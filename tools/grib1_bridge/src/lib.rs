//! Deterministic parallel CPU kernels for GPUWM native preprocessing.
//!
//! The GRIB bridges and this library intentionally share one Rust package so
//! a distributable preprocessor can reuse the already-vendored decoder build
//! without introducing a second native toolchain.  The C ABI is deliberately
//! small: Python owns metadata/shape validation and passes contiguous FP32
//! buffers; Rust partitions independent target points or columns into fixed,
//! contiguous ranges.  No reduction crosses a worker boundary, so changing
//! the worker count cannot change an output element's arithmetic.

pub mod dealias;
pub mod chem_conversions;
pub mod chem_diagnostics;
pub mod chem_fire_reduce;
pub mod eta;
pub mod ensemble;
pub mod grib2_supplement;
pub mod quantization;
pub mod surface_pressure;
pub mod aerosol_emission;
pub mod aerosol_numbers;
// The static-dataset ingest arm: a WPS intermediate READER (the
// inverse of src/bin/met_intermediate.rs's writer) and the
// global-source bilinear the seam of a cyclic lat/lon grid needs.
pub mod wif_ffi;
pub mod chem_remap;
pub mod grib2_stack;
pub mod wps_intermediate;
// WPS metgrid's masked-field chain (soil, snow, skin, sea ice) in float64,
// byte-identical to the NumPy transcription it replaced.
pub mod wps_masked;
// The native HRRR route's land-only bilinear stencil for soil, built in
// float64 and applied in float32, byte-identical to the NumPy builder.
pub mod masked_stencil;
// The lake skin search and the water-temperature blends in float64,
// byte-identical to the NumPy code they replaced.
pub mod water_blend;
// The water-temperature box repairs and the per-body assembly in float64,
// byte-identical to the NumPy code they replaced.
pub mod water_repair;
// Which water body owns each source cell, byte-identical to the NumPy
// code it replaced.
pub mod water_owner;
// The bounded surface-nearest search of the CPU backend in float32,
// byte-identical to the NumPy code it replaced.
pub mod surface_nearest;
pub mod missing_value_chain;
// Elementwise exp, log, pow and the trigonometric functions through the
// vendored libm crate, so a host preparation's transcendentals are the same
// on every CPU and C library (gpuwm/core/portable_math.py).
pub mod portable_math;
pub mod parallel;
pub mod glibc239_math;
pub mod prepared_io;
pub mod vertical_plan;
pub mod health_scan;
pub mod ozone;
pub mod cold_start;
pub mod host_arrays;
pub mod analysis_exchange;
pub mod preparation_fingerprints;
pub mod noah_init;
// IMS daily snow and ice (NSIDC G02156) sampled at target points: the
// observed lake-ice provider's reader (gpuwm/ingest/lake_ice.py).
pub mod ims_snow_ice;
pub mod slab_init;
pub mod soil_points;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI32, Ordering};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// artifact was built from, embedded so the release cut can prove a
/// staged binary matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`), never by
/// executing it.  `build.rs` injects the value; `unknown` marks a build
/// the cut must refuse (outside git, or a dirty tree).  Every
/// executable in this package references this constant from `main` so
/// the linker cannot discard the bytes.
pub static SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The stamp for C callers, NUL-terminated -- and the exported
/// reference that keeps the stamp bytes present in the cdylib the
/// release ships, where no `main` exists to hold them.
#[no_mangle]
pub extern "C" fn gpuwm_preprocess_cpu_source_rev()
-> *const std::os::raw::c_char {
    static C_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    C_STAMP.as_ptr().cast()
}

pub(crate) const OK: i32 = 0;
pub(crate) const ERR_NULL: i32 = 1;
pub(crate) const ERR_DIMENSION: i32 = 2;
pub(crate) const ERR_NONFINITE: i32 = 3;
pub(crate) const ERR_PRESSURE_ORDER: i32 = 4;
pub(crate) const ERR_SURFACE_BRACKET: i32 = 5;
pub(crate) const ERR_TARGET_ABOVE_TOP: i32 = 6;
pub(crate) const ERR_INTERPOLATION_WINDOW: i32 = 7;

/// Relative distance (2^-16) within which a target above the highest
/// source level is co-located with it instead of refused.
///
/// The gap it absorbs is physical, not only rounding.  The source top is
/// the native model's top mass level as a FULL (moist) pressure, and WRF's
/// integ_moist leaves that level undried (pd(top) = p(top)), while the
/// target is the same eta level as a DRY pressure.  Their difference is
/// the water-vapour weight in the top half-layer, about
/// (p_level - p_lid) * qv, so relative to the level it is at most qv_top.
/// Native HRRR at a 1500 Pa lid measures 1731.4753..1731.4758 Pa against
/// the 1731.475 Pa dry target: 231.475 Pa * qv for qv = 0.9..3.3e-6 kg/kg,
/// plus GRIB packing steps of 1.3e-4 Pa and FP32 rounding.  The previous
/// four-epsilon (2^-21) bound covered rounding only and refused a moist
/// stratosphere (2025-03-14 06Z: 4.9e-7 relative).  2^-16 = 1.53e-5 still
/// covers qv_top = 1e-5 kg/kg, above any stratospheric water vapour, with
/// FP32 and GRIB rounding on top.  In height it is about 0.1 m (scale
/// height * 2^-16), far below any vertical grid spacing, so the clamped
/// target reads the top source value with no new layer extrapolated.
/// A power of two keeps `tolerance * top` exact in FP32, so the Rust,
/// CUDA and launcher comparisons give the same verdict on every column.
// Preserve the exact power-of-two tolerance shared with CUDA.
#[allow(clippy::excessive_precision)]
pub(crate) const TOP_COLOCATION_RTOL: f32 = 1.52587890625e-5;
pub(crate) const ERR_PANIC: i32 = 127;

#[inline]
pub(crate) fn worker_ranges(length: usize, workers: usize) -> Vec<(usize, usize)> {
    let count = workers.min(length);
    let quotient = length / count;
    let remainder = length % count;
    let mut ranges = Vec::with_capacity(count);
    let mut start = 0usize;
    for index in 0..count {
        let stop = start + quotient + usize::from(index < remainder);
        ranges.push((start, stop));
        start = stop;
    }
    ranges
}

#[inline]
fn nearest_even_nonnegative(value: f32) -> usize {
    let lower = value.floor();
    let fraction = value - lower;
    let lower_index = lower as usize;
    if fraction < 0.5 {
        lower_index
    } else if fraction > 0.5 || lower_index % 2 == 1 {
        lower_index + 1
    } else {
        lower_index
    }
}

#[inline]
fn cuda_ftz_product_nonzero(left: f32, right: f32) -> bool {
    // NVIDIA FP32 arithmetic flushes subnormal operands/results to zero.  The
    // WPS `oned` missing-value branch tests the product rather than the two
    // operands independently, so preserving host subnormals here can select a
    // different interpolation polynomial at legitimate zero-valued stencils.
    // Mirror the CUDA authority explicitly instead of depending on host MXCSR
    // state or compiler flags.
    left.is_normal() && right.is_normal() && {
        let product = left * right;
        product.is_normal() || product.is_infinite()
    }
}

#[inline]
fn host_ieee_product_normal(left: f32, right: f32) -> bool {
    // The OTHER missing-value predicate: a literal mirror of NumPy's
    //     np.abs(np.multiply(b, c, dtype=np.float32))
    //         >= np.finfo(np.float32).tiny
    // as `gpuwm/ingest/hrrr.py:_wps_oned_cpu` writes it.  It differs from
    // `cuda_ftz_product_nonzero` in exactly one corner -- a SUBNORMAL
    // OPERAND whose product with its partner is normal -- because the
    // NumPy form flushes only the RESULT (the `>= tiny` compare) and
    // never the operands, while CUDA flushes both.  Measured on the box
    // that built this: `np.float32(1e-40) * np.float32(1e10)` is
    // 9.999946e-31, a normal number, so the NumPy predicate says "both"
    // and the CUDA one says "neither".
    //
    // Which one is correct is not this function's question.  The two
    // predicates serve two different pinned authorities: the projected
    // HRRR host route is pinned to the NumPy operator it is replacing,
    // the regular-grid route is pinned to the CUDA plan beside it.  A
    // port that "improved" the corner would move production bits, so
    // this reproduces it and says so.
    //
    // f32::MIN_POSITIVE is the smallest positive NORMAL, which is
    // exactly what np.finfo(np.float32).tiny names.  NaN compares false
    // (NumPy agrees), infinities compare true (NumPy agrees).
    (left * right).abs() >= f32::MIN_POSITIVE
}

#[inline]
fn wps_oned_with(x: f32, a: f32, b: f32, c: f32, d: f32, both: bool) -> f32 {
    let zero = 0.0f32;
    let half = 0.5f32;
    let one = 1.0f32;
    let regular = (one - x) * (b + x * (half * (c - a) + x * (half * (c + a) - b)))
        + x * (c + (one - x) * (half * (b - d) + (one - x) * (half * (b + d) - c)));
    let mut out = zero;
    if x == zero {
        out = b;
    }
    if x == one {
        out = c;
    }
    if both && a == zero && d == zero {
        out = b * (one - x) + c * x;
    }
    if both && a != zero && d == zero {
        out = b + x * (half * (c - a) + x * (half * (c + a) - b));
    }
    if both && a == zero && d != zero {
        out = c + (one - x) * (half * (b - d) + (one - x) * (half * (b + d) - c));
    }
    if both && a != zero && d != zero {
        out = regular;
    }
    out
}

#[inline]
fn wps_oned(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    wps_oned_with(x, a, b, c, d, cuda_ftz_product_nonzero(b, c))
}

#[inline]
fn wps_oned_host(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    wps_oned_with(x, a, b, c, d, host_ieee_product_normal(b, c))
}

#[inline]
fn source_at(source: &[f32], lead: usize, y: usize, x: usize, ny: usize, nx: usize) -> f32 {
    source[(lead * ny + y) * nx + x]
}

fn horizontal_point(
    source: &[f32],
    lead: usize,
    y: f32,
    x: f32,
    ny: usize,
    nx: usize,
    method: i32,
) -> Result<f32, i32> {
    if !y.is_finite()
        || !x.is_finite()
        || y < 0.0
        || x < 0.0
        || y > (ny - 1) as f32
        || x > (nx - 1) as f32
    {
        return Err(ERR_NONFINITE);
    }
    if method == 0 {
        let iy = nearest_even_nonnegative(y).min(ny - 1);
        let ix = nearest_even_nonnegative(x).min(nx - 1);
        return Ok(source_at(source, lead, iy, ix, ny, nx));
    }
    if method == 1 {
        let iy = (y.floor() as usize).min(ny - 2);
        let ix = (x.floor() as usize).min(nx - 2);
        let fy = y - iy as f32;
        let fx = x - ix as f32;
        let one = 1.0f32;
        let lower = (one - fx) * source_at(source, lead, iy, ix, ny, nx)
            + fx * source_at(source, lead, iy, ix + 1, ny, nx);
        let upper = (one - fx) * source_at(source, lead, iy + 1, ix, ny, nx)
            + fx * source_at(source, lead, iy + 1, ix + 1, ny, nx);
        return Ok((one - fy) * lower + fy * upper);
    }
    if method != 2 {
        return Err(ERR_DIMENSION);
    }

    let iy = y.floor() as isize;
    let ix = x.floor() as isize;
    let fy = y - iy as f32;
    let fx = x - ix as f32;
    let tiny = 1.0e-20f32;
    let mut rows = [0.0f32; 4];
    for (row_index, y_offset) in (-1isize..=2).enumerate() {
        let jy = (iy + y_offset).clamp(0, ny as isize - 1) as usize;
        let mut values = [0.0f32; 4];
        for (column_index, x_offset) in (-1isize..=2).enumerate() {
            let jx = (ix + x_offset).clamp(0, nx as isize - 1) as usize;
            let value = source_at(source, lead, jy, jx, ny, nx);
            values[column_index] = if value == 0.0 { tiny } else { value };
        }
        rows[row_index] = wps_oned(fx, values[0], values[1], values[2], values[3]);
    }
    let result = wps_oned(fy, rows[0], rows[1], rows[2], rows[3]);
    Ok(if result == tiny { 0.0 } else { result })
}

/// Interpolate one or more regular-grid FP32 fields to precomputed source
/// coordinates. `method` is 0=nearest, 1=bilinear, 2=WPS parabolic.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_regular_interp_f32(
    source: *const f32,
    target_y: *const f32,
    target_x: *const f32,
    output: *mut f32,
    nlead: usize,
    source_ny: usize,
    source_nx: usize,
    ntarget: usize,
    method: i32,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null() || target_y.is_null() || target_x.is_null() || output.is_null() {
            return ERR_NULL;
        }
        if nlead == 0
            || source_ny < 2
            || source_nx < 2
            || ntarget == 0
            || workers == 0
            || !(0..=2).contains(&method)
        {
            return ERR_DIMENSION;
        }
        let source_length = match nlead
            .checked_mul(source_ny)
            .and_then(|n| n.checked_mul(source_nx))
        {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let output_length = match nlead.checked_mul(ntarget) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let source_slice = std::slice::from_raw_parts(source, source_length);
        let y_slice = std::slice::from_raw_parts(target_y, ntarget);
        let x_slice = std::slice::from_raw_parts(target_x, ntarget);
        let output_address = output as usize;
        let error = AtomicI32::new(OK);
        parallel::run_ranges(ntarget, workers, |start, stop| {
                    let error = &error;
                    let output_ptr = output_address as *mut f32;
                    for target_index in start..stop {
                        if error.load(Ordering::Relaxed) != OK {
                            break;
                        }
                        for lead in 0..nlead {
                            match horizontal_point(
                                source_slice,
                                lead,
                                y_slice[target_index],
                                x_slice[target_index],
                                source_ny,
                                source_nx,
                                method,
                            ) {
                                Ok(value) => unsafe {
                                    *output_ptr.add(lead * ntarget + target_index) = value;
                                },
                                Err(code) => {
                                    error
                                        .compare_exchange(
                                            OK,
                                            code,
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                        )
                                        .ok();
                                    break;
                                }
                            }
                        }
                    }
        });
        let code = error.load(Ordering::Relaxed);
        if code == OK
            && std::slice::from_raw_parts(output, output_length)
                .iter()
                .any(|value| !value.is_finite())
        {
            ERR_NONFINITE
        } else {
            code
        }
    }))
    .unwrap_or(ERR_PANIC)
}

// ---------------------------------------------------------------------------
// Indexed-donor horizontal interpolation
// ---------------------------------------------------------------------------
//
// `gpuwm_regular_interp_f32` above takes FRACTIONAL source coordinates and
// derives the donor cell from them with an FP32 floor.  A projected source
// (HRRR on its own Lambert grid) cannot use that: it selects the donor in
// FP64 and keeps it, because a local coordinate just below an integer can
// advance its donor once it is rounded to FP32.  So the projected route had
// no Rust boundary at all and ran the whole operator in NumPy.
//
// This is that boundary.  It takes the donor as an EXACT integer pair plus
// the FP32 fraction the caller already holds -- the same `(iy, ix, fy, fx)`
// split `_ProjectedGpuPlan` hands the CUDA kernel -- and never re-derives
// either one.  The arithmetic below is a statement-order mirror of
// `gpuwm/ingest/hrrr.py:_ProjectedCpuPlan.apply`, down to the zero/`tiny`
// sentinel round trip and the host-IEEE `oned` predicate; see
// `host_ieee_product_normal` for the one place that differs from the CUDA
// mirror in this same file, and why it must.

#[inline]
#[allow(clippy::too_many_arguments)]
fn indexed_point(
    source: &[f32],
    lead: usize,
    iy: i64,
    ix: i64,
    fy: f32,
    fx: f32,
    ny: usize,
    nx: usize,
    method: i32,
) -> f32 {
    let one = 1.0f32;
    if method == 1 {
        // NumPy indexes iy/iy+1 and ix/ix+1 WITHOUT clipping here; the
        // caller's geometry guarantees the four-point halo and the entry
        // point below rejects a donor that would leave the window.
        let iy = iy as usize;
        let ix = ix as usize;
        let lower = (one - fx) * source_at(source, lead, iy, ix, ny, nx)
            + fx * source_at(source, lead, iy, ix + 1, ny, nx);
        let upper = (one - fx) * source_at(source, lead, iy + 1, ix, ny, nx)
            + fx * source_at(source, lead, iy + 1, ix + 1, ny, nx);
        return (one - fy) * lower + fy * upper;
    }
    // Parabolic.  np.clip(ix + offset, 0, nx - 1), in i64 so an extreme
    // donor cannot wrap the way int32 index arithmetic would.
    let tiny = 1.0e-20f32;
    let mut rows = [0.0f32; 4];
    for (row_index, y_offset) in (-1i64..=2).enumerate() {
        let jy = (iy + y_offset).clamp(0, ny as i64 - 1) as usize;
        let mut values = [0.0f32; 4];
        for (column_index, x_offset) in (-1i64..=2).enumerate() {
            let jx = (ix + x_offset).clamp(0, nx as i64 - 1) as usize;
            let value = source_at(source, lead, jy, jx, ny, nx);
            values[column_index] = if value == 0.0 { tiny } else { value };
        }
        rows[row_index] = wps_oned_host(fx, values[0], values[1], values[2], values[3]);
    }
    let result = wps_oned_host(fy, rows[0], rows[1], rows[2], rows[3]);
    if result == tiny {
        0.0
    } else {
        result
    }
}

/// Interpolate FP32 fields onto targets given by an EXACT integer donor
/// index plus its FP32 fraction.  `method` is 1=bilinear, 2=WPS parabolic.
///
/// Nearest is deliberately absent: it is a pure gather off a SEPARATE
/// donor pair (the caller's round-to-nearest indices, not this floor), so
/// routing it here would mean a second index contract for an operation
/// NumPy fancy-indexing already does exactly and in well under a percent
/// of this operator's wall time.
///
/// Non-finite values are propagated, not refused, because the operator
/// this mirrors propagates them: the `oned` missing-value predicate is
/// false for a NaN product, so a NaN donor resolves to the polynomial's
/// zero default rather than to NaN, and refusing here would turn a
/// reproduction into a behaviour change.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_indexed_interp_f32(
    source: *const f32,
    donor_y: *const i32,
    donor_x: *const i32,
    fraction_y: *const f32,
    fraction_x: *const f32,
    output: *mut f32,
    nlead: usize,
    source_ny: usize,
    source_nx: usize,
    ntarget: usize,
    method: i32,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if source.is_null()
            || donor_y.is_null()
            || donor_x.is_null()
            || fraction_y.is_null()
            || fraction_x.is_null()
            || output.is_null()
        {
            return ERR_NULL;
        }
        if nlead == 0
            || source_ny < 2
            || source_nx < 2
            || ntarget == 0
            || workers == 0
            || !(1..=2).contains(&method)
        {
            return ERR_DIMENSION;
        }
        let source_length = match nlead
            .checked_mul(source_ny)
            .and_then(|n| n.checked_mul(source_nx))
        {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        if nlead.checked_mul(ntarget).is_none() {
            return ERR_DIMENSION;
        }
        let source_slice = std::slice::from_raw_parts(source, source_length);
        let donor_y_slice = std::slice::from_raw_parts(donor_y, ntarget);
        let donor_x_slice = std::slice::from_raw_parts(donor_x, ntarget);
        let fy_slice = std::slice::from_raw_parts(fraction_y, ntarget);
        let fx_slice = std::slice::from_raw_parts(fraction_x, ntarget);
        // Bilinear reads iy+1/ix+1 unguarded, exactly as the mirrored
        // NumPy expression does.  NumPy would WRAP a negative index and
        // raise on an overrun; neither is a reproduction worth having, so
        // the whole donor field is checked once, up front, before any
        // worker starts.  Parabolic clamps every offset and needs nothing.
        if method == 1 {
            let fits = (0..ntarget).all(|index| {
                let iy = donor_y_slice[index] as i64;
                let ix = donor_x_slice[index] as i64;
                iy >= 0 && ix >= 0 && iy + 1 < source_ny as i64 && ix + 1 < source_nx as i64
            });
            if !fits {
                return ERR_DIMENSION;
            }
        }
        let output_address = output as usize;
        parallel::run_ranges(ntarget, workers, |start, stop| {
                    let output_ptr = output_address as *mut f32;
                    // Lead outer, target inner: within one level the source
                    // stencil sweeps the window in raster order, so the
                    // four donor rows stay resident, and the stores are
                    // contiguous.  Pure loop order -- no element's
                    // arithmetic depends on it, and neither does the
                    // partition, which is why worker count cannot move a
                    // bit here any more than it can in the two entry
                    // points above.
                    for lead in 0..nlead {
                        for target_index in start..stop {
                            let value = indexed_point(
                                source_slice,
                                lead,
                                donor_y_slice[target_index] as i64,
                                donor_x_slice[target_index] as i64,
                                fy_slice[target_index],
                                fx_slice[target_index],
                                source_ny,
                                source_nx,
                                method,
                            );
                            unsafe {
                                *output_ptr.add(lead * ntarget + target_index) = value;
                            }
                        }
                    }
        });
        OK
    }))
    .unwrap_or(ERR_PANIC)
}

#[inline]
fn lagrange(x: &[f32], y: &[f32], order: usize, target: f32) -> f32 {
    let mut result = 0.0f32;
    for term in 0..=order {
        let mut numerator = 1.0f32;
        let mut denominator = 1.0f32;
        for index in 0..=order {
            if index == term {
                continue;
            }
            numerator *= target - x[index];
            denominator *= x[term] - x[index];
        }
        if denominator != 0.0 {
            result += y[term] * numerator / denominator;
        }
    }
    result
}

struct VerticalScratch {
    ox: Vec<f32>,
    oy: Vec<f32>,
    x: Vec<f32>,
}

impl VerticalScratch {
    fn new(levels: usize) -> Self {
        Self { ox: Vec::with_capacity(levels), oy: Vec::with_capacity(levels),
               x: Vec::with_capacity(levels) }
    }
}

#[allow(clippy::too_many_arguments)]
fn vertical_column(
    field: &[f32],
    surface_field: &[f32],
    source_pressure: &[f32],
    surface_pressure: &[f32],
    target_pressure: &[f32],
    output_address: usize,
    nsource: usize,
    ntarget: usize,
    ncolumn: usize,
    column: usize,
    interp_in_logp: bool,
    extrap_temperature: bool,
    force_surface: usize,
    zap_close_levels: f32,
    vboundb: usize,
    scratch: &mut VerticalScratch,
) -> Result<(), i32> {
    let psfc = surface_pressure[column];
    if !psfc.is_finite() || psfc <= 0.0 || !surface_field[column].is_finite() {
        return Err(ERR_NONFINITE);
    }
    let mut previous = f32::INFINITY;
    let mut first_above = None;
    for level in 0..nsource {
        let pressure = source_pressure[level * ncolumn + column];
        let value = field[level * ncolumn + column];
        if !pressure.is_finite() || pressure <= 0.0 || !value.is_finite() {
            return Err(ERR_NONFINITE);
        }
        if level > 0 && pressure >= previous {
            return Err(ERR_PRESSURE_ORDER);
        }
        previous = pressure;
        if first_above.is_none() && pressure < psfc {
            first_above = Some(level);
        }
    }
    let first_above = first_above.ok_or(ERR_SURFACE_BRACKET)?;
    let VerticalScratch { ox, oy, x } = scratch;
    ox.clear();
    oy.clear();
    x.clear();
    if first_above > 0 {
        for level in 0..first_above {
            ox.push(source_pressure[level * ncolumn + column]);
            oy.push(field[level * ncolumn + column]);
        }
        if ox[ox.len() - 1] - psfc < zap_close_levels {
            ox.pop();
            oy.pop();
        }
        ox.push(psfc);
        oy.push(surface_field[column]);
        let mut next = first_above;
        if force_surface > 0 {
            let force_pressure = target_pressure[(force_surface - 1) * ncolumn + column];
            for level in first_above..nsource {
                if source_pressure[level * ncolumn + column] <= force_pressure {
                    next = level;
                    break;
                }
            }
        }
        let start =
            if ox[ox.len() - 1] - source_pressure[next * ncolumn + column] < zap_close_levels {
                next + 1
            } else {
                next
            };
        for level in start..nsource {
            ox.push(source_pressure[level * ncolumn + column]);
            oy.push(field[level * ncolumn + column]);
        }
    } else {
        ox.push(psfc);
        oy.push(surface_field[column]);
        let mut next = 0usize;
        if force_surface > 0 {
            let force_pressure = target_pressure[(force_surface - 1) * ncolumn + column];
            for level in 0..nsource {
                if source_pressure[level * ncolumn + column] <= force_pressure {
                    next = level;
                    break;
                }
            }
        }
        for level in next..nsource {
            let pressure = source_pressure[level * ncolumn + column];
            if ox[ox.len() - 1] - pressure < zap_close_levels && level < nsource - 1 {
                continue;
            }
            ox.push(pressure);
            oy.push(field[level * ncolumn + column]);
        }
    }
    if ox.len() < 2 {
        return Err(ERR_INTERPOLATION_WINDOW);
    }
    if interp_in_logp {
        x.extend(ox.iter().map(|value| value.ln()));
    } else {
        x.extend_from_slice(ox);
    }
    let output = output_address as *mut f32;
    for target_level in 0..ntarget {
        let mut pressure = target_pressure[target_level * ncolumn + column];
        if !pressure.is_finite() || pressure <= 0.0 {
            return Err(ERR_NONFINITE);
        }
        // A native top mass level carries its top half-layer's vapour
        // weight that the dry target does not (TOP_COLOCATION_RTOL).
        // Co-locate only that gap; a target meaningfully above the source
        // still fails, because its value would have to be extrapolated
        // past the top of the analysis.
        let top = ox[ox.len() - 1];
        if pressure < top {
            if top - pressure > TOP_COLOCATION_RTOL * top.abs() {
                // Check pressure before logarithms: two distinct endpoint
                // pressures can have the same rounded logarithm.
                return Err(ERR_TARGET_ABOVE_TOP);
            }
            pressure = top;
        }
        let target_x = if interp_in_logp {
            pressure.ln()
        } else {
            pressure
        };
        let mut found = None;
        for lower in 0..x.len() - 1 {
            if (target_x - x[lower]) * (target_x - x[lower + 1]) <= 0.0 {
                found = Some(lower);
                break;
            }
        }
        let result = if let Some(lower) = found {
            // WRF's `k+1 >= 1+vboundb` in wrapping unsigned arithmetic, as the CUDA kernel
            // does: vboundb = -1 (usize::MAX) wraps to 0 and admits every level. The
            // clippy rewrite to `k >= vboundb` (3b41df516e) dropped the wrap.
            if target_level.wrapping_add(1) >= vboundb.wrapping_add(1) {
                let fits_upper = lower + 2 < x.len();
                let fits_lower = lower >= 1;
                if fits_upper && fits_lower {
                    0.5 * (lagrange(&x[lower..lower + 3], &oy[lower..lower + 3], 2, target_x)
                        + lagrange(
                            &x[lower - 1..lower + 2],
                            &oy[lower - 1..lower + 2],
                            2,
                            target_x,
                        ))
                } else if fits_upper {
                    lagrange(&x[lower..lower + 3], &oy[lower..lower + 3], 2, target_x)
                } else if fits_lower {
                    lagrange(
                        &x[lower - 1..lower + 2],
                        &oy[lower - 1..lower + 2],
                        2,
                        target_x,
                    )
                } else {
                    return Err(ERR_INTERPOLATION_WINDOW);
                }
            } else {
                lagrange(&x[lower..lower + 2], &oy[lower..lower + 2], 1, target_x)
            }
        } else if pressure > ox[0] {
            if extrap_temperature {
                let t1 = oy[0] * (ox[0] / 100000.0).powf(0.2857143);
                let average_pressure = 0.5 * (pressure + ox[0]);
                let dhdp = 11880.516 * 0.1902632 * (average_pressure / 100.0).powf(0.1902632 - 1.0);
                let dt = dhdp * ((pressure - ox[0]) / 100.0) * 0.0065;
                (t1 + dt) * (100000.0 / pressure).powf(0.2857143)
            } else {
                oy[0]
            }
        } else {
            return Err(ERR_TARGET_ABOVE_TOP);
        };
        if !result.is_finite() {
            return Err(ERR_NONFINITE);
        }
        unsafe {
            *output.add(target_level * ncolumn + column) = result;
        }
    }
    Ok(())
}

/// Apply WRF-real vertical interpolation to independent FP32 columns.
/// Unlike the CUDA v1 kernel, source columns are dynamically sized.
///
/// # Safety
///
/// Every pointer must address the complete contiguous buffer implied by the
/// dimensions.  Input and output buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_vert_interp_f32(
    field: *const f32,
    surface_field: *const f32,
    source_pressure: *const f32,
    surface_pressure: *const f32,
    target_pressure: *const f32,
    output: *mut f32,
    nsource: usize,
    ntarget: usize,
    ncolumn: usize,
    interp_in_logp: i32,
    extrap_temperature: i32,
    force_surface: usize,
    zap_close_levels: f32,
    vboundb: usize,
    workers: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if field.is_null()
            || surface_field.is_null()
            || source_pressure.is_null()
            || surface_pressure.is_null()
            || target_pressure.is_null()
            || output.is_null()
        {
            return ERR_NULL;
        }
        if nsource < 2
            || ntarget == 0
            || ncolumn == 0
            || workers == 0
            || force_surface > ntarget
            || !zap_close_levels.is_finite()
            || zap_close_levels < 0.0
            || !(0..=1).contains(&interp_in_logp)
            || !(0..=1).contains(&extrap_temperature)
        {
            return ERR_DIMENSION;
        }
        let source_length = match nsource.checked_mul(ncolumn) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let target_length = match ntarget.checked_mul(ncolumn) {
            Some(value) => value,
            None => return ERR_DIMENSION,
        };
        let field_slice = std::slice::from_raw_parts(field, source_length);
        let source_slice = std::slice::from_raw_parts(source_pressure, source_length);
        let surface_field_slice = std::slice::from_raw_parts(surface_field, ncolumn);
        let surface_pressure_slice = std::slice::from_raw_parts(surface_pressure, ncolumn);
        let target_slice = std::slice::from_raw_parts(target_pressure, target_length);
        let output_address = output as usize;
        let error = AtomicI32::new(OK);
        parallel::run_ranges(ncolumn, workers, |start, stop| {
                    let error = &error;
                    // Three source-level vectors per worker, reused for
                    // every column instead of three allocations per cell.
                    let mut scratch = VerticalScratch::new(nsource + 1);
                    for column in start..stop {
                        if error.load(Ordering::Relaxed) != OK {
                            break;
                        }
                        if let Err(code) = vertical_column(
                            field_slice,
                            surface_field_slice,
                            source_slice,
                            surface_pressure_slice,
                            target_slice,
                            output_address,
                            nsource,
                            ntarget,
                            ncolumn,
                            column,
                            interp_in_logp != 0,
                            extrap_temperature != 0,
                            force_surface,
                            zap_close_levels,
                            vboundb,
                            &mut scratch,
                        ) {
                            error
                                .compare_exchange(OK, code, Ordering::Relaxed, Ordering::Relaxed)
                                .ok();
                            break;
                        }
                    }
        });
        error.load(Ordering::Relaxed)
    }))
    .unwrap_or(ERR_PANIC)
}

/// The ABI generation of the entry points already in the contract.
///
/// It stays 1 across the addition of `gpuwm_indexed_interp_f32`, and that
/// is the whole versioning rule: this number describes the SHAPE of the
/// existing calls, and Python refuses a library whose number is not the
/// one it was built against (`gpuwm/ingest/cpu_backend.py`).  Bumping it
/// to advertise a NEW symbol would refuse every correctly-built older
/// library over a call those libraries were never asked to make.  A
/// caller that wants the new entry looks the symbol up instead and keeps
/// its own fallback -- which is also the only accurate answer, since a
/// staged bundle can be older than the checkout driving it.
///
/// Changing an existing signature is the thing that bumps this.
#[no_mangle]
pub extern "C" fn gpuwm_preprocess_cpu_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_partition_is_complete_and_stable() {
        assert_eq!(worker_ranges(10, 3), vec![(0, 4), (4, 7), (7, 10)]);
        assert_eq!(worker_ranges(3, 9), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn nearest_uses_ties_to_even() {
        assert_eq!(nearest_even_nonnegative(0.5), 0);
        assert_eq!(nearest_even_nonnegative(1.5), 2);
        assert_eq!(nearest_even_nonnegative(2.5), 2);
    }

    #[test]
    fn parabolic_zero_sentinel_matches_cuda_flush_to_zero() {
        // This is the reduced 700-hPa GFS RH stencil that exposed a 0 versus
        // 3.28849-percent CPU/CUDA branch split in the full native pipeline.
        let source = [
            8.4, 9.3, 10.2, 11.4, 1.1, 2.0, 3.4, 5.1, 0.0, 0.0, 0.0, 0.2, 2.2, 1.7, 1.2, 0.8,
        ];
        let actual = horizontal_point(&source, 0, 1.002_899_2, 1.937_393_2, 4, 4, 2)
            .expect("valid parabolic stencil");
        assert_eq!(actual.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn this_build_does_not_flush_subnormals() {
        // The host-IEEE predicate is only a mirror of NumPy if THIS
        // binary keeps subnormals.  x86-64 Rust does not set MXCSR
        // FTZ/DAZ and does not enable fast-math, so it does not -- but
        // that is a property of the build, not of the language, and a
        // future flag that changed it would silently move the projected
        // route's bits.  Assert it where it would be noticed.
        let subnormal = f32::from_bits(0x0001_1682); // np.float32(1e-40)
        assert!(!subnormal.is_normal());
        assert_ne!(subnormal, 0.0);
        // DAZ off: the subnormal operand is not read as zero.
        let product = subnormal * 1.0e10f32;
        assert!(product.is_normal());
        // FTZ off: a subnormal RESULT survives as a subnormal.
        let squeezed = subnormal * 0.5f32;
        assert!(!squeezed.is_normal());
        assert_ne!(squeezed, 0.0);
    }

    #[test]
    fn host_and_cuda_predicates_split_only_on_a_non_normal_operand() {
        let subnormal = f32::from_bits(0x0001_1682); // np.float32(1e-40)
        // (left, right, host/NumPy answer, cuda/FTZ answer).  The two
        // split exactly when an OPERAND is not normal but the host
        // product still is -- CUDA flushes the operand, NumPy does not.
        for (left, right, host, cuda) in [
            (subnormal, 1.0e10f32, true, false),
            (1.0e10f32, subnormal, true, false),
            (f32::INFINITY, 5.0f32, true, false),
            // Agreement everywhere else, including the subnormal PRODUCT
            // of two normal operands, which is what `>= tiny` is for.
            (1.0e-20f32, 1.0e-20f32, false, false),
            (subnormal, 1.0e-10f32, false, false),
            (0.0f32, 5.0f32, false, false),
            (5.0f32, 0.0f32, false, false),
            (f32::NAN, 5.0f32, false, false),
            (3.0e38f32, 3.0e38f32, true, true),
            (1.0f32, 1.0f32, true, true),
            (-2.5f32, 4.0f32, true, true),
        ] {
            assert_eq!(
                host_ieee_product_normal(left, right),
                host,
                "host predicate on ({left}, {right})"
            );
            assert_eq!(
                cuda_ftz_product_nonzero(left, right),
                cuda,
                "cuda predicate on ({left}, {right})"
            );
        }
    }

    #[test]
    fn indexed_parabolic_is_worker_count_invariant() {
        let ny = 24usize;
        let nx = 31usize;
        let nlead = 3usize;
        let ntarget = 97usize;
        let source: Vec<f32> = (0..nlead * ny * nx)
            .map(|index| {
                let value = ((index * 37) % 211) as f32 * 0.125 - 6.0;
                if index % 17 == 0 {
                    0.0
                } else {
                    value
                }
            })
            .collect();
        let donor_y: Vec<i32> = (0..ntarget).map(|i| (i % (ny - 3) + 1) as i32).collect();
        let donor_x: Vec<i32> = (0..ntarget).map(|i| (i % (nx - 3) + 1) as i32).collect();
        let fy: Vec<f32> = (0..ntarget).map(|i| (i % 8) as f32 / 8.0).collect();
        let fx: Vec<f32> = (0..ntarget).map(|i| (i % 5) as f32 / 5.0).collect();
        let run = |workers: usize| {
            let mut output = vec![0.0f32; nlead * ntarget];
            let code = unsafe {
                gpuwm_indexed_interp_f32(
                    source.as_ptr(),
                    donor_y.as_ptr(),
                    donor_x.as_ptr(),
                    fy.as_ptr(),
                    fx.as_ptr(),
                    output.as_mut_ptr(),
                    nlead,
                    ny,
                    nx,
                    ntarget,
                    2,
                    workers,
                )
            };
            assert_eq!(code, OK);
            output
        };
        let serial = run(1);
        for workers in [2usize, 3, 7, 64] {
            let parallel = run(workers);
            assert!(
                serial
                    .iter()
                    .zip(parallel.iter())
                    .all(|(left, right)| left.to_bits() == right.to_bits()),
                "worker count {workers} moved a bit"
            );
        }
    }

    #[test]
    fn indexed_bilinear_refuses_a_donor_outside_the_halo() {
        let source = [1.0f32; 16];
        let donor_y = [3i32];
        let donor_x = [0i32];
        let fy = [0.5f32];
        let fx = [0.5f32];
        let mut output = [0.0f32; 1];
        let code = unsafe {
            gpuwm_indexed_interp_f32(
                source.as_ptr(),
                donor_y.as_ptr(),
                donor_x.as_ptr(),
                fy.as_ptr(),
                fx.as_ptr(),
                output.as_mut_ptr(),
                1,
                4,
                4,
                1,
                1,
                1,
            )
        };
        assert_eq!(code, ERR_DIMENSION);
    }

    #[test]
    fn vertical_column_has_no_64_level_ceiling() {
        let nsource = 96usize;
        let ntarget = 137usize;
        let ncolumn = 1usize;
        let source_pressure: Vec<f32> = (0..nsource)
            .map(|level| 100000.0 - level as f32 * 900.0)
            .collect();
        let field: Vec<f32> = source_pressure
            .iter()
            .map(|pressure| 200.0 + pressure * 0.001)
            .collect();
        let surface_pressure = [100500.0f32];
        let surface_field = [300.5f32];
        let target_pressure: Vec<f32> = (0..ntarget)
            .map(|level| 100250.0 - level as f32 * 600.0)
            .collect();
        let mut output = vec![0.0f32; ntarget];
        let code = unsafe {
            gpuwm_wrf_vert_interp_f32(
                field.as_ptr(),
                surface_field.as_ptr(),
                source_pressure.as_ptr(),
                surface_pressure.as_ptr(),
                target_pressure.as_ptr(),
                output.as_mut_ptr(),
                nsource,
                ntarget,
                ncolumn,
                0,
                0,
                1,
                0.0,
                4,
                7,
            )
        };
        assert_eq!(code, OK);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    fn top_column(source: &[f32], target: f32, logp: bool) -> (Result<(), i32>, f32) {
        let field: Vec<f32> = (0..source.len()).map(|level| 289.0 - 10.0 * level as f32).collect();
        let mut output = [0.0f32];
        let result = vertical_column(
            &field, &[291.0], source, &[100000.0], &[target],
            output.as_mut_ptr() as usize, source.len(), 1, 1, 0,
            logp, false, 0, 500.0, 4,
            &mut VerticalScratch::new(source.len() + 1));
        (result, output[0])
    }

    /// The target reads exactly what a target AT the source top reads:
    /// the top source value, with no layer extrapolated beyond it.
    fn assert_colocated(source: &[f32], target: f32, logp: bool) {
        let top = source[source.len() - 1];
        let (result, value) = top_column(source, target, logp);
        assert_eq!(result, Ok(()), "top {top} target {target} logp {logp}");
        let (at_top, endpoint) = top_column(source, top, logp);
        assert_eq!(at_top, Ok(()));
        assert_eq!(value.to_bits(), endpoint.to_bits(), "top {top} target {target}");
        let top_value = 289.0 - 10.0 * (source.len() - 1) as f32;
        assert!((value - top_value).abs() <= 2.0e-6 * top_value.abs(), "{value} vs {top_value}");
    }

    #[test]
    // Keep the recorded pressure words and their original decimal spellings.
    #[allow(clippy::excessive_precision)]
    fn a_moist_native_top_is_colocated_with_its_dry_target() {
        // HRRR's top mass level at a 1500 Pa lid: dry 1500 + 0.00235 *
        // 98500 = 1731.475 Pa.  The native file holds the FULL pressure,
        // which carries the top half-layer's vapour, 231.475 Pa * qv/(1-qv);
        // integ_moist leaves the top level undried, the target is dry.
        let dry_target = 1731.475f32;
        let mut tops: Vec<f32> = [0.0f64, 1.0e-6, 3.34e-6, 1.0e-5]
            .iter()
            .map(|qv| (1731.475f64 + 231.475 * qv / (1.0 - qv)) as f32)
            .collect();
        // Decoded maxima of the native top level: 2026-10-03 00Z passed
        // under the old four-epsilon bound, 2025-03-14 06Z was refused.
        tops.extend([1731.4755859375f32, 1731.475830078125]);
        for top in tops {
            let source = [98000.0f32, 70000.0, 30000.0, 5000.0, 2209.2018, top];
            for logp in [false, true] {
                assert_colocated(&source, dry_target, logp);
            }
        }
        // A target below the source top (1731.5 Pa under a 1731.475 Pa top)
        // was never at risk and still interpolates inside the column.
        let source = [98000.0f32, 70000.0, 30000.0, 5000.0, 2209.2018, 1731.475];
        for logp in [false, true] {
            assert_eq!(top_column(&source, 1731.5, logp).0, Ok(()));
        }
    }

    #[test]
    // Keep the recorded pressure words and their original decimal spellings.
    #[allow(clippy::excessive_precision)]
    fn a_target_meaningfully_above_the_source_top_is_still_refused() {
        let source = [98000.0f32, 70000.0, 30000.0, 5000.0, 2209.2018, 1731.4755];
        for logp in [false, true] {
            // 50 Pa above, and the first FP32 word past the 2^-16 bound.
            let top = source[source.len() - 1];
            let mut outside = top;
            while top - outside <= TOP_COLOCATION_RTOL * top {
                outside = f32::from_bits(outside.to_bits() - 1);
            }
            let inside = f32::from_bits(outside.to_bits() + 1);
            for target in [top - 50.0, 1500.0, 1731.40, outside] {
                assert_eq!(top_column(&source, target, logp).0,
                           Err(ERR_TARGET_ABOVE_TOP), "target {target}");
            }
            assert_colocated(&source, inside, logp);
        }
    }

    #[test]
    fn the_colocation_bound_is_decided_in_raw_pressure_at_every_top() {
        // The check runs on pressures before any logarithm, so two
        // distinct pressures with one rounded logarithm cannot slip a
        // target past the bound.  Sweep source tops from 10 Pa to 10 kPa.
        let mut top = 10.0f32;
        while top < 10000.0 {
            let mut outside = top;
            while top - outside <= TOP_COLOCATION_RTOL * top {
                outside = f32::from_bits(outside.to_bits() - 1);
            }
            let inside = f32::from_bits(outside.to_bits() + 1);
            let source = [top * 9.0, top * 6.0, top * 4.0, top];
            for logp in [false, true] {
                assert_eq!(top_column(&source, outside, logp).0,
                           Err(ERR_TARGET_ABOVE_TOP), "top {top}");
                assert_colocated(&source, inside, logp);
            }
            top *= 1.37;
        }
    }
}

pub mod lake_state;
