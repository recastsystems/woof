//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
//! Case-7 terrain, coordinate surfaces and omega metrics, in native RKIND order.
//! The original recipe fixes hybrid=true and zh=30000. The configurable
//! recipe follows 44c0849 cases.F:4675-4698. Damping remains fixed.
//! Each protocol call owns the complete mesh (or a caller-owned working halo).
//! No contraction, parallel reduction or algebraic rearrangement is used.
//! Every loop below that runs on all cores computes each stored word on its
//! own, in the same operation order as the serial loop (one cell, edge or
//! level per task, no shared accumulator), so the output words are the
//! serial words. The serial loops ran on one core for minutes on a 2.3 M-cell
//! mesh (RRFS v2 conus3km, 2026-10-28) while the other cores waited.

use crate::geometry::Real;
use rayon::prelude::*;
use serde::Deserialize;
use std::io::{Read, Write};

pub const ABI_MARKER: &str = "rw_mpas_geometry --protocol hex-vertical-v1";

/// Coordinate policy supplied by the producer table, independent of model names.
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all="kebab-case")]
pub enum CoordinateRecipe {
    #[default]
    FixedHybrid,
    ConfiguredHybrid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub mode: String,
    pub width: usize,
    pub terrain: Vec<f64>,
    pub counts: Vec<usize>,
    pub edges_on_cell: Vec<Vec<i64>>,
    pub cells_on_cell: Vec<Vec<i64>>,
    pub cells_on_edge: Vec<[i64; 2]>,
    pub dc: Vec<f64>,
    pub dv: Vec<f64>,
    pub area: Vec<f64>,
    pub levels: usize,
    pub top: f64,
    pub scheme: String,
    pub specified: Option<Vec<f64>>,
    pub projection: String,
    pub terrain_passes: usize,
    pub smooth: bool,
    pub surface_passes: usize,
    pub minimum_fraction: f64,
    pub hybrid: bool,
    pub transition: f64,
    #[serde(default)]
    pub coordinate_recipe: CoordinateRecipe,
    pub damping: f64,
    pub damping_start: f64,
    pub regional_halo: bool,
    pub order: i32,
    /// Edge-major, side-major, coefficient-fastest.
    pub deriv: Vec<Vec<Vec<f64>>>,
    /// Level-major, cell-fastest; only the edge mode reads it.
    pub zgrid: Vec<Vec<f64>>,
    #[serde(default)]
    pub boundary: Option<Boundary>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub met_path: String,
    pub lat: Vec<f32>, pub lon: Vec<f32>, pub mask: Vec<i32>,
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
fn fail(message: &str) -> Result<(), String> { Err(message.to_string()) }

/// Validate array topology before any indexing; invalid neighbors otherwise
/// address unrelated cells or panic. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4490-4558,4854-4895 at 1d6e370e.
fn validate(r: &Request) -> Result<(), String> {
    let nc = r.terrain.len();
    let ne = r.dc.len();
    if nc == 0 || ne == 0 || r.counts.len() != nc || r.edges_on_cell.len() != nc
        || r.cells_on_cell.len() != nc || r.dv.len() != ne || r.area.len() != nc
        || r.cells_on_edge.len() != ne {
        return fail("vertical topology/metric dimensions are inconsistent");
    }
    // RRFSx/MPAS-Model cases.F:4490-4558,4805-4895 at 1d6e370e
    // performs IEEE arithmetic on stored fields, including NaN terrain from
    // empty geography sampling. The pristine strict vertical-only decks emit
    // these fields successfully; field values are not an admission condition.
    for c in 0..nc {
        let n = r.counts[c];
        if n == 0 || r.edges_on_cell[c].len() < n || r.cells_on_cell[c].len() < n {
            return fail("nEdgesOnCell exceeds connectivity rows or is zero");
        }
        for j in 0..n {
            if r.edges_on_cell[c][j] < 0 || r.edges_on_cell[c][j] >= ne as i64 || r.cells_on_cell[c][j] < -1
                || r.cells_on_cell[c][j] >= nc as i64 {
                return fail("vertical connectivity contains an out-of-range cell or edge");
            }
        }
    }
    for (edge, &[a,b]) in r.cells_on_edge.iter().enumerate() {
        if a < -1 || b < -1 || a >= nc as i64 || b >= nc as i64 || (a == -1 && b == -1) {
            return Err(if a == -1 && b == -1 { format!("edge {edge} has no interior cell") } else { format!("edge {edge} references an out-of-range cell") });
        }
        if (a == -1 || b == -1) && !r.regional_halo {
            return fail("regional_boundary_metrics requires a caller-owned working halo");
        }
    }
    Ok(())
}

/// Ordered unscaled Laplacian and two-pass smoother, including garbage-cell
/// substitution and the exact-zero guard. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4490-4558 at 1d6e370e.
fn laplacian<T: Real>(r: &Request, field: &[T], skip_zero: bool) -> Vec<T> {
    let mut result = vec![T::zero(); field.len()];
    result.par_iter_mut().enumerate().for_each(|(c, out)| {
        if skip_zero && field[c] == T::zero() { return; }
        let mut sum = T::zero();
        for j in 0..r.counts[c] {
            let e = r.edges_on_cell[c][j] as usize;
            let neighbor = r.cells_on_cell[c][j];
            let value = if neighbor < 0 { field[c] } else { field[neighbor as usize] };
            sum = sum + T::number(r.dv[e])/T::number(r.dc[e])*(value-field[c]);
        }
        *out = sum;
    });
    result
}

/// Fourth-order terrain smoothing. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4490-4558 at 1d6e370e.
fn smooth_terrain<T: Real>(r: &Request) -> Vec<T> {
    let mut terrain: Vec<T> = r.terrain.iter().map(|&v| T::number(v)).collect();
    for _ in 0..r.terrain_passes {
        let first = laplacian(r, &terrain, true);
        let hs: Vec<T> = terrain.iter().zip(first).map(|(&v,l)| v+T::number(0.216)*l).collect();
        let second = laplacian(r, &hs, true);
        terrain = hs.iter().zip(second).map(|(&v,l)| v-T::number(0.216)*l).collect();
    }
    terrain
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
fn write<T: Real>(out: &mut impl Write, values: &[T]) -> Result<(),String> {
    for &v in values { v.encode(out)?; }
    Ok(())
}

/// All three interface branches and both projections; hybrid/basic coordinate,
/// iterative surface acceptance and metric closure. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4568-4853 at 1d6e370e.
fn grid<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
    let n = r.levels;
    let nc = r.terrain.len();
    if n < 3 { return fail("lower-boundary extrapolation requires at least three levels"); }
    if !r.minimum_fraction.is_finite() || r.minimum_fraction <= 0.0 || r.minimum_fraction >= 1.0 {
        return fail("minimum layer fraction must lie strictly between zero and one");
    }
    let one = T::one();
    let half = T::half();
    let pi = T::number(std::f64::consts::PI);
    let mut top = T::number(r.top);
    let terrain = smooth_terrain::<T>(r);
    let mut zw = vec![T::zero(); n+1];
    if r.scheme == "specified" {
        let specified = r.specified.as_ref().ok_or("specified grid needs interface heights")?;
        if specified.len() != n+1 { return fail("specified heights must contain nVertLevels+1 values"); }
        for k in 0..=n { zw[k] = T::number(specified[k]); }
        top = zw[n];
    } else {
        if !top.finite() || top <= T::zero() { return fail("model top must be finite and positive"); }
        let dz = top/T::count(n);
        let (als,alt,zetal) = if n >= 55 { (0.075,1.70,0.75) } else { (0.075,1.23,0.31) };
        let (als,alt,zetal) = (T::number(als),T::number(alt),T::number(zetal));
        for k in 0..=n {
            let eta = T::count(k)/T::count(n);
            zw[k] = match r.scheme.as_str() {
                "legacy" => eta.pow(T::number(1.5))*top,
                "tc" if eta < zetal => {
                    let x = T::count(k)*dz/(top*zetal);
                    // GNU integer powers: x**2 = x*x; x**3 = (x*x)*x.
                    (als*T::count(k)/T::count(n)
                        +(T::number(3.0)*(one-alt)+T::number(2.0)*(alt-als)*zetal)*(x*x)
                        -(T::number(2.0)*(one-alt)+(alt-als)*zetal)*((x*x)*x))*top
                }
                "tc" => {
                    let zl = one-alt*(one-zetal);
                    (zl+alt*(eta-zetal))*top
                }
                _ => return fail("vertical scheme must be tc, legacy or specified"),
            };
        }
    }
    if zw.iter().any(|v| !v.finite()) || zw.windows(2).any(|p| p[1] <= p[0]) {
        return fail("vertical interfaces must be finite and strictly increasing");
    }
    if zw[0] != T::zero() { return fail("first vertical interface must be zero"); }
    let (hybrid, transition) = match r.coordinate_recipe {
        CoordinateRecipe::FixedHybrid => (true, 30000.0),
        CoordinateRecipe::ConfiguredHybrid => (r.hybrid, r.transition),
    };
    let zh=if hybrid {T::number(transition)} else {top};
    if !zh.finite() || zh <= T::zero() { return fail("hybrid transition must be finite and positive"); }
    let mut kz = n; // Fortran kz=nz => zero-based n (even if no height level).
    let mut ah = vec![T::zero(); n+1];
    for k in 0..=n {
        ah[k] = if hybrid {
            if zw[k] < zh {
                let a = (half*pi*zw[k]/zh).cos();
                let a2 = a*a;
                // GNU powi(6) multiplies the cube by itself.
                let a3 = a2*a;
                a3*a3
            } else { kz = kz.min(k); T::zero() }
        } else { one-zw[k]/top };
    }
    let mut dzw = vec![T::zero(); n];
    let mut rdzw = dzw.clone();
    let mut zu = dzw.clone();
    let mut dzu = dzw.clone();
    let mut rdzu = dzw.clone();
    let mut rdzwp = dzw.clone();
    let mut rdzwm = dzw.clone();
    let mut fzp = dzw.clone();
    let mut fzm = dzw.clone();
    for k in 0..n {
        dzw[k] = zw[k+1]-zw[k];
        rdzw[k] = one/dzw[k];
        zu[k] = half*(zw[k]+zw[k+1]);
    }
    for k in 1..n {
        dzu[k] = half*(dzw[k]+dzw[k-1]);
        rdzu[k] = one/dzu[k];
        rdzwp[k] = dzw[k-1]/(dzw[k]*(dzw[k]+dzw[k-1]));
        rdzwm[k] = dzw[k]/(dzw[k-1]*(dzw[k]+dzw[k-1]));
        let a = half*dzw[k]/dzu[k];
        let b = half*dzw[k-1]/dzu[k];
        match r.projection.as_str() {
            "linear_interpolation" => { fzp[k] = a; fzm[k] = b; }
            "layer_integral" => { fzm[k] = a; fzp[k] = b; }
            _ => return fail("projection must be linear_interpolation or layer_integral"),
        }
    }
    let cof1 = (T::number(2.0)*dzu[1]+dzu[2])/(dzu[1]+dzu[2])*dzw[0]/dzu[1];
    let cof2 = dzu[1]/(dzu[1]+dzu[2])*dzw[0]/dzu[2];
    let cf = [fzp[1]+cof1, (fzm[1]-cof1)-cof2, cof2];
    let mut hx: Vec<Vec<T>> = vec![terrain; n+1];
    if r.smooth {
        let mut sm0 = vec![T::zero(); nc];
        sm0.par_iter_mut().enumerate().for_each(|(c, out)| {
            let mut sum = T::zero();
            for j in 0..r.counts[c] { sum = sum+T::number(r.dc[r.edges_on_cell[c][j] as usize]); }
            sum = sum/T::count(r.counts[c]);
            let mut factor = T::number(3000.0)/sum;
            if factor > one { factor = one; }
            factor = T::number(0.125)*factor;
            if factor < T::number(0.01) { factor = T::number(0.01); }
            *out = factor;
        });
        for k in 1..kz {
            hx[k] = hx[k-1].clone();
            let a = T::number(3.0)*zw[k]/top;
            let mut factor = a*a;
            if factor > one { factor = one; }
            for _ in 0..r.surface_passes+k+1 {
                let lap = laplacian(r, &hx[k], false);
                let (below, level) = hx.split_at_mut(k);
                let previous = &below[k-1];
                let limit = T::number(r.minimum_fraction)*(zw[k]-zw[k-1]);
                let (zk, zk1, ahk, ahk1) = (zw[k], zw[k-1], ah[k], ah[k-1]);
                level[0].par_iter_mut().enumerate().for_each(|(c, h)| {
                    let sm = sm0[c]*factor;
                    let hs = *h+sm*lap[c];
                    let dz = (zk+ahk*hs)-(zk1+ahk1*previous[c]);
                    if dz > limit { *h = hs; }
                });
            }
        }
        for row in &mut hx[kz..] { row.fill(T::zero()); }
    }
    let mut zg = vec![vec![T::zero();nc];n+1];
    let mut zz = vec![vec![T::zero();nc];n];
    zg.par_iter_mut().enumerate().for_each(|(k, row)| { for c in 0..nc { row[c] = zw[k]+ah[k]*hx[k][c]; } });
    zz.par_iter_mut().enumerate().for_each(|(k, row)| { for c in 0..nc {
        let dz = zg[k+1][c]-zg[k][c];
        row[c] = (zw[k+1]-zw[k])/dz;
    } });
    let mut zxu = vec![vec![T::zero();r.dc.len()];n];
    zxu.par_iter_mut().enumerate().for_each(|(k, row)| {
    for e in 0..r.dc.len() {
        let [a,b] = r.cells_on_edge[e];
        {
            // RRFSx/MPAS-Model cases.F:4826-4834 at 1d6e370e reads the
            // cold extra-cell zgrid here. The explicit neighbor substitution
            // at 4854-4863 belongs only to the following omega metrics.
            // Whole public CONUS replay measured 184257 exterior-edge words
            // differing when an interior height was substituted here.
            let a0 = if a < 0 {T::zero()} else {zg[k][a as usize]};
            let a1 = if a < 0 {T::zero()} else {zg[k+1][a as usize]};
            let b0 = if b < 0 {T::zero()} else {zg[k][b as usize]};
            let b1 = if b < 0 {T::zero()} else {zg[k+1][b as usize]};
            row[e] = half*(((b0-a0)+b1)-a1)/T::number(r.dc[e]);
        }
    }
    });
    let mut dss = vec![vec![T::zero();nc];n];
    // Fork cases.F:4450-4451 fixes the initialization damping fields.
    let zd = T::number(12000.0);
    let amp = T::zero();
    if !zd.finite() || !amp.finite() || (amp != T::zero() && zd >= top) {
        return fail("nonzero damping requires a finite start below model top");
    }
    dss.par_iter_mut().enumerate().for_each(|(k, row)| { for c in 0..nc {
        if zg[k][c] > zd+T::number(0.1) {
            let a = (half*pi*(zg[k][c]-zd)/(top-zd)).sin();
            row[c] = T::zero()+amp*(a*a);
        }
    } });
    for array in [&zw,&dzw,&rdzw,&zu,&dzu,&rdzu,&rdzwp,&rdzwm,&fzp,&fzm,&ah] { write(out,array)?; }
    for matrix in [&hx,&zg,&zz,&zxu,&dss] { for row in matrix { write(out,row)?; } }
    write(out,&cf)?;
    out.write_all(&(kz as u64).to_le_bytes()).map_err(|e|e.to_string())
}

/// Omega edge metrics. Arithmetic deliberately matches multiply then divide,
/// rather than pre-dividing geometric factors. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4840-4895 at 1d6e370e.
fn edge<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
    let n = r.levels;
    let nc = r.terrain.len();
    let ne = r.dc.len();
    if ![2,3,4].contains(&r.order) { return fail("theta advection order must be 2, 3 or 4"); }
    if r.zgrid.len() != n+1 || r.zgrid.iter().any(|row| row.len() != nc) {
        return fail("edge metric zgrid dimensions are inconsistent");
    }
    // The widest cell is found once. It used to be found again inside the
    // check for every edge and side, a full pass over the cells each time:
    // 7.0 M edges x 2 x 2.3 M cells on RRFS v2 conus3km, tens of minutes on
    // one core before any metric was computed (2026-10-28).
    let widest = r.counts.iter().copied().max().unwrap_or(0);
    if r.order != 2 && (r.deriv.len() != ne || r.deriv.iter().any(|sides|
        sides.len() != 2 || sides.iter().any(|row| row.len() <= widest))) {
        return fail("deriv_two has insufficient edge/side/neighbor coefficients");
    }
    let zg: Vec<Vec<T>> = r.zgrid.iter().map(|row| row.iter().map(|&v| T::number(v)).collect()).collect();
    let mut zb = vec![T::zero();(n+1)*2*ne];
    let mut zb3 = zb.clone();
    zb.par_chunks_mut(2*ne).zip(zb3.par_chunks_mut(2*ne)).enumerate().take(n).for_each(|(k, (zb_k, zb3_k))| {
    for e in 0..ne {
        let [a,b] = r.cells_on_edge[e];
        let c1 = if a < 0 { b as usize } else { a as usize };
        let c2 = if b < 0 { c1 } else { b as usize };
        {
            // RRFSx/MPAS-Model cases.F:4875-4890 at 1d6e370e does not set z_edge3 in
            // order2. The strict fork stores this inherited scratch word in
            // repeated complete-stage runs at 1/2 ranks on both hosts;
            // receipts/native-stage-word-witness-cross-host-v1.json records
            // all eight matching affected-word inventories.
            let mut z3 = if r.order == 2 {
                T::number(f32::from_bits(0x20202020) as f64)
            } else {T::zero()};
            let z = if r.order == 2 { (zg[k][c1]+zg[k][c2])/T::number(2.0) } else {
                let mut d = [T::zero();2];
                for (side,c) in [c1,c2].into_iter().enumerate() {
                    d[side] = T::number(r.deriv[e][side][0])*zg[k][c];
                    for j in 0..r.counts[c] {
                        let neighbor = r.cells_on_cell[c][j];
                        if neighbor >= 0 {
                            d[side] = d[side]+T::number(r.deriv[e][side][j+1])*zg[k][neighbor as usize];
                        }
                    }
                }
                let dc = T::number(r.dc[e]);
                // Strict fork ELF 0x7f77c-0x7f78b evaluates the product and
                // quotient before xor-ing the result sign. Negating dc**2
                // instead changes the stored NaN sign in whole-file replay.
                // RRFSx/MPAS-Model cases.F:4884 at 1d6e370e.
                if r.order == 3 {
                    // LLVM may move unary negation through multiplication or
                    // division even without fast-math, changing NaN signs.
                    // Keep the strict ELF instruction boundaries observable.
                    let difference = std::hint::black_box(d[0]-d[1]);
                    let product = std::hint::black_box((dc*dc)*difference);
                    let quotient = std::hint::black_box(product/T::number(12.0));
                    z3 = std::hint::black_box(-quotient);
                }
                T::half()*(zg[k][c1]+zg[k][c2])-(dc*dc)*(d[0]+d[1])/T::number(12.0)
            };
            for (side,c) in [c1,c2].into_iter().enumerate() {
                let index = side*ne+e;
                zb_k[index] = (z-zg[k][c])*T::number(r.dv[e])/T::number(r.area[c]);
                zb3_k[index] = z3*T::number(r.dv[e])/T::number(r.area[c]);
            }
        }
    }
    });
    write(out,&zb)?;
    write(out,&zb3)
}

/// Versioned JSON topology/parameter input and raw IEEE output. The binary
/// protocol preserves every field word, including unused +0 slots.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
pub fn run(mut reader: impl Read, mut writer: impl Write) -> Result<(),String> {
    let _abi = std::hint::black_box(ABI_MARKER);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let mut r: Request = serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    run_request(&mut r, writer)
}

/// Typed file-producer entry preserves nonfinite inactive coefficient words.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
pub fn run_request(r: &mut Request, mut writer: impl Write) -> Result<(),String> {
    if let Some(b) = &r.boundary {
        if r.width != 4 || !["grid","terrain"].contains(&r.mode.as_str()) {
            return fail("boundary blending uses native single precision before terrain/grid construction");
        }
        let field = super::metfile::read_first_named_field(std::path::Path::new(&b.met_path),"SOILHGT")
            .map_err(|e|e.to_string())?.ok_or("SOILHGT field not found for boundary blending")?;
        let mut terrain: Vec<f32> = r.terrain.iter().map(|&v|v as f32).collect();
        super::boundary::blend_bdy_terrain(&field,&b.lat,&b.lon,&b.mask,&mut terrain)?;
        r.terrain=terrain.into_iter().map(f64::from).collect();
    }
    validate(&r)?;
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
    fn process<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
        match r.mode.as_str() {
            "terrain" => write(out,&smooth_terrain::<T>(r)),
            "grid" => grid::<T>(r,out),
            "edge" => edge::<T>(r,out),
            _ => fail("vertical mode must be terrain, grid or edge"),
        }
    }
    match r.width {
        4 => process::<f32>(&r,&mut writer),
        8 => process::<f64>(&r,&mut writer),
        _ => fail("vertical width must be float32 or float64"),
    }
}
