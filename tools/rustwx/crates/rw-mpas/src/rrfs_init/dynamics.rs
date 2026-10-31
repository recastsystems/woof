//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
//! Whole RRFS moisture, base-state, balance and wind sibling.
//! Moisture, the base state, hydrostatic balance and the wind fields.
//!
//! Everything from the vertically interpolated `t`, `pressure` and `relhum`
//! to the model state the dycore starts from, in the order
//! `init_atm_case_gfs` does it -- the order is essential, because `relhum`
//! is used for `qv` and *then* rewritten with respect to ice, and because the
//! hydrostatic fixed point runs on a `pp` that the base-state step has
//! already coupled to the vertical metric.
//!
//! ## The inherited arithmetic quirk, carried deliberately, and what it is not
//! The virtual-temperature factor is `rvord - 1` (0.60836...) when `rho_zz` is
//! first formed, and the literal `1.61` in the level-1 seed, in the
//! hydrostatic loop, in `theta_m`, and in the recovery of `theta` from
//! `theta_m`.
//!
//! One correction to the design note, found by test rather than by reading:
//! **the `theta` round trip does close exactly.**  Step 4 builds `theta` as
//! `T*(p0/p)^(R/cp)` with no virtual factor at all: the `rvord - 1` there
//! sits in `rho_zz`'s denominator, not in `theta`, and steps 9 and 12
//! multiply and divide by the *same* `1.61` factor.  What the mix actually
//! moves is `rho_zz`: the value formed at step 4 with `rvord - 1` is
//! recomputed inside the hydrostatic loop with `1.61`, so the density (and
//! through it `pressure`, `rho_p`, `theta_m`, `ru` and `w`) carries the
//! inconsistency while `theta` does not.
//!
//! [`VirtualFactor`] carries both arms.  `ReproduceFortran` is the comparison
//! arm; `Consistent` uses `rvord - 1` in all five places.  Neither is silently
//! made the other.

#![allow(clippy::needless_range_loop)]

use super::vinterp::{vertical_interp, Extrap};

/// MPAS's own constants, from `mpas_constants.F`.
pub mod constants {
    pub const GRAVITY: f32 = 9.80616;
    pub const RGAS: f32 = 287.0;
    pub const RV: f32 = 461.6;
    pub const CP: f32 = 7.0 * RGAS / 2.0;
    pub const CV: f32 = CP - RGAS;
    pub const RVORD: f32 = RV / RGAS;
    pub const P0: f32 = 1.0e5;
    /// The isothermal base state's temperature, a `parameter` inside
    /// `init_atm_case_gfs`.
    pub const T0B: f32 = 250.0;
    pub const SVP1: f32 = 0.6112;
    pub const SVP2: f32 = 17.67;
    pub const SVP3: f32 = 29.65;
    pub const SVPT0: f32 = 273.15;
}

use constants::*;

/// Which virtual-temperature factor the chain uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualFactor {
    /// `rvord - 1` at first formation, `1.61` in the three later places.
    /// What the Fortran does; the comparison arm.
    ReproduceFortran,
    /// `rvord - 1` throughout, so the `theta` round trip closes.
    Consistent,
}

impl VirtualFactor {
    /// The factor the *later* three uses take.
    #[inline]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
    fn late(self) -> f32 {
        match self {
            VirtualFactor::ReproduceFortran => 1.61,
            VirtualFactor::Consistent => RVORD - 1.0,
        }
    }
}

/// Liquid saturation mixing ratio, the Thompson polynomial MPAS carries.
///
/// The comment at the call site is essential: ungrib's RH is always with
/// respect to liquid water (see `fix_gfs_rh` in `WPS/ungrib/src/rrpr.F`), so
/// this is the right saturation to divide by and an ice saturation is not.
#[allow(clippy::excessive_precision)]
/// MPAS v8.4.2 core_atmosphere/physics/mpas_atmphys_functions.F:156-186.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn rslf(p: f32, t: f32) -> f32 {
    // Transcribed at the Fortran's own written precision, not rounded to what
    // f32 can hold: the digits are the provenance.
    const C: [f32; 9] = [
        0.611583699e03,
        0.444606896e02,
        0.143177157e01,
        0.264224321e-1,
        0.299291081e-3,
        0.203154182e-5,
        0.702620698e-8,
        0.379534310e-11,
        -0.321582393e-13,
    ];
    let x = (t - 273.16).max(-80.0);
    let mut esl = C[8];
    for c in C.iter().take(8).rev() {
        esl = c + x * esl;
    }
    let esl = esl.min(p * 0.15);
    0.622 * esl / (p - esl)
}

/// Rewrite RH with respect to ice below freezing, after it has been used for
/// `qv`.  The ungrib reconstruction, including the linear blend GFS applies
/// between -20 C and 0 C.
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:7990-8035.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn convert_relhum_wrt_ice(t: &[f32], relhum: &mut [f32]) {
    for (i, &temp) in t.iter().enumerate() {
        if temp <= 273.15 {
            let eis = 0.01
                * (9.550426 - (5723.265 / temp) + (3.53068 * temp.ln()) - (0.00728332 * temp))
                    .exp();
            let mut ews = 6.112 * (17.67 * (temp - 273.15) / ((temp - 273.15) + 243.5)).exp();
            let r1 = if temp > 253.15 {
                let f = (273.15 - temp) / 20.0;
                (f * eis) + ((1.0 - f) * ews)
            } else {
                eis
            };
            let r1 = r1.max(1.0e-12);
            ews = ews.max(0.0);
            relhum[i] = (ews / r1 * relhum[i]).min(100.0).max(0.0);
        }
    }
}

/// Two-metre specific humidity from the two-metre temperature, surface
/// pressure and two-metre relative humidity.
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:5248-5251.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn diagnose_q2(t2m: f32, psfc: f32, rh2: f32) -> f32 {
    let es = 6.112 * ((17.27 * (t2m - 273.16)) / (t2m - 35.86)).exp();
    let rs = 0.622 * es * 100.0 / (psfc - es * 100.0);
    0.01 * rs * rh2
}

/// The first-guess surface value of `q2` the case code computes while the
/// surface level is still in hand, with `svp1/svp2/svp3` rather than the
/// 6.112/17.27/35.86 triple used later.  Both are in the Fortran; both are
/// kept, because the later one overwrites the earlier one and a reader
/// comparing the two needs to see that.
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:4128-4130.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn diagnose_q2_surface_pass(t2m: f32, psfc: f32, rh2: f32) -> f32 {
    let mut es = SVP1 * 10.0 * (SVP2 * (t2m - SVPT0) / (t2m - SVP3)).exp();
    es = es.min(0.99 * 0.01 * psfc);
    let rs = 0.622 * es * 100.0 / (psfc - es * 100.0);
    0.01 * rs * rh2
}

/// One column's worth of the state the chain builds.
#[derive(Debug, Default, Clone)]
pub struct ColumnState {
    pub qv: Vec<f32>,
    pub theta: Vec<f32>,
    pub theta_m: Vec<f32>,
    pub pressure: Vec<f32>,
    pub pressure_base: Vec<f32>,
    pub pressure_p: Vec<f32>,
    pub exner: Vec<f32>,
    pub exner_base: Vec<f32>,
    pub rho_zz: Vec<f32>,
    pub rho: Vec<f32>,
    pub rho_base: Vec<f32>,
    pub rho_p: Vec<f32>,
    pub theta_base: Vec<f32>,
    pub rtheta_base: Vec<f32>,
    pub precipw: f32,
    pub surface_pressure: f32,
    /// Hydrostatic passes actually used, per level.  Reported as a
    /// distribution so a delta traceable to the fixed point can be attributed
    /// to it rather than absorbed into a tolerance.
    pub hydrostatic_iterations: Vec<u32>,
}

/// Build one cell's column, from the vertically interpolated fields up.
///
/// `t` enters as temperature and leaves the function having been turned into
/// potential temperature and then `theta_m`, exactly as the Fortran reuses its
/// own array.
#[allow(clippy::too_many_arguments)]
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:5177-5410.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn build_column(
    t_in: &[f32],
    pressure_in: &[f32],
    relhum: &[f32],
    spechum: &[f32],
    use_spechumd: bool,
    zgrid: &[f32],
    zz: &[f32],
    fzm: &[f32],
    fzp: &[f32],
    dzu: &[f32],
    rdzw0: f32,
    factor: VirtualFactor,
) -> ColumnState {
    let nz = t_in.len();
    let mut s = ColumnState {
        qv: vec![0.0; nz],
        theta: vec![0.0; nz],
        theta_m: vec![0.0; nz],
        pressure: pressure_in.to_vec(),
        pressure_base: vec![0.0; nz],
        pressure_p: vec![0.0; nz],
        exner: vec![0.0; nz],
        exner_base: vec![0.0; nz],
        rho_zz: vec![0.0; nz],
        rho: vec![0.0; nz],
        rho_base: vec![0.0; nz],
        rho_p: vec![0.0; nz],
        theta_base: vec![0.0; nz],
        rtheta_base: vec![0.0; nz],
        precipw: 0.0,
        surface_pressure: 0.0,
        hydrostatic_iterations: vec![0; nz],
    };

    // 1. Water vapour.
    for k in 0..nz {
        s.qv[k] = if use_spechumd {
            spechum[k] / (1.0 - spechum[k])
        } else {
            let rs = rslf(s.pressure[k], t_in[k]);
            0.01 * rs * relhum[k]
        };
    }

    // 4. Exner, potential temperature, dry density.
    let mut theta = vec![0.0f32; nz];
    for k in 0..nz {
        s.exner[k] = (s.pressure[k] / P0).powf(RGAS / CP);
        theta[k] = t_in[k] * (P0 / s.pressure[k]).powf(RGAS / CP);
        let early = match factor {
            VirtualFactor::ReproduceFortran => RVORD - 1.0,
            VirtualFactor::Consistent => RVORD - 1.0,
        };
        s.rho_zz[k] = s.pressure[k] / RGAS / (s.exner[k] * theta[k] * (1.0 + early * s.qv[k]));
        s.rho_zz[k] /= 1.0 + s.qv[k];
    }

    // 5. Precipitable water, before the metric coupling.
    for k in 0..nz {
        s.precipw += s.rho_zz[k] * s.qv[k] * (zgrid[k + 1] - zgrid[k]);
    }

    // 6. Base state: a dry isothermal atmosphere at t0b.
    for k in 0..nz {
        let ztemp = 0.5 * (zgrid[k + 1] + zgrid[k]);
        // Strict fork ELF 0x8e8c1-0x8e8d1 divides the positive product,
        // then flips its sign bit before expf. Moving the sign into the
        // gravity constant changes the measured NaN sign.
        let quotient = std::hint::black_box(GRAVITY * ztemp / (RGAS * T0B));
        s.pressure_base[k] = P0 * (-quotient).exp();
        s.exner_base[k] = (s.pressure_base[k] / P0).powf(RGAS / CP);
        s.rho_base[k] = s.pressure_base[k] / (RGAS * T0B);
        s.theta_base[k] = T0B / s.exner_base[k];
        s.rtheta_base[k] = s.rho_base[k] * s.theta_base[k];
        s.exner[k] = s.exner_base[k];
        s.pressure_p[k] = 0.0;
        s.rho_p[k] = 0.0;
    }

    // 7. Couple to the vertical metric.
    for k in 0..nz {
        s.rho_base[k] /= zz[k];
        s.rho_zz[k] /= zz[k];
        s.pressure_p[k] = s.pressure[k] - s.pressure_base[k];
        s.rho_p[k] = s.rho_zz[k] - s.rho_base[k];
    }

    // 8. Hydrostatic rebalance.  Level 1 is seeded, then each level above runs
    //    a fixed point of at most thirty passes, converged when |dpp| <= 1e-4 Pa.
    let late = factor.late();
    s.rho_zz[0] = ((s.pressure[0] / P0).powf(CV / CP)) * (P0 / RGAS)
        / (theta[0] * (1.0 + late * s.qv[0]))
        / zz[0];
    s.rho_p[0] = s.rho_zz[0] - s.rho_base[0];
    for k in 1..nz {
        let mut it = 0u32;
        let mut p_check = 2.0 * 0.0001f32;
        while it < 30 && p_check > 0.0001 {
            p_check = s.pressure_p[k];
            s.pressure_p[k] = s.pressure_p[k - 1]
                - (fzm[k] * s.rho_p[k] + fzp[k] * s.rho_p[k - 1]) * GRAVITY * dzu[k]
                - (fzm[k] * s.rho_zz[k] * s.qv[k] + fzp[k] * s.rho_zz[k - 1] * s.qv[k - 1])
                    * GRAVITY
                    * dzu[k];
            s.pressure[k] = s.pressure_p[k] + s.pressure_base[k];
            s.exner[k] = (s.pressure[k] / P0).powf(RGAS / CP);
            s.rho_zz[k] = s.pressure[k]
                / RGAS
                / (s.exner[k] * theta[k] * (1.0 + late * s.qv[k]))
                / zz[k];
            s.rho_p[k] = s.rho_zz[k] - s.rho_base[k];
            p_check = (p_check - s.pressure_p[k]).abs();
            it += 1;
        }
        s.hydrostatic_iterations[k] = it;
    }

    // 9. theta_m, and decouple rho_p from the metric.
    for k in 0..nz {
        s.theta_m[k] = theta[k] * (1.0 + late * s.qv[k]);
        s.rho_p[k] *= zz[k];
    }

    // 11. The explicitly ad-hoc diagnostic surface pressure.
    s.surface_pressure = 0.5 * GRAVITY / rdzw0
        * (1.25 * s.rho_zz[0] * (1.0 + s.qv[0]) - 0.25 * s.rho_zz[1] * (1.0 + s.qv[1]))
        + s.pressure_p[0]
        + s.pressure_base[0];

    // 12. rho and the recovered theta.
    for k in 0..nz {
        s.rho[k] = s.rho_zz[k] * zz[k];
        s.theta[k] = s.theta_m[k] / (1.0 + late * s.qv[k]);
    }

    s
}

/// Coupled horizontal/vertical winds at the case-7 driver boundary.
pub struct WindState {pub ru:Vec<Vec<f32>>,pub rw:Vec<Vec<f32>>,pub w:Vec<Vec<f32>>}
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_cases.F:5336-5395.
/// Connectivity here retains the file's one-based indexing.
#[allow(clippy::too_many_arguments)]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7670-7721 at 1d6e370e.
/// Preserve the pinned strict GNU scalar instruction's destination operand.
#[inline]
fn wind_add(mut destination:f32, source:f32)->f32 {
    unsafe { core::arch::asm!("addss {destination}, {source}",
        destination=inout(xmm_reg) destination, source=in(xmm_reg) source,
        options(pure,nomem,nostack,preserves_flags)); }
    destination
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7670-7721 at 1d6e370e.
/// Preserve the pinned strict GNU scalar instruction's destination operand.
#[inline]
fn wind_mul(mut destination:f32, source:f32)->f32 {
    unsafe { core::arch::asm!("mulss {destination}, {source}",
        destination=inout(xmm_reg) destination, source=in(xmm_reg) source,
        options(pure,nomem,nostack,preserves_flags)); }
    destination
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7670-7721 at 1d6e370e.
/// Preserve the pinned strict GNU scalar instruction's destination operand.
#[inline]
fn wind_sub(mut destination:f32, source:f32)->f32 {
    unsafe { core::arch::asm!("subss {destination}, {source}",
        destination=inout(xmm_reg) destination, source=in(xmm_reg) source,
        options(pure,nomem,nostack,preserves_flags)); }
    destination
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7670-7721 at 1d6e370e.
/// Preserve the pinned strict GNU scalar instruction's destination operand.
#[inline]
fn wind_div(mut destination:f32, source:f32)->f32 {
    unsafe { core::arch::asm!("divss {destination}, {source}",
        destination=inout(xmm_reg) destination, source=in(xmm_reg) source,
        options(pure,nomem,nostack,preserves_flags)); }
    destination
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn diagnose_winds(u_edge:&[Vec<f32>],rho_zz:&[&[f32]],owners:&[[usize;2]],
    counts:&[usize],edges:&[Vec<usize>],metrics:&super::capsule::VerticalMetrics,
    order:i32,coefficient:f32)->WindState {
    use rayon::prelude::*;
    let n_cells=rho_zz.len();let n_edges=owners.len();let nz=metrics.n_vert_levels;
    // ru, rw and w. Every edge's ru and every cell's rw and w depend only on
    // their own inputs (rw sums over the cell's own edges in stored order),
    // so edges and cells run on every core with the serial loops' arithmetic.
    let ru: Vec<Vec<f32>> = (0..n_edges).into_par_iter().map(|e| {
        let [c1, c2] = owners[e];
        // The regional extra-cell pool is cold, as in edge interpolation.
        let density = |cell:usize,k:usize| cell.checked_sub(1)
            .filter(|&c|c<n_cells).map_or(0.0,|c|rho_zz[c][k]);
        (0..nz).map(|k| {
            // Strict ELF 0x8d0c0..0x8d0e5: the density sum is
            // the final MULSS destination, so both-NaN operands retain
            // its payload rather than the wind's payload.
            wind_mul(wind_add(density(c2,k),density(c1,k)),
                wind_mul(u_edge[e][k],0.5))
        }).collect()
    }).collect();
    // w and rw are dimensioned against nVertLevelsP1 in the file, and the
    // case code fills only k = 2 .. nVertLevels; the bottom and the top stay
    // at zero.  Allocating nz here instead would emit a short slab.
    let nzp1 = nz + 1;
    let mut rw = super::zero_columns(n_cells, nzp1);
    rw.par_iter_mut().enumerate().for_each(|(c, rw_c)| {
        for i in 0..counts[c] {
            let e = edges[c][i];
            if e == 0 || e > n_edges {
                continue;
            }
            let e = e - 1;
            let side_one = owners[e][0] == c + 1;
            for k in 1..nz {
                // Strict ELF 0x8e01b..0x8e067 and 0x8e150..0x8e1f6.
                let flux = wind_add(wind_mul(metrics.fzp[k],ru[e][k-1]),
                    wind_mul(ru[e][k],metrics.fzm[k]));
                let metric = wind_add(wind_mul(metrics.fzp[k],metrics.zz[c][k-1]),
                    wind_mul(metrics.fzm[k],metrics.zz[c][k]));
                if side_one {
                    rw_c[k] = wind_sub(rw_c[k],wind_mul(wind_mul(metric,metrics.zb[e][0][k]),flux));
                } else {
                    rw_c[k] = wind_add(wind_mul(wind_mul(metric,metrics.zb[e][1][k]),flux),rw_c[k]);
                }
                if order == 3 {
                    let sign = if ru[e][k].is_sign_negative() { -1.0f32 } else { 1.0f32 };
                    if side_one {
                        rw_c[k] = wind_add(wind_mul(wind_mul(wind_mul(metric,wind_mul(sign,coefficient)),metrics.zb3[e][0][k]),flux),rw_c[k]);
                    } else {
                        rw_c[k] = wind_sub(rw_c[k],wind_mul(wind_mul(wind_mul(metric,wind_mul(sign,coefficient)),metrics.zb3[e][1][k]),flux));
                    }
                }
            }
        }
    });
    let mut w = super::zero_columns(n_cells, nzp1);
    w.par_iter_mut().enumerate().for_each(|(c, w_c)| {
        for k in 1..nz {
            // Strict ELF 0x8da40..0x8da73, numerator destination.
            w_c[k] = wind_div(rw[c][k],wind_add(
                wind_mul(rho_zz[c][k-1],metrics.fzp[k]),
                wind_mul(metrics.fzm[k],rho_zz[c][k])));
        }
    });

    WindState {ru,rw,w}
}

/// Rotate a first-guess wind pair to the edge normal.
#[inline]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn edge_normal_wind(angle_edge: f32, u: f32, v: f32) -> f32 {
    angle_edge.cos() * u + angle_edge.sin() * v
}

/// The two-cell mean coordinate an edge column is interpolated on.
#[inline]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn edge_column_coordinate(z1: f32, z2: f32) -> f32 {
    0.5 * (z1 + z2)
}

/// The four-point mean an edge's target height is.
#[inline]
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn edge_target_height(zk1: f32, zkp1: f32, zk2: f32, zkp2: f32) -> f32 {
    0.25 * (zk1 + zkp1 + zk2 + zkp2)
}

/// Interpolate one edge column with `extrap = 0`.
///
/// Case 7 uses constant extrapolation at edges; `init_atm_case_lbc` uses
/// linear for the same field.  That difference is real and the two are not
/// unified here.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7499-7742,10305-10365 at 1d6e370e.
pub fn interp_edge_column(target: f32, coords: &[f32], values: &[f32]) -> f32 {
    vertical_interp(target, coords, values, Extrap::Constant).unwrap_or(values[values.len() - 1])
}

