//! Dry-pressure coefficients of the WRF vertical coordinate on interfaces.
//!
//! WRF ARW v4 technical note [R1] section 2.1: the dry hydrostatic pressure
//! on a full (interface) level is
//!
//! ```text
//! p_d = B(eta) (p_s - p_t) + (eta - B(eta)) (p_0 - p_t) + p_t
//!     = C3F mu_d + C4F + p_t,       mu_d = MU + MUB = p_s - p_t
//! ```
//!
//! with p_0 = 1e5 Pa and, for the hybrid coordinate (HYBRID_OPT = 2), the
//! cubic B(eta) = c1 + c2 eta + c3 eta^2 + c4 eta^3 above eta_c (0 at and
//! below eta_c) whose coefficients make B continuous with zero slope at
//! eta_c and B(1) = 1:
//!
//! ```text
//! c1 = 2 eta_c^2 / (1 - eta_c)^3
//! c2 = -eta_c (4 + eta_c + eta_c^2) / (1 - eta_c)^3
//! c3 = 2 (1 + eta_c + eta_c^2) / (1 - eta_c)^3
//! c4 = -(1 + eta_c) / (1 - eta_c)^3
//! ```
//!
//! For the terrain-following coordinate (HYBRID_OPT = 0) B(eta) = eta.
//! A history that stores C3F and C4F is used as written.

use thiserror::Error;

/// Where the interface coefficients came from (recorded in manifests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientSource {
    /// C3F and C4F read from the history.
    Stored,
    /// Derived from ZNW and ETAC for HYBRID_OPT = 2.
    HybridFromEtac,
    /// Derived from ZNW for the terrain-following coordinate.
    TerrainFollowing,
    /// A height-coordinate frame (an MPAS history gathered onto a WRF-shaped
    /// grid: no MU, MUB or ZNW).  The mass-level pressure is the history's own
    /// full pressure P + PB, and the interface pressures are filled after the
    /// state is built by [`crate::state::fill_height_coordinate_interfaces`]
    /// (log-linear in height between mass levels, hypsometric beyond them).
    HeightCoordinate,
}

impl CoefficientSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stored => "history C3F/C4F",
            Self::HybridFromEtac => "ZNW + ETAC hybrid B(eta) of NCAR/TN-556 section 2.1",
            Self::TerrainFollowing => "ZNW terrain-following, B(eta) = eta",
            Self::HeightCoordinate => {
                "height coordinate (no MU/ZNW): mass pressure = P + PB; interfaces log-linear in height between mass levels, \
                 PSFC at the ground, hypsometric (layer virtual temperature) above the top mass level"
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum HybridError {
    #[error("HYBRID_OPT = {0} has no ETAC and no stored C3F/C4F")]
    MissingEtac(i32),
    #[error("ETAC = {0} is outside [0, 1)")]
    BadEtac(f64),
    #[error("ZNW has {got} values, expected {want}")]
    Length { got: usize, want: usize },
}

/// Interface coefficients, bottom (eta = 1) to top (eta = 0), binary32 as
/// both devices consume them.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceCoefficients {
    pub c3f: Vec<f32>,
    pub c4f: Vec<f32>,
    pub source: CoefficientSource,
}

/// B(eta) of [R1] for a given eta_c (binary64).
pub fn hybrid_b(eta: f64, etac: f64) -> f64 {
    if eta <= etac {
        return 0.0;
    }
    let one = 1.0 - etac;
    let d = one * one * one;
    let c1 = 2.0 * etac * etac / d;
    let c2 = -etac * (4.0 + etac + etac * etac) / d;
    let c3 = 2.0 * (1.0 + etac + etac * etac) / d;
    let c4 = -(1.0 + etac) / d;
    c1 + c2 * eta + c3 * eta * eta + c4 * eta * eta * eta
}

impl InterfaceCoefficients {
    /// Choose the coefficients for a history: stored C3F/C4F when present,
    /// else derived from ZNW per HYBRID_OPT (and ETAC).
    pub fn resolve(
        nz: usize,
        stored: Option<(Vec<f32>, Vec<f32>)>,
        znw: &[f32],
        hybrid_opt: i32,
        etac: Option<f64>,
        p_top: f32,
    ) -> Result<Self, HybridError> {
        if let Some((c3f, c4f)) = stored {
            if c3f.len() == nz + 1 && c4f.len() == nz + 1 {
                return Ok(Self { c3f, c4f, source: CoefficientSource::Stored });
            }
        }
        if znw.len() != nz + 1 {
            return Err(HybridError::Length { got: znw.len(), want: nz + 1 });
        }
        let p0_minus_pt = 100000.0 - p_top as f64;
        if hybrid_opt == 2 {
            let etac = etac.ok_or(HybridError::MissingEtac(hybrid_opt))?;
            if !(0.0..1.0).contains(&etac) {
                return Err(HybridError::BadEtac(etac));
            }
            let mut c3f = Vec::with_capacity(nz + 1);
            let mut c4f = Vec::with_capacity(nz + 1);
            for &e in znw {
                let eta = e as f64;
                let b = hybrid_b(eta, etac);
                c3f.push(b as f32);
                c4f.push(((eta - b) * p0_minus_pt) as f32);
            }
            Ok(Self { c3f, c4f, source: CoefficientSource::HybridFromEtac })
        } else {
            Ok(Self {
                c3f: znw.to_vec(),
                c4f: vec![0.0; nz + 1],
                source: CoefficientSource::TerrainFollowing,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b_is_continuous_and_normalised() {
        for etac in [0.0, 0.1, 0.2, 0.3] {
            assert!((hybrid_b(1.0, etac) - 1.0).abs() < 1e-14);
            assert!(hybrid_b(etac + 1e-9, etac).abs() < 1e-12);
            // zero slope at eta_c
            let h = 1e-6;
            let slope = (hybrid_b(etac + h, etac) - hybrid_b(etac, etac)) / h;
            assert!(slope.abs() < 1e-4, "{etac} {slope}");
            // monotone increasing above eta_c
            let mut prev = 0.0;
            for i in 0..=100 {
                let eta = etac + (1.0 - etac) * i as f64 / 100.0;
                let b = hybrid_b(eta, etac);
                assert!(b >= prev - 1e-15);
                prev = b;
            }
        }
    }

    #[test]
    fn interfaces_reproduce_surface_and_top_pressure() {
        // p_d at eta = 1 is the surface dry pressure, at eta = 0 it is p_top.
        let znw: Vec<f32> = (0..=10).map(|i| 1.0 - i as f32 / 10.0).collect();
        let c = InterfaceCoefficients::resolve(10, None, &znw, 2, Some(0.2), 1500.0).unwrap();
        let mud = 98000.0f32;
        let bottom = c.c3f[0] * mud + c.c4f[0] + 1500.0;
        let top = c.c3f[10] * mud + c.c4f[10] + 1500.0;
        assert_eq!(bottom, mud + 1500.0);
        assert_eq!(top, 1500.0);
    }
}
