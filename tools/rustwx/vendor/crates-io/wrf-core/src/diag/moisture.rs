//! Moisture diagnostic variables:
//! pw, rh2m, dp2m, mixing_ratio, specific_humidity

use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Precipitable water (mm). `[ny, nx]`
pub fn compute_pw(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let qv = f.qvapor(t)?;
    let pres = f.full_pressure(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;

    Ok(crate::met::composite::compute_pw(&qv, &pres, nx, ny, nz))
}

/// 2-m relative humidity (%). `[ny, nx]`
pub fn compute_rh2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let psfc = f.psfc(t)?; // Pa

    Ok(t2
        .iter()
        .zip(q2.iter())
        .zip(psfc.iter())
        .map(|((t_k, q), p_pa)| {
            let t_c = t_k - 273.15;
            let p_hpa = p_pa / 100.0;
            let e = q * p_hpa / (0.622 + q);
            let es = 6.112 * (17.67 * t_c / (t_c + 243.5)).exp();
            (e / es * 100.0).clamp(0.0, 100.0)
        })
        .collect())
}

/// 2-m dewpoint (°C). `[ny, nx]`
pub fn compute_dp2m(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let q2 = f.q2_for_opts(t, opts)?;
    let psfc = f.psfc(t)?;

    Ok(q2
        .iter()
        .zip(psfc.iter())
        .map(|(&q, &p_pa)| dewpoint_2m_c(q, p_pa))
        .collect())
}

/// 2-m dewpoint (°C) from the 2-m mixing ratio (kg/kg) and surface pressure
/// (Pa).  A missing input stays missing: NaN in either gives NaN.
///
/// Breakage this prevents (2026-10-29, hex Kansas polygon f000/f001): the
/// MPAS converter writes NaN at every window point off a limited-area mesh,
/// and `f64::max(NaN, 1e-10)` returns `1e-10`, so the floor turned each
/// missing Q2 into a -150 C dewpoint and the whole frame outside the mesh
/// was painted with the -40 F end of the 2 m dewpoint scale, while T2 and
/// MSLP left it blank.  Finite inputs give the same bits as before.
pub fn dewpoint_2m_c(q: f64, p_pa: f64) -> f64 {
    if q.is_nan() || p_pa.is_nan() {
        return f64::NAN;
    }
    let q = q.max(1e-10);
    let p_hpa = p_pa / 100.0;
    let e = q * p_hpa / (0.622 + q);
    let ln_e = (e / 6.112).max(1e-10).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

/// Water vapor mixing ratio (kg/kg). `[nz, ny, nx]`
pub fn compute_mixing_ratio(f: &WrfFile, t: usize, _opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    f.qvapor(t).map(|v| v.to_vec())
}

/// Specific humidity (kg/kg). `[nz, ny, nx]`
/// q = qv / (1 + qv)
pub fn compute_specific_humidity(
    f: &WrfFile,
    t: usize,
    _opts: &ComputeOpts,
) -> WrfResult<Vec<f64>> {
    let qv = f.qvapor(t)?;
    Ok(qv.iter().map(|q| q / (1.0 + q)).collect())
}
