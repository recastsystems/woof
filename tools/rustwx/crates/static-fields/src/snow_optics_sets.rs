//! Ice refractive-index sets and dust optical sets for the paper-written
//! snow optics. Re-implemented from published descriptions, not bitwise to
//! upstream. No SNICAR program, optical library or upstream table is read.
//!
//! Ice sets (Flanner et al. 2021, doi:10.5194/gmd-14-7673-2021, section 3.1,
//! describes the three sets; the numbers come from the original data):
//!   1  Warren (1984), doi:10.1364/AO.23.001206, CC0 database copy.
//!   2  Warren & Brandt (2008), doi:10.1029/2007JD009744 (existing default).
//!   3  Warren & Brandt (2008) real part with the Picard et al. (2016)
//!      absorption, doi:10.5194/tc-10-2655-2016 Supplement section 2 table,
//!      CC BY 3.0, over 350-600 nm, its 350 nm value held constant below.
//! Dust sets (Flanner et al. 2021 section 3.3.3 describes the three named
//! compositions; mineral indices are the CC-BY-4.0 Zhang et al. 2024 group
//! indices, doi:10.5281/zenodo.14171144, not the Scanza 2015 compilation):
//!   0  Zhang et al. (2024) coarse-mode mean index (existing default).
//!   1  Saharan: Balkanski et al. (2007) doi:10.5194/acp-7-81-2007 Table 2,
//!      central hematite S1 volume fractions, Bruggeman mixed.
//!   2  Colorado: Skiles et al. (2017) doi:10.1017/jog.2016.126 Table 1.
//!   3  Greenland: Polashenski et al. (2015) doi:10.1002/2015GL065912 central
//!      scenario volume fractions as reported by Flanner et al. 2021, Bruggeman.
use crate::snow_optics::{Complex, NW};

pub fn csqrt(z: Complex) -> Complex {
    let r = (z.re * z.re + z.im * z.im).sqrt();
    let re = (0.5 * (r + z.re)).max(0.).sqrt();
    let im = (0.5 * (r - z.re)).max(0.).sqrt();
    Complex::new(re, if z.im < 0. { -im } else { im })
}

/// Comma-separated wavelength_um,n,k rows; '#' lines are attribution.
pub fn read_csv_nk(path: &std::path::Path) -> Result<Vec<[f64; 3]>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = line
            .split(',')
            .take(3)
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        if v.len() != 3 || !(v[0] > 0. && v[1] > 0. && v[2] >= 0.) {
            return Err(format!("{}: invalid nk row {line}", path.display()));
        }
        if rows.last().is_some_and(|p| p[0] >= v[0]) {
            return Err(format!("{}: wavelengths not increasing", path.display()));
        }
        rows.push([v[0], v[1], v[2]]);
    }
    if rows.len() < 2 {
        return Err(format!("{}: no rows", path.display()));
    }
    Ok(rows)
}

/// Linear n and k in wavelength; held constant outside the tabulated range
/// (a declared extension, used for Skiles below 0.35 and above 1.5 um).
pub fn linear_nk(rows: &[[f64; 3]], w: f64) -> Complex {
    if w <= rows[0][0] {
        return Complex::new(rows[0][1], rows[0][2]);
    }
    let last = rows[rows.len() - 1];
    if w >= last[0] {
        return Complex::new(last[1], last[2]);
    }
    let hi = rows.partition_point(|p| p[0] < w);
    let (a, b) = (rows[hi - 1], rows[hi]);
    let f = (w - a[0]) / (b[0] - a[0]);
    Complex::new(a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]))
}

/// Picard table: wavelength_nm,absorption_m-1[,uncertainty].
pub fn read_picard(path: &std::path::Path) -> Result<Vec<[f64; 2]>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut out: Vec<[f64; 2]> = Vec::new();
    for line in s.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let v: Vec<f64> = line
            .split(',')
            .take(2)
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        if v.len() != 2 || !(v[1] > 0.) || out.last().is_some_and(|p| p[0] >= v[0]) {
            return Err(format!("invalid Picard row {line}"));
        }
        out.push([v[0], v[1]]);
    }
    if out.len() != 51 || out[0][0] != 350. || out[50][0] != 600. {
        return Err("Picard 2016 table must be the 51 rows 350-600 nm".into());
    }
    Ok(out)
}

/// k = alpha * lambda / (4 pi), lambda in metres: the absorption
/// coefficient definition Picard et al. 2016 use (their section 2).
pub fn picard_k(alpha_per_m: f64, nm: f64) -> f64 {
    alpha_per_m * nm * 1e-9 / (4. * std::f64::consts::PI)
}

/// Merged set 3: WB2008 rows outside 350-600 nm keep their n and k, except
/// that below 350 nm k is Picard's 350 nm value (the constancy rule Flanner
/// et al. 2021 describe below the first Picard wavelength). Inside the
/// interval the 51 Picard wavelengths carry WB2008's interpolated n.
pub fn picard_merge(wb: &[[f64; 3]], alpha: &[[f64; 2]]) -> Vec<[f64; 3]> {
    let k350 = picard_k(alpha[0][1], alpha[0][0]);
    let mut out = Vec::with_capacity(wb.len() + alpha.len());
    for r in wb.iter().filter(|r| r[0] < 0.35) {
        out.push([r[0], r[1], k350]);
    }
    for a in alpha {
        let w = a[0] * 1e-3;
        out.push([w, crate::snow_optics::index_at(wb, w).re, picard_k(a[1], a[0])]);
    }
    for r in wb.iter().filter(|r| r[0] > 0.6) {
        out.push(*r);
    }
    out
}

/// Bruggeman effective medium: sum_i V_i (e_i - e)/(e_i + 2e) = 0
/// (Flanner et al. 2021 eq. 20). Newton iteration from the volume mean.
pub fn bruggeman(eps: &[Complex], vol: &[f64]) -> Complex {
    assert_eq!(eps.len(), vol.len());
    let total: f64 = vol.iter().sum();
    let mut e = Complex::new(0., 0.);
    for (x, v) in eps.iter().zip(vol) {
        e = e + *x * Complex::new(v / total, 0.);
    }
    for _ in 0..100 {
        let (mut f, mut d) = (Complex::new(0., 0.), Complex::new(0., 0.));
        for (x, v) in eps.iter().zip(vol) {
            let w = Complex::new(v / total, 0.);
            let den = *x + e * Complex::new(2., 0.);
            f = f + w * (*x - e) / den;
            d = d + w * (Complex::new(-3., 0.) * *x) / (den * den);
        }
        let step = f / d;
        e = e - step;
        let size = (e.re * e.re + e.im * e.im).sqrt();
        if (step.re * step.re + step.im * step.im).sqrt() < 1e-15 * size {
            break;
        }
    }
    assert!(e.re.is_finite() && e.im.is_finite() && e.im >= -1e-15);
    e
}

/// Mineral order: illite, kaolinite, montmorillonite, quartz, calcite, hematite.
pub const MINERALS: [&str; 6] = ["illite", "kaolinite", "montmorillonite", "quartz", "calcite", "hematite"];
/// Balkanski et al. 2007 Table 2, S1 "central hematite", % volume.
pub const SAHARAN_VOLUME: [f64; 6] = [31.5, 24.0, 23.0, 14.0, 6.0, 1.5];
/// Polashenski et al. 2015 central scenario, % volume (Flanner 2021 s3.3.3).
pub const GREENLAND_VOLUME: [f64; 6] = [31.0, 15.2, 7.3, 8.3, 35.4, 2.8];
/// Particle densities, kg m-3: set 0 as before; sets 1-3 as Flanner et al.
/// 2021 section 3.3.3 state them.
pub const DUST_DENSITY: [f64; 4] = [2500., 2645., 2600., 2747.];

pub fn mixed_index(minerals: &[Vec<[f64; 3]>], volume: &[f64; 6], w: f64) -> Complex {
    let eps: Vec<Complex> = minerals
        .iter()
        .map(|m| {
            let n = linear_nk(m, w);
            n * n
        })
        .collect();
    csqrt(bruggeman(&eps, volume))
}

pub const DUST_SETS: usize = 4;
pub const DUST_WORDS: usize = DUST_SETS * NW * 3;
pub const DUST_MAGIC: &[u8; 8] = b"SNDUST01";

pub fn decode_dust(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() != 8 + DUST_WORDS * 4 || &bytes[..8] != DUST_MAGIC {
        return Err("snow dust rows layout mismatch".into());
    }
    let v: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for p in v.chunks_exact(3) {
        if !(p.iter().all(|x| x.is_finite()) && p[0] > 0. && p[1] >= 0. && p[1] <= p[0] && p[2].abs() <= 1.) {
            return Err("nonphysical snow dust optics".into());
        }
    }
    Ok(v)
}

/// Strict C ABI, the same contract as gpuwm_snow_optics_load_v1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_snow_dust_rows_load_v1(
    path: *const std::ffi::c_char,
    digest: *const std::ffi::c_char,
    out: *mut f32,
    n: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        use sha2::{Digest, Sha256};
        if path.is_null() || digest.is_null() || out.is_null() || n != DUST_WORDS {
            return Err("snow dust rows ABI mismatch".to_string());
        }
        let path = unsafe { std::ffi::CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let expected = unsafe { std::ffi::CStr::from_ptr(digest) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("snow dust rows SHA256 mismatch".into());
        }
        let v = decode_dust(&bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), out, n);
        }
        Ok(())
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => crate::capi::set_error(e),
        Err(_) => crate::capi::set_error("snow dust rows panic"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bruggeman_hand_case() {
        // e1=2.25, e2=1, equal volume: 4e^2 - 3.25e - 4.5 = 0 by hand,
        // e = (3.25 + sqrt(3.25^2 + 72)) / 8 = 1.54205...
        let e = bruggeman(&[Complex::new(2.25, 0.), Complex::new(1., 0.)], &[0.5, 0.5]);
        let want = (3.25 + (3.25f64 * 3.25 + 72.).sqrt()) / 8.;
        assert!((e.re - want).abs() < 1e-12 && e.im.abs() < 1e-15, "{e:?}");
        // A single component returns itself; a dilute absorber stays absorbing.
        let one = bruggeman(&[Complex::new(2.4, 0.01)], &[1.]);
        assert!((one.re - 2.4).abs() < 1e-14 && (one.im - 0.01).abs() < 1e-14);
        let mix = bruggeman(&[Complex::new(2.4, 0.), Complex::new(7., 4.)], &[0.985, 0.015]);
        assert!(mix.im > 0. && mix.re > 2.4);
    }
    #[test]
    fn picard_conversion_hand_case() {
        // Picard 2016 Supplement table: 440 nm, 0.01820 m-1.
        // k = 0.0182 * 440e-9 / (4 pi) = 6.37256e-10 (8.008e-9 / 12.5664).
        let k = picard_k(0.01820, 440.);
        assert!((k - 6.372564e-10).abs() < 1e-15, "{k}");
    }
    #[test]
    fn complex_sqrt() {
        let z = csqrt(Complex::new(-4., 0.));
        assert!(z.re.abs() < 1e-15 && (z.im - 2.).abs() < 1e-15);
        let m = Complex::new(1.5, 0.01);
        let r = csqrt(m * m);
        assert!((r.re - 1.5).abs() < 1e-14 && (r.im - 0.01).abs() < 1e-14);
    }
}
