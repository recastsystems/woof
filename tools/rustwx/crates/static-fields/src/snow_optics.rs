//! Independently written Lorenz-Mie optics; no SNICAR code or optical tables.
//! Equations: Bohren & Huffman (1983), ch. 4; optical-depth mixing and
//! delta-Eddington are described in doi:10.5194/gmd-14-7673-2021.
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Default, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}
impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn norm(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}
impl Add for Complex {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Div for Complex {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let c = self * b.conj();
        Self::new(c.re / b.norm(), c.im / b.norm())
    }
}
fn c(v: f64) -> Complex {
    Complex::new(v, 0.)
}

/// Efficiencies Qext, Qsca and scattering asymmetry for m=n+ik.
/// Downward logarithmic-derivative recursion, upward real Riccati-Bessel
/// recursion, and the full adjacent-order/cross-polarisation g sums.
pub fn mie(x: f64, m: Complex) -> [f64; 3] {
    assert!(x > 0. && m.re > 0. && m.im >= 0.);
    let stop = (x + 4.05 * x.cbrt() + 2.).ceil() as usize;
    let z = m * c(x);
    let depth = stop.max((z.norm().sqrt()).ceil() as usize) + 32;
    let mut d = vec![Complex::default(); depth + 1];
    for n in (1..=depth).rev() {
        let q = c(n as f64) / z;
        d[n - 1] = q - c(1.) / (d[n] + q);
    }
    // chi_n=-x*y_n and xi_n=psi_n-i*chi_n, outgoing wave, m=n+ik.
    let (mut p0, mut p1) = (x.sin(), x.sin() / x - x.cos());
    let (mut q0, mut q1) = (x.cos(), x.cos() / x + x.sin());
    let mut prev_a = Complex::default();
    let mut prev_b = prev_a;
    let (mut ext, mut sca, mut asym) = (0., 0., 0.);
    for n in 1..=stop {
        let nf = n as f64;
        let a1 = d[n] / m + c(nf / x);
        let b1 = d[n] * m + c(nf / x);
        let xi1 = Complex::new(p1, -q1);
        let xi0 = Complex::new(p0, -q0);
        let a = (a1 * c(p1) - c(p0)) / (a1 * xi1 - xi0);
        let b = (b1 * c(p1) - c(p0)) / (b1 * xi1 - xi0);
        ext += (2. * nf + 1.) * (a.re + b.re);
        sca += (2. * nf + 1.) * (a.norm() + b.norm());
        asym += (2. * nf + 1.) / (nf * (nf + 1.)) * (a * b.conj()).re;
        if n > 1 {
            asym += (nf - 1.) * (nf + 1.) / nf * ((prev_a * a.conj()).re + (prev_b * b.conj()).re);
        }
        prev_a = a;
        prev_b = b;
        let p2 = (2. * nf + 1.) / x * p1 - p0;
        p0 = p1;
        p1 = p2;
        let q2 = (2. * nf + 1.) / x * q1 - q0;
        q0 = q1;
        q1 = q2;
    }
    let qe = 2. * ext / (x * x);
    let qs = 2. * sca / (x * x);
    assert!(qe.is_finite() && qs.is_finite() && qe + 1e-8 >= qs && qs > 0.);
    [qe, qs.min(qe), (4. * asym / (x * x * qs)).clamp(-1., 1.)]
}

/// Read CC0 refractiveindex.info tabulated-nk rows. Metadata is preserved in
/// the source files, but cannot silently become numeric data.
pub fn read_index(path: &std::path::Path) -> Result<Vec<[f64; 3]>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for line in s.lines() {
        let v: Vec<_> = line.split_whitespace().collect();
        if v.len() != 3 {
            continue;
        }
        if let (Ok(w), Ok(n), Ok(k)) = (
            v[0].parse::<f64>(),
            v[1].parse::<f64>(),
            v[2].parse::<f64>(),
        ) {
            // Only the solar interval is used. Querry has out-of-order IR
            // points beyond 7 um; those are outside this asset.
            if w > 5.1 {
                continue;
            }
            if !(w > 0. && n > 0. && k > 0.) || rows.last().is_some_and(|p: &[f64; 3]| p[0] >= w) {
                return Err("invalid nk rows".into());
            }
            rows.push([w, n, k]);
        }
    }
    if rows.len() < 2 {
        return Err("no tabulated nk rows".into());
    }
    Ok(rows)
}
/// n linear in log wavelength, log(k) linear in log wavelength (WB2008).
pub fn index_at(rows: &[[f64; 3]], w: f64) -> Complex {
    assert!(w >= rows[0][0] && w <= rows.last().unwrap()[0]);
    let hi = rows.partition_point(|p| p[0] < w).clamp(1, rows.len() - 1);
    let (a, b) = (rows[hi - 1], rows[hi]);
    let f = (w / a[0]).ln() / (b[0] / a[0]).ln();
    Complex::new(
        a[1] + f * (b[1] - a[1]),
        (a[2].ln() + f * (b[2] / a[2]).ln()).exp(),
    )
}

pub const NR: usize = 32;
pub const NW: usize = 480;
pub const OPTICS_START:usize=2*NR+3*NW;
pub const OPTICS_ROWS:usize=5*NR+2; // four snow habits, bubbles, BC, dust
pub const FRESNEL_START:usize=OPTICS_START+OPTICS_ROWS*NW*3;
pub const MIX_START:usize=FRESNEL_START+NW*4;
pub const AGE_START:usize=MIX_START+32*NW*3;
pub const TABLE_WORDS:usize=AGE_START+crate::snow_aging::AGING_WORDS;
pub const MAGIC: &[u8; 8] = b"SNMIE002";

/// The on-disk format is fixed, little-endian f32, with an eight-byte magic.
/// SHA256 is checked by the caller before this decoder reads any values.
pub fn read_table(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    decode_bytes(&bytes)
}

/// Strict C ABI. Output belongs to the caller. No allocation crosses FFI.
/// A digest pins the same bytes used by forecast/restart receipts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_snow_optics_load_v1(
    path: *const std::ffi::c_char,
    digest: *const std::ffi::c_char,
    out: *mut f32,
    n: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        use sha2::{Digest, Sha256};
        if path.is_null() || digest.is_null() || out.is_null() || n != TABLE_WORDS {
            return Err("snow optics ABI mismatch".to_string());
        }
        let path = unsafe { std::ffi::CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let expected = unsafe { std::ffi::CStr::from_ptr(digest) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("snow optics SHA256 mismatch".into());
        }
        // Decode the already verified bytes; do not reopen a mutable file.
        let tmp = decode_bytes(&bytes)?;
        unsafe {
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), out, n);
        }
        Ok(())
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => crate::capi::set_error(e),
        Err(_) => crate::capi::set_error("snow optics panic"),
    }
}

fn decode_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() != 8 + TABLE_WORDS * 4 || &bytes[..8] != MAGIC {
        return Err("snow optics layout mismatch".into());
    }
    let v: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite snow optics".into());
    }
    if v[..NR].windows(2).any(|p| p[0] >= p[1]) || v[0] != 30. || v[NR - 1] != 1500. {
        return Err("snow radius axis mismatch".into());
    }
    if v[NR..2*NR].windows(2).any(|p|p[0]>=p[1]) || v[NR]!=30. || v[2*NR-1]!=20000. {
        return Err("bubble radius axis mismatch".into());
    }
    for i in 0..NW {
        if (v[2*NR + i] - (0.205 + i as f32 * 0.01)).abs() > 2e-6 || (v[2*NR + NW + i] < 0. || v[2*NR + 2*NW + i] < 0.) {
            return Err("spectral axis mismatch".into());
        }
    }
    for p in v[OPTICS_START..FRESNEL_START].chunks_exact(3) {
        if p[0] <= 0. || p[1] < 0. || p[1] > p[0] || p[2].abs() > 1. {
            return Err("nonphysical snow optics".into());
        }
    }
    for p in v[FRESNEL_START..MIX_START].chunks_exact(4) {
        if p[0]<=0. || p[1]<0. || !(0.0..=1.0).contains(&p[2]) || !(0.0..=1.0).contains(&p[3]) {
            return Err("invalid ice interface optics".into());
        }
    }
    if v[MIX_START..AGE_START].iter().any(|e|*e<1.) {return Err("invalid internal-mixing enhancement".into());}
    if v[AGE_START..].iter().any(|e|*e<0.) {return Err("invalid generated aging coefficients".into());}
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rayleigh_limit() {
        let m = Complex::new(1.31, 0.);
        let x = 0.001;
        let v = mie(x, m);
        let p = (m * m - c(1.)) / (m * m + c(2.));
        let want = 8. / 3. * x.powi(4) * p.norm();
        assert!((v[1] / want - 1.).abs() < 1e-5);
        assert!((v[0] - v[1]).abs() < 1e-18);
        assert!(v[2].abs() < 1e-6);
    }
    #[test]
    fn published_mie_benchmark() {
        // Standard nonabsorbing x=2, m=1.5 benchmark.
        let v = mie(2., Complex::new(1.5, 0.));
        assert!((v[0] - 1.7984181632262635).abs() < 1e-12);
        assert!((v[2] - 0.6259531461237373).abs() < 1e-12);
    }
    #[test]
    fn absorptive_and_large_spheres() {
        for x in [0.01, 1., 100., 10000.] {
            let v = mie(x, Complex::new(1.31, 0.01));
            assert!(v[0] > v[1] && v[2] >= 0. && v[2] < 1.);
        }
    }
    #[test]
    fn bad_binary_rejected() {
        assert!(decode_bytes(b"SNMIE002").is_err());
    }
}
