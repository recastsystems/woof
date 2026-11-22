//! Light-absorbing-particle optics for snow (clean-snicar-lap).
//! Re-implemented from published descriptions, not bitwise to upstream
//! Noah-MP; no SNICAR program or optical table is an input.
//!
//! Rows (480 bins, 0.205-4.995 um; ext m2/kg, abs m2/kg, g):
//!   0  organic carbon. RI from Brown et al. 2018 ACP 18:17745 eqs 1-3
//!      (the Saleh et al. 2014 brown-carbon fit): n = 1.7,
//!      k = k550 (0.55/lambda)^w, k550 = 0.016 log10(R) + 0.04,
//!      w = 0.21/(R + 0.7) as printed there, R = 0.08 (their global mean
//!      biomass-burning BC-to-OA). Lognormal, effective radius 0.1 um,
//!      sigma_g 1.5; density 1000 kg/m3 (Brown 2018 sect. 2.1).
//!   1-5 dust in diameter bins 0.1-1, 1-2.5, 2.5-5, 5-10, 10-100 um (the
//!      bins Lin et al. 2025 JHM name), Zhang et al. 2024 mineral RI
//!      (CC-BY-4.0, already packaged for the BC/dust row), density
//!      2500 kg/m3, mass log-uniform in diameter inside each bin (24
//!      equal-mass sub-bins), self-written Mie.
//! Usage: snow-lap-optics DUST.dat OUT.bin   (all cores)
use static_fields::snow_optics::*;
use std::{io::Write, path::PathBuf};

fn dust_index(path: &std::path::Path) -> Vec<[f64; 3]> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| {
        let v: Vec<f64> = l.split(',').take(3).map(|v| v.parse().unwrap()).collect();
        [v[0], v[1], v[2]]
    }).collect()
}
/// Lognormal size distribution (sigma_g 1.5), the form snow-optics uses.
fn psd(re: f64, w: f64, m: Complex, rho: f64) -> [f64; 3] {
    let mut s = [0.; 4];
    let l = 1.5_f64.ln();
    for j in 0..24 {
        let z = -4. + (j as f64 + 0.5) / 3.;
        let r = re * (-2.5 * l * l + z * l).exp();
        let n = (-0.5 * z * z).exp();
        let q = mie(2. * std::f64::consts::PI * r / w, m);
        let a = n * r * r;
        s[0] += a * q[0]; s[1] += a * (q[0] - q[1]); s[2] += a * q[1] * q[2]; s[3] += n * r * r * r;
    }
    let factor = 3e6 / (4. * rho * s[3]);
    [s[0] * factor, s[1] * factor, s[2] / (s[0] - s[1])]
}
/// Bin-mean specific optics for mass log-uniform in diameter.
fn bin(dlo: f64, dhi: f64, w: f64, m: Complex, rho: f64) -> [f64; 3] {
    let (mut e, mut a, mut gs, mut sc) = (0., 0., 0., 0.);
    let k = 24;
    for i in 0..k {
        let d = dlo * (dhi / dlo).powf((i as f64 + 0.5) / k as f64);
        let r = 0.5 * d;
        let q = mie(2. * std::f64::consts::PI * r / w, m);
        let f = 3e6 / (4. * rho * r) / k as f64;
        e += q[0] * f; a += (q[0] - q[1]) * f; sc += q[1] * f; gs += q[1] * q[2] * f;
    }
    [e, a, gs / sc]
}
pub fn oc_index(w: f64) -> Complex {
    let r: f64 = 0.08;
    let k550 = 0.016 * r.log10() + 0.04;
    let ww = 0.21 / (r + 0.7);
    Complex::new(1.7, k550 * (0.55 / w).powf(ww))
}
pub const DUST_EDGES: [f64; 6] = [0.1, 1., 2.5, 5., 10., 100.];
fn main() {
    let a: Vec<_> = std::env::args_os().collect();
    assert_eq!(a.len(), 3, "snow-lap-optics DUST.dat OUT.bin");
    let dust = dust_index(&PathBuf::from(&a[1]));
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rows: Vec<(usize, Vec<[f64; 3]>)> = std::thread::scope(|s| {
        let jobs: Vec<_> = (0..workers).map(|worker| {
            let dust = &dust;
            s.spawn(move || {
                let mut out = Vec::new();
                for iw in (worker..NW).step_by(workers) {
                    let w = 0.205 + iw as f64 * 0.01;
                    let mut p = vec![psd(0.1, w, oc_index(w), 1000.)];
                    let m = index_at(dust, w);
                    for b in 0..5 { p.push(bin(DUST_EDGES[b], DUST_EDGES[b + 1], w, m, 2500.)); }
                    out.push((iw, p));
                }
                out
            })
        }).collect();
        jobs.into_iter().flat_map(|j| j.join().unwrap()).collect()
    });
    let mut words = vec![0_f32; LAP_TABLE_WORDS];
    for (iw, p) in rows { for (r, v) in p.iter().enumerate() { for k in 0..3 { words[(r * NW + iw) * 3 + k] = v[k] as f32; } } }
    let mut f = std::fs::File::create(&a[2]).unwrap();
    f.write_all(LAP_MAGIC).unwrap();
    for v in &words { f.write_all(&v.to_le_bytes()).unwrap(); }
    drop(f);
    let bytes = std::fs::read(&a[2]).unwrap();
    decode_lap_bytes(&bytes).unwrap();
    println!("generated {} words: OC + five dust bins, {} workers", LAP_TABLE_WORDS, workers);
}
