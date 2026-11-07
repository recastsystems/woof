//! TEMPO microphysics initialization for WOOF's GPU TEMPO port.
//!
//! Port of `tempo_init` from TEMPO (https://github.com/NCAR/TEMPO, commit
//! 9adf9ef2827104640125f0c322e1218fce74e94c, tag tempo_v3.0.3, the RRFS v2
//! pin): module_mp_tempo_params.F90 (parameters, initialize_graupel_vars,
//! initialize_parameters, initialize_bins_*, create_bins),
//! module_mp_tempo_utils.F90 (compute_efrw, compute_efsw, compute_drop_evap,
//! qi_aut_qs, calc_gamma_p/series/cf, get_nuc) and the table readers of
//! module_mp_tempo_driver.F90.
//!
//! Copyright 2026, NOAA, UCAR/NCAR CU/CIRES.  Licensed under the Apache
//! License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0);
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//! KIND.  Modifications: translated from Fortran to Rust; the module state
//! becomes one named float64 blob (gpuwm/core/kernels/tempo/tempo_common.cuh
//! TEMPO_FIELDS order, integers stored as float64).
//!
//! Arithmetic follows the reference build (gfortran -fdefault-real-8):
//! every value is f64; Fortran `**` with a non-constant integer exponent is
//! libgcc's `__powidf2` ([`powi`]); `x**2.0` with a constant exponent is
//! `x*x` (GCC folds it); other real powers, exp, log, log10, atan and
//! gamma (C `tgamma`) call the platform C library as gfortran does.
//! Expressions gfortran folds at compile time from named constants are
//! evaluated here so they round the same way (see [`cr_cube`]).

#![no_builtins]

pub mod capi;
mod fields;
mod wrf_tables;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub use fields::FIELDS;

// The platform C library's elementary functions, called by name as
// gfortran calls them.  `no_builtins` keeps LLVM from recognising these
// calls and rewriting them (pow(x, 0.5) -> sqrt, constant folding), which
// gfortran -O0 does not do.
unsafe extern "C" {
    fn tgamma(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn log10(x: f64) -> f64;
    fn atan(x: f64) -> f64;
}

/// C library calls on f64, spelled as methods to keep the transliteration
/// readable.
trait Libm {
    fn cpow(self, y: f64) -> f64;
    fn cexp(self) -> f64;
    fn cln(self) -> f64;
    fn clog10(self) -> f64;
    fn catan(self) -> f64;
}

impl Libm for f64 {
    // SAFETY (all five): pure C library functions on plain doubles.
    fn cpow(self, y: f64) -> f64 {
        unsafe { pow(self, y) }
    }
    fn cexp(self) -> f64 {
        unsafe { exp(self) }
    }
    fn cln(self) -> f64 {
        unsafe { log(self) }
    }
    fn clog10(self) -> f64 {
        unsafe { log10(self) }
    }
    fn catan(self) -> f64 {
        unsafe { atan(self) }
    }
}

/// Fortran GAMMA: gfortran calls the C library's tgamma.
fn gamma(x: f64) -> f64 {
    // SAFETY: tgamma is a pure C library function on a plain double.
    unsafe { tgamma(x) }
}

/// libgcc `__powidf2`: real ** integer as gfortran evaluates it.
pub fn powi(mut x: f64, m: i32) -> f64 {
    let mut n: u32 = if m < 0 { (m as i64).unsigned_abs() as u32 } else { m as u32 };
    let mut y = if n % 2 == 1 { x } else { 1.0 };
    loop {
        n >>= 1;
        if n == 0 {
            break;
        }
        x *= x;
        if n % 2 == 1 {
            y *= x;
        }
    }
    if m < 0 { 1.0 / y } else { y }
}

/// Fortran NINT: round half away from zero.
fn nint(x: f64) -> i64 {
    x.round() as i64
}

/// Correctly rounded x**3, which is what gfortran's compile-time folding
/// (MPFR) gives for `const**3` and `const**3.0` with named constants.
fn cr_cube(x: f64) -> f64 {
    let x2 = x * x;
    let e2 = x.mul_add(x, -x2);
    let x3 = x2 * x;
    let e3 = x2.mul_add(x, -x3) + e2 * x;
    x3 + e3
}

/// Fortran literal `<m>.e<e>_wp`: the correctly rounded decimal value.
fn lit(m: u32, e: i32) -> f64 {
    format!("{m}e{e}").parse::<f64>().expect("decimal literal")
}

/// The decade tables of module_mp_tempo_params: 1..9 times each decade from
/// 10**e0 to 10**(e1-1), then 10**e1.
fn decades(e0: i32, e1: i32) -> Vec<f64> {
    let mut v = Vec::new();
    for e in e0..e1 {
        for m in 1..=9 {
            v.push(lit(m, e));
        }
    }
    v.push(lit(1, e1));
    v
}

/// The named float64 blob, laid out as [`FIELDS`].
pub struct Blob {
    pub data: Vec<f64>,
    index: HashMap<&'static str, (usize, usize)>,
}

impl Blob {
    pub fn new() -> Self {
        let mut index = HashMap::new();
        let mut off = 0usize;
        for (name, n) in FIELDS {
            index.insert(*name, (off, *n));
            off += n;
        }
        Blob { data: vec![0.0; off], index }
    }

    fn slot(&self, name: &str) -> (usize, usize) {
        *self.index.get(name).unwrap_or_else(|| panic!("TEMPO blob has no field {name}"))
    }

    pub fn set(&mut self, name: &str, values: &[f64]) {
        let (o, n) = self.slot(name);
        assert_eq!(n, values.len(), "TEMPO blob field {name}: {} values, layout says {n}", values.len());
        self.data[o..o + n].copy_from_slice(values);
    }

    pub fn set1(&mut self, name: &str, value: f64) {
        self.set(name, &[value]);
    }

    pub fn get(&self, name: &str) -> &[f64] {
        let (o, n) = self.slot(name);
        &self.data[o..o + n]
    }

    /// The named-record file tools/tempo_ref/tempo_ref_driver.F90 writes
    /// (48-byte space-padded name, little-endian i64 count, f64 values), so
    /// gpuwm.core.tempo_runtime.load_blob reads either source unchanged.
    pub fn to_records(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 8 + FIELDS.len() * 56);
        for (name, n) in FIELDS {
            let mut nm = [b' '; 48];
            nm[..name.len()].copy_from_slice(name.as_bytes());
            out.extend_from_slice(&nm);
            out.extend_from_slice(&(*n as i64).to_le_bytes());
            for v in self.get(name) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

impl Default for Blob {
    fn default() -> Self {
        Self::new()
    }
}

/// ty_tempo_cfgs flags that change what tempo_init builds.
#[derive(Clone, Copy, Debug)]
pub struct InitOptions {
    pub hailaware: bool,
    pub refl10cm_from_melting: bool,
    pub max_hail_diameter: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        InitOptions { hailaware: true, refl10cm_from_melting: false, max_hail_diameter: true }
    }
}

// ---------------------------------------------------------------------------
// parameters (module_mp_tempo_params.F90 lines 27-288)
// ---------------------------------------------------------------------------
const AV_R: f64 = 4854.0;
const BV_R: f64 = 1.0;
const AV_S: f64 = 40.0;
const BV_S: f64 = 0.55;
const FV_S: f64 = 100.0;
const BV_C: f64 = 2.0;
const BV_I: f64 = 1.0;
const AV_G_OLD: f64 = 442.0;
const BV_G_OLD: f64 = 0.89;
const AV_G_NEW: f64 = 161.794724;
const BV_G_NEW: f64 = 0.640961647;
const FV_R: f64 = 195.0;
const AV_C: f64 = 0.316946e8;
const A_COEFF: f64 = 0.47244157;
const B_COEFF: f64 = 0.54698726;
const AV_I: f64 = 1493.9;
const AM_S: f64 = 0.069;
const BM_S: f64 = 2.0;
const BM_G: f64 = 3.0;
const BM_I: f64 = 3.0;
const BM_R: f64 = 3.0;
const RHO_I: f64 = 890.0;
const XM0I: f64 = 1.0e-12;
const D0C: f64 = 1.0e-6;
const D0R: f64 = 50.0e-6;
const D0S: f64 = 300.0e-6;
const D0G: f64 = 350.0e-6;
const D0R_MAX: f64 = 2.5e-3;
const C_CUBE: f64 = 0.5;
const C_SQRD: f64 = 0.15;
const MU_R: f64 = 0.0;
const MU_S: f64 = 0.6357;
const MU_G: f64 = 0.0;
const MU_I: f64 = 0.0;
const NU_C_SCALE: f64 = 1000.0e6;
const NAIN1: f64 = 0.5e6;
const RHO_W: f64 = 1000.0;
const SC: f64 = 0.632;
const T0: f64 = 273.15;
const NBINS: usize = 100;
const NBC: usize = NBINS;
const NBR: usize = NBINS;
const NBS: usize = NBINS;
const NBI: usize = NBINS;
const NBG: usize = NBINS;
const NTB_I: usize = 64;
const NTB_I1: usize = 55;
const NTB_C: usize = 37;
const NRHG: usize = 9;
const IDX_BG1: usize = 6;
const NHBINS: usize = 50;
const RADAR_BINS: usize = 50;

// host-model variables (no parameter attribute), TEMPO defaults
const PI: f64 = 3.1415926536;
const LSUB: f64 = 2.834e6;
const LVAP0: f64 = 2.5e6;
const RV: f64 = 461.5;
const RDRY: f64 = 287.04;

fn set_parameters(b: &mut Blob) {
    let scalars: &[(&str, f64)] = &[
        ("av_r", AV_R), ("bv_r", BV_R), ("av_s", AV_S), ("bv_s", BV_S), ("fv_s", FV_S),
        ("bv_c", BV_C), ("bv_i", BV_I), ("av_g_old", AV_G_OLD), ("bv_g_old", BV_G_OLD),
        ("av_g_new", AV_G_NEW), ("bv_g_new", BV_G_NEW), ("fv_r", FV_R), ("av_c", AV_C),
        ("a_coeff", A_COEFF), ("b_coeff", B_COEFF), ("av_i", AV_I), ("am_s", AM_S),
        ("bm_s", BM_S), ("bm_g", BM_G), ("bm_i", BM_I), ("bm_r", BM_R), ("rho_i", RHO_I),
        ("xm0i", XM0I), ("d0c", D0C), ("d0r", D0R), ("d0s", D0S), ("d0g", D0G),
        ("d0r_max", D0R_MAX), ("c_cube", C_CUBE), ("c_sqrd", C_SQRD), ("mu_r", MU_R),
        ("mu_s", MU_S), ("mu_g", MU_G), ("mu_i", MU_I), ("nu_c_scale", NU_C_SCALE),
        ("nu_c_max", 15.0), ("nu_c_min", 2.0), ("naccn0", 300.0e6), ("naccn1", 50.0e6),
        ("nain0", 1.5e6), ("nain1", NAIN1), ("nwfa_default", 11.1e6),
        ("nifa_default", NAIN1 * 0.01), ("aero_max", 9999.0e6), ("hgfrz", 235.16),
        ("nt_c_o", 50.0e6), ("nt_c_l", 100.0e6), ("nt_c_max", 1999.0e6), ("nt_c_min", 2.0),
        ("tno", 5.0), ("ato", 0.304), ("rho_s", 100.0), ("demott_nuc_ssati", 0.25),
        ("max_ni", 4999.0e3), ("icenuc_max", 1000.0e3), ("rime_threshold", 2.0),
        ("rime_conversion", 0.95), ("ef_si", 0.05), ("ef_rs", 0.95), ("ef_rg", 0.75),
        ("ef_ri", 0.95), ("autocon_nr_factor", 10.0),
        ("timestep_conversion_rime_to_rain", 120.0), ("nrhg", NRHG as f64), ("nrhg1", 1.0),
        ("min_qv", 1.0e-10), ("r1", 1.0e-12), ("r2", 1.0e-6),
        ("low_limit_mass_for_precip", 1.0e-9), ("eps", 1.0e-15), ("meters3_to_liters", 1000.0),
        ("gonv_min", 1.0e2), ("gonv_max", 1.0e6), ("t0", T0), ("rho_w", RHO_W), ("sc", SC),
        ("pi", PI), ("lsub", LSUB), ("lvap0", LVAP0), ("rv", RV), ("rdry", RDRY),
        ("roverrv", 0.622), ("r", 287.04), ("cp", 1004.0), ("r_uni", 8.314),
        ("kap0", 490.6), ("kap1", 17.46), ("lam0", 20.78), ("lam1", 3.29),
        ("nbins", NBINS as f64), ("nbc", NBC as f64), ("nbr", NBR as f64), ("nbs", NBS as f64),
        ("nbi", NBI as f64), ("nbg", NBG as f64), ("ntb_i", NTB_I as f64),
        ("ntb_i1", NTB_I1 as f64), ("ntb_c", NTB_C as f64), ("ntb_t", 9.0), ("ntb_g1", 37.0),
        ("ntb_s", 37.0), ("ntb_g", 37.0), ("ntb_r", 37.0), ("ntb_r1", 37.0), ("ntb_t1", 45.0),
        ("ntb_in", 55.0), ("ntb_arc", 7.0), ("ntb_arw", 9.0), ("ntb_art", 7.0), ("ntb_arr", 5.0),
        ("ntb_ark", 4.0), ("idx_bg1", IDX_BG1 as f64), ("nhbins", NHBINS as f64),
        ("radar_bins", RADAR_BINS as f64),
    ];
    for (name, v) in scalars {
        b.set1(name, *v);
    }
    b.set("rho_g", &[50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]);
    b.set("r_c", &decades(-6, -2));
    b.set("r_i", &decades(-10, -3));
    b.set("r_r", &decades(-6, -2));
    b.set("r_s", &decades(-6, -2));
    b.set("r_g", &decades(-6, -2));
    b.set("n0r_exp", &decades(6, 10));
    b.set("n0g_exp", &decades(2, 6));
    b.set("nt_i", &decades(0, 6));
    b.set("nt_in", &decades(0, 6));
    b.set("ta_na", &[10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0]);
    b.set("ta_ww", &[0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0]);
    b.set("ta_tk", &[243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15]);
    b.set("ta_ra", &[0.01, 0.02, 0.04, 0.08, 0.16]);
    b.set("ta_ka", &[0.2, 0.4, 0.6, 0.8]);
    b.set("sa", &[5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204,
                  0.003199, 0.0, -0.015952]);
    b.set("sb", &[0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079,
                  0.000594, 0.0, -0.003577]);
    b.set("tc", &[-0.01, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0]);
}

/// initialize_graupel_vars (params.F90 410-425); returns (av_g, bv_g, dim_nrhg).
fn graupel_vars(hail: bool) -> ([f64; NRHG], [f64; NRHG], usize) {
    let mut av_g = [45.9173813, 67.0867386, 98.0158463, 122.353378, 143.204224, 161.794724,
                    178.762115, 194.488785, 209.225876];
    let mut bv_g = [0.640961647; NRHG];
    if hail {
        av_g[IDX_BG1 - 1] = AV_G_NEW;
        bv_g[IDX_BG1 - 1] = BV_G_NEW;
        (av_g, bv_g, NRHG)
    } else {
        av_g[IDX_BG1 - 1] = AV_G_OLD;
        bv_g[IDX_BG1 - 1] = BV_G_OLD;
        (av_g, bv_g, 1)
    }
}

/// Derived constants of initialize_parameters (params.F90 428-608).
struct Derived {
    am_i: f64,
    am_r: f64,
    am_g: [f64; NRHG],
    lfus: f64,
    olfus: f64,
    obmr: f64,
    obmi: f64,
    oig1: f64,
    d0i: f64,
    cce: [[f64; 15]; 5],  // cce[row-1][n-1]
    ccg: [[f64; 15]; 5],
    ocg1: [f64; 15],
    cie: [f64; 7],
    cig: [f64; 7],
}

fn initialize_parameters(b: &mut Blob, bv_g: &[f64; NRHG]) -> Derived {
    let am_i = PI * RHO_I / 6.0;
    let am_r = PI * RHO_W / 6.0;
    let mut am_g = [0.0; NRHG];
    let rho_g = b.get("rho_g").to_vec();
    for m in 0..NRHG {
        am_g[m] = PI * rho_g[m] / 6.0;
    }
    // 4/3 and (2.5e-6)**3 are folded at compile time (named constants only)
    let ar_volume = (4.0 / 3.0) * PI * cr_cube(2.5e-6);
    let lfus = LSUB - LVAP0;
    let olfus = 1.0 / lfus;
    let orv = 1.0 / RV;
    let rho_not = 101325.0 / (RDRY * 298.0);
    let rho_not0 = 101325.0 / (RDRY * T0);
    // sc**(1/3): folded by gfortran (MPFR, correctly rounded)
    let sc3 = SC.cpow(1.0 / 3.0);
    let d0i = (XM0I / am_i).cpow(1.0 / BM_I);
    // am_s*d0s**bm_s folded: d0s**2.0 correctly rounded is d0s*d0s
    let xm0s = AM_S * (D0S * D0S);
    // d0g**bm_g folded (correctly rounded cube), times the runtime am_g(nrhg)
    let xm0g = am_g[NRHG - 1] * cr_cube(D0G);
    let obmi = 1.0 / BM_I;
    let obmr = 1.0 / BM_R;
    let oams = 1.0 / AM_S;
    let obms = 1.0 / BM_S;
    let ocms = oams.cpow(obms);
    let obmg = 1.0 / BM_G;
    let mut oamg = [0.0; NRHG];
    let mut ocmg = [0.0; NRHG];
    for m in 0..NRHG {
        oamg[m] = 1.0 / am_g[m];
        ocmg[m] = oamg[m].cpow(obmg);
    }

    let mut cce = [[0.0; 15]; 5];
    let mut ccg = [[0.0; 15]; 5];
    let mut ocg1 = [0.0; 15];
    let mut ocg2 = [0.0; 15];
    for n in 1..=15usize {
        let nf = n as f64;
        cce[0][n - 1] = nf + 1.0;
        cce[1][n - 1] = BM_R + nf + 1.0;
        cce[2][n - 1] = BM_R + nf + 4.0;
        cce[3][n - 1] = nf + BV_C + 1.0;
        cce[4][n - 1] = BM_R + nf + BV_C + 1.0;
        for r in 0..5 {
            ccg[r][n - 1] = gamma(cce[r][n - 1]);
        }
        ocg1[n - 1] = 1.0 / ccg[0][n - 1];
        ocg2[n - 1] = 1.0 / ccg[1][n - 1];
    }

    let cie = [
        MU_I + 1.0,
        BM_I + MU_I + 1.0,
        BM_I + MU_I + BV_I + 1.0,
        MU_I + BV_I + 1.0,
        MU_I + 2.0,
        BM_I * 0.5 + MU_I + BV_I + 1.0,
        BM_I * 0.5 + MU_I + 1.0,
    ];
    let mut cig = [0.0; 7];
    for i in 0..7 {
        cig[i] = gamma(cie[i]);
    }
    let oig1 = 1.0 / cig[0];
    let oig2 = 1.0 / cig[1];

    let cre = [
        BM_R + 1.0,
        MU_R + 1.0,
        BM_R + MU_R + 1.0,
        BM_R * 2.0 + MU_R + 1.0,
        MU_R + BV_R + 1.0,
        BM_R + MU_R + BV_R + 1.0,
        BM_R * 0.5 + MU_R + BV_R + 1.0,
        BM_R + MU_R + BV_R + 3.0,
        MU_R + BV_R + 3.0,
        MU_R + 2.0,
        0.5 * (BV_R + 5.0 + 2.0 * MU_R),
        BM_R * 0.5 + MU_R + 1.0,
        BM_R * 2.0 + MU_R + BV_R + 1.0,
    ];
    let mut crg = [0.0; 13];
    for n in 0..13 {
        crg[n] = gamma(cre[n]);
    }
    let ore1 = 1.0 / cre[0];
    let org1 = 1.0 / crg[0];
    let org2 = 1.0 / crg[1];
    let org3 = 1.0 / crg[2];

    let cse = [
        BM_S + 1.0,
        BM_S + 2.0,
        BM_S * 2.0,
        BM_S + BV_S + 1.0,
        BM_S * 2.0 + BV_S + 1.0,
        BM_S * 2.0 + 1.0,
        BM_S + MU_S + 1.0,
        BM_S + MU_S + 2.0,
        BM_S + MU_S + 3.0,
        BM_S + MU_S + BV_S + 1.0,
        BM_S * 2.0 + MU_S + BV_S + 1.0,
        BM_S * 2.0 + MU_S + 1.0,
        BV_S + 2.0,
        BM_S + BV_S,
        MU_S + 1.0,
        1.0 + (1.0 + BV_S) / 2.0,
        BM_S + BV_S + 2.0,
    ];
    let mut csg = [0.0; 17];
    for n in 0..17 {
        csg[n] = gamma(cse[n]);
    }

    // cge(12, nrhg), cgg(12, nrhg); Fortran column-major flattening
    let mut cge = [[0.0; NRHG]; 12];
    for m in 0..NRHG {
        cge[0][m] = BM_G + 1.0;
        cge[1][m] = MU_G + 1.0;
        cge[2][m] = BM_G + MU_G + 1.0;
        cge[3][m] = BM_G * 2.0 + MU_G + 1.0;
        cge[9][m] = MU_G + 2.0;
        cge[11][m] = BM_G * 0.5 + MU_G + 1.0;
    }
    for m in 0..NRHG {
        cge[4][m] = BM_G * 2.0 + MU_G + bv_g[m] + 1.0;
        cge[5][m] = BM_G + MU_G + bv_g[m] + 1.0;
        cge[6][m] = BM_G * 0.5 + MU_G + bv_g[m] + 1.0;
        cge[7][m] = MU_G + bv_g[m] + 1.0;
        cge[8][m] = MU_G + bv_g[m] + 3.0;
        cge[10][m] = 0.5 * (bv_g[m] + 5.0 + 2.0 * MU_G);
    }
    let mut cgg = [[0.0; NRHG]; 12];
    for m in 0..NRHG {
        for n in 0..12 {
            cgg[n][m] = gamma(cge[n][m]);
        }
    }
    let oge1 = 1.0 / cge[0][0];
    let ogg1 = 1.0 / cgg[0][0];
    let ogg2 = 1.0 / cgg[1][0];
    let ogg3 = 1.0 / cgg[2][0];

    let t1_qr_qc = PI * 0.25 * AV_R * crg[8];
    let t1_qr_qi = PI * 0.25 * AV_R * crg[8];
    let t2_qr_qi = PI * 0.25 * am_r * AV_R * crg[7];
    let t1_qs_qc = PI * 0.25 * AV_S;
    let t1_qs_qi = PI * 0.25 * AV_S;
    let t1_qr_ev = 0.78 * crg[9];
    let t2_qr_ev = 0.308 * sc3 * AV_R.sqrt() * crg[10];
    let t1_qs_sd = 0.86;
    let t2_qs_sd = 0.28 * sc3 * AV_S.sqrt();
    let t1_qs_me = PI * 4.0 * C_SQRD * olfus * 0.86;
    let t2_qs_me = PI * 4.0 * C_SQRD * olfus * 0.28 * sc3 * AV_S.sqrt();
    let t1_qg_sd = 0.86 * cgg[9][0];
    let t1_qg_me = PI * 4.0 * C_CUBE * olfus * 0.86 * cgg[9][0];

    b.set1("am_i", am_i);
    b.set1("am_r", am_r);
    b.set("am_g", &am_g);
    b.set1("lfus", lfus);
    b.set1("olfus", olfus);
    b.set1("orv", orv);
    b.set1("ar_volume", ar_volume);
    b.set1("sc3", sc3);
    b.set1("d0i", d0i);
    b.set1("xm0s", xm0s);
    b.set1("xm0g", xm0g);
    b.set1("obmi", obmi);
    b.set1("obmr", obmr);
    b.set1("oams", oams);
    b.set1("obms", obms);
    b.set1("ocms", ocms);
    b.set("oamg", &oamg);
    b.set("ocmg", &ocmg);
    b.set1("obmg", obmg);
    b.set1("rho_not", rho_not);
    b.set1("rho_not0", rho_not0);
    let flat = |a: &[[f64; 15]; 5]| -> Vec<f64> {
        let mut v = Vec::with_capacity(75);
        for n in 0..15 {
            for r in 0..5 {
                v.push(a[r][n]);
            }
        }
        v
    };
    b.set("cce", &flat(&cce));
    b.set("ccg", &flat(&ccg));
    b.set("ocg1", &ocg1);
    b.set("ocg2", &ocg2);
    b.set("cie", &cie);
    b.set("cig", &cig);
    b.set1("oig1", oig1);
    b.set1("oig2", oig2);
    b.set("cre", &cre);
    b.set("crg", &crg);
    b.set1("ore1", ore1);
    b.set1("org1", org1);
    b.set1("org2", org2);
    b.set1("org3", org3);
    b.set("cse", &cse);
    b.set("csg", &csg);
    let flatg = |a: &[[f64; NRHG]; 12]| -> Vec<f64> {
        let mut v = Vec::with_capacity(12 * NRHG);
        for m in 0..NRHG {
            for n in 0..12 {
                v.push(a[n][m]);
            }
        }
        v
    };
    b.set("cge", &flatg(&cge));
    b.set("cgg", &flatg(&cgg));
    b.set1("oge1", oge1);
    b.set1("ogg1", ogg1);
    b.set1("ogg2", ogg2);
    b.set1("ogg3", ogg3);
    b.set1("t1_qr_qc", t1_qr_qc);
    b.set1("t1_qr_qi", t1_qr_qi);
    b.set1("t2_qr_qi", t2_qr_qi);
    b.set1("t1_qs_qc", t1_qs_qc);
    b.set1("t1_qs_qi", t1_qs_qi);
    b.set1("t1_qr_ev", t1_qr_ev);
    b.set1("t2_qr_ev", t2_qr_ev);
    b.set1("t1_qs_sd", t1_qs_sd);
    b.set1("t2_qs_sd", t2_qs_sd);
    b.set1("t1_qs_me", t1_qs_me);
    b.set1("t2_qs_me", t2_qs_me);
    b.set1("t1_qg_sd", t1_qg_sd);
    b.set1("t1_qg_me", t1_qg_me);

    Derived { am_i, am_r, am_g, lfus, olfus, obmr, obmi, oig1, d0i, cce, ccg, ocg1, cie, cig }
}

/// create_bins (params.F90 693-720): (bins, deltabins).
fn create_bins(numbins: usize, lowbin: f64, highbin: f64) -> (Vec<f64>, Vec<f64>) {
    let mut xdx = vec![0.0; numbins + 1];
    xdx[0] = lowbin;
    xdx[numbins] = highbin;
    for n in 2..=numbins {
        xdx[n - 1] = ((n - 1) as f64 / numbins as f64 * (xdx[numbins] / xdx[0]).cln() + xdx[0].cln()).cexp();
    }
    let bins = (0..numbins).map(|n| (xdx[n] * xdx[n + 1]).sqrt()).collect();
    let delta = (0..numbins).map(|n| xdx[n + 1] - xdx[n]).collect();
    (bins, delta)
}

struct Bins {
    dc: Vec<f64>,
    dtc: Vec<f64>,
    di: Vec<f64>,
    dti: Vec<f64>,
    dr: Vec<f64>,
    ds: Vec<f64>,
    t_nc: Vec<f64>,
}

/// initialize_bins_for_tables (params.F90 611-656).
fn initialize_bins_for_tables(b: &mut Blob, d0i: f64) -> Bins {
    let log10_nint = |v: f64| nint(v.clog10()) as f64;
    b.set1("nic2", log10_nint(b.get("r_c")[0]));
    b.set1("nii2", log10_nint(b.get("r_i")[0]));
    b.set1("nir2", log10_nint(b.get("r_r")[0]));
    b.set1("nis2", log10_nint(b.get("r_s")[0]));
    b.set1("nig2", log10_nint(b.get("r_g")[0]));
    b.set1("nii3", log10_nint(b.get("nt_i")[0]));
    b.set1("nir3", log10_nint(b.get("n0r_exp")[0]));
    b.set1("nig3", log10_nint(b.get("n0g_exp")[0]));
    b.set1("niin2", log10_nint(b.get("nt_in")[0]));

    let mut dc = vec![0.0; NBC];
    let mut dtc = vec![0.0; NBC];
    dc[0] = D0C;
    dtc[0] = D0C;
    for n in 1..NBC {
        dc[n] = dc[n - 1] + 1.0e-6;
        dtc[n] = dc[n] - dc[n - 1];
    }
    let (di, dti) = create_bins(NBI, d0i, 2.0 * D0S);
    let (dr, dtr) = create_bins(NBR, D0R, 0.005);
    let (ds, dts) = create_bins(NBS, D0S, 0.02);
    let (dg, dtg) = create_bins(NBG, D0G, 0.05);
    let (t_nc0, _) = create_bins(NBC, 1.0, 3000.0);
    let t_nc: Vec<f64> = t_nc0.iter().map(|v| v * 1.0e6).collect();
    let nic1 = (t_nc[NBC - 1] / t_nc[0]).cln();
    b.set("dc", &dc);
    b.set("dtc", &dtc);
    b.set("di", &di);
    b.set("dti", &dti);
    b.set("dr", &dr);
    b.set("dtr", &dtr);
    b.set("ds", &ds);
    b.set("dts", &dts);
    b.set("dg", &dg);
    b.set("dtg", &dtg);
    b.set("t_nc", &t_nc);
    b.set1("nic1", nic1);
    Bins { dc, dtc, di, dti, dr, ds, t_nc }
}

/// compute_efrw (utils.F90 322-373).
fn compute_efrw(bins: &Bins) -> Vec<f64> {
    let mut t = vec![0.0; NBR * NBC];
    for j in 0..NBC {
        for i in 0..NBR {
            let (dc, dr) = (bins.dc[j], bins.dr[i]);
            let mut ef_rw = 0.0;
            let p = dc / dr;
            if dr < 50.0e-6 || dc < 3.0e-6 {
                t[i + NBR * j] = 0.0;
            } else if p > 0.25 {
                let x = dc * 1.0e6;
                ef_rw = if dr < 75.0e-6 {
                    0.026794 * x - 0.20604
                } else if dr < 125.0e-6 {
                    -0.00066842 * x * x + 0.061542 * x - 0.37089
                } else if dr < 175.0e-6 {
                    4.091e-06 * x * x * x * x - 0.00030908 * x * x * x + 0.0066237 * x * x
                        - 0.0013687 * x - 0.073022
                } else if dr < 250.0e-6 {
                    9.6719e-5 * x * x * x - 0.0068901 * x * x + 0.17305 * x - 0.65988
                } else if dr < 350.0e-6 {
                    9.0488e-5 * x * x * x - 0.006585 * x * x + 0.16606 * x - 0.56125
                } else {
                    0.00010721 * x * x * x - 0.0072962 * x * x + 0.1704 * x - 0.46929
                };
            } else {
                let vtr = -0.1021 + 4.932e3 * dr - 0.9551e6 * dr * dr + 0.07934e9 * dr * dr * dr
                    - 0.002362e12 * dr * dr * dr * dr;
                let stokes = dc * dc * vtr * RHO_W / (9.0 * 1.718e-5 * dr);
                let reynolds = 9.0 * stokes / (p * p * RHO_W);
                let f = reynolds.cln();
                let g = -0.1007 - 0.358 * f + 0.0261 * f * f;
                let k0 = g.cexp();
                let z = (stokes / (k0 + 1.0e-15)).cln();
                let h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
                let yc0 = 2.0 / PI * h.catan();
                ef_rw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p));
            }
            t[i + NBR * j] = 0.0f64.max(ef_rw.min(0.95));
        }
    }
    t
}

/// compute_efsw (utils.F90 376-413).
fn compute_efsw(bins: &Bins, am_r: f64, obmr: f64) -> Vec<f64> {
    let mut t = vec![0.0; NBS * NBC];
    for j in 0..NBC {
        let dc = bins.dc[j];
        let vtc = 1.19e4 * (1.0e4 * dc * dc * 0.25);
        for i in 0..NBS {
            let ds = bins.ds[i];
            let vts = AV_S * ds.cpow(BV_S) * (-FV_S * ds).cexp() - vtc;
            let ds_m = (AM_S * (ds * ds) / am_r).cpow(obmr);
            let p = dc / ds_m;
            if p > 0.25 || ds < D0S || dc < 6.0e-6 || vts < 1.0e-3 {
                t[i + NBS * j] = 0.0;
            } else {
                let stokes = dc * dc * vts * RHO_W / (9.0 * 1.718e-5 * ds_m);
                let reynolds = 9.0 * stokes / (p * p * RHO_W);
                let f = reynolds.cln();
                let g = -0.1007 - 0.358 * f + 0.0261 * f * f;
                let k0 = g.cexp();
                let z = (stokes / (k0 + 1.0e-15)).cln();
                let h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
                let yc0 = 2.0 / PI * h.catan();
                let ef_sw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p));
                t[i + NBS * j] = 0.0f64.max(ef_sw.min(0.95));
            }
        }
    }
    t
}

fn get_nuc(nc: f64) -> usize {
    if NU_C_SCALE / nc >= 12.5 {
        15
    } else {
        (nint(NU_C_SCALE / nc) + 2).min(15) as usize
    }
}

/// compute_drop_evap (utils.F90 466-500): (tpc_wev, tnc_wev).
fn compute_drop_evap(bins: &Bins, d: &Derived, r_c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut tpc = vec![0.0; NBC * NTB_C * NBC];
    let mut tnc = vec![0.0; NBC * NTB_C * NBC];
    let massc: Vec<f64> = (0..NBC).map(|n| d.am_r * bins.dc[n].cpow(BM_R)).collect();
    let mut n_c = vec![0.0; NBC];
    for k in 0..NBC {
        let nu_c = get_nuc(bins.t_nc[k]);
        for j in 0..NTB_C {
            let lamc = (bins.t_nc[k] * d.am_r * d.ccg[1][nu_c - 1] * d.ocg1[nu_c - 1] / r_c[j]).cpow(d.obmr);
            let n0_c = bins.t_nc[k] * d.ocg1[nu_c - 1] * lamc.cpow(d.cce[0][nu_c - 1]);
            for i in 0..NBC {
                n_c[i] = n0_c * powi(bins.dc[i], nu_c as i32) * (-lamc * bins.dc[i]).cexp() * bins.dtc[i];
                let mut summ = 0.0;
                let mut summ2 = 0.0;
                for n in 0..=i {
                    summ += massc[n] * n_c[n];
                    summ2 += n_c[n];
                }
                let idx = i + NBC * (j + NTB_C * k);
                tpc[idx] = summ;
                tnc[idx] = summ2;
            }
        }
    }
    (tpc, tnc)
}

/// calc_gamma_series (utils.F90, 9adf9ef).
fn calc_gamma_series(a: f64, x: f64) -> f64 {
    let it_max = 100;
    let smallvalue = 1.0e-7;
    let mut ap1 = a;
    let mut sum_term = 1.0 / gamma(ap1 + 1.0);
    let mut sum = sum_term;
    for _k in 1..=it_max {
        ap1 += 1.0;
        sum_term = sum_term * x / ap1;
        sum += sum_term;
        if sum_term.abs() < sum.abs() * smallvalue {
            break;
        }
    }
    sum * x.cpow(a) * (-x).cexp()
}

/// calc_gamma_cf (utils.F90, 9adf9ef).
fn calc_gamma_cf(a: f64, x: f64) -> f64 {
    let it_max = 100;
    let smallvalue = 1.0e-7;
    let offset = 1.0e-30;
    let mut b = 1.0 - a + x;
    let mut d = 1.0 / b;
    let h0 = offset;
    let mut c = b + 1.0 / offset;
    let mut delta = c * d;
    let mut h = h0 * delta;
    for k in 1..=it_max {
        let kf = k as f64;
        let aj = kf * (a - kf);
        b += 2.0;
        d = b + aj * d;
        if d.abs() < offset {
            d = offset;
        }
        c = b + aj / c;
        if c.abs() < offset {
            c = offset;
        }
        d = 1.0 / d;
        delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < smallvalue {
            break;
        }
    }
    (-x + a * x.cln()).cexp() * h / gamma(a)
}

fn calc_gamma_p(a: f64, x: f64) -> f64 {
    if x < a + 1.0 { calc_gamma_series(a, x) } else { 1.0 - calc_gamma_cf(a, x) }
}

/// qi_aut_qs (utils.F90 416-463): (tps_iaus, tni_iaus, tpi_ide).
fn qi_aut_qs(bins: &Bins, d: &Derived, nt_i: &[f64], r_i: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut tps = vec![0.0; NTB_I * NTB_I1];
    let mut tni = vec![0.0; NTB_I * NTB_I1];
    let mut tpi = vec![0.0; NTB_I * NTB_I1];
    let mut n_i = vec![0.0; NBI];
    for j in 0..NTB_I1 {
        for i in 0..NTB_I {
            let lami = (d.am_i * d.cig[1] * d.oig1 * nt_i[j] / r_i[i]).cpow(d.obmi);
            let di_mean = (BM_I + MU_I + 1.0) / lami;
            let n0_i = nt_i[j] * d.oig1 * lami.cpow(d.cie[0]);
            let mut t1 = 0.0;
            let mut t2 = 0.0;
            let idx = i + NTB_I * j;
            if di_mean > 5.0 * D0S {
                t1 = r_i[i];
                t2 = nt_i[j];
                tpi[idx] = 0.0;
            } else if di_mean < d.d0i {
                t1 = 0.0;
                t2 = 0.0;
                tpi[idx] = 1.0;
            } else {
                let xlimit_intg = lami * D0S;
                tpi[idx] = calc_gamma_p(MU_I + 2.0, xlimit_intg);
                for n2 in 0..NBI {
                    // di**mu_i with mu_i = 0.0 folds to 1.0
                    n_i[n2] = n0_i * 1.0 * (-lami * bins.di[n2]).cexp() * bins.dti[n2];
                    if bins.di[n2] >= D0S {
                        t1 += n_i[n2] * d.am_i * bins.di[n2].cpow(BM_I);
                        t2 += n_i[n2];
                    }
                }
            }
            tps[idx] = t1;
            tni[idx] = t2;
        }
    }
    (tps, tni, tpi)
}

// ---------------------------------------------------------------------------
// table readers (driver.F90 455-560), big-endian
// ---------------------------------------------------------------------------
fn read_exact_size(dir: &Path, name: &str, size: u64) -> Result<Vec<u8>, String> {
    let path = dir.join(name);
    let meta = fs::metadata(&path).map_err(|e| format!(
        "TEMPO table {} was not found ({e}); TEMPO needs ccn_activate.bin, freeze_water_data_tempo_v3, \
         qr_acr_qs_data_tempo_v3 and qr_acr_qg_data_tempo_v3 in one directory", path.display()))?;
    if meta.len() != size {
        return Err(format!("TEMPO table {} is {} bytes, expected {size}", path.display(), meta.len()));
    }
    fs::read(&path).map_err(|e| format!("reading TEMPO table {}: {e}", path.display()))
}

fn be_f64(bytes: &[u8]) -> Vec<f64> {
    bytes.chunks_exact(8).map(|c| f64::from_be_bytes(c.try_into().unwrap())).collect()
}

const N_CCN: usize = 7 * 9 * 7 * 5 * 4;
const N_FZ_R: usize = 37 * 37 * 45 * 55;
const N_FZ_C: usize = 37 * 100 * 45 * 55;
const N_QS: usize = 37 * 9 * 37 * 37;
const N_QG: usize = 37 * 37 * 9 * 37 * 37;

fn read_tables(b: &mut Blob, dir: &Path) -> Result<(), String> {
    // ccn: Fortran sequential unformatted record of real32 (4-byte markers)
    let raw = read_exact_size(dir, "ccn_activate.bin", (4 * N_CCN + 8) as u64)?;
    let head = u32::from_be_bytes(raw[0..4].try_into().unwrap()) as usize;
    let tail = u32::from_be_bytes(raw[raw.len() - 4..].try_into().unwrap()) as usize;
    if head != 4 * N_CCN || tail != 4 * N_CCN {
        return Err(format!("ccn_activate.bin record markers {head}/{tail}, expected {}", 4 * N_CCN));
    }
    let ccn: Vec<f64> = raw[4..4 + 4 * N_CCN]
        .chunks_exact(4)
        .map(|c| f32::from_be_bytes(c.try_into().unwrap()) as f64)
        .collect();
    b.set("tnccn_act", &ccn);

    let raw = read_exact_size(dir, "freeze_water_data_tempo_v3", (8 * (4 * N_FZ_R + 2 * N_FZ_C)) as u64)?;
    let mut o = 0usize;
    for (name, n) in [("tpi_qrfz", N_FZ_R), ("tni_qrfz", N_FZ_R), ("tpg_qrfz", N_FZ_R), ("tnr_qrfz", N_FZ_R),
                      ("tpi_qcfz", N_FZ_C), ("tni_qcfz", N_FZ_C)] {
        b.set(name, &be_f64(&raw[o..o + 8 * n]));
        o += 8 * n;
    }
    drop(raw);

    let raw = read_exact_size(dir, "qr_acr_qs_data_tempo_v3", (8 * 12 * N_QS) as u64)?;
    let mut o = 0usize;
    for name in ["tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1", "tms_sacr1",
                 "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"] {
        b.set(name, &be_f64(&raw[o..o + 8 * N_QS]));
        o += 8 * N_QS;
    }
    drop(raw);

    let raw = read_exact_size(dir, "qr_acr_qg_data_tempo_v3", (8 * 5 * N_QG) as u64)?;
    let mut o = 0usize;
    for name in ["tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"] {
        b.set(name, &be_f64(&raw[o..o + 8 * N_QG]));
        o += 8 * N_QG;
    }
    Ok(())
}

/// tempo_init (driver.F90 tempo_init) into the blob.
pub fn tempo_init(table_dir: &Path, opt: InitOptions) -> Result<Blob, String> {
    let mut b = Blob::new();
    set_parameters(&mut b);
    let (av_g, bv_g, dim_nrhg) = graupel_vars(opt.hailaware);
    b.set("av_g", &av_g);
    b.set("bv_g", &bv_g);
    b.set1("dim_nrhg", dim_nrhg as f64);
    let d = initialize_parameters(&mut b, &bv_g);
    let bins = initialize_bins_for_tables(&mut b, d.d0i);
    let _ = (d.lfus, d.olfus, d.am_g);

    b.set("t_efrw", &compute_efrw(&bins));
    b.set("t_efsw", &compute_efsw(&bins, d.am_r, d.obmr));
    let r_c = b.get("r_c").to_vec();
    let (tpc, tnc) = compute_drop_evap(&bins, &d, &r_c);
    b.set("tpc_wev", &tpc);
    b.set("tnc_wev", &tnc);
    let nt_i = b.get("nt_i").to_vec();
    let r_i = b.get("r_i").to_vec();
    let (tps, tni, tpi) = qi_aut_qs(&bins, &d, &nt_i, &r_i);
    b.set("tps_iaus", &tps);
    b.set("tni_iaus", &tni);
    b.set("tpi_ide", &tpi);

    read_tables(&mut b, table_dir)?;

    if opt.refl10cm_from_melting {
        let (s, ds) = create_bins(RADAR_BINS, 100.0e-6, 0.02);
        let (g, dg) = create_bins(RADAR_BINS, 100.0e-6, 0.05);
        b.set("sbins_radar", &s);
        b.set("dsbins_radar", &ds);
        b.set("gbins_radar", &g);
        b.set("dgbins_radar", &dg);
    }
    if opt.max_hail_diameter {
        let (h, dh) = create_bins(NHBINS, 500.0e-6, 0.075);
        b.set("hbins", &h);
        b.set("dhbins", &dh);
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_complete_and_sized() {
        let b = Blob::new();
        assert_eq!(FIELDS.len(), 249);
        assert_eq!(b.data.len(), FIELDS.iter().map(|(_, n)| n).sum::<usize>());
        assert_eq!(b.get("tcg_racg").len(), N_QG);
    }

    #[test]
    fn powi_matches_libgcc_order() {
        let x = 1.2345678901234567e-5;
        assert_eq!(powi(x, 3), x * (x * x));
        assert_eq!(powi(x, 4), (x * x) * (x * x));
        assert_eq!(powi(x, 0), 1.0);
    }

    #[test]
    fn decade_tables_have_tempo_sizes() {
        assert_eq!(decades(-6, -2).len(), 37);
        assert_eq!(decades(-10, -3).len(), 64);
        assert_eq!(decades(0, 6).len(), 55);
    }
}
