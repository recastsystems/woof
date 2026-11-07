// gpuwm/core/kernels/wdm6_refl.cu
//
// WDM6 radar reflectivity (WRF do_radar_ref=1, REFL_10CM at mp_physics=16).
// Transcription authority (byte-frozen WRF v4.6.1 reference bundle):
//   phys/module_mp_wdm6.F: refl10cm_wdm6 (:2957-3131) and the m(D)
//     parameters wdm6init hands radar_init (:2196-2208; xmu_r = 1 for rain,
//     0 for snow and graupel).
//   phys/module_mp_radar.F: the Blahak melting-particle backscatter
//     (rayleigh_soak_wetgraupel, get_m_mix_nested, get_m_mix,
//     m_complex_maxwellgarnett).
//
// WHY THIS IS A SEPARATE TRANSLATION UNIT.  Everything below the header is
// refl.cu's device prologue, copied verbatim.  It is copied rather than
// shared because refl.cu is BYTE-FROZEN: tests/test_mp8_frozen.py pins its
// file and compiled-source sha256 and its own docstring states the rule --
// a new scheme gets a new .cu file, so that no scheme's compiled numerics
// can move because another scheme was added.  gpuwm/core/kernels/wdm6.cu
// makes the same trade against wsm6.cu, for the same reason.
//
// The duplication is guarded rather than trusted: tests/test_wdm6.py
// asserts this file's copy of the prologue is byte-identical to refl.cu's,
// so the two cannot drift apart silently.  Because the copy is verbatim it
// carries Morrison constants WDM6 does not use; the two that would MISLEAD
// (RXAM_R at RHOW = 997, RXAM_S) are #undef'd immediately after the
// prologue ends, so this file cannot reach them.
//
// One CUDA thread per (j, i) column; the melting-level scan k_0 is a
// column-serial dependency.  radar_init products (size bins, Simpson
// weights, K_w, m_w_0, m_i_0) arrive from gpuwm.core.refl.radar_init_wdm6.

// Compile-time bound on the three column kernels' per-thread arrays.  The
// launchers (gpuwm/core/refl.py) specialize it to the field's own nz through
// get_kernel_int_defines; 256 is the unspecialized ceiling this source keeps
// when nothing overrides it, and the value gpuwm/core/refl.py validates nz
// against.  Nothing here reads the bound as a value -- every loop runs to
// the runtime nz -- so specializing it moves no arithmetic.  It costs the
// widest kernel (refl10cm_morrison_column) 72 B of local frame per level,
// which the driver answers with (frame - 1024) * 1536 * 170 bytes of backing
// store at first launch: 4,335 MiB at 256 against 624 MiB at 49.
#ifndef REFL_KMAX
#define REFL_KMAX 256
#endif
#define REFL_NRBINS 50

// Morrison module PI (module_mp_morr_two_moment.F:99), also the radar
// module PIx (module_mp_radar.F:283).
#define RPI 3.1415926535897932384626434
// module_model_constants.F:19 r_d = 287., Morrison's R (F:92).
#define R_D 287.0
// m(D) prefactors set before "call radar_init" (F:532-542): rain
// PI*RHOW/6 (RHOW = 997), snow CS = 100*PI/6.  Dense-ice CG is a
// runtime scalar because morr_rimed_ice selects hail or graupel.
#define RXAM_R (RPI * 997.0 / 6.0)
#define RXAM_S (100.0 * RPI / 6.0)
// radar_init gamma moments for xbm = 3, xmu = 0 (module_mp_radar.F:
// 150-175): xc?g(3) = Gamma(4), xc?g(4) = Gamma(7); the 1/Gamma(1+mu)
// factors xorg2/xosg2/xogg2 are exactly 1.
#define RGAMMA_4 6.0
#define RGAMMA_7 720.0
// PI5 with WRF's truncated pi, and lamda_radar = 0.10 m (:41/:76-77).
#define RPI5 (3.14159 * 3.14159 * 3.14159 * 3.14159 * 3.14159)
#define RLAMDA4 (0.10 * 0.10 * 0.10 * 0.10)
// melt_outside_s == melt_outside_g = 0.9 (:61-62).
#define RMELT_OUTSIDE 0.9

// ---- binary64 complex helpers (COMPLEX*16 arithmetic) --------------------

struct cd { double re, im; };

__device__ __forceinline__ cd cd_make(double re, double im)
{
    cd c; c.re = re; c.im = im; return c;
}

__device__ __forceinline__ cd cd_add(cd a, cd b)
{
    return cd_make(a.re + b.re, a.im + b.im);
}

__device__ __forceinline__ cd cd_sub(cd a, cd b)
{
    return cd_make(a.re - b.re, a.im - b.im);
}

__device__ __forceinline__ cd cd_mul(cd a, cd b)
{
    return cd_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ cd cd_div(cd a, cd b)
{
    double d = b.re * b.re + b.im * b.im;
    return cd_make((a.re * b.re + a.im * b.im) / d,
                   (a.im * b.re - a.re * b.im) / d);
}

__device__ __forceinline__ double cd_abs(cd a)
{
    return hypot(a.re, a.im);
}

// Principal square root (Fortran SQRT on COMPLEX*16).
__device__ __forceinline__ cd cd_sqrt(cd a)
{
    double m = cd_abs(a);
    double re = sqrt(fmax(0.5 * (m + a.re), 0.0));
    double im = sqrt(fmax(0.5 * (m - a.re), 0.0));
    return cd_make(re, a.im < 0.0 ? -im : im);
}

// Principal logarithm (Fortran LOG on COMPLEX*16).
__device__ __forceinline__ cd cd_log(cd a)
{
    return cd_make(log(cd_abs(a)), atan2(a.im, a.re));
}

__device__ __forceinline__ cd cd_scale(double s, cd a)
{
    return cd_make(s * a.re, s * a.im);
}

// ---- module_mp_radar.F per-particle machinery -----------------------------

// m_complex_maxwellgarnett (:544-590) on the only inclusion string
// radar_init configures ('spheroidal', :106-118; betas :574-576).  The
// Fortran's volume-closure/-999 error path (:558) is unreachable here:
// every call below passes fractions that sum to 1 by construction.
__device__ cd refl_maxwellgarnett(double vol1, double vol2, double vol3,
                                  cd m1, cd m2, cd m3)
{
    cd m1t = cd_mul(m1, m1);
    cd m2t = cd_mul(m2, m2);
    cd m3t = cd_mul(m3, m3);
    cd d2 = cd_sub(m2t, m1t);
    cd d3 = cd_sub(m3t, m1t);
    cd one = cd_make(1.0, 0.0);
    cd beta2 = cd_mul(cd_div(cd_scale(2.0, m1t), d2),
                      cd_sub(cd_mul(cd_div(m2t, d2),
                                    cd_log(cd_div(m2t, m1t))), one));
    cd beta3 = cd_mul(cd_div(cd_scale(2.0, m1t), d3),
                      cd_sub(cd_mul(cd_div(m3t, d3),
                                    cd_log(cd_div(m3t, m1t))), one));
    cd num = cd_add(cd_scale(1.0 - vol2 - vol3, m1t),
                    cd_add(cd_scale(vol2, cd_mul(beta2, m2t)),
                           cd_scale(vol3, cd_mul(beta3, m3t))));
    cd den = cd_add(cd_make(1.0 - vol2 - vol3, 0.0),
                    cd_add(cd_scale(vol2, beta2), cd_scale(vol3, beta3)));
    return cd_sqrt(cd_div(num, den));
}

// get_m_mix_nested (:362-489) on radar_init's Morrison string set --
// host='air', matrix='water', hostmatrix='icewater' (:106-118) -- which
// resolves to exactly two nested Maxwell-Garnett evaluations along the
// host='air' branch (:384-413): the ice-in-water inclusion mix
// (get_m_mix matrix='water', :517-519), then that mixture as 'ice'
// inclusions in the air host (get_m_mix matrix='ice', :514-516).
__device__ cd refl_m_mix_nested(cd m_a, cd m_i, cd m_w, double volair,
                                double volice, double volwater)
{
    double vol1 = volice / fmax(volice + volwater, 1.0e-10);   // :391
    double vol2 = 1.0 - vol1;                                  // :392
    cd mtmp = refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i);
    return refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                               mtmp, m_a, cd_scale(2.0, m_a));
}

// rayleigh_soak_wetgraupel (:265-358): Rayleigh backscattering cross
// section (m2) of one melting, water-coated ice particle of mass x_g.
__device__ double refl_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i)
{
    double fm = fmin(fmax(fmelt, 0.0), 1.0);                   // :290
    double mra = fmin(fmax(meltratio_outside, 0.0), 1.0);      // :292
    mra = mra + (1.0 - mra) * fm;                              // :299
    double x_w = x_g * fm;                                     // :301
    double d_g = a_geo * pow(x_g, b_geo);                      // :303
    if (d_g < 1.0e-12) return 0.0;                             // :305/:354-356
    double vg = RPI / 6.0 * d_g * d_g * d_g;                   // :307
    double rhog = fmin(fmax(x_g / vg, 10.0), 900.0);           // :308
    vg = x_g / rhog;                                           // :309
    double grenz = 1.0 - rhog / 1000.0;                        // :311
    double volg;
    if (mra <= grenz) {                                        // :313-317
        volg = vg * (1.0 - mra * fm);
    } else {                                                   // :319-331
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    double d_large = pow(6.0 / RPI * volg, 1.0 / 3.0);         // :335
    double volice = (x_g - x_w) / (volg * 900.0);              // :336
    double volwater = x_w / (1000.0 * volg);                   // :337
    double volair = 1.0 - volice - volwater;                   // :338
    cd m_core = refl_m_mix_nested(cd_make(1.0, 0.0), m_i, m_w,
                                  volair, volice, volwater);   // :342
    cd m2 = cd_mul(m_core, m_core);
    double kfac = cd_abs(cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                                cd_add(m2, cd_make(2.0, 0.0))));
    double d3 = d_large * d_large * d_large;
    return kfac * kfac * RPI5 * d3 * d3 / RLAMDA4;             // :351-352
}

// ---- Morrison refl10cm_hm column kernel -----------------------------------

// ^^ that heading is the last line of refl.cu's copied prologue, and this is
// where the copy ends.  RXAM_R and RXAM_S are DEAD below it and are removed
// here rather than left lying around: any later use is now a compile error,
// not a wrong number.  They cannot be deleted at their definition -- the
// block above is refl.cu's prologue byte-for-byte and
// tests/test_wdm6.py::test_the_wdm6_refl_prologue_is_byte_identical_to_
// refl_cu_s pins it that way -- and they are Morrison's constants, not
// WDM6's: RXAM_R uses RHOW = 997 (module_mp_morr_two_moment.F:532-542),
// while wdm6init calls radar_init with denr = rhowater = 1000
// (module_mp_wdm6.F:2196-2208).  The xam_r this kernel uses arrives from
// gpuwm.core.refl.radar_init_wdm6 and is PI*1000/6.  A reader checking rain
// density against WRF must not find 997 answering for WDM6.
#undef RXAM_R
#undef RXAM_S

// WDM6 refl10cm_wdm6 (module_mp_wdm6.F:2957-3131) with the radar module's
// particle routines (module_mp_radar.F:265-590), transcribed in the TYPES
// the Fortran declares, which is what makes it WRF's word:
//
//   * temp, qv, pres, rho, rr, nr, rs, rg, temp_C and the three ze arrays
//     are default REAL, so they are float32 here;
//   * lamr/lams/lamg, ilam*, N0_* are DOUBLE PRECISION, but lamr's RHS is
//     all REAL (radar_init's xam_r, xcrg, xorg2 and xobmr are REAL), so it
//     is a float32 pow assigned to a double; N0_s's exp(-alpha*temp_C) is a
//     float32 exp of WDM6's REAL alpha;
//   * every radar_init product (xam_*, xc?g, xc?e, xocm?, xobm?) is REAL and
//     arrives as the float32 word WRF computes (gpuwm.core.refl.
//     WDM6_RADAR_WORDS); K_w, PI5, lamda4, m_w_0 and m_i_0 arrive as WRF's
//     binary64 words, which are NOT the float64 formulas: PI5 is a REAL
//     product, lamda_radar = 0.10 is a REAL literal, and m_w_0 is the root
//     of a default (single) COMPLEX.
//
// Binary64 exp/log/pow are WOOF's own (g64_exp, g64_log, g64_pow), float32
// exp/pow are gfk_exp/gfk_pow.  LOG10, ATAN2 and HYPOT (complex ABS) are the
// device library's binary64 functions; their words are not the oracle
// host's in every case, which can only reach the float32 output through a
// rounding boundary of the final SNGL (tests/test_wdm6_wrf461_oracle.py
// measures it).  Complex division is gfortran's Fortran-rules lowering
// (Smith's ratio form, -fcx-fortran-rules), complex multiplication the
// plain component form.
//
// INHERITED INCONSISTENCY, transcribed as written (not a bug in the sense
// the "never bit-exact to a bug" rule covers -- nothing here is undefined,
// the two routines simply disagree): wdm62D consumes ncr as a VOLUMETRIC
// number, dividing by (q*den) at :2251, while this routine forms
// nr = nr1d*rho and rr = qr1d*rho so the densities cancel and nr1d enters
// as a PER-MASS number.  The two lamr definitions therefore differ by a
// factor rho inside the cube root (about 3 per cent in lamr near the
// surface).  WRF does this; reproducing it is the port's job, and the
// divergence is recorded here rather than silently repaired.

// Float32 words: the WDM6_RW_* indices of the launcher's word table.
#define WDM6_RW_PI     0    // module_mp_wdm6's REAL pi (4.*atan(1.))
#define WDM6_RW_XAM_R  1
#define WDM6_RW_XCRG3  2
#define WDM6_RW_XCRG4  3
#define WDM6_RW_XORG2  4
#define WDM6_RW_XOBMR  5
#define WDM6_RW_XCRE2  6
#define WDM6_RW_XCRE4  7
#define WDM6_RW_XAM_S  8
#define WDM6_RW_XCSG3  9
#define WDM6_RW_XCSG4  10
#define WDM6_RW_XCSE1  11
#define WDM6_RW_XCSE4  12
#define WDM6_RW_XOCMS  13
#define WDM6_RW_XOBMS  14
#define WDM6_RW_XBM_S  15
#define WDM6_RW_XMU_S  16
#define WDM6_RW_XAM_G  17
#define WDM6_RW_XCGG3  18
#define WDM6_RW_XCGG4  19
#define WDM6_RW_XCGE1  20
#define WDM6_RW_XCGE4  21
#define WDM6_RW_XOCMG  22
#define WDM6_RW_XOBMG  23
#define WDM6_RW_XBM_G  24
#define WDM6_RW_XMU_G  25
#define WDM6_RW_N0G    26
#define WDM6_RW_COUNT  27
// Binary64 words.
#define WDM6_RD_K_W     0
#define WDM6_RD_PI5     1
#define WDM6_RD_LAMDA4  2
#define WDM6_RD_MW_RE   3
#define WDM6_RD_MW_IM   4
#define WDM6_RD_MI_RE   5
#define WDM6_RD_MI_IM   6
#define WDM6_RD_COUNT   7

// Complex division as gfortran lowers it (-fcx-fortran-rules): Smith's
// ratio form on the larger of |c|, |d|, three divisions, no libcall.
__device__ __forceinline__ cd wdm6_cd_div(cd a, cd b)
{
    double ratio, div, tr, ti;
    if (fabs(b.re) < fabs(b.im)) {
        ratio = b.re / b.im;
        div = (b.re * ratio) + b.im;
        tr = (a.re * ratio) + a.im;
        ti = (a.im * ratio) - a.re;
    } else {
        ratio = b.im / b.re;
        div = (b.im * ratio) + b.re;
        tr = (a.im * ratio) + a.re;
        ti = a.im - (a.re * ratio);
    }
    return cd_make(tr / div, ti / div);
}

// Complex LOG and SQRT from their definitions: log|z| + i arg z, and the
// half-angle square root formed on the side that does not cancel.
__device__ __forceinline__ cd wdm6_cd_log(cd a)
{
    return cd_make(g64_log::log(hypot(a.re, a.im)), atan2(a.im, a.re));
}

__device__ __forceinline__ cd wdm6_cd_sqrt(cd a)
{
    double d = hypot(a.re, a.im);
    double r, s;
    if (a.re > 0.0) {
        r = sqrt(0.5 * (d + a.re));
        s = 0.5 * (a.im / r);
    } else {
        s = sqrt(0.5 * (d - a.re));
        r = fabs(0.5 * (a.im / s));
    }
    return cd_make(r, copysign(s, a.im));
}

// m_complex_maxwellgarnett (module_mp_radar.F:544-590), 'spheroidal'.
// Returns false on the volume-closure error (:556-562), where WRF's caller
// gives the particle no backscatter.
__device__ bool wdm6_maxwellgarnett(double vol1, double vol2, double vol3,
                                    cd m1, cd m2, cd m3, cd *out)
{
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6) return false;
    cd m1t = cd_mul(m1, m1);
    cd m2t = cd_mul(m2, m2);
    cd m3t = cd_mul(m3, m3);
    cd one = cd_make(1.0, 0.0);
    cd d2 = cd_sub(m2t, m1t);
    cd d3 = cd_sub(m3t, m1t);
    cd beta2 = cd_mul(wdm6_cd_div(cd_scale(2.0, m1t), d2),
                      cd_sub(cd_mul(wdm6_cd_div(m2t, d2),
                                    wdm6_cd_log(wdm6_cd_div(m2t, m1t))), one));
    cd beta3 = cd_mul(wdm6_cd_div(cd_scale(2.0, m1t), d3),
                      cd_sub(cd_mul(wdm6_cd_div(m3t, d3),
                                    wdm6_cd_log(wdm6_cd_div(m3t, m1t))), one));
    // ((1-vol2-vol3)*m1t + vol2*beta2*m2t) + vol3*beta3*m3t, each product
    // left to right, over ((1-vol2-vol3) + vol2*beta2) + vol3*beta3.
    double w1 = 1.0 - vol2 - vol3;
    cd num = cd_add(cd_add(cd_scale(w1, m1t),
                           cd_mul(cd_scale(vol2, beta2), m2t)),
                    cd_mul(cd_scale(vol3, beta3), m3t));
    cd den = cd_add(cd_add(cd_make(w1, 0.0), cd_scale(vol2, beta2)),
                    cd_scale(vol3, beta3));
    *out = wdm6_cd_sqrt(wdm6_cd_div(num, den));
    return true;
}

// rayleigh_soak_wetgraupel (module_mp_radar.F:265-358) through
// get_m_mix_nested on radar_init's strings (host 'air', matrix 'water',
// hostmatrix 'icewater', :106-118, :384-413) and get_m_mix (:494-541).
__device__ double wdm6_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i, double pi5, double lamda4)
{
    const double pix = 3.1415926535897932384626434;
    const cd m_air = cd_make(1.0, 0.0);
    double fm = fmax(fmin(fmelt, 1.0), 0.0);
    double mra = fmax(fmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    double x_w = x_g * fm;
    double d_g = a_geo * g64_pow::pow(x_g, b_geo);
    if (!(d_g >= 1.0e-12)) return 0.0;
    double vg = pix / 6.0 * (d_g * d_g * d_g);
    double rhog = fmax(fmin(x_g / vg, 900.0), 10.0);
    vg = x_g / rhog;
    double grenz = 1.0 - rhog / 1000.0;
    double volg;
    if (mra <= grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    // (6.0/PIx*volg)**(1./3.): the exponent is the REAL 1./3.
    double d_large = g64_pow::pow(6.0 / pix * volg, (double)(1.0f / 3.0f));
    double volice = (x_g - x_w) / (volg * 900.0);
    double volwater = x_w / (1000.0 * volg);
    double volair = 1.0 - volice - volwater;
    // get_m_mix_nested: the ice/water inclusion mix in a water matrix,
    // then that mixture as the 'ice' matrix around air with 2*m_a.
    double vol1 = volice / fmax(volice + volwater, 1.0e-10);
    double vol2 = 1.0 - vol1;
    cd mtmp, m_core;
    if (!wdm6_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &mtmp))
        return 0.0;
    if (!wdm6_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                             cd_scale(2.0, m_air), &m_core))
        return 0.0;
    cd m2 = cd_mul(m_core, m_core);
    cd q = wdm6_cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                       cd_add(m2, cd_make(2.0, 0.0)));
    double kfac = hypot(q.re, q.im);
    double d3 = d_large * d_large * d_large;
    // ABS(..)**2 * PI5 * D_large**6 / lamda4, D**6 = (D**3)*(D**3).
    return kfac * kfac * pi5 * (d3 * d3) / lamda4;
}

extern "C" __global__ void refl10cm_wdm6_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ nr_in, const real* __restrict__ qs,
    const real* __restrict__ qg,
    const real* __restrict__ t, const real* __restrict__ p,
    const double* __restrict__ tables,
    const float* __restrict__ rw, const double* __restrict__ rd,
    real* __restrict__ refl,
    const int nz, const int ny, const int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx, i = col - j * nx;
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const float pi = rw[WDM6_RW_PI];
    const float xam_r = rw[WDM6_RW_XAM_R], xam_s = rw[WDM6_RW_XAM_S];
    const float xam_g = rw[WDM6_RW_XAM_G];
    const float n0g = rw[WDM6_RW_N0G];
    const double k_w = rd[WDM6_RD_K_W], pi5 = rd[WDM6_RD_PI5];
    const double lamda4 = rd[WDM6_RD_LAMDA4];
    const cd mw = cd_make(rd[WDM6_RD_MW_RE], rd[WDM6_RD_MW_IM]);
    const cd mi = cd_make(rd[WDM6_RD_MI_RE], rd[WDM6_RD_MI_IM]);
    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0r[REFL_KMAX], n0s[REFL_KMAX];
    const double n0g_d = (double)n0g;   // N0_g(k) = n0g at every level
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];
    for (int k = 0; k < nz; ++k) {                         // :2996-3037
        size_t idx = I3(k, j, i, ny, nx);
        const float tk = t[idx];
        float temp_c = fminf(-0.001f, tk - 273.15f);
        float qvk = fmaxf(1.0e-10f, qv[idx]);
        float pres = p[idx];
        float rho = 0.622f * pres / (287.0f * tk * (qvk + 0.622f));
        lqr[k] = lqs[k] = lqg[k] = false;
        if (qr[idx] > 1.0e-9f) {
            rr[k] = qr[idx] * rho;
            float nrk = nr_in[idx] * rho;
            double lamr = (double)gfk_pow(xam_r * rw[WDM6_RW_XCRG3]
                                          * rw[WDM6_RW_XORG2] * nrk / rr[k],
                                          rw[WDM6_RW_XOBMR]);
            ilamr[k] = 1.0 / lamr;
            n0r[k] = (double)(nrk * rw[WDM6_RW_XORG2])
                   * g64_pow::pow(lamr, (double)rw[WDM6_RW_XCRE2]);
            lqr[k] = true;
        } else {
            rr[k] = 1.0e-12f;
        }
        if (qs[idx] > 1.0e-9f) {
            rs[k] = qs[idx] * rho;
            n0s[k] = (double)fminf(1.0e11f, 2.0e6f * gfk_exp(-(0.12f * temp_c)));
            double lams = g64_pow::pow(
                (double)(xam_s * rw[WDM6_RW_XCSG3]) * n0s[k] / (double)rs[k],
                (double)(1.0f / rw[WDM6_RW_XCSE1]));
            ilams[k] = 1.0 / lams;
            lqs[k] = true;
        } else {
            rs[k] = 1.0e-12f;
        }
        if (qg[idx] > 1.0e-9f) {
            rg[k] = qg[idx] * rho;
            double lamg = g64_pow::pow(
                (double)(xam_g * rw[WDM6_RW_XCGG3]) * n0g_d / (double)rg[k],
                (double)(1.0f / rw[WDM6_RW_XCGE1]));
            ilamg[k] = 1.0 / lamg;
            lqg[k] = true;
        } else {
            rg[k] = 1.0e-12f;
        }
    }
    bool melti = false; int k0 = 0;                         // :3042-3052
    for (int k = nz - 2; k >= 0; --k) {
        if (t[I3(k, j, i, ny, nx)] > 273.15f && lqr[k]
            && (lqs[k + 1] || lqg[k + 1])) {
            k0 = max(k + 1, k0); melti = true; break;
        }
    }
    // (0.176/0.93)*(6.0/PI)*(6.0/PI)*(xam/900.0)*(xam/900.0), all REAL.
    const float six_pi = 6.0f / pi;
    const float xs900 = __fdiv_rn(xam_s, 900.0f);
    const float xg900 = __fdiv_rn(xam_g, 900.0f);
    const float cs = (0.176f / 0.93f) * six_pi * six_pi * xs900 * xs900;
    const float cg = (0.176f / 0.93f) * six_pi * six_pi * xg900 * xg900;
    for (int k = 0; k < nz; ++k) {                          // :3060-3071
        zer[k] = zes[k] = zeg[k] = 1.0e-22f;
        if (lqr[k]) zer[k] = (float)(n0r[k] * (double)rw[WDM6_RW_XCRG4]
                            * g64_pow::pow(ilamr[k], (double)rw[WDM6_RW_XCRE4]));
        if (lqs[k]) zes[k] = (float)((double)cs * n0s[k]
                            * (double)rw[WDM6_RW_XCSG4]
                            * g64_pow::pow(ilams[k], (double)rw[WDM6_RW_XCSE4]));
        if (lqg[k]) zeg[k] = (float)((double)cg * n0g_d
                            * (double)rw[WDM6_RW_XCGG4]
                            * g64_pow::pow(ilamg[k], (double)rw[WDM6_RW_XCGE4]));
    }
    if (melti && k0 >= 1) {                                 // :3082-3124
        for (int k = k0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k0]) {
                double fm = fmax(0.005, fmin(1.0 - (double)(rs[k] / rs[k0]), 0.99));
                double eta = 0.0, lams = 1.0 / ilams[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = (double)xam_s
                             * g64_pow::pow(xxds[n], (double)rw[WDM6_RW_XBM_S]);
                    double cb = wdm6_rayleigh_soak_wetgraupel(
                        x, (double)rw[WDM6_RW_XOCMS], (double)rw[WDM6_RW_XOBMS],
                        fm, RMELT_OUTSIDE, mw, mi, pi5, lamda4);
                    double f_d = n0s[k]
                               * g64_pow::pow(xxds[n], (double)rw[WDM6_RW_XMU_S])
                               * g64_exp::exp(-lams * xxds[n]);
                    eta = eta + f_d * cb * simpson[n] * xdts[n];
                }
                zes[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
            if (lqg[k] && lqg[k0]) {
                double fm = fmax(0.005, fmin(1.0 - (double)(rg[k] / rg[k0]), 0.99));
                double eta = 0.0, lamg = 1.0 / ilamg[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = (double)xam_g
                             * g64_pow::pow(xxdg[n], (double)rw[WDM6_RW_XBM_G]);
                    double cb = wdm6_rayleigh_soak_wetgraupel(
                        x, (double)rw[WDM6_RW_XOCMG], (double)rw[WDM6_RW_XOBMG],
                        fm, RMELT_OUTSIDE, mw, mi, pi5, lamda4);
                    double f_d = n0g_d
                               * g64_pow::pow(xxdg[n], (double)rw[WDM6_RW_XMU_G])
                               * g64_exp::exp(-lamg * xxdg[n]);
                    eta = eta + f_d * cb * simpson[n] * xdtg[n];
                }
                zeg[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
        }
    }
    for (int k = 0; k < nz; ++k) {                          // :3126-3128, :294
        float zsum = zer[k] + zes[k] + zeg[k];
        float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[I3(k, j, i, ny, nx)] = fmaxf(-35.0f, dbz);
    }
}
