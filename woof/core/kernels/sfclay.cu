// gpuwm/core/kernels/sfclay.cu
//
// WRF v4.6.1 MM5 surface-layer schemes, transcribed from
//   phys/module_sf_sfclay.F          SFCLAY1D (option 91), and
//   phys/physics_mmm/sf_sfclayrev.F90 sf_sfclayrev_run (option 1).
// module_sf_sfclayrev.F is only the WRF/CCPP array wrapper.  One CUDA thread
// handles one (j,i) surface column; inputs are the lowest mass-level values
// already interpolated to mass points exactly as the WRF wrappers do.
//
// The incoming znt, ust, ustm, mol, hfx, qfx, qsfc, and zol are previous-
// step/inout fields.  This matters: classic unstable z/L uses old MOL, its
// strong-stable branch leaves ZOL untouched, and both schemes average newly
// diagnosed u* with old UST and old USTM.  Outputs remain FP32 model state;
// gpuwm.verify.npref.np_sfclay is the float64 transcription mirror.
//
// Bitwise WRF for option 1 (tools/sfclayrev_wrf461_oracle, graded by
// tests/test_sfclayrev_wrf461_parity.py against the pinned strict recipe's
// build: gfortran 13.3 -O2 -fno-fast-math -ffp-contract=off
// -fno-tree-vectorize, the wrf.exe every strict proof runs against).
// What that takes, each one measured on the oracle:
//   * the unit compiles without multiply-add contraction (kernels/__init__.py
//     _NO_FMAD_MODULES), as gfortran on baseline x86-64 never contracts;
//   * ALOG, EXP, real ** and ATAN are WOOF's own float32 routines
//     (sf_log/sf_exp/sf_pow/sf_atan below), which return the words the
//     reference's libm returns; CUDA's logf/expf/powf/atanf are different
//     functions;
//   * every REAL power is the powf call gfortran emits, including x**0.5
//     (not sqrtf); integer powers stay products; and x**2. is the product
//     x*x, because at -O2 gfortran folds powf(x, 2.0) into a multiply (the
//     strict wrf.exe's psim/psih_unstable_full call powf twice, not four
//     times).  Keeping those four as powf calls, the -O0 spelling, put
//     1-ULP surface fluxes in 3 of 12,000 UAE 9 km columns after one
//     physics-on step against the strict wrf.exe 4.6.1 (2026-10-29);
//   * every expression keeps WRF's own association, read from gfortran's
//     -fdump-tree-original of the pinned source: the Exner factor's
//     (p1000mb*0.001)/(p/1000.) base, rhox's (psfc/1000.)*1000., the land
//     scalar's (0.01/za)*zol, sqrt(sqrt(restar)), -((zl/0.07)*0.40), and
//     the wind speed as sqrt(u*u+v*v), not hypotf.

// WOOF's own float32 libm words (glibc_flt32.cuh's gfk_*, prepended by the
// loader for this unit) in place of CUDA's builtins.
//
// __noinline__, not __forceinline__: inlining the glibc routines into every
// call site of this unit made NVRTC take 114.6 s for this one module on a
// cold kernel cache (RTX PRO 6000, sm_120, NVRTC 13.4), on one core, with
// the card idle inside the first step of every fresh box's first forecast
// (2.9.2 box door test).  As calls they compile in 14.7 s.  The unit is
// built with --fmad=false, so no multiply-add can be contracted across the
// call boundary: every float operation and its rounding is unchanged.
__device__ __noinline__ real sf_log(real x) { return gfk_log(x); }
__device__ __noinline__ real sf_exp(real x) { return gfk_exp(x); }
__device__ __noinline__ real sf_pow(real x, real y) { return gfk_pow(x, y); }

// WOOF's own float32 atanf: the same routine and the same 19 words as
// mynn_pbl.cu's mynn_glibc_atanf / MYNN_ATANF_TAB (the gate in
// tests/test_sfclayrev_wrf461_parity.py holds the two tables equal).  Every
// operation is a plain FP32 multiply or add, none contracted.
__constant__ unsigned int SFC_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__device__ real sf_atan(real x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int) hx;
    int id;
#define SFC_ATAN_HI(i) __uint_as_float(SFC_ATANF_TAB[(i)])
#define SFC_ATAN_LO(i) __uint_as_float(SFC_ATANF_TAB[4 + (i)])
#define SFC_ATAN_T(i)  __uint_as_float(SFC_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return FADD(x, x);
        if (signed_hx > 0) return FADD(SFC_ATAN_HI(3), SFC_ATAN_LO(3));
        return FSUB(-SFC_ATAN_HI(3), SFC_ATAN_LO(3));
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);                 // |x|
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = FDIV(FSUB(FMUL(2.0f, x), 1.0f), FADD(2.0f, x));
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = FDIV(FSUB(x, 1.0f), FADD(x, 1.0f));
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = FDIV(FSUB(x, 1.5f), FADD(1.0f, FMUL(1.5f, x)));
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = FDIV(-1.0f, x);
        }
    }
    real z = FMUL(x, x);
    real w = FMUL(z, z);
    real s1 = FMUL(w, SFC_ATAN_T(10));
    s1 = FMUL(w, FADD(SFC_ATAN_T(8), s1));
    s1 = FMUL(w, FADD(SFC_ATAN_T(6), s1));
    s1 = FMUL(w, FADD(SFC_ATAN_T(4), s1));
    s1 = FMUL(w, FADD(SFC_ATAN_T(2), s1));
    s1 = FMUL(z, FADD(SFC_ATAN_T(0), s1));
    real s2 = FMUL(w, SFC_ATAN_T(9));
    s2 = FMUL(w, FADD(SFC_ATAN_T(7), s2));
    s2 = FMUL(w, FADD(SFC_ATAN_T(5), s2));
    s2 = FMUL(w, FADD(SFC_ATAN_T(3), s2));
    s2 = FMUL(w, FADD(SFC_ATAN_T(1), s2));
    real s = FADD(s1, s2);
    if (id < 0) return FSUB(x, FMUL(x, s));
    real r = FSUB(SFC_ATAN_HI(id),
                  FSUB(FSUB(FMUL(x, s), SFC_ATAN_LO(id)), x));
    return (signed_hx < 0) ? -r : r;
#undef SFC_ATAN_HI
#undef SFC_ATAN_LO
#undef SFC_ATAN_T
}

// Constants gfortran folds at compile time in the pinned source, written as
// the words it folded (-fdump-tree-original): 2.*ATAN(1.), SQRT(3.),
// 4.*ATAN(1.)/SQRT(3.), 1./1.1, 1./2.5, SQRT(0.71), SQRT(0.60).
#define SF_TWO_ATAN1   1.57079637050628662109375f
#define SF_SQRT3       1.73205077648162841796875f
#define SF_FOUR_ATAN1_OVER_SQRT3 1.81379950046539306640625f
#define SF_INV_1P1     9.0909087657928466796875e-1f
#define SF_INV_2P5     4.000000059604644775390625e-1f
#define SF_SQRT_0P71   8.42614948749542236328125e-1f
#define SF_SQRT_0P60   7.74596691131591796875e-1f

__device__ __forceinline__ double sf_f2d(real x)
{
    // Bit-decoded float->double widening.  The f32->f64 cvt this compile
    // route emits DAZes subnormal inputs (measured on the RTX 5090, same
    // finding as rrtmg_sw.cu's rsw_f2d), so (double)z0 of a subnormal
    // roughness would arrive as 0.0.
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix & 0x80000000u) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

__device__ __forceinline__ real sf_log_zratio(real num, real z0)
{
    // A NaN argument is already-corrupted state, not a degenerate-but-real
    // roughness, and the rescue below would launder it: its floor is
    // written !(zd > 0.0), which is true for NaN, so a NaN znt would leave
    // here as log(num) - log(FLT_MIN) -- an ordinary finite contrast around
    // 90 that no downstream check can tell from a real answer.  Zero
    // roughness is a physical input this function defines; NaN roughness is
    // a defect, and propagating it is what keeps it visible.
    if (isnan(num) || isnan(z0)) return num + z0;   // NaN, payload preserved
    // Healthy path first: WRF's own single-quotient spelling, so every
    // z0 the quotient survives keeps its exact FP32 word
    // (module_sf_sfclay.F:494 GZ1OZ0 = ALOG(ZA/ZNT);
    // sf_sfclayrev.F90:318 alog((za+znt)/znt)).
    real r = sf_log(num / z0);
    if (isfinite(r)) return r;
    // num/z0 overflows FP32 once z0 < num/FLT_MAX (~2.9e-38 for a 10 m
    // level) -- long before the logarithm itself is out of range -- and
    // z0 == 0 divides by zero.  WRF v4.6.1 has no guard at either site:
    // ZNT=0 sends GZ1OZ0=Inf into PSIX and U10 = Inf/Inf = NaN, which is
    // undefined behaviour, not an answer.  Rescue in float64, where the
    // quotient of any two nonzero floats is finite: log(num) - log(z0).
    // FP32 logf is unusable here because this route flushes subnormal
    // inputs (logf(1e-38f) measures -inf on the 5090), hence sf_f2d and
    // double log.  z0 <= 0 floors at FLT_MIN, turning zero roughness
    // into a defined huge-but-finite contrast (gz1 ~ 90) whose drag and
    // exchange coefficients go to zero smoothly.  Documented divergence:
    // defined behaviour where WRF is undefined.
    double zd = sf_f2d(z0);
    if (!(zd > 0.0)) zd = 1.1754943508222875e-38;   // FLT_MIN
    return (real)(log((double)num) - log(zd));
}

__device__ __forceinline__ real sf_psim_classic_full(real z)
{
    // module_sf_sfclay.F:961-963 (sfclayinit).
    real x = sf_pow(1.0f - 16.0f * z, 0.25f);
    return 2.0f * sf_log(0.5f * (1.0f + x))
         + sf_log(0.5f * (1.0f + x * x)) - 2.0f * sf_atan(x)
         + SF_TWO_ATAN1;
}

__device__ __forceinline__ real sf_psih_classic_full(real z)
{
    // module_sf_sfclay.F:964: (1-16*ZOLN)**0.5 is a powf call, not sqrtf.
    real y = sf_pow(1.0f - 16.0f * z, 0.5f);
    return 2.0f * sf_log(0.5f * (1.0f + y));
}

__device__ __forceinline__ real sf_classic_table(real z, bool heat)
{
    real x = fminf(fmaxf(-z, 0.0f), 9.9999f) * 100.0f;
    int n = (int)x;
    real r = x - (real)n;
    real z0 = -0.01f * (real)n, z1 = -0.01f * (real)(n + 1);
    real f0 = heat ? sf_psih_classic_full(z0) : sf_psim_classic_full(z0);
    real f1 = heat ? sf_psih_classic_full(z1) : sf_psim_classic_full(z1);
    return f0 + r * (f1 - f0);
}

__device__ __forceinline__ real sf_psim_stable_full(real z)
{
    // sf_sfclayrev.F90:991.
    return -6.1f * sf_log(z + sf_pow(1.0f + sf_pow(z, 2.5f), SF_INV_2P5));
}

__device__ __forceinline__ real sf_psih_stable_full(real z)
{
    // sf_sfclayrev.F90:999.
    return -5.3f * sf_log(z + sf_pow(1.0f + sf_pow(z, 1.1f), SF_INV_1P1));
}

__device__ __forceinline__ real sf_psim_unstable_full(real z)
{
    // sf_sfclayrev.F90:1008-1014.  ym**2. and zolf**2. are products: the
    // strict -O2 build folds powf(x, 2.0) into x*x (header note); zolf**2
    // (integer power) is a product at any optimisation level.
    real x = sf_pow(1.0f - 16.0f * z, 0.25f);
    real psimk = 2.0f * sf_log(0.5f * (1.0f + x))
               + sf_log(0.5f * (1.0f + x * x)) - 2.0f * sf_atan(x)
               + SF_TWO_ATAN1;
    real ym = sf_pow(1.0f - 10.0f * z, 0.33f); // file literal .33
    real psimc = 1.5f * sf_log(__fdiv_rn((ym * ym + ym + 1.0f), 3.0f))
                - SF_SQRT3 * sf_atan(__fdiv_rn((2.0f * ym + 1.0f), SF_SQRT3))
                + SF_FOUR_ATAN1_OVER_SQRT3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

__device__ __forceinline__ real sf_psih_unstable_full(real z)
{
    // sf_sfclayrev.F90:1023-1029.  (1.-16.*zolf)**.5 is a powf call; yh**2.
    // and zolf**2. are products, as in psim above.
    real y = sf_pow(1.0f - 16.0f * z, 0.5f);
    real psihk = 2.0f * sf_log((1.0f + y) / 2.0f);
    real yh = sf_pow(1.0f - 34.0f * z, 0.33f);
    real psihc = 1.5f * sf_log(__fdiv_rn((yh * yh + yh + 1.0f), 3.0f))
                - SF_SQRT3 * sf_atan(__fdiv_rn((2.0f * yh + 1.0f), SF_SQRT3))
                + SF_FOUR_ATAN1_OVER_SQRT3;
    return (psihk + z * z * psihc) / (1.0f + z * z);
}

__device__ __forceinline__ real sf_rev_table(real z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable
    bool unstable = which >= 2;
    z = unstable ? fminf(z, 0.0f) : fmaxf(z, 0.0f);
    real x = (unstable ? -z : z) * 100.0f;
    int n = (int)x;
    if (n + 1 >= 1000) {
        if (which == 0) return sf_psim_stable_full(z);
        if (which == 1) return sf_psih_stable_full(z);
        if (which == 2) return sf_psim_unstable_full(z);
        return sf_psih_unstable_full(z);
    }
    real r = x - (real)n;
    real sign = unstable ? -1.0f : 1.0f;
    real z0 = sign * 0.01f * (real)n;
    real z1 = sign * 0.01f * (real)(n + 1);
    real f0, f1;
    if (which == 0) { f0 = sf_psim_stable_full(z0); f1 = sf_psim_stable_full(z1); }
    else if (which == 1) { f0 = sf_psih_stable_full(z0); f1 = sf_psih_stable_full(z1); }
    else if (which == 2) { f0 = sf_psim_unstable_full(z0); f1 = sf_psim_unstable_full(z1); }
    else { f0 = sf_psih_unstable_full(z0); f1 = sf_psih_unstable_full(z1); }
    return f0 + r * (f1 - f0);
}

__device__ __forceinline__ real sf_psim_stable(real z) { return sf_rev_table(z, 0); }
__device__ __forceinline__ real sf_psih_stable(real z) { return sf_rev_table(z, 1); }
__device__ __forceinline__ real sf_psim_unstable(real z) { return sf_rev_table(z, 2); }
__device__ __forceinline__ real sf_psih_unstable(real z) { return sf_rev_table(z, 3); }

__device__ __forceinline__ real sf_zolri_residual(real &zeta, real ri,
                                                   real z, real z0)
{
    if (zeta * ri < 0.0f) zeta = 0.0f;
    real zeta0 = zeta * z0 / z;
    real zeta3 = zeta + zeta0;
    real fm, fh;
    // sf_log_zratio, not bare logf: a degenerate z0 otherwise turns the
    // residual into zeta*Inf/Inf = NaN and poisons the whole zol solve.
    if (ri < 0.0f) {
        fm = sf_log_zratio(z + z0, z0)
           - (sf_psim_unstable(zeta3) - sf_psim_unstable(zeta0));
        fh = sf_log_zratio(z + z0, z0)
           - (sf_psih_unstable(zeta3) - sf_psih_unstable(zeta0));
    } else {
        fm = sf_log_zratio(z + z0, z0)
           - (sf_psim_stable(zeta3) - sf_psim_stable(zeta0));
        fh = sf_log_zratio(z + z0, z0)
           - (sf_psih_stable(zeta3) - sf_psih_stable(zeta0));
    }
    return zeta * fh / (fm * fm) - ri;
}

__device__ __forceinline__ real sf_zolri(real ri, real z, real z0)
{
    real x1 = ri < 0.0f ? -5.0f : 0.0f;
    real x2 = ri < 0.0f ? 0.0f : 5.0f;
    real fx1 = sf_zolri_residual(x1, ri, z, z0);
    real fx2 = sf_zolri_residual(x2, ri, z, z0);
    real result = fabsf(fx1) < fabsf(fx2) ? x1 : x2;
    int iter = 0;
    while (fabsf(x1 - x2) > 0.01f) {
        if (iter == 10 || fx1 == fx2) return result;
        if (fabsf(fx2) < fabsf(fx1)) {
            x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1);
            fx1 = sf_zolri_residual(x1, ri, z, z0);
            result = x1;
        } else {
            x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1);
            fx2 = sf_zolri_residual(x2, ri, z, z0);
            result = x2;
        }
        ++iter;
    }
    return result;
}

__device__ __forceinline__ real sf_rev_heat_psi(real zol, real za,
                                                 real rough, real height)
{
    real zh = zol * (height + rough) / za;
    real z0 = zol * rough / za;
    // sf_sfclayrev.F90:545-559's test order: > 0, then == 0, else unstable.
    if (zol > 0.0f) return sf_psih_stable(zh) - sf_psih_stable(z0);
    if (zol == 0.0f) return 0.0f;
    return sf_psih_unstable(zh) - sf_psih_unstable(z0);
}

extern "C" __global__
void sfclay_column(
    const real* __restrict__ u, const real* __restrict__ v,
    const real* __restrict__ t, const real* __restrict__ qv,
    const real* __restrict__ p, const real* __restrict__ dz8w,
    const real* __restrict__ psfc, const real* __restrict__ tsk,
    const real* __restrict__ pblh, const real* __restrict__ mavail,
    const real* __restrict__ xland, const real* __restrict__ lakemask,
    real* __restrict__ znt, real* __restrict__ ust,
    real* __restrict__ ustm, real* __restrict__ mol,
    real* __restrict__ hfx, real* __restrict__ qfx, real* __restrict__ qsfc,
    real* __restrict__ zol_o, real* __restrict__ regime_o,
    real* __restrict__ psim_o, real* __restrict__ psih_o,
    real* __restrict__ fm_o, real* __restrict__ fh_o,
    real* __restrict__ lh_o, real* __restrict__ u10_o,
    real* __restrict__ v10_o, real* __restrict__ th2_o,
    real* __restrict__ t2_o, real* __restrict__ q2_o,
    real* __restrict__ chs_o, real* __restrict__ chs2_o,
    real* __restrict__ cqs2_o, real* __restrict__ flhc_o,
    real* __restrict__ flqc_o, real* __restrict__ qgh_o,
    real* __restrict__ rmol_o, real* __restrict__ wspd_o,
    real* __restrict__ br_o, real* __restrict__ gz1_o,
    real* __restrict__ cpm_o, real* __restrict__ ck_o,
    real* __restrict__ cka_o, real* __restrict__ cd_o,
    real* __restrict__ cda_o,
    real dx, int option, int isfflx, int isftcflx, int iz0tlnd, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (option == 91) {
        // Classic MM5 (module_sf_sfclay.F) is its own transcription,
        // sfclay_classic.cuh, graded bitwise against the WRF Fortran.  The
        // option == 91 arms left in the body below are no longer reached.
        sfclay_classic_point(u[idx], v[idx], t[idx], qv[idx], p[idx],
            dz8w[idx], psfc[idx], tsk[idx], pblh[idx], mavail[idx],
            xland[idx], lakemask[idx], idx,
            znt, ust, ustm, mol, hfx, qfx, qsfc,
            zol_o, regime_o, psim_o, psih_o, fm_o, fh_o, lh_o, u10_o, v10_o,
            th2_o, t2_o, q2_o, chs_o, chs2_o, cqs2_o, flhc_o, flqc_o, qgh_o,
            rmol_o, wspd_o, br_o, gz1_o, cpm_o, ck_o, cka_o, cd_o, cda_o,
            dx, isfflx, isftcflx, iz0tlnd);
        return;
    }

    const real karman = 0.4f, ep1 = RV / RD - 1.0f, xka = 2.4e-5f;
    bool land = xland[idx] < 1.5f;
    real uu = u[idx], vv = v[idx], temp = t[idx], qvx = qv[idx];
    real press = p[idx], ps = psfc[idx], ground_t = tsk[idx];
    real z0 = znt[idx], old_ust = ust[idx], old_mol = mol[idx];
    real old_ustm = ustm[idx];
    real old_zol = zol_o[idx];
    real old_hfx = hfx[idx], old_qfx = qfx[idx], qs = qsfc[idx];

    // Both schemes: THGB=TSK*(P1000mb/PSFCPA)**ROVCP, and the air's Exner
    // factor from pressure in cb, THCON=(P1000mb*0.001/PL)**ROVCP with
    // PL=P/1000. (sf_sfclayrev.F90:230,255-259; module_sf_sfclay.F:405,
    // 430-434).  P1000mb*0.001 rounds to 100.0000076 in float32, so this is
    // not (P0/p)**ROVCP.
    real thgb = ground_t * sf_pow(P0 / ps, RCP);
    real pl = __fdiv_rn(press, 1000.0f);
    real thx = temp * sf_pow(__fdiv_rn(__fmul_rn(P0, 0.001f), pl), RCP);
    real thvx = thx * (1.0f + ep1 * qvx);
    real tv = temp * (1.0f + ep1 * qvx);
    real cpm = CP * (1.0f + 0.8f * qvx);
    real psfc_cb = __fdiv_rn(ps, 1000.0f);
    real es = SVP1 * sf_exp(SVP2 * (ground_t - SVPT0) / (ground_t - SVP3));
    if (!land && lakemask[idx] == 0.0f) es *= 0.98f;
    if (!land || surface_le_zero(qs)) qs = EP2 * es / (psfc_cb - es);
    real es_air = SVP1 * sf_exp(SVP2 * (temp - SVPT0) / (temp - SVP3));
    real qgh = EP2 * es_air / (pl - es_air);
    // RHOX=PSFC*1000./(R*SCR4) with PSFC=PSFCPA/1000. (sf_sfclayrev.F90:221,
    // 300; module_sf_sfclay.F:396,475): the cb round trip is WRF's word.
    real rho = __fmul_rn(psfc_cb, 1000.0f) / (RD * tv);
    real za = 0.5f * dz8w[idx];
    real gz1, gz2, gz10;
    if (option == 91) {
        gz1 = sf_log_zratio(za, z0);
        gz2 = sf_log_zratio(2.0f, z0);
        gz10 = sf_log_zratio(10.0f, z0);
    } else {
        gz1 = sf_log_zratio(za + z0, z0);
        gz2 = sf_log_zratio(2.0f + z0, z0);
        gz10 = sf_log_zratio(10.0f + z0, z0);
    }

    real tskv = thgb * (1.0f + ep1 * qs);
    real dthv = thvx - tskv;
    real wspd0 = sqrtf(uu * uu + vv * vv), vconv;   // WRF's SQRT, not hypot
    if (land) {
        real fluxc = fmaxf(__fdiv_rn(old_hfx / rho, CP) + ep1 * tskv * old_qfx / rho,
                           0.0f);
        vconv = sf_pow(G / ground_t * pblh[idx] * fluxc, 0.33f);
    } else {
        vconv = sqrtf(fmaxf(-dthv, 0.0f));
    }
    real vsgd = 0.32f * sf_pow(fmaxf(__fdiv_rn(dx, 5000.0f) - 1.0f, 0.0f), 0.33f);
    real wspd = fmaxf(sqrtf(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd),
                       0.1f);
    real br = G / thx * za * dthv / (wspd * wspd);
    if (surface_lt_zero(old_mol)) br = fminf(br, 0.0f);

    real psim = 0.0f, psih = 0.0f, psim10 = 0.0f, psih10 = 0.0f;
    real psim2 = 0.0f, psih2 = 0.0f, pq = 0.0f, pq2 = 0.0f, pq10 = 0.0f;
    real zol = old_zol, regime, rmol;
    if (option == 91) {
        if (br >= 0.2f) {
            regime = 1.0f;
            psim = fmaxf(-10.0f * gz1, -10.0f); psih = psim;
            psim10 = fmaxf(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmaxf(2.0f / za * psim, -10.0f); psih2 = psim2;
            real za_over_l = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = fminf(za_over_l, 9.999f) / za;
        } else if (br > 0.0f) {
            regime = 2.0f;
            psim = fmaxf(-5.0f * br * gz1 / (1.1f - 5.0f * br), -10.0f);
            psih = psim;
            psim10 = fmaxf(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmaxf(2.0f / za * psim, -10.0f); psih2 = psim2;
            zol = br * gz1 / (1.00001f - 5.0f * br);
            if (zol > 0.5f) {
                zol = (1.89f * gz1 + 44.2f) * br * br
                    + (1.18f * gz1 - 1.37f) * br;
                zol = fminf(zol, 9.999f);
            }
            rmol = zol / za;
        } else if (br == 0.0f) {
            regime = 3.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = zol / za;
        } else {
            regime = 4.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            real zol10 = fmaxf(fminf(10.0f / za * zol, 0.0f), -9.9999f);
            real zol2 = fmaxf(fminf(2.0f / za * zol, 0.0f), -9.9999f);
            zol = fmaxf(fminf(zol, 0.0f), -9.9999f);
            psim = sf_classic_table(zol, false); psih = sf_classic_table(zol, true);
            psim10 = sf_classic_table(zol10, false); psih10 = sf_classic_table(zol10, true);
            psim2 = sf_classic_table(zol2, false); psih2 = sf_classic_table(zol2, true);
            psih = fminf(psih, 0.9f * gz1); psim = fminf(psim, 0.9f * gz1);
            psih2 = fminf(psih2, 0.9f * gz2);
            psim10 = fminf(psim10, 0.9f * gz10);
            psih10 = fminf(psih10, 0.9f * gz10);
            rmol = zol / za;
        }
    } else {
        zol = 0.0f;
        if (br > 0.0f) zol = sf_zolri(fminf(br, 250.0f), za, z0);
        else if (br < 0.0f)
            zol = old_ust < 0.001f ? br * gz1
                  : sf_zolri(fmaxf(br, -250.0f), za, z0);
        real zz = zol * (za + z0) / za, z10 = zol * (10.0f + z0) / za;
        real z2 = zol * (2.0f + z0) / za, zz0 = zol * z0 / za;
        // ZL=(0.01)/ZA*ZOL over land (sf_sfclayrev.F90:413), WRF's order.
        real scalar_z = land ? __fdiv_rn(0.01f, za) * zol : zz0;
        if (br > 0.0f) {
            regime = 1.0f;
            psim = sf_psim_stable(zz) - sf_psim_stable(zz0);
            psih = sf_psih_stable(zz) - sf_psih_stable(zz0);
            psim10 = sf_psim_stable(z10) - sf_psim_stable(zz0);
            psih10 = sf_psih_stable(z10) - sf_psih_stable(zz0);
            psim2 = sf_psim_stable(z2) - sf_psim_stable(zz0);
            psih2 = sf_psih_stable(z2) - sf_psih_stable(zz0);
            pq = sf_psih_stable(zol) - sf_psih_stable(scalar_z);
            pq2 = sf_psih_stable(2.0f / za * zol) - sf_psih_stable(scalar_z);
            pq10 = sf_psih_stable(10.0f / za * zol) - sf_psih_stable(scalar_z);
        } else if (br == 0.0f) {
            regime = 3.0f; zol = 0.0f;
        } else {
            regime = 4.0f;
            psim = sf_psim_unstable(zz) - sf_psim_unstable(zz0);
            psih = sf_psih_unstable(zz) - sf_psih_unstable(zz0);
            psim10 = sf_psim_unstable(z10) - sf_psim_unstable(zz0);
            psih10 = sf_psih_unstable(z10) - sf_psih_unstable(zz0);
            psim2 = sf_psim_unstable(z2) - sf_psim_unstable(zz0);
            psih2 = sf_psih_unstable(z2) - sf_psih_unstable(zz0);
            pq = sf_psih_unstable(zol) - sf_psih_unstable(scalar_z);
            pq2 = sf_psih_unstable(2.0f / za * zol) - sf_psih_unstable(scalar_z);
            pq10 = sf_psih_unstable(10.0f / za * zol) - sf_psih_unstable(scalar_z);
            psih = fminf(psih, 0.9f * gz1); psim = fminf(psim, 0.9f * gz1);
            psih2 = fminf(psih2, 0.9f * gz2);
            psim10 = fminf(psim10, 0.9f * gz10);
            psih10 = fminf(psih10, 0.9f * gz10);
        }
        rmol = zol / za;
    }

    real dtg = thx - thgb;
    real psix = gz1 - psim, psix10 = gz10 - psim10;
    real psit = option == 91 ? fmaxf(gz1 - psih, 2.0f) : gz1 - psih;
    real psit2 = gz2 - psih2;
    real zl = land ? 0.01f : z0;
    real psiq = sf_log(__fdiv_rn(karman * old_ust * za, xka) + __fdiv_rn(za, zl))
              - (option == 91 ? psih : pq);
    real psiq2 = sf_log(__fdiv_rn(karman * old_ust * 2.0f, xka) + __fdiv_rn(2.0f, zl))
               - (option == 91 ? psih2 : pq2);
    real psiq10 = sf_log(__fdiv_rn(karman * old_ust * 10.0f, xka) + __fdiv_rn(10.0f, zl))
                - (option == 91 ? psih10 : pq10);

    if (!land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        // Z0T=MIN(Z0T,1.0E-4) then MAX(Z0T,2.0E-9), WRF's order.
        real z0t = fmaxf(fminf(5.5e-5f * sf_pow(restar, -0.60f), 1.0e-4f),
                         2.0e-9f);
        if (option == 91) {
            psiq = fmaxf(sf_log((za + z0t) / z0t) - psih, 2.0f);
            psit = fmaxf(sf_log((za + z0t) / z0t) - psih, 2.0f);
            psiq2 = fmaxf(sf_log((2.0f + z0t) / z0t) - psih2, 2.0f);
            psit2 = fmaxf(sf_log((2.0f + z0t) / z0t) - psih2, 2.0f);
            psiq10 = fmaxf(sf_log((10.0f + z0t) / z0t) - psih10, 2.0f);
        } else {
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = sf_log((za + z0t) / z0t) - psih;
            psit2 = sf_log((2.0f + z0t) / z0t) - psih2;
            psiq = psit; psiq2 = psit2;
            psiq10 = sf_log((10.0f + z0t) / z0t) - psih10;
        }
    }

    if (isftcflx == 1 && !land) {
        real z0q = 1.0e-4f;
        if (option == 91) {
            psiq = sf_log(__fdiv_rn(za, z0q)) - psih;
            psiq2 = sf_log(2.0f / z0q) - psih2;
            psiq10 = sf_log(10.0f / z0q) - psih10;
        } else {
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_log(__fdiv_rn((za + z0q), z0q)) - psih;
            psiq2 = sf_log((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_log((10.0f + z0q) / z0q) - psih10;
        }
        psit = psiq; psit2 = psiq2;
    } else if (isftcflx == 2 && !land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        // 0.40*(7.3*SQRT(SQRT(RESTAR))*SQRT(0.71)-5.): two square roots,
        // not a quarter power (sf_sfclayrev.F90:636,668;
        // module_sf_sfclay.F:761-762).
        real gz0t = 0.4f * (7.3f * sqrtf(sqrtf(restar)) * SF_SQRT_0P71 - 5.0f);
        real gz0q = 0.4f * (7.3f * sqrtf(sqrtf(restar)) * SF_SQRT_0P60 - 5.0f);
        if (option == 91) {
            psit = gz1 - psih + gz0t; psiq = gz1 - psih + gz0q;
            psit2 = gz2 - psih2 + gz0t; psiq2 = gz2 - psih2 + gz0q;
            psiq10 = gz10 - psih + gz0q;
        } else {
            real z0t = z0 / sf_exp(gz0t), z0q = z0 / sf_exp(gz0q);
            real pht = sf_rev_heat_psi(zol, za, z0t, za);
            real pht2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psit = sf_log((za + z0t) / z0t) - pht;
            psit2 = sf_log((2.0f + z0t) / z0t) - pht2;
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_log((za + z0q) / z0q) - psih;
            psiq2 = sf_log((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_log((10.0f + z0q) / z0q) - psih10;
        }
    }

    real ck = (karman / psix10) * (karman / psiq10);
    real cd = (karman / psix10) * (karman / psix10);
    real cka = (karman / psix) * (karman / psiq);
    real cda = (karman / psix) * (karman / psix);
    bool scalar_exp_overflow = false;
    if (iz0tlnd >= 1 && land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        // CZIL=10.0**(-0.40*(ZL/0.07)): the quotient first
        // (sf_sfclayrev.F90:716; module_sf_sfclay.F:786).
        real czil = iz0tlnd == 1
                  ? sf_pow(10.0f, -(__fdiv_rn(z0, 0.07f) * 0.40f)) : 0.1f;
        if (option == 91) {
            real add = czil * karman * sqrtf(restar);
            psit = psiq = gz1 - psih + add;
            psit2 = psiq2 = gz2 - psih2 + add;
        } else {
            real scalar_exponent = czil * karman * sqrtf(restar);
            // Revised MM5 land scalar roughness overflow defect:
            // sf_sfclayrev.F90:723 overflows EXP, makes Z0T zero and then
            // FH=Inf and TH2/T2/Q2=NaN.  This is the last FP32 argument
            // whose exponential is finite.  Keep WRF's zero-exchange limit
            // beyond it, but diagnose the scalar resistance in log space.
            // The ordinary arm keeps its exact expression and FP32 word.
            scalar_exp_overflow = scalar_exponent > 88.72283172607421875f;
            real z0t = scalar_exp_overflow ? 0.0f : z0 / sf_exp(scalar_exponent);
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = psiq = scalar_exp_overflow
                ? sf_log_zratio(za, z0) + scalar_exponent - psih
                : sf_log((za + z0t) / z0t) - psih;
            psit2 = psiq2 = scalar_exp_overflow
                ? sf_log_zratio(2.0f, z0) + scalar_exponent - psih2
                : sf_log((2.0f + z0t) / z0t) - psih2;
        }
    }

    real new_ust = 0.5f * old_ust + 0.5f * karman * wspd / psix;
    // TKE coupling (module_sf_sfclay.F:800-804,
    // physics_mmm/sf_sfclayrev.F90:759-763): USTM repeats the UST
    // relaxation on the wind speed WITHOUT the Beljaars/Mahrt-Sun
    // vconv/vsgd correction and without WSPD's 0.1 floor, and takes
    // none of the land floor UST gets below.
    real wspdi = sqrtf(uu * uu + vv * vv);
    real new_ustm = 0.5f * old_ustm + 0.5f * karman * wspdi / psix;
    real u10 = uu * psix10 / psix, v10 = vv * psix10 / psix;
    real th2 = thgb + (thx - thgb) * psit2 / psit;
    real q2 = qs + (qvx - qs) * psiq2 / psiq;
    real t2 = th2 * sf_pow(__fdiv_rn(ps, P0), RCP);
    if (land) new_ust = fmaxf(new_ust, option == 91 ? 0.1f : 0.001f);
    real mol_numerator = karman * (thx - thgb);
    real new_mol = scalar_exp_overflow ? copysignf(0.0f, mol_numerator)
                                       : mol_numerator / psit;

    real z0out = z0;
    if (isfflx && !land) {
        z0out = fminf(__fdiv_rn(0.0185f * new_ust * new_ust, G)
                      + 0.11f * 1.5e-5f / new_ust, 2.85e-3f);
        if (isftcflx != 0) {
            real zw = fminf(sf_pow(__fdiv_rn(new_ust, 1.06f), 0.3f), 1.0f);
            real zn1 = __fdiv_rn(0.011f * new_ust * new_ust, G) + 1.59e-5f;
            real zn2 = 10.0f * sf_exp(-9.5f * sf_pow(new_ust, -0.3333f))
                       + 0.11f * 1.5e-5f / fmaxf(new_ust, 0.01f);
            z0out = fminf(fmaxf((1.0f - zw) * zn1 + zw * zn2, 1.27e-7f),
                           2.85e-3f);
        }
    }

    real flhc = 0.0f, flqc = 0.0f, new_hfx = 0.0f, new_qfx = 0.0f;
    // Flux-off carry defect: WRF skips the CHS/CHS2/CQS2 assignments
    // (sf_sfclayrev.F90:794,893,902,903), retaining the caller's buffers.
    // LH is unassigned in WRF on this path; WOOF defines it as zero.
    real lh = 0.0f, chs = isfflx ? 0.0f : chs_o[idx];
    real chs2 = isfflx ? 0.0f : chs2_o[idx];
    real cqs2 = isfflx ? 0.0f : cqs2_o[idx];
    if (isfflx) {
        real flqc_numerator = rho * mavail[idx] * new_ust * karman;
        flqc = scalar_exp_overflow ? copysignf(0.0f, flqc_numerator)
                                  : flqc_numerator / psiq;
        if (fabsf(thx - thgb) > 1.0e-5f)
            flhc = cpm * rho * new_ust * new_mol / (thx - thgb);
        new_qfx = flqc * (qs - qvx); lh = XLV * new_qfx;
        new_hfx = flhc * (thgb - thx);
        real chs_numerator = new_ust * karman;
        chs = scalar_exp_overflow ? copysignf(0.0f, chs_numerator) : chs_numerator / psiq;
        cqs2 = scalar_exp_overflow ? copysignf(0.0f, chs_numerator) : chs_numerator / psiq2;
        chs2 = scalar_exp_overflow ? copysignf(0.0f, chs_numerator) : chs_numerator / psit2;
    }

    znt[idx] = z0out; ust[idx] = new_ust; ustm[idx] = new_ustm;
    mol[idx] = new_mol;
    hfx[idx] = new_hfx; qfx[idx] = new_qfx; qsfc[idx] = qs;
    zol_o[idx] = zol; regime_o[idx] = regime; psim_o[idx] = psim;
    psih_o[idx] = psih; fm_o[idx] = psix; fh_o[idx] = psit; lh_o[idx] = lh;
    u10_o[idx] = u10; v10_o[idx] = v10; th2_o[idx] = th2; t2_o[idx] = t2;
    q2_o[idx] = q2;
    // SFCLAYREV leaves the incoming exchange coefficients alone when
    // ISFFLX is zero. Clearing them erased the previous surface call.
    if (isfflx) {
        chs_o[idx] = chs; chs2_o[idx] = chs2; cqs2_o[idx] = cqs2;
    }
    flhc_o[idx] = flhc; flqc_o[idx] = flqc;
    qgh_o[idx] = qgh; rmol_o[idx] = rmol; wspd_o[idx] = wspd;
    br_o[idx] = br; gz1_o[idx] = gz1; cpm_o[idx] = cpm;
    ck_o[idx] = ck; cka_o[idx] = cka; cd_o[idx] = cd; cda_o[idx] = cda;
}
