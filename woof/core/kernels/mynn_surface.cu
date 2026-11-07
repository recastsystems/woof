// WRF v4.6.1 MYNN surface layer (sf_sfclay_physics=5).
// With MYNN_SFCLAY_GSL_WRF39 defined (mynn_sfclay_variant = "gsl_wrf39") the
// same kernel compiles the GSL WRF 3.9 fork's form instead: its z/L search,
// caps, Richardson clamp, heat log terms and psih lower limit.
//
// This is the CUDA transcription of module_sf_mynn.F:SFCLAY1D_mynn for the
// pinned option identities: every defined isftcflx over water, with
// iz0tlnd=0, spp_pbl=0 and psi_opt=0.  One thread handles one independent
// horizontal column.  The MYNN surface-layer runtime selector remains
// fail-closed until the PBL kernel and full driver coupling pass their
// separate WRF oracles.

// WRF arithmetic, word for word (lane/verify-mynn-sfclay-wrf461).  Graded at
// 0 ULP on every output against SFCLAY1D_mynn by
// tools/mynn_sfclay_wrf461_column_oracle.  Three rules hold the unit there:
//   * every EXP/LOG/ATAN/REAL**REAL is WOOF's own float32 libm word
//     (mynn_libm.cuh), never CUDA's expf/logf/atanf/powf;
//   * the unit compiles with --fmad=false (gpuwm/core/kernels/__init__.py
//     _NO_FMAD_MODULES) and every expression keeps gfortran's association:
//     X**2 is X*X, A*B**2 is A*(B*B), and a constant subexpression gfortran
//     folds (ATAN(1.), LOG(10./1e-4), e**2.) is the folded word;
//   * no fused multiply-add anywhere, including the table interpolation.
#define SL_LOG(x) mynn_glibc_logf(x)
#define SL_EXP(x) mynn_glibc_expf(x)
#define SL_POW(x, y) mynn_glibc_powf((x), (y))
#define SL_ATAN(x) mynn_glibc_atanf(x)
// gfortran folds ATAN(1.) at compile time to the correctly rounded pi/4.
#define SL_ATAN1 __uint_as_float(0x3F490FDBu)
//
// COMPILE TIME.  The four psi forms, the table lookup, Li et al. (2010) and
// the three z/L solvers (zolri, zolri2, zolrib) are __noinline__.  Inlined,
// every one of the table's call sites expands two psi forms of several libm
// routines each, inside solver loops: on the reference host under load NVRTC took 33-40 s
// for the wrf_461 form and 12-13 minutes for gsl_wrf39 (3-5 s before
// the libm header).  With these call boundaries both forms take about 1.5 s.
// The unit compiles with --fmad=false, so a call boundary cannot change a
// rounding; the column oracle grades the words either way.

// module_sf_mynn.F:2088-2130, psi_opt=0.  1./2.5 and 1./1.1 are the folded
// quotients; sqrt(3.) is correctly rounded on both sides.
__device__ __noinline__ real mynn_psim_stable_full(real z)
{
    return -6.1f * SL_LOG(z + SL_POW(1.0f + SL_POW(z, 2.5f),
                                     __fdiv_rn(1.0f, 2.5f)));
}

__device__ __noinline__ real mynn_psih_stable_full(real z)
{
    return -5.3f * SL_LOG(z + SL_POW(1.0f + SL_POW(z, 1.1f),
                                     __fdiv_rn(1.0f, 1.1f)));
}

__device__ __noinline__ real mynn_psim_unstable_full(real z)
{
    real x = SL_POW(1.0f - 16.0f * z, 0.25f);
    real psimk = 2.0f * SL_LOG(0.5f * (1.0f + x))
               + SL_LOG(0.5f * (1.0f + x * x))
               - 2.0f * SL_ATAN(x) + 2.0f * SL_ATAN1;
    real ym = SL_POW(1.0f - 10.0f * z, 0.33f);
    real rt3 = sqrtf(3.0f);
    real psimc = 1.5f * SL_LOG(__fdiv_rn((ym * ym + ym + 1.0f), 3.0f))
               - rt3 * SL_ATAN(__fdiv_rn((2.0f * ym + 1.0f), rt3))
               + __fdiv_rn(4.0f * SL_ATAN1, rt3);
    return (psimk + (z * z) * psimc) / (1.0f + z * z);
}

// (1.-16.*zolf)**.5 is a powf call in WRF (gfortran keeps signed zeros, so
// it does not become SQRT), and powf is not sqrtf.
__device__ __noinline__ real mynn_psih_unstable_full(real z)
{
    real y = SL_POW(1.0f - 16.0f * z, 0.5f);
    real psihk = 2.0f * SL_LOG(__fdiv_rn((1.0f + y), 2.0f));
    real yh = SL_POW(1.0f - 34.0f * z, 0.33f);
    real rt3 = sqrtf(3.0f);
    real psihc = 1.5f * SL_LOG(__fdiv_rn((yh * yh + yh + 1.0f), 3.0f))
               - rt3 * SL_ATAN(__fdiv_rn((2.0f * yh + 1.0f), rt3))
               + __fdiv_rn(4.0f * SL_ATAN1, rt3);
    return (psihk + (z * z) * psihc) / (1.0f + z * z);
}

__device__ __noinline__ real mynn_table(real z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable.
    // module_sf_mynn.F:2197-2271 on the psi_init tables (:2051-2084), whose
    // entries are the full functions at zolf = +-float(n)*0.01, evaluated
    // here on demand.  nzol = INT(zolf*100.), rzol = zolf*100. - nzol, and
    // tab(n) + rzol*(tab(n+1)-tab(n)) is a rounded product and a rounded
    // sum: gfortran on x86-64 has no FMA, so neither may this.  (Before
    // lane/verify-mynn-sfclay-wrf461 the interpolation was one __fmaf_rn,
    // pinned for sm_89/sm_120 identity; the unfused order is just as pinned
    // and is the oracle's.)
    bool unstable = which >= 2;
    real scaled = __fmul_rn(unstable ? -z : z, 100.0f);
    int index = (int)scaled;
    if (index + 1 > 1000) {
        if (which == 0) return mynn_psim_stable_full(z);
        if (which == 1) return mynn_psih_stable_full(z);
        if (which == 2) return mynn_psim_unstable_full(z);
        return mynn_psih_unstable_full(z);
    }
    real fraction = __fsub_rn(scaled, (real)index);
    real sign = unstable ? -1.0f : 1.0f;
    real z0 = sign * __fmul_rn((real)index, 0.01f);
    real z1 = sign * __fmul_rn((real)(index + 1), 0.01f);
    real f0, f1;
    if (which == 0) {
        f0 = mynn_psim_stable_full(z0); f1 = mynn_psim_stable_full(z1);
    } else if (which == 1) {
        f0 = mynn_psih_stable_full(z0); f1 = mynn_psih_stable_full(z1);
    } else if (which == 2) {
        f0 = mynn_psim_unstable_full(z0); f1 = mynn_psim_unstable_full(z1);
    } else {
        f0 = mynn_psih_unstable_full(z0); f1 = mynn_psih_unstable_full(z1);
    }
    return __fadd_rn(f0, __fmul_rn(fraction, __fsub_rn(f1, f0)));
}

__device__ __forceinline__ real mynn_psim_stable(real z)
{ return mynn_table(z, 0); }
__device__ __forceinline__ real mynn_psih_stable(real z)
{ return mynn_table(z, 1); }
__device__ __forceinline__ real mynn_psim_unstable(real z)
{ return mynn_table(z, 2); }
__device__ __forceinline__ real mynn_psih_unstable(real z)
{ return mynn_table(z, 3); }

__device__ __noinline__ real mynn_li_etal_2010(
    real rib, real zaz0, real z0zt)
{
    real zaz02 = fminf(fmaxf(zaz0, 100.0f), 100000.0f);
    real z0zt2 = fminf(fmaxf(z0zt, 0.5f), 100.0f);
    // module_sf_mynn.F:1860-1885.  Rib**2, alfa**2 and beta**2 are integer
    // powers, a rounded square that the coefficient then multiplies.
    real alfa = SL_LOG(zaz02), beta = SL_LOG(z0zt2), zl;
    if (rib <= 0.0f) {
        zl = 0.045f * alfa * (rib * rib)
           + ((0.003f * beta + 0.0059f) * (alfa * alfa)
              + (-0.0828f * beta + 0.8845f) * alfa
              + (0.1739f * (beta * beta) - 0.9213f * beta - 0.1057f)) * rib;
        return fminf(fmaxf(zl, -15.0f), 0.0f);
    }
    if (rib <= 0.2f) {
        zl = ((0.5738f * beta - 0.4399f) * alfa
              + (-4.901f * beta + 52.50f)) * (rib * rib)
           + ((-0.0539f * beta + 1.540f) * alfa
              + (-0.669f * beta - 3.282f)) * rib;
        return fminf(fmaxf(zl, 0.0f), 4.0f);
    }
    zl = (0.7529f * alfa + 14.94f) * rib
       + 0.1569f * alfa - 0.3091f * beta - 1.303f;
    return fminf(fmaxf(zl, 1.0f), 20.0f);
}

__device__ __noinline__ real mynn_zolrib(
    real ri, real za, real z0, real zt, real logz0, real logzt, real zol1)
{
    if (zol1 * ri < 0.0f) zol1 = 0.0f;
    real zolold = ri < 0.0f ? -99999.0f : 99999.0f;
    real result = ri < 0.0f ? -66666.0f : 66666.0f;
    int iteration = 1;
    while (fabsf(zolold - result) > 0.01f && iteration < 20) {
        zolold = iteration == 1 ? zol1 : result;
        real zol20 = zolold * z0 / za;
        real zol3 = zolold + zol20;
        real zolt = zolold * zt / za;
        real psit2, psix2;
        if (ri < 0.0f) {
            psit2 = fmaxf(logzt
                - (mynn_psih_unstable(zol3) - mynn_psih_unstable(zolt)), 1.0f);
            psix2 = fmaxf(logz0
                - (mynn_psim_unstable(zol3) - mynn_psim_unstable(zol20)), 1.0f);
        } else {
            psit2 = fmaxf(logzt
                - (mynn_psih_stable(zol3) - mynn_psih_stable(zolt)), 1.0f);
            psix2 = fmaxf(logz0
                - (mynn_psim_stable(zol3) - mynn_psim_stable(zol20)), 1.0f);
        }
        result = ri * (psix2 * psix2) / psit2;   // :2031 ri*psix2**2/psit2
        ++iteration;
    }
    if (iteration == 20 && fabsf(zolold - result) > 0.01f)
        result = mynn_li_etal_2010(ri, za / z0, z0 / zt);
    return result;
}

#ifdef MYNN_SFCLAY_GSL_WRF39
// mynn_sfclay_variant = "gsl_wrf39": the MYNN surface layer of the GSL WRF
// 3.9 fork (NOAA-EMC/HRRR v4.1.21 sorc/hrrr_wrfarw.fd/WRFV3.9/phys/
// module_sf_mynn.F).  The loader defines MYNN_SFCLAY_GSL_WRF39 only for that
// variant, so the default ("wrf_461") translation unit is the WRF v4.6.1
// transcription above, unchanged.
//
// fork :1954-1993 zolri2.  ZOL2 is INTENT(INOUT): an iterate of the wrong
// sign is reset to zero here and the caller sees the reset value.  No floors
// on either log term, and the heat term's lower argument is z0/L.  The
// operations are pinned unfused so the 0.01 bracket test in zolri sees the
// rounding gfortran gives the fork source.
__device__ __noinline__ real mynn_zolri2(
    real &zol2, real ri2, real za, real z0, real zt)
{
    if (zol2 * ri2 < 0.0f) zol2 = 0.0f;
    // A non-finite secant iterate must not reach the table's integer index.
    if (!isfinite(zol2)) return __int_as_float(0x7fc00000);
    real zol20 = __fmul_rn(zol2, z0) / za;
    real zol3 = __fadd_rn(zol2, zol20);
    if (!isfinite(zol20) || !isfinite(zol3)) return __int_as_float(0x7fc00000);
    real psix2, psit2;
    if (ri2 < 0.0f) {
        psix2 = __fsub_rn(SL_LOG((za + z0) / z0),
            __fsub_rn(mynn_psim_unstable(zol3), mynn_psim_unstable(zol20)));
        psit2 = __fsub_rn(SL_LOG((za + zt) / zt),
            __fsub_rn(mynn_psih_unstable(zol3), mynn_psih_unstable(zol20)));
    } else {
        psix2 = __fsub_rn(SL_LOG((za + z0) / z0),
            __fsub_rn(mynn_psim_stable(zol3), mynn_psim_stable(zol20)));
        psit2 = __fsub_rn(SL_LOG((za + zt) / zt),
            __fsub_rn(mynn_psih_stable(zol3), mynn_psih_stable(zol20)));
    }
    return __fsub_rn(__fmul_rn(zol2, psit2) / __fmul_rn(psix2, psix2), ri2);
}

// fork :1899-1952 zolri.  A secant search over a bracket that starts at
// [first guess - 0.02, 0] (unstable) or [0, first guess + 0.02] (stable): at
// most 5 passes, each replacing the end point with the larger residual, and
// stopping when the end points are 0.01 apart or less.  After 5 passes with
// the end points still 0.01 or more apart it gives up and returns 5*Ri
// (unstable) or 8*Ri (stable), the fork's own values (:1938-1946).
//
// Undefined in the fork, defined here: a secant step whose denominator
// fx2 - fx1 is exactly zero divides by zero, and a residual that is not
// finite feeds Inf or NaN into the next table lookup (an out-of-range
// integer conversion).  Both take the give-up value at once.  The fork's
// result variable is unset when the loop body never runs; with a finite
// first guess the bracket starts at least 0.02 wide, so the body always
// runs, and the give-up value is the defined initial value regardless.
__device__ __noinline__ real mynn_zolri(
    real ri, real za, real z0, real zt, real zol1)
{
    real giveup = ri < 0.0f ? __fmul_rn(ri, 5.0f) : __fmul_rn(ri, 8.0f);
    real x1, x2;
    if (ri < 0.0f) { x1 = __fsub_rn(zol1, 0.02f); x2 = 0.0f; }
    else           { x1 = 0.0f; x2 = __fadd_rn(zol1, 0.02f); }
    real result = giveup;
    real fx1 = mynn_zolri2(x1, ri, za, z0, zt);
    real fx2 = mynn_zolri2(x2, ri, za, z0, zt);
    if (!isfinite(fx1) || !isfinite(fx2)) return giveup;
    int n = 0;
    while (fabsf(__fsub_rn(x1, x2)) > 0.01f && n < 5) {
        real denom = __fsub_rn(fx2, fx1);
        if (denom == 0.0f) return giveup;
        if (fabsf(fx2) < fabsf(fx1)) {
            x1 = __fsub_rn(x1, __fmul_rn(fx1 / denom, __fsub_rn(x2, x1)));
            fx1 = mynn_zolri2(x1, ri, za, z0, zt);
            result = x1;
            if (!isfinite(fx1)) return giveup;
        } else {
            x2 = __fsub_rn(x2, __fmul_rn(fx2 / denom, __fsub_rn(x2, x1)));
            fx2 = mynn_zolri2(x2, ri, za, z0, zt);
            result = x2;
            if (!isfinite(fx2)) return giveup;
        }
        ++n;
    }
    if (n == 5 && fabsf(__fsub_rn(x1, x2)) >= 0.01f) result = giveup;
    return result;
}
#endif

__device__ __forceinline__ void mynn_zilitinkevich_land(
    real z0, real restar, real &zt, real &zq)
{
    zt = z0 * SL_EXP(-0.4f * 0.085f * sqrtf(restar));
    zt = fminf(zt, 0.75f * z0);
    zq = zt;
}

__device__ __forceinline__ real mynn_charnock_1955(
    real ustar, real wsp, real visc, real zu)
{
    // :1351.  LOG(10./1e-4) is a constant gfortran folds: the correctly
    // rounded log of 100000., 0x413834F1.
    real wsp10 = wsp * __uint_as_float(0x413834F1u)
               / SL_LOG(__fdiv_rn(zu, 1.0e-4f));
    real czc = 0.011f + 0.007f
        * fminf(fmaxf((wsp10 - 10.0f) / 8.0f, 0.0f), 1.0f);
    real z0 = __fdiv_rn(czc * ustar * ustar, 9.81f)
            + 0.11f * visc / fmaxf(ustar, 0.05f);
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1281.  ISFTCFLX=1 and 2.  A function of u* alone: the
// Fortran takes no wind speed.
__device__ __forceinline__ real mynn_davis_etal_2008(real ustar)
{
    real zw = fminf(SL_POW(__fdiv_rn(ustar, 1.06f), 0.3f), 1.0f);
    real zn1 = __fdiv_rn(0.011f * ustar * ustar, 9.81f) + 1.59e-5f;
    real zn2 = 10.0f * SL_EXP(-9.5f * SL_POW(ustar, -0.3333f))
             + 0.11f * 1.5e-5f / fmaxf(ustar, 0.01f);
    real z0 = (1.0f - zw) * zn1 + zw * zn2;
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1311.  ISFTCFLX=3.  ustar is an argument of the Fortran
// subroutine and is never read in its body, so it is not taken here either.
__device__ __forceinline__ real mynn_taylor_yelland_2001(real wsp)
{
    real hs = 0.0248f * (wsp * wsp);   // :1323 wsp10**2.
    real tp = 0.729f * fmaxf(wsp, 0.1f);
    real lp = __fdiv_rn(9.81f * (tp * tp), (2.0f * 3.14159265f));
    real z0 = 1200.0f * hs * SL_POW(hs / lp, 4.5f);
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1425.  COARE 3.0 zt/zq.  The Ren<=2 test at :1442 selects
// between two arms that compute the identical expression.  Zq is assigned
// from the already-clamped Zt (:1466-1467), so zq == zt.
__device__ __forceinline__ void mynn_fairall_etal_2003(
    real restar, real &zt, real &zq)
{
    zt = 5.5e-5f * SL_POW(restar, -0.60f);
    zt = fminf(fmaxf(zt, 2.0e-9f), 1.0e-4f);
    zq = zt;
}

// module_sf_mynn.F:1392.  ISFTCFLX=2 zt/zq.  This leaf makes its own
// landsea-1.5 .GT. 0 test, which is not the .GE. 0 at :625 that selected the
// water roughness, so a column at exactly XLAND=1.5 arrives here and takes
// the LAND arm.
__device__ __forceinline__ void mynn_garratt_1992(
    real z0, real restar, real xland, real &zt, real &zq)
{
    if (xland - 1.5f > 0.0f) {
        real quarter = SL_POW(restar, 0.25f);
        zt = z0 * SL_EXP(2.0f - 2.48f * quarter);
        zq = z0 * SL_EXP(2.0f - 2.28f * quarter);
        zq = fmaxf(fminf(zq, 5.5e-5f), 2.0e-9f);
        zt = fmaxf(fminf(zt, 5.5e-5f), 2.0e-9f);
    } else {
        // :1417 e**2. with e a PARAMETER: gfortran folds it to 0x40EC7325.
        zq = __fdiv_rn(z0, __uint_as_float(0x40EC7325u));
        zt = zq;
    }
}

// module_sf_mynn.F:631-710.  ISFTCFLX picks the over-water z0 and then, from
// the restar that new z0 implies (:675), the zt/zq pair.  COARE_OPT is a REAL
// PARAMETER fixed at 3.0 (:85), so edson_etal_2013 and fairall_etal_2014 are
// unreachable through SFCLAY1D_mynn and are deliberately absent from this
// kernel: they exist in the CPU reference only because the leaf oracle can
// call them at their own entry points.  ISFTCFLX=4 sets a z0 and then leaves
// z_t/z_q unassigned (:680-702 has no arm for it), so the host rejects it
// before launch and this switch has no arm for it either.
__device__ __forceinline__ void mynn_water_roughness(
    int isftcflx, real ustar, real wsp, real visc, real za, real xland,
    real &z0, real &restar, real &zt, real &zq)
{
    if (isftcflx == 1 || isftcflx == 2) z0 = mynn_davis_etal_2008(ustar);
    else if (isftcflx == 3) z0 = mynn_taylor_yelland_2001(wsp);
    else z0 = mynn_charnock_1955(ustar, wsp, visc, za);
    restar = fmaxf(ustar * z0 / visc, 0.1f);
    if (isftcflx == 2) mynn_garratt_1992(z0, restar, xland, zt, zq);
    else mynn_fairall_etal_2003(restar, zt, zq);
}

__device__ __forceinline__ void mynn_andreas_snow(
    real visc, real ustar, real &zt, real &zq)
{
    // :1580-1582: 0.035*(ustar*ustar)/9.8 and the explicit product of the
    // two (ustar-0.18)/0.1 factors, not a power.
    real dev = __fdiv_rn((ustar - 0.18f), 0.1f);
    real zntsno = 0.135f * visc / ustar
        + __fdiv_rn(0.035f * (ustar * ustar), 9.8f)
        * (5.0f * SL_EXP(-(dev * dev)) + 1.0f);
    real ren = fminf(ustar * zntsno / visc, 1000.0f);
    real log_ren = SL_LOG(ren);
    real bt0, bt1, bt2, bq0, bq1, bq2;
    if (ren <= 0.135f) {
        bt0 = 1.25f; bt1 = 0.0f; bt2 = 0.0f;
        bq0 = 1.61f; bq1 = 0.0f; bq2 = 0.0f;
    } else if (ren < 2.5f) {
        bt0 = 0.149f; bt1 = -0.55f; bt2 = 0.0f;
        bq0 = 0.351f; bq1 = -0.628f; bq2 = 0.0f;
    } else {
        bt0 = 0.317f; bt1 = -0.565f; bt2 = -0.183f;
        bq0 = 0.396f; bq1 = -0.512f; bq2 = -0.180f;
    }
    // :1591-1602: LOG(Ren2)**2 is a rounded square.
    zt = zntsno * SL_EXP(bt0 + bt1 * log_ren + bt2 * (log_ren * log_ren));
    zq = zntsno * SL_EXP(bq0 + bq1 * log_ren + bq2 * (log_ren * log_ren));
}

extern "C" __global__
void mynn_surface_column(
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ t1_a, const real* __restrict__ qv1_a,
    const real* __restrict__ p1_a, const real* __restrict__ rho1_a,
    const real* __restrict__ dz1_a, const real* __restrict__ u2_a,
    const real* __restrict__ v2_a, const real* __restrict__ dz2_a,
    const real* __restrict__ psfc_a, const real* __restrict__ tsk_a,
    const real* __restrict__ pblh_a, const real* __restrict__ mavail_a,
    const real* __restrict__ hfx_a, const real* __restrict__ qfx_a,
    const real* __restrict__ znt_a, const real* __restrict__ qsfc_a,
    const real* __restrict__ ust_a, const real* __restrict__ xland_a,
    const real* __restrict__ snowh_a, const real* __restrict__ mol_a,
    const real* __restrict__ ustm_a,
    real* __restrict__ regime_o, real* __restrict__ zol_o,
    real* __restrict__ rmol_o, real* __restrict__ ust_o,
    real* __restrict__ ustm_o, real* __restrict__ mol_o,
    real* __restrict__ psim_o, real* __restrict__ psih_o,
    real* __restrict__ chs_o, real* __restrict__ chs2_o,
    real* __restrict__ cqs2_o, real* __restrict__ ch_o,
    real* __restrict__ flhc_o, real* __restrict__ flqc_o,
    real* __restrict__ qgh_o, real* __restrict__ qsfc_o,
    real* __restrict__ hfx_o, real* __restrict__ qfx_o,
    real* __restrict__ lh_o, real* __restrict__ u10_o,
    real* __restrict__ v10_o, real* __restrict__ th2_o,
    real* __restrict__ t2_o, real* __restrict__ q2_o,
    real* __restrict__ gz1oz0_o, real* __restrict__ wspd_o,
    real* __restrict__ br_o, real* __restrict__ ck_o,
    real* __restrict__ cka_o, real* __restrict__ cd_o,
    real* __restrict__ cda_o, real* __restrict__ wstar_o,
    real* __restrict__ qstar_o, real* __restrict__ cpm_o,
    real* __restrict__ znt_o,
#ifdef OCEAN_WAVE_ROUGHNESS
    const real* __restrict__ ocean_z0,
#endif
    real dx, int itimestep, int isfflx, int isftcflx, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const real cp = 1004.5f, grav = 9.81f, r = 287.0f, rv = 461.6f;
    const real rovcp = r / cp, xlv = 2.5e6f;
    const real svp1 = 0.6112f, svp2 = 17.67f, svp3 = 29.65f;
    const real svpt0 = 273.15f, ep1 = rv / r - 1.0f;
    const real ep2 = r / rv, ep3 = 1.0f - ep2;
    const real karman = 0.4f, prt = 1.0f, wmin = 0.1f, vconvc = 1.25f;

    real u1 = u1_a[idx], v1 = v1_a[idx], t1 = t1_a[idx];
    real qv1 = qv1_a[idx], p1 = p1_a[idx], rho1 = rho1_a[idx];
    real dz1 = dz1_a[idx], u2 = u2_a[idx], v2 = v2_a[idx];
    real dz2 = dz2_a[idx], psfcpa = psfc_a[idx], tsk = tsk_a[idx];
    real pblh = pblh_a[idx], mavail = mavail_a[idx];
    real hfx = hfx_a[idx], qfx = qfx_a[idx], z0 = znt_a[idx];
    real qsfc = qsfc_a[idx], ust = ust_a[idx], xland = xland_a[idx];
    real snowh = snowh_a[idx], mol = mol_a[idx], ustm = ustm_a[idx];

    real psfc = __fdiv_rn(psfcpa, 1000.0f);
    real thgb = tsk * SL_POW(100.0f / psfc, rovcp);
    real pl = __fdiv_rn(p1, 1000.0f);
    real th1 = t1 * SL_POW(100.0f / pl, rovcp);
    real tc1 = t1 - 273.15f;
    real qvsh = qv1 / (1.0f + qv1);
    // See the THVGB comment below: TVCON/THV1D are pinned to unfused
    // round-to-nearest so they contract identically to their surface twins.
    real tvcon = __fadd_rn(1.0f, __fmul_rn(ep1, qvsh));
    real thv1 = __fmul_rn(th1, tvcon);
    real za = 0.5f * dz1;
    real za2 = dz1 + 0.5f * dz2;
    real govrth = grav / th1;

    real e1;
    if (tsk < 273.15f) {
        e1 = svp1 * SL_EXP(4648.0f * (1.0f / 273.15f - 1.0f / tsk)
             - 11.64f * SL_LOG(273.15f / tsk) + 0.02265f * (273.15f - tsk));
    } else {
        e1 = svp1 * SL_EXP(svp2 * (tsk - svpt0) / (tsk - svp3));
    }
    // module_sf_mynn.F:532 tests the INCOMING QSFC: a land column whose LSM
    // never set QSFC is recomputed from saturation here.
    real qsfcmr;
    if (xland > 1.5f || surface_le_zero(qsfc)) {
        qsfc = ep2 * e1 / (psfc - ep3 * e1);
        qsfcmr = ep2 * e1 / (psfc - e1);
    } else {
        qsfcmr = qsfc / (1.0f - qsfc);
    }

    if (tsk < 273.15f) {
        e1 = svp1 * SL_EXP(4648.0f * (1.0f / 273.15f - 1.0f / t1)
             - 11.64f * SL_LOG(273.15f / t1) + 0.02265f * (273.15f - t1));
    } else {
        e1 = svp1 * SL_EXP(svp2 * (t1 - svpt0) / (t1 - svp3));
    }
    real qgh = ep2 * e1 / (pl - e1);
    real cpm = cp * (1.0f + 0.84f * qv1);

    // module_sf_mynn.F:556 -- WSPD is SQRT(u*u+v*v), not hypot().
    real wsp = sqrtf(u1 * u1 + v1 * v1);
    // THV1D (:513) and THVGB (:559) are the same product of a temperature and
    // (1+EP1*q).  gfortran evaluates both without contraction, so an exactly
    // saturated neutral column cancels to zero.  NVRTC is free to contract one
    // and not the other; pin both to unfused round-to-nearest so the identity
    // holds for the same reason it holds in WRF instead of by special case.
    real tvcon_gb = __fadd_rn(1.0f, __fmul_rn(ep1, qsfc));
    real thvgb = __fmul_rn(thgb, tvcon_gb);
    real dthvdz = thv1 - thvgb;
    real fluxc = fmaxf(__fdiv_rn(hfx / rho1, cp) + __fdiv_rn(ep1 * thvgb * qfx, rho1), 0.0f);
    // module_sf_mynn.F:573 spells the :532 predicate a second time, but it runs
    // after the :533 saturation update, so a land column that entered with
    // QSFC<=0 now takes the LAND height scale.
    bool wstar_water = xland > 1.5f || surface_le_zero(qsfc);
    real height = wstar_water ? pblh : fminf(1.5f * pblh, 4000.0f);
    real wstar = vconvc * SL_POW(grav / tsk * height * fluxc, 0.33f);
    real vsgd = 0.32f * SL_POW(fmaxf(__fdiv_rn(dx, 5000.0f) - 1.0f, 0.0f), 0.33f);
    wsp = fmaxf(sqrtf(wsp * wsp + wstar * wstar + vsgd * vsgd), wmin);
    real br = govrth * za * dthvdz / (wsp * wsp);
#ifdef MYNN_SFCLAY_GSL_WRF39
    real limit = itimestep == 1 ? 2.0f : 50.0f;   // fork :613-614
#else
    real limit = itimestep == 1 ? 2.0f : 4.0f;
#endif
    br = fminf(fmaxf(br, -limit), limit);

    real visc = 1.326e-5f * (1.0f + 6.542e-3f * tc1
              + 8.301e-6f * tc1 * tc1 - 4.84e-9f * tc1 * tc1 * tc1);
    real restar, zt, zq;
    if (xland >= 1.5f) {
        mynn_water_roughness(isftcflx, ust, wsp, visc, za, xland,
                             z0, restar, zt, zq);
    } else {
        restar = fmaxf(ust * z0 / visc, 0.1f);
        if (snowh >= 0.1f) mynn_andreas_snow(visc, ust, zt, zq);
        else mynn_zilitinkevich_land(z0, restar, zt, zq);
    }

#if defined(OCEAN_WAVE_ROUGHNESS) && !defined(OCEAN_SPP_SURFACE)
    // Use the wave-age momentum roughness before calculating exchange.
    if (xland >= 1.5f && ocean_z0[idx] > 0.0f) {
        z0 = ocean_z0[idx];
        restar = fmaxf(ust * z0 / visc, 0.1f);
        if (isftcflx == 2) mynn_garratt_1992(z0, restar, xland, zt, zq);
        else mynn_fairall_etal_2003(restar, zt, zq);
    }
#endif
    real zratio = z0 / zt;
    real gz1oz0 = SL_LOG((za + z0) / z0);
#ifdef MYNN_SFCLAY_GSL_WRF39
    // fork :771-775: zt, not z0, in the numerator of the heat log terms.
    real gz1ozt = SL_LOG((za + zt) / zt);
    real gz2ozt = SL_LOG((2.0f + zt) / zt);
    real gz10oz0 = SL_LOG((10.0f + z0) / z0);
    real gz10ozt = SL_LOG((10.0f + zt) / zt);
    // fork :813-819, :894-900: z/L capped at 50, not 20.
    const real zol_cap = 50.0f;
#else
    real gz1ozt = SL_LOG((za + z0) / zt);
    real gz2ozt = SL_LOG((2.0f + z0) / zt);
    real gz10oz0 = SL_LOG((10.0f + z0) / z0);
    real gz10ozt = SL_LOG((10.0f + z0) / zt);
    const real zol_cap = 20.0f;
#endif

    real regime, zol, psim, psih, psim10, psih10, psih2;
    if (br > 0.0f) {
        regime = br > 0.2f ? 1.0f : 2.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmaxf(ust * ust, 0.0001f));
            zol = fminf(fmaxf(zol, 0.0f), zol_cap);
        }
#ifdef MYNN_SFCLAY_GSL_WRF39
        zol = fminf(fmaxf(mynn_zolri(br, za, z0, zt, zol), 0.0f), zol_cap);
#else
        zol = fminf(fmaxf(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           0.0f), zol_cap);
#endif
        real zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        real zolza = zol * (za + z0) / za;
        real zol10 = zol * (10.0f + z0) / za;
        real zol2 = zol * (2.0f + z0) / za;
#ifdef MYNN_SFCLAY_GSL_WRF39
        zolzt = zolz0;   // fork :836, :849: psih's lower limit is z0/L
#endif
        psim = mynn_psim_stable(zolza) - mynn_psim_stable(zolz0);
        psih = mynn_psih_stable(zolza) - mynn_psih_stable(zolzt);
        psim10 = mynn_psim_stable(zol10) - mynn_psim_stable(zolz0);
        psih10 = mynn_psih_stable(zol10) - mynn_psih_stable(zolz0);
        psih2 = mynn_psih_stable(zol2) - mynn_psih_stable(zolz0);
    } else if (br == 0.0f) {
        regime = 3.0f; zol = 0.0f;
#ifdef MYNN_SFCLAY_GSL_WRF39
        // fork :875-880, from the incoming u* and MOL.
        if (ust < 0.01f) zol = br * gz1oz0;
        else zol = karman * govrth * za * mol / fmaxf(ust * ust, 0.001f);
#endif
        psim = 0.0f; psih = 0.0f; psim10 = 0.0f; psih10 = 0.0f; psih2 = 0.0f;
    } else {
        regime = 4.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmaxf(ust * ust, 0.001f));
            zol = fminf(fmaxf(zol, -zol_cap), 0.0f);
        }
#ifdef MYNN_SFCLAY_GSL_WRF39
        zol = fminf(fmaxf(mynn_zolri(br, za, z0, zt, zol), -zol_cap), 0.0f);
#else
        zol = fminf(fmaxf(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           -zol_cap), 0.0f);
#endif
        real zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        real zolza = zol * (za + z0) / za;
        real zol10 = zol * (10.0f + z0) / za;
        real zol2 = zol * (2.0f + z0) / za;
#ifdef MYNN_SFCLAY_GSL_WRF39
        zolzt = zolz0;   // fork :917, :928
#endif
        psim = mynn_psim_unstable(zolza) - mynn_psim_unstable(zolz0);
        psih = mynn_psih_unstable(zolza) - mynn_psih_unstable(zolzt);
        psim10 = mynn_psim_unstable(zol10) - mynn_psim_unstable(zolz0);
        psih10 = mynn_psih_unstable(zol10) - mynn_psih_unstable(zolz0);
        psih2 = mynn_psih_unstable(zol2) - mynn_psih_unstable(zolz0);
        psih = fminf(psih, 0.9f * gz1ozt);
        psim = fminf(psim, 0.9f * gz1oz0);
        psih2 = fminf(psih2, 0.9f * gz2ozt);
        psim10 = fminf(psim10, 0.9f * gz10oz0);
        psih10 = fminf(psih10, 0.9f * gz10ozt);
    }
    real rmol = zol / za;

    real psix = gz1oz0 - psim;
    real psix10 = gz10oz0 - psim10;
    ust = 0.5f * ust + 0.5f * karman * wsp / psix;
    // module_sf_mynn.F:953 -- WSPDI is SQRT(u*u+v*v), not hypot().
    real wspdi = fmaxf(sqrtf(u1 * u1 + v1 * v1), wmin);
    ustm = 0.5f * ustm + 0.5f * karman * wspdi / psix;
    if (xland < 1.5f) {
        ust = fmaxf(ust, 0.005f);
        ustm = ust;
    }

#ifdef MYNN_SFCLAY_GSL_WRF39
    // fork :978-986: zt and zq in the numerators, also for the qstar copy.
    gz1ozt = SL_LOG((za + zt) / zt);
    gz2ozt = SL_LOG((2.0f + zt) / zt);
    real psit = fmaxf(gz1ozt - psih, 1.0f);
    real psit2 = fmaxf(gz2ozt - psih2, 1.0f);
    real psiq = fmaxf(SL_LOG((za + zq) / zq) - psih, 1.0f);
    real psiq2 = fmaxf(SL_LOG((2.0f + zq) / zq) - psih2, 1.0f);
    real psiq10 = fmaxf(SL_LOG((10.0f + zq) / zq) - psih10, 1.0f);
#else
    gz1ozt = SL_LOG((za + z0) / zt);
    gz2ozt = SL_LOG((2.0f + z0) / zt);
    real psit = fmaxf(gz1ozt - psih, 1.0f);
    real psit2 = fmaxf(gz2ozt - psih2, 1.0f);
    real psiq = fmaxf(SL_LOG((za + z0) / zq) - psih, 1.0f);
    real psiq2 = fmaxf(SL_LOG((2.0f + z0) / zq) - psih2, 1.0f);
    real psiq10 = fmaxf(SL_LOG((10.0f + z0) / zq) - psih10, 1.0f);
#endif
    mol = karman * (thv1 - thvgb) / psit / prt;
    // :988-989: DQG = (QVSH-qsfc)*1000. first, then KARMAN*DQG/PSIQ/PRT.
    real qstar = karman * ((qvsh - qsfc) * 1000.0f) / psiq / prt;

    // WRF deliberately recomputes moisture resistance with zq in numerator.
    psiq = fmaxf(SL_LOG((za + zq) / zq) - psih, 1.0f);
    psiq2 = fmaxf(SL_LOG((2.0f + zq) / zq) - psih2, 1.0f);
    psiq10 = fmaxf(SL_LOG((10.0f + zq) / zq) - psih10, 1.0f);
    real flhc, flqc, lh, chs, ch, cqs2, chs2, ck, cka, cd, cda;
    if (isfflx < 1) {
        qfx = 0.0f; hfx = 0.0f; flhc = 0.0f; flqc = 0.0f; lh = 0.0f;
        chs = 0.0f; ch = 0.0f; chs2 = 0.0f; cqs2 = 0.0f;
        ck = 0.0f; cka = 0.0f; cd = 0.0f; cda = 0.0f;
    } else {
        flqc = rho1 * mavail * ust * karman / psiq;
        flhc = rho1 * cpm * ust * karman / psit;
        qfx = fmaxf(flqc * (qsfcmr - qv1), -0.02f);
        lh = xlv * qfx;
        // module_sf_mynn.F:1065-1076.  The arms test XLAND-1.5 for .GT. and
        // .LT. and nothing else, so a column at exactly XLAND=1.5 takes
        // neither and HFX keeps the value it entered with.
        if (xland > 1.5f) {
            hfx = flhc * (thgb - th1);
            // :1067-1072, the AHW dissipative-heating term -- the third and
            // last thing ISFTCFLX selects, water-only, and keyed on
            // ISFTCFLX.NE.0, so every non-default roughness identity adds it.
            // USTM is the u* computed without the vconv/vsgd gust (:953-956).
            if (isftcflx != 0) hfx = hfx + rho1 * ustm * ustm * wspdi;
        } else if (xland < 1.5f) {
            hfx = flhc * (thgb - th1);
            hfx = fmaxf(hfx, -250.0f);
        }
        chs = ust * karman / psit;
        ch = flhc / (cpm * rho1);
        cqs2 = ust * karman / psiq2;
        chs2 = ust * karman / psit2;
        ck = (karman / psix10) * (karman / psiq10);
        cd = (karman / psix10) * (karman / psix10);
        cka = (karman / psix) * (karman / psiq);
        cda = (karman / psix) * (karman / psix);
    }

    real u10, v10;
    if (za <= 7.0f) {
        if (za2 > 7.0f && za2 < 13.0f) { u10 = u2; v10 = v2; }
        else {
            // :1117: U1D*log(10./z0)/log(ZA/z0), left to right.
            real l10 = SL_LOG(10.0f / z0), lza = SL_LOG(za / z0);
            u10 = u1 * l10 / lza; v10 = v1 * l10 / lza;
        }
    } else if (za < 13.0f) {
        real l10 = SL_LOG(10.0f / z0), lza = SL_LOG(za / z0);
        u10 = u1 * l10 / lza; v10 = v1 * l10 / lza;
    } else {
        u10 = u1 * psix10 / psix; v10 = v1 * psix10 / psix;
    }

    real th2 = thgb + (th1 - thgb) * psit2 / psit;
    if ((th1 > thgb && !(th2 >= thgb && th2 <= th1))
        || (th1 < thgb && !(th2 >= th1 && th2 <= thgb)))
        th2 = thgb + 2.0f * (th1 - thgb) / za;
    real t2 = th2 * SL_POW(__fdiv_rn(psfc, 100.0f), rovcp);
    real q2 = qsfcmr + (qv1 - qsfcmr) * psiq2 / psiq;
    q2 = fmaxf(q2, fminf(qsfcmr, qv1));
    q2 = fminf(q2, 1.05f * qv1);

    regime_o[idx] = regime; zol_o[idx] = zol; rmol_o[idx] = rmol;
    ust_o[idx] = ust; ustm_o[idx] = ustm; mol_o[idx] = mol;
    psim_o[idx] = psim; psih_o[idx] = psih; chs_o[idx] = chs;
    chs2_o[idx] = chs2; cqs2_o[idx] = cqs2; ch_o[idx] = ch;
    flhc_o[idx] = flhc; flqc_o[idx] = flqc; qgh_o[idx] = qgh;
    qsfc_o[idx] = qsfc; hfx_o[idx] = hfx; qfx_o[idx] = qfx;
    lh_o[idx] = lh; u10_o[idx] = u10; v10_o[idx] = v10;
    th2_o[idx] = th2; t2_o[idx] = t2; q2_o[idx] = q2;
    gz1oz0_o[idx] = gz1oz0; wspd_o[idx] = wsp; br_o[idx] = br;
    ck_o[idx] = ck; cka_o[idx] = cka; cd_o[idx] = cd; cda_o[idx] = cda;
    wstar_o[idx] = wstar; qstar_o[idx] = qstar; cpm_o[idx] = cpm;
    // module_sf_mynn.F:436 declares ZNT INTENT(INOUT); :657 (charnock_1955)
    // mutates it in place over water and the updated value persists into the
    // next step's ZNTstoch/restar/z_t/z_q chain.
    znt_o[idx] = z0;
}

// WRF's wrapper seed uses the same unflushed arithmetic as the column.
extern "C" __global__ void mynn_surface_seed(
    const real* u, const real* v, const real* qv,
    real* ust, real* mol, real* qsfc, real* qstar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real speed = 0.04f * sqrtf(u[i] * u[i] + v[i] * v[i]);
    ust[i] = isnan(speed) ? speed : fmaxf(speed, 0.001f);
    mol[i] = 0.0f;
    qsfc[i] = qv[i] / (1.0f + qv[i]);
    qstar[i] = 0.0f;
}
