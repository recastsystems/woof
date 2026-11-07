// gpuwm/core/kernels/noah.cu
//
// Noah land surface model, transcribed line-faithfully from WRF v4.6.1
// phys/module_sf_noahdrv.F (subroutine lsm: per-column prep + post-SFLX
// updates) and phys/module_sf_noahlsm.F (SFLX and its full subtree:
// PENMAN, CANRES, NOPAC, SNOPAC, SMFLX/SRT/SSTEP, SHFLX/HRT/HSTEP,
// ROSR12, EVAPO/DEVAP/TRANSP, SNKSRC/FRH2O/TMPAVG/TBND, TDFCND/WDFCND,
// CSNOW/SNFRAC/ALCALC/SNOWPACK/SNOW_NEW/SNOWZ0, REDPRM from packed
// tables).  One thread per (i, j) column, 4 soil layers, FP32
// throughout with FP64 used only for the energy-residual diagnostics.
// Land-ice columns (ivgtyp == isice) run SFLX_GLACIAL
// (module_sf_noahlsm_glacial_only.F) in noah_glacial_column below.
// NOT ported (see gpuwm/core/noah.py): WRF-Hydro,
// SFCDIF_off.  The urban canopy
// models run AFTER this kernel (gpuwm/core/urban_driver.py); this kernel
// carries their hand-over only: the NATURAL remap of urban columns
// (module_sf_noahdrv.F:964-990), TSK_RURAL_BEP, and the rural Q1/Q2K/ZLVL,
// all behind urban_opt > 0.  The plain VEGTYP==ISURBAN overrides are ported.
// Float64 mirror: gpuwm/verify/npref.py np_noah_column.
//
// Constant discipline: module_sf_noahlsm's own parameters are the file
// literals below (NRD=287.04, NSIGMA=5.67e-8, PENMAN's CP=1004.6, ...);
// quantities from module_model_constants come from the launch-time
// defines (CP=1004.5, XLV, RCP, RHOWATER) or the literals NSTBOLT/NXLF.

#define NSOIL 4

#ifndef NOAH_WRF_SCALAR_RCT
#define NOAH_WRF_SCALAR_RCT 0
#endif

#define NRD      287.04f      // noahlsm module RD
#define NSIGMA   5.67e-8f     // noahlsm SIGMA
#define NCPH2O   4.218e+3f    // noahlsm CPH2O
#define NCPICE   2.106e+3f    // noahlsm CPICE
#define NLSUBF   3.335e+5f    // noahlsm LSUBF
#define NEMISSI_S 0.95f       // noahlsm EMISSI_S
#define NTFREEZ  273.15f      // SFLX TFREEZ
#define NLVH2O   2.501e+6f    // SFLX LVH2O
#define NLSUBS   2.83e+6f     // SFLX/PENMAN/SNOPAC LSUBS
#define NR_SHEAT 287.04f      // SFLX local R
#define NCP_PEN  1004.6f      // PENMAN's local CP
#define NELCP    2.4888e+3f   // PENMAN ELCP
#define NLSUBC   2.501e+6f    // PENMAN/SNOPAC LSUBC
#define NSTBOLT  5.67051e-8f  // module_model_constants STBOLT (NOAHRES)
#define NXLF     3.50e+5f     // module_model_constants XLF (SNOPCX)

// ------------------------------------------------- float32 libm words
// The ordinary-land column calls WOOF's own float32 expf/logf/powf
// (gfk_exp, gfk_log, gfk_pow of glibc_flt32.cuh), the same words the
// land-ice arm below uses, so its exponentials, logarithms and powers are
// the ones WRF's gfortran build gets from the C library rather than CUDA's.
//
// noah_log10f: log10(x) = k*log10(2) + log10(e)*ln(m) for x = m*2^k with m
// in [1,2) (or [0.5,1) when k is negative), with log10(2) split into a head
// and a tail so k*log10(2) carries no rounding of its own.  Every operation
// is spelled with an __f*_rn intrinsic so no contraction can fuse them.
// MEASURED exhaustively on x86-64 (glibc 2.39): over all
// 2,139,095,039 positive finite float32 inputs this formula on the host
// libm's logf gives the same word as the host's log10f; that log10f is NOT
// correctly rounded (29,787,059 inputs differ from log10 in double rounded
// once), so a correctly rounded log10f would not reproduce WRF here.
// tools/noah_wrf461_oracle/libm_exhaustive.py repeats the comparison for
// this device function against the host log10f.
__device__ __forceinline__ float noah_log10f(float x)
{
#if NOAH_WRF_MODERN_LOG10
    // glibc 2.43's log10f is correctly rounded. A binary64 log10 followed
    // by one FP32 rounding matches the captured real native columns;
    // the legacy decomposition below preserves the 2.39 referee words.
    return __double2float_rn(log10((double)x));
#else
    unsigned int hx = __float_as_uint(x);
    if ((hx & 0x7fffffffu) == 0u) return __uint_as_float(0xff800000u);
    if (hx >= 0x7f800000u) return (hx > 0x7f800000u || (hx >> 31))
                               ? __uint_as_float(0x7fc00000u) : x;
    int k = 0;
    if (hx < 0x00800000u) {               // subnormal: scale by 2^25
        x = __fmul_rn(x, 33554432.0f);
        hx = __float_as_uint(x);
        k -= 25;
    }
    k += (int)(hx >> 23) - 127;
    int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    float y = (float)(k + i);
    float z = __fadd_rn(__fmul_rn(y, __uint_as_float(0x355427dbu)),
                        __fmul_rn(__uint_as_float(0x3ede5bd9u),
                                  gfk_log(__uint_as_float(hx))));
    return __fadd_rn(z, __fmul_rn(y, __uint_as_float(0x3e9a2080u)));
#endif
}

// noah_powi: X**N for an INTEGER N, the binary powering gfortran -O0 hands
// to libgcc's __powisf2 (module_sf_noahlsm.F:3803 ACRT**(CVFRZ-J) and
// module_sf_noahdrv.F:1312 T1**4): y = x if n is odd else 1, then square x
// and multiply it in for each further bit, every product rounded to float.
__device__ __forceinline__ float noah_powi(float x, int m)
{
    unsigned int n = m < 0 ? (unsigned int)(-m) : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = __fmul_rn(x, x);
        if (n % 2u) y = __fmul_rn(y, x);
    }
    return m < 0 ? __fdiv_rn(1.0f, y) : y;
}

// packed-table column indices (gpuwm/core/noah.py VEG_COLS/SOIL_COLS/GEN)
#define NVEGC 15
#define VG_NROOT 0
#define VG_RSMIN 1
#define VG_RGL 2
#define VG_HS 3
#define VG_SNUP 4
#define VG_LAIMIN 5
#define VG_LAIMAX 6
#define VG_EMISSMIN 7
#define VG_EMISSMAX 8
#define VG_ALBEDOMIN 9
#define VG_ALBEDOMAX 10
#define VG_Z0MIN 11
#define VG_Z0MAX 12
#define VG_SHDTBL 13
#define VG_MAXALB 14
#define NSOILC 10
#define SO_BEXP 0
#define SO_SMCDRY 1
#define SO_F1 2
#define SO_SMCMAX 3
#define SO_SMCREF 4
#define SO_PSISAT 5
#define SO_DKSAT 6
#define SO_DWSAT 7
#define SO_SMCWLT 8
#define SO_QUARTZ 9
#define GEN_TOPT 0
#define GEN_CMCMAX 1
#define GEN_CFACTR 2
#define GEN_RSMAX 3
#define GEN_SBETA 4
#define GEN_FXEXP 5
#define GEN_CSOIL 6
#define GEN_SALP 7
#define GEN_REFDK 8
#define GEN_REFKDT 9
#define GEN_FRZK 10
#define GEN_ZBOT 11
#define GEN_LVCOEF 12
#define GEN_SLOPE 13
#define GEN_BARE 14
#define GEN_NATURAL 15

// ---------------------------------------------------------------- FRH2O
__device__ static real noah_frh2o(real tkelv, real smc, real sh2o,
                                  real smcmax, real bexp, real psis)
{
    const real ck = 8.0f, blim = 5.5f, error = 0.005f;
    const real hlice = 3.335e5f, gs = 9.81f, t0 = 273.15f;
    real bx = (bexp <= blim) ? bexp : blim;
    int nlog = 0, kcount = 0;
    if (tkelv > (t0 - 1.0e-3f)) return smc;
    real swl = smc - sh2o;
    if (swl > (smc - 0.02f)) swl = smc - 0.02f;
    if (swl < 0.0f) swl = 0.0f;
    while (nlog < 10 && kcount == 0) {
        nlog += 1;
        real df = gfk_log(__fdiv_rn(psis * gs, hlice)
                       * gfk_pow(1.0f + ck * swl, 2.0f)
                       * gfk_pow(smcmax / (smc - swl), bx))
                  - gfk_log(-(tkelv - t0) / tkelv);
        real denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
        real swlk = swl - df / denom;
        if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
        if (swlk < 0.0f) swlk = 0.0f;
        real dswl = fabsf(swlk - swl);
        swl = swlk;
        if (dswl <= error) kcount += 1;
    }
    real freew = smc - swl;
    if (kcount == 0) {                 // Flerchinger explicit fallback
        real fk = gfk_pow((hlice / (gs * (-psis)))
                       * ((tkelv - t0) / tkelv), -1.0f / bx) * smcmax;
        if (fk < 0.02f) fk = 0.02f;
        freew = fminf(fk, smc);
    }
    return freew;
}

// ---------------------------------------------------------------- CSNOW
__device__ static real noah_csnow(real dsnow)
{
    return 2.0f * 0.11631f * (0.328f * gfk_pow(10.0f, 2.25f * dsnow));
}

// ------------------------------------------------------------- SNOW_NEW
__device__ static void noah_snow_new(real temp, real newsn, real& snowh,
                                     real& sndens)
{
    real snowhc = snowh * 100.0f;
    real newsnc = newsn * 100.0f;
    real tempc = temp - 273.15f;
    real dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = 0.05f + 0.0017f * gfk_pow(tempc + 15.0f, 1.5f);
    real hnewc = newsnc / dsnew;
    if (snowhc + hnewc < 1.0e-3f) sndens = fmaxf(dsnew, sndens);
    else sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc);
    snowhc = snowhc + hnewc;
    snowh = snowhc * 0.01f;
}

// --------------------------------------------------------------- SNFRAC
__device__ static real noah_snfrac(real sneqv, real snup, real salp)
{
    if (sneqv < snup) {
        real rsnow = sneqv / snup;
        return 1.0f - (gfk_exp(-salp * rsnow) - rsnow * gfk_exp(-salp));
    }
    return 1.0f;
}

// --------------------------------------------------------------- ALCALC
__device__ static void noah_alcalc(real alb, real snoalb, real embrd,
                                   real sncovr, real dt, bool snowng,
                                   real& snotime1, real lvcoef,
                                   real& albedo, real& emissi)
{
    const real snacca = 0.94f, snaccb = 0.58f;
    albedo = alb + sncovr * (snoalb - alb);
    emissi = embrd + sncovr * (NEMISSI_S - embrd);
    real snoalb1 = snoalb + lvcoef * (0.85f - snoalb);
    real snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = snotime1 + dt;
        snoalb2 = snoalb1 * gfk_pow(snacca,
                                 gfk_pow(__fdiv_rn(snotime1, 86400.0f), snaccb));
    }
    snoalb2 = fmaxf(snoalb2, alb);
    albedo = alb + sncovr * (snoalb2 - alb);
    if (albedo > snoalb2) albedo = snoalb2;
}

// --------------------------------------------------------------- TDFCND
__device__ static real noah_tdfcnd(real smc, real qz, real smcmax,
                                   real sh2o, real bexp, real psisat,
                                   int soiltyp, int opt_thcnd)
{
    if (opt_thcnd == 1 || (opt_thcnd == 2 && soiltyp != 4
                           && soiltyp != 3)) {
        real satratio = smc / smcmax;
        const real thkice = 2.2f, thkw = 0.57f, thko = 2.0f,
                   thkqtz = 7.7f;
        real thks = gfk_pow(thkqtz, qz) * gfk_pow(thko, 1.0f - qz);
        real xunfroz = sh2o / smc;
        real xu = xunfroz * smcmax;
        real thksat = gfk_pow(thks, 1.0f - smcmax)
                      * gfk_pow(thkice, smcmax - xu) * gfk_pow(thkw, xu);
        real gammd = (1.0f - smcmax) * 2700.0f;
        real thkdry = (0.135f * gammd + 64.7f)
                      / (2700.0f - 0.947f * gammd);
        real akei = satratio;
        real akel;
        if (satratio > 0.1f) akel = noah_log10f(satratio) + 1.0f;
        else akel = 0.0f;
        real ake = ((smc - sh2o) * akei + sh2o * akel) / smc;
        return ake * (thksat - thkdry) + thkdry;
    }
    real psif = psisat * 100.0f * gfk_pow(smcmax / smc, bexp);
    real pf = noah_log10f(fabsf(psif));
    if (pf <= 5.1f) return 420.0f * gfk_exp(-(pf + 2.7f));
    return 0.1744f;
}

// --------------------------------------------------------------- SNOWZ0
__device__ static real noah_snowz0(real sncovr, real z0brd, real snowh)
{
    const real z0s = 0.001f;
    real burial = 7.0f * z0brd - snowh;
    real z0eff;
    if (burial <= 0.0007f) z0eff = z0s;
    else z0eff = __fdiv_rn(burial, 7.0f);
    return (1.0f - sncovr) * z0brd + sncovr * z0eff;
}

// --------------------------------------------------------------- PENMAN
__device__ static void noah_penman(real sfctmp, real sfcprs, real ch,
                                   real t2v, real th2, real prcp,
                                   real fdown, real ssoil, real q2,
                                   real q2sat, real dqsdt2, bool snowng,
                                   bool frzgra, real emissi_in,
                                   real sncovr, real& etp, real& rch,
                                   real& rr, real& epsca, real& t24,
                                   real& flx2
#if NOAH_UA_PHYS
,real swe,real albedo,real solar,real fvb,real gama,real soil,real& etpn,real& flx4
#endif
)
{
    real emissi = emissi_in;
    real elcp1 = (1.0f - sncovr) * NELCP
                 + __fdiv_rn(sncovr * NELCP * NLSUBS, NLSUBC);
    real lvs = (1.0f - sncovr) * NLSUBC + sncovr * NLSUBS;
    flx2 = 0.0f;
    real delta = elcp1 * dqsdt2;
    t24 = sfctmp * sfctmp * sfctmp * sfctmp;
    rr = emissi * t24 * 6.48e-8f / (sfcprs * ch) + 1.0f;
    real rho = sfcprs / (NRD * t2v);
    rch = rho * NCP_PEN * ch;
    if (!snowng) {
        if (prcp > 0.0f) rr = rr + NCPH2O * prcp / rch;
    } else {
        rr = rr + NCPICE * prcp / rch;
    }
    real fnet = fdown - emissi * NSIGMA * t24 - ssoil;

#if NOAH_UA_PHYS
    real fnetn=noah_ua_canopy_flux(swe,fnet,solar,albedo,fvb,gama,soil,sfctmp,flx4);
#endif
    if (frzgra) {
        flx2 = -NLSUBF * prcp;
        fnet = fnet - flx2;
#if NOAH_UA_PHYS
        fnetn=__fsub_rn(fnetn,flx2);
#endif

    }
    real rad = fnet / rch + th2 - sfctmp;
    real a = elcp1 * (q2sat - q2);
    epsca = (a * rr + rad * delta) / (delta + rr);

#if NOAH_UA_PHYS
    real radn=__fsub_rn(__fadd_rn(__fdiv_rn(fnetn,rch),th2),sfctmp);
    real epscan=__fdiv_rn(__fadd_rn(__fmul_rn(a,rr),__fmul_rn(radn,delta)),__fadd_rn(delta,rr));
    etpn=__fdiv_rn(__fmul_rn(epscan,rch),lvs);
#endif
    etp = epsca * rch / lvs;           // AOASIS = 1 without the UCM
}

// --------------------------------------------------------------- CANRES
__device__ static void noah_canres(real solar, real ch, real sfctmp,
                                   real q2, const real* sh2o,
                                   const real* zsoil, real smcwlt,
                                   real smcref, real rsmin, int nroot,
                                   real q2sat, real dqsdt2, real topt,
                                   real rsmax, real rgl, real hs,
                                   real xlai, real sfcprs, real emissi,
                                   real& rc, real& pc)
{
    const real slv = 2.501000e6f;
    real ff = 0.55f * 2.0f * solar / (rgl * xlai);
    real rcs = (ff + rsmin / rsmax) / (1.0f + ff);
    rcs = fmaxf(rcs, 0.0001f);
#if NOAH_WRF_SCALAR_RCT || NOAH_UA_PHYS
    // The strict referee recipe builds Noah at -O2 (-fno-fast-math
    // -ffp-contract=off -fno-tree-vectorize). GNU Fortran 13.3 and 15.2 at
    // -O1 and above fold CANRES's constant **2.0 (module_sf_noahlsm.F:1083)
    // to one FP32 self multiply; at -O0 the same source calls powf, which is
    // the gfk_pow arm below and differs in rare words. Keep the subtract,
    // square, coefficient and final subtract separately rounded, as in the
    // -O2 executable (release box and a development machine disassembly, 2026-11-02).
    real temperature_delta = __fsub_rn(topt, sfctmp);
    real rct = __fsub_rn(1.0f, __fmul_rn(0.0016f,
                                    __fmul_rn(temperature_delta, temperature_delta)));
#else
    real rct = 1.0f - 0.0016f * gfk_pow(topt - sfctmp, 2.0f);
#endif
    rct = fmaxf(rct, 0.0001f);
    real rcq = 1.0f / (1.0f + hs * (q2sat - q2));
    rcq = fmaxf(rcq, 0.01f);
    real rcsoil = 0.0f;
    real gx = (sh2o[0] - smcwlt) / (smcref - smcwlt);
    gx = fminf(fmaxf(gx, 0.0f), 1.0f);
    real part[NSOIL];
    part[0] = (zsoil[0] / zsoil[nroot - 1]) * gx;
    for (int k = 1; k < nroot; ++k) {
        gx = (sh2o[k] - smcwlt) / (smcref - smcwlt);
        gx = fminf(fmaxf(gx, 0.0f), 1.0f);
        part[k] = ((zsoil[k] - zsoil[k - 1]) / zsoil[nroot - 1]) * gx;
    }
    for (int k = 0; k < nroot; ++k) rcsoil = rcsoil + part[k];
    rcsoil = fmaxf(rcsoil, 0.0001f);
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil);

#if NOAH_UA_PHYS
    // The strict native build calls libm powf for the REAL exponent **4.0.
    real tfour=gfk_pow(sfctmp,4.0f);
#else
    real tfour=powf(sfctmp,4.0f);
#endif
    real rr2 = (__fdiv_rn(4.0f * emissi * NSIGMA * NRD, CP))
#if NOAH_UA_PHYS
               * tfour / (sfcprs * ch) + 1.0f;
#else
               * gfk_pow(sfctmp, 4.0f) / (sfcprs * ch) + 1.0f;
#endif
    real delta = (slv / CP) * dqsdt2;
    pc = (rr2 + delta) / (rr2 * (1.0f + rc * ch) + delta);
}

// ---------------------------------------------------------------- DEVAP
__device__ static real noah_devap(real etp1, real smc, real shdfac,
                                  real smcmax, real smcdry, real fxexp)
{
    real sratio = (smc - smcdry) / (smcmax - smcdry);
    real fx;
    if (sratio > 0.0f) {
        fx = gfk_pow(sratio, fxexp);
        fx = fmaxf(fminf(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return fx * (1.0f - shdfac) * etp1;
}

// --------------------------------------------------------------- TRANSP
__device__ static void noah_transp(real* et, real etp1,
                                   const real* sh2o, real cmc,
                                   real shdfac, real smcwlt,
                                   real cmcmax, real pc, real cfactr,
                                   real smcref, int nroot,
                                   const real* rtdis)
{
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    real etp1a;
    if (cmc != 0.0f)
        etp1a = shdfac * pc * etp1
                * (1.0f - gfk_pow(cmc / cmcmax, cfactr));
    else
        etp1a = shdfac * pc * etp1;
    real gx[NSOIL];
    real sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (sh2o[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = fmaxf(fminf(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (real)nroot;
    real denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        real rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * fmaxf(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

// ---------------------------------------------------------------- EVAPO
__device__ static void noah_evapo(real& eta1, const real* smc, real cmc,
                                  real etp1, real dt, const real* sh2o,
                                  real smcmax, real pc, real smcwlt,
                                  real smcref, real shdfac, real cmcmax,
                                  real smcdry, real cfactr, int nroot,
                                  const real* rtdis, real fxexp,
                                  real& edir, real& ec, real* et,
                                  real& ett)
{
    edir = 0.0f;
    ec = 0.0f;
    ett = 0.0f;
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    if (etp1 > 0.0f) {
        if (shdfac < 1.0f)
            edir = noah_devap(etp1, smc[0], shdfac, smcmax, smcdry,
                              fxexp);
        if (shdfac > 0.0f) {
            noah_transp(et, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax,
                        pc, cfactr, smcref, nroot, rtdis);
            for (int k = 0; k < NSOIL; ++k) ett = ett + et[k];
            if (cmc > 0.0f)
                ec = shdfac * gfk_pow(cmc / cmcmax, cfactr) * etp1;
            else
                ec = 0.0f;
            real cmc2ms = cmc / dt;
            ec = fminf(cmc2ms, ec);
        }
    }
    eta1 = edir + ett + ec;
}

// -------------------------------------------------------------- FAC2MIT
__device__ static real noah_fac2mit(real smcmax)
{
    real flimit = 0.90f;
    if (smcmax == 0.395f) flimit = 0.59f;
    else if (smcmax == 0.434f || smcmax == 0.404f) flimit = 0.85f;
    else if (smcmax == 0.465f || smcmax == 0.406f) flimit = 0.86f;
    else if (smcmax == 0.476f || smcmax == 0.439f) flimit = 0.74f;
    else if (smcmax == 0.200f || smcmax == 0.464f) flimit = 0.80f;
    return flimit;
}

// --------------------------------------------------------------- WDFCND
__device__ static void noah_wdfcnd(real& wdf, real& wcnd, real smc,
                                   real smcmax, real bexp, real dksat,
                                   real dwsat, real sicemax)
{
    real factr1 = 0.05f / smcmax;
    real factr2 = smc / smcmax;
    factr1 = fminf(factr1, factr2);
    real expon = bexp + 2.0f;
    wdf = dwsat * gfk_pow(factr2, expon);
    if (sicemax > 0.0f) {
        real vkwgt = 1.0f / (1.0f + gfk_pow(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat
              * gfk_pow(factr1, expon);
    }
    expon = 2.0f * bexp + 3.0f;
    wcnd = dksat * gfk_pow(factr2, expon);
}

// ------------------------------------------------------------------ SRT
__device__ static void noah_srt(real* rhstt, real edir, const real* et,
                                const real* sh2o, const real* sh2oa,
                                real pcpdrp, const real* zsoil,
                                real dwsat, real dksat, real smcmax,
                                real bexp, real dt, real smcwlt,
                                real slope, real kdt, real frzx,
                                const real* sice, real* ai, real* bi,
                                real* ci, real& runoff1, real& runoff2)
{
    const int cvfrz = 3;
    real dmax[NSOIL];
    real sicemax = 0.0f;
    for (int ks = 0; ks < NSOIL; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    real pddum = pcpdrp;
    runoff1 = 0.0f;
    runoff2 = 0.0f;
    if (pcpdrp != 0.0f) {
        real dt1 = __fdiv_rn(dt, 86400.0f);
        real smcav = smcmax - smcwlt;
        dmax[0] = -zsoil[0] * smcav;
        real dice = -zsoil[0] * sice[0];
        dmax[0] = dmax[0] * (1.0f - (sh2oa[0] + sice[0] - smcwlt)
                             / smcav);
        real dd = dmax[0];
        for (int ks = 1; ks < NSOIL; ++ks) {
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks];
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav;
            dmax[ks] = dmax[ks] * (1.0f - (sh2oa[ks] + sice[ks]
                                           - smcwlt) / smcav);
            dd = dd + dmax[ks];
        }
        real val = 1.0f - gfk_exp(-kdt * dt1);
        real ddt = dd * val;
        real px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;
        real infmax = (px * (ddt / (px + ddt))) / dt;
        real fcr = 1.0f;
        if (dice > 1.0e-2f) {
            real acrt = (real)cvfrz * frzx / dice;
            real ssum = 1.0f;
            int ialp1 = cvfrz - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int k = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) k = k * jj;
                ssum = ssum + noah_powi(acrt, cvfrz - j) / (real)k;
            }
            fcr = 1.0f - gfk_exp(-acrt) * ssum;
        }
        infmax = infmax * fcr;
        real wdf0, wcnd0;
        noah_wdfcnd(wdf0, wcnd0, sh2oa[0], smcmax, bexp, dksat, dwsat,
                    sicemax);
        infmax = fmaxf(infmax, wcnd0);
        infmax = fminf(infmax, px / dt);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }
    real wdf, wcnd;
    noah_wdfcnd(wdf, wcnd, sh2oa[0], smcmax, bexp, dksat, dwsat,
                sicemax);
    real ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    real dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];
    real ddz2 = 0.0f;
    for (int k = 1; k < NSOIL; ++k) {
        real denom2 = zsoil[k - 1] - zsoil[k];
        real wdf2, wcnd2, dsmdz2, slopx;
        if (k != NSOIL - 1) {
            slopx = 1.0f;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[k], smcmax, bexp, dksat,
                        dwsat, sicemax);
            real denom = zsoil[k - 1] - zsoil[k + 1];
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k] = -wdf2 * ddz2 / denom2;
        } else {
            slopx = slope;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[NSOIL - 1], smcmax, bexp,
                        dksat, dwsat, sicemax);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        real numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf * dsmdz
                     - wcnd + et[k];
        rhstt[k] = numer / (-denom2);
        ai[k] = -wdf * ddz / denom2;
        bi[k] = -(ai[k] + ci[k]);
        if (k == NSOIL - 1) runoff2 = slopx * wcnd2;
        if (k != NSOIL - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// --------------------------------------------------------------- ROSR12
__device__ static void noah_rosr12(real* p, const real* a,
                                   const real* b, const real* c_in,
                                   const real* d)
{
    real c_[NSOIL], delta[NSOIL];
    for (int k = 0; k < NSOIL; ++k) c_[k] = c_in[k];
    c_[NSOIL - 1] = 0.0f;
    p[0] = -c_[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < NSOIL; ++k) {
        p[k] = -c_[k] * (1.0f / (b[k] + a[k] * p[k - 1]));
        delta[k] = (d[k] - a[k] * delta[k - 1])
                   * (1.0f / (b[k] + a[k] * p[k - 1]));
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        int kk = NSOIL - k - 1;
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

// ---------------------------------------------------------------- SSTEP
__device__ static void noah_sstep(real* sh2oout, const real* sh2oin,
                                  real& cmc, const real* rhstt_in,
                                  real rhsct, real dt, real smcmax,
                                  real cmcmax, const real* zsoil,
                                  real* smc, const real* sice,
                                  const real* ai_in, const real* bi_in,
                                  const real* ci_in, real& runoff3,
                                  bool update_cmc)
{
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhstt[k] = rhstt_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhstt);
    real wplus = 0.0f;
    real ddz = -zsoil[0];
    for (int k = 0; k < NSOIL; ++k) {
        if (k != 0) ddz = zsoil[k - 1] - zsoil[k];
        sh2oout[k] = sh2oin[k] + p[k] + wplus / ddz;
        real stot = sh2oout[k] + sice[k];
        if (stot > smcmax) {
            if (k == 0) ddz = -zsoil[0];
            else ddz = -zsoil[k] + zsoil[k - 1];
            wplus = (stot - smcmax) * ddz;
        } else {
            wplus = 0.0f;
        }
        smc[k] = fmaxf(fminf(stot, smcmax), 0.02f);
        sh2oout[k] = fmaxf(smc[k] - sice[k], 0.0f);
    }
    runoff3 = wplus;
    if (update_cmc) {
        cmc = cmc + dt * rhsct;
        if (cmc < 1.0e-20f) cmc = 0.0f;
        cmc = fminf(cmc, cmcmax);
    }
}

// ---------------------------------------------------------------- SMFLX
template<bool IRRIGATION = false>
__device__ static void noah_smflx(real* smc, real& cmc, real dt,
                                  real prcp1, const real* zsoil,
                                  real* sh2o, real slope, real kdt,
                                  real frzfact, real smcmax, real bexp,
                                  real smcwlt, real dksat, real dwsat,
                                  real shdfac, real cmcmax, real edir,
                                  real ec, const real* et,
                                  real& runoff1, real& runoff2,
                                  real& runoff3, real& drip, real irrigation_rate = 0.0f)
{
    real rhsct = shdfac * prcp1 - ec;
    drip = 0.0f;
    real trhsct = dt * rhsct;
    real excess = cmc + trhsct;
    if (excess > cmcmax) drip = excess - cmcmax;
    real pcpdrp = (1.0f - shdfac) * prcp1 + drip / dt;
    if constexpr (IRRIGATION) {
        if (irrigation_rate != 0.0f)
            pcpdrp = __fadd_rn(pcpdrp, __fmul_rn(0.001f, irrigation_rate));
    }
    real sice[NSOIL];
    for (int i = 0; i < NSOIL; ++i) sice[i] = smc[i] - sh2o[i];
    real fac2 = 0.0f;
    for (int i = 0; i < NSOIL; ++i)
        fac2 = fmaxf(fac2, sh2o[i] / smcmax);
    real flimit = noah_fac2mit(smcmax);
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
    if ((pcpdrp * dt) > (0.0001f * 1000.0f * (-zsoil[0]) * smcmax)
        || (fac2 > flimit)) {
        real sh2ofg[NSOIL], sh2oa[NSOIL], sh2onew[NSOIL];
        real dummy = 0.0f;
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   false);
        for (int k = 0; k < NSOIL; ++k)
            sh2oa[k] = (sh2o[k] + sh2ofg[k]) * 0.5f;
        noah_srt(rhstt, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    } else {
        real sh2onew[NSOIL];
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    }
}

// ------------------------------------------------------------------ TBND
__device__ static real noah_tbnd(real tu, real tb, const real* zsoil,
                                 real zbot, int k)
{
    real zup, zb;
    if (k == 0) zup = 0.0f;
    else zup = zsoil[k - 1];
    if (k == NSOIL - 1) zb = 2.0f * zbot - zsoil[k];
    else zb = zsoil[k + 1];
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb);
}

// ---------------------------------------------------------------- TMPAVG
__device__ static real noah_tmpavg(real tup, real tm, real tdn,
                                   const real* zsoil, int k)
{
    const real t0 = 2.7315e2f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    real dzh = dz * 0.5f;
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                return (tup + 2.0f * tm + tdn) / 4.0f;
            }
            real x0 = (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * dzh + tm * (dzh + x0)
                           + t0 * (2.0f * dzh - x0)) / dz;
        }
        if (tdn < t0) {
            real xup = (t0 - tup) * dzh / (tm - tup);
            real xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * xup + t0 * (2.0f * dz - xup - xdn)
                           + tdn * xdn) / dz;
        }
        real xup = (t0 - tup) * dzh / (tm - tup);
        return 0.5f * (tup * xup + t0 * (2.0f * dz - xup)) / dz;
    }
    if (tm < t0) {
        if (tdn < t0) {
            real xup = dzh - (t0 - tup) * dzh / (tm - tup);
            return 0.5f * (t0 * (dz - xup) + tm * (dzh + xup)
                           + tdn * dzh) / dz;
        }
        real xup = dzh - (t0 - tup) * dzh / (tm - tup);
        real xdn = (t0 - tm) * dzh / (tdn - tm);
        return 0.5f * (t0 * (2.0f * dz - xup - xdn)
                       + tm * (xup + xdn)) / dz;
    }
    if (tdn < t0) {
        real xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
        return (t0 * (dz - xdn) + 0.5f * (t0 + tdn) * xdn) / dz;
    }
    return (tup + 2.0f * tm + tdn) / 4.0f;
}

// ---------------------------------------------------------------- SNKSRC
__device__ static real noah_snksrc(real qtot, real tavg, real smc,
                                   real& sh2o, const real* zsoil,
                                   real smcmax, real psisat, real bexp,
                                   real dt, int k)
{
    const real dh2o = 1.0000e3f, hlice = 3.3350e5f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    real freew = noah_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    real xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
    if (xh2o < sh2o && xh2o < freew) {
        if (freew > sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o > sh2o && xh2o > freew) {
        if (freew < sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc) xh2o = smc;
    real tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

// ------------------------------------------------------------------ HRT
__device__ static void noah_hrt(real* rhsts, const real* stc,
                                const real* smc, real smcmax,
                                const real* zsoil, real yy, real zz1,
                                real tbot, real zbot, real psisat,
                                real* sh2o, real dt, real bexp,
                                int soiltyp, int opt_thcnd, real df1,
                                real quartz, real csoil, int vegtyp,
                                int isurban, real* ai, real* bi,
                                real* ci)
{
    const real t0 = 273.15f, cair = 1004.0f, cice = 2.106e6f,
               ch2o = 4.2e6f;
    real csoil_loc;
    if (vegtyp == isurban) csoil_loc = 3.0e6f;
    else csoil_loc = csoil;
    real hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc
                 + (smcmax - smc[0]) * cair
                 + (smc[0] - sh2o[0]) * cice;
    real ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    real dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    real ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    real denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    real qtot = -1.0f * rhsts[0] * denom;
    real sice = smc[0] - sh2o[0];
    real tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    real tbk = noah_tbnd(stc[0], stc[1], zsoil, zbot, 0);
    if (sice > 0.0f || stc[0] < t0 || tsurf < t0 || tbk < t0) {
        real tavg = noah_tmpavg(tsurf, stc[0], tbk, zsoil, 0);
        real tsnsr = noah_snksrc(qtot, tavg, smc[0], sh2o[0], zsoil,
                                 smcmax, psisat, bexp, dt, 0);
        rhsts[0] = rhsts[0] - tsnsr / denom;
    }
    real ddz2 = 0.0f;
    real df1k = df1;
    real dtsdz2;
    for (int k = 1; k < NSOIL; ++k) {
        hcpct = sh2o[k] * ch2o + (1.0f - smcmax) * csoil_loc
                + (smcmax - smc[k]) * cair
                + (smc[k] - sh2o[k]) * cice;
        real df1n, tbk1;
        if (k != NSOIL - 1) {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] - zsoil[k + 1]);
            dtsdz2 = (stc[k] - stc[k + 1]) / denom;
            ddz2 = 2.0f / (zsoil[k - 1] - zsoil[k + 1]);
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct);
            tbk1 = noah_tbnd(stc[k], stc[k + 1], zsoil, zbot, k);
        } else {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] + zsoil[k]) - zbot;
            dtsdz2 = (stc[k] - tbot) / denom;
            ci[k] = 0.0f;
            tbk1 = noah_tbnd(stc[k], tbot, zsoil, zbot, k);
        }
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct;
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom;
        qtot = -1.0f * denom * rhsts[k];
        sice = smc[k] - sh2o[k];
        real tavg = noah_tmpavg(tbk, stc[k], tbk1, zsoil, k);
        if (sice > 0.0f || stc[k] < t0 || tbk < t0 || tbk1 < t0) {
            real tsnsr = noah_snksrc(qtot, tavg, smc[k], sh2o[k],
                                     zsoil, smcmax, psisat, bexp, dt,
                                     k);
            rhsts[k] = rhsts[k] - tsnsr / denom;
        }
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct);
        bi[k] = -(ai[k] + ci[k]);
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ---------------------------------------------------------------- HSTEP
__device__ static void noah_hstep(real* stcout, const real* stcin,
                                  const real* rhsts_in, real dt,
                                  const real* ai_in, const real* bi_in,
                                  const real* ci_in)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhsts[k] = rhsts_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhsts);
    for (int k = 0; k < NSOIL; ++k) stcout[k] = stcin[k] + p[k];
}

// ---------------------------------------------------------------- SHFLX
__device__ static void noah_shflx(real* stc, const real* smc,
                                  real smcmax, real& t1, real dt,
                                  real yy, real zz1, const real* zsoil,
                                  real tbot, real zbot, real psisat,
                                  real* sh2o, real bexp, real df1,
                                  real quartz, real csoil, int vegtyp,
                                  int isurban, int soiltyp,
                                  int opt_thcnd, real& ssoil)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], stcf[NSOIL];
    noah_hrt(rhsts, stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot,
             psisat, sh2o, dt, bexp, soiltyp, opt_thcnd, df1, quartz,
             csoil, vegtyp, isurban, ai, bi, ci);
    noah_hstep(stcf, stc, rhsts, dt, ai, bi, ci);
    for (int i = 0; i < NSOIL; ++i) stc[i] = stcf[i];
    t1 = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    ssoil = df1 * (stc[0] - t1) / (0.5f * zsoil[0]);
}

// -------------------------------------------------------------- SNOWPACK
__device__ static void noah_snowpack(real esd, real dtsec, real& snowh,
                                     real& sndens, real tsnow,
                                     real tsoil
#if NOAH_UA_PHYS
,real snomlt
#endif
)
{
    const real c1k = 0.01f, c2k = 21.0f;
    real snowhc = snowh * 100.0f;
    real esdc = esd * 100.0f;
    real dthr = __fdiv_rn(dtsec, 3600.0f);
    real tsnowc = tsnow - 273.15f;
    real tsoilc = tsoil - 273.15f;
    real tavgc = 0.5f * (tsnowc + tsoilc);
    real esdcx;
    if (esdc > 1.0e-2f) esdcx = esdc;
    else esdcx = 1.0e-2f;
    real bfac = dthr * c1k * gfk_exp(0.08f * tavgc - c2k * sndens);
    const int ipol = 4;
    real pexp = 0.0f;
    for (int j = ipol; j >= 1; --j)
        pexp = __fdiv_rn((1.0f + pexp) * bfac * esdcx, (real)(j + 1));
    pexp = pexp + 1.0f;
    real dsx = sndens * pexp;
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        real dw = __fdiv_rn(0.13f * dthr, 24.0f);

#if NOAH_UA_PHYS
        if(tsoilc>=0.0f)dw=noah_ua_retention(dw,snomlt,esdcx);
#endif
        sndens = sndens * (1.0f - dw) + dw;
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowhc = esdc / sndens;
    snowh = snowhc * 0.01f;
}

// ------------------------------------------------- exact sign compare
// Subnormal-preserving float -> double, the pattern of shinhong.cu's sh_f2d
// and rrtmg_sw.cu's rsw_f2d.  CuPy compiles every kernel with -ftz=true, so a
// plain `x > 0.0f` is setp.gt.ftz.f32 and reads a positive subnormal as zero;
// a plain (double)x cast is no better, because the cvt.f64.f32 it emits
// DAZes the input too.  The bits are decoded instead, so the compare is the
// IEEE one gfortran's `.GT. 0.0` makes.
__device__ __forceinline__ double noah_f2d(real x)
{
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix >> 31) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

// ====================================================================
// LAND ICE: module_sf_noahdrv.F lsm's ICE == -1 arm (:1140-1183) and
// SFLX_GLACIAL with its whole subtree (module_sf_noahlsm_glacial_only.F,
// WRF v4.6.1, sha256 f43c6482 in the oracle receipt).  Until 2.8.6 the
// column returned untouched (TSK 255.0 against WRF's 273.15 K on the oracle's
// land-ice column).  The expressions are those noah_mosaic.cu's
// noah_mosaic_tile_glacial already carries, graded there against WRF's
// mosaic driver; this arm restates them for the ordinary driver, whose prep
// and post-update differ from lsm_mosaic's in four places (the SWDOWN > 10
// term of the snow DQSDT2 test, no CQS2 = CHS rewrite, SMSTAV = SOILW = 0,
// and noahres in float32 as written).
//
// Every operation is the rounding gfortran -O0 performs: no contraction
// (the __f*_rn intrinsics), glibc's own float32 exp/pow words (gfk_exp,
// gfk_pow from glibc_flt32.cuh), and gfortran's compile-time folds spelled
// as the same single-precision operations at run time.  The ordinary-land
// path above keeps its own arithmetic (row 18 of the parity sweep).
// ====================================================================

#define NGL_CAPA __uint_as_float(0x3e924925u)   // CAPA = R_D/CP (noahdrv:700)

// glacial CSNOW (:578-591)
__device__ static real nglac_csnow(real dsnow)
{
    real c = __fmul_rn(0.328f, gfk_pow(10.0f, __fmul_rn(2.25f, dsnow)));
    return __fmul_rn(__fmul_rn(2.0f, 0.11631f), c);
}

// glacial SNOW_NEW (:1228-1278)
__device__ static void nglac_snow_new(real temp, real newsn, real& snowh,
                                      real& sndens)
{
    real tempc = __fsub_rn(temp, 273.15f);
    real dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = __fadd_rn(0.05f, __fmul_rn(0.0017f,
                     gfk_pow(__fadd_rn(tempc, 15.0f), 1.5f)));
    real snowhc = __fmul_rn(snowh, 100.0f);
    real newsnc = __fmul_rn(newsn, 100.0f);
    real hnewc = __fdiv_rn(newsnc, dsnew);
    if (__fadd_rn(snowhc, hnewc) < 1.0e-3f) {
        sndens = fmaxf(dsnew, sndens);
    } else {
        sndens = __fdiv_rn(__fadd_rn(__fmul_rn(snowhc, sndens),
                                     __fmul_rn(hnewc, dsnew)),
                           __fadd_rn(snowhc, hnewc));
    }
    snowhc = __fadd_rn(snowhc, hnewc);
    snowh = __fmul_rn(snowhc, 0.01f);
}

// glacial ALCALC (:412-523): complete snow cover, so no SNCOVR weights.
__device__ static void nglac_alcalc(real alb, real snoalb, real embrd,
                                    real dt, bool snowng, real lvcoef,
                                    real& snotime1, real& albedo,
                                    real& emissi)
{
    const real snacca = 0.94f, snaccb = 0.58f;
    emissi = __fadd_rn(embrd, __fsub_rn(NEMISSI_S, embrd));
    real snoalb1 = __fadd_rn(snoalb, __fmul_rn(lvcoef,
                                               __fsub_rn(0.85f, snoalb)));
    real snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = __fadd_rn(snotime1, dt);
        snoalb2 = __fmul_rn(snoalb1, gfk_pow(snacca,
                      gfk_pow(__fdiv_rn(snotime1, 86400.0f), snaccb)));
    }
    snoalb2 = fmaxf(snoalb2, alb);
    albedo = __fadd_rn(alb, __fsub_rn(snoalb2, alb));
    if (albedo > snoalb2) albedo = snoalb2;
}

// glacial HRTICE (:565-707): ice thermal properties, ZBOT = -25 m.
__device__ static void nglac_hrtice(real* rhsts, const real* stc, real tbot,
                                    const real* zsoil, real yy, real zz1,
                                    real df1, real* ai, real* bi, real* ci)
{
    real hcpct = __fmul_rn(1.0e6f, __fsub_rn(0.8194f,
                     __fmul_rn(__fmul_rn(0.1309f, 0.5f), zsoil[0])));
    real df1k = df1;
    real ddz = __fdiv_rn(1.0f, __fmul_rn(-0.5f, zsoil[1]));
    ai[0] = 0.0f;
    ci[0] = __fdiv_rn(__fmul_rn(df1, ddz), __fmul_rn(zsoil[0], hcpct));
    bi[0] = __fadd_rn(-ci[0], __fdiv_rn(df1,
                __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(0.5f, zsoil[0]),
                                              zsoil[0]), hcpct), zz1)));
    real dtsdz = __fdiv_rn(__fsub_rn(stc[0], stc[1]),
                           __fmul_rn(-0.5f, zsoil[1]));
    real ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(stc[0], yy)),
                           __fmul_rn(__fmul_rn(0.5f, zsoil[0]), zz1));
    rhsts[0] = __fdiv_rn(__fsub_rn(__fmul_rn(df1, dtsdz), ssoil),
                         __fmul_rn(zsoil[0], hcpct));
    real ddz2 = 0.0f;
    df1k = df1;
    real df1n = df1;
    for (int k = 1; k < NSOIL; ++k) {
        real zmd = __fmul_rn(0.5f, __fadd_rn(zsoil[k], zsoil[k - 1]));
        hcpct = __fmul_rn(1.0e6f, __fsub_rn(0.8194f, __fmul_rn(0.1309f, zmd)));
        df1n = __fsub_rn(0.32333f, __fmul_rn(0.10073f, zmd));
        real dtsdz2;
        if (k != NSOIL - 1) {
            real denom = __fmul_rn(0.5f, __fsub_rn(zsoil[k - 1], zsoil[k + 1]));
            dtsdz2 = __fdiv_rn(__fsub_rn(stc[k], stc[k + 1]), denom);
            ddz2 = __fdiv_rn(2.0f, __fsub_rn(zsoil[k - 1], zsoil[k + 1]));
            ci[k] = -__fdiv_rn(__fmul_rn(df1n, ddz2),
                               __fmul_rn(__fsub_rn(zsoil[k - 1], zsoil[k]),
                                         hcpct));
        } else {
            dtsdz2 = __fdiv_rn(__fsub_rn(stc[k], tbot),
                               __fsub_rn(__fmul_rn(0.5f, __fadd_rn(zsoil[k - 1],
                                                                   zsoil[k])),
                                         -25.0f));
            ci[k] = 0.0f;
        }
        real denom = __fmul_rn(__fsub_rn(zsoil[k], zsoil[k - 1]), hcpct);
        rhsts[k] = __fdiv_rn(__fsub_rn(__fmul_rn(df1n, dtsdz2),
                                       __fmul_rn(df1k, dtsdz)), denom);
        ai[k] = -__fdiv_rn(__fmul_rn(df1k, ddz),
                           __fmul_rn(__fsub_rn(zsoil[k - 1], zsoil[k]), hcpct));
        bi[k] = -__fadd_rn(ai[k], ci[k]);
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// glacial SHFLX (:824-853) -> HRTICE, then HSTEP (:710-754) and the shared
// noahlsm ROSR12 (module_sf_noahlsm.F:2537).
// HSTEP (module_sf_noahlsm.F:1853, and the glacial copy :710-754) with the
// noahlsm ROSR12 it calls; shared by the land-ice and sea-ice arms.
__device__ static void nice_hstep(real* stc, real dt, real* rhsts, real* ai,
                                  real* bi, real* ci)
{
    for (int k = 0; k < NSOIL; ++k) {
        rhsts[k] = __fmul_rn(rhsts[k], dt);
        ai[k] = __fmul_rn(ai[k], dt);
        bi[k] = __fadd_rn(1.0f, __fmul_rn(bi[k], dt));
        ci[k] = __fmul_rn(ci[k], dt);
    }
    // ROSR12(P=CI, A=AI, B=BI, C=CIin, D=RHSTSin, DELTA=RHSTS)
    real p[NSOIL], delta[NSOIL], c[NSOIL];
    for (int k = 0; k < NSOIL; ++k) c[k] = ci[k];
    c[NSOIL - 1] = 0.0f;
    p[0] = __fdiv_rn(-c[0], bi[0]);
    delta[0] = __fdiv_rn(rhsts[0], bi[0]);
    for (int k = 1; k < NSOIL; ++k) {
        real rb = __fdiv_rn(1.0f, __fadd_rn(bi[k], __fmul_rn(ai[k], p[k - 1])));
        p[k] = __fmul_rn(-c[k], rb);
        delta[k] = __fmul_rn(__fsub_rn(rhsts[k], __fmul_rn(ai[k], delta[k - 1])),
                             rb);
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        int kk = NSOIL - k - 1;
        p[kk] = __fadd_rn(__fmul_rn(p[kk], p[kk + 1]), delta[kk]);
    }
    for (int k = 0; k < NSOIL; ++k) stc[k] = __fadd_rn(stc[k], p[k]);
}

__device__ static void nglac_shflx(real* stc, real dt, real yy, real zz1,
                                   const real* zsoil, real tbot, real df1)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
    nglac_hrtice(rhsts, stc, tbot, zsoil, yy, zz1, df1, ai, bi, ci);
    nice_hstep(stc, dt, rhsts, ai, bi, ci);
}

// glacial SNOWPACK (:1076-1199)
__device__ static void nglac_snowpack(real sneqv, real dtsec, real& snowh,
                                      real& sndens, real tsnow, real tsoil)
{
    real esdc = __fmul_rn(sneqv, 100.0f);
    real dthr = __fdiv_rn(dtsec, 3600.0f);
    real tsnowc = __fsub_rn(tsnow, 273.15f);
    real tsoilc = __fsub_rn(tsoil, 273.15f);
    real tavgc = __fmul_rn(0.5f, __fadd_rn(tsnowc, tsoilc));
    real esdcx = (esdc > 1.0e-2f) ? esdc : 1.0e-2f;
    real bfac = __fmul_rn(__fmul_rn(dthr, 0.01f),
                          gfk_exp(__fsub_rn(__fmul_rn(0.08f, tavgc),
                                            __fmul_rn(21.0f, sndens))));
    real pexp = 0.0f;
    for (int j = 4; j >= 1; --j)
        pexp = __fdiv_rn(__fmul_rn(__fmul_rn(__fadd_rn(1.0f, pexp), bfac),
                                   esdcx), (real)(j + 1));
    pexp = __fadd_rn(pexp, 1.0f);
    real dsx = __fmul_rn(sndens, pexp);
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        real dw = __fdiv_rn(__fmul_rn(0.13f, dthr), 24.0f);
        sndens = __fadd_rn(__fmul_rn(sndens, __fsub_rn(1.0f, dw)), dw);
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowh = __fmul_rn(__fdiv_rn(esdc, sndens), 0.01f);
}

// One land-ice column: lsm's prep (module_sf_noahdrv.F:792-981), the ICE == -1
// arm and SFLX_GLACIAL (:1140-1183), then lsm's post-update (:1210-1597).
// __noinline__ keeps the ordinary-land body's instruction selection where it
// was: this arm is a call from the place the column used to return.
__device__ __noinline__ static void noah_glacial_column(
                 size_t idx, size_t plane,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ smois_a,
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ genp,
                 const real* __restrict__ dzs,
                 real dt, int frpcpn,
                 int bep_opt, real* __restrict__ rural_bep)
{
    // ---------------- lsm prep (the statements before the ICE branch that
    // SFLX_GLACIAL reads; the vegetation, soil and urban ones it does not)
    const real a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __fmul_rn(a2, __fsub_rn(a3, a4));
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = __fdiv_rn(qv1_a[idx], __fadd_rn(1.0f, qv1_a[idx]));
    real q2sat = __fdiv_rn(qgh_a[idx], __fadd_rn(1.0f, qgh_a[idx]));
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(__fdiv_rn(1.0e5f, psfc), NGL_CAPA);
    real apelm = gfk_pow(__fdiv_rn(1.0e5f, sfcprs), NGL_CAPA);
    real sfcth2 = __fmul_rn(sfctmp, apelm);
    real th2 = __fdiv_rn(sfcth2, apes);
    real emissi = emiss_a[idx];
    real lwdn = __fmul_rn(glw_a[idx], emissi);
    real soldn = swdown_a[idx];
    real solnet = __fmul_rn(soldn, __fsub_rn(1.0f, albedo_a[idx]));
    real prcp = __fdiv_rn(rainbl_a[idx], dt);
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real sneqv = __fmul_rn(snow_a[idx], 0.001f);
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;
    real sfcma4 = __fsub_rn(sfctmp, a4);
    real dqsdt2 = __fdiv_rn(__fmul_rn(q2sat, a23m4), __fmul_rn(sfcma4, sfcma4));
    if (noah_f2d(snow_a[idx]) > 0.0) {
        real sfctsno = sfctmp;
        real e2sat = __fmul_rn(611.2f, gfk_exp(__fmul_rn(6174.0f,
                         __fsub_rn(__fdiv_rn(1.0f, 273.15f),
                                   __fdiv_rn(1.0f, sfctsno)))));
        real q2sati = __fdiv_rn(__fmul_rn(0.622f, e2sat),
                                __fsub_rn(sfcprs, e2sat));
        q2sati = __fdiv_rn(q2sati, __fadd_rn(1.0f, q2sati));
        real snowc = snowc_a[idx];
        real dqi = __fdiv_rn(__fmul_rn(q2sati, 6174.0f),
                             __fmul_rn(sfctsno, sfctsno));
        if (t1 > 273.14f) {
            q2sat = __fadd_rn(__fmul_rn(q2sat, __fsub_rn(1.0f, snowc)),
                              __fmul_rn(q2sati, snowc));
            dqsdt2 = __fadd_rn(__fmul_rn(dqsdt2, __fsub_rn(1.0f, snowc)),
                               __fmul_rn(dqi, snowc));
        } else {
            q2sat = q2sati;
            dqsdt2 = dqi;
        }
        if (t1 > 273.0f && noah_f2d(snowc) > 0.0 && swdown_a[idx] > 10.0f)
            dqsdt2 = __fmul_rn(dqsdt2, __fsub_rn(1.0f, snowc));
    }
    real tbot = tmn_a[idx];
    real snoalb1 = snoalb_a[idx];
    real cmc = __fdiv_rn(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real stc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) stc[k] = tslb_a[k * plane + idx];
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = __fmul_rn(5.0f, sneqv);

    // ---------------- SFLX_GLACIAL (:32-409)
    real zsoil[NSOIL];
    zsoil[0] = -dzs[0];
    for (int k = 1; k < NSOIL; ++k) zsoil[k] = __fadd_rn(-dzs[k], zsoil[k - 1]);
    if (sneqv < 0.10f) {
        sneqv = 0.10f;
        snowhk = 0.50f;
    }
    real sndens = __fdiv_rn(sneqv, snowhk);
    real sncond = nglac_csnow(sndens);
    bool snowng = false, frzgra = false;
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= NTFREEZ) frzgra = true;
    }
    real prcpf;
    if (snowng || frzgra) {
        real sn_new = __fmul_rn(__fmul_rn(prcp, dt), 0.001f);
        sneqv = __fadd_rn(sneqv, sn_new);
        prcpf = 0.0f;
        nglac_snow_new(sfctmp, sn_new, snowhk, sndens);
        if (sncovr > 0.99f) {
            if (stc[0] < __fsub_rn(NTFREEZ, 5.0f)) sndens = 0.2f;
            if (snowng && t1 < 273.0f && sfctmp < 273.0f) sndens = 0.2f;
        }
        sncond = nglac_csnow(sndens);
    } else {
        prcpf = prcp;
    }
    (void)prcpf;   // SNOPAC's PRCP1 = PRCPF*0.001 is never read again
    sncovr = 1.0f;
    real albedo;
    nglac_alcalc(alb, snoalb1, embrd, dt, snowng, genp[GEN_LVCOEF],
                 snotime1, albedo, emissi);
    real df1 = sncond;
    real dsoil = -__fmul_rn(0.5f, zsoil[0]);
    real dtot = __fadd_rn(snowhk, dsoil);
    real frcsno = __fdiv_rn(snowhk, dtot);
    real frcsoi = __fdiv_rn(dsoil, dtot);
    df1 = __fadd_rn(__fmul_rn(frcsno, sncond), __fmul_rn(frcsoi, df1));
    if (dtot > __fmul_rn(2.0f, dsoil)) dtot = __fmul_rn(2.0f, dsoil);
    real ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
    // glacial SNOWZ0 (:1202-1226)
    real burial = __fsub_rn(__fmul_rn(7.0f, z0brd), snowhk);
    real z0k = (burial <= 0.0007f) ? 0.001f : __fdiv_rn(burial, 7.0f);
    real fdown = __fadd_rn(solnet, lwdn);
    real t2v = __fmul_rn(sfctmp, __fadd_rn(1.0f, __fmul_rn(0.61f, q2k)));
    real rho = __fdiv_rn(sfcprs, __fmul_rn(NRD, t2v));
    real rch = __fmul_rn(__fmul_rn(rho, NCP_PEN), chk);
    real t24 = __fmul_rn(__fmul_rn(__fmul_rn(sfctmp, sfctmp), sfctmp), sfctmp);
    // glacial PENMAN (:756-821)
    real elcp1, lvs;
    if (t1 > 273.15f) {
        elcp1 = NELCP;
        lvs = NLSUBC;
    } else {
        elcp1 = __fdiv_rn(__fmul_rn(NELCP, NLSUBS), NLSUBC);
        lvs = NLSUBS;
    }
    real delta = __fmul_rn(elcp1, dqsdt2);
    real a = __fmul_rn(elcp1, __fsub_rn(q2sat, q2k));
    real rr = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(emissi, t24), 6.48e-8f),
                                  __fmul_rn(sfcprs, chk)), 1.0f);
    if (!snowng) {
        if (prcp > 0.0f) rr = __fadd_rn(rr, __fdiv_rn(__fmul_rn(NCPH2O, prcp), rch));
    } else {
        rr = __fadd_rn(rr, __fdiv_rn(__fmul_rn(NCPICE, prcp), rch));
    }
    real flx2 = frzgra ? __fmul_rn(-NLSUBF, prcp) : 0.0f;
    real fnet = __fsub_rn(__fsub_rn(__fsub_rn(fdown,
                    __fmul_rn(__fmul_rn(emissi, NSIGMA), t24)), ssoil), flx2);
    real rad = __fsub_rn(__fadd_rn(__fdiv_rn(fnet, rch), th2), sfctmp);
    real epsca = __fdiv_rn(__fadd_rn(__fmul_rn(a, rr), __fmul_rn(rad, delta)),
                           __fadd_rn(delta, rr));
    real etp = __fdiv_rn(__fmul_rn(epsca, rch), lvs);
    // glacial SNOPAC (:856-1073)
    const real esdmin = 1.0e-6f;
    real snomlt = 0.0f, esnow = 0.0f, esnow2 = 0.0f, etanrg;
    if (etp <= 0.0f) {
        if (ribb >= 0.1f && fdown > 150.0f) {
            etp = __fdiv_rn(__fadd_rn(
                      __fdiv_rn(fminf(__fmul_rn(etp, __fsub_rn(1.0f, ribb)), 0.0f),
                                0.980f),
                      __fmul_rn(etp, __fsub_rn(0.980f, 1.0f))), 0.980f);
        }
        real etp1 = __fmul_rn(etp, 0.001f);
        esnow2 = __fmul_rn(etp1, dt);
        etanrg = __fmul_rn(etp, NLSUBS);
    } else {
        esnow = etp;
        real esnow1 = __fmul_rn(esnow, 0.001f);
        esnow2 = __fmul_rn(esnow1, dt);
        etanrg = __fmul_rn(esnow, NLSUBS);
    }
    real flx1 = 0.0f;
    if (snowng) flx1 = __fmul_rn(__fmul_rn(NCPICE, prcp), __fsub_rn(t1, sfctmp));
    else if (prcp > 0.0f)
        flx1 = __fmul_rn(__fmul_rn(NCPH2O, prcp), __fsub_rn(t1, sfctmp));
    dsoil = -__fmul_rn(0.5f, zsoil[0]);
    dtot = __fadd_rn(snowhk, dsoil);
    real denom = __fadd_rn(1.0f, __fdiv_rn(df1, __fmul_rn(__fmul_rn(dtot, rr), rch)));
    real t12a = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(__fdiv_rn(
                    __fsub_rn(__fsub_rn(__fsub_rn(fdown, flx1), flx2),
                              __fmul_rn(__fmul_rn(emissi, NSIGMA), t24)), rch),
                    th2), sfctmp), __fdiv_rn(etanrg, rch)), rr);
    real t12b = __fdiv_rn(__fmul_rn(df1, stc[0]),
                          __fmul_rn(__fmul_rn(dtot, rr), rch));
    real t12 = __fdiv_rn(__fadd_rn(__fadd_rn(sfctmp, t12a), t12b), denom);
    real stc1_old = stc[0];
    real flx3 = 0.0f;
    int ebal_case;
    double reslin;
    if (t12 <= NTFREEZ) {
        ebal_case = 1;
        t1 = t12;
        ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
        sneqv = fmaxf(0.0f, __fsub_rn(sneqv, esnow2));
        reslin = ((double)fdown - (double)flx1 - (double)flx2
                  - (double)emissi * (double)NSIGMA * (double)t24
                  + (double)rch * ((double)th2 - (double)sfctmp)
                  - (double)etanrg
                  - (double)rch * (double)rr * ((double)t12 - (double)sfctmp)
                  - (double)df1 * ((double)t12 - (double)stc1_old)
                    / (double)dtot);
    } else {
        bool clipped = false;
        t1 = NTFREEZ;
        if (dtot > __fmul_rn(2.0f, dsoil)) dtot = __fmul_rn(2.0f, dsoil);
        ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
        real seh = __fmul_rn(rch, __fsub_rn(t1, th2));
        real t14 = __fmul_rn(__fmul_rn(t1, t1), __fmul_rn(t1, t1));
        if (__fsub_rn(sneqv, esnow2) <= esdmin) {
            sneqv = 0.0f;
            snomlt = 0.0f;
            flx3 = 0.0f;
            clipped = true;
        } else {
            sneqv = __fsub_rn(sneqv, esnow2);
            flx3 = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(
                       __fsub_rn(fdown, flx1), flx2),
                       __fmul_rn(__fmul_rn(emissi, NSIGMA), t14)), ssoil),
                       seh), etanrg);
            if (flx3 <= 0.0f) {
                flx3 = 0.0f;
                clipped = true;
            }
            real ex = __fdiv_rn(__fmul_rn(flx3, 0.001f), NLSUBF);
            snomlt = __fmul_rn(ex, dt);
            if (__fsub_rn(sneqv, snomlt) >= esdmin) {
                sneqv = __fsub_rn(sneqv, snomlt);
            } else {
                ex = __fdiv_rn(sneqv, dt);
                flx3 = __fmul_rn(__fmul_rn(ex, 1000.0f), NLSUBF);
                snomlt = sneqv;
                sneqv = 0.0f;
                clipped = true;
            }
        }
        ebal_case = clipped ? 3 : 2;
        reslin = ((double)fdown - (double)flx1 - (double)flx2
                  - (double)emissi * (double)NSIGMA * (double)t14
                  - (double)ssoil - (double)seh
                  - (double)etanrg - (double)flx3);
    }
    real yy = __fsub_rn(stc[0], __fdiv_rn(__fmul_rn(__fmul_rn(
                  __fmul_rn(0.5f, ssoil), zsoil[0]), 1.0f), df1));
    nglac_shflx(stc, dt, yy, 1.0f, zsoil, tbot, df1);
    if (sneqv >= 0.10f) {
        nglac_snowpack(sneqv, dt, snowhk, sndens, t1, yy);
    } else {
        sneqv = 0.10f;
        snowhk = 0.50f;
    }
    // SFLX_GLACIAL epilogue
    real eta_kin = etp;
    real q1 = __fadd_rn(q2k, __fdiv_rn(__fmul_rn(eta_kin, CP), rch));
    real sheat = __fmul_rn(__fdiv_rn(-__fmul_rn(__fmul_rn(chk, CP), sfcprs),
                                     __fmul_rn(NR_SHEAT, t2v)),
                           __fsub_rn(th2, t1));
    real esnow_e = __fmul_rn(esnow, NLSUBS);
    real etp_e = __fmul_rn(etp, NLSUBS);
    real eta_e = (etp_e > 0.0f) ? esnow_e : etp_e;
    real ssoil_out = __fmul_rn(-1.0f, ssoil);
    real runoff1 = __fdiv_rn(snomlt, dt);

    // ---------------- lsm post-update (ICE == -1 sets SOILM = 0,
    // XLAI = 0.01, RUNOFF2 = RUNOFF3 = 0, SWC = SMC = SMAV = 1; SOILW keeps
    // the 0 written at :799)
    lai_a[idx] = 0.01f;
    canwat_a[idx] = __fmul_rn(cmc, 1000.0f);
    snow_a[idx] = __fmul_rn(sneqv, 1000.0f);
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
    tsk_a[idx] = t1;
    if (bep_opt >= 2 && rural_bep != nullptr) rural_bep[idx] = t1;
    hfx_a[idx] = sheat;
    potevp_a[idx] = __fadd_rn(potevp_a[idx], __fmul_rn(etp_e,
                        __fdiv_rn(dt, __fmul_rn(XLV, RHOWATER))));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    qsfc_a[idx] = __fdiv_rn(q1, __fsub_rn(1.0f, q1));
    for (int k = 0; k < NSOIL; ++k) {
        smois_a[k * plane + idx] = 1.0f;
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = 1.0f;
        smcrel_a[k * plane + idx] = 1.0f;
    }
    real t14 = __fmul_rn(__fmul_rn(t1, t1), __fmul_rn(t1, t1));
    noahres_a[idx] = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(
                         __fadd_rn(__fsub_rn(__fadd_rn(solnet, lwdn), sheat),
                                   ssoil_out), eta_e),
                         __fmul_rn(__fmul_rn(emissi, NSTBOLT), t14)),
                         flx1), flx2), flx3);
    reslin_a[idx] = (real)reslin;
    ebal_a[idx] = ebal_case;
    smstav_a[idx] = 0.0f;
    smstot_a[idx] = __fmul_rn(0.0f, 1000.0f);
    sfcrunoff_a[idx] = __fadd_rn(sfcrunoff_a[idx],
                                 __fmul_rn(__fmul_rn(runoff1, dt), 1000.0f));
    udrunoff_a[idx] = __fadd_rn(udrunoff_a[idx],
                                __fmul_rn(__fmul_rn(0.0f, dt), 1000.0f));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = __fadd_rn(acsnow_a[idx], __fmul_rn(prcp, dt));
    if (noah_f2d(snow_a[idx]) > 0.0) {
        acsnom_a[idx] = __fadd_rn(acsnom_a[idx], __fmul_rn(snomlt, 1000.0f));
        snopcx_a[idx] = __fsub_rn(snopcx_a[idx], __fdiv_rn(snomlt,
                            __fdiv_rn(dt, __fmul_rn(1000.0f, NXLF))));
    }
}

// ---------------------------------------------------------------- kernel
// ONE BODY, TWO ENTRY POINTS.  noah_column is the default path (no urban
// model): it instantiates the body with URBAN = false, where every urban
// statement sits under `if constexpr` and is not compiled, so the kernel is
// built from exactly the statements it had before the urban hand-over was
// added.  Compiling the hand-over into the one default kernel behind a
// runtime `urban_opt > 0` test moved default forecasts: NVRTC contracts
// multiplies into adds (FMA), and the conditional rewrites of vegtyp,
// shdfac, alb, emissi, lwdn, solnet and t1 changed which products it
// fused, so a 1 h default forecast drifted from its 15-minute history on
// (measured on the RTX 5090, 2026-09-30).  noah_column_urban carries the
// hand-over for sf_urban_physics > 0.
template <bool URBAN, bool FASDAS = false, bool IRRIGATION = false>
__device__ __forceinline__
void noah_column_body(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
#if NOAH_UA_PHYS
                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a,
#endif

                 // URBAN HANDOVER (sf_urban_physics > 0), read only by the
                 // URBAN instantiation; noah_column passes zeros.
                 int urban_opt,
                 const int* __restrict__ urban_cat,   // category -> 0/1
                 int urban_cat_size, int natural,
                 const real* __restrict__ frc_urb2d,
                 const real* __restrict__ ts_urb2d,
                 real* __restrict__ tsk_rural_bep,    // options 2/3 only
                 real* __restrict__ rural_q1,
                 real* __restrict__ rural_q2k,
                 real* __restrict__ rural_zlvl,
                 const real* fasdas_h = nullptr,
                 const real* fasdas_q = nullptr,
                 const real* fasdas_norm = nullptr,
                 const real* irrigation_channel_a = nullptr)
{
    real irrigation_rate = 0.0f;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    if constexpr (IRRIGATION)
        irrigation_rate = __fdiv_rn(irrigation_channel_a[col], dt);
    size_t idx = (size_t)col;
    size_t plane = (size_t)ny * nx;

    // WRF module_sf_noahdrv.F:749-788 initializes diagnostic soil state on
    // the first call BEFORE the open-water/sea-ice early-return branches.
    if (itimestep == 1) {
        if ((xland_a[idx] - 1.5f) >= 0.0f) {
            smstav_a[idx] = 1.0f;
            smstot_a[idx] = 1.0f;
            for (int k = 0; k < NSOIL; ++k) {
                smois_a[k * plane + idx] = 1.0f;
                tslb_a[k * plane + idx] = 273.16f;
                smcrel_a[k * plane + idx] = 1.0f;
            }
        } else if (xice_a[idx] >= xice_threshold) {
            smstav_a[idx] = 1.0f;
            smstot_a[idx] = 1.0f;
            for (int k = 0; k < NSOIL; ++k) {
                smois_a[k * plane + idx] = 1.0f;
                smcrel_a[k * plane + idx] = 1.0f;
            }
        }
    }

    // ================= driver (module_sf_noahdrv.F lsm) prep =========
    // module_sf_noahdrv.F:809-814 writes CHKLOWQ for EVERY column, before the
    // XLAND land/sea branch at :869 -- so open water, sea ice and land ice all
    // leave the driver with CHKLOWQ = 1.  This kernel used to return from the
    // three skip paths below without writing it, leaving whatever the caller
    // had in the array; the WRF oracle's water and sea-ice rows found it (0.0
    // against 1.0).  The `myj` arm that can set 0 instead is not ported --
    // launch_noah has no myj argument -- so the value is unconditionally 1.
    chklowq_a[idx] = 1.0f;
    if ((xland_a[idx] - 1.5f) >= 0.0f) return;        // open water
    if (xice_a[idx] >= xice_threshold) {              // sea ice
        for (int k = 0; k < NSOIL; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];
    if (vegtyp == isice) {         // land ice: ICE = -1, SFLX_GLACIAL
        // TSK_RURAL_BEP = T1 (module_sf_noahdrv.F:1243-1247) holds for a
        // land-ice column too; only the URBAN instantiation passes it.
        int glacial_bep_opt = 0;
        real* glacial_bep = nullptr;
        if constexpr (URBAN) {
            glacial_bep_opt = urban_opt;
            glacial_bep = tsk_rural_bep;
        }
        noah_glacial_column(idx, plane, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                            qgh_a, glw_a, swdown_a, rainbl_a, sr_a, chs_a,
                            cqs2_a, chs2_a, rib_a, tmn_a, snoalb_a, embck_a,
                            tsk_a, hfx_a, qfx_a, lh_a, grdflx_a, qsfc_a,
                            canwat_a, snow_a, snowc_a, snowh_a, albedo_a,
                            albbck_a, emiss_a, znt_a, z0_a, snotime_a, lai_a,
                            smstav_a, smstot_a, sfcrunoff_a, udrunoff_a,
                            acsnow_a, acsnom_a, snopcx_a, potevp_a,
                            noahres_a, reslin_a, smois_a, tslb_a, sh2o_a,
                            smcrel_a, ebal_a, genp, dzs, dt, frpcpn,
                            glacial_bep_opt, glacial_bep);
        return;
    }

    const real tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = a2 * (a3 - a4);
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = qv1_a[idx] / (1.0f + qv1_a[idx]);
    real q2sat = qgh_a[idx] / (1.0f + qgh_a[idx]);
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(1.0e5f / psfc, RCP);             // CAPA = R_d/CP
    real apelm = gfk_pow(1.0e5f / sfcprs, RCP);
    real sfcth2 = sfctmp * apelm;
    real th2 = sfcth2 / apes;
    real emissi = emiss_a[idx];
    real lwdn = glw_a[idx] * emissi;
    real soldn = swdown_a[idx];
    real solnet = soldn * (1.0f - albedo_a[idx]);
    real prcp = rainbl_a[idx] / dt;
    real shdfac = __fdiv_rn(vegfra_a[idx], 100.0f);
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real shmin = __fdiv_rn(shdmin_a[idx], 100.0f);
    real shmax = __fdiv_rn(shdmax_a[idx], 100.0f);
    real sneqv = snow_a[idx] * 0.001f;
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;

    real dqsdt2 = q2sat * a23m4 / ((sfctmp - a4) * (sfctmp - a4));
    // WRF's IF(SNOW(I,J).GT.0.0) (module_sf_noahdrv.F:893) is TRUE for a
    // positive subnormal SNOW; `snow_a[idx] > 0.0f` under CuPy's -ftz=true
    // was FALSE, so the column took the snow-free saturation (oracle case 18:
    // TSK 0.083 K, HFX 1.95 and LH 2.87 W/m2 from WRF).  The other two
    // compares on this snow state cannot change a value: a subnormal SNEQV
    // is zeroed by SFLX's `<= 1.0e-7` reset whichever way the first test
    // goes, and a subnormal SNOW leaving SFLX comes only from SNOPAC's cold
    // arm, where SNOMLT = 0 makes the ACSNOM/SNOPCX updates below no-ops.
    if (noah_f2d(snow_a[idx]) > 0.0) {
        real sfctsno = sfctmp;
        real e2sat = 611.2f * gfk_exp(6174.0f * (1.0f / 273.15f
                                              - 1.0f / sfctsno));
        real q2sati = 0.622f * e2sat / (sfcprs - e2sat);
        q2sati = q2sati / (1.0f + q2sati);
        if (t1 > 273.14f) {
            q2sat = q2sat * (1.0f - snowc_a[idx])
                    + q2sati * snowc_a[idx];
            dqsdt2 = dqsdt2 * (1.0f - snowc_a[idx])
                     + q2sati * 6174.0f / (sfctsno * sfctsno)
                       * snowc_a[idx];
        } else {
            q2sat = q2sati;
            dqsdt2 = q2sati * 6174.0f / (sfctsno * sfctsno);
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f && soldn > 10.0f)
            dqsdt2 = dqsdt2 * (1.0f - snowc_a[idx]);
    }
    real tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    real snoalb1 = snoalb_a[idx];
    real cmc = __fdiv_rn(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real smc[NSOIL], stc[NSOIL], swc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = 5.0f * sneqv;
    // URBAN NATURAL-SURFACE REMAP, module_sf_noahdrv.F:964-990.  With an
    // urban model on, the natural fraction of a city cell is WRF's NATURAL
    // vegetation category and its skin temperature is the RURAL one: for
    // the UCM the blended TSK with the urban part taken back out, for
    // BEP/BEM the rural skin BEP kept from the last call.  The test reads
    // IVGTYP (the category array), as WRF's does, not the local VEGTYP.
    // Explicit _rn intrinsics keep the T1 recovery out of an FMA, which
    // gfortran -O0 does not form.
    bool urban_col = false;   // written only by the URBAN instantiation
    if constexpr (URBAN) {
        if (urban_opt > 0) {
            int cat = ivgtyp[idx];
            if (cat >= 0 && cat < urban_cat_size && urban_cat[cat] != 0) {
                urban_col = true;
                vegtyp = natural;
                shdfac = vegtbl[(size_t)(natural - 1) * NVEGC + VG_SHDTBL];
                alb = 0.2f;                       // ALBBRD (ALBEDOK is SFLX's)
                emissi = 0.98f;
                lwdn = glw_a[idx] * emissi;
                solnet = soldn * (1.0f - 0.2f);
                real frc = frc_urb2d[idx];
                if (frc < 0.99f) {
                    if (urban_opt == 1) {
                        t1 = __fdiv_rn(__fsub_rn(tsk_a[idx],
                                                 __fmul_rn(frc, ts_urb2d[idx])),
                                       __fsub_rn(1.0f, frc));
                    } else {
                        t1 = tsk_rural_bep[idx];
                    }
                } else {
                    t1 = tsk_a[idx];
                }
            }
        }
    }
    real xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }

    // ======================== SFLX ===================================
    real sldpth[NSOIL], zsoil[NSOIL];
    for (int k = 0; k < NSOIL; ++k) sldpth[k] = dzs[k];
    zsoil[0] = -sldpth[0];
    for (int kz = 1; kz < NSOIL; ++kz)
        zsoil[kz] = -sldpth[kz] + zsoil[kz - 1];

    // REDPRM from the packed tables (SLOPETYP = 1 as the driver sets)
    const real* sv = soiltbl + (size_t)(soiltyp - 1) * NSOILC;
    const real* vv = vegtbl + (size_t)(vegtyp - 1) * NVEGC;
    real csoil = genp[GEN_CSOIL];
    real bexp = sv[SO_BEXP];
    real dksat = sv[SO_DKSAT];
    real dwsat = sv[SO_DWSAT];
    real f1 = sv[SO_F1];
    real psisat = sv[SO_PSISAT];
    real quartz = sv[SO_QUARTZ];
    real smcdry = sv[SO_SMCDRY];
    real smcmax = sv[SO_SMCMAX];
    real smcref = sv[SO_SMCREF];
    real smcwlt = sv[SO_SMCWLT];
    real zbot = genp[GEN_ZBOT];
    real salp = genp[GEN_SALP];
    real sbeta = genp[GEN_SBETA];
    real refdk = genp[GEN_REFDK];
    real frzk = genp[GEN_FRZK];
    real fxexp = genp[GEN_FXEXP];
    real refkdt = genp[GEN_REFKDT];
    real kdt = refkdt * dksat / refdk;
    real slope = genp[GEN_SLOPE];
    real lvcoef = genp[GEN_LVCOEF];
    real frzfact = (smcmax / smcref) * (0.412f / 0.468f);
    real frzx = frzk * frzfact;
    real topt = genp[GEN_TOPT];
    real cmcmax = genp[GEN_CMCMAX];
    real cfactr = genp[GEN_CFACTR];
    real rsmax = genp[GEN_RSMAX];
    int nroot = (int)vv[VG_NROOT];
    real snup = vv[VG_SNUP];
    real rsmin = vv[VG_RSMIN];
    real rgl = vv[VG_RGL];
    real hs = vv[VG_HS];
    real emissmin = vv[VG_EMISSMIN];
    real emissmax = vv[VG_EMISSMAX];
    real laimin = vv[VG_LAIMIN];
    real laimax = vv[VG_LAIMAX];
    real z0min = vv[VG_Z0MIN];
    real z0max = vv[VG_Z0MAX];
    real albedomin = vv[VG_ALBEDOMIN];
    real albedomax = vv[VG_ALBEDOMAX];
    int bare = (int)genp[GEN_BARE];
    if (vegtyp == bare) shdfac = 0.0f;
    real rtdis[NSOIL];
    for (int i = 0; i < NSOIL; ++i) rtdis[i] = 0.0f;
    for (int i = 0; i < nroot; ++i)
        rtdis[i] = -sldpth[i] / zsoil[nroot - 1];

    // urban parameter overrides (plain Noah, no UCM)
    if (vegtyp == isurban) {
        shdfac = 0.05f;
        rsmin = 400.0f;
        smcmax = 0.45f;
        smcref = 0.42f;
        smcwlt = 0.40f;
        smcdry = 0.40f;
    }

    // background emissivity / LAI / albedo / roughness interpolation
    real embrd_o;
    if (shdfac >= shmax) {
        embrd_o = emissmax;
        if (!rdlai2d) xlai = laimax;
        if (!usemonalb) alb = albedomin;
        z0brd = z0max;
    } else if (shdfac <= shmin) {
        embrd_o = emissmin;
        if (!rdlai2d) xlai = laimin;
        if (!usemonalb) alb = albedomax;
        z0brd = z0min;
    } else {
        if (shmax > shmin) {
            real interp_fraction = (shdfac - shmin) / (shmax - shmin);
            interp_fraction = fminf(interp_fraction, 1.0f);
            interp_fraction = fmaxf(interp_fraction, 0.0f);
            embrd_o = (1.0f - interp_fraction) * emissmin
                      + interp_fraction * emissmax;
            if (!rdlai2d) xlai = (1.0f - interp_fraction) * laimin
                                 + interp_fraction * laimax;
            if (!usemonalb) alb = (1.0f - interp_fraction) * albedomax
                                  + interp_fraction * albedomin;
            z0brd = (1.0f - interp_fraction) * z0min
                    + interp_fraction * z0max;
        } else {
            embrd_o = 0.5f * emissmin + 0.5f * emissmax;
            if (!rdlai2d) xlai = 0.5f * laimin + 0.5f * laimax;
            if (!usemonalb) alb = 0.5f * albedomin + 0.5f * albedomax;
            z0brd = 0.5f * z0min + 0.5f * z0max;
        }
    }
    embrd = embrd_o;

    // snowpack density / precipitation type
    bool snowng = false, frzgra = false;
    real sndens, sncond;
    if (sneqv <= 1.0e-7f) {
        sneqv = 0.0f;
        sndens = 0.0f;
        snowhk = 0.0f;
        sncond = 1.0f;
    } else {
        sndens = sneqv / snowhk;
        sncond = noah_csnow(sndens);
    }
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= NTFREEZ) frzgra = true;
    }
    real prcpf;
    if (snowng || frzgra) {
        real sn_new = prcp * dt * 0.001f;
        sneqv = sneqv + sn_new;
        prcpf = 0.0f;
        noah_snow_new(sfctmp, sn_new, snowhk, sndens);
        sncond = noah_csnow(sndens);
    } else {
        prcpf = prcp;
    }


#if NOAH_UA_PHYS
    real flx4=0.0f,fvb=0.0f,gama=0.0f,fbur=0.0f,fgsn=0.0f,etpn=0.0f;
#endif
    // snow cover fraction, snow albedo, emissivity
    real albedo;
    if (sneqv == 0.0f) {
        sncovr = 0.0f;
        albedo = alb;
        emissi = embrd;
    } else {
        sncovr = noah_snfrac(sneqv, snup, salp);
#if NOAH_UA_PHYS
        noah_ua_cover(sneqv,snowhk,shdfac,canopy[2*(vegtyp-1)],canopy[2*(vegtyp-1)+1],xlai,fvb,gama,fbur,fgsn);
        chk=noah_ua_exchange(sfctmp,t1,shdfac,fgsn,xlai,chk);
#endif

        sncovr = fminf(sncovr, 0.98f);
        noah_alcalc(alb, snoalb1, embrd, sncovr, dt, snowng, snotime1,
                    lvcoef, albedo, emissi);
    }

    // surface thermal conductivity + first-guess soil heat flux
    real df1 = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                           psisat, soiltyp, opt_thcnd);
    if (vegtyp == isurban) df1 = 3.24f;
    df1 = df1 * gfk_exp(sbeta * shdfac);
    if (sncovr > 0.97f) df1 = sncond;
    real dsoil = -(0.5f * zsoil[0]);
    real dtot = 0.0f;
    real ssoil;
    if (sneqv == 0.0f) {
        ssoil = df1 * (t1 - stc[0]) / dsoil;
    } else {
        dtot = snowhk + dsoil;
        real frcsno = snowhk / dtot;
        real frcsoi = dsoil / dtot;
        real df1a = frcsno * sncond + frcsoi * df1;
        df1 = df1a * sncovr + df1 * (1.0f - sncovr);
        ssoil = df1 * (t1 - stc[0]) / dtot;
    }

    // roughness over snow
    real z0k;
    if (sncovr > 0.0f) z0k = noah_snowz0(sncovr, z0brd, snowhk);
    else z0k = z0brd;


#if NOAH_UA_PHYS
    z0k=noah_ua_roughness(shmax,fbur,fgsn,z0brd);
#endif
    // Penman potential evaporation
    real fdown = solnet + lwdn;
    real t2v = sfctmp * (1.0f + 0.61f * q2k);
    real etp, rch, rr, epsca, t24, flx2;
    noah_penman(sfctmp, sfcprs, chk, t2v, th2, prcp, fdown, ssoil, q2k,
                q2sat, dqsdt2, snowng, frzgra, emissi, sncovr, etp,
                rch, rr, epsca, t24, flx2
#if NOAH_UA_PHYS
,sneqv,albedo,soldn,fvb,gama,stc[0],etpn,flx4
#endif
);

    // canopy resistance
    // WRF skips CANRES after complete burial but leaves PC undefined.
    // Keep zero transpiration for that bare-canopy state.
    real rc = 0.0f, pc = 0.0f;
    if (shdfac > 0.0f && xlai > 0.0f)
        noah_canres(soldn, chk, sfctmp, q2k, swc, zsoil, smcwlt,
                    smcref, rsmin, nroot, q2sat, dqsdt2, topt, rsmax,
                    rgl, hs, xlai, sfcprs, emissi, rc, pc);

    // ---- NOPAC / SNOPAC -------------------------------------------
    real eta_kin, beta, flx1, flx3, ssoil_pac;
    real runoff1, runoff2, runoff3, dew, drip;
    real edir1k = 0.0f, ec1k = 0.0f, ett1k = 0.0f;
    real et1k[NSOIL];
    for (int k = 0; k < NSOIL; ++k) et1k[k] = 0.0f;
    real etns = 0.0f, esnow = 0.0f, snomlt = 0.0f;
    int ebal_case;
    double reslin;
    real fasdas_capacity = 0.0f;

    if (sneqv == 0.0f) {
        // =================== NOPAC ===================================
        ebal_case = 0;
        real prcp1 = prcp * 0.001f;
        real etp1 = etp * 0.001f;
        dew = 0.0f;
        real eta = 0.0f;
        if (etp > 0.0f) {
            real eta1;
            noah_evapo(eta1, smc, cmc, etp1, dt, swc, smcmax, pc,
                       smcwlt, smcref, shdfac, cmcmax, smcdry, cfactr,
                       nroot, rtdis, fxexp, edir1k, ec1k, et1k, ett1k);
            // WRF v4.6.1 NOPAC: evaporation corrections precede SMFLX.
            // ETA1 and ETT1 retain the pre-correction values, as in WRF.
            if constexpr (FASDAS) {
                real total = 0.0f;
                for (int k=0; k<NSOIL; ++k) total = total + et1k[k];
                total = edir1k + ec1k + total;
                if (total != 0.0f) {
                    real q = fasdas_q[idx], norm = fasdas_norm[idx];
                    real wet = (smc[0] >= smcref && q > 0.0f) ? 0.0f : 1.0f;
                    real direct = __fdiv_rn(edir1k,total)*q*1.0e-3f*norm;
                    direct = direct*wet;
                    edir1k = edir1k + direct;
                    real canopy = __fdiv_rn(ec1k,total)*q*1.0e-3f*norm;
                    ec1k = ec1k + canopy;
                    for (int k=0; k<NSOIL; ++k) {
                        wet = (smc[k] >= smcref && q > 0.0f) ? 0.0f : 1.0f;
                        real root = __fdiv_rn(et1k[k],total)*q*1.0e-3f*norm;
                        root = root*wet;
                        et1k[k] = et1k[k] + root;
                    }
                }
            }
            noah_smflx<IRRIGATION>(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip, irrigation_rate);
            eta = eta1 * 1000.0f;
        } else {
            dew = -etp1;
            prcp1 = prcp1 + dew;
            noah_smflx<IRRIGATION>(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip, irrigation_rate);
        }
        if (etp <= 0.0f) {
            beta = 0.0f;
            eta = etp;
            if (etp < 0.0f) beta = 1.0f;
        } else {
            beta = eta / etp;
        }
        real df1n = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                                psisat, soiltyp, opt_thcnd);
        if (vegtyp == isurban) df1n = 3.24f;
        df1n = df1n * gfk_exp(sbeta * shdfac);
        real yynum = fdown - emissi * NSIGMA * t24;
        real yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca)
                           / rr;
        real zz1 = df1n / (-0.5f * zsoil[0] * rch * rr) + 1.0f;
        if constexpr (FASDAS) {
            real soil_capacity = (vegtyp == isurban) ? 3.0e6f : csoil;
            fasdas_capacity = swc[0]*4.2e6f + (1.0f-smcmax)*soil_capacity
                            + (smcmax-smc[0])*1004.0f + (smc[0]-swc[0])*2.106e6f;
        }
        noah_shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1n, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil_pac);
        flx1 = NCPH2O * prcp * (t1 - sfctmp);
        flx3 = 0.0f;
        eta_kin = eta;
        reslin = 0.0;                       // filled below (needs eta_e)
    } else {
        // =================== SNOPAC ==================================
        const real esdmin = 1.0e-6f, snoexp = 2.0f;
        real esd = sneqv;
        dew = 0.0f;
        real etns1 = 0.0f;
        real esnow1, esnow2;
        esnow = 0.0f;
        esnow1 = 0.0f;
        esnow2 = 0.0f;
        real prcp1 = prcpf * 0.001f;
        beta = 1.0f;
        real etanrg, etp1;
        if (etp <= 0.0f) {
            if (ribb >= 0.1f && fdown > 150.0f) {
                etp = __fdiv_rn((__fdiv_rn(fminf(etp * (1.0f - ribb), 0.0f) * sncovr,
                       0.980f) + etp * (0.980f - sncovr)), 0.980f);
            }
            if (etp == 0.0f) beta = 0.0f;
            etp1 = etp * 0.001f;
            dew = -etp1;
            esnow2 = etp1 * dt;
            etanrg = etp * ((1.0f - sncovr) * NLSUBC
                            + sncovr * NLSUBS);
        } else {
            etp1 = etp * 0.001f;
            if (sncovr < 1.0f) {
                noah_evapo(etns1, smc, cmc, etp1, dt, swc, smcmax, pc,
                           smcwlt, smcref, shdfac, cmcmax, smcdry,
                           cfactr, nroot, rtdis, fxexp, edir1k, ec1k,
                           et1k, ett1k);
                edir1k = edir1k * (1.0f - sncovr);
                ec1k = ec1k * (1.0f - sncovr);
                for (int k = 0; k < NSOIL; ++k)
                    et1k[k] = et1k[k] * (1.0f - sncovr);
                ett1k = ett1k * (1.0f - sncovr);
                etns1 = etns1 * (1.0f - sncovr);
                etns = etns1 * 1000.0f;
            }
            esnow = etp * sncovr;
#if NOAH_UA_PHYS
            esnow=__fmul_rn(etpn,sncovr);
#endif

            esnow1 = esnow * 0.001f;
            esnow2 = esnow1 * dt;
            etanrg = esnow * NLSUBS + etns * NLSUBC;
        }
        flx1 = 0.0f;
        if (snowng) flx1 = NCPICE * prcp * (t1 - sfctmp);
        else if (prcp > 0.0f) flx1 = NCPH2O * prcp * (t1 - sfctmp);
        real dsoil2 = -0.5f * zsoil[0];
        dtot = snowhk + dsoil2;
        real denom = 1.0f + df1 / (dtot * rr * rch);
        real t12a = ((fdown - flx1 - flx2 - emissi * NSIGMA * t24)
                     / rch + th2 - sfctmp - etanrg / rch) / rr;
        real t12b = df1 * stc[0] / (dtot * rr * rch);
        real t12 = (sfctmp + t12a + t12b) / denom;
        real stc1_old = stc[0];
        real ex = 0.0f;
        snomlt = 0.0f;
        if (t12 <= NTFREEZ) {
            ebal_case = 1;
            t1 = t12;
            ssoil_pac = df1 * (t1 - stc[0]) / dtot;
            esd = fmaxf(0.0f, esd - esnow2);
            flx3 = 0.0f;
#if NOAH_UA_PHYS
            flx4=0.0f;
#endif
            reslin = ((double)fdown - (double)flx1 - (double)flx2
                      - (double)emissi * (double)NSIGMA * (double)t24
                      + (double)rch * ((double)th2 - (double)sfctmp)
                      - (double)etanrg
                      - (double)rch * (double)rr
                        * ((double)t12 - (double)sfctmp)
                      - (double)df1 * ((double)t12 - (double)stc1_old)
                        / (double)dtot);
        } else {
            t1 = NTFREEZ * fmaxf(0.01f, gfk_pow(sncovr, snoexp))
                 + t12 * (1.0f - fmaxf(0.01f, gfk_pow(sncovr, snoexp)));
            beta = 1.0f;
            ssoil_pac = df1 * (t1 - stc[0]) / dtot;
            bool clipped = false;
            if (esd - esnow2 <= esdmin) {
                esd = 0.0f;
                ex = 0.0f;
                snomlt = 0.0f;
                flx3 = 0.0f;
#if NOAH_UA_PHYS
                flx4=0.0f;
#endif
                clipped = true;
            } else {
                esd = esd - esnow2;
                real seh = rch * (t1 - th2);
                real t14 = t1 * t1;
                t14 = t14 * t14;
                flx3 = fdown - flx1 - flx2 - emissi * NSIGMA * t14
                       - ssoil_pac - seh - etanrg;
                real flx3_def = flx3;
                if (flx3 <= 0.0f) flx3 = 0.0f;
                if (flx3 != flx3_def) clipped = true;

#if NOAH_UA_PHYS
                noah_ua_melt_flux(flx3,flx4);
#endif
                ex = __fdiv_rn(flx3 * 0.001f, NLSUBF);
                snomlt = ex * dt;
                if (esd - snomlt >= esdmin) {
                    esd = esd - snomlt;
                } else {
                    ex = esd / dt;
                    flx3 = ex * 1000.0f * NLSUBF;
                    snomlt = esd;
                    esd = 0.0f;
                    clipped = true;
                }
            }
            ebal_case = clipped ? 3 : 2;
            prcp1 = prcp1 + ex;
            real seh = rch * (t1 - th2);
            real t14 = t1 * t1;
            t14 = t14 * t14;
            reslin = ((double)fdown - (double)flx1 - (double)flx2
                      - (double)emissi * (double)NSIGMA * (double)t14
                      - (double)ssoil_pac - (double)seh
                      - (double)etanrg - (double)flx3);
        }
        noah_smflx<IRRIGATION>(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                   frzx, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                   cmcmax, edir1k, ec1k, et1k, runoff1, runoff2,
                   runoff3, drip);
        real zz1 = 1.0f;
        real yy = stc[0] - 0.5f * ssoil_pac * zsoil[0] * zz1 / df1;
        real t11 = t1;
        real ssoil1;
        if constexpr (FASDAS) {
            real soil_capacity = (vegtyp == isurban) ? 3.0e6f : csoil;
            fasdas_capacity = swc[0]*4.2e6f + (1.0f-smcmax)*soil_capacity
                            + (smcmax-smc[0])*1004.0f + (smc[0]-swc[0])*2.106e6f;
        }
        noah_shflx(stc, smc, smcmax, t11, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil1);
        if (esd > 0.0f) {
            noah_snowpack(esd, dt, snowhk, sndens, t1, yy
#if NOAH_UA_PHYS
,snomlt
#endif
);
        } else {
            esd = 0.0f;
            snowhk = 0.0f;
            sndens = 0.0f;
            sncovr = 0.0f;
        }
        sneqv = esd;
        eta_kin = esnow + etns - 1000.0f * dew;
    }

    real q1 = q2k + eta_kin * CP / rch;
    real sheat = -(chk * CP * sfcprs) / (NR_SHEAT * t2v) * (th2 - t1);


#if NOAH_UA_PHYS
    sheat=__fadd_rn(sheat,flx4);
#endif
    // kinematic -> energy conversions (SFLX epilogue)
    real edir_e = (edir1k * 1000.0f) * NLVH2O;
    real ec_e = (ec1k * 1000.0f) * NLVH2O;
    real ett_e = (ett1k * 1000.0f) * NLVH2O;
    real esnow_e = esnow * NLSUBS;
    real etp_e = etp * ((1.0f - sncovr) * NLVH2O + sncovr * NLSUBS);
    real eta_e;
    if (etp_e > 0.0f) eta_e = edir_e + ec_e + ett_e + esnow_e;
    else eta_e = etp_e;
    if (etp_e == 0.0f) beta = 0.0f;
    else beta = eta_e / etp_e;

    if (ebal_case == 0) {
        reslin = ((double)fdown
                  - (double)emissi * (double)NSIGMA * (double)t24
                  + (double)rch * ((double)th2 - (double)sfctmp)
                  - (double)eta_e
                  - (double)rch * (double)rr
                    * ((double)t1 - (double)sfctmp)
                  - (double)ssoil_pac);
    }

    real ssoil_out = -1.0f * ssoil_pac;
    runoff3 = runoff3 / dt;
    runoff2 = runoff2 + runoff3;
    real soilm = -1.0f * smc[0] * zsoil[0];
    for (int k = 1; k < NSOIL; ++k)
        soilm = soilm + smc[k] * (zsoil[k - 1] - zsoil[k]);
    real soilwm = -1.0f * (smcmax - smcwlt) * zsoil[0];
    real soilww = -1.0f * (smc[0] - smcwlt) * zsoil[0];
    real smav[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        smav[k] = (smc[k] - smcwlt) / (smcmax - smcwlt);
    if (nroot >= 2) {
        for (int k = 1; k < nroot; ++k) {
            soilwm = soilwm + (smcmax - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k]);
            soilww = soilww + (smc[k] - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k]);
        }
    }
    real soilw;
    if (soilwm < 1.0e-6f) {
        soilwm = 0.0f;
        soilw = 0.0f;
        soilm = 0.0f;
    } else {
        soilw = soilww / soilwm;
    }

    // ================= driver post-SFLX updates ======================
    lai_a[idx] = xlai;
    canwat_a[idx] = cmc * 1000.0f;
    snow_a[idx] = sneqv * 1000.0f;
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
    // LSM applies the FASDAS skin correction after SFLX has diagnosed its
    // fluxes; those fluxes keep their pre-correction T1.
    if constexpr (FASDAS) {
        real latent = fasdas_q[idx]*2.5e6f*fasdas_norm[idx];
        t1 = t1 + __fdiv_rn((fasdas_h[idx]-latent)*dt,fasdas_capacity*dzs[0]);
    }
    tsk_a[idx] = t1;
    // module_sf_noahdrv.F:1243-1247: TSK_RURAL_BEP = T1 for BEP/BEM.
    if constexpr (URBAN) {
        if (urban_opt >= 2) tsk_rural_bep[idx] = t1;
    }
    hfx_a[idx] = sheat;
    potevp_a[idx] = potevp_a[idx] + etp_e * (__fdiv_rn(dt, (XLV * RHOWATER)));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    // WRF module_sf_noahdrv.F:1275: ordinary-land Noah makes the 2 m
    // thermal exchange coefficient identical to its moisture coefficient.
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    qsfc_a[idx] = q1 / (1.0f - q1);
    if constexpr (URBAN) {
        if (urban_col) {
            // The rural values the UCM blend reads and QSFC cannot give back
            // exactly: Q1 before the mixing-ratio conversion, the specific
            // humidity Q2K and ZLVL = 0.5*DZ8W (module_sf_noahdrv.F:798, 817).
            rural_q1[idx] = q1;
            rural_q2k[idx] = q2k;
            rural_zlvl[idx] = 0.5f * dz8w1_a[idx];
        }
    }
    for (int k = 0; k < NSOIL; ++k) {
        smois_a[k * plane + idx] = smc[k];
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = swc[k];
        smcrel_a[k * plane + idx] = smav[k];
    }
    // module_sf_noahdrv.F:1312: the residual is a float32 sum, left to
    // right, with T1**4 as gfortran's -O0 __powisf2 forms it.  It cancels
    // terms of order 500 W/m2 into a residual of order 0.01, so the float32
    // rounding of each partial sum IS the answer; a float64 sum rounded once
    // is a different function of the same terms.
    noahres_a[idx] = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(
                         __fadd_rn(__fsub_rn(__fadd_rn(solnet, lwdn), sheat),
                                   ssoil_out), eta_e),
                         __fmul_rn(__fmul_rn(emissi, NSTBOLT),
                                   noah_powi(t1, 4))),
                         flx1), flx2), flx3);

#if NOAH_UA_PHYS
    real t14ua=__fmul_rn(__fmul_rn(t1,t1),__fmul_rn(t1,t1));
    noahres_a[idx]=(solnet+lwdn)-sheat+ssoil_out-eta_e-(emissi*NSTBOLT*t14ua)-flx1-flx2-flx3-flx4;
    flx4_a[idx]=flx4;fvb_a[idx]=fvb;fbur_a[idx]=fbur;fgsn_a[idx]=fgsn;
#endif
    reslin_a[idx] = (real)reslin;
    ebal_a[idx] = ebal_case;
    chklowq_a[idx] = 1.0f;
    smstav_a[idx] = soilw;
    smstot_a[idx] = soilm * 1000.0f;
    sfcrunoff_a[idx] = sfcrunoff_a[idx] + runoff1 * dt * 1000.0f;
    udrunoff_a[idx] = udrunoff_a[idx] + runoff2 * dt * 1000.0f;
    if (ffrozp > 0.5f)
        acsnow_a[idx] = acsnow_a[idx] + prcp * dt;
    if (snow_a[idx] > 0.0f) {
        acsnom_a[idx] = acsnom_a[idx] + snomlt * 1000.0f;
#if NOAH_UA_PHYS
        snopcx_a[idx]=__fsub_rn(snopcx_a[idx],__fdiv_rn(snomlt,__fdiv_rn(dt,350000000.0f)));
#else
        // module_sf_noahdrv.F:1595 SNOPCX - SNOMLT/FDTLIW with FDTLIW =
        // DT/ROWLIW (:730) and ROWLIW = 1.E3*XLF, as the land-ice arm spells
        // it; SNOMLT*1000*XLF/DT is the same quantity rounded differently
        // (1 ULP on seven melting columns of the oracle).
        snopcx_a[idx] = __fsub_rn(snopcx_a[idx], __fdiv_rn(snomlt,
                            __fdiv_rn(dt, __fmul_rn(1000.0f, NXLF))));
#endif
    }
}

// Keep the ordinary-land column below the register-allocation step that
// would remove resident column blocks. The scalar arithmetic is unchanged.
extern "C" __global__ __launch_bounds__(64, 8)
void noah_column(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx
#if NOAH_UA_PHYS
,                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a
#endif
                 )
{
    noah_column_body<false>(ivgtyp, isltyp, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                            qgh_a, dz8w1_a, glw_a, swdown_a, rainbl_a, sr_a,
                            chs_a, cqs2_a, chs2_a, rib_a, vegfra_a, shdmin_a,
                            shdmax_a, tmn_a, xland_a, xice_a, snoalb_a,
                            embck_a, tsk_a, hfx_a, qfx_a, lh_a, grdflx_a,
                            qsfc_a, canwat_a, snow_a, snowc_a, snowh_a,
                            albedo_a, albbck_a, emiss_a, znt_a, z0_a,
                            snotime_a, lai_a, smstav_a, smstot_a, sfcrunoff_a,
                            udrunoff_a, acsnow_a, acsnom_a, snopcx_a,
                            potevp_a, noahres_a, reslin_a, chklowq_a, smois_a,
                            tslb_a, sh2o_a, smcrel_a, ebal_a, vegtbl, soiltbl,
                            genp, dzs, dt, lucats, slcats, isurban, isice,
                            xice_threshold, itimestep, frpcpn, usemonalb,
                            rdlai2d, opt_thcnd, ny, nx,
#if NOAH_UA_PHYS
 canopy,flx4_a,fvb_a,fbur_a,fgsn_a,
#endif
 0, nullptr, 0, 0,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr);
}

extern "C" __global__
void noah_column_urban(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
#if NOAH_UA_PHYS
                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a,
#endif

                 // URBAN HANDOVER (sf_urban_physics > 0), read only by the
                 // URBAN instantiation; noah_column passes zeros.
                 int urban_opt,
                 const int* __restrict__ urban_cat,   // category -> 0/1
                 int urban_cat_size, int natural,
                 const real* __restrict__ frc_urb2d,
                 const real* __restrict__ ts_urb2d,
                 real* __restrict__ tsk_rural_bep,    // options 2/3 only
                 real* __restrict__ rural_q1,
                 real* __restrict__ rural_q2k,
                 real* __restrict__ rural_zlvl)
{
    noah_column_body<true>(ivgtyp, isltyp, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                           qgh_a, dz8w1_a, glw_a, swdown_a, rainbl_a, sr_a,
                           chs_a, cqs2_a, chs2_a, rib_a, vegfra_a, shdmin_a,
                           shdmax_a, tmn_a, xland_a, xice_a, snoalb_a,
                           embck_a, tsk_a, hfx_a, qfx_a, lh_a, grdflx_a,
                           qsfc_a, canwat_a, snow_a, snowc_a, snowh_a,
                           albedo_a, albbck_a, emiss_a, znt_a, z0_a,
                           snotime_a, lai_a, smstav_a, smstot_a, sfcrunoff_a,
                           udrunoff_a, acsnow_a, acsnom_a, snopcx_a, potevp_a,
                           noahres_a, reslin_a, chklowq_a, smois_a, tslb_a,
                           sh2o_a, smcrel_a, ebal_a, vegtbl, soiltbl, genp,
                           dzs, dt, lucats, slcats, isurban, isice,
                           xice_threshold, itimestep, frpcpn, usemonalb,
                           rdlai2d, opt_thcnd, ny, nx,
#if NOAH_UA_PHYS
 canopy,flx4_a,fvb_a,fbur_a,fgsn_a,
#endif
 urban_opt, urban_cat,
                           urban_cat_size, natural, frc_urb2d, ts_urb2d,
                           tsk_rural_bep, rural_q1, rural_q2k, rural_zlvl);
}


// ====================================================================
// SEA ICE: WRF v4.6.1 seaice_noah (module_sf_noah_seaice_drv.F) and
// SFLX_SEAICE with its subtree (module_sf_noah_seaice.F), which the surface
// driver calls after lsm / lsm_mosaic for every Noah run
// (module_surface_driver.F:2917-2929).  lsm itself only sets SH2O = 1 and
// LAI = 0.01 on a sea-ice column and returns; until 2.8.6 nothing advanced
// the column afterwards, so TSK stayed at its initial SKINTEMP over sea ice.
//
// The arm is WRF's default namelist path, the only one gpuwm admits:
// seaice_albedo_opt = 0 (albedo = seaice_albedo_default, emissivity 0.98,
// full snow cover), seaice_thickness_opt = 0 (seaice_thickness_default),
// seaice_snowdepth_opt = 0 (depth clamped to [seaice_snowdepth_min,
// seaice_snowdepth_max]) and fractional_seaice = 0.  DT is the model step,
// not DTBL: the driver hands seaice_noah `dt` where it hands lsm `dtbl`.
// Rounding follows gfortran -O0 exactly, as the land-ice arm does.
// ====================================================================

// SFLX_SEAICE's HRTICE (module_sf_noah_seaice.F:585-703): constant ice heat
// capacity and conductivity, lower boundary at the ice base ZSOIL(NSOIL).
__device__ static void nsice_hrtice(real* rhsts, const real* stc, real tbot,
                                    const real* zsoil, real yy, real zz1,
                                    real df1, real* ai, real* bi, real* ci)
{
    const real hcpct = 1.72396e+6f;
    real zbot = zsoil[NSOIL - 1];
    real ddz = __fdiv_rn(1.0f, __fmul_rn(-0.5f, zsoil[1]));
    ai[0] = 0.0f;
    ci[0] = __fdiv_rn(__fmul_rn(df1, ddz), __fmul_rn(zsoil[0], hcpct));
    bi[0] = __fadd_rn(-ci[0], __fdiv_rn(df1,
                __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(0.5f, zsoil[0]),
                                              zsoil[0]), hcpct), zz1)));
    real dtsdz = __fdiv_rn(__fsub_rn(stc[0], stc[1]),
                           __fmul_rn(-0.5f, zsoil[1]));
    real ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(stc[0], yy)),
                           __fmul_rn(__fmul_rn(0.5f, zsoil[0]), zz1));
    rhsts[0] = __fdiv_rn(__fsub_rn(__fmul_rn(df1, dtsdz), ssoil),
                         __fmul_rn(zsoil[0], hcpct));
    real ddz2 = 0.0f;
    real df1k = df1, df1n = df1;
    for (int k = 1; k < NSOIL; ++k) {
        real dtsdz2;
        if (k != NSOIL - 1) {
            real denom = __fmul_rn(0.5f, __fsub_rn(zsoil[k - 1], zsoil[k + 1]));
            dtsdz2 = __fdiv_rn(__fsub_rn(stc[k], stc[k + 1]), denom);
            ddz2 = __fdiv_rn(2.0f, __fsub_rn(zsoil[k - 1], zsoil[k + 1]));
            ci[k] = -__fdiv_rn(__fmul_rn(df1n, ddz2),
                               __fmul_rn(__fsub_rn(zsoil[k - 1], zsoil[k]),
                                         hcpct));
        } else {
            dtsdz2 = __fdiv_rn(__fsub_rn(stc[k], tbot),
                               __fsub_rn(__fmul_rn(0.5f, __fadd_rn(zsoil[k - 1],
                                                                   zsoil[k])),
                                         zbot));
            ci[k] = 0.0f;
        }
        real denom = __fmul_rn(__fsub_rn(zsoil[k], zsoil[k - 1]), hcpct);
        rhsts[k] = __fdiv_rn(__fsub_rn(__fmul_rn(df1n, dtsdz2),
                                       __fmul_rn(df1k, dtsdz)), denom);
        ai[k] = -__fdiv_rn(__fmul_rn(df1k, ddz),
                           __fmul_rn(__fsub_rn(zsoil[k - 1], zsoil[k]), hcpct));
        bi[k] = -__fadd_rn(ai[k], ci[k]);
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// seaice_snowdepth_opt = 0 (module_sf_noah_seaice.F, both clamp sites):
// keep the density, clamp the depth, rebuild the water equivalent.
__device__ __forceinline__ void nsice_clamp_depth(real& sneqv, real& snowh,
                                                  real dmin, real dmax)
{
    real sndens = __fdiv_rn(sneqv, snowh);
    snowh = fmaxf(dmin, fminf(snowh, dmax));
    sneqv = __fmul_rn(snowh, sndens);
}

extern "C" __global__
#if GPUWM_SEAICE_OPTIONS
void noah_seaice_options_column(
#else
void noah_seaice_column(
#endif
const real* __restrict__ xice_a,
                        const real* __restrict__ psfc_a,
                        const real* __restrict__ sfcprs_a,
                        const real* __restrict__ sfctmp_a,
                        const real* __restrict__ qv1_a,
                        const real* __restrict__ qgh_a,
                        const real* __restrict__ glw_a,
                        const real* __restrict__ swdown_a,
                        const real* __restrict__ rainbl_a,
                        const real* __restrict__ sr_a,
                        const real* __restrict__ rib_a,
                        const real* __restrict__ chs_a,
                        real* __restrict__ cqs2_a,
                        real* __restrict__ chs2_a,
                        real* __restrict__ tsk_a,
                        real* __restrict__ emiss_a,
                        real* __restrict__ albedo_a,
                        real* __restrict__ z0_a,
                        real* __restrict__ snow_a,
                        real* __restrict__ snowc_a,
                        real* __restrict__ snowh_a,
                        real* __restrict__ znt_a,
                        real* __restrict__ lh_a,
                        real* __restrict__ hfx_a,
                        real* __restrict__ qfx_a,
                        real* __restrict__ potevp_a,
                        real* __restrict__ grdflx_a,
                        real* __restrict__ qsfc_a,
                        real* __restrict__ acsnow_a,
                        real* __restrict__ acsnom_a,
                        real* __restrict__ snopcx_a,
                        real* __restrict__ sfcrunoff_a,
                        real* __restrict__ noahres_a,
                        real* __restrict__ tslb_a,       // (4, ny, nx)
                        int* __restrict__ fatal,         // (1,)
                        real dt, real xice_threshold, int frpcpn,
                        real albedo_default, real sithick,
                        real snowdepth_min, real snowdepth_max,
                        int ny, int nx
#if GPUWM_SEAICE_OPTIONS
                        , const real* albsi_a, real* icedepth_a, real* snowsi_a,
                        int albedo_opt, int thickness_opt, int snowdepth_opt
#endif
                        )
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    size_t idx = (size_t)col;
    size_t plane = (size_t)ny * nx;
#if GPUWM_SEAICE_OPTIONS
    if (xice_a[idx] < xice_threshold) {
        if (thickness_opt == 1) icedepth_a[idx] = 0.0f;
        if (snowdepth_opt == 1) snowsi_a[idx] = 0.0f;
        return;
    }
    if (thickness_opt == 1) {
        sithick = fminf(fmaxf(0.10f, icedepth_a[idx]), 10.0f);
        icedepth_a[idx] = sithick;
    }
    if (albedo_opt == 2) albedo_a[idx] = albsi_a[idx];
    real snowonsi = 0.0f;
    if (snowdepth_opt == 1)
        snowonsi = fmaxf(snowdepth_min, fminf(snowsi_a[idx], snowdepth_max));
#else
    if (xice_a[idx] < xice_threshold) return;
#endif      // driver :CYCLE

    // ---------------- seaice_noah prep
    const real a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __fmul_rn(a2, __fsub_rn(a3, a4));
    real sfctmp = sfctmp_a[idx];
    real t1 = tsk_a[idx];
    // SNOALB is read only under seaice_albedo_opt = 1 or 2, so this arm
    // takes no SNOALB argument.
    real emissi = emiss_a[idx];
    real lwdn = __fmul_rn(glw_a[idx], emissi);
    real sneqv = __fmul_rn(snow_a[idx], 0.001f);
    real snowh = snowh_a[idx];
#if GPUWM_SEAICE_OPTIONS
    if (snowdepth_opt == 1) sneqv = __fmul_rn(snowonsi, 0.3f);
#endif
    real sncovr = snowc_a[idx];
    real solnet = __fmul_rn(swdown_a[idx], __fsub_rn(1.0f, albedo_a[idx]));
    real sfcprs = sfcprs_a[idx];
    real psfc = psfc_a[idx];
    real q2 = __fdiv_rn(qv1_a[idx], __fadd_rn(1.0f, qv1_a[idx]));
    real apes = gfk_pow(__fdiv_rn(1.0e5f, psfc), NGL_CAPA);
    real apelm = gfk_pow(__fdiv_rn(1.0e5f, sfcprs), NGL_CAPA);
    real th2 = __fdiv_rn(__fmul_rn(sfctmp, apelm), apes);
    real q2sat = __fdiv_rn(qgh_a[idx], __fadd_rn(1.0f, qgh_a[idx]));
    real sfcma4 = __fsub_rn(sfctmp, a4);
    real dqsdt2 = __fdiv_rn(__fmul_rn(q2sat, a23m4), __fmul_rn(sfcma4, sfcma4));
    if (noah_f2d(snow_a[idx]) > 0.0) {             // IF(SNOW .GT. 0.0)
        real sfctsno = sfctmp;
        real e2sat = __fmul_rn(611.2f, gfk_exp(__fmul_rn(6174.0f,
                         __fsub_rn(__fdiv_rn(1.0f, 273.15f),
                                   __fdiv_rn(1.0f, sfctsno)))));
        real q2sati = __fdiv_rn(__fmul_rn(0.622f, e2sat),
                                __fsub_rn(sfcprs, e2sat));
        q2sati = __fdiv_rn(q2sati, __fadd_rn(1.0f, q2sati));
        real snowc = snowc_a[idx];
        real dqi = __fdiv_rn(__fmul_rn(q2sati, 6174.0f),
                             __fmul_rn(sfctsno, sfctsno));
        if (t1 > 273.14f) {
            q2sat = __fadd_rn(__fmul_rn(q2sat, __fsub_rn(1.0f, snowc)),
                              __fmul_rn(q2sati, snowc));
            dqsdt2 = __fadd_rn(__fmul_rn(dqsdt2, __fsub_rn(1.0f, snowc)),
                               __fmul_rn(dqi, snowc));
        } else {
            q2sat = q2sati;
            dqsdt2 = dqi;
        }
        if (t1 > 273.0f && noah_f2d(snowc) > 0.0)
            dqsdt2 = __fmul_rn(dqsdt2, __fsub_rn(1.0f, snowc));
    }
    real prcp = __fdiv_rn(rainbl_a[idx], dt);
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;
    const real tbot = 271.36f;
    real stc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) stc[k] = tslb_a[k * plane + idx];
    real ch = chs_a[idx];
    real ribb = rib_a[idx];

    // ---------------- SFLX_SEAICE (module_sf_noah_seaice.F:32-541)
    real snomlt = 0.0f;
    real zsoil[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        zsoil[k] = -__fdiv_rn(__fmul_rn(sithick, (real)(k + 1)), (real)NSOIL);
    real z0brd = 0.001f;
    bool snowng = false, frzgra = false;
#if GPUWM_SEAICE_OPTIONS
    const real snow_floor = albedo_opt == 1 ? 0.0001f : 0.01f;
    const real depth_floor = albedo_opt == 1 ? 0.0005f : 0.05f;
    if (sneqv < snow_floor) {
        sneqv = snow_floor;
        snowh = depth_floor;
    }
#else
    if (sneqv < 0.01f) {                            // albedo_opt 0 floor
        sneqv = 0.01f;
        snowh = 0.05f;
    }
#endif
#if GPUWM_SEAICE_OPTIONS
    if (snowdepth_opt == 1) {
        snowh = snowonsi;
        sneqv = __fmul_rn(snowh, 0.3f);
    } else
#endif
    nsice_clamp_depth(sneqv, snowh, snowdepth_min, snowdepth_max);
    real sndens = __fdiv_rn(sneqv, snowh);
    // WRF: FATAL_ERROR('Physical snow depth is less than snow water equiv.')
    // A device cannot abort; the launcher raises with WRF's words.
    if (sndens > 1.0f) atomicExch(fatal, 1);
    real sncond = nglac_csnow(sndens);
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= NTFREEZ) frzgra = true;
    }
    if (snowng || frzgra) {
        real sn_new = __fmul_rn(__fmul_rn(prcp, dt), 0.001f);
        sneqv = __fadd_rn(sneqv, sn_new);
        nglac_snow_new(sfctmp, sn_new, snowh, sndens);
        if (sncovr > 0.99f) {
            if (stc[0] < __fsub_rn(NTFREEZ, 5.0f)) sndens = 0.2f;
            if (snowng && t1 < 273.0f && sfctmp < 273.0f) sndens = 0.2f;
        }
        sncond = nglac_csnow(sndens);
    }
    real albedo;
#if GPUWM_SEAICE_OPTIONS
    if (albedo_opt == 1) {
        real alb_snow = t1 < 268.15f ? 0.8f : (t1 < 273.15f
            ? __fsub_rn(0.65f, __fmul_rn(0.03f, __fsub_rn(t1, 273.15f))) : 0.65f);
        real alb_ice = sfctmp <= 273.15f ? 0.65f : (sfctmp < 278.15f
            ? __fsub_rn(0.65f, __fmul_rn(0.04f, __fsub_rn(sfctmp, 273.15f))) : 0.45f);
        real sncovrr = __fdiv_rn(snowh, __fadd_rn(snowh, 0.10f));
        albedo = __fadd_rn(__fmul_rn(sncovrr, alb_snow),
                          __fmul_rn(__fsub_rn(1.0f, sncovrr), alb_ice));
    } else {
        sncovr = 1.0f;
        emissi = 0.98f;
        albedo = albedo_opt == 2 ? albsi_a[idx] : albedo_default;
    }
#else
    sncovr = 1.0f;
    emissi = 0.98f;
    albedo = albedo_default;
#endif
    real df1 = 2.2f;
    real dsoil = -__fmul_rn(0.5f, zsoil[0]);
    real dtot = __fadd_rn(snowh, dsoil);
    real frcsno = __fdiv_rn(snowh, dtot);
    real frcsoi = __fdiv_rn(dsoil, dtot);
    real df1a = __fadd_rn(__fmul_rn(frcsno, sncond), __fmul_rn(frcsoi, df1));
    df1 = __fadd_rn(__fmul_rn(df1a, sncovr),
                    __fmul_rn(df1, __fsub_rn(1.0f, sncovr)));
    real ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
    // SNOWZ0 (:1206-1235), weighted by SNCOVR
    real burial = __fsub_rn(__fmul_rn(7.0f, z0brd), snowh);
    real z0eff = (burial <= 0.0007f) ? 0.001f : __fdiv_rn(burial, 7.0f);
    real z0 = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, sncovr), z0brd),
                        __fmul_rn(sncovr, z0eff));
    real fdown = __fadd_rn(solnet, lwdn);
    real t2v = __fmul_rn(sfctmp, __fadd_rn(1.0f, __fmul_rn(0.61f, q2)));
    real t24 = __fmul_rn(__fmul_rn(__fmul_rn(sfctmp, sfctmp), sfctmp), sfctmp);
    real rho = __fdiv_rn(sfcprs, __fmul_rn(NRD, t2v));
    real rch = __fmul_rn(__fmul_rn(rho, NCP_PEN), ch);
    // PENMAN (:706-778)
    real elcp1, lvs;
    if (t1 > 273.15f) {
        elcp1 = NELCP;
        lvs = NLSUBC;
    } else {
        elcp1 = __fdiv_rn(__fmul_rn(NELCP, NLSUBS), NLSUBC);
        lvs = NLSUBS;
    }
    real flx2 = 0.0f;
    real delta = __fmul_rn(elcp1, dqsdt2);
    real rr = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(emissi, t24), 6.48e-8f),
                                  __fmul_rn(sfcprs, ch)), 1.0f);
    if (prcp > 0.0f) {
        if (!snowng) rr = __fadd_rn(rr, __fdiv_rn(__fmul_rn(NCPH2O, prcp), rch));
        else rr = __fadd_rn(rr, __fdiv_rn(__fmul_rn(NCPICE, prcp), rch));
    }
    real fnet = __fsub_rn(__fsub_rn(fdown,
                    __fmul_rn(__fmul_rn(emissi, NSIGMA), t24)), ssoil);
    if (frzgra) {
        flx2 = __fmul_rn(-NLSUBF, prcp);
        fnet = __fsub_rn(fnet, flx2);
    }
    real rad = __fsub_rn(__fadd_rn(__fdiv_rn(fnet, rch), th2), sfctmp);
    real a = __fmul_rn(elcp1, __fsub_rn(q2sat, q2));
    real epsca = __fdiv_rn(__fadd_rn(__fmul_rn(a, rr), __fmul_rn(rad, delta)),
                           __fadd_rn(delta, rr));
    real etp = __fdiv_rn(__fmul_rn(epsca, rch), lvs);
    // SNOPAC (:811-1075)
    const real esdmin = 1.0e-6f;
    real esnow = 0.0f, esnow2 = 0.0f, etanrg;
    if (etp <= 0.0f) {
        if (ribb >= 0.1f && fdown > 150.0f) {
            etp = __fdiv_rn(__fadd_rn(
                      __fdiv_rn(__fmul_rn(fminf(__fmul_rn(etp,
                                                          __fsub_rn(1.0f, ribb)),
                                                0.0f), sncovr), 0.980f),
                      __fmul_rn(etp, __fsub_rn(0.980f, sncovr))), 0.980f);
        }
        real etp1 = __fmul_rn(etp, 0.001f);
        esnow2 = __fmul_rn(etp1, dt);
        etanrg = __fmul_rn(etp, __fadd_rn(__fmul_rn(__fsub_rn(1.0f, sncovr),
                                                    NLSUBC),
                                          __fmul_rn(sncovr, NLSUBS)));
    } else {
        esnow = __fmul_rn(etp, sncovr);
        real esnow1 = __fmul_rn(esnow, 0.001f);
        esnow2 = __fmul_rn(esnow1, dt);
        etanrg = __fmul_rn(esnow, NLSUBS);
    }
    real flx1 = 0.0f;
    if (snowng) flx1 = __fmul_rn(__fmul_rn(NCPICE, prcp), __fsub_rn(t1, sfctmp));
    else if (prcp > 0.0f)
        flx1 = __fmul_rn(__fmul_rn(NCPH2O, prcp), __fsub_rn(t1, sfctmp));
    dsoil = -__fmul_rn(0.5f, zsoil[0]);
    dtot = __fadd_rn(snowh, dsoil);
    real denom = __fadd_rn(1.0f, __fdiv_rn(df1, __fmul_rn(__fmul_rn(dtot, rr), rch)));
    real t12a = __fdiv_rn(__fsub_rn(__fsub_rn(__fadd_rn(__fdiv_rn(
                    __fsub_rn(__fsub_rn(__fsub_rn(fdown, flx1), flx2),
                              __fmul_rn(__fmul_rn(emissi, NSIGMA), t24)), rch),
                    th2), sfctmp), __fdiv_rn(etanrg, rch)), rr);
    real t12b = __fdiv_rn(__fmul_rn(df1, stc[0]),
                          __fmul_rn(__fmul_rn(dtot, rr), rch));
    real t12 = __fdiv_rn(__fadd_rn(__fadd_rn(sfctmp, t12a), t12b), denom);
    real esd = sneqv;
    real flx3 = 0.0f;
    if (t12 <= NTFREEZ) {
        t1 = t12;
        ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
        esd = fmaxf(0.0f, __fsub_rn(esd, esnow2));
    } else {
        t1 = NTFREEZ;
        ssoil = __fdiv_rn(__fmul_rn(df1, __fsub_rn(t1, stc[0])), dtot);
        if (__fsub_rn(esd, esnow2) <= esdmin) {
            esd = 0.0f;
        } else {
            esd = __fsub_rn(esd, esnow2);
            real seh = __fmul_rn(rch, __fsub_rn(t1, th2));
            real t14 = __fmul_rn(__fmul_rn(t1, t1), __fmul_rn(t1, t1));
            flx3 = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(
                       __fsub_rn(fdown, flx1), flx2),
                       __fmul_rn(__fmul_rn(emissi, NSIGMA), t14)), ssoil),
                       seh), etanrg);
            if (flx3 <= 0.0f) flx3 = 0.0f;
            real ex = __fdiv_rn(__fmul_rn(flx3, 0.001f), NLSUBF);
            snomlt = __fmul_rn(ex, dt);
            if (__fsub_rn(esd, snomlt) >= esdmin) {
                esd = __fsub_rn(esd, snomlt);
            } else {
                ex = __fdiv_rn(esd, dt);
                flx3 = __fmul_rn(__fmul_rn(ex, 1000.0f), NLSUBF);
                snomlt = esd;
                esd = 0.0f;
            }
        }
    }
    real yy = __fsub_rn(stc[0], __fdiv_rn(__fmul_rn(__fmul_rn(
                  __fmul_rn(0.5f, ssoil), zsoil[0]), 1.0f), df1));
    {
        real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
        nsice_hrtice(rhsts, stc, tbot, zsoil, yy, 1.0f, df1, ai, bi, ci);
        nice_hstep(stc, dt, rhsts, ai, bi, ci);
    }
#if GPUWM_SEAICE_OPTIONS
    if (esd >= snow_floor) {
#else
    if (esd >= 0.01f) {
#endif                             // albedo_opt 0
        nglac_snowpack(esd, dt, snowh, sndens, t1, yy);
    } else {
#if GPUWM_SEAICE_OPTIONS
        esd = snow_floor;
        snowh = depth_floor;
        sncovr = albedo_opt == 1 ? 0.005f : 1.0f;
#else
        esd = 0.01f;
        snowh = 0.05f;
        sncovr = 1.0f;
#endif
    }
    sneqv = esd;
    // SFLX_SEAICE epilogue
    real eta_kin = etp;
#if GPUWM_SEAICE_OPTIONS
    if (snowdepth_opt == 1) {
        snowh = snowonsi;
        sneqv = __fmul_rn(snowh, 0.3f);
        snowsi_a[idx] = snowonsi;
    } else
#endif
    nsice_clamp_depth(sneqv, snowh, snowdepth_min, snowdepth_max);
    real q1 = __fadd_rn(q2, __fdiv_rn(__fmul_rn(eta_kin, CP), rch));
    real sheat = __fmul_rn(__fdiv_rn(-__fmul_rn(__fmul_rn(ch, CP), sfcprs),
                                     __fmul_rn(NR_SHEAT, t2v)),
                           __fsub_rn(th2, t1));
    esnow = __fmul_rn(esnow, NLSUBS);
    etp = __fmul_rn(etp, __fadd_rn(__fmul_rn(__fsub_rn(1.0f, sncovr), NLVH2O),
                                   __fmul_rn(sncovr, NLSUBS)));
    real eta = (etp > 0.0f) ? esnow : etp;
    ssoil = __fmul_rn(-1.0f, ssoil);
    real runoff1 = __fdiv_rn(snomlt, dt);

    // ---------------- seaice_noah post-update
    albedo_a[idx] = albedo;
    emiss_a[idx] = emissi;
    tsk_a[idx] = t1;
    z0_a[idx] = z0brd;
    snowh_a[idx] = snowh;
    snowc_a[idx] = sncovr;
    snow_a[idx] = __fmul_rn(sneqv, 1000.0f);
    for (int k = 0; k < NSOIL; ++k) tslb_a[k * plane + idx] = stc[k];
    znt_a[idx] = z0;
    lh_a[idx] = eta;
    hfx_a[idx] = sheat;
    qfx_a[idx] = eta_kin;
    potevp_a[idx] = __fadd_rn(potevp_a[idx], __fmul_rn(etp,
                        __fdiv_rn(dt, __fmul_rn(XLV, RHOWATER))));
    grdflx_a[idx] = ssoil;
    chs2_a[idx] = cqs2_a[idx];
    if (q1 > qsfc_a[idx]) cqs2_a[idx] = chs_a[idx];
    qsfc_a[idx] = __fdiv_rn(q1, __fsub_rn(1.0f, q1));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = __fadd_rn(acsnow_a[idx], __fmul_rn(prcp, dt));
    acsnom_a[idx] = __fadd_rn(acsnom_a[idx], __fmul_rn(snomlt, 1000.0f));
    snopcx_a[idx] = __fsub_rn(snopcx_a[idx], __fdiv_rn(snomlt,
                        __fdiv_rn(dt, __fmul_rn(1000.0f, NXLF))));
    sfcrunoff_a[idx] = __fadd_rn(sfcrunoff_a[idx],
                                 __fmul_rn(__fmul_rn(runoff1, dt), 1000.0f));
    real t14 = __fmul_rn(__fmul_rn(t1, t1), __fmul_rn(t1, t1));
    noahres_a[idx] = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(
                         __fadd_rn(__fsub_rn(__fadd_rn(solnet, lwdn), sheat),
                                   ssoil), eta),
                         __fmul_rn(__fmul_rn(emissi, NSTBOLT), t14)),
                         flx1), flx2), flx3);
}

extern "C" __global__ __launch_bounds__(64, 8)
void noah_column_fasdas(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
#if NOAH_UA_PHYS
                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a,
#endif
 const real* fasdas_h, const real* fasdas_q, const real* fasdas_norm)
{
    noah_column_body<false, true>(ivgtyp, isltyp, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                            qgh_a, dz8w1_a, glw_a, swdown_a, rainbl_a, sr_a,
                            chs_a, cqs2_a, chs2_a, rib_a, vegfra_a, shdmin_a,
                            shdmax_a, tmn_a, xland_a, xice_a, snoalb_a,
                            embck_a, tsk_a, hfx_a, qfx_a, lh_a, grdflx_a,
                            qsfc_a, canwat_a, snow_a, snowc_a, snowh_a,
                            albedo_a, albbck_a, emiss_a, znt_a, z0_a,
                            snotime_a, lai_a, smstav_a, smstot_a, sfcrunoff_a,
                            udrunoff_a, acsnow_a, acsnom_a, snopcx_a,
                            potevp_a, noahres_a, reslin_a, chklowq_a, smois_a,
                            tslb_a, sh2o_a, smcrel_a, ebal_a, vegtbl, soiltbl,
                            genp, dzs, dt, lucats, slcats, isurban, isice,
                            xice_threshold, itimestep, frpcpn, usemonalb,
                            rdlai2d, opt_thcnd, ny, nx,
#if NOAH_UA_PHYS
                            canopy, flx4_a, fvb_a, fbur_a, fgsn_a,
#endif
 0, nullptr, 0, 0,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, fasdas_h, fasdas_q, fasdas_norm);
}

extern "C" __global__ __launch_bounds__(64, 8)
void noah_column_urban_fasdas(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
#if NOAH_UA_PHYS
                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a,
#endif

                 // URBAN HANDOVER (sf_urban_physics > 0), read only by the
                 // URBAN instantiation; noah_column passes zeros.
                 int urban_opt,
                 const int* __restrict__ urban_cat,   // category -> 0/1
                 int urban_cat_size, int natural,
                 const real* __restrict__ frc_urb2d,
                 const real* __restrict__ ts_urb2d,
                 real* __restrict__ tsk_rural_bep,    // options 2/3 only
                 real* __restrict__ rural_q1,
                 real* __restrict__ rural_q2k,
                 real* __restrict__ rural_zlvl, const real* fasdas_h, const real* fasdas_q, const real* fasdas_norm)
{
    noah_column_body<true, true>(ivgtyp, isltyp, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                           qgh_a, dz8w1_a, glw_a, swdown_a, rainbl_a, sr_a,
                           chs_a, cqs2_a, chs2_a, rib_a, vegfra_a, shdmin_a,
                           shdmax_a, tmn_a, xland_a, xice_a, snoalb_a,
                           embck_a, tsk_a, hfx_a, qfx_a, lh_a, grdflx_a,
                           qsfc_a, canwat_a, snow_a, snowc_a, snowh_a,
                           albedo_a, albbck_a, emiss_a, znt_a, z0_a,
                           snotime_a, lai_a, smstav_a, smstot_a, sfcrunoff_a,
                           udrunoff_a, acsnow_a, acsnom_a, snopcx_a, potevp_a,
                           noahres_a, reslin_a, chklowq_a, smois_a, tslb_a,
                           sh2o_a, smcrel_a, ebal_a, vegtbl, soiltbl, genp,
                           dzs, dt, lucats, slcats, isurban, isice,
                           xice_threshold, itimestep, frpcpn, usemonalb,
                           rdlai2d, opt_thcnd, ny, nx,
#if NOAH_UA_PHYS
                            canopy, flx4_a, fvb_a, fbur_a, fgsn_a,
#endif
 urban_opt, urban_cat,
                           urban_cat_size, natural, frc_urb2d, ts_urb2d,
                           tsk_rural_bep, rural_q1, rural_q2k, rural_zlvl, fasdas_h, fasdas_q, fasdas_norm);
}

extern "C" __global__ __launch_bounds__(64, 8)
void noah_column_irrigation(const int* __restrict__ ivgtyp,
                 const int* __restrict__ isltyp,
                 const real* __restrict__ psfc_a,
                 const real* __restrict__ sfcprs_a,
                 const real* __restrict__ sfctmp_a,
                 const real* __restrict__ qv1_a,
                 const real* __restrict__ qgh_a,
                 const real* __restrict__ dz8w1_a,
                 const real* __restrict__ glw_a,
                 const real* __restrict__ swdown_a,
                 const real* __restrict__ rainbl_a,
                 const real* __restrict__ sr_a,
                 const real* __restrict__ chs_a,
                 const real* __restrict__ cqs2_a,
                 real* __restrict__ chs2_a,
                 const real* __restrict__ rib_a,
                 const real* __restrict__ vegfra_a,
                 const real* __restrict__ shdmin_a,
                 const real* __restrict__ shdmax_a,
                 const real* __restrict__ tmn_a,
                 const real* __restrict__ xland_a,
                 const real* __restrict__ xice_a,
                 const real* __restrict__ snoalb_a,
                 const real* __restrict__ embck_a,
                 real* __restrict__ tsk_a,
                 real* __restrict__ hfx_a,
                 real* __restrict__ qfx_a,
                 real* __restrict__ lh_a,
                 real* __restrict__ grdflx_a,
                 real* __restrict__ qsfc_a,
                 real* __restrict__ canwat_a,
                 real* __restrict__ snow_a,
                 real* __restrict__ snowc_a,
                 real* __restrict__ snowh_a,
                 real* __restrict__ albedo_a,
                 real* __restrict__ albbck_a,
                 real* __restrict__ emiss_a,
                 real* __restrict__ znt_a,
                 real* __restrict__ z0_a,
                 real* __restrict__ snotime_a,
                 real* __restrict__ lai_a,
                 real* __restrict__ smstav_a,
                 real* __restrict__ smstot_a,
                 real* __restrict__ sfcrunoff_a,
                 real* __restrict__ udrunoff_a,
                 real* __restrict__ acsnow_a,
                 real* __restrict__ acsnom_a,
                 real* __restrict__ snopcx_a,
                 real* __restrict__ potevp_a,
                 real* __restrict__ noahres_a,
                 real* __restrict__ reslin_a,
                 real* __restrict__ chklowq_a,
                 real* __restrict__ smois_a,      // (4, ny, nx)
                 real* __restrict__ tslb_a,
                 real* __restrict__ sh2o_a,
                 real* __restrict__ smcrel_a,
                 int* __restrict__ ebal_a,
                 const real* __restrict__ vegtbl,   // (lucats, NVEGC)
                 const real* __restrict__ soiltbl,  // (slcats, NSOILC)
                 const real* __restrict__ genp,     // (16,)
                 const real* __restrict__ dzs,      // (4,)
                 real dt, int lucats, int slcats, int isurban,
                 int isice, real xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
#if NOAH_UA_PHYS
                 const real* canopy, real* flx4_a, real* fvb_a, real* fbur_a, real* fgsn_a,
#endif
 const real* irrigation_channel_a)
{
    noah_column_body<false, false, true>(ivgtyp, isltyp, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                            qgh_a, dz8w1_a, glw_a, swdown_a, rainbl_a, sr_a,
                            chs_a, cqs2_a, chs2_a, rib_a, vegfra_a, shdmin_a,
                            shdmax_a, tmn_a, xland_a, xice_a, snoalb_a,
                            embck_a, tsk_a, hfx_a, qfx_a, lh_a, grdflx_a,
                            qsfc_a, canwat_a, snow_a, snowc_a, snowh_a,
                            albedo_a, albbck_a, emiss_a, znt_a, z0_a,
                            snotime_a, lai_a, smstav_a, smstot_a, sfcrunoff_a,
                            udrunoff_a, acsnow_a, acsnom_a, snopcx_a,
                            potevp_a, noahres_a, reslin_a, chklowq_a, smois_a,
                            tslb_a, sh2o_a, smcrel_a, ebal_a, vegtbl, soiltbl,
                            genp, dzs, dt, lucats, slcats, isurban, isice,
                            xice_threshold, itimestep, frpcpn, usemonalb,
                            rdlai2d, opt_thcnd, ny, nx,
#if NOAH_UA_PHYS
                            canopy, flx4_a, fvb_a, fbur_a, fgsn_a,
#endif
 0, nullptr, 0, 0,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, irrigation_channel_a);
}
