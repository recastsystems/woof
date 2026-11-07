// Eta Ferrier microphysics (WRF mp_physics = 95, package ETAMPNEW).
//
// Transcription authority: byte-unmodified WRF v4.6.1
// phys/module_mp_etanew.F (2,676 lines, commit d66e442).  Every block below
// cites the Fortran lines it carries.  The operation order is the Fortran's
// left-to-right evaluation as gfortran lowers it with no reassociation:
//
//   * real exponents other than 2.0, -1.0 and 1.0 go through WOOF's own
//     float32 pow (gfk_pow): TK**1.5, TK**1.81, (1.E5/PP)**C1, x**.5,
//     XLI**.42066, RR**.1947 and the rest; EXP and ALOG go through gfk_exp
//     and gfk_log (glibc_flt32.cuh, WOOF's float32 math routines);
//   * integer exponents are multiplication chains;
//   * constant expressions gfortran folds at compile time (PARAMETER
//     arithmetic, and powers of literals such as (1.E5)**C1 and
//     1.1**Nrime) are folded once per operation in binary32, the way the
//     front end folds them; the five folded POWERS are carried as exact
//     hexadecimal literals, each recomputed by
//     tests/test_etanew_mp95.py with exact rational arithmetic and
//     pinned by the init oracle;
//   * Fortran MAX(a, b) is (a > b ? a : b) and MIN(a, b) is (a < b ? a : b),
//     the argument order and the signed-zero / NaN behaviour gfortran 13
//     produces (measured on the oracle host with -O0 and -O2);
//   * INT() truncates, with x86's 0x80000000 for NaN and out-of-range input.
//
// CONDENSE and DEPOSIT (module_mp_etanew.F:1981-2076) carry COND, SSAT,
// WCdum and DEP in REAL(KIND=8) exactly as the Fortran declares them; the
// mixed-kind assignments round where the Fortran rounds.
//
// The kernel is compiled with -fmad=false and IEEE division in strict mode
// (GPUWM_WRF_EXACT=1: gpuwm.wrf_exact.STRICT_OPTIONS); the default mode
// compiles the same source with the same non-contracting arithmetic.
//
// CONDENSE and DEPOSIT retain WRF's convergence conditions and iteration
// order.  No partial adjustment is returned from an artificial iteration cap.
//
// Layout: every 3-D field is (nz, ny, nx) C-order, level k = 0 at the
// surface (WRF's kts); a column is one thread.  WRF's microphysics flips the
// column so L = 1 is the model top (EGCP01DRV, :408-422); the level loop here
// walks L = 1..nz, k = nz - L, and streams each level's driver preparation,
// column physics and driver write-back in that order, which is the order of
// every data dependence the Fortran has (a level reads only its own state,
// the precipitation fluxes ARAIN/ASNOW falling from above, and the rime
// factor of the level above, RimeF_col(L-1), :1101-1102).

#define ETA_NX 7501

// --- table offsets (ETAMPNEW_DATA.expanded_rain record order, :2269-2280)
#define ETA_NR 951            // MDRmin:MDRmax = 50:1000 (also MDImin:MDImax)
#define ETA_VENTR1 0
#define ETA_VENTR2 (1 * ETA_NR)
#define ETA_ACCRR (2 * ETA_NR)
#define ETA_MASSR (3 * ETA_NR)
#define ETA_VRAIN (4 * ETA_NR)
#define ETA_RRATE (5 * ETA_NR)
#define ETA_VENTI1 (6 * ETA_NR)
#define ETA_VENTI2 (7 * ETA_NR)
#define ETA_ACCRI (8 * ETA_NR)
#define ETA_MASSI (9 * ETA_NR)
#define ETA_VSNOWI (10 * ETA_NR)
#define ETA_VEL_RF (11 * ETA_NR)    // VEL_RF(2:9, 0:Nrime), column-major
#define ETA_TABLE_WORDS (11 * ETA_NR + 8 * 41)

// --- MP_RESTART_STATE layout (:161-170, :308-316)
#define ETA_RS_MY_GROWTH 0          // MY_GROWTH(1:35)
#define ETA_RS_C1XPVS0 35
#define ETA_RS_C2XPVS0 36
#define ETA_RS_C1XPVS 37
#define ETA_RS_C2XPVS 38
#define ETA_RS_CIACW 39
#define ETA_RS_CIACR 40
#define ETA_RS_CRACW 41
#define ETA_RS_CRAUT 42
#define ETA_RS_WORDS 43

// --- module constants ETANEWinit sets that are NOT in the restart vector
#define ETA_K_ABFR 0
#define ETA_K_CBFR 1
#define ETA_K_RR_DRMIN 2
#define ETA_K_RR_DR1 3
#define ETA_K_RR_DR2 4
#define ETA_K_RR_DR3 5
#define ETA_K_RR_DR4 6
#define ETA_K_RR_DR5 7
#define ETA_K_RR_DRMAX 8
#define ETA_K_RQR_DRMIN 9
#define ETA_K_RQR_DRMAX 10
#define ETA_K_CN0R0 11
#define ETA_K_CN0R_DMRMIN 12
#define ETA_K_CN0R_DMRMAX 13
#define ETA_K_ESW0 14
#define ETA_K_RFMAX 15
#define ETA_K_QAUT0 16
#define ETA_K_WORDS 17

// --- integer parameters (:12-31, :53-54)
#define ETA_MY_T1 1
#define ETA_MY_T2 35
#define ETA_MDIMIN 50
#define ETA_MDIMAX 1000
#define ETA_MDRMIN 50
#define ETA_MDRMAX 1000
#define ETA_MDR1 100
#define ETA_MDR2 200
#define ETA_MDR3 320
#define ETA_MDR4 450
#define ETA_MDR5 670
#define ETA_NRIME 40

// --- PARAMETER constants (:36-82), folded one binary32 operation at a time
#define ETA_CP 1004.6f
#define ETA_EPSQ 1.E-12f
#define ETA_GRAV 9.806f
#define ETA_RHOL 1000.f
#define ETA_RD 287.04f
#define ETA_RV 461.5f
#define ETA_T0C 273.15f
#define ETA_XLS 2.834E6f
#define ETA_T_ICE (-40.f)
#define ETA_T_ICE_INIT (-5.f)
#define ETA_NLIMAX 5.E3f
#define ETA_NLIMIN 1.E3f
#define ETA_N0R0 8.E6f
#define ETA_N0RMIN 1.E4f
#define ETA_NCW 250.E6f
#define ETA_FLARGE1 1.f
#define ETA_FLARGE2 .2f
#define ETA_RHGRD 1.f
#define ETA_TOLER 5.E-7f
#define ETA_RHO0 1.194f
#define ETA_XRATIO .025f
#define ETA_BLEND 1.f
#define ETA_XMRMIN 50.f                       // 1.e6*DMRmin, 0x42480000
#define ETA_XMRMAX 0x1.f40002p+9f             // 1.e6*DMRmax, 0x447A0001

// Folded powers of literals (exact rational arithmetic, see the header):
#define ETA_DMRMIN4 0x1.cd2b26p-58f           // DMRmin**4, 0x22E69593
#define ETA_DMRMAX4 0x1.19799cp-40f           // DMRmax**4, 0x2B8CBCCE
#define ETA_RFMAX_FOLD 0x1.6a130ap+5f         // 1.1**Nrime, 0x42350985
#define ETA_C20E6_3 0x1.203af8p-47f           // (20.E-6)**3, 0x28101D7C
#define ETA_1E5_POW_C1 0x1.7353c0p+5f         // (1.E5)**C1, 0x4239A9E0
#define ETA_PI 0x1.921fb6p+1f                 // ACOS(-1.), 0x40490FDB

struct EtaParams {
    float EPS, EPS1, EPSQ1, rcp, RCPRV, RGRAV, RRHOL, XLS1, XLS2, XLS3;
    float CLIMIT, C1, C2, T_ICEK;
};

__device__ __forceinline__ EtaParams eta_params()
{
    // Intrinsics keep these out of any multiply-add contraction in the
    // default build, so both modes see the same folded constants.
    EtaParams q;
    q.EPS = __fdiv_rn(ETA_RD, ETA_RV);                        // RD/RV
    q.EPS1 = __fsub_rn(__fdiv_rn(ETA_RV, ETA_RD), 1.f);       // RV/RD-1.
    q.EPSQ1 = __fmul_rn(1.001f, ETA_EPSQ);                    // 1.001*EPSQ
    q.rcp = __fdiv_rn(1.f, ETA_CP);                           // 1./CP
    q.RCPRV = __fdiv_rn(q.rcp, ETA_RV);                       // RCP/RV
    q.RGRAV = __fdiv_rn(1.f, ETA_GRAV);                       // 1./GRAV
    q.RRHOL = __fdiv_rn(1.f, ETA_RHOL);                       // 1./RHOL
    q.XLS1 = __fmul_rn(ETA_XLS, q.rcp);                       // XLS*RCP
    q.XLS2 = __fmul_rn(__fmul_rn(ETA_XLS, ETA_XLS), q.RCPRV); // XLS*XLS*RCPRV
    q.XLS3 = __fdiv_rn(__fmul_rn(ETA_XLS, ETA_XLS), ETA_RV);  // XLS*XLS/RV
    q.CLIMIT = __fmul_rn(10.f, ETA_EPSQ);                     // 10.*EPSQ
    q.C1 = __fdiv_rn(1.f, 3.f);                               // 1./3.
    q.C2 = __fdiv_rn(1.f, 6.f);                               // 1./6.
    q.T_ICEK = __fadd_rn(ETA_T0C, ETA_T_ICE);                 // T0C+T_ICE
    return q;
}

// gfortran 13 MAX / MIN on REAL (argument order preserved).
__device__ __forceinline__ float fmaxf_wrf(float a, float b)
{
    return (a > b) ? a : b;
}
__device__ __forceinline__ float fminf_wrf(float a, float b)
{
    return (a < b) ? a : b;
}
__device__ __forceinline__ double fmax_wrf(double a, double b)
{
    return (a > b) ? a : b;
}

// INT(x) for REAL x: truncation; cvttss2si's 0x80000000 for NaN and for
// anything outside the int range.
__device__ __forceinline__ int fint_wrf(float x)
{
    if (!(x > -2147483648.f && x < 2147483648.f)) {
        return (int)0x80000000;
    }
    return (int)x;
}

__device__ __forceinline__ int imin_wrf(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax_wrf(int a, int b) { return a > b ? a : b; }

// FPVS / FPVS0 (:2541-2596): linear interpolation in the saturation tables.
__device__ __forceinline__ float eta_fpvs(float t, float c1, float c2,
                                          const float* __restrict__ tb)
{
    float xj = fminf_wrf(fmaxf_wrf(c1 + c2 * t, 1.f), (float)ETA_NX);
    int jx = fint_wrf(fminf_wrf(xj, (float)ETA_NX - 1.f));
    float lo = tb[jx - 1];
    return lo + (xj - (float)jx) * (tb[jx] - lo);
}

// FPVSX / FPVSX0 (:2600-2674): the exact saturation pressure the tables hold.
__device__ float eta_fpvsx(float t, bool water_only)
{
    const float TTP = 2.7316E+2f, HVAP = 2.5000E+6f, PSAT = 6.1078E+2f;
    const float CLIQ = 4.1855E+3f, CVAP = 1.8460E+3f, CICE = 2.1060E+3f;
    const float HSUB = 2.8340E+6f;
    const float PSATK = __fmul_rn(PSAT, 1.E-3f);
    const float DLDT = __fsub_rn(CVAP, CLIQ);
    const float XA = -__fdiv_rn(DLDT, ETA_RV);
    const float XB = __fadd_rn(XA, __fdiv_rn(HVAP, __fmul_rn(ETA_RV, TTP)));
    const float DLDTI = __fsub_rn(CVAP, CICE);
    const float XAI = -__fdiv_rn(DLDTI, ETA_RV);
    const float XBI = __fadd_rn(XAI, __fdiv_rn(HSUB, __fmul_rn(ETA_RV, TTP)));
    float tr = TTP / t;
    if (water_only || t >= TTP) {
        return PSATK * gfk_pow(tr, XA) * gfk_exp(XB * (1.f - tr));
    }
    return PSATK * gfk_pow(tr, XAI) * gfk_exp(XBI * (1.f - tr));
}

// GPVS (:2489-2537): one thread per table entry.
extern "C" __global__ void etanew_gpvs(float* __restrict__ tbpvs,
                                       float* __restrict__ tbpvs0)
{
    int jx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (jx > ETA_NX) return;
    const float XMIN = 180.0f;
    const float XINC = __fdiv_rn(150.f, 7500.f);   // (XMAX-XMIN)/(NX-1)
    float x = XMIN + (float)(jx - 1) * XINC;
    tbpvs[jx - 1] = eta_fpvsx(x, false);
    tbpvs0[jx - 1] = eta_fpvsx(x, true);
}

// The scalar part of ETANEWinit (:2203-2423) and MY_GROWTH_RATES
// (:2440-2483), after etanew_gpvs has filled the tables.  One thread.
// rs: MP_RESTART_STATE(1:43); k: the non-restart module constants;
// dt_slot[0]: the DTPH the restart vector was last built for.  With
// only_if_changed != 0 the kernel returns at once when dt_slot[0] already
// equals this call's DTPH, so the forecast adapter can ask every step
// without a host round trip.
extern "C" __global__ void etanew_constants(
    const float* __restrict__ tables, const float* __restrict__ tbpvs0,
    float dt, float gsmdt, float* __restrict__ rs, float* __restrict__ k,
    float* __restrict__ dt_slot, int only_if_changed)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const EtaParams q = eta_params();
    // DTPH=MAX(GSMDT*60.,DT); DTPH=NINT(DTPH/DT)*DT (:2237-2238)
    float dtph = fmaxf_wrf(gsmdt * 60.f, dt);
    dtph = (float)(int)lroundf(dtph / dt) * dt;
    if (only_if_changed && dt_slot[0] == dtph) return;
    dt_slot[0] = dtph;
    // GPVS scalars (:2524-2528)
    const float XINC = __fdiv_rn(150.f, 7500.f);
    float c1xpvs = 1.f - 180.0f / XINC;
    float c2xpvs = 1.f / XINC;
    rs[ETA_RS_C1XPVS0] = c1xpvs;
    rs[ETA_RS_C2XPVS0] = c2xpvs;
    rs[ETA_RS_C1XPVS] = c1xpvs;
    rs[ETA_RS_C2XPVS] = c2xpvs;
    // MY_GROWTH_RATES (:2463-2479)
    const float MY_600[35] = {
        5.5e-8f, 1.4E-7f, 2.8E-7f, 6.E-7f, 3.3E-6f,
        2.E-6f, 9.E-7f, 8.8E-7f, 8.2E-7f, 9.4e-7f,
        1.2E-6f, 1.85E-6f, 5.5E-6f, 1.5E-5f, 1.7E-5f,
        1.5E-5f, 1.E-5f, 3.4E-6f, 1.85E-6f, 1.35E-6f,
        1.05E-6f, 1.E-6f, 9.5E-7f, 9.0E-7f, 9.5E-7f,
        9.5E-7f, 9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f,
        9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f};
    if (dtph >= 0.0f) {
        float dt_ice = gfk_pow(__fdiv_rn(dtph, 600.f), 1.5f);
        for (int i = 0; i < 35; ++i) {
            rs[ETA_RS_MY_GROWTH + i] = dt_ice * MY_600[i] * 1.e-3f;
        }
    } else {
        for (int i = 0; i < 35; ++i) rs[ETA_RS_MY_GROWTH + i] = 0.0f;
    }
    const float PI = ETA_PI;
    const float* RRATE = tables + ETA_RRATE - ETA_MDRMIN;
    const float* MASSR = tables + ETA_MASSR - ETA_MDRMIN;
    // :2308-2310
    k[ETA_K_ABFR] = -0.66f;
    const float BBFR = 100.f;
    k[ETA_K_CBFR] = 20.f * PI * PI * BBFR * ETA_RHOL * 1.E-21f;
    // :2316, :2322
    rs[ETA_RS_CIACW] = 0.5f * dtph * 0.25f * PI * ETA_1E5_POW_C1;
    rs[ETA_RS_CIACR] = 1.0f * PI * dtph;
    // :2330-2346
    k[ETA_K_RR_DRMIN] = ETA_N0R0 * RRATE[ETA_MDRMIN];
    k[ETA_K_RR_DR1] = ETA_N0R0 * RRATE[ETA_MDR1];
    k[ETA_K_RR_DR2] = ETA_N0R0 * RRATE[ETA_MDR2];
    k[ETA_K_RR_DR3] = ETA_N0R0 * RRATE[ETA_MDR3];
    k[ETA_K_RR_DR4] = ETA_N0R0 * RRATE[ETA_MDR4];
    k[ETA_K_RR_DR5] = ETA_N0R0 * RRATE[ETA_MDR5];
    k[ETA_K_RR_DRMAX] = ETA_N0R0 * RRATE[ETA_MDRMAX];
    k[ETA_K_RQR_DRMIN] = ETA_N0R0 * MASSR[ETA_MDRMIN];
    k[ETA_K_RQR_DRMAX] = ETA_N0R0 * MASSR[ETA_MDRMAX];
    // :2347-2350
    float c_n0r0 = PI * ETA_RHOL * ETA_N0R0;
    k[ETA_K_CN0R0] = 1.E6f / gfk_pow(c_n0r0, .25f);
    k[ETA_K_CN0R_DMRMIN] = 1.f / (PI * ETA_RHOL * ETA_DMRMIN4);
    k[ETA_K_CN0R_DMRMAX] = 1.f / (PI * ETA_RHOL * ETA_DMRMAX4);
    // :2355
    rs[ETA_RS_CRACW] = 1.0f * dtph * 0.25f * PI;
    // :2357-2358, ESW0 from the module's own fresh tables
    k[ETA_K_ESW0] = 1000.f * eta_fpvs(ETA_T0C, c1xpvs, c2xpvs, tbpvs0);
    k[ETA_K_RFMAX] = ETA_RFMAX_FOLD;
    // :2370, :2378
    rs[ETA_RS_CRAUT] = 1.f - gfk_exp(-(1.E-3f * dtph));
    k[ETA_K_QAUT0] = __fdiv_rn(PI * ETA_RHOL * ETA_NCW * ETA_C20E6_3, 6.f);
    (void)q;
}

// CONDENSE (:1981-2031).  Returns the Fortran's REAL result.
__device__ float eta_condense(const EtaParams& q, float PP, float QW, float TK,
                              float WV, const float* __restrict__ rs,
                              const float* __restrict__ tbpvs0)
{
    const double RHLIMIT = (double).001f;
    const double RHLIMIT1 = -RHLIMIT;
    const float c1 = rs[ETA_RS_C1XPVS0], c2 = rs[ETA_RS_C2XPVS0];
    float XLVT = 3.148E6f - 2370.f * TK;
    float XLV1 = XLVT * q.rcp;
    float XLV2 = XLVT * XLVT * q.RCPRV;
    float Tdum = TK;
    float WVdum = WV;
    double WCdum = (double)QW;
    float ESW = fminf_wrf(1000.f * eta_fpvs(Tdum, c1, c2, tbpvs0), 0.99f * PP);
    float WS = ETA_RHGRD * q.EPS * ESW / (PP - ESW);
    float DWV = WVdum - WS;
    double SSAT = (double)(DWV / WS);
    float CONDENSE = 0.f;
    int n = 0;
    while (((SSAT < RHLIMIT1 && WCdum > (double)ETA_EPSQ) || SSAT > RHLIMIT)
          ) {
        ++n;
        double COND = (double)(DWV / (1.f + XLV2 * WS / (Tdum * Tdum)));
        COND = fmax_wrf(COND, -WCdum);
        Tdum = (float)((double)Tdum + (double)XLV1 * COND);
        WVdum = (float)((double)WVdum - COND);
        WCdum = WCdum + COND;
        CONDENSE = (float)((double)CONDENSE + COND);
        ESW = fminf_wrf(1000.f * eta_fpvs(Tdum, c1, c2, tbpvs0), 0.99f * PP);
        WS = ETA_RHGRD * q.EPS * ESW / (PP - ESW);
        DWV = WVdum - WS;
        SSAT = (double)(DWV / WS);
    }
    return CONDENSE;
}

// DEPOSIT (:2037-2076).  Tdum and WVdum are INTENT(INOUT) in WRF and the
// caller passes its DUM1/DUM2 scratch, which it does not read afterwards;
// they are local copies here.
__device__ float eta_deposit(const EtaParams& q, float PP, float Tdum,
                             float WVdum, const float* __restrict__ rs,
                             const float* __restrict__ tbpvs)
{
    const double RHLIMIT = (double).001f;
    const double RHLIMIT1 = -RHLIMIT;
    const float c1 = rs[ETA_RS_C1XPVS], c2 = rs[ETA_RS_C2XPVS];
    float ESI = fminf_wrf(1000.f * eta_fpvs(Tdum, c1, c2, tbpvs), 0.99f * PP);
    float WS = ETA_RHGRD * q.EPS * ESI / (PP - ESI);
    float DWV = WVdum - WS;
    double SSAT = (double)(DWV / WS);
    float DEPOSIT = 0.f;
    int n = 0;
    while ((SSAT > RHLIMIT || SSAT < RHLIMIT1)) {
        ++n;
        double DEP = (double)(DWV / (1.f + q.XLS2 * WS / (Tdum * Tdum)));
        Tdum = (float)((double)Tdum + (double)q.XLS1 * DEP);
        WVdum = (float)((double)WVdum - DEP);
        DEPOSIT = (float)((double)DEPOSIT + DEP);
        ESI = fminf_wrf(1000.f * eta_fpvs(Tdum, c1, c2, tbpvs), 0.99f * PP);
        WS = ETA_RHGRD * q.EPS * ESI / (PP - ESI);
        DWV = WVdum - WS;
        SSAT = (double)(DWV / WS);
    }
    return DEPOSIT;
}

// Mean rain-drop diameter (microns) from a rain rate, :1411-1493 / :1737-1757.
// ``with_max`` is the first form (RR above RR_DRmax pins MDRmax); the second
// form, used for VRAIN2, has no RR_DRmax arm.
__device__ __forceinline__ int eta_drop_index(float RR, const float* __restrict__ k,
                                              bool with_max)
{
    int idx;
    if (RR <= k[ETA_K_RR_DRMIN]) {
        idx = ETA_MDRMIN;
    } else if (RR <= k[ETA_K_RR_DR1]) {
        idx = fint_wrf(1.123E3f * gfk_pow(RR, .1947f) + .5f);
        idx = imax_wrf(ETA_MDRMIN, imin_wrf(idx, ETA_MDR1));
    } else if (RR <= k[ETA_K_RR_DR2]) {
        idx = fint_wrf(1.225E3f * gfk_pow(RR, .2017f) + .5f);
        idx = imax_wrf(ETA_MDR1, imin_wrf(idx, ETA_MDR2));
    } else if (RR <= k[ETA_K_RR_DR3]) {
        idx = fint_wrf(1.3006E3f * gfk_pow(RR, .2083f) + .5f);
        idx = imax_wrf(ETA_MDR2, imin_wrf(idx, ETA_MDR3));
    } else if (RR <= k[ETA_K_RR_DR4]) {
        idx = fint_wrf(1.354E3f * gfk_pow(RR, .2143f) + .5f);
        idx = imax_wrf(ETA_MDR3, imin_wrf(idx, ETA_MDR4));
    } else if (RR <= k[ETA_K_RR_DR5]) {
        idx = fint_wrf(1.404E3f * gfk_pow(RR, .2213f) + .5f);
        idx = imax_wrf(ETA_MDR4, imin_wrf(idx, ETA_MDR5));
    } else if (!with_max || RR <= k[ETA_K_RR_DRMAX]) {
        idx = fint_wrf(1.4457E3f * gfk_pow(RR, .2303f) + .5f);
        idx = imax_wrf(ETA_MDR5, imin_wrf(idx, ETA_MDRMAX));
    } else {
        idx = ETA_MDRMAX;
    }
    return idx;
}

// ETAMP_NEW (:92-323) with EGCP01DRV (:327-584) and EGCP01COLUMN
// (:638-2078), one thread per column.
//
// In/out: th (full theta), qv (water vapour mixing ratio), qt (total
// condensate, the advected scalar), f_ice, f_rain, f_rimef (the carried
// fractions), rainnc (mm).  Out: qc, qr, qs (rebuilt from qt and the
// fractions, :268-297), rainncv (mm this call), sr.  In: p (Pa), pii
// (Exner), rho (density), dz8w (layer depth, m).  rs/tbpvs/tbpvs0 are the
// restart state (MP_RESTART_STATE, TBPVS_STATE, TBPVS0_STATE); k the
// non-restart module constants; tables the ETAMPNEW_DATA arrays.
extern "C" __global__ void etanew_column(
    float* __restrict__ th, float* __restrict__ qv, float* __restrict__ qt,
    float* __restrict__ qc, float* __restrict__ qr, float* __restrict__ qs,
    float* __restrict__ f_ice, float* __restrict__ f_rain,
    float* __restrict__ f_rimef,
    const float* __restrict__ p, const float* __restrict__ pii,
    const float* __restrict__ rho, const float* __restrict__ dz8w,
    float* __restrict__ rainnc, float* __restrict__ rainncv,
    float* __restrict__ sr,
    const float* __restrict__ rs, const float* __restrict__ tbpvs,
    const float* __restrict__ tbpvs0, const float* __restrict__ k,
    const float* __restrict__ tables,
    float DTPH, int nz, int ncol)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const EtaParams q = eta_params();

    const float* VENTR1 = tables + ETA_VENTR1 - ETA_MDRMIN;
    const float* VENTR2 = tables + ETA_VENTR2 - ETA_MDRMIN;
    const float* ACCRR = tables + ETA_ACCRR - ETA_MDRMIN;
    const float* VRAIN = tables + ETA_VRAIN - ETA_MDRMIN;
    const float* VENTI1 = tables + ETA_VENTI1 - ETA_MDIMIN;
    const float* VENTI2 = tables + ETA_VENTI2 - ETA_MDIMIN;
    const float* ACCRI = tables + ETA_ACCRI - ETA_MDIMIN;
    const float* MASSI = tables + ETA_MASSI - ETA_MDIMIN;
    const float* VSNOWI = tables + ETA_VSNOWI - ETA_MDIMIN;
    const float* VEL_RF = tables + ETA_VEL_RF;   // (ixs-2) + 8*ixrf
    const float* MY_GROWTH = rs + ETA_RS_MY_GROWTH - ETA_MY_T1;

    const float CIACW = rs[ETA_RS_CIACW], CIACR = rs[ETA_RS_CIACR];
    const float CRACW = rs[ETA_RS_CRACW], CRAUT = rs[ETA_RS_CRAUT];
    const float c1x0 = rs[ETA_RS_C1XPVS0], c2x0 = rs[ETA_RS_C2XPVS0];
    const float c1x = rs[ETA_RS_C1XPVS], c2x = rs[ETA_RS_C2XPVS];
    const float ABFR = k[ETA_K_ABFR], CBFR = k[ETA_K_CBFR];
    const float RQR_DRmin = k[ETA_K_RQR_DRMIN], RQR_DRmax = k[ETA_K_RQR_DRMAX];
    const float CN0r0 = k[ETA_K_CN0R0];
    const float CN0r_DMRmin = k[ETA_K_CN0R_DMRMIN];
    const float CN0r_DMRmax = k[ETA_K_CN0R_DMRMAX];
    const float ESW0 = k[ETA_K_ESW0], RFmax = k[ETA_K_RFMAX];
    const float QAUT0 = k[ETA_K_QAUT0];
    const float EPSQ = ETA_EPSQ, CLIMIT = q.CLIMIT;

    float ARAIN = 0.f;     // :840
    float ASNOW = 0.f;     // :841
    float rimef_above = 1.f;   // RimeF_col(L-1) as the level above left it
    float QSI = 0.f;       // stale-by-design: read at TC >= 0 only where
                           // the result cannot depend on it (:886, :1057)

    for (int L = 1; L <= nz; ++L) {
        const int idx = (nz - L) * ncol + col;

        // ---- ETAMP_NEW entry (:178-179) and EGCP01DRV preparation
        //      (:410-497) for this level
        const float t_phy = th[idx] * pii[idx];
        const float q_phy = qv[idx] / (1.f + qv[idx]);
        float CWM = qt[idx];
        float Fice_in = f_ice[idx];
        float Frain_in = f_rain[idx];
        float Frimef_in = f_rimef[idx];
        if (L == 1) {                              // :436-440
            if (CWM <= EPSQ) CWM = EPSQ;
            Fice_in = 1.f;
            Frain_in = 0.f;
            Frimef_in = 1.f;
        }
        const float PP_col = p[idx];
        const float DPCOL = rho[idx] * ETA_GRAV * dz8w[idx];      // :430
        const float THICK_col = DPCOL * q.RGRAV;                 // :449
        float T_col = t_phy;
        float TC = T_col - ETA_T0C;
        float QV_col = fmaxf_wrf(EPSQ, q_phy);                   // :452
        float WC_col;
        if (CWM <= q.EPSQ1) {                                    // :453-464
            WC_col = 0.f;
            Fice_in = (TC < ETA_T_ICE) ? 1.f : 0.f;
            Frain_in = 0.f;
            Frimef_in = 1.f;
        } else {
            WC_col = CWM;
        }
        // Composition of the condensate (:469-496)
        float QI_col = 0.f, QR_col = 0.f, QW_col = 0.f;
        if (Fice_in >= 1.f) {
            QI_col = WC_col;
        } else if (Fice_in <= 0.f) {
            QW_col = WC_col;
        } else {
            QI_col = Fice_in * WC_col;
            QW_col = WC_col - QI_col;
        }
        if (QW_col > 0.f && Frain_in > 0.f) {
            if (Frain_in >= 1.f) {
                QR_col = QW_col;
                QW_col = 0.f;
            } else {
                QR_col = Frain_in * QW_col;
                QW_col = QW_col - QR_col;
            }
        }
        if (QI_col <= 0.f) Frimef_in = 1.f;
        float RimeF_col = Frimef_in;
        const float rimef_lbef = (L == 1) ? RimeF_col : rimef_above;

        // ---- EGCP01COLUMN level body (:848-1962)
        do {
            if (QV_col <= EPSQ && WC_col <= EPSQ) break;         // :853
            const float TK = T_col;
            TC = TK - ETA_T0C;
            const float PP = PP_col;
            const float QV = QV_col;
            const float WV = QV / (1.f - QV);
            const float WC = WC_col;
            bool CLEAR = true;
            float ESW = fminf_wrf(1000.f * eta_fpvs(TK, c1x0, c2x0, tbpvs0),
                                  0.99f * PP);                    // :874
            const float QSW = q.EPS * ESW / (PP - ESW);
            float WS = QSW;
            if (TC < 0.f) {                                       // :877-881
                float ESI = fminf_wrf(1000.f * eta_fpvs(TK, c1x, c2x, tbpvs),
                                      0.99f * PP);
                QSI = q.EPS * ESI / (PP - ESI);
                WS = QSI;
            }
            const float QSWgrd = ETA_RHGRD * QSW;                 // :885-887
            const float QSIgrd = ETA_RHGRD * QSI;
            const float WSgrd = ETA_RHGRD * WS;
            if (WV > WSgrd || WC > EPSQ) CLEAR = false;          // :891
            if (ARAIN > CLIMIT) {                                 // :895-899
                CLEAR = false;
            } else {
                ARAIN = 0.f;
            }
            if (ASNOW > CLIMIT) {                                 // :906-910
                CLEAR = false;
            } else {
                ASNOW = 0.f;
            }
            if (CLEAR) break;                                     // :916

            const float RHO = PP / (ETA_RD * TK * (1.f + q.EPS1 * QV));   // :926
            const float RRHO = 1.f / RHO;
            const float DTRHO = DTPH * RHO;
            const float BLDTRH = ETA_BLEND * DTRHO;
            const float THICK = THICK_col;
            float ARAINnew = 0.f, ASNOWnew = 0.f;
            const float QI = QI_col;
            float QInew = 0.f;
            const float QR = QR_col;
            float QRnew = 0.f;
            const float QW = QW_col;
            float QWnew = 0.f;
            float PCOND = 0.f, PIDEP = 0.f, PIACW = 0.f, PIACWI = 0.f;
            float PIACWR = 0.f, PIACR = 0.f, PICND = 0.f, PIEVP = 0.f;
            float PIMLT = 0.f, PRAUT = 0.f, PRACW = 0.f, PREVP = 0.f;

            const float XLVT = 3.148E6f - 2370.f * TK;             // :976-982
            const float XLF = ETA_XLS - XLVT;
            const float XLV1 = XLVT * q.rcp;
            const float XLF1 = XLF * q.rcp;
            const float TK2 = 1.f / (TK * TK);
            const float XLV2 = __fdiv_rn(XLVT * XLVT * QSW * TK2, ETA_RV);
            const float DENOMW = 1.f + XLV2 * q.rcp;
            const float TFACTOR = gfk_pow(TK, 1.5f) / (TK + 120.f);   // :989-992
            const float DYNVIS = 1.496E-6f * TFACTOR;
            const float THERM_COND = 2.116E-3f * TFACTOR;
            const float DIFFUS = 8.794E-5f * gfk_pow(TK, 1.81f) / PP;
            const float GAMMAS = gfk_pow(1.E5f / PP, q.C1);      // :997
            const float GAMMAR = gfk_pow(ETA_RHO0 / RHO, .4f);   // :1001

            bool ICE_logical;
            float QLICE = 0.f, QTICE = 0.f;
            if (TC < 0.f || QI > EPSQ || ASNOW > CLIMIT) {       // :1009-1015
                ICE_logical = true;
            } else {
                ICE_logical = false;
                QLICE = 0.f;
                QTICE = 0.f;
            }
            bool RAIN_logical = false;                            // :1019-1020
            if (ARAIN > CLIMIT || QR > EPSQ) RAIN_logical = true;

            int INDEXS = ETA_MDIMIN;
            float TOT_ICE = 0.f, PILOSS = 0.f, RimeF1 = 1.f, VrimeF = 1.f;
            float VEL_INC = GAMMAS, VSNOW = 0.f, EMAIRI = THICK;
            float XLIMASS = 0.f, FLIMASS = 0.f, NLICE = 0.f, NSmICE = 0.f;
            float FLARGE, FSMALL = 0.f, XSIMASS = 0.f;
            float WVQW = WV + QW;
            if (ICE_logical) {
                WVQW = WV + QW;                                   // :1053
                if (TC >= 0.f || WVQW < QSIgrd) {                 // :1057-1073
                    FLARGE = ETA_FLARGE1;
                } else {
                    FLARGE = ETA_FLARGE2;
                    if (TC >= -8.f && TC <= -3.f) FLARGE = .5f * FLARGE;
                }
                FSMALL = (1.f - FLARGE) / FLARGE;                 // :1074
                XSIMASS = RRHO * MASSI[ETA_MDIMIN] * FSMALL;
                if (QI <= EPSQ && ASNOW <= CLIMIT) {              // :1076-1090
                    INDEXS = ETA_MDIMIN;
                    TOT_ICE = 0.f;
                    PILOSS = 0.f;
                    RimeF1 = 1.f;
                    VrimeF = 1.f;
                    VEL_INC = GAMMAS;
                    VSNOW = 0.f;
                    EMAIRI = THICK;
                    XLIMASS = RRHO * RimeF1 * MASSI[INDEXS];
                    FLIMASS = XLIMASS / (XLIMASS + XSIMASS);
                    QLICE = 0.f;
                    QTICE = 0.f;
                    NLICE = 0.f;
                    NSmICE = 0.f;
                } else {
                    float DUM = (float)ETA_MDIMAX * gfk_exp(.0536f * TC);   // :1097
                    INDEXS = imin_wrf(ETA_MDIMAX,
                                      imax_wrf(ETA_MDIMIN, fint_wrf(DUM)));
                    TOT_ICE = THICK * QI + ETA_BLEND * ASNOW;     // :1099
                    PILOSS = -TOT_ICE / THICK;
                    const float DUM1 = rimef_lbef;                // :1101-1102
                    const float DUM2 = RimeF_col;
                    RimeF1 = (DUM2 * THICK * QI + DUM1 * ETA_BLEND * ASNOW)
                             / TOT_ICE;                           // :1104
                    RimeF1 = fminf_wrf(RimeF1, RFmax);
                    for (int IPASS = 0; IPASS <= 1; ++IPASS) {    // :1106-1168
                        if (RimeF1 <= 1.f) {
                            RimeF1 = 1.f;
                            VrimeF = 1.f;
                        } else {
                            const int IXS = imax_wrf(2, imin_wrf(INDEXS / 100, 9));
                            const float XRF = 10.492f * gfk_log(RimeF1);
                            const int IXRF = imax_wrf(0, imin_wrf(fint_wrf(XRF),
                                                                  ETA_NRIME));
                            if (IXRF >= ETA_NRIME) {
                                VrimeF = VEL_RF[(IXS - 2) + 8 * ETA_NRIME];
                            } else {
                                const float v0 = VEL_RF[(IXS - 2) + 8 * IXRF];
                                const float v1 = VEL_RF[(IXS - 2) + 8 * (IXRF + 1)];
                                VrimeF = v0 + (XRF - (float)IXRF) * (v1 - v0);
                            }
                        }
                        VEL_INC = GAMMAS * VrimeF;                // :1121-1129
                        VSNOW = VEL_INC * VSNOWI[INDEXS];
                        EMAIRI = THICK + BLDTRH * VSNOW;
                        XLIMASS = RRHO * RimeF1 * MASSI[INDEXS];
                        FLIMASS = XLIMASS / (XLIMASS + XSIMASS);
                        QTICE = TOT_ICE / EMAIRI;
                        QLICE = FLIMASS * QTICE;
                        NLICE = QLICE / XLIMASS;
                        NSmICE = FSMALL * NLICE;
                        if ((NLICE >= ETA_NLIMIN && NLICE <= ETA_NLIMAX)
                            || IPASS == 1) {
                            break;
                        }
                        if (TC < 0.f) {                           // :1135-1148
                            const float XLI = RHO * (QTICE / DUM - XSIMASS) / RimeF1;
                            if (XLI <= MASSI[ETA_MDIMIN]) {
                                INDEXS = ETA_MDIMIN;
                            } else if (XLI <= MASSI[450]) {
                                const float DLI = 9.5885E5f * gfk_pow(XLI, .42066f);
                                INDEXS = imin_wrf(ETA_MDIMAX,
                                                  imax_wrf(ETA_MDIMIN, fint_wrf(DLI)));
                            } else if (XLI <= MASSI[ETA_MDIMAX]) {
                                const float DLI = 3.9751E6f * gfk_pow(XLI, .49870f);
                                INDEXS = imin_wrf(ETA_MDIMAX,
                                                  imax_wrf(ETA_MDIMIN, fint_wrf(DLI)));
                            } else {
                                INDEXS = ETA_MDIMAX;
                            }
                        }
                        DUM = fmaxf_wrf(ETA_NLIMIN, fminf_wrf(ETA_NLIMAX, NLICE));  // :1160
                        if (DUM >= ETA_NLIMAX && INDEXS >= ETA_MDIMAX) {
                            RimeF1 = RHO * (__fdiv_rn(QTICE, ETA_NLIMAX) - XSIMASS)
                                     / MASSI[INDEXS];
                        }
                    }
                }
            }

            // ---- autoconversion and riming (:1178-1196)
            if (QW > EPSQ && TC >= ETA_T_ICE) {
                const float QW0 = QAUT0 * RRHO;
                PRAUT = fmaxf_wrf(0.f, fminf_wrf(QW - QW0, QW0)) * CRAUT;
                if (QLICE > EPSQ) {
                    const float FWS = fminf_wrf(1.f, CIACW * VEL_INC * NLICE
                                                * ACCRI[INDEXS]
                                                / gfk_pow(PP, q.C1));
                    PIACW = FWS * QW;
                    if (TC < 0.f) PIACWI = PIACW;
                }
            }

            // ---- ice deposition / sublimation (:1202-1287)
            if (ICE_logical) {
                if (TC < ETA_T_ICE && (WV > QSIgrd || QW > EPSQ)) {
                    PCOND = -QW;                                  // :1214-1219
                    const float DUM1 = TK + XLV1 * PCOND;
                    const float DUM2 = WV + QW;
                    float DUM = fminf_wrf(1000.f * eta_fpvs(DUM1, c1x, c2x, tbpvs),
                                          0.99f * PP);
                    DUM = ETA_RHGRD * q.EPS * DUM / (PP - DUM);
                    if (DUM2 > DUM) PIDEP = eta_deposit(q, PP, DUM1, DUM2, rs, tbpvs);
                } else if (TC < 0.f) {
                    const float DENOMI = 1.f + q.XLS2 * QSI * TK2;    // :1227-1229
                    const float DWVi = fminf_wrf(WVQW, QSW) - QSI;
                    const float PIDEP_max = fmaxf_wrf(PILOSS, DWVi / DENOMI);
                    if (QTICE > 0.f) {                            // :1230-1261
                        const float SFACTOR = gfk_pow(VEL_INC, .5f)
                            * gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), q.C2);
                        const float ABI = 1.f / (RHO * q.XLS3 * QSI * TK2 / THERM_COND
                                                 + 1.f / DIFFUS);
                        const float VENTIL = (VENTI1[INDEXS] + SFACTOR * VENTI2[INDEXS])
                                             * NLICE;
                        const float VENTIS = (VENTI1[ETA_MDIMIN]
                                              + SFACTOR * VENTI2[ETA_MDIMIN]) * NSmICE;
                        float DIDEP = ABI * (VENTIL + VENTIS) * DTPH;
                        if (DIDEP >= ETA_XRATIO) {
                            DIDEP = (1.f - gfk_exp(-(DIDEP * DENOMI))) / DENOMI;
                        }
                        if (DWVi > 0.f) {
                            PIDEP = fminf_wrf(DWVi * DIDEP, PIDEP_max);
                        } else if (DWVi < 0.f) {
                            PIDEP = fmaxf_wrf(DWVi * DIDEP, PIDEP_max);
                        }
                    } else if (WVQW > QSI && TC <= ETA_T_ICE_INIT) {   // :1263-1283
                        const int INDEX_MY = imax_wrf(ETA_MY_T1,
                            imin_wrf(fint_wrf(.5f - TC), ETA_MY_T2));
                        const float DUM1 = QSW / QSI - 1.f;
                        const float DUM2 = 1.E3f * gfk_exp(12.96f * DUM1 - .639f);
                        PIDEP = fminf_wrf(PIDEP_max, DUM2 * MY_GROWTH[INDEX_MY] * RRHO);
                    }
                }
            }

            // ---- cloud water condensation (:1299-1323)
            if (TC >= ETA_T_ICE && (QW > EPSQ || WV > QSWgrd)) {
                if (PIACWI == 0.f && PIDEP == 0.f) {
                    PCOND = eta_condense(q, PP, QW, TK, WV, rs, tbpvs0);
                } else {
                    float DUM = XLVT * QSWgrd * q.RCPRV * TK2;
                    const float DENOMWI = 1.f + ETA_XLS * DUM;
                    const float DENOMF = XLF * DUM;
                    DUM = fmaxf_wrf(0.f, PIDEP);
                    PCOND = (WV - QSWgrd - DENOMWI * DUM - DENOMF * PIACWI) / DENOMW;
                    const float DUM1 = -QW;
                    const float DUM2 = PCOND - PIACW;
                    if (DUM2 < DUM1) {
                        DUM = DUM1 / DUM2;
                        PCOND = DUM * PCOND;
                        PIACW = DUM * PIACW;
                        PIACWI = DUM * PIACWI;
                    }
                }
            }

            // ---- Schumann-Ludlam limit and melting (:1328-1375)
            float TCC = TC + XLV1 * PCOND + q.XLS1 * PIDEP + XLF1 * PIACWI;
            if (TCC > 0.f) {
                PIACWI = 0.f;
                TCC = TC + XLV1 * PCOND + q.XLS1 * PIDEP;
            }
            if (TC > 0.f && TCC > 0.f && ICE_logical) {
                const float SFACTOR = gfk_pow(VEL_INC, .5f)
                    * gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), q.C2);
                const float VENTIL = NLICE * (VENTI1[INDEXS] + SFACTOR * VENTI2[INDEXS]);
                const float AIEVP = VENTIL * DIFFUS * DTPH;
                float DIEVP;
                if (AIEVP < ETA_XRATIO) {
                    DIEVP = AIEVP;
                } else {
                    DIEVP = 1.f - gfk_exp(-AIEVP);
                }
                const float QSW0 = q.EPS * ESW0 / (PP - ESW0);
                const float DWV0 = fminf_wrf(WV, QSW) - QSW0;
                float DUM = QW + PCOND;
                if (WV < QSW && DUM <= EPSQ) {
                    DUM = DWV0 * DIEVP;
                    PIEVP = fmaxf_wrf(fminf_wrf(0.f, DUM), PILOSS);
                    PICND = fmaxf_wrf(0.f, DUM);
                }
                PIMLT = THERM_COND * TCC * VENTIL * RRHO * DTPH / XLF;
                float DUM1 = fmaxf_wrf(0.f, (TCC + XLV1 * PIEVP) / XLF1);
                PIMLT = fminf_wrf(PIMLT, DUM1);
                DUM = PIEVP - PIMLT;
                if (DUM < PILOSS) {
                    DUM1 = PILOSS / DUM;
                    PIMLT = PIMLT * DUM1;
                    PIEVP = PIEVP * DUM1;
                }
            }

            // ---- rain (:1390-1592)
            float TOT_RAIN = 0.f, VRAIN1 = 0.f, QTRAIN = 0.f, PRLOSS = 0.f;
            float RQR = 0.f, N0r = 0.f;
            int INDEXR = ETA_MDRMIN;
            if (RAIN_logical) {
                if (ARAIN <= 0.f) {
                    INDEXR = ETA_MDRMIN;
                    VRAIN1 = 0.f;
                } else {
                    const float RR = ARAIN / (DTPH * GAMMAR);
                    INDEXR = eta_drop_index(RR, k, true);
                    VRAIN1 = GAMMAR * VRAIN[INDEXR];
                }
                TOT_RAIN = THICK * QR + ETA_BLEND * ARAIN;        // :1498-1501
                QTRAIN = TOT_RAIN / (THICK + BLDTRH * VRAIN1);
                PRLOSS = -TOT_RAIN / THICK;
                RQR = RHO * QTRAIN;
                if (RQR <= RQR_DRmin) {                           // :1505-1514
                    N0r = fmaxf_wrf(ETA_N0RMIN, CN0r_DMRmin * RQR);
                    INDEXR = ETA_MDRMIN;
                } else if (RQR >= RQR_DRmax) {
                    N0r = CN0r_DMRmax * RQR;
                    INDEXR = ETA_MDRMAX;
                } else {
                    N0r = ETA_N0R0;
                    INDEXR = fint_wrf(fmaxf_wrf(ETA_XMRMIN,
                        fminf_wrf(CN0r0 * gfk_pow(RQR, .25f), ETA_XMRMAX)));
                }
                if (RQR <= EPSQ) {                                // :1515-1519
                    VRAIN1 = 0.f;
                } else {
                    VRAIN1 = GAMMAR * VRAIN[INDEXR];
                }
                if (TC < ETA_T_ICE) {                             // :1521-1522
                    PIACR = -PRLOSS;
                } else {
                    const float DWVr = WV - PCOND - QSW;          // :1524-1525
                    float DUM = QW + PCOND;
                    if (DWVr < 0.f && DUM <= EPSQ) {
                        const float RFACTOR = gfk_pow(GAMMAR, .5f)
                            * gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), q.C2);
                        const float ABW = 1.f / (RHO * XLV2 / THERM_COND
                                                 + 1.f / DIFFUS);
                        const float VENTR = N0r * (VENTR1[INDEXR]
                                                   + RFACTOR * VENTR2[INDEXR]);
                        const float CREVP = ABW * VENTR * DTPH;
                        if (CREVP < ETA_XRATIO) {
                            DUM = DWVr * CREVP;
                        } else {
                            DUM = DWVr * (1.f - gfk_exp(-(CREVP * DENOMW))) / DENOMW;
                        }
                        PREVP = fmaxf_wrf(DUM, PRLOSS);
                    } else if (QW > EPSQ) {
                        const float FWR = CRACW * GAMMAR * N0r * ACCRR[INDEXR];
                        PRACW = fminf_wrf(1.0f, FWR) * QW;
                    }
                    if (TC < 0.f && TCC < 0.f) {                  // :1556-1590
                        DUM = .001f * (float)INDEXR;
                        DUM = (gfk_exp(ABFR * TC) - 1.f) * DUM * DUM * DUM * DUM
                              * DUM * DUM * DUM;
                        PIACR = fminf_wrf(CBFR * N0r * RRHO * DUM, QTRAIN);
                        if (QLICE > EPSQ) {
                            DUM = VRAIN1;
                            float DUM1 = DUM - VSNOW;
                            const float DUM2 = gfk_pow(DUM1 * DUM1 + .04f * DUM * VSNOW, .5f);
                            DUM1 = 5.E-12f * (float)INDEXR * (float)INDEXR
                                   + 2.E-12f * (float)INDEXR * (float)INDEXS
                                   + .5E-12f * (float)INDEXS * (float)INDEXS;
                            const float FIR = fminf_wrf(1.f, CIACR * NLICE * DUM1 * DUM2);
                            PIACR = fminf_wrf(PIACR + FIR * QTRAIN, QTRAIN);
                        }
                        DUM = PREVP - PIACR;
                        if (DUM < PRLOSS) {
                            const float DUM1 = PRLOSS / DUM;
                            PREVP = DUM1 * PREVP;
                            PIACR = DUM1 * PIACR;
                        }
                    }
                }
            }

            // ---- budget: cloud water (:1605-1624)
            {
                const float DUM1 = PIACW + PRAUT + PRACW - fminf_wrf(0.f, PCOND);
                if (DUM1 > QW) {
                    const float DUM = QW / DUM1;
                    PIACW = DUM * PIACW;
                    PIACWI = DUM * PIACWI;
                    PRAUT = DUM * PRAUT;
                    PRACW = DUM * PRACW;
                    if (PCOND < 0.f) PCOND = DUM * PCOND;
                }
            }
            PIACWR = PIACW - PIACWI;
            const float DELW = PCOND - PIACW - PRAUT - PRACW;
            QWnew = QW + DELW;
            if (QWnew <= EPSQ) QWnew = 0.f;
            if (QW > 0.f && QWnew != 0.f) {
                const float DUM = QWnew / QW;
                if (DUM < ETA_TOLER) QWnew = 0.f;
            }

            // ---- temperature and vapour (:1628-1633)
            const float DELT = XLV1 * (PCOND + PIEVP + PICND + PREVP)
                               + q.XLS1 * PIDEP + XLF1 * (PIACWI + PIACR - PIMLT);
            const float Tnew = TK + DELT;
            const float DELV = -PCOND - PIDEP - PIEVP - PICND - PREVP;
            const float WVnew = WV + DELV;

            // ---- ice (:1650-1697)
            float RimeF = 1.f;
            if (ICE_logical) {
                const float DELI = PIDEP + PIEVP + PIACWI + PIACR - PIMLT;
                float TOT_ICEnew = TOT_ICE + THICK * DELI;
                if (TOT_ICE > 0.f && TOT_ICEnew != 0.f) {
                    const float DUM = TOT_ICEnew / TOT_ICE;
                    if (DUM < ETA_TOLER) TOT_ICEnew = 0.f;
                }
                if (TOT_ICEnew <= CLIMIT) {
                    TOT_ICEnew = 0.f;
                    RimeF = 1.f;
                    QInew = 0.f;
                    ASNOWnew = 0.f;
                } else {
                    const float DUM = PIACWI + PIACR;
                    if (DUM <= EPSQ && PIDEP <= EPSQ) {
                        RimeF = RimeF1;
                    } else {
                        const float DUM1 = TOT_ICE + THICK * (PIDEP + DUM);
                        const float DUM2 = TOT_ICE / RimeF1 + THICK * PIDEP;
                        if (DUM2 <= 0.f) {
                            RimeF = RFmax;
                        } else {
                            RimeF = fminf_wrf(RFmax, fmaxf_wrf(1.f, DUM1 / DUM2));
                        }
                    }
                    QInew = TOT_ICEnew / (THICK + BLDTRH * FLIMASS * VSNOW);
                    if (QInew <= EPSQ) QInew = 0.f;
                    if (QI > 0.f && QInew != 0.f) {
                        const float D = QInew / QI;
                        if (D < ETA_TOLER) QInew = 0.f;
                    }
                    ASNOWnew = BLDTRH * FLIMASS * VSNOW * QInew;
                    if (ASNOW > 0.f && ASNOWnew != 0.f) {
                        const float D = ASNOWnew / ASNOW;
                        if (D < ETA_TOLER) ASNOWnew = 0.f;
                    }
                }
            }

            // ---- rain (:1713-1771)
            {
                const float DELR = PRAUT + PRACW + PIACWR - PIACR + PIMLT + PREVP + PICND;
                float TOT_RAINnew = TOT_RAIN + THICK * DELR;
                if (TOT_RAIN > 0.f && TOT_RAINnew != 0.f) {
                    const float DUM = TOT_RAINnew / TOT_RAIN;
                    if (DUM < ETA_TOLER) TOT_RAINnew = 0.f;
                }
                if (TOT_RAINnew <= CLIMIT) {
                    TOT_RAINnew = 0.f;
                    QRnew = 0.f;
                    ARAINnew = 0.f;
                } else {
                    const float RR = TOT_RAINnew / (DTPH * GAMMAR);
                    const int IDR = eta_drop_index(RR, k, false);
                    const float VRAIN2 = .5f * (VRAIN1 + GAMMAR * VRAIN[IDR]);
                    QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
                    if (QRnew <= EPSQ) QRnew = 0.f;
                    if (QR > 0.f && QRnew != 0.f) {
                        const float DUM = QRnew / QR;
                        if (DUM < ETA_TOLER) QRnew = 0.f;
                    }
                    ARAINnew = BLDTRH * VRAIN2 * QRnew;
                    if (ARAIN > 0.f && ARAINnew != 0.f) {
                        const float DUM = ARAINnew / ARAIN;
                        if (DUM < ETA_TOLER) ARAINnew = 0.f;
                    }
                }
            }
            const float WCnew = QWnew + QRnew + QInew;           // :1773

            // ---- update arrays (:1953-1962)
            T_col = Tnew;
            QV_col = fmaxf_wrf(EPSQ, WVnew / (1.f + WVnew));
            WC_col = fmaxf_wrf(EPSQ, WCnew);
            QI_col = fmaxf_wrf(EPSQ, QInew);
            QR_col = fmaxf_wrf(EPSQ, QRnew);
            QW_col = fmaxf_wrf(EPSQ, QWnew);
            RimeF_col = RimeF;
            ASNOW = ASNOWnew;
            ARAIN = ARAINnew;
        } while (false);
        rimef_above = RimeF_col;

        // ---- EGCP01DRV storage (:516-540)
        float Fi, Frf, Fr;
        if (QI_col <= EPSQ) {
            Fi = 0.f;
            if (T_col < q.T_ICEK) Fi = 1.f;
            Frf = 1.f;
        } else {
            Fi = fmaxf_wrf(0.f, fminf_wrf(1.f, QI_col / WC_col));
            Frf = fmaxf_wrf(1.f, RimeF_col);
        }
        if (QR_col <= EPSQ) {
            Fr = 0.f;
        } else {
            Fr = QR_col / (QR_col + QW_col);
        }

        // ---- ETAMP_NEW exit (:268-297)
        th[idx] = T_col / pii[idx];
        qv[idx] = QV_col / (1.f - QV_col);
        const float WCo = WC_col;
        float QSo = 0.f, QRo = 0.f, QCo = 0.f;
        if (Fi >= 1.f) {
            QSo = WCo;
        } else if (Fi <= 0.f) {
            QCo = WCo;
        } else {
            QSo = Fi * WCo;
            QCo = WCo - QSo;
        }
        if (QCo > 0.f && Fr > 0.f) {
            if (Fr >= 1.f) {
                QRo = QCo;
                QCo = 0.f;
            } else {
                QRo = Fr * QCo;
                QCo = QCo - QRo;
            }
        }
        qt[idx] = WCo;
        qs[idx] = QSo;
        qr[idx] = QRo;
        qc[idx] = QCo;
        f_ice[idx] = Fi;
        f_rain[idx] = Fr;
        f_rimef[idx] = Frf;
    }

    // ---- surface precipitation (:547-554, :301-305)
    const float APREC = (ARAIN + ASNOW) * q.RRHOL;
    float SR;
    if (APREC < 1.E-8f) {
        SR = 0.f;
    } else {
        SR = q.RRHOL * ASNOW / APREC;
    }
    sr[col] = SR;
    rainnc[col] = APREC * 1000.f + rainnc[col];
    rainncv[col] = APREC * 1000.f;
}

// WRF module_radiation_driver.F:3861-3865, 3902-3922, 3945-3979.
// Older Eta's QS is total frozen condensate; F_ICE_PHY supplies its weight.
extern "C" __global__ void etanew_cloud_fraction(
    int n, const float* qv, const float* qc, const float* qs,
    const float* t, const float* p, const float* ice, float* cloud) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  float tc = t[i] - 273.15f;
  float esw = (1000.f * .61078f) * gfk_exp(17.2693882f * tc / (t[i] - 35.86f));
  float esi = (1000.f * .61078f) * gfk_exp(21.8745584f * tc / (t[i] - 7.66f));
  float qvsw = (287.f / 461.6f) * esw / (p[i] - esw);
  float qvsi = (287.f / 461.6f) * esi / (p[i] - esi);
  float qcld = qc[i] + qs[i];
  float weight = qcld < 1.e-12f ? 0.f : ice[i];
  float qvs = (1.f - weight) * qvsw + weight * qvsi;
  float rh = qv[i] / qvs;
  if (qcld < 1.e-12f) { cloud[i] = 0.f; return; }
  if (rh >= 1.f) { cloud[i] = 1.f; return; }
  float subsat = qvs - qv[i];
  subsat = 1.e-10f > subsat ? 1.e-10f : subsat;
  float denom = gfk_pow(subsat, .49f);
  float arg = -100.f * qcld / denom;
  arg = -6.9f > arg ? -6.9f : arg;
  rh = 1.e-10f > rh ? 1.e-10f : rh;
  float fraction = gfk_pow(rh, .25f) * (1.f - gfk_exp(arg));
  cloud[i] = fraction < .01f ? 0.f : fraction;
}
