// One shared column kernel for selectors 5 and 15.
// mp15 derives rime factor from QRIMEF/QI (:615), keeps QT/F_* local,
// and writes QRIMEF=QI*F_RIMEF_PHY after repartition (:690).
// gpuwm/core/kernels/fer_hires.cu
//
// Ferrier-Aligo microphysics, WRF mp_physics = 5 (package fer_mp_hires),
// transcribed statement by statement from WRF v4.6.1
// phys/module_mp_fer_hires.F (sha256 69e937f0...): FER_HIRES (:240-471),
// EGCP01DRV (:717-1005), EGCP01COLUMN with CONDENSE and DEPOSIT
// (:1051-2783), fer_hires_init (:2793-3150), MY_GROWTH_RATES (:3152-3190),
// GPVS / FPVS / FPVS0 / FPVSX / FPVSX0 (:3196-3397).
//
// One thread per (i, j) column.  WRF flips the column so L = 1 is the model
// top (EGCP01DRV :799-813) and integrates top-down; the only state one level
// hands the next is the falling rain and ice (ARAIN, ASNOW, VRAIN1, VSNOW1),
// the melting-layer bookkeeping (INDEXS0C, RHOX0C, TIMLT, STRAT, DRZL,
// INDEXRmin) and the updated rime factor of the level above
// (RimeF_col(LBEF), :1575-1577).  Every other column array in the Fortran
// is written and read at the same L, so the kernel streams the column from
// k = nz-1 down to 0 and never stores a level array.
//
// ARITHMETIC.  One Fortran statement is the same left-to-right sequence of
// binary32 operations here, every literal carries its binary32 suffix, and
// the strict build (GPUWM_WRF_EXACT=1: --fmad=false --ftz=false
// --prec-div=true --prec-sqrt=true) contracts nothing, which is what grades
// at 0 ULP against the gfortran -O0 reference.  EXP, ALOG and REAL ** REAL
// are WOOF's own float32 libm (gfk_exp, gfk_log, gfk_pow in
// glibc_flt32.cuh); SQRT is the correctly rounded sqrtf.  CONDENSE and
// DEPOSIT carry COND/DEP/SSAT/WCdum in binary64 exactly as their
// REAL(KIND=8) declarations do (:2571-2574, :2639-2642): the binary32
// right-hand sides are evaluated in binary32 and widened, the updates of
// Tdum/WVdum are binary64 sums rounded once to binary32.  Compile-time
// constants the Fortran front end folds (parameters, ACOS(-1.),
// (1.E5)**C1, 1.1**Nrime, (20.E-6)**3, ...) are written as the exact
// binary32 words gfortran folds them to; tests/test_fer_hires_port.py
// recomputes each one.
//
// Fortran MAX/MIN are fh_max/fh_min, with the operand choice gfortran 13
// -O0 makes on x86-64 (maxss/minss): MAX(a1,a2) returns a1 only when
// a1 > a2 and a2 otherwise, so MAX(0.,-0.) is -0. and MAX(NaN,1.) is 1.
// (MIN: a1 only when a1 < a2).  Measured with a probe program on the
// oracle's own toolchain.  INT() of a real is fh_int: a truncation that
// maps NaN and out-of-range values to INT_MIN as cvttss2si does.
//
// Rcw deliberately follows WRF's DUM2 reuse at :1734/:1748.  Strict
// parity grades the stock module, including this arithmetic order.

#define FH_MDIMIN 50
#define FH_MDIMAX 1000
#define FH_MDRMIN 50
#define FH_MDRMAX 1000
#define FH_NTAB 951            // MDImax - MDImin + 1 = MDRmax - MDRmin + 1
#define FH_NRIME 40
#define FH_NX 7501
#define FH_MY_T1 1
#define FH_MY_T2 35
#define FH_INDEXRSTRMAX 500
#define FH_MAX_ITERATIONS 10

// --- Physical constants, module_mp_fer_hires.F:166-187 (binary32).
#define FH_CP     1004.6f
#define FH_EPSQ   1.E-12f
#define FH_GRAV   9.806f
#define FH_RHOL   1000.f
#define FH_RD     287.04f
#define FH_RV     461.5f
#define FH_T0C    273.15f
#define FH_XLV    2.5E6f
#define FH_XLF    3.3358e+5f
#define FH_PI     3.141592653589793f
// Derived parameters, folded by gfortran one binary32 operation at a time.
#define FH_EPS    0x1.3e7318p-1f     // RD/RV
#define FH_EPS1   0x1.373038p-1f     // RV/RD-1.
#define FH_EPSQ1  0x1.19c1a8p-40f    // 1.001*EPSQ
#define FH_RGRAV  0x1.a1b416p-4f     // 1./GRAV
#define FH_RRHOL  0x1.0624dep-10f    // 1./RHOL
#define FH_XLV1   0x1.3711b0p+11f    // XLV/CP
#define FH_XLF1   0x1.4c0d74p+8f     // XLF/CP
#define FH_XLS1   0x1.60935ep+11f    // XLS/CP, XLS=XLV+XLF
#define FH_XLV2   0x1.939b54p+33f    // XLV*XLV/RV
#define FH_XLS2   0x1.034018p+34f    // XLS*XLS/RV
#define FH_XLV3   0x1.9b669ep+23f    // XLV*XLV*RCPRV, RCPRV=RCP/RV, RCP=1./CP
#define FH_XLS3   0x1.0841bcp+24f    // XLS*XLS*RCPRV
#define FH_CLIMIT 0x1.5fd7fep-37f    // 10.*EPSQ
#define FH_C1     0x1.555556p-2f     // 1./3.
#define FH_XMRMIN 50.f               // 1.e6*DMRmin
#define FH_XMRMAX 0x1.f40002p+9f     // 1.e6*DMRmax  (1000.00006)
#define FH_XMIMAX 0x1.f40002p+9f     // 1.e6*DMImax  (1000.00006)
#define FH_XMIEXP .0536f
#define FH_XMIN   180.0f
#define FH_XMAX   330.0f
// Tunable parameters exported by the module (:216-229).
#define FH_RHGRD_IN     1.f
#define FH_RHGRD_OUT    0.975f
#define FH_P_RHGRD_OUT  850.E2f
#define FH_T_ICE        (-40.f)
#define FH_T_ICEK       0x1.d24cccp+7f   // T0C+T_ICE
#define FH_T_ICE_INIT   (-12.f)
#define FH_NSI_MAX      250.E3f
#define FH_NLIMIN       1.E3f
#define FH_N0R0         8.E6f
#define FH_NCW          250.E6f
// EGCP01COLUMN local parameters (:1190-1203).
#define FH_TOLER  5.E-7f
#define FH_C2     0x1.555556p-3f     // 1./6.
#define FH_RHO0   1.194f
#define FH_XRATIO .025f
#define FH_BLEND  1.f
// CONDENSE / DEPOSIT REAL(KIND=8) parameters RHLIMIT=.001: the binary32
// literal widened, not the binary64 decimal (:2572-2573, :2640-2641).
#define FH_RHLIMIT  ((double)0.001f)
// Front-end folded constant subexpressions.
#define FH_1E5_POW_C1 0x1.7353c0p+5f  // (1.E5)**C1, FER_HIRES :329
#define FH_RFMAX_FOLDED 0x1.6a130ap+5f // 1.1**Nrime, fer_hires_init :3055

// Offsets into the table buffer the init kernel fills (see
// gpuwm/core/fer_hires.py, which owns the same numbers).
#define FH_OFF_VENTR1   0
#define FH_OFF_VENTR2   (1 * FH_NTAB)
#define FH_OFF_ACCRR    (2 * FH_NTAB)
#define FH_OFF_MASSR    (3 * FH_NTAB)
#define FH_OFF_VRAIN    (4 * FH_NTAB)
#define FH_OFF_RRATE    (5 * FH_NTAB)
#define FH_OFF_VENTI1   (6 * FH_NTAB)
#define FH_OFF_VENTI2   (7 * FH_NTAB)
#define FH_OFF_ACCRI    (8 * FH_NTAB)
#define FH_OFF_MASSI    (9 * FH_NTAB)
#define FH_OFF_VSNOWI   (10 * FH_NTAB)
#define FH_OFF_VEL_RF   (11 * FH_NTAB)                 // (2:9, 0:Nrime)
#define FH_OFF_TBPVS    (FH_OFF_VEL_RF + 8 * (FH_NRIME + 1))
#define FH_OFF_TBPVS0   (FH_OFF_TBPVS + FH_NX)
#define FH_OFF_SCALARS  (FH_OFF_TBPVS0 + FH_NX)
// Scalars fer_hires_init leaves in module SAVE variables.
#define FH_S_C1XPVS      0
#define FH_S_C2XPVS      1
#define FH_S_C1XPVS0     2
#define FH_S_C2XPVS0     3
#define FH_S_PI_E        4
#define FH_S_ABFR        5
#define FH_S_CBFR        6
#define FH_S_RQR_DRMIN   7
#define FH_S_RQR_DRMAX   8
#define FH_S_CN0R0       9
#define FH_S_ESW0        10
#define FH_S_RFMAX       11
#define FH_S_RFMX1       12
#define FH_S_BETA6       13
#define FH_S_ARAUT       14
#define FH_S_QAUT0       15
#define FH_S_ARCW        16
#define FH_S_RH_NGC      17
#define FH_S_RQHAIL      18
#define FH_S_BVHAIL      19
#define FH_S_AVHAIL      20
#define FH_S_RH_NGT      21
#define FH_S_DTPH        22
#define FH_S_CRAIN       23
#define FH_N_SCALARS     24
#define FH_TABLE_WORDS   (FH_OFF_SCALARS + FH_N_SCALARS)

// gfortran MAX/MIN on reals and INT() of a real (see the header comment).
__device__ __forceinline__ float fh_max(float a1, float a2)
{ return (a1 > a2) ? a1 : a2; }
__device__ __forceinline__ float fh_min(float a1, float a2)
{ return (a1 < a2) ? a1 : a2; }
__device__ __forceinline__ double fh_maxd(double a1, double a2)
{ return (a1 > a2) ? a1 : a2; }
// gfortran O0 keeps the first operand for these expression/variable
// call sites when either operand is NaN. Finite and signed-zero behavior
// still follows the ordinary helper. The all-step fallback trace pins it.
__device__ __forceinline__ float fh_max_first(float a1, float a2)
{ return (isnan(a1) || isnan(a2)) ? a1 : fh_max(a1, a2); }
__device__ __forceinline__ float fh_min_first(float a1, float a2)
{ return (isnan(a1) || isnan(a2)) ? a1 : fh_min(a1, a2); }
__device__ __forceinline__ int fh_int(float x)
{
    if (!(x > -2147483904.0f && x < 2147483648.0f)) return -2147483647 - 1;
    return (int)x;
}

// --- Saturation vapour pressure lookups (FPVS :3248-3299, FPVS0 :3302-3319)
__device__ __forceinline__ float fh_fpvs(const real* __restrict__ tab, float T)
{
    if (T >= FH_XMIN && T <= FH_XMAX) {
        const float c1 = tab[FH_OFF_SCALARS + FH_S_C1XPVS];
        const float c2 = tab[FH_OFF_SCALARS + FH_S_C2XPVS];
        float XJ = fh_min(fh_max(c1 + c2 * T, 1.f), (float)FH_NX);
        int JX = fh_int(fh_min(XJ, (float)FH_NX - 1.f));
        const real* tb = tab + FH_OFF_TBPVS - 1;          // TBPVS(1:NX)
        return tb[JX] + (XJ - (float)JX) * (tb[JX + 1] - tb[JX]);
    } else if (T > FH_XMAX) {
        return 0.61078f * gfk_exp(17.2694f * (T - 273.16f) / (T - 35.86f));
    } else {
        return 0.61078f * gfk_exp(21.8746f * (T - 273.16f) / (T - 7.66f));
    }
}

__device__ __forceinline__ float fh_fpvs0(const real* __restrict__ tab, float T)
{
    if (T >= FH_XMIN && T <= FH_XMAX) {
        const float c1 = tab[FH_OFF_SCALARS + FH_S_C1XPVS0];
        const float c2 = tab[FH_OFF_SCALARS + FH_S_C2XPVS0];
        float XJ1 = fh_min(fh_max(c1 + c2 * T, 1.f), (float)FH_NX);
        int JX1 = fh_int(fh_min(XJ1, (float)FH_NX - 1.f));
        const real* tb = tab + FH_OFF_TBPVS0 - 1;         // TBPVS0(1:NX)
        return tb[JX1] + (XJ1 - (float)JX1) * (tb[JX1 + 1] - tb[JX1]);
    }
    return 0.61078f * gfk_exp(17.2694f * (T - 273.16f) / (T - 35.86f));
}

// --- CONDENSE (:2561-2626): Asai (1965) water adjustment.
__device__ float fh_condense(const real* __restrict__ tab, float PP, float QW,
                             float TK, float WV, float RHgrd)
{
    float Tdum = TK;
    float WVdum = WV;
    double WCdum = (double)QW;
    float ESW = fh_min(1000.f * fh_fpvs0(tab, Tdum), 0.99f * PP);
    float WS = RHgrd * FH_EPS * ESW / (PP - ESW);
    float DWV = WVdum - WS;
    double SSAT = (double)(DWV / WS);
    float CONDENSE = 0.f;
    const float DWVinp = DWV;
    int NITER;
    for (NITER = 1; NITER <= FH_MAX_ITERATIONS; ++NITER) {
        double COND = (double)(DWV / (1.f + FH_XLV3 * WS / (Tdum * Tdum)));
        COND = fh_maxd(COND, -WCdum);
        Tdum = (float)((double)Tdum + (double)FH_XLV1 * COND);
        WVdum = (float)((double)WVdum - COND);
        WCdum = WCdum + COND;
        CONDENSE = (float)((double)CONDENSE + COND);
        ESW = fh_min(1000.f * fh_fpvs0(tab, Tdum), 0.99f * PP);
        WS = RHgrd * FH_EPS * ESW / (PP - ESW);
        DWV = WVdum - WS;
        SSAT = (double)(DWV / WS);
        if (SSAT >= -FH_RHLIMIT && SSAT <= FH_RHLIMIT) break;
        if (SSAT < -FH_RHLIMIT && WCdum <= (double)FH_EPSQ) break;
    }
    if (NITER > FH_MAX_ITERATIONS) CONDENSE = DWVinp;   // :2614-2623
    return CONDENSE;
}

// --- DEPOSIT (:2632-2690): Asai (1965) ice adjustment.  Tdum and WVdum are
// INTENT(INOUT) in the Fortran; the caller's copies are never read again
// after the call, so they are values here.
__device__ float fh_deposit(const real* __restrict__ tab, float PP, float Tdum,
                            float WVdum, float RHgrd)
{
    float ESI = fh_min(1000.f * fh_fpvs(tab, Tdum), 0.99f * PP);
    float WS = RHgrd * FH_EPS * ESI / (PP - ESI);
    float DWV = WVdum - WS;
    double SSAT = (double)(DWV / WS);
    float DEPOSIT = 0.f;
    const float DWVinp = DWV;
    int NITER;
    for (NITER = 1; NITER <= FH_MAX_ITERATIONS; ++NITER) {
        double DEP = (double)(DWV / (1.f + FH_XLS3 * WS / (Tdum * Tdum)));
        Tdum = (float)((double)Tdum + (double)FH_XLS1 * DEP);
        WVdum = (float)((double)WVdum - DEP);
        DEPOSIT = (float)((double)DEPOSIT + DEP);
        ESI = fh_min(1000.f * fh_fpvs(tab, Tdum), 0.99f * PP);
        WS = RHgrd * FH_EPS * ESI / (PP - ESI);
        DWV = WVdum - WS;
        SSAT = (double)(DWV / WS);
        if (SSAT >= -FH_RHLIMIT && SSAT <= FH_RHLIMIT) break;
    }
    if (NITER > FH_MAX_ITERATIONS) DEPOSIT = DWVinp;    // :2678-2687
    return DEPOSIT;
}

// ---------------------------------------------------------------------------
// fer_hires_init (:2793-3150) and GPVS (:3196-3244), once per domain.
// Thread t < NX fills TBPVS(t+1)/TBPVS0(t+1); thread NX fills the scalars.
// The rain/ice integral tables (ETAMPNEW_DATA.expanded_rain, read at
// :2972-2983) are already in tab[0 .. FH_OFF_TBPVS) when this runs.
// Every operation is rounding-pinned, so the tables are WRF's words in the
// default build as well as the strict one.
// ---------------------------------------------------------------------------
__device__ float fh_fpvsx(float T)                       // :3323-3379
{
    const float TTP = 2.7316E+2f;
    const float PSATK = 0x1.38b82ap-1f;     // PSAT*1.E-3
    const float XA = 0x1.44700ep+2f;        // -DLDT/RV
    const float XB = 0x1.8e6906p+4f;        // XA+HVAP/(RV*TTP)
    const float XAI = 0x1.207362p-1f;       // -DLDTI/RV
    const float XBI = 0x1.70b4d0p+4f;       // XAI+HSUB/(RV*TTP)
    float TR = FDIV(TTP, T);
    if (T >= TTP)
        return FMUL(FMUL(PSATK, gfk_pow(TR, XA)),
                    gfk_exp(FMUL(XB, FSUB(1.f, TR))));
    return FMUL(FMUL(PSATK, gfk_pow(TR, XAI)),
                gfk_exp(FMUL(XBI, FSUB(1.f, TR))));
}

__device__ float fh_fpvsx0(float T)                      // :3382-3397
{
    const float TTP = 2.7316E+2f;
    const float PSATK = 0x1.38b82ap-1f;
    const float XA = 0x1.44700ep+2f;
    const float XB = 0x1.8e6906p+4f;
    float TR = FDIV(TTP, T);
    return FMUL(FMUL(PSATK, gfk_pow(TR, XA)),
                gfk_exp(FMUL(XB, FSUB(1.f, TR))));
}


// MY_GROWTH_RATES DATA MY_600 (:3174-3181), 1-based.
__device__ const float FH_MY_600[35] = {
    5.5e-8f, 1.4E-7f, 2.8E-7f, 6.E-7f, 3.3E-6f,
    2.E-6f, 9.E-7f, 8.8E-7f, 8.2E-7f, 9.4e-7f,
    1.2E-6f, 1.85E-6f, 5.5E-6f, 1.5E-5f, 1.7E-5f,
    1.5E-5f, 1.E-5f, 3.4E-6f, 1.85E-6f, 1.35E-6f,
    1.05E-6f, 1.E-6f, 9.5E-7f, 9.0E-7f, 9.5E-7f,
    9.5E-7f, 9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f,
    9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f, 9.E-7f };

// GPVS (:3231-3242): TBPVS(JX), TBPVS0(JX) for JX = t+1.
extern "C" __global__
void fer_hires_init_tables(real* __restrict__ tab)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= FH_NX) return;
    const float XINC = FDIV(FSUB(FH_XMAX, FH_XMIN), (float)(FH_NX - 1));
    float X = FADD(FH_XMIN, FMUL((float)t, XINC));       // XMIN+(JX-1)*XINC
    tab[FH_OFF_TBPVS + t] = fh_fpvsx(X);
    tab[FH_OFF_TBPVS0 + t] = fh_fpvsx0(X);
}

// The SAVE scalars of fer_hires_init (:2929-3115) and GPVS (:3231-3235).
// One thread, after fer_hires_init_tables (ESW0 reads TBPVS0).
extern "C" __global__
void fer_hires_init_scalars(real* __restrict__ tab, real dtph)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    real* s = tab + FH_OFF_SCALARS;
    const real* MASSR = tab + FH_OFF_MASSR - FH_MDRMIN;  // MASSR(MDRmin:)
    const float XINC = FDIV(FSUB(FH_XMAX, FH_XMIN), (float)(FH_NX - 1));
    const float C1XPVS = FSUB(1.f, FDIV(FH_XMIN, XINC));
    const float C2XPVS = FDIV(1.f, XINC);
    s[FH_S_C1XPVS] = C1XPVS;
    s[FH_S_C2XPVS] = C2XPVS;
    s[FH_S_C1XPVS0] = C1XPVS;
    s[FH_S_C2XPVS0] = C2XPVS;
    // DTPH=MAX(GSMDT*60.,DT); DTPH=NINT(DTPH/DT)*DT (:2937-2938).  WRF's
    // microphysics_init passes GSMDT = MPDT = 0, so DTPH = DT; the launcher
    // passes that DT.
    s[FH_S_DTPH] = dtph;
    const float PI_E = 0x1.921fb6p+1f;                   // ACOS(-1.) :3006
    s[FH_S_PI_E] = PI_E;
    s[FH_S_ABFR] = -0.66f;                               // :3011
    const float BBFR = 100.f;                            // :3012
    // CBFR=20.*PI_E*PI_E*BBFR*RHOL*1.E-21*DTPH  (:3013)
    s[FH_S_CBFR] = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(20.f, PI_E), PI_E), BBFR),
                                  FH_RHOL), 1.E-21f), dtph);
    s[FH_S_RQR_DRMIN] = FMUL(FH_N0R0, MASSR[FH_MDRMIN]); // :3040
    s[FH_S_RQR_DRMAX] = FMUL(FH_N0R0, MASSR[FH_MDRMAX]); // :3041
    const float C_N0r0 = FMUL(FMUL(PI_E, FH_RHOL), FH_N0R0);   // :3044
    s[FH_S_CN0R0] = FDIV(1.E6f, FSQRT(FSQRT(C_N0r0)));   // :3045
    // ESW0=1000.*FPVS0(T0C) (:3054), the table lookup, rounding-pinned.
    {
        const float T = FH_T0C;
        float XJ1 = fh_min(fh_max(FADD(C1XPVS, FMUL(C2XPVS, T)), 1.f),
                           (float)FH_NX);
        int JX1 = fh_int(fh_min(XJ1, (float)FH_NX - 1.f));
        const real* tb = tab + FH_OFF_TBPVS0 - 1;
        float v = FADD(tb[JX1], FMUL(FSUB(XJ1, (float)JX1),
                                     FSUB(tb[JX1 + 1], tb[JX1])));
        s[FH_S_ESW0] = FMUL(1000.f, v);
    }
    s[FH_S_RFMAX] = FH_RFMAX_FOLDED;                     // 1.1**Nrime :3055
    s[FH_S_RFMX1] = 0x1.616ddep+11f;                     // PI*900. :3056
    // BETA6 (:3068-3069), RDIS=0.5.
    const float RDIS = 0.5f;
    float a = FADD(1.f, FMUL(FMUL(3.f, RDIS), RDIS));
    float b = FADD(1.f, FMUL(FMUL(4.f, RDIS), RDIS));
    float c = FADD(1.f, FMUL(FMUL(5.f, RDIS), RDIS));
    float d = FADD(1.f, FMUL(RDIS, RDIS));
    float e = FADD(1.f, FMUL(FMUL(2.f, RDIS), RDIS));
    s[FH_S_BETA6] = FDIV(FMUL(FMUL(a, b), c), FMUL(d, e));
    s[FH_S_ARAUT] = 0x1.3e14a6p+21f;   // 1.03e19/(NCW*SQRT(NCW)) :3075
    s[FH_S_QAUT0] = 0x1.12843ap-10f;   // PI*RHOL*NCW*(20.E-6)**3/6. :3086
    s[FH_S_ARCW] = 0x1.89e5dep+6f;     // 1.E6*(0.75/(PI*NCW*RHOL))**C1 :3090
    const float RH_NgC0 = 0x1.7f7ec6p+20f;    // PI*500.*1.E3 :3101
    s[FH_S_RQHAIL] = FMUL(RH_NgC0, 0x1.12e0c2p-30f);     // *(1.E-3)**3 :3102
    const float Bvhail = 0x1.17e4b2p-2f;      // 0.82*C1 :3103
    s[FH_S_BVHAIL] = Bvhail;
    s[FH_S_AVHAIL] = FMUL(1353.f, gfk_pow(FDIV(1.f, RH_NgC0), Bvhail));  // :3104
    s[FH_S_RH_NGC] = FMUL(1.E6f, gfk_pow(FDIV(1.f, RH_NgC0), FH_C1));    // :3105
    const float RH_NgT0 = 0x1.1f9f14p+22f;    // PI*300.*5.E3 :3114
    s[FH_S_RH_NGT] = FMUL(1.E6f, gfk_pow(FDIV(1.f, RH_NgT0), FH_C1));    // :3115
    // Crain=720.E18*C_NR*C_NR with C_NR=1./(PI*RHOL) folded (:3042-3043).
    s[FH_S_CRAIN] = FMUL(FMUL(720.E18f, 0x1.4dc5a8p-12f), 0x1.4dc5a8p-12f);
}

// ---------------------------------------------------------------------------
// FER_HIRES (:240-471) + EGCP01DRV (:717-1005) + EGCP01COLUMN (:1051-2548)
// for one (i, j) column per thread, in place.
//
//   th           full potential temperature (K)        (nz, ny, nx) inout
//   qv qc qr qi  mixing ratios (kg/kg)                  (nz, ny, nx) inout
//   qt           total condensate QT (kg/kg)             (nz, ny, nx) out
//   f_ice f_rain f_rimef  F_ICE_PHY / F_RAIN_PHY / F_RIMEF_PHY, inout
//   p pii rho dz8w  P_PHY (Pa), PI_PHY, RHO_PHY (kg m-3), dz8w (m), in
//   rainnc rainncv sr  (ny, nx): RAINNC += APREC*1000, RAINNCV, SR
//   refl         DBZ_col (dBZ, (nz, ny, nx)), written when refl_due != 0
//   tab          the buffer fer_hires_init_* filled
//   dt           the step DT; gid the WRF domain id GID (RHgrd_out on 1)
//
// REFLECTIVITY.  EGCP01COLUMN computes a reflectivity column DBZ_col from
// the scheme's own rain and precipitating-ice moments (:2232-2251,
// :2339-2345, :2494-2496) and EGCP01DRV discards it, so WRF v4.6.1 never
// writes REFL_10CM under mp=5.  This kernel keeps that column and writes it
// to ``refl`` on output steps: the scheme's own dBZ (clear-air value
// DBZmin = -20) instead of a field WRF leaves at its allocation value.
// The diagnostic changes no prognostic word; 10*ALOG10 uses CUDA's
// log10f, so this one diagnostic is not part of the 0 ULP claim.
// ---------------------------------------------------------------------------
// Shared native radar moments for forecast and initial-state diagnostics.
__device__ __forceinline__ float fh_snow_z(float rqs, float rime, float mass) {
    float ns = rqs / (rime * mass);
    if (rqs > 0.0025f && rime > 5.f) {
        if (rqs <= 0.005f) {
            const float b = fh_min(1.f, 400.f * rqs - 1.f);
            ns = 1.E3f * b + ns * (1.f - b);
        } else {
            const float b = 180.f * (rqs - 0.005f);
            ns = 1.E3f * (1.f - fh_min(b, 0.8f));
        }
    }
    return 1.634e13f * rqs * rqs / ns;
}
__device__ __forceinline__ float fh_rain_z(float rqr, float n0, int index, float crain) {
    const float nr = n0 * 1.E-6f * (float)index;
    return crain * rqr * rqr / nr;
}

extern "C" __global__
void fer_hires_column(real* __restrict__ th, real* __restrict__ qv,
                      real* __restrict__ qc, real* __restrict__ qr,
                      real* __restrict__ qi, real* __restrict__ qt,
                      real* __restrict__ f_ice, real* __restrict__ f_rain,
                      real* __restrict__ f_rimef,
                      const real* __restrict__ p,
                      const real* __restrict__ pii,
                      const real* __restrict__ rho,
                      const real* __restrict__ dz8w,
                      real* __restrict__ rainnc, real* __restrict__ rainncv,
                      real* __restrict__ sr,
                      real* __restrict__ refl,
                      const real* __restrict__ tab,
                      real dt, int gid, int refl_due, int nz, int ny, int nx,
                      int analysis_only, real* __restrict__ qrimef, int advected_rime)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    const real* s = tab + FH_OFF_SCALARS;
    // Lookup tables at their Fortran bounds MDRmin:MDRmax / MDImin:MDImax.
    const real* VENTR1 = tab + FH_OFF_VENTR1 - FH_MDRMIN;
    const real* VENTR2 = tab + FH_OFF_VENTR2 - FH_MDRMIN;
    const real* ACCRR = tab + FH_OFF_ACCRR - FH_MDRMIN;
    const real* MASSR = tab + FH_OFF_MASSR - FH_MDRMIN;
    const real* VRAIN = tab + FH_OFF_VRAIN - FH_MDRMIN;
    const real* VENTI1 = tab + FH_OFF_VENTI1 - FH_MDIMIN;
    const real* VENTI2 = tab + FH_OFF_VENTI2 - FH_MDIMIN;
    const real* ACCRI = tab + FH_OFF_ACCRI - FH_MDIMIN;
    const real* MASSI = tab + FH_OFF_MASSI - FH_MDIMIN;
    const real* VSNOWI = tab + FH_OFF_VSNOWI - FH_MDIMIN;
    const real* VEL_RF = tab + FH_OFF_VEL_RF;            // (2:9, 0:Nrime)

    const float PI_E = s[FH_S_PI_E];
    const float ABFR = s[FH_S_ABFR];
    const float CBFR = s[FH_S_CBFR];
    const float RQR_DRmin = s[FH_S_RQR_DRMIN];
    const float RQR_DRmax = s[FH_S_RQR_DRMAX];
    const float CN0r0 = s[FH_S_CN0R0];
    const float ESW0 = s[FH_S_ESW0];
    const float RFmax = s[FH_S_RFMAX];
    const float RFmx1 = s[FH_S_RFMX1];
    const float BETA6 = s[FH_S_BETA6];
    const float ARAUT = s[FH_S_ARAUT];
    const float QAUT0 = s[FH_S_QAUT0];
    const float ARcw = s[FH_S_ARCW];
    const float RH_NgC = s[FH_S_RH_NGC];
    const float RQhail = s[FH_S_RQHAIL];
    const float BVhail = s[FH_S_BVHAIL];
    const float AVhail = s[FH_S_AVHAIL];
    const float RH_NgT = s[FH_S_RH_NGT];
    const float Crain = s[FH_S_CRAIN];

    // FER_HIRES :323-343: the constants that follow this call's DT.
    // MY_GROWTH_RATES(DT) sets DT_ICE=(DTPH/600.)**1.5 (:3185); the table
    // MY_GROWTH=DT_ICE*MY_600*1.E-3 (:3186) is formed at its one read.
    const float DT_ICE = gfk_pow(__fdiv_rn(dt, 600.f), 1.5f);
    const float CIACW = dt * 0.25f * PI_E * 0.5f * FH_1E5_POW_C1;   // :329
    const float CIACR = PI_E * dt;                                    // :334
    const float CRACW = dt * 0.25f * PI_E * 1.0f;                     // :339
    const float BRAUT = __fdiv_rn(dt * 1.1E10f * BETA6, FH_NCW);                // :343
    const float DTPH = analysis_only ? 0.f : dt;               // EGCP01DRV(GID,DT,...) :407

    // EGCP01COLUMN :1245-1253, column-scope state.
    float ARAIN = 0.f, VRAIN1 = 0.f, VSNOW1 = 0.f, ASNOW = 0.f;
    int INDEXS0C = FH_MDIMIN;
    float RHOX0C = 22.5f;
    float TIMLT = 0.f;
    bool STRAT = false, DRZL = false;
    int INDEXRmin = FH_MDRMIN;
    float RimeF_above = 1.f;          // RimeF_col(L-1) after its update

    for (int L = 1; L <= nz; ++L) {
        const int k = nz - L;                            // KFLIP=KTE+1-L
        const size_t idx = IDX3(k, j, i);

        // ---- FER_HIRES :355-403, this level's entry state.
        const float pi_k = pii[idx];
        const float t_phy = th[idx] * pi_k;
        const float qv_in = qv[idx];
        const float Qsh = qv_in / (1.f + qv_in);         // specific humidity
        const float qc_in = qc[idx], qr_in = qr[idx], qi_in = qi[idx];
        const float QTk = qc_in + qr_in + qi_in;
        float Fice, Frain, FrimeF;
        if (qi_in <= FH_EPSQ) {
            Fice = 0.f;
            FrimeF = 1.f;
            if (t_phy < FH_T_ICEK) Fice = 1.f;
        } else {
            Fice = fh_max(0.f, fh_min(1.f, qi_in / QTk));
            FrimeF = advected_rime ? qrimef[idx] / qi_in : f_rimef[idx];
        }
        if (qr_in <= FH_EPSQ) Frain = 0.f;
        else Frain = qr_in / (qr_in + qc_in);

        // ---- EGCP01DRV :801-912, the column setup for this L.
        float CWM = QTk;
        if (L == 1 && !analysis_only) {                  // :827-830
            if (CWM <= FH_EPSQ) CWM = FH_EPSQ;
            Fice = 1.f;
            Frain = 0.f;
            FrimeF = 1.f;
        }
        const float DPCOL = rho[idx] * FH_GRAV * dz8w[idx];
        const float P_col = p[idx];
        const float THICK_col = DPCOL * FH_RGRAV;
        float T_col = t_phy;
        const float TCs = T_col - FH_T0C;
        float QV_col = fh_max(FH_EPSQ, Qsh);
        float WC_col;
        if (CWM <= FH_EPSQ1) {
            WC_col = 0.f;
            if (TCs < FH_T_ICE) Fice = 1.f; else Fice = 0.f;
            Frain = 0.f;
            FrimeF = 1.f;
        } else {
            WC_col = CWM;
        }
        float QI_col, QR_col, QW_col;
        {
            const float WCs = WC_col;
            float QIs = 0.f, QRs = 0.f, QWs = 0.f;
            if (Fice >= 1.f) QIs = WCs;
            else if (Fice <= 0.f) QWs = WCs;
            else { QIs = Fice * WCs; QWs = WCs - QIs; }
            if (QWs > 0.f && Frain > 0.f) {
                if (Frain >= 1.f) { QRs = QWs; QWs = 0.f; }
                else { QRs = Frain * QWs; QWs = QWs - QRs; }
            }
            if (QIs <= 0.f) FrimeF = 1.f;
            QI_col = QIs; QR_col = QRs; QW_col = QWs;
        }
        float RimeF_col = FrimeF;
        const float RHC_col = (gid == 1 && P_col < FH_P_RHGRD_OUT)
                              ? FH_RHGRD_OUT : FH_RHGRD_IN;
        // LBEF=MAX(1,L-1) (:1575): at L = 1 that is this level's own value.
        const float RimeF_LBEF = (L == 1) ? RimeF_col : RimeF_above;

        // ---- EGCP01COLUMN big_loop body (:1259-2545) for this L.
        bool CLEAR;
        if (ARAIN > FH_CLIMIT) { CLEAR = false; VRAIN1 = 0.f; }
        else { CLEAR = true; ARAIN = 0.f; }
        if (ASNOW > FH_CLIMIT) { CLEAR = false; VSNOW1 = 0.f; }
        else { ASNOW = 0.f; }

        const float TK = T_col;
        const float TC = TK - FH_T0C;
        const float PP = P_col;
        const float Q = QV_col;
        const float WV = Q / (1.f - Q);
        const float WC = WC_col;
        const float RHgrd = RHC_col;

        const float ESW = fh_min(1000.f * fh_fpvs0(tab, TK), 0.99f * PP);
        const float QSW = FH_EPS * ESW / (PP - ESW);
        float WS = QSW;
        float QSI = QSW;
        if (TC < 0.f) {
            const float ESI = fh_min(1000.f * fh_fpvs(tab, TK), 0.99f * PP);
            QSI = FH_EPS * ESI / (PP - ESI);
            WS = QSI;
        }
        const float QSWgrd = RHgrd * QSW;
        const float QSIgrd = RHgrd * QSI;
        const float WSgrd = RHgrd * WS;
        if (WV > WSgrd || WC > FH_EPSQ) CLEAR = false;

        float DBZ_col = -20.f;                           // DBZmin, :1279
        if (CLEAR) {                                     // :1353-1359
            STRAT = false;
            DRZL = false;
            INDEXRmin = FH_MDRMIN;
            TIMLT = 0.f;
        } else {
            const float RHO = PP / (FH_RD * TK * (1.f + FH_EPS1 * Q));
            const float RRHO = 1.f / RHO;
            const float DTRHO = DTPH * RHO;
            const float BLDTRH = FH_BLEND * DTRHO;
            const float THICK = THICK_col;

            float ARAINnew = 0.f, ASNOWnew = 0.f;
            const float QI = QI_col;
            float QInew = 0.f;
            const float QR = QR_col;
            float QRnew = 0.f;
            const float QW = QW_col;
            float QWnew = 0.f;

            float PCOND = 0.f, PIDEP = 0.f, PINIT = 0.f, PIACW = 0.f,
                  PIACWI = 0.f, PIACWR = 0.f, PIACR = 0.f, PICND = 0.f,
                  PIEVP = 0.f, PIMLT = 0.f, PRAUT = 0.f, PRACW = 0.f,
                  PREVP = 0.f;
            float NSmICE = 0.f;
            int INDEXR = FH_MDRMIN;
            int INDEXR1 = INDEXR;
            int INDEXR2 = INDEXR;
            float N0r = 0.f;
            float DUM1 = fh_min(0.f, TC);
            float DUM = FH_XMIMAX * gfk_exp(FH_XMIEXP * DUM1);
            int INDEXS = min(FH_MDIMAX, max(FH_MDIMIN, fh_int(DUM)));
            float VCI = 0.f;
            float VSNOW = 0.f;
            float VRAIN2 = 0.f;
            float RimeF1 = 1.f;
            float DUM2 = 0.f, DUM3 = 0.f;

            const float TK2 = 1.f / (TK * TK);
            const float TFACTOR = sqrtf(TK * TK * TK) / (TK + 120.f);
            const float DYNVIS = 1.496E-6f * TFACTOR;
            const float THERM_COND = 2.116E-3f * TFACTOR;
            const float DIFFUS = 8.794E-5f * gfk_pow(TK, 1.81f) / PP;
            const float GAMMAS = fh_min(1.5f, gfk_pow(1.E5f / PP, FH_C1));
            const float GAMMAR = gfk_pow(FH_RHO0 / RHO, .4f);

            bool ICE_logical;
            float QLICE = 0.f, QTICE = 0.f;
            if (TC < 0.f || QI > FH_EPSQ || ASNOW > FH_CLIMIT) {
                ICE_logical = true;
            } else {
                ICE_logical = false;
                QLICE = 0.f;
                QTICE = 0.f;
            }
            // T_ICE <= -100. (:1473) is false for T_ICE = -40.
            bool RAIN_logical = false;
            if (ARAIN > FH_CLIMIT || QR > FH_EPSQ) RAIN_logical = true;

            float NInuclei = 0.f, QSmICE = 0.f, Rcw = 0.f;
            float TOT_ICE = 0.f, PILOSS = 0.f, VEL_INC = 0.f,
                  FLIMASS = 0.f, RQLICE = 0.f, NLICE = 0.f;
            if (ICE_logical) {                           // ice_test :1484
                NInuclei = 0.f;
                NSmICE = 0.f;
                QSmICE = 0.f;
                Rcw = 0.f;
                if (TC < 0.f) {
                    const float NSImax = fh_max(FH_NSI_MAX,
                        0.1f * RHO * QI / MASSI[FH_MDIMIN]);
                    NInuclei = fh_min(0.01f * gfk_exp(-0.6f * TC), NSImax);
                    NInuclei = fh_min(5.f * gfk_exp(-0.304f * TC), NInuclei);
                    if (QI > FH_EPSQ) {
                        DUM = RRHO * MASSI[FH_MDIMIN];
                        NSmICE = fh_min(NInuclei, QI / DUM);
                        QSmICE = NSmICE * DUM;
                    }
                }
                if (QI <= FH_EPSQ && ASNOW <= FH_CLIMIT) {   // init_ice
                    TOT_ICE = 0.f;
                    PILOSS = 0.f;
                    RimeF1 = 1.f;
                    VEL_INC = GAMMAS;
                    VSNOW = 0.f;
                    VSNOW1 = 0.f;
                    VCI = 0.f;
                    FLIMASS = 1.f;
                    QLICE = 0.f;
                    RQLICE = 0.f;
                    QTICE = 0.f;
                    NLICE = 0.f;
                } else {
                    TOT_ICE = THICK * QI + FH_BLEND * ASNOW;
                    PILOSS = -TOT_ICE / THICK;
                    const float QLgICE = fh_max(0.f, QI - QSmICE);
                    VCI = GAMMAS * VSNOWI[FH_MDIMIN];
                    RimeF1 = (RimeF_col * THICK * QLgICE
                              + RimeF_LBEF * FH_BLEND * ASNOW) / TOT_ICE;
                    if (RimeF1 > 2.f) {
                        DUM3 = RH_NgC * gfk_pow(RHO * QLgICE, FH_C1);
                        DUM2 = RH_NgT * gfk_pow(RHO * QLgICE, FH_C1);
                        if (RimeF1 >= 10.f) {
                            DUM = DUM3;
                        } else if (RimeF1 >= 5.f) {
                            DUM = 0.2f * (RimeF1 - 5.f);
                            DUM = DUM3 * DUM + DUM2 * (1.f - DUM);
                        } else {
                            DUM1 = (float)INDEXS;
                            DUM = 0.33333f * (RimeF1 - 2.f);
                            DUM = DUM2 * DUM + DUM1 * (1.f - DUM);
                        }
                        INDEXS = min(FH_MDIMAX, max(FH_MDIMIN, fh_int(DUM)));
                    }
                    const float EMAIRI = THICK + BLDTRH * VSNOW1;
                    QLICE = (THICK * QLgICE + FH_BLEND * ASNOW) / EMAIRI;
                    RQLICE = RHO * QLICE;
                    QTICE = QLICE + QSmICE;
                    FLIMASS = QLICE / QTICE;
                    for (int IPASS = 1; IPASS <= 2; ++IPASS) {   // two_pass
                        DUM = 1.E-6f * (float)INDEXS;
                        const float RFmx = RFmx1 * DUM * DUM * DUM / MASSI[INDEXS];
                        RimeF1 = fh_min(RimeF1, RFmx);
                        float VrimeF;
                        if (RimeF1 <= 1.f) {
                            RimeF1 = 1.f;
                            VrimeF = 1.f;
                        } else {
                            RimeF1 = fh_min(RimeF1, RFmax);
                            const int IXS = max(2, min(INDEXS / 100, 9));
                            const float XRF = 10.492f * gfk_log(RimeF1);
                            const int IXRF = max(0, min(fh_int(XRF), FH_NRIME));
                            if (IXRF >= FH_NRIME) {
                                VrimeF = VEL_RF[(IXS - 2) + 8 * FH_NRIME];
                            } else {
                                const float v0 = VEL_RF[(IXS - 2) + 8 * IXRF];
                                const float v1 = VEL_RF[(IXS - 2) + 8 * (IXRF + 1)];
                                VrimeF = v0 + (XRF - (float)IXRF) * (v1 - v0);
                            }
                            VrimeF = fh_max(1.f, VrimeF);
                        }
                        VEL_INC = GAMMAS * VrimeF;
                        VSNOW = VEL_INC * VSNOWI[INDEXS];
                        if (RimeF1 >= 5.f && INDEXS == FH_MDIMAX
                            && RQLICE > RQhail) {
                            DUM = GAMMAS * AVhail * gfk_pow(RQLICE, BVhail);
                            if (DUM > VSNOW) {
                                VSNOW = DUM;
                                VEL_INC = VSNOW / VSNOWI[INDEXS];
                            }
                        }
                        const float XLIMASS = RimeF1 * MASSI[INDEXS];
                        NLICE = RQLICE / XLIMASS;
                        if (IPASS >= 2
                            || (NLICE >= FH_NLIMIN && NLICE <= FH_NSI_MAX))
                            break;
                        NLICE = fh_max(FH_NLIMIN, fh_min(FH_NSI_MAX, NLICE));
                        const float XLI = RQLICE / (NLICE * RimeF1);
                        if (XLI <= MASSI[FH_MDIMIN]) {
                            INDEXS = FH_MDIMIN;
                        } else if (XLI <= MASSI[450]) {
                            const float DLI = 9.5885E5f * gfk_pow(XLI, .42066f);
                            INDEXS = min(FH_MDIMAX, max(FH_MDIMIN, fh_int(DLI)));
                        } else if (XLI < MASSI[FH_MDIMAX]) {
                            const float DLI = 3.9751E6f * gfk_pow(XLI, .49870f);
                            INDEXS = min(FH_MDIMAX, max(FH_MDIMIN, fh_int(DLI)));
                        } else {
                            INDEXS = FH_MDIMAX;
                        }
                    }
                }
            }

            // :1688-1719 melting-layer and drizzle flags.
            if (TC <= 0.f) {
                STRAT = false;
                INDEXRmin = FH_MDRMIN;
                TIMLT = 0.f;
                INDEXS0C = INDEXS;
                RHOX0C = 22.5f * fh_max(1.f, fh_min(RimeF1, 40.f));
            } else {
                if (!RAIN_logical) {
                    STRAT = false;
                    INDEXRmin = FH_MDRMIN;
                    if (!ICE_logical) TIMLT = 0.f;
                } else {
                    if (TIMLT > FH_EPSQ && RHOX0C <= 225.f) {
                        STRAT = true;
                    } else {
                        STRAT = false;
                        INDEXRmin = FH_MDRMIN;
                    }
                    if (STRAT && INDEXRmin <= FH_MDRMIN) {
                        INDEXRmin = fh_int((float)INDEXS0C
                                           * gfk_pow(0.001f * RHOX0C, FH_C1));
                        INDEXRmin = max(FH_MDRMIN, min(INDEXRmin, FH_INDEXRSTRMAX));
                    }
                }
            }
            if (STRAT || TIMLT > FH_EPSQ) DRZL = false;
            else DRZL = true;

            if (analysis_only) {
                float zs = 0.f, zr = 0.f;
                const float rqs = RHO * QLICE;
                if (rqs > 0.f) zs = fh_snow_z(rqs, RimeF1, MASSI[INDEXS]);
                const float rqr = RHO * QR;
                if (rqr > FH_EPSQ) {
                    // Native rain_pass PSD reconstruction, without fallout.
                    if (rqr <= RQR_DRmin) {
                        INDEXR = FH_MDRMIN;
                        N0r = rqr / MASSR[INDEXR];
                    } else if (DRZL) {
                        DUM = fh_max(1.f, 0.5E-3f / rqr);
                        N0r = fh_min(1.E9f, FH_N0R0 * DUM * sqrtf(DUM));
                        DUM1 = rqr / (FH_PI * FH_RHOL * N0r);
                        DUM = 1.E6f * sqrtf(sqrtf(DUM1));
                        INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                        N0r = rqr / MASSR[INDEXR];
                    } else if (rqr >= RQR_DRmax) {
                        INDEXR = FH_MDRMAX;
                        N0r = rqr / MASSR[INDEXR];
                    } else {
                        N0r = FH_N0R0;
                        DUM = CN0r0 * sqrtf(sqrtf(rqr));
                        INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                    }
                    zr = fh_rain_z(rqr, N0r, INDEXR, Crain);
                }
                refl[idx] = zs + zr > 0.01f ? 10.f * log10f(zs + zr) : -20.f;
                RimeF_above = RimeF_col;
                continue;
            }

            // :1729-1751 autoconversion and riming.
            if (QW > FH_EPSQ && TC >= FH_T_ICE) {
                DUM2 = RHO * QW;
                if (DUM2 > QAUT0) {
                    DUM2 = DUM2 * DUM2;
                    DUM = BRAUT * DUM2 * QW;
                    DUM1 = ARAUT * DUM2;
                    PRAUT = fh_min(QW, DUM * (1.f - gfk_exp(-(DUM1 * DUM1))));
                }
                if (QLICE > FH_EPSQ) {
                    const float FWS = fh_min(1.f, CIACW * VEL_INC * NLICE
                                                  * ACCRI[INDEXS]);
                    PIACW = FWS * QW;
                    if (TC < 0.f) {
                        PIACWI = PIACW;
                        Rcw = ARcw * gfk_pow(DUM2, FH_C1);
                    }
                }
            }

            // :1757-1853 homogeneous freezing and ice deposition.
            if (TC < FH_T_ICE && (WV > QSWgrd || QW > FH_EPSQ)) {
                PIACW = QW;
                PIACWI = PIACW;
                Rcw = 0.f;
                DUM1 = TK + FH_XLF1 * PIACW;
                DUM2 = WV;
                DUM = fh_min(1000.f * fh_fpvs(tab, DUM1), 0.99f * PP);
                DUM = RHgrd * FH_EPS * DUM / (PP - DUM);
                if (DUM2 > DUM)
                    PIDEP = fh_deposit(tab, PP, DUM1, DUM2, RHgrd);
            } else if (TC < 0.f) {
                const float DENOMI = 1.f + FH_XLS3 * QSI * TK2;
                const float DWVi = WV - QSIgrd;
                const float PIDEP_max = fh_max_first(PILOSS, DWVi / DENOMI);
                float DIDEP = 0.f;
                if (QTICE > 0.f) {
                    const float ABI = 1.f / (RHO * FH_XLS2 * QSI * TK2 / THERM_COND
                                             + 1.f / DIFFUS);
                    DUM = gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), FH_C2);
                    float SFACTOR = sqrtf(GAMMAS) * DUM;
                    const float VENTIS = (VENTI1[FH_MDIMIN]
                                          + SFACTOR * VENTI2[FH_MDIMIN]) * NSmICE;
                    SFACTOR = sqrtf(VEL_INC) * DUM;
                    const float VENTIL = (VENTI1[INDEXS]
                                          + SFACTOR * VENTI2[INDEXS]) * NLICE;
                    DIDEP = ABI * (VENTIL + VENTIS) * DTPH;
                }
                if (WV > QSWgrd && TC < FH_T_ICE_INIT && NSmICE < NInuclei) {
                    const int INDEX_MY = max(FH_MY_T1,
                                             min(fh_int(.5f - TC), FH_MY_T2));
                    DUM = fh_max(0.f, NInuclei - NSmICE);
                    // MY_GROWTH(INDEX_MY) = DT_ICE*MY_600(INDEX_MY)*1.E-3
                    const float MY_GROWTH = DT_ICE * FH_MY_600[INDEX_MY - 1] * 1.E-3f;
                    PINIT = fh_max(0.f, DUM * MY_GROWTH * RRHO);
                }
                if (DWVi > 0.f)
                    PIDEP = fh_min_first(DWVi * DIDEP + PINIT, PIDEP_max);
                else if (DWVi < 0.f)
                    PIDEP = fh_max_first(DWVi * DIDEP, PIDEP_max);
            }

            // :1858-1865 cloud water condensation.
            if (TC >= FH_T_ICE && (QW > FH_EPSQ || WV > QSWgrd)) {
                DUM1 = QW - PIACWI;
                DUM2 = TK + FH_XLS1 * PIDEP + FH_XLF1 * PIACWI;
                DUM3 = WV - PIDEP;
                PCOND = fh_condense(tab, PP, DUM1, DUM2, DUM3, RHgrd);
            }

            // :1870-1876 Schumann-Ludlam limit.
            float TCC = TC + FH_XLV1 * PCOND + FH_XLS1 * PIDEP + FH_XLF1 * PIACWI;
            if (TCC > 0.f) {
                PIACWI = 0.f;
                Rcw = 0.f;
                PIDEP = 0.f;
                TCC = TC + FH_XLV1 * PCOND;
            }

            // :1878-1921 melting and evaporation of ice.
            if (TC > 0.f && TCC > 0.f && ICE_logical) {
                const float SFACTOR = sqrtf(VEL_INC)
                    * gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), FH_C2);
                const float VENTIL = NLICE * (VENTI1[INDEXS]
                                              + SFACTOR * VENTI2[INDEXS]);
                const float AIEVP = VENTIL * DIFFUS * DTPH;
                float DIEVP;
                if (AIEVP < FH_XRATIO) DIEVP = AIEVP;
                else DIEVP = 1.f - gfk_exp(-AIEVP);
                const float QSW0 = FH_EPS * ESW0 / (PP - ESW0);
                const float DWV0 = fh_min(WV, QSW) - QSW0;
                DUM = QW + PCOND;
                if (WV < QSW && DUM <= FH_EPSQ) {
                    DUM = DWV0 * DIEVP;
                    PIEVP = fh_max(fh_min(0.f, DUM), PILOSS);
                    PICND = fh_max(0.f, DUM);
                }
                PIMLT = __fdiv_rn(THERM_COND * TCC * VENTIL * RRHO * DTPH, FH_XLF);
                DUM1 = fh_max(0.f, __fdiv_rn(TCC + FH_XLV1 * PIEVP, FH_XLF1));
                PIMLT = fh_min(PIMLT, DUM1);
                DUM = PIEVP - PIMLT;
                if (DUM < PILOSS) {
                    DUM1 = PILOSS / DUM;
                    PIMLT = PIMLT * DUM1;
                    PIEVP = PIEVP * DUM1;
                }
            }

            // :1937-2051 rain.
            float TOT_RAIN = 0.f, QTRAIN = 0.f, PRLOSS = 0.f, RQR = 0.f;
            if (RAIN_logical) {
                TOT_RAIN = THICK * QR + FH_BLEND * ARAIN;
                PRLOSS = -TOT_RAIN / THICK;
                QTRAIN = TOT_RAIN / (THICK + BLDTRH * VRAIN1);
                RQR = RHO * QTRAIN;
                if (STRAT && RQR > 1.E-3f) STRAT = false;
                if (DRZL && PIMLT > FH_EPSQ) DRZL = false;
                if (RQR <= RQR_DRmin) {
                    INDEXR = FH_MDRMIN;
                    N0r = RQR / MASSR[INDEXR];
                } else if (DRZL) {
                    DUM = fh_max(1.0f, 0.5E-3f / RQR);
                    N0r = fh_min(1.E9f, FH_N0R0 * DUM * sqrtf(DUM));
                    DUM1 = RQR / (FH_PI * FH_RHOL * N0r);
                    DUM = 1.E6f * sqrtf(sqrtf(DUM1));
                    INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                    N0r = RQR / MASSR[INDEXR];
                } else if (RQR >= RQR_DRmax) {
                    INDEXR = FH_MDRMAX;
                    N0r = RQR / MASSR[INDEXR];
                } else {
                    N0r = FH_N0R0;
                    DUM = CN0r0 * sqrtf(sqrtf(RQR));
                    INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                    if (STRAT && INDEXR < INDEXRmin) {
                        INDEXR = INDEXRmin;
                        N0r = RQR / MASSR[INDEXR];
                    }
                }
                VRAIN2 = GAMMAR * VRAIN[INDEXR];

                if (TC < FH_T_ICE) {
                    PIACR = -PRLOSS;
                } else {
                    const float DWVr = WV - PCOND - QSW;
                    DUM = QW + PCOND;
                    if (DWVr < 0.f && DUM <= FH_EPSQ) {
                        const float RFACTOR = sqrtf(GAMMAR)
                            * gfk_pow(RHO / (DIFFUS * DIFFUS * DYNVIS), FH_C2);
                        const float ABW = 1.f / (RHO * FH_XLV2 * QSW * TK2
                                                 / THERM_COND + 1.f / DIFFUS);
                        const float VENTR = N0r * (VENTR1[INDEXR]
                                                   + RFACTOR * VENTR2[INDEXR]);
                        const float CREVP = ABW * VENTR * DTPH;
                        PREVP = fh_max_first(DWVr * CREVP, PRLOSS);
                    } else if (QW > FH_EPSQ) {
                        const float FWR = CRACW * GAMMAR * N0r * ACCRR[INDEXR];
                        PRACW = fh_min(1.f, FWR) * QW;
                    }
                    if (ICE_logical && TC < 0.f && TCC < 0.f) {
                        DUM = .001f * (float)INDEXR;
                        DUM = (gfk_exp(ABFR * TC) - 1.f) * DUM * DUM * DUM * DUM
                              * DUM * DUM * DUM;
                        PIACR = fh_min_first(CBFR * N0r * RRHO * DUM, QTRAIN);
                        if (QLICE > FH_EPSQ) {
                            DUM = GAMMAR * VRAIN[INDEXR];
                            DUM1 = DUM - VSNOW;
                            DUM2 = sqrtf(DUM1 * DUM1 + .04f * DUM * VSNOW);
                            DUM1 = 5.E-12f * (float)INDEXR * (float)INDEXR
                                   + 2.E-12f * (float)INDEXR * (float)INDEXS
                                   + .5E-12f * (float)INDEXS * (float)INDEXS;
                            const float FIR = fh_min(1.f, CIACR * NLICE * DUM1 * DUM2);
                            PIACR = fh_min_first(PIACR + FIR * QTRAIN, QTRAIN);
                        }
                        DUM = PREVP - PIACR;
                        if (DUM < PRLOSS) {
                            DUM1 = PRLOSS / DUM;
                            PREVP = DUM1 * PREVP;
                            PIACR = DUM1 * PIACR;
                        }
                    }
                }
            }
            INDEXR1 = INDEXR;

            // :2066-2085 cloud water budget.
            DUM1 = PIACW + PRAUT + PRACW - fh_min(0.f, PCOND);
            if (DUM1 > QW) {
                DUM = QW / DUM1;
                PIACW = DUM * PIACW;
                PIACWI = DUM * PIACWI;
                PRAUT = DUM * PRAUT;
                PRACW = DUM * PRACW;
                if (PCOND < 0.f) PCOND = DUM * PCOND;
            }
            PIACWR = PIACW - PIACWI;
            const float DELW = PCOND - PIACW - PRAUT - PRACW;
            QWnew = QW + DELW;
            if (QWnew <= FH_EPSQ) QWnew = 0.f;
            if (QW > 0.f && QWnew != 0.f) {
                DUM = QWnew / QW;
                if (DUM < FH_TOLER) QWnew = 0.f;
            }

            // :2089-2094 temperature and vapour.
            const float DELT = FH_XLV1 * (PCOND + PIEVP + PICND + PREVP)
                             + FH_XLS1 * PIDEP + FH_XLF1 * (PIACWI + PIACR - PIMLT);
            const float Tnew = TK + DELT;
            const float DELV = -PCOND - PIDEP - PIEVP - PICND - PREVP;
            const float WVnew = WV + DELV;

            // :2118-2253 ice.
            float RimeF = 1.f;
            float Zsnow = 0.f, Zrain = 0.f;
            if (ICE_logical) {
                const float DELI = PIDEP + PIEVP + PIACWI + PIACR - PIMLT;
                float TOT_ICEnew = TOT_ICE + THICK * DELI;
                if (TOT_ICE > 0.f && TOT_ICEnew != 0.f) {
                    DUM = TOT_ICEnew / TOT_ICE;
                    if (DUM < FH_TOLER) TOT_ICEnew = 0.f;
                }
                if (TOT_ICEnew <= FH_CLIMIT) {
                    TOT_ICEnew = 0.f;
                    RimeF = 1.f;
                    QInew = 0.f;
                    ASNOWnew = 0.f;
                } else {
                    float PSDEP;
                    if (PINIT <= FH_EPSQ) PSDEP = fh_max(0.f, PIDEP);
                    else PSDEP = 0.f;
                    const float DELS = PIACWI + PIACR + PSDEP;
                    if (DELS <= FH_EPSQ || Tnew >= FH_T0C) {
                        RimeF = RimeF1;
                    } else {
                        DUM = 1.E-6f * (float)INDEXS;
                        DUM1 = DUM * DUM * DUM / MASSI[INDEXS];
                        const float RFmx = RFmx1 * DUM1;
                        float RFrime;
                        if (PIACWI > 0.f && Rcw > 0.f) {
                            DUM = -Rcw * VSNOW / fh_min(-0.5f, TC);
                            DUM = fh_min(12.14f, fh_max(0.275f, DUM));
                            DUM = 300.f * gfk_pow(DUM, 0.44f);
                            DUM = fh_min(900.f, fh_max(170.f, DUM));
                            RFrime = FH_PI * DUM * DUM1;
                        } else {
                            RFrime = 1.f;
                        }
                        const float RFnew = (PSDEP + RFrime * PIACWI
                                             + RFmx * PIACR) / DELS;
                        DUM2 = TOT_ICE + THICK * DELS;
                        DUM3 = RimeF1 * TOT_ICE + RFnew * THICK * DELS;
                        if (DUM2 <= FH_CLIMIT) RimeF = RimeF1;
                        else RimeF = fh_min_first(RFmx, fh_max(1.f, DUM3 / DUM2));
                    }
                    DUM1 = BLDTRH * (FLIMASS * VSNOW + (1.f - FLIMASS) * VCI);
                    QInew = TOT_ICEnew / (THICK + DUM1);
                    if (QInew <= FH_EPSQ) QInew = 0.f;
                    if (QI > 0.f && QInew != 0.f) {
                        DUM = QInew / QI;
                        if (DUM < FH_TOLER) QInew = 0.f;
                    }
                    ASNOWnew = QInew * DUM1;
                    if (ASNOW > 0.f && ASNOWnew != 0.f) {
                        DUM = ASNOWnew / ASNOW;
                        if (DUM < FH_TOLER) ASNOWnew = 0.f;
                    }
                    // :2232-2251, the precipitating-ice reflectivity.
                    const float RQSnew = FLIMASS * RHO * QInew;
                    if (refl_due && RQSnew > 0.f) {
                        Zsnow = fh_snow_z(RQSnew, RimeF, MASSI[INDEXS]);
                    }
                }
            }

            // :2268-2347 rain update.
            TIMLT = TIMLT + PIMLT * THICK;
            const float DELR = PRAUT + PRACW + PIACWR - PIACR + PIMLT + PREVP + PICND;
            float TOT_RAINnew = TOT_RAIN + THICK * DELR;
            if (TOT_RAIN > 0.f && TOT_RAINnew != 0.f) {
                DUM = TOT_RAINnew / TOT_RAIN;
                if (DUM < FH_TOLER) TOT_RAINnew = 0.f;
            }
            if (TOT_RAINnew <= FH_CLIMIT) {
                TOT_RAINnew = 0.f;
                VRAIN2 = 0.f;
                QRnew = 0.f;
                ARAINnew = 0.f;
            } else {
                QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
                float RQRnew = RHO * QRnew;
                if (STRAT && RQRnew > 1.E-3f) STRAT = false;
                if (DRZL && TIMLT > FH_EPSQ) DRZL = false;
                INDEXR2 = INDEXR1;
                for (int IPASS = 1; IPASS <= 3; ++IPASS) {   // rain_pass
                    if (RQRnew <= RQR_DRmin) {
                        INDEXR = FH_MDRMIN;
                        N0r = RQRnew / MASSR[INDEXR];
                    } else if (DRZL) {
                        DUM = fh_max(1.0f, 0.5E-3f / RQRnew);
                        N0r = fh_min(1.E9f, FH_N0R0 * DUM * sqrtf(DUM));
                        DUM1 = RQRnew / (FH_PI * FH_RHOL * N0r);
                        DUM = 1.E6f * sqrtf(sqrtf(DUM1));
                        INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                        N0r = RQRnew / MASSR[INDEXR];
                    } else if (RQRnew >= RQR_DRmax) {
                        INDEXR = FH_MDRMAX;
                        N0r = RQRnew / MASSR[INDEXR];
                    } else {
                        N0r = FH_N0R0;
                        DUM = CN0r0 * sqrtf(sqrtf(RQRnew));
                        INDEXR = fh_int(fh_max(FH_XMRMIN, fh_min(DUM, FH_XMRMAX)));
                        if (STRAT && INDEXR < INDEXRmin) {
                            INDEXR = INDEXRmin;
                            N0r = RQRnew / MASSR[INDEXR];
                        }
                    }
                    VRAIN2 = GAMMAR * VRAIN[INDEXR];
                    QRnew = TOT_RAINnew / (THICK + BLDTRH * VRAIN2);
                    RQRnew = RHO * QRnew;
                    const int IDR = abs(INDEXR - INDEXR2);
                    INDEXR2 = INDEXR;
                    if (IDR <= 5) break;
                }
                if (QRnew <= FH_EPSQ) QRnew = 0.f;
                if (QR > 0.f && QRnew != 0.f) {
                    DUM = QRnew / QR;
                    if (DUM < FH_TOLER) QRnew = 0.f;
                }
                ARAINnew = BLDTRH * VRAIN2 * QRnew;
                if (ARAIN > 0.f && ARAINnew != 0.f) {
                    DUM = ARAINnew / ARAIN;
                    if (DUM < FH_TOLER) ARAINnew = 0.f;
                }
                // :2339-2345, the rain reflectivity.
                const float RQRo = RHO * QRnew;
                if (refl_due && RQRo > FH_EPSQ) {
                    Zrain = fh_rain_z(RQRo, N0r, INDEXR2, Crain);
                }
            }
            const float WCnew = QWnew + QRnew + QInew;
            // :2357-2476, the budget check and the PRINT_diag writes, set
            // only locals that are re-set before their next read.

            // :2482-2500 the level and the falling fluxes.
            T_col = Tnew;
            QV_col = fh_max_first(FH_EPSQ, WVnew / (1.f + WVnew));
            WC_col = fh_max_first(FH_EPSQ, WCnew);
            QI_col = fh_max_first(FH_EPSQ, QInew);
            QR_col = fh_max_first(FH_EPSQ, QRnew);
            QW_col = fh_max_first(FH_EPSQ, QWnew);
            RimeF_col = RimeF;
            ASNOW = ASNOWnew;
            VSNOW1 = VSNOW;
            ARAIN = ARAINnew;
            VRAIN1 = VRAIN2;
            const float Ztot = Zrain + Zsnow;                // :2494-2496
            if (Ztot > 0.01f) DBZ_col = 10.f * log10f(Ztot);
        }
        RimeF_above = RimeF_col;
        if (refl_due) refl[idx] = DBZ_col;

        if (analysis_only) continue;

        // ---- EGCP01DRV :936-960, storage update.
        float Fice_o, FrimeF_o, Frain_o;
        if (QI_col <= FH_EPSQ) {
            Fice_o = 0.f;
            if (T_col < FH_T_ICEK) Fice_o = 1.f;
            FrimeF_o = 1.f;
        } else {
            Fice_o = fh_max(0.f, fh_min(1.f, QI_col / WC_col));
            FrimeF_o = fh_max(1.f, RimeF_col);
        }
        if (QR_col <= FH_EPSQ) Frain_o = 0.f;
        else Frain_o = QR_col / (QR_col + QW_col);

        // ---- FER_HIRES :416-445, back to the model's arrays.
        // The stock O0 invalid-temperature path carries x86's negative
        // quiet NaN. Its payload is an output word in the strict oracle.
        th[idx] = isnan(T_col) ? __int_as_float(0xffc00000) : T_col / pi_k;
        qv[idx] = QV_col / (1.f - QV_col);
        const float WCo = WC_col;
        float QIo = 0.f, QRo = 0.f, QCo = 0.f;
        if (Fice_o >= 1.f) QIo = WCo;
        else if (Fice_o <= 0.f) QCo = WCo;
        else { QIo = Fice_o * WCo; QCo = WCo - QIo; }
        if (QCo > 0.f && Frain_o > 0.f) {
            if (Frain_o >= 1.f) { QRo = QCo; QCo = 0.f; }
            else { QRo = Frain_o * QCo; QCo = QCo - QRo; }
        }
        if (!advected_rime) qt[idx] = WCo;
        qi[idx] = QIo;
        qr[idx] = QRo;
        qc[idx] = QCo;
        if (advected_rime) {
            qrimef[idx] = QIo * FrimeF_o;
        } else {
            f_ice[idx] = Fice_o;
            f_rain[idx] = Frain_o;
            f_rimef[idx] = FrimeF_o;
        }
    }

    if (analysis_only) return;

    // ---- EGCP01DRV :967-974 and FER_HIRES :449-454, the surface.
    const float APREC = (ARAIN + ASNOW) * FH_RRHOL;
    const size_t sidx = (size_t)j * nx + i;
    sr[sidx] = (APREC < 1.E-8f) ? 0.f : FH_RRHOL * ASNOW / APREC;
    rainnc[sidx] = APREC * 1000.f + rainnc[sidx];
    rainncv[sidx] = APREC * 1000.f;
}
