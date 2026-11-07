// sas.cu -- WRF v4.6.1 2015 GFS SAS deep convection with the ARW shallow
// branch (cu_physics = 94, module_cu_sas.F), transcribed to CUDA.  WOOF's own
// work, Apache-2.0; WRF's code is public domain (WRF LICENSE.txt).
//
// Stages, in WRF's call order, each its own entry point so each is graded
// against its own oracle record (tools/sas_wrf461_oracle):
//   sas_tables       module_gfs_funcphys GFUNCPHYS: the gpvs, gtdp, gthe,
//                    gtma and gtlcl tables SAS reads (fpvs/ftdp/fthe/stma/
//                    ftlcl), built here by WOOF's own binary64 exp and pow.
//   sas_stage_deep   SASCNVN (:2599-4441) on CU_SAS's kind_phys inputs.
//   sas_stage_shal   OLD_ARW_SHALCV (:2297) with MSTADBT3 (:2517) and
//                    TRIDI2T3 (:2475).
//   sas_prepare_*    CU_SAS inputs, ordered column sums and deep limits.
//   sas_deep_prepare  Independent adjacent-level SAS thermodynamics.
//   sas_column       SASCNVN, the shallow branch and the rain split.
//   sas_finish       Level-parallel CU_SAS tendencies.
//
// Arithmetic.  kind_phys is binary64 (module_gfs_machine.F), WRF's REAL is
// binary32, and module_cu_sas.F mixes them the way gfortran's -O2
// -fno-fast-math -ffp-contract=off build evaluates them:
// Strict numerics only:
//   * every binary64 operation is a rounding-pinned intrinsic through SD
//     (__dadd_rn ...), so NVRTC's default --fmad=true can never contract a
//     multiply into an add the x86-64 build rounded twice;
//   * the few binary32 operations of the driver (QV/(1.+QV), PCPS*.001,
//     W+W, DT*STEPCU, 1./FLOAT(KBCON)) are __f*_rn intrinsics;
//   * a literal without _kind_phys is a default REAL: its value is the
//     float32 rounding of the decimal, then widened exactly -- SL(.002) is
//     (double)0.002f, never 0.002;
//   * x**2 / x**3 with integer exponents are multiplications ((x*x)*x);
//     real exponents call WOOF's own binary64 pow, and EXP WOOF's own exp,
//     from the shared binary64 libm header the loader prepends for this
//     module (glibc_flt64.cuh);
//   * MAX(a,b) = (a > b || isnan(a)) ? a : b, MIN likewise, as the gfortran
//     oracle toolchain evaluates them (ties return the second argument).
//
// One thread owns each ordered column recurrence. Independent level work
// uses a two-dimensional grid.  WRF runs a tile row; the oracle proves the
// row call equals the one-column call word for word on every case
// (run_sas.f90 "row"), and module_cu_sas.F's only cross-column statements
// are early RETURNs taken when every column is already inactive.
//
// Storage.  Inputs and outputs are level-major, element (k, col) at
// [(k-1)*ncol + col], interface arrays with nz+1 rows.  The per-column
// working arrays live in one global binary64 workspace, slot-major with the
// column fastest: slot s, level k at ws[((s*kw) + k-1)*ncolw + col].
#ifndef SAS_CU
#define SAS_CU

// Strict mode preserves the native scalar operation order and libm words.
// Fast mode permits CUDA contraction and uses CUDA's binary64 exp/pow.
#ifndef SAS_STRICT
#define SAS_STRICT 0
#endif
// ---- kind_phys binary64 ----------------------------------------------------
struct SD {
    double v;
    __device__ __forceinline__ SD() {}
    __device__ __forceinline__ SD(double x) : v(x) {}
};
#if SAS_STRICT
__device__ __forceinline__ SD operator+(SD a, SD b) { return SD(__dadd_rn(a.v, b.v)); }
__device__ __forceinline__ SD operator-(SD a, SD b) { return SD(__dsub_rn(a.v, b.v)); }
__device__ __forceinline__ SD operator*(SD a, SD b) { return SD(__dmul_rn(a.v, b.v)); }
__device__ __forceinline__ SD operator/(SD a, SD b) { return SD(__ddiv_rn(a.v, b.v)); }
#else
__device__ __forceinline__ SD operator+(SD a, SD b) { return SD(a.v + b.v); }
__device__ __forceinline__ SD operator-(SD a, SD b) { return SD(a.v - b.v); }
__device__ __forceinline__ SD operator*(SD a, SD b) { return SD(a.v * b.v); }
// Fast mode shares binary64 reciprocals across common denominators in
// the heat, moisture and momentum updates. No kind_phys value is narrowed.
// Keep IEEE division for denominators whose reciprocal overflows.
__device__ __forceinline__ SD operator/(SD a, SD b) {
    const double inverse = 1.0 / b.v;
    return SD(isfinite(inverse) ? a.v * inverse : a.v / b.v);
}
#endif
__device__ __forceinline__ SD operator-(SD a) { return SD(-a.v); }
__device__ __forceinline__ bool operator<(SD a, SD b) { return a.v < b.v; }
__device__ __forceinline__ bool operator>(SD a, SD b) { return a.v > b.v; }
__device__ __forceinline__ bool operator<=(SD a, SD b) { return a.v <= b.v; }
__device__ __forceinline__ bool operator>=(SD a, SD b) { return a.v >= b.v; }
__device__ __forceinline__ bool operator==(SD a, SD b) { return a.v == b.v; }
__device__ __forceinline__ SD smax(SD a, SD b) { return (a.v > b.v || a.v != a.v) ? a : b; }
__device__ __forceinline__ SD smin(SD a, SD b) { return (a.v < b.v || a.v != a.v) ? a : b; }
__device__ __forceinline__ SD sabs(SD a) { return SD(fabs(a.v)); }
#if SAS_STRICT
__device__ __forceinline__ SD ssqrt(SD a) { return SD(__dsqrt_rn(a.v)); }
__device__ __forceinline__ SD sexp(SD a) { return SD(glibc_exp(a.v)); }
__device__ __forceinline__ SD spow(SD a, SD b) { return SD(glibc_pow(a.v, b.v)); }
#else
__device__ __forceinline__ SD ssqrt(SD a) { return SD(sqrt(a.v)); }
__device__ __forceinline__ SD sexp(SD a) { return SD(exp(a.v)); }
__device__ __forceinline__ SD spow(SD a, SD b) { return SD(pow(a.v, b.v)); }
#endif
__device__ __forceinline__ SD sq(SD a) { return a * a; }              // x**2
__device__ __forceinline__ SD cube(SD a) { return (a * a) * a; }      // x**3
__device__ __forceinline__ SD wid(float x) { return SD((double)x); }  // REAL -> kind_phys
__device__ __forceinline__ float nar(SD x) { return __double2float_rn(x.v); }
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
// A default-REAL literal in a kind_phys expression.
#define SL(x) SD((double)(x##f))

// ---- module_gfs_physcons: single-precision literals widened -----------------
// (each real(kind_phys) parameter is assigned a default-REAL literal, so its
// value is the float32 word; derived parameters are binary64 operations on
// those words, which gfortran folds correctly rounded -- the same result the
// pinned intrinsics give at run time)
#define C_G     SL(9.80665e+0)
#define C_RD    SL(2.8705e+2)
#define C_RV    SL(4.6150e+2)
#define C_CP    SL(1.0046e+3)
#define C_CVAP  SL(1.8460e+3)
#define C_CLIQ  SL(4.1855e+3)
#define C_CSOL  SL(2.1060e+3)
#define C_HVAP  SL(2.5000e+6)
#define C_HFUS  SL(3.3358e+5)
#define C_PSAT  SL(6.1078e+2)
#define C_T0C   SL(2.7315e+2)
#define C_TTP   SL(2.7316e+2)
__device__ __forceinline__ SD c_rocp()  { return C_RD / C_CP; }
__device__ __forceinline__ SD c_cpor()  { return C_CP / C_RD; }
__device__ __forceinline__ SD c_fvirt() { return C_RV / C_RD - SD(1.0); }
__device__ __forceinline__ SD c_eps()   { return C_RD / C_RV; }
__device__ __forceinline__ SD c_epsm1() { return C_RD / C_RV - SD(1.0); }
__device__ __forceinline__ SD c_dldt()  { return C_CVAP - C_CLIQ; }
__device__ __forceinline__ SD c_xpona() { return -c_dldt() / C_RV; }
__device__ __forceinline__ SD c_xponb() { return -c_dldt() / C_RV + C_HVAP / (C_RV * C_TTP); }

// ---- funcphys tables ---------------------------------------------------------
#define NXPVS 7501
#define NXTDP 5001
#define NXTHE 241
#define NYTHE 151
#define NXMA 151
#define NYMA 121
#define NXTLCL 151
#define NYTLCL 61
// Layout of the table buffer (binary64), the same order tables.bin holds:
#define TB_PVS   0                                  // c1 c2 tb[7501]
#define TB_TDP   (TB_PVS + 2 + NXPVS)               // c1 c2 tb[5001]
#define TB_THE   (TB_TDP + 2 + NXTDP)               // c1x c2x c1y c2y tb[241*151]
#define TB_MA    (TB_THE + 4 + NXTHE * NYTHE)       // c1x c2x c1y c2y tbt[151*121] tbq[...]
#define TB_TLCL  (TB_MA + 4 + 2 * NXMA * NYMA)      // c1x c2x c1y c2y tb[151*61]
#define TB_TOTAL (TB_TLCL + 4 + NXTLCL * NYTLCL)

// fpvsx (:814): exact saturation vapour pressure, liquid above the triple
// point, ice 20 K below it, linear in between.
__device__ SD fpvsx(SD t) {
    const SD tliq = C_TTP, tice = C_TTP - SD(20.0);
    const SD dldtl = C_CVAP - C_CLIQ, heatl = C_HVAP;
    const SD xponal = -dldtl / C_RV, xponbl = -dldtl / C_RV + heatl / (C_RV * C_TTP);
    const SD dldti = C_CVAP - C_CSOL, heati = C_HVAP + C_HFUS;
    const SD xponai = -dldti / C_RV, xponbi = -dldti / C_RV + heati / (C_RV * C_TTP);
    const SD tr = C_TTP / t;
    if (t >= tliq) return C_PSAT * spow(tr, xponal) * sexp(xponbl * (SD(1.0) - tr));
    if (t < tice) return C_PSAT * spow(tr, xponai) * sexp(xponbi * (SD(1.0) - tr));
    const SD w = (t - tice) / (tliq - tice);
    const SD pvl = C_PSAT * spow(tr, xponal) * sexp(xponbl * (SD(1.0) - tr));
    const SD pvi = C_PSAT * spow(tr, xponai) * sexp(xponbi * (SD(1.0) - tr));
    return w * pvl + (SD(1.0) - w) * pvi;
}

// ftdpxg (:1588): dewpoint by Newton iteration from a guess.
__device__ SD ftdpxg(SD tg, SD pv) {
    const SD terrm = SL(1.e-6);
    const SD tliq = C_TTP, tice = C_TTP - SD(20.0);
    const SD dldtl = C_CVAP - C_CLIQ, heatl = C_HVAP;
    const SD xponal = -dldtl / C_RV, xponbl = -dldtl / C_RV + heatl / (C_RV * C_TTP);
    const SD dldti = C_CVAP - C_CSOL, heati = C_HVAP + C_HFUS;
    const SD xponai = -dldti / C_RV, xponbi = -dldti / C_RV + heati / (C_RV * C_TTP);
    SD t = tg;
    for (int i = 1; i <= 100; ++i) {
        const SD tr = C_TTP / t;
        SD pvt, dpvt;
        if (t >= tliq) {
            pvt = C_PSAT * spow(tr, xponal) * sexp(xponbl * (SD(1.0) - tr));
            const SD el = heatl + dldtl * (t - C_TTP);
            dpvt = el * pvt / (C_RV * sq(t));
        } else if (t < tice) {
            pvt = C_PSAT * spow(tr, xponai) * sexp(xponbi * (SD(1.0) - tr));
            const SD el = heati + dldti * (t - C_TTP);
            dpvt = el * pvt / (C_RV * sq(t));
        } else {
            const SD w = (t - tice) / (tliq - tice);
            const SD pvtl = C_PSAT * spow(tr, xponal) * sexp(xponbl * (SD(1.0) - tr));
            const SD pvti = C_PSAT * spow(tr, xponai) * sexp(xponbi * (SD(1.0) - tr));
            pvt = w * pvtl + (SD(1.0) - w) * pvti;
            const SD ell = heatl + dldtl * (t - C_TTP);
            const SD eli = heati + dldti * (t - C_TTP);
            dpvt = (w * ell * pvtl + (SD(1.0) - w) * eli * pvti) / (C_RV * sq(t));
        }
        const SD terr = (pvt - pv) / dpvt;
        t = t - terr;
        if (sabs(terr) <= terrm) break;
    }
    return t;
}

// fthex (:1860): equivalent potential temperature at the LCL.
__device__ SD fthex(SD t, SD pk) {
    const SD psatb = C_PSAT * SL(1.e-5);
    const SD p = spow(pk, c_cpor());
    const SD tr = C_TTP / t;
    const SD pv = psatb * spow(tr, c_xpona()) * sexp(c_xponb() * (SD(1.0) - tr));
    const SD pd = p - pv;
    if (pd > pv) {
        const SD el = C_HVAP + c_dldt() * (t - C_TTP);
        const SD expo = el * c_eps() * pv / (C_CP * t * pd);
        return t * spow(pd, -c_rocp()) * sexp(expo);
    }
    return SD(0.0);
}

// stmaxg (:2165): parcel temperature and humidity on a moist adiabat.
__device__ void stmaxg(SD tg, SD the, SD pk, SD& tma, SD& qma) {
    const SD terrm = SL(1.e-4);
    const SD psatb = C_PSAT * SL(1.e-5);
    SD t = tg;
    const SD p = spow(pk, c_cpor());
    for (int i = 1; i <= 100; ++i) {
        const SD tr = C_TTP / t;
        const SD pv = psatb * spow(tr, c_xpona()) * sexp(c_xponb() * (SD(1.0) - tr));
        const SD pd = p - pv;
        const SD el = C_HVAP + c_dldt() * (t - C_TTP);
        const SD expo = el * c_eps() * pv / (C_CP * t * pd);
        const SD thet = t * spow(pd, -c_rocp()) * sexp(expo);
        const SD dthet = thet / t * (SD(1.0) + expo * (c_dldt() * t / el
                                                     + el * p / (C_RV * t * pd)));
        const SD terr = (thet - the) / dthet;
        t = t - terr;
        if (sabs(terr) <= terrm) break;
    }
    tma = t;
    const SD tr = C_TTP / t;
    const SD pv = psatb * spow(tr, c_xpona()) * sexp(c_xponb() * (SD(1.0) - tr));
    const SD pd = p - pv;
    qma = c_eps() * pv / (pd + c_eps() * pv);
}

// ftlclx (:2862): LCL temperature by Newton iteration, floored at 180 K.
__device__ SD ftlclx(SD t, SD tdpd) {
    const SD terrm = SL(1.e-4), tlmin = SD(180.0), tlminx = tlmin - SD(5.0);
    SD tr = C_TTP / (t - tdpd);
    const SD pvdew = C_PSAT * spow(tr, c_xpona()) * sexp(c_xponb() * (SD(1.0) - tr));
    SD tlcl = t - tdpd;
    const SD inv_rocp = SD(1.0) / c_rocp();
    for (int i = 1; i <= 100; ++i) {
        tr = C_TTP / tlcl;
        const SD ta = t / tlcl;
        const SD pvlcl = C_PSAT * spow(tr, c_xpona()) * sexp(c_xponb() * (SD(1.0) - tr))
                         * spow(ta, inv_rocp);
        const SD el = C_HVAP + c_dldt() * (tlcl - C_TTP);
        const SD dpvlcl = (el / (C_RV * sq(t)) + SD(1.0) / (c_rocp() * tlcl)) * pvlcl;
        const SD terr = (pvlcl - pvdew) / dpvlcl;
        tlcl = tlcl - terr;
        if (sabs(terr) <= terrm || tlcl < tlminx) break;
    }
    return smax(tlcl, tlmin);
}

// The table bounds whose second axis is p**kappa.  ymin/ymax of gthe and gtma
// are constant expressions (0.04_krealfp**con_rocp ...) that gfortran folds
// at compile time; the launcher hands in those words (gpuwm.core.sas
// .TABLE_BOUNDS, measured against the oracle's c1y/c2y) rather than
// evaluating pow here.

// GFUNCPHYS, the five tables SAS reads.  One launch; thread roles:
//   [0, NXPVS)                    gpvs entries (independent)
//   NXPVS                         gtdp (a Newton chain: each entry seeds the next)
//   (+1) .. (+NXTHE*NYTHE)        gthe entries (independent)
//   (+NYMA)                       gtma rows (a chain along the theta axis)
//   (+NXTLCL*NYTLCL)              gtlcl entries (independent)
// Thread 0 also writes every c1/c2 coefficient.
extern "C" __global__ void sas_tables(double* tb, double the_ymin, double the_ymax,
                                      double ma_ymin, double ma_ymax) {
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long r = id;
    if (r < NXPVS) {                                   // gpvs (:671)
        const SD xmin = SD(180.0), xmax = SD(330.0);
        const SD xinc = (xmax - xmin) / SD((double)(NXPVS - 1));
        const SD c2 = SD(1.0) / xinc;
        const SD c1 = SD(1.0) - xmin * c2;
        if (r == 0) { tb[TB_PVS] = c1.v; tb[TB_PVS + 1] = c2.v; }
        const int jx = (int)r + 1;
        const SD x = xmin + SD((double)(jx - 1)) * xinc;
        tb[TB_PVS + 2 + (jx - 1)] = fpvsx(x).v;
        return;
    }
    r -= NXPVS;
    if (r == 0) {                                      // gtdp (:1395)
        const SD xmin = SL(0.5), xmax = SL(10000.5);
        const SD xinc = (xmax - xmin) / SD((double)(NXTDP - 1));
        tb[TB_TDP] = (SD(1.0) - xmin / xinc).v;
        tb[TB_TDP + 1] = (SD(1.0) / xinc).v;
        SD t = SL(208.0);
        for (int jx = 1; jx <= NXTDP; ++jx) {
            const SD x = xmin + SD((double)(jx - 1)) * xinc;
            t = ftdpxg(t, x);
            tb[TB_TDP + 2 + (jx - 1)] = t.v;
        }
        return;
    }
    r -= 1;
    if (r < (long long)NXTHE * NYTHE) {                // gthe (:1678)
        const SD xmin = C_TTP - SD(90.0), xmax = C_TTP + SD(30.0);
        const SD ymin = SD(the_ymin), ymax = SD(the_ymax);
        const SD xinc = (xmax - xmin) / SD((double)(NXTHE - 1));
        const SD yinc = (ymax - ymin) / SD((double)(NYTHE - 1));
        if (r == 0) {
            tb[TB_THE] = (SD(1.0) - xmin / xinc).v;
            tb[TB_THE + 1] = (SD(1.0) / xinc).v;
            tb[TB_THE + 2] = (SD(1.0) - ymin / yinc).v;
            tb[TB_THE + 3] = (SD(1.0) / yinc).v;
        }
        const int jx = (int)(r % NXTHE) + 1, jy = (int)(r / NXTHE) + 1;
        const SD y = ymin + SD((double)(jy - 1)) * yinc;
        const SD x = xmin + SD((double)(jx - 1)) * xinc;
        tb[TB_THE + 4 + (jy - 1) * NXTHE + (jx - 1)] = fthex(x, y).v;
        return;
    }
    r -= (long long)NXTHE * NYTHE;
    if (r < NYMA) {                                    // gtma (:1918)
        const SD xmin = SD(200.0), xmax = SD(500.0);
        const SD ymin = SD(ma_ymin), ymax = SD(ma_ymax);
        const SD xinc = (xmax - xmin) / SD((double)(NXMA - 1));
        const SD yinc = (ymax - ymin) / SD((double)(NYMA - 1));
        if (r == 0) {
            tb[TB_MA] = (SD(1.0) - xmin / xinc).v;
            tb[TB_MA + 1] = (SD(1.0) / xinc).v;
            tb[TB_MA + 2] = (SD(1.0) - ymin / yinc).v;
            tb[TB_MA + 3] = (SD(1.0) / yinc).v;
        }
        const int jy = (int)r + 1;
        const SD y = ymin + SD((double)(jy - 1)) * yinc;
        SD tg = xmin * y;
        for (int jx = 1; jx <= NXMA; ++jx) {
            const SD x = xmin + SD((double)(jx - 1)) * xinc;
            SD t, q;
            stmaxg(tg, x, y, t, q);
            tb[TB_MA + 4 + (jy - 1) * NXMA + (jx - 1)] = t.v;
            tb[TB_MA + 4 + NXMA * NYMA + (jy - 1) * NXMA + (jx - 1)] = q.v;
            tg = t;
        }
        return;
    }
    r -= NYMA;
    if (r < (long long)NXTLCL * NYTLCL) {              // gtlcl (:2654)
        const SD xmin = SD(180.0), xmax = SD(330.0), ymin = SD(0.0), ymax = SD(60.0);
        const SD xinc = (xmax - xmin) / SD((double)(NXTLCL - 1));
        const SD yinc = (ymax - ymin) / SD((double)(NYTLCL - 1));
        if (r == 0) {
            tb[TB_TLCL] = (SD(1.0) - xmin / xinc).v;
            tb[TB_TLCL + 1] = (SD(1.0) / xinc).v;
            tb[TB_TLCL + 2] = (SD(1.0) - ymin / yinc).v;
            tb[TB_TLCL + 3] = (SD(1.0) / yinc).v;
        }
        const int jx = (int)(r % NXTLCL) + 1, jy = (int)(r / NXTLCL) + 1;
        const SD y = ymin + SD((double)(jy - 1)) * yinc;
        const SD x = xmin + SD((double)(jx - 1)) * xinc;
        tb[TB_TLCL + 4 + (jy - 1) * NXTLCL + (jx - 1)] = ftlclx(x, y).v;
    }
}

// ---- table lookups -----------------------------------------------------------
__device__ __forceinline__ SD fpvs(const double* tb, SD t) {          // :718
    const SD c1 = SD(tb[TB_PVS]), c2 = SD(tb[TB_PVS + 1]);
    const double* t1 = tb + TB_PVS + 2 - 1;                            // 1-based
    const SD xj = smin(smax(c1 + c2 * t, SD(1.0)), SD((double)NXPVS));
    const int jx = (int)smin(xj, SD((double)NXPVS - 1.0)).v;
    return SD(t1[jx]) + (xj - SD((double)jx)) * (SD(t1[jx + 1]) - SD(t1[jx]));
}
__device__ __forceinline__ SD ftdp(const double* tb, SD pv) {         // :1443
    const SD c1 = SD(tb[TB_TDP]), c2 = SD(tb[TB_TDP + 1]);
    const double* t1 = tb + TB_TDP + 2 - 1;
    const SD xj = smin(smax(c1 + c2 * pv, SD(1.0)), SD((double)NXTDP));
    const int jx = (int)smin(xj, SD((double)NXTDP - 1.0)).v;
    return SD(t1[jx]) + (xj - SD((double)jx)) * (SD(t1[jx + 1]) - SD(t1[jx]));
}
__device__ __forceinline__ SD fthe(const double* tb, SD t, SD pk) {   // :1734
    const SD c1x = SD(tb[TB_THE]), c2x = SD(tb[TB_THE + 1]);
    const SD c1y = SD(tb[TB_THE + 2]), c2y = SD(tb[TB_THE + 3]);
    const double* a = tb + TB_THE + 4;
    const SD xj = smin(c1x + c2x * t, SD((double)NXTHE));
    const SD yj = smin(c1y + c2y * pk, SD((double)NYTHE));
    if (xj >= SD(1.0) && yj >= SD(1.0)) {
        const int jx = (int)smin(xj, SD((double)NXTHE - 1.0)).v;
        const int jy = (int)smin(yj, SD((double)NYTHE - 1.0)).v;
#define THE(i, j) SD(a[((j) - 1) * NXTHE + ((i) - 1)])
        const SD ftx1 = THE(jx, jy) + (xj - SD((double)jx)) * (THE(jx + 1, jy) - THE(jx, jy));
        const SD ftx2 = THE(jx, jy + 1) + (xj - SD((double)jx)) * (THE(jx + 1, jy + 1) - THE(jx, jy + 1));
#undef THE
        return ftx1 + (yj - SD((double)jy)) * (ftx2 - ftx1);
    }
    return SD(0.0);
}
__device__ __forceinline__ void stma(const double* tb, SD the, SD pk, SD& tma, SD& qma) {  // :1978
    const SD c1x = SD(tb[TB_MA]), c2x = SD(tb[TB_MA + 1]);
    const SD c1y = SD(tb[TB_MA + 2]), c2y = SD(tb[TB_MA + 3]);
    const double* tt = tb + TB_MA + 4;
    const double* tq = tt + NXMA * NYMA;
    const SD xj = smin(smax(c1x + c2x * the, SD(1.0)), SD((double)NXMA));
    const SD yj = smin(smax(c1y + c2y * pk, SD(1.0)), SD((double)NYMA));
    const int jx = (int)smin(xj, SD((double)NXMA - 1.0)).v;
    const int jy = (int)smin(yj, SD((double)NYMA - 1.0)).v;
#define TT(i, j) SD(tt[((j) - 1) * NXMA + ((i) - 1)])
#define TQ(i, j) SD(tq[((j) - 1) * NXMA + ((i) - 1)])
    const SD dx = xj - SD((double)jx), dy = yj - SD((double)jy);
    const SD ftx1 = TT(jx, jy) + dx * (TT(jx + 1, jy) - TT(jx, jy));
    const SD ftx2 = TT(jx, jy + 1) + dx * (TT(jx + 1, jy + 1) - TT(jx, jy + 1));
    tma = ftx1 + dy * (ftx2 - ftx1);
    const SD qx1 = TQ(jx, jy) + dx * (TQ(jx + 1, jy) - TQ(jx, jy));
    const SD qx2 = TQ(jx, jy + 1) + dx * (TQ(jx + 1, jy + 1) - TQ(jx, jy + 1));
    qma = qx1 + dy * (qx2 - qx1);
#undef TT
#undef TQ
}
__device__ __forceinline__ SD ftlcl(const double* tb, SD t, SD tdpd) {  // :2707
    const SD c1x = SD(tb[TB_TLCL]), c2x = SD(tb[TB_TLCL + 1]);
    const SD c1y = SD(tb[TB_TLCL + 2]), c2y = SD(tb[TB_TLCL + 3]);
    const double* a = tb + TB_TLCL + 4;
    const SD xj = smin(smax(c1x + c2x * t, SD(1.0)), SD((double)NXTLCL));
    const SD yj = smin(smax(c1y + c2y * tdpd, SD(1.0)), SD((double)NYTLCL));
    const int jx = (int)smin(xj, SD((double)NXTLCL - 1.0)).v;
    const int jy = (int)smin(yj, SD((double)NYTLCL - 1.0)).v;
#define TL(i, j) SD(a[((j) - 1) * NXTLCL + ((i) - 1)])
    const SD ftx1 = TL(jx, jy) + (xj - SD((double)jx)) * (TL(jx + 1, jy) - TL(jx, jy));
    const SD ftx2 = TL(jx, jy + 1) + (xj - SD((double)jx)) * (TL(jx + 1, jy + 1) - TL(jx, jy + 1));
#undef TL
    return ftx1 + (yj - SD((double)jy)) * (ftx2 - ftx1);
}

// ---- column views ------------------------------------------------------------
struct V {                      // binary64 column array, 1-based
    double* p;
    int s;
    __device__ __forceinline__ SD& operator()(int k) const {
        return *reinterpret_cast<SD*>(p + (long long)(k - 1) * s);
    }
};
struct Ws {                     // this column's workspace
    double* base;               // ws + col
    int kw;                     // slot length (levels)
    int stride;                 // columns sharing the workspace
    __device__ __forceinline__ V slot(int s) const {
        return V{base + (long long)s * kw * stride, stride};
    }
};
// Slots.  SASCNVN's automatic arrays first, then CU_SAS's locals, then the
// shallow branch's.
enum {
    S_PFLD, S_TO, S_QO, S_UO, S_VO, S_QESO, S_DELLAL, S_DBYO, S_ZO, S_XLAMUE,
    S_FENT1, S_FENT2, S_FRH, S_HEO, S_HESO, S_QRCD, S_DELLAH, S_DELLAQ,
    S_DELLAU, S_DELLAV, S_HCKO, S_UCKO, S_VCKO, S_QCKO, S_ETA, S_ETAD, S_ZI,
    S_QRCDO, S_PWO, S_PWDO, S_HCDO, S_UCDO, S_VCDO, S_QCDO, S_PO,
    S_DEL, S_PRSL, S_PHIL, S_DOT, S_Q1, S_T1, S_U1, S_V1, S_QL1, S_QL2,
    S_PRSLK, S_PRSI, S_DZI, S_DZL,
    S_Q2, S_T2, S_AL, S_AD, S_AU,
    S_COUNT
};

// ---- SASCNVN (:2599) for one column (im = ix = 1, ncloud = 1, rcs = 1) -----
struct DeepOut { SD cldwrk, rn; int kbot, ktop, kcnv; };

template<bool PREPARED = false>
__device__ DeepOut sascnvn(const double* tb, const Ws& w, int km, SD delt,
                           V del, V prsl, SD ps, V phil, V ql1, V ql2,
                           V q1, V t1, V u1, V v1, SD rcs, SD slimsk, V dot,
                           SD pgcon, SD sas_mass_flux)
{
    const int ncloud = 1;
    const SD g = C_G, cp = C_CP, hvap = C_HVAP, rv = C_RV;
    const SD fv = c_fvirt(), t0c = C_T0C, cvap = C_CVAP, cliq = C_CLIQ;
    const SD eps = c_eps(), epsm1 = c_epsm1();
    const SD elocp = hvap / cp, el2orc = hvap * hvap / (rv * cp);
    const SD c0 = SL(.002), c1 = SL(.002), delta = fv;
    const SD fact1 = (cvap - cliq) / rv, fact2 = hvap / rv - fact1 * t0c;
    const SD cthk = SD(150.0), cincrmax = SD(180.0), cincrmin = SD(120.0), dthk = SD(25.0);
    const SD tf = SL(233.16), tcr = SL(263.16), tcrf = SD(1.0) / (tcr - tf);

    V pfld = w.slot(S_PFLD), to = w.slot(S_TO), qo = w.slot(S_QO), uo = w.slot(S_UO),
      vo = w.slot(S_VO), qeso = w.slot(S_QESO), dellal = w.slot(S_DELLAL),
      dbyo = w.slot(S_DBYO), zo = w.slot(S_ZO), xlamue = w.slot(S_XLAMUE),
      fent1 = w.slot(S_FENT1), fent2 = w.slot(S_FENT2), frh = w.slot(S_FRH),
      heo = w.slot(S_HEO), heso = w.slot(S_HESO), qrcd = w.slot(S_QRCD),
      dellah = w.slot(S_DELLAH), dellaq = w.slot(S_DELLAQ),
      dellau = w.slot(S_DELLAU), dellav = w.slot(S_DELLAV),
      hcko = w.slot(S_HCKO), ucko = w.slot(S_UCKO), vcko = w.slot(S_VCKO),
      qcko = w.slot(S_QCKO), eta = w.slot(S_ETA), etad = w.slot(S_ETAD),
      zi = w.slot(S_ZI), qrcdo = w.slot(S_QRCDO), pwo = w.slot(S_PWO),
      pwdo = w.slot(S_PWDO), hcdo = w.slot(S_HCDO), ucdo = w.slot(S_UCDO),
      vcdo = w.slot(S_VCDO), qcdo = w.slot(S_QCDO), po = w.slot(S_PO);

    // pcrit / acritt (:2715-2718), data statements of default-REAL literals
    const float pcrit_f[15] = {850.f, 800.f, 750.f, 700.f, 650.f, 600.f, 550.f, 500.f,
                               450.f, 400.f, 350.f, 300.f, 250.f, 200.f, 150.f};
    const float acritt_f[15] = {.0633f, .0445f, .0553f, .0664f, .075f, .1082f, .1521f,
                                .2216f, .3151f, .3677f, .41f, .5255f, .7663f, 1.1686f,
                                1.6851f};
#define PCRIT(k) wid(pcrit_f[(k) - 1])

    const int km1 = km - 1;
    DeepOut out;
    out.kcnv = 0;
    bool cnvflg = true;
    out.rn = SD(0.0);
    out.kbot = km + 1;
    out.ktop = 0;
    int kbcon = km, ktcon = 1;
    SD dtconv = SD(3600.0);
    out.cldwrk = SD(0.0);
    SD pdot = SD(0.0), pbcdif = SD(0.0);
    int lmin = 1, jmin = 1;
    SD qlko_ktcon = SD(0.0), edt = SD(0.0), edto = SD(0.0), edtx = SD(0.0);
    SD acrt = SD(0.0), acrtfct = SD(1.0), aa1 = SD(0.0), aa2 = SD(0.0), xaa0 = SD(0.0);
    SD pwavo = SD(0.0), pwevo = SD(0.0), xpwav = SD(0.0), xpwev = SD(0.0), vshear = SD(0.0);
    SD acrit[16];
    for (int k = 1; k <= 15; ++k) acrit[k] = wid(acritt_f[k - 1]) * (SD(975.0) - PCRIT(k));
    const SD dt2 = delt;
    SD val = SD(1200.0);
    const SD dtmin = smax(dt2, val);
    val = SD(3600.0);
    const SD dtmax = smax(dt2, val);
    const SD mbdt = SD(10.0), edtmaxl = SL(.3), edtmaxs = SL(.3), clam = SL(.1), aafac = SL(.1);
    // EM_CORE (:2804-2810): betal = betas = .05, evfact = evfactl = 0.5
    const SD betal = SL(.05), betas = SL(.05), evfact = SL(0.5), evfactl = SL(0.5);
    const SD cxlamu = SL(1.0e-4), xlamde = SL(1.0e-4), xlamdd = SL(1.0e-4);
    const SD w1l = -SL(8.e-3), w2l = -SL(4.e-2), w3l = -SL(5.e-3), w4l = -SL(5.e-4);
    const SD w1s = -SL(2.e-4), w2s = -SL(2.e-3), w3s = -SL(1.e-3), w4s = -SL(2.e-5);

    int kbmax = km, kbm = km, kmax = km;
    int kb = 1;
    if constexpr (PREPARED) {
        const V meta = w.slot(S_DZI);
        kmax = (int)meta(1).v; kbmax = (int)meta(2).v;
        kbm = (int)meta(3).v; kb = (int)w.slot(S_DZL)(1).v;
    } else {
    const SD tx1 = SD(1.0) / ps;
    for (int k = 1; k <= km; ++k) {
        if (prsl(k) * tx1 > SL(0.04)) kmax = imin(km, k + 1);
        if (prsl(k) * tx1 > SL(0.45)) kbmax = k + 1;
        if (prsl(k) * tx1 > SL(0.70)) kbm = k + 1;
    }
    kbmax = imin(kbmax, kmax);
    kbm = imin(kbm, kmax);
    for (int k = 1; k <= km; ++k) zo(k) = phil(k) / g;
    for (int k = 1; k <= km1; ++k) {
        zi(k) = SD(0.5) * (zo(k) + zo(k + 1));
        xlamue(k) = clam / zi(k);
    }
    for (int k = 1; k <= imin(km, kmax); ++k) {
        pfld(k) = prsl(k) * SD(10.0);
        to(k) = t1(k); qo(k) = q1(k);
        uo(k) = u1(k) * rcs; vo(k) = v1(k) * rcs;
        qeso(k) = SL(0.01) * fpvs(tb, to(k));
        qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
        qeso(k) = smax(qeso(k), SL(1.e-8));
        qo(k) = smax(qo(k), SL(1.e-10));
        const SD tem = phil(k) + cp * to(k);
        heo(k) = tem + hvap * qo(k);
        heso(k) = tem + hvap * qeso(k);
    }
    SD hmax = heo(1);
    kb = 1;
    for (int k = 2; k <= km; ++k) {
        if (k <= kbm && heo(k) > hmax) { kb = k; hmax = heo(k); }
    }
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            const SD dz = SD(.5) * (zo(k + 1) - zo(k));
            const SD dp = SD(.5) * (pfld(k + 1) - pfld(k));
            const SD es = SL(0.01) * fpvs(tb, to(k + 1));
            const SD pprime = pfld(k + 1) + epsm1 * es;
            const SD qs = eps * es / pprime;
            const SD dqsdp = -(qs / pprime);
            const SD desdt = es * (fact1 / to(k + 1) + fact2 / sq(to(k + 1)));
            const SD dqsdt = qs * pfld(k + 1) * desdt / (es * pprime);
            const SD gamma = el2orc * qeso(k + 1) / sq(to(k + 1));
            const SD dt = (g * dz + hvap * dqsdp * dp) / (cp * (SD(1.0) + gamma));
            const SD dq = dqsdt * dt + dqsdp * dp;
            to(k) = to(k + 1) + dt;
            qo(k) = qo(k + 1) + dq;
            po(k) = SD(.5) * (pfld(k) + pfld(k + 1));
            qeso(k) = SL(0.01) * fpvs(tb, to(k));
            qeso(k) = eps * qeso(k) / (po(k) + epsm1 * qeso(k));
            qeso(k) = smax(qeso(k), SL(1.e-8));
            qo(k) = smax(qo(k), SL(1.e-10));
            frh(k) = SD(1.0) - smin(qo(k) / qeso(k), SD(1.0));
            heo(k) = SD(.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qo(k);
            heso(k) = SD(.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qeso(k);
            uo(k) = SD(.5) * (uo(k) + uo(k + 1));
            vo(k) = SD(.5) * (vo(k) + vo(k + 1));
        }
    }
    }
    // level of free convection
    {
        bool flg = true;
        kbcon = kmax;
        for (int k = 1; k <= km1; ++k) {
            if (flg && k <= kbmax && k > kb && heo(kb) > heso(k)) { kbcon = k; flg = false; }
        }
    }
    if (kbcon == kmax) cnvflg = false;
    if (!cnvflg) return out;
    // critical convective inhibition from cloud-base vertical velocity
    pdot = SD(10.0) * dot(kbcon);
    SD w1, w2, w3, w4;
    if (slimsk == SD(1.0)) { w1 = w1l; w2 = w2l; w3 = w3l; w4 = w4l; }
    else { w1 = w1s; w2 = w2s; w3 = w3s; w4 = w4s; }
    {
        SD tem;
        if (pdot <= w4) tem = (pdot - w4) / (w3 - w4);
        else if (pdot >= -w4) tem = -(pdot + w4) / (w4 - w3);
        else tem = SD(0.0);
        tem = smax(tem, SD(-1.0));
        tem = smin(tem, SD(1.0));
        tem = SD(1.0) - tem;
        const SD tem1 = SD(.5) * (cincrmax - cincrmin);
        const SD cincr = cincrmax - tem * tem1;
        pbcdif = pfld(kb) - pfld(kbcon);
        if (pbcdif > cincr) cnvflg = false;
    }
    if (!cnvflg) return out;
    // Only surviving deep columns need these scratch initial values. Neither
    // rejection check above reads them. Keep every native binary64 operation.
    for (int k = 1; k <= kmax; ++k) {
            eta(k) = SD(1.0); fent1(k) = SD(1.0); fent2(k) = SD(1.0);
            hcko(k) = SD(0.0); qcko(k) = SD(0.0); ucko(k) = SD(0.0); vcko(k) = SD(0.0);
            etad(k) = SD(1.0);
            hcdo(k) = SD(0.0); qcdo(k) = SD(0.0); ucdo(k) = SD(0.0); vcdo(k) = SD(0.0);
            qrcd(k) = SD(0.0); qrcdo(k) = SD(0.0); dbyo(k) = SD(0.0);
            pwo(k) = SD(0.0); pwdo(k) = SD(0.0); dellal(k) = SD(0.0);
    }
    for (int k = 2; k <= km1; ++k)
        if (k > kbcon && k < kmax) xlamue(k) = xlamue(kbcon);
    const SD xlamud = xlamue(kbcon);
    for (int k = imax(2, kbcon + 1); k <= imin(km1, kmax - 1); ++k) {
        const SD tem = qeso(k) / qeso(kbcon);
        fent1(k) = sq(tem);
        fent2(k) = cube(tem);
    }
    for (int k = imax(2, kbcon); k <= imin(km1, kmax - 1); ++k) {
        const SD tem = cxlamu * frh(k) * fent2(k);
        xlamue(k) = xlamue(k) * fent1(k) + tem;
    }
    for (int k = imin(km1, kbcon - 1); k >= imax(1, kb); --k) {
        const SD dz = zi(k + 1) - zi(k);
        const SD ptem = SD(0.5) * (xlamue(k) + xlamue(k + 1)) - xlamud;
        eta(k) = eta(k + 1) / (SD(1.0) + ptem * dz);
    }
    for (int k = imax(2, kbcon + 1); k <= imin(km1, kmax - 1); ++k) {
        const SD dz = zi(k) - zi(k - 1);
        const SD ptem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) - xlamud;
        eta(k) = eta(k - 1) * (SD(1.0) + ptem * dz);
    }
    hcko(kb) = heo(kb); ucko(kb) = uo(kb); vcko(kb) = vo(kb);
    pwavo = SD(0.0);
    for (int k = imax(2, kb + 1); k <= imin(km1, kmax - 1); ++k) {
        const SD dz = zi(k) - zi(k - 1);
        const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) * dz;
        const SD tem1 = SD(0.5) * xlamud * dz;
        const SD factor = SD(1.0) + tem - tem1;
        const SD ptem = SD(0.5) * tem + pgcon;
        const SD ptem1 = SD(0.5) * tem - pgcon;
        hcko(k) = ((SD(1.0) - tem1) * hcko(k - 1) + tem * SD(0.5) * (heo(k) + heo(k - 1))) / factor;
        ucko(k) = ((SD(1.0) - tem1) * ucko(k - 1) + ptem * uo(k) + ptem1 * uo(k - 1)) / factor;
        vcko(k) = ((SD(1.0) - tem1) * vcko(k - 1) + ptem * vo(k) + ptem1 * vo(k - 1)) / factor;
        dbyo(k) = hcko(k) - heso(k);
    }
    int kbcon1 = kmax;
    {
        bool flg = cnvflg;
        for (int k = 2; k <= km1; ++k) {
            if (flg && k < kmax && k >= kbcon && dbyo(k) > SD(0.0)) { kbcon1 = k; flg = false; }
        }
    }
    if (kbcon1 == kmax) cnvflg = false;
    if (cnvflg && pfld(kbcon) - pfld(kbcon1) > dthk) cnvflg = false;
    if (!cnvflg) return out;
    {
        bool flg = cnvflg;
        ktcon = 1;
        for (int k = 2; k <= km1; ++k) {
            if (flg && k < kmax && k > kbcon1 && dbyo(k) < SD(0.0)) { ktcon = k; flg = false; }
        }
    }
    if (pfld(kbcon) - pfld(ktcon) < cthk) cnvflg = false;
    if (!cnvflg) return out;
    // downdraft originating level
    SD hmin = heo(kbcon1);
    lmin = kbmax; jmin = kbmax;
    for (int k = 2; k <= km1; ++k) {
        if (k <= kbmax && k > kbcon1 && heo(k) < hmin) { lmin = k + 1; hmin = heo(k); }
    }
    jmin = imin(lmin, ktcon - 1);
    jmin = imax(jmin, kbcon1 + 1);
    if (jmin >= ktcon) cnvflg = false;
    if (!cnvflg) return out;
    SD xmbmax;
    {
        const int k = kbcon;
        const SD dp = SD(1000.0) * del(k);
        xmbmax = dp / (g * dt2);
        xmbmax = smin(sas_mass_flux, xmbmax);
    }
    // cloud moisture property and precipitation
    aa1 = SD(0.0);
    qcko(kb) = qo(kb);
    for (int k = imax(2, kb + 1); k <= imin(km1, ktcon - 1); ++k) {
        const SD dz = zi(k) - zi(k - 1);
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD qrch = qeso(k) + gamma * dbyo(k) / (hvap * (SD(1.0) + gamma));
        const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) * dz;
        const SD tem1 = SD(0.5) * xlamud * dz;
        const SD factor = SD(1.0) + tem - tem1;
        qcko(k) = ((SD(1.0) - tem1) * qcko(k - 1) + tem * SD(0.5) * (qo(k) + qo(k - 1))) / factor;
        const SD dq = eta(k) * (qcko(k) - qrch);
        if (k >= kbcon && dq > SD(0.0)) {
            const SD etah = SD(.5) * (eta(k) + eta(k - 1));
            SD qlk;
            if (ncloud > 0 && k > jmin) {
                const SD dp = SD(1000.0) * del(k);
                qlk = dq / (eta(k) + etah * (c0 + c1) * dz);
                dellal(k) = etah * c1 * dz * qlk * g / dp;
            } else {
                qlk = dq / (eta(k) + etah * c0 * dz);
            }
            aa1 = aa1 - dz * g * qlk;
            qcko(k) = qlk + qrch;
            pwo(k) = etah * c0 * dz * qlk;
            pwavo = pwavo + pwo(k);
        }
    }
    // cloud work function
    for (int k = imax(2, kbcon); k <= imin(km1, ktcon - 1); ++k) {
        const SD dz1 = zo(k + 1) - zo(k);
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD rfact = SD(1.0) + delta * cp * gamma * to(k) / hvap;
        aa1 = aa1 + dz1 * (g / (cp * to(k))) * dbyo(k) / (SD(1.0) + gamma) * rfact;
        aa1 = aa1 + dz1 * g * delta * smax(SD(0.0), qeso(k) - qo(k));
    }
    if (aa1 <= SD(0.0)) cnvflg = false;
    if (!cnvflg) return out;
    // overshooting: final cloud top where aafac*aa1 is used up
    aa2 = aafac * aa1;
    int ktcon1 = kmax - 1;
    {
        bool flg = cnvflg;
        for (int k = 2; k <= km1; ++k) {
            if (flg && k >= ktcon && k < kmax) {
                const SD dz1 = zo(k + 1) - zo(k);
                const SD gamma = el2orc * qeso(k) / sq(to(k));
                const SD rfact = SD(1.0) + delta * cp * gamma * to(k) / hvap;
                aa2 = aa2 + dz1 * (g / (cp * to(k))) * dbyo(k) / (SD(1.0) + gamma) * rfact;
                if (aa2 < SD(0.0)) { ktcon1 = k; flg = false; }
            }
        }
    }
    for (int k = imax(2, ktcon); k <= imin(km1, ktcon1 - 1); ++k) {
        const SD dz = zi(k) - zi(k - 1);
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD qrch = qeso(k) + gamma * dbyo(k) / (hvap * (SD(1.0) + gamma));
        const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) * dz;
        const SD tem1 = SD(0.5) * xlamud * dz;
        const SD factor = SD(1.0) + tem - tem1;
        qcko(k) = ((SD(1.0) - tem1) * qcko(k - 1) + tem * SD(0.5) * (qo(k) + qo(k - 1))) / factor;
        const SD dq = eta(k) * (qcko(k) - qrch);
        if (dq > SD(0.0)) {
            const SD etah = SD(.5) * (eta(k) + eta(k - 1));
            SD qlk;
            if (ncloud > 0) {
                const SD dp = SD(1000.0) * del(k);
                qlk = dq / (eta(k) + etah * (c0 + c1) * dz);
                dellal(k) = etah * c1 * dz * qlk * g / dp;
            } else {
                qlk = dq / (eta(k) + etah * c0 * dz);
            }
            qcko(k) = qlk + qrch;
            pwo(k) = etah * c0 * dz * qlk;
            pwavo = pwavo + pwo(k);
        }
    }
    { const int kk = ktcon; ktcon = ktcon1; ktcon1 = kk; }
    if (ncloud > 0) {
        const int k = ktcon - 1;
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD qrch = qeso(k) + gamma * dbyo(k) / (hvap * (SD(1.0) + gamma));
        const SD dq = qcko(k) - qrch;
        if (dq > SD(0.0)) { qlko_ktcon = dq; qcko(k) = qrch; }
    }
    // precipitation efficiency from wind shear
    vshear = SD(0.0);
    for (int k = imax(2, kb + 1); k <= imin(km, ktcon); ++k) {
        const SD shear = ssqrt(sq(uo(k) - uo(k - 1)) + sq(vo(k) - vo(k - 1)));
        vshear = vshear + shear;
    }
    vshear = SD(1.e3) * vshear / (zi(ktcon) - zi(kb));
    {
        const SD e1 = SL(1.591) - SL(.639) * vshear + SL(.0953) * sq(vshear)
                      - SL(.00496) * cube(vshear);
        edt = SD(1.0) - e1;
        edt = smin(edt, SL(.9));
        edt = smax(edt, SL(.0));
        edto = edt;
        edtx = edt;
    }
    // detrainment rate between 1 and kbcon
    SD xlamd;
    {
        SD sumx = SD(0.0);
        for (int k = 1; k <= km1; ++k)
            if (k >= 1 && k < kbcon) sumx = sumx + (zi(k + 1) - zi(k));
        SD beta = betas;
        if (slimsk == SD(1.0)) beta = betal;
        const SD dz = (sumx + zi(1)) / wid((float)kbcon);
        const SD tem = wid(__fdiv_rn(1.0f, (float)kbcon));
        xlamd = (SD(1.0) - spow(beta, tem)) / dz;
    }
    // downdraft mass flux
    for (int k = km1; k >= 1; --k) {
        if (k <= kmax - 1) {
            if (k < jmin && k >= kbcon) {
                const SD dz = zi(k + 1) - zi(k);
                const SD ptem = xlamdd - xlamde;
                etad(k) = etad(k + 1) * (SD(1.0) - ptem * dz);
            } else if (k < kbcon) {
                const SD dz = zi(k + 1) - zi(k);
                const SD ptem = xlamd + xlamdd - xlamde;
                etad(k) = etad(k + 1) * (SD(1.0) - ptem * dz);
            }
        }
    }
    // downdraft moisture properties
    {
        const int jmn = jmin;
        hcdo(jmn) = heo(jmn); qcdo(jmn) = qo(jmn); qrcdo(jmn) = qeso(jmn);
        ucdo(jmn) = uo(jmn); vcdo(jmn) = vo(jmn);
        pwevo = SD(0.0);
    }
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD dz = zi(k + 1) - zi(k);
        SD tem, tem1;
        if (k >= kbcon) { tem = xlamde * dz; tem1 = SD(0.5) * xlamdd * dz; }
        else { tem = xlamde * dz; tem1 = SD(0.5) * (xlamd + xlamdd) * dz; }
        const SD factor = SD(1.0) + tem - tem1;
        const SD ptem = SD(0.5) * tem - pgcon;
        const SD ptem1 = SD(0.5) * tem + pgcon;
        hcdo(k) = ((SD(1.0) - tem1) * hcdo(k + 1) + tem * SD(0.5) * (heo(k) + heo(k + 1))) / factor;
        ucdo(k) = ((SD(1.0) - tem1) * ucdo(k + 1) + ptem * uo(k + 1) + ptem1 * uo(k)) / factor;
        vcdo(k) = ((SD(1.0) - tem1) * vcdo(k + 1) + ptem * vo(k + 1) + ptem1 * vo(k)) / factor;
        dbyo(k) = hcdo(k) - heso(k);
    }
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        qrcdo(k) = qeso(k) + (SD(1.0) / hvap) * (gamma / (SD(1.0) + gamma)) * dbyo(k);
        const SD dz = zi(k + 1) - zi(k);
        SD tem, tem1;
        if (k >= kbcon) { tem = xlamde * dz; tem1 = SD(0.5) * xlamdd * dz; }
        else { tem = xlamde * dz; tem1 = SD(0.5) * (xlamd + xlamdd) * dz; }
        const SD factor = SD(1.0) + tem - tem1;
        qcdo(k) = ((SD(1.0) - tem1) * qcdo(k + 1) + tem * SD(0.5) * (qo(k) + qo(k + 1))) / factor;
        pwdo(k) = etad(k + 1) * (qcdo(k) - qrcdo(k));
        qcdo(k) = qrcdo(k);
        pwevo = pwevo + pwdo(k);
    }
    {
        SD edtmax = edtmaxl;
        if (slimsk == SD(0.0)) edtmax = edtmaxs;
        if (pwevo < SD(0.0)) {
            edto = -(edto * pwavo / pwevo);
            edto = smin(edto, edtmax);
        } else {
            edto = SD(0.0);
        }
    }
    // downdraft cloud work functions
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD dhh = hcdo(k), dt = to(k), dg = gamma, dh = heso(k);
        const SD dz = -(zo(k + 1) - zo(k));
        aa1 = aa1 + edto * dz * (g / (cp * dt)) * ((dhh - dh) / (SD(1.0) + dg))
                    * (SD(1.0) + delta * cp * dg * dt / hvap);
        aa1 = aa1 + edto * dz * g * delta * smax(SD(0.0), qeso(k) - qo(k));
    }
    if (aa1 <= SD(0.0)) cnvflg = false;
    if (!cnvflg) return out;
    // change a cloud of unit mass flux makes to the environment
    for (int k = 1; k <= imin(km, kmax); ++k) { dellah(k) = SD(0.0); dellaq(k) = SD(0.0); dellau(k) = SD(0.0); dellav(k) = SD(0.0); }
    {
        const SD dp = SD(1000.0) * del(1);
        dellah(1) = edto * etad(1) * (hcdo(1) - heo(1)) * g / dp;
        dellaq(1) = edto * etad(1) * (qcdo(1) - qo(1)) * g / dp;
        dellau(1) = edto * etad(1) * (ucdo(1) - uo(1)) * g / dp;
        dellav(1) = edto * etad(1) * (vcdo(1) - vo(1)) * g / dp;
    }
    for (int k = 2; k <= imin(km1, ktcon - 1); ++k) {
        SD aup = SD(1.0);
        if (k <= kb) aup = SD(0.0);
        SD adw = SD(1.0);
        if (k > jmin) adw = SD(0.0);
        const SD dp = SD(1000.0) * del(k);
        const SD dz = zi(k) - zi(k - 1);
        const SD dv1h = heo(k), dv2h = SD(.5) * (heo(k) + heo(k - 1)), dv3h = heo(k - 1);
        const SD dv1q = qo(k), dv2q = SD(.5) * (qo(k) + qo(k - 1)), dv3q = qo(k - 1);
        const SD dv1u = uo(k), dv2u = SD(.5) * (uo(k) + uo(k - 1)), dv3u = uo(k - 1);
        const SD dv1v = vo(k), dv2v = SD(.5) * (vo(k) + vo(k - 1)), dv3v = vo(k - 1);
        const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1));
        const SD tem1 = xlamud;
        SD ptem, ptem1;
        if (k <= kbcon) { ptem = xlamde; ptem1 = xlamd + xlamdd; }
        else { ptem = xlamde; ptem1 = xlamdd; }
        dellah(k) = dellah(k) +
            ((aup * eta(k) - adw * edto * etad(k)) * dv1h
             - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3h
             - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2h * dz
             + aup * tem1 * eta(k - 1) * SD(.5) * (hcko(k) + hcko(k - 1)) * dz
             + adw * edto * ptem1 * etad(k) * SD(.5) * (hcdo(k) + hcdo(k - 1)) * dz) * g / dp;
        dellaq(k) = dellaq(k) +
            ((aup * eta(k) - adw * edto * etad(k)) * dv1q
             - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3q
             - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2q * dz
             + aup * tem1 * eta(k - 1) * SD(.5) * (qcko(k) + qcko(k - 1)) * dz
             + adw * edto * ptem1 * etad(k) * SD(.5) * (qrcdo(k) + qrcdo(k - 1)) * dz) * g / dp;
        dellau(k) = dellau(k) +
            ((aup * eta(k) - adw * edto * etad(k)) * dv1u
             - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3u
             - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2u * dz
             + aup * tem1 * eta(k - 1) * SD(.5) * (ucko(k) + ucko(k - 1)) * dz
             + adw * edto * ptem1 * etad(k) * SD(.5) * (ucdo(k) + ucdo(k - 1)) * dz
             - pgcon * (aup * eta(k - 1) - adw * edto * etad(k)) * (dv1u - dv3u)) * g / dp;
        dellav(k) = dellav(k) +
            ((aup * eta(k) - adw * edto * etad(k)) * dv1v
             - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3v
             - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2v * dz
             + aup * tem1 * eta(k - 1) * SD(.5) * (vcko(k) + vcko(k - 1)) * dz
             + adw * edto * ptem1 * etad(k) * SD(.5) * (vcdo(k) + vcdo(k - 1)) * dz
             - pgcon * (aup * eta(k - 1) - adw * edto * etad(k)) * (dv1v - dv3v)) * g / dp;
    }
    {   // cloud top
        const int indx = ktcon;
        const SD dp = SD(1000.0) * del(indx);
        dellah(indx) = eta(indx - 1) * (hcko(indx - 1) - heo(indx - 1)) * g / dp;
        dellaq(indx) = eta(indx - 1) * (qcko(indx - 1) - qo(indx - 1)) * g / dp;
        dellau(indx) = eta(indx - 1) * (ucko(indx - 1) - uo(indx - 1)) * g / dp;
        dellav(indx) = eta(indx - 1) * (vcko(indx - 1) - vo(indx - 1)) * g / dp;
        dellal(indx) = eta(indx - 1) * qlko_ktcon * g / dp;
    }
    // final changed variable per unit mass flux
    for (int k = 1; k <= imin(km, kmax); ++k) {
        if (k > ktcon) { qo(k) = q1(k); to(k) = t1(k); }
        if (k <= ktcon) {
            qo(k) = dellaq(k) * mbdt + q1(k);
            const SD dellat = (dellah(k) - hvap * dellaq(k)) / cp;
            to(k) = dellat * mbdt + t1(k);
            qo(k) = smax(qo(k), SL(1.e-10));
        }
    }
    for (int k = 1; k <= imin(km, kmax); ++k) {
        qeso(k) = SL(0.01) * fpvs(tb, to(k));
        qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
        qeso(k) = smax(qeso(k), SL(1.e-8));
    }
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            const SD dz = SD(.5) * (zo(k + 1) - zo(k));
            const SD dp = SD(.5) * (pfld(k + 1) - pfld(k));
            const SD es = SL(0.01) * fpvs(tb, to(k + 1));
            const SD pprime = pfld(k + 1) + epsm1 * es;
            const SD qs = eps * es / pprime;
            const SD dqsdp = -(qs / pprime);
            const SD desdt = es * (fact1 / to(k + 1) + fact2 / sq(to(k + 1)));
            const SD dqsdt = qs * pfld(k + 1) * desdt / (es * pprime);
            const SD gamma = el2orc * qeso(k + 1) / sq(to(k + 1));
            const SD dt = (g * dz + hvap * dqsdp * dp) / (cp * (SD(1.0) + gamma));
            const SD dq = dqsdt * dt + dqsdp * dp;
            to(k) = to(k + 1) + dt;
            qo(k) = qo(k + 1) + dq;
            po(k) = SD(.5) * (pfld(k) + pfld(k + 1));
        }
    }
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            qeso(k) = SL(0.01) * fpvs(tb, to(k));
            qeso(k) = eps * qeso(k) / (po(k) + epsm1 * qeso(k));
            qeso(k) = smax(qeso(k), SL(1.e-8));
            qo(k) = smax(qo(k), SL(1.e-10));
            heo(k) = SD(.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qo(k);
            heso(k) = SD(.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qeso(k);
        }
    }
    {
        const int k = kmax;
        heo(k) = g * zo(k) + cp * to(k) + hvap * qo(k);
        heso(k) = g * zo(k) + cp * to(k) + hvap * qeso(k);
    }
    // static control: moisture and cloud work functions
    xaa0 = SD(0.0);
    xpwav = SD(0.0);
    hcko(kb) = heo(kb);
    qcko(kb) = qo(kb);
    for (int k = imax(2, kb + 1); k <= imin(km1, ktcon); ++k) {
        const SD dz = zi(k) - zi(k - 1);
        const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) * dz;
        const SD tem1 = SD(0.5) * xlamud * dz;
        const SD factor = SD(1.0) + tem - tem1;
        hcko(k) = ((SD(1.0) - tem1) * hcko(k - 1) + tem * SD(0.5) * (heo(k) + heo(k - 1))) / factor;
    }
    {
        SD xdby = SD(0.0);   // a scalar carried across k in WRF; always set before read
        for (int k = 2; k <= km1; ++k) {
            if (k > kb && k < ktcon) {
                const SD dz = zi(k) - zi(k - 1);
                const SD gamma = el2orc * qeso(k) / sq(to(k));
                xdby = hcko(k) - heso(k);
                const SD xqrch = qeso(k) + gamma * xdby / (hvap * (SD(1.0) + gamma));
                const SD tem = SD(0.5) * (xlamue(k) + xlamue(k - 1)) * dz;
                const SD tem1 = SD(0.5) * xlamud * dz;
                const SD factor = SD(1.0) + tem - tem1;
                qcko(k) = ((SD(1.0) - tem1) * qcko(k - 1) + tem * SD(0.5) * (qo(k) + qo(k - 1))) / factor;
                const SD dq = eta(k) * (qcko(k) - xqrch);
                if (k >= kbcon && dq > SD(0.0)) {
                    const SD etah = SD(.5) * (eta(k) + eta(k - 1));
                    SD qlk;
                    if (ncloud > 0 && k > jmin) qlk = dq / (eta(k) + etah * (c0 + c1) * dz);
                    else qlk = dq / (eta(k) + etah * c0 * dz);
                    if (k < ktcon1) xaa0 = xaa0 - dz * g * qlk;
                    qcko(k) = qlk + xqrch;
                    const SD xpw = etah * c0 * dz * qlk;
                    xpwav = xpwav + xpw;
                }
            }
            if (k >= kbcon && k < ktcon1) {
                const SD dz1 = zo(k + 1) - zo(k);
                const SD gamma = el2orc * qeso(k) / sq(to(k));
                const SD rfact = SD(1.0) + delta * cp * gamma * to(k) / hvap;
                xaa0 = xaa0 + dz1 * (g / (cp * to(k))) * xdby / (SD(1.0) + gamma) * rfact;
                xaa0 = xaa0 + dz1 * g * delta * smax(SD(0.0), qeso(k) - qo(k));
            }
        }
    }
    {
        const int jmn = jmin;
        hcdo(jmn) = heo(jmn); qcdo(jmn) = qo(jmn); qrcd(jmn) = qeso(jmn);
        xpwev = SD(0.0);
    }
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD dz = zi(k + 1) - zi(k);
        SD tem, tem1;
        if (k >= kbcon) { tem = xlamde * dz; tem1 = SD(0.5) * xlamdd * dz; }
        else { tem = xlamde * dz; tem1 = SD(0.5) * (xlamd + xlamdd) * dz; }
        const SD factor = SD(1.0) + tem - tem1;
        hcdo(k) = ((SD(1.0) - tem1) * hcdo(k + 1) + tem * SD(0.5) * (heo(k) + heo(k + 1))) / factor;
    }
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD dq = qeso(k), dt = to(k);
        const SD gamma = el2orc * dq / sq(dt);
        const SD dh = hcdo(k) - heso(k);
        qrcd(k) = dq + (SD(1.0) / hvap) * (gamma / (SD(1.0) + gamma)) * dh;
        const SD dz = zi(k + 1) - zi(k);
        SD tem, tem1;
        if (k >= kbcon) { tem = xlamde * dz; tem1 = SD(0.5) * xlamdd * dz; }
        else { tem = xlamde * dz; tem1 = SD(0.5) * (xlamd + xlamdd) * dz; }
        const SD factor = SD(1.0) + tem - tem1;
        qcdo(k) = ((SD(1.0) - tem1) * qcdo(k + 1) + tem * SD(0.5) * (qo(k) + qo(k + 1))) / factor;
        const SD xpwd = etad(k + 1) * (qcdo(k) - qrcd(k));
        qcdo(k) = qrcd(k);
        xpwev = xpwev + xpwd;
    }
    {
        SD edtmax = edtmaxl;
        if (slimsk == SD(0.0)) edtmax = edtmaxs;
        if (xpwev >= SD(0.0)) {
            edtx = SD(0.0);
        } else {
            edtx = -(edtx * xpwav / xpwev);
            edtx = smin(edtx, edtmax);
        }
    }
    for (int k = imin(km1, jmin - 1); k >= 1; --k) {
        const SD gamma = el2orc * qeso(k) / sq(to(k));
        const SD dhh = hcdo(k), dt = to(k), dg = gamma, dh = heso(k);
        const SD dz = -(zo(k + 1) - zo(k));
        xaa0 = xaa0 + edtx * dz * (g / (cp * dt)) * ((dhh - dh) / (SD(1.0) + dg))
                      * (SD(1.0) + delta * cp * dg * dt / hvap);
        xaa0 = xaa0 + edtx * dz * g * delta * smax(SD(0.0), qeso(k) - qo(k));
    }
    // critical cloud work function
    {
        const SD pk = pfld(ktcon);
        if (pk < PCRIT(15)) {
            acrt = acrit[15] * (SD(975.0) - pk) / (SD(975.0) - PCRIT(15));
        } else if (pk > PCRIT(1)) {
            acrt = acrit[1];
        } else {
            int k = (int)((SD(850.0) - pk) / SD(50.0)).v + 2;
            k = imin(k, 15);
            k = imax(k, 2);
            acrt = acrit[k] + (acrit[k - 1] - acrit[k]) * (pk - PCRIT(k)) / (PCRIT(k - 1) - PCRIT(k));
        }
    }
    {
        if (pdot <= w4) acrtfct = (pdot - w4) / (w3 - w4);
        else if (pdot >= -w4) acrtfct = -(pdot + w4) / (w4 - w3);
        else acrtfct = SD(0.0);
        acrtfct = smax(acrtfct, SD(-1.0));
        acrtfct = smin(acrtfct, SD(1.0));
        acrtfct = SD(1.0) - acrtfct;
        dtconv = dt2 + smax(SD(1800.0) - dt2, SD(0.0)) * (pdot - w2) / (w1 - w2);
        dtconv = smax(dtconv, dtmin);
        dtconv = smin(dtconv, dtmax);
    }
    // large-scale forcing and the cloud-base mass flux
    SD xmb;
    {
        const SD fld = (aa1 - acrt * acrtfct) / dtconv;
        if (fld <= SD(0.0)) cnvflg = false;
        if (cnvflg) {
            const SD xk = (xaa0 - aa1) / mbdt;
            if (xk >= SD(0.0)) cnvflg = false;
            if (cnvflg) {
                xmb = -(fld / xk);
                xmb = smin(xmb, xmbmax);
            }
        }
    }
    if (!cnvflg) return out;
    // restore to, qo, uo, vo in case convection stops
    for (int k = 1; k <= imin(km, kmax); ++k) {
        to(k) = t1(k); qo(k) = q1(k); uo(k) = u1(k); vo(k) = v1(k);
        qeso(k) = SL(0.01) * fpvs(tb, t1(k));
        qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
        qeso(k) = smax(qeso(k), SL(1.e-8));
    }
    // feedback
    for (int k = 1; k <= imin(imin(km, kmax), ktcon); ++k) {
        const SD dellat = (dellah(k) - hvap * dellaq(k)) / cp;
        t1(k) = t1(k) + dellat * xmb * dt2;
        q1(k) = q1(k) + dellaq(k) * xmb * dt2;
        const SD tem = SD(1.0) / rcs;
        u1(k) = u1(k) + dellau(k) * xmb * dt2 * tem;
        v1(k) = v1(k) + dellav(k) * xmb * dt2 * tem;
    }
    for (int k = 1; k <= imin(imin(km, kmax), ktcon); ++k) {
        qeso(k) = SL(0.01) * fpvs(tb, t1(k));
        qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
        qeso(k) = smax(qeso(k), SL(1.e-8));
    }
    SD rntot = SD(0.0), delqev = SD(0.0), delq2 = SD(0.0);
    bool flg = cnvflg;
    for (int k = imin(imin(km, kmax), ktcon - 1); k >= 1; --k) {
        SD aup = SD(1.0);
        if (k <= kb) aup = SD(0.0);
        SD adw = SD(1.0);
        if (k >= jmin) adw = SD(0.0);
        const SD rain = aup * pwo(k) + adw * edto * pwdo(k);
        rntot = rntot + rain * xmb * SL(.001) * dt2;
    }
    for (int k = imin(km, kmax); k >= 1; --k) {
        SD qevap = SD(0.0);
        if (k < ktcon) {
            SD aup = SD(1.0);
            if (k <= kb) aup = SD(0.0);
            SD adw = SD(1.0);
            if (k >= jmin) adw = SD(0.0);
            const SD rain = aup * pwo(k) + adw * edto * pwdo(k);
            out.rn = out.rn + rain * xmb * SL(.001) * dt2;
        }
        if (flg && k < ktcon) {
            SD evef = edt * evfact;
            if (slimsk == SD(1.0)) evef = edt * evfactl;
            const SD qcond = evef * (q1(k) - qeso(k)) / (SD(1.0) + el2orc * qeso(k) / sq(t1(k)));
            const SD dp = SD(1000.0) * del(k);
            if (out.rn > SD(0.0) && qcond < SD(0.0)) {
                qevap = -(qcond * (SD(1.0) - sexp(-(SL(.32) * ssqrt(dt2 * out.rn)))));
                qevap = smin(qevap, out.rn * SD(1000.0) * g / dp);
                delq2 = delqev + SL(.001) * qevap * dp / g;
            }
            if (out.rn > SD(0.0) && qcond < SD(0.0) && delq2 > rntot) {
                qevap = SD(1000.0) * g * (rntot - delqev) / dp;
                flg = false;
            }
            if (out.rn > SD(0.0) && qevap > SD(0.0)) {
                q1(k) = q1(k) + qevap;
                t1(k) = t1(k) - elocp * qevap;
                out.rn = out.rn - SL(.001) * qevap * dp / g;
                delqev = delqev + SL(.001) * dp * qevap / g;
            }
        }
    }
    // precipitation in m; back out when rain evaporation made it negative
    if (out.rn < SD(0.0) && !flg) out.rn = SD(0.0);
    if (out.rn <= SD(0.0)) {
        out.rn = SD(0.0);
    } else {
        out.ktop = ktcon;
        out.kbot = kbcon;
        out.kcnv = 1;
        out.cldwrk = aa1;
    }
    // cloud water (ncloud > 0): detrained condensate split ice/water by T
    if (out.rn > SD(0.0)) {
        for (int k = imax(1, kb + 1); k <= imin(km, ktcon); ++k) {
            const SD tem = dellal(k) * xmb * dt2;
            const SD tem1 = smax(SD(0.0), smin(SD(1.0), (tcr - t1(k)) * tcrf));
            if (ql2(k) > SD(-999.0)) {
                ql1(k) = ql1(k) + tem * tem1;
                ql2(k) = ql2(k) + tem * (SD(1.0) - tem1);
            } else {
                ql1(k) = ql1(k) + tem;
            }
        }
    } else {
        for (int k = 1; k <= imin(km, kmax); ++k) { t1(k) = to(k); q1(k) = qo(k); u1(k) = uo(k); v1(k) = vo(k); }
    }
#undef PCRIT
    return out;
}

// ---- MSTADBT3 (:2517), one column, levels 1..km ------------------------------
__device__ void mstadbt3(const double* tb, int km, int k1, int k2, V prsl, V prslk,
                         V tenv, V qenv, int& klcl, int& kbot, int& ktop, V tcld, V qcld)
{
    const SD eps = c_eps(), epsm1 = c_epsm1(), fv = c_fvirt();
    SD slkma = SD(0.0), thema = SD(0.0);
    for (int k = k1; k <= k2; ++k) {
        const SD pv = SD(1000.0) * prsl(k) * qenv(k) / (eps - epsm1 * qenv(k));
        const SD tdpd = tenv(k) - ftdp(tb, pv);
        SD tlcl, slklcl;
        if (tdpd > SD(0.0)) {
            tlcl = ftlcl(tb, tenv(k), tdpd);
            slklcl = prslk(k) * tlcl / tenv(k);
        } else {
            tlcl = tenv(k);
            slklcl = prslk(k);
        }
        const SD thelcl = fthe(tb, tlcl, slklcl);
        if (thelcl > thema) { slkma = slklcl; thema = thelcl; }
    }
    klcl = km + 1; kbot = km + 1; ktop = 0;
    for (int k = 1; k <= km; ++k) { tcld(k) = SD(0.0); qcld(k) = SD(0.0); }
    for (int k = k1; k <= km; ++k) {
        if (prslk(k) <= slkma) {
            klcl = imin(klcl, k);
            SD tma, qma;
            stma(tb, thema, prslk(k), tma, qma);
            const SD tvcld = tma * (SD(1.0) + fv * qma);
            const SD tvenv = tenv(k) * (SD(1.0) + fv * qenv(k));
            if (tvcld > tvenv) {
                kbot = imin(kbot, k);
                ktop = imax(ktop, k);
                tcld(k) = tma - tenv(k);
                qcld(k) = qma - qenv(k);
            }
        }
    }
}

// ---- OLD_ARW_SHALCV (:2297) with TRIDI2T3 (:2475), one column -----------------
// A column alone in WRF's compressed set (N2 = 1).  WRF solves the whole
// set over [min KBOT, max KTOP]; the rows outside a column's own range are
// identity rows (CK = 0 makes AL = AU = -0, AD = 1), which leave its words
// unchanged -- the oracle's row check holds that on every case.
__device__ void old_arw_shalcv(const double* tb, const Ws& w, int km, SD dt,
                               V del, V prsi, V prsl, V prslk, int kuo,
                               V q, V t, SD dpshc)
{
    const SD g = C_G, cp = C_CP, hvap = C_HVAP, rd = C_RD;
    const SD gocp = g / cp;
    const int kliftl = 2, kliftu = 2;
    V q2 = w.slot(S_Q2), t2 = w.slot(S_T2), al = w.slot(S_AL), ad = w.slot(S_AD),
      au = w.slot(S_AU);
    bool lshc = false;
    if (kuo == 0) {
        for (int k = 1; k <= km - 1; ++k) {
            const SD eldq = hvap * (q(k) - q(k + 1));
            const SD cpdt = cp * (t(k) - t(k + 1));
            const SD rtdls = (prsl(k) - prsl(k + 1)) / prsi(k + 1) * rd * SD(0.5) * (t(k) + t(k + 1));
            const SD dmse = eldq + cpdt - rtdls;
            lshc = lshc || dmse > SD(0.0);
        }
    }
    if (!lshc) return;
    for (int k = 1; k <= km; ++k) { q2(k) = q(k); t2(k) = t(k); }
    int ktopm = km;
    for (int k = 2; k <= km; ++k)
        if (prsi(1) - prsi(k) <= dpshc) ktopm = k;
    int klcl, kbot, ktop;
    mstadbt3(tb, km - 1, kliftl, kliftu, prsl, prslk, t2, q2, klcl, kbot, ktop, al, au);
    kbot = imin(klcl - 1, ktopm - 1);
    ktop = imin(ktop + 1, ktopm);
    lshc = false;
    for (int k = 1; k <= km - 1; ++k) {
        if (k >= kbot && k < ktop) {
            const SD eldq = hvap * (q2(k) - q2(k + 1));
            const SD cpdt = cp * (t2(k) - t2(k + 1));
            const SD rtdls = (prsl(k) - prsl(k + 1)) / prsi(k + 1) * rd * SD(0.5) * (t2(k) + t2(k + 1));
            const SD dmse = eldq + cpdt - rtdls;
            lshc = lshc || dmse > SD(0.0);
            au(k) = g / rtdls;
        }
    }
    if (!lshc) { kbot = km + 1; ktop = 0; }
    const int k1 = kbot, k2 = ktop;
    const int kt = k2 - k1 + 1;
    if (kt < 2) return;
    ad(k1) = SD(1.0);
    for (int k = k1; k <= k2 - 1; ++k) {
        const SD dtodsl = dt / del(k);
        const SD dtodsu = dt / del(k + 1);
        const SD dsig = prsl(k) - prsl(k + 1);
        SD ck;
        if (k == kbot) ck = SD(1.5);
        else if (k == ktop - 1) ck = SD(1.0);
        else if (k == ktop - 2) ck = SD(3.0);
        else if (k > kbot && k < ktop - 2) ck = SD(5.0);
        else ck = SD(0.0);
        const SD dsdz1 = ck * dsig * au(k) * gocp;
        const SD dsdz2 = ck * dsig * au(k) * au(k);
        au(k) = -(dtodsl * dsdz2);
        al(k) = -(dtodsu * dsdz2);
        ad(k) = ad(k) - au(k);
        ad(k + 1) = SD(1.0) - al(k);
        t2(k) = t2(k) + dtodsl * dsdz1;
        t2(k + 1) = t2(k + 1) - dtodsu * dsdz1;
    }
    // TRIDI2T3(L=1, N=kt, CL=AL(k1..), CM=AD(k1..), CU=AU(k1..), R1=Q2, R2=T2,
    //          AU, A1=Q2, A2=T2 in place).  Local row n is level k1+n-1;
    //          CL(n) is AL of the row before it.
    {
        const int n = kt;
#define CL(r) al(k1 + (r) - 2)
#define CM(r) ad(k1 + (r) - 1)
#define CU(r) au(k1 + (r) - 1)
#define A1(r) q2(k1 + (r) - 1)
#define A2(r) t2(k1 + (r) - 1)
        SD fk = SD(1.0) / CM(1);
        CU(1) = fk * CU(1);
        A1(1) = fk * A1(1);
        A2(1) = fk * A2(1);
        for (int r = 2; r <= n - 1; ++r) {
            fk = SD(1.0) / (CM(r) - CL(r) * CU(r - 1));
            CU(r) = fk * CU(r);
            A1(r) = fk * (A1(r) - CL(r) * A1(r - 1));
            A2(r) = fk * (A2(r) - CL(r) * A2(r - 1));
        }
        fk = SD(1.0) / (CM(n) - CL(n) * CU(n - 1));
        A1(n) = fk * (A1(n) - CL(n) * A1(n - 1));
        A2(n) = fk * (A2(n) - CL(n) * A2(n - 1));
        for (int r = n - 1; r >= 1; --r) {
            A1(r) = A1(r) - CU(r) * A1(r + 1);
            A2(r) = A2(r) - CU(r) * A2(r + 1);
        }
#undef CL
#undef CM
#undef CU
#undef A1
#undef A2
    }
    for (int k = k1; k <= k2; ++k) { q(k) = q2(k); t(k) = t2(k); }
}

// ---- entry points --------------------------------------------------------------
__device__ __forceinline__ Ws ws_for(double* ws, int kw, int ncolw, int col) {
    return Ws{ws + col, kw, ncolw};
}
__device__ __forceinline__ V gview(double* a, int ncol, int col) { return V{a + col, ncol}; }

// Stage 1: SASCNVN on the kind_phys inputs CU_SAS forms (oracle stage1.bin).
// Level arrays are (nzmax, ncol) level-major; km per column in kmcol.
extern "C" __global__ void sas_stage_deep(
    const double* tb, double* ws, int kw, int ncol, const int* kmcol,
    const double* delt, const double* pgcon, const double* massf,
    const double* ps, const double* slimsk,
    double* del, double* prsl, double* phil, double* ql1, double* ql2,
    double* q1, double* t1, double* u1, double* v1, double* dot,
    double* cldwrk, double* rn, int* kbot, int* ktop, int* kcnv)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const Ws w = ws_for(ws, kw, ncol, col);
    const DeepOut o = sascnvn(tb, w, kmcol[col], SD(delt[col]),
        gview(del, ncol, col), gview(prsl, ncol, col), SD(ps[col]),
        gview(phil, ncol, col), gview(ql1, ncol, col), gview(ql2, ncol, col),
        gview(q1, ncol, col), gview(t1, ncol, col), gview(u1, ncol, col),
        gview(v1, ncol, col), SD(1.0), SD(slimsk[col]), gview(dot, ncol, col),
        SD(pgcon[col]), SD(massf[col]));
    cldwrk[col] = o.cldwrk.v; rn[col] = o.rn.v;
    kbot[col] = o.kbot; ktop[col] = o.ktop; kcnv[col] = o.kcnv;
}

// Stage 2: OLD_ARW_SHALCV on the post-SASCNVN state (oracle stage2.bin).
extern "C" __global__ void sas_stage_shal(
    const double* tb, double* ws, int kw, int ncol, const int* kmcol,
    const int* kcnv, const double* delt, const double* dpshc,
    double* del, double* prsi, double* prsl, double* prslk, double* q, double* t)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const Ws w = ws_for(ws, kw, ncol, col);
    old_arw_shalcv(tb, w, kmcol[col], SD(delt[col]), gview(del, ncol, col),
                   gview(prsi, ncol, col), gview(prsl, ncol, col),
                   gview(prslk, ncol, col), kcnv[col], gview(q, ncol, col),
                   gview(t, ncol, col), SD(dpshc[col]));
}

// Stage 3: CU_SAS (:7) for one column, WRF REAL in and out.
//   level arrays (nz rows) and interface arrays w, p8w (nz+1 rows),
//   level-major [(k-1)*stride + col0 + col]; this launch covers columns
//   col0 .. col0+ncol-1 of the grid, the workspace holds ncol columns;
//   kmcol == nullptr means every column has nz.
//   shalconv: CU_SAS's shalconv_use (1 in an ARW build, :233-245).
//   has_qc / has_qi: P_QC / P_QI >= P_FIRST_SCALAR (:482, :490).
#define SAS_COLUMN_ARGS const double* tb, double* ws, int kw, int ncol, int col0, long long stride, \
    int nz, const int* kmcol, \
    const float* t3d, const float* qv3d, const float* qc3d, const float* qi3d, \
    const float* u3d, const float* v3d, const float* w3d, const float* pcps, \
    const float* pi3d, const float* rho3d, const float* dz8w, const float* p8w, \
    const float* xland, const float* dtcol, const int* stepcucol, \
    const float* pgconcol, float dt_all, int stepcu_all, float pgcon_all, \
    int shalconv, int has_qc, int has_qi, \
    float* rthcuten, float* rqvcuten, float* rqccuten, float* rqicuten, \
    float* raincv, float* pratec, float* hbot, float* htop, \
    int* kbot_out, int* ktop_out, int* kcnv_out, double* cldwrk_out

// Parallel independent level work; ordered recurrences stay column-owned.
extern "C" __global__ void sas_prepare_fields(SAS_COLUMN_ARGS)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y + 1;
    if (col >= ncol) return;
    const long long g = (long long)col0 + col;
    const int km = kmcol ? kmcol[g] : nz;
    if (k > km) return;
    const Ws w = ws_for(ws, kw, ncol, col);
#define LV(a, k) a[(long long)((k) - 1) * stride + g]
    V del = w.slot(S_DEL), prsl = w.slot(S_PRSL), dot = w.slot(S_DOT),
      q1 = w.slot(S_Q1), t1 = w.slot(S_T1), u1 = w.slot(S_U1), v1 = w.slot(S_V1),
      ql1 = w.slot(S_QL1), ql2 = w.slot(S_QL2), prslk = w.slot(S_PRSLK);
    const SD grav = C_G, rd = C_RD, rovcp = c_rocp();
        prsl(k) = wid(__fmul_rn(LV(pcps, k), .001f));
        dot(k) = -(wid(5.0e-4f) * grav * wid(LV(rho3d, k))
                   * wid(__fadd_rn(LV(w3d, k), LV(w3d, k + 1))));
        del(k) = prsl(k) * grav / rd * wid(LV(dz8w, k)) / wid(LV(t3d, k));
        u1(k) = wid(LV(u3d, k));
        v1(k) = wid(LV(v3d, k));
        const float qv = LV(qv3d, k), qi = LV(qi3d, k), qc = LV(qc3d, k);
        q1(k) = wid(__fdiv_rn(qv, __fadd_rn(1.0f, qv)));
        t1(k) = wid(LV(t3d, k));
        ql1(k) = wid(__fdiv_rn(qi, __fadd_rn(1.0f, qi)));
        ql2(k) = wid(__fdiv_rn(qc, __fadd_rn(1.0f, qc)));
        prslk(k) = spow(prsl(k) * SL(.01), rovcp);
#undef LV
}

extern "C" __global__ void sas_prepare_columns(SAS_COLUMN_ARGS)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const long long g = (long long)col0 + col;
    const int km = kmcol ? kmcol[g] : nz;
    const Ws w = ws_for(ws, kw, ncol, col);
#define LV(a, k) a[(long long)((k) - 1) * stride + g]
    V phil = w.slot(S_PHIL), del = w.slot(S_DEL), prsi = w.slot(S_PRSI);
    // The two native prefix sums retain their scalar operation order.
    SD z = SD(0.0), mid;
    for (int k = 2; k <= km; ++k) {
        const SD before = z;
        z = z + wid(LV(dz8w, k - 1));
        mid = (z + before) * SD(0.5);
        phil(k - 1) = mid * C_G;
    }
    phil(km) = (SD(2.0) * z - mid) * C_G;
    SD pressure = wid(__fmul_rn(LV(p8w, 1), .001f));
    prsi(1) = pressure;
    for (int k = 2; k <= km + 1; ++k) {
        pressure = pressure - del(k - 1);
        prsi(k) = pressure;
    }
    const SD ps = wid(__fmul_rn(LV(p8w, 1), .001f));
    const SD tx1 = SD(1.0) / ps;
    V prsl = w.slot(S_PRSL), t1 = w.slot(S_T1), q1 = w.slot(S_Q1);
    int kmax = km, kbmax = km, kbm = km;
    for (int k = 1; k <= km; ++k) {
        if (prsl(k) * tx1 > SL(0.04)) kmax = imin(km, k + 1);
        if (prsl(k) * tx1 > SL(0.45)) kbmax = k + 1;
        if (prsl(k) * tx1 > SL(0.70)) kbm = k + 1;
    }
    kbmax = imin(kbmax, kmax); kbm = imin(kbm, kmax);
    SD hmax = phil(1) + C_CP * t1(1) + C_HVAP * smax(q1(1), SL(1.e-10));
    int kb = 1;
    for (int k = 2; k <= kbm; ++k) {
        const SD heo = phil(k) + C_CP * t1(k) + C_HVAP * smax(q1(k), SL(1.e-10));
        if (heo > hmax) { kb = k; hmax = heo; }
    }
    const V meta = w.slot(S_DZI);
    meta(1) = SD((double)kmax); meta(2) = SD((double)kbmax);
    meta(3) = SD((double)kbm); w.slot(S_DZL)(1) = SD((double)kb);
#undef LV
}

extern "C" __global__ void sas_deep_prepare(SAS_COLUMN_ARGS)

{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y + 1;
    if (col >= ncol) return;
    const long long g = (long long)col0 + col;
    const int km = kmcol ? kmcol[g] : nz;
    if (k > km) return;
    const Ws w = ws_for(ws, kw, ncol, col);
#define LV(a, k) a[(long long)((k) - 1) * stride + g]
    V del = w.slot(S_DEL), prsl = w.slot(S_PRSL), dot = w.slot(S_DOT),
      q1 = w.slot(S_Q1), t1 = w.slot(S_T1), u1 = w.slot(S_U1), v1 = w.slot(S_V1),
      ql1 = w.slot(S_QL1), ql2 = w.slot(S_QL2), prslk = w.slot(S_PRSLK);
    const int kmax = (int)w.slot(S_DZI)(1).v;
    V phil = w.slot(S_PHIL), pfld = w.slot(S_PFLD), to = w.slot(S_TO), qo = w.slot(S_QO),
      uo = w.slot(S_UO), vo = w.slot(S_VO), qeso = w.slot(S_QESO),
      heo = w.slot(S_HEO), heso = w.slot(S_HESO), zo = w.slot(S_ZO),
      zi = w.slot(S_ZI), xlamue = w.slot(S_XLAMUE), frh = w.slot(S_FRH), po = w.slot(S_PO);
    const SD grav = C_G, cp = C_CP, hvap = C_HVAP, rv = C_RV;
    const SD eps = c_eps(), epsm1 = c_epsm1();
    const SD fact1 = (C_CVAP - C_CLIQ) / rv, fact2 = hvap / rv - fact1 * C_T0C;
    const SD el2orc = hvap * hvap / (rv * cp);
    const SD z = phil(k) / grav;
    zo(k) = z;
    if (k <= km - 1) {
        zi(k) = SD(0.5) * (z + phil(k + 1) / grav);
        xlamue(k) = SL(.1) / zi(k);
    }
    if (k <= kmax) {
        pfld(k) = prsl(k) * SD(10.0);
        to(k) = t1(k); qo(k) = smax(q1(k), SL(1.e-10));
        uo(k) = u1(k); vo(k) = v1(k);
        SD qe = SL(0.01) * fpvs(tb, t1(k));
        qe = eps * qe / (pfld(k) + epsm1 * qe);
        qeso(k) = smax(qe, SL(1.e-8));
        const SD tem = phil(k) + cp * to(k);
        heo(k) = tem + hvap * qo(k); heso(k) = tem + hvap * qeso(k);
    }
    if (k <= km - 1 && k <= kmax - 1) {
        const SD next_p = prsl(k + 1) * SD(10.0), next_t = t1(k + 1);
        SD next_qes = SL(0.01) * fpvs(tb, next_t);
        next_qes = eps * next_qes / (next_p + epsm1 * next_qes);
        next_qes = smax(next_qes, SL(1.e-8));
        const SD dz = SD(.5) * (phil(k + 1) / grav - z);
        const SD dp = SD(.5) * (next_p - pfld(k));
        const SD es = SL(0.01) * fpvs(tb, next_t);
        const SD pprime = next_p + epsm1 * es;
        const SD qs = eps * es / pprime;
        const SD dqsdp = -(qs / pprime);
        const SD desdt = es * (fact1 / next_t + fact2 / sq(next_t));
        const SD dqsdt = qs * next_p * desdt / (es * pprime);
        const SD gamma = el2orc * next_qes / sq(next_t);
        const SD dt = (grav * dz + hvap * dqsdp * dp) / (cp * (SD(1.0) + gamma));
        const SD dq = dqsdt * dt + dqsdp * dp;
        to(k) = next_t + dt;
        qo(k) = smax(q1(k + 1), SL(1.e-10)) + dq;
        po(k) = SD(.5) * (pfld(k) + next_p);
        qeso(k) = SL(0.01) * fpvs(tb, to(k));
        qeso(k) = eps * qeso(k) / (po(k) + epsm1 * qeso(k));
        qeso(k) = smax(qeso(k), SL(1.e-8));
        qo(k) = smax(qo(k), SL(1.e-10));
        frh(k) = SD(1.0) - smin(qo(k) / qeso(k), SD(1.0));
        heo(k) = SD(.5) * grav * (z + phil(k + 1) / grav) + cp * to(k) + hvap * qo(k);
        heso(k) = SD(.5) * grav * (z + phil(k + 1) / grav) + cp * to(k) + hvap * qeso(k);
        uo(k) = SD(.5) * (u1(k) + u1(k + 1));
        vo(k) = SD(.5) * (v1(k) + v1(k + 1));
    }
#undef LV
}

extern "C" __global__ void sas_column(SAS_COLUMN_ARGS)

{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const long long g = (long long)col0 + col;          // grid column
    const int km = kmcol ? kmcol[g] : nz;
    const float dt = dtcol ? dtcol[g] : dt_all;
    const int stepcu = stepcucol ? stepcucol[g] : stepcu_all;
    const float pgconf = pgconcol ? pgconcol[g] : pgcon_all;
    const Ws w = ws_for(ws, kw, ncol, col);
#define LV(a, k) a[(long long)((k) - 1) * stride + g]
    V del = w.slot(S_DEL), prsl = w.slot(S_PRSL),
      phil = w.slot(S_PHIL), dot = w.slot(S_DOT), q1 = w.slot(S_Q1), t1 = w.slot(S_T1),
      u1 = w.slot(S_U1), v1 = w.slot(S_V1), ql1 = w.slot(S_QL1), ql2 = w.slot(S_QL2),
      prslk = w.slot(S_PRSLK), prsi = w.slot(S_PRSI);
    const SD dpshc = SD(30.0);                          // 30_kind_phys
    const SD delt = wid(__fmul_rn(dt, (float)stepcu));   // DT*STEPCU, default REAL
    const SD pgcon_use = wid(pgconf);
    const SD massf = SL(9e9);                            // sas_mass_flux absent in ARW
    const SD ps = wid(__fmul_rn(LV(p8w, 1), .001f));
    const SD slimsk = wid(fabsf(__fsub_rn(xland[g], 2.0f)));
    const DeepOut o = sascnvn<true>(tb, w, km, delt, del, prsl, ps, phil, ql1, ql2,
                              q1, t1, u1, v1, SD(1.0), slimsk, dot, pgcon_use, massf);
    const float raincv1 = nar(o.rn * SD(1000.0) / SD((double)stepcu));
    const float pratec1 = nar(o.rn * SD(1000.0) / wid(__fmul_rn((float)stepcu, dt)));
    if (shalconv == 1)
        old_arw_shalcv(tb, w, km, delt, del, prsi, prsl, prslk, o.kcnv, q1, t1, dpshc);
    raincv[g] = __fadd_rn(raincv1, 0.0f);
    pratec[g] = __fadd_rn(pratec1, 0.0f);
    hbot[g] = (float)o.kbot;
    htop[g] = (float)o.ktop;
    if (kbot_out) kbot_out[g] = o.kbot;
    if (ktop_out) ktop_out[g] = o.ktop;
    if (kcnv_out) kcnv_out[g] = o.kcnv;
    if (cldwrk_out) cldwrk_out[g] = o.cldwrk.v;
#undef LV
}

extern "C" __global__ void sas_finish(SAS_COLUMN_ARGS)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y + 1;
    if (col >= ncol) return;
    const long long g = (long long)col0 + col;
    const int km = kmcol ? kmcol[g] : nz;
    if (k > km) return;
    const Ws w = ws_for(ws, kw, ncol, col);
#define LV(a, k) a[(long long)((k) - 1) * stride + g]
    V del = w.slot(S_DEL), prsl = w.slot(S_PRSL), dot = w.slot(S_DOT),
      q1 = w.slot(S_Q1), t1 = w.slot(S_T1), u1 = w.slot(S_U1), v1 = w.slot(S_V1),
      ql1 = w.slot(S_QL1), ql2 = w.slot(S_QL2), prslk = w.slot(S_PRSLK);
    const SD grav = C_G, rd = C_RD, rovcp = c_rocp();
    const float dt = dtcol ? dtcol[g] : dt_all;
    const int stepcu = stepcucol ? stepcucol[g] : stepcu_all;
    const SD delt = wid(__fmul_rn(dt, (float)stepcu));
    const SD rdelt = SD(1.0) / delt;
        LV(rthcuten, k) = nar((t1(k) - wid(LV(t3d, k))) / wid(LV(pi3d, k)) * rdelt);
        LV(rqvcuten, k) = nar((q1(k) / (SD(1.0) - q1(k)) - wid(LV(qv3d, k))) * rdelt);
    if (has_qc)
            LV(rqccuten, k) = nar((ql2(k) / (SD(1.0) - ql2(k)) - wid(LV(qc3d, k))) * rdelt);
    if (has_qi)
            LV(rqicuten, k) = nar((ql1(k) / (SD(1.0) - ql1(k)) - wid(LV(qi3d, k))) * rdelt);
#undef LV
}

#undef SAS_COLUMN_ARGS

#endif  // SAS_CU
