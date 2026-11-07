// cu_osas.cu -- WRF v4.6.1 cu_physics = 95: the old GFS simplified
// Arakawa-Schubert deep convection (OSASCNV) with its shallow moist-adiabat
// mixing (SHALCV, MSTADBT3, TRIDI2T3), the CU_OSAS column driver and the
// gfuncphys lookup tables it reads (phys/module_cu_osas.F,
// phys/module_gfs_funcphys.F, phys/module_gfs_physcons.F).  ArWen's own
// transcription, Apache-2.0.  One thread per column.
//
// The loader (gpuwm/core/kernels/__init__.py _EXTRA_HEADERS["cu_osas"])
// prepends the binary64 column vocabulary (column_r8.cuh). Default physics
// uses CUDA exp/log/pow and permits fused operations. Strict replay
// (GPUWM_WRF_EXACT=1) and table generation use individually rounded
// operations and the pinned binary64 exp/log/pow (glibc_flt64.cuh).
//
// Arithmetic contract, matched against the WRF 0 ULP referee build
// (gfortran 13.3 -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize,
// tools/osas_wrf461_oracle):
//   * kind_phys is binary64.  A literal without _kind_phys is a default
//     REAL: its value is the float32 rounding of the decimal, widened
//     exactly.  OSAS_F(1.e-8) spells that; never R8(1.e-8).
//   * Driver statements that are pure default-REAL (Pcps*.001,
//     QV/(1.+QV), ABS(XLAND-2.), STEPCU*DT, w+w) round in float32 first.
//   * x**2 and x**3 are multiply chains; every other power with a real
//     exponent is pow.  NINT is round half away from zero.
//   * Operand order is Fortran's left-to-right; each line cites its source.
//
// The shipped identity differs from WRF in two declared places, both
// switchable for the parity suite only (flags argument):
//   * XKT2 (the cloud-top selector) is an INPUT here.  WRF draws it from a
//     clock-seeded random_number on every call and reseeds per J row from
//     the same clock, so WRF is not reproducible and rows share their
//     "random" sequence.  The launcher fills it: strict mode takes an
//     injected value, default mode a counter-based draw keyed on the step
//     and the column (cu_osas_xkt2 below).
//   * OSAS_FLAG_WRF_QL_BACKOUT off (the shipped identity): when the
//     column's rain evaporates to RN <= 0, WRF backs the heating and the
//     moistening out (module_cu_osas.F:2174-2183) but keeps the cloud-top
//     detrainment it already added to QL (:2044-2058) -- condensate created
//     with no vapour removed.  The shipped identity restores QL with T1 and
//     Q1.  Set, the kernel reproduces WRF's water source exactly.

#define OSAS_F(x) R8((double)x##f)

#define OSAS_FLAG_WRF_QL_BACKOUT 1
#define OSAS_FLAG_STAGES 2

// ---- gfuncphys table layout (one binary64 buffer) --------------------------
#define OSAS_NXPVS 7501
#define OSAS_NXTDP 5001
#define OSAS_NXTHE 241
#define OSAS_NYTHE 151
#define OSAS_NXMA 151
#define OSAS_NYMA 121
#define OSAS_NXTLCL 151
#define OSAS_NYTLCL 61
#define OSAS_TB_C1XPVS 0
#define OSAS_TB_C2XPVS 1
#define OSAS_TB_C1XTDP 2
#define OSAS_TB_C2XTDP 3
#define OSAS_TB_C1XTHE 4
#define OSAS_TB_C2XTHE 5
#define OSAS_TB_C1YTHE 6
#define OSAS_TB_C2YTHE 7
#define OSAS_TB_C1XMA 8
#define OSAS_TB_C2XMA 9
#define OSAS_TB_C1YMA 10
#define OSAS_TB_C2YMA 11
#define OSAS_TB_C1XTLCL 12
#define OSAS_TB_C2XTLCL 13
#define OSAS_TB_C1YTLCL 14
#define OSAS_TB_C2YTLCL 15
#define OSAS_TB_PVS 16
#define OSAS_TB_TDP (OSAS_TB_PVS + OSAS_NXPVS)
#define OSAS_TB_THE (OSAS_TB_TDP + OSAS_NXTDP)
#define OSAS_TB_TMA (OSAS_TB_THE + OSAS_NXTHE * OSAS_NYTHE)
#define OSAS_TB_QMA (OSAS_TB_TMA + OSAS_NXMA * OSAS_NYMA)
#define OSAS_TB_TLCL (OSAS_TB_QMA + OSAS_NXMA * OSAS_NYMA)
#define OSAS_TB_WORDS (OSAS_TB_TLCL + OSAS_NXTLCL * OSAS_NYTLCL)

// ---- module_gfs_physcons (each a default-REAL literal widened) ----------------
struct OsasPhys {
    R8 g, rd, rv, cp, cvap, cliq, csol, hvap, hfus, psat, t0c, ttp;
    R8 rocp, cpor, fvirt, eps, epsm1, dldt, xpona, xponb, psatb;
};

__device__ __forceinline__ OsasPhys osas_phys()
{
    OsasPhys c;
    c.g = OSAS_F(9.80665e+0);       // module_gfs_physcons.F con_g
    c.rd = OSAS_F(2.8705e+2);       // con_rd
    c.rv = OSAS_F(4.6150e+2);       // con_rv
    c.cp = OSAS_F(1.0046e+3);       // con_cp
    c.cvap = OSAS_F(1.8460e+3);     // con_cvap
    c.cliq = OSAS_F(4.1855e+3);     // con_cliq
    c.csol = OSAS_F(2.1060e+3);     // con_csol
    c.hvap = OSAS_F(2.5000e+6);     // con_hvap
    c.hfus = OSAS_F(3.3358e+5);     // con_hfus
    c.psat = OSAS_F(6.1078e+2);     // con_psat
    c.t0c = OSAS_F(2.7315e+2);      // con_t0c
    c.ttp = OSAS_F(2.7316e+2);      // con_ttp
    c.rocp = c.rd / c.cp;           // con_rocp
    c.cpor = c.cp / c.rd;           // con_cpor
    c.fvirt = c.rv / c.rd - R8(1.0);    // con_fvirt
    c.eps = c.rd / c.rv;            // con_eps
    c.epsm1 = c.rd / c.rv - R8(1.0);    // con_epsm1
    c.dldt = c.cvap - c.cliq;       // con_dldt
    c.xpona = -c.dldt / c.rv;       // con_xpona
    c.xponb = -c.dldt / c.rv + c.hvap / (c.rv * c.ttp);   // con_xponb
    c.psatb = c.psat * OSAS_F(1.e-5);  // funcphys psatb
    return c;
}

// ---- gfuncphys: the saturation and moist-adiabat functions ------------------
// fpvsx (module_gfs_funcphys.F:814): mixed-phase saturation vapour pressure.
__device__ __forceinline__ R8 osas_fpvsx(R8 t, const OsasPhys& c)
{
    const R8 tliq = c.ttp;
    const R8 tice = c.ttp - R8(20.0);
    const R8 dldtl = c.cvap - c.cliq;
    const R8 heatl = c.hvap;
    const R8 xponal = -dldtl / c.rv;
    const R8 xponbl = -dldtl / c.rv + heatl / (c.rv * c.ttp);
    const R8 dldti = c.cvap - c.csol;
    const R8 heati = c.hvap + c.hfus;
    const R8 xponai = -dldti / c.rv;
    const R8 xponbi = -dldti / c.rv + heati / (c.rv * c.ttp);
    const R8 tr = c.ttp / t;
    if (t >= tliq) {
        return c.psat * uw_pow(tr, xponal) * uw_exp(xponbl * (R8(1.0) - tr));
    } else if (t < tice) {
        return c.psat * uw_pow(tr, xponai) * uw_exp(xponbi * (R8(1.0) - tr));
    }
    const R8 w = (t - tice) / (tliq - tice);
    const R8 pvl = c.psat * uw_pow(tr, xponal) * uw_exp(xponbl * (R8(1.0) - tr));
    const R8 pvi = c.psat * uw_pow(tr, xponai) * uw_exp(xponbi * (R8(1.0) - tr));
    return w * pvl + (R8(1.0) - w) * pvi;
}

// ftdpxg (:1588): dewpoint by Newton iteration from a first guess.
__device__ __forceinline__ R8 osas_ftdpxg(R8 tg, R8 pv, const OsasPhys& c)
{
    const R8 terrm = OSAS_F(1.e-6);
    const R8 tliq = c.ttp;
    const R8 tice = c.ttp - R8(20.0);
    const R8 dldtl = c.cvap - c.cliq;
    const R8 heatl = c.hvap;
    const R8 xponal = -dldtl / c.rv;
    const R8 xponbl = -dldtl / c.rv + heatl / (c.rv * c.ttp);
    const R8 dldti = c.cvap - c.csol;
    const R8 heati = c.hvap + c.hfus;
    const R8 xponai = -dldti / c.rv;
    const R8 xponbi = -dldti / c.rv + heati / (c.rv * c.ttp);
    R8 t = tg;
    for (int i = 1; i <= 100; ++i) {
        const R8 tr = c.ttp / t;
        R8 pvt, dpvt;
        if (t >= tliq) {
            pvt = c.psat * uw_pow(tr, xponal) * uw_exp(xponbl * (R8(1.0) - tr));
            const R8 el = heatl + dldtl * (t - c.ttp);
            dpvt = el * pvt / (c.rv * uw_sq(t));
        } else if (t < tice) {
            pvt = c.psat * uw_pow(tr, xponai) * uw_exp(xponbi * (R8(1.0) - tr));
            const R8 el = heati + dldti * (t - c.ttp);
            dpvt = el * pvt / (c.rv * uw_sq(t));
        } else {
            const R8 w = (t - tice) / (tliq - tice);
            const R8 pvtl = c.psat * uw_pow(tr, xponal) * uw_exp(xponbl * (R8(1.0) - tr));
            const R8 pvti = c.psat * uw_pow(tr, xponai) * uw_exp(xponbi * (R8(1.0) - tr));
            pvt = w * pvtl + (R8(1.0) - w) * pvti;
            const R8 ell = heatl + dldtl * (t - c.ttp);
            const R8 eli = heati + dldti * (t - c.ttp);
            dpvt = (w * ell * pvtl + (R8(1.0) - w) * eli * pvti) / (c.rv * uw_sq(t));
        }
        const R8 terr = (pvt - pv) / dpvt;
        t = t - terr;
        if (uw_abs(terr) <= terrm) break;
    }
    return t;
}

// fthex (:1860): equivalent potential temperature at T and scaled P**kappa.
__device__ __forceinline__ R8 osas_fthex(R8 t, R8 pk, const OsasPhys& c)
{
    const R8 p = uw_pow(pk, c.cpor);
    const R8 tr = c.ttp / t;
    const R8 pv = c.psatb * uw_pow(tr, c.xpona) * uw_exp(c.xponb * (R8(1.0) - tr));
    const R8 pd = p - pv;
    if (pd > pv) {
        const R8 el = c.hvap + c.dldt * (t - c.ttp);
        const R8 expo = el * c.eps * pv / (c.cp * t * pd);
        return t * uw_pow(pd, -c.rocp) * uw_exp(expo);
    }
    return R8(0.0);
}

// stmaxg (:2165): moist-adiabat temperature and humidity by Newton iteration.
__device__ __forceinline__ void osas_stmaxg(R8 tg, R8 the, R8 pk, R8& tma, R8& qma,
                                            const OsasPhys& c)
{
    const R8 terrm = OSAS_F(1.e-4);
    R8 t = tg;
    const R8 p = uw_pow(pk, c.cpor);
    for (int i = 1; i <= 100; ++i) {
        const R8 tr = c.ttp / t;
        const R8 pv = c.psatb * uw_pow(tr, c.xpona) * uw_exp(c.xponb * (R8(1.0) - tr));
        const R8 pd = p - pv;
        const R8 el = c.hvap + c.dldt * (t - c.ttp);
        const R8 expo = el * c.eps * pv / (c.cp * t * pd);
        const R8 thet = t * uw_pow(pd, -c.rocp) * uw_exp(expo);
        const R8 dthet = thet / t * (R8(1.0) + expo * (c.dldt * t / el
                                                    + el * p / (c.rv * t * pd)));
        const R8 terr = (thet - the) / dthet;
        t = t - terr;
        if (uw_abs(terr) <= terrm) break;
    }
    tma = t;
    const R8 tr = c.ttp / t;
    const R8 pv = c.psatb * uw_pow(tr, c.xpona) * uw_exp(c.xponb * (R8(1.0) - tr));
    const R8 pd = p - pv;
    qma = c.eps * pv / (pd + c.eps * pv);
}

// ftlclx (:2862): LCL temperature from T and the dewpoint depression.
__device__ __forceinline__ R8 osas_ftlclx(R8 t, R8 tdpd, const OsasPhys& c)
{
    const R8 terrm = OSAS_F(1.e-4);
    const R8 tlmin = OSAS_F(180.);
    const R8 tlminx = tlmin - OSAS_F(5.);
    R8 tr = c.ttp / (t - tdpd);
    const R8 pvdew = c.psat * uw_pow(tr, c.xpona) * uw_exp(c.xponb * (R8(1.0) - tr));
    R8 tlcl = t - tdpd;
    const R8 inv_rocp = R8(1.0) / c.rocp;   // 1/con_rocp, a folded constant
    for (int i = 1; i <= 100; ++i) {
        tr = c.ttp / tlcl;
        const R8 ta = t / tlcl;
        const R8 pvlcl = c.psat * uw_pow(tr, c.xpona) * uw_exp(c.xponb * (R8(1.0) - tr))
                         * uw_pow(ta, inv_rocp);
        const R8 el = c.hvap + c.dldt * (tlcl - c.ttp);
        const R8 dpvlcl = (el / (c.rv * uw_sq(t)) + R8(1.0) / (c.rocp * tlcl)) * pvlcl;
        const R8 terr = (pvlcl - pvdew) / dpvlcl;
        tlcl = tlcl - terr;
        if (uw_abs(terr) <= terrm || tlcl < tlminx) break;
    }
    return uw_max(tlcl, tlmin);
}

// ---- the table builds (gpvs :671, gtdp :1395, gthe :1678, gtma :1918,
// gtlcl :2654).  The coefficient words are written by thread 0 of each. -----
extern "C" __global__ void osas_gpvs(double* tb)
{
    const int jx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (jx > OSAS_NXPVS) return;
    const OsasPhys c = osas_phys();
    const R8 xmin = R8(180.0), xmax = R8(330.0);
    const R8 xinc = (xmax - xmin) / uw_real(OSAS_NXPVS - 1);
    const R8 c2x = R8(1.0) / xinc;
    const R8 c1x = R8(1.0) - xmin * c2x;
    if (jx == 1) { tb[OSAS_TB_C1XPVS] = c1x.v; tb[OSAS_TB_C2XPVS] = c2x.v; }
    const R8 x = xmin + uw_real(jx - 1) * xinc;
    tb[OSAS_TB_PVS + jx - 1] = osas_fpvsx(x, c).v;
}

// Independent 32-point segments start at pinned native predecessor roots.
// The recurrence inside each segment preserves the native table words.
extern "C" __global__ void osas_gtdp(double* tb)
{
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = segment * 32 + 1;
    if (first > OSAS_NXTDP) return;
    const OsasPhys c = osas_phys();
    const R8 xmin = OSAS_F(0.5), xmax = OSAS_F(10000.5);
    const R8 xinc = (xmax - xmin) / uw_real(OSAS_NXTDP - 1);
    if (segment == 0) {
        tb[OSAS_TB_C1XTDP] = (R8(1.0) - xmin / xinc).v;
        tb[OSAS_TB_C2XTDP] = (R8(1.0) / xinc).v;
    }
    R8 t = R8(osas_tdp_seed[segment]);
    const int last = min(first + 31, OSAS_NXTDP);
    for (int jx = first; jx <= last; ++jx) {
        const R8 x = xmin + uw_real(jx - 1) * xinc;
        t = osas_ftdpxg(t, x, c);
        tb[OSAS_TB_TDP + jx - 1] = t.v;
    }
}

// ymin/ymax of gthe and gtma are 0.04_krealfp**con_rocp, 0.01_krealfp**con_rocp
// and 1.10_krealfp**con_rocp: constant expressions gfortran folds at compile
// time with MPFR, so they arrive as correctly rounded words.  They are passed
// in (osas_tables.py derives them and the table digest pins the result)
// rather than recomputed with a pow that is not guaranteed correctly rounded.
extern "C" __global__ void osas_gthe(double* tb, double ymin_v, double ymax_v)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= OSAS_NXTHE * OSAS_NYTHE) return;
    const int jx = n % OSAS_NXTHE + 1, jy = n / OSAS_NXTHE + 1;
    const OsasPhys c = osas_phys();
    const R8 xmin = c.ttp - R8(90.0), xmax = c.ttp + R8(30.0);
    const R8 ymin = R8(ymin_v), ymax = R8(ymax_v);
    const R8 xinc = (xmax - xmin) / uw_real(OSAS_NXTHE - 1);
    const R8 yinc = (ymax - ymin) / uw_real(OSAS_NYTHE - 1);
    if (n == 0) {
        tb[OSAS_TB_C1XTHE] = (R8(1.0) - xmin / xinc).v;
        tb[OSAS_TB_C2XTHE] = (R8(1.0) / xinc).v;
        tb[OSAS_TB_C1YTHE] = (R8(1.0) - ymin / yinc).v;
        tb[OSAS_TB_C2YTHE] = (R8(1.0) / yinc).v;
    }
    const R8 y = ymin + uw_real(jy - 1) * yinc;
    const R8 x = xmin + uw_real(jx - 1) * xinc;
    tb[OSAS_TB_THE + n] = osas_fthex(x, y, c).v;
}

// gtma warm-starts each point from the previous one along x, so one thread
// per y row walks that row.
extern "C" __global__ void osas_gtma(double* tb, double ymin_v, double ymax_v)
{
    const int jy = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (jy > OSAS_NYMA) return;
    const OsasPhys c = osas_phys();
    const R8 xmin = R8(200.0), xmax = R8(500.0);
    const R8 ymin = R8(ymin_v), ymax = R8(ymax_v);
    const R8 xinc = (xmax - xmin) / uw_real(OSAS_NXMA - 1);
    const R8 yinc = (ymax - ymin) / uw_real(OSAS_NYMA - 1);
    if (jy == 1) {
        tb[OSAS_TB_C1XMA] = (R8(1.0) - xmin / xinc).v;
        tb[OSAS_TB_C2XMA] = (R8(1.0) / xinc).v;
        tb[OSAS_TB_C1YMA] = (R8(1.0) - ymin / yinc).v;
        tb[OSAS_TB_C2YMA] = (R8(1.0) / yinc).v;
    }
    const R8 y = ymin + uw_real(jy - 1) * yinc;
    const R8 pk = y;
    R8 tg = xmin * y;
    for (int jx = 1; jx <= OSAS_NXMA; ++jx) {
        const R8 the = xmin + uw_real(jx - 1) * xinc;
        R8 t, q;
        osas_stmaxg(tg, the, pk, t, q, c);
        const int at = (jx - 1) + (jy - 1) * OSAS_NXMA;
        tb[OSAS_TB_TMA + at] = t.v;
        tb[OSAS_TB_QMA + at] = q.v;
        tg = t;
    }
}

extern "C" __global__ void osas_gtlcl(double* tb)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= OSAS_NXTLCL * OSAS_NYTLCL) return;
    const int jx = n % OSAS_NXTLCL + 1, jy = n / OSAS_NXTLCL + 1;
    const OsasPhys c = osas_phys();
    const R8 xmin = R8(180.0), xmax = R8(330.0);
    const R8 ymin = R8(0.0), ymax = R8(60.0);
    const R8 xinc = (xmax - xmin) / uw_real(OSAS_NXTLCL - 1);
    const R8 yinc = (ymax - ymin) / uw_real(OSAS_NYTLCL - 1);
    if (n == 0) {
        tb[OSAS_TB_C1XTLCL] = (R8(1.0) - xmin / xinc).v;
        tb[OSAS_TB_C2XTLCL] = (R8(1.0) / xinc).v;
        tb[OSAS_TB_C1YTLCL] = (R8(1.0) - ymin / yinc).v;
        tb[OSAS_TB_C2YTLCL] = (R8(1.0) / yinc).v;
    }
    const R8 y = ymin + uw_real(jy - 1) * yinc;
    const R8 x = xmin + uw_real(jx - 1) * xinc;
    tb[OSAS_TB_TLCL + n] = osas_ftlclx(x, y, c).v;
}

// ---- table lookups ------------------------------------------------------------
// fpvs (:718): linear interpolation in tbpvs.
__device__ __forceinline__ R8 osas_fpvs(R8 t, const double* __restrict__ tb)
{
    const R8 c1 = R8(tb[OSAS_TB_C1XPVS]), c2 = R8(tb[OSAS_TB_C2XPVS]);
    const R8 xj = uw_min(uw_max(c1 + c2 * t, R8(1.0)), uw_real(OSAS_NXPVS));
    const int jx = uw_int(uw_min(xj, R8(OSAS_NXPVS - 1.0)));
    const R8 lo = R8(tb[OSAS_TB_PVS + jx - 1]), hi = R8(tb[OSAS_TB_PVS + jx]);
    return lo + (xj - uw_real(jx)) * (hi - lo);
}

// ftdp (:1443)
__device__ __forceinline__ R8 osas_ftdp(R8 pv, const double* __restrict__ tb)
{
    const R8 c1 = R8(tb[OSAS_TB_C1XTDP]), c2 = R8(tb[OSAS_TB_C2XTDP]);
    const R8 xj = uw_min(uw_max(c1 + c2 * pv, R8(1.0)), uw_real(OSAS_NXTDP));
    const int jx = uw_int(uw_min(xj, R8(OSAS_NXTDP - 1.0)));
    const R8 lo = R8(tb[OSAS_TB_TDP + jx - 1]), hi = R8(tb[OSAS_TB_TDP + jx]);
    return lo + (xj - uw_real(jx)) * (hi - lo);
}

// fthe (:1734): zero outside the table's low edges.
__device__ __forceinline__ R8 osas_fthe(R8 t, R8 pk, const double* __restrict__ tb)
{
    const R8 xj = uw_min(R8(tb[OSAS_TB_C1XTHE]) + R8(tb[OSAS_TB_C2XTHE]) * t,
                         uw_real(OSAS_NXTHE));
    const R8 yj = uw_min(R8(tb[OSAS_TB_C1YTHE]) + R8(tb[OSAS_TB_C2YTHE]) * pk,
                         uw_real(OSAS_NYTHE));
    if (xj >= R8(1.0) && yj >= R8(1.0)) {
        const int jx = uw_int(uw_min(xj, R8(OSAS_NXTHE - 1.0)));
        const int jy = uw_int(uw_min(yj, R8(OSAS_NYTHE - 1.0)));
        const double* a = tb + OSAS_TB_THE;
        const int a00 = (jx - 1) + (jy - 1) * OSAS_NXTHE;
        const int a01 = (jx - 1) + jy * OSAS_NXTHE;
        const R8 fx = xj - uw_real(jx);
        const R8 ftx1 = R8(a[a00]) + fx * (R8(a[a00 + 1]) - R8(a[a00]));
        const R8 ftx2 = R8(a[a01]) + fx * (R8(a[a01 + 1]) - R8(a[a01]));
        return ftx1 + (yj - uw_real(jy)) * (ftx2 - ftx1);
    }
    return R8(0.0);
}

// stma (:1978)
__device__ __forceinline__ void osas_stma(R8 the, R8 pk, R8& tma, R8& qma,
                                          const double* __restrict__ tb)
{
    const R8 xj = uw_min(uw_max(R8(tb[OSAS_TB_C1XMA]) + R8(tb[OSAS_TB_C2XMA]) * the,
                                R8(1.0)), uw_real(OSAS_NXMA));
    const R8 yj = uw_min(uw_max(R8(tb[OSAS_TB_C1YMA]) + R8(tb[OSAS_TB_C2YMA]) * pk,
                                R8(1.0)), uw_real(OSAS_NYMA));
    const int jx = uw_int(uw_min(xj, R8(OSAS_NXMA - 1.0)));
    const int jy = uw_int(uw_min(yj, R8(OSAS_NYMA - 1.0)));
    const int a00 = (jx - 1) + (jy - 1) * OSAS_NXMA;
    const int a01 = (jx - 1) + jy * OSAS_NXMA;
    const R8 fx = xj - uw_real(jx), fy = yj - uw_real(jy);
    const double* t = tb + OSAS_TB_TMA;
    const double* q = tb + OSAS_TB_QMA;
    const R8 ftx1 = R8(t[a00]) + fx * (R8(t[a00 + 1]) - R8(t[a00]));
    const R8 ftx2 = R8(t[a01]) + fx * (R8(t[a01 + 1]) - R8(t[a01]));
    tma = ftx1 + fy * (ftx2 - ftx1);
    const R8 qx1 = R8(q[a00]) + fx * (R8(q[a00 + 1]) - R8(q[a00]));
    const R8 qx2 = R8(q[a01]) + fx * (R8(q[a01 + 1]) - R8(q[a01]));
    qma = qx1 + fy * (qx2 - qx1);
}

// ftlcl (:2707)
__device__ __forceinline__ R8 osas_ftlcl(R8 t, R8 tdpd, const double* __restrict__ tb)
{
    const R8 xj = uw_min(uw_max(R8(tb[OSAS_TB_C1XTLCL]) + R8(tb[OSAS_TB_C2XTLCL]) * t,
                                R8(1.0)), uw_real(OSAS_NXTLCL));
    const R8 yj = uw_min(uw_max(R8(tb[OSAS_TB_C1YTLCL]) + R8(tb[OSAS_TB_C2YTLCL]) * tdpd,
                                R8(1.0)), uw_real(OSAS_NYTLCL));
    const int jx = uw_int(uw_min(xj, R8(OSAS_NXTLCL - 1.0)));
    const int jy = uw_int(uw_min(yj, R8(OSAS_NYTLCL - 1.0)));
    const double* a = tb + OSAS_TB_TLCL;
    const int a00 = (jx - 1) + (jy - 1) * OSAS_NXTLCL;
    const int a01 = (jx - 1) + jy * OSAS_NXTLCL;
    const R8 fx = xj - uw_real(jx);
    const R8 ftx1 = R8(a[a00]) + fx * (R8(a[a00 + 1]) - R8(a[a00]));
    const R8 ftx2 = R8(a[a01]) + fx * (R8(a[a01 + 1]) - R8(a[a01]));
    return ftx1 + (yj - uw_real(jy)) * (ftx2 - ftx1);
}

// Fortran NINT for binary64: round half away from zero (gfortran lround).
__device__ __forceinline__ int osas_nint(R8 x) { return (int)round(x.v); }

// ---- OSASCNV (module_cu_osas.F:471-2186), one column -------------------------
// The Fortran works on IM columns at once and RETURNs when no column is
// left convecting (TOTFLG); every statement after each of those returns is
// guarded by CNVFLG or writes only scratch, so for one column the RETURN is
// "this column is done".  The oracle's 12 x 8 and 1 x 96 layouts agree word
// for word, which is that argument measured.
struct OsasColumn {
    V del, prsl, phil, dot, q1, t1, u1, v1, ql1, ql2;
};

// module_cu_osas.F:575-580, :607-609. These are the same default-REAL
// constants widened before the multiply. Keep the 15-word immutable table
// in constant memory instead of constructing two indexed binary64 arrays
// in every thread's local memory. All pressure entries are exact integers.
__device__ __constant__ float osas_acritt[15] = {
    .0633f, .0445f, .0553f, .0664f, .075f, .1082f, .1521f, .2216f,
    .3151f, .3677f, .41f, .5255f, .7663f, 1.1686f, 1.6851f
};
__device__ __forceinline__ R8 osas_pcrit(int k) {
    return uw_real(900 - 50 * k);
}
__device__ __forceinline__ R8 osas_acrit(int k) {
    return uw_widen(osas_acritt[k - 1]) * (OSAS_F(975.) - osas_pcrit(k));
}

__device__ void osas_deep(int km, R8 delt, R8 ps, const OsasColumn& col, R8 rcs,
                          R8& cldwrk, R8& rn, int& kbot, int& ktop, int& kuo,
                          R8 slimsk, R8 xkt2, int ncloud, int flags,
                          const double* __restrict__ tb, const OsasPhys& c, Ws& ws)
{
    WsMark mark(ws);
    const V del = col.del, prsl = col.prsl, phil = col.phil, dot = col.dot;
    const V q1 = col.q1, t1 = col.t1, u1 = col.u1, v1 = col.v1;
    const V ql1 = col.ql1, ql2 = col.ql2;
    V po = ws.r8(km), pfld = ws.r8(km), to = ws.r8(km), qo = ws.r8(km);
    V uo = ws.r8(km), vo = ws.r8(km), qeso = ws.r8(km), dbyo = ws.r8(km);
    V zo = ws.r8(km), sumz = ws.r8(km), sumh = ws.r8(km), heo = ws.r8(km);
    V heso = ws.r8(km), qrcd = ws.r8(km), dellah = ws.r8(km), dellaq = ws.r8(km);
    V dellau = ws.r8(km), dellav = ws.r8(km), hcko = ws.r8(km), ucko = ws.r8(km);
    V vcko = ws.r8(km), qcko = ws.r8(km), eta = ws.r8(km), etau = ws.r8(km);
    V etad = ws.r8(km), qrcdo = ws.r8(km), pwo = ws.r8(km), pwdo = ws.r8(km);

    // PHYSICAL PARAMETERS (:554-558, :586)
    const R8 g = c.g, cp = c.cp, hvap = c.hvap;
    const R8 elocp = hvap / cp;
    const R8 el2orc = hvap * hvap / (c.rv * cp);
    const R8 c0 = OSAS_F(.002);
    const R8 delta = c.fvirt;
    const R8 fact1 = (c.cvap - c.cliq) / c.rv;
    const R8 fact2 = hvap / c.rv - fact1 * c.t0c;
    const R8 tcr = OSAS_F(263.16), tf = OSAS_F(233.16);
    const R8 tcrf = R8(1.0) / (tcr - tf);
    const R8 eps = c.eps, epsm1 = c.epsm1;

    const int km1 = km - 1;
    // INITIALIZE ARRAYS (:593-605)
    rn = R8(0.0);
    kbot = km + 1;
    ktop = 0;
    kuo = 0;
    bool cnvflg = true;
    R8 dtconv = R8(3600.0);
    cldwrk = R8(0.0);
    R8 pdot = R8(0.0);
    int kt2 = 0;
    R8 qlko_ktcon = R8(0.0);
    R8 dellal = R8(0.0);
    const R8 dt2 = delt;
    const R8 dtmin = uw_max(dt2, OSAS_F(1200.));
    const R8 dtmax = uw_max(dt2, OSAS_F(3600.));
    // MODEL TUNABLE PARAMETERS (:618-662), the EM_CORE values (:636-644)
    const R8 mbdt = OSAS_F(10.);
    const R8 edtmaxl = OSAS_F(.3), edtmaxs = OSAS_F(.3);
    const R8 alphal = OSAS_F(.5), alphas = OSAS_F(.75);
    const R8 betal = OSAS_F(.05), betas = OSAS_F(.05);
    const R8 evfact = OSAS_F(0.5), evfactl = OSAS_F(0.5);
    const R8 pdpdwn = OSAS_F(0.);
    const R8 xlambu = OSAS_F(1.e-4);
    const R8 w1l = OSAS_F(-8.E-3), w2l = OSAS_F(-4.E-2), w3l = OSAS_F(-5.E-3),
             w4l = OSAS_F(-5.E-4);
    const R8 w1s = OSAS_F(-2.E-4), w2s = OSAS_F(-2.E-3), w3s = OSAS_F(-1.E-3),
             w4s = OSAS_F(-2.E-5);

    // DEFINE TOP LAYER FOR SEARCH OF THE DOWNDRAFT ORIGINATING LAYER (:675-692)
    int kbmax = km, kbm = km, kmax = km;
    const R8 tx1 = R8(1.0) / ps;
    for (int k = 1; k <= km; ++k) {
        if (prsl(k) * tx1 > OSAS_F(0.45)) kbmax = k + 1;
        if (prsl(k) * tx1 > OSAS_F(0.70)) kbm = k + 1;
        if (prsl(k) * tx1 > OSAS_F(0.04)) kmax = uw_imin(km, k + 1);
    }
    kbmax = uw_imin(kbmax, kmax);
    kbm = uw_imin(kbm, kmax);

    // (:697-761). These level-local initializations have no recurrence.
    // Keep their intermediate values in registers instead of reloading
    // them through five separate passes over the column workspace.
    for (int k = 1; k <= km; ++k) {
        zo(k) = phil(k) / g;
        if (k <= kmax) {
            const R8 pk = prsl(k) * R8(10.0);
            const R8 tk = t1(k);
            const R8 qk = uw_max(q1(k), OSAS_F(1.e-10));
            R8 qes = OSAS_F(0.01) * osas_fpvs(tk, tb);
            qes = eps * qes / (pk + epsm1 * qes);
            qes = uw_max(qes, OSAS_F(1.E-8));
            pfld(k) = pk;
            pwo(k) = R8(0.0);
            pwdo(k) = R8(0.0);
            to(k) = tk;
            qo(k) = qk;
            uo(k) = u1(k);
            vo(k) = v1(k);
            dbyo(k) = R8(0.0);
            sumz(k) = R8(0.0);
            sumh(k) = R8(0.0);
            qeso(k) = qes;
            const R8 tem = phil(k) + cp * tk;
            heo(k) = tem + hvap * qk;
            heso(k) = tem + hvap * qes;
        }
    }
    // LEVEL WITH LARGEST MOIST STATIC ENERGY (:766-780)
    R8 hmax = heo(1);
    int kb = 1;
    for (int k = 2; k <= km; ++k) {
        if (k <= kbm) {
            if (heo(k) > hmax && cnvflg) { kb = k; hmax = heo(k); }
        }
    }
    // (:788-835). Ascending k reads the untouched k+1 values in both
    // native passes. Fusing the passes preserves that dependence and each
    // arithmetic expression, including the original qeso(k+1).
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            const R8 tn = to(k + 1);
            const R8 dz = R8(0.5) * (zo(k + 1) - zo(k));
            const R8 dp = R8(0.5) * (pfld(k + 1) - pfld(k));
            const R8 es = OSAS_F(0.01) * osas_fpvs(tn, tb);
            const R8 pprime = pfld(k + 1) + epsm1 * es;
            const R8 qs = eps * es / pprime;
            const R8 dqsdp = -(qs / pprime);
            const R8 desdt = es * (fact1 / tn + fact2 / uw_sq(tn));
            const R8 dqsdt = qs * pfld(k + 1) * desdt / (es * pprime);
            const R8 gamma = el2orc * qeso(k + 1) / uw_sq(tn);
            const R8 dt = (g * dz + hvap * dqsdp * dp) / (cp * (R8(1.0) + gamma));
            const R8 dq = dqsdt * dt + dqsdp * dp;
            const R8 tk = tn + dt;
            const R8 qk = uw_max(qo(k + 1) + dq, OSAS_F(1.e-10));
            const R8 pk = R8(0.5) * (pfld(k) + pfld(k + 1));
            R8 qes = OSAS_F(0.01) * osas_fpvs(tk, tb);
            qes = eps * qes / (pk + epsm1 * qes);
            qes = uw_max(qes, OSAS_F(1.E-8));
            to(k) = tk;
            qo(k) = qk;
            po(k) = pk;
            qeso(k) = qes;
            heo(k) = R8(0.5) * g * (zo(k) + zo(k + 1)) + cp * tk + hvap * qk;
            heso(k) = R8(0.5) * g * (zo(k) + zo(k + 1)) + cp * tk + hvap * qes;
            uo(k) = R8(0.5) * (uo(k) + uo(k + 1));
            vo(k) = R8(0.5) * (vo(k) + vo(k + 1));
        }
    }
    // LOOK FOR CONVECTIVE CLOUD BASE AS THE LEVEL OF FREE CONVECTION (:854-892)
    R8 hkbo = R8(0.0), qkbo = R8(0.0), ukbo = R8(0.0), vkbo = R8(0.0);
    if (cnvflg) {
        hkbo = heo(kb);
        qkbo = qo(kb);
        ukbo = uo(kb);
        vkbo = vo(kb);
    }
    bool flg = cnvflg;
    int kbcon = kmax;
    for (int k = 1; k <= km; ++k) {
        if (k <= kbmax) {
            if (flg && k > kb) {
                const R8 hsbar = heso(k);
                if (hkbo > hsbar) { flg = false; kbcon = k; }
            }
        }
    }
    if (cnvflg) {
        const R8 pbcdif = -pfld(kbcon) + pfld(kb);
        pdot = R8(10.0) * dot(kbcon);
        if (pbcdif > OSAS_F(150.)) cnvflg = false;
        if (kbcon == kmax) cnvflg = false;
    }
    if (!cnvflg) return;

    // ENTRAINMENT RATE BETWEEN KB AND KBCON (:901-918)
    R8 xlamb = R8(0.0);
    {
        R8 alpha = alphas;
        if (slimsk == R8(1.0)) alpha = alphal;
        R8 dz;
        if (kb == 1) {
            dz = R8(0.5) * (zo(kbcon) + zo(kbcon - 1)) - zo(1);
        } else {
            dz = R8(0.5) * (zo(kbcon) + zo(kbcon - 1))
                 - R8(0.5) * (zo(kb) + zo(kb - 1));
        }
        if (kbcon != kb) xlamb = -(uw_log(alpha) / dz);
        else xlamb = R8(0.0);
    }
    // UPDRAFT MASS FLUX (:920-945)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) { eta(k) = R8(1.0); etau(k) = R8(1.0); }
    }
    for (int k = km1; k >= 2; --k) {
        if (k <= kbmax) {
            if (k < kbcon && k >= kb) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                eta(k) = eta(k + 1) * uw_exp(-xlamb * dz);
                etau(k) = eta(k);
            }
        }
    }
    if (kb == 1 && kbcon > 1) {
        const R8 dz = R8(0.5) * (zo(2) - zo(1));
        eta(1) = eta(2) * uw_exp(-xlamb * dz);
        etau(1) = eta(1);
    }
    // UPDRAFT CLOUD PROPERTIES (:949-958)
    hcko(kb) = hkbo;
    qcko(kb) = qkbo;
    ucko(kb) = ukbo;
    vcko(kb) = vkbo;
    R8 pwavo = R8(0.0);
    // BELOW CLOUD BASE, MODIFIED BY ENTRAINMENT (:962-984)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k > kb && k <= kbcon) {
                const R8 factor = eta(k - 1) / eta(k);
                const R8 onemf = R8(1.0) - factor;
                hcko(k) = factor * hcko(k - 1) + onemf * R8(0.5) * (heo(k) + heo(k + 1));
                ucko(k) = factor * ucko(k - 1) + onemf * R8(0.5) * (uo(k) + uo(k + 1));
                vcko(k) = factor * vcko(k - 1) + onemf * R8(0.5) * (vo(k) + vo(k + 1));
                dbyo(k) = hcko(k) - heso(k);
            }
            if (k > kbcon) {
                hcko(k) = hcko(k - 1);
                ucko(k) = ucko(k - 1);
                vcko(k) = vcko(k - 1);
                dbyo(k) = hcko(k) - heso(k);
            }
        }
    }
    // DETERMINE CLOUD TOP (:986-1015)
    flg = true;
    int ktcon = 1;
    for (int k = 2; k <= km; ++k) {
        if (k <= kmax) {
            if (dbyo(k) < R8(0.0) && flg && k > kbcon) { ktcon = k; flg = false; }
        }
    }
    if ((pfld(kbcon) - pfld(ktcon)) < OSAS_F(150.)) cnvflg = false;
    if (!cnvflg) return;

    // DOWNDRAFT ORIGINATING LEVEL ABOVE THETA-E MINIMUM (:1019-1041)
    R8 hmin = heo(kbcon);
    int lmin = kbmax;
    int jmin = kbmax;
    for (int k = kbcon; k <= kbmax; ++k) {
        if (heo(k) < hmin && cnvflg) { lmin = k + 1; hmin = heo(k); }
    }
    jmin = uw_imin(lmin, ktcon - 1);
    const R8 xmbmax = OSAS_F(.1);
    jmin = uw_imax(jmin, kbcon + 1);
    // ENTRAINING CLOUD (:1045-1055)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k > jmin && k <= ktcon) {
                sumz(k) = sumz(k - 1) + R8(0.5) * (zo(k + 1) - zo(k - 1));
                sumh(k) = sumh(k - 1) + R8(0.5) * (zo(k + 1) - zo(k - 1)) * heo(k);
            }
        }
    }
    // (:1057-1077)
    {
        kt2 = osas_nint(xkt2 * uw_widen((float)(ktcon - jmin)) - OSAS_F(.5)) + jmin + 1;
        const R8 tem1 = hcko(jmin) - heso(kt2);
        const R8 tem2 = sumz(kt2) * heso(kt2) - sumh(kt2);
        if (uw_abs(tem2) > OSAS_F(0.000001)) xlamb = tem1 / tem2;
        else cnvflg = false;
        xlamb = uw_max(xlamb, R8(0.0));
        xlamb = uw_min(xlamb, OSAS_F(2.3) / sumz(kt2));
    }
    // (:1079-1094)
    bool dwnflg = cnvflg, dwnflg2 = cnvflg;
    if (cnvflg) {
        if (kt2 >= ktcon) dwnflg = false;
        if (xlamb <= OSAS_F(1.e-30) || hcko(jmin) - heso(kt2) <= OSAS_F(1.e-30))
            dwnflg = false;
        for (int k = jmin; k <= kt2; ++k) {
            if (dwnflg && heo(k) > heso(kt2)) dwnflg = false;
        }
        if (cnvflg && (pfld(kbcon) - pfld(ktcon)) < pdpdwn) dwnflg2 = false;
    }
    // A column that lost CNVFLG at the XLAMB test is done: everything below is
    // guarded by CNVFLG or writes scratch only.
    if (!cnvflg) return;
    // (:1096-1121)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (dwnflg && k > jmin && k <= kt2) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                eta(k) = eta(k - 1) * (R8(1.0) + xlamb * dz);
                etau(k) = etau(k - 1) * (R8(1.0) + (xlamb + xlambu) * dz);
            }
        }
    }
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (!dwnflg && k > jmin && k <= ktcon) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                etau(k) = etau(k - 1) * (R8(1.0) + xlambu * dz);
            }
        }
    }
    if (dwnflg) ktcon = kt2;   // (:1134-1138)
    // ABOVE CLOUD BASE, MODIFIED BY DETRAINMENT (:1142-1162)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k > kbcon && k <= ktcon) {
                const R8 factor = eta(k - 1) / eta(k);
                const R8 onemf = R8(1.0) - factor;
                const R8 fuv = etau(k - 1) / etau(k);
                const R8 onemfu = R8(1.0) - fuv;
                hcko(k) = factor * hcko(k - 1) + onemf * R8(0.5) * (heo(k) + heo(k + 1));
                ucko(k) = fuv * ucko(k - 1) + onemfu * R8(0.5) * (uo(k) + uo(k + 1));
                vcko(k) = fuv * vcko(k - 1) + onemfu * R8(0.5) * (vo(k) + vo(k + 1));
                dbyo(k) = hcko(k) - heso(k);
            }
        }
    }
    // (:1167-1180)
    if (dwnflg2 && jmin <= kbcon) {
        cnvflg = false;
        dwnflg = false;
        dwnflg2 = false;
    }
    if (!cnvflg) return;

    // CLOUD MOISTURE PROPERTY AND PRECIPITATION (:1185-1219); RHBAR, which
    // nothing reads, is not formed.
    R8 aa1 = R8(0.0);
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k > kb && k < ktcon) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                const R8 dz1 = zo(k) - zo(k - 1);
                const R8 gamma = el2orc * qeso(k) / uw_sq(to(k));
                const R8 qrch = qeso(k) + gamma * dbyo(k) / (hvap * (R8(1.0) + gamma));
                const R8 factor = eta(k - 1) / eta(k);
                const R8 onemf = R8(1.0) - factor;
                qcko(k) = factor * qcko(k - 1) + onemf * R8(0.5) * (qo(k) + qo(k + 1));
                const R8 dq = eta(k) * qcko(k) - eta(k) * qrch;
                if (dq > R8(0.0)) {
                    const R8 etah = R8(0.5) * (eta(k) + eta(k - 1));
                    const R8 qlk = dq / (eta(k) + etah * c0 * dz);
                    aa1 = aa1 - dz1 * g * qlk;
                    const R8 qc = qlk + qrch;
                    pwo(k) = etah * c0 * dz * qlk;
                    qcko(k) = qc;
                    pwavo = pwavo + pwo(k);
                }
            }
        }
    }
    // LIQUID AND VAPOUR SEPARATION AT CLOUD TOP (:1226-1246)
    if (ncloud > 0) {
        const int k = ktcon;
        const R8 gamma = el2orc * qeso(k) / uw_sq(to(k));
        const R8 qrch = qeso(k) + gamma * dbyo(k) / (hvap * (R8(1.0) + gamma));
        const R8 dq = qcko(k - 1) - qrch;
        if (dq > R8(0.0)) {
            qlko_ktcon = dq;
            qcko(k - 1) = qrch;
        }
    }
    // CLOUD WORK FUNCTION AT T+DT (:1250-1281)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k > kbcon && k <= ktcon) {
                const R8 dz1 = zo(k) - zo(k - 1);
                const R8 gamma = el2orc * qeso(k - 1) / uw_sq(to(k - 1));
                const R8 rfact = R8(1.0) + delta * cp * gamma * to(k - 1) / hvap;
                aa1 = aa1 + dz1 * (g / (cp * to(k - 1))) * dbyo(k - 1)
                            / (R8(1.0) + gamma) * rfact;
                aa1 = aa1 + dz1 * g * delta * uw_max(R8(0.0), qeso(k - 1) - qo(k - 1));
            }
        }
    }
    if (aa1 <= R8(0.0)) {
        dwnflg = false;
        dwnflg2 = false;
        cnvflg = false;
    }
    if (!cnvflg) return;

    // DOWNDRAFT STRENGTH IN TERMS OF WINDSHEAR (:1292-1326)
    R8 vshear = R8(0.0);
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k >= kb && k <= ktcon) {
                const R8 shear = rcs * uw_sqrt(uw_sq(uo(k + 1) - uo(k))
                                               + uw_sq(vo(k + 1) - vo(k)));
                vshear = vshear + shear;
            }
        }
    }
    vshear = R8(1.e3) * vshear / (zo(ktcon) - zo(kb));
    const R8 e1 = OSAS_F(1.591) - OSAS_F(.639) * vshear
                  + OSAS_F(.0953) * uw_sq(vshear) - OSAS_F(.00496) * uw_powi(vshear, 3);
    R8 edt = R8(1.0) - e1;
    edt = uw_min(edt, OSAS_F(.9));
    edt = uw_max(edt, OSAS_F(.0));
    R8 edto = edt;
    R8 edtx = edt;
    // DETRAINMENT RATE BETWEEN 1 AND KBDTR (:1328-1342)
    int kbdtr = kbcon;
    R8 xlamd = R8(0.0);
    {
        R8 beta = betas;
        if (slimsk == R8(1.0)) beta = betal;
        kbdtr = uw_imax(kbdtr, 1);
        if (kbdtr > 1) {
            const R8 dz = R8(0.5) * zo(kbdtr) + R8(0.5) * zo(kbdtr - 1) - zo(1);
            xlamd = uw_log(beta) / dz;
        }
    }
    // DOWNDRAFT MASS FLUX (:1344-1370)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            etad(k) = R8(1.0);
            qrcdo(k) = R8(0.0);
        }
    }
    for (int k = km1; k >= 2; --k) {
        if (k <= kbmax) {
            if (k < kbdtr) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                etad(k) = etad(k + 1) * uw_exp(xlamd * dz);
            }
        }
    }
    if (kbdtr > 1) {
        const R8 dz = R8(0.5) * (zo(2) - zo(1));
        etad(1) = etad(2) * uw_exp(xlamd * dz);
    }
    // DOWNDRAFT MOISTURE PROPERTIES (:1374-1407)
    R8 pwevo = R8(0.0);
    const int jmn = jmin;
    R8 hcdo = heo(jmn);
    R8 qcdo = qo(jmn);
    qrcdo(jmn) = qeso(jmn);
    const R8 ucdo = uo(jmn);
    const R8 vcdo = vo(jmn);
    for (int k = km1; k >= 1; --k) {
        if (k <= kmax - 1) {
            if (k < jmin) {
                const R8 dq = qeso(k);
                const R8 dt = to(k);
                const R8 gamma = el2orc * dq / uw_sq(dt);
                const R8 dh = hcdo - heso(k);
                qrcdo(k) = dq + (R8(1.0) / hvap) * (gamma / (R8(1.0) + gamma)) * dh;
                const R8 detad = etad(k + 1) - etad(k);
                pwdo(k) = etad(k + 1) * qcdo - etad(k) * qrcdo(k);
                pwdo(k) = pwdo(k) - detad * R8(0.5) * (qrcdo(k) + qrcdo(k + 1));
                qcdo = qrcdo(k);
                pwevo = pwevo + pwdo(k);
            }
        }
    }
    // FINAL DOWNDRAFT STRENGTH (:1416-1429)
    {
        R8 edtmax = edtmaxl;
        if (slimsk == R8(0.0)) edtmax = edtmaxs;
        if (dwnflg2) {
            if (pwevo < R8(0.0)) {
                edto = -(edto * pwavo / pwevo);
                edto = uw_min(edto, edtmax);
            } else {
                edto = R8(0.0);
            }
        } else {
            edto = R8(0.0);
        }
    }
    // DOWNDRAFT CLOUDWORK FUNCTIONS (:1435-1468)
    for (int k = km1; k >= 1; --k) {
        if (k <= kmax - 1) {
            if (dwnflg2 && k < jmin) {
                const R8 gamma = el2orc * qeso(k + 1) / uw_sq(to(k + 1));
                const R8 dhh = hcdo;
                const R8 dt = to(k + 1);
                const R8 dg = gamma;
                const R8 dh = heso(k + 1);
                const R8 dz = R8(-1.0) * (zo(k + 1) - zo(k));
                aa1 = aa1 + edto * dz * (g / (cp * dt)) * ((dhh - dh) / (R8(1.0) + dg))
                            * (R8(1.0) + delta * cp * dg * dt / hvap);
                aa1 = aa1 + edto * dz * g * delta
                            * uw_max(R8(0.0), qeso(k + 1) - qo(k + 1));
            }
        }
    }
    if (aa1 <= R8(0.0)) {
        cnvflg = false;
        dwnflg = false;
        dwnflg2 = false;
    }
    if (!cnvflg) return;

    // CHANGE A CLOUD WITH UNIT MASS FLUX WOULD DO (:1475-1497)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            dellah(k) = R8(0.0);
            dellaq(k) = R8(0.0);
            dellau(k) = R8(0.0);
            dellav(k) = R8(0.0);
        }
    }
    {
        const R8 dp = R8(1000.0) * del(1);
        dellah(1) = edto * etad(1) * (hcdo - heo(1)) * g / dp;
        dellaq(1) = edto * etad(1) * (qcdo - qo(1)) * g / dp;
        dellau(1) = edto * etad(1) * (ucdo - uo(1)) * g / dp;
        dellav(1) = edto * etad(1) * (vcdo - vo(1)) * g / dp;
    }
    // CHANGED DUE TO SUBSIDENCE AND ENTRAINMENT (:1501-1550)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k < ktcon) {
                R8 aup = R8(1.0);
                if (k <= kb) aup = R8(0.0);
                R8 adw = R8(1.0);
                if (k > jmin) adw = R8(0.0);
                const R8 dv1 = heo(k);
                const R8 dv2 = R8(0.5) * (heo(k) + heo(k + 1));
                const R8 dv3 = heo(k - 1);
                const R8 dv1q = qo(k);
                const R8 dv2q = R8(0.5) * (qo(k) + qo(k + 1));
                const R8 dv3q = qo(k - 1);
                const R8 dv1u = uo(k);
                const R8 dv2u = R8(0.5) * (uo(k) + uo(k + 1));
                const R8 dv3u = uo(k - 1);
                const R8 dv1v = vo(k);
                const R8 dv2v = R8(0.5) * (vo(k) + vo(k + 1));
                const R8 dv3v = vo(k - 1);
                const R8 dp = R8(1000.0) * del(k);
                const R8 deta = eta(k) - eta(k - 1);
                const R8 detad = etad(k) - etad(k - 1);
                const R8 a = aup * eta(k) - adw * edto * etad(k);
                const R8 b = aup * eta(k - 1) - adw * edto * etad(k - 1);
                dellah(k) = dellah(k)
                    + (a * dv1 - b * dv3 - aup * deta * dv2
                       + adw * edto * detad * hcdo) * g / dp;
                dellaq(k) = dellaq(k)
                    + (a * dv1q - b * dv3q - aup * deta * dv2q
                       + adw * edto * detad * R8(0.5) * (qrcdo(k) + qrcdo(k - 1))) * g / dp;
                dellau(k) = dellau(k)
                    + (a * dv1u - b * dv3u - aup * deta * dv2u
                       + adw * edto * detad * ucdo) * g / dp;
                dellav(k) = dellav(k)
                    + (a * dv1v - b * dv3v - aup * deta * dv2v
                       + adw * edto * detad * vcdo) * g / dp;
            }
        }
    }
    // CLOUD TOP (:1554-1575)
    {
        const int indx = ktcon;
        const R8 dp = R8(1000.0) * del(indx);
        const R8 dv1 = heo(indx - 1);
        dellah(indx) = eta(indx - 1) * (hcko(indx - 1) - dv1) * g / dp;
        const R8 dvq1 = qo(indx - 1);
        dellaq(indx) = eta(indx - 1) * (qcko(indx - 1) - dvq1) * g / dp;
        const R8 dv1u = uo(indx - 1);
        dellau(indx) = eta(indx - 1) * (ucko(indx - 1) - dv1u) * g / dp;
        const R8 dv1v = vo(indx - 1);
        dellav(indx) = eta(indx - 1) * (vcko(indx - 1) - dv1v) * g / dp;
        dellal = eta(indx - 1) * qlko_ktcon * g / dp;
    }
    // FINAL CHANGED VARIABLE PER UNIT MASS FLUX (:1579-1598)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k > ktcon) {
                qo(k) = q1(k);
                to(k) = t1(k);
                uo(k) = u1(k);
                vo(k) = v1(k);
            }
            if (k <= ktcon) {
                qo(k) = dellaq(k) * mbdt + q1(k);
                const R8 dellat = (dellah(k) - hvap * dellaq(k)) / cp;
                to(k) = dellat * mbdt + t1(k);
                qo(k) = uw_max(qo(k), OSAS_F(1.e-10));
            }
        }
    }
    // ENVIRONMENT AGAIN (:1610-1624)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            qeso(k) = OSAS_F(0.01) * osas_fpvs(to(k), tb);
            qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
            qeso(k) = uw_max(qeso(k), OSAS_F(1.E-8));
        }
    }
    R8 xaa0 = R8(0.0);
    R8 xpwav = R8(0.0);
    // MOIST STATIC ENERGY (:1652-1704)
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            const R8 dz = R8(0.5) * (zo(k + 1) - zo(k));
            const R8 dp = R8(0.5) * (pfld(k + 1) - pfld(k));
            const R8 es = OSAS_F(0.01) * osas_fpvs(to(k + 1), tb);
            const R8 pprime = pfld(k + 1) + epsm1 * es;
            const R8 qs = eps * es / pprime;
            const R8 dqsdp = -(qs / pprime);
            const R8 desdt = es * (fact1 / to(k + 1) + fact2 / uw_sq(to(k + 1)));
            const R8 dqsdt = qs * pfld(k + 1) * desdt / (es * pprime);
            const R8 gamma = el2orc * qeso(k + 1) / uw_sq(to(k + 1));
            const R8 dt = (g * dz + hvap * dqsdp * dp) / (cp * (R8(1.0) + gamma));
            const R8 dq = dqsdt * dt + dqsdp * dp;
            to(k) = to(k + 1) + dt;
            qo(k) = qo(k + 1) + dq;
            po(k) = R8(0.5) * (pfld(k) + pfld(k + 1));
        }
    }
    for (int k = 1; k <= km1; ++k) {
        if (k <= kmax - 1) {
            qeso(k) = OSAS_F(0.01) * osas_fpvs(to(k), tb);
            qeso(k) = eps * qeso(k) / (po(k) + epsm1 * qeso(k));
            qeso(k) = uw_max(qeso(k), OSAS_F(1.E-8));
            qo(k) = uw_max(qo(k), OSAS_F(1.e-10));
            heo(k) = R8(0.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qo(k);
            heso(k) = R8(0.5) * g * (zo(k) + zo(k + 1)) + cp * to(k) + hvap * qeso(k);
        }
    }
    {
        const int k = kmax;
        heo(k) = g * zo(k) + cp * to(k) + hvap * qo(k);
        heso(k) = g * zo(k) + cp * to(k) + hvap * qeso(k);
    }
    {
        const R8 xhkb = heo(kb);
        const R8 xqkb = qo(kb);
        hcko(kb) = xhkb;
        qcko(kb) = xqkb;
    }
    // STATIC CONTROL: MOISTURE AND CLOUD WORK FUNCTIONS (:1721-1783)
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k > kb && k <= ktcon) {
                const R8 factor = eta(k - 1) / eta(k);
                const R8 onemf = R8(1.0) - factor;
                hcko(k) = factor * hcko(k - 1) + onemf * R8(0.5) * (heo(k) + heo(k + 1));
            }
        }
    }
    for (int k = 2; k <= km1; ++k) {
        if (k <= kmax - 1) {
            if (k > kb && k < ktcon) {
                const R8 dz = R8(0.5) * (zo(k + 1) - zo(k - 1));
                const R8 gamma = el2orc * qeso(k) / uw_sq(to(k));
                R8 xdby = hcko(k) - heso(k);
                xdby = uw_max(xdby, R8(0.0));
                const R8 xqrch = qeso(k) + gamma * xdby / (hvap * (R8(1.0) + gamma));
                const R8 factor = eta(k - 1) / eta(k);
                const R8 onemf = R8(1.0) - factor;
                qcko(k) = factor * qcko(k - 1) + onemf * R8(0.5) * (qo(k) + qo(k + 1));
                const R8 dq = eta(k) * qcko(k) - eta(k) * xqrch;
                if (dq > R8(0.0)) {
                    const R8 etah = R8(0.5) * (eta(k) + eta(k - 1));
                    const R8 qlk = dq / (eta(k) + etah * c0 * dz);
                    xaa0 = xaa0 - (zo(k) - zo(k - 1)) * g * qlk;
                    const R8 xqc = qlk + xqrch;
                    const R8 xpw = etah * c0 * dz * qlk;
                    qcko(k) = xqc;
                    xpwav = xpwav + xpw;
                }
            }
            if (k > kbcon && k <= ktcon) {
                const R8 dz1 = zo(k) - zo(k - 1);
                const R8 gamma = el2orc * qeso(k - 1) / uw_sq(to(k - 1));
                const R8 rfact = R8(1.0) + delta * cp * gamma * to(k - 1) / hvap;
                const R8 xdby = hcko(k - 1) - heso(k - 1);
                xaa0 = xaa0 + dz1 * (g / (cp * to(k - 1))) * xdby
                              / (R8(1.0) + gamma) * rfact;
                xaa0 = xaa0 + dz1 * g * delta * uw_max(R8(0.0), qeso(k - 1) - qo(k - 1));
            }
        }
    }
    // DOWNDRAFT MOISTURE PROPERTIES (:1793-1822)
    R8 xpwev = R8(0.0);
    R8 xhcd = R8(0.0);
    if (dwnflg2) {
        xhcd = heo(jmn);
        qrcd(jmn) = qeso(jmn);
    }
    for (int k = km1; k >= 1; --k) {
        if (k <= kmax - 1) {
            if (dwnflg2 && k < jmin) {
                const R8 dq = qeso(k);
                const R8 dt = to(k);
                const R8 gamma = el2orc * dq / uw_sq(dt);
                const R8 dh = xhcd - heso(k);
                qrcd(k) = dq + (R8(1.0) / hvap) * (gamma / (R8(1.0) + gamma)) * dh;
                const R8 detad = etad(k + 1) - etad(k);
                R8 xpwd = etad(k + 1) * qrcd(k + 1) - etad(k) * qrcd(k);
                xpwd = xpwd - detad * R8(0.5) * (qrcd(k) + qrcd(k + 1));
                xpwev = xpwev + xpwd;
            }
        }
    }
    // (:1824-1837)
    {
        R8 edtmax = edtmaxl;
        if (slimsk == R8(0.0)) edtmax = edtmaxs;
        if (dwnflg2) {
            if (xpwev >= R8(0.0)) {
                edtx = R8(0.0);
            } else {
                edtx = -(edtx * xpwav / xpwev);
                edtx = uw_min(edtx, edtmax);
            }
        } else {
            edtx = R8(0.0);
        }
    }
    // DOWNDRAFT CLOUDWORK FUNCTIONS (:1844-1863)
    for (int k = km1; k >= 1; --k) {
        if (k <= kmax - 1) {
            if (dwnflg2 && k < jmin) {
                const R8 gamma = el2orc * qeso(k + 1) / uw_sq(to(k + 1));
                const R8 dhh = xhcd;
                const R8 dt = to(k + 1);
                const R8 dg = gamma;
                const R8 dh = heso(k + 1);
                const R8 dz = R8(-1.0) * (zo(k + 1) - zo(k));
                xaa0 = xaa0 + edtx * dz * (g / (cp * dt)) * ((dhh - dh) / (R8(1.0) + dg))
                              * (R8(1.0) + delta * cp * dg * dt / hvap);
                xaa0 = xaa0 + edtx * dz * g * delta
                              * uw_max(R8(0.0), qeso(k + 1) - qo(k + 1));
            }
        }
    }
    // CRITICAL CLOUD WORK FUNCTION (:1870-1890)
    R8 acrt = R8(0.0);
    {
        const R8 ptop = pfld(ktcon);
        if (ptop < osas_pcrit(15)) {
            acrt = osas_acrit(15) * (OSAS_F(975.) - ptop) / (OSAS_F(975.) - osas_pcrit(15));
        } else if (ptop > osas_pcrit(1)) {
            acrt = osas_acrit(1);
        } else {
            int k = uw_int((OSAS_F(850.) - ptop) / OSAS_F(50.)) + 2;
            k = uw_imin(k, 15);
            k = uw_imax(k, 2);
            const R8 acrit_k = osas_acrit(k), acrit_prev = osas_acrit(k - 1);
            const R8 pcrit_k = osas_pcrit(k), pcrit_prev = osas_pcrit(k - 1);
            acrt = acrit_k + (acrit_prev - acrit_k) * (ptop - pcrit_k)
                              / (pcrit_prev - pcrit_k);
        }
    }
    // (:1891-1941)
    R8 acrtfct = R8(1.0);
    {
        R8 w1, w2, w3, w4;
        if (slimsk == R8(1.0)) { w1 = w1l; w2 = w2l; w3 = w3l; w4 = w4l; }
        else { w1 = w1s; w2 = w2s; w3 = w3s; w4 = w4s; }
        if (pdot <= w4) {
            acrtfct = (pdot - w4) / (w3 - w4);
        } else if (pdot >= -w4) {
            acrtfct = -((pdot + w4) / (w4 - w3));
        } else {
            acrtfct = R8(0.0);
        }
        acrtfct = uw_max(acrtfct, OSAS_F(-1.));
        acrtfct = uw_min(acrtfct, OSAS_F(1.));
        acrtfct = R8(1.0) - acrtfct;
        dtconv = dt2 + uw_max(OSAS_F(1800.) - dt2, R8(0.0)) * (pdot - w2) / (w1 - w2);
        dtconv = uw_max(dtconv, dtmin);
        dtconv = uw_min(dtconv, dtmax);
    }
    // LARGE SCALE FORCING, CLOUD BASE MASS FLUX (:1945-1976)
    R8 xmb = R8(0.0);
    {
        flg = cnvflg;
        const R8 fld = (aa1 - acrt * acrtfct) / dtconv;
        if (fld <= R8(0.0)) flg = false;
        cnvflg = flg;
        R8 xk = R8(0.0);
        if (cnvflg) {
            xk = (xaa0 - aa1) / mbdt;
            if (xk >= R8(0.0)) flg = false;
        }
        cnvflg = flg;
        if (cnvflg) {
            xmb = -(fld / xk);
            xmb = uw_min(xmb, xmbmax);
        }
    }
    if (!cnvflg) return;

    // RESTORE TO, QO TO T1, Q1 IN CASE CONVECTION STOPS (:1980-1995)
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            to(k) = t1(k);
            qo(k) = q1(k);
            qeso(k) = OSAS_F(0.01) * osas_fpvs(t1(k), tb);
            qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
            qeso(k) = uw_max(qeso(k), OSAS_F(1.E-8));
        }
    }
    // FEEDBACK (:2008-2028); DELHBAR/DELQBAR/DELTBAR are diagnostics nothing
    // reads and are not formed.
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k <= ktcon) {
                const R8 dellat = (dellah(k) - hvap * dellaq(k)) / cp;
                t1(k) = t1(k) + dellat * xmb * dt2;
                q1(k) = q1(k) + dellaq(k) * xmb * dt2;
                u1(k) = u1(k) + dellau(k) * xmb * dt2;
                v1(k) = v1(k) + dellav(k) * xmb * dt2;
            }
        }
    }
    // (:2029-2062), and the cloud-top QL as it stood before the detrainment,
    // for the shipped identity's back-out below.
    R8 ql1_before = R8(0.0), ql2_before = R8(0.0);
    for (int k = 1; k <= km; ++k) {
        if (k <= kmax) {
            if (k <= ktcon) {
                qeso(k) = OSAS_F(0.01) * osas_fpvs(t1(k), tb);
                qeso(k) = eps * qeso(k) / (pfld(k) + epsm1 * qeso(k));
                qeso(k) = uw_max(qeso(k), OSAS_F(1.E-8));
                if (ncloud > 0 && k == ktcon) {
                    ql1_before = ql1(k);
                    ql2_before = ql2(k);
                    const R8 tem = dellal * xmb * dt2;
                    const R8 tem1 = uw_max(R8(0.0), uw_min(R8(1.0), (tcr - t1(k)) * tcrf));
                    if (ql2(k) > OSAS_F(-999.0)) {
                        ql1(k) = ql1(k) + tem * tem1;
                        ql2(k) = ql2(k) + tem * (R8(1.0) - tem1);
                    } else {
                        const R8 tem2 = ql1(k) + tem;
                        ql1(k) = tem2 * tem1;
                        ql2(k) = tem2 - ql1(k);
                    }
                }
            }
        }
    }
    // (:2074-2093)
    R8 rntot = R8(0.0), delqev = R8(0.0), delq2 = R8(0.0);
    flg = cnvflg;
    for (int k = km; k >= 1; --k) {
        if (k <= kmax) {
            if (k <= ktcon) {
                R8 aup = R8(1.0);
                if (k <= kb) aup = R8(0.0);
                R8 adw = R8(1.0);
                if (k > jmin) adw = R8(0.0);
                const R8 rain = aup * pwo(k) + adw * edto * pwdo(k);
                rntot = rntot + rain * xmb * OSAS_F(.001) * dt2;
            }
        }
    }
    // RAIN AND ITS EVAPORATION, TOP DOWN (:2094-2140)
    for (int k = km; k >= 1; --k) {
        if (k <= kmax) {
            R8 qevap = R8(0.0);
            if (k <= ktcon) {
                R8 aup = R8(1.0);
                if (k <= kb) aup = R8(0.0);
                R8 adw = R8(1.0);
                if (k > jmin) adw = R8(0.0);
                const R8 rain = aup * pwo(k) + adw * edto * pwdo(k);
                rn = rn + rain * xmb * OSAS_F(.001) * dt2;
            }
            if (flg && k <= ktcon) {
                R8 evef = edt * evfact;
                if (slimsk == R8(1.0)) evef = edt * evfactl;
                const R8 qcond = evef * (q1(k) - qeso(k))
                                 / (R8(1.0) + el2orc * qeso(k) / uw_sq(t1(k)));
                const R8 dp = R8(1000.0) * del(k);
                if (rn > R8(0.0) && qcond < R8(0.0)) {
                    qevap = -(qcond * (R8(1.0) - uw_exp(-(OSAS_F(.32)
                                                          * uw_sqrt(dt2 * rn)))));
                    qevap = uw_min(qevap, rn * R8(1000.0) * g / dp);
                    delq2 = delqev + OSAS_F(.001) * qevap * dp / g;
                }
                if (rn > R8(0.0) && qcond < R8(0.0) && delq2 > rntot) {
                    qevap = R8(1000.0) * g * (rntot - delqev) / dp;
                    flg = false;
                }
                if (rn > R8(0.0) && qevap > R8(0.0)) {
                    q1(k) = q1(k) + qevap;
                    t1(k) = t1(k) - elocp * qevap;
                    rn = rn - OSAS_F(.001) * qevap * dp / g;
                    delqev = delqev + OSAS_F(.001) * dp * qevap / g;
                }
            }
        }
    }
    // PRECIPITATION RATE CONVERTED TO ACTUAL PRECIP (:2156-2173)
    if (rn < R8(0.0) && !flg) rn = R8(0.0);
    if (rn <= R8(0.0)) {
        rn = R8(0.0);
    } else {
        ktop = ktcon;
        kbot = kbcon;
        kuo = 1;
        cldwrk = aa1;
    }
    // BACK OUT THE HEATING AND THE MOISTENING (:2174-2183)
    if (rn <= R8(0.0)) {
        for (int k = 1; k <= km; ++k) {
            if (k <= kmax) {
                t1(k) = to(k);
                q1(k) = qo(k);
            }
        }
        if (!(flags & OSAS_FLAG_WRF_QL_BACKOUT) && ncloud > 0 && ktcon <= kmax) {
            ql1(ktcon) = ql1_before;
            ql2(ktcon) = ql2_before;
        }
    }
}

// ---- MSTADBT3 (:2409-2489) for one column, as SHALCV calls it (K1 = K2 = 2);
// only KLCL and KTOP reach SHALCV.  TCLD/QCLD are not formed: within the
// column's own KBOT..KTOP window SHALCV overwrites every AU it reads.
__device__ void osas_mstadbt3(int km, const V prsl, const V prslk, const V tenv,
                              const V qenv, int& klcl, int& ktop,
                              const double* __restrict__ tb, const OsasPhys& c)
{
    const int k1 = 2, k2 = 2;
    R8 slkma = R8(0.0), thema = R8(0.0);
    for (int k = k1; k <= k2; ++k) {
        const R8 pv = R8(1000.0) * prsl(k) * qenv(k) / (c.eps - c.epsm1 * qenv(k));
        const R8 tdpd = tenv(k) - osas_ftdp(pv, tb);
        R8 tlcl, slklcl;
        if (tdpd > R8(0.0)) {
            tlcl = osas_ftlcl(tenv(k), tdpd, tb);
            slklcl = prslk(k) * tlcl / tenv(k);
        } else {
            tlcl = tenv(k);
            slklcl = prslk(k);
        }
        const R8 thelcl = osas_fthe(tlcl, slklcl, tb);
        if (thelcl > thema) { slkma = slklcl; thema = thelcl; }
    }
    klcl = km + 1;
    ktop = 0;
    for (int k = k1; k <= km; ++k) {
        if (prslk(k) <= slkma) {
            klcl = uw_imin(klcl, k);
            R8 tma, qma;
            osas_stma(thema, prslk(k), tma, qma, tb);
            const R8 tvcld = tma * (R8(1.0) + c.fvirt * qma);
            const R8 tvenv = tenv(k) * (R8(1.0) + c.fvirt * qenv(k));
            if (tvcld > tvenv) ktop = uw_imax(ktop, k);
        }
    }
}

// ---- SHALCV (:2190-2365) and TRIDI2T3 (:2367-2406) for one column ------------
// The Fortran compresses the row's shallow candidates and solves one
// tridiagonal system over the union window K1..K2 of their KBOT..KTOP.
// Outside a column's own window its rows are exact identities (CK = 0 makes
// every off-diagonal a signed zero and every diagonal 1), so solving over the
// column's own window gives the same words; the oracle's layout check
// measures it.  The solve runs in place, as the Fortran's aliased call does
// (R1 = A1 = Q, R2 = A2 = T, CU = AU).
__device__ void osas_shallow(int km, R8 dt, const V del, const V prsi, const V prsl,
                             const V prslk, int kuo, V q, V t, R8 dpshc,
                             const double* __restrict__ tb, const OsasPhys& c, Ws& ws)
{
    if (kuo != 0) return;
    const R8 g = c.g, cp = c.cp, hvap = c.hvap, rd = c.rd;
    const R8 gocp = g / cp;
    bool lshc = false;
    for (int k = 1; k <= km - 1; ++k) {
        const R8 eldq = hvap * (q(k) - q(k + 1));
        const R8 cpdt = cp * (t(k) - t(k + 1));
        const R8 rtdls = (prsl(k) - prsl(k + 1)) / prsi(k + 1) * rd * R8(0.5)
                         * (t(k) + t(k + 1));
        const R8 dmse = eldq + cpdt - rtdls;
        lshc = lshc || dmse > R8(0.0);
    }
    if (!lshc) return;
    int ktopm = km;
    for (int k = 2; k <= km; ++k) {
        if (prsi(1) - prsi(k) <= dpshc) ktopm = k;
    }
    int klcl, ktop;
    osas_mstadbt3(km - 1, prsl, prslk, t, q, klcl, ktop, tb, c);
    int kbot = uw_imin(klcl - 1, ktopm - 1);
    ktop = uw_imin(ktop + 1, ktopm);
    WsMark mark(ws);
    V al = ws.r8(km), ad = ws.r8(km), au = ws.r8(km);
    lshc = false;
    for (int k = 1; k <= km - 1; ++k) {
        if (k >= kbot && k < ktop) {
            const R8 eldq = hvap * (q(k) - q(k + 1));
            const R8 cpdt = cp * (t(k) - t(k + 1));
            const R8 rtdls = (prsl(k) - prsl(k + 1)) / prsi(k + 1) * rd * R8(0.5)
                             * (t(k) + t(k + 1));
            const R8 dmse = eldq + cpdt - rtdls;
            lshc = lshc || dmse > R8(0.0);
            au(k) = g / rtdls;
        }
    }
    if (!lshc) return;
    const int k1 = kbot, k2 = ktop;
    const int n = k2 - k1 + 1;
    if (n < 2) return;
    ad(k1) = R8(1.0);
    for (int k = k1; k <= k2 - 1; ++k) {
        const R8 dtodsl = dt / del(k);
        const R8 dtodsu = dt / del(k + 1);
        const R8 dsig = prsl(k) - prsl(k + 1);
        R8 ck;
        if (k == kbot) ck = R8(1.5);
        else if (k == ktop - 1) ck = R8(1.0);
        else if (k == ktop - 2) ck = R8(3.0);
        else if (k > kbot && k < ktop - 2) ck = R8(5.0);
        else ck = R8(0.0);
        const R8 dsdz1 = ck * dsig * au(k) * gocp;
        const R8 dsdz2 = ck * dsig * au(k) * au(k);
        au(k) = -(dtodsl * dsdz2);
        al(k) = -(dtodsu * dsdz2);
        ad(k) = ad(k) - au(k);
        ad(k + 1) = R8(1.0) - al(k);
        t(k) = t(k) + dtodsl * dsdz1;
        t(k + 1) = t(k + 1) - dtodsu * dsdz1;
    }
    // TRIDI2T3 over rows k1..k2: CL(r) = AL(k1+r-2), CM(r) = AD, CU(r) = AU.
    {
        R8 fk = R8(1.0) / ad(k1);
        au(k1) = fk * au(k1);
        q(k1) = fk * q(k1);
        t(k1) = fk * t(k1);
        for (int k = k1 + 1; k <= k2 - 1; ++k) {
            fk = R8(1.0) / (ad(k) - al(k - 1) * au(k - 1));
            au(k) = fk * au(k);
            q(k) = fk * (q(k) - al(k - 1) * q(k - 1));
            t(k) = fk * (t(k) - al(k - 1) * t(k - 1));
        }
        fk = R8(1.0) / (ad(k2) - al(k2 - 1) * au(k2 - 1));
        q(k2) = fk * (q(k2) - al(k2 - 1) * q(k2 - 1));
        t(k2) = fk * (t(k2) - al(k2 - 1) * t(k2 - 1));
        for (int k = k2 - 1; k >= k1; --k) {
            q(k) = q(k) - au(k) * q(k + 1);
            t(k) = t(k) - au(k) * t(k + 1);
        }
    }
}

// ---- XKT2 for the default arm: a counter-based draw ---------------------------
// splitmix64 over (seed, step, column): reproducible, independent of the
// launch decomposition, and with no state to carry across a restart.
__device__ __forceinline__ unsigned long long osas_mix64(unsigned long long z)
{
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void cu_osas_xkt2(double* xkt2, long long ncols,
                                        unsigned long long seed, long long step,
                                        int mode, double fixed,
                                        const unsigned long long* column_indices)
{
    const long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;
    if (mode != 0) { xkt2[col] = fixed; return; }
    const unsigned long long h = osas_mix64(
        osas_mix64(seed ^ osas_mix64((unsigned long long)step))
        ^ (column_indices ? column_indices[col] : (unsigned long long)col));
    xkt2[col] = (double)(h >> 11) * 0x1.0p-53;   // uniform on [0, 1)
}

// ---- CU_OSAS (:7-394): the column driver ---------------------------------------
// Layout: every 3-D field is WRF-ordered (k = 1 at the bottom), level-major
// with the column fastest: element k of column c is base[(k-1)*ncols + c].
// w and p8w carry kx+1 levels.  The binary64 pool is slot-major with the
// chunk's thread index fastest.
#define OSAS_COLUMN_ARGS \
const float* __restrict__ t3d, const float* __restrict__ qv3d, \
const float* __restrict__ qc3d, const float* __restrict__ qi3d, \
const float* __restrict__ u3d, const float* __restrict__ v3d, \
const float* __restrict__ w, const float* __restrict__ pcps, \
const float* __restrict__ p8w, const float* __restrict__ pi3d, \
const float* __restrict__ rho3d, const float* __restrict__ dz8w, \
const float* __restrict__ xland, const double* __restrict__ xkt2, \
float* rthcuten, float* rqvcuten, float* rqccuten, float* rqicuten, \
float* raincv, float* pratec, float* htop, float* hbot, \
const double* __restrict__ tb, \
long long ncols, long long col0, int nchunk, int kx, float dt, int stepcu, \
int flags, double* r8pool, int r8cap, int* err, double* stages
#define OSAS_COLUMN_PASS t3d, qv3d, qc3d, qi3d, u3d, v3d, w, pcps, p8w, pi3d, rho3d, dz8w, xland, xkt2, rthcuten, rqvcuten, rqccuten, rqicuten, raincv, pratec, htop, hbot, tb, ncols, col0, nchunk, kx, dt, stepcu, flags, r8pool, r8cap, err, stages

template<int PHASE>
__device__ __forceinline__ void osas_columns_impl(OSAS_COLUMN_ARGS)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int active = uw_imin(nchunk, (int)(ncols - col0));
    const int tid = PHASE == 3 ? linear % active : linear;
    const int out_k = PHASE == 3 ? linear / active + 1 : 1;
    if (PHASE == 3 && out_k > kx) return;
    if (tid >= nchunk) return;
    const long long cc = col0 + tid;
    if (cc >= ncols) return;
    const long long s = ncols;
    #define LV(a, k) (a)[(long long)((k) - 1) * s + cc]
    Ws ws{r8pool + tid, nullptr, nchunk, 0, r8cap, 0, 0, err};
    const OsasPhys c = osas_phys();
    const int km = kx;
    const R8 g = c.g, rd = c.rd;

    // (:224-230)
    const int jcap = 126;
    (void)jcap;    // OSASCNV forms FJCAP from it and never reads FJCAP
    const R8 dpshc = R8(30.0);
    const float delt_f = __fmul_rn(dt, (float)stepcu);
    const R8 delt = uw_widen(delt_f);
    const R8 rdelt = R8(1.0) / delt;
    const int ncloud = 1;

    OsasColumn col;
    V zi = ws.r8(km), zl = ws.r8(km), prslk = ws.r8(km), prsi = ws.r8(km + 1);
    col.del = ws.r8(km); col.prsl = ws.r8(km); col.phil = ws.r8(km);
    col.dot = ws.r8(km); col.q1 = ws.r8(km); col.t1 = ws.r8(km);
    col.u1 = ws.r8(km); col.v1 = ws.r8(km); col.ql1 = ws.r8(km); col.ql2 = ws.r8(km);

    // (:295-332)
    const float psfc = p8w[cc];                       // P8w(i,kms,j)
    const R8 ps = uw_widen(__fmul_rn(psfc, .001f));
    const R8 rcs = R8(1.0);
    const R8 slimsk = uw_widen(fabsf(__fsub_rn(xland[cc], 2.f)));
    if constexpr (PHASE == 0 || PHASE == 1) {
        // compute zi and zl (:247-267)
        zi(1) = R8(0.0);
        for (int k = 2; k <= km; ++k) zi(k) = zi(k - 1) + uw_widen(LV(dz8w, k - 1));
        for (int k = 2; k <= km; ++k) zl(k - 1) = (zi(k) + zi(k - 1)) * R8(0.5);
        zl(km) = R8(2.0) * zi(km) - zl(km - 1);

        prsi(1) = ps;
        const R8 dotfac = OSAS_F(5.0E-4) * g;
        for (int k = 1; k <= km; ++k) {
            col.prsl(k) = uw_widen(__fmul_rn(LV(pcps, k), .001f));
            col.phil(k) = zl(k) * g;
            col.dot(k) = -(dotfac * uw_widen(LV(rho3d, k))
                           * uw_widen(__fadd_rn(LV(w, k), LV(w, k + 1))));
        }
        for (int k = 1; k <= km; ++k) {
            const float tk = LV(t3d, k);
            col.del(k) = col.prsl(k) * g / rd * uw_widen(LV(dz8w, k)) / uw_widen(tk);
            col.u1(k) = uw_widen(LV(u3d, k));
            col.v1(k) = uw_widen(LV(v3d, k));
            const float qv = LV(qv3d, k), qc = LV(qc3d, k), qi = LV(qi3d, k);
            col.q1(k) = uw_widen(__fdiv_rn(qv, __fadd_rn(1.f, qv)));
            col.t1(k) = uw_widen(tk);
            col.ql1(k) = uw_widen(__fdiv_rn(qi, __fadd_rn(1.f, qi)));
            col.ql2(k) = uw_widen(__fdiv_rn(qc, __fadd_rn(1.f, qc)));
            prslk(k) = uw_pow(col.prsl(k) * OSAS_F(.01), c.rocp);
        }
        for (int k = 2; k <= km + 1; ++k) prsi(k) = prsi(k - 1) - col.del(k - 1);

    }
    if constexpr (PHASE == 0 || PHASE == 2) {
        R8 cldwrk, rn;
        int kbot, ktop, kuo;
        osas_deep(km, delt, ps, col, rcs, cldwrk, rn, kbot, ktop, kuo, slimsk,
                  R8(xkt2[cc]), ncloud, flags, tb, c, ws);
        if (stages != nullptr) {
            // Stage record (the oracle's after-deep hook): t1, q1, ql1, ql2 per
            // level, then rn, cldwrk, kbot, ktop, kuo; then t1, q1 after SHALCV.
            double* st = stages + (size_t)cc * (size_t)(6 * km + 5);
            for (int k = 1; k <= km; ++k) {
                st[k - 1] = col.t1(k).v;
                st[km + k - 1] = col.q1(k).v;
                st[2 * km + k - 1] = col.ql1(k).v;
                st[3 * km + k - 1] = col.ql2(k).v;
            }
            st[4 * km] = rn.v;
            st[4 * km + 1] = cldwrk.v;
            st[4 * km + 2] = (double)kbot;
            st[4 * km + 3] = (double)ktop;
            st[4 * km + 4] = (double)kuo;
        }
        osas_shallow(km, delt, col.del, prsi, col.prsl, prslk, kuo, col.q1, col.t1,
                     dpshc, tb, c, ws);
        if (stages != nullptr) {
            double* st = stages + (size_t)cc * (size_t)(6 * km + 5) + 4 * km + 5;
            for (int k = 1; k <= km; ++k) {
                st[k - 1] = col.t1(k).v;
                st[km + k - 1] = col.q1(k).v;
            }
        }

        // (:345-390)
        const float stepdt = __fmul_rn((float)stepcu, dt);
        raincv[cc] = uw_narrow(rn * R8(1000.0) / uw_real(stepcu));
        pratec[cc] = uw_narrow(rn * R8(1000.0) / uw_widen(stepdt));
        hbot[cc] = (float)kbot;
        htop[cc] = (float)ktop;
    }
    if constexpr (PHASE == 0 || PHASE == 3) {
        for (int k = out_k; k <= (PHASE == 3 ? out_k : km); ++k) {
            LV(rthcuten, k) = uw_narrow((col.t1(k) - uw_widen(LV(t3d, k)))
                                        / uw_widen(LV(pi3d, k)) * rdelt);
            LV(rqvcuten, k) = uw_narrow((col.q1(k) / (R8(1.0) - col.q1(k))
                                         - uw_widen(LV(qv3d, k))) * rdelt);
            if (rqccuten != nullptr) {
                LV(rqccuten, k) = uw_narrow((col.ql2(k) / (R8(1.0) - col.ql2(k))
                                             - uw_widen(LV(qc3d, k))) * rdelt);
            }
            if (rqicuten != nullptr) {
                LV(rqicuten, k) = uw_narrow((col.ql1(k) / (R8(1.0) - col.ql1(k))
                                             - uw_widen(LV(qi3d, k))) * rdelt);
            }
        }
    }
    #undef LV
}

// PHASE 0 remains the complete single-kernel referee entry. The production
// launcher uses the same statements in three separately compiled images:
// low-register driver preparation, column physics, and level-parallel output.
extern "C" __global__ __launch_bounds__(64) void cu_osas_columns(OSAS_COLUMN_ARGS) {
    osas_columns_impl<0>(OSAS_COLUMN_PASS);
}
extern "C" __global__ __launch_bounds__(64) void cu_osas_columns_prepare(OSAS_COLUMN_ARGS) {
    osas_columns_impl<1>(OSAS_COLUMN_PASS);
}
extern "C" __global__ __launch_bounds__(64) void cu_osas_columns_physics(OSAS_COLUMN_ARGS) {
    osas_columns_impl<2>(OSAS_COLUMN_PASS);
}
extern "C" __global__ __launch_bounds__(256) void cu_osas_columns_finish(OSAS_COLUMN_ARGS) {
    osas_columns_impl<3>(OSAS_COLUMN_PASS);
}
#undef OSAS_COLUMN_ARGS
#undef OSAS_COLUMN_PASS
