// gfs_pbl.cu -- the GFS nonlocal PBL (WRF v4.6.1 phys/module_bl_gfs.F,
// bl_pbl_physics = 3): the BL_GFS wrapper, MONINP and the TRIDI2 / TRIDIN /
// TRIDIT solvers, one thread per column.  ArWen's own work, Apache-2.0.
//
// Precision follows the Fortran declaration by declaration:
//
//  * BL_GFS receives default REAL (binary32) fields and holds its column in
//    REAL(kind_phys), which module_gfs_machine.F makes binary64.  Every
//    expression whose operands are all REAL is evaluated in binary32 and
//    only then widened (RRHOX, CPM, PSK, Q1, PRSL, PHII, PHIL, the STRESS
//    numerator); mixed expressions widen first.  The results go back to
//    REAL on assignment to the tendency arrays.
//  * MONINP is binary64 throughout.  Its PARAMETERs and the
//    module_gfs_physcons con_* constants are binary64 variables initialised
//    from default-REAL literals, so each is the float32 rounding of its
//    decimal, widened: gfs_pbl_constants.cuh holds those words, generated
//    from the pinned source by tools/gfs_pbl_wrf461_oracle/gen_constants.py.
//
// Under GPUWM_WRF_EXACT arithmetic is pinned with rounding-explicit intrinsics,
// so NVRTC can never contract a multiply-add the SSE2 gfortran build does
// not form (configure.wrf FCOPTIM -ffp-contract=off).  The module is loaded
// with --ftz=false, so binary32 subnormals survive as on the CPU.
//
// gfortran -O2 lowering, as the oracle build sees it: x**2 (integer) and
// x**2.0 (the PFAC parameter) are x*x; the two non-trivial real powers,
// (PSFC*.00001)**ROVCP in binary32 and (PRSL*.01)**ROVCP in binary64, call
// WOOF's own gfk_pow and glibc_pow.  MIN/MAX follow GFS_MIN/GFS_MAX below.
//
// Workspace: the per-column arrays live in a global binary64 workspace,
// slot-major with the thread fastest (element k of slot s for thread t is
// ws[((s*(nz+1)) + k-1) * nchunk + t]), 1-based like the Fortran.

// Strict arithmetic is the native replay opt-in. Fast runs permit contraction
// and use CUDA's math while retaining WRF's declared binary64 column state.
#ifndef GPUWM_WRF_EXACT
#define glibc_pow pow
#define glibc_exp exp
#define glibc_log log
#define gfk_pow powf
#define __dsqrt_rn sqrt
#define __fsqrt_rn sqrtf
#endif

#define GFS_WS_SLOTS 10
#define S_DEL 0
#define S_PRSI 1
#define S_THETA 2
#define S_DKU 3
#define S_DKT 4
#define S_AU 5
#define S_A1 6
#define S_A2 7
#define S_A3 8
#define S_A4 9

#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gA(double a, double b) { return __dadd_rn(a, b); }
#else
__device__ __forceinline__ double gA(double a, double b) { return a + b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gS(double a, double b) { return __dsub_rn(a, b); }
#else
__device__ __forceinline__ double gS(double a, double b) { return a - b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gM(double a, double b) { return __dmul_rn(a, b); }
#else
__device__ __forceinline__ double gM(double a, double b) { return a * b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gD(double a, double b) { return __ddiv_rn(a, b); }
#else
__device__ __forceinline__ double gD(double a, double b) { return a / b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float fA(float a, float b) { return __fadd_rn(a, b); }
#else
__device__ __forceinline__ float fA(float a, float b) { return a + b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float fS(float a, float b) { return __fsub_rn(a, b); }
#else
__device__ __forceinline__ float fS(float a, float b) { return a - b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float fM(float a, float b) { return __fmul_rn(a, b); }
#else
__device__ __forceinline__ float fM(float a, float b) { return a * b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float fD(float a, float b) { return __fdiv_rn(a, b); }
#else
__device__ __forceinline__ float fD(float a, float b) { return a / b; }
#endif
// Fortran MAX/MIN as gfortran 13.3 lowers REAL(8) MAX/MIN at the oracle's
// flags, measured by tools/gfs_pbl_wrf461_oracle/minmax_probe.F90 on ties
// of signed zeros and NaN operands: MAX(a, b) returns a only when a > b
// (else b); MIN(a, b) returns b only when b < a (else a).
__device__ __forceinline__ double GFS_MAX(double a, double b)
{ return (a > b) ? a : b; }
__device__ __forceinline__ double GFS_MIN(double a, double b)
{ return (b < a) ? b : a; }
// binary64 -> binary32 on assignment to a REAL, subnormals included.
__device__ __forceinline__ float g2f(double x) { return gfk_d2f_rn(x); }

extern "C" __global__ void gfs_pbl_columns(
    const float* __restrict__ u3d, const float* __restrict__ v3d,
    const float* __restrict__ t3d, const float* __restrict__ qv3d,
    const float* __restrict__ qc3d, const float* __restrict__ qi3d,
    const float* __restrict__ p3d, const float* __restrict__ pi3d,
    const float* __restrict__ dz8w, const float* __restrict__ z,
    const float* __restrict__ psfc, const float* __restrict__ psim,
    const float* __restrict__ psih, const float* __restrict__ hfx,
    const float* __restrict__ qfx, const float* __restrict__ gz1oz0,
    const float* __restrict__ br,
    float* __restrict__ wspd, float* __restrict__ ust,
    float* __restrict__ pbl, int* __restrict__ kpbl2d,
    float* __restrict__ rublten, float* __restrict__ rvblten,
    float* __restrict__ rthblten, float* __restrict__ rqvblten,
    float* __restrict__ rqcblten, float* __restrict__ rqiblten,
    double* __restrict__ ws,
    const float cp_wrf, const float g_wrf, const float rovcp,
    const float r_wrf, const float rovg, const float ep1, const float karman,
    const float dtf,
    const int flqi, const int km, const int ncol, const int col0,
    const int nchunk)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nchunk) return;
    const int i = col0 + t;
    if (i >= ncol) return;
    const size_t nc = (size_t)ncol;
    const size_t lev = (size_t)km + 1;
#define IN(a, k) (a)[(size_t)((k) - 1) * nc + i]
#define WS(s, k) ws[((size_t)(s) * lev + (size_t)((k) - 1)) * (size_t)nchunk + t]

    // ---- BL_GFS: module_bl_gfs.F:279-411 --------------------------------
    const int ntrac = flqi ? 3 : 2;
    const float qv1 = IN(qv3d, 1);
    const float t1f = IN(t3d, 1);
    const double rrhox = (double)fD(fM(fM(r_wrf, t1f), fA(1.0f, fM(ep1, qv1))),
                                    psfc[i]);
    const double cpm = (double)fM(cp_wrf, fA(1.0f, fM(0.8f, qv1)));
    const float fmtmp_f = fS(gz1oz0[i], psim[i]);
    const double fmtmp = (double)fmtmp_f;
    const double psk = (double)gfk_pow(fM(psfc[i], 0.00001f), rovcp);
    const double fm = fmtmp;
    const double fh = (double)fS(gz1oz0[i], psih[i]);
    const double heat = gM(gD((double)hfx[i], cpm), rrhox);
    const double evap = gM((double)qfx[i], rrhox);
    const float wspd_in = wspd[i];
    const double stress = gD(
        (double)fM(fM(fM(karman, karman), wspd_in), wspd_in),
        gM(fmtmp, fmtmp));
    const double spd1 = (double)wspd_in;
    const double rcl = 1.0;
    const double rbsoil = (double)br[i];
    WS(S_PRSI, 1) = (double)fM(psfc[i], 0.001f);

#define U1(k) ((double)IN(u3d, k))
#define V1(k) ((double)IN(v3d, k))
#define T1(k) ((double)IN(t3d, k))
#define PRSL(k) ((double)fM(IN(p3d, k), 0.001f))
#define Q1V(k) ((double)fD(IN(qv3d, k), fA(1.0f, IN(qv3d, k))))
#define Q1C(k) ((double)fD(IN(qc3d, k), fA(1.0f, IN(qc3d, k))))
#define Q1I(k) ((double)fD(IN(qi3d, k), fA(1.0f, IN(qi3d, k))))

    // DEL, PRSI (:380-395); PHII/PHIL are recomputed where read (below).
    for (int k = 2; k <= km; ++k) {
        const int m = k - 1;
        WS(S_DEL, m) = gD(gM(gD(PRSL(m), (double)rovg), (double)IN(dz8w, m)),
                          T1(m));
        WS(S_PRSI, k) = gS(WS(S_PRSI, m), WS(S_DEL, m));
    }
    WS(S_DEL, km) = WS(S_DEL, km - 1);
    WS(S_PRSI, km + 1) = gS(WS(S_PRSI, km), WS(S_DEL, km - 1));

    // ---- MONINP: module_bl_gfs.F:729-1298 --------------------------------
    const float zs = IN(z, 1);
    // PHII(k) = (Z(k)-Z(kts))*G, PHII(kts) = 0, PHII(kte+1) =
    // PHII(kte) + dz8w(kte)*G; PHIL(m) = 0.5*(Z(m+1)+Z(m)-2*Z(kts))*G,
    // PHIL(kte) = PHII(kte)-PHIL(kte-1)+PHII(kte)  (:385-394).
    auto phii = [&](int k) -> double {
        if (k == 1) return 0.0;
        const double top = (double)fM(fS(IN(z, (k <= km ? k : km)), zs), g_wrf);
        if (k <= km) return top;
        return gA(top, (double)fM(IN(dz8w, km), g_wrf));
    };
    auto phil = [&](int m) -> double {
        if (m < km)
            return (double)fM(fM(0.5f, fS(fA(IN(z, m + 1), IN(z, m)),
                                          fM(2.0f, zs))), g_wrf);
        const double pk = phii(km);
        const double pm = (double)fM(fM(0.5f, fS(fA(IN(z, km), IN(z, km - 1)),
                                                 fM(2.0f, zs))), g_wrf);
        return gA(gS(pk, pm), pk);
    };
    auto ZI = [&](int k) -> double { return gM(phii(k), GFS_GRAVI); };
    auto ZL = [&](int k) -> double { return gM(phil(k), GFS_GRAVI); };

    const double dt2 = gM(2.0, (double)dtf);
    const double rdt = gD(1.0, dt2);
    const int kmpbl = km / 2;
    const double rbcr = 0.25;

    // Native computes theta up to KM/2, but each height search stops
    // reading it at the first crossing. Evaluate those same words on
    // demand; the enhanced search extends the cache only if it goes higher.
    auto theta_at = [&](int k) -> double {
        const double prslk = glibc_pow(gM(PRSL(k), GFS_F_P01), (double)rovcp);
        return gD(gM(T1(k), psk), prslk);
    };
    WS(S_THETA, 1) = theta_at(1);
#define RDZT(k) gD(gM(GFS_GOR, WS(S_PRSI, (k) + 1)), gS(PRSL(k), PRSL((k) + 1)))

    double hgamt = 0.0, hgamq = 0.0, wscale = 0.0;
    int kpbl = 1;
    double hpbl = ZI(2);
    bool pblflg = true;
    bool sfcflg = true;
    if (rbsoil > 0.0) sfcflg = false;
    const double rdzt1 = gD(gM(GFS_GOR, PRSL(1)), WS(S_DEL, 1));
    const double beta = gD(gM(dt2, rdzt1), T1(1));
    const double ustar = __dsqrt_rn(stress);
    const double the1 = WS(S_THETA, 1);
    const double the1v = gM(the1, gA(1.0, gM(GFS_FV, GFS_MAX(Q1V(1), GFS_QMIN))));
    double thermal = the1v;

    // First guess of the PBL height (:809-840).
    bool stable = false;
    double rbup = rbsoil, rbdn = rbsoil;
    for (int k = 2; k <= kmpbl; ++k) {
        if (!stable) {
            WS(S_THETA, k) = theta_at(k);
            rbdn = rbup;
            const double thekv = gM(WS(S_THETA, k),
                                    gA(1.0, gM(GFS_FV, GFS_MAX(Q1V(k), GFS_QMIN))));
            const double spdk2 = GFS_MAX(gM(rcl, gA(gM(U1(k), U1(k)),
                                                    gM(V1(k), V1(k)))), 1.0);
            rbup = gD(gM(gS(thekv, the1v), gD(gM(GFS_G, ZL(k)), the1v)), spdk2);
            kpbl = k;
            stable = rbup > rbcr;
        }
    }
    const int first_theta_top = kpbl;
    {
        const int k = kpbl;
        double rbint;
        if (rbdn >= rbcr) rbint = 0.0;
        else if (rbup <= rbcr) rbint = 1.0;
        else rbint = gD(gS(rbcr, rbdn), gS(rbup, rbdn));
        const double zlm = ZL(k - 1);
        hpbl = gA(zlm, gM(rbint, gS(ZL(k), zlm)));
        if (hpbl < ZI(kpbl)) kpbl = kpbl - 1;
    }
    // Stability functions and the velocity scale (:842-865).
    double phih, phim;
    {
        double hol = GFS_MAX(gD(gM(gM(rbsoil, fm), fm), fh), GFS_RIMIN);
        if (sfcflg) hol = GFS_MIN(hol, -GFS_ZFMIN);
        else hol = GFS_MAX(hol, GFS_ZFMIN);
        hol = gM(gD(gM(hol, hpbl), ZL(1)), GFS_SFCFRAC);
        if (sfcflg) {
            const double tem = gD(1.0, gS(1.0, gM(GFS_APHI16, hol)));
            phih = __dsqrt_rn(tem);
            phim = __dsqrt_rn(phih);
        } else {
            phim = gA(1.0, gM(GFS_APHI5, hol));
            phih = phim;
        }
        wscale = gD(ustar, phim);
        wscale = GFS_MIN(wscale, gM(ustar, GFS_APHI16));
        wscale = GFS_MAX(wscale, gD(ustar, GFS_APHI5));
    }
    // Surface variables for the unstable PBL estimate (:870-883).
    {
        const double sflux = gA(heat, gM(gM(evap, GFS_FV), the1));
        if (sfcflg && sflux > 0.0) {
            hgamt = GFS_MIN(gD(gM(GFS_CFAC, heat), wscale), GFS_GAMCRT);
            hgamq = GFS_MIN(gD(gM(GFS_CFAC, evap), wscale), GFS_GAMCRQ);
            double vpert = gA(hgamt, gM(gM(GFS_FV, the1), hgamq));
            vpert = GFS_MIN(vpert, GFS_GAMCRT);
            thermal = gA(thermal, GFS_MAX(vpert, 0.0));
            hgamt = GFS_MAX(hgamt, 0.0);
            hgamq = GFS_MAX(hgamq, 0.0);
        } else {
            pblflg = false;
        }
    }
    if (pblflg) { kpbl = 1; hpbl = ZI(2); }
    // Enhance the PBL height by the thermal (:894-937).
    if (pblflg) {
        stable = false;
        rbup = rbsoil;
        for (int k = 2; k <= kmpbl; ++k) {
            if (!stable) {
                if (k > first_theta_top) WS(S_THETA, k) = theta_at(k);
                rbdn = rbup;
                const double thekv = gM(WS(S_THETA, k),
                                        gA(1.0, gM(GFS_FV, GFS_MAX(Q1V(k), GFS_QMIN))));
                const double spdk2 = GFS_MAX(gM(rcl, gA(gM(U1(k), U1(k)),
                                                        gM(V1(k), V1(k)))), 1.0);
                rbup = gD(gM(gS(thekv, thermal), gD(gM(GFS_G, ZL(k)), the1v)),
                          spdk2);
                kpbl = k;
                stable = rbup > rbcr;
            }
        }
        const int k = kpbl;
        double rbint;
        if (rbdn >= rbcr) rbint = 0.0;
        else if (rbup <= rbcr) rbint = 1.0;
        else rbint = gD(gS(rbcr, rbdn), gS(rbup, rbdn));
        const double zlm = ZL(k - 1);
        hpbl = gA(zlm, gM(rbint, gS(ZL(k), zlm)));
        if (hpbl < ZI(kpbl)) kpbl = kpbl - 1;
        if (kpbl <= 1) pblflg = false;
    }
    // Diffusivities below the PBL (:970-1072; ALPHA = 1, the non-HWRF
    // PARAMETER, so ABS(ALPHA) is an exact 1).
    // DEL is no longer read as a pressure thickness after BETA above;
    // retain DT2/DEL once for the two matrix builds. THETA is no longer
    // read after the PBL searches, so reuse that slot for RDZT. These are
    // the same native divisions, shared without changing their order or
    // precision, and the workspace remains bounded by the existing pool.
    for (int k = 1; k <= km - 1; ++k) {
        WS(S_THETA, k) = RDZT(k);
        // PRSI(k) is dead now; the next RDZT only reads PRSI(k+2).
        // Both implicit matrices use this identical native division.
        WS(S_PRSI, k) = gD(gM(WS(S_THETA, k), 2.0), gA(T1(k), T1(k + 1)));
        WS(S_DEL, k) = gD(dt2, WS(S_DEL, k));
    }
    WS(S_DEL, km) = gD(dt2, WS(S_DEL, km));
#undef RDZT
#define RDZT(k) WS(S_THETA, k)
    const double zl1 = ZL(1);
    for (int k = 1; k <= kmpbl; ++k) {
        if (kpbl > k) {
            double prinv = gD(1.0, gA(gD(phih, phim), GFS_CFAC_VK_P1));
            prinv = GFS_MIN(prinv, GFS_PRMAX);
            prinv = GFS_MAX(prinv, GFS_PRMIN);
            double zfac = GFS_MAX(gS(1.0, gD(gS(ZI(k + 1), zl1), gS(hpbl, zl1))),
                                  GFS_ZFMIN);
            double dku = gA(GFS_XKZO, gM(gM(gM(gM(wscale, GFS_VK), ZI(k + 1)),
                                            1.0), gM(zfac, zfac)));
            double dkt = gM(dku, prinv);
            dku = GFS_MIN(dku, GFS_DKMAX);
            dku = GFS_MAX(dku, GFS_DKMIN);
            dkt = GFS_MIN(dkt, GFS_DKMAX);
            dkt = GFS_MAX(dkt, GFS_DKMIN);
            WS(S_DKU, k) = dku;
            WS(S_DKT, k) = dkt;
        }
    }
    // Diffusivities over the PBL, the free atmosphere (:1080-1132).
    for (int k = 1; k <= km - 1; ++k) {
        if (k >= kpbl) {
            const double ti = gD(2.0, gA(T1(k), T1(k + 1)));
            const double rdz = gM(RDZT(k), ti);
            const double du_ = gS(U1(k), U1(k + 1));
            const double dv_ = gS(V1(k), V1(k + 1));
            const double dw2 = gM(rcl, gA(gM(du_, du_), gM(dv_, dv_)));
            const double shr2 = gM(GFS_MAX(dw2, GFS_DW2MIN), gM(rdz, rdz));
            const double tvd = gM(T1(k), gA(1.0, gM(GFS_FV, GFS_MAX(Q1V(k), GFS_QMIN))));
            const double tvu = gM(T1(k + 1),
                                  gA(1.0, gM(GFS_FV, GFS_MAX(Q1V(k + 1), GFS_QMIN))));
            const double bvf2 = gM(gM(GFS_G, gA(GFS_GOCP, gM(rdz, gS(tvu, tvd)))), ti);
            const double ri = GFS_MAX(gD(bvf2, shr2), GFS_RIMIN);
            const double zk = gM(GFS_VK, ZI(k + 1));
            const double rl2 = gD(gM(zk, GFS_RLAM), gA(GFS_RLAM, zk));
            const double dk = gM(gM(rl2, rl2), __dsqrt_rn(shr2));
            double dku, dkt;
            if (ri < 0.0) {
                const double sri = __dsqrt_rn(-ri);
                dku = gA(GFS_XKZO, gM(dk, gA(1.0, gD(gM(GFS_F_8, -ri),
                                                      gA(1.0, gM(GFS_F_1746, sri))))));
                const double tem = gM(dk, gA(1.0, gD(gM(GFS_F_8, -ri),
                                                     gA(1.0, gM(GFS_F_1286, sri)))));
                dkt = gA(GFS_XKZO, tem);
            } else {
                const double b = gA(1.0, gM(GFS_F_5, ri));
                const double tem = gD(dk, gM(b, b));
                dkt = gA(GFS_XKZO, tem);
                double prnum = gA(1.0, gM(GFS_F_21, ri));
                prnum = GFS_MIN(prnum, GFS_PRMAX);
                dku = gA(gM(gS(dkt, GFS_XKZO), prnum), GFS_XKZO);
            }
            dku = GFS_MIN(dku, GFS_DKMAX);
            dku = GFS_MAX(dku, GFS_DKMIN);
            dkt = GFS_MIN(dkt, GFS_DKMAX);
            dkt = GFS_MAX(dkt, GFS_DKMIN);
            WS(S_DKU, k) = dku;
            WS(S_DKT, k) = dkt;
        }
    }

    // ---- Heat, moisture and condensates (:1137-1193, :1246-1295) ----------
    // Heat/vapour and condensates have exactly the same AL/AD/AU matrix:
    // both start AD(1)=1 and use DKT, DEL, DSIG and RDZ in the same order.
    // Only their right-hand sides differ. Factor that matrix once for all
    // four right-hand sides, retaining every native recurrence and rounding.
    // Momentum below has its own DKU matrix and surface stress condition.
    // Stream matrix construction directly through native Thomas forward
    // elimination. Carry the preceding row in registers: the original
    // separate sweeps wrote AL/AD and reread every RHS from global memory.
    // Each row retains precisely the same native operation order.
    double diag = 1.0;
    double heat_rhs = gA(T1(1), gM(beta, heat));
    double vapor_rhs = gA(Q1V(1), gM(beta, evap));
    double prev_au = 0.0, prev_al = 0.0;
    double prev_t = 0.0, prev_q = 0.0, prev_c = 0.0, prev_i = 0.0;
    for (int k = 1; k <= km; ++k) {
        double au = 0.0, al = 0.0;
        double next_t = 0.0, next_q = 0.0;
        if (k < km) {
            const double dtodsd = WS(S_DEL, k);
            const double dtodsu = WS(S_DEL, k + 1);
            const double dsig = gS(PRSL(k), PRSL(k + 1));
            const double rdz = WS(S_PRSI, k);
            const double tem1 = gM(gM(dsig, WS(S_DKT, k)), rdz);
            double dsdzt;
            if (pblflg && k < kpbl) {
                const double tem = gD(1.0, hpbl);
                dsdzt = gM(tem1, gS(GFS_GOCP, gM(hgamt, tem)));
                const double dsdzq = gM(tem1, -gM(hgamq, tem));
                vapor_rhs = gA(vapor_rhs, gM(dtodsd, dsdzq));
                next_q = gS(Q1V(k + 1), gM(dtodsu, dsdzq));
            } else {
                dsdzt = gM(tem1, GFS_GOCP);
                next_q = Q1V(k + 1);
            }
            const double dsdz2 = gM(tem1, rdz);
            au = -gM(dtodsd, dsdz2);
            al = -gM(dtodsu, dsdz2);
            diag = gS(diag, au);
            heat_rhs = gA(heat_rhs, gM(dtodsd, dsdzt));
            next_t = gS(T1(k + 1), gM(dtodsu, dsdzt));
        }
        const double fk = (k == 1) ? gD(1.0, diag)
            : gD(1.0, gS(diag, gM(prev_al, prev_au)));
        const double tk = (k == 1) ? gM(fk, heat_rhs)
            : gM(fk, gS(heat_rhs, gM(prev_al, prev_t)));
        const double qk = (k == 1) ? gM(fk, vapor_rhs)
            : gM(fk, gS(vapor_rhs, gM(prev_al, prev_q)));
        const double ck = (k == 1) ? gM(fk, Q1C(k))
            : gM(fk, gS(Q1C(k), gM(prev_al, prev_c)));
        WS(S_A1, k) = prev_t = tk;
        WS(S_A2, k) = prev_q = qk;
        WS(S_A3, k) = prev_c = ck;
        if (ntrac == 3) {
            const double ik = (k == 1) ? gM(fk, Q1I(k))
                : gM(fk, gS(Q1I(k), gM(prev_al, prev_i)));
            WS(S_A4, k) = prev_i = ik;
        }
        if (k < km) {
            WS(S_AU, k) = prev_au = gM(fk, au);
            prev_al = al;
            diag = gS(1.0, al);
            heat_rhs = next_t;
            vapor_rhs = next_q;
        }
    }
    // RTHBLTEN = TAU/PI3D, RQVBLTEN = RTG/(1-Q1)**2 (:445-452); TAU and
    // RTG start at BL_GFS's zero, so each is 0 + TTEND (a -0 becomes +0).
    for (int k = km; k >= 1; --k) {
        if (k < km) {
            const double au = WS(S_AU, k);
            prev_t = gS(WS(S_A1, k), gM(au, prev_t));
            prev_q = gS(WS(S_A2, k), gM(au, prev_q));
            prev_c = gS(WS(S_A3, k), gM(au, prev_c));
            if (ntrac == 3) prev_i = gS(WS(S_A4, k), gM(au, prev_i));
        }
        const double ttend = gM(gS(prev_t, T1(k)), rdt);
        const double q1 = Q1V(k);
        const double qtend = gM(gS(prev_q, q1), rdt);
        const double tau = gA(0.0, ttend);
        const double rtg = gA(0.0, qtend);
        IN(rthblten, k) = g2f(gD(tau, (double)IN(pi3d, k)));
        const double om = gS(1.0, q1);
        IN(rqvblten, k) = g2f(gD(rtg, gM(om, om)));
        const double qc1 = Q1C(k);
        const double rtc = gA(0.0, gM(gS(prev_c, qc1), rdt));
        const double oc = gS(1.0, qc1);
        IN(rqcblten, k) = g2f(gD(rtc, gM(oc, oc)));
        if (ntrac == 3) {
            const double qi1 = Q1I(k);
            const double rti = gA(0.0, gM(gS(prev_i, qi1), rdt));
            const double oi = gS(1.0, qi1);
            IN(rqiblten, k) = g2f(gD(rti, gM(oi, oi)));
        }
    }

    // ---- Momentum (:1197-1241) -------------------------------------------
    diag = gA(1.0, gD(gM(beta, stress), spd1));
    prev_au = 0.0; prev_al = 0.0;
    double prev_u = 0.0, prev_v = 0.0;
    for (int k = 1; k <= km; ++k) {
        double au = 0.0, al = 0.0;
        if (k < km) {
            const double dtodsd = WS(S_DEL, k);
            const double dtodsu = WS(S_DEL, k + 1);
            const double dsig = gS(PRSL(k), PRSL(k + 1));
            const double rdz = WS(S_PRSI, k);
            const double dsdz2 = gM(gM(gM(dsig, WS(S_DKU, k)), rdz), rdz);
            au = -gM(dtodsd, dsdz2);
            al = -gM(dtodsu, dsdz2);
            diag = gS(diag, au);
        }
        const double fk = (k == 1) ? gD(1.0, diag)
            : gD(1.0, gS(diag, gM(prev_al, prev_au)));
        const double uk = (k == 1) ? gM(fk, U1(k))
            : gM(fk, gS(U1(k), gM(prev_al, prev_u)));
        const double vk = (k == 1) ? gM(fk, V1(k))
            : gM(fk, gS(V1(k), gM(prev_al, prev_v)));
        WS(S_A1, k) = prev_u = uk;
        WS(S_A2, k) = prev_v = vk;
        if (k < km) {
            WS(S_AU, k) = prev_au = gM(fk, au);
            prev_al = al;
            diag = gS(1.0, al);
        }
    }
    for (int k = km; k >= 1; --k) {
        if (k < km) {
            const double au = WS(S_AU, k);
            prev_u = gS(WS(S_A1, k), gM(au, prev_u));
            prev_v = gS(WS(S_A2, k), gM(au, prev_v));
        }
        IN(rublten, k) = g2f(gA(0.0, gM(gS(prev_u, U1(k)), rdt)));
        IN(rvblten, k) = g2f(gA(0.0, gM(gS(prev_v, V1(k)), rdt)));
    }

    // ---- BL_GFS epilogue (:463-475) --------------------------------------
    ust[i] = g2f(__dsqrt_rn(stress));
    {
        const float us = IN(u3d, 1), vs = IN(v3d, 1);
        wspd[i] = fA(__fsqrt_rn(fA(fM(us, us), fM(vs, vs))), 1.0e-9f);
    }
    pbl[i] = g2f(hpbl);
    kpbl2d[i] = kpbl;
#undef IN
#undef WS
#undef U1
#undef V1
#undef T1
#undef PRSL
#undef Q1V
#undef Q1C
#undef Q1I
#undef RDZT
}

// One coalesced validation pass replaces eight temporary arrays and
// reductions. Status bits retain the first-field failure identity, and
// the existing health ledger owns the sole deferred readback.
extern "C" __global__ void gfs_pbl_finite(
    const unsigned long long n, const unsigned long long ns,
    const float* a0, const float* a1, const float* a2,
    const float* a3, const float* a4, const float* a5,
    const float* a6, const float* a7, unsigned int* status)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bad = 0;
    if (i < n) {
        if (!isfinite(a0[i])) bad |= 1u;
        if (!isfinite(a1[i])) bad |= 2u;
        if (!isfinite(a2[i])) bad |= 4u;
        if (!isfinite(a3[i])) bad |= 8u;
        if (!isfinite(a4[i])) bad |= 16u;
        if (!isfinite(a5[i])) bad |= 32u;
    }
    if (i < ns) {
        if (!isfinite(a6[i])) bad |= 64u;
        if (!isfinite(a7[i])) bad |= 128u;
    }
    if (bad) atomicOr(status, bad);
}
