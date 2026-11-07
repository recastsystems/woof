// sf_gfs.cu -- the GFS surface layer, WRF v4.6.1 sf_sfclay_physics = 3
// (phys/module_sf_gfs.F: SF_GFS over PROGTM, and module_gfs_funcphys.F's
// FPVS table).  WOOF's own work, Apache-2.0.
//
// The scheme's column is kind_phys = selected_real_kind(13,60), binary64
// (module_gfs_machine.F:7), so this unit computes in binary64 and does not
// demote. Under GPUWM_WRF_EXACT three rules make it equal WRF's words (graded against
// tools/sfgfs_wrf461_oracle, which is built with WRF's own referee flags,
// -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize, and checked
// byte-identical to the referee build's object files):
//
//  1. Every arithmetic operation is a rounding-pinned intrinsic
//     (__dadd_rn/__dsub_rn/__dmul_rn/__ddiv_rn/__dsqrt_rn, and the __f*_rn
//     forms for the REAL statements of SF_GFS), so NVRTC can never contract
//     a multiply-add the SSE2 build does not. Default arithmetic permits
//     contraction and uses CUDA's math functions.
//  2. Fortran literal kinds.  WRF builds with 4-byte default REAL, so a
//     literal without _kind_phys is the float32 rounding of the decimal,
//     widened exactly: written F(.014f) below, never .014.  That includes
//     the con_* constants of module_gfs_physcons.F (con_g = 9.80665e+0 is a
//     float32 literal).  PARAMETER expressions are folded by the compiler
//     one correctly rounded binary64 operation at a time, which is what the
//     same intrinsics compute at run time.
//  3. Real exponents.  gfortran folds x**2 (integer) to x*x; every REAL
//     exponent here (0.25, -0.25, -0.5, ROVCP, the FPVSX exponents) is a
//     pow call (nm -u module_sf_gfs.o lists pow, exp and log only).
//     exp/log/pow are WOOF's own binary64 routines (glibc_flt64.cuh, which
//     the loader places ahead of this unit).
//
// Declared divergences from the shipped WRF statements live in the
// FRACTIONAL_SEAICE wrapper kernel below and in gpuwm/core/sf_gfs.py.
//
// What is NOT here, on purpose: PROGT2, KTSOIL, FUNCDF, FUNCKT, THSAT, TWLT
// and module_progtm's DFK/KTK/DFKT/TSAT tables.  WRF v4.6.1 never executes
// them: PROGTM jumps over its whole soil model (GOTO 1111/2222/5555,
// module_sf_gfs.F:489, :525, :753) and nothing in the tree calls PROGT2.
// Code no WRF run reaches has no output to grade and no user to serve.

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

#define D(x) ((double)(x))
#define F(x) ((double)(float)(x))

#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gd_add(double a, double b) { return __dadd_rn(a, b); }
#else
__device__ __forceinline__ double gd_add(double a, double b) { return a + b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gd_sub(double a, double b) { return __dsub_rn(a, b); }
#else
__device__ __forceinline__ double gd_sub(double a, double b) { return a - b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gd_mul(double a, double b) { return __dmul_rn(a, b); }
#else
__device__ __forceinline__ double gd_mul(double a, double b) { return a * b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ double gd_div(double a, double b) { return __ddiv_rn(a, b); }
#else
__device__ __forceinline__ double gd_div(double a, double b) {
    // A shared, correctly rounded binary64 reciprocal lets NVRTC reuse
    // denominators (notably z1, fm and fh) across the stability equations.
    return a * __drcp_rn(b);
}
#endif
__device__ __forceinline__ double gd_sqrt(double a) { return __dsqrt_rn(a); }
__device__ __forceinline__ double gd_max(double a, double b) { return (a > b) ? a : b; }
__device__ __forceinline__ double gd_min(double a, double b) { return (a < b) ? a : b; }
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float gf_add(float a, float b) { return __fadd_rn(a, b); }
#else
__device__ __forceinline__ float gf_add(float a, float b) { return a + b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float gf_sub(float a, float b) { return __fsub_rn(a, b); }
#else
__device__ __forceinline__ float gf_sub(float a, float b) { return a - b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float gf_mul(float a, float b) { return __fmul_rn(a, b); }
#else
__device__ __forceinline__ float gf_mul(float a, float b) { return a * b; }
#endif
#ifdef GPUWM_WRF_EXACT
__device__ __forceinline__ float gf_div(float a, float b) { return __fdiv_rn(a, b); }
#else
__device__ __forceinline__ float gf_div(float a, float b) { return a / b; }
#endif
__device__ __forceinline__ float gf_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float gd_narrow(double a) { return __double2float_rn(a); }

// The fast arm keeps binary64 but reduces fixed powers to roots. The
// native pow implementation remains the strict replay referee, including
// its own rounding. Nonpositive/special arguments keep CUDA pow semantics.
__device__ __forceinline__ double gfs_quarter(double x) {
#ifdef GPUWM_WRF_EXACT
    return glibc_pow(x, F(.25f));
#else
    return x > 0.0 ? sqrt(sqrt(x)) : pow(x, F(.25f));
#endif
}
__device__ __forceinline__ double gfs_neg_quarter(double x) {
#ifdef GPUWM_WRF_EXACT
    return glibc_pow(x, F(-.25f));
#else
    return x > 0.0 ? rsqrt(sqrt(x)) : pow(x, F(-.25f));
#endif
}
__device__ __forceinline__ double gfs_neg_half(double x) {
#ifdef GPUWM_WRF_EXACT
    return glibc_pow(x, F(-.5f));
#else
    return x > 0.0 ? rsqrt(x) : pow(x, F(-.5f));
#endif
}

// ---- module_gfs_physcons.F (float32 literals widened) ----------------------
#define CON_G     F(9.80665e+0f)
#define CON_RD    F(2.8705e+2f)
#define CON_RV    F(4.6150e+2f)
#define CON_CVAP  F(1.8460e+3f)
#define CON_CLIQ  F(4.1855e+3f)
#define CON_CSOL  F(2.1060e+3f)
#define CON_HVAP  F(2.5000e+6f)
#define CON_HFUS  F(3.3358e+5f)
#define CON_PSAT  F(6.1078e+2f)
#define CON_TTP   F(2.7316e+2f)

//: Number of FPVS table words (module_gfs_funcphys.F:254, nxpvs).
#define GFS_NXPVS 7501

// ---- module_gfs_funcphys.F FPVSX (:? fpvsx), binary64 ------------------------
__device__ double gfs_fpvsx(double t) {
    const double tliq = CON_TTP;
    const double tice = gd_sub(CON_TTP, F(20.0f));
    const double dldtl = gd_sub(CON_CVAP, CON_CLIQ);
    const double heatl = CON_HVAP;
    const double xponal = gd_div(-dldtl, CON_RV);
    const double xponbl = gd_add(gd_div(-dldtl, CON_RV),
                                 gd_div(heatl, gd_mul(CON_RV, CON_TTP)));
    const double dldti = gd_sub(CON_CVAP, CON_CSOL);
    const double heati = gd_add(CON_HVAP, CON_HFUS);
    const double xponai = gd_div(-dldti, CON_RV);
    const double xponbi = gd_add(gd_div(-dldti, CON_RV),
                                 gd_div(heati, gd_mul(CON_RV, CON_TTP)));
    const double tr = gd_div(CON_TTP, t);
    const double omtr = gd_sub(1.0, tr);
    if (t >= tliq) {
        return gd_mul(gd_mul(CON_PSAT, glibc_pow(tr, xponal)),
                      glibc_exp(gd_mul(xponbl, omtr)));
    } else if (t < tice) {
        return gd_mul(gd_mul(CON_PSAT, glibc_pow(tr, xponai)),
                      glibc_exp(gd_mul(xponbi, omtr)));
    }
    const double w = gd_div(gd_sub(t, tice), gd_sub(tliq, tice));
    const double pvl = gd_mul(gd_mul(CON_PSAT, glibc_pow(tr, xponal)),
                              glibc_exp(gd_mul(xponbl, omtr)));
    const double pvi = gd_mul(gd_mul(CON_PSAT, glibc_pow(tr, xponai)),
                              glibc_exp(gd_mul(xponbi, omtr)));
    return gd_add(gd_mul(w, pvl), gd_mul(gd_sub(1.0, w), pvi));
}

// GPVS (module_gfs_funcphys.F gpvs): the table FPVS interpolates, word for
// word in WRF's construction order.  Thread jx-1 writes tbpvs(jx); thread 0
// also writes c1xpvs and c2xpvs after the table, at [GFS_NXPVS] and
// [GFS_NXPVS + 1].
extern "C" __global__ void gfs_fpvs_table(double* __restrict__ table) {
    const int j0 = blockIdx.x * blockDim.x + threadIdx.x;   // jx - 1
    const double xmin = 180.0, xmax = 330.0;
    const double xinc = gd_div(gd_sub(xmax, xmin), (double)(GFS_NXPVS - 1));
    if (j0 < GFS_NXPVS) {
        const double x = gd_add(xmin, gd_mul((double)j0, xinc));
        table[j0] = gfs_fpvsx(x);
    }
    if (j0 == 0) {
        const double c2 = gd_div(1.0, xinc);
        table[GFS_NXPVS] = gd_sub(1.0, gd_mul(xmin, c2));
        table[GFS_NXPVS + 1] = c2;
    }
}

// FPVS: linear interpolation in the table (module_gfs_funcphys.F fpvs).
__device__ __forceinline__ double gfs_fpvs(const double* __restrict__ tb,
                                           double t) {
    const double c1 = tb[GFS_NXPVS], c2 = tb[GFS_NXPVS + 1];
    const double xj = gd_min(gd_max(gd_add(c1, gd_mul(c2, t)), 1.0),
                             (double)GFS_NXPVS);
    const int jx = (int)gd_min(xj, (double)(GFS_NXPVS - 1));   // INT() truncates
    const double lo = tb[jx - 1], hi = tb[jx];
    return gd_add(lo, gd_mul(gd_sub(xj, (double)jx), gd_sub(hi, lo)));
}

struct GfsIn {
    float u1, v1, t1, qv1, p1, psfc, tsk, xland;
};
struct GfsOut {
    float br, chs, chs2, cpm, cqs2, flhc, flqc, gz1oz0, hfx, lh,
          psim, psih, qfx, qgh, qsfc, u10, v10, wspd;
};
struct GfsConst {
    float cp, rovcp, r, xlv, ep1, ep2, karman;
    int isfflx;
};

// One column of SF_GFS (module_sf_gfs.F:182-282) with PROGTM's executed
// statements (:436-753 then :5555-1486) inline.  znt and ust are INOUT.
__device__ void gfs_sfclay_one(const GfsIn& in, float& znt, float& ust,
                               GfsOut& o, const GfsConst& k,
                               const double* __restrict__ tb) {
    // ---- SF_GFS :191-215 ----------------------------------------------------
    const double prsl1 = D(gf_mul(in.p1, .001f));          // REAL*REAL
    const double ps = D(gf_mul(in.psfc, .001f));
    double q1 = D(in.qv1);
    const double slimsk = D(fabsf(gf_sub(in.xland, 2.f)));
    const double tskin = D(in.tsk);
    const double t1 = D(in.t1);
    const double u1 = D(in.u1);
    double ustar = D(ust);
    const double v1 = D(in.v1);
    double z0rl = D(gf_mul(znt, 100.f));
    const double rovcp = D(k.rovcp);
#ifdef GPUWM_WRF_EXACT
    const double prslki = glibc_pow(gd_div(ps, prsl1), rovcp);
#else
    // The hydrostatic height and pressure power use this same logarithm.
    // Keep both transcendental evaluations and their operands binary64.
    const double log_pressure_ratio = log(gd_div(prsl1, ps));
    const double prslki = exp(-rovcp * log_pressure_ratio);
#endif
#ifdef GPUWM_WRF_EXACT
    const double surface_exner = glibc_pow(gd_div(F(100.f), ps), rovcp);
#else
    const double surface_exner = exp(rovcp * log(gd_div(F(100.f), ps)));
#endif
    const double thgb = gd_mul(tskin, surface_exner);
#ifdef GPUWM_WRF_EXACT
    const double thx = gd_mul(t1, glibc_pow(gd_div(F(100.f), prsl1), rovcp));
#else
    // (100/prsl1)^rovcp = (100/ps)^rovcp * (ps/prsl1)^rovcp.
    // Both factors are already required by the native surface equations.
    const double thx = gd_mul(t1, gd_mul(surface_exner, prslki));
#endif
    const double rho1 = gd_div(gd_mul(prsl1, F(1000.f)),
                               gd_mul(gd_mul(D(k.r), t1),
                                      gd_add(1.0, gd_mul(D(k.ep1), q1))));
    q1 = gd_div(q1, gd_add(1.0, q1));

    // ---- PROGTM :453-487 (RCL = 1, DDVEL = 0) --------------------------------
    const double xrcl = gd_sqrt(1.0);
    const double psurf = gd_mul(F(1000.f), ps);
    const double ps1 = gd_mul(F(1000.f), prsl1);
    double wind = gd_add(gd_mul(xrcl, gd_sqrt(gd_add(gd_mul(u1, u1),
                                                     gd_mul(v1, v1)))),
                         gd_max(0.0, gd_min(0.0, 30.0)));
    wind = gd_max(wind, 1.0);
    const double q0 = gd_max(q1, 1.e-8);                     // 1.E-8_kind_phys
    const double tsurf = tskin;
    const double theta1 = gd_mul(t1, prslki);
    const double rvrdm1 = gd_sub(gd_div(CON_RV, CON_RD), F(1.f));
    const double eps = gd_div(CON_RD, CON_RV);
    const double epsm1 = gd_sub(gd_div(CON_RD, CON_RV), F(1.f));
    const double tv1 = gd_mul(t1, gd_add(F(1.f), gd_mul(rvrdm1, q0)));
    const double thv1 = gd_mul(theta1, gd_add(F(1.f), gd_mul(rvrdm1, q0)));
    const double tvs = gd_mul(tsurf, gd_add(F(1.f), gd_mul(rvrdm1, q0)));
    // QS1 and the MIN(QS1,Q0) clip feed only the skipped soil model.
    double qss = gfs_fpvs(tb, tskin);
    qss = gd_div(gd_mul(eps, qss), gd_add(psurf, gd_mul(epsm1, qss)));
    double z0 = gd_mul(F(.01f), z0rl);

    // ---- :543-546 ------------------------------------------------------------
    const double g = CON_G;
#ifdef GPUWM_WRF_EXACT
    const double log_pressure_height = glibc_log(gd_div(ps1, psurf));
#else
    const double log_pressure_height = log_pressure_ratio;
#endif
    const double z1 = gd_div(gd_mul(gd_mul(-CON_RD, tv1),
                                    log_pressure_height), g);

    // ---- :577-601 ------------------------------------------------------------
    const double charnock = F(.014f), ca = F(.4f), vis = F(1.4e-5f);
    if (slimsk == 0.0) {
        ustar = gd_sqrt(gd_div(gd_mul(g, z0), charnock));
    }
    const double z0max = gd_min(z0, gd_mul(F(0.1f), z1));
    double ztmax = z0max;
    if (slimsk == 0.0) {
        double restar = gd_div(gd_mul(ustar, z0max), vis);
        restar = gd_max(restar, .000001);                     // _kind_phys
        double rat = gd_sub(gd_mul(F(2.67f), gfs_quarter(restar)),
                            F(2.57f));
        rat = gd_min(rat, 7.0);
        ztmax = gd_mul(z0max, glibc_exp(-rat));
    }

    // ---- :607-622 ------------------------------------------------------------
    double dtv = gd_sub(thv1, tvs);
    double adtv = fabs(dtv);
    adtv = gd_max(adtv, .001);                                // _kind_phys
    dtv = gd_mul(copysign(1.0, dtv), adtv);
    double rb = gd_div(gd_mul(gd_mul(g, dtv), z1),
                       gd_mul(gd_mul(gd_mul(F(.5f), gd_add(thv1, tvs)), wind),
                              wind));
    rb = gd_max(rb, -5000.0);
    // Equal roughness gives bit-identical arguments in the native
    // expressions. Share their logarithms and stable corrections without
    // instantiating the complete divergent column twice.
    const double log_z1_z0 = glibc_log(gd_div(z1, z0));
    double fm = (z0max == z0) ? log_z1_z0 : glibc_log(gd_div(z1, z0max));
    double fh = (ztmax == z0max) ? fm : glibc_log(gd_div(z1, ztmax));
    double hlinf = gd_div(gd_mul(gd_mul(rb, fm), fm), fh);
    double fm10 = glibc_log(gd_div(gd_add(z0max, F(10.f)), z0max));
    double fh2 = glibc_log(gd_div(gd_add(ztmax, F(2.f)), ztmax));

    const double four_alpha = gd_mul(F(4.f), F(5.f));          // 4.*ALPHA
    double pm = 0.0, ph = 0.0, pm10 = 0.0, ph2 = 0.0, hl1 = 0.0;
    if (dtv >= 0.0) {
        // ---- stable, :630-669 ------------------------------------------------
        hl1 = hlinf;
        if (hlinf > F(.25f)) {
            const double hl0inf = gd_div(gd_mul(z0max, hlinf), z1);
            const double hltinf = gd_div(gd_mul(ztmax, hlinf), z1);
            const double aa = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hlinf)));
            const double aa0 = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hl0inf)));
            const double bb = aa;
            const double bb0 = (ztmax == z0max) ? aa0 : gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hltinf)));
            pm = gd_add(gd_sub(aa0, aa), glibc_log(gd_div(gd_add(aa, 1.0),
                                                          gd_add(aa0, 1.0))));
            ph = (ztmax == z0max) ? pm : gd_add(gd_sub(bb0, bb), glibc_log(gd_div(gd_add(bb, 1.0),
                                                          gd_add(bb0, 1.0))));
            const double fms = gd_sub(fm, pm);
            const double fhs = gd_sub(fh, ph);
            hl1 = gd_div(gd_mul(gd_mul(fms, fms), rb), fhs);
        }
        const double hl0 = gd_div(gd_mul(z0max, hl1), z1);
        const double hlt = gd_div(gd_mul(ztmax, hl1), z1);
        double aa = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hl1)));
        const double aa0 = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hl0)));
        double bb = aa;
        const double bb0 = (ztmax == z0max) ? aa0 : gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hlt)));
        pm = gd_add(gd_sub(aa0, aa), glibc_log(gd_div(gd_add(aa, 1.0),
                                                      gd_add(aa0, 1.0))));
        ph = (ztmax == z0max) ? pm : gd_add(gd_sub(bb0, bb), glibc_log(gd_div(gd_add(bb, 1.0),
                                                      gd_add(bb0, 1.0))));
        const double hl110 = gd_div(gd_mul(hl1, F(10.f)), z1);
        aa = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hl110)));
        pm10 = gd_add(gd_sub(aa0, aa), glibc_log(gd_div(gd_add(aa, 1.0),
                                                        gd_add(aa0, 1.0))));
        const double hl12 = gd_div(gd_mul(hl1, F(2.f)), z1);
        bb = gd_sqrt(gd_add(1.0, gd_mul(four_alpha, hl12)));
        ph2 = gd_add(gd_sub(bb0, bb), glibc_log(gd_div(gd_add(bb, 1.0),
                                                       gd_add(bb0, 1.0))));
    } else {
        // ---- unstable, :681-715 ----------------------------------------------
        const double olinf = gd_div(z1, hlinf);
        if (fabs(olinf) <= gd_mul(F(50.f), z0max)) {
            hlinf = gd_div(-z1, gd_mul(F(50.f), z0max));
        }
        const double a0 = F(-3.975f), a1 = F(12.32f), b1 = F(-7.755f),
                     b2 = F(6.041f);
        const double a0p = F(-7.941f), a1p = F(24.75f), b1p = F(-8.705f),
                     b2p = F(7.899f);
        if (hlinf >= F(-.5f)) {
            hl1 = hlinf;
            pm = gd_div(gd_mul(gd_add(a0, gd_mul(a1, hl1)), hl1),
                        gd_add(gd_add(1.0, gd_mul(b1, hl1)),
                               gd_mul(gd_mul(b2, hl1), hl1)));
            ph = gd_div(gd_mul(gd_add(a0p, gd_mul(a1p, hl1)), hl1),
                        gd_add(gd_add(1.0, gd_mul(b1p, hl1)),
                               gd_mul(gd_mul(b2p, hl1), hl1)));
            const double hl110 = gd_div(gd_mul(hl1, F(10.f)), z1);
            pm10 = gd_div(gd_mul(gd_add(a0, gd_mul(a1, hl110)), hl110),
                          gd_add(gd_add(1.0, gd_mul(b1, hl110)),
                                 gd_mul(gd_mul(b2, hl110), hl110)));
            const double hl12 = gd_div(gd_mul(hl1, F(2.f)), z1);
            ph2 = gd_div(gd_mul(gd_add(a0p, gd_mul(a1p, hl12)), hl12),
                         gd_add(gd_add(1.0, gd_mul(b1p, hl12)),
                                gd_mul(gd_mul(b2p, hl12), hl12)));
        }
        if (hlinf < F(-.5f)) {
            hl1 = -hlinf;
            pm = gd_sub(gd_add(glibc_log(hl1),
                               gd_mul(F(2.f), gfs_neg_quarter(hl1))),
                        F(.8776f));
            ph = gd_add(gd_add(glibc_log(hl1),
                               gd_mul(F(.5f), gfs_neg_half(hl1))),
                        F(1.386f));
            const double hl110 = gd_div(gd_mul(hl1, F(10.f)), z1);
            pm10 = gd_sub(gd_add(glibc_log(hl110),
                                 gd_mul(F(2.f), gfs_neg_quarter(hl110))),
                          F(.8776f));
            const double hl12 = gd_div(gd_mul(hl1, F(2.f)), z1);
            ph2 = gd_add(gd_add(glibc_log(hl12),
                                gd_mul(F(.5f), gfs_neg_half(hl12))),
                         F(1.386f));
        }
    }

    // ---- :719-751 ------------------------------------------------------------
    fm = gd_sub(fm, pm);
    fh = gd_sub(fh, ph);
    fm10 = gd_sub(fm10, pm10);
    fh2 = gd_sub(fh2, ph2);
    const double ca2 = gd_mul(ca, ca);
#ifdef GPUWM_WRF_EXACT
    const double cm = gd_div(ca2, gd_mul(fm, fm));
    const double ch = gd_div(ca2, gd_mul(fm, fh));
    const double stress = gd_mul(gd_mul(cm, wind), wind);
    ustar = gd_sqrt(stress);
#else
    // cm = (ca/fm)^2 and sqrt(cm*wind^2) = abs(ca*wind/fm).
    // Share the momentum reciprocal with the 10 m wind below; retain the
    // native expression and every rounding in the replay arm.
    const double inv_fm = __drcp_rn(fm);
    const double ch = ca2 * inv_fm * __drcp_rn(fh);
    ustar = fabs(ca * wind * inv_fm);
#endif
    if (slimsk == 0.0) {
        z0 = gd_mul(gd_div(charnock, g), gd_mul(ustar, ustar));
        z0 = gd_min(z0, .1);                                  // _kind_phys
        z0 = gd_max(z0, 1.e-7);                               // _kind_phys
        z0rl = gd_mul(F(100.f), z0);
    }

    // ---- :1453-1457 ----------------------------------------------------------
    double f10m = gd_div(fm10, fm);
    f10m = gd_min(f10m, 1.0);
    const double u10m = gd_mul(gd_mul(f10m, xrcl), u1);
    const double v10m = gd_mul(gd_mul(f10m, xrcl), v1);

    // ---- SF_GFS :234-276 -----------------------------------------------------
    o.u10 = gd_narrow(u10m);
    o.v10 = gd_narrow(v10m);
    o.br = gd_narrow(rb);
    o.chs = gd_narrow(gd_mul(ch, wind));
    o.chs2 = gd_narrow(gd_div(gd_mul(ustar, D(k.karman)), fh2));
    o.cpm = gf_mul(k.cp, gf_add(1.f, gf_mul(.8f, in.qv1)));
    const float esat = gd_narrow(gfs_fpvs(tb, t1));
    o.qgh = gd_narrow(gd_div(D(gf_mul(k.ep2, esat)),
                             gd_sub(gd_mul(F(1000.f), ps), D(esat))));
    o.qsfc = gd_narrow(qss);
    o.psih = gd_narrow(ph);
    o.psim = gd_narrow(pm);
    ust = gd_narrow(ustar);
    o.wspd = gd_narrow(wind);
    znt = gd_narrow(gd_mul(z0rl, F(.01f)));

    o.flhc = gd_narrow(gd_mul(gd_mul(D(o.cpm), rho1), D(o.chs)));
    o.flqc = gd_narrow(gd_mul(rho1, D(o.chs)));
    o.gz1oz0 = gd_narrow((slimsk != 0.0) ? log_z1_z0
        : glibc_log(gd_div(z1, gd_mul(z0rl, F(.01f)))));
    o.cqs2 = o.chs2;

    if (k.isfflx == 0) {
        o.hfx = 0.f;
        o.lh = 0.f;
        o.qfx = 0.f;
    } else {
        const float dx = gf_sub(in.xland, 1.5f);
        if (dx > 0.f) {
            o.hfx = gd_narrow(gd_mul(D(o.flhc), gd_sub(thgb, thx)));
        } else if (dx < 0.f) {
            o.hfx = gd_narrow(gd_mul(D(o.flhc), gd_sub(thgb, thx)));
            o.hfx = gf_max(o.hfx, -250.f);
        }
        o.qfx = gd_narrow(gd_mul(D(o.flqc), gd_sub(D(o.qsfc), q1)));
        o.qfx = gf_max(o.qfx, 0.f);
        o.lh = gf_mul(k.xlv, o.qfx);
    }
}

__device__ __forceinline__ void gfs_store(const GfsOut& o, long long i,
        float* br, float* chs, float* chs2, float* cpm, float* cqs2,
        float* flhc, float* flqc, float* gz1oz0, float* hfx, float* lh,
        float* psim, float* psih, float* qfx, float* qgh, float* qsfc,
        float* u10, float* v10, float* wspd) {
    br[i] = o.br; chs[i] = o.chs; chs2[i] = o.chs2; cpm[i] = o.cpm;
    cqs2[i] = o.cqs2; flhc[i] = o.flhc; flqc[i] = o.flqc;
    gz1oz0[i] = o.gz1oz0; hfx[i] = o.hfx; lh[i] = o.lh; psim[i] = o.psim;
    psih[i] = o.psih; qfx[i] = o.qfx; qgh[i] = o.qgh; qsfc[i] = o.qsfc;
    u10[i] = o.u10; v10[i] = o.v10; wspd[i] = o.wspd;
}

// SF_GFS over n columns (FRACTIONAL_SEAICE = 0, module_surface_driver.F:2341).
// HFX is left untouched on a column whose XLAND is exactly 1.5, as WRF's
// IF/ELSEIF leaves it (module_sf_gfs.F:266-271).
extern "C" __global__ void gfs_sfclay(
        int n,
        const float* __restrict__ u1, const float* __restrict__ v1,
        const float* __restrict__ t1, const float* __restrict__ qv1,
        const float* __restrict__ p1, const float* __restrict__ psfc,
        const float* __restrict__ tsk, const float* __restrict__ xland,
        float* znt, float* ust,
        float* br, float* chs, float* chs2, float* cpm, float* cqs2,
        float* flhc, float* flqc, float* gz1oz0, float* hfx, float* lh,
        float* psim, float* psih, float* qfx, float* qgh, float* qsfc,
        float* u10, float* v10, float* wspd,
        const double* __restrict__ tb,
        float cp, float rovcp, float r, float xlv, float ep1, float ep2,
        float karman, int isfflx) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const GfsConst k = {cp, rovcp, r, xlv, ep1, ep2, karman, isfflx};
    const GfsIn in = {u1[i], v1[i], t1[i], qv1[i], p1[i], psfc[i], tsk[i],
                      xland[i]};
    GfsOut o;
    o.hfx = hfx[i];
    float z = znt[i], u = ust[i];
    gfs_sfclay_one(in, z, u, o, k, tb);
    znt[i] = z; ust[i] = u;
    gfs_store(o, i, br, chs, chs2, cpm, cqs2, flhc, flqc, gz1oz0, hfx, lh,
              psim, psih, qfx, qgh, qsfc, u10, v10, wspd);
}

// SF_GFS_SEAICE_WRAPPER (module_surface_driver.F:5564-5815) with
// get_local_ice_tsk (:7117-7208, SFC_LAYER_VALUES = .true.,
// TICE2TSK_IF2COLD = .false.), FRACTIONAL_SEAICE = 1.
//
// DECLARED DIVERGENCE (a WRF defect repaired, never transcribed): WRF builds
// XLAND_SEA = 2 for the open-water component of a sea-ice cell (:5751) and
// then passes XLAND to the open-water call (:5772), so that component runs
// as LAND -- no Charnock roughness, no Zeng thermal roughness, the land
// -250 W m-2 HFX floor -- on a cell real.exe marked XLAND = 1.  Every other
// *_seaice_wrapper in the same file passes XLAND_SEA.  This kernel passes
// it.  The oracle (tools/sfgfs_wrf461_oracle) records the shipped and the
// repaired call side by side.
//
// sst is WOOF's SST/TSK_SEA carrier (INOUT, like WRF's SST): ice cells
// receive get_local_ice_tsk's adjusted SST, which is also the open-water
// skin temperature the post-LSM reblend reads.  The *_sea outputs are the
// open-water call's results for every column, exactly as WRF leaves them.
extern "C" __global__ void gfs_sfclay_seaice(
        int n,
        const float* __restrict__ u1, const float* __restrict__ v1,
        const float* __restrict__ t1, const float* __restrict__ qv1,
        const float* __restrict__ p1, const float* __restrict__ psfc,
        const float* __restrict__ tsk, const float* __restrict__ xland,
        const float* __restrict__ xice, float* sst,
        float* znt, float* ust,
        float* br, float* chs, float* chs2, float* cpm, float* cqs2,
        float* flhc, float* flqc, float* gz1oz0, float* hfx, float* lh,
        float* psim, float* psih, float* qfx, float* qgh, float* qsfc,
        float* u10, float* v10, float* wspd,
        float* chs_sea, float* chs2_sea, float* cpm_sea, float* cqs2_sea,
        float* flhc_sea, float* flqc_sea, float* hfx_sea, float* lh_sea,
        float* qfx_sea, float* qgh_sea, float* qsfc_sea, float* ust_sea,
        float* znt_sea,
        const double* __restrict__ tb,
        float cp, float rovcp, float r, float xlv, float ep1, float ep2,
        float karman, int isfflx, float xice_threshold, int itimestep) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const GfsConst k = {cp, rovcp, r, xlv, ep1, ep2, karman, isfflx};
    const float xi = xice[i];
    const bool ice = (xi >= xice_threshold) && (xi <= 1.0f);
    float s = sst[i];
    float tsk_sea, tsk_ice;
    // ---- get_local_ice_tsk :7158-7206 ----------------------------------------
    if (ice) {
        if (s < 271.4f) s = 271.4f;
        if (s > 273.f && itimestep <= 3) {
            if (xi >= 0.6f) s = 271.4f;
            else if (xi >= 0.4f) s = 273.f;
            else if (xi >= 0.2f && s > 275.f) s = 275.f;
            else if (s > 278.f) s = 278.f;
        }
        tsk_sea = s;
        tsk_ice = gf_div(gf_sub(tsk[i], gf_mul(gf_sub(1.0f, xi), s)), xi);
        if (tsk_ice < 221.4f) tsk_ice = 221.4f;
        if (xi < 0.2f && tsk[i] < 253.15f) tsk_ice = 253.15f;
        if (xi < 0.1f && tsk[i] < 263.15f) tsk_ice = 263.15f;
    } else {
        tsk_sea = tsk[i];
        tsk_ice = tsk[i];
    }
    const float znt_hold = znt[i], ust_hold = ust[i];

    // ---- the ice / true-land call, :5734-5743 (CPM goes to CPM_SEA) ------------
    GfsIn in = {u1[i], v1[i], t1[i], qv1[i], p1[i], psfc[i], tsk_ice,
                xland[i]};
    GfsOut o;
    o.hfx = hfx[i];
    float z = znt[i], u = ust[i];
    gfs_sfclay_one(in, z, u, o, k, tb);

    // ---- open-water set-up, :5747-5765 ----------------------------------------
    float zs, us, xland_sea;
    if (ice) {
        xland_sea = 2.f;
        zs = 0.0001f;
        if (s < 271.4f) s = 271.4f;
        tsk_sea = s;
        us = ust_sea[i];
    } else {
        xland_sea = xland[i];
        zs = znt_hold;
        us = ust_hold;
        tsk_sea = tsk[i];
    }
    // ---- the open-water call, :5769-5778, with XLAND_SEA (see header) --------
    GfsIn sea_in = in;
    sea_in.tsk = tsk_sea;
    sea_in.xland = xland_sea;
    GfsOut os;
    os.hfx = hfx_sea[i];
    gfs_sfclay_one(sea_in, zs, us, os, k, tb);

    // ---- the ice-fraction blend, :5782-5813 -----------------------------------
    if (ice) {
        const float omx = gf_sub(1.0f, xi);
        o.br = gf_add(gf_mul(o.br, xi), gf_mul(omx, os.br));
        o.gz1oz0 = gf_add(gf_mul(o.gz1oz0, xi), gf_mul(omx, os.gz1oz0));
        o.psim = gf_add(gf_mul(o.psim, xi), gf_mul(omx, os.psim));
        o.psih = gf_add(gf_mul(o.psih, xi), gf_mul(omx, os.psih));
        o.u10 = gf_add(gf_mul(o.u10, xi), gf_mul(omx, os.u10));
        o.v10 = gf_add(gf_mul(o.v10, xi), gf_mul(omx, os.v10));
        o.wspd = gf_add(gf_mul(o.wspd, xi), gf_mul(omx, os.wspd));
    }
    // WRF's argument swap (:5735 hands the first call CPM_SEA, :5770 hands
    // the second CPM) is transcribed as written: both calls compute CPM from
    // the same QV, so the two words are equal either way.
    const float cpm_first = o.cpm;
    o.cpm = os.cpm;
    gfs_store(o, i, br, chs, chs2, cpm, cqs2, flhc, flqc, gz1oz0, hfx, lh,
              psim, psih, qfx, qgh, qsfc, u10, v10, wspd);
    znt[i] = z; ust[i] = u;
    chs_sea[i] = os.chs; chs2_sea[i] = os.chs2; cpm_sea[i] = cpm_first;
    cqs2_sea[i] = os.cqs2; flhc_sea[i] = os.flhc; flqc_sea[i] = os.flqc;
    hfx_sea[i] = os.hfx; lh_sea[i] = os.lh; qfx_sea[i] = os.qfx;
    qgh_sea[i] = os.qgh; qsfc_sea[i] = os.qsfc;
    ust_sea[i] = us; znt_sea[i] = zs;
    if (ice) sst[i] = s;
}
