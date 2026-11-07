// YSU's topo_wind arm (kernels/ysu.cu ysu_column_topo), prepended to
// ysu.cu by the kernel loader (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS["ysu"]) after glibc_flt32.cuh.

// --------------------------------------------------------------------------
// topo_wind arm (WRF v4.7.1 &physics topo_wind = 1 or 2): bl_ysu.F90 with
// ctopo/ctopo2 present, MMM-physics 20240626-MPASv8.2.
//   ctopo  scales the first-level surface drag in the momentum solve,
//          weighted by how convective the column is (:1254-1314: the paj TKE
//          profile, the hybrid PBL height of get_pblh, and the Beljaars
//          convective velocity vconv);
//   ctopo2 blends the 10 m wind toward the first-level wind on hill tops
//          (:1402-1408), which the kernel writes to u10o/v10o.
// Both come from gpuwm/core/kernels/terrain_drag.cu::topo_wind_static
// (start_em.F:1579-1626).  Everything in ysu.cu is `if constexpr`; since
// lane/parity-286 every entry point instantiates TOPO = true (see below).
// This header holds the arm's own pieces so that ysu.cu's
// lines before its momentum assembly keep their numbers (the physics
// registry and the FTZ claim census cite them).
// The new statements spell every product and sum as an IEEE intrinsic (WRF
// rounds each; this module lets NVRTC fuse otherwise) and take WOOF's own powf
// (gfk_pow) and get_pblh's tanh as ysu_tanhf below.
//
// The DEFAULT kernel (ysu_column) takes this arm too, with ctopo = ctopo2 = 1:
// WRF's driver passes both on every column of every run
// (module_bl_ysu.F:404, Registry default 1.0), topo_wind or not, so
// bl_ysu.F90:1308 is the surface-drag diagonal of every default WRF run and
// the ctopo-absent :1315 (`ad(1) = 1+fric`) is reached by no WRF run at all.
// A YsuTopo with null pointers means exactly that: ctopo reads as 1.0f and
// the :1402-1408 blend, which is u10*1 + 0*ux(1) = u10, writes nothing.
struct YsuTopo {
    const real *ctopo, *ctopo2;
    real *u10o, *v10o;
};

// WRF's thx, bl_ysu.F90:419 `thx(i,k) = tx(i,k)/pi2d(i,k)`, where the driver's
// tx is phy_prep's t_phy = th_phy*pi_phy.  So WRF's YSU never sees the model's
// theta: it sees (theta*pi)/pi, which can sit one ULP away, and its potential-
// temperature tendency leaves as (ttend*pi2d)/pi2d (module_bl_ysu.F:452 with
// bl_ysu.F90:1102).  The kernel spells both round trips so that, in the
// model, every theta it reads and the tendency it writes are WRF's words.
__device__ __forceinline__ real ysu_thx(const real *theta, const real *exner,
                                        int q) {
    return __fdiv_rn(__fmul_rn(theta[q], exner[q]), exner[q]);
}

// A divisor WRF uses as it stands: bl_ysu.F90 divides by wspd (:693, :895,
// :1306-1307) and by wscale**4 (:693, :895) with no floor.  The kernel used
// to floor them at 1e-9 and 1e-20, which changed the answer on every finite
// input below the floor (the wide oracle's wspd = 1e-10 probe: du/dv 10x
// WRF's).  Now the value passes through untouched whenever it is positive,
// and the floor applies only where WRF would divide by zero (or by a
// negative wspd, which no surface layer writes), so the kernel keeps a
// defined value exactly where WRF's own result is Inf/NaN.  A NaN passes
// through, as ysu_max lets it, so a poisoned input still reaches
// validate_ysu_outputs.
__device__ __forceinline__ real ysu_pos(real x, real fallback) {
    return (x > 0.0f || x != x) ? x : fallback;
}

// Subnormal-preserving float -> double (shinhong.cu sh_f2d, rrtmg_sw.cu
// rsw_f2d).  bl_ysu.F90:613 `if(br(i).gt.0.0) sfcflg(i) = .false.` must see a
// positive subnormal br as positive; sm_120 DAZes FP32 subnormals in every
// float32 operation including compares, and the cvt.f64.f32 a plain
// `(double)br > 0.0` emits DAZes its input too, so the subnormal is decoded
// from its bits.  A NaN br compares false here, as in WRF, so sfcflg stays
// true (WRF initialises it .true. and only clears it on br > 0).
__device__ __forceinline__ double ysu_f2d(real x) {
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix >> 31) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

// tanhf for get_pblh's weight, :1684 `wt=.5*TANH((zi - sbl_lim)/sbl_damp) + .5`.
//
// WHY NOT tanh() IN DOUBLE.  Until lane/verify-ysu this read
// __double2float_rn(tanh((double)x)), the correctly rounded tanh, which is
// what glibc 2.43 returns (the topo_wind oracle's platform,
// tests/test_terrain_drag_wrf471_parity.py).  The reference platform of this
// verification (Ubuntu 24.04, glibc 2.39; the MYNN oracles' platform too)
// returns the classic expm1f-based tanhf instead, and the two disagree on
// 113,763,220 of the 2,149,580,801 floats in [-0.5, 10] (5.3%), measured on
// the reference host.  get_pblh's weight sets pblh_ysu, which sets vconvlim
// and so the surface-drag diagonal on every column where vconvlim < 1: the
// wide oracle's 20,000-column sweep showed six columns off WRF in du/dv (up
// to 819 ULP) and nowhere else until this function replaced the double tanh.
// ysu_tanhf and ysu_expm1f are WOOF's own float32 code: the same algorithm
// and the same words as the MYNN unit's mynn_tanhf/mynn_expm1f
// (kernels/mynn_pbl.cu), held to the reference platform's tanhf over every
// float in [-0.5, 22] by tools/ysu_wrf461_oracle/tanhf_sweep.py and to a
// stored sample of it by tests/test_ysu_wrf461_wide_parity.py.
__device__ __forceinline__ real ysu_scale_exponent(real y, int k)
{
    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));
}

__device__ real ysu_expm1f(real x)
{
    const real ln2_hi = __uint_as_float(0x3F317180u);
    const real ln2_lo = __uint_as_float(0x3717F7D1u);
    const real invln2 = __uint_as_float(0x3FB8AA3Bu);
    const real q1 = __uint_as_float(0xBD088889u);
    const real q2 = __uint_as_float(0x3AD00D01u);
    const real q3 = __uint_as_float(0xB8A670CDu);
    const real q4 = __uint_as_float(0x36867E54u);
    const real q5 = __uint_as_float(0xB457EDBBu);
    const real tiny = 1.0e-30f;

    unsigned word = __float_as_uint(x);
    unsigned sign = word & 0x80000000u;
    unsigned magnitude = word & 0x7FFFFFFFu;
    if (magnitude >= 0x4195B844u) {              // |x| >= 27*ln2
        if (magnitude >= 0x42B17218u) {          // |x| >= 88.72
            if (magnitude > 0x7F800000u) return __fadd_rn(x, x);
            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);
        }
        if (sign != 0u) return __fsub_rn(tiny, 1.0f);
    }
    int k;
    real correction;
    if (magnitude > 0x3EB17218u) {               // |x| > 0.5*ln2
        real hi, lo;
        if (magnitude < 0x3F851592u) {           // |x| < 1.5*ln2
            if (sign == 0u) {
                hi = __fsub_rn(x, ln2_hi); lo = ln2_lo; k = 1;
            } else {
                hi = __fadd_rn(x, ln2_hi); lo = -ln2_lo; k = -1;
            }
        } else {
            k = (int)__fadd_rn(__fmul_rn(invln2, x), sign == 0u ? 0.5f : -0.5f);
            real scale = (real)k;
            hi = __fsub_rn(x, __fmul_rn(scale, ln2_hi));
            lo = __fmul_rn(scale, ln2_lo);
        }
        x = __fsub_rn(hi, lo);
        correction = __fsub_rn(__fsub_rn(hi, x), lo);
    } else if (magnitude < 0x33000000u) {        // |x| < 2**-25
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    real hfx = __fmul_rn(0.5f, x);
    real hxs = __fmul_rn(x, hfx);
    real r1 = __fadd_rn(1.0f, __fmul_rn(hxs, __fadd_rn(q1, __fmul_rn(hxs,
        __fadd_rn(q2, __fmul_rn(hxs, __fadd_rn(q3, __fmul_rn(hxs,
            __fadd_rn(q4, __fmul_rn(hxs, q5))))))))));
    real t = __fsub_rn(3.0f, __fmul_rn(r1, hfx));
    real e = __fmul_rn(hxs, __fdiv_rn(__fsub_rn(r1, t),
                                      __fsub_rn(6.0f, __fmul_rn(x, t))));
    if (k == 0) return __fsub_rn(x, __fsub_rn(__fmul_rn(x, e), hxs));
    e = __fsub_rn(__fmul_rn(x, __fsub_rn(e, correction)), correction);
    e = __fsub_rn(e, hxs);
    if (k == -1) return __fsub_rn(__fmul_rn(0.5f, __fsub_rn(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return __fmul_rn(-2.0f, __fsub_rn(e, __fadd_rn(x, 0.5f)));
        return __fadd_rn(1.0f, __fmul_rn(2.0f, __fsub_rn(x, e)));
    }
    real y;
    if (k <= -2 || k > 56) {
        y = __fsub_rn(1.0f, __fsub_rn(e, x));
        y = ysu_scale_exponent(y, k);
        return __fsub_rn(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));
        y = __fsub_rn(t, __fsub_rn(e, x));
    } else {
        t = __uint_as_float((unsigned)(0x7F - k) << 23);
        y = __fsub_rn(x, __fadd_rn(e, t));
        y = __fadd_rn(y, 1.0f);
    }
    return ysu_scale_exponent(y, k);
}

// 1 - 2/(expm1f(2|x|)+2) at |x| >= 1, -t/(t+2) below, saturated at 22.
__device__ real ysu_tanhf(real x)
{
    const real tiny = 1.0e-30f;
    unsigned word = __float_as_uint(x);
    unsigned magnitude = word & 0x7FFFFFFFu;
    real z;
    if (magnitude >= 0x7F800000u)                // inf or NaN: one/x +- one
        return (word & 0x80000000u) == 0u
            ? __fadd_rn(__fdiv_rn(1.0f, x), 1.0f)
            : __fsub_rn(__fdiv_rn(1.0f, x), 1.0f);
    if (magnitude < 0x41B00000u) {               // |x| < 22
        if (magnitude < 0x24000000u)             // |x| < 2**-55
            return __fmul_rn(x, __fadd_rn(1.0f, x));
        real ax = __uint_as_float(magnitude);
        if (magnitude >= 0x3F800000u) {          // |x| >= 1
            real t = ysu_expm1f(__fmul_rn(2.0f, ax));
            z = __fsub_rn(1.0f, __fdiv_rn(2.0f, __fadd_rn(t, 2.0f)));
        } else {
            real t = ysu_expm1f(__fmul_rn(-2.0f, ax));
            z = __fdiv_rn(-t, __fadd_rn(t, 2.0f));
        }
    } else {
        z = __fsub_rn(1.0f, tiny);
    }
    return (word & 0x80000000u) == 0u ? z : -z;
}

// bl_ysu.F90 get_pblh (:1586-1693, "Copied from MYNN PBL"), one column, on
// the kernel's 0-based workspace: thv[k] is thetav1d(k+1), tke[k] is
// qke1d(k+1) (written for k <= nz-2, which is all WRF writes and all the
// two searches below can reach), zq[k] is zw1d(k+1), and dz1d(k) is WRF's
// dzq(i,k) = zq(i,k+1)-zq(i,k) (:522), formed here from zq the same way.
template <class Col>
__device__ real ysu_get_pblh(Col thv, Col tke, Col zq, int nz, real landsea) {
    const real sbl_lim = 200.0f, sbl_damp = 400.0f;
    // Fortran index helpers (1-based, kts = 1, kte = nz).
    #define YT_THV(k) thv[(k) - 1]
    #define YT_QKE(k) tke[(k) - 1]
    #define YT_ZW(k) zq[(k) - 1]
    #define YT_DZ(k) __fsub_rn(zq[(k)], zq[(k) - 1])
    const int kte = nz, kts = 1;
    int k = kts + 1, kthv = 1, ktke = 1;
    real maxqke = 0.0f, minthv = 9.e9f;
    while (k <= kte && YT_ZW(k) <= 500.0f) {
        real qtke = fmaxf(YT_QKE(k), 0.0f);
        if (maxqke < qtke) { maxqke = qtke; ktke = k; }
        if (minthv > YT_THV(k)) { minthv = YT_THV(k); kthv = k; }
        k = k + 1;
    }
    real tkeeps = __fdiv_rn(maxqke, 40.0f);
    tkeeps = fmaxf(tkeeps, 0.025f);
    tkeeps = fminf(tkeeps, 0.25f);
    real delt_thv = (__fsub_rn(landsea, 1.5f) >= 0.0f) ? 0.75f : 1.5f;
    real zi = 0.0f;
    k = kthv + 1;
    real thr = __fadd_rn(minthv, delt_thv);
    while (zi == 0.0f && k <= kte) {
        if (YT_THV(k) >= thr) {
            real frac = fminf(__fdiv_rn(__fsub_rn(YT_THV(k), thr),
                                        fmaxf(__fsub_rn(YT_THV(k), YT_THV(k - 1)),
                                              1e-6f)), 1.0f);
            zi = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
        }
        k = k + 1;
        if (k == kte - 1) zi = YT_ZW(kts + 1);
    }
    real pblh_tke = 0.0f;
    k = ktke + 1;
    while (pblh_tke == 0.0f && k <= kte) {
        real qtke = fmaxf(__fdiv_rn(YT_QKE(k), 2.0f), 0.0f);
        real qtkem1 = fmaxf(__fdiv_rn(YT_QKE(k - 1), 2.0f), 0.0f);
        if (qtke <= tkeeps) {
            real frac = fminf(__fdiv_rn(__fsub_rn(tkeeps, qtke),
                                        fmaxf(__fsub_rn(qtkem1, qtke), 1e-6f)),
                              1.0f);
            pblh_tke = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
            pblh_tke = fmaxf(pblh_tke, YT_ZW(kts + 1));
        }
        k = k + 1;
        if (k == kte - 1) pblh_tke = YT_ZW(kts + 1);
    }
    real th = ysu_tanhf(__fdiv_rn(__fsub_rn(zi, sbl_lim), sbl_damp));
    real wt = __fadd_rn(__fmul_rn(0.5f, th), 0.5f);
    #undef YT_THV
    #undef YT_QKE
    #undef YT_ZW
    #undef YT_DZ
    return __fadd_rn(__fmul_rn(pblh_tke, __fsub_rn(1.0f, wt)),
                     __fmul_rn(zi, wt));
}

// bl_ysu.F90:1402-1408: u10 = ctopo2*u10 + (1-ctopo2)*ux(1), the same for v,
// with ux(1), vx(1) the first-level wind the call was handed.  ctopo2 = 1
// everywhere but on hill tops.
__device__ __forceinline__ void ysu_topo_blend_u10(
        const YsuTopo &topo, const real *u10, const real *v10, real u1,
        real v1, int col) {
    if (topo.u10o == nullptr) return;   // ctopo2 = 1: u10*1 + 0*ux(1) = u10
    real c2 = topo.ctopo2[col];
    topo.u10o[col] = __fadd_rn(__fmul_rn(c2, u10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), u1));
    topo.v10o[col] = __fadd_rn(__fmul_rn(c2, v10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), v1));
}

