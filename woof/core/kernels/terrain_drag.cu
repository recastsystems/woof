// Sub-grid terrain drag from WRF v4.7.1: the topo_wind static coefficients
// and the two orographic drag schemes gwd_opt = 1 and gwd_opt = 3.
//
//   topo_wind_static  dyn_em/start_em.F:1539-1626.  LAP_HGT from the terrain
//                     and CTOPO / CTOPO2 for topo_wind = 1 (Jimenez and
//                     Dudhia 2012: drag scaled by ln of the sub-grid terrain
//                     deviation, tapered to zero on hill tops) and
//                     topo_wind = 2 (drag scaled by (VAR*0.4/200+1.175)^2,
//                     capped at 1.575^2).  YSU reads them (kernels/ysu.cu).
//   gwdo_column       gwd_opt = 1: phys/module_bl_gwdo.F (gwdo) calling
//                     phys/physics_mmm/bl_gwdo.F90 (bl_gwdo_run), the KIM
//                     orographic gravity-wave drag with flow blocking
//                     (Choi and Hong 2015).
//   gwdo_gsl_column   gwd_opt = 3: phys/module_bl_gwdo_gsl.F, the GSL drag
//                     suite: large-scale gravity-wave drag and flow
//                     blocking (tapered off between 13 and 3 km), small-scale
//                     gravity-wave drag (Steeneveld et al. 2008) and
//                     turbulent orographic form drag (Beljaars et al. 2004),
//                     both tapered off between 12 and 1 km.
//
// Every statement follows WRF's own, in WRF's order of evaluation; the module
// is compiled with -fmad=false so no product is fused into a sum, as gfortran
// fuses none.  Fortran's real powers (x**2., x**(-1.2), ...) are glibc's
// powf (gfk_pow; gfortran calls powf for a real exponent even when it is
// 2.0), integer powers are products, expf and logf are glibc's (gfk_exp,
// gfk_log), sinf is glibc's (glibc_sinf).  atan2f is td_atan2f below, WOOF's
// own float32 atan2f, whose words are the reference platform's (Ubuntu 24.04,
// glibc 2.39, x86-64): see its comment.  The oracles that hold these kernels
// to WRF are tools/terrain_drag_wrf471_oracle
// (tests/test_terrain_drag_wrf471_parity.py) and, for gwd_opt = 1 on the
// reference platform, tools/gwdo_kim_wrf461_oracle
// (tests/test_gwdo_kim_wrf461_parity.py).
//
// Layout: 3-D fields are (nz, ny, nx) mass levels (p_interface nz+1), 2-D
// fields (ny, nx); the four directional statistics OA1-4 and OL1-4 are
// (4, ny, nx).  One thread owns one column.

#define TD_KMAX 128

__device__ __forceinline__ float td_max(float a, float b)
{ return fmaxf(a, b); }
__device__ __forceinline__ float td_min(float a, float b)
{ return fminf(a, b); }

// atan2f for the wind-sector choice, bl_gwdo.F90:338 and
// module_bl_gwdo_gsl.F's `wdir = atan2(ubar,vbar) + pi`: the angle only picks
// one of eight sectors (which OA/OL statistic and grid length the drag uses),
// so its last bit matters only on a sector boundary, and there it decides
// the whole column.
//
// WHY NOT atan2() IN DOUBLE.  Until lane/verify-gwdo-kim this read
// __double2float_rn(atan2((double)y, (double)x)), the correctly rounded
// angle, which is what glibc 2.43 returns (the WRF v4.7.1 oracle's
// platform).  The reference platform of this verification (Ubuntu 24.04,
// glibc 2.39; the MYNN, YSU and Kessler oracles' platform) returns the
// classic atanf-based atan2f instead, 1 ULP off the correctly rounded angle
// on 16% of random wind pairs (measured on x86-64), and on a sector
// boundary that 1 ULP picks the other sector: in the WRF v4.6.1 column
// oracle (tests/data/oracles/gwdo_kim) one sector-boundary column per grid
// took idir 1 under WRF and 8 under the double atan2 (50 levels), 5 and 4
// (90 levels), and every drag word of those columns differed (up to
// 0.41 m/s2).
//
// td_atanf and td_atan2f are WOOF's own float32 code: td_atanf is the same
// algorithm and the same words as the MYNN unit's mynn_glibc_atanf
// (kernels/mynn_pbl.cu), and td_atan2f reduces to it through one IEEE
// division y/x and a quadrant fix-up with float32 pi constants.  They are
// held to the reference platform's atanf over EVERY float32 and to its
// atan2f on boundary, special and random pairs by
// tools/gwdo_kim_wrf461_oracle/sector_sweep.py.  Every operation is one
// plain float32 operation; the module compiles with -fmad=false.
__constant__ unsigned int TD_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,   // atan hi parts
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,   // atan lo parts
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,   // odd series
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__device__ float td_atanf(float x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int)hx;
    int id;
#define TD_ATAN_HI(i) __uint_as_float(TD_ATANF_TAB[(i)])
#define TD_ATAN_LO(i) __uint_as_float(TD_ATANF_TAB[4 + (i)])
#define TD_ATAN_T(i)  __uint_as_float(TD_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return x + x;      // NaN
        if (signed_hx > 0) return TD_ATAN_HI(3) + TD_ATAN_LO(3);
        return -TD_ATAN_HI(3) - TD_ATAN_LO(3);
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);                 // |x|
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0f * x - 1.0f) / (2.0f + x);
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0f) / (x + 1.0f);
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = (x - 1.5f) / (1.0f + 1.5f * x);
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = -1.0f / x;
        }
    }
    float z = x * x;
    float w = z * z;
    float s1 = w * TD_ATAN_T(10);
    s1 = w * (TD_ATAN_T(8) + s1);
    s1 = w * (TD_ATAN_T(6) + s1);
    s1 = w * (TD_ATAN_T(4) + s1);
    s1 = w * (TD_ATAN_T(2) + s1);
    s1 = z * (TD_ATAN_T(0) + s1);
    float s2 = w * TD_ATAN_T(9);
    s2 = w * (TD_ATAN_T(7) + s2);
    s2 = w * (TD_ATAN_T(5) + s2);
    s2 = w * (TD_ATAN_T(3) + s2);
    s2 = w * (TD_ATAN_T(1) + s2);
    float s = s1 + s2;
    if (id < 0) return x - x * s;
    float r = TD_ATAN_HI(id) - ((x * s - TD_ATAN_LO(id)) - x);
    return (signed_hx < 0) ? -r : r;
#undef TD_ATAN_HI
#undef TD_ATAN_LO
#undef TD_ATAN_T
}

__device__ float td_atan2f(float y, float x)
{
    const float tiny = 1.0e-30f;
    const float pi_o_4 = __uint_as_float(0x3F490FDBu);
    const float pi_o_2 = __uint_as_float(0x3FC90FDBu);
    const float pi = __uint_as_float(0x40490FDBu);
    const float pi_lo = __uint_as_float(0xB3BBBD2Eu);   // pi - float(pi)
    int hx = __float_as_int(x), hy = __float_as_int(y);
    int ix = hx & 0x7FFFFFFF, iy = hy & 0x7FFFFFFF;
    if (ix > 0x7F800000 || iy > 0x7F800000) return x + y;   // NaN
    if (hx == 0x3F800000) return td_atanf(y);               // x = 1
    int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);            // 2*sign(x)+sign(y)
    if (iy == 0) {                                          // y = +-0
        if (m <= 1) return y;
        return (m == 2) ? pi + tiny : -pi - tiny;
    }
    if (ix == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    if (ix == 0x7F800000) {                                 // x = +-Inf
        if (iy == 0x7F800000) {
            switch (m) {
            case 0: return pi_o_4 + tiny;
            case 1: return -pi_o_4 - tiny;
            case 2: return 3.0f * pi_o_4 + tiny;
            default: return -3.0f * pi_o_4 - tiny;
            }
        }
        switch (m) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return pi + tiny;
        default: return -pi - tiny;
        }
    }
    if (iy == 0x7F800000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
    int k = (iy - ix) >> 23;
    float z;
    if (k > 60) z = pi_o_2 + 0.5f * pi_lo;                  // |y/x| > 2**60
    else if (hx < 0 && k < -60) z = 0.0f;                   // |y|/x < -2**60
    else z = td_atanf(fabsf(y / x));
    switch (m) {
    case 0: return z;
    case 1: return -z;
    case 2: return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
    }
}

// Fortran NINT: round half away from zero.
__device__ __forceinline__ int td_nint(float x) { return (int)roundf(x); }

// --------------------------------------------------------------------------
// topo_wind: LAP_HGT, CTOPO, CTOPO2 (start_em.F:1539-1626)
// --------------------------------------------------------------------------
extern "C" __global__
void topo_wind_static(const real *ht, const real *var_sso, const real *var2d,
                      const real *xland, real *lap_hgt, real *ctopo,
                      real *ctopo2, int nx, int ny, int topo_wind)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;
    int i = idx % nx, j = idx / nx;
    // :1542-1555, the non-periodic arm: neighbours clamp to the domain.
    int im1 = max(i - 1, 0), ip1 = min(i + 1, nx - 1);
    int jm1 = max(j - 1, 0), jp1 = min(j + 1, ny - 1);
    real h = ht[idx];
    // :1558  (ht(ip1)+ht(im1)+ht(jp1)+ht(jm1)-ht*4.)/4.
    real lap = (((ht[j * nx + ip1] + ht[j * nx + im1]) + ht[jp1 * nx + i])
                + ht[jm1 * nx + i]) - h * 4.0f;
    lap = lap / 4.0f;
    lap_hgt[idx] = lap;
    // :1581-1582
    real c = 1.0f, c2 = 1.0f;
    if (topo_wind == 1) {
        // :1587-1592
        if (xland[idx] < 1.5f) c = sqrtf(var_sso[idx]);
        if (c <= 2.718f) c = 1.0f;
        else c = gfk_log(c);
        // :1594-1608
        if (lap > -10.0f) {
            // ctopo unchanged
        } else if (lap >= -20.0f) {
            real alpha = (lap + 20.0f) / 10.0f;
            c = alpha * c + (1.0f - alpha);
        } else if (lap >= -30.0f) {
            c = (lap + 30.0f) / 10.0f;
            c2 = (lap + 30.0f) / 10.0f;
        } else {
            c = 0.0f;
            c2 = 0.0f;
        }
    } else if (topo_wind == 2) {
        // :1615-1623
        real vfac;
        if (xland[idx] < 1.5f) {
            real x = var2d[idx] * 0.4f / 200.0f + 1.175f;
            vfac = td_min(1.575f, x);
            vfac = vfac * vfac;
        } else {
            vfac = 1.0f;
        }
        c = c * vfac;
    }
    ctopo[idx] = c;
    ctopo2[idx] = c2;
}

// The SASE bulk-Richardson height is not a WRF PBL package output.  Derive
// GSL's one-based upper-bracket mass level from that diagnosed height,
// using the height diagnostic's FP64 cumulative layer-center arithmetic.
extern "C" __global__
void terrain_pbl_top(const real *dz, const real *pblh, int *kpbl,
                     int nz, int ny, int nx)
{
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ncol = (size_t)ny * nx;
    if (col >= ncol) return;
    double depth = 0.0;
    int top = nz;
    for (int k = 0; k < nz; ++k) {
        double thick = (double)dz[(size_t)k * ncol + col];
        double center = depth + 0.5 * thick;
        depth += thick;
        // Round as the height diagnostic rounds its output.  This keeps
        // its exact first-interior/top-center fallbacks at their level.
        if (k >= 1 && (real)center >= pblh[col]) {
            top = k + 1;
            break;
        }
    }
    kpbl[col] = top;
}

// --------------------------------------------------------------------------
// gwd_opt = 1: bl_gwdo_run (phys/physics_mmm/bl_gwdo.F90), through WRF's
// wrapper module_bl_gwdo.F::gwdo (rublten += rotated drag).
// --------------------------------------------------------------------------
// The scheme's per-level arrays are recomputed from the inputs where it reads
// them rather than stored: every recomputation is the same statement on the
// same words, so the value is the one WRF stored.  usqj below the blending
// height and the averaged usqj(1) are written by WRF and never read again, so
// they are not formed.
extern "C" __global__
void gwdo_column(const real *u, const real *v, const real *t,
                 const real *qv, const real *p, const real *p_interface,
                 const real *pi, const real *z,
                 const real *var, const real *oc1, const real *oa4,
                 const real *ol4, const real *sina, const real *cosa,
                 real *du, real *dv,
                 real *dtaux3d, real *dtauy3d, real *dusfcg, real *dvsfcg,
                 real dxmeter, real deltim, real g_, real cp_, real rd_,
                 real fv_, real pi_, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const real ric = 0.25f, dw2min = 1.0f, rimin = -100.0f, bnv2min = 1.0e-5f;
    const real efmin = 0.0f, efmax = 10.0f, gmax = 1.0f;
    const real veleps = 1.0f, frc = 1.0f, ce = 0.8f, cg = 0.5f;
    const int kpblmin = 2;
    const real frmax = 10.0f, olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const int kte = nz;                    // Fortran kts = 1 .. kte = nz
    const int mdir = 8;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    real ca = cosa[col], sa = sina[col];

    // Fortran-indexed accessors (k = 1..nz).
    #define Q(k) ((size_t)((k) - 1) * st + col)
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv_ * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RHO(k) (1.0f / rd_ * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(size_t)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (z[Q(k)])

    const real fdir = (real)mdir / (2.0f * pi_);
    real delx = dxmeter, dely = dxmeter;
    real dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrtf(gfk_pow(delx, 2.0f) + gfk_pow(dely, 2.0f));
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];
    real cleff = dxmeter;
    real varc = var[col];

    // klowtop: first level whose height above the lowest reaches 2*var.
    real zlowtop = 2.0f * varc;
    int klowtop = 0;
    if (zlowtop > 0.0f) {
        for (int k = 2; k <= kte; ++k) {
            if (ZL(k) - ZL(1) >= zlowtop) { klowtop = k + 1; break; }
        }
    }
    const int kpblmax = kte;
    int kbl = klowtop;
    kbl = max(min(kbl, kpblmax), kpblmin);
    real delks = 1.0f / (PRSI(1) - PRSI(kbl));
    real delks1 = 1.0f / (PRSL(1) - PRSL(kbl));
    real ubar = 0.0f, vbar = 0.0f, rhobar = 0.0f;
    for (int k = 1; k <= kpblmax; ++k) {
        if (k < kbl) {
            real rcsks = DEL(k) * delks;
            real rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            rhobar = rhobar + rdelks * RHO(k);
        }
    }
    // wind direction sector
    real oa, ol, olp, od, dxy, dxyp;
    {
        real oa4c[4], ol4c[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(size_t)m * st + col];
            ol4c[m] = ol4[(size_t)m * st + col];
        }
        real wdir = td_atan2f(ubar, vbar) + pi_;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        oa = (real)(1 - 2 * ((nwd - 1) / 4)) * oa4c[m];
        ol = ol4c[m];
        real ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = ol4p[m];
        od = olp / td_max(ol, olmin);
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

    // usqj(k), bnv2(k) for k = 1..kte-1 (bnv2 unclamped in this scheme);
    // bnv2(kte) keeps its initial 0.
    #define TI(k) (2.0f / (t[Q(k)] + t[Q((k) + 1)]))
    #define RDZ(k) (1.0f / (ZL((k) + 1) - ZL(k)))
    auto usqj_at = [&](int k) -> real {
        real ti = TI(k);
        real rdz = RDZ(k);
        real tem1 = U1(k) - U1(k + 1);
        real tem2 = V1(k) - V1(k + 1);
        real dw2 = tem1 * tem1 + tem2 * tem2;
        real shr2 = td_max(dw2, dw2min) * rdz * rdz;
        real bvf2 = g_ * (g_ / cp_ + rdz * (VTJ(k + 1) - VTJ(k))) * ti;
        return td_max(bvf2 / shr2, rimin);
    };
    auto bnv2_at = [&](int k) -> real {
        if (k >= kte) return 0.0f;
        real rdz = RDZ(k);
        return 2.0f * g_ * rdz * (VTK(k + 1) - VTK(k))
               / (VTK(k + 1) + VTK(k));
    };

    real ulow = td_max(sqrtf(ubar * ubar + vbar * vbar), 1.0f);
    real rulow = 1.0f / ulow;
    auto velco_at = [&](int k) -> real {
        real vc = 0.5f * ((U1(k) + U1(k + 1)) * ubar
                          + (V1(k) + V1(k + 1)) * vbar);
        vc = vc * rulow;
        if (vc < veleps && vc > 0.0f) vc = veleps;
        return vc;
    };

    bool ldrag = velco_at(1) <= 0.0f;
    for (int k = kpblmin; k <= kpblmax; ++k)
        if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;
    real wtkbj = (PRSL(1) - PRSL(2)) * delks1;
    real bnv2_1 = wtkbj * bnv2_at(1);
    for (int k = kpblmin; k <= kpblmax; ++k) {
        if (k < kbl) {
            real rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
            bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
        }
    }
    ldrag = ldrag || bnv2_1 <= 0.0f;
    ldrag = ldrag || ulow == 1.0f;
    ldrag = ldrag || varc <= 0.0f;

    real bnv = 0.0f, fr = 0.0f, xn = 0.0f, yn = 0.0f;
    if (!ldrag) {
        bnv = sqrtf(bnv2_1);
        fr = bnv * rulow * varc * od;
        fr = td_min(fr, frmax);
        xn = ubar * rulow;
        yn = vbar * rulow;
    }
    real taub, coefm = 0.0f;
    if (!ldrag) {
        real efact = gfk_pow(oa + 2.0f, ce * fr / frc);
        efact = td_min(td_max(efact, efmin), efmax);
        coefm = gfk_pow(1.0f + ol, oa + 1.0f);
        real xlinv = coefm / cleff;
        real tem = fr * fr * oc1[col];
        real gfobnv = gmax * tem / ((tem + cg) * bnv);
        taub = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact;
    } else {
        taub = 0.0f;
        xn = 0.0f;
        yn = 0.0f;
    }

    // Vertical structure of the stress.
    real taup[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) taup[k - 1] = 0.0f;
    for (int k = 1; k <= kpblmax; ++k)
        if (k <= kbl) taup[k - 1] = taub;
    bool icrilv = false;
    real brvf = 0.0f;
    for (int k = kpblmin; k <= kte - 1; ++k) {
        int kp1 = k + 1;
        if (k >= kbl) {
            real usqj = usqj_at(k);
            icrilv = icrilv || (usqj < ric) || (velco_at(k) <= 0.0f);
            brvf = td_max(bnv2_at(k), bnv2min);
            brvf = sqrtf(brvf);
        }
        if (k >= kbl && !ldrag) {
            if (!icrilv && taup[k - 1] > 0.0f) {
                real velco = velco_at(k);
                real usqj = usqj_at(k);
                real temv = 1.0f / velco;
                real tem1 = coefm / dxy * (RHO(kp1) + RHO(k)) * brvf * velco
                            * 0.5f;
                real hd = sqrtf(taup[k - 1] / tem1);
                real fro = brvf * hd * temv;
                real tem2 = sqrtf(usqj);
                real tem = 1.0f + tem2 * fro;
                real rim = usqj * (1.0f - fro) / (tem * tem);
                if (rim <= ric) {
                    if ((oa <= 0.0f) || (kp1 >= kpblmin)) {
                        real temc = 2.0f + 1.0f / tem2;
                        hd = velco * (2.0f * sqrtf(temc) - temc) / brvf;
                        taup[kp1 - 1] = tem1 * hd * hd;
                    }
                } else {
                    taup[kp1 - 1] = taup[k - 1];
                }
            }
        }
    }
    // Flow blocking, summed into the stress.
    if (!ldrag) {
        int kblk = 0;
        real fbdpe = 0.0f, fbdke = 0.0f, zblk = 0.0f;
        for (int k = kte; k >= kpblmin; --k) {
            if (kblk == 0 && k <= kbl) {
                fbdpe = fbdpe + bnv2_at(k) * (ZL(kbl) - ZL(k)) * DEL(k) / g_
                        / RHO(k);
                fbdke = 0.5f * (gfk_pow(U1(k), 2.0f) + gfk_pow(V1(k), 2.0f));
                if (fbdpe >= fbdke) {
                    kblk = k;
                    kblk = min(kblk, kbl);
                    zblk = ZL(kblk) - ZL(1);
                }
            }
        }
        if (kblk != 0) {
            real fbdcd = td_max(2.0f - 1.0f / od, 0.0f);
            real taufb = 0.5f * rhobar * coefm / (dxmeter * dxmeter) * fbdcd
                         * dxyp * olp * zblk * (ulow * ulow);
            real tautem = taufb / (real)(kblk - 1);
            taup[0] = taup[0] + taufb;
            for (int k = 2; k <= kblk; ++k) {
                taufb = taufb - tautem;
                taup[k - 1] = taup[k - 1] + taufb;
            }
            // taufb is 0 above kblk: taup there gains + 0.
            for (int k = kblk + 1; k <= kte + 1; ++k)
                taup[k - 1] = taup[k - 1] + 0.0f;
        }
    }
    // -(g) d(tau)/dp and the critical-line limiter.
    auto taud_at = [&](int k) -> real {
        return 1.0f * (taup[k] - taup[k - 1]) * g_ / DEL(k);
    };
    real dtfac = 1.0f;
    for (int k = 1; k <= kpblmax - 1; ++k) {
        if (k <= kbl) {
            real td = taud_at(k);
            if (td != 0.0f)
                dtfac = td_min(dtfac, fabsf(velco_at(k) / (deltim * td)));
        }
    }
    real dusfc = 0.0f, dvsfc = 0.0f;
    for (int k = 1; k <= kte; ++k) {
        real td = taud_at(k) * dtfac;
        real dtaux = td * xn;
        real dtauy = td * yn;
        dusfc = dusfc + dtaux * DEL(k);
        dvsfc = dvsfc + dtauy * DEL(k);
        size_t q = Q(k);
        // module_bl_gwdo.F hands bl_gwdo_run rublten and copies it back;
        // bl_gwdo.F90 rotates the drag to the model grid and adds it.
        du[q] = du[q] + dtaux * ca + dtauy * sa;
        dv[q] = dv[q] - dtaux * sa + dtauy * ca;
        if (dtaux3d != nullptr) {
            dtaux3d[q] = dtaux * ca + dtauy * sa;
            dtauy3d[q] = -dtaux * sa + dtauy * ca;
        }
    }
    dusfc = (-1.0f / g_) * dusfc;
    dvsfc = (-1.0f / g_) * dvsfc;
    if (dusfcg != nullptr) {
        dusfcg[col] = dusfc * ca + dvsfc * sa;
        dvsfcg[col] = -dusfc * sa + dvsfc * ca;
    }
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RHO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
    #undef TI
    #undef RDZ
}

// --------------------------------------------------------------------------
// gwd_opt = 3: module_bl_gwdo_gsl.F (gwdo_gsl -> gwdo2d), the GSL suite.
// --------------------------------------------------------------------------
// gsl_gwd_ls = gsl_gwd_bl = gsl_gwd_ss = gsl_gwd_fd = 1 and
// gsl_diss_ht_opt = 0 are compile-time PARAMETERs in WRF; with the
// dissipation heating off the scheme never writes RTHBLTEN.  rcl = 1 (the
// wrapper passes it), so rcs = cs = 1 and csg = g.  rstoch is the SPP
// pattern's lowest level, pattern_spp_pbl(i,kts,j), when spp_pbl = 1 and a
// null pointer (rstoch = 0, every _stoch quantity its plain value) when
// spp_pbl = 0, as gwdo_gsl forms rstoch1D (module_bl_gwdo_gsl.F, the "Add
// SPP" block); the _stoch arithmetic stays spelled out as WRF evaluates it.
// lcap = kte, so the lcap < kte stress extension never runs.
//
// dtau (when non-null) is (8, nz, ny, nx): dtaux3d_ls, dtauy3d_ls,
// dtaux3d_bl, dtauy3d_bl, dtaux3d_ss, dtauy3d_ss, dtaux3d_fd, dtauy3d_fd;
// dsfc (when non-null) is (8, ny, nx) in the same order (dusfcg_ls, ...).
extern "C" __global__
void gwdo_gsl_column(const real *u, const real *v, const real *t,
                     const real *qv, const real *p, const real *p_interface,
                     const real *pi, const real *z, const real *dz,
                     const real *var, const real *oc1, const real *oa4,
                     const real *ol4, const real *varss, const real *oc1ss,
                     const real *oa4ss, const real *ol4ss,
                     const real *sina, const real *cosa,
                     const real *xland, const real *br, const real *hpbl,
                     const int *kpbl, const real *rstoch_pattern,
                     real *du, real *dv, real *dtau, real *dsfc,
                     real dxmeter, real deltim, real g, real cp, real rd,
                     real fv, real pi_c, int kpblmax, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const real dxmin_ss = 1000.0f, dxmax_ss = 12000.0f;
    const real dxmin_ls = 3000.0f, dxmax_ls = 13000.0f;
    const real ric = 0.25f, dw2min = 1.0f, rimin = -100.0f, bnv2min = 1.0e-5f;
    const real efmin = 0.0f, efmax = 10.0f, xl = 4.0e4f, gmax = 1.0f;
    const real veleps = 1.0f, factop = 0.5f, frc = 1.0f, ce = 0.8f, cg = 0.5f;
    const int kpblmin = 2;
    const real varmax_ss = 35.0f, varmax_fd = 160.0f, beta_ss = 0.1f;
    const real beta_fd = 0.2f;
    const real frmax = 10.0f, olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const real clf_coeff = 3.4E+07f, clf_coeff_ss = 0.1f;
    const real a1_coeff = 0.00026615161f, a2_coeff = 0.005363f;
    const real TOFD_coeff = 0.0759f, Hefold_nom = 1500.0f;
    const int kte = nz, kts = 1;
    const int mdir = 8;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const real rcl = 1.0f;
    const real rcs = sqrtf(rcl);
    const real cs = 1.0f / sqrtf(rcl);
    const real csg = cs * g;
    const int lcap = kte;
    const real fdir = (real)mdir / (2.0f * pi_c);
    real ca = cosa[col], sa = sina[col];

    #define Q(k) ((size_t)((k) - 1) * st + col)
    // gwdo_gsl rotates the winds to zonal/meridional before gwdo2d.
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RO(k) (1.0f / rd * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(size_t)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (z[Q(k)])

    // Scale-aware tapers.
    real ls_taper, ss_taper;
    if (dxmeter >= dxmax_ls) {
        ls_taper = 1.0f;
    } else if (dxmeter <= dxmin_ls) {
        ls_taper = 0.0f;
    } else {
        ls_taper = 0.5f * (glibc_sinf(pi_c * (dxmeter - 0.5f
                                               * (dxmax_ls + dxmin_ls))
                                      / (dxmax_ls - dxmin_ls)) + 1.0f);
    }
    if (dxmeter >= dxmax_ss) {
        ss_taper = 1.0f;
    } else if (dxmeter <= dxmin_ss) {
        ss_taper = 0.0f;
    } else {
        ss_taper = dxmax_ss * (1.0f - dxmin_ss / dxmeter)
                   / (dxmax_ss - dxmin_ss);
    }
    real delx = dxmeter, dely = dxmeter;
    real dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrtf(delx * delx + dely * dely);
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];

    const real rstoch = rstoch_pattern != nullptr ? rstoch_pattern[col]
                                                  : 0.0f;
    real varc = var[col], varssc = varss[col];
    real var_stoch = varc + varc * 0.666f * rstoch;
    real varss_stoch = varssc + varssc * 0.666f * rstoch;
    real varmax_ss_stoch = varmax_ss + varmax_ss * 0.5f * rstoch;
    real varmax_fd_stoch = varmax_fd + varmax_fd * 0.5f * rstoch;

    // Reference level: the higher of 2*var and the PBL top.
    real zlowtop = 2.0f * var_stoch;
    int klowtop = 0;
    for (int k = kts + 1; k <= kte; ++k) {
        if (ZL(k) - ZL(1) >= zlowtop) { klowtop = k + 1; break; }
    }
    int kpblc = kpbl[col];
    int kbl = max(kpblc, klowtop);
    kbl = max(min(kbl, kpblmax), kpblmin);
    int komax = klowtop - 1;
    real delks = 1.0f / (PRSI(1) - PRSI(kbl));
    real delks1 = 1.0f / (PRSL(1) - PRSL(kbl));
    real ubar = 0.0f, vbar = 0.0f, roll = 0.0f;
    for (int k = kts; k <= kpblmax; ++k) {
        if (k < kbl) {
            real rcsks = rcs * DEL(k) * delks;
            real rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            roll = roll + rdelks * RO(k);
        }
    }
    real oa, ol, oass, olss, olp, od, dxy, dxyp;
    {
        real oa4c[4], ol4c[4], oa4ssc[4], ol4ssc[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(size_t)m * st + col];
            ol4c[m] = ol4[(size_t)m * st + col];
            oa4ssc[m] = oa4ss[(size_t)m * st + col];
            ol4ssc[m] = ol4ss[(size_t)m * st + col];
        }
        real wdir = td_atan2f(ubar, vbar) + pi_c;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        real sgn = (real)(1 - 2 * ((nwd - 1) / 4));
        oa = sgn * oa4c[m];
        ol = ol4c[m];
        oass = sgn * oa4ssc[m];
        olss = ol4ssc[m];
        real ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = ol4p[m];
        od = olp / td_max(ol, olmin);
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

    auto bnv2_at = [&](int k) -> real {        // bnv2(k) as WRF stores it
        if (k >= kte) return 0.0f;
        real rdz = 1.0f / (ZL(k + 1) - ZL(k));
        real b = 2.0f * g * rdz * (VTK(k + 1) - VTK(k))
                 / (VTK(k + 1) + VTK(k));
        return td_max(b, bnv2min);
    };
    auto usqj_at = [&](int k) -> real {
        real ti = 2.0f / (t[Q(k)] + t[Q(k + 1)]);
        real rdz = 1.0f / (ZL(k + 1) - ZL(k));
        real tem1 = U1(k) - U1(k + 1);
        real tem2 = V1(k) - V1(k + 1);
        real dw2 = rcl * (tem1 * tem1 + tem2 * tem2);
        real shr2 = td_max(dw2, dw2min) * rdz * rdz;
        real bvf2 = g * (g / cp + rdz * (VTJ(k + 1) - VTJ(k))) * ti;
        return td_max(bvf2 / shr2, rimin);
    };

    const bool ls_on = ls_taper > 1.0e-02f;
    const bool ss_on = ss_taper > 1.0e-02f;
    bool ldrag = false;
    real ulow = 0.0f, rulow = 0.0f, xn = 0.0f, yn = 0.0f;
    real taub = 0.0f, coefm = 0.0f, bnv = 0.0f, fr = 0.0f;
    auto velco_at = [&](int k) -> real {
        real vc = (0.5f * rcs) * ((U1(k) + U1(k + 1)) * ubar
                                  + (V1(k) + V1(k + 1)) * vbar);
        vc = vc * rulow;
        if (vc < veleps && vc > 0.0f) vc = veleps;
        return vc;
    };
    if (ls_on) {
        ulow = td_max(sqrtf(ubar * ubar + vbar * vbar), 1.0f);
        rulow = 1.0f / ulow;
        ldrag = velco_at(1) <= 0.0f;
        for (int k = kpblmin; k <= kpblmax; ++k)
            if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;
        for (int k = kts; k <= kpblmax; ++k)
            if (k < kbl) ldrag = ldrag || bnv2_at(k) < 0.0f;
        real wtkbj = (PRSL(1) - PRSL(2)) * delks1;
        real bnv2_1 = wtkbj * bnv2_at(1);
        for (int k = kpblmin; k <= kpblmax; ++k) {
            if (k < kbl) {
                real rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
                bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
            }
        }
        ldrag = ldrag || bnv2_1 <= 0.0f;
        ldrag = ldrag || ulow == 1.0f;
        ldrag = ldrag || var_stoch <= 0.0f;
        if (!ldrag) {
            bnv = sqrtf(bnv2_1);
            fr = bnv * rulow * 2.0f * var_stoch * od;
            fr = td_min(fr, frmax);
            xn = ubar * rulow;
            yn = vbar * rulow;
        }
        if (!ldrag) {
            real efact = gfk_pow(oa + 2.0f, ce * fr / frc);
            efact = td_min(td_max(efact, efmin), efmax);
            real cleff = clf_coeff / sqrtf(dxmeter);
            coefm = gfk_pow(1.0f + ol, oa + 1.0f);
            real xlinv = coefm / cleff;
            real tem = fr * fr * oc1[col];
            real gfobnv = gmax * tem / ((tem + cg) * bnv);
            taub = xlinv * roll * ulow * ulow * ulow * gfobnv * efact;
        } else {
            taub = 0.0f;
            xn = 0.0f;
            yn = 0.0f;
        }
    }

    // Small-scale gravity-wave drag (stable boundary layer).
    real tauwavex0 = 0.0f, tauwavey0 = 0.0f, hpbl2 = 0.0f;
    bool wave = false;
    int kpblw = kpblc;
    real xlandc = xland[col];
    if (ss_on) {
        real hpblc = hpbl[col];
        hpbl2 = hpblc + 10.0f;
        int kpbl2 = kpblc;
        const int kvar = 1;
        // za(k) from the cumulative layer depths, in WRF's order.
        real zq = 0.0f;
        real za_k = 0.0f;
        {
            real zqk = 0.0f;
            int kend = max(kpblc, kts + 1);
            for (int k = kts; k <= kend; ++k) {
                real zqn = dz[Q(k)] + zqk;
                za_k = 0.5f * (zqk + zqn);
                if (k >= kts + 1 && za_k > 300.0f) {
                    kpbl2 = k;
                    if (k == kpblc) hpbl2 = hpblc + 10.0f;
                    else hpbl2 = za_k + 10.0f;
                    break;
                }
                zqk = zqn;
            }
        }
        (void)zq;
        if ((xlandc - 1.5f) <= 0.0f && 2.0f * varss_stoch <= hpblc) {
            real thv1 = t[Q(1)] / pi[Q(1)] * (1.0f + fv * qv[Q(1)]);
            real thvk = t[Q(kpbl2)] / pi[Q(kpbl2)] * (1.0f + fv * qv[Q(kpbl2)]);
            if (br[col] > 0.0f && thvk - thv1 > 0.0f) {
                real cleff_ss = sqrtf(dxy * dxy + dxyp * dxyp);
                cleff_ss = clf_coeff_ss * td_max(dxmax_ss, cleff_ss);
                real coefm_ss = gfk_pow(1.0f + olss, oass + 1.0f);
                real xlinv = coefm_ss / cleff_ss;
                real govrth = g / (0.5f * (thvk + thv1));
                real xnbv = sqrtf(govrth * (thvk - thv1) / hpbl2);
                real var_temp = td_min(varss_stoch, varmax_ss_stoch)
                    + td_max(0.0f, beta_ss * (varss_stoch - varmax_ss_stoch));
                real tv2 = 2.0f * var_temp;
                if (fabsf(xnbv / U1(kpbl2)) > xlinv) {
                    tauwavex0 = 0.5f * xnbv * xlinv * (tv2 * tv2) * RO(kvar)
                                * U1(kvar);
                    tauwavex0 = tauwavex0 * ss_taper;
                } else {
                    tauwavex0 = 0.0f;
                }
                if (fabsf(xnbv / V1(kpbl2)) > xlinv) {
                    tauwavey0 = 0.5f * xnbv * xlinv * (tv2 * tv2) * RO(kvar)
                                * V1(kvar);
                    tauwavey0 = tauwavey0 * ss_taper;
                } else {
                    tauwavey0 = 0.0f;
                }
                wave = true;
            }
        }
    }
    // Form drag constants.
    real a2 = 0.0f, H_efold = 0.0f;
    bool form = ss_on && (xlandc - 1.5f) <= 0.0f;
    if (form) {
        real var_temp = td_min(varss_stoch, varmax_fd_stoch)
            + td_max(0.0f, beta_fd * (varss_stoch - varmax_fd_stoch));
        real a1 = a1_coeff * (var_temp * var_temp);
        a2 = a1 * a2_coeff;
        H_efold = td_max(2.0f * varss_stoch, hpbl[col]);
        H_efold = td_min(H_efold, Hefold_nom);
    }

    // Large-scale stress profile.
    real taup[TD_KMAX + 1], taufb[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) { taup[k - 1] = 0.0f; taufb[k - 1] = 0.0f; }
    if (ls_on) {
        for (int k = kts; k <= kpblmax; ++k)
            if (k <= kbl) taup[k - 1] = taub;
        bool icrilv = false;
        real brvf = 0.0f;
        for (int k = kpblmin; k <= kte - 1; ++k) {
            int kp1 = k + 1;
            if (k >= kbl) {
                icrilv = icrilv || (usqj_at(k) < ric) || (velco_at(k) <= 0.0f);
                brvf = td_max(bnv2_at(k), bnv2min);
                brvf = sqrtf(brvf);
            }
            if (k >= kbl && !ldrag) {
                if (!icrilv && taup[k - 1] > 0.0f) {
                    real velco = velco_at(k);
                    real usqj = usqj_at(k);
                    real temv = 1.0f / velco;
                    real tem1 = coefm / dxy * (RO(kp1) + RO(k)) * brvf * velco
                                * 0.5f;
                    real hd = sqrtf(taup[k - 1] / tem1);
                    real fro = brvf * hd * temv;
                    real tem2 = sqrtf(usqj);
                    real tem = 1.0f + tem2 * fro;
                    real rim = usqj * (1.0f - fro) / (tem * tem);
                    if (rim <= ric) {
                        if ((oa <= 0.0f) || (kp1 >= kpblmin)) {
                            real temc = 2.0f + 1.0f / tem2;
                            hd = velco * (2.0f * sqrtf(temc) - temc) / brvf;
                            taup[kp1 - 1] = tem1 * hd * hd;
                        }
                    } else {
                        taup[kp1 - 1] = taup[k - 1];
                    }
                }
            }
        }
        // Flow blocking (kept apart from taup, as WRF keeps it).
        if (!ldrag) {
            int kblk = 0;
            real pe = 0.0f, zblk = 0.0f;
            for (int k = kte; k >= kpblmin; --k) {
                if (kblk == 0 && k <= komax) {
                    pe = pe + bnv2_at(k) * (ZL(komax) - ZL(k)) * DEL(k) / g
                         / RO(k);
                    real ke = 0.5f * (gfk_pow(rcs * U1(k), 2.0f)
                                      + gfk_pow(rcs * V1(k), 2.0f));
                    if (pe >= ke) {
                        kblk = k;
                        kblk = min(kblk, kbl);
                        zblk = ZL(kblk) - ZL(kts);
                    }
                }
            }
            if (kblk != 0) {
                real cd = td_max(2.0f - 1.0f / od, 0.0f);
                real mx = td_max(dxmax_ls, dxy);
                taufb[0] = 0.5f * roll * coefm / (mx * mx) * cd * dxyp * olp
                           * zblk * (ulow * ulow);
                real tautem = taufb[0] / (real)(kblk - kts);
                for (int k = kts + 1; k <= kblk; ++k)
                    taufb[k - 1] = taufb[k - 2] - tautem;
            }
        }
    }
    // Deceleration terms and the critical-line limiter (ls_on only).
    auto taud_ls_at = [&](int k) -> real {
        real td = 1.0f * (taup[k] - taup[k - 1]) * csg / DEL(k);
        if (k >= lcap) td = td * factop;
        return td;
    };
    auto taud_bl_at = [&](int k) -> real {
        real td = 1.0f * (taufb[k] - taufb[k - 1]) * csg / DEL(k);
        if (k >= lcap) td = td * factop;
        return td;
    };
    real dtfac = 1.0f;
    if (ls_on) {
        for (int k = kts; k <= kpblmax - 1; ++k) {
            if (k <= kbl) {
                real s = taud_ls_at(k) + taud_bl_at(k);
                if (s != 0.0f)
                    dtfac = td_min(dtfac, fabsf(velco_at(k)
                                                / (deltim * rcs * s)));
            }
        }
    }

    // Per level: the four components in WRF's order of accumulation into
    // dudt (ss, then fd, then ls + bl), rotated and added to the PBL
    // tendencies as gwdo_gsl does.
    real dus[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dvs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    real zqk = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        size_t q = Q(k);
        real del = DEL(k);
        real zqn = dz[q] + zqk;
        real za = 0.5f * (zqk + zqn);
        zqk = zqn;
        real dudt = 0.0f, dvdt = 0.0f;
        real ut_ss = 0.0f, vt_ss = 0.0f, ut_fd = 0.0f, vt_fd = 0.0f;
        real ut_ls = 0.0f, vt_ls = 0.0f, ut_bl = 0.0f, vt_bl = 0.0f;
        if (ss_on) {
            if (wave && k <= kpblw) {
                real shape = td_max((1.0f - za / hpbl2), 0.0f);
                ut_ss = -1.0f * tauwavex0 * 2.0f * shape / hpbl2;
                vt_ss = -1.0f * tauwavey0 * 2.0f * shape / hpbl2;
            }
            dudt = dudt + ut_ss;
            dvdt = dvdt + vt_ss;
            dus[2] = dus[2] + ut_ss * del;
            dvs[2] = dvs[2] + vt_ss * del;
            if (form) {
                real u1 = U1(k), v1 = V1(k);
                real wsp = sqrtf(u1 * u1 + v1 * v1);
                real ef = gfk_exp(-gfk_pow(za / H_efold, 1.5f));
                real zp = gfk_pow(za, -1.2f);
                ut_fd = -(TOFD_coeff * wsp * u1 * ef * a2 * zp * ss_taper);
                vt_fd = -(TOFD_coeff * wsp * v1 * ef * a2 * zp * ss_taper);
            }
            dudt = dudt + ut_fd;
            dvdt = dvdt + vt_fd;
            dus[3] = dus[3] + ut_fd * del;
            dvs[3] = dvs[3] + vt_fd * del;
        }
        if (ls_on) {
            real tls = taud_ls_at(k) * dtfac * ls_taper;
            real tbl = taud_bl_at(k) * dtfac * ls_taper;
            ut_ls = tls * xn;
            vt_ls = tls * yn;
            ut_bl = tbl * xn;
            vt_bl = tbl * yn;
            dudt = ut_ls + ut_bl + dudt;
            dvdt = vt_ls + vt_bl + dvdt;
            dus[0] = dus[0] + ut_ls * del;
            dvs[0] = dvs[0] + vt_ls * del;
            dus[1] = dus[1] + ut_bl * del;
            dvs[1] = dvs[1] + vt_bl * del;
        }
        du[q] = du[q] + dudt * ca + dvdt * sa;
        dv[q] = dv[q] - dudt * sa + dvdt * ca;
        if (dtau != nullptr) {
            size_t n3 = (size_t)nz * st;
            real cx[4] = {ut_ls, ut_bl, ut_ss, ut_fd};
            real cy[4] = {vt_ls, vt_bl, vt_ss, vt_fd};
            for (int m = 0; m < 4; ++m) {
                dtau[(size_t)(2 * m) * n3 + q] = cx[m] * ca + cy[m] * sa;
                dtau[(size_t)(2 * m + 1) * n3 + q] = -cx[m] * sa + cy[m] * ca;
            }
        }
    }
    if (dsfc != nullptr) {
        for (int m = 0; m < 4; ++m) {
            real us = (-1.0f / g * rcs) * dus[m];
            real vs = (-1.0f / g * rcs) * dvs[m];
            dsfc[(size_t)(2 * m) * st + col] = us * ca + vs * sa;
            dsfc[(size_t)(2 * m + 1) * st + col] = -us * sa + vs * ca;
        }
    }
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
}
