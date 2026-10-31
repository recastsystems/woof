// Fused large-step pressure-gradient, buoyancy, and geopotential RHS terms.
//
// These kernels replace eager CuPy expression trees.  Each CuPy binary
// operator formerly wrote an FP32 temporary, so a compiler-contracted FMA
// would change the rounding tree.  The rn_* helpers spell out the same
// round-to-nearest FP32 boundary while retaining the repository-wide NVRTC
// flags (default contraction policy, no fast math).

static __device__ __forceinline__ real rn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#else
// Face coordinates are in [0, extent]. Avoid integer division in the
// periodic interior while preserving the duplicated face at extent.
static __device__ __forceinline__
int periodic_face(int face, int extent)
{
    return face == extent ? 0 : face;
}

static __device__ __forceinline__
int periodic_previous(int face, int extent)
{
    return face == 0 ? extent - 1 : face - 1;
}

// Centered stencils normally fit in the extent. Keep the original remainder
// semantics for smaller direct-kernel callers as well.
static __device__ __forceinline__
int periodic_offset(int cell, int offset, int extent)
{
    if (extent < (offset < 0 ? -offset : offset))
        return (cell + offset + extent) % extent;
    int shifted = cell + offset;
    return shifted < 0 ? shifted + extent
         : shifted >= extent ? shifted - extent : shifted;
}
#endif

static __device__ __forceinline__
real base_value(const real* __restrict__ field, int k, size_t c,
                size_t st, int base3d)
{
    return field[base3d ? (size_t)k * st + c : (size_t)k];
}

static __device__ __forceinline__
real mut_value(const real* __restrict__ mub2d,
               const real* __restrict__ mup, size_t c)
{
    return rn_add(mub2d[c], mup[c]);
}

#if GPUWM_WRF_EXACT
// WRF calc_mu_uv's face mass, the grid%muu/muv that rhs_ph and
// horizontal_pressure_gradient read: 0.5*(MU(i)+MU(i-1)+MUB(i)+MUB(i-1))
// left to right, ca the face's own column and cb the one before.  Summing
// (MUB+MU) per column first, as the default does, agreed with it on the
// time-t state of the round-3 pair and parted from the second RK stage
// (2 ULP in ~3200 u and v pressure-gradient words, 1 ULP in 1862 rhs_ph
// words).
static __device__ __forceinline__
real wrf_face_mu(const real* __restrict__ mup,
                 const real* __restrict__ mub2d, size_t ca, size_t cb)
{
    return rn_mul(0.5f, rn_add(rn_add(rn_add(mup[ca], mup[cb]), mub2d[ca]),
                               mub2d[cb]));
}

#endif
static __device__ __forceinline__
real pp_value(const real* __restrict__ p, const real* __restrict__ pb,
              int k, size_t c, size_t st, int base3d)
{
#if GPUWM_WRF_EXACT_D_DIAGNOSTICS
    return p[(size_t)k * st + c];
#else
    return rn_sub(p[(size_t)k * st + c],
                  base_value(pb, k, c, st, base3d));
#endif
}

static __device__ __forceinline__
real dpn_value(const real* __restrict__ p, const real* __restrict__ pb,
               int kf, size_t c, size_t st, int nz, int base3d,
               const real* __restrict__ fnm,
               const real* __restrict__ fnp,
               real cf1, real cf2, real cf3,
               int top_lid, real cfn, real cfn1)
{
    if (kf == nz) {
        if (!top_lid) return 0.0f;
        real top = rn_mul(cfn, pp_value(p, pb, nz - 1, c, st, base3d));
        real below = rn_mul(cfn1, pp_value(p, pb, nz - 2, c, st, base3d));
        return rn_add(top, below);
    }
    if (kf == 0) {
        real t0 = rn_mul(cf1, pp_value(p, pb, 0, c, st, base3d));
        real t1 = rn_mul(cf2, pp_value(p, pb, 1, c, st, base3d));
        real t2 = rn_mul(cf3, pp_value(p, pb, 2, c, st, base3d));
        return rn_add(rn_add(t0, t1), t2);
    }
    real hi = rn_mul(fnm[kf], pp_value(p, pb, kf, c, st, base3d));
    real lo = rn_mul(fnp[kf], pp_value(p, pb, kf - 1, c, st, base3d));
    return rn_add(hi, lo);
}

#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#else
// Reuse each pressure perturbation at both adjacent full faces. The
// interpolation retains the same rounded products and left-to-right sum.
static __device__ __forceinline__
void dpn_pair_value(const real* __restrict__ p, const real* __restrict__ pb,
                    int k, size_t c, size_t st, int nz, int base3d,
                    const real* __restrict__ fnm,
                    const real* __restrict__ fnp,
                    real cf1, real cf2, real cf3,
                    int top_lid, real cfn, real cfn1, real pp_k,
                    real& hi, real& lo)
{
    if (nz < 3) {
        hi = dpn_value(p, pb, k + 1, c, st, nz, base3d,
                       fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
        lo = dpn_value(p, pb, k, c, st, nz, base3d,
                       fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
        return;
    }
    real pp_below = k > 0 ? pp_value(p, pb, k - 1, c, st, base3d) : 0.0f;
    real pp_above = k + 1 < nz ? pp_value(p, pb, k + 1, c, st, base3d) : 0.0f;
    if (k + 1 == nz) {
        if (top_lid) {
            real top = rn_mul(cfn, pp_k);
            real below = rn_mul(cfn1, pp_below);
            hi = rn_add(top, below);
        } else {
            hi = 0.0f;
        }
    } else {
        real upper = rn_mul(fnm[k + 1], pp_above);
        real lower = rn_mul(fnp[k + 1], pp_k);
        hi = rn_add(upper, lower);
    }
    if (k == 0) {
        real t0 = rn_mul(cf1, pp_k);
        real t1 = rn_mul(cf2, pp_above);
        real t2 = rn_mul(cf3, pp_value(p, pb, 2, c, st, base3d));
        lo = rn_add(rn_add(t0, t1), t2);
    } else {
        real upper = rn_mul(fnm[k], pp_k);
        real lower = rn_mul(fnp[k], pp_below);
        lo = rn_add(upper, lower);
    }
}
#endif

static __device__ __forceinline__
real php_half_value(const real* __restrict__ php,
                    const real* __restrict__ phb,
                    int k, size_t c, size_t st, int base3d)
{
    real perturbation = rn_add(php[(size_t)k * st + c],
                               php[(size_t)(k + 1) * st + c]);
    if (!base3d)
        return rn_mul(0.5f, perturbation);
    real base = rn_add(phb[(size_t)k * st + c],
                       phb[(size_t)(k + 1) * st + c]);
#if GPUWM_WRF_EXACT
    // WRF calc_php (module_big_step_utilities_em.F:1261) sums left to right:
    // 0.5*(phb(k)+phb(k+1)+ph(k)+ph(k+1)).  The grouped form below differs
    // in ~24 % of words (combo-sweep round2 LOCALIZE.md, cause 1).
    return rn_mul(0.5f, rn_add(rn_add(base, php[(size_t)k * st + c]),
                               php[(size_t)(k + 1) * st + c]));
#else
    return rn_mul(0.5f, rn_add(base, perturbation));
#endif
}


#if GPUWM_WRF_EXACT_C_BIGSTEP
static __device__ __forceinline__
real wrf_dpn_face(const real* __restrict__ p, const real* __restrict__ pb,
                  int kf, size_t ca, size_t cb, size_t st, int nz, int base3d,
                  const real* __restrict__ fnm, const real* __restrict__ fnp,
                  real cf1, real cf2, real cf3, int top_lid, real cfn, real cfn1)
{
    if (kf == nz && !top_lid) return 0.0f;
    int k0 = kf == nz ? nz - 1 : kf;
    real pair0 = rn_add(pp_value(p, pb, k0, cb, st, base3d),
                        pp_value(p, pb, k0, ca, st, base3d));
    if (kf == 0) {
        real pair1 = rn_add(pp_value(p, pb, 1, cb, st, base3d), pp_value(p, pb, 1, ca, st, base3d));
        real pair2 = rn_add(pp_value(p, pb, 2, cb, st, base3d), pp_value(p, pb, 2, ca, st, base3d));
        return rn_mul(0.5f, rn_add(rn_add(rn_mul(cf1, pair0), rn_mul(cf2, pair1)), rn_mul(cf3, pair2)));
    }
    int k1 = kf == nz ? nz - 2 : kf - 1;
    real pair1 = rn_add(pp_value(p, pb, k1, cb, st, base3d),
                        pp_value(p, pb, k1, ca, st, base3d));
    real high = rn_mul(kf == nz ? cfn : fnm[kf], pair0);
    real low = rn_mul(kf == nz ? cfn1 : fnp[kf], pair1);
    return rn_mul(0.5f, rn_add(high, low));
}
#endif

static __device__
real pgf_face(size_t c_a, size_t c_b, int k, size_t st, int nz,
              real rd, real half_rd,
              const real* __restrict__ p, const real* __restrict__ pb,
              const real* __restrict__ al, const real* __restrict__ alt,
              const real* __restrict__ php, const real* __restrict__ phb,
              const real* __restrict__ mup, const real* __restrict__ mub2d,
              const real* __restrict__ c1h, const real* __restrict__ c2h,
              const real* __restrict__ rdnw,
              const real* __restrict__ fnm, const real* __restrict__ fnp,
              real cf1, real cf2, real cf3, int base3d,
              int top_lid, real cfn, real cfn1)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    size_t a = (size_t)k * st + c_a;
    size_t b = (size_t)k * st + c_b;

    // muf/dmu retain the old (sum -> multiply by 0.5) temporary boundary.
#if GPUWM_WRF_EXACT
    real muf = wrf_face_mu(mup, mub2d, c_a, c_b);
#else
    real muf = rn_mul(0.5f, rn_add(mut_value(mub2d, mup, c_a),
                                   mut_value(mub2d, mup, c_b)));
#endif
    real dmu = rn_mul(0.5f, rn_add(mup[c_a], mup[c_b]));
    real layer_mass = rn_add(rn_mul(c1h[k], muf), c2h[k]);

    real dph_hi = rn_sub(php[a + st], php[b + st]);
    real dph_lo = rn_sub(php[a], php[b]);
    #if GPUWM_WRF_EXACT_C_BIGSTEP
    real bracket = rn_sub(rn_add(dph_hi, php[a]), php[b]);
#else
    real bracket = rn_add(dph_hi, dph_lo);
#endif

    real alt_sum = rn_add(alt[a], alt[b]);
    real dpp = rn_sub(pp_value(p, pb, k, c_a, st, base3d),
                      pp_value(p, pb, k, c_b, st, base3d));
    bracket = rn_add(bracket, rn_mul(alt_sum, dpp));

    real al_sum = rn_add(al[a], al[b]);
    real dpb = rn_sub(base_value(pb, k, c_a, st, base3d),
                      base_value(pb, k, c_b, st, base3d));
    bracket = rn_add(bracket, rn_mul(al_sum, dpb));
    real left = rn_mul(rn_mul(half_rd, layer_mass), bracket);

    real dphp = rn_sub(php_half_value(php, phb, k, c_a, st, base3d),
                        php_half_value(php, phb, k, c_b, st, base3d));
#if GPUWM_WRF_EXACT_C_BIGSTEP
    real dpf_hi = wrf_dpn_face(p, pb, k + 1, c_a, c_b, st, nz, base3d,
                               fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
    real dpf_lo = wrf_dpn_face(p, pb, k, c_a, c_b, st, nz, base3d,
                               fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
#else
    real dpf_hi = rn_mul(
        0.5f,
        rn_add(dpn_value(p, pb, k + 1, c_a, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1),
               dpn_value(p, pb, k + 1, c_b, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1)));
    real dpf_lo = rn_mul(
        0.5f,
        rn_add(dpn_value(p, pb, k, c_a, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1),
               dpn_value(p, pb, k, c_b, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1)));
#endif
    real vertical = rn_mul(rdnw[k], rn_sub(dpf_hi, dpf_lo));
    #if GPUWM_WRF_EXACT_C_BIGSTEP
    real perturbation_mass = rn_mul(0.5f, rn_add(rn_mul(c1h[k], mup[c_b]), rn_mul(c1h[k], mup[c_a])));
    vertical = rn_sub(vertical, perturbation_mass);
#else
    vertical = rn_sub(vertical, rn_mul(c1h[k], dmu));
#endif
    real right = rn_mul(rn_mul(rd, dphp), vertical);
    return rn_add(left, right);
#else
    size_t a = (size_t)k * st + c_a;
    size_t b = (size_t)k * st + c_b;

    // muf/dmu retain the old (sum -> multiply by 0.5) temporary boundary.
    real muf = rn_mul(0.5f, rn_add(mut_value(mub2d, mup, c_a),
                                   mut_value(mub2d, mup, c_b)));
    real dmu = rn_mul(0.5f, rn_add(mup[c_a], mup[c_b]));
    real layer_mass = rn_add(rn_mul(c1h[k], muf), c2h[k]);

    real dph_hi = rn_sub(php[a + st], php[b + st]);
    real dph_lo = rn_sub(php[a], php[b]);
    real bracket = rn_add(dph_hi, dph_lo);

    real alt_sum = rn_add(alt[a], alt[b]);
    real pp_a = pp_value(p, pb, k, c_a, st, base3d);
    real pp_b = pp_value(p, pb, k, c_b, st, base3d);
    real dpp = rn_sub(pp_a, pp_b);
    bracket = rn_add(bracket, rn_mul(alt_sum, dpp));

    real al_sum = rn_add(al[a], al[b]);
    real dpb = rn_sub(base_value(pb, k, c_a, st, base3d),
                      base_value(pb, k, c_b, st, base3d));
    bracket = rn_add(bracket, rn_mul(al_sum, dpb));
    real left = rn_mul(rn_mul(half_rd, layer_mass), bracket);

    real dphp = rn_sub(php_half_value(php, phb, k, c_a, st, base3d),
                        php_half_value(php, phb, k, c_b, st, base3d));
    real dp_a_hi, dp_a_lo, dp_b_hi, dp_b_lo;
    dpn_pair_value(p, pb, k, c_a, st, nz, base3d,
                   fnm, fnp, cf1, cf2, cf3,
                   top_lid, cfn, cfn1, pp_a, dp_a_hi, dp_a_lo);
    dpn_pair_value(p, pb, k, c_b, st, nz, base3d,
                   fnm, fnp, cf1, cf2, cf3,
                   top_lid, cfn, cfn1, pp_b, dp_b_hi, dp_b_lo);
    real dpf_hi = rn_mul(0.5f, rn_add(dp_a_hi, dp_b_hi));
    real dpf_lo = rn_mul(0.5f, rn_add(dp_a_lo, dp_b_lo));
    real vertical = rn_mul(rdnw[k], rn_sub(dpf_hi, dpf_lo));
    vertical = rn_sub(vertical, rn_mul(c1h[k], dmu));
    real right = rn_mul(rn_mul(rd, dphp), vertical);
    return rn_add(left, right);
#endif
}

extern "C" __global__
void slow_pgf(real* __restrict__ ru_t, real* __restrict__ rv_t,
              const real* __restrict__ p, const real* __restrict__ pb,
              const real* __restrict__ al, const real* __restrict__ alt,
              const real* __restrict__ php, const real* __restrict__ phb,
              const real* __restrict__ mup, const real* __restrict__ mub2d,
              const real* __restrict__ c1h, const real* __restrict__ c2h,
              const real* __restrict__ rdnw,
              const real* __restrict__ fnm, const real* __restrict__ fnp,
              real cf1, real cf2, real cf3,
              int top_lid, real cfn, real cfn1,
              const real* __restrict__ cqu,
              const real* __restrict__ cqv, int use_cq,
              real rdx, real rdy, real half_rdx, real half_rdy,
              int boundary_x, int boundary_y, int base3d,
              int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int rem = tid - k * nyf * nxf;
    int j = rem / nxf;
    int i = rem - j * nxf;
    size_t st = (size_t)ny * nx;

    if (j < ny && ((!boundary_x && i <= nx)
                   || (boundary_x && i > 0 && i < nx))) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int ia = i % nx, ib = (i - 1 + nx) % nx;
#else
        int ia = periodic_face(i, nx), ib = periodic_previous(i, nx);
#endif
        size_t ca = (size_t)j * nx + ia, cb = (size_t)j * nx + ib;
        real term = pgf_face(ca, cb, k, st, nz, rdx, half_rdx,
                             p, pb, al, alt, php, phb, mup, mub2d,
                             c1h, c2h, rdnw, fnm, fnp,
                             cf1, cf2, cf3, base3d,
                             top_lid, cfn, cfn1);
        size_t out = I3S(k, j, i, ny, nxf);
        if (use_cq) term = rn_mul(cqu[out], term);
        ru_t[out] = rn_sub(ru_t[out], term);
    }
    if (i < nx && ((!boundary_y && j <= ny)
                   || (boundary_y && j > 0 && j < ny))) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int ja = j % ny, jb = (j - 1 + ny) % ny;
#else
        int ja = periodic_face(j, ny), jb = periodic_previous(j, ny);
#endif
        size_t ca = (size_t)ja * nx + i, cb = (size_t)jb * nx + i;
        real term = pgf_face(ca, cb, k, st, nz, rdy, half_rdy,
                             p, pb, al, alt, php, phb, mup, mub2d,
                             c1h, c2h, rdnw, fnm, fnp,
                             cf1, cf2, cf3, base3d,
                             top_lid, cfn, cfn1);
        size_t out = I3S(k, j, i, nyf, nx);
        if (use_cq) term = rn_mul(cqv[out], term);
        rv_t[out] = rn_sub(rv_t[out], term);
    }
}

static __device__ __forceinline__
real q_total(const real* __restrict__ qv, const real* __restrict__ qc,
             const real* __restrict__ qr, const real* __restrict__ qi,
             const real* __restrict__ qs, const real* __restrict__ qg,
             const real* __restrict__ qh,
             size_t ix, int moist_mode)
{
    real q = rn_add(qv[ix], qc[ix]);
    q = rn_add(q, qr[ix]);
    if (moist_mode >= 2) {
        q = rn_add(q, qi[ix]);
        q = rn_add(q, qs[ix]);
        q = rn_add(q, qg[ix]);
    }
    if (moist_mode == 3) q = rn_add(q, qh[ix]);
    return q;
}

#if GPUWM_WRF_EXACT
// WRF calc_cq's cqw = 0.5*qtot with qtot accumulated species by species,
// qtot = qtot + moist(k) + moist(k-1), in the moist array's order.  Summing
// each level's total first and then the pair, the default spelling, moved
// pg_buoy_w in 14471 words of the first RK stage wherever hydrometeors
// exist (combo-sweep round 3, A088).
static __device__ __forceinline__
real wrf_cqw_pair(const real* __restrict__ qv, const real* __restrict__ qc,
                  const real* __restrict__ qr, const real* __restrict__ qi,
                  const real* __restrict__ qs, const real* __restrict__ qg,
                  const real* __restrict__ qh, size_t a, size_t b,
                  int moist_mode)
{
    real q = rn_add(rn_add(0.0f, qv[a]), qv[b]);
    q = rn_add(rn_add(q, qc[a]), qc[b]);
    q = rn_add(rn_add(q, qr[a]), qr[b]);
    if (moist_mode >= 2) {
        q = rn_add(rn_add(q, qi[a]), qi[b]);
        q = rn_add(rn_add(q, qs[a]), qs[b]);
        q = rn_add(rn_add(q, qg[a]), qg[b]);
    }
    if (moist_mode == 3) q = rn_add(rn_add(q, qh[a]), qh[b]);
    return rn_mul(0.5f, q);
}

// WRF pg_buoy_w (module_big_step_utilities_em.F:2480, :2490) scales the
// bracket by (1./msfty(i,j))*g, evaluated left to right, where the default
// path takes (g*bracket)/msft: two roundings of the same quantity
// (combo-sweep round2 LOCALIZE.md, cause 2).  Without map factors WRF's
// msfty is 1 and (1./1.)*g is g exactly.
static __device__ __forceinline__
real buoyancy_scale(const real* __restrict__ msft, size_t c, int has_msf)
{
    if (!has_msf) return G;
    return rn_mul(rn_div(1.0f, msft[c]), G);
}
#endif

extern "C" __global__
void slow_buoyancy(real* __restrict__ rw_t,
                   const real* __restrict__ p,
                   const real* __restrict__ pb,
                   const real* __restrict__ mup,
                   const real* __restrict__ mub2d,
                   const real* __restrict__ qv,
                   const real* __restrict__ qc,
                   const real* __restrict__ qr,
                   const real* __restrict__ qi,
                   const real* __restrict__ qs,
                   const real* __restrict__ qg,
                   const real* __restrict__ qh,
                   const real* __restrict__ rdn,
                   const real* __restrict__ rdnw,
                   const real* __restrict__ c1f,
                   const real* __restrict__ c2f,
                   const real* __restrict__ msft,
                   int moist_mode, int has_msf, int base3d,
                   int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    size_t im = ix - st;
    if (k == nz) {
        real cqw = 0.0f;
        if (moist_mode) {
            size_t top = (size_t)(nz - 1) * st + c;
            size_t below = top - st;
#if GPUWM_WRF_EXACT
            cqw = wrf_cqw_pair(qv, qc, qr, qi, qs, qg, qh, top, below,
                               moist_mode);
#else
            cqw = rn_mul(0.5f,
                         rn_add(q_total(qv, qc, qr, qi, qs, qg, qh,
                                        top, moist_mode),
                                q_total(qv, qc, qr, qi, qs, qg, qh,
                                        below, moist_mode)));
#endif
        }
        real cq1 = rn_div(1.0f, rn_add(1.0f, cqw));
        real pp_top = pp_value(p, pb, nz - 1, c, st, base3d);
        real pressure = rn_mul(
            rn_mul(rn_mul(cq1, 2.0f), rdnw[nz - 1]),
            rn_sub(0.0f, pp_top));
        real term = rn_sub(pressure, rn_mul(c1f[nz], mup[c]));
        if (moist_mode) {
            real loading = rn_mul(cqw, cq1);
            real base_mass = rn_add(rn_mul(c1f[nz], mub2d[c]), c2f[nz]);
            term = rn_sub(term, rn_mul(loading, base_mass));
        }
#if GPUWM_WRF_EXACT
        real buoy = rn_mul(buoyancy_scale(msft, c, has_msf), term);
#else
        real buoy = rn_mul(G, term);
        if (has_msf) buoy = rn_div(buoy, msft[c]);
#endif
        rw_t[ix] = rn_add(rw_t[ix], buoy);
        return;
    }
    real dp = rn_sub(pp_value(p, pb, k, c, st, base3d),
                     pp_value(p, pb, k - 1, c, st, base3d));
    real term;
    if (moist_mode) {
#if GPUWM_WRF_EXACT
        real cqw = wrf_cqw_pair(qv, qc, qr, qi, qs, qg, qh, ix, im,
                                moist_mode);
#else
        real cqw = rn_mul(0.5f,
                          rn_add(q_total(qv, qc, qr, qi, qs, qg, qh,
                                         ix, moist_mode),
                                 q_total(qv, qc, qr, qi, qs, qg, qh,
                                         im, moist_mode)));
#endif
        real cq1 = rn_div(1.0f, rn_add(1.0f, cqw));
        term = rn_mul(rn_mul(cq1, rdn[k]), dp);
        term = rn_sub(term, rn_mul(c1f[k], mup[c]));
        real loading = rn_mul(cqw, cq1);
        real base_mass = rn_add(rn_mul(c1f[k], mub2d[c]), c2f[k]);
        term = rn_sub(term, rn_mul(loading, base_mass));
    } else {
        term = rn_mul(rdn[k], dp);
        term = rn_sub(term, rn_mul(c1f[k], mup[c]));
    }
#if GPUWM_WRF_EXACT
    real buoy = rn_mul(buoyancy_scale(msft, c, has_msf), term);
#else
    real buoy = rn_mul(G, term);
    if (has_msf) buoy = rn_div(buoy, msft[c]);
#endif
    rw_t[ix] = rn_add(rw_t[ix], buoy);
}

static __device__ __forceinline__
real ph_total(const real* __restrict__ php, const real* __restrict__ phb,
              int k, size_t c, size_t st, int base3d)
{
    return rn_add(base_value(phb, k, c, st, base3d),
                  php[(size_t)k * st + c]);
}

static __device__ __forceinline__
real avg_mut(const real* __restrict__ mup,
             const real* __restrict__ mub2d, size_t ca, size_t cb)
{
#if GPUWM_WRF_EXACT
    return wrf_face_mu(mup, mub2d, ca, cb);
#else
    return rn_mul(0.5f, rn_add(mut_value(mub2d, mup, ca),
                               mut_value(mub2d, mup, cb)));
#endif
}

static __device__
real fcx_value(int face, int j, int k, size_t st, int ny, int nx,
               const real* __restrict__ u,
               const real* __restrict__ mup,
               const real* __restrict__ mub2d,
               const real* __restrict__ c1f,
               const real* __restrict__ c2f,
               const real* __restrict__ msfu, int has_msf,
               int top, real cfn, real cfn1)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int fidx = face % nx;
#else
    int fidx = periodic_face(face, nx);
#endif
    int ia = fidx, ib = (fidx - 1 + nx) % nx;
#else
    int fidx = periodic_face(face, nx);
    int ia = fidx, ib = periodic_previous(fidx, nx);
#endif
    size_t ca = (size_t)j * nx + ia, cb = (size_t)j * nx + ib;
    real mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    size_t f = I3S(k - 1, j, fidx, ny, nx + 1);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        size_t fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(size_t)j * (nx + 1) + fidx]);
    return value;
}

static __device__
real fcy_value(int row, int i, int k, size_t st, int ny, int nx,
               const real* __restrict__ v,
               const real* __restrict__ mup,
               const real* __restrict__ mub2d,
               const real* __restrict__ c1f,
               const real* __restrict__ c2f,
               const real* __restrict__ msfv, int has_msf,
               int top, real cfn, real cfn1)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int ridx = row % ny;
#else
    int ridx = periodic_face(row, ny);
#endif
    int ja = ridx, jb = (ridx - 1 + ny) % ny;
#else
    int ridx = periodic_face(row, ny);
    int ja = ridx, jb = periodic_previous(ridx, ny);
#endif
    size_t ca = (size_t)ja * nx + i, cb = (size_t)jb * nx + i;
    real mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    size_t f = I3S(k - 1, ridx, i, ny + 1, nx);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        size_t fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(size_t)ridx * nx + i]);
    return value;
}

// Compatibility rhs_ph entry points accept already-averaged face masses.
// Keep these separate from fcx/fcy_value so the production fused kernel's
// reconstructed-mass hot path and generated arithmetic remain unchanged.
static __device__
real supplied_fcx_value(int face, int j, int k, int ny, int nx,
                        const real* __restrict__ u,
                        const real* __restrict__ mux,
                        const real* __restrict__ c1f,
                        const real* __restrict__ c2f,
                        const real* __restrict__ msfu, int has_msf,
                        int top, real cfn, real cfn1)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int fidx = face % nx;
#else
    int fidx = periodic_face(face, nx);
#endif
    real mass = rn_add(rn_mul(c1f[k], mux[(size_t)j * (nx + 1) + fidx]),
                       c2f[k]);
    size_t f = I3S(k - 1, j, fidx, ny, nx + 1);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        size_t fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(size_t)j * (nx + 1) + fidx]);
    return value;
}

static __device__
real supplied_fcy_value(int row, int i, int k, int ny, int nx,
                        const real* __restrict__ v,
                        const real* __restrict__ muy,
                        const real* __restrict__ c1f,
                        const real* __restrict__ c2f,
                        const real* __restrict__ msfv, int has_msf,
                        int top, real cfn, real cfn1)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int ridx = row % ny;
#else
    int ridx = periodic_face(row, ny);
#endif
    real mass = rn_add(rn_mul(c1f[k], muy[(size_t)ridx * nx + i]), c2f[k]);
    size_t f = I3S(k - 1, ridx, i, ny + 1, nx);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        size_t fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(size_t)ridx * nx + i]);
    return value;
}

static __device__ __forceinline__
real wrf_phi_field_difference(const real* __restrict__ field, int axis,
                              int i, int j, int k, int offset,
                              int ny, int nx, size_t st, int profile)
{
    if (profile) return 0.0f;
    int pos = axis ? i : j;
    int length = axis ? nx : ny;
    int pp = (pos + offset + length) % length;
    int pm = (pos - offset + length) % length;
    size_t cp = axis ? (size_t)j * nx + pp : (size_t)pp * nx + i;
    size_t cm = axis ? (size_t)j * nx + pm : (size_t)pm * nx + i;
    return rn_sub(field[(size_t)k * st + cp], field[(size_t)k * st + cm]);
}

static __device__ __forceinline__
real x_difference(int i, int offset, int j, int k, size_t st, int nx,
                  const real* __restrict__ php,
                  const real* __restrict__ phb, int base3d)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int ip = (i + offset + nx) % nx;
    int im = (i - offset + nx) % nx;
#else
    int ip = periodic_offset(i, offset, nx);
    int im = periodic_offset(i, -offset, nx);
#endif
    return rn_sub(ph_total(php, phb, k, (size_t)j * nx + ip, st, base3d),
                  ph_total(php, phb, k, (size_t)j * nx + im, st, base3d));
}

static __device__ __forceinline__
real y_difference(int j, int offset, int i, int k, size_t st, int ny, int nx,
                  const real* __restrict__ php,
                  const real* __restrict__ phb, int base3d)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int jp = (j + offset + ny) % ny;
    int jm = (j - offset + ny) % ny;
#else
    int jp = periodic_offset(j, offset, ny);
    int jm = periodic_offset(j, -offset, ny);
#endif
    return rn_sub(ph_total(php, phb, k, (size_t)jp * nx + i, st, base3d),
                  ph_total(php, phb, k, (size_t)jm * nx + i, st, base3d));
}

static __device__ __forceinline__
real centered_x(int i, int j, int k, size_t st, int nx,
                const real* __restrict__ php,
                const real* __restrict__ phb, int base3d,
                real w1, real w2, real w3, real denom)
{
    real t1 = rn_mul(w1, x_difference(i, 1, j, k, st, nx, php, phb, base3d));
    real t2 = rn_mul(w2, x_difference(i, 2, j, k, st, nx, php, phb, base3d));
    real t3 = rn_mul(w3, x_difference(i, 3, j, k, st, nx, php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

static __device__ __forceinline__
real centered_y(int j, int i, int k, size_t st, int ny, int nx,
                const real* __restrict__ php,
                const real* __restrict__ phb, int base3d,
                real w1, real w2, real w3, real denom)
{
    real t1 = rn_mul(w1, y_difference(j, 1, i, k, st, ny, nx,
                                      php, phb, base3d));
    real t2 = rn_mul(w2, y_difference(j, 2, i, k, st, ny, nx,
                                      php, phb, base3d));
    real t3 = rn_mul(w3, y_difference(j, 3, i, k, st, ny, nx,
                                      php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

// Invalid horizontal-advection configurations historically failed only
// after rhs_ph had applied these vertical terms.  This cold-path kernel
// preserves that partial-state contract without branching the fused kernel.
extern "C" __global__
void slow_geopotential_vertical(real* __restrict__ rph_t,
                                const real* __restrict__ ww,
                                const real* __restrict__ w,
                                const real* __restrict__ php,
                                const real* __restrict__ phb,
                                const real* __restrict__ mup,
                                const real* __restrict__ mub2d,
                                const real* __restrict__ rdnw,
                                const real* __restrict__ fnm,
                                const real* __restrict__ fnp,
                                const real* __restrict__ c1f,
                                const real* __restrict__ c2f,
                                const real* __restrict__ msft,
                                int has_msf, int base3d,
                                int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = top ? 0.0f : rph_t[ix];

    if (!top) {
        real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
        real ph_k = ph_total(php, phb, k, c, st, base3d);
        real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
        real wd_lo = rn_mul(rn_mul(rn_mul(0.5f,
                                         rn_add(ww[ix], ww[ix - st])),
                                  rdnw[k - 1]),
                            rn_sub(ph_k, ph_km));
        real wd_hi = rn_mul(rn_mul(rn_mul(0.5f,
                                         rn_add(ww[ix + st], ww[ix])),
                                  rdnw[k]),
                            rn_sub(ph_kp, ph_k));
        real omega = rn_add(rn_mul(fnm[k], wd_hi),
                            rn_mul(fnp[k], wd_lo));
        tendency = rn_sub(tendency, omega);
    }

    real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
    #if GPUWM_WRF_EXACT_C_BIGSTEP
    real gw = rn_mul(rn_mul(mass, G), w[ix]);
#else
    real gw = rn_mul(mass, rn_mul(G, w[ix]));
#endif
    if (has_msf) gw = rn_div(gw, msft[c]);
    rph_t[ix] = rn_add(tendency, gw);
}

// Native rhs_ph on open boundaries. Interior differences retain the
// separate PH/PHB operation order; outer upwinding uses PH alone.
// WRF v4.7.1 evaluates the west top row at kz=kde. No initializer assigns
// FNM/FNP at kde (they loop k=2..kde-1) and the U level there is never
// written, so at run time all three hold their zero allocation value: the
// donor speed is zero and the term subtracts a signed zero, as WRF does.
extern "C" __global__
void slow_geopotential_open(real* rph_t,const real* ww,const real* w,
    const real* u,const real* v,const real* php,const real* phb,
    const real* mup,const real* mub,const real* rdnw,const real* fnm,const real* fnp,
    const real* c1f,const real* c2f,real cfn,real cfn1,
    const real* msft,const real* msfu,const real* msfv,
    const real* mux,const real* muy,real quarter_rdx,real quarter_rdy,
    int mapped,int open_x,int open_y,int supplied,int add_vertical,int order,int base3d,
    int nz,int ny,int nx)
{
    size_t tid=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)ny*nx;
    if(tid>=(size_t)nz*st)return;
    int k=int(tid/st)+1;size_t c=tid-(size_t)(k-1)*st,ix=(size_t)k*st+c;
    int j=int(c/nx),i=int(c-(size_t)j*nx),top=k==nz;
    real tendency=(top&&add_vertical)?0.0f:rph_t[ix];
    real mass=rn_add(rn_mul(c1f[k],mut_value(mub,mup,c)),c2f[k]);
    if(add_vertical) {
        if(!top) {
            real dl=rn_sub(rn_add(rn_sub(php[ix],php[ix-st]),base_value(phb,k,c,st,base3d)),base_value(phb,k-1,c,st,base3d));
            real dh=rn_sub(rn_add(rn_sub(php[ix+st],php[ix]),base_value(phb,k+1,c,st,base3d)),base_value(phb,k,c,st,base3d));
            real lo=rn_mul(rn_mul(rn_mul(0.5f,rn_add(ww[ix],ww[ix-st])),rdnw[k-1]),dl);
            real hi=rn_mul(rn_mul(rn_mul(0.5f,rn_add(ww[ix+st],ww[ix])),rdnw[k]),dh);
            tendency=rn_sub(tendency,rn_add(rn_mul(fnm[k],hi),rn_mul(fnp[k],lo)));
        }
        real gw=rn_mul(rn_mul(mass,G),w[ix]);
        tendency=rn_add(tendency,mapped?rn_div(gw,msft[c]):gw);
    }
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(boundary&&(pos==0||pos==length-1))continue;
        int stencil=(order==2||(boundary&&(pos==1||pos==length-2)))?2:
                    ((boundary&&(pos==2||pos==length-3))?4:6);
        real rd=axis?quarter_rdx:quarter_rdy;
        if(top)rd=rn_mul(2.0f,rd);
        real coefficient=mapped?rn_div(rd,msft[c]):rd;
        real f0,f1;
        if(axis) {
            f0=supplied?supplied_fcx_value(i,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcx_value(i+1,j,k,ny,nx,u,mux,c1f,c2f,msfu,mapped,top,cfn,cfn1):fcx_value(i+1,j,k,st,ny,nx,u,mup,mub,c1f,c2f,msfu,mapped,top,cfn,cfn1);
        } else {
            f0=supplied?supplied_fcy_value(j,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
            f1=supplied?supplied_fcy_value(j+1,i,k,ny,nx,v,muy,c1f,c2f,msfv,mapped,top,cfn,cfn1):fcy_value(j+1,i,k,st,ny,nx,v,mup,mub,c1f,c2f,msfv,mapped,top,cfn,cfn1);
        }
        real contribution;
        if(stencil==2) {
            int pm=(pos-1+length)%length,pp=(pos+1)%length;
            size_t cm=axis?(size_t)j*nx+pm:(size_t)pm*nx+i;
            size_t cp=axis?(size_t)j*nx+pp:(size_t)pp*nx+i;
            real dl=rn_sub(rn_add(rn_sub(base_value(phb,k,c,st,base3d),base_value(phb,k,cm,st,base3d)),php[ix]),php[(size_t)k*st+cm]);
            real dh=rn_sub(rn_add(rn_sub(base_value(phb,k,cp,st,base3d),base_value(phb,k,c,st,base3d)),php[(size_t)k*st+cp]),php[ix]);
            contribution=rn_mul(coefficient,rn_add(rn_mul(f1,dh),rn_mul(f0,dl)));
        } else {
            real p1=wrf_phi_field_difference(php,axis,i,j,k,1,ny,nx,st,0);
            real p2=wrf_phi_field_difference(php,axis,i,j,k,2,ny,nx,st,0);
            real b1=wrf_phi_field_difference(phb,axis,i,j,k,1,ny,nx,st,!base3d);
            real b2=wrf_phi_field_difference(phb,axis,i,j,k,2,ny,nx,st,!base3d),raw,reciprocal;
            if(stencil==4) {
                raw=rn_sub(rn_add(rn_sub(rn_mul(8.0f,p1),p2),rn_mul(8.0f,b1)),b2);
                reciprocal=__fdiv_rn(1.0f,12.0f);
            } else {
                real p3=wrf_phi_field_difference(php,axis,i,j,k,3,ny,nx,st,0);
                real b3=wrf_phi_field_difference(phb,axis,i,j,k,3,ny,nx,st,!base3d);
                raw=rn_add(rn_sub(rn_mul(45.0f,p1),rn_mul(9.0f,p2)),p3);
                raw=rn_add(rn_sub(rn_add(raw,rn_mul(45.0f,b1)),rn_mul(9.0f,b2)),b3);
                reciprocal=__fdiv_rn(1.0f,60.0f);
            }
            contribution=rn_mul(coefficient,rn_mul(rn_mul(rn_add(f1,f0),reciprocal),raw));
        }
        tendency=rn_sub(tendency,contribution);
    }
    // The native routine applies outer y, then outer x after interior terms.
    for(int axis=0;axis<2;++axis) {
        int pos=axis?i:j,length=axis?nx:ny,boundary=axis?open_x:open_y;
        if(!boundary||(pos!=0&&pos!=length-1))continue;
        int west_top=axis&&pos==0&&top;
        int kz=west_top?k:min(k,nz-1);
        real wl,wh,hm=west_top?0.0f:fnm[kz],hp=west_top?0.0f:fnp[kz];
        if(axis) {
            size_t face=(size_t)j*(nx+1)+i,plane=(size_t)ny*(nx+1);
            wh=west_top?0.0f:rn_add(u[(size_t)kz*plane+face+1],u[(size_t)kz*plane+face]);
            wl=rn_add(u[(size_t)(kz-1)*plane+face+1],u[(size_t)(kz-1)*plane+face]);
        } else {
            size_t face=(size_t)j*nx+i,plane=(size_t)(ny+1)*nx;
            wh=rn_add(v[(size_t)kz*plane+face+nx],v[(size_t)kz*plane+face]);
            wl=rn_add(v[(size_t)(kz-1)*plane+face+nx],v[(size_t)(kz-1)*plane+face]);
        }
        real speed=rn_mul(0.5f,rn_add(rn_mul(hm,wh),rn_mul(hp,wl)));
        speed=pos==0?fminf(speed,0.0f):fmaxf(speed,0.0f);
        size_t ci=axis?(pos==0?c+1:c-1):(pos==0?c+nx:c-nx);
        real diff=pos==0?rn_sub(php[(size_t)k*st+ci],php[ix]):rn_sub(php[ix],php[(size_t)k*st+ci]);
        real rd=rn_mul(4.0f,axis?quarter_rdx:quarter_rdy);
        tendency=rn_sub(tendency,rn_mul(rn_mul(rd,mass),rn_mul(speed,diff)));
    }
    rph_t[ix]=tendency;
}

extern "C" __global__
void slow_geopotential(real* __restrict__ rph_t,
                       const real* __restrict__ ww,
                       const real* __restrict__ w,
                       const real* __restrict__ u,
                       const real* __restrict__ v,
                       const real* __restrict__ php,
                       const real* __restrict__ phb,
                       const real* __restrict__ mup,
                       const real* __restrict__ mub2d,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ fnm,
                       const real* __restrict__ fnp,
                       const real* __restrict__ c1f,
                       const real* __restrict__ c2f,
                       real cfn, real cfn1,
                       const real* __restrict__ msft,
                       const real* __restrict__ msfu,
                       const real* __restrict__ msfv,
                       real quarter_rdx, real quarter_rdy,
                       int has_msf, int boundary_x, int boundary_y,
                       int specified, int order, int add_vertical,
                       int base3d, int nz, int ny, int nx)
{
#if GPUWM_WRF_EXACT_C_BIGSTEP
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = (top && add_vertical) ? 0.0f : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }
    if (add_vertical) {
        if (!top) {
            real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            real ph_k = ph_total(php, phb, k, c, st, base3d);
            real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            real wd_lo = rn_mul(rn_mul(rn_mul(0.5f, rn_add(ww[ix], ww[ix - st])), rdnw[k - 1]), rn_sub(rn_add(rn_sub(php[ix], php[ix - st]), base_value(phb, k, c, st, base3d)), base_value(phb, k - 1, c, st, base3d)));
            real wd_hi = rn_mul(rn_mul(rn_mul(0.5f, rn_add(ww[ix + st], ww[ix])), rdnw[k]), rn_sub(rn_add(rn_sub(php[ix + st], php[ix]), base_value(phb, k + 1, c, st, base3d)), base_value(phb, k, c, st, base3d)));
            real omega = rn_add(rn_mul(fnm[k], wd_hi), rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }
        real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        real gw = rn_mul(rn_mul(mass, G), w[ix]);
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }
    // WRF performs the y contribution first, then x. A direction's
    // map-factor division precedes the face sum and centered stencil.
    for (int axis = 0; axis < 2; ++axis) {
        int pos = axis ? i : j;
        int length = axis ? nx : ny;
        int boundary = axis ? boundary_x : boundary_y;
        real quarter_rd = axis ? quarter_rdx : quarter_rdy;
        real coefficient = has_msf ? rn_div(quarter_rd, msft[c]) : quarter_rd;
        int stencil = order == 2 ? 2 : 6;
        if (boundary && (pos == 0 || pos == length - 1)) continue;
        if (order != 2 && specified && (pos < 3 || pos >= length - 3)) {
            if (pos == 1 || pos == length - 2) stencil = 2;
            else if (!axis && (pos == 2 || pos == length - 3)) stencil = 4;
            else continue;
        }
        real face0 = axis ? fcx_value(i, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        real face1 = axis ? fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d, c1f, c2f, msfu, has_msf, top, cfn, cfn1)
                          : fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d, c1f, c2f, msfv, has_msf, top, cfn, cfn1);
        real contribution;
        if (stencil == 2) {
            int pm = (pos - 1 + length) % length;
            int pp = (pos + 1) % length;
            size_t cm = axis ? (size_t)j * nx + pm : (size_t)pm * nx + i;
            size_t cp = axis ? (size_t)j * nx + pp : (size_t)pp * nx + i;
            real d0 = rn_sub(rn_add(rn_sub(base_value(phb, k, c, st, base3d), base_value(phb, k, cm, st, base3d)), php[(size_t)k * st + c]), php[(size_t)k * st + cm]);
            real d1 = rn_sub(rn_add(rn_sub(base_value(phb, k, cp, st, base3d), base_value(phb, k, c, st, base3d)), php[(size_t)k * st + cp]), php[(size_t)k * st + c]);
            real sum = rn_add(rn_mul(face1, d1), rn_mul(face0, d0));
            contribution = rn_mul(coefficient, sum);
        } else {
            real p1 = wrf_phi_field_difference(php, axis, i, j, k, 1, ny, nx, st, 0);
            real p2 = wrf_phi_field_difference(php, axis, i, j, k, 2, ny, nx, st, 0);
            real b1 = wrf_phi_field_difference(phb, axis, i, j, k, 1, ny, nx, st, !base3d);
            real b2 = wrf_phi_field_difference(phb, axis, i, j, k, 2, ny, nx, st, !base3d);
            real raw;
            real reciprocal;
            if (stencil == 4) {
                raw = rn_sub(rn_mul(8.0f, p1), p2);
                raw = rn_sub(rn_add(raw, rn_mul(8.0f, b1)), b2);
                reciprocal = __fdiv_rn(1.0f, 12.0f);
            } else {
                real p3 = wrf_phi_field_difference(php, axis, i, j, k, 3, ny, nx, st, 0);
                real b3 = wrf_phi_field_difference(phb, axis, i, j, k, 3, ny, nx, st, !base3d);
                raw = rn_add(rn_sub(rn_mul(45.0f, p1), rn_mul(9.0f, p2)), p3);
                raw = rn_add(raw, rn_mul(45.0f, b1));
                raw = rn_add(rn_sub(raw, rn_mul(9.0f, b2)), b3);
                reciprocal = __fdiv_rn(1.0f, 60.0f);
            }
            // WRF: (0.25*rd/msft) * ( (face1 + face0) * (1./60.) * (stencil) ),
            // evaluated left to right inside the parentheses.
            real weighted_faces = rn_mul(rn_add(face1, face0), reciprocal);
            contribution = rn_mul(coefficient, rn_mul(weighted_faces, raw));
        }
        tendency = rn_sub(tendency, contribution);
    }
    rph_t[ix] = tendency;

#else
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = (top && add_vertical) ? 0.0f : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }

    if (add_vertical) {
        if (!top) {
            real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            real ph_k = ph_total(php, phb, k, c, st, base3d);
            real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            real wd_lo = rn_mul(rn_mul(rn_mul(0.5f,
                                             rn_add(ww[ix], ww[ix - st])),
                                      rdnw[k - 1]),
                                rn_sub(ph_k, ph_km));
            real wd_hi = rn_mul(rn_mul(rn_mul(0.5f,
                                             rn_add(ww[ix + st], ww[ix])),
                                      rdnw[k]),
                                rn_sub(ph_kp, ph_k));
            real omega = rn_add(rn_mul(fnm[k], wd_hi),
                                rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }

        real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        real gw = rn_mul(mass, rn_mul(G, w[ix]));
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }

    if (order == 2) {
        real dphx;
        if (boundary_x && i == 0) {
            dphx = 0.0f;
        } else {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int im = (i - 1 + nx) % nx;
#else
            int im = periodic_previous(i, nx);
#endif
#else
            int im = periodic_previous(i, nx);
#endif
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)j * nx + im, st, base3d));
        }
        real fx = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(size_t)j * (nx + 1) + i]);
        real fxp;
        if (!(boundary_x && i == nx - 1)) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int ip = (i + 1) % nx;
#else
            int ip = periodic_face(i + 1, nx);
#endif
#else
            int ip = periodic_face(i + 1, nx);
#endif
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(size_t)j * (nx + 1)
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                                       + ((i + 1) % nx)]);
#else
                                       + periodic_face(i + 1, nx)]);
#endif
#else
                                       + periodic_face(i + 1, nx)]);
#endif
        } else {
            fxp = 0.0f;
        }
        real hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        // WRF rhs_ph excludes the entire normal-direction contribution
        // on the outer open/specified row, including its interior face.
        if (boundary_x && (i == 0 || i == nx - 1)) hx = 0.0f;
        tendency = rn_sub(tendency, hx);

        real dphy;
        if (boundary_y && j == 0) {
            dphy = 0.0f;
        } else {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int jm = (j - 1 + ny) % ny;
#else
            int jm = periodic_previous(j, ny);
#endif
#else
            int jm = periodic_previous(j, ny);
#endif
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)jm * nx + i, st, base3d));
        }
        real fy = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(size_t)j * nx + i]);
        real fyp;
        if (!(boundary_y && j == ny - 1)) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int jp = (j + 1) % ny;
#else
            int jp = periodic_face(j + 1, ny);
#endif
#else
            int jp = periodic_face(j + 1, ny);
#endif
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                fyp = rn_mul(fyp, msfv[(size_t)((j + 1) % ny) * nx + i]);
#else
                fyp = rn_mul(fyp, msfv[(size_t)periodic_face(j + 1, ny) * nx + i]);
#endif
#else
                fyp = rn_mul(fyp, msfv[(size_t)periodic_face(j + 1, ny) * nx + i]);
#endif
        } else {
            fyp = 0.0f;
        }
        real hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        if (boundary_y && (j == 0 || j == ny - 1)) hy = 0.0f;
        tendency = rn_sub(tendency, hy);
    } else {
        real hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
#else
                int im = periodic_previous(i, nx), ip = periodic_face(i + 1, nx);
#endif
#else
                int im = periodic_previous(i, nx), ip = periodic_face(i + 1, nx);
#endif
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)j * nx + im, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d0);
                real f1 = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = 0.0f;
            }
        } else {
            real csx = rn_add(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                        c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1),
                              fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                        mub2d, c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1));
            real diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }

        real hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
#else
                int jm = periodic_previous(j, ny), jp = periodic_face(j + 1, ny);
#endif
#else
                int jm = periodic_previous(j, ny), jp = periodic_face(j + 1, ny);
#endif
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)jm * nx + i, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d0);
                real f1 = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                real csy = rn_add(
                    fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1),
                    fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1));
                real diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, 8.0f, -1.0f, 0.0f, 12.0f);
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = 0.0f;
            }
        } else {
            real csy = rn_add(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                        c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1),
                              fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                        mub2d, c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1));
            real diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;

#endif
}

extern "C" __global__
void slow_geopotential_faces(real* __restrict__ rph_t,
                             const real* __restrict__ u,
                             const real* __restrict__ v,
                             const real* __restrict__ php,
                             const real* __restrict__ phb,
                             const real* __restrict__ mux,
                             const real* __restrict__ muy,
                             const real* __restrict__ c1f,
                             const real* __restrict__ c2f,
                             real cfn, real cfn1,
                             const real* __restrict__ msft,
                             const real* __restrict__ msfu,
                             const real* __restrict__ msfv,
                             real quarter_rdx, real quarter_rdy,
                             int has_msf, int boundary_x, int boundary_y,
                             int specified, int order, int base3d,
                             int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }

    if (order == 2) {
        real dphx;
        if (boundary_x && i == 0) {
            dphx = 0.0f;
        } else {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int im = (i - 1 + nx) % nx;
#else
            int im = periodic_previous(i, nx);
#endif
#else
            int im = periodic_previous(i, nx);
#endif
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)j * nx + im, st, base3d));
        }
        real fx = rn_mul(supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(size_t)j * (nx + 1) + i]);
        real fxp;
        if (!(boundary_x && i == nx - 1)) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int ip = (i + 1) % nx;
#else
            int ip = periodic_face(i + 1, nx);
#endif
#else
            int ip = periodic_face(i + 1, nx);
#endif
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(size_t)j * (nx + 1)
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                                       + ((i + 1) % nx)]);
#else
                                       + periodic_face(i + 1, nx)]);
#endif
#else
                                       + periodic_face(i + 1, nx)]);
#endif
        } else {
            fxp = 0.0f;
        }
        real hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        if (boundary_x && (i == 0 || i == nx - 1)) hx = 0.0f;
        tendency = rn_sub(tendency, hx);

        real dphy;
        if (boundary_y && j == 0) {
            dphy = 0.0f;
        } else {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int jm = (j - 1 + ny) % ny;
#else
            int jm = periodic_previous(j, ny);
#endif
#else
            int jm = periodic_previous(j, ny);
#endif
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)jm * nx + i, st, base3d));
        }
        real fy = rn_mul(supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(size_t)j * nx + i]);
        real fyp;
        if (!(boundary_y && j == ny - 1)) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
            int jp = (j + 1) % ny;
#else
            int jp = periodic_face(j + 1, ny);
#endif
#else
            int jp = periodic_face(j + 1, ny);
#endif
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                fyp = rn_mul(fyp, msfv[(size_t)((j + 1) % ny) * nx + i]);
#else
                fyp = rn_mul(fyp, msfv[(size_t)periodic_face(j + 1, ny) * nx + i]);
#endif
#else
                fyp = rn_mul(fyp, msfv[(size_t)periodic_face(j + 1, ny) * nx + i]);
#endif
        } else {
            fyp = 0.0f;
        }
        real hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        if (boundary_y && (j == 0 || j == ny - 1)) hy = 0.0f;
        tendency = rn_sub(tendency, hy);
    } else {
        real hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
#else
                int im = periodic_previous(i, nx), ip = periodic_face(i + 1, nx);
#endif
#else
                int im = periodic_previous(i, nx), ip = periodic_face(i + 1, nx);
#endif
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)j * nx + im, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(supplied_fcx_value(
                                     i, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d0);
                real f1 = rn_mul(supplied_fcx_value(
                                     i + 1, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = 0.0f;
            }
        } else {
            real csx = rn_add(
                supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1),
                supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1));
            real diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }

        real hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
#else
                int jm = periodic_previous(j, ny), jp = periodic_face(j + 1, ny);
#endif
#else
                int jm = periodic_previous(j, ny), jp = periodic_face(j + 1, ny);
#endif
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)jm * nx + i, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(supplied_fcy_value(
                                     j, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d0);
                real f1 = rn_mul(supplied_fcy_value(
                                     j + 1, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                real csy = rn_add(
                    supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1),
                    supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1));
                real diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, 8.0f, -1.0f, 0.0f, 12.0f);
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = 0.0f;
            }
        } else {
            real csy = rn_add(
                supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1),
                supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1));
            real diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

// ---- RK small-step preparation/finalization -------------------------------
//
// The former CuPy implementation materialized every binary operator below as
// a separate FP32 array.  rn_* preserves each of those store/load rounding
// boundaries explicitly, so batching the pointwise chains changes neither
// expression order nor contraction while removing the temporary traffic.
//
// THE FACE-MASS NEIGHBOUR IS BOUNDARY-DEPENDENT.  Both uv kernels below build
// WRF's muu/muus and muv/muvs inline -- the column mass carried to a u/v face
// as the mean of the two mass cells that share it (module_small_step_em.F
// small_step_prep :225-228 and small_step_finish :96-98 consume exactly those
// arrays).  WRF builds them in calc_mu_uv
// (module_big_step_utilities_em.F:59-115 x / :118-174 y) and that routine
// chooses the off-domain neighbour from ``config_flags%periodic_x`` /
// ``periodic_y``:
//
//   west face  i = ids:  im = its   normally, its-1 under periodic_x
//   east face  i = ide:  im = ite-1 normally, ite   under periodic_x
//
// The periodic branch reads the halo cell, which the period fill has loaded
// with the opposite edge -- the ``i % nx`` / ``(i-1+nx) % nx`` below.  The
// NON-periodic branch names the boundary cell twice, so the boundary face
// carries that cell's own mass: a zero-gradient ghost, which is min/max
// clamping of the same two indices.  ``advance_mu_th`` (kernels/acoustic.cu)
// already spells the identical choice for its four face masses.
//
// Until this flag arrived both kernels wrapped unconditionally, so on a
// specified, nested or open-lateral domain the west boundary face took its
// mass from the EAST-most mass column and vice versa -- a real coupling of
// the two physical boundaries, at both ends of every acoustic rung of every
// RK stage.  These were the last two sites where a wrapped index SURVIVED on
// a non-periodic domain: ``advance_uv`` (kernels/acoustic.cu) also spells
// ``i % nx``, but under specified/nested its spec_zone guard never reaches
// the boundary face, and under open boundaries acoustic.py overwrites both
// face sets from their saved pre-substep values immediately after the
// launch.  MEASURED after the fix, 96x64x24 dry, one step, 25 Pa on one
// west mass column, over all 76 device arrays the state owns: the influence
// stops at column 10 on an open domain and at column 0 on a specified one,
// while the periodic control still reaches the far edge in 33 arrays.
// Before the fix the same probe moved the far edge in 34 open-domain arrays
// and in the two stage mass fluxes of a specified one.
//
// The two boundary-normal face columns are the whole footprint of the
// change: interior faces 1 <= i <= nx-1 index i-1 and i on both branches, so
// a periodic run is bit-identical (proved by tilestream/test_gate.py, which
// is periodic throughout, and by tests/test_small_step_lateral_wrap.py's
// periodic rows).  A specified domain moves by exactly one float32 ULP at
// one step, in 22 of 148992 ``ru`` faces and 26 of 149760 ``rv`` faces --
// all of them at i = 0 / i = nx and j = 0 / j = ny, free interior exactly
// zero; the persisted state first differs at step 3 and by step 8 is
// 1.1e-3 m/s in u and 1.9e-2 Pa in mu.

extern "C" __global__
void small_step_init_uv(real* __restrict__ u_pp,
                        real* __restrict__ v_pp,
                        const real* __restrict__ u0,
                        const real* __restrict__ v0,
                        const real* __restrict__ u,
                        const real* __restrict__ v,
                        const real* __restrict__ mup0,
                        const real* __restrict__ mup,
                        const real* __restrict__ mub2d,
                        const real* __restrict__ c1h,
                        const real* __restrict__ c2h,
                        const real* __restrict__ msfu,
                        const real* __restrict__ msfv,
                        int has_msf, int boundary_x, int boundary_y,
                        int nz, int ny, int nx
#if GPUWM_WRF_EXACT
                        , int rk_step
#endif
                        )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;

    if (j < ny) {
        int ia = boundary_x ? min(i, nx - 1) : i % nx;
        int ib = boundary_x ? max(i - 1, 0)  : (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ia;
        size_t cb = (size_t)j * nx + ib;
        real mt_a = rn_add(mub2d[ca], mup0[ca]);
        real mt_b = rn_add(mub2d[cb], mup0[cb]);
        real ms_a = rn_add(mub2d[ca], mup[ca]);
        real ms_b = rn_add(mub2d[cb], mup[cb]);
        real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));
        real msf = rn_mul(0.5f, rn_add(ms_a, ms_b));
#if GPUWM_WRF_EXACT
        msf = rn_mul(0.5f, rn_add(rn_add(rn_add(mup[ca], mup[cb]),
                                              mub2d[ca]), mub2d[cb]));
        mtf = rk_step == 1 ? msf : rn_mul(0.5f,
            rn_add(rn_add(rn_add(mub2d[ca], mup0[ca]), mub2d[cb]), mup0[cb]));
#endif
        real ct = rn_add(rn_mul(c1h[k], mtf), c2h[k]);
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        size_t ix = I3S(k, j, i, ny, nxf);
        real value = rn_sub(rn_mul(ct, u0[ix]), rn_mul(cs, u[ix]));
        if (has_msf) value = rn_div(value, msfu[(size_t)j * nxf + i]);
        u_pp[ix] = value;
    }
    if (i < nx) {
        int ja = boundary_y ? min(j, ny - 1) : j % ny;
        int jb = boundary_y ? max(j - 1, 0)  : (j - 1 + ny) % ny;
        size_t ca = (size_t)ja * nx + i;
        size_t cb = (size_t)jb * nx + i;
        real mt_a = rn_add(mub2d[ca], mup0[ca]);
        real mt_b = rn_add(mub2d[cb], mup0[cb]);
        real ms_a = rn_add(mub2d[ca], mup[ca]);
        real ms_b = rn_add(mub2d[cb], mup[cb]);
        real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));
        real msf = rn_mul(0.5f, rn_add(ms_a, ms_b));
#if GPUWM_WRF_EXACT
        msf = rn_mul(0.5f, rn_add(rn_add(rn_add(mup[ca], mup[cb]),
                                              mub2d[ca]), mub2d[cb]));
        mtf = rk_step == 1 ? msf : rn_mul(0.5f,
            rn_add(rn_add(rn_add(mub2d[ca], mup0[ca]), mub2d[cb]), mup0[cb]));
#endif
        real ct = rn_add(rn_mul(c1h[k], mtf), c2h[k]);
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        size_t ix = I3S(k, j, i, nyf, nx);
        real value = rn_sub(rn_mul(ct, v0[ix]), rn_mul(cs, v[ix]));
#if GPUWM_WRF_EXACT
        if (has_msf) {
            value = rn_mul(value, msfv[(size_t)j * nx + i]); // strict MF_VX_INV
        }
#else
        if (has_msf) value = rn_div(value, msfv[(size_t)j * nx + i]);
#endif
        v_pp[ix] = value;
    }
}

extern "C" __global__
void small_step_init_column(real* __restrict__ w_pp,
                            real* __restrict__ th_pp,
                            real* __restrict__ ph_pp,
                            real* __restrict__ mu_pp,
                            real* __restrict__ al_pp,
                            real* __restrict__ p_pp,
                            real* __restrict__ p_pp_old,
                            const real* __restrict__ w0,
                            const real* __restrict__ w,
                            const real* __restrict__ thp0,
                            const real* __restrict__ thp,
                            const real* __restrict__ php0,
                            const real* __restrict__ php,
                            const real* __restrict__ p,
                            const real* __restrict__ alt,
                            const real* __restrict__ mup0,
                            const real* __restrict__ mup,
                            const real* __restrict__ mub2d,
                            const real* __restrict__ thb,
                            const real* __restrict__ c1h,
                            const real* __restrict__ c2h,
                            const real* __restrict__ c1f,
                            const real* __restrict__ c2f,
                            const real* __restrict__ rdnw,
                            const real* __restrict__ msft,
                            int has_msf, int base3d,
                            int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mut = rn_add(mub2d[c], mup0[c]);
    real mus = rn_add(mub2d[c], mup[c]);
    real mupp = rn_sub(mup0[c], mup[c]);
    mu_pp[c] = mupp;

    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        real cft = rn_add(rn_mul(c1f[k], mut), c2f[k]);
        real cfs = rn_add(rn_mul(c1f[k], mus), c2f[k]);
        real value = rn_sub(rn_mul(cft, w0[f]), rn_mul(cfs, w[f]));
        if (has_msf) value = rn_div(value, msft[c]);
        w_pp[f] = value;
        ph_pp[f] = rn_sub(php0[f], php[f]);
    }

    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * bstr + boff;
        real cht = rn_add(rn_mul(c1h[k], mut), c2h[k]);
        real chs = rn_add(rn_mul(c1h[k], mus), c2h[k]);
        real tht = rn_add(thb[b], thp0[h]);
        real ths = rn_add(thb[b], thp[h]);
#if GPUWM_WRF_EXACT
        tht = thp0[h];
        ths = thp[h];
#endif
        real thpp = rn_sub(rn_mul(cht, tht), rn_mul(chs, ths));
        th_pp[h] = thpp;

        real c2a = rn_div(rn_mul(GAMMA, p[h]), alt[h]);
        real al0 = rn_mul(c1h[k], mupp);
        real al1 = rn_mul(alt[h], al0);
        real dph = rn_sub(ph_pp[h + st], ph_pp[h]);
        real al2 = rn_mul(rdnw[k], dph);
        real al = rn_div(-rn_add(al1, al2), cht);
#if GPUWM_WRF_EXACT
        al = rn_mul(rn_div(-1.0f, cht), rn_add(al1, al2));
#endif
        al_pp[h] = al;

        real p0 = rn_mul(c1h[k], mupp);
        real p1 = rn_mul(p0, ths);
        real p2 = rn_sub(thpp, p1);
        real p3 = rn_mul(alt[h], p2);
        real p4 = rn_mul(cht, ths);
#if GPUWM_WRF_EXACT
        p4 = rn_mul(cht, rn_add(300.0f, ths));
#endif
        real p5 = rn_div(p3, p4);
        real p6 = rn_sub(p5, al);
        real pnew = rn_mul(c2a, p6);
        p_pp[h] = pnew;
        p_pp_old[h] = pnew;
    }
}

extern "C" __global__
void small_step_finish_uv(real* __restrict__ u,
                          real* __restrict__ v,
                          const real* __restrict__ u_pp,
                          const real* __restrict__ v_pp,
                          const real* __restrict__ mup,
                          const real* __restrict__ mu_pp,
                          const real* __restrict__ mub2d,
                          const real* __restrict__ c1h,
                          const real* __restrict__ c2h,
                          const real* __restrict__ msfu,
                          const real* __restrict__ msfv,
                          int has_msf, int boundary_x, int boundary_y,
#if GPUWM_WRF_EXACT
                          int nz, int ny, int nx,
                          const real* __restrict__ muts_wrf)
#else
                          int nz, int ny, int nx)
#endif
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;

    if (j < ny) {
        int ia = boundary_x ? min(i, nx - 1) : i % nx;
        int ib = boundary_x ? max(i - 1, 0)  : (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ia;
        size_t cb = (size_t)j * nx + ib;
        real msa = rn_add(mub2d[ca], mup[ca]);
        real msb = rn_add(mub2d[cb], mup[cb]);
#if GPUWM_WRF_EXACT
        // calc_mu_uv_1 of WRF's grid%muts (the ring's is its own carrier).
        real mna = muts_wrf[ca];
        real mnb = muts_wrf[cb];
#else
        real mna = rn_add(msa, mu_pp[ca]);
        real mnb = rn_add(msb, mu_pp[cb]);
#endif
        real msf = rn_mul(0.5f, rn_add(msa, msb));
        real mnf = rn_mul(0.5f, rn_add(mna, mnb));
#if GPUWM_WRF_EXACT
        // muu: WRF calc_mu_uv, 0.5*(mu(i)+mu(i-1)+mub(i)+mub(i-1)).
        // muus: calc_mu_uv_1 of muts = (mub+mu)+mu'' (advance_mu_t,
        // module_small_step_em.F:1106; small_step_finish :392), which is
        // the mna/mnb order above -- a regrouped muus differed from WRF in
        // 12 % of face masses (combo-sweep round2 LOCALIZE.md, cause 7).
        msf = rn_mul(0.5f, rn_add(rn_add(rn_add(mup[ca], mup[cb]),
                                              mub2d[ca]), mub2d[cb]));
#endif
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        real cn = rn_add(rn_mul(c1h[k], mnf), c2h[k]);
        size_t ix = I3S(k, j, i, ny, nxf);
        real pp = u_pp[ix];
        if (has_msf) pp = rn_mul(pp, msfu[(size_t)j * nxf + i]);
        u[ix] = rn_div(rn_add(rn_mul(cs, u[ix]), pp), cn);
    }
    if (i < nx) {
        int ja = boundary_y ? min(j, ny - 1) : j % ny;
        int jb = boundary_y ? max(j - 1, 0)  : (j - 1 + ny) % ny;
        size_t ca = (size_t)ja * nx + i;
        size_t cb = (size_t)jb * nx + i;
        real msa = rn_add(mub2d[ca], mup[ca]);
        real msb = rn_add(mub2d[cb], mup[cb]);
#if GPUWM_WRF_EXACT
        // calc_mu_uv_1 of WRF's grid%muts (the ring's is its own carrier).
        real mna = muts_wrf[ca];
        real mnb = muts_wrf[cb];
#else
        real mna = rn_add(msa, mu_pp[ca]);
        real mnb = rn_add(msb, mu_pp[cb]);
#endif
        real msf = rn_mul(0.5f, rn_add(msa, msb));
        real mnf = rn_mul(0.5f, rn_add(mna, mnb));
#if GPUWM_WRF_EXACT
        // muu: WRF calc_mu_uv, 0.5*(mu(i)+mu(i-1)+mub(i)+mub(i-1)).
        // muus: calc_mu_uv_1 of muts = (mub+mu)+mu'' (advance_mu_t,
        // module_small_step_em.F:1106; small_step_finish :392), which is
        // the mna/mnb order above -- a regrouped muus differed from WRF in
        // 12 % of face masses (combo-sweep round2 LOCALIZE.md, cause 7).
        msf = rn_mul(0.5f, rn_add(rn_add(rn_add(mup[ca], mup[cb]),
                                              mub2d[ca]), mub2d[cb]));
#endif
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        real cn = rn_add(rn_mul(c1h[k], mnf), c2h[k]);
        size_t ix = I3S(k, j, i, nyf, nx);
        real pp = v_pp[ix];
        if (has_msf) pp = rn_mul(pp, msfv[(size_t)j * nx + i]);
        v[ix] = rn_div(rn_add(rn_mul(cs, v[ix]), pp), cn);
    }
}

extern "C" __global__
void small_step_finish_column(real* __restrict__ w,
                              real* __restrict__ thp,
                              real* __restrict__ php,
                              real* __restrict__ mup,
                              const real* __restrict__ w_pp,
                              const real* __restrict__ th_pp,
                              const real* __restrict__ ph_pp,
                              const real* __restrict__ mu_pp,
                              const real* __restrict__ mub2d,
                              const real* __restrict__ thb,
                              const real* __restrict__ c1h,
                              const real* __restrict__ c2h,
                              const real* __restrict__ c1f,
                              const real* __restrict__ c2f,
                              const real* __restrict__ msft,
                              const real* __restrict__ h_diabatic,
                              real hdiab_dt, int remove_hdiab,
                              int has_msf, int base3d,
#if GPUWM_WRF_EXACT
                              int nz, int ny, int nx,
                              const real* __restrict__ muts_wrf)
#else
                              int nz, int ny, int nx)
#endif
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mus = rn_add(mub2d[c], mup[c]);
#if GPUWM_WRF_EXACT
    real mun = muts_wrf[c];
#else
    real mun = rn_add(mus, mu_pp[c]);
#endif

    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        real pp = w_pp[f];
        if (has_msf) pp = rn_mul(pp, msft[c]);
        real cs = rn_add(rn_mul(c1f[k], mus), c2f[k]);
        real cn = rn_add(rn_mul(c1f[k], mun), c2f[k]);
        w[f] = rn_div(rn_add(rn_mul(cs, w[f]), pp), cn);
        php[f] = rn_add(php[f], ph_pp[f]);
    }

    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * bstr + boff;
        real cs = rn_add(rn_mul(c1h[k], mus), c2h[k]);
        real theta = rn_add(thb[b], thp[h]);
        real numerator = rn_add(rn_mul(cs, theta), th_pp[h]);
#if GPUWM_WRF_EXACT
        theta = thp[h];
        numerator = th_pp[h];
        if (remove_hdiab) {
            real removal = rn_mul(rn_mul(hdiab_dt, cs), h_diabatic[h]);
            numerator = rn_sub(numerator, removal);
        }
        numerator = rn_add(numerator, rn_mul(theta, cs));
#else
        if (remove_hdiab) {
            real removal = rn_mul(rn_mul(hdiab_dt, cs), h_diabatic[h]);
            numerator = rn_sub(numerator, removal);
        }
#endif
        real cn = rn_add(rn_mul(c1h[k], mun), c2h[k]);
#if GPUWM_WRF_EXACT
        thp[h] = rn_div(numerator, cn);
#else
        thp[h] = rn_sub(rn_div(numerator, cn), thb[b]);
#endif
    }
    mup[c] = rn_add(mup[c], mu_pp[c]);
}
