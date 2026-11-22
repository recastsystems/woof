// gpuwm/core/kernels/sase.cu
//
// SASE-L1 device kernels (stage-3 Tasks 2-4 + 6, amended by S3-6c):
// horizontal top-hat test filters, structure-function block partial
// sums, strain (clamped uniform/variable dz), Germano-lift helpers,
// the deviatoric model stress, the dynamic-solve reductions, the S3-6c
// SPLIT-STEP kernels (vertical channel K_v/l_B, horizontal-explicit
// tendencies + P_h, FP64 per-column backward-Euler Thomas sweeps for
// the implicit vertical channel, implicit-flux production P_v, the
// split e update + FP64 ledger partials), and the driver-coupling
// kernels (per-column dz coefficients, N^2, horizontal K_h scalar-mix
// flux/divergence).  FP32 (`real`) throughout except the reduction
// accumulators (FP64 per block, host-side final sum) and the Thomas
// column sweeps, which run FP64 end to end (S3-6b concern 3).
// FP64 verification authority: gpuwm.verify.sase_ref (the split step
// mirrors sase_split_step); parity gates live in tests/test_sase_gpu.py.
// Array layout (nz, ny, nx), x fastest; horizontal directions wrap
// periodically; the vertical is CLAMPED in every mode -- the v0
// roll-based periodic vertical (z_mode 2) retired with the explicit
// fused step (S3-6b report section 7: the split ledger theorem needs
// no periodic vertical).

// SASE_TPB and SASE_KMAX are injected as a compile-time tier by the
// launcher (gpuwm/core/sase.py _INT_DEFINES, through
// get_kernel_int_defines), so the block size and shared-memory extents
// here and the host launch configs there cannot drift.
//
// The #ifndef guards below are the same idiom kf.cu and refl.cu use for
// their own level bounds, and they exist for a specific reason rather
// than as defensive habit: the local-frame census
// (tests/test_preflight.py::test_the_recorded_local_frames_match_the_
// driver) compiles every translation unit in this directory through the
// PLAIN loader to measure its per-thread frame, and a unit that cannot
// compile without a tier is invisible to it.  A kernel that grows its
// frame silently moves the whole process's device footprint by
// gigabytes, so being measurable matters more than being unbuildable
// without the launcher.  The defaults MUST equal the launcher's tier;
// tests/test_sase_gpu.py::test_sase_tpb_single_source pins that.
#ifndef SASE_TPB
#define SASE_TPB 128
#endif
#ifndef SASE_KMAX
#define SASE_KMAX 128
#endif

// Top-hat test filter of nominal width 2 or 4 grid cells, x then y
// (authority box_filter): weights [1/4, 1/2, 1/4] or
// [1/8, 1/4, 1/4, 1/4, 1/8], periodic wrap in x and y, vertical
// untouched.  Fused separable form: the inner x-weighted row sum is the
// x-pass value at (k, jc, i); the outer loop applies the y weights.
extern "C" __global__
void sase_box_filter(const real* __restrict__ f, real* __restrict__ out,
                     int width, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const real w2[3] = {0.25f, 0.5f, 0.25f};
    const real w4[5] = {0.125f, 0.25f, 0.25f, 0.25f, 0.125f};
    const real* wt = (width == 2) ? w2 : w4;
    int half = (width == 2) ? 1 : 2;
    real acc = 0.0f;
    for (int jj = -half; jj <= half; ++jj) {
        int jc = PERIODIC(j + jj, ny);
        real row = 0.0f;
        for (int ii = -half; ii <= half; ++ii)
            row += wt[ii + half] * f[IDX3(k, jc, PERIODIC(i + ii, nx))];
        acc += wt[jj + half] * row;
    }
    out[IDX3(k, j, i)] = acc;
}

// Structure-function block partial sums: for one velocity component,
// accumulate sum(dx_inc^2 + dy_inc^2) over all cells for r in {1, 2, 4}
// (horizontal, periodic).  Increments are FP32 (production arithmetic);
// squares and sums are FP64.  partials has shape (3, nblocks); the host
// finishes with 0.5 * sum(partials[t]) / ncell per component (the
// authority's 0.5*(mean_x + mean_y)).
extern "C" __global__
void sase_structure_partial(const real* __restrict__ f,
                            double* __restrict__ partials,
                            int nz, int ny, int nx, int nblocks)
{
    __shared__ double sdata[3][SASE_TPB];
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncell = (long long)nz * ny * nx;
    double acc[3] = {0.0, 0.0, 0.0};
    if (idx < ncell) {
        int i = (int)(idx % nx);
        int j = (int)((idx / nx) % ny);
        int k = (int)(idx / ((long long)nx * ny));
        real fc = f[idx];
        const int rs[3] = {1, 2, 4};
        for (int t = 0; t < 3; ++t) {
            real dxr = f[IDX3(k, j, PERIODIC(i + rs[t], nx))] - fc;
            real dyr = f[IDX3(k, PERIODIC(j + rs[t], ny), i)] - fc;
            acc[t] = (double)dxr * (double)dxr + (double)dyr * (double)dyr;
        }
    }
    for (int t = 0; t < 3; ++t) sdata[t][threadIdx.x] = acc[t];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int t = 0; t < 3; ++t)
                sdata[t][threadIdx.x] += sdata[t][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int t = 0; t < 3; ++t)
            partials[(size_t)t * nblocks + blockIdx.x] = sdata[t][0];
}

// Vertical neighbor indices: every surviving z_mode (0/1/3) clamps to
// the boundary rows (the edge branches of sase_ddz ignore the clamped
// reads).  The z_mode-2 periodic wrap retired with the explicit fused
// step (S3-6c); the z_mode argument stays so call sites keep one
// signature shape with sase_ddz.
__device__ __forceinline__ int sase_km(int k, int nz, int z_mode)
{
    (void)z_mode;
    return (k > 0) ? k - 1 : 0;
}

__device__ __forceinline__ int sase_kp(int k, int nz, int z_mode)
{
    (void)z_mode;
    return (k < nz - 1) ? k + 1 : nz - 1;
}

// Vertical derivative at level k from the three vertical neighbors.
// z_mode=0 mirrors the authority's uniform clamped expressions:
// (f[k+1]-f[k-1])/(2*dz) interior, one-sided (f1-f0)/dz at the edges.
// z_mode=1 is the variable-spacing three-point Lagrange stencil in
// COEFFICIENT FORM: cm/c0/cp are precomputed on host in FP64 exactly as
// _ddz_var groups them (cm = -(h_p/(h_m*(h_p+h_m))), c0 =
// (h_p-h_m)/(h_p*h_m), cp = h_m/(h_p*(h_p+h_m)) on center spacings from
// the cumsum-half-layer construction), then cast to FP32; edge rows are
// one-sided over the FP64-computed edge center spacings h_lo/h_hi.
// z_mode=3 (stage-3 Task 6, the model's terrain-following columns) is
// the SAME Lagrange coefficient form with PER-CELL (nz,ny,nx)
// coefficient fields indexed by the flat cell index q
// (sase_ddz_coefficients builds them on device in FP64); the clamped
// edge rows are FOLDED INTO the coefficient rows (k=0: cm=0,
// c0=-1/h_lo, cp=+1/h_lo with the clamped fm read multiplied by zero;
// k=nz-1 mirrored), so the branch-free per-cell expression covers every
// level.  The z_mode 0/1/3 arithmetic is untouched from Task 6 -- the
// pinned device goldens stay bitwise; the retired z_mode-2 roll-based
// periodic branch (v0 conservation-ledger box) was removed with the
// explicit fused step (S3-6c).
__device__ __forceinline__ real
sase_ddz(real fm, real fc, real fp, int k, size_t q, int nz,
         const real* __restrict__ cm, const real* __restrict__ c0,
         const real* __restrict__ cp, real dz, real two_dz,
         real h_lo, real h_hi, int z_mode)
{
    if (z_mode == 3) return cm[q] * fm + c0[q] * fc + cp[q] * fp;
    if (k == 0)      return (fp - fc) / (z_mode ? h_lo : dz);
    if (k == nz - 1) return (fc - fm) / (z_mode ? h_hi : dz);
    if (z_mode)      return cm[k] * fm + c0[k] * fc + cp[k] * fp;
    return (fp - fm) / two_dz;
}

// Per-column FP64 build of the z_mode=3 coefficient fields from FP32
// layer thicknesses t (nz,ny,nx).  Center spacings come DIRECTLY from
// the thicknesses -- h_m(k) = z_k - z_{k-1} = 0.5*(t[k-1] + t[k]) by
// the telescoping of the cumsum-half-layer construction (exact in
// real arithmetic; the host's FP64 cumsum-then-difference grouping
// can round differently in the last ULP -- S3-6c review Minor, inside
// the coefficient-cast tolerance) -- so no cumulative sum is needed;
// FP32 thicknesses promote to FP64 exactly and only the final
// coefficient cast meets FP32 (matching the host _ddz_coefficients
// precision contract without a per-step host round trip of the
// model's evolving dz field).  Edge
// rows fold the one-sided two-point stencil into coefficient form (see
// sase_ddz z_mode=3).  One thread per column.
extern "C" __global__
void sase_ddz_coefficients(const real* __restrict__ t,
                           real* __restrict__ cm, real* __restrict__ c0,
                           real* __restrict__ cp,
                           int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double tk = (double)t[q];
        if (k == 0) {
            double h = 0.5 * (tk + (double)t[q + ncol]);
            cm[q] = 0.0f;
            c0[q] = (real)(-1.0 / h);
            cp[q] = (real)(1.0 / h);
        } else if (k == nz - 1) {
            double h = 0.5 * ((double)t[q - ncol] + tk);
            cm[q] = (real)(-1.0 / h);
            c0[q] = (real)(1.0 / h);
            cp[q] = 0.0f;
        } else {
            double hm = 0.5 * ((double)t[q - ncol] + tk);
            double hp = 0.5 * (tk + (double)t[q + ncol]);
            cm[q] = (real)(-(hp / (hm * (hp + hm))));
            c0[q] = (real)((hp - hm) / (hp * hm));
            cp[q] = (real)(hm / (hp * (hp + hm)));
        }
    }
}

// Resolved strain tensor [xx, yy, zz, xy, xz, yz], centered horizontal
// differences (periodic) and the clamped z_mode vertical above
// (authority strain; the split step is clamped-vertical in BOTH modes).
extern "C" __global__
void sase_strain(const real* __restrict__ u, const real* __restrict__ v,
                 const real* __restrict__ w,
                 real* __restrict__ sxx, real* __restrict__ syy,
                 real* __restrict__ szz, real* __restrict__ sxy,
                 real* __restrict__ sxz, real* __restrict__ syz,
                 const real* __restrict__ cm, const real* __restrict__ c0,
                 const real* __restrict__ cp,
                 real two_dx, real two_dy, real dz, real two_dz,
                 real h_lo, real h_hi, int z_mode,
                 int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    int km = sase_km(k, nz, z_mode), kp = sase_kp(k, nz, z_mode);
    size_t q = IDX3(k, j, i);
    real dudx = (u[IDX3(k, j, ip)] - u[IDX3(k, j, im)]) / two_dx;
    real dvdx = (v[IDX3(k, j, ip)] - v[IDX3(k, j, im)]) / two_dx;
    real dwdx = (w[IDX3(k, j, ip)] - w[IDX3(k, j, im)]) / two_dx;
    real dudy = (u[IDX3(k, jp, i)] - u[IDX3(k, jm, i)]) / two_dy;
    real dvdy = (v[IDX3(k, jp, i)] - v[IDX3(k, jm, i)]) / two_dy;
    real dwdy = (w[IDX3(k, jp, i)] - w[IDX3(k, jm, i)]) / two_dy;
    real dudz = sase_ddz(u[IDX3(km, j, i)], u[q], u[IDX3(kp, j, i)],
                         k, q, nz, cm, c0, cp, dz, two_dz, h_lo, h_hi,
                         z_mode);
    real dvdz = sase_ddz(v[IDX3(km, j, i)], v[q], v[IDX3(kp, j, i)],
                         k, q, nz, cm, c0, cp, dz, two_dz, h_lo, h_hi,
                         z_mode);
    real dwdz = sase_ddz(w[IDX3(km, j, i)], w[q], w[IDX3(kp, j, i)],
                         k, q, nz, cm, c0, cp, dz, two_dz, h_lo, h_hi,
                         z_mode);
    sxx[q] = dudx;
    syy[q] = dvdy;
    szz[q] = dwdz;
    sxy[q] = 0.5f * (dudy + dvdx);
    sxz[q] = 0.5f * (dudz + dwdx);
    syz[q] = 0.5f * (dvdz + dwdy);
}

// Pointwise velocity products in the authority's _PAIRS order
// (uu, vv, ww, uv, uw, vw) -- the fine-level fields the Germano lift
// filters.
extern "C" __global__
void sase_velocity_products(const real* __restrict__ u,
                            const real* __restrict__ v,
                            const real* __restrict__ w,
                            real* __restrict__ uu, real* __restrict__ vv,
                            real* __restrict__ ww, real* __restrict__ uv,
                            real* __restrict__ uw, real* __restrict__ vw,
                            long long ncell)
{
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncell) return;
    real a = u[q], b = v[q], c = w[q];
    uu[q] = a * a;
    vv[q] = b * b;
    ww[q] = c * c;
    uv[q] = a * b;
    uw[q] = a * c;
    vw[q] = b * c;
}

// Germano lift finalize: L_ij = filt(vel_i*vel_j) - filt_i*filt_j
// (authority grouping: the filtered product minus the product of
// filtered velocities).
extern "C" __global__
void sase_lift_combine(const real* __restrict__ fuu,
                       const real* __restrict__ fvv,
                       const real* __restrict__ fww,
                       const real* __restrict__ fuv,
                       const real* __restrict__ fuw,
                       const real* __restrict__ fvw,
                       const real* __restrict__ fu,
                       const real* __restrict__ fv,
                       const real* __restrict__ fw,
                       real* __restrict__ l0, real* __restrict__ l1,
                       real* __restrict__ l2, real* __restrict__ l3,
                       real* __restrict__ l4, real* __restrict__ l5,
                       long long ncell)
{
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncell) return;
    real a = fu[q], b = fv[q], c = fw[q];
    l0[q] = fuu[q] - a * a;
    l1[q] = fvv[q] - b * b;
    l2[q] = fww[q] - c * c;
    l3[q] = fuv[q] - a * b;
    l4[q] = fuw[q] - a * c;
    l5[q] = fvw[q] - b * c;
}

// SASE-L1 modeled SGS stress (authority model_stress): dynamic eddy
// viscosity on delta_eddy blended with the equilibrium momentum
// background on delta_mom, acting on the deviatoric strain, plus the
// isotropic (2/3)e term.  Scalar constants (C_MOM_BG -- the fixed
// momentum-background coefficient, formerly spelled C_K/PR_T; E_MIN)
// arrive as arguments single-sourced from gpuwm.verify.sase_ref by
// the launcher (S3-6g decision table: this is a momentum-channel
// constant, NOT a Prandtl consumer -- the regime blend never touches
// it; the ck_over_prt parameter name is kept for signature stability).
//
// Realizability closure: tau_zz is computed FROM the trace identity,
// tzz = 2e - txx - tyy, which is algebraically identical to the
// authority's per-component expression (the identity tau_kk = 2e holds
// exactly in R) but bounds the FP32 trace residual by ~1.5 ULP of the
// largest participating term BY CONSTRUCTION -- the ledger's
// realizability contract.  Direct per-component evaluation would leave
// the residual at the mercy of ~6 independent roundings at the
// deviatoric magnitude.  The parity cost against the authority's tau_zz
// is a few ULP absolute, far inside the 2e-6 scale-relative gate.
extern "C" __global__
void sase_model_stress(const real* __restrict__ e,
                       const real* __restrict__ s0,
                       const real* __restrict__ s1,
                       const real* __restrict__ s2,
                       const real* __restrict__ s3,
                       const real* __restrict__ s4,
                       const real* __restrict__ s5,
                       real* __restrict__ t0, real* __restrict__ t1,
                       real* __restrict__ t2, real* __restrict__ t3,
                       real* __restrict__ t4, real* __restrict__ t5,
                       real c_nu, real f_blend, real delta_eddy,
                       real delta_mom, real ck_over_prt, real e_min,
                       long long ncell)
{
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncell) return;
    real ef = fmaxf(e[q], e_min);
    real root_e = sqrtf(ef);
    real nu_eddy = c_nu * delta_eddy * root_e;
    real nu_mom = ck_over_prt * delta_mom * root_e;
    real nu = f_blend * nu_eddy + (1.0f - f_blend) * nu_mom;
    real a = s0[q], b = s1[q];
    real div3 = __fdiv_rn((a + b + s2[q]), 3.0f);
    real iso_visc = 2.0f * nu * div3;
    real iso_e = (2.0f / 3.0f) * ef;
    real txx = (-2.0f * nu * a + iso_visc) + iso_e;
    real tyy = (-2.0f * nu * b + iso_visc) + iso_e;
    real tzz = (2.0f * ef - txx) - tyy;        // trace closure (see above)
    t0[q] = txx;
    t1[q] = tyy;
    t2[q] = tzz;
    t3[q] = -2.0f * nu * s3[q];
    t4[q] = -2.0f * nu * s4[q];
    t5[q] = -2.0f * nu * s5[q];
}

// S3-6e RANS-governed horizontal stress (authority governed_stress):
// nu = f*c_nu*delta*sqrt(e) + (1-f)*K_smag with the audited 2-D
// Smagorinsky deformation diffusivity K_smag = min((c_s*delta)^2*|D_h|,
// cap*delta), |D_h| = sqrt((Sxx - Syy)^2 + 4*Sxy^2) on the SASE strain
// (A-grid transcription of WRF's def2 -- constants single-sourced from
// gpuwm.verify.sase_ref by the launcher, NOT re-derived).  Outputs the
// stress (trace closure exactly as sase_model_stress), the governed
// diffusivity field km (one field serves stress, e-transport, and the
// scalar K_h = km/Pr_t(f) -- S3-6g blended Prandtl number, applied by
// the launcher/driver), and the smag share r = nu_smag/max(nu, eps)
// in [0, 1] (nu >= nu_smag >= 0 termwise) that weights the production
// heat bypass.  At f = 1 the smag term is an FP-exact zero and tau
// reduces bitwise to sase_model_stress at delta/delta.
extern "C" __global__
void sase_model_stress_gov(const real* __restrict__ e,
                           const real* __restrict__ s0,
                           const real* __restrict__ s1,
                           const real* __restrict__ s2,
                           const real* __restrict__ s3,
                           const real* __restrict__ s4,
                           const real* __restrict__ s5,
                           real* __restrict__ t0, real* __restrict__ t1,
                           real* __restrict__ t2, real* __restrict__ t3,
                           real* __restrict__ t4, real* __restrict__ t5,
                           real* __restrict__ km, real* __restrict__ r_out,
                           real c_nu, real f_blend, real delta,
                           real c_s, real km_cap, real nu_eps, real e_min,
                           long long ncell)
{
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncell) return;
    real ef = fmaxf(e[q], e_min);
    real root_e = sqrtf(ef);
    real a = s0[q], b = s1[q], d12 = s3[q];
    real def_h = sqrtf((a - b) * (a - b) + 4.0f * (d12 * d12));
    real cd = c_s * delta;
    real k_smag = fminf(cd * cd * def_h, km_cap * delta);
    real nu_eddy = c_nu * delta * root_e;
    real nu_smag = (1.0f - f_blend) * k_smag;
    real nu = f_blend * nu_eddy + nu_smag;
    real div3 = __fdiv_rn((a + b + s2[q]), 3.0f);
    real iso_visc = 2.0f * nu * div3;
    real iso_e = (2.0f / 3.0f) * ef;
    real txx = (-2.0f * nu * a + iso_visc) + iso_e;
    real tyy = (-2.0f * nu * b + iso_visc) + iso_e;
    real tzz = (2.0f * ef - txx) - tyy;        // trace closure (see above)
    t0[q] = txx;
    t1[q] = tyy;
    t2[q] = tzz;
    t3[q] = -2.0f * nu * d12;
    t4[q] = -2.0f * nu * s4[q];
    t5[q] = -2.0f * nu * s5[q];
    km[q] = nu;
    r_out[q] = nu_smag / fmaxf(nu, nu_eps);
}

// Dynamic-solve basis premultiply (authority _identity_rows): from the
// RAW fine strain, form the width-independent refilter integrand
// p_k = -2*delta*sqrt(max(e, E_MIN)) * S_k^dev (deviatoric taken here,
// matching the authority's s_fine = _deviatoric(strain(...)) before the
// multiply).  Box-filtering p_k at each test width gives the shared
// refiltered fine-level term of both basis columns.
extern "C" __global__
void sase_basis_premultiply(const real* __restrict__ e,
                            const real* __restrict__ s0,
                            const real* __restrict__ s1,
                            const real* __restrict__ s2,
                            const real* __restrict__ s3,
                            const real* __restrict__ s4,
                            const real* __restrict__ s5,
                            real* __restrict__ p0, real* __restrict__ p1,
                            real* __restrict__ p2, real* __restrict__ p3,
                            real* __restrict__ p4, real* __restrict__ p5,
                            real delta, real e_min, long long ncell)
{
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncell) return;
    real root_e = sqrtf(fmaxf(e[q], e_min));
    real c = -2.0f * delta * root_e;
    real div3 = __fdiv_rn((s0[q] + s1[q] + s2[q]), 3.0f);
    p0[q] = c * (s0[q] - div3);
    p1[q] = c * (s1[q] - div3);
    p2[q] = c * (s2[q] - div3);
    p3[q] = c * s3[q];
    p4[q] = c * s4[q];
    p5[q] = c * s5[q];
}

// Dynamic-solve block partial sums for one test width (authority
// _identity_rows + the Gram/projection contractions of dynamic_solve):
// per cell and component k,
//   a_k = -2*delta_a*root_e*Sc_k^dev - rf_k     (eddy: delta_a = width*delta)
//   b_k = -2*delta_b*root_e*Sc_k^dev - rf_k     (momentum: delta_b = delta)
//   r_k = L_k - (k<3) * trace(L)/3              (deviatoric lift)
// with Sc the RAW coarse strain (deviatoric taken here) and rf the
// box-filtered premultiplied fine strain.  The five scalars a.a, a.b,
// b.b, a.r, b.r accumulate in FP64 (products of FP32 values promoted
// before multiply, structure-partial idiom); partials has shape
// (5, nblocks) and the host finishes the sum over blocks and widths,
// then runs the authority's 2x2 tail.
//
// bw > 0 (Task-6 fix round, registered specified-boundary adjudication)
// EXCLUDES the outer bw rows on all four lateral edges from the
// reductions: cells there contribute exactly zero to every moment,
// implemented as an in-kernel mask (chosen over an interior-slice
// launch so the FP64 block-reduction layout, block count, and
// deterministic summation order stay IDENTICAL to the golden-pinned
// bw=0 path -- with bw == 0 the only added work is one comparison and
// the accumulation arithmetic is untouched, so the pinned device
// goldens remain bitwise).
extern "C" __global__
void sase_solve_partial(const real* __restrict__ e,
                        const real* __restrict__ sc0,
                        const real* __restrict__ sc1,
                        const real* __restrict__ sc2,
                        const real* __restrict__ sc3,
                        const real* __restrict__ sc4,
                        const real* __restrict__ sc5,
                        const real* __restrict__ rf0,
                        const real* __restrict__ rf1,
                        const real* __restrict__ rf2,
                        const real* __restrict__ rf3,
                        const real* __restrict__ rf4,
                        const real* __restrict__ rf5,
                        const real* __restrict__ l0,
                        const real* __restrict__ l1,
                        const real* __restrict__ l2,
                        const real* __restrict__ l3,
                        const real* __restrict__ l4,
                        const real* __restrict__ l5,
                        double* __restrict__ partials,
                        real delta_a, real delta_b, real e_min,
                        int ny, int nx, int bw,
                        int nblocks, long long ncell)
{
    __shared__ double sdata[5][SASE_TPB];
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double m[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool excluded = false;
    if (bw > 0 && q < ncell) {
        int i = (int)(q % nx);
        int j = (int)((q / nx) % ny);
        excluded = (i < bw || i >= nx - bw || j < bw || j >= ny - bw);
    }
    if (q < ncell && !excluded) {
        real root_e = sqrtf(fmaxf(e[q], e_min));
        real ca = -2.0f * delta_a * root_e;
        real cb = -2.0f * delta_b * root_e;
        real sc[6] = {sc0[q], sc1[q], sc2[q], sc3[q], sc4[q], sc5[q]};
        real rf[6] = {rf0[q], rf1[q], rf2[q], rf3[q], rf4[q], rf5[q]};
        real lv[6] = {l0[q], l1[q], l2[q], l3[q], l4[q], l5[q]};
        real div3 = __fdiv_rn((sc[0] + sc[1] + sc[2]), 3.0f);
        real trl = __fdiv_rn((lv[0] + lv[1] + lv[2]), 3.0f);
        for (int k = 0; k < 6; ++k) {
            real scd = (k < 3) ? sc[k] - div3 : sc[k];
            real av = ca * scd - rf[k];
            real bv = cb * scd - rf[k];
            real rv = (k < 3) ? lv[k] - trl : lv[k];
            m[0] += (double)av * (double)av;
            m[1] += (double)av * (double)bv;
            m[2] += (double)bv * (double)bv;
            m[3] += (double)av * (double)rv;
            m[4] += (double)bv * (double)rv;
        }
    }
    for (int t = 0; t < 5; ++t) sdata[t][threadIdx.x] = m[t];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int t = 0; t < 5; ++t)
                sdata[t][threadIdx.x] += sdata[t][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int t = 0; t < 5; ++t)
            partials[(size_t)t * nblocks + blockIdx.x] = sdata[t][0];
}

// ---------------------------------------------------------------------
// S3-6c: split-step kernels -- explicit horizontal channel + implicit
// vertical channel (authority sase_split_step).  The v0 explicit
// fused-step kernels (sase_e_flux, sase_tendencies, sase_e_update)
// retired here together with the z_mode-2 periodic vertical: the split
// ledger theorem needs no periodic vertical, and the device must never
// mirror the superseded formulation against the new authority (S3-6b
// report section 7).
// ---------------------------------------------------------------------

// Layer thickness at level k under the split step's thickness modes
// (the per-column geometry the Thomas sweeps and the vertical channel
// rebuild in FP64):
//   t_mode 0 -- uniform column: scalar dz; t is a NEVER-DEREFERENCED
//               dummy pointer (launcher passes a 1-element placeholder);
//   t_mode 1 -- shared (nz,) float32 thickness column;
//   t_mode 3 -- per-column (nz, ny, nx) float32 thickness field (the
//               same object the z_mode-3 coefficient build consumes).
// FP32 thicknesses promote to FP64 exactly; face spacings h_k =
// 0.5*(t_k + t_{k+1}) equal the authority's z-center differences by
// the telescoping of the cumsum-half-layer construction -- exact in
// real arithmetic; the authority's FP64 cumsum-then-difference path
// can round differently in the last ULP (S3-6c review Minor), which
// the parity gates absorb (sase_ddz_coefficients has the same
// argument).
__device__ __forceinline__ double
sase_thick(const real* __restrict__ t, real dz, int t_mode, int k,
           long long ncol, long long col)
{
    if (t_mode == 0) return (double)dz;
    if (t_mode == 1) return (double)t[k];
    return (double)t[(size_t)k * ncol + col];
}

// S3-6h BL89 segment crossing (authority _bl89_first_crossing): the
// smallest s in [0, h] with c2*s^2 + c1*s == rem, or -1.0 when the
// parcel traverses the whole segment.  Derivation at the authority:
// within a piecewise-linear theta_v segment the spent energy is
// EXACTLY I(s) = c1*s + c2*s^2, so the stop point is a quadratic
// root; the numerically stable q-form mirrors the host (q = -(c1 +
// sign(c1)*sqrt(disc))/2, roots q/c2 and -rem/q), with the exact
// c2 == 0 linear branch.  All FP64.
__device__ __forceinline__
double sase_bl89_crossing(double c1, double c2, double rem, double h)
{
    double tol = 1.0e-12 * h;
    double best = -1.0;
    if (c2 == 0.0) {
        if (c1 > 0.0) {
            double s = rem / c1;
            if (s >= -tol && s <= h + tol)
                best = fmin(fmax(s, 0.0), h);
        }
        return best;
    }
    double disc = c1 * c1 + 4.0 * c2 * rem;
    if (disc < 0.0) return -1.0;
    double sq = sqrt(disc);
    double qq = -0.5 * (c1 + ((c1 >= 0.0) ? sq : -sq));
    double roots[2];
    roots[0] = qq / c2;
    roots[1] = (qq != 0.0) ? (-rem / qq) : 0.0;
    for (int i = 0; i < 2; ++i) {
        double r = roots[i];
        if (isfinite(r) && r >= -tol && r <= h + tol) {
            double s = fmin(fmax(r, 0.0), h);
            if (best < 0.0 || s < best) best = s;
        }
    }
    return best;
}

// Vertical mixing channel at e^n (authority _blackadar_length /
// vertical_mixing_length / bl89_displacement_lengths / bl89_combine /
// bl89_rans_lengths -- S3-6h), one thread per column: FP64 z centers
// by cumulative thickness sum (uniform mode gives z_k = (k+1/2)*dz
// exactly), then per level
//   l_B     = kappa*(z+z0)/(1 + kappa*(z+z0)/lambda)
//   l_les   = min(l_B, LS_COEF*sqrt(e)/N)      [stable cells; the
//             frozen pre-S3-6h length = the LES limb of the blend]
//   l_up/l_down : the BL89 parcel-energetics displacement pair by the
//             in-thread column sweep (accumulate exact segment
//             quadratures against the LIVE theta profile; fractional
//             segment by sase_bl89_crossing; l_up bounded by the top
//             interface, l_down by the surface; theta held constant
//             below the first center / above the last -- authority
//             conventions, derivations there)
//   l_mix_bl = (0.5*(l_up^-p + l_down^-p))^(-1/p), p = BL89_MIX_EXP
//   l_mix_r  = min(l_les, l_mix_bl),  l_eps_r = min(l_B, min(l_up,
//             l_down))
//   SASE-M1b (has_moist; S4-3c mirror of the authority
//             bl89_rans_lengths n2_dry seam -- sase_ref module
//             docstring, SASE-M1b section, MOIST_MASTER_LENGTH =
//             "bl89-n2eff-excursion-min-v1"): where n2 (= n2_eff)
//             departs bitwise from n2d (dry) -- exactly the
//             M1-substituted cells, the e-update kernel's point-2
//             mask idiom -- BOTH composed lengths are ADDITIONALLY
//             min-bounded by the moist parcel-excursion length
//               l_m = min(l_up_m, l_down_m)
//             of the BL89-family up/down excursion integrals against
//             the n2_eff FIELD: R(z') = int N^2_eff ds from the
//             parcel level, N^2_eff held piecewise-constant per
//             segment at the arithmetic face mean
//             0.5*(n2_eff[j] + n2_eff[j+1]) (R piecewise linear, the
//             outer integral exactly quadratic -- the same
//             sase_bl89_crossing quadrature, c1 = R at the segment
//             start accumulated as run += slope*h, c2 = slope/2),
//             constant-extension (slope 0) end segments, the same
//             E_MIN-floored e budget and surface/top geometric
//             bounds as the dry pair.  In-body comment has the
//             authority transcription notes.  Applied BEFORE rho/C_r
//             below (the authority evaluates C_r AT the bounded
//             length) and to the stored leps.  Unsubstituted cells
//             keep their bits verbatim (the branch never runs --
//             nothing is added, not even +0.0); has_moist == 0
//             leaves n2d a never-dereferenced gated dummy and the
//             kernel bitwise pre-M1b.
//   rho      = min(l_mix_r/l_s, 1)             [0 where N^2 <= 0 or
//             n2 absent -- S3-6i stable_limit_coefficient]
//   C_r      = C_KV + (C_KS/LS_COEF - C_KV)*rho^CKS_BLEND_EXP
//   K_v      = f*(C_KV*l_les*sqrt(e)) + (1-f)*(C_r*l_mix_r*sqrt(e))
//             [S3-6i two-product K blend: f = 1 FP-exact LES limb,
//              f = 0 the decoupled RANS channel -- where l_s binds,
//              rho == 1 and K_v -> C_KS*e/N, the registered stable
//              limit; neutral cells keep C_r = C_KV exactly]
// and the stored leps field is l_eps_r -- the RANS limb the e-update
// kernel's l_d blend consumes (S3-6h; formerly the bare l_B).  All
// arithmetic FP64 through the final FP32 stores; the theta/e columns
// (and the n2_eff column under has_moist) are staged into per-thread
// local arrays (the Thomas-sweep idiom, SASE_KMAX bound) because the
// O(nz^2) sweeps re-read them.  n2 is a gated dummy pointer when
// has_n2 == 0 (never dereferenced -- the sase_solve/e-update restrict
// landmine idiom), and n2d likewise when has_moist == 0.
extern "C" __global__
void sase_vertical_channel(const real* __restrict__ e,
                           const real* __restrict__ n2, int has_n2,
                           const real* __restrict__ n2d, int has_moist,
                           const real* __restrict__ theta,
                           const real* __restrict__ t, real dz, int t_mode,
                           real* __restrict__ kv, real* __restrict__ leps,
                           real z0, real karman, real blk_lambda,
                           real c_kv, real ls_coef, real e_min,
                           real f_blend, real g_accel, real mix_exp,
                           real c_ks, real cks_exp,
                           int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double zc_[SASE_KMAX], th_[SASE_KMAX], e_[SASE_KMAX],
           n2e_[SASE_KMAX];
    double zc = 0.0;
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double tk = sase_thick(t, dz, t_mode, k, ncol, col);
        zc_[k] = zc + 0.5 * tk;
        zc += tk;
        th_[k] = (double)theta[q];
        e_[k] = (double)fmaxf(e[q], e_min);
        // SASE-M1b: the moist excursion sweep re-reads the n2_eff
        // column O(nz) times per substituted level (header comment).
        if (has_moist) n2e_[k] = (double)n2[q];
    }
    double htop = zc;
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double z = zc_[k];
        double kz = (double)karman * (z + (double)z0);
        double lb64 = kz / (1.0 + kz / (double)blk_lambda);
        double e64 = e_[k];
        double root_e = sqrt(e64);
        double l_les = lb64;
        double ls_v = 0.0;         // S3-6i: l_s where stable, else 0
        if (has_n2) {
            double n2v = (double)n2[q];
            if (n2v > 0.0) {
                double ls = (double)ls_coef * root_e / sqrt(n2v);
                ls_v = ls;
                l_les = fmin(l_les, ls);
            }
        }
        double beta = (double)g_accel / th_[k];
        double bl[2];                          // l_up, l_down
        for (int down = 0; down < 2; ++down) {
            double rem = e64;
            double acc = 0.0;
            double got = -1.0;
            double prev_z = zc_[k], prev_t = th_[k];
            int nsteps = down ? (k + 1) : (nz - k);
            for (int m = 0; m < nsteps && got < 0.0; ++m) {
                int j = down ? (k - 1 - m) : (k + 1 + m);
                double node_z, node_t, h, c1, c2d;
                if (down) {
                    node_z = (j >= 0) ? zc_[j] : 0.0;
                    node_t = (j >= 0) ? th_[j] : th_[0];
                    h = prev_z - node_z;
                    c1 = beta * (th_[k] - prev_t);
                    c2d = beta * (prev_t - node_t);
                } else {
                    node_z = (j < nz) ? zc_[j] : htop;
                    node_t = (j < nz) ? th_[j] : th_[nz - 1];
                    h = node_z - prev_z;
                    c1 = beta * (prev_t - th_[k]);
                    c2d = beta * (node_t - prev_t);
                }
                if (h > 0.0) {
                    double c2 = c2d / (2.0 * h);
                    double s = sase_bl89_crossing(c1, c2, rem, h);
                    if (s >= 0.0) {
                        got = acc + s;
                    } else {
                        double seg = (c1 + c2 * h) * h;
                        rem = fmax(rem - seg, 0.0);
                    }
                    acc += h;
                }
                prev_z = node_z;
                prev_t = node_t;
            }
            if (got < 0.0) got = down ? z : (htop - z);
            bl[down] = got;
        }
        double p = (double)mix_exp;
        double l_mix_bl = pow(0.5 * (pow(bl[0], -p) + pow(bl[1], -p)),
                              -1.0 / p);
        double l_eps_bl = fmin(bl[0], bl[1]);
        double l_mix_r = fmin(l_les, l_mix_bl);
        double l_eps_r = fmin(lb64, l_eps_bl);
        // SASE-M1b moist master-length limb (S4-3c mirror of the
        // authority bl89_rans_lengths n2_dry seam composed with
        // bl89_moist_excursion_lengths; header comment has the
        // formulation).  Engaged ONLY in M1-substituted cells --
        // FP32 inequality against the dry field, exactly the
        // e-update kernel's point-2 mask idiom (the moist-n2 kernel
        // copies unsaturated dry bits).  TRANSCRIPTION NOTES
        // (authority op order, line for line): rem starts at the
        // staged E_MIN-floored e; per segment the stratification
        // slope is the FP64 arithmetic face mean
        // 0.5*(n2e_[j] + n2e_[j+1]) (constant-extension slope 0.0
        // beyond the outermost centers), c1 = run (R at the segment
        // start, accumulated afterwards as run += slope*h), c2 =
        // 0.5*slope gated on h > 0, the fractional segment solved by
        // the SAME sase_bl89_crossing quadrature the dry pair rides,
        // the exact segment quadrature seg = (c1 + c2*h)*h subtracted
        // through fmax(rem - seg, 0.0) only when no crossing, and the
        // surface/top geometric fallback bounds z / htop - z.
        // Moist-unstable segments (slope < 0) contribute negatively
        // -- rem GROWS -- which is the deck-transparency the limb
        // exists for (a moist-unstable deck spends nothing until the
        // moist-stable lid).  The min member l_m = min(l_up, l_down)
        // bounds BOTH compositions before the C_r evaluation below
        // (the authority evaluates C_r AT the bounded length).
        if (has_moist && n2[q] != n2d[q]) {
            double lm[2];                      // l_up_m, l_down_m
            for (int down = 0; down < 2; ++down) {
                double rem = e64;
                double acc = 0.0;
                double run = 0.0;
                double got = -1.0;
                double prev_z = zc_[k];
                int nsteps = down ? (k + 1) : (nz - k);
                for (int m = 0; m < nsteps && got < 0.0; ++m) {
                    int j = down ? (k - 1 - m) : (k + 1 + m);
                    double node_z, h, slope;
                    if (down) {
                        node_z = (j >= 0) ? zc_[j] : 0.0;
                        h = prev_z - node_z;
                        slope = (j >= 0)
                                ? 0.5 * (n2e_[j] + n2e_[j + 1]) : 0.0;
                    } else {
                        node_z = (j < nz) ? zc_[j] : htop;
                        h = node_z - prev_z;
                        slope = (j < nz)
                                ? 0.5 * (n2e_[j - 1] + n2e_[j]) : 0.0;
                    }
                    double c1 = run;
                    double c2 = (h > 0.0) ? 0.5 * slope : 0.0;
                    double s = sase_bl89_crossing(c1, c2, rem, h);
                    if (s >= 0.0) {
                        got = acc + s;
                    } else if (h > 0.0) {
                        double seg = (c1 + c2 * h) * h;
                        rem = fmax(rem - seg, 0.0);
                    }
                    acc += h;
                    if (h > 0.0) run += slope * h;
                    prev_z = node_z;
                }
                if (got < 0.0) got = down ? z : (htop - z);
                lm[down] = got;
            }
            double l_m = fmin(lm[0], lm[1]);
            l_mix_r = fmin(l_mix_r, l_m);
            l_eps_r = fmin(l_eps_r, l_m);
        }
        // S3-6i decoupled stable-limit coefficient (authority
        // stable_limit_coefficient): neutral/unstable cells keep
        // C_r = C_KV exactly (the branch never fires), l_s-binding
        // cells land C_KS/LS_COEF (rho == 1 -- l_s is a term of the
        // l_mix_r min), the blend between is C^1 in N (comment above).
        double c_r = (double)c_kv;
        if (ls_v > 0.0) {
            double rho = fmin(l_mix_r / ls_v, 1.0);
            c_r += ((double)c_ks / (double)ls_coef - (double)c_kv)
                   * pow(rho, (double)cks_exp);
        }
        double fb = (double)f_blend;
        kv[q] = (real)(fb * ((double)c_kv * l_les * root_e)
                       + (1.0 - fb) * (c_r * l_mix_r * root_e));
        leps[q] = (real)l_eps_r;
    }
}

// Split-step explicit horizontal channel (authority sase_split_step
// step 4): momentum tendencies keep ONLY the horizontal tau flux
// divergences, du_i = -(ddx tau_ix + ddy tau_iy) -- the d/dz(tau_i3)
// fluxes are REMODELED as the implicit -K_v*d(phi)/dz channel, so
// tau_zz never enters the split step -- plus the horizontal production
// pairing P_h,tot = -(tau_xx*ddx u + tau_yy*ddy v + tau_xy*(ddy u +
// ddx v) + tau_xz*ddx w + tau_yz*ddy w) against the SAME u^n gradients
// (identity (i) of the split ledger theorem: horizontal summation by
// parts on the periodic directions).  S3-6e split: the smag share
//   ph_heat = r*(P_h,tot + (2/3)*max(e, e_min)*(ddx u + ddy v))
//           [= 2*nu_smag*G, the deformation component's viscous
//            pairing -- bypasses e, deposits to heat]
//   ph_e    = P_h,tot - ph_heat        [dynamic + isotropic, feeds e]
// with r the stress kernel's smag-share field; at r = 0 ph_heat is an
// FP-exact zero and ph_e is bitwise P_h,tot.  The u^n horizontal
// derivatives are recomputed here with the identical centered FP32
// expressions the strain kernel used, so the pairing is
// arithmetic-consistent.
extern "C" __global__
void sase_split_tendencies(const real* __restrict__ t0,
                           const real* __restrict__ t1,
                           const real* __restrict__ t3,
                           const real* __restrict__ t4,
                           const real* __restrict__ t5,
                           const real* __restrict__ u,
                           const real* __restrict__ v,
                           const real* __restrict__ w,
                           const real* __restrict__ e,
                           const real* __restrict__ r,
                           real* __restrict__ du, real* __restrict__ dv,
                           real* __restrict__ dw, real* __restrict__ ph_e,
                           real* __restrict__ ph_heat,
                           real two_dx, real two_dy, real e_min,
                           int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    size_t q = IDX3(k, j, i);
#define SASE_DDX(t) ((t[IDX3(k, j, ip)] - t[IDX3(k, j, im)]) / two_dx)
#define SASE_DDY(t) ((t[IDX3(k, jp, i)] - t[IDX3(k, jm, i)]) / two_dy)
    du[q] = -(SASE_DDX(t0) + SASE_DDY(t3));
    dv[q] = -(SASE_DDX(t3) + SASE_DDY(t1));
    dw[q] = -(SASE_DDX(t4) + SASE_DDY(t5));
    real dudx = SASE_DDX(u), dvdy = SASE_DDY(v);
    real dudy = SASE_DDY(u), dvdx = SASE_DDX(v);
    real dwdx = SASE_DDX(w), dwdy = SASE_DDY(w);
#undef SASE_DDX
#undef SASE_DDY
    real ph_tot = -(t0[q] * dudx + t1[q] * dvdy + t3[q] * (dudy + dvdx)
                    + t4[q] * dwdx + t5[q] * dwdy);
    real phh = r[q] * (ph_tot
                       + (2.0f / 3.0f) * fmaxf(e[q], e_min)
                         * (dudx + dvdy));
    ph_heat[q] = phh;
    ph_e[q] = ph_tot - phh;
}

// Horizontal subgrid-energy diffusive fluxes f_i = 2*K_m * d(e64)/dx_i
// (authority sase_split_step step 6 transport integrand -- the split
// step's e-transport is horizontal-explicit ONLY; the vertical
// e-transport is the implicit 2*K_v Thomas solve).  S3-6e: K_m is the
// GOVERNED diffusivity field from the stress kernel (one horizontal
// diffusivity serves stress, e-transport, and scalars; the v0 bare-C_K
// km_coef*sqrt(e) blend is retired on this path).  Neighbor reads
// floor e exactly like the authority's precomputed e64 array.
extern "C" __global__
void sase_e_hflux(const real* __restrict__ e,
                  const real* __restrict__ km,
                  real* __restrict__ fx, real* __restrict__ fy,
                  real two_dx, real two_dy,
                  real e_min, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    size_t q = IDX3(k, j, i);
    real k2 = 2.0f * km[q];
    real dedx = (fmaxf(e[IDX3(k, j, ip)], e_min)
                 - fmaxf(e[IDX3(k, j, im)], e_min)) / two_dx;
    real dedy = (fmaxf(e[IDX3(k, jp, i)], e_min)
                 - fmaxf(e[IDX3(k, jm, i)], e_min)) / two_dy;
    fx[q] = k2 * dedx;
    fy[q] = k2 * dedy;
}

// One FP64 backward-Euler Thomas sweep for one column of one field
// (authority implicit_vertical_diffusion, line-mirrored): flux-form
// zero-flux operator with face conductances r[k] = dt*K_f/h
// precomputed by the caller; sub_k = -r[k-1]/thick_k, sup_k =
// -r[k]/thick_k, diag_k = 1 - sub_k - sup_k (unit row sums split, the
// M-matrix that gives unconditional stability + the max principle).
// rhs is phi (+ dt*dphi when dphi != nullptr -- the u* explicit deposit
// of the momentum channel).  S3-6j ``bottom`` (authority drag_bottom):
// the precomputed dt*c/thick_0 surface-stress term added to the BOTTOM
// diagonal (0.0 = the zero-flux end, an FP-exact no-op) -- the
// implicit linearization F_sfc = c*phi_new_0 of the YSU
// ``diag[0] = 1 + fric`` pattern; the added term is positive, so the
// M-matrix dominance (and unconditional stability) strengthens.
// S3-11b ``dep0`` (authority surface_scalar_flux_deposit, the
// registered EXPLICIT-deposit-BEFORE-solve seam SFC_SCALAR_FLUX =
// "explicit-deposit-v1"): the precomputed FP64 surface scalar-flux
// increment dt*flux/(rho1*fac*thick_0) added to the BOTTOM rhs before
// the sweep -- the fused form of the authority composition
// implicit_solve(deposit(phi)), op-order-identical to the FP64
// authority because the deposited bottom value never rounds through
// an intermediate FP32 store.  GUARDED: dep0 == +-0.0 adds NOTHING
// (not even +0.0), so the zero-flux column is BITWISE the pre-S3-11b
// sweep unconditionally -- including a -0.0 bottom value, which the
// authority's unguarded ``x + 0.0`` would flip to +0.0 (the S3-11a
// docstring caveat; unreachable for physical theta > 0 / qv >= +0.0,
// the one documented FP divergence, bounded by the zero-flux
// identity gate in tests/test_sase_gpu.py).  Momentum callers pass
// 0.0 (their surface seam is the S3-6j drag diagonal).
// cpr/dpr are the caller's per-thread FP64
// sweep buffers (SASE_KMAX doubles each; they live in registers/local
// memory -- NO global workspace, which is what the preflight
// transcription counts on).  The back-substitution chain stays FP64
// (prev carries the unrounded value); only the store rounds to FP32.
// Ledger pairings (split theorem channels), accumulated when the
// pointers are given:
//   m_expl += phi^n * (rhs - phi^n)              [dKE_expl at u^n]
//   m_impl += phi32^{n+1} * (phi32^{n+1} - rhs)  [dKE_impl at u^{n+1}]
// m_impl reads the STORED FP32 value so the pairing is consistent with
// sase_vertical_production, which builds P_v from the stored fields.
// With a nonzero bottom term, m_impl inherently CONTAINS the drag work
// (authority S3-6j identity); the caller measures dKE_sfc separately.
__device__ void
sase_thomas_column(real* __restrict__ phi, const real* __restrict__ dphi,
                   const double* r, const real* __restrict__ t, real dz,
                   int t_mode, double dt, double bottom, double dep0,
                   real floor_val, int has_floor,
                   int nz, long long ncol, long long col,
                   double* cpr, double* dpr,
                   double* m_expl, double* m_impl)
{
    size_t q0 = (size_t)col;
    double tb = sase_thick(t, dz, t_mode, 0, ncol, col);
    double sup = -(r[0] / tb);
    double diag = 1.0 - sup + bottom;          // sub_0 = 0; S3-6j drag
    double phin = (double)phi[q0];
    double rhs = (dphi != nullptr) ? phin + dt * (double)dphi[q0] : phin;
    // S3-11b surface scalar-flux deposit (header comment): guarded so
    // a zero increment leaves the sweep bitwise-untouched.
    if (dep0 != 0.0) rhs += dep0;
    if (m_expl) *m_expl += phin * (rhs - phin);
    cpr[0] = sup / diag;
    dpr[0] = rhs / diag;
    for (int k = 1; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        tb = sase_thick(t, dz, t_mode, k, ncol, col);
        double sub = -(r[k - 1] / tb);
        sup = (k < nz - 1) ? -(r[k] / tb) : 0.0;
        diag = 1.0 - sub - sup;
        phin = (double)phi[q];
        rhs = (dphi != nullptr) ? phin + dt * (double)dphi[q] : phin;
        if (m_expl) *m_expl += phin * (rhs - phin);
        double m = diag - sub * cpr[k - 1];
        cpr[k] = sup / m;
        dpr[k] = (rhs - sub * dpr[k - 1]) / m;
    }
    double prev = 0.0;
    for (int k = nz - 1; k >= 0; --k) {
        size_t q = (size_t)k * ncol + col;
        double out = (k == nz - 1) ? dpr[k] : dpr[k] - cpr[k] * prev;
        prev = out;
        real o32 = (real)out;
        if (has_floor) o32 = fmaxf(o32, floor_val);
        if (m_impl) {
            double phin_k = (double)phi[q];    // still the pre-solve value
            double rhs_k = (dphi != nullptr)
                ? phin_k + dt * (double)dphi[q] : phin_k;
            // Keep the pairing consistent with the forward pass (S3-11b;
            // every current m_impl caller passes dep0 = 0.0).
            if (k == 0 && dep0 != 0.0) rhs_k += dep0;
            double o64 = (double)o32;
            *m_impl += o64 * (o64 - rhs_k);
        }
        phi[q] = o32;
    }
}

// Implicit vertical momentum channel (authority sase_split_step step
// 5): per column, build the face conductances r[k] = dt*K_f/h once
// (K_f = arithmetic face mean of the K_v field, h = FP64 center
// spacing), then run the FP64 Thomas sweep for u, v, w with rhs =
// phi + dt*dphi (the horizontal-explicit u* deposit), writing the
// solved fields in place and accumulating the dKE ledger channels
// (structure-partial FP64 block-reduction idiom).  S3-6j surface
// stress (authority module docstring, S3-6j section): with
// has_drag != 0, ``ust`` is the (ny, nx) friction-velocity field and
// the u/v sweeps carry the implicit drag term
//   c      = ust^2 / max(hypot(u1^n, v1^n), sfc_floor)   [PRE-solve
//            level-1 winds -- the YSU wspd1 linearization; sfc_floor
//            is the FP64 SFC_WSPD_FLOOR, sfclay's audited 0.1]
//   bottom = dt*c/thick_0                                [diag_0 +=]
// S3-9c gustiness correction (authority sase_split_step / module
// docstring, S3-9c section): with has_wspd != 0, ``wspd`` is
// sfclay's gust-ENHANCED (ny, nx) speed field and c gains the
// audited YSU factor (npref.py:6495-6496 ``* (wspd1/max(wspd,
// 1.0e-9))**2``), line-mirroring the authority:
//   c *= (spd1 / max(wspd, 1e-9))^2       [spd1 = the floored
//                                          resolved speed above]
// -- no-gust wspd == spd1 multiplies by exactly 1.0; has_wspd == 0
// forms no factor (the S3-6j arithmetic bitwise; the wspd pointer is
// then a never-dereferenced gated dummy, the restrict-alias idiom).
// The launcher rejects wspd without ust.
// (w keeps the zero-flux end: the stress is horizontal).  After the
// u/v solves the third ledger channel accumulates the MEASURED drag
// work from the STORED FP32 fields (the m_impl convention):
//   m_sfc += -dt*c*((u1^{n+1})^2 + (v1^{n+1})^2)/thick_0   [<= 0]
// With has_drag == 0 the ust pointer is a never-dereferenced dummy
// (the restrict-alias gated idiom) and every drag term is exactly
// absent.  nz == 1 columns
// have no faces: the solve is the identity on u* (authority nz-1
// branch, drag deliberately NOT applied -- faceless), with the
// channels still measured.  One thread per column;
// launcher enforces nz <= SASE_KMAX.
extern "C" __global__
void sase_thomas_momentum(real* __restrict__ u, real* __restrict__ v,
                          real* __restrict__ w,
                          const real* __restrict__ du,
                          const real* __restrict__ dv,
                          const real* __restrict__ dw,
                          const real* __restrict__ kv,
                          const real* __restrict__ ust, int has_drag,
                          const real* __restrict__ wspd, int has_wspd,
                          double sfc_floor,
                          const real* __restrict__ t, real dz, int t_mode,
                          double* __restrict__ partials,
                          real dt, int nblocks, int nz, int ny, int nx)
{
    __shared__ double sdata[3][SASE_TPB];
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    double m_expl = 0.0, m_impl = 0.0, m_sfc = 0.0;
    if (col < ncol && nz > 1) {
        double r[SASE_KMAX], cpr[SASE_KMAX], dpr[SASE_KMAX];
        double dt64 = (double)dt;
        double t0 = sase_thick(t, dz, t_mode, 0, ncol, col);
        double tk = t0;
        for (int k = 0; k < nz - 1; ++k) {
            double tk1 = sase_thick(t, dz, t_mode, k + 1, ncol, col);
            double h = 0.5 * (tk + tk1);
            size_t q = (size_t)k * ncol + col;
            double kf = 0.5 * ((double)kv[q] + (double)kv[q + ncol]);
            r[k] = dt64 * kf / h;
            tk = tk1;
        }
        double cdrag = 0.0, bottom = 0.0;
        if (has_drag) {
            // PRE-solve level-1 winds (authority: hypot at u^n).
            double u1 = (double)u[col], v1 = (double)v[col];
            double spd1 = fmax(hypot(u1, v1), sfc_floor);
            double us = (double)ust[col];
            cdrag = us * us / spd1;
            if (has_wspd) {
                // S3-9c: the audited YSU gustiness factor
                // (header comment; authority ratio*ratio grouping).
                double ratio = spd1 / fmax((double)wspd[col], 1.0e-9);
                cdrag = cdrag * (ratio * ratio);
            }
            bottom = dt64 * cdrag / t0;
        }
        sase_thomas_column(u, du, r, t, dz, t_mode, dt64, bottom, 0.0,
                           0.0f, 0, nz, ncol, col, cpr, dpr,
                           &m_expl, &m_impl);
        sase_thomas_column(v, dv, r, t, dz, t_mode, dt64, bottom, 0.0,
                           0.0f, 0, nz, ncol, col, cpr, dpr,
                           &m_expl, &m_impl);
        sase_thomas_column(w, dw, r, t, dz, t_mode, dt64, 0.0, 0.0,
                           0.0f, 0, nz, ncol, col, cpr, dpr,
                           &m_expl, &m_impl);
        if (has_drag) {
            // Measured drag work at the SOLVED (stored FP32) winds.
            double u1n = (double)u[col], v1n = (double)v[col];
            m_sfc += -dt64 * cdrag * (u1n * u1n + v1n * v1n) / t0;
        }
    } else if (col < ncol) {
        // nz == 1: no faces -- identity solve on u* (authority branch).
        real* fields[3] = {u, v, w};
        const real* tends[3] = {du, dv, dw};
        for (int c = 0; c < 3; ++c) {
            double phin = (double)fields[c][col];
            double rhs = phin + (double)dt * (double)tends[c][col];
            m_expl += phin * (rhs - phin);
            real o32 = (real)rhs;
            double o64 = (double)o32;
            m_impl += o64 * (o64 - rhs);
            fields[c][col] = o32;
        }
    }
    sdata[0][threadIdx.x] = m_expl;
    sdata[1][threadIdx.x] = m_impl;
    sdata[2][threadIdx.x] = m_sfc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int c = 0; c < 3; ++c)
                sdata[c][threadIdx.x] += sdata[c][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int c = 0; c < 3; ++c)
            partials[(size_t)c * nblocks + blockIdx.x] = sdata[c][0];
}

// Implicit e-transport / scalar vertical channel (authority
// implicit_vertical_diffusion at the sase_split_step step-7 and driver
// scalar call sites): the same FP64 Thomas sweep with face diffusivity
// kfac*K_f (kfac = 2 for the e-transport convention, 1/Pr_t(f) for
// the scalars -- the S3-6g blended Prandtl number, supplied by the
// driver) and an optional post-solve floor (the e channel's E_MIN
// re-floor, which folds solver roundoff dips into the measured
// transport channel).  kfac*K_f is formed in FP64 before the r build,
// matching the authority's 2.0*k_face grouping exactly (the factor is
// exact in FP64).  No ledger channels here.
//
// S3-11b surface scalar-flux deposit (authority
// surface_scalar_flux_deposit; registered SFC_SCALAR_FLUX =
// "explicit-deposit-v1"; root cause
// docs/internal/sdd/lake-momentum-root-cause.md): with has_sfc != 0,
// ``sfc_flux`` is the (ny, nx) DIMENSIONAL post-sfclay surface flux
// (HFX [W m^-2] for the theta row, QFX [kg m^-2 s^-1] for the qv
// row -- positive UPWARD), ``sfc_rho1`` the (ny, nx) lowest-level
// MOIST density -- physics.sase_surface_rho1, the SAME field the
// surface e source consumes (the S3-11a rho-consistency obligation:
// a deposit at a different density would silently rescale the flux
// against the e source's own buoyancy bookkeeping) -- and ``sfc_fac``
// the row constant (CP_AIR for theta, 1.0 for qv; *1.0 is FP-exact,
// so one kernel serves both authority rows bitwise).  The bottom rhs
// gains the authority's exact FP64 expression
//   dep = dt*flux/((rho1*fac)*thick_0)
// BEFORE the sweep (sase_thomas_column dep0 -- the fused explicit
// pre-deposit; op order identical to the authority composition
// implicit_solve(deposit(phi)), no intermediate FP32 round).
// GUARDED at flux == +-0.0: the deposit is then exactly absent and
// the sweep is BITWISE the has_sfc == 0 path -- the seam is OFF-able
// only through the fluxes themselves (S3-11b driver contract).  The
// former comment here ("their surface fluxes arrive via sfclay/Noah")
// recorded a FALSE premise -- sfclay only diagnoses HFX/QFX and Noah
// consumes them in the GROUND budget; no path deposited them into
// theta/qv until this seam (the G-LAKE root cause).  With
// has_sfc == 0 the flux/rho pointers are never-dereferenced gated
// dummies (the module's restrict-alias idiom) and the arithmetic is
// bitwise the pre-S3-11b kernel.  qc/qi rows pass has_sfc == 0 (YSU's
// cloud/ice rows carry no surface source); e-transport likewise.
extern "C" __global__
void sase_thomas_scalar(real* __restrict__ phi,
                        const real* __restrict__ kv,
                        const real* __restrict__ sfc_flux, int has_sfc,
                        const real* __restrict__ sfc_rho1,
                        double sfc_fac,
                        const real* __restrict__ t, real dz, int t_mode,
                        real dt, real kfac, real floor_val, int has_floor,
                        int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol || nz < 2) return;
    double r[SASE_KMAX], cpr[SASE_KMAX], dpr[SASE_KMAX];
    double dt64 = (double)dt;
    double kf64 = (double)kfac;
    double tk = sase_thick(t, dz, t_mode, 0, ncol, col);
    double dep = 0.0;
    if (has_sfc) {
        // Authority op order: dt*flux / ((rho1*fac)*thick_0) -- the
        // numpy grouping of surface_scalar_flux_deposit, with tk still
        // holding thick_0 here.  fl == +-0.0 leaves dep = +-0.0 and
        // the column guard skips the add entirely (header comment).
        double fl = (double)sfc_flux[col];
        if (fl != 0.0)
            dep = dt64 * fl / ((double)sfc_rho1[col] * sfc_fac * tk);
    }
    for (int k = 0; k < nz - 1; ++k) {
        double tk1 = sase_thick(t, dz, t_mode, k + 1, ncol, col);
        double h = 0.5 * (tk + tk1);
        size_t q = (size_t)k * ncol + col;
        double kf = kf64 * (0.5 * ((double)kv[q] + (double)kv[q + ncol]));
        r[k] = dt64 * kf / h;
        tk = tk1;
    }
    // bottom = 0.0: the S3-6j drag DIAGONAL stays a momentum-only seam
    // (a drag row here would double-count); the scalar surface flux
    // enters through dep0 above, and e keeps the zero-flux end (its
    // surface source is the named physics.sase_surface_e_source).
    sase_thomas_column(phi, nullptr, r, t, dz, t_mode, dt64, 0.0, dep,
                       floor_val, has_floor, nz, ncol, col, cpr, dpr,
                       nullptr, nullptr);
}

// Implicit-flux vertical production P_v (authority
// _vertical_production): per interior face, eps_f = sum_i
// K_f*(delta phi_i)^2/h from the IMPLICIT-SOLVED fields -- the pairing
// that makes identity (ii) of the split ledger theorem hold -- split
// half to each neighbor cell over its thickness, upper face added
// first (authority accumulation order).  Pointwise non-negative by
// construction.  K_f and h are rebuilt from the same kv field and
// thicknesses the Thomas sweep used, so the pairing is consistent.
// Arithmetic here is FP32 (unlike the FP64 sweeps; S3-6c review
// Minor): P_v feeds the FP32 e update, and the FP32/FP64 mismatch of
// the P_v-vs-dKE_impl pairing is part of what the characterized FP32
// closure bound measures.
extern "C" __global__
void sase_vertical_production(const real* __restrict__ u,
                              const real* __restrict__ v,
                              const real* __restrict__ w,
                              const real* __restrict__ kv,
                              const real* __restrict__ t, real dz,
                              int t_mode,
                              real* __restrict__ pv,
                              int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    long long ncol = (long long)ny * nx;
    long long col = (long long)j * nx + i;
    size_t q = IDX3(k, j, i);
    if (nz < 2) { pv[q] = 0.0f; return; }
    double tk = sase_thick(t, dz, t_mode, k, ncol, col);
    real eps_lo = 0.0f, eps_hi = 0.0f;
    if (k > 0) {
        size_t qm = q - (size_t)ncol;
        real h = (real)(0.5 * (sase_thick(t, dz, t_mode, k - 1, ncol, col)
                               + tk));
        real kf = 0.5f * (kv[qm] + kv[q]);
        real duz = u[q] - u[qm];
        real dvz = v[q] - v[qm];
        real dwz = w[q] - w[qm];
        eps_lo = kf * duz * duz / h;
        eps_lo += kf * dvz * dvz / h;
        eps_lo += kf * dwz * dwz / h;
    }
    if (k < nz - 1) {
        size_t qp = q + (size_t)ncol;
        real h = (real)(0.5 * (tk + sase_thick(t, dz, t_mode, k + 1,
                                               ncol, col)));
        real kf = 0.5f * (kv[q] + kv[qp]);
        real duz = u[qp] - u[q];
        real dvz = v[qp] - v[q];
        real dwz = w[qp] - w[q];
        eps_hi = kf * duz * duz / h;
        eps_hi += kf * dvz * dvz / h;
        eps_hi += kf * dwz * dwz / h;
    }
    real th = (real)tk;
    pv[q] = 0.5f * eps_hi / th + 0.5f * eps_lo / th;
}

// S3-12 state-independent Blackadar reference length (authority
// _blackadar_length on the _column_geometry z centers): one thread per
// column writes lb[k] = kz/(1 + kz/lambda), kz = kappa*(z_k + z0), the
// RANS member of the additive channel's neutral_dissipation_length
// blend l_ref = delta^f * l_B(z+z0)^(1-f) -- the e-update kernel forms
// the blend itself (its l_d endpoint-branch idiom).  DELIBERATELY a
// separate kernel rather than a third sase_vertical_channel output:
// the field depends on GEOMETRY ONLY (t, z0 -- no e, no theta, no n2;
// that state-independence is the entire content of the S3-12 fix), it
// is needed only under the additive switch, and a gated launch keeps
// the default path's allocation set and every existing kernel
// signature untouched.  FP64 z in the AUTHORITY'S OWN OP ORDER, which
// differs from the vertical-channel kernel's half-step accumulation in
// the last ulp: t_mode 0 mirrors _column_geometry's uniform branch
// z_k = (k + 0.5)*dz EXACTLY (product, not accumulation); t_mode 1/3
// mirror np.cumsum(t) - 0.5*t (running FP64 sum, then the half-layer
// SUBTRACTED -- not zc + 0.5*tk).  The constants arrive as doubles
// (the moist-n2/vent kernel convention), so the stored FP32 value is
// the correctly-rounded image of the authority's FP64 l_B wherever the
// FP32 thicknesses are what the authority consumed -- the parity gate
// in tests/test_sase_gpu.py pins that bitwise (max ULP 0).
extern "C" __global__
void sase_blackadar_length(const real* __restrict__ t, real dz,
                           int t_mode, real* __restrict__ lb,
                           double z0, double karman, double blk_lambda,
                           int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double csum = 0.0;
    for (int k = 0; k < nz; ++k) {
        double z;
        if (t_mode == 0) {
            z = ((double)k + 0.5) * (double)dz;
        } else {
            double tk = sase_thick(t, dz, t_mode, k, ncol, col);
            csum += tk;
            z = csum - 0.5 * tk;
        }
        double kz = karman * (z + z0);
        lb[(size_t)k * ncol + col] = (real)(kz / (1.0 + kz / blk_lambda));
    }
}

// Split e update + FP64 ledger partials (authority sase_split_step
// steps 6-8, explicit part; S3-6d analytic dissipation substep; S3-6e
// production split + damping-layer taper).  Per cell, at e^n:
//   buoy   = -(g/theta) * (K_v/pr_t) * dtheta/dz   [the VERTICAL
//            channel's K_h -- kv field, not the K_m blend; pr_t is
//            the S3-6g blended Prandtl number Pr_t(f_used), computed
//            host-side by the launcher (PR_LES at f = 1, FP-exact).
//            SASE-M1 (has_moist): where n2 (= n2_eff) departs bitwise
//            from n2d (dry), buoy = -(K_v/pr_t)*n2_eff instead --
//            the point-2 substitution; in-body comment]
//   l_d    = min(delta**f * l_eps_rans**(1-f), LS_COEF*sqrt(e)/N)
//            [the regime blend -- GEOMETRIC since S3-9 (authority
//            module docstring, S3-9 section; was the linear
//            f*delta + (1-f)*l_eps_rans); S3-6h: the RANS limb is
//            the vertical-channel kernel's l_eps_r =
//            min(l_B, l_eps_BL89) field (formerly the bare l_B);
//            l_s branch on stable cells only, at e^n]
//   t_h    = ddx(fx) + ddy(fy)          [horizontal transport only]
//   g      = has_taper ? gtap : 1       [damp_opt=3 taper weight field
//            from the launcher; 1.0 is an FP-exact no-op]
//   x      = ph_e + pv;  src = g*x;  gb = g*buoy
//   e*     = e + dt*((src + gb) + t_h)  [no explicit dissipation
//            source any more -- S3-6d]
//   ANALYTIC decay substep (the exact solution of
//   de/dt = -C_E*e^{3/2}/l_d over dt from e*), FP64 per cell from the
//   FP32 e* and l_d with ONE FP32 rounding at the e_dec store (the
//   ratified S3-6d precision requirement -- b*dt reaches O(1) at d01
//   parameters, where FP32 forward arithmetic would visibly bias the
//   equilibrium):
//     b      = C_eps*sqrt(max(e*, E_MIN))/(2*l_d)
//              [C_eps == C_E unless has_ces (S3-6k, authority
//              stable_dissipation_coefficient), in which case
//              rho = min(l_d/l_s, 1), w = rho^CKS_BLEND_EXP,
//              C_rans = (1-w)*C_E + w*C_ES, C_eps = f*C_E +
//              (1-f)*C_rans -- bitwise C_E at has_ces == 0, at
//              n2 <= 0, and at f == 1; in-body comment.
//              S3-12 (has_ced; authority additive_dissipation_
//              coefficient + neutral_dissipation_length): the second,
//              grid-scale Deardorff channel then ADDS to whichever
//              base the line above selected,
//                l_ref  = delta^f * lb^(1-f)   [lb = the
//                         sase_blackadar_length field; the l_d
//                         blend's own endpoint branches]
//                C_eps += (1-f)*w*C_ED*(l_d/l_ref)
//              with the SAME rho/w the S3-6k line forms -- bitwise
//              the selected base at has_ced == 0, at n2 <= 0 / n2
//              absent (the shared ls_v > 0.0f SELECTION gate), and
//              at f == 1 (the added term is arithmetic +0.0 and
//              c_eps > 0, the authority's own LES-limb argument);
//              in-body comment]
//     e_dec  = e*/(1 + b*dt)^2
//     D      = e* - e_dec              [FP32 pairing on the stored
//              e_dec, consistent with the m_impl stored-value idiom]
//   e_clip = max(e_dec, E_MIN);  clip = e_clip - e_dec
//   heat   = (D - clip) + dt*(ph_heat + (x - src))   [S3-6e: decay +
//            smag bypass + taper redirect; no longer pointwise
//            sign-definite -- authority module docstring]
//   e = e_clip (the implicit 2*K_v transport solve follows), heat out.
// Ledger channels, FP64 per block (structure-partial reduction idiom):
//   m0 = e_clip - e_old   [with the implicit-transport increment
//        excluded by the theorem's dE definition, sum(e^{n+1} - e^n) -
//        sum(e^{n+1} - e_clip) telescopes to exactly this]
//   m1 = dt*gb,  m2 = dt*t_h            (the dE exclusions; the
//        measured buoyancy channel is the TAPERED gb -- S3-6e theorem)
//   m3 = heat                           (dHeat)
// The host finishes dE = m0 - m1 - m2, dHeat = m3; together with the
// Thomas kernel's dKE channels the closure residual on the
// horizontally periodic uniform-dz box is the characterized FP32
// bound (spec 4.2 artifact, gate 1e-5 rel in tests).
extern "C" __global__
void sase_split_e_update(real* __restrict__ e, real* __restrict__ heat,
                         const real* __restrict__ theta,
                         const real* __restrict__ n2, int has_n2,
                         const real* __restrict__ n2d, int has_moist,
                         const real* __restrict__ kv,
                         const real* __restrict__ leps,
                         const real* __restrict__ lb, int has_ced,
                         const real* __restrict__ ph_e,
                         const real* __restrict__ ph_heat,
                         const real* __restrict__ pv,
                         const real* __restrict__ fx,
                         const real* __restrict__ fy,
                         const real* __restrict__ gtap, int has_taper,
                         double* __restrict__ partials,
                         const real* __restrict__ cm,
                         const real* __restrict__ c0,
                         const real* __restrict__ cp,
                         real two_dx, real two_dy, real dz, real two_dz,
                         real h_lo, real h_hi, int z_mode,
                         real dt, real f_blend, real delta, real pr_t,
                         real c_e, real ls_coef,
                         real c_es, real cks_exp, int has_ces,
                         real c_ed,
                         real g_accel, real e_min,
                         int nblocks, int nz, int ny, int nx)
{
    __shared__ double sdata[4][SASE_TPB];
    long long qq = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncell = (long long)nz * ny * nx;
    double m[4] = {0.0, 0.0, 0.0, 0.0};
    if (qq < ncell) {
        int i = (int)(qq % nx);
        int j = (int)((qq / nx) % ny);
        int k = (int)(qq / ((long long)nx * ny));
        int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
        int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
        int km = sase_km(k, nz, z_mode), kp = sase_kp(k, nz, z_mode);
        size_t q = (size_t)qq;
        real ec = fmaxf(e[q], e_min);
        real root_e = sqrtf(ec);
        real dthdz = sase_ddz(theta[IDX3(km, j, i)], theta[q],
                              theta[IDX3(kp, j, i)], k, q, nz, cm, c0, cp,
                              dz, two_dz, h_lo, h_hi, z_mode);
        real buoy = -(g_accel / theta[q]) * (kv[q] / pr_t) * dthdz;
        // SASE-M1 point 2 (S4-2 mirror of the authority seam; sase_ref
        // module docstring, SASE-M1 section): with the moist seam
        // engaged the n2 slot carries n2_eff (the l_s min below rides
        // it -- points 1/3) and n2d the DRY field; WHERE the moist
        // field departed from the dry field -- the moist-n2 kernel
        // copies unsaturated dry bits, so FP32 inequality identifies
        // exactly the substituted cells (a saturated cell whose DK82
        // value coincides bitwise is substitution-inert by definition,
        // the authority convention) -- the buoyancy source becomes
        // -(K_v/Pr_t)*N^2_m; elsewhere the literal dry expression
        // above stands UNCHANGED (nothing is added, not even +0.0 --
        // the S3-11b zero-guard idiom).  has_moist == 0 leaves n2d a
        // never-dereferenced gated dummy (restrict-alias idiom) and
        // the kernel bitwise pre-M1.
        if (has_moist) {
            real n2e = n2[q];
            if (n2e != n2d[q])
                buoy = -(kv[q] / pr_t) * n2e;
        }
        // S3-9b GEOMETRIC l_d blend (authority dissipation_length,
        // S3-9; was the linear f_blend*delta + (1-f_blend)*leps[q]):
        // FP64 pow per cell with ONE FP32 rounding at l -- the
        // vertical-channel kernel's transcendental convention joined
        // to this kernel's S3-6d FP64-decay idiom; the FP32 l_s min
        // below is untouched.  The f = 0 / f = 1 ENDPOINTS take the
        // explicit branch: the authority's bitwise endpoint contract
        // (f = 0 -> leps, f = 1 -> delta; the retired linear form's
        // own FP-exact endpoint arithmetic) rides numpy's
        // special-cased pow (x**0.0 == 1.0, x**1.0 == x through its
        // exact integral-exponent path), while device pow guarantees
        // pow(x, 0.0) == 1.0 but NOT pow(x, 1.0) == x (measured
        // 1-ulp misses on the RTX 5090, S3-9b report), so the
        // endpoint values are transcribed directly -- same values,
        // same contract, no branch divergence (f_blend is a
        // kernel-wide scalar, exact at 0 and 1 through the FP32
        // argument cast).
        real l;
        if (f_blend == 0.0f) {
            l = leps[q];
        } else if (f_blend == 1.0f) {
            l = delta;
        } else {
            l = (real)(pow((double)delta, (double)f_blend)
                       * pow((double)leps[q], 1.0 - (double)f_blend));
        }
        // The stability length is HOISTED into a named temp so the
        // S3-6k rho below can be formed locally with no new state and
        // no new device field; the value written into l is unchanged
        // (same fminf, same operands, same order).  ls_v == 0.0f marks
        // "no stability limit here" -- the unstable/neutral cells and
        // the has_n2 == 0 path both leave it there, and both are the
        // cells S3-6k must not touch.
        real ls_v = 0.0f;
        if (has_n2) {
            real n2v = n2[q];
            if (n2v > 0.0f) {
                ls_v = ls_coef * root_e / sqrtf(n2v);
                l = fminf(l, ls_v);
            }
        }
        real t_h = (fx[IDX3(k, j, ip)] - fx[IDX3(k, j, im)]) / two_dx
                   + (fy[IDX3(k, jp, i)] - fy[IDX3(k, jm, i)]) / two_dy;
        // S3-6e taper: g = 1.0f is an FP-exact no-op (src == x,
        // x - src == 0, gb == buoy bitwise).
        real g = has_taper ? gtap[q] : 1.0f;
        real x_prod = ph_e[q] + pv[q];
        real src = g * x_prod;
        real gb = g * buoy;
        real e_old = e[q];
        real e_star = e_old + dt * ((src + gb) + t_h);
        // S3-6d analytic decay substep, FP64 per cell (header comment).
        double es64 = (double)e_star;
        // S3-6k decoupled stable-limb dissipation coefficient (authority
        // stable_dissipation_coefficient; sase_ref module docstring,
        // S3-6k section).  has_ces == 0 leaves the multiplicand the
        // LITERAL (double)c_e -- bitwise the pre-S3-6k kernel, nothing
        // added, not even *1.0.  With it on: rho rides l_d against the
        // stability length it was just min-ed with, so where l_s bound
        // l == ls_v BITWISE (it came out of that fminf) and rho is
        // exactly 1.0, matching the authority; the TWO-PRODUCT blends
        // then land C_ES exactly there and c_e exactly at f_blend == 1
        // (the LES limb, RANS-only by construction).
        // ls_v > 0.0f IS THE AUTHORITY'S np.where(stable, blend, C_E):
        // the neutral/unstable return must be the SELECTED literal c_e,
        // never the blend evaluated where crn happens to equal c_e --
        // f*c_e + (1-f)*c_e misses c_e by 1 ulp at 36.6% of f (measured
        // on the authority this session, including the recorded
        // production f = 4.1188928660938e-05).  Both halves branch on
        // the same predicate so the FP64 authority and this FP32 mirror
        // agree BITWISE on the cells S3-6k must not touch.
        double c_eps = (double)c_e;
        if (has_ces && ls_v > 0.0f) {
            double rho = fmin((double)l / (double)ls_v, 1.0);
            double w = pow(rho, (double)cks_exp);
            double crn = (1.0 - w) * (double)c_e + w * (double)c_es;
            c_eps = (double)f_blend * (double)c_e
                    + (1.0 - (double)f_blend) * crn;
        }
        // S3-12 additive e^{3/2} channel (authority
        // additive_dissipation_coefficient on the SAME (l_d, e^n,
        // n2_eff, f), l_ref = neutral_dissipation_length; sase_ref
        // module docstring, S3-12 section).  has_ced == 0 adds NOTHING
        // -- not even +0.0 -- so the kernel is bitwise pre-S3-12 by
        // construction, and the lb slot is then a never-dereferenced
        // gated dummy (the module's restrict-alias idiom).  The
        // ls_v > 0.0f half of the gate IS the authority's
        // np.where(stable, base + added, base) SELECTION: neutral and
        // unstable cells (and the whole has_n2 == 0 test-box path)
        // return the SELECTED base, never base + 0.0 (which would
        // flip a -0.0 and, at n2 <= 0, evaluate ls off its guarded
        // branch).  rho/w are the S3-6k line's own values -- both
        // authority functions form them from the same inputs with the
        // same ops, so one evaluation serves both amendments (re-formed
        // here rather than hoisted so has_ces == 0 stays literally the
        // pre-S3-6k arithmetic).  l_ref rides the l_d blend's OWN
        // endpoint-branch idiom (kernel comment at the l_d blend:
        // device pow(x, 1.0) misses x by 1 ulp on the RTX 5090, while
        // the authority's numpy x**1.0/x**0.0 are exact), so f = 0
        // lands the LITERAL FP32 lb -- the authority's l_B bitwise --
        // and f = 1 the literal delta.  At f = 1 the added term is the
        // arithmetic +0.0 the authority also forms ((1-f) = 0 exactly;
        // every other factor finite and non-negative), and
        // c_eps + 0.0 == c_eps bitwise because c_eps > 0 -- the
        // authority's own LES-limb argument, mirrored, so no f branch
        // guards the ADD itself.
        if (has_ced && ls_v > 0.0f) {
            double rho = fmin((double)l / (double)ls_v, 1.0);
            double w = pow(rho, (double)cks_exp);
            double lref;
            if (f_blend == 0.0f) {
                lref = (double)lb[q];
            } else if (f_blend == 1.0f) {
                lref = (double)delta;
            } else {
                lref = pow((double)delta, (double)f_blend)
                       * pow((double)lb[q], 1.0 - (double)f_blend);
            }
            c_eps += (1.0 - (double)f_blend) * w * (double)c_ed
                     * ((double)l / lref);
        }
        double b = c_eps * sqrt(fmax(es64, (double)e_min))
                   / (2.0 * (double)l);
        double fac = 1.0 + b * (double)dt;
        real e_dec = (real)(es64 / (fac * fac));
        real decay = e_star - e_dec;
        real e_clip = fmaxf(e_dec, e_min);
        real clip_gain = e_clip - e_dec;
        real ht = (decay - clip_gain)
                  + dt * (ph_heat[q] + (x_prod - src));
        e[q] = e_clip;
        heat[q] = ht;
        double dtd = (double)dt;
        m[0] = (double)e_clip - (double)e_old;
        m[1] = dtd * (double)gb;
        m[2] = dtd * (double)t_h;
        m[3] = (double)ht;
    }
    for (int c = 0; c < 4; ++c) sdata[c][threadIdx.x] = m[c];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int c = 0; c < 4; ++c)
                sdata[c][threadIdx.x] += sdata[c][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int c = 0; c < 4; ++c)
            partials[(size_t)c * nblocks + blockIdx.x] = sdata[c][0];
}

// ---------------------------------------------------------------------
// S3-6f: partition-bound kernels (authority bulk_richardson_zi /
// w_structure_functions -- mesoscale sensing concession).
// ---------------------------------------------------------------------

// Per-column bulk-Richardson BL height (authority bulk_richardson_zi),
// one thread per column, FP64 through the final FP32 store: layer
// centers by the in-thread cumulative thickness sum (the
// sase_vertical_channel construction), Rib(k) = (theta_k - theta_1)
// * (g*z_k/theta_1) / max(u_k^2 + v_k^2, wspd2_floor) against the
// level-1 thermal, first upward crossing of rib_crit with linear
// interpolation in Rib between the bracketing layer centers (the
// audited YSU diagnose crossing at one registered critical value).
// No crossing -> the top layer center (permissive fallback); the
// result floors at the FIRST INTERIOR layer center (stable-BL
// fallback); nz = 1 stores z[0].  All conventions documented at the
// authority (sase_ref module docstring, S3-6f section).
extern "C" __global__
void sase_zi_column(const real* __restrict__ u, const real* __restrict__ v,
                    const real* __restrict__ theta,
                    const real* __restrict__ t, real dz, int t_mode,
                    real* __restrict__ zi,
                    real rib_crit, real wspd2_floor, real g_accel,
                    int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double zc = 0.0;
    double t0 = sase_thick(t, dz, t_mode, 0, ncol, col);
    double z_prev = 0.5 * t0;                  // z[0]
    zc = t0;
    if (nz == 1) { zi[col] = (real)z_prev; return; }
    double thermal = (double)theta[col];
    double rib_prev = 0.0;                     // own-level Rib, exact
    double z1_int = 0.0;                       // first interior center
    double out = -1.0;
    double zk = z_prev;
    for (int k = 1; k < nz; ++k) {
        double tk = sase_thick(t, dz, t_mode, k, ncol, col);
        zk = zc + 0.5 * tk;
        zc += tk;
        if (k == 1) z1_int = zk;
        size_t q = (size_t)k * ncol + col;
        double uu = (double)u[q], vv = (double)v[q];
        double spd2 = fmax(uu * uu + vv * vv, (double)wspd2_floor);
        double rib = ((double)theta[q] - thermal)
                     * ((double)g_accel * zk / thermal) / spd2;
        if (out < 0.0 && rib >= (double)rib_crit) {
            double frac = ((double)rib_crit - rib_prev) / (rib - rib_prev);
            out = z_prev + frac * (zk - z_prev);
        }
        z_prev = zk;
        rib_prev = rib;
    }
    if (out < 0.0) out = zk;                   // no crossing: top center
    zi[col] = (real)fmax(out, z1_int);
}

// N^2-screened w structure-function partial sums + interior floored-e
// sum (authority w_structure_functions + the split step's e_mean):
// per SCREEN-PASSING cell (dry n2 <= n2_screen at the anchor cell;
// has_n2 == 0 treats every cell as neutral = passing -- the gated
// dummy-pointer idiom of the other kernels) accumulate
// sum(dx_inc^2 + dy_inc^2) of w for r in {1, 2, 4} plus the passing
// count; per INTERIOR cell (the bw anchor mask of sase_solve_partial,
// screen-independent) accumulate fmaxf(e, e_min).  partials has shape
// (5, nblocks): [d2_r1, d2_r2, d2_r4, count, e_sum]; the host tail
// finishes 0.5*sum/count per r and e_sum/ncell_interior, then applies
// the authority _w_bound_tail.  Increments are FP32 (production
// arithmetic), squares/sums FP64 -- the sase_structure_partial
// contract.
extern "C" __global__
void sase_w_sensor_partial(const real* __restrict__ w,
                           const real* __restrict__ e,
                           const real* __restrict__ n2, int has_n2,
                           real n2_screen, real e_min,
                           double* __restrict__ partials,
                           int ny, int nx, int bw,
                           int nblocks, long long ncell)
{
    __shared__ double sdata[5][SASE_TPB];
    long long qq = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double m[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (qq < ncell) {
        int i = (int)(qq % nx);
        int j = (int)((qq / nx) % ny);
        int k = (int)(qq / ((long long)nx * ny));
        bool interior = !(bw > 0 && (i < bw || i >= nx - bw
                                     || j < bw || j >= ny - bw));
        if (interior) {
            m[4] = (double)fmaxf(e[qq], e_min);
            bool pass = (has_n2 == 0) || (n2[qq] <= n2_screen);
            if (pass) {
                real wc = w[qq];
                const int rs[3] = {1, 2, 4};
                for (int tt = 0; tt < 3; ++tt) {
                    real dxr = w[IDX3(k, j, PERIODIC(i + rs[tt], nx))] - wc;
                    real dyr = w[IDX3(k, PERIODIC(j + rs[tt], ny), i)] - wc;
                    m[tt] = (double)dxr * (double)dxr
                            + (double)dyr * (double)dyr;
                }
                m[3] = 1.0;
            }
        }
    }
    for (int t = 0; t < 5; ++t) sdata[t][threadIdx.x] = m[t];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int t = 0; t < 5; ++t)
                sdata[t][threadIdx.x] += sdata[t][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int t = 0; t < 5; ++t)
            partials[(size_t)t * nblocks + blockIdx.x] = sdata[t][0];
}

// ---------------------------------------------------------------------
// Stage-3 Task 6: driver coupling -- N^2 from the model theta profile
// and K_h down-gradient scalar mixing.
// ---------------------------------------------------------------------

// Brunt-Vaisala N^2 = (g/theta) * d(theta)/dz on the shared z_mode
// vertical stencil (authority brunt_vaisala_n2): the stability-length
// input the driver computes from the model's own theta profile, on
// EXACTLY the operator every other SASE vertical derivative uses.
extern "C" __global__
void sase_n2(const real* __restrict__ theta, real* __restrict__ n2,
             const real* __restrict__ cm, const real* __restrict__ c0,
             const real* __restrict__ cp,
             real two_dx, real two_dy, real dz, real two_dz,
             real h_lo, real h_hi, int z_mode,
             real g_accel, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int km = sase_km(k, nz, z_mode), kp = sase_kp(k, nz, z_mode);
    size_t q = IDX3(k, j, i);
    real dthdz = sase_ddz(theta[IDX3(km, j, i)], theta[q],
                          theta[IDX3(kp, j, i)], k, q, nz, cm, c0, cp,
                          dz, two_dz, h_lo, h_hi, z_mode);
    n2[q] = (g_accel / theta[q]) * dthdz;
}

// SASE-M1 effective stability N^2_eff (S4-2 device mirror of the
// authority ``moist_n2``; sase_ref module docstring, SASE-M1 section
// has the DK82 derivation): the Durran & Klemp (1982, JAS 39,
// 2152-2158, Eq. 36) saturated moist N^2_m with condensate loading in
// saturated cells, the dry n2 input BITWISE elsewhere -- the
// unsaturated branch copies the literal FP32 dry bits
// (out[q] = n2_dry[q]), so a switch-off cell adds/changes NOTHING
// (the M1 unsaturated-identity contract, the S3-11b zero-guard
// idiom).  One thread per column, FP64 end to end in the authority's
// exact op order with ONE FP32 rounding at the saturated store:
//   T   = theta*(p/P0)^kappa          [kappa = RD/CP, host FP64]
//   es  = 1000*SVP1*exp(SVP2*(T - SVPT0)/(T - SVP3))  [Tetens liquid]
//   qs  = EP2*es/(p - es)
//   a   = 1 + XLV*qs/(RD*T)
//   b   = 1 + EP2*XLV^2*qs/(CP*RD*T^2)
//   N2m = g*((a/b)*(ddz(theta)/theta + (XLV/(CP*T))*ddz(qs))
//            - ddz(qv + qc))
//   sat = (qc > 0) || (qv >= qs)      [MOIST_STABILITY_SWITCH =
//                                      "binary-qc-or-rh100-liquid"]
// Every vertical derivative rides the authority ``_ddz_var`` stencil
// rebuilt in-thread in FP64: z centers by the SEQUENTIAL
// cumsum-half-layer grouping (s += t_k; z_k = s - 0.5*t_k -- exactly
// the authority ``_z_centers`` partial-sum-then-subtract order, not
// the telescoped half-sum form of sase_ddz_coefficients), interior
// three-point Lagrange coefficients in the authority's literal
// grouping, one-sided linear-exact edge rows; nz == 1 degenerates to
// zero derivatives (the authority's single-level branch).  The
// theta/T/qs/qw columns and z centers stage into per-thread local
// arrays (the Thomas-sweep idiom, SASE_KMAX bound; 5 KMAX doubles).
// Constants arrive as FP64 kernel arguments single-sourced from the
// authority registry through the launcher (no in-source twins).
// DOCUMENTED DOMAIN (authority convention): physical states have
// T > SVP3 and p > es; violations surface as non-finite output, never
// a silent switch flip.
extern "C" __global__
void sase_moist_n2(const real* __restrict__ theta,
                   const real* __restrict__ qv,
                   const real* __restrict__ qc,
                   const real* __restrict__ p,
                   const real* __restrict__ n2_dry,
                   real* __restrict__ out,
                   const real* __restrict__ t, real dz, int t_mode,
                   double kappa, double p0, double svp1, double svp2,
                   double svp3, double svpt0, double ep2, double xlv,
                   double rd, double cp_air, double g_accel,
                   int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double zc_[SASE_KMAX], th_[SASE_KMAX], tt_[SASE_KMAX],
           qs_[SASE_KMAX], qw_[SASE_KMAX];
    double s = 0.0;
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double tk = sase_thick(t, dz, t_mode, k, ncol, col);
        s += tk;
        zc_[k] = s - 0.5 * tk;
        double thv = (double)theta[q];
        double pv = (double)p[q];
        double tv = thv * pow(pv / p0, kappa);
        double es = 1000.0 * svp1
                    * exp(svp2 * (tv - svpt0) / (tv - svp3));
        th_[k] = thv;
        tt_[k] = tv;
        qs_[k] = ep2 * es / (pv - es);
        qw_[k] = (double)qv[q] + (double)qc[q];
    }
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double dth, dqs, dqw;
        if (nz == 1) {
            dth = dqs = dqw = 0.0;     // authority nz == 1: zeros
        } else if (k == 0) {
            double h = zc_[1] - zc_[0];
            dth = (th_[1] - th_[0]) / h;
            dqs = (qs_[1] - qs_[0]) / h;
            dqw = (qw_[1] - qw_[0]) / h;
        } else if (k == nz - 1) {
            double h = zc_[nz - 1] - zc_[nz - 2];
            dth = (th_[nz - 1] - th_[nz - 2]) / h;
            dqs = (qs_[nz - 1] - qs_[nz - 2]) / h;
            dqw = (qw_[nz - 1] - qw_[nz - 2]) / h;
        } else {
            double hm = zc_[k] - zc_[k - 1];
            double hp = zc_[k + 1] - zc_[k];
            double cmv = -(hp / (hm * (hp + hm)));
            double c0v = (hp - hm) / (hp * hm);
            double cpv = hm / (hp * (hp + hm));
            dth = cmv * th_[k - 1] + c0v * th_[k] + cpv * th_[k + 1];
            dqs = cmv * qs_[k - 1] + c0v * qs_[k] + cpv * qs_[k + 1];
            dqw = cmv * qw_[k - 1] + c0v * qw_[k] + cpv * qw_[k + 1];
        }
        double a_fac = 1.0 + xlv * qs_[k] / (rd * tt_[k]);
        double b_fac = 1.0 + ep2 * xlv * xlv * qs_[k]
                       / (cp_air * rd * tt_[k] * tt_[k]);
        double bracket = dth / th_[k]
                         + (xlv / (cp_air * tt_[k])) * dqs;
        double n2m = g_accel * ((a_fac / b_fac) * bracket - dqw);
        bool sat = ((double)qc[q] > 0.0) || ((double)qv[q] >= qs_[k]);
        out[q] = sat ? (real)n2m : n2_dry[q];
    }
}

// Horizontal scalar-mixing fluxes f_i = K_h * d(s)/dx_i (the
// horizontal-explicit half of the S3-6c split scalar channel; the
// vertical half is the K_v/Pr_t(f) sase_thomas_scalar solve): K_h =
// kh_coef*sqrt(max(e, E_MIN)) with kh_coef =
// (f*c_nu + (1-f)*C_K)*delta/Pr_t(f) supplied by the launcher -- K_m's
// blend over the (S3-6g regime-blended) turbulent Prandtl number.  Unlike sase_e_hflux the
// scalar itself is NOT floored (only e is) and the transport carries
// no factor 2 (that factor is the e-equation's own transport
// constant).  The retired full-3D pair (sase_scalar_flux /
// sase_flux_div) mirrored the v0 authority scalar_mix, whose explicit
// vertical leg is superseded by the implicit channel.
extern "C" __global__
void sase_scalar_hflux(const real* __restrict__ s,
                       const real* __restrict__ e,
                       real* __restrict__ fx, real* __restrict__ fy,
                       real two_dx, real two_dy,
                       real kh_coef, real e_min, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    size_t q = IDX3(k, j, i);
    real kh = kh_coef * sqrtf(fmaxf(e[q], e_min));
    real dsdx = (s[IDX3(k, j, ip)] - s[IDX3(k, j, im)]) / two_dx;
    real dsdy = (s[IDX3(k, jp, i)] - s[IDX3(k, jm, i)]) / two_dy;
    fx[q] = kh * dsdx;
    fy[q] = kh * dsdy;
}

// S3-6e field-mode horizontal scalar fluxes: K_h = kh_fac * km with km
// the GOVERNED horizontal diffusivity field the split step exported
// (kh_fac carries the 1/Pr_t(f) scalar convention -- S3-6g blended
// Prandtl number at the step's used f).  Same two-pass
// grouping as sase_scalar_hflux (authority ``scalar_hmix``); the
// kh_coef*sqrt(e) kernel above stays for the v0 seam and its parity
// fixtures.
extern "C" __global__
void sase_scalar_hflux_km(const real* __restrict__ s,
                          const real* __restrict__ km,
                          real* __restrict__ fx, real* __restrict__ fy,
                          real two_dx, real two_dy,
                          real kh_fac, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    size_t q = IDX3(k, j, i);
    real kh = kh_fac * km[q];
    real dsdx = (s[IDX3(k, j, ip)] - s[IDX3(k, j, im)]) / two_dx;
    real dsdy = (s[IDX3(k, jp, i)] - s[IDX3(k, jm, i)]) / two_dy;
    fx[q] = kh * dsdx;
    fy[q] = kh * dsdy;
}

// Horizontal flux divergence out = d(fx)/dx + d(fy)/dy (second pass of
// the horizontal scalar channel).
extern "C" __global__
void sase_hflux_div(const real* __restrict__ fx,
                    const real* __restrict__ fy,
                    real* __restrict__ out,
                    real two_dx, real two_dy,
                    int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int ip = PERIODIC(i + 1, nx), im = PERIODIC(i - 1, nx);
    int jp = PERIODIC(j + 1, ny), jm = PERIODIC(j - 1, ny);
    size_t q = IDX3(k, j, i);
    out[q] = (fx[IDX3(k, j, ip)] - fx[IDX3(k, j, im)]) / two_dx
             + (fy[IDX3(k, jp, i)] - fy[IDX3(k, jm, i)]) / two_dy;
}

// ---------------------------------------------------------------------
// SASE-M2 conditional venting limb (S4-5 device mirror of the authority
// ``plume_vent_flux``) + the S4-5 driver deposit seam's cap-family
// rescale (authority ``vent_deposit_rescale``).
// ---------------------------------------------------------------------

// SASE-M2 parcel saturation adjustment (authority
// ``_vent_saturation_adjust``): the (theta, qv, qc) of a parcel with
// liquid-water potential temperature th_l and total water qt at
// pressure p, by a FIXED VENT_SAT_ADJUST_ITERS Newton iteration on
// f(T) = T - Pi*theta_l - (L/cp)*qc(T) from the unsaturated first guess
// T0 = Pi*theta_l, with the analytic derivative.  Transcribed line for
// line in the authority's op order (the <= 2e-6 parity tier is
// sensitive to expression grouping): es/qs are re-formed INSIDE the
// loop before the residual, the derivative rides the SAME es, and the
// post-loop es/qs recomputation is a THIRTEENTH evaluation, not the
// loop's last one.  ``iters`` arrives from the launcher's read of the
// registered module constant (the authority's runtime-live default).
__device__ __forceinline__ void
sase_vent_sat_adjust(double thl, double qt, double pv, int iters,
                     double kappa, double p0, double svp1, double svp2,
                     double svp3, double svpt0, double ep2,
                     double xlv_cp, double* th_out, double* qv_out,
                     double* qc_out)
{
    double pi = pow(pv / p0, kappa);
    double t = pi * thl;
    for (int it = 0; it < iters; ++it) {
        double es = 1000.0 * svp1 * exp(svp2 * (t - svpt0) / (t - svp3));
        double qs = ep2 * es / (pv - es);
        double qc_i = fmax(qt - qs, 0.0);
        double res = t - pi * thl - xlv_cp * qc_i;
        double des = es * svp2 * (svpt0 - svp3)
                     / ((t - svp3) * (t - svp3));
        double dqs = ep2 * pv * des / ((pv - es) * (pv - es));
        double slope = (qc_i > 0.0) ? (1.0 + xlv_cp * dqs) : 1.0;
        t = t - res / slope;
    }
    double es = 1000.0 * svp1 * exp(svp2 * (t - svpt0) / (t - svp3));
    double qs = ep2 * es / (pv - es);
    double qc_p = fmax(qt - qs, 0.0);
    *th_out = t / pi;
    *qv_out = qt - qc_p;
    *qc_out = qc_p;
}

// SASE-M2 conditional venting limb (S4-5 device mirror of the authority
// ``plume_vent_flux``; sase_ref module docstring, SASE-M2 section, has
// the complete formulation and every derivation).  ONE THREAD PER
// COLUMN, sequential FP64 sweep in the authority's exact op order (the
// sase_vertical_channel pattern, spec C12), with a single FP32 rounding
// at the face stores.  Writes the three face-registered flux profiles
// (nz+1, ny, nx) with F[0] = F[nz] = +0.0 written as the LITERAL +0.0
// (never a computed-then-zeroed value, never negative zero) and, under
// has_idx, the seven diagnosed indices the S4-5 parity gate compares
// BIT-EXACTLY against the authority
//   idx rows: 0 k_base, 1 k_top, 2 k_r, 3 k_lid, 4 k_lfc, 5 k_nb, 6 kb
// (-1 on every row of a stood-down column).  Index agreement is a
// SEPARATE pass/fail gate from the flux tolerance: a one-cell k_base
// move is a median 35% flux change on real fields (design doc SASE-M2
// amendment "root / anchor separation"), so "within tolerance" is not a
// defence for a differing index.
//
// Structure, and the seven places a mirror of this function goes wrong
// (each has a verification receipt in the CPU history -- S4-4 report
// rounds 1-7):
//  1. THE WHOLE LAYER STRUCTURE rides qt >= qs (VENT_K_LID_MEMBERSHIP =
//     "qt-ge-qs-v1"), NOT qc > 0: membership, the run base, contiguity
//     and the run top are all the saturation-state criterion, and the
//     M1 mask (here the bitwise n2_moist != n2_dry departure, the M1b
//     seam's own idiom) is a VETO that can never EXTEND a run.  The
//     limb is therefore insensitive to +-1e-12 kg/kg condensate shifts.
//  2. ROOT / ANCHOR SEPARATION: the thermodynamic root k_r is the
//     highest interior theta_es maximum at or below k_base, floored
//     structurally at k_base - (k_top - k_base) - 1, falling back to
//     k_base under monotone theta_es; theta_es is a function of
//     (theta, p) ONLY, so the root is condensate-free by construction.
//     The AMPLITUDE does not ride on it: the ebar window and the shape
//     normalization both anchor at k_base (with the structural
//     k_base = 0 guard max(k_base, 1), since z_f[0] = 0 -- a guard the
//     step-1a stand-down now makes unreachable on an active column).
//  3. k_lid = k_top + 2 -- the entrainment-zone cell's OWN top face
//     (C9 amendment: the single cell above the saturated run is the
//     discrete entrainment zone and the only legitimate detrainment
//     recipient; the cap proper receives bitwise zero).
//  4. The ebar window spans k_base .. min(k_top + 1, nz - 1) -- the
//     saturated-layer base through the entrainment-zone cell, NOT the
//     raw masked run and NOT the root.
//  5. The natural-NB / buoyancy-peak / taper branch is live physics
//     that is degenerate on most columns (a build with those searches
//     deleted agreed with 16 of 17 CPU fixtures bitwise) -- it is
//     transcribed here in full.
//  6. STAND-DOWN (bitwise +0.0 on every face): no qualifying run, a
//     qualifying run based in the LOWEST MODEL LEVEL (k_base == 0 --
//     VENT_ANCHOR_RULE, step 1a), no LFC below k_lid, k_lid beyond
//     VENT_DEPTH_CAP of the root, k_lid past the column top.
//  7. f_blend = 1 gives bitwise +0.0 through the two-product blend
//     m_used = (1 - f)*m_base -- the gate (m_used > 0.0) then closes
//     every face.
//
// Per-thread FP64 column arrays (spec C12 budget ~4-6 KMAX doubles):
// thes_, b_, dth_, dqv_, dqc_, ss_ = SIX, plus a one-byte membership
// column.  Layer thicknesses come from sase_thick (the shared t_mode
// contract) and the z centers/faces are re-derived by the SAME
// sequential partial sums the authority ``_column_geometry`` uses
// (z = cumsum(t) - 0.5*t centers; z_f = cumsum(t) faces with
// z_f[0] = 0), so no thickness array is staged.
extern "C" __global__
void sase_plume_vent_flux(const real* __restrict__ theta,
                          const real* __restrict__ qv,
                          const real* __restrict__ qc,
                          const real* __restrict__ p,
                          const real* __restrict__ e_sgs,
                          const real* __restrict__ n2m,
                          const real* __restrict__ n2d,
                          const real* __restrict__ rho1,
                          real* __restrict__ f_th,
                          real* __restrict__ f_qv,
                          real* __restrict__ f_qc,
                          int* __restrict__ idx, int has_idx,
                          const real* __restrict__ t, real dz, int t_mode,
                          double f_blend, double kappa, double p0,
                          double svp1, double svp2, double svp3,
                          double svpt0, double ep2, double xlv_cp,
                          double xlv, double cp_air,
                          double rvm1, double g_accel, double e_min,
                          double mb_coef, double ent_coef,
                          double sigw_share, double depth_cap,
                          int min_run_cells, int sat_iters, int per_level,
                          int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double thes_[SASE_KMAX], b_[SASE_KMAX], dth_[SASE_KMAX],
           dqv_[SASE_KMAX], dqc_[SASE_KMAX], ss_[SASE_KMAX];
    unsigned char memb_[SASE_KMAX];

    // Structural end faces, written as the LITERAL +0.0 (interface
    // contract): never computed, never negative zero.  The interior
    // faces are pre-zeroed too so a stood-down column and every
    // out-of-support face carry the same literal.
    for (int j = 0; j <= nz; ++j) {
        size_t qj = (size_t)j * ncol + col;
        f_th[qj] = 0.0f; f_qv[qj] = 0.0f; f_qc[qj] = 0.0f;
    }
    if (has_idx)
        for (int r = 0; r < 7; ++r)
            idx[(size_t)r * ncol + col] = -1;

    // -- thermodynamic state (the M1 Tetens/Exner primitives) ---------
    // thes = theta*exp(XLV*qs/(CP*T)) and the membership test
    // qt = qv + qc >= qs INSIDE the M1 mask (the bitwise n2_moist !=
    // n2_dry departure -- the seam's own mask, never re-derived here).
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double thv = (double)theta[q];
        double pv = (double)p[q];
        double tv = thv * pow(pv / p0, kappa);
        double es = 1000.0 * svp1 * exp(svp2 * (tv - svpt0) / (tv - svp3));
        double qs = ep2 * es / (pv - es);
        // NOTE the grouping: the authority writes
        //   thes = th*exp(XLV*qs_env/(CP_AIR*t_env)),
        // NOT (XLV/CP_AIR)*qs/t -- the two differ in the last ulp, and
        // theta_es feeds the peak/decrease COMPARISONS that select
        // k_r, k_base and k_top, where a last-ulp difference is a
        // one-cell index move and a ~35% flux change.  xlv_cp is the
        // authority's OWN (XLV/CP_AIR) grouping and is used only where
        // the authority parenthesizes it that way (thl_env and the
        // saturation adjustment).
        thes_[k] = thv * exp(xlv * qs / (cp_air * tv));
        double qt = (double)qv[q] + (double)qc[q];
        bool mask = (n2m[q] != n2d[q]);
        memb_[k] = (unsigned char)((mask && (qt >= qs)) ? 1 : 0);
    }

    // -- step 1: lowest qualifying saturated moist-unstable run -------
    bool in_run = false, mono = false, chosen = false;
    int run_s = 0, k_base = -1, k_top = -1;
    double base_thes = 0.0, prev_thes = 0.0;
    for (int k = 0; k < nz; ++k) {
        bool mk = memb_[k] != 0;
        bool start = mk && !in_run;
        bool cont = mk && in_run;
        if (start) run_s = k;
        if (start) base_thes = thes_[k];
        if (start) mono = true;
        else if (cont) mono = mono && (thes_[k] < prev_thes);
        if (mk) prev_thes = thes_[k];
        bool end_here = (k + 1 < nz) ? (mk && (memb_[k + 1] == 0)) : mk;
        bool long_enough = (k - run_s) >= (min_run_cells - 1);
        bool reading = per_level ? mono : ((thes_[k] - base_thes) < 0.0);
        bool take = end_here && long_enough && reading && !chosen;
        if (take) { k_base = run_s; k_top = k; }
        chosen = chosen || take;
        in_run = mk;
    }
    if (!chosen) return;                       // stand-down: zeros

    // -- step 1a: SURFACE-BASED STAND-DOWN ----------------------------
    // VENT_ANCHOR_RULE = "cloud-base-face-standdown-v1" (authority
    // ``plume_vent_flux`` step 1a; design doc SASE-M section 4,
    // amendment "a surface-based saturated layer stands the limb
    // down").  The FOURTH registered stand-down: a run based in the
    // LOWEST MODEL LEVEL has no cloud-base face for the step-5 shape to
    // normalize on (z_f[0] = 0 identically), so the pre-amendment
    // max(k_base, 1) guard anchored it on the lowest layer THICKNESS
    // and made the amplitude a function of the vertical grid.  Every
    // face is already the literal +0.0 and every idx row already -1 at
    // this point, so the bare return IS the bitwise stand-down.
    if (k_base == 0) return;                   // stand-down: zeros

    // -- step 1b: root = the theta_es-decrease base, depth-bounded ----
    int k_r = k_base;
    int k_r_floor = k_base - (k_top - k_base) - 1;
    for (int k = 1; k < nz - 1; ++k) {
        bool is_peak = (thes_[k] > thes_[k - 1])
                       && (thes_[k] > thes_[k + 1]);
        if (is_peak && k <= k_base && k >= k_r_floor) k_r = k;
    }

    // -- step 1c: BL-integrated ebar from CLOUD BASE up ---------------
    int k_hi = (k_top + 1 < nz - 1) ? (k_top + 1) : (nz - 1);
    double ebar_num = 0.0, ebar_den = 0.0;
    for (int k = 0; k < nz; ++k) {
        if (k >= k_base && k <= k_hi) {
            size_t q = (size_t)k * ncol + col;
            double tk = sase_thick(t, dz, t_mode, k, ncol, col);
            double ek = fmax((double)e_sgs[q], e_min);
            ebar_num = ebar_num + tk * ek;
            ebar_den = ebar_den + tk;
        }
    }

    // -- step 2: amplitude (the FP-exact two-product blend) -----------
    double ebar = fmax(ebar_num / ((ebar_den > 0.0) ? ebar_den : 1.0),
                       e_min);
    double sigma_w = sqrt(sigw_share * ebar);
    double m_base = mb_coef * (double)rho1[col] * sigma_w;
    double m_used = (1.0 - f_blend) * m_base;

    // -- step 3: entraining parcel ascent -----------------------------
    // Segment-exact update against face-mean environments with
    // eps = VENT_ENT_COEF/z; the previous level's (thl_env, qt_env, z)
    // are carried forward in registers rather than staged (identical
    // arithmetic, three fewer KMAX arrays).
    double thl_p = 0.0, qt_p = 0.0;
    bool started = false;
    double thl_prev = 0.0, qt_prev = 0.0, z_prev = 0.0, z_run = 0.0;
    for (int k = 0; k < nz; ++k) {
        size_t q = (size_t)k * ncol + col;
        double tk = sase_thick(t, dz, t_mode, k, ncol, col);
        z_run += tk;
        double zk = z_run - 0.5 * tk;           // _column_geometry centers
        double thv = (double)theta[q];
        double qvv = (double)qv[q];
        double qcv = (double)qc[q];
        double pv = (double)p[q];
        double tv = thv * pow(pv / p0, kappa);
        double qt_env = qvv + qcv;
        double thl_env = thv - xlv_cp * qcv * thv / tv;
        b_[k] = 0.0; dth_[k] = 0.0; dqv_[k] = 0.0; dqc_[k] = 0.0;
        if (k > 0 && started) {                 // adv = started & active
            double thl_f = 0.5 * (thl_prev + thl_env);
            double qt_f = 0.5 * (qt_prev + qt_env);
            double fac = pow(z_prev / zk, ent_coef);
            double thl_n = thl_f + (thl_p - thl_f) * fac;
            double qt_n = qt_f + (qt_p - qt_f) * fac;
            double th_p, qv_p, qc_p;
            sase_vent_sat_adjust(thl_n, qt_n, pv, sat_iters, kappa, p0,
                                 svp1, svp2, svp3, svpt0, ep2, xlv_cp,
                                 &th_p, &qv_p, &qc_p);
            double tv_p = th_p * (1.0 + rvm1 * qv_p - qc_p);
            double tv_e = thv * (1.0 + rvm1 * qvv - qcv);
            b_[k] = g_accel * (tv_p - tv_e) / tv_e;
            dth_[k] = th_p - thv;
            dqv_[k] = qv_p - qvv;
            dqc_[k] = qc_p - qcv;
            thl_p = thl_n;
            qt_p = qt_n;
        }
        if (k_r == k) {                         // init = active & (k_r==k)
            thl_p = thl_env;
            qt_p = qt_env;
            started = true;
        }
        thl_prev = thl_env; qt_prev = qt_env; z_prev = zk;
    }

    // -- step 4: LFC / NB termination + inversion-base cap ------------
    // k_lid = k_top + 2 is the entrainment-zone cell's own TOP face
    // (C9 amendment); a parcel still buoyant when the search REACHES
    // k_lid terminates there -- termination never crosses into the cap
    // proper above the entrainment zone.
    int k_lid = k_top + 2;
    double zr = 0.0;
    {
        double run = 0.0;
        for (int k = 0; k <= k_r; ++k) {
            double tk = sase_thick(t, dz, t_mode, k, ncol, col);
            run += tk;
            zr = run - 0.5 * tk;
        }
    }
    bool lfc_found = false, nb_found = false;
    int k_nb = -1, kb = -1, k_lfc = -1;
    double bmax = 0.0;
    {
        double run = 0.0;
        for (int k = 0; k < nz; ++k) {
            double tk = sase_thick(t, dz, t_mode, k, ncol, col);
            run += tk;
            double zk = run - 0.5 * tk;
            bool above = k > k_r;
            bool within_layer = above && (k < k_lid);
            bool incap = within_layer && ((zk - zr) <= depth_cap);
            bool pos = b_[k] > 0.0;
            bool upd_lfc = incap && !lfc_found && pos;
            if (upd_lfc) k_lfc = k;
            lfc_found = lfc_found || upd_lfc;
            bool cand = incap && lfc_found && !nb_found && pos;
            bool upd_b = cand && (b_[k] > bmax);
            if (upd_b) { bmax = b_[k]; kb = k; }
            bool upd_nb = incap && lfc_found && !nb_found && !pos;
            if (upd_nb) k_nb = k;
            nb_found = nb_found || upd_nb;
            bool at_lid = above && (k == k_lid) && lfc_found && !nb_found
                          && ((zk - zr) <= depth_cap);
            if (at_lid) k_nb = k;
            nb_found = nb_found || at_lid;
        }
    }
    if (!nb_found) return;                     // stand-down: zeros
    if (has_idx) {
        idx[col] = k_base;
        idx[(size_t)1 * ncol + col] = k_top;
        idx[(size_t)2 * ncol + col] = k_r;
        idx[(size_t)3 * ncol + col] = k_lid;
        idx[(size_t)4 * ncol + col] = k_lfc;
        idx[(size_t)5 * ncol + col] = k_nb;
        idx[(size_t)6 * ncol + col] = kb;
    }

    // -- step 5: face mass-flux shape M_hat ---------------------------
    // ss = the REVERSE cumulative sum of the remaining-buoyancy weight
    // (authority flip/cumsum/flip: accumulated from the column top
    // DOWNWARD, which is what makes the taper's roundoff bit-identical).
    {
        double acc = 0.0;
        for (int k = nz - 1; k >= 0; --k) {
            double w = 0.0;
            if (k > kb && k < k_nb)
                w = b_[k] * sase_thick(t, dz, t_mode, k, ncol, col);
            acc += w;
            ss_[k] = acc;
        }
    }
    // k_base >= 1 on every column that reaches here (step 1a), so this
    // max() is a provable NO-OP kept as the structural statement that
    // face 0 is never a normalization face -- mirror of the authority's
    // np.maximum(k_base, 1).  Do not delete it; do not read it as
    // active protection either.
    int ja = (k_base > 1) ? k_base : 1;         // k_base = 0 guard
    double zf_a = 0.0, zf_pk = 0.0;
    {
        double run = 0.0;
        for (int j = 0; j <= nz; ++j) {
            if (j == ja) zf_a = run;
            if (j == kb + 1) zf_pk = run;
            if (j < nz) run += sase_thick(t, dz, t_mode, j, ncol, col);
        }
    }
    double den = ss_[kb + 1];
    double mh_pk = pow(zf_pk / zf_a, ent_coef);

    // -- step 6: face fluxes ------------------------------------------
    {
        double run = sase_thick(t, dz, t_mode, 0, ncol, col);   // z_f[1]
        for (int j = 1; j < nz; ++j) {
            bool grow = (j >= k_r + 1) && (j <= kb + 1) && (j < k_nb);
            bool tap = (j > kb + 1) && (j < k_nb) && (den > 0.0);
            double g = pow(run / zf_a, ent_coef);
            double r = ss_[j] / ((den > 0.0) ? den : 1.0);
            double mh = grow ? g : (tap ? (mh_pk * r) : 0.0);
            if ((mh > 0.0) && (m_used > 0.0)) {
                size_t qj = (size_t)j * ncol + col;
                f_th[qj] = (real)(m_used * mh
                                  * (0.5 * (dth_[j - 1] + dth_[j])));
                f_qv[qj] = (real)(m_used * mh
                                  * (0.5 * (dqv_[j - 1] + dqv_[j])));
                f_qc[qj] = (real)(m_used * mh
                                  * (0.5 * (dqc_[j - 1] + dqc_[j])));
            }
            run += sase_thick(t, dz, t_mode, j, ncol, col);
        }
    }
}

// S4-5 deposit seam, pass 1 (authority ``vent_deposit_rescale``): the
// registered CAP-FAMILY uniform rescale factor
//   s = min(1, THETA_CAP/|dtheta|max, QT_CAP/|dqv|max, QT_CAP/|dqc|max)
// over the column's own maxima of the UNSCALED deposits.  One thread per
// column, FP64 in and out (the scale plane is the driver's one FP64
// field -- rounding it to FP32 would put a needless second rounding
// between the authority's cap and the deposited value).  A per-level
// clip destroys the ledger (measured sum thick*dtheta = -3.74 against
// 0.0 under this uniform rescale), which is why the factor is a
// per-COLUMN scalar applied to all three rows.  DIVIDE GUARD: an
// inactive column has every |d|max == +0.0, so the quotient is skipped
// entirely and the running minimum keeps 1.0 (the authority's np.where
// guard; the authority's -W error::RuntimeWarning policy is what makes
// the unguarded division a failure there).
extern "C" __global__
void sase_vent_deposit_scale(const real* __restrict__ f_th,
                             const real* __restrict__ f_qv,
                             const real* __restrict__ f_qc,
                             const real* __restrict__ rho1,
                             double* __restrict__ scale,
                             const real* __restrict__ t, real dz,
                             int t_mode, double dt, double theta_cap,
                             double qt_cap, int nz, int ny, int nx)
{
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = (long long)ny * nx;
    if (col >= ncol) return;
    double rho = (double)rho1[col];
    double dmax[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < nz; ++k) {
        double tk = sase_thick(t, dz, t_mode, k, ncol, col);
        size_t q0 = (size_t)k * ncol + col;
        size_t q1 = (size_t)(k + 1) * ncol + col;
        double d[3];
        d[0] = ((double)f_th[q0] - (double)f_th[q1]) * dt / (rho * tk);
        d[1] = ((double)f_qv[q0] - (double)f_qv[q1]) * dt / (rho * tk);
        d[2] = ((double)f_qc[q0] - (double)f_qc[q1]) * dt / (rho * tk);
        for (int r = 0; r < 3; ++r)
            dmax[r] = fmax(dmax[r], fabs(d[r]));
    }
    double s = 1.0;
    const double caps[3] = {theta_cap, qt_cap, qt_cap};
    for (int r = 0; r < 3; ++r)
        if (dmax[r] > 0.0) s = fmin(s, caps[r] / dmax[r]);
    scale[col] = s;
}

// S4-5 deposit seam, pass 2 (authority ``vent_deposit_rescale``): the
// EXPLICIT flux-form RHS deposit of ONE scalar row,
//   phi*[k] += (Fs[k] - Fs[k+1])*dt/(rho1*thick_k),  Fs = s*F,
// added to the pre-solve state phi* in the driver's scalar loop BEFORE
// launch_implicit_vertical_diffusion -- the registered deposit-then-
// solve order generalizing SFC_SCALAR_FLUX = "explicit-deposit-v1".
// NOTHING is inserted in sase_split_step and NO Thomas row is touched
// (the solver's pinned max principle is exactly what non-local
// transport must be free to violate).  The scale multiplies the FLUXES,
// not the formed deposit (authority wording ``F_phi *= s``; the two
// differ in the last ulp).  LEDGER: F[0] = F[nz] = +0.0 by the
// interface contract, so sum_k thick_k*dphi_k = (dt/rho1)*(Fs[0] -
// Fs[nz]) = 0 -- the interior faces telescope and the surface flux
// stays owned by the S3-11a deposit (no double counting).
extern "C" __global__
void sase_vent_deposit(real* __restrict__ phi,
                       const real* __restrict__ f_row,
                       const double* __restrict__ scale,
                       const real* __restrict__ rho1,
                       const real* __restrict__ t, real dz, int t_mode,
                       double dt, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    long long ncol = (long long)ny * nx;
    long long col = (long long)j * nx + i;
    double s = scale[col];
    double tk = sase_thick(t, dz, t_mode, k, ncol, col);
    double rho = (double)rho1[col];
    double fk = s * (double)f_row[(size_t)k * ncol + col];
    double fk1 = s * (double)f_row[(size_t)(k + 1) * ncol + col];
    double d = (fk - fk1) * dt / (rho * tk);
    size_t q = (size_t)k * ncol + col;
    phi[q] = (real)((double)phi[q] + d);
}

// ---------------------------------------------------------------------
// SPLIT SUBGRID-FLUX DIAGNOSTIC (output-only).
//
// The two kernels below RECORD the closure's own vertical subgrid
// fluxes as face-registered fields so the M2 vent channel and the K_v
// implicit-diffusion channel can be read separately off a history
// file.  They write ONLY their own `out` buffer: no prognostic, no
// workspace and no input array is touched, so the diagnostic cannot
// perturb the state (the driver's guard blocks sit AFTER the
// state-affecting launch on their line).
//
// SHARED CONVENTIONS, and they are load-bearing:
//   * Registration -- z FACE on the mass column, (nz+1, ny, nx),
//     identical to W; faces 0 and nz are the literal +0.0.
//   * Sign -- POSITIVE UPWARD for BOTH channels, which is why the
//     diffusion form below carries the leading minus against the
//     gradient.  This is the model's own vent convention: the deposit
//     kernel above forms dphi_k = (F[k] - F[k+1])*dt/(rho1*thick_k),
//     the flux-form convergence of an upward flux.
//   * Density -- BOTH channels divide by the SAME lowest-level moist
//     density plane rho1 the M2 deposit divides by (sase_vent_deposit
//     above; physics.sase_surface_rho1).  A true 3-D density would
//     make the two numbers unsummable and would break the identity
//       (F_vent[k]-F_vent[k+1] + F_diff[k]-F_diff[k+1])*dt/(rho1*t_k)
//         == phi_new[k] - phi_star[k]
//     which is the whole point of splitting them.
//   * Units -- carried by the caller's `fac`: 1.0 leaves the model's
//     own row units (kg m^-2 s^-1 on a moisture row), CP_AIR converts
//     a theta row [K kg m^-2 s^-1] to a heat flux [W m^-2] (the exact
//     inverse of the model's own theta-row convention -- the S3-11b
//     surface deposit is dt*HFX/(rho1*CP_AIR*thick_0), so HFX/CP_AIR
//     IS a theta-row flux).  No exner factor: the model does not use
//     one at this seam.
//   * Face 0 -- written +0.0 for BOTH channels even on the theta/qv
//     rows, whose true bottom-face flux is the S3-11b surface deposit
//     (HFX/CP_AIR, QFX).  That deposit is fused into the Thomas bottom
//     rhs and touches no interior face, so the interior recovery is
//     unaffected; the surface flux is already on disk as HFX/QFX and
//     is deliberately not double-counted here.

// M2 vent channel: the CAP-SCALED flux the deposit actually applied.
// sase_plume_vent_flux returns the UNSCALED profile and the deposit
// multiplies it in-kernel by the per-column cap factor `scale`, so a
// diagnostic recording the raw profile would record a flux the model
// did not apply.  The product is formed in the deposit's own op order
// (s * (double)F) with a single FP32 round at the store, so with
// fac == 1.0 the store is exactly the FP32 image of the FP64 value
// sase_vent_deposit consumed.
extern "C" __global__
void sase_vent_flux_diag(const real* __restrict__ f_row,
                         const double* __restrict__ scale,
                         real* __restrict__ out,
                         double fac, int nzp1, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nzp1) return;
    long long ncol = (long long)ny * nx;
    long long col = (long long)j * nx + i;
    size_t q = (size_t)k * ncol + col;
    out[q] = (real)(fac * (scale[col] * (double)f_row[q]));
}

// K_v channel: the implicit vertical diffusion flux RECOVERED from the
// POST-SOLVE field.  sase_thomas_scalar holds its face coefficients in
// per-thread registers and materializes no face flux, so there is
// nothing to copy -- but backward Euler evaluates the operator at the
// NEW state, so the post-solve field determines the flux exactly:
//
//     h[k]   = 0.5*(thick[k-1] + thick[k])
//     K_f[k] = kfac * 0.5*(kv[k-1] + kv[k])
//     F[k]   = -rho1 * K_f[k] * (phi[k] - phi[k-1]) / h[k]
//
// h and K_f are built in sase_thomas_scalar's verbatim op order (its
// r[k] = dt*kf/h with kf = kfac*(0.5*(kv[k]+kv[k+1])) and
// h = 0.5*(t_k + t_{k+1}) is this face at index k+1), so this is the
// flux the solver used, not a nearby number.  `phi` is the field AFTER
// launch_implicit_vertical_diffusion returns and BEFORE the driver
// converts it back to rate form.
extern "C" __global__
void sase_diff_flux_diag(const real* __restrict__ phi,
                         const real* __restrict__ kv,
                         const real* __restrict__ rho1,
                         real* __restrict__ out,
                         const real* __restrict__ t, real dz, int t_mode,
                         double kfac, double fac, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;                       // FACE index, 0 .. nz
    if (i >= nx || j >= ny || k > nz) return;
    long long ncol = (long long)ny * nx;
    long long col = (long long)j * nx + i;
    size_t q = (size_t)k * ncol + col;
    if (k == 0 || k == nz) {                  // interface contract
        out[q] = 0.0f;
        return;
    }
    size_t qm = q - (size_t)ncol;
    double tkm = sase_thick(t, dz, t_mode, k - 1, ncol, col);
    double tk = sase_thick(t, dz, t_mode, k, ncol, col);
    double h = 0.5 * (tkm + tk);
    double kf = kfac * (0.5 * ((double)kv[qm] + (double)kv[q]));
    double grad = ((double)phi[q] - (double)phi[qm]) / h;
    out[q] = (real)(fac * (-(double)rho1[col] * kf * grad));
}
