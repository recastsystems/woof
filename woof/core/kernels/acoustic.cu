// gpuwm/core/kernels/acoustic.cu
//
// Split-explicit acoustic substeps (ARW Tech Note sec. 3.1.2 eqns 3.7-3.14;
// WRF dyn_em/module_small_step_em.F subroutines advance_uv / advance_mu_t /
// calc_coef_w / advance_w / calc_p_rho).  General hybrid/terrain form of
// the WRF code (stage-moist cqu/cqv/cqw, exactly-one dry bypass, periodic x/y,
// WRF-default open top with selectable top_lid, no external-mode emdiv
// filter): every
// column-mass coupling carries the hybrid increments c1h[k]*mu + c2h[k]
// (half levels) / c1f[k]*mu + c2f[k] (full levels); the dry mass is the
// 2-D field mub2d + mup; base profiles are 1-D flat columns (base3d = 0)
// or per-column 3-D fields over terrain (base3d = 1).  For flat sigma runs
// (c1 = 1, c2 = 0, base3d = 0) everything reduces to the Phase 1 forms.
//
// Map-scale factors (Phase 3 Task 3): u_pp/v_pp/w_pp are the msf-coupled
// WRF small-step momenta ((c1h*mu+c2h)*u/msfu)'' etc. the coupling and
// uncoupling live in dycore._init_small_steps/_finish_small_steps.
// advance_uv is msf-free in the isotropic single-msf-per-staggering case
// gpuwm carries (the Fortran's msfux/msfuy and msfvy/msfvx pressure-
// gradient ratios are identically 1); advance_mu_th and advance_w_phi
// take msft/msfu/msfv + a has_msf flag and apply WRF advance_mu_t /
// advance_w's factors when it is set.  has_msf == 0 keeps the original
// expressions verbatim (bitwise Phase 2, regression-pinned).
//
// Perturbation ('' = deviation from the RK stage reference t*) fields:
//   u_pp, v_pp   coupled momenta ((c1h*mu+c2h)*u)'' at u/v points
//   mu_pp        column mass mu''
//   th_pp        coupled (mu*theta)''
//   ph_pp        geopotential phi'' at full levels
//   p_pp, al_pp  linearized-EOS pressure p'' and specific volume alpha'';
//                p_pp_old = previous substep (divergence damping: the
//                gradient uses p_pp + smdiv*(p_pp - p_pp_old)); al'' rides
//                the base-pressure gradient in advance_uv (terrain)
//   ww_pp        eta mass flux Omega'' at w levels (implicit solve input)

// Explicit round points for fusions of eager CuPy expressions.  The core
// acoustic algebra below intentionally remains on the repository's default
// contraction policy; these helpers are used only where a former global
// temporary separated two operators.
static __device__ __forceinline__ real arn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real arn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real arn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real arn_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Divergence-damped p'' (forward pressure weighting; smdiv = 0 on the
// first substep, when there is no history).
static __device__ __forceinline__
real pdmp(const real* __restrict__ p, const real* __restrict__ pold,
          size_t idx, real smdiv)
{
    return p[idx] + smdiv * (p[idx] - pold[idx]);
}

// WRF calc_cq (module_big_step_utilities_em.F:787-906): sum every active
// Registry ``moist`` mass species at the two cells surrounding a face.
// Morrison number moments are Registry ``scalar`` entries and do not enter.
static __device__ __forceinline__
real cq_pair(const real* __restrict__ qv, const real* __restrict__ qc,
             const real* __restrict__ qr, const real* __restrict__ qi,
             const real* __restrict__ qs, const real* __restrict__ qg,
             const real* __restrict__ qh,
             size_t a, size_t b, int n_mass)
{
    real qtot = 0.0f;
    qtot = arn_add(qtot, qv[a]);
    qtot = arn_add(qtot, qv[b]);
    if (n_mass >= 3) {
        qtot = arn_add(qtot, qc[a]);
        qtot = arn_add(qtot, qc[b]);
        qtot = arn_add(qtot, qr[a]);
        qtot = arn_add(qtot, qr[b]);
    }
    if (n_mass >= 6) {
        qtot = arn_add(qtot, qi[a]);
        qtot = arn_add(qtot, qi[b]);
        qtot = arn_add(qtot, qs[a]);
        qtot = arn_add(qtot, qs[b]);
        qtot = arn_add(qtot, qg[a]);
        qtot = arn_add(qtot, qg[b]);
    }
    if (n_mass == 7) {
        qtot = arn_add(qtot, qh[a]);
        qtot = arn_add(qtot, qh[b]);
    }
    return 1.0f / (1.0f + 0.5f * qtot);
}

// One RK-stage cq preparation.  Periodic duplicate u/v faces are filled;
// open/spec boundary-normal faces are later excluded by advance_uv.  cqw is
// already in pg_buoy_w's consumed form 1/(1+qtot_w), not calc_cq's temporary
// half-sum.  Its unused surface/top rows are overwritten with one.
extern "C" __global__
void calc_cq(const real* __restrict__ qv, const real* __restrict__ qc,
             const real* __restrict__ qr, const real* __restrict__ qi,
             const real* __restrict__ qs, const real* __restrict__ qg,
             const real* __restrict__ qh,
             real* __restrict__ cqu, real* __restrict__ cqv,
             real* __restrict__ cqw,
             int n_mass, int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nxf = nx + 1, nyf = ny + 1;
    size_t nu = (size_t)nz * ny * nxf;
    size_t nv = (size_t)nz * nyf * nx;
    size_t nw = (size_t)(nz + 1) * ny * nx;
    if ((size_t)tid < nu) {
        int k = tid / (ny * nxf);
        int rem = tid - k * ny * nxf;
        int j = rem / nxf, i = rem - j * nxf;
        int ia = i % nx, ib = (i - 1 + nx) % nx;
        size_t a = ((size_t)k * ny + j) * nx + ia;
        size_t b = ((size_t)k * ny + j) * nx + ib;
        cqu[tid] = cq_pair(qv, qc, qr, qi, qs, qg, qh, a, b, n_mass);
    }
    if ((size_t)tid < nv) {
        int k = tid / (nyf * nx);
        int rem = tid - k * nyf * nx;
        int j = rem / nx, i = rem - j * nx;
        int ja = j % ny, jb = (j - 1 + ny) % ny;
        size_t a = ((size_t)k * ny + ja) * nx + i;
        size_t b = ((size_t)k * ny + jb) * nx + i;
        cqv[tid] = cq_pair(qv, qc, qr, qi, qs, qg, qh, a, b, n_mass);
    }
    if ((size_t)tid < nw) {
        int k = tid / (ny * nx);
        int rem = tid - k * ny * nx;
        if (k == 0 || k == nz) {
            cqw[tid] = 1.0f;
        } else {
            size_t a = ((size_t)k * ny * nx) + rem;
            size_t b = ((size_t)(k - 1) * ny * nx) + rem;
            cqw[tid] = cq_pair(qv, qc, qr, qi, qs, qg, qh, a, b, n_mass);
        }
    }
}

// Damped p'' at full level kf, averaged across the two mass columns
// cA/cB of a face (WRF dpn).  Surface value extrapolated with cf1..cf3;
// zero at the model top (no lid).  st = level stride (ny*nx).
static __device__
real dpn_face(const real* __restrict__ p, const real* __restrict__ pold,
              real smdiv, int kf, int nz, size_t cA, size_t cB, size_t st,
              const real* __restrict__ fnm, const real* __restrict__ fnp,
              real cf1, real cf2, real cf3, int top_lid)
{
    if (kf == nz) {
        if (!top_lid) return 0.0f;
        return 0.5f * (
            cf1 * (pdmp(p, pold, cA + (size_t)(nz - 1) * st, smdiv)
                 + pdmp(p, pold, cB + (size_t)(nz - 1) * st, smdiv))
          + cf2 * (pdmp(p, pold, cA + (size_t)(nz - 2) * st, smdiv)
                 + pdmp(p, pold, cB + (size_t)(nz - 2) * st, smdiv))
          + cf3 * (pdmp(p, pold, cA + (size_t)(nz - 3) * st, smdiv)
                 + pdmp(p, pold, cB + (size_t)(nz - 3) * st, smdiv)));
    }
    if (kf == 0)
        return 0.5f * (cf1 * (pdmp(p, pold, cA,        smdiv)
                            + pdmp(p, pold, cB,        smdiv))
                     + cf2 * (pdmp(p, pold, cA + st,   smdiv)
                            + pdmp(p, pold, cB + st,   smdiv))
                     + cf3 * (pdmp(p, pold, cA + 2*st, smdiv)
                            + pdmp(p, pold, cB + 2*st, smdiv)));
    return 0.5f * (fnm[kf] * (pdmp(p, pold, cA + (size_t)kf * st, smdiv)
                            + pdmp(p, pold, cB + (size_t)kf * st, smdiv))
                 + fnp[kf] * (pdmp(p, pold, cA + (size_t)(kf-1) * st, smdiv)
                            + pdmp(p, pold, cB + (size_t)(kf-1) * st, smdiv)));
}

// Horizontal pressure-gradient term (WRF dpxy) at half level k on the face
// between mass columns cB (left/south) and cA (right/north), spacing 1/rd:
//   0.5*rd*(c1h*mu_face+c2h)*( d(phi'')[k]+d(phi'')[k+1]
//                              + (alt_A+alt_B)*d(pe)
//                              + (al''_A+al''_B)*d(pb) )   [terrain term]
// + rd*d(phi_FULL_half)*( rdnw[k]*(dpn[k+1]-dpn[k]) - c1h*mu''_face )
// Term 4's coefficient is the FULL t* half-level geopotential -- WRF's
// php argument is calc_php's 0.5*(phb(k)+phb(k+1)+ph(k)+ph(k+1))
// (module_big_step_utilities_em.F:1261, consumed at
// module_small_step_em.F:861-862/935-936) -- so the base phb joins the
// perturbation php over terrain (base3d); with a flat base phb is
// horizontally constant, its face difference is exactly zero, and the
// perturbation-only expression below is kept bitwise unchanged.
static __device__
real pgrad_face(size_t cA, size_t cB, int k, int nz, size_t st, real rd,
                const real* __restrict__ ph_pp, const real* __restrict__ php,
                const real* __restrict__ phb,
                const real* __restrict__ alt,
                const real* __restrict__ al_pp, const real* __restrict__ pb,
                const real* __restrict__ p_pp, const real* __restrict__ p_old,
                real smdiv,
                const real* __restrict__ mup, const real* __restrict__ mu_pp,
                const real* __restrict__ mub2d,
                const real* __restrict__ c1h, const real* __restrict__ c2h,
                const real* __restrict__ fnm, const real* __restrict__ fnp,
                const real* __restrict__ rdnw,
                real cf1, real cf2, real cf3, int base3d, int top_lid)
{
#if GPUWM_WRF_EXACT
    real muf = 0.5f * (mup[cA] + mup[cB] + mub2d[cA] + mub2d[cB]);
#else
    real muf = 0.5f * ((mub2d[cB] + mup[cB]) + (mub2d[cA] + mup[cA]));
#endif
    size_t a = cA + (size_t)k * st, b = cB + (size_t)k * st;
    real dph = (ph_pp[a + st] - ph_pp[b + st]) + (ph_pp[a] - ph_pp[b]);
    real dpe = pdmp(p_pp, p_old, a, smdiv) - pdmp(p_pp, p_old, b, smdiv);
    real dpb = base3d ? (pb[a] - pb[b]) : 0.0f;   // flat: d(pb)/dx == 0
    real dpxy = 0.5f * rd * (c1h[k] * muf + c2h[k])
              * (dph + (alt[a] + alt[b]) * dpe + (al_pp[a] + al_pp[b]) * dpb);
    real dpn_up = dpn_face(p_pp, p_old, smdiv, k + 1, nz, cA, cB, st,
                           fnm, fnp, cf1, cf2, cf3, top_lid);
    real dpn_dn = dpn_face(p_pp, p_old, smdiv, k,     nz, cA, cB, st,
                           fnm, fnp, cf1, cf2, cf3, top_lid);
#if GPUWM_WRF_EXACT
    size_t ba = base3d ? a : (size_t)k;
    size_t bb = base3d ? b : (size_t)k;
    size_t bst = base3d ? st : 1;
    real half_a = 0.5f * (phb[ba] + phb[ba + bst] + php[a] + php[a + st]);
    real half_b = 0.5f * (phb[bb] + phb[bb + bst] + php[b] + php[b + st]);
    real dphp = half_a - half_b;
    dpxy += rd * dphp * (rdnw[k] * (dpn_up - dpn_dn)
                         - 0.5f * ((c1h[k] * mu_pp[cB]) + (c1h[k] * mu_pp[cA])));
#else
    real dphp = 0.5f * ((php[a] + php[a + st]) - (php[b] + php[b + st]));
    if (base3d)                    // full geopotential: add the base part
        dphp += 0.5f * ((phb[a] + phb[a + st]) - (phb[b] + phb[b + st]));
    dpxy += rd * dphp * (rdnw[k] * (dpn_up - dpn_dn)
                         - 0.5f * c1h[k] * (mu_pp[cB] + mu_pp[cA]));
#endif
    return dpxy;
}

// Forward step of u''/v'': large-step tendency as constant forcing minus
// the damped horizontal pressure gradient.  One thread per (k, j, i) on
// the union of the u grid (j < ny) and v grid (i < nx); the periodic
// duplicate faces (u face nx, v row ny) are computed like any other face
// and stay consistent because all mass-point reads are index-wrapped.
extern "C" __global__
void advance_uv(real* __restrict__ u_pp, real* __restrict__ v_pp,
                const real* __restrict__ ru_t, const real* __restrict__ rv_t,
                const real* __restrict__ p_pp,
                const real* __restrict__ p_pp_old,
                const real* __restrict__ ph_pp, const real* __restrict__ php,
                const real* __restrict__ phb,
                const real* __restrict__ alt,
                const real* __restrict__ al_pp, const real* __restrict__ pb,
                const real* __restrict__ mup, const real* __restrict__ mu_pp,
                const real* __restrict__ mub2d,
                 const real* __restrict__ c1h, const real* __restrict__ c2h,
                 const real* __restrict__ fnm, const real* __restrict__ fnp,
                 const real* __restrict__ rdnw,
                 const real* __restrict__ cqu,
                 const real* __restrict__ cqv, int moist_cq,
                 real cf1, real cf2, real cf3,
                int top_lid,
                real rdx, real rdy, real dtau, real smdiv,
#if GPUWM_WRF_EXACT
                int spec_zone, int base3d, int nz, int ny, int nx,
                const real* __restrict__ mudf,
                const real* __restrict__ msfu, const real* __restrict__ msfv,
                real xscale, real yscale, int use_emdiv, int has_msf)
#else
                int spec_zone, int base3d, int nz, int ny, int nx)
#endif
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    size_t st = (size_t)ny * nx;

    if (j < ny
        && j >= spec_zone && j < ny - spec_zone
        && i >= spec_zone && i <= nx - spec_zone) {
                                                // u face i = 0..nx in row j
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int iA = i % nx, iB = (i - 1 + nx) % nx;
#else
        int iA = (i == nx ? 0 : i), iB = (i == 0 ? nx - 1 : i - 1);
#endif
        size_t cA = (size_t)j * nx + iA, cB = (size_t)j * nx + iB;
        real dpxy = pgrad_face(cA, cB, k, nz, st, rdx, ph_pp, php, phb, alt,
                               al_pp, pb, p_pp, p_pp_old, smdiv,
                               mup, mu_pp, mub2d, c1h, c2h,
                               fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                               top_lid);
        size_t uix = I3S(k, j, i, ny, nxf);
#if GPUWM_WRF_EXACT
        u_pp[uix] = u_pp[uix] + dtau * ru_t[uix];
        real cq = moist_cq ? cqu[uix] : 1.0f;
        u_pp[uix] = u_pp[uix] - dtau * cq * dpxy;
        if (use_emdiv) {
            // advance_uv: u = u - dts*cqu*dpxy + (c1h*mudf_xy), the
            // external-mode term last, with mudf_xy = -emdiv*dx*
            // (MUDF(i)-MUDF(i-1))/msfuy.  Applied before the substep (the
            // default) it rounds differently once MUDF is non-zero, from
            // the second RK stage on (combo-sweep round 3).
            real gx = xscale * (mudf[cA] - mudf[cB]);
            if (has_msf) gx = gx / msfu[(size_t)j * nxf + i];
            u_pp[uix] = u_pp[uix] + c1h[k] * gx;
        }
#else
        if (moist_cq)
            u_pp[uix] += dtau * (ru_t[uix] - cqu[uix] * dpxy);
        else
            u_pp[uix] += dtau * (ru_t[uix] - dpxy);
#endif
    } else if (j < ny) {
        size_t uix = I3S(k, j, i, ny, nxf);
        u_pp[uix] = arn_add(u_pp[uix], arn_mul(dtau, ru_t[uix]));
    }
    if (i < nx
        && i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j <= ny - spec_zone) {
                                                // v face j = 0..ny, column i
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int jA = j % ny, jB = (j - 1 + ny) % ny;
#else
        int jA = (j == ny ? 0 : j), jB = (j == 0 ? ny - 1 : j - 1);
#endif
        size_t cA = (size_t)jA * nx + i, cB = (size_t)jB * nx + i;
        real dpxy = pgrad_face(cA, cB, k, nz, st, rdy, ph_pp, php, phb, alt,
                               al_pp, pb, p_pp, p_pp_old, smdiv,
                               mup, mu_pp, mub2d, c1h, c2h,
                               fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                               top_lid);
        size_t vix = I3S(k, j, i, nyf, nx);
#if GPUWM_WRF_EXACT
        v_pp[vix] = v_pp[vix] + dtau * rv_t[vix];
        real cq = moist_cq ? cqv[vix] : 1.0f;
        v_pp[vix] = v_pp[vix] - dtau * cq * dpxy;
        if (use_emdiv) {
            // mudf_xy = -emdiv*dy*(MUDF(j)-MUDF(j-1))*msfvx_inv.
            real gy = yscale * (mudf[cA] - mudf[cB]);
            if (has_msf) gy = gy * msfv[(size_t)j * nx + i]; // strict argument is MF_VX_INV
            v_pp[vix] = v_pp[vix] + c1h[k] * gy;
        }
#else
        if (moist_cq)
            v_pp[vix] += dtau * (rv_t[vix] - cqv[vix] * dpxy);
        else
            v_pp[vix] += dtau * (rv_t[vix] - dpxy);
#endif
    } else if (i < nx) {
        size_t vix = I3S(k, j, i, nyf, nx);
        v_pp[vix] = arn_add(v_pp[vix], arn_mul(dtau, rv_t[vix]));
    }
}


#if GPUWM_WRF_EXACT
static __device__ __forceinline__
real wrf_total_divergence(int k, int j, int i, int ny, int nx,
                         const real* u_pp, const real* v_pp,
                         const real* u, const real* v,
                         const real* c1h, const real* c2h,
                         real muw, real mue, real mus, real mun,
                         real msfu_w, real msfu_e, real msfv_s, real msfv_n,
                         real msft_c, real rdx, real rdy)
{
    size_t uw = I3S(k,j,i,ny,nx+1), ue = uw+1;
    size_t vs = I3S(k,j,i,ny+1,nx), vn = vs+nx;
    real vs_full = v_pp[vs] + (c1h[k]*mus+c2h[k])*v[vs]*msfv_s;
    real vn_full = v_pp[vn] + (c1h[k]*mun+c2h[k])*v[vn]*msfv_n;
    real uw_full = u_pp[uw] + __fdiv_rn((c1h[k]*muw+c2h[k])*u[uw],msfu_w);
    real ue_full = u_pp[ue] + __fdiv_rn((c1h[k]*mue+c2h[k])*u[ue],msfu_e);
    return msft_c*msft_c*(rdy*(vn_full-vs_full)+rdx*(ue_full-uw_full));
}

static __device__
void wrf_advance_mu_theta(const real* u_pp, const real* v_pp,
                          const real* u, const real* v, const real* mup,
                          real* mu_pp, real* mu_old, const real* rmu_t,
                          real* mudf, int write_mudf, const real* theta,
                          real* th_pp, real* th_old, const real* rth_t,
                          real* ww_pp, const real* ww_ref,
                          const real* p_pp, real* p_old,
                          const real* dnw, const real* rdnw,
                          const real* fnm, const real* fnp,
                          const real* c1h, const real* c2h, const real* mub2d,
                          const real* msft, const real* msfu, const real* msfv,
                          int mapped, real rdx, real rdy, real dtau,
                          int nz, int ny, int nx, int open_x, int open_y,
                          int spec_zone)
{
    int c = blockIdx.x*blockDim.x+threadIdx.x;
    if (c >= ny*nx) return;
    int j=c/nx, i=c-j*nx;
    size_t st=(size_t)ny*nx;
    mu_old[c]=mu_pp[c];
    for (int k=0;k<nz;++k) {
        size_t h=(size_t)k*st+c;
        th_old[h]=th_pp[h]; p_old[h]=p_pp[h];
    }
    if (i<spec_zone || i>=nx-spec_zone || j<spec_zone || j>=ny-spec_zone) {
        if (write_mudf) mudf[c]=0.0f;
        return;
    }
    int ip=open_x?min(i+1,nx-1):(i+1)%nx;
    int im=open_x?max(i-1,0):(i-1+nx)%nx;
    int jp=open_y?min(j+1,ny-1):(j+1)%ny;
    int jm=open_y?max(j-1,0):(j-1+ny)%ny;
    size_t cw=(size_t)j*nx+im, ce=(size_t)j*nx+ip;
    size_t cs=(size_t)jm*nx+i, cn=(size_t)jp*nx+i;
    real muw=0.5f*(mup[c]+mup[cw]+mub2d[c]+mub2d[cw]);
    real mue=0.5f*(mup[ce]+mup[c]+mub2d[ce]+mub2d[c]);
    real mus=0.5f*(mup[c]+mup[cs]+mub2d[c]+mub2d[cs]);
    real mun=0.5f*(mup[cn]+mup[c]+mub2d[cn]+mub2d[c]);
    real mc=mapped?msft[c]:1.0f;
    real uxw=mapped?msfu[(size_t)j*(nx+1)+i]:1.0f;
    real uxe=mapped?msfu[(size_t)j*(nx+1)+i+1]:1.0f;
    real vys=mapped?msfv[(size_t)j*nx+i]:1.0f;
    real vyn=mapped?msfv[(size_t)(j+1)*nx+i]:1.0f;
    real dmdt=0.0f;
    for (int k=0;k<nz;++k) {
        real dvdxi=wrf_total_divergence(k,j,i,ny,nx,u_pp,v_pp,u,v,c1h,c2h,
                              muw,mue,mus,mun,uxw,uxe,vys,vyn,mc,rdx,rdy);
        dmdt=dmdt+dnw[k]*dvdxi;
    }
    mu_pp[c]=mu_pp[c]+dtau*(dmdt+rmu_t[c]);
    if (write_mudf) mudf[c]=dmdt+rmu_t[c];
    real total_ww=0.0f;
    ww_pp[c]=total_ww-ww_ref[c];
    for (int kk=1;kk<nz;++kk) {
        int k=kk-1;
        real dvdxi=wrf_total_divergence(k,j,i,ny,nx,u_pp,v_pp,u,v,c1h,c2h,
                              muw,mue,mus,mun,uxw,uxe,vys,vyn,mc,rdx,rdy);
        total_ww=total_ww-__fdiv_rn(dnw[k]*(c1h[k]*dmdt+dvdxi+c1h[k]*rmu_t[c]),mc);
        ww_pp[(size_t)kk*st+c]=total_ww-ww_ref[(size_t)kk*st+c];
    }
    ww_pp[(size_t)nz*st+c]=0.0f;
    real wdtn=0.0f;
    for (int k=0;k<nz;++k) {
        size_t h=(size_t)k*st+c;
        real wdtn_up=0.0f;
        if (k+1<nz) wdtn_up=ww_pp[h+st]*(fnm[k+1]*theta[h+st]+fnp[k+1]*theta[h]);
        real hy=0.5f*rdy*(v_pp[I3S(k,j+1,i,ny+1,nx)]*(theta[IDX3(k,jp,i)]+theta[h])
                         -v_pp[I3S(k,j,i,ny+1,nx)]*(theta[h]+theta[IDX3(k,jm,i)]));
        real hx=0.5f*rdx*(u_pp[I3S(k,j,i+1,ny,nx+1)]*(theta[IDX3(k,j,ip)]+theta[h])
                         -u_pp[I3S(k,j,i,ny,nx+1)]*(theta[h]+theta[IDX3(k,j,im)]));
        th_pp[h]=th_pp[h]+mc*dtau*rth_t[h];
        th_pp[h]=th_pp[h]-dtau*mc*(mc*(hy+hx)+rdnw[k]*(wdtn_up-wdtn));
        wdtn=wdtn_up;
    }
}

// solve_em applies these specified tendencies after advance_mu_t and before
// advance_w.  Keeping the driver operation separate preserves signed zeros
// in the acoustic routine's untouched frame.
extern "C" __global__
void advance_exact_frame_mu_t(real* __restrict__ mu_pp,
                              real* __restrict__ th_pp,
                              const real* __restrict__ rmu_t,
                              const real* __restrict__ rth_t,
                              real dtau, int spec_zone, int periodic_x,
                              int nz, int ny, int nx,
                              real* __restrict__ muts,
                              const real* __restrict__ mub2d,
                              const real* __restrict__ mup)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=ny*nx) return;
    int j=c/nx, i=c-j*nx;
    bool frame=j<spec_zone || j>=ny-spec_zone
                || (!periodic_x && (i<spec_zone || i>=nx-spec_zone));
    // WRF's grid%muts: advance_mu_t sets MUTS = MUT + MU'' where it
    // integrates (module_small_step_em.F); on the specified ring solve_em
    // instead advances it like mu_2, spec_bdyupdate(muts, mu_tend, dts),
    // from small_step_prep's MUB+MU_2 (rk_step 1) or MUB+MU_1 (later).
    // The two agree on RK stage 1 only.
    if (!frame) {
        muts[c]=(mub2d[c]+mup[c])+mu_pp[c];
        return;
    }
    muts[c]=muts[c]+dtau*rmu_t[c];
    mu_pp[c]=mu_pp[c]+dtau*rmu_t[c];
    size_t st=(size_t)ny*nx;
    for (int k=0;k<nz;++k) {
        size_t h=(size_t)k*st+c;
        th_pp[h]=th_pp[h]+dtau*rth_t[h];
    }
}
#endif

// Forward steps of mu'' and (mu*theta)'' plus the Omega'' diagnosis (ARW
// eqns 3.9-3.10), from the just-updated u''/v''.  One thread per column:
//  - mu'' integrates the column divergence of the *total* coupled momentum
//    (perturbation + reference (c1h*mu_face+c2h)*u_t*, the fixed
//    large-step part of R_mu) plus the rmu_t forcing;
//  - Omega'' (ww_pp) integrates the perturbation-only divergence upward
//    from Omega''(surface) = 0 with the c1h-weighted column-mass tendency
//    (WRF ww(k) = ww(k-1) - dnw(k-1)*(c1h(k-1)*(dmdt+mu_tend) + dvdxi)),
//    zero at the top by construction;
//  - (mu*theta)'' adds rth_t and the divergence of the perturbation
//    momenta advecting the reference theta_t* = thb + thp;
//  - the pre-update mu''/(mu*theta)'' and the post-advance_uv p'' are saved
//    to their history buffers while this kernel already traverses them.
extern "C" __global__
void advance_mu_th(const real* __restrict__ u_pp,
                   const real* __restrict__ v_pp,
                   const real* __restrict__ u, const real* __restrict__ v,
                   const real* __restrict__ mup,
                   real* __restrict__ mu_pp,
                   real* __restrict__ mu_pp_old,
                   const real* __restrict__ rmu_t,
                   real* __restrict__ mudf, int write_mudf,
                   const real* __restrict__ thp,
                   const real* __restrict__ thb,
                   real* __restrict__ th_pp,
                   real* __restrict__ th_pp_old,
                   const real* __restrict__ rth_t,
                   real* __restrict__ ww_pp,
#if GPUWM_WRF_EXACT
                   const real* __restrict__ ww_ref,
#endif
                   const real* __restrict__ p_pp,
                   real* __restrict__ p_pp_old,
                   const real* __restrict__ dnw,
                   const real* __restrict__ rdnw,
                   const real* __restrict__ fnm,
                   const real* __restrict__ fnp,
                   const real* __restrict__ c1h, const real* __restrict__ c2h,
                   const real* __restrict__ mub2d,
                   real rdx, real rdy, real dtau,
                   int base3d, int nz, int ny, int nx,
                   int open_x, int open_y, int spec_zone)
{
#if GPUWM_WRF_EXACT
    wrf_advance_mu_theta(u_pp,v_pp,u,v,mup,mu_pp,mu_pp_old,rmu_t,
                         mudf,write_mudf,thp,th_pp,th_pp_old,rth_t,
                         ww_pp,ww_ref,p_pp,p_pp_old,dnw,rdnw,fnm,fnp,
                         c1h,c2h,mub2d,mub2d, mub2d, mub2d,0,rdx,rdy,dtau,
                         nz,ny,nx,open_x,open_y,spec_zone);
#else
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    mu_pp_old[c] = mu_pp[c];
    for (int k = 0; k < nz; ++k) {
        size_t ci = (size_t)k * st + c;
        th_pp_old[ci] = th_pp[ci];
        p_pp_old[ci] = p_pp[ci];
    }
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) {
        real rmu = rmu_t[c];
        if (write_mudf) mudf[c] = 0.0f;
        mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, rmu));
        for (int k = 0; k < nz; ++k) {
            size_t ci = (size_t)k * st + c;
            real mt0 = arn_mul(300.0f, c1h[k]);
            real mt1 = arn_mul(mt0, rmu);
            real full_tendency = arn_add(rth_t[ci], mt1);
            th_pp[ci] = arn_add(th_pp[ci], arn_mul(dtau, full_tendency));
        }
        return;
    }
    // Cross-boundary neighbours: periodic wrap, or WRF's zero-gradient
    // ghost (clamp) at open lateral boundaries -- the boundary-face
    // reference mass becomes the boundary cell's own and the theta_t*
    // advection sees its own value across the boundary face.
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
    int nxf = nx + 1;
    size_t bstr = base3d ? st : 1;             // base-profile level stride

    // Reference column mass at the four faces (fixed over the substeps);
    // coupled per level below with c1h[k]*mu + c2h[k].
    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    real muw = 0.5f * ((mub2d[cw] + mup[cw]) + (mub2d[c] + mup[c]));
    real mue = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[ce] + mup[ce]));
    real mus = 0.5f * ((mub2d[cs] + mup[cs]) + (mub2d[c] + mup[c]));
    real mun = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[cn] + mup[cn]));

    real dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        real dvr = rdx * ((c1h[k] * mue + c2h[k]) * u[ue]
                        - (c1h[k] * muw + c2h[k]) * u[uw])
                 + rdy * ((c1h[k] * mun + c2h[k]) * v[vn]
                        - (c1h[k] * mus + c2h[k]) * v[vs]);
        dmdt_pp += dnw[k] * dvp;
        dmdt_rf += dnw[k] * dvr;
    }
    real rmu = rmu_t[c];
    real mass_tendency = arn_add(arn_add(dmdt_pp, dmdt_rf), rmu);
    if (write_mudf) mudf[c] = mass_tendency;
    mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, mass_tendency));

    ww_pp[c] = 0.0f;
    real wwk = 0.0f;
    for (int kk = 1; kk < nz; ++kk) {
        int k = kk - 1;
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        wwk -= dnw[k] * (c1h[k] * (dmdt_pp + rmu) + dvp);
        ww_pp[(size_t)kk * st + c] = wwk;
    }
    ww_pp[(size_t)nz * st + c] = 0.0f;

    real wdtn_k = 0.0f;                        // vertical theta flux, level k
    for (int k = 0; k < nz; ++k) {
        real wdtn_kp1 = 0.0f;
        if (k + 1 < nz) {
            real t_up = thb[(size_t)(k + 1) * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)(k + 1) * st + c];
            real t_dn = thb[(size_t)k * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)k * st + c];
            wdtn_kp1 = ww_pp[(size_t)(k + 1) * st + c]
                     * (fnm[k + 1] * t_up + fnp[k + 1] * t_dn);
        }
        size_t bk = (size_t)k * bstr;
        real t_c  = thb[bk + (base3d ? (size_t)c : 0)]
                  + thp[(size_t)k * st + c];
        real t_ip = thb[bk + (base3d ? ce : 0)] + thp[IDX3(k, j, ip)];
        real t_im = thb[bk + (base3d ? cw : 0)] + thp[IDX3(k, j, im)];
        real t_jp = thb[bk + (base3d ? cn : 0)] + thp[IDX3(k, jp, i)];
        real t_jm = thb[bk + (base3d ? cs : 0)] + thp[IDX3(k, jm, i)];
        real hx = 0.5f * rdx * (u_pp[I3S(k, j, i + 1, ny, nxf)] * (t_ip + t_c)
                              - u_pp[I3S(k, j, i,     ny, nxf)] * (t_c + t_im));
        real hy = 0.5f * rdy * (v_pp[I3S(k, j + 1, i, ny + 1, nx)] * (t_jp + t_c)
                              - v_pp[I3S(k, j,     i, ny + 1, nx)] * (t_c + t_jm));
        size_t ci = (size_t)k * st + c;
        th_pp[ci] += dtau * (rth_t[ci]
                             - (hx + hy + rdnw[k] * (wdtn_kp1 - wdtn_k)));
        wdtn_k = wdtn_kp1;
    }
#endif
}

// Map-scale-factor variant of advance_mu_th (Phase 3 Task 3, WRF
// advance_mu_t with map factors; launched instead of advance_mu_th when
// any msf != 1; the arithmetic update in the msf==1 kernel above remains
// unchanged apart from the history stores):
//  - the layer divergence dvdxi carries msftx*msfty = msft^2 and the
//    REFERENCE coupled momenta divide by their face msf (u_pp/v_pp
//    already carry theirs from small_step_prep);
//  - the Omega'' recurrence divides by msfty = msft;
//  - the theta'' update multiplies the whole RHS by msfty and the
//    horizontal advection by msftx (WRF: t += msfty*dts*ft then
//    t -= dts*msfty*(msftx*(hx+hy) + rdnw*d(wdtn))).
extern "C" __global__
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#else
__launch_bounds__(256, 4)
#endif
void advance_mu_th_msf(const real* __restrict__ u_pp,
                       const real* __restrict__ v_pp,
                       const real* __restrict__ u, const real* __restrict__ v,
                       const real* __restrict__ mup,
                       real* __restrict__ mu_pp,
                       real* __restrict__ mu_pp_old,
                       const real* __restrict__ rmu_t,
                       real* __restrict__ mudf, int write_mudf,
                       const real* __restrict__ thp,
                       const real* __restrict__ thb,
                       real* __restrict__ th_pp,
                       real* __restrict__ th_pp_old,
                       const real* __restrict__ rth_t,
                       real* __restrict__ ww_pp,
#if GPUWM_WRF_EXACT
                   const real* __restrict__ ww_ref,
#endif
                       const real* __restrict__ p_pp,
                       real* __restrict__ p_pp_old,
                       const real* __restrict__ dnw,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ fnm,
                       const real* __restrict__ fnp,
                       const real* __restrict__ c1h,
                       const real* __restrict__ c2h,
                       const real* __restrict__ mub2d,
                       const real* __restrict__ msft,   // (ny, nx)
                       const real* __restrict__ msfu,   // (ny, nx+1)
                       const real* __restrict__ msfv,   // (ny+1, nx)
                       real rdx, real rdy, real dtau,
                       int base3d, int nz, int ny, int nx,
                       int open_x, int open_y, int spec_zone)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#if GPUWM_WRF_EXACT
    wrf_advance_mu_theta(u_pp,v_pp,u,v,mup,mu_pp,mu_pp_old,rmu_t,
                         mudf,write_mudf,thp,th_pp,th_pp_old,rth_t,
                         ww_pp,ww_ref,p_pp,p_pp_old,dnw,rdnw,fnm,fnp,
                         c1h,c2h,mub2d,msft, msfu, msfv,1,rdx,rdy,dtau,
                         nz,ny,nx,open_x,open_y,spec_zone);
#else
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    mu_pp_old[c] = mu_pp[c];
    for (int k = 0; k < nz; ++k) {
        size_t ci = (size_t)k * st + c;
        th_pp_old[ci] = th_pp[ci];
        p_pp_old[ci] = p_pp[ci];
    }
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) {
        real rmu = rmu_t[c];
        if (write_mudf) mudf[c] = 0.0f;
        mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, rmu));
        for (int k = 0; k < nz; ++k) {
            size_t ci = (size_t)k * st + c;
            real mt0 = arn_mul(300.0f, c1h[k]);
            real mt1 = arn_mul(mt0, rmu);
            real full_tendency = arn_add(rth_t[ci], mt1);
            th_pp[ci] = arn_add(th_pp[ci], arn_mul(dtau, full_tendency));
        }
        return;
    }
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1) % nx;
    int im = open_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1) % ny;
    int jm = open_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
    int nxf = nx + 1;
    size_t bstr = base3d ? st : 1;

    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    real muw = 0.5f * ((mub2d[cw] + mup[cw]) + (mub2d[c] + mup[c]));
    real mue = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[ce] + mup[ce]));
    real mus = 0.5f * ((mub2d[cs] + mup[cs]) + (mub2d[c] + mup[c]));
    real mun = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[cn] + mup[cn]));
    real m2 = msft[c] * msft[c];
    real msfu_w = msfu[(size_t)j * nxf + i];
    real msfu_e = msfu[(size_t)j * nxf + i + 1];
    real msfv_s = msfv[(size_t)j * nx + i];
    real msfv_n = msfv[(size_t)(j + 1) * nx + i];

    real dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        real dvr = rdx * ((c1h[k] * mue + c2h[k]) * u[ue] / msfu_e
                        - (c1h[k] * muw + c2h[k]) * u[uw] / msfu_w)
                 + rdy * ((c1h[k] * mun + c2h[k]) * v[vn] / msfv_n
                        - (c1h[k] * mus + c2h[k]) * v[vs] / msfv_s);
        dmdt_pp += dnw[k] * (m2 * dvp);
        dmdt_rf += dnw[k] * (m2 * dvr);
    }
    real rmu = rmu_t[c];
    real mass_tendency = arn_add(arn_add(dmdt_pp, dmdt_rf), rmu);
    if (write_mudf) mudf[c] = mass_tendency;
    mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, mass_tendency));

    ww_pp[c] = 0.0f;
    real wwk = 0.0f;
    for (int kk = 1; kk < nz; ++kk) {
        int k = kk - 1;
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        wwk -= dnw[k] * (c1h[k] * (dmdt_pp + rmu) + m2 * dvp) / msft[c];
        ww_pp[(size_t)kk * st + c] = wwk;
    }
    ww_pp[(size_t)nz * st + c] = 0.0f;

    real wdtn_k = 0.0f;
    for (int k = 0; k < nz; ++k) {
        real wdtn_kp1 = 0.0f;
        if (k + 1 < nz) {
            real t_up = thb[(size_t)(k + 1) * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)(k + 1) * st + c];
            real t_dn = thb[(size_t)k * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)k * st + c];
            wdtn_kp1 = ww_pp[(size_t)(k + 1) * st + c]
                     * (fnm[k + 1] * t_up + fnp[k + 1] * t_dn);
        }
        size_t bk = (size_t)k * bstr;
        real t_c  = thb[bk + (base3d ? (size_t)c : 0)]
                  + thp[(size_t)k * st + c];
        real t_ip = thb[bk + (base3d ? ce : 0)] + thp[IDX3(k, j, ip)];
        real t_im = thb[bk + (base3d ? cw : 0)] + thp[IDX3(k, j, im)];
        real t_jp = thb[bk + (base3d ? cn : 0)] + thp[IDX3(k, jp, i)];
        real t_jm = thb[bk + (base3d ? cs : 0)] + thp[IDX3(k, jm, i)];
        real hx = 0.5f * rdx * (u_pp[I3S(k, j, i + 1, ny, nxf)] * (t_ip + t_c)
                              - u_pp[I3S(k, j, i,     ny, nxf)] * (t_c + t_im));
        real hy = 0.5f * rdy * (v_pp[I3S(k, j + 1, i, ny + 1, nx)] * (t_jp + t_c)
                              - v_pp[I3S(k, j,     i, ny + 1, nx)] * (t_c + t_jm));
        size_t ci = (size_t)k * st + c;
        th_pp[ci] += dtau * msft[c]
                   * (rth_t[ci]
                      - (msft[c] * (hx + hy)
                         + rdnw[k] * (wdtn_kp1 - wdtn_k)));
        wdtn_k = wdtn_kp1;
    }
#endif
#else
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    size_t st = (size_t)ny * nx;
    mu_pp_old[c] = mu_pp[c];
    for (int k = 0; k < nz; ++k) {
        size_t ci = (size_t)k * st + c;
        th_pp_old[ci] = th_pp[ci];
        p_pp_old[ci] = p_pp[ci];
    }
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) {
        real rmu = rmu_t[c];
        if (write_mudf) mudf[c] = 0.0f;
        mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, rmu));
        for (int k = 0; k < nz; ++k) {
            size_t ci = (size_t)k * st + c;
            real mt0 = arn_mul(300.0f, c1h[k]);
            real mt1 = arn_mul(mt0, rmu);
            real full_tendency = arn_add(rth_t[ci], mt1);
            th_pp[ci] = arn_add(th_pp[ci], arn_mul(dtau, full_tendency));
        }
        return;
    }
    int ip = open_x ? min(i + 1, nx - 1) : (i + 1 == nx ? 0 : i + 1);
    int im = open_x ? max(i - 1, 0)      : (i == 0 ? nx - 1 : i - 1);
    int jp = open_y ? min(j + 1, ny - 1) : (j + 1 == ny ? 0 : j + 1);
    int jm = open_y ? max(j - 1, 0)      : (j == 0 ? ny - 1 : j - 1);
    int nxf = nx + 1;
    size_t bstr = base3d ? st : 1;

    size_t cw = (size_t)j * nx + im, ce = (size_t)j * nx + ip;
    size_t cs = (size_t)jm * nx + i, cn = (size_t)jp * nx + i;
    real muw = 0.5f * ((mub2d[cw] + mup[cw]) + (mub2d[c] + mup[c]));
    real mue = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[ce] + mup[ce]));
    real mus = 0.5f * ((mub2d[cs] + mup[cs]) + (mub2d[c] + mup[c]));
    real mun = 0.5f * ((mub2d[c] + mup[c]) + (mub2d[cn] + mup[cn]));
    real m2 = msft[c] * msft[c];
    real msfu_w = msfu[(size_t)j * nxf + i];
    real msfu_e = msfu[(size_t)j * nxf + i + 1];
    real msfv_s = msfv[(size_t)j * nx + i];
    real msfv_n = msfv[(size_t)(j + 1) * nx + i];

    real dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real dvp = rdx * (u_pp[ue] - u_pp[uw]) + rdy * (v_pp[vn] - v_pp[vs]);
        real dvr = rdx * (__fdiv_rn((c1h[k] * mue + c2h[k]) * u[ue], msfu_e)
                        - __fdiv_rn((c1h[k] * muw + c2h[k]) * u[uw], msfu_w))
                 + rdy * (__fdiv_rn((c1h[k] * mun + c2h[k]) * v[vn], msfv_n)
                        - __fdiv_rn((c1h[k] * mus + c2h[k]) * v[vs], msfv_s));
        dmdt_pp += dnw[k] * (m2 * dvp);
        dmdt_rf += dnw[k] * (m2 * dvr);
    }
    real rmu = rmu_t[c];
    real mass_tendency = arn_add(arn_add(dmdt_pp, dmdt_rf), rmu);
    if (write_mudf) mudf[c] = mass_tendency;
    mu_pp[c] = arn_add(mu_pp[c], arn_mul(dtau, mass_tendency));

    // The next Omega row is consumed immediately by this theta row.
    // Both vertical recurrences retain their original bottom-up order.
    ww_pp[c] = 0.0f;
    real wwk = 0.0f;
    real wdtn_k = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t uw = I3S(k, j, i, ny, nxf), ue = I3S(k, j, i + 1, ny, nxf);
        size_t vs = I3S(k, j, i, ny + 1, nx), vn = I3S(k, j + 1, i, ny + 1, nx);
        real u_w = u_pp[uw], u_e = u_pp[ue];
        real v_s = v_pp[vs], v_n = v_pp[vn];
        real wdtn_kp1 = 0.0f;
        if (k + 1 < nz) {
            real dvp = rdx * (u_e - u_w) + rdy * (v_n - v_s);
            // Preserve the unfused column loop's contracted mass term.
            real mass_div = __fmaf_rn(c1h[k], dmdt_pp + rmu, arn_mul(m2, dvp));
            wwk -= __fdiv_rn(dnw[k] * mass_div, msft[c]);
            ww_pp[(size_t)(k + 1) * st + c] = wwk;
            real t_up = thb[(size_t)(k + 1) * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)(k + 1) * st + c];
            real t_dn = thb[(size_t)k * bstr + (base3d ? (size_t)c : 0)]
                      + thp[(size_t)k * st + c];
            wdtn_kp1 = wwk * (fnm[k + 1] * t_up + fnp[k + 1] * t_dn);
        }
        size_t bk = (size_t)k * bstr;
        real t_c  = thb[bk + (base3d ? (size_t)c : 0)]
                  + thp[(size_t)k * st + c];
        real t_ip = thb[bk + (base3d ? ce : 0)] + thp[IDX3(k, j, ip)];
        real t_im = thb[bk + (base3d ? cw : 0)] + thp[IDX3(k, j, im)];
        real t_jp = thb[bk + (base3d ? cn : 0)] + thp[IDX3(k, jp, i)];
        real t_jm = thb[bk + (base3d ? cs : 0)] + thp[IDX3(k, jm, i)];
        real hx = 0.5f * rdx * (u_e * (t_ip + t_c)
                              - u_w * (t_c + t_im));
        real hy = 0.5f * rdy * (v_n * (t_jp + t_c)
                              - v_s * (t_c + t_jm));
        size_t ci = (size_t)k * st + c;
        th_pp[ci] += dtau * msft[c]
                   * (rth_t[ci]
                      - (msft[c] * (hx + hy)
                         + rdnw[k] * (wdtn_kp1 - wdtn_k)));
        wdtn_k = wdtn_kp1;
    }
    ww_pp[(size_t)nz * st + c] = 0.0f;
#endif
}

// ---- Vertically implicit w''-phi'' solve (ARW eqns 3.11-3.14; WRF
// calc_coef_w / advance_w / calc_p_rho, general hybrid/terrain form:
// kinematic terrain lower BC, configurable WRF top_lid, moist cqw). ----------

/* Max full levels for the in-thread solve.  This bound sizes exactly ONE
 * per-thread object in this translation unit -- the RHS column
 * `real rhs[WPHI_MAX_LEV]` that advance_w_phi and advance_w_phi_msf each
 * declare -- so it costs 4 B of local frame per admitted full level and
 * nothing else.  No expression in either kernel READS it: every loop runs
 * to the runtime nz and the highest index any of them forms is nz, so
 * raising it moves no arithmetic.  The a/alpha/gamma coefficient columns
 * are global scratch (gpuwm/core/acoustic.py), not in-thread.
 *
 * The launcher selects it from a small tier ladder by nz through
 * gpuwm.core.kernels.get_kernel_int_defines, exactly the way kf.cu's
 * KF_KMAX is selected.  129 is the tier this source keeps when nothing
 * overrides it, and every nz <= 128 configuration compiles the
 * UNSPECIALIZED module -- no define is injected, so the string handed to
 * NVRTC is byte-identical to what it was before the ladder existed.
 * gpuwm/core/acoustic.py owns the ladder; tests/test_acoustic_nz_tiers.py
 * pins both halves of that claim.
 */
#ifndef WPHI_MAX_LEV
#define WPHI_MAX_LEV 129
#endif

// Linearized-EOS p''/alpha'' diagnosis (WRF calc_p_rho, non-hydrostatic
// branch; with full theta the WRF t0-offset terms cancel):
//   alpha'' = -(alpha_t*'(c1h*mu'') + d(phi'')/d(eta)) / (c1h*mu_ts+c2h)
//   p''     = c2a*(alpha_t*'((mu*theta)'' - c1h*mu''*theta_t*)
//                  / ((c1h*mu_ts+c2h)*theta_t*) - alpha'')
// The owning solve/frame thread diagnoses its complete column after updating
// phi''.  alpha'' is stored for the next substep's advance_uv terrain term.
static __device__ __forceinline__
void diagnose_p_column(real* __restrict__ p_pp,
                       real* __restrict__ al_pp,
                       const real* __restrict__ th_pp,
                       const real* __restrict__ ph_pp,
                       const real* __restrict__ mu_pp,
                       const real* __restrict__ thp,
                       const real* __restrict__ thb,
                       const real* __restrict__ alt,
                       const real* __restrict__ c2a,
                       const real* __restrict__ mup,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ c1h,
                       const real* __restrict__ c2h,
                       const real* __restrict__ mub2d,
                       size_t c, size_t st, int base3d, int nz)
{
    for (int k = 0; k < nz; ++k) {
        size_t tid = (size_t)k * st + c;
        real chm = c1h[k] * (mub2d[c] + mup[c] + mu_pp[c]) + c2h[k];
#if GPUWM_WRF_EXACT
        real th_ref = thp[tid];
        real al = __fdiv_rn(-1.0f, chm)
                * (alt[tid] * (c1h[k] * mu_pp[c])
                   + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid]));
        al_pp[tid] = al;
        p_pp[tid] = c2a[tid]
                  * (__fdiv_rn(alt[tid]
                      * (th_pp[tid] - (c1h[k] * mu_pp[c]) * th_ref),
                      chm * (300.0f + th_ref)) - al);
#else
        real th_ref = thb[base3d ? tid : (size_t)k] + thp[tid];
        real al = -(alt[tid] * (c1h[k] * mu_pp[c])
                    + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid])) / chm;
        al_pp[tid] = al;
        p_pp[tid] = c2a[tid]
                  * (alt[tid]
                     * (th_pp[tid] - c1h[k] * mu_pp[c] * th_ref)
                     / (chm * th_ref) - al);
#endif
    }
}

#if GPUWM_WRF_EXACT
// diagnose_p_column with WRF's grid%muts for the column supplied: the
// specified ring's muts is the spec_bdyupdate carrier, not (mub+mu)+mu''.
static __device__ __forceinline__
void diagnose_p_column_muts(real* __restrict__ p_pp,
                            real* __restrict__ al_pp,
                            const real* __restrict__ th_pp,
                            const real* __restrict__ ph_pp,
                            const real* __restrict__ mu_pp,
                            const real* __restrict__ thp,
                            const real* __restrict__ alt,
                            const real* __restrict__ c2a,
                            const real* __restrict__ rdnw,
                            const real* __restrict__ c1h,
                            const real* __restrict__ c2h,
                            size_t c, size_t st, int nz, real muts)
{
    for (int k = 0; k < nz; ++k) {
        size_t tid = (size_t)k * st + c;
        real chm = c1h[k] * muts + c2h[k];
        real th_ref = thp[tid];
        real al = __fdiv_rn(-1.0f, chm)
                * (alt[tid] * (c1h[k] * mu_pp[c])
                   + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid]));
        al_pp[tid] = al;
        p_pp[tid] = c2a[tid]
                  * (__fdiv_rn(alt[tid]
                      * (th_pp[tid] - (c1h[k] * mu_pp[c]) * th_ref),
                      chm * (300.0f + th_ref)) - al);
    }
}

#endif
// Sound-speed factor c2a = gamma*p/alpha and the LU-factored tridiagonal
// coefficients a/alpha/gam of the implicit w'' system, all from the fixed
// t* state (WRF calc_coef_w; lid_flag = !top_lid).  One thread per column;
// a/alpha/gam live at w levels (row k couples w[k-1], w[k], w[k+1]); each
// c2a is divided by its own half-level (c1h*mut + c2h) and the
// (c1f*mut + c2f) of the full level whose phi'' update carries the
// coupled w (WRF indexing: a row kk uses levels kk-1, the diagonal row k
// uses cfm(k), the upper coefficient cfm(k+1)).
extern "C" __global__
void calc_coefs(const real* __restrict__ p, const real* __restrict__ alt,
                const real* __restrict__ mup,
                real* __restrict__ c2a, real* __restrict__ a,
                real* __restrict__ alpha, real* __restrict__ gam,
                const real* __restrict__ rdn, const real* __restrict__ rdnw,
                const real* __restrict__ c1h, const real* __restrict__ c2h,
                const real* __restrict__ c1f, const real* __restrict__ c2f,
                const real* __restrict__ mub2d,
                const real* __restrict__ cqw, int moist_cq,
                real dtau, real epssm, int top_lid,
                int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    real mut = mub2d[c] + mup[c];
    real cof = 0.5f * dtau * G * (1.0f + epssm);
    cof *= cof;

    for (int k = 0; k < nz; ++k)
        c2a[(size_t)k * st + c] = GAMMA * p[(size_t)k * st + c]
                                        / alt[(size_t)k * st + c];

    gam[c] = 0.0f;                       // gamma[0] seeds the recurrence
    a[(size_t)st + c] = 0.0f;            // row 1: w[0] is a BC, not solved
    for (int k = 2; k < nz; ++k) {
        if (moist_cq)
            a[(size_t)k * st + c] =
                -cqw[(size_t)k * st + c] * cof * rdn[k] * rdnw[k - 1]
                * c2a[(size_t)(k - 1) * st + c]
                / ((c1h[k - 1] * mut + c2h[k - 1])
                   * (c1f[k - 1] * mut + c2f[k - 1]));
        else
            a[(size_t)k * st + c] = -cof * rdn[k] * rdnw[k - 1]
                                    * c2a[(size_t)(k - 1) * st + c]
                                    / ((c1h[k - 1] * mut + c2h[k - 1])
                                       * (c1f[k - 1] * mut + c2f[k - 1]));
    }
    if (top_lid) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        a[(size_t)nz * st + c] = -0.0f;
#else
        a[(size_t)nz * st + c] = 0.0f;   // WRF lid_flag = 0
#endif
    } else {
        // WRF v4.6.1 module_small_step_em.F:619-626: the default
        // lid_flag=1 keeps the one-sided top row coupled to w[nz-1].
#if GPUWM_WRF_EXACT
        // WRF writes rdnw(kde-1)**2, and Fortran raises the power before
        // the product: (-2.*cof)*(rdnw*rdnw), then *c2a*lid_flag.  The
        // default left-to-right ((-2*cof)*rdnw)*rdnw rounds differently
        // and the top alpha carries it down the back-substitution, the
        // levels 44-50 w/ph difference of combo-sweep round 2
        // (LOCALIZE.md item 6).  lid_flag is 1 on this branch.
        a[(size_t)nz * st + c] =
            -2.0f * cof * (rdnw[nz - 1] * rdnw[nz - 1])
            * c2a[(size_t)(nz - 1) * st + c]
            / ((c1h[nz - 1] * mut + c2h[nz - 1])
               * (c1f[nz - 1] * mut + c2f[nz - 1]));
#else
        a[(size_t)nz * st + c] =
            -2.0f * cof * rdnw[nz - 1] * rdnw[nz - 1]
            * c2a[(size_t)(nz - 1) * st + c]
            / ((c1h[nz - 1] * mut + c2h[nz - 1])
               * (c1f[nz - 1] * mut + c2f[nz - 1]));
#endif
    }
    for (int k = 1; k < nz; ++k) {
        real chm_k  = c1h[k] * mut + c2h[k];
        real chm_km = c1h[k - 1] * mut + c2h[k - 1];
        real cfm_k  = c1f[k] * mut + c2f[k];
        real cfm_kp = c1f[k + 1] * mut + c2f[k + 1];
        real b, cc;
        if (moist_cq) {
            real cq = cqw[(size_t)k * st + c];
            b = 1.0f + cq * cof * rdn[k]
                * (rdnw[k] * c2a[(size_t)k * st + c] / (chm_k * cfm_k)
                   + rdnw[k - 1] * c2a[(size_t)(k - 1) * st + c]
                     / (chm_km * cfm_k));
            cc = -cq * cof * rdn[k] * rdnw[k]
                 * c2a[(size_t)k * st + c] / (chm_k * cfm_kp);
        } else {
            b = 1.0f + cof * rdn[k]
                * (rdnw[k] * c2a[(size_t)k * st + c] / (chm_k * cfm_k)
                   + rdnw[k - 1] * c2a[(size_t)(k - 1) * st + c]
                     / (chm_km * cfm_k));
            cc = -cof * rdn[k] * rdnw[k] * c2a[(size_t)k * st + c]
                 / (chm_k * cfm_kp);
        }
        real al = 1.0f / (b - a[(size_t)k * st + c]
                              * gam[(size_t)(k - 1) * st + c]);
        alpha[(size_t)k * st + c] = al;
        gam[(size_t)k * st + c] = cc * al;
    }
#if GPUWM_WRF_EXACT
    // b = 1.+2.*cof*rdnw(kde-1)**2*c2a/(...): the power first, as above.
    real b_top = 1.0f + 2.0f * cof * (rdnw[nz - 1] * rdnw[nz - 1])
                        * c2a[(size_t)(nz - 1) * st + c]
                        / ((c1h[nz - 1] * mut + c2h[nz - 1])
                           * (c1f[nz] * mut + c2f[nz]));
#else
    real b_top = 1.0f + 2.0f * cof * rdnw[nz - 1] * rdnw[nz - 1]
                        * c2a[(size_t)(nz - 1) * st + c]
                        / ((c1h[nz - 1] * mut + c2h[nz - 1])
                           * (c1f[nz] * mut + c2f[nz]));
#endif
    alpha[(size_t)nz * st + c] = 1.0f / (b_top - a[(size_t)nz * st + c]
                                         * gam[(size_t)(nz - 1) * st + c]);
    gam[(size_t)nz * st + c] = 0.0f;
}

#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
static __device__ __forceinline__
real wrf_theta_average(real current, real previous, real muave,
                       real c1, real c2, real muts, real theta, real epssm)
{
    real average = 0.5f*((1.0f+epssm)*current+(1.0f-epssm)*previous);
    return __fdiv_rn(average+(c1*muave)*300.0f,
                     (c1*muts+c2)*(300.0f+theta));
}

// Convert the full-theta acoustic increments to WRF's perturbation words
// before averaging. Each stored REAL conversion rounds independently.
static __device__ __forceinline__
real wrf_full_theta_average(real current, real previous, real mu, real mu_old,
                            real muave, real c1, real c2, real muts,
                            real theta, real epssm)
{
    real current_wrf = current - (c1 * mu) * 300.0f;
    real previous_wrf = previous - (c1 * mu_old) * 300.0f;
    return wrf_theta_average(current_wrf, previous_wrf, muave,
                             c1, c2, muts, theta, epssm);
}
#endif

// Crank-Nicolson (off-centered by epssm) step of the coupled w''/phi''
// equations: build the phi''-equation RHS, the kinematic terrain lower BC
// on w'' from the just-updated u''/v'' (WRF advance_w; identically zero
// when flat), the w'' forcing rows, solve the tridiagonal system in-thread
// (Thomas, using the precomputed factors), apply the damp_opt=3 implicit w
// damper (dampmag > 0), then update phi'' from the new w''.  One thread
// per column.  top_lid pins w''(nz)=phi''(nz)=0; the WRF-default open branch
// retains the one-sided top forcing/coupling/update.  phi''(0) is never
// touched (fixed terrain surface).
// mu_pp/th_pp are the post-advance_mu_th fields; *_old the pre-substep
// copies (for the epssm averages muave / t_2ave).  w_ref is the uncoupled
// reference w_t* (WRF w_save) consumed only by the damper; dampmag is
// dtau*dampcoef when damp_opt == 3 and 0 otherwise.
extern "C" __global__
void advance_w_phi(real* __restrict__ w_pp, real* __restrict__ ph_pp,
                   const real* __restrict__ rw_t,
                   const real* __restrict__ rph_t,
                   const real* __restrict__ ww_pp,
                   const real* __restrict__ mu_pp,
                   const real* __restrict__ mu_pp_old,
                   const real* __restrict__ th_pp,
                   const real* __restrict__ th_pp_old,
                   const real* __restrict__ thp, const real* __restrict__ thb,
                   const real* __restrict__ php, const real* __restrict__ phb,
                   const real* __restrict__ alt, const real* __restrict__ c2a,
                   const real* __restrict__ a, const real* __restrict__ alpha,
                   const real* __restrict__ gam,
                   const real* __restrict__ mup,
                   const real* __restrict__ u_pp,
                   const real* __restrict__ v_pp,
                   const real* __restrict__ w_ref,
                   const real* __restrict__ ht,
                   const real* __restrict__ rdn, const real* __restrict__ rdnw,
                   const real* __restrict__ fnm, const real* __restrict__ fnp,
                   const real* __restrict__ c1h, const real* __restrict__ c2h,
                   const real* __restrict__ c1f, const real* __restrict__ c2f,
                   const real* __restrict__ mub2d,
                   const real* __restrict__ cqw, int moist_cq,
                   real* __restrict__ p_pp, real* __restrict__ al_pp,
                   real cf1, real cf2, real cf3, real rdx, real rdy,
                   real dtau, real epssm,
                   real dampmag, real zdamp,
                   int boundary_x, int boundary_y,
                   int spec_zone, int base3d, int top_lid,
                   int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WPHI_MAX_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;             // base-profile level stride
    size_t boff = base3d ? (size_t)c : 0;
    real mut = mub2d[c] + mup[c];
    real muts = mut + mu_pp[c];
    real muave = 0.5f * ((1.0f + epssm) * mu_pp[c]
                       + (1.0f - epssm) * mu_pp_old[c]);

    // RHS of the phi'' equation at full levels: large-step forcing, the
    // explicit half of the g*w term, minus Omega''*d(phi_t*)/d(eta).
    real rhs[WPHI_MAX_LEV];
    rhs[0] = 0.0f;
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        rhs[k] = dtau * (rph_t[f]
                         + 0.5f * G * (1.0f - epssm) * w_pp[f]);
    }
    // wdwn[m] = 0.5*(ww[m]+ww[m-1]) * rdnw[m-1] * (phi_t*[m]-phi_t*[m-1]),
    // rolled through registers (each value feeds rows m-1 and m).
    real ph_lo = phb[boff] + php[c];
    real ph_hi = phb[bstr + boff] + php[(size_t)st + c];
    real wd_lo = 0.5f * (ww_pp[(size_t)st + c] + ww_pp[c])
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
               * rdnw[0] * (php[st+c]-php[c]+phb[bstr+boff]-phb[boff]);
#else
               * rdnw[0] * (ph_hi - ph_lo);
#endif
    for (int k = 1; k < nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = phb[(size_t)(k + 1) * bstr + boff]
              + php[(size_t)(k + 1) * st + c];
        real wd_hi = 0.5f * (ww_pp[(size_t)(k + 1) * st + c]
                           + ww_pp[(size_t)k * st + c])
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
                   * rdnw[k] * (php[(size_t)(k+1)*st+c]-php[(size_t)k*st+c]
                     +phb[(size_t)(k+1)*bstr+boff]-phb[(size_t)k*bstr+boff]);
#else
                   * rdnw[k] * (ph_hi - ph_lo);
#endif
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k <= nz; ++k)
        rhs[k] = ph_pp[(size_t)k * st + c]
               + rhs[k] / (c1f[k] * mut + c2f[k]);
    if (top_lid) rhs[nz] = 0.0f;                       // rigid lid only

    // Kinematic terrain lower BC on the coupled w'' (WRF advance_w), from
    // the cf1..cf3-weighted lowest half levels of the post-advance_uv
    // momenta; identically zero over flat terrain (ht differences all 0).
    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        size_t cip = (size_t)j * nx + ip, cim = (size_t)j * nx + im;
        size_t cjp = (size_t)jp * nx + i, cjm = (size_t)jm * nx + i;
        real ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        real uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        real vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        real vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
        w_pp[c] = 0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                + (ht[c] - ht[cjm]) * vs)
                + 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                + (ht[c] - ht[cim]) * uw);
    }

    // Forcing rows of the implicit w'' system (interior w levels).
    // t_2ave: off-centered (mu*theta)'' average normalized by
    // (c1h*mu_ts+c2h)*theta_t*.
#if GPUWM_WRF_EXACT
    real t2_dn = wrf_theta_average(th_pp[c],th_pp_old[c],muave,
                                        c1h[0],c2h[0],muts,thp[c],epssm);
#else
#if GPUWM_WRF_EXACT_C_DIFFUSION
    real t2_dn = wrf_full_theta_average(th_pp[c], th_pp_old[c], mu_pp[c], mu_pp_old[c],
                                        muave, c1h[0], c2h[0], muts,
                                        thp[c] + (thb[boff] - 300.0f), epssm);
#else
    real t2_dn = 0.5f * ((1.0f + epssm) * th_pp[c]
                       + (1.0f - epssm) * th_pp_old[c])
               / ((c1h[0] * muts + c2h[0]) * (thb[boff] + thp[c]));
#endif
#endif
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c;                 // half level above
        size_t hm = h - st;                            // half level below
#if GPUWM_WRF_EXACT
        real t2_up = wrf_theta_average(th_pp[h],th_pp_old[h],muave,
                                            c1h[k],c2h[k],muts,thp[h],epssm);
#else
#if GPUWM_WRF_EXACT_C_DIFFUSION
        real t2_up = wrf_full_theta_average(th_pp[h], th_pp_old[h], mu_pp[c], mu_pp_old[c],
                                            muave, c1h[k], c2h[k], muts,
                                            thp[h] + (thb[(size_t)k * bstr + boff] - 300.0f), epssm);
#else
        real t2_up = 0.5f * ((1.0f + epssm) * th_pp[h]
                           + (1.0f - epssm) * th_pp_old[h])
                   / ((c1h[k] * muts + c2h[k])
                      * (thb[(size_t)k * bstr + boff] + thp[h]));
#endif
#endif
        real dph_up = (1.0f + epssm) * (rhs[k + 1] - rhs[k])
                    + (1.0f - epssm) * (ph_pp[(size_t)(k + 1) * st + c]
                                        - ph_pp[(size_t)k * st + c]);
        real dph_dn = (1.0f + epssm) * (rhs[k] - rhs[k - 1])
                    + (1.0f - epssm) * (ph_pp[(size_t)k * st + c]
                                        - ph_pp[(size_t)(k - 1) * st + c]);
        size_t f = (size_t)k * st + c;
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        if (moist_cq) {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + cqw[f] * (0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        } else {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + 0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn)
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        }
#else
        if (moist_cq) {
            w_pp[f] += dtau * rw_t[f]
                     + cqw[f] * (0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        } else {
            w_pp[f] += dtau * rw_t[f]
                     + 0.5f * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn)
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        }
#endif
        t2_dn = t2_up;
    }
    if (top_lid) {
        w_pp[(size_t)nz * st + c] = 0.0f;              // legacy rigid lid
    } else {
        // WRF v4.6.1 module_small_step_em.F:1420-1431.  The open top row
        // uses a one-sided pressure/buoyancy forcing and no cqw factor.
        size_t h = (size_t)(nz - 1) * st + c;
        size_t f = (size_t)nz * st + c;
        real dph_dn = (1.0f + epssm) * (rhs[nz] - rhs[nz - 1])
                    + (1.0f - epssm)
                      * (ph_pp[f] - ph_pp[(size_t)(nz - 1) * st + c]);
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + (__fdiv_rn(-0.5f*dtau*G,c1h[nz-1]*mut+c2h[nz-1])
                    * (rdnw[nz-1]*rdnw[nz-1])*2.0f*c2a[h]*dph_dn
                    -dtau*G*(2.0f*rdnw[nz-1]*c2a[h]*alt[h]*t2_dn+c1f[nz]*muave));
#else
        w_pp[f] += dtau * rw_t[f]
                 - dtau * G * c2a[h] * rdnw[nz - 1] * rdnw[nz - 1]
                   / (c1h[nz - 1] * mut + c2h[nz - 1]) * dph_dn
                 - dtau * G * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h]
                               * t2_dn + c1f[nz] * muave);
#endif
    }

    // Thomas solve with the precomputed factors (a[1] decouples the lower
    // boundary; a[nz] is live only for the WRF-default open top).
    for (int k = 1; k <= nz; ++k)
        w_pp[(size_t)k * st + c] =
            (w_pp[(size_t)k * st + c]
             - a[(size_t)k * st + c] * w_pp[(size_t)(k - 1) * st + c])
            * alpha[(size_t)k * st + c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(size_t)k * st + c] -= gam[(size_t)k * st + c]
                                  * w_pp[(size_t)(k + 1) * st + c];

    // WRF damp_opt=3 (KDH 2008): implicit w-only Rayleigh damping of the
    // solved w'' against the coupled reference w_t* (advance_w); per-column
    // heights from the t* geopotential, layer depth zdamp below the top.
    if (dampmag > 0.0f) {
        real htop = __fdiv_rn((phb[(size_t)nz * bstr + boff]
                     + php[(size_t)nz * st + c]), G);
        real hbot = htop - zdamp;
        for (int k = 1; k <= nz; ++k) {
            real hk = __fdiv_rn((phb[(size_t)k * bstr + boff]
                       + php[(size_t)k * st + c]), G);
            if (hk >= hbot) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
                real angle = __fdiv_rn(1.5707963267948966f*(hk-hbot),zdamp);
                real sn = glibc_sinf(angle);
#else
                real sn = sinf(1.5707963267948966f * (hk - hbot) / zdamp);
#endif
                real dw = dampmag * sn * sn;
                real cfm = c1f[k] * mut + c2f[k];
                size_t f = (size_t)k * st + c;
                w_pp[f] = (w_pp[f] - dw * cfm * w_ref[f]) / (1.0f + dw);
            }
        }
    }

    // phi'' forward update from the implicit half of the g*w term (eq 3.11).
    for (int k = 1; k <= nz; ++k)
        ph_pp[(size_t)k * st + c] =
            rhs[k] + 0.5f * dtau * G * (1.0f + epssm)
                     * w_pp[(size_t)k * st + c]
                     / (c1f[k] * muts + c2f[k]);

    // Pure column chain: global FP32 ph_pp and an FP32 register have the
    // same rounding before the fused diagnosis.
    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (size_t)c, st, base3d, nz);
}

// Map-scale-factor variant of advance_w_phi (Phase 3 Task 3; WRF advance_w
// with map factors).  Launched instead of advance_w_phi when any msf != 1;
// the msf==1 kernel above stays byte-identical to Phase 2 so its codegen,
// and hence the pinned bitwise regression, cannot drift.  Differences from
// advance_w_phi, all per WRF advance_w (isotropic msft = msfty = msftx):
//   - rhs coupling:   rhs[k] = ph'' + msft*rhs[k]/C_f(mut)
//   - kinematic BC:   w''(0) = msft*(u''.grad ht terms)
//   - forcing rows:   the implicit pressure-gradient and buoyancy chunks
//                     carry msft_inv = 1/msft (the rw_t forcing is already
//                     (1/my)-coupled)
//   - phi'' update:   + msft*0.5*dtau*g*(1+epssm)*w''/C_f(muts)
//   - damper: WRF's damp target (c1f*mut+c2f)*w_save has NO msf factor
//     even though w'' is msf-coupled -- transcribed literally (the
//     Fortran's own comment: "have not thought through w equation").
extern "C" __global__
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
#else
__launch_bounds__(256, 4)
#endif
void advance_w_phi_msf(real* __restrict__ w_pp, real* __restrict__ ph_pp,
                       const real* __restrict__ rw_t,
                       const real* __restrict__ rph_t,
                       const real* __restrict__ ww_pp,
                       const real* __restrict__ mu_pp,
                       const real* __restrict__ mu_pp_old,
                       const real* __restrict__ th_pp,
                       const real* __restrict__ th_pp_old,
                       const real* __restrict__ thp,
                       const real* __restrict__ thb,
                       const real* __restrict__ php,
                       const real* __restrict__ phb,
                       const real* __restrict__ alt,
                       const real* __restrict__ c2a,
                       const real* __restrict__ a,
                       const real* __restrict__ alpha,
                       const real* __restrict__ gam,
                       const real* __restrict__ mup,
                       const real* __restrict__ u_pp,
                       const real* __restrict__ v_pp,
                       const real* __restrict__ w_ref,
                       const real* __restrict__ ht,
                       const real* __restrict__ rdn,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ fnm,
                       const real* __restrict__ fnp,
                       const real* __restrict__ c1h,
                       const real* __restrict__ c2h,
                       const real* __restrict__ c1f,
                       const real* __restrict__ c2f,
                       const real* __restrict__ mub2d,
                       const real* __restrict__ cqw, int moist_cq,
                       const real* __restrict__ msft,      // (ny, nx)
                       real* __restrict__ p_pp,
                       real* __restrict__ al_pp,
                       real cf1, real cf2, real cf3, real rdx, real rdy,
                       real dtau, real epssm,
                       real dampmag, real zdamp,
                       int boundary_x, int boundary_y,
                       int spec_zone, int base3d, int top_lid,
                       int nz, int ny, int nx)
{
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WPHI_MAX_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;             // base-profile level stride
    size_t boff = base3d ? (size_t)c : 0;
    real mut = mub2d[c] + mup[c];
    real muts = mut + mu_pp[c];
    real muave = 0.5f * ((1.0f + epssm) * mu_pp[c]
                       + (1.0f - epssm) * mu_pp_old[c]);
    real msf_c = msft[c];
    real msf_i = 1.0f / msf_c;

    // RHS of the phi'' equation at full levels.
    real rhs[WPHI_MAX_LEV];
    rhs[0] = 0.0f;
    for (int k = 1; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        rhs[k] = dtau * (rph_t[f]
                         + 0.5f * G * (1.0f - epssm) * w_pp[f]);
    }
    real ph_lo = phb[boff] + php[c];
    real ph_hi = phb[bstr + boff] + php[(size_t)st + c];
    real wd_lo = 0.5f * (ww_pp[(size_t)st + c] + ww_pp[c])
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
               * rdnw[0] * (php[st+c]-php[c]+phb[bstr+boff]-phb[boff]);
#else
               * rdnw[0] * (ph_hi - ph_lo);
#endif
    for (int k = 1; k < nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = phb[(size_t)(k + 1) * bstr + boff]
              + php[(size_t)(k + 1) * st + c];
        real wd_hi = 0.5f * (ww_pp[(size_t)(k + 1) * st + c]
                           + ww_pp[(size_t)k * st + c])
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
                   * rdnw[k] * (php[(size_t)(k+1)*st+c]-php[(size_t)k*st+c]
                     +phb[(size_t)(k+1)*bstr+boff]-phb[(size_t)k*bstr+boff]);
#else
                   * rdnw[k] * (ph_hi - ph_lo);
#endif
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k <= nz; ++k)          // WRF: + msfty*rhs/C_f(mut)
        rhs[k] = ph_pp[(size_t)k * st + c]
               + msf_c * rhs[k] / (c1f[k] * mut + c2f[k]);
    if (top_lid) rhs[nz] = 0.0f;                       // rigid lid only

    // Kinematic terrain lower BC (WRF advance_w: msfty*(v part) +
    // msftx*(u part); the isotropic single factor multiplies the sum).
    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        size_t cip = (size_t)j * nx + ip, cim = (size_t)j * nx + im;
        size_t cjp = (size_t)jp * nx + i, cjm = (size_t)jm * nx + i;
        real ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        real uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        real vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        real vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        w_pp[c] = msf_c * 0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + msf_c * 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw);
#else
        w_pp[c] = msf_c * (0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw));
#endif
    }

    // Forcing rows of the implicit w'' system (interior w levels).
#if GPUWM_WRF_EXACT
    real t2_dn = wrf_theta_average(th_pp[c],th_pp_old[c],muave,
                                        c1h[0],c2h[0],muts,thp[c],epssm);
#else
#if GPUWM_WRF_EXACT_C_DIFFUSION
    real t2_dn = wrf_full_theta_average(th_pp[c], th_pp_old[c], mu_pp[c], mu_pp_old[c],
                                        muave, c1h[0], c2h[0], muts,
                                        thp[c] + (thb[boff] - 300.0f), epssm);
#else
    real t2_dn = 0.5f * ((1.0f + epssm) * th_pp[c]
                       + (1.0f - epssm) * th_pp_old[c])
               / ((c1h[0] * muts + c2h[0]) * (thb[boff] + thp[c]));
#endif
#endif
    for (int k = 1; k < nz; ++k) {
        size_t h = (size_t)k * st + c;                 // half level above
        size_t hm = h - st;                            // half level below
#if GPUWM_WRF_EXACT
        real t2_up = wrf_theta_average(th_pp[h],th_pp_old[h],muave,
                                            c1h[k],c2h[k],muts,thp[h],epssm);
#else
#if GPUWM_WRF_EXACT_C_DIFFUSION
        real t2_up = wrf_full_theta_average(th_pp[h], th_pp_old[h], mu_pp[c], mu_pp_old[c],
                                            muave, c1h[k], c2h[k], muts,
                                            thp[h] + (thb[(size_t)k * bstr + boff] - 300.0f), epssm);
#else
        real t2_up = 0.5f * ((1.0f + epssm) * th_pp[h]
                           + (1.0f - epssm) * th_pp_old[h])
                   / ((c1h[k] * muts + c2h[k])
                      * (thb[(size_t)k * bstr + boff] + thp[h]));
#endif
#endif
        real dph_up = (1.0f + epssm) * (rhs[k + 1] - rhs[k])
                    + (1.0f - epssm) * (ph_pp[(size_t)(k + 1) * st + c]
                                        - ph_pp[(size_t)k * st + c]);
        real dph_dn = (1.0f + epssm) * (rhs[k] - rhs[k - 1])
                    + (1.0f - epssm) * (ph_pp[(size_t)k * st + c]
                                        - ph_pp[(size_t)(k - 1) * st + c]);
        size_t f = (size_t)k * st + c;
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        if (moist_cq) {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + msf_i * cqw[f] * (0.5f * dtau * G * rdn[k]
                                        * (c2a[h] * rdnw[k]
                                           / (c1h[k] * mut + c2h[k]) * dph_up
                                           - c2a[hm] * rdnw[k - 1]
                                             / (c1h[k - 1] * mut + c2h[k - 1])
                                             * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        } else {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + msf_i * (0.5f * dtau * G * rdn[k]
                                * (c2a[h] * rdnw[k]
                                   / (c1h[k] * mut + c2h[k]) * dph_up
                                   - c2a[hm] * rdnw[k - 1]
                                     / (c1h[k - 1] * mut + c2h[k - 1])
                                     * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        }
#else
        if (moist_cq) {
            w_pp[f] += dtau * rw_t[f]
                     + msf_i * cqw[f] * (0.5f * dtau * G * rdn[k]
                                        * (c2a[h] * rdnw[k]
                                           / (c1h[k] * mut + c2h[k]) * dph_up
                                           - c2a[hm] * rdnw[k - 1]
                                             / (c1h[k - 1] * mut + c2h[k - 1])
                                             * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        } else {
            w_pp[f] += dtau * rw_t[f]
                     + msf_i * (0.5f * dtau * G * rdn[k]
                                * (c2a[h] * rdnw[k]
                                   / (c1h[k] * mut + c2h[k]) * dph_up
                                   - c2a[hm] * rdnw[k - 1]
                                     / (c1h[k - 1] * mut + c2h[k - 1])
                                     * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        }
#endif
        t2_dn = t2_up;
    }
    if (top_lid) {
        w_pp[(size_t)nz * st + c] = 0.0f;              // legacy rigid lid
    } else {
        // WRF v4.6.1 module_small_step_em.F:1420-1431, with msfty^-1.
        size_t h = (size_t)(nz - 1) * st + c;
        size_t f = (size_t)nz * st + c;
        real dph_dn = (1.0f + epssm) * (rhs[nz] - rhs[nz - 1])
                    + (1.0f - epssm)
                      * (ph_pp[f] - ph_pp[(size_t)(nz - 1) * st + c]);
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
        w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + msf_i
                   * (__fdiv_rn(-0.5f*dtau*G,c1h[nz-1]*mut+c2h[nz-1])
                      * (rdnw[nz-1]*rdnw[nz-1])*2.0f*c2a[h]*dph_dn
                      -dtau*G*(2.0f*rdnw[nz-1]*c2a[h]*alt[h]*t2_dn+c1f[nz]*muave));
#else
        w_pp[f] += dtau * rw_t[f]
                 + msf_i
                   * (-dtau * G * c2a[h] * rdnw[nz - 1] * rdnw[nz - 1]
                      / (c1h[nz - 1] * mut + c2h[nz - 1]) * dph_dn
                      - dtau * G
                        * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h] * t2_dn
                           + c1f[nz] * muave));
#endif
    }

    // Thomas solve with the precomputed factors.
    for (int k = 1; k <= nz; ++k)
        w_pp[(size_t)k * st + c] =
            (w_pp[(size_t)k * st + c]
             - a[(size_t)k * st + c] * w_pp[(size_t)(k - 1) * st + c])
            * alpha[(size_t)k * st + c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(size_t)k * st + c] -= gam[(size_t)k * st + c]
                                  * w_pp[(size_t)(k + 1) * st + c];

    // damp_opt=3 implicit w damper (msf-free damp target: WRF literal).
    if (dampmag > 0.0f) {
        real htop = __fdiv_rn((phb[(size_t)nz * bstr + boff]
                     + php[(size_t)nz * st + c]), G);
        real hbot = htop - zdamp;
        for (int k = 1; k <= nz; ++k) {
            real hk = __fdiv_rn((phb[(size_t)k * bstr + boff]
                       + php[(size_t)k * st + c]), G);
            if (hk >= hbot) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_DIFFUSION
                real angle = __fdiv_rn(1.5707963267948966f*(hk-hbot),zdamp);
                real sn = glibc_sinf(angle);
#else
                real sn = sinf(1.5707963267948966f * (hk - hbot) / zdamp);
#endif
                real dw = dampmag * sn * sn;
                real cfm = c1f[k] * mut + c2f[k];
                size_t f = (size_t)k * st + c;
                w_pp[f] = (w_pp[f] - dw * cfm * w_ref[f]) / (1.0f + dw);
            }
        }
    }

    // phi'' forward update (WRF: + msfty*0.5*dts*g*(1+epssm)*w/C_f(muts)).
    for (int k = 1; k <= nz; ++k)
        ph_pp[(size_t)k * st + c] =
            rhs[k] + msf_c * 0.5f * dtau * G * (1.0f + epssm)
                     * w_pp[(size_t)k * st + c]
                     / (c1f[k] * muts + c2f[k]);

    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (size_t)c, st, base3d, nz);
#else
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx || nz + 1 > WPHI_MAX_LEV) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;             // base-profile level stride
    size_t boff = base3d ? (size_t)c : 0;
    real mut = mub2d[c] + mup[c];
    real muts = mut + mu_pp[c];
    real muave = 0.5f * ((1.0f + epssm) * mu_pp[c]
                       + (1.0f - epssm) * mu_pp_old[c]);
    real msf_c = msft[c];
    real msf_i = __fdiv_rn(1.0f, msf_c);

    // Keep only the three RHS rows consumed by each forcing row.  After
    // an old phi row has had its last forcing consumer, its storage holds
    // that row's RHS until the final phi update.  The surface phi remains
    // untouched because the separate surface RHS is exactly zero.
    real rhs_lo = 0.0f;
    real ph_lo = phb[boff] + php[c];
    real ph_hi = phb[bstr + boff] + php[(size_t)st + c];
    real wd_lo = 0.5f * (ww_pp[(size_t)st + c] + ww_pp[c])
               * rdnw[0] * (ph_hi - ph_lo);
    // The outer multiply preserves the original local-array round point.
    real rhs_mid = arn_mul(dtau, rph_t[st + c]
                          + 0.5f * G * (1.0f - epssm) * w_pp[st + c]);
    if (nz > 1) {
        int k = 1;
        ph_lo = ph_hi;
        ph_hi = phb[(size_t)(k + 1) * bstr + boff]
              + php[(size_t)(k + 1) * st + c];
        real wd_hi = 0.5f * (ww_pp[(size_t)(k + 1) * st + c]
                           + ww_pp[(size_t)k * st + c])
                   * rdnw[k] * (ph_hi - ph_lo);
        rhs_mid -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    rhs_mid = ph_pp[st + c]
            + __fdiv_rn(msf_c * rhs_mid, c1f[1] * mut + c2f[1]);
    if (top_lid && nz == 1) rhs_mid = 0.0f;

    // Kinematic terrain lower BC (WRF advance_w: msfty*(v part) +
    // msftx*(u part); the isotropic single factor multiplies the sum).
    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0)      : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0)      : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        size_t cip = (size_t)j * nx + ip, cim = (size_t)j * nx + im;
        size_t cjp = (size_t)jp * nx + i, cjm = (size_t)jm * nx + i;
        real ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        real uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        real vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        real vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
        w_pp[c] = msf_c * (0.5f * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + 0.5f * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw));
    }

    // Forcing rows of the implicit w'' system (interior w levels).
    real t2_dn = __fdiv_rn(0.5f * ((1.0f + epssm) * th_pp[c]
                       + (1.0f - epssm) * th_pp_old[c]),
               (c1h[0] * muts + c2h[0]) * (thb[boff] + thp[c]));
    for (int k = 1; k < nz; ++k) {
        int kr = k + 1;
        size_t fr = (size_t)kr * st + c;
        real rhs_hi = arn_mul(dtau, rph_t[fr]
                             + 0.5f * G * (1.0f - epssm) * w_pp[fr]);
        if (kr < nz) {
            ph_lo = ph_hi;
            ph_hi = phb[(size_t)(kr + 1) * bstr + boff]
                  + php[(size_t)(kr + 1) * st + c];
            real wd_hi = 0.5f * (ww_pp[(size_t)(kr + 1) * st + c]
                               + ww_pp[(size_t)kr * st + c])
                       * rdnw[kr] * (ph_hi - ph_lo);
            rhs_hi -= dtau * (fnm[kr] * wd_hi + fnp[kr] * wd_lo);
            wd_lo = wd_hi;
        }
        rhs_hi = ph_pp[fr]
               + __fdiv_rn(msf_c * rhs_hi, c1f[kr] * mut + c2f[kr]);
        if (top_lid && kr == nz) rhs_hi = 0.0f;

        size_t h = (size_t)k * st + c;                 // half level above
        size_t hm = h - st;                            // half level below
        real t2_up = __fdiv_rn(0.5f * ((1.0f + epssm) * th_pp[h]
                           + (1.0f - epssm) * th_pp_old[h]),
                   (c1h[k] * muts + c2h[k])
                      * (thb[(size_t)k * bstr + boff] + thp[h]));
        real dph_up = (1.0f + epssm) * (rhs_hi - rhs_mid)
                    + (1.0f - epssm) * (ph_pp[(size_t)(k + 1) * st + c]
                                        - ph_pp[(size_t)k * st + c]);
        real dph_dn = (1.0f + epssm) * (rhs_mid - rhs_lo)
                    + (1.0f - epssm) * (ph_pp[(size_t)k * st + c]
                                        - ph_pp[(size_t)(k - 1) * st + c]);
        size_t f = (size_t)k * st + c;
        if (moist_cq) {
            w_pp[f] += dtau * rw_t[f]
                     + msf_i * cqw[f] * (0.5f * dtau * G * rdn[k]
                                        * (__fdiv_rn(c2a[h] * rdnw[k], c1h[k] * mut + c2h[k]) * dph_up
                                           - __fdiv_rn(c2a[hm] * rdnw[k - 1], c1h[k - 1] * mut + c2h[k - 1])
                                             * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        } else {
            w_pp[f] += dtau * rw_t[f]
                     + msf_i * (0.5f * dtau * G * rdn[k]
                                * (__fdiv_rn(c2a[h] * rdnw[k], c1h[k] * mut + c2h[k]) * dph_up
                                   - __fdiv_rn(c2a[hm] * rdnw[k - 1], c1h[k - 1] * mut + c2h[k - 1])
                                     * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        }
        t2_dn = t2_up;
        if (k > 1) ph_pp[(size_t)(k - 1) * st + c] = rhs_lo;
        rhs_lo = rhs_mid;
        rhs_mid = rhs_hi;
    }
    if (top_lid) {
        w_pp[(size_t)nz * st + c] = 0.0f;              // legacy rigid lid
    } else {
        // WRF v4.6.1 module_small_step_em.F:1420-1431, with msfty^-1.
        size_t h = (size_t)(nz - 1) * st + c;
        size_t f = (size_t)nz * st + c;
        real dph_dn = (1.0f + epssm) * (rhs_mid - rhs_lo)
                    + (1.0f - epssm)
                      * (ph_pp[f] - ph_pp[(size_t)(nz - 1) * st + c]);
        w_pp[f] += dtau * rw_t[f]
                 + msf_i
                   * (__fdiv_rn(-dtau * G * c2a[h] * rdnw[nz - 1] * rdnw[nz - 1],
                      c1h[nz - 1] * mut + c2h[nz - 1]) * dph_dn
                      - dtau * G
                        * (2.0f * rdnw[nz - 1] * c2a[h] * alt[h] * t2_dn
                           + c1f[nz] * muave));
    }
    if (nz > 1) ph_pp[(size_t)(nz - 1) * st + c] = rhs_lo;
    ph_pp[(size_t)nz * st + c] = rhs_mid;

    // Thomas solve with the precomputed factors.
    for (int k = 1; k <= nz; ++k)
        w_pp[(size_t)k * st + c] =
            (w_pp[(size_t)k * st + c]
             - a[(size_t)k * st + c] * w_pp[(size_t)(k - 1) * st + c])
            * alpha[(size_t)k * st + c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(size_t)k * st + c] -= gam[(size_t)k * st + c]
                                  * w_pp[(size_t)(k + 1) * st + c];

    // damp_opt=3 implicit w damper (msf-free damp target: WRF literal).
    if (dampmag > 0.0f) {
        real htop = __fdiv_rn((phb[(size_t)nz * bstr + boff]
                     + php[(size_t)nz * st + c]), G);
        real hbot = htop - zdamp;
        for (int k = 1; k <= nz; ++k) {
            real hk = __fdiv_rn((phb[(size_t)k * bstr + boff]
                       + php[(size_t)k * st + c]), G);
            if (hk >= hbot) {
                real sn = sinf(__fdiv_rn(1.5707963267948966f * (hk - hbot), zdamp));
                real dw = dampmag * sn * sn;
                real cfm = c1f[k] * mut + c2f[k];
                size_t f = (size_t)k * st + c;
                w_pp[f] = __fdiv_rn(w_pp[f] - dw * cfm * w_ref[f], 1.0f + dw);
            }
        }
    }

    // phi'' forward update (WRF: + msfty*0.5*dts*g*(1+epssm)*w/C_f(muts)).
    for (int k = 1; k <= nz; ++k)
        ph_pp[(size_t)k * st + c] =
            ph_pp[(size_t)k * st + c]
                     + __fdiv_rn(msf_c * 0.5f * dtau * G * (1.0f + epssm)
                     * w_pp[(size_t)k * st + c],
                     c1f[k] * muts + c2f[k]);

    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (size_t)c, st, base3d, nz);
#endif
}

// Specified-frame phi update followed by zero-gradient w.  The two outputs
// are independent; one thread owns one frame point, and every source w point
// is clamped to the untouched interior.  arn_* matches each former eager
// CuPy operator boundary in acoustic._advance_specified_phi.
extern "C" __global__
void advance_specified_phi_w(real* __restrict__ ph_pp,
                             real* __restrict__ w_pp,
                             real* __restrict__ p_pp,
                             real* __restrict__ al_pp,
                             const real* __restrict__ rph_t,
                             const real* __restrict__ th_pp,
                             const real* __restrict__ mup,
                             const real* __restrict__ mu_pp,
                             const real* __restrict__ mub2d,
                             const real* __restrict__ rmu_t,
                             const real* __restrict__ php,
                             const real* __restrict__ thp,
                             const real* __restrict__ thb,
                             const real* __restrict__ alt,
                             const real* __restrict__ c2a,
                             const real* __restrict__ rdnw,
                             const real* __restrict__ c1h,
                             const real* __restrict__ c2h,
                             const real* __restrict__ c1f,
                             const real* __restrict__ c2f,
                             real dtau, int spec_zone, int base3d,
#if GPUWM_WRF_EXACT
                             int nz, int ny, int nx,
                             const real* __restrict__ muts_wrf)
#else
                             int nz, int ny, int nx)
#endif
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    if (i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j < ny - spec_zone) return;

#if GPUWM_WRF_EXACT
    real muts = muts_wrf[c];
#else
    real mut = arn_add(mub2d[c], mup[c]);
    real muts = arn_add(mut, mu_pp[c]);
#endif
    real dt_rmu = arn_mul(dtau, rmu_t[c]);
    real mu_old = arn_sub(muts, dt_rmu);
    int source_j = min(max(j, spec_zone), ny - 1 - spec_zone);
    int source_i = min(max(i, spec_zone), nx - 1 - spec_zone);
    size_t source_c = (size_t)source_j * nx + source_i;
    for (int k = 0; k <= nz; ++k) {
        size_t tid = (size_t)k * st + c;
        real numerator = arn_add(arn_mul(c1f[k], mu_old), c2f[k]);
        real denominator = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        real ratio = arn_div(numerator, denominator);
#if GPUWM_WRF_EXACT
        // spec_bdyupdate_ph: field*(c1*mu_old+c2)/(c1*muts+c2), i.e.
        // (field*num)/den.  field*(num/den) agrees while phi'' is zero on
        // the ring (RK stage 1) and rounds apart from stage 2 on.
        real value0 = arn_div(arn_mul(ph_pp[tid], numerator), denominator);
#else
        real value0 = arn_mul(ph_pp[tid], ratio);
#endif
        real value1 = arn_mul(dtau, rph_t[tid]);
        real denominator2 = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        real value2 = arn_div(value1, denominator2);
        real value3 = arn_add(value0, value2);
        real value4 = arn_sub(ratio, 1.0f);
        real value5 = arn_mul(php[tid], value4);
        ph_pp[tid] = arn_add(value3, value5);
        w_pp[tid] = w_pp[(size_t)k * st + source_c];
    }
#if GPUWM_WRF_EXACT
    diagnose_p_column_muts(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, alt, c2a,
                           rdnw, c1h, c2h, c, st, nz, muts);
#else
    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      c, st, base3d, nz);
#endif
}

// Nested counterpart of advance_specified_phi_w.  solve_em.F:1602-1611
// applies spec_bdyupdate(w_2,rw_tend,dts_rk) instead of the root domain's
// zero-gradient copy.  The phi and diagnostic trees stay identical to the
// dyn fused specified-frame path above; only the independent w operation
// differs.
extern "C" __global__
void advance_nested_phi_w(real* __restrict__ ph_pp,
                          real* __restrict__ w_pp,
                          real* __restrict__ p_pp,
                          real* __restrict__ al_pp,
                          const real* __restrict__ rph_t,
                          const real* __restrict__ rw_t,
                          const real* __restrict__ th_pp,
                          const real* __restrict__ mup,
                          const real* __restrict__ mu_pp,
                          const real* __restrict__ mub2d,
                          const real* __restrict__ rmu_t,
                          const real* __restrict__ php,
                          const real* __restrict__ thp,
                          const real* __restrict__ thb,
                          const real* __restrict__ alt,
                          const real* __restrict__ c2a,
                          const real* __restrict__ rdnw,
                          const real* __restrict__ c1h,
                          const real* __restrict__ c2h,
                          const real* __restrict__ c1f,
                          const real* __restrict__ c2f,
                          real dtau, int spec_zone, int base3d,
#if GPUWM_WRF_EXACT
                          int nz, int ny, int nx,
                          const real* __restrict__ muts_wrf)
#else
                          int nz, int ny, int nx)
#endif
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    if (i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j < ny - spec_zone) return;

#if GPUWM_WRF_EXACT
    real muts = muts_wrf[c];
#else
    real mut = arn_add(mub2d[c], mup[c]);
    real muts = arn_add(mut, mu_pp[c]);
#endif
    real dt_rmu = arn_mul(dtau, rmu_t[c]);
    real mu_old = arn_sub(muts, dt_rmu);
    for (int k = 0; k <= nz; ++k) {
        size_t tid = (size_t)k * st + c;
        real numerator = arn_add(arn_mul(c1f[k], mu_old), c2f[k]);
        real denominator = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        real ratio = arn_div(numerator, denominator);
#if GPUWM_WRF_EXACT
        // spec_bdyupdate_ph: field*(c1*mu_old+c2)/(c1*muts+c2), i.e.
        // (field*num)/den.  field*(num/den) agrees while phi'' is zero on
        // the ring (RK stage 1) and rounds apart from stage 2 on.
        real value0 = arn_div(arn_mul(ph_pp[tid], numerator), denominator);
#else
        real value0 = arn_mul(ph_pp[tid], ratio);
#endif
        real value1 = arn_mul(dtau, rph_t[tid]);
        real denominator2 = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        real value2 = arn_div(value1, denominator2);
        real value3 = arn_add(value0, value2);
        real value4 = arn_sub(ratio, 1.0f);
        real value5 = arn_mul(php[tid], value4);
        ph_pp[tid] = arn_add(value3, value5);
        w_pp[tid] = arn_add(w_pp[tid], arn_mul(dtau, rw_t[tid]));
    }
#if GPUWM_WRF_EXACT
    diagnose_p_column_muts(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, alt, c2a,
                           rdnw, c1h, c2h, c, st, nz, muts);
#else
    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      c, st, base3d, nz);
#endif
}

// WRF external-mode filter.  gx/gy were formerly full-domain CuPy
// temporaries; arn_* retains their subtraction, scale, optional division,
// c1h multiplication, and final addition as five distinct FP32 operations.
extern "C" __global__
void apply_emdiv(real* __restrict__ u_pp,
                 real* __restrict__ v_pp,
                 const real* __restrict__ mudf,
                 const real* __restrict__ mu_pp,
                 real* __restrict__ mu_prev,
                 const real* __restrict__ c1h,
                 const real* __restrict__ msfu,
                 const real* __restrict__ msfv,
                 real xscale, real yscale,
                 int has_msf, int boundary_x, int boundary_y,
                 int specified, int spec_zone, int save_mu,
                 int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;

    if (save_mu && k == 0 && j < ny && i < nx)
        mu_prev[(size_t)j * nx + i] = mu_pp[(size_t)j * nx + i];

    bool do_u = false;
    int gi = i;
    if (j < ny) {
        if (specified)
            do_u = j >= spec_zone && j < ny - spec_zone
                && i >= spec_zone && i <= nx - spec_zone;
        else if (boundary_x)
            do_u = i >= 1 && i < nx;
        else {
            do_u = true;
            if (i == nx) gi = 0;
        }
    }
    if (do_u) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int gim = (gi - 1 + nx) % nx;
#else
        int gim = (gi == 0 ? nx - 1 : gi - 1);
#endif
        size_t c = (size_t)j * nx + gi;
        size_t cm = (size_t)j * nx + gim;
        real gx = arn_sub(mudf[c], mudf[cm]);
        gx = arn_mul(xscale, gx);
        if (has_msf) gx = arn_div(gx, msfu[(size_t)j * nxf + gi]);
        real increment = arn_mul(c1h[k], gx);
        size_t ix = I3S(k, j, i, ny, nxf);
        u_pp[ix] = arn_add(u_pp[ix], increment);
    }

    bool do_v = false;
    int gj = j;
    if (i < nx) {
        if (specified)
            do_v = i >= spec_zone && i < nx - spec_zone
                && j >= spec_zone && j <= ny - spec_zone;
        else if (boundary_y)
            do_v = j >= 1 && j < ny;
        else {
            do_v = true;
            if (j == ny) gj = 0;
        }
    }
    if (do_v) {
#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || GPUWM_WRF_EXACT_D_DIAGNOSTICS
        int gjm = (gj - 1 + ny) % ny;
#else
        int gjm = (gj == 0 ? ny - 1 : gj - 1);
#endif
        size_t c = (size_t)gj * nx + i;
        size_t cm = (size_t)gjm * nx + i;
        real gy = arn_sub(mudf[c], mudf[cm]);
        gy = arn_mul(yscale, gy);
#if GPUWM_WRF_EXACT
        if (has_msf) gy = arn_mul(gy, msfv[(size_t)gj * nx + i]); // strict MF_VX_INV
#else
        if (has_msf) gy = arn_div(gy, msfv[(size_t)gj * nx + i]);
#endif
        real increment = arn_mul(c1h[k], gy);
        size_t ix = I3S(k, j, i, nyf, nx);
        v_pp[ix] = arn_add(v_pp[ix], increment);
    }
}

extern "C" __global__
void update_mudf(real* __restrict__ mudf,
                 const real* __restrict__ mu_pp,
                 const real* __restrict__ mu_prev,
                 real dtau, int boundary_x, int boundary_y,
                 int width, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    int j = c / nx, i = c - j * nx;
    if ((boundary_x && (i < width || i >= nx - width))
        || (boundary_y && (j < width || j >= ny - width))) {
        mudf[c] = 0.0f;
        return;
    }
    real difference = arn_sub(mu_pp[c], mu_prev[c]);
    mudf[c] = arn_div(difference, dtau);
}

// The three WRF sumflux arrays have different staggers but no cross-field
// dependencies.  A union launch preserves each element's scalar operation.
extern "C" __global__
void zero_sumflux(real* __restrict__ ru_m,
                  real* __restrict__ rv_m,
                  real* __restrict__ ww_m,
                  size_t nu, size_t nv, size_t nw, size_t nmax)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nmax) return;
    if (tid < nu) ru_m[tid] = 0.0f;
    if (tid < nv) rv_m[tid] = 0.0f;
    if (tid < nw) ww_m[tid] = 0.0f;
}

extern "C" __global__
void accumulate_sumflux(real* __restrict__ ru_m,
                        real* __restrict__ rv_m,
                        real* __restrict__ ww_m,
                        const real* __restrict__ u_pp,
                        const real* __restrict__ v_pp,
                        const real* __restrict__ ww_pp,
                        size_t nu, size_t nv, size_t nw, size_t nmax)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nmax) return;
    if (tid < nu) ru_m[tid] = arn_add(ru_m[tid], u_pp[tid]);
    if (tid < nv) rv_m[tid] = arn_add(rv_m[tid], v_pp[tid]);
    if (tid < nw) ww_m[tid] = arn_add(ww_m[tid], ww_pp[tid]);
}

extern "C" __global__
void finish_sumflux(real* __restrict__ ru_m,
                    real* __restrict__ rv_m,
                    real* __restrict__ ww_m,
                    const real* __restrict__ ru,
                    const real* __restrict__ rv,
                    const real* __restrict__ ww,
                    real nsub, size_t nu, size_t nv, size_t nw, size_t nmax)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nmax) return;
    if (tid < nu) {
        real mean = arn_div(ru_m[tid], nsub);
        ru_m[tid] = arn_add(mean, ru[tid]);
    }
    if (tid < nv) {
        real mean = arn_div(rv_m[tid], nsub);
        rv_m[tid] = arn_add(mean, rv[tid]);
    }
    if (tid < nw) {
        real mean = arn_div(ww_m[tid], nsub);
        ww_m[tid] = arn_add(mean, ww[tid]);
    }
}
