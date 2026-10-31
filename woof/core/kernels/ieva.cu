// WRF 4.7.1's implicit-explicit vertical advection (IEVA, namelist
// zadvect_implicit = 1; Wicker and Skamarock 2020, after Shchepetkin 2015),
// dyn_em/module_ieva_em.F, as rk_tendency and rk_scalar_tend call it on the
// last RK3 substep (module_em.F:436-715, 1216-1364).
// The wrf_legacy variant also preserves the earlier module_advect_em
// current-mass operator and module_big_step_utilities_em directional split.
// Source pins and the reproduced WRF public-domain notice: root NOTICE,
// licenses/LICENSE-WRF-public-domain.txt, tools/ieva_wrf_oracle/legacy_build.py.
//
// ieva_ww_split     WW_SPLIT: the eta mass flux ww into its explicit part
//                   wwE (advected by the usual upwind operators) and its
//                   implicit part wwI (the column solves below).
// ieva_mut_new      CALC_MUT_NEW: the column mass after dt from the stage's
//                   horizontal mass-flux divergence.
// ieva_solve_s      advect_s_implicit (theta and every transported scalar).
// ieva_solve_u/_v   advect_u_implicit / advect_v_implicit.
// ieva_solve_ph     advect_ph_implicit (advective form, centred wwI).
// ieva_solve_w      advect_w_implicit, with its terrain lower boundary and
//                   geopotential upper boundary.
//
// Arithmetic is WRF's, operator by operator: REAL (FP32) where the Fortran
// is REAL, REAL(KIND=8) where the tridiagonal arrays are, the FP32
// coefficients rounded to FP32 before they are widened, Fortran's
// left-to-right grouping, and every operation spelled with a
// round-to-nearest intrinsic so NVRTC cannot contract a product and a sum
// into an FMA that gfortran (x86-64 SSE2, no -mfma) never forms.  Every
// divisor that is a Fortran constant (alpha_max, the 0.25 and 0.5 halving
// is a product) and g arrives as a kernel argument, not a literal: NVRTC
// may turn a division by a literal into a multiply by its reciprocal on
// sm_120 (A146).
//
// Index conventions (0-based, ArWen's (k, j, i) C order): w levels
// 0..nz, mass levels 0..nz-1; WRF's 1-based w level k is kw = k-1 here and
// its mass level k is km = k-1.  rdnw[km] = 1/dnw, rdn[kw] = 1/dn.

#ifndef IEVA_KMAX
#define IEVA_KMAX 65
#endif

static __device__ __forceinline__ float f_add(float a, float b) { return __fadd_rn(a, b); }
static __device__ __forceinline__ float f_sub(float a, float b) { return __fsub_rn(a, b); }
static __device__ __forceinline__ float f_mul(float a, float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float f_div(float a, float b) { return __fdiv_rn(a, b); }
static __device__ __forceinline__ double d_add(double a, double b) { return __dadd_rn(a, b); }
static __device__ __forceinline__ double d_sub(double a, double b) { return __dsub_rn(a, b); }
static __device__ __forceinline__ double d_mul(double a, double b) { return __dmul_rn(a, b); }
static __device__ __forceinline__ double d_div(double a, double b) { return __ddiv_rn(a, b); }

// gfortran's MAX/MIN of two REAL variables compile to maxss/minss with the
// Fortran operands in order: the first argument when it is strictly
// greater (smaller), else the second.  Only the sign of an exact zero
// depends on it.
static __device__ __forceinline__ float f_max(float a, float b) { return (a > b) ? a : b; }
static __device__ __forceinline__ float f_min(float a, float b) { return (a < b) ? a : b; }

// Thomas algorithm, WRF TRIDIAG2D, one step of the forward sweep at level
// k.  ``c_prev`` is the super-diagonal of level k-1.
static __device__ __forceinline__ void thomas_forward(
    int k, int kstart, double a, double b, double c_prev, double r,
    double &bet, double *gam, double *bb)
{
    if (k == kstart) {
        bet = b;
        bb[k] = d_div(r, bet);
    } else {
        gam[k] = d_div(c_prev, bet);
        bet = d_sub(b, d_mul(a, gam[k]));
        bb[k] = d_div(d_sub(r, d_mul(a, bb[k - 1])), bet);
    }
}

static __device__ __forceinline__ void thomas_backward(
    int kstart, int kend, const double *gam, double *bb)
{
    for (int k = kend - 1; k >= kstart; --k)
        bb[k] = d_sub(bb[k], d_mul(gam[k + 1], bb[k + 1]));
}

// The FP32 coefficient triple of the flux-form upwind solves (s, u, v, w):
//   at = -dt*max(wiL,0);  ct = dt*min(wiR,0);  btmp = dt*(max(wiR,0) - min(wiL,0))
// each rounded to FP32 before it is widened, as the Fortran assigns a REAL
// expression to a REAL(KIND=8) array.
static __device__ __forceinline__ void upwind_coefficients(
    float wiL, float wiR, float dt, float zero,
    double &at, double &ct, double &btmp)
{
    at = (double)(-f_mul(dt, f_max(wiL, zero)));
    ct = (double)f_mul(dt, f_min(wiR, zero));
    btmp = (double)f_mul(dt, f_sub(f_max(wiR, zero), f_min(wiL, zero)));
}

// rt = single - mass*(at*x_km1 + btmp*x_k + ct*x_kp1): the product sum in
// REAL(KIND=8) left to right, the FP32 mass widened, the FP32 lead term
// widened last.
static __device__ __forceinline__ double upwind_rhs(
    float lead, float mass, double at, double btmp, double ct,
    float x_km1, float x_k, float x_kp1)
{
    double sum = d_add(d_add(d_mul(at, (double)x_km1),
                             d_mul(btmp, (double)x_k)),
                       d_mul(ct, (double)x_kp1));
    return d_sub((double)lead, d_mul((double)mass, sum));
}

// ---------------------------------------------------------------------------
// WW_SPLIT (module_ieva_em.F:49-323), the IEVA branch.  One thread per w
// point, (nz+1)*ny*nx.  Surface and top: all explicit.  u/v are the
// uncoupled winds the caller hands WW_SPLIT (the stage's u_2/v_2 in
// rk_tendency, the time-n u_1/v_1 in rk_scalar_tend), mut that call's
// column mass, dt the full step.
extern "C" __global__
void ieva_ww_split(const float* __restrict__ ww,
                   const float* __restrict__ u,
                   const float* __restrict__ v,
                   const float* __restrict__ mut,
                   const float* __restrict__ rdnw,
                   const float* __restrict__ c1f,
                   const float* __restrict__ c2f,
                   float rdx, float rdy, float dt,
                   float alpha_max, float ceps, float cmnx_ratio,
                   float cutoff, float r4cmx, float quarter, float one,
                   float zero,
                   float* __restrict__ wwE, float* __restrict__ wwI,
                   const float* __restrict__ msft, int has_msf, int legacy,
                   int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)(nz + 1) * st) return;
    int k = (int)(tid / st);
    size_t c = tid - (size_t)k * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    float w = ww[tid];
    if (k == 0 || k == nz) {
        wwE[tid] = w;
        wwI[tid] = zero;
        return;
    }
    const size_t nxu = (size_t)nx + 1;
    const size_t ua = ((size_t)k * ny + j) * nxu + i;          // mass level k
    const size_t ub = ((size_t)(k - 1) * ny + j) * nxu + i;    // mass level k-1
    // cx = 0.25*rdx*(u(i+1,k) + u(i,k) + u(i+1,k-1) + u(i,k-1))
    float su = f_add(f_add(f_add(u[ua + 1], u[ua]), u[ub + 1]), u[ub]);
    float cx = f_mul(f_mul(quarter, rdx), su);
    const size_t va = ((size_t)k * (ny + 1) + j) * nx + i;
    const size_t vb = ((size_t)(k - 1) * (ny + 1) + j) * nx + i;
    float sv = f_add(f_add(f_add(v[va + nx], v[va]), v[vb + nx]), v[vb]);
    float cy = f_mul(f_mul(quarter, rdy), sv);
    // cw_max = max(alpha_max - dt*Ceps*sqrt(cx**2 + cy**2), 0.0)
    float hyp = __fsqrt_rn(f_add(f_mul(cx, cx), f_mul(cy, cy)));
    if (legacy) {
        // The earlier WW_SPLIT takes the flow on the upwind mass level,
        // including the mass-point map factor, and uses cx+cy.
        size_t ul = (w < zero) ? ua : ub;
        size_t vl = (w < zero) ? va : vb;
        float msf = has_msf ? msft[c] : one;
        cx = f_mul(f_mul(msf, rdx),
                   f_sub(f_max(u[ul + 1], zero), f_min(u[ul], zero)));
        cy = f_mul(f_mul(msf, rdy),
                   f_sub(f_max(v[vl + nx], zero), f_min(v[vl], zero)));
        hyp = f_add(cx, cy);
    }
    float cw_max = f_max(f_sub(alpha_max, f_mul(f_mul(dt, ceps), hyp)), zero);
    // cr = ww*dt*rdnw(k)/(c1f(k)*mut + c2f(k))
    float mass = f_add(f_mul(c1f[k], mut[c]), c2f[k]);
    float cr = f_div(f_mul(f_mul(w, dt), rdnw[k]), mass);
    if (cw_max > zero) {
        float cw_max2 = f_mul(cw_max, cw_max);
        float cw_min = f_mul(cw_max, cmnx_ratio);
        float cw = fabsf(cr);
        float cff;
        if (cw < cw_min) {
            cff = cw_max2;
        } else if (cw < f_mul(cutoff, cw_min)) {
            float d = f_sub(cw, cw_min);
            cff = f_add(cw_max2, f_mul(r4cmx, f_mul(d, d)));
        } else {
            cff = f_mul(cw_max, cw);
        }
        float wfrac = f_div(cw_max2, cff);
        wfrac = f_max(f_min(wfrac, one), zero);        // amax1(amin1(.,1),0)
        wwE[tid] = f_mul(w, wfrac);
        wwI[tid] = f_mul(w, f_sub(one, wfrac));
    } else {
        wwE[tid] = zero;
        wwI[tid] = w;
    }
}

// ---------------------------------------------------------------------------
// CALC_MUT_NEW (module_ieva_em.F:431-557).  One thread per mass column.
// muu/muv are the stage's face column masses (the ones that couple the
// stage's ru/rv), u/v the stage's uncoupled winds.  divv in WRF's grouping:
//   (msftx/rdnw(k)) * ( rdx*( C(muu(i+1))*u(i+1)/msfuy(i+1) - C(muu(i))*u(i)/msfuy(i) )
//                      + rdy*( C(muv(j+1))*v(j+1)*msfvx_inv(j+1) - C(muv(j))*v(j)*msfvx_inv(j) ) )
// summed bottom-up in one FP32 accumulator; mut_new = mut_old - dt*dmdt.
// Without map factors WRF's msftx = msfuy = msfvx_inv = 1, so the
// numerator is ``one`` and the other two factors are skipped (exact).
extern "C" __global__
void ieva_mut_new(const float* __restrict__ u,
                  const float* __restrict__ v,
                  const float* __restrict__ muu,
                  const float* __restrict__ muv,
                  const float* __restrict__ c1h,
                  const float* __restrict__ c2h,
                  const float* __restrict__ rdnw,
                  const float* __restrict__ mut_old,
                  const float* __restrict__ msft,
                  const float* __restrict__ msfu,
                  const float* __restrict__ msfv,
                  float rdx, float rdy, float dt, float one, int has_msf,
                  float* __restrict__ mut_new,
                  int nz, int ny, int nx)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    const size_t nxu = (size_t)nx + 1;
    const size_t fu = (size_t)j * nxu + i;          // west face of column
    const size_t fv = (size_t)j * nx + i;           // south face of column
    float numer = has_msf ? msft[c] : one;
    float inv_n = one, inv_s = one;
    if (has_msf) {                                  // msfvx_inv = 1./msfvx
        #if GPUWM_WRF_EXACT
        inv_n = msfv[fv + nx];
        inv_s = msfv[fv];
#else
        inv_n = f_div(one, msfv[fv + nx]);
        inv_s = f_div(one, msfv[fv]);
#endif
    }
    float dmdt = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const size_t uk = (size_t)k * ny * nxu + fu;
        const size_t vk = (size_t)k * (ny + 1) * nx + fv;
        float ae = f_mul(f_add(f_mul(c1h[k], muu[fu + 1]), c2h[k]), u[uk + 1]);
        float aw = f_mul(f_add(f_mul(c1h[k], muu[fu]), c2h[k]), u[uk]);
        if (has_msf) {
            ae = f_div(ae, msfu[fu + 1]);
            aw = f_div(aw, msfu[fu]);
        }
        float bn = f_mul(f_add(f_mul(c1h[k], muv[fv + nx]), c2h[k]), v[vk + nx]);
        float bs = f_mul(f_add(f_mul(c1h[k], muv[fv]), c2h[k]), v[vk]);
        if (has_msf) {
            bn = f_mul(bn, inv_n);
            bs = f_mul(bs, inv_s);
        }
        float bracket = f_add(f_mul(rdx, f_sub(ae, aw)), f_mul(rdy, f_sub(bn, bs)));
        float divv = f_mul(f_div(numer, rdnw[k]), bracket);
        dmdt = f_add(dmdt, divv);
    }
    mut_new[c] = f_sub(mut_old[c], f_mul(dt, dmdt));
}

// calc_mu_uv_1 on one face: 0.5*(mu(i)+mu(i-1)) inside, the edge cell's
// own mass on a non-periodic edge face (0.5*(mu+mu) is exact), wrapped
// when periodic.
static __device__ __forceinline__ float face_mass(
    const float* __restrict__ mu, size_t row, int f, int n, int stride,
    int periodic, float half)
{
    int a = f, b = f - 1;
    if (periodic) {
        a = (a % n + n) % n;
        b = (b % n + n) % n;
    } else {
        if (a > n - 1) a = n - 1;
        if (b < 0) b = 0;
        if (f == 0) b = a;
        if (f == n) a = b;
    }
    return f_mul(half, f_add(mu[row + (size_t)a * stride], mu[row + (size_t)b * stride]));
}

// ---------------------------------------------------------------------------
// advect_s_implicit (module_ieva_em.F:705-835).  One thread per mass
// column, levels 0..nz-1.  ``tend`` holds the explicit coupled tendency on
// entry and the IEVA tendency bb/dt on exit.  s_old = (base - shift) +
// pert (base optional: a 1-D column when base_mode == 1, 3-D when 2) is
// the time-n scalar the Fortran calls s_old; mut_old is the time-n column
// mass and mut_new the one the implicit coefficients divide by.  Theta
// passes WRF's t0 as ``shift``, so the solve reads WRF's t_1 = theta - t0
// and never the 300 K constant (see gpuwm/core/ieva.py).
extern "C" __global__
void ieva_solve_s(float* __restrict__ tend,
                  const float* __restrict__ pert,
                  const float* __restrict__ base, int base_mode, float shift,
                  const float* __restrict__ rom,
                  const float* __restrict__ c1h,
                  const float* __restrict__ c2h,
                  const float* __restrict__ mut_old,
                  const float* __restrict__ mut_new,
                  const float* __restrict__ rdnw,
                  float dt, float one, float zero,
                  int nz, int ny, int nx)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st) return;
    double gam[IEVA_KMAX], bb[IEVA_KMAX];
    double bet = 0.0, ct_prev = 0.0;
    const float mo = mut_old[c], mn = mut_new[c];
    for (int k = 0; k < nz; ++k) {
        int km1 = (k == 0) ? 0 : k - 1;
        int kp1 = (k == nz - 1) ? nz - 1 : k + 1;
        float s_m, s_0, s_p;
        if (base_mode == 0) {
            s_m = pert[(size_t)km1 * st + c];
            s_0 = pert[(size_t)k * st + c];
            s_p = pert[(size_t)kp1 * st + c];
        } else if (base_mode == 1) {
            s_m = f_add(f_sub(base[km1], shift), pert[(size_t)km1 * st + c]);
            s_0 = f_add(f_sub(base[k], shift), pert[(size_t)k * st + c]);
            s_p = f_add(f_sub(base[kp1], shift), pert[(size_t)kp1 * st + c]);
        } else {
            s_m = f_add(f_sub(base[(size_t)km1 * st + c], shift), pert[(size_t)km1 * st + c]);
            s_0 = f_add(f_sub(base[(size_t)k * st + c], shift), pert[(size_t)k * st + c]);
            s_p = f_add(f_sub(base[(size_t)kp1 * st + c], shift), pert[(size_t)kp1 * st + c]);
        }
        float mass_new = f_add(f_mul(c1h[k], mn), c2h[k]);
        float wiL = f_div(f_mul(rom[(size_t)k * st + c], rdnw[k]), mass_new);
        float wiR = f_div(f_mul(rom[(size_t)(k + 1) * st + c], rdnw[k]), mass_new);
        double at, ct, btmp;
        upwind_coefficients(wiL, wiR, dt, zero, at, ct, btmp);
        double bt = d_add((double)one, btmp);
        float mass_old = f_add(f_mul(c1h[k], mo), c2h[k]);
        double rt = upwind_rhs(f_mul(dt, tend[(size_t)k * st + c]), mass_old,
                               at, btmp, ct, s_m, s_0, s_p);
        thomas_forward(k, 0, at, bt, ct_prev, rt, bet, gam, bb);
        ct_prev = ct;
    }
    thomas_backward(0, nz - 1, gam, bb);
    for (int k = 0; k < nz; ++k)
        tend[(size_t)k * st + c] = f_div(__double2float_rn(bb[k]), dt);
}

// ---------------------------------------------------------------------------
// advect_u_implicit (module_ieva_em.F:838-977).  One thread per u face
// column, faces [f_lo, f_hi] of every row: WRF's i_start..i_end, i.e. the
// interior faces 1..nx-1 on a specified or nested domain and every face
// when x is periodic (face nx duplicates face 0).  wwI is averaged to the
// face, the face masses are calc_mu_uv_1's of mut_old and mut_new.
extern "C" __global__
void ieva_solve_u(float* __restrict__ tend,
                  const float* __restrict__ u_old,
                  const float* __restrict__ rom,
                  const float* __restrict__ c1h,
                  const float* __restrict__ c2h,
                  const float* __restrict__ mut_old,
                  const float* __restrict__ mut_new,
                  const float* __restrict__ rdnw,
                  const float* __restrict__ msfu,
                  float dt, float one, float zero, float half,
                  int has_msf, int periodic, int f_lo, int f_hi,
                  const float* __restrict__ legacy_face, int legacy,
                  int nz, int ny, int nx)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nfaces = f_hi - f_lo + 1;
    if (nfaces <= 0 || t >= (size_t)ny * nfaces) return;
    int j = (int)(t / nfaces);
    int f = f_lo + (int)(t - (size_t)j * nfaces);
    const size_t nxu = (size_t)nx + 1;
    const size_t st = (size_t)ny * nx, stu = (size_t)ny * nxu;
    const size_t row = (size_t)j * nx;
    int ia = periodic ? (f % nx) : f;               // column east of face
    int ib = periodic ? ((f - 1 + nx) % nx) : f - 1; // column west of face
    const size_t fc = (size_t)j * nxu + f;
    float mo = face_mass(mut_old, row, f, nx, 1, periodic, half);
    float mn = face_mass(mut_new, row, f, nx, 1, periodic, half);
    if (legacy) mo = mn = legacy_face[fc];
    float msf = has_msf ? msfu[fc] : one;
    double gam[IEVA_KMAX], bb[IEVA_KMAX];
    double bet = 0.0, ct_prev = 0.0;
    for (int k = 0; k < nz; ++k) {
        int km1 = (k == 0) ? 0 : k - 1;
        int kp1 = (k == nz - 1) ? nz - 1 : k + 1;
        float mass_new = f_add(f_mul(c1h[k], mn), c2h[k]);
        float rl = f_mul(half, f_add(rom[(size_t)k * st + row + ib],
                                     rom[(size_t)k * st + row + ia]));
        float rr = f_mul(half, f_add(rom[(size_t)(k + 1) * st + row + ib],
                                     rom[(size_t)(k + 1) * st + row + ia]));
        float wiL = f_mul(rl, rdnw[k]);
        float wiR = f_mul(rr, rdnw[k]);
        if (has_msf) {
            wiL = f_mul(wiL, msf);
            wiR = f_mul(wiR, msf);
        }
        wiL = f_div(wiL, mass_new);
        wiR = f_div(wiR, mass_new);
        double at, ct, btmp;
        upwind_coefficients(wiL, wiR, dt, zero, at, ct, btmp);
        double bt = d_add((double)one, btmp);
        float mass_old = f_add(f_mul(c1h[k], mo), c2h[k]);
        float lead = f_mul(dt, tend[(size_t)k * stu + fc]);
        if (has_msf) lead = f_mul(lead, msf);
        double rt = upwind_rhs(lead, mass_old, at, btmp, ct,
                               u_old[(size_t)km1 * stu + fc],
                               u_old[(size_t)k * stu + fc],
                               u_old[(size_t)kp1 * stu + fc]);
        thomas_forward(k, 0, at, bt, ct_prev, rt, bet, gam, bb);
        ct_prev = ct;
    }
    thomas_backward(0, nz - 1, gam, bb);
    for (int k = 0; k < nz; ++k) {
        float out = f_div(__double2float_rn(bb[k]), dt);
        if (has_msf) out = f_div(out, msf);
        tend[(size_t)k * stu + fc] = out;
    }
}

// ---------------------------------------------------------------------------
// advect_v_implicit (module_ieva_em.F:980-1116).  One thread per v face
// column, faces [f_lo, f_hi] of every column i.  WRF divides the implicit
// wind by msfvy and couples the tendency with msfvx; gpuwm's map factor
// is isotropic, so both are msfv.
extern "C" __global__
void ieva_solve_v(float* __restrict__ tend,
                  const float* __restrict__ v_old,
                  const float* __restrict__ rom,
                  const float* __restrict__ c1h,
                  const float* __restrict__ c2h,
                  const float* __restrict__ mut_old,
                  const float* __restrict__ mut_new,
                  const float* __restrict__ rdnw,
                  const float* __restrict__ msfv,
                  float dt, float one, float zero, float half,
                  int has_msf, int periodic, int f_lo, int f_hi,
                  const float* __restrict__ legacy_face, int legacy,
                  int nz, int ny, int nx)
{
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nfaces = f_hi - f_lo + 1;
    if (nfaces <= 0 || t >= (size_t)nx * nfaces) return;
    int f = f_lo + (int)(t / nx);
    int i = (int)(t - (size_t)(f - f_lo) * nx);
    const size_t st = (size_t)ny * nx, stv = (size_t)(ny + 1) * nx;
    int ja = periodic ? (f % ny) : f;
    int jb = periodic ? ((f - 1 + ny) % ny) : f - 1;
    const size_t fc = (size_t)f * nx + i;
    float mo = face_mass(mut_old, (size_t)i, f, ny, nx, periodic, half);
    float mn = face_mass(mut_new, (size_t)i, f, ny, nx, periodic, half);
    if (legacy) mo = mn = legacy_face[fc];
    float msf = has_msf ? msfv[fc] : one;
    double gam[IEVA_KMAX], bb[IEVA_KMAX];
    double bet = 0.0, ct_prev = 0.0;
    for (int k = 0; k < nz; ++k) {
        int km1 = (k == 0) ? 0 : k - 1;
        int kp1 = (k == nz - 1) ? nz - 1 : k + 1;
        float mass_new = f_add(f_mul(c1h[k], mn), c2h[k]);
        float rl = f_mul(half, f_add(rom[(size_t)k * st + (size_t)jb * nx + i],
                                     rom[(size_t)k * st + (size_t)ja * nx + i]));
        float rr = f_mul(half, f_add(rom[(size_t)(k + 1) * st + (size_t)jb * nx + i],
                                     rom[(size_t)(k + 1) * st + (size_t)ja * nx + i]));
        float wiL = f_mul(rl, rdnw[k]);
        float wiR = f_mul(rr, rdnw[k]);
        if (has_msf) {
            wiL = f_mul(wiL, msf);
            wiR = f_mul(wiR, msf);
        }
        wiL = f_div(wiL, mass_new);
        wiR = f_div(wiR, mass_new);
        double at, ct, btmp;
        upwind_coefficients(wiL, wiR, dt, zero, at, ct, btmp);
        double bt = d_add((double)one, btmp);
        float mass_old = f_add(f_mul(c1h[k], mo), c2h[k]);
        float lead = f_mul(dt, tend[(size_t)k * stv + fc]);
        if (has_msf) lead = f_mul(lead, msf);
        double rt = upwind_rhs(lead, mass_old, at, btmp, ct,
                               v_old[(size_t)km1 * stv + fc],
                               v_old[(size_t)k * stv + fc],
                               v_old[(size_t)kp1 * stv + fc]);
        thomas_forward(k, 0, at, bt, ct_prev, rt, bet, gam, bb);
        ct_prev = ct;
    }
    thomas_backward(0, nz - 1, gam, bb);
    for (int k = 0; k < nz; ++k) {
        float out = f_div(__double2float_rn(bb[k]), dt);
        if (has_msf) out = f_div(out, msf);
        tend[(size_t)k * stv + fc] = out;
    }
}

// ---------------------------------------------------------------------------
// advect_ph_implicit (module_ieva_em.F:560-702).  One thread per mass
// column; interior w levels 1..nz-1.  Advective form: the centred wwI at
// the w level, upwinded, and the solve is for the perturbation increment
// with the base geopotential's advection kept on the right-hand side.
// ``tend`` is the geopotential tendency after rhs_ph's explicit terms
// (computed with wwE); mut is the stage's column mass.
extern "C" __global__
void ieva_solve_ph(float* __restrict__ tend,
                   const float* __restrict__ ph_old,
                   const float* __restrict__ phb, int base3d,
                   const float* __restrict__ rom,
                   const float* __restrict__ c1f,
                   const float* __restrict__ c2f,
                   const float* __restrict__ mut,
                   const float* __restrict__ rdnw,
                   const float* __restrict__ msft,
                   float dt, float one, float zero, float half,
                   int has_msf, int nz, int ny, int nx)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st || nz < 2) return;
    const float mt = mut[c];
    const float msf = has_msf ? msft[c] : one;
    double gam[IEVA_KMAX], bb[IEVA_KMAX];
    double bet = 0.0, ct_prev = 0.0;
    for (int k = 1; k < nz; ++k) {
        float mass = f_add(f_mul(c1f[k], mt), c2f[k]);
        // wiC = 0.5*wwI(k)*(rdzw(k-1)+rdzw(k)) * msfty / (c1(k)*mut+c2(k))
        float wiC = f_mul(f_mul(half, rom[(size_t)k * st + c]),
                          f_add(rdnw[k - 1], rdnw[k]));
        if (has_msf) wiC = f_mul(wiC, msf);
        wiC = f_div(wiC, mass);
        double at = (double)(-f_mul(dt, f_max(wiC, zero)));
        double ct = (double)f_mul(dt, f_min(wiC, zero));
        double btmp = d_sub(-at, ct);
        double bt = d_add((double)one, btmp);
        // rt = tendency*dt*msfty/(c1*mut+c2) - at*pho(k-1) - btmp*pho(k)
        //      - ct*pho(k+1) - at*phb(k-1) - btmp*phb(k) - ct*phb(k+1)
        float lead = f_mul(tend[(size_t)k * st + c], dt);
        if (has_msf) lead = f_mul(lead, msf);
        lead = f_div(lead, mass);
        float pm = ph_old[(size_t)(k - 1) * st + c];
        float p0 = ph_old[(size_t)k * st + c];
        float pp = ph_old[(size_t)(k + 1) * st + c];
        float bm = phb[base3d ? (size_t)(k - 1) * st + c : (size_t)(k - 1)];
        float b0 = phb[base3d ? (size_t)k * st + c : (size_t)k];
        float bp = phb[base3d ? (size_t)(k + 1) * st + c : (size_t)(k + 1)];
        double rt = (double)lead;
        rt = d_sub(rt, d_mul(at, (double)pm));
        rt = d_sub(rt, d_mul(btmp, (double)p0));
        rt = d_sub(rt, d_mul(ct, (double)pp));
        rt = d_sub(rt, d_mul(at, (double)bm));
        rt = d_sub(rt, d_mul(btmp, (double)b0));
        rt = d_sub(rt, d_mul(ct, (double)bp));
        thomas_forward(k, 1, at, bt, ct_prev, rt, bet, gam, bb);
        ct_prev = ct;
    }
    thomas_backward(1, nz - 1, gam, bb);
    for (int k = 1; k < nz; ++k) {
        float mass = f_add(f_mul(c1f[k], mt), c2f[k]);
        float out = f_div(f_mul(__double2float_rn(bb[k]), mass), dt);
        if (has_msf) out = f_div(out, msf);
        tend[(size_t)k * st + c] = out;
    }
}

// Terrain height one cell over, with set_w_surface's one-sided edge on a
// non-periodic axis (the forced outer row this reaches is never read by
// the acoustic w solve, which skips it).
static __device__ __forceinline__ float ht_at(
    const float* __restrict__ ht, int j, int i, int ny, int nx,
    int bx, int by)
{
    if (bx) { if (i < 0) i = 0; if (i > nx - 1) i = nx - 1; }
    else    { i = (i % nx + nx) % nx; }
    if (by) { if (j < 0) j = 0; if (j > ny - 1) j = ny - 1; }
    else    { j = (j % ny + ny) % ny; }
    return ht[(size_t)j * nx + i];
}

// A179: one level of a coupled u or v tendency as the acceleration it is,
// tend*msf/(c1h*mu_face + c2h) in m s-2: the map factor and face column
// mass WRF's small_step_finish uncouples u_2 and v_2 with
// (module_small_step_em.F:383, :392), multiplied, then divided, in that
// order.  Without map factors the product is skipped (x*1 is exact).
static __device__ __forceinline__ float uncoupled_tendency(
    float tend, float msf, int has_msf, float c1, float c2, float mu)
{
    float t = has_msf ? f_mul(tend, msf) : tend;
    return f_div(t, f_add(f_mul(c1, mu), c2));
}

// ---------------------------------------------------------------------------
// advect_w_implicit (module_ieva_em.F:1120-1271).  One thread per mass
// column; interior w levels 1..nz-1.  wwI is averaged to the half levels
// either side of the w level; the lower boundary adds the w increment the
// terrain slope makes of the u/v tendencies (ru_t/rv_t after their own
// implicit solves), the upper boundary the one the geopotential makes
// (rph_t after its implicit solve).
//
// DECLARED DIVERGENCE FROM WRF 4.7.1 (A179): both boundary terms are put in
// the units of the w_old terms beside them, m s-1 (times the column mass
// the right-hand side carries).  WRF's lower-boundary dw (:1231-1244) sums
// the COUPLED ru_tend/rv_tend (Pa m s-2), about one column mass too large:
// on the terrain clock's ridge probe it took the crest's w tendency from
// 4.4e4 to 9.3e7 and the forecast went NaN in step 3.  Here each level is
// uncoupled first (uncoupled_tendency).  WRF's upper-boundary dw (:1248-1253)
// takes (ph_new - ph_old)/dt_rk, m2 s-3, beside ph_tend/mass/g, m s-1; the
// first term is divided by g as well.  mux/muy are the stage face masses
// stage_fluxes couples ru/rv with; tools/ieva_wrf_oracle grades this kernel
// against WRF's own routine with the same two corrections.
extern "C" __global__
void ieva_solve_w(float* __restrict__ tend,
                  const float* __restrict__ utend,
                  const float* __restrict__ vtend,
                  const float* __restrict__ mux,
                  const float* __restrict__ muy,
                  const float* __restrict__ c1h,
                  const float* __restrict__ c2h,
                  const float* __restrict__ msfu,
                  const float* __restrict__ msfv,
                  const float* __restrict__ ht,
                  const float* __restrict__ rom,
                  const float* __restrict__ ph_new,
                  const float* __restrict__ ph_old,
                  const float* __restrict__ ph_tend,
                  const float* __restrict__ w_old,
                  const float* __restrict__ c1f,
                  const float* __restrict__ c2f,
                  float cf1, float cf2, float cf3,
                  const float* __restrict__ mut,
                  const float* __restrict__ mut_old,
                  const float* __restrict__ mut_new,
                  const float* __restrict__ rdn,
                  const float* __restrict__ msft,
                  float rdx, float rdy, float dt, float g,
                  float one, float zero, float half,
                  int has_msf, int bx, int by, int legacy,
                  int nz, int ny, int nx)
{
    size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (c >= st || nz < 2) return;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    const float mt = mut[c], mo = mut_old[c], mn = mut_new[c];
    const float msf = has_msf ? msft[c] : one;
    const int kend = nz - 1;
    double gam[IEVA_KMAX], bb[IEVA_KMAX];
    double bet = 0.0, ct_prev = 0.0;
    for (int k = 1; k <= kend; ++k) {
        float mass_new = f_add(f_mul(c1f[k], mn), c2f[k]);
        float rl = f_mul(half, f_add(rom[(size_t)(k - 1) * st + c],
                                     rom[(size_t)k * st + c]));
        float rr = f_mul(half, f_add(rom[(size_t)(k + 1) * st + c],
                                     rom[(size_t)k * st + c]));
        float wiL = f_mul(rl, rdn[k]);
        float wiR = f_mul(rr, rdn[k]);
        if (has_msf) {
            wiL = f_mul(wiL, msf);
            wiR = f_mul(wiR, msf);
        }
        wiL = f_div(wiL, mass_new);
        wiR = f_div(wiR, mass_new);
        double at, ct, btmp;
        upwind_coefficients(wiL, wiR, dt, zero, at, ct, btmp);
        double bt = d_add((double)one, btmp);
        float mass_old = f_add(f_mul(c1f[k], mo), c2f[k]);
        float lead = f_mul(dt, tend[(size_t)k * st + c]);
        if (has_msf) lead = f_mul(lead, msf);
        double rt = upwind_rhs(lead, mass_old, at, btmp, ct,
                               w_old[(size_t)(k - 1) * st + c],
                               w_old[(size_t)k * st + c],
                               w_old[(size_t)(k + 1) * st + c]);
        float mass = f_add(f_mul(c1f[k], mt), c2f[k]);
        if (k == 1) {
            // dw = msfty*.5*rdy*( (ht(j+1)-ht(j))*(cf1*vt(1,j+1)+cf2*vt(2,j+1)+cf3*vt(3,j+1))
            //                    +(ht(j)-ht(j-1))*(cf1*vt(1,j)+cf2*vt(2,j)+cf3*vt(3,j)) )
            //    + msftx*.5*rdx*( ... utend at i+1 and i ... )
            // with each vt(l) = vtend(l)*msfvx/(c1h(l)*muv+c2h(l)) and
            // ut(l) = utend(l)*msfuy/(c1h(l)*muu+c2h(l)) (A179; WRF reads
            // the coupled vtend(l) and utend(l) here).
            const size_t svl = (size_t)(ny + 1) * nx;
            const size_t vn = (size_t)(j + 1) * nx + i, vs = (size_t)j * nx + i;
            const float mvn = has_msf ? msfv[vn] : one, mvs = has_msf ? msfv[vs] : one;
            float vcn = f_add(f_add(
                f_mul(cf1, uncoupled_tendency(vtend[vn], mvn, has_msf, c1h[0], c2h[0], muy[vn])),
                f_mul(cf2, uncoupled_tendency(vtend[svl + vn], mvn, has_msf, c1h[1], c2h[1], muy[vn]))),
                f_mul(cf3, uncoupled_tendency(vtend[2 * svl + vn], mvn, has_msf, c1h[2], c2h[2], muy[vn])));
            float vcs = f_add(f_add(
                f_mul(cf1, uncoupled_tendency(vtend[vs], mvs, has_msf, c1h[0], c2h[0], muy[vs])),
                f_mul(cf2, uncoupled_tendency(vtend[svl + vs], mvs, has_msf, c1h[1], c2h[1], muy[vs]))),
                f_mul(cf3, uncoupled_tendency(vtend[2 * svl + vs], mvs, has_msf, c1h[2], c2h[2], muy[vs])));
            const size_t sul = (size_t)ny * (nx + 1);
            const size_t ue = (size_t)j * (nx + 1) + i + 1, uw = (size_t)j * (nx + 1) + i;
            const float mue = has_msf ? msfu[ue] : one, muw = has_msf ? msfu[uw] : one;
            float uce = f_add(f_add(
                f_mul(cf1, uncoupled_tendency(utend[ue], mue, has_msf, c1h[0], c2h[0], mux[ue])),
                f_mul(cf2, uncoupled_tendency(utend[sul + ue], mue, has_msf, c1h[1], c2h[1], mux[ue]))),
                f_mul(cf3, uncoupled_tendency(utend[2 * sul + ue], mue, has_msf, c1h[2], c2h[2], mux[ue])));
            float ucw = f_add(f_add(
                f_mul(cf1, uncoupled_tendency(utend[uw], muw, has_msf, c1h[0], c2h[0], mux[uw])),
                f_mul(cf2, uncoupled_tendency(utend[sul + uw], muw, has_msf, c1h[1], c2h[1], mux[uw]))),
                f_mul(cf3, uncoupled_tendency(utend[2 * sul + uw], muw, has_msf, c1h[2], c2h[2], mux[uw])));
            float h0 = ht[c];
            float hn = ht_at(ht, j + 1, i, ny, nx, bx, by);
            float hs = ht_at(ht, j - 1, i, ny, nx, bx, by);
            float he = ht_at(ht, j, i + 1, ny, nx, bx, by);
            float hw = ht_at(ht, j, i - 1, ny, nx, bx, by);
            float yterm = f_add(f_mul(f_sub(hn, h0), vcn), f_mul(f_sub(h0, hs), vcs));
            float xterm = f_add(f_mul(f_sub(he, h0), uce), f_mul(f_sub(h0, hw), ucw));
            float dw = f_add(f_mul(f_mul(f_mul(msf, half), rdy), yterm),
                             f_mul(f_mul(f_mul(msf, half), rdx), xterm));
            // rt = rt - (c1(k)*mut+c2(k))*at*dt_rk*dw
            rt = d_sub(rt, d_mul(d_mul(d_mul((double)mass, at), (double)dt),
                                 (double)dw));
        }
        if (k == kend) {
            // dw = msfty*((ph_new(k+1)-ph_old(k+1))/dt_rk/g - ph_tend(k+1)/(c1(k)*mut+c2(k))/g)
            // (A179: WRF's first term is not divided by g, m2 s-3 beside m s-1.)
            const size_t top = (size_t)(k + 1) * st + c;
            float dph = f_div(f_div(f_sub(ph_new[top], ph_old[top]), dt), g);
            float pt = f_div(f_div(ph_tend[top], mass), g);
            float dw = f_sub(dph, pt);
            if (has_msf) dw = f_mul(msf, dw);
            if (legacy) {
                // Earlier WRF already divides the complete boundary
                // increment by g. Preserve that FP32 grouping exactly.
                dph = f_div(f_sub(ph_new[top], ph_old[top]), dt);
                pt = f_div(ph_tend[top], mass);
                dw = f_sub(dph, pt);
                if (has_msf) dw = f_mul(msf, dw);
                dw = f_div(dw, g);
            }
            // rt = rt - (c1(k)*mut+c2(k))*ct*(dw - w_old(k+1))
            rt = d_sub(rt, d_mul(d_mul((double)mass, ct),
                                 (double)f_sub(dw, w_old[top])));
        }
        thomas_forward(k, 1, at, bt, ct_prev, rt, bet, gam, bb);
        ct_prev = ct;
    }
    thomas_backward(1, kend, gam, bb);
    for (int k = 1; k <= kend; ++k) {
        float out = f_div(__double2float_rn(bb[k]), dt);
        if (has_msf) out = f_div(out, msf);
        tend[(size_t)k * st + c] = out;
    }
}
