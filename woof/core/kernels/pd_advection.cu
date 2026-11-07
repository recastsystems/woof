// gpuwm/core/kernels/pd_advection.cu
//
// Positive-definite scalar transport (WRF module_advect_em.F
// advect_scalar_pd; Skamarock 2006 MWR), applied on the RK3 FINAL stage
// only, exactly as WRF does.  Two kernels:
//
//   pd_fluxes        per face: the CFL-clamped 1st-order upwind flux of the
//                    time-t scalar q0 (WRF field_old) and the correction
//                    F_corr = F_high - F_upwind1, where F_high is bitwise
//                    the unlimited flux of advection.cu (5th-order upwind
//                    horizontal; vertical WRF vert_order ladder selected
//                    by the trailing vorder argument, 3 or 5, with
//                    2nd-order faces one cell in from the eta boundaries)
//                    evaluated on the RK stage estimate q.  Upwind Courant numbers divide by
//                    the hybrid face mass c1h*<mu>_face + c2h (mut = WRF
//                    muts, the post-acoustic stage mass); the eta face
//                    spacing dz = 2/(rdnw[kf]+rdnw[kf-1]) is negative, so
//                    the Omega-signed upwinding follows automatically.
//
//   pd_renorm_apply  per mass cell: WRF's renormalization -- compute the
//                    upwind-updated coupled scalar ph_low = (c1h*mu_old +
//                    c2h)*q0 - dt*div(F_low) (positive by construction) and
//                    the total outgoing correction flux_out; where
//                    flux_out > ph_low, ALL outgoing correction fluxes of
//                    that cell are scaled by r = ph_low/(flux_out + eps),
//                    clamped [0,1].  Each face flux drains exactly one
//                    donor cell (by its sign), so the donor's scale is
//                    recomputed on the fly (race-free, deterministic) and
//                    the kernel ADDS -div(F_low + r*F_corr) into tend_out.
//
// x and y are periodic by default (redundant staggered faces nx / ny are
// computed with wrapped stencils, hence bitwise equal to face 0); boundary
// eta faces carry zero flux.  Mirrors: gpuwm/verify/npref.py np_pd_fluxes /
// np_pd_renorm_apply.
//
// Specified/open lateral boundaries (Phase 4 transport FIX-C; WRF
// advect_scalar_pd fully supports them): with open_x/open_y (the dycore
// maps specified onto them),
//   - the boundary-normal faces 0/nx (0/ny) carry zero flux and the
//     near-boundary high-order fluxes degrade to 2nd order one face in
//     and WRF's horizontal flux3 two faces in, exactly like
//     advect_scalar (the degrade_xs/xe/ys/ye bands, F:6069ff blocks);
//   - the limiter skips the outermost cells: WRF's specified AND open
//     bounds both set i_start=ids+1, i_end=ide-2 / j analogues
//     (module_advect_em.F:7697-7715), so boundary cells never
//     renormalize (pd_scale returns 1 there);
//   - the applied tendency gives boundary cells the vertical PD
//     divergence only: the x divergence covers ids+1..ide-2
//     (F:7817-7821), the y divergence jds+1..jde-2 (F:7852-7856), the
//     vertical the full ids..ide-1 (F:7787-7791).
// The periodic path (open_x = open_y = 0) keeps the ORIGINAL expressions
// verbatim in dedicated branches (bitwise-pinned by the phase2 moist
// regression).
//
// Map-scale factors (Phase 3 Task 3; WRF advect_scalar_pd): when
// has_msf != 0, the horizontal upwind fluxes/Courant numbers use the
// PHYSICAL face spacing dx*2/(msft_A + msft_B) (WRF "ADT eqn 48 d/dx";
// the Fortran's msfty average at an x face equals the isotropic face
// msft), ph_low/flux_out weight their horizontal divergence by
// msftx*msfty = msft^2 and the vertical by msfty = msft, and the applied
// tendency weights the horizontal divergence by msftx = msft ("un-canceled
// map scale factor").  has_msf == 0 keeps the original expressions
// verbatim (bitwise Phase 2, regression-pinned).

#ifndef GPUWM_HORDER
#define GPUWM_HORDER 5
#endif

// Periodic indices retain the original normalization in the exact build.
__device__ __forceinline__
int pd_periodic(int index, int extent)
{
#if GPUWM_WRF_EXACT_C_ADVECTION
    return PERIODIC(index, extent);
#else
    if ((unsigned int)index < (unsigned int)extent) return index;
    int wrapped = index % extent;
    return wrapped < 0 ? wrapped + extent : wrapped;
#endif
}

// --- flux operators, duplicated from advection.cu (modules compile
// --- standalone); any change must be made in both files and the mirrors.
__device__ __forceinline__
real pd_flux5(real qm3, real qm2, real qm1, real q0, real qp1, real qp2,
              real vel)
{
#if GPUWM_HORDER == 2
    return 0.5f*vel*(q0+qm1);
#elif GPUWM_HORDER == 3
    real center=0.5833333134651184f*(q0+qm1)-0.0833333358168602f*(qp1+qm2);
    real dissipation=copysignf(1.0f,vel)*0.0833333358168602f*((qp1-qm2)-3.0f*(q0-qm1));
    return vel*(center+dissipation);
#elif GPUWM_HORDER == 4
    return vel*(0.5833333134651184f*(q0+qm1)-0.0833333358168602f*(qp1+qm2));
#elif GPUWM_HORDER == 6
    return vel*(0.6166666746139526f*(q0+qm1)-0.13333334028720856f*(qp1+qm2)+0.01666666753590107f*(qp2+qm3));
#else
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = 0.6166666746139526f * (q0 + qm1)
                - 0.13333334028720856f * (qp1 + qm2)
                + 0.01666666753590107f * (qp2 + qm3);
    real dissipation = copysignf(1.0f, vel) * 0.01666666753590107f
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1));
    return vel * (center - dissipation);
#else
    return __fdiv_rn((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            - fabsf(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
#endif
#endif
}

__device__ __forceinline__
real pd_flux3(real qm2, real qm1, real q0, real qp1, real vel)
{
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = 0.5833333134651184f * (q0 + qm1)
                - 0.0833333358168602f * (qp1 + qm2);
    real dissipation = copysignf(1.0f, vel) * 0.0833333358168602f
        * ((qp1 - qm2) - 3.0f * (q0 - qm1));
    return vel * (center - dissipation);
#else
    return __fdiv_rn((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            + fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
#endif
}

// Vertical 5th-order face flux: WRF advect_scalar_pd's flux5 with -vel
// (HRRR fork module_advect_em.F F:8412-8413, the vert_order == 5 arm), the
// Omega-signed dissipation of pd_flux3 above; mirrors advection.cu flux5v.
__device__ __forceinline__
real pd_flux5v(real qm3, real qm2, real qm1, real q0, real qp1, real qp2,
               real vel)
{
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = 0.6166666746139526f * (q0 + qm1)
                - 0.13333334028720856f * (qp1 + qm2)
                + 0.01666666753590107f * (qp2 + qm3);
    real dissipation = copysignf(1.0f, vel) * 0.01666666753590107f
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1));
    return vel * (center + dissipation);
#else
    return __fdiv_rn((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            + fabsf(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
#endif
}

// Horizontal 3rd-order face flux (WRF flux3 with the flux5 upwinding
// sign) -- used two faces in from a specified/open lateral boundary.
__device__ __forceinline__
real pd_flux3h(real qm2, real qm1, real q0, real qp1, real vel)
{
#if GPUWM_HORDER == 4 || GPUWM_HORDER == 6
    return vel*(0.5833333134651184f*(q0+qm1)-0.0833333358168602f*(qp1+qm2));
#else
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = 0.5833333134651184f * (q0 + qm1)
                - 0.0833333358168602f * (qp1 + qm2);
    real dissipation = copysignf(1.0f, vel) * 0.0833333358168602f
        * ((qp1 - qm2) - 3.0f * (q0 - qm1));
    return vel * (center + dissipation);
#else
    return __fdiv_rn((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            - fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
#endif
#endif
}

// WRF's stretched-grid fnm/fnp weights at the 2nd-order eta faces
// (advect_scalar_pd fqz = rom*(fzm(k)*field(k)+fzp(k)*field(k-1)) at
// k=kts+1 and k=ktf, module_advect_em.F:7631/:7641); 0.5/0.5 on a uniform
// grid, bitwise.  Then the vorder ladder of advection.cu zface_half:
// 3 is pd_flux3 between; 5 is WRF's vert_order == 5 arm of advect_scalar_pd
// (HRRR fork F:8378-8501): pd_flux3 at faces 2 and nz-2 (k = kts+2, ktf-1),
// pd_flux5v at faces 3 .. nz-3 (k = kts+3 .. ktf-2).
__device__ __forceinline__
real pd_zface_half(const real* q, real vel, int kf, int j, int i,
                   int nz, int ny, int nxs,
                   const real* fnm, const real* fnp, int vorder)
{
    if (kf == 0 || kf == nz) return 0.0f;
    if (vorder == 2 || kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[I3S(kf, j, i, ny, nxs)]
                      + fnp[kf] * q[I3S(kf - 1, j, i, ny, nxs)]);
    if (vorder == 6 && kf >= 3 && kf <= nz - 3)
        return vel * (0.6166666746139526f *
                          (q[I3S(kf,j,i,ny,nxs)] + q[I3S(kf-1,j,i,ny,nxs)])
                      - 0.13333334028720856f *
                          (q[I3S(kf+1,j,i,ny,nxs)] + q[I3S(kf-2,j,i,ny,nxs)])
                      + 0.01666666753590107f *
                          (q[I3S(kf+2,j,i,ny,nxs)] + q[I3S(kf-3,j,i,ny,nxs)]));
    if (vorder == 4 || vorder == 6)
        return vel * (0.5833333134651184f *
                          (q[I3S(kf,j,i,ny,nxs)] + q[I3S(kf-1,j,i,ny,nxs)])
                      - 0.0833333358168602f *
                          (q[I3S(kf+1,j,i,ny,nxs)] + q[I3S(kf-2,j,i,ny,nxs)]));
    if (vorder == 5 && kf >= 3 && kf <= nz - 3)
        return pd_flux5v(q[I3S(kf - 3, j, i, ny, nxs)],
                         q[I3S(kf - 2, j, i, ny, nxs)],
                         q[I3S(kf - 1, j, i, ny, nxs)],
                         q[I3S(kf,     j, i, ny, nxs)],
                         q[I3S(kf + 1, j, i, ny, nxs)],
                         q[I3S(kf + 2, j, i, ny, nxs)], vel);
    return pd_flux3(q[I3S(kf - 2, j, i, ny, nxs)],
                    q[I3S(kf - 1, j, i, ny, nxs)],
                    q[I3S(kf,     j, i, ny, nxs)],
                    q[I3S(kf + 1, j, i, ny, nxs)], vel);
}

// WRF flux_upwind: CFL-clamped 1st-order upwind face value * Courant
// number; face f lies between cells f-1 (qm1) and f (q0v).
__device__ __forceinline__
real flux_upwind(real qm1, real q0v, real cr)
{
    return 0.5f * fminf(1.0f, cr + fabsf(cr)) * qm1
         + 0.5f * fmaxf(-1.0f, cr - fabsf(cr)) * q0v;
}

// ---------------------------------------------------------------------------
// Kernel 1: flux decomposition.  Launched over faces: i = 0..nx (fastest),
// j = 0..ny, k = 0..nz; each thread computes the x face (j<ny, k<nz), the
// y face (i<nx, k<nz), and the eta face (i<nx, j<ny) at its index.
// ---------------------------------------------------------------------------
extern "C" __global__
void pd_fluxes(const real* __restrict__ q,      // (nz, ny, nx) stage estimate
               const real* __restrict__ q0,     // (nz, ny, nx) time-t scalar
               const real* __restrict__ ru,     // (nz, ny, nx+1) coupled flux
               const real* __restrict__ rv,     // (nz, ny+1, nx)
               const real* __restrict__ rw,     // (nz+1, ny, nx) Omega-signed
               const real* __restrict__ mut,    // (ny, nx) stage column mass
               const real* __restrict__ c1h,    // (nz,)
               const real* __restrict__ c2h,    // (nz,)
               const real* __restrict__ rdnw,   // (nz,) 1/dnw (< 0)
               const real* __restrict__ fnm,    // (nz,) eta-face weights
               const real* __restrict__ fnp,    // (nz,)
               const real* __restrict__ msft,   // (ny, nx) mass-pt map factor
               real dx, real dy, real dt,
               real* __restrict__ fxl,          // (nz, ny, nx+1) upwind
               real* __restrict__ fxc,          // (nz, ny, nx+1) correction
               real* __restrict__ fyl,          // (nz, ny+1, nx)
               real* __restrict__ fyc,          // (nz, ny+1, nx)
               real* __restrict__ fzl,          // (nz+1, ny, nx)
               real* __restrict__ fzc,          // (nz+1, ny, nx)
               int nz, int ny, int nx, int has_msf,
               int open_x, int open_y, int vorder)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    if (j < ny && k < nz) {                       // x face f = i (0..nx)
        if (open_x && (i == 0 || i == nx)) {      // boundary-normal faces:
            fxl[I3(k, j, i, ny, nx + 1)] = 0.0f;  // never consumed (WRF
            fxc[I3(k, j, i, ny, nx + 1)] = 0.0f;  // specified/open bounds)
        } else if (open_x) {                      // degraded, no wrap
            real vel = ru[I3(k, j, i, ny, nx + 1)];
            #if GPUWM_WRF_EXACT_C_ADVECTION
real muf=0.5f*((c1h[k]*mut[(size_t)j * nx + i]+c2h[k])+(c1h[k]*mut[(size_t)j * nx + i - 1]+c2h[k]));
#else
real muf = c1h[k] * 0.5f * (mut[(size_t)j * nx + i]
                                        + mut[(size_t)j * nx + i - 1])
                     + c2h[k];
#endif
            real dxp = dx;
            if (has_msf)
                #if GPUWM_WRF_EXACT_C_ADVECTION
dxp=__fdiv_rn(__fdiv_rn(2.0f,msft[(size_t)j * nx + i]+msft[(size_t)j * nx + i - 1]),__fdiv_rn(1.0f,dx));
#else
dxp = dx * 2.0f / (msft[(size_t)j * nx + i]
                                   + msft[(size_t)j * nx + i - 1]);
#endif
            real cr = vel * dt / dxp / muf;
#if GPUWM_WRF_EXACT_C_ADVECTION
            // The western second-order face divides the velocity by its
            // face mass before constructing its Courant number in WRF.
            if (i == 1 && GPUWM_HORDER != 2) cr = __fdiv_rn(__fdiv_rn(vel, muf) * dt, dxp);
#endif
            real fl = muf * (dxp / dt)
                    * flux_upwind(q0[IDX3(k, j, i - 1)],
                                  q0[IDX3(k, j, i)], cr);
            int d = min(i, nx - i);
            real fh;
            if (d == 1 || GPUWM_HORDER == 2)                           // 2nd order at the edge
                fh = 0.5f * vel * (q[IDX3(k, j, i)] + q[IDX3(k, j, i - 1)]);
            else if (d == 2)                      // 3rd order one face in
                fh = pd_flux3h(q[IDX3(k, j, i - 2)], q[IDX3(k, j, i - 1)],
                               q[IDX3(k, j, i)], q[IDX3(k, j, i + 1)], vel);
            else
                fh = pd_flux5(q[IDX3(k, j, i - 3)], q[IDX3(k, j, i - 2)],
                              q[IDX3(k, j, i - 1)], q[IDX3(k, j, i)],
                              q[IDX3(k, j, i + 1)], q[IDX3(k, j, i + 2)],
                              vel);
            fxl[I3(k, j, i, ny, nx + 1)] = fl;
            fxc[I3(k, j, i, ny, nx + 1)] = fh - fl;
        } else {                                  // ORIGINAL periodic path
            real vel = ru[I3(k, j, i, ny, nx + 1)];
            #if GPUWM_WRF_EXACT_C_ADVECTION
real muf=0.5f*((c1h[k]*mut[(size_t)j * nx + pd_periodic(i, nx)]+c2h[k])+(c1h[k]*mut[(size_t)j * nx + pd_periodic(i - 1, nx)]+c2h[k]));
#else
real muf = c1h[k] * 0.5f * (mut[(size_t)j * nx + pd_periodic(i, nx)]
                                        + mut[(size_t)j * nx + pd_periodic(i - 1, nx)])
                     + c2h[k];
#endif
            real dxp = dx;                        // physical face spacing
            if (has_msf)                          // WRF: 2/(msf_A+msf_B)/rdx
                #if GPUWM_WRF_EXACT_C_ADVECTION
dxp=__fdiv_rn(__fdiv_rn(2.0f,msft[(size_t)j * nx + pd_periodic(i, nx)]+msft[(size_t)j * nx + pd_periodic(i - 1, nx)]),__fdiv_rn(1.0f,dx));
#else
dxp = dx * 2.0f / (msft[(size_t)j * nx + pd_periodic(i, nx)]
                                   + msft[(size_t)j * nx + pd_periodic(i - 1, nx)]);
#endif
            real cr = vel * dt / dxp / muf;
            real fl = muf * (dxp / dt)
                    * flux_upwind(q0[IDX3(k, j, pd_periodic(i - 1, nx))],
                                  q0[IDX3(k, j, pd_periodic(i, nx))], cr);
            real fh = pd_flux5(q[IDX3(k, j, pd_periodic(i - 3, nx))],
                               q[IDX3(k, j, pd_periodic(i - 2, nx))],
                               q[IDX3(k, j, pd_periodic(i - 1, nx))],
                               q[IDX3(k, j, pd_periodic(i,     nx))],
                               q[IDX3(k, j, pd_periodic(i + 1, nx))],
                               q[IDX3(k, j, pd_periodic(i + 2, nx))], vel);
            fxl[I3(k, j, i, ny, nx + 1)] = fl;
            fxc[I3(k, j, i, ny, nx + 1)] = fh - fl;
        }
    }

    if (i < nx && k < nz) {                       // y face g = j (0..ny)
        if (open_y && (j == 0 || j == ny)) {
            fyl[I3(k, j, i, ny + 1, nx)] = 0.0f;
            fyc[I3(k, j, i, ny + 1, nx)] = 0.0f;
        } else if (open_y) {                      // degraded, no wrap
            real vel = rv[I3(k, j, i, ny + 1, nx)];
            #if GPUWM_WRF_EXACT_C_ADVECTION
real muf=0.5f*((c1h[k]*mut[(size_t)j * nx + i]+c2h[k])+(c1h[k]*mut[(size_t)(j - 1) * nx + i]+c2h[k]));
#else
real muf = c1h[k] * 0.5f * (mut[(size_t)j * nx + i]
                                        + mut[(size_t)(j - 1) * nx + i])
                     + c2h[k];
#endif
            real dyp = dy;
            if (has_msf)
                #if GPUWM_WRF_EXACT_C_ADVECTION
dyp=__fdiv_rn(__fdiv_rn(2.0f,msft[(size_t)j * nx + i]+msft[(size_t)(j - 1) * nx + i]),__fdiv_rn(1.0f,dy));
#else
dyp = dy * 2.0f / (msft[(size_t)j * nx + i]
                                   + msft[(size_t)(j - 1) * nx + i]);
#endif
            real cr = vel * dt / dyp / muf;
            real fl = muf * (dyp / dt)
                    * flux_upwind(q0[IDX3(k, j - 1, i)],
                                  q0[IDX3(k, j, i)], cr);
            int d = min(j, ny - j);
            real fh;
            if (d == 1 || GPUWM_HORDER == 2)
                fh = 0.5f * vel * (q[IDX3(k, j, i)] + q[IDX3(k, j - 1, i)]);
            else if (d == 2)
                fh = pd_flux3h(q[IDX3(k, j - 2, i)], q[IDX3(k, j - 1, i)],
                               q[IDX3(k, j, i)], q[IDX3(k, j + 1, i)], vel);
            else
                fh = pd_flux5(q[IDX3(k, j - 3, i)], q[IDX3(k, j - 2, i)],
                              q[IDX3(k, j - 1, i)], q[IDX3(k, j, i)],
                              q[IDX3(k, j + 1, i)], q[IDX3(k, j + 2, i)],
                              vel);
            fyl[I3(k, j, i, ny + 1, nx)] = fl;
            fyc[I3(k, j, i, ny + 1, nx)] = fh - fl;
        } else {                                  // ORIGINAL periodic path
            real vel = rv[I3(k, j, i, ny + 1, nx)];
            #if GPUWM_WRF_EXACT_C_ADVECTION
real muf=0.5f*((c1h[k]*mut[(size_t)pd_periodic(j, ny) * nx + i]+c2h[k])+(c1h[k]*mut[(size_t)pd_periodic(j - 1, ny) * nx + i]+c2h[k]));
#else
real muf = c1h[k] * 0.5f * (mut[(size_t)pd_periodic(j, ny) * nx + i]
                                        + mut[(size_t)pd_periodic(j - 1, ny) * nx + i])
                     + c2h[k];
#endif
            real dyp = dy;
            if (has_msf)
                #if GPUWM_WRF_EXACT_C_ADVECTION
dyp=__fdiv_rn(__fdiv_rn(2.0f,msft[(size_t)pd_periodic(j, ny) * nx + i]+msft[(size_t)pd_periodic(j - 1, ny) * nx + i]),__fdiv_rn(1.0f,dy));
#else
dyp = dy * 2.0f / (msft[(size_t)pd_periodic(j, ny) * nx + i]
                                   + msft[(size_t)pd_periodic(j - 1, ny) * nx + i]);
#endif
            real cr = vel * dt / dyp / muf;
            real fl = muf * (dyp / dt)
                    * flux_upwind(q0[IDX3(k, pd_periodic(j - 1, ny), i)],
                                  q0[IDX3(k, pd_periodic(j, ny), i)], cr);
            real fh = pd_flux5(q[IDX3(k, pd_periodic(j - 3, ny), i)],
                               q[IDX3(k, pd_periodic(j - 2, ny), i)],
                               q[IDX3(k, pd_periodic(j - 1, ny), i)],
                               q[IDX3(k, pd_periodic(j,     ny), i)],
                               q[IDX3(k, pd_periodic(j + 1, ny), i)],
                               q[IDX3(k, pd_periodic(j + 2, ny), i)], vel);
            fyl[I3(k, j, i, ny + 1, nx)] = fl;
            fyc[I3(k, j, i, ny + 1, nx)] = fh - fl;
        }
    }

    if (i < nx && j < ny) {                       // eta face kf = k (0..nz)
        real fl = 0.0f, fc = 0.0f;
        if (k > 0 && k < nz) {
            real vel = rw[IDX3(k, j, i)];
            real dz = 2.0f / (rdnw[k] + rdnw[k - 1]);        // < 0
            real muf = c1h[k] * mut[(size_t)j * nx + i] + c2h[k];
#if GPUWM_WRF_EXACT_C_ADVECTION
            real cr = __fdiv_rn(__fdiv_rn(vel * dt, dz), muf);
#else
            real cr = vel * dt / (dz * muf);
#endif
            fl = muf * (dz / dt)
               * flux_upwind(q0[IDX3(k - 1, j, i)], q0[IDX3(k, j, i)], cr);
            fc = pd_zface_half(q, vel, k, j, i, nz, ny, nx, fnm, fnp, vorder) - fl;
        }
        fzl[IDX3(k, j, i)] = fl;
        fzc[IDX3(k, j, i)] = fc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: renormalize and apply.  pd_scale computes WRF's per-cell
// renormalization factor from the stored fluxes; the main kernel evaluates
// it for the cell and (on demand) its six face-donor neighbors.
// ---------------------------------------------------------------------------
__device__
real pd_scale(int k, int j, int i,
              const real* q0, const real* mu_old,
              const real* c1h, const real* c2h, const real* rdnw,
              const real* fxl, const real* fxc,
              const real* fyl, const real* fyc,
              const real* fzl, const real* fzc,
              const real* msft,
              real dx_inv, real dy_inv, real dt, int ny, int nx,
              int has_msf, int open_x, int open_y
#if GPUWM_WRF_EXACT_C_ADVECTION
              ,const real* mub, const real* mup_old
#endif
              )
{
    // WRF limiter bounds: specified AND open both exclude the outermost
    // cells (module_advect_em.F:7697-7715) -- their fluxes are never
    // renormalized.
    if ((open_x && (i == 0 || i == nx - 1))
        || (open_y && (j == 0 || j == ny - 1)))
        return 1.0f;
#if GPUWM_WRF_EXACT_C_ADVECTION
    real chm0=(c1h[k]*mub[(size_t)j*nx+i]+c2h[k])+(c1h[k]*mup_old[(size_t)j*nx+i]);
#else
    real chm0 = c1h[k] * mu_old[(size_t)j * nx + i] + c2h[k];
#endif
    real ph_low, fo;
    if (has_msf) {                       // WRF: msftx*msfty on horizontal,
        real m = msft[(size_t)j * nx + i];   // msfty on vertical divergence
        real m2 = m * m;
        ph_low = chm0 * q0[IDX3(k, j, i)]
               - dt * (m2 * (dx_inv * (fxl[I3(k, j, i + 1, ny, nx + 1)]
                                       - fxl[I3(k, j, i, ny, nx + 1)])
                             + dy_inv * (fyl[I3(k, j + 1, i, ny + 1, nx)]
                                         - fyl[I3(k, j, i, ny + 1, nx)]))
                       + m * rdnw[k] * (fzl[IDX3(k + 1, j, i)]
                                        - fzl[IDX3(k, j, i)]));
        fo = dt * (m2 * (dx_inv * (fmaxf(0.0f, fxc[I3(k, j, i + 1, ny, nx + 1)])
                                   - fminf(0.0f, fxc[I3(k, j, i, ny, nx + 1)]))
                         + dy_inv * (fmaxf(0.0f, fyc[I3(k, j + 1, i, ny + 1, nx)])
                                     - fminf(0.0f, fyc[I3(k, j, i, ny + 1, nx)])))
                   + m * rdnw[k] * (fminf(0.0f, fzc[IDX3(k + 1, j, i)])
                                    - fmaxf(0.0f, fzc[IDX3(k, j, i)])));
    } else {                             // original: bitwise Phase 2
        real dl = dx_inv * (fxl[I3(k, j, i + 1, ny, nx + 1)]
                            - fxl[I3(k, j, i, ny, nx + 1)])
                + dy_inv * (fyl[I3(k, j + 1, i, ny + 1, nx)]
                            - fyl[I3(k, j, i, ny + 1, nx)])
                + rdnw[k] * (fzl[IDX3(k + 1, j, i)] - fzl[IDX3(k, j, i)]);
        ph_low = chm0 * q0[IDX3(k, j, i)] - dt * dl;
        fo = dt * (dx_inv * (fmaxf(0.0f, fxc[I3(k, j, i + 1, ny, nx + 1)])
                             - fminf(0.0f, fxc[I3(k, j, i, ny, nx + 1)]))
                   + dy_inv * (fmaxf(0.0f, fyc[I3(k, j + 1, i, ny + 1, nx)])
                               - fminf(0.0f, fyc[I3(k, j, i, ny + 1, nx)]))
                   + rdnw[k] * (fminf(0.0f, fzc[IDX3(k + 1, j, i)])
                                - fmaxf(0.0f, fzc[IDX3(k, j, i)])));
    }
    if (fo > ph_low)
#if GPUWM_WRF_EXACT_C_ADVECTION
        return fmaxf(0.0f, __fdiv_rn(ph_low,fo+1e-20f));
#else
        return fminf(1.0f, fmaxf(0.0f, ph_low / (fo + 1e-20f)));
#endif
    return 1.0f;
}

extern "C" __global__
void pd_renorm_apply(const real* __restrict__ q0,      // (nz, ny, nx)
                     const real* __restrict__ mu_old,  // (ny, nx) time-t mass
                     const real* __restrict__ fxl,
                     const real* __restrict__ fxc,
                     const real* __restrict__ fyl,
                     const real* __restrict__ fyc,
                     const real* __restrict__ fzl,
                     const real* __restrict__ fzc,
                     const real* __restrict__ c1h,
                     const real* __restrict__ c2h,
                     const real* __restrict__ rdnw,
                     const real* __restrict__ msft,    // (ny, nx)
                     real dx_inv, real dy_inv, real dt,
                     real* __restrict__ tend_out,      // (nz, ny, nx) +=
                     int nz, int ny, int nx, int has_msf,
                     int open_x, int open_y
#if GPUWM_WRF_EXACT_C_ADVECTION
                     ,const real* __restrict__ mub, const real* __restrict__ mup_old
#endif
                     )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
#if GPUWM_WRF_EXACT_C_ADVECTION
    if (i >= nx || j >= ny || k >= nz) return;
#else
    unsigned int active = __ballot_sync(0xffffffffu,
                                        i < nx && j < ny && k < nz);
    if (i >= nx || j >= ny || k >= nz) return;
#endif

#if GPUWM_WRF_EXACT_C_ADVECTION
#define PD_SCALE(kk, jj, ii) pd_scale((kk), (jj), (ii), q0, mu_old, c1h,   \
        c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv,     \
        dt, ny, nx, has_msf, open_x, open_y,mub,mup_old)
#else
#define PD_SCALE(kk, jj, ii) pd_scale((kk), (jj), (ii), q0, mu_old, c1h,   \
        c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv,     \
        dt, ny, nx, has_msf, open_x, open_y)
#endif

    real sc = PD_SCALE(k, j, i);

#if GPUWM_WRF_EXACT_C_ADVECTION
    real fxc_l = fxc[I3(k, j, i, ny, nx + 1)];
    real sx_l = (fxc_l > 0.0f) ? PD_SCALE(k, j, pd_periodic(i - 1, nx))
              : ((fxc_l < 0.0f) ? sc : 1.0f);
    real fxc_r = fxc[I3(k, j, i + 1, ny, nx + 1)];
    real sx_r = (fxc_r > 0.0f) ? sc
              : ((fxc_r < 0.0f) ? PD_SCALE(k, j, pd_periodic(i + 1, nx)) : 1.0f);

    real fyc_l = fyc[I3(k, j, i, ny + 1, nx)];
    real sy_l = (fyc_l > 0.0f) ? PD_SCALE(k, pd_periodic(j - 1, ny), i)
              : ((fyc_l < 0.0f) ? sc : 1.0f);
    real fyc_r = fyc[I3(k, j + 1, i, ny + 1, nx)];
    real sy_r = (fyc_r > 0.0f) ? sc
              : ((fyc_r < 0.0f) ? PD_SCALE(k, pd_periodic(j + 1, ny), i) : 1.0f);

#else
    // Adjacent x donors already evaluate the identical cell scale in
    // this warp. Exchange the result without changing its arithmetic;
    // warp edges and periodic seams retain the original evaluation.
    int lane = threadIdx.x & 31;
    real sc_l = __shfl_up_sync(active, sc, 1);
    real sc_r = __shfl_down_sync(active, sc, 1);
    bool have_l = lane > 0 && (active & (1u << (lane - 1)));
    bool have_r = lane < 31 && (active & (1u << (lane + 1)));

    real fxc_l = fxc[I3(k, j, i, ny, nx + 1)];
    real sx_l = (fxc_l > 0.0f) ? (have_l ? sc_l
                                         : PD_SCALE(k, j, pd_periodic(i - 1, nx)))
              : ((fxc_l < 0.0f) ? sc : 1.0f);
    real fxc_r = fxc[I3(k, j, i + 1, ny, nx + 1)];
    real sx_r = (fxc_r > 0.0f) ? sc
              : ((fxc_r < 0.0f) ? (have_r ? sc_r
                                        : PD_SCALE(k, j, pd_periodic(i + 1, nx)))
                                 : 1.0f);

    real fyc_l = fyc[I3(k, j, i, ny + 1, nx)];
    real sy_l = (fyc_l > 0.0f) ? PD_SCALE(k, pd_periodic(j - 1, ny), i)
              : ((fyc_l < 0.0f) ? sc : 1.0f);
    real fyc_r = fyc[I3(k, j + 1, i, ny + 1, nx)];
    real sy_r = (fyc_r > 0.0f) ? sc
              : ((fyc_r < 0.0f) ? PD_SCALE(k, pd_periodic(j + 1, ny), i) : 1.0f);

#endif

    // eta face kf: positive (downward) drains the upper cell kf, negative
    // (upward, Omega-signed) drains the lower cell kf-1.  Boundary faces
    // carry zero flux, so the out-of-range donors are never evaluated.
    real fzc_b = fzc[IDX3(k, j, i)];
    real sz_b = (fzc_b > 0.0f) ? sc
              : ((fzc_b < 0.0f) ? PD_SCALE(k - 1, j, i) : 1.0f);
    real fzc_t = fzc[IDX3(k + 1, j, i)];
    real sz_t = (fzc_t < 0.0f) ? sc
              : ((fzc_t > 0.0f) ? PD_SCALE(k + 1, j, i) : 1.0f);
#undef PD_SCALE
#if GPUWM_WRF_EXACT_C_ADVECTION

    // The native limiter first advances z, then x, then y.  Correction
    // differences and the two low-order fluxes retain their round points.
    real result=tend_out[IDX3(k,j,i)];
    real zbracket=sz_t*fzc_t-sz_b*fzc_b+fzl[IDX3(k+1,j,i)]-fzl[IDX3(k,j,i)];
    result=result-rdnw[k]*zbracket;
    real m=has_msf?msft[(size_t)j*nx+i]:1.0f;
    if(!open_x || (i>=1 && i<=nx-2)) {
        real xbracket=sx_r*fxc_r-sx_l*fxc_l+fxl[I3(k,j,i+1,ny,nx+1)]-fxl[I3(k,j,i,ny,nx+1)];
        result=result-m*(dx_inv*xbracket);
    }
    if(!open_y || (j>=1 && j<=ny-2)) {
        real ybracket=sy_r*fyc_r-sy_l*fyc_l+fyl[I3(k,j+1,i,ny+1,nx)]-fyl[I3(k,j,i,ny+1,nx)];
        result=result-m*(dy_inv*ybracket);
    }
    tend_out[IDX3(k,j,i)]=result;
#else

    if (!open_x && !open_y) {            // ORIGINAL periodic apply
        if (has_msf) {                   // WRF "un-canceled" msftx weighting
            real m = msft[(size_t)j * nx + i];
            tend_out[IDX3(k, j, i)] +=
                -m * dx_inv * ((sx_r * fxc_r + fxl[I3(k, j, i + 1, ny, nx + 1)])
                               - (sx_l * fxc_l + fxl[I3(k, j, i, ny, nx + 1)]))
                - m * dy_inv * ((sy_r * fyc_r + fyl[I3(k, j + 1, i, ny + 1, nx)])
                                - (sy_l * fyc_l + fyl[I3(k, j, i, ny + 1, nx)]))
                - rdnw[k] * ((sz_t * fzc_t + fzl[IDX3(k + 1, j, i)])
                             - (sz_b * fzc_b + fzl[IDX3(k, j, i)]));
        } else {                         // original: bitwise Phase 2
            tend_out[IDX3(k, j, i)] +=
                -dx_inv * ((sx_r * fxc_r + fxl[I3(k, j, i + 1, ny, nx + 1)])
                           - (sx_l * fxc_l + fxl[I3(k, j, i, ny, nx + 1)]))
                - dy_inv * ((sy_r * fyc_r + fyl[I3(k, j + 1, i, ny + 1, nx)])
                            - (sy_l * fyc_l + fyl[I3(k, j, i, ny + 1, nx)]))
                - rdnw[k] * ((sz_t * fzc_t + fzl[IDX3(k + 1, j, i)])
                             - (sz_b * fzc_b + fzl[IDX3(k, j, i)]));
        }
        return;
    }

    // Specified/open apply: the vertical PD divergence covers every cell
    // (WRF F:7787-7791); the horizontal divergences skip the boundary
    // cells (x: ids+1..ide-2, F:7817-7821; y: jds+1..jde-2, F:7852-7856).
    real m = has_msf ? msft[(size_t)j * nx + i] : 1.0f;
    real t = -rdnw[k] * ((sz_t * fzc_t + fzl[IDX3(k + 1, j, i)])
                         - (sz_b * fzc_b + fzl[IDX3(k, j, i)]));
    if (!open_x || (i >= 1 && i <= nx - 2))
        t += -m * dx_inv * ((sx_r * fxc_r + fxl[I3(k, j, i + 1, ny, nx + 1)])
                            - (sx_l * fxc_l + fxl[I3(k, j, i, ny, nx + 1)]));
    if (!open_y || (j >= 1 && j <= ny - 2))
        t += -m * dy_inv * ((sy_r * fyc_r + fyl[I3(k, j + 1, i, ny + 1, nx)])
                            - (sy_l * fyc_l + fyl[I3(k, j, i, ny + 1, nx)]));
    tend_out[IDX3(k, j, i)] += t;
#endif
}
