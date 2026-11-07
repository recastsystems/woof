// gpuwm/core/kernels/advection.cu
//
// Flux-form advection on the Arakawa C-grid (WRF-ARW schemes):
//   horizontal: 5th-order upwind-biased fluxes (WRF flux5),
//   vertical:   WRF's vert_order ladder, selected per launch by the
//               trailing vorder argument (RunConfig v_sca_adv_order for
//               scalars and w, v_mom_adv_order for u and v):
//                 3: 3rd-order upwind-biased fluxes (WRF flux3), with a
//                    2nd-order centered fallback one face in from each
//                    boundary and zero flux through the domain top/bottom
//                    faces (every run before the argument existed);
//                 5: WRF's vert_order == 5 ladder (NOAA-EMC/HRRR
//                    40ee6058c, WRFV3.9 module_advect_em.F, the same
//                    text as WRF 4.7.1): 2nd order one face in, flux3
//                    two faces in, flux5 (Omega-signed, flux5v) between
//                    (advect_scalar F:5106-5147, advect_u F:2270-2310,
//                    advect_v F:3751-3796, advect_w F:6782-6830).
//
// All four kernels ADD  -dF_x/dx - dF_y/dy - dF_eta/deta  into tend_out.
// ru (nz,ny,nx+1), rv (nz,ny+1,nx), rw (nz+1,ny,nx) are coupled mass
// fluxes (mu*u etc.) precomputed by the caller.  rw is the eta-directed
// advecting flux, Omega-signed: negative for physically upward motion,
// since eta decreases with height.  Vertical fluxes therefore upwind on
// -vel -- WRF module_advect_em's flux3(..., -vel) convention -- which is
// why flux3's dissipation term carries the opposite sign to flux5's.
// x and y are periodic: stencil reads of the advected field wrap with
// period nx (ny), so the redundant staggered columns u[...,nx] / v[:,ny,:]
// are never read and their tendencies are computed identically to column 0.
//
// Face-flux index convention (matches gpuwm/verify/npref.py mirrors):
// the u-face f lies between cells f-1 and f, so
//   F(f) = vel*(37*(q[f]+q[f-1]) - 8*(q[f+1]+q[f-2]) + (q[f+2]+q[f-3]))/60
//        - |vel|*(10*(q[f]-q[f-1]) - 5*(q[f+1]-q[f-2]) + (q[f+2]-q[f-3]))/60
// and a cell's divergence uses F(right) - F(left).
//
// Open lateral boundaries (Task 11 prerequisite; WRF v4.6.1
// module_advect_em.F transcription): with open_x/open_y each kernel takes
// a dedicated path that (a) degrades the 5th-order stencil to 2nd order
// one face in from the boundary and WRF's horizontal flux3 two faces in
// (the degrade_xs/xe/ys/ye blocks), (b) applies WRF's loop-bound
// exclusions -- no advection normal to an open boundary at the boundary
// cells/faces, no cross-boundary stencil wrap anywhere -- and (c) adds
// WRF's non-cb open advective terms at the boundary cells of
// scalar/w/tangential-velocity fields ("the computations that don't
// require cb", with field_old == field per the rk_tendency call).  The
// boundary-normal velocity faces get NO horizontal advection here; the
// additive cb radiative term (openbc.cu) stands in for it.  Their
// vertical advection is retained with the boundary cell's Omega (WRF's
// zero-gradient rom ghost) for u unless open_y is also set (the Fortran's
// commented-out open_x bounds + active open_ys/ye bounds), and dropped
// for v whenever open_y is set -- both asymmetries transcribed verbatim.
// Periodic tendencies below the model lid retain the original arithmetic
// (tests/data/advection_periodic_regression.npz). The w lid row includes
// the native WRF flux contribution, checked by the compiled Fortran oracle.
//
// Specified boundaries (Phase 4 transport FIX-D; WRF's `specified`
// logical): with spec != 0 (dycore sets it from cfg.specified, always
// together with open_x/open_y) the non-cb open advective terms do NOT
// fire -- WRF gates them on open_xs/xe/ys/ye only and simply excludes the
// boundary cells from the tendency loops under specified -- and the
// boundary-adjacent 2nd-order u/v fluxes take WRF's "specified uses
// upstream normal wind at boundaries" substitution (advect_u F:690-723,
// advect_v F:1978-2013).  Vertical faces one in from the eta boundaries
// and advect_w's horizontal advecting velocities carry WRF's
// stretched-grid fnm/fnp weights (see zface_half; identical to the old
// 0.5/0.5 averages on uniform grids, bitwise).
//
// Map-scale factors (Phase 3 Task 3 + Phase 4 transport FIX-A; ARW tech
// note eqns 2.23-2.26, WRF module_advect_em.F mrdx/mrdy): every kernel
// takes the 2-D map factor at its TENDENCY points (msfu for u, msfv for v,
// msft for scalars and w) and, when has_msf != 0, weights the horizontal
// flux divergence by it, WRF's mrdx = msf*rdx / mrdy = msf*rdy, identical
// for both axes at a given point, in BOTH the periodic and the
// open/specified paths (WRF applies mrdx/mrdy unconditionally: scalar
// F:3534/3644, u F:633/740, v F:2050/2676, w F:5096/5548; real74 runs
// specified + Lambert msf through the open path every step).  The non-cb
// open advective terms follow WRF exactly: u's y-term and v's x-term are
// msf-weighted (mrdy at F:1296, mrdx at F:2766), the scalar and w terms
// carry plain rdx/rdy (F:4119-4177, F:5695-5806).  Vertical divergence
// stays unweighted: exact for u (eqn 2.23), w (2.25) and the my-divided
// scalar form (2.26); for v the Fortran's (msfvy/msfvx) vertical ratio is
// identically 1 with the isotropic single msfv gpuwm carries
// (Lambert/Mercator/polar).  has_msf == 0 keeps the original expressions
// verbatim so the msf==1 path stays BITWISE Phase 2 (regression-pinned by
// tests/data/phase2_step_regression.npz) regardless of FMA contraction
// choices.

#if !GPUWM_WRF_EXACT_C_ADVECTION
// Equivalent periodic indices without the second signed remainder.
__device__ __forceinline__ int adv_periodic(int i, int n)
{
    int r = i % n;
    return r < 0 ? r + n : r;
}

#endif

#if GPUWM_WENO_MOM
// WRF 4.7.1 advect_weno_{u,v,w}: REAL polynomial/smoothness expressions
// are assigned to DOUBLE PRECISION locals before nonlinear weighting.
__device__ __forceinline__
real weno_momentum(real qm3, real qm2, real qm1, real q0, real qp1,
                   real qp2, real vel, bool vertical)
{
    bool positive = vertical ? -vel >= 0.0f : vel >= 0.0f;
    real a = positive ? qm3 : qp2;
    real b = positive ? qm2 : qp1;
    real c = positive ? qm1 : q0;
    real d = positive ? q0 : qm1;
    real e = positive ? qp1 : qm2;
#if GPUWM_WRF_EXACT_C_ADVECTION
    typedef double weight_real;
#else
    // Native REAL32 stencil expressions feed DOUBLE weighting. Strict
    // replay retains that path. Fast numerics use normalized FP32 weights
    // so a GPU forecast does not pay the card's FP64 throughput penalty.
    typedef float weight_real;
#endif
    weight_real f0 = 0.3333333432674408f*a - 1.1666666269302368f*b
              + 1.8333333730697632f*c;
    weight_real f1 = -0.1666666716337204f*b + 0.8333333134651184f*c
              + 0.3333333432674408f*d;
    weight_real f2 = 0.3333333432674408f*c + 0.8333333134651184f*d
              - 0.1666666716337204f*e;
    real x0=a-2.0f*b+c, y0=a-4.0f*b+3.0f*c;
    real x1=b-2.0f*c+d, y1=b-d;
    real x2=c-2.0f*d+e, y2=e-4.0f*d+3.0f*c;
    weight_real beta0=1.0833333730697632f*(x0*x0)+0.25f*(y0*y0);
    weight_real beta1=1.0833333730697632f*(x1*x1)+0.25f*(y1*y1);
    weight_real beta2=1.0833333730697632f*(x2*x2)+0.25f*(y2*y2);
    weight_real eps=(weight_real)1.0e-18;
    weight_real t0=eps+beta0, t1=eps+beta1, t2=eps+beta2;
#if GPUWM_WRF_EXACT_C_ADVECTION
    weight_real w0=0.1/(t0*t0), w1=0.6/(t1*t1), w2=0.3/(t2*t2);
#else
    weight_real w0=__fdividef(0.1f,t0*t0);
    weight_real w1=__fdividef(0.6f,t1*t1);
    weight_real w2=__fdividef(0.3f,t2*t2);
#endif
#if GPUWM_WRF_EXACT_C_ADVECTION
    return (real)((double)vel*(w0*f0+w1*f1+w2*f2)/(w0+w1+w2));
#else
    // Normalize before multiplying by the field and mass flux: constant
    // stencils can have weights near 1e36, within FP32 but not their flux.
    real inverse=__fdividef(1.0f,w0+w1+w2);
    return vel*((w0*inverse)*f0+(w1*inverse)*f1+(w2*inverse)*f2);
#endif
}
#endif

__device__ __forceinline__
real flux5(real qm3, real qm2, real qm1, real q0, real qp1, real qp2, real vel)
{
#if GPUWM_WENO_MOM
    return weno_momentum(qm3,qm2,qm1,q0,qp1,qp2,vel,false);
#elif GPUWM_HORDER == 3
    real center = __fdiv_rn(7.0f*(q0+qm1)-(qp1+qm2),12.0f);
    real dissipation = __fdiv_rn(copysignf(1.0f,vel)
        *((qp1-qm2)-3.0f*(q0-qm1)),12.0f);
    return vel*(center+dissipation);
#elif GPUWM_HORDER == 4
    return vel*__fdiv_rn(7.0f*(q0+qm1)-(qp1+qm2),12.0f);
#elif GPUWM_HORDER == 6
    return vel*__fdiv_rn(37.0f*(q0+qm1)-8.0f*(qp1+qm2)+(qp2+qm3),60.0f);
#else
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = __fdiv_rn(37.0f * (q0 + qm1)
                    - 8.0f * (qp1 + qm2) + (qp2 + qm3), 60.0f);
    real dissipation = __fdiv_rn(copysignf(1.0f, vel)
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1)), 60.0f);
    return vel * (center - dissipation);
#else
    return __fdiv_rn((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            - fabsf(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
#endif
#endif
}

__device__ __forceinline__
real flux3(real qm2, real qm1, real q0, real qp1, real vel)
{
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = __fdiv_rn(7.0f * (q0 + qm1) - (qp1 + qm2), 12.0f);
    real dissipation = __fdiv_rn(copysignf(1.0f, vel)
        * ((qp1 - qm2) - 3.0f * (q0 - qm1)), 12.0f);
    return vel * (center - dissipation);
#else
    return __fdiv_rn((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            + fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
#endif
}

// Vertical 5th-order face flux: WRF flux5 called with -vel (module_advect_em.F
// vert_order == 5, vflux = vel*flux5(q(k-3..k+2), -vel)), so its dissipation
// carries the Omega-signed sign of the vertical flux3 above, opposite to the
// horizontal flux5.  Its own function rather than flux5 with a flipped
// argument: under the exact-C build the Fortran expression order is
// vel*(flux6 - sign(1,-vel)*D/60) = vel*(center + dissipation), and the
// fabsf form of the default build rounds the same way as flux3 does.
__device__ __forceinline__
real flux5v(real qm3, real qm2, real qm1, real q0, real qp1, real qp2, real vel)
{
#if GPUWM_WENO_MOM
    return weno_momentum(qm3,qm2,qm1,q0,qp1,qp2,vel,true);
#endif
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = __fdiv_rn(37.0f * (q0 + qm1)
                    - 8.0f * (qp1 + qm2) + (qp2 + qm3), 60.0f);
    real dissipation = __fdiv_rn(copysignf(1.0f, vel)
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1)), 60.0f);
    return vel * (center + dissipation);
#else
    return __fdiv_rn((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            + fabsf(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
#endif
}

// Horizontal 3rd-order face flux (WRF flux3 with the flux5 upwinding
// sign; the vertical flux3 above carries the opposite, Omega-signed
// dissipation) -- used two faces in from an open boundary.
__device__ __forceinline__
real flux3h(real qm2, real qm1, real q0, real qp1, real vel)
{
#if GPUWM_HORDER == 4 || GPUWM_HORDER == 6
    return vel*__fdiv_rn(7.0f*(q0+qm1)-(qp1+qm2),12.0f);
#endif
#if GPUWM_WRF_EXACT_C_ADVECTION
    real center = __fdiv_rn(7.0f * (q0 + qm1) - (qp1 + qm2), 12.0f);
    real dissipation = __fdiv_rn(copysignf(1.0f, vel)
        * ((qp1 - qm2) - 3.0f * (q0 - qm1)), 12.0f);
    return vel * (center + dissipation);
#else
    return __fdiv_rn((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            - fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
#endif
}

__device__ __forceinline__
real flux2(real qm1, real q0, real vel)
{
    return 0.5f * vel * (q0 + qm1);
}

// Native centered flux operators (module_advect_em.F flux4/flux6).
__device__ __forceinline__
real flux4(real qm2, real qm1, real q0, real qp1, real vel)
{
    return vel * __fdiv_rn(7.0f * (q0 + qm1) - (qp1 + qm2), 12.0f);
}

__device__ __forceinline__
real flux6(real qm3, real qm2, real qm1, real q0, real qp1, real qp2, real vel)
{
    return vel * __fdiv_rn(37.0f * (q0 + qm1)
                          - 8.0f * (qp1 + qm2) + (qp2 + qm3), 60.0f);
}

// x-face flux (face f = 0..n) of a CELL-type field q (row width nxs) with
// an OPEN x boundary pair: 0 at the never-consumed boundary faces, WRF's
// degraded orders near them, full 5th order in the interior (no wrap).
__device__ __forceinline__
real xface_cell_open(const real* q, real vel, int k, int j, int f,
                     int n, int ny, int nxs)
{
#if GPUWM_HORDER == 2
    return 0.5f*vel*(q[I3S(k,j,f,ny,nxs)]+q[I3S(k,j,f-1,ny,nxs)]);
#endif

    if (f <= 0 || f >= n) return 0.0f;
    int d = min(f, n - f);
    if (d == 1)
        return flux2(q[I3S(k, j, f - 1, ny, nxs)],
                     q[I3S(k, j, f, ny, nxs)], vel);
    if (d == 2)
        return flux3h(q[I3S(k, j, f - 2, ny, nxs)],
                      q[I3S(k, j, f - 1, ny, nxs)],
                      q[I3S(k, j, f, ny, nxs)],
                      q[I3S(k, j, f + 1, ny, nxs)], vel);
    return flux5(q[I3S(k, j, f - 3, ny, nxs)],
                 q[I3S(k, j, f - 2, ny, nxs)],
                 q[I3S(k, j, f - 1, ny, nxs)],
                 q[I3S(k, j, f, ny, nxs)],
                 q[I3S(k, j, f + 1, ny, nxs)],
                 q[I3S(k, j, f + 2, ny, nxs)], vel);
}

// Periodic counterpart (full 5th order, wrapped reads).
__device__ __forceinline__
real xface_cell_per(const real* q, real vel, int k, int j, int f,
                    int n, int ny, int nxs)
{
#if GPUWM_HORDER == 2
    return 0.5f*vel*(q[I3S(k,j,PERIODIC(f,n),ny,nxs)]+q[I3S(k,j,PERIODIC(f-1,n),ny,nxs)]);
#endif

    return flux5(q[I3S(k, j, PERIODIC(f - 3, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f - 2, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f - 1, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f + 1, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f + 2, n), ny, nxs)], vel);
}

// y-face analogues (face g = 0..n over the nys rows of q).
__device__ __forceinline__
real yface_cell_open(const real* q, real vel, int k, int g, int i,
                     int n, int nys, int nxs)
{
#if GPUWM_HORDER == 2
    return 0.5f*vel*(q[I3S(k,g,i,nys,nxs)]+q[I3S(k,g-1,i,nys,nxs)]);
#endif

    if (g <= 0 || g >= n) return 0.0f;
    int d = min(g, n - g);
    if (d == 1)
        return flux2(q[I3S(k, g - 1, i, nys, nxs)],
                     q[I3S(k, g, i, nys, nxs)], vel);
    if (d == 2)
        return flux3h(q[I3S(k, g - 2, i, nys, nxs)],
                      q[I3S(k, g - 1, i, nys, nxs)],
                      q[I3S(k, g, i, nys, nxs)],
                      q[I3S(k, g + 1, i, nys, nxs)], vel);
    return flux5(q[I3S(k, g - 3, i, nys, nxs)],
                 q[I3S(k, g - 2, i, nys, nxs)],
                 q[I3S(k, g - 1, i, nys, nxs)],
                 q[I3S(k, g, i, nys, nxs)],
                 q[I3S(k, g + 1, i, nys, nxs)],
                 q[I3S(k, g + 2, i, nys, nxs)], vel);
}

__device__ __forceinline__
real yface_cell_per(const real* q, real vel, int k, int g, int i,
                    int n, int nys, int nxs)
{
#if GPUWM_HORDER == 2
    return 0.5f*vel*(q[I3S(k,PERIODIC(g,n),i,nys,nxs)]+q[I3S(k,PERIODIC(g-1,n),i,nys,nxs)]);
#endif

    return flux5(q[I3S(k, PERIODIC(g - 3, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g - 2, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g - 1, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g + 1, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g + 2, n), i, nys, nxs)], vel);
}

// Vertical face flux for a half-level (mass/u/v-point) field q at w-level
// kf (0..nz): zero at the boundaries, WRF's stretched-grid fnm/fnp-weighted
// 2nd-order face value one face in (module_advect_em.F vert_order 3:
// vflux = rom*(fzm(k)*f(k) + fzp(k)*f(k-1)) at k=kts+1 and k=ktf, scalars
// :4322/:4327, u :1486/:1490, v mirrors; reduces bitwise to the 0.5/0.5
// average on a uniform grid), then the vorder ladder:
//   3: flux3 everywhere between (every run before vorder existed);
//   5: WRF vert_order == 5 (HRRR fork module_advect_em.F advect_scalar
//      F:5106-5147, advect_u F:2270-2310, advect_v F:3751-3796): flux3 at
//      k = kts+2 and k = ktf-1 (0-based w faces 2 and nz-2), flux5 with
//      -vel (flux5v) at k = kts+3 .. ktf-2 (faces 3 .. nz-3).
// The 2nd-order test comes first, so a column too short for WRF's loops
// (nz < 4, where the Fortran would read below its own memory bounds) runs
// the 2nd-order faces alone, as vorder 3 does.  q is indexed with row
// width nxs (nx or nx+1).
__device__ __forceinline__
real zface_half(const real* q, real vel, int kf, int j, int i,
                int nz, int ny, int nxs,
                const real* fnm, const real* fnp, int vorder)
{
    if (kf == 0 || kf == nz) return 0.0f;
    if (vorder == 2 || kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[I3S(kf,     j, i, ny, nxs)]
                      + fnp[kf] * q[I3S(kf - 1, j, i, ny, nxs)]);
    if (vorder == 6 && kf >= 3 && kf <= nz - 3)
        return flux6(q[I3S(kf - 3, j, i, ny, nxs)],
                     q[I3S(kf - 2, j, i, ny, nxs)],
                     q[I3S(kf - 1, j, i, ny, nxs)],
                     q[I3S(kf,     j, i, ny, nxs)],
                     q[I3S(kf + 1, j, i, ny, nxs)],
                     q[I3S(kf + 2, j, i, ny, nxs)], vel);
    if (vorder == 4 || vorder == 6)
        return flux4(q[I3S(kf - 2, j, i, ny, nxs)],
                     q[I3S(kf - 1, j, i, ny, nxs)],
                     q[I3S(kf,     j, i, ny, nxs)],
                     q[I3S(kf + 1, j, i, ny, nxs)], vel);
    if (vorder == 5 && kf >= 3 && kf <= nz - 3)
        return flux5v(q[I3S(kf - 3, j, i, ny, nxs)],
                      q[I3S(kf - 2, j, i, ny, nxs)],
                      q[I3S(kf - 1, j, i, ny, nxs)],
                      q[I3S(kf,     j, i, ny, nxs)],
                      q[I3S(kf + 1, j, i, ny, nxs)],
                      q[I3S(kf + 2, j, i, ny, nxs)], vel);
    return flux3(q[I3S(kf - 2, j, i, ny, nxs)],
                 q[I3S(kf - 1, j, i, ny, nxs)],
                 q[I3S(kf,     j, i, ny, nxs)],
                 q[I3S(kf + 1, j, i, ny, nxs)], vel);
}

// Full-level W transport: its faces are on mass levels, so the centered
// ladder is shifted one level relative to zface_half.
__device__ __forceinline__
real zface_w_centered(const real* w, real vel, int m, int j, int i,
                      int nz, int ny, int nx, int vorder)
{
    if (vorder == 2 || m == 0 || m == nz - 1)
        return flux2(w[IDX3(m,j,i)], w[IDX3(m+1,j,i)], vel);
    if (vorder == 6 && m >= 2 && m <= nz - 3)
        return flux6(w[IDX3(m-2,j,i)], w[IDX3(m-1,j,i)],
                     w[IDX3(m,j,i)], w[IDX3(m+1,j,i)],
                     w[IDX3(m+2,j,i)], w[IDX3(m+3,j,i)], vel);
    return flux4(w[IDX3(m-1,j,i)], w[IDX3(m,j,i)],
                 w[IDX3(m+1,j,i)], w[IDX3(m+2,j,i)], vel);
}

// ---------------------------------------------------------------------------
// Scalar q at mass points (nz, ny, nx).
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_scalar(const real* __restrict__ q,
                     const real* __restrict__ ru,       // (nz, ny, nx+1)
                     const real* __restrict__ rv,       // (nz, ny+1, nx)
                     const real* __restrict__ rw,       // (nz+1, ny, nx)
                     real* __restrict__ tend_out,       // (nz, ny, nx) +=
                     const real* __restrict__ rdnw,     // (nz,)
                     const real* __restrict__ fnm,      // (nz,) face weights
                     const real* __restrict__ fnp,      // (nz,)
                     const real* __restrict__ msf,      // (ny, nx) mass-pt
                     real dx_inv, real dy_inv,
                     int nz, int ny, int nx,
                     int open_x, int open_y, int has_msf, int spec,
                     int vorder)
{
#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_NO_HADV || GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4 || GPUWM_HORDER == 6

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    real mscale=has_msf?msf[(size_t)j*nx+i]:1.0f;



    if (open_x || open_y) {                            // WRF advect_scalar
        real th = 0.0f; real tx=0.0f,ty=0.0f; bool applyx=false,applyy=false;                                // open path with msf
        real tb = 0.0f;                                // (FIX-A): weighted
                                                       // flux divergences /
                                                       // plain non-cb terms
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: no x tendency at boundary cells
                // (F:4037-4038); non-cb terms are open-only (F:4115).
            } else if (i == 0) {                       // (plain rdx, WRF
                real ru0 = ru[I3(k, j, 0, ny, nx + 1)];        // F:4119)
                real ru1 = ru[I3(k, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ru0 + ru1), 0.0f);
                tx = dx_inv * (ub * (q[IDX3(k, j, 1)] - q[IDX3(k, j, 0)])
                                 + q[IDX3(k, j, 0)] * (ru1 - ru0)); applyx=true;
            } else {
                real ru0 = ru[I3(k, j, nx - 1, ny, nx + 1)];
                real ru1 = ru[I3(k, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ru0 + ru1), 0.0f);
                tx = dx_inv * (ub * (q[IDX3(k, j, nx - 1)]
                                       - q[IDX3(k, j, nx - 2)])
                                 + q[IDX3(k, j, nx - 1)] * (ru1 - ru0)); applyx=true;
            }
        } else {
            real fx0, fx1;
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            tx=(mscale*dx_inv)*(fx1-fx0); applyx=true;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rv0 = rv[I3(k, 0, i, ny + 1, nx)];
                real rv1 = rv[I3(k, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rv0 + rv1), 0.0f);
                ty = dy_inv * (vb * (q[IDX3(k, 1, i)] - q[IDX3(k, 0, i)])
                                 + q[IDX3(k, 0, i)] * (rv1 - rv0)); applyy=true;
            } else {
                real rv0 = rv[I3(k, ny - 1, i, ny + 1, nx)];
                real rv1 = rv[I3(k, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rv0 + rv1), 0.0f);
                ty = dy_inv * (vb * (q[IDX3(k, ny - 1, i)]
                                       - q[IDX3(k, ny - 2, i)])
                                 + q[IDX3(k, ny - 1, i)] * (rv1 - rv0)); applyy=true;
            }
        } else {
            real fy0, fy1;
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            ty=(mscale*dy_inv)*(fy1-fy0); applyy=true;
        }
        real fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[I3(kf, j, i, ny, nx)], kf, j, i,
                                nz, ny, nx, fnm, fnp, vorder);
        }

#if GPUWM_NO_HADV
        applyx=applyx && open_x && (i==0 || i==nx-1) && !spec;
        applyy=applyy && open_y && (j==0 || j==ny-1) && !spec;
#elif GPUWM_HORDER == 2
        if(!(open_x && (i==0 || i==nx-1))) {
            int ip=open_x?i+1:PERIODIC(i+1,nx), im=open_x?i-1:PERIODIC(i-1,nx);
            tx=(mscale*dx_inv)*0.5f*(ru[I3(k,j,i+1,ny,nx+1)]*(q[IDX3(k,j,ip)]+q[IDX3(k,j,i)])
                -ru[I3(k,j,i,ny,nx+1)]*(q[IDX3(k,j,i)]+q[IDX3(k,j,im)]));
        }
        if(!(open_y && (j==0 || j==ny-1))) {
            int jp=open_y?j+1:PERIODIC(j+1,ny), jm=open_y?j-1:PERIODIC(j-1,ny);
            ty=(mscale*dy_inv)*0.5f*(rv[I3(k,j+1,i,ny+1,nx)]*(q[IDX3(k,jp,i)]+q[IDX3(k,j,i)])
                -rv[I3(k,j,i,ny+1,nx)]*(q[IDX3(k,j,i)]+q[IDX3(k,jm,i)]));
        }
#endif
        real t = tend_out[IDX3(k,j,i)];
#if GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4
        if(applyx)t=t-tx;
        if(applyy)t=t-ty;
#else
        if(applyy)t=t-ty;
        if(applyx)t=t-tx;
#endif    // WRF mrdx/mrdy
#ifdef GPUWM_THETA_OFFSET
        // Native disables advection of theta-300. The fast acoustic path
        // transports full theta: its 300 K reference still follows mass.
        // Radiative edge terms above already carry the constant there.
        if (spec || !open_x || (i>0 && i<nx-1))
            t -= (mscale*dx_inv)*300.0f*(ru[I3(k,j,i+1,ny,nx+1)]-ru[I3(k,j,i,ny,nx+1)]);
        if (spec || !open_y || (j>0 && j<ny-1))
            t -= (mscale*dy_inv)*300.0f*(rv[I3(k,j+1,i,ny+1,nx)]-rv[I3(k,j,i,ny+1,nx)]);
#endif
        tend_out[IDX3(k,j,i)] = t - rdnw[k]*(fzo[1]-fzo[0]);
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face index 0..nx
        fx[s] = flux5(q[IDX3(k, j, PERIODIC(f - 3, nx))],
                      q[IDX3(k, j, PERIODIC(f - 2, nx))],
                      q[IDX3(k, j, PERIODIC(f - 1, nx))],
                      q[IDX3(k, j, PERIODIC(f,     nx))],
                      q[IDX3(k, j, PERIODIC(f + 1, nx))],
                      q[IDX3(k, j, PERIODIC(f + 2, nx))],
                      ru[I3(k, j, f, ny, nx + 1)]);
        int g = j + s;                                 // v-face index 0..ny
        fy[s] = flux5(q[IDX3(k, PERIODIC(g - 3, ny), i)],
                      q[IDX3(k, PERIODIC(g - 2, ny), i)],
                      q[IDX3(k, PERIODIC(g - 1, ny), i)],
                      q[IDX3(k, PERIODIC(g,     ny), i)],
                      q[IDX3(k, PERIODIC(g + 1, ny), i)],
                      q[IDX3(k, PERIODIC(g + 2, ny), i)],
                      rv[I3(k, g, i, ny + 1, nx)]);
        int kf = k + s;                                // w-face index 0..nz
        real vel = rw[I3(kf, j, i, ny, nx)];
        fz[s] = zface_half(q, vel, kf, j, i, nz, ny, nx, fnm, fnp, vorder);
    }
    real result=tend_out[IDX3(k,j,i)];
#if GPUWM_NO_HADV
#ifdef GPUWM_THETA_OFFSET
    result -= (mscale*dx_inv)*300.0f*(ru[I3(k,j,i+1,ny,nx+1)]-ru[I3(k,j,i,ny,nx+1)]);
    result -= (mscale*dy_inv)*300.0f*(rv[I3(k,j+1,i,ny+1,nx)]-rv[I3(k,j,i,ny+1,nx)]);
#endif
#elif GPUWM_HORDER == 2
    int ip=PERIODIC(i+1,nx),im=PERIODIC(i-1,nx),jp=PERIODIC(j+1,ny),jm=PERIODIC(j-1,ny);
    result=result-(mscale*dx_inv)*0.5f*(ru[I3(k,j,i+1,ny,nx+1)]*(q[IDX3(k,j,ip)]+q[IDX3(k,j,i)])
          -ru[I3(k,j,i,ny,nx+1)]*(q[IDX3(k,j,i)]+q[IDX3(k,j,im)]));
    result=result-(mscale*dy_inv)*0.5f*(rv[I3(k,j+1,i,ny+1,nx)]*(q[IDX3(k,jp,i)]+q[IDX3(k,j,i)])
          -rv[I3(k,j,i,ny+1,nx)]*(q[IDX3(k,j,i)]+q[IDX3(k,jm,i)]));
#elif GPUWM_HORDER == 3 || GPUWM_HORDER == 4
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
#else
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
#endif
    result=result-rdnw[k]*(fz[1]-fz[0]);
    tend_out[IDX3(k,j,i)]=result;
#else

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    if ((open_x || open_y) && !has_msf) {              // WRF advect_scalar
        real t = 0.0f;                                 // open path, msf==1:
                                                       // ORIGINAL Task-11
                                                       // body kept verbatim
                                                       // (bitwise-pinned)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ru0 = ru[I3(k, j, 0, ny, nx + 1)];
                real ru1 = ru[I3(k, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[IDX3(k, j, 1)] - q[IDX3(k, j, 0)])
                                + q[IDX3(k, j, 0)] * (ru1 - ru0));
            } else {
                real ru0 = ru[I3(k, j, nx - 1, ny, nx + 1)];
                real ru1 = ru[I3(k, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[IDX3(k, j, nx - 1)]
                                      - q[IDX3(k, j, nx - 2)])
                                + q[IDX3(k, j, nx - 1)] * (ru1 - ru0));
            }
        } else {
            real fx0, fx1;
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            t += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rv0 = rv[I3(k, 0, i, ny + 1, nx)];
                real rv1 = rv[I3(k, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[IDX3(k, 1, i)] - q[IDX3(k, 0, i)])
                                + q[IDX3(k, 0, i)] * (rv1 - rv0));
            } else {
                real rv0 = rv[I3(k, ny - 1, i, ny + 1, nx)];
                real rv1 = rv[I3(k, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[IDX3(k, ny - 1, i)]
                                      - q[IDX3(k, ny - 2, i)])
                                + q[IDX3(k, ny - 1, i)] * (rv1 - rv0));
            }
        } else {
            real fy0, fy1;
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            t += -(fy1 - fy0) * dy_inv;
        }
        real fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[I3(kf, j, i, ny, nx)], kf, j, i,
                                nz, ny, nx, fnm, fnp, vorder);
        }
        tend_out[IDX3(k, j, i)] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }

    if (open_x || open_y) {                            // WRF advect_scalar
        real th = 0.0f;                                // open path with msf
        real tb = 0.0f;                                // (FIX-A): weighted
                                                       // flux divergences /
                                                       // plain non-cb terms
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: no x tendency at boundary cells
                // (F:4037-4038); non-cb terms are open-only (F:4115).
            } else if (i == 0) {                       // (plain rdx, WRF
                real ru0 = ru[I3(k, j, 0, ny, nx + 1)];        // F:4119)
                real ru1 = ru[I3(k, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[IDX3(k, j, 1)] - q[IDX3(k, j, 0)])
                                 + q[IDX3(k, j, 0)] * (ru1 - ru0));
            } else {
                real ru0 = ru[I3(k, j, nx - 1, ny, nx + 1)];
                real ru1 = ru[I3(k, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[IDX3(k, j, nx - 1)]
                                       - q[IDX3(k, j, nx - 2)])
                                 + q[IDX3(k, j, nx - 1)] * (ru1 - ru0));
            }
        } else {
            real fx0, fx1;
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            th += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rv0 = rv[I3(k, 0, i, ny + 1, nx)];
                real rv1 = rv[I3(k, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[IDX3(k, 1, i)] - q[IDX3(k, 0, i)])
                                 + q[IDX3(k, 0, i)] * (rv1 - rv0));
            } else {
                real rv0 = rv[I3(k, ny - 1, i, ny + 1, nx)];
                real rv1 = rv[I3(k, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[IDX3(k, ny - 1, i)]
                                       - q[IDX3(k, ny - 2, i)])
                                 + q[IDX3(k, ny - 1, i)] * (rv1 - rv0));
            }
        } else {
            real fy0, fy1;
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            th += -(fy1 - fy0) * dy_inv;
        }
        real fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[I3(kf, j, i, ny, nx)], kf, j, i,
                                nz, ny, nx, fnm, fnp, vorder);
        }
        real t = msf[(size_t)j * nx + i] * th + tb;    // WRF mrdx/mrdy
        tend_out[IDX3(k, j, i)] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face index 0..nx
        fx[s] = flux5(q[IDX3(k, j, adv_periodic(f - 3, nx))],
                      q[IDX3(k, j, adv_periodic(f - 2, nx))],
                      q[IDX3(k, j, adv_periodic(f - 1, nx))],
                      q[IDX3(k, j, adv_periodic(f,     nx))],
                      q[IDX3(k, j, adv_periodic(f + 1, nx))],
                      q[IDX3(k, j, adv_periodic(f + 2, nx))],
                      ru[I3(k, j, f, ny, nx + 1)]);
        int g = j + s;                                 // v-face index 0..ny
        fy[s] = flux5(q[IDX3(k, adv_periodic(g - 3, ny), i)],
                      q[IDX3(k, adv_periodic(g - 2, ny), i)],
                      q[IDX3(k, adv_periodic(g - 1, ny), i)],
                      q[IDX3(k, adv_periodic(g,     ny), i)],
                      q[IDX3(k, adv_periodic(g + 1, ny), i)],
                      q[IDX3(k, adv_periodic(g + 2, ny), i)],
                      rv[I3(k, g, i, ny + 1, nx)]);
        int kf = k + s;                                // w-face index 0..nz
        real vel = rw[I3(kf, j, i, ny, nx)];
        fz[s] = zface_half(q, vel, kf, j, i, nz, ny, nx, fnm, fnp, vorder);
    }
    if (has_msf) {                       // WRF mrdx/mrdy = msft*rdx/rdy
        tend_out[IDX3(k, j, i)] += msf[(size_t)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[IDX3(k, j, i)] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdnw[k];
    }
#endif
}

// ---------------------------------------------------------------------------
// u momentum at u-points (nz, ny, nx+1).  x-faces of the u control volume
// sit at mass centers m (advecting flux 0.5*(ru[m]+ru[m+1])); y- and z-faces
// sit at corner points (rv, rw averaged across the two mass columns
// straddling the u-point).
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_u(const real* __restrict__ u,            // (nz, ny, nx+1)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz, ny, nx+1) +=
                const real* __restrict__ rdnw,         // (nz,)
                const real* __restrict__ fnm,          // (nz,) face weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny, nx+1) u-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder)
{
#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_NO_HADV || GPUWM_HORDER == 2

    int i = blockIdx.x * blockDim.x + threadIdx.x;     // u-point 0..nx
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;
    real mscale=has_msf?msf[(size_t)j*(nx+1)+i]:1.0f;

    if (open_x || open_y) {                            // WRF advect_u
        int nxs = nx + 1;                              // open-BC path
        bool xedge = open_x && (i == 0 || i == nx);    // boundary-normal face
        int cl = open_x ? max(i - 1, 0) : PERIODIC(i - 1, nx);
        int cr = open_x ? min(i, nx - 1) : PERIODIC(i, nx);
        int iu = open_x ? i : cr;                      // u read column: the
        real th = 0.0f; real tx=0.0f,ty=0.0f; bool applyx=false,applyy=false;                                // periodic duplicate
                                                       // face nx reads col 0

        if (!xedge) {                                  // x advection
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? cl : cr;            // mass-center face
                real velx = 0.5f * (ru[I3(k, j, m, ny, nxs)]
                                    + ru[I3(k, j, m + 1, ny, nxs)]);
                if (open_x) {
                    int d = min(m, nx - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_u F:690-723): fqx(ids+1)
                        // takes ub = u(ids+1) when u(ids+1) < 0, fqx(ide)
                        // takes ub = u(ide-1) when u(ide-1) > 0.
                        real qa = u[I3(k, j, m, ny, nxs)];
                        real qb = u[I3(k, j, m + 1, ny, nxs)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fx2[s] = flux2(qa, qb, velx);
                    }
                    else if (d == 1)
                        fx2[s] = flux3h(u[I3(k, j, m - 1, ny, nxs)],
                                        u[I3(k, j, m, ny, nxs)],
                                        u[I3(k, j, m + 1, ny, nxs)],
                                        u[I3(k, j, m + 2, ny, nxs)], velx);
                    else
                        fx2[s] = flux5(u[I3(k, j, m - 2, ny, nxs)],
                                       u[I3(k, j, m - 1, ny, nxs)],
                                       u[I3(k, j, m, ny, nxs)],
                                       u[I3(k, j, m + 1, ny, nxs)],
                                       u[I3(k, j, m + 2, ny, nxs)],
                                       u[I3(k, j, m + 3, ny, nxs)], velx);
                } else {
                    fx2[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m - 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 3, nx), ny, nxs)],
                                   velx);
                }
            }
            tx = mscale*dx_inv*(fx2[1]-fx2[0]); applyx=true;
        }

        if (!xedge) {                                  // y advection
            if (open_y && (j == 0 || j == ny - 1)) {   // non-cb open y-term
                if (spec) {
                    // WRF specified: u rows jds/jde-1 get no y tendency
                    // (degrade bounds j_start=jds+1, j_end=jde-2); the
                    // non-cb y-terms are open-only (F:1292/:1314).
                } else if (j == 0) {                   // (msf-weighted with
                    real vw = 0.5f * (rv[I3(k, 0, cl, ny + 1, nx)]   // th:
                                      + rv[I3(k, 0, cr, ny + 1, nx)]);
                    real vb = fminf(vw, 0.0f);         // WRF mrdy, F:1296)
                    real dsum = (rv[I3(k, 1, cl, ny + 1, nx)]
                                 - rv[I3(k, 0, cl, ny + 1, nx)])
                              + (rv[I3(k, 1, cr, ny + 1, nx)]
                                 - rv[I3(k, 0, cr, ny + 1, nx)]);
                    ty = mscale*dy_inv * (vb * (u[I3(k, 1, iu, ny, nxs)]
                                           - u[I3(k, 0, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, 0, iu, ny, nxs)] * dsum); applyy=true;
                } else {
                    real vw = 0.5f * (rv[I3(k, ny, cl, ny + 1, nx)]
                                      + rv[I3(k, ny, cr, ny + 1, nx)]);
                    real vb = fmaxf(vw, 0.0f);
                    real dsum = (rv[I3(k, ny, cl, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cl, ny + 1, nx)])
                              + (rv[I3(k, ny, cr, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cr, ny + 1, nx)]);
                    ty = mscale*dy_inv * (vb * (u[I3(k, ny - 1, iu, ny, nxs)]
                                           - u[I3(k, ny - 2, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, ny - 1, iu, ny, nxs)]
                                       * dsum); applyy=true;
                }
            } else {
                real fy2[2];
                for (int s = 0; s < 2; ++s) {
                    int g = j + s;                     // corner row 0..ny
                    real vely = 0.5f * (rv[I3(k, g, cl, ny + 1, nx)]
                                        + rv[I3(k, g, cr, ny + 1, nx)]);
                    fy2[s] = open_y
                        ? yface_cell_open(u, vely, k, g, iu, ny, ny, nxs)
                        : yface_cell_per(u, vely, k, g, iu, ny, ny, nxs);
                }
                ty = mscale*dy_inv*(fy2[1]-fy2[0]); applyy=true;
            }
        }

        // WRF advect_u weights every horizontal term (flux divergences AND
        // the non-cb open y-term) by msfux at the u point (mrdx F:740,
        // mrdy F:633/1296); the vertical term below stays unweighted.

#if GPUWM_NO_HADV
        applyx=false;
        applyy=applyy && open_y && (j==0 || j==ny-1) && !spec;
#elif GPUWM_HORDER == 2
        applyx=!xedge;
        if(applyx) {
            int ip=open_x ? i+1 : PERIODIC(i+1,nx);
            int im=open_x ? i-1 : PERIODIC(i-1,nx);
            real uc=u[I3(k,j,iu,ny,nxs)], up=u[I3(k,j,ip,ny,nxs)], um=u[I3(k,j,im,ny,nxs)];
            if(spec && open_x && i==1 && uc<0.0f)um=uc;
            if(spec && open_x && i==nx-1 && uc>0.0f)up=uc;
            tx=(mscale*dx_inv)*0.25f*((ru[I3(k,j,i+1,ny,nxs)]+ru[I3(k,j,i,ny,nxs)])*(up+uc)
               -(ru[I3(k,j,i,ny,nxs)]+ru[I3(k,j,i-1,ny,nxs)])*(uc+um));
        }
        if(!(open_y && (j==0 || j==ny-1))) {
            applyy=!xedge && (!spec || !open_x || (i>=2 && i<=nx-2));
            if(applyy) {
                int jp=open_y ? j+1 : PERIODIC(j+1,ny), jm=open_y ? j-1 : PERIODIC(j-1,ny);
                real uc=u[I3(k,j,iu,ny,nxs)];
                ty=(mscale*dy_inv)*0.25f*((rv[I3(k,j+1,cr,ny+1,nx)]+rv[I3(k,j+1,cl,ny+1,nx)])*(u[I3(k,jp,iu,ny,nxs)]+uc)
                   -(rv[I3(k,j,cr,ny+1,nx)]+rv[I3(k,j,cl,ny+1,nx)])*(uc+u[I3(k,jm,iu,ny,nxs)]));
            }
        }
#endif
        real t = tend_out[I3(k,j,i,ny,nx+1)];
        #if GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4
        // Native advect_u applies x before y for orders 3/4.
        if(applyx)t=t-tx;
        if(applyy)t=t-ty;
#else
        if(applyy)t=t-ty;
        if(applyx)t=t-tx;
#endif

        // z advection: retained at the boundary-normal faces with the
        // boundary cell's Omega unless open_y (Fortran advect_u vertical
        // bounds: open_x exclusions commented out, open_ys/ye + specified
        // active).
        if (!(xedge && open_y)) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz;
                if (open_x && i == 0)
                    velz = rw[I3(kf, j, 0, ny, nx)];
                else if (open_x && i == nx)
                    velz = rw[I3(kf, j, nx - 1, ny, nx)];
                else
                    velz = 0.5f * (rw[I3(kf, j, cl, ny, nx)]
                                   + rw[I3(kf, j, cr, ny, nx)]);
                fz2[s] = zface_half(u, velz, kf, j, iu, nz, ny, nxs, fnm, fnp, vorder);
            }
            t = t - rdnw[k]*(fz2[1]-fz2[0]);
        }
        tend_out[I3(k,j,i,ny,nx+1)] = t;
        return;
    }

    int ic   = PERIODIC(i, nx);                        // right mass column
    int im1c = PERIODIC(i - 1, nx);                    // left mass column

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int m = (s == 0) ? im1c : ic;                  // mass-center face
        real velx = 0.5f * (ru[I3(k, j, m,     ny, nx + 1)]
                          + ru[I3(k, j, m + 1, ny, nx + 1)]);
        fx[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m - 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m,     nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 3, nx), ny, nx + 1)],
                      velx);
        int g = j + s;                                 // corner row 0..ny
        real vely = 0.5f * (rv[I3(k, g, im1c, ny + 1, nx)]
                          + rv[I3(k, g, ic,   ny + 1, nx)]);
        fy[s] = flux5(u[I3(k, PERIODIC(g - 3, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 2, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g,     ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 2, ny), ic, ny, nx + 1)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, j, im1c, ny, nx)]
                          + rw[I3(kf, j, ic,   ny, nx)]);
        fz[s] = zface_half(u, velz, kf, j, ic, nz, ny, nx + 1, fnm, fnp, vorder);
    }

#if GPUWM_HORDER == 2
    real uc=u[I3(k,j,ic,ny,nx+1)];
    real tx=(mscale*dx_inv)*0.25f*((ru[I3(k,j,ic+1,ny,nx+1)]+ru[I3(k,j,ic,ny,nx+1)])*(u[I3(k,j,PERIODIC(ic+1,nx),ny,nx+1)]+uc)
        -(ru[I3(k,j,ic,ny,nx+1)]+ru[I3(k,j,im1c,ny,nx+1)])*(uc+u[I3(k,j,im1c,ny,nx+1)]));
    real ty=(mscale*dy_inv)*0.25f*((rv[I3(k,j+1,ic,ny+1,nx)]+rv[I3(k,j+1,im1c,ny+1,nx)])*(u[I3(k,PERIODIC(j+1,ny),ic,ny,nx+1)]+uc)
        -(rv[I3(k,j,ic,ny+1,nx)]+rv[I3(k,j,im1c,ny+1,nx)])*(uc+u[I3(k,PERIODIC(j-1,ny),ic,ny,nx+1)]));
#endif
    real result=tend_out[I3(k,j,i,ny,nx+1)];
    #if GPUWM_NO_HADV
    #elif GPUWM_HORDER == 2
    result=result-tx;
    result=result-ty;
    #elif GPUWM_HORDER == 3 || GPUWM_HORDER == 4
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
#else
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
#endif
    result=result-rdnw[k]*(fz[1]-fz[0]);
    tend_out[I3(k,j,i,ny,nx+1)]=result;
#else

    int i = blockIdx.x * blockDim.x + threadIdx.x;     // u-point 0..nx
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;

    if (open_x || open_y) {                            // WRF advect_u
        int nxs = nx + 1;                              // open-BC path
        bool xedge = open_x && (i == 0 || i == nx);    // boundary-normal face
        int cl = open_x ? max(i - 1, 0) : PERIODIC(i - 1, nx);
        int cr = open_x ? min(i, nx - 1) : PERIODIC(i, nx);
        int iu = open_x ? i : cr;                      // u read column: the
        real th = 0.0f;                                // periodic duplicate
                                                       // face nx reads col 0

        if (!xedge) {                                  // x advection
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? cl : cr;            // mass-center face
                real velx = 0.5f * (ru[I3(k, j, m, ny, nxs)]
                                    + ru[I3(k, j, m + 1, ny, nxs)]);
                if (open_x) {
                    int d = min(m, nx - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_u F:690-723): fqx(ids+1)
                        // takes ub = u(ids+1) when u(ids+1) < 0, fqx(ide)
                        // takes ub = u(ide-1) when u(ide-1) > 0.
                        real qa = u[I3(k, j, m, ny, nxs)];
                        real qb = u[I3(k, j, m + 1, ny, nxs)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fx2[s] = flux2(qa, qb, velx);
                    }
                    else if (d == 1)
                        fx2[s] = flux3h(u[I3(k, j, m - 1, ny, nxs)],
                                        u[I3(k, j, m, ny, nxs)],
                                        u[I3(k, j, m + 1, ny, nxs)],
                                        u[I3(k, j, m + 2, ny, nxs)], velx);
                    else
                        fx2[s] = flux5(u[I3(k, j, m - 2, ny, nxs)],
                                       u[I3(k, j, m - 1, ny, nxs)],
                                       u[I3(k, j, m, ny, nxs)],
                                       u[I3(k, j, m + 1, ny, nxs)],
                                       u[I3(k, j, m + 2, ny, nxs)],
                                       u[I3(k, j, m + 3, ny, nxs)], velx);
                } else {
                    fx2[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m - 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 3, nx), ny, nxs)],
                                   velx);
                }
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }

        if (!xedge) {                                  // y advection
            if (open_y && (j == 0 || j == ny - 1)) {   // non-cb open y-term
                if (spec) {
                    // WRF specified: u rows jds/jde-1 get no y tendency
                    // (degrade bounds j_start=jds+1, j_end=jde-2); the
                    // non-cb y-terms are open-only (F:1292/:1314).
                } else if (j == 0) {                   // (msf-weighted with
                    real vw = 0.5f * (rv[I3(k, 0, cl, ny + 1, nx)]   // th:
                                      + rv[I3(k, 0, cr, ny + 1, nx)]);
                    real vb = fminf(vw, 0.0f);         // WRF mrdy, F:1296)
                    real dsum = (rv[I3(k, 1, cl, ny + 1, nx)]
                                 - rv[I3(k, 0, cl, ny + 1, nx)])
                              + (rv[I3(k, 1, cr, ny + 1, nx)]
                                 - rv[I3(k, 0, cr, ny + 1, nx)]);
                    th += -dy_inv * (vb * (u[I3(k, 1, iu, ny, nxs)]
                                           - u[I3(k, 0, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, 0, iu, ny, nxs)] * dsum);
                } else {
                    real vw = 0.5f * (rv[I3(k, ny, cl, ny + 1, nx)]
                                      + rv[I3(k, ny, cr, ny + 1, nx)]);
                    real vb = fmaxf(vw, 0.0f);
                    real dsum = (rv[I3(k, ny, cl, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cl, ny + 1, nx)])
                              + (rv[I3(k, ny, cr, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cr, ny + 1, nx)]);
                    th += -dy_inv * (vb * (u[I3(k, ny - 1, iu, ny, nxs)]
                                           - u[I3(k, ny - 2, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, ny - 1, iu, ny, nxs)]
                                       * dsum);
                }
            } else {
                real fy2[2];
                for (int s = 0; s < 2; ++s) {
                    int g = j + s;                     // corner row 0..ny
                    real vely = 0.5f * (rv[I3(k, g, cl, ny + 1, nx)]
                                        + rv[I3(k, g, cr, ny + 1, nx)]);
                    fy2[s] = open_y
                        ? yface_cell_open(u, vely, k, g, iu, ny, ny, nxs)
                        : yface_cell_per(u, vely, k, g, iu, ny, ny, nxs);
                }
                th += -(fy2[1] - fy2[0]) * dy_inv;
            }
        }

        // WRF advect_u weights every horizontal term (flux divergences AND
        // the non-cb open y-term) by msfux at the u point (mrdx F:740,
        // mrdy F:633/1296); the vertical term below stays unweighted.
        real t = has_msf ? msf[(size_t)j * (nx + 1) + i] * th : th;

        // z advection: retained at the boundary-normal faces with the
        // boundary cell's Omega unless open_y (Fortran advect_u vertical
        // bounds: open_x exclusions commented out, open_ys/ye + specified
        // active).
        if (!(xedge && open_y)) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz;
                if (open_x && i == 0)
                    velz = rw[I3(kf, j, 0, ny, nx)];
                else if (open_x && i == nx)
                    velz = rw[I3(kf, j, nx - 1, ny, nx)];
                else
                    velz = 0.5f * (rw[I3(kf, j, cl, ny, nx)]
                                   + rw[I3(kf, j, cr, ny, nx)]);
                fz2[s] = zface_half(u, velz, kf, j, iu, nz, ny, nxs, fnm, fnp, vorder);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[I3(k, j, i, ny, nxs)] += t;
        return;
    }

    int ic   = PERIODIC(i, nx);                        // right mass column
    int im1c = PERIODIC(i - 1, nx);                    // left mass column

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int m = (s == 0) ? im1c : ic;                  // mass-center face
        real velx = 0.5f * (ru[I3(k, j, m,     ny, nx + 1)]
                          + ru[I3(k, j, m + 1, ny, nx + 1)]);
        fx[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m - 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m,     nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 3, nx), ny, nx + 1)],
                      velx);
        int g = j + s;                                 // corner row 0..ny
        real vely = 0.5f * (rv[I3(k, g, im1c, ny + 1, nx)]
                          + rv[I3(k, g, ic,   ny + 1, nx)]);
        fy[s] = flux5(u[I3(k, PERIODIC(g - 3, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 2, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g,     ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 2, ny), ic, ny, nx + 1)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, j, im1c, ny, nx)]
                          + rw[I3(kf, j, ic,   ny, nx)]);
        fz[s] = zface_half(u, velz, kf, j, ic, nz, ny, nx + 1, fnm, fnp, vorder);
    }
    if (has_msf) {                       // WRF advect_u: msfux at the u point
        tend_out[I3(k, j, i, ny, nx + 1)] += msf[(size_t)j * (nx + 1) + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[I3(k, j, i, ny, nx + 1)] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
#endif
}

// ---------------------------------------------------------------------------
// v momentum at v-points (nz, ny+1, nx).  Mirror image of flux_div_u.
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_v(const real* __restrict__ v,            // (nz, ny+1, nx)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz, ny+1, nx) +=
                const real* __restrict__ rdnw,         // (nz,)
                const real* __restrict__ fnm,          // (nz,) face weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny+1, nx) v-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder)
{
#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_NO_HADV || GPUWM_HORDER == 2

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;     // v-point 0..ny
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;
    real mscale=has_msf?msf[(size_t)j*nx+i]:1.0f;

    if (open_x || open_y) {                            // WRF advect_v
        int nys = ny + 1;                              // open-BC path
        bool yedge = open_y && (j == 0 || j == ny);    // boundary-normal face
        int rs = open_y ? max(j - 1, 0) : PERIODIC(j - 1, ny);
        int rn = open_y ? min(j, ny - 1) : PERIODIC(j, ny);
        int jv = open_y ? j : rn;                      // v read row: the
        real th = 0.0f; real tx=0.0f,ty=0.0f; bool applyx=false,applyy=false;                                // periodic duplicate
                                                       // face ny reads row 0

        if (!yedge) {                                  // y advection
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? rs : rn;            // mass-center face
                real vely = 0.5f * (rv[I3(k, m, i, nys, nx)]
                                    + rv[I3(k, m + 1, i, nys, nx)]);
                if (open_y) {
                    int d = min(m, ny - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_v F:1978-2013): fqy(jds+1)
                        // takes vb = v(jds+1) when v(jds+1) < 0, fqy(jde)
                        // takes vb = v(jde-1) when v(jde-1) > 0.
                        real qa = v[I3(k, m, i, nys, nx)];
                        real qb = v[I3(k, m + 1, i, nys, nx)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fy2[s] = flux2(qa, qb, vely);
                    }
                    else if (d == 1)
                        fy2[s] = flux3h(v[I3(k, m - 1, i, nys, nx)],
                                        v[I3(k, m, i, nys, nx)],
                                        v[I3(k, m + 1, i, nys, nx)],
                                        v[I3(k, m + 2, i, nys, nx)], vely);
                    else
                        fy2[s] = flux5(v[I3(k, m - 2, i, nys, nx)],
                                       v[I3(k, m - 1, i, nys, nx)],
                                       v[I3(k, m, i, nys, nx)],
                                       v[I3(k, m + 1, i, nys, nx)],
                                       v[I3(k, m + 2, i, nys, nx)],
                                       v[I3(k, m + 3, i, nys, nx)], vely);
                } else {
                    fy2[s] = flux5(v[I3(k, PERIODIC(m - 2, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m - 1, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 1, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 2, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 3, ny), i, nys, nx)],
                                   vely);
                }
            }
            ty = mscale*dy_inv*(fy2[1]-fy2[0]); applyy=true;
        }

        if (!yedge) {                                  // x advection
            if (open_x && (i == 0 || i == nx - 1)) {   // non-cb open x-term:
                // corner-averaged ru AT the boundary face (WRF advect_v
                // open_xs/xe, module_advect_em.F:2763/2785).
                if (spec) {
                    // WRF specified: v columns ids/ide-1 get no x tendency
                    // (F:2660-2661); the non-cb x-terms are open-only
                    // (F:2763/:2785).
                } else if (i == 0) {
                    real uw0 = 0.5f * (ru[I3(k, rs, 0, ny, nx + 1)]
                                       + ru[I3(k, rn, 0, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, 1, ny, nx + 1)]
                                       + ru[I3(k, rn, 1, ny, nx + 1)]);
                    real ub = fminf(uw0, 0.0f);
                    tx = mscale*dx_inv * (ub * (v[I3(k, jv, 1, nys, nx)]
                                           - v[I3(k, jv, 0, nys, nx)])
                                     + v[I3(k, jv, 0, nys, nx)] * (uw1 - uw0)); applyx=true;
                } else {
                    real uw0 = 0.5f * (ru[I3(k, rs, nx - 1, ny, nx + 1)]
                                       + ru[I3(k, rn, nx - 1, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, nx, ny, nx + 1)]
                                       + ru[I3(k, rn, nx, ny, nx + 1)]);
                    real ub = fmaxf(uw1, 0.0f);
                    tx = mscale*dx_inv * (ub * (v[I3(k, jv, nx - 1, nys, nx)]
                                           - v[I3(k, jv, nx - 2, nys, nx)])
                                     + v[I3(k, jv, nx - 1, nys, nx)]
                                       * (uw1 - uw0)); applyx=true;
                }
            } else {
                real fx2[2];
                for (int s = 0; s < 2; ++s) {
                    int f = i + s;                     // corner column 0..nx
                    real velx = 0.5f * (ru[I3(k, rs, f, ny, nx + 1)]
                                        + ru[I3(k, rn, f, ny, nx + 1)]);
                    fx2[s] = open_x
                        ? xface_cell_open(v, velx, k, jv, f, nx, nys, nx)
                        : xface_cell_per(v, velx, k, jv, f, nx, nys, nx);
                }
                tx = mscale*dx_inv*(fx2[1]-fx2[0]); applyx=true;
            }
        }

        // WRF advect_v weights every horizontal term (flux divergences AND
        // the non-cb open x-term) by msfvy at the v point (mrdy F:2050,
        // mrdx F:2676/2766); the vertical term below stays unweighted.

#if GPUWM_NO_HADV
        applyy=false;
        applyx=applyx && open_x && (i==0 || i==nx-1) && !spec;
#elif GPUWM_HORDER == 2
        applyy=!yedge;
        if(applyy) {
            int jp=open_y ? j+1 : PERIODIC(j+1,ny), jm=open_y ? j-1 : PERIODIC(j-1,ny);
            real vc=v[I3(k,jv,i,nys,nx)], vp=v[I3(k,jp,i,nys,nx)], vm=v[I3(k,jm,i,nys,nx)];
            if(spec && open_y && j==1 && vc<0.0f)vm=vc;
            if(spec && open_y && j==ny-1 && vc>0.0f)vp=vc;
            ty=(mscale*dy_inv)*0.25f*((rv[I3(k,j+1,i,nys,nx)]+rv[I3(k,j,i,nys,nx)])*(vp+vc)
               -(rv[I3(k,j,i,nys,nx)]+rv[I3(k,j-1,i,nys,nx)])*(vc+vm));
        }
        if(!(open_x && (i==0 || i==nx-1))) {
            applyx=!yedge && (!spec || !open_y || (j>=2 && j<=ny-2));
            if(applyx) {
                int ip=open_x ? i+1 : PERIODIC(i+1,nx), im=open_x ? i-1 : PERIODIC(i-1,nx);
                real vc=v[I3(k,jv,i,nys,nx)];
                tx=(mscale*dx_inv)*0.25f*((ru[I3(k,rn,i+1,ny,nx+1)]+ru[I3(k,rs,i+1,ny,nx+1)])*(v[I3(k,jv,ip,nys,nx)]+vc)
                   -(ru[I3(k,rn,i,ny,nx+1)]+ru[I3(k,rs,i,ny,nx+1)])*(vc+v[I3(k,jv,im,nys,nx)]));
            }
        }
#endif
        real t = tend_out[I3(k,j,i,ny+1,nx)];
        if(applyy)t=t-ty;
        if(applyx)t=t-tx;

        // z advection: excluded at the boundary-normal faces whenever
        // open_y or specified (Fortran advect_v vertical open_ys/ye
        // j-bounds).
        if (!yedge) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz = 0.5f * (rw[I3(kf, rs, i, ny, nx)]
                                    + rw[I3(kf, rn, i, ny, nx)]);
                fz2[s] = zface_half(v, velz, kf, jv, i, nz, nys, nx, fnm, fnp, vorder);
            }
            t = t - rdnw[k]*(fz2[1]-fz2[0]);
        }
        tend_out[I3(k,j,i,ny+1,nx)] = t;
        return;
    }

    int jc   = PERIODIC(j, ny);                        // north mass row
    int jm1c = PERIODIC(j - 1, ny);                    // south mass row

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // corner column 0..nx
        real velx = 0.5f * (ru[I3(k, jm1c, f, ny, nx + 1)]
                          + ru[I3(k, jc,   f, ny, nx + 1)]);
        fx[s] = flux5(v[I3(k, jc, PERIODIC(f - 3, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f - 2, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f - 1, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f,     nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f + 1, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f + 2, nx), ny + 1, nx)],
                      velx);
        int m = (s == 0) ? jm1c : jc;                  // mass-center face
        real vely = 0.5f * (rv[I3(k, m,     i, ny + 1, nx)]
                          + rv[I3(k, m + 1, i, ny + 1, nx)]);
        fy[s] = flux5(v[I3(k, PERIODIC(m - 2, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m - 1, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m,     ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 1, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 2, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 3, ny), i, ny + 1, nx)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, jm1c, i, ny, nx)]
                          + rw[I3(kf, jc,   i, ny, nx)]);
        fz[s] = zface_half(v, velz, kf, jc, i, nz, ny + 1, nx, fnm, fnp, vorder);
    }

#if GPUWM_HORDER == 2
    real vc=v[I3(k,jc,i,ny+1,nx)];
    real ty=(mscale*dy_inv)*0.25f*((rv[I3(k,jc+1,i,ny+1,nx)]+rv[I3(k,jc,i,ny+1,nx)])*(v[I3(k,PERIODIC(jc+1,ny),i,ny+1,nx)]+vc)
        -(rv[I3(k,jc,i,ny+1,nx)]+rv[I3(k,jm1c,i,ny+1,nx)])*(vc+v[I3(k,jm1c,i,ny+1,nx)]));
    real tx=(mscale*dx_inv)*0.25f*((ru[I3(k,jc,i+1,ny,nx+1)]+ru[I3(k,jm1c,i+1,ny,nx+1)])*(v[I3(k,jc,PERIODIC(i+1,nx),ny+1,nx)]+vc)
        -(ru[I3(k,jc,i,ny,nx+1)]+ru[I3(k,jm1c,i,ny,nx+1)])*(vc+v[I3(k,jc,PERIODIC(i-1,nx),ny+1,nx)]));
#endif
    real result=tend_out[I3(k,j,i,ny+1,nx)];
    #if GPUWM_NO_HADV
    #elif GPUWM_HORDER == 2
    result=result-ty;
    result=result-tx;
    #else
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
    #endif
    result=result-rdnw[k]*(fz[1]-fz[0]);
    tend_out[I3(k,j,i,ny+1,nx)]=result;
#else

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;     // v-point 0..ny
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;

    if (open_x || open_y) {                            // WRF advect_v
        int nys = ny + 1;                              // open-BC path
        bool yedge = open_y && (j == 0 || j == ny);    // boundary-normal face
        int rs = open_y ? max(j - 1, 0) : adv_periodic(j - 1, ny);
        int rn = open_y ? min(j, ny - 1) : adv_periodic(j, ny);
        int jv = open_y ? j : rn;                      // v read row: the
        real th = 0.0f;                                // periodic duplicate
                                                       // face ny reads row 0

        if (!yedge) {                                  // y advection
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? rs : rn;            // mass-center face
                real vely = 0.5f * (rv[I3(k, m, i, nys, nx)]
                                    + rv[I3(k, m + 1, i, nys, nx)]);
                if (open_y) {
                    int d = min(m, ny - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_v F:1978-2013): fqy(jds+1)
                        // takes vb = v(jds+1) when v(jds+1) < 0, fqy(jde)
                        // takes vb = v(jde-1) when v(jde-1) > 0.
                        real qa = v[I3(k, m, i, nys, nx)];
                        real qb = v[I3(k, m + 1, i, nys, nx)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fy2[s] = flux2(qa, qb, vely);
                    }
                    else if (d == 1)
                        fy2[s] = flux3h(v[I3(k, m - 1, i, nys, nx)],
                                        v[I3(k, m, i, nys, nx)],
                                        v[I3(k, m + 1, i, nys, nx)],
                                        v[I3(k, m + 2, i, nys, nx)], vely);
                    else
                        fy2[s] = flux5(v[I3(k, m - 2, i, nys, nx)],
                                       v[I3(k, m - 1, i, nys, nx)],
                                       v[I3(k, m, i, nys, nx)],
                                       v[I3(k, m + 1, i, nys, nx)],
                                       v[I3(k, m + 2, i, nys, nx)],
                                       v[I3(k, m + 3, i, nys, nx)], vely);
                } else {
                    fy2[s] = flux5(v[I3(k, adv_periodic(m - 2, ny), i, nys, nx)],
                                   v[I3(k, adv_periodic(m - 1, ny), i, nys, nx)],
                                   v[I3(k, adv_periodic(m, ny), i, nys, nx)],
                                   v[I3(k, adv_periodic(m + 1, ny), i, nys, nx)],
                                   v[I3(k, adv_periodic(m + 2, ny), i, nys, nx)],
                                   v[I3(k, adv_periodic(m + 3, ny), i, nys, nx)],
                                   vely);
                }
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }

        if (!yedge) {                                  // x advection
            if (open_x && (i == 0 || i == nx - 1)) {   // non-cb open x-term:
                // corner-averaged ru AT the boundary face (WRF advect_v
                // open_xs/xe, module_advect_em.F:2763/2785).
                if (spec) {
                    // WRF specified: v columns ids/ide-1 get no x tendency
                    // (F:2660-2661); the non-cb x-terms are open-only
                    // (F:2763/:2785).
                } else if (i == 0) {
                    real uw0 = 0.5f * (ru[I3(k, rs, 0, ny, nx + 1)]
                                       + ru[I3(k, rn, 0, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, 1, ny, nx + 1)]
                                       + ru[I3(k, rn, 1, ny, nx + 1)]);
                    real ub = fminf(uw0, 0.0f);
                    th += -dx_inv * (ub * (v[I3(k, jv, 1, nys, nx)]
                                           - v[I3(k, jv, 0, nys, nx)])
                                     + v[I3(k, jv, 0, nys, nx)] * (uw1 - uw0));
                } else {
                    real uw0 = 0.5f * (ru[I3(k, rs, nx - 1, ny, nx + 1)]
                                       + ru[I3(k, rn, nx - 1, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, nx, ny, nx + 1)]
                                       + ru[I3(k, rn, nx, ny, nx + 1)]);
                    real ub = fmaxf(uw1, 0.0f);
                    th += -dx_inv * (ub * (v[I3(k, jv, nx - 1, nys, nx)]
                                           - v[I3(k, jv, nx - 2, nys, nx)])
                                     + v[I3(k, jv, nx - 1, nys, nx)]
                                       * (uw1 - uw0));
                }
            } else {
                real fx2[2];
                for (int s = 0; s < 2; ++s) {
                    int f = i + s;                     // corner column 0..nx
                    real velx = 0.5f * (ru[I3(k, rs, f, ny, nx + 1)]
                                        + ru[I3(k, rn, f, ny, nx + 1)]);
                    fx2[s] = open_x
                        ? xface_cell_open(v, velx, k, jv, f, nx, nys, nx)
                        : xface_cell_per(v, velx, k, jv, f, nx, nys, nx);
                }
                th += -(fx2[1] - fx2[0]) * dx_inv;
            }
        }

        // WRF advect_v weights every horizontal term (flux divergences AND
        // the non-cb open x-term) by msfvy at the v point (mrdy F:2050,
        // mrdx F:2676/2766); the vertical term below stays unweighted.
        real t = has_msf ? msf[(size_t)j * nx + i] * th : th;

        // z advection: excluded at the boundary-normal faces whenever
        // open_y or specified (Fortran advect_v vertical open_ys/ye
        // j-bounds).
        if (!yedge) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz = 0.5f * (rw[I3(kf, rs, i, ny, nx)]
                                    + rw[I3(kf, rn, i, ny, nx)]);
                fz2[s] = zface_half(v, velz, kf, jv, i, nz, nys, nx, fnm, fnp, vorder);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[I3(k, j, i, nys, nx)] += t;
        return;
    }

    int jc   = adv_periodic(j, ny);                        // north mass row
    int jm1c = adv_periodic(j - 1, ny);                    // south mass row

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // corner column 0..nx
        real velx = 0.5f * (ru[I3(k, jm1c, f, ny, nx + 1)]
                          + ru[I3(k, jc,   f, ny, nx + 1)]);
        fx[s] = flux5(v[I3(k, jc, adv_periodic(f - 3, nx), ny + 1, nx)],
                      v[I3(k, jc, adv_periodic(f - 2, nx), ny + 1, nx)],
                      v[I3(k, jc, adv_periodic(f - 1, nx), ny + 1, nx)],
                      v[I3(k, jc, adv_periodic(f,     nx), ny + 1, nx)],
                      v[I3(k, jc, adv_periodic(f + 1, nx), ny + 1, nx)],
                      v[I3(k, jc, adv_periodic(f + 2, nx), ny + 1, nx)],
                      velx);
        int m = (s == 0) ? jm1c : jc;                  // mass-center face
        real vely = 0.5f * (rv[I3(k, m,     i, ny + 1, nx)]
                          + rv[I3(k, m + 1, i, ny + 1, nx)]);
        fy[s] = flux5(v[I3(k, adv_periodic(m - 2, ny), i, ny + 1, nx)],
                      v[I3(k, adv_periodic(m - 1, ny), i, ny + 1, nx)],
                      v[I3(k, adv_periodic(m,     ny), i, ny + 1, nx)],
                      v[I3(k, adv_periodic(m + 1, ny), i, ny + 1, nx)],
                      v[I3(k, adv_periodic(m + 2, ny), i, ny + 1, nx)],
                      v[I3(k, adv_periodic(m + 3, ny), i, ny + 1, nx)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, jm1c, i, ny, nx)]
                          + rw[I3(kf, jc,   i, ny, nx)]);
        fz[s] = zface_half(v, velz, kf, jc, i, nz, ny + 1, nx, fnm, fnp, vorder);
    }
    if (has_msf) {                       // WRF advect_v: msfvy at the v point
        tend_out[I3(k, j, i, ny + 1, nx)] += msf[(size_t)j * nx + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[I3(k, j, i, ny + 1, nx)] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
#endif
}

// ---------------------------------------------------------------------------
// w momentum at w-points (nz+1, ny, nx).  Horizontal advecting fluxes are
// ru/rv averaged across the two half levels straddling the w-level; the
// vertical flux of w lives at mass (half) levels m with advecting flux
// 0.5*(rw[m]+rw[m+1]). The surface has no tendency. The top carries WRF's
// extrapolated horizontal advection and one-sided vertical flux contribution;
// the acoustic solver separately enforces a rigid lid when configured.
// ---------------------------------------------------------------------------
__device__ __forceinline__
real w_lid_velocity(const real* flux, int j, int i, int nz, int nys, int nxs,
                    const real* fnm, const real* fnp)
{
    return (2.0f - fnm[nz - 1]) * flux[I3S(nz - 1, j, i, nys, nxs)]
           - fnp[nz - 1] * flux[I3S(nz - 2, j, i, nys, nxs)];
}

extern "C" __global__
void flux_div_w(const real* __restrict__ w,            // (nz+1, ny, nx)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz+1, ny, nx) +=
                const real* __restrict__ rdn,          // (nz,)
                const real* __restrict__ fnm,          // (nz,) w-level weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny, nx) mass-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder)
{
#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_NO_HADV || GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4 || GPUWM_HORDER == 6

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;     // w-level 0..nz
    if (i >= nx || j >= ny || k > nz) return;
    real mscale=has_msf?msf[(size_t)j*nx+i]:1.0f;
    if (k == 0 || (k == nz && nz<2)) return;                     // BC levels: no tendency

    if (k == nz) {
        // WRF v4.7.1 advect_w: lid face winds use (2-fzm)*RU - fzp*RUbelow.
        // Its vertical lid contribution is 2*rdzu(ktf)*vflux(ktf+1).
        real th=0.0f; real tx=0.0f,ty=0.0f; bool applyx=false,applyy=false;
        real tb = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (!spec) {
                int f0 = i == 0 ? 0 : nx - 1;
                real r0 = w_lid_velocity(ru, j, f0, nz, ny, nx + 1, fnm, fnp);
                real r1 = w_lid_velocity(ru, j, f0 + 1, nz, ny, nx + 1, fnm, fnp);
                real ub = i == 0 ? fminf(0.5f * (r0 + r1), 0.0f)
                                : fmaxf(0.5f * (r0 + r1), 0.0f);
                real dq = i == 0 ? w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)]
                                : w[IDX3(k, j, nx - 1)] - w[IDX3(k, j, nx - 2)];
                tx = dx_inv * (ub * dq + w[IDX3(k, j, i)] * (r1 - r0));
            }
        } else {
            real r0 = w_lid_velocity(ru, j, i, nz, ny, nx + 1, fnm, fnp);
            real r1 = w_lid_velocity(ru, j, i + 1, nz, ny, nx + 1, fnm, fnp);
            real f0 = open_x ? xface_cell_open(w, r0, k, j, i, nx, ny, nx)
                             : xface_cell_per(w, r0, k, j, i, nx, ny, nx);
            real f1 = open_x ? xface_cell_open(w, r1, k, j, i + 1, nx, ny, nx)
                             : xface_cell_per(w, r1, k, j, i + 1, nx, ny, nx);
            tx=(mscale*dx_inv)*(f1-f0); applyx=true;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (!spec) {
                int g0 = j == 0 ? 0 : ny - 1;
                real r0 = w_lid_velocity(rv, g0, i, nz, ny + 1, nx, fnm, fnp);
                real r1 = w_lid_velocity(rv, g0 + 1, i, nz, ny + 1, nx, fnm, fnp);
                real vb = j == 0 ? fminf(0.5f * (r0 + r1), 0.0f)
                                : fmaxf(0.5f * (r0 + r1), 0.0f);
                real dq = j == 0 ? w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)]
                                : w[IDX3(k, ny - 1, i)] - w[IDX3(k, ny - 2, i)];
                ty = dy_inv * (vb * dq + w[IDX3(k, j, i)] * (r1 - r0));
            }
        } else {
            real r0 = w_lid_velocity(rv, j, i, nz, ny + 1, nx, fnm, fnp);
            real r1 = w_lid_velocity(rv, j + 1, i, nz, ny + 1, nx, fnm, fnp);
            real f0 = open_y ? yface_cell_open(w, r0, k, j, i, ny, ny, nx)
                             : yface_cell_per(w, r0, k, j, i, ny, ny, nx);
            real f1 = open_y ? yface_cell_open(w, r1, k, j + 1, i, ny, ny, nx)
                             : yface_cell_per(w, r1, k, j + 1, i, ny, ny, nx);
            ty=(mscale*dy_inv)*(f1-f0); applyy=true;
        }

#if GPUWM_NO_HADV
        applyx=applyx && open_x && (i==0 || i==nx-1) && !spec;
        applyy=applyy && open_y && (j==0 || j==ny-1) && !spec;
#elif GPUWM_HORDER == 2
        if(!(open_x && (i==0 || i==nx-1))) {
            int ip=open_x?i+1:PERIODIC(i+1,nx),im=open_x?i-1:PERIODIC(i-1,nx);
            tx=(mscale*dx_inv)*0.5f*(w_lid_velocity(ru,j,i+1,nz,ny,nx+1,fnm,fnp)*(w[IDX3(k,j,ip)]+w[IDX3(k,j,i)])
                -w_lid_velocity(ru,j,i,nz,ny,nx+1,fnm,fnp)*(w[IDX3(k,j,i)]+w[IDX3(k,j,im)]));
        }
        if(!(open_y && (j==0 || j==ny-1))) {
            int jp=open_y?j+1:PERIODIC(j+1,ny),jm=open_y?j-1:PERIODIC(j-1,ny);
            ty=(mscale*dy_inv)*0.5f*(w_lid_velocity(rv,j+1,i,nz,ny+1,nx,fnm,fnp)*(w[IDX3(k,jp,i)]+w[IDX3(k,j,i)])
                -w_lid_velocity(rv,j,i,nz,ny+1,nx,fnm,fnp)*(w[IDX3(k,j,i)]+w[IDX3(k,jm,i)]));
        }
#endif
        real horizontal=tend_out[IDX3(k,j,i)];
#if GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4
        if(applyx)horizontal-=tx;
        if(applyy)horizontal-=ty;
#else
        if(applyy)horizontal-=ty;
        if(applyx)horizontal-=tx;
#endif
        real top_flux = 0.25f * (rw[IDX3(k, j, i)] + rw[IDX3(k - 1, j, i)])
                             * (w[IDX3(k, j, i)] + w[IDX3(k - 1, j, i)]);
        tend_out[IDX3(k, j, i)] = horizontal + 2.0f * rdn[k - 1] * top_flux;
        return;
    }



    if (open_x || open_y) {                            // WRF advect_w with
        real th = 0.0f; real tx=0.0f,ty=0.0f; bool applyx=false,applyy=false;                                // msf (FIX-A):
                                                       // weighted flux
        real tb = 0.0f;                                // divergences / plain
                                                       // non-cb open terms
        // horizontal advecting fluxes interpolated to the w level with
        // WRF's fnm/fnp weights (advect_w F:5004/:4531)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ruw0 = fnm[k] * ru[I3(k, j, 0, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 0, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ruw0 + ruw1), 0.0f);
                tx = dx_inv * (ub * (w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)])
                                 + w[IDX3(k, j, 0)] * (ruw1 - ruw0)); applyx=true;
            } else {
                real ruw0 = fnm[k] * ru[I3(k, j, nx - 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx - 1, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, nx, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ruw0 + ruw1), 0.0f);
                tx = dx_inv * (ub * (w[IDX3(k, j, nx - 1)]
                                       - w[IDX3(k, j, nx - 2)])
                                 + w[IDX3(k, j, nx - 1)] * (ruw1 - ruw0)); applyx=true;
            }
        } else {
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                real velx = fnm[k] * ru[I3(k, j, f, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            tx = mscale*dx_inv*(fx2[1]-fx2[0]); applyx=true;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rvw0 = fnm[k] * rv[I3(k, 0, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 0, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rvw0 + rvw1), 0.0f);
                ty = dy_inv * (vb * (w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)])
                                 + w[IDX3(k, 0, i)] * (rvw1 - rvw0)); applyy=true;
            } else {
                real rvw0 = fnm[k] * rv[I3(k, ny - 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny - 1, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, ny, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rvw0 + rvw1), 0.0f);
                ty = dy_inv * (vb * (w[IDX3(k, ny - 1, i)]
                                       - w[IDX3(k, ny - 2, i)])
                                 + w[IDX3(k, ny - 1, i)] * (rvw1 - rvw0)); applyy=true;
            }
        } else {
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                real vely = fnm[k] * rv[I3(k, g, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            ty = mscale*dy_inv*(fy2[1]-fy2[0]); applyy=true;
        }
        // Vertical faces on mass levels m (WRF advect_w vflux(k), m = k-2):
        // m == 0 and nz-1 are WRF's 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1))
        // faces (flux2), and under vorder 5 (HRRR fork module_advect_em.F
        // advect_w F:6782-6830, which keys on v_sca_adv_order) m == 1 and
        // nz-2 are flux3 (k = kts+2, ktf) with flux5v on w[m-2..m+3] for
        // m = 2 .. nz-3 (k = kts+3 .. ktf-1); vorder 3 keeps flux3 between.
        // The lid row's 2*rdzu(ktf)*vflux(ktf+1) reads the top flux2 face.
        real fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;                         // mass-level face
            real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
            if (vorder == 2 || vorder == 4 || vorder == 6)
                fz2[s] = zface_w_centered(w, velz, m, j, i, nz, ny, nx, vorder);

            else if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
            else if (vorder == 5 && m >= 2 && m <= nz - 3)
                fz2[s] = flux5v(w[IDX3(m - 2, j, i)], w[IDX3(m - 1, j, i)],
                                w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)],
                                w[IDX3(m + 2, j, i)], w[IDX3(m + 3, j, i)],
                                velz);
            else
                fz2[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                               w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)],
                               velz);
        }
        // WRF advect_w mrdx/mrdy = msftx*rdx/rdy (F:5096); the non-cb open
        // terms carry plain rdx/rdy (F:5695).

#if GPUWM_NO_HADV
        applyx=applyx && open_x && (i==0 || i==nx-1) && !spec;
        applyy=applyy && open_y && (j==0 || j==ny-1) && !spec;
#elif GPUWM_HORDER == 2
        if(!(open_x && (i==0 || i==nx-1))) {
            int ip=open_x?i+1:PERIODIC(i+1,nx),im=open_x?i-1:PERIODIC(i-1,nx);
            tx=(mscale*dx_inv)*0.5f*((fnm[k]*ru[I3(k,j,i+1,ny,nx+1)]+fnp[k]*ru[I3(k-1,j,i+1,ny,nx+1)])*(w[IDX3(k,j,ip)]+w[IDX3(k,j,i)])
                -(fnm[k]*ru[I3(k,j,i,ny,nx+1)]+fnp[k]*ru[I3(k-1,j,i,ny,nx+1)])*(w[IDX3(k,j,i)]+w[IDX3(k,j,im)]));
        }
        if(!(open_y && (j==0 || j==ny-1))) {
            int jp=open_y?j+1:PERIODIC(j+1,ny),jm=open_y?j-1:PERIODIC(j-1,ny);
            ty=(mscale*dy_inv)*0.5f*((fnm[k]*rv[I3(k,j+1,i,ny+1,nx)]+fnp[k]*rv[I3(k-1,j+1,i,ny+1,nx)])*(w[IDX3(k,jp,i)]+w[IDX3(k,j,i)])
                -(fnm[k]*rv[I3(k,j,i,ny+1,nx)]+fnp[k]*rv[I3(k-1,j,i,ny+1,nx)])*(w[IDX3(k,j,i)]+w[IDX3(k,jm,i)]));
        }
#endif
        real t = tend_out[IDX3(k,j,i)];
#if GPUWM_HORDER == 2 || GPUWM_HORDER == 3 || GPUWM_HORDER == 4
        if(applyx)t=t-tx;
        if(applyy)t=t-ty;
#else
        if(applyy)t=t-ty;
        if(applyx)t=t-tx;
#endif
        tend_out[IDX3(k,j,i)] = t - rdn[k]*(fz2[1]-fz2[0]);
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face column 0..nx
        real velx = fnm[k] * ru[I3(k,     j, f, ny, nx + 1)]
                  + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
        fx[s] = flux5(w[IDX3(k, j, PERIODIC(f - 3, nx))],
                      w[IDX3(k, j, PERIODIC(f - 2, nx))],
                      w[IDX3(k, j, PERIODIC(f - 1, nx))],
                      w[IDX3(k, j, PERIODIC(f,     nx))],
                      w[IDX3(k, j, PERIODIC(f + 1, nx))],
                      w[IDX3(k, j, PERIODIC(f + 2, nx))],
                      velx);
        int g = j + s;                                 // v-face row 0..ny
        real vely = fnm[k] * rv[I3(k,     g, i, ny + 1, nx)]
                  + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
        fy[s] = flux5(w[IDX3(k, PERIODIC(g - 3, ny), i)],
                      w[IDX3(k, PERIODIC(g - 2, ny), i)],
                      w[IDX3(k, PERIODIC(g - 1, ny), i)],
                      w[IDX3(k, PERIODIC(g,     ny), i)],
                      w[IDX3(k, PERIODIC(g + 1, ny), i)],
                      w[IDX3(k, PERIODIC(g + 2, ny), i)],
                      vely);
        int m = k - 1 + s;                             // mass-level face 0..nz-1
        real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
        if (vorder == 2 || vorder == 4 || vorder == 6)
            fz[s] = zface_w_centered(w, velz, m, j, i, nz, ny, nx, vorder);

        else if (m == 0 || m == nz - 1)
            fz[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
        else if (vorder == 5 && m >= 2 && m <= nz - 3)
            fz[s] = flux5v(w[IDX3(m - 2, j, i)], w[IDX3(m - 1, j, i)],
                            w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)],
                            w[IDX3(m + 2, j, i)], w[IDX3(m + 3, j, i)], velz);
        else
            fz[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                          w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)], velz);
    }
    real tx=0.0f,ty=0.0f; bool applyx=true,applyy=true;

#if GPUWM_NO_HADV
    applyx=applyx && open_x && (i==0 || i==nx-1) && !spec;
    applyy=applyy && open_y && (j==0 || j==ny-1) && !spec;
#elif GPUWM_HORDER == 2
    {
        int ip=open_x?i+1:PERIODIC(i+1,nx),im=open_x?i-1:PERIODIC(i-1,nx);
        tx=(mscale*dx_inv)*0.5f*((fnm[k]*ru[I3(k,j,i+1,ny,nx+1)]+fnp[k]*ru[I3(k-1,j,i+1,ny,nx+1)])*(w[IDX3(k,j,ip)]+w[IDX3(k,j,i)])
        -(fnm[k]*ru[I3(k,j,i,ny,nx+1)]+fnp[k]*ru[I3(k-1,j,i,ny,nx+1)])*(w[IDX3(k,j,i)]+w[IDX3(k,j,im)]));
    }
    {
        int jp=open_y?j+1:PERIODIC(j+1,ny),jm=open_y?j-1:PERIODIC(j-1,ny);
        ty=(mscale*dy_inv)*0.5f*((fnm[k]*rv[I3(k,j+1,i,ny+1,nx)]+fnp[k]*rv[I3(k-1,j+1,i,ny+1,nx)])*(w[IDX3(k,jp,i)]+w[IDX3(k,j,i)])
        -(fnm[k]*rv[I3(k,j,i,ny+1,nx)]+fnp[k]*rv[I3(k-1,j,i,ny+1,nx)])*(w[IDX3(k,j,i)]+w[IDX3(k,jm,i)]));
    }
#endif
    real result=tend_out[IDX3(k,j,i)];
#if GPUWM_NO_HADV
#elif GPUWM_HORDER == 2
    result=result-tx;
    result=result-ty;
#elif GPUWM_HORDER == 3 || GPUWM_HORDER == 4
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
#else
    result=result-(mscale*dy_inv)*(fy[1]-fy[0]);
    result=result-(mscale*dx_inv)*(fx[1]-fx[0]);
#endif
    result=result-rdn[k]*(fz[1]-fz[0]);
    tend_out[IDX3(k,j,i)]=result;
#else

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;     // w-level 0..nz
    if (i >= nx || j >= ny || k > nz) return;
    // A one-level primitive has no second mass level for lid extrapolation.
    if (k == 0 || (k == nz && nz < 2)) return;

    if (k == nz) {
        // WRF v4.7.1 advect_w: lid face winds use (2-fzm)*RU - fzp*RUbelow.
        // Its vertical lid contribution is 2*rdzu(ktf)*vflux(ktf+1).
        real th = 0.0f;
        real tb = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (!spec) {
                int f0 = i == 0 ? 0 : nx - 1;
                real r0 = w_lid_velocity(ru, j, f0, nz, ny, nx + 1, fnm, fnp);
                real r1 = w_lid_velocity(ru, j, f0 + 1, nz, ny, nx + 1, fnm, fnp);
                real ub = i == 0 ? fminf(0.5f * (r0 + r1), 0.0f)
                                : fmaxf(0.5f * (r0 + r1), 0.0f);
                real dq = i == 0 ? w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)]
                                : w[IDX3(k, j, nx - 1)] - w[IDX3(k, j, nx - 2)];
                tb += -dx_inv * (ub * dq + w[IDX3(k, j, i)] * (r1 - r0));
            }
        } else {
            real r0 = w_lid_velocity(ru, j, i, nz, ny, nx + 1, fnm, fnp);
            real r1 = w_lid_velocity(ru, j, i + 1, nz, ny, nx + 1, fnm, fnp);
            real f0 = open_x ? xface_cell_open(w, r0, k, j, i, nx, ny, nx)
                             : xface_cell_per(w, r0, k, j, i, nx, ny, nx);
            real f1 = open_x ? xface_cell_open(w, r1, k, j, i + 1, nx, ny, nx)
                             : xface_cell_per(w, r1, k, j, i + 1, nx, ny, nx);
            th += -(f1 - f0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (!spec) {
                int g0 = j == 0 ? 0 : ny - 1;
                real r0 = w_lid_velocity(rv, g0, i, nz, ny + 1, nx, fnm, fnp);
                real r1 = w_lid_velocity(rv, g0 + 1, i, nz, ny + 1, nx, fnm, fnp);
                real vb = j == 0 ? fminf(0.5f * (r0 + r1), 0.0f)
                                : fmaxf(0.5f * (r0 + r1), 0.0f);
                real dq = j == 0 ? w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)]
                                : w[IDX3(k, ny - 1, i)] - w[IDX3(k, ny - 2, i)];
                tb += -dy_inv * (vb * dq + w[IDX3(k, j, i)] * (r1 - r0));
            }
        } else {
            real r0 = w_lid_velocity(rv, j, i, nz, ny + 1, nx, fnm, fnp);
            real r1 = w_lid_velocity(rv, j + 1, i, nz, ny + 1, nx, fnm, fnp);
            real f0 = open_y ? yface_cell_open(w, r0, k, j, i, ny, ny, nx)
                             : yface_cell_per(w, r0, k, j, i, ny, ny, nx);
            real f1 = open_y ? yface_cell_open(w, r1, k, j + 1, i, ny, ny, nx)
                             : yface_cell_per(w, r1, k, j + 1, i, ny, ny, nx);
            th += -(f1 - f0) * dy_inv;
        }
        real horizontal = has_msf ? msf[(size_t)j * nx + i] * th + tb : th + tb;
        real top_flux = 0.25f * (rw[IDX3(k, j, i)] + rw[IDX3(k - 1, j, i)])
                             * (w[IDX3(k, j, i)] + w[IDX3(k - 1, j, i)]);
        tend_out[IDX3(k, j, i)] += horizontal + 2.0f * rdn[k - 1] * top_flux;
        return;
    }

    if ((open_x || open_y) && !has_msf) {              // WRF advect_w
        real t = 0.0f;                                 // ORIGINAL Task-11
                                                       // body kept verbatim
                                                       // (bitwise-pinned)
        // horizontal advecting fluxes interpolated to the w level with
        // WRF's fnm/fnp weights (advect_w F:5004/:4531)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ruw0 = fnm[k] * ru[I3(k, j, 0, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 0, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)])
                                + w[IDX3(k, j, 0)] * (ruw1 - ruw0));
            } else {
                real ruw0 = fnm[k] * ru[I3(k, j, nx - 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx - 1, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, nx, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[IDX3(k, j, nx - 1)]
                                      - w[IDX3(k, j, nx - 2)])
                                + w[IDX3(k, j, nx - 1)] * (ruw1 - ruw0));
            }
        } else {
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                real velx = fnm[k] * ru[I3(k, j, f, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            t += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rvw0 = fnm[k] * rv[I3(k, 0, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 0, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)])
                                + w[IDX3(k, 0, i)] * (rvw1 - rvw0));
            } else {
                real rvw0 = fnm[k] * rv[I3(k, ny - 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny - 1, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, ny, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[IDX3(k, ny - 1, i)]
                                      - w[IDX3(k, ny - 2, i)])
                                + w[IDX3(k, ny - 1, i)] * (rvw1 - rvw0));
            }
        } else {
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                real vely = fnm[k] * rv[I3(k, g, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            t += -(fy2[1] - fy2[0]) * dy_inv;
        }
        // Vertical faces on mass levels m (WRF advect_w vflux(k), m = k-2):
        // m == 0 and nz-1 are WRF's 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1))
        // faces (flux2), and under vorder 5 (HRRR fork module_advect_em.F
        // advect_w F:6782-6830, which keys on v_sca_adv_order) m == 1 and
        // nz-2 are flux3 (k = kts+2, ktf) with flux5v on w[m-2..m+3] for
        // m = 2 .. nz-3 (k = kts+3 .. ktf-1); vorder 3 keeps flux3 between.
        // The lid row's 2*rdzu(ktf)*vflux(ktf+1) reads the top flux2 face.
        real fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;                         // mass-level face
            real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
            if (vorder == 2 || vorder == 4 || vorder == 6)
                fz2[s] = zface_w_centered(w, velz, m, j, i, nz, ny, nx, vorder);

            else if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
            else if (vorder == 5 && m >= 2 && m <= nz - 3)
                fz2[s] = flux5v(w[IDX3(m - 2, j, i)], w[IDX3(m - 1, j, i)],
                                w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)],
                                w[IDX3(m + 2, j, i)], w[IDX3(m + 3, j, i)],
                                velz);
            else
                fz2[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                               w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)],
                               velz);
        }
        tend_out[IDX3(k, j, i)] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }

    if (open_x || open_y) {                            // WRF advect_w with
        real th = 0.0f;                                // msf (FIX-A):
                                                       // weighted flux
        real tb = 0.0f;                                // divergences / plain
                                                       // non-cb open terms
        // horizontal advecting fluxes interpolated to the w level with
        // WRF's fnm/fnp weights (advect_w F:5004/:4531)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ruw0 = fnm[k] * ru[I3(k, j, 0, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 0, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)])
                                 + w[IDX3(k, j, 0)] * (ruw1 - ruw0));
            } else {
                real ruw0 = fnm[k] * ru[I3(k, j, nx - 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx - 1, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, nx, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[IDX3(k, j, nx - 1)]
                                       - w[IDX3(k, j, nx - 2)])
                                 + w[IDX3(k, j, nx - 1)] * (ruw1 - ruw0));
            }
        } else {
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                real velx = fnm[k] * ru[I3(k, j, f, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rvw0 = fnm[k] * rv[I3(k, 0, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 0, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)])
                                 + w[IDX3(k, 0, i)] * (rvw1 - rvw0));
            } else {
                real rvw0 = fnm[k] * rv[I3(k, ny - 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny - 1, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, ny, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[IDX3(k, ny - 1, i)]
                                       - w[IDX3(k, ny - 2, i)])
                                 + w[IDX3(k, ny - 1, i)] * (rvw1 - rvw0));
            }
        } else {
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                real vely = fnm[k] * rv[I3(k, g, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }
        // Vertical faces on mass levels m (WRF advect_w vflux(k), m = k-2):
        // m == 0 and nz-1 are WRF's 0.25*(rom(k)+rom(k-1))*(w(k)+w(k-1))
        // faces (flux2), and under vorder 5 (HRRR fork module_advect_em.F
        // advect_w F:6782-6830, which keys on v_sca_adv_order) m == 1 and
        // nz-2 are flux3 (k = kts+2, ktf) with flux5v on w[m-2..m+3] for
        // m = 2 .. nz-3 (k = kts+3 .. ktf-1); vorder 3 keeps flux3 between.
        // The lid row's 2*rdzu(ktf)*vflux(ktf+1) reads the top flux2 face.
        real fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;                         // mass-level face
            real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
            if (vorder == 2 || vorder == 4 || vorder == 6)
                fz2[s] = zface_w_centered(w, velz, m, j, i, nz, ny, nx, vorder);

            else if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
            else if (vorder == 5 && m >= 2 && m <= nz - 3)
                fz2[s] = flux5v(w[IDX3(m - 2, j, i)], w[IDX3(m - 1, j, i)],
                                w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)],
                                w[IDX3(m + 2, j, i)], w[IDX3(m + 3, j, i)],
                                velz);
            else
                fz2[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                               w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)],
                               velz);
        }
        // WRF advect_w mrdx/mrdy = msftx*rdx/rdy (F:5096); the non-cb open
        // terms carry plain rdx/rdy (F:5695).
        real t = msf[(size_t)j * nx + i] * th + tb;
        tend_out[IDX3(k, j, i)] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face column 0..nx
        real velx = fnm[k] * ru[I3(k,     j, f, ny, nx + 1)]
                  + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
        fx[s] = flux5(w[IDX3(k, j, PERIODIC(f - 3, nx))],
                      w[IDX3(k, j, PERIODIC(f - 2, nx))],
                      w[IDX3(k, j, PERIODIC(f - 1, nx))],
                      w[IDX3(k, j, PERIODIC(f,     nx))],
                      w[IDX3(k, j, PERIODIC(f + 1, nx))],
                      w[IDX3(k, j, PERIODIC(f + 2, nx))],
                      velx);
        int g = j + s;                                 // v-face row 0..ny
        real vely = fnm[k] * rv[I3(k,     g, i, ny + 1, nx)]
                  + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
        fy[s] = flux5(w[IDX3(k, PERIODIC(g - 3, ny), i)],
                      w[IDX3(k, PERIODIC(g - 2, ny), i)],
                      w[IDX3(k, PERIODIC(g - 1, ny), i)],
                      w[IDX3(k, PERIODIC(g,     ny), i)],
                      w[IDX3(k, PERIODIC(g + 1, ny), i)],
                      w[IDX3(k, PERIODIC(g + 2, ny), i)],
                      vely);
        int m = k - 1 + s;                             // mass-level face 0..nz-1
        real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
        if (vorder == 2 || vorder == 4 || vorder == 6)
            fz[s] = zface_w_centered(w, velz, m, j, i, nz, ny, nx, vorder);

        else if (m == 0 || m == nz - 1)
            fz[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
        else if (vorder == 5 && m >= 2 && m <= nz - 3)
            fz[s] = flux5v(w[IDX3(m - 2, j, i)], w[IDX3(m - 1, j, i)],
                            w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)],
                            w[IDX3(m + 2, j, i)], w[IDX3(m + 3, j, i)], velz);
        else
            fz[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                          w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)], velz);
    }
    if (has_msf) {                       // WRF advect_w: msftx at mass point
        tend_out[IDX3(k, j, i)] += msf[(size_t)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdn[k];
    } else {                             // original: bitwise Phase 2
        tend_out[IDX3(k, j, i)] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdn[k];
    }
#endif
}
