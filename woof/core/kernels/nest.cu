// gpuwm/core/kernels/nest.cu
//
// Phase 5 Task 10 (panel lane L3): parent->child nest interpolation
// operators, transliterated from WRF v4.6.1:
//
//   - SINT / SINTB           share/sint.F:2-198 / :203-347
//   - bdy_interp1            share/interp_fcn.F:2423-2626
//   - blend_terrain          dyn_em/nest_init_utils.F:712-785
//   - adjust_tempqv          dyn_em/nest_init_utils.F:812-890
//   - copy_fcn/fcnm/fcni     share/interp_fcn.F:1397-1906 (dormant, P5b)
//
// The horizontal operator authority is share/sint.F ITSELF (F6 amendment):
// the field-dependent DONOR/TR4 flux statement functions and the nonlinear
// overshoot/undershoot limiter are evaluated per field at force time.  Only
// GEOMETRY is precomputed (donor index maps + the XIG/XJG offset coefficient
// tables, sint.F:46-59): FP64-built on host, stored FP32 on device -- a
// registered deviation from WRF's on-the-fly REAL construction (sint.F:13-14,
// :31), tested by N1.5.  The FP64 mirrors (gpuwm/verify/npref.py) consume the
// SAME FP32-rounded tables.
//
// This module is compiled WITHOUT FMA contraction (-fmad=false, see
// gpuwm/core/nest_interp.py) so the summation paths round like the mirrors.

// ---------------------------------------------------------------------------
// share/sint.F statement functions (:37, :39-41, :43-44) and constants.
// ---------------------------------------------------------------------------

// DATA EP/ 1.E-10/ (sint.F:26)
#define SINT_EP 1.0e-10f
// PARAMETER(one12=1./12.,one24=1./24.) (sint.F:14)
#define SINT_ONE12 (1.0f / 12.0f)
#define SINT_ONE24 (1.0f / 24.0f)

// DONOR(Y1,Y2,A)=(Y1*AMAX1(0.,SIGN(1.,A))-Y2*AMIN1(0.,SIGN(1.,A)))*A
// (sint.F:37).  SIGN(1.,A) = copysignf(1.f, a), signed zero included.
__device__ __forceinline__ float sint_donor(float y1, float y2, float a)
{
    float s = copysignf(1.0f, a);
    return (y1 * fmaxf(0.0f, s) - y2 * fminf(0.0f, s)) * a;
}

// TR4(YM1,Y0,YP1,YP2,A) (sint.F:39-41), the 4th-order transport flux.
__device__ __forceinline__ float sint_tr4(float ym1, float y0, float yp1,
                                          float yp2, float a)
{
    return a * SINT_ONE12 * (7.0f * (yp1 + y0) - (yp2 + ym1))
         - a * a * SINT_ONE24 * (15.0f * (yp1 - y0) - (yp2 - ym1))
         - a * a * a * SINT_ONE12 * ((yp1 + y0) - (yp2 + ym1))
         + a * a * a * a * SINT_ONE24 * (3.0f * (yp1 - y0) - (yp2 - ym1));
}

// One 1-D residual-advection pass of SINTB (sint.F:286-310): donor fluxes,
// low-order update W, TR4 fluxes, antidiffusive fluxes, and the OV/UN
// min/max limiter.  PP(X)=AMAX1(0.,X), PN(X)=AMIN1(0.,X) (sint.F:43-44).
__device__ __forceinline__ float sint_pass(float ym2, float ym1, float y0,
                                           float yp1, float yp2, float a)
{
    float fl0 = sint_donor(ym1, y0, a);                       // :286
    float fl1 = sint_donor(y0, yp1, a);                       // :287
    float w = y0 - (fl1 - fl0);                               // :288
    float mxm = fmaxf(fmaxf(ym1, y0), fmaxf(yp1, w));         // :289-291
    float mn = fminf(fminf(ym1, y0), fminf(yp1, w));          // :292
    float f0 = sint_tr4(ym2, ym1, y0, yp1, a) - fl0;          // :293-299
    float f1 = sint_tr4(ym1, y0, yp1, yp2, a) - fl1;          // :296-300
    float pp0 = fmaxf(0.0f, f0), pn0 = fminf(0.0f, f0);
    float pp1 = fmaxf(0.0f, f1), pn1 = fminf(0.0f, f1);
    float ov = (mxm - w) / (-pn1 + pp0 + SINT_EP);            // :301-302
    float un = (w - mn) / (pp1 - pn0 + SINT_EP);              // :303-304
    float c0 = pp0 * fminf(1.0f, ov) + pn0 * fminf(1.0f, un); // :305-306
    float c1 = pp1 * fminf(1.0f, un) + pn1 * fminf(1.0f, ov); // :307-308
    return w - (c1 - c0);                                     // :309
}

// Full SINT evaluation at one child point: x-pass over the five j-rows
// J=-2..2 (ior=2, sint.F:15), then one y-pass -- the SINT/SINTB two-pass
// structure (sint.F:66-192 / :274-341) collapsed to the single (II,JJ,IIM)
// this thread owns (per-plane writes happen only after all reads in the
// Fortran, so per-point evaluation is arithmetic-identical).
__device__ __forceinline__ float sint_point(const float* __restrict__ cfld,
                                            int k, int cjd, int cid,
                                            float ax, float ay,
                                            int nyp, int nxp)
{
    float z[5];
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
        const float* row = cfld + I3(k, cjd + j, cid - 2, nyp, nxp);
        z[j + 2] = sint_pass(row[0], row[1], row[2], row[3], row[4], ax);
    }
    return sint_pass(z[0], z[1], z[2], z[3], z[4], ay);
}

// ---------------------------------------------------------------------------
// nest_sint: full-extent child capture (interp_fcn_sint semantics,
// interp_fcn.F:874-993 -- the tile/stagger wrapper that calls SINT at :971;
// child pickup nfld(ni-ioff,nk,nj-joff) = psca(ci,cj,ip+1+jp*nri) at :985).
// The donor maps ci/ip/cj/jp already carry the wrapper's ioff/joff shift
// (built host-side, gpuwm/core/nest_interp.py).
// ---------------------------------------------------------------------------
extern "C" __global__ void nest_sint(
    const float* __restrict__ cfld,    // parent (nz, nyp, nxp)
    float* __restrict__ nfld,          // child  (nz, nyc, nxc)
    const int* __restrict__ ci_map,    // (nxc,) 0-based donor parent i
    const int* __restrict__ ip_map,    // (nxc,) sub-cell x position
    const int* __restrict__ cj_map,    // (nyc,)
    const int* __restrict__ jp_map,    // (nyc,)
    const float* __restrict__ xig,     // (nri,) FP32-stored offsets
    const float* __restrict__ xjg,     // (nrj,)
    int nz, int nyc, int nxc, int nyp, int nxp)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nz * nyc * nxc;
    if (tid >= total) return;
    int ic = (int)(tid % nxc);
    int jc = (int)((tid / nxc) % nyc);
    int k = (int)(tid / ((size_t)nxc * nyc));
    float ax = xig[ip_map[ic]];
    float ay = xjg[jp_map[jc]];
    nfld[tid] = sint_point(cfld, k, cj_map[jc], ci_map[ic], ax, ay, nyp, nxp);
}

// ---------------------------------------------------------------------------
// nest_bdy_interp1: one boundary side's VALUE and TENDENCY tables
// (bdy_interp1, interp_fcn.F:2423-2626; SINTB per side at :2539-2559).
//
//   value    = child's current state           (bdy_xs = nfld, :2584)
//   tendency = rdt * (SINT(parent) - child)    (:2583)
//
// REAL*8-rdt precision scheme, exact (F7 amendment): cdt is the
// PARENT/COARSE step ("Time step size for CG and FG", :2320; decls :2345,
// :2472); rdt = 1.D0/cdt in double (:2480, :2500); the difference is formed
// in FP32, promoted to FP64, multiplied by the FP64 reciprocal, stored FP32
// (:2583).  Table layout: west/east (nz, nyc, sz), south/north (nz, sz,
// nxc), width index 0 at the domain edge for east/north (WRF's
// bdy_xe(nj,k,nide-ni[+1]) indexing, :2593-2617) -- the Phase-4
// lateral_bc.py orientation.  side: 0=west, 1=east, 2=south, 3=north.
// ---------------------------------------------------------------------------
extern "C" __global__ void nest_bdy_interp1(
    const float* __restrict__ cfld,    // coupled parent @ t+dtp
    const float* __restrict__ nfld,    // coupled child current
    float* __restrict__ bdy_val,
    float* __restrict__ bdy_tend,
    const int* __restrict__ ci_map,    // bdy-wrapper donor maps (ioff =
    const int* __restrict__ ip_map,    //   MAX((nri-1)/2,1) on stagger,
    const int* __restrict__ cj_map,    //   interp_fcn.F:2504-2510)
    const int* __restrict__ jp_map,
    const float* __restrict__ xig,
    const float* __restrict__ xjg,
    float cdt,                         // PARENT step, REAL (:2345/:2472)
    int side, int sz,
    int nz, int nyc, int nxc, int nyp, int nxp)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int longn = (side < 2) ? nyc : nxc;
    size_t total = (size_t)nz * longn * sz;
    if (tid >= total) return;
    int w = (int)(tid % sz);
    int l = (int)((tid / sz) % longn);
    int k = (int)(tid / ((size_t)sz * longn));
    int ic, jc;
    size_t slot;
    if (side == 0) {           // WEST: ni = nids..nids+sz-1 (:2582)
        ic = w; jc = l; slot = I3(k, l, w, nyc, sz);
    } else if (side == 1) {    // EAST: width 0 at the edge (:2593-2604)
        ic = nxc - 1 - w; jc = l; slot = I3(k, l, w, nyc, sz);
    } else if (side == 2) {    // SOUTH: nj = njds..njds+sz-1 (:2588)
        ic = l; jc = w; slot = I3(k, w, l, sz, nxc);
    } else {                   // NORTH: width 0 at the edge (:2606-2617)
        ic = l; jc = nyc - 1 - w; slot = I3(k, w, l, sz, nxc);
    }
    float ax = xig[ip_map[ic]];
    float ay = xjg[jp_map[jc]];
    float psca = sint_point(cfld, k, cj_map[jc], ci_map[ic], ax, ay,
                            nyp, nxp);
    float nv = nfld[I3(k, jc, ic, nyc, nxc)];
    double rdt = 1.0 / (double)cdt;                            // :2500
    bdy_tend[slot] = (float)(rdt * (double)(psca - nv));       // :2583
    bdy_val[slot] = nv;                                        // :2584
}

// Enumerate the same independent side tables in one launch. Each side keeps
// its original coordinate decode, SINT expression and stored REAL*8 divisor.
extern "C" __global__ void nest_bdy_interp1_all_sides(
    const float* __restrict__ cfld,    // coupled parent @ t+dtp
    const float* __restrict__ nfld,    // coupled child current
    float* __restrict__ west_val, float* __restrict__ west_tend,
    float* __restrict__ east_val, float* __restrict__ east_tend,
    float* __restrict__ south_val, float* __restrict__ south_tend,
    float* __restrict__ north_val, float* __restrict__ north_tend,
    const int* __restrict__ ci_map,    // bdy-wrapper donor maps (ioff =
    const int* __restrict__ ip_map,    //   MAX((nri-1)/2,1) on stagger,
    const int* __restrict__ cj_map,    //   interp_fcn.F:2504-2510)
    const int* __restrict__ jp_map,
    const float* __restrict__ xig,
    const float* __restrict__ xjg,
    float cdt,                         // PARENT step, REAL (:2345/:2472)
    int sz,
    int nz, int nyc, int nxc, int nyp, int nxp)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ew = (size_t)nz * nyc * sz;
    size_t ns = (size_t)nz * nxc * sz;
    if (tid >= 2*(ew + ns)) return;
    int side;
    float* bdy_val;
    float* bdy_tend;
    if (tid < ew) {
        side = 0; bdy_val = west_val; bdy_tend = west_tend;
    } else if (tid < 2*ew) {
        side = 1; tid -= ew; bdy_val = east_val; bdy_tend = east_tend;
    } else if (tid < 2*ew + ns) {
        side = 2; tid -= 2*ew; bdy_val = south_val; bdy_tend = south_tend;
    } else {
        side = 3; tid -= 2*ew + ns; bdy_val = north_val; bdy_tend = north_tend;
    }
    int longn = (side < 2) ? nyc : nxc;
    int w = (int)(tid % sz);
    int l = (int)((tid / sz) % longn);
    int k = (int)(tid / ((size_t)sz * longn));
    int ic, jc;
    size_t slot;
    if (side == 0) {           // WEST: ni = nids..nids+sz-1 (:2582)
        ic = w; jc = l; slot = I3(k, l, w, nyc, sz);
    } else if (side == 1) {    // EAST: width 0 at the edge (:2593-2604)
        ic = nxc - 1 - w; jc = l; slot = I3(k, l, w, nyc, sz);
    } else if (side == 2) {    // SOUTH: nj = njds..njds+sz-1 (:2588)
        ic = l; jc = w; slot = I3(k, w, l, sz, nxc);
    } else {                   // NORTH: width 0 at the edge (:2606-2617)
        ic = l; jc = nyc - 1 - w; slot = I3(k, w, l, sz, nxc);
    }
    float ax = xig[ip_map[ic]];
    float ay = xjg[jp_map[jc]];
    float psca = sint_point(cfld, k, cj_map[jc], ci_map[ic], ax, ay,
                            nyp, nxp);
    float nv = nfld[I3(k, jc, ic, nyc, nxc)];
    double rdt = 1.0 / (double)cdt;                            // :2500
    bdy_tend[slot] = (float)(rdt * (double)(psca - nv));       // :2583
    bdy_val[slot] = nv;                                        // :2584
}

// ---------------------------------------------------------------------------
// nest_blend_terrain (nest_init_utils.F:712-785): rows <= spec_bdy_width
// take the parent-interpolated value (:766-769); the next blend_width
// frames blend linearly with weights blend_cell/(blend_width+1) (:759-765,
// r_blend_zones = 1./(blend_width+1) at :755); interior stays fine.  The
// descending blend_cell loop is transliterated so the smallest matching
// frame wins at corners exactly as the Fortran overwrite order does.
// In-place on ter_input; per-point (no neighbors), so in-place is exact.
// 1-based WRF i/ide map to 0-based i0 = i-1 with ide = nx+1.
// ---------------------------------------------------------------------------
extern "C" __global__ void nest_blend_terrain(
    const float* __restrict__ ter_interpolated,
    float* __restrict__ ter_input,
    int spec_bdy_width, int blend_width,
    int nk, int ny, int nx,
    int domain_ny, int domain_nx, int j0, int i0)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nk * ny * nx;
    if (tid >= total) return;
    int i = (int)(tid % nx) + 1 + i0;        // WRF 1-based domain i
    int j = (int)((tid / nx) % ny) + 1 + j0; // WRF 1-based domain j
    int ide = domain_nx + 1, jde = domain_ny + 1;
    float fine = ter_input[tid];
    float coarse = ter_interpolated[tid];
    float r_blend_zones = 1.0f / (float)(blend_width + 1);     // :755
    float value = fine;                                        // :742-748
    for (int blend_cell = blend_width; blend_cell >= 1; --blend_cell) {
        if (i == spec_bdy_width + blend_cell ||
            j == spec_bdy_width + blend_cell ||
            i == ide - spec_bdy_width - blend_cell ||
            j == jde - spec_bdy_width - blend_cell) {          // :760-761
            value = ((float)blend_cell * fine
                     + (float)(blend_width + 1 - blend_cell) * coarse)
                    * r_blend_zones;                           // :762-763
        }
    }
    if (i <= spec_bdy_width || j <= spec_bdy_width ||
        i >= ide - spec_bdy_width || j >= jde - spec_bdy_width) {  // :766-767
        value = coarse;                                        // :768
    }
    ter_input[tid] = value;
}

// ---------------------------------------------------------------------------
// nest_adjust_tempqv (nest_init_utils.F:812-890): correct theta/qv for the
// MUB change from terrain blending, conserving RH.  Full pressure
// p = c4(k) + c3(k)*mub + p_top + pp (:851/:867; pp is read, never
// written), two-step lapse correction (:874-875, coefficient -191.86e-3 =
// 2*(g/cp-6.5e-3)*R_dry/g per the :868 comment), Magnus constants
// 610.78/17.0809/234.175 and epsilon 0.622 as WRF literals (:857-858,
// :882-884).  Both use_theta_m branches (:852-856, :869-880).  rvord =
// R_v/R_d arrives as an FP64 argument from the constants module (never
// hardcoded).  The Fortran's znw dummy argument is unused in its body and
// is dropped.
//
// PRECISION (documented deviation, flagged for PROVENANCE by the owning
// init lane): evaluated in FP64 and stored FP32, where WRF evaluates in
// REAL.  The (th+300)*(p/1e5)**(2/7) - 273.15 chain cancels ~275 K of
// magnitude, so REAL evaluation carries an irreducible multi-100-ULP
// spread in qv that no FP64 mirror could floor at 8 ULPs; this one-shot
// init-time operator computes in double instead, keeping the pinned
// fp32_floor oracle discriminating.  Numeric consequence vs WRF REAL is
// O(1e-5) relative in the blend rows, bounded by the N1 HGT/MUB static
// oracles and the N3 statistical gates.
// ---------------------------------------------------------------------------
extern "C" __global__ void nest_adjust_tempqv(
    const float* __restrict__ mub,       // (ny, nx) blended
    const float* __restrict__ save_mub,  // (ny, nx) pre-blend
    const float* __restrict__ c3,        // (nz,) half-level hybrid c3h
    const float* __restrict__ c4,        // (nz,) half-level hybrid c4h
    double p_top,
    float* __restrict__ th,              // (nz, ny, nx) perturbation theta
    const float* __restrict__ pp,        // (nz, ny, nx) perturbation p
    float* __restrict__ qv,              // (nz, ny, nx)
    double rvord,                        // R_v/R_d (module_model_constants)
    int use_theta_m,
    int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nz * ny * nx;
    if (tid >= total) return;
    int k = (int)(tid / ((size_t)ny * nx));
    size_t col = tid % ((size_t)ny * nx);
    double mu_new = (double)mub[col];
    double mu_old = (double)save_mub[col];
    double thp = (double)th[tid];
    double ppd = (double)pp[tid];
    double qvd = (double)qv[tid];
    // Pass 1: pre-blend pressure and conserved RH (:848-862).
    double p_old = (double)c4[k] + (double)c3[k] * mu_old + p_top + ppd;
    double tc;
    if (use_theta_m == 1) {
        tc = (thp + 300.0) * pow(p_old / 1.0e5, 2.0 / 7.0)
             / (1.0 + rvord * qvd) - 273.15;                   // :853
    } else {
        tc = (thp + 300.0) * pow(p_old / 1.0e5, 2.0 / 7.0) - 273.15;  // :855
    }
    double es = 610.78 * exp(17.0809 * tc / (234.175 + tc));   // :857
    double e = qvd * p_old / (0.622 + qvd);                    // :858
    double rh = e / es;                                        // :859
    // Pass 2: post-blend pressure, theta correction, RH -> qv (:864-887).
    double p_new = (double)c4[k] + (double)c3[k] * mu_new + p_top + ppd;
    double thloc = (use_theta_m == 1) ? (thp + 300.0) / (1.0 + rvord * qvd)
                                      : (thp + 300.0);         // :870/:872
    double dth1 = -191.86e-3 * thloc / (p_new + p_old)
                  * (p_new - p_old);                           // :874
    double dth = -191.86e-3 * (thloc + 0.5 * dth1) / (p_new + p_old)
                 * (p_new - p_old);                            // :875
    double th_new = (use_theta_m == 1)
                    ? (thloc + dth) * (1.0 + rvord * qvd) - 300.0   // :877
                    : (thloc + dth) - 300.0;                   // :879
    tc = (thloc + dth) * pow(p_new / 1.0e5, 2.0 / 7.0) - 273.15;    // :881
    es = 610.78 * exp(17.0809 * tc / (234.175 + tc));          // :882
    e = rh * es;                                               // :883
    th[tid] = (float)th_new;
    qv[tid] = (float)(0.622 * e / (p_new - e));                // :884
}

// ---------------------------------------------------------------------------
// copy_fcn feedback family (dormant Phase-5b machinery, oracled at N1).
// Parent loop bounds shared by all branches (interp_fcn.F:1466/:1470 etc.):
//   ci = ipos+spec_zone .. ipos+(nide-nids)/nri - istag - spec_zone
// with istag = 0 in the staggered direction, 1 otherwise (:1459-1461).
// nide_span/njde_span are the child MASS counts ((nide-nids), (njde-njds)).
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool feedback_cell(size_t tid, int ipos, int jpos,
                                              int spec_zone,
                                              int nide_span, int njde_span,
                                              int nri, int nrj,
                                              int xstag, int ystag,
                                              int nz, int nyp, int nxp,
                                              int* ci, int* cj, int* k)
{
    int istag = xstag ? 0 : 1;                                 // :1459-1461
    int jstag = ystag ? 0 : 1;
    int ci_lo = ipos + spec_zone;
    int ci_hi = ipos + nide_span / nri - istag - spec_zone;
    int cj_lo = jpos + spec_zone;
    int cj_hi = jpos + njde_span / nrj - jstag - spec_zone;
    int ni_cells = ci_hi - ci_lo + 1;
    int nj_cells = cj_hi - cj_lo + 1;
    if (ni_cells <= 0 || nj_cells <= 0) return false;
    size_t total = (size_t)nz * nj_cells * ni_cells;
    if (tid >= total) return false;
    *ci = ci_lo + (int)(tid % ni_cells);
    *cj = cj_lo + (int)((tid / ni_cells) % nj_cells);
    *k = (int)(tid / ((size_t)ni_cells * nj_cells));
    return true;
}

// copy_fcn (interp_fcn.F:1397-1742): BOTH parity branches.  Odd mass
// (:1463-1517) and even mass (:1567-1663) cell-average all nri*nrj child
// points with weight 1./REAL(nri*nrj); u/v average 1/nri along the face
// (odd :1519-1562, even :1667-1737).  The accumulation is sequential FP32
// exactly as the Fortran DO ijpoints loop.
extern "C" __global__ void nest_copy_fcn(
    float* __restrict__ cfld,          // parent (nz, nyp, nxp)
    const float* __restrict__ nfld,    // child  (nz, nyc, nxc)
    int ipos, int jpos,                // i/j_parent_start (1-based)
    int nri, int nrj, int spec_zone,
    int xstag, int ystag,
    int nz, int nyp, int nxp, int nyc, int nxc,
    int parent_i0, int parent_j0, int child_i0, int child_j0,
    int loop_i0, int loop_j0, int loop_ni, int loop_nj)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * loop_ni * loop_nj) return;
    int ci = loop_i0 + (int)(tid % loop_ni) + 1;
    int cj = loop_j0 + (int)((tid / loop_ni) % loop_nj) + 1;
    int k = (int)(tid / ((size_t)loop_ni * loop_nj));
    bool odd = (nrj % 2) != 0;                                 // :1463
    int ni, nj, ij_lo, ij_hi, ij_stride, sub;
    float w;
    if (odd) {
        if (!xstag && !ystag) {                                // :1465-1517
            ni = (ci - ipos) * nri + nri / 2 + 1;
            nj = (cj - jpos) * nrj + nrj / 2 + 1;
            ij_lo = 1; ij_hi = nri * nrj; ij_stride = 1;       // :1473
            w = 1.0f / (float)(nri * nrj);                     // :1477
        } else if (xstag) {                                    // :1519-1539
            ni = (ci - ipos) * nri + 1;
            nj = (cj - jpos) * nrj + nrj / 2 + 1;
            ij_lo = (nri + 1) / 2;                             // :1527
            ij_hi = (nri + 1) / 2 + nri * (nri - 1);
            ij_stride = nri;
            w = 1.0f / (float)nri;                             // :1531
        } else {                                               // :1541-1562
            ni = (ci - ipos) * nri + nri / 2 + 1;
            nj = (cj - jpos) * nrj + 1;
            ij_lo = (nrj * nrj + 1) / 2 - nrj / 2;             // :1549
            ij_hi = ij_lo + nrj - 1;
            ij_stride = 1;
            w = 1.0f / (float)nrj;                             // :1553
        }
        sub = 1;   // odd branches: ipoints = MOD(ij-1,nri)+1-nri/2-1 (:1474)
    } else {
        // Even branches all anchor at the SW child of the cell: mass
        // ni = (ci-ipos)*nri + istag with istag=1 (:1648); u/v + 1
        // (:1700/:1723); nj likewise (:1644/:1696/:1719).
        ni = (ci - ipos) * nri + 1;
        nj = (cj - jpos) * nrj + 1;
        if (!xstag && !ystag) {                                // :1643-1663
            ij_lo = 1; ij_hi = nri * nrj; ij_stride = 1;       // :1650
            w = 1.0f / (float)(nri * nrj);                     // :1654
        } else if (xstag) {                                    // :1695-1713
            ij_lo = 1; ij_hi = nri * nrj; ij_stride = nri;     // :1702
            w = 1.0f / (float)nri;                             // :1706
        } else {                                               // :1717-1737
            ij_lo = 1; ij_hi = nri; ij_stride = 1;             // :1725
            w = 1.0f / (float)nri;                             // :1729
        }
        sub = 0;   // even branches: ipoints = MOD(ij-1,nri) (:1651)
    }
    float acc = 0.0f;                                          // :1472/:1649
    for (int ij = ij_lo; ij <= ij_hi; ij += ij_stride) {
        int ipoints = (ij - 1) % nri + sub * (1 - nri / 2 - 1);
        int jpoints = (ij - 1) / nri + sub * (1 - nrj / 2 - 1);
        acc = acc + w * nfld[I3(k, nj + jpoints - 1 - child_j0,
                                ni + ipoints - 1 - child_i0,
                                nyc, nxc)];                    // :1476-1477
    }
    cfld[I3(k, cj - 1 - parent_j0, ci - 1 - parent_i0, nyp, nxp)] = acc;
}

// copy_fcnm (interp_fcn.F:1747-1824): 1-pt masked-field feedback -- odd
// ratio picks the center child (ni = (ci-ipos)*nri+istag+1, :1800), even
// ratio picks the SW-corner nearest neighbor (ni = (ci-ipos)*nri+1 with
// ipoints = nri/2-1, :1812-1815).
template <typename T> __device__ void copy_one_feedback(
    T* __restrict__ cfld,
    const T* __restrict__ nfld,
    int ipos, int jpos, int nri, int nrj, int spec_zone,
    int xstag, int ystag,
    int nz, int nyp, int nxp, int nyc, int nxc,
    int parent_i0, int parent_j0, int child_i0, int child_j0,
    int loop_i0, int loop_j0, int loop_ni, int loop_nj)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)nz * loop_ni * loop_nj) return;
    int ci = loop_i0 + (int)(tid % loop_ni) + 1;
    int cj = loop_j0 + (int)((tid / loop_ni) % loop_nj) + 1;
    int k = (int)(tid / ((size_t)loop_ni * loop_nj));
    int istag = xstag ? 0 : 1;
    int jstag = ystag ? 0 : 1;
    int ni, nj;
    if ((nrj % 2) != 0) {                                      // :1793-1804
        ni = (ci - ipos) * nri + istag + 1;
        nj = (cj - jpos) * nrj + jstag + 1;
    } else {                                                   // :1806-1818
        ni = (ci - ipos) * nri + 1 + (nri / 2 - 1);
        nj = (cj - jpos) * nrj + 1 + (nrj / 2 - 1);
    }
    cfld[I3(k, cj - 1 - parent_j0, ci - 1 - parent_i0, nyp, nxp)] =
        nfld[I3(k, nj - 1 - child_j0, ni - 1 - child_i0, nyc, nxc)];
}

// Integer uses the identical index expressions, with no conversion to REAL.
#define COPY_ONE_KERNEL(name, type) \
extern "C" __global__ void name(type* cfld, const type* nfld, \
    int ipos, int jpos, int nri, int nrj, int spec_zone, int xstag, int ystag, \
    int nz, int nyp, int nxp, int nyc, int nxc, \
    int parent_i0, int parent_j0, int child_i0, int child_j0, \
    int loop_i0, int loop_j0, int loop_ni, int loop_nj) { \
    copy_one_feedback(cfld, nfld, ipos, jpos, nri, nrj, spec_zone, xstag, ystag, \
        nz, nyp, nxp, nyc, nxc, parent_i0, parent_j0, child_i0, child_j0, \
        loop_i0, loop_j0, loop_ni, loop_nj); \
}
COPY_ONE_KERNEL(nest_copy_fcnm, float)
COPY_ONE_KERNEL(nest_copy_fcni, int)
COPY_ONE_KERNEL(nest_copy_fcni64, long long)
#undef COPY_ONE_KERNEL

// ---------------------------------------------------------------------------
// The feedback smoothers (interp_fcn.F:3794-4014), applied to the parent
// AFTER the restriction unpack -- feedback_domain_em_part2.F:176-193 runs
// nest_feedbackup_smooth.inc as the last act of the feedback transaction.
// Registry flag `s` marks every fed-back field (Registry.EM_COMMON: u :159,
// v :172, w :183, ph :199, t :211, mu :288, moist :454ff), default fcn
// `smoother` (reg_parse.c:650), so the smoothed inventory IS the feedback
// inventory.
//
// Both smoothers share one two-stage shape per (pass, k):
//   stage 1 filters along J from the parent field into a scratch plane and
//     COPIES a one-cell ring around the window (sm121's cfldnew init loop,
//     :3910-3915; smdsm's committed cfld outside the window);
//   stage 2 filters along I from the scratch back into the parent field.
// That reproduces both Fortran dataflows exactly: sm121's I pass reads
// cfldnew, which holds J-filtered values in the window and the pre-pass
// copy outside it (:3927-3931); smdsm COMMITS its J pass before the I pass
// reads (:4001-4004), so in-window neighbours are J-filtered and ring
// neighbours are the original field either way.
//
// The window per axis is [pos+2, pos+span-2-stag] in WRF's 1-based domain
// cells (:3918-3921, :3993-3994), span = (n?de-n?ds)/nr?, stag = 0 on the
// field's staggered axis and 1 otherwise.  The module compiles with
// -fmad=false, so the FP32 expression order below is the rounding order.

// MOD 0: sm121 (:3864-3935), one pass.  The Fortran sums HIGH-side
//        neighbour first -- `0.25 * ( cfld(j+1) + 2.*cfld(j) + cfld(j-1) )`
//        left-associates as ((c + 2b) + a), and FP32 addition is not
//        associative, so the order below IS the value (measured: the
//        (a + 2b) + c spelling lands 1 ULP off on ~40% of cells).
// MOD 1: smdsm (:3937-4014), b + xnu*((c + a)*0.5 - b), passes 1 and 2
//        with xnu = +0.50 / -0.52 (:3976); the two-term sum commutes.
__device__ __forceinline__ float smooth_filter(int mode, float xnu,
                                               float a, float b, float c)
{
    if (mode == 0)
        return 0.25f * ((c + 2.0f * b) + a);                   // :3927/:3934
    return b + xnu * ((c + a) * 0.5f - b);                     // :3999/:4007
}

// Window decode shared by both stages: threads cover (niw + 2*ring) x
// (njw + 2*ring) x nz cells at 0-based origin (i0-ring, j0-ring).
__device__ __forceinline__ bool smooth_cell(size_t tid, int i0, int j0,
                                            int niw, int njw, int ring,
                                            int nz, int* i, int* j, int* k)
{
    int nit = niw + 2 * ring;
    int njt = njw + 2 * ring;
    size_t total = (size_t)nz * njt * nit;
    if (tid >= total) return false;
    *i = (i0 - ring) + (int)(tid % nit);
    *j = (j0 - ring) + (int)((tid / nit) % njt);
    *k = (int)(tid / ((size_t)nit * njt));
    return true;
}

// Stage 1: scratch[window] = J-filter(parent); scratch[ring] = parent.
extern "C" __global__ void nest_smooth_j(
    const float* __restrict__ cfld,    // parent (nz, nyp, nxp)
    float* __restrict__ scr,           // scratch, same extents
    int mode, float xnu,
    int i0, int j0, int niw, int njw,  // 0-based window, inclusive counts
    int nz, int nyp, int nxp)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k;
    if (!smooth_cell(tid, i0, j0, niw, njw, 1, nz, &i, &j, &k)) return;
    bool inside = (i >= i0 && i < i0 + niw && j >= j0 && j < j0 + njw);
    float b = cfld[I3(k, j, i, nyp, nxp)];
    if (!inside) { scr[I3(k, j, i, nyp, nxp)] = b; return; }
    float a = cfld[I3(k, j - 1, i, nyp, nxp)];
    float c = cfld[I3(k, j + 1, i, nyp, nxp)];
    scr[I3(k, j, i, nyp, nxp)] = smooth_filter(mode, xnu, a, b, c);
}

// Stage 2: parent[window] = I-filter(scratch).
extern "C" __global__ void nest_smooth_i(
    float* __restrict__ cfld,
    const float* __restrict__ scr,
    int mode, float xnu,
    int i0, int j0, int niw, int njw,
    int nz, int nyp, int nxp)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i, j, k;
    if (!smooth_cell(tid, i0, j0, niw, njw, 0, nz, &i, &j, &k)) return;
    float a = scr[I3(k, j, i - 1, nyp, nxp)];
    float b = scr[I3(k, j, i, nyp, nxp)];
    float c = scr[I3(k, j, i + 1, nyp, nxp)];
    cfld[I3(k, j, i, nyp, nxp)] = smooth_filter(mode, xnu, a, b, c);
}

// One block per exposed cell. Integer squared distances and the first donor
// on ties reproduce NumPy argmin without materialising a pairwise matrix.
extern "C" __global__ void relocation_nearest_donor(
    const long long* targets, const long long* donors,
    long long* picked, int donor_count) {
    __shared__ unsigned long long distances[128];
    __shared__ long long indices[128];
    const int lane = threadIdx.x;
    const long long y = targets[2 * blockIdx.x];
    const long long x = targets[2 * blockIdx.x + 1];
    unsigned long long best = ~0ULL;
    long long index = 0x7fffffffffffffffLL;
    for (long long d = lane; d < donor_count; d += blockDim.x) {
        const long long dy = y - donors[2 * d];
        const long long dx = x - donors[2 * d + 1];
        const unsigned long long distance = dy * dy + dx * dx;
        if (distance < best || (distance == best && d < index)) {
            best = distance;
            index = d;
        }
    }
    distances[lane] = best;
    indices[lane] = index;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) {
            const int other = lane + stride;
            if (distances[other] < distances[lane] ||
                (distances[other] == distances[lane] && indices[other] < indices[lane])) {
                distances[lane] = distances[other];
                indices[lane] = indices[other];
            }
        }
        __syncthreads();
    }
    if (!lane) picked[blockIdx.x] = indices[0];
}
