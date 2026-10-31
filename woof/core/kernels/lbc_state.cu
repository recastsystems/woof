// Resident, state-aware WRF specified/relaxation lateral boundaries.
// Side layout: W/E (nz,ny,width), S/N (nz,width,nx), distance 0 outermost.
// The explicit round-to-nearest intrinsics reproduce the separate CuPy
// float32 ufuncs used by the former full-domain coupling/finalization path.

enum LbcFieldKind {
    LBC_U = 0, LBC_V = 1, LBC_THETA = 2,
    LBC_PHI = 3, LBC_MU = 4, LBC_QV = 5,
    LBC_W = 6, LBC_SCALAR = 7
};

static __device__ __forceinline__
int min4(int a, int b, int c, int d)
{
    return min(min(a, b), min(c, d));
}

// Ring d of the perimeter frames: its south row (j = d), north row
// (j = ny-1-d), west column (i = d) and east column (i = nx-1-d); the rows
// own the corners.  On a whole domain every ring has two distinct rows and
// two distinct columns.  A streamed tile window narrower than two
// relaxation zones along an odd side has a middle ring whose two rows (or
// two columns) are the SAME line, and that line is listed once: listed
// twice, two threads would read, modify and write each of its cells.
static __device__ __forceinline__
int frame_rows(int d, int ny) { return (ny - 1 - d > d) ? 2 : 1; }

static __device__ __forceinline__
int frame_cols(int d, int nx) { return (nx - 1 - d > d) ? 2 : 1; }

static __device__ __forceinline__
bool frame_point(int p, int ny, int nx, int width,
                 int* j, int* i, int* distance)
{
    for (int d = 0; d < width; ++d) {
        int ni = nx - 2*d;
        int nj = max(ny - 2*d - 2, 0);
        int rows = frame_rows(d, ny);
        int ring = rows*ni + frame_cols(d, nx)*nj;
        if (p >= ring) {
            p -= ring;
            continue;
        }
        *distance = d;
        if (p < ni) {
            *j = d;
            *i = d + p;
        } else if (p < rows*ni) {
            *j = ny - 1 - d;
            *i = d + p - ni;
        } else if (p < rows*ni + nj) {
            *j = d + 1 + p - rows*ni;
            *i = d;
        } else {
            *j = d + 1 + p - rows*ni - nj;
            *i = nx - 1 - d;
        }
        return true;
    }
    return false;
}

static __device__ __forceinline__
int frame_offset(int j, int i, int ny, int nx, int width)
{
    int d = min4(j, ny - 1 - j, i, nx - 1 - i);
    if (d < 0 || d >= width) return -1;
    int offset = 0;
    for (int q = 0; q < d; ++q) {
        offset += frame_rows(q, ny)*(nx - 2*q)
                  + frame_cols(q, nx)*max(ny - 2*q - 2, 0);
    }
    int ni = nx - 2*d;
    int nj = max(ny - 2*d - 2, 0);
    int rows = frame_rows(d, ny);
    if (j == d) return offset + i - d;
    if (j == ny - 1 - d) return offset + ni + i - d;
    if (i == d) return offset + rows*ni + j - d - 1;
    return offset + rows*ni + nj + j - d - 1;
}

static __device__ __forceinline__
real current_mu(const real* mub2d, const real* mup,
                int j, int i, int nx)
{
    size_t q = (size_t)j*nx + i;
    return __fadd_rn(mub2d[q], mup[q]);
}

// WRF couple_or_uncouple_em.F keeps the base and perturbation mass terms
// separated in REAL: (c1*Mub+c2) + (c1*Mu_2).  These helpers deliberately
// do not reuse current_mu(), because c1*(Mub+Mu_2)+c2 is a different FP32
// expression tree.
static __device__ __forceinline__
real hybrid_point_current(real c1, real c2,
                          const real* mub2d, const real* mup,
                          int j, int i, int nx)
{
    size_t q = (size_t)j*nx + i;
    real base = __fadd_rn(__fmul_rn(c1, mub2d[q]), c2);
    real perturbation = __fmul_rn(c1, mup[q]);
    return __fadd_rn(base, perturbation);
}

static __device__ __forceinline__
real u_face_hybrid_current(real c1, real c2,
                           const real* mub2d, const real* mup,
                           int j, int i, int ny, int nx)
{
    if (i == 0) {
        // WRF's generic loop includes the west physical face.  Its i-1
        // halo duplicates i=0 here, but the literal four-term REAL tree
        // remains roundoff-visible.  Only the east face is repaired to the
        // one-sided point form below (couple_or_uncouple_em.F:140-166).
        size_t point = (size_t)j*nx;
        real base = __fadd_rn(__fmul_rn(c1, mub2d[point]), c2);
        real perturbation = __fmul_rn(c1, mup[point]);
        real sum = __fadd_rn(base, base);
        sum = __fadd_rn(sum, perturbation);
        sum = __fadd_rn(sum, perturbation);
        return __fmul_rn(0.5f, sum);
    }
    if (i == nx)
        return hybrid_point_current(c1, c2, mub2d, mup, j, nx - 1, nx);
    size_t right = (size_t)j*nx + i;
    size_t left = right - 1;
    real base_right = __fadd_rn(__fmul_rn(c1, mub2d[right]), c2);
    real base_left = __fadd_rn(__fmul_rn(c1, mub2d[left]), c2);
    real sum = __fadd_rn(base_right, base_left);
    sum = __fadd_rn(sum, __fmul_rn(c1, mup[right]));
    sum = __fadd_rn(sum, __fmul_rn(c1, mup[left]));
    return __fmul_rn(0.5f, sum);
}

static __device__ __forceinline__
real v_face_hybrid_current(real c1, real c2,
                           const real* mub2d, const real* mup,
                           int j, int i, int ny, int nx)
{
    if (j == 0) {
        // As for u-west, WRF evaluates its generic j/j-1 expression at the
        // south face with a duplicated mass halo.  The north face alone is
        // overwritten by WRF's one-sided repair.
        size_t point = (size_t)i;
        real base = __fadd_rn(__fmul_rn(c1, mub2d[point]), c2);
        real perturbation = __fmul_rn(c1, mup[point]);
        real sum = __fadd_rn(base, base);
        sum = __fadd_rn(sum, perturbation);
        sum = __fadd_rn(sum, perturbation);
        return __fmul_rn(0.5f, sum);
    }
    if (j == ny)
        return hybrid_point_current(c1, c2, mub2d, mup, ny - 1, i, nx);
    size_t north = (size_t)j*nx + i;
    size_t south = north - nx;
    real base_north = __fadd_rn(__fmul_rn(c1, mub2d[north]), c2);
    real base_south = __fadd_rn(__fmul_rn(c1, mub2d[south]), c2);
    real sum = __fadd_rn(base_north, base_south);
    sum = __fadd_rn(sum, __fmul_rn(c1, mup[north]));
    sum = __fadd_rn(sum, __fmul_rn(c1, mup[south]));
    return __fmul_rn(0.5f, sum);
}

static __device__ __forceinline__
real old_mu(const real* old_mup_frame,
            const real* mub2d, const real* new_mup,
            int j, int i, int ny, int nx, int spec_zone)
{
    int frame = frame_offset(j, i, ny, nx, spec_zone);
    real mup = frame >= 0 ? old_mup_frame[frame]
                          : new_mup[(size_t)j*nx + i];
    return __fadd_rn(mub2d[(size_t)j*nx + i], mup);
}

static __device__ __forceinline__
real u_face_mu_current(const real* mub2d, const real* mup,
                       int j, int i, int ny, int nx)
{
    if (i == 0) return current_mu(mub2d, mup, j, 0, nx);
    if (i == nx) return current_mu(mub2d, mup, j, nx - 1, nx);
    real right = current_mu(mub2d, mup, j, i, nx);
    real left = current_mu(mub2d, mup, j, i - 1, nx);
    return __fmul_rn(0.5f, __fadd_rn(right, left));
}

static __device__ __forceinline__
real v_face_mu_current(const real* mub2d, const real* mup,
                       int j, int i, int ny, int nx)
{
    if (j == 0) return current_mu(mub2d, mup, 0, i, nx);
    if (j == ny) return current_mu(mub2d, mup, ny - 1, i, nx);
    real north = current_mu(mub2d, mup, j, i, nx);
    real south = current_mu(mub2d, mup, j - 1, i, nx);
    return __fmul_rn(0.5f, __fadd_rn(north, south));
}

#if GPUWM_WRF_EXACT
// Strict mode: the face mass of WRF calc_mu_uv (module_big_step_utilities_em.F),
// MUU = 0.5*(MU(i)+MU(i-1)+MUB(i)+MUB(i-1)) summed left to right, and on the
// first and last face of a non-periodic domain the edge column twice
// (im = its, resp. i-1 = im = ite-1).  relax_bdy_dry relaxes WRF's grid%ru /
// grid%rv, which couple_momentum forms from exactly this MUU/MUV.  The default
// helpers above sum (MUB+MU) per column first and take MUB+MU at the edge
// faces, a different rounding (combo-sweep round 2, LOCALIZE.md item 5).
static __device__ __forceinline__
real u_face_mu_wrf(const real* mub2d, const real* mup, int j, int i, int nx)
{
    size_t r = (size_t)j*nx + min(i, nx - 1);
    size_t l = (size_t)j*nx + max(i - 1, 0);
    return __fmul_rn(0.5f, __fadd_rn(__fadd_rn(__fadd_rn(mup[r], mup[l]),
                                               mub2d[r]), mub2d[l]));
}

static __device__ __forceinline__
real v_face_mu_wrf(const real* mub2d, const real* mup, int j, int i, int ny,
                   int nx)
{
    size_t n = (size_t)min(j, ny - 1)*nx + i;
    size_t s = (size_t)max(j - 1, 0)*nx + i;
    return __fmul_rn(0.5f, __fadd_rn(__fadd_rn(__fadd_rn(mup[n], mup[s]),
                                               mub2d[n]), mub2d[s]));
}

// WRF relax_bdytend_core: field_tend = field_tend + fcx*fls0
//                                     - gcx*(fls1+fls2+fls3+fls4-4.*fls0),
// left to right.  The held rows start from zero, where this equals the
// default grouping; the mass row adds into the big-step mu tendency, where
// it does not (LOCALIZE.md item 5, MU in row 1).
#define LBC_RELAX_ADD(result, fc, gc, f0, f1, f2, f3, f4) \
    (result) = __fsub_rn(__fadd_rn((result), __fmul_rn((fc), (f0))), \
        __fmul_rn((gc), __fsub_rn(__fadd_rn(__fadd_rn(__fadd_rn((f1), (f2)), \
        (f3)), (f4)), __fmul_rn(4.0f, (f0)))))

#endif
static __device__ __forceinline__
real u_face_mu_old(const real* old_mup_frame,
                   const real* mub2d, const real* mup,
                   int j, int i, int ny, int nx, int spec_zone)
{
    if (i == 0)
        return old_mu(old_mup_frame, mub2d, mup, j, 0,
                      ny, nx, spec_zone);
    if (i == nx)
        return old_mu(old_mup_frame, mub2d, mup, j, nx - 1,
                      ny, nx, spec_zone);
    real right = old_mu(old_mup_frame, mub2d, mup, j, i,
                        ny, nx, spec_zone);
    real left = old_mu(old_mup_frame, mub2d, mup, j, i - 1,
                       ny, nx, spec_zone);
    return __fmul_rn(0.5f, __fadd_rn(right, left));
}

static __device__ __forceinline__
real v_face_mu_old(const real* old_mup_frame,
                   const real* mub2d, const real* mup,
                   int j, int i, int ny, int nx, int spec_zone)
{
    if (j == 0)
        return old_mu(old_mup_frame, mub2d, mup, 0, i,
                      ny, nx, spec_zone);
    if (j == ny)
        return old_mu(old_mup_frame, mub2d, mup, ny - 1, i,
                      ny, nx, spec_zone);
    real north = old_mu(old_mup_frame, mub2d, mup, j, i,
                        ny, nx, spec_zone);
    real south = old_mu(old_mup_frame, mub2d, mup, j - 1, i,
                        ny, nx, spec_zone);
    return __fmul_rn(0.5f, __fadd_rn(north, south));
}

static __device__ __forceinline__
real coupled_current(int kind, int k, int j, int i,
                     const real* mub2d, const real* mup,
                     const real* u, const real* v, const real* w,
                     const real* thp, const real* thb,
                     const real* php, const real* scalar,
                     const real* c1h, const real* c2h,
                     const real* c1f, const real* c2f,
                     const real* msft, const real* msfu, const real* msfv,
                     int has_msf, int thb_3d,
                     int ny, int nx)
{
    int mny = ny - (kind == LBC_V);
    int mnx = nx - (kind == LBC_U);
    size_t idx = I3(k, j, i, ny, nx);
    if (kind == LBC_MU) return mup[(size_t)j*mnx + i];
    real mass;
    real ch;
    if (kind == LBC_U) {
#if GPUWM_WRF_EXACT
        mass = u_face_mu_wrf(mub2d, mup, j, i, mnx);
#else
        mass = u_face_mu_current(mub2d, mup, j, i, mny, mnx);
#endif
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real result = __fmul_rn(ch, u[idx]);
        return has_msf ? __fdiv_rn(result, msfu[(size_t)j*nx + i])
                       : result;
    }
    if (kind == LBC_V) {
#if GPUWM_WRF_EXACT
        mass = v_face_mu_wrf(mub2d, mup, j, i, mny, mnx);
#else
        mass = v_face_mu_current(mub2d, mup, j, i, mny, mnx);
#endif
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real result = __fmul_rn(ch, v[idx]);
#if GPUWM_WRF_EXACT
        // couple_momentum: rv = v*(c1h*muv+c2h)*msfv_inv, the stored inverse.
        return has_msf ? __fmul_rn(result,
                                   msfv[(size_t)j*nx + i])
                       : result;
#else
        return has_msf ? __fdiv_rn(result, msfv[(size_t)j*nx + i])
                       : result;
#endif
    }
    mass = current_mu(mub2d, mup, j, i, mnx);
    if (kind == LBC_W) {
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        // WRF relax_bdy_dry -> mass_weight has no map-factor argument.
        // The force-time w tables remain /msfty in couple_nest_field below.
        return __fmul_rn(ch, w[idx]);
    }
    if (kind == LBC_PHI) {
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        return __fmul_rn(ch, php[idx]);
    }
    ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
    if (kind == LBC_THETA) {
#if GPUWM_WRF_EXACT_C_BIGSTEP
        return __fmul_rn(ch, thp[idx]);
#else
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        real total_theta = __fadd_rn(thb[h], thp[idx]);
        return __fmul_rn(ch, __fsub_rn(total_theta, 300.0f));
#endif
    }
    return __fmul_rn(ch, scalar[idx]);
}

static __device__ __forceinline__
real lbc_value(const real* value, const real* tendency, size_t index, real dtbc)
{
    // Matches the existing spec_bdy.cu expression (including compiler FMA
    // policy) used by relaxation forcing.
    return value[index] + dtbc * tendency[index];
}

static __device__ __forceinline__
real installed_value(const real* value, const real* tendency,
                     size_t index, real dtbc)
{
    // spec_bdy_final formerly used two separate CuPy ufuncs.
    return __fadd_rn(value[index], __fmul_rn(dtbc, tendency[index]));
}

static __device__ __forceinline__
bool boundary_index(int k, int j, int i, int ny, int nx, int spec_zone,
                    int width, const real* west, const real* west_t,
                    const real* east, const real* east_t,
                    const real* south, const real* south_t,
                    const real* north, const real* north_t,
                    real dtbc, real* value, bool interpolate)
{
    int ds = j;
    int dn = ny - 1 - j;
    int dw = i;
    int de = nx - 1 - i;
    const real* bvalue = nullptr;
    const real* btend = nullptr;
    size_t b = 0;
    if (ds < spec_zone && i >= ds && i < nx - ds) {
        b = ((size_t)k*width + ds)*nx + i;
        bvalue = south; btend = south_t;
    } else if (dn < spec_zone && i >= dn && i < nx - dn) {
        b = ((size_t)k*width + dn)*nx + i;
        bvalue = north; btend = north_t;
    } else if (dw < spec_zone && j >= dw + 1 && j < ny - dw - 1) {
        b = ((size_t)k*ny + j)*width + dw;
        bvalue = west; btend = west_t;
    } else if (de < spec_zone && j >= de + 1 && j < ny - de - 1) {
        b = ((size_t)k*ny + j)*width + de;
        bvalue = east; btend = east_t;
    } else {
        return false;
    }
    *value = interpolate ? installed_value(bvalue, btend, b, dtbc)
                         : btend[b];
    return true;
}

extern "C" __global__
void state_specified_relaxation(
    real* __restrict__ field_tend,
    const real* __restrict__ held,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ u,
    const real* __restrict__ v,
    const real* __restrict__ w,
    const real* __restrict__ thp,
    const real* __restrict__ thb,
    const real* __restrict__ php,
    const real* __restrict__ scalar,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    const real* __restrict__ west,
    const real* __restrict__ west_t,
    const real* __restrict__ east,
    const real* __restrict__ east_t,
    const real* __restrict__ south,
    const real* __restrict__ south_t,
    const real* __restrict__ north,
    const real* __restrict__ north_t,
    const real* __restrict__ fcx,
    const real* __restrict__ gcx,
    real dtbc, int width, int spec_zone, int relax_zone,
    int apply_relax, int clear_specified, int add_held, int divide_msf,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int frame_count, int relax_sides)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*frame_count) return;
    int k = tid/frame_count;
    int p = tid - k*frame_count;
    int j, i, mapped_d;
    if (!frame_point(p, ny, nx, max(spec_zone, relax_zone),
                     &j, &i, &mapped_d)) return;
    // relax_sides: bit 1 south, 2 north, 4 west, 8 east.  15 on a whole
    // domain, where every test below reduces to WRF's own corner rule
    // (Y sides own the corners).  A streamed tile clears the bit of each
    // interior seam: a seam relaxes nothing, and its distance stops
    // bounding which true edge owns a corner cell, because in the domain
    // that cell is nowhere near the seam's side.
    const bool own_s = (relax_sides & 1) != 0;
    const bool own_n = (relax_sides & 2) != 0;
    const bool own_w = (relax_sides & 4) != 0;
    const bool own_e = (relax_sides & 8) != 0;
    size_t idx = I3(k, j, i, ny, nx);
    real result = field_tend[idx];
    real specified;
    if (boundary_index(k, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &specified, false)) {
        result = clear_specified ? 0.0f : specified;
    } else if (apply_relax) {
        int ds = j;
        int dn = ny - 1 - j;
        int dw = i;
        int de = nx - 1 - i;
        real f0, f1, f2, f3, f4;
        if (own_s && ds >= spec_zone && ds < relax_zone
            && (!own_w || i >= ds) && (!own_e || i < nx - ds)) {
            int d = ds;
            size_t b0 = ((size_t)k*width + d)*nx + i;
            size_t bm = ((size_t)k*width + d)*nx + max(i - 1, 0);
            size_t bp = ((size_t)k*width + d)*nx + min(i + 1, nx - 1);
            size_t bo = ((size_t)k*width + d - 1)*nx + i;
            size_t bi = ((size_t)k*width + d + 1)*nx + i;
            f0 = lbc_value(south, south_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(south, south_t, bm, dtbc)
                 - coupled_current(kind, k, j, max(i - 1, 0), mub2d, mup,
                                   u, v, w, thp, thb, php, scalar, c1h, c2h,
                                   c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
                                   ny, nx);
            f2 = lbc_value(south, south_t, bp, dtbc)
                 - coupled_current(kind, k, j, min(i + 1, nx - 1),
                                   mub2d, mup, u, v, w, thp, thb, php, scalar,
                                   c1h, c2h, c1f, c2f, msft, msfu, msfv,
                                   has_msf, thb_3d, ny, nx);
            f3 = lbc_value(south, south_t, bo, dtbc)
                 - coupled_current(kind, k, j - 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(south, south_t, bi, dtbc)
                 - coupled_current(kind, k, j + 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
#if GPUWM_WRF_EXACT
            LBC_RELAX_ADD(result, fcx[d], gcx[d], f0, f1, f2, f3, f4);
#else
            result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
#endif
        } else if (own_n && dn >= spec_zone && dn < relax_zone
                   && (!own_w || i >= dn) && (!own_e || i < nx - dn)) {
            int d = dn;
            size_t b0 = ((size_t)k*width + d)*nx + i;
            size_t bm = ((size_t)k*width + d)*nx + max(i - 1, 0);
            size_t bp = ((size_t)k*width + d)*nx + min(i + 1, nx - 1);
            size_t bo = ((size_t)k*width + d - 1)*nx + i;
            size_t bi = ((size_t)k*width + d + 1)*nx + i;
            f0 = lbc_value(north, north_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(north, north_t, bm, dtbc)
                 - coupled_current(kind, k, j, max(i - 1, 0), mub2d, mup,
                                   u, v, w, thp, thb, php, scalar, c1h, c2h,
                                   c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
                                   ny, nx);
            f2 = lbc_value(north, north_t, bp, dtbc)
                 - coupled_current(kind, k, j, min(i + 1, nx - 1),
                                   mub2d, mup, u, v, w, thp, thb, php, scalar,
                                   c1h, c2h, c1f, c2f, msft, msfu, msfv,
                                   has_msf, thb_3d, ny, nx);
            f3 = lbc_value(north, north_t, bo, dtbc)
                 - coupled_current(kind, k, j + 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(north, north_t, bi, dtbc)
                 - coupled_current(kind, k, j - 1, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
#if GPUWM_WRF_EXACT
            LBC_RELAX_ADD(result, fcx[d], gcx[d], f0, f1, f2, f3, f4);
#else
            result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
#endif
        } else if (own_w && dw >= spec_zone && dw < relax_zone
                   && (!own_s || j >= dw + 1)
                   && (!own_n || j < ny - dw - 1)) {
            int d = dw;
            // Clamped like the south/north branches' i neighbours.  Inert
            // on a whole domain, where j >= dw + 1 >= 2 and
            // j <= ny - dw - 2; reachable only on a tile whose south or
            // north side is a seam, at the tile's own outer row.
            int jm = max(j - 1, 0);
            int jp = min(j + 1, ny - 1);
            size_t b0 = ((size_t)k*ny + j)*width + d;
            size_t bm = ((size_t)k*ny + jm)*width + d;
            size_t bp = ((size_t)k*ny + jp)*width + d;
            size_t bo = ((size_t)k*ny + j)*width + d - 1;
            size_t bi = ((size_t)k*ny + j)*width + d + 1;
            f0 = lbc_value(west, west_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(west, west_t, bm, dtbc)
                 - coupled_current(kind, k, jm, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f2 = lbc_value(west, west_t, bp, dtbc)
                 - coupled_current(kind, k, jp, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f3 = lbc_value(west, west_t, bo, dtbc)
                 - coupled_current(kind, k, j, i - 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(west, west_t, bi, dtbc)
                 - coupled_current(kind, k, j, i + 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
#if GPUWM_WRF_EXACT
            LBC_RELAX_ADD(result, fcx[d], gcx[d], f0, f1, f2, f3, f4);
#else
            result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
#endif
        } else if (own_e && de >= spec_zone && de < relax_zone
                   && (!own_s || j >= de + 1)
                   && (!own_n || j < ny - de - 1)) {
            int d = de;
            int jm = max(j - 1, 0);
            int jp = min(j + 1, ny - 1);
            size_t b0 = ((size_t)k*ny + j)*width + d;
            size_t bm = ((size_t)k*ny + jm)*width + d;
            size_t bp = ((size_t)k*ny + jp)*width + d;
            size_t bo = ((size_t)k*ny + j)*width + d - 1;
            size_t bi = ((size_t)k*ny + j)*width + d + 1;
            f0 = lbc_value(east, east_t, b0, dtbc)
                 - coupled_current(kind, k, j, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f1 = lbc_value(east, east_t, bm, dtbc)
                 - coupled_current(kind, k, jm, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f2 = lbc_value(east, east_t, bp, dtbc)
                 - coupled_current(kind, k, jp, i, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f3 = lbc_value(east, east_t, bo, dtbc)
                 - coupled_current(kind, k, j, i + 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
            f4 = lbc_value(east, east_t, bi, dtbc)
                 - coupled_current(kind, k, j, i - 1, mub2d, mup, u, v, w,
                                   thp, thb, php, scalar, c1h, c2h, c1f, c2f,
                                   msft, msfu, msfv, has_msf, thb_3d, ny, nx);
#if GPUWM_WRF_EXACT
            LBC_RELAX_ADD(result, fcx[d], gcx[d], f0, f1, f2, f3, f4);
#else
            result += fcx[d]*f0 - gcx[d]*(f1+f2+f3+f4-4.0f*f0);
#endif
        }
    }
    if (divide_msf) result = __fdiv_rn(result, msft[(size_t)j*nx + i]);
    if (add_held) result = __fadd_rn(result, held[idx]);
    field_tend[idx] = result;
}

extern "C" __global__
void install_mu_boundary(
    real* __restrict__ mup,
    real* __restrict__ old_mup_frame,
    const real* __restrict__ west,
    const real* __restrict__ west_t,
    const real* __restrict__ east,
    const real* __restrict__ east_t,
    const real* __restrict__ south,
    const real* __restrict__ south_t,
    const real* __restrict__ north,
    const real* __restrict__ north_t,
    real dtbc, int width, int spec_zone, int ny, int nx, int frame_count)
{
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= frame_count) return;
    int j, i, d;
    if (!frame_point(p, ny, nx, spec_zone, &j, &i, &d)) return;
    size_t idx = (size_t)j*nx + i;
    old_mup_frame[p] = mup[idx];
    real value;
    if (boundary_index(0, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &value, true)) {
        mup[idx] = value;
    }
}

// F16 force-time coupling: one caller-owned full field at a time.  The
// output is a shared arena prefix and is overwritten before every read.
// WRF v4.6.1 coupling authority: dyn_em/couple_or_uncouple_em.F:119-144
// (c1h/c1f mass and u/v/w map factors), nested physical-face repair
// :149-166, and uncoupling inverses :193-223/:228-245.  Full-level w is
// subsequently relaxed only on nests at module_bc_em.F:320-345.  gpuwm
// transliterates WRF's separated Mub/Mu_2 REAL expression tree without
// mutating the source state.  WRF stores the map-factor-divided u/v/w
// weights at :123/:143-144 before multiplying the fields at :279/:288-289.
extern "C" __global__
void couple_nest_field(
    const real* __restrict__ target,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ thb,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    real* __restrict__ out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*ny*nx) return;
    int k = tid/(ny*nx);
    int rem = tid - k*ny*nx;
    int j = rem/nx;
    int i = rem - j*nx;
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    real ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfu[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfv[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msft[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    real value = target[tid];
    if (kind == LBC_THETA) {
#if !GPUWM_WRF_EXACT_C_BIGSTEP
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        value = __fsub_rn(__fadd_rn(thb[h], value), 300.0f);
#endif
    }
    out[tid] = __fmul_rn(ch, value);
}

// Same expression tree as the full-field coupling above. Only enumeration
// changes; global coordinates preserve physical-face clamps and map factors.
extern "C" __global__
void couple_nest_field_window(
    const real* __restrict__ target,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ thb,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    real* __restrict__ out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx,
    int j0, int i0, int nj, int ni)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*nj*ni) return;
    int k = tid/(nj*ni);
    int rem = tid - k*nj*ni;
    int j = j0 + rem/ni;
    int i = i0 + rem%ni;
    tid = I3(k, j, i, ny, nx);
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    real ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfu[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfv[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msft[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    real value = target[tid];
    if (kind == LBC_THETA) {
#if !GPUWM_WRF_EXACT_C_BIGSTEP
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        value = __fsub_rn(__fadd_rn(thb[h], value), 300.0f);
#endif
    }
    out[tid] = __fmul_rn(ch, value);
}

// Boundary interpolation reads only this child frame. The existing ring
// enumerator gives each corner one writer and retains physical coordinates.
extern "C" __global__
void couple_nest_field_frame(
    const real* __restrict__ target,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ thb,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    real* __restrict__ out,
    int has_msf, int thb_3d, int kind,
    int nz, int ny, int nx, int mny, int mnx,
    int width, int points)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*points) return;
    int k = tid/points;
    int j, i, distance;
    if (!frame_point(tid - k*points, ny, nx, width, &j, &i, &distance)) return;
    tid = I3(k, j, i, ny, nx);
    if (kind == LBC_MU) {
        out[tid] = target[tid];
        return;
    }
    real ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfu[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msfv[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            ch = __fdiv_rn(ch, msft[(size_t)j*nx + i]);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[tid] = __fmul_rn(ch, target[tid]);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    real value = target[tid];
    if (kind == LBC_THETA) {
#if !GPUWM_WRF_EXACT_C_BIGSTEP
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        value = __fsub_rn(__fadd_rn(thb[h], value), 300.0f);
#endif
    }
    out[tid] = __fmul_rn(ch, value);
}

// Child-to-parent feedback inverse. copy_fcn has already written the
// restricted coupled values into the parent-shaped `coupled` buffer and MU
// has already been restricted in the live parent. Only the registered
// parent overlap is visited, so no couple/uncouple round trip can perturb
// the one-way exterior.
extern "C" __global__
void uncouple_feedback_field(
    real* __restrict__ target,
    const real* __restrict__ coupled,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ thb,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    int has_msf, int thb_3d, int kind,
    int i_lo, int i_hi, int j_lo, int j_hi,
    int nz, int ny, int nx, int mny, int mnx)
{
    int ni = i_hi - i_lo + 1;
    int nj = j_hi - j_lo + 1;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*nj*ni) return;
    int k = tid/(nj*ni);
    int rem = tid - k*nj*ni;
    int j = j_lo + rem/ni;
    int i = i_lo + rem%ni;
    size_t idx = I3(k, j, i, ny, nx);
    real value = coupled[idx];
    real ch;
    if (kind == LBC_U) {
        ch = u_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            value = __fmul_rn(value, msfu[(size_t)j*nx + i]);
        target[idx] = __fdiv_rn(value, ch);
        return;
    }
    if (kind == LBC_V) {
        ch = v_face_hybrid_current(
            c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf)
            value = __fmul_rn(value, msfv[(size_t)j*nx + i]);
        target[idx] = __fdiv_rn(value, ch);
        return;
    }
    if (kind == LBC_W) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf)
            value = __fmul_rn(value, msft[(size_t)j*nx + i]);
        target[idx] = __fdiv_rn(value, ch);
        return;
    }
    if (kind == LBC_PHI) {
        ch = hybrid_point_current(
            c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        target[idx] = __fdiv_rn(value, ch);
        return;
    }
    ch = hybrid_point_current(
        c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    real result = __fdiv_rn(value, ch);
    if (kind == LBC_THETA) {
#if !GPUWM_WRF_EXACT_C_BIGSTEP
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        result = __fsub_rn(__fadd_rn(result, 300.0f), thb[h]);
#endif
    }
    target[idx] = result;
}

static __device__ __forceinline__
real coupled_old_target(int kind, int k, int j, int i,
                        const real* target,
                        const real* old_mup_frame,
                        const real* mub2d, const real* mup,
                        const real* thb, const real* c1h, const real* c2h,
                        const real* c1f, const real* c2f,
                        const real* msft, const real* msfu, const real* msfv,
                        int has_msf, int thb_3d, int spec_zone,
                        int ny, int nx, int mny, int mnx)
{
    size_t idx = I3(k, j, i, ny, nx);
    real mass;
    real ch;
    if (kind == LBC_U) {
        mass = u_face_mu_old(old_mup_frame, mub2d, mup, j, i,
                             mny, mnx, spec_zone);
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real result = __fmul_rn(ch, target[idx]);
        return has_msf ? __fdiv_rn(result, msfu[(size_t)j*nx + i])
                       : result;
    }
    if (kind == LBC_V) {
        mass = v_face_mu_old(old_mup_frame, mub2d, mup, j, i,
                             mny, mnx, spec_zone);
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real result = __fmul_rn(ch, target[idx]);
        return has_msf ? __fdiv_rn(result, msfv[(size_t)j*nx + i])
                       : result;
    }
    mass = old_mu(old_mup_frame, mub2d, mup, j, i,
                  mny, mnx, spec_zone);
    if (kind == LBC_W) {
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        real result = __fmul_rn(ch, target[idx]);
        return has_msf ? __fdiv_rn(result, msft[(size_t)j*nx + i])
                       : result;
    }
    if (kind == LBC_PHI) {
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        return __fmul_rn(ch, target[idx]);
    }
    ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
    if (kind == LBC_THETA) {
#if GPUWM_WRF_EXACT_C_BIGSTEP
        return __fmul_rn(ch, target[idx]);
#else
        size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
        real total_theta = __fadd_rn(thb[h], target[idx]);
        return __fmul_rn(ch, __fsub_rn(total_theta, 300.0f));
#endif
    }
    return __fmul_rn(ch, target[idx]);
}

extern "C" __global__
void finalize_state_field(
    real* __restrict__ target,
    const real* __restrict__ old_mup_frame,
    const real* __restrict__ mub2d,
    const real* __restrict__ mup,
    const real* __restrict__ thb,
    const real* __restrict__ scalar_unused,
    const real* __restrict__ c1h,
    const real* __restrict__ c2h,
    const real* __restrict__ c1f,
    const real* __restrict__ c2f,
    const real* __restrict__ msft,
    const real* __restrict__ msfu,
    const real* __restrict__ msfv,
    const real* __restrict__ west,
    const real* __restrict__ west_t,
    const real* __restrict__ east,
    const real* __restrict__ east_t,
    const real* __restrict__ south,
    const real* __restrict__ south_t,
    const real* __restrict__ north,
    const real* __restrict__ north_t,
    real dtbc, int width, int spec_zone, int has_msf, int thb_3d,
#if GPUWM_WRF_EXACT
    int kind, int nz, int ny, int nx, int mny, int mnx,
    const real* __restrict__ muts_wrf)
#else
    int kind, int nz, int ny, int nx, int mny, int mnx)
#endif
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= nz*ny*nx) return;
    int k = tid/(ny*nx);
    int rem = tid - k*ny*nx;
    int j = rem/nx;
    int i = rem - j*nx;
#if GPUWM_WRF_EXACT
    // WRF spec_bdy_final writes the specified rows only:
    // field = xmsf*bfield/xmu, bfield = bdy + dtbc*bdy_tend, xmsf the
    // field's map factor for u/v/w (1 otherwise), xmu = c1*mu+c2 with mu
    // the acoustic loop's grid%muts (t, ph, w, moist) or its calc_mu_uv_1
    // faces grid%muus/muvs (u, v) -- not the installed new MU.  Every other
    // cell keeps its value: re-coupling and uncoupling the whole field, the
    // default path, moved interior words by an ULP at every step end
    // (combo-sweep round 3).
    if (muts_wrf != nullptr) {
        real bfield;
        if (!boundary_index(k, j, i, ny, nx, spec_zone, width,
                            west, west_t, east, east_t, south, south_t,
                            north, north_t, dtbc, &bfield, true))
            return;
        real mass;
        if (kind == LBC_U) {
            size_t r = (size_t)j*mnx + min(i, mnx - 1);
            size_t l = (size_t)j*mnx + max(i - 1, 0);
            mass = __fmul_rn(0.5f, __fadd_rn(muts_wrf[r], muts_wrf[l]));
        } else if (kind == LBC_V) {
            size_t n = (size_t)min(j, mny - 1)*mnx + i;
            size_t s2 = (size_t)max(j - 1, 0)*mnx + i;
            mass = __fmul_rn(0.5f, __fadd_rn(muts_wrf[n], muts_wrf[s2]));
        } else {
            mass = muts_wrf[(size_t)j*mnx + i];
        }
        bool full = kind == LBC_W || kind == LBC_PHI;
        real xmu = __fadd_rn(__fmul_rn(full ? c1f[k] : c1h[k], mass),
                             full ? c2f[k] : c2h[k]);
        real num = bfield;
        if (has_msf) {
            if (kind == LBC_U) num = __fmul_rn(msfu[(size_t)j*nx + i], num);
            else if (kind == LBC_V) num = __fmul_rn(msfv[(size_t)j*nx + i], num);
            else if (kind == LBC_W) num = __fmul_rn(msft[(size_t)j*nx + i], num);
        }
        target[tid] = __fdiv_rn(num, xmu);
        return;
    }
#endif
#if GPUWM_WRF_EXACT_C_BIGSTEP
    real coupled = coupled_old_target(
        kind, k, j, i, target, old_mup_frame, mub2d, mup, thb,
        c1h, c2h, c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
        spec_zone, ny, nx, mny, mnx);
    real installed;
    if (boundary_index(k, j, i, ny, nx, spec_zone, width,
                       west, west_t, east, east_t, south, south_t,
                       north, north_t, dtbc, &installed, true)) {
        coupled = installed;
    }
#else
    real coupled;
    // The installed value replaces the old coupling at specified cells.
    // Do not load or couple a value that has no consumer.
    if (!boundary_index(k, j, i, ny, nx, spec_zone, width,
                        west, west_t, east, east_t, south, south_t,
                        north, north_t, dtbc, &coupled, true)) {
        coupled = coupled_old_target(
            kind, k, j, i, target, old_mup_frame, mub2d, mup, thb,
            c1h, c2h, c1f, c2f, msft, msfu, msfv, has_msf, thb_3d,
            spec_zone, ny, nx, mny, mnx);
    }
#endif
    real mass;
    real ch;
    if (kind == LBC_U) {
        mass = u_face_mu_current(mub2d, mup, j, i, mny, mnx);
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real remapped = __fmul_rn(coupled, msfu[(size_t)j*nx + i]);
        target[tid] = __fdiv_rn(remapped, ch);
    } else if (kind == LBC_V) {
        mass = v_face_mu_current(mub2d, mup, j, i, mny, mnx);
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real remapped = __fmul_rn(coupled, msfv[(size_t)j*nx + i]);
        target[tid] = __fdiv_rn(remapped, ch);
    } else if (kind == LBC_PHI) {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        target[tid] = __fdiv_rn(coupled, ch);
    } else if (kind == LBC_W) {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = __fadd_rn(__fmul_rn(c1f[k], mass), c2f[k]);
        real remapped = has_msf
            ? __fmul_rn(coupled, msft[(size_t)j*nx + i]) : coupled;
        target[tid] = __fdiv_rn(remapped, ch);
    } else {
        mass = current_mu(mub2d, mup, j, i, mnx);
        ch = __fadd_rn(__fmul_rn(c1h[k], mass), c2h[k]);
        real result = __fdiv_rn(coupled, ch);
        if (kind == LBC_THETA) {
#if !GPUWM_WRF_EXACT_C_BIGSTEP
            size_t h = thb_3d ? I3(k, j, i, mny, mnx) : (size_t)k;
            result = __fsub_rn(__fadd_rn(result, 300.0f), thb[h]);
#endif
        } else if (!isnan(result)) {
            result = fmaxf(result, 0.0f);
        }
        target[tid] = result;
    }
}
