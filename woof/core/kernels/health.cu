// Combined integration-health reductions.  Two device launches produce one
// compact host record for u/w/theta maxima, co-located vertical CFL rate,
// w argmax, and real74 boundary / interior w maxima.  Max comparisons are
// exact for non-negative FP32 abs values; the lowest 64-bit flat index wins
// ties, matching argmax.  The compact float record stores that index as two
// exact uint32 payload words.  Partial records are [u, w, theta, edge-w,
// interior-w, max(|w_upper|/dz_cell), index-lo, index-hi, mask]; the final
// record drops the mask and keeps the first eight fields.

#define HEALTH_THREADS 256
#define HEALTH_FIELDS 9

static __device__ __forceinline__
void update_max(real value, unsigned long long index, real& current,
                unsigned long long& current_index)
{
    if (value > current || (value == current && index < current_index)) {
        current = value;
        current_index = index;
    }
}

extern "C" __global__
void health_partial(const real* __restrict__ u,
                    const real* __restrict__ w,
                    const real* __restrict__ thp,
                    const real* __restrict__ ph,
                    const real* __restrict__ phb,
                    real* __restrict__ partial,
                    unsigned long long nu, unsigned long long nw,
                    unsigned long long nth,
                    unsigned long long ncells,
                    int phb_full, int width, int ny, int nx, real gravity)
{
    real umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    real edge = 0.0f, interior = 0.0f, vertical_rate = 0.0f;
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    unsigned long long start = ((unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x);
    unsigned long long stride = ((unsigned long long)gridDim.x * blockDim.x);

    for (unsigned long long ix = start; ix < nu; ix += stride) {
        real value = fabsf(u[ix]);
        if (isnan(value)) mask |= 1u;
        else if (value > umax) umax = value;
    }
    for (unsigned long long ix = start; ix < nw; ix += stride) {
        real value = fabsf(w[ix]);
        bool boundary = false;
        if (width > 0) {
            unsigned long long c = ix % ((unsigned long long)ny * nx);
            int j = (int)(c / nx), i = (int)(c - (unsigned long long)j * nx);
            boundary = (j < width || j >= ny - width
                        || i < width || i >= nx - width);
        }
        if (isnan(value)) {
            mask |= 2u;
            if (width > 0) mask |= boundary ? 8u : 16u;
        } else {
            update_max(value, ix, wmax, windex);
            if (width > 0) {
                if (boundary) edge = value > edge ? value : edge;
                else interior = value > interior ? value : interior;
            }
        }
    }
    for (unsigned long long ix = start; ix < nth; ix += stride) {
        real value = fabsf(thp[ix]);
        if (isnan(value)) mask |= 4u;
        else if (value > thmax) thmax = value;
    }
    // One mass cell owns dz between its lower/upper geopotential faces and
    // the w value on its upper face.  Reducing this ratio preserves
    // co-location: a strong upper-level updraft over a thick layer can no
    // longer be paired with an unrelated thin surface layer.
    unsigned long long plane = (unsigned long long)ny * nx;
    for (unsigned long long ix = start; ix < ncells; ix += stride) {
        unsigned long long k = ix / plane;
        unsigned long long upper = ix + plane;
        real base_lower = phb_full ? phb[ix] : phb[k];
        real base_upper = phb_full ? phb[upper] : phb[k + 1];
        real dz = ((ph[upper] + base_upper) - (ph[ix] + base_lower))
                  / gravity;
        real speed = fabsf(w[upper]);
        if (!isfinite(dz) || dz <= 0.0f || !isfinite(speed)) {
            mask |= 32u;
        } else {
            real rate = speed / dz;
            vertical_rate = rate > vertical_rate ? rate : vertical_rate;
        }
    }

    __shared__ real values[6][HEALTH_THREADS];
    __shared__ unsigned long long indices[HEALTH_THREADS];
    __shared__ unsigned masks[HEALTH_THREADS];
    int lane = threadIdx.x;
    values[0][lane] = umax;
    values[1][lane] = wmax;
    values[2][lane] = thmax;
    values[3][lane] = edge;
    values[4][lane] = interior;
    values[5][lane] = vertical_rate;
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            update_max(values[1][lane + offset], indices[lane + offset],
                       values[1][lane], indices[lane]);
            values[2][lane] = values[2][lane + offset] > values[2][lane]
                              ? values[2][lane + offset] : values[2][lane];
            values[3][lane] = values[3][lane + offset] > values[3][lane]
                              ? values[3][lane + offset] : values[3][lane];
            values[4][lane] = values[4][lane + offset] > values[4][lane]
                              ? values[4][lane + offset] : values[4][lane];
            values[5][lane] = values[5][lane + offset] > values[5][lane]
                              ? values[5][lane + offset] : values[5][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        size_t out = (size_t)blockIdx.x * HEALTH_FIELDS;
        for (int field = 0; field < 6; ++field)
            partial[out + field] = values[field][0];
        partial[out + 6] = __uint_as_float((unsigned)indices[0]);
        partial[out + 7] = __uint_as_float((unsigned)(indices[0] >> 32));
        partial[out + 8] = __uint_as_float(masks[0]);
    }
}

extern "C" __global__
void health_final(const real* __restrict__ partial,
                  real* __restrict__ result, int nblocks)
{
    real maxima[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    int lane = threadIdx.x;
    for (int block = lane; block < nblocks; block += HEALTH_THREADS) {
        size_t src = (size_t)block * HEALTH_FIELDS;
        maxima[0] = partial[src] > maxima[0] ? partial[src] : maxima[0];
        unsigned long long index = __float_as_uint(partial[src + 6]);
        index |= ((unsigned long long)__float_as_uint(partial[src + 7])) << 32;
        update_max(partial[src + 1], index, maxima[1], windex);
        for (int field = 2; field < 6; ++field)
            maxima[field] = partial[src + field] > maxima[field]
                            ? partial[src + field] : maxima[field];
        mask |= __float_as_uint(partial[src + 8]);
    }

    __shared__ real values[6][HEALTH_THREADS];
    __shared__ unsigned long long indices[HEALTH_THREADS];
    __shared__ unsigned masks[HEALTH_THREADS];
    for (int field = 0; field < 6; ++field)
        values[field][lane] = maxima[field];
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            update_max(values[1][lane + offset], indices[lane + offset],
                       values[1][lane], indices[lane]);
            values[2][lane] = values[2][lane + offset] > values[2][lane]
                              ? values[2][lane + offset] : values[2][lane];
            values[3][lane] = values[3][lane + offset] > values[3][lane]
                              ? values[3][lane + offset] : values[3][lane];
            values[4][lane] = values[4][lane + offset] > values[4][lane]
                              ? values[4][lane + offset] : values[4][lane];
            values[5][lane] = values[5][lane + offset] > values[5][lane]
                              ? values[5][lane + offset] : values[5][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        result[0] = (masks[0] & 1u) ? nanf("") : values[0][0];
        result[1] = (masks[0] & 2u) ? nanf("") : values[1][0];
        result[2] = (masks[0] & 4u) ? nanf("") : values[2][0];
        result[3] = (masks[0] & 8u) ? nanf("") : values[3][0];
        result[4] = (masks[0] & 16u) ? nanf("") : values[4][0];
        result[5] = (masks[0] & 32u) ? nanf("") : values[5][0];
        result[6] = __uint_as_float((unsigned)indices[0]);
        result[7] = __uint_as_float((unsigned)(indices[0] >> 32));
    }
}

// -------------------------------------------------------------------------
// Phase-5 full mutable-state health gate.
//
// Pointer/size/status metadata is represented as pairs of uint32 words in
// the model's sanctioned FP32 scratch pool.  blockIdx.y selects the
// descriptor and blockIdx.x a ``chunk`` of contiguous elements within it, so
// the launch stays one kernel while the largest descriptor is spread over
// many multiprocessors instead of stalling a single one.  The compact result
// is [uint64 ORed-status, uint64 packed-first-bad], where packed-first-bad is
// (field_id << 48) | flat_index.  atomicMin therefore gives deterministic
// field-first, then index-first attribution independent of block scheduling,
// and therefore independent of how the elements are split into chunks.
// -------------------------------------------------------------------------

#define VALIDATE_LOWER       1u
#define VALIDATE_UPPER       2u
#define VALIDATE_STRICT_LOW  4u
#define VALIDATE_ADD_AUX     8u
#define VALIDATE_AUX_LEVEL  16u
#define VALIDATE_INT32      32u
#define VALIDATE_INACTIVE_POSINF 64u

static __device__ __forceinline__
unsigned long long words_u64(const unsigned* words, int index)
{
    int offset = 2 * index;
    return ((unsigned long long)words[offset]
            | ((unsigned long long)words[offset + 1] << 32));
}

extern "C" __global__
void validate_full_state(
        const unsigned* __restrict__ field_pointer_words,
        const unsigned* __restrict__ auxiliary_pointer_words,
        const unsigned* __restrict__ field_size_words,
        const real* __restrict__ bounds,
        const unsigned* __restrict__ rule_flags,
        const unsigned* __restrict__ plane_sizes,
        const unsigned* __restrict__ status_bit_words,
        const unsigned* __restrict__ rim_geometry,
        unsigned long long* __restrict__ result,
        int nfields, unsigned long long chunk)
{
    int field = blockIdx.y;
    if (field >= nfields) return;
    unsigned long long values_address = words_u64(
        field_pointer_words, field);
    const real* values = reinterpret_cast<const real*>(values_address);
    const int* integer_values = reinterpret_cast<const int*>(values_address);
    const real* auxiliary = reinterpret_cast<const real*>(
        words_u64(auxiliary_pointer_words, field));
    unsigned long long size = words_u64(field_size_words, field);
    // Block-uniform: chunks past the end of a short descriptor leave before
    // the __syncthreads() below, so the reduction is never split.
    unsigned long long begin = (unsigned long long)blockIdx.x * chunk;
    if (begin >= size) return;
    unsigned long long end = begin + chunk < size ? begin + chunk : size;
    unsigned flags = rule_flags[field];
    real lower = bounds[2 * field];
    real upper = bounds[2 * field + 1];
    bool failed = false;
    unsigned long long first = 0xffffffffffffffffull;
    for (unsigned long long index = begin + threadIdx.x; index < end;
         index += blockDim.x) {
        real value = ((flags & VALIDATE_INT32)
                      ? (real)integer_values[index] : values[index]);
        if (flags & VALIDATE_ADD_AUX) {
            unsigned long long aux_index = index;
            if (flags & VALIDATE_AUX_LEVEL)
                aux_index = index / (unsigned long long)plane_sizes[field];
            if (flags & VALIDATE_INACTIVE_POSINF) {
                unsigned long long outgoing = index + plane_sizes[field];
                real activity = outgoing < size ? auxiliary[outgoing] : 0.0f;
                if (isinf(value) && value > 0.0f && activity == 0.0f)
                    value = 0.0f;
            } else {
                value += auxiliary[aux_index];
            }
        }
        bool bad = !isfinite(value);
        unsigned ny = rim_geometry[3 * field];
        unsigned nx = rim_geometry[3 * field + 1];
        unsigned width = rim_geometry[3 * field + 2];
        bool rim = false;
        if (width && nx && ny) {
            unsigned i = index % nx, j = (index / nx) % ny;
            rim = (i < width || nx - 1 - i < width
                   || j < width || ny - 1 - j < width);
        }
        if ((flags & VALIDATE_LOWER) && !rim)
            bad |= ((flags & VALIDATE_STRICT_LOW)
                    ? value <= lower : value < lower);
        if (flags & VALIDATE_UPPER)
            bad |= value > upper;
        if (bad) {
            failed = true;
            first = index < first ? index : first;
        }
    }
    __shared__ unsigned long long first_by_thread[HEALTH_THREADS];
    __shared__ unsigned failed_by_thread[HEALTH_THREADS];
    int lane = threadIdx.x;
    first_by_thread[lane] = first;
    failed_by_thread[lane] = failed ? 1u : 0u;
    __syncthreads();
    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            unsigned long long other = first_by_thread[lane + offset];
            first_by_thread[lane] = other < first_by_thread[lane]
                                    ? other : first_by_thread[lane];
            failed_by_thread[lane] |= failed_by_thread[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && failed_by_thread[0]) {
        atomicOr(result, words_u64(status_bit_words, field));
        unsigned long long packed = ((unsigned long long)field << 48)
                                    | first_by_thread[0];
        atomicMin(result + 1, packed);
    }
}

// -------------------------------------------------------------------------
// The same reductions, over ONE TILE'S INTERIOR, in DOMAIN coordinates.
//
// Why this exists: under [tiles] the domain lives in a host store and the
// resident DomainState the run loop hands to stability_report is never
// written again, so the loop's nan / w_max / CFL gates observe a corpse.  The
// reductions are max folds and an OR fold, so they are associative and can be
// taken per tile inside the sweep instead -- which reads the memory the
// forecast is actually in.
//
// Three things make this EQUAL to the whole-domain reduction rather than
// merely similar, and all three are parameters rather than assumptions:
//
//   * the window is the tile's INTERIOR (the set the scatter writes), never
//     the halo -- halo cells are a neighbour's copy this tile did not
//     integrate, and on a domain edge they are seam fill;
//   * every index compared or classified is a DOMAIN index, built from
//     (dj0, di0), because health_partial breaks argmax ties by LOWEST FLAT
//     INDEX and splits boundary from interior using the DOMAIN's (j, i);
//   * the u window is passed separately from the mass window, because u is
//     (nz, ny, nx+1) and exactly one tile owns the closing face.
//
// Partial records are laid out exactly as health_partial's, so health_final
// reduces a whole sweep's tiles in one launch with no special case.
// -------------------------------------------------------------------------

extern "C" __global__
void health_partial_tile(const real* __restrict__ u,
                         const real* __restrict__ w,
                         const real* __restrict__ thp,
                         const real* __restrict__ ph,
                         const real* __restrict__ phb,
                         real* __restrict__ partial,
                         int tny, int tnx,
                         int jm0, int jmn, int im0, int imn,
                         int ju0, int jun, int iu0, int iun,
                         int nz, int dj0, int di0,
                         int ny, int nx, int width,
                         int phb_full, int do_vertical, real gravity)
{
    real umax = 0.0f, wmax = 0.0f, thmax = 0.0f;
    real edge = 0.0f, interior = 0.0f, vertical_rate = 0.0f;
    unsigned long long windex = 0xffffffffffffffffull;
    unsigned mask = 0u;
    unsigned long long start = ((unsigned long long)blockIdx.x * blockDim.x
                                + threadIdx.x);
    unsigned long long stride = ((unsigned long long)gridDim.x * blockDim.x);

    const unsigned long long plane_m = (unsigned long long)tny * tnx;
    const unsigned long long plane_u = (unsigned long long)tny * (tnx + 1);
    const unsigned long long dplane   = (unsigned long long)ny * nx;

    // ---- u over the owned faces (no argmax: health_partial keeps none) ---
    unsigned long long nu = (unsigned long long)nz * jun * iun;
    for (unsigned long long ix = start; ix < nu; ix += stride) {
        unsigned long long rem = ix;
        int ii = (int)(rem % (unsigned long long)iun); rem /= (unsigned long long)iun;
        int jj = (int)(rem % (unsigned long long)jun); rem /= (unsigned long long)jun;
        int k  = (int)rem;
        real value = fabsf(u[(unsigned long long)k * plane_u
                             + (unsigned long long)(ju0 + jj) * (tnx + 1)
                             + (unsigned long long)(iu0 + ii)]);
        if (isnan(value)) mask |= 1u;
        else if (value > umax) umax = value;
    }

    // ---- w over the owned mass columns, all nz+1 faces -------------------
    unsigned long long nw = (unsigned long long)(nz + 1) * jmn * imn;
    for (unsigned long long ix = start; ix < nw; ix += stride) {
        unsigned long long rem = ix;
        int ii = (int)(rem % (unsigned long long)imn); rem /= (unsigned long long)imn;
        int jj = (int)(rem % (unsigned long long)jmn); rem /= (unsigned long long)jmn;
        int k  = (int)rem;
        real value = fabsf(w[(unsigned long long)k * plane_m
                             + (unsigned long long)(jm0 + jj) * tnx
                             + (unsigned long long)(im0 + ii)]);
        int dj = dj0 + jj, di = di0 + ii;
        unsigned long long dindex = (unsigned long long)k * dplane
                                    + (unsigned long long)dj * nx
                                    + (unsigned long long)di;
        bool boundary = false;
        if (width > 0) {
            boundary = (dj < width || dj >= ny - width
                        || di < width || di >= nx - width);
        }
        if (isnan(value)) {
            mask |= 2u;
            if (width > 0) mask |= boundary ? 8u : 16u;
        } else {
            update_max(value, dindex, wmax, windex);
            if (width > 0) {
                if (boundary) edge = value > edge ? value : edge;
                else interior = value > interior ? value : interior;
            }
        }
    }

    // ---- theta' over the owned mass cells --------------------------------
    unsigned long long nth = (unsigned long long)nz * jmn * imn;
    for (unsigned long long ix = start; ix < nth; ix += stride) {
        unsigned long long rem = ix;
        int ii = (int)(rem % (unsigned long long)imn); rem /= (unsigned long long)imn;
        int jj = (int)(rem % (unsigned long long)jmn); rem /= (unsigned long long)jmn;
        int k  = (int)rem;
        real value = fabsf(thp[(unsigned long long)k * plane_m
                               + (unsigned long long)(jm0 + jj) * tnx
                               + (unsigned long long)(im0 + ii)]);
        if (isnan(value)) mask |= 4u;
        else if (value > thmax) thmax = value;
    }

    // ---- co-located |w_upper| / dz_cell ----------------------------------
    if (do_vertical) {
        for (unsigned long long ix = start; ix < nth; ix += stride) {
            unsigned long long rem = ix;
            int ii = (int)(rem % (unsigned long long)imn); rem /= (unsigned long long)imn;
            int jj = (int)(rem % (unsigned long long)jmn); rem /= (unsigned long long)jmn;
            int k  = (int)rem;
            unsigned long long idx = (unsigned long long)k * plane_m
                                     + (unsigned long long)(jm0 + jj) * tnx
                                     + (unsigned long long)(im0 + ii);
            unsigned long long upper = idx + plane_m;
            real base_lower = phb_full ? phb[idx] : phb[k];
            real base_upper = phb_full ? phb[upper] : phb[k + 1];
            real dz = ((ph[upper] + base_upper) - (ph[idx] + base_lower))
                      / gravity;
            real speed = fabsf(w[upper]);
            if (!isfinite(dz) || dz <= 0.0f || !isfinite(speed)) {
                mask |= 32u;
            } else {
                real rate = speed / dz;
                vertical_rate = rate > vertical_rate ? rate : vertical_rate;
            }
        }
    }

    __shared__ real values[6][HEALTH_THREADS];
    __shared__ unsigned long long indices[HEALTH_THREADS];
    __shared__ unsigned masks[HEALTH_THREADS];
    int lane = threadIdx.x;
    values[0][lane] = umax;
    values[1][lane] = wmax;
    values[2][lane] = thmax;
    values[3][lane] = edge;
    values[4][lane] = interior;
    values[5][lane] = vertical_rate;
    indices[lane] = windex;
    masks[lane] = mask;
    __syncthreads();

    for (int offset = HEALTH_THREADS / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            values[0][lane] = values[0][lane + offset] > values[0][lane]
                              ? values[0][lane + offset] : values[0][lane];
            update_max(values[1][lane + offset], indices[lane + offset],
                       values[1][lane], indices[lane]);
            for (int field = 2; field < 6; ++field)
                values[field][lane] = values[field][lane + offset]
                                      > values[field][lane]
                                      ? values[field][lane + offset]
                                      : values[field][lane];
            masks[lane] |= masks[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        size_t out = (size_t)blockIdx.x * HEALTH_FIELDS;
        for (int field = 0; field < 6; ++field)
            partial[out + field] = values[field][0];
        partial[out + 6] = __uint_as_float((unsigned)indices[0]);
        partial[out + 7] = __uint_as_float((unsigned)(indices[0] >> 32));
        partial[out + 8] = __uint_as_float(masks[0]);
    }
}
