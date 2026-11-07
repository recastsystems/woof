// Read-only validation of PhysicsDriver's canonical surface diagnostics.
//
// The scheme kernels remain in their own translation units.  This compact
// status reduction changes only how their completed outputs are checked.

extern "C" __global__
void microphysics_validate_outputs(
        const real *rainnc, const real *rainncv, const real *sr,
        const real *snownc, const real *snowncv,
        const real *graupelnc, const real *graupelncv,
        const real *hailnc, const real *hailncv,
        unsigned int active, real sr_upper,
        unsigned int *status, long long count) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count) return;

    unsigned int invalid = 0;
    if ((active & (1u << 0)) && !isfinite(rainnc[index]))
        invalid |= 1u << 0;
    if ((active & (1u << 1)) && !isfinite(rainncv[index]))
        invalid |= 1u << 1;
    if ((active & (1u << 2)) && !isfinite(sr[index]))
        invalid |= 1u << 2;
    if ((active & (1u << 3)) && !isfinite(snownc[index]))
        invalid |= 1u << 3;
    if ((active & (1u << 4)) && !isfinite(snowncv[index]))
        invalid |= 1u << 4;
    if ((active & (1u << 5)) && !isfinite(graupelnc[index]))
        invalid |= 1u << 5;
    if ((active & (1u << 6)) && !isfinite(graupelncv[index]))
        invalid |= 1u << 6;
    if ((active & (1u << 7)) && !isfinite(hailnc[index]))
        invalid |= 1u << 7;
    if ((active & (1u << 8)) && !isfinite(hailncv[index]))
        invalid |= 1u << 8;
    // CAMMGMP exports an un-clipped ratio of signed sedimentation sums.
    // Its native finite SR can be negative after cancellation; the other
    // schemes retain their established [0, sr_upper] fraction contract.
    if (!(active & (1u << 31))) {
        if (sr[index] < 0.0f) invalid |= 1u << 16;
        if (sr[index] > sr_upper) invalid |= 1u << 17;
    }
    if (invalid != 0) atomicOr(status, invalid);
}

// ---------------------------------------------------------------------------
// The specified-zone ring guard around every scheme call, in ONE launch each
// way.  It lives beside the validation kernel because both run on every
// microphysics call: no extra translation unit is compiled for it, and like
// the validation kernel it holds no per-thread frame.
// ---------------------------------------------------------------------------
//
// gpuwm.core.microphysics snapshots the outermost spec_zone ring of each
// array a scheme call can touch before the call and restores it after
// (WRF's clipped microphysics tiles, solve_em.F:3618-3639).  Done as one
// CuPy slice copy per (array, ring edge), that was one small launch per
// array and edge each way per call; this kernel does the same copies from
// one descriptor table.  It moves 32-bit words and does no arithmetic, so every copied
// value, NaN payloads included, is bit for bit what the slice copy wrote.
//
// Descriptor row (MP_RING_ROW int64 words):
//   0 array base address, 1 buffer base address (0 = zero-fill the ring
//   section instead), 2 leading extent (levels, 1 for a 2-D field),
//   3 first row j0, 4 row count nj, 5 first column i0, 6 column count ni,
//   7 row length nx, 8 plane size ny*nx.
// The buffer holds the section C-contiguously, exactly as
// ``buf[...] = arr[slc]`` lays it out: level, then row, then column.
// direction 0 gathers array -> buffer, 1 scatters buffer -> array.
#define MP_RING_ROW 9

extern "C" __global__ void mp_ring_copy(
    const long long* __restrict__ table, int rows, int direction)
{
    const int r = blockIdx.y;
    if (r >= rows) return;
    const long long* d = table + (long long)r * MP_RING_ROW;
    unsigned int* arr = (unsigned int*)d[0];
    unsigned int* buf = (unsigned int*)d[1];
    const long long nlev = d[2], j0 = d[3], nj = d[4], i0 = d[5], ni = d[6];
    const long long nx = d[7], plane = d[8];
    const long long section = nj * ni;
    const long long count = nlev * section;
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         k < count; k += (long long)gridDim.x * blockDim.x) {
        const long long lev = k / section;
        const long long rem = k - lev * section;
        const long long jj = rem / ni;
        const long long ii = rem - jj * ni;
        const long long a = lev * plane + (j0 + jj) * nx + (i0 + ii);
        if (buf == nullptr) {
            arr[a] = 0u;
        } else if (direction == 0) {
            buf[k] = arr[a];
        } else {
            arr[a] = buf[k];
        }
    }
}
