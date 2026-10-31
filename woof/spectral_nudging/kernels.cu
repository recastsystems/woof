// Original implementation of the spectral relaxation equations in:
// von Storch, Langenberg, Feser (2000), Eq. 2,
// DOI 10.1175/1520-0493(2000)128<3664:ASNTFD>2.0.CO;2.
// Miguez-Macho, Stenchikov, Robock (2004), Eqs. 1-2 and section 4,
// DOI 10.1029/2003JD004495.
// The regional convention uses physical T; WRF FDDA uses potential T.
// WRF clock and mask semantics are ported under WRF-NOTICE.txt.
#include <cuComplex.h>

__device__ bool due(const double *clock, int cadence) {
    long long step = llround(clock[0] / clock[1]) + 1;
    return step == 1 || step % cadence == 0;
}
__device__ bool clear_between_calls(const double *clock, int cadence) {
    // WRF clears at mod(itimestep-1,STEPFG)==0, then calculates at
    // step 1 or multiples of STEPFG. The first extra call is held until
    // the first regular call even when fgdtzero is enabled.
    return llround(clock[0]/clock[1]) % cadence == 0;
}
__device__ float time_weight(const double *clock, double end, double ramp) {
    if (end < 0.) return 1.f;
    double stop = end + fmax(0., ramp);
    if (ramp != 0.) return (float)fmin(1., fmax(0., (stop-clock[0])/fabs(ramp)));
    return clock[0] <= end ? 1.f : 0.f;
}

extern "C" __global__ void difference(
    const float *value, const float *a, const float *b,
    cuFloatComplex *out, const double *clock, unsigned int *error, int n,
    int wrf, int cadence, double end, double ramp, int grid_nudging)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    if (!due(clock, cadence) || time_weight(clock, end, ramp) == 0.f) {
        out[i] = make_cuFloatComplex(0.f, 0.f); return;
    }
    // Midpoint forcing. clock = elapsed, dt, interval_start, interval_end.
    bool valid = clock[1] > 0 && clock[0] >= clock[2]
        && (clock[0] + clock[1] <= clock[3] || (end >= 0. && clock[0] >= end));
    if (!valid) {
        atomicOr(error, 1u);
        out[i] = make_cuFloatComplex(0.f, 0.f);
        return;
    }
    float alpha = (float)((clock[0] + (wrf ? 0. : 0.5 * clock[1]) - clock[2])
                          / (clock[3] - clock[2]));
    // Grid FDDA holds the last analysis during a positive end ramp.
    // Spectral FDDA extrapolates the last two analyses instead.
    if (wrf && grid_nudging && end >= 0. && ramp > 0. && clock[0] >= end)
        alpha = (float)((end-clock[2])/(clock[3]-clock[2]));
    out[i] = make_cuFloatComplex(a[i] + alpha * (b[i] - a[i]) - value[i], 0.f);
}

extern "C" __global__ void truncate_modes(
    cuFloatComplex *spectrum, int n, int nx, int ny, int cutoff_x, int cutoff_y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int x = i % nx, y = (i / nx) % ny;
    int kx = min(x, nx - x), ky = min(y, ny - y);
    if (kx > cutoff_x || ky > cutoff_y)
        spectrum[i] = make_cuFloatComplex(0.f, 0.f);
}

extern "C" __global__ void relax(
    float *value, const cuFloatComplex *filtered,
    const float *height_agl, const float *pblh, const double *clock,
    int n, int nx, int ny, int boundary, float floor_m, float ramp_m, float tau,
    const float *vertical_weight, float *held, int wrf, int cadence, int fgdtzero,
    double end, double ramp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    if (!(clock[1] > 0 && clock[0] >= clock[2] &&
          (clock[0] + clock[1] <= clock[3] || (end >= 0. && clock[0] >= end)))) return;
    int x = i % nx, y = (i / nx) % ny;
    if (x < boundary || x >= nx - boundary || y < boundary || y >= ny - boundary) return;
    if (wrf) {
        if (due(clock, cadence)) {
            held[i] = vertical_weight[i] * time_weight(clock, end, ramp) * cuCrealf(filtered[i]) / tau;
        } else if (fgdtzero && clear_between_calls(clock,cadence)) {
            held[i] = 0.f;
        }
        if (time_weight(clock, end, ramp) > 0.f && held[i] != 0.f)
            value[i] += (float)clock[1] * held[i];
        return;
    }
    float z0 = fmaxf(floor_m, pblh[i % (nx * ny)]);
    float r = fminf(1.f, fmaxf(0.f, (height_agl[i] - z0) / ramp_m));
    // Post-filter masking preserves the PBL exactly. Spatially varying masks
    // mix Fourier modes; do not claim high-mode invariance in that case.
    float weight = r * r * (3.f - 2.f * r);
    if (cadence > 1 || end >= 0.) {
        if (due(clock,cadence)) held[i] = weight*time_weight(clock,end,ramp)*cuCrealf(filtered[i])/tau;
        else if (fgdtzero && clear_between_calls(clock,cadence)) held[i] = 0.f;
        if (time_weight(clock,end,ramp) > 0.f && held[i] != 0.f) value[i] += (float)clock[1]*held[i];
        return;
    }
    if (weight > 0.f) value[i] += (float)(clock[1] / tau) * weight * cuCrealf(filtered[i]);
}

extern "C" __global__ void tick(double *clock, unsigned int *error, unsigned long long *count)
{
    if (*error == 0) {clock[0] += clock[1]; *count += 1;}
}

// Device-side timing samples for receipts, no host synchronization per step.
extern "C" __global__ void stamp(unsigned long long *times, unsigned long long *sums, int slot) {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    times[slot] = value;
    if (slot & 1) sums[slot/2] += value-times[slot-1];
}
