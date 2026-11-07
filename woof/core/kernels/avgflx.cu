// WRF v4.6.1 module_avgflx_em.F::upd_avgflx, single precision.
extern "C" __global__ void update_avgflx(
    float* mean, const float* flux, const int* clock, long long size,
    int nx, int ny, int axis)
{
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= size) return;
    const int sx = nx + (axis == 0), sy = ny + (axis == 1);
    const int i = q % sx, j = (q / sx) % sy;
    const int c = min(j, ny - 1) * nx + min(i, nx - 1);
    const bool reset = clock[nx * ny + c] != 0;
    const float count = __int2float_rn(reset ? 0 : clock[c]);
    const float old = reset ? 0.f : mean[q];
    // Explicit RN operations preserve WRF's recurrence even in fast mode.
    mean[q] = __fdiv_rn(__fadd_rn(__fmul_rn(count, old), flux[q]),
                       __fadd_rn(count, 1.f));
}

extern "C" __global__ void advance_avgflx_count(int* clock, int columns)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= columns) return;
    clock[c] = (clock[columns + c] ? 0 : clock[c]) + 1;
    clock[columns + c] = 0;
}
