// Read-only validation, separate from the MRF column arithmetic.
extern "C" __global__
void mrf_validate_outputs(
        const real *du, const real *dv, const real *dtheta,
        const real *dqv, const real *dqc, const real *dqi,
        const real *pblh, const real *ust, const real *hol,
        unsigned int *status, long long count_3d, long long count_2d) {
    const long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int bad = 0;
    if (i < count_3d) {
        if (!isfinite(du[i])) bad |= 1u << 0;
        if (!isfinite(dv[i])) bad |= 1u << 1;
        if (!isfinite(dtheta[i])) bad |= 1u << 2;
        if (!isfinite(dqv[i])) bad |= 1u << 3;
        if (!isfinite(dqc[i])) bad |= 1u << 4;
        if (!isfinite(dqi[i])) bad |= 1u << 5;
    }
    if (i < count_2d) {
        if (!isfinite(pblh[i])) bad |= 1u << 6;
        if (!isfinite(ust[i])) bad |= 1u << 7;
        if (!isfinite(hol[i])) bad |= 1u << 8;
    }
    if (bad) atomicOr(status, bad);
}
