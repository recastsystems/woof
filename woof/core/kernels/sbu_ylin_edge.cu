// RIMI entry closure from WRF 4.6.1 module_mp_sbu_ylin.F:500-520.
// One thread owns a column. Source mass fields are physical mixing ratios.
// The first maximum is propagated downward before interpolation/restriction.
static __device__ __forceinline__ float sbu_edge_ri(float alt, float qc, float qs) {
    const float rho = __fdiv_rn(1.0f, alt);
    qc = qc > 0.0f ? qc : 0.0f;
    qs = qs > 0.0f ? qs : 0.0f;
    if (rho * qc <= 1e-5f || rho * qs <= 1e-5f) return 0.0f;
    return __fdiv_rn(1.0f, 1.0f + __fdiv_rn(6e-5f,
        (gfk_pow(rho, 1.170f) * qc) * gfk_pow(qs, 0.170f)));
}
// Reuse the established P3 exit policy's 50..900 kg/m3 bounds and split.
// Riming must use the target snow mass after that split, not an absent QS.
static __device__ __forceinline__ float sbu_edge_p3_snow(float ice, float rime, float volume) {
    ice = fmaxf(ice, 0.0f);
    rime = fmaxf(rime, 0.0f);
    volume = fmaxf(volume, 0.0f);
    float density = 0.0f;
    if (volume >= 1.0e-15f) {
        density = __fdiv_rn(rime, volume);
        if (density < 50.0f) { density = 50.0f; volume = __fdiv_rn(rime, density); }
        else if (density > 900.0f) { density = 900.0f; volume = __fdiv_rn(rime, density); }
    } else { rime = 0.0f; volume = 0.0f; }
    if (rime > ice && density > 0.0f) { rime = ice; volume = __fdiv_rn(rime, density); }
    if (rime < 1.0e-14f) { rime = 0.0f; volume = 0.0f; }
    float graupel = 0.0f;
    if (rime > 0.0f) {
        graupel = __fmul_rn(__fsub_rn(rime, __fmul_rn(100.0f, volume)), __fdiv_rn(400.0f, 300.0f));
        graupel = fminf(fmaxf(graupel, 0.0f), rime);
    }
    return __fsub_rn(rime, graupel);
}
extern "C" __global__ void sbu_ylin_edge_rimi(
    const float* alt, const float* qc, const float* qs,
    const float* qi, const float* qir, const float* qib,
    const float* mub, const float* mup, const float* c1, const float* c2,
    float* out, int snow_present, int source_mp, int coupled, int nz, int ncol) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float maximum = 0.0f;
    int first_maximum = 0;
    for (int k = 0; k < nz; ++k) {
        const int i = k * ncol + col;
        const float snow = source_mp == 50 ? sbu_edge_p3_snow(qi[i], qir[i], qib[i]) :
            (snow_present ? qs[i] : 0.0f);
        const float ri = sbu_edge_ri(alt[i], qc[i], snow);
        if (ri > maximum) { maximum = ri; first_maximum = k; }
    }
    for (int k = 0; k < nz; ++k) {
        const int i = k * ncol + col;
        const float snow = source_mp == 50 ? sbu_edge_p3_snow(qi[i], qir[i], qib[i]) :
            (snow_present ? qs[i] : 0.0f);
        float ri = (k <= first_maximum) ? maximum : sbu_edge_ri(alt[i], qc[i], snow);
        ri = fmaxf(0.0f, fminf(ri, 1.0f));
        if (coupled) {
            const float base = __fadd_rn(__fmul_rn(c1[k], mub[col]), c2[k]);
            const float mass = __fadd_rn(base, __fmul_rn(c1[k], mup[col]));
            ri = __fmul_rn(mass, ri);
        }
        out[i] = ri;
    }
}
