// WRF 4.7.1 phys/module_sf_idealscmsfclay.F:121-123,239-252.
// WRF public-domain source; retain the native REAL32 operation order.
extern "C" __global__ void ideal_scm_advance(
    float* heat, float* latent, float* skin,
    const float* heat_rate, const float* latent_rate, const float* skin_rate,
    float dt) {
    if (blockIdx.x || threadIdx.x) return;
    heat[0] = __fadd_rn(heat[0], __fmul_rn(dt, heat_rate[0]));
    latent[0] = __fadd_rn(latent[0], __fmul_rn(dt, latent_rate[0]));
    skin[0] = __fadd_rn(skin[0], __fmul_rn(dt, skin_rate[0]));
}

extern "C" __global__ void ideal_scm_surface(
    int n, const float* th, const float* pi, const float* rho,
    const float* mavail, const float* heat, const float* latent,
    const float* skin, float cp, float xlv, float* hfx, float* lh,
    float* qfx, float* tsk, float* flhc, float* exch, float* flqc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hfx[i] = heat[0];
    lh[i] = latent[0];
    qfx[i] = __fdiv_rn(lh[i], xlv);
    tsk[i] = skin[0];
    flhc[i] = __fdiv_rn(hfx[i], __fsub_rn(tsk[i], __fmul_rn(th[i], pi[i])));
    exch[i] = __fdiv_rn(flhc[i], __fmul_rn(rho[i], cp));
    flqc[i] = __fmul_rn(exch[i], mavail[i]);
    // Native U10/V10/TH2/T2/Q2 are INTENT(OUT) but never assigned.
    // All other INOUT fields are untouched by the native routine.
}
