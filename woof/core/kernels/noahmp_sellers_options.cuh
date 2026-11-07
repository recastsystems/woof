// WRF v4.6.1 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422,
// module_sf_noahmplsm.F:2175,2188-2191. WRF public-domain distribution.
// Dry and urban caps remain in ENERGY after this leaf, as in native WRF.
extern "C" __global__ void noahmp_sellers_resistance(
    const float* water, const float* smcmax, const float* fsno,
    float* resistance, int n)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
#ifdef GPUWM_WRF_EXACT
    float bevap=__fdiv_rn(water[c],smcmax[c]);
    bevap=0.0f>bevap?0.0f:bevap;
    float exponent=__fsub_rn(8.25f,__fmul_rn(NMP_SELLERS_COEFF,bevap));
    resistance[c]=__fadd_rn(__fmul_rn(fsno[c],1.0f),
        __fmul_rn(__fsub_rn(1.0f,fsno[c]),r_exp(exponent)));
#else
    float bevap=water[c]/smcmax[c];
    bevap=0.0f>bevap?0.0f:bevap;
    resistance[c]=fsno[c]+(1.0f-fsno[c])*expf(8.25f-NMP_SELLERS_COEFF*bevap);
#endif
}
