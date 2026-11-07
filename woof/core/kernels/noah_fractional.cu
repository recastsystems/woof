// WRF 4.7.1 module_surface_driver.F:4677,4780. WRF public domain.
extern "C" __global__ void eta_ice_qsat(const float* tsk,const float* psfc,
                                        float* qs,int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=n)return;
    float pq0sea=__fmul_rn(379.90516f,.98f);
    float arg=__fdiv_rn(__fmul_rn(17.2693882f,__fsub_rn(tsk[c],273.16f)),
                         __fsub_rn(tsk[c],35.86f));
    qs[c]=__fmul_rn(__fdiv_rn(pq0sea,psfc[c]),gfk_exp(arg));
}
