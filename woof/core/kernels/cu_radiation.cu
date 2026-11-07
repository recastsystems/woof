// Public WRF v4.7.1 module_radiation_driver.F:1264-1282,1335-1345.
// Licence: WRF public domain, checked in cloud_aerosol_wrf471_oracle/sources.json.
// Only private radiation moisture is written. Native restores QC/QI after use.
#ifdef GPUWM_WRF_EXACT
#define FADD(a,b) __fadd_rn(a,b)
#define FSUB(a,b) __fsub_rn(a,b)
#define FMUL(a,b) __fmul_rn(a,b)
#else
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#endif
extern "C" __global__ void cu_radiation_merge(
    long long n, int mode, int fqc, int fqi,
    const float *qc_cu, const float *qi_cu,
    const float *deep, const float *shallow,
    float *qc, float *qi, float *fraction, float *cu_fraction) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    if(mode==1) {
        if(fqc) qc[i]=FADD(qc[i],qc_cu[i]);
        if(fqi) qi[i]=FADD(qi[i],qi_cu[i]);
    } else {
        float cu=FADD(deep[i],shallow[i]);
        cu_fraction[i]=cu;
        float cf=FMUL(FSUB(1.f,cu),fraction[i]);
        cf=FADD(cf,cu);
        fraction[i]=fminf(1.f,cf);
        qc[i]=FADD(qc[i],FMUL(qc_cu[i],cu));
        qi[i]=FADD(qi[i],FMUL(qi_cu[i],cu));
    }
}
