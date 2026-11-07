// WRF v4.6.1 module_surface_driver.F:3689-3697, public domain.
// PX diagnoses its own T2/Q2. No SFCDIAGS follows the land call.
extern "C" __global__ void pxlsm_post(
    int n, float dtbl, const float *psfc, const float *t2, const float *qfx,
    float *th2, float *chklowq, float *sfcevp, float *rainbl, float *pending) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    chklowq[c] = 1.0f;
    th2[c] = t2[c] * gfk_pow(1.e5f / psfc[c], 287.f / 1004.5f);
    sfcevp[c] = sfcevp[c] + qfx[c] * dtbl;
    rainbl[c] = 0.f;
    pending[c] = 0.f;
}

// module_surface_driver.F: get_local_ice_tsk; the surface-layer wrapper has
// already applied its SST floor and first-three-step bounds on this call.
extern "C" __global__ void pxlsm_ice_prepare(
    int n, int surface_called, const float* xice, float* sst,
    float* sea, float* tsk) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    float x=xice[c], t=tsk[c];
    if (surface_called) {
        if(x>=.02f && x<=1.f) sst[c]=sea[c];
        return;
    }
    if(x>=.02f && x<=1.f) {
        sst[c]=fmaxf(sst[c],271.4f);
        sea[c]=sst[c];
        float ice=fmaxf((t-(1.f-x)*sea[c])/x,221.4f);
        if(x<.2f && t<253.15f) ice=253.15f;
        if(x<.1f && t<263.15f) ice=263.15f;
        tsk[c]=ice;
    } else sea[c]=t;
}
extern "C" __global__ void pxlsm_ice_blend(
    int n, int surface_called, const float* xice, const float* sea, float* tsk, float* save,
    float* flhc,
    float* flqc,
    float* cpm,
    float* cqs2,
    float* chs2,
    float* chs,
    float* qsfc,
    float* qgh,
    float* hfx,
    float* qfx,
    float* lh,
    const float* flhc_sea, const float* flqc_sea, const float* cpm_sea, const float* cqs2_sea, const float* chs2_sea, const float* chs_sea, const float* qsfc_sea, const float* qgh_sea, const float* hfx_sea, const float* qfx_sea, const float* lh_sea) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    float x=xice[c];
    if(x>=.02f && x<=1.f) {
      if(surface_called) {
        flhc[c]=flhc[c]*x+(1.f-x)*flhc_sea[c];
        flqc[c]=flqc[c]*x+(1.f-x)*flqc_sea[c];
        cpm[c]=cpm[c]*x+(1.f-x)*cpm_sea[c];
        cqs2[c]=cqs2[c]*x+(1.f-x)*cqs2_sea[c];
        chs2[c]=chs2[c]*x+(1.f-x)*chs2_sea[c];
        chs[c]=chs[c]*x+(1.f-x)*chs_sea[c];
        qsfc[c]=qsfc[c]*x+(1.f-x)*qsfc_sea[c];
        qgh[c]=qgh[c]*x+(1.f-x)*qgh_sea[c];
        hfx[c]=hfx[c]*x+(1.f-x)*hfx_sea[c];
        qfx[c]=qfx[c]*x+(1.f-x)*qfx_sea[c];
        lh[c]=lh[c]*x+(1.f-x)*lh_sea[c];
      }
        save[c]=tsk[c];
        tsk[c]=tsk[c]*x+(1.f-x)*sea[c];
    }
}
