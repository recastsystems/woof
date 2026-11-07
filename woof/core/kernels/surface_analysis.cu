// WRF v4.7.1 phys/module_surface_driver.F:1637-1792, public domain.
// Source concentration is loaded into XICE. XICEM remains the preceding step.
extern "C" __global__ void load_surface_field(
    const float* input, const float* water, const float* lake, float* output,
    int mask, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !isfinite(input[i])) return;
    if(mask!=3 && (water[i]<.5f || (mask==1 && lake[i]>.5f) || (mask==2 && lake[i]<.5f))) return;
    output[i]=input[i];
}
extern "C" __global__ void surface_update_column(
    const float* sst, const float* lake, float* xice, float* xicem,
    float* xice_save, float* xland, int* ivgtyp, int* isltyp, float* vegfra,
    float* tmn, float* albedo, float* albbck, float* emiss, float* embck,
    const float* albsi, float* tsk, float* tsk_save, float* snow, float* snowc,
    float* snowh, float* tslb, float* smois, float* sh2o,
    int lake_model, int fractional, int iswater, int isice,
    int albedo_opt, float ice_albedo, int nsoil, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float threshold=fractional ? .02f : .5f;
    if(lake_model && lake[i]==1.f && xice[i]>threshold) xice[i]=0.f;
    xice_save[i]=xicem[i];
    float ice=xice[i], old=xicem[i];
    if(fractional && ice!=old && old>threshold) {
        float a=albedo_opt==2 ? albsi[i] : albedo[i];
        albedo[i]=.08f+ice/old*(a-.08f);
        emiss[i]=.98f+ice/old*(emiss[i]-.98f);
        tsk[i]=tsk_save[i]*ice+(1.f-ice)*sst[i];
    }
    if(xland[i]>1.5f && ice>=threshold && old<threshold) {
        xicem[i]=ice; xland[i]=1.f; ivgtyp[i]=isice; isltyp[i]=16;
        vegfra[i]=0.f; tmn[i]=271.4f;
        float a=albedo_opt==2 ? albsi[i] : ice_albedo;
        albedo[i]=a*ice+.08f*(1.f-ice); albbck[i]=a;
        emiss[i]=.98f*ice+.98f*(1.f-ice); embck[i]=.98f;
        for(int k=0;k<nsoil;k++) {
            tslb[k*n+i]=tsk[i]; smois[k*n+i]=1.f; sh2o[k*n+i]=0.f;
        }
    }
    if(xland[i]>1.5f && (!lake_model || lake[i]!=1.f) && sst[i]<350.f && sst[i]>250.f) {
        tsk[i]=sst[i]; tslb[i]=sst[i];
    }
    if(xland[i]<1.5f && xicem[i]>=threshold && ice<threshold) {
        xicem[i]=ice; xland[i]=2.f; ivgtyp[i]=iswater; isltyp[i]=14;
        vegfra[i]=0.f; snow[i]=0.f; snowc[i]=0.f; snowh[i]=0.f;
        tmn[i]=sst[i]; albedo[i]=.08f; albbck[i]=.08f;
        emiss[i]=.98f; embck[i]=.98f;
        for(int k=0;k<nsoil;k++) {
            tslb[k*n+i]=sst[i]; smois[k*n+i]=1.f; sh2o[k*n+i]=1.f;
        }
    }
    xice_save[i]=xicem[i]; xicem[i]=ice; tsk_save[i]=tsk[i];
}

// Bit-preserving bounded form of WRF share/interp_fcn.F copy_fcnm/fcni.
// The host has already selected the first donor and the exact rectangle.
extern "C" __global__ void surface_feedback_pick(
    unsigned int* output, const unsigned int* input,
    int nx, int ny, int sx, int sy, int ri, int rj, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int x=i%nx, y=(i/nx)%ny, k=i/(nx*ny);
    output[i]=input[(k*sy+y*rj)*sx+x*ri];
}
