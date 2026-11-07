// Full rad_all column for WRF's stock FLG control path (naero=0).
// Inputs use WRF's top-to-surface ordering and native units.
extern "C" __global__ void flg_column_driver(
    int ncol,int nv1,const float* pressure,const float* temperature,
    const float* humidity,const float* ozone,const float* ice,const float* ice_size,
    const float* liquid,const float* liquid_radius,const float* rain,const float* graupel,
    const float* cloud_fraction,const float* inhomogeneity,
    const float* albedo,const float* emissivity,const float* coszen,
    const float* solar_constant,const float* skin_temperature,
    const float* tables,const int* offsets,const float* gas_values,const int* controls,
    float* fds,float* fus,float* dts,float* fdir,float* fuir,float* dtir,
    float* fd,float* fu,float* dt,float* direct,float* diffuse,int* errors,float* workspace)
{
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol) return;
    const int nv=nv1-1;
    int flags[29];
    for(int i=0;i<29;++i) flags[i]=controls[col*29+i];
    errors[col]=0;
    flgc_rad_all(workspace+(col/32)*(32*FLG_WORKSPACE_WORDS)+(col%32),nv,nv1,albedo+col*6,coszen[col],solar_constant[col],skin_temperature[col],emissivity+col*12,
        pressure+col*nv1,temperature+col*nv1,humidity+col*nv1,ozone+col*nv1,
        ice+col*nv,ice_size+col*nv,liquid+col*nv,liquid_radius+col*nv,
        rain+col*nv,graupel+col*nv,cloud_fraction+col*nv,inhomogeneity+col*nv,
        nullptr,nullptr,nullptr,
        fds+col*nv1,fus+col*nv1,dts+col*nv,fdir+col*nv1,fuir+col*nv1,dtir+col*nv,
        fd+col*nv1,fu+col*nv1,dt+col*nv,direct+col*nv1,diffuse+col*nv1,
        tables,offsets,gas_values+col*11,flags,errors+col);
}
