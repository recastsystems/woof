// Four callable FLG stream arms, with the original closure controls.
extern "C" __global__ void flg_stream_driver(int ncol,int nv1,
    const int* bands,const float* albedo,const float* coszen,const float* f0,
    const float* emissivity,const float* bf,const float* bs,
    const float* moments,const float* scattering,const float* depth,
    const float* tables,const int* offsets,float* up,float* down,
    float* direct,float* diffuse,int* errors)
{
    int task=blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=ncol*4) return;
    int col=task/4,arm=task%4,nv=nv1-1; int controls[27]={};
    float unused_gases[11]={};
    controls[20]=arm!=1; controls[21]=arm==1;
    controls[22]=arm==2; controls[23]=arm!=2;
    controls[24]=arm==1; controls[25]=0; controls[26]=arm!=1;
    const float* w1=moments+col*4*nv;
    const float* w2=w1+nv; const float* w3=w2+nv; const float* w4=w3+nv;
    const float* w=scattering+col*nv; const float* t=depth+col*nv;
    errors[task]=0; float* fu=up+task*nv1; float* fd=down+task*nv1;
    if(arm==0)
        flgs_qfts(nv,nv1,bands[col*2],albedo[col],coszen[col],f0[col],w1,w2,w3,w4,w,t,fu,fd,
                   direct+col*nv1,diffuse+col*nv1,tables,offsets,unused_gases,controls,errors+task);
    else if(arm==1)
        flgs_qftsts(nv,nv1,bands[col*2],albedo[col],coszen[col],f0[col],w1,w2,w3,w4,w,t,fu,fd,
                     tables,offsets,unused_gases,controls,errors+task);
    else if(arm==2)
        flgs_qfti(nv,nv1,bands[col*2+1],emissivity[col],bf+col*nv1,bs[col],w1,w2,w3,w4,w,t,fu,fd,
                   tables,offsets,unused_gases,controls,errors+task);
    else
        flgs_qftisf(nv,nv1,bands[col*2+1],emissivity[col],bf+col*nv1,bs[col],w1,w2,w3,w4,w,t,fu,fd,
                     tables,offsets,unused_gases,controls,errors+task);
}

extern "C" __global__ void flg_zero_pivot_probe(int* error){
    float a[2]={},b[2]={},c[2]={},r[2]={},u[2]={},gam[2]={};
    float unused_gases[11]={}; int flags[27]={};
    *error=0;
    flgs_tridag(a,b,c,r,u,gam,2,nullptr,nullptr,unused_gases,flags,error);
}
