// GFDL longwave table interpolation helpers from WRF 4.6.1.
// This translation unit is staged; it does not implement LWR88 or FST88.
// WOOF's own logarithm runs with separate operations and round-to-nearest.
#define GD_A(a,b) __fadd_rn((a),(b))
#define GD_S(a,b) __fsub_rn((a),(b))
#define GD_M(a,b) __fmul_rn((a),(b))
#define GD_D(a,b) __fdiv_rn((a),(b))

__device__ float gd_log10(float x) {
    return __double2float_rn(__dmul_rn(plm_log((double)x),0x1.bcb7b1526e50ep-2));
}
__device__ float gd_interp(float phi,float fx,float dt,
        const float *t1,const float *t2,const float *t4) {
    float tmp3=GD_A(gd_log10(phi),16.0f);
    float fyo=truncf(GD_M(tmp3,10.0f));
    float du=GD_S(tmp3,GD_M(0.1f,fyo));
    int iv=(int)GD_A(GD_M(28.0f,fyo),fx)-1;
    return GD_A(GD_A(t1[iv],GD_M(du,t2[iv])),GD_M(dt,t4[iv]));
}

extern "C" __global__ void gd_e3v88(const float *tv,const float *av,
        const float *em3,float *out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    float fxo=truncf(GD_M(tv[i],0.1f));
    float tmp3=GD_A(gd_log10(av[i]),16.0f);
    float dt=GD_S(tv[i],GD_M(10.0f,fxo));
    float fyo=truncf(GD_M(tmp3,10.0f));
    float du=GD_S(tmp3,GD_M(0.1f,fyo));
    int it=(int)GD_A(fxo,GD_M(fyo,28.0f));
    float w1=GD_S(10.0f,dt),w2=GD_S(0.1f,du);
    float a=GD_M(GD_M(w1,w2),em3[it-10]);
    float b=GD_M(GD_M(w2,dt),em3[it-9]);
    float c=GD_M(GD_M(w1,du),em3[it+18]);
    float d=GD_M(GD_M(dt,du),em3[it+19]);
    out[i]=GD_A(GD_A(GD_A(a,b),c),d);
}

extern "C" __global__ void gd_e2spec(const float *phi,const float *fx,
        const float *dt,const float *t1,const float *t2,const float *t4,
        float *out,int ncol,int nlay) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncol)return;
    int nv=nlay+1;
    for(int k=0;k<2;k++) {
        int ix=c*nv+k;
        out[ix]=gd_interp(phi[ix],fx[ix],dt[ix],t1,t2,t4);
    }
}

extern "C" __global__ void gd_e290(const float *phi,const float *fx,
        const float *dt,const float *t1,const float *t2,const float *t4,
        float *em,float *emb,int ncol,int nlay,int klen) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncol)return;
    int nv=nlay+1,off=c*nv;
    for(int k=klen-1;k<nv;k++)
        em[off+k]=gd_interp(phi[off+k],fx[off+k],dt[off+k],t1,t2,t4);
    em[off+nlay-1]=GD_M(0.5f,GD_A(em[off+nlay-1],em[off+nlay]));
    for(int k=klen-1;k<nlay;k++)
        emb[off+k]=gd_interp(phi[off+k],fx[off+klen-2],dt[off+klen-2],t1,t2,t4);
}

__device__ float gd_e1_two(const float *tab,int iv,float dt) {
    float w1=GD_S(10.0f,dt);
    return GD_A(GD_M(GD_M(w1,0.1f),tab[iv]),GD_M(GD_M(0.1f,dt),tab[iv+1]));
}
__device__ float gd_e1_four(const float *tab,int iv,float dt,float du) {
    float w1=GD_S(10.0f,dt),w2=GD_S(0.1f,du);
    float a=GD_M(GD_M(w1,w2),tab[iv]);
    float b=GD_M(GD_M(w2,dt),tab[iv+1]);
    float c=GD_M(GD_M(w1,du),tab[iv+28]);
    float d=GD_M(GD_M(dt,du),tab[iv+29]);
    return GD_A(GD_A(GD_A(a,b),c),d);
}
extern "C" __global__ void gd_e1e290(const float *phi,const float *fx1,
        const float *dt1,const float *fx2,const float *dt2,
        const float *t1,const float *t2,const float *t4,const float *em1,
        const float *em1w,float *em,float *g1,float *g2,float *g3,float *g4,
        float *g5,int ncol,int nlay) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncol)return;
    int nv=nlay+1,off=c*nv,off2=c*nlay;
    for(int k=0;k<nv;k++)em[off+k]=gd_interp(phi[off+k],fx2[off+k],dt2[off+k],t1,t2,t4);
    em[off+nlay-1]=GD_M(0.5f,GD_A(em[off+nlay-1],em[off+nlay]));
    int base=(int)fx1[off]-1;
    g1[off]=gd_e1_two(em1,base,dt1[off]);
    g3[off]=g1[off];
    g4[off]=gd_e1_two(em1w,base,dt1[off]);
    for(int k=0;k<nlay;k++) {
        float tmp3=GD_A(gd_log10(phi[off+k]),16.0f);
        float fyo=truncf(GD_M(tmp3,10.0f));
        float du=GD_S(tmp3,GD_M(0.1f,fyo));
        fyo=GD_M(28.0f,fyo);
        int ix1=(int)GD_A(fyo,fx1[off+k+1])-1;
        int ix2=(int)GD_A(fyo,fx1[off+k])-1;
        int ix3=(int)GD_A(fyo,fx1[off])-1;
        g1[off+k+1]=gd_e1_four(em1,ix1,dt1[off+k+1],du);
        g2[off2+k]=gd_e1_four(em1,ix2,dt1[off+k],du);
        g3[off+k+1]=gd_e1_four(em1,ix3,dt1[off],du);
        g4[off+k+1]=gd_e1_four(em1w,ix1,dt1[off+k+1],du);
        g5[off2+k]=gd_e1_four(em1w,ix2,dt1[off+k],du);
    }
}
