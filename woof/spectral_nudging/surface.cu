// WRF v4.6.1 phys/module_fdda_psufddagd.F:693-1320,1553-1595.
// Public-domain WRF notice: WRF-NOTICE.txt. See surface_sources.json.
// This is the SFDDAGD module operator, not an admitted forecast selector.
// Arrays use (level,y,x), with the native high x/y boundary included.
// The native caller owns wpbl, wzfac, ramping, cadence and halo exchange.
// Each column thread preserves the Fortran expression order. No atomics,
// host reductions or allocations occur in these kernels.

#ifdef SURFACE_FDDA_HOST_ORACLE
#include <cmath>
#include <algorithm>
using std::min;
using std::max;
#define __device__
#define __global__
struct SurfaceFddaIndex { int x; };
static SurfaceFddaIndex blockIdx, blockDim, threadIdx;
#define SFC_EXP expf
#define SFC_LOG logf
#define SFC_POW powf
#define SFC_DIV(a,b) ((a)/(b))
#else
// The setup adapter prepends the existing scalar libm header only in strict
// numerics. Default numerics uses CUDA's own fast column arithmetic.
#ifdef SURFACE_FDDA_STRICT
#define SFC_EXP gfk_exp
#define SFC_LOG gfk_log
#define SFC_POW gfk_pow
#define SFC_DIV(a,b) __fdiv_rn((a),(b))
#else
#define SFC_EXP expf
#define SFC_LOG logf
#define SFC_POW powf
#define SFC_DIV(a,b) ((a)/(b))
#endif
#endif

// SFDDAGD:1074-1100. Distances and radius are km; dx is m. Store the
// unsmoothed endpoints separately, because SMTHER reads an immutable copy.
extern "C" __global__ void sfc_density(
    const float *odis_old, const float *odis_new, float *raw,
    int nx, int ny, float dx, float rinblw) {
    int c=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny;
    if(c>=n) return;
    float rindx=rinblw*1000.f/dx;
    float m=-0.8f*2.f/rindx;
    for(int e=0;e<2;e++) {
        float d=e?odis_new[c]:odis_old[c];
        float w;
        if(d<0.5f*rinblw) w=1.f;
        else w=m*fminf(d*1000.f/dx,rindx)+1.8f;
        raw[e*n+c]=w;
    }
}

// SMTHER:1573-1591, one pass: XNU(1)=0.5. Native physical boundaries
// stay unsmoothed. Clamp only AFTER smoothing, as in SFDDAGD:1108-1119.
extern "C" __global__ void sfc_smooth(
    const float *raw, float *weight, int nx, int ny) {
    int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;
    if(c>=n) return;
    int i=c%nx,j=c/nx;
    for(int e=0;e<2;e++) {
        int q=e*n+c; float w=raw[q];
        // nx-1,ny-1 are ide,jde, excluded by the native loops.
        if(i>0 && i<nx-2 && j>0 && j<ny-2) {
            float avg=((raw[q+1]+raw[q-1])+raw[q+nx])+raw[q-nx];
            avg=avg*0.25f;
            w=raw[q]+0.5f*(avg-raw[q]);
        }
        weight[q]=fminf(1.f,fmaxf(0.f,w));
    }
}

// SFDDAGD:960-1037. The native U/V endpoint fields are already staggered.
// REGIME, ZNT, HT and the lowest Z are on mass points.
__device__ float sfc_wind_correction(
    const float *regime,const float *znt,const float *ht,const float *z,
    int a,int b) {
    float reg=0.5f*(regime[a]+regime[b]);
    float difz=0.5f*(((z[a]-ht[a])+z[b])-ht[b]);
    float correction=1.f;
    if(reg>3.5f && difz>10.f) {
        float roughness=0.5f*(znt[a]+znt[b]);
        if(roughness<=0.2f) correction=1.f+0.320f*SFC_POW(roughness,0.2f);
        else correction=1.169f+0.315f*roughness;
        // gfortran folds the two constant LOGs to these float32 words.
        const float wr14=6.684611797332763671875f;
        const float wr24=3.688879489898681640625f;
        float wr1z=SFC_LOG(difz/0.05f),wr2z=SFC_LOG(difz/1.f);
        float wndfac=0.5f*(wr1z/wr14+wr2z/wr24);
        correction=wndfac*correction;
    }
    return correction;
}

// A native saturation call uses R_d/R_v, NOT the rounded 0.622 literal.
__device__ float sfc_qsat(float temperature,float pressure) {
    float es=0.6112f*SFC_EXP(SFC_DIV(17.67f*(temperature-273.15f),(temperature-29.65f)))*10.f;
    return SFC_DIV((287.f/461.6f)*es,(SFC_DIV(pressure,100.f)-es));
}

// Endpoints: [U10,V10,TH2,Q2], each (ny,nx). wpbl is
// (4,nz,ny,nx), wzfac is (4,nz). The supplied weights are separate:
// (1-wpbl)*wzfac cannot be obtained by complementing their product.
// output is [RUNDGDTEN,RVNDGDTEN,RTHNDGDTEN,RQVNDGDTEN].
// At excluded native boundary points output retains its incoming bits.
// SDA_HFX/SDA_QFX are written only for FASDAS, matching the native routine.
extern "C" __global__ void sfc_tendencies(
    const float *u,const float *v,const float *theta,const float *temperature,
    const float *qv,const float *pressure,const float *z,const float *ht,
    const float *regime,const float *znt,const float *old_analysis,
    const float *new_analysis,const float *density,const float *wpbl,
    const float *wzfac,float *output,float *sda_hfx,float *sda_qfx,
    int nx,int ny,int nz,int fasdas,const float *time_factors,
    float guv,float gt,float gq) {
    int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny,volume=n*nz;
    if(c>=n) return;
    int i=c%nx,j=c/nx;
    if(i>=nx-1 || j>=ny-1) return;
    float coef=time_factors[0],tfac=time_factors[1];
    float one_minus_coef=1.f-coef;
    float blw=density[c]*one_minus_coef+density[n+c]*coef;
    float th_analysis=old_analysis[2*n+c]*one_minus_coef+new_analysis[2*n+c]*coef;
    float q_analysis=old_analysis[3*n+c]*one_minus_coef+new_analysis[3*n+c]*coef;
    float bu=0.f,bv=0.f,ua=0.f,va=0.f;
    if(i>0) {
        bu=(0.5f*(density[c-1]+density[c]))*one_minus_coef
            +(0.5f*(density[n+c-1]+density[n+c]))*coef;
        ua=old_analysis[c]*one_minus_coef+new_analysis[c]*coef;
        ua=ua*sfc_wind_correction(regime,znt,ht,z,c-1,c);
    }
    if(j>0) {
        bv=(0.5f*(density[c-nx]+density[c]))*one_minus_coef
            +(0.5f*(density[n+c-nx]+density[n+c]))*coef;
        va=old_analysis[n+c]*one_minus_coef+new_analysis[n+c]*coef;
        va=va*sfc_wind_correction(regime,znt,ht,z,c-nx,c);
    }
    for(int k=0;k<nz;k++) {
        int q=k*n+c;
        if(i>0) output[q]=guv*(1.f-wpbl[q])*wzfac[k]*tfac*bu*(ua-u[c]);
        if(j>0) output[volume+q]=guv*(1.f-wpbl[volume+q])*wzfac[nz+k]*tfac*bv*(va-v[c]);
        float qa=q_analysis,qs=sfc_qsat(temperature[q],pressure[q]);
        if(qa>qs) qa=qs;
        if(fasdas==1) {
            float stability=regime[c]<=1.1f ? 1.f/3.f:1.f;
            float w=k==0?0.f:1.f;
            float t=stability*gt*(1.f-w)*wzfac[2*nz+k]*tfac*blw*(th_analysis-theta[c]);
            float r=stability*gq*(1.f-w)*wzfac[3*nz+k]*tfac*blw*(qa-qv[q]);
            if(k==0) {
                float exf=SFC_POW(1.e5f/pressure[q],287.f/1004.f);
                sda_hfx[c]=t/exf;
                sda_qfx[c]=r;
            } else {t=0.f;r=0.f;}
            output[2*volume+q]=t;
            output[3*volume+q]=r;
        } else {
            output[2*volume+q]=gt*(1.f-wpbl[2*volume+q])*wzfac[2*nz+k]*tfac*blw*(th_analysis-theta[c]);
            output[3*volume+q]=gq*(1.f-wpbl[3*volume+q])*wzfac[3*nz+k]*tfac*blw*(qa-qv[q]);
        }
    }
}

// PXLSM:492-504. Endpoint RH uses the current model surface pressure at
// BOTH endpoints. Interpolating T/Q first and computing RH afterwards
// would change the native nonlinear calculation. SN is snow water
// equivalent in mm (Registry.EM_COMMON:2054-2055), without clipping.
extern "C" __global__ void sfc_px_targets(
    const float *t_old,const float *t_new,const float *q_old,const float *q_new,
    const float *sn_old,const float *sn_new,const float *psfc,
    float *t2obs,float *q2obs,float *rh2obs,float *snobs,
    int n,const float *time_factors) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    float core=time_factors[0],corb=1.f-core;
    float p=psfc[c]/1000.f;
    float e=0.6112f*SFC_EXP(17.67f*(t_old[c]-273.15f)/(t_old[c]-29.65f));
    float qs=(287.f/461.6f)*e/(p-e);
    float rh_old=q_old[c]/qs;
    e=0.6112f*SFC_EXP(17.67f*(t_new[c]-273.15f)/(t_new[c]-29.65f));
    qs=(287.f/461.6f)*e/(p-e);
    float rh_new=q_new[c]/qs;
    rh2obs[c]=corb*rh_old+core*rh_new;
    t2obs[c]=corb*t_old[c]+core*t_new[c];
    q2obs[c]=corb*q_old[c]+core*q_new[c];
    snobs[c]=corb*sn_old[c]+core*sn_new[c];
}

#ifdef SURFACE_FDDA_HOST_ORACLE
// Test-only entry points execute the SAME arithmetic as the device source.
// They are never loaded by the forecast and provide no CPU backend.
extern "C" void sfc_host_density(const float *a,const float *b,float *raw,float *w,
    int nx,int ny,float dx,float radius) {
    blockDim.x=1;threadIdx.x=0;
    for(int c=0;c<nx*ny;c++){blockIdx.x=c;sfc_density(a,b,raw,nx,ny,dx,radius);}
    for(int c=0;c<nx*ny;c++){blockIdx.x=c;sfc_smooth(raw,w,nx,ny);}
}
extern "C" void sfc_host_tendencies(
    const float *u,const float *v,const float *th,const float *t,const float *q,
    const float *p,const float *z,const float *h,const float *r,const float *znt,
    const float *a,const float *b,const float *density,const float *wpbl,
    const float *wzfac,float *out,float *sda_hfx,float *sda_qfx,
    int nx,int ny,int nz,int fasdas,float coef,float tfac,float guv,float gt,float gq) {
    blockDim.x=1;threadIdx.x=0;
    const float time_factors[2]={coef,tfac};
    for(int c=0;c<nx*ny;c++){
        blockIdx.x=c;sfc_tendencies(u,v,th,t,q,p,z,h,r,znt,a,b,density,
            wpbl,wzfac,out,sda_hfx,sda_qfx,nx,ny,nz,fasdas,time_factors,guv,gt,gq);
    }
}
extern "C" void sfc_host_px_targets(
    const float *t0,const float *t1,const float *q0,const float *q1,
    const float *s0,const float *s1,const float *p,float *t,float *q,
    float *rh,float *s,int n,float core) {
    blockDim.x=1;threadIdx.x=0;
    const float time_factors[2]={core,1.f};
    for(int c=0;c<n;c++) {
        blockIdx.x=c;sfc_px_targets(t0,t1,q0,q1,s0,s1,p,t,q,rh,s,n,time_factors);
    }
}
#endif
