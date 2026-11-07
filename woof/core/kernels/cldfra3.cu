// WRF 4.7.1 module_radiation_driver.F:4003-4473 (unchanged since 4.6.1).
// WRF public-domain notice: gpuwm/data/wrf_radiation/LICENSE-WRF.txt.
// Strict log10 uses CORE-MATH; its MIT notice is in wrf471_log10f.cuh.
// One thread per bottom-up column. Explicit rounded operations retain order.
#ifdef GPUWM_WRF_EXACT
#define A __fadd_rn
#define S __fsub_rn
#define M __fmul_rn
#define D __fdiv_rn
#else
#define A(a,b) ((a)+(b))
#define S(a,b) ((a)-(b))
#define M(a,b) ((a)*(b))
#define D(a,b) ((a)/(b))
#define gfk_pow powf
#define gfk_log logf
#endif
__device__ float cf_sat(float p,float t,bool ice) {
 float x=fmaxf(-80.f,S(t,273.16f)),e;
 if(ice) e=A(.609868993e3f,M(x,A(.499320233e2f,M(x,A(.184672631e1f,M(x,A(.402737184e-1f,M(x,A(.565392987e-3f,M(x,A(.521693933e-5f,M(x,A(.307839583e-7f,M(x,A(.105785160e-9f,M(x,.161444444e-12f))))))))))))))));
 else e=A(.611583699e3f,M(x,A(.444606896e2f,M(x,A(.143177157e1f,M(x,A(.264224321e-1f,M(x,A(.299291081e-3f,M(x,A(.203154182e-5f,M(x,A(.702620698e-8f,M(x,A(.379534310e-11f,M(x,-.321582393e-13f))))))))))))))));
 e=fminf(e,M(p,.15f)); return D(M(.622f,e),ice?fmaxf(1.e-4f,S(p,e)):S(p,e));
}
// WOOF's float32 base-10 helper, shared arithmetic with r_log10.
__device__ float cf_log10(float x) {
#ifdef GPUWM_WRF_EXACT
 return wrf471_log10f(x);
#else
 return log10f(x);
#endif
}
__device__ void cf_adjust(float *cf,float *q,const float *qs,const float *qvs,const float *t,const float *dz,int k1,int k2,bool ice) {
 float total=0.f;for(int k=k1;k<=k2;k++)total=A(total,dz[k]);
 float maxq=fmaxf(0.f,S(qvs[k1],qvs[k2]));
 for(int k=k1;k<=k2;k++)maxq=fmaxf(1.e-6f,S(maxq,ice?A(q[k],qs[k]):q[k]));
 maxq=fminf(1.e-4f,maxq);float depth=0.f;
 for(int k=k1;k<=k2;k++) {depth=A(depth,k==k1?M(.5f,dz[k]):dz[k]);float amount=fmaxf(1.e-6f,M(D(M(maxq,depth),total),.5f));
 if(cf[k]>0.f&&cf[k]<1.f&&t[k]>=(ice?203.16f:258.16f))q[k]=A(q[k],M(M(cf[k],cf[k]),amount));}
}
extern "C" __global__ void cal_cldfra3_column(int nc,int nz,float *qv,float *qc,float *qi,const float *qs,const float *dz,const float *p,const float *t,const float *land,const float *grid,int modify,float maxrh,float *cf,float *work,unsigned *status) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
 int off=c*nz;qv+=off;qc+=off;qi+=off;qs+=off;dz+=off;p+=off;t+=off;cf+=off;
 float *qvs=work+off,*rh=work+nc*nz+off,*rho=work+2*nc*nz+off,*theta=work+3*nc*nz+off;
 for(int k=0;k<nz;k++) {
 float sw=cf_sat(p[k],t[k],false),si=cf_sat(p[k],t[k],true),tc=S(t[k],273.15f);
 qvs[k]=tc>=-12.f?sw:(tc<-35.f?si:S(sw,D(M(S(sw,si),S(-12.f,tc)),23.f)));
 if(modify) {if(qc[k]>1.e-8f){qv[k]=fmaxf(qv[k],sw);qvs[k]=sw;}if(qc[k]<=1.e-8f&&qi[k]>=1.e-9f){qv[k]=fmaxf(qv[k],M(si,1.005f));qvs[k]=si;}}
 rh[k]=fmaxf(.01f,D(qv[k],qvs[k]));rho[k]=D(p[k],M(287.f,t[k]));
 }
 for(int k=0;k<nz;k++) {
 float delz=fmaxf(100.f,dz[k]);float scale=__fsqrt_rn(D(1.f,A(50.f,M(M(M(grid[c],grid[c]),delz),.01f))));
 float rhl=A(.77f,fminf(.22f,scale)),rhocean=A(.85f,fminf(.14f,scale));
 float sum=A(qc[k],qi[k]);
 if(qc[k]>1.e-6f||qi[k]>=1.e-7f||(qs[k]>1.e-6f&&t[k]<273.f))cf[k]=1.f;
 else if(sum>1.e-10f&&sum<1.e-6f)cf[k]=fminf(.99f,M(.1f,A(11.f,cf_log10(sum))));
 else {
 float rh0=S(land[c],1.5f)>0.f?rhocean:rhl;float tc=fmaxf(-80.f,S(t[k],273.15f));if(tc<-12.f)rh0=rhl;
 if(tc>=25.f)cf[k]=0.f;
 else if(tc>=-12.f)cf[k]=fmaxf(0.f,S(1.f,__fsqrt_rn(D(S(1.001f,fminf(rh[k],1.f)),S(1.001f,rh0)))));
 else {float cap=(maxrh>1.12f||!modify)?1.45f:1.05f;float lim=(maxrh>1.12f||!modify)?1.46f:1.06f;rh0=A(rh0,D(M(S(cap,rh0),S(-12.f,tc)),73.f));cf[k]=fmaxf(0.f,S(1.f,__fsqrt_rn(D(S(lim,fminf(rh[k],cap)),S(lim,rh0)))));}
 if(cf[k]>0.f)cf[k]=fmaxf(.01f,fminf(cf[k],.99f));
 }
 if(cf[k]>0.f&&p[k]<7000.f)cf[k]=0.f;
 }
 int km=-1,kp=-1;
 for(int k=nz-1;k>=0;k--){theta[k]=M(t[k],gfk_pow(D(100000.f,p[k]),287.05f/1004.f));if(S(t[k],273.16f)>-12.f&&p[k]>10100.f)km=max(km,k);if(p[k]>14999.f&&kp==-1)kp=k;}
 km=max(km,0);if(km>nz-4){atomicOr(status,1u);return;}
 if(nz-1-kp<3)kp=nz-4;
 int k;
 for(k=kp-2;k>=0;k--){float depth=A(A(M(.5f,dz[k]),dz[k+1]),M(.5f,dz[k+2]));if(D(S(theta[k+2],theta[k]),depth)<10.f/1500.f||p[k]>70000.f)break;}
 int trop=max(2,min(k+2,nz-2));
 if(trop>kp){for(k=nz-4;k>=kp-2;k--){float depth=A(A(M(.5f,dz[k]),dz[k+1]),M(.5f,dz[k+2]));if(D(S(theta[k+2],theta[k]),depth)<10.f/1500.f&&p[k]>9500.f)break;}trop=max(kp-1,min(k+2,nz-2));}
 for(k=trop+1;k<nz;k++)if(cf[k]>0.f&&cf[k]<1.f)cf[k]=0.f;
 for(k=1;k<=km;k++)if(S(theta[k],theta[k-1])>M(.010e-3f,dz[k]))break;
 int bot=max(1,k-2);
 for(k=0;k<=bot;k++)if(cf[k]>0.f&&cf[k]<1.f)cf[k]=fmaxf(.01f,M(.5f,cf[k]));
 for(k=0;k<=trop;k++)if(cf[k]>0.f)bot=min(k,bot);
 int base=trop;
 for(k=trop;k>km+1;k--) {int top=-1;if(cf[k]>=.01f){top=k;for(int j=top-1;j>=km;j--)if(cf[j]<.01f||j==km){base=j+1;break;}}
 int depth=top-base+1;if(depth>=2){cf_adjust(cf,qi,qs,qvs,t,dz,base,top,true);k=base;}else if(depth==1){if(cf[base]>0.f&&cf[base]<1.f)qi[base]=A(qi[base],M(M(.05f,qvs[base]),cf[base]));k=base;}}
 base=km+3;
 for(k=km+2;k>bot;k--) {int top=-1;if(cf[k]>=.01f){top=k;for(int j=top-1;j>=bot;j--)if(cf[j]<.01f||j==bot){base=j+1;break;}}
 int depth=top-base+1;if(depth>=2){cf_adjust(cf,qc,qs,qvs,t,dz,base,top,false);k=base;}else if(depth==1){if(cf[base]>0.f&&cf[base]<1.f)qc[base]=A(qc[base],M(M(.05f,qvs[base]),cf[base]));k=base;}}
 float lwp=0.f,iwp=0.f;
 for(k=0;k<nz;k++)if(cf[k]>0.f&&cf[k]<1.f){lwp=A(lwp,M(M(qc[k],rho[k]),dz[k]));iwp=A(iwp,M(M(qi[k],rho[k]),dz[k]));}
 if(lwp>1.f){float factor=D(1.f,lwp);for(k=0;k<nz;k++)if(cf[k]>0.f&&cf[k]<1.f)qc[k]=M(qc[k],factor);}
 if(iwp>1.f){float factor=D(1.f,iwp);for(k=0;k<nz;k++)if(cf[k]>0.f&&cf[k]<1.f)qi[k]=M(qi[k],factor);}
 if(modify)for(k=0;k<nz;k++)if(cf[k]>.20f&&cf[k]<1.f)qv[k]=fmaxf(qv[k],qvs[k]);
}
#undef A
#undef S
#undef M
#undef D

// Radiation diagnostic: one maximum per column, with the fraction's word retained.
extern "C" __global__ void cldfra3_column_max(int nc,int nz,const float *cf,float *out) {
 int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc)return;
 float value=cf[c*nz];for(int k=1;k<nz;k++)value=fmaxf(value,cf[c*nz+k]);out[c]=value;
}
