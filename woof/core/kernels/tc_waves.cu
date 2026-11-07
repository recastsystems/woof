// Fetch-limited deep-water JONSWAP state, equations 2.4.3--2.4.4 and
// spectral shape of Hasselmann et al. (1973), doi:10.2312/rrh.1973.12.
// Whitecap dissipation: Komen et al. (1984),
// doi:10.1175/1520-0485(1984)014<1271:OTEOAF>2.0.CO;2.
// Roughness: Drennan et al. (2003), doi:10.1029/2000JC000715.
// Original equations, no external source code. Spectral bandwidth and
// quadrature are explicit inputs because the slope depends on bandwidth.
__device__ float tc_spec(float f,float fp,float alpha,float gamma) {
 float sigma=f<fp?.07f:.09f;
 float peak=expf(-.5f*powf((f-fp)/(sigma*fp),2.f));
 return __fdiv_rn(alpha*9.81f*9.81f, powf(6.28318530718f,4.f))*powf(f,-5.f)
       *expf(-1.25f*powf(fp/f,4.f))*powf(gamma,peak);
}
extern "C" __global__ void tc_fetch_waves(
 const float* u10,const float* v10,const float* ust,const float* land,
 float* hs,float* phase,float* diss,float* slope,float* z0,
 float fetch,float flo,float fhi,int bins,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float speed=hypotf(u10[i],v10[i]);
 if(land[i]<1.5f || speed<1.f) {hs[i]=phase[i]=diss[i]=slope[i]=z0[i]=0;return;}
 float X=9.81f*fetch/(speed*speed);
 // Beyond the fetch-growth range use the PM fully developed limit.
 float alpha=fmaxf(.0081f,.076f*powf(X,-.22f));
 float fp=fmaxf(.13f,3.5f*powf(X,-1.f/3.f))*9.81f/speed;
 float gamma=fp>.13001f*9.81f/speed?3.3f:1.f;
 float dl=logf(fhi/flo)/bins;
 float m0=0,invomega=0,invrootk=0,mss=0,mk=0,mkk=0;
 for(int b=0;b<bins;++b){
  float f=flo*expf((b+.5f)*dl),df=2.f*f*sinhf(.5f*dl);
  float sig=6.28318530718f*f,k=__fdiv_rn(sig*sig, 9.81f);
  float en=tc_spec(f,fp,alpha,gamma)*df;
  m0+=en;invomega+=en/sig;invrootk+=en/sqrtf(k);
  mss+=k*k*en;mk+=k*en;mkk+=k*k*en;
 }
 float sigbar=m0/fmaxf(invomega,1.e-30f);
 float kbar=powf(m0/fmaxf(invrootk,1.e-30f),2.f);
 float steep=kbar*sqrtf(m0);
 float rate=4.10e-5f*powf(__fdiv_rn(steep, sqrtf(.00302f)),4.f)*sigbar;
 hs[i]=4.f*sqrtf(m0);phase[i]=9.81f/(6.28318530718f*fp);
 diss[i]=1025.f*9.81f*rate*(.5f*mk/fmaxf(kbar,1.e-30f)+.5f*mkk/fmaxf(kbar*kbar,1.e-30f));
 slope[i]=mss;
 float us=fmaxf(ust[i],.001f);
 z0[i]=fminf(.02f,3.35f*hs[i]*powf(us/phase[i],3.4f))+.11f*1.5e-5f/us;
}

// The same whitecap source used by the resident wave model, integrated
// with its own frequencies and finite-depth wavenumbers. No host copies.
extern "C" __global__ void tc_spectral_waves(
 const float* e,const float* f,const float* df,const float* k,
 const float* moments,const float* depth,const float* hs,
 const float* tp,const float* ust,float* phase,float* diss,float* slope,
 float* rough,int nf,int nd,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 phase[i]=diss[i]=slope[i]=rough[i]=0;
 if(depth[i]<=0 || hs[i]<=0 || tp[i]<=0) return;
 const float* m=moments+10*i;
 float kbar=fmaxf(m[2],1e-12f),steep=kbar*sqrtf(m[0]);
 float factor=4.10e-5f*powf(__fdiv_rn(steep, sqrtf(.00302f)),4.f)*m[1];
 float dd=0,ss=0;
 for(int a=0;a<nf;++a){
  float kk=k[i*nf+a],r=kk/kbar;
  for(int b=0;b<nd;++b){
   float en=e[(i*nf+a)*nd+b]*df[a]*6.28318530718f/nd;
   ss+=kk*kk*en;dd+=factor*(.5f+.5f*r)*r*en;
  }
 }
 float omega=6.28318530718f/tp[i],kp=__fdiv_rn(omega*omega, 9.81f);
 for(int j=0;j<12;++j){
  float kh=kp*depth[i],th=tanhf(kh);
  kp=fmaxf(1e-10f,kp-(9.81f*kp*th-omega*omega)/(9.81f*(th+kh*(1.f-th*th))));
 }
 float cp=omega/kp,us=fmaxf(ust[i],.001f);
 phase[i]=cp;diss[i]=1025.f*9.81f*dd;slope[i]=ss;
 rough[i]=fminf(.02f,3.35f*hs[i]*powf(us/cp,3.4f))+.11f*1.5e-5f/us;
}
