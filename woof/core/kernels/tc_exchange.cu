// Independent neutral-table surface route. Green and Zhang (2013),
// doi:10.1175/MWR-D-12-00274.1. Tables carry each observational source.
// Businger-Dyer MO functions, with stability from the stock surface call.
// This module is never compiled or launched when the TC bundle is off.
__device__ float tc_pm(float z) {
    if (z >= 0) return -5.f*z;
    float x=powf(1.f-16.f*z,.25f);
    return 2.f*logf((1.f+x)/2.f)+logf((1.f+x*x)/2.f)-2.f*atanf(x)+1.5707963267948966f;
}
__device__ float tc_ph(float z) {
    if(z>=0) return -5.f*z;
    return 2.f*logf((1.f+sqrtf(1.f-16.f*z))/2.f);
}
__device__ float tc_lookup(const float* tab, float wind, int col) {
    float pos=fminf(1200.f,fmaxf(0.f,wind*10.f));
    int a=min(1199,(int)pos); float frac=pos-a;
    return tab[3*a+col]*(1.f-frac)+tab[3*(a+1)+col]*frac;
}
extern "C" __global__ void tc_exchange(
 const float* u,const float* v,const float* t,const float* q,const float* p,
 const float* dz,const float* ps,const float* ts,const float* land,
 const float* lake,const float* ice,const float* tab,const float* wave_z0,int wave_on,
 float* z0,float* ust,float* ustm,float* mol,float* hfx,float* qfx,
 const float* qs,float* zol,float* psim,float* psih,float* fm,float* fh,
 float* lh,float* u10,float* v10,float* th2,float* t2,float* q2,
 float* chs,float* chs2,float* cqs2,float* flhc,float* flqc,float* rmol,
 float* cd,float* cda,float* ck,float* cka,
 float* cdn,float* chn,float* cqn,float* zt_out,float* zq_out,
 float* mynn_ch,float* qstar,float* cpm,float* gz1oz0,int mynn,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n || land[i]<1.5f || lake[i]>.5f || ice[i]>.01f) return;
 float height=fmaxf(.5f*dz[i],2.01f), speed=hypotf(u[i],v[i]);
 // Freeze the old MO stability over this table solve. The next surface call
 // updates stability with the actual exchanged heat and momentum.
 float invL=rmol[i];
 float neutral=fmaxf(0.f,hypotf(u10[i],v10[i]));
 float dm=0,dh=0,dq=0,zm=0,zt=0,zq=0,cm=0,ch=0,cq=0;
 for(int it=0;it<12;++it) {
   cm=tc_lookup(tab,neutral,0); ch=tc_lookup(tab,neutral,1);cq=tc_lookup(tab,neutral,2);
   float lm=.4f/sqrtf(cm);
   zm=10.f*expf(-lm);zt=10.f*expf(-.16f/(lm*ch));zq=10.f*expf(-.16f/(lm*cq));
   // Wave roughness replaces only momentum roughness; scalar roughness
   // retains the selected curve. Sea state is lagged one surface call.
   if(wave_on && wave_z0[i]>0.f) {
     zm=wave_z0[i];lm=logf(10.f/zm);cm=.16f/(lm*lm);
     ch=.16f/(lm*logf(10.f/zt));cq=.16f/(lm*logf(10.f/zq));
   }
   dm=fmaxf(.5f,logf(height/zm)-tc_pm(height*invL)+tc_pm(zm*invL));
   dh=fmaxf(.5f,logf(height/zt)-tc_ph(height*invL)+tc_ph(zt*invL));
   dq=fmaxf(.5f,logf(height/zq)-tc_ph(height*invL)+tc_ph(zq*invL));
   neutral=.5f*neutral+.5f*speed*lm/dm;
 }
 float us=.4f*speed/dm;
 float dm10=fmaxf(.5f,logf(10.f/zm)-tc_pm(10.f*invL)+tc_pm(zm*invL));
 float dh10=fmaxf(.5f,logf(10.f/zt)-tc_ph(10.f*invL)+tc_ph(zt*invL));
 float dq10=fmaxf(.5f,logf(10.f/zq)-tc_ph(10.f*invL)+tc_ph(zq*invL));
 float dh2=fmaxf(.5f,logf(2.f/zt)-tc_ph(2.f*invL)+tc_ph(zt*invL));
 float dq2=fmaxf(.5f,logf(2.f/zq)-tc_ph(2.f*invL)+tc_ph(zq*invL));
 float theta=t[i]*powf(1.e5f/p[i],287.f/1004.5f);
 float theta0=ts[i]*powf(1.e5f/ps[i],287.f/1004.5f);
 float rho=ps[i]/(287.f*t[i]*(1.f+.608f*q[i]));
 float cp=1004.5f*(1.f+(mynn?.84f:.8f)*q[i]);
 float surface_q=mynn ? qs[i]/(1.f-qs[i]) : qs[i];
 z0[i]=zm;ust[i]=us;ustm[i]=us;mol[i]=.4f*(theta-theta0)/dh;
 cdn[i]=cm;chn[i]=ch;cqn[i]=cq;zt_out[i]=zt;zq_out[i]=zq;
 flhc[i]=rho*cp*.4f*us/dh;flqc[i]=rho*.4f*us/dq;
 hfx[i]=flhc[i]*(theta0-theta);qfx[i]=flqc[i]*(surface_q-q[i]);lh[i]=2.5e6f*qfx[i];
 u10[i]=u[i]*dm10/dm;v10[i]=v[i]*dm10/dm;
 th2[i]=theta0+(theta-theta0)*dh2/dh;
 t2[i]=th2[i]*powf(__fdiv_rn(ps[i], 1.e5f),287.f/1004.5f);
 q2[i]=surface_q+(q[i]-surface_q)*dq2/dq;
 chs[i]=.4f*us/dq;chs2[i]=.4f*us/dh2;cqs2[i]=.4f*us/dq2;
 fm[i]=dm;fh[i]=dh;psim[i]=tc_pm(height*invL);psih[i]=tc_ph(height*invL);
 zol[i]=height*invL;cd[i]=.16f/(dm10*dm10);cda[i]=.16f/(dm*dm);
 // Preserve stock CK semantics: moisture exchange, not an enthalpy-weighted Ck.
 ck[i]=.16f/(dm10*dq10);cka[i]=.16f/(dm*dq);
 if(mynn) {
   // Match MYNN's input contract: virtual temperature scale, specific
   // humidity scale in g/kg, heat CHS/CH, and moist-air heat capacity.
   // These are carrier conversions, not another coefficient formulation.
   float qa=q[i]/(1.f+q[i]);
   mol[i]=.4f*(theta*(1.f+.608f*qa)-theta0*(1.f+.608f*qs[i]))/dh;
   qstar[i]=.4f*(qa-qs[i])*1000.f/dq;
   chs[i]=.4f*us/dh;mynn_ch[i]=chs[i];cpm[i]=cp;
   gz1oz0[i]=logf((height+zm)/zm);
 }
}
