// WRF v4.7.1 module_diag_afwa.F and module_diag_functions.F, public
// domain. Immutable source hashes and notice: tools/diag_wrf471_oracle/.
// Each CUDA thread owns a native column; no host reduction or readback.
#ifdef GPUWM_WRF_EXACT
#define af_exp gfk_exp
#define af_pow gfk_pow
// This native capture uses glibc 2.43 LOG10(default REAL). The older shared
// log10_flt32 helper reproduces glibc 2.39's different rounding. Evaluate
// in binary64 and round once here; this matches every real captured word.
__device__ float af_log10(float x) { return (float)log10((double)x); }
#else
#define af_exp expf
#define af_pow powf
#define af_log10 log10f
#endif
__device__ float af_rh(float p,float t,float qv) {
    float q=qv/(1.0f+qv);
    float qs=379.90516f/p*af_exp(17.2693882f*(t-273.16f)/(t-35.86f));
    float rh=100.0f*q/qs;
    if(rh>100.0f)rh=100.0f;else if(rh<1.0f)rh=1.0f;
    return rh;
}
__device__ float af_mslp(float h,float p,float z,float q,float t) {
    const float gi=1.0f/9.81f, taucr=287.0f*gi*290.66f;
    const float correction=0.005f*9.81f/287.0f;
    float tv=t*(1.0f+0.608f*q);
    float ts=(tv+(z-h)*0.0065f)*287.0f*gi;
    float tl=(tv+(z-0.0f)*0.0065f)*287.0f*gi;
    if(tl>taucr && ts<=taucr)tl=taucr;
    else if(tl>taucr && ts>taucr)tl=taucr-correction*((ts-taucr)*(ts-taucr));
    float avg=0.5f*(tl+ts);
    return p*af_exp(h/avg);
}
__device__ float af_heat(float t,float rh) {
    if(!(t>294.26111f))return t;
    float tf=(t-273.15f)*(9.0f/5.0f)+32.0f, tf2=tf*tf, rh2=rh*rh;
    float hi=-42.379f+2.04901523f*tf+10.14333127f*rh
        -0.22475541f*tf*rh-6.83783E-03f*tf2-5.481717E-02f*rh2
        +1.22874E-03f*tf2*rh+8.5282E-04f*tf*rh2-1.99E-06f*tf2*rh2;
    return (hi-32.0f)*(5.0f/9.0f)+273.15f;
}
__device__ float af_chill(float t,float wind) {
    float mph=wind*2.23693629f, tf=(t-273.15f)*(9.0f/5.0f)+32.0f;
    float wc=35.74f+0.6215f*tf-35.75f*af_pow(mph,0.16f)
        +0.4275f*tf*af_pow(mph,0.16f);
    return (wc-32.0f)*(5.0f/9.0f)+273.15f;
}
__device__ float af_fits(float p,float t,float rh) {
    float tc=t-273.15f;
    float p1=11.344f-0.0303998f*t, p2=3.49149f-1302.8844f/t;
    float c1=23.832241f-5.02808f*af_log10(t);
    float es=af_pow(10.0f,c1-1.3816E-7f*af_pow(10.0f,p1)
        +8.1328E-3f*af_pow(10.0f,p2)-2949.076f/t);
    float svapr=es*100.0f, vapr=svapr*(rh/100.0f);
    float expon=(7.5f*tc)/(237.7f+tc);
    float de=6.112f*af_pow(10.0f,expon)*(rh/100.0f);
    float logs=af_log10(de/6.112f);
    float td=(237.7f*logs)/(7.5f-logs);
    float gamma=0.00066f*(p/1000.0f);
    float delta=4098.0f*(vapr/1000.0f)/((tc+237.3f)*(tc+237.3f));
    float wb=((gamma*tc+delta*td)/(gamma+delta))+273.15f;
    float fits=0.8281f*(wb-273.15f)+0.3549f*(t-273.15f)+5.08f;
    return fits+273.15f;
}
extern "C" __global__ void afwa_prepare_microphysics(
        const float* alt,const float* ph,const float* phb,
        float* rho,float* dz,float* z1,float* z,int profile,int nz,int n,int phb3) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float lower=(ph[c]+phb[phb3?c:0])/G;
    for(int k=0;k<nz;++k) {
        size_t x=(size_t)k*n+c;
        float upper=(ph[x+n]+phb[phb3?x+n:k+1])/G;
        rho[x]=1.0f/alt[x];dz[x]=upper-lower;
        if(k==0)z1[c]=0.5f*(lower+upper);
        if(profile)z[x]=0.5f*(lower+upper);
        lower=upper;
    }
}
extern "C" __global__ void afwa_base(
        const float* qv,const float* rho,const float* dz,const float* t1,const float* z1,
        const float* ht,const float* psfc,const float* raincv,const float* rainncv,
        const float* rainc,const float* rainnc,
        float* precip,float* total,float* pwat,float* mslp,int nz,int n,int water,int output) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    precip[c]=raincv[c]+rainncv[c];total[c]=rainc[c]+rainnc[c];
    if(water) {
        float sum=0.0f;
        for(int k=0;k<nz;++k){size_t x=(size_t)k*n+c;sum=sum+qv[x]*dz[x]*rho[x];}
        pwat[c]=sum;
    }
    if(output)mslp[c]=af_mslp(ht[c],psfc[c],z1[c],qv[c],t1[c]);
}
extern "C" __global__ void afwa_thermal(
        const float* t2,const float* p,const float* q2,const float* u10,const float* v10,
        float* heat,float* chill,float* fits,int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float rh=af_rh(p[c],t2[c],q2[c]);
    float wind=sqrtf(u10[c]*u10[c]+v10[c]*v10[c]);
    heat[c]=af_heat(t2[c],rh);chill[c]=af_chill(t2[c],wind);fits[c]=af_fits(p[c],t2[c],rh);
}

extern "C" __global__ void afwa_ptype(
 const float* t,const float* p,const float* q,const float* z,const float* dz,
 const float* ht,const float* t2,const float* sw,const float* precip,
 float* rain,float* snow,float* ice,float* frz,float* snowfall,
 float ccn,float total_melt,int nz,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n||!(precip[c]>0.0f))return;
 float mod=t2[c]+sw[c]/100.0f,maxtmp=mod,melt=0.0f;
 int type=0,top=nz-1;bool saturation=false;
 // Native cloud_top_tmp has no assignment if this cold precipitating
 // column never reaches RH>80%. Preserve that undefined result explicitly.
 float cloudtop=nanf("");
 if(mod>275.15f)type=1;
 else {
  for(int k=nz-1;k>=0;--k){size_t x=(size_t)k*n+c;
   if(z[x]-ht[c]>0.0f){
    maxtmp=fmaxf(t[x],maxtmp);float rh=af_rh(p[x],t[x],q[x]);
    if(rh>80.0f&&!saturation){cloudtop=t[x];top=k;saturation=true;type=2;}
    if(rh<=70.0f&&saturation)saturation=false;
   }
  }
  if(isnan(cloudtop)){rain[c]=snow[c]=ice[c]=frz[c]=snowfall[c]=cloudtop;return;}
  if(cloudtop<=ccn&&maxtmp<=273.15f)type=2;
  for(int k=top;k>=0;--k){size_t x=(size_t)k*n+c;
   if(z[x]-ht[c]>0.0f){
    if(cloudtop==t[x]&&cloudtop>ccn)type=1;
    if((type==2||type==3)&&t[x]>273.15f){
     melt=melt+9.8f*(((t[x]-273.15f)/273.15f)*dz[x]);
     if(melt>total_melt){type=1;melt=0.0f;}
    }
    if(t[x]<=273.15f&&melt>total_melt/4.0f&&(type==2||type==3)){type=3;melt=0.0f;}
    if(type==1&&t[x]<=ccn)type=3;
   }
  }
 }
 if(type==3)ice[c]=ice[c]+precip[c];
 if(type==2){snow[c]=snow[c]+precip[c];snowfall[c]=snowfall[c]+5.0f*precip[c]*af_pow(5.0f-mod+273.15f,0.4f);}
 if(type==1){if(mod>273.15f)rain[c]=rain[c]+precip[c];else frz[c]=frz[c]+precip[c];}
}

extern "C" __global__ void afwa_dbz(const float* rho,const float* t,
 const float* qr,const float* qs,const float* qg,float* ze,int count) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=count)return;
 const float pi=3.1415926536f;
 const float factor_r=0x1.f0bade0000000p+48f;
 const float factor_s=0x1.f490060000000p+45f;
 const float factor_g=0x1.61f3800000000p+46f;
 float fs=factor_s,fg=factor_g;
 if(t[x]>273.15f){fs=factor_s/0.224f;fg=factor_g/0.224f;}
 float tc=fminf(-0.001f,t[x]-273.15f);
 float son=fminf(2.0e8f,2.0e6f*af_exp(-0.12f*tc)),gon=5.0e7f;
 float qgr=qg[x],qra=qr[x],qsn=qs[x];
 if(qgr>1.e-15f){gon=2.38f*af_pow(pi*400.0f/(rho[x]*qgr),0.92f);gon=fmaxf(1.e4f,fminf(gon,5.e7f));}
 float ron=1.e10f;
 if(qra>1.e-15f){
  float arg=(0.0001f-qra)/(0.25f*0.0001f);
#ifdef GPUWM_WRF_EXACT
  float v=(float)tanh((double)arg);
#else
  float v=tanhf(arg);
#endif
  ron=((1.e10f-8.e6f)*0.5f)*v+((1.e10f+8.e6f)*0.5f);
 }
 qra=fmaxf(qra,0.0f);qsn=fmaxf(qsn,0.0f);qgr=fmaxf(qgr,0.0f);
 ze[x]=factor_r*af_pow(rho[x]*qra,1.75f)/af_pow(ron,.75f)
       +fs*af_pow(rho[x]*qsn,1.75f)/af_pow(son,.75f)
       +fg*af_pow(rho[x]*qgr,1.75f)/af_pow(gon,.75f);
 if(ze[x]<0.0f)ze[x]=0.0f;
}

extern "C" __global__ void afwa_radar_vil(const float* qr,const float* qs,const float* qg,
 const float* rho,const float* dz,const float* z,const float* ze,
 float* vil,float* rvil,float* refd,float* comp,float* etop,int nz,int n,int dovil,int doradar) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 float v=0.0f,rv=0.0f,r=0.0f,rc=0.0f,top=0.0f;int kt=-1;
 for(int k=0;k<nz;++k){size_t x=(size_t)k*n+c;
  if(dovil){v=v+(qr[x]+qs[x]+qg[x])*dz[x]*rho[x];rv=rv+0.00344f*af_pow(ze[x],.57143f)*dz[x]/1000.0f;}
  if(doradar){
   if(ze[x]>1.e-20f){float db=10.0f*af_log10(ze[x]);if(k==0)r=fmaxf(db,0.0f);if(db>rc)rc=db;}
   if(ze[x]>63.0957f)kt=k;
  }
 }
 if(dovil){vil[c]=v;rvil[c]=rv;}
 if(doradar){
  // Native reads the undefined extra mass row if an echo reaches model top.
  if(kt==nz-1)top=nanf("");
  else if(kt>=0){size_t x=(size_t)kt*n+c;top=z[x]+((63.0957f-ze[x])/(ze[x+n]-ze[x]))*(z[x+n]-z[x]);}
  refd[c]=r;comp[c]=rc;etop[c]=top;
 }
}

extern "C" __global__ void afwa_visibility(
 const float* t,const float* p,const float* q,const float* z,const float* u,const float* v,
 const float* ht,const float* qr,const float* qs,const float* qg,const float* rho,
 const float* psfc,const float* t2,const float* q2,const float* pwat,
 float* vis,float* dust,float* alpha,int nz,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 float wind=-999.0f;
 for(int k=0;k<nz;++k){size_t x=(size_t)k*n+c;
  float zagl=z[x]-ht[c];
  if(zagl>=125.0f){float speed=sqrtf(u[x]*u[x]+v[x]*v[x]);
   if(k==0)wind=speed;
   else{float previous=sqrtf(u[x-n]*u[x-n]+v[x-n]*v[x-n]);
    wind=previous+((125.0f-(z[x-n]-ht[c]))/(zagl-(z[x-n]-ht[c])))*(speed-previous);}
   break;
  }
 }
 float br=1.1f*af_pow(1000.0f*rho[c]*(qr[c]+qg[c]),.75f);
 float bs=10.36f*af_pow(1000.0f*rho[c]*qs[c],.78f);
 float ext=(br+bs)/1000.0f,rh=af_rh(psfc[c],t2[c],q2[c]);
 float haze=999999.0f,ah=3.6f;
 if(q2[c]>0.0f){
  haze=1500.0f*(105.0f-rh)*(5.0f/fminf(1000.0f*q2[c],5.0f));
  ah=0.1f+pwat[c]/25.0f+wind/3.0f+(100.0f-rh)/10.0f+1.0f/(1000.0f*q2[c]);ah=fminf(ah,3.6f);
 }
 float hydro=ext>0.0f?fminf(3.912f/ext,999999.0f):999999.0f;
 if(hydro<haze){vis[c]=hydro;alpha[c]=3.6f;}else{vis[c]=haze;alpha[c]=ah;}
 // This engine's WRF_CHEM=0 path has the native five dust bins equal to zero.
 dust[c]=999999.0f;
}

// WRF f52c197e module_diag_afwa.F:160-181. NaNs deliberately retain
// native comparison semantics: none of these ordered comparisons is true.
extern "C" __global__ void afwa_bad_data_check(
 const real* u,const real* v,const real* theta,unsigned int* status,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(u[i]>300.f || u[i]<-300.f || v[i]>300.f || v[i]<-300.f
    || theta[i]>9999.f || theta[i]<99.f)atomicOr(status,1u);
}

// WRF 4.7.1 module_diag_afwa.F:1377-1614. Default implicit typing makes
// native nc an INTEGER; retain its truncation before the WDM6 exponential.
#ifdef GPUWM_WRF_EXACT
#define af_log gfk_log
#else
#define af_log logf
#endif
extern "C" __global__ void afwa_icing_mixing(const float* t,const float* rho,const float* qr,const float* qc,const float* number,float* qlarge,float* qsmall,int nz,int levels,int opt,int count){
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nz*count)return;
 if(x<levels*count && t[x]<273.15f){
   // Native implicit INTEGER nc uses x86 CVTTSS2SI: an out-of-range
   // concentration yields INT_MIN. CUDA conversion saturates at INT_MAX.
   // The real forecast witnesses this native-undefined conversion; retain
   // its observed branch while converting every representable value exactly.
   int nc=(number[x]>=2147483648.f || number[x]<-2147483648.f || !isfinite(number[x]))
          ? (-2147483647-1) : __float2int_rz(number[x]);
   float cloud=qc[x],rain=qr[x],n=(float)nc,den=rho[x];
   float fl=0.f,fs=0.f;
   if(opt==2){
    if(cloud<2.5e-4f)fl=395000.f*af_pow(cloud,2.f)+102.9f*cloud;
    else if(cloud<1.4e-3f)fl=276.1f*cloud-0.01861f;
    else fl=.3f*af_log(641.789f*cloud)+.4f;
   }else if(opt==3 || opt==4){
    if(cloud<5.e-4f)fl=50420.f*af_pow(cloud,2.f)+29.39f*cloud;
    else if(cloud<1.4e-3f)fl=97.65f*cloud-.02152f;
    else fl=.2f*af_log(646.908f*cloud)+.135f;
   }else if(opt==5){
    if(cloud<1.4e-3f)fl=28000.f*af_pow(cloud,2.f)+.1f*cloud;
    else if(cloud<2.6e-3f)fl=112.351f*cloud-.102272f;
    else fl=.3f*af_log(654.92f*cloud)*.301607f;
   }else if(opt==6 || opt==7){
    if(cloud>1.e-12f && n>1.e-12f){
     float a=(-n)*3141.59265f*__uint_as_float(0x2a0cbccbu)/(6000.f*den*cloud);
     float b=(-n)*3141.59265f*8.f/(6000.f*den*cloud);
     float ns=(-n)*af_exp(a)+n,all=(-n)*af_exp(b)+n;
     fl=1.f-ns/all;
    }
   }
   fl=fmaxf(fl,0.f);if(opt!=0)fs=1.f-fl;
   float ql=fmaxf(rain+fl*cloud,0.f),qs=fmaxf(fs*cloud,0.f);
   qlarge[x]=ql;qsmall[x]=qs;
}else {qlarge[x]=0.f;qsmall[x]=0.f;}}
extern "C" __global__ void afwa_icing_integrate(const float* t,const float* z,
 const float* dz,const float* rho,const float* qr,const float* qc,const float* number,
 float* fz,float* large,float* small,float* large_max,float* small_max,
 float* top,float* bottom,float* qlarge,float* qsmall,
 int nz,int levels,int native_top,int opt,int count) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
 float freeze=-999.f,bot=-999.f,ceiling=-999.f,lg=0.f,sm=0.f,lgmax=0.f,smmax=0.f;
 int kt=-1,kb=-1;
 for(int k=0;k<levels;++k){
  int x=k*count+c;
  if(t[x]<273.15f){
   float ql=qlarge[x],qs=qsmall[x];
   lg=lg+(ql*dz[x])*rho[x];sm=sm+(qs*dz[x])*rho[x];
   if(ql>lgmax)lgmax=ql;if(qs>smmax)smmax=qs;
   if(freeze==-999.f){
    if(k!=0){int y=x-count;freeze=z[y]+((273.15f-t[y])/(t[x]-t[y]))*(z[x]-z[y]);}
    else freeze=z[x];
   }
   if(ql+qs>=1.e-5f){if(kb==-1)kb=k;kt=k;}
  }
 }
 if(kb!=-1){
  int x=kb*count+c;
  if(kb!=0){int y=x-count;float qa=qlarge[y]+qsmall[y],qb=qlarge[x]+qsmall[x];
   bot=z[y]+((1.e-5f-qa)/(qb-qa))*(z[x]-z[y]);bot=fmaxf(bot,freeze);
  }else bot=z[x];
 }
 if(kt!=-1 && kt!=native_top){
  int x=kt*count+c,y=x+count;
  // Native kpe includes the extra allocated top slot. Its qicing is
  // initialized here and its unfilled z retains WRF's zero allocation.
  float qa=qlarge[x]+qsmall[x],qb=kt+1<nz?qlarge[y]+qsmall[y]:0.f;
  float zn=kt+1<nz?z[y]:0.f;
  ceiling=z[x]+((1.e-5f-qa)/(qb-qa))*(zn-z[x]);ceiling=fmaxf(ceiling,bot);
 }
 fz[c]=freeze;large[c]=lg;small[c]=sm;large_max[c]=lgmax;small_max[c]=smmax;top[c]=ceiling;bottom[c]=bot;
}
