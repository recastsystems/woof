// WRF 4.7.1 module_ra_farms.F (byte-identical to 4.6.1).
// WRF public-domain notice: gpuwm/data/wrf_radiation/LICENSE-WRF.txt.
// Strict log10 uses CORE-MATH; its MIT notice is in wrf471_log10f.cuh.
// Every scalar operation is rounded in WRF statement order.
// Math routines are WOOF float32 routines.
// The shared cal_cldfra1 array composition uses native float32 libm even
// in fast forecasts. Fuse its operations without contracting their roundings.
// Keep this entry before the FARMS-only fast libm aliases below.
extern "C" __global__ void farms_cloud_fraction(
 const float* qv,const float* qc,const float* qi,const float* qs,
 const float* t,const float* p,float* out,int n,int fc,int fi,int fs,int ferrier){
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;
 if(!fc){out[x]=0.f;return;}
 float tc=__fsub_rn(t[x],273.15f);
 float esw=__fmul_rn(__fmul_rn(1000.f,.61078f),
   gfk_exp(__fdiv_rn(__fmul_rn(17.2693882f,tc),__fsub_rn(t[x],35.86f))));
 float esi=__fmul_rn(__fmul_rn(1000.f,.61078f),
   gfk_exp(__fdiv_rn(__fmul_rn(21.8745584f,tc),__fsub_rn(t[x],7.66f))));
 float ep2=__fdiv_rn(287.f,461.6f);
 float qvsw=__fdiv_rn(__fmul_rn(ep2,esw),__fsub_rn(p[x],esw));
 float qvsi=__fdiv_rn(__fmul_rn(ep2,esi),__fsub_rn(p[x],esi));
 float qcld,weight;
 if(fi&&fs){qcld=__fadd_rn(__fadd_rn(qi[x],qc[x]),qs[x]);
   weight=qcld<1.e-12f?0.f:__fdiv_rn(__fadd_rn(qi[x],qs[x]),fmaxf(qcld,1.e-12f));}
 else if(fi){qcld=__fadd_rn(qi[x],qc[x]);
   weight=qcld<1.e-12f?0.f:__fdiv_rn(qi[x],fmaxf(qcld,1.e-12f));
   if(ferrier&&qcld>=1.e-12f&&tc< -40.f)weight=1.f;}
 else{qcld=qc[x];weight=qcld<1.e-12f?0.f:(t[x]>273.15f?0.f:1.f);}
 float qvs=__fadd_rn(__fmul_rn(__fsub_rn(1.f,weight),qvsw),__fmul_rn(weight,qvsi));
 float rh=__fdiv_rn(qv[x],qvs);
 float subsat=fmaxf(1.e-10f,__fsub_rn(qvs,qv[x]));
 float arg=fmaxf(-6.9f,__fdiv_rn(__fmul_rn(-100.f,qcld),gfk_pow(subsat,.49f)));
 float frac=__fmul_rn(gfk_pow(fmaxf(1.e-10f,rh),.25f),__fsub_rn(1.f,gfk_exp(arg)));
 if(frac<.01f)frac=0.f;
 out[x]=qcld<1.e-12f?0.f:(rh>=1.f?1.f:frac);
}

#ifndef GPUWM_WRF_EXACT
#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#define FDIV(a,b) ((a)/(b))
#define gfk_pow powf
#define gfk_exp expf
#define gfk_log logf
#define glibc_acosf acosf
#define glibc_cosf cosf
#define glibc_sinf sinf
#endif
const float ptau_min=1.e-6f,delta_min=1.e-6f,de_ice_min=5.f,de_ice_max=140.f,de_cloud_min=5.f,de_cloud_max=120.f,tau_min=.0001f,tau_max=300.f;
__device__ float farms_log10(float x) {
#ifdef GPUWM_WRF_EXACT
 return wrf471_log10f(x);
#else
 return log10f(x);
#endif
}
__device__ float farms_sinh(float x) {
#ifndef GPUWM_WRF_EXACT
return sinhf(x);
#else
float a=fabsf(x);
// Stable hyperbolic identity with explicit binary32 rounding at each step.
float t=__double2float_rn(plm_exp((double)a)-1.0);
float s=a<1.f?FMUL(.5f,FSUB(FMUL(2.f,t),FDIV(FMUL(t,t),FADD(t,1.f)))):FMUL(.5f,FADD(t,FDIV(t,FADD(t,1.f))));
return x<0.f?-s:s;
#endif
}
__device__ void farms_sunearth(int juday, float &r) {
float pi,b,r1;
pi=glibc_acosf(-1.0f);
b=FDIV(FMUL(FMUL(2.0f,pi),juday),365.0f);
r1=FADD(FADD(FADD(FADD(1.00011f,FMUL(0.034221f,glibc_cosf(b))),FMUL(0.00128f,glibc_sinf(b))),FMUL(0.000719f,glibc_cosf(FMUL(2.0f,b)))),FMUL(7.7e-05f,glibc_sinf(FMUL(2.0f,b))));
r=gfk_pow(r1,-0.5f);
}
__device__ void farms_bird(float p,float albdo,float ssa,float g,float z,float radius,float beta,float alpha,float ozone,float w,float &tddclr,float &tduclr,float &ruuclr) {
float degrad,airmass,airmassp,t_rayleigh,x0,t_o,t_gases,xw,t_water,tau038,tau050,taua,t_aerosol,t_aa,t_as,f0,fdif,ftotal,fddclr;
degrad=0.017453293f;
airmass=FDIV(1.0f,FADD(glibc_cosf(FMUL(z,degrad)),FMUL(0.15f,gfk_pow(FSUB(93.885f,z),-1.25f))));
airmassp=FDIV(FMUL(p,airmass),1013.0f);
t_rayleigh=gfk_exp(FMUL(FMUL(-0.0903f,gfk_pow(airmassp,0.84f)),FSUB(FADD(1.0f,airmassp),gfk_pow(airmassp,1.01f))));
x0=FMUL(ozone,airmass);
t_o=FSUB(FSUB(1.0f,FMUL(FMUL(0.1611f,x0),gfk_pow(FADD(1.0f,FMUL(139.48f,x0)),-0.3035f))),FDIV(FMUL(0.002715f,x0),FADD(FADD(1.0f,FMUL(0.044f,x0)),FMUL(0.0003f,FMUL(x0,x0)))));
t_gases=gfk_exp(FMUL(-0.0127f,gfk_pow(airmassp,0.26f)));
xw=FMUL(w,airmass);
t_water=FSUB(1.0f,FDIV(FMUL(2.4959f,xw),FADD(gfk_pow(FADD(1.0f,FMUL(79.034f,xw)),0.6828f),FMUL(6.385f,xw))));
tau038=FMUL(beta,gfk_pow(0.38f,-alpha));
tau050=FMUL(beta,gfk_pow(0.5f,-alpha));
taua=FADD(FMUL(0.2758f,tau038),FMUL(0.35f,tau050));
t_aerosol=gfk_exp(FMUL(FMUL(-gfk_pow(taua,0.873f),FSUB(FADD(1.0f,taua),gfk_pow(taua,0.7088f))),gfk_pow(airmass,0.9108f)));
t_aa=FSUB(1.0f,FMUL(FMUL(0.1f,FADD(FSUB(1.0f,airmass),gfk_pow(airmass,1.06f))),FSUB(1.0f,t_aerosol)));
t_as=FDIV(t_aerosol,t_aa);
ruuclr=FADD(0.0685f,FMUL(FSUB(1.0f,g),FSUB(1.0f,t_as)));
f0=1.0f;
fddclr=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(f0,glibc_cosf(FMUL(z,degrad))),0.9662f),t_rayleigh),t_o),t_gases),t_water),t_aerosol);
fdif=FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(f0,glibc_cosf(FMUL(z,degrad))),0.79f),t_o),t_gases),t_water),t_aa),FADD(FMUL(0.5f,FSUB(1.0f,t_rayleigh)),FMUL(g,FSUB(1.0f,t_as)))),FADD(FSUB(1.0f,airmass),gfk_pow(airmass,1.02f)));
ftotal=FDIV(FADD(fddclr,fdif),FSUB(1.0f,FMUL(albdo,ruuclr)));
tddclr=FDIV(fddclr,FMUL(f0,glibc_cosf(FMUL(z,degrad))));
tduclr=FDIV(FSUB(ftotal,fddclr),FMUL(f0,glibc_cosf(FMUL(z,degrad))));
}
__device__ void farms_watermodel(float tau,float de,float solarangle,float &tducld,float &ruucld) {
float solarconst,ptau,pdhi,delta,y,ppdhi;
solarconst=1385.7218f;
ptau=FSUB(FMUL(FADD(2.885f,FMUL(0.002f,FSUB(de,60.0f))),solarangle),0.007347f);
ptau=fmaxf(ptau,ptau_min);
pdhi=FMUL(FMUL(FADD(1.0f,FMUL(FSUB(de,60.0f),0.0002f)),1087.24f),gfk_pow(solarangle,1.1605f));
delta=FADD(FSUB(FADD(FADD(FMUL(-0.644531f,solarangle),1.20117f),FDIV(0.129807f,solarangle)),FDIV(0.00121096f,FMUL(solarangle,solarangle))),FDIV(1.52587e-07f,FMUL(FMUL(solarangle,solarangle),solarangle)));
delta=fmaxf(delta,delta_min);
y=FMUL(FMUL(0.012f,FSUB(tau,ptau)),solarangle);
ppdhi=FMUL(FMUL(FADD(1.0f,farms_sinh(y)),pdhi),gfk_exp(FDIV(-FMUL(FSUB(farms_log10(tau),farms_log10(ptau)),FSUB(farms_log10(tau),farms_log10(ptau))),delta)));
tducld=FDIV(ppdhi,FMUL(solarconst,solarangle));
if (tau<1.0f) {
ruucld=FMUL(0.107359f,tau);
}
if (tau>=1.0f) {
ruucld=FSUB(1.03f,gfk_exp(FDIV(FMUL(-FADD(0.5f,farms_log10(tau)),FADD(0.5f,farms_log10(tau))),3.105f)));
}
}
__device__ void farms_icemodel(float tau,float de,float solarangle,float &tducld,float &ruucld) {
float solarconst,ptau,pdhi,delta,y,ppdhi;
solarconst=1385.7218f;
if (de<=26.0f) {
ptau=FSUB(FMUL(2.8487f,solarangle),0.0029f);
}
if (de>26.0f) {
ptau=FSUB(FMUL(FADD(2.8355f,FMUL(FSUB(100.0f,de),0.006f)),solarangle),0.00612f);
}
ptau=fmaxf(ptau,ptau_min);
pdhi=FMUL(1047.6367f,gfk_pow(solarangle,1.0883f));
delta=FSUB(FADD(FADD(FMUL(-0.0549531f,solarangle),0.617632f),FDIV(0.17876f,solarangle)),FDIV(0.002174f,FMUL(solarangle,solarangle)));
delta=fmaxf(delta,delta_min);
y=FMUL(FMUL(0.01f,FSUB(tau,ptau)),solarangle);
ppdhi=FMUL(FMUL(FADD(1.0f,farms_sinh(y)),pdhi),gfk_exp(FDIV(-FMUL(FSUB(farms_log10(tau),farms_log10(ptau)),FSUB(farms_log10(tau),farms_log10(ptau))),delta)));
tducld=FDIV(ppdhi,FMUL(solarconst,solarangle));
if (tau<1.0f) {
ruucld=FMUL(0.094039f,tau);
}
if (tau>=1.0f) {
ruucld=FSUB(1.02f,gfk_exp(FDIV(FMUL(-FADD(0.5f,farms_log10(tau)),FADD(0.5f,farms_log10(tau))),3.25f)));
}
}
__device__ void farms_clearskyall(float p,float albdo,float ssa,float g,float z,float radius,float beta,float alpha,float ozone,float w,float &tddclr,float &tduclr,float &ruuclr,float &tuuclr) {
#ifdef GPUWM_WRF_EXACT
float aa[10],mu[10];
for(int i=0;i<10;i++) { mu[i]=FADD(FMUL(FSUB((float)(i+1),1.0f),.1f),.1f); float angle=FDIV(FMUL(glibc_acosf(mu[i]),180.f),3.1415927410125732f); farms_bird(p,albdo,ssa,g,angle,radius,beta,alpha,ozone,w,tddclr,tduclr,ruuclr); aa[i]=tddclr; }
tuuclr=0.f;
for(int i=0;i<10;i++) tuuclr=FADD(tuuclr,FMUL(FMUL(FMUL(2.f,mu[i]),aa[i]),.1f));
#else
// Accumulate each quadrature ray as it is computed. Unrolling exposes its
// fixed angle to the compiler and removes two per-thread local arrays.
tuuclr=0.f;
#pragma unroll
for(int i=0;i<10;i++) {
 float mu=(float)(i+1)*.1f;
 float angle=acosf(mu)*180.f/3.1415927410125732f;
 farms_bird(p,albdo,ssa,g,angle,radius,beta,alpha,ozone,w,tddclr,tduclr,ruuclr);
 tuuclr+=2.f*mu*tddclr*.1f;
}
#endif
farms_bird(p,albdo,ssa,g,z,radius,beta,alpha,ozone,w,tddclr,tduclr,ruuclr);
}
__device__ void farms_farms(float p_pa,float albdo,float ssa,float g,float solarangle,float beta,float alpha,float w_mm,float tau_qv,float tau_qi,float tau_qs,float cldfra_h,float re_cloud_path_m,float re_ice_path_m,float re_snow_path_m,int juday,float &ghi,float &dni,float &dif,float &dir,float &ghi_clear,float &dni_clear) {
float de_cloud,de_cloud2,de_ice,de_ice2,de_snow,de_snow2,f0,f1,ftotal,ozone,p,radius,ruucld,pi,ruucld_water,ruucld_ice,ruucld_snow,ruuclr,tau_qv2,tau_qi2,tau_qs2,tau,tddcld,tddclr,tducld,tducld_ice,tducld_snow,tducld_water,tduclr,tuuclr,w,z,tau_tot;
pi=glibc_acosf(-1.0f);
p=FDIV(p_pa,100.0f);
ozone=0.265f;
w=FDIV(w_mm,10.0f);
de_cloud=FDIV(FMUL(2.0f,re_cloud_path_m),1e-06f);
de_ice=FDIV(FMUL(2.0f,re_ice_path_m),1e-06f);
de_snow=FDIV(FMUL(2.0f,re_snow_path_m),1e-06f);
de_cloud=FDIV(FMUL(2.0f,re_cloud_path_m),1e-06f);
de_cloud2=fmaxf(de_cloud,de_cloud_min);
de_cloud2=fminf(de_cloud2,de_cloud_max);
de_ice=FDIV(FMUL(2.0f,re_ice_path_m),1e-06f);
de_ice2=fmaxf(de_ice,de_ice_min);
de_ice2=fminf(de_ice2,de_ice_max);
de_snow=FDIV(FMUL(2.0f,re_snow_path_m),1e-06f);
de_snow2=fmaxf(de_snow,de_ice_min);
de_snow2=fminf(de_snow2,de_ice_max);
tau_qv2=fmaxf(tau_qv,tau_min);
tau_qv2=fminf(tau_qv2,tau_max);
tau_qi2=fmaxf(tau_qi,tau_min);
tau_qi2=fminf(tau_qi2,tau_max);
tau_qs2=fmaxf(tau_qs,tau_min);
tau_qs2=fminf(tau_qs2,tau_max);
tau_tot=0.0f;
if (tau_qv>tau_min) tau_tot=FADD(tau_tot,tau_qv2);
if (tau_qi>tau_min) tau_tot=FADD(tau_tot,tau_qi2);
if (tau_qs>tau_min) tau_tot=FADD(tau_tot,tau_qs2);
tau_tot=fminf(tau_tot,tau_max);
z=FDIV(FMUL(glibc_acosf(solarangle),180.0f),pi);
farms_sunearth( juday, radius );
f0=FDIV(1361.2f,FMUL(radius,radius));
farms_clearskyall(p, albdo, ssa, g, z, radius, beta, alpha, ozone, w, tddclr, tduclr, ruuclr, tuuclr);
if (tau_qv>tau_min) {
farms_watermodel(tau_tot, de_cloud2, solarangle, tducld_water, ruucld_water);
} else {
tducld_water=0.0f;
ruucld_water=0.0f;
tau_qv2=0.0f;
}
if (tau_qi>tau_min) {
farms_icemodel(tau_tot, de_ice2, solarangle, tducld_ice, ruucld_ice);
} else {
tducld_ice=0.0f;
ruucld_ice=0.0f;
tau_qi2=0.0f;
}
if (tau_qs>tau_min) {
farms_icemodel(tau_tot, de_snow2, solarangle, tducld_snow, ruucld_snow);
} else {
tducld_snow=0.0f;
ruucld_snow=0.0f;
tau_qs2=0.0f;
}
if (tau_tot>0.0f) {
tducld=FDIV(FADD(FADD(FMUL(tau_qv2,tducld_water),FMUL(tau_qi2,tducld_ice)),FMUL(tau_qs2,tducld_snow)),tau_tot);
tducld=fminf(tducld,1.0f);
tducld=fmaxf(tducld,0.0f);
ruucld=FDIV(FADD(FADD(FMUL(tau_qv2,ruucld_water),FMUL(tau_qi2,ruucld_ice)),FMUL(tau_qs2,ruucld_snow)),tau_tot);
ruucld=fminf(ruucld,1.0f);
ruucld=fmaxf(ruucld,0.0f);
} else {
tducld=0.0f;
ruucld=0.0f;
}
tau=FADD(FADD(tau_qv2,tau_qi2),tau_qs2);
tau=fminf(tau,tau_max);
tau=fmaxf(tau,tau_min);
tddcld=gfk_exp(FDIV(-tau,solarangle));
dni_clear=FMUL(f0,tddclr);
dni=FMUL(dni_clear,tddcld);
dir=FMUL(dni,solarangle);
f1=FMUL(FMUL(solarangle,f0),FADD(FMUL(tddcld,FADD(tddclr,tduclr)),FMUL(tducld,tuuclr)));
ftotal=FDIV(f1,FSUB(1.0f,FMUL(albdo,FADD(ruuclr,FMUL(FMUL(ruucld,tuuclr),tuuclr)))));
ghi=ftotal;
ghi_clear=FMUL(FMUL(solarangle,f0),FDIV(FADD(tddclr,tduclr),FSUB(1.0f,FMUL(albdo,ruuclr))));
ghi=FADD(FMUL(cldfra_h,ghi),FMUL(FSUB(1.0f,cldfra_h),ghi_clear));
dni=FADD(FMUL(cldfra_h,dni),FMUL(FSUB(1.0f,cldfra_h),dni_clear));
dif=FSUB(ghi,dir);
}

// One thread per horizontal column; arrays are bottom-up (nz,ncol).
extern "C" __global__ void farms_column(
int n,int nz,int has_reqc,int has_reqi,int has_reqs,float julian,
const float *p,const float *rho,const float *dz,const float *alb,
const float *ssa,const float *g,const float *aod,const float *ang,
const float *cz,const float *qv,const float *qi,const float *qs,const float *qc,
const float *rc,const float *ri,const float *rs,const float *cf,
float *out) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
if(cz[i]<=0.f) { for(int a=0;a<6;a++) out[a*n+i]=0.f; return; }
float pmw=0.f,lwp=0.f,iwp=0.f,swp=0.f,nrc=0.f,nri=0.f,nrs=0.f,cwp=0.f,ncf=0.f;
for(int k=0;k<nz;k++) {
int j=k*n+i;
float rhodz=FDIV(FMUL(rho[j],dz[j]),FADD(1.f,qv[j]));
float c=has_reqc?rc[j]:8.e-6f, ice=has_reqi?ri[j]:24.e-6f, snow=has_reqs?rs[j]:24.e-6f;
if(has_reqc && cf[j]>0.f && c<2.5e-6f) c=8.e-6f;
if(has_reqi && cf[j]>0.f && ice<5.e-6f) ice=24.e-6f;
if(has_reqs && cf[j]>0.f && snow<10.e-6f) snow=24.e-6f;
pmw=FADD(pmw,FMUL(qv[j],rhodz));
lwp=FADD(lwp,FMUL(qc[j],rhodz));
iwp=FADD(iwp,FMUL(qi[j],rhodz));
swp=FADD(swp,FMUL(qs[j],rhodz));
nrc=FADD(nrc,FMUL(FMUL(qc[j],c),rhodz));
nri=FADD(nri,FMUL(FMUL(qi[j],ice),rhodz));
nrs=FADD(nrs,FMUL(FMUL(qs[j],snow),rhodz));
float q=FADD(FADD(qc[j],qi[j]),qs[j]);
cwp=FADD(cwp,FMUL(q,rhodz)); ncf=FADD(ncf,FMUL(FMUL(q,cf[j]),rhodz));
}
float c=lwp>0.f?FDIV(nrc,lwp):0.f, ice=iwp>0.f?FDIV(nri,iwp):0.f,snow=swp>0.f?FDIV(nrs,swp):0.f;
float tc=c>0.f?FDIV(FDIV(FMUL(1.5f,lwp),c),1000.f):0.f;
float ti=ice>0.f?FMUL(FMUL(iwp,1000.f),FADD(.02f,FDIV(4.2f,FMUL(FMUL(2.f,ice),1.e6f)))):0.f;
float ts=snow>0.f?FMUL(FMUL(swp,1000.f),FADD(.02f,FDIV(4.2f,FMUL(FMUL(2.f,snow),1.e6f)))):0.f;
float beta=FMUL(aod[i],gfk_pow(FDIV(1000.f,550.f),-ang[i]));
farms_farms(p[i],alb[i],ssa[i],g[i],cz[i],beta,ang[i],pmw,tc,ti,ts,cwp>0.f?FDIV(ncf,cwp):0.f,c,ice,snow,(int)julian,out[i],out[2*n+i],out[3*n+i],out[n+i],out[4*n+i],out[5*n+i]);
}

#ifndef GPUWM_WRF_EXACT
// A tile reads 32 adjacent columns at eight levels at a time. Only the
// fast reduction changes association; the ordered referee entry stays above.
extern "C" __global__ void farms_column_fast(
int n,int nz,int has_reqc,int has_reqi,int has_reqs,float julian,
const float *p,const float *rho,const float *dz,const float *alb,
const float *ssa,const float *g,const float *aod,const float *ang,
const float *cz,const float *qv,const float *qi,const float *qs,const float *qc,
const float *rc,const float *ri,const float *rs,const float *cf,
float *out) {
 const int x=threadIdx.x, y=threadIdx.y, i=blockIdx.x*32+x;
 const bool daylight=i<n && cz[i]>0.f;
 float v[9]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
 if(daylight) for(int k=y;k<nz;k+=8) {
  const int j=k*n+i;
  const float rhodz=rho[j]*dz[j]/(1.f+qv[j]);
  float c=has_reqc?rc[j]:8.e-6f, ice=has_reqi?ri[j]:24.e-6f, snow=has_reqs?rs[j]:24.e-6f;
  if(has_reqc && cf[j]>0.f && c<2.5e-6f) c=8.e-6f;
  if(has_reqi && cf[j]>0.f && ice<5.e-6f) ice=24.e-6f;
  if(has_reqs && cf[j]>0.f && snow<10.e-6f) snow=24.e-6f;
  v[0]+=qv[j]*rhodz; v[1]+=qc[j]*rhodz;
  v[2]+=qi[j]*rhodz; v[3]+=qs[j]*rhodz;
  v[4]+=qc[j]*c*rhodz; v[5]+=qi[j]*ice*rhodz; v[6]+=qs[j]*snow*rhodz;
  const float q=qc[j]+qi[j]+qs[j];
  v[7]+=q*rhodz; v[8]+=q*cf[j]*rhodz;
 }
 __shared__ float partial[9][8][32];
 #pragma unroll
 for(int a=0;a<9;a++) partial[a][y][x]=v[a];
 __syncthreads();
 if(y || i>=n) return;
 if(!daylight) {
  #pragma unroll
  for(int a=0;a<6;a++) out[a*n+i]=0.f;
  return;
 }
 #pragma unroll
 for(int a=0;a<9;a++) {
  v[a]=partial[a][0][x];
  #pragma unroll
  for(int lane=1;lane<8;lane++) v[a]+=partial[a][lane][x];
 }
 const float pmw=v[0],lwp=v[1],iwp=v[2],swp=v[3];
 const float c=lwp>0.f?v[4]/lwp:0.f,ice=iwp>0.f?v[5]/iwp:0.f,snow=swp>0.f?v[6]/swp:0.f;
 const float tc=c>0.f?1.5f*lwp/c/1000.f:0.f;
 const float ti=ice>0.f?iwp*1000.f*(.02f+4.2f/(2.f*ice*1.e6f)):0.f;
 const float ts=snow>0.f?swp*1000.f*(.02f+4.2f/(2.f*snow*1.e6f)):0.f;
 const float beta=aod[i]*powf(1000.f/550.f,-ang[i]);
 farms_farms(p[i],alb[i],ssa[i],g[i],cz[i],beta,ang[i],pmw,tc,ti,ts,
  v[7]>0.f?v[8]/v[7]:0.f,c,ice,snow,(int)julian,
  out[i],out[2*n+i],out[3*n+i],out[n+i],out[4*n+i],out[5*n+i]);
}
#endif
