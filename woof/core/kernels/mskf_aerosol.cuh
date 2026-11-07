// Generated closed transcription from WRF 4.6.1. Source SHA256: 4bca54b294def08ffc98ef69a30650c943e2c394f34802e75e50c7f1a2896fcb
// Generation refuses unknown statements. Each emitted statement has its WRF line.

// Runtime arrays use one double-sized cell per element, interleaved across lanes.
template<class T> struct MaArray {
 double* ptr; int n1,n2,n3,l1,l2,l3,step=1;
 __device__ T& at(int x) const { return *reinterpret_cast<T*>(ptr+(size_t)x*step*32); }
 __device__ MaArray<T> slice(int i,int j) const { return {ptr+(size_t)((i-l1)+n1*(j-l2))*32,n3,1,1,l3,1,1,n1*n2}; }
 __device__ int size() const { return n1*n2*n3; }
 __device__ T& operator()(int i) const { return at(i-l1); }
 __device__ T& operator()(int i,int j) const { return at((i-l1)+n1*(j-l2)); }
 __device__ T& operator()(int i,int j,int k) const { return at((i-l1)+n1*((j-l2)+n2*(k-l3))); }
};
struct MaArena {
 double* ptr; int used;
 template<class T> __device__ MaArray<T> make(int a,int b,int c=1,int d=1,int e=1,int f=1) {
  MaArray<T> x{ptr+(size_t)used*32,b-a+1,d-c+1,f-e+1,a,c,e};
  used+=x.size(); return x;
 }
};
template<class A,class B> __device__ auto ma_max(A a,B b)->decltype(a+b) { using T=decltype(a+b); return a>b?(T)a:(T)b; }
template<class A,class B,class... C> __device__ auto ma_max(A a,B b,C... c)->decltype(a+b) { return ma_max(ma_max(a,b),c...); }
template<class A,class B> __device__ auto ma_min(A a,B b)->decltype(a+b) { using T=decltype(a+b); return a<b?(T)a:(T)b; }
template<class A,class B,class... C> __device__ auto ma_min(A a,B b,C... c)->decltype(a+b) { return ma_min(ma_min(a,b),c...); }
#ifdef GPUWM_MSKF_FAST
__device__ double ma_exp(double x) {return exp(x);}
__device__ float ma_exp(float x) {return gfk_exp(x);}
__device__ double ma_log(double x) {return log(x);}
__device__ float ma_log(float x) {return gfk_log(x);}
#else
__device__ double ma_exp(double x) {return g64_exp::exp(x);}
__device__ float ma_exp(float x) {return gfk_exp(x);}
__device__ double ma_log(double x) {return g64_log::log(x);}
__device__ float ma_log(float x) {return gfk_log(x);}
#endif
__device__ double ma_sqrt(double x) {return sqrt(x);}
__device__ float ma_sqrt(float x) {return sqrtf(x);}
__device__ double ma_abs(double x) {return fabs(x);}
__device__ float ma_abs(float x) {return fabsf(x);}
__device__ int ma_abs(int x) {return abs(x);}
__device__ double ma_aint(double x) {return trunc(x);}
__device__ float ma_aint(float x) {return truncf(x);}
__device__ double ma_log10(double x) {return log10(x);}
template<class A,class B> __device__ auto ma_pow(A x,B y)->decltype(x+y) {
 using T=decltype(x+y);
#ifdef GPUWM_MSKF_FAST
 if constexpr(sizeof(T)==8) return pow((double)x,(double)y);
 else return gfk_pow((float)x,(float)y);
#else
 if constexpr(sizeof(T)==8) return g64_pow::pow((double)x,(double)y);
 else return gfk_pow((float)x,(float)y);
#endif
}
template<int N,class T> __device__ T ma_powi(T x) {
 if constexpr(N==0) return T(1);
 else if constexpr(N<0) return T(1)/ma_powi<-N>(x);
 else if constexpr(N==1) return x;
 else if constexpr(N==2) return x*x;
 else if constexpr(N==3) return (x*x)*x;
 else if constexpr(N==4) {T a=x*x; return a*a;}
 else if constexpr(N==5) {T a=x*x;return (a*x)*a;}
 else if constexpr(N==6) {T a=(x*x)*x;return a*a;}
 else {T a=ma_powi<N/2>(x); if constexpr(N%2) return (a*a)*x; else return a*a;}
}

struct MaPhysics { MaArena arena; int error=0;
static constexpr int naer_cu=10;
static constexpr int pcols=1;
double g;
double mw;
double r;
double rv;
double rr;
double cpp;
double rhow;
double xlf;
static constexpr double gravit=9.80616;
static constexpr double rair=287.04239;
static constexpr double tmelt=273.15;
static constexpr double cpair=1.00464e3;
static constexpr double rh2o=461.915;
static constexpr double r_universal=8.31447e3;
static constexpr double mwh2o=18.;
static constexpr double rhoh2o=1.000e3;
static constexpr double latvap=2.501e6;
static constexpr double latice=3.337e5;
static constexpr double epsilo=0.622;
double rhosn;
double rhoi;
double ac;
double bc;
double as;
double bs;
double ai;
double bi;
double ar;
double br;
double ci;
double di;
double cs;
double ds;
double cr;
double dr;
double eii;
double ecc;
double ecr;
double dcs;
double f14;
double qsmall;
double bimm;
double aimm;
double rhosu;
double mi0;
double rin;
double pi;
double rn_dst1;
double rn_dst2;
double rn_dst3;
double rn_dst4;
double t0;
static constexpr int psat=6;
double aten;
MaArray<double> alogsig;
MaArray<double> exp45logsig;
MaArray<double> argfactor;
MaArray<double> amcube;
MaArray<double> smcrit;
MaArray<double> lnsm;
MaArray<double> amcubesulfate;
MaArray<double> smcritsulfate;
MaArray<double> amcubefactor;
MaArray<double> smcritfactor;
MaArray<double> super;
double alogten;
double alog2;
double alog3;
double alogaten;
MaArray<double> supersat;
MaArray<double> ccnfact;
MaArray<double> f1;
MaArray<double> f2;
double third;
double sixth;
double zero;
double sq2;
double sqpi;
int idxsul=1;
int idxdst1=3;
int idxdst2=4;
int idxdst3=5;
int idxdst4=6;
int idxbcphi=10;
MaArray<double> dryrad_aer;
MaArray<double> density_aer;
MaArray<double> hygro_aer;
MaArray<double> dispersion_aer;
MaArray<double> num_to_mass_aer;
__device__ MaPhysics(double* ptr, double* cached=nullptr):arena{ptr,0} {
if(!ptr) return;
// Only species constants use the shared table. Keep the original arena offset
// for every mutable per-column array, including the strict replay layout.
MaArena table_arena{cached ? cached : ptr,0};
alogsig=table_arena.make<double>(1,naer_cu);
exp45logsig=table_arena.make<double>(1,naer_cu);
argfactor=table_arena.make<double>(1,naer_cu);
amcube=table_arena.make<double>(1,naer_cu);
smcrit=table_arena.make<double>(1,naer_cu);
lnsm=table_arena.make<double>(1,naer_cu);
amcubesulfate=table_arena.make<double>(1,pcols);
smcritsulfate=table_arena.make<double>(1,pcols);
amcubefactor=table_arena.make<double>(1,naer_cu);
smcritfactor=table_arena.make<double>(1,naer_cu);
super=table_arena.make<double>(1,psat);
supersat=table_arena.make<double>(1,psat);
ccnfact=table_arena.make<double>(1,psat,1,naer_cu);
f1=table_arena.make<double>(1,naer_cu);
f2=table_arena.make<double>(1,naer_cu);
dryrad_aer=table_arena.make<double>(1,naer_cu);
density_aer=table_arena.make<double>(1,naer_cu);
hygro_aer=table_arena.make<double>(1,naer_cu);
dispersion_aer=table_arena.make<double>(1,naer_cu);
num_to_mass_aer=table_arena.make<double>(1,naer_cu);
arena.used=table_arena.used;
if(!cached) {
dryrad_aer.at(0)=0.695e-7; dryrad_aer.at(1)=0.200e-5; dryrad_aer.at(2)=0.151e-5; dryrad_aer.at(3)=0.151e-5; dryrad_aer.at(4)=0.151e-5; dryrad_aer.at(5)=0.151e-5; dryrad_aer.at(6)=0.212e-7; dryrad_aer.at(7)=0.118e-7; dryrad_aer.at(8)=0.212e-7; dryrad_aer.at(9)=0.118e-7;
density_aer.at(0)=1770.; density_aer.at(1)=2200.; density_aer.at(2)=2600.; density_aer.at(3)=2600.; density_aer.at(4)=2600.; density_aer.at(5)=2600.; density_aer.at(6)=1800.; density_aer.at(7)=1000.; density_aer.at(8)=2600.; density_aer.at(9)=1000.;
hygro_aer.at(0)=0.507; hygro_aer.at(1)=1.160; hygro_aer.at(2)=0.140; hygro_aer.at(3)=0.140; hygro_aer.at(4)=0.140; hygro_aer.at(5)=0.140; hygro_aer.at(6)=0.100; hygro_aer.at(7)=0.100; hygro_aer.at(8)=0.140; hygro_aer.at(9)=0.100;
dispersion_aer.at(0)=2.030; dispersion_aer.at(1)=1.3732; dispersion_aer.at(2)=1.900; dispersion_aer.at(3)=1.900; dispersion_aer.at(4)=1.900; dispersion_aer.at(5)=1.900; dispersion_aer.at(6)=2.240; dispersion_aer.at(7)=2.000; dispersion_aer.at(8)=2.240; dispersion_aer.at(9)=2.000;
num_to_mass_aer.at(0)=42097098109277080.; num_to_mass_aer.at(1)=8626504211623.; num_to_mass_aer.at(2)=3484000000000000.; num_to_mass_aer.at(3)=213800000000000.; num_to_mass_aer.at(4)=22050000000000.; num_to_mass_aer.at(5)=3165000000000.; num_to_mass_aer.at(6)=0.745645e+18; num_to_mass_aer.at(7)=0.167226e+20; num_to_mass_aer.at(8)=0.516216e+18; num_to_mass_aer.at(9)=0.167226e+20;
supersat.at(0)=0.02f;
supersat.at(1)=0.05f;
supersat.at(2)=0.1f;
supersat.at(3)=0.2f;
supersat.at(4)=0.5f;
supersat.at(5)=1.0f;
}
mskf_mphyi(cached != nullptr); }
__device__ void mskf_mphyi(bool cached=false) {
int arena_saved=arena.used;
int k;
int l;
int m;
int iaer;
double surften;
// WRF:241
xlf = latice;
// WRF:249
rhosn = 100.;
// WRF:250
rhoi = 500.;
// WRF:251
rhow = 1000.;
// WRF:257
ac = 3.e7;
// WRF:258
bc = 2.;
// WRF:261
as = 11.72;
// WRF:262
bs = 0.41;
// WRF:265
ai = 700.;
// WRF:266
bi = 1.;
// WRF:269
ar = 841.99667;
// WRF:270
br = 0.8;
// WRF:276
pi = 3.1415927;
// WRF:280
ci = ((rhoi*pi)/6.);
// WRF:281
di = 3.;
// WRF:285
cs = ((rhosn*pi)/6.);
// WRF:286
ds = 3.;
// WRF:290
cr = ((rhow*pi)/6.);
// WRF:291
dr = 3.;
// WRF:295
eii = 0.1;
// WRF:299
ecr = 1.0;
// WRF:304
dcs = 200.e-6;
// WRF:307
f14 = 100.0f;
// WRF:312
qsmall = 1.e-28;
// WRF:316
bimm = 100.;
// WRF:317
aimm = 0.66;
// WRF:323
rn_dst1 = 0.258e-6;
// WRF:324
rn_dst2 = 0.717e-6;
// WRF:325
rn_dst3 = 1.576e-6;
// WRF:326
rn_dst4 = 3.026e-6;
// WRF:330
rhosu = (85000./((rair*tmelt)));
// WRF:334
mi0 = ((((((4./3.)*pi)*rhoi)*(10.e-6))*(10.e-6))*(10.e-6));
// WRF:338
rin = 0.1e-6;
// WRF:341
t0 = 273.15;
// WRF:350
zero = 0.;
// WRF:351
third = (1.f/3.);
// WRF:352
sixth = (1.f/6.);
// WRF:353
sq2 = ma_sqrt(2.);
// WRF:354
pi = (4.*atan(1.0));
// WRF:355
sqpi = ma_sqrt(pi);
// WRF:357
surften = 0.076;
// WRF:358
aten = (((2.f*mwh2o)*surften)/(((r_universal*t0)*rhoh2o)));
// WRF:359
alogaten = ma_log(aten);
// WRF:360
alog2 = ma_log(2.);
// WRF:361
alog3 = ma_log(3.);
if(!cached) {
// WRF:362
for(int _slice=0; _slice<(psat)-(1)+1; ++_slice) super((1+_slice)) = (0.01f*supersat((1+_slice)));
// WRF:364
for(m=1; m<=naer_cu; m+=1) {
// WRF:366
alogsig(m) = ma_log(dispersion_aer(m));
// WRF:367
exp45logsig(m) = ma_exp(((4.5f*alogsig(m))*alogsig(m)));
// WRF:368
argfactor(m) = (2.f/(((3.f*ma_sqrt(2.f))*alogsig(m))));
// WRF:369
f1(m) = (0.5f*ma_exp(((2.5f*alogsig(m))*alogsig(m))));
// WRF:370
f2(m) = (1.f+(0.25f*alogsig(m)));
// WRF:371
amcubefactor(m) = (3./((((4.*pi)*exp45logsig(m))*density_aer(m))));
// WRF:372
smcritfactor(m) = ((2.*aten)*ma_sqrt((aten/((27.*ma_max(1.e-10,hygro_aer(m)))))));
// WRF:374
amcube(m) = (amcubefactor(m)/(num_to_mass_aer(m)));
// WRF:376
if((hygro_aer(m)>1.e-10f)) {
// WRF:377
smcrit(m) = (smcritfactor(m)/ma_sqrt(amcube(m)));
// WRF:378
} else {
// WRF:379
smcrit(m) = 100.f;
// WRF:380
}
// WRF:381
lnsm(m) = ma_log(smcrit(m));
// WRF:383
}
}
// WRF:385
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ void mskf_mphy(MaArray<double> su,MaArray<double> qu,MaArray<double> mu,MaArray<double> du,MaArray<double> cmel,MaArray<double> cmei,MaArray<double> zf,MaArray<double> pm,MaArray<double> te,MaArray<double> qe,MaArray<double> eps0,MaArray<int> jb,MaArray<int> jt,MaArray<int> jlcl,int msg,int il2g,double grav,double cp,double rd,MaArray<double> qc,MaArray<double> qi,MaArray<double> qr,MaArray<double> qni,MaArray<double> rprd,MaArray<double> wu,MaArray<double> eu,MaArray<double> nc,MaArray<double> ni,MaArray<double> nr,MaArray<double> ns,MaArray<double> dum2l,MaArray<double> sprd,MaArray<double> frz,MaArray<double> aer_mmr,double deltat,int pver,int pverp,MaArray<double> gamhat,MaArray<double> qsatzm,MaArray<double> wu_mskf_act,MaArray<double> qc_mskf_act,MaArray<double> qi_mskf_act,MaArray<double> effc,MaArray<double> effi,MaArray<double> effs) {
int arena_saved=arena.used;
const int naer_cu=10;
const int pcols=1;
MaArray<double> autolm=arena.make<double>(1,pcols,1,pver);
MaArray<double> accrlm=arena.make<double>(1,pcols,1,pver);
MaArray<double> bergnm=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtimm=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtctm=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhmlm=arena.make<double>(1,pcols,1,pver);
MaArray<double> hmpim=arena.make<double>(1,pcols,1,pver);
MaArray<double> accslm=arena.make<double>(1,pcols,1,pver);
MaArray<double> dlfm=arena.make<double>(1,pcols,1,pver);
MaArray<double> autoln=arena.make<double>(1,pcols,1,pver);
MaArray<double> accrln=arena.make<double>(1,pcols,1,pver);
MaArray<double> bergnn=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtimn=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtctn=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhmln=arena.make<double>(1,pcols,1,pver);
MaArray<double> accsln=arena.make<double>(1,pcols,1,pver);
MaArray<double> activn=arena.make<double>(1,pcols,1,pver);
MaArray<double> dlfn=arena.make<double>(1,pcols,1,pver);
MaArray<double> autoim=arena.make<double>(1,pcols,1,pver);
MaArray<double> accsim=arena.make<double>(1,pcols,1,pver);
MaArray<double> difm=arena.make<double>(1,pcols,1,pver);
MaArray<double> nuclin=arena.make<double>(1,pcols,1,pver);
MaArray<double> nuclim=arena.make<double>(1,pcols,1,pver);
MaArray<double> collrm=arena.make<double>(1,pcols,1,pver);
MaArray<double> collrn=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtcrm=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhtcrn=arena.make<double>(1,pcols,1,pver);
MaArray<double> autorn=arena.make<double>(1,pcols,1,pver);
MaArray<double> aggrn=arena.make<double>(1,pcols,1,pver);
MaArray<double> aggsn=arena.make<double>(1,pcols,1,pver);
MaArray<double> autoin=arena.make<double>(1,pcols,1,pver);
MaArray<double> accsin=arena.make<double>(1,pcols,1,pver);
MaArray<double> hmpin=arena.make<double>(1,pcols,1,pver);
MaArray<double> difn=arena.make<double>(1,pcols,1,pver);
MaArray<double> trspcm=arena.make<double>(1,pcols,1,pver);
MaArray<double> trspcn=arena.make<double>(1,pcols,1,pver);
MaArray<double> trspim=arena.make<double>(1,pcols,1,pver);
MaArray<double> trspin=arena.make<double>(1,pcols,1,pver);
MaArray<double> ncadj=arena.make<double>(1,pcols,1,pver);
MaArray<double> niadj=arena.make<double>(1,pcols,1,pver);
MaArray<double> qcadj=arena.make<double>(1,pcols,1,pver);
MaArray<double> qiadj=arena.make<double>(1,pcols,1,pver);
MaArray<double> nimey=arena.make<double>(1,pcols,1,pver);
MaArray<double> nihf=arena.make<double>(1,pcols,1,pver);
MaArray<double> nidep=arena.make<double>(1,pcols,1,pver);
MaArray<double> niimm=arena.make<double>(1,pcols,1,pver);
double omsm;
double dum;
double arg;
double dum1;
double dum2;
MaArray<double> q=arena.make<double>(1,pcols,1,pver);
MaArray<double> t=arena.make<double>(1,pcols,1,pver);
MaArray<double> rho=arena.make<double>(1,pcols,1,pver);
MaArray<double> dz=arena.make<double>(1,pcols,1,pver);
MaArray<double> qcic=arena.make<double>(1,pcols,1,pver);
MaArray<double> qiic=arena.make<double>(1,pcols,1,pver);
double tot_qc_qi;
MaArray<double> qniic=arena.make<double>(1,pcols,1,pver);
MaArray<double> qric=arena.make<double>(1,pcols,1,pver);
MaArray<double> ncic=arena.make<double>(1,pcols,1,pver);
MaArray<double> niic=arena.make<double>(1,pcols,1,pver);
MaArray<double> nsic=arena.make<double>(1,pcols,1,pver);
MaArray<double> nric=arena.make<double>(1,pcols,1,pver);
MaArray<double> lami=arena.make<double>(1,pver);
MaArray<double> n0i=arena.make<double>(1,pver);
MaArray<double> lamc=arena.make<double>(1,pver);
MaArray<double> n0c=arena.make<double>(1,pver);
MaArray<double> lams=arena.make<double>(1,pver);
MaArray<double> n0s=arena.make<double>(1,pver);
MaArray<double> lamr=arena.make<double>(1,pver);
MaArray<double> n0r=arena.make<double>(1,pver);
MaArray<double> cdist1=arena.make<double>(1,pver);
MaArray<double> pgam=arena.make<double>(1,pver);
double lammax;
double lammin;
MaArray<double> mnuccc=arena.make<double>(1,pver);
MaArray<double> nnuccc=arena.make<double>(1,pver);
MaArray<double> mnucct=arena.make<double>(1,pver);
MaArray<double> nnucct=arena.make<double>(1,pver);
MaArray<double> msacwi=arena.make<double>(1,pver);
MaArray<double> nsacwi=arena.make<double>(1,pver);
MaArray<double> prf=arena.make<double>(1,pver);
MaArray<double> psf=arena.make<double>(1,pver);
MaArray<double> pnrf=arena.make<double>(1,pver);
MaArray<double> pnsf=arena.make<double>(1,pver);
MaArray<double> prc=arena.make<double>(1,pver);
MaArray<double> nprc=arena.make<double>(1,pver);
MaArray<double> nprc1=arena.make<double>(1,pver);
MaArray<double> nsagg=arena.make<double>(1,pver);
double dc0;
double ds0;
double eci;
MaArray<double> dv=arena.make<double>(1,pcols,1,pver);
MaArray<double> mua=arena.make<double>(1,pcols,1,pver);
MaArray<double> psacws=arena.make<double>(1,pver);
MaArray<double> npsacws=arena.make<double>(1,pver);
MaArray<double> pracs=arena.make<double>(1,pver);
MaArray<double> npracs=arena.make<double>(1,pver);
MaArray<double> mnuccr=arena.make<double>(1,pver);
MaArray<double> nnuccr=arena.make<double>(1,pver);
MaArray<double> pra=arena.make<double>(1,pver);
MaArray<double> npra=arena.make<double>(1,pver);
MaArray<double> nragg=arena.make<double>(1,pver);
MaArray<double> prci=arena.make<double>(1,pver);
MaArray<double> nprci=arena.make<double>(1,pver);
MaArray<double> prai=arena.make<double>(1,pver);
MaArray<double> nprai=arena.make<double>(1,pver);
MaArray<double> prb=arena.make<double>(1,pver);
MaArray<double> nprb=arena.make<double>(1,pver);
MaArray<double> arn=arena.make<double>(1,pcols,1,pver);
MaArray<double> asn=arena.make<double>(1,pcols,1,pver);
MaArray<double> acn=arena.make<double>(1,pcols,1,pver);
MaArray<double> ain=arena.make<double>(1,pcols,1,pver);
MaArray<double> uns=arena.make<double>(1,pver);
MaArray<double> ums=arena.make<double>(1,pver);
MaArray<double> unr=arena.make<double>(1,pver);
MaArray<double> umr=arena.make<double>(1,pver);
double qce;
double qie;
double nce;
double nie;
double qre;
double nre;
double qnie;
double nse;
double ratio;
MaArray<double> qctend=arena.make<double>(1,pcols,1,pver);
MaArray<double> qitend=arena.make<double>(1,pcols,1,pver);
MaArray<double> nctend=arena.make<double>(1,pcols,1,pver);
MaArray<double> nitend=arena.make<double>(1,pcols,1,pver);
MaArray<double> qnitend=arena.make<double>(1,pcols,1,pver);
MaArray<double> nstend=arena.make<double>(1,pcols,1,pver);
MaArray<double> qrtend=arena.make<double>(1,pcols,1,pver);
MaArray<double> nrtend=arena.make<double>(1,pcols,1,pver);
double bergtsf;
double plevap;
MaArray<double> naermod=arena.make<double>(1,naer_cu);
MaArray<double> naer2=arena.make<double>(1,pcols,1,pver,1,naer_cu);
MaArray<double> naer2h=arena.make<double>(1,pcols,1,pver,1,naer_cu);
MaArray<double> maerosol=arena.make<double>(1,1,1,naer_cu);
MaArray<double> naer=arena.make<double>(1,pcols);
MaArray<double> npccn=arena.make<double>(1,pver);
double ncmax;
double mtimec;
MaArray<double> dum2i=arena.make<double>(1,pcols,1,pver);
MaArray<double> qs=arena.make<double>(1,pcols,1,pver);
MaArray<double> es=arena.make<double>(1,pcols,1,pver);
MaArray<double> relhum=arena.make<double>(1,pcols,1,pver);
MaArray<double> esi=arena.make<double>(1,pcols,1,pver);
MaArray<double> nnuccd=arena.make<double>(1,pver);
MaArray<double> mnuccd=arena.make<double>(1,pver);
double nimax;
double mtime;
int i;
int k;
int nstep;
int n;
int l;
int ii;
int kk;
int m;
int iter;
int it;
MaArray<int> ltrue=arena.make<int>(1,pcols);
double tcnt;
double viscosity;
double mfp;
double slip1;
double slip2;
double slip3;
double slip4;
double dfaer1;
double dfaer2;
double dfaer3;
double dfaer4;
double nacon1;
double nacon2;
double nacon3;
double nacon4;
MaArray<double> ttend=arena.make<double>(1,pver);
double naimm;
MaArray<double> ntaer=arena.make<double>(1,pcols,1,pver);
MaArray<double> ntaerh=arena.make<double>(1,pcols,1,pver);
double ni_secp;
MaArray<double> th=arena.make<double>(1,pcols,1,pver);
MaArray<double> qh=arena.make<double>(1,pcols,1,pver);
MaArray<double> zkine=arena.make<double>(1,pcols,1,pver);
MaArray<double> zbuo=arena.make<double>(1,pcols,1,pver);
double zfacbuo;
double cwdrag;
double cwifrac;
double retv;
double zbuoc;
double zbc;
double zbe;
double zdkbuo;
double zdken;
MaArray<double> arcf=arena.make<double>(1,pcols,1,pver);
MaArray<double> p=arena.make<double>(1,pcols,1,pver);
MaArray<double> ph=arena.make<double>(1,pcols,1,pver);
MaArray<double> rhoh=arena.make<double>(1,pcols,1,pver);
MaArray<double> rhom=arena.make<double>(1,pcols,1,pver);
MaArray<double> tu=arena.make<double>(1,pcols,1,pver);
MaArray<double> fhmrm=arena.make<double>(1,pcols,1,pver);
double ncorg;
double niorg;
double qcorg;
double qiorg;
MaArray<int> kqi=arena.make<int>(1,pcols);
MaArray<int> kqc=arena.make<int>(1,pcols);
MaArray<bool> lcbase=arena.make<bool>(1,pcols);
MaArray<bool> libase=arena.make<bool>(1,pcols);
// WRF:708
omsm = 0.99999;
// WRF:709
zfacbuo = (0.5/((1.+0.5)));
// WRF:710
cwdrag = (1.875*0.506);
// WRF:711
cwifrac = 0.5;
// WRF:712
retv = 0.608;
// WRF:713
bergtsf = 1800.;
// WRF:716
for(i=1; i<=il2g; i+=1) {
// WRF:717
for(k=1; k<=pver; k+=1) {
// WRF:718
q(i,k) = qu(i,k);
// WRF:719
tu(i,k) = (su(i,k)-((grav/cp)*zf(i,k)));
// WRF:720
t(i,k) = (su(i,k)-((grav/cp)*zf(i,k)));
// WRF:721
p(i,k) = (100.*pm(i,k));
// WRF:722
wu(i,k) = 0.;
// WRF:723
zkine(i,k) = 0.;
// WRF:724
arcf(i,k) = 0.;
// WRF:725
zbuo(i,k) = 0.;
// WRF:726
nc(i,k) = 0.;
// WRF:727
ni(i,k) = 0.;
// WRF:728
qc(i,k) = 0.;
// WRF:729
qi(i,k) = 0.;
// WRF:730
qcic(i,k) = 0.;
// WRF:731
qiic(i,k) = 0.;
// WRF:732
ncic(i,k) = nc(i,k);
// WRF:733
niic(i,k) = ni(i,k);
// WRF:734
qr(i,k) = 0.;
// WRF:735
qni(i,k) = 0.;
// WRF:736
nr(i,k) = 0.;
// WRF:737
ns(i,k) = 0.;
// WRF:738
qric(i,k) = qr(i,k);
// WRF:739
qniic(i,k) = qni(i,k);
// WRF:740
nric(i,k) = nr(i,k);
// WRF:741
nsic(i,k) = ns(i,k);
// WRF:742
nimey(i,k) = 0.;
// WRF:743
nihf(i,k) = 0.;
// WRF:744
nidep(i,k) = 0.;
// WRF:745
niimm(i,k) = 0.;
// WRF:747
autolm(i,k) = 0.;
// WRF:748
accrlm(i,k) = 0.;
// WRF:749
bergnm(i,k) = 0.;
// WRF:750
fhtimm(i,k) = 0.;
// WRF:751
fhtctm(i,k) = 0.;
// WRF:752
fhmlm(i,k) = 0.;
// WRF:753
hmpim(i,k) = 0.;
// WRF:754
accslm(i,k) = 0.;
// WRF:755
dlfm(i,k) = 0.;
// WRF:756
collrm(i,k) = 0.;
// WRF:757
collrn(i,k) = 0.;
// WRF:758
fhtcrm(i,k) = 0.;
// WRF:759
fhtcrn(i,k) = 0.;
// WRF:760
autorn(i,k) = 0.;
// WRF:761
aggrn(i,k) = 0.;
// WRF:762
aggsn(i,k) = 0.;
// WRF:764
autoln(i,k) = 0.;
// WRF:766
accrln(i,k) = 0.;
// WRF:767
bergnn(i,k) = 0.;
// WRF:768
fhtimn(i,k) = 0.;
// WRF:769
fhtctn(i,k) = 0.;
// WRF:770
fhmln(i,k) = 0.;
// WRF:771
accsln(i,k) = 0.;
// WRF:772
activn(i,k) = 0.;
// WRF:773
dlfn(i,k) = 0.;
// WRF:774
ncadj(i,k) = 0.;
// WRF:775
qcadj(i,k) = 0.;
// WRF:777
autoim(i,k) = 0.;
// WRF:778
accsim(i,k) = 0.;
// WRF:779
difm(i,k) = 0.;
// WRF:780
nuclin(i,k) = 0.;
// WRF:781
nuclim(i,k) = 0.;
// WRF:782
autoin(i,k) = 0.;
// WRF:783
accsin(i,k) = 0.;
// WRF:784
hmpin(i,k) = 0.;
// WRF:785
difn(i,k) = 0.;
// WRF:786
niadj(i,k) = 0.;
// WRF:787
qiadj(i,k) = 0.;
// WRF:789
trspcm(i,k) = 0.;
// WRF:790
trspcn(i,k) = 0.;
// WRF:791
trspim(i,k) = 0.;
// WRF:792
trspin(i,k) = 0.;
// WRF:794
effc(i,k) = 0.;
// WRF:795
effi(i,k) = 0.;
// WRF:796
effs(i,k) = 0.;
// WRF:798
fhmrm(i,k) = 0.;
// WRF:799
}
// WRF:800
}
// WRF:804
for(k=1; k<=pver; k+=1) {
// WRF:805
for(i=1; i<=il2g; i+=1) {
// WRF:807
if((k==1)) {
// WRF:808
rhoh(i,k) = (p(i,k)/((t(i,k)*rd)));
// WRF:809
rhom(i,k) = (p(i,k)/((t(i,k)*rd)));
// WRF:810
th(i,k) = te(i,k);
// WRF:811
qh(i,k) = qe(i,k);
// WRF:812
dz(i,k) = (zf(i,k)-zf(i,(k+1)));
// WRF:813
ph(i,k) = p(i,k);
// WRF:814
} else {
// WRF:816
rhoh(i,k) = ((0.5*((p(i,k)+p(i,(k-1)))))/((t(i,k)*rd)));
// WRF:817
if((k==pver)) {
// WRF:818
rhom(i,k) = (p(i,k)/((rd*t(i,k))));
// WRF:819
} else {
// WRF:820
rhom(i,k) = ((2.0*p(i,k))/((rd*((t(i,k)+t(i,(k+1)))))));
// WRF:821
}
// WRF:822
th(i,k) = (0.5*((te(i,k)+te(i,(k-1)))));
// WRF:823
qh(i,k) = (0.5*((qe(i,k)+qe(i,(k-1)))));
// WRF:824
dz(i,k) = (zf(i,(k-1))-zf(i,k));
// WRF:825
ph(i,k) = (0.5*((p(i,k)+p(i,(k-1)))));
// WRF:826
}
// WRF:828
dv(i,k) = ((8.794e-5*ma_pow(t(i,k),1.81))/ph(i,k));
// WRF:829
mua(i,k) = ((1.496e-6*ma_pow(t(i,k),1.5))/((t(i,k)+120.)));
// WRF:832
rho(i,k) = rhoh(i,k);
// WRF:838
arn(i,k) = (ar*ma_pow(((rhosu/rho(i,k))),0.54f));
// WRF:839
asn(i,k) = (as*ma_pow(((rhosu/rho(i,k))),0.54f));
// WRF:840
acn(i,k) = (ac*ma_pow(((rhosu/rho(i,k))),0.54f));
// WRF:841
ain(i,k) = (ai*ma_pow(((rhosu/rho(i,k))),0.54f));
// WRF:843
}
// WRF:844
}
// WRF:847
for(k=1; k<=pver; k+=1) {
// WRF:848
for(i=1; i<=il2g; i+=1) {
// WRF:849
for(int _slice=0; _slice<(naer_cu)-(1)+1; ++_slice) naer2(i,k,(1+_slice)) = 0.;
// WRF:850
for(int _slice=0; _slice<(naer_cu)-(1)+1; ++_slice) naer2h(i,k,(1+_slice)) = 0.;
// WRF:851
dum2l(i,k) = 0.;
// WRF:852
dum2i(i,k) = 0.;
// WRF:853
}
// WRF:854
}
// WRF:856
for(k=1; k<=pver; k+=1) {
// WRF:857
for(i=1; i<=il2g; i+=1) {
// WRF:858
ntaer(i,k) = 0.0;
// WRF:859
ntaerh(i,k) = 0.0;
// WRF:860
for(m=1; m<=naer_cu; m+=1) {
// WRF:862
maerosol(1,m) = (aer_mmr(i,k,m)*rhom(i,k));
// WRF:871
if((m==idxsul)) {
// WRF:872
naer2(i,k,m) = (5.64259e13*ma_pow(maerosol(1,m),0.58f));
// WRF:873
} else {
// WRF:874
naer2(i,k,m) = (maerosol(1,m)*num_to_mass_aer(m));
// WRF:875
}
// WRF:876
ntaer(i,k) = (ntaer(i,k)+naer2(i,k,m));
// WRF:877
}
// WRF:878
}
// WRF:879
}
// WRF:881
for(i=1; i<=il2g; i+=1) {
// WRF:882
ltrue(i) = 0;
// WRF:883
for(k=1; k<=pver; k+=1) {
// WRF:884
if(((((qc(i,k)>=qsmall)||(qi(i,k)>=qsmall))||(cmel(i,k)>=qsmall))||(cmei(i,k)>=qsmall))) { ltrue(i) = 1; }
// WRF:886
}
// WRF:887
}
// WRF:890
for(i=1; i<=il2g; i+=1) {
// WRF:891
if((ltrue(i)==0)) {
// WRF:892
for(k=1; k<=pver; k+=1) {
// WRF:893
qctend(i,k) = 0.;
// WRF:894
qitend(i,k) = 0.;
// WRF:895
qnitend(i,k) = 0.;
// WRF:896
qrtend(i,k) = 0.;
// WRF:897
nctend(i,k) = 0.;
// WRF:898
nitend(i,k) = 0.;
// WRF:899
nrtend(i,k) = 0.;
// WRF:900
nstend(i,k) = 0.;
// WRF:901
qniic(i,k) = 0.;
// WRF:902
qric(i,k) = 0.;
// WRF:903
nsic(i,k) = 0.;
// WRF:904
nric(i,k) = 0.;
// WRF:905
qni(i,k) = 0.;
// WRF:906
qr(i,k) = 0.;
// WRF:907
ns(i,k) = 0.;
// WRF:908
nr(i,k) = 0.;
// WRF:909
qc(i,k) = 0.;
// WRF:910
qi(i,k) = 0.;
// WRF:911
nc(i,k) = 0.;
// WRF:912
ni(i,k) = 0.;
// WRF:913
rprd(i,k) = 0.;
// WRF:914
sprd(i,k) = 0.;
// WRF:915
frz(i,k) = 0.;
// WRF:916
}
// WRF:917
goto ma_label_300;
// WRF:918
}
// WRF:920
kqc(i) = 1;
// WRF:921
kqi(i) = 1;
// WRF:922
lcbase(i) = true;
// WRF:923
libase(i) = true;
// WRF:927
iter = 2;
// WRF:933
for(it=1; it<=iter; it+=1) {
// WRF:935
for(k=1; k<=pver; k+=1) {
// WRF:936
qctend(i,k) = 0.;
// WRF:937
qitend(i,k) = 0.;
// WRF:938
qnitend(i,k) = 0.;
// WRF:939
qrtend(i,k) = 0.;
// WRF:940
nctend(i,k) = 0.;
// WRF:941
nitend(i,k) = 0.;
// WRF:942
nrtend(i,k) = 0.;
// WRF:943
nstend(i,k) = 0.;
// WRF:944
rprd(i,k) = 0.;
// WRF:945
sprd(i,k) = 0.;
// WRF:946
frz(i,k) = 0.;
// WRF:947
qniic(i,k) = 0.;
// WRF:948
qric(i,k) = 0.;
// WRF:949
nsic(i,k) = 0.;
// WRF:950
nric(i,k) = 0.;
// WRF:951
qiic(i,k) = 0.;
// WRF:952
qcic(i,k) = 0.;
// WRF:953
niic(i,k) = 0.;
// WRF:954
ncic(i,k) = 0.;
// WRF:956
accrlm(i,k) = 0.;
// WRF:957
bergnm(i,k) = 0.;
// WRF:958
fhtimm(i,k) = 0.;
// WRF:959
fhtctm(i,k) = 0.;
// WRF:960
fhmlm(i,k) = 0.;
// WRF:961
hmpim(i,k) = 0.;
// WRF:962
accslm(i,k) = 0.;
// WRF:963
dlfm(i,k) = 0.;
// WRF:965
autoln(i,k) = 0.;
// WRF:966
accrln(i,k) = 0.;
// WRF:967
bergnn(i,k) = 0.;
// WRF:968
fhtimn(i,k) = 0.;
// WRF:969
fhtctn(i,k) = 0.;
// WRF:970
fhmln(i,k) = 0.;
// WRF:971
accsln(i,k) = 0.;
// WRF:972
activn(i,k) = 0.;
// WRF:973
dlfn(i,k) = 0.;
// WRF:974
ncadj(i,k) = 0.;
// WRF:975
qcadj(i,k) = 0.;
// WRF:977
autoim(i,k) = 0.;
// WRF:978
accsim(i,k) = 0.;
// WRF:979
difm(i,k) = 0.;
// WRF:981
nuclin(i,k) = 0.;
// WRF:982
nuclim(i,k) = 0.;
// WRF:983
autoin(i,k) = 0.;
// WRF:984
accsin(i,k) = 0.;
// WRF:985
hmpin(i,k) = 0.;
// WRF:986
difn(i,k) = 0.;
// WRF:987
niadj(i,k) = 0.;
// WRF:988
qiadj(i,k) = 0.;
// WRF:990
trspcm(i,k) = 0.;
// WRF:991
trspcn(i,k) = 0.;
// WRF:992
trspim(i,k) = 0.;
// WRF:993
trspin(i,k) = 0.;
// WRF:995
effc(i,k) = 0.;
// WRF:996
effi(i,k) = 0.;
// WRF:997
effs(i,k) = 0.;
// WRF:999
fhmrm(i,k) = 0.;
// WRF:1002
}
// WRF:1007
for(k=pver; k>=(msg+2); k+=(-1)) {
// WRF:1010
if((((((k>jt(i))&&(k<=jb(i)))&&(eps0(i)>0.))&&(mu(i,k)>0.))&&(mu(i,(k-1))>0.))) {
// WRF:1014
ums(k) = 0.;
// WRF:1015
uns(k) = 0.;
// WRF:1016
umr(k) = 0.;
// WRF:1017
unr(k) = 0.;
// WRF:1018
prf(k) = 0.;
// WRF:1019
pnrf(k) = 0.;
// WRF:1020
psf(k) = 0.;
// WRF:1021
pnsf(k) = 0.;
// WRF:1022
ttend(k) = 0.;
// WRF:1023
nnuccd(k) = 0.;
// WRF:1024
npccn(k) = 0.;
// WRF:1036
nc(i,k) = ma_max(nc(i,k),0.);
// WRF:1037
ni(i,k) = ma_max(ni(i,k),0.);
// WRF:1038
if((it==1)) {
// WRF:1039
qcic(i,k) = qc(i,k);
// WRF:1040
qiic(i,k) = qi(i,k);
// WRF:1046
ncic(i,k) = nc(i,k);
// WRF:1047
niic(i,k) = ni(i,k);
// WRF:1048
qniic(i,k) = qni(i,k);
// WRF:1049
qric(i,k) = qr(i,k);
// WRF:1050
nsic(i,k) = ns(i,k);
// WRF:1051
nric(i,k) = nr(i,k);
// WRF:1052
} else {
// WRF:1053
if((k<=kqc(i))) {
// WRF:1054
qcic(i,k) = qc(i,k);
// WRF:1055
ncic(i,k) = nc(i,k);
// WRF:1056
if((k==kqc(i))) {
// WRF:1057
qcic(i,k) = qc(i,(k-1));
// WRF:1058
ncic(i,k) = nc(i,(k-1));
// WRF:1059
}
// WRF:1061
for(kk=k; kk>=(jt(i)+2); kk+=(-1)) {
// WRF:1062
qric(i,k) = (qr(i,k)+ma_max(0.,(qr(i,(kk-1))-qr(i,(kk-2)))));
// WRF:1063
if((qr(i,(kk-1))>0.)) { nric(i,k) = (nr(i,k)+((ma_max(0.,(qr(i,(kk-1))-qr(i,(kk-2))))/qr(i,(kk-1)))*nr(i,(kk-1)))); }
// WRF:1065
}
// WRF:1066
}
// WRF:1067
if((k<=kqi(i))) {
// WRF:1068
qiic(i,k) = qi(i,k);
// WRF:1069
niic(i,k) = ni(i,k);
// WRF:1070
if((k==kqi(i))) {
// WRF:1071
qiic(i,k) = qi(i,(k-1));
// WRF:1072
niic(i,k) = ni(i,(k-1));
// WRF:1073
}
// WRF:1075
for(kk=k; kk>=(jt(i)+2); kk+=(-1)) {
// WRF:1076
qniic(i,k) = (qni(i,k)+ma_max(0.,(qni(i,(kk-1))-qni(i,(kk-2)))));
// WRF:1077
if((qni(i,(kk-1))>0.)) { nsic(i,k) = (ns(i,k)+((ma_max(0.,(qni(i,(kk-1))-qni(i,(kk-2))))/qni(i,(kk-1)))*ns(i,(kk-1)))); }
// WRF:1079
}
// WRF:1080
}
// WRF:1081
}
// WRF:1083
if((it==1)) {
// WRF:1086
}
// WRF:1092
if((((cmel(i,(k-1))>qsmall)&&lcbase(i))&&(it==1))) {
// WRF:1093
kqc(i) = k;
// WRF:1094
lcbase(i) = false;
// WRF:1095
qcic(i,k) = ((dz(i,k)*cmel(i,(k-1)))/((mu(i,(k-1))+(dz(i,k)*du(i,(k-1))))));
// WRF:1096
if((qcic(i,k)==0.0f)) {
// WRF:1097
if((it==1)) {
// WRF:1100
}
// WRF:1101
}
// WRF:1102
ncic(i,k) = (qcic(i,k)/(((((4./3.)*pi)*ma_powi<3>(11.e-6))*rhow)));
// WRF:1103
}
// WRF:1106
if((((qiic(i,k)>qsmall)&&libase(i))&&(it==1))) {
// WRF:1107
kqi(i) = k;
// WRF:1108
libase(i) = false;
// WRF:1109
} else if((((((cmei(i,(k-1))>qsmall)&&(cmei(i,k)<qsmall))&&(k<jb(i)))&&libase(i))&&(it==1))) {
// WRF:1111
kqi(i) = k;
// WRF:1112
libase(i) = false;
// WRF:1113
qiic(i,k) = ((dz(i,k)*cmei(i,(k-1)))/((mu(i,(k-1))+(dz(i,k)*du(i,(k-1))))));
// WRF:1114
niic(i,k) = (qiic(i,k)/(((((4./3.)*pi)*ma_powi<3>(25.e-6))*rhoi)));
// WRF:1115
}
// WRF:1122
if((qiic(i,k)>=qsmall)) {
// WRF:1124
niic(i,k) = ma_min(niic(i,k),(qiic(i,k)*1.e20));
// WRF:1125
lami(k) = ma_pow(((((mskf_gamma((1.+di))*ci)*niic(i,k))/qiic(i,k))),((1./di)));
// WRF:1127
n0i(k) = (niic(i,k)*lami(k));
// WRF:1129
lammax = (1./10.e-6);
// WRF:1130
lammin = (1./((2.*dcs)));
// WRF:1132
if((lami(k)<lammin)) {
// WRF:1133
lami(k) = lammin;
// WRF:1134
n0i(k) = ((ma_pow(lami(k),((di+1.)))*qiic(i,k))/((ci*mskf_gamma((1.+di)))));
// WRF:1135
niic(i,k) = (n0i(k)/lami(k));
// WRF:1136
} else if((lami(k)>lammax)) {
// WRF:1137
lami(k) = lammax;
// WRF:1138
n0i(k) = ((ma_pow(lami(k),((di+1.)))*qiic(i,k))/((ci*mskf_gamma((1.+di)))));
// WRF:1139
niic(i,k) = (n0i(k)/lami(k));
// WRF:1140
}
// WRF:1141
} else {
// WRF:1142
lami(k) = 0.;
// WRF:1143
n0i(k) = 0.;
// WRF:1144
}
// WRF:1147
if((qcic(i,k)>=qsmall)) {
// WRF:1150
ncic(i,k) = ma_min(ncic(i,k),(qcic(i,k)*1.e20));
// WRF:1154
pgam(k) = ((0.0005714*(((ncic(i,k)/1.e6)*rho(i,k))))+0.2714);
// WRF:1155
pgam(k) = ((1./(ma_powi<2>(pgam(k))))-1.);
// WRF:1156
pgam(k) = ma_max(pgam(k),2.);
// WRF:1157
pgam(k) = ma_min(pgam(k),15.);
// WRF:1160
lamc(k) = ma_pow(((((((pi/6.)*rhow)*ncic(i,k))*mskf_gamma((pgam(k)+4.)))/((qcic(i,k)*mskf_gamma((pgam(k)+1.)))))),((1./3.)));
// WRF:1164
lammin = (((pgam(k)+1.))/50.e-6);
// WRF:1165
lammax = (((pgam(k)+1.))/2.e-6);
// WRF:1167
if((lamc(k)<lammin)) {
// WRF:1168
lamc(k) = lammin;
// WRF:1169
ncic(i,k) = ((((6.*ma_powi<3>(lamc(k)))*qcic(i,k))*mskf_gamma((pgam(k)+1.)))/(((pi*rhow)*mskf_gamma((pgam(k)+4.)))));
// WRF:1172
} else if((lamc(k)>lammax)) {
// WRF:1173
lamc(k) = lammax;
// WRF:1174
ncic(i,k) = ((((6.*ma_powi<3>(lamc(k)))*qcic(i,k))*mskf_gamma((pgam(k)+1.)))/(((pi*rhow)*mskf_gamma((pgam(k)+4.)))));
// WRF:1177
}
// WRF:1181
cdist1(k) = (ncic(i,k)/mskf_gamma((pgam(k)+1.)));
// WRF:1182
} else {
// WRF:1183
lamc(k) = 0.;
// WRF:1184
cdist1(k) = 0.;
// WRF:1185
}
// WRF:1187
if((kqc(i)==k)) {
// WRF:1188
qc(i,k) = 0.;
// WRF:1189
nc(i,k) = 0.;
// WRF:1190
}
// WRF:1192
if((kqi(i)==k)) {
// WRF:1193
qi(i,k) = 0.;
// WRF:1194
ni(i,k) = 0.;
// WRF:1195
}
// WRF:1206
if((qcic(i,k)>=1.e-8)) {
// WRF:1216
prc(k) = ((7.98e10*ma_pow(qcic(i,k),4.22))*ma_pow((((ncic(i,k)/1.e6)*rho(i,k))),((-3.01))));
// WRF:1219
nprc(k) = (prc(k)/(((((4./3.)*pi)*rhow)*ma_powi<3>((25.e-6)))));
// WRF:1220
nprc1(k) = (prc(k)/((qcic(i,k)/ncic(i,k))));
// WRF:1221
} else {
// WRF:1222
prc(k) = 0.;
// WRF:1223
nprc(k) = 0.;
// WRF:1224
nprc1(k) = 0.;
// WRF:1225
}
// WRF:1230
if(((k==kqc(i))&&(it==1))) {
// WRF:1231
qric(i,k) = ((prc(k)*dz(i,k))/0.55);
// WRF:1232
nric(i,k) = ((nprc(k)*dz(i,k))/0.55);
// WRF:1233
qr(i,k) = 0.0;
// WRF:1234
nr(i,k) = 0.0;
// WRF:1235
}
// WRF:1247
if(((t(i,k)<=273.15)&&(qiic(i,k)>=qsmall))) {
// WRF:1252
nprci(k) = ((n0i(k)/((lami(k)*f14)))*ma_exp(((-lami(k))*dcs)));
// WRF:1254
prci(k) = (((((pi*rhoi)*n0i(k))/((6.*f14)))*(((((ma_powi<3>(dcs)/lami(k))+((3.*ma_powi<2>(dcs))/ma_powi<2>(lami(k))))+((6.*dcs)/ma_powi<3>(lami(k))))+(6./ma_powi<4>(lami(k))))))*ma_exp(((-lami(k))*dcs)));
// WRF:1257
} else {
// WRF:1258
prci(k) = 0.;
// WRF:1259
nprci(k) = 0.;
// WRF:1260
}
// WRF:1265
if(((k==kqi(i))&&(it==1))) {
// WRF:1266
qniic(i,k) = ((prci(k)*dz(i,k))*0.25);
// WRF:1267
nsic(i,k) = ((nprci(k)*dz(i,k))*0.25);
// WRF:1268
qni(i,k) = 0.0;
// WRF:1269
ns(i,k) = 0.0;
// WRF:1270
}
// WRF:1272
if((qniic(i,k)<qsmall)) {
// WRF:1273
qniic(i,k) = 0.;
// WRF:1274
nsic(i,k) = 0.;
// WRF:1275
}
// WRF:1276
if((qric(i,k)<qsmall)) {
// WRF:1277
qric(i,k) = 0.;
// WRF:1278
nric(i,k) = 0.;
// WRF:1279
}
// WRF:1283
nric(i,k) = ma_max(nric(i,k),0.);
// WRF:1284
nsic(i,k) = ma_max(nsic(i,k),0.);
// WRF:1291
if((qric(i,k)>=qsmall)) {
// WRF:1292
lamr(k) = ma_pow(((((pi*rhow)*nric(i,k))/qric(i,k))),((1./3.)));
// WRF:1293
n0r(k) = (nric(i,k)*lamr(k));
// WRF:1295
lammax = (1./20.e-6);
// WRF:1296
lammin = (1./500.e-6);
// WRF:1298
if((lamr(k)<lammin)) {
// WRF:1299
lamr(k) = lammin;
// WRF:1300
n0r(k) = ((ma_powi<4>(lamr(k))*qric(i,k))/((pi*rhow)));
// WRF:1301
nric(i,k) = (n0r(k)/lamr(k));
// WRF:1302
} else if((lamr(k)>lammax)) {
// WRF:1303
lamr(k) = lammax;
// WRF:1304
n0r(k) = ((ma_powi<4>(lamr(k))*qric(i,k))/((pi*rhow)));
// WRF:1305
nric(i,k) = (n0r(k)/lamr(k));
// WRF:1306
}
// WRF:1310
unr(k) = ma_min(((arn(i,k)*mskf_gamma((1.+br)))/ma_pow(lamr(k),br)),10.);
// WRF:1311
umr(k) = ma_min(((arn(i,k)*mskf_gamma((4.+br)))/((6.*ma_pow(lamr(k),br)))),10.);
// WRF:1312
} else {
// WRF:1313
lamr(k) = 0.;
// WRF:1314
n0r(k) = 0.;
// WRF:1315
umr(k) = 0.;
// WRF:1316
unr(k) = 0.;
// WRF:1317
}
// WRF:1320
if((qniic(i,k)>=qsmall)) {
// WRF:1321
lams(k) = ma_pow(((((mskf_gamma((1.+ds))*cs)*nsic(i,k))/qniic(i,k))),((1./ds)));
// WRF:1323
n0s(k) = (nsic(i,k)*lams(k));
// WRF:1326
lammax = (1./10.e-6);
// WRF:1327
lammin = (1./2000.e-6);
// WRF:1329
if((lams(k)<lammin)) {
// WRF:1330
lams(k) = lammin;
// WRF:1331
n0s(k) = ((ma_pow(lams(k),((ds+1.)))*qniic(i,k))/((cs*mskf_gamma((1.+ds)))));
// WRF:1332
nsic(i,k) = (n0s(k)/lams(k));
// WRF:1333
} else if((lams(k)>lammax)) {
// WRF:1334
lams(k) = lammax;
// WRF:1335
n0s(k) = ((ma_pow(lams(k),((ds+1.)))*qniic(i,k))/((cs*mskf_gamma((1.+ds)))));
// WRF:1336
nsic(i,k) = (n0s(k)/lams(k));
// WRF:1337
}
// WRF:1340
ums(k) = ma_min(((asn(i,k)*mskf_gamma((4.+bs)))/((6.*ma_pow(lams(k),bs)))),3.6);
// WRF:1341
uns(k) = ma_min(((asn(i,k)*mskf_gamma((1.+bs)))/ma_pow(lams(k),bs)),3.6);
// WRF:1342
} else {
// WRF:1343
lams(k) = 0.;
// WRF:1344
n0s(k) = 0.;
// WRF:1345
ums(k) = 0.;
// WRF:1346
uns(k) = 0.;
// WRF:1347
}
// WRF:1354
if(((qniic(i,k)>=qsmall)&&(t(i,k)<=273.15))) {
// WRF:1355
nsagg(k) = (((((((((-1108.)*asn(i,k))*eii)*ma_pow(pi,((((1.-bs))/3.))))*ma_pow(rhosn,(((((-2.)-bs))/3.))))*ma_pow(rho(i,k),((((2.+bs))/3.))))*ma_pow(qniic(i,k),((((2.+bs))/3.))))*ma_pow(((nsic(i,k)*rho(i,k))),((((4.-bs))/3.))))/(((4.*720.)*rho(i,k))));
// WRF:1360
} else {
// WRF:1361
nsagg(k) = 0.;
// WRF:1362
}
// WRF:1372
if((((qniic(i,k)>=qsmall)&&(t(i,k)<=273.15))&&(qcic(i,k)>=qsmall))) {
// WRF:1380
dc0 = (((pgam(k)+1.))/lamc(k));
// WRF:1381
ds0 = (1./lams(k));
// WRF:1382
dum = ((((dc0*dc0)*uns(k))*rhow)/(((9.*mua(i,k))*ds0)));
// WRF:1383
eci = ((dum*dum)/((((dum+0.4))*((dum+0.4)))));
// WRF:1384
eci = ma_max(eci,0.);
// WRF:1385
eci = ma_min(eci,1.);
// WRF:1387
psacws(k) = ((((((((pi/4.)*asn(i,k))*qcic(i,k))*rho(i,k))*n0s(k))*eci)*mskf_gamma((bs+3.)))/ma_pow(lams(k),((bs+3.))));
// WRF:1390
npsacws(k) = ((((((((pi/4.)*asn(i,k))*ncic(i,k))*rho(i,k))*n0s(k))*eci)*mskf_gamma((bs+3.)))/ma_pow(lams(k),((bs+3.))));
// WRF:1393
} else {
// WRF:1394
psacws(k) = 0.;
// WRF:1395
npsacws(k) = 0.;
// WRF:1396
}
// WRF:1400
if((((t(i,k)<270.16))&&((t(i,k)>=268.16)))) {
// WRF:1401
ni_secp = (((3.5e8*((270.16-t(i,k))))/2.0)*psacws(k));
// WRF:1402
nsacwi(k) = ni_secp;
// WRF:1403
msacwi(k) = ma_min((ni_secp*mi0),psacws(k));
// WRF:1404
} else if((((t(i,k)<268.16))&&((t(i,k)>=265.16)))) {
// WRF:1405
ni_secp = (((3.5e8*((t(i,k)-265.16)))/3.0)*psacws(k));
// WRF:1406
nsacwi(k) = ni_secp;
// WRF:1407
msacwi(k) = ma_min((ni_secp*mi0),psacws(k));
// WRF:1408
} else {
// WRF:1409
ni_secp = 0.0;
// WRF:1410
nsacwi(k) = 0.0;
// WRF:1411
msacwi(k) = 0.0;
// WRF:1412
}
// WRF:1413
psacws(k) = ma_max(0.0,(psacws(k)-(ni_secp*mi0)));
// WRF:1419
if((((qric(i,k)>=1.e-8)&&(qniic(i,k)>=1.e-8))&&(t(i,k)<=273.15))) {
// WRF:1422
pracs(k) = (((pi*pi)*ecr)*((((((ma_pow(((ma_powi<2>((((1.2*umr(k))-(0.95*ums(k)))))+((0.08*ums(k))*umr(k)))),0.5)*rhow)*rho(i,k))*n0r(k))*n0s(k))*((((5./((ma_powi<6>(lamr(k))*lams(k))))+(2./((ma_powi<5>(lamr(k))*ma_powi<2>(lams(k))))))+(0.5/((ma_powi<4>(lamr(k))*ma_powi<3>(lams(k))))))))));
// WRF:1429
npracs(k) = (((((((pi/2.)*rho(i,k))*ecr)*ma_pow((((1.7*ma_powi<2>(((unr(k)-uns(k)))))+((0.3*unr(k))*uns(k)))),0.5))*n0r(k))*n0s(k))*((((1./((ma_powi<3>(lamr(k))*lams(k))))+(1./((ma_powi<2>(lamr(k))*ma_powi<2>(lams(k))))))+(1./((lamr(k)*ma_powi<3>(lams(k))))))));
// WRF:1434
} else {
// WRF:1435
pracs(k) = 0.;
// WRF:1436
npracs(k) = 0.;
// WRF:1437
}
// WRF:1443
if(((t(i,k)<269.15)&&(qric(i,k)>=qsmall))) {
// WRF:1445
mnuccr(k) = ((((((((20.*pi)*pi)*rhow)*nric(i,k))*bimm)*ma_exp((aimm*((273.15-t(i,k))))))/ma_powi<3>(lamr(k)))/ma_powi<3>(lamr(k)));
// WRF:1449
nnuccr(k) = ((((pi*nric(i,k))*bimm)*ma_exp((aimm*((273.15-t(i,k))))))/ma_powi<3>(lamr(k)));
// WRF:1451
} else {
// WRF:1452
mnuccr(k) = 0.;
// WRF:1453
nnuccr(k) = 0.;
// WRF:1454
}
// WRF:1461
if(((qric(i,k)>=qsmall)&&(qcic(i,k)>=qsmall))) {
// WRF:1462
pra(k) = (67.*ma_pow(((qcic(i,k)*qric(i,k))),1.15));
// WRF:1463
npra(k) = (pra(k)/((qcic(i,k)/ncic(i,k))));
// WRF:1464
} else {
// WRF:1465
pra(k) = 0.;
// WRF:1466
npra(k) = 0.;
// WRF:1467
}
// WRF:1473
if((qric(i,k)>=qsmall)) {
// WRF:1474
nragg(k) = ((((-8.)*nric(i,k))*qric(i,k))*rho(i,k));
// WRF:1475
} else {
// WRF:1476
nragg(k) = 0.;
// WRF:1477
}
// WRF:1484
if((((qniic(i,k)>=qsmall)&&(qiic(i,k)>=qsmall))&&(t(i,k)<=273.15))) {
// WRF:1486
prai(k) = ((((((((pi/4.)*asn(i,k))*qiic(i,k))*rho(i,k))*n0s(k))*eii)*mskf_gamma((bs+3.)))/ma_pow(lams(k),((bs+3.))));
// WRF:1489
nprai(k) = ((((((((pi/4.)*asn(i,k))*niic(i,k))*rho(i,k))*n0s(k))*eii)*mskf_gamma((bs+3.)))/ma_pow(lams(k),((bs+3.))));
// WRF:1492
} else {
// WRF:1493
prai(k) = 0.;
// WRF:1494
nprai(k) = 0.;
// WRF:1495
}
// WRF:1499
prf(k) = (((-umr(k))*qric(i,k))/dz(i,k));
// WRF:1500
pnrf(k) = (((-unr(k))*nric(i,k))/dz(i,k));
// WRF:1501
psf(k) = (((-ums(k))*qniic(i,k))/dz(i,k));
// WRF:1502
pnsf(k) = (((-uns(k))*nsic(i,k))/dz(i,k));
// WRF:1507
if((k==jb(i))) {
// WRF:1508
zkine(i,jb(i)) = 0.5;
// WRF:1509
wu(i,jb(i)) = 1.;
// WRF:1510
zbuo(i,jb(i)) = ((((tu(i,jb(i))*((1.+(retv*qu(i,jb(i))))))-(th(i,jb(i))*((1.+(retv*qh(i,jb(i))))))))/((th(i,jb(i))*((1.+(retv*qh(i,jb(i))))))));
// WRF:1513
} else {
// WRF:1514
if(true) {
// WRF:1518
zbc = (tu(i,k)*((((((1.+(retv*qu(i,k)))-qr(i,k))-qni(i,k))-qi(i,k))-qc(i,k))));
// WRF:1519
zbe = (th(i,k)*((1.+(retv*qh(i,k)))));
// WRF:1520
zbuo(i,k) = (((zbc-zbe))/zbe);
// WRF:1521
zbuoc = (((zbuo(i,k)+zbuo(i,(k+1))))*0.5);
// WRF:1522
zdkbuo = (((dz(i,(k+1))*grav)*zfacbuo)*zbuoc);
// WRF:1523
zdken = ma_min(.99,(((((1.+cwdrag))*ma_max(du(i,k),eu(i,k)))*dz(i,(k+1)))/ma_max(1.e-10,mu(i,(k+1)))));
// WRF:1525
zkine(i,k) = ((((zkine(i,(k+1))*((1.-zdken)))+zdkbuo))/((1.+zdken)));
// WRF:1528
} else {
// WRF:1531
zbc = (tu(i,k)*((1.+(retv*qu(i,k)))));
// WRF:1532
zbe = (th(i,k)*((1.+(retv*qh(i,k)))));
// WRF:1533
zbuo(i,k) = (((((((zbc-zbe))/zbe)-qr(i,k))-qni(i,k))-qi(i,k))-qc(i,k));
// WRF:1534
zbuoc = (((zbuo(i,k)+zbuo(i,(k+1))))*0.5);
// WRF:1535
zdkbuo = ((((dz(i,(k+1))*grav)*zbuoc)*((1.0f-0.25f)))/6.f);
// WRF:1536
zdken = ((du(i,k)*dz(i,(k+1)))/ma_max(1.e-10,mu(i,(k+1))));
// WRF:1537
zkine(i,k) = ((((zkine(i,(k+1))*((1.-zdken)))+zdkbuo))/((1.+zdken)));
// WRF:1539
}
// WRF:1540
wu(i,k) = ma_min(15.,ma_sqrt((2.*ma_max(0.1,zkine(i,k)))));
// WRF:1542
}
// WRF:1545
arcf(i,k) = (mu(i,k)/wu(i,k));
// WRF:1553
for(int _slice=0; _slice<(naer_cu)-(1)+1; ++_slice) naer2h(i,k,(1+_slice)) = (0.5*((naer2(i,k,(1+_slice))+naer2(i,(k+1),(1+_slice)))));
// WRF:1554
ntaerh(i,k) = (0.5*((ntaer(i,k)+ntaer(i,(k+1)))));
// WRF:1558
if(((qcic(i,k)>=qsmall)||(cmel(i,(k+1))>=qsmall))) {
// WRF:1563
mskf_activate(wu(i,k),t(i,k),rho(i,k),naer2h.slice(i,k),naer_cu,naer_cu,maerosol,dispersion_aer,hygro_aer,density_aer,dum2,qsatzm(i,k)); if(error) { arena.used=arena_saved; return; }
// WRF:1568
dum2l(i,k) = dum2;
// WRF:1569
} else {
// WRF:1570
dum2l(i,k) = 0.;
// WRF:1571
}
// WRF:1574
if((((qcic(i,k)>=qsmall)&&(t(i,k)>238.15))&&(k>(jt(i)+2)))) {
// WRF:1578
if((k==kqc(i))) {
// WRF:1579
npccn(k) = (dum2l(i,k)/deltat);
// WRF:1580
} else {
// WRF:1581
npccn(k) = (((dum2l(i,k)-ncic(i,k)))/deltat);
// WRF:1582
}
// WRF:1584
npccn(k) = ma_max(0.,npccn(k));
// WRF:1585
ncmax = dum2l(i,k);
// WRF:1586
} else {
// WRF:1587
npccn(k) = 0.;
// WRF:1588
ncmax = 0.;
// WRF:1589
}
// WRF:1594
esi(i,k) = mskf_polysvp(t(i,k),1);
// WRF:1595
es(i,k) = mskf_polysvp(t(i,k),0);
// WRF:1596
qs(i,k) = ((0.622*es(i,k))/((ph(i,k)-(((1.0-0.622))*es(i,k)))));
// WRF:1597
qs(i,k) = ma_min(1.0,qs(i,k));
// WRF:1598
if((qs(i,k)<0.0)) { qs(i,k) = 1.0; }
// WRF:1600
relhum(i,k) = 1.0;
// WRF:1602
if((t(i,k)<tmelt)) {
// WRF:1603
if(true) {
// WRF:1608
mskf_nucleati(wu(i,k),t(i,k),p(i,k),q(i,k),qcic(i,k),rho(i,k),naer2h.slice(i,k),naer_cu,dum2i(i,k),nihf(i,k),niimm(i,k),nidep(i,k),nimey(i,k)); if(error) { arena.used=arena_saved; return; }
// WRF:1613
nihf(i,k) = (nihf(i,k)*rho(i,k));
// WRF:1614
niimm(i,k) = (niimm(i,k)*rho(i,k));
// WRF:1615
nidep(i,k) = (nidep(i,k)*rho(i,k));
// WRF:1616
nimey(i,k) = (nimey(i,k)*rho(i,k));
// WRF:1617
} else {
// WRF:1620
dum2i(i,k) = ((0.005*ma_exp((0.304*((273.15-t(i,k))))))*1000.);
// WRF:1623
dum2i(i,k) = (ma_min(dum2i(i,k),208.9e3)/rho(i,k));
// WRF:1624
}
// WRF:1625
} else {
// WRF:1626
dum2i(i,k) = 0.;
// WRF:1627
}
// WRF:1633
if(((((dum2i(i,k)>0.)&&(t(i,k)<tmelt))&&(((relhum(i,k)*es(i,k))/esi(i,k))>1.05))&&(k>(jt(i)+1)))) {
// WRF:1636
if((k==kqi(i))) {
// WRF:1637
nnuccd(k) = (dum2i(i,k)/deltat);
// WRF:1638
} else {
// WRF:1639
nnuccd(k) = (((dum2i(i,k)-niic(i,k)))/deltat);
// WRF:1640
}
// WRF:1641
nnuccd(k) = ma_max(nnuccd(k),0.);
// WRF:1642
nimax = dum2i(i,k);
// WRF:1646
mnuccd(k) = (nnuccd(k)*mi0);
// WRF:1647
} else {
// WRF:1648
nnuccd(k) = 0.;
// WRF:1649
nimax = 0.;
// WRF:1650
mnuccd(k) = 0.;
// WRF:1651
}
// WRF:1656
if(((((t(i,k)<=273.15)&&(t(i,k)>233.15))&&(qiic(i,k)>0.5e-6))&&(qcic(i,k)>qsmall))) {
// WRF:1658
plevap = (qcic(i,k)/bergtsf);
// WRF:1659
prb(k) = ma_max(0.,plevap);
// WRF:1660
nprb(k) = (prb(k)/((qcic(i,k)/ncic(i,k))));
// WRF:1661
} else {
// WRF:1662
prb(k) = 0.;
// WRF:1663
nprb(k) = 0.;
// WRF:1664
}
// WRF:1669
if((((((qcic(i,k)>=qsmall)&&(ncic(i,k)>0.))&&(ntaerh(i,k)>0.))&&(t(i,k)<=268.15))&&(t(i,k)>238.15))) {
// WRF:1672
if(false) {
// WRF:1674
ttend(k) = ((((-grav)*wu(i,k))/cp)/((1.0+gamhat(i,k))));
// WRF:1675
naimm = ((((0.00291*naer2h(i,k,idxbcphi))+(32.3*((((naer2h(i,k,idxdst1)+naer2h(i,k,idxdst2))+naer2h(i,k,idxdst3))+naer2h(i,k,idxdst4))))))/ntaerh(i,k));
// WRF:1678
if((ttend(k)<0.)) {
// WRF:1679
nnuccc(k) = (((((-naimm)*ma_exp((273.15-t(i,k))))*ttend(k))*qcic(i,k))/rhow);
// WRF:1680
mnuccc(k) = ((nnuccc(k)*qcic(i,k))/ncic(i,k));
// WRF:1681
}
// WRF:1682
} else {
// WRF:1686
mnuccc(k) = (((((((((pi*pi)/36.)*rhow)*cdist1(k))*mskf_gamma((7.+pgam(k))))*bimm)*ma_exp((aimm*((273.15-t(i,k))))))/ma_powi<3>(lamc(k)))/ma_powi<3>(lamc(k)));
// WRF:1691
nnuccc(k) = ((((((pi/6.)*cdist1(k))*mskf_gamma((pgam(k)+4.)))*bimm)*ma_exp((aimm*((273.15-t(i,k))))))/ma_powi<3>(lamc(k)));
// WRF:1693
}
// WRF:1697
tcnt = ma_pow(((270.16-t(i,k))),1.3);
// WRF:1698
viscosity = (1.8e-5*ma_pow(((t(i,k)/298.0)),0.85));
// WRF:1699
mfp = ((2.0*viscosity)/((ph(i,k)*ma_sqrt(((8.0*28.96e-3)/(((pi*8.314409)*t(i,k))))))));
// WRF:1702
slip1 = (1.0+(((mfp/rn_dst1))*((1.257+((0.4*ma_exp((-(((1.1*rn_dst1)/mfp))))))))));
// WRF:1703
slip2 = (1.0+(((mfp/rn_dst2))*((1.257+((0.4*ma_exp((-(((1.1*rn_dst2)/mfp))))))))));
// WRF:1704
slip3 = (1.0+(((mfp/rn_dst3))*((1.257+((0.4*ma_exp((-(((1.1*rn_dst3)/mfp))))))))));
// WRF:1705
slip4 = (1.0+(((mfp/rn_dst4))*((1.257+((0.4*ma_exp((-(((1.1*rn_dst4)/mfp))))))))));
// WRF:1707
dfaer1 = (((1.381e-23*t(i,k))*slip1)/((((6.*pi)*viscosity)*rn_dst1)));
// WRF:1708
dfaer2 = (((1.381e-23*t(i,k))*slip2)/((((6.*pi)*viscosity)*rn_dst2)));
// WRF:1709
dfaer3 = (((1.381e-23*t(i,k))*slip3)/((((6.*pi)*viscosity)*rn_dst3)));
// WRF:1710
dfaer4 = (((1.381e-23*t(i,k))*slip4)/((((6.*pi)*viscosity)*rn_dst4)));
// WRF:1712
nacon1 = 0.0;
// WRF:1713
nacon2 = 0.0;
// WRF:1714
nacon3 = 0.0;
// WRF:1715
nacon4 = 0.0;
// WRF:1718
if((idxdst1>0)) {
// WRF:1719
nacon1 = ((naer2(i,k,idxdst1)*tcnt)*0.0);
// WRF:1720
}
// WRF:1721
if((idxdst2>0)) {
// WRF:1722
nacon2 = (naer2(i,k,idxdst2)*tcnt);
// WRF:1723
}
// WRF:1724
if((idxdst3>0)) {
// WRF:1725
nacon3 = (naer2(i,k,idxdst3)*tcnt);
// WRF:1726
}
// WRF:1727
if((idxdst4>0)) {
// WRF:1728
nacon4 = (naer2(i,k,idxdst4)*tcnt);
// WRF:1729
}
// WRF:1731
mnucct(k) = ((((((((((((dfaer1*nacon1)+(dfaer2*nacon2))+(dfaer3*nacon3))+(dfaer4*nacon4)))*pi)*pi)/3.)*rhow)*cdist1(k))*mskf_gamma((pgam(k)+5.)))/ma_powi<4>(lamc(k)));
// WRF:1734
nnucct(k) = ((((((((((dfaer1*nacon1)+(dfaer2*nacon2))+(dfaer3*nacon3))+(dfaer4*nacon4)))*2.)*pi)*cdist1(k))*mskf_gamma((pgam(k)+2.)))/lamc(k));
// WRF:1744
} else {
// WRF:1745
mnuccc(k) = 0.;
// WRF:1746
nnuccc(k) = 0.;
// WRF:1747
mnucct(k) = 0.;
// WRF:1748
nnucct(k) = 0.;
// WRF:1749
}
// WRF:1762
mtime = (deltat/900.);
// WRF:1763
mtimec = (deltat/900.);
// WRF:1765
mtime = ma_max(1.0,mtime);
// WRF:1766
mtimec = ma_max(1.0,mtimec);
// WRF:1770
qce = ((mu(i,k)*qc(i,k))+(dz(i,k)*((cmel(i,(k-1))-(du(i,(k-1))*qc(i,k))))));
// WRF:1771
dum = ((arcf(i,k)*(((((((pra(k)+prc(k))+prb(k))+mnuccc(k))+mnucct(k))+msacwi(k))+psacws(k))))*dz(i,k));
// WRF:1773
if((qce<0.)) {
// WRF:1774
prc(k) = 0.;
// WRF:1775
pra(k) = 0.;
// WRF:1776
prb(k) = 0.;
// WRF:1777
mnuccc(k) = 0.;
// WRF:1778
mnucct(k) = 0.;
// WRF:1779
msacwi(k) = 0.;
// WRF:1780
psacws(k) = 0.;
// WRF:1781
} else if((dum>qce)) {
// WRF:1782
ratio = ((qce/dum)*omsm);
// WRF:1783
prc(k) = (prc(k)*ratio);
// WRF:1784
pra(k) = (pra(k)*ratio);
// WRF:1785
prb(k) = (prb(k)*ratio);
// WRF:1786
mnuccc(k) = (mnuccc(k)*ratio);
// WRF:1787
mnucct(k) = (mnucct(k)*ratio);
// WRF:1788
msacwi(k) = (msacwi(k)*ratio);
// WRF:1789
psacws(k) = (psacws(k)*ratio);
// WRF:1790
}
// WRF:1793
nce = ((mu(i,k)*nc(i,k))+(((((arcf(i,k)*npccn(k))*mtimec)-(du(i,(k-1))*nc(i,k))))*dz(i,k)));
// WRF:1794
dum = ((arcf(i,k)*dz(i,k))*((((((nprc1(k)+npra(k))+nnuccc(k))+nnucct(k))+npsacws(k))+nprb(k))));
// WRF:1796
if((nce<0.)) {
// WRF:1797
nprc1(k) = 0.;
// WRF:1799
npra(k) = 0.;
// WRF:1800
nnuccc(k) = 0.;
// WRF:1801
nnucct(k) = 0.;
// WRF:1802
npsacws(k) = 0.;
// WRF:1803
nprb(k) = 0.;
// WRF:1804
} else if((dum>nce)) {
// WRF:1805
ratio = ((nce/dum)*omsm);
// WRF:1806
nprc1(k) = (nprc1(k)*ratio);
// WRF:1807
npra(k) = (npra(k)*ratio);
// WRF:1808
nnuccc(k) = (nnuccc(k)*ratio);
// WRF:1809
nnucct(k) = (nnucct(k)*ratio);
// WRF:1810
npsacws(k) = (npsacws(k)*ratio);
// WRF:1811
nprb(k) = (nprb(k)*ratio);
// WRF:1812
}
// WRF:1815
qie = ((mu(i,k)*qi(i,k))+(dz(i,k)*(((cmei(i,(k-1))-(du(i,(k-1))*qi(i,k)))+(((((mnuccc(k)+mnucct(k))+msacwi(k))+prb(k)))*arcf(i,k))))));
// WRF:1817
dum = ((arcf(i,k)*((prci(k)+prai(k))))*dz(i,k));
// WRF:1818
if((qie<0.)) {
// WRF:1819
prci(k) = 0.;
// WRF:1820
prai(k) = 0.;
// WRF:1821
} else if((dum>qie)) {
// WRF:1822
ratio = ((qie/dum)*omsm);
// WRF:1823
prci(k) = (prci(k)*ratio);
// WRF:1824
prai(k) = (prai(k)*ratio);
// WRF:1825
}
// WRF:1828
nie = ((mu(i,k)*ni(i,k))+(dz(i,k)*(((((nnuccd(k)*mtime)*arcf(i,k))-(du(i,(k-1))*ni(i,k)))+(nnucct(k)*arcf(i,k))))));
// WRF:1830
dum = ((arcf(i,k)*dz(i,k))*((((-nsacwi(k))+nprci(k))+nprai(k))));
// WRF:1832
if((nie<0.)) {
// WRF:1833
nsacwi(k) = 0.;
// WRF:1834
nprci(k) = 0.;
// WRF:1835
nprai(k) = 0.;
// WRF:1836
} else if((dum>nie)) {
// WRF:1837
ratio = ((nie/dum)*omsm);
// WRF:1838
nsacwi(k) = (nsacwi(k)*ratio);
// WRF:1839
nprci(k) = (nprci(k)*ratio);
// WRF:1840
nprai(k) = (nprai(k)*ratio);
// WRF:1841
}
// WRF:1845
qre = ((mu(i,k)*qr(i,k))+((dz(i,k)*((pra(k)+prc(k))))*arcf(i,k)));
// WRF:1846
dum = ((arcf(i,k)*dz(i,k))*(((pracs(k)+mnuccr(k))-prf(k))));
// WRF:1847
if((qre<0.)) {
// WRF:1848
prf(k) = 0.;
// WRF:1849
pracs(k) = 0.;
// WRF:1850
mnuccr(k) = 0.;
// WRF:1851
} else if((dum>qre)) {
// WRF:1852
ratio = ((qre/dum)*omsm);
// WRF:1853
prf(k) = (prf(k)*ratio);
// WRF:1854
pracs(k) = (pracs(k)*ratio);
// WRF:1855
mnuccr(k) = (mnuccr(k)*ratio);
// WRF:1856
}
// WRF:1859
nre = (mu(i,k)*nr(i,k));
// WRF:1860
dum = ((arcf(i,k)*dz(i,k))*((((((-nprc(k))+npracs(k))+nnuccr(k))-nragg(k))-pnrf(k))));
// WRF:1862
if((nre<0.)) {
// WRF:1863
nprc(k) = 0.;
// WRF:1864
npracs(k) = 0.;
// WRF:1865
nnuccr(k) = 0.;
// WRF:1866
nragg(k) = 0.;
// WRF:1867
pnrf(k) = 0.;
// WRF:1868
} else if((dum>nre)) {
// WRF:1869
ratio = ((nre/dum)*omsm);
// WRF:1870
nprc(k) = (nprc(k)*ratio);
// WRF:1871
npracs(k) = (npracs(k)*ratio);
// WRF:1872
nnuccr(k) = (nnuccr(k)*ratio);
// WRF:1873
nragg(k) = (nragg(k)*ratio);
// WRF:1874
pnrf(k) = (pnrf(k)*ratio);
// WRF:1875
}
// WRF:1879
qnie = ((mu(i,k)*qni(i,k))+(dz(i,k)*(((((((prai(k)+psacws(k))+prci(k))+pracs(k))+mnuccr(k)))*arcf(i,k)))));
// WRF:1881
dum = ((arcf(i,k)*dz(i,k))*((-psf(k))));
// WRF:1883
if((qnie<0.)) {
// WRF:1884
psf(k) = 0.;
// WRF:1885
} else if((dum>qnie)) {
// WRF:1886
ratio = ((qnie/dum)*omsm);
// WRF:1887
psf(k) = (psf(k)*ratio);
// WRF:1888
}
// WRF:1891
nse = ((mu(i,k)*ns(i,k))+((dz(i,k)*((nprci(k)+nnuccr(k))))*arcf(i,k)));
// WRF:1892
dum = ((arcf(i,k)*dz(i,k))*(((-nsagg(k))-pnsf(k))));
// WRF:1893
if((nse<0.)) {
// WRF:1894
nsagg(k) = 0.;
// WRF:1895
pnsf(k) = 0.;
// WRF:1896
} else if((dum>nse)) {
// WRF:1897
ratio = ((nse/dum)*omsm);
// WRF:1898
nsagg(k) = (nsagg(k)*ratio);
// WRF:1899
pnsf(k) = (pnsf(k)*ratio);
// WRF:1900
}
// WRF:1906
if((k<=kqc(i))) {
// WRF:1907
qctend(i,k) = (qctend(i,k)+((((((((-pra(k))-prc(k))-prb(k))-mnuccc(k))-mnucct(k))-msacwi(k))-psacws(k))));
// WRF:1914
qitend(i,k) = (qitend(i,k)+(((((((prb(k)+mnuccc(k))+mnucct(k))+msacwi(k))-prci(k))-prai(k))+(mnuccd(k)*mtimec))));
// WRF:1918
qrtend(i,k) = ((qrtend(i,k)+((pra(k)+prc(k))))+(((-pracs(k))-mnuccr(k))));
// WRF:1923
qnitend(i,k) = ((qnitend(i,k)+(((prai(k)+psacws(k))+prci(k))))+((pracs(k)+mnuccr(k))));
// WRF:1929
nctend(i,k) = ((nctend(i,k)+(npccn(k)*mtimec))+(((((((-nnuccc(k))-nnucct(k))-npsacws(k))-npra(k))-nprc1(k))-nprb(k))));
// WRF:1933
nitend(i,k) = ((nitend(i,k)+(nnuccd(k)*mtime))+(((((nnuccc(k)+nnucct(k))+nsacwi(k))-nprci(k))-nprai(k))));
// WRF:1937
nstend(i,k) = ((nstend(i,k)+((nsagg(k)+nnuccr(k))))+nprci(k));
// WRF:1940
nrtend(i,k) = ((nrtend(i,k)+nprc(k))+((((-npracs(k))-nnuccr(k))+nragg(k))));
// WRF:1945
autolm(i,k) = ((-prc(k))*arcf(i,k));
// WRF:1946
accrlm(i,k) = ((-pra(k))*arcf(i,k));
// WRF:1947
bergnm(i,k) = ((-prb(k))*arcf(i,k));
// WRF:1948
fhtimm(i,k) = ((-mnuccc(k))*arcf(i,k));
// WRF:1949
fhtctm(i,k) = ((-mnucct(k))*arcf(i,k));
// WRF:1950
hmpim(i,k) = ((-msacwi(k))*arcf(i,k));
// WRF:1951
accslm(i,k) = ((-psacws(k))*arcf(i,k));
// WRF:1952
collrm(i,k) = ((-pracs(k))*arcf(i,k));
// WRF:1953
collrn(i,k) = ((-npracs(k))*arcf(i,k));
// WRF:1954
fhtcrm(i,k) = ((-mnuccr(k))*arcf(i,k));
// WRF:1955
fhtcrn(i,k) = ((-nnuccr(k))*arcf(i,k));
// WRF:1956
dlfm(i,k) = ((-du(i,k))*qc(i,k));
// WRF:1958
autoln(i,k) = (((-nprc1(k))*arcf(i,k))*rho(i,k));
// WRF:1959
autorn(i,k) = (((-nprc(k))*arcf(i,k))*rho(i,k));
// WRF:1960
aggrn(i,k) = ((nragg(k)*arcf(i,k))*rho(i,k));
// WRF:1961
aggsn(i,k) = ((nsagg(k)*arcf(i,k))*rho(i,k));
// WRF:1962
accrln(i,k) = (((-npra(k))*arcf(i,k))*rho(i,k));
// WRF:1963
bergnn(i,k) = (((-nprb(k))*arcf(i,k))*rho(i,k));
// WRF:1964
fhtimn(i,k) = (((-nnuccc(k))*arcf(i,k))*rho(i,k));
// WRF:1965
fhtctn(i,k) = (((-nnucct(k))*arcf(i,k))*rho(i,k));
// WRF:1966
accsln(i,k) = (((-npsacws(k))*arcf(i,k))*rho(i,k));
// WRF:1967
activn(i,k) = (((npccn(k)*mtimec)*arcf(i,k))*rho(i,k));
// WRF:1968
dlfn(i,k) = (((-du(i,k))*nc(i,k))*rho(i,k));
// WRF:1970
autoim(i,k) = ((-prci(k))*arcf(i,k));
// WRF:1971
accsim(i,k) = ((-prai(k))*arcf(i,k));
// WRF:1972
difm(i,k) = ((-du(i,k))*qi(i,k));
// WRF:1974
nuclin(i,k) = (((nnuccd(k)*mtime)*arcf(i,k))*rho(i,k));
// WRF:1975
nuclim(i,k) = (((mnuccd(k)*mtime)*arcf(i,k))*rho(i,k));
// WRF:1976
autoin(i,k) = (((-nprci(k))*arcf(i,k))*rho(i,k));
// WRF:1977
accsin(i,k) = (((-nprai(k))*arcf(i,k))*rho(i,k));
// WRF:1978
hmpin(i,k) = ((nsacwi(k)*arcf(i,k))*rho(i,k));
// WRF:1979
difn(i,k) = (((-du(i,k))*ni(i,k))*rho(i,k));
// WRF:1980
} else {
// WRF:1981
qctend(i,k) = 0.;
// WRF:1982
qitend(i,k) = 0.;
// WRF:1983
qrtend(i,k) = 0.;
// WRF:1984
qnitend(i,k) = 0.;
// WRF:1985
nctend(i,k) = 0.;
// WRF:1986
nitend(i,k) = 0.;
// WRF:1987
nstend(i,k) = 0.;
// WRF:1988
nrtend(i,k) = 0.;
// WRF:1989
}
// WRF:1995
if((k<=kqi(i))) {
// WRF:1996
qni(i,(k-1)) = ((1./mu(i,(k-1)))*(((mu(i,k)*qni(i,k))+((dz(i,k)*((qnitend(i,k)+psf(k))))*arcf(i,k)))));
// WRF:1999
ns(i,(k-1)) = ((1./mu(i,(k-1)))*(((mu(i,k)*ns(i,k))+((dz(i,k)*((nstend(i,k)+pnsf(k))))*arcf(i,k)))));
// WRF:2002
} else {
// WRF:2003
qni(i,(k-1)) = 0.;
// WRF:2004
ns(i,(k-1)) = 0.;
// WRF:2005
}
// WRF:2007
if((qni(i,(k-1))<=0.)) {
// WRF:2008
qni(i,(k-1)) = 0.;
// WRF:2009
ns(i,(k-1)) = 0.;
// WRF:2010
}
// WRF:2013
if((k<=kqc(i))) {
// WRF:2014
qr(i,(k-1)) = ((1./mu(i,(k-1)))*(((mu(i,k)*qr(i,k))+((dz(i,k)*((qrtend(i,k)+prf(k))))*arcf(i,k)))));
// WRF:2017
nr(i,(k-1)) = ((1./mu(i,(k-1)))*(((mu(i,k)*nr(i,k))+((dz(i,k)*((nrtend(i,k)+pnrf(k))))*arcf(i,k)))));
// WRF:2020
} else {
// WRF:2021
qr(i,(k-1)) = 0.;
// WRF:2022
nr(i,(k-1)) = 0.;
// WRF:2023
}
// WRF:2025
if((qr(i,(k-1))<=0.)) {
// WRF:2026
qr(i,(k-1)) = 0.;
// WRF:2027
nr(i,(k-1)) = 0.;
// WRF:2028
}
// WRF:2032
if(((t(i,(k-1))<233.15)&&(qr(i,(k-1))>0.))) {
// WRF:2035
dum = ((xlf/cp)*qr(i,(k-1)));
// WRF:2036
if(((t(i,(k-1))+dum)>233.15)) {
// WRF:2037
dum = (((-((t(i,(k-1))-233.15)))*cp)/xlf);
// WRF:2039
dum = (dum/qr(i,(k-1)));
// WRF:2040
dum = ma_max(0.,dum);
// WRF:2041
dum = ma_min(1.,dum);
// WRF:2042
} else {
// WRF:2043
dum = 1.;
// WRF:2044
}
// WRF:2045
qni(i,(k-1)) = (qni(i,(k-1))+(dum*qr(i,(k-1))));
// WRF:2046
ns(i,(k-1)) = (ns(i,(k-1))+(dum*nr(i,(k-1))));
// WRF:2047
qr(i,(k-1)) = (((1.-dum))*qr(i,(k-1)));
// WRF:2048
nr(i,(k-1)) = (((1.-dum))*nr(i,(k-1)));
// WRF:2049
fhmrm(i,(k-1)) = ((((-mu(i,(k-1)))*dum)*qr(i,(k-1)))/dz(i,k));
// WRF:2050
}
// WRF:2058
if((k<=kqc(i))) {
// WRF:2059
qc(i,(k-1)) = ((1./mu(i,(k-1)))*(((((mu(i,k)*qc(i,k))-((dz(i,k)*du(i,(k-1)))*qc(i,k)))+((dz(i,k)*qctend(i,k))*arcf(i,k)))+(dz(i,k)*cmel(i,(k-1))))));
// WRF:2063
nc(i,(k-1)) = ((1./mu(i,(k-1)))*((((mu(i,k)*nc(i,k))-((dz(i,k)*du(i,(k-1)))*nc(i,k)))+((dz(i,k)*nctend(i,k))*arcf(i,k)))));
// WRF:2067
} else {
// WRF:2068
qc(i,(k-1)) = 0.;
// WRF:2069
nc(i,(k-1)) = 0.;
// WRF:2070
}
// WRF:2072
qcorg = qc(i,(k-1));
// WRF:2073
ncorg = nc(i,(k-1));
// WRF:2074
if((qc(i,(k-1))<=0.)) {
// WRF:2075
qc(i,(k-1)) = 0.;
// WRF:2076
nc(i,(k-1)) = 0.;
// WRF:2077
}
// WRF:2078
qcadj(i,(k-1)) = (((((qc(i,(k-1))-qcorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2079
ncadj(i,(k-1)) = (((((nc(i,(k-1))-ncorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2082
if((k<=kqi(i))) {
// WRF:2083
qi(i,(k-1)) = ((1./mu(i,(k-1)))*(((((mu(i,k)*qi(i,k))-((dz(i,k)*du(i,(k-1)))*qi(i,k)))+((dz(i,k)*qitend(i,k))*arcf(i,k)))+(dz(i,k)*cmei(i,(k-1))))));
// WRF:2087
ni(i,(k-1)) = ((1./mu(i,(k-1)))*((((mu(i,k)*ni(i,k))-((dz(i,k)*du(i,(k-1)))*ni(i,k)))+((dz(i,k)*nitend(i,k))*arcf(i,k)))));
// WRF:2091
} else {
// WRF:2092
qi(i,(k-1)) = 0.;
// WRF:2093
ni(i,(k-1)) = 0.;
// WRF:2094
}
// WRF:2096
qiorg = qi(i,(k-1));
// WRF:2097
niorg = ni(i,(k-1));
// WRF:2098
if((qi(i,(k-1))<=0.)) {
// WRF:2099
qi(i,(k-1)) = 0.;
// WRF:2100
ni(i,(k-1)) = 0.;
// WRF:2101
}
// WRF:2102
qiadj(i,(k-1)) = (((((qi(i,(k-1))-qiorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2103
niadj(i,(k-1)) = (((((ni(i,(k-1))-niorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2114
if(((t(i,(k-1))<233.15)&&(qc(i,(k-1))>0.))) {
// WRF:2116
dum = ((xlf/cp)*qc(i,(k-1)));
// WRF:2117
if(((t(i,(k-1))+dum)>233.15)) {
// WRF:2118
dum = (((-((t(i,(k-1))-233.15)))*cp)/xlf);
// WRF:2120
dum = (dum/qc(i,(k-1)));
// WRF:2121
dum = ma_max(0.,dum);
// WRF:2122
dum = ma_min(1.,dum);
// WRF:2123
} else {
// WRF:2124
dum = 1.;
// WRF:2125
}
// WRF:2126
qi(i,(k-1)) = (qi(i,(k-1))+(dum*qc(i,(k-1))));
// WRF:2127
ni(i,(k-1)) = (ni(i,(k-1))+(dum*nc(i,(k-1))));
// WRF:2128
fhmlm(i,(k-1)) = ((((-mu(i,(k-1)))*dum)*qc(i,(k-1)))/dz(i,k));
// WRF:2129
fhmln(i,(k-1)) = (((((-mu(i,(k-1)))*dum)*nc(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2130
qc(i,(k-1)) = (((1.-dum))*qc(i,(k-1)));
// WRF:2131
nc(i,(k-1)) = (((1.-dum))*nc(i,(k-1)));
// WRF:2132
}
// WRF:2134
frz(i,(k-1)) = (((cmei(i,(k-1))+(arcf(i,k)*(((((((prb(k)+mnuccc(k))+mnucct(k))+msacwi(k))+pracs(k))+mnuccr(k))+psacws(k)))))-fhmlm(i,(k-1)))-fhmrm(i,(k-1)));
// WRF:2149
niorg = ni(i,(k-1));
// WRF:2152
if((qi(i,(k-1))>=qsmall)) {
// WRF:2154
ni(i,(k-1)) = ma_min(ni(i,(k-1)),(qi(i,(k-1))*1.e20));
// WRF:2155
lami((k-1)) = ma_pow(((((mskf_gamma((1.+di))*ci)*ni(i,(k-1)))/qi(i,(k-1)))),((1./di)));
// WRF:2157
n0i((k-1)) = (ni(i,(k-1))*lami((k-1)));
// WRF:2159
lammax = (1./10.e-6);
// WRF:2160
lammin = (1./((2.*dcs)));
// WRF:2162
if((lami((k-1))<lammin)) {
// WRF:2163
lami((k-1)) = lammin;
// WRF:2164
n0i((k-1)) = ((ma_pow(lami((k-1)),((di+1.)))*qi(i,(k-1)))/((ci*mskf_gamma((1.+di)))));
// WRF:2165
ni(i,(k-1)) = (n0i((k-1))/lami((k-1)));
// WRF:2166
} else if((lami((k-1))>lammax)) {
// WRF:2167
lami((k-1)) = lammax;
// WRF:2168
n0i((k-1)) = ((ma_pow(lami((k-1)),((di+1.)))*qi(i,(k-1)))/((ci*mskf_gamma((1.+di)))));
// WRF:2169
ni(i,(k-1)) = (n0i((k-1))/lami((k-1)));
// WRF:2170
}
// WRF:2171
effi(i,(k-1)) = ((1.5/lami((k-1)))*1.e6);
// WRF:2172
} else {
// WRF:2173
lami((k-1)) = 0.;
// WRF:2174
n0i((k-1)) = 0.;
// WRF:2175
effi(i,(k-1)) = 0.;
// WRF:2176
}
// WRF:2179
niadj(i,(k-1)) = (niadj(i,(k-1))+(((((ni(i,(k-1))-niorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k)));
// WRF:2182
ncorg = nc(i,(k-1));
// WRF:2185
if((qc(i,(k-1))>=qsmall)) {
// WRF:2188
nc(i,(k-1)) = ma_min(nc(i,(k-1)),(qc(i,(k-1))*1.e20));
// WRF:2192
pgam((k-1)) = ((0.0005714*(((nc(i,(k-1))/1.e6)*rho(i,(k-1)))))+0.2714);
// WRF:2193
pgam((k-1)) = ((1./(ma_powi<2>(pgam((k-1)))))-1.);
// WRF:2194
pgam((k-1)) = ma_max(pgam((k-1)),2.);
// WRF:2195
pgam((k-1)) = ma_min(pgam((k-1)),15.);
// WRF:2198
lamc((k-1)) = ma_pow(((((((pi/6.)*rhow)*nc(i,(k-1)))*mskf_gamma((pgam((k-1))+4.)))/((qc(i,(k-1))*mskf_gamma((pgam((k-1))+1.)))))),((1./3.)));
// WRF:2202
lammin = (((pgam((k-1))+1.))/50.e-6);
// WRF:2203
lammax = (((pgam((k-1))+1.))/2.e-6);
// WRF:2205
if((lamc((k-1))<lammin)) {
// WRF:2206
lamc((k-1)) = lammin;
// WRF:2207
nc(i,(k-1)) = ((((6.*ma_powi<3>(lamc((k-1))))*qc(i,(k-1)))*mskf_gamma((pgam((k-1))+1.)))/(((pi*rhow)*mskf_gamma((pgam((k-1))+4.)))));
// WRF:2210
} else if((lamc((k-1))>lammax)) {
// WRF:2211
lamc((k-1)) = lammax;
// WRF:2212
nc(i,(k-1)) = ((((6.*ma_powi<3>(lamc((k-1))))*qc(i,(k-1)))*mskf_gamma((pgam((k-1))+1.)))/(((pi*rhow)*mskf_gamma((pgam((k-1))+4.)))));
// WRF:2215
}
// WRF:2216
effc(i,(k-1)) = ((((mskf_gamma((pgam((k-1))+4.))/mskf_gamma((pgam((k-1))+3.)))/lamc((k-1)))/2.)*1.e6);
// WRF:2220
cdist1((k-1)) = (nc(i,(k-1))/mskf_gamma((pgam((k-1))+1.)));
// WRF:2221
} else {
// WRF:2222
lamc((k-1)) = 0.;
// WRF:2223
cdist1((k-1)) = 0.;
// WRF:2224
effc(i,(k-1)) = 0.;
// WRF:2225
}
// WRF:2228
ncadj(i,(k-1)) = (ncadj(i,(k-1))+(((((nc(i,(k-1))-ncorg))*mu(i,(k-1)))/dz(i,k))*rho(i,k)));
// WRF:2230
trspcm(i,(k-1)) = ((((mu(i,k)*qc(i,k))-(mu(i,(k-1))*qc(i,(k-1)))))/dz(i,k));
// WRF:2231
trspcn(i,(k-1)) = (((((mu(i,k)*nc(i,k))-(mu(i,(k-1))*nc(i,(k-1)))))/dz(i,k))*rho(i,k));
// WRF:2232
trspim(i,(k-1)) = ((((mu(i,k)*qi(i,k))-(mu(i,(k-1))*qi(i,(k-1)))))/dz(i,k));
// WRF:2233
trspin(i,(k-1)) = (((((mu(i,k)*ni(i,k))-(mu(i,(k-1))*ni(i,(k-1)))))/dz(i,k))*rho(i,k));
// WRF:2235
if(((k-1)==(jt(i)+1))) {
// WRF:2236
trspcm(i,(k-2)) = ((mu(i,(k-1))*qc(i,(k-1)))/dz(i,k));
// WRF:2237
trspcn(i,(k-2)) = (((mu(i,(k-1))*nc(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2238
trspim(i,(k-2)) = ((mu(i,(k-1))*qi(i,(k-1)))/dz(i,k));
// WRF:2239
trspin(i,(k-2)) = (((mu(i,(k-1))*ni(i,(k-1)))/dz(i,k))*rho(i,k));
// WRF:2240
dlfm(i,(k-2)) = ((-du(i,(k-2)))*qc(i,(k-1)));
// WRF:2241
dlfn(i,(k-2)) = (((-du(i,(k-2)))*nc(i,(k-1)))*rho(i,k));
// WRF:2242
difm(i,(k-2)) = ((-du(i,(k-2)))*qi(i,(k-1)));
// WRF:2243
difn(i,(k-2)) = (((-du(i,(k-2)))*ni(i,(k-1)))*rho(i,k));
// WRF:2244
}
// WRF:2249
if((qr(i,(k-1))>=qsmall)) {
// WRF:2251
lamr((k-1)) = ma_pow(((((pi*rhow)*nr(i,(k-1)))/qr(i,(k-1)))),((1./3.)));
// WRF:2252
n0r((k-1)) = (nr(i,(k-1))*lamr((k-1)));
// WRF:2254
lammax = (1./20.e-6);
// WRF:2255
lammin = (1./500.e-6);
// WRF:2257
if((lamr((k-1))<lammin)) {
// WRF:2258
lamr((k-1)) = lammin;
// WRF:2259
n0r((k-1)) = ((ma_powi<4>(lamr((k-1)))*qr(i,(k-1)))/((pi*rhow)));
// WRF:2260
nr(i,(k-1)) = (n0r((k-1))/lamr((k-1)));
// WRF:2261
} else if((lamr((k-1))>lammax)) {
// WRF:2262
lamr((k-1)) = lammax;
// WRF:2263
n0r((k-1)) = ((ma_powi<4>(lamr((k-1)))*qr(i,(k-1)))/((pi*rhow)));
// WRF:2264
nr(i,(k-1)) = (n0r((k-1))/lamr((k-1)));
// WRF:2265
}
// WRF:2266
} else {
// WRF:2267
lamr((k-1)) = 0.;
// WRF:2268
n0r((k-1)) = 0.;
// WRF:2269
}
// WRF:2272
if((qni(i,(k-1))>=qsmall)) {
// WRF:2273
lams((k-1)) = ma_pow(((((mskf_gamma((1.+ds))*cs)*ns(i,(k-1)))/qni(i,(k-1)))),((1./ds)));
// WRF:2275
n0s((k-1)) = (ns(i,(k-1))*lams((k-1)));
// WRF:2278
lammax = (1./10.e-6);
// WRF:2279
lammin = (1./2000.e-6);
// WRF:2281
if((lams((k-1))<lammin)) {
// WRF:2282
lams((k-1)) = lammin;
// WRF:2283
n0s((k-1)) = ((ma_pow(lams((k-1)),((ds+1.)))*qni(i,(k-1)))/((cs*mskf_gamma((1.+ds)))));
// WRF:2284
ns(i,(k-1)) = (n0s((k-1))/lams((k-1)));
// WRF:2285
} else if((lams((k-1))>lammax)) {
// WRF:2286
lams((k-1)) = lammax;
// WRF:2287
n0s((k-1)) = ((ma_pow(lams((k-1)),((ds+1.)))*qni(i,(k-1)))/((cs*mskf_gamma((1.+ds)))));
// WRF:2288
ns(i,(k-1)) = (n0s((k-1))/lams((k-1)));
// WRF:2289
}
// WRF:2290
effs(i,(k-1)) = ((1.5/lams((k-1)))*1.e6);
// WRF:2291
} else {
// WRF:2292
lams((k-1)) = 0.;
// WRF:2293
n0s((k-1)) = 0.;
// WRF:2294
effs(i,(k-1)) = 0.;
// WRF:2295
}
// WRF:2300
rprd(i,(k-1)) = (qrtend(i,k)*arcf(i,k));
// WRF:2301
sprd(i,(k-1)) = (qnitend(i,k)*arcf(i,k));
// WRF:2307
}
// WRF:2311
if((qni(i,(k-1))<qsmall)) {
// WRF:2312
qni(i,(k-1)) = 0.;
// WRF:2313
ns(i,(k-1)) = 0.;
// WRF:2314
}
// WRF:2316
if((qr(i,(k-1))<qsmall)) {
// WRF:2317
qr(i,(k-1)) = 0.;
// WRF:2318
nr(i,(k-1)) = 0.;
// WRF:2319
}
// WRF:2320
if((qi(i,(k-1))<qsmall)) {
// WRF:2321
qi(i,(k-1)) = 0.;
// WRF:2322
ni(i,(k-1)) = 0.;
// WRF:2323
}
// WRF:2325
if((qc(i,(k-1))<qsmall)) {
// WRF:2326
qc(i,(k-1)) = 0.;
// WRF:2327
nc(i,(k-1)) = 0.;
// WRF:2328
}
// WRF:2333
nr(i,(k-1)) = ma_max(nr(i,(k-1)),0.);
// WRF:2334
ns(i,(k-1)) = ma_max(ns(i,(k-1)),0.);
// WRF:2335
ni(i,(k-1)) = ma_max(ni(i,(k-1)),0.);
// WRF:2336
nc(i,(k-1)) = ma_max(nc(i,(k-1)),0.);
// WRF:2340
}
// WRF:2342
}
// WRF:2343
ma_label_300:;
// WRF:2344
}
// WRF:2359
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ double mskf_gamma(double x) {
int arena_saved=arena.used;
double retval;
int i;
int n;
bool parity;
MaArray<double> c=arena.make<double>(1,7);
double conv;
double eps;
double fact;
double half;
double one;
MaArray<double> p=arena.make<double>(1,8);
double pi;
MaArray<double> q=arena.make<double>(1,8);
double res;
double sqrtpi;
double sum;
double twelve;
double two;
double xbig;
double xden;
double xinf;
double xminin;
double xnum;
double y;
double y1;
double ysq;
double z;
double zero;
// WRF:2473
one=1.0e0; half=0.5e0; twelve=12.0e0; two=2.0e0; zero=0.0e0; sqrtpi=0.9189385332046727417803297e0; pi=3.1415926535897932384626434e0;
// WRF:2482
xbig=35.040e0; xminin=1.18e-38; eps=1.19e-7; xinf=3.4e38;
// WRF:2490
p.at(0)=(-1.71618513886549492533811e+0); p.at(1)=2.47656508055759199108314e+1; p.at(2)=(-3.79804256470945635097577e+2); p.at(3)=6.29331155312818442661052e+2; p.at(4)=8.66966202790413211295064e+2; p.at(5)=(-3.14512729688483675254357e+4); p.at(6)=(-3.61444134186911729807069e+4); p.at(7)=6.64561438202405440627855e+4;
// WRF:2494
q.at(0)=(-3.08402300119738975254353e+1); q.at(1)=3.15350626979604161529144e+2; q.at(2)=(-1.01515636749021914166146e+3); q.at(3)=(-3.10777167157231109440444e+3); q.at(4)=2.25381184209801510330112e+4; q.at(5)=4.75584627752788110767815e+3; q.at(6)=(-1.34659959864969306392456e+5); q.at(7)=(-1.15132259675553483497211e+5);
// WRF:2509
c.at(0)=(-1.910444077728e-03); c.at(1)=8.4171387781295e-04; c.at(2)=(-5.952379913043012e-04); c.at(3)=7.93650793500350248e-04; c.at(4)=(-2.777777777777681622553e-03); c.at(5)=8.333333333333333331554247e-02; c.at(6)=5.7083835261e-03;
// WRF:2522
parity = false;
// WRF:2523
fact = one;
// WRF:2524
n = 0;
// WRF:2525
y = x;
// WRF:2526
if((y<=zero)) {
// WRF:2530
y = (-x);
// WRF:2531
y1 = ma_aint(y);
// WRF:2532
res = (y-y1);
// WRF:2533
if((res!=zero)) {
// WRF:2534
if((y1!=(ma_aint((y1*half))*two))) { parity = true; }
// WRF:2535
fact = ((-pi)/sin((pi*res)));
// WRF:2536
y = (y+one);
// WRF:2537
} else {
// WRF:2538
res = xinf;
// WRF:2539
goto ma_label_900;
// WRF:2540
}
// WRF:2541
}
// WRF:2545
if((y<eps)) {
// WRF:2549
if((y>=xminin)) {
// WRF:2550
res = (one/y);
// WRF:2551
} else {
// WRF:2552
res = xinf;
// WRF:2553
goto ma_label_900;
// WRF:2554
}
// WRF:2555
} else if((y<twelve)) {
// WRF:2556
y1 = y;
// WRF:2557
if((y<one)) {
// WRF:2561
z = y;
// WRF:2562
y = (y+one);
// WRF:2563
} else {
// WRF:2567
n = ((int)(y)-1);
// WRF:2568
y = (y-(double)(n));
// WRF:2569
z = (y-one);
// WRF:2570
}
// WRF:2574
xnum = zero;
// WRF:2575
xden = one;
// WRF:2576
for(i=1; i<=8; i+=1) {
// WRF:2577
xnum = (((xnum+p(i)))*z);
// WRF:2578
xden = ((xden*z)+q(i));
// WRF:2579
}
// WRF:2580
res = ((xnum/xden)+one);
// WRF:2581
if((y1<y)) {
// WRF:2585
res = (res/y1);
// WRF:2586
} else if((y1>y)) {
// WRF:2590
for(i=1; i<=n; i+=1) {
// WRF:2591
res = (res*y);
// WRF:2592
y = (y+one);
// WRF:2593
}
// WRF:2594
}
// WRF:2595
} else {
// WRF:2599
if((y<=xbig)) {
// WRF:2600
ysq = (y*y);
// WRF:2601
sum = c(7);
// WRF:2602
for(i=1; i<=6; i+=1) {
// WRF:2603
sum = ((sum/ysq)+c(i));
// WRF:2604
}
// WRF:2605
sum = (((sum/y)-y)+sqrtpi);
// WRF:2606
sum = (sum+(((y-half))*ma_log(y)));
// WRF:2607
res = ma_exp(sum);
// WRF:2608
} else {
// WRF:2609
res = xinf;
// WRF:2610
goto ma_label_900;
// WRF:2611
}
// WRF:2612
}
// WRF:2616
if(parity) { res = (-res); }
// WRF:2617
if((fact!=one)) { res = (fact/res); }
// WRF:2618
ma_label_900: retval = res;
// WRF:2620
{ arena.used=arena_saved; return retval; }
arena.used=arena_saved; return retval;
}
__device__ void mskf_activate(double wbar,double tair,double rhoair,MaArray<double> na,int pmode,int nmode,MaArray<double> ma,MaArray<double> sigman,MaArray<double> hygro,MaArray<double> rhodry,double& nact,double qs) {
int arena_saved=arena.used;
int ptype;
double derf;
double derfc;
double erf_alt;
const int nx=200;
int maxmodes;
double surften;
double p0;
MaArray<double> volc=arena.make<double>(1,naer_cu);
double tmass;
double rm;
double pres;
double path;
double diff;
double conduct;
double diff0;
double conduct0;
double dqsdt;
double dqsdp;
double gloc;
double zeta;
MaArray<double> eta=arena.make<double>(1,naer_cu);
MaArray<double> smc=arena.make<double>(1,naer_cu);
double lnsmax;
double alpha;
double gammaloc;
double beta;
double sqrtg;
double alogam;
double rlo;
double rhi;
double xint1;
double xint2;
double xint3;
double xint4;
double w;
double wnuc;
double wb;
double alw;
double sqrtalw;
double smax;
double x;
double arg;
double xmincoeff;
double xcut;
double volcut;
double surfcut;
double z;
double z1;
double z2;
double wf1;
double wf2;
double zf1;
double zf2;
double gf1;
double gf2;
double gf;
double etafactor1;
double etafactor2max;
MaArray<double> etafactor2=arena.make<double>(1,naer_cu);
double es;
int m;
int n;
MaArray<double> amcubeloc=arena.make<double>(1,naer_cu);
MaArray<double> lnsmloc=arena.make<double>(1,naer_cu);
// WRF:2844
surften=0.076f;
// WRF:2847
p0=1013.25e2f;
// WRF:2885
maxmodes = naer_cu;
// WRF:2887
if((maxmodes<pmode)) {
// WRF:2890
}
// WRF:2892
nact = 0.;
// WRF:2894
if(((nmode==1)&&(na(1)<1.e-20f))) { { arena.used=arena_saved; return; } }
// WRF:2896
if((wbar<=0.f)) { { arena.used=arena_saved; return; } }
// WRF:2898
pres = ((rair*rhoair)*tair);
// WRF:2899
diff0 = ((0.211e-4f*((p0/pres)))*ma_pow(((tair/t0)),1.94f));
// WRF:2900
conduct0 = ((((5.69f+(0.017f*((tair-t0)))))*4.186e2f)*1.e-5f);
// WRF:2904
dqsdt = ((latvap/(((rh2o*tair)*tair)))*qs);
// WRF:2905
alpha = (gravit*(((latvap/((((cpair*rh2o)*tair)*tair)))-(1.f/((rair*tair))))));
// WRF:2906
gammaloc = (((1+((latvap/cpair)*dqsdt)))/((rhoair*qs)));
// WRF:2909
gloc = (1.f/(((rhoh2o/(((diff0*rhoair)*qs)))+(((latvap*rhoh2o)/((conduct0*tair)))*(((latvap/((rh2o*tair)))-1.f))))));
// WRF:2911
sqrtg = ma_sqrt(gloc);
// WRF:2912
beta = ((((4.f*pi)*rhoh2o)*gloc)*gammaloc);
// WRF:2913
etafactor2max = (1.e10f/ma_pow(((alpha*wbar)),1.5f));
// WRF:2915
for(m=1; m<=nmode; m+=1) {
// WRF:2917
volc(m) = (ma(m)/(rhodry(m)));
// WRF:2918
if(((volc(m)>1.e-39)&&(na(m)>1.e-39))) {
// WRF:2919
etafactor2(m) = (1.f/(((na(m)*beta)*sqrtg)));
// WRF:2921
amcubeloc(m) = (((3.f*volc(m))/((((4.f*pi)*exp45logsig(m))*na(m)))));
// WRF:2922
smc(m) = smcrit(m);
// WRF:2925
if((hygro(m)>1.e-10f)) {
// WRF:2926
smc(m) = ((2.f*aten)*ma_sqrt((aten/(((27.f*hygro(m))*amcubeloc(m))))));
// WRF:2927
} else {
// WRF:2928
smc(m) = 100.f;
// WRF:2929
}
// WRF:2930
} else {
// WRF:2931
smc(m) = 1.f;
// WRF:2932
etafactor2(m) = etafactor2max;
// WRF:2933
}
// WRF:2934
lnsmloc(m) = ma_log(smc(m));
// WRF:2935
}
// WRF:2938
wnuc = wbar;
// WRF:2941
w = wbar;
// WRF:2942
alw = (alpha*wnuc);
// WRF:2943
sqrtalw = ma_sqrt(alw);
// WRF:2944
zeta = (((2.f*sqrtalw)*aten)/((3.f*sqrtg)));
// WRF:2945
etafactor1 = ((2.f*alw)*sqrtalw);
// WRF:2947
for(m=1; m<=nmode; m+=1) {
// WRF:2948
eta(m) = (etafactor1*etafactor2(m));
// WRF:2949
}
// WRF:2952
mskf_maxsat(zeta,eta,nmode,smc,smax); if(error) { arena.used=arena_saved; return; }
// WRF:2954
lnsmax = ma_log(smax);
// WRF:2956
xmincoeff = ((alogaten-((2.f*third)*((lnsmax-alog2))))-alog3);
// WRF:2958
nact = 0.;
// WRF:2959
for(m=1; m<=nmode; m+=1) {
// WRF:2960
x = ((2*((lnsmloc(m)-lnsmax)))/(((3*sq2)*alogsig(m))));
// WRF:2966
nact = (nact+((0.5f*((1.f-cam_erf(x))))*na(m)));
// WRF:2971
}
// WRF:2972
nact = (nact/rhoair);
// WRF:2984
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ void mskf_maxsat(double zeta,MaArray<double> eta,int nmode,MaArray<double> smc,double& smax) {
int arena_saved=arena.used;
int m;
double sum;
double g1;
double g2;
// WRF:3005
for(m=1; m<=nmode; m+=1) {
// WRF:3006
if(((zeta>(1.e5f*eta(m)))||((smc(m)*smc(m))>(1.e5f*eta(m))))) {
// WRF:3008
smax = 1.e-20f;
// WRF:3009
} else {
// WRF:3011
goto ma_label_1;
// WRF:3012
}
// WRF:3013
}
// WRF:3015
{ arena.used=arena_saved; return; }
// WRF:3017
ma_label_1:;
// WRF:3019
sum = 0;
// WRF:3020
for(m=1; m<=nmode; m+=1) {
// WRF:3021
if((eta(m)>1.e-20f)) {
// WRF:3022
g1 = ma_sqrt((zeta/eta(m)));
// WRF:3023
g1 = ((g1*g1)*g1);
// WRF:3024
g2 = (smc(m)/ma_sqrt((eta(m)+(3*zeta))));
// WRF:3025
g2 = ma_sqrt(g2);
// WRF:3026
g2 = ((g2*g2)*g2);
// WRF:3027
sum = (sum+((((f1(m)*g1)+(f2(m)*g2)))/((smc(m)*smc(m)))));
// WRF:3029
} else {
// WRF:3030
sum = 1.e20f;
// WRF:3031
}
// WRF:3032
}
// WRF:3034
smax = (1.f/ma_sqrt(sum));
// WRF:3036
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ void mskf_nucleati(double wbar,double tair,double pair,double qv,double qc,double rhoair,MaArray<double> na,int naer_all,double& nuci,double& onihf,double& oniimm,double& onidep,double& onimey) {
int arena_saved=arena.used;
double so4_num;
double soot_num;
double dst1_num;
double dst2_num;
double dst3_num;
double dst4_num;
double dst_num;
double nihf;
double niimm;
double nidep;
double nimey;
double n1;
double ni;
double tc;
double a;
double b;
double c;
double regm;
double esl;
double esi;
double deles;
double qsi;
double qsl;
double relhum;
double dst_scale;
double subgrid;
double dmc;
double ssmc;
// WRF:3094
so4_num = 0.0;
// WRF:3095
soot_num = 0.0;
// WRF:3096
dst_num = 0.0;
// WRF:3097
dst1_num = 0.0;
// WRF:3098
dst2_num = 0.0;
// WRF:3099
dst3_num = 0.0;
// WRF:3100
dst4_num = 0.0;
// WRF:3108
if((idxsul>0)) {
// WRF:3109
so4_num = (na(idxsul)*1.0e-6);
// WRF:3110
}
// WRF:3112
if((idxbcphi>0)) {
// WRF:3113
soot_num = (na(idxbcphi)*1.0e-6);
// WRF:3114
}
// WRF:3116
if((idxdst1>0)) {
// WRF:3117
dst1_num = (na(idxdst1)*1.0e-6);
// WRF:3118
}
// WRF:3120
if((idxdst2>0)) {
// WRF:3121
dst2_num = (na(idxdst2)*1.0e-6);
// WRF:3122
}
// WRF:3124
if((idxdst3>0)) {
// WRF:3125
dst3_num = (na(idxdst3)*1.0e-6);
// WRF:3126
}
// WRF:3128
if((idxdst4>0)) {
// WRF:3129
dst4_num = (na(idxdst4)*1.0e-6);
// WRF:3130
}
// WRF:3132
dst_num = (((dst1_num+dst2_num)+dst3_num)+dst4_num);
// WRF:3136
ni = 0.;
// WRF:3137
tc = (tair-273.15);
// WRF:3140
esi = (611.2f*ma_exp(((21.87f*((tair-273.16f)))/((tair-7.66f)))));
// WRF:3141
esl = (611.2f*ma_exp(((17.67f*((tair-273.16f)))/(((243.5f+tair)-273.16f)))));
// WRF:3142
qsi = ((0.622f*esi)/((pair-esi)));
// WRF:3143
qsl = ((0.622f*esl)/((pair-esl)));
// WRF:3144
deles = (qv/qsi);
// WRF:3145
relhum = (qv/qsl);
// WRF:3148
niimm = 0.;
// WRF:3149
nidep = 0.;
// WRF:3150
nihf = 0.;
// WRF:3152
if(((so4_num>=1.0e-10)&&(((soot_num+dst_num))>=1.0e-10))) {
// WRF:3154
subgrid = 1.0;
// WRF:3157
if(((((wbar<4.0))&&((tc<=(-35.0))))&&((((deles*subgrid))>=1.0)))) {
// WRF:3160
a = (((-1.4938)*ma_log((soot_num+dst_num)))+12.884);
// WRF:3161
b = (((-10.41)*ma_log((soot_num+dst_num)))-67.69);
// WRF:3162
regm = ((a*ma_log(wbar))+b);
// WRF:3165
if((tc>regm)) {
// WRF:3166
if(((tc<(-40.))&&(wbar>1.))) {
// WRF:3167
mskf_hf(tc,wbar,relhum,subgrid,so4_num,nihf); if(error) { arena.used=arena_saved; return; }
// WRF:3168
niimm = 0.;
// WRF:3169
nidep = 0.;
// WRF:3170
n1 = nihf;
// WRF:3171
} else {
// WRF:3172
mskf_hetero(tc,wbar,(soot_num+dst_num),niimm,nidep); if(error) { arena.used=arena_saved; return; }
// WRF:3173
nihf = 0.;
// WRF:3174
n1 = (niimm+nidep);
// WRF:3175
}
// WRF:3176
} else if((tc<(regm-5.))) {
// WRF:3177
mskf_hf(tc,wbar,relhum,subgrid,so4_num,nihf); if(error) { arena.used=arena_saved; return; }
// WRF:3178
niimm = 0.;
// WRF:3179
nidep = 0.;
// WRF:3180
n1 = nihf;
// WRF:3181
} else {
// WRF:3182
if(((tc<(-40.))&&(wbar>1.))) {
// WRF:3183
mskf_hf(tc,wbar,relhum,subgrid,so4_num,nihf); if(error) { arena.used=arena_saved; return; }
// WRF:3184
niimm = 0.;
// WRF:3185
nidep = 0.;
// WRF:3186
n1 = nihf;
// WRF:3187
} else {
// WRF:3188
mskf_hf((regm-5.),wbar,relhum,subgrid,so4_num,nihf); if(error) { arena.used=arena_saved; return; }
// WRF:3189
mskf_hetero(regm,wbar,(soot_num+dst_num),niimm,nidep); if(error) { arena.used=arena_saved; return; }
// WRF:3190
if((nihf<=((niimm+nidep)))) {
// WRF:3191
n1 = nihf;
// WRF:3192
} else {
// WRF:3193
n1 = (((niimm+nidep))*ma_pow(((((niimm+nidep))/nihf)),((((tc-regm))/5.))));
// WRF:3194
}
// WRF:3195
}
// WRF:3196
}
// WRF:3198
ni = n1;
// WRF:3200
}
// WRF:3201
}
// WRF:3202
ma_label_1100:;
// WRF:3209
if((((tc<0.)&&(tc>(-37.)))&&(qc>1.e-12))) {
// WRF:3215
if((deles>1.5f)) {
// WRF:3216
deles = 1.5f;
// WRF:3217
}
// WRF:3218
nimey = (1.e-3*ma_exp(((12.96*((deles-1.0)))-0.639)));
// WRF:3219
} else {
// WRF:3220
nimey = 0.;
// WRF:3221
}
// WRF:3223
nuci = (ni+nimey);
// WRF:3224
if(((nuci>9999.)||(nuci<0.))) {
// WRF:3227
nuci = 0.;
// WRF:3228
error=32; { arena.used=arena_saved; return; }
// WRF:3229
}
// WRF:3231
nuci = ((nuci*1.e+6)/rhoair);
// WRF:3232
onimey = ((nimey*1.e+6)/rhoair);
// WRF:3233
onidep = ((nidep*1.e+6)/rhoair);
// WRF:3234
oniimm = ((niimm*1.e+6)/rhoair);
// WRF:3235
onihf = ((nihf*1.e+6)/rhoair);
// WRF:3240
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ void mskf_hetero(double t,double ww,double ns,double& nis,double& nid) {
int arena_saved=arena.used;
double a11;
double a12;
double a21;
double a22;
double b11;
double b12;
double b21;
double b22;
double a;
double b;
double c;
// WRF:3255
a11 = 0.0263;
// WRF:3256
a12 = (-0.0185);
// WRF:3257
a21 = 2.758;
// WRF:3258
a22 = 1.3221;
// WRF:3259
b11 = (-0.008);
// WRF:3260
b12 = (-0.0468);
// WRF:3261
b21 = (-0.2667);
// WRF:3262
b22 = (-1.4588);
// WRF:3266
b = ((((a11+(b11*ma_log(ns))))*ma_log(ww))+((a12+(b12*ma_log(ns)))));
// WRF:3267
c = (a21+(b21*ma_log(ns)));
// WRF:3269
nis = (((ma_exp(a22)*ma_pow(ns,b22))*ma_exp((b*t)))*ma_pow(ww,c));
// WRF:3270
nis = ma_min(nis,ns);
// WRF:3272
nid = 0.0;
// WRF:3274
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ void mskf_hf(double t,double ww,double rh,double subgrid,double na,double& ni) {
int arena_saved=arena.used;
double a1_fast;
double a21_fast;
double a22_fast;
double b1_fast;
double b21_fast;
double b22_fast;
double a2_fast;
double b2_fast;
double c1_fast;
double c2_fast;
double k1_fast;
double k2_fast;
double a1_slow;
double a2_slow;
double b1_slow;
double b2_slow;
double b3_slow;
double c1_slow;
double c2_slow;
double k1_slow;
double k2_slow;
double regm;
double a;
double b;
double c;
double rhw;
// WRF:3296
a1_fast = 0.0231;
// WRF:3297
a21_fast = (-1.6387);
// WRF:3298
a22_fast = (-6.045);
// WRF:3299
b1_fast = (-0.008);
// WRF:3300
b21_fast = (-0.042);
// WRF:3301
b22_fast = (-0.112);
// WRF:3302
c1_fast = 0.0739;
// WRF:3303
c2_fast = 1.2372;
// WRF:3305
a1_slow = (-0.3949);
// WRF:3306
a2_slow = 1.282;
// WRF:3307
b1_slow = (-0.0156);
// WRF:3308
b2_slow = 0.0111;
// WRF:3309
b3_slow = 0.0217;
// WRF:3310
c1_slow = 0.120;
// WRF:3311
c2_slow = 2.312;
// WRF:3313
ni = 0.0;
// WRF:3319
a = ((6.0e-4*ma_log(ww))+6.6e-3);
// WRF:3320
b = ((6.0e-2*ma_log(ww))+1.052);
// WRF:3321
c = ((1.68*ma_log(ww))+129.35);
// WRF:3322
rhw = ((((((a*t)*t)+(b*t))+c))*0.01);
// WRF:3324
if((((t<=(-37.0)))&&((((rh*subgrid))>=rhw)))) {
// WRF:3327
regm = ((6.07*ma_log(ww))-55.0);
// WRF:3329
if((t>=regm)) {
// WRF:3331
if((t>(-64.0))) {
// WRF:3332
a2_fast = a21_fast;
// WRF:3333
b2_fast = b21_fast;
// WRF:3334
} else {
// WRF:3335
a2_fast = a22_fast;
// WRF:3336
b2_fast = b22_fast;
// WRF:3337
}
// WRF:3339
k1_fast = ma_exp(((a2_fast+(b2_fast*t))+(c2_fast*ma_log(ww))));
// WRF:3340
k2_fast = ((a1_fast+(b1_fast*t))+(c1_fast*ma_log(ww)));
// WRF:3342
ni = (k1_fast*ma_pow(na,(k2_fast)));
// WRF:3343
ni = ma_min(ni,na);
// WRF:3344
} else {
// WRF:3346
k1_slow = ma_exp(((a2_slow+(((b2_slow+(b3_slow*ma_log(ww))))*t))+(c2_slow*ma_log(ww))));
// WRF:3347
k2_slow = ((a1_slow+(b1_slow*t))+(c1_slow*ma_log(ww)));
// WRF:3349
ni = (k1_slow*ma_pow(na,(k2_slow)));
// WRF:3350
ni = ma_min(ni,na);
// WRF:3351
}
// WRF:3352
}
// WRF:3354
{ arena.used=arena_saved; return; }
arena.used=arena_saved;
}
__device__ double mskf_polysvp(double t,int type) {
int arena_saved=arena.used;
double retval;
double dum;
// WRF:3373
if((type==1)) {
// WRF:3377
retval = (ma_pow(10.,((((((-9.09718)*(((273.16/t)-1.)))-(3.56654*ma_log10((273.16/t))))+(0.876793*((1.-(t/273.16)))))+ma_log10(6.1071))))*100.);
// WRF:3381
}
// WRF:3386
if((type==0)) {
// WRF:3387
retval = (ma_pow(10.,(((((((-7.90298)*(((373.16/t)-1.)))+(5.02808*ma_log10((373.16/t))))-(1.3816e-7*((ma_pow(10.,((11.344*((1.-(t/373.16))))))-1.))))+(8.1328e-3*((ma_pow(10.,(((-3.49149)*(((373.16/t)-1.)))))-1.))))+ma_log10(1013.246))))*100.);
// WRF:3392
}
arena.used=arena_saved; return retval;
}
__device__ void calerf_r8(double arg,double& result,int jint) {
int arena_saved=arena.used;
int i;
double x;
double y;
double ysq;
double xnum;
double xden;
double del;
const double zero=0.0e0;
const double four=4.0e0;
const double one=1.0e0;
const double half=0.5e0;
const double two=2.0e0;
const double sqrpi=5.6418958354775628695e-1;
const double thresh=0.46875e0;
const double sixten=16.0e0;
const double xinf=1.79e308;
const double xneg=(-26.628e0);
const double xsmall=1.11e-16;
const double xbig=26.543e0;
const double xhuge=6.71e7;
const double xmax=2.53e307;
MaArray<double> a=arena.make<double>(1,5);
a.at(0)=3.16112374387056560e00;
a.at(1)=1.13864154151050156e02;
a.at(2)=3.77485237685302021e02;
a.at(3)=3.20937758913846947e03;
a.at(4)=1.85777706184603153e-1;
MaArray<double> b=arena.make<double>(1,4);
b.at(0)=2.36012909523441209e01;
b.at(1)=2.44024637934444173e02;
b.at(2)=1.28261652607737228e03;
b.at(3)=2.84423683343917062e03;
MaArray<double> c=arena.make<double>(1,9);
c.at(0)=5.64188496988670089e-1;
c.at(1)=8.88314979438837594e00;
c.at(2)=6.61191906371416295e01;
c.at(3)=2.98635138197400131e02;
c.at(4)=8.81952221241769090e02;
c.at(5)=1.71204761263407058e03;
c.at(6)=2.05107837782607147e03;
c.at(7)=1.23033935479799725e03;
c.at(8)=2.15311535474403846e-8;
MaArray<double> d=arena.make<double>(1,8);
d.at(0)=1.57449261107098347e01;
d.at(1)=1.17693950891312499e02;
d.at(2)=5.37181101862009858e02;
d.at(3)=1.62138957456669019e03;
d.at(4)=3.29079923573345963e03;
d.at(5)=4.36261909014324716e03;
d.at(6)=3.43936767414372164e03;
d.at(7)=1.23033935480374942e03;
MaArray<double> p=arena.make<double>(1,6);
p.at(0)=3.05326634961232344e-1;
p.at(1)=3.60344899949804439e-1;
p.at(2)=1.25781726111229246e-1;
p.at(3)=1.60837851487422766e-2;
p.at(4)=6.58749161529837803e-4;
p.at(5)=1.63153871373020978e-2;
MaArray<double> q=arena.make<double>(1,5);
q.at(0)=2.56852019228982242e00;
q.at(1)=1.87295284992346047e00;
q.at(2)=5.27905102951428412e-1;
q.at(3)=6.05183413124413191e-2;
q.at(4)=2.33520497626869185e-3;
// WRF:248
x = arg;
// WRF:249
y = ma_abs(x);
// WRF:250
if((y<=thresh)) {
// WRF:254
ysq = zero;
// WRF:255
if((y>xsmall)) { ysq = (y*y); }
// WRF:256
xnum = (a(5)*ysq);
// WRF:257
xden = ysq;
// WRF:258
for(i=1; i<=3; i+=1) {
// WRF:259
xnum = (((xnum+a(i)))*ysq);
// WRF:260
xden = (((xden+b(i)))*ysq);
// WRF:261
}
// WRF:262
result = ((x*((xnum+a(4))))/((xden+b(4))));
// WRF:263
if((jint!=0)) { result = (one-result); }
// WRF:264
if((jint==2)) { result = (ma_exp(ysq)*result); }
// WRF:265
goto ma_label_80;
// WRF:266
} else if((y<=four)) {
// WRF:270
xnum = (c(9)*y);
// WRF:271
xden = y;
// WRF:272
for(i=1; i<=7; i+=1) {
// WRF:273
xnum = (((xnum+c(i)))*y);
// WRF:274
xden = (((xden+d(i)))*y);
// WRF:275
}
// WRF:276
result = (((xnum+c(8)))/((xden+d(8))));
// WRF:277
if((jint!=2)) {
// WRF:278
ysq = (ma_aint((y*sixten))/sixten);
// WRF:279
del = (((y-ysq))*((y+ysq)));
// WRF:280
result = ((ma_exp(((-ysq)*ysq))*ma_exp((-del)))*result);
// WRF:281
}
// WRF:282
} else {
// WRF:286
result = zero;
// WRF:287
if((y>=xbig)) {
// WRF:288
if((((jint!=2))||((y>=xmax)))) { goto ma_label_30; }
// WRF:289
if((y>=xhuge)) {
// WRF:290
result = (sqrpi/y);
// WRF:291
goto ma_label_30;
// WRF:292
}
// WRF:293
}
// WRF:294
ysq = (one/((y*y)));
// WRF:295
xnum = (p(6)*ysq);
// WRF:296
xden = ysq;
// WRF:297
for(i=1; i<=4; i+=1) {
// WRF:298
xnum = (((xnum+p(i)))*ysq);
// WRF:299
xden = (((xden+q(i)))*ysq);
// WRF:300
}
// WRF:301
result = ((ysq*((xnum+p(5))))/((xden+q(5))));
// WRF:302
result = (((sqrpi-result))/y);
// WRF:303
if((jint!=2)) {
// WRF:304
ysq = (ma_aint((y*sixten))/sixten);
// WRF:305
del = (((y-ysq))*((y+ysq)));
// WRF:306
result = ((ma_exp(((-ysq)*ysq))*ma_exp((-del)))*result);
// WRF:307
}
// WRF:308
}
// WRF:309
ma_label_30:;
// WRF:313
if((jint==0)) {
// WRF:314
result = (((half-result))+half);
// WRF:315
if((x<zero)) { result = (-result); }
// WRF:316
} else if((jint==1)) {
// WRF:317
if((x<zero)) { result = (two-result); }
// WRF:318
} else {
// WRF:319
if((x<zero)) {
// WRF:320
if((x<xneg)) {
// WRF:321
result = xinf;
// WRF:322
} else {
// WRF:323
ysq = (ma_aint((x*sixten))/sixten);
// WRF:324
del = (((x-ysq))*((x+ysq)));
// WRF:325
y = (ma_exp((ysq*ysq))*ma_exp(del));
// WRF:326
result = (((y+y))-result);
// WRF:327
}
// WRF:328
}
// WRF:329
}
// WRF:330
ma_label_80:;
arena.used=arena_saved;
}
__device__ double cam_erf(double x) { double y;int j=0;calerf_r8(x,y,j);return y;}
__device__ double cam_erfc(double x) { double y;int j=1;calerf_r8(x,y,j);return y;}
};
/* Upstream notice retained verbatim.
!    Copyright(C) 1996 Takuya OOURA (email: ooura@mmm.t.u-tokyo.ac.jp).
!    You may use, copy, modify this code for any purpose and
!    without fee. You may distribute this ORIGINAL package.
*/
