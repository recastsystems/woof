#ifndef GPUWM_WRF_EXACT
#define glibc_sinf sinf
#define glibc_cosf cosf
#define glibc_tanf tanf
#define glibc_atanf atanf
#define glibc_asinf asinf
#endif
// WRF 4.6.1 phys/module_ra_eclipse.F, public-domain eclipse geometry.
// Float operations follow the scalar Fortran statements. shared math helpers
// are provided by the shared kernel loader; no fast-math or FTZ is used.
// Explicit division preserves WRF rounding at constant divisors too.
__device__ void eclipse_point(const float *b, float t1, float lat, float xlon,
                             float *obscur, int *mask, float *track) {
  const float pi=3.1415926535897932384626433f, epsil=0.0818192f;
  const float degrad=pi/180.0f;
  float t=(t1+__fdiv_rn(b[31],3600.0f))-b[2];
  float value[6]={0,0,0,0,0,0};
  for(int i=0;i<4;i++) {
    float power = i==0 ? 1.0f : (i==1 ? t : (i==2 ? t*t : (t*t)*t));
    for(int q=0;q<6;q++) value[q]=value[q]+b[5+4*q+i]*power;
  }
  float cx=value[0],cy=value[1],cd=value[2],cl1=value[3],cl2=value[4],cmu=value[5];
  float tanf1=b[29],tanf2=b[30],Dt=b[31];
  float rmu=cmu*degrad,rd=cd*degrad;
  float E=__fsqrt_rn(1.0f-epsil*epsil),E_1=__fdiv_rn(1.0f,E);
  float tan_rd1=E_1*glibc_tanf(rd),rd1=glibc_atanf(tan_rd1);
  float rho1=__fdiv_rn(glibc_sinf(rd),glibc_sinf(rd1)),cy1=__fdiv_rn(cy,rho1);
  float tan_rd2=E*glibc_tanf(rd),rd2=glibc_atanf(tan_rd2);
  float rho2=__fdiv_rn(glibc_cosf(rd),glibc_cosf(rd2));
  float tan_gamma=__fdiv_rn(cx,cy1),cgamma=glibc_atanf(tan_gamma);
  // gfortran-12 -O2 rewrites SIN(ATAN(x)) even with -fno-fast-math.
  // The native capture build's optimized tree is retained with the oracle.
  float sg = fabsf(tan_gamma) < 1.844674297419792384e19f
      ? __fdiv_rn(tan_gamma,__fsqrt_rn(tan_gamma*tan_gamma+1.0f)) : 1.0f;
  float sin_beta=__fdiv_rn(cx,sg);
  track[0]=1e30f; track[1]=1e30f;
  if(fabsf(sin_beta)<1.0f) {
    float beta=glibc_asinf(sin_beta);
    float tan_C=__fdiv_rn(cy1,glibc_cosf(beta)),C=glibc_atanf(tan_C);
    float cm=__fdiv_rn(cy1,glibc_sinf(C));
    float sin_lat1=cm*glibc_sinf(C+rd1);
    float cos_lat1=__fsqrt_rn(1.0f-sin_lat1*sin_lat1);
    float tan_lat1=__fdiv_rn(sin_lat1,cos_lat1),tan_lat=E_1*tan_lat1;
    track[0]=__fdiv_rn(glibc_atanf(tan_lat),degrad);
    float sin_H=__fdiv_rn(cx,cos_lat1),cos_H=__fdiv_rn(cm*glibc_cosf(C+rd1),cos_lat1);
    float tan_H=__fdiv_rn(sin_H,cos_H),H=glibc_atanf(tan_H);
    if(cos_H<0.0f) H=sin_H>=0.0f ? pi-fabsf(H) : pi+fabsf(H);
    track[1]=__fdiv_rn(rmu-H,degrad);
    if(track[1]>360.0f) track[1]=track[1]-360.0f;
    if(track[1]>180.0f) track[1]=track[1]-360.0f;
    if(track[1]<=-180.0f) track[1]=track[1]+360.0f;
    track[1]=-track[1]+__fdiv_rn(1.002738f*15.0f*Dt,3600.0f);
  }
  float lon=-xlon-__fdiv_rn(1.002738f*15.0f*Dt,3600.0f);
  if(lon<0.0f) lon=lon+360.0f;
  if(lon>360.0f) lon=lon-360.0f;
  float rlat=lat*degrad,rlon=lon*degrad,H=rmu-rlon;
  float tmp=epsil*glibc_sinf(rlat);
  float A=__fsqrt_rn(1.0f-tmp*tmp);
  float cos_lat1=__fdiv_rn(glibc_cosf(rlat),A),sin_lat1=__fdiv_rn(glibc_sinf(rlat)*E,A);
  float xi=cos_lat1*glibc_sinf(H);
  float eta1=sin_lat1*glibc_cosf(rd1)*E-cos_lat1*glibc_sinf(rd1)*glibc_cosf(H);
  float zeta1=sin_lat1*glibc_sinf(rd2)*E+cos_lat1*glibc_cosf(rd2)*glibc_cosf(H);
  cy1=__fdiv_rn(cy,rho1);
  float dx=xi-cx,dy=eta1-cy1,DELTA2=dx*dx+dy*dy;
  float DELTA=__fsqrt_rn(DELTA2),LL1=cl1-zeta1*tanf1,LL2=cl2-zeta1*tanf2;
  if(DELTA>LL1) { *obscur=0.0f; *mask=0; }
  else if(DELTA<=LL1 && DELTA>fabsf(LL2)) { *obscur=__fdiv_rn(LL1-DELTA,LL1+LL2); *mask=1; }
  else if(DELTA<=fabsf(LL2)) {
    if(LL2<0.0f) { *obscur=1.0f; *mask=2; }
    else { *obscur=__fdiv_rn(LL1-DELTA,LL1+LL2); *mask=3; }
  }
  if(*obscur>1.0f) *obscur=1.0f;
}
#ifndef ECLIPSE_HOST_ORACLE
extern "C" __global__ void solar_eclipse(int n, const float *b, float t1,
  const float *lat,const float *lon,float *obscur,int *mask,float *elat,float *elon) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=n) return;
  float track[2];
  eclipse_point(b,t1,lat[p],lon[p],obscur+p,mask+p,track);
  if(p==0) { elat[0]=track[0]; elon[0]=track[1]; }
}
#endif
