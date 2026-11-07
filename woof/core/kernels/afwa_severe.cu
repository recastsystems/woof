// WRF 4.7.1 phys/module_diag_afwa.F, public-domain upstream pin f52c197.
// Native arithmetic and vertical order; independent resident GPU columns.
#ifdef GPUWM_WRF_EXACT
#define sv_exp gfk_exp
#define sv_pow gfk_pow
#else
#define sv_exp expf
#define sv_pow powf
#endif
__device__ float severe_rh(float p,float t,float qv) {
 float q=qv/(1.f+qv),qs=379.90516f/p*sv_exp(17.2693882f*(t-273.16f)/(t-35.86f));
 float rh=100.f*q/qs;if(rh>100.f)rh=100.f;else if(rh<1.f)rh=1.f;return rh;
}
extern "C" __global__ void afwa_severe_humidity(const float *p,const float *t,const float *q,float *rh,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)rh[i]=severe_rh(p[i],t[i],q[i]);
}
extern "C" __global__ void afwa_severe_driver(
 const float *u,const float *v,const float *z,const float *ht,
 const float *u10,const float *v10,const float *precip,
 float *gust,float *llmax,float *llws,float *wu,float *wd,
 float *tcoli,float *flux,float *uhmax,float *tor,
 float *oldrh,float *minrh,float *hail,
 int reset,float dt,int nz,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 if(reset){gust[c]=0.f;llmax[c]=0.f;wu[c]=0.f;wd[c]=0.f;
   tcoli[c]=0.f;flux[c]=0.f;uhmax[c]=0.f;tor[c]=0.f;
   oldrh[c]=minrh[c];minrh[c]=999.f;hail[c]=0.f;}
 float vel=sqrtf(u10[c]*u10[c]+v10[c]*v10[c]);
 float prate=(precip[c]/dt)*3600.f;
 if(prate>50.f){
   for(int k=1;k<nz;k++){int a=(k-1)*n+c,b=k*n+c;
    float h=z[b]-ht[c];if(h>=1000.f){
     // WRF divides by the upper AGL height, literally, not its difference.
     float uu=u[a]+(1000.f-(z[a]-ht[c]))*((u[b]-u[a])/h);
     float vv=v[a]+(1000.f-(z[a]-ht[c]))*((v[b]-v[a])/h);
     float f=fmaxf((150.f-prate)/100.f,0.f);
     float ub=u10[c]*f+uu*(1.f-f),vb=v10[c]*f+vv*(1.f-f);
     float blend=sqrtf(ub*ub+vb*vb);if(blend>vel)vel=blend;break;
    }}
 }
 if(vel>gust[c])gust[c]=vel;
 for(int k=1;k<nz;k++){int a=(k-1)*n+c,b=k*n+c;float h=z[b]-ht[c];
  if(h>=609.6f){
   float uu=u[a]+(609.6f-(z[a]-ht[c]))*((u[b]-u[a])/h)-u10[c];
   float vv=v[a]+(609.6f-(z[a]-ht[c]))*((v[b]-v[a])/h)-v10[c];
   float shear=sqrtf(uu*uu+vv*vv);llws[c]=shear;
   if(shear>llmax[c])llmax[c]=shear;break;
  }}
}
extern "C" __global__ void afwa_severe_column(
 const float *w,const float *t,const float *z,const float *p,
 const float *qi,const float *qs,const float *qg,const float *rho,const float *dz,
 const float *rh,const float *ht,const float *t2,const float *uh,
 const float *oldrh,float *minrh,float *wu,float *wd,float *hail,
 float *tcoli_max,float *flux_max,int nz,int n,const float *qv,int compute_rh) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 for(int k=1;k<nz;k++){int a=(k-1)*n+c,b=k*n+c;
  if(z[b]-ht[c]>=3500.f){
   // Only this pair of RH values reaches any native severe result.
   // Fast forecasts fuse the unchanged RH expressions into their consumer.
   float ra=compute_rh?severe_rh(p[a],t[a],qv[a]):rh[a];
   float rb=compute_rh?severe_rh(p[b],t[b],qv[b]):rh[b];
   float h=ra+(3500.f-(z[a]-ht[c]))*((rb-ra)/(z[b]-z[a]));
   if(h<minrh[c])minrh[c]=h;break;
  }}
 float up=0.f,dn=0.f,ice=0.f,flux=0.f,wn=-999.f;
 for(int k=0;k<nz;k++){int b=k*n+c;
  float mb=p[b]/100.f;
  if(mb>400.f&&w[b]>up){up=w[b];if(up>wu[c])wu[c]=up;}
  if(mb>400.f&&w[b]<dn){dn=w[b];if(dn<wd[c])wd[c]=dn;}
  ice=ice+((qi[b]+qs[b])+qg[b])*dz[b]*rho[b];
  if(t[b]<=258.15f&&wn==-999.f&&k>0&&qg[b]>1.e-20f){
   wn=w[b];float q=1000.f*qg[b];flux=q*wn;
  }
 }
 float melt=fmaxf(t2[c]-288.15f,0.f);
 float moisture=fmaxf(2.f*(fminf(minrh[c],oldrh[c])-70.f),0.f);
 float h=fmaxf((sv_pow(up/1.4f,1.1f)-melt)-moisture,0.f);
 h=h*(uh[c]/100.f+.25f);if(h>hail[c])hail[c]=h;
 if(ice>tcoli_max[c])tcoli_max[c]=ice;
 if(flux>flux_max[c])flux_max[c]=flux;
}
extern "C" __global__ void afwa_severe_tornado(
 const float *uh,const float *llws,const float *zlfc,
 const float *minrh,const float *oldrh,const float *mask_prev,const float *dur_prev,
 float *mask,float *dur,float *maximum,float dt,int ny,int nx,
 int west,int east,int south,int north,int periodic_x,int periodic_y) {
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=ny*nx;if(c>=n)return;
 int i=c%nx,j=c/nx;mask[c]=0.f;dur[c]=0.f;
 if(i<west||i>=nx-east||j<south||j>=ny-north)return;
 float tornado=0.f;
 if(zlfc[c]>=0.f){
  float a=fminf(fmaxf((uh[c]-25.f)/50.f,0.f),1.f);
  float b=fminf(fmaxf((llws[c]-2.f)/10.f,0.f),1.f);
  float d=fminf(fmaxf((3000.f-zlfc[c])/1500.f,0.f),1.f);
  float e=fminf(fmaxf((90.f-fminf(minrh[c],oldrh[c]))/30.f,0.f),1.f);
  tornado=(((50.f*a)*b)*d)*e;
 }
 if(tornado>0.f)mask[c]=1.f;
 float pm=0.f,pd=0.f;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  int yy=j+y,xx=i+x;
  if(periodic_x)xx=(xx+nx)%nx;if(periodic_y)yy=(yy+ny)%ny;
  if(yy<0||yy>=ny||xx<0||xx>=nx)continue;
  pm=fmaxf(pm,mask_prev[yy*nx+xx]);pd=fmaxf(pd,dur_prev[yy*nx+xx]);
 }
 if(mask[c]>.5f||pm>.5f)dur[c]=pd+dt;
 tornado=tornado*fminf(dur[c]/900.f,1.f);
 if(tornado>maximum[c])maximum[c]=tornado;
}
