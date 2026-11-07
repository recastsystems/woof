// Complete spatial WRF-v4 Miguez-Macho/Fan groundwater coupling.
// Original WRF public domain. Source pin: noahmp_options_wrf461_oracle.
namespace noahmp_water_options {
__constant__ float groundwater_klatfactor[19]={2,3,4,10,10,12,14,20,24,28,40,48,2,0,10,0,20,2,2};
__device__ float groundwater_river(const float*s,float wtd,float dt) {
 float rc=s[6];if(wtd>s[5]&&s[3]>s[5])rc=FMUL(rc,glibc_expf(FMUL(s[4],FSUB(wtd,s[3]))));
 return f_max(FDIV(FMUL(FMUL(rc,FSUB(wtd,s[5])),dt),s[1]),0.0f);
}
}
// Static rows: FDEPTH,AREA,TOPO,EQWTD,PEXP,RIVERBED,RIVERCOND,RECHCLIM,XLAND,XICE.
// Parameter rows: BEXP,DKSAT,negative PSISAT,SMCMAX,SMCWLT,DWSAT.
// State rows: WTD,SMCWTD,DEEPRECH,QSLAT,QRFS,QSPRINGS,RECH,QLAT,QRF,QSPRING.
extern "C" __global__ void k_groundwater_heads(const float*p,const float*s,const float*y,const int*soil,float*kh,int n) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10;float w=y[(size_t)c*10],k=0.0f;
 if(a[0]>0.0f){float kl=FMUL(p[(size_t)c*6+1],groundwater_klatfactor[soil[c]-1]);k=w<-1.5f?FMUL(FMUL(a[0],kl),glibc_expf(FDIV(FADD(w,1.5f),a[0]))):FMUL(kl,FADD(FADD(w,1.5f),a[0]));}
 kh[(size_t)c*2]=k;kh[(size_t)c*2+1]=FADD(a[2],w);
}
extern "C" __global__ void k_groundwater_lateral(const float*s,const float*kh,const int*land,float*y,int nx,int ny,float dt) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nx*ny)return;int i=c%nx,j=c/nx;float q=0.0f;
 if(i>0&&j>0&&i<nx-1&&j<ny-1&&land[c]){
  int offsets[8]={nx-1,-1,-nx-1,nx,-nx,nx+1,1,-nx+1};
  for(int k=0;k<8;++k){int nb=c+offsets[k];float f=FMUL(FADD(kh[(size_t)nb*2],kh[(size_t)c*2]),FSUB(kh[(size_t)nb*2+1],kh[(size_t)c*2+1]));if(k==0||k==2||k==5||k==7)f=FDIV(f,1.41421353816986083984375f);q=FADD(q,f);}
  q=FDIV(FMUL(FMUL(0.22754493f,q),dt),s[(size_t)c*10+1]);
 }
 y[(size_t)c*10+7]=q;
}
extern "C" __global__ void k_groundwater_update(const float*p,const float*s,const float*dz,const int*land,float*y,float*z,int n,float dt) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10;const float*b=p+(size_t)c*6;float*r=y+(size_t)c*10,*v=z+(size_t)c*12;
 float zsoil[5]={0.0f};for(int k=1;k<=4;++k)zsoil[k]=FSUB(zsoil[k-1],dz[k-1]);
 r[8]=land[c]?groundwater_river(a,r[0],dt):0.0f;
 if(land[c]){
  if(r[0]<FSUB(zsoil[4],dz[3])){
   float ddz=FSUB(zsoil[4],r[0]),mid=FMUL(0.5f,FADD(r[1],b[3]));
   float psi=FMUL(b[2],runoff_powf(FDIV(b[3],r[1]),b[0]));
   float wc=FMUL(b[1],runoff_powf(FDIV(mid,b[3]),FADD(FMUL(2.0f,b[0]),3.0f)));
   float flux=FMUL(FMUL(f_neg(dt),wc),FSUB(FDIV(FSUB(b[2],psi),ddz),1.0f));
   r[1]=FADD(r[1],FDIV(FSUB(r[2],flux),ddz));
   float plus=FMUL(f_max(FSUB(r[1],b[3]),0.0f),ddz),minus=FMUL(f_max(FSUB(1.0e-4f,r[1]),0.0f),ddz);
   r[1]=f_max(f_min(r[1],b[3]),1.0e-4f);r[2]=FSUB(FADD(flux,plus),minus);
  }
  float water=FADD(FSUB(r[7],r[8]),r[2]);
  groundwater_updatewtd(dz,zsoil,v+8,b[3],b[4],b[2],b[0],water,r[0],v,v+4,r[1],r[9]);
 }
 r[3]=FADD(r[3],FMUL(r[7],1000.0f));r[4]=FADD(r[4],FMUL(r[8],1000.0f));r[5]=FADD(r[5],FMUL(r[9],1000.0f));r[6]=FADD(r[6],FMUL(r[2],1000.0f));r[2]=0.0f;
}
extern "C" __global__ void k_groundwater_cold_adjust(float*y,const int*land,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n&&land[c]&&y[(size_t)c*10+7]>0.01f)y[(size_t)c*10]=f_min(FADD(y[(size_t)c*10],0.25f),0.0f);
}
extern "C" __global__ void k_groundwater_cold_river(float*s,const float*y,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;float*a=s+(size_t)c*10;const float*r=y+(size_t)c*10;
 a[3]=r[0];float ddz=FSUB(a[3],FSUB(a[5],a[2]));if(ddz<0.0f){a[5]=FADD(a[2],a[3]);ddz=0.0f;}
 float total=FDIV(FMUL(a[1],FADD(r[7],FMUL(a[7],0.001f))),31536000.0f);
 if(total>0.0f)a[6]=FDIV(total,f_max(ddz,0.05f));else{a[6]=0.01f;a[5]=FADD(a[2],a[3]);}
 a[5]=f_min(FSUB(a[5],a[2]),0.0f);
}
extern "C" __global__ void k_groundwater_cold_soil(const float*p,const float*s,const float*dz,const int*land,float*y,float*z,int n,float dt) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10,*b=p+(size_t)c*6;float*r=y+(size_t)c*10,*v=z+(size_t)c*12;
 float zsoil[4];zsoil[0]=f_neg(dz[0]);for(int k=1;k<4;++k)zsoil[k]=FSUB(zsoil[k-1],dz[k]);
 float qrf=land[c]?groundwater_river(a,r[0],dt):0.0f;
 groundwater_gw_cold(zsoil,dz,b[0],b[3],b[4],b[5],b[1],b[2],dt,r[7],qrf,r[0],v+8,v,v+4,r[1]);
 for(int k=2;k<10;++k)r[k]=0.0f;
}
