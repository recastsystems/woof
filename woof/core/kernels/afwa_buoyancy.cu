// WRF 4.7.1 module_diag_functions.F:89-351,438-834 and
// module_diag_afwa.F:619-674. Public domain; immutable oracle manifest.
#ifdef GPUWM_WRF_EXACT
#define bp_exp gfk_exp
#define bp_pow gfk_pow
#define bp_log gfk_log
#else
#define bp_exp expf
#define bp_pow powf
#define bp_log logf
#endif
__device__ float bp_lcl(float t,float rh) {
 float term1=1.f/(t-55.f);
 float term2=rh>0.f?bp_log(rh/100.f)/2840.f:bp_log(.001f/1.f)/2840.f;
 return 1.f/(term1-term2)+55.f;
}
__device__ float bp_theta(float t,float p){return t*bp_pow(1000.f/p,287.04f/1004.67f);}
__device__ float bp_thetae(float t,float p,float rh,float w) {
 float tlc=bp_lcl(t,rh);
 return (t*bp_pow(1000.f/p,(287.04f/1004.67f)*(1.f-(.28E-3f*w*1000.f))))
  *bp_exp(((3.376f/tlc)-.00254f)*(w*1000.f*(1.f+.81E-3f*w*1000.f)));
}
__device__ float bp_ws(float t,float p) {
 float es=6.122f*bp_exp((17.67f*(t-273.15f))/(t-29.66f));
 return (.622f*es)/((p/100.f)-es);
}
__device__ float bp_virtual(float t,float w){return t*(1.f+w/.622f)/(1.f+w);}
__device__ float bp_the2t(float thetae,float p) {
 float t=thetae/bp_exp(2.500E6f*.012f/(1004.67f*295.f))*bp_pow(p/100000.f,287.04f/1004.67f);
 for(int iter=1;iter<=105;++iter){
  float t2=t+1.f;
  float es=6.122f*bp_exp((17.67f*(t-273.15f))/(t-29.66f));
  float w=.622f*es/((p/100.f)-es);
  float es2=6.122f*bp_exp((17.67f*(t2-273.15f))/(t2-29.66f));
  float w2=.622f*es2/((p/100.f)-es2);
  float check=bp_thetae(t,p/100.f,100.f,w),check2=bp_thetae(t2,p/100.f,100.f,w2);
  if(fabsf(thetae-check)<.001f)break;
  t=t+(thetae-check)/(check2-check);
 }
 return t;
}
__device__ float bp_rh(float p,float t,float qv) {
 float q=qv/(1.f+qv),qs=379.90516f/p*bp_exp(17.2693882f*(t-273.16f)/(t-35.86f));
 float rh=100.f*q/qs;if(rh>100.f)rh=100.f;else if(rh<1.f)rh=1.f;return rh;
}

// Independent environmental thermodynamics are evaluated once per voxel.
extern "C" __global__ void afwa_buoy_environment(const float* t,const float* p,
 const float* q,float* work_w,int nz,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nz*n)return;
 float rh=bp_rh(p[i],t[i],q[i]),w=(rh/100.f)*bp_ws(t[i],p[i]);
 work_w[i]=w;work_w[(size_t)nz*n+i]=bp_thetae(t[i],p[i]/100.f,rh,w);
}
// The native most-unstable search remains in its original vertical order.
extern "C" __global__ void afwa_buoy_source(const float* t,const float* p,
 const float* q,const float* work_w,float* meta,int nz,int n) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=2*n)return;
 int c=x%n,parcel=x/n;size_t base=(size_t)parcel*6*n+c;
 #define AT(a,k) a[(size_t)(k)*n+c]
 #define M(k) meta[base+(size_t)(k)*n]
 int src=0;float st=AT(t,0),sp=AT(p,0),rh=bp_rh(sp,st,AT(q,0)),sw=AT(work_w,0);
 float theta=bp_theta(st,sp/100.f),e=bp_thetae(st,sp/100.f,rh,sw);
 for(int k=0;k<nz;++k){if(AT(p,k)<=50000.f)break;
  float et=work_w[(size_t)nz*n+(size_t)k*n+c];
  if(parcel==1&&et>e){src=k;st=AT(t,k);sp=AT(p,k);rh=bp_rh(sp,st,AT(q,k));sw=AT(work_w,k);theta=bp_theta(st,sp/100.f);e=et;}
 }
 float te=bp_thetae(st,sp/100.f,rh,sw);
 float tlcl=bp_lcl(AT(t,src),bp_rh(AT(p,src),AT(t,src),AT(q,src)));
 float pcl=AT(p,src)*bp_pow(tlcl/AT(t,src),1004.67f/287.058f);
 int first=nz;for(int k=src;k<nz;++k){if(AT(p,k)<=pcl){first=k;break;}}
 M(0)=(float)src;M(1)=theta;M(2)=te;M(3)=sw;M(4)=pcl;M(5)=(float)first;
 #undef AT
 #undef M
}
// Each moist parcel level has its own Newton iteration. It has no dependency
// on the solution at another level; only final integrals must stay ordered.
extern "C" __global__ void afwa_buoy_lift(const float* t,const float* p,
 const float* work_w,const float* meta,float* work_dtv,float* work_b,int nz,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*nz*n)return;
 int c=i%n,k=(i/n)%nz,parcel=i/(nz*n);size_t base=(size_t)parcel*6*n+c;
 #define M(j) meta[base+(size_t)(j)*n]
 if(k<(int)M(0)){work_dtv[i]=0.f;work_b[i]=0.f;return;}
 float pk=p[(size_t)k*n+c],pt,ptv,tv;
 float sw=M(3);float ew=work_w[(size_t)k*n+c];
 if(pk<=M(4)&&k>(int)M(5)){
  pt=bp_the2t(M(2),pk);float pw=bp_ws(pt,pk);
  ptv=bp_virtual(pt,pw);tv=bp_virtual(t[(size_t)k*n+c],ew);
  float freeze=.033f*(263.15f-ptv);if(freeze>1.f)freeze=1.f;if(freeze<0.f)freeze=0.f;
  freeze=freeze*333700.f*(sw-pw)/1005.7f;ptv=ptv-ptv*(sw-pw)+freeze;
 }else{pt=M(1)/bp_pow(100000.f/pk,287.058f/1004.67f);ptv=bp_virtual(pt,sw);tv=bp_virtual(t[(size_t)k*n+c],ew);}
 float d=ptv-tv;work_dtv[i]=d;work_b[i]=9.80665f*(d/tv);
 #undef M
}
extern "C" __global__ void afwa_buoy_integrate(const float* p,const float* z,const float* ht,
 const float* meta,const float* work_dtv,const float* work_b,
 float* cape,float* cin,float* zlfc,float* plfc,float* lidx,float* capemu,float* cinmu,int nz,int n) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=2*n)return;
 int c=x%n,parcel=x/n==0?3:1;size_t base=(size_t)(x/n)*nz*n+c;
 #define AT(a,k) a[(size_t)(k)*n+c]
 #define D(k) work_dtv[base+(size_t)(k)*n]
 #define B(k) work_b[base+(size_t)(k)*n]
 const float rd=287.058f,cp=1004.67f,undef=-9.999E30f;
 float ca=0.f,ci=undef,zl=undef,pl=undef,li=nanf("");float pcl=meta[(size_t)(x/n)*6*n+4*n+c];
 bool flag=false;int lfc=-1,el=-1;
 for(int k=0;k<nz;++k){
  if(!flag&&B(k)>0.f&&AT(p,k)<=pcl){flag=true;lfc=k;}
  if(k>=1&&flag&&B(k)<0.f&&B(k-1)>=0.f)el=k;
  if(B(k)<0.f&&flag)flag=false;
 }
 if(el>=0&&lfc>=0){pl=AT(p,lfc);float pel=AT(p,el);ci=0.f;
  for(int k=1;k<nz;++k){
   float pk=AT(p,k),dz=AT(z,k)-AT(z,k-1);
   if(pk<=pcl&&pk>pl)ci=ci+fminf(B(k),0.f)*dz;
   if(pk<=pl&&pk>pel)ca=ca+fmaxf(B(k),0.f)*dz;
  }
 }
 for(int k=1;k<nz;++k){float pu=AT(p,k),pd=AT(p,k-1),pm=50000.f;
  if(pd<=pm){li=0.f;break;}
  else if(pu<=pm&&pd>pm){float lu=-D(k)*bp_pow(pu/100000.f,rd/cp),ld=-D(k-1)*bp_pow(pd/100000.f,rd/cp);li=(lu*(pm-pd)+ld*(pu-pm))/(pu-pd);break;}
 }
 if(lfc>=0){pl=AT(p,lfc);zl=AT(z,lfc);}
 if(isnan(pl)||pl<0.f){pl=-1.f;zl=-1.f;}if(isnan(ca))ca=0.f;if(isnan(ci))ci=undef;
 if(parcel==3){cape[c]=ca;cin[c]=ci;plfc[c]=pl;zlfc[c]=zl>=ht[c]?zl-ht[c]:-1.f;lidx[c]=li!=999.f?li+273.15f:li;}
 else {capemu[c]=ca;cinmu[c]=ci;}
 #undef AT
 #undef D
 #undef B
}

extern "C" __global__ void afwa_severe_buoy_source(const float* t,const float* p,
 const float* q,const float* work_w,float* meta,int nz,int n) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;
 int c=x%n,parcel=x/n;size_t base=(size_t)parcel*6*n+c;
 #define AT(a,k) a[(size_t)(k)*n+c]
 #define M(k) meta[base+(size_t)(k)*n]
 int src=0;float st=AT(t,0),sp=AT(p,0),rh=bp_rh(sp,st,AT(q,0)),sw=AT(work_w,0);
 float theta=bp_theta(st,sp/100.f),e=bp_thetae(st,sp/100.f,rh,sw);
 for(int k=0;k<nz;++k){if(AT(p,k)<=50000.f)break;
  float et=work_w[(size_t)nz*n+(size_t)k*n+c];
  if(parcel==1&&et>e){src=k;st=AT(t,k);sp=AT(p,k);rh=bp_rh(sp,st,AT(q,k));sw=AT(work_w,k);theta=bp_theta(st,sp/100.f);e=et;}
 }
 float te=bp_thetae(st,sp/100.f,rh,sw);
 float tlcl=bp_lcl(AT(t,src),bp_rh(AT(p,src),AT(t,src),AT(q,src)));
 float pcl=AT(p,src)*bp_pow(tlcl/AT(t,src),1004.67f/287.058f);
 int first=nz;for(int k=src;k<nz;++k){if(AT(p,k)<=pcl){first=k;break;}}
 M(0)=(float)src;M(1)=theta;M(2)=te;M(3)=sw;M(4)=pcl;M(5)=(float)first;
 #undef AT
 #undef M
}

extern "C" __global__ void afwa_severe_buoy_lift(const float* t,const float* p,
 const float* work_w,const float* meta,float* work_dtv,float* work_b,int nz,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nz*n)return;
 int c=i%n,k=(i/n)%nz,parcel=i/(nz*n);size_t base=(size_t)parcel*6*n+c;
 #define M(j) meta[base+(size_t)(j)*n]
 if(k<(int)M(0)){work_dtv[i]=0.f;work_b[i]=0.f;return;}
 float pk=p[(size_t)k*n+c],pt,ptv,tv;
 float sw=M(3);float ew=work_w[(size_t)k*n+c];
 if(pk<=M(4)&&k>(int)M(5)){
  pt=bp_the2t(M(2),pk);float pw=bp_ws(pt,pk);
  ptv=bp_virtual(pt,pw);tv=bp_virtual(t[(size_t)k*n+c],ew);
  float freeze=.033f*(263.15f-ptv);if(freeze>1.f)freeze=1.f;if(freeze<0.f)freeze=0.f;
  freeze=freeze*333700.f*(sw-pw)/1005.7f;ptv=ptv-ptv*(sw-pw)+freeze;
 }else{pt=M(1)/bp_pow(100000.f/pk,287.058f/1004.67f);ptv=bp_virtual(pt,sw);tv=bp_virtual(t[(size_t)k*n+c],ew);}
 float d=ptv-tv;work_dtv[i]=d;work_b[i]=9.80665f*(d/tv);
 #undef M
}

extern "C" __global__ void afwa_severe_buoy_integrate(const float* p,const float* z,const float* ht,
 const float* meta,const float* work_dtv,const float* work_b,
 float* cape,float* cin,float* zlfc,float* plfc,float* lidx,float* capemu,float* cinmu,int nz,int n) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;
 int c=x%n,parcel=x/n==0?3:1;size_t base=(size_t)(x/n)*nz*n+c;
 #define AT(a,k) a[(size_t)(k)*n+c]
 #define D(k) work_dtv[base+(size_t)(k)*n]
 #define B(k) work_b[base+(size_t)(k)*n]
 const float rd=287.058f,cp=1004.67f,undef=-9.999E30f;
 float ca=0.f,ci=undef,zl=undef,pl=undef,li=nanf("");float pcl=meta[(size_t)(x/n)*6*n+4*n+c];
 bool flag=false;int lfc=-1,el=-1;
 for(int k=0;k<nz;++k){
  if(!flag&&B(k)>0.f&&AT(p,k)<=pcl){flag=true;lfc=k;}
  if(k>=1&&flag&&B(k)<0.f&&B(k-1)>=0.f)el=k;
  if(B(k)<0.f&&flag)flag=false;
 }
 if(el>=0&&lfc>=0){pl=AT(p,lfc);float pel=AT(p,el);ci=0.f;
  for(int k=1;k<nz;++k){
   float pk=AT(p,k),dz=AT(z,k)-AT(z,k-1);
   if(pk<=pcl&&pk>pl)ci=ci+fminf(B(k),0.f)*dz;
   if(pk<=pl&&pk>pel)ca=ca+fmaxf(B(k),0.f)*dz;
  }
 }
 for(int k=1;k<nz;++k){float pu=AT(p,k),pd=AT(p,k-1),pm=50000.f;
  if(pd<=pm){li=0.f;break;}
  else if(pu<=pm&&pd>pm){float lu=-D(k)*bp_pow(pu/100000.f,rd/cp),ld=-D(k-1)*bp_pow(pd/100000.f,rd/cp);li=(lu*(pm-pd)+ld*(pu-pm))/(pu-pd);break;}
 }
 if(lfc>=0){pl=AT(p,lfc);zl=AT(z,lfc);}
 if(isnan(pl)||pl<0.f){pl=-1.f;zl=-1.f;}if(isnan(ca))ca=0.f;if(isnan(ci))ci=undef;
 if(parcel==3){cape[c]=ca;cin[c]=ci;plfc[c]=pl;zlfc[c]=zl>=0.f?zl-ht[c]:-1.f;lidx[c]=li!=999.f?li+273.15f:li;}
 else {capemu[c]=ca;cinmu[c]=ci;}
 #undef AT
 #undef D
 #undef B
}
