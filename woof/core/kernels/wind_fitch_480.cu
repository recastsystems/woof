// WRF v4.8.0 phys/module_wind_fitch.F (06d4240ae989), the original Fitch branch
// (windfarm_induction=0) with the THRUST diagnostic 4.8.0 adds. Public domain, plus
// the Fraunhofer IWES MIT notice for the THRUST lines (tools/wind_mav_wrf471_oracle).
// Arithmetic and evaluation order follow dragforce/dragcof line for line. The
// windfarm_induction=1 branch is fitch_induction_480 below.
// One block per occupied column; turbines sharing a cell accumulate in file order.
// REAL**3. compiles to glibc powf(x,3.0f) under the strict recipe (GCC folds only
// **2. to x*x without -funsafe-math), so the cubes call glibc's own powf word;
// (x*x)*x differs by one ulp on real captures. Fast numerics uses the product.
#ifdef GPUWM_WIND480_FAST
#define f480_cube(x) (((x)*(x))*(x))
#else
#define f480_cube(x) gfk_pow((x),3.f)
#endif
__device__ __forceinline__ float f480_area(float z1,float z2,float diameter,float density){
 float r=diameter/2.f;z1=r-z1;z2=r-z2;float a=fabsf(z1),b=fabsf(z2),v;
 float aa=a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r);
 float bb=b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r);
 if(z1>0.f && z2>0.f)v=aa-bb;
 else if(z1<0.f && z2<0.f)v=bb-aa;
 else v=b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r)+a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r);
 return v*density;
}
// dragcof; the BAMS thrcof clamp applies only with induction_correction=1.
__device__ __forceinline__ void f480_drag(float s,int t,int nt,const int* nv,const float* ws,const float* tc,const float* pw,
 const float* diam,const float* stc,const float* stc2,float factor,int clamp,float& tk,float& pc,float& ct){
 int n=nv[t],k=1;for(;k<n;++k)if(ws[k*nt+t]>s)break;
 if(s<=ws[t])ct=stc[t];else if(s>=ws[(n-1)*nt+t])ct=stc2[t];
 else ct=tc[(k-1)*nt+t]+(tc[k*nt+t]-tc[(k-1)*nt+t])/(ws[k*nt+t]-ws[(k-1)*nt+t])*(s-ws[(k-1)*nt+t]);
 if(clamp)ct=fmaxf(fminf(ct,.9999f),.0001f);  // BAMS clamp: induction branch only
 pc=0.f;
 if(s>ws[t] && s<ws[(n-1)*nt+t]){
  float area=(3.14159265358979323846f/4.f)*(diam[t]*diam[t]);
  float x=ws[(k-1)*nt+t],y=ws[k*nt+t];
  float f1=1000.f/(.5f*1.23f*f480_cube(x)*area),f2=1000.f/(.5f*1.23f*f480_cube(y)*area);
  pc=pw[(k-1)*nt+t]*f1+(pw[k*nt+t]*f2-pw[(k-1)*nt+t]*f1)/(y-x)*(s-x);
 }
 tk=factor*(ct-pc);if(tk<0.f)tk=0.f;
}
// Same argument list as fitch_induction so the resident farm swaps entry points only.
extern "C" __global__ void fitch_original_480(
 const int* cols,const int* ptr,const int* ids,const float* diam,const float* hub,const int* nv,
 const float* ws,const float* tc,const float* pw,const float* stc,const float* stc2,
 const float* u,const float* v,const float* dz,const float* zw,
 float* du,float* dv,float* qke,float* adv,float* power,float* thrust,float* tke,float* tinc,
 float* hubout,float* turbine,int nc,int nz,int nt,float dx,float dt,float factor){
 int c=cols[blockIdx.x],k=threadIdx.x;extern __shared__ float ps[];
 float pk=0.f,tkall=0.f,thall=0.f;
 for(int it=ptr[c];it<ptr[c+1];++it){int t=ids[it];float lo=hub[t]-diam[t]/2.f,hi=hub[t]+diam[t]/2.f;
  int bot=-1,top=-2,kbot=0,ktop=0;float lower=0.f,z1=0.f,z2=0.f;
  for(int z=0;z<nz;++z){float upper=lower+dz[z*nc+c];
   if(lo>=lower && lo<=upper)bot=z;
   if(hi>=lower && hi<=upper){top=z;break;}lower=upper;
  }
  if(top<bot || bot<0)continue;
  lower=0.f;
  // Native: IF(hubheight .GE. z2) kbot=k ELSE ktop=k.
  for(int z=0;z<nz;++z){z2=lower+.5f*dz[z*nc+c];if(hub[t]>=z2)kbot=z;else{ktop=z;break;}z1=z2;lower=z2+.5f*dz[z*nc+c];}
  float area=(3.14159265358979323846f/4.f)*(diam[t]*diam[t]),sp;
  if(ktop==0)sp=__fsqrt_rn(u[c]*u[c]+v[c]*v[c])*hub[t]/z1;
  else{float s1=__fsqrt_rn(u[kbot*nc+c]*u[kbot*nc+c]+v[kbot*nc+c]*v[kbot*nc+c]);
   float s2=__fsqrt_rn(u[ktop*nc+c]*u[ktop*nc+c]+v[ktop*nc+c]*v[ktop*nc+c]);sp=s1+((s2-s1)/(z2-z1))*(hub[t]-z1);}
  float tk,pc,ct;f480_drag(sp,t,nt,nv,ws,tc,pw,diam,stc,stc2,factor,0,tk,pc,ct);
  float p1=.5f*1.23f*f480_cube(sp)*area*pc;
  float ta=0.f,speed=0.f,p2k=0.f;
  if(k<nz && k>=bot && k<=top){int p=k*nc+c;float a1=fmaxf(zw[p]-lo-zw[c],0.f),a2=fminf(zw[p+nc]-lo-zw[c],diam[t]);
   ta=f480_area(a1,a2,diam[t],1.f/(dx*dx));speed=__fsqrt_rn(u[p]*u[p]+v[p]*v[p]);
   p2k=.5f*pc*1.23f*f480_cube(speed)*ta/(1.f/(dx*dx));}
  if(k<nz)ps[k]=p2k;__syncthreads();float p2=0.f;for(int z=0;z<nz;++z)p2=p2+ps[z];
  float ec=(p1==0.f || p2==0.f)?1.f:p1/p2;
  if(k<nz && k>=bot && k<=top){int p=k*nc+c;
   float delta=f480_cube(speed)*ta*tk*dt/dz[p]*ec;qke[p]=qke[p]+delta;tkall=tkall+delta;
   du[p]=du[p]-.5f*u[p]*ct*speed*ta/dz[p]*ec;
   dv[p]=dv[p]-.5f*v[p]*ct*speed*ta/dz[p]*ec;}
  if(k==0){pk=p1+pk;thall=.5f*1.23f*(sp*sp)*area*ct+thall;hubout[10*t]=sp;turbine[t]=p1;}
  __syncthreads();
 }
 if(k<nz){tinc[k*nc+c]=tkall;tke[k*nc+c]=tkall/dt;adv[k*nc+c]=qke[k*nc+c];}
 if(k==0){power[c]=pk;thrust[c]=thall;}
}
// windfarm_induction=1 branch of the same 4.8.0 module (Fraunhofer IWES MIT lines).
// wind_mav.cu's fitch_induction (kept unchanged for the wrf_471 pin) spells the cubes
// (x*x)*x; native compiles REAL**3. to glibc powf, so this sibling differs only there
// (f480_cube).
__device__ double f480_atan_series(double x) {
#ifdef GPUWM_WIND480_FAST
    return double(atanf(float(x)));
#endif
    double ax=fabs(x),offset=0.0;
    bool inv=ax>1.0;
    if(inv) ax=1.0/ax;
    if(ax>0.4142135623730950488) {ax=(ax-1.0)/(ax+1.0);offset=0.7853981633974483096;}
    double term=ax,sum=ax,xx=ax*ax;
    for(int n=1;n<=36;++n) {term=-term*xx;sum=sum+term/double(2*n+1);}
    sum=offset+sum;
    if(inv) sum=1.5707963267948966192-sum;
    return copysign(sum,x);
}
__device__ __forceinline__ float f480_powi(float x,int n){float y=1.f;for(;;){if(n&1)y=y*x;n>>=1;if(!n)return y;x=x*x;}}
extern "C" __global__ void fitch_induction_480(
 const int* cols,const int* ptr,const int* ids,const float* diam,const float* hub,const int* nv,
 const float* ws,const float* tc,const float* pw,const float* stc,const float* stc2,
 const float* u,const float* v,const float* dz,const float* zw,
 float* du,float* dv,float* qke,float* adv,float* power,float* thrust,float* tke,float* tinc,
 float* hubout,float* turbine,int nc,int nz,int nt,float dx,float dt,float factor){
 int c=cols[blockIdx.x],k=threadIdx.x;extern __shared__ float ps[];
 float pk=0.f,tkall=0.f,thall=0.f;int count=ptr[c+1]-ptr[c];
 for(int it=ptr[c];it<ptr[c+1];++it){int t=ids[it];float lo=hub[t]-diam[t]/2.f,hi=hub[t]+diam[t]/2.f;
  int bot=-1,top=-2,kbot=0,ktop=0;float lower=0.f,z1=0.f,z2=0.f;
  for(int z=0;z<nz;++z){float upper=lower+dz[z*nc+c];
   if(lo>=lower && lo<=upper)bot=z;
   if(hi>=lower && hi<=upper){top=z;break;}lower=upper;
  }
  if(top<bot || bot<0)continue;
  lower=0.f;
  for(int z=0;z<nz;++z){z2=lower+.5f*dz[z*nc+c];if(hub[t]>=z2)kbot=z;else{ktop=z;break;}z1=z2;lower=z2+.5f*dz[z*nc+c];}
  float area=(3.14159265358979323846f/4.f)*(diam[t]*diam[t]);
  float uh=(u[ktop*nc+c]+u[kbot*nc+c])/2.f,vh=(v[ktop*nc+c]+v[kbot*nc+c])/2.f;
  float angle=__double2float_rn(f480_atan_series(double(vh/uh)));
  float a=.25f*area/(diam[t]*dx*fminf(fabsf(1.f/glibc_cosf(angle)),fabsf(1.f/glibc_sinf(angle)))),base;
  if(ktop==0)base=__fsqrt_rn(u[c]*u[c]+v[c]*v[c])*hub[t]/z1;
  else{float s1=__fsqrt_rn(u[kbot*nc+c]*u[kbot*nc+c]+v[kbot*nc+c]*v[kbot*nc+c]);
   float s2=__fsqrt_rn(u[ktop*nc+c]*u[ktop*nc+c]+v[ktop*nc+c]*v[ktop*nc+c]);base=s1+((s2-s1)/(z2-z1))*(hub[t]-z1);}
  float sp=base/f480_powi(1.f-a,count),tk,pc,ct;
  f480_drag(sp,t,nt,nv,ws,tc,pw,diam,stc,stc2,factor,1,tk,pc,ct);
  a=(.5f*(1.f-__fsqrt_rn(1.f-ct)))*area/(diam[t]*dx*fminf(fabsf(1.f/glibc_cosf(angle)),fabsf(1.f/glibc_sinf(angle))));
  float scale=f480_powi(1.f-a,count);sp=base/scale;
  f480_drag(sp,t,nt,nv,ws,tc,pw,diam,stc,stc2,factor,1,tk,pc,ct);
  float p1=.5f*1.23f*f480_cube(sp)*area*pc;
  float ta=0.f,speed=0.f,p2k=0.f;
  if(k<nz && k>=bot && k<=top){int p=k*nc+c;float a1=fmaxf(zw[p]-lo-zw[c],0.f),a2=fminf(zw[p+nc]-lo-zw[c],diam[t]);
   ta=f480_area(a1,a2,diam[t],1.f/(dx*dx));speed=__fsqrt_rn(u[p]*u[p]+v[p]*v[p])/scale;
   p2k=.5f*pc*1.23f*f480_cube(speed)*ta/(1.f/(dx*dx));}
  if(k<nz)ps[k]=p2k;__syncthreads();float p2=0.f;for(int z=0;z<nz;++z)p2=p2+ps[z];
  float ec=(p1==0.f || p2==0.f)?1.f:p1/p2;
  if(k<nz && k>=bot && k<=top){int p=k*nc+c;
   float delta=f480_cube(speed)*ta*tk*dt/dz[p]*ec;qke[p]=qke[p]+delta;tkall=tkall+delta;
   du[p]=du[p]-.5f*(u[p]/scale)*ct*speed*ta/dz[p]*ec;
   dv[p]=dv[p]-.5f*(v[p]/scale)*ct*speed*ta/dz[p]*ec;}
  if(k==0){pk=p1+pk;thall=.5f*1.23f*(sp*sp)*area*ct+thall;hubout[10*t]=sp;turbine[t]=p1;}
  __syncthreads();
 }
 if(k<nz){tinc[k*nc+c]=tkall;tke[k*nc+c]=tkall/dt;adv[k*nc+c]=qke[k*nc+c];}
 if(k==0){power[c]=pk;thrust[c]=thall;}
}
