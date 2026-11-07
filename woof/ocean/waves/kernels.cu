// Original CUDA implementation of the published spectral balance equations.
// Equations and units are documented in SCIENCE.md. No external model source.
#define PI 3.14159265358979323846f
#define G 9.80665f
#define R 6371000.f
#ifndef DIA_COEFFICIENT
#define DIA_COEFFICIENT 3e7f
#endif
__device__ float sq(float x){return x*x;}
__device__ float pos(float x){return fmaxf(x,0.f);}
__device__ int wrap(int j,int n){return (j%n+n)%n;}
__device__ float growth(float sig,float k,float cs,float us,float ze){
    if(cs<=0.f || us<=0.f)return 0.f;
    float c=sig/k;
    float r=.41f*c/(us*cs);
    // Evaluate the critical height in log space to avoid overflow.
    float loglam=logf(G*ze/(c*c))+r;
    if(loglam>=0.f || loglam < -80.f)return 0.f;
    float beta=(1.2f/(.41f*.41f))*expf(loglam)*sq(sq(loglam));
    return beta*(1.225f/1025.f)*sq(us/c)*sq(cs)*sig;
}
extern "C" __global__ void geometry(const float* depth,const float* lat,
 const float* freq,float* k,float* cg,float* turn,float* cfl,
 int nx,int ny,int nf,int nd,float dxdeg,float dydeg){
 int z=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nf*nd;if(z>=n)return;
 int d=z%nd,f=(z/nd)%nf,c=z/(nd*nf),x=c%nx,y=c/nx;
 float h=depth[c],s=2*PI*freq[f],a=2*PI*d/nd,phi=lat[y]*PI/180;
 if(h<=0){turn[z]=0;cfl[z]=0;return;}
 float kk=fmaxf(s*s/G,s/sqrtf(G*h));
 for(int it=0;it<12;it++){
   float t=tanhf(kk*h),se=1-t*t;
   kk=fmaxf(1e-8f,kk-(G*kk*t-s*s)/(G*(t+kk*h*se)));
 }
 float kh=kk*h,sh=kh<40 ? sinhf(2*kh):1e30f;
 float group=.5f*s/kk*(1+2*kh/sh);
 if(d==0){k[c*nf+f]=kk;cg[c*nf+f]=group;}
 float dx=R*cosf(phi)*dxdeg*PI/180,dy=R*dydeg*PI/180;
 int l=x?c-1:c,rr=x+1<nx?c+1:c,b=y?c-nx:c,t=y+1<ny?c+nx:c;
 float hl=depth[l]>0?depth[l]:h,hr=depth[rr]>0?depth[rr]:h;
 float hb=depth[b]>0?depth[b]:h,ht=depth[t]>0?depth[t]:h;
 float hx=(hr-hl)/((rr==c||l==c)?dx:2*dx);
 float hy=(ht-hb)/((t==c||b==c)?dy:2*dy);
 float ct=-group*cosf(a)*tanf(phi)/R+s/sh*(sinf(a)*hx-cosf(a)*hy);
 turn[z]=ct;
 // Directional turning is implicit: only spatial faces constrain this step.
 cfl[z]=2*group*(1/dx+1/dy);
}
__device__ float edgeflux(float velocity,float left,float right){
 return velocity*(velocity>=0?left:right);
}
extern "C" __global__ void propagate(const float* e,float* out,const float* depth,
 const float* lat,const float* cg,const float* turn,int nx,int ny,int nf,int nd,
 float dxdeg,float dydeg,float dt){
 int z=blockIdx.x*blockDim.x+threadIdx.x,ns=nf*nd,n=nx*ny*ns;if(z>=n)return;
 int d=z%nd,f=(z/nd)%nf,c=z/ns,x=c%nx,y=c/nx;
 if(depth[c]<=0){out[z]=0;return;}
 float phi=lat[y]*PI/180,a=2*PI*d/nd,cp=cg[c*nf+f];
 float dx=R*cosf(phi)*dxdeg*PI/180,dy=R*dydeg*PI/180,da=2*PI/nd;
 float fxp,fxm,fyp,fym;
 if(x+1<nx){int cc=c+1;float v=.5f*(cp+(depth[cc]>0?cg[cc*nf+f]:cp))*cosf(a);
 fxp=edgeflux(v,e[z],depth[cc]>0?e[z+ns]:0.f);}else fxp=edgeflux(cp*cosf(a),e[z],0.f);
 if(x){int cc=c-1;float v=.5f*(cp+(depth[cc]>0?cg[cc*nf+f]:cp))*cosf(a);
 fxm=edgeflux(v,depth[cc]>0?e[z-ns]:0.f,e[z]);}else fxm=edgeflux(cp*cosf(a),0.f,e[z]);
 float cn=cosf(phi+.5f*dydeg*PI/180),cs=cosf(phi-.5f*dydeg*PI/180);
 if(y+1<ny){int cc=c+nx;float v=.5f*(cp+(depth[cc]>0?cg[cc*nf+f]:cp))*sinf(a);
 fyp=cn*edgeflux(v,e[z],depth[cc]>0?e[z+nx*ns]:0.f);}else fyp=cn*edgeflux(cp*sinf(a),e[z],0.f);
 if(y){int cc=c-nx;float v=.5f*(cp+(depth[cc]>0?cg[cc*nf+f]:cp))*sinf(a);
 fym=cs*edgeflux(v,depth[cc]>0?e[z-nx*ns]:0.f,e[z]);}else fym=cs*edgeflux(cp*sinf(a),0.f,e[z]);
 out[z]=e[z]-dt*((fxp-fxm)/dx+(fyp-fym)/(dy*cosf(phi)));
}
// Conservative backward Euler in direction. Each CUDA thread owns one whole
// periodic directional ring, so the cyclic tridiagonal solve has no atomics or
// inter-thread dependencies. The upwind matrix is an M-matrix: positivity holds
// for arbitrary dt, and its column sums of one preserve directional variance.
#ifndef WAVES_DIRECTIONS
#define WAVES_DIRECTIONS 180
#endif
extern "C" __global__ void refract(float* e,const float* turn,const float* depth,
 int nc,int nf,int nd,float dt){
 int ring=blockIdx.x*blockDim.x+threadIdx.x;if(ring>=nc*nf)return;
 if(depth[ring/nf]<=0)return;
 int base=ring*nd;double h=dt/(2*PI/nd);
 double cp[WAVES_DIRECTIONS],yy[WAVES_DIRECTIONS],zz[WAVES_DIRECTIONS];
 double left0=.5f*(turn[base+nd-1]+turn[base]);
 double right0=.5f*(turn[base]+turn[base+1]);
 double diag0=1+h*(pos(right0)+pos(-left0));
 // Remove cyclic corners with a rank-one correction, solve two ordinary
 // tridiagonal systems together, then restore that correction exactly.
 double alpha=-h*pos(left0),beta=-h*pos(-left0),gamma=-diag0;
 for(int d=0;d<nd;d++){
  double left=.5f*(turn[base+wrap(d-1,nd)]+turn[base+d]);
  double right=.5f*(turn[base+d]+turn[base+wrap(d+1,nd)]);
  double lower=-h*pos(left),upper=-h*pos(-right);
  double diag=1+h*(pos(right)+pos(-left));
  if(d==0)diag-=gamma;
  if(d==nd-1)diag-=alpha*beta/gamma;
  double denom=diag-(d?lower*cp[d-1]:0.f);
  cp[d]=upper/denom;
  yy[d]=(e[base+d]-(d?lower*yy[d-1]:0.f))/denom;
  double rhs=d==0?gamma:(d==nd-1?beta:0.f);
  zz[d]=(rhs-(d?lower*zz[d-1]:0.f))/denom;
 }
 for(int d=nd-2;d>=0;d--){yy[d]-=cp[d]*yy[d+1];zz[d]-=cp[d]*zz[d+1];}
 double factor=(yy[0]+alpha/gamma*yy[nd-1])/(1+zz[0]+alpha/gamma*zz[nd-1]);
 for(int d=0;d<nd;d++)e[base+d]=yy[d]-factor*zz[d];
}
// Block reductions: no field reduction or source integration on the CPU.
__device__ float reduce(float v,float* s){
 int t=threadIdx.x;s[t]=v;__syncthreads();
 for(int off=blockDim.x/2;off;off/=2){if(t<off)s[t]+=s[t+off];__syncthreads();}
 return s[0];
}
__device__ void spectral_moments(const float* e,const float* freq,const float* df,
 const float* k,const float* depth,const float* u0,const float* v0,
 const float* u1,const float* v1,float alpha,float* wind,float* m,
 int nc,int nf,int nd){
 int c=blockIdx.x,t=threadIdx.x,ns=nf*nd;if(c>=nc)return;
 __shared__ float sh[128];__shared__ float vv[10];
 float uu=(1-alpha)*u0[c]+alpha*u1[c],v=(1-alpha)*v0[c]+alpha*v1[c];
 if(t==0){wind[2*c]=uu;wind[2*c+1]=v;}__syncthreads();
 if(depth[c]<=0){if(t==0)for(int j=0;j<10;j++)m[c*10+j]=0;return;}
 float sw=0,si=0,sk=0,sx=0,sy=0,sf=0,sea=0;
 float speed=hypotf(uu,v),a=atan2f(v,uu),da=2*PI/nd;
 for(int z=t;z<ns;z+=blockDim.x){int f=z/nd,d=z%nd;
 float en=e[c*ns+z]*df[f]*da,sig=2*PI*freq[f],kk=k[c*nf+f],th=da*d;
 sw+=en;si+=en/sig;sk+=en/sqrtf(kk);sx+=en*cosf(th);sy+=en*sinf(th);sf+=freq[f]*en;
 if(sig/kk < 1.2f*speed*fmaxf(0.f,cosf(th-a)))sea+=en;
 }
 float et=reduce(sw,sh);if(t==0)vv[0]=et;__syncthreads();
 float x=reduce(si,sh);if(t==0)vv[1]=vv[0]/fmaxf(x,1e-30f);__syncthreads();
 x=reduce(sk,sh);if(t==0)vv[2]=sq(vv[0]/fmaxf(x,1e-30f));__syncthreads();
 x=reduce(sx,sh);if(t==0)vv[3]=x;__syncthreads();
 x=reduce(sy,sh);if(t==0)vv[4]=x;__syncthreads();
 x=reduce(sf,sh);if(t==0)vv[5]=x/fmaxf(vv[0],1e-30f);__syncthreads();
 x=reduce(sea,sh);if(t==0)vv[6]=x;__syncthreads();
 if(t==0){vv[7]=fmaxf(.001f,speed*.04f);vv[8]=.01f*sq(vv[7])/G+.11f*1.5e-5f/vv[7];}__syncthreads();
 // Coupled surface-layer fixed point for wind input. Atmospheric feedback is
 // exposed as z0/ustar and is applied by the separate coupling lane.
 for(int it=0;it<8;it++){
  float stressx=0,stressy=0;
  for(int z=t;z<ns;z+=blockDim.x){int f=z/nd,d=z%nd;float sig=2*PI*freq[f],kk=k[c*nf+f];
   float cs=cosf(da*d-a),b=growth(sig,kk,cs,vv[7],vv[8]);
   float force=G*b*e[c*ns+z]*df[f]*da*kk/sig;
   stressx+=force*cosf(da*d);stressy+=force*sinf(da*d);
  }
  float tx=reduce(stressx,sh);
  if(t==0)vv[9]=tx;__syncthreads();
  float ty=reduce(stressy,sh);
  float st=hypotf(vv[9],ty);
  if(t==0){float ratio=fminf(.95f,pos(st)*1025.f/(1.225f*sq(vv[7])));
   float z0=.01f*sq(vv[7])/G+.11f*1.5e-5f/vv[7];
   vv[8]=z0/sqrtf(1-ratio);
   float us=.41f*speed/logf((10+vv[8]-z0)/vv[8]);
   vv[7]=.5f*(vv[7]+fmaxf(.001f,us));vv[9]=z0;
  }__syncthreads();
 }
 if(t==0)for(int j=0;j<10;j++)m[c*10+j]=vv[j];
}
extern "C" __global__ void moments(const float* e,const float* freq,const float* df,
 const float* k,const float* depth,const float* u0,const float* v0,
 const float* u1,const float* v1,float alpha,float* wind,float* m,
 int nc,int nf,int nd){
 spectral_moments(e,freq,df,k,depth,u0,v0,u1,v1,alpha,wind,m,nc,nf,nd);
}
__device__ void source_bin(int z,float* e,const float* freq,const float* k,
 const float* depth,const float* wind,const float* m,int nc,int nf,int nd,float dt,
 float* budgets){
 int ns=nf*nd;
 int d=z%nd,f=(z/nd)%nf,c=z/ns;if(depth[c]<=0){e[z]=0;return;}
 const float* mm=m+10*c;float sig=2*PI*freq[f],kk=k[c*nf+f];
 float u=wind[2*c],v=wind[2*c+1],us=mm[7],cs=cosf(2*PI*d/nd-atan2f(v,u));
 float b=growth(sig,kk,cs,us,mm[8]);
 float sigpm=2*PI*.13f*G/(28*us),filter=expf(-powf(sigpm/sig,4.f));
 // Convert linear input per radian frequency to per Hz.
 float lin=1.5e-3f/(G*G)*powf(us*pos(cs),4.f)*filter;
 float steep=mm[2]*sqrtf(mm[0]),r=kk/fmaxf(mm[2],1e-12f);
 float white=4.10e-5f*(.5f+.5f*r)*powf(steep/sqrtf(.00302f),4.f)*mm[1]*r;
 float kh=kk*depth[c],sh=kh<40?sinhf(kh):1e30f;
 float bottom=.038f*sq(sig/G)/sq(sh);
 // Solve (1-Q)/log(Q) = -8*m0/Hmax^2 on (0,1).
 float hmax=.73f*depth[c],ratio=8*mm[0]/sq(hmax),qb;
 if(ratio>=1)qb=1;else if(ratio<1e-5f)qb=0;else{
  // Solve in log(Q): linear Q bisection injects spurious breaking when
  // the physical probability is far below floating point epsilon.
  float lo=-fminf(100.f,1/ratio+1),hi=-1e-7f;
  for(int it=0;it<28;it++){float mid=.5f*(lo+hi);float rr=expm1f(mid)/mid;if(rr>ratio)hi=mid;else lo=mid;}qb=expf(.5f*(lo+hi));
 }
 float breaking=.25f*qb*mm[5]*sq(hmax)/fmaxf(mm[0],1e-30f);
 // Positive split integration: exact exponential input, implicit dissipation.
 // Source substeps are chosen by the driver from the measured growth bound.
 float old=e[z],input=(old+lin*dt)*expf(b*dt),den=1+dt*(white+bottom+breaking);
 float val=input/den;e[z]=val;
 budgets[z*4]=input-old;budgets[z*4+1]=-dt*white*val;
 budgets[z*4+2]=-dt*bottom*val;budgets[z*4+3]=-dt*breaking*val;
}
extern "C" __global__ void sources(float* e,const float* freq,const float* k,
 const float* depth,const float* wind,const float* m,int nc,int nf,int nd,float dt,
 float* budgets){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=nc*nf*nd)return;
 source_bin(z,e,freq,k,depth,wind,m,nc,nf,nd,dt,budgets);
}
// Source subcycling stays inside a cell block for the asynchronous shared ABI.
// Its clock and growth reduction are device state; no host scalar reads or
// per-step synchronization are needed. Both doors use the same equations above.
extern "C" __global__ void source_integrate(float* e,const float* freq,const float* df,
 const float* k,const float* depth,const float* u,const float* v,float* wind,
 float* m,int nc,int nf,int nd,float interval,float* budgets){
 int c=blockIdx.x,t=threadIdx.x,ns=nf*nd;if(c>=nc||depth[c]<=0)return;
 __shared__ float sh[128];__shared__ float clock,step;
 if(t==0)clock=0;__syncthreads();
 while(clock<interval){
  spectral_moments(e,freq,df,k,depth,u,v,u,v,0,wind,m,nc,nf,nd);
  __syncthreads();
  float bound=0,a=atan2f(v[c],u[c]);
  for(int z=t;z<ns;z+=blockDim.x){int f=z/nd,d=z%nd;
   bound=fmaxf(bound,growth(2*PI*freq[f],k[c*nf+f],cosf(2*PI*d/nd-a),m[c*10+7],m[c*10+8]));
  }
  sh[t]=bound;__syncthreads();
  for(int off=blockDim.x/2;off;off/=2){if(t<off)sh[t]=fmaxf(sh[t],sh[t+off]);__syncthreads();}
  if(t==0)step=fminf(interval-clock,.2f/fmaxf(sh[0],1e-20f));__syncthreads();
  for(int z=t;z<ns;z+=blockDim.x)source_bin(c*ns+z,e,freq,k,depth,wind,m,nc,nf,nd,step,budgets);
  __syncthreads();if(t==0)clock=fminf(interval,clock+step);__syncthreads();
 }
}
__device__ float sample(const float* e,int base,int nf,int nd,float fi,float di){
 int f=(int)floorf(fi),d=(int)floorf(di);float a=fi-f,b=di-d;
 return (1-a)*((1-b)*e[base+f*nd+wrap(d,nd)]+b*e[base+f*nd+wrap(d+1,nd)])+
 a*((1-b)*e[base+(f+1)*nd+wrap(d,nd)]+b*e[base+(f+1)*nd+wrap(d+1,nd)]);
}
__device__ void scatter(float* out,const float* freq,const float* df,int base,int nd,
 float fi,float di,float q){
 int f=(int)floorf(fi),d=(int)floorf(di);float a=fi-f,b=di-d,da=2*PI/nd;
 for(int j=0;j<2;j++)for(int l=0;l<2;l++){
 int ff=f+j,dd=wrap(d+l,nd);float w=(j?a:1-a)*(l?b:1-b);
 atomicAdd(out+base+ff*nd+dd,q*w*(2*PI*freq[ff])/(df[ff]*da));
 }
}
__device__ float donor_limit(const float* e,const float* freq,const float* df,int base,int nd,
 float fi,float di,float dt){
 int f=(int)floorf(fi),d=(int)floorf(di);float a=fi-f,b=di-d,lim=1e30f;
 for(int j=0;j<2;j++)for(int l=0;l<2;l++){
  float w=(j?a:1-a)*(l?b:1-b);if(w<=1e-8f)continue;
  int ff=f+j,dd=wrap(d+l,nd);
  lim=fminf(lim,.2f*e[base+ff*nd+dd]*df[ff]*(2*PI/nd)/(2*PI*freq[ff]*w*dt));
 }return lim;
}
__device__ void dia_bin(int z,const float* e,float* rate,const float* freq,
 const float* df,const float* depth,const float* m,int nc,int nf,int nd,float logq,float dt,bool limited,float* diagonal=nullptr){
 int ns=nf*nd,n=nc*ns;if(z>=n)return;
 int d=z%nd,f=(z/nd)%nf,c=z/ns;if(depth[c]<=0)return;
 float fp=f+logf(1.25f)/logq,fm=f+logf(.75f)/logq;
 if(fm<0 || fp>=nf-1)return;
 // Interpolation linear in physical frequency preserves action and energy.
 int ip=(int)fp,im=(int)fm;
 fp=ip+(1.25f*freq[f]-freq[ip])/(freq[ip+1]-freq[ip]);
 fm=im+(.75f*freq[f]-freq[im])/(freq[im+1]-freq[im]);
 float en=e[z],f0=freq[f],sig=2*PI*f0;
 float kd=fmaxf(.5f,.75f*m[c*10+2]*depth[c]);
 float dep=1+5.5f/kd*(1-5.f/6.f*kd)*expf(-1.25f*kd);
 for(int mirror=-1;mirror<=1;mirror+=2){
  float dp=d+mirror*11.48f/360*nd,dm=d-mirror*33.56f/360*nd;
  float ep=sample(e,c*ns,nf,nd,fp,dp),em=sample(e,c*ns,nf,nd,fm,dm);
  float coeff=DIA_COEFFICIENT/powf(G,4.f)*powf(f0,11.f)*dep;
  float ap=1/powf(1.25f,4.f),am=1/powf(.75f,4.f),ac=1/powf(.9375f,4.f);
  float transfer=coeff*(en*en*(ep*ap+em*am)-2*en*ep*em*ac);
  float action=transfer*df[f]*(2*PI/nd)/sig;
  // Limit a whole resonant exchange, never its individual legs. This avoids
  // withdrawing action from an empty interpolation donor and keeps both
  // discrete action and variance invariants under the positivity remedy.
  if(limited && action>=0){action=fminf(action,.2f*en*df[f]*(2*PI/nd)/(2*sig*dt));}
  else if(limited){float available=fminf(donor_limit(e,freq,df,c*ns,nd,fp,dp,dt),donor_limit(e,freq,df,c*ns,nd,fm,dm,dt));action=-fminf(-action,available);}
  atomicAdd(rate+z,-2*action*sig/(df[f]*(2*PI/nd)));
  scatter(rate,freq,df,c*ns,nd,fp,dp,action);
  scatter(rate,freq,df,c*ns,nd,fm,dm,action);
  if(diagonal){
   // Differentiate this same discrete quartet, including interpolation and
   // deposition weights. HJ99 source integration uses the diagonal of the
   // TOTAL source Jacobian, not separately capped source operators.
   atomicAdd(diagonal+z,-2*coeff*(2*en*(ep*ap+em*am)-2*ep*em*ac));
   float locations[2]={fp,fm},angles[2]={dp,dm};
   float derivative[2]={coeff*(en*en*ap-2*en*em*ac),coeff*(en*en*am-2*en*ep*ac)};
   for(int leg=0;leg<2;leg++){
    int ff=(int)floorf(locations[leg]),dd=(int)floorf(angles[leg]);
    float fa=locations[leg]-ff,da=angles[leg]-dd;
    for(int j=0;j<2;j++)for(int l=0;l<2;l++){
     float weight=(j?fa:1-fa)*(l?da:1-da);int dest=c*ns+(ff+j)*nd+wrap(dd+l,nd);
     atomicAdd(diagonal+dest,derivative[leg]*weight*weight*df[f]*freq[ff+j]/(f0*df[ff+j]));
    }
   }
  }
 }
}
extern "C" __global__ void dia(const float* e,float* rate,const float* freq,
 const float* df,const float* depth,const float* m,int nc,int nf,int nd,float logq,float dt){
 dia_bin(blockIdx.x*blockDim.x+threadIdx.x,e,rate,freq,df,depth,m,nc,nf,nd,logq,dt,true);
}
// Diagnostic source at the captured state, without finite-step donor clipping.
// It is never applied to the prognostic spectrum.
extern "C" __global__ void dia_unlimited(const float* e,float* rate,const float* freq,
 const float* df,const float* depth,const float* m,int nc,int nf,int nd,float logq){
 dia_bin(blockIdx.x*blockDim.x+threadIdx.x,e,rate,freq,df,depth,m,nc,nf,nd,logq,0,false);
}
// Original analytic derivative of Hasselmann et al. 1985 eq 5.5, evaluated
// with the source itself. Used by the joint semi-implicit source update.
extern "C" __global__ void wave_dia_rates(const float* e,float* rate,float* diagonal,
 const float* freq,const float* df,const float* depth,const float* m,int nc,int nf,int nd,float logq){
 dia_bin(blockIdx.x*blockDim.x+threadIdx.x,e,rate,freq,df,depth,m,nc,nf,nd,logq,0,false,diagonal);
}
extern "C" __global__ void dia_apply(float* e,const float* rate,float* limiter,
 int nc,int nf,int nd,float dt){
 int c=blockIdx.x,t=threadIdx.x,ns=nf*nd;__shared__ float sh[128];
 float lim=1;
 for(int z=t;z<ns;z+=blockDim.x){int i=c*ns+z;if(rate[i]<0)lim=fminf(lim,.9f*e[i]/(-dt*rate[i]));}
 sh[t]=lim;__syncthreads();for(int off=blockDim.x/2;off;off/=2){if(t<off)sh[t]=fminf(sh[t],sh[t+off]);__syncthreads();}
 for(int z=t;z<ns;z+=blockDim.x){int i=c*ns+z;e[i]+=sh[0]*dt*rate[i];}
 if(t==0)limiter[c]=sh[0];
}
extern "C" __global__ void diagnose(const float* e,const float* df,const float* freq,
 const float* m,const float* depth,float* out,int nc,int nf,int nd){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(depth[c]<=0){for(int j=0;j<7;j++)out[j*nc+c]=nanf("");return;}
 float peak=-1;int pf=0;
 for(int f=0;f<nf;f++){float sum=0;for(int d=0;d<nd;d++)sum+=e[(c*nf+f)*nd+d];if(sum>peak){peak=sum;pf=f;}}
 const float* mm=m+c*10;float dir=fmodf(270-atan2f(mm[4],mm[3])*180/PI+360,360);
 out[c]=4*sqrtf(pos(mm[0]));out[nc+c]=mm[0]>1e-15f?1/freq[pf]:nanf("");
 out[2*nc+c]=mm[0]>1e-15f?dir:nanf("");out[3*nc+c]=4*sqrtf(pos(mm[6]));
 out[4*nc+c]=4*sqrtf(pos(mm[0]-mm[6]));out[5*nc+c]=mm[8];out[6*nc+c]=mm[7];
}
extern "C" __global__ void input_bound(const float* freq,const float* k,
 const float* depth,const float* wind,const float* m,float* bounds,int nc,int nf,int nd){
 int c=blockIdx.x,t=threadIdx.x;__shared__ float sh[128];float b=0;
 if(depth[c]>0){float a=atan2f(wind[2*c+1],wind[2*c]);for(int z=t;z<nf*nd;z+=blockDim.x){int f=z/nd,d=z%nd;
 b=fmaxf(b,growth(2*PI*freq[f],k[c*nf+f],cosf(2*PI*d/nd-a),m[c*10+7],m[c*10+8]));}}
 sh[t]=b;__syncthreads();for(int off=blockDim.x/2;off;off/=2){if(t<off)sh[t]=fmaxf(sh[t],sh[t+off]);__syncthreads();}
 if(t==0)bounds[c]=sh[0];
}
