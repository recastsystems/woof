// Original action-space kernels. Equations only, no external model source.
// Intrinsic frequency uses Hz as the integration measure: N=E/(2*pi*f).
#define PI 3.14159265358979323846f
#define G 9.80665f
#define RW 1025.f
#define KAP .41f
#ifndef NF
#define NF 36
#endif
#ifndef ND
#define ND 36
#endif
#ifndef WAVES_BACKEND_HANDLES_TURN
#define WAVES_BACKEND_HANDLES_TURN 0
#endif
__device__ float sqr(float x){return x*x;}
__device__ float positive(float x){return fmaxf(x,0.f);}
__device__ float sum_block(float x,float* sh){
 int t=threadIdx.x,lane=t%32,warp=t/32;
 for(int j=16;j;j/=2)x+=__shfl_down_sync(0xffffffff,x,j);
 // A single warp already owns the complete reduction. Broadcast its result
 // without staging through shared memory and a second reduction.
 if(blockDim.x==32)return __shfl_sync(0xffffffff,x,0);
 if(lane==0)sh[warp]=x;__syncthreads();
 if(warp==0){x=lane<blockDim.x/32?sh[lane]:0;
  for(int j=16;j;j/=2)x+=__shfl_down_sync(0xffffffff,x,j);
  if(lane==0)sh[0]=x;
 }__syncthreads();
 return sh[0];
}
// Conservative piecewise-linear angular transport with monotonized-central
// slopes: van Leer (1977), JCP23,276-299, doi:10.1016/0021-9991(77)90095-X.
// SSPRK2: Gottlieb & Shu (1998), Math.Comp67,73-85,
// doi:10.1090/S0025-5718-98-00913-2. Ring-local angular CFL<=.45;
// there is no source microstepping or host synchronization. Periodic face
// fluxes cancel exactly in real arithmetic. Both angular entry points use it.
__device__ float wave_mc(float l,float r){
 if(l*r<=0)return 0;
 return copysignf(fminf(.5f*fabsf(l+r),2*fminf(fabsf(l),fabsf(r))),l);
}
__device__ void wave_direction_solve(float* e,int ring,int nd,float dt,
 float left,float right,float left0,float right0){
 int d=threadIdx.x,count=blockDim.x;long long base=(long long)ring*nd;
 extern __shared__ float work[];
 float* q=work;float* original=q+count;float* slope=original+count;
 float* flux=slope+count;float* vmax=flux+count;
 vmax[d]=d<nd?fmaxf(fabsf(left),fabsf(right)):0;
 __syncthreads();
 for(int stride=count/2;stride;stride/=2){
  if(d<stride)vmax[d]=fmaxf(vmax[d],vmax[d+stride]);__syncthreads();
 }
 float delta=2*PI/nd;int pieces=max(1,(int)ceilf(dt*vmax[0]/(.45f*delta)));
 float h=dt/(pieces*delta);
 q[d]=d<nd?e[base+d]:0;__syncthreads();
 for(int step=0;step<pieces;step++){
  original[d]=q[d];__syncthreads();
  for(int stage=0;stage<2;stage++){
   if(d<nd)slope[d]=wave_mc(q[d]-q[(d+nd-1)%nd],q[(d+1)%nd]-q[d]);
   __syncthreads();
   if(d<nd){int next=(d+1)%nd;
    flux[d]=right*(right>=0?q[d]+.5f*slope[d]:q[next]-.5f*slope[next]);
   }__syncthreads();
   float next=d<nd?q[d]-h*(flux[d]-flux[(d+nd-1)%nd]):0;
   __syncthreads();q[d]=stage?.5f*(original[d]+next):next;__syncthreads();
  }
 }
 if(d<nd)e[base+d]=q[d];
}
extern "C" __global__ void wave_refract(float* e,const float* turn,const float* depth,
 int nc,int nf,int nd,float dt,int first,int number){
 int job=blockIdx.x,d=threadIdx.x;if(job>=nc*number||depth[job/number]<=0)return;
 int ring=(job/number)*nf+first+job%number;long long base=(long long)ring*nd;
 float left0=.5f*(turn[base+nd-1]+turn[base]);
 float right0=.5f*(turn[base]+turn[base+1]);
 float left=0,right=0;
 if(d<nd){left=.5f*(turn[base+(d+nd-1)%nd]+turn[base+d]);
  right=.5f*(turn[base+d]+turn[base+(d+1)%nd]);}
 wave_direction_solve(e,ring,nd,dt,left,right,left0,right0);
}
// Same analytic ray rate as wave_turn, evaluated where consumed. No scratch
// lifetime crosses the geographic operator, which borrows the turning buffer.
extern "C" __global__ void wave_refract_geometry(float* e,const float* depth,
 const float* lat,const float* hx,const float* hy,const float* freq,const float* k,
 const float* cg,int nc,int nf,int nd,float dt,int first,int number){
 int job=blockIdx.x,d=threadIdx.x;if(job>=nc*number||depth[job/number]<=0)return;
 int c=job/number,f=first+job%number,ring=c*nf+f;
 float kh=k[ring]*depth[c],inverse=kh>0&&kh<40?1/sinhf(2*kh):0;
 float a=2*PI*freq[f]*inverse*hx[c],b=-2*PI*freq[f]*inverse*hy[c];
#if !WAVES_BACKEND_HANDLES_TURN
 b-=cg[ring]*tanf(lat[c]*PI/180)/6371000.f;
#endif
 float angle=2*PI*d/nd,delta=2*PI/nd;
 float center=a*sinf(angle)+b*cosf(angle);
 float left=.5f*(center+a*sinf(angle-delta)+b*cosf(angle-delta));
 float right=.5f*(center+a*sinf(angle+delta)+b*cosf(angle+delta));
 float left0=.5f*(b-a*sinf(delta)+b*cosf(delta));
 float right0=.5f*(b+a*sinf(delta)+b*cosf(delta));
 wave_direction_solve(e,ring,nd,dt,left,right,left0,right0);
}
// Linear dispersion and Hamiltonian rays: Holthuijsen (2007), chapter 7,
// Cambridge, doi:10.1017/CBO9780511618536. Depth change is local Eulerian dh/dt;
// steady spatial depth enters refraction/shoaling, NOT intrinsic frequency drift.
extern "C" __global__ void wave_geometry(const float* h,const float* lat,
 const float* hx,const float* hy,const float* dhdt,const float* freq,
 float* k,float* cg,float* fdot,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(z>=(long long)nc*NF)return;
 int f=z%NF,c=z/NF;float depth=h[c],sig=2*PI*freq[f];
 if(depth<=0){k[z]=0;cg[z]=0;fdot[z]=0;return;}
 float kk=fmaxf(sig*sig/G,sig/sqrtf(G*depth));
 for(int it=0;it<10;it++){float th=tanhf(kk*depth);
  kk=fmaxf(1e-9f,kk-(G*kk*th-sig*sig)/(G*(th+kk*depth*(1-th*th))));}
 float kh=kk*depth,invsh=kh<40?1/sinhf(2*kh):0;
 float group=.5f*sig/kk*(1+2*kh*invsh);
 k[z]=kk;cg[z]=group;fdot[z]=freq[f]*kk*invsh*dhdt[c];
}
extern "C" __global__ void wave_gather(const float* full,const int* wet,float* compact,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)compact[c]=full[wet[c]];
}
extern "C" __global__ void wave_turn(const float* h,const float* lat,const float* hx,
 const float* hy,const float* freq,const float* k,const float* cg,float* turn,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int c=z/(NF*ND),f=(z/ND)%NF,d=z%ND;float kh=k[c*NF+f]*h[c],a=2*PI*d/ND;
 float invsh=kh>0&&kh<40?1/sinhf(2*kh):0;
 turn[z]=h[c]>0?2*PI*freq[f]*invsh*(sinf(a)*hx[c]-cosf(a)*hy[c]):0;
#if !WAVES_BACKEND_HANDLES_TURN
 if(h[c]>0)turn[z]-=cg[c*NF+f]*cosf(a)*tanf(lat[c]*PI/180)/6371000.f;
#endif
}
extern "C" __global__ void wave_wind_mask(const float* beta,unsigned int* mask,int nc){
 int word=blockIdx.x*blockDim.x+threadIdx.x,words=(NF*ND+31)/32;if(word>=nc*words)return;
 int c=word/words,start=(word%words)*32;unsigned int bits=0;
 for(int j=0;j<32&&start+j<NF*ND;j++)if(beta[(long long)c*NF*ND+start+j]>0)bits|=1u<<j;
 mask[word]=bits;
}
extern "C" __global__ void wave_limiter_reset(float* limiter,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<nc)limiter[c]=1;
}
extern "C" __global__ void wave_water_level(const float* eta,const float* base,
 const int* wet,float* h,float* level,float* previous,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 level[c]=eta[wet[c]];h[c]=fmaxf(base[c]+level[c],0.f);
}
extern "C" __global__ void wave_depth_rate(const float* h,float* previous,float* rate,int n,float dt){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n){rate[c]=(h[c]-previous[c])/dt;previous[c]=h[c];}
}
extern "C" __global__ void wave_convert(const float* src,float* dest,const float* f,int n,int to_energy){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)n*NF*ND)return;
 float sig=2*PI*f[(z/ND)%NF];dest[z]=to_energy?src[z]*sig:src[z]/sig;
}
extern "C" __global__ void wave_dia_moments(const float* m,float* legacy,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 for(int j=0;j<10;j++)legacy[c*10+j]=0;
 legacy[c*10]=m[c*16];legacy[c*10+2]=m[c*16+3];
}
extern "C" __global__ void wave_zero(float* a,long long n){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z<n)a[z]=0;
}
extern "C" __global__ void wave_copy(const float* a,float* b,long long n){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z<n)b[z]=a[z];
}
extern "C" __global__ void wave_copy_group(const float* a,float* b,int nc,int first,int count){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*count*ND)return;
 long long q=(z/(count*ND))*NF*ND+first*ND+z%(count*ND);b[q]=a[q];
}
extern "C" __global__ void wave_clock(double* clock,long long* steps,double dt){
 if(blockIdx.x==0&&threadIdx.x==0){clock[0]+=dt;steps[0]++;}
}
extern "C" __global__ void wave_timer(unsigned long long* values,int stage,int finish){
 if(blockIdx.x==0&&threadIdx.x==0){unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  if(finish)values[stage*2+1]+=now-values[stage*2];else values[stage*2]=now;
 }
}
extern "C" __global__ void wave_forcing_bracket(const float* u0,const float* v0,
 const float* u1,const float* v1,const int* wet,const double* clock,const double* times,
 float* u,float* v,int* errors,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 double a=(clock[0]-times[0])/(times[1]-times[0]);
 if(a<0||a>1||!isfinite(a)){if(c==0)atomicAdd(errors,1);u[c]=nanf("");v[c]=nanf("");return;}
 int i=wet[c];u[c]=u0[i]+(float)a*(u1[i]-u0[i]);v[c]=v0[i]+(float)a*(v1[i]-v0[i]);
}
extern "C" __global__ void wave_boundary(const float* e0,const float* e1,
 float* n,const float* f,int cells,const double* clock,const double* bracket,
 double offset,int* errors){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)cells*NF*ND)return;
 double alpha=(clock[0]+offset-bracket[0])/(bracket[1]-bracket[0]);
 if(alpha<0||alpha>1||!isfinite(alpha)){if(z==0)atomicAdd(errors,1);n[z]=nanf("");return;}
 float a=(float)alpha;
 n[z]=(e0[z]+a*(e1[z]-e0[z]))/(2*PI*f[(z/ND)%NF]);
}
// Conservative backward Euler frequency advection. Closed frequency ends retain
// action instead of deleting it outside a finite grid; edge energy is diagnosed.
extern "C" __global__ void wave_frequency(float* n,const float* fdot,
 const float* df,int nc,float dt){
 int ring=blockIdx.x*blockDim.x+threadIdx.x;if(ring>=nc*ND)return;
 int c=ring/ND,d=ring%ND;float upper[NF],rhs[NF];
 for(int f=0;f<NF;f++){
  float l=f?.5f*(fdot[c*NF+f-1]+fdot[c*NF+f]):0;
  float r=f<NF-1?.5f*(fdot[c*NF+f]+fdot[c*NF+f+1]):0;
  float low=-dt*positive(l)/df[f],up=-dt*positive(-r)/df[f];
  float diag=1+dt*(positive(r)+positive(-l))/df[f];
  float den=diag-(f?low*upper[f-1]:0);
  upper[f]=up/den;rhs[f]=(n[((long long)c*NF+f)*ND+d]-(f?low*rhs[f-1]:0))/den;
 }
 for(int f=NF-2;f>=0;f--)rhs[f]-=upper[f]*rhs[f+1];
 for(int f=0;f<NF;f++)n[((long long)c*NF+f)*ND+d]=rhs[f];
}
// Moments in E units. m[0:7]=m0,m1,m2,kmean,mean_sigma,windsea_m0,
// windsea_inverse_f; m[7:16]=ustar,z0_eff,charnock,tau_w_e,tau_w_n,
// a_orb,u_orb,peak_f,reserved. BAJ positive-root-k mean, TM509 eqs 6-7.
extern "C" __global__ void wave_moments(const float* n,const float* freq,
 const float* df,const float* k,const float* u,const float* v,const unsigned int* windmask,float* m,int nc,float ratio){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;
 __shared__ float sh[128];float a[7]={0,0,0,0,0,0,0};
 float wind=hypotf(u[c],v[c]),angle=atan2f(v[c],u[c]);
 for(int b=t;b<NF*ND;b+=blockDim.x){int f=b/ND,d=b%ND;
  float sig=2*PI*freq[f],en=n[(long long)c*NF*ND+b]*sig*df[f]*(2*PI/ND);
  float kk=k[c*NF+f];a[0]+=en;a[1]+=freq[f]*en;a[2]+=freq[f]*freq[f]*en;
  a[3]+=sqrtf(fmaxf(kk,0.f))*en;a[4]+=en/sig;
  // TM509 eq 10: input-positive OR the 28*u* windsea criterion. This is
  // intentionally distinct from the output partition's 1.5*U10 criterion.
  bool input_positive=(windmask[c*((NF*ND+31)/32)+b/32]>>(b%32))&1u;
  if(kk>0 && (input_positive || sig/kk<28*m[c*16+7]*positive(cosf(2*PI*d/ND-angle)))){a[5]+=en;a[6]+=en/freq[f];}
 }
 // Integrate the f^-5 tail above the highest bin EDGE, not its centre, to
 // avoid double counting the top bin. Integral of A*f^(p-5)=A*fb^(p-4)/(4-p).
 // TM509 eqs 6-9; Ardhuin et al. 2010 eq 7. High-frequency dispersion is deep.
 for(int d=t;d<ND;d+=blockDim.x){
  float ft=freq[NF-1],fb=ft*sqrtf(ratio);
  float amplitude=n[((long long)c*NF+NF-1)*ND+d]*(2*PI*ft)*powf(ft,5.f)*(2*PI/ND);
  float t0=amplitude/(4*powf(fb,4.f)),t1=amplitude/(3*powf(fb,3.f));
  a[0]+=t0;a[1]+=t1;a[2]+=amplitude/(2*fb*fb);
  a[3]+=(2*PI/sqrtf(G))*t1;a[4]+=amplitude/(10*PI*powf(fb,5.f));
  float along=28*m[c*16+7]*positive(cosf(2*PI*d/ND-angle));
  if(along>0){float cutoff=fmaxf(fb,G/(2*PI*along));
   a[5]+=amplitude/(4*powf(cutoff,4.f));a[6]+=amplitude/(5*powf(cutoff,5.f));}
 }
 for(int j=0;j<7;j++){float val=sum_block(a[j],sh);if(t==0)m[c*16+j]=val;__syncthreads();}
 if(t==0){float* mm=m+c*16;float en=fmaxf(mm[0],1e-30f);
  mm[3]=sqr(mm[3]/en);mm[4]=en/fmaxf(mm[4],1e-30f);
  mm[12]=2*sqrtf(positive(mm[0]));mm[13]=4*PI*sqrtf(positive(mm[2]));}
}
// Janssen (1991), JPO 21, doi:10.1175/1520-0485(1991)021<1631:QLTOWW>2.0.CO;2.
// Ardhuin et al. (2010), JPO 40, doi:10.1175/2010JPO4324.1, eqs 19-24.
__device__ float wind_beta(float sig,float kk,float cs,float us,float z0,
 float rho,float beta_max,float za){
 if(cs<=0||us<=0||kk<=0)return 0;
 float c=sig/kk,z=logf(fmaxf(kk*z0,1e-30f))+KAP/(cs*(us/c+za));
 if(z>=0||z<-80)return 0;
 return rho/RW*beta_max/(KAP*KAP)*expf(z)*sqr(sqr(z))*sqr(us/c)*cs*cs*sig;
}
extern "C" __global__ void wave_wind(const float* n,const float* freq,
 const float* df,const float* k,const float* u,const float* v,const float* rho,
 const float* shelter,float* m,float* beta,int nc,int pack){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;__shared__ float sh[128];
 float* mm=m+c*16;float speed=hypotf(u[c],v[c]),wa=atan2f(v[c],u[c]);
 float us=fmaxf(mm[7],.001f),z0=fmaxf(mm[8],1e-6f),tx=0,ty=0;
 float bm=pack==0?1.52f:1.2f,za=pack==0?.006f:.011f;
 for(int b=t;b<NF*ND;b+=blockDim.x){int f=b/ND,d=b%ND;float angle=2*PI*d/ND;
  float ux=us*us*cosf(wa),uy=us*us*sinf(wa);
  if(pack==0){ux-=shelter[(c*NF+f)*2];uy-=shelter[(c*NF+f)*2+1];}
  // The residual stress vector changes input direction as well as magnitude.
  float ur=sqrtf(hypotf(ux,uy)),wr=atan2f(uy,ux);
  float sig=2*PI*freq[f],kk=k[c*NF+f];
  float bval=wind_beta(sig,kk,cosf(angle-wr),ur,z0,rho[c],bm,za);
  beta[(long long)c*NF*ND+b]=bval;
  float stress=RW*G*kk*bval*n[(long long)c*NF*ND+b]*df[f]*(2*PI/ND);
  tx+=stress*cosf(angle);ty+=stress*sinf(angle);
 }
 // Published f^-5 diagnostic stress tail, integrated on 24 log-frequency bins.
 // Eq 25 is cumulative over ALL directions below each frequency. Updating
 // separate directional residuals gives each direction its own stress budget
 // and overcounts the tail. Reduce each ring before the next frequency.
 __shared__ float totalx,totaly;
 float sx=sum_block(tx,sh);if(t==0)totalx=sx;__syncthreads();
 float sy=sum_block(ty,sh);if(t==0)totaly=sy;__syncthreads();
 const float ratio=1.189207115f;
 float fmax=freq[NF-1],edge=fmax*sqrtf(freq[NF-1]/freq[NF-2]);
 for(int j=0;j<24;j++){
  float ux=us*us*cosf(wa),uy=us*us*sinf(wa);
  if(pack==0){
   float scale=fminf(1.f,.95f*us*us*rho[c]/fmaxf(hypotf(totalx,totaly),1e-30f));
   ux-=scale*totalx/rho[c];uy-=scale*totaly/rho[c];
  }
  float f=edge*powf(ratio,j+.5f),sig=2*PI*f,kk=sig*sig/G,dx=0,dy=0;
  float width=f*(sqrtf(ratio)-1/sqrtf(ratio));
  for(int d=t;d<ND;d+=blockDim.x){
   float angle=2*PI*d/ND,emax=n[((long long)c*NF+NF-1)*ND+d]*(2*PI*fmax);
   float bval=wind_beta(sig,kk,cosf(angle-atan2f(uy,ux)),sqrtf(hypotf(ux,uy)),z0,rho[c],bm,za);
   float stress=RW*G*kk/sig*bval*emax*powf(f/fmax,-5.f)*width*(2*PI/ND);
   dx+=stress*cosf(angle);dy+=stress*sinf(angle);
  }
  sx=sum_block(dx,sh);if(t==0)totalx+=sx;__syncthreads();
  sy=sum_block(dy,sh);if(t==0)totaly+=sy;__syncthreads();
 }
 if(t==0){mm[10]=totalx;mm[11]=totaly;
  if(speed<1e-6f){mm[7]=0;mm[8]=0;mm[9]=.0095f;mm[10]=0;mm[11]=0;return;}
  float ratio=fminf(.95f,hypotf(totalx,totaly)/(rho[c]*us*us));
  float base=fminf(.0095f*us*us/G,.002f)+.11f*1.5e-5f/us;
  float effective=base/sqrtf(1-ratio);
  float next=KAP*speed/logf(10/fminf(effective,1.f));
  mm[7]=speed>0?.5f*(us+next):0;
  mm[8]=effective;mm[9]=effective*G/fmaxf(us*us,1e-12f);
 }
}
extern "C" __global__ void wave_wind_fused(const float* n,const float* freq,
 const float* df,const float* k,const float* u,const float* v,const float* rho,
 float* shelter,float* m,float* beta,int nc,int pack){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;__shared__ float sh[128];
 float* mm=m+c*16;float speed=hypotf(u[c],v[c]),wa=atan2f(v[c],u[c]);
 __shared__ float ringx[NF],ringy[NF];
 // Tail dispersion and quadrature geometry do not change across the eight
 // surface-layer iterations. Each ring is evaluated once per cell.
 __shared__ float tail_frequency[24],tail_wavenumber[24],tail_power[24],tail_width[24];
 float top_frequency=freq[NF-1];
 float tail_edge=top_frequency*sqrtf(freq[NF-1]/freq[NF-2]);
 const float tail_ratio=1.189207115f;
 for(int j=t;j<24;j+=blockDim.x){
  float f=tail_edge*powf(tail_ratio,j+.5f),sig=2*PI*f;
  tail_frequency[j]=f;tail_wavenumber[j]=sig*sig/G;
  tail_power[j]=powf(f/top_frequency,-5.f);
  tail_width[j]=f*(sqrtf(tail_ratio)-1/sqrtf(tail_ratio));
 }__syncthreads();
 for(int iteration=0;iteration<8;iteration++){
 float us=fmaxf(mm[7],.001f),z0=fmaxf(mm[8],1e-6f),tx=0,ty=0;
 float bm=pack==0?1.52f:1.2f,za=pack==0?.006f:.011f;
 for(int b=t;b<NF*ND;b+=blockDim.x){int f=b/ND,d=b%ND;float angle=2*PI*d/ND;
  float ux=us*us*cosf(wa),uy=us*us*sinf(wa);
  if(pack==0){ux-=shelter[(c*NF+f)*2];uy-=shelter[(c*NF+f)*2+1];}
  // The residual stress vector changes input direction as well as magnitude.
  float ur=sqrtf(hypotf(ux,uy)),wr=atan2f(uy,ux);
  float sig=2*PI*freq[f],kk=k[c*NF+f];
  float bval=wind_beta(sig,kk,cosf(angle-wr),ur,z0,rho[c],bm,za);
  beta[(long long)c*NF*ND+b]=bval;
  float stress=RW*G*kk*bval*n[(long long)c*NF*ND+b]*df[f]*(2*PI/ND);
  tx+=stress*cosf(angle);ty+=stress*sinf(angle);
 }
 // Published f^-5 diagnostic stress tail, integrated on 24 log-frequency bins.
 // Eq 25 is cumulative over ALL directions below each frequency. Updating
 // separate directional residuals gives each direction its own stress budget
 // and overcounts the tail. Reduce each ring before the next frequency.
 __shared__ float totalx,totaly;
 float sx=sum_block(tx,sh);if(t==0)totalx=sx;__syncthreads();
 float sy=sum_block(ty,sh);if(t==0)totaly=sy;__syncthreads();
 float fmax=top_frequency;
 for(int j=0;j<24;j++){
  float ux=us*us*cosf(wa),uy=us*us*sinf(wa);
  if(pack==0){
   float scale=fminf(1.f,.95f*us*us*rho[c]/fmaxf(hypotf(totalx,totaly),1e-30f));
   ux-=scale*totalx/rho[c];uy-=scale*totaly/rho[c];
  }
  float f=tail_frequency[j],sig=2*PI*f,kk=tail_wavenumber[j],dx=0,dy=0;
  for(int d=t;d<ND;d+=blockDim.x){
   float angle=2*PI*d/ND,emax=n[((long long)c*NF+NF-1)*ND+d]*(2*PI*fmax);
   float bval=wind_beta(sig,kk,cosf(angle-atan2f(uy,ux)),sqrtf(hypotf(ux,uy)),z0,rho[c],bm,za);
   float stress=RW*G*kk/sig*bval*emax*tail_power[j]*tail_width[j]*(2*PI/ND);
   dx+=stress*cosf(angle);dy+=stress*sinf(angle);
  }
  sx=sum_block(dx,sh);if(t==0)totalx+=sx;__syncthreads();
  sy=sum_block(dy,sh);if(t==0)totaly+=sy;__syncthreads();
 }
 if(t==0){mm[10]=totalx;mm[11]=totaly;
  if(speed<1e-6f){mm[7]=0;mm[8]=0;mm[9]=.0095f;mm[10]=0;mm[11]=0;}else{
  float ratio=fminf(.95f,hypotf(totalx,totaly)/(rho[c]*us*us));
  float base=fminf(.0095f*us*us/G,.002f)+.11f*1.5e-5f/us;
  float effective=base/sqrtf(1-ratio);
  float next=KAP*speed/logf(10/fminf(effective,1.f));
  mm[7]=speed>0?.5f*(us+next):0;
  mm[8]=effective;mm[9]=effective*G/fmaxf(us*us,1e-12f);
  }
 }__syncthreads();
 if(pack==0){
  // Each ring stress is reduced once, then summed in frequency order. This
  // replaces NF copies of the lower-frequency spectral integration.
  for(int f=t;f<NF;f+=blockDim.x){float tx=0,ty=0;
   for(int d=0;d<ND;d++){long long z=((long long)c*NF+f)*ND+d;
    float w=RW/rho[c]*G*k[c*NF+f]*beta[z]*n[z]*df[f]*(2*PI/ND);
    tx+=w*cosf(2*PI*d/ND);ty+=w*sinf(2*PI*d/ND);}
   ringx[f]=tx;ringy[f]=ty;
  }__syncthreads();
  for(int f=t;f<NF;f+=blockDim.x){float tx=0,ty=0;
   for(int j=0;j<f;j++){tx+=ringx[j];ty+=ringy[j];}
   float us=mm[7],scale=fminf(1.f,.95f*us*us/fmaxf(hypotf(tx,ty),1e-30f));
   shelter[(c*NF+f)*2]=tx*scale;shelter[(c*NF+f)*2+1]=ty*scale;
  }__syncthreads();
 }
 }
}
extern "C" __global__ void wave_shelter(const float* n,const float* beta,
 const float* freq,const float* df,const float* k,const float* rho,
 const float* u,const float* v,const float* m,float* out,int nc){
 int ring=blockIdx.x*blockDim.x+threadIdx.x;if(ring>=nc*NF)return;
 int c=ring/NF,f=ring%NF;float tx=0,ty=0;
 for(int j=0;j<f;j++)for(int d=0;d<ND;d++){
  long long z=((long long)c*NF+j)*ND+d;
  float a=2*PI*d/ND,w=RW/rho[c]*G*k[c*NF+j]*beta[z]*n[z]*df[j]*(2*PI/ND);
  tx+=w*cosf(a);ty+=w*sinf(a);
 }
 // Do not let an iterative stress estimate reverse the residual wind stress.
 // This bound is a positivity condition of the stress partition, not a source toggle.
 float us=m[c*16+7],s=hypotf(tx,ty),scale=fminf(1.f,.95f*us*us/fmaxf(s,1e-30f));
 out[ring*2]=tx*scale;out[ring*2+1]=ty*scale;
}
// Directional saturation, Ardhuin et al. 2010 eqs 12-15, DOI above.
extern "C" __global__ void wave_saturation(const float* n,const float* freq,
 const float* k,const float* cg,float* sat,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int d=z%ND,f=(z/ND)%NF,c=z/(NF*ND);float sum=0,kk=k[c*NF+f];
 for(int j=0;j<ND;j++){float cs=cosf(2*PI*(j-d)/ND);if(cs>=.1736481777f)
  sum+=cs*cs*n[((long long)c*NF+f)*ND+j]*(2*PI*freq[f])*(2*PI/ND);}
 sat[z]=kk*kk*kk*cg[c*NF+f]/(2*PI)*sum;
}
// Original source assembly from published equations. Arrays are E tendencies.
extern "C" __global__ void wave_depth_breaking(const float* depth,const float* m,float* out,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(depth[c]<=0){out[c]=0;return;}const float* mm=m+c*16;
 float hmax=.73f*depth[c],qratio=8*mm[0]/(hmax*hmax),qb=0;
 if(qratio>=1)qb=1;else if(qratio>1e-5f){
  float lo=-fminf(100.f,1/qratio+1),hi=-1e-7f;
  for(int j=0;j<24;j++){float mid=.5f*(lo+hi);if(expm1f(mid)/mid>qratio)hi=mid;else lo=mid;}
  qb=expf(.5f*(lo+hi));}
 out[c]=.25f*qb*(mm[1]/fmaxf(mm[0],1e-30f))*hmax*hmax/fmaxf(mm[0],1e-30f);
}
// Cache the direction-dependent breaking weight used by every higher bin.
// This is the identical eq 18 integrand, before multiplying relative speed.
extern "C" __global__ void wave_breaking_weight(const float* sat,const float* freq,
 const float* df,const float* k,const float* cg,const float* depth,float* weight,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int f=(z/ND)%NF,c=z/(NF*ND);float y=tanhf(k[c*NF+f]*depth[c]);
 float br=.0009f*y*(1.3286f*y*y*y-2.5709f*y*y+1.9995f*y+.2428f);
 float p=positive(sqrtf(positive(sat[z]))-sqrtf(positive(br)));
 weight[z]=.4f*(28.4f/PI)*p*p/fmaxf(cg[c*NF+f],1e-12f)*df[f]*(2*PI/ND);
}
// Bottom: Hasselmann et al. 1973, Deutsche Hydrographische Zeitschrift A8(12),
// JONSWAP report (no DOI verified); institutional primary text in SCIENCE.md.
// Breaking: Battjes & Janssen 1978, ICCE, doi:10.9753/icce.v16.32.
// Seed: Cavaleri & Malanotte-Rizzoli 1981, JGR, doi:10.1029/JC086iC11p10961.
// White: Bidlot et al. 2007, ECMWF TM509, doi:10.21957/m97gmhqze, eqs 4-7.
extern "C" __global__ void wave_source_rates(const float* n,const float* freq,
 const float* df,const float* k,const float* cg,const float* depth,const float* u,
 const float* v,const float* rho,const float* m,const float* beta,const float* sat,const float* breaking,const float* depth_breaking,
 float* loss,float* seed,int nc,int pack,const int* probe_map,float* components,int probes){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int d=z%ND,f=(z/ND)%NF,c=z/(NF*ND);const float* mm=m+c*16;
 if(depth[c]<=0){loss[z]=0;seed[z]=0;return;}
 if(!isfinite(u[c])||!isfinite(v[c])||!isfinite(rho[c])||rho[c]<=0){loss[z]=nanf("");seed[z]=nanf("");return;}
 float sig=2*PI*freq[f],kk=k[c*NF+f],kh=kk*depth[c],wa=atan2f(v[c],u[c]);
 float us=mm[7],cs=positive(cosf(2*PI*d/ND-wa));
 float fpm=.13f*G/fmaxf(28*us,1e-12f);
 seed[z]=.0015f/(G*G)*powf(us*cs,4.f)*expf(-powf(fpm/freq[f],4.f));
 float invsh=kh<40?1/sinhf(kh):0;
 float sink=.038f*sqr(sig/G*invsh);
 int probe=probe_map?probe_map[c]:-1;
 if(probe>=0){components[(long long)probe*NF*ND+f*ND+d]=sink;
  components[((long long)probes+probe)*NF*ND+f*ND+d]=depth_breaking[c];}
 float before_white=sink+depth_breaking[c],before_cumulative=before_white,before_swell=before_white;
 sink+=depth_breaking[c];
 if(pack==1){
  float kr=fmaxf(mm[3],1e-12f),r=kk/kr;
  float mean_sigma=2*PI*mm[1]/fmaxf(mm[0],1e-30f);
  sink+=2.1f*mean_sigma*sqr(kr*kr*mm[0])*(.4f*r+.6f*r*r);
  before_cumulative=sink;before_swell=sink;
 }else{
  // Published Table A2 coefficient, finite-depth threshold eq 15 active.
  float y=tanhf(kh),br=.0009f*y*(1.3286f*y*y*y-2.5709f*y*y+1.9995f*y+.2428f);
  br=fmaxf(br,1e-12f);float maxb=0;
  for(int j=0;j<ND;j++)maxb=fmaxf(maxb,sat[((long long)c*NF+f)*ND+j]);
  sink+=sig*.000022f*(.3f*sqr(positive(maxb/br-1))+.7f*sqr(positive(sat[z]/br-1)));
  before_cumulative=sink;
  float cp=sig/fmaxf(kk,1e-12f),a=2*PI*d/ND;
  // Cumulative breaking, eq 18 (published; eq 19 in preprint).
  for(int j=0;j<f&&freq[j]<.5f*freq[f];j++){
   float kp=k[c*NF+j],cprime=2*PI*freq[j]/fmaxf(kp,1e-12f);
   for(int l=0;l<ND;l++){
    float w=breaking[((long long)c*NF+j)*ND+l];
    if(w>0){float relative=sqrtf(fmaxf(0.f,cp*cp+cprime*cprime-2*cp*cprime*cosf(a-2*PI*l/ND)));
     sink+=w*relative;}
   }
  }
  before_swell=sink;
  // Swell air friction: 2010 JPO eqs 8-9, doi:10.1175/2010JPO4324.1,
  // updated by 2016 GRL eqs 3-4 and section 3, doi:10.1002/2015GL067566.
  // The later paper corrects the factor-two viscous error and reports the
  // 1050-track median fe=.0037. Use that external population value, not a
  // coefficient selected from this model's buoy errors. Retain the published
  // 2010 Cdsv=1.2 and Reynolds switch; only the swell-friction term changes.
  float re=4*mm[12]*mm[13]/1.5e-5f,hs=4*sqrtf(positive(mm[0]));
  sink+=rho[c]/RW*(re<2e5f/fmaxf(hs,1e-6f)
      ?1.2f*kk*sqrtf(2*1.5e-5f*sig):16*.0037f*sig*sig*mm[13]/G);
 }
 if(probe>=0){
  components[((long long)2*probes+probe)*NF*ND+f*ND+d]=before_cumulative-before_white;
  components[((long long)3*probes+probe)*NF*ND+f*ND+d]=before_swell-before_cumulative;
  components[((long long)4*probes+probe)*NF*ND+f*ND+d]=sink-before_swell;
 }
 loss[z]=sink;
}
// Semi-implicit diagonal source update. HJ99 limiter revised in TM509 eq 13.
// Each call takes the entire requested source interval, with no local clocks.
extern "C" __global__ void wave_sources(float* n,const float* freq,const float* m,
 const float* beta,const float* loss,const float* seed,const float* nonlinear,
 const float* nonlinear_diagonal,float* limiter,int nc,int nfx,float dt){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int f=(z/ND)%NF,c=z/(NF*ND);const float* mm=m+c*16;float sig=2*PI*freq[f],old=n[z]*sig;
 long long ext=((long long)c*nfx+f)*ND+z%ND;
 float b=beta[z],d=loss[z],rate=seed[z]+(b-d)*old+nonlinear[ext];
 if(!isfinite(old)||!isfinite(rate)||!isfinite(d)){n[z]=nanf("");atomicMin((unsigned int*)(limiter+c),0u);return;}
 // Hersbach & Janssen (1999), DOI in SCIENCE.md; fully implicit weighting. All terms
 // share one increment, so an equilibrium of their SUM remains an equilibrium.
 // Off-diagonal coupling is omitted. Loss coefficients are frozen in this
 // diagonal linearization; DIA's local derivative is evaluated analytically.
 float delta=dt*rate/(1+dt*fmaxf(0.f,d-b-nonlinear_diagonal[ext]));
 float fm=mm[5]>1e-20f?mm[5]/fmaxf(mm[6],1e-30f):freq[NF-1]/2.5f;
 float cap=3e-7f*G*fmaxf(mm[7],G*.0056f/freq[f])*powf(freq[f],-4.f)*fm*dt;
 float limited=fmaxf(-old,fminf(cap,fmaxf(-cap,delta)));
 n[z]=(old+limited)/sig;
 float factor=fabsf(delta)>1e-30f?fabsf(limited/delta):1.f;
#if __CUDA_ARCH__ >= 800
 // All members of a warp addressing the same cell have the same atomic
 // destination. Integer minimum preserves the exact positive-float minimum.
 unsigned int group=__match_any_sync(__activemask(),c);
 unsigned int minimum=__reduce_min_sync(group,__float_as_uint(factor));
 if((threadIdx.x%32)==__ffs(group)-1)atomicMin((unsigned int*)(limiter+c),minimum);
#else
 atomicMin((unsigned int*)(limiter+c),__float_as_uint(factor));
#endif
}
// Diagnostic high-frequency tail: Bidlot et al. 2007, TM509 eq 9;
// Ardhuin et al. 2010, JPO eq 7. Anchor all directions before writing bins.
extern "C" __global__ void wave_tail(float* n,const float* freq,const float* m,
 int nc,int pack,float logratio){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;
 __shared__ float anchor[ND];const float* mm=m+c*16;
 float fm=pack==1?(mm[5]>1e-20f?mm[5]/fmaxf(mm[6],1e-30f):freq[NF-1])
                 :(mm[0]>1e-20f?mm[1]/mm[0]:freq[NF-1]);
 // TM509 equation 9 replaces the old cycle-4 max(2.5*fm,4*fPM) rule.
 // The revised windsea mean must not retain that superseded PM floor.
 float fc=fmaxf(freq[0],(pack==1?2.5f:9.9f)*fm);
 if(fc>=freq[NF-1])return;
 int f0=(int)floorf(logf(fc/freq[0])/logratio);f0=max(0,min(NF-2,f0));
 for(int d=t;d<ND;d+=blockDim.x){
  anchor[d]=n[((long long)c*NF+f0)*ND+d]*2*PI*freq[f0];
 }__syncthreads();
 for(int b=t;b<NF*ND;b+=blockDim.x){int f=b/ND,d=b%ND;
  if(f>f0)n[(long long)c*NF*ND+b]=anchor[d]*powf(freq[f]/freq[f0],-5.f)/(2*PI*freq[f]);}
}
// Hasselmann et al. 1985 eq 5.5; Tolman 2003 MMAB226 section 2.1 explicitly
// extends the spectral grid so every quartet feeding a resolved bin is present.
// The coefficient is for E(f,theta): C*g^-4*f^11, with NO extra (2*pi)^2.
extern "C" __global__ void wave_extend_dia(const float* n,const float* freq,
 float* energy,int nc,int nfx){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*nfx*ND)return;
 int c=z/(nfx*ND),f=(z/ND)%nfx,d=z%ND;
 if(f<NF)energy[z]=n[((long long)c*NF+f)*ND+d]*(2*PI*freq[f]);
 else energy[z]=n[((long long)c*NF+NF-1)*ND+d]*(2*PI*freq[NF-1])*powf(freq[f]/freq[NF-1],-5.f);
}
extern "C" __global__ void wave_contract_dia(const float* energy,const float* freq,
 float* n,int nc,int nfx){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int c=z/(NF*ND),f=(z/ND)%NF,d=z%ND;n[z]=energy[((long long)c*nfx+f)*ND+d]/(2*PI*freq[f]);
}
