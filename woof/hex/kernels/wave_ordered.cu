// Fixed-order reductions for the spectral wave model: the DIA quartet exchange and the partition sums.
// Original Apache-2.0. The equations are those of dia_bin (kernels.cu) and of the partition diagnostics
// (diagnostics.cu): Hasselmann et al. (1985), JPO 15, 1378-1391, doi:10.1175/1520-0485(1985)015<1378:CAPOTN>2.0.CO;2;
// Hanson & Phillips (2001), JAOT 18, 277-293. Only the order of the sums is new, and it is fixed.
//
// THE BREAKAGE THIS PREVENTS (gpuwm-hex 0.4.14, 2026-11-06): those sums used float atomicAdd, so threads added
// in whatever order they arrived. Two identical 1-card runs (x1.40962 120 km, waves on, 3 h) differed in
// 4,863,958 of 44,807,348 wave words, 22,315 of them already in the t=0 frame, and with roughness feedback on
// the atmosphere of 2 rank processes vs 2 rank threads differed in 386,196,279 words. Here every output word
// has one owner thread that adds its terms in a fixed order: the same bits on every run and every executor.
//
// Self-contained (its own helpers), so gpuwm and gpuwm-hex compile the same bytes into the same kernels.
#ifndef ORDERED_PI
#define ORDERED_PI 3.14159265358979323846f
#endif
#ifndef DIA_COEFFICIENT
#define DIA_COEFFICIENT 3e7f
#endif
#ifndef NF
#define NF 36
#endif
#ifndef ND
#define ND 36
#endif
#define ORDERED_G 9.80665f
#define DIA_CORNERS 16   // per source bin: 2 mirrors x 2 legs (f+, f-) x 4 bilinear corners

__device__ int ordered_wrap(int j,int n){return (j%n+n)%n;}
__device__ float ordered_sample(const float* e,int nd,float fi,float di){
 int f=(int)floorf(fi),d=(int)floorf(di);float a=fi-f,b=di-d;
 return (1-a)*((1-b)*e[f*nd+ordered_wrap(d,nd)]+b*e[f*nd+ordered_wrap(d+1,nd)])+
 a*((1-b)*e[(f+1)*nd+ordered_wrap(d,nd)]+b*e[(f+1)*nd+ordered_wrap(d+1,nd)]);
}
__device__ float ordered_donor_limit(const float* e,const float* freq,const float* df,int nd,
 float fi,float di,float dt){
 int f=(int)floorf(fi),d=(int)floorf(di);float a=fi-f,b=di-d,lim=1e30f;
 for(int j=0;j<2;j++)for(int l=0;l<2;l++){
  float w=(j?a:1-a)*(l?b:1-b);if(w<=1e-8f)continue;
  int ff=f+j,dd=ordered_wrap(d+l,nd);
  lim=fminf(lim,.2f*e[ff*nd+dd]*df[ff]*(2*ORDERED_PI/nd)/(2*ORDERED_PI*freq[ff]*w*dt));
 }return lim;
}

// ---- DIA: the quartet exchange as a gather -------------------------------------------------------------------
// Run once per (frequency table, nd). For each source bin: the quartet geometry (fp, dp, fm, dm per mirror) and
// the 16 target bins of its bilinear corners (-1 for a bin whose quartet leaves the frequency range). The host
// inverts the target table once into a destination-sorted list, so each destination later adds its terms in one
// fixed order. The geometry is cell-independent; only the spectrum and the depth scaling are per cell.
extern "C" __global__ void wave_dia_quartets(const float* freq,int nf,int nd,float logq,
 float* geometry,int* targets){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=nf*nd)return;
 int d=z%nd,f=z/nd;
 float fp=f+logf(1.25f)/logq,fm=f+logf(.75f)/logq;
 bool valid=!(fm<0 || fp>=nf-1);
 if(valid){
  // Interpolation linear in physical frequency preserves action and energy.
  int ip=(int)fp,im=(int)fm;
  fp=ip+(1.25f*freq[f]-freq[ip])/(freq[ip+1]-freq[ip]);
  fm=im+(.75f*freq[f]-freq[im])/(freq[im+1]-freq[im]);
 }
 for(int mi=0;mi<2;mi++){int mirror=2*mi-1;
  float dp=d+mirror*11.48f/360*nd,dm=d-mirror*33.56f/360*nd;
  float* g=geometry+((long long)z*2+mi)*4;g[0]=fp;g[1]=dp;g[2]=fm;g[3]=dm;
  for(int leg=0;leg<2;leg++){
   float loc=leg?fm:fp,ang=leg?dm:dp;int ff=(int)floorf(loc),dd=(int)floorf(ang);
   for(int j=0;j<2;j++)for(int l=0;l<2;l++)
    targets[(long long)z*DIA_CORNERS+mi*8+leg*4+j*2+l]=valid?(ff+j)*nd+ordered_wrap(dd+l,nd):-1;
  }
 }
}
// One block per wet cell (grid-stride over cells). work = the cell's spectrum, each source bin's action per
// mirror and, with a diagonal, each leg's derivative (words = 3 or 7 floats per bin): dynamic shared memory, or
// a per-block global slab where the spectrum is too large for shared memory. Phase 1: each bin's own quartet
// (the source's own terms go to its own words; one owner thread each). Phase 2: each bin gathers what the
// quartets of other bins give it, in the fixed list order. Adds onto rate/diagonal as the scatter version did.
extern "C" __global__ void wave_dia_ordered(const float* e,float* rate,float* diagonal,
 const float* freq,const float* df,const float* depth,const float* m,const float* geometry,const int* targets,
 const int* offsets,const int* entries,float* slab,int nc,int nf,int nd,float dt,int limited,int words){
 extern __shared__ float dynamic_work[];
 const int ns=nf*nd;const bool with_diagonal=diagonal!=nullptr;
 float* work=slab?slab+(long long)blockIdx.x*words:dynamic_work;
 float* en=work;float* act=work+ns;float* der=work+3*ns;
 const float da=2*ORDERED_PI/nd;
 const float ap=1/powf(1.25f,4.f),am=1/powf(.75f,4.f),ac=1/powf(.9375f,4.f);
 for(int c=blockIdx.x;c<nc;c+=gridDim.x){
  if(depth[c]<=0)continue;   // uniform over the block
  const long long base=(long long)c*ns;
  for(int z=threadIdx.x;z<ns;z+=blockDim.x)en[z]=e[base+z];
  __syncthreads();
  const float kd=fmaxf(.5f,.75f*m[c*10+2]*depth[c]);
  const float dep=1+5.5f/kd*(1-5.f/6.f*kd)*expf(-1.25f*kd);
  for(int z=threadIdx.x;z<ns;z+=blockDim.x){
   const int f=z/nd;const bool valid=targets[(long long)z*DIA_CORNERS]>=0;
   float own_rate=0,own_diagonal=0;
   for(int mi=0;mi<2;mi++){
    float action=0,dplus=0,dminus=0;
    if(valid){
     const float* g=geometry+((long long)z*2+mi)*4;
     const float fp=g[0],dp=g[1],fm=g[2],dm=g[3];
     const float e0=en[z],f0=freq[f],sig=2*ORDERED_PI*f0;
     const float ep=ordered_sample(en,nd,fp,dp),em=ordered_sample(en,nd,fm,dm);
     const float coeff=DIA_COEFFICIENT/powf(ORDERED_G,4.f)*powf(f0,11.f)*dep;
     const float transfer=coeff*(e0*e0*(ep*ap+em*am)-2*e0*ep*em*ac);
     action=transfer*df[f]*da/sig;
     // Limit a whole resonant exchange, never its individual legs.
     if(limited && action>=0){action=fminf(action,.2f*e0*df[f]*da/(2*sig*dt));}
     else if(limited){float available=fminf(ordered_donor_limit(en,freq,df,nd,fp,dp,dt),
      ordered_donor_limit(en,freq,df,nd,fm,dm,dt));action=-fminf(-action,available);}
     own_rate+=-2*action*sig/(df[f]*da);
     if(with_diagonal){
      // Diagonal of the TOTAL source Jacobian of this same discrete quartet (HJ99 source integration).
      own_diagonal+=-2*coeff*(2*e0*(ep*ap+em*am)-2*ep*em*ac);
      dplus=coeff*(e0*e0*ap-2*e0*em*ac);dminus=coeff*(e0*e0*am-2*e0*ep*ac);
     }
    }
    act[mi*ns+z]=action;
    if(with_diagonal){der[(mi*2)*ns+z]=dplus;der[(mi*2+1)*ns+z]=dminus;}
   }
   if(valid){rate[base+z]+=own_rate;if(with_diagonal)diagonal[base+z]+=own_diagonal;}
  }
  __syncthreads();
  for(int z=threadIdx.x;z<ns;z+=blockDim.x){
   const int fd=z/nd,first=offsets[z],last=offsets[z+1];
   float gathered_rate=0,gathered_diagonal=0;
   for(int i=first;i<last;i++){
    const int code=entries[i],s=code/DIA_CORNERS,k=code%DIA_CORNERS;
    const int mi=k>>3,leg=(k>>2)&1,j=(k>>1)&1,l=k&1;
    const float* g=geometry+((long long)s*2+mi)*4;
    const float loc=g[leg*2],ang=g[leg*2+1];
    const int ff=(int)floorf(loc),dd=(int)floorf(ang);
    const float a=loc-ff,b=ang-dd,w=(j?a:1-a)*(l?b:1-b);
    gathered_rate+=act[mi*ns+s]*w*(2*ORDERED_PI*freq[fd])/(df[fd]*da);
    if(with_diagonal){const int fs=s/nd;
     gathered_diagonal+=der[(mi*2+leg)*ns+s]*w*w*df[fs]*freq[fd]/(freq[fs]*df[fd]);}
   }
   if(last>first){rate[base+z]+=gathered_rate;if(with_diagonal)diagonal[base+z]+=gathered_diagonal;}
  }
  __syncthreads();
 }
}

// ---- Partition diagnostics: basin sums and the partition spectra ------------------------------------------
__device__ int ordered_periodic(int d){return (d+ND)%ND;}
__device__ float ordered_sqr(float x){return x*x;}
__device__ float ordered_bearing(float x,float y){return fmodf(630.f-atan2f(y,x)*180/ORDERED_PI,360.f);}
__device__ float ordered_peak_frequency(const float* es,const float* f,int pf){
 if(pf==0||pf==NF-1)return f[pf];
 float x0=f[pf-1],x1=f[pf],x2=f[pf+1];
 float a=((es[pf+1]-es[pf])/(x2-x1)-(es[pf]-es[pf-1])/(x1-x0))/(x2-x0);
 float b=(es[pf]-es[pf-1])/(x1-x0)-a*(x0+x1);
 if(a>=0||!isfinite(a)||fabsf(a)<1e-30f)return x1;
 return fminf(x2,fmaxf(x0,-b/(2*a)));
}
// HP01 section 2a: peak wave age, adjacent mutual peaks, updated moments after every merge. Model-spectrum
// settings kappa=.5, zeta=.75 (Table 1). Low-energy variance .0025 m2: Portilla et al. (2009), JAOT 26, 107-122,
// doi:10.1175/2008JTECHO609.1, section 2a/Table 1. Merge, never discard. One block per cell (blockDim a multiple
// of 32) keeps the watershed closure and basin statistics on chip.
extern "C" __global__ void wave_merge_basins_ordered(const float* n,const float* smooth,
 int* labels,const float* freq,const float* df,const float* k,
 const float* u,const float* v,float* workspace,int nc){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;
 const int ns=NF*ND;long long base=(long long)c*ns;
 // The exposed spectral dimensions can exceed a block's shared-memory limit.
 // Use the same block algorithm in tiled global scratch in that case.
#if NF*ND <= 1600
 __shared__ int root[NF*ND],next[NF*ND],wind[NF*ND];
 __shared__ float energy[NF*ND],mx[NF*ND],my[NF*ND],m2[NF*ND];
#else
 int* root=(int*)(workspace+(long long)c*ns*7);int* next=root+ns;int* wind=next+ns;
 float* energy=(float*)(wind+ns);float* mx=energy+ns;float* my=mx+ns;float* m2=my+ns;
#endif
 __shared__ int selected[3];
 __shared__ unsigned long long best;
 for(int b=t;b<ns;b+=blockDim.x){root[b]=labels[base+b];energy[b]=mx[b]=my[b]=m2[b]=0;
  int f=b/ND,d=b%ND;float cs,ss;sincosf(2*ORDERED_PI*d/ND,&ss,&cs);
  wind[b]=k[c*NF+f]>0&&2*ORDERED_PI*freq[f]/k[c*NF+f]<1.5f*(u[c]*cs+v[c]*ss);
 }__syncthreads();
 for(int span=1;span<ns;span*=2){
  int changed=0;
  for(int b=t;b<ns;b+=blockDim.x){next[b]=root[root[b]];changed|=next[b]!=root[b];}
  // Stop only when every pointer is already a root.
  if(!__syncthreads_or(changed))break;
  for(int b=t;b<ns;b+=blockDim.x)root[b]=next[b];__syncthreads();
 }
 // Basin sums in a fixed order: one warp per root, its lanes stride the bins in index order, then a fixed
 // shuffle tree; lane 0 of that warp is the only writer of the root's sums. next[] is free once the closure
 // has converged and holds each bin's variance (as float bits).
 for(int b=t;b<ns;b+=blockDim.x){int f=b/ND;
  next[b]=__float_as_int(n[base+b]*(2*ORDERED_PI*freq[f])*df[f]*(2*ORDERED_PI/ND));}
 __syncthreads();
 {const int lane=t&31,warp=t>>5,warps=blockDim.x>>5;
  for(int r=warp;r<ns;r+=warps){if(root[r]!=r)continue;   // uniform over the warp
   float se=0,sx=0,sy=0,s2=0;
   for(int b=lane;b<ns;b+=32)if(root[b]==r){int f=b/ND,d=b%ND;float cs,ss;
    sincosf(2*ORDERED_PI*d/ND,&ss,&cs);float e=__int_as_float(next[b]);
    se+=e;sx+=e*freq[f]*cs;sy+=e*freq[f]*ss;s2+=e*freq[f]*freq[f];}
   for(int o=16;o;o>>=1){se+=__shfl_down_sync(0xffffffffu,se,o);sx+=__shfl_down_sync(0xffffffffu,sx,o);
    sy+=__shfl_down_sync(0xffffffffu,sy,o);s2+=__shfl_down_sync(0xffffffffu,s2,o);}
   if(lane==0){energy[r]=se;mx[r]=sx;my[r]=sy;m2[r]=s2;}
  }}
 __syncthreads();
 for(int iteration=0;iteration<ns-1;iteration++){
  if(t==0)best=0;__syncthreads();
  for(int b=t;b<ns;b+=blockDim.x){int a=root[b],f=b/ND,d=b%ND;
   if(energy[a]<=0)continue;
   for(int j=-1;j<=1;j++)if(f+j>=0&&f+j<NF)for(int l=-1;l<=1;l++){
    int other=(f+j)*ND+ordered_periodic(d+l),r=root[other];if(a>=r||energy[r]<=0)continue;
    float ac,as,rc,rs;sincosf(2*ORDERED_PI*(a%ND)/ND,&as,&ac);sincosf(2*ORDERED_PI*(r%ND)/ND,&rs,&rc);
    float dist=ordered_sqr(freq[a/ND]*ac-freq[r/ND]*rc)+ordered_sqr(freq[a/ND]*as-freq[r/ND]*rs);
    float score=0;
    if(fminf(energy[a],energy[r])<.0025f)score=3.f+1.f/(1.f+dist);
    else if(!wind[a]&&!wind[r]){
     float spread=fmaxf(m2[a]/energy[a]-ordered_sqr(mx[a]/energy[a])-ordered_sqr(my[a]/energy[a]),
                       m2[r]/energy[r]-ordered_sqr(mx[r]/energy[r])-ordered_sqr(my[r]/energy[r]));
     if(dist<=.5f*spread)score=1.f+(.5f*spread-dist)/fmaxf(.5f*spread,1e-30f);
     float saddle=fminf(smooth[base+b],smooth[base+other]);
     float contrast=saddle/fmaxf(fminf(smooth[base+a],smooth[base+r]),1e-30f);
     if(contrast>=.75f)score=fmaxf(score,1.f+contrast);
    }
    // An integer maximum: the same winner whatever the arrival order.
    if(score>0){unsigned int pair=(unsigned int)(a*ns+r);
     atomicMax(&best,((unsigned long long)__float_as_uint(score)<<32)|pair);}
   }
  }__syncthreads();if(best==0)break;
  unsigned int pair=(unsigned int)best;int a=pair/ns,r=pair%ns;
  // Keep the stronger peak as representative, including its wave-age class.
  if(smooth[base+a]<smooth[base+r]){int swap=a;a=r;r=swap;}
  if(t==0){energy[a]+=energy[r];mx[a]+=mx[r];my[a]+=my[r];m2[a]+=m2[r];energy[r]=0;}
  for(int b=t;b<ns;b+=blockDim.x)if(root[b]==r)root[b]=a;
  __syncthreads();
 }
 if(t==0){float e[3]={0,0,0};for(int j=0;j<3;j++)selected[j]=-1;
  for(int b=0;b<ns;b++)if(!wind[b]&&energy[b]>0)
   for(int j=0;j<3;j++)if(energy[b]>e[j]){
    for(int l=2;l>j;l--){e[l]=e[l-1];selected[l]=selected[l-1];}
    e[j]=energy[b];selected[j]=b;break;}
 }__syncthreads();
 for(int b=t;b<ns;b+=blockDim.x){int r=root[b],part=1;
  if(!wind[r]){float best_distance=1e30f,rc,rs;
   sincosf(2*ORDERED_PI*(r%ND)/ND,&rs,&rc);
   for(int j=0;j<3;j++)if(selected[j]>=0){int a=selected[j];float ac,as;
    sincosf(2*ORDERED_PI*(a%ND)/ND,&as,&ac);
    float dist=ordered_sqr(freq[a/ND]*ac-freq[r/ND]*rc)+ordered_sqr(freq[a/ND]*as-freq[r/ND]*rs);
    if(dist<best_distance){best_distance=dist;part=j+2;}}
  }
  labels[base+b]=part;
 }
}
// Bulk and partition products. The per-frequency spectra of the total (slot 0) and of each partition: one owner
// thread per frequency adds its directions in order.
extern "C" __global__ void wave_diagnose_ordered(const float* n,const int* label,
 const float* freq,const float* df,const float* variance,const float* wind,
 const int* select,const float* moments,const int* wet,float* out,int nc,int nfull){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;
 __shared__ float es[5*NF],xspec[5*NF],yspec[5*NF];
 for(int j=t;j<5*NF;j+=blockDim.x){es[j]=0;xspec[j]=0;yspec[j]=0;}__syncthreads();
 for(int f=t;f<NF;f+=blockDim.x){float total=0,x_total=0,y_total=0;
  for(int d=0;d<ND;d++){long long z=(long long)c*NF*ND+f*ND+d;
   int p=label[z];float a=2*ORDERED_PI*d/ND;
   float e=n[z]*(2*ORDERED_PI*freq[f])*(2*ORDERED_PI/ND);float ex=e*cosf(a),ey=e*sinf(a);
   total+=e;x_total+=ex;y_total+=ey;
   es[p*NF+f]+=e;xspec[p*NF+f]+=ex;yspec[p*NF+f]+=ey;
  }
  es[f]+=total;xspec[f]+=x_total;yspec[f]+=y_total;
 }__syncthreads();
 if(t==0){int cell=wet[c];const float* mm=moments+c*16;
  for(int p=0;p<5;p++){
   float m0=0,m1=0,m2=0,x=0,y=0;int peak=0;
   for(int f=0;f<NF;f++){float e=es[p*NF+f]*df[f];m0+=e;m1+=e*freq[f];m2+=e*freq[f]*freq[f];
    x+=xspec[p*NF+f]*df[f];y+=yspec[p*NF+f]*df[f];if(es[p*NF+f]>es[p*NF+peak])peak=f;}
   float hs=m0>1e-15f?4*sqrtf(m0):0;
   float tp=hs?1/ordered_peak_frequency(es+p*NF,freq,peak):0;
   float dp=hs?ordered_bearing(xspec[p*NF+peak],yspec[p*NF+peak]):0;
   if(p==0){out[cell]=hs;out[nfull+cell]=tp;
    out[2*nfull+cell]=hs?m0/fmaxf(m1,1e-30f):0;
    out[3*nfull+cell]=hs?sqrtf(m0/fmaxf(m2,1e-30f)):0;
    out[4*nfull+cell]=dp;out[5*nfull+cell]=hs?ordered_bearing(x,y):0;
    out[6*nfull+cell]=hs?sqrtf(fmaxf(0.f,2*(1-hypotf(x,y)/m0)))*180/ORDERED_PI:0;
    out[22*nfull+cell]=hs?fmodf(ordered_bearing(x,y)+180,360.f):0;
   }else{int offset=7+(p-1)*3;out[offset*nfull+cell]=hs;out[(offset+1)*nfull+cell]=tp;out[(offset+2)*nfull+cell]=dp;}
  }
  out[19*nfull+cell]=mm[7];out[20*nfull+cell]=mm[9];out[21*nfull+cell]=mm[8];
  out[23*nfull+cell]=mm[10];out[24*nfull+cell]=mm[11];
 }
}
