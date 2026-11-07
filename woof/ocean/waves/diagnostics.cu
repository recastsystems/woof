// Original device reconstruction and watershed diagnostics.
// Hanson & Phillips (2001), JAOT 18, 277-293,
// doi:10.1175/1520-0426(2001)018<0277:AAOOSD>2.0.CO;2.
// Spectral shapes and coefficients are supplied by partition_shapes.v1.json.
#define PI 3.14159265358979323846f
#ifndef NF
#define NF 36
#endif
#ifndef ND
#define ND 36
#endif
__device__ int periodic(int d){return (d+ND)%ND;}
__device__ float bearing(float x,float y){return fmodf(630.f-atan2f(y,x)*180/PI,360.f);}
__device__ float shape(float f,int d,float tp,float dp,const float* p){
 float peak=1/tp,sigma=f<=peak?p[4]:p[5],x=f/peak;
 float bell=expf(-sqr((x-1)/sigma)*.5f);
 float theta=(270-dp)*PI/180,cs=fabsf(cosf((2*PI*d/ND-theta)*.5f));
 // Log evaluation avoids overflow for narrow swell or peaks outside the grid.
 return expf(-5*logf(x)-1.25f*powf(x,-4.f)+logf(p[3])*bell
             +2*p[6]*logf(fmaxf(cs,1e-30f)));
}
// rows = kind,tp_min,tp_max,gamma,sigma_a,sigma_b,s; upper limit exclusive.
extern "C" __global__ void wave_reconstruct(const float* partitions,
 const int* point_map,const float* freq,const float* df,const float* rows,
 float* energy,int* errors,int points,int nrow){
 int c=blockIdx.x,t=threadIdx.x;if(c>=points)return;
 __shared__ float sh[128];__shared__ float norm;__shared__ int row;
 for(int b=t;b<NF*ND;b+=blockDim.x)energy[(long long)c*NF*ND+b]=0;
 for(int part=0;part<4;part++){
  int src=point_map?point_map[c]:c;const float* pp=partitions+((long long)src*4+part)*4;
  float hs=pp[0],tp=pp[1],dp=pp[2],kind=pp[3];
  if(hs==0)continue;
  if(t==0){row=-1;
   if(isfinite(hs)&&hs>0&&isfinite(tp)&&tp>0&&isfinite(dp)&&(kind==0||kind==1))
    for(int r=0;r<nrow;r++)if(rows[7*r]==kind&&tp>=rows[7*r+1]&&tp<rows[7*r+2]){row=r;break;}
   if(row<0)atomicAdd(errors,1);
  }__syncthreads();if(row<0)continue;
  const float* p=rows+row*7;float total=0;
  for(int b=t;b<NF*ND;b+=blockDim.x)total+=shape(freq[b/ND],b%ND,tp,dp,p)*df[b/ND]*(2*PI/ND);
  float sum=sum_block(total,sh);if(t==0){norm=sum;if(!(sum>1e-30f))atomicAdd(errors,1);}__syncthreads();
  if(norm>1e-30f)for(int b=t;b<NF*ND;b+=blockDim.x)
   energy[(long long)c*NF*ND+b]+=hs*hs/(16*norm)*shape(freq[b/ND],b%ND,tp,dp,p);
  __syncthreads();
 }
}
// One pass of symmetric smoothing for basin topology only. Bulk values always
// integrate the original spectrum; no diagnostic smoothing changes state.
extern "C" __global__ void wave_smooth(const float* n,const float* freq,float* smooth,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int c=z/(NF*ND),f=(z/ND)%NF,d=z%ND;float sum=0,w=0;
 for(int j=-1;j<=1;j++)if(f+j>=0&&f+j<NF)for(int l=-1;l<=1;l++){
  float weight=(j==0?2.f:1.f)*(l==0?2.f:1.f);
  sum+=weight*n[((long long)c*NF+f+j)*ND+periodic(d+l)]*(2*PI*freq[f+j]);w+=weight;}
 smooth[z]=sum/w;
}
extern "C" __global__ void wave_watershed(const float* smooth,int* labels,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int c=z/(NF*ND),f=(z/ND)%NF,d=z%ND,b=z%(NF*ND),best=b;float top=smooth[z];
 for(int j=-1;j<=1;j++)if(f+j>=0&&f+j<NF)for(int l=-1;l<=1;l++){
  int candidate=(f+j)*ND+periodic(d+l);float value=smooth[(long long)c*NF*ND+candidate];
  if(value>top||(value==top&&candidate<best)){top=value;best=candidate;}}
 labels[z]=best;
}
extern "C" __global__ void wave_label_jump(const int* src,int* dest,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 long long base=z/(NF*ND)*(NF*ND);dest[z]=src[base+src[z]];
}
extern "C" __global__ void wave_basin_energy(const float* n,const int* label,
 const float* freq,const float* df,const float* k,const float* u,const float* v,
 float* variance,float* wind_variance,int nc){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nc*NF*ND)return;
 int c=z/(NF*ND),f=(z/ND)%NF,d=z%ND;float sig=2*PI*freq[f];
 float en=n[z]*sig*df[f]*(2*PI/ND);long long target=(long long)c*NF*ND+label[z];
 atomicAdd(variance+target,en);
 float projected=u[c]*cosf(2*PI*d/ND)+v[c]*sinf(2*PI*d/ND);
 if(k[c*NF+f]>0&&sig/k[c*NF+f]<1.2f*projected)atomicAdd(wind_variance+target,en);
}
extern "C" __global__ void wave_choose_basins(const float* variance,const float* wind,
 int* selection,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int r[3]={-1,-1,-1};float e[3]={0,0,0};
 for(int b=0;b<NF*ND;b++){long long z=(long long)c*NF*ND+b;
  float v=variance[z];if(v<=0||wind[z]>.333333333f*v)continue;
  for(int j=0;j<3;j++)if(v>e[j]){for(int l=2;l>j;l--){e[l]=e[l-1];r[l]=r[l-1];}e[j]=v;r[j]=b;break;}}
 for(int j=0;j<3;j++)selection[c*3+j]=r[j];
}
// HP01 section 2a: peak wave age, adjacent mutual peaks, updated moments after
// every merge. Model-spectrum settings kappa=.5, zeta=.75 (Table 1).
// Low-energy variance .0025 m2: Portilla et al. (2009), JAOT 26,107-122,
// doi:10.1175/2008JTECHO609.1, section 2a/Table 1. Merge, never discard.
// One block per cell keeps the watershed closure and basin statistics on chip.
extern "C" __global__ void wave_merge_basins(const float* n,const float* smooth,
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
  int f=b/ND,d=b%ND;float cs,ss;sincosf(2*PI*d/ND,&ss,&cs);
  wind[b]=k[c*NF+f]>0&&2*PI*freq[f]/k[c*NF+f]<1.5f*(u[c]*cs+v[c]*ss);
 }__syncthreads();
 for(int span=1;span<ns;span*=2){
  int changed=0;
  for(int b=t;b<ns;b+=blockDim.x){next[b]=root[root[b]];changed|=next[b]!=root[b];}
  // Stop only when every pointer is already a root. This is the same closure,
  // with no fixed extra passes after it has converged.
  if(!__syncthreads_or(changed))break;
  for(int b=t;b<ns;b+=blockDim.x)root[b]=next[b];__syncthreads();
 }
 for(int b=t;b<ns;b+=blockDim.x){int f=b/ND,d=b%ND,r=root[b];float cs,ss;
  sincosf(2*PI*d/ND,&ss,&cs);float e=n[base+b]*(2*PI*freq[f])*df[f]*(2*PI/ND);
  atomicAdd(energy+r,e);atomicAdd(mx+r,e*freq[f]*cs);atomicAdd(my+r,e*freq[f]*ss);
  atomicAdd(m2+r,e*freq[f]*freq[f]);
 }__syncthreads();
 for(int iteration=0;iteration<ns-1;iteration++){
  if(t==0)best=0;__syncthreads();
  for(int b=t;b<ns;b+=blockDim.x){int a=root[b],f=b/ND,d=b%ND;
   if(energy[a]<=0)continue;
   for(int j=-1;j<=1;j++)if(f+j>=0&&f+j<NF)for(int l=-1;l<=1;l++){
    int other=(f+j)*ND+periodic(d+l),r=root[other];if(a>=r||energy[r]<=0)continue;
    float ac,as,rc,rs;sincosf(2*PI*(a%ND)/ND,&as,&ac);sincosf(2*PI*(r%ND)/ND,&rs,&rc);
    float dist=sqr(freq[a/ND]*ac-freq[r/ND]*rc)+sqr(freq[a/ND]*as-freq[r/ND]*rs);
    float score=0;
    if(fminf(energy[a],energy[r])<.0025f)score=3.f+1.f/(1.f+dist);
    else if(!wind[a]&&!wind[r]){
     float spread=fmaxf(m2[a]/energy[a]-sqr(mx[a]/energy[a])-sqr(my[a]/energy[a]),
                       m2[r]/energy[r]-sqr(mx[r]/energy[r])-sqr(my[r]/energy[r]));
     if(dist<=.5f*spread)score=1.f+(.5f*spread-dist)/fmaxf(.5f*spread,1e-30f);
     float saddle=fminf(smooth[base+b],smooth[base+other]);
     float contrast=saddle/fmaxf(fminf(smooth[base+a],smooth[base+r]),1e-30f);
     if(contrast>=.75f)score=fmaxf(score,1.f+contrast);
    }
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
   sincosf(2*PI*(r%ND)/ND,&rs,&rc);
   for(int j=0;j<3;j++)if(selected[j]>=0){int a=selected[j];float ac,as;
    sincosf(2*PI*(a%ND)/ND,&as,&ac);
    float dist=sqr(freq[a/ND]*ac-freq[r/ND]*rc)+sqr(freq[a/ND]*as-freq[r/ND]*rs);
    if(dist<best_distance){best_distance=dist;part=j+2;}}
  }
  labels[base+b]=part;
 }
}
__device__ int partition_id(int root,int c,const float* variance,const float* wind,const int* select){
 long long z=(long long)c*NF*ND+root;
 if(wind[z]>.333333333f*variance[z])return 1;
 if(root==select[c*3])return 2;
 if(root==select[c*3+1])return 3;
 // Residual low-energy basins join the third swell, preserving total variance.
 return 4;
}
__device__ float peak_frequency(const float* es,const float* f,int pf){
 if(pf==0||pf==NF-1)return f[pf];
 float x0=f[pf-1],x1=f[pf],x2=f[pf+1];
 float a=((es[pf+1]-es[pf])/(x2-x1)-(es[pf]-es[pf-1])/(x1-x0))/(x2-x0);
 float b=(es[pf]-es[pf-1])/(x1-x0)-a*(x0+x1);
 if(a>=0||!isfinite(a)||fabsf(a)<1e-30f)return x1;
 return fminf(x2,fmaxf(x0,-b/(2*a)));
}
extern "C" __global__ void wave_diagnose(const float* n,const int* label,
 const float* freq,const float* df,const float* variance,const float* wind,
 const int* select,const float* moments,const int* wet,float* out,int nc,int nfull){
 int c=blockIdx.x,t=threadIdx.x;if(c>=nc)return;
 __shared__ float es[5*NF],xspec[5*NF],yspec[5*NF],sh[128];
 for(int j=t;j<5*NF;j+=blockDim.x){es[j]=0;xspec[j]=0;yspec[j]=0;}__syncthreads();
 for(int b=t;b<NF*ND;b+=blockDim.x){int f=b/ND,d=b%ND;long long z=(long long)c*NF*ND+b;
  int p=label[z];float a=2*PI*d/ND;
  float e=n[z]*(2*PI*freq[f])*(2*PI/ND);
  atomicAdd(es+f,e);atomicAdd(xspec+f,e*cosf(a));atomicAdd(yspec+f,e*sinf(a));
  atomicAdd(es+p*NF+f,e);atomicAdd(xspec+p*NF+f,e*cosf(a));atomicAdd(yspec+p*NF+f,e*sinf(a));
 }__syncthreads();
 if(t==0){int cell=wet[c];const float* mm=moments+c*16;
  for(int p=0;p<5;p++){
   float m0=0,m1=0,m2=0,x=0,y=0;int peak=0;
   for(int f=0;f<NF;f++){float e=es[p*NF+f]*df[f];m0+=e;m1+=e*freq[f];m2+=e*freq[f]*freq[f];
    x+=xspec[p*NF+f]*df[f];y+=yspec[p*NF+f]*df[f];if(es[p*NF+f]>es[p*NF+peak])peak=f;}
   float hs=m0>1e-15f?4*sqrtf(m0):0;
   float tp=hs?1/peak_frequency(es+p*NF,freq,peak):0;
   float dp=hs?bearing(xspec[p*NF+peak],yspec[p*NF+peak]):0;
   if(p==0){out[cell]=hs;out[nfull+cell]=tp;
    out[2*nfull+cell]=hs?m0/fmaxf(m1,1e-30f):0;
    out[3*nfull+cell]=hs?sqrtf(m0/fmaxf(m2,1e-30f)):0;
    out[4*nfull+cell]=dp;out[5*nfull+cell]=hs?bearing(x,y):0;
    out[6*nfull+cell]=hs?sqrtf(fmaxf(0.f,2*(1-hypotf(x,y)/m0)))*180/PI:0;
    out[22*nfull+cell]=hs?fmodf(bearing(x,y)+180,360.f):0;
   }else{int offset=7+(p-1)*3;out[offset*nfull+cell]=hs;out[(offset+1)*nfull+cell]=tp;out[(offset+2)*nfull+cell]=dp;}
  }
  out[19*nfull+cell]=mm[7];out[20*nfull+cell]=mm[9];out[21*nfull+cell]=mm[8];
  out[23*nfull+cell]=mm[10];out[24*nfull+cell]=mm[11];
 }
}
extern "C" __global__ void wave_station_spectra(const float* action,const int* compact,
 const float* freq,float* output,int count){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)count*NF*ND)return;
 int s=z/(NF*ND),bin=z%(NF*ND);output[z]=action[(long long)compact[s]*NF*ND+bin]*(2*PI*freq[bin/ND]);
}
extern "C" __global__ void wave_station_bulk(const float* output,const int* full,
 float* stations,int count,int nfull){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z<25*count)stations[z]=output[(z/count)*nfull+full[z%count]];
}
