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
extern "C" __global__ void wave_choose_basins(const float* variance,const float* wind,
 int* selection,int nc){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int r[3]={-1,-1,-1};float e[3]={0,0,0};
 for(int b=0;b<NF*ND;b++){long long z=(long long)c*NF*ND+b;
  float v=variance[z];if(v<=0||wind[z]>.333333333f*v)continue;
  for(int j=0;j<3;j++)if(v>e[j]){for(int l=2;l>j;l--){e[l]=e[l-1];r[l]=r[l-1];}e[j]=v;r[j]=b;break;}}
 for(int j=0;j<3;j++)selection[c*3+j]=r[j];
}
// Basin merging (HP01 section 2a) and the bulk/partition products are wave_merge_basins_ordered and
// wave_diagnose_ordered in ordered.cu: one owner thread per sum, fixed order. The float atomicAdd sums
// that were here made two identical runs differ already at t=0 (gpuwm-hex 0.4.14, 2026-11-06).
__device__ int partition_id(int root,int c,const float* variance,const float* wind,const int* select){
 long long z=(long long)c*NF*ND+root;
 if(wind[z]>.333333333f*variance[z])return 1;
 if(root==select[c*3])return 2;
 if(root==select[c*3+1])return 3;
 // Residual low-energy basins join the third swell, preserving total variance.
 return 4;
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
