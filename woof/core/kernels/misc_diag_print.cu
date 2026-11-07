// WRF 4.7.1 module_diag_misc.F: diagnostic_output_calc. Public domain.
__device__ float md_value(int q,int c,const float* dps,const float* dmu,
 const float* rc,const float* rn,const float* ac,const float* an,
 const float* ev,const float* hf,const float* lh,float dt){
 switch(q){case 0:return fabsf(dps[c]);case 1:return fabsf(dmu[c]);
 case 2:return fabsf(rc[c]);case 3:return fabsf(rn[c]);
 case 4:return fabsf(rc[c])+fabsf(rn[c]);case 5:return fabsf(ac[c]);
 case 6:return fabsf(an[c]);case 7:return fabsf(ac[c])+fabsf(an[c]);
 case 8:return fabsf(ev[c]);case 9:return fabsf(hf[c]);case 10:return fabsf(lh[c]);
 case 11:return ac[c];case 12:return an[c];default:return fabsf(dmu[c]*dt);}
}
extern "C" __global__ void misc_print_fields(const float *ps,const float *mu,
 float *pk,float *oldmu,float *dps,float *dmu,float dt,int first,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 dps[c]=first?0.f:(ps[c]-pk[c])/dt;dmu[c]=first?0.f:(mu[c]-oldmu[c])/dt;
 pk[c]=ps[c];oldmu[c]=mu[c];
}
// A fast tree over independent columns, including only owned tile interiors.
extern "C" __global__ void misc_print_partial(const float *dps,const float *dmu,
 const float* rc,const float* rn,const float* ac,const float* an,
 const float* ev,const float* hf,const float* lh,float *out,int *outidx,float dt,
 int nx,int x0,int x1,int y0,int y1,int mode,int ox,int oy,int dnx){
 __shared__ float values[256];__shared__ int ix[256];int q=blockIdx.y,l=threadIdx.x;
 int w=x1-x0,n=w*(y1-y0),c0=blockIdx.x*256+l;
 float v=0.f;if(c0<n){int c=(y0+c0/w)*nx+x0+c0%w;
  if(mode==2||q<2||q==13)v=md_value(q,c,dps,dmu,rc,rn,ac,an,ev,hf,lh,dt);}
 values[l]=v;ix[l]=c0<n?((y0+c0/w+oy)*dnx+x0+c0%w+ox):2147483647;__syncthreads();
 for(int s=128;s;s>>=1){if(l<s){if(q>=11){if(values[l+s]>values[l]||(values[l+s]==values[l]&&ix[l+s]<ix[l])){values[l]=values[l+s];ix[l]=ix[l+s];}}else values[l]+=values[l+s];}__syncthreads();}
 if(!l){out[q*gridDim.x+blockIdx.x]=values[0];if(q>=11)outidx[(q-11)*gridDim.x+blockIdx.x]=ix[0];}
}
extern "C" __global__ void misc_print_finish(const float* partial,const int* indices,float* out,int* outidx,int blocks){
 __shared__ float value[128];__shared__ int ix[128];int q=blockIdx.x,l=threadIdx.x;float v=0.f;int id=2147483647;
 for(int k=l;k<blocks;k+=128){float a=partial[q*blocks+k];if(q>=11){int j=indices[(q-11)*blocks+k];if(a>v||(a==v&&j<id)){v=a;id=j;}}else v+=a;}
 value[l]=v;ix[l]=id;__syncthreads();for(int s=64;s;s>>=1){if(l<s){if(q>=11){if(value[l+s]>value[l]||(value[l+s]==value[l]&&ix[l+s]<ix[l])){value[l]=value[l+s];ix[l]=ix[l+s];}}else value[l]+=value[l+s];}__syncthreads();}
 if(!l){out[q]=q==13?value[0]*1.e-5f:value[0];if(q>=11)outidx[q-11]=ix[0];}
}
// Strict oracle only: preserve each MPI patch's native FP32 left fold.
// Default forecasts use the parallel kernels above.
extern "C" __global__ void misc_print_native_order(const float *dps,const float *dmu,
 const float* rc,const float* rn,const float* ac,const float* an,
 const float* ev,const float* hf,const float* lh,float *out,float dt,int n){
 int q=threadIdx.x;if(q>=14)return;float s=0.f;
 for(int c=0;c<n;c++){
  if(q==4){s=s+fabsf(rc[c]);s=s+fabsf(rn[c]);}
  else if(q==7){s=s+fabsf(ac[c]);s=s+fabsf(an[c]);}
  else{float v=md_value(q,c,dps,dmu,rc,rn,ac,an,ev,hf,lh,dt);s=q>=11?fmaxf(s,v):s+v;}
 }
 out[q]=q==13?s*1.e-5f:s;
}

// moist_physics_prep_em replaces P8W after RK and before microphysics.
// This surface value must survive the post-microphysics EOS refresh.
extern "C" __global__ void misc_print_surface_pressure(const float *p,
 const float *ph,const float *phb,float *ps,int n,int phb3){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 float a=(ph[c]+phb[phb3?c:0])/9.81f;
 float b=(ph[n+c]+phb[phb3?n+c:1])/9.81f;
 float d=(ph[2*n+c]+phb[phb3?2*n+c:2])/9.81f;
 float z1=.5f*(a+b),z2=.5f*(b+d);
 float w1=(a-z2)/(z1-z2),w2=1.f-w1;
 ps[c]=w1*p[c]+w2*p[n+c];
}
