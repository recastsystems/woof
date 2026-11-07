// Original finite-volume MUSCL/SSP-RK2 transport, no model implementation used.
// van Leer (1979), JCP 32, 101-136, doi:10.1016/0021-9991(79)90145-1.
// Shu & Osher (1988), JCP 77, 439-471, doi:10.1016/0021-9991(88)90177-5.
// Spherical balance: Holthuijsen (2007), Cambridge,
// doi:10.1017/CBO9780511618536. Metrics are prepared on the device.
#define PI 3.14159265358979323846f
__device__ float minmod3(float a,float b,float c){
 if(a*b<=0 || a*c<=0)return 0;
 return copysignf(fminf(fabsf(a),fminf(fabsf(b),fabsf(c))),a);
}
__device__ float slope(const float* q,const int* nb,const float* ds,
 int c,int axis,int bin,int ns){
 int l=nb[4*c+axis*2],r=nb[4*c+axis*2+1];
 if(l<0||r<0)return 0;
 float v=q[(long long)c*ns+bin];
 float dl=ds[4*c+axis*2],dr=ds[4*c+axis*2+1];
 float a=(v-q[(long long)l*ns+bin])/dl;
 float b=(q[(long long)r*ns+bin]-v)/dr;
 return minmod3(2*a,(a*dl+b*dr)/(dl+dr),2*b);
}
extern "C" __global__ void wave_depth_gradient(const float* h,const int* nb,
 const float* ds,const float* ca,const float* sa,float* east,float* north,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 float v[4];
 for(int j=0;j<4;j++){int b=nb[4*c+j];v[j]=b>=0&&h[b]>0?h[b]:h[c];}
 float gx=(v[1]-v[0])/(ds[4*c]+ds[4*c+1]);
 float gy=(v[3]-v[2])/(ds[4*c+2]+ds[4*c+3]);
 east[c]=ca[c]*gx-sa[c]*gy;north[c]=sa[c]*gx+ca[c]*gy;
}
extern "C" __global__ void wave_structured_transport(
 const float* q,const float* original,float* out,const float* cg,const float* depth,
 const int* nb,const int* boundary,const float* ne,const float* nn,
 const float* length,const float* distance,const float* area,const float* ca,const float* sa,
 const float* incident,float* budget,int n,int nf,int nd,float dt,float horizon,float dfrel,
 int final_stage,int phase){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 int ns=nf*nd;if(z>=(long long)n*ns)return;
 int c=z/ns,bin=z%ns,f=bin/nd,d=bin%nd;
 if(depth[c]<=0){if(phase)out[z]=0;else budget[z]=0;return;}
 float ct,st;sincosf(2*PI*d/nd,&st,&ct);
 float sum=0,center=q[z];
 for(int side=0;side<4;side++){
  int edge=4*c+side,b=nb[edge];bool wet=b>=0&&depth[b]>0;
  float speed=wet?.5f*(cg[c*nf+f]+cg[b*nf+f]):cg[c*nf+f];
  float vel=speed*(ct*ne[edge]+st*nn[edge]);
  float sign=(side%2)?1.f:-1.f;
  float a=center+sign*.5f*distance[edge]*slope(q,nb,distance,c,side/2,bin,ns);
  float other=0;
  if(wet)other=q[(long long)b*ns+bin]-sign*.5f*distance[edge]*slope(q,nb,distance,b,side/2,bin,ns);
  else if(b<0&&boundary[edge]>=0&&incident)other=incident[(long long)boundary[edge]*ns+bin];
  float flux=vel*(vel>=0?a:other);
  if(wet){
   // Booij & Holthuijsen (1987), JCP 68, 307-326,
   // doi:10.1016/0021-9991(87)90060-X. Full tensor, including cross derivative.
   // Fixed spreading horizon matches the Voronoi backend; it is a numerical
   // approximation to swell age and is recorded, not an observed age estimate.
   float along=speed*dfrel,across=speed*(2*PI/nd);
   float dss=along*along*horizon/12,dnn=across*across*horizon/12;
   float gxc=slope(q,nb,distance,c,0,bin,ns),gyc=slope(q,nb,distance,c,1,bin,ns);
   float gxb=slope(q,nb,distance,b,0,bin,ns),gyb=slope(q,nb,distance,b,1,bin,ns);
   float gx=.5f*(ca[c]*gxc-sa[c]*gyc+ca[b]*gxb-sa[b]*gyb);
   float gy=.5f*(sa[c]*gxc+ca[c]*gyc+sa[b]*gxb+ca[b]*gyb);
   float gn=(q[(long long)b*ns+bin]-center)/distance[edge];
   float correction=gn-gx*ne[edge]-gy*nn[edge];
   gx+=correction*ne[edge];gy+=correction*nn[edge];
   flux-=dnn*gn+(dss-dnn)*(ct*ne[edge]+st*nn[edge])*(ct*gx+st*gy);
  }
  flux*=length[edge];
  if(!phase)sum+=fmaxf(flux,0.f);
  else sum+=flux*(flux>=0?budget[z]:(wet?budget[(long long)b*ns+bin]:1.f));
 }
 if(!phase){budget[z]=sum>0?fminf(1.f,center*area[c]/(dt*sum)):1.f;return;}
 float next=center-dt*sum/area[c];
 out[z]=final_stage?.5f*(original[z]+next):next;
}
// Cached MUSCL slopes and one signed flux per shared face. Equations and
// limiter are those cited at the start of this file. Spectral tiles bound VRAM.
extern "C" __global__ void wave_structured_slopes(const float* q,const int* nb,
 const float* distance,float* slopes,int n,int ns,int first,int count){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)n*count)return;
 int c=z/count,b=first+z%count;
 slopes[z]=slope(q,nb,distance,c,0,b,ns);
 slopes[(long long)n*count+z]=slope(q,nb,distance,c,1,b,ns);
}
extern "C" __global__ void wave_structured_faces(const float* q,const float* slopes,
 const float* cg,const float* depth,const int* nb,const int* boundary,const int* owner,
 const float* ne,const float* nn,const float* length,const float* distance,
 const float* ca,const float* sa,const float* incident,float* flux,
 int n,int nface,int nf,int nd,int first,int count,float horizon,float dfrel){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)nface*count)return;
 int edge=owner[z/count],c=edge/4,side=edge%4,b=nb[edge],local=z%count,bin=first+local;
 bool here=depth[c]>0,wet=b>=0&&depth[b]>0;
 if(!here&&!wet){flux[z]=0;return;}
 int f=bin/nd,d=bin%nd,ns=nf*nd;long long cz=(long long)c*count+local;
 float ct,st;sincosf(2*PI*d/nd,&st,&ct);
 float speed=here?(wet?.5f*(cg[c*nf+f]+cg[b*nf+f]):cg[c*nf+f]):cg[b*nf+f];
 float vel=speed*(ct*ne[edge]+st*nn[edge]),sign=(side%2)?1.f:-1.f;
 float center=q[(long long)c*ns+bin];
 float a=here?center+sign*.5f*distance[edge]*slopes[(long long)(side/2)*n*count+cz]:0;
 float other=0;long long bz=(long long)b*count+local;
 if(wet)other=q[(long long)b*ns+bin]-sign*.5f*distance[edge]*slopes[(long long)(side/2)*n*count+bz];
 else if(b<0&&boundary[edge]>=0&&incident)other=incident[(long long)boundary[edge]*ns+bin];
 float value=vel*(vel>=0?a:other);
 if(here&&wet){float along=speed*dfrel,across=speed*(2*PI/nd);
  float dss=along*along*horizon/12,dnn=across*across*horizon/12;
  float gxc=slopes[cz],gyc=slopes[(long long)n*count+cz];
  float gxb=slopes[bz],gyb=slopes[(long long)n*count+bz];
  float gx=.5f*(ca[c]*gxc-sa[c]*gyc+ca[b]*gxb-sa[b]*gyb);
  float gy=.5f*(sa[c]*gxc+ca[c]*gyc+sa[b]*gxb+ca[b]*gyb);
  float gn=(q[(long long)b*ns+bin]-center)/distance[edge];
  float correction=gn-gx*ne[edge]-gy*nn[edge];
  gx+=correction*ne[edge];gy+=correction*nn[edge];
  value-=dnn*gn+(dss-dnn)*(ct*ne[edge]+st*nn[edge])*(ct*gx+st*gy);
 }
 flux[z]=value*length[edge];
}
extern "C" __global__ void wave_structured_gather(const float* q,const float* original,
 float* out,const float* depth,const int* nb,const int* face_id,const float* area,
 const float* flux,float* budget,int n,int ns,int first,int count,float dt,int final_stage,int phase){
 long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(z>=(long long)n*count)return;
 int c=z/count,local=z%count;long long qz=(long long)c*ns+first+local;
 if(depth[c]<=0){if(phase)out[qz]=0;else budget[z]=0;return;}
 float sum=0;
 for(int side=0;side<4;side++){int edge=4*c+side,b=nb[edge];
  float value=flux[(long long)face_id[edge]*count+local];
  if(side%2==0&&b>=0)value=-value;
  if(!phase)sum+=fmaxf(value,0.f);
  else sum+=value*(value>=0?budget[z]:(b>=0&&depth[b]>0?budget[(long long)b*count+local]:1.f));
 }
 if(!phase)budget[z]=sum>0?fminf(1.f,q[qz]*area[c]/(dt*sum)):1.f;
 else{float next=q[qz]-dt*sum/area[c];out[qz]=final_stage?.5f*(original[qz]+next):next;}
}
