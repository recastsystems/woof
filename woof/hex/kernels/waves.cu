// Original Apache-2.0: device exchange and conservative spectral remapping.
// NVRTC supplies CUDA math declarations without host system headers.
typedef long long int64_t;

extern "C" __global__ void hex_wave_forcing(const int *local,const int *global,
 const float *u,const float *v,const float *p,const float *ust,const float *density,
 float *ou,float *ov,float *op,float *oa,float *tx,float *ty,float *rho,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int a=local[i],b=global[i];float x=u[a],y=v[a],us=fmaxf(ust[a],0.f);
 float r=density[a];
 float stress=r*us*us/fmaxf(hypotf(x,y),1.e-12f);
 ou[b]=x;ov[b]=y;op[b]=p[a];oa[b]=us;tx[b]=stress*x;ty[b]=stress*y;rho[b]=r;
}

// Positive polygon-overlap remap of cell-average action. Direction rotation
// is parallel transport on the sphere; frequency measure is unchanged.
// Conservative remapping: Jones 1999, Monthly Weather Review 127, 2204-2210,
// doi:10.1175/1520-0493(1999)127<2204:FASOCR>2.0.CO;2.
extern "C" __global__ void hex_wave_remap(const float *old,const float *seed,float *out,
 const int *ptr,const int *column,const double *overlap,const double *target_area,
 const int *source_wet,const int *target_full,const double *slat,const double *slon,
 const double *tlat,const double *tlon,int nt,int nf,int nd){
 int64_t k=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=(int64_t)nt*nf*nd)return;
 int bin=k%(nf*nd),f=bin/nd,d=bin%nd,i=k/(nf*nd),target=target_full[i];
 double la=tlat[target],lo=tlon[target],cs=cos(la),sn=sin(la),cl=cos(lo),ss=sin(lo);
 double tn[3]={cs*cl,cs*ss,sn},te[3]={-ss,cl,0},ty[3]={-sn*cl,-sn*ss,cs};
 double theta=6.283185307179586*d/nd,v[3];
 for(int a=0;a<3;++a)v[a]=cos(theta)*te[a]+sin(theta)*ty[a];
 double value=0,covered=0;
 for(int j=ptr[target];j<ptr[target+1];++j){
  int c=column[j],wet=source_wet[c];double weight=overlap[j];if(wet<0)continue;
  // Formerly dry overlap has no spectrum to carry and needs incident seeding.
  covered+=weight;
  double sl=slat[c],so=slon[c],sc=cos(sl),st=sin(sl),lc=cos(so),ls=sin(so);
  double normal[3]={sc*lc,sc*ls,st},east[3]={-ls,lc,0},north[3]={-st*lc,-st*ls,sc};
  double dot=0,vn=0;for(int a=0;a<3;++a){dot+=tn[a]*normal[a];vn+=v[a]*normal[a];}
  double x=0,y=0;for(int a=0;a<3;++a){double w=v[a]-vn/(1.+dot)*(tn[a]+normal[a]);x+=w*east[a];y+=w*north[a];}
  double index=atan2(y,x)*nd/6.283185307179586;if(index<0)index+=nd;
  int d0=(int)floor(index);double blend=index-d0;d0%=nd;int d1=(d0+1)%nd;
  int64_t base=((int64_t)wet*nf+f)*nd;
  value+=weight*((1.-blend)*old[base+d0]+blend*old[base+d1]);
 }
 double area=target_area[target],uncovered=fmax(0.,area-covered);
 out[k]=(float)((value+uncovered*seed[k])/area);
}
