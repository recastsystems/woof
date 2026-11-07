// WRF v4.6.1 phys/module_sf_3dpwp.F port. See WRF-NOTICE.txt.
// Layout: level, y, x. All circulation terms remain enabled.
#ifndef OCEAN_LEVELS
#define OCEAN_LEVELS 30
#endif
#define L OCEAN_LEVELS
__device__ float sigma(float tt,float s) {
 float t=tt-273.15f,sg=((6.76786136e-6f*s-4.8249614e-4f)*s+0.814876577f)*s-0.0934458632f;
 return ((((-1.43803061e-7f*t-1.98248399e-3f)*t-0.545939111f)*t+4.53168426f)*t)/(t+67.26f)
 +((((1.667e-8f*t-8.164e-7f)*t+1.803e-5f)*t)*sg+((-1.0843e-6f*t+9.8185e-5f)*t-4.7867e-3f)*t+1.f)*sg;
}
__device__ void mix(float *a,const float *z,int m) {
 float sum=0.f,depth=0.f;
 for(int k=0;k<m;k++){float dz=z[k]-(k?z[k-1]:0.f);sum+=a[k]*dz;depth+=dz;}
 for(int k=0;k<m;k++)a[k]=sum/depth;
}
__device__ int mld(const float *r) {
 for(int k=0;k<L-1;k++)if(fabsf(r[k+1]-r[k])>1.e-5f)return k+1;
 return L;
}
__device__ float depth_mld(const float *z,int m){return z[m-1];}
__device__ void rotate(float *u,float *v,float sa,float ca){
 for(int k=0;k<L;k++){float a=u[k],b=v[k];u[k]=a*ca+b*sa;v[k]=b*ca-a*sa;}
}
__device__ void stir(float *a,const float *z,int k,float ff){
 float dz=z[k]-(k?z[k-1]:0.f),dn=z[k+1]-z[k],da=(a[k+1]-a[k])*ff*0.5f;
 // WRF declares b INTEGER. REAL prevents truncation from leaving a
 // nonuniform-depth interface unmixed forever.
 float b=2.f/(1.f+dz/dn);a[k+1]-=da*b*dz/dn;a[k]+=da*b;
}
__device__ float richardson(const float *r,const float *z,const float *u,const float *v,int k){
 float dz=(z[k+1]-(k?z[k-1]:0.f))*0.5f,du=u[k+1]-u[k],dv=v[k+1]-v[k];
 return 9.8f*dz*fmaxf(r[k+1]-r[k],1.e-3f)/(1000.f*fmaxf(du*du+dv*dv,1.e-6f));
}
extern "C" __global__ void pwp_geometry(
 const float *t,const float *s,const float *u,const float *v,const float *z,
 const float *tr,const float *sr,const float *land,const float *mx,const float *my,
 float *w,float *p,float rdx,float rdy,int nx,int ny){
 int i=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;if(i>=n)return;
 int x=i%nx,y=i/nx,cx=min(max(x,1),nx-2),cy=min(max(y,1),ny-2),c=cy*nx+cx;
 bool interior=land[i]>1.5f&&land[c]>1.5f&&land[c+1]>1.5f&&land[c-1]>1.5f&&land[c+nx]>1.5f&&land[c-nx]>1.5f;
 float div=0.f,press=0.f;
 for(int k=0;k<L;k++){
  int q=k*n+c;float dz=z[q]-(k?z[q-n]:0.f);
  if(interior)div+=dz*((u[q+1]-u[q-1])*rdx*mx[c]+(v[q+nx]-v[q-nx])*rdy*my[c])*0.5f;
  w[k*n+i]=div;
 }
 for(int k=L-1;k>=0;k--){
  int q=k*n+i;float dz=z[q]-(k?z[q-n]:0.f);
  float alpha=sigma(tr[q]+0.5f,sr[q])-sigma(tr[q]-0.5f,sr[q]);
  float beta=sigma(tr[q],sr[q]+0.5f)-sigma(tr[q],sr[q]-0.5f);
  if(land[i]>1.5f)press-=9.8f*dz*(alpha*(t[q]-tr[q])+beta*(s[q]-sr[q]));p[q]=press;
 }
}
__device__ float transport(const float *a,const float *z,const float *w,
 const float *land,const float *mx,const float *my,float u,float v,
 int i,int k,int nx,int ny,float rdx,float rdy){
 int n=nx*ny,q=k*n+i,x=i%nx,y=i/nx;float tendency=0.f;
 if(x==0||x==nx-1||y==0||y==ny-1){
  if(x==0&&land[i+1]>1.5f)tendency-=1.5f*rdx*mx[i]*(a[q]-a[q+1]);
  if(x==nx-1&&land[i-1]>1.5f)tendency-=1.5f*rdx*mx[i]*(a[q]-a[q-1]);
  if(y==0&&land[i+nx]>1.5f)tendency-=1.5f*rdy*my[i]*(a[q]-a[q+nx]);
  if(y==ny-1&&land[i-nx]>1.5f)tendency-=1.5f*rdy*my[i]*(a[q]-a[q-nx]);return tendency;
 }
 int ix=u>=0.f?-1:1,iy=v>=0.f?-nx:nx;
 if(land[i+ix]>1.5f)tendency-=fabsf(u)*(a[q]-a[q+ix])*rdx*mx[i];
 if(land[i+iy]>1.5f)tendency-=fabsf(v)*(a[q]-a[q+iy])*rdy*my[i];
 // w > 0 is upward (divergence integrated from the surface). The column has no
 // water below its floor, so water rising through the floor carries the bottom
 // layer's own properties (zero-gradient inflow) and adds nothing there. WRF's
 // one-sided k-1 difference at the floor is downwind for upwelling and grows any
 // bottom gradient at rate w/dz without bound; convective mixing then carried that
 // runaway (246-308 K bottom water) to the surface on the SF Bay 250 m nest
 // (HRRR 2026-10-21T12, RRTMGP refused TSK at +41.85 h).
 if(k>0){int km=q-n,kp=k==L-1?q:q+n;float wv=0.5f*(w[q-n]+w[q]);
  if(k<L-1||wv<0.f)tendency+=wv*(a[km]-a[kp])/(z[km]-z[kp]);}
 return tendency;
}
extern "C" __global__ void pwp_transport(
 const float *t,const float *s,const float *u,const float *v,const float *z,
 const float *w,const float *p,const float *land,const float *mx,const float *my,
 float *tend,float rdx,float rdy,int nx,int ny,int first){
 int q=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;if(q>=n*L)return;
 int i=q%n,k=q/n,x=i%nx,y=i/nx;
 if(first)for(int a=0;a<4;a++)tend[a*n*L+q]=0.f;
 if(land[i]<=1.5f)return;
 const float *a[4]={t,s,u,v};
 for(int j=0;j<4;j++)tend[j*n*L+q]+=transport(a[j],z,w,land,mx,my,u[q],v[q],i,k,nx,ny,rdx,rdy);
 if(x>0&&x<nx-1&&y>0&&y<ny-1){
  if(land[i+1]>1.5f&&land[i-1]>1.5f)tend[2*n*L+q]-=(p[q+1]-p[q-1])*rdx*mx[i]/2046.f;
  if(land[i+nx]>1.5f&&land[i-nx]>1.5f)tend[3*n*L+q]-=(p[q+nx]-p[q-nx])*rdy*my[i]/2046.f;
 }
}
extern "C" __global__ void pwp_column(
 float *tout,float *sout,float *uout,float *vout,float *density,float *mlout,float *tsk,
 const float *base,const float *zall,const float *trall,const float *srall,
 const float *hfx,const float *lh,const float *sw,const float *lw,const float *qfx,
 const float *ust,const float *ua,const float *va,const float *rhoair,const float *f,const float *land,
 const float *ice_mask,float *tend,int *status,float dt,float stage_dt,float stbolt,int coupled_ice,int n){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||land[i]<=1.5f)return;
 float t[L],s[L],u[L],v[L],r[L],z[L],ri[L-1];
 for(int k=0;k<L;k++){int q=k*n+i;t[k]=base[q];s[k]=base[n*L+q];u[k]=base[2*n*L+q];v[k]=base[3*n*L+q];z[k]=zall[q];}
 float tr=trall[i],sr=srall[i],dr=sigma(tr,sr)+1000.f;
 float alpha=sigma(tr+0.5f,sr)-sigma(tr-0.5f,sr),beta=sigma(tr,sr+0.5f)-sigma(tr,sr-0.5f);
 // Where the sea-ice step set this column's surface fluxes (ice present), it
 // owns the surface energy balance and hands the ocean ICE_OCEAN_HEAT (its
 // open-water flux, minus the ocean-to-ice heat flux, plus melt).
 // reconcile_ocean applies that to the mixed layer, as the OML route replaces
 // its surface flux; the atmospheric fluxes here would count the ice-covered
 // surface twice and never take out the ice's heat. Ice-free water keeps them.
 bool ice_owned=coupled_ice&&ice_mask[i]>0.f;
 // Incoming flux times absorption is finite even when GSW+GLW=0.
 for(int k=0;k<L&&!ice_owned;k++){
  float z1=k?z[k-1]:0.f,dz=z[k]-z1;
  // Paulson and Simpson (1977), DOI 10.1175/1520-0485(1977)007<0952:IMITUO>2.0.CO;2.
  // Two solar bands; longwave is absorbed at the surface, not at 20 m.
  float heat=sw[i]*(0.62f*(expf(-z1/1.5f)-expf(-z[k]/1.5f))
                  +0.38f*(expf(-z1/20.f)-expf(-z[k]/20.f)));
  if(k==0)heat+=lw[i]-lh[i]-hfx[i]-stbolt*t[0]*t[0]*t[0]*t[0];
  t[k]+=dt*heat/(dz*1024.f*4183.3f);
 }
 s[0]+=s[0]*qfx[i]*dt/(1024.f*z[0]);
 for(int k=0;k<L;k++)r[k]=dr+alpha*(t[k]-tr)+beta*(s[k]-sr);
 int ml=1;
 for(int k=1;k<L;k++){if(r[k]>r[k-1])break;if(r[k]<r[k-1]){ml=k+1;mix(r,z,ml);}}
 if(ml>1){mix(t,z,ml);mix(s,z,ml);mix(u,z,ml);mix(v,z,ml);mix(r,z,ml);}
 float sa=sinf(f[i]*dt*0.5f),ca=cosf(f[i]*dt*0.5f);rotate(u,v,sa,ca);
 ml=mld(r);float dml=depth_mld(z,ml),speed=fmaxf(sqrtf(ua[i]*ua[i]+va[i]*va[i]),0.1f);
 // UST is atmospheric friction velocity: stress is rho_air * UST^2.
 float du=rhoair[i]*ust[i]*ust[i]*ua[i]/speed*dt/(1024.f*dml),dv=rhoair[i]*ust[i]*ust[i]*va[i]/speed*dt/(1024.f*dml);
 for(int k=0;k<ml;k++){u[k]+=du;v[k]+=dv;}
 for(int k=ml-1;k<L-1;k++){
  float dz=depth_mld(z,k+1),delr=(r[k+1]-r[k])/1024.f,a=u[k+1]-u[k],b=v[k+1]-v[k];
  if(9.8f*delr*dz/(a*a+b*b+1.e-8f)>0.65f)break;
  mix(t,z,k+2);mix(s,z,k+2);mix(u,z,k+2);mix(v,z,k+2);mix(r,z,k+2);
 }
 for(int k=0;k<L-1;k++)ri[k]=richardson(r,z,u,v,k);
 int iterations=0;
 while(true){
  int ks=0;for(int k=1;k<L-1;k++)if(ri[k]<ri[ks])ks=k;
  if(ri[ks]>=0.25f)break;
  // Report at history/checkpoint boundaries, without a per-step host sync.
  if(++iterations>100000){atomicOr(status,1);break;}
  float ff=1.f-ri[ks]/0.3f;stir(t,z,ks,ff);stir(s,z,ks,ff);stir(u,z,ks,ff);stir(v,z,ks,ff);stir(r,z,ks,ff);
  for(int k=max(0,ks-2);k<=min(L-2,ks+2);k++)ri[k]=richardson(r,z,u,v,k);
 }
 rotate(u,v,sa,ca);mlout[i]=depth_mld(z,mld(r));
 float *out[4]={tout,sout,uout,vout};const float *a[4]={t,s,u,v};
 for(int j=0;j<4;j++)for(int k=0;k<L;k++){
  int q=k*n+i,off=j*n*L+q;tend[off]+=(a[j][k]-base[off])/dt;
  float value=base[off]+stage_dt*tend[off];out[j][q]=value;if(!isfinite(value))atomicOr(status,2);
 }
 for(int k=0;k<L;k++){int q=k*n+i;density[q]=dr+alpha*(tout[q]-tr)+beta*(sout[q]-sr);}
 tsk[i]=tout[i];
}
extern "C" __global__ void pwp_fallback(float *t,float *s,float *u,float *v,float *z,const float *tsk,float hml,float gamma,int n){
 int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n*L)return;
 int i=q%n,k=q/n;float depth=5.f*(k+1);z[q]=depth;t[q]=tsk[i]-gamma*fmaxf(depth-hml,0.f);s[q]=35.f;u[q]=0.f;v[q]=0.f;
}
// One-degree spatial bins bound work to nearby wet profiles, including the
// dateline and polar caps. Final selection uses exact great-circle distance.
__device__ int ocean_bin(float lat,float lon){
 int y=min(179,max(0,(int)floorf(lat+90.f)));
 float x=lon-floorf(lon/360.f)*360.f;
 return y*360+min(359,max(0,(int)floorf(x)));
}
extern "C" __global__ void ocean_bin_count(const float *lat,const float *lon,const float *st,int *counts,int n){
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<n&&isfinite(lat[i])&&isfinite(lon[i])&&isfinite(st[i])&&st[i]>=260.f&&st[i]<=330.f)
  atomicAdd(counts+ocean_bin(lat[i],lon[i]),1);
}
extern "C" __global__ void ocean_bin_scatter(const float *lat,const float *lon,const float *st,
 const int *offset,int *cursor,int *members,int n){
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<n&&isfinite(lat[i])&&isfinite(lon[i])&&isfinite(st[i])&&st[i]>=260.f&&st[i]<=330.f){
  int bin=ocean_bin(lat[i],lon[i]);members[offset[bin]+atomicAdd(cursor+bin,1)]=i;
 }
}
extern "C" __global__ void ocean_nearest(const float *slat,const float *slon,
 const int *offset,const int *members,const float *tlat,const float *tlon,const float *land,int *index,int nt){
 int i=blockIdx.x;if(i>=nt)return;
 if(land[i]<=1.5f){if(threadIdx.x==0)index[i]=-2;return;}
 __shared__ float scores[256];__shared__ int indices[256];
 int tid=threadIdx.x,best=-1;float score=1.e30f,lat=tlat[i]*0.01745329252f,lon=tlon[i]*0.01745329252f;
 int bin=ocean_bin(tlat[i],tlon[i]),cy=bin/360,cx=bin%360;
 // Bound longitude by the smallest cosine inside the one-degree cap.
 float pole=fminf(90.f,fabsf(tlat[i])+1.f);
 int width=min(180,(int)ceilf(1.f/fmaxf(0.00001f,cosf(pole*0.01745329252f)))+1);
 for(int y=max(0,cy-1);y<=min(179,cy+1);y++)for(int d=-width;d<=min(width,359-width);d++){
  int x=(cx+d+360)%360,b=y*360+x;
  for(int p=offset[b]+tid;p<offset[b+1];p+=blockDim.x){
   int j=members[p];float a=slat[j]*0.01745329252f,bb=slon[j]*0.01745329252f;
   float dy=sinf((lat-a)*0.5f),dx=sinf((lon-bb)*0.5f);
   float dist=dy*dy+cosf(lat)*cosf(a)*dx*dx;
   if(dist<score||(dist==score&&(best<0||j<best))){score=dist;best=j;}
  }
 }
 scores[tid]=score;indices[tid]=best;__syncthreads();
 for(int off=128;off>0;off>>=1){
  if(tid<off&&(scores[tid+off]<scores[tid]||(scores[tid+off]==scores[tid]&&indices[tid+off]<indices[tid]))){scores[tid]=scores[tid+off];indices[tid]=indices[tid+off];}__syncthreads();
 }
 if(tid==0)index[i]=(scores[0]>0.00007615242f?-1:indices[0]);
}
extern "C" __global__ void ocean_profiles(const float *src,const float *depth,const int *index,
 float *out,float *zout,int ns,int nt,int nk,int celsius,float layer_m){
 int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=nt*L)return;
 int i=q%nt,k=q/nt,j=index[i];float bottom=layer_m*(k+1),top=layer_m*k;zout[q]=bottom;
 if(j<0){out[q]=(j==-2?0.f:nanf(""));return;}
 // Conservative mean of the analysis's piecewise-linear vertical profile.
 // Constant extension above its shallowest sample, no bottom extrapolation.
 float total=0.f,left=top;
 if(left<depth[0]){float right=fminf(bottom,depth[0]);total+=(right-left)*src[j];left=right;}
 for(int hi=1;hi<nk&&left<bottom;hi++){
  if(depth[hi]<=left)continue;
  float right=fminf(bottom,depth[hi]),zlo=depth[hi-1],dz=depth[hi]-zlo;
  float a=src[(hi-1)*ns+j],slope=(src[hi*ns+j]-a)/dz;
  total+=(right-left)*(a+0.5f*slope*(left+right-2.f*zlo));left=right;
 }
 out[q]=total/layer_m+(celsius?273.15f:0.f);
}

extern "C" __global__ void pwp_diagnose(const float *t,const float *s,const float *z,
 const float *tr,const float *sr,float *rout,float *ml,const float *land,int n){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||land[i]<=1.5f)return;
 float r[L],depth[L],dr=sigma(tr[i],sr[i])+1000.f;
 float alpha=sigma(tr[i]+0.5f,sr[i])-sigma(tr[i]-0.5f,sr[i]);
 float beta=sigma(tr[i],sr[i]+0.5f)-sigma(tr[i],sr[i]-0.5f);
 for(int k=0;k<L;k++){int q=k*n+i;r[k]=dr+alpha*(t[q]-tr[i])+beta*(s[q]-sr[i]);rout[q]=r[k];depth[k]=z[q];}
 ml[i]=depth_mld(depth,mld(r));
}
extern "C" __global__ void ocean_mld_profiles(const float *mld,const int *index,const float *sst,
 float *t,float *s,float *u,float *v,float *z,float gamma,int n){
 int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=n*L)return;
 int i=q%n,k=q/n,j=index[i];float depth=5.f*(k+1);z[q]=depth;
 // Below the mixed layer the configured lapse rate stops at sea-water freezing
 // (-1.9 C at 35 PSU), or at the SST where the surface is already colder: a
 // 150 m column under 275 K water otherwise fell below 260 K and was refused.
 t[q]=(j<0?0.f:fmaxf(sst[i]-gamma*fmaxf(depth-mld[j],0.f),fminf(sst[i],271.25f)));
 s[q]=(j<0?0.f:35.f);u[q]=0.f;v[q]=0.f;
}
