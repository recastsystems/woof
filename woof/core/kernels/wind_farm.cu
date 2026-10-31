// Original equation implementations; no source imported from research models.
// Fitch et al. (2012), doi:10.1175/MWR-D-11-00352.1, equations 8-12.
// Archer et al. (2020), doi:10.1175/MWR-D-20-0097.1: CTKE correction and
// retained/advection carrier (the Python driver supplies both QKE arrays).
// Volker et al. (2015), doi:10.5194/gmd-8-3715-2015, equations 12,14,15.
// Ma et al. (2022), doi:10.5194/wes-7-2407-2022, equations 1-20.
// Mirocha et al. (2014), doi:10.1063/1.4861061, appendix A10-A25.
// Float32, z/y/x C layout. Each output has one writer; no atomic reductions.
#define PI 3.14159265358979323846f
// Nearest generator on the unit sphere is spherical Voronoi ownership.
extern "C" __global__ void wind_locate_cells(int nt,int nc,const float* ll,
 const float* lat,const float* lon,int* indices){
 int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=nt)return;
 double la=(double)ll[2*a]*3.141592653589793/180.,lo=(double)ll[2*a+1]*3.141592653589793/180.;
 double sl=sin(la),cl=cos(la),best=-2.;int cell=0;
 for(int c=0;c<nc;++c){double dot=sl*sin((double)lat[c])+cl*cos((double)lat[c])*cos((double)lon[c]-lo);
  if(dot>best){best=dot;cell=c;}}
 indices[a]=cell;
}
__device__ float clampw(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
// Setup only: all turbines locate concurrently; no host per-turbine launches.
extern "C" __global__ void wind_locate(int nt,int nx,int ny,const float* ll,
 const float* lat,const float* lon,float* xy){
 int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=nt)return;
 float la=ll[2*a],lo=ll[2*a+1],scale=cosf(__fdiv_rn(la*PI,180.f)),best=1.e30f;int c=0;
 for(int q=0;q<nx*ny;++q){float dl=remainderf(lon[q]-lo,360.f)*scale,dlat=lat[q]-la;
  float d=dl*dl+dlat*dlat;if(d<best){best=d;c=q;}}
 int i=c%nx,j=c/nx,il=max(i-1,0),ir=min(i+1,nx-1),jl=max(j-1,0),jr=min(j+1,ny-1);
 float ax=remainderf(lon[j*nx+ir]-lon[j*nx+il],360.f)*scale/(ir-il);
 float ay=(lat[j*nx+ir]-lat[j*nx+il])/(ir-il);
 float bx=remainderf(lon[jr*nx+i]-lon[jl*nx+i],360.f)*scale/(jr-jl);
 float by=(lat[jr*nx+i]-lat[jl*nx+i])/(jr-jl),det=ax*by-ay*bx;
 float x=-remainderf(lon[c]-lo,360.f)*scale,y=la-lat[c];
 xy[2*a]=i+(x*by-y*bx)/det;xy[2*a+1]=j+(y*ax-x*ay)/det;
}
// Curves are rows [speed,CT,CP,rpm,pitch]. Outside supplied range: parked.
__device__ float curve(const float* c,int start,int n,float u,int field){
 if(u<c[5*start] || u>c[5*(start+n-1)]) return 0.f;
 int j=0; while(j<n-2 && c[5*(start+j+1)]<u) ++j;
 const float* a=c+5*(start+j); const float* b=a+5;
 float t=(u-a[0])/(b[0]-a[0]);return a[field]+t*(b[field]-a[field]);
}
// WRF's standing/cut-out thrust and zero power outside the operating interval.
// Generic inventory curves retain the lab's original interpolation unchanged.
__device__ float fitch_curve(const float* c,int start,int n,float u,int field,const float* limits){
 if(limits[0]>=0.f){
  if(u<=limits[0])return field==1?limits[2]:0.f;
  if(u>=limits[1])return field==1?limits[3]:0.f;
 }
 return curve(c,start,n,u,field);
}
__device__ float diskprimitive(float z,float r){
 z=clampw(z,-r,r);return z*sqrtf(fmaxf(0.f,r*r-z*z))+r*r*asinf(z/r);
}
__device__ float overlap(float r,float R,float d){
 if(d>=r+R)return 0.f;
 if(d<=fabsf(R-r)){float a=fminf(r,R);return PI*a*a;}
 float x=clampw((d*d+r*r-R*R)/(2*d*r),-1.f,1.f);
 float y=clampw((d*d+R*R-r*r)/(2*d*R),-1.f,1.f);
 return r*r*acosf(x)+R*R*acosf(y)-.5f*sqrtf(fmaxf(0.f,(-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)));
}
// Fitch et al. (2012), doi:10.1175/MWR-D-11-00352.1;
// Archer et al. (2020), doi:10.1175/MWR-D-20-0097.1.
// One block per occupied column. Hub interpolation, rotor normalization,
// drag and both TKE carriers share one launch. No atomics or full-grid pass.
// All threads participate in reductions, including threads above the top.
extern "C" __global__ void wind_fitch_sparse(int nc,int nz,const int* cols,
 const int* ptr,const int* ids,const float* t,const int* meta,const float* curves,const float* limits,
 const float* zi,const float* u,const float* v,const float* rho,const float* msf,
 float area,float dt,float factor,int hubnormalize,
 float* du,float* dv,float* qke,float* adv,float* hub,float* tke_rate){
 int c=cols[blockIdx.x],k=threadIdx.x,p=k*nc+c;
 __shared__ float sums[128];
 __shared__ float hs[7];
 float su=0.f,sv=0.f,st=0.f;
 float U=k<nz?hypotf(u[p],v[p]):0.f;
 for(int j=ptr[c];j<ptr[c+1];++j){
  int a=ids[j];float R=t[5*a+3],h=t[5*a+2]+zi[c];
  if(k==0){
   int z=0;while(z<nz-2 && .5f*(zi[(z+1)*nc+c]+zi[(z+2)*nc+c])<h)++z;
   int q=z*nc+c;float z0=.5f*(zi[q]+zi[q+nc]),z1=.5f*(zi[q+nc]+zi[q+2*nc]);
   float f=clampw((h-z0)/(z1-z0),0.f,1.f);
   float uh=u[q]+f*(u[q+nc]-u[q]),vh=v[q]+f*(v[q+nc]-v[q]);
   float sp=hypotf(u[q],v[q])+f*(hypotf(u[q+nc],v[q+nc])-hypotf(u[q],v[q]));
   float mag=hypotf(uh,vh);if(mag>0.f){uh*=sp/mag;vh*=sp/mag;}
   hs[0]=sp;hs[1]=fitch_curve(curves,meta[5*a],meta[5*a+1],sp,1,limits+5*a)*t[5*a+4];
   hs[2]=fitch_curve(curves,meta[5*a],meta[5*a+1],sp,2,limits+5*a);hs[3]=limits[5*a+4]>0.f?limits[5*a+4]:rho[q]+f*(rho[q+nc]-rho[q]);
   hs[4]=uh;hs[5]=vh;
  }
  float A=k<nz?diskprimitive(zi[p+nc]-h,R)-diskprimitive(zi[p]-h,R):0.f;
  sums[k]=A*U*U*U;__syncthreads();
  for(int stride=64;stride>0;stride>>=1){if(k<stride)sums[k]+=sums[k+stride];__syncthreads();}
  float ec=(hs[2]>0.f && sums[0]>0.f)?PI*R*R*hs[0]*hs[0]*hs[0]/sums[0]:1.f;
  if(k==0){float* o=hub+10*a;o[0]=hs[4];o[1]=hs[5];o[2]=hs[0];o[3]=hs[1];o[4]=hs[3];o[5]=0.f;o[6]=h;
   o[7]=.5f*hs[3]*PI*R*R*hs[2]*hs[0]*hs[0]*hs[0]*t[5*a+4];o[8]=hs[2];o[9]=ec;}
  if(k<nz && A>0.f && t[5*a+4]!=0.f){
   float ct=hubnormalize?hs[1]:fitch_curve(curves,meta[5*a],meta[5*a+1],U,1,limits+5*a);
   float cp=hubnormalize?hs[2]:fitch_curve(curves,meta[5*a],meta[5*a+1],U,2,limits+5*a);
   if(!hubnormalize)ec=1.f;
   float coeff=.5f*A*msf[c]*msf[c]/area/(zi[p+nc]-zi[p]);
   float drag=ct*U*ec;su-=coeff*drag*u[p];sv-=coeff*drag*v[p];
   st+=coeff*factor*fmaxf(0.f,ct-cp)*U*U*U*ec;
  }
  __syncthreads();
 }
 if(k<nz){du[p]+=su;dv[p]+=sv;qke[p]+=2.f*dt*st;if(adv!=qke)adv[p]+=2.f*dt*st;tke_rate[p]=st;}
}
// Geometry rows: x(m),y(m),hub(AGL),radius,operating. type metadata:
// curve offset,count,blade offset,count,number of blades.
extern "C" __global__ void wind_sample(int nt,int nc,int nz,const int* col,
 const float* t,const int* meta,const float* curves,const float* limits,const float* zi,
 const float* u,const float* v,const float* rho,const float* km,float* hub){
 int a=blockDim.x*blockIdx.x+threadIdx.x;if(a>=nt)return;
 int c=col[a];float h=t[5*a+2]+zi[c];int k=0;
 while(k<nz-2 && .5f*(zi[(k+1)*nc+c]+zi[(k+2)*nc+c])<h)++k;
 float z0=.5f*(zi[k*nc+c]+zi[(k+1)*nc+c]);
 float z1=.5f*(zi[(k+1)*nc+c]+zi[(k+2)*nc+c]);
 float f=clampw((h-z0)/(z1-z0),0.f,1.f);int p=k*nc+c;
 float uh=u[p]+f*(u[p+nc]-u[p]),vh=v[p]+f*(v[p+nc]-v[p]);
 float mag=hypotf(uh,vh);
 float sp=hypotf(u[p],v[p])+f*(hypotf(u[p+nc],v[p+nc])-hypotf(u[p],v[p]));
 if(mag>0.f){uh*=sp/mag;vh*=sp/mag;}
 float* o=hub+10*a;
 o[0]=uh;o[1]=vh;o[2]=sp;o[3]=fitch_curve(curves,meta[5*a],meta[5*a+1],sp,1,limits+5*a)*t[5*a+4];
 o[4]=limits[5*a+4]>0.f?limits[5*a+4]:rho[p]+f*(rho[p+nc]-rho[p]);o[5]=fmaxf(0.f,km[p]+f*(km[p+nc]-km[p]));
 o[6]=zi[c]+t[5*a+2];o[8]=fitch_curve(curves,meta[5*a],meta[5*a+1],sp,2,limits+5*a);
 float integral=0.f,R=t[5*a+3];
 for(int kk=0;kk<nz;++kk){float A=diskprimitive(zi[(kk+1)*nc+c]-h,R)-diskprimitive(zi[kk*nc+c]-h,R);float U=hypotf(u[kk*nc+c],v[kk*nc+c]);integral+=A*U*U*U;}
 o[9]=(o[8]>0.f && integral>0.f)?PI*R*R*sp*sp*sp/integral:1.f;
 o[7]=0.5f*o[4]*PI*t[5*a+3]*t[5*a+3]*o[8]*sp*sp*sp*t[5*a+4];
}
// Seven direction samples, Gaussian probability at the paper's offsets.
extern "C" __global__ void wind_wake(int nt,const float* t,const float* h,
 float kw,float maxd,float sd,int superposition,float* speed){
 int index=blockDim.x*blockIdx.x+threadIdx.x;if(index>=nt*7)return;
 int a=index/7,s=index%7;const float offsets[7]={-2.5f,-1.5f,-.5f,0.f,.5f,1.5f,2.5f};
 float base=h[10*a+2],sum=0.f;int count=0;
 float deltaangle=__fdiv_rn(offsets[s]*PI,180.f);
 for(int b=0;b<nt;++b){
  if(b==a || t[5*b+4]==0.f)continue;
  float theta=atan2f(h[10*a+1]+h[10*b+1],h[10*a]+h[10*b])+deltaangle;
  float cs=cosf(theta),sn=sinf(theta),dx=t[5*a]-t[5*b],dy=t[5*a+1]-t[5*b+1];
  float x=dx*cs+dy*sn,y=-dx*sn+dy*cs,R=t[5*b+3];
  if(x<=0.f || x>2.f*R*maxd || fabsf(y)>x*.577350269f)continue;
  float r=t[5*a+3],d=hypotf(y,h[10*a+6]-h[10*b+6]);
  float fraction=overlap(r,R+kw*x,d)/(PI*r*r);if(fraction<=0.f)continue;
  float def=(1.f-sqrtf(fmaxf(0.f,1.f-h[10*b+3])))/((1.f+kw*x/R)*(1.f+kw*x/R));
  float loss=h[10*b+2]*def*fraction;
  if(superposition==1)sum+=loss;
  else if(superposition==2)sum+=loss*loss;
  else {float mix=base*(1.f-fraction)+h[10*b+2]*(1.f-def)*fraction;sum+=mix*mix;}
  ++count;
 }
 speed[index]=fmaxf(0.f,count==0?base:superposition==1?base-sum:superposition==2?base-sqrtf(sum):sqrtf(sum/count));
}
extern "C" __global__ void wind_source(int scheme,int nc,int nz,int nx,int ny,
 const int* ptr,const int* ids,const float* t,const int* meta,const float* curves,const float* limits,
 const float* hub,const float* wake,const float* zi,const float* u,const float* v,
 const float* msf,float area,float dx,float dt,float factor,float sigma0factor,float sd,int hubnormalize,
 float* du,float* dv,float* qke,float* adv,float* tke_rate){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=nc*nz)return;
 int c=p%nc;float su=0.f,sv=0.f,st=0.f;
 float z0=zi[p],z1=zi[p+nc],dz=z1-z0,U=hypotf(u[p],v[p]);
 const float offsets[7]={-2.5f,-1.5f,-.5f,0.f,.5f,1.5f,2.5f};
 float weights[7],wt=0.f;
 for(int s=0;s<7;++s){weights[s]=sd>0.f?expf(-.5f*offsets[s]*offsets[s]/(sd*sd)):(s==3?1.f:0.f);wt+=weights[s];}
 for(int j=ptr[c];j<ptr[c+1];++j){
  int a=ids[j];if(t[5*a+4]==0.f)continue;
  float R=t[5*a+3],h=hub[10*a+6],Uh=hub[10*a+2];
  float A=diskprimitive(z1-h,R)-diskprimitive(z0-h,R);
  float inv=msf[c]*msf[c]/area;
  int off=meta[5*a],n=meta[5*a+1];
  if(scheme==2){
   float s0=sigma0factor*R,b=hub[10*a+5]*dx/fmaxf(Uh,1.e-6f);
   // Stable integral of sqrt(s0^2 + 2 K x / U), L=dx/2.
   float s1=sqrtf(s0*s0+b),sig=(2.f/3.f)*(s1*s1+s1*s0+s0*s0)/(s1+s0);
   float z=.5f*(z0+z1)-h;
   float F=sqrtf(PI/8.f)*hub[10*a+3]*R*R*Uh*Uh*inv/sig*expf(-.5f*z*z/(sig*sig));
   if(U>0.f){su-=F*u[p]/U;sv-=F*v[p]/U;}continue;
  }
  if(A<=0.f)continue;
  float drag=0.f,tk=0.f;
  if(scheme==3){
   for(int s=0;s<7;++s){float ws=wake[7*a+s],r=Uh>0.f?ws/Uh:0.f;
    float ct=curve(curves,off,n,ws,1),cp=curve(curves,off,n,ws,2),uk=r*U;
    drag+=weights[s]*ct*uk;tk+=weights[s]*factor*(ct-cp)*uk*uk*uk;
   }drag/=wt;tk/=wt;
  }else{float ct=hubnormalize?hub[10*a+3]:fitch_curve(curves,off,n,U,1,limits+5*a),cp=hubnormalize?hub[10*a+8]:fitch_curve(curves,off,n,U,2,limits+5*a);
   float ec=hubnormalize?hub[10*a+9]:1.f;drag=ct*U*ec;tk=factor*fmaxf(0.f,ct-cp)*U*U*U*ec;}
  float coeff=.5f*A*inv/dz;su-=coeff*drag*u[p];sv-=coeff*drag*v[p];st+=coeff*tk;
 }
 du[p]+=su;dv[p]+=sv;tke_rate[p]=st;
 // qke is twice TKE. Both carriers persist; never overwrite advected TKE
 // with the previous step's pre-farm local MYNN value.
 qke[p]+=2.f*dt*st;if(adv!=qke)adv[p]+=2.f*dt*st;
}
// Flat polar rows [alpha degrees, CL, CD], linear interpolation.
__device__ float polar(const float* rows,const int* pm,int id,float alpha,int field){
 int o=pm[2*id],n=pm[2*id+1],j=0;
 if(alpha<=rows[3*o])return rows[3*o+field];
 if(alpha>=rows[3*(o+n-1)])return rows[3*(o+n-1)+field];
 while(j<n-2 && rows[3*(o+j+1)]<alpha)++j;
 const float* p=rows+3*(o+j);float f=(alpha-p[0])/(p[3]-p[0]);return p[field]+f*(p[3+field]-p[field]);
}
extern "C" __global__ void wind_gad(int nc,int nz,int nx,float dx,float dy,float sigma,
 const int* ptr,const int* ids,const float* t,const int* meta,const float* curves,
 const float* blades,const float* polars,const int* pm,const float* hub,
 const float* zi,const float* u,const float* v,float tol,int maxiter,
 float* du,float* dv,float* dw,int* failures){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=nc*nz)return;
 int c=p%nc;float x=(c%nx)*dx,y=(c/nx)*dy,z=.5f*(zi[p]+zi[p+nc]);
 float fx=0.f,fy=0.f,fz=0.f;int failed=0;
 for(int j=ptr[c];j<ptr[c+1];++j){
  int a=ids[j];if(t[5*a+4]==0.f)continue;
  float sp=hub[10*a+2];if(sp<=1.e-6f)continue;
  float cs=hub[10*a]/sp,sn=hub[10*a+1]/sp;
  float dn=(x-t[5*a])*cs+(y-t[5*a+1])*sn;
  float cross=-(x-t[5*a])*sn+(y-t[5*a+1])*cs;
  float vertical=z-hub[10*a+6],r=hypotf(cross,vertical),R=t[5*a+3];
  int bo=meta[5*a+2],bn=meta[5*a+3];float rhub=blades[4*bo]*R;
  if(r<=rhub || r>=R || fabsf(dn)>4.f*sigma)continue;
  float speed=fmaxf(0.f,u[p]*cs+v[p]*sn),rpm=curve(curves,meta[5*a],meta[5*a+1],sp,3);
  if(rpm<=0.f || speed<=0.f)continue;
  float pitch=curve(curves,meta[5*a],meta[5*a+1],sp,4),omega=rpm*(2.f*PI/60.f);
  int b=0;while(b<bn-2 && blades[4*(bo+b+1)]*R<r)++b;
  const float* bb=blades+4*(bo+b);float f=clampw((r/R-bb[0])/(bb[4]-bb[0]),0.f,1.f);
  float chord=bb[1]+f*(bb[5]-bb[1]),twist=bb[2]+f*(bb[6]-bb[2]);
  float B=(float)meta[5*a+4],sol=B*chord/(2.f*PI*r),at=0.f,cn=0.f,ct=0.f,vr2=0.f;bool converged=false;
  for(int it=0;it<maxiter;++it){
   float vn=speed*.98f,vt=omega*r*(1.f+at),psi=atan2f(vn,vt),s=sinf(psi),co=cosf(psi);
   float alpha=__fdiv_rn(psi*180.f,PI)-twist+pitch;
   float cl0=polar(polars,pm,(int)bb[3],alpha,1),cd0=polar(polars,pm,(int)bb[3],alpha,2);
   float cl=cl0+f*(polar(polars,pm,(int)bb[7],alpha,1)-cl0);
   float cd=cd0+f*(polar(polars,pm,(int)bb[7],alpha,2)-cd0);
   cn=cl*co+cd*s;ct=cl*s-cd*co;vr2=vn*vn+vt*vt;
   float F=2.f/PI*(acosf(expf(-B*(R-r)/(2.f*r*s)))+acosf(expf(-B*(r-rhub)/(2.f*r*s))));
   float den=4.f*F*s*co-sol*ct;
   float next=fabsf(den)>1.e-8f?sol*ct/den:at;
   if(!isfinite(next) || fabsf(next)>2.f){break;}
   if(fabsf(next-at)<tol){converged=true;break;}
   at=.5f*(next+at); // under-relaxation preserves the BEM fixed point
  }
  if(!converged)++failed;
  float gaussian=expf(-.5f*dn*dn/(sigma*sigma))/(sqrtf(2.f*PI)*sigma);
  float coeff=.5f*vr2*B*chord/(2.f*PI*r)*gaussian;
  float fn=coeff*cn,ft=coeff*ct;
  fx-=fn*cs+ft*(vertical/r)*sn;fy-=fn*sn-ft*(vertical/r)*cs;fz+=ft*(cross/r);
 }
 du[p]=fx;dv[p]=fy;dw[p]=fz;failures[p]+=failed;
}

extern "C" __global__ void wind_power(int nt,const float* t,const int* meta,
 const float* curves,const float* wake,float sd,float* hub){
 int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=nt)return;
 const float angles[7]={-2.5f,-1.5f,-.5f,0.f,.5f,1.5f,2.5f};
 float power=0.f,total=0.f;
 for(int s=0;s<7;++s){float w=sd>0.f?expf(-.5f*angles[s]*angles[s]/(sd*sd)):(s==3?1.f:0.f);
 float u=wake[7*a+s];power+=w*curve(curves,meta[5*a],meta[5*a+1],u,2)*u*u*u;total+=w;}
 hub[10*a+7]=.5f*hub[10*a+4]*PI*t[5*a+3]*t[5*a+3]*power/total*t[5*a+4];
}
