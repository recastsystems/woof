// SPDX-License-Identifier: Apache-2.0
// Independently implemented equations 3.1-3.9 of Kurihara et al. (1993),
// DOI 10.1175/1520-0493(1993)121<2030:AISOHM>2.0.CO;2.
// Radial mapping: Liu et al. (2020), DOI 10.3390/atmos11090968.
// Regular isobaric patch, Earth-relative winds; never a native eta-level shift.
#if !defined(__CUDACC_RTC__)
#include <math.h>
#else
#define NAN __longlong_as_double(0x7ff8000000000000LL)
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#endif
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
using int64_t = long long;
#endif
constexpr double ER=6371000., DR=0.017453292519943295, PI=3.14159265358979323846;
__device__ double profile_sample(const double*r,const double*v,int n,double x){
 if(x<=r[0])return v[0];if(x>=r[n-1])return v[n-1];
 int lo=0,hi=n-1;while(hi-lo>1){int mid=(lo+hi)/2;if(r[mid]>x)hi=mid;else lo=mid;}
 double a=(x-r[lo])/(r[hi]-r[lo]);return (1.-a)*v[lo]+a*v[hi];
}
struct Grid {int x,y;double lat0,lon0,dlat,dlon;};
__device__ double sample(const double*f,Grid g,double lat,double lon){
 double dl=remainder(lon-g.lon0,360.);
 double x=dl/g.dlon,y=(lat-g.lat0)/g.dlat;
 if(x < -1.e-9 || y < -1.e-9 || x>g.x-1+1.e-9 || y>g.y-1+1.e-9)return NAN;
 x=fmin(fmax(x,0.),g.x-1.);y=fmin(fmax(y,0.),g.y-1.);
 int i=min(int(x),g.x-2),j=min(int(y),g.y-2);double a=x-i,b=y-j;
 return (1-b)*((1-a)*f[j*g.x+i]+a*f[j*g.x+i+1])+b*((1-a)*f[(j+1)*g.x+i]+a*f[(j+1)*g.x+i+1]);
}
// theta is great-circle initial bearing clockwise from geographic north.
__device__ void polar(double clat,double clon,double lat,double lon,double&r,double&theta){
 double p=clat*DR,q=lat*DR,d=(lon-clon)*DR;
 double e=cos(q)*sin(d),n=cos(p)*sin(q)-sin(p)*cos(q)*cos(d);
 r=ER*atan2(hypot(e,n),sin(p)*sin(q)+cos(p)*cos(q)*cos(d));theta=atan2(e,n);
}
__device__ void destination(double clat,double clon,double r,double theta,double&lat,double&lon){
 double p=clat*DR,d=r/ER;
 double q=asin(fmin(1.,fmax(-1.,sin(p)*cos(d)+cos(p)*sin(d)*cos(theta))));
 lat=q/DR;lon=clon+atan2(sin(theta)*sin(d)*cos(p),cos(d)-sin(p)*sin(q))/DR;
}
// Evaluate paired source profiles at inverse-mapped destination radii.
// Vector input groups are tangential (CCW), radial (outward); output E,N.
extern "C" __global__ void tc_profile_field(const double*r,const double*v,double*out,
 int R,int P,int X,int Y,double lat0,double lon0,double dlat,double dlon,
 double clat,double clon,double a,double b,double outer,double outside,int vector){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,N=int64_t(X)*Y;
 int levels=vector?P/2:P;if(i>=N*levels)return;
 int p=i/N;double lat=lat0+(i/X%Y)*dlat,lon=lon0+(i%X)*dlon,d,t;
 polar(clat,clon,lat,lon,d,t);
 if(d>=outer){out[i]=outside;if(vector)out[i+levels*N]=outside;return;}
 double s=b==0.?d/a:2.*d/(a+sqrt(a*a+2.*b*d));
 double value=profile_sample(r,v+p*R,R,s);
 if(!vector){out[i]=value;return;}
 double radial=profile_sample(r,v+(p+levels)*R,R,s),dummy,back;
 polar(lat,lon,clat,clon,dummy,back);double bearing=d<1.e-8?t:back+PI;
 out[i]=-value*cos(bearing)+radial*sin(bearing);
 out[i+levels*N]=value*sin(bearing)+radial*cos(bearing);
}
// Intersection of |environment+asymmetry + gain*axisymmetric| <= target.
// The largest feasible positive gain places at least one resolved point on
// the requested maximum, while retaining vector background and asymmetries.
extern "C" __global__ void tc_gain_bounds(const double*wind,const double*axis,
 const unsigned char*mask,double*lower,double*upper,int C,double target){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=C)return;
 if(!mask[i]){lower[i]=0.;upper[i]=INFINITY;return;}
 double ax=axis[i],ay=axis[C+i],bx=wind[i]-ax,by=wind[C+i]-ay;
 double A=ax*ax+ay*ay,B=ax*bx+ay*by,D=bx*bx+by*by-target*target;
 if(!isfinite(A)||!isfinite(B)||!isfinite(D)){lower[i]=INFINITY;upper[i]=-INFINITY;return;}
 if(A==0.){lower[i]=D<=0.?0.:INFINITY;upper[i]=D<=0.?INFINITY:-INFINITY;return;}
 double discriminant=B*B-A*D;
 if(discriminant<0.){lower[i]=INFINITY;upper[i]=-INFINITY;return;}
 double root=sqrt(discriminant),q=-B-copysign(root,B);
 double r0=q/A,r1=q==0.?0.:D/q;
 lower[i]=fmax(0.,fmin(r0,r1));upper[i]=fmax(r0,r1);
}
// Largest connected interval from g=0 with P(g) >= observed pressure floor.
// P(g)-floor=A*g^2+B*g+C. For a cyclonic pressure anomaly A,B <= 0;
// positive anomalies with nonnegative coefficients cannot impose an upper
// bound. Mixed signs are solved too, without assuming a monotone residual.
extern "C" __global__ void tc_pressure_gain_bounds(const double*moved,const double*axis,
 const double*quadratic,const double*linear,const unsigned char*mask,double*upper,
 int N,double floor){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 if(!mask[i]){upper[i]=INFINITY;return;}
 double A=axis[i]*quadratic[i],B=axis[i]*linear[i],C=moved[i]-axis[i]-floor;
 if(!isfinite(A)||!isfinite(B)||!isfinite(C)||C<0.){upper[i]=NAN;return;}
 if(A==0.){upper[i]=B<0.?-C/B:INFINITY;return;}
 if(C==0.){upper[i]=B<0.?0.:(A<0.?-B/A:INFINITY);return;}
 double D=B*B-4.*A*C;
 if(D<=0.){upper[i]=A<0.?NAN:INFINITY;return;}
 double q=-.5*(B+copysign(sqrt(D),B));
 double r0=q/A,r1=q==0.?INFINITY:C/q;
 double bound=INFINITY;
 if(r0>0.)bound=r0;if(r1>0.)bound=fmin(bound,r1);
 upper[i]=bound;
}
// Hydrostatic INCREMENT on fixed pressure levels, anchored at model top.
// Retains each background column's discretization/nonhydrostatic residual.
// p is ascending (top to bottom), q is mixing ratio, z is geometric height.
extern "C" __global__ void tc_hydrostatic_increment(const double*p,const double*T0,
 const double*q0,const double*T1,const double*q1,const double*z0,double*z1,int C,int K){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=C)return;
 double delta=0.,prev=0.;bool bad=false;
 for(int k=0;k<K;++k){
  int64_t i=int64_t(k)*C+c;
  bad|=!isfinite(T0[i])||!isfinite(T1[i])||T0[i]<=0.||T1[i]<=0.
   ||!isfinite(q0[i])||!isfinite(q1[i])||q0[i]<0.||q1[i]<0.||!isfinite(z0[i])
   ||!isfinite(p[k])||p[k]<=0.||(k>0&&p[k]<=p[k-1]);
  double dT=T1[i]*(1.+q1[i]/.622)/(1.+q1[i])-T0[i]*(1.+q0[i]/.622)/(1.+q0[i]);
  if(k)delta-=287.05/9.80665*.5*(prev+dT)*log(p[k]/p[k-1]);
  z1[i]=bad?NAN:z0[i]+delta;prev=dT;
 }
}
// Reconcile the height increment with the changed sea-level pressure anchor.
// Uses the lowest pressure-level virtual temperature for the extrapolation;
// preserves the old column's extrapolation residual. Surface pressure uses
// the separately supplied near-surface virtual temperatures and fixed terrain.
extern "C" __global__ void tc_pressure_anchor(const double*p,const double*T0,
 const double*q0,const double*T1,const double*q1,const double*z0,const double*z_top,
 const double*mslp0,const double*mslp1,const double*ps0,const double*terrain,
 const double*tv_surface0,const double*tv_surface1,double*z1,double*ps1,int C,int K){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=C)return;
 int64_t bottom=int64_t(K-1)*C+c;
 double tv0=T0[bottom]*(1.+q0[bottom]/.622)/(1.+q0[bottom]);
 double tv1=T1[bottom]*(1.+q1[bottom]/.622)/(1.+q1[bottom]);
 bool bad=!isfinite(mslp0[c])||!isfinite(mslp1[c])||mslp0[c]<=0.||mslp1[c]<=0.
  ||!isfinite(ps0[c])||ps0[c]<=0.||!isfinite(terrain[c])
  ||!isfinite(tv_surface0[c])||!isfinite(tv_surface1[c])||tv_surface0[c]<=0.||tv_surface1[c]<=0.;
 double bottom_delta=287.05/9.80665*(tv1*log(mslp1[c]/p[K-1])-tv0*log(mslp0[c]/p[K-1]));
 double offset=bottom_delta-(z_top[bottom]-z0[bottom]);
 for(int k=0;k<K;++k){int64_t i=int64_t(k)*C+c;z1[i]=bad?NAN:z_top[i]+offset;}
 double exponent=log(mslp1[c]/mslp0[c])+9.80665*terrain[c]/287.05*(1./tv_surface0[c]-1./tv_surface1[c]);
 ps1[c]=bad?NAN:ps0[c]*exp(exponent);
}
extern "C" __global__ void tc_filter(const double*in,double*out,int X,int Y,int P,int axis,double k){
 int64_t t=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=int64_t(X)*Y*P)return;
 int x=t%X,y=t/X%Y;
 if((axis==0&&(x==0||x==X-1))||(axis==1&&(y==0||y==Y-1))){out[t]=in[t];return;}
 int stride=axis==0?1:X;out[t]=in[t]+k*(in[t-stride]+in[t+stride]-2.*in[t]);
}
extern "C" __global__ void tc_ring(const double*field,const double*basic,double*ring,double*mean,
 int X,int Y,double lat0,double lon0,double dlat,double dlon,
 int BX,int BY,double blat0,double blon0,int P,int angles,double clat,double clon,double radius){
 int p=blockIdx.x,a=threadIdx.x;extern __shared__ double values[];
 double value=0.;if(a<angles){
  double lat,lon;destination(clat,clon,radius,2.*PI*a/angles,lat,lon);
  value=sample(field+int64_t(p)*X*Y,Grid{X,Y,lat0,lon0,dlat,dlon},lat,lon)
       -sample(basic+int64_t(p)*BX*BY,Grid{BX,BY,blat0,blon0,1.,1.},lat,lon);
  ring[p*angles+a]=value;
 }
 values[a]=value;__syncthreads();
 // Fixed ordered reduction of the small boundary quadrature. No host stage.
 if(a==0){double sum=0.;for(int j=0;j<angles;++j)sum+=values[j];mean[p]=sum/angles;}
}
__device__ double anomaly(const double*field,const double*basic,const double*ring,double mean,
 Grid g,Grid bg,int angles,double clat,double clon,double radius,double lat,double lon){
 double r,t;polar(clat,clon,lat,lon,r,t);if(r>=radius)return 0.;
 double a=t*angles/(2.*PI);if(a<0)a+=angles;
 int j=int(a);double b=a-j,bound=(1.-b)*ring[j%angles]+b*ring[(j+1)%angles];
 // l = r0/5, as the original paper specifies. Stable expm1 denominator.
 double l=radius*.2,exp0=exp(-radius*radius/(l*l));
 double E=(exp(-(radius-r)*(radius-r)/(l*l))-exp0)/(1.-exp0);
 return sample(field,g,lat,lon)-sample(basic,bg,lat,lon)-bound*E-mean*(1.-E);
}
extern "C" __global__ void tc_move(const double*field,const double*basic,const double*ring,const double*mean,double*out,
 int X,int Y,double lat0,double lon0,double dlat,double dlon,
 int BX,int BY,double blat0,double blon0,int P,int angles,
 double oldlat,double oldlon,double newlat,double newlon,double radius,
 double map_a,double map_b,double target_radius){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,N=int64_t(X)*Y;
 if(i>=N*P)return;int p=i/N,x=i%X,y=i/X%Y;
 Grid g{X,Y,lat0,lon0,dlat,dlon},bg{BX,BY,blat0,blon0,1.,1.};
 double lat=lat0+y*dlat,lon=lon0+x*dlon,ro,to,rn,tn;
 polar(oldlat,oldlon,lat,lon,ro,to);polar(newlat,newlon,lat,lon,rn,tn);
 // Exact untouched environment: do not even evaluate subtraction there.
 if(ro>=radius&&rn>=target_radius){out[i]=field[i];return;}
 const double*f=field+p*N,*b=basic+int64_t(p)*BX*BY,*r=ring+p*angles;
 double removed=anomaly(f,b,r,mean[p],g,bg,angles,oldlat,oldlon,radius,lat,lon),added=0.;
 if(rn<target_radius){
  // Forward r_new=a*r_old+0.5*b*r_old^2. Stable positive inverse.
  double source_r=map_b==0.?rn/map_a:2.*rn/(map_a+sqrt(map_a*map_a+2.*map_b*rn));
  double slat,slon;destination(oldlat,oldlon,source_r,tn,slat,slon);
  added=anomaly(f,b,r,mean[p],g,bg,angles,oldlat,oldlon,radius,slat,slon);
 }
 out[i]=field[i]-removed+added;
}
// Nonnegative constituents cannot use signed anomaly subtraction. Fill the
// vacated disk from its boundary and blend a full donor column into the new
// disk. All moments use identical nonnegative interpolation weights. This
// is a bounded initialization remap, not a conservative transport operator.
extern "C" __global__ void tc_move_constituents(const double*field,double*out,
 int X,int Y,double lat0,double lon0,double dlat,double dlon,int P,int angles,
 double oldlat,double oldlon,double newlat,double newlon,double radius,
 double map_a,double map_b,double target_radius){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,N=int64_t(X)*Y;
 if(i>=N*P)return;int p=i/N,x=i%X,y=i/X%Y;
 Grid g{X,Y,lat0,lon0,dlat,dlon};const double*f=field+p*N;
 double lat=lat0+y*dlat,lon=lon0+x*dlon,ro,to,rn,tn;
 polar(oldlat,oldlon,lat,lon,ro,to);polar(newlat,newlon,lat,lon,rn,tn);
 if(ro>=radius&&rn>=target_radius){out[i]=field[i];return;}
 double background=field[i];
 if(ro<radius){
  double mean=0.;
  for(int a=0;a<angles;++a){double la,lo;
   destination(oldlat,oldlon,radius,2.*PI*a/angles,la,lo);
   mean+=sample(f,g,la,lo)/angles;
  }
  double la,lo;destination(oldlat,oldlon,radius,to,la,lo);
  double s=ro/radius;
  background=(1.-s)*mean+s*sample(f,g,la,lo);
 }
 if(rn>=target_radius){out[i]=background;return;}
 double sr=map_b==0.?rn/map_a:2.*rn/(map_a+sqrt(map_a*map_a+2.*map_b*rn));
 double la,lo;destination(oldlat,oldlon,sr,tn,la,lo);
 // The same r0/5 boundary layer used by the vortex separation; exactly
 // donor at the centre and exactly environment at the outer radius.
 double s=sr/radius,e0=exp(-25.);
 double environment_weight=(exp(-25.*(1.-s)*(1.-s))-e0)/(1.-e0);
 out[i]=(1.-environment_weight)*sample(f,g,la,lo)+environment_weight*background;
}
// Translate a vector anomaly in radial/tangential bases, including meridian
// convergence along each great circle. Planes are [east levels,north levels].
extern "C" __global__ void tc_move_vector(const double*field,const double*basic,const double*ring,const double*mean,double*out,
 int X,int Y,double lat0,double lon0,double dlat,double dlon,
 int BX,int BY,double blat0,double blon0,int P,int angles,
 double oldlat,double oldlon,double newlat,double newlon,double radius,
 double map_a,double map_b,double target_radius){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,N=int64_t(X)*Y;
 int levels=P/2;if(i>=N*levels)return;int p=i/N,x=i%X,y=i/X%Y;
 Grid g{X,Y,lat0,lon0,dlat,dlon},bg{BX,BY,blat0,blon0,1.,1.};
 double lat=lat0+y*dlat,lon=lon0+x*dlon,ro,to,rn,tn;
 polar(oldlat,oldlon,lat,lon,ro,to);polar(newlat,newlon,lat,lon,rn,tn);
 int64_t north=i+levels*N;
 if(ro>=radius&&rn>=target_radius){out[i]=field[i];out[north]=field[north];return;}
 double removed[2],added[2]={0.,0.};
 for(int c=0;c<2;++c){int q=p+c*levels;
  removed[c]=anomaly(field+q*N,basic+int64_t(q)*BX*BY,ring+q*angles,mean[q],g,bg,angles,oldlat,oldlon,radius,lat,lon);
 }
 if(rn<target_radius){
  double sr=map_b==0.?rn/map_a:2.*rn/(map_a+sqrt(map_a*map_a+2.*map_b*rn));
  double slat,slon;destination(oldlat,oldlon,sr,tn,slat,slon);
  double source[2];for(int c=0;c<2;++c){int q=p+c*levels;
   source[c]=anomaly(field+q*N,basic+int64_t(q)*BX*BY,ring+q*angles,mean[q],g,bg,angles,oldlat,oldlon,radius,slat,slon);
  }
  double dummy,source_back,target_back;
  polar(slat,slon,oldlat,oldlon,dummy,source_back);
  polar(lat,lon,newlat,newlon,dummy,target_back);
  double rotation=rn<1.e-8?0.:target_back-source_back;
  // A clockwise bearing change rotates (east,north) by this convention.
  added[0]=cos(rotation)*source[0]+sin(rotation)*source[1];
  added[1]=-sin(rotation)*source[0]+cos(rotation)*source[1];
 }
 out[i]=field[i]-removed[0]+added[0];out[north]=field[north]-removed[1]+added[1];
}
// Gradient-wind pressure ratio, equations 10-16 and 45-48 of
// DOI 10.3390/atmos11090968. The two profiles correspond point-for-point
// under the radial map, but integration uses each profile's physical radii.
extern "C" __global__ void tc_balance_ratio(const double*r0,const double*v0,
 const double*r1,const double*v1,double f0,double f1,int N,double*gamma){
 extern __shared__ double work[];
 double *a0=work,*a1=work+N,*b0=work+2*N,*b1=work+3*N;
 __shared__ int bad;if(threadIdx.x==0)bad=0;__syncthreads();
 for(int j=threadIdx.x;j<N;j+=blockDim.x){
  bool invalid=!isfinite(r0[j])||!isfinite(r1[j])||!isfinite(v0[j])||!isfinite(v1[j])
   ||r0[j]<0||r1[j]<0||f0*v0[j]<0||f1*v1[j]<0;
  if(j==0)invalid|=r0[j]!=0.||r1[j]!=0.||v0[j]!=0.||v1[j]!=0.;
  else invalid|=r0[j]<=r0[j-1]||r1[j]<=r1[j-1];
  if(j==N-1)invalid|=v0[j]!=0.||v1[j]!=0.;
  if(invalid)atomicExch(&bad,1);
  int i=N-1-j;double old=0.,now=0.;
  if(i<N-1){
   double g0=i==0?0.:v0[i]*v0[i]/r0[i]+f0*v0[i];
   double g1=i==0?0.:v1[i]*v1[i]/r1[i]+f1*v1[i];
   old=.5*(g0+v0[i+1]*v0[i+1]/r0[i+1]+f0*v0[i+1])*(r0[i+1]-r0[i]);
   now=.5*(g1+v1[i+1]*v1[i+1]/r1[i+1]+f1*v1[i+1])*(r1[i+1]-r1[i]);
  }
  a0[j]=old;a1[j]=now;
 }
 __syncthreads();
 // Parallel inclusive prefix from the outer boundary toward the eye.
 for(int stride=1;stride<N;stride*=2){
  for(int j=threadIdx.x;j<N;j+=blockDim.x){
   b0[j]=a0[j]+(j>=stride?a0[j-stride]:0.);
   b1[j]=a1[j]+(j>=stride?a1[j-stride]:0.);
  }
  __syncthreads();double*t=a0;a0=b0;b0=t;t=a1;a1=b1;b1=t;__syncthreads();
 }
 for(int j=threadIdx.x;j<N;j+=blockDim.x)
  gamma[N-1-j]=bad?NAN:(a0[j]==0.?(a1[j]==0.?1.:NAN):a1[j]/a0[j]);
}
// Gamma is a horizontal field, common to all levels (paper equation 24).
// Temperature anomaly is the axisymmetric vortex component. Humidity uses
// the same-pressure constant-RH approximation (equations 26 and 30-32).
extern "C" __global__ void tc_thermal_balance(const double*T,const double*q,
 const double*axisymmetric_T,const double*gamma,double*newT,double*newq,int C,int K){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(C)*K)return;
 double g=gamma[i%C];
 if(!isfinite(T[i])||T[i]<=29.66||!isfinite(q[i])||q[i]<0.){newT[i]=newq[i]=NAN;return;}
 if(g==1.){newT[i]=T[i];newq[i]=q[i];return;}
 double t=T[i]+(g-1.)*axisymmetric_T[i];
 bool bad=!isfinite(g)||g<0.||!isfinite(t)||t<=29.66||!isfinite(T[i])||T[i]<=29.66||!isfinite(q[i])||q[i]<0.;
 newT[i]=bad?NAN:t;
 newq[i]=bad?NAN:q[i]*exp(17.67*243.5*(t-T[i])/((t-29.66)*(T[i]-29.66)));
}
// Azimuthal average of the separated vortex on physical radii. For winds,
// use the terminal great-circle basis at each sample, not the centre bearing.
extern "C" __global__ void tc_profiles(const double*field,const double*basic,
 const double*ring,const double*mean,const double*radii,double*out,
 int X,int Y,double lat0,double lon0,double dlat,double dlon,
 int BX,int BY,double blat0,double blon0,int P,int angles,
 double clat,double clon,double radius,int R,int vector){
 int j=blockIdx.x,p=blockIdx.y,a=threadIdx.x;
 int levels=vector?P/2:P; if(j>=R||p>=levels)return;
 extern __shared__ double values[];
 double value=0.,radial=0.;
 if(a<angles && j<R-1){
  double r=radii[j],bearing=2.*PI*a/angles,lat,lon;
  destination(clat,clon,r,bearing,lat,lon);
  Grid g{X,Y,lat0,lon0,dlat,dlon},bg{BX,BY,blat0,blon0,1.,1.};
  double u=anomaly(field+int64_t(p)*X*Y,basic+int64_t(p)*BX*BY,
   ring+p*angles,mean[p],g,bg,angles,clat,clon,radius,lat,lon);
  if(vector){
   int q=p+levels;
   double v=anomaly(field+int64_t(q)*X*Y,basic+int64_t(q)*BX*BY,
    ring+q*angles,mean[q],g,bg,angles,clat,clon,radius,lat,lon);
   double dummy,back;polar(lat,lon,clat,clon,dummy,back);
   double t=r<1.e-8?bearing:back+PI;
   // Positive tangential is counterclockwise viewed from above (NH cyclonic).
   value=-u*cos(t)+v*sin(t);radial=u*sin(t)+v*cos(t);
   if(r==0.)value=radial=0.;
  }else value=u;
 }
 values[a]=value;if(vector)values[blockDim.x+a]=radial;__syncthreads();
 for(int stride=blockDim.x/2;stride;stride/=2){
  if(a<stride){values[a]+=values[a+stride];if(vector)values[blockDim.x+a]+=values[blockDim.x+a+stride];}
  __syncthreads();
 }
 if(a==0){out[p*R+j]=values[0]/angles;if(vector)out[(p+levels)*R+j]=values[blockDim.x]/angles;}
}
