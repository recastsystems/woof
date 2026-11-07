// WRF v4.7.1 phys/module_wind_mav.F (public domain, UCAR notice in tools/wind_mav_wrf471_oracle/NOTICE.txt).
// Resident graph-frontier wake solve; one block per direction/model arm.
// Rotor levels and occupied columns have unique writers; native sum order is retained.
// MAV column stage from WRF 4.6.1 module_wind_mav.F, lines 828-942.
// No atomics: turbines sharing a cell accumulate in input order.
__device__ float mav_interp(float x,const float* grid,const float* data,int n,int stride,int t) {
    if(x<grid[t]) return data[t];
    if(x>=grid[t+(n-1)*stride]) return data[t+(n-1)*stride];
    int k=0;
    for(;k<n-1;++k) if(x>=grid[t+k*stride] && x<grid[t+(k+1)*stride]) break;
    float zk=float(k+1)+(x-grid[t+k*stride])/(grid[t+(k+1)*stride]-grid[t+k*stride]);
    int k1=int(zk);
    float d=zk-float(k1),dm=float(k1+1)-zk;
    return dm*data[t+(k1-1)*stride]+d*data[t+k1*stride];
}
__device__ void mav_drag(float speed,const float* ws,const float* tc,const float* pc,
const int* nv,const float* stc,const float* stc2,int nt,int t,float correction,
float& tk,float& pw,float& th) {
    if(speed<ws[t]) {th=stc[t];pw=0.0f;}
    else if(speed>ws[t+(nv[t]-1)*nt]) {th=stc2[t];pw=0.0f;}
    else {th=mav_interp(speed,ws,tc,nv[t],nt,t);pw=mav_interp(speed,ws,pc,nv[t],nt,t);}
    tk=correction*fmaxf(0.0f,th-pw);
}
__device__ float mav_turbine_area(float z1,float z2,float diameter) {
    float r=0.5f*diameter;
    z1=r-z1;z2=r-z2;
    float a=fabsf(z1),b=fabsf(z2);
    // Keep the full WRF left-to-right expression, not a subtraction of
    // separately rounded circular-segment areas.
    if(z1>0.0f && z2>0.0f)
        return a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r)
             -b*__fsqrt_rn(r*r-b*b)-r*r*glibc_asinf(b/r);
    if(z1<0.0f && z2<0.0f)
        return b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r)
             -a*__fsqrt_rn(r*r-a*a)-r*r*glibc_asinf(a/r);
    return b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r)
         +a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r);
}
__device__ double mav_atan_series(double x) {
#ifdef GPUWM_WIND_FAST
    // Native Fitch ATAN has REAL32 input and result.
    return double(atanf(float(x)));
#endif
    double ax=fabs(x),offset=0.0;
    bool inv=ax>1.0;
    if(inv) ax=1.0/ax;
    if(ax>0.4142135623730950488) {ax=(ax-1.0)/(ax+1.0);offset=0.7853981633974483096;}
    double term=ax,sum=ax,xx=ax*ax;
    for(int n=1;n<=36;++n) {term=-term*xx;sum=sum+term/double(2*n+1);}
    sum=offset+sum;
    if(inv) sum=1.5707963267948966192-sum;
    return copysign(sum,x);
}
__device__ float mav_atan2(float y,float x) {
#ifdef GPUWM_WIND_FAST
    return atan2f(y,x);
#endif
    // Preserve quadrant identity at signed zero, before forming a ratio.
    if(y==0.0f) return signbit(x)?copysignf(3.14159265358979323846f,y):y;
    if(x==0.0f) return copysignf(1.5707963267948966192f,y);
    double a=mav_atan_series(double(y)/double(x));
    if(x<0.0f) a=a+copysign(3.14159265358979323846,double(y));
    return __double2float_rn(a);
}
__device__ float mav_erf(float x) {
#ifdef GPUWM_WIND_FAST
    return erff(x);
#endif
    // Original positive-series evaluation of the erf integral, in double.
    // For |x| >= 4 the binary32 result rounds to one.
    if(!isfinite(x)) return isnan(x)?x:copysignf(1.0f,x);
    double a=fabs(double(x));
    if(a>=4.0) return copysignf(1.0f,x);
    double term=a,sum=a,xx=2.0*a*a;
    for(int n=1;n<=100;++n) {term=term*xx/double(2*n+1);sum=sum+term;}
    return copysignf(__double2float_rn(1.12837916709551257390*plm_exp(-a*a)*sum),x);
}
__device__ float mav_overlap(float y1,float y2,float z1,float z2,float r1,float r2) {
    float dz=z1-z2,dy=y1-y2,c=__fsqrt_rn(dz*dz+dy*dy);
    float small=fminf(r2,r1),large=fmaxf(r2,r1);
    if(c+small<=large) return 3.14159265358979323846f*(small*small);
    if(r1+r2<=c) return 0.0f;
    float cbd=glibc_acosf((r2*r2+c*c-r1*r1)/(2.0f*r2*c));
    float cad=glibc_acosf((r1*r1+c*c-r2*r2)/(2.0f*r1*c));
    return cbd*(r2*r2)+cad*(r1*r1)-0.5f*(r2*r2)*glibc_sinf(2.0f*cbd)
          -0.5f*(r1*r1)*glibc_sinf(2.0f*cad);
}
__device__ float mav_gaussian(float ch,float ck,float r,float sx,float sy) {
    const float ww[24]={.1279381953467522f,.1279381953467522f,.1258374563468283f,.1258374563468283f,
    .1216704729278034f,.1216704729278034f,.1155056680537256f,.1155056680537256f,
    .1074442701159656f,.1074442701159656f,.0976186521041139f,.0976186521041139f,
    .0861901615319533f,.0861901615319533f,.0733464814110803f,.0733464814110803f,
    .0592985849154368f,.0592985849154368f,.0442774388174198f,.0442774388174198f,
    .0285313886289337f,.0285313886289337f,.0123412297999872f,.0123412297999872f};
    const float xx[24]={-.0640568928626056f,.0640568928626056f,-.1911188674736163f,.1911188674736163f,
    -.3150426796961634f,.3150426796961634f,-.4337935076260451f,.4337935076260451f,
    -.5454214713888396f,.5454214713888396f,-.6480936519369755f,.6480936519369755f,
    -.7401241915785544f,.7401241915785544f,-.8200019859739029f,.8200019859739029f,
    -.8864155270044011f,.8864155270044011f,-.9382745520027328f,.9382745520027328f,
    -.9747285559713095f,.9747285559713095f,-.9951872199970213f,.9951872199970213f};
    float sum=0.0f;
    for(int i=0;i<24;++i) {
        float t=0.5f*xx[i]+0.5f;
        float d0=(ck-r*t*__fsqrt_rn(2.0f-t*t))/(__fsqrt_rn(2.0f)*sy);
        float d1=(ck+r*t*__fsqrt_rn(2.0f-t*t))/(__fsqrt_rn(2.0f)*sy);
        float am=(ch-r*(1.0f-t*t))/sx,ap=(ch+r*(1.0f-t*t))/sx;
        float p=(gfk_exp(-0.5f*(am*am))+gfk_exp(-0.5f*(ap*ap)))*(mav_erf(d1)-mav_erf(d0))*t;
        sum=sum+0.5f*ww[i]*p;
    }
    float a=(2.0f*3.14159265358979323846f*sx*sy)*r/sx/__fsqrt_rn(2.0f*3.14159265358979323846f);
    return a*sum/(3.14159265358979323846f*(r*r));
}
__device__ void mav_rotation(float x,float y,float theta,float& xr,float& yr) {
    xr=x*glibc_cosf(theta)+y*glibc_sinf(theta);
    yr=-x*glibc_sinf(theta)+y*glibc_cosf(theta);
}

__device__ void mav_solve(const float* ui,float* ul,const float* diameter,const float* area,const float* kw,
const int* nup,const int* up,const float* ax,const float* ay,const float* az,const float* ao,
const int* nv,const float* ws,const float* tc,const float* pc,const float* stc,const float* stc2,
float* def,int nt,int t,int model,int overlap,float correction) {
        if(nup[t]==0) return;
        if(model==3) {
            int points=0,blocked=0;
            float sumdist=0.0f,sumrecip=0.0f;
            // WRF packs blocked distances first and trailing ones next.
            // Accumulate blocked values in sample order, then the ones.
            for(int jd=1;jd<=50;++jd) for(int kd=1;kd<=50;++kd) {
                float y=-1.0f+(float(jd)-0.5f)/50.0f*2.0f;
                float z=-1.0f+(float(kd)-0.5f)/50.0f*2.0f;
                if(y*y+z*z>=1.0f) continue;
                ++points;
                y=y*(0.5f*diameter[t]);z=z*(0.5f*diameter[t]);
                for(int p=nup[t]-1;p>=0;--p) {
                    int j=up[t*nt+p],q=t*nt+j;
                    float dy=y-ay[q],dz=z-az[q],r=0.5f*diameter[j];
                    if(dy*dy+dz*dz<r*r) {
                        ++blocked;
                        sumdist=sumdist+ax[q]/(20.0f*2.0f*r);
                        sumrecip=sumrecip+1.0f/ax[q];
                        break;
                    }
                }
            }
            for(int p=blocked;p<points;++p) sumdist=sumdist+1.0f;
            float bf=float(blocked)/float(points),bd=sumdist/float(points),br=sumrecip/float(points);
            if(bd>1.0f) bf=0.0f;
            float g=bf==0.0f?1.0f:0.9615f-0.1549f*bf-0.0114f*br*20.0f*2.0f*(0.5f*diameter[t]);
            ul[t]=ui[t]*g;
            return;
        }
        for(int j=0;j<nt;++j) def[j]=0.0f;
        for(int p=0;p<nup[t];++p) {
            int j=up[t*nt+p],q=t*nt+j;
            float tmp,pw,th;
            mav_drag(ul[j],ws,tc,pc,nv,stc,stc2,nt,j,correction,tmp,pw,th);
            float deficit;
            if(model==1) {
                float den=1.0f+kw[j]*ax[q]/(0.5f*diameter[j]);
                deficit=(1.0f-__fsqrt_rn(1.0f-th))/(den*den)*ao[q];
                if(overlap==1 || overlap==2) def[j]=ui[j]*deficit*ao[q];
                else if(overlap==3) def[j]=ul[j]*deficit*ao[q];
                else def[j]=ui[t]*(1.0f-ao[q])+ui[j]*(1.0f-deficit)*ao[q];
            } else {
                float beta=0.5f*(1.0f+__fsqrt_rn(1.0f-th))/__fsqrt_rn(1.0f-th);
                float eps=0.25f*__fsqrt_rn(beta),r=0.5f*diameter[j];
                float sy=0.025f*ax[q]+eps*2.0f*r,sz=0.0175f*ax[q]+eps*2.0f*r;
                float avg=mav_gaussian(ay[q],az[q],0.5f*diameter[t],sy,sz);
                deficit=(1.0f-__fsqrt_rn(1.0f-(r*r)*th/sy/sz/2.0f))*avg;
                if(overlap==1 || overlap==2) def[j]=ui[j]*deficit;
                else if(overlap==3) def[j]=ul[j]*deficit;
                else def[j]=ui[j]*(1.0f-deficit);
            }
        }
        float sum=0.0f;
        for(int j=0;j<nt;++j) sum=sum+(overlap==1?def[j]:def[j]*def[j]);
        if(overlap==1) ul[t]=ui[t]-sum;
        else if(overlap==2 || overlap==3) ul[t]=ui[t]-__fsqrt_rn(sum);
        else ul[t]=__fsqrt_rn(sum/float(nup[t]));
}


#ifdef GPUWM_WIND_FAST
// The 24 independent native XA nodes share a warp. Coefficients live in
// constant storage, rather than dynamically indexed per-thread local arrays.
__device__ __constant__ float mav_quad_w[24]={.1279381953467522f,.1279381953467522f,.1258374563468283f,.1258374563468283f,
    .1216704729278034f,.1216704729278034f,.1155056680537256f,.1155056680537256f,
    .1074442701159656f,.1074442701159656f,.0976186521041139f,.0976186521041139f,
    .0861901615319533f,.0861901615319533f,.0733464814110803f,.0733464814110803f,
    .0592985849154368f,.0592985849154368f,.0442774388174198f,.0442774388174198f,
    .0285313886289337f,.0285313886289337f,.0123412297999872f,.0123412297999872f};
__device__ __constant__ float mav_quad_x[24]={-.0640568928626056f,.0640568928626056f,-.1911188674736163f,.1911188674736163f,
    -.3150426796961634f,.3150426796961634f,-.4337935076260451f,.4337935076260451f,
    -.5454214713888396f,.5454214713888396f,-.6480936519369755f,.6480936519369755f,
    -.7401241915785544f,.7401241915785544f,-.8200019859739029f,.8200019859739029f,
    -.8864155270044011f,.8864155270044011f,-.9382745520027328f,.9382745520027328f,
    -.9747285559713095f,.9747285559713095f,-.9951872199970213f,.9951872199970213f};

__device__ float mav_gaussian_warp(float ch,float ck,float r,float sx,float sy) {
    int lane=threadIdx.x&31;
    float sum=0.f;
    if(lane<24) {
        float t=.5f*mav_quad_x[lane]+.5f;
        float d0=(ck-r*t*sqrtf(2.f-t*t))/(sqrtf(2.f)*sy);
        float d1=(ck+r*t*sqrtf(2.f-t*t))/(sqrtf(2.f)*sy);
        float am=(ch-r*(1.f-t*t))/sx,ap=(ch+r*(1.f-t*t))/sx;
        sum=.5f*mav_quad_w[lane]*(expf(-.5f*am*am)+expf(-.5f*ap*ap))*(erff(d1)-erff(d0))*t;
    }
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    sum=__shfl_sync(0xffffffff,sum,0);
    float a=(2.f*3.14159265358979323846f*sx*sy)*r/sx/sqrtf(2.f*3.14159265358979323846f);
    return a*sum/(3.14159265358979323846f*r*r);
}
__device__ void mav_solve_warp(const float* ui,float* ul,const float* diameter,const float* area,const float* kw,
const int* nup,const int* up,const float* ax,const float* ay,const float* az,const float* ao,
const int* nv,const float* ws,const float* tc,const float* pc,const float* stc,const float* stc2,
float* def,int nt,int t,int model,int overlap,float correction) {
    int lane=threadIdx.x&31;
    if(nup[t]==0)return;
    if(model==1) {
        float sum=0.f;
        for(int p=lane;p<nup[t];p+=32){
            int j=up[t*nt+p],q=t*nt+j;float tk,pw,th;
            mav_drag(ul[j],ws,tc,pc,nv,stc,stc2,nt,j,correction,tk,pw,th);
            float den=1.f+kw[j]*ax[q]/(.5f*diameter[j]);
            float deficit=(1.f-sqrtf(1.f-th))/(den*den)*ao[q];
            float value=(overlap==1 || overlap==2)?ui[j]*deficit*ao[q]:
                        overlap==3?ul[j]*deficit*ao[q]:ui[t]*(1.f-ao[q])+ui[j]*(1.f-deficit)*ao[q];
            sum+=overlap==1?value:value*value;
        }
        for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if(lane==0)ul[t]=overlap==1?ui[t]-sum:
                        (overlap==2 || overlap==3)?ui[t]-sqrtf(sum):sqrtf(sum/float(nup[t]));
        return;
    }
    float sum=0.f;
    for(int p=0;p<nup[t];++p) {
        int j=up[t*nt+p],q=t*nt+j;
        float tmp,pw,th;
        mav_drag(ul[j],ws,tc,pc,nv,stc,stc2,nt,j,correction,tmp,pw,th);
        float root=sqrtf(1.f-th),beta=.5f*(1.f+root)/root;
        float eps=.25f*sqrtf(beta),r=.5f*diameter[j];
        float sy=.025f*ax[q]+eps*2.f*r,sz=.0175f*ax[q]+eps*2.f*r;
        float avg=mav_gaussian_warp(ay[q],az[q],.5f*diameter[t],sy,sz);
        float deficit=(1.f-sqrtf(1.f-r*r*th/sy/sz/2.f))*avg;
        float value=(overlap==1 || overlap==2)?ui[j]*deficit:
                    overlap==3?ul[j]*deficit:ui[j]*(1.f-deficit);
        sum+=overlap==1?value:value*value;
    }
    if(lane==0)ul[t]=overlap==1?ui[t]-sum:
                    (overlap==2 || overlap==3)?ui[t]-sqrtf(sum):sqrtf(sum/float(nup[t]));
}
#endif

__device__ __forceinline__ float mav_mass_at(const float* zw,const float* zm,int p,int nc){
#ifdef GPUWM_WIND_FAST
 // Only turbine columns need heights. Avoid a full-domain preparation pass.
 return .5f*(zw[p]+zw[p+nc]);
#else
 return zm[p];
#endif
}
extern "C" __global__ void mav_wake(
const float* u,const float* v,const float* zw,const float* zm,const float* land,const float* co,const float* si,
const float* xt,const float* yt,const float* hub,const float* diameter,const float* area,const int* col,
const int* nv,const float* ws,const float* tc,const float* pc,const float* stc,const float* stc2,
const int* models,const int* methods,const float* directions,
float* all_ui,float* all_ul,float* work,int* iwork,int* status,int nc,int nz,int nt,float correction) {
 int arm=blockIdx.x,nn=nt*nt,model=models[arm],overlap=methods[arm];
 float dtheta=directions[arm];
 float* ui=all_ui+arm*nt;float* ul=all_ul+arm*nt;
 float* hu=work+arm*(5*nt+6*nn);float* hv=hu+nt;float* terrain=hv+nt;float* kw=terrain+nt;float* xl=kw+nt;
 float* ax=xl+nt;float* ay=ax+nn;float* az=ay+nn;float* ao=az+nn;float* defs=ao+nn;
 int* nup=iwork+arm*(3*nt+nn);int* flag=nup+nt;int* ready=flag+nt;int* up=ready+nt;
 for(int t=threadIdx.x;t<nt;t+=blockDim.x) {
  int c=col[t];hu[t]=hv[t]=terrain[t]=0.f;xl[t]=0.f;
  if(c>=0 && c<nc){
   float h=hub[t],ground=zw[c];int k=0;
   // Native TO_ZK2 clamps to the first/last mass level, with its one-based fractional index.
   if(h<mav_mass_at(zw,zm,c,nc)-ground)k=0;
   else if(h>=mav_mass_at(zw,zm,(nz-1)*nc+c,nc)-ground)k=nz-1;
   else for(;k<nz-1;++k)if(h>=mav_mass_at(zw,zm,k*nc+c,nc)-ground && h<mav_mass_at(zw,zm,(k+1)*nc+c,nc)-ground)break;
   int p=k*nc+c;
   float a=u[p]*co[c]-v[p]*si[c],b=v[p]*co[c]+u[p]*si[c];
   if(k<nz-1 && h>=mav_mass_at(zw,zm,c,nc)-ground){
    float z0=mav_mass_at(zw,zm,p,nc)-ground,z1=mav_mass_at(zw,zm,p+nc,nc)-ground;
    float zk=float(k+1)+(h-z0)/(z1-z0),d=zk-float(k+1),dm=float(k+2)-zk;
    a=dm*a+d*(u[p+nc]*co[c]-v[p+nc]*si[c]);
    b=dm*b+d*(v[p+nc]*co[c]+u[p+nc]*si[c]);
   }
   hu[t]=a;hv[t]=b;terrain[t]=ground;xl[t]=land[c];
  }else{hu[t]=hv[t]=1.e-3f;}
  ui[t]=__fsqrt_rn(hu[t]*hu[t]+hv[t]*hv[t]);ul[t]=ui[t];
  kw[t]=xl[t]>1.5f?.04f:.0075f;flag[t]=0;
 }
 __syncthreads();
 for(int t=threadIdx.x;t<nt;t+=blockDim.x){
  nup[t]=0;
  if(col[t]<0 || col[t]>=nc)continue;
  for(int j=0;j<nt;++j){
   if(j==t)continue;
   float xp=xt[j]-xt[t],yp=yt[j]-yt[t];
   if(__fsqrt_rn(yp*yp+xp*xp)>20.f*diameter[0])continue;
#ifdef GPUWM_WIND_FAST
   // The native double ACOS is used only for a 30-degree predicate.
   // Its cosine test needs neither an inverse trig call nor binary64.
   float dot=-(hu[t]*xp+hv[t]*yp);
   float norm2=(hu[t]*hu[t]+hv[t]*hv[t])*(xp*xp+yp*yp);
   if(dot<0.f || dot*dot<.75f*norm2)continue;
   float theta;
#else
   double a=double(-(hu[t]*xp+hv[t]*yp));
   double b=double(__fsqrt_rn((hu[t]*hu[t]+hv[t]*hv[t])*(xp*xp+yp*yp)));
   if(fabs(b)<fabs(a))b=copysign(a,b);
   double ratio=a/b;
   float theta=(ratio< -1. || ratio>1.)?0.f:__double2float_rn(uw_acos(ratio));
   if(fabsf(theta)>30.f*3.14159265358979323846f/180.f)continue;
#endif
   theta=mav_atan2(hv[t]+hv[j],hu[t]+hu[j])+dtheta;
   float x,y,xj,yj;mav_rotation(xt[t],yt[t],theta,x,y);mav_rotation(xt[j],yt[j],theta,xj,yj);
   float dist=x-xj,aover=0.f;
   if(dist>0.f){
    float width=.5f*diameter[j]+(model==3?0.f:(model==2?5.f*kw[j]:kw[j]))*dist;
    aover=mav_overlap(y,yj,hub[t]+terrain[t],hub[j]+terrain[j],.5f*diameter[t],width);
   }
   if(aover/area[t]>.01f){
    int q=t*nt+j;up[t*nt+nup[t]++]=j;
    ax[q]=dist;ay[q]=y-yj;az[q]=(hub[t]+terrain[t])-(hub[j]+terrain[j]);ao[q]=aover/area[t];
   }
  }
  if(model==3)for(int p=0;p<nup[t]-1;++p){
   int best=p;for(int q=p+1;q<nup[t];++q)if(ax[t*nt+up[t*nt+q]]>ax[t*nt+up[t*nt+best]])best=q;
   int chosen=up[t*nt+best];for(int q=best;q>p;--q)up[t*nt+q]=up[t*nt+q-1];up[t*nt+p]=chosen;
  }
 }
 __syncthreads();
#ifdef GPUWM_WIND_FAST
 // GM depends on ambient speed alone, so its 2500-point disk samples
 // need no graph frontier. A warp owns each turbine; arms are blocks.
 if(model==3){
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int t=warp;t<nt;t+=blockDim.x/32){
   float dist=0.f,recip=0.f;int points=0,blocked=0;
   int count=nup[t];
   // Cache the common single-upstream geometry outside the disk loop.
   // This avoids reloading the same global graph edge 2500/32 times.
   float gy=0.f,gz=0.f,gr=0.f,gx=0.f;
   if(count==1){int j=up[t*nt],q=t*nt+j;gy=ay[q];gz=az[q];gr=.5f*diameter[j];gx=ax[q];}
   if(count)for(int sample=lane;sample<2500;sample+=32){
    int jd=sample/50+1,kd=sample%50+1;
    float y=-1.f+(float(jd)-.5f)/50.f*2.f,z=-1.f+(float(kd)-.5f)/50.f*2.f;
    if(y*y+z*z>=1.f)continue;++points;y*=.5f*diameter[t];z*=.5f*diameter[t];
    if(count==1){
     float dy=y-gy,dz=z-gz;
     if(dy*dy+dz*dz<gr*gr){++blocked;dist+=gx/(20.f*2.f*gr);recip+=1.f/gx;}
    }else for(int p=count-1;p>=0;--p){int j=up[t*nt+p],q=t*nt+j;float dy=y-ay[q],dz=z-az[q],r=.5f*diameter[j];
     if(dy*dy+dz*dz<r*r){++blocked;dist+=ax[q]/(20.f*2.f*r);recip+=1.f/ax[q];break;}
    }
   }
   for(int delta=16;delta;delta/=2){points+=__shfl_down_sync(0xffffffff,points,delta);blocked+=__shfl_down_sync(0xffffffff,blocked,delta);dist+=__shfl_down_sync(0xffffffff,dist,delta);recip+=__shfl_down_sync(0xffffffff,recip,delta);}
   if(lane==0 && points){dist+=float(points-blocked);float bf=float(blocked)/float(points),bd=dist/float(points),br=recip/float(points);if(bd>1.f)bf=0.f;
    float g=bf==0.f?1.f:.9615f-.1549f*bf-.0114f*br*20.f*2.f*(.5f*diameter[t]);ul[t]=ui[t]*g;}
  }
  return;
 }
#endif
 for(int depth=0;depth<nt;++depth){
  int pending=0;
  for(int t=threadIdx.x;t<nt;t+=blockDim.x){
   ready[t]=0;
   if(flag[t])continue;
   bool ok=true;for(int p=0;p<nup[t];++p)if(!flag[up[t*nt+p]])ok=false;
   ready[t]=ok;pending++;
  }
  if(__syncthreads_count(pending)==0)return;
#ifdef GPUWM_WIND_FAST
  for(int t=threadIdx.x/32;t<nt;t+=blockDim.x/32)if(ready[t]){
   mav_solve_warp(ui,ul,diameter,area,kw,nup,up,ax,ay,az,ao,nv,ws,tc,pc,stc,stc2,defs+t*nt,nt,t,model,overlap,correction);
  }
#else
  for(int t=threadIdx.x;t<nt;t+=blockDim.x)if(ready[t]){
   mav_solve(ui,ul,diameter,area,kw,nup,up,ax,ay,az,ao,nv,ws,tc,pc,stc,stc2,defs+t*nt,nt,t,model,overlap,correction);
  }
#endif
  __syncthreads();
  int progress=0;for(int t=threadIdx.x;t<nt;t+=blockDim.x)if(ready[t]){flag[t]=1;progress++;}
  if(__syncthreads_count(progress)==0){if(threadIdx.x==0)status[arm]=-2;return;}
 }
}
// A block owns one rotor in one arm; all rotor layers participate concurrently.
extern "C" __global__ void mav_rotor(
const float* u,const float* v,const float* dz,const float* zw,const float* ui,const float* ul,
const float* diameter,const float* hub,const float* area,const int* col,const int* nv,
const float* ws,const float* tc,const float* pc,const float* stc,const float* stc2,const float* weights,
float* rotor,float* turbine_power,int* status,int nc,int nz,int nt,float dt,float density,float correction){
 int t=blockIdx.x,arm=blockIdx.y,c=col[t],k=threadIdx.x;
 extern __shared__ float values[];float* powerz=values;
 float tk,pw,th;mav_drag(ul[arm*nt+t],ws,tc,pc,nv,stc,stc2,nt,t,correction,tk,pw,th);
 float sp=ul[arm*nt+t],power1=.5f*1.23f*((sp*sp)*sp)*area[t],weight=weights[arm];power1=power1*pw;
 if(k==0)turbine_power[arm*nt+t]=(c>=0 && c<nc)?power1*weight:0.f;
 float tarea=0.f,speed=0.f,powerk=0.f;
 int bot=-1,top=-2;
 if(c>=0 && c<nc){
  float ground=zw[c],lo=hub[t]-.5f*diameter[t],hi=hub[t]+.5f*diameter[t];
  for(int z=0;z<nz;++z){float a=zw[z*nc+c]-ground,b=zw[(z+1)*nc+c]-ground;
   if(lo>=a && lo<b)bot=z;if(hi>=a && hi<b)top=z;
  }
  if(bot<0 || top<bot){if(k==0)status[arm]=t+1;}
  else if(k>=bot && k<=top && k<nz){
   int p=k*nc+c;float z1=fmaxf(zw[p]-ground-lo,0.f),z2=fminf(zw[p+nc]-ground-lo,diameter[t]);
   tarea=mav_turbine_area(z1,z2,diameter[t]);speed=(sp/ui[arm*nt+t])*__fsqrt_rn(u[p]*u[p]+v[p]*v[p]);
   powerk=.5f*1.23f*((speed*speed)*speed)*tarea*pw;
  }
 }
 if(k<nz)powerz[k]=powerk;
 __syncthreads();
 // Identical native left-fold in each level lane, avoiding a serial reduction stage.
 float power2=0.f;for(int z=0;z<nz;++z)power2=power2+powerz[z];
 float ec=(power1==0.f || power2==0.f)?1.f:power1/power2;ec=ec*density*weight;
 if(k<nz){
  float a=0.f,b=0.f,d=0.f;
  if(c>=0 && c<nc && k>=bot && k<=top && bot>=0){int p=k*nc+c;
   a=0.f-.5f*u[p]*th*speed*tarea/dz[p]*ec;b=0.f-.5f*v[p]*th*speed*tarea/dz[p]*ec;
   d=.5f*((speed*speed)*speed)*tk*tarea/dz[p]*dt*ec;
  }
  int p=((arm*nt+t)*nz+k)*3;rotor[p]=a;rotor[p+1]=b;rotor[p+2]=d;
 }
 if(k==0)turbine_power[nt*gridDim.y+arm*nt+t]=power2*weight;
}
extern "C" __global__ void mav_accumulate(
const int* cols,const int* ptr,const int* ids,const float* rotor,const float* powers,
float* du,float* dv,float* qke,float* adv,float* tke,float* tke_increment,float* power,float* turbine,
const float* ul,const float* ui,const float* weights,float* hub,
int nc,int nz,int nt,int arms,int models,int tkeadvect,float dt){
 int c=cols[blockIdx.x];
#ifdef GPUWM_WIND_FAST
 // One cooperative block per occupied column and level. Parallelize the
 // independent arm/turbine contributions instead of serial memory loads
 // in just one block for the entire vertical column. Strict mode retains
 // WRF's left fold below for the bitwise oracle.
 int k=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x>>5;
 __shared__ float partials[12];
 float a=0.f,b=0.f,d=0.f;
 int count=ptr[c+1]-ptr[c];
 for(int j=threadIdx.x;j<arms*count;j+=blockDim.x){
  int arm=j/count,t=ids[ptr[c]+j%count],p=((arm*nt+t)*nz+k)*3;
  a+=rotor[p];b+=rotor[p+1];d+=rotor[p+2];
 }
 for(int offset=16;offset;offset>>=1){
  a+=__shfl_down_sync(0xffffffff,a,offset);
  b+=__shfl_down_sync(0xffffffff,b,offset);
  d+=__shfl_down_sync(0xffffffff,d,offset);
 }
 if(lane==0){partials[3*warp]=a;partials[3*warp+1]=b;partials[3*warp+2]=d;}
 __syncthreads();
 if(threadIdx.x==0){
  a=b=d=0.f;
  for(int w=0;w<4;++w){a+=partials[3*w];b+=partials[3*w+1];d+=partials[3*w+2];}
#else
 int k=threadIdx.x;
 if(k<nz){
  float a=0.f,b=0.f,d=0.f;
  for(int arm=0;arm<arms;++arm)for(int j=ptr[c];j<ptr[c+1];++j){int t=ids[j],p=((arm*nt+t)*nz+k)*3;
   a=a+rotor[p];b=b+rotor[p+1];d=d+rotor[p+2];
  }
#endif
  int p=k*nc+c;du[p]=du[p]+a/float(models);dv[p]=dv[p]+b/float(models);float tk=d/float(models);tke[p]=tk/dt;tke_increment[p]=tk;
  if(tkeadvect){qke[p]=qke[p]+2.f*tk;adv[p]=qke[p];}
 }
#ifdef GPUWM_WIND_FAST
 // A warp owns a turbine's independent arm diagnostics. The column sum
 // shares the same block, so there is no extra launch or atomic update.
 if(k==0){
  float column_power=0.f;
  for(int j=warp;j<count;j+=4){
   int t=ids[ptr[c]+j];float p=0.f,w=0.f,p2=0.f;
   for(int arm=lane;arm<arms;arm+=32){
    p+=powers[arm*nt+t];p2+=powers[nt*arms+arm*nt+t];w+=ul[arm*nt+t]*weights[arm];
   }
   for(int d=16;d;d>>=1){p+=__shfl_down_sync(0xffffffff,p,d);p2+=__shfl_down_sync(0xffffffff,p2,d);w+=__shfl_down_sync(0xffffffff,w,d);}
   if(lane==0){turbine[t]=p/float(models);hub[10*t]=w/float(models);hub[10*t+1]=p/float(models);column_power+=p2;}
  }
  if(lane==0)partials[warp]=column_power;
  __syncthreads();
  if(threadIdx.x==0)power[c]=(partials[0]+partials[1]+partials[2]+partials[3])/float(models);
 }
#else
 if(k==0 && threadIdx.x==0){
  float value=0.f;
  for(int arm=0;arm<arms;++arm)for(int j=ptr[c];j<ptr[c+1];++j)value=value+powers[nt*arms+arm*nt+ids[j]];
  power[c]=value/float(models);
  for(int j=ptr[c];j<ptr[c+1];++j){int t=ids[j];float p=0.f,w=0.f;
   // power_nt_md is accumulated over directions separately, then over models.
   for(int m=0;m<models;++m){float pm=0.f;for(int dir=0;dir<7;++dir)pm=pm+powers[(dir*models+m)*nt+t];p=p+pm;}
   for(int arm=0;arm<arms;++arm)w=w+ul[arm*nt+t]*weights[arm];
   turbine[t]=p/float(models);hub[10*t]=w/float(models);hub[10*t+1]=p/float(models);
  }
 }
#endif
}
extern "C" __global__ void mav_layout_xy(const float* xy,float* xt,float* yt,int* col,int nt,int nx,int ny,float dx,float deg){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nt)return;
 float xc=0.f,yc=0.f;for(int j=0;j<nt;++j){xc=xc+xy[2*j];yc=yc+xy[2*j+1];}xc=xc/float(nt);yc=yc/float(nt);
 float x,y;mav_rotation(xy[2*t]-xc,xy[2*t+1]-yc,deg/180.f*3.14159265358979323846f,x,y);xt[t]=x;yt[t]=y;
 float px=float((nx+1)/3-1)*dx+x,py=float((ny+1)/3-1)*dx+y;int i=-1,j=-1;
 for(int z=0;z<nx;++z)if(float(z)*dx<=px && float(z+1)*dx>px){i=z;break;}
 for(int z=0;z<ny;++z)if(float(z)*dx<=py && float(z+1)*dx>py){j=z;break;}
 col[t]=(i<0 || j<0)?-1:j*nx+i;
}
extern "C" __global__ void mav_mass_height(const float* zi,float* zm,int nc,int nz){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<nc*nz)zm[p]=.5f*(zi[p]+zi[p+nc]);
}
extern "C" __global__ void mav_prepare_arms(float* weights,float* directions,int models){
 int arm=threadIdx.x;if(arm>=7*models)return;
 const float angles[7]={-2.5f,-1.5f,-.5f,0.f,.5f,1.5f,2.5f};
 float total=0.f;for(int j=0;j<7;++j)total=total+(j==3?1.f:gfk_exp(-(angles[j]*angles[j])/(2.f*(2.f*2.f))));
 int j=arm/models;weights[arm]=(j==3?1.f:gfk_exp(-(angles[j]*angles[j])/(2.f*(2.f*2.f))))/total;
 directions[arm]=angles[j]/180.f*3.14159265358979323846f;
}
// WRF share/module_llxy.F: set_lc/llij_lc, set_ps/llij_ps and set_merc/llij_merc.
__device__ float mav_wrap(float a){if(a>180.f)a=a-360.f;if(a< -180.f)a=a+360.f;return a;}
extern "C" __global__ void mav_project_ll(const float* ll,const float* lat,const float* lon,int* col,
 int nt,int nx,int ny,int projection,float dx,float tl1,float tl2,float stdlon){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nt)return;
 const float rad=3.14159265358979323846f/180.f,deg=180.f/3.14159265358979323846f;
 float lat1=lat[0],lon1=lon[0],la=ll[2*t],lo=ll[2*t+1],h=tl1<0.f?-1.f:1.f,re=6370000.f/dx,i,j;
 if(projection==1){
  float cone;
  if(fabsf(tl1-tl2)>.1f){
   cone=log10f(glibc_cosf(tl1*rad))-log10f(glibc_cosf(tl2*rad));
   cone=cone/(log10f(glibc_tanf((45.f-fabsf(tl1)/2.f)*rad))-log10f(glibc_tanf((45.f-fabsf(tl2)/2.f)*rad)));
  }else cone=glibc_sinf(fabsf(tl1)*rad);
  float ctl=glibc_cosf(tl1*rad),tan1=glibc_tanf((90.f*h-tl1)*rad/2.f);
  float rsw=re*ctl/cone*gfk_pow(glibc_tanf((90.f*h-lat1)*rad/2.f)/tan1,cone);
  float arg=cone*(mav_wrap(lon1-stdlon)*rad),pi=h-h*rsw*glibc_sinf(arg),pj=h+rsw*glibc_cosf(arg);
  float rm=re*ctl/cone*gfk_pow(glibc_tanf((90.f*h-la)*rad/2.f)/tan1,cone);
  arg=cone*(mav_wrap(lo-stdlon)*rad);i=h*(pi+h*rm*glibc_sinf(arg));j=h*(pj-rm*glibc_cosf(arg));
 }else if(projection==2){
  float ref=stdlon+90.f,scale=1.f+h*glibc_sinf(tl1*rad),a=lat1*rad;
  float rsw=re*glibc_cosf(a)*scale/(1.f+h*glibc_sinf(a)),alo=(lon1-ref)*rad;
  float pi=1.f-rsw*glibc_cosf(alo),pj=1.f-h*rsw*glibc_sinf(alo);
  a=la*rad;float rm=re*glibc_cosf(a)*scale/(1.f+h*glibc_sinf(a));alo=(lo-ref)*rad;
  i=pi+rm*glibc_cosf(alo);j=pj+h*rm*glibc_sinf(alo);
 }else{
  float dl=dx/(6370000.f*glibc_cosf(rad*tl1)),rsw=lat1==0.f?0.f:gfk_log(glibc_tanf(.5f*((lat1+90.f)*rad)))/dl;
  i=1.f+mav_wrap(lo-lon1)/(dl*deg);j=1.f+gfk_log(glibc_tanf(.5f*((la+90.f)*rad)))/dl-rsw;
 }
 int ix=int(roundf(i))-1,iy=int(roundf(j))-1;col[t]=(ix<0 || iy<0 || ix>=nx || iy>=ny)?-1:iy*nx+ix;
}
// WRF upstream Fitch induction additions: MIT, Fraunhofer IWES (2022).
// Full notice and immutable source pin are in tools/wind_mav_wrf471_oracle.
__device__ __forceinline__ float fi_area(float z1,float z2,float diameter,float density){
 float r=diameter/2.f;z1=r-z1;z2=r-z2;float a=fabsf(z1),b=fabsf(z2),v;
 float aa=a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r);
 float bb=b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r);
 if(z1>0.f && z2>0.f)v=aa-bb;
 else if(z1<0.f && z2<0.f)v=bb-aa;
 else v=b*__fsqrt_rn(r*r-b*b)+r*r*glibc_asinf(b/r)+a*__fsqrt_rn(r*r-a*a)+r*r*glibc_asinf(a/r);
 return v*density;
}
__device__ __forceinline__ float fi_powi(float x,int n){float y=1.f;for(;;){if(n&1)y=y*x;n>>=1;if(!n)return y;x=x*x;}}
__device__ __forceinline__ void fi_drag(float s,int t,int nt,const int* nv,const float* ws,const float* tc,const float* pw,
 const float* diam,const float* stc,const float* stc2,float factor,float& tk,float& pc,float& ct){
 int n=nv[t],k=1;for(;k<n;++k)if(ws[k*nt+t]>s)break;
 if(s<=ws[t])ct=stc[t];else if(s>=ws[(n-1)*nt+t])ct=stc2[t];
 else ct=tc[(k-1)*nt+t]+(tc[k*nt+t]-tc[(k-1)*nt+t])/(ws[k*nt+t]-ws[(k-1)*nt+t])*(s-ws[(k-1)*nt+t]);
 ct=fmaxf(fminf(ct,.9999f),.0001f);pc=0.f;
 if(s>ws[t] && s<ws[(n-1)*nt+t]){
 float area=(3.14159265358979323846f/4.f)*(diam[t]*diam[t]);
 float x=ws[(k-1)*nt+t],y=ws[k*nt+t];
 float f1=1000.f/(.5f*1.23f*((x*x)*x)*area),f2=1000.f/(.5f*1.23f*((y*y)*y)*area);
 pc=pw[(k-1)*nt+t]*f1+(pw[k*nt+t]*f2-pw[(k-1)*nt+t]*f1)/(y-x)*(s-x);
 }
 tk=factor*(ct-pc);if(tk<0.f)tk=0.f;
}
extern "C" __global__ void fitch_induction(
 const int* cols,const int* ptr,const int* ids,const float* diam,const float* hub,const int* nv,
 const float* ws,const float* tc,const float* pw,const float* stc,const float* stc2,
 const float* u,const float* v,const float* dz,const float* zw,
 float* du,float* dv,float* qke,float* adv,float* power,float* thrust,float* tke,float* tinc,
 float* hubout,float* turbine,int nc,int nz,int nt,float dx,float dt,float factor){
 int c=cols[blockIdx.x],k=threadIdx.x;extern __shared__ float ps[];
 float pk=0.f,tkall=0.f,thall=0.f;int count=ptr[c+1]-ptr[c];
 for(int it=ptr[c];it<ptr[c+1];++it){int t=ids[it];float lo=hub[t]-diam[t]/2.f,hi=hub[t]+diam[t]/2.f;
 int bot=-1,top=-2,kbot=0,ktop=0;float lower=0.f,z1=0.f,z2=0.f;
 for(int z=0;z<nz;++z){float upper=lower+dz[z*nc+c];
  if(lo>=lower && lo<=upper)bot=z;
  if(hi>=lower && hi<=upper){top=z;break;}lower=upper;
 }
 if(top<bot || bot<0)continue;
 lower=0.f;
 for(int z=0;z<nz;++z){z2=lower+.5f*dz[z*nc+c];if(hub[t]>z2)kbot=z;else{ktop=z;break;}z1=z2;lower=z2+.5f*dz[z*nc+c];}
 float area=(3.14159265358979323846f/4.f)*(diam[t]*diam[t]);
 float uh=(u[ktop*nc+c]+u[kbot*nc+c])/2.f,vh=(v[ktop*nc+c]+v[kbot*nc+c])/2.f;
 float angle=__double2float_rn(mav_atan_series(double(vh/uh)));
 float geom=area/(diam[t]*dx*fminf(fabsf(1.f/glibc_cosf(angle)),fabsf(1.f/glibc_sinf(angle))));
 float a=.25f*area/(diam[t]*dx*fminf(fabsf(1.f/glibc_cosf(angle)),fabsf(1.f/glibc_sinf(angle)))),base;
 if(ktop==0)base=__fsqrt_rn(u[c]*u[c]+v[c]*v[c])*hub[t]/z1;
 else{float s1=__fsqrt_rn(u[kbot*nc+c]*u[kbot*nc+c]+v[kbot*nc+c]*v[kbot*nc+c]);
 float s2=__fsqrt_rn(u[ktop*nc+c]*u[ktop*nc+c]+v[ktop*nc+c]*v[ktop*nc+c]);base=s1+((s2-s1)/(z2-z1))*(hub[t]-z1);}
 float sp=base/fi_powi(1.f-a,count),tk,pc,ct;fi_drag(sp,t,nt,nv,ws,tc,pw,diam,stc,stc2,factor,tk,pc,ct);
 // Preserve native multiplication/division order in the geometric correction.
 a=(.5f*(1.f-__fsqrt_rn(1.f-ct)))*area/(diam[t]*dx*fminf(fabsf(1.f/glibc_cosf(angle)),fabsf(1.f/glibc_sinf(angle))));
 float scale=fi_powi(1.f-a,count);sp=base/scale;
 fi_drag(sp,t,nt,nv,ws,tc,pw,diam,stc,stc2,factor,tk,pc,ct);
 float p1=.5f*1.23f*((sp*sp)*sp)*area*pc;
 float ta=0.f,speed=0.f,p2k=0.f;
 if(k<nz && k>=bot && k<=top){int p=k*nc+c;float a1=fmaxf(zw[p]-lo-zw[c],0.f),a2=fminf(zw[p+nc]-lo-zw[c],diam[t]);
 ta=fi_area(a1,a2,diam[t],1.f/(dx*dx));speed=__fsqrt_rn(u[p]*u[p]+v[p]*v[p])/scale;
 p2k=.5f*pc*1.23f*((speed*speed)*speed)*ta/(1.f/(dx*dx));}
 if(k<nz)ps[k]=p2k;__syncthreads();float p2=0.f;for(int z=0;z<nz;++z)p2=p2+ps[z];
 float ec=(p1==0.f || p2==0.f)?1.f:p1/p2;
 if(k<nz && k>=bot && k<=top){int p=k*nc+c;
 float delta=((speed*speed)*speed)*ta*tk*dt/dz[p]*ec;qke[p]=qke[p]+delta;tkall=tkall+delta;
 du[p]=du[p]-.5f*(u[p]/scale)*ct*speed*ta/dz[p]*ec;
 dv[p]=dv[p]-.5f*(v[p]/scale)*ct*speed*ta/dz[p]*ec;}
 if(k==0){pk=p1+pk;thall=.5f*1.23f*(sp*sp)*area*ct+thall;hubout[10*t]=sp;turbine[t]=p1;}
 __syncthreads();
 }
 if(k<nz){tinc[k*nc+c]=tkall;tke[k*nc+c]=tkall/dt;adv[k*nc+c]=qke[k*nc+c];}
 if(k==0){power[c]=pk;thrust[c]=thall;}
}
