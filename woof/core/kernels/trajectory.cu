// WRF 4.6.1 module_em:trajectory and module_llxy, public-domain source.
// Licence and byte pins: tools/trajectory_wrf461_oracle/source_manifest.json.
// Coordinates remain WRF's one-based floating mass-grid indices.
#if WRF_EXACT
#define TSIN glibc_sinf
#define TCOS glibc_cosf
#define TTAN glibc_tanf
#define TATAN glibc_atanf
#define TLOG gfk_log
#define TLOG10 gfk_log10
#define TEXP gfk_exp
#define TPOW gfk_pow
__device__ float tatan2(float y,float x) { return (float)atan2((double)y,(double)x); }
#else
#define TSIN sinf
#define TCOS cosf
#define TTAN tanf
#define TATAN atanf
#define TLOG logf
#define TLOG10 log10f
#define TEXP expf
#define TPOW powf
__device__ float tatan2(float y,float x) { return atan2f(y,x); }
#endif
const float TRAD=0.01745329238474369f, TDEG=57.2957763671875f;
// h, re/dx, tl1, tl2, stdlon, polei, polej, cone, rsw, dlon,
// lat1, lon1, latinc, loninc. Same float fields as native proj_info.
__device__ float twrap(float x) { if(x>180.f)x-=360.f; if(x< -180.f)x+=360.f; return x; }
__device__ void tllij(int code,const float* p,float lat,float lon,float &i,float &j) {
    float h=p[0],re=p[1],tl=p[2],std=p[4];
    if(code==1) {
        float dl=twrap(lon-std),ct=TCOS(tl*TRAD);
        float rm=re*ct/p[7]*TPOW(TTAN((90.f*h-lat)*TRAD/2.f)/TTAN((90.f*h-tl)*TRAD/2.f),p[7]);
        float arg=p[7]*(dl*TRAD);
        i=h*(p[5]+h*rm*TSIN(arg)); j=h*(p[6]-rm*TCOS(arg));
    } else if(code==2) {
        float a=lat*TRAD,top=1.f+h*TSIN(tl*TRAD);
        float rm=re*TCOS(a)*top/(1.f+h*TSIN(a)),lo=(lon-(std+90.f))*TRAD;
        i=p[5]+rm*TCOS(lo);j=p[6]+h*rm*TSIN(lo);
    } else if(code==3) {
        i=1.f+twrap(lon-p[11])/(p[9]*TDEG);
        j=1.f+TLOG(TTAN(.5f*((lat+90.f)*TRAD)))/p[9]-p[8];
    } else { // trajmapproj's Cassini has lat0=90, lon0=0: native cylindrical branch.
        float dl=lon-p[11];if(dl<0)dl+=360.f;if(dl>360.f)dl-=360.f;
        i=dl/p[13];j=(lat-p[10])/p[12];
        if(i<=0.f)i+=360.f/p[13];if(i>360.f/p[13])i-=360.f/p[13];i+=1.f;j+=1.f;
    }
}
__device__ void tijll(int code,const float* p,float i,float j,float &lat,float &lon) {
    if(code==1) {
        float h=p[0],chi1=(90.f-h*p[2])*TRAD,chi2=(90.f-h*p[3])*TRAD;
        float xx=h*i-p[5],yy=p[6]-h*j,r2=xx*xx+yy*yy,r=sqrtf(r2)/p[1];
        if(r2==0.f){lat=h*90.f;lon=p[4];return;}
        lon=p[4]+TDEG*tatan2(h*xx,yy)/p[7];lon=fmodf(lon+360.f,360.f);
        float chi;
        if(chi1==chi2)chi=2.f*TATAN(TPOW(r/TTAN(chi1),1.f/p[7])*TTAN(chi1*.5f));
        else chi=2.f*TATAN(TPOW(r*p[7]/TSIN(chi1),1.f/p[7])*TTAN(chi1*.5f));
        lat=(90.f-chi*TDEG)*h;lon=twrap(lon);
    } else if(code==3) {
        lat=2.f*TATAN(TEXP(p[9]*(p[8]+j-1.f)))*TDEG-90.f;
        lon=twrap((i-1.f)*p[9]*TDEG+p[11]);
    } else {
        float iw=i-1.f,jw=j-1.f;
        if(iw<0.f)iw+=360.f/p[13];if(iw>=360.f/p[13])iw-=360.f/p[13];
        lat=jw*p[12]+p[10];lon=twrap(iw*p[13]+p[11]);
    }
}
extern "C" __global__ void trajectory_projection(float* p,int code,float dx,float dy,
    float lat1,float lon1,float tl1,float tl2,float std) {
    if(blockIdx.x || threadIdx.x)return;
    float h=tl1<0.f?-1.f:1.f,re=6370000.f/dx;
    p[0]=h;p[1]=re;p[2]=tl1;p[3]=tl2;p[4]=std;p[10]=lat1;p[11]=lon1;
    p[12]=dy*360.f/(2.f*6370000.f*3.1415927410125732f);
    p[13]=dx*360.f/(2.f*6370000.f*3.1415927410125732f);
    if(code==1) {
        float cone;
        if(fabsf(tl1-tl2)>.1f)cone=(TLOG10(TCOS(tl1*TRAD))-TLOG10(TCOS(tl2*TRAD)))/
            (TLOG10(TTAN((45.f-fabsf(tl1)/2.f)*TRAD))-TLOG10(TTAN((45.f-fabsf(tl2)/2.f)*TRAD)));
        else cone=TSIN(fabsf(tl1)*TRAD);
        float rsw=re*TCOS(tl1*TRAD)/cone*TPOW(TTAN((90.f*h-lat1)*TRAD/2.f)/TTAN((90.f*h-tl1)*TRAD/2.f),cone);
        float arg=cone*(twrap(lon1-std)*TRAD);
        p[7]=cone;p[8]=rsw;p[5]=h-h*rsw*TSIN(arg);p[6]=h+rsw*TCOS(arg);
    } else if(code==2) {
        float rsw=re*TCOS(lat1*TRAD)*(1.f+h*TSIN(tl1*TRAD))/(1.f+h*TSIN(lat1*TRAD));
        float alo=(lon1-(std+90.f))*TRAD;
        p[8]=rsw;p[5]=1.f-rsw*TCOS(alo);p[6]=1.f-h*rsw*TSIN(alo);
    } else if(code==3) {
        p[9]=dx/(6370000.f*TCOS(TRAD*tl1));p[8]=lat1==0.f?0.f:TLOG(TTAN(.5f*((lat1+90.f)*TRAD)))/p[9];
    }
}
// A column/face view. i,j,k supplied in native one-based coordinates.
struct TV {
 const float *q; int nx,ny;
 __device__ float at(int i,int k,int j)const{return q[((size_t)(k-1)*ny+j-1)*nx+i-1];}
};
struct TM {
 const float *q; int nx;
 __device__ float at(int i,int j)const{return q[(size_t)(j-1)*nx+i-1];}
};
struct TW {
 TV ru,rv,ww; TM mut,muu,muv,mt,muf,mvf; const float *c1h,*c2h,*c1f,*c2f;
 __device__ float at(int comp,int i,int k,int j)const {
  if(k<1)return 0.f;
  if(comp==0)return ru.at(i,k,j)/(c1h[k-1]*muu.at(i,j)+c2h[k-1])*muf.at(i,j);
  if(comp==1)return rv.at(i,k,j)/(c1h[k-1]*muv.at(i,j)+c2h[k-1])*mvf.at(i,j);
  return ww.at(i,k,j)/(c1f[k-1]*mut.at(i,j)+c2f[k-1])*mt.at(i,j);
 }
};
__device__ float tplane(TW w,int comp,int i,int j,int k,float x,float y,float z) {
 float a,b,c,d,da,db,dc,dd;
 if(comp!=2) {
  a=w.at(comp,i,k,j+1);b=w.at(comp,i,k,j);c=w.at(comp,i+1,k,j);d=w.at(comp,i+1,k,j+1);
  float xx=x+(comp==0?.5f:0.f),yy=y+(comp==1?.5f:0.f);
  da=fabsf((float(i+1)-xx)*(float(j)-yy));db=fabsf((float(i+1)-xx)*(float(j+1)-yy));
  dc=fabsf((float(i)-xx)*(float(j+1)-yy));dd=fabsf((float(i)-xx)*(float(j)-yy));
 } else {
  a=w.at(2,i,k+1,j);b=w.at(2,i,k,j);c=w.at(2,i+1,k,j);d=w.at(2,i+1,k+1,j);
  da=fabsf((float(i+1)-x)*(float(k)-(z+.5f)));db=fabsf((float(i+1)-x)*(float(k+1)-(z+.5f)));
  dc=fabsf((float(i)-x)*(float(k+1)-(z+.5f)));dd=fabsf((float(i)-x)*(float(k)-(z+.5f)));
 }
 return (a*da+b*db+c*dc+d*dd)/(da+db+dc+dd);
}
extern "C" __global__ void trajectory_advance(float *coords,int capacity,int count,
 const int* moving,int code,const float* proj,float dt,float rdx,float rdy,
 const float* ru,const float* rv,const float* ww,const float* mut,const float* muu,const float* muv,
 const float* mt,const float* muf,const float* mvf,const float* c1h,const float* c2h,
 const float* c1f,const float* c2f,const float* znw,int nx,int ny,int nz) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=count)return;
 float &x=coords[n],&y=coords[capacity+n],&z=coords[2*capacity+n],&lon=coords[3*capacity+n],&lat=coords[4*capacity+n];
 if(x==-9999.f)return;
 // Native has_proj_map omits spherical polar stereographic (code=2).
 bool has=code!=2;
 if(has)tllij(code,proj,lat,lon,x,y);
 int i=(int)floorf(x),j=(int)floorf(y),k=(int)floorf(z);
 if(i<1||i>nx||j<1||j>ny||k>nz) {x=y=z=lon=lat=-9999.f;return;}
 if(!moving[n])return;
 // module_em.F:2661-2674 only initializes local u/v/w through
 // ide-1,jde-1. Staggered interpolation in the final half-cell reads
 // an uninitialized native temporary. No deterministic word exists there.
 if(x>=float(nx)-.5f||y>=float(ny)-.5f||k<1||k>=nz) {x=y=z=lon=lat=__int_as_float(0x7fc00000);return;}
 TW w={{ru,nx+1,ny},{rv,nx,ny+1},{ww,nx,ny},{mut,nx},{muu,nx+1},{muv,nx},{mt,nx},{muf,nx+1},{mvf,nx},c1h,c2h,c1f,c2f};
 int iu=i+(x-float(i)>=.5f),jv=j+(y-float(j)>=.5f),kw=k+(z-float(k)>=.5f);
 float ul=tplane(w,0,iu,j,k,x,y,z),uu=tplane(w,0,iu,j,k+1,x,y,z);
 float u=uu*fabsf(float(k)-z)+ul*fabsf(float(k+1)-z);
 float vl=tplane(w,1,i,jv,k,x,y,z),vu=tplane(w,1,i,jv,k+1,x,y,z);
 float v=vu*fabsf(float(k)-z)+vl*fabsf(float(k+1)-z);
 float wl=tplane(w,2,i,j,kw,x,y,z),wu=tplane(w,2,i,j+1,kw,x,y,z);
 float omega=wu*fabsf(float(j)-y)+wl*fabsf(float(j+1)-y);
 float eta=znw[kw]*fabsf(z+.5f-float(kw))+znw[kw-1]*fabsf(z+.5f-float(kw+1));
 float rdxx=rdx*w.mt.at(i,j),rdyy=rdy*w.mt.at(i,j);
 x=x+(u*dt)*rdxx;y=y+(v*dt)*rdyy;eta=eta+omega*dt;
 int kt=0;for(int l=1;l<=nz;l++)if(eta<=znw[l-1]&&eta>znw[l])kt=l;
 if(kt)z=(float(kt)*fabsf(eta-znw[kt])+float(kt+1)*fabsf(eta-znw[kt-1]))/(znw[kt-1]-znw[kt])-.5f;
 if(has)tijll(code,proj,x,y,lat,lon);
}
extern "C" __global__ void trajectory_initialize(float* coords,int capacity,int count,
 const float* input,const int* agl,int code,const float* proj,const float* ph,const float* phb,
 int base_column,int nx,int ny,int nz) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=capacity)return;
 for(int c=0;c<5;c++)coords[c*capacity+n]=-9999.f;
 if(n>=count)return;
 float lon=input[n*3],lat=input[n*3+1],lev=input[n*3+2],x,y;
 tllij(code,proj,lat,lon,x,y);
 if(x<1.f||x>nx||y<1.f||y>ny)return;
 int i=(int)roundf(x),j=(int)roundf(y);TV phi={ph,nx,ny},base={phb,nx,ny};
 float bot=(phi.at(i,1,j)+(base_column?phb[0]:base.at(i,1,j)))/9.81f;
 float top=(phi.at(i,nz,j)+(base_column?phb[nz-1]:base.at(i,nz,j)))/9.81f;
 if(agl[n])lev+=bot;if(lev<bot||lev>top)return;
 float zi=-9999.f,low=bot;
 for(int k=2;k<=nz;k++) {
  float high=(phi.at(i,k,j)+(base_column?phb[k-1]:base.at(i,k,j)))/9.81f;
  if(lev<=high){zi=float(k-1)+(lev-low)/(high-low);break;}low=high;
 }
 coords[n]=x;coords[capacity+n]=y;coords[2*capacity+n]=zi;
 coords[3*capacity+n]=lon;coords[4*capacity+n]=lat;
}
// Generic native Registry field sampler: descriptors select direct fields,
// staggering and the three derived dynamic variables. One thread per
// particle/variable, never a host gather of a weather field.
struct TF { const float *a,*b,*p; int kind,nx,ny,column; };
__device__ float tfvalue(TF f,int i,int k,int j) {
 TV a={f.a,f.nx,f.ny},b={f.b,f.nx,f.ny};
 float q=a.at(i,f.kind==7?1:k,j);
 if(f.kind==1)return .5f*(q+a.at(i+1,k,j));
 if(f.kind==2)return .5f*(q+a.at(i,k,j+1));
 if(f.kind==3)return .5f*(q+a.at(i,k+1,j));
 if(f.kind==4)return q+b.at(i,k,j);
 if(f.kind==5){TV p={f.p,f.nx,f.ny};return (q+300.f)*TPOW((b.at(i,k,j)+p.at(i,k,j))*1.e-5f,287.f/1004.5f);}
 if(f.kind==8){TV p={f.p,f.nx,f.ny};return (q+300.f)*TPOW(p.at(i,k,j)*1.e-5f,287.f/1004.5f);}
 if(f.kind==9)return q-(f.column?f.b[k-1]:b.at(i,k,j));
 if(f.kind==6) {
  float b0=f.column?f.b[k-1]:b.at(i,k,j),b1=f.column?f.b[k]:b.at(i,k+1,j);
  return .5f*(q+a.at(i,k+1,j)+b0+b1)*(1.f/9.81f);
 }
 return q;
}
extern "C" __global__ void trajectory_sample(const float* coords,int capacity,int count,
 const unsigned long long* ptrs,const int* desc,const int* requested,const int* active,
 float* out,int slot,int variables,int nx,int ny,int nz) {
 int flat=blockIdx.x*blockDim.x+threadIdx.x;if(flat>=count*(variables+5))return;
 int c=flat/count,n=flat%count;
 size_t target=((size_t)slot*(variables+5)+c)*count+n;
 float x=coords[n],y=coords[capacity+n],z=coords[2*capacity+n];
 // Native always records coordinates, including inactive time intervals.
 if(c<5){out[target]=coords[c*capacity+n];return;}
 int v=c-5;
 if(!active[n]||!requested[v*count+n]||x<1.f||x>nx||y<1.f||y>ny||z<1.f||z>nz){out[target]=-9999.f;return;}
 const int* d=desc+v*4;
 TF f={(const float*)ptrs[v*3],(const float*)ptrs[v*3+1],(const float*)ptrs[v*3+2],d[0],d[1],d[2],d[3]};
 int i=(int)x,j=(int)y,k=(int)z;
 if(i>=nx||j>=ny||k>=nz) {out[target]=__int_as_float(0x7fc00000);return;}
 // Native reads its halo for a mass-grid point in the last cell. The
 // bindings carry that neighbor whenever the selected field is staggered.
 float dx=x-floorf(x),dy=y-floorf(y),dz=z-floorf(z),ox=1.f-dx,oy=1.f-dy;
 float lo=ox*oy*tfvalue(f,i,k,j)+oy*dx*tfvalue(f,i+1,k,j)+dy*ox*tfvalue(f,i,k,j+1)+dx*dy*tfvalue(f,i+1,k,j+1);
 float hi=ox*oy*tfvalue(f,i,k+1,j)+oy*dx*tfvalue(f,i+1,k+1,j)+dy*ox*tfvalue(f,i,k+1,j+1)+dx*dy*tfvalue(f,i+1,k+1,j+1);
 out[target]=dz*hi+(1.f-dz)*lo;
}
// Activation uses native next_time=current_time+INT(dt), including time zero.
extern "C" __global__ void trajectory_clock(const double* intervals,double seconds,
 int* moving,int* active,int count) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=count)return;
 bool enabled=seconds>=intervals[3*n]&&seconds<=intervals[3*n+1];
 active[n]=enabled;moving[n]=enabled&&intervals[3*n+2]!=0.;
}
// external/RSL_LITE/feedback_domain_em_part2.F:102-123. Longitude and
// latitude update sequentially; i/j remain in each domain's coordinates.
extern "C" __global__ void trajectory_feedback(float* parent,float* child,int capacity) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=capacity)return;
 if(child[3*capacity+n]==-9999.f){child[3*capacity+n]=parent[3*capacity+n];child[2*capacity+n]=parent[2*capacity+n];}
 else {parent[3*capacity+n]=child[3*capacity+n];parent[2*capacity+n]=child[2*capacity+n];}
 if(child[4*capacity+n]==-9999.f){child[4*capacity+n]=parent[4*capacity+n];child[2*capacity+n]=parent[2*capacity+n];}
 else {parent[4*capacity+n]=child[4*capacity+n];parent[2*capacity+n]=child[2*capacity+n];}
}
// Average the already-rounded total post-acoustic mass, matching MUUS/MUVS.
extern "C" __global__ void trajectory_face_mass(const float* mut,float* muu,float* muv,int nx,int ny) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;
 if(n<(nx+1)*ny){int i=n%(nx+1),j=n/(nx+1);muu[n]=.5f*(mut[j*nx+(i?i-1:0)]+mut[j*nx+(i<nx?i:nx-1)]);}
 if(n<nx*(ny+1)){int i=n%nx,j=n/nx;muv[n]=.5f*(mut[(j?j-1:0)*nx+i]+mut[(j<ny?j:ny-1)*nx+i]);}
}
