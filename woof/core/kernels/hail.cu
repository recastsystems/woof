// WRF public-domain diagnostics. Source and notice in tools/hail_wrf461.
// Arrays are mass [k,j,i], except w and geopotential on vertical faces.
extern "C" __global__ void hail_prepare(
    const float* theta,const float* p,const float* alt,const float* qv,
    const float* phb,const float* php,const float* w,
    float* t,float* z,float* rho,float* mask,
    int nx,int ny,int nz,int base_profile) {
    int c=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;
    if(c>=n) return;
    float strong=0;
    for(int k=0;k<nz;k++) {
        int i=k*n+c;
        t[i]=theta[i]*powf(__fdiv_rn(p[i],100000.f),287.f/1004.f);
        rho[i]=(1.f+qv[i])/alt[i];
        float za=__fdiv_rn(phb[base_profile?k:i]+php[i],9.81f);
        float zb=__fdiv_rn(phb[base_profile?k+1:i+n]+php[i+n],9.81f);
        z[i]=.5f*(za+zb);
        if(w[i]>=10.f) strong=1;
    }
    if(w[nz*n+c]>=10.f) strong=1;
    mask[c]=strong;
}
extern "C" __global__ void hail_duration(
    const float* mask,const float* oldmask,const float* olddur,float* dur,
    int nx,int ny,float dt,int periodic_x,int periodic_y) {
    int c=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;
    if(c>=n) return;
    int x=c%nx,y=c/nx;
    dur[c]=0;
    if((!periodic_x&&(x==0||x==nx-1))||(!periodic_y&&(y==0||y==ny-1))) return;
    float seen=mask[c],longest=0;
    for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
        int xx=(x+dx+nx)%nx,yy=(y+dy+ny)%ny,j=yy*nx+xx;
        seen=fmaxf(seen,oldmask[j]); longest=fmaxf(longest,olddur[j]);
    }
    if(seen==1.f) dur[c]=longest+dt;
}
// One column of HAILCAST; Column is the generation (hailcast_wrf.cuh = WRF
// v4.6.1, hailcast_wrf39.cuh = operational HRRR v4.1.21, gpuwm.core.hail).
template<class Column> __device__ void hailcast_column(
    float* t,float* z,float* p,float* rho,float* qv,
    float* qi,float* qc,float* qr,float* qs,float* qg,float* w,
    const float* ht,const float* dur,float* e1,float* e2,float* e3,float* e4,float* e5,
    float* maximum,float* mean,float* sd,float* runmax,
    int nx,int ny,int nz) {
    int c=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;
    if(c>=n||dur[c]<=900.f) return;
    HailArray<float> ta{t+c,n},za{z+c,n},pa{p+c,n},ra{rho+c,n},
        va{qv+c,n},ia{qi+c,n},ca{qc+c,n},raina{qr+c,n},
        sa{qs+c,n},ga{qg+c,n},wa{w+c,n};
    float h=ht[c],duration=dur[c],d[5]={0,0,0,0,0};
    Column column;
    column.hailstone_driver(ta,za,h,pa,ra,va,ia,ca,raina,sa,ga,wa,
        duration,nz,d[0],d[1],d[2],d[3],d[4]);
    float* embryos[5]={e1,e2,e3,e4,e5};
    // Invalid integration must remain visible to history/verification. Never
    // turn a failed stone trajectory into a plausible no-hail observation.
    if(!isfinite(runmax[c])) return;
    for(int j=0;j<5;j++) if(!isfinite(d[j])) {
        maximum[c]=mean[c]=sd[c]=runmax[c]=nanf("");
        embryos[j][c]=nanf("");return;
    }
    float m=0,mx=0,s=0;
    for(int j=0;j<5;j++) {
        float value=fmaxf(d[j],0.f);
        embryos[j][c]=fmaxf(embryos[j][c],value);
        m+=embryos[j][c]*.2f; mx=fmaxf(mx,embryos[j][c]);
    }
    for(int j=0;j<5;j++) { float e=embryos[j][c]-m; s+=e*e; }
    maximum[c]=mx; mean[c]=m; sd[c]=sqrtf(s*.25f); runmax[c]=fmaxf(runmax[c],mx);
}
extern "C" __global__ void hailcast_columns(
    float* t,float* z,float* p,float* rho,float* qv,
    float* qi,float* qc,float* qr,float* qs,float* qg,float* w,
    const float* ht,const float* dur,float* e1,float* e2,float* e3,float* e4,float* e5,
    float* maximum,float* mean,float* sd,float* runmax,
    int nx,int ny,int nz) {
    hailcast_column<HailColumn<256>>(t,z,p,rho,qv,qi,qc,qr,qs,qg,w,ht,dur,
        e1,e2,e3,e4,e5,maximum,mean,sd,runmax,nx,ny,nz);
}
extern "C" __global__ void hailcast_columns_wrf39(
    float* t,float* z,float* p,float* rho,float* qv,
    float* qi,float* qc,float* qr,float* qs,float* qg,float* w,
    const float* ht,const float* dur,float* e1,float* e2,float* e3,float* e4,float* e5,
    float* maximum,float* mean,float* sd,float* runmax,
    int nx,int ny,int nz) {
    hailcast_column<HailColumnWrf39<256>>(t,z,p,rho,qv,qi,qc,qr,qs,qg,w,ht,dur,
        e1,e2,e3,e4,e5,maximum,mean,sd,runmax,nx,ny,nz);
}
// N*Q(alpha+1,lambda*D) = 0.0005 m^-3, WRF HAILMAXD criterion.
// Closed forms cover the default exponential graupel and alpha=1 hail.
__device__ double hail_tail(double a,double x) {
    if(x<=0) return 1;
    if(a==1) return exp(-x);
    if(a==2) return exp(-x)*(1+x);
    double logterm=a*log(x)-x-lgamma(a);
    if(x<a+1) {
        double term=1/a,sum=term;
        for(int j=1;j<=160;j++) { term*=x/(a+j);sum+=term;if(fabs(term)<fabs(sum)*1.e-13) break; }
        return fmax(0.,1-sum*exp(logterm));
    }
    double b=x+1-a,c=1.e300,d=1/b,h=d;
    for(int j=1;j<=160;j++) {
        double an=-j*(j-a); b+=2;
        d=an*d+b;if(fabs(d)<1.e-300)d=1.e-300;
        c=b+an/c;if(fabs(c)<1.e-300)c=1.e-300;
        d=1/d;double delta=d*c;h*=delta;if(fabs(delta-1)<1.e-13)break;
    }
    return fmax(0.,h*exp(logterm));
}
__device__ float hail_psd_size(double mass,double number,double density,double alpha) {
    if(!(mass>1.e-6&&number>0.0005&&density>0&&alpha>-1)) return 0;
    double lambda=cbrt(3.141592653589793/6*density*(alpha+1)*(alpha+2)*(alpha+3)*number/mass);
    double target=.0005/number,lo=0,hi=fmax(64.,alpha*4+64);
    if(alpha==0) return (float)(-log(target)/lambda);
    for(int j=0;j<40;j++) {
        double mid=.5*(lo+hi);
        if(hail_tail(alpha+1,mid)>target)lo=mid;else hi=mid;
    }
    return (float)(.5*(lo+hi)/lambda);
}
extern "C" __global__ void hail_psd(
    const float* rho,const float* qg,const float* ng,const float* vg,
    const float* qh,const float* nh,const float* vh,
    float* lowest,float* column,float* runlowest,float* runcolumn,
    int nx,int ny,int nz,int number_g,int volume_g,int number_h,int volume_h,
    int thompson,float density_g,float density_h,float alpha_g,float alpha_h,float intercept,float intercept_h) {
    int c=blockDim.x*blockIdx.x+threadIdx.x,n=nx*ny;
    if(c>=n) return;
    float mx=0,low=0;
    for(int k=0;k<nz;k++) {
        int i=k*n+c;
        double m=fmaxf(qg[i],0.f)*rho[i];
        double density=volume_g&&vg[i]>0?qg[i]/vg[i]:density_g;
        double number=number_g?fmaxf(ng[i],0.f)*rho[i]:0;
        if(!number_g&&m>1.e-6) {
            double n0=intercept;
            if(thompson) {
                float lg=log10f(fmaxf(1.e-9f,(float)m));
                float exponent=fmaxf(2.f,fminf(6.f,3.f+2.f/7.f*(lg+8.f)));
                n0=powf(10.f,exponent);
            }
            double lambda=pow(n0*3.141592653589793*density/m,.25);
            number=n0/lambda;
        }
        float size=hail_psd_size(m,number,density,alpha_g);
        if(number_h || intercept_h>0) {
            double mh=fmaxf(qh[i],0.f)*rho[i];
            double dh=volume_h&&vh[i]>0?qh[i]/vh[i]:density_h;
            double number=number_h?fmaxf(nh[i],0.f)*rho[i]:0;
            if(!number_h && mh>1.e-6) {
                double lambda=pow(intercept_h*3.141592653589793*dh/mh,.25);
                number=intercept_h/lambda;
            }
            size=fmaxf(size,hail_psd_size(mh,number,dh,alpha_h));
        }
        mx=fmaxf(mx,size);
        if(k==0)low=size;
    }
    lowest[c]=fmaxf(lowest[c],low);
    column[c]=fmaxf(column[c],mx);
    runlowest[c]=fmaxf(runlowest[c],low);runcolumn[c]=fmaxf(runcolumn[c],mx);
}
