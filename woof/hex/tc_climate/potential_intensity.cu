// CUDA adaptation of the MIT-licensed tcpyPI CAPE/PI algorithm.
// Copyright (c) 2020 Daniel Gilford. See adjacent LICENSE.tcpypi.
// Gilford (2021), doi:10.5194/gmd-14-2351-2021; Bister and Emanuel (2002),
// doi:10.1029/2001JD000776. Reversible ascent, dissipative heating, Ck/Cd=.9,
// surface reduction=.8. No ice thermodynamics, matching the cited algorithm.
// The source profile is [level,cell], pressure Pa, T K, vapor mixing ratio kg/kg.
#ifndef __CUDACC_RTC__
#include <math.h>
#else
#define NAN nanf("")
#endif
constexpr double RD=287.04, RV=461.5, EPS=RD/RV, CPD=1005.7, CL=2500.;
__device__ double es(double t) { double c=t-273.15; return 6.112*exp(17.67*c/(243.5+c)); }
__device__ double latent(double t) { return 2.501e6-630.*(t-273.15); }
__device__ double trho(double t,double rt,double r) { return t*(1.+r/EPS)/(1.+rt); }
__device__ double entropy(double t,double r,double p) {
    double e=r*p/(EPS+r), rh=fmin(1.,e/es(t));
    return (CPD+r*CL)*log(t)-RD*log(p-e)+latent(t)*r/t-r*RV*log(rh);
}
struct Cape { double value, outflow; int flag; };
__device__ Cape cape(double tp,double rp,double pp,int levels,int stride,int cell,
    const float* pressure,const float* temperature,const float* vapor) {
    Cape out={0,temperature[cell],1};
    if(rp<1e-6 || tp<200 || !isfinite(rp) || !isfinite(tp)) {out.flag=0;return out;}
    double rh=fmin(1.,rp*pp/(EPS+rp)/es(tp));
    if(rh<=0) {out.flag=0;return out;}
    double s=entropy(tp,rp,pp), plcl=pp*pow(rh,tp/(1669.-122.*rh-tp));
    // One thread per sounding, bounded by the launch validation.
    double buoy[128];
    int inb=0;
    for(int k=0;k<levels;++k) {
        int i=k*stride+cell;
        double p=pressure[i]*.01,t=temperature[i],r=vapor[i],tg,rg;
        if(p>=plcl) {tg=tp*pow(p/pp,RD/CPD); rg=rp;}
        else {
            double next=t; bool converged=false;
            for(int it=1;it<=501;++it) {
                tg=next; double e=es(tg);
                if(!isfinite(e) || e>p-1 || tg<=0) {out.flag=2;return out;}
                rg=EPS*e/(p-e);
                double lv=latent(tg);
                double slope=(CPD+rp*CL+lv*lv*rg/(RV*tg*tg))/tg;
                double sg=(CPD+rp*CL)*log(tg)-RD*log(p-e)+lv*rg/tg;
                next=tg+(it<3?.3:1.)*(s-sg)/slope;
                if(!isfinite(next)) {out.flag=2;return out;}
                if(fabs(next-tg)<=.001) {converged=true;break;}
            }
            if(!converged) {out.flag=2;return out;}
        }
        buoy[k]=trho(tg,p>=plcl?rg:rp,rg)-trho(t,r,r);
        if(k>0 && buoy[k]>0) inb=k;
    }
    if(!inb) return out;
    double area=0;
    for(int k=1;k<=inb;++k) {
        double p0=pressure[(k-1)*stride+cell]*.01,p1=pressure[k*stride+cell]*.01;
        area+=RD*(buoy[k]+buoy[k-1])*(p0-p1)/(p0+p1);
    }
    double p0=pressure[cell]*.01;
    area+=RD*(pp-p0)/(pp+p0)*buoy[0];
    out.outflow=temperature[inb*stride+cell];
    if(inb<levels-1) {
        double p=pressure[inb*stride+cell]*.01,pnext=pressure[(inb+1)*stride+cell]*.01;
        double pinb=(pnext*buoy[inb]-p*buoy[inb+1])/(buoy[inb]-buoy[inb+1]);
        area+=RD*buoy[inb]*(p-pinb)/(p+pinb);
        out.outflow=(temperature[inb*stride+cell]*(pinb-pnext)+
                      temperature[(inb+1)*stride+cell]*(p-pinb))/(p-pnext);
    }
    out.value=fmax(0.,area); return out;
}
extern "C" __global__ void potential_intensity(int n,int nz,
    const float* pressure,const float* temperature,const float* vapor,
    const float* sst,const float* slp,const int* ocean,float ckcd,float reduction,
    float ptop,float* pi,int* status) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=n) return;
    pi[c]=NAN; status[c]=0;
    if(!ocean[c]) return;
    double ts=sst[c],msl=slp[c]*.01;
    if(!isfinite(ts) || ts<=278.15 || ts>373.15 || !isfinite(msl) || msl<=0) return;
    int levels=0; double mind=1e30;
    for(int k=0;k<nz;++k) {
        int i=k*n+c;
        if(!isfinite(pressure[i]) || !isfinite(temperature[i]) || !isfinite(vapor[i]) ||
           temperature[i]<=100 || temperature[i]>373.15 || vapor[i]<0 || pressure[i]<=0) return;
        if(k && pressure[i]>=pressure[i-n]) return;
        double d=fabs(pressure[i]*.01-ptop);
        if(d<mind) {mind=d;levels=k;}
    }
    if(levels<3 || levels>128 || pressure[(nz-1)*n+c]>ptop*100) return;
    double t=temperature[c],r=vapor[c];
    Cape env=cape(t,r,pressure[c]*.01,levels,n,c,pressure,temperature,vapor);
    if(env.flag!=1) {status[c]=env.flag;return;}
    double pm=970,ratio=0,fac=0;
    for(int it=0;it<=200;++it) {
        double pp=fmin(pm,1000.);
        double rp=EPS*r*msl/(pp*(EPS+r)-r*msl);
        Cape a=cape(t,rp,pp,levels,n,c,pressure,temperature,vapor);
        double rs=EPS*es(ts)/(pp-es(ts));
        Cape b=cape(ts,rs,pp,levels,n,c,pressure,temperature,vapor);
        if(a.flag!=1 || b.flag!=1) {status[c]=2;return;}
        ratio=ts/b.outflow; fac=fmax(0.,b.value-a.value);
        double tv=.5*(trho(t,r,r)+trho(ts,rs,rs));
        double cat=fmax(0.,a.value-env.value+.5*ckcd*ratio*(b.value-a.value));
        double next=msl*exp(-cat/(RD*tv));
        if(!isfinite(next) || next<400 || it==200) {status[c]=2;return;}
        if(fabs(next-pm)<=.5) {pi[c]=reduction*sqrt(ckcd*ratio*fac);status[c]=1;return;}
        pm=next;
    }
}
// Least-squares spherical wind curl. Uses true angular differences and the
// metric term u*tan(latitude)/R. This diagnostic is not a native-vorticity
// bitwise claim. Missing neighbors propagate a missing column.
extern "C" __global__ void vorticity850(int n,int degree,const int* graph,
    const int* counts,const float* lat,const float* lon,const float* u,
    const float* v,float* eta) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=n) return;
    double xx=0,xy=0,yy=0,xv=0,yv=0,xu=0,yu=0;
    eta[c]=NAN;
    if(!isfinite(u[c]) || !isfinite(v[c]) || fabs(lat[c])>1.0471975512) return;
    for(int k=0;k<counts[c];++k) {
        int j=graph[c*degree+k];
        if(j<0 || j>=n || !isfinite(u[j]) || !isfinite(v[j])) return;
        double dl=remainder((double)lon[j]-lon[c],6.283185307179586);
        double x=6371229.*cos(lat[c])*dl,y=6371229.*(lat[j]-lat[c]);
        double w=1./fmax(1.,x*x+y*y);
        xx+=w*x*x;xy+=w*x*y;yy+=w*y*y;
        xv+=w*x*(v[j]-v[c]);yv+=w*y*(v[j]-v[c]);
        xu+=w*x*(u[j]-u[c]);yu+=w*y*(u[j]-u[c]);
    }
    double det=xx*yy-xy*xy;
    if(det<=1e-12) return;
    eta[c]=(yy*xv-xy*yv-xx*yu+xy*xu)/det+u[c]*tan(lat[c])/6371229.
           +2.*7.292115e-5*sin(lat[c]);
}
