# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""MPAS-A v8.4.2 radiation support, independent of the WOOF driver.

Device arrays use float32 C-contiguous (column, bottom-up level) storage.
Ozone monthly tables use (column, ozone level, month); radii are meters.
Ozone input records are authenticated and decoded by the shared framework
reader. The remaining bindings allocate buffers and launch CUDA. No default
suite or WOOF implementation is changed by importing this module.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np



def merge_mynn_bl_clouds(qc, qi, cldfra, *, qc_bl, qi_bl, cldfra_bl,
                         itimestep, xp):
    """WRF v4.6.1 module_radiation_driver.F:1403-1429, on radiation copies.

    The modern MYNN provider supplies grid-mean phase-separated condensate.
    Do not multiply by fraction again or add it over resolved cloud. The
    thresholds are Fortran default REAL constants, including strict bounds.
    Carried MYNN fraction replaces the diagnosis after the first timestep.
    Vertical overlap is separately controlled by the existing RRTMG setting.
    """
    cloudy = cldfra_bl > xp.float32(0.001)
    if itimestep != 1:
        cldfra[...] = cldfra_bl
    qc[...] = xp.where((qc < xp.float32(1.e-6)) & cloudy, qc + qc_bl, qc)
    qi[...] = xp.where((qi < xp.float32(1.e-8)) & cloudy, qi + qi_bl, qi)


CLOUD_SCHEMES = ("cld_incidence", "cld_fraction", "cld_fraction_thompson")
OVERLAP = {"none": 0, "random": 1, "maximum_random": 2, "maximum": 3,
           "exponential": 4, "exponential_random": 5}
RADIUS_SCHEMES = ("mp_thompson", "mp_thompson_aerosols", "mp_wsm6")


@dataclass(frozen=True)
class RadiationOptions:
    cldovrlp: int
    idcor: int
    o3input: int
    has_reqc: int
    has_reqi: int
    has_reqs: int


def radiation_options(config: Mapping[str, object]) -> RadiationOptions:
    """mpas_atmphys_driver_radiation_lw.F:845-863; SW same mapping.

    Radius presence: mpas_atmphys_manager.F:747-759. Defaults belong to
    the namelist resolver; all relevant keys must be explicitly resolved.
    """
    overlap = str(config["config_radt_cld_overlap"]).strip()
    dcorr = str(config["config_radt_cld_dcorrlen"]).strip()
    # LW:852-857 / SW:905-910 leave unknown strings unset. The published
    # strict executable reproduces 'none' in the repeated two-host, 1/2-rank,
    # full-RRTMG witness receipts/native-overlap-policy-witness-v1.json.
    # Maximum-random controls change 222 fields, so this witness is sensitive
    # to overlap. Retain these words inside the comparison, without an abort.
    cldovrlp = OVERLAP.get(overlap, 0)
    # LW:858-862 / SW:911-916 initialize idcor to zero before comparing
    # strings. An unrecognized decorrelation leaves that defined zero.
    has_re = int(bool(config["config_microp_re"]) and
                 str(config["config_microp_scheme"]).strip() in RADIUS_SCHEMES and
                 str(config["config_radt_lw_scheme"]).strip() == "rrtmg_lw" and
                 str(config["config_radt_sw_scheme"]).strip() == "rrtmg_sw")
    return RadiationOptions(cldovrlp, int(cldovrlp >= 4 and
                            dcorr == "latitude_varying"),
                            2 if config["config_o3climatology"] else 0,
                            has_re, has_re, has_re)


# Arithmetic pins come from the shared glibc device header. --fmad=false
# protects all ordinary expressions; FADD/FMUL/FDIV additionally pin steps.
CUDA_SOURCE = r'''
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:156-217 (RSLF / RSIF)
__device__ float sat(float p, float t, bool ice) {
    const float water[9]={.611583699E03f,.444606896E02f,.143177157E01f,
        .264224321E-1f,.299291081E-3f,.203154182E-5f,.702620698E-8f,
        .379534310E-11f,-.321582393E-13f};
    const float frozen[9]={.609868993E03f,.499320233E02f,.184672631E01f,
        .402737184E-1f,.565392987E-3f,.521693933E-5f,.307839583E-7f,
        .105785160E-9f,.161444444E-12f};
    const float *c = ice ? frozen : water;
    float x=fmaxf(-80.f,FSUB(t,273.16f)), e=c[8];
    for(int n=7;n>=0;--n) e=FADD(c[n],FMUL(x,e));
    e=fminf(e,FMUL(p,.15f));
    return FDIV(FMUL(.622f,e),FSUB(p,e));
}

// MPAS mpas_atmphys_driver_cloudiness.F:64-143
extern "C" __global__ void cloud_from(int nc,int nz,float len,
    const float *density,const float *xland,const float *qv,const float *qc,
    const float *qi,const float *qs,float *dx,float *land,float *v,float *c,
    float *ice,float *snow,float *cf) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc*nz) return;
    v[i]=qv[i]; c[i]=qc[i]; ice[i]=qi[i]; snow[i]=qs[i]; cf[i]=0.f;
    if(i%nz==0) {
        int col=i/nz;
        dx[col]=FMUL(FDIV(len,gfk_pow(density[col],.25f)),.001f);
        land[col]=xland[col];
    }
}

// MPAS mpas_atmphys_driver_cloudiness.F:237-370 (both non-Thompson schemes)
extern "C" __global__ void cloud_simple(int n,int scheme,int fqc,int fqi,
    const float *v,const float *c,const float *ice,const float *snow,
    const float *t,const float *p,float *cf) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float out=0.f;
    if(scheme==0) {
        if(fqc && fqi) out=FADD(c[i],ice[i])>1.e-6f ? 1.f:0.f;
        else if(fqc) out=c[i]>1.e-6f ? 1.f:0.f;
    } else {
        float esw=FMUL(1000.f*.61078f,gfk_exp(FDIV(FMUL(17.2693882f,
            FSUB(t[i],273.15f)),FSUB(t[i],35.86f))));
        float esi=FMUL(1000.f*.61078f,gfk_exp(FDIV(FMUL(21.8745584f,
            FSUB(t[i],273.15f)),FSUB(t[i],7.66f))));
        const float ep2=287.f/461.6f;
        float sw=FDIV(FMUL(ep2,esw),FSUB(p[i],esw));
        float si=FDIV(FMUL(ep2,esi),FSUB(p[i],esi));
        float qcld=FADD(FADD(c[i],ice[i]),snow[i]);
        float w=qcld<1.e-12f ? 0.f:FDIV(FADD(ice[i],snow[i]),qcld);
        float qvs=FADD(FMUL(FSUB(1.f,w),sw),FMUL(w,si));
        float rh=FDIV(v[i],qvs);
        if(qcld<1.e-12f) out=0.f;
        else if(rh>=1.f) out=1.f;
        else {
            float subsat=fmaxf(1.e-10f,FSUB(qvs,v[i]));
            float denom=gfk_pow(subsat,.49f);
            float arg=fmaxf(-6.9f,FDIV(FMUL(-100.f,qcld),denom));
            rh=fmaxf(1.e-10f,rh);
            out=FMUL(gfk_pow(rh,.25f),FSUB(1.f,gfk_exp(arg)));
            if(out<.01f) out=0.f;
        }
    }
    cf[i]=out;
}

// MPAS module_mp_thompson_cldfra3.F:384-469 (Ice and H2O adjustments)
__device__ void adjust_layer(bool ice,float *cf,float *qc,float *qi,
    const float *qs,const float *qvs,const float *t,const float *rho,
    const float *dz,int k1,int k2) {
    float tdz=0.f;
    for(int k=k1;k<=k2;++k) tdz=FADD(tdz,dz[k]);
    float max_wc=fabsf(FSUB(qvs[k2-1],qvs[k1]));
    float exists=0.f;
    for(int k=k1;k<=k2;++k) {
        float q=ice ? FADD(qi[k],qs[k]):qc[k];
        exists=FADD(exists,FMUL(FMUL(q,rho[k]),dz[k]));
    }
    if(exists>1.f) return;
    float this_dz=0.f;
    for(int k=k1;k<=k2;++k) {
        this_dz=FADD(this_dz,k==k1 ? FMUL(.5f,dz[k]):dz[k]);
        float this_wc=FDIV(FMUL(max_wc,this_dz),tdz);
        float wc=fmaxf(1.e-6f,FMUL(this_wc,.5f));
        if(ice) {
            if(cf[k]>.01f && cf[k]<.99f && t[k]>=203.16f)
                qi[k]=FADD(qi[k],FMUL(FMUL(.1f,cf[k]),wc));
            else if(qi[k]<1.e-5f && cf[k]>=.99f && t[k]>=203.16f)
                qi[k]=FADD(qi[k],FMUL(.01f,wc));
        } else {
            if(cf[k]>.01f && cf[k]<.99f && t[k]<298.16f && t[k]>=253.16f)
                qc[k]=FADD(qc[k],FMUL(FMUL(cf[k],cf[k]),wc));
            else if(cf[k]>=.99f && qc[k]<1.e-5f && t[k]<298.16f && t[k]>=253.16f)
                qc[k]=FADD(qc[k],FMUL(.1f,wc));
        }
    }
}

// MPAS module_mp_thompson_cldfra3.F:476-518
__device__ void adjust_final(float *cf,float *qc,float *qi,
    const float *rho,const float *dz,int tropo) {
    float lwp=0.f,iwp=0.f;
    for(int k=0;k<=tropo;++k) if(cf[k]>.01f && cf[k]<.99f)
        lwp=FADD(lwp,FMUL(FMUL(qc[k],rho[k]),dz[k]));
    for(int k=0;k<=tropo;++k) if(cf[k]>.01f && cf[k]<.99f)
        iwp=FADD(iwp,FMUL(FMUL(qi[k],rho[k]),dz[k]));
    if(lwp>1.f) {
        float xfac=FDIV(1.f,lwp);
        for(int k=0;k<=tropo;++k) if(cf[k]>.01f && cf[k]<.99f)
            qc[k]=FMUL(qc[k],xfac);
    }
    if(iwp>1.f) {
        float xfac=FDIV(1.f,iwp);
        for(int k=0;k<=tropo;++k) if(cf[k]>.01f && cf[k]<.99f)
            qi[k]=FMUL(qi[k],xfac);
    }
}

// MPAS module_mp_thompson_cldfra3.F:221-227
__device__ float pressure_height(float p) {
    return FMUL(44307.692f,FSUB(1.f,gfk_pow(FDIV(p,101325.f),.190f)));
}
// MPAS module_mp_thompson_cldfra3.F:191-380
__device__ void find_layers(float *qvs,float *cf,const float *t,
    const float *p,const float *rho,float *qc,float *qi,const float *qs,
    float *theta,float *dz,int nz) {
    int m12=-1,m40=-1;
    for(int k=nz-1;k>=0;--k) {
        theta[k]=FMUL(t[k],gfk_pow(FDIV(100000.f,p[k]),287.05f/1004.f));
        if(FSUB(t[k],273.16f)>-40.f) m40=max(m40,k);
        if(FSUB(t[k],273.16f)>-12.f) m12=max(m12,k);
    }
    m12=max(m12,0); m40=max(m40,0);
    float z2=pressure_height(p[nz-1]);
    for(int k=nz-2;k>=0;--k) {
        float z1=pressure_height(p[k]); dz[k+1]=FSUB(z2,z1); z2=z1;
    }
    dz[0]=dz[1];
    int k;
    for(k=nz-4;k>=0;--k) {
        float h1=pressure_height(p[k]),h2=pressure_height(p[k+2]);
        if(FDIV(FSUB(theta[k+2],theta[k]),FSUB(h2,h1))<10.f/1500.f
           && h1<19000.f && h1>4000.f) break;
    }
    int tropo=max(2,k+2);
    for(k=tropo+1;k<nz;++k) if(cf[k]>0.f && cf[k]<.999f) cf[k]=0.f;
    for(k=2;k<=m12;++k)
        if(FSUB(theta[k],theta[k-1])>FMUL(.05E-3f,dz[k])) break;
    int bot=max(1,k-2);
    for(k=0;k<=bot;++k) if(cf[k]>0.f && cf[k]<.999f) cf[k]=0.f;
    int base=tropo;
    k=tropo;
    while(k>m12) {
        int top=-1;
        if(cf[k]>=.01f) {
            top=k;
            for(int k2=top-1;k2>=m12;--k2)
                if(cf[k2]<.01f || k2==m12) {base=k2+1;break;}
        }
        if(top-base+1>=2) {
            adjust_layer(true,cf,qc,qi,qs,qvs,t,rho,dz,base,top); k=base;
        } else if(cf[base]>0.f && qi[base]<1.e-6f)
            qi[base]=FMUL(1.e-5f,cf[base]);
        --k;
    }
    base=tropo;
    // MPAS's min(m12,m12+2), deliberately different from WRF.
    k=min(m12,m12+2);
    while(k>bot) {
        int top=-1;
        if(cf[k]>=.01f) {
            top=k;
            for(int k2=top-1;k2>=bot;--k2)
                if(cf[k2]<.01f || k2==bot) {base=k2+1;break;}
        }
        if(top-base+1>=2) {
            adjust_layer(false,cf,qc,qi,qs,qvs,t,rho,dz,base,top); k=base;
        } else if(cf[base]>0.f && qc[base]<1.e-6f)
            qc[base]=FMUL(1.e-5f,cf[base]);
        --k;
    }
    adjust_final(cf,qc,qi,rho,dz,tropo);
}

// MPAS module_mp_thompson_cldfra3.F:44-184
extern "C" __global__ void cloud_thompson(int nc,int nz,
    const float *v,float *qc,float *qi,float *qs,const float *p,
    const float *t,const float *rho,const float *land,const float *dx,
    float *cf,float *scratch) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=nc) return;
    int offset=col*nz;
    v+=offset; qc+=offset; qi+=offset; qs+=offset; p+=offset;
    t+=offset; rho+=offset; cf+=offset;
    float *qvs=scratch+col*nz*3, *theta=qvs+nz,*dz=theta+nz;
    float cube=FMUL(FMUL(FMUL(dx[col],dx[col]),dx[col]),.5f);
    float rhl=FADD(.781f,FSQRT(FDIV(1.f,FADD(35.f,cube))));
    float rhocean=FADD(.831f,FSQRT(FDIV(1.f,FADD(70.f,cube))));
    for(int k=0;k<nz;++k) {
        cf[k]=0.f;
        if(qc[k]>1.E-6f || qi[k]>=1.E-7f || qs[k]>1.E-5f) {
            cf[k]=1.f; qvs[k]=v[k];
        } else {
            float tc=FSUB(t[k],273.16f),sw=sat(p[k],t[k],false),si=sat(p[k],t[k],true);
            if(tc>=-12.f) qvs[k]=sw;
            else if(tc<-20.f) qvs[k]=si;
            else qvs[k]=FSUB(sw,FDIV(FMUL(FSUB(sw,si),FSUB(-12.f,tc)),8.f));
            float rh=fmaxf(.01f,fminf(FDIV(v[k],qvs[k]),.9999f));
            float rh00=FSUB(land[col],1.5f)>0.f ? rhocean:rhl;
            if(tc>=-12.f) {
                rh=fminf(.999f,rh);
                cf[k]=fmaxf(0.f,FSUB(1.f,FSQRT(FDIV(FSUB(1.f,rh),FSUB(1.f,rh00)))));
            } else if(tc<-12.f && tc>-70.f && rh>rhocean) {
                rh=fmaxf(.01f,fminf(FDIV(v[k],qvs[k]),FSUB(FDIV(sw,si),1.E-6f)));
                float rhimax=fmaxf(FADD(rh,1.E-6f),FDIV(sw,si));
                cf[k]=fmaxf(0.f,FSUB(1.f,FSQRT(FDIV(FSUB(rhimax,rh),FSUB(rhimax,rhocean)))));
            }
            cf[k]=fminf(.90f,cf[k]);
        }
    }
    find_layers(qvs,cf,t,p,rho,qc,qi,qs,theta,dz,nz);
}

// MPAS mpas_atmphys_o3climatology.F:59-194, decoded table input.
extern "C" __global__ void ozone_latitude(int nc,int nl,const float *lat,
    const float *lat_oz,const float *table,float *monthly) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nc*nl*12) return;
    int m=idx%12,lev=(idx/12)%nl,col=idx/(nl*12);
    float y=FDIV(lat[col],3.1415926f/180.f);
    int i1=0,i2=0;
    // Native F:164-167 overruns at the exact final latitude. Full strict
    // executable repetitions witness unstable words. The in-bounds endpoint
    // keeps the scheme running; its word exclusion is proposed, not accepted
    // under NATIVE-DEFECT-POLICY item 4.
    if(y>=lat_oz[63]) i1=i2=63;
    else if(y<lat_oz[0]) i1=i2=0;
    else {
        for(int i=0;i<63;++i) if(y>=lat_oz[i] && y<lat_oz[i+1]) {i1=i;i2=i+1;break;}
    }
    float d=FSUB(lat_oz[i2],lat_oz[i1]),dy=FSUB(y,lat_oz[i1]);
    float a=table[(m*64+i1)*nl+lev],b=table[(m*64+i2)*nl+lev];
    monthly[idx]=d==0.f ? a:FADD(a,FDIV(FMUL(FSUB(b,a),dy),d));
}

// MPAS mpas_atmphys_o3climatology.F:234-327 + CAM getfactors.
// MPAS module_ra_cam_support.F:2528-2632 (cyclic calendar weights).
extern "C" __global__ void ozone_time(int n,float julian,const float *monthly,float *out) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    const int dates[12]={16,45,75,105,136,166,197,228,258,289,319,350};
    double day=(double)FADD(julian,1.f);
    int iday=(int)day;
    day=DSUB(day,(double)iday); iday=iday%365; if(iday==0) iday=365;
    day=DADD(day,(double)iday);
    int next=0;
    for(int m=0;m<12;++m) if((double)dates[m]>day) {next=m;break;}
    int prev=next>0 ? next-1:11;
    double plus=dates[next],minus=dates[prev],f1,f2;
    if(next==0) {
        double delta=DSUB(DADD(plus,365.),minus);
        if(day>plus) {
            f1=DDIV(DSUB(DADD(plus,365.),day),delta);
            f2=DDIV(DSUB(day,minus),delta);
        } else {
            f1=DDIV(DSUB(plus,day),delta);
            f2=DDIV(DSUB(DADD(day,365.),minus),delta);
        }
    } else {
        double delta=DSUB(plus,minus);
        f1=DDIV(DSUB(plus,day),delta);f2=DDIV(DSUB(day,minus),delta);
    }
    out[idx]=FADD(FMUL((float)f1,monthly[idx*12+prev]),FMUL((float)f2,monthly[idx*12+next]));
}

// MPAS mpas_atmphys_o3climatology.F:197-231 / monthly_interp_to_date.
extern "C" __global__ void ozone_update(int n,int month1,int month2,int d1,int d2,
    const float *monthly,float *out) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    out[idx]=FDIV(FADD(FMUL(monthly[idx*12+month2],(float)d1),
                           FMUL(monthly[idx*12+month1],(float)d2)),(float)(d1+d2));
}

// MPAS module_ra_rrtmg_vinterp.F:22-177. The native lower search bound
// is the minimum previous upper index across ALL columns, not this column.
extern "C" __global__ void ozone_min_upper(int nc,int nl,const int *upper,int *start) {
    if(blockIdx.x*blockDim.x+threadIdx.x != 0) return;
    int low=nl;
    for(int i=0;i<nc;++i) low=min(low,upper[i]);
    start[0]=low;
}
// MPAS module_ra_rrtmg_vinterp.F:22-177
extern "C" __global__ void ozone_vertical_level(int nc,int nz,int nl,int k,int pascal,
    const float *p,const float *pin,const float *ozone,const int *start,int *upper,float *out) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=nc) return;
    float press=pascal ? FDIV(p[col*nz+k],100.f):p[col*nz+k];
    int u=upper[col];
    for(int kk=start[0];kk<nl-1;++kk)
        if(pin[kk]<press && press<=pin[kk+1]) {u=kk;break;}
    upper[col]=u;
    float value;
    if(press<pin[0]) value=FDIV(FMUL(ozone[col*nl],press),pin[0]);
    else if(press>pin[nl-1]) value=ozone[col*nl+nl-1];
    else {
        float dpu=FSUB(press,pin[u]),dpl=FSUB(pin[u+1],press);
        value=FDIV(FADD(FMUL(ozone[col*nl+u],dpl),
                        FMUL(ozone[col*nl+u+1],dpu)),FADD(dpl,dpu));
    }
    out[col*nz+k]=value;
}

// MPAS mpas_atmphys_driver_radiation_lw.F:414-454,688-712;
// MPAS mpas_atmphys_driver_radiation_sw.F:495-531.
extern "C" __global__ void copy_triplet(int n,int enabled,const float *a,
    const float *b,const float *c,float *x,float *y,float *z) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    x[idx]=enabled ? a[idx]:0.f;y[idx]=enabled ? b[idx]:0.f;z[idx]=enabled ? c[idx]:0.f;
}
// MPAS mpas_atmphys_driver_cloudiness.F:130-140,164-170.
// Public seam is [level,cell], local radiation storage is [cell,level].
extern "C" __global__ void profile_from_mpas(int nc,int nz,int ntotal,
    const float *src,float *dst) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nc*nz) dst[i]=src[(i%nz)*ntotal+i/nz];
}
// MPAS mpas_atmphys_driver_cloudiness.F:164-170
extern "C" __global__ void profile_to_mpas(int nc,int nz,int ntotal,
    const float *src,float *dst) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nc*nz) dst[(i%nz)*ntotal+i/nz]=src[i];
}

'''


def _cp():
    """CUDA allocation ABI for mpas_atmphys_driver_cloudiness.F:64-75."""
    import cupy
    return cupy


CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")


def _module():
    """The cloudiness driver module loaded on the calling card."""
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _module_on(int(_cp().cuda.Device().id))


@lru_cache(maxsize=None)
def _module_on(device: int):
    """Compile the driver kernels for mpas_atmphys_driver_cloudiness.F:175-234."""
    # The base's hex libm_identity/rkind_libm modules expose CPU hooks only.
    # Reuse the pinned engine's shared device implementation until f4 supplies
    # the v8.4.2 header; never substitute CUDA transcendental builtins.
    import woof.core.kernels
    from cupy.cuda import compiler, function
    header = Path(woof.core.kernels.__file__).with_name("glibc_flt32.cuh")
    # RawModule's cache compiler appends forced FTZ. Direct NVRTC preserves
    # native RKIND subnormals in the ozone interpolation multiply/add chain.
    binary, _ = compiler.compile_using_nvrtc(
        header.read_text()+"\n"+CUDA_SOURCE, options=_hex_numerics.options(CUDA_OPTIONS),
        arch=_cp().cuda.Device(device).compute_capability)
    with _cp().cuda.Device(device):
        module = function.Module()
        module.load(binary)
    return module


def _array(value, shape, name):
    """Validate pool array ABI from mpas_atmphys_driver_cloudiness.F:92-143."""
    cp = _cp()
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
        raise TypeError(f"{name} must be a device float32 array; prevents silent RKIND conversion")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous with shape {shape}; prevents MPAS stride corruption")
    return value


def _launch(name, count, args):
    """Launch column loops from mpas_atmphys_driver_cloudiness.F:237-370."""
    if count:
        _module().get_function(name)((int((count+127)//128),), (128,), args)


def cloudiness_from_MPAS(*, qv, qc, qi, qs, meshDensity, xland, len_disp):
    """mpas_atmphys_driver_cloudiness.F:64-143: allocate/initialize driver arrays."""
    cp = _cp()
    if qv.ndim != 2 or qv.shape[1] == 0:
        raise ValueError("qv must be (ncol,nlev), nlev>0; prevents invalid MPAS bounds")
    nc, nz = qv.shape
    inputs = tuple(_array(x, qv.shape, name) for x, name in
                   ((qv,"qv"), (qc,"qc"), (qi,"qi"), (qs,"qs")))
    density = _array(meshDensity, (nc,), "meshDensity")
    land = _array(xland, (nc,), "xland")
    out = {name: cp.empty(qv.shape, dtype=cp.float32) for name in
           ("qvrad", "qcrad", "qirad", "qsrad", "cldfrac")}
    out["dx"] = cp.empty(nc, dtype=cp.float32)
    out["xland"] = cp.empty(nc, dtype=cp.float32)
    _launch("cloud_from",nc*nz,(np.int32(nc),np.int32(nz),np.float32(len_disp),
            density,land,*inputs,out["dx"],out["xland"],out["qvrad"],out["qcrad"],
            out["qirad"],out["qsrad"],out["cldfrac"]))
    return out


def cloudiness_to_MPAS(local, cldfrac):
    """mpas_atmphys_driver_cloudiness.F:146-172: write diagnostic fraction."""
    _array(cldfrac, local["cldfrac"].shape, "cldfrac")
    _cp().copyto(cldfrac, local["cldfrac"])


def driver_cloudiness(*, scheme, qv, qc, qi, qs, t, pres, pres_hyd, rho,
                      meshDensity, xland, len_disp, f_qc, f_qi):
    """mpas_atmphys_driver_cloudiness.F:175-234: full cloudiness driver slice."""
    if scheme not in CLOUD_SCHEMES:
        raise ValueError(f"unknown MPAS cloud scheme {scheme!r}: prevents an unselected cloudiness slot")
    fields = {name: _array(x, qv.shape, name) for name,x in
              (("t",t),("pres",pres),("pres_hyd",pres_hyd),("rho",rho))}
    nc,nz = qv.shape
    if scheme == "cld_fraction_thompson" and nz < 3:
        raise ValueError("MPAS Thompson find_cloudLayers reads kts+2 and dz(kts+1); requires >=3 levels")
    out = cloudiness_from_MPAS(qv=qv,qc=qc,qi=qi,qs=qs,meshDensity=meshDensity,
                              xland=xland,len_disp=len_disp)
    if scheme == "cld_fraction_thompson":
        scratch = _cp().empty((nc,nz,3), dtype=_cp().float32)
        _launch("cloud_thompson",nc,(np.int32(nc),np.int32(nz),out["qvrad"],
                out["qcrad"],out["qirad"],out["qsrad"],fields["pres_hyd"],fields["t"],
                fields["rho"],out["xland"],out["dx"],out["cldfrac"],scratch))
    else:
        _launch("cloud_simple",nc*nz,(np.int32(nc*nz),np.int32(CLOUD_SCHEMES.index(scheme)),
                np.int32(f_qc),np.int32(f_qi),out["qvrad"],out["qcrad"],out["qirad"],
                out["qsrad"],fields["t"],fields["pres"],out["cldfrac"]))
    return out


def init_o3climatology(*, latCell, lat_ozone, ozmixin):
    """mpas_atmphys_o3climatology.F:59-194, with table decode owned by Rust.

    MPAS overruns lat_ozone at exactly its final latitude. The full native
    executable varies there across hosts/ranks; the native-defect policy
    endpoint is a proposed exclusion requiring the owner's case-specific approval;
    REPORT.md records the exact affected words and witnesses.
    """
    cp = _cp()
    nc = latCell.size
    _array(latCell,(nc,),"latCell")
    _array(lat_ozone,(64,),"lat_ozone")
    if ozmixin.ndim != 3 or ozmixin.shape[:2] != (12,64):
        raise ValueError("ozmixin must be (12,64,nOznLevels); prevents wrong table order")
    _array(ozmixin,ozmixin.shape,"ozmixin")
    nl = ozmixin.shape[2]
    out = cp.empty((nc,nl,12),dtype=cp.float32)
    _launch("ozone_latitude",out.size,(np.int32(nc),np.int32(nl),latCell,lat_ozone,ozmixin,out))
    return out


def o3climatology_from_MPAS(julian, ozmixm):
    """mpas_atmphys_o3climatology.F:234-327: CAM calendar interpolation."""
    cp = _cp()
    if ozmixm.ndim != 3 or ozmixm.shape[2] != 12:
        raise ValueError("ozmixm must be (ncol,nOznLevels,12); prevents invalid month indices")
    _array(ozmixm,ozmixm.shape,"ozmixm")
    out=cp.empty(ozmixm.shape[:2],dtype=cp.float32)
    _launch("ozone_time",out.size,(np.int32(out.size),np.float32(julian),ozmixm,out))
    return out


@dataclass
class OzoneCalendarState:
    """Restart-owned MPAS get_julgmt DATA mmd(2), including its saved leap-day state."""
    february_days: int = 28


def _monthly_date_weights(current_date, calendar_state):
    """src/core_atmosphere/physics/mpas_atmphys_date_time.F:51-84,148-219: saved calendar and month weights."""
    # MPAS get_julgmt's DATA mmd persists. Once a leap-year call changes
    # February to 29 it is never reset to 28, even in a later non-leap year.
    # Keep that source behavior explicitly in caller-owned restart metadata.
    from datetime import date
    year,month,day=map(int,current_date[:10].split("-"))
    leap=(year%4==0 and year%100!=0) or year%400==0
    if calendar_state is None:
        calendar_state=OzoneCalendarState()
    if leap:
        calendar_state.february_days=29
    if calendar_state.february_days not in (28,29):
        raise ValueError("invalid saved MPAS February length; prevents corrupt calendar restart")
    extra=calendar_state.february_days-(29 if leap else 28)
    target=date(year,month,day).timetuple().tm_yday+(extra if month>2 else 0)
    middle=[date(year,m,15).timetuple().tm_yday+(extra if m>2 else 0) for m in range(1,13)]
    middle=[middle[0]-31,*middle,middle[-1]+31]
    l=next(i for i in range(13) if middle[i]<target<=middle[i+1])
    month1,month2=(11,0) if l in (0,12) else (l-1,l)
    return month1,month2,target-middle[l],middle[l+1]-target


def update_o3climatology(current_date, ozmixm, *, calendar_state=None):
    """mpas_atmphys_o3climatology.F:197-231; src/core_atmosphere/physics/mpas_atmphys_date_time.F:148-219.

    Only calendar metadata is calculated in Python; field interpolation
    uses MPAS's multiply/add/divide order on the device. A repeated run
    supplies its OzoneCalendarState through the restart-owned pools.
    """
    month1,month2,d1,d2=_monthly_date_weights(current_date,calendar_state)
    cp=_cp()
    if ozmixm.ndim != 3 or ozmixm.shape[2] != 12:
        raise ValueError("ozmixm must be (ncol,nOznLevels,12); prevents invalid month indices")
    _array(ozmixm,ozmixm.shape,"ozmixm")
    out=cp.empty(ozmixm.shape[:2],dtype=cp.float32)
    _launch("ozone_update",out.size,(np.int32(out.size),np.int32(month1),np.int32(month2),
            np.int32(d1),np.int32(d2),ozmixm,out))
    return out


def vinterp_ozn(pmid, pin, o3clim, *, pressure_in_pa=False):
    """module_ra_rrtmg_vinterp.F:22-177; LW driver Pa conversion: 481-506."""
    cp=_cp()
    if pmid.ndim != 2 or o3clim.ndim != 2:
        raise ValueError("ozone/pressure must be profiles; prevents wrong pressure level order")
    nc,nz=pmid.shape; nl=pin.size
    if nl<2:
        raise ValueError("MPAS vinterp_ozn reads upper+1; requires >=2 ozone levels")
    _array(pmid,(nc,nz),"pmid");_array(pin,(nl,),"pin");_array(o3clim,(nc,nl),"o3clim")
    out=cp.empty_like(pmid)
    upper=cp.zeros(nc,dtype=cp.int32)
    start=cp.empty(1,dtype=cp.int32)
    for k in range(nz-1,-1,-1):
        _launch("ozone_min_upper",1,(np.int32(nc),np.int32(nl),upper,start))
        _launch("ozone_vertical_level",nc,(np.int32(nc),np.int32(nz),np.int32(nl),
                np.int32(k),np.int32(pressure_in_pa),pmid,pin,o3clim,start,upper,out))
    return out


def radii_from_MPAS(*, microp_scheme, microp_re, re_cloud, re_ice, re_snow):
    """mpas_atmphys_driver_radiation_lw.F:414-454; SW:495-531.

    Returns None for MPAS's default (untouched) arm; the caller keeps its
    allocated driver arrays in that case.
    """
    if microp_scheme.strip() not in RADIUS_SCHEMES:
        return None
    return _copy_radii(re_cloud,re_ice,re_snow,bool(microp_re))


def radii_to_MPAS(*, microp_scheme, microp_re, rre_cloud, rre_ice, rre_snow):
    """mpas_atmphys_driver_radiation_lw.F:688-712: LW diagnostic radii."""
    if microp_scheme.strip() not in RADIUS_SCHEMES:
        return None
    return _copy_radii(rre_cloud,rre_ice,rre_snow,bool(microp_re))


def _copy_radii(a,b,c,enabled):
    """mpas_atmphys_driver_radiation_lw.F:414-454,688-712: radius copies."""
    cp=_cp()
    for name,x in (("re_cloud",a),("re_ice",b),("re_snow",c)):
        _array(x,a.shape,name)
    out=tuple(cp.empty_like(a) for _ in range(3))
    _launch("copy_triplet",a.size,(np.int32(a.size),np.int32(enabled),a,b,c,*out))
    return out



def _from_public(profile, nsolve):
    """mpas_atmphys_driver_cloudiness.F:130-140: input profile order."""
    cp = _cp()
    if profile.ndim != 2:
        raise ValueError("MPAS public profile must be [level,cell]; prevents transposed native fields")
    nz,ntotal=profile.shape
    if not 0 <= nsolve <= ntotal:
        raise ValueError("nCellsSolve exceeds profile extent; prevents halo overrun")
    _array(profile,(nz,ntotal),"MPAS profile")
    out=cp.empty((nsolve,nz),dtype=cp.float32)
    _launch("profile_from_mpas",out.size,(np.int32(nsolve),np.int32(nz),
            np.int32(ntotal),profile,out))
    return out


def _to_public(local, public):
    """mpas_atmphys_driver_cloudiness.F:164-170: output profile order."""
    nc,nz=local.shape
    if public.ndim != 2 or public.shape[0] != nz or public.shape[1] < nc:
        raise ValueError("MPAS output extent cannot hold solve columns; prevents halo overwrite")
    _array(public,public.shape,"MPAS output")
    _launch("profile_to_mpas",local.size,(np.int32(nc),np.int32(nz),
            np.int32(public.shape[1]),local,public))


def cloudiness_slot(call):
    """mpas_atmphys_driver_cloudiness.F:175-234, f4 SlotCall callback.

    pools: prepared (qv,qc,qi,qs,t,pres,pres_hyd,rho); mesh
    (meshDensity); sfc_input (xland); diag_physics (cldfrac);
    nCellsSolve, config_len_disp, f_qc, f_qi. Public profiles use f4's
    [level,cell] layout. radshared holds the published [cell,level]
    diagnostic-only radiation carriers for p22/p23 and restart glue.
    """
    pools=call.pools
    nc=int(pools["nCellsSolve"])
    profiles={name:_from_public(pools["prepared"][name],nc) for name in
              ("qv","qc","qi","qs","t","pres","pres_hyd","rho")}
    # Column vectors already have the MPAS layout. Slicing solve extents
    # creates views, preserving halos and avoiding any host field processing.
    out=driver_cloudiness(scheme=call.scheme,**profiles,
        meshDensity=pools["mesh"]["meshDensity"][:nc],
        xland=pools["sfc_input"]["xland"][:nc],
        len_disp=pools["config_len_disp"],f_qc=pools["f_qc"],f_qi=pools["f_qi"])
    if icloud_bl_enabled(call.options):
        # WRF v4.6.1 merge for modern MYNN grid-mean condensate.
        # Modify radiation-only carriers, never the prognostic qc/qi state.
        diag = pools["diag_physics"]
        cf_bl = _from_public(diag["cldfrac_bl"], nc)
        qc_bl = _from_public(diag["qc_bl"], nc)
        qi_bl = _from_public(diag["qi_bl"], nc)
        merge_mynn_bl_clouds(out["qcrad"], out["qirad"], out["cldfrac"],
            qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cf_bl,
            itimestep=call.itimestep, xp=_cp())
    _to_public(out["cldfrac"],pools["diag_physics"]["cldfrac"])
    # MPAS cloudiness_from_MPAS:131-145 publishes these prepared globals.
    # Native radiation consumes q*rad_p (LW:868-869, SW:924-925), so the
    # mixed-backend path must commit them as well as the diagnostic field.
    prepared_radiation=pools.get("radiation_prepared")
    if prepared_radiation is not None:
        for name in ("qvrad","qcrad","qirad","qsrad","cldfrac"):
            _to_public(out[name],prepared_radiation[name])
        for name in ("dx","xland"):
            prepared_radiation[name][:nc]=out[name]
    # Ozone calendar/table state is restart-owned in this same pool. A
    # cloudiness alarm updates only the cloud carriers, preserving that state.
    pools.setdefault("radshared",{}).update(out)
    return out


def register_hex_cuda(registry):
    """mpas_atmphys_driver_cloudiness.F:203-234: register the three names."""
    for scheme in CLOUD_SCHEMES:
        registry.register("cloudiness",scheme,"hex-cuda",cloudiness_slot)


def bind_native_pools(provider, call):
    """MPAS mpas_atmphys_driver_cloudiness.F:92-172,175-234.

    Bind f4's already prepared native globals, without recomputing fields.
    The native extra top workspace row is outside kts:kte and is preserved
    by the provider's commit. Shared cloud carriers persist for p22/p23.
    """
    dims=provider.session.layout.dimensions
    nz=int(dims['nVertLevels']);nc=int(dims['nCellsSolve'])
    prepared={name:provider.prepared(name+'_p')[:nz] for name in
              ('qv','qc','qi','qs','t','pres','pres_hyd','rho')}
    radiation={name:provider.prepared(name+'_p')[:nz] for name in
               ('qvrad','qcrad','qirad','qsrad','cldfrac')}
    radiation.update({name:provider.prepared(name+'_p') for name in ('dx','xland')})
    shared=getattr(provider,'radshared_v842_state',None)
    if shared is None:
        shared={};provider.radshared_v842_state=shared
    return {'prepared':prepared,'radiation_prepared':radiation,'mesh':provider['mesh'],
            'sfc_input':provider['sfc_input'],'diag_physics':provider['diag_physics'],
            'nCellsSolve':nc,'config_len_disp':provider['configs']['config_len_disp'],
            'f_qc':provider['scalar_state']['f_qc'],
            'f_qi':provider['scalar_state']['f_qi'],'radshared':shared}


def load_ozone_resources(resources, device):
    """mpas_atmphys_o3climatology.F:90-140: authenticated table words.

    Keep the original resource binding usable with published decoder packets.
    Their optional ``rust_decoder`` entry need not export an ozone ABI: these
    three formatted input tables use the same reader as framework init.
    """
    from .framework_resources_v842 import decode_ozone_tables
    return decode_ozone_tables(resources, device)


def bind_native_ozone_pools(provider, call):
    """mpas_atmphys_o3climatology.F:234-275: time-update public pool binding."""
    shared=getattr(provider,'radshared_v842_state',None)
    if shared is None:
        shared={};provider.radshared_v842_state=shared
    return {'nCellsSolve':int(provider.session.layout.dimensions['nCellsSolve']),
            'atm_input':provider['atm_input'],'diag_physics':provider['diag_physics'],
            'julian':provider.session.arguments.calendar['julian'],
            'current_date':provider.session.arguments.calendar['current_date'],'radshared':shared}


def bind_native_ozone_initial_pools(provider, call):
    """mpas_atmphys_o3climatology.F:90-98,146-187: complete nCells initialization.

    The ordinary slot provider packs nCellsSolve. Initialization instead writes
    every native nCells word, including halos, and preserves extra ABI padding.
    Tables must be supplied before the nested native initialization boundary.
    """
    nc=int(provider.session.layout.dimensions['nCells'])
    def full_cell_field(pool,name):
        host=provider.native.get_field(pool,name)
        selection=(...,slice(0,nc))
        packed=provider.cp.ascontiguousarray(provider.cp.asarray(host[selection]))
        provider.buffers[pool,name,1]=(host,packed,selection)
        return packed
    shared=getattr(provider,'radshared_v842_state',None)
    if shared is None:
        shared={};provider.radshared_v842_state=shared
    return {'nCells':nc,'mesh':{'latCell':full_cell_field('mesh','latCell')},
            'atm_input':{'ozmixm':full_cell_field('atm_input','ozmixm'),
                         'pin':provider['atm_input']['pin']},
            'radshared_tables':provider.radshared_v842_tables,'radshared':shared}


def ozone_slot(call):
    """mpas_atmphys_o3climatology.F:234-327, f4 framework hook slice.

    pools: atm_input.ozmixm [month,ozone level,cell],
    diag_physics.o3clim [ozone level,cell], nCellsSolve and julian.
    Tables and calendar metadata remain in the caller-owned restart pools.
    """
    pools=call.pools
    local=_monthly_from_pool(pools,int(pools["nCellsSolve"]))
    out=o3climatology_from_MPAS(pools["julian"],local)
    _to_public(out,pools["diag_physics"]["o3clim"])
    pools.setdefault("radshared",{})["o3clim"]=out
    return out


def _monthly_from_pool(pools,nc):
    """mpas_atmphys_o3climatology.F:221-227,274-275: monthly pool layout."""
    monthly=pools["atm_input"]["ozmixm"]
    if monthly.ndim != 3 or monthly.shape[0] != 12 or not 0 <= nc <= monthly.shape[2]:
        raise ValueError("MPAS monthly ozone must be [12,level,cell] with cell extent; prevents month/halo corruption")
    _array(monthly,monthly.shape,"MPAS monthly ozone")
    return _cp().ascontiguousarray(monthly[:,:,:nc].transpose(2,1,0))


def init_ozone_slot(call):
    """mpas_atmphys_o3climatology.F:59-194: initialization pool slice.

    radshared_tables holds Rust-decoded device lat_ozone, ozmixin and pin.
    Initialization fills nCells (including native halo cells), unlike the
    time update's nCellsSolve. No table files are decoded in Python.
    """
    cp=_cp();pools=call.pools;nc=int(pools["nCells"])
    tables=pools["radshared_tables"]
    latitude=pools["mesh"]["latCell"]
    _array(latitude,(latitude.size,),"MPAS latCell")
    if not 0 <= nc <= latitude.size:
        raise ValueError("nCells exceeds ozone latitude extent; prevents initialization halo overrun")
    out=init_o3climatology(latCell=latitude[:nc],
                         lat_ozone=tables["lat_ozone"],ozmixin=tables["ozmixin"])
    monthly=pools["atm_input"]["ozmixm"]
    if monthly.ndim != 3 or monthly.shape[:2] != (12,out.shape[1]) or monthly.shape[2] < nc:
        raise ValueError("MPAS ozone table/pool extents differ; prevents wrong pressure interpolation or halo overrun")
    _array(monthly,monthly.shape,"MPAS monthly ozone")
    pressure=_array(tables["pin"],(out.shape[1],),"ozone table pin")
    target=_array(pools["atm_input"]["pin"],pressure.shape,"MPAS pin")
    cp.copyto(target,pressure)
    cp.copyto(monthly[:,:,:nc],out.transpose(2,1,0))
    # The climatology lives on in the ozmixm pool field only.  THE BREAKAGE THIS PREVENTS: a second (nCells, levels, 12)
    # copy kept in the provider shared dict for the whole run (2.8 KB per cell, 0.8 GB on a conus3km rank); no slot or
    # replay reads radshared["ozmixm"] (ozone_slot and the RRFS drivers read atm_input.ozmixm).
    return out


def update_ozone_slot(call):
    """mpas_atmphys_o3climatology.F:197-231: date-based pool slice."""
    pools=call.pools;shared=pools.setdefault("radshared",{})
    state=shared.setdefault("ozone_calendar",OzoneCalendarState())
    out=update_o3climatology(pools["current_date"],
        _monthly_from_pool(pools,int(pools["nCellsSolve"])),calendar_state=state)
    _to_public(out,pools["diag_physics"]["o3clim"])
    shared["o3clim"]=out
    return out


def icloud_bl_enabled(options):
    value = options.get('config_hex_icloud_bl', False)
    if type(value) is not bool:
        raise ValueError('config_hex_icloud_bl must be a boolean')
    if value and options.get('config_pbl_scheme') != 'bl_mynn':
        raise ValueError('config_hex_icloud_bl requires bl_mynn')
    return value
