"""MPAS-A v8.4.2 GSL orographic UGWP, RKIND=4, resident CUDA columns.

Authority: physics_noaa/UGWP/bl_ugwp.F:1-1114 (UGWP c1c893e),
physics_wrf/module_bl_ugwp_gwdo.F:15-357,385-469, and
mpas_atmphys_driver_gwdo.F:365-459,520-549,555-689,779-822.
The nonstationary call at wrapper lines 362-382 belongs to p15.

Arrays are float32 C [level,column], surface [column], with the lowest
layer first. kpbl is int32 and retains Fortran's one-based index. Python
only allocates, copies, validates descriptors, and launches device code.
All atmospheric arithmetic, including driver dx and rounded tendency
increments, runs in kernels. The four lifecycle routines are native no-ops.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache, wraps
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEME = "bl_ugwp_gwdo"
BACKEND = "hex-cuda"
MPAS_SOURCE = "physics_noaa/UGWP/bl_ugwp.F:1-1114"
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
LAYER_INPUTS = ("uproj", "vproj", "t1", "q1", "prsl", "prslk", "zl", "dz",
                "rublten", "rvblten")
SURFACE_INPUTS = ("sina", "cosa", "hpbl", "br1", "xland1", "dxmeter",
                  "var", "oc1", "oa2d1", "oa2d2", "oa2d3", "oa2d4",
                  "ol2d1", "ol2d2", "ol2d3", "ol2d4", "varss", "oc1ss",
                  "oa2d1ss", "oa2d2ss", "oa2d3ss", "oa2d4ss",
                  "ol2d1ss", "ol2d2ss", "ol2d3ss", "ol2d4ss")
LAYER_OUTPUTS = ("rublten", "rvblten", "dtaux3d", "dtauy3d",
                 "dtaux3d_ls", "dtauy3d_ls", "dtaux3d_bl", "dtauy3d_bl",
                 "dtaux3d_ss", "dtauy3d_ss", "dtaux3d_fd", "dtauy3d_fd")
SURFACE_OUTPUTS = ("dusfcg", "dvsfcg", "dusfc_ls", "dvsfc_ls", "dusfc_bl",
                   "dvsfc_bl", "dusfc_ss", "dvsfc_ss", "dusfc_fd", "dvsfc_fd")

# All arithmetic follows the source expressions with contraction disabled.
# Explicit RN sqrt/division preserve the strict Fortran result and subnormals.
# Level arrays retain Fortran indexing to keep the source comparison direct.
CUDA_SOURCE = r'''
// MPAS physics_noaa/UGWP/bl_ugwp.F:15-1059, complete bl_ugwp_run.
extern "C" __global__ void ugwp_oro_run(
    const float *layers, const float *prsi0, const float *surface,
    const int *kpbl0, float *out, float *sout, int *status,
    float g, float cp, float rd, float rv, float fv, float pi_c,
    float dt, int nz, int nc, int diags)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nc) return;
    const int kte = nz, kpblmax = nz, kpblmin = 2;
    const size_t n3 = (size_t)nz * nc;
    #define IN(m,k) layers[(size_t)(m)*n3 + (size_t)((k)-1)*nc + i]
    #define S(m) surface[(size_t)(m)*nc + i]
    #define P(k) prsi0[(size_t)((k)-1)*nc + i]
    #define O(m,k) out[(size_t)(m)*n3 + (size_t)((k)-1)*nc + i]
    #define SO(m) sout[(size_t)(m)*nc + i]
    const float sa = S(0), ca = S(1), hpbl = S(2), br = S(3);
    const float xland = S(4), dx = S(5), var = S(6), oc1 = S(7), varss = S(16);
    const int kpbl = kpbl0[i];
    const float inv_g = 1.0f / g, fdir = 8.0f / (2.0f*pi_c);
    float u[UGWP_LEVELS+2], v[UGWP_LEVELS+2], vtj[UGWP_LEVELS+2];
    float vtk[UGWP_LEVELS+2], rho[UGWP_LEVELS+2], del[UGWP_LEVELS+2];
    float za[UGWP_LEVELS+2], bnv2[UGWP_LEVELS+2], ri[UGWP_LEVELS+2];
    float velco[UGWP_LEVELS+2], taup[UGWP_LEVELS+2], taufb[UGWP_LEVELS+2];
    float taud_ls[UGWP_LEVELS+2], taud_bl[UGWP_LEVELS+2];
    float dudt[UGWP_LEVELS+2], dvdt[UGWP_LEVELS+2];
    float dusfc = 0.0f, dvsfc = 0.0f;
    float dus[4] = {0.0f,0.0f,0.0f,0.0f}, dvs[4] = {0.0f,0.0f,0.0f,0.0f};
    // bl_ugwp.F:275-354: constants, taper, initialization.
    float ls_taper;
    if (dx >= 13000.0f) ls_taper = 1.0f;
    else if (dx <= 3000.0f) ls_taper = 0.0f;
    else ls_taper = 0.5f * (glibc_sinf(pi_c * (dx-0.5f*(13000.0f+3000.0f))
                                    / (13000.0f-3000.0f)) + 1.0f);
    const float ss_taper = 1.0f;
    float dxy4[4] = {dx, dx, __fsqrt_rn(gfk_pow(dx,2.0f)+gfk_pow(dx,2.0f)),0.0f};
    dxy4[3] = dxy4[2];
    for (int k=1; k<=kte+1; ++k) { taup[k]=0.0f; taufb[k]=0.0f; }
    const float zs = IN(6,1) - 0.5f * IN(7,1);
    // bl_ugwp.F:356-380: carriers and earth-relative winds.
    for (int k=1; k<=kte; ++k) {
        vtj[k] = IN(2,k) * (1.0f + fv*IN(3,k));
        vtk[k] = vtj[k] / IN(5,k);
        rho[k] = 1.0f / rd * IN(4,k) / vtj[k];
        del[k] = P(k)-P(k+1);
        u[k] = IN(0,k)*ca - IN(1,k)*sa;
        v[k] = IN(0,k)*sa + IN(1,k)*ca;
        za[k] = IN(6,k)-zs;
        bnv2[k]=0.0f; ri[k]=0.0f; velco[k]=0.0f;
        taud_ls[k]=0.0f; taud_bl[k]=0.0f; dudt[k]=0.0f; dvdt[k]=0.0f;
        for(int m=2;m<12;++m) O(m,k)=0.0f;
    }
    // bl_ugwp.F:382-481: reference layer and direction.
    int klowtop=0;
    const float zlowtop=2.0f*var;
    if(zlowtop>0.0f) for(int k=2;k<=kte;++k) {
        if(IN(6,k)-IN(6,1)>=zlowtop) { klowtop=k+1; break; }
    }
    const int kbl = max(min(max(kpbl,klowtop),kpblmax),kpblmin);
    const int komax = klowtop-1;
    const float delks = 1.0f/(P(1)-P(kbl));
    const float delks1 = 1.0f/(IN(4,1)-IN(4,kbl));
    float ubar=0.0f,vbar=0.0f,rhobar=0.0f;
    for(int k=1;k<kbl;++k) {
        const float rcsks=del[k]*delks, rdelks=del[k]*delks;
        ubar=ubar+rcsks*u[k]; vbar=vbar+rcsks*v[k]; rhobar=rhobar+rdelks*rho[k];
    }
    // Same shared atan2 policy as terrain_drag.cu: double atan2 -> RN float.
    const float wdir=__double2float_rn(atan2((double)ubar,(double)vbar))+pi_c;
    const int idir=((int)roundf(fdir*wdir))%8;
    const int nwdir[8]={6,7,5,8,2,3,1,4};
    const int nwd=nwdir[idir], m=(nwd-1)%4;
    const float oa=(float)(1-2*((nwd-1)/4))*S(8+m), ol=S(12+m);
    const int cross[4]={1,0,3,2};
    const float olp=S(12+cross[m]);
    const float od=fmaxf(fminf(olp/fmaxf(ol,1.e-5f),10.0f),0.1f);
    const float dxy=dxy4[m];
    const bool ls_on=ls_taper>1.e-02f;
    bool ldrag=false;
    float ulow=0.0f, rulow=0.0f, xn=0.0f,yn=0.0f,coefm=0.0f,taub=0.0f;
    // bl_ugwp.F:487-603: mesoscale base stress, stability and propagation.
    if(ls_on) {
        for(int k=1;k<kte;++k) {
            const float ti=2.0f/(IN(2,k)+IN(2,k+1));
            const float rdz=1.0f/(IN(6,k+1)-IN(6,k));
            const float tem1=u[k]-u[k+1],tem2=v[k]-v[k+1];
            const float dw2=tem1*tem1+tem2*tem2;
            const float shr2=fmaxf(dw2,1.0f)*rdz*rdz;
            const float bvf2=g*(g/cp+rdz*(vtj[k+1]-vtj[k]))*ti;
            ri[k]=fmaxf(bvf2/shr2,-100.0f);
            bnv2[k]=2.0f*g*rdz*(vtk[k+1]-vtk[k])/(vtk[k+1]+vtk[k]);
            bnv2[k]=fmaxf(bnv2[k],1.e-5f);
        }
        ulow=fmaxf(__fsqrt_rn(ubar*ubar+vbar*vbar),1.0f);
        rulow=1.0f/ulow;
        for(int k=1;k<kte;++k) {
            velco[k]=0.5f*((u[k]+u[k+1])*ubar+(v[k]+v[k+1])*vbar);
            velco[k]=velco[k]*rulow;
            if(velco[k]<1.0f && velco[k]>0.0f) velco[k]=1.0f;
        }
        ldrag=velco[1]<=0.0f;
        for(int k=2;k<kbl;++k) ldrag=ldrag || velco[k]<=0.0f;
        const float wtkbj=(IN(4,1)-IN(4,2))*delks1;
        bnv2[1]=wtkbj*bnv2[1]; ri[1]=wtkbj*ri[1];
        for(int k=2;k<kbl;++k) {
            const float rdelks=(IN(4,k)-IN(4,k+1))*delks1;
            bnv2[1]=bnv2[1]+bnv2[k]*rdelks; ri[1]=ri[1]+ri[k]*rdelks;
        }
        ldrag=ldrag || bnv2[1]<=0.0f;
        ldrag=ldrag || ulow==1.0f;
        ldrag=ldrag || var<=0.0f;
        const float ksmax=4.0f*pi_c/dx;
        if(bnv2[1]>0.0f) ldrag=ldrag || __fsqrt_rn(bnv2[1])*rulow<ksmax;
        for(int k=2;k<kbl;++k) ri[k]=ri[1];
        if(!ldrag) {
            const float bnv=__fsqrt_rn(bnv2[1]);
            float fr=bnv*rulow*var*od; fr=fminf(fr,10.0f);
            xn=ubar*rulow; yn=vbar*rulow;
            float efact=gfk_pow(oa+2.0f,0.8f*fr/1.0f);
            efact=fminf(fmaxf(efact,0.0f),10.0f);
            coefm=gfk_pow(1.0f+ol,oa+1.0f);
            const float xlinv=coefm*8.84e-6f;
            const float tem=fr*fr*oc1;
            const float gfobnv=1.0f*tem/((tem+0.5f)*bnv);
            taub=xlinv*rhobar*ulow*ulow*ulow*gfobnv*efact;
        }
    }
    // bl_ugwp.F:609-731: small-scale stable boundary-layer waves.
    float hpbl2=hpbl+10.0f;
    int kpbl2=kpbl;
    for(int k=2;k<=max(kpbl,2);++k) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:634-646 reads za(k), but may
        // exit before reaching an excessive KPBL. Check the actual read.
        if(k>nz) { atomicExch(status,1); return; }
        if(za[k]>300.0f) {
            kpbl2=k;
            hpbl2=(k==kpbl)?hpbl+10.0f:za[k]+10.0f;
            break;
        }
    }
    float tauwavex0=0.0f,tauwavey0=0.0f;
    bool wave=false;
    // bl_ugwp.F:614-615,645-709: for kpbl<=0 the native write loop
    // is empty and initialized SS tendencies remain zero, even if the
    // preceding thvx(kpbl2) read is outside storage. Preserve those
    // observable outputs without refusing a native-completing column.
    if(xland-1.5f<=0.0f && 2.0f*varss<=hpbl && br>0.0f && kpbl>0) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:645-685 dereferences kpbl2 only
        // in this branch. Native water/inactive columns may carry zero.
        if(kpbl2<1 || kpbl2>nz) { atomicExch(status,1); return; }
        const float thv1=(IN(2,1)/IN(5,1))*(1.0f+fv*IN(3,1));
        const float thvk=(IN(2,kpbl2)/IN(5,kpbl2))*(1.0f+fv*IN(3,kpbl2));
        if(br>0.0f && thvk-thv1>0.0f) {
            const float xlinv=0.001f*pi_c;
            const float govrth=g/(0.5f*(thvk+thv1));
            const float xnbv=__fsqrt_rn(govrth*(thvk-thv1)/hpbl2);
            if(u[kpbl2]==0.0f || fabsf(xnbv/u[kpbl2])>xlinv) {
                const float tv=2.0f*varss;
                const float vtemp2=0.5f*xnbv*xlinv*(tv*tv)*rho[1];
                tauwavex0=vtemp2*u[1]/(1.0f+vtemp2*dt);
                tauwavex0=tauwavex0*ss_taper;
            }
            if(v[kpbl2]==0.0f || fabsf(xnbv/v[kpbl2])>xlinv) {
                const float tv=2.0f*varss;
                const float vtemp2=0.5f*xnbv*xlinv*(tv*tv)*rho[1];
                tauwavey0=vtemp2*v[1]/(1.0f+vtemp2*dt);
                tauwavey0=tauwavey0*ss_taper;
            }
            wave=true;
        }
    }
    // UGWP_INVALID_KPBL: bl_ugwp.F:701-709 writes through kpbl when
    // waves are active. Inactive columns do not perform those writes.
    if(wave && kpbl>nz) { atomicExch(status,1); return; }
    for(int k=1;k<=kte;++k) {
        float ut=0.0f,vt=0.0f;
        if(wave && k<=kpbl) {
            ut=-1.0f*tauwavex0*2.0f*fmaxf(1.0f-za[k]/hpbl2,0.0f)/hpbl2;
            vt=-1.0f*tauwavey0*2.0f*fmaxf(1.0f-za[k]/hpbl2,0.0f)/hpbl2;
        }
        dudt[k]=dudt[k]+ut; dvdt[k]=dvdt[k]+vt;
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*del[k]; dvsfc=dvsfc+vt*del[k];
        if(diags) { O(8,k)=ut; O(9,k)=vt; dus[2]=dus[2]+ut*del[k]; dvs[2]=dvs[2]+vt*del[k]; }
    }
    // bl_ugwp.F:737-807: semi-implicit Beljaars topographic form drag.
    float var_temp=fminf(varss,500.0f)+fmaxf(0.0f,0.2f*(varss-500.0f));
    const float a1=0.00026615161f*(var_temp*var_temp), a2=a1*0.005363f;
    bool form=xland-1.5f<=0.0f;
    for(int k=1;k<=kte;++k) {
        float ut=0.0f,vt=0.0f;
        if(form) {
            const float wsp=__fsqrt_rn(u[k]*u[k]+v[k]*v[k]);
            var_temp=12.0f*6.325e-3f*gfk_exp(-gfk_pow(za[k]/1500.0f,1.5f))*a2
                     *gfk_pow(za[k],-1.2f)*ss_taper;
            ut=-var_temp*wsp*u[k]/(1.0f+var_temp*dt*wsp);
            vt=-var_temp*wsp*v[k]/(1.0f+var_temp*dt*wsp);
            if(za[k]/1500.0f>4.0f) form=false;
        }
        dudt[k]=dudt[k]+ut; dvdt[k]=dvdt[k]+vt;
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*del[k]; dvsfc=dvsfc+vt*del[k];
        if(diags) { O(10,k)=ut; O(11,k)=vt; dus[3]=dus[3]+ut*del[k]; dvs[3]=dvs[3]+vt*del[k]; }
    }
    // bl_ugwp.F:814-878: vertical mesoscale stress and Lindzen saturation.
    if(ls_on) {
        for(int k=1;k<=kbl;++k) taup[k]=taub;
        bool icrilv=false;
        float brvf=0.0f;
        for(int k=2;k<kte;++k) {
            const int kp1=k+1;
            if(k>=kbl) {
                icrilv=icrilv || ri[k]<0.25f || velco[k]<=0.0f;
                brvf=__fsqrt_rn(fmaxf(bnv2[k],1.e-5f));
            }
            if(k>=kbl && !ldrag) {
                if(!icrilv && taup[k]>0.0f) {
                    const float temv=1.0f/velco[k];
                    const float tem1=coefm/dxy*(rho[kp1]+rho[k])*brvf*velco[k]*0.5f;
                    float hd=__fsqrt_rn(taup[k]/tem1);
                    const float fro=brvf*hd*temv;
                    const float tem2=__fsqrt_rn(ri[k]), tem=1.0f+tem2*fro;
                    const float rim=ri[k]*(1.0f-fro)/(tem*tem);
                    if(rim<=0.25f) {
                        if(oa<=0.0f || kp1>=kpblmin) {
                            const float temc=2.0f+1.0f/tem2;
                            hd=velco[k]*(2.0f*__fsqrt_rn(temc)-temc)/brvf;
                            taup[kp1]=tem1*hd*hd;
                        }
                    } else taup[kp1]=taup[k];
                }
            }
        }
        // bl_ugwp.F:884-940: flow-blocking layer and stress.
        if(!ldrag) {
            int kblk=0;
            float pe=0.0f,zblk=0.0f;
            for(int k=kte;k>=2;--k) {
                if(kblk==0 && k<=komax) {
                    pe=pe+bnv2[k]*(IN(6,komax)-IN(6,k))*del[k]/g/rho[k];
                    const float ke=0.5f*(gfk_pow(u[k],2.0f)+gfk_pow(v[k],2.0f));
                    if(pe>=ke) { kblk=min(k,kbl); zblk=za[kblk]; }
                }
            }
            if(kblk!=0) {
                const float cd=fmaxf(2.0f-1.0f/od,0.0f);
                taufb[1]=0.5f*1.92e-4f*rhobar*coefm*cd*olp*zblk*(ulow*ulow);
                const float tautem=taufb[1]/(float)(kblk-1);
                for(int k=2;k<=kblk;++k) taufb[k]=taufb[k-1]-tautem;
            }
        }
        // bl_ugwp.F:943-1025: tendencies, surface sigma and mesospheric limiters.
        taup[kte+1]=taup[kte];
        for(int k=1;k<=kte;++k) {
            taud_ls[k]=1.0f*(taup[k+1]-taup[k])*g/del[k];
            taud_bl[k]=1.0f*(taufb[k+1]-taufb[k])*g/del[k];
        }
        float dtfac=1.0f;
        for(int k=1;k<kpblmax;++k) {
            if(P(k)>=0.5f*P(1)) {
                const float s=taud_ls[k]+taud_bl[k];
                if(s!=0.0f) dtfac=fminf(dtfac,fabsf(velco[k]/(dt*s)));
            } else break;
        }
        for(int k=1;k<=kte;++k) {
            float dtfac_meso=1.0f;
            if(IN(4,k)<=70.0f) {
                if(taud_ls[k]!=0.0f)
                    dtfac_meso=fminf(dtfac_meso,0.5f*fabsf(velco[k]/(dt*taud_ls[k])));
            }
            taud_ls[k]=taud_ls[k]*dtfac*dtfac_meso*ls_taper;
            taud_bl[k]=taud_bl[k]*dtfac*dtfac_meso*ls_taper;
            const float ux=taud_ls[k]*xn,vy=taud_ls[k]*yn;
            const float ub=taud_bl[k]*xn,vb=taud_bl[k]*yn;
            dudt[k]=ux+ub+dudt[k]; dvdt[k]=vy+vb+dvdt[k];
            O(2,k)=O(2,k)+ux+ub; O(3,k)=O(3,k)+vy+vb;
            dusfc=dusfc+(ux+ub)*del[k]; dvsfc=dvsfc+(vy+vb)*del[k];
            if(diags) {
                O(4,k)=ux; O(5,k)=vy; O(6,k)=ub; O(7,k)=vb;
                dus[0]=dus[0]+ux*del[k]; dvs[0]=dvs[0]+vy*del[k];
                dus[1]=dus[1]+ub*del[k]; dvs[1]=dvs[1]+vb*del[k];
            }
        }
    }
    // bl_ugwp.F:1019-1059: integrate earth-relative stresses, rotate tendencies.
    SO(0)=-inv_g*dusfc; SO(1)=-inv_g*dvsfc;
    if(diags) for(int m=0;m<4;++m) { SO(2+2*m)=-inv_g*dus[m]; SO(3+2*m)=-inv_g*dvs[m]; }
    for(int k=1;k<=kte;++k) {
        O(0,k)=IN(8,k)+dudt[k]*ca+dvdt[k]*sa;
        O(1,k)=IN(9,k)-dudt[k]*sa+dvdt[k]*ca;
    }
    #undef IN
    #undef S
    #undef P
    #undef O
    #undef SO
}
// MPAS mpas_atmphys_driver_gwdo.F:531-534, effective local grid spacing.
extern "C" __global__ void ugwp_oro_dx(const float *density,float *dx,float length,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n) dx[i]=length/gfk_pow(density[i],0.25f);
}
// MPAS mpas_atmphys_driver_gwdo.F:679-680, rounded tendency differences.
extern "C" __global__ void ugwp_oro_diff(const float *before,const float *after,float *diff,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n) diff[i]=__fsub_rn(after[i],before[i]);
}
'''


def _shared_math_source() -> str:
    """MPAS bl_ugwp.F:458-1053 math support from pinned glibc device headers."""
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    return "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("glibc_flt32.cuh", "glibc_trig_flt32.cuh")
    )


@lru_cache(maxsize=None)
def _module(device: int, levels: int):
    """MPAS bl_ugwp.F:15-1059 compilation preserving native gradual underflow."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    source = _shared_math_source() + f"\n#define UGWP_LEVELS {levels}\n" + CUDA_SOURCE
    with cp.cuda.Device(device):
        ptx, _ = compiler.compile_using_nvrtc(source, _hex_numerics.options(COMPILE_OPTIONS), None, "ugwp_oro_v842.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    return module


def _resident(name, value, shape, dtype):
    """MPAS module_bl_ugwp_gwdo.F:116-164 argument descriptor validation."""
    import cupy as cp
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a resident CuPy array")
    if value.dtype != np.dtype(dtype) or value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C {dtype} {shape}; got {value.dtype} {value.shape}")
    if value.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"{name} belongs to a different CUDA device")
    return value


def _on_stream(function):
    """MPAS module_bl_ugwp_gwdo.F:325-429 ordered call/output orchestration."""
    @wraps(function)
    def run(*args, **kwargs):
        """MPAS module_bl_ugwp_gwdo.F:325-429 ordered resident invocation."""
        import cupy as cp
        stream = kwargs.get("stream")
        with stream if stream is not None else cp.cuda.get_current_stream():
            return function(*args, **kwargs)
    return run


@_on_stream
def bl_ugwp_run(columns: Mapping[str, object], *, g, cp, rd, rv, fv, pi, deltim,
                ugwp_diags: bool = False, stream=None) -> dict[str, object]:
    """MPAS bl_ugwp.F:15-1059; module_bl_ugwp_gwdo.F:325-357,385-429.

    Returns candidate outputs without mutating PBL tendencies. Optional
    diagnostics are returned only when enabled, as in the native wrapper.
    """
    import cupy as cupy
    shape = columns["uproj"].shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 1:
        raise ValueError("UGWP requires at least two levels and one column")
    nz, nc = shape
    for name in LAYER_INPUTS:
        _resident(name, columns[name], shape, "float32")
    for name in SURFACE_INPUTS:
        _resident(name, columns[name], (nc,), "float32")
    _resident("prsi", columns["prsi"], (nz+1, nc), "float32")
    _resident("kpbl", columns["kpbl"], (nc,), "int32")
    layers = cupy.stack([columns[name] for name in LAYER_INPUTS])
    surface = cupy.stack([columns[name] for name in SURFACE_INPUTS])
    out = cupy.empty((12,nz,nc), dtype=cupy.float32)
    sout = cupy.empty((10,nc), dtype=cupy.float32)
    status = cupy.zeros((1,), dtype=cupy.int32)
    module = _module(cupy.cuda.runtime.getDevice(), nz)
    kernel = module.get_function("ugwp_oro_run")
    kernel(((nc+63)//64,), (64,),
           (layers, columns["prsi"], surface, columns["kpbl"], out, sout, status,
            *(np.float32(x) for x in (g,cp,rd,rv,fv,pi,deltim)),
            np.int32(nz),np.int32(nc),np.int32(ugwp_diags)), stream=stream)
    message = "UGWP_INVALID_KPBL: native small-scale path would read beyond layer storage"
    if stream is None:
        # Read off the step path (deferred_checks_v842); a forecast still stops on it, a replay at once.
        # THE COST THIS REMOVES: a device-to-host copy and a host sync every drag call (hrrrv5, one per step).
        from .deferred_checks_v842 import defer
        defer(status, message, error=ValueError)
    elif int(status.get(stream=stream)[0]):
        raise ValueError(message)
    count = 12 if ugwp_diags else 4
    result = {name: out[index] for index,name in enumerate(LAYER_OUTPUTS[:count])}
    count = 10 if ugwp_diags else 2
    result.update({name:sout[index] for index,name in enumerate(SURFACE_OUTPUTS[:count])})
    return result


@_on_stream
def gwdo_from_MPAS(fields: Mapping[str, object], *, len_disp, stream=None) -> dict[str, object]:
    """MPAS mpas_atmphys_driver_gwdo.F:365-459,520-549.

    len_disp must already resolve config_len_disp=0 to nominalMinDc as in
    physics initialization; layer carriers come from the common physics prep.
    """
    import cupy as cp
    dynamic = {"uproj":"u_p", "vproj":"v_p", "t1":"t_p", "q1":"qv_p",
               "prsl":"pres_hydd_p", "prslk":"pi_p", "zl":"zmid_p", "dz":"dz_p",
               "rublten":"rublten", "rvblten":"rvblten", "hpbl":"hpbl",
               "br1":"br", "xland1":"xland", "prsi":"pres2_hydd_p", "kpbl":"kpbl"}
    result = {key:fields[value] for key,value in dynamic.items()}
    nc = result["uproj"].shape[1]
    density = _resident("meshDensity",fields["meshDensity"],(nc,),"float32")
    result["sina"] = cp.zeros((nc,),dtype=cp.float32)
    result["cosa"] = cp.ones((nc,),dtype=cp.float32)
    result["dxmeter"] = cp.empty((nc,),dtype=cp.float32)
    _module(cp.cuda.runtime.getDevice(),result["uproj"].shape[0]).get_function("ugwp_oro_dx")(
        ((nc+127)//128,),(128,), (density,result["dxmeter"],np.float32(len_disp),np.int32(nc)),stream=stream)
    for scale,suffix in (("ls",""),("ss","ss")):
        result["var"+suffix] = fields["var2d"+scale]
        result["oc1"+suffix] = fields["con"+scale]
        for direction in range(1,5):
            for statistic in ("oa","ol"):
                result[f"{statistic}2d{direction}{suffix}"] = fields[f"{statistic}{direction}{scale}"]
    return result


@_on_stream
def gwdo_to_MPAS(fields: Mapping[str, object], outputs: Mapping[str, object], *, stream=None):
    """MPAS mpas_atmphys_driver_gwdo.F:604-652,667-687.

    Builds a candidate mapping including native rounded rubldiff/rvbldiff;
    rthblten passes through. The seam commits these after the slot succeeds.
    """
    import cupy as cp
    result = dict(outputs)
    for name, diff in (("rublten","rubldiff"),("rvblten","rvbldiff")):
        before = fields[name]
        after = outputs[name]
        _resident(name,after,before.shape,"float32")
        result[diff] = cp.empty_like(before)
        _module(cp.cuda.runtime.getDevice(),before.shape[0]).get_function("ugwp_oro_diff")(
            ((before.size+127)//128,),(128,),
            (before,after,result[diff],np.int32(before.size)),stream=stream)
    result["rthblten"] = outputs.get("rthblten", fields["rthblten"])
    return result


@_on_stream
def driver_gwdo(fields: Mapping[str, object], *, len_disp, g, cp, rd, rv, fv, pi,
                deltim, ugwp_diags=False, stream=None):
    """MPAS mpas_atmphys_driver_gwdo.F:753-822,829: orographic driver slice.

    p15 composes its nonstationary call after this operator when enabled.
    """
    columns = gwdo_from_MPAS(fields,len_disp=len_disp,stream=stream)
    outputs = bl_ugwp_run(columns,g=g,cp=cp,rd=rd,rv=rv,fv=fv,pi=pi,
                          deltim=deltim,ugwp_diags=ugwp_diags,stream=stream)
    return gwdo_to_MPAS(fields,outputs,stream=stream)


def bl_ugwp_init():
    """MPAS bl_ugwp.F:1062-1072: successful stateless initialization."""
    return "", 0


def bl_ugwp_final():
    """MPAS bl_ugwp.F:1075-1085: successful stateless finalization."""
    return "", 0


def bl_ugwp_timestep_init():
    """MPAS bl_ugwp.F:1088-1098: successful stateless step initialization."""
    return "", 0


def bl_ugwp_timestep_final():
    """MPAS bl_ugwp.F:1101-1111: successful stateless step finalization."""
    return "", 0


def register_backend(registry, *, get_fields, get_constants, commit_outputs,
                     get_stream=None, nonstationary=None):
    """Register f4 SlotCall launcher for MPAS mpas_atmphys_driver_gwdo.F:779-829.

    Providers belong to the common adapter: get_fields supplies the named
    resident carriers documented by gwdo_from_MPAS; get_constants supplies
    g, cp, rd, rv, fv, pi and already resolved len_disp. commit_outputs
    commits candidates into diag_physics and tend_physics. No hidden scheme
    state is retained. p15 may provide nonstationary(call, fields, outputs),
    returning its updated outputs before the final rounded tendency diff.
    """
    def launch(call):
        """MPAS mpas_atmphys_driver_gwdo.F:779-829 combined slot invocation."""
        # Defect UGWP_NONSTATIONARY_UNREGISTERED: enabling native NGW but
        # dropping module_bl_ugwp_gwdo.F:362-382 loses momentum and temperature forcing.
        ngw = bool(call.options.get("config_ngw_scheme", False))
        if ngw and nonstationary is None:
            raise RuntimeError("UGWP_NONSTATIONARY_UNREGISTERED: config_ngw_scheme requires the p15 callback")
        fields = get_fields(call)
        constants = dict(get_constants(call))
        stream = get_stream(call) if get_stream is not None else None
        columns = gwdo_from_MPAS(fields, len_disp=constants.pop("len_disp"), stream=stream)
        outputs = bl_ugwp_run(columns, **constants, deltim=call.dt,
                             ugwp_diags=bool(call.options["config_ugwp_diags"]), stream=stream)
        if ngw:
            outputs = nonstationary(call, fields, outputs)
        outputs = gwdo_to_MPAS(fields, outputs, stream=stream)
        commit_outputs(call, outputs)
        return outputs
    registry.register("gwdo", SCHEME, BACKEND, launch)


def bind_native_pools(provider, call):
    """Select persistent buffers only for the RRFS resident facade."""
    if call.slot != "gwdo" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("UGWP binder requires its registered orographic slot")
    from .rrfs.resident_gwdo_rrfs import resident_slot
    slot = resident_slot(provider, call)
    if slot is not None:
        return {"ugwp_oro_resident": slot}
    return _bind_native_pools(provider, call)


def _bind_native_pools(provider, call):
    """MPAS mpas_atmphys_driver_gwdo.F:365-459,520-549,604-687.

    Bind f4's NativeDevicePoolViews transaction. The native provider must
    export get_real_config(name) from the initialized configs pool so len_disp
    includes native's nominalMinDc resolution. No host expression recreates
    that resolved value or compiled physics constants.
    """
    if call.slot != "gwdo" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("UGWP binder requires its registered orographic slot")
    nz = provider.session.layout.dimensions["nVertLevels"]
    fields = {name: provider.prepared(name)[:nz] for name in
              ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pi_p", "zmid_p", "dz_p")}
    fields["pres2_hydd_p"] = provider.prepared("pres2_hydd_p")[:nz+1]
    for name in ("rublten", "rvblten", "rthblten"):
        fields[name] = provider.field("tend_physics", name)
    for name in ("kpbl", "hpbl", "br"):
        fields[name] = provider.field("diag_physics", name)
    fields["xland"] = provider.field("sfc_input", "xland")
    fields["meshDensity"] = provider.field("mesh", "meshDensity")
    for scale in ("ls", "ss"):
        for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
            fields[name+scale] = provider.field("sfc_input", name+scale)
    constants = {name: provider.native.get_constant(native) for name, native in
                 (("g", "gravity"), ("cp", "cp"), ("rd", "R_d"),
                  ("rv", "R_v"), ("fv", "ep_1"), ("pi", "pii"))}
    constants["len_disp"] = provider.native.get_real_config("config_len_disp")
    outputs = ("rublten", "rvblten", "rthblten", "rubldiff", "rvbldiff",
               "dtaux3d", "dtauy3d", "dusfcg", "dvsfcg")
    if call.options["config_ugwp_diags"]:
        outputs += LAYER_OUTPUTS[4:] + SURFACE_OUTPUTS[2:]
        if call.options.get("config_ngw_scheme", False):
            outputs += ("dudt_ngw", "dvdt_ngw", "dtdt_ngw")
    targets = {name: provider.field("tend_physics" if name in
               ("rublten", "rvblten", "rthblten") else "diag_physics", name)
               for name in outputs}
    prepared_targets = {name: provider.prepared(name+"_p") for name in outputs
                        if name not in ("rubldiff", "rvbldiff")}
    from_targets = {name: provider.prepared(name+"_p") for name in
                    ("rublten", "rvblten", "rthblten", "kpbl", "hpbl", "br", "xland")}
    for scale in ("ls", "ss"):
        for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
            from_targets[name+scale] = provider.prepared(name+scale+"_p")
    return {"ugwp_oro_v842": dict(fields=fields, constants=constants,
            targets=targets, prepared_targets=prepared_targets,
            from_targets=from_targets, orientation=tuple(provider.prepared(name)
            for name in ("sina_p", "cosa_p", "dx_p")),
            nonstationary=getattr(provider, "ugwp_nonstationary", None))}


def run_slot(call):
    """MPAS mpas_atmphys_driver_gwdo.F:753-829 whole resident slot boundary.

    The f4 transaction commits pool/global storage afterward. Assignment only
    copies device words here; all driver arithmetic uses the proven kernels.
    Prepared globals retain top storage; the provider retains native halos.
    """
    if call.slot != "gwdo" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("UGWP callback requires its registered orographic slot")
    if "ugwp_oro_resident" in call.pools:
        return call.pools["ugwp_oro_resident"].run(call)
    packet = call.pools["ugwp_oro_v842"]
    fields, constants = packet["fields"], dict(packet["constants"])
    ngw = bool(call.options.get("config_ngw_scheme", False))
    nonstationary = packet.get("nonstationary")
    # UGWP_NONSTATIONARY_UNREGISTERED: wrapper362-382 would lose native forcing.
    if ngw and nonstationary is None:
        raise RuntimeError("UGWP_NONSTATIONARY_UNREGISTERED: config_ngw_scheme requires the p15 callback")
    columns = gwdo_from_MPAS(fields, len_disp=constants.pop("len_disp"))
    for name, target in packet["from_targets"].items():
        value = fields[name]
        target[:value.shape[0]] = value
    for name, target in zip(("sina", "cosa", "dxmeter"), packet["orientation"]):
        target[...] = columns[name]
    outputs = bl_ugwp_run(columns, **constants, deltim=call.dt,
                         ugwp_diags=bool(call.options["config_ugwp_diags"]))
    if ngw:
        outputs = nonstationary(call, fields, outputs)
    outputs = gwdo_to_MPAS(fields, outputs)
    for name, target in packet["targets"].items():
        target[...] = outputs[name]
    for name, target in packet["prepared_targets"].items():
        value = outputs[name]
        target[:value.shape[0]] = value
    return outputs
