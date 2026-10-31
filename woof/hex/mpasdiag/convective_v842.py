# Numerical CUDA derivative of MPAS-A v8.4.2; original source remains unchanged.
# Copyright (c) 2016, Los Alamos National Security, LLC (LANS)
# and the University Corporation for Atmospheric Research (UCAR).
# MPAS distribution license:
# Copyright (c) 2013-2019,  Los Alamos National Security, LLC (LANS) (Ocean: LA-CC-13-047;
# Land Ice: LA-CC-13-117) and the University Corporation for Atmospheric Research (UCAR).
#
# All rights reserved.
#
# LANS is the operator of the Los Alamos National Laboratory under Contract No.
# DE-AC52-06NA25396 with the U.S. Department of Energy.  UCAR manages the National
# Center for Atmospheric Research under Cooperative Agreement ATM-0753581 with the
# National Science Foundation.  The U.S. Government has rights to use, reproduce,
# and distribute this software.  NO WARRANTY, EXPRESS OR IMPLIED IS OFFERED BY
# LANS, UCAR OR THE GOVERNMENT AND NONE OF THEM ASSUME ANY LIABILITY FOR THE USE
# OF THIS SOFTWARE.  If software is modified to produce derivative works, such
# modified software should be clearly marked, so as not to confuse it with the
# version available from LANS and UCAR.
#
# Additionally, redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1) Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2) Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3) None of the names of LANS, UCAR or the names of its contributors, if any, may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Complete MPAS-A v8.4.2 convective diagnostic driver on CUDA.

Inputs are resident float32 C arrays (cell, level), zgrid/w include the top
interface, and topology uses MPAS's ONE-based indices including nCells+1.
The caller supplies output stream decisions; update is called after each
step, compute before output and reset AFTER that output has been consumed.
Outputs include allocated halos, while compute/update only change solve cells.

The native module never assigns lcl or lfc. Their incoming words are retained.
No host processing, stream scheduling, thermodynamic alternatives or defaults
are implemented here. MPAS's compile-time source=2 and adiabat=1 are retained;
the inactive source/adiabat branches are eliminated just as in the GNU build.
"""
from __future__ import annotations
from woof.hex.device_memory_release import shared_card_memory
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SOURCE_SHA256 = '45dfff935a361572d12088334b4a7bf64815278ed6f035ff64c3a976fec4046b'

FIELDS = (
    'cape', 'cin', 'lcl', 'lfc', 'srh_0_1km', 'srh_0_3km',
    'uzonal_surface', 'uzonal_1km', 'uzonal_6km',
    'umeridional_surface', 'umeridional_1km', 'umeridional_6km',
    'temperature_surface', 'dewpoint_surface',
    'updraft_helicity_max', 'w_velocity_max', 'wind_speed_level1_max',
)
MAXIMA = FIELDS[14:]

# A tiny arithmetic type pins every source-level RKIND operation. Shared
# engine libm owns all elementary functions. The source is also compiled on
# the host by replay tests, with this same arithmetic and the live libm.
CUDA_SOURCE = r'''
struct R {
    float v;
    __device__ R() {}
    __device__ R(float x): v(x) {}
    __device__ R& operator+=(R x) { v=__fadd_rn(v,x.v); return *this; }
};
__device__ R operator+(R a,R b) {return __fadd_rn(a.v,b.v);}
__device__ R operator-(R a,R b) {return __fsub_rn(a.v,b.v);}
__device__ R operator*(R a,R b) {return __fmul_rn(a.v,b.v);}
__device__ R operator/(R a,R b) {return __fdiv_rn(a.v,b.v);}
__device__ R operator-(R a) {return -a.v;}
__device__ bool operator<(R a,R b) {return a.v<b.v;}
__device__ bool operator>(R a,R b) {return a.v>b.v;}
__device__ bool operator<=(R a,R b) {return a.v<=b.v;}
__device__ bool operator>=(R a,R b) {return a.v>=b.v;}
__device__ R rmax(R a,R b) {return a.v>b.v?a:b;}
__device__ R rmin(R a,R b) {return a.v<b.v?a:b;}
__device__ R rabs(R a) {return fabsf(a.v);}
__device__ R rlog(R a) {return gfk_log(a.v);}
__device__ R rexp(R a) {return gfk_exp_fma(a.v);}
__device__ R rpow(R a,R b) {return gfk_pow(a.v,b.v);}
__device__ R rsqrt(R a) {return __fsqrt_rn(a.v);}

// mpas_convective_diagnostics.F:528-544
__device__ R column_height_value(const float* x,const float* z,R zi,int n) {
    int kz=0;
    for(int k=0;k<n;k++) if(R(z[k])<=zi) kz=k;
    kz=kz<n-1?kz:n-2;
    R wz=(R(z[kz+1])-zi)/(R(z[kz+1])-R(z[kz]));
    R wzp1=R(1.0f)-wz;
    return wz*R(x[kz])+wzp1*R(x[kz+1]);
}
// mpas_convective_diagnostics.F:548-565
__device__ R integral_zstaggered(const float* x,const float* z,R bot,R top,int n) {
    R result=0.0f;
    for(int k=0;k<n;k++) {
        R zb=rmax(z[k],bot),zt=rmin(z[k+1],top);
        result+=R(x[k])*rmax(0.0f,zt-zb);
    }
    return result;
}
// mpas_convective_diagnostics.F:569-589
__device__ R integral_zpoint(const float* x,const float* z,R bot,R top,int n) {
    R result=0.0f;
    for(int k=0;k<n-1;k++) {
        R zb=rmax(z[k],bot),zt=rmin(z[k+1],top);
        R dz=rmax(0.0f,zt-zb);
        R mid=(R(0.5f)*(zt+zb)-R(z[k]))/(R(z[k+1])-R(z[k]));
        R val=R(x[k])+(R(x[k+1])-R(x[k]))*mid;
        result+=dz*val;
    }
    return result;
}
// mpas_convective_diagnostics.F:1043-1054
__device__ R getqvs(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(17.67f)*(t-R(273.15f))/(t-R(29.65f)));
    return eps*es/(p-es);
}
// mpas_convective_diagnostics.F:1060-1071
__device__ R getqvi(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(21.8745584f)*(t-R(273.15f))/(t-R(7.66f)));
    return eps*es/(p-es);
}
// mpas_convective_diagnostics.F:1077-1093
__device__ R getthe(R p,R t,R td,R q) {
    R tlcl;
    if((td-t)>=R(-0.1f)) tlcl=t;
    else tlcl=R(56.0f)+R(1.0f)/(R(1.0f)/(td-R(56.0f))+R(0.00125f)*rlog(t/td));
    return t*rpow(R(100000.0f)/p,R(0.2854f)*(R(1.0f)-R(0.28f)*q))
        *rexp((R(3376.0f)/tlcl-R(2.54f))*q*(R(1.0f)+R(0.81f)*q));
}
// mpas_convective_diagnostics.F:595-1037
// source=2 (most unstable), adiabat=1 (pseudoadiabatic, liquid only).
__device__ void getcape(int nk,const float* pin,const float* tin,const float* tdin,
                       float* ws,float& cape_out,float& cin_out) {
    float *p=ws,*t=p+nk,*td=t+nk,*pi=td+nk,*q=pi+nk,*th=q+nk,*thv=th+nk;
    const R rd=287.04f,rv=461.5f,cp=1005.7f,g=9.81f,cpv=1875.0f,cpl=4190.0f,cpi=2118.636f;
    const R rp00=1.0f/100000.0f,reps=461.5f/287.04f,rddcp=287.04f/1005.7f,cpdg=1005.7f/9.81f;
    const R lv1=2501000.0f+(4190.0f-1875.0f)*273.15f,lv2=4190.0f-1875.0f;
    const R ls1=2836017.0f+(2118.636f-1875.0f)*273.15f,ls2=2118.636f-1875.0f;
    for(int k=0;k<nk;k++) {
        p[k]=(R(100.0f)*R(pin[k])).v;
        t[k]=(R(273.15f)+R(tin[k])).v;
        td[k]=(R(273.15f)+R(tdin[k])).v;
        pi[k]=rpow(R(p[k])*rp00,rddcp).v;
        q[k]=getqvs(p[k],td[k]).v;
        th[k]=(R(t[k])/R(pi[k])).v;
        thv[k]=(R(th[k])*(R(1.0f)+reps*R(q[k]))/(R(1.0f)+R(q[k]))).v;
    }
    int kmax=0;
    if(R(p[0])>=R(50000.0f)) {
        R maxthe=0.0f;
        for(int k=0;k<nk;k++) if(R(p[k])>=R(50000.0f)) {
            R the=getthe(p[k],t[k],td[k],q[k]);
            if(the>maxthe) {maxthe=the;kmax=k;}
        }
    }
    R narea=0.0f,cape=0.0f,cin=0.0f;
    R th2=th[kmax],pi2=pi[kmax],p2=p[kmax],t2=t[kmax],thv2=thv[kmax],qv2=q[kmax],b2=0.0f;
    R ql2=0.0f,qi2=0.0f,qt=qv2;
    for(int k=kmax+1;k<nk;k++) {
        R b1=b2;
        R dp=R(p[k-1])-R(p[k]);
        int nloop=1;
        if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
        for(int n=0;n<nloop;n++) {
            R p1=p2,t1=t2,th1=th2,qv1=qv2,ql1=ql2,qi1=qi2;
            p2=p2-dp;
            pi2=rpow(p2*rp00,rddcp);
            R thlast=th1;
            int i=0;
            while(true) {
                i++;
                t2=thlast*pi2;
                // Preserve the source's zero-weight ice saturation term.
                qv2=rmin(qt,R(1.0f)*getqvs(p2,t2)+R(0.0f)*getqvi(p2,t2));
                qi2=rmax(R(0.0f)*(qt-qv2),0.0f);
                ql2=rmax(qt-qv2-qi2,0.0f);
                R tbar=R(0.5f)*(t1+t2),qvbar=R(0.5f)*(qv1+qv2);
                R qlbar=R(0.5f)*(ql1+ql2),qibar=R(0.5f)*(qi1+qi2);
                R lhv=lv1-lv2*tbar,lhs=ls1-ls2*tbar;
                R rm=rd+rv*qvbar,cpm=cp+cpv*qvbar+cpl*qlbar+cpi*qibar;
                th2=th1*rexp(lhv*(ql2-ql1)/(cpm*tbar)+lhs*(qi2-qi1)/(cpm*tbar)
                             +(rm/cpm-rd/cp)*rlog(p2/p1));
                // Native returns the accumulated CAPE/CIN on nonconvergence.
                if(i>100) {cape_out=cape.v;cin_out=cin.v;return;}
                if(rabs(th2-thlast)>R(0.0002f)) thlast=thlast+R(0.3f)*(th2-thlast);
                else break;
            }
            qt=qv2;ql2=0.0f;qi2=0.0f;
        }
        thv2=th2*(R(1.0f)+reps*qv2)/(R(1.0f)+qv2+ql2+qi2);
        b2=g*(thv2-R(thv[k]))/R(thv[k]);
        R dz=-cpdg*R(0.5f)*(R(thv[k])+R(thv[k-1]))*(R(pi[k])-R(pi[k-1]));
        R parea;
        if(b2>=R(0.0f)&&b1<R(0.0f)) {
            R frac=b2/(b2-b1);
            parea=R(0.5f)*b2*dz*frac;
            narea=narea-R(0.5f)*b1*dz*(R(1.0f)-frac);
            cin+=narea;narea=0.0f;
        } else if(b2<R(0.0f)&&b1>R(0.0f)) {
            R frac=b1/(b1-b2);
            parea=R(0.5f)*b1*dz*frac;
            narea=-R(0.5f)*b2*dz*(R(1.0f)-frac);
        } else if(b2<R(0.0f)) {
            parea=0.0f;narea=narea-R(0.5f)*dz*(b1+b2);
        } else {parea=R(0.5f)*dz*(b1+b2);narea=0.0f;}
        cape+=rmax(0.0f,parea);
        if(R(p[k])<=R(10000.0f)&&b2<R(0.0f)) break;
    }
    cape_out=cape.v;cin_out=cin.v;
}
// mpas_convective_diagnostics.F:227-487
extern "C" __global__ void compute_diag(int nc,int ns,int nl,unsigned mask,float rvord,
    const float* height,const float* u,const float* v,const float* theta,
    const float* qv,const float* exner,const float* pb,const float* pp,float* out,float* work) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=ns) return;
    const float *z=height+c*(nl+1),*uc=u+c*nl,*vc=v+c*nl;
    float* ws=work+c*(17*nl+3);
    float *temp=ws,*dew=temp+nl,*zp=dew+nl,*zr=zp+nl,*dudz=zr+nl+1,*dvdz=dudz+nl;
    float *srh=dvdz+nl,*pin=srh+nl+1,*tin=pin+nl,*tdin=tin+nl,*cape_ws=tdin+nl;
    for(int k=0;k<nl;k++) {
        int ix=c*nl+k;
        temp[k]=(R(theta[ix])/(R(1.0f)+R(rvord)*R(qv[ix]))*R(exner[ix])).v;
        R evp=R(0.01f)*(R(pb[ix])+R(pp[ix]))*R(qv[ix])/(R(qv[ix])+R(0.622f));
        evp=rmax(evp,1.0e-8f);
        R lg=rlog(evp/R(6.112f));
        dew[k]=(R(243.5f)*lg/(R(17.67f)-lg)+R(273.15f)).v;
        zp[k]=(R(0.5f)*(R(z[k])+R(z[k+1]))-R(z[0])).v;
    }
    for(int k=0;k<=nl;k++) zr[k]=(R(z[k])-R(z[0])).v;
    out[6*nc+c]=uc[0];out[9*nc+c]=vc[0];out[12*nc+c]=temp[0];out[13*nc+c]=dew[0];
    if(mask&(1u<<7)) out[7*nc+c]=column_height_value(uc,zp,1000.0f,nl).v;
    if(mask&(1u<<8)) out[8*nc+c]=column_height_value(uc,zp,6000.0f,nl).v;
    if(mask&(1u<<10)) out[10*nc+c]=column_height_value(vc,zp,1000.0f,nl).v;
    if(mask&(1u<<11)) out[11*nc+c]=column_height_value(vc,zp,6000.0f,nl).v;
    if(mask&((1u<<4)|(1u<<5))) {
        R ub=uc[0],vb=vc[0];
        if(R(0.0f)>R(zp[0])) {ub=column_height_value(uc,zp,0.0f,nl);vb=column_height_value(vc,zp,0.0f,nl);}
        R ut=column_height_value(uc,zp,6000.0f,nl),vt=column_height_value(vc,zp,6000.0f,nl);
        R us=ut-ub,vs=vt-vb;
        R um=integral_zstaggered(uc,zr,0.0f,6000.0f,nl)/R(6000.0f);
        R vm=integral_zstaggered(vc,zr,0.0f,6000.0f,nl)/R(6000.0f);
        R sm=rmax(0.0001f,rsqrt(us*us+vs*vs));
        R ustorm=um+R(7.5f)*vs/sm,vstorm=vm-R(7.5f)*us/sm;
        for(int k=1;k<nl-1;k++) {
            R dz=R(0.5f)*(R(z[k+1])-R(z[k-1]));
            dudz[k]=((R(uc[k])-R(uc[k-1]))/dz).v;
            dvdz[k]=((R(vc[k])-R(vc[k-1]))/dz).v;
        }
        dudz[0]=dudz[1];dvdz[0]=dvdz[1];dudz[nl-1]=dudz[nl-2];dvdz[nl-1]=dvdz[nl-2];
        for(int k=1;k<nl;k++) srh[k]=(-(R(0.5f)*(R(uc[k])+R(uc[k-1]))-ustorm)*R(dvdz[k])
                                  +(R(0.5f)*(R(vc[k])+R(vc[k-1]))-vstorm)*R(dudz[k])).v;
        srh[0]=(-(R(uc[0])-ustorm)*R(dvdz[0])+(R(vc[0])-vstorm)*R(dudz[0])).v;
        srh[nl]=srh[nl-1];
        for(int k=0;k<=nl;k++) srh[k]=rmax(0.0f,srh[k]).v;
        if(mask&(1u<<4)) out[4*nc+c]=integral_zpoint(srh,zr,0.0f,1000.0f,nl+1).v;
        if(mask&(1u<<5)) out[5*nc+c]=integral_zpoint(srh,zr,0.0f,3000.0f,nl+1).v;
    }
    if(mask&3u) {
        for(int k=0;k<nl;k++) {
            int ix=c*nl+k;
            pin[k]=((R(pp[ix])+R(pb[ix]))/R(100.0f)).v;
            tin[k]=(R(temp[k])-R(273.15f)).v;tdin[k]=(R(dew[k])-R(273.15f)).v;
        }
        getcape(nl,pin,tin,tdin,cape_ws,out[c],out[nc+c]);
    }
}
// mpas_convective_diagnostics.F:99-198
// One thread per level performs the native ordered scatter. No atomics.
extern "C" __global__ void vertex_helicity(int nc,int nl,int nv,int degree,
    const int* cov,const float* kite,const float* vort,float* uh) {
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if(k>=nl) return;
    for(int c=0;c<=nc;c++) uh[c*nl+k]=0.0f;
    for(int i=0;i<nv;i++) for(int j=0;j<degree;j++) {
        int ix=i*degree+j,c=cov[ix]-1;
        uh[c*nl+k]=(R(uh[c*nl+k])+R(kite[ix])*R(vort[i*nl+k])).v;
    }
}
// mpas_convective_diagnostics.F:99-198, as a gather. One thread per
// (cell,level) adds that cell's kite*vorticity terms in the order native's
// ordered scatter adds them (offsets/entries list each cell's (vertex,i)
// slots in ascending scatter order, built once per mesh), starting from the
// same zero word: the same additions in the same order, so the same words,
// without the serial 55-thread walk over every vertex.
extern "C" __global__ void vertex_helicity_gather(int nc,int nl,int degree,
    const int* offsets,const int* entries,const float* kite,const float* vort,float* uh) {
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=(long long)(nc+1)*nl) return;
    int c=(int)(t/nl),k=(int)(t%nl);
    R s=0.0f;
    for(int e=offsets[c];e<offsets[c+1];e++) {
        int ix=entries[e];
        s=s+R(kite[ix])*R(vort[(long long)(ix/degree)*nl+k]);
    }
    uh[(long long)c*nl+k]=s.v;
}
extern "C" __global__ void update_diag(int nc,int ns,int nl,unsigned mask,
    const float* z,const float* w,const float* u,const float* v,const float* area,float* uh,float* out) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=ns) return;
    const float *wc=w+c*(nl+1),*zc=z+c*(nl+1);
    if(mask&(1u<<15)) {
        // GNU's scalar MAXVAL/interval MAX keeps the accumulator on ties.
        // MAXVAL finds its first non-NaN operand before folding later ones.
        int k=0;while(k<nl && wc[k]!=wc[k]) ++k;
        if(k<nl) {
            R wm=wc[k];for(++k;k<nl;k++) wm=rmax(wc[k],wm);
            out[15*nc+c]=rmax(wm,out[15*nc+c]).v;
        } // An all-NaN MAXVAL is ignored by GNU's final MAX with the prior.
    }
    if(mask&(1u<<16)) {
        R uu=u[c*nl],vv=v[c*nl];
        out[16*nc+c]=rmax(rsqrt(uu*uu+vv*vv),out[16*nc+c]).v;
    }
    if(mask&(1u<<14)) {
        R result=0.0f;
        for(int k=0;k<nl;k++) {
            R h=rmax(R(0.5f)*(R(wc[k])+R(wc[k+1])),0.0f)*rmax(R(uh[c*nl+k])/R(area[c]),0.0f);
            R zb=rmax(R(zc[k])-R(zc[0]),2000.0f),zt=rmin(R(zc[k+1])-R(zc[0]),5000.0f);
            result+=h*rmax(0.0f,zt-zb);
        }
        out[14*nc+c]=rmax(result,out[14*nc+c]).v;
    }
}
// mpas_convective_diagnostics.F:501-525
extern "C" __global__ void reset_diag(int nc,unsigned mask,float* out) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    for(int j=14;j<17;j++) if(mask&(1u<<j)) out[j*nc+c]=0.0f;
}
'''


def _mask(fields: Iterable[str]) -> int:
    names = frozenset(fields)
    unknown = names.difference(FIELDS)
    if unknown:
        raise ValueError(f'unknown convective diagnostics: {sorted(unknown)}')
    return sum(1 << FIELDS.index(name) for name in names)


def device_source():
    """Assemble device code using the pinned engine's shared maths headers."""
    import woof

    root = Path(woof.__file__).resolve().parent / 'core' / 'kernels'
    # Reuse the engine's shared implementations, including the reference
    # host's explicitly fused expf selection; do not copy or rewrite libm.
    headers = '\n'.join((root / name).read_text(encoding='utf-8') for name in
                        ('glibc_flt32.cuh', 'flt32_expf_fma.cuh'))
    return headers + '\n' + CUDA_SOURCE


@lru_cache(maxsize=None)
def _module(device_id, artifact=None):
    import cupy as cp
    from cupy.cuda import compiler
    # RawModule's cache compiler appends -ftz=true. Direct NVRTC preserves
    # native subnormal interval maxima; test_subnormal_cuda guards this.
    with cp.cuda.Device(device_id):
        options = _hex_numerics.options(('--std=c++17', '--fmad=false', '--ftz=false',
                   '--prec-div=true', '--prec-sqrt=true'))
        arch = cp.cuda.Device(device_id).compute_capability
        source = device_source()
        if artifact:
            # A CPU-only NVRTC image built through the build queue avoids
            # holding a card while queued for a compiler. Authenticate every input so a
            # stale image cannot silently execute a different port or FTZ.
            import hashlib
            import json
            path = Path(artifact)
            record = json.loads(path.with_suffix(path.suffix+'.json').read_text())
            binary = path.read_bytes()
            kind = record.get('image_kind', 'ptx')
            target = {'ptx': 'compute_', 'cubin': 'sm_'}.get(kind)
            if (target is None
                    or (kind == 'cubin' and not binary.startswith(b'\x7fELF'))
                    or record.get('source_sha256') != hashlib.sha256(source.encode()).hexdigest()
                    or record.get('image_sha256', record.get('ptx_sha256')) != hashlib.sha256(binary).hexdigest()
                    or record.get('compile_options') != list(options)+['--gpu-architecture='+target+arch]):
                raise ValueError('convective CUDA artifact source, options or bytes differ')
        else:
            binary, _ = compiler.compile_using_nvrtc(source, options=options, arch=arch)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def helicity_plan(cov, n_cells, vertex_order=None):
    """CSR of each cell's cellsOnVertex slots, in native scatter order.

    ``cov`` is resident int32 (nVertices, vertexDegree), one-based with the
    nCells+1 sentinel. Native adds slot ix = iVertex*degree+i in ascending
    ix (or in ``vertex_order``'s vertex order for a same-host rank). Returns
    (offsets[n_cells+2], entries[nnz]) int32 device arrays.
    """
    import cupy as cp
    nv, degree = cov.shape
    if vertex_order is None:
        sequence = cp.arange(nv*degree, dtype=cp.int64)
    else:
        sequence = (cp.asarray(vertex_order, dtype=cp.int64)[:, None]*degree
                    + cp.arange(degree, dtype=cp.int64)[None, :]).ravel()
    cells = cov.ravel()[sequence].astype(cp.int64) - 1
    if cells.size and (int(cells.min()) < 0 or int(cells.max()) > n_cells):
        raise ValueError('cellsOnVertex must hold one-based indices 1..nCells+1')
    # Unique composite keys: cell first, then position in scatter order.
    order = cp.argsort(cells*cells.size + cp.arange(cells.size, dtype=cp.int64))
    entries = cp.ascontiguousarray(sequence[order].astype(cp.int32))
    counts = cp.bincount(cells, minlength=n_cells+1)
    offsets = cp.zeros(n_cells+2, dtype=cp.int32)
    offsets[1:] = cp.cumsum(counts).astype(cp.int32)
    return offsets, entries


class ConvectiveDiagnostics:
    """Stream-driven driver, preserving the MPAS output/update/reset boundary."""

    def __init__(self, outputs, n_cells_solve: int, n_levels: int,
                 output_stream_fields: Iterable[str]):
        """mpas_convective_diagnostics.F:53-85 (setup).

        ``outputs`` is a writable resident C float32 array (17, nCells).
        It must contain native initialization/restart values on entry.
        No implicit initialization occurs; lcl/lfc always retain those words.
        """
        import cupy as cp
        self.outputs = outputs
        self.n_cells = outputs.shape[1] if outputs.ndim == 2 else 0
        self.n_solve = int(n_cells_solve)
        self.n_levels = int(n_levels)
        self._array(outputs, (len(FIELDS), self.n_cells), 'outputs')
        if not 0 <= self.n_solve <= self.n_cells or self.n_levels < 3:
            # Native SRH indexes level 2 and nLevels-1; reject invalid input
            # dimensions before a kernel can read outside allocated storage.
            raise ValueError('require 0 <= nCellsSolve <= nCells and nLevels >= 3')
        self.update_mask = _mask(output_stream_fields) & _mask(MAXIMA)
        # compute uses 17 nl-sized arrays and two extra interface entries.
        self._work = (None if shared_card_memory() else
                      cp.empty((self.n_solve, 17*self.n_levels+3), dtype=cp.float32))
        self._uh = None
        # A same-host rank's local vertices in global-id order (None: local
        # order). The helicity gather adds each cell's terms in this order.
        self.vertex_order = None
        self._helicity_plan = None

    @staticmethod
    def _array(array, shape, name, dtype=np.float32):
        import cupy as cp
        if not isinstance(array, cp.ndarray) or array.dtype != np.dtype(dtype):
            raise TypeError(f'{name} must be a resident {np.dtype(dtype)} CUDA array')
        if array.shape != shape or not array.flags.c_contiguous:
            raise ValueError(f'{name} must be C contiguous with shape {shape}')
        return array

    def _inputs(self, arrays: Mapping[str, object], names):
        shapes = {'zgrid': (self.n_cells, self.n_levels+1),
                  'w': (self.n_cells, self.n_levels+1), 'areaCell': (self.n_cells,)}
        return tuple(self._array(arrays[name], shapes.get(name, (self.n_cells, self.n_levels)), name)
                     for name in names)

    def _launch(self, name, count, args):
        if count:
            import cupy as cp
            import os
            _module(cp.cuda.Device().id, os.environ.get('MPAS_CONVECTIVE_IMAGE') or
                    os.environ.get('MPAS_CONVECTIVE_PTX')).get_function(name)(
                ((count+127)//128,), (128,), args)

    def _helicity_plan_for(self, cov, n_cells):
        """Per-cell (vertex,i) slots in native scatter order; built once per mesh.

        MPAS's mesh pool connectivity is fixed for a run, so the plan is kept
        for the driver's lifetime (keyed on its shape and vertex order).
        """
        key = (cov.shape, int(n_cells), id(self.vertex_order))
        if self._helicity_plan is None or self._helicity_plan[0] != key:
            self._helicity_plan = (key, helicity_plan(cov, n_cells, self.vertex_order))
        return self._helicity_plan[1]

    def compute(self, arrays: Mapping[str, object], fields_will_be_written: Iterable[str],
                *, rvord: float = float(np.float32(461.6)/np.float32(287.0))):
        """mpas_convective_diagnostics.F:227-487; compute before stream output.

        qv is a view of the native moisture slot materialized by the caller.
        relhum is unused by the pinned native routine and is not required.
        ``rvord`` can be supplied from MPAS's selected constants package.
        """
        mask = _mask(fields_will_be_written) & ((1 << 14)-1)
        if not mask or not self.n_solve:
            return self.outputs
        args = self._inputs(arrays, ('zgrid', 'uReconstructZonal', 'uReconstructMeridional',
                                    'theta_m', 'qv', 'exner', 'pressure_base', 'pressure_p'))
        work = self._work
        if work is None:
            import cupy as cp
            work = cp.empty((self.n_solve, 17*self.n_levels+3), dtype=cp.float32)
        self._launch('compute_diag', self.n_solve,
                     (np.int32(self.n_cells), np.int32(self.n_solve), np.int32(self.n_levels),
                      np.uint32(mask), np.float32(rvord), *args, self.outputs, work))
        return self.outputs

    def update(self, arrays: Mapping[str, object]):
        """mpas_convective_diagnostics.F:99-198; update after each timestep.

        cellsOnVertex is resident C int32 (nVertices, vertexDegree), with
        valid one-based indices 1..nCells+1, including the MPAS sentinel.
        kiteAreasOnVertex has that same shape; vorticity is (nVertices,nLevels).
        """
        import cupy as cp
        if not self.update_mask or not self.n_solve:
            return self.outputs
        # Only demand data that the selected native branch reads.
        dummy = self.outputs
        z = w = u = v = area = dummy
        if self.update_mask & (1 << 14):
            z, w, area = self._inputs(arrays, ('zgrid', 'w', 'areaCell'))
            cov = arrays['cellsOnVertex']
            if cov.ndim != 2 or cov.shape[1] < 1:
                raise ValueError('cellsOnVertex must have shape (nVertices, vertexDegree)')
            nv, degree = cov.shape
            self._array(cov, (nv, degree), 'cellsOnVertex', np.int32)
            kite = self._array(arrays['kiteAreasOnVertex'], (nv, degree), 'kiteAreasOnVertex')
            vort = self._array(arrays['vorticity'], (nv, self.n_levels), 'vorticity')
            if self._uh is None:
                self._uh = cp.empty((self.n_cells+1, self.n_levels), dtype=cp.float32)
            offsets, entries = self._helicity_plan_for(cov, self.n_cells)
            self._launch('vertex_helicity_gather', (self.n_cells+1)*self.n_levels,
                         (np.int32(self.n_cells), np.int32(self.n_levels), np.int32(degree),
                          offsets, entries, kite, vort, self._uh))
        if self.update_mask & (1 << 15):
            w, = self._inputs(arrays, ('w',))
        if self.update_mask & (1 << 16):
            u, v = self._inputs(arrays, ('uReconstructZonal', 'uReconstructMeridional'))
        self._launch('update_diag', self.n_solve,
                     (np.int32(self.n_cells), np.int32(self.n_solve), np.int32(self.n_levels),
                      np.uint32(self.update_mask), z, w, u, v, area,
                      self._uh if self._uh is not None else dummy, self.outputs))
        return self.outputs

    def reset(self, fields_will_be_written: Iterable[str]):
        """mpas_convective_diagnostics.F:501-525; reset after output consumption.

        Reset decisions use this event's stream fields, independently of the
        setup inclusion flags, and reset all allocated cells including halos.
        """
        mask = _mask(fields_will_be_written) & _mask(MAXIMA)
        if mask:
            self._launch('reset_diag', self.n_cells,
                         (np.int32(self.n_cells), np.uint32(mask), self.outputs))
        return self.outputs


class ConvectiveDiagnosticsModule:
    """Pool adapter for the diagnostics manager supplied by o2-diag-isobaric.

    Pool fields use native (level,cell) and (scalar,level,cell) resident arrays.
    Transposition/copying runs on the device. Output pool arrays retain their
    identity, so stream writers and restart services can keep their views.
    """

    def setup(self, pools, clock, utils):
        """mpas_convective_diagnostics.F:53-85; bind pools and stream queries."""
        import cupy as cp
        from .utils import Direction
        self.pools = pools
        self.mesh, self.state, self.diag = (pools[name] for name in ('mesh', 'state', 'diag'))
        self.clock, self.utils = clock, utils
        nc = int(self.mesh['nCells'])
        self.n_alloc = self.diag[FIELDS[0]].size
        if self.n_alloc not in (nc, nc+1):
            raise ValueError('diagnostic allocation must include nCells, optionally its sentinel')
        values = [ConvectiveDiagnostics._array(self.diag[name], (self.n_alloc,), name) for name in FIELDS]
        included = [name for name in MAXIMA
                    if utils.stream_inclusion_count(name, direction=Direction.OUTPUT)>0]
        self.driver = ConvectiveDiagnostics(cp.stack(values), self.mesh['nCellsSolve'],
                                            self.mesh['nVertLevels'], included)

    def _column(self, value, name, levels):
        import cupy as cp
        if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
            raise TypeError(f'{name} must be resident native RKIND')
        if value.ndim != 2 or value.shape[0] != levels or value.shape[1] < self.n_alloc:
            raise ValueError(f'{name} requires native (level,allocatedCell) shape')
        return cp.ascontiguousarray(value[:, :self.n_alloc].T)

    def _arrays(self, names):
        import cupy as cp
        # The driver may rotate the entire native level-1 state pool.
        m,s,d = self.mesh,self.pools['state'],self.diag
        nl = self.driver.n_levels
        arrays={}
        for name in names:
            if name in ('zgrid','w'):
                value=m[name] if name=='zgrid' else s[name]
                arrays[name]=self._column(value,name,nl+1)
            elif name=='qv':
                scalars=s['scalars']
                if not isinstance(scalars,cp.ndarray) or scalars.dtype != cp.float32 or scalars.ndim!=3:
                    raise TypeError('scalars must be resident (scalar,level,cell) RKIND')
                iqv=int(s['index_qv'])-1
                if not 0<=iqv<scalars.shape[0]:
                    raise ValueError('index_qv must be native one-based moisture slot')
                arrays[name]=self._column(scalars[iqv],name,nl)
            elif name=='theta_m':
                arrays[name]=self._column(s[name],name,nl)
            elif name=='areaCell':
                arrays[name]=cp.ascontiguousarray(m[name][:self.n_alloc])
            elif name in ('cellsOnVertex','kiteAreasOnVertex'):
                value=m[name]
                shape=(int(m['vertexDegree']),int(m['nVertices']))
                ConvectiveDiagnostics._array(value,shape,name,np.int32 if name=='cellsOnVertex' else np.float32)
                arrays[name]=cp.ascontiguousarray(value.T)
            elif name=='vorticity':
                value=d[name]
                ConvectiveDiagnostics._array(value,value.shape,name)
                if value.ndim!=2 or value.shape[0]!=nl or value.shape[1]<m['nVertices']:
                    raise ValueError('vorticity requires native (level,vertex) shape')
                arrays[name]=cp.ascontiguousarray(value[:,:m['nVertices']].T)
            else:
                arrays[name]=self._column(d[name],name,nl)
        return arrays

    def _sync_maxima(self):
        import cupy as cp
        for name in MAXIMA:
            cp.copyto(self.driver.outputs[FIELDS.index(name)], self.diag[name])

    def _publish(self, names, *, all_cells=False):
        import cupy as cp
        count = self.n_alloc if all_cells else self.driver.n_solve
        for name in names:
            cp.copyto(self.diag[name][:count], self.driver.outputs[FIELDS.index(name), :count])

    def update(self):
        """mpas_convective_diagnostics.F:99-198; consume live native level 1."""
        mask=self.driver.update_mask
        if not mask:
            return
        names=set()
        if mask&(1<<14):
            names.update(('zgrid','w','areaCell','cellsOnVertex','kiteAreasOnVertex','vorticity'))
        if mask&(1<<15):
            names.add('w')
        if mask&(1<<16):
            names.update(('uReconstructZonal','uReconstructMeridional'))
        self._sync_maxima()
        self.driver.update(self._arrays(names))
        self._publish(name for name in MAXIMA if mask&(1<<FIELDS.index(name)))

    def compute(self):
        """mpas_convective_diagnostics.F:227-487; consume current stream alarms."""
        names=[name for name in FIELDS[:14] if self.utils.field_will_be_written(name)]
        if not names:
            return
        arrays=self._arrays(('zgrid','uReconstructZonal','uReconstructMeridional',
                             'theta_m','qv','exner','pressure_base','pressure_p'))
        self.driver.compute(arrays,names)
        changed={'uzonal_surface','umeridional_surface','temperature_surface','dewpoint_surface'}
        changed.update(set(names)&{'uzonal_1km','uzonal_6km','umeridional_1km','umeridional_6km',
                                  'srh_0_1km','srh_0_3km'})
        if set(names)&{'cape','cin'}:
            changed.update(('cape','cin'))
        # LCL/LFC and unrelated output words remain in their original arrays.
        self._publish(changed)

    def reset(self):
        """mpas_convective_diagnostics.F:501-525; reset before alarm reset."""
        names=[name for name in MAXIMA if self.utils.field_will_be_written(name)]
        self.driver.reset(names)
        self._publish(names, all_cells=True)


class NativeConvectiveDiagnostics(ConvectiveDiagnosticsModule):
    """Thin binding to f4's real native session and native output alarms.

    The caller supplies primary diagnostics at MPAS's lifecycle points. This
    adapter copies their words and never recomputes vorticity or other inputs.
    Native streams retain ownership of output files and their alarm resets.
    """

    def __init__(self, session, streams):
        """mpas_convective_diagnostics.F:53-85; bind native pools/stream fields."""
        import cupy as cp
        self.session, self.streams = session, streams
        owner = self

        class Pool:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, name):
                # Copies preserve the native sentinel and current time level.
                if self.name == 'mesh' and name in session.layout.dimensions:
                    return session.layout.dimensions[name]
                if self.name == 'state' and name == 'index_qv':
                    return session.layout.indices['state', name]
                value = cp.asarray(owner.session.native.get_field(self.name, name, level=1))
                if self.name == 'mesh' and name in ('cellsOnVertex', 'kiteAreasOnVertex'):
                    value = value[:, :session.layout.dimensions['nVertices']]
                return cp.ascontiguousarray(value)

        self.pools = {name: Pool(name) for name in ('mesh', 'state', 'diag')}
        self.mesh, self.state, self.diag = (self.pools[name] for name in ('mesh', 'state', 'diag'))
        self.utils = self
        self.clock = None
        self.n_alloc = self.diag[FIELDS[0]].size
        nc = int(self.mesh['nCells'])
        if self.n_alloc not in (nc, nc+1):
            raise ValueError('diagnostic allocation must include nCells, optionally its sentinel')
        values = [ConvectiveDiagnostics._array(self.diag[name], (self.n_alloc,), name)
                  for name in FIELDS]
        included = {name for stream in streams if 'output' in stream.directions
                    for name in stream.fields if name in MAXIMA}
        self.driver = ConvectiveDiagnostics(cp.stack(values), self.mesh['nCellsSolve'],
                                            self.mesh['nVertLevels'], included)

    def field_will_be_written(self, name):
        """mpas_convective_diagnostics.F:263-295; query native output alarms."""
        return any('output' in stream.directions and stream.output_interval != 'none'
                   and name in stream.fields
                   and self.session.native.alarm_ringing(stream.name+':out')
                   for stream in self.streams)

    def _publish(self, names, *, all_cells=False):
        """mpas_convective_diagnostics.F:99-198,227-487,501-525; copy output words."""
        import cupy as cp
        count = self.n_alloc if all_cells else self.driver.n_solve
        for name in names:
            value = self.diag[name]
            cp.copyto(value[:count], self.driver.outputs[FIELDS.index(name), :count])
            self.session.put('diag', name, value)
