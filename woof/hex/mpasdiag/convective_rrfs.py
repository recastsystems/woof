# Numerical CUDA derivative of RRFS-MPAS 1d6e370e; original source remains unchanged.
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
"""Complete RRFS-MPAS 1d6e370e convective diagnostic driver on CUDA.

Inputs are resident float32 arrays in the pools' native (level, cell) layout, read
in place; zgrid/w include the top interface, and topology uses MPAS's ONE-based
indices including nCells+1.
The caller supplies output stream decisions; update is called after each
step, compute before output and reset AFTER that output has been consumed.
Outputs include allocated halos, while compute/update only change solve cells.

The native module never assigns lcl or lfc. Their incoming words are retained.
All field arithmetic runs on the device; the adapter reads stream intervals. MPAS's compile-time source=2 and adiabat=1 are retained;
the inactive source/adiabat branches are eliminated just as in the GNU build.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SOURCE_SHA256 = '1fc114ab954e6ebc9a0073ec70c6222029fcd3de68d215958599c4f03fbaf0fa'

FIELDS = (
    'cape', 'cin', 'lcl', 'lfc', 'srh_0_1km', 'srh_0_3km',
    'uzonal_surface', 'uzonal_1km', 'uzonal_6km',
    'umeridional_surface', 'umeridional_1km', 'umeridional_6km',
    'temperature_surface', 'dewpoint_surface',
)
UH_FIELDS = ('updraft_helicity_max', 'updraft_helicity_max_01', 'updraft_helicity_max_16',
    'updraft_helicity_max0_15', 'updraft_helicity_max05_15', 'updraft_helicity_max_02',
    'updraft_helicity_max_03', 'updraft_helicity_min', 'updraft_helicity_min_01',
    'updraft_helicity_min_16', 'updraft_helicity_min0_15', 'updraft_helicity_min05_15',
    'updraft_helicity_min_02', 'updraft_helicity_min_03')
FIELDS = FIELDS + UH_FIELDS + ('w_velocity_max', 'w_velocity_min', 'w_velocity_mean',
    'wind_speed_level1_max', 'wind02_max', 'wind02_integral_max', 'wind_speed_10m_max',
    'wind_speed_10m_mean', 'u10m_mean', 'v10m_mean', 'prec_acc_c', 'prec_acc_nc', 'snow_acc_nc')
ACCUMULATORS = FIELDS[14:]
MAXIMA = ACCUMULATORS  # compatibility with the whole-module oracle
HAIL_FIELDS = ('max_hail_diameter_sfc_acc', 'max_hail_diameter_column_acc')



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

// Column access in the pools' native (level, cell) layout: element k of a cell's column is p[k*s].
struct S {
    const float* p; long long s;
    __device__ float operator[](int k) const {return p[(long long)k*s];}
};
// A writable level-interleaved per-cell work column.
struct W {
    float* p; long long s;
    __device__ float& operator[](int k) const {return p[(long long)k*s];}
};

// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1127-1143
template<class X,class Z> __device__ R column_height_value(X x,Z z,R zi,int n) {
    int kz=0;
    for(int k=0;k<n;k++) if(R(z[k])<=zi) kz=k;
    kz=kz<n-1?kz:n-2;
    R wz=(R(z[kz+1])-zi)/(R(z[kz+1])-R(z[kz]));
    R wzp1=R(1.0f)-wz;
    return wz*R(x[kz])+wzp1*R(x[kz+1]);
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1168-1184
template<class X,class Z> __device__ R integral_zstaggered(X x,Z z,R bot,R top,int n) {
    R result=0.0f;
    for(int k=0;k<n;k++) {
        R zb=rmax(z[k],bot),zt=rmin(z[k+1],top);
        result+=R(x[k])*rmax(0.0f,zt-zb);
    }
    return result;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1189-1209
template<class X,class Z> __device__ R integral_zpoint(X x,Z z,R bot,R top,int n) {
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
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1688-1699
__device__ R getqvs(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(17.67f)*(t-R(273.15f))/(t-R(29.65f)));
    return eps*es/(p-es);
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1705-1716
__device__ R getqvi(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(21.8745584f)*(t-R(273.15f))/(t-R(7.66f)));
    return eps*es/(p-es);
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1722-1738
__device__ R getthe(R p,R t,R td,R q) {
    R tlcl;
    if((td-t)>=R(-0.1f)) tlcl=t;
    else tlcl=R(56.0f)+R(1.0f)/(R(1.0f)/(td-R(56.0f))+R(0.00125f)*rlog(t/td));
    return t*rpow(R(100000.0f)/p,R(0.2854f)*(R(1.0f)-R(0.28f)*q))
        *rexp((R(3376.0f)/tlcl-R(2.54f))*q*(R(1.0f)+R(0.81f)*q));
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1240-1682
// source=2 (most unstable), adiabat=1 (pseudoadiabatic, liquid only). One thread per column; the column's work
// arrays are level-interleaved accessors (W). Same operations in the same order as the native routine, with three
// changes of schedule that leave every word as it was:
//  * The pressure path of the ascent (p2 -= dp per substep) depends only on the level pressures, so pi2 =
//    (p2/p00)^(rd/cp) and log(p2/p1) of every substep are formed first (prep[s], prep[cap+s]) and read back by
//    the ascent; log(p2/p1) is the loop-invariant of the native iteration, evaluated once per substep.
//  * The ascent is one flat loop of saturation iterations (a lane finishing a substep or level moves on while
//    its neighbours keep iterating), so the 32 columns of a warp no longer serialise on each other's loop exits.
//  * The ice-saturation term enters only as 0*qvi. When qvi is certainly finite (es_ice < 0.999 p2 by a 1e-5
//    accurate bound, p2 > 0), 0*qvi is a signed zero and qvs (>= +0 there) + that zero is qvs, so qvi is not
//    formed; otherwise it is formed and added exactly as native does (an infinite or NaN qvi still poisons qv2).
template<class A> __device__ void getcape_cols(int nk,A pin,A tin,A tdin,A p,A t,A td,A pi,A q,A th,A thv,
                                               A prep,int cap,float& cape_out,float& cin_out) {
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
    // Pressure path of every substep the ascent can take (it may stop earlier), up to cap substeps.
    int prepared=0;
    {
        R p2=p[kmax];int s=0;
        for(int k=kmax+1;k<nk&&s<cap;k++) {
            R dp=R(p[k-1])-R(p[k]);
            int nloop=1;
            if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
            for(int n=0;n<nloop&&s<cap;n++,s++) {
                R p1=p2;p2=p2-dp;
                prep[s]=rpow(p2*rp00,rddcp).v;prep[cap+s]=rlog(p2/p1).v;
            }
        }
        prepared=s;
    }
    R narea=0.0f,cape=0.0f,cin=0.0f;
    R th2=th[kmax],pi2=pi[kmax],p2=p[kmax],t2=t[kmax],thv2=thv[kmax],qv2=q[kmax],b2=0.0f;
    R ql2=0.0f,qi2=0.0f,qt=qv2;
    int k=kmax+1;
    if(k>=nk) {cape_out=cape.v;cin_out=cin.v;return;}
    // Level k and its first substep.
    R b1=b2,dp=R(p[k-1])-R(p[k]);
    int nloop=1,n=0,s=0,i=0;
    if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
    R p1=p2,t1=t2,th1=th2,qv1=qv2,ql1=ql2,qi1=qi2,lg;
    p2=p2-dp;
    if(s<cap) {pi2=prep[s];lg=prep[cap+s];} else {pi2=rpow(p2*rp00,rddcp);lg=rlog(p2/p1);}
    R thlast=th1;
    // Keep this level's environmental words in registers throughout its ascent.
    // Read the following substep before the saturation chain starts; volatile loads
    // keep the compiler from sinking them back into the converged branch.
    R env_thv=thv[k],prev_thv=thv[k-1],env_pi=pi[k],prev_pi=pi[k-1],env_p=p[k];
    R next_pi=0.0f,next_lg=0.0f;
    while(true) {
        if(i==0 && s+1<prepared) {
            next_pi=static_cast<volatile float&>(prep[s+1]);
            next_lg=static_cast<volatile float&>(prep[cap+s+1]);
        }
        i++;
        t2=thlast*pi2;
        const R qvs=getqvs(p2,t2);
        const float targ=(R(21.8745584f)*(t2-R(273.15f))/(t2-R(7.66f))).v;
        if(p2.v>0.0f&&611.2f*__expf(targ)<0.999f*p2.v) qv2=rmin(qt,R(1.0f)*qvs);
        else qv2=rmin(qt,R(1.0f)*qvs+R(0.0f)*getqvi(p2,t2));
        qi2=rmax(R(0.0f)*(qt-qv2),0.0f);
        ql2=rmax(qt-qv2-qi2,0.0f);
        R tbar=R(0.5f)*(t1+t2),qvbar=R(0.5f)*(qv1+qv2);
        R qlbar=R(0.5f)*(ql1+ql2),qibar=R(0.5f)*(qi1+qi2);
        R lhv=lv1-lv2*tbar,lhs=ls1-ls2*tbar;
        R rm=rd+rv*qvbar,cpm=cp+cpv*qvbar+cpl*qlbar+cpi*qibar;
        th2=th1*rexp(lhv*(ql2-ql1)/(cpm*tbar)+lhs*(qi2-qi1)/(cpm*tbar)
                     +(rm/cpm-rd/cp)*lg);
        // Native returns the accumulated CAPE/CIN on nonconvergence.
        if(i>100) {cape_out=cape.v;cin_out=cin.v;return;}
        if(rabs(th2-thlast)>R(0.0002f)) {thlast=thlast+R(0.3f)*(th2-thlast);continue;}
        // The substep converged.
        qt=qv2;ql2=0.0f;qi2=0.0f;
        n++;s++;
        if(n>=nloop) {
            // The level is done.
            thv2=th2*(R(1.0f)+reps*qv2)/(R(1.0f)+qv2+ql2+qi2);
            b2=g*(thv2-env_thv)/env_thv;
            R dz=-cpdg*R(0.5f)*(env_thv+prev_thv)*(env_pi-prev_pi);
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
            if(env_p<=R(10000.0f)&&b2<R(0.0f)) break;
            if(++k>=nk) break;
            prev_thv=env_thv;prev_pi=env_pi;
            env_thv=thv[k];env_pi=pi[k];
            R prev_p=env_p;env_p=p[k];
            b1=b2;dp=prev_p-env_p;
            nloop=1;n=0;
            if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
        }
        // Next substep.
        p1=p2;t1=t2;th1=th2;qv1=qv2;ql1=ql2;qi1=qi2;
        p2=p2-dp;
        if(s<cap) {pi2=next_pi;lg=next_lg;} else {pi2=rpow(p2*rp00,rddcp);lg=rlog(p2/p1);}
        thlast=th1;i=0;
    }
    cape_out=cape.v;cin_out=cin.v;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:710-970
// One thread per column of this launch's slice [c0, c0+n) of the solve cells, 32-thread blocks (one warp per SM:
// the parcel ascent is a long serial chain through FP64 libm, so it is spread over as many SMs as there are
// warps). Inputs are the pools' (level, cell) words read in place (row stride s*); outs[j] is output row j's own
// allocation, written in place. work is level-interleaved per slice column (work[r*n+i]): 17*nl+3 rows, then the
// ascent's substep pressure path (2*cap rows).
extern "C" __global__ void compute_diag_cols(int c0,int n,int nl,int cap,unsigned long long mask,float rvord,
    const float* height,long long sz,const float* u,long long su,const float* v,long long sv,
    const float* theta,long long st,const float* qv,long long sq,const float* exner,long long se,
    const float* pb,long long sb,const float* pp,long long sp,float* const* outs,float* work) {
    const int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    const int c=c0+i;
    const S z{height+c,sz},uc{u+c,su},vc{v+c,sv};
    #define ROW(r) W{work+(long long)(r)*n+i,n}
    const W temp=ROW(0),dew=ROW(nl),zp=ROW(2*nl),zr=ROW(3*nl),dudz=ROW(4*nl+1),dvdz=ROW(5*nl+1);
    const W srh=ROW(6*nl+1),pin=ROW(7*nl+2),tin=ROW(8*nl+2),tdin=ROW(9*nl+2);
    const W cp_=ROW(10*nl+2),ct=ROW(11*nl+2),ctd=ROW(12*nl+2),cpi=ROW(13*nl+2),cq=ROW(14*nl+2),cth=ROW(15*nl+2);
    const W cthv=ROW(16*nl+2),prep=ROW(17*nl+3);
    #undef ROW
    for(int k=0;k<nl;k++) {
        const float thk=theta[(long long)k*st+c],qk=qv[(long long)k*sq+c],exk=exner[(long long)k*se+c];
        const float pbk=pb[(long long)k*sb+c],ppk=pp[(long long)k*sp+c];
        temp[k]=(R(thk)/(R(1.0f)+R(rvord)*R(qk))*R(exk)).v;
        R evp=R(0.01f)*(R(pbk)+R(ppk))*R(qk)/(R(qk)+R(0.622f));
        evp=rmax(evp,1.0e-8f);
        R lg=rlog(evp/R(6.112f));
        dew[k]=(R(243.5f)*lg/(R(17.67f)-lg)+R(273.15f)).v;
        zp[k]=(R(0.5f)*(R(z[k])+R(z[k+1]))-R(z[0])).v;
    }
    for(int k=0;k<=nl;k++) zr[k]=(R(z[k])-R(z[0])).v;
    outs[6][c]=uc[0];outs[9][c]=vc[0];outs[12][c]=temp[0];outs[13][c]=dew[0];
    if(mask&(1u<<7)) outs[7][c]=column_height_value(uc,zp,1000.0f,nl).v;
    if(mask&(1u<<8)) outs[8][c]=column_height_value(uc,zp,6000.0f,nl).v;
    if(mask&(1u<<10)) outs[10][c]=column_height_value(vc,zp,1000.0f,nl).v;
    if(mask&(1u<<11)) outs[11][c]=column_height_value(vc,zp,6000.0f,nl).v;
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
        if(mask&(1u<<4)) outs[4][c]=integral_zpoint(srh,zr,0.0f,1000.0f,nl+1).v;
        if(mask&(1u<<5)) outs[5][c]=integral_zpoint(srh,zr,0.0f,3000.0f,nl+1).v;
    }
    if(mask&3u) {
        for(int k=0;k<nl;k++) {
            const float pbk=pb[(long long)k*sb+c],ppk=pp[(long long)k*sp+c];
            pin[k]=((R(ppk)+R(pbk))/R(100.0f)).v;
            tin[k]=(R(temp[k])-R(273.15f)).v;tdin[k]=(R(dew[k])-R(273.15f)).v;
        }
        float cape,cin;
        getcape_cols(nl,pin,tin,tdin,cp_,ct,ctd,cpi,cq,cth,cthv,prep,cap,cape,cin);
        outs[0][c]=cape;outs[1][c]=cin;
    }
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// Native vertex order is retained, including the nCells+1 sentinel.
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
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// The same scatter as a gather: one thread per (cell,level) adds its slots in
// native scatter order from the same zero word (convective_v842.helicity_plan).
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
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// vertex_helicity_gather reading vorticity in the pool's own (level, vertex) layout (row stride sv).
extern "C" __global__ void vertex_helicity_gather_pool(int nc,int nl,int degree,
    const int* offsets,const int* entries,const float* kite,const float* vort,long long sv,float* uh) {
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=(long long)(nc+1)*nl) return;
    int c=(int)(t/nl),k=(int)(t%nl);
    R s=0.0f;
    for(int e=offsets[c];e<offsets[c+1];e++) {
        int ix=entries[e];
        s=s+R(kite[ix])*R(vort[(long long)k*sv+ix/degree]);
    }
    uh[(long long)c*nl+k]=s.v;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// w in the pool's own (level, cell) layout (row stride sw).
extern "C" __global__ void split_helicity_pool(int nc,int nl,const float* w,long long sw,
    const float* area,float* positive,float* negative) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>nc) return;
    for(int k=0;k<nl;k++) {
        R h=positive[c*nl+k];
        if(c<nc) {
            R wm=R(0.5f)*(R(w[(long long)k*sw+c])+R(w[(long long)(k+1)*sw+c]));
            positive[c*nl+k]=(rmax(wm,0.0f)*rmax(h/R(area[c]),0.0f)).v;
            negative[c*nl+k]=rmin(wm*h/R(area[c]),0.0f).v;
        } else negative[c*nl+k]=h.v;
    }
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
extern "C" __global__ void smooth_helicity(int ns,int nl,int maxedges,
    const int* bdy,const int* nedges,const int* edges,const int* cells,
    const float* dv,const float* dc,const float* positive,const float* negative,
    float* spos,float* sneg) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=ns) return;
    for(int k=0;k<nl;k++) {
        R sp=0.0f,sn=0.0f;
        if(bdy[c]==0) for(int j=0;j<nedges[c];j++) {
            int ix=c*maxedges+j,e=edges[ix]-1,other=cells[ix]-1;
            R weight=R(dv[e])/R(dc[e]);
            sp+=weight*(R(positive[other*nl+k])-R(positive[c*nl+k]));
            sn+=weight*(R(negative[other*nl+k])-R(negative[c*nl+k]));
        }
        spos[c*nl+k]=(R(positive[c*nl+k])+R(0.1f)*sp).v;
        sneg[c*nl+k]=(R(negative[c*nl+k])+R(0.1f)*sn).v;
    }
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1146-1165
template<class X,class Z> __device__ R col_depth_max(X values,Z z,R bot,R top,int n) {
    int kb=0,kt=0;
    for(int k=0;k<n;k++) {if(R(z[k])<=bot) kb=k;if(R(z[k])>=top) kt=k;}
    kb=kb<n-1?kb:n-2;kt=kt<n-1?kt:n-2;
    R result=-__int_as_float(0x7f800000);
    for(int k=kb;k<=kt;k++) result=rmax(values[k],result);
    return result;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1213-1233
template<class X,class P> __device__ R mean_pressure(X values,P p,R bot,R top,int n) {
    R result=0.0f;int count=0;
    for(int k=0;k<n;k++) if(R(p[k])<=bot&&R(p[k])>=top) {result+=values[k];count++;}
    // The strict GNU 0/0 result is x86's negative indefinite NaN, not
    // CUDA's 0x7fffffff canonical NaN (measured empty-window replay).
    if(count==0) return __int_as_float(0xffc00000);
    return result/R(float(count));
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// One thread per cell. Column inputs are the pools' (level, cell) words read in place (row stride s*); outs[j] is
// output row j's own allocation, updated in place. The work area is level-interleaved (work[k*ns+c]): zg is rows
// 0..nl, pres nl+1..2nl, speed 2nl+1..3nl. zg[nl] is the native unassigned wind-integration word: tail[c] when the
// caller restores a measured native word, else zero.
extern "C" __global__ void update_diag_pool(int ns,int nl,unsigned long long mask,float steps,int convection,
    const float* z,long long sz,const float* w,long long sw,const float* u,long long su,const float* v,long long sv,
    const float* pb,long long sb,const float* pp,long long spr,const float* refl,long long sr,
    const float* u10,const float* v10,const float* t2,
    const float* rainc,const float* rainnc,const float* snow,const float* uh,const float* neg,
    const float* tail,float* const* outs,float* work) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=ns) return;
    const S zc{z+c,sz},wc{w+c,sw},uc{u+c,su},vc{v+c,sv},pbc{pb+c,sb},ppc{pp+c,spr},rc{refl+c,sr};
    const W zg{work+c,ns},pres{work+(long long)(nl+1)*ns+c,ns},speed{work+(long long)(2*nl+1)*ns+c,ns};
    zg[nl]=tail?tail[c]:0.0f;
    #define NEED(j) (mask&(1ull<<(j)))
    #define OUT(j) outs[j][c]
    if(NEED(28)) {R wm=wc[0];for(int k=1;k<nl;k++) wm=rmax(wc[k],wm);OUT(28)=rmax(wm,OUT(28)).v;}
    if(NEED(29)) {R wm=wc[0];for(int k=1;k<nl;k++) wm=rmin(wc[k],wm);OUT(29)=rmin(wm,OUT(29)).v;}
    if(NEED(30)) {
        R span=R(pbc[0])+R(ppc[0])-R(pbc[nl-1])-R(ppc[nl-1]);
        for(int k=0;k<nl;k++) pres[k]=(R(pbc[k])+R(ppc[k])).v;
        R mean=mean_pressure(wc,pres,R(0.8f)*span,R(0.5f)*span,nl);
        // Preserve GNU's quiet NaN word through multiplication/addition;
        // CUDA otherwise canonicalizes it on each arithmetic instruction.
        if(OUT(30)!=OUT(30)) {} // GNU propagates the existing accumulator.
        else if(mean.v!=mean.v) OUT(30)=mean.v;
        else OUT(30)=(R(OUT(30))+(R(1.0f)/R(steps))*mean).v;
    }
    if(NEED(31)) {R a=uc[0],b=vc[0];OUT(31)=rmax(rsqrt(a*a+b*b),OUT(31)).v;}
    if(NEED(32)||NEED(33)) {
        // The fork initializes only the first nl entries (zg[nl] is the word set above).
        for(int k=0;k<nl;k++) zg[k]=(R(0.5f)*(R(zc[k])+R(zc[k+1]))-R(zc[0])).v;
        if(column_height_value(rc,zg,1000.0f,nl)>R(20.0f)) {
            if(NEED(32)) {
                for(int k=0;k<nl;k++) {R a=uc[k],b=vc[k];speed[k]=rsqrt(a*a+b*b).v;}
                OUT(32)=rmax(col_depth_max(speed,zg,0.0f,2000.0f,nl),OUT(32)).v;
            }
            if(NEED(33)) {
                R a=integral_zstaggered(uc,zg,0.0f,2000.0f,nl);
                OUT(33)=rmax(rsqrt(a*a+a*a),OUT(33)).v;
            }
        }
    }
    if(NEED(34)||NEED(35)) {R a=u10[c],b=v10[c],speed10=rsqrt(a*a+b*b);
        if(NEED(34)) OUT(34)=rmax(speed10,OUT(34)).v;
        if(NEED(35)) OUT(35)=(R(OUT(35))+speed10*(R(1.0f)/R(steps))).v;
    }
    if(NEED(36)) OUT(36)=(R(OUT(36))+R(u10[c])*(R(1.0f)/R(steps))).v;
    if(NEED(37)) OUT(37)=(R(OUT(37))+R(v10[c])*(R(1.0f)/R(steps))).v;
    if(NEED(38)&&convection) OUT(38)=rmax(R(OUT(38))+R(rainc[c]),0.0f).v;
    if(NEED(39)) OUT(39)=rmax(R(OUT(39))+R(rainnc[c]),0.0f).v;
    if(NEED(40)) {
        OUT(40)=(R(OUT(40))+R(snow[c])).v;
        if(R(t2[c])<R(273.15f)) {
            if(convection) OUT(40)=(R(OUT(40))+R(rainc[c])).v;
            OUT(40)=rmax(OUT(40),0.0f).v;
        }
    }
    if(mask&(((1ull<<14)-1)<<14)) {
        for(int k=0;k<=nl;k++) zg[k]=(R(zc[k])-R(zc[0])).v;
        const float bot[7]={2000,0,1000,0,500,0,0},top[7]={5000,1000,6000,1500,1500,2000,3000};
        for(int j=0;j<14;j++) if(NEED(14+j)) {
            R h=integral_zstaggered((j<7?uh:neg)+(long long)c*nl,zg,bot[j%7],top[j%7],nl);
            OUT(14+j)=(j<7?rmax(h,OUT(14+j)):rmin(h,OUT(14+j))).v;
        }
    }
    #undef NEED
    #undef OUT
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:984-1124
extern "C" __global__ void reset_diag_pool(int nc,unsigned long long mask,float* const* outs) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=nc) return;
    for(int j=14;j<41;j++) if(mask&(1ull<<j)) outs[j][c]=0.0f;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:1350-1381
extern "C" __global__ void hail_update(int nc,int its,int ite,int input_start,const float* sfc,const float* column,float* sacc,float* cacc) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;if(c<its||c>=ite||c>=nc) return;
    sacc[c]=rmax(sfc[c-input_start],sacc[c]).v;cacc[c]=rmax(column[c-input_start],cacc[c]).v;
}
// RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:972-1003
extern "C" __global__ void physics_reset(int n,float* values) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;if(c<n) values[c]=0.0f;
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
def _module(device_id):
    import cupy as cp
    from cupy.cuda import compiler
    # RawModule's cache compiler appends -ftz=true. Direct NVRTC preserves
    # native subnormal interval maxima; test_subnormal_cuda guards this.
    with cp.cuda.Device(device_id):
        binary, _ = compiler.compile_using_nvrtc(
            device_source(), options=_hex_numerics.options(('--std=c++17', '--fmad=false', '--ftz=false',
                                     '--prec-div=true', '--prec-sqrt=true')),
            arch=cp.cuda.Device(device_id).compute_capability)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def _update_array_names(mask: int, convection: bool) -> set[str]:
    """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681.

    Pool lookups may return unassociated arrays for inactive physics packages.
    Only the enabled native arithmetic branches dereference these arrays.
    """
    requirements = {
        'w_velocity_max': ('w',), 'w_velocity_min': ('w',),
        'w_velocity_mean': ('w','pressure_base','pressure_p'),
        'wind_speed_level1_max': ('uReconstructZonal','uReconstructMeridional'),
        'wind02_max': ('zgrid','refl10cm','uReconstructZonal','uReconstructMeridional'),
        'wind02_integral_max': ('zgrid','refl10cm','uReconstructZonal'),
        'wind_speed_10m_max': ('u10','v10'), 'wind_speed_10m_mean': ('u10','v10'),
        'u10m_mean': ('u10',), 'v10m_mean': ('v10',),
        'prec_acc_nc': ('rainncv',), 'snow_acc_nc': ('snowncv','t2m'),
    }
    needed = set()
    for field, names in requirements.items():
        if mask & (1<<FIELDS.index(field)): needed.update(names)
    if convection and mask & _mask(('prec_acc_c','snow_acc_nc')): needed.add('raincv')
    if mask & _mask(UH_FIELDS): needed.update(('zgrid','w','areaCell'))
    return needed


class ConvectiveDiagnostics:
    """Stream-driven driver, preserving the MPAS output/update/reset boundary.

    Every kernel reads its inputs in the pools' native (level, cell) layout in place and writes each output row
    in place through a device table of row pointers, so neither a transpose nor a copy-in/copy-out crosses the
    device per call.
    """
    #: Substeps of the CAPE ascent whose pressure path is formed ahead (100 Pa substeps: a 1,200 hPa column fits;
    #: any further substep forms its own, the same words).
    CAPE_SUBSTEPS = 1280
    #: Byte budget of the compute work area (level-interleaved). One launch takes as many solve columns as fit
    #: (about 37,000 at 59 levels: every column of a 15 km mesh, about 7 warps per SM of a 170-SM card on a 3 km
    #: one); a larger mesh runs consecutive launches of that many columns. Each column's words depend only on
    #: that column, so the split changes no word.
    COMPUTE_WORK_BYTES = 512 << 20

    def __init__(self, outputs, n_cells_solve: int, n_levels: int,
                 output_stream_fields: Iterable[str]):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:79-283 (setup).

        ``outputs`` is either one writable resident C float32 array (41, allocatedCells) or a sequence of 41
        such (allocatedCells,) arrays in FIELDS order (each output's own allocation, written in place). It must
        contain native initialization/restart values on entry. No implicit initialization occurs; lcl/lfc
        always retain those words.
        """
        import cupy as cp
        self.n_solve = int(n_cells_solve)
        self.n_levels = int(n_levels)
        if isinstance(outputs, cp.ndarray):
            self.outputs = outputs
            self.n_cells = outputs.shape[1] if outputs.ndim == 2 else 0
            self._array(outputs, (len(FIELDS), self.n_cells), 'outputs')
            rows = [outputs[i] for i in range(len(FIELDS))]
        else:
            self.outputs = None
            rows = list(outputs)
            if len(rows) != len(FIELDS):
                raise ValueError(f'outputs must hold the {len(FIELDS)} convective fields in FIELDS order')
            self.n_cells = rows[0].shape[0] if getattr(rows[0], 'ndim', 0) == 1 else 0
        if not 0 <= self.n_solve <= self.n_cells or self.n_levels < 3:
            # Native SRH indexes level 2 and nLevels-1; reject invalid input
            # dimensions before a kernel can read outside allocated storage.
            raise ValueError('require 0 <= nCellsSolve <= nCells and nLevels >= 3')
        self.bind_outputs(rows)
        self.update_mask = _mask(output_stream_fields) & _mask(MAXIMA)
        self._helicity_plan = None
        self._helicity = None
        self._work = None
        self._compute_work = None
        # A same-host rank's native vertex scatter order (output_diagnostics_rrfs); None is the one-card order.
        self.vertex_order = None

    def bind_outputs(self, rows):
        """Point the kernels' output table at these 41 row allocations (FIELDS order)."""
        import cupy as cp
        rows = [self._array(row, (self.n_cells,), FIELDS[i]) for i, row in enumerate(rows)]
        pointers = tuple(row.data.ptr for row in rows)
        if pointers != getattr(self, '_pointers', None):
            self._rows, self._pointers = rows, pointers
            self._table = cp.asarray(np.asarray(pointers, dtype=np.uint64))
        return self

    @staticmethod
    def _array(array, shape, name, dtype=np.float32):
        import cupy as cp
        if not isinstance(array, cp.ndarray) or array.dtype != np.dtype(dtype):
            raise TypeError(f'{name} must be a resident {np.dtype(dtype)} CUDA array')
        if array.shape != shape or not array.flags.c_contiguous:
            raise ValueError(f'{name} must be C contiguous with shape {shape}')
        return array

    def _column(self, array, name, levels):
        """A pool (level, cell) array the kernels read in place: (levels, >= nCellsSolve), unit cell stride.

        Returns the array and its row stride in words."""
        import cupy as cp
        if not isinstance(array, cp.ndarray) or array.dtype != np.float32:
            raise TypeError(f'{name} must be a resident float32 CUDA array')
        if (array.ndim != 2 or array.shape[0] != levels or array.shape[1] < self.n_solve
                or array.strides[1] != 4 or array.strides[0] % 4 or array.strides[0] < 4*self.n_solve):
            raise ValueError(f'{name} must have the native ({levels}, >= nCellsSolve) layout with unit cell stride')
        return [array, np.int64(array.strides[0]//4)]

    def _cells(self, array, name, count=None, dtype=np.float32):
        """A per-cell vector the kernels index by cell: (>= count,) contiguous."""
        import cupy as cp
        count = self.n_solve if count is None else count
        if not isinstance(array, cp.ndarray) or array.dtype != np.dtype(dtype):
            raise TypeError(f'{name} must be a resident {np.dtype(dtype)} CUDA array')
        if array.ndim != 1 or array.shape[0] < count or not array.flags.c_contiguous:
            raise ValueError(f'{name} must be contiguous with at least {count} cells')
        return array

    def _launch(self, name, count, args):
        if count:
            import cupy as cp
            _module(cp.cuda.Device().id).get_function(name)(((count+127)//128,), (128,), args)

    def compute(self, arrays: Mapping[str, object], fields_will_be_written: Iterable[str],
                *, rvord: float = float(np.float32(461.6)/np.float32(287.0))):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:710-970; compute before stream output.

        Column inputs are native (level, cell) arrays (zgrid with its top interface). qv is the native
        moisture slot's (level, cell) view. relhum is unused by the pinned native routine and is not required.
        ``rvord`` can be supplied from MPAS's selected constants package.
        """
        import cupy as cp
        mask = _mask(fields_will_be_written) & ((1 << 14)-1)
        if not mask or not self.n_solve:
            return self.outputs
        args = []
        for name in ('zgrid', 'uReconstructZonal', 'uReconstructMeridional',
                     'theta_m', 'qv', 'exner', 'pressure_base', 'pressure_p'):
            args += self._column(arrays[name], name, self.n_levels + (name == 'zgrid'))
        # Columns per launch: the level-interleaved work (17*nl+3 rows plus the 2*CAPE_SUBSTEPS rows of the
        # ascent's pressure path) within COMPUTE_WORK_BYTES, whole warps.
        rows = 17*self.n_levels + 3 + 2*self.CAPE_SUBSTEPS
        n = min(self.n_solve, max(32, self.COMPUTE_WORK_BYTES // (4*rows) // 32 * 32))
        if self._compute_work is None or self._compute_work.shape != (rows, n):
            self._compute_work = cp.empty((rows, n), dtype=cp.float32)
        kernel = _module(cp.cuda.Device().id).get_function('compute_diag_cols')
        for c0 in range(0, self.n_solve, n):
            count = min(n, self.n_solve - c0)
            # 32-thread blocks: the ascent's FP64 libm chain is spread over as many SMs as there are warps.
            kernel(((count+31)//32,), (32,),
                   (np.int32(c0), np.int32(count), np.int32(self.n_levels), np.int32(self.CAPE_SUBSTEPS),
                    np.uint64(mask), np.float32(rvord), *args, self._table, self._compute_work))
        return self.outputs

    def update(self, arrays: Mapping[str, object], *, nsteps: int, convection: bool,
               n_mesh_cells: int | None = None):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681.

        nsteps is the native integer quotient of diagnostics interval by dt. Column inputs are native
        (level, cell) arrays; cellsOnVertex/kiteAreasOnVertex are (vertex, degree) and edgesOnCell/cellsOnCell
        (cell, maxEdges); vorticity is native (level, vertex). Topology indices are one-based, including the
        nCells+1 sentinel.
        """
        import cupy as cp
        if not self.update_mask or not self.n_solve:
            return self.outputs
        nc = self.n_cells if n_mesh_cells is None else int(n_mesh_cells)
        nl = self.n_levels
        needed = _update_array_names(self.update_mask, convection)
        columns = {}
        for name in ('zgrid', 'w', 'uReconstructZonal', 'uReconstructMeridional',
                     'pressure_base', 'pressure_p', 'refl10cm'):
            if name in needed:
                columns[name] = self._column(arrays[name], name, nl + (name in ('zgrid', 'w')))
        for name in ('u10', 'v10', 't2m', 'raincv', 'rainncv', 'snowncv'):
            if name in needed:
                self._cells(arrays[name], name)
        dummy = self._rows[0]
        uh = neg = dummy
        if self.update_mask & _mask(UH_FIELDS):
            cov = arrays['cellsOnVertex']; nv, degree = cov.shape
            self._array(cov, (nv, degree), 'cellsOnVertex', np.int32)
            kite = self._array(arrays['kiteAreasOnVertex'], (nv, degree), 'kiteAreasOnVertex')
            vort, vstride = self._column(arrays['vorticity'], 'vorticity', nl)
            if vort.shape[1] < nv:
                raise ValueError('vorticity must span nVertices')
            order = self.vertex_order
            key = (cov.shape, nc, id(order))
            if self._helicity_plan is None or self._helicity_plan[0] != key:
                from .convective_v842 import helicity_plan
                # Mesh connectivity is fixed for a run; build the gather once.
                self._helicity_plan = (key, helicity_plan(cov, nc, order))
            offsets, entries = self._helicity_plan[1]
            if self._helicity is None or self._helicity[0].shape != (nc+1, nl):
                # Gather writes every word, split every cell, smooth the solve cells the update reads.
                self._helicity = tuple(cp.empty((nc+1, nl), dtype=cp.float32) for _ in range(4))
            uh, neg, spos, sneg = self._helicity
            w, wstride = columns['w']
            area = self._cells(arrays['areaCell'], 'areaCell', nc)
            self._launch('vertex_helicity_gather_pool', (nc+1)*nl,
                (np.int32(nc), np.int32(nl), np.int32(degree), offsets, entries, kite, vort, vstride, uh))
            self._launch('split_helicity_pool', nc+1,
                (np.int32(nc), np.int32(nl), w, wstride, area, uh, neg))
            maxedges = arrays['cellsOnCell'].shape[1]
            self._launch('smooth_helicity', self.n_solve,
                (np.int32(self.n_solve), np.int32(nl), np.int32(maxedges),
                 *(arrays[name] for name in ('bdyMaskCell','nEdgesOnCell','edgesOnCell','cellsOnCell','dvEdge','dcEdge')),
                 uh, neg, spos, sneg))
            uh, neg = spos, sneg
        if self._work is None or self._work.shape != (3*nl+1, self.n_solve):
            # Level-interleaved per-cell work (zg, pres, speed); every word a cell reads is written that call.
            self._work = cp.empty((3*nl+1, self.n_solve), dtype=cp.float32)
        # For wind integration the last interface is a native unassigned word. Zero reproduces the standalone
        # replay only; wind_z_agl_tail restores a measured native word for the witness.
        tail = arrays.get('wind_z_agl_tail')
        if tail is not None:
            tail = self._cells(tail, 'wind_z_agl_tail')
        column_args = []
        for name in ('zgrid', 'w', 'uReconstructZonal', 'uReconstructMeridional',
                     'pressure_base', 'pressure_p', 'refl10cm'):
            column_args += columns.get(name, (dummy, np.int64(0)))
        self._launch('update_diag_pool', self.n_solve,
            (np.int32(self.n_solve), np.int32(nl), np.uint64(self.update_mask), np.float32(nsteps),
             np.int32(convection), *column_args,
             *(arrays.get(name, dummy) for name in ('u10', 'v10', 't2m', 'raincv', 'rainncv', 'snowncv')),
             uh, neg, np.uint64(0) if tail is None else tail, self._table, self._work))
        return self.outputs

    def reset(self, fields_will_be_written: Iterable[str]):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:984-1124; reset after output consumption.

        Reset decisions use this event's stream fields, independently of the
        setup inclusion flags, and reset all allocated cells including halos.
        """
        mask = _mask(fields_will_be_written) & _mask(MAXIMA)
        if mask:
            self._launch('reset_diag_pool', self.n_cells, (np.int32(self.n_cells), np.uint64(mask), self._table))
        return self.outputs


class ConvectiveDiagnosticsModule:
    """Native pool lifecycle adapter selected by the diagnostics provider table.

    With resident pools (a ``view`` accessor: ResidentOutputDiagnostics) the kernels read the current authority
    in place and update the produced output allocations in place. Any other pool provider keeps the copy-in and
    copy-out of one stacked output array around each call.
    """

    def setup(self, pools, clock, utils):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:79-283."""
        import cupy as cp
        from .utils import Direction
        self.pools,self.clock,self.utils = pools,clock,utils
        self.mesh,self.diag,self.physics = (pools[name] for name in ('mesh','diag','diag_physics'))
        self.n_alloc = self.diag[FIELDS[0]].size
        included = [name for name in ACCUMULATORS if utils.stream_inclusion_count(name,direction=Direction.OUTPUT)>0]
        self.physics['do_diag_dbz'][...] = int(utils.stream_inclusion_count('refl10cm_1km_max',direction=Direction.OUTPUT)>0)
        self.in_place = callable(getattr(self.diag, 'view', None))
        rows = [self.diag[name] for name in FIELDS]
        self.driver = ConvectiveDiagnostics(rows if self.in_place else cp.stack(rows),
                                            self.mesh['nCellsSolve'],self.mesh['nVertLevels'],included)
        self._mesh = None

    @staticmethod
    def _view(pool, name):
        view = getattr(pool, 'view', None)
        return view(name) if callable(view) else pool[name]

    def _check_outputs(self):
        if self.in_place and tuple(self.diag[name].data.ptr for name in FIELDS) != self.driver._pointers:
            # THE BREAKAGE THIS PREVENTS: a replaced output allocation would silently stop receiving the
            # running maxima and the computed words (the kernels write the bound rows in place).
            raise RuntimeError('convective output allocations changed after setup')

    def _arrays(self, *, compute=False, convection=False):
        import cupy as cp
        m,d,p,s = self.mesh,self.diag,self.physics,self.pools['state']
        arrays = {}
        if compute:
            names = {'zgrid','uReconstructZonal','uReconstructMeridional','theta_m',
                     'exner','pressure_base','pressure_p'}
        else:
            names = _update_array_names(self.driver.update_mask, convection)
        column_names = {'zgrid','w','uReconstructZonal','uReconstructMeridional',
                        'theta_m','exner','pressure_base','pressure_p','refl10cm'}
        for name in names & column_names:
            pool = m if name=='zgrid' else s if name in ('w','theta_m') else p if name=='refl10cm' else d
            arrays[name] = self._view(pool, name)
        if compute:
            arrays['qv'] = self._view(s, 'scalars')[int(s['index_qv'])-1]
            return arrays
        if self.driver.update_mask & _mask(UH_FIELDS):
            if self._mesh is None:
                # Mesh connectivity and metrics are fixed for a run (as the helicity plan assumes): built once.
                mesh = {name: cp.ascontiguousarray(m[name].T)
                        for name in ('cellsOnVertex','kiteAreasOnVertex','edgesOnCell','cellsOnCell')}
                for name in ('areaCell','bdyMaskCell','nEdgesOnCell','dvEdge','dcEdge'):
                    mesh[name] = cp.ascontiguousarray(m[name][:int(m['nCells'])] if name=='areaCell' else m[name])
                self._mesh = mesh
            arrays.update(self._mesh)
            arrays['vorticity'] = d['vorticity']
        for name in names - column_names - {'areaCell'}:
            arrays[name] = self._view(p, name)
        if 'wind_z_agl_tail' in p:
            arrays['wind_z_agl_tail'] = cp.ascontiguousarray(p['wind_z_agl_tail'][:self.driver.n_solve])
        return arrays

    def _sync(self):
        import cupy as cp
        if not self.in_place:
            for i,name in enumerate(FIELDS): cp.copyto(self.driver.outputs[i],self.diag[name])

    def _publish(self,names,all_cells=False):
        import cupy as cp
        if self.in_place:
            return
        count = self.n_alloc if all_cells else self.driver.n_solve
        for name in names: cp.copyto(self.diag[name][:count],self.driver.outputs[FIELDS.index(name),:count])

    def update(self, configs, stream_mgr, dt):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681."""
        from .utils import Direction
        interval = stream_mgr.get_stream_interval('diagnostics', Direction.OUTPUT)
        nsteps = interval // dt
        self._sync()
        convection = configs['config_convection_scheme'].rstrip()!='off'
        self.driver.update(self._arrays(convection=convection),nsteps=nsteps,
                           convection=convection,n_mesh_cells=int(self.mesh['nCells']))
        self._publish(name for name in ACCUMULATORS if self.driver.update_mask & (1<<FIELDS.index(name)))

    def compute(self):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:710-970."""
        names = [name for name in FIELDS[:14] if self.utils.field_will_be_written(name)]
        if not names: return
        self._check_outputs()
        self._sync();self.driver.compute(self._arrays(compute=True),names)
        changed = {'uzonal_surface','umeridional_surface','temperature_surface','dewpoint_surface'}
        changed.update(set(names)-{'lcl','lfc','cape','cin'})
        if set(names)&{'cape','cin'}: changed.update(('cape','cin'))
        self._publish(changed)

    def reset(self):
        """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:984-1124."""
        names = [name for name in ACCUMULATORS if self.utils.field_will_be_written(name)]
        self._check_outputs()
        self.driver.reset(names);self._publish(names,all_cells=True)


def update_hail_accumulators(surface, column, surface_acc, column_acc, its, ite, *, input_start=1):
    """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:1350-1381.

    Consume TEMPO's resident tile outputs, its/ite are native inclusive one-based indices.
    input_start is the native ims lower bound of those input tile arrays.
    """
    import cupy as cp
    n = surface_acc.size
    for name,array in (('surface',surface),('column',column)):
        ConvectiveDiagnostics._array(array,(surface.size,),name)
    for name,array in (('surface_acc',surface_acc),('column_acc',column_acc)):
        ConvectiveDiagnostics._array(array,(n,),name)
    if not n or ite < its: return
    if not (1 <= its <= ite <= n and input_start <= its and ite < input_start+surface.size):
        if n: raise ValueError('native hail input/output tile bounds exceed resident arrays')
    _module(cp.cuda.Device().id).get_function('hail_update')(((n+127)//128,),(128,),
        (np.int32(n),np.int32(its-1),np.int32(ite),np.int32(input_start-1),surface,column,surface_acc,column_acc))


def reset_physics_accumulators(diag_physics):
    """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:972-1003.

    Called after a diagnostics output, independent of field inclusion alarms.
    """
    import cupy as cp
    for name in ('refl10cm_1km_max',*HAIL_FIELDS):
        values = diag_physics.get(name)
        if values is not None and values.size:
            ConvectiveDiagnostics._array(values,(values.size,),name)
            _module(cp.cuda.Device().id).get_function('physics_reset')(((values.size+127)//128,),(128,),
                (np.int32(values.size),values))


def diagnostics_manager(*, isobaric, cloud, pv, soundings):
    """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_atm_diagnostics_manager.F:31-173.

    Table-selected factory; inherits the unchanged compute/reset/cleanup order.
    The fork's update receives the native config pool, stream manager and dt.
    """
    from .manager import DiagnosticsManager

    class Manager(DiagnosticsManager):
        def setup(self, streams, configs, pools, clock, dminfo):
            """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_atm_diagnostics_manager.F:31-63."""
            self.utils.init(streams)
            self.template.setup(configs,pools,clock)
            self.isobaric.setup(pools,clock,self.utils)
            self.cloud.setup(pools,clock)
            self.convective.setup(pools,clock,self.utils)
            self.pv.setup(pools,clock)
            self.soundings.setup(configs,pools,clock,dminfo)

        def update(self, configs, stream_mgr, dt):
            """RRFSx/MPAS-Model@1d6e370e MPAS-Model-rrfs/src/core_atmosphere/diagnostics/mpas_atm_diagnostics_manager.F:76-92."""
            self.template.update()
            self.convective.update(configs,stream_mgr,dt)

    return Manager(isobaric=isobaric,cloud=cloud,convective=ConvectiveDiagnosticsModule(),
                   pv=pv,soundings=soundings)
