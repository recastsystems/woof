"""MPAS-A v8.4.2 UGWP non-stationary gravity wave column boundary.

Pinned authority: UGWP c1c893e, MPAS f34984b. The source below ports the
full reachable default MPAS non-stationary operator, including background
initialization, source interpolation, dissipative propagation, eddy smoothing,
and the MPAS tendency/diagnostic conversion. WOOF is untouched.

Layout is resident FP32 C [level, column], bottom to top; climate ABSMF is
C [day, latitude] (Fortran [latitude, day]). Indices in CUDA local scratch
remain one based to preserve the pristine solver's loop boundaries.

The launcher loads the pinned engine shared glibc device mathematics header
defining ngw_exp/log/pow/sin/cos. An explicit header supports seam integration.
Native module knobs are fixed controls, not MPAS namelist options. Conv/front/okw initialization spectra are preserved in the initialization
diagnostic state. Compile-time disabled RMS diagnostics do not affect the
operator and are not exposed as extra user options.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

SOURCE_PINS = {
    'bl_ugwpv1_ngw.F': 'fde23ef2624d754a9b8b689cd5c237e372db5a4188a3818dd544d8cbee4ea7ef',
    'cires_tauamf_data.F': '2b8c0af74003665e21e9e0f9863d168b61c74e735d947aac7c6d147a1c903461',
    'cires_ugwpv1_initialize.F': 'd066b26674284312a54d8ed6c0818ea5768c225c1e9e5e1f9df5386354bc8575',
    'cires_ugwpv1_module.F': '10d9e9511892c9cb823f0c8aa491835ced53ef049fc8c4c9a99fad4df2c7dee3',
    'cires_ugwpv1_solv2.F': '41ad723de0f69ff23c039f014b5be14b5c7d0d5a87ac8d5a37b3af8e8dd63d80',
    'cires_ugwpv1_triggers.F': '2c2d15db5b40937f8e601d45a0572c1e8cb8ebeea1218bcb9079b4e749b08e61',
}

COMPILE_OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false',
                   '--prec-div=true', '--prec-sqrt=true')

CUDA_SOURCE = r'''


#ifndef NGW_HOST
#define NGW_DEVICE __device__
// bl_ugwpv1_ngw.F:211-231, single-precision additive arithmetic.
// cires_tauamf_data.F:100-109; cires_ugwpv1_solv2.F:732-733,807-808:
// strict x86 arithmetic preserves quiet operand NaNs and produces negative
// indefinite for invalid arithmetic. CUDA's canonical 0x7fffffff differs.
NGW_DEVICE float ngw_native_nan(float result,float a,float b) {
  if(result==result) return result;
  if(a!=a) return __int_as_float(__float_as_int(a)|0x00400000);
  if(b!=b) return __int_as_float(__float_as_int(b)|0x00400000);
  return __int_as_float(0xffc00000u);
}
NGW_DEVICE float ngw_add(float a,float b) {return ngw_native_nan(__fadd_rn(a,b),a,b);}
NGW_DEVICE float ngw_sub(float a,float b) {return ngw_native_nan(__fsub_rn(a,b),a,b);}
NGW_DEVICE float ngw_mul(float a,float b) {return ngw_native_nan(__fmul_rn(a,b),a,b);}
NGW_DEVICE float ngw_div(float a,float b) {return ngw_native_nan(__fdiv_rn(a,b),a,b);}
NGW_DEVICE float ngw_sqrt(float a) {return ngw_native_nan(__fsqrt_rn(a),a,a);}
// cires_ugwpv1_solv2.F:732-733,807-808: preserve native NaN payload/sign
// for absolute value and unary negation as well as binary arithmetic.
NGW_DEVICE float ngw_neg(float a) {return __int_as_float(__float_as_int(a)^0x80000000u);}
NGW_DEVICE float ngw_abs(float a) {return __int_as_float(__float_as_int(a)&0x7fffffffu);}
#else
#define NGW_DEVICE
inline float ngw_add(float a,float b) {return a+b;}
inline float ngw_sub(float a,float b) {return a-b;}
inline float ngw_mul(float a,float b) {return a*b;}
inline float ngw_div(float a,float b) {return a/b;}
inline float ngw_sqrt(float a) {return sqrtf(a);}
inline float ngw_neg(float a) {return -a;}
inline float ngw_abs(float a) {return fabsf(a);}
#endif
template<typename T> NGW_DEVICE T ngw_min(T a,T b) {return a<b?a:b;}
template<typename T> NGW_DEVICE T ngw_max(T a,T b) {return a>b?a:b;}
// cires_ugwpv1_solv2.F:272,732-733,807-808,885: strict GNU keeps the
// measured unordered cases here. Constant clamp at line 885 is ordered
// separately to match its optimized GNU evaluation. Retain finite arithmetic.
#ifndef NGW_HOST
NGW_DEVICE float ngw_min(float a,float b) {
  unsigned int ia=__float_as_uint(a),ib=__float_as_uint(b),chosen;
  if((ia&0x7fffffffu)>0x7f800000u || (ib&0x7fffffffu)>0x7f800000u)
    return __uint_as_float(ia);
  // Prevent NVRTC's min/max recognition from canonicalizing operand NaNs.
  asm volatile("{.reg .pred p; setp.lt.f32 p,%1,%2; selp.b32 %0,%3,%4,p;}"
               : "=r"(chosen) : "f"(a),"f"(b),"r"(ia),"r"(ib));
  return __uint_as_float(chosen);
}
NGW_DEVICE float ngw_max(float a,float b) {
  unsigned int ia=__float_as_uint(a),ib=__float_as_uint(b),chosen;
  if((ia&0x7fffffffu)>0x7f800000u || (ib&0x7fffffffu)>0x7f800000u)
    return __uint_as_float(ia);
  asm volatile("{.reg .pred p; setp.gt.f32 p,%1,%2; selp.b32 %0,%3,%4,p;}"
               : "=r"(chosen) : "f"(a),"f"(b),"r"(ia),"r"(ib));
  return __uint_as_float(chosen);
}
#else
NGW_DEVICE float ngw_min(float a,float b) {return a!=a || b!=b ? a : (a<b?a:b);}
NGW_DEVICE float ngw_max(float a,float b) {return a!=a || b!=b ? a : (a>b?a:b);}
#endif
NGW_DEVICE float ngw_min(float a,int b) {return ngw_min(a,float(b));}
NGW_DEVICE float ngw_max(float a,int b) {return ngw_max(a,float(b));}

// cires_ugwpv1_initialize.F:65-100, init_nazdir; cardinal directions are exact for naz=4.
NGW_DEVICE void ngw_init_nazdir(int naz,float pi2,float *xaz,float *yaz) {
  float drad=ngw_div(pi2,float(naz));
  if(naz!=4) {
    for(int i=1;i<=naz;++i) {
      float phic=ngw_mul(drad,ngw_sub(float(i),1.f));
      xaz[i]=ngw_cos(phic);yaz[i]=ngw_sin(phic);
    }
  } else {
    xaz[1]=1.f;yaz[1]=0.f;xaz[2]=0.f;yaz[2]=1.f;
    xaz[3]=-1.f;yaz[3]=0.f;xaz[4]=0.f;yaz[4]=-1.f;
  }
}
// cires_ugwpv1_initialize.F:216-269,302-334,363-391, full default source spectra.
NGW_DEVICE void ngw_init_sources(float pi2,float *out) {
  float ch[NW+1],spf[NW+1],xaz[NA+1],yaz[NA+1];
  const float bns=.02f,bns2=bns*bns,bns4=bns2*bns2,mstar=6.28e-3f/2.f;
  float dc=ngw_div(ngw_sub(95.f,5.f),float(NW-1));
  float snorm=0.f;
  for(int k=1;k<=NW;++k) {
    float chk=ngw_add(5.f,ngw_mul(float(k-1),dc));
    float m=ngw_mul(mstar,chk),m2=ngw_mul(m,m),tn4=ngw_mul(m2,m2);
    ch[k]=chk;spf[k]=ngw_div(ngw_mul(bns4,chk),ngw_add(bns4,tn4));
  }
  for(int k=1;k<=NW;++k) snorm=ngw_add(snorm,spf[k]);
  for(int k=1;k<=NW;++k) spf[k]=ngw_mul(ngw_div(spf[k],snorm),1.5f);
  ngw_init_nazdir(NA,pi2,xaz,yaz);
  for(int k=1;k<=NW;++k) {out[k-1]=ch[k];out[NW+k-1]=spf[k];}
  for(int a=1;a<=NA;++a) {out[2*NW+a-1]=xaz[a];out[2*NW+NA+a-1]=yaz[a];}
  for(int source=1;source<=2;++source) {
    float jetdc=ngw_div(ngw_sub(67.5f,2.5f),float(NW-1));
    for(int k=1;k<=NW;++k) {
      out[source*(2*NW+2*NA)+k-1]=ngw_add(2.5f,ngw_mul(float(k-1),jetdc));
      out[source*(2*NW+2*NA)+NW+k-1]=1.f;
    }
    for(int a=1;a<=NA;++a) {
      out[source*(2*NW+2*NA)+2*NW+a-1]=xaz[a];
      out[source*(2*NW+2*NA)+2*NW+NA+a-1]=yaz[a];
    }
  }
}
// cires_ugwpv1_initialize.F:416-440, unused default alternative solver metadata initialization.
NGW_DEVICE void ngw_initsolv_lsatdis(int nwaves,int nazdir,int nstoch,float effac,
                                    int &nwav,int &nazd,int &nst,float &eff) {
  if(nwaves==0 || nstoch==1) {nwav=4;nazd=4;nst=0;eff=1.f;}
  else {nwav=nwaves;nazd=nazdir;nst=nstoch;eff=effac;}
}

// cires_ugwpv1_solv2.F:17-1020; ugwpv1_ngw_init/run in bl_ugwpv1_ngw.F:30-236.
NGW_DEVICE void ngw_column(int col,int nc,int ntlat,int ntday,float dtp,float amp,float day,
const float *rdzw,const float *dzu,const float *lat_r,const float *taulat,const float *days,const float *tau,
const float *um,const float *vm,const float *tm,const float *qm,const float *prsl,const float *prslk,
const float *prsi,const float *zmet,const float *zmeti,const float *raincv,const float *rainncv,
float *rublten,float *rvblten,float *rthblten,float *pdudt,float *pdvdt,float *pdtdt,float *dked,float *zngw,
float *init_state,int *index_state,float *weight_state,int *status,
const float *native_xlat=0,const int *native_indices=0,const float *native_weights=0,int *launch_state=0,bool initialize_only=false) {
// bl_ugwpv1_ngw.F:54-104, native single-precision constants and zeta centers.
const int levs=NL, nwav=NW, nazd=NA;
int ilaunch=55;
const float pi=3.141592653589793f, grav=9.80616f, cpd=1004.5f, rd=287.f, rv=461.6f;
const float omega1=7.29212e-5f;
const float fv=ngw_sub(ngw_div(rv,rd),1.f);
const float grav2=ngw_add(grav,grav), rgrav=ngw_div(1.f,grav), rgrav2=ngw_mul(rgrav,rgrav);
const float rdi=ngw_div(1.f,rd), rcpd=ngw_div(1.f,cpd), rcpd2=ngw_div(.5f,cpd);
const float gor=ngw_div(grav,rd), gr2=ngw_mul(grav,gor), grcp=ngw_mul(grav,rcpd), gocp=grcp;
const float rcpdl=ngw_mul(cpd,rgrav), grav2cpd=ngw_mul(grav,grcp);
const float pi2=ngw_mul(2.f,pi), pih=ngw_mul(.5f,pi);
const float rad_to_deg=ngw_div(180.f,pi), deg_to_rad=ngw_div(pi,180.f);
const float bnv2min=ngw_mul(ngw_div(pi2,1800.f),ngw_div(pi2,1800.f));
const float bnv2max=ngw_mul(ngw_div(pi2,30.f),ngw_div(pi2,30.f));
const float dw2min=1.f, omega2=ngw_mul(2.f,omega1);
const float hpscale=7000.f, rhp=ngw_div(1.f,hpscale), rh4=ngw_mul(.25f,rhp);
const float mkzmin=ngw_div(pi2,80000.f), mkz2min=ngw_mul(mkzmin,mkzmin);
const float mkzmax=ngw_div(pi2,500.f), ucrit=ngw_div(.02f,mkzmax);
// cires_ugwpv1_module.F:29-76,115-154, fixed MPAS module controls.
const float maxdudt=450.e-5f, max_eps=450.f*4.e-4f, maxdtdt=(450.f*4.e-4f)*1.e-3f;
const float dked_min=.01f, dked_max=450.f, ipr_ktgw=1.f/3.f, pr_kdis=1.f/(1.f+1.f);
const int knob_ugwp_doheat=1, knob_ugwp_dokdis=2;
const float psrc=27500.f, nslope=1.f, lzstar=2000.f, lzmax=15750.f, lzmin=750.f, lhmet=200000.f;
const float tau_min=.25e-3f, gw_eff=1.f, sc2=30.f*30.f;
float init_zu[NL+2],pmb[NL+2],zkm[NL+2];
float kvg[NL+2],ktg[NL+2],krad[NL+2],kion[NL+2];
float zci[NW+1],zdci[NW+1],zcosang[NA+1],zsinang[NA+1];
init_zu[1]=ngw_div(.5f,rdzw[0]);
for(int k=2;k<=levs;++k) init_zu[k]=ngw_add(init_zu[k-1],dzu[k-1]);
// cires_ugwpv1_module.F:214-245, background pressure and initial launch layer.
for(int k=1;k<=levs;++k) {
  zkm[k]=ngw_mul(.001f,init_zu[k]);
  pmb[k]=ngw_mul(101325.f,ngw_exp(ngw_div(ngw_neg(init_zu[k]),hpscale)));
}
int initial_k;
for(initial_k=levs;initial_k>=1;--initial_k) if(pmb[initial_k]>psrc) break;
ilaunch=ngw_max(initial_k-1,5);
// cires_ugwpv1_initialize.F:103-178, init_global_gwdis, including super RF.
const float vusurf=2.e-5f, musurf=2.e-5f/1.95f, alpha=1.f/86400.f/15.f, kdrag=1.f/86400.f/30.f;
float rtau_fv3=ngw_div(ngw_div(1.f,86400.f),10.f);
float pih_dlog=ngw_div(pih,ngw_log(ngw_div(750.f,pmb[levs])));
for(int k=1;k<=levs;++k) {
  float ae1=ngw_div(zkm[k],7.f), ae2=ngw_div(ngw_sub(zkm[k],130.f),30.f);
  float vumol=ngw_mul(vusurf,ngw_exp(ae1));
  float mumol=ngw_mul(musurf,ngw_exp(ae1));
  float keddy=ngw_mul(100.f,ngw_exp(ngw_neg(ngw_mul(ae2,ae2))));
  kvg[k]=ngw_add(vumol,keddy); ktg[k]=ngw_add(mumol,ngw_mul(keddy,3.f));
  krad[k]=alpha; kion[k]=kdrag;
  if(pmb[k]<=750.f) {
    float s=ngw_sin(ngw_mul(pih_dlog,ngw_log(ngw_div(750.f,pmb[k]))));
    float rf=ngw_mul(rtau_fv3,ngw_mul(s,s));
    krad[k]=ngw_add(krad[k],rf); kion[k]=ngw_add(kion[k],rf);
  }
}
kion[levs+1]=kion[levs]; krad[levs+1]=krad[levs]; kvg[levs+1]=kvg[levs]; ktg[levs+1]=ktg[levs];
// cires_ugwpv1_initialize.F:545-646, initsolv_wmsdis, MPAS version=1.
const float v_kxw=ngw_div(pi2,lhmet), rv_kxw=ngw_div(1.f,v_kxw), zms=ngw_div(pi2,lzstar);
float zang=ngw_div(pi2,float(nazd)), znorm=0.f;
for(int a=1;a<=nazd;++a) {
  float zang1=ngw_mul(float(a-1),zang);
  zcosang[a]=ngw_cos(zang1); zsinang[a]=ngw_sin(zang1);
  znorm=ngw_add(znorm,ngw_abs(zcosang[a]));
}
float zaz_fct=ngw_div(2.f,znorm);
// cires_ugwpv1_initialize.F:588-615, native Scinocca stretching, before version=1 overwrites it.
float zxmax=ngw_div(1.f,2.5f),zxmin=ngw_div(1.f,125.f);
float zxran=ngw_sub(zxmax,zxmin), zdx0=ngw_div(zxran,float(nwav-1));
float ae1=ngw_div(zxran,.25f);
float zx1=ngw_div(zxran,ngw_sub(ngw_exp(ae1),1.f)),zx2=ngw_sub(zxmin,zx1);
for(int w=1;w<=nwav;++w) {
  float ztx=ngw_add(ngw_mul(float(w-1),zdx0),zxmin);
  float a=ngw_div(ngw_sub(ztx,zxmin),.25f);
  float zx=ngw_add(ngw_mul(zx1,ngw_exp(a)),zx2);
  zci[w]=ngw_div(1.f,zx);
  zdci[w]=ngw_mul(ngw_mul(ngw_mul(ngw_mul(zci[w],zci[w]),ngw_div(zx1,.25f)),ngw_exp(a)),zdx0);
}
// cires_ugwpv1_initialize.F:621-642, version is fixed to 1 by native module controls.
const float bnfix=6.28f/300.f;
float dlzmet=ngw_div(ngw_sub(lzmax,lzmin),float(nwav-1));
for(int w=1;w<=nwav;++w) {
  float lz=ngw_add(lzmin,ngw_mul(float(w-1),dlzmet));
  zci[w]=ngw_div(lz,ngw_div(pi2,bnfix));
}
float zdx=ngw_div(ngw_sub(zci[nwav],zci[1]),float(nwav-1));
for(int w=1;w<=nwav;++w) zdci[w]=zdx;
if(init_state && col==0) {
  ngw_init_sources(pi2,init_state+2*nwav+2*nazd+2*(levs+1));
  for(int w=1;w<=nwav;++w) {init_state[w-1]=zci[w];init_state[nwav+w-1]=zdci[w];}
  for(int a=1;a<=nazd;++a) {init_state[2*nwav+a-1]=zcosang[a];init_state[2*nwav+nazd+a-1]=zsinang[a];}
  for(int k=1;k<=levs+1;++k) {init_state[2*nwav+2*nazd+k-1]=kvg[k];init_state[2*nwav+2*nazd+levs+1+k-1]=kion[k];}
}
// cires_tauamf_data.F:14-57, cires_indx_ugwp latitude indices (one based).
float lat_deg=ngw_mul(rad_to_deg,lat_r[col]);
int j2=ntlat;
for(int i=1;i<=ntlat;++i) if(lat_deg<taulat[i-1]) {j2=i;break;}
j2=ngw_min(j2,ntlat); int j1=ngw_max(j2-1,1);
float lat_w2=j1!=j2?ngw_div(ngw_sub(lat_deg,taulat[j1-1]),ngw_sub(taulat[j2-1],taulat[j1-1])):1.f;
float lat_w1=ngw_sub(1.f,lat_w2);
// bl_ugwpv1_ngw.F:132-157, preserve the inner driver's saved latitude arguments.
if(native_indices) {j1=native_indices[col];j2=native_indices[nc+col];
lat_w1=native_weights[col];lat_w2=native_weights[nc+col];}
if(index_state) {index_state[col]=j1;index_state[nc+col]=j2;}
if(weight_state) {weight_state[col]=lat_w1;weight_state[nc+col]=lat_w2;}
if(initialize_only) return;
// mpas_atmphys_driver_gwdo.F:304,474-511, intentionally different rad2deg constant.
float xlat_value=ngw_mul(lat_r[col],180.f/3.1415926f);
if(native_xlat) xlat_value=native_xlat[col];
float local_xlat[1]={xlat_value};
// cires_tauamf_data.F:60-110, tau_amf_interp; one time search per column.
// cires_tauamf_data.F:82-110: outside the final DAYS entry native leaves
// it2 unset. Published strict runs sometimes fault, but every completed
// probe has the ordinary finite NGW rates (the climate multiplier is zero).
// Continue with the final bracket; this is not an identity claim for a
// native run which faults before producing affected words.
int it2=ntday;
for(int i=1;i<=ntday;++i) if(day<days[i-1]) {it2=i;break;}
// Before the first entry, retain native's zero denominator and IEEE result.
it2=ngw_min(it2,ntday);int it1=ngw_max(it2-1,1);
float tw2=ngw_div(ngw_sub(day,days[it1-1]),ngw_sub(days[it2-1],days[it1-1]));
float tw1=ngw_sub(1.f,tw2);
float tx1_source=ngw_add(ngw_mul(tau[(it1-1)*ntlat+j1-1],lat_w1),ngw_mul(tau[(it1-1)*ntlat+j2-1],lat_w2));
float tx2_source=ngw_add(ngw_mul(tau[(it2-1)*ntlat+j1-1],lat_w1),ngw_mul(tau[(it2-1)*ntlat+j2-1],lat_w2));
float tau_amf=ngw_add(ngw_mul(tx1_source,tw1),ngw_mul(tw2,tx2_source));
// cires_ugwpv1_triggers.F:11-62, slat_geos5_2020.
float latdeg=ngw_abs(xlat_value),flat_gw,tem;
if(latdeg<15.3f) {
  tem=ngw_div(ngw_sub(latdeg,3.f),12.5f);
  flat_gw=ngw_mul(1.25f,ngw_exp(ngw_mul(ngw_neg(tem),tem)));
  if(latdeg<=3.f) flat_gw=1.25f;
} else if(latdeg<31.f && latdeg>=15.3f) flat_gw=.25f;
else if(latdeg<60.f && latdeg>=31.f) {
  tem=ngw_div(ngw_sub(latdeg,60.f),23.f); flat_gw=ngw_exp(ngw_mul(ngw_neg(tem),tem));
} else {
  tem=ngw_div(ngw_sub(latdeg,60.f),25.f); flat_gw=ngw_exp(ngw_mul(ngw_neg(tem),tem));
}
if(xlat_value<=-31.f) {
  if(latdeg<60.f && latdeg>=31.f) {tem=ngw_div(ngw_sub(latdeg,60.f),23.f);flat_gw=ngw_exp(ngw_mul(ngw_neg(tem),tem));}
  if(latdeg>=60.f) {tem=ngw_div(ngw_sub(latdeg,60.f),30.f);flat_gw=ngw_exp(ngw_mul(ngw_neg(tem),tem));}
}
float tau_value=ngw_mul(amp,flat_gw);
// bl_ugwpv1_ngw.F:183-204 and cires_ugwpv1_module.F:326-383, rain conversion and ngwflux_update.
float rain=ngw_mul(ngw_add(raincv[col],rainncv[col]),.001f);
float tau_3dt=ngw_add(ngw_mul(tau_value,1.f),ngw_mul(0.f,tau_amf));
if(rain>0.f) {
  float wlat=ngw_abs(xlat_value);
  if(wlat<=41.f && rain>1.e-5f) {
    float rw1=ngw_mul(.75f,ngw_div(wlat,41.f)),rw2=ngw_sub(1.f,rw1);
    float tau_rain=ngw_add(ngw_mul(tau_3dt,rw1),ngw_div(ngw_mul(ngw_mul(rw2,.001f),ngw_min(8.e-5f,rain)),1.e-5f));
    tau_rain=ngw_add(ngw_mul(tau_3dt,0.f),ngw_mul(1.f,tau_rain));
    if(tau_rain<ngw_mul(.5f,tau_3dt)) tau_rain=ngw_mul(.5f,tau_3dt);
    if(tau_rain>ngw_mul(2.f,tau_3dt)) tau_rain=ngw_mul(2.f,tau_3dt);
    tau_3dt=tau_rain;
  }
}
float local_tau[1]={tau_3dt};
const float zsp_gw=106500.f;
const float linsat2=1.f;
const float dturb_max=100.f;
const int ener_norm=0;
const int ener_lsat=0;
const int nstdif=1;
const int wave_sponge=1;
float apc;
float bdif;
float bt_dif;
float bvi;
float bvi2;
float bvi3;
float bvi4;
float c2f2;
float cdf1;
float cdf2;
float cf1;
float cgz;
float dtau;
float dtdif;
float dtstab;
float dzcrit;
float dzmetf;
float dzmetm;
float dzmetp;
float ek1;
float ek2;
float ekin_norm;
float exp_sponge;
float expdis;
float fcorsat;
float fdis;
float flux_2_sig;
float flux_norm;
float fluxs;
float fmode;
float gipr;
int iaz;
int inc;
float ipr_max;
int ist;
int j;
int jj;
int jk;
int jkp;
int jl;
int k;
float kamp;
float kdsat;
int kk;
int km1;
int km2;
int kp1;
int ksrc;
int ktop;
float ktur;
float kturp;
float kzw2;
float kzw3;
float mf_diss_heat;
float mi_sponge;
float nslope3;
int nstab;
float pr_kdis_eff;
float ptc;
float ptm;
float ptrms;
float pwrms;
float rcms;
float rdtp;
float rdtp2;
float rho_src;
float rhp_wam;
float ritur;
float rstar;
float rstar2;
float shr2;
float sigu2;
float sigu2_ch;
float snorm_ener;
float stab;
float stab_dt;
float taub_ch;
float taub_src;
float taud_min;
float tvc;
float tvm;
float tx1;
float ucrit3;
float ucrit_max;
float ucrit_maxdc;
float ugw;
float umfc;
float umfm;
float umfp;
float uz;
float v_cdp;
float v_cdp2;
float v_kzi;
float v_kzw;
float v_wdi;
float v_wdp;
float v_wdpc;
float vc_zflx_mode;
float vgw;
float vm_zflx_mode;
float vmdiff;
float vz;
float w1;
float w2;
float w3;
float wave_distot;
float wdop2;
float zatmp;
float zcin;
float zcin2;
float zcin3;
float zcin4;
float zcinc;
float zcool;
float zdelm;
float zdelp;
float zdep;
float ze1;
float ze2;
float zgrow;
float zmetk;
float zthm;
float zu;
float sinlat[NL+2];
float wrk1[NL+2];
float wrk2[NL+2];
float taux[NL+2];
float tauy[NL+2];
float fden_bn[NL+2];
float bn[NL+2];
float bn2[NL+2];
float rhoint[NL+2];
float uint[NL+2];
float vint[NL+2];
float tint[NL+2];
float irhodz_mid[NL+2];
float suprf[NL+2];
float cstar[NL+2];
float cstar2[NL+2];
float v_zmet[NL+2];
float vueff[NL+2];
float atm[NL+2];
float aum[NL+2];
float avm[NL+2];
float aqm[NL+2];
float aprsl[NL+2];
float azmet[NL+2];
float dz_met[NL+2];
float aprsi[NL+2];
float azmeti[NL+2];
float dz_meti[NL+2];
float wrk3[NL+2];
float uold[NL+2];
float vold[NL+2];
float told[NL+2];
float unew[NL+2];
float vnew[NL+2];
float tnew[NL+2];
float rho[NL+2];
float rhomid[NL+2];
float adif[NL+2];
float cdif[NL+2];
float acdif[NL+2];
float qmid[NL+2];
float akt[NL+2];
float dktur[NL+2];
float ktint[NL+2];
float kvint[NL+2];
float fden_lsat[NL+2];
float fden_bnen[NL+2];
int anstab[NL+2];
float sig_u2az[NA+1];
float sig_u2az_m[NA+1];
float wave_disaz[NA+1];
float rdci[NW+1];
float rci[NW+1];
float ul[NA+1];
float zwrms[NA+1][NW+1];
float atrms[NL+2][NA+1];
float awrms[NL+2][NA+1];
float fpu[NL+2][NA+1];
float ui[NL+2][NA+1];
float flux[NA+1][NW+1];
float flux_m[NA+1][NW+1];
float dfdz_v[NL+2][NA+1];
float dfdz_heat[NL+2][NA+1];
float wave_dis[NA+1][NW+1];
float wave_act[NA+1][NW+1];
float akzw[NL+2][NA+1][NW+1];
// cires_ugwpv1_solv2.F:195
nslope3 = ngw_add(nslope, 3.0f);
// cires_ugwpv1_solv2.F:196
pr_kdis_eff = ngw_mul(gw_eff, pr_kdis);
// cires_ugwpv1_solv2.F:197
ipr_max = ngw_max(1.0f,ipr_ktgw);
// cires_ugwpv1_solv2.F:198
gipr = ngw_mul(grav, ipr_ktgw);
// cires_ugwpv1_solv2.F:205
for(int s0=1;s0<=nwav;++s0) { rci[s0] = ngw_div(1.0f, zci[s0]); }
// cires_ugwpv1_solv2.F:206
for(int s0=1;s0<=nwav;++s0) { rdci[s0] = ngw_div(1.0f, zdci[s0]); }
// cires_ugwpv1_solv2.F:208
rdtp = ngw_div(1.0f, dtp);
// cires_ugwpv1_solv2.F:209
rdtp2 = ngw_mul(0.5f, rdtp);
// cires_ugwpv1_solv2.F:211
ksrc = ngw_max(ilaunch,3);
// cires_ugwpv1_solv2.F:212
km2 = (ksrc-2);
// cires_ugwpv1_solv2.F:213
km1 = (ksrc-1);
// cires_ugwpv1_solv2.F:214
kp1 = (ksrc+1);
// cires_ugwpv1_solv2.F:215
ktop = (levs+1);
// cires_ugwpv1_solv2.F:217
suprf[ktop] = kion[levs];
// cires_ugwpv1_solv2.F:219
for(k=1;k<=levs;k+=1) {
// cires_ugwpv1_solv2.F:220
suprf[k] = kion[k];
// cires_ugwpv1_solv2.F:221
for(int s0=1;s0<=1;++s0) { pdvdt[((k-1)*nc)+col] = 0.0f; }
// cires_ugwpv1_solv2.F:222
for(int s0=1;s0<=1;++s0) { pdudt[((k-1)*nc)+col] = 0.0f; }
// cires_ugwpv1_solv2.F:223
for(int s0=1;s0<=1;++s0) { pdtdt[((k-1)*nc)+col] = 0.0f; }
// cires_ugwpv1_solv2.F:224
for(int s0=1;s0<=1;++s0) { dked[((k-1)*nc)+col] = 0.0f; }
// cires_ugwpv1_solv2.F:225
}
// cires_ugwpv1_solv2.F:228
for(j=1;j<=1;j+=1) {
// cires_ugwpv1_solv2.F:229
sinlat[j] = ngw_sin(ngw_mul(xlat_value, deg_to_rad));
// cires_ugwpv1_solv2.F:230
}
// cires_ugwpv1_solv2.F:235
for(j=1;j<=1;j+=1) {
// cires_ugwpv1_solv2.F:236
jl = j;
// cires_ugwpv1_solv2.F:237
tx1 = ngw_mul(ngw_mul(omega2, sinlat[j]), rv_kxw);
// cires_ugwpv1_solv2.F:238
cf1 = ngw_abs(tx1);
// cires_ugwpv1_solv2.F:239
c2f2 = ngw_mul(tx1, tx1);
// cires_ugwpv1_solv2.F:240
ucrit_max = ngw_max(ucrit,cf1);
// cires_ugwpv1_solv2.F:241
ucrit3 = ngw_mul(ngw_mul(ucrit_max, ucrit_max), ucrit_max);
// cires_ugwpv1_solv2.F:245
for(int s0=1;s0<=levs;++s0) { aprsl[s0] = prsl[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:249
for(k=1;k<=levs;k+=1) {
// cires_ugwpv1_solv2.F:250
if((aprsl[k]<psrc)) { break; }
// cires_ugwpv1_solv2.F:251
}
// cires_ugwpv1_solv2.F:252
ilaunch = ngw_max(ngw_sub(k, 1),3);
// cires_ugwpv1_solv2.F:253
ksrc = ngw_max(ilaunch,3);
// cires_ugwpv1_solv2.F:255
zngw[col] = zmet[((ksrc-1)*nc)+col];
// cires_ugwpv1_solv2.F:257
km2 = (ksrc-2);
// cires_ugwpv1_solv2.F:258
km1 = (ksrc-1);
// cires_ugwpv1_solv2.F:259
kp1 = (ksrc+1);
// cires_ugwpv1_solv2.F:263
for(int s0=1;s0<=levs;++s0) { aum[s0] = um[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:264
for(int s0=1;s0<=levs;++s0) { avm[s0] = vm[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:265
for(int s0=1;s0<=levs;++s0) { atm[s0] = tm[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:266
for(int s0=1;s0<=levs;++s0) { aqm[s0] = qm[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:267
for(int s0=1;s0<=levs;++s0) { azmet[s0] = zmet[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:268
for(int s0=1;s0<=(levs+1);++s0) { aprsi[s0] = prsi[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:269
for(int s0=1;s0<=(levs+1);++s0) { azmeti[s0] = zmeti[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:271
rho_src = ngw_div(ngw_mul(aprsl[ksrc], rdi), atm[ksrc]);
// cires_ugwpv1_solv2.F:272
taub_ch = ngw_max(tau_3dt,tau_min);
// cires_ugwpv1_solv2.F:273
taub_src = taub_ch;
// cires_ugwpv1_solv2.F:276
sigu2 = ngw_mul(ngw_div(ngw_div(taub_src, rho_src), v_kxw), zms);
// cires_ugwpv1_solv2.F:277
for(int s0=1;s0<=nazd;++s0) { sig_u2az[s0] = sigu2; }
// cires_ugwpv1_solv2.F:281
for(jk=km2;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:282
dz_meti[jk] = ngw_sub(azmeti[(jk+1)], azmeti[jk]);
// cires_ugwpv1_solv2.F:283
// Native reads azmeti(0) at jk=1. dz_met(1) is dead: all consumers start
// at km1=ksrc-1>=2 (lines 288-304,860-865,907-912,984-1018).
if(jk>1) dz_met[jk] = ngw_sub(azmet[jk], azmeti[(jk-1)]);
// cires_ugwpv1_solv2.F:284
}
// cires_ugwpv1_solv2.F:288
for(jk=km1;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:289
tvc = ngw_mul(atm[jk], ngw_add(1.0f, ngw_mul(fv, aqm[jk])));
// cires_ugwpv1_solv2.F:290
tvm = ngw_mul(atm[(jk-1)], ngw_add(1.0f, ngw_mul(fv, aqm[(jk-1)])));
// cires_ugwpv1_solv2.F:291
ptc = ngw_div(tvc, prslk[((jk-1)*nc)+col]);
// cires_ugwpv1_solv2.F:292
ptm = ngw_div(tvm, prslk[(((jk-1)-1)*nc)+col]);
// cires_ugwpv1_solv2.F:294
zthm = ngw_div(2.0f, ngw_add(tvc, tvm));
// cires_ugwpv1_solv2.F:295
rhp_wam = ngw_mul(zthm, gor);
// cires_ugwpv1_solv2.F:297
uint[jk] = ngw_mul(0.5f, ngw_add(aum[(jk-1)], aum[jk]));
// cires_ugwpv1_solv2.F:298
vint[jk] = ngw_mul(0.5f, ngw_add(avm[(jk-1)], avm[jk]));
// cires_ugwpv1_solv2.F:299
tint[jk] = ngw_mul(0.5f, ngw_add(tvc, tvm));
// cires_ugwpv1_solv2.F:300
rhomid[jk] = ngw_div(ngw_mul(aprsl[jk], rdi), atm[jk]);
// cires_ugwpv1_solv2.F:301
rhoint[jk] = ngw_mul(ngw_mul(aprsi[jk], rdi), zthm);
// cires_ugwpv1_solv2.F:302
zdelp = dz_meti[jk];
// cires_ugwpv1_solv2.F:303
v_zmet[jk] = ngw_mul(2.0f, zdelp);
// cires_ugwpv1_solv2.F:304
zdelm = ngw_div(1.0f, dz_met[jk]);
// cires_ugwpv1_solv2.F:308
bn2[jk] = ngw_mul(ngw_mul(grav2cpd, zthm), ngw_add(1.0f, ngw_mul(ngw_mul(rcpdl, ngw_sub(tvc, tvm)), zdelm)));
// cires_ugwpv1_solv2.F:309
bn2[jk] = ngw_max(ngw_min(bn2[jk],bnv2max),bnv2min);
// cires_ugwpv1_solv2.F:310
bn[jk] = ngw_sqrt(bn2[jk]);
// cires_ugwpv1_solv2.F:313
wrk3[jk] = ngw_div(ngw_div(1.0f, zdelp), rhomid[jk]);
// cires_ugwpv1_solv2.F:314
irhodz_mid[jk] = ngw_div(ngw_mul(ngw_mul(rdtp, zdelp), rhomid[jk]), rho_src);
// cires_ugwpv1_solv2.F:319
uz = ngw_sub(aum[jk], aum[(jk-1)]);
// cires_ugwpv1_solv2.F:320
vz = ngw_sub(avm[jk], avm[(jk-1)]);
// cires_ugwpv1_solv2.F:321
shr2 = ngw_mul(ngw_mul(ngw_max(ngw_add(ngw_mul(uz, uz), ngw_mul(vz, vz)),dw2min), zdelm), zdelm);
// cires_ugwpv1_solv2.F:323
zmetk = ngw_mul(azmet[jk], rh4);
// cires_ugwpv1_solv2.F:324
zgrow = ngw_exp(zmetk);
// cires_ugwpv1_solv2.F:325
ritur = ngw_div(bn2[jk], shr2);
// cires_ugwpv1_solv2.F:326
kamp = ngw_mul(ngw_mul(ngw_sqrt(shr2), sc2), zgrow);
// cires_ugwpv1_solv2.F:327
w1 = ngw_div(1.0f, ngw_add(1.0f, ngw_mul(5, ritur)));
// cires_ugwpv1_solv2.F:328
ktur = ngw_min(ngw_max(ngw_mul(ngw_mul(kamp, w1), w1),dked_min),dked_max);
// cires_ugwpv1_solv2.F:329
zmetk = ngw_mul(azmet[jk], rhp);
// cires_ugwpv1_solv2.F:330
vueff[jk] = ngw_add(ktur, kvg[jk]);
// cires_ugwpv1_solv2.F:332
akt[jk] = ngw_div(gipr, tvc);
// cires_ugwpv1_solv2.F:333
}
// cires_ugwpv1_solv2.F:345
jk = levs;
// cires_ugwpv1_solv2.F:347
rhoint[ktop] = ngw_div(ngw_mul(ngw_mul(0.5f, aprsi[levs]), rdi), atm[jk]);
// cires_ugwpv1_solv2.F:348
tint[ktop] = ngw_mul(atm[jk], ngw_add(1.0f, ngw_mul(fv, aqm[jk])));
// cires_ugwpv1_solv2.F:349
uint[ktop] = aum[jk];
// cires_ugwpv1_solv2.F:350
vint[ktop] = avm[jk];
// cires_ugwpv1_solv2.F:352
v_zmet[ktop] = v_zmet[jk];
// cires_ugwpv1_solv2.F:353
vueff[ktop] = vueff[jk];
// cires_ugwpv1_solv2.F:354
bn2[ktop] = bn2[jk];
// cires_ugwpv1_solv2.F:355
bn[ktop] = bn[jk];
// cires_ugwpv1_solv2.F:359
for(jk=km1;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:360
akt[jk] = ngw_mul(ngw_neg(akt[jk]), ngw_add(gor, ngw_div(ngw_sub(tint[(jk+1)], tint[jk]), dz_meti[jk])));
// cires_ugwpv1_solv2.F:361
}
// cires_ugwpv1_solv2.F:364
bvi = bn[ksrc];
// cires_ugwpv1_solv2.F:364
bvi2 = ngw_mul(bvi, bvi);
// cires_ugwpv1_solv2.F:365
bvi3 = ngw_mul(bvi2, bvi);
// cires_ugwpv1_solv2.F:365
bvi4 = ngw_mul(bvi2, bvi2);
// cires_ugwpv1_solv2.F:365
rcms = ngw_div(zms, bvi);
// cires_ugwpv1_solv2.F:369
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:370
ul[iaz] = ngw_add(ngw_mul(zcosang[iaz], uint[ksrc]), ngw_mul(zsinang[iaz], vint[ksrc]));
// cires_ugwpv1_solv2.F:371
}
// cires_ugwpv1_solv2.F:374
for(jk=ksrc;jk<=ktop;jk+=1) {
// cires_ugwpv1_solv2.F:375
cstar[jk] = ngw_div(bn[jk], zms);
// cires_ugwpv1_solv2.F:376
cstar2[jk] = ngw_mul(cstar[jk], cstar[jk]);
// cires_ugwpv1_solv2.F:378
fden_lsat[jk] = ngw_mul(ngw_mul(ngw_div(rhoint[jk], bn[jk]), v_kxw), linsat2);
// cires_ugwpv1_solv2.F:380
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:381
zu = ngw_add(ngw_mul(zcosang[iaz], uint[jk]), ngw_mul(zsinang[iaz], vint[jk]));
// cires_ugwpv1_solv2.F:382
ui[jk][iaz] = zu;
// cires_ugwpv1_solv2.F:383
}
// cires_ugwpv1_solv2.F:384
}
// cires_ugwpv1_solv2.F:386
rstar = ngw_div(1.0f, cstar[ksrc]);
// cires_ugwpv1_solv2.F:387
rstar2 = ngw_mul(rstar, rstar);
// cires_ugwpv1_solv2.F:392
for(int s0=km2;s0<=ktop;++s0) { for(int s1=1;s1<=nazd;++s1) { fpu[s0][s1] = 0.0f; } }
// cires_ugwpv1_solv2.F:394
for(inc=1;inc<=nwav;inc+=1) {
// cires_ugwpv1_solv2.F:396
zcin = ngw_mul(zci[inc], rstar);
// cires_ugwpv1_solv2.F:401
flux[1][inc] = ngw_div(ngw_mul(rstar, ngw_mul(zcin, zcin)), ngw_add(1.0f, ngw_pow(zcin,nslope3)));
// cires_ugwpv1_solv2.F:405
fpu[ksrc][1] = ngw_add(fpu[ksrc][1], ngw_mul(flux[1][inc], zdci[inc]));
// cires_ugwpv1_solv2.F:407
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:408
akzw[ksrc][iaz][inc] = ngw_mul(bvi, rci[inc]);
// cires_ugwpv1_solv2.F:409
}
// cires_ugwpv1_solv2.F:411
}
// cires_ugwpv1_solv2.F:415
flux_norm = ngw_div(taub_src, fpu[ksrc][1]);
// cires_ugwpv1_solv2.F:416
ze1 = ngw_mul(ngw_mul(ngw_div(ngw_mul(flux_norm, bvi), rhoint[ksrc]), rstar), rstar2);
// cires_ugwpv1_solv2.F:417
for(jk=ksrc;jk<=ktop;jk+=1) {
// cires_ugwpv1_solv2.F:418
fden_bn[jk] = ngw_div(ngw_mul(ze1, rhoint[jk]), bn[jk]);
// cires_ugwpv1_solv2.F:419
}
// cires_ugwpv1_solv2.F:421
for(inc=1;inc<=nwav;inc+=1) {
// cires_ugwpv1_solv2.F:422
flux[1][inc] = ngw_mul(flux_norm, flux[1][inc]);
// cires_ugwpv1_solv2.F:423
}
// cires_ugwpv1_solv2.F:457
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:458
fpu[ksrc][iaz] = taub_src;
// cires_ugwpv1_solv2.F:459
fpu[km1][iaz] = taub_src;
// cires_ugwpv1_solv2.F:460
}
// cires_ugwpv1_solv2.F:466
for(iaz=2;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:467
for(inc=1;inc<=nwav;inc+=1) {
// cires_ugwpv1_solv2.F:468
flux[iaz][inc] = flux[1][inc];
// cires_ugwpv1_solv2.F:469
}
// cires_ugwpv1_solv2.F:470
}
// cires_ugwpv1_solv2.F:489
for(int a=1;a<=nazd;++a) for(int w=1;w<=nwav;++w) wave_act[a][w]=1.f;
// cires_ugwpv1_solv2.F:491
for(jk=ksrc;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:493
jkp = (jk+1);
// cires_ugwpv1_solv2.F:495
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:497
sig_u2az_m[iaz] = sig_u2az[iaz];
// cires_ugwpv1_solv2.F:499
umfp = ui[jkp][iaz];
// cires_ugwpv1_solv2.F:500
umfm = ui[jk][iaz];
// cires_ugwpv1_solv2.F:501
umfc = ngw_mul(0.5f, ngw_add(umfm, umfp));
// cires_ugwpv1_solv2.F:503
dfdz_v[jk][iaz] = 0.0f;
// cires_ugwpv1_solv2.F:504
dfdz_heat[jk][iaz] = 0.0f;
// cires_ugwpv1_solv2.F:505
fpu[jkp][iaz] = 0.0f;
// cires_ugwpv1_solv2.F:506
sig_u2az[iaz] = 0.0f;
// cires_ugwpv1_solv2.F:509
for(inc=1;inc<=nwav;inc+=1) {
// cires_ugwpv1_solv2.F:510
flux_m[iaz][inc] = flux[iaz][inc];
// cires_ugwpv1_solv2.F:512
zcin = zci[inc];
// cires_ugwpv1_solv2.F:513
zcinc = rci[inc];
// cires_ugwpv1_solv2.F:515
if((wave_act[iaz][inc]==1.0f)) {
// cires_ugwpv1_solv2.F:522
v_cdp = ngw_sub(zcin, umfp);
// cires_ugwpv1_solv2.F:523
v_cdp2 = ngw_mul(v_cdp, v_cdp);
// cires_ugwpv1_solv2.F:524
cdf2 = ngw_sub(v_cdp2, c2f2);
// cires_ugwpv1_solv2.F:525
if(((v_cdp<=ucrit_max) || (cdf2<=0.0f))) {
// cires_ugwpv1_solv2.F:529
wave_act[iaz][inc] = 0.0f;
// cires_ugwpv1_solv2.F:530
akzw[jkp][iaz][inc] = ngw_div(pi, dz_meti[jk]);
// cires_ugwpv1_solv2.F:531
fluxs = 0.0f;
// cires_ugwpv1_solv2.F:532
flux[iaz][inc] = fluxs;
// cires_ugwpv1_solv2.F:534
} else {
// cires_ugwpv1_solv2.F:536
v_wdp = ngw_mul(v_kxw, v_cdp);
// cires_ugwpv1_solv2.F:537
wdop2 = ngw_mul(v_wdp, v_wdp);
// cires_ugwpv1_solv2.F:542
kzw2 = ngw_div(ngw_sub(bn2[jkp], wdop2), cdf2);
// cires_ugwpv1_solv2.F:546
if((kzw2>mkz2min)) {
// cires_ugwpv1_solv2.F:547
v_kzw = ngw_sqrt(kzw2);
// cires_ugwpv1_solv2.F:548
akzw[jkp][iaz][inc] = v_kzw;
// cires_ugwpv1_solv2.F:555
v_cdp = ngw_sqrt(cdf2);
// cires_ugwpv1_solv2.F:556
v_wdp = ngw_mul(v_kxw, v_cdp);
// cires_ugwpv1_solv2.F:557
v_wdi = ngw_add(ngw_mul(kzw2, vueff[jk]), kion[jk]);
// cires_ugwpv1_solv2.F:558
v_wdpc = ngw_sqrt(ngw_add(ngw_mul(v_wdp, v_wdp), ngw_mul(v_wdi, v_wdi)));
// cires_ugwpv1_solv2.F:559
v_kzi = ngw_div(ngw_mul(v_kzw, v_wdi), v_wdpc);
// cires_ugwpv1_solv2.F:562
ze1 = ngw_mul(v_kzi, v_zmet[jk]);
// cires_ugwpv1_solv2.F:564
if((ze1>=0.01f)) {
// cires_ugwpv1_solv2.F:565
expdis = ngw_max(ngw_exp(ngw_neg(ze1)),0.01f);
// cires_ugwpv1_solv2.F:566
} else {
// cires_ugwpv1_solv2.F:567
expdis = ngw_div(1.0f, ngw_add(1.0f, ze1));
// cires_ugwpv1_solv2.F:568
}
// cires_ugwpv1_solv2.F:571
wave_act[iaz][inc] = 1.0f;
// cires_ugwpv1_solv2.F:572
fmode = flux[iaz][inc];
// cires_ugwpv1_solv2.F:574
flux_2_sig = ngw_div(ngw_div(v_kzw, v_kxw), rhoint[jkp]);
// cires_ugwpv1_solv2.F:575
w1 = ngw_div(ngw_div(ngw_div(v_wdpc, kzw2), v_kzw), v_zmet[jk]);
// cires_ugwpv1_solv2.F:576
} else {
// cires_ugwpv1_solv2.F:578
expdis = 1.0f;
// cires_ugwpv1_solv2.F:579
v_kzw = mkzmin;
// cires_ugwpv1_solv2.F:581
v_cdp = 0.0f;
// cires_ugwpv1_solv2.F:582
wave_act[iaz][inc] = 0.0f;
// cires_ugwpv1_solv2.F:583
akzw[jkp][iaz][inc] = v_kzw;
// cires_ugwpv1_solv2.F:584
fmode = 0.0f;
// cires_ugwpv1_solv2.F:585
w1 = 0.0f;
// cires_ugwpv1_solv2.F:586
}
// cires_ugwpv1_solv2.F:589
fdis = ngw_mul(ngw_mul(fmode, expdis), wave_act[iaz][inc]);
// cires_ugwpv1_solv2.F:612
if((ener_norm==0)) { fluxs = ngw_mul(ngw_mul(fden_bn[jkp], cdf2), wave_act[iaz][inc]); }
// cires_ugwpv1_solv2.F:634
zdep = ngw_sub(fdis, fluxs);
// cires_ugwpv1_solv2.F:635
if((zdep>0.0f)) {
// cires_ugwpv1_solv2.F:637
ze1 = flux[iaz][inc];
// cires_ugwpv1_solv2.F:638
flux[iaz][inc] = fluxs;
// cires_ugwpv1_solv2.F:639
ze2 = ngw_mul(ngw_log(ngw_div(ze1, fluxs)), w1);
// cires_ugwpv1_solv2.F:641
} else {
// cires_ugwpv1_solv2.F:643
flux[iaz][inc] = fdis;
// cires_ugwpv1_solv2.F:644
}
// cires_ugwpv1_solv2.F:646
dtau = ngw_sub(flux_m[iaz][inc], flux[iaz][inc]);
// cires_ugwpv1_solv2.F:647
if((dtau<0)) {
// cires_ugwpv1_solv2.F:648
flux[iaz][inc] = flux_m[iaz][inc];
// cires_ugwpv1_solv2.F:649
}
// cires_ugwpv1_solv2.F:653
if((azmeti[jkp]>=zsp_gw)) {
// cires_ugwpv1_solv2.F:654
mi_sponge = ngw_div(0.5f, dz_meti[jk]);
// cires_ugwpv1_solv2.F:655
ze2 = ngw_mul(ngw_div(v_wdp, v_kzw), mi_sponge);
// cires_ugwpv1_solv2.F:656
v_wdi = ngw_add(ze2, ngw_mul(v_wdi, 0.25f));
// cires_ugwpv1_solv2.F:657
v_wdpc = ngw_sqrt(ngw_add(ngw_mul(v_wdp, v_wdp), ngw_mul(v_wdi, v_wdi)));
// cires_ugwpv1_solv2.F:658
v_kzi = ngw_div(ngw_mul(v_kzw, v_wdi), v_wdpc);
// cires_ugwpv1_solv2.F:660
ze1 = ngw_mul(v_kzi, v_zmet[jk]);
// cires_ugwpv1_solv2.F:661
exp_sponge = ngw_exp(ngw_neg(ze1));
// cires_ugwpv1_solv2.F:665
flux[iaz][inc] = ngw_mul(flux[iaz][inc], exp_sponge);
// cires_ugwpv1_solv2.F:666
}
// cires_ugwpv1_solv2.F:668
}
// cires_ugwpv1_solv2.F:669
}
// cires_ugwpv1_solv2.F:673
if((wave_act[iaz][inc]==1)) {
// cires_ugwpv1_solv2.F:675
zcinc = zdci[inc];
// cires_ugwpv1_solv2.F:676
vc_zflx_mode = flux[iaz][inc];
// cires_ugwpv1_solv2.F:677
vmdiff = ngw_max(0.0f,ngw_sub(flux_m[iaz][inc], vc_zflx_mode));
// cires_ugwpv1_solv2.F:678
if((vmdiff<=0.0f)) { vc_zflx_mode = flux_m[iaz][inc]; }
// cires_ugwpv1_solv2.F:679
ze1 = ngw_mul(vc_zflx_mode, zcinc);
// cires_ugwpv1_solv2.F:680
fpu[jkp][iaz] = ngw_add(fpu[jkp][iaz], ze1);
// cires_ugwpv1_solv2.F:681
sig_u2az[iaz] = ngw_add(sig_u2az[iaz], ngw_mul(ze1, flux_2_sig));
// cires_ugwpv1_solv2.F:690
zdelp = ngw_mul(ngw_mul(ngw_mul(wrk3[jk], v_cdp), zcinc), vmdiff);
// cires_ugwpv1_solv2.F:699
dfdz_v[jk][iaz] = ngw_add(dfdz_v[jk][iaz], zdelp);
// cires_ugwpv1_solv2.F:700
dfdz_heat[jk][iaz] = ngw_add(dfdz_heat[jk][iaz], zdelp);
// cires_ugwpv1_solv2.F:701
}
// cires_ugwpv1_solv2.F:703
}
// cires_ugwpv1_solv2.F:705
ze1 = fpu[jk][iaz];
// cires_ugwpv1_solv2.F:706
if((fpu[jkp][iaz]>ze1)) { fpu[jkp][iaz] = ze1; }
// cires_ugwpv1_solv2.F:726
}
// cires_ugwpv1_solv2.F:731
float sum_heat=0.f; for(int a=1;a<=nazd;++a) sum_heat=ngw_add(sum_heat,ngw_abs(dfdz_heat[jk][a])); tx1 = ngw_div(sum_heat, bn2[jk]);
// cires_ugwpv1_solv2.F:732
ze1 = ngw_max(dked_min,tx1);
// cires_ugwpv1_solv2.F:733
ze2 = ngw_min(dked_max,ze1);
// cires_ugwpv1_solv2.F:734
vueff[jkp] = ngw_add(ze2, vueff[jkp]);
// cires_ugwpv1_solv2.F:736
}
// cires_ugwpv1_solv2.F:752
for(jk=ksrc;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:753
taux[jk] = 0.0f;
// cires_ugwpv1_solv2.F:754
tauy[jk] = 0.0f;
// cires_ugwpv1_solv2.F:755
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:756
taux[jk] = ngw_add(taux[jk], ngw_mul(fpu[jk][iaz], zcosang[iaz]));
// cires_ugwpv1_solv2.F:757
tauy[jk] = ngw_add(tauy[jk], ngw_mul(fpu[jk][iaz], zsinang[iaz]));
// cires_ugwpv1_solv2.F:758
pdtdt[((jk-1)*nc)+col] = ngw_add(pdtdt[((jk-1)*nc)+col], dfdz_v[jk][iaz]);
// cires_ugwpv1_solv2.F:759
dked[((jk-1)*nc)+col] = ngw_add(dked[((jk-1)*nc)+col], dfdz_heat[jk][iaz]);
// cires_ugwpv1_solv2.F:760
}
// cires_ugwpv1_solv2.F:761
}
// cires_ugwpv1_solv2.F:762
jk = ktop;
// cires_ugwpv1_solv2.F:762
taux[jk] = 0.0f;
// cires_ugwpv1_solv2.F:762
tauy[jk] = 0.0f;
// cires_ugwpv1_solv2.F:763
for(iaz=1;iaz<=nazd;iaz+=1) {
// cires_ugwpv1_solv2.F:764
taux[jk] = ngw_add(taux[jk], ngw_mul(fpu[jk][iaz], zcosang[iaz]));
// cires_ugwpv1_solv2.F:765
tauy[jk] = ngw_add(tauy[jk], ngw_mul(fpu[jk][iaz], zsinang[iaz]));
// cires_ugwpv1_solv2.F:766
}
// cires_ugwpv1_solv2.F:779
for(jk=(ksrc+1);jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:780
jkp = (jk+1);
// cires_ugwpv1_solv2.F:781
zdelp = ngw_mul(wrk3[jk], gw_eff);
// cires_ugwpv1_solv2.F:782
ze1 = ngw_mul(ngw_sub(taux[jkp], taux[jk]), zdelp);
// cires_ugwpv1_solv2.F:783
ze2 = ngw_mul(ngw_sub(tauy[jkp], tauy[jk]), zdelp);
// cires_ugwpv1_solv2.F:785
if((ngw_abs(ze1)>=maxdudt)) {
// cires_ugwpv1_solv2.F:786
ze1 = copysignf(maxdudt,ze1);
// cires_ugwpv1_solv2.F:787
}
// cires_ugwpv1_solv2.F:788
if((ngw_abs(ze2)>=maxdudt)) {
// cires_ugwpv1_solv2.F:789
ze2 = copysignf(maxdudt,ze2);
// cires_ugwpv1_solv2.F:790
}
// cires_ugwpv1_solv2.F:792
pdudt[((jk-1)*nc)+col] = ngw_neg(ze1);
// cires_ugwpv1_solv2.F:793
pdvdt[((jk-1)*nc)+col] = ngw_neg(ze2);
// cires_ugwpv1_solv2.F:798
if((knob_ugwp_doheat==1)) {
// cires_ugwpv1_solv2.F:802
pdtdt[((jk-1)*nc)+col] = ngw_mul(pdtdt[((jk-1)*nc)+col], gw_eff);
// cires_ugwpv1_solv2.F:803
ze2 = pdtdt[((jk-1)*nc)+col];
// cires_ugwpv1_solv2.F:804
if((ngw_abs(ze2)>=max_eps)) { pdtdt[((jk-1)*nc)+col] = copysignf(max_eps,ze2); }
// cires_ugwpv1_solv2.F:806
dked[((jk-1)*nc)+col] = ngw_div(dked[((jk-1)*nc)+col], bn2[jk]);
// cires_ugwpv1_solv2.F:807
ze1 = ngw_max(dked_min,dked[((jk-1)*nc)+col]);
// cires_ugwpv1_solv2.F:808
dked[((jk-1)*nc)+col] = ngw_min(dked_max,ze1);
// cires_ugwpv1_solv2.F:809
qmid[jk] = pdtdt[((jk-1)*nc)+col];
// cires_ugwpv1_solv2.F:810
}
// cires_ugwpv1_solv2.F:811
}
// cires_ugwpv1_solv2.F:819
for(int s0=1;s0<=levs;++s0) { dktur[s0] = dked[((s0-1)*nc)+col]; }
// cires_ugwpv1_solv2.F:821
for(ist=1;ist<=nstdif;ist+=1) {
// cires_ugwpv1_solv2.F:822
for(jk=ksrc;jk<=(levs-1);jk+=1) {
// cires_ugwpv1_solv2.F:823
adif[jk] = ngw_add(ngw_mul(0.25f, ngw_add(dktur[(jk-1)], dktur[(jk+1)])), ngw_mul(0.5f, dktur[jk]));
// cires_ugwpv1_solv2.F:824
}
// cires_ugwpv1_solv2.F:825
for(int s0=ksrc;s0<=(levs-1);++s0) { dktur[s0] = adif[s0]; }
// cires_ugwpv1_solv2.F:826
}
// cires_ugwpv1_solv2.F:827
dktur[levs] = ngw_mul(0.5f, ngw_add(dked[((levs-1)*nc)+col], dked[(((levs-1)-1)*nc)+col]));
// cires_ugwpv1_solv2.F:828
dktur[(levs+1)] = dktur[levs];
// cires_ugwpv1_solv2.F:830
for(jk=ksrc;jk<=(levs+1);jk+=1) {
// cires_ugwpv1_solv2.F:831
ze1 = ngw_mul(0.5f, ngw_add(dktur[jk], dktur[(jk-1)]));
// cires_ugwpv1_solv2.F:832
kvint[jk] = ze1;
// cires_ugwpv1_solv2.F:833
ktint[jk] = ngw_mul(ze1, ipr_ktgw);
// cires_ugwpv1_solv2.F:834
}
// cires_ugwpv1_solv2.F:839
for(jk=(ksrc+1);jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:840
ze2 = ngw_add(ngw_add(qmid[jk], ngw_mul(dktur[jk], akt[jk])), ngw_div(ngw_mul(grav, ngw_sub(ktint[(jk+1)], ktint[jk])), dz_meti[jk]));
// cires_ugwpv1_solv2.F:841
qmid[jk] = ze2;
// cires_ugwpv1_solv2.F:842
if((ngw_abs(ze2)>=max_eps)) { qmid[jk] = copysignf(max_eps,ze2); }
// cires_ugwpv1_solv2.F:843
pdtdt[((jk-1)*nc)+col] = ngw_mul(qmid[jk], rcpd);
// cires_ugwpv1_solv2.F:844
dked[((jk-1)*nc)+col] = dktur[jk];
// cires_ugwpv1_solv2.F:845
}
// cires_ugwpv1_solv2.F:853
for(int s0=km2;s0<=levs;++s0) { uold[s0] = ngw_add(aum[s0], ngw_mul(pdudt[((s0-1)*nc)+col], dtp)); }
// cires_ugwpv1_solv2.F:854
for(int s0=km2;s0<=levs;++s0) { vold[s0] = ngw_add(avm[s0], ngw_mul(pdvdt[((s0-1)*nc)+col], dtp)); }
// cires_ugwpv1_solv2.F:855
for(int s0=km2;s0<=levs;++s0) { told[s0] = ngw_add(atm[s0], ngw_mul(pdtdt[((s0-1)*nc)+col], dtp)); }
// cires_ugwpv1_solv2.F:860
for(int s0=km2;s0<=levs;++s0) { dktur[s0] = dked_min; }
// cires_ugwpv1_solv2.F:862
for(jk=km1;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:863
uz = ngw_sub(uold[jk], uold[(jk-1)]);
// cires_ugwpv1_solv2.F:864
vz = ngw_sub(vold[jk], vold[(jk-1)]);
// cires_ugwpv1_solv2.F:865
ze1 = dz_met[jk];
// cires_ugwpv1_solv2.F:866
zdelm = ngw_div(1.0f, ze1);
// cires_ugwpv1_solv2.F:868
tvc = ngw_mul(told[jk], ngw_add(1.0f, ngw_mul(fv, aqm[jk])));
// cires_ugwpv1_solv2.F:869
tvm = ngw_mul(told[(jk-1)], ngw_add(1.0f, ngw_mul(fv, aqm[(jk-1)])));
// cires_ugwpv1_solv2.F:870
zthm = ngw_div(2.0f, ngw_add(tvc, tvm));
// cires_ugwpv1_solv2.F:871
shr2 = ngw_mul(ngw_mul(ngw_max(ngw_add(ngw_mul(uz, uz), ngw_mul(vz, vz)),dw2min), zdelm), zdelm);
// cires_ugwpv1_solv2.F:873
bn2[jk] = ngw_mul(ngw_mul(grav2cpd, zthm), ngw_add(1.0f, ngw_mul(ngw_mul(rcpdl, ngw_sub(tvc, tvm)), zdelm)));
// cires_ugwpv1_solv2.F:875
bn2[jk] = ngw_max(ngw_min(bn2[jk],bnv2max),bnv2min);
// cires_ugwpv1_solv2.F:876
zmetk = ngw_mul(azmet[jk], rh4);
// cires_ugwpv1_solv2.F:877
zgrow = ngw_exp(zmetk);
// cires_ugwpv1_solv2.F:878
ritur = ngw_div(bn2[jk], shr2);
// cires_ugwpv1_solv2.F:879
w1 = ngw_div(1.0f, ngw_add(1.0f, ngw_mul(5, ritur)));
// cires_ugwpv1_solv2.F:880
ze2 = ngw_min(ngw_mul(sc2, zgrow),ngw_mul(ngw_mul(4.0f, ze1), ze1));
// cires_ugwpv1_solv2.F:884
kamp = ngw_mul(ngw_mul(ngw_mul(ngw_sqrt(shr2), ze2), w1), w1);
// cires_ugwpv1_solv2.F:885
// GNU folds the lower clamp with the finite bound as unordered result.
ktur = ngw_min(ngw_max(dked_min,kamp),dked_max);
// cires_ugwpv1_solv2.F:886
dktur[jk] = ktur;
// cires_ugwpv1_solv2.F:890
dked[((jk-1)*nc)+col] = ngw_add(dked[((jk-1)*nc)+col], ktur);
// cires_ugwpv1_solv2.F:892
}
// cires_ugwpv1_solv2.F:897
if((knob_ugwp_dokdis==2)) {
// cires_ugwpv1_solv2.F:899
for(jk=ksrc;jk<=levs;jk+=1) {
// cires_ugwpv1_solv2.F:900
ze1 = ngw_min(ngw_mul(0.5f, ngw_add(dktur[jk], dktur[(jk-1)])),dturb_max);
// cires_ugwpv1_solv2.F:901
kvint[jk] = ngw_add(kvint[jk], ze1);
// cires_ugwpv1_solv2.F:903
}
// cires_ugwpv1_solv2.F:904
kvint[km1] = kvint[ksrc];
// cires_ugwpv1_solv2.F:905
kvint[ktop] = kvint[levs];
// cires_ugwpv1_solv2.F:907
dzmetm = ngw_div(1.0f, dz_met[km1]);
// cires_ugwpv1_solv2.F:908
for(int s0=km1;s0<=levs;++s0) { adif[s0] = 0.0f; }
// cires_ugwpv1_solv2.F:909
for(int s0=km1;s0<=levs;++s0) { cdif[s0] = 0.0f; }
// cires_ugwpv1_solv2.F:910
for(jk=km1;jk<=(levs-1);jk+=1) {
// cires_ugwpv1_solv2.F:912
dzmetp = ngw_div(1.0f, dz_met[(jk+1)]);
// cires_ugwpv1_solv2.F:913
dzmetf = ngw_div(1.0f, ngw_mul(dz_meti[jk], rhomid[jk]));
// cires_ugwpv1_solv2.F:916
ktur = ngw_mul(ngw_mul(kvint[jk], rhoint[jk]), dzmetf);
// cires_ugwpv1_solv2.F:917
kturp = ngw_mul(ngw_mul(kvint[(jk+1)], rhoint[(jk+1)]), dzmetf);
// cires_ugwpv1_solv2.F:919
adif[jk] = ngw_mul(ktur, dzmetm);
// cires_ugwpv1_solv2.F:920
cdif[jk] = ngw_mul(kturp, dzmetp);
// cires_ugwpv1_solv2.F:921
apc = ngw_add(adif[jk], cdif[jk]);
// cires_ugwpv1_solv2.F:922
acdif[jk] = apc;
// cires_ugwpv1_solv2.F:924
w1 = ngw_mul(apc, ipr_max);
// cires_ugwpv1_solv2.F:925
if((rdtp<w1)) {
// cires_ugwpv1_solv2.F:926
anstab[jk] = ngw_add(floorf(ngw_mul(w1, dtp)), 1);
// cires_ugwpv1_solv2.F:927
} else {
// cires_ugwpv1_solv2.F:928
anstab[jk] = 1;
// cires_ugwpv1_solv2.F:929
}
// cires_ugwpv1_solv2.F:930
dzmetm = dzmetp;
// cires_ugwpv1_solv2.F:931
}
// cires_ugwpv1_solv2.F:933
int max_stab=anstab[ksrc]; for(int s=ksrc+1;s<=levs-1;++s) max_stab=ngw_max(max_stab,anstab[s]); nstab = max_stab;
// cires_ugwpv1_solv2.F:939
dtdif = ngw_div(dtp, float(nstab));
// cires_ugwpv1_solv2.F:940
ze1 = ngw_div(1.0f, dtdif);
// cires_ugwpv1_solv2.F:942
for(ist=1;ist<=nstab;ist+=1) {
// cires_ugwpv1_solv2.F:943
for(k=ksrc;k<=(levs-1);k+=1) {
// cires_ugwpv1_solv2.F:944
bdif = ngw_sub(ze1, acdif[k]);
// cires_ugwpv1_solv2.F:945
bt_dif = ngw_sub(ze1, ngw_mul(acdif[k], ipr_ktgw));
// cires_ugwpv1_solv2.F:946
unew[k] = ngw_add(ngw_add(ngw_mul(uold[k], bdif), ngw_mul(uold[(k-1)], adif[k])), ngw_mul(uold[(k+1)], cdif[k]));
// cires_ugwpv1_solv2.F:947
vnew[k] = ngw_add(ngw_add(ngw_mul(vold[k], bdif), ngw_mul(vold[(k-1)], adif[k])), ngw_mul(vold[(k+1)], cdif[k]));
// cires_ugwpv1_solv2.F:948
tnew[k] = ngw_add(ngw_mul(told[k], bt_dif), ngw_mul(ngw_add(ngw_mul(told[(k-1)], adif[k]), ngw_mul(told[(k+1)], cdif[k])), ipr_ktgw));
// cires_ugwpv1_solv2.F:949
}
// cires_ugwpv1_solv2.F:951
for(int s0=ksrc;s0<=(levs-1);++s0) { uold[s0] = ngw_mul(unew[s0], dtdif); }
// cires_ugwpv1_solv2.F:952
for(int s0=ksrc;s0<=(levs-1);++s0) { vold[s0] = ngw_mul(vnew[s0], dtdif); }
// cires_ugwpv1_solv2.F:953
for(int s0=ksrc;s0<=(levs-1);++s0) { told[s0] = ngw_mul(tnew[s0], dtdif); }
// cires_ugwpv1_solv2.F:957
uold[levs] = uold[(levs-1)];
// cires_ugwpv1_solv2.F:958
vold[levs] = vold[(levs-1)];
// cires_ugwpv1_solv2.F:959
told[levs] = told[(levs-1)];
// cires_ugwpv1_solv2.F:960
}
// cires_ugwpv1_solv2.F:964
for(k=ksrc;k<=(levs-1);k+=1) {
// cires_ugwpv1_solv2.F:968
ze2 = ngw_mul(rdtp, ngw_sub(uold[k], aum[k]));
// cires_ugwpv1_solv2.F:969
ze1 = ngw_mul(rdtp, ngw_sub(vold[k], avm[k]));
// cires_ugwpv1_solv2.F:970
pdtdt[((k-1)*nc)+col] = ngw_mul(rdtp, ngw_sub(told[k], atm[k]));
// cires_ugwpv1_solv2.F:972
if((ngw_abs(pdtdt[((k-1)*nc)+col])>=maxdtdt)) { pdtdt[((k-1)*nc)+col] = copysignf(maxdtdt,pdtdt[((k-1)*nc)+col]); }
// cires_ugwpv1_solv2.F:973
if((ngw_abs(ze1)>=maxdudt)) {
// cires_ugwpv1_solv2.F:974
ze1 = copysignf(maxdudt,ze1);
// cires_ugwpv1_solv2.F:975
}
// cires_ugwpv1_solv2.F:976
if((ngw_abs(ze2)>=maxdudt)) {
// cires_ugwpv1_solv2.F:977
ze2 = copysignf(maxdudt,ze2);
// cires_ugwpv1_solv2.F:978
}
// cires_ugwpv1_solv2.F:980
pdudt[((k-1)*nc)+col] = ze2;
// cires_ugwpv1_solv2.F:981
pdvdt[((k-1)*nc)+col] = ze1;
// cires_ugwpv1_solv2.F:982
uz = ngw_sub(uold[(k+1)], uold[(k-1)]);
// cires_ugwpv1_solv2.F:983
vz = ngw_sub(vold[(k+1)], vold[(k-1)]);
// cires_ugwpv1_solv2.F:984
ze2 = ngw_div(1.0f, ngw_add(dz_met[(k+1)], dz_met[k]));
// cires_ugwpv1_solv2.F:985
mf_diss_heat = ngw_mul(ngw_mul(ngw_mul(ngw_mul(rcpd, kvint[k]), ngw_add(ngw_mul(uz, uz), ngw_mul(vz, vz))), ze2), ze2);
// cires_ugwpv1_solv2.F:986
pdtdt[((k-1)*nc)+col] = ngw_add(pdtdt[((k-1)*nc)+col], mf_diss_heat);
// cires_ugwpv1_solv2.F:988
}
// cires_ugwpv1_solv2.F:991
}
// cires_ugwpv1_solv2.F:993
}
// bl_ugwpv1_ngw.F:211-231, additive driver tendencies; diagnostics remain temperature rates.
for(int k=1;k<=levs;++k) {
  int i=(k-1)*nc+col;
  rthblten[i]=ngw_add(rthblten[i],ngw_div(pdtdt[i],prslk[i]));
}
for(int k=1;k<=levs;++k) {
  int i=(k-1)*nc+col;
  rublten[i]=ngw_add(rublten[i],pdudt[i]);
  rvblten[i]=ngw_add(rvblten[i],pdvdt[i]);
}
status[col]=0;
// cires_ugwpv1_solv2.F:249-253, SAVE ilaunch after the final native column.
if(launch_state) launch_state[col]=ilaunch;
}

#ifndef NGW_HOST
// bl_ugwpv1_ngw.F:30-104: same initialization arithmetic, without evaluating a timestep.
extern "C" __global__ void ngw_initialize(int nc,int ntlat,const float *rdzw,const float *dzu,
const float *lat_r,const float *taulat,float *initial,int *indices,float *weights) {
int col=blockDim.x*blockIdx.x+threadIdx.x;
if(col<nc) ngw_column(col,nc,ntlat,0,1.f,0.f,0.f,rdzw,dzu,lat_r,taulat,
  nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
  nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
  initial,indices,weights,nullptr,nullptr,nullptr,nullptr,nullptr,true);
}
// bl_ugwpv1_ngw.F:167-236, whole-column driver launch.
extern "C" __global__ void ngw_run(int nc,int ntlat,int ntday,float dtp,float amp,float day,
const float *rdzw,const float *dzu,const float *lat_r,const float *taulat,const float *days,const float *tau,
const float *um,const float *vm,const float *tm,const float *qm,const float *prsl,const float *prslk,
const float *prsi,const float *zmet,const float *zmeti,const float *raincv,const float *rainncv,
float *rublten,float *rvblten,float *rthblten,float *pdudt,float *pdvdt,float *pdtdt,float *dked,float *zngw,
float *init_state,int *index_state,float *weight_state,int *status) {
int col=blockDim.x*blockIdx.x+threadIdx.x;
if(col<nc) ngw_column(col,nc,ntlat,ntday,dtp,amp,day,rdzw,dzu,lat_r,taulat,days,tau,um,vm,tm,qm,prsl,prslk,prsi,zmet,zmeti,raincv,rainncv,rublten,rvblten,rthblten,pdudt,pdvdt,pdtdt,dked,zngw,init_state,index_state,weight_state,status);
}
// bl_ugwpv1_ngw.F:120-236, exact captured inner MPAS argument boundary.
extern "C" __global__ void ngw_run_boundary(int nc,int ntlat,int ntday,float dtp,float amp,float day,
const float *rdzw,const float *dzu,const float *lat_r,const float *taulat,const float *days,const float *tau,
const float *um,const float *vm,const float *tm,const float *qm,const float *prsl,const float *prslk,
const float *prsi,const float *zmet,const float *zmeti,const float *raincv,const float *rainncv,
float *rublten,float *rvblten,float *rthblten,float *pdudt,float *pdvdt,float *pdtdt,float *dked,float *zngw,
float *init_state,int *index_state,float *weight_state,int *status,
const float *native_xlat,const int *native_indices,const float *native_weights,int *launch_state) {
int col=blockDim.x*blockIdx.x+threadIdx.x;
if(col<nc) ngw_column(col,nc,ntlat,ntday,dtp,amp,day,rdzw,dzu,lat_r,taulat,days,tau,um,vm,tm,qm,prsl,prslk,prsi,zmet,zmeti,raincv,rainncv,rublten,rvblten,rthblten,pdudt,pdvdt,pdtdt,dked,zngw,init_state,index_state,weight_state,status,native_xlat,native_indices,native_weights,launch_state);
}
// mpas_atmphys_driver_gwdo.F:648-657,676-683, whole-GWDO subtraction and writeback.
extern "C" __global__ void ngw_writeback(int nword,int diags,
const float *ru,const float *rv,const float *rt,const float *du,const float *dv,const float *dt,
const float *before_u,const float *before_v,
float *out_u,float *out_v,float *out_t,float *diff_u,float *diff_v,
float *out_du,float *out_dv,float *out_dt) {
int i=blockDim.x*blockIdx.x+threadIdx.x;
if(i<nword) {
  diff_u[i]=ngw_sub(ru[i],before_u[i]);diff_v[i]=ngw_sub(rv[i],before_v[i]);
  out_u[i]=ru[i];out_v[i]=rv[i];out_t[i]=rt[i];
  if(diags) {out_du[i]=du[i];out_dv[i]=dv[i];out_dt[i]=dt[i];}
}
}
#endif
'''

MASS_INPUTS = ("um", "vm", "tm", "qm", "prsl", "prslk", "zmet")
INTERFACE_INPUTS = ("prsi", "zmeti")
SURFACE_INPUTS = ("lat_r", "raincv", "rainncv")
DIAGNOSTIC_FIELDS = ("dudt_ngw", "dvdt_ngw", "dtdt_ngw")
TENDENCY_FIELDS = ("rublten", "rvblten", "rthblten")
# mpas_atmphys_driver_gwdo.F:781-799, native prepared argument aliases.
PREPARED_ALIASES = {
    "um": "u_p", "vm": "v_p", "tm": "t_p", "qm": "qv_p",
    "prsl": "pres_hydd_p", "prslk": "pi_p", "zmet": "zmid_p",
    "prsi": "pres2_hydd_p", "zmeti": "z_p",
}
STREAM_CONTRACT = {
    "input": "ugwp_ngw_in", "input_fields": ("LATS", "DAYS", "ABSMF"),
    "output": "diag_ugwp", "output_fields": DIAGNOSTIC_FIELDS,
    "diagnostic_selector": "config_ugwp_diags",
    "selector": "config_ngw_scheme", "parent_scheme": "bl_ugwp_gwdo",
}


@dataclass(frozen=True)
class NgwResult:
    """bl_ugwpv1_ngw.F:128-165,211-231: temperatures and additive rates."""

    rublten: object
    rvblten: object
    rthblten: object
    dudt_ngw: object
    dvdt_ngw: object
    dtdt_ngw: object
    dked: object
    zngw: object
    initial_state: object
    latitude_indices: object
    latitude_weights: object
    status: object

    def diagnostics(self, enabled: bool) -> dict[str, object]:
        """mpas_atmphys_driver_gwdo.F:648-657, only when both options are on."""
        return {name: getattr(self, name) for name in DIAGNOSTIC_FIELDS} if enabled else {}


def shared_math_header() -> str:
    """Maths for cires_ugwpv1_solv2.F:17-1018 and cires_ugwpv1_initialize.F:65-675.

    Shared pinned engine float and glibc 2.43 trigonometric device headers.

    These are reused unchanged; the port introduces no elementary function
    implementation. Header bytes belong to the existing engine pin.
    """
    import woof.core.kernels as kernels

    root = Path(kernels.__file__).parent
    source = "\n".join((root / name).read_text(encoding="utf-8") for name in
                       ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"))
    # cires_ugwpv1_solv2.F:565,661,678: shared exp/log arithmetic may
    # canonicalize a NaN on CUDA; retain the strict native operand words.
    # Every elementary function still delegates to the unchanged shared header.
    return source + "\n__device__ float ngw_native_nan(float,float,float);\n" \
        "#define ngw_exp(x) ngw_native_nan(gfk_exp(x),(x),(x))\n" \
        "#define ngw_log(x) ngw_native_nan(gfk_log(x),(x),(x))\n" \
        "#define ngw_pow(x,y) ngw_native_nan(gfk_pow((x),(y)),(x),(y))\n" \
        "#define ngw_sin(x) ngw_native_nan(glibc_sinf(x),(x),(x))\n" \
        "#define ngw_cos(x) ngw_native_nan(glibc_cosf(x),(x),(x))\n"


def cuda_source(nlev: int, math_header: str | None = None) -> str:
    """One source for packaged and runtime NGW run, init and writeback entries."""
    header = shared_math_header() if math_header is None else math_header
    return f"#define NL {nlev}\n#define NW 25\n#define NA 4\n" + header + CUDA_SOURCE


# Per-card module cache, unbounded on purpose: entries are cards x variants. A bounded LRU thrashes once
# ranks x variants exceed it, and the evicted card reloads its module on every call (thompson_v842._module
# at maxsize=8 on 8 cards, measured 2026-10-28: 1.28 s/step against 0.25 s unbounded).
@lru_cache(maxsize=None)
def _module(nlev: int, math_header: str, device_id: int):
    """cires_ugwpv1_solv2.F:17-1020; separate kernel specialization per depth."""
    import cupy as cp

    if not math_header.strip():
        raise RuntimeError("UGWP needs the shared glibc device math header; "
                           "the lane base rkind_libm.py has host hooks only")
    source = cuda_source(nlev, math_header)
    with cp.cuda.Device(device_id):
        # Direct NVRTC preserves --ftz=false; CuPy RawModule appends --ftz=true.
        # The native physics door need not install the generic CuPy AOT bridge.
        # Use the authenticated direct-image lookup here, as Thompson does,
        # so an NGW pack is consumed on the actual forecast route too.
        from ..cuda_backend.aot import direct_image
        ptx = direct_image(source, _hex_numerics.options(COMPILE_OPTIONS),
                           cp.cuda.compiler._get_arch(), filename="ugwp_ngw_v842.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
        return module


def initialize(latitude, climate_latitude, rdzw, dzu):
    """ugwpv1_ngw_init, bl_ugwpv1_ngw.F:30-104, with no native module state.

    The forecast operator uses the same initialization body for its local spectra
    and profiles. Persist only the latitude fields the MPAS pools expose.
    """
    import cupy as cp
    latitude, climate_latitude, rdzw, dzu = (
        cp.ascontiguousarray(value, dtype=cp.float32)
        for value in (latitude, climate_latitude, rdzw, dzu))
    nlev, ncol = rdzw.size, latitude.size
    if nlev < 5 or dzu.size != nlev or climate_latitude.size < 2:
        raise ValueError("NGW initialization needs five levels and two climate latitudes")
    initial = cp.empty(4 * 58 + 2 * (nlev + 1), cp.float32)
    indices = cp.empty((2, ncol), cp.int32)
    weights = cp.empty((2, ncol), cp.float32)
    kernel = _module(nlev, shared_math_header(), latitude.device.id).get_function("ngw_initialize")
    kernel(((ncol + 31) // 32,), (32,), (np.int32(ncol), np.int32(climate_latitude.size),
           rdzw, dzu, latitude, climate_latitude, initial, indices, weights))
    return initial, indices, weights


def run(columns: Mapping[str, object], climate: Mapping[str, object], *,
        dtp: float, amplitude: float, day: float, math_header: str | None = None,
        native_boundary: Mapping[str, object] | None = None) -> NgwResult:
    """ugwpv1_ngw_init/run, bl_ugwpv1_ngw.F:30-236.

    Resident inputs only. The shared header defines device ngw_exp, ngw_log,
    ngw_pow, ngw_sin and ngw_cos from the shared math implementation. Existing
    PBL/orographic tendencies are copied before adding this slot's contribution.
    Decoding climate NetCDF and transporting model state remain the Rust seam's
    responsibility. ``day`` is the driver's fractional Julian day, not a date.
    """
    import cupy as cp

    base = columns["um"]
    if not isinstance(base, cp.ndarray) or base.ndim != 2:
        raise TypeError("um must be resident C float32 [level,column]")
    nlev, ncol = base.shape
    if nlev < 5 or ncol < 1:
        raise ValueError("UGWP requires at least five levels for launch and km2 scratch")
    device_id = base.device.id

    def resident(name, array, shape, dtype=cp.float32):
        """Resident argument storage; bl_ugwpv1_ngw.F:167-178."""
        if not isinstance(array, cp.ndarray) or array.dtype != dtype or array.shape != shape \
                or not array.flags.c_contiguous or array.device.id != device_id:
            raise TypeError(f"{name} requires resident C {dtype} {shape} on device {device_id}")
        return array

    for name in MASS_INPUTS + TENDENCY_FIELDS:
        resident(name, columns[name], (nlev, ncol))
    for name in INTERFACE_INPUTS:
        resident(name, columns[name], (nlev + 1, ncol))
    for name in SURFACE_INPUTS:
        resident(name, columns[name], (ncol,))
    for name in ("rdzw", "dzu"):
        resident(name, climate[name], (nlev,))
    ntlat = int(climate["LATS"].size)
    ntday = int(climate["DAYS"].size)
    if ntlat < 2 or ntday < 2:
        raise ValueError("cires_tauamf_data.F:90-98 needs latitude and time brackets")
    resident("LATS", climate["LATS"], (ntlat,))
    resident("DAYS", climate["DAYS"], (ntday,))
    resident("ABSMF", climate["ABSMF"], (ntday, ntlat))
    if not np.isfinite(dtp) or dtp <= 0 or not np.isfinite(amplitude) or not np.isfinite(day):
        raise ValueError("finite amplitude/day and positive dtp are required by solver reciprocal dtp")

    with cp.cuda.Device(device_id):
        tendency = [columns[name].copy() for name in TENDENCY_FIELDS]
        diagnostics = [cp.empty((nlev, ncol), cp.float32) for _ in range(4)]
        zngw = cp.empty(ncol, cp.float32)
        initial_state = cp.empty(4 * 58 + 2 * (nlev + 1), cp.float32)
        indices = cp.empty((2, ncol), cp.int32)
        weights = cp.empty((2, ncol), cp.float32)
        status = cp.empty(ncol, cp.int32)
        args = (np.int32(ncol), np.int32(ntlat), np.int32(ntday), np.float32(dtp),
                np.float32(amplitude), np.float32(day), climate["rdzw"], climate["dzu"],
                columns["lat_r"], climate["LATS"], climate["DAYS"], climate["ABSMF"],
                columns["um"], columns["vm"], columns["tm"], columns["qm"], columns["prsl"],
                columns["prslk"], columns["prsi"], columns["zmet"], columns["zmeti"],
                columns["raincv"], columns["rainncv"], *tendency, *diagnostics,
                zngw, initial_state, indices, weights, status)
        entry = "ngw_run"
        if native_boundary is not None:
            # bl_ugwpv1_ngw.F:132-157; cires_ugwpv1_solv2.F:249-253.
            # Captured resident arguments and observable SAVE output, without a second latitude conversion.
            for name, shape, dtype in (("xlatd", (ncol,), cp.float32),
                                       ("indices", (2, ncol), cp.int32),
                                       ("weights", (2, ncol), cp.float32),
                                       ("launch_state", (ncol,), cp.int32)):
                resident(name, native_boundary[name], shape, dtype)
            args += tuple(native_boundary[name] for name in ("xlatd", "indices", "weights", "launch_state"))
            entry = "ngw_run_boundary"
        kernel = _module(nlev, shared_math_header() if math_header is None else math_header, device_id).get_function(entry)
        kernel(((ncol + 31) // 32,), (32,), args)
    return NgwResult(*tendency, *diagnostics, zngw, initial_state, indices, weights, status)


def from_mpas(mesh: Mapping[str, object], diag: Mapping[str, object],
              tendencies: Mapping[str, object], prepared: Mapping[str, object], *,
              convection_on: bool, microphysics_on: bool) -> dict[str, object]:
    """mpas_atmphys_driver_gwdo.F:474-514,540-547; module_bl_ugwp_gwdo.F:220-294.

    The Rust/driver preparation owns thermodynamic transforms. Layout already
    matches MPAS [level,column]. Latitude conversion and climate indices occur
    in the kernel, preserving the driver and initialization's different pi.
    """
    import cupy as cp

    result = {name: prepared[name] for name in MASS_INPUTS + INTERFACE_INPUTS}
    result.update({name: tendencies[name] for name in TENDENCY_FIELDS})
    result["lat_r"] = mesh["latCell"]
    result["raincv"] = diag["raincv"] if convection_on else cp.zeros_like(mesh["latCell"])
    result["rainncv"] = diag["rainncv"] if microphysics_on else cp.zeros_like(mesh["latCell"])
    return result


def to_mpas(result: NgwResult, diag: Mapping[str, object],
            tendencies: Mapping[str, object], *, gwdo_before: Mapping[str, object],
            ngw_input: Mapping[str, object], ugwp_diags: bool) -> None:
    """mpas_atmphys_driver_gwdo.F:595-597,648-657,676-683; module_bl_ugwp_gwdo.F:384-423."""
    import cupy as cp

    # The shared module keeps subtraction gradual-underflow safe (--ftz=false).
    nlev, ncol = result.rublten.shape
    device_id = result.rublten.device.id
    def field(array, shape, dtype):
        """Output pool descriptors; mpas_atmphys_driver_gwdo.F:595-600."""
        if not isinstance(array, cp.ndarray) or array.shape != shape or array.dtype != dtype \
                or not array.flags.c_contiguous or array.device.id != device_id:
            raise TypeError("NGW writeback requires matching resident pool descriptors")
    for name in TENDENCY_FIELDS:
        field(tendencies[name], (nlev,ncol), cp.float32)
    for name in ("rubldiff", "rvbldiff"):
        field(diag[name], (nlev,ncol), cp.float32)
    for name in ("rublten", "rvblten"):
        field(gwdo_before[name], (nlev,ncol), cp.float32)
    if ugwp_diags:
        for name in DIAGNOSTIC_FIELDS:
            field(diag[name], (nlev,ncol), cp.float32)
    for name in ("jindx1_tau", "jindx2_tau", "ddy_j1tau", "ddy_j2tau"):
        field(ngw_input[name], (ncol,), cp.int32 if name.startswith("jindx") else cp.float32)
    with cp.cuda.Device(device_id):
        kernel = _module(nlev, shared_math_header(), result.rublten.device.id).get_function("ngw_writeback")
        # Disabled diagnostics have valid pointers but are never accessed.
        targets = [diag[name] for name in DIAGNOSTIC_FIELDS] if ugwp_diags else [result.dtdt_ngw]*3
        args = (np.int32(nlev*ncol), np.int32(ugwp_diags), result.rublten, result.rvblten,
                result.rthblten, result.dudt_ngw, result.dvdt_ngw, result.dtdt_ngw,
                gwdo_before["rublten"], gwdo_before["rvblten"],
                tendencies["rublten"], tendencies["rvblten"], tendencies["rthblten"],
                diag["rubldiff"], diag["rvbldiff"], *targets)
        kernel(((nlev*ncol+127)//128,), (128,), args)
    # mpas_atmphys_init.F:464-472, persistent input-pool latitude interpolation.
    for name, array in (("jindx1_tau", result.latitude_indices[0]),
                        ("jindx2_tau", result.latitude_indices[1]),
                        ("ddy_j1tau", result.latitude_weights[0]),
                        ("ddy_j2tau", result.latitude_weights[1])):
        cp.copyto(ngw_input[name], array)


def bind_native_pools(provider, call):
    """Native driver storage binding; mpas_atmphys_driver_gwdo.F:474-514,595-597,781-813.

    MPAS initialization supplies climate/vertical words (mpas_atmphys_init.F:
    244-245,463-472). The native calendar supplies fractional Julian day
    (mpas_atmphys_manager.F:204-213). This copies and associates storage only;
    native preparation and the CUDA operator retain every physics calculation.
    """
    if call.slot != "ngw" or str(call.backend) != "hex-cuda":
        raise ValueError("NGW pool binding requires a registered ngw CUDA slot")
    cp = provider.cp
    mesh, diag, tendencies, ngw_input = (provider[name] for name in
        ("mesh", "diag_physics", "tend_physics", "ngw_input"))
    nlev = int(mesh["nVertLevels"])
    ncol = int(provider.session.layout.dimensions["nCellsSolve"])
    prepared = {name: cp.ascontiguousarray(provider.prepared(alias)[
        :nlev + int(name in INTERFACE_INPUTS)])
        for name, alias in PREPARED_ALIASES.items()}
    climate = {name: mesh[name][:nlev] for name in ("rdzw", "dzu")}
    climate.update({name: ngw_input[name] for name in ("LATS", "DAYS")})
    # Registry.xml:3755; cires_tauamf_data.F:68,106-109: native pool
    # dimensions are [latitude,day]; the kernel consumes C [day,latitude].
    climate["ABSMF"] = cp.ascontiguousarray(ngw_input["ABSMF"].T)
    before = getattr(provider.session, "gwdo_before", None)
    if before is None:
        raise RuntimeError("pre-both GWDO wind snapshots are required for native rubldiff/rvbldiff")
    gwdo_before = {name: cp.ascontiguousarray(cp.asarray(before[name][:nlev, :ncol]))
                   for name in ("rublten", "rvblten")}
    return {"mesh": mesh, "diag_physics": diag, "tend_physics": tendencies,
            "prepared": prepared, "climate": climate, "ngw_input": ngw_input,
            "gwdo_before": gwdo_before,
            "curr_julday": provider.session.arguments.calendar["julian"]}


def _launch_mpas(call, values):
    """mpas_atmphys_driver_gwdo.F:474-514,648-683,807-813, separate NGW driver slice."""
    options = call.options
    columns = from_mpas(values["mesh"], values["diag_physics"],
                        values["tend_physics"], values["prepared"],
                        convection_on=options["config_convection_scheme"] != "off",
                        microphysics_on=options["config_microp_scheme"] != "off")
    result = run(columns, values["climate"], dtp=call.dt,
                 amplitude=options["config_knob_ugwp_tauamp"], day=values["curr_julday"])
    to_mpas(result, values["diag_physics"], values["tend_physics"],
            gwdo_before=values["gwdo_before"], ngw_input=values["ngw_input"],
            ugwp_diags=options["config_ugwp_diags"])
    return result


def launch_slot(call):
    """Table callback; mpas_atmphys_driver_gwdo.F:781-813."""
    return _launch_mpas(call, call.pools)


def register_backend(registry, *, arguments) -> None:
    """Driver registration; mpas_atmphys_driver_gwdo.F:474-514,648-683,807-813.

    ``arguments(call)`` supplies resident ``mesh``, ``diag_physics``,
    ``tend_physics``, ``prepared``, ``climate``, ``ngw_input``,
    ``gwdo_before`` and scalar
    ``curr_julday``. Prepared fields come from the shared MPAS thermodynamic
    preparation, not file decoding in Python. ``gwdo_before`` preserves the
    two wind tendencies before orographic drag; it must not alias writeback.
    The integrator adds the ngw table row and separate driver dispatch.
    """
    def invoke(call):
        """Slot invocation; mpas_atmphys_driver_gwdo.F:781-813."""
        return _launch_mpas(call, arguments(call))

    registry.register("ngw", "ugwp_ngw", "hex-cuda", invoke)


def finalize() -> None:
    """cires_ugwpv1_module.F:305-321, release cached device module ownership.

    Native SAVE work arrays are column-local device scratch here. Result arrays
    belong to the caller; their lifetime follows the enclosing model pools.
    """
    _module.cache_clear()
