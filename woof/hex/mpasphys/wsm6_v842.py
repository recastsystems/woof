"""MPAS-A v8.4.2 WSM6, effective radii and radar, isolated from WOOF.

Authority: pinned physics_mmm/mp_wsm6.F90, mp_wsm6_effectRad.F90,
mp_radar.F90, and physics_wrf/module_mp_wsm6.F (WRF_CHEM=0).
Every translated routine and executable statement retains its native citation.
Device transcendental and complex division operations require the shared math
provider; no CUDA builtin fallback is selected. Host-oracle agreement alone
is not GPU or forecast acceptance. Arrays are RKIND binary32; radar complex
intermediates retain native real(8), including default-complex rounding.
"""
from __future__ import annotations
import re
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from typing import Mapping, Callable, Any

MAX_LEVELS = 256
# Slots 8:11 are reserved, matching the standalone C oracle ABI. They are not read.
CONSTANT_NAMES = (
    "g", "cpd", "cpv", "rd", "rv", "t0c", "ep1", "ep2",
    "reserved0", "reserved1", "reserved2", "qmin", "xls", "xlv0", "xlf0",
    "den0", "denr", "cliq", "cice", "psat", "re_qc_bg", "re_qi_bg", "re_qs_bg",
    "re_qc_max", "re_qi_max", "re_qs_max",
)
VOLUME_NAMES = ("th_or_t", "qv", "qc", "qi", "qr", "qs", "qg", "den", "p", "delz")
SURFACE_NAMES = ("rain", "rainncv", "snow", "snowncv", "graupel", "graupelncv", "sr")

CUDA_ARITHMETIC = r'''
namespace wsm6_detail {
template<class A,class B> struct is_same { static constexpr bool value=false; };
template<class A> struct is_same<A,A> { static constexpr bool value=true; };
template<class A> struct is_arithmetic { static constexpr bool value=false; };
template<> struct is_arithmetic<int> { static constexpr bool value=true; };
template<> struct is_arithmetic<float> { static constexpr bool value=true; };
template<> struct is_arithmetic<double> { static constexpr bool value=true; };
template<bool,class T=int> struct enable_if {};
template<class T> struct enable_if<true,T> { using type=T; };
}
#include <cupy/complex.cuh>
using cx=thrust::complex<double>;
using cx4=thrust::complex<float>;
// The shared libm provider is inserted here. It defines mp_exp/log/pow/sqrt/
// log10/sin/cos/atan for float/double, complex log/sqrt/abs and complex division.
/* WSM6_SHARED_DEVICE_MATH */
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ double mp_pow(double a,float b) {return mp_pow(a,(double)b);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ double mp_pow(float a,double b) {return mp_pow((double)a,b);}
#define WSM6_KMAX 256
// mp_wsm6.F90:213-1468; GNU scalar SSE exceptional arithmetic lowering.
// Reuses p3 Thompson's scalar operator support, not its scheme arithmetic or libm.
__device__ bool wsm_nan(float x) {return (__float_as_uint(x)&0x7fffffffU)>0x7f800000U;}
// mp_wsm6.F90:213-1468; GNU scalar SSE exceptional arithmetic lowering.
__device__ bool wsm_nan(double x) {return ((unsigned long long)__double_as_longlong(x)&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;}
// mp_wsm6.F90:213-1468; quiet an input NaN, preserving its sign and payload.
__device__ float wsm_quiet(float x) {return __uint_as_float(__float_as_uint(x)|0x00400000U);}
// mp_wsm6.F90:213-1468; quiet an input NaN, preserving its sign and payload.
__device__ double wsm_quiet(double x) {return __longlong_as_double(__double_as_longlong(x)|0x0008000000000000LL);}
// mp_wsm6.F90:213-1468; invalid arithmetic produces GNU SSE's negative indefinite.
__device__ float wsm_invalid(float x) {return wsm_nan(x)?__uint_as_float(0xffc00000U):x;}
// mp_wsm6.F90:213-1468; invalid arithmetic produces GNU SSE's negative indefinite.
__device__ double wsm_invalid(double x) {return wsm_nan(x)?__longlong_as_double(0xfff8000000000000ULL):x;}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto add(A a,B b) {
 using P=decltype(a+b);
 if constexpr(wsm6_detail::is_same<P,double>::value || wsm6_detail::is_same<P,float>::value) {
   P x=P(a),y=P(b);if(wsm_nan(x))return wsm_quiet(x);if(wsm_nan(y))return wsm_quiet(y);
   if constexpr(wsm6_detail::is_same<P,double>::value)return wsm_invalid(__dadd_rn(x,y));
   else return wsm_invalid(__fadd_rn(x,y));
 }
 else return a+b;
}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto sub(A a,B b) {
 using P=decltype(a-b);
 if constexpr(wsm6_detail::is_same<P,double>::value || wsm6_detail::is_same<P,float>::value) {
   P x=P(a),y=P(b);if(wsm_nan(x))return wsm_quiet(x);if(wsm_nan(y))return wsm_quiet(y);
   if constexpr(wsm6_detail::is_same<P,double>::value)return wsm_invalid(__dsub_rn(x,y));
   else return wsm_invalid(__fsub_rn(x,y));
 }
 else return a-b;
}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto mul(A a,B b) {
 using P=decltype(a*b);
 if constexpr(wsm6_detail::is_same<P,double>::value || wsm6_detail::is_same<P,float>::value) {
   P x=P(a),y=P(b);if(wsm_nan(x))return wsm_quiet(x);if(wsm_nan(y))return wsm_quiet(y);
   if constexpr(wsm6_detail::is_same<P,double>::value)return wsm_invalid(__dmul_rn(x,y));
   else return wsm_invalid(__fmul_rn(x,y));
 }
 else return a*b;
}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto div(A a,B b) {
 using P=decltype(a/b);
 if constexpr(wsm6_detail::is_same<P,double>::value || wsm6_detail::is_same<P,float>::value) {
   P x=P(a),y=P(b);if(wsm_nan(x))return wsm_quiet(x);if(wsm_nan(y))return wsm_quiet(y);
   if constexpr(wsm6_detail::is_same<P,double>::value)return wsm_invalid(__ddiv_rn(x,y));
   else return wsm_invalid(__fdiv_rn(x,y));
 }
 else return a/b;
}
// Promotion follows Fortran: default-real subtrees round before conversion to
// real(8), while expressions containing complex(8) promote their other operand.
#define CXOPS(OP,NAME) \
/* mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support */ \
template<class T,class B, typename wsm6_detail::enable_if<wsm6_detail::is_arithmetic<B>::value,int>::type=0> __device__ auto NAME(thrust::complex<T> a,B b) { using P=decltype(T{}+B{}); return NAME(thrust::complex<P>(a),thrust::complex<P>(P(b),P(0))); } \
/* mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support */ \
template<class A,class T, typename wsm6_detail::enable_if<wsm6_detail::is_arithmetic<A>::value,int>::type=0> __device__ auto NAME(A a,thrust::complex<T> b) { using P=decltype(A{}+T{}); return NAME(thrust::complex<P>(P(a),P(0)),thrust::complex<P>(b)); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto add(thrust::complex<A> a,thrust::complex<B> b) { using P=decltype(A{}+B{}); return thrust::complex<P>(add(P(a.real()),P(b.real())),add(P(a.imag()),P(b.imag()))); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto sub(thrust::complex<A> a,thrust::complex<B> b) { using P=decltype(A{}+B{}); return thrust::complex<P>(sub(P(a.real()),P(b.real())),sub(P(a.imag()),P(b.imag()))); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto mul(thrust::complex<A> a,thrust::complex<B> b) { using P=decltype(A{}+B{}); return thrust::complex<P>(sub(mul(P(a.real()),P(b.real())),mul(P(a.imag()),P(b.imag()))),add(mul(P(a.real()),P(b.imag())),mul(P(a.imag()),P(b.real())))); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto div(thrust::complex<A> a,thrust::complex<B> b) { using P=decltype(A{}+B{}); return mp_complex_div(thrust::complex<P>(a),thrust::complex<P>(b)); }
CXOPS(+,add)
CXOPS(-,sub)
CXOPS(*,mul)
CXOPS(/,div)
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto mp_min(A a,B b) { using P=decltype(a+b); return P(a)<P(b)?P(a):P(b); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<class A,class B> __device__ auto mp_max(A a,B b) { using P=decltype(a+b); return P(a)>P(b)?P(a):P(b); }
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ float mp_abs(float a) {return fabsf(a);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ double mp_abs(double a) {return fabs(a);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ int mp_abs(int a) {return abs(a);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ cx mp_conjg(cx a) {return cx(a.real(),-a.imag());}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ cx4 mp_cmplx(double a,double b) {return cx4((float)a,(float)b);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
__device__ int mp_nint(float a) {return (int)roundf(a);}
// mp_wsm6.F90:213-1468; Fortran scalar operator/intrinsic lowering support
template<int N,class T> __device__ T ipow(T a) {
 if constexpr(N==0) return T(1);
 else if constexpr(N==1) return a;
 else if constexpr(N==2) return mul(a,a);
 else if constexpr(N==3) return mul(mul(a,a),a);
 else if constexpr(N==4) { T a2=mul(a,a);return mul(a2,a2); }
 else if constexpr(N==5) { T a2=mul(a,a);return mul(mul(a2,a2),a); }
 else if constexpr(N==6) { T a3=mul(mul(a,a),a);return mul(a3,a3); }
 else if constexpr(N==7) { T a2=mul(a,a),a3=mul(a2,a);return mul(mul(a3,a3),a); }
 else if constexpr(N==8) { T a2=mul(a,a),a4=mul(a2,a2);return mul(a4,a4); }
}
// mp_radar.F90:375-477; implements native character equality
__device__ bool str_eq(const char *a,const char *b) { while(*a && *a==*b) {++a;++b;} return *a==*b; }
'''

CUDA_MODULES = r'''
struct Wsm6Workspace;
struct Wsm6State {
Wsm6Workspace *workspace;
const char *last_message;
float dtcldcr;
float n0r;
float avtr;
float bvtr;
float r0;
float peaut;
float xncr;
float xmyu;
float avts;
float bvts;
float lamdarmax;
float lamdasmax;
float dicon;
float dimax;
float pfrz1;
float pfrz2;
float qcrmin;
float eacrc;
float dens;
float qs0;
float n0smax;
float n0s;
float alpha;
float qc0;
float qck1;
float bvtr1;
float bvtr2;
float bvtr3;
float bvtr4;
float g1pbr;
float g3pbr;
float g4pbr;
float g5pbro2;
float pvtr;
float eacrr;
float pacrr;
float bvtr6;
float g6pbr;
float precr1;
float precr2;
float roqimax;
float bvts1;
float bvts2;
float bvts3;
float bvts4;
float g1pbs;
float g3pbs;
float g4pbs;
float n0g;
float avtg;
float bvtg;
float deng;
float lamdagmax;
float g5pbso2;
float pvts;
float pacrs;
float precs1;
float precs2;
float pidn0r;
float xlv1;
float pacrc;
float pi;
float bvtg1;
float bvtg2;
float bvtg3;
float bvtg4;
float g1pbg;
float g3pbg;
float g4pbg;
float g5pbgo2;
float pvtg;
float pacrg;
float precg1;
float precg2;
float pidn0g;
float rslopermax;
float rslopesmax;
float rslopegmax;
float rsloperbmax;
float rslopesbmax;
float rslopegbmax;
float rsloper2max;
float rslopes2max;
float rslopeg2max;
float rsloper3max;
float rslopes3max;
float rslopeg3max;
float pidn0s;
float pidnc;
int nrbins;
int slen;
const char *mixingrulestring_s;
const char *matrixstring_s;
const char *inclusionstring_s;
const char *hoststring_s;
const char *hostmatrixstring_s;
const char *hostinclusionstring_s;
const char *mixingrulestring_g;
const char *matrixstring_g;
const char *inclusionstring_g;
const char *hoststring_g;
const char *hostmatrixstring_g;
const char *hostinclusionstring_g;
cx m_w_0;
cx m_i_0;
double xxdx[(50+1)];
double xxds[(50)];
double xdts[(50)];
double xxdg[(50)];
double xdtg[(50)];
double lamda_radar;
double k_w;
double pi5;
double lamda4;
double simpson[(50+1)];
double basis[(3)];
float xcre[(4)];
float xcse[(4)];
float xcge[(4)];
float xcrg[(4)];
float xcsg[(4)];
float xcgg[(4)];
float xam_r;
float xbm_r;
float xmu_r;
float xobmr;
float xam_s;
float xbm_s;
float xmu_s;
float xoams;
float xobms;
float xocms;
float xam_g;
float xbm_g;
float xmu_g;
float xoamg;
float xobmg;
float xocmg;
float xorg2;
float xosg2;
float xogg2;
const char *radar_debug;
double melt_outside_s;
double melt_outside_g;
};
// MPAS-A v8.4.2 mp_wsm6.F90:213-2444; mp_wsm6_effectRad.F90:59-193.
// One global workspace per independent column. Distinct native automatic arrays
// retain distinct storage across nested calls. Padding stays unassigned.
struct Wsm6Workspace {
  Wsm6State state;
  float mp_wsm6_run_rh[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_qsat[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_rslope[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_rslope2[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_rslope3[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_rslopeb[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_qrs_tmp[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_falk[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_fall[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_work1[WSM6_KMAX * 3 + 3];
  float mp_wsm6_run_fallc[WSM6_KMAX + 3];
  float mp_wsm6_run_falkc[WSM6_KMAX + 3];
  float mp_wsm6_run_work1c[WSM6_KMAX + 3];
  float mp_wsm6_run_work2c[WSM6_KMAX + 3];
  float mp_wsm6_run_workr[WSM6_KMAX + 3];
  float mp_wsm6_run_worka[WSM6_KMAX + 3];
  float mp_wsm6_run_den_tmp[WSM6_KMAX + 3];
  float mp_wsm6_run_delz_tmp[WSM6_KMAX + 3];
  float mp_wsm6_run_pigen[WSM6_KMAX + 3];
  float mp_wsm6_run_pidep[WSM6_KMAX + 3];
  float mp_wsm6_run_pcond[WSM6_KMAX + 3];
  float mp_wsm6_run_prevp[WSM6_KMAX + 3];
  float mp_wsm6_run_psevp[WSM6_KMAX + 3];
  float mp_wsm6_run_pgevp[WSM6_KMAX + 3];
  float mp_wsm6_run_psdep[WSM6_KMAX + 3];
  float mp_wsm6_run_pgdep[WSM6_KMAX + 3];
  float mp_wsm6_run_praut[WSM6_KMAX + 3];
  float mp_wsm6_run_psaut[WSM6_KMAX + 3];
  float mp_wsm6_run_pgaut[WSM6_KMAX + 3];
  float mp_wsm6_run_piacr[WSM6_KMAX + 3];
  float mp_wsm6_run_pracw[WSM6_KMAX + 3];
  float mp_wsm6_run_praci[WSM6_KMAX + 3];
  float mp_wsm6_run_pracs[WSM6_KMAX + 3];
  float mp_wsm6_run_psacw[WSM6_KMAX + 3];
  float mp_wsm6_run_psaci[WSM6_KMAX + 3];
  float mp_wsm6_run_psacr[WSM6_KMAX + 3];
  float mp_wsm6_run_pgacw[WSM6_KMAX + 3];
  float mp_wsm6_run_pgaci[WSM6_KMAX + 3];
  float mp_wsm6_run_pgacr[WSM6_KMAX + 3];
  float mp_wsm6_run_pgacs[WSM6_KMAX + 3];
  float mp_wsm6_run_paacw[WSM6_KMAX + 3];
  float mp_wsm6_run_psmlt[WSM6_KMAX + 3];
  float mp_wsm6_run_pgmlt[WSM6_KMAX + 3];
  float mp_wsm6_run_pseml[WSM6_KMAX + 3];
  float mp_wsm6_run_pgeml[WSM6_KMAX + 3];
  float mp_wsm6_run_qsum[WSM6_KMAX + 3];
  float mp_wsm6_run_xl[WSM6_KMAX + 3];
  float mp_wsm6_run_cpm[WSM6_KMAX + 3];
  float mp_wsm6_run_work2[WSM6_KMAX + 3];
  float mp_wsm6_run_denfac[WSM6_KMAX + 3];
  float mp_wsm6_run_xni[WSM6_KMAX + 3];
  float mp_wsm6_run_denqrs1[WSM6_KMAX + 3];
  float mp_wsm6_run_denqrs2[WSM6_KMAX + 3];
  float mp_wsm6_run_denqrs3[WSM6_KMAX + 3];
  float mp_wsm6_run_denqci[WSM6_KMAX + 3];
  float mp_wsm6_run_n0sfac[WSM6_KMAX + 3];
  float slope_wsm6_n0sfac[WSM6_KMAX + 3];
  float slope_rain_n0sfac[WSM6_KMAX + 3];
  float slope_snow_n0sfac[WSM6_KMAX + 3];
  float slope_graup_n0sfac[WSM6_KMAX + 3];
  float nislfv_rain_plm_dz[WSM6_KMAX + 3];
  float nislfv_rain_plm_ww[WSM6_KMAX + 3];
  float nislfv_rain_plm_qq[WSM6_KMAX + 3];
  float nislfv_rain_plm_wd[WSM6_KMAX + 3];
  float nislfv_rain_plm_wa[WSM6_KMAX + 3];
  float nislfv_rain_plm_was[WSM6_KMAX + 3];
  float nislfv_rain_plm_den[WSM6_KMAX + 3];
  float nislfv_rain_plm_denfac[WSM6_KMAX + 3];
  float nislfv_rain_plm_tk[WSM6_KMAX + 3];
  float nislfv_rain_plm_qn[WSM6_KMAX + 3];
  float nislfv_rain_plm_qr[WSM6_KMAX + 3];
  float nislfv_rain_plm_tmp[WSM6_KMAX + 3];
  float nislfv_rain_plm_tmp1[WSM6_KMAX + 3];
  float nislfv_rain_plm_tmp2[WSM6_KMAX + 3];
  float nislfv_rain_plm_tmp3[WSM6_KMAX + 3];
  float nislfv_rain_plm_wi[WSM6_KMAX + 3];
  float nislfv_rain_plm_zi[WSM6_KMAX + 3];
  float nislfv_rain_plm_za[WSM6_KMAX + 3];
  float nislfv_rain_plm_dza[WSM6_KMAX + 3];
  float nislfv_rain_plm_qa[WSM6_KMAX + 3];
  float nislfv_rain_plm_qmi[WSM6_KMAX + 3];
  float nislfv_rain_plm_qpi[WSM6_KMAX + 3];
  float nislfv_rain_plm6_dz[WSM6_KMAX + 3];
  float nislfv_rain_plm6_ww[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qq[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qq2[WSM6_KMAX + 3];
  float nislfv_rain_plm6_wd[WSM6_KMAX + 3];
  float nislfv_rain_plm6_wa[WSM6_KMAX + 3];
  float nislfv_rain_plm6_wa2[WSM6_KMAX + 3];
  float nislfv_rain_plm6_was[WSM6_KMAX + 3];
  float nislfv_rain_plm6_den[WSM6_KMAX + 3];
  float nislfv_rain_plm6_denfac[WSM6_KMAX + 3];
  float nislfv_rain_plm6_tk[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qn[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qr[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qr2[WSM6_KMAX + 3];
  float nislfv_rain_plm6_tmp[WSM6_KMAX + 3];
  float nislfv_rain_plm6_tmp1[WSM6_KMAX + 3];
  float nislfv_rain_plm6_tmp2[WSM6_KMAX + 3];
  float nislfv_rain_plm6_tmp3[WSM6_KMAX + 3];
  float nislfv_rain_plm6_wi[WSM6_KMAX + 3];
  float nislfv_rain_plm6_zi[WSM6_KMAX + 3];
  float nislfv_rain_plm6_za[WSM6_KMAX + 3];
  float nislfv_rain_plm6_dza[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qa[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qa2[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qmi[WSM6_KMAX + 3];
  float nislfv_rain_plm6_qpi[WSM6_KMAX + 3];
  bool refl10cm_wsm6_l_qr[WSM6_KMAX + 3];
  bool refl10cm_wsm6_l_qs[WSM6_KMAX + 3];
  bool refl10cm_wsm6_l_qg[WSM6_KMAX + 3];
  float refl10cm_wsm6_temp[WSM6_KMAX + 3];
  float refl10cm_wsm6_pres[WSM6_KMAX + 3];
  float refl10cm_wsm6_qv[WSM6_KMAX + 3];
  float refl10cm_wsm6_rho[WSM6_KMAX + 3];
  float refl10cm_wsm6_rr[WSM6_KMAX + 3];
  float refl10cm_wsm6_rs[WSM6_KMAX + 3];
  float refl10cm_wsm6_rg[WSM6_KMAX + 3];
  float refl10cm_wsm6_ze_rain[WSM6_KMAX + 3];
  float refl10cm_wsm6_ze_snow[WSM6_KMAX + 3];
  float refl10cm_wsm6_ze_graupel[WSM6_KMAX + 3];
  double refl10cm_wsm6_ilamr[WSM6_KMAX + 3];
  double refl10cm_wsm6_ilams[WSM6_KMAX + 3];
  double refl10cm_wsm6_ilamg[WSM6_KMAX + 3];
  double refl10cm_wsm6_n0_r[WSM6_KMAX + 3];
  double refl10cm_wsm6_n0_s[WSM6_KMAX + 3];
  double refl10cm_wsm6_n0_g[WSM6_KMAX + 3];
  float mp_wsm6_effectrad_run_ni[WSM6_KMAX + 3];
  float mp_wsm6_effectrad_run_rqc[WSM6_KMAX + 3];
  float mp_wsm6_effectrad_run_rqi[WSM6_KMAX + 3];
  float mp_wsm6_effectrad_run_rni[WSM6_KMAX + 3];
  float mp_wsm6_effectrad_run_rqs[WSM6_KMAX + 3];

  float column_local[10*WSM6_KMAX], column_surface[7];
  float column_radii[3*WSM6_KMAX], column_radar[WSM6_KMAX];
  float column_production[2*WSM6_KMAX];
};


// mp_wsm6.F90:20-70; module parameter initialization
__device__ void state_parameters(Wsm6State &S) {
S.dtcldcr = 120.f;
S.n0r = 8.e6f;
S.avtr = 841.9f;
S.bvtr = 0.8f;
S.r0 = .8e-5f;
S.peaut = .55f;
S.xncr = 3.e8f;
S.xmyu = 1.718e-5f;
S.avts = 11.72f;
S.bvts = .41f;
S.lamdarmax = 8.e4f;
S.lamdasmax = 1.e5f;
S.dicon = 11.9f;
S.dimax = 500.e-6f;
S.pfrz1 = 100.f;
S.pfrz2 = 0.66f;
S.qcrmin = 1.e-9f;
S.eacrc = 1.0f;
S.dens = 100.0f;
S.qs0 = 6.e-4f;
S.n0smax = 1.e11f;
S.n0s = 2.e6f;
S.alpha = .12f;
S.nrbins = 50;
S.slen = 20;
S.lamda_radar = 0.10f;
S.basis[0] = div(1.e0, 3.e0);
S.basis[1] = div(4.e0, 3.e0);
S.basis[2] = div(1.e0, 3.e0);
S.melt_outside_s = 0.9e0;
S.melt_outside_g = 0.9e0;
}

// MPAS-A v8.4.2 mp_wsm6.F90:73-189
__device__ void mp_wsm6_init(Wsm6State &S, float den0, float denr, float dens, float cl, float cpv, int hail_opt, int &errflg);

// MPAS-A v8.4.2 mp_wsm6.F90:195-207
__device__ void mp_wsm6_finalize(Wsm6State &S, int &errflg);

// MPAS-A v8.4.2 mp_wsm6.F90:213-1468
__device__ void mp_wsm6_run(Wsm6State &S, float *t, float *q, float *qc, float *qi, float *qr, float *qs, float *qg, const float *den, const float *p, const float *delz, float delt, float g, float cpd, float cpv, float rd, float rv, float t0c, float ep1, float ep2, float qmin, float xls, float xlv0, float xlf0, float den0, float denr, float cliq, float cice, float psat, float *rain, float *rainncv, float *sr, float *snow, float *snowncv, float *graupel, float *graupelncv, float *rainprod2d, float *evapprod2d, int its, int ite, int kts, int kte, int &errflg);

// MPAS-A v8.4.2 mp_wsm6.F90:1471-1494
__device__ float rgmma(Wsm6State &S, float x);

// MPAS-A v8.4.2 mp_wsm6.F90:1497-1522
__device__ float fpvs(Wsm6State &S, float t, int ice, float rd, float rv, float cvap, float cliq, float cice, float hvap, float hsub, float psat, float t0c);

// MPAS-A v8.4.2 mp_wsm6.F90:1525-1602
__device__ void slope_wsm6(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte);

// MPAS-A v8.4.2 mp_wsm6.F90:1605-1648
__device__ void slope_rain(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte);

// MPAS-A v8.4.2 mp_wsm6.F90:1651-1699
__device__ void slope_snow(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte);

// MPAS-A v8.4.2 mp_wsm6.F90:1702-1748
__device__ void slope_graup(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte);

// MPAS-A v8.4.2 mp_wsm6.F90:1751-1994
__device__ void nislfv_rain_plm(Wsm6State &S, int im, int km, const float *denl, const float *denfacl, const float *tkl, const float *dzl, float *wwl, float *rql, float *precip, float dt, int id, int iter);

// MPAS-A v8.4.2 mp_wsm6.F90:1997-2272
__device__ void nislfv_rain_plm6(Wsm6State &S, int im, int km, const float *denl, const float *denfacl, const float *tkl, const float *dzl, float *wwl, float *rql, float *rql2, float *precip1, float *precip2, float dt, int id, int iter);

// MPAS-A v8.4.2 mp_wsm6.F90:2275-2444
__device__ void refl10cm_wsm6(Wsm6State &S, const float *qv1d, const float *qr1d, const float *qs1d, const float *qg1d, const float *t1d, const float *p1d, float *dbz, int kts, int kte);

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:23-35
__device__ void mp_wsm6_effectrad_init(Wsm6State &S, int &errflg);

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:41-53
__device__ void mp_wsm6_effectrad_finalize(Wsm6State &S, int &errflg);

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:59-193
__device__ void mp_wsm6_effectrad_run(Wsm6State &S, bool do_microp_re, const float *t, const float *qc, const float *qi, const float *qs, const float *rho, float qmin, float t0c, float re_qc_bg, float re_qi_bg, float re_qs_bg, float re_qc_max, float re_qi_max, float re_qs_max, float *re_qc, float *re_qi, float *re_qs, int its, int ite, int kts, int kte, int &errflg);

// MPAS-A v8.4.2 mp_radar.F90:72-189
__device__ void radar_init(Wsm6State &S);

// MPAS-A v8.4.2 mp_radar.F90:192-292
__device__ void rayleigh_soak_wetgraupel(Wsm6State &S, double x_g, double a_geo, double b_geo, double fmelt, double meltratio_outside, cx m_w, cx m_i, double lambda, double &c_back, const char *mixingrule, const char *matrix, const char *inclusion, const char *host, const char *hostmatrix, const char *hostinclusion);

// MPAS-A v8.4.2 mp_radar.F90:295-306
__device__ float wgamma(Wsm6State &S, float y);

// MPAS-A v8.4.2 mp_radar.F90:309-342
__device__ float gammln(Wsm6State &S, float xx);

// MPAS-A v8.4.2 mp_radar.F90:345-483
__device__ cx get_m_mix_nested(Wsm6State &S, cx m_a, cx m_i, cx m_w, double volair, double volice, double volwater, const char *mixingrule, const char *host, const char *matrix, const char *inclusion, const char *hostmatrix, const char *hostinclusion, int &cumulerror);

// MPAS-A v8.4.2 mp_radar.F90:486-533
__device__ cx get_m_mix(Wsm6State &S, cx m_a, cx m_i, cx m_w, double volair, double volice, double volwater, const char *mixingrule, const char *matrix, const char *inclusion, int &error);

// MPAS-A v8.4.2 mp_radar.F90:536-589
__device__ cx m_complex_maxwellgarnett(Wsm6State &S, double vol1, double vol2, double vol3, cx m1, cx m2, cx m3, const char *inclusion, int &error);

// MPAS-A v8.4.2 mp_radar.F90:592-629
__device__ cx m_complex_water_ray(Wsm6State &S, double lambda, double t);

// MPAS-A v8.4.2 mp_radar.F90:632-673
__device__ cx m_complex_ice_maetzler(Wsm6State &S, double lambda, double t);

// MPAS-A v8.4.2 mp_wsm6.F90:73-189
__device__ void mp_wsm6_init(Wsm6State &S, float den0, float denr, float dens, float cl, float cpv, int hail_opt, int &errflg) {
  if ((hail_opt == 1)) { // mp_wsm6.F90:88
  S.n0g = 4.e4f; // mp_wsm6.F90:89
  S.deng = 700.f; // mp_wsm6.F90:90
  S.avtg = 285.0f; // mp_wsm6.F90:91
  S.bvtg = 0.8f; // mp_wsm6.F90:92
  S.lamdagmax = 2.e4f; // mp_wsm6.F90:93
  } else { // mp_wsm6.F90:94
  S.n0g = 4.e6f; // mp_wsm6.F90:95
  S.deng = 500; // mp_wsm6.F90:96
  S.avtg = 330.0f; // mp_wsm6.F90:97
  S.bvtg = 0.8f; // mp_wsm6.F90:98
  S.lamdagmax = 6.e4f; // mp_wsm6.F90:99
  } // mp_wsm6.F90:100
  S.pi = mul(4.f, mp_atan(1.f)); // mp_wsm6.F90:102
  S.xlv1 = sub(cl, cpv); // mp_wsm6.F90:103
  S.qc0 = div(mul(mul(mul(mul(div(4.f, 3.f), S.pi), denr), ipow<3>(S.r0)), S.xncr), den0); // mp_wsm6.F90:105
  S.qck1 = mul(div(div(mul(mul(.104f, 9.8f), S.peaut), mp_pow(mul(S.xncr, denr), div(1.f, 3.f))), S.xmyu), mp_pow(den0, div(4.f, 3.f))); // mp_wsm6.F90:106
  S.pidnc = div(mul(S.pi, denr), 6.f); // mp_wsm6.F90:107
  S.bvtr1 = add(1.f, S.bvtr); // mp_wsm6.F90:109
  S.bvtr2 = add(2.5f, mul(.5f, S.bvtr)); // mp_wsm6.F90:110
  S.bvtr3 = add(3.f, S.bvtr); // mp_wsm6.F90:111
  S.bvtr4 = add(4.f, S.bvtr); // mp_wsm6.F90:112
  S.bvtr6 = add(6.f, S.bvtr); // mp_wsm6.F90:113
  S.g1pbr = rgmma(S, S.bvtr1); // mp_wsm6.F90:114
  S.g3pbr = rgmma(S, S.bvtr3); // mp_wsm6.F90:115
  S.g4pbr = rgmma(S, S.bvtr4); // mp_wsm6.F90:116
  S.g6pbr = rgmma(S, S.bvtr6); // mp_wsm6.F90:117
  S.g5pbro2 = rgmma(S, S.bvtr2); // mp_wsm6.F90:118
  S.pvtr = div(mul(S.avtr, S.g4pbr), 6.f); // mp_wsm6.F90:119
  S.eacrr = 1.0f; // mp_wsm6.F90:120
  S.pacrr = mul(mul(mul(mul(mul(S.pi, S.n0r), S.avtr), S.g3pbr), .25f), S.eacrr); // mp_wsm6.F90:121
  S.precr1 = mul(mul(mul(2.f, S.pi), S.n0r), .78f); // mp_wsm6.F90:122
  S.precr2 = mul(mul(mul(mul(mul(2.f, S.pi), S.n0r), .31f), mp_pow(S.avtr, .5f)), S.g5pbro2); // mp_wsm6.F90:123
  S.roqimax = mul(2.08e22f, ipow<8>(S.dimax)); // mp_wsm6.F90:124
  S.bvts1 = add(1.f, S.bvts); // mp_wsm6.F90:126
  S.bvts2 = add(2.5f, mul(.5f, S.bvts)); // mp_wsm6.F90:127
  S.bvts3 = add(3.f, S.bvts); // mp_wsm6.F90:128
  S.bvts4 = add(4.f, S.bvts); // mp_wsm6.F90:129
  S.g1pbs = rgmma(S, S.bvts1); // mp_wsm6.F90:130
  S.g3pbs = rgmma(S, S.bvts3); // mp_wsm6.F90:131
  S.g4pbs = rgmma(S, S.bvts4); // mp_wsm6.F90:132
  S.g5pbso2 = rgmma(S, S.bvts2); // mp_wsm6.F90:133
  S.pvts = div(mul(S.avts, S.g4pbs), 6.f); // mp_wsm6.F90:134
  S.pacrs = mul(mul(mul(mul(S.pi, S.n0s), S.avts), S.g3pbs), .25f); // mp_wsm6.F90:135
  S.precs1 = mul(mul(4.f, S.n0s), .65f); // mp_wsm6.F90:136
  S.precs2 = mul(mul(mul(mul(4.f, S.n0s), .44f), mp_pow(S.avts, .5f)), S.g5pbso2); // mp_wsm6.F90:137
  S.pidn0r = mul(mul(S.pi, denr), S.n0r); // mp_wsm6.F90:138
  S.pidn0s = mul(mul(S.pi, dens), S.n0s); // mp_wsm6.F90:139
  S.pacrc = mul(mul(mul(mul(mul(S.pi, S.n0s), S.avts), S.g3pbs), .25f), S.eacrc); // mp_wsm6.F90:141
  S.bvtg1 = add(1.f, S.bvtg); // mp_wsm6.F90:143
  S.bvtg2 = add(2.5f, mul(.5f, S.bvtg)); // mp_wsm6.F90:144
  S.bvtg3 = add(3.f, S.bvtg); // mp_wsm6.F90:145
  S.bvtg4 = add(4.f, S.bvtg); // mp_wsm6.F90:146
  S.g1pbg = rgmma(S, S.bvtg1); // mp_wsm6.F90:147
  S.g3pbg = rgmma(S, S.bvtg3); // mp_wsm6.F90:148
  S.g4pbg = rgmma(S, S.bvtg4); // mp_wsm6.F90:149
  S.pacrg = mul(mul(mul(mul(S.pi, S.n0g), S.avtg), S.g3pbg), .25f); // mp_wsm6.F90:150
  S.g5pbgo2 = rgmma(S, S.bvtg2); // mp_wsm6.F90:151
  S.pvtg = div(mul(S.avtg, S.g4pbg), 6.f); // mp_wsm6.F90:152
  S.precg1 = mul(mul(mul(2.f, S.pi), S.n0g), .78f); // mp_wsm6.F90:153
  S.precg2 = mul(mul(mul(mul(mul(2.f, S.pi), S.n0g), .31f), mp_pow(S.avtg, .5f)), S.g5pbgo2); // mp_wsm6.F90:154
  S.pidn0g = mul(mul(S.pi, S.deng), S.n0g); // mp_wsm6.F90:155
  S.rslopermax = div(1.f, S.lamdarmax); // mp_wsm6.F90:157
  S.rslopesmax = div(1.f, S.lamdasmax); // mp_wsm6.F90:158
  S.rslopegmax = div(1.f, S.lamdagmax); // mp_wsm6.F90:159
  S.rsloperbmax = mp_pow(S.rslopermax, S.bvtr); // mp_wsm6.F90:160
  S.rslopesbmax = mp_pow(S.rslopesmax, S.bvts); // mp_wsm6.F90:161
  S.rslopegbmax = mp_pow(S.rslopegmax, S.bvtg); // mp_wsm6.F90:162
  S.rsloper2max = mul(S.rslopermax, S.rslopermax); // mp_wsm6.F90:163
  S.rslopes2max = mul(S.rslopesmax, S.rslopesmax); // mp_wsm6.F90:164
  S.rslopeg2max = mul(S.rslopegmax, S.rslopegmax); // mp_wsm6.F90:165
  S.rsloper3max = mul(S.rsloper2max, S.rslopermax); // mp_wsm6.F90:166
  S.rslopes3max = mul(S.rslopes2max, S.rslopesmax); // mp_wsm6.F90:167
  S.rslopeg3max = mul(S.rslopeg2max, S.rslopegmax); // mp_wsm6.F90:168
  S.xam_r = div(mul(S.pi, denr), 6.f); // mp_wsm6.F90:174
  S.xbm_r = 3.f; // mp_wsm6.F90:175
  S.xmu_r = 0.f; // mp_wsm6.F90:176
  S.xam_s = div(mul(S.pi, dens), 6.f); // mp_wsm6.F90:177
  S.xbm_s = 3.f; // mp_wsm6.F90:178
  S.xmu_s = 0.f; // mp_wsm6.F90:179
  S.xam_g = div(mul(S.pi, S.deng), 6.f); // mp_wsm6.F90:180
  S.xbm_g = 3.f; // mp_wsm6.F90:181
  S.xmu_g = 0.f; // mp_wsm6.F90:182
  radar_init(S); // mp_wsm6.F90:184
  // CCPP status text // mp_wsm6.F90:186
  S.last_message = "mp_wsm6_init OK"; // mp_wsm6.F90:186
  errflg = 0; // mp_wsm6.F90:187
}

// MPAS-A v8.4.2 mp_wsm6.F90:195-207
__device__ void mp_wsm6_finalize(Wsm6State &S, int &errflg) {
  // CCPP status text // mp_wsm6.F90:204
  S.last_message = "mp_wsm6_finalize OK"; // mp_wsm6.F90:204
  errflg = 0; // mp_wsm6.F90:205
}

// MPAS-A v8.4.2 mp_wsm6.F90:213-1468
__device__ void mp_wsm6_run(Wsm6State &S, float *t, float *q, float *qc, float *qi, float *qr, float *qs, float *qg, const float *den, const float *p, const float *delz, float delt, float g, float cpd, float cpv, float rd, float rv, float t0c, float ep1, float ep2, float qmin, float xls, float xlv0, float xlf0, float den0, float denr, float cliq, float cice, float psat, float *rain, float *rainncv, float *sr, float *snow, float *snowncv, float *graupel, float *graupelncv, float *rainprod2d, float *evapprod2d, int its, int ite, int kts, int kte, int &errflg) {
#ifdef __CUDA_ARCH__
float *rh=S.workspace->mp_wsm6_run_rh;
#else
float rh[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *qsat=S.workspace->mp_wsm6_run_qsat;
#else
float qsat[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *rslope=S.workspace->mp_wsm6_run_rslope;
#else
float rslope[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *rslope2=S.workspace->mp_wsm6_run_rslope2;
#else
float rslope2[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *rslope3=S.workspace->mp_wsm6_run_rslope3;
#else
float rslope3[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *rslopeb=S.workspace->mp_wsm6_run_rslopeb;
#else
float rslopeb[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *qrs_tmp=S.workspace->mp_wsm6_run_qrs_tmp;
#else
float qrs_tmp[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *falk=S.workspace->mp_wsm6_run_falk;
#else
float falk[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *fall=S.workspace->mp_wsm6_run_fall;
#else
float fall[WSM6_KMAX * 3 + 3];
#endif
#ifdef __CUDA_ARCH__
float *work1=S.workspace->mp_wsm6_run_work1;
#else
float work1[WSM6_KMAX * 3 + 3];
#endif
// Native column-sized storage: mp_wsm6.F90:331-392.
#ifdef __CUDA_ARCH__
float *fallc=S.workspace->mp_wsm6_run_fallc;
#else
float fallc[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *falkc=S.workspace->mp_wsm6_run_falkc;
#else
float falkc[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *work1c=S.workspace->mp_wsm6_run_work1c;
#else
float work1c[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *work2c=S.workspace->mp_wsm6_run_work2c;
#else
float work2c[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *workr=S.workspace->mp_wsm6_run_workr;
#else
float workr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *worka=S.workspace->mp_wsm6_run_worka;
#else
float worka[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *den_tmp=S.workspace->mp_wsm6_run_den_tmp;
#else
float den_tmp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *delz_tmp=S.workspace->mp_wsm6_run_delz_tmp;
#else
float delz_tmp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pigen=S.workspace->mp_wsm6_run_pigen;
#else
float pigen[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pidep=S.workspace->mp_wsm6_run_pidep;
#else
float pidep[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pcond=S.workspace->mp_wsm6_run_pcond;
#else
float pcond[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *prevp=S.workspace->mp_wsm6_run_prevp;
#else
float prevp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psevp=S.workspace->mp_wsm6_run_psevp;
#else
float psevp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgevp=S.workspace->mp_wsm6_run_pgevp;
#else
float pgevp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psdep=S.workspace->mp_wsm6_run_psdep;
#else
float psdep[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgdep=S.workspace->mp_wsm6_run_pgdep;
#else
float pgdep[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *praut=S.workspace->mp_wsm6_run_praut;
#else
float praut[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psaut=S.workspace->mp_wsm6_run_psaut;
#else
float psaut[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgaut=S.workspace->mp_wsm6_run_pgaut;
#else
float pgaut[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *piacr=S.workspace->mp_wsm6_run_piacr;
#else
float piacr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pracw=S.workspace->mp_wsm6_run_pracw;
#else
float pracw[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *praci=S.workspace->mp_wsm6_run_praci;
#else
float praci[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pracs=S.workspace->mp_wsm6_run_pracs;
#else
float pracs[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psacw=S.workspace->mp_wsm6_run_psacw;
#else
float psacw[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psaci=S.workspace->mp_wsm6_run_psaci;
#else
float psaci[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psacr=S.workspace->mp_wsm6_run_psacr;
#else
float psacr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgacw=S.workspace->mp_wsm6_run_pgacw;
#else
float pgacw[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgaci=S.workspace->mp_wsm6_run_pgaci;
#else
float pgaci[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgacr=S.workspace->mp_wsm6_run_pgacr;
#else
float pgacr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgacs=S.workspace->mp_wsm6_run_pgacs;
#else
float pgacs[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *paacw=S.workspace->mp_wsm6_run_paacw;
#else
float paacw[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *psmlt=S.workspace->mp_wsm6_run_psmlt;
#else
float psmlt[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgmlt=S.workspace->mp_wsm6_run_pgmlt;
#else
float pgmlt[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pseml=S.workspace->mp_wsm6_run_pseml;
#else
float pseml[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pgeml=S.workspace->mp_wsm6_run_pgeml;
#else
float pgeml[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qsum=S.workspace->mp_wsm6_run_qsum;
#else
float qsum[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *xl=S.workspace->mp_wsm6_run_xl;
#else
float xl[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *cpm=S.workspace->mp_wsm6_run_cpm;
#else
float cpm[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *work2=S.workspace->mp_wsm6_run_work2;
#else
float work2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denfac=S.workspace->mp_wsm6_run_denfac;
#else
float denfac[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *xni=S.workspace->mp_wsm6_run_xni;
#else
float xni[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denqrs1=S.workspace->mp_wsm6_run_denqrs1;
#else
float denqrs1[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denqrs2=S.workspace->mp_wsm6_run_denqrs2;
#else
float denqrs2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denqrs3=S.workspace->mp_wsm6_run_denqrs3;
#else
float denqrs3[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denqci=S.workspace->mp_wsm6_run_denqci;
#else
float denqci[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *n0sfac=S.workspace->mp_wsm6_run_n0sfac;
#else
float n0sfac[WSM6_KMAX + 3];
#endif
float delqrs1[(1)];
float delqrs2[(1)];
float delqrs3[(1)];
float delqi[(1)];
float tstepsnow[(1)];
float tstepgraup[(1)];
int mstep[(1)];
int numdt[(1)];
bool flgcld[(1)];
float x;
float y;
float z;
float a;
float b;
float c;
float d;
float e;
float qdt;
float holdrr;
float holdrs;
float holdrg;
float supcol;
float supcolt;
float pvt;
float coeres;
float supsat;
float dtcld;
float xmi;
float eacrs;
float satdt;
float qimax;
float diameter;
float xni0;
float roqi0;
float fallsum;
float fallsum_qsi;
float fallsum_qg;
float vt2i;
float vt2r;
float vt2s;
float vt2g;
float acrfac;
float egs;
float egi;
float xlwork2;
float factor;
float source;
float value;
float xlf;
float pfrzdtc;
float pfrzdtr;
float supice;
float alpha2;
float delta2;
float delta3;
float vt2ave;
float holdc;
float holdci;
int i;
int j;
int k;
int mstepmax;
int iprt;
int latd;
int lond;
int loop;
int loops;
int ifsat;
int n;
int idim;
int kdim;
float dldti;
float xb;
float xai;
float tr;
float xbi;
float xa;
float hvap;
float cvap;
float hsub;
float dldt;
float ttp;
float dvec1[(1)];
float tvec1[(1)];
float temp;
  auto cpmcal = [&](float x) { return add(mul(cpd, sub(1.f, mp_max(x, qmin))), mul(mp_max(x, qmin), cpv)); }; // mp_wsm6.F90:418
  auto xlcal = [&](float x) { return sub(xlv0, mul(S.xlv1, sub(x, t0c))); }; // mp_wsm6.F90:419
  auto diffus = [&](float x, float y) { return div(mul(8.794e-5f, mp_exp(mul(mp_log(x), 1.81f))), y); }; // mp_wsm6.F90:425
  auto viscos = [&](float x, float y) { return div(div(mul(1.496e-6f, mul(x, mp_sqrt(x))), add(x, 120.f)), y); }; // mp_wsm6.F90:426
  auto xka = [&](float x, float y) { return mul(mul(1.414e3f, viscos(x, y)), y); }; // mp_wsm6.F90:427
  auto diffac = [&](float a, float b, float c, float d, float e) { return add(div(mul(mul(d, a), a), mul(mul(mul(xka(c, d), rv), c), c)), div(1.f, mul(e, diffus(c, b)))); }; // mp_wsm6.F90:428
  auto venfac = [&](float a, float b, float c) { return mul(div(mp_exp(mul(mp_log(div(viscos(b, c), diffus(b, a))), .3333333f)), mp_sqrt(viscos(b, c))), mp_sqrt(mp_sqrt(div(den0, c)))); }; // mp_wsm6.F90:429
  auto conden = [&](float a, float b, float c, float d, float e) { return div(sub(mp_max(b, qmin), c), add(1.f, div(mul(div(mul(d, d), mul(rv, e)), c), mul(a, a)))); }; // mp_wsm6.F90:431
  idim = ((ite - its) + 1); // mp_wsm6.F90:434
  kdim = ((kte - kts) + 1); // mp_wsm6.F90:435
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:440
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:441
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:442
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:443
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:444
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:445
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:446
   } // mp_wsm6.F90:447
   } // mp_wsm6.F90:448
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:454
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:455
  cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = cpmcal(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:456
  xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = xlcal(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:457
   } // mp_wsm6.F90:458
   } // mp_wsm6.F90:459
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:460
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:461
  delz_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:462
  den_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:463
   } // mp_wsm6.F90:464
   } // mp_wsm6.F90:465
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:470
  rainncv[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:471
  if (((snowncv != nullptr) && (snow != nullptr))) { snowncv[((i)-(its))*(1)] = 0.f; } // mp_wsm6.F90:472
  if (((graupelncv != nullptr) && (graupel != nullptr))) { graupelncv[((i)-(its))*(1)] = 0.f; } // mp_wsm6.F90:473
  sr[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:474
  tstepsnow[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:476
  tstepgraup[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:477
   } // mp_wsm6.F90:478
  loops = mp_max(mp_nint(div(delt, S.dtcldcr)), 1); // mp_wsm6.F90:483
  dtcld = div(delt, loops); // mp_wsm6.F90:484
  if ((delt <= S.dtcldcr)) { dtcld = delt; } // mp_wsm6.F90:485
  for (loop=1; loop<=loops; loop+=1) { // mp_wsm6.F90:487
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:492
  mstep[((i)-(its))*(1)] = 1; // mp_wsm6.F90:493
  flgcld[((i)-(its))*(1)] = true; // mp_wsm6.F90:494
   } // mp_wsm6.F90:495
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:502
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:503
  dvec1[((i)-(its))*(1)] = den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:504
   } // mp_wsm6.F90:505
  for(int vi=0; vi<((ite - its) + 1); ++vi) tvec1[vi] = div(1.0f, dvec1[vi]); // mp_wsm6.F90:506
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:507
  tvec1[((i)-(its))*(1)] = mul(tvec1[((i)-(its))*(1)], den0); // mp_wsm6.F90:508
   } // mp_wsm6.F90:509
  for(int vi=0; vi<((ite - its) + 1); ++vi) dvec1[vi] = mp_sqrt(tvec1[vi]); // mp_wsm6.F90:510
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:511
  denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = dvec1[((i)-(its))*(1)]; // mp_wsm6.F90:512
   } // mp_wsm6.F90:513
   } // mp_wsm6.F90:514
  hsub = xls; // mp_wsm6.F90:519
  hvap = xlv0; // mp_wsm6.F90:520
  cvap = cpv; // mp_wsm6.F90:521
  ttp = add(t0c, 0.01f); // mp_wsm6.F90:522
  dldt = sub(cvap, cliq); // mp_wsm6.F90:523
  xa = div((-dldt), rv); // mp_wsm6.F90:524
  xb = add(xa, div(hvap, mul(rv, ttp))); // mp_wsm6.F90:525
  dldti = sub(cvap, cice); // mp_wsm6.F90:526
  xai = div((-dldti), rv); // mp_wsm6.F90:527
  xbi = add(xai, div(hsub, mul(rv, ttp))); // mp_wsm6.F90:528
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:529
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:530
  tr = div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:531
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xa))), mp_exp(mul(xb, sub(1.f, tr)))); // mp_wsm6.F90:532
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:533
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:534
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:535
  rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(div(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qmin); // mp_wsm6.F90:536
  tr = div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:537
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < ttp)) { // mp_wsm6.F90:538
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xai))), mp_exp(mul(xbi, sub(1.f, tr)))); // mp_wsm6.F90:539
  } else { // mp_wsm6.F90:540
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xa))), mp_exp(mul(xb, sub(1.f, tr)))); // mp_wsm6.F90:541
  } // mp_wsm6.F90:542
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:543
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:544
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:545
  rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(div(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qmin); // mp_wsm6.F90:546
   } // mp_wsm6.F90:547
   } // mp_wsm6.F90:548
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:554
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:555
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:556
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:557
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:558
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:559
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:560
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:561
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:562
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:563
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:564
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:565
  psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:566
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:567
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:568
  pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:569
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:570
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:571
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:572
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:573
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:574
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:575
  pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:576
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:577
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:578
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:579
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:580
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:581
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:582
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:583
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:584
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:585
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:586
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:587
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:588
  fallc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:589
  falkc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:590
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 1.e3f; // mp_wsm6.F90:591
   } // mp_wsm6.F90:592
   } // mp_wsm6.F90:593
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:597
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:598
  temp = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mp_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin)); // mp_wsm6.F90:599
  temp = mp_sqrt(mp_sqrt(mul(mul(temp, temp), temp))); // mp_wsm6.F90:600
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(mul(5.38e7f, temp), 1.e3f), 1.e6f); // mp_wsm6.F90:601
   } // mp_wsm6.F90:602
   } // mp_wsm6.F90:603
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:609
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:610
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:611
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:612
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:613
   } // mp_wsm6.F90:614
   } // mp_wsm6.F90:615
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:616
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:619
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:620
  workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:621
  qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 1.e-15f); // mp_wsm6.F90:622
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:623
  worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(add(mul(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mul(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:624
  } else { // mp_wsm6.F90:626
  worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:627
  } // mp_wsm6.F90:628
  denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:629
  denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:630
  denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:631
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= 0.0f)) { workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:632
   } // mp_wsm6.F90:633
   } // mp_wsm6.F90:634
  nislfv_rain_plm(S, idim, kdim, den_tmp, denfac, t, delz_tmp, workr, denqrs1, delqrs1, dtcld, 1, 1); // mp_wsm6.F90:635
  nislfv_rain_plm6(S, idim, kdim, den_tmp, denfac, t, delz_tmp, worka, denqrs2, denqrs3, delqrs2, delqrs3, dtcld, 1, 1); // mp_wsm6.F90:637
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:639
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:640
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(div(denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:641
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(div(denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:642
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(div(denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:643
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:644
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:645
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:646
   } // mp_wsm6.F90:647
   } // mp_wsm6.F90:648
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:649
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(div(delqrs1[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:650
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(div(delqrs2[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:651
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(div(delqrs3[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:652
   } // mp_wsm6.F90:653
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:654
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:655
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:656
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:657
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:658
   } // mp_wsm6.F90:659
   } // mp_wsm6.F90:660
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:661
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:664
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:665
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:666
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mp_exp(mul(S.alpha, supcol)), div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:667
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > t0c)) { // mp_wsm6.F90:668
  xlf = xlf0; // mp_wsm6.F90:673
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = venfac(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:674
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:675
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:676
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(div(mul(mul(div(xka(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xlf), sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), S.pi), 2.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), add(mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:677
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(div(mul(psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld), mstep[((i)-(its))*(1)]), div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mstep[((i)-(its))*(1)])), 0.f); // mp_wsm6.F90:680
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:682
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:683
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:684
  } // mp_wsm6.F90:685
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:690
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:691
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(div(xka(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xlf), sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), add(mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:692
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(div(mul(pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld), mstep[((i)-(its))*(1)]), div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mstep[((i)-(its))*(1)])), 0.f); // mp_wsm6.F90:695
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:697
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:698
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:699
  } // mp_wsm6.F90:700
  } // mp_wsm6.F90:701
   } // mp_wsm6.F90:702
   } // mp_wsm6.F90:703
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:707
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:708
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= 0.f)) { // mp_wsm6.F90:709
  work1c[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:710
  } else { // mp_wsm6.F90:711
  xmi = div(mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:712
  diameter = mp_max(mp_min(mul(S.dicon, mp_sqrt(xmi)), S.dimax), 1.e-25f); // mp_wsm6.F90:713
  work1c[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(1.49e4f, mp_exp(mul(mp_log(diameter), 1.31f))); // mp_wsm6.F90:714
  } // mp_wsm6.F90:715
   } // mp_wsm6.F90:716
   } // mp_wsm6.F90:717
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:721
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:722
  denqci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:723
   } // mp_wsm6.F90:724
   } // mp_wsm6.F90:725
  nislfv_rain_plm(S, idim, kdim, den_tmp, denfac, t, delz_tmp, work1c, denqci, delqi, dtcld, 1, 0); // mp_wsm6.F90:726
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:728
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:729
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(div(denqci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:730
   } // mp_wsm6.F90:731
   } // mp_wsm6.F90:732
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:733
  fallc[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))] = div(div(delqi[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:734
   } // mp_wsm6.F90:735
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:740
  fallsum = add(add(add(fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), fallc[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:741
  fallsum_qsi = add(fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], fallc[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:742
  fallsum_qg = fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:743
  if ((fallsum > 0.f)) { // mp_wsm6.F90:744
  rainncv[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), rainncv[((i)-(its))*(1)]); // mp_wsm6.F90:745
  rain[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), rain[((i)-(its))*(1)]); // mp_wsm6.F90:746
  } // mp_wsm6.F90:747
  if ((fallsum_qsi > 0.f)) { // mp_wsm6.F90:748
  tstepsnow[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), tstepsnow[((i)-(its))*(1)]); // mp_wsm6.F90:749
  if (((snowncv != nullptr) && (snow != nullptr))) { // mp_wsm6.F90:751
  snowncv[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), snowncv[((i)-(its))*(1)]); // mp_wsm6.F90:752
  snow[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), snow[((i)-(its))*(1)]); // mp_wsm6.F90:754
  } // mp_wsm6.F90:755
  } // mp_wsm6.F90:756
  if ((fallsum_qg > 0.f)) { // mp_wsm6.F90:757
  tstepgraup[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), tstepgraup[((i)-(its))*(1)]); // mp_wsm6.F90:758
  if (((graupelncv != nullptr) && (graupel != nullptr))) { // mp_wsm6.F90:760
  graupelncv[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), graupelncv[((i)-(its))*(1)]); // mp_wsm6.F90:761
  graupel[((i)-(its))*(1)] = add(mul(mul(div(mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), graupel[((i)-(its))*(1)]); // mp_wsm6.F90:763
  } // mp_wsm6.F90:764
  } // mp_wsm6.F90:765
  if ((snowncv != nullptr)) { // mp_wsm6.F90:766
  if ((fallsum > 0.f)) { sr[((i)-(its))*(1)] = div(add(snowncv[((i)-(its))*(1)], graupelncv[((i)-(its))*(1)]), add(rainncv[((i)-(its))*(1)], 1.e-12f)); } // mp_wsm6.F90:767
  } else { // mp_wsm6.F90:768
  if ((fallsum > 0.f)) { sr[((i)-(its))*(1)] = div(add(tstepsnow[((i)-(its))*(1)], tstepgraup[((i)-(its))*(1)]), add(rainncv[((i)-(its))*(1)], 1.e-12f)); } // mp_wsm6.F90:769
  } // mp_wsm6.F90:770
   } // mp_wsm6.F90:771
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:777
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:778
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:779
  xlf = sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:780
  if ((supcol < 0.f)) { xlf = xlf0; } // mp_wsm6.F90:781
  if (((supcol < 0) && (qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:782
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:783
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:784
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:785
  } // mp_wsm6.F90:786
  if (((supcol > 40.f) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:791
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:792
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:793
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:794
  } // mp_wsm6.F90:795
  if (((supcol > 0.f) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:800
  supcolt = mp_min(supcol, 50.f); // mp_wsm6.F90:803
  pfrzdtc = mp_min(mul(mul(mul(div(div(mul(mul(S.pfrz1, sub(mp_exp(mul(S.pfrz2, supcolt)), 1.f)), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denr), S.xncr), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:804
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtc); // mp_wsm6.F90:806
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pfrzdtc)); // mp_wsm6.F90:807
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtc); // mp_wsm6.F90:808
  } // mp_wsm6.F90:809
  if (((supcol > 0.f) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:814
  temp = rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:818
  temp = mul(mul(temp, temp), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:819
  supcolt = mp_min(supcol, 50.f); // mp_wsm6.F90:820
  pfrzdtr = mp_min(mul(mul(mul(div(mul(mul(mul(mul(20.f, mul(S.pi, S.pi)), S.pfrz1), S.n0r), denr), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), sub(mp_exp(mul(S.pfrz2, supcolt)), 1.f)), temp), dtcld), qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:821
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtr); // mp_wsm6.F90:824
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pfrzdtr)); // mp_wsm6.F90:825
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtr); // mp_wsm6.F90:826
  } // mp_wsm6.F90:827
   } // mp_wsm6.F90:828
   } // mp_wsm6.F90:829
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:835
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:836
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:837
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:838
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:839
   } // mp_wsm6.F90:840
   } // mp_wsm6.F90:841
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:842
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:850
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:851
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = diffac(xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:852
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = diffac(xls, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:853
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = venfac(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:854
   } // mp_wsm6.F90:855
   } // mp_wsm6.F90:856
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:866
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:867
  supsat = sub(mp_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin), qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:868
  satdt = div(supsat, dtcld); // mp_wsm6.F90:869
  if ((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qc0)) { // mp_wsm6.F90:874
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(S.qck1, mp_pow(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], div(7.f, 3.f))); // mp_wsm6.F90:875
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:876
  } // mp_wsm6.F90:877
  if (((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:882
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mul(mul(mul(mul(S.pacrr, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:883
  } // mp_wsm6.F90:885
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:890
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:891
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), add(mul(S.precr1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precr2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:892
  if ((prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:894
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:895
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)); // mp_wsm6.F90:896
  } else { // mp_wsm6.F90:897
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)); // mp_wsm6.F90:898
  } // mp_wsm6.F90:899
  } // mp_wsm6.F90:900
   } // mp_wsm6.F90:901
   } // mp_wsm6.F90:902
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:917
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:918
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:919
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mp_exp(mul(S.alpha, supcol)), div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:920
  supsat = sub(mp_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin), qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:921
  satdt = div(supsat, dtcld); // mp_wsm6.F90:922
  ifsat = 0; // mp_wsm6.F90:923
  temp = mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mp_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin)); // mp_wsm6.F90:929
  temp = mp_sqrt(mp_sqrt(mul(mul(temp, temp), temp))); // mp_wsm6.F90:930
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(mul(5.38e7f, temp), 1.e3f), 1.e6f); // mp_wsm6.F90:931
  eacrs = mp_exp(mul(0.07f, (-supcol))); // mp_wsm6.F90:932
  xmi = div(mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:934
  diameter = mp_min(mul(S.dicon, mp_sqrt(xmi)), S.dimax); // mp_wsm6.F90:935
  vt2i = mul(1.49e4f, mp_pow(diameter, 1.31f)); // mp_wsm6.F90:936
  vt2r = mul(mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:937
  vt2s = mul(mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:938
  vt2g = mul(mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:939
  qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 1.e-15f); // mp_wsm6.F90:940
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:941
  vt2ave = div(add(mul(vt2s, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mul(vt2g, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:942
  } else { // mp_wsm6.F90:943
  vt2ave = 0.f; // mp_wsm6.F90:944
  } // mp_wsm6.F90:945
  if (((supcol > 0) && (qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:946
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:947
  acrfac = add(add(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:952
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(mul(mul(S.pi, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), S.n0r), mp_abs(sub(vt2r, vt2i))), acrfac), 4.f); // mp_wsm6.F90:954
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(mp_min(mp_max(div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:956
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:957
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(div(mul(mul(mul(mul(mul(mul(mul(mul(mul(ipow<2>(S.pi), S.avtr), S.n0r), denr), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), S.g6pbr), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), 24.f), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:962
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(mp_min(mp_max(div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:966
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:967
  } // mp_wsm6.F90:968
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:973
  acrfac = add(add(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:974
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(mul(mul(mul(mul(S.pi, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), eacrs), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mp_abs(sub(vt2ave, vt2i))), acrfac), 4.f); // mp_wsm6.F90:976
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:978
  } // mp_wsm6.F90:979
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:984
  egi = mp_exp(mul(0.07f, (-supcol))); // mp_wsm6.F90:985
  acrfac = add(add(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:986
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(mul(mul(mul(S.pi, egi), qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), S.n0g), mp_abs(sub(vt2ave, vt2i))), acrfac), 4.f); // mp_wsm6.F90:988
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:989
  } // mp_wsm6.F90:990
  } // mp_wsm6.F90:991
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:996
  psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mul(mul(mul(mul(mul(mul(S.pacrc, n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), ipow<2>(mp_min(mp_max(div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:997
  } // mp_wsm6.F90:1001
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:1006
  pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mul(mul(mul(mul(mul(S.pacrg, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), ipow<2>(mp_min(mp_max(div(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1007
  } // mp_wsm6.F90:1011
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:1016
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(add(mul(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1017
  } // mp_wsm6.F90:1019
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1024
  if ((supcol > 0)) { // mp_wsm6.F90:1025
  acrfac = add(add(mul(mul(mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(mul(mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1026
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(mul(mul(mul(mul(ipow<2>(S.pi), S.n0r), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mp_abs(sub(vt2r, vt2ave))), div(S.dens, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1029
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(mp_min(mp_max(div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1032
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1033
  } // mp_wsm6.F90:1034
  acrfac = add(add(mul(mul(mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(mul(mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1039
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(mul(mul(mul(mul(ipow<2>(S.pi), S.n0r), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mp_abs(sub(vt2ave, vt2r))), div(denr, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1042
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(mp_min(mp_max(div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1045
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1046
  } // mp_wsm6.F90:1047
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1052
  acrfac = add(add(mul(mul(mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), mul(mul(mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1053
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(mul(mul(mul(ipow<2>(S.pi), S.n0r), S.n0g), mp_abs(sub(vt2ave, vt2r))), div(denr, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1056
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(mp_min(mp_max(div(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1059
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1060
  } // mp_wsm6.F90:1061
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1068
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:1069
  } // mp_wsm6.F90:1070
  if ((supcol <= 0)) { // mp_wsm6.F90:1071
  xlf = xlf0; // mp_wsm6.F90:1072
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(div(mul(mul(cliq, supcol), add(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), xlf), div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); } // mp_wsm6.F90:1077
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(div(mul(mul(cliq, supcol), add(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), xlf), div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); } // mp_wsm6.F90:1084
  } // mp_wsm6.F90:1087
  if ((supcol > 0)) { // mp_wsm6.F90:1088
  if (((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0) && (ifsat != 1))) { // mp_wsm6.F90:1093
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(mul(4.f, diameter), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f)), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1094
  supice = sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1095
  if ((pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1096
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_max(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1097
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1098
  } else { // mp_wsm6.F90:1099
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_min(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1100
  } // mp_wsm6.F90:1101
  if ((mp_abs(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= mp_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1102
  } // mp_wsm6.F90:1103
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (ifsat != 1))) { // mp_wsm6.F90:1108
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1109
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), add(mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1110
  supice = sub(sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1112
  if ((psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1113
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1114
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_max(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1115
  } else { // mp_wsm6.F90:1116
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_min(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1117
  } // mp_wsm6.F90:1118
  if ((mp_abs(add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= mp_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1119
  } // mp_wsm6.F90:1121
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (ifsat != 1))) { // mp_wsm6.F90:1126
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1127
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), add(mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1128
  supice = sub(sub(sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1130
  if ((pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1131
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1132
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_max(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1133
  } else { // mp_wsm6.F90:1134
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_min(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div(satdt, 2)), supice); // mp_wsm6.F90:1135
  } // mp_wsm6.F90:1136
  if ((mp_abs(add(add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= mp_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1137
  } // mp_wsm6.F90:1139
  if (((supsat > 0) && (ifsat != 1))) { // mp_wsm6.F90:1144
  supice = sub(sub(sub(sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1145
  xni0 = mul(1.e3f, mp_exp(mul(0.1f, supcol))); // mp_wsm6.F90:1146
  roqi0 = mul(4.92e-11f, mp_pow(xni0, 1.33f)); // mp_wsm6.F90:1147
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(div(sub(div(roqi0, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mp_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.f)), dtcld), 0.f); // mp_wsm6.F90:1148
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_min(pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], satdt), supice); // mp_wsm6.F90:1149
  } // mp_wsm6.F90:1150
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:1156
  qimax = div(S.roqimax, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1157
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(div(sub(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qimax), dtcld), 0.f); // mp_wsm6.F90:1158
  } // mp_wsm6.F90:1159
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:1165
  alpha2 = mul(1.e-3f, mp_exp(mul(0.09f, (-supcol)))); // mp_wsm6.F90:1166
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(mul(alpha2, sub(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], S.qs0)), 0.f), div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1167
  } // mp_wsm6.F90:1168
  } // mp_wsm6.F90:1169
  if ((supcol < 0.f)) { // mp_wsm6.F90:1175
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 1.f))) { // mp_wsm6.F90:1176
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1177
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(mul(sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), add(mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1178
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1181
  } // mp_wsm6.F90:1182
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 1.f))) { // mp_wsm6.F90:1187
  coeres = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mp_sqrt(mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1188
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mul(sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), add(mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), mul(mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1189
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1191
  } // mp_wsm6.F90:1192
  } // mp_wsm6.F90:1193
   } // mp_wsm6.F90:1194
   } // mp_wsm6.F90:1195
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1202
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1203
  delta2 = 0.f; // mp_wsm6.F90:1205
  delta3 = 0.f; // mp_wsm6.F90:1206
  if (((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f) && (qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f))) { delta2 = 1.f; } // mp_wsm6.F90:1207
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f)) { delta3 = 1.f; } // mp_wsm6.F90:1208
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= t0c)) { // mp_wsm6.F90:1209
  value = mp_max(qmin, qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1213
  source = mul(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1214
  if ((source > value)) { // mp_wsm6.F90:1215
  factor = div(value, source); // mp_wsm6.F90:1216
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1217
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1218
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1219
  } // mp_wsm6.F90:1220
  value = mp_max(qmin, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1224
  source = mul(add(add(add(sub(sub(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1225
  if ((source > value)) { // mp_wsm6.F90:1227
  factor = div(value, source); // mp_wsm6.F90:1228
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1229
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1230
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1231
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1232
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1233
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1234
  } // mp_wsm6.F90:1235
  value = mp_max(qmin, qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1239
  source = mul(add(add(add(sub(sub((-praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1240
  if ((source > value)) { // mp_wsm6.F90:1242
  factor = div(value, source); // mp_wsm6.F90:1243
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1244
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1245
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1246
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1247
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1248
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1249
  } // mp_wsm6.F90:1250
  value = mp_max(qmin, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1254
  source = mul((-sub(add(add(sub(add(add(add(sub(add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta2)), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1255
  if ((source > value)) { // mp_wsm6.F90:1258
  factor = div(value, source); // mp_wsm6.F90:1259
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1260
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1261
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1262
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1263
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1264
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1265
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1266
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1267
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1268
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1269
  } // mp_wsm6.F90:1270
  value = mp_max(qmin, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1274
  source = mul((-add(add(add(add(add(add(add(add(add(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta3))), mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta3))), mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1275
  if ((source > value)) { // mp_wsm6.F90:1279
  factor = div(value, source); // mp_wsm6.F90:1280
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1281
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1282
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1283
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1284
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1285
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1286
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1287
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1288
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1289
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1290
  } // mp_wsm6.F90:1291
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = (-add(add(add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1293
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1295
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1296
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(sub(sub(sub(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1298
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(sub(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(sub(sub(add(add(add(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1301
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(sub(sub(add(add(add(sub(add(add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta2)), dtcld)), 0.f); // mp_wsm6.F90:1304
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(add(add(add(add(add(add(add(add(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta3))), mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta3))), mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], sub(1.f, delta2))), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1309
  xlf = sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1314
  xlwork2 = sub(sub(mul((-xls), add(add(add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), mul(xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), mul(xlf, add(add(add(add(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6.F90:1315
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlwork2, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1318
  } else { // mp_wsm6.F90:1319
  value = mp_max(qmin, qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1323
  source = mul(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1324
  if ((source > value)) { // mp_wsm6.F90:1325
  factor = div(value, source); // mp_wsm6.F90:1326
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1327
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1328
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1329
  } // mp_wsm6.F90:1330
  value = mp_max(qmin, qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1334
  source = mul(sub(sub(sub(add(add(sub((-paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1335
  if ((source > value)) { // mp_wsm6.F90:1337
  factor = div(value, source); // mp_wsm6.F90:1338
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1339
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1340
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1341
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1342
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1343
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1344
  } // mp_wsm6.F90:1345
  value = mp_max(S.qcrmin, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1349
  source = mul(sub(sub(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1350
  if ((source > value)) { // mp_wsm6.F90:1351
  factor = div(value, source); // mp_wsm6.F90:1352
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1353
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1354
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1355
  } // mp_wsm6.F90:1356
  value = mp_max(S.qcrmin, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1360
  source = mul((-add(add(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1361
  if ((source > value)) { // mp_wsm6.F90:1362
  factor = div(value, source); // mp_wsm6.F90:1363
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1364
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1365
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1366
  } // mp_wsm6.F90:1367
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = (-add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1369
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1371
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1372
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(sub(sub(add(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1374
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(sub(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1377
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(add(add(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1379
  xlf = sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1381
  xlwork2 = sub(mul((-xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), mul(xlf, add(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6.F90:1382
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(xlwork2, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1384
  } // mp_wsm6.F90:1385
   } // mp_wsm6.F90:1386
   } // mp_wsm6.F90:1387
  hsub = xls; // mp_wsm6.F90:1392
  hvap = xlv0; // mp_wsm6.F90:1393
  cvap = cpv; // mp_wsm6.F90:1394
  ttp = add(t0c, 0.01f); // mp_wsm6.F90:1395
  dldt = sub(cvap, cliq); // mp_wsm6.F90:1396
  xa = div((-dldt), rv); // mp_wsm6.F90:1397
  xb = add(xa, div(hvap, mul(rv, ttp))); // mp_wsm6.F90:1398
  dldti = sub(cvap, cice); // mp_wsm6.F90:1399
  xai = div((-dldti), rv); // mp_wsm6.F90:1400
  xbi = add(xai, div(hsub, mul(rv, ttp))); // mp_wsm6.F90:1401
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1402
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1403
  tr = div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1404
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xa))), mp_exp(mul(xb, sub(1.f, tr)))); // mp_wsm6.F90:1405
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1406
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1407
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:1408
  tr = div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1409
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < ttp)) { // mp_wsm6.F90:1410
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xai))), mp_exp(mul(xbi, sub(1.f, tr)))); // mp_wsm6.F90:1411
  } else { // mp_wsm6.F90:1412
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(psat, mp_exp(mul(mp_log(tr), xa))), mp_exp(mul(xb, sub(1.f, tr)))); // mp_wsm6.F90:1413
  } // mp_wsm6.F90:1414
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1415
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1416
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:1417
   } // mp_wsm6.F90:1418
   } // mp_wsm6.F90:1419
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1426
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1427
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = conden(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1428
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1429
  pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(div(mp_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.f), dtcld), mp_max(div(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], dtcld), 0.f)); // mp_wsm6.F90:1430
  if (((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 0.f))) { pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(mp_max(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], (-qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), dtcld); } // mp_wsm6.F90:1431
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = sub(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1433
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)), 0.f); // mp_wsm6.F90:1434
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mul(div(mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1435
   } // mp_wsm6.F90:1436
   } // mp_wsm6.F90:1437
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1443
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1444
  if ((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= qmin)) { qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1445
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= qmin)) { qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1446
   } // mp_wsm6.F90:1447
   } // mp_wsm6.F90:1448
   } // mp_wsm6.F90:1449
  if (((rainprod2d != nullptr) && (evapprod2d != nullptr))) { // mp_wsm6.F90:1451
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1452
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1453
  rainprod2d[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = add(add(add(add(add(add(add(add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1454
  evapprod2d[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = (-add(add(add(add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1456
   } // mp_wsm6.F90:1457
   } // mp_wsm6.F90:1458
  } // mp_wsm6.F90:1459
  // CCPP status text // mp_wsm6.F90:1465
  S.last_message = "mp_wsm6_run OK"; // mp_wsm6.F90:1465
  errflg = 0; // mp_wsm6.F90:1466
}

// MPAS-A v8.4.2 mp_wsm6.F90:1471-1494
__device__ float rgmma(Wsm6State &S, float x) {
int i;
float euler = 0.577215664901532f;
float y;
float result;
  if ((x == 1.f)) { // mp_wsm6.F90:1483
  result = 0.f; // mp_wsm6.F90:1484
  } else { // mp_wsm6.F90:1485
  result = mul(x, mp_exp(mul(euler, x))); // mp_wsm6.F90:1486
  for (i=1; i<=10000; i+=1) { // mp_wsm6.F90:1487
  y = (float)(i); // mp_wsm6.F90:1488
  result = mul(mul(result, add(1.000f, div(x, y))), mp_exp(div((-x), y))); // mp_wsm6.F90:1489
   } // mp_wsm6.F90:1490
  result = div(1.f, result); // mp_wsm6.F90:1491
  } // mp_wsm6.F90:1492
return result;
}

// MPAS-A v8.4.2 mp_wsm6.F90:1497-1522
__device__ float fpvs(Wsm6State &S, float t, int ice, float rd, float rv, float cvap, float cliq, float cice, float hvap, float hsub, float psat, float t0c) {
float tr;
float ttp;
float dldt;
float dldti;
float xa;
float xb;
float xai;
float xbi;
float result;
  ttp = add(t0c, 0.01f); // mp_wsm6.F90:1508
  dldt = sub(cvap, cliq); // mp_wsm6.F90:1509
  xa = div((-dldt), rv); // mp_wsm6.F90:1510
  xb = add(xa, div(hvap, mul(rv, ttp))); // mp_wsm6.F90:1511
  dldti = sub(cvap, cice); // mp_wsm6.F90:1512
  xai = div((-dldti), rv); // mp_wsm6.F90:1513
  xbi = add(xai, div(hsub, mul(rv, ttp))); // mp_wsm6.F90:1514
  tr = div(ttp, t); // mp_wsm6.F90:1515
  if (((t < ttp) && (ice == 1))) { // mp_wsm6.F90:1516
  result = mul(mul(psat, mp_pow(tr, xai)), mp_exp(mul(xbi, sub(1.f, tr)))); // mp_wsm6.F90:1517
  } else { // mp_wsm6.F90:1518
  result = mul(mul(psat, mp_pow(tr, xa)), mp_exp(mul(xb, sub(1.f, tr)))); // mp_wsm6.F90:1519
  } // mp_wsm6.F90:1520
return result;
}

// MPAS-A v8.4.2 mp_wsm6.F90:1525-1602
__device__ void slope_wsm6(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte) {
int i;
int k;
float t0c = 273.15f;
float x;
float y;
float z;
float supcol;
#ifdef __CUDA_ARCH__
float *n0sfac=S.workspace->slope_wsm6_n0sfac;
#else
float n0sfac[WSM6_KMAX + 3];
#endif
  auto lamdar = [&](float x, float y) { return mp_sqrt(mp_sqrt(div(S.pidn0r, mul(x, y)))); }; // mp_wsm6.F90:1549
  auto lamdas = [&](float x, float y, float z) { return mp_sqrt(mp_sqrt(div(mul(S.pidn0s, z), mul(x, y)))); }; // mp_wsm6.F90:1550
  auto lamdag = [&](float x, float y) { return mp_sqrt(mp_sqrt(div(S.pidn0g, mul(x, y)))); }; // mp_wsm6.F90:1551
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1553
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1554
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1555
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mp_exp(mul(S.alpha, supcol)), div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:1559
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1560
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopermax; // mp_wsm6.F90:1561
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloperbmax; // mp_wsm6.F90:1562
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloper2max; // mp_wsm6.F90:1563
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloper3max; // mp_wsm6.F90:1564
  } else { // mp_wsm6.F90:1565
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(1.f, lamdar(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1566
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvtr); // mp_wsm6.F90:1567
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1568
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1569
  } // mp_wsm6.F90:1570
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1571
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopesmax; // mp_wsm6.F90:1572
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopesbmax; // mp_wsm6.F90:1573
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopes2max; // mp_wsm6.F90:1574
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopes3max; // mp_wsm6.F90:1575
  } else { // mp_wsm6.F90:1576
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(1.f, lamdas(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1577
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvts); // mp_wsm6.F90:1578
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1579
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1580
  } // mp_wsm6.F90:1581
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1582
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopegmax; // mp_wsm6.F90:1583
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopegbmax; // mp_wsm6.F90:1584
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopeg2max; // mp_wsm6.F90:1585
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopeg3max; // mp_wsm6.F90:1586
  } else { // mp_wsm6.F90:1587
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = div(1.f, lamdag(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1588
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvtg); // mp_wsm6.F90:1589
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1590
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1591
  } // mp_wsm6.F90:1592
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1593
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1594
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = mul(mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1595
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1596
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1597
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1598
   } // mp_wsm6.F90:1599
   } // mp_wsm6.F90:1600
}

// MPAS-A v8.4.2 mp_wsm6.F90:1605-1648
__device__ void slope_rain(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte) {
int i;
int k;
float t0c = 273.15f;
float x;
float y;
#ifdef __CUDA_ARCH__
float *n0sfac=S.workspace->slope_rain_n0sfac;
#else
float n0sfac[WSM6_KMAX + 3];
#endif
  auto lamdar = [&](float x, float y) { return mp_sqrt(mp_sqrt(div(S.pidn0r, mul(x, y)))); }; // mp_wsm6.F90:1628
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1630
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1631
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1632
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopermax; // mp_wsm6.F90:1633
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloperbmax; // mp_wsm6.F90:1634
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloper2max; // mp_wsm6.F90:1635
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloper3max; // mp_wsm6.F90:1636
  } else { // mp_wsm6.F90:1637
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(1.f, lamdar(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1638
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvtr); // mp_wsm6.F90:1639
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1640
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1641
  } // mp_wsm6.F90:1642
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1643
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1644
   } // mp_wsm6.F90:1645
   } // mp_wsm6.F90:1646
}

// MPAS-A v8.4.2 mp_wsm6.F90:1651-1699
__device__ void slope_snow(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte) {
int i;
int k;
float t0c = 273.15f;
float x;
float y;
float z;
float supcol;
#ifdef __CUDA_ARCH__
float *n0sfac=S.workspace->slope_snow_n0sfac;
#else
float n0sfac[WSM6_KMAX + 3];
#endif
  auto lamdas = [&](float x, float y, float z) { return mp_sqrt(mp_sqrt(div(mul(S.pidn0s, z), mul(x, y)))); }; // mp_wsm6.F90:1674
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1676
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1677
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1678
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mp_exp(mul(S.alpha, supcol)), div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:1682
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1683
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopesmax; // mp_wsm6.F90:1684
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopesbmax; // mp_wsm6.F90:1685
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopes2max; // mp_wsm6.F90:1686
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopes3max; // mp_wsm6.F90:1687
  } else { // mp_wsm6.F90:1688
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(1.f, lamdas(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1689
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvts); // mp_wsm6.F90:1690
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1691
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1692
  } // mp_wsm6.F90:1693
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1694
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1695
   } // mp_wsm6.F90:1696
   } // mp_wsm6.F90:1697
}

// MPAS-A v8.4.2 mp_wsm6.F90:1702-1748
__device__ void slope_graup(Wsm6State &S, const float *qrs, const float *den, const float *denfac, const float *t, float *rslope, float *rslopeb, float *rslope2, float *rslope3, float *vt, int its, int ite, int kts, int kte) {
int i;
int k;
float t0c = 273.15f;
float x;
float y;
#ifdef __CUDA_ARCH__
float *n0sfac=S.workspace->slope_graup_n0sfac;
#else
float n0sfac[WSM6_KMAX + 3];
#endif
  auto lamdag = [&](float x, float y) { return mp_sqrt(mp_sqrt(div(S.pidn0g, mul(x, y)))); }; // mp_wsm6.F90:1725
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1727
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1728
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1732
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopegmax; // mp_wsm6.F90:1733
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopegbmax; // mp_wsm6.F90:1734
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopeg2max; // mp_wsm6.F90:1735
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopeg3max; // mp_wsm6.F90:1736
  } else { // mp_wsm6.F90:1737
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = div(1.f, lamdag(qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1738
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvtg); // mp_wsm6.F90:1739
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1740
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1741
  } // mp_wsm6.F90:1742
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mul(mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1743
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1744
   } // mp_wsm6.F90:1745
   } // mp_wsm6.F90:1746
}

// MPAS-A v8.4.2 mp_wsm6.F90:1751-1994
__device__ void nislfv_rain_plm(Wsm6State &S, int im, int km, const float *denl, const float *denfacl, const float *tkl, const float *dzl, float *wwl, float *rql, float *precip, float dt, int id, int iter) {
int i;
int k;
int n;
int m;
int kk;
int kb;
int kt;
float tl;
float tl2;
float qql;
float dql;
float qqd;
float th;
float th2;
float qqh;
float dqh;
float zsum;
float qsum;
float dim;
float dip;
float c1;
float con1;
float fa1;
float fa2;
float allold;
float allnew;
float zz;
float dzamin;
float cflmax;
float decfl;
// Native km/km+1 arrays: mp_wsm6.F90:1789-1793,2036-2041.
#ifdef __CUDA_ARCH__
float *dz=S.workspace->nislfv_rain_plm_dz;
#else
float dz[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *ww=S.workspace->nislfv_rain_plm_ww;
#else
float ww[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qq=S.workspace->nislfv_rain_plm_qq;
#else
float qq[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wd=S.workspace->nislfv_rain_plm_wd;
#else
float wd[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wa=S.workspace->nislfv_rain_plm_wa;
#else
float wa[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *was=S.workspace->nislfv_rain_plm_was;
#else
float was[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *den=S.workspace->nislfv_rain_plm_den;
#else
float den[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denfac=S.workspace->nislfv_rain_plm_denfac;
#else
float denfac[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tk=S.workspace->nislfv_rain_plm_tk;
#else
float tk[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qn=S.workspace->nislfv_rain_plm_qn;
#else
float qn[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qr=S.workspace->nislfv_rain_plm_qr;
#else
float qr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp=S.workspace->nislfv_rain_plm_tmp;
#else
float tmp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp1=S.workspace->nislfv_rain_plm_tmp1;
#else
float tmp1[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp2=S.workspace->nislfv_rain_plm_tmp2;
#else
float tmp2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp3=S.workspace->nislfv_rain_plm_tmp3;
#else
float tmp3[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wi=S.workspace->nislfv_rain_plm_wi;
#else
float wi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *zi=S.workspace->nislfv_rain_plm_zi;
#else
float zi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *za=S.workspace->nislfv_rain_plm_za;
#else
float za[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *dza=S.workspace->nislfv_rain_plm_dza;
#else
float dza[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qa=S.workspace->nislfv_rain_plm_qa;
#else
float qa[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qmi=S.workspace->nislfv_rain_plm_qmi;
#else
float qmi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qpi=S.workspace->nislfv_rain_plm_qpi;
#else
float qpi[WSM6_KMAX + 3];
#endif
  for(int slice0=1;slice0<=im;++slice0) precip[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1798
  for (i=1; i<=im; i+=1) { // mp_wsm6.F90:1800
  for(int slice0=1;slice0<=km;++slice0) dz[((slice0)-(1))*(1)] = dzl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1801
  for(int slice0=1;slice0<=km;++slice0) qq[((slice0)-(1))*(1)] = rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1802
  for(int slice0=1;slice0<=km;++slice0) ww[((slice0)-(1))*(1)] = wwl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1803
  for(int slice0=1;slice0<=km;++slice0) den[((slice0)-(1))*(1)] = denl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1804
  for(int slice0=1;slice0<=km;++slice0) denfac[((slice0)-(1))*(1)] = denfacl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1805
  for(int slice0=1;slice0<=km;++slice0) tk[((slice0)-(1))*(1)] = tkl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1806
  allold = 0.0f; // mp_wsm6.F90:1808
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1809
  allold = add(allold, qq[((k)-(1))*(1)]); // mp_wsm6.F90:1810
   } // mp_wsm6.F90:1811
  if ((allold <= 0.0f)) { // mp_wsm6.F90:1812
  goto i_loop_cycle; // mp_wsm6.F90:1813
  } // mp_wsm6.F90:1814
  zi[((1)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1817
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1818
  zi[(((k + 1))-(1))*(1)] = add(zi[((k)-(1))*(1)], dz[((k)-(1))*(1)]); // mp_wsm6.F90:1819
   } // mp_wsm6.F90:1820
  for(int slice0=1;slice0<=km;++slice0) wd[((slice0)-(1))*(1)] = ww[((slice0)-(1))*(1)]; // mp_wsm6.F90:1823
  n = 1; // mp_wsm6.F90:1824
  label100:; // mp_wsm6.F90:1825
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:1828
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:1829
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1830
  wi[((k)-(1))*(1)] = div(add(mul(ww[((k)-(1))*(1)], dz[(((k - 1))-(1))*(1)]), mul(ww[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])), add(dz[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])); // mp_wsm6.F90:1831
   } // mp_wsm6.F90:1832
  fa1 = div(9.f, 16.f); // mp_wsm6.F90:1834
  fa2 = div(1.f, 16.f); // mp_wsm6.F90:1835
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:1836
  wi[((2)-(1))*(1)] = mul(0.5f, add(ww[((2)-(1))*(1)], ww[((1)-(1))*(1)])); // mp_wsm6.F90:1837
  for (k=3; k<=(km - 1); k+=1) { // mp_wsm6.F90:1838
  wi[((k)-(1))*(1)] = sub(mul(fa1, add(ww[((k)-(1))*(1)], ww[(((k - 1))-(1))*(1)])), mul(fa2, add(ww[(((k + 1))-(1))*(1)], ww[(((k - 2))-(1))*(1)]))); // mp_wsm6.F90:1839
   } // mp_wsm6.F90:1840
  wi[((km)-(1))*(1)] = mul(0.5f, add(ww[((km)-(1))*(1)], ww[(((km - 1))-(1))*(1)])); // mp_wsm6.F90:1841
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:1842
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1845
  if ((ww[((k)-(1))*(1)] == 0.0f)) { wi[((k)-(1))*(1)] = ww[(((k - 1))-(1))*(1)]; } // mp_wsm6.F90:1846
   } // mp_wsm6.F90:1847
  con1 = 0.05f; // mp_wsm6.F90:1850
  for (k=km; k>=1; k+=(-1)) { // mp_wsm6.F90:1851
  decfl = div(mul(sub(wi[(((k + 1))-(1))*(1)], wi[((k)-(1))*(1)]), dt), dz[((k)-(1))*(1)]); // mp_wsm6.F90:1852
  if ((decfl > con1)) { // mp_wsm6.F90:1853
  wi[((k)-(1))*(1)] = sub(wi[(((k + 1))-(1))*(1)], div(mul(con1, dz[((k)-(1))*(1)]), dt)); // mp_wsm6.F90:1854
  } // mp_wsm6.F90:1855
   } // mp_wsm6.F90:1856
  for (k=1; k<=(km + 1); k+=1) { // mp_wsm6.F90:1858
  za[((k)-(1))*(1)] = sub(zi[((k)-(1))*(1)], mul(wi[((k)-(1))*(1)], dt)); // mp_wsm6.F90:1859
   } // mp_wsm6.F90:1860
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1862
  dza[((k)-(1))*(1)] = sub(za[(((k + 1))-(1))*(1)], za[((k)-(1))*(1)]); // mp_wsm6.F90:1863
   } // mp_wsm6.F90:1864
  dza[(((km + 1))-(1))*(1)] = sub(zi[(((km + 1))-(1))*(1)], za[(((km + 1))-(1))*(1)]); // mp_wsm6.F90:1865
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1868
  qa[((k)-(1))*(1)] = div(mul(qq[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:1869
  qr[((k)-(1))*(1)] = div(qa[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:1870
   } // mp_wsm6.F90:1871
  qa[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:1872
  if ((n <= iter)) { // mp_wsm6.F90:1877
  slope_rain(S, qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, 1, 1, 1, km); // mp_wsm6.F90:1878
  if ((n >= 2)) { for(int slice0=1;slice0<=km;++slice0) wa[((slice0)-(1))*(1)] = mul(0.5f, add(wa[((slice0)-(1))*(1)], was[((slice0)-(1))*(1)])); } // mp_wsm6.F90:1879
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1880
  ww[((k)-(1))*(1)] = mul(0.5f, add(wd[((k)-(1))*(1)], wa[((k)-(1))*(1)])); // mp_wsm6.F90:1885
   } // mp_wsm6.F90:1886
  for(int slice0=1;slice0<=km;++slice0) was[((slice0)-(1))*(1)] = wa[((slice0)-(1))*(1)]; // mp_wsm6.F90:1887
  n = (n + 1); // mp_wsm6.F90:1888
  goto label100; // mp_wsm6.F90:1889
  } // mp_wsm6.F90:1890
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1893
  dip = div(sub(qa[(((k + 1))-(1))*(1)], qa[((k)-(1))*(1)]), add(dza[(((k + 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1894
  dim = div(sub(qa[((k)-(1))*(1)], qa[(((k - 1))-(1))*(1)]), add(dza[(((k - 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1895
  if ((mul(dip, dim) <= 0.0f)) { // mp_wsm6.F90:1896
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1897
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1898
  } else { // mp_wsm6.F90:1899
  qpi[((k)-(1))*(1)] = add(qa[((k)-(1))*(1)], mul(mul(0.5f, add(dip, dim)), dza[((k)-(1))*(1)])); // mp_wsm6.F90:1900
  qmi[((k)-(1))*(1)] = sub(mul(2.0f, qa[((k)-(1))*(1)]), qpi[((k)-(1))*(1)]); // mp_wsm6.F90:1901
  if (((qpi[((k)-(1))*(1)] < 0.0f) || (qmi[((k)-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:1902
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1903
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1904
  } // mp_wsm6.F90:1905
  } // mp_wsm6.F90:1906
   } // mp_wsm6.F90:1907
  qpi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:1908
  qmi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:1909
  qmi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:1910
  qpi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:1911
  for(int slice0=1;slice0<=km;++slice0) qn[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1914
  kb = 1; // mp_wsm6.F90:1915
  kt = 1; // mp_wsm6.F90:1916
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1917
  kb = mp_max((kb - 1), 1); // mp_wsm6.F90:1918
  kt = mp_max((kt - 1), 1); // mp_wsm6.F90:1919
  if ((zi[((k)-(1))*(1)] >= za[(((km + 1))-(1))*(1)])) { // mp_wsm6.F90:1921
  goto intp_exit; // mp_wsm6.F90:1922
  } else { // mp_wsm6.F90:1923
  for (kk=kb; kk<=km; kk+=1) { // mp_wsm6.F90:1924
  if ((zi[((k)-(1))*(1)] <= za[(((kk + 1))-(1))*(1)])) { // mp_wsm6.F90:1925
  kb = kk; // mp_wsm6.F90:1926
  goto find_kb_exit; // mp_wsm6.F90:1927
  } else { // mp_wsm6.F90:1928
  goto find_kb_cycle; // mp_wsm6.F90:1929
  } // mp_wsm6.F90:1930
  find_kb_cycle:; } find_kb_exit:; // mp_wsm6.F90:1931
  for (kk=kt; kk<=km; kk+=1) { // mp_wsm6.F90:1932
  if ((zi[(((k + 1))-(1))*(1)] <= za[((kk)-(1))*(1)])) { // mp_wsm6.F90:1933
  kt = kk; // mp_wsm6.F90:1934
  goto find_kt_exit; // mp_wsm6.F90:1935
  } else { // mp_wsm6.F90:1936
  goto find_kt_cycle; // mp_wsm6.F90:1937
  } // mp_wsm6.F90:1938
  find_kt_cycle:; } find_kt_exit:; // mp_wsm6.F90:1939
  kt = (kt - 1); // mp_wsm6.F90:1940
  if ((kt == kb)) { // mp_wsm6.F90:1942
  tl = div(sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1943
  th = div(sub(zi[(((k + 1))-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1944
  tl2 = mul(tl, tl); // mp_wsm6.F90:1945
  th2 = mul(th, th); // mp_wsm6.F90:1946
  qqd = mul(0.5f, sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:1947
  qqh = add(mul(qqd, th2), mul(qmi[((kb)-(1))*(1)], th)); // mp_wsm6.F90:1948
  qql = add(mul(qqd, tl2), mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:1949
  qn[((k)-(1))*(1)] = div(sub(qqh, qql), sub(th, tl)); // mp_wsm6.F90:1950
  } else if ((kt > kb)) { // mp_wsm6.F90:1951
  tl = div(sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1952
  tl2 = mul(tl, tl); // mp_wsm6.F90:1953
  qqd = mul(0.5f, sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:1954
  qql = add(mul(qqd, tl2), mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:1955
  dql = sub(qa[((kb)-(1))*(1)], qql); // mp_wsm6.F90:1956
  zsum = mul(sub(1.f, tl), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1957
  qsum = mul(dql, dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1958
  if (((kt - kb) > 1)) { // mp_wsm6.F90:1959
  for (m=(kb + 1); m<=(kt - 1); m+=1) { // mp_wsm6.F90:1960
  zsum = add(zsum, dza[((m)-(1))*(1)]); // mp_wsm6.F90:1961
  qsum = add(qsum, mul(qa[((m)-(1))*(1)], dza[((m)-(1))*(1)])); // mp_wsm6.F90:1962
   } // mp_wsm6.F90:1963
  } // mp_wsm6.F90:1964
  th = div(sub(zi[(((k + 1))-(1))*(1)], za[((kt)-(1))*(1)]), dza[((kt)-(1))*(1)]); // mp_wsm6.F90:1965
  th2 = mul(th, th); // mp_wsm6.F90:1966
  qqd = mul(0.5f, sub(qpi[((kt)-(1))*(1)], qmi[((kt)-(1))*(1)])); // mp_wsm6.F90:1967
  dqh = add(mul(qqd, th2), mul(qmi[((kt)-(1))*(1)], th)); // mp_wsm6.F90:1968
  zsum = add(zsum, mul(th, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:1969
  qsum = add(qsum, mul(dqh, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:1970
  qn[((k)-(1))*(1)] = div(qsum, zsum); // mp_wsm6.F90:1971
  } // mp_wsm6.F90:1972
  goto intp_cycle; // mp_wsm6.F90:1973
  } // mp_wsm6.F90:1974
  intp_cycle:; } intp_exit:; // mp_wsm6.F90:1976
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1979
  if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:1980
  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1981
  goto sum_precip_cycle; // mp_wsm6.F90:1982
  } else if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] >= 0.0f))) { // mp_wsm6.F90:1983
  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:1984
  goto sum_precip_exit; // mp_wsm6.F90:1985
  } // mp_wsm6.F90:1986
  goto sum_precip_exit; // mp_wsm6.F90:1987
  sum_precip_cycle:; } sum_precip_exit:; // mp_wsm6.F90:1988
  for(int slice0=1;slice0<=km;++slice0) rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:1991
  i_loop_cycle:; } i_loop_exit:; // mp_wsm6.F90:1992
}

// MPAS-A v8.4.2 mp_wsm6.F90:1997-2272
__device__ void nislfv_rain_plm6(Wsm6State &S, int im, int km, const float *denl, const float *denfacl, const float *tkl, const float *dzl, float *wwl, float *rql, float *rql2, float *precip1, float *precip2, float dt, int id, int iter) {
int i;
int ist;
int k;
int n;
int m;
int kk;
int kb;
int kt;
float tl;
float tl2;
float qql;
float dql;
float qqd;
float th;
float th2;
float qqh;
float dqh;
float zsum;
float qsum;
float dim;
float dip;
float c1;
float con1;
float fa1;
float fa2;
float allold;
float allnew;
float zz;
float dzamin;
float cflmax;
float decfl;
// Native km/km+1 arrays: mp_wsm6.F90:1789-1793,2036-2041.
#ifdef __CUDA_ARCH__
float *dz=S.workspace->nislfv_rain_plm6_dz;
#else
float dz[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *ww=S.workspace->nislfv_rain_plm6_ww;
#else
float ww[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qq=S.workspace->nislfv_rain_plm6_qq;
#else
float qq[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qq2=S.workspace->nislfv_rain_plm6_qq2;
#else
float qq2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wd=S.workspace->nislfv_rain_plm6_wd;
#else
float wd[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wa=S.workspace->nislfv_rain_plm6_wa;
#else
float wa[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wa2=S.workspace->nislfv_rain_plm6_wa2;
#else
float wa2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *was=S.workspace->nislfv_rain_plm6_was;
#else
float was[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *den=S.workspace->nislfv_rain_plm6_den;
#else
float den[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *denfac=S.workspace->nislfv_rain_plm6_denfac;
#else
float denfac[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tk=S.workspace->nislfv_rain_plm6_tk;
#else
float tk[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qn=S.workspace->nislfv_rain_plm6_qn;
#else
float qn[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qr=S.workspace->nislfv_rain_plm6_qr;
#else
float qr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qr2=S.workspace->nislfv_rain_plm6_qr2;
#else
float qr2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp=S.workspace->nislfv_rain_plm6_tmp;
#else
float tmp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp1=S.workspace->nislfv_rain_plm6_tmp1;
#else
float tmp1[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp2=S.workspace->nislfv_rain_plm6_tmp2;
#else
float tmp2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *tmp3=S.workspace->nislfv_rain_plm6_tmp3;
#else
float tmp3[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *wi=S.workspace->nislfv_rain_plm6_wi;
#else
float wi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *zi=S.workspace->nislfv_rain_plm6_zi;
#else
float zi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *za=S.workspace->nislfv_rain_plm6_za;
#else
float za[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *dza=S.workspace->nislfv_rain_plm6_dza;
#else
float dza[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qa=S.workspace->nislfv_rain_plm6_qa;
#else
float qa[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qa2=S.workspace->nislfv_rain_plm6_qa2;
#else
float qa2[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qmi=S.workspace->nislfv_rain_plm6_qmi;
#else
float qmi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qpi=S.workspace->nislfv_rain_plm6_qpi;
#else
float qpi[WSM6_KMAX + 3];
#endif
float precip[1]; // mp_wsm6.F90:2041; im=1 per column launch
  for(int slice0=1;slice0<=im;++slice0) precip[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2045
  for(int slice0=1;slice0<=im;++slice0) precip1[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2046
  for(int slice0=1;slice0<=im;++slice0) precip2[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2047
  for (i=1; i<=im; i+=1) { // mp_wsm6.F90:2049
  for(int slice0=1;slice0<=km;++slice0) dz[((slice0)-(1))*(1)] = dzl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2050
  for(int slice0=1;slice0<=km;++slice0) qq[((slice0)-(1))*(1)] = rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2051
  for(int slice0=1;slice0<=km;++slice0) qq2[((slice0)-(1))*(1)] = rql2[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2052
  for(int slice0=1;slice0<=km;++slice0) ww[((slice0)-(1))*(1)] = wwl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2053
  for(int slice0=1;slice0<=km;++slice0) den[((slice0)-(1))*(1)] = denl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2054
  for(int slice0=1;slice0<=km;++slice0) denfac[((slice0)-(1))*(1)] = denfacl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2055
  for(int slice0=1;slice0<=km;++slice0) tk[((slice0)-(1))*(1)] = tkl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2056
  allold = 0.0f; // mp_wsm6.F90:2058
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2059
  allold = add(add(allold, qq[((k)-(1))*(1)]), qq2[((k)-(1))*(1)]); // mp_wsm6.F90:2060
   } // mp_wsm6.F90:2061
  if ((allold <= 0.0f)) { // mp_wsm6.F90:2062
  goto i_loop_cycle; // mp_wsm6.F90:2063
  } // mp_wsm6.F90:2064
  zi[((1)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2067
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2068
  zi[(((k + 1))-(1))*(1)] = add(zi[((k)-(1))*(1)], dz[((k)-(1))*(1)]); // mp_wsm6.F90:2069
   } // mp_wsm6.F90:2070
  for(int slice0=1;slice0<=km;++slice0) wd[((slice0)-(1))*(1)] = ww[((slice0)-(1))*(1)]; // mp_wsm6.F90:2073
  n = 1; // mp_wsm6.F90:2074
  label100:; // mp_wsm6.F90:2075
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:2078
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:2079
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2080
  wi[((k)-(1))*(1)] = div(add(mul(ww[((k)-(1))*(1)], dz[(((k - 1))-(1))*(1)]), mul(ww[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])), add(dz[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])); // mp_wsm6.F90:2081
   } // mp_wsm6.F90:2082
  fa1 = div(9.f, 16.f); // mp_wsm6.F90:2084
  fa2 = div(1.f, 16.f); // mp_wsm6.F90:2085
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:2086
  wi[((2)-(1))*(1)] = mul(0.5f, add(ww[((2)-(1))*(1)], ww[((1)-(1))*(1)])); // mp_wsm6.F90:2087
  for (k=3; k<=(km - 1); k+=1) { // mp_wsm6.F90:2088
  wi[((k)-(1))*(1)] = sub(mul(fa1, add(ww[((k)-(1))*(1)], ww[(((k - 1))-(1))*(1)])), mul(fa2, add(ww[(((k + 1))-(1))*(1)], ww[(((k - 2))-(1))*(1)]))); // mp_wsm6.F90:2089
   } // mp_wsm6.F90:2090
  wi[((km)-(1))*(1)] = mul(0.5f, add(ww[((km)-(1))*(1)], ww[(((km - 1))-(1))*(1)])); // mp_wsm6.F90:2091
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:2092
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2095
  if ((ww[((k)-(1))*(1)] == 0.0f)) { wi[((k)-(1))*(1)] = ww[(((k - 1))-(1))*(1)]; } // mp_wsm6.F90:2096
   } // mp_wsm6.F90:2097
  con1 = 0.05f; // mp_wsm6.F90:2100
  for (k=km; k>=1; k+=(-1)) { // mp_wsm6.F90:2101
  decfl = div(mul(sub(wi[(((k + 1))-(1))*(1)], wi[((k)-(1))*(1)]), dt), dz[((k)-(1))*(1)]); // mp_wsm6.F90:2102
  if ((decfl > con1)) { // mp_wsm6.F90:2103
  wi[((k)-(1))*(1)] = sub(wi[(((k + 1))-(1))*(1)], div(mul(con1, dz[((k)-(1))*(1)]), dt)); // mp_wsm6.F90:2104
  } // mp_wsm6.F90:2105
   } // mp_wsm6.F90:2106
  for (k=1; k<=(km + 1); k+=1) { // mp_wsm6.F90:2108
  za[((k)-(1))*(1)] = sub(zi[((k)-(1))*(1)], mul(wi[((k)-(1))*(1)], dt)); // mp_wsm6.F90:2109
   } // mp_wsm6.F90:2110
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2112
  dza[((k)-(1))*(1)] = sub(za[(((k + 1))-(1))*(1)], za[((k)-(1))*(1)]); // mp_wsm6.F90:2113
   } // mp_wsm6.F90:2114
  dza[(((km + 1))-(1))*(1)] = sub(zi[(((km + 1))-(1))*(1)], za[(((km + 1))-(1))*(1)]); // mp_wsm6.F90:2115
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2118
  qa[((k)-(1))*(1)] = div(mul(qq[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:2119
  qa2[((k)-(1))*(1)] = div(mul(qq2[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:2120
  qr[((k)-(1))*(1)] = div(qa[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:2121
  qr2[((k)-(1))*(1)] = div(qa2[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:2122
   } // mp_wsm6.F90:2123
  qa[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:2124
  qa2[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:2125
  if ((n <= iter)) { // mp_wsm6.F90:2130
  slope_snow(S, qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, 1, 1, 1, km); // mp_wsm6.F90:2131
  slope_graup(S, qr2, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa2, 1, 1, 1, km); // mp_wsm6.F90:2132
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2133
  tmp[((k)-(1))*(1)] = mp_max(add(qr[((k)-(1))*(1)], qr2[((k)-(1))*(1)]), 1.e-15f); // mp_wsm6.F90:2134
  if ((tmp[((k)-(1))*(1)] > 1.e-15f)) { // mp_wsm6.F90:2135
  wa[((k)-(1))*(1)] = div(add(mul(wa[((k)-(1))*(1)], qr[((k)-(1))*(1)]), mul(wa2[((k)-(1))*(1)], qr2[((k)-(1))*(1)])), tmp[((k)-(1))*(1)]); // mp_wsm6.F90:2136
  } else { // mp_wsm6.F90:2137
  wa[((k)-(1))*(1)] = 0.f; // mp_wsm6.F90:2138
  } // mp_wsm6.F90:2139
   } // mp_wsm6.F90:2140
  if ((n >= 2)) { for(int slice0=1;slice0<=km;++slice0) wa[((slice0)-(1))*(1)] = mul(0.5f, add(wa[((slice0)-(1))*(1)], was[((slice0)-(1))*(1)])); } // mp_wsm6.F90:2141
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2142
  ww[((k)-(1))*(1)] = mul(0.5f, add(wd[((k)-(1))*(1)], wa[((k)-(1))*(1)])); // mp_wsm6.F90:2148
   } // mp_wsm6.F90:2149
  for(int slice0=1;slice0<=km;++slice0) was[((slice0)-(1))*(1)] = wa[((slice0)-(1))*(1)]; // mp_wsm6.F90:2150
  n = (n + 1); // mp_wsm6.F90:2151
  goto label100; // mp_wsm6.F90:2152
  } // mp_wsm6.F90:2153
  for (ist=1; ist<=2; ist+=1) { // mp_wsm6.F90:2155
  if ((ist == 2)) { // mp_wsm6.F90:2156
  for(int slice0=1;slice0<=(km + 1);++slice0) qa[((slice0)-(1))*(1)] = qa2[((slice0)-(1))*(1)]; // mp_wsm6.F90:2157
  } // mp_wsm6.F90:2158
  precip[((i)-(1))*(1)] = 0.f; // mp_wsm6.F90:2160
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2163
  dip = div(sub(qa[(((k + 1))-(1))*(1)], qa[((k)-(1))*(1)]), add(dza[(((k + 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2164
  dim = div(sub(qa[((k)-(1))*(1)], qa[(((k - 1))-(1))*(1)]), add(dza[(((k - 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2165
  if ((mul(dip, dim) <= 0.0f)) { // mp_wsm6.F90:2166
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2167
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2168
  } else { // mp_wsm6.F90:2169
  qpi[((k)-(1))*(1)] = add(qa[((k)-(1))*(1)], mul(mul(0.5f, add(dip, dim)), dza[((k)-(1))*(1)])); // mp_wsm6.F90:2170
  qmi[((k)-(1))*(1)] = sub(mul(2.0f, qa[((k)-(1))*(1)]), qpi[((k)-(1))*(1)]); // mp_wsm6.F90:2171
  if (((qpi[((k)-(1))*(1)] < 0.0f) || (qmi[((k)-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:2172
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2173
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2174
  } // mp_wsm6.F90:2175
  } // mp_wsm6.F90:2176
   } // mp_wsm6.F90:2177
  qpi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:2178
  qmi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:2179
  qmi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:2180
  qpi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:2181
  for(int slice0=1;slice0<=km;++slice0) qn[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2184
  kb = 1; // mp_wsm6.F90:2185
  kt = 1; // mp_wsm6.F90:2186
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2187
  kb = mp_max((kb - 1), 1); // mp_wsm6.F90:2188
  kt = mp_max((kt - 1), 1); // mp_wsm6.F90:2189
  if ((zi[((k)-(1))*(1)] >= za[(((km + 1))-(1))*(1)])) { // mp_wsm6.F90:2191
  goto intp_exit; // mp_wsm6.F90:2192
  } else { // mp_wsm6.F90:2193
  for (kk=kb; kk<=km; kk+=1) { // mp_wsm6.F90:2194
  if ((zi[((k)-(1))*(1)] <= za[(((kk + 1))-(1))*(1)])) { // mp_wsm6.F90:2195
  kb = kk; // mp_wsm6.F90:2196
  goto find_kb_exit; // mp_wsm6.F90:2197
  } else { // mp_wsm6.F90:2198
  goto find_kb_cycle; // mp_wsm6.F90:2199
  } // mp_wsm6.F90:2200
  find_kb_cycle:; } find_kb_exit:; // mp_wsm6.F90:2201
  for (kk=kt; kk<=km; kk+=1) { // mp_wsm6.F90:2202
  if ((zi[(((k + 1))-(1))*(1)] <= za[((kk)-(1))*(1)])) { // mp_wsm6.F90:2203
  kt = kk; // mp_wsm6.F90:2204
  goto find_kt_exit; // mp_wsm6.F90:2205
  } else { // mp_wsm6.F90:2206
  goto find_kt_cycle; // mp_wsm6.F90:2207
  } // mp_wsm6.F90:2208
  find_kt_cycle:; } find_kt_exit:; // mp_wsm6.F90:2209
  kt = (kt - 1); // mp_wsm6.F90:2210
  if ((kt == kb)) { // mp_wsm6.F90:2212
  tl = div(sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2213
  th = div(sub(zi[(((k + 1))-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2214
  tl2 = mul(tl, tl); // mp_wsm6.F90:2215
  th2 = mul(th, th); // mp_wsm6.F90:2216
  qqd = mul(0.5f, sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:2217
  qqh = add(mul(qqd, th2), mul(qmi[((kb)-(1))*(1)], th)); // mp_wsm6.F90:2218
  qql = add(mul(qqd, tl2), mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:2219
  qn[((k)-(1))*(1)] = div(sub(qqh, qql), sub(th, tl)); // mp_wsm6.F90:2220
  } else if ((kt > kb)) { // mp_wsm6.F90:2221
  tl = div(sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2222
  tl2 = mul(tl, tl); // mp_wsm6.F90:2223
  qqd = mul(0.5f, sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:2224
  qql = add(mul(qqd, tl2), mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:2225
  dql = sub(qa[((kb)-(1))*(1)], qql); // mp_wsm6.F90:2226
  zsum = mul(sub(1.f, tl), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2227
  qsum = mul(dql, dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2228
  if (((kt - kb) > 1)) { // mp_wsm6.F90:2229
  for (m=(kb + 1); m<=(kt - 1); m+=1) { // mp_wsm6.F90:2230
  zsum = add(zsum, dza[((m)-(1))*(1)]); // mp_wsm6.F90:2231
  qsum = add(qsum, mul(qa[((m)-(1))*(1)], dza[((m)-(1))*(1)])); // mp_wsm6.F90:2232
   } // mp_wsm6.F90:2233
  } // mp_wsm6.F90:2234
  th = div(sub(zi[(((k + 1))-(1))*(1)], za[((kt)-(1))*(1)]), dza[((kt)-(1))*(1)]); // mp_wsm6.F90:2235
  th2 = mul(th, th); // mp_wsm6.F90:2236
  qqd = mul(0.5f, sub(qpi[((kt)-(1))*(1)], qmi[((kt)-(1))*(1)])); // mp_wsm6.F90:2237
  dqh = add(mul(qqd, th2), mul(qmi[((kt)-(1))*(1)], th)); // mp_wsm6.F90:2238
  zsum = add(zsum, mul(th, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:2239
  qsum = add(qsum, mul(dqh, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:2240
  qn[((k)-(1))*(1)] = div(qsum, zsum); // mp_wsm6.F90:2241
  } // mp_wsm6.F90:2242
  goto intp_cycle; // mp_wsm6.F90:2243
  } // mp_wsm6.F90:2244
  intp_cycle:; } intp_exit:; // mp_wsm6.F90:2246
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2249
  if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:2250
  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2251
  goto sum_precip_cycle; // mp_wsm6.F90:2252
  } else if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] >= 0.0f))) { // mp_wsm6.F90:2253
  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:2254
  goto sum_precip_exit; // mp_wsm6.F90:2255
  } // mp_wsm6.F90:2256
  goto sum_precip_exit; // mp_wsm6.F90:2257
  sum_precip_cycle:; } sum_precip_exit:; // mp_wsm6.F90:2258
  if ((ist == 1)) { // mp_wsm6.F90:2261
  for(int slice0=1;slice0<=km;++slice0) rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:2262
  precip1[((i)-(1))*(1)] = precip[((i)-(1))*(1)]; // mp_wsm6.F90:2263
  } else { // mp_wsm6.F90:2264
  for(int slice0=1;slice0<=km;++slice0) rql2[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:2265
  precip2[((i)-(1))*(1)] = precip[((i)-(1))*(1)]; // mp_wsm6.F90:2266
  } // mp_wsm6.F90:2267
  ist_loop_cycle:; } ist_loop_exit:; // mp_wsm6.F90:2268
  i_loop_cycle:; } i_loop_exit:; // mp_wsm6.F90:2270
}

// MPAS-A v8.4.2 mp_wsm6.F90:2275-2444
__device__ void refl10cm_wsm6(Wsm6State &S, const float *qv1d, const float *qr1d, const float *qs1d, const float *qg1d, const float *t1d, const float *p1d, float *dbz, int kts, int kte) {
bool melti;
// Native level arrays: mp_wsm6.F90:2284-2299.
#ifdef __CUDA_ARCH__
bool *l_qr=S.workspace->refl10cm_wsm6_l_qr;
#else
bool l_qr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
bool *l_qs=S.workspace->refl10cm_wsm6_l_qs;
#else
bool l_qs[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
bool *l_qg=S.workspace->refl10cm_wsm6_l_qg;
#else
bool l_qg[WSM6_KMAX + 3];
#endif
int i;
int k;
int k_0;
int kbot;
int n;
float r = 287.f;
float temp_c;
#ifdef __CUDA_ARCH__
float *temp=S.workspace->refl10cm_wsm6_temp;
#else
float temp[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *pres=S.workspace->refl10cm_wsm6_pres;
#else
float pres[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *qv=S.workspace->refl10cm_wsm6_qv;
#else
float qv[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rho=S.workspace->refl10cm_wsm6_rho;
#else
float rho[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rr=S.workspace->refl10cm_wsm6_rr;
#else
float rr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rs=S.workspace->refl10cm_wsm6_rs;
#else
float rs[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rg=S.workspace->refl10cm_wsm6_rg;
#else
float rg[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *ze_rain=S.workspace->refl10cm_wsm6_ze_rain;
#else
float ze_rain[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *ze_snow=S.workspace->refl10cm_wsm6_ze_snow;
#else
float ze_snow[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *ze_graupel=S.workspace->refl10cm_wsm6_ze_graupel;
#else
float ze_graupel[WSM6_KMAX + 3];
#endif
double fmelt_s;
double fmelt_g;
double cback;
double x;
double eta;
double f_d;
#ifdef __CUDA_ARCH__
double *ilamr=S.workspace->refl10cm_wsm6_ilamr;
#else
double ilamr[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
double *ilams=S.workspace->refl10cm_wsm6_ilams;
#else
double ilams[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
double *ilamg=S.workspace->refl10cm_wsm6_ilamg;
#else
double ilamg[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
double *n0_r=S.workspace->refl10cm_wsm6_n0_r;
#else
double n0_r[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
double *n0_s=S.workspace->refl10cm_wsm6_n0_s;
#else
double n0_s[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
double *n0_g=S.workspace->refl10cm_wsm6_n0_g;
#else
double n0_g[WSM6_KMAX + 3];
#endif
double lamr;
double lams;
double lamg;
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:2304
  dbz[((k)-(kts))*(1)] = (-35.0f); // mp_wsm6.F90:2305
   } // mp_wsm6.F90:2306
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:2311
  temp[((k)-(kts))*(1)] = t1d[((k)-(kts))*(1)]; // mp_wsm6.F90:2312
  temp_c = mp_min(sub(temp[((k)-(kts))*(1)], 273.15f), (-0.001f)); // mp_wsm6.F90:2313
  qv[((k)-(kts))*(1)] = mp_max(qv1d[((k)-(kts))*(1)], 1.e-10f); // mp_wsm6.F90:2314
  pres[((k)-(kts))*(1)] = p1d[((k)-(kts))*(1)]; // mp_wsm6.F90:2315
  rho[((k)-(kts))*(1)] = div(mul(0.622f, pres[((k)-(kts))*(1)]), mul(mul(r, temp[((k)-(kts))*(1)]), add(qv[((k)-(kts))*(1)], 0.622f))); // mp_wsm6.F90:2316
  if ((qr1d[((k)-(kts))*(1)] > 1.e-9f)) { // mp_wsm6.F90:2318
  rr[((k)-(kts))*(1)] = mul(qr1d[((k)-(kts))*(1)], rho[((k)-(kts))*(1)]); // mp_wsm6.F90:2319
  n0_r[((k)-(kts))*(1)] = S.n0r; // mp_wsm6.F90:2320
  lamr = mp_pow(div(mul(mul(S.xam_r, S.xcrg[((3)-(1))*(1)]), n0_r[((k)-(kts))*(1)]), rr[((k)-(kts))*(1)]), div(1.f, S.xcre[((1)-(1))*(1)])); // mp_wsm6.F90:2321
  ilamr[((k)-(kts))*(1)] = div(1.f, lamr); // mp_wsm6.F90:2322
  l_qr[((k)-(kts))*(1)] = true; // mp_wsm6.F90:2323
  } else { // mp_wsm6.F90:2324
  rr[((k)-(kts))*(1)] = 1.e-12f; // mp_wsm6.F90:2325
  l_qr[((k)-(kts))*(1)] = false; // mp_wsm6.F90:2326
  } // mp_wsm6.F90:2327
  if ((qs1d[((k)-(kts))*(1)] > 1.e-9f)) { // mp_wsm6.F90:2329
  rs[((k)-(kts))*(1)] = mul(qs1d[((k)-(kts))*(1)], rho[((k)-(kts))*(1)]); // mp_wsm6.F90:2330
  n0_s[((k)-(kts))*(1)] = mp_min(S.n0smax, mul(S.n0s, mp_exp(mul((-S.alpha), temp_c)))); // mp_wsm6.F90:2331
  lams = mp_pow(div(mul(mul(S.xam_s, S.xcsg[((3)-(1))*(1)]), n0_s[((k)-(kts))*(1)]), rs[((k)-(kts))*(1)]), div(1.f, S.xcse[((1)-(1))*(1)])); // mp_wsm6.F90:2332
  ilams[((k)-(kts))*(1)] = div(1.f, lams); // mp_wsm6.F90:2333
  l_qs[((k)-(kts))*(1)] = true; // mp_wsm6.F90:2334
  } else { // mp_wsm6.F90:2335
  rs[((k)-(kts))*(1)] = 1.e-12f; // mp_wsm6.F90:2336
  l_qs[((k)-(kts))*(1)] = false; // mp_wsm6.F90:2337
  } // mp_wsm6.F90:2338
  if ((qg1d[((k)-(kts))*(1)] > 1.e-9f)) { // mp_wsm6.F90:2340
  rg[((k)-(kts))*(1)] = mul(qg1d[((k)-(kts))*(1)], rho[((k)-(kts))*(1)]); // mp_wsm6.F90:2341
  n0_g[((k)-(kts))*(1)] = S.n0g; // mp_wsm6.F90:2342
  lamg = mp_pow(div(mul(mul(S.xam_g, S.xcgg[((3)-(1))*(1)]), n0_g[((k)-(kts))*(1)]), rg[((k)-(kts))*(1)]), div(1.f, S.xcge[((1)-(1))*(1)])); // mp_wsm6.F90:2343
  ilamg[((k)-(kts))*(1)] = div(1.f, lamg); // mp_wsm6.F90:2344
  l_qg[((k)-(kts))*(1)] = true; // mp_wsm6.F90:2345
  } else { // mp_wsm6.F90:2346
  rg[((k)-(kts))*(1)] = 1.e-12f; // mp_wsm6.F90:2347
  l_qg[((k)-(kts))*(1)] = false; // mp_wsm6.F90:2348
  } // mp_wsm6.F90:2349
   } // mp_wsm6.F90:2350
  melti = false; // mp_wsm6.F90:2355
  k_0 = kts; // mp_wsm6.F90:2356
  for (k=(kte - 1); k>=kts; k+=(-1)) { // mp_wsm6.F90:2357
  if ((((temp[((k)-(kts))*(1)] > 273.15f) && l_qr[((k)-(kts))*(1)]) && (l_qs[(((k + 1))-(kts))*(1)] || l_qg[(((k + 1))-(kts))*(1)]))) { // mp_wsm6.F90:2358
  k_0 = mp_max((k + 1), k_0); // mp_wsm6.F90:2360
  melti = true; // mp_wsm6.F90:2361
  goto label195; // mp_wsm6.F90:2362
  } // mp_wsm6.F90:2363
   } // mp_wsm6.F90:2364
  label195:; // mp_wsm6.F90:2365
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:2373
  ze_rain[((k)-(kts))*(1)] = 1.e-22f; // mp_wsm6.F90:2374
  ze_snow[((k)-(kts))*(1)] = 1.e-22f; // mp_wsm6.F90:2375
  ze_graupel[((k)-(kts))*(1)] = 1.e-22f; // mp_wsm6.F90:2376
  if (l_qr[((k)-(kts))*(1)]) { ze_rain[((k)-(kts))*(1)] = mul(mul(n0_r[((k)-(kts))*(1)], S.xcrg[((4)-(1))*(1)]), mp_pow(ilamr[((k)-(kts))*(1)], S.xcre[((4)-(1))*(1)])); } // mp_wsm6.F90:2377
  if (l_qs[((k)-(kts))*(1)]) { ze_snow[((k)-(kts))*(1)] = mul(mul(mul(mul(mul(mul(mul(div(0.176f, 0.93f), div(6.0f, S.pi)), div(6.0f, S.pi)), div(S.xam_s, 900.0f)), div(S.xam_s, 900.0f)), n0_s[((k)-(kts))*(1)]), S.xcsg[((4)-(1))*(1)]), mp_pow(ilams[((k)-(kts))*(1)], S.xcse[((4)-(1))*(1)])); } // mp_wsm6.F90:2378
  if (l_qg[((k)-(kts))*(1)]) { ze_graupel[((k)-(kts))*(1)] = mul(mul(mul(mul(mul(mul(mul(div(0.176f, 0.93f), div(6.0f, S.pi)), div(6.0f, S.pi)), div(S.xam_g, 900.0f)), div(S.xam_g, 900.0f)), n0_g[((k)-(kts))*(1)]), S.xcgg[((4)-(1))*(1)]), mp_pow(ilamg[((k)-(kts))*(1)], S.xcge[((4)-(1))*(1)])); } // mp_wsm6.F90:2381
   } // mp_wsm6.F90:2384
  if ((melti && (k_0 >= (kts + 1)))) { // mp_wsm6.F90:2395
  for (k=(k_0 - 1); k>=kts; k+=(-1)) { // mp_wsm6.F90:2396
  if ((l_qs[((k)-(kts))*(1)] && l_qs[((k_0)-(kts))*(1)])) { // mp_wsm6.F90:2399
  fmelt_s = mp_max(mp_min(sub(1.0e0, div(rs[((k)-(kts))*(1)], rs[((k_0)-(kts))*(1)])), 0.99e0), 0.005e0); // mp_wsm6.F90:2400
  eta = 0.e0; // mp_wsm6.F90:2401
  lams = div(1.f, ilams[((k)-(kts))*(1)]); // mp_wsm6.F90:2402
  for (n=1; n<=S.nrbins; n+=1) { // mp_wsm6.F90:2403
  x = mul(S.xam_s, mp_pow(S.xxds[((n)-(1))*(1)], S.xbm_s)); // mp_wsm6.F90:2404
  rayleigh_soak_wetgraupel(S, x, (double)(S.xocms), (double)(S.xobms), fmelt_s, S.melt_outside_s, S.m_w_0, S.m_i_0, S.lamda_radar, cback, S.mixingrulestring_s, S.matrixstring_s, S.inclusionstring_s, S.hoststring_s, S.hostmatrixstring_s, S.hostinclusionstring_s); // mp_wsm6.F90:2405
  f_d = mul(mul(n0_s[((k)-(kts))*(1)], mp_pow(S.xxds[((n)-(1))*(1)], S.xmu_s)), mp_exp(mul((-lams), S.xxds[((n)-(1))*(1)]))); // mp_wsm6.F90:2410
  eta = add(eta, mul(mul(mul(f_d, cback), S.simpson[((n)-(1))*(1)]), S.xdts[((n)-(1))*(1)])); // mp_wsm6.F90:2411
   } // mp_wsm6.F90:2412
  ze_snow[((k)-(kts))*(1)] = (float)(mul(div(S.lamda4, mul(S.pi5, S.k_w)), eta)); // mp_wsm6.F90:2413
  } // mp_wsm6.F90:2414
  if ((l_qg[((k)-(kts))*(1)] && l_qg[((k_0)-(kts))*(1)])) { // mp_wsm6.F90:2419
  fmelt_g = mp_max(mp_min(sub(1.0e0, div(rg[((k)-(kts))*(1)], rg[((k_0)-(kts))*(1)])), 0.99e0), 0.005e0); // mp_wsm6.F90:2420
  eta = 0.e0; // mp_wsm6.F90:2421
  lamg = div(1.f, ilamg[((k)-(kts))*(1)]); // mp_wsm6.F90:2422
  for (n=1; n<=S.nrbins; n+=1) { // mp_wsm6.F90:2423
  x = mul(S.xam_g, mp_pow(S.xxdg[((n)-(1))*(1)], S.xbm_g)); // mp_wsm6.F90:2424
  rayleigh_soak_wetgraupel(S, x, (double)(S.xocmg), (double)(S.xobmg), fmelt_g, S.melt_outside_g, S.m_w_0, S.m_i_0, S.lamda_radar, cback, S.mixingrulestring_g, S.matrixstring_g, S.inclusionstring_g, S.hoststring_g, S.hostmatrixstring_g, S.hostinclusionstring_g); // mp_wsm6.F90:2425
  f_d = mul(mul(n0_g[((k)-(kts))*(1)], mp_pow(S.xxdg[((n)-(1))*(1)], S.xmu_g)), mp_exp(mul((-lamg), S.xxdg[((n)-(1))*(1)]))); // mp_wsm6.F90:2430
  eta = add(eta, mul(mul(mul(f_d, cback), S.simpson[((n)-(1))*(1)]), S.xdtg[((n)-(1))*(1)])); // mp_wsm6.F90:2431
   } // mp_wsm6.F90:2432
  ze_graupel[((k)-(kts))*(1)] = (float)(mul(div(S.lamda4, mul(S.pi5, S.k_w)), eta)); // mp_wsm6.F90:2433
  } // mp_wsm6.F90:2434
   } // mp_wsm6.F90:2436
  } // mp_wsm6.F90:2437
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:2439
  dbz[((k)-(kts))*(1)] = mul(10.f, mp_log10(mul(add(add(ze_rain[((k)-(kts))*(1)], ze_snow[((k)-(kts))*(1)]), ze_graupel[((k)-(kts))*(1)]), 1.e18))); // mp_wsm6.F90:2440
   } // mp_wsm6.F90:2441
}

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:23-35
__device__ void mp_wsm6_effectrad_init(Wsm6State &S, int &errflg) {
  // CCPP status text // mp_wsm6_effectRad.F90:32
  S.last_message = "mp_wsm6_effectRad_init OK"; // mp_wsm6_effectRad.F90:32
  errflg = 0; // mp_wsm6_effectRad.F90:33
}

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:41-53
__device__ void mp_wsm6_effectrad_finalize(Wsm6State &S, int &errflg) {
  // CCPP status text // mp_wsm6_effectRad.F90:50
  S.last_message = "mp_wsm6_effectRad_finalize OK"; // mp_wsm6_effectRad.F90:50
  errflg = 0; // mp_wsm6_effectRad.F90:51
}

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:59-193
__device__ void mp_wsm6_effectrad_run(Wsm6State &S, bool do_microp_re, const float *t, const float *qc, const float *qi, const float *qs, const float *rho, float qmin, float t0c, float re_qc_bg, float re_qi_bg, float re_qs_bg, float re_qc_max, float re_qi_max, float re_qs_max, float *re_qc, float *re_qi, float *re_qs, int its, int ite, int kts, int kte, int &errflg) {
int i;
int k;
int inu_c;
#ifdef __CUDA_ARCH__
float *ni=S.workspace->mp_wsm6_effectrad_run_ni;
#else
float ni[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rqc=S.workspace->mp_wsm6_effectrad_run_rqc;
#else
float rqc[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rqi=S.workspace->mp_wsm6_effectrad_run_rqi;
#else
float rqi[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rni=S.workspace->mp_wsm6_effectrad_run_rni;
#else
float rni[WSM6_KMAX + 3];
#endif
#ifdef __CUDA_ARCH__
float *rqs=S.workspace->mp_wsm6_effectrad_run_rqs;
#else
float rqs[WSM6_KMAX + 3];
#endif
float temp;
float lamdac;
float supcol;
float n0sfac;
float lamdas;
float diai;
bool has_qc;
bool has_qi;
bool has_qs;
float r1 = 1.e-12f;
float r2 = 1.e-6f;
float bm_r = 3.0f;
float obmr = div(1.0f, bm_r);
float nc0 = 3.e8f;
  if ((!do_microp_re)) { return; } // mp_wsm6_effectRad.F90:115
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:118
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:119
  re_qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = re_qc_bg; // mp_wsm6_effectRad.F90:120
  re_qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = re_qi_bg; // mp_wsm6_effectRad.F90:121
  re_qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = re_qs_bg; // mp_wsm6_effectRad.F90:122
   } // mp_wsm6_effectRad.F90:123
   } // mp_wsm6_effectRad.F90:124
  has_qc = false; // mp_wsm6_effectRad.F90:127
  has_qi = false; // mp_wsm6_effectRad.F90:128
  has_qs = false; // mp_wsm6_effectRad.F90:129
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:131
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:132
  rqc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(r1, mul(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], rho[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6_effectRad.F90:134
  if ((rqc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > r1)) { has_qc = true; } // mp_wsm6_effectRad.F90:135
  rqi[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(r1, mul(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], rho[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6_effectRad.F90:137
  temp = mul(rho[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], mp_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin)); // mp_wsm6_effectRad.F90:138
  temp = mp_sqrt(mp_sqrt(mul(mul(temp, temp), temp))); // mp_wsm6_effectRad.F90:139
  ni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_min(mp_max(mul(5.38e7f, temp), 1.e3f), 1.e6f); // mp_wsm6_effectRad.F90:140
  rni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(r2, mul(ni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rho[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6_effectRad.F90:141
  if (((rqi[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > r1) && (rni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > r2))) { has_qi = true; } // mp_wsm6_effectRad.F90:142
  rqs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = mp_max(r1, mul(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], rho[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6_effectRad.F90:144
  if ((rqs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > r1)) { has_qs = true; } // mp_wsm6_effectRad.F90:145
   } // mp_wsm6_effectRad.F90:146
   } // mp_wsm6_effectRad.F90:147
  if (has_qc) { // mp_wsm6_effectRad.F90:149
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:150
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:151
  if ((rqc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= r1)) { continue; } // mp_wsm6_effectRad.F90:152
  lamdac = mp_pow(div(mul(S.pidnc, nc0), rqc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), obmr); // mp_wsm6_effectRad.F90:153
  re_qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mul(1.5f, div(1.0f, lamdac)), re_qc_max), 2.51e-6f); // mp_wsm6_effectRad.F90:154
   } // mp_wsm6_effectRad.F90:155
   } // mp_wsm6_effectRad.F90:156
  } // mp_wsm6_effectRad.F90:157
  if (has_qi) { // mp_wsm6_effectRad.F90:159
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:160
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:161
  if (((rqi[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= r1) || (rni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= r2))) { continue; } // mp_wsm6_effectRad.F90:162
  diai = mul(11.9f, mp_sqrt(div(rqi[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6_effectRad.F90:163
  re_qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mul(mul(0.75f, 0.163f), diai), re_qi_max), 10.01e-6f); // mp_wsm6_effectRad.F90:164
   } // mp_wsm6_effectRad.F90:165
   } // mp_wsm6_effectRad.F90:166
  } // mp_wsm6_effectRad.F90:167
  if (has_qs) { // mp_wsm6_effectRad.F90:169
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:170
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:171
  if ((rqs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= r1)) { continue; } // mp_wsm6_effectRad.F90:172
  supcol = sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6_effectRad.F90:173
  n0sfac = mp_max(mp_min(mp_exp(mul(S.alpha, supcol)), div(S.n0smax, S.n0s)), 1.f); // mp_wsm6_effectRad.F90:174
  lamdas = mp_sqrt(mp_sqrt(div(mul(S.pidn0s, n0sfac), rqs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6_effectRad.F90:175
  re_qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(mp_min(mul(0.5f, div(1.f, lamdas)), re_qs_max), 25.e-6f); // mp_wsm6_effectRad.F90:176
   } // mp_wsm6_effectRad.F90:177
   } // mp_wsm6_effectRad.F90:178
  } // mp_wsm6_effectRad.F90:179
  for (k=kts; k<=kte; k+=1) { // mp_wsm6_effectRad.F90:182
  for (i=its; i<=ite; i+=1) { // mp_wsm6_effectRad.F90:183
  re_qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(re_qc_bg, mp_min(re_qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], re_qc_max)); // mp_wsm6_effectRad.F90:184
  re_qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(re_qi_bg, mp_min(re_qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], re_qi_max)); // mp_wsm6_effectRad.F90:185
  re_qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = mp_max(re_qs_bg, mp_min(re_qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], re_qs_max)); // mp_wsm6_effectRad.F90:186
   } // mp_wsm6_effectRad.F90:187
   } // mp_wsm6_effectRad.F90:188
  // CCPP status text // mp_wsm6_effectRad.F90:190
  S.last_message = "mp_wsm6_effectRad_run OK"; // mp_wsm6_effectRad.F90:190
  errflg = 0; // mp_wsm6_effectRad.F90:191
}

// MPAS-A v8.4.2 mp_radar.F90:72-189
__device__ void radar_init(Wsm6State &S) {
int n;
  S.pi5 = mul(mul(mul(mul(3.14159f, 3.14159f), 3.14159f), 3.14159f), 3.14159f); // mp_radar.F90:80
  S.lamda4 = mul(mul(mul(S.lamda_radar, S.lamda_radar), S.lamda_radar), S.lamda_radar); // mp_radar.F90:81
  S.m_w_0 = cx(0x1.213d1a0000000p+3, -0x1.634d6c0000000p+0); // GNU 15.2 strict constant fold (radar_init fixed parameters) // mp_radar.F90:82
  S.m_i_0 = cx(0x1.c91dae74d3b24p+0, -0x1.1acb622ae704dp-13); // GNU 15.2 strict constant fold (radar_init fixed parameters) // mp_radar.F90:83
  S.k_w = 0x1.de4b5a79d9de5p-1; // GNU 15.2 strict constant fold (radar_init fixed parameters) // mp_radar.F90:84
  for (n=1; n<=(S.nrbins + 1); n+=1) { // mp_radar.F90:86
  S.simpson[((n)-(1))*(1)] = 0.0e0; // mp_radar.F90:87
   } // mp_radar.F90:88
  for (n=1; n<=(S.nrbins - 1); n+=2) { // mp_radar.F90:89
  S.simpson[((n)-(1))*(1)] = add(S.simpson[((n)-(1))*(1)], S.basis[((1)-(1))*(1)]); // mp_radar.F90:90
  S.simpson[(((n + 1))-(1))*(1)] = add(S.simpson[(((n + 1))-(1))*(1)], S.basis[((2)-(1))*(1)]); // mp_radar.F90:91
  S.simpson[(((n + 2))-(1))*(1)] = add(S.simpson[(((n + 2))-(1))*(1)], S.basis[((3)-(1))*(1)]); // mp_radar.F90:92
   } // mp_radar.F90:93
  for (n=1; n<=S.slen; n+=1) { // mp_radar.F90:95
  // string initialized below // mp_radar.F90:96
  // string initialized below // mp_radar.F90:97
  // string initialized below // mp_radar.F90:98
  // string initialized below // mp_radar.F90:99
  // string initialized below // mp_radar.F90:100
  // string initialized below // mp_radar.F90:101
  // string initialized below // mp_radar.F90:102
  // string initialized below // mp_radar.F90:103
  // string initialized below // mp_radar.F90:104
  // string initialized below // mp_radar.F90:105
  // string initialized below // mp_radar.F90:106
  // string initialized below // mp_radar.F90:107
   } // mp_radar.F90:108
  S.mixingrulestring_s = "maxwellgarnett"; // mp_radar.F90:110
  S.hoststring_s = "air"; // mp_radar.F90:111
  S.matrixstring_s = "water"; // mp_radar.F90:112
  S.inclusionstring_s = "spheroidal"; // mp_radar.F90:113
  S.hostmatrixstring_s = "icewater"; // mp_radar.F90:114
  S.hostinclusionstring_s = "spheroidal"; // mp_radar.F90:115
  S.mixingrulestring_g = "maxwellgarnett"; // mp_radar.F90:117
  S.hoststring_g = "air"; // mp_radar.F90:118
  S.matrixstring_g = "water"; // mp_radar.F90:119
  S.inclusionstring_g = "spheroidal"; // mp_radar.F90:120
  S.hostmatrixstring_g = "icewater"; // mp_radar.F90:121
  S.hostinclusionstring_g = "spheroidal"; // mp_radar.F90:122
  S.xxdx[((1)-(1))*(1)] = 100.e-6; // mp_radar.F90:125
  S.xxdx[(((S.nrbins + 1))-(1))*(1)] = 0.02e0; // mp_radar.F90:126
  for (n=2; n<=S.nrbins; n+=1) { // mp_radar.F90:127
  S.xxdx[((n)-(1))*(1)] = mp_exp(add(mul(div((double)((n - 1)), (double)(S.nrbins)), mp_log(div(S.xxdx[(((S.nrbins + 1))-(1))*(1)], S.xxdx[((1)-(1))*(1)]))), mp_log(S.xxdx[((1)-(1))*(1)]))); // mp_radar.F90:128
   } // mp_radar.F90:130
  for (n=1; n<=S.nrbins; n+=1) { // mp_radar.F90:131
  S.xxds[((n)-(1))*(1)] = mp_sqrt(mul(S.xxdx[((n)-(1))*(1)], S.xxdx[(((n + 1))-(1))*(1)])); // mp_radar.F90:132
  S.xdts[((n)-(1))*(1)] = sub(S.xxdx[(((n + 1))-(1))*(1)], S.xxdx[((n)-(1))*(1)]); // mp_radar.F90:133
   } // mp_radar.F90:134
  S.xxdx[((1)-(1))*(1)] = 100.e-6; // mp_radar.F90:137
  S.xxdx[(((S.nrbins + 1))-(1))*(1)] = 0.05e0; // mp_radar.F90:138
  for (n=2; n<=S.nrbins; n+=1) { // mp_radar.F90:139
  S.xxdx[((n)-(1))*(1)] = mp_exp(add(mul(div((double)((n - 1)), (double)(S.nrbins)), mp_log(div(S.xxdx[(((S.nrbins + 1))-(1))*(1)], S.xxdx[((1)-(1))*(1)]))), mp_log(S.xxdx[((1)-(1))*(1)]))); // mp_radar.F90:140
   } // mp_radar.F90:142
  for (n=1; n<=S.nrbins; n+=1) { // mp_radar.F90:143
  S.xxdg[((n)-(1))*(1)] = mp_sqrt(mul(S.xxdx[((n)-(1))*(1)], S.xxdx[(((n + 1))-(1))*(1)])); // mp_radar.F90:144
  S.xdtg[((n)-(1))*(1)] = sub(S.xxdx[(((n + 1))-(1))*(1)], S.xxdx[((n)-(1))*(1)]); // mp_radar.F90:145
   } // mp_radar.F90:146
  S.xcre[((1)-(1))*(1)] = add(1.f, S.xbm_r); // mp_radar.F90:154
  S.xcre[((2)-(1))*(1)] = add(1.f, S.xmu_r); // mp_radar.F90:155
  S.xcre[((3)-(1))*(1)] = add(4.f, S.xmu_r); // mp_radar.F90:156
  S.xcre[((4)-(1))*(1)] = add(7.f, S.xmu_r); // mp_radar.F90:157
  for (n=1; n<=4; n+=1) { // mp_radar.F90:158
  S.xcrg[((n)-(1))*(1)] = wgamma(S, S.xcre[((n)-(1))*(1)]); // mp_radar.F90:159
   } // mp_radar.F90:160
  S.xorg2 = div(1.f, S.xcrg[((2)-(1))*(1)]); // mp_radar.F90:161
  S.xcse[((1)-(1))*(1)] = add(1.f, S.xbm_s); // mp_radar.F90:163
  S.xcse[((2)-(1))*(1)] = add(1.f, S.xmu_s); // mp_radar.F90:164
  S.xcse[((3)-(1))*(1)] = add(4.f, S.xmu_s); // mp_radar.F90:165
  S.xcse[((4)-(1))*(1)] = add(7.f, S.xmu_s); // mp_radar.F90:166
  for (n=1; n<=4; n+=1) { // mp_radar.F90:167
  S.xcsg[((n)-(1))*(1)] = wgamma(S, S.xcse[((n)-(1))*(1)]); // mp_radar.F90:168
   } // mp_radar.F90:169
  S.xosg2 = div(1.f, S.xcsg[((2)-(1))*(1)]); // mp_radar.F90:170
  S.xcge[((1)-(1))*(1)] = add(1.f, S.xbm_g); // mp_radar.F90:172
  S.xcge[((2)-(1))*(1)] = add(1.f, S.xmu_g); // mp_radar.F90:173
  S.xcge[((3)-(1))*(1)] = add(4.f, S.xmu_g); // mp_radar.F90:174
  S.xcge[((4)-(1))*(1)] = add(7.f, S.xmu_g); // mp_radar.F90:175
  for (n=1; n<=4; n+=1) { // mp_radar.F90:176
  S.xcgg[((n)-(1))*(1)] = wgamma(S, S.xcge[((n)-(1))*(1)]); // mp_radar.F90:177
   } // mp_radar.F90:178
  S.xogg2 = div(1.f, S.xcgg[((2)-(1))*(1)]); // mp_radar.F90:179
  S.xobmr = div(1.f, S.xbm_r); // mp_radar.F90:181
  S.xoams = div(1.f, S.xam_s); // mp_radar.F90:182
  S.xobms = div(1.f, S.xbm_s); // mp_radar.F90:183
  S.xocms = mp_pow(S.xoams, S.xobms); // mp_radar.F90:184
  S.xoamg = div(1.f, S.xam_g); // mp_radar.F90:185
  S.xobmg = div(1.f, S.xbm_g); // mp_radar.F90:186
  S.xocmg = mp_pow(S.xoamg, S.xobmg); // mp_radar.F90:187
}

// MPAS-A v8.4.2 mp_radar.F90:192-292
__device__ void rayleigh_soak_wetgraupel(Wsm6State &S, double x_g, double a_geo, double b_geo, double fmelt, double meltratio_outside, cx m_w, cx m_i, double lambda, double &c_back, const char *mixingrule, const char *matrix, const char *inclusion, const char *host, const char *hostmatrix, const char *hostinclusion) {
int error;
cx m_core;
cx m_air;
double pix = 3.1415926535897932384626434e0;
double d_large;
double d_g;
double rhog;
double x_w;
double xw_a;
double fm;
double fmgrenz;
double volg;
double vg;
double volair;
double volice;
double volwater;
double meltratio_outside_grenz;
double mra;
  m_air = cx(1.0e0, 0.0e0); // mp_radar.F90:221
  fm = mp_max(mp_min(fmelt, 1.0e0), 0.0e0); // mp_radar.F90:224
  mra = mp_max(mp_min(meltratio_outside, 1.0e0), 0.0e0); // mp_radar.F90:226
  mra = add(mra, mul(sub(1.0e0, mra), fm)); // mp_radar.F90:233
  x_w = mul(x_g, fm); // mp_radar.F90:235
  d_g = mul(a_geo, mp_pow(x_g, b_geo)); // mp_radar.F90:237
  if ((d_g >= 1e-12)) { // mp_radar.F90:239
  vg = mul(div(pix, 6.f), ipow<3>(d_g)); // mp_radar.F90:241
  rhog = mp_max(mp_min(div(x_g, vg), 900.0e0), 10.0e0); // mp_radar.F90:242
  vg = div(x_g, rhog); // mp_radar.F90:243
  meltratio_outside_grenz = sub(1.0e0, div(rhog, 1000.f)); // mp_radar.F90:245
  if ((mra <= meltratio_outside_grenz)) { // mp_radar.F90:247
  volg = mul(vg, sub(1.0e0, mul(mra, fm))); // mp_radar.F90:251
  } else { // mp_radar.F90:253
  fmgrenz = div(sub(900.0f, rhog), add(sub(mul(mra, 900.0f), rhog), div(mul(900.0f, rhog), 1000.f))); // mp_radar.F90:256
  if ((fm <= fmgrenz)) { // mp_radar.F90:258
  volg = mul(sub(1.0f, mul(mra, fm)), vg); // mp_radar.F90:260
  } else { // mp_radar.F90:261
  volg = add(div(sub(x_g, x_w), 900.0f), div(x_w, 1000.f)); // mp_radar.F90:264
  } // mp_radar.F90:265
  } // mp_radar.F90:267
  d_large = mp_pow(mul(div(6.0f, pix), volg), div(1.f, 3.f)); // mp_radar.F90:269
  volice = div(sub(x_g, x_w), mul(volg, 900.0f)); // mp_radar.F90:270
  volwater = div(x_w, mul(1000.f, volg)); // mp_radar.F90:271
  volair = sub(sub(1.0f, volice), volwater); // mp_radar.F90:272
  m_core = get_m_mix_nested(S, m_air, m_i, m_w, volair, volice, volwater, mixingrule, host, matrix, inclusion, hostmatrix, hostinclusion, error); // mp_radar.F90:276
  if ((error != 0)) { // mp_radar.F90:279
  c_back = 0.0e0; // mp_radar.F90:280
  return; // mp_radar.F90:281
  } // mp_radar.F90:282
  c_back = div(mul(mul(ipow<2>(mp_abs(div(sub(ipow<2>(m_core), 1.0e0), add(ipow<2>(m_core), 2.0e0)))), S.pi5), ipow<6>(d_large)), S.lamda4); // mp_radar.F90:285
  } else { // mp_radar.F90:288
  c_back = 0.0e0; // mp_radar.F90:289
  } // mp_radar.F90:290
}

// MPAS-A v8.4.2 mp_radar.F90:295-306
__device__ float wgamma(Wsm6State &S, float y) {
float result;
  result = mp_exp(gammln(S, y)); // mp_radar.F90:304
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:309-342
__device__ float gammln(Wsm6State &S, float xx) {
int j;
double stp = 2.5066282746310005e0;
double cof[(6-1+1)] = {76.18009172947146e0, (-86.50532032941677e0), 24.01409824083091e0, (-1.231739572450155e0), .1208650973866179e-2, (-.5395239384953e-5)};
double ser;
double tmp;
double x;
double y;
float result;
  x = xx; // mp_radar.F90:330
  y = x; // mp_radar.F90:331
  tmp = add(x, 5.5e0); // mp_radar.F90:332
  tmp = sub(mul(add(x, 0.5e0), mp_log(tmp)), tmp); // mp_radar.F90:333
  ser = 1.000000000190015e0; // mp_radar.F90:334
  for (j=1; j<=6; j+=1) { // mp_radar.F90:335
  y = add(y, 1.e0); // mp_radar.F90:336
  ser = add(ser, div(cof[((j)-(1))*(1)], y)); // mp_radar.F90:337
   } // mp_radar.F90:338
  result = add(tmp, mp_log(div(mul(stp, ser), x))); // mp_radar.F90:340
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:345-483
__device__ cx get_m_mix_nested(Wsm6State &S, cx m_a, cx m_i, cx m_w, double volair, double volice, double volwater, const char *mixingrule, const char *host, const char *matrix, const char *inclusion, const char *hostmatrix, const char *hostinclusion, int &cumulerror) {
int error;
cx mtmp;
double vol1;
double vol2;
cx result;
  cumulerror = 0; // mp_radar.F90:372
  result = mp_cmplx(1.0e0, 0.0e0); // mp_radar.F90:373
  if (str_eq(host, "air")) { // mp_radar.F90:375
  if (str_eq(matrix, "air")) { // mp_radar.F90:376
  // diagnostic only // mp_radar.F90:377
  cumulerror = (cumulerror + 1); // mp_radar.F90:379
  } else { // mp_radar.F90:380
  vol1 = div(volice, mp_max(add(volice, volwater), 1e-10)); // mp_radar.F90:381
  vol2 = sub(1.0e0, vol1); // mp_radar.F90:382
  mtmp = get_m_mix(S, m_a, m_i, m_w, 0.0e0, vol1, vol2, mixingrule, matrix, inclusion, error); // mp_radar.F90:383
  cumulerror = (cumulerror + error); // mp_radar.F90:385
  if (str_eq(hostmatrix, "air")) { // mp_radar.F90:387
  result = get_m_mix(S, m_a, mtmp, mul(2.0f, m_a), volair, sub(1.0e0, volair), 0.0e0, mixingrule, hostmatrix, hostinclusion, error); // mp_radar.F90:388
  cumulerror = (cumulerror + error); // mp_radar.F90:391
  } else if (str_eq(hostmatrix, "icewater")) { // mp_radar.F90:392
  result = get_m_mix(S, m_a, mtmp, mul(2.0f, m_a), volair, sub(1.0e0, volair), 0.0e0, mixingrule, "ice", hostinclusion, error); // mp_radar.F90:393
  cumulerror = (cumulerror + error); // mp_radar.F90:396
  } else { // mp_radar.F90:397
  // diagnostic only // mp_radar.F90:398
  cumulerror = (cumulerror + 1); // mp_radar.F90:400
  } // mp_radar.F90:401
  } // mp_radar.F90:402
  } else if (str_eq(host, "ice")) { // mp_radar.F90:404
  if (str_eq(matrix, "ice")) { // mp_radar.F90:406
  // diagnostic only // mp_radar.F90:407
  cumulerror = (cumulerror + 1); // mp_radar.F90:409
  } else { // mp_radar.F90:410
  vol1 = div(volair, mp_max(add(volair, volwater), 1e-10)); // mp_radar.F90:411
  vol2 = sub(1.0e0, vol1); // mp_radar.F90:412
  mtmp = get_m_mix(S, m_a, m_i, m_w, vol1, 0.0e0, vol2, mixingrule, matrix, inclusion, error); // mp_radar.F90:413
  cumulerror = (cumulerror + error); // mp_radar.F90:415
  if (str_eq(hostmatrix, "ice")) { // mp_radar.F90:417
  result = get_m_mix(S, mtmp, m_i, mul(2.0f, m_a), sub(1.0e0, volice), volice, 0.0e0, mixingrule, hostmatrix, hostinclusion, error); // mp_radar.F90:418
  cumulerror = (cumulerror + error); // mp_radar.F90:421
  } else if (str_eq(hostmatrix, "airwater")) { // mp_radar.F90:422
  result = get_m_mix(S, mtmp, m_i, mul(2.0f, m_a), sub(1.0e0, volice), volice, 0.0e0, mixingrule, "air", hostinclusion, error); // mp_radar.F90:423
  cumulerror = (cumulerror + error); // mp_radar.F90:426
  } else { // mp_radar.F90:427
  // diagnostic only // mp_radar.F90:428
  cumulerror = (cumulerror + 1); // mp_radar.F90:430
  } // mp_radar.F90:431
  } // mp_radar.F90:432
  } else if (str_eq(host, "water")) { // mp_radar.F90:434
  if (str_eq(matrix, "water")) { // mp_radar.F90:436
  // diagnostic only // mp_radar.F90:437
  cumulerror = (cumulerror + 1); // mp_radar.F90:439
  } else { // mp_radar.F90:440
  vol1 = div(volair, mp_max(add(volice, volair), 1e-10)); // mp_radar.F90:441
  vol2 = sub(1.0e0, vol1); // mp_radar.F90:442
  mtmp = get_m_mix(S, m_a, m_i, m_w, vol1, vol2, 0.0e0, mixingrule, matrix, inclusion, error); // mp_radar.F90:443
  cumulerror = (cumulerror + error); // mp_radar.F90:445
  if (str_eq(hostmatrix, "water")) { // mp_radar.F90:447
  result = get_m_mix(S, mul(2, m_a), mtmp, m_w, 0.0e0, sub(1.0e0, volwater), volwater, mixingrule, hostmatrix, hostinclusion, error); // mp_radar.F90:448
  cumulerror = (cumulerror + error); // mp_radar.F90:451
  } else if (str_eq(hostmatrix, "airice")) { // mp_radar.F90:452
  result = get_m_mix(S, mul(2, m_a), mtmp, m_w, 0.0e0, sub(1.0e0, volwater), volwater, mixingrule, "ice", hostinclusion, error); // mp_radar.F90:453
  cumulerror = (cumulerror + error); // mp_radar.F90:456
  } else { // mp_radar.F90:457
  // diagnostic only // mp_radar.F90:458
  cumulerror = (cumulerror + 1); // mp_radar.F90:460
  } // mp_radar.F90:461
  } // mp_radar.F90:462
  } else if (str_eq(host, "none")) { // mp_radar.F90:464
  result = get_m_mix(S, m_a, m_i, m_w, volair, volice, volwater, mixingrule, matrix, inclusion, error); // mp_radar.F90:466
  cumulerror = (cumulerror + error); // mp_radar.F90:469
  } else { // mp_radar.F90:471
  // diagnostic only // mp_radar.F90:472
  cumulerror = (cumulerror + 1); // mp_radar.F90:474
  } // mp_radar.F90:475
  if ((cumulerror != 0)) { // mp_radar.F90:477
  // diagnostic only // mp_radar.F90:478
  result = mp_cmplx(1.0e0, 0.0e0); // mp_radar.F90:480
  } // mp_radar.F90:481
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:486-533
__device__ cx get_m_mix(Wsm6State &S, cx m_a, cx m_i, cx m_w, double volair, double volice, double volwater, const char *mixingrule, const char *matrix, const char *inclusion, int &error) {
cx result;
  error = 0; // mp_radar.F90:503
  result = mp_cmplx(1.0e0, 0.0e0); // mp_radar.F90:504
  if (str_eq(mixingrule, "maxwellgarnett")) { // mp_radar.F90:506
  if (str_eq(matrix, "ice")) { // mp_radar.F90:507
  result = m_complex_maxwellgarnett(S, volice, volair, volwater, m_i, m_a, m_w, inclusion, error); // mp_radar.F90:508
  } else if (str_eq(matrix, "water")) { // mp_radar.F90:510
  result = m_complex_maxwellgarnett(S, volwater, volair, volice, m_w, m_a, m_i, inclusion, error); // mp_radar.F90:511
  } else if (str_eq(matrix, "air")) { // mp_radar.F90:513
  result = m_complex_maxwellgarnett(S, volair, volwater, volice, m_a, m_w, m_i, inclusion, error); // mp_radar.F90:514
  } else { // mp_radar.F90:516
  // diagnostic only // mp_radar.F90:517
  error = 1; // mp_radar.F90:519
  } // mp_radar.F90:520
  } else { // mp_radar.F90:522
  // diagnostic only // mp_radar.F90:523
  error = 2; // mp_radar.F90:525
  } // mp_radar.F90:526
  if ((error != 0)) { // mp_radar.F90:528
  // diagnostic only // mp_radar.F90:529
  } // mp_radar.F90:531
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:536-589
__device__ cx m_complex_maxwellgarnett(Wsm6State &S, double vol1, double vol2, double vol3, cx m1, cx m2, cx m3, const char *inclusion, int &error) {
cx beta2;
cx beta3;
cx m1t;
cx m2t;
cx m3t;
cx result;
  error = 0; // mp_radar.F90:557
  if ((mp_abs(sub(add(add(vol1, vol2), vol3), 1.0e0)) > 1e-6)) { // mp_radar.F90:559
  // diagnostic only // mp_radar.F90:560
  result = mp_cmplx((-999.99e0), (-999.99e0)); // mp_radar.F90:563
  error = 1; // mp_radar.F90:564
  return result; // mp_radar.F90:565
  } // mp_radar.F90:566
  m1t = ipow<2>(m1); // mp_radar.F90:568
  m2t = ipow<2>(m2); // mp_radar.F90:569
  m3t = ipow<2>(m3); // mp_radar.F90:570
  if (str_eq(inclusion, "spherical")) { // mp_radar.F90:572
  beta2 = div(mul(3.0e0, m1t), add(m2t, mul(2.0e0, m1t))); // mp_radar.F90:573
  beta3 = div(mul(3.0e0, m1t), add(m3t, mul(2.0e0, m1t))); // mp_radar.F90:574
  } else if (str_eq(inclusion, "spheroidal")) { // mp_radar.F90:575
  beta2 = mul(div(mul(2.0e0, m1t), sub(m2t, m1t)), sub(mul(div(m2t, sub(m2t, m1t)), mp_log(div(m2t, m1t))), 1.0e0)); // mp_radar.F90:576
  beta3 = mul(div(mul(2.0e0, m1t), sub(m3t, m1t)), sub(mul(div(m3t, sub(m3t, m1t)), mp_log(div(m3t, m1t))), 1.0e0)); // mp_radar.F90:577
  } else { // mp_radar.F90:578
  // diagnostic only // mp_radar.F90:579
  result = cx(-999.99e0, -999.99e0); // mp_radar.F90:581; explicit kind=R8KIND
  error = 1; // mp_radar.F90:582
  return result; // mp_radar.F90:583
  } // mp_radar.F90:584
  result = mp_sqrt(div(add(add(mul(sub(sub(1.0e0, vol2), vol3), m1t), mul(mul(vol2, beta2), m2t)), mul(mul(vol3, beta3), m3t)), add(add(sub(sub(1.0e0, vol2), vol3), mul(vol2, beta2)), mul(vol3, beta3)))); // mp_radar.F90:586
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:592-629
__device__ cx m_complex_water_ray(Wsm6State &S, double lambda, double t) {
double pix = 3.1415926535897932384626434e0;
double epsinf;
double epss;
double epsr;
double epsi;
double alpha;
double lambdas;
double sigma;
double nenner;
cx i = cx(0e0, 1e0);
cx result;
  epsinf = sub(add(5.27137e0, mul(0.02164740e0, t)), mul(mul(0.00131198e0, t), t)); // mp_radar.F90:612
  epss = mul(78.54e+0, sub(add(sub(1.0f, mul(4.579e-3, sub(t, 25.0f))), mul(mul(1.190e-5, sub(t, 25.0f)), sub(t, 25.0f))), mul(mul(mul(2.800e-8, sub(t, 25.0f)), sub(t, 25.0f)), sub(t, 25.0f)))); // mp_radar.F90:613
  alpha = add(div((-16.8129e0), add(t, 273.16f)), 0.0609265e0); // mp_radar.F90:616
  lambdas = mul(mul(0.00033836e0, mp_exp(div(2513.98e0, add(t, 273.16f)))), 1e-2f); // mp_radar.F90:617
  nenner = add(add(1.e0, mul(mul(2.e0, mp_pow(div(lambdas, lambda), sub(1e0, alpha))), mp_sin(mul(mul(alpha, pix), 0.5f)))), mp_pow(div(lambdas, lambda), sub(2e0, mul(2e0, alpha)))); // mp_radar.F90:619
  epsr = add(epsinf, div(mul(sub(epss, epsinf), add(mul(mp_pow(div(lambdas, lambda), sub(1e0, alpha)), mp_sin(mul(mul(alpha, pix), 0.5f))), 1e0)), nenner)); // mp_radar.F90:621
  epsi = add(div(mul(sub(epss, epsinf), add(mul(mp_pow(div(lambdas, lambda), sub(1e0, alpha)), mp_cos(mul(mul(alpha, pix), 0.5f))), 0e0)), nenner), div(mul(lambda, 1.25664f), 1.88496f)); // mp_radar.F90:623
  result = mp_sqrt(mp_cmplx(epsr, (-epsi))); // mp_radar.F90:627
return result;
}

// MPAS-A v8.4.2 mp_radar.F90:632-673
__device__ cx m_complex_ice_maetzler(Wsm6State &S, double lambda, double t) {
double f;
double c;
double tk;
double b1;
double b2;
double b;
double deltabeta;
double betam;
double beta;
double theta;
double alfa;
cx result;
  c = 2.99e8; // mp_radar.F90:656
  tk = add(t, 273.16f); // mp_radar.F90:657
  f = mul(div(c, lambda), 1e-9); // mp_radar.F90:658
  b1 = 0.0207f; // mp_radar.F90:660
  b2 = 1.16e-11; // mp_radar.F90:661
  b = 335.0e0; // mp_radar.F90:662
  deltabeta = mp_exp(add((-10.02f), mul(0.0364f, sub(tk, 273.16f)))); // mp_radar.F90:663
  betam = add(mul(div(b1, tk), div(mp_exp(div(b, tk)), ipow<2>(sub(mp_exp(div(b, tk)), 1)))), mul(mul(b2, f), f)); // mp_radar.F90:664
  beta = add(betam, deltabeta); // mp_radar.F90:665
  theta = sub(div(300.f, tk), 1.f); // mp_radar.F90:666
  alfa = mul(add(0.00504e0, mul(0.0062e0, theta)), mp_exp(mul((-22.1e0), theta))); // mp_radar.F90:667
  result = add(3.1884f, mul(9.1e-4f, sub(tk, 273.16f))); // mp_radar.F90:668
  result = add(result, mp_cmplx(0.0e0, add(div(alfa, f), mul(beta, f)))); // mp_radar.F90:669
  result = mp_sqrt(mp_conjg(result)); // mp_radar.F90:671
return result;
}
'''

CUDA_COLUMNS = r'''
// MPAS-A v8.4.2 module_mp_wsm6.F:18-239; mpas_atmphys_driver_microphysics.F:437-461.
// All arrays [component,level,cell] with one scalar column owned by each thread.
extern "C" __global__ void wsm6_columns(int nz,int nc,int ns,float dt,int hail,
 const float *c,float *v,float *s,float *r,float *dbz,float *prod,const float *pi,
 int wrapper,int radii_on,int radar_on,int production_on,int *status,char *messages=nullptr,int first=0,Wsm6Workspace *workspace=nullptr) {
 int i=first+blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
#ifdef __CUDA_ARCH__
 Wsm6Workspace &B=workspace[i-first];
 float *local=B.column_local,*surface=B.column_surface,*radii=B.column_radii;
 float *radar=B.column_radar,*production=B.column_production;
#else
 float local[10*WSM6_KMAX],surface[7],radii[3*WSM6_KMAX],radar[WSM6_KMAX],production[2*WSM6_KMAX];
#endif
 for(int n=0;n<10;++n) for(int k=0;k<nz;++k) local[n*nz+k]=v[(n*nz+k)*nc+i];
 for(int n=0;n<7;++n) surface[n]=s[n*nc+i];
 for(int n=0;n<3;++n) for(int k=0;k<nz;++k) radii[n*nz+k]=r[(n*nz+k)*nc+i];
 if(wrapper) for(int k=0;k<nz;++k) local[k]=mul(local[k],pi[k*nc+i]);
#ifdef __CUDA_ARCH__
 Wsm6State &S=B.state;S=Wsm6State{};S.workspace=&B;
#else
 Wsm6State S{};
#endif
 state_parameters(S);int err=0;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],hail,err);
 if(err) {status[i]=err;return;}
 mp_wsm6_run(S,local,local+nz,local+2*nz,local+3*nz,local+4*nz,local+5*nz,local+6*nz,local+7*nz,local+8*nz,local+9*nz,dt,
 c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],
 surface,surface+1,surface+6,surface+2,surface+3,surface+4,surface+5,
 production_on?production:nullptr,production_on?production+nz:nullptr,1,1,1,nz,err);
 if(err) {status[i]=err;return;}
 if(wrapper) for(int k=0;k<nz;++k) {
   float theta=div(local[k],pi[k*nc+i]);v[k*nc+i]=theta;
   local[k]=mul(theta,pi[k*nc+i]);
 }
 mp_wsm6_effectrad_run(S,radii_on,local,local+2*nz,local+3*nz,local+5*nz,local+7*nz,c[11],c[5],c[20],c[21],c[22],c[23],c[24],c[25],radii,radii+nz,radii+2*nz,1,1,1,nz,err);
 if(radar_on) refl10cm_wsm6(S,local+nz,local+4*nz,local+5*nz,local+6*nz,local,local+8*nz,radar,1,nz);
 for(int n=wrapper?1:0;n<10;++n) for(int k=0;k<nz;++k) v[(n*nz+k)*nc+i]=local[n*nz+k];
 for(int n=0;n<7;++n) s[n*nc+i]=surface[n];
 for(int n=0;n<3;++n) for(int k=0;k<nz;++k) r[(n*nz+k)*nc+i]=radii[n*nz+k];
 if(radar_on) for(int k=0;k<nz;++k) dbz[k*nc+i]=radar[k];
 if(production_on) for(int n=0;n<2;++n) for(int k=0;k<nz;++k) prod[(n*nz+k)*nc+i]=production[n*nz+k];
 if(messages) {
   // Fortran character assignment: module_mp_wsm6.F:223; mp_wsm6_effectRad.F90:190.
   int n=0;while(S.last_message[n] && n<256){messages[n*nc+i]=S.last_message[n];++n;}
   for(;n<256;++n)messages[n*nc+i]=' ';
 }
 status[i]=err;
}
'''

FIELD_NAMES = ('theta_m', 'rho_zz', 'zz', 'exner', 'pressure_base', 'pressure_p', 'rtheta_base', 'rtheta_p', 'exner_base', 'zgrid', 'qv', 'qc', 'qi', 'qr', 'qs', 'qg', 'rainprod', 'evapprod', 're_cloud', 're_ice', 're_snow', 'rthmpten', 'rqvmpten', 'rqcmpten', 'rqimpten', 'rqrmpten', 'rqsmpten', 'rqgmpten', 'rt_diabatic_tend', 'dtheta_dt_mp', 'surface_pressure', 'tend_sfc_pressure', 'precipw', 'rainnc', 'rainncv', 'snownc', 'snowncv', 'graupelnc', 'graupelncv', 'sr', 'refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max')

CUDA_DRIVER = r'''
#define FIELD_theta_m 0
#define FIELD_rho_zz 1
#define FIELD_zz 2
#define FIELD_exner 3
#define FIELD_pressure_base 4
#define FIELD_pressure_p 5
#define FIELD_rtheta_base 6
#define FIELD_rtheta_p 7
#define FIELD_exner_base 8
#define FIELD_zgrid 9
#define FIELD_qv 10
#define FIELD_qc 11
#define FIELD_qi 12
#define FIELD_qr 13
#define FIELD_qs 14
#define FIELD_qg 15
#define FIELD_rainprod 16
#define FIELD_evapprod 17
#define FIELD_re_cloud 18
#define FIELD_re_ice 19
#define FIELD_re_snow 20
#define FIELD_rthmpten 21
#define FIELD_rqvmpten 22
#define FIELD_rqcmpten 23
#define FIELD_rqimpten 24
#define FIELD_rqrmpten 25
#define FIELD_rqsmpten 26
#define FIELD_rqgmpten 27
#define FIELD_rt_diabatic_tend 28
#define FIELD_dtheta_dt_mp 29
#define FIELD_surface_pressure 30
#define FIELD_tend_sfc_pressure 31
#define FIELD_precipw 32
#define FIELD_rainnc 33
#define FIELD_rainncv 34
#define FIELD_snownc 35
#define FIELD_snowncv 36
#define FIELD_graupelnc 37
#define FIELD_graupelncv 38
#define FIELD_sr 39
#define FIELD_refl10cm 40
#define FIELD_refl10cm_max 41
#define FIELD_refl10cm_1km 42
#define FIELD_refl10cm_1km_max 43
#define F(name,k) fields[FIELD_##name][(k)*nc+i]
#define SF(name) fields[FIELD_##name][i]
__device__ const int SPECIES[] = {FIELD_qv, FIELD_qc, FIELD_qi, FIELD_qr, FIELD_qs, FIELD_qg};
__device__ const int RADII[] = {FIELD_re_cloud, FIELD_re_ice, FIELD_re_snow};
__device__ const int PRODUCTION[] = {FIELD_rainprod, FIELD_evapprod};
__device__ const int TENDENCIES[] = {FIELD_rqvmpten, FIELD_rqcmpten, FIELD_rqimpten, FIELD_rqrmpten, FIELD_rqsmpten, FIELD_rqgmpten};
// MPAS-A v8.4.2 mpas_atmphys_interface.F:598-823; mpas_atmphys_driver_microphysics.F:505-563.
extern "C" __global__ void wsm6_from_mpas(int nz,int nc,int ns,float rv_rd,
 float **fields,float *v,float *s,float *r,float *prod,float *pi) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
 for(int k=0;k<nz;++k) {
  const int x=k*nc+i;
  v[x]=div(F(theta_m,k),add(1.f,mul(rv_rd,mp_max(F(qv,k), 0.f))));
  for(int n=0;n<6;++n) v[(n+1)*nz*nc+x]=fields[SPECIES[n]][x];
  v[7*nz*nc+x]=mul(F(zz,k),F(rho_zz,k));
  v[8*nz*nc+x]=add(F(pressure_base,k),F(pressure_p,k));
  v[9*nz*nc+x]=sub(F(zgrid,k+1),F(zgrid,k));pi[x]=F(exner,k);
  for(int n=0;n<3;++n) r[n*nz*nc+x]=fields[RADII[n]][x];
  for(int n=0;n<2;++n) prod[n*nz*nc+x]=fields[PRODUCTION[n]][x];
  F(rthmpten,k)=v[x];
  for(int n=0;n<6;++n) fields[TENDENCIES[n]][x]=fields[SPECIES[n]][x];
 }
 for(int n=0;n<7;++n)s[n*nc+i]=0.f;
}
// MPAS-A v8.4.2 mpas_atmphys_driver_microphysics.F:690-740. Called before *_to_MPAS.
extern "C" __global__ void wsm6_radar_to_mpas(int nz,int nc,int ns,float **fields,
 const float *raw,int *status) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
 int kp=0;float maximum=-35.f;
 // Native driver zp(1:kte) and refl10cm(:,1:kte,:) have no assigned
 // kp+1 value when every centre is below 1 km (:711-727). Strict native
 // continues, with measured run-to-run variation in the 1 km outputs.
 // Keep an unassigned trailing storage word for those reads, rather than
 // inventing extrapolated values or refusing the complete scheme call.
 float heights[WSM6_KMAX+1], reflectivity[WSM6_KMAX+1];
 for(int k=0;k<nz;++k) {
  float dbz=mp_max(raw[k*nc+i], -35.f);F(refl10cm,k)=dbz;maximum=mp_max(maximum,dbz);
  float zp=add(sub(F(zgrid,k),F(zgrid,0)),mul(.5f,sub(F(zgrid,k+1),F(zgrid,k))));
  heights[k]=zp;reflectivity[k]=dbz;
  if(zp<1000.f)kp=k;
 }
 float z1=heights[kp],z2=heights[kp+1];
 float w1=div(sub(z2,1000.f),sub(z2,z1)),w2=sub(1.f,w1);
 SF(refl10cm_max)=maximum;
 SF(refl10cm_1km)=add(mul(w1,reflectivity[kp]),mul(w2,reflectivity[kp+1]));
 SF(refl10cm_1km_max)=mp_max(SF(refl10cm_1km_max),SF(refl10cm_1km));
}
// MPAS-A v8.4.2 mpas_atmphys_interface.F:826-1138 (WSM6 branch).
// mpas_atmphys_driver_microphysics.F:568-653: exactly one bucket subtraction, strictly >.
extern "C" __global__ void wsm6_to_mpas(int nz,int nc,int ns,int top,float dt_dyn,
 float rd,float rv_rd,float rvord,float rcv,float p0,float g,float bucket,int acrain,
 const float *c,float **fields,int *i_rainnc,const float *v,const float *s,
 const float *r,const float *prod) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
 // precip_to_MPAS precedes microphysics_to_MPAS; use complete scheme column.
 float precipw=0.f;
 for(int k=0;k<nz;++k) {
  int x=k*nc+i;float q=v[nz*nc+x];
  float rho_a=div(v[7*nz*nc+x],add(1.f,q));
  precipw=add(precipw,mul(mul(q,rho_a),v[9*nz*nc+x]));
 }
 SF(precipw)=precipw;SF(rainncv)=s[i];SF(rainnc)=add(SF(rainnc),SF(rainncv));
 if(acrain && bucket>0.f && SF(rainnc)>bucket){++i_rainnc[i];SF(rainnc)=sub(SF(rainnc),bucket);}
 SF(snowncv)=s[2*nc+i];SF(graupelncv)=s[4*nc+i];
 SF(sr)=div(add(s[2*nc+i],s[4*nc+i]),add(s[i],1.e-12f));
 SF(snownc)=add(SF(snownc),SF(snowncv));SF(graupelnc)=add(SF(graupelnc),SF(graupelncv));
 for(int k=0;k<top;++k) {
  int x=k*nc+i;float oldtheta=F(theta_m,k),olddry=div(oldtheta,add(1.f,mul(rvord,F(qv,k))));
  for(int n=0;n<6;++n) fields[SPECIES[n]][x]=v[(n+1)*nz*nc+x];
  F(theta_m,k)=mul(v[x],add(1.f,mul(rvord,F(qv,k))));
  F(rt_diabatic_tend,k)=div(sub(F(theta_m,k),oldtheta),dt_dyn);
  F(dtheta_dt_mp,k)=div(sub(div(F(theta_m,k),add(1.f,mul(rvord,F(qv,k)))),olddry),dt_dyn);
  F(rtheta_p,k)=sub(mul(F(rho_zz,k),F(theta_m,k)),F(rtheta_base,k));
  F(exner,k)=mp_pow(mul(mul(F(zz,k),div(rd,p0)),add(F(rtheta_p,k),F(rtheta_base,k))),rcv);
  F(pressure_p,k)=mul(mul(F(zz,k),rd),add(mul(F(exner,k),F(rtheta_p,k)),mul(sub(F(exner,k),F(exner_base,k)),F(rtheta_base,k))));
  for(int n=0;n<3;++n) fields[RADII[n]][x]=r[n*nz*nc+x];
  for(int n=0;n<2;++n) fields[PRODUCTION[n]][x]=prod[n*nz*nc+x];
  F(rthmpten,k)=div(sub(div(F(theta_m,k),add(1.f,mul(rv_rd,mp_max(F(qv,k), 0.f)))),F(rthmpten,k)),dt_dyn);
  for(int n=0;n<6;++n) fields[TENDENCIES[n]][x]=div(sub(fields[SPECIES[n]][x],fields[TENDENCIES[n]][x]),dt_dyn);
 }
 for(int k=top;k<nz;++k) {
  int x=k*nc+i;
  for(int n=1;n<6;++n) fields[SPECIES[n]][x]=0.f;
  F(rt_diabatic_tend,k)=0.f;F(rthmpten,k)=0.f;
  for(int n=0;n<6;++n) fields[TENDENCIES[n]][x]=0.f;
  for(int n=0;n<2;++n)fields[PRODUCTION[n]][x]=0.f;
  for(int n=0;n<3;++n)fields[RADII[n]][x]=c[20+n];
 }
 float tem1=sub(F(zgrid,1),F(zgrid,0)),tem2,rho2;
 float rho1=mul(mul(F(rho_zz,0),F(zz,0)),add(1.f,v[nz*nc+i]));
 if(nz>=2) {
  tem2=sub(F(zgrid,2),F(zgrid,1));
  rho2=mul(mul(F(rho_zz,1),F(zz,1)),add(1.f,v[(nz+1)*nc+i]));
 } else {
  // mpas_atmphys_interface.F:952-962 reads absent second-level storage.
  // All72 measured one-level strict-native children continue;16cases vary.
  // Leave those four padding words unassigned. No invented field value,
  // global out-of-bounds access, exclusion or parity claim is made.
  float unassigned_second[4];
  tem2=sub(unassigned_second[0],F(zgrid,1));
  rho2=mul(mul(unassigned_second[1],unassigned_second[2]),add(1.f,unassigned_second[3]));
 }
 float oldp=SF(surface_pressure);
 SF(surface_pressure)=mul(mul(mul(.5f,g),sub(F(zgrid,1),F(zgrid,0))),sub(rho1,div(mul(mul(.5f,sub(rho2,rho1)),tem1),add(tem1,tem2))));
 SF(surface_pressure)=add(add(SF(surface_pressure),F(pressure_p,0)),F(pressure_base,0));
 SF(tend_sfc_pressure)=div(sub(SF(surface_pressure),oldp),dt_dyn);
}
'''


#: Correct mode (CORRECTIONS_V842): exact text, corrected text, expected count.  WB3:
#: mpas_atmphys_driver_microphysics.F:614-617 divides the dry rho_p by (1+qv) a second time.  WB5: the
#: rain-out of the straddling sedimentation cell (mp_wsm6.F90:1984, :2254) books the below-ground part at
#: the flat cell mean although the remap moved it with the piecewise-linear profile; correct mode books it
#: with that profile, as the native correct-mode build's wsm6_plm_surface switch does.
CORRECT_MODE_SOURCE = {
    "WB5": (('  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:1984',
             '  th = div(sub(0.0f, za[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(dza[((k)-(1))*(1)], add(mul(qmi[((k)-(1))*(1)], th), mul(mul(mul(0.5f, sub(qpi[((k)-(1))*(1)], qmi[((k)-(1))*(1)])), th), th)))); // mp_wsm6.F90:1984 (WB5 correct mode)', 1),
            ('  precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(qa[((k)-(1))*(1)], sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:2254',
             '  th = div(sub(0.0f, za[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); precip[((i)-(1))*(1)] = add(precip[((i)-(1))*(1)], mul(dza[((k)-(1))*(1)], add(mul(qmi[((k)-(1))*(1)], th), mul(mul(mul(0.5f, sub(qpi[((k)-(1))*(1)], qmi[((k)-(1))*(1)])), th), th)))); // mp_wsm6.F90:2254 (WB5 correct mode)', 1)),
    "WB3": (('float rho_a=div(v[7*nz*nc+x],add(1.f,q));', 'float rho_a=v[7*nz*nc+x];', 1),),
}


#: Fast numerics (the default; owner ruling 2026-10-16 "strict is opt-in"). THE BREAKAGE THIS
#: PREVENTS: mesoscale_reference at 0.210 s/step against 0.0638 for convection_permitting on the
#: same RTX PRO 6000 (x1.2562 24 h fast, fix/hex-wsm6-speed 0336dbf8), 3.3x against the 1.2x bar,
#: because fast WSM6 still ran the strict operator wrappers (a NaN payload test on both operands
#: and a round-to-nearest intrinsic that forbids contraction, on every add/sub/mul/div) and
#: glibc's binary32 exp/log/log10/pow replicas, which evaluate in binary64 (1/64 rate on RTX-class
#: cards). Fast mode compiles plain C++ operators (same operand types and promotion, contraction
#: allowed by --fmad=true) and CUDA's binary32 libm (2-4 ulp), as Thompson's _fast_support does.
#: Strict text stays byte for byte what the bitwise proofs measured (and its kernel-pack key).
_FAST_OPERATORS = (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/"))
_FAST_MATHS = (
    ("__device__ float mp_exp(float x) {return thompson_aa_expf(x);}",
     "__device__ __forceinline__ float mp_exp(float x) {return expf(x);}"),
    ("__device__ float mp_log(float x) {return thompson_aa_logf(x);}",
     "__device__ __forceinline__ float mp_log(float x) {return logf(x);}"),
    ("__device__ float mp_log10(float x) {return ubm_log10f(x);}",
     "__device__ __forceinline__ float mp_log10(float x) {return log10f(x);}"),
    ("__device__ float mp_pow(float x,float y) {return tc_pow(x,y);}",
     "__device__ __forceinline__ float mp_pow(float x,float y) {return powf(x,y);}"),
)


def _fast_arithmetic(arithmetic: str) -> str:
    """Plain operators for the four scalar wrappers; complex and integer overloads unchanged."""
    import re
    for name, op in _FAST_OPERATORS:
        pattern = (r"template<class A,class B> __device__ auto " + name
                   + r"\(A a,B b\) \{\n using P=decltype\(a" + re.escape(op) + r"b\);\n.*?\n else return a"
                   + re.escape(op) + r"b;\n\}\n")
        arithmetic, count = re.subn(
            pattern, f"template<class A,class B> __device__ __forceinline__ auto {name}(A a,B b) "
                     f"{{return a{op}b;}}\n", arithmetic, flags=re.S)
        if count != 1:
            raise ValueError(f"WSM6 fast arithmetic: expected one scalar {name} wrapper, found {count}")
    return arithmetic


def _fast_maths(header: str) -> str:
    """CUDA binary32 exp/log/log10/pow for the shared provider's float entries (binary64 and complex kept)."""
    present = [old in header for old, _ in _FAST_MATHS]
    if not any(present):
        return header  # a stand-in provider (source-shape tests); nothing to replace
    for (old, new), here in zip(_FAST_MATHS, present):
        if not here or header.count(old) != 1:
            raise ValueError(f"WSM6 fast maths: expected one {old}")
        header = header.replace(old, new)
    return header


def cuda_source(device_math_header: str, *, max_levels=MAX_LEVELS, corrections=(), numerics=None) -> str:
    """Assemble the audited source with the campaign's shared math definitions.

    Assembly for mp_wsm6.F90:213-1468 (orchestration only).
    The provider must export mp_exp/log/log10/sqrt/sin/cos/atan(float/double),
    mp_pow(float,float)/mp_pow(double,double), mp_log(cx), mp_sqrt(cx/cx4),
    mp_abs(cx), and mp_complex_div(cx/cx4,cx/cx4). cx is complex<double>,
    cx4 complex<float>. Scalar sqrt also follows the provider's identity path.
    ``numerics`` lets the offline pack compose both modes in one process (_FAST_OPERATORS).
    """
    if not isinstance(device_math_header, str) or not device_math_header.strip():
        raise RuntimeError("WSM6 requires the shared RKIND/glibc CUDA math header")
    if isinstance(max_levels, bool) or not isinstance(max_levels, int) or not 2 <= max_levels <= MAX_LEVELS:
        raise ValueError(f"WSM6 buffer capacity must be 2..{MAX_LEVELS} levels")
    # Native automatic arrays follow the call's level bounds. Reserving all
    # 256 levels for every thread made even a two-level launch run out of
    # memory on the actual RTX4090 replay. Only storage capacity changes;
    # indices, loop bounds, statements and scalar operation order are retained.
    arithmetic = CUDA_ARITHMETIC.replace('#define WSM6_KMAX 256',
                                         f'#define WSM6_KMAX {max_levels}')
    mode = _hex_numerics.mode() if numerics is None else _hex_numerics.normalise(numerics)
    if mode == "fast":
        arithmetic = _fast_arithmetic(arithmetic)
        device_math_header = _fast_maths(device_math_header)
    modules, driver = CUDA_MODULES, CUDA_DRIVER
    if corrections:
        # The cooperative source clones the sedimentation routines. Apply
        # WB5 to their one original before cloning, retaining the count-one
        # guards. Applying it after cloning refused every fast/correct run
        # with "WB5 occurs 2 times, not 1" before the first physics step.
        from .driver_v842 import correct_source
        modules = correct_source(modules, {"WB5": CORRECT_MODE_SOURCE["WB5"]},
                                 corrections, "wsm6_v842")
        driver = correct_source(driver, {"WB3": CORRECT_MODE_SOURCE["WB3"]},
                                corrections, "wsm6_v842")
    if coop_columns(mode):
        scheme = modules + _coop_functions(modules) + CUDA_COOP_COLUMNS
    else:
        scheme = _interleaved(modules + CUDA_COLUMNS)
    text = (arithmetic.replace("/* WSM6_SHARED_DEVICE_MATH */", device_math_header)
            + scheme + driver + _state_snapshot_source())
    return text


#: Column workspace layout (WSM6 speed, 2026-10-29). The automatic arrays of every column used
#: to sit in one Wsm6Workspace struct per thread, so the 32 lanes of a warp touched 32 cache
#: lines ~45 KB apart on every array access, and the launcher ran 1,024 columns per launch
#: (3 serial launches of 32 blocks for x1.2562). THE BREAKAGE THIS PREVENTS: mesoscale_reference
#: at 0.50 s/step against 0.064 s/step for convection_permitting on the same RTX PRO 6000
#: (x1.2562 24 h fast, ffa87990), against the 1.2x bar. Now a warp's 32 columns share one block
#: of the workspace in which every struct word is stored 32 times, lane fastest (the interleaving
#: CUDA local memory uses): word w of lane l lives at block + w*32 + l. Array element k of a
#: column is then one coalesced access for the warp. Only storage moves: every statement,
#: index expression and operand order of the audited text is unchanged, and the per-column
#: native state stays one Wsm6State per column after the interleaved blocks.
WS_LANES = 32
_WS_SUPPORT = r'''
#ifdef __CUDA_ARCH__
#define WSM6_WS_LANES 32
template<class T> struct WsP {
 T* p;
 __device__ __forceinline__ WsP():p(nullptr) {}
 __device__ __forceinline__ WsP(decltype(nullptr)):p(nullptr) {}
 __device__ __forceinline__ explicit WsP(T* q):p(q) {}
 template<class U> __device__ __forceinline__ WsP(WsP<U> o):p(o.p) {}
 __device__ __forceinline__ T& operator[](int k) const {return p[k*WSM6_WS_LANES];}
 __device__ __forceinline__ WsP operator+(int n) const {return WsP(p+n*WSM6_WS_LANES);}
 __device__ __forceinline__ bool operator==(decltype(nullptr)) const {return p==nullptr;}
 __device__ __forceinline__ bool operator!=(decltype(nullptr)) const {return p!=nullptr;}
};
struct Wsm6Workspace;
// Lane l's view of the array at byte offset `offset` of the struct, in the warp block `block`.
template<class T> __device__ __forceinline__ WsP<T> wsm6_view(Wsm6Workspace *block,unsigned long long offset) {
 return WsP<T>((T*)((unsigned char*)block+offset*WSM6_WS_LANES)+(threadIdx.x&(WSM6_WS_LANES-1)));
}
#else
template<class T> using WsP = T*;
#endif
// Byte offset of a Wsm6Workspace member. NVRTC 13.4 takes the builtin offsetof (the compiler the
// interleaved layout was proven with, kept as is); NVRTC 12.9, 13.0 and 13.2 reject it ("type name is
// not allowed"): the CUDA 12 pip, driver-580 system and DA image compilers take the member address.
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
#define WSM6_OFFSET(member) __builtin_offsetof(Wsm6Workspace,member)
#else
#define WSM6_OFFSET(member) ((unsigned long long)&(((Wsm6Workspace*)0)->member))
#endif
'''


def _interleaved(text: str) -> str:
    """Store the column workspace lane-interleaved; storage only, statements unchanged (WS_LANES)."""
    import re
    marker = "struct Wsm6Workspace;\nstruct Wsm6State {"
    if text.count(marker) != 1 or text.count("  Wsm6State state;\n") != 1:
        raise ValueError("WSM6 workspace declarations not found exactly once")
    text = text.replace(marker, _WS_SUPPORT + marker, 1).replace("  Wsm6State state;\n", "", 1)
    binding = re.compile(r"^(float|double|bool) \*(\w+)=S\.workspace->(\w+);$", re.M)
    text, bound = binding.subn(
        lambda m: (f"WsP<{m.group(1)}> {m.group(2)}=wsm6_view<{m.group(1)}>"
                   f"(S.workspace,WSM6_OFFSET({m.group(3)}));"), text)
    if bound != 132 or "S.workspace->" in text:
        raise ValueError(f"WSM6 workspace bindings changed ({bound}); reconcile the interleaved layout")

    # nislfv_rain_plm/plm6 write their surface precipitation into the caller's one-word
    # automatic arrays (delqrs1[1] ... in mp_wsm6_run), which are not workspace arrays.
    scalar_outputs = {"precip", "precip1", "precip2"}

    def params(m):
        line = m.group(0)
        line = re.sub(r"\bconst (float|double|bool) \*(\w+)", r"WsP<const \1> \2", line)
        return re.sub(r"(?<![\w<])(float|double|bool) \*(\w+)",
                      lambda p: p.group(0) if p.group(2) in scalar_outputs else f"WsP<{p.group(1)}> {p.group(2)}",
                      line)
    text, signatures = re.subn(r"^__device__ [^\n]*\(Wsm6State &S,[^\n]*$", params, text, flags=re.M)
    if not signatures:
        raise ValueError("WSM6 scheme signatures not found")
    old = (" Wsm6Workspace &B=workspace[i-first];\n"
           " float *local=B.column_local,*surface=B.column_surface,*radii=B.column_radii;\n"
           " float *radar=B.column_radar,*production=B.column_production;\n")
    view = lambda name: f"wsm6_view<float>(B,WSM6_OFFSET(column_{name}))"
    new = (" const int wl=i-first;\n"
           " Wsm6Workspace *B=(Wsm6Workspace*)((unsigned char*)workspace"
           "+(unsigned long long)(wl/WSM6_WS_LANES)*WSM6_WS_LANES*sizeof(Wsm6Workspace));\n"
           f" WsP<float> local={view('local')},surface={view('surface')},radii={view('radii')};\n"
           f" WsP<float> radar={view('radar')},production={view('production')};\n")
    state_old = " Wsm6State &S=B.state;S=Wsm6State{};S.workspace=&B;\n"
    state_new = (" Wsm6State &S=((Wsm6State*)((unsigned char*)workspace"
                 "+(unsigned long long)gridDim.x*WSM6_WS_LANES*sizeof(Wsm6Workspace)))[wl];"
                 "S=Wsm6State{};S.workspace=B;\n")
    production_old = "production_on?production:nullptr,production_on?production+nz:nullptr"
    production_new = "production_on?production:WsP<float>(),production_on?production+nz:WsP<float>()"
    for before, after in ((old, new), (state_old, state_new), (production_old, production_new)):
        if text.count(before) != 1:
            raise ValueError("WSM6 column kernel prologue changed; reconcile the interleaved layout")
        text = text.replace(before, after, 1)
    return text


#: Lanes over levels (fast numerics; WSM6 speed step 2, 2026-10-29). THE BREAKAGE THIS PREVENTS: one
#: thread per column left x1.2562 (2,562 columns) at 81 warps on a 188-SM card, so every WSM6 call took
#: the latency of one column run serially through 15 cloud sub-steps (mesoscale_reference 3.3x
#: convection_permitting). In fast numerics one warp now runs one column: every level loop of
#: mp_wsm6_run and slope_wsm6 that touches only its own level (checked below, refused otherwise) runs
#: its levels across the 32 lanes; column statements that write shared memory (sedimentation, surface
#: precipitation, the native state) run on lane 0 and their scalars are broadcast; statements that only
#: form scalars run on every lane. Every statement, operand order and loop body is the audited text;
#: the column workspace is the native one-struct-per-column layout, so a level is one coalesced access.
#: Strict numerics keeps the column-per-thread kernel (the bitwise proofs) unless coop_columns says so.
COOP_LANES = 32
_COOP = None  # None: fast numerics only. True/False: forced (the strict coop-vs-serial identity check).
_COOP_FUNCTIONS = ("slope_wsm6", "slope_rain", "slope_snow", "slope_graup",
                   "nislfv_rain_plm", "nislfv_rain_plm6", "mp_wsm6_run", "refl10cm_wsm6")
#: Device functions a level loop may call: they read the native state and write only their arguments
#: (mp_radar.F90:192-673; _coop_functions refuses one that writes a state member or the workspace).
_PURE_HELPERS = ("rayleigh_soak_wetgraupel", "get_m_mix_nested", "get_m_mix", "m_complex_maxwellgarnett",
                 "m_complex_water_ray", "m_complex_ice_maetzler")
_LEVEL_LOOP = re.compile(r"^for \(k=([^;]+); k<=([^;]+); k\+=1\) \{$|^for \(k=([^;]+); k>=([^;]+); k\+=\(-1\)\) \{$")
_SLICE_LOOP = re.compile(r"^for\(int slice0=1;slice0<=(\w+);\+\+slice0\) ([^{}]*;)$")


def coop_columns(mode: str) -> bool:
    """Whether the column kernel runs one warp per column (lanes over levels).

    Default: fast numerics only. WOOF_HEX_WSM6_COOP=1/0 forces it either way, for the identity check
    that a strict coop forecast equals the strict column-per-thread forecast word for word.
    """
    import os
    if _COOP is not None:
        return bool(_COOP)
    forced = os.environ.get("WOOF_HEX_WSM6_COOP", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return mode == "fast"


def _code(line: str) -> str:
    return line.split("//", 1)[0]


def _statements(lines):
    """Split lines at one brace depth into whole statements (each a list of lines)."""
    out, current, depth = [], [], 0
    for line in lines:
        current.append(line)
        code = _code(line)
        depth += code.count("{") - code.count("}")
        if depth < 0:
            raise ValueError("WSM6 coop: unbalanced braces")
        if depth == 0:
            out.append(current)
            current = []
    if current:
        raise ValueError("WSM6 coop: statement not closed")
    return out


def _coop_function(text: str, name: str, calls) -> str:
    """The lane-cooperative copy ``name_coop(..., int lane)`` of one device function definition."""
    import re
    m = re.search(r"^__device__ (\w+) " + name + r"\((Wsm6State &S[^\n]*)\) \{[^\n]*\n", text, re.M)
    if m is None or len(re.findall(r"^__device__ \w+ " + name + r"\(Wsm6State &S[^\n]*\) \{", text, re.M)) != 1:
        raise ValueError(f"WSM6 coop: one definition of {name} expected")
    rest = text[m.end():].split("\n")
    body, depth = [], 1
    for line in rest:
        depth += _code(line).count("{") - _code(line).count("}")
        if depth == 0:
            break
        body.append(line)
    else:
        raise ValueError(f"WSM6 coop: {name} not closed")
    declaration = re.compile(r"^(#|//|float\b|int\b|bool\b|double\b|const char\b|cx\b|auto \w+ = \[&\])")
    first = next(i for i, line in enumerate(body) if line.strip() and not declaration.match(line.strip()))
    decls, executable = body[:first], body[first:]
    params = m.group(2)
    shared = set(re.findall(r"(?:float|double|bool|int) \*(\w+)", params))
    shared |= set(re.findall(r"^(?:float|double|bool|int) \*(\w+)=S\.workspace->", "\n".join(decls), re.M))
    scalars = {n: t for t, n in re.findall(r"^(float|int|bool|double) (\w+)(?: = [^;\[]*)?;$",
                                           "\n".join(line.strip() for line in decls), re.M)}
    scalars.update({n: t for t, n in re.findall(r"(int|float|bool) &(\w+)", params)})
    local_arrays = {n for n in re.findall(r"^(?:float|int|bool|double) (\w+)\[", "\n".join(
        line.strip() for line in decls), re.M)} - shared
    assign = lambda n: re.compile(r"(?:^|[;{]|\)\s*)\s*" + re.escape(n) + r"\s*(?:\[[^\n=]*\])?\s*=[^=]")

    def subscripts(code, array):
        """Every subscript text of array[...] in code (balanced brackets)."""
        found, start = [], 0
        token = re.compile(r"\b" + re.escape(array) + r"\[")
        while True:
            m_ = token.search(code, start)
            if m_ is None:
                return found
            depth_, j = 1, m_.end()
            while depth_:
                depth_ += {"[": 1, "]": -1}.get(code[j], 0)
                j += 1
            found.append(code[m_.end():j - 1])
            start = j

    def level_loop(statement):
        """The lanes-over-levels header of a level loop that is safe to split, else None.

        Safe: no jumps; every array the loop writes is written and read only at the loop's own level
        (arrays it only reads may be read at any level); every scalar it writes is written before it is read.
        """
        head = _code(statement[0]).strip()
        one_line = _SLICE_LOOP.match(head)
        loop = _LEVEL_LOOP.match(head)
        if one_line and one_line.group(1) == "im":
            return None  # im is the column count (1 per call), not a level loop
        if one_line:
            var, high, body = "slice0", one_line.group(1), [one_line.group(2)]
        elif loop:
            var = "k"
            low, high = (loop.group(1), loop.group(2)) if loop.group(1) is not None else (loop.group(4), loop.group(3))
            body = [_code(line) for line in statement[1:]]
        else:
            return None
        code = "\n".join(body)
        if re.search(r"\bgoto\b|\bbreak\b|\bcontinue\b|\breturn\b|^\s*\w+:;", code, re.M):
            return None
        if set(re.findall(r"\b(\w+)\(S,", code)) - set(_PURE_HELPERS):
            return None
        own = re.compile(r"\(\(" + var + r"\)-\((?:kts|1)\)\)")
        written = {n for n in set(shared) | local_arrays for line in body
                   if re.search(r"(?:^|[;{]|\)\s*)\s*" + re.escape(n) + r"\s*\[", line) and assign(n).search(line)}
        private = set()  # per-thread arrays not indexed by level (dvec1[i]): each lane's own, treated as scalars
        for n in written:
            subs = subscripts(code, n)
            if n in local_arrays and not any(re.search(r"\b" + var + r"\b", sub) for sub in subs):
                private.add(n)
                continue
            for sub in subs:
                if not own.search(sub) or re.search(r"\b" + var + r"\b", own.sub("", sub)):
                    return None
        for n in list(scalars) + sorted(private):
            token = re.compile(r"\b" + re.escape(n) + r"\b")
            hits = [i for i, line in enumerate(body) if token.search(line)]
            if not hits or not any(assign(n).search(line) for line in body):
                continue
            line = body[hits[0]]
            target = re.compile(r"(?:^|(?<=[;{])|(?<=\)))\s*" + re.escape(n) + r"\s*(?:\[[^\n=]*\])?\s*=(?!=)")
            # its first line must write it and not also read it (allold = add(allold, ...) is a reduction)
            if not assign(n).search(line) or token.search(target.sub(" ", line)) or n == var:
                return None
        if one_line:
            return f"for(int slice0=1+lane;slice0<={high};slice0+={COOP_LANES}) {one_line.group(2)}"
        return f"for ({var}=({low})+lane; {var}<={high}; {var}+={COOP_LANES}) {{"

    used = {"k_loops": [], "lane0": [], "all": []}
    out = []

    def refs(lines, names):
        text_ = "\n".join(_code(line) for line in lines)
        return {n for n in names if re.search(r"\b" + re.escape(n) + r"\b", text_)}

    def writes(lines, names):
        return {n for n in names for line in lines if assign(n).search(_code(line))}

    def flush(group):
        if not group:
            return
        lines = [line for statement in group for line in statement]
        codes = "\n".join(_code(line) for line in lines)
        shared_write = bool(writes(lines, shared)) or bool(re.search(r"\bS\.\w+\s*=[^=]", codes)) \
            or bool(re.search(r"\b\w+\(S,", codes))
        if not shared_write:
            out.extend(lines)
            used["all"].append(lines)
            return
        out.append("  if (lane == 0) { // WSM6 coop: column statements on lane 0")
        out.extend(lines)
        out.append("  } __syncwarp();")
        for n in sorted(writes(lines, scalars)):
            if scalars[n] == "bool":
                out.append(f"  {n} = (bool)__shfl_sync(0xffffffffu, (int){n}, 0);")
            else:
                out.append(f"  {n} = __shfl_sync(0xffffffffu, {n}, 0);")
        used["lane0"].append(lines)

    def walk(lines):
        group = []
        for statement in _statements(lines):
            head = _code(statement[0]).strip()
            comment = statement[0][len(_code(statement[0])):]
            header = level_loop(statement)
            if header is not None:
                flush(group); group = []
                indent = statement[0][:len(statement[0]) - len(statement[0].lstrip())]
                out.append(f"{indent}{header} {comment} (lanes over levels)")
                out.extend(statement[1:])
                out.append("  __syncwarp();")
                used["k_loops"].append(statement)
                continue
            call = re.match(r"(\w+)\(S, (.*)\);$", head)
            if call and call.group(1) in calls:
                flush(group); group = []
                out.append(f"  {call.group(1)}_coop(S, {call.group(2)}, lane); __syncwarp(); " + comment)
                continue
            if re.match(r"^\w+:;$", head):  # a jump target stays on every lane, never inside a lane-0 block
                flush(group); group = []
                out.extend(statement)
                continue
            codes = "\n".join(_code(line) for line in statement)
            outward = set(re.findall(r"\bgoto (\w+);", codes)) - set(re.findall(r"\b(\w+):;", codes))
            container = (len(statement) > 2 and head.endswith("{")
                         and (head.startswith("for (loop=") or head.startswith("for (i=1; i<=im;") or head.startswith("for (ist=1; ist<=2;")
                              or head.startswith("if ("))
                         and any(_LEVEL_LOOP.match(_code(line).strip()) or _code(line).strip().split("(S,")[0] in calls
                                 for line in statement[1:]))
            if container:
                flush(group); group = []
                out.append(statement[0])
                walk(statement[1:-1])
                out.append(statement[-1])
                continue
            if outward:  # a jump out of the statement is taken by every lane together (uniform condition)
                if writes(statement, shared) or re.search(r"\bS\.\w+\s*=[^=]|\b\w+\(S,", codes):
                    raise ValueError(f"WSM6 coop: {name} jumps out of a statement that writes shared memory")
                flush(group); group = []
                out.extend(statement)
                used["all"].append(statement)
                continue
            group.append(statement)
        flush(group)

    walk(executable)
    # Lane-0 groups leave their per-thread arrays current on lane 0 only; level loops keep theirs per lane.
    lane0_arrays = set().union(*(writes(g, local_arrays) for g in used["lane0"])) if used["lane0"] else set()
    def reads(lines, names):
        found = set()
        for n in names:
            target = re.compile(r"(?:^|(?<=[;{])|(?<=\)))\s*" + re.escape(n) + r"\s*\[[^\n=]*\]\s*=(?!=)")
            if any(re.search(r"\b" + re.escape(n) + r"\b", target.sub(" ", _code(line))) for line in lines):
                found.add(n)
        return found

    for g in used["k_loops"] + used["all"]:
        clash = reads(g, lane0_arrays)
        if clash:
            raise ValueError(f"WSM6 coop: {name} reads lane-0 arrays {sorted(clash)} on every lane")
    for g in used["k_loops"]:
        mine = writes(g, local_arrays)
        for other in used["k_loops"] + used["all"] + used["lane0"]:
            if other is not g and refs(other, mine):
                raise ValueError(f"WSM6 coop: {name} per-level array {sorted(mine)} used outside its loop")
    # A scalar a level loop leaves behind holds each lane's own last level: no other statement may read it before
    # writing it (the native code never reads a loop's final value; refused if it would).
    for g in used["k_loops"]:
        for n in writes(g, scalars):
            target = re.compile(r"(?:^|(?<=[;{])|(?<=\)))\s*" + re.escape(n) + r"\s*=(?!=)")
            token = re.compile(r"\b" + re.escape(n) + r"\b")
            for other in used["k_loops"] + used["all"] + used["lane0"]:
                first = next((_code(line) for line in other if token.search(_code(line))), None)
                if other is not g and first is not None and (not target.search(first)
                                                             or token.search(target.sub(" ", first))):
                    raise ValueError(f"WSM6 coop: {name} reads {n} left behind by a level loop")
    header = f"// Lane-cooperative copy of {name} (COOP_LANES)\n__device__ {m.group(1)} {name}_coop({params}, int lane) {{"
    return "\n".join([header, *decls, *out, "}", ""])


@lru_cache(maxsize=4)
def _coop_functions(modules: str) -> str:
    """Lane-cooperative copies of the WSM6 level loops (COOP_LANES)."""
    for helper in _PURE_HELPERS:
        m = re.search(r"^__device__ \w+ " + helper + r"\(Wsm6State &S[^\n]*\) \{\n(.*?)^\}", modules, re.M | re.S)
        if m is None or re.search(r"\bS\.\w+\s*=[^=]|workspace", m.group(1)):
            raise ValueError(f"WSM6 coop: {helper} is not a pure helper")
    text, done = "\n// WSM6 lane-cooperative functions (fast numerics)\n", []
    for name in _COOP_FUNCTIONS:
        text += _coop_function(modules, name, tuple(done)) + "\n"
        done.append(name)
    return text


CUDA_COOP_COLUMNS = r'''
// MPAS-A v8.4.2 module_mp_wsm6.F:18-239; mpas_atmphys_driver_microphysics.F:437-461 (COOP_LANES).
// One warp per column: lane l moves levels l, l+32, ...; column statements run on lane 0.
// At most 144 registers, so 14 one-warp blocks fit an SM's 64K registers and x1.2562's 2,562 columns run in
// one wave on 188 SMs (166 registers gave 12 per SM: a second wave of 306 columns doubled the call).
extern "C" __global__ void __launch_bounds__(32, 14) wsm6_columns(int nz,int nc,int ns,float dt,int hail,
 const float *c,float *v,float *s,float *r,float *dbz,float *prod,const float *pi,
 int wrapper,int radii_on,int radar_on,int production_on,int *status,char *messages,int first,Wsm6Workspace *workspace,
 const Wsm6State *tmpl) {
 const int lane=threadIdx.x&31;
 const int i=first+blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);if(i>=ns)return;
 Wsm6Workspace &B=workspace[i-first];
 float *local=B.column_local,*surface=B.column_surface,*radii=B.column_radii;
 float *radar=B.column_radar,*production=B.column_production;
 for(int n=0;n<10;++n) for(int k=lane;k<nz;k+=32) local[n*nz+k]=v[(n*nz+k)*nc+i];
 if(lane==0) for(int n=0;n<7;++n) surface[n]=s[n*nc+i];
 for(int n=0;n<3;++n) for(int k=lane;k<nz;k+=32) radii[n*nz+k]=r[(n*nz+k)*nc+i];
 if(wrapper) for(int k=lane;k<nz;k+=32) local[k]=mul(local[k],pi[k*nc+i]);
 // The native initialisation (mp_wsm6.F90:73-189: 13 rgmma products of 10,000 terms and radar_init) depends only on
 // the constants and hail, so it runs once per card (wsm6_state_template) and every column starts from a copy.
 Wsm6State &S=B.state;int err=0;
 for(int w=lane;w<(int)(sizeof(Wsm6State)/sizeof(unsigned long long));w+=32)
   ((unsigned long long*)&S)[w]=((const unsigned long long*)tmpl)[w];
 __syncwarp();
 if(lane==0) S.workspace=&B;
 __syncwarp();
 mp_wsm6_run_coop(S,local,local+nz,local+2*nz,local+3*nz,local+4*nz,local+5*nz,local+6*nz,local+7*nz,local+8*nz,local+9*nz,dt,
 c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],
 surface,surface+1,surface+6,surface+2,surface+3,surface+4,surface+5,
 production_on?production:nullptr,production_on?production+nz:nullptr,1,1,1,nz,err,lane);
 __syncwarp();err=__shfl_sync(0xffffffffu,err,0);
 if(err) {if(lane==0)status[i]=err;return;}
 if(wrapper) for(int k=lane;k<nz;k+=32) {
   float theta=div(local[k],pi[k*nc+i]);v[k*nc+i]=theta;
   local[k]=mul(theta,pi[k*nc+i]);
 }
 __syncwarp();
 if(lane==0) {
 mp_wsm6_effectrad_run(S,radii_on,local,local+2*nz,local+3*nz,local+5*nz,local+7*nz,c[11],c[5],c[20],c[21],c[22],c[23],c[24],c[25],radii,radii+nz,radii+2*nz,1,1,1,nz,err);
 }
 __syncwarp();err=__shfl_sync(0xffffffffu,err,0);
 if(radar_on) {refl10cm_wsm6_coop(S,local+nz,local+4*nz,local+5*nz,local+6*nz,local,local+8*nz,radar,1,nz,lane);__syncwarp();}
 for(int n=wrapper?1:0;n<10;++n) for(int k=lane;k<nz;k+=32) v[(n*nz+k)*nc+i]=local[n*nz+k];
 if(lane==0) for(int n=0;n<7;++n) s[n*nc+i]=surface[n];
 for(int n=0;n<3;++n) for(int k=lane;k<nz;k+=32) r[(n*nz+k)*nc+i]=radii[n*nz+k];
 if(radar_on) for(int k=lane;k<nz;k+=32) dbz[k*nc+i]=radar[k];
 if(production_on) for(int n=0;n<2;++n) for(int k=lane;k<nz;k+=32) prod[(n*nz+k)*nc+i]=production[n*nz+k];
 if(lane==0) {
 if(messages) {
   // Fortran character assignment: module_mp_wsm6.F:223; mp_wsm6_effectRad.F90:190.
   int n=0;while(S.last_message[n] && n<256){messages[n*nc+i]=S.last_message[n];++n;}
   for(;n<256;++n)messages[n*nc+i]=' ';
 }
 status[i]=err;
 }
}
// MPAS-A v8.4.2 mp_wsm6.F90:73-189 (state_parameters, mp_wsm6_init, radar_init) once: the state every column copies.
static_assert(sizeof(Wsm6State)%sizeof(unsigned long long)==0,"Wsm6State copies in 8-byte words");
extern "C" __global__ void wsm6_state_template(const float *c,int hail,Wsm6State *out,int *status) {
 if(blockIdx.x || threadIdx.x)return;
 Wsm6State S{};state_parameters(S);int err=0;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],hail,err);
 S.workspace=nullptr;*out=S;*status=err;
}
'''


def _state_layout():
    """Typed replay metadata for mp_wsm6.F90:20-71; mp_radar.F90:35-70."""
    import re
    declarations = CUDA_MODULES.split('struct Wsm6State {', 1)[1].split('};', 1)[0]
    rows = []
    for line in declarations.splitlines():
        if not line.strip():
            continue
        if line.strip() == 'Wsm6Workspace *workspace;':
            continue
        kind, name, array = re.fullmatch(
            r'(float|double|int|cx|const char \*)\s*(\w+)(\[.*\])?;', line.strip()).groups()
        if name in ('last_message', 'workspace'):
            continue  # Lowered public character output, not native module state.
        count = 1
        for expression in re.findall(r'\[([()\d+]+)\]', array or ''):
            # Only pinned integer literal bounds and sums occur in this struct.
            count *= sum(int(part) for part in expression.replace('(', '').replace(')', '').split('+'))
        width = {'float': 4, 'double': 8, 'int': 4, 'cx': 16}.get(kind)
        length = width * count if width else (256 if name == 'radar_debug' else 20)
        rows.append((name, kind, length))
    return tuple(rows)


def _state_snapshot_source():
    """Initialization replay view; mp_wsm6.F90:73-191; mp_radar.F90:72-190."""
    branches = []
    for index, (name, kind, size) in enumerate(_state_layout()):
        if kind == 'const char *':
            copy = (f'int n=0;if(S.{name}){{while(n<{size} && S.{name}[n])'
                    f'{{out[n]=S.{name}[n];++n;}}}}'
                    f'for(;n<{size};++n)out[n]=S.{name}?32:0;')
        else:
            copy = (f'static_assert(sizeof(S.{name})=={size},"native state width");'
                    f'const unsigned char *src=(const unsigned char*)&S.{name};'
                    f'for(int n=0;n<{size};++n)out[n]=src[n];')
        branches.append(f'if(member=={index}){{{copy}return;}}')
    return '''
// MPAS mp_wsm6.F90:73-191; mp_radar.F90:72-190: typed initialized state.
extern "C" __global__ void wsm6_initialized_state(const float *c,int hail,int member,unsigned char *out) {
 if(blockIdx.x || threadIdx.x)return;
 Wsm6State S{};state_parameters(S);int err=0;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],hail,err);
''' + '\n'.join(branches) + '\n}\n'


def _card() -> int:
    """The calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return int(cp.cuda.Device().id)


def _state_kernel(device_math_header):
    return _state_kernel_on(_card(), device_math_header)


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _state_kernel_on(device, device_math_header):
    """Compile replay view of mp_wsm6.F90:73-191; mp_radar.F90:72-190."""
    # Initialization has no vertical arrays; avoid allocating storage for the
    # unused column entrypoints present in the same compilation unit.
    return _compile_cuda(cuda_source(device_math_header, max_levels=2),
                         ('wsm6_initialized_state',))[0]


def initialized_state_bytes(constants, *, hail_opt, device_math_header):
    """Expose device initialized native state for f1 replay, as typed byte buffers.

    mp_wsm6.F90:73-191; mp_radar.F90:72-190. Snapshot contains no expected
    native output. The same initialization runs in every production column.
    """
    import cupy as cp
    import numpy as np
    _array(constants, (26,), 'constants')
    kernel = _state_kernel(device_math_header)
    result = {}
    for index, (name, kind, size) in enumerate(_state_layout()):
        output = cp.empty(size, dtype=cp.uint8)
        kernel((1,), (1,), (constants, np.int32(hail_opt), np.int32(index), output))
        result[name] = output
    return result


def compiler_options(numerics=None):
    from pathlib import Path, PurePosixPath
    try:
        import cupy as cp
    except ImportError:
        # A pack inventory on a host with no CuPy (the CPU tier, aot_build.physics_inventory): CuPy's include root in
        # the portable form the recorded rows carry (aot.portable_options writes @SITE@ for site-packages), so the
        # row is the one a CuPy host records.  Every compile runs where CuPy is installed and takes the real path.
        include = PurePosixPath("@SITE@/cupy/_core/include")
    else:
        include = Path(cp.__file__).parent / "_core/include"
    return _hex_numerics.options((f"-I{include}", f"-I{include / 'cupy/_cccl/thrust'}",
               f"-I{include / 'cupy/_cccl/libcudacxx'}", "--std=c++17",
               "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"),
               numerics=numerics)


def _compile_cuda(source: str, names, modules=None):
    """Load the packaged direct-NVRTC image, preserving native subnormals.

    Every public entry uses extern C. Its literal symbol is also the lowered
    name, so wheel images and the fallback share the same function lookup.
    ``modules`` (a list) receives the loaded module, for later lookups of other entries.
    """
    import cupy as cp
    from ..cuda_backend.aot import direct_image
    image = direct_image(source, compiler_options(), cp.cuda.Device().compute_capability)
    module = cp.cuda.function.Module()
    module.load(image)
    if modules is not None:
        modules.append(module)
    return tuple(module.get_function(name) for name in names)


def _kernels(device_math_header: str, max_levels=MAX_LEVELS, corrections=()):
    return _kernels_on(_card(), device_math_header, max_levels, corrections)


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _kernels_on(device, device_math_header: str, max_levels=MAX_LEVELS, corrections=()):
    """Compile launcher entrypoints for mp_wsm6.F90:213-1468."""
    names = ("wsm6_columns", "wsm6_from_mpas", "wsm6_radar_to_mpas", "wsm6_to_mpas")
    modules = []
    kernels = _compile_cuda(cuda_source(device_math_header, max_levels=max_levels, corrections=corrections), names,
                            modules)
    _MODULE_OF[kernels[0].ptr] = modules[0]
    return kernels


_MODULE_OF = {}  # column kernel handle -> its loaded module (the state template comes from the same module)
_TEMPLATES = {}  # (column kernel handle, constants bytes, hail) -> device Wsm6State the columns copy


def _state_template(kernel, constants, host_constants, hail_opt, state_bytes):
    """The once-per-card initialised Wsm6State the lane-cooperative columns copy (CUDA_COOP_COLUMNS)."""
    import cupy as cp
    import numpy as np
    host = np.asarray(cp.asnumpy(constants) if host_constants is None else host_constants, dtype=np.float32)
    key = (kernel.ptr, host.tobytes(), int(hail_opt))
    if key not in _TEMPLATES:
        out = cp.empty(state_bytes, cp.uint8)
        err = cp.zeros(1, cp.int32)
        _MODULE_OF[kernel.ptr].get_function("wsm6_state_template")((1,), (1,), (constants, np.int32(hail_opt), out, err))
        value = int(err.get()[0])
        if value:
            raise RuntimeError(f"MPAS WSM6 status {value}")
        _TEMPLATES[key] = out
    return _TEMPLATES[key]


def _workspace_size_kernel(device_math_header, max_levels):
    return _workspace_size_kernel_on(_card(), device_math_header, max_levels)


def workspace_size_source(device_math_header, max_levels, numerics=None):
    return cuda_source(device_math_header, max_levels=max_levels, numerics=numerics) + r'''
// MPAS-A v8.4.2 mp_wsm6.F90:213-2444; automatic storage metadata only.
extern "C" __global__ void wsm6_workspace_size(unsigned long long *out) {
 if(blockIdx.x || threadIdx.x)return;out[0]=sizeof(Wsm6Workspace);out[1]=sizeof(Wsm6State);
}
'''


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _workspace_size_kernel_on(device, device_math_header, max_levels):
    """Storage metadata for mp_wsm6.F90:213-2444 automatic arrays."""
    return _compile_cuda(workspace_size_source(device_math_header, max_levels),
                         ('wsm6_workspace_size',))[0]


@lru_cache(maxsize=16)
def _workspace_size(device_math_header, max_levels):
    """Storage metadata for mp_wsm6.F90:213-2444 without reading weather arrays.

    Returns (bytes of one column's arrays, bytes of one column's native state); a launch
    holds the arrays of its warps lane-interleaved, then one state per column (WS_LANES).
    """
    import cupy as cp
    answer=cp.empty(2, cp.uint64)
    _workspace_size_kernel(device_math_header,max_levels)((1,),(1,),(answer,))
    arrays, state = (int(v) for v in answer.get())
    return arrays, state


#: Byte budget of one WSM6 workspace tile. THE BREAKAGE THIS PREVENTS: a whole-rank tile at
#: ~45 KB per 55-level column is 7.4 GB for a 164k-column rank (a 30 km globe over 4 cards),
#: the same out-of-memory the Thompson workspace budget names (hex-global-fit, 2026-10-27).
#: WOOF_HEX_WSM6_WORKSPACE_MIB overrides it. Columns are independent, so the tile changes
#: only how many launches a call takes, never a value.
WORKSPACE_BUDGET_MIB = 1536


def _tile_columns(ns, column_bytes):
    """Columns per launch: every solve column at once, capped by the byte budget, whole warps."""
    import os
    value = os.environ.get("WOOF_HEX_WSM6_WORKSPACE_MIB", "").strip().lower()
    mib = WORKSPACE_BUDGET_MIB if value in ("", "auto") else int(value)
    if mib < 64:
        raise ValueError("WOOF_HEX_WSM6_WORKSPACE_MIB must be at least 64")
    budget = max(WS_LANES, ((mib << 20) // column_bytes) // WS_LANES * WS_LANES)
    return min(-(-ns // WS_LANES) * WS_LANES, budget)


def _array(a, shape, name, dtype="float32"):
    """Validate native argument storage: mp_wsm6.F90:213-330."""
    import cupy as cp
    if not isinstance(a, cp.ndarray) or a.dtype != cp.dtype(dtype):
        raise TypeError(f"{name} must be a CUDA {dtype} array")
    if a.shape != shape or not a.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous with shape {shape}")
    return a


def columns(volume, surface, radii, reflectivity, production, constants, *, dt,
            hail_opt, device_math_header, exner=None, solve_cells=None,
            radii_on=True, radar_on=True, production_on=True, messages=None, corrections=(), host_constants=None):
    """Run complete MMM columns, or the MPAS wrapper when exner is supplied.

    volume[10,level,cell] follows VOLUME_NAMES; surface[7,cell] follows
    SURFACE_NAMES; radii[3,level,cell], production[2,level,cell], dBZ[level,cell].
    Volume component zero is temperature for direct MMM calls and dry theta
    for wrapper calls. Native wrapper (WRF_CHEM=0) leaves production unchanged;
    production_on must be False in wrapper mode. Optional radius/radar outputs
    retain their caller values when disabled. Halos beyond solve_cells are kept.
    MPAS sources: mp_wsm6.F90:213-1468; module_mp_wsm6.F:18-239.
    """
    import cupy as cp
    import numpy as np
    if len(volume.shape) != 3 or volume.shape[0] != 10:
        raise ValueError("volume must have ten components")
    _, nz, nc = volume.shape
    if not 1 <= nz <= MAX_LEVELS:
        raise ValueError(f"WSM6 column storage supports 1..{MAX_LEVELS} levels")
    ns = nc if solve_cells is None else int(solve_cells)
    if not 0 <= ns <= nc:
        raise ValueError("solve_cells must exclude only trailing halo cells")
    args = [_array(volume, (10,nz,nc), "volume"),
            _array(surface, (7,nc), "surface"),
            _array(radii, (3,nz,nc), "radii"),
            _array(reflectivity, (nz,nc), "reflectivity"),
            _array(production, (2,nz,nc), "production")]
    _array(constants, (26,), "constants")
    wrapper = exner is not None
    if wrapper and production_on:
        raise ValueError("MPAS WRF_CHEM=0 wrapper does not write production diagnostics")
    pi = _array(exner, (nz,nc), "exner") if wrapper else reflectivity
    if messages is not None:
        _array(messages, (256,nc), "messages", "uint8")
    status = cp.zeros(nc, dtype=cp.int32)
    if ns:
        # Native one-level sedimentation reads outside ww's logical bounds
        # (mp_wsm6.F90:1837,1841,2087,2091), but the strict module continues.
        # Whole native/device boundary measurements match for positive and
        # negative dt. Retain a two-level allocation without changing the
        # native logical bounds or assigning invented values to its padding.
        # Correct mode (CORRECT_MODE_SOURCE): the scheme comes from the corrected text.
        kernel = (_kernels(device_math_header, max(2, nz), tuple(corrections)) if corrections
                  else _kernels(device_math_header, max(2, nz)))[0]
        # MPAS runs independent im=1 columns (mp_wsm6.F90:213-327). Each launch
        # keeps the original array strides and global cell index; all columns,
        # outputs, and halo words retain their original ownership. One launch
        # takes every solve column unless the workspace byte budget caps the
        # tile (_tile_columns); 1,024-column launches left most of the card idle.
        coop = coop_columns(_hex_numerics.mode())
        arrays_bytes, state_bytes = _workspace_size(device_math_header, max(2, nz))
        batch = _tile_columns(ns, arrays_bytes + state_bytes)
        workspace = cp.empty(batch * (arrays_bytes + state_bytes), cp.uint8)
        first = 0
        while first < ns:
            stop = min(first + batch, ns)
            # One warp per column in fast numerics (COOP_LANES), one thread per column otherwise.
            launch = (((stop-first),), (COOP_LANES,)) if coop else ((((stop-first)+WS_LANES-1)//WS_LANES,), (WS_LANES,))
            try:
                kernel(*launch,
                    (np.int32(nz), np.int32(nc), np.int32(stop), np.float32(dt),
                     np.int32(hail_opt), constants, *args, pi, np.int32(wrapper),
                     np.int32(radii_on), np.int32(radar_on), np.int32(production_on), status,
                     messages if messages is not None else np.uintp(0), np.int32(first), workspace)
                    + ((_state_template(kernel, constants, host_constants, hail_opt, state_bytes),) if coop else ()))
            except cp.cuda.driver.CUDADriverError as error:
                # CUDA_ERROR_OUT_OF_MEMORY from cuLaunchKernel means the
                # launch was not submitted. The current capture failed here
                # while other queued contexts occupied the card; reduce only
                # independent-column concurrency, never retry executed work.
                # One warp is the smallest useful allocation with this block.
                if error.status != 2 or batch <= WS_LANES:
                    raise
                batch = max(WS_LANES, batch // 2 // WS_LANES * WS_LANES)
                continue
            first = stop
        # Read off the step path inside a forecast (deferred_checks_v842, U-MPASPHYS-1): a
        # forecast still stops on a nonzero status, at most one step later; a replay at once.
        from .deferred_checks_v842 import defer
        defer(status[:ns], "MPAS WSM6 status {value}", reduce="first")
    return status


def step(fields: Mapping[str, Any], constants, *, dt_microp, dt_dyn,
         hail_opt, mp_top_level, solve_cells, rvord, rcv, p0,
         bucket_rainnc, l_acrain, l_diags, has_reqc, has_reqi, has_reqs,
         device_math_header, corrections=(), host_constants=None, radar_kernel=None,
         relhum_kernel=None, relhum=None):
    """Full MPAS WSM6 driver boundary, in-place on explicit field bindings.

    Field names equal MPAS pool names except qv/qc/qi/qr/qs/qg, which bind
    registry species views. Every level field is [level,cell]; zgrid has one
    extra interface. Counters are int32 [cell]; other surface fields [cell].
    Scheme runs through kte; mp_top_level limits native writeback only.
    No physical state is retained outside the caller's fields.
    MPAS sources: mpas_atmphys_interface.F:598-1138;
    mpas_atmphys_driver_microphysics.F:437-461,505-740.
    """
    import cupy as cp
    import numpy as np
    nz, nc = fields["theta_m"].shape
    if not 1 <= nz <= MAX_LEVELS or not 1 <= mp_top_level <= nz:
        raise ValueError("MPAS column storage needs 1..256 levels and a valid mp_top_level")
    ns = int(solve_cells)
    if not 0 <= ns <= nc:
        raise ValueError("solve_cells must exclude only trailing halo cells")
    _array(constants, (26,), "constants")
    surface_fields = set(FIELD_NAMES[30:40]) | set(FIELD_NAMES[41:])
    arrays = [_array(fields[name], (nz+1,nc) if name == "zgrid" else
                     (nc,) if name in surface_fields else (nz,nc), name)
              for name in FIELD_NAMES]
    counter = _array(fields["i_rainnc"], (nc,), "i_rainnc", "int32")
    if not ns:
        return
    pointers = cp.asarray([a.data.ptr for a in arrays], dtype=cp.uint64)
    v=cp.empty((10,nz,nc),dtype=cp.float32);s=cp.empty((7,nc),dtype=cp.float32)
    r=cp.empty((3,nz,nc),dtype=cp.float32);prod=cp.empty((2,nz,nc),dtype=cp.float32)
    pi=cp.empty((nz,nc),dtype=cp.float32);raw=cp.empty((nz,nc),dtype=cp.float32)
    kernels=_kernels(device_math_header, max(2, nz)); grid=((ns+31)//32,);block=(32,)
    # Correct mode (CORRECT_MODE_SOURCE): the writeback kernel comes from the corrected text.
    to_mpas=_kernels(device_math_header, max(2, nz), tuple(corrections))[3] if corrections else kernels[3]
    # All constants are scalar RKIND. Read only metadata-sized constant table.
    c=cp.asnumpy(constants) if host_constants is None else host_constants
    ratio=np.float32(np.float32(c[4])/np.float32(c[3]))
    kernels[1](grid,block,(np.int32(nz),np.int32(nc),np.int32(ns),ratio,
                          pointers,v,s,r,prod,pi))
    status=columns(v,s,r,raw,prod,constants,dt=dt_microp,hail_opt=hail_opt,
        device_math_header=device_math_header,exner=pi,solve_cells=ns,
        radii_on=(has_reqc==has_reqi==has_reqs==1),radar_on=l_diags,production_on=False,host_constants=c,
        **({"corrections":tuple(corrections)} if corrections else {}))
    if l_diags:
        (radar_kernel or kernels[2])(grid,block,(np.int32(nz),np.int32(nc),np.int32(ns),pointers,raw,status))
        if relhum_kernel is not None:
            _array(relhum, (nz, nc), 'relhum')
            relhum_kernel(((nz*nc+127)//128,), (128,),
                          (v, pi, relhum, np.int32(nz), np.int32(nc), np.int32(ns)))
        elif fields.get("relhum") is not None:
            # mpas_atmphys_driver_microphysics.F:479-481,798-850 compute_relhum, after the scheme and before
            # precip_to_MPAS/microphysics_to_MPAS: the updated th_p (v[0]) and qv_p (v[1]), pres_p (v[8]), pi_p.
            # THE BREAKAGE THIS PREVENTS: on the resident route nothing else formed relhum for WSM6, so every
            # relhum word (and the eight relhum isobaric diagnostics) kept a stale value (x1.2562 24 h strict
            # mesoscale_reference: 140,910 of 140,910 words per history frame). eca0a789 added this; the
            # rel/hex-0.4.10 merge ffa87990 kept only the RRFS relhum_kernel branch and dropped it, and
            # ffa87990 measured 1,196,453 of 36,335,253 words different again, all relhum and relhum_*hPa.
            relhum = fields["relhum"]
            _array(relhum, (nz,nc), "relhum")
            relhum[:, ns:] = 0
            _relhum_module(_card()).get_function("wsm6_relhum")(((nz*ns+63)//64,), (64,), (
                v[0], pi, v[1], v[8], relhum, np.int32(nz), np.int32(nc), np.int32(ns)))
    to_mpas(grid,block,(np.int32(nz),np.int32(nc),np.int32(ns),np.int32(mp_top_level),
        np.float32(dt_dyn),np.float32(c[3]),ratio,np.float32(rvord),np.float32(rcv),
        np.float32(p0),np.float32(c[0]),np.float32(bucket_rainnc),np.int32(l_acrain),
        constants,pointers,counter,v,s,r,prod))


CUDA_HEIGHT_GUARD = r'''
// MPAS-A v8.4.2 mpas_atmphys_driver_microphysics.F:711-727; test-only defect classifier.
// Legacy helper name retained for the standalone test ABI; never refuses a production call.
extern "C" __global__ void wsm6_height_guard(int nz,int nc,int ns,const float *z,int *status) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ns)return;
 int kp=0;
 for(int k=0;k<nz;++k) {
  float zp=__fadd_rn(__fsub_rn(z[k*nc+i],z[i]),__fmul_rn(.5f,__fsub_rn(z[(k+1)*nc+i],z[k*nc+i])));
  if(zp<1000.f)kp=k;
 }
 status[i]=(kp+1>=nz);
}
'''


def register(registry, bindings_from_call: Callable, constants_from_call: Callable,
             device_math_header: str):
    """Register on f4's SlotRegistry without owning its files or pool bindings.

    Native driver argument bindings: mpas_atmphys_driver_microphysics.F:681-687.
    bindings_from_call(call) returns (field_mapping, step_keyword_options),
    including dt_microp/dt_dyn and MPAS flags. It binds the current time level,
    scalar registry indices and solve range from call.pools. The integrator
    supplies this small mapping after the native bridge publishes its schema.
    """
    cuda_source(device_math_header)  # reject a missing provider before registration
    def launch(call):
        """Dispatch mpas_atmphys_driver_microphysics.F:681-687."""
        fields, options = bindings_from_call(call)
        step(fields, constants_from_call(call),
             device_math_header=device_math_header, **options)
    registry.register("microphysics", "mp_wsm6", "hex-cuda", launch)
    return launch


# Native pool ownership: mpas_atmphys_interface.F:637-657,875-902;
# mpas_atmphys_driver_microphysics.F:593-607,706-709.
NATIVE_FIELD_POOLS = {
    **dict.fromkeys(("zz", "zgrid"), "mesh"),
    **dict.fromkeys(("theta_m", "rho_zz"), "state"),
    **dict.fromkeys(("exner", "pressure_base", "pressure_p", "rtheta_base",
                     "rtheta_p", "exner_base", "surface_pressure", "dtheta_dt_mp"), "diag"),
    **dict.fromkeys(("rt_diabatic_tend", "tend_sfc_pressure"), "tend"),
    **dict.fromkeys(("rthmpten", "rqvmpten", "rqcmpten", "rqimpten",
                     "rqrmpten", "rqsmpten", "rqgmpten"), "tend_physics"),
    **dict.fromkeys(("rainprod", "evapprod", "re_cloud", "re_ice", "re_snow",
                     "precipw", "rainnc", "rainncv", "snownc", "snowncv",
                     "graupelnc", "graupelncv", "sr", "refl10cm", "refl10cm_max",
                     "refl10cm_1km", "refl10cm_1km_max", "i_rainnc"), "diag_physics"),
}
# Exact compiled-constant argument aliases: mpas_atmphys_driver_microphysics.F:443-455.
NATIVE_CONSTANT_NAMES = (
    "gravity", "cp", "cpv", "R_d", "R_v", "svpt0", "ep_1", "ep_2",
    None, None, None, "epsilon", "xls", "xlv", "xlf", "rho_a", "rho_w",
    "cliq", "cice", "psat", "re_qc_bg", "re_qi_bg", "re_qs_bg",
    "re_qc_max", "re_qi_max", "re_qs_max",
)


def bind_native(provider, call, *, writeback_levels=None):
    """Bind f4's native pool transaction to the complete MPAS driver boundary.

    mpas_atmphys_interface.F:637-657,875-902;
    mpas_atmphys_driver_microphysics.F:437-461,593-607.
    The provider packs solve cells and selects state time level two before this
    call. Species views share its committed scalars allocation; no weather
    arithmetic or host field decoding occurs here. This callback owns from/to
    conversion. The seam's native pre_microphysics/post_microphysics hooks are
    strict CPU no-ops, as specified by mpas-physics-seam.md:170-175.
    """
    cp = provider.cp
    fields = {name: provider[pool][name] for name, pool in NATIVE_FIELD_POOLS.items()}
    scalars = provider["state"]["scalars"]
    for name in VOLUME_NAMES[1:7]:
        index = provider.session.layout.indices["state", "index_" + name]
        if not 1 <= index <= scalars.shape[0]:
            raise ValueError(f"native WSM6 species index {name} is not allocated")
        fields[name] = scalars[index - 1]
    # compute_relhum's target, mpas_atmphys_driver_microphysics.F:818 (bound as Kessler and Thompson bind it;
    # eca0a789, dropped by the ffa87990 merge).
    fields["relhum"] = provider["diag"]["relhum"]
    constants = provider["constants"]
    import numpy as np
    host_constants = np.asarray([constants[name] if name is not None else 0
                                 for name in NATIVE_CONSTANT_NAMES], dtype=np.float32)
    packed = cp.asarray(host_constants)
    scalars_native = provider["scalar_state"]
    options = dict(
        dt_microp=scalars_native["dt_microp"], dt_dyn=scalars_native["dt_dyn"],
        hail_opt=0,  # native module parameter: mpas_atmphys_driver_microphysics.F:91
        mp_top_level=(int(provider["tend_physics"]["mp_top_level"].item())
                      if writeback_levels is None else int(writeback_levels)),
        host_constants=host_constants,
        solve_cells=provider.session.layout.dimensions["nCellsSolve"],
        rvord=constants["rvord"], rcv=constants["rcv"], p0=constants["P0"],
        bucket_rainnc=provider["configs"]["config_bucket_rainnc"],
        l_acrain=bool(scalars_native["l_acrain"]), l_diags=bool(scalars_native["l_diags"]),
        has_reqc=int(scalars_native["has_reqc"]), has_reqi=int(scalars_native["has_reqi"]),
        has_reqs=int(scalars_native["has_reqs"]),
    )
    return {"fields": fields, "constants": packed, "options": options}


def cuda_entry(call):
    """Table-selected callback; mpas_atmphys_driver_microphysics.F:437-461.

    The shared math owner exports rkind_libm.wsm6_device_math_header(). An
    embedding may also inject that header in this SlotCall's options. It is
    launcher metadata, never an MPAS namelist/Registry config. An absent
    provider fails before any field mutation or commit.
    NativeDevicePoolViews supplies the bind_native result in call.pools.
    """
    header = call.options.get("wsm6_device_math_header")
    if header is None:
        from .. import rkind_libm
        factory = getattr(rkind_libm, "wsm6_device_math_header", None)
        if not callable(factory):
            raise RuntimeError("shared rkind_libm.wsm6_device_math_header is not published")
        header = factory()
    cuda_source(header)
    binding = call.pools
    # Correct mode: the applied findings this implementation has text for (CORRECTIONS_V842 declaration).
    from .driver_v842 import slot_corrections
    corrections = slot_corrections(call, CORRECT_MODE_SOURCE)
    return step(binding["fields"], binding["constants"],
                device_math_header=header, **binding["options"],
                **({"corrections": corrections} if corrections else {}))


# MPAS-Model-v8.4.2 module_mp_thompson.F:4597-4658 rslf/rsif and mpas_atmphys_driver_microphysics.F:798-850
# compute_relhum, the same words as kessler_v842.CUDA_RELHUM_SOURCE, on WSM6's [level,cell] column carriers.
WSM6_RELHUM_SOURCE = r"""
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
__device__ float wsm6_rslf(float p,float t) {
 const float c[9]={.611583699E03f,.444606896E02f,.143177157E01f,.264224321E-1f,.299291081E-3f,
  .203154182E-5f,.702620698E-8f,.379534310E-11f,-.321582393E-13f};
 float x=fmaxf(-80.0f,S(t,273.16f)), es=c[8];
 for(int i=7;i>=0;--i) es=A(c[i],M(x,es));
 es=fminf(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
__device__ float wsm6_rsif(float p,float t) {
 const float c[9]={.609868993E03f,.499320233E02f,.184672631E01f,.402737184E-1f,.565392987E-3f,
  .521693933E-5f,.307839583E-7f,.105785160E-9f,.161444444E-12f};
 float x=fmaxf(-80.0f,S(t,273.16f)), es=c[8];
 for(int i=7;i>=0;--i) es=A(c[i],M(x,es));
 es=fminf(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
extern "C" __global__ void wsm6_relhum(const float *th,const float *pii,const float *qv,const float *pres,
 float *relhum,int nz,int nc,int ns) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nz*ns)return;
 int k=n/ns,i=n-k*ns,x=k*nc+i;
 float t=M(th[x],pii[x]);
 float qvs=wsm6_rslf(pres[x],t);
 if(S(t,273.16f)<=0.0f) qvs=wsm6_rsif(pres[x],t);
 relhum[x]=M(D(qv[x],qvs),100.0f);
}
"""


# Per-card module cache, unbounded on purpose (cards x variants), as _kernels_on.
@lru_cache(maxsize=None)
def _relhum_module(device: int):
    """compute_relhum for WSM6; explicit round-to-nearest arithmetic, so fast and strict give the same words."""
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc
    options = _hex_numerics.options(("--std=c++17", "--fmad=false", "--ftz=false",
                                     "--prec-div=true", "--prec-sqrt=true"))
    image, _ = compile_using_nvrtc(WSM6_RELHUM_SOURCE, options, None, "wsm6_relhum_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module
