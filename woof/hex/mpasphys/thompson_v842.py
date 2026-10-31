# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""MPAS-A v8.4.2 Thompson full-column CUDA transcription.

Numerical authority: physics_wrf/module_mp_thompson.F and its imported
physics_mmm/mp_radar.F90, with MPAS single RKIND and FP64 lookup records.
Immutable coefficients come from strict GNU initialization of the pinned
source. Table decoding belongs to the Rust preparation owner; this module
accepts already validated, Fortran-ordered device arrays.

The launcher implements MPAS's own driver interface and the published f4
registration protocol. WOOF defaults remain unchanged; real-door integration
and whole-model acceptance are still required.
"""
from __future__ import annotations
from woof.hex.device_memory_release import shared_card_memory
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from typing import Mapping

import numpy as np

SCHEME = "mp_thompson"
BACKEND = "hex-cuda"
TABLE_FILE_SHA256 = {
    "MP_THOMPSON_QIautQS_DATA.DBL": "9d270e5364b0c3e0a1f80d3251d67e673206e7203fadc905a16774bd3f43acf1",
    "MP_THOMPSON_QRacrQG_DATA.DBL": "3762e7f77abde1248a64e650372dd71505e20478173bfc7ca04b8fda424b1872",
    "MP_THOMPSON_QRacrQS_DATA.DBL": "5880b9d3b6bbe936101ab1db5316659f5563136371c36da47b14c13d23fbf9ee",
    "MP_THOMPSON_freezeH2O_DATA.DBL": "081ced9efcfd7c4034616435fa9d29a36bc5cc1b6f0b0edfee56260ea73c9edf",
}
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MAX_LEVELS = 256

_SUPPORT = r'''

// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ bool tc_nan(float a) {return (__float_as_uint(a)&0x7fffffffU)>0x7f800000U;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ bool tc_nan(double a) {return ((unsigned long long)__double_as_longlong(a)&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ float tc_quiet(float a) {return __uint_as_float(__float_as_uint(a)|0x00400000U);}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ double tc_quiet(double a) {return __longlong_as_double(__double_as_longlong(a)|0x0008000000000000LL);}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ float tc_add(float a,float b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);float r=__fadd_rn(a,b);return tc_nan(r)?__uint_as_float(0xffc00000U):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ double tc_add(double a,double b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);double r=__dadd_rn(a,b);return tc_nan(r)?__longlong_as_double(0xfff8000000000000ULL):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ int tc_add(int a,int b) {return a+b;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
template<class A,class B> __device__ auto tc_add(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_add(T(a),T(b));}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ float tc_sub(float a,float b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);float r=__fsub_rn(a,b);return tc_nan(r)?__uint_as_float(0xffc00000U):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ double tc_sub(double a,double b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);double r=__dsub_rn(a,b);return tc_nan(r)?__longlong_as_double(0xfff8000000000000ULL):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ int tc_sub(int a,int b) {return a-b;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
template<class A,class B> __device__ auto tc_sub(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_sub(T(a),T(b));}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ float tc_mul(float a,float b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);float r=__fmul_rn(a,b);return tc_nan(r)?__uint_as_float(0xffc00000U):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ double tc_mul(double a,double b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);double r=__dmul_rn(a,b);return tc_nan(r)?__longlong_as_double(0xfff8000000000000ULL):r;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
__device__ int tc_mul(int a,int b) {return a*b;}
// MPAS module_mp_thompson.F:1400-3642; strict native SSE scalar arithmetic.
template<class A,class B> __device__ auto tc_mul(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_mul(T(a),T(b));}

// Scalar arithmetic for generated Thompson columns. The caller supplies
// shared MPAS device mathematics; there is no scheme-local libm fallback.
#ifndef THOMPSON_COLUMN_KMAX
#define THOMPSON_COLUMN_KMAX 64
#endif
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B> __device__ auto tc_min(A a,B b)->decltype(a+b) {return a<b?a:b;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B,class C> __device__ auto tc_min(A a,B b,C c)->decltype(a+b+c) {return tc_min(tc_min(a,b),c);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B> __device__ auto tc_max(A a,B b)->decltype(a+b) {return a>b?a:b;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B,class C> __device__ auto tc_max(A a,B b,C c)->decltype(a+b+c) {return tc_max(tc_max(a,b),c);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_div(float a,float b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);float r=__fdiv_rn(a,b);return tc_nan(r)?__uint_as_float(0xffc00000U):r;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_div(double a,double b) {if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);double r=__ddiv_rn(a,b);return tc_nan(r)?__longlong_as_double(0xfff8000000000000ULL):r;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ int tc_div(int a,int b) {return a/b;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B> __device__ auto tc_div(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_div(T(a),T(b));}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_abs(float a) {return __uint_as_float(__float_as_uint(a)&0x7fffffffU);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_abs(double a) {return __longlong_as_double(__double_as_longlong(a)&0x7fffffffffffffffLL);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ int tc_abs(int a) {return a<0?-a:a;}
// MPAS module_mp_thompson.F:2655-2665; SIGN/SNGL intrinsic lowering.
__device__ float tc_sign(float a,float b) {return __uint_as_float((__float_as_uint(a)&0x7fffffffU)|(__float_as_uint(b)&0x80000000U));}
// MPAS module_mp_thompson.F:2655-2665; SIGN/SNGL intrinsic lowering.
__device__ double tc_sign(double a,double b) {return copysign(tc_abs(a),b);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_sqrt(float a) {if(tc_nan(a))return tc_quiet(a);float r=__fsqrt_rn(a);return tc_nan(r)?__uint_as_float(0xffc00000U):r;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_sqrt(double a) {if(tc_nan(a))return tc_quiet(a);double r=__dsqrt_rn(a);return tc_nan(r)?__longlong_as_double(0xfff8000000000000ULL):r;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_exp(float a) {return mpas_expf(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_exp(double a) {return mpas_exp(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_log(float a) {return mpas_logf(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_log(double a) {return mpas_log(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_log10(float a) {return mpas_log10f(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_log10(double a) {return mpas_log10(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_pow(float a,float b) {float r=mpas_powf(a,b);if(!tc_nan(r))return r;if(tc_nan(a))return tc_quiet(a);if(tc_nan(b))return tc_quiet(b);return __uint_as_float(0xffc00000U);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_pow(double a,double b) {return mpas_pow(a,b);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class T> __device__ T tc_pow(T a,int b) {
    bool neg=b<0; unsigned int n=neg?0U-(unsigned int)b:(unsigned int)b;
    T y=(n&1U)?a:T(1);
    while(n>>=1U) {a=tc_mul(a,a);if(n&1U)y=tc_mul(y,a);}
    return neg?tc_div(T(1),y):y;
}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
template<class A,class B> __device__ auto tc_pow(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_pow(T(a),T(b));}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_atan(double a) {return mpas_atan(a);}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ float tc_atan(float a) {return mpas_atanf(a);}
// MPAS module_mp_thompson.F:2655-2665; SIGN/SNGL intrinsic lowering.
__device__ float tc_sngl(double a) {return (float)a;}
// MPAS module_mp_thompson.F:1400-3642; scalar operator/intrinsic lowering for the column statements.
__device__ double tc_dfloat(int a) {return (double)a;}
// MPAS module_mp_thompson.F:1196-1200; PRESENT intrinsic lowering.
template<class T> __device__ bool tc_present(T a) {return true;}
// MPAS module_mp_thompson.F:3243-3278; ANY intrinsic lowering.
__device__ bool tc_any(const bool* a,int lo,int hi) {for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; initialized strings select air/water/icewater/spheroidal.
struct cd { double re, im; };

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_make(double re, double im)
{
    cd c; c.re = re; c.im = im; return c;
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_add(cd a, cd b)
{
    return cd_make(a.re + b.re, a.im + b.im);
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_sub(cd a, cd b)
{
    return cd_make(a.re - b.re, a.im - b.im);
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_mul(cd a, cd b)
{
    return cd_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_div(cd a, cd b)
{
    double d = b.re * b.re + b.im * b.im;
    return cd_make((a.re * b.re + a.im * b.im) / d,
                   (a.im * b.re - a.re * b.im) / d);
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ double cd_abs(cd a)
{
    return mpas_hypot(a.re, a.im);
}

// Principal square root (Fortran SQRT on COMPLEX*16).
// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_tc_sqrt(cd a)
{
    double m = cd_abs(a);
    double re = tc_sqrt(fmax(0.5 * (m + a.re), 0.0));
    double im = tc_sqrt(fmax(0.5 * (m - a.re), 0.0));
    return cd_make(re, a.im < 0.0 ? -im : im);
}

// Principal logarithm (Fortran LOG on COMPLEX*16).
// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_log(cd a)
{
    return cd_make(tc_log(cd_abs(a)), mpas_atan2(a.im, a.re));
}

// MPAS physics_mmm/mp_radar.F90:192-292,345-589; COMPLEX*16 operator lowering.
__device__ __forceinline__ cd cd_scale(double s, cd a)
{
    return cd_make(s * a.re, s * a.im);
}

// ---- imported physics_mmm/mp_radar.F90 per-particle machinery -----------------------------

// MPAS mp_radar.F90:536-589 m_complex_maxwellgarnett on the only inclusion string
// radar_init configures ('spheroidal', :110-122; betas :576-577).  The
// Fortran's volume-closure/-999 error path (:559-565) is unreachable here:
// every call below passes fractions that sum to 1 by construction.
__device__ cd refl_maxwellgarnett(double vol1, double vol2, double vol3,
                                  cd m1, cd m2, cd m3)
{
    cd m1t = cd_mul(m1, m1);
    cd m2t = cd_mul(m2, m2);
    cd m3t = cd_mul(m3, m3);
    cd d2 = cd_sub(m2t, m1t);
    cd d3 = cd_sub(m3t, m1t);
    cd one = cd_make(1.0, 0.0);
    cd beta2 = cd_mul(cd_div(cd_scale(2.0, m1t), d2),
                      cd_sub(cd_mul(cd_div(m2t, d2),
                                    cd_log(cd_div(m2t, m1t))), one));
    cd beta3 = cd_mul(cd_div(cd_scale(2.0, m1t), d3),
                      cd_sub(cd_mul(cd_div(m3t, d3),
                                    cd_log(cd_div(m3t, m1t))), one));
    cd num = cd_add(cd_add(cd_scale(1.0 - vol2 - vol3, m1t),
                    cd_mul(cd_scale(vol2,beta2),m2t)),
                    cd_mul(cd_scale(vol3,beta3),m3t));
    cd den = cd_add(cd_add(cd_make(1.0 - vol2 - vol3,0.0),
                    cd_scale(vol2,beta2)),cd_scale(vol3,beta3));
    return cd_tc_sqrt(cd_div(num, den));
}

// MPAS mp_radar.F90:345-483 get_m_mix_nested on radar_init's Morrison string set --
// host='air', matrix='water', hostmatrix='icewater' (:110-122) -- which
// resolves to exactly two nested Maxwell-Garnett evaluations along the
// host='air' branch (:375-402): the ice-in-water inclusion mix
// (get_m_mix matrix='water', :511-512), then that mixture as 'ice'
// inclusions in the air host (get_m_mix matrix='ice', :508-509).
__device__ cd refl_m_mix_nested(cd m_a, cd m_i, cd m_w, double volair,
                                double volice, double volwater)
{
    double vol1 = volice / fmax(volice + volwater, 1.0e-10);   // MPAS mp_radar.F90:381
    double vol2 = 1.0 - vol1;                                  // MPAS mp_radar.F90:382
    cd mtmp = refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i);
    return refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                               mtmp, m_a, cd_scale(2.0, m_a));
}

// MPAS mp_radar.F90:192-292 rayleigh_soak_wetgraupel: Rayleigh backscattering cross
// section (m2) of one melting, water-coated ice particle of mass x_g.
__device__ double refl_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i, double generation_pi5, double generation_lamda4)
{
    double fm = fmin(fmax(fmelt, 0.0), 1.0);                   // MPAS mp_radar.F90:224
    double mra = fmin(fmax(meltratio_outside, 0.0), 1.0);      // MPAS mp_radar.F90:226
    mra = mra + (1.0 - mra) * fm;                              // MPAS mp_radar.F90:233
    double x_w = x_g * fm;                                     // MPAS mp_radar.F90:235
    double d_g = a_geo * tc_pow(x_g, b_geo);                      // MPAS mp_radar.F90:237
    if (d_g < 1.0e-12) return 0.0;                             // MPAS mp_radar.F90:239,288-289
    double vg = 3.1415926535897932384626434 / 6.0 * tc_pow(d_g,3);                   // MPAS mp_radar.F90:241
    double rhog = fmin(fmax(x_g / vg, 10.0), 900.0);           // MPAS mp_radar.F90:242
    vg = x_g / rhog;                                           // MPAS mp_radar.F90:243
    double grenz = 1.0 - rhog / 1000.0;                        // MPAS mp_radar.F90:245
    double volg;
    if (mra <= grenz) {                                        // MPAS mp_radar.F90:247-251
        volg = vg * (1.0 - mra * fm);
    } else {                                                   // MPAS mp_radar.F90:253-267
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    double d_large = tc_pow(6.0 / 3.1415926535897932384626434 * volg, (double)(1.0f / 3.0f));         // MPAS mp_radar.F90:269
    double volice = (x_g - x_w) / (volg * 900.0);              // MPAS mp_radar.F90:270
    double volwater = x_w / (1000.0 * volg);                   // MPAS mp_radar.F90:271
    double volair = 1.0 - volice - volwater;                   // MPAS mp_radar.F90:272
    cd m_core = refl_m_mix_nested(cd_make(1.0, 0.0), m_i, m_w,
                                  volair, volice, volwater);   // MPAS mp_radar.F90:276-278
    cd m2 = cd_mul(m_core, m_core);
    double kfac = cd_abs(cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                                cd_add(m2, cd_make(2.0, 0.0))));
    return tc_pow(kfac,2) * generation_pi5 * tc_pow(d_large,6) / generation_lamda4;             // MPAS mp_radar.F90:285-286
}

'''

_COLUMN = r'''
// MPAS module_mp_thompson.F:433-993; mp_radar.F90:72-189.
// Immutable coefficients capture strict GNU initialization; external lookup
// arrays retain native kinds and layouts. Nt_c and mu_c remain column inputs.
__device__ __constant__ double tc_xxds[50] = {0x1.ba409db297486p-14,0x1.ebb079752770dp-14,0x1.115387d5665e6p-13,0x1.2fe142ad6d844p-13,0x1.51d95692c1748p-13,0x1.779d809403699p-13,0x1.a19a6ab6f8cf9p-13,0x1.d048e4a0ce200p-13,0x1.02179f98cdd44p-12,0x1.1ef1677dab285p-12,0x1.3f04ccf611fb4p-12,0x1.62ae1a802bfa6p-12,0x1.8a53ebad47393p-12,0x1.b668545d00f11p-12,0x1.e76a28f903d41p-12,0x1.0ef335b0e2af3p-11,0x1.2d3cf0531fea0p-11,0x1.4ee96a0b0304cp-11,0x1.7459865fc6260p-11,0x1.9df8fd7ecb26ep-11,0x1.cc3f922c031bep-11,0x1.ffb26a57ce929p-11,0x1.1c72c71ce74e4p-10,0x1.3c3ec91f79b6bp-10,0x1.5f98b84a26d3dp-10,0x1.86e64bd092f83p-10,0x1.b29899ad9698ap-10,0x1.e32d5c0476ac2p-10,0x1.0c982d70b9635p-9,0x1.2a9e7f3542e0cp-9,0x1.4c000702c7f67p-9,0x1.711cd0bd3ecb9p-9,0x1.9a5fa4d3e7318p-9,0x1.c83f3b7fe38ffp-9,0x1.fb3f925b2e709p-9,0x1.19f9b413630ebp-8,0x1.397ef18262e6fp-8,0x1.5c8a32f2a4f82p-8,0x1.83804d36655c9p-8,0x1.aed15a98c9f98p-8,0x1.def9fd6aeec8dp-8,0x1.0a42634faa2efp-7,0x1.2805e23dc0717p-7,0x1.491d1eed7d837p-7,0x1.6de74f7f81201p-7,0x1.96ce4eba7ca80p-7,0x1.c447cc9dfcb0fp-7,0x1.f6d6a1011ebcap-7,0x1.17862205c08b5p-6,0x1.36c53861d22d4p-6};
__device__ __constant__ double tc_xxdg[50] = {0x1.be52c6d4d4e0bp-14,0x1.f9648b0c9dd5dp-14,0x1.1e23cff932616p-13,0x1.440271c8b315bp-13,0x1.6ee420edffc82p-13,0x1.9f72ac2b16e9fp-13,0x1.d66e5b77868e7p-13,0x1.0a5874b5b6693p-12,0x1.2d9870b7f02abp-12,0x1.5582b8057a06bp-12,0x1.82b55a30b2b62p-12,0x1.b5e35203d655fp-12,0x1.efd74a3fc3bd5p-12,0x1.18bb6013c913bp-11,0x1.3de2c8217d90bp-11,0x1.67f4fdb31b216p-11,0x1.979899f513f38p-11,0x1.cd8a428d880e8p-11,0x1.054fca51a4b10p-10,0x1.27e539615bdb2p-10,0x1.4f0e615b10661p-10,0x1.7b6655115131dp-10,0x1.ad9cad5844970p-10,0x1.e678405f4a72ap-10,0x1.136d1a88cc628p-9,0x1.37e0bf55e34bbp-9,0x1.612767286fb31p-9,0x1.8fe4851e6e59cp-9,0x1.c4d12e21bf0e1p-9,0x1.005f7ae866566p-8,0x1.224d963359ebdp-8,0x1.48b9453ca2efep-8,0x1.743aac8b1aefcp-8,0x1.a57e135a9f49cp-8,0x1.dd468dd15f31bp-8,0x1.0e3880bf712d4p-7,0x1.31fbc80be7278p-7,0x1.5a7abafb253fep-7,0x1.8855b5d8a8dfbp-7,0x1.bc424e122d453p-7,0x1.f70e214646e8ep-7,0x1.1cd101be97692p-6,0x1.4282cc91ff6b1p-6,0x1.6d31b586abf0ap-6,0x1.9d86c24ece8bbp-6,0x1.d4415736099dap-6,0x1.091d16837085ep-5,0x1.2c33559f20ba0p-5,0x1.53ee59e9c3495p-5,0x1.80eb77ce9713fp-5};
__device__ __constant__ double tc_xdts[50] = {0x1.77161f03b8d68p-17,0x1.a103e6f958e30p-17,0x1.cfa18da335f10p-17,0x1.01ba99bc1ebd0p-16,0x1.1e89fb9bee440p-16,0x1.3e91d17cbba40p-16,0x1.622e44991cba8p-16,0x1.89c5cb86e2e28p-16,0x1.b5ca5107dd508p-16,0x1.e6ba7bcf49940p-16,0x1.0e918d7584e60p-15,0x1.2cd05d7491c10p-15,0x1.4e70b4239fe30p-15,0x1.73d3521e19ef8p-15,0x1.9d63c8be3d3e0p-15,0x1.cb99af9dcbc90p-15,0x1.fef9fcafb9c00p-15,0x1.1c0c416875130p-14,0x1.3bcccd8e5ca20p-14,0x1.5f19fee6907a0p-14,0x1.865967fb87488p-14,0x1.b1fbf605cdee8p-14,0x1.e27f35d7f9700p-14,0x1.0c375e8e694d8p-13,0x1.2a32ddfb908e0p-13,0x1.4b885dc355db0p-13,0x1.7097c72b987d8p-13,0x1.99cbbc26dc8e0p-13,0x1.c79aca2426cc0p-13,0x1.fa88bf2ad43b0p-13,0x1.1994128bf2170p-12,0x1.390df39ff1ab0p-12,0x1.5c0c93994f538p-12,0x1.82f4a2f2b1680p-12,0x1.ae36138fa9020p-12,0x1.de4d5ad57ce60p-12,0x1.09e26be2abe10p-11,0x1.279b308f1b160p-11,0x1.48a680000f378p-11,0x1.6d636e051cc08p-11,0x1.963baf3de1ce0p-11,0x1.c3a4c93f09170p-11,0x1.f62164b6f3720p-11,0x1.172162af7b9a8p-10,0x1.365535f972f98p-10,0x1.5905f370ba5e0p-10,0x1.7f976b77487e8p-10,0x1.aa7896cb11678p-10,0x1.da24d5d1f3a40p-10,0x1.0792a9cb73498p-9};
__device__ __constant__ double tc_xdtg[50] = {0x1.bc14e866a2140p-17,0x1.f6dab983eb268p-17,0x1.1cb3e70a34428p-16,0x1.4261d7ca34128p-16,0x1.6d0c642a7d4a8p-16,0x1.9d5c8097b0790p-16,0x1.d4117dceb0db0p-16,0x1.0901ff3a518e8p-15,0x1.2c14a87960f38p-15,0x1.53cb9d6d8b830p-15,0x1.80c4226e5e400p-15,0x1.b3b04c25b0230p-15,0x1.ed59c0bf1f4e0p-15,0x1.17526b35e1998p-14,0x1.3c4a0dca10fd8p-14,0x1.66262b5a79fb0p-14,0x1.958c86de8ede0p-14,0x1.cb38d36df1978p-14,0x1.03ffcdc37b48cp-13,0x1.2668c559f03b8p-13,0x1.4d5f9349d7a20p-13,0x1.797e8302391e8p-13,0x1.ab744b9043188p-13,0x1.e406c37adda40p-13,0x1.120af81772098p-12,0x1.364fbe8d28898p-12,0x1.5f6154179d6a8p-12,0x1.8de259ac34ad0p-12,0x1.c28af63be1408p-12,0x1.fe2bafed72e80p-12,0x1.20d852ed55ae8p-11,0x1.47129b8afb4a0p-11,0x1.725c12b7b19a8p-11,0x1.a360222cc0320p-11,0x1.dae0e33cd3220p-11,0x1.0cdd0fb8e0b38p-10,0x1.30725b71ceb38p-10,0x1.58bd3cdfc5510p-10,0x1.865d421e79ea8p-10,0x1.ba0717221de28p-10,0x1.f48751296f8f0p-10,0x1.1b62cc6fcf5b8p-9,0x1.40e41fdb0a0c0p-9,0x1.6b5c27535fa78p-9,0x1.9b730f38289c8p-9,0x1.d1e745bf18448p-9,0x1.07c8368599540p-8,0x1.2ab15888706a8p-8,0x1.5239473e3d5a8p-8,0x1.7efc8cc80fdb0p-8};
__device__ __constant__ float tc_ta_ww[9] = {0x1.47ae140000000p-7f,0x1.02de000000000p-5f,0x1.99999a0000000p-4f,0x1.4395820000000p-2f,0x1.0000000000000p+0f,0x1.947ae20000000p+1f,0x1.4000000000000p+3f,0x1.f9999a0000000p+4f,0x1.9000000000000p+6f};
__device__ __constant__ float tc_ta_tk[7] = {0x1.e64ccc0000000p+7f,0x1.fa4ccc0000000p+7f,0x1.0726660000000p+8f,0x1.1126660000000p+8f,0x1.1b26660000000p+8f,0x1.2526660000000p+8f,0x1.2f26660000000p+8f};
__device__ __constant__ float tc_ta_na[7] = {0x1.4000000000000p+3f,0x1.f9999a0000000p+4f,0x1.9000000000000p+6f,0x1.3c00000000000p+8f,0x1.f400000000000p+9f,0x1.8b00000000000p+11f,0x1.3880000000000p+13f};
__device__ __constant__ double tc_t_nc[100] = {0x1.fc397d2e29693p+19,0x1.134b48c43cf34p+20,0x1.2a3dd352f6082p+20,0x1.431a0b96efa44p+20,0x1.5e08c2e2e9a8fp+20,0x1.7b36318d4d12ap+20,0x1.9ad23f8ae0103p+20,0x1.bd10d316ca88dp+20,0x1.e22a25e91018fp+20,0x1.052d90c3b435fp+21,0x1.1af2e1a4042f1p+21,0x1.3288c45548635p+21,0x1.4c15f25c83e36p+21,0x1.67c45f98cacedp+21,0x1.85c17f24e7439p+21,0x1.a63e8df6dd417p+21,0x1.c970e3b7d34d8p+21,0x1.ef924a591c26cp+21,0x1.0c70ae7b172b6p+22,0x1.22d0f74fa97d3p+22,0x1.3b0ebcdae8ba0p+22,0x1.5551cc48379bep+22,0x1.71c544172c402p+22,0x1.9097dae784d1fp+22,0x1.b1fc2c2bd9fe4p+22,0x1.d6290b40fd1e8p+22,0x1.fd59dd726f44dp+22,0x1.13e77dbfdec53p+23,0x1.2ae70da17073bp+23,0x1.43d161128c52bp+23,0x1.5ecf608f160a4p+23,0x1.7c0d5d84d527cp+23,0x1.9bbb5b175ba7bp+23,0x1.be0d5cf4ac06bp+23,0x1.e33bbcbc0d301p+23,0x1.05c1c341a246bp+24,0x1.1b936e890f542p+24,0x1.3336b33c6c333p+24,0x1.4cd260d9d958bp+24,0x1.6890830e6b719p+24,0x1.869ea6bee5fa3p+24,0x1.a72e24d196d2ap+24,0x1.ca7473341e1f9p+24,0x1.f0ab7ca22a1b9p+24,0x1.0d08ffdea29e7p+25,0x1.2375fb08ec365p+25,0x1.3bc181d943659p+25,0x1.5613781090f5fp+25,0x1.729714a5dddf0p+25,0x1.917b28ba73cf6p+25,0x1.b2f26c7814b82p+25,0x1.d733d2577ebc4p+25,0x1.fe7ae157f7177p+25,0x1.14840b5dfe1e0p+26,0x1.2b90a7f5cc246p+26,0x1.44891e9516a9dp+26,0x1.5f966eee0969fp+26,0x1.7ce503940740cp+26,0x1.9ca4fae8da851p+26,0x1.bf0a761e15f8fp+26,0x1.e44deecc5e17fp+26,0x1.065649d6914b2p+27,0x1.1c345687835adp+27,0x1.33e504d4f3eb1p+27,0x1.4d8f3a4298a05p+27,0x1.695d1a59088e2p+27,0x1.877c4bd5a2a50p+27,0x1.a81e439ed8fc5p+27,0x1.cb7895f7f1f8ap+27,0x1.f1c54e79926e0p+27,0x1.0da1a7afc0728p+28,0x1.241b5c640e2d5p+28,0x1.3c74ac4785f82p+28,0x1.56d591bd6736fp+28,0x1.73695c420fbb3p+28,0x1.925ef7875ef59p+28,0x1.b3e9387e8b4a4p+28,0x1.d83f30cde6866p+28,0x1.ff9c893b999e2p+28,0x1.1520f1d0e5ff1p+29,0x1.2c3aa28685515p+29,0x1.45414459958afp+29,0x1.605dee3fb63eep+29,0x1.7dbd24002a692p+29,0x1.9d8f1f4a6a06ep+29,0x1.c0081ee4574a8p+29,0x1.e560bc7218c78p+29,0x1.06eb24b238155p+30,0x1.1cd599d311433p+30,0x1.3493b956df97ep+30,0x1.4e4c7ed36cc62p+30,0x1.6a2a25ba5bcc4p+30,0x1.885a6eb05170ap+30,0x1.a90eeaabc757fp+30,0x1.cc7d4c56e087bp+30,0x1.f2dfc039de18cp+30,0x1.0e3aa61f7b1d9p+31,0x1.24c11b9630558p+31,0x1.3d283c5f3f1d1p+31,0x1.5798198d15441p+31};
__device__ __constant__ double tc_simpson[51] = {0x1.5555555555555p-2,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-1,0x1.5555555555555p+0,0x1.5555555555555p-2};
__device__ __constant__ float tc_sb[10] = {0x1.e7a67a0000000p-2f,-0x1.0470a80000000p-6f,0x1.53ebc00000000p-3f,0x1.e96c400000000p-8f,-0x1.27b2cc0000000p-13f,0x1.ee84ae0000000p-5f,0x1.4b599a0000000p-14f,0x1.376d540000000p-11f,0x0.0p+0f,-0x1.d4d8340000000p-9f};
__device__ __constant__ float tc_sa[10] = {0x1.442e840000000p+2f,-0x1.00a6ba0000000p-4f,-0x1.8424700000000p+1f,0x1.e2d1f20000000p-6f,-0x1.2ad81a0000000p-12f,0x1.400d1c0000000p-2f,0x1.abd1aa0000000p-13f,0x1.a34ca00000000p-9f,0x0.0p+0f,-0x1.055b8a0000000p-6f};
__device__ __constant__ float tc_r_s[28] = {0x1.4f8b580000000p-17f,0x1.4f8b580000000p-16f,0x1.f751040000000p-16f,0x1.4f8b580000000p-15f,0x1.a36e2e0000000p-15f,0x1.f751040000000p-15f,0x1.2599ee0000000p-14f,0x1.4f8b580000000p-14f,0x1.797cc40000000p-14f,0x1.a36e2e0000000p-14f,0x1.a36e2e0000000p-13f,0x1.3a92a40000000p-12f,0x1.a36e2e0000000p-12f,0x1.0624de0000000p-11f,0x1.3a92a40000000p-11f,0x1.6f00680000000p-11f,0x1.a36e2e0000000p-11f,0x1.d7dbf40000000p-11f,0x1.0624de0000000p-10f,0x1.0624de0000000p-9f,0x1.89374c0000000p-9f,0x1.0624de0000000p-8f,0x1.47ae140000000p-8f,0x1.89374c0000000p-8f,0x1.cac0840000000p-8f,0x1.0624de0000000p-7f,0x1.26e9780000000p-7f,0x1.47ae140000000p-7f};
__device__ __constant__ float tc_r_r[37] = {0x1.0c6f7a0000000p-20f,0x1.0c6f7a0000000p-19f,0x1.92a7380000000p-19f,0x1.0c6f7a0000000p-18f,0x1.4f8b580000000p-18f,0x1.92a7380000000p-18f,0x1.d5c3160000000p-18f,0x1.0c6f7a0000000p-17f,0x1.2dfd6a0000000p-17f,0x1.4f8b580000000p-17f,0x1.4f8b580000000p-16f,0x1.f751040000000p-16f,0x1.4f8b580000000p-15f,0x1.a36e2e0000000p-15f,0x1.f751040000000p-15f,0x1.2599ee0000000p-14f,0x1.4f8b580000000p-14f,0x1.797cc40000000p-14f,0x1.a36e2e0000000p-14f,0x1.a36e2e0000000p-13f,0x1.3a92a40000000p-12f,0x1.a36e2e0000000p-12f,0x1.0624de0000000p-11f,0x1.3a92a40000000p-11f,0x1.6f00680000000p-11f,0x1.a36e2e0000000p-11f,0x1.d7dbf40000000p-11f,0x1.0624de0000000p-10f,0x1.0624de0000000p-9f,0x1.89374c0000000p-9f,0x1.0624de0000000p-8f,0x1.47ae140000000p-8f,0x1.89374c0000000p-8f,0x1.cac0840000000p-8f,0x1.0624de0000000p-7f,0x1.26e9780000000p-7f,0x1.47ae140000000p-7f};
__device__ __constant__ float tc_r_i[64] = {0x1.b7cdfe0000000p-34f,0x1.b7cdfe0000000p-33f,0x1.49da7e0000000p-32f,0x1.b7cdfe0000000p-32f,0x1.12e0be0000000p-31f,0x1.49da7e0000000p-31f,0x1.80d43e0000000p-31f,0x1.b7cdfe0000000p-31f,0x1.eec7be0000000p-31f,0x1.12e0be0000000p-30f,0x1.12e0be0000000p-29f,0x1.9c511e0000000p-29f,0x1.12e0be0000000p-28f,0x1.5798ee0000000p-28f,0x1.9c511e0000000p-28f,0x1.e1094e0000000p-28f,0x1.12e0be0000000p-27f,0x1.353cd60000000p-27f,0x1.5798ee0000000p-27f,0x1.5798ee0000000p-26f,0x1.01b2b20000000p-25f,0x1.5798ee0000000p-25f,0x1.ad7f2a0000000p-25f,0x1.01b2b20000000p-24f,0x1.2ca5d00000000p-24f,0x1.5798ee0000000p-24f,0x1.828c0c0000000p-24f,0x1.ad7f2a0000000p-24f,0x1.ad7f2a0000000p-23f,0x1.421f600000000p-22f,0x1.ad7f2a0000000p-22f,0x1.0c6f7a0000000p-21f,0x1.421f600000000p-21f,0x1.77cf440000000p-21f,0x1.ad7f2a0000000p-21f,0x1.e32f0e0000000p-21f,0x1.0c6f7a0000000p-20f,0x1.0c6f7a0000000p-19f,0x1.92a7380000000p-19f,0x1.0c6f7a0000000p-18f,0x1.4f8b580000000p-18f,0x1.92a7380000000p-18f,0x1.d5c3160000000p-18f,0x1.0c6f7a0000000p-17f,0x1.2dfd6a0000000p-17f,0x1.4f8b580000000p-17f,0x1.4f8b580000000p-16f,0x1.f751040000000p-16f,0x1.4f8b580000000p-15f,0x1.a36e2e0000000p-15f,0x1.f751040000000p-15f,0x1.2599ee0000000p-14f,0x1.4f8b580000000p-14f,0x1.797cc40000000p-14f,0x1.a36e2e0000000p-14f,0x1.a36e2e0000000p-13f,0x1.3a92a40000000p-12f,0x1.a36e2e0000000p-12f,0x1.0624de0000000p-11f,0x1.3a92a40000000p-11f,0x1.6f00680000000p-11f,0x1.a36e2e0000000p-11f,0x1.d7dbf40000000p-11f,0x1.0624de0000000p-10f};
__device__ __constant__ float tc_r_g[28] = {0x1.4f8b580000000p-17f,0x1.4f8b580000000p-16f,0x1.f751040000000p-16f,0x1.4f8b580000000p-15f,0x1.a36e2e0000000p-15f,0x1.f751040000000p-15f,0x1.2599ee0000000p-14f,0x1.4f8b580000000p-14f,0x1.797cc40000000p-14f,0x1.a36e2e0000000p-14f,0x1.a36e2e0000000p-13f,0x1.3a92a40000000p-12f,0x1.a36e2e0000000p-12f,0x1.0624de0000000p-11f,0x1.3a92a40000000p-11f,0x1.6f00680000000p-11f,0x1.a36e2e0000000p-11f,0x1.d7dbf40000000p-11f,0x1.0624de0000000p-10f,0x1.0624de0000000p-9f,0x1.89374c0000000p-9f,0x1.0624de0000000p-8f,0x1.47ae140000000p-8f,0x1.89374c0000000p-8f,0x1.cac0840000000p-8f,0x1.0624de0000000p-7f,0x1.26e9780000000p-7f,0x1.47ae140000000p-7f};
__device__ __constant__ float tc_r_c[37] = {0x1.0c6f7a0000000p-20f,0x1.0c6f7a0000000p-19f,0x1.92a7380000000p-19f,0x1.0c6f7a0000000p-18f,0x1.4f8b580000000p-18f,0x1.92a7380000000p-18f,0x1.d5c3160000000p-18f,0x1.0c6f7a0000000p-17f,0x1.2dfd6a0000000p-17f,0x1.4f8b580000000p-17f,0x1.4f8b580000000p-16f,0x1.f751040000000p-16f,0x1.4f8b580000000p-15f,0x1.a36e2e0000000p-15f,0x1.f751040000000p-15f,0x1.2599ee0000000p-14f,0x1.4f8b580000000p-14f,0x1.797cc40000000p-14f,0x1.a36e2e0000000p-14f,0x1.a36e2e0000000p-13f,0x1.3a92a40000000p-12f,0x1.a36e2e0000000p-12f,0x1.0624de0000000p-11f,0x1.3a92a40000000p-11f,0x1.6f00680000000p-11f,0x1.a36e2e0000000p-11f,0x1.d7dbf40000000p-11f,0x1.0624de0000000p-10f,0x1.0624de0000000p-9f,0x1.89374c0000000p-9f,0x1.0624de0000000p-8f,0x1.47ae140000000p-8f,0x1.89374c0000000p-8f,0x1.cac0840000000p-8f,0x1.0624de0000000p-7f,0x1.26e9780000000p-7f,0x1.47ae140000000p-7f};
__device__ __constant__ float tc_ocg2[15] = {0x1.5555560000000p-5f,0x1.1111100000000p-7f,0x1.6c16c00000000p-10f,0x1.a019f80000000p-13f,0x1.a01a040000000p-16f,0x1.71de3c0000000p-19f,0x1.27e4f20000000p-22f,0x1.ae64560000000p-26f,0x1.1eed840000000p-29f,0x1.61245c0000000p-33f,0x1.9397480000000p-37f,0x1.ae7f480000000p-41f,0x1.ae7f480000000p-45f,0x1.952c760000000p-49f,0x1.6827620000000p-53f};
__device__ __constant__ float tc_ocg1[15] = {0x1.0000000000000p+0f,0x1.0000000000000p-1f,0x1.5555560000000p-3f,0x1.5555560000000p-5f,0x1.1111100000000p-7f,0x1.6c16c00000000p-10f,0x1.a019f80000000p-13f,0x1.a01a040000000p-16f,0x1.71de3c0000000p-19f,0x1.27e4f20000000p-22f,0x1.ae64560000000p-26f,0x1.1eed840000000p-29f,0x1.61245c0000000p-33f,0x1.9397480000000p-37f,0x1.ae7f480000000p-41f};
__device__ __constant__ float tc_nt_in[55] = {0x1.0000000000000p+0f,0x1.0000000000000p+1f,0x1.8000000000000p+1f,0x1.0000000000000p+2f,0x1.4000000000000p+2f,0x1.8000000000000p+2f,0x1.c000000000000p+2f,0x1.0000000000000p+3f,0x1.2000000000000p+3f,0x1.4000000000000p+3f,0x1.4000000000000p+4f,0x1.e000000000000p+4f,0x1.4000000000000p+5f,0x1.9000000000000p+5f,0x1.e000000000000p+5f,0x1.1800000000000p+6f,0x1.4000000000000p+6f,0x1.6800000000000p+6f,0x1.9000000000000p+6f,0x1.9000000000000p+7f,0x1.2c00000000000p+8f,0x1.9000000000000p+8f,0x1.f400000000000p+8f,0x1.2c00000000000p+9f,0x1.5e00000000000p+9f,0x1.9000000000000p+9f,0x1.c200000000000p+9f,0x1.f400000000000p+9f,0x1.f400000000000p+10f,0x1.7700000000000p+11f,0x1.f400000000000p+11f,0x1.3880000000000p+12f,0x1.7700000000000p+12f,0x1.b580000000000p+12f,0x1.f400000000000p+12f,0x1.1940000000000p+13f,0x1.3880000000000p+13f,0x1.3880000000000p+14f,0x1.d4c0000000000p+14f,0x1.3880000000000p+15f,0x1.86a0000000000p+15f,0x1.d4c0000000000p+15f,0x1.1170000000000p+16f,0x1.3880000000000p+16f,0x1.5f90000000000p+16f,0x1.86a0000000000p+16f,0x1.86a0000000000p+17f,0x1.24f8000000000p+18f,0x1.86a0000000000p+18f,0x1.e848000000000p+18f,0x1.24f8000000000p+19f,0x1.55cc000000000p+19f,0x1.86a0000000000p+19f,0x1.b774000000000p+19f,0x1.e848000000000p+19f};
__device__ __constant__ float tc_nt_i[55] = {0x1.0000000000000p+0f,0x1.0000000000000p+1f,0x1.8000000000000p+1f,0x1.0000000000000p+2f,0x1.4000000000000p+2f,0x1.8000000000000p+2f,0x1.c000000000000p+2f,0x1.0000000000000p+3f,0x1.2000000000000p+3f,0x1.4000000000000p+3f,0x1.4000000000000p+4f,0x1.e000000000000p+4f,0x1.4000000000000p+5f,0x1.9000000000000p+5f,0x1.e000000000000p+5f,0x1.1800000000000p+6f,0x1.4000000000000p+6f,0x1.6800000000000p+6f,0x1.9000000000000p+6f,0x1.9000000000000p+7f,0x1.2c00000000000p+8f,0x1.9000000000000p+8f,0x1.f400000000000p+8f,0x1.2c00000000000p+9f,0x1.5e00000000000p+9f,0x1.9000000000000p+9f,0x1.c200000000000p+9f,0x1.f400000000000p+9f,0x1.f400000000000p+10f,0x1.7700000000000p+11f,0x1.f400000000000p+11f,0x1.3880000000000p+12f,0x1.7700000000000p+12f,0x1.b580000000000p+12f,0x1.f400000000000p+12f,0x1.1940000000000p+13f,0x1.3880000000000p+13f,0x1.3880000000000p+14f,0x1.d4c0000000000p+14f,0x1.3880000000000p+15f,0x1.86a0000000000p+15f,0x1.d4c0000000000p+15f,0x1.1170000000000p+16f,0x1.3880000000000p+16f,0x1.5f90000000000p+16f,0x1.86a0000000000p+16f,0x1.86a0000000000p+17f,0x1.24f8000000000p+18f,0x1.86a0000000000p+18f,0x1.e848000000000p+18f,0x1.24f8000000000p+19f,0x1.55cc000000000p+19f,0x1.86a0000000000p+19f,0x1.b774000000000p+19f,0x1.e848000000000p+19f};
__device__ __constant__ double tc_ds[100] = {0x1.ad333ee243328p-13,0x1.c16d81a78be12p-13,0x1.d69bcfaade57ep-13,0x1.ecc9a94a4d8f2p-13,0x1.02018cd3b94bep-12,0x1.0e2a5e98cc55cp-12,0x1.1ae5e43f67fbap-12,0x1.283b07bbcf881p-12,0x1.3631066c9bf1dp-12,0x1.44cf750921f39p-12,0x1.541e43bf463eep-12,0x1.6425c282fc07dp-12,0x1.74eea591c5216p-12,0x1.86820a2ca71b1p-12,0x1.98e97b8b26644p-12,0x1.ac2ef809f7722p-12,0x1.c05cf69836519p-12,0x1.d57e6c6616fefp-12,0x1.eb9ed2d8246e4p-12,0x1.016516e124b35p-11,0x1.0d8688f702a7ep-11,0x1.1a3a55f7abf18p-11,0x1.278763a60cb63p-11,0x1.3574eafcd1076p-11,0x1.440a7c1a68ab6p-11,0x1.5350025c5c3fap-11,0x1.634dc8ac2e968p-11,0x1.740c7e00201a6p-11,0x1.85953a12462afp-11,0x1.97f1825085ee5p-11,0x1.ab2b4f0820ee2p-11,0x1.bf4d10cfa2349p-11,0x1.d461b6321c7cbp-11,0x1.ea74b19ecd7efp-11,0x1.00c8ffd02fc4bp-10,0x1.0ce316af94b94p-10,0x1.198f2fb8fadacp-10,0x1.26d42c8082108p-10,0x1.34b9419f9318ep-10,0x1.4345fa9e8277ep-10,0x1.52823e0d680cap-10,0x1.627651ce64d49p-10,0x1.732adf93ab3f1p-10,0x1.84a8f993ba89bp-10,0x1.96fa1f765b0edp-10,0x1.aa28437d0841dp-10,0x1.be3dcfe995567p-10,0x1.d345aca5fb610p-10,0x1.e94b453063179p-10,0x1.002d476750b85p-9,0x1.0c40078642915p-9,0x1.18e471443dd61p-9,0x1.262162091f8a0p-9,0x1.33fe0a0fb50efp-9,0x1.4281f04cffa83p-9,0x1.51b4f6869004cp-9,0x1.619f5d9a32002p-9,0x1.7249c9f93b8cdp-9,0x1.83bd4859edcabp-9,0x1.960352a174a50p-9,0x1.a925d50930130p-9,0x1.bd2f3382124a1p-9,0x1.d22a4f58ffe0ep-9,0x1.e8228d1f42349p-9,0x1.ff23dada4156cp-9,0x1.0b9d5b3ef0bffp-8,0x1.183a1a5a84432p-8,0x1.256f03fdfd262p-8,0x1.3343440833c64p-8,0x1.41be5cdd9c775p-8,0x1.50e82b7c2884fp-8,0x1.60c8ebc05981bp-8,0x1.71693cddd870fp-8,0x1.82d2260dfe51ep-8,0x1.950d1b76d8dd2p-8,0x1.a824034d54f12p-8,0x1.bc213b3558474p-8,0x1.d10f9de2b5b08p-8,0x1.e6fa88fe0a4d5p-8,0x1.fdede350b7137p-8,0x1.0afb119da8498p-7,0x1.17902abd03ae2p-7,0x1.24bd121d5adf1p-7,0x1.3288ef4435f40p-7,0x1.40fb400840f41p-7,0x1.501bdca2b3cf4p-7,0x1.5ff2fbf1cecf9p-7,0x1.708937eebba8ep-7,0x1.81e792593f31fp-7,0x1.9417799bc50fap-7,0x1.a722cdea6d303p-7,0x1.bb13e69fe301ap-7,0x1.cff597dae7daap-7,0x1.e5d3385f9d2eep-7,0x1.fcb8a7bfbff3ep-7,0x1.0a592a66968e2p-6,0x1.16e6a22d17b5bp-6,0x1.240b8c25a08f0p-6,0x1.31cf0b7f0c0efp-6,0x1.4038998500e50p-6};
__device__ __constant__ double tc_dr[100] = {0x1.ad333ee243328p-15,0x1.c16d81a78be10p-15,0x1.d69bcfaade57dp-15,0x1.ecc9a94a4d8f0p-15,0x1.02018cd3b94bdp-14,0x1.0e2a5e98cc55bp-14,0x1.1ae5e43f67fb8p-14,0x1.283b07bbcf880p-14,0x1.3631066c9bf1cp-14,0x1.44cf750921f38p-14,0x1.541e43bf463edp-14,0x1.6425c282fc07cp-14,0x1.74eea591c5215p-14,0x1.86820a2ca71b0p-14,0x1.98e97b8b26640p-14,0x1.ac2ef809f7721p-14,0x1.c05cf6983651bp-14,0x1.d57e6c6616feep-14,0x1.eb9ed2d8246e0p-14,0x1.016516e124b34p-13,0x1.0d8688f702a7fp-13,0x1.1a3a55f7abf17p-13,0x1.278763a60cb60p-13,0x1.3574eafcd1071p-13,0x1.440a7c1a68ab2p-13,0x1.5350025c5c3f9p-13,0x1.634dc8ac2e967p-13,0x1.740c7e00201a8p-13,0x1.85953a12462b4p-13,0x1.97f1825085ee7p-13,0x1.ab2b4f0820ee0p-13,0x1.bf4d10cfa2344p-13,0x1.d461b6321c7c9p-13,0x1.ea74b19ecd7f2p-13,0x1.00c8ffd02fc4ap-12,0x1.0ce316af94b93p-12,0x1.198f2fb8fada9p-12,0x1.26d42c8082105p-12,0x1.34b9419f9318dp-12,0x1.4345fa9e8277dp-12,0x1.52823e0d680c6p-12,0x1.627651ce64d45p-12,0x1.732adf93ab3f1p-12,0x1.84a8f993ba89ap-12,0x1.96fa1f765b0ebp-12,0x1.aa28437d0841bp-12,0x1.be3dcfe995566p-12,0x1.d345aca5fb60ep-12,0x1.e94b453063177p-12,0x1.002d476750b84p-11,0x1.0c40078642914p-11,0x1.18e471443dd5fp-11,0x1.262162091f89fp-11,0x1.33fe0a0fb50eep-11,0x1.4281f04cffa82p-11,0x1.51b4f6869004bp-11,0x1.619f5d9a32001p-11,0x1.7249c9f93b8cbp-11,0x1.83bd4859edcaap-11,0x1.960352a174a4ep-11,0x1.a925d5093012fp-11,0x1.bd2f3382124a0p-11,0x1.d22a4f58ffe0dp-11,0x1.e8228d1f42348p-11,0x1.ff23dada4156bp-11,0x1.0b9d5b3ef0bffp-10,0x1.183a1a5a84431p-10,0x1.256f03fdfd261p-10,0x1.3343440833c64p-10,0x1.41be5cdd9c774p-10,0x1.50e82b7c2884ep-10,0x1.60c8ebc05981ap-10,0x1.71693cddd870ep-10,0x1.82d2260dfe51cp-10,0x1.950d1b76d8dd0p-10,0x1.a824034d54f0fp-10,0x1.bc213b3558472p-10,0x1.d10f9de2b5b06p-10,0x1.e6fa88fe0a4d3p-10,0x1.fdede350b7135p-10,0x1.0afb119da8497p-9,0x1.17902abd03ae1p-9,0x1.24bd121d5adefp-9,0x1.3288ef4435f3ep-9,0x1.40fb400840f40p-9,0x1.501bdca2b3cf3p-9,0x1.5ff2fbf1cecf8p-9,0x1.708937eebba8cp-9,0x1.81e792593f31dp-9,0x1.9417799bc50f9p-9,0x1.a722cdea6d302p-9,0x1.bb13e69fe3019p-9,0x1.cff597dae7da9p-9,0x1.e5d3385f9d2edp-9,0x1.fcb8a7bfbff3dp-9,0x1.0a592a66968e1p-8,0x1.16e6a22d17b5ap-8,0x1.240b8c25a08efp-8,0x1.31cf0b7f0c0eep-8,0x1.4038998500e4fp-8};
__device__ __constant__ float tc_csg[18] = {0x1.0000000000000p+1f,0x1.8000000000000p+2f,0x1.8000000000000p+2f,0x1.c1b23e0000000p+1f,0x1.c5fb860000000p+5f,0x1.8000000000000p+4f,0x1.ef90cc0000000p+1f,0x1.c26ea40000000p+3f,0x1.0502320000000p+6f,0x1.e7381a0000000p+2f,0x1.4a7bda0000000p+7f,0x1.0502320000000p+6f,0x1.60b3f40000000p+0f,0x1.60b3f40000000p+0f,0x1.cbcaca0000000p-1f,0x1.d997ba0000000p-1f,0x1.81fe7e0000000p+1f,0x1.e7381a0000000p+2f};
__device__ __constant__ float tc_cse[18] = {0x1.8000000000000p+1f,0x1.0000000000000p+2f,0x1.0000000000000p+2f,0x1.c666660000000p+1f,0x1.6333340000000p+2f,0x1.4000000000000p+2f,0x1.d15e9e0000000p+1f,0x1.28af500000000p+2f,0x1.68af500000000p+2f,0x1.0be2820000000p+2f,0x1.8be2840000000p+2f,0x1.68af500000000p+2f,0x1.4666660000000p+1f,0x1.4666660000000p+1f,0x1.a2bd3c0000000p+0f,0x1.c666660000000p+0f,0x1.b491d00000000p+1f,0x1.0be2820000000p+2f};
__device__ __constant__ float tc_crg[13] = {0x1.8000000000000p+2f,0x1.0000000000000p+0f,0x1.8000000000000p+2f,0x1.6800020000000p+9f,0x1.0000000000000p+0f,0x1.8000000000000p+4f,0x1.a963920000000p+1f,0x1.6800020000000p+9f,0x1.8000000000000p+2f,0x1.0000000000000p+0f,0x1.0000000000000p+1f,0x1.544fa60000000p+0f,0x1.3b00080000000p+12f};
__device__ __constant__ float tc_cre[13] = {0x1.0000000000000p+2f,0x1.0000000000000p+0f,0x1.0000000000000p+2f,0x1.c000000000000p+2f,0x1.0000000000000p+1f,0x1.4000000000000p+2f,0x1.c000000000000p+1f,0x1.c000000000000p+2f,0x1.0000000000000p+2f,0x1.0000000000000p+1f,0x1.8000000000000p+1f,0x1.4000000000000p+1f,0x1.0000000000000p+3f};
__device__ __constant__ float tc_cig[7] = {0x1.0000000000000p+0f,0x1.8000000000000p+2f,0x1.8000000000000p+4f,0x1.0000000000000p+0f,0x1.0000000000000p+0f,0x1.a963920000000p+1f,0x1.544fa60000000p+0f};
__device__ __constant__ float tc_cie[7] = {0x1.0000000000000p+0f,0x1.0000000000000p+2f,0x1.4000000000000p+2f,0x1.0000000000000p+1f,0x1.0000000000000p+1f,0x1.c000000000000p+1f,0x1.4000000000000p+1f};
__device__ __constant__ float tc_cgg[12] = {0x1.8000000000000p+2f,0x1.0000000000000p+0f,0x1.8000000000000p+2f,0x1.6800020000000p+9f,0x1.f9204c0000000p+11f,0x1.45cfc80000000p+4f,0x1.8e4e040000000p+6f,0x1.2540780000000p+9f,0x1.4f065a0000000p+2f,0x1.0000000000000p+0f,0x1.e6f4a80000000p+0f,0x1.e6f4a80000000p+0f};
__device__ __constant__ float tc_cge[12] = {0x1.0000000000000p+2f,0x1.0000000000000p+0f,0x1.0000000000000p+2f,0x1.c000000000000p+2f,0x1.f8f5c20000000p+2f,0x1.38f5c20000000p+2f,0x1.78f5c20000000p+2f,0x1.b8f5c20000000p+2f,0x1.f1eb840000000p+1f,0x1.0000000000000p+1f,0x1.78f5c20000000p+1f,0x1.78f5c20000000p+1f};
__device__ __constant__ float tc_ccg[75] = {0x1.0000000000000p+0f,0x1.8000000000000p+4f,0x1.3b00080000000p+12f,0x1.8000000000000p+2f,0x1.6800020000000p+9f,0x1.0000000000000p+1f,0x1.e000020000000p+6f,0x1.3afffe0000000p+15f,0x1.8000000000000p+4f,0x1.3b00080000000p+12f,0x1.8000000000000p+2f,0x1.6800020000000p+9f,0x1.625ffe0000000p+18f,0x1.e000020000000p+6f,0x1.3afffe0000000p+15f,0x1.8000000000000p+4f,0x1.3b00080000000p+12f,0x1.baf80e0000000p+21f,0x1.6800020000000p+9f,0x1.625ffe0000000p+18f,0x1.e000020000000p+6f,0x1.3afffe0000000p+15f,0x1.308a800000000p+25f,0x1.3b00080000000p+12f,0x1.baf80e0000000p+21f,0x1.6800020000000p+9f,0x1.625ffe0000000p+18f,0x1.c8cfd00000000p+28f,0x1.3afffe0000000p+15f,0x1.308a800000000p+25f,0x1.3b00080000000p+12f,0x1.baf80e0000000p+21f,0x1.7328d20000000p+32f,0x1.625ffe0000000p+18f,0x1.c8cfd00000000p+28f,0x1.3afffe0000000p+15f,0x1.308a800000000p+25f,0x1.44c3b40000000p+36f,0x1.baf80e0000000p+21f,0x1.7328d20000000p+32f,0x1.625ffe0000000p+18f,0x1.c8cfd00000000p+28f,0x1.3077700000000p+40f,0x1.308a800000000p+25f,0x1.44c3b40000000p+36f,0x1.baf80e0000000p+21f,0x1.7328d20000000p+32f,0x1.3077700000000p+44f,0x1.c8cfd00000000p+28f,0x1.3077700000000p+40f,0x1.308a800000000p+25f,0x1.44c3b40000000p+36f,0x1.437ef00000000p+48f,0x1.7328d20000000p+32f,0x1.3077700000000p+44f,0x1.c8cfd00000000p+28f,0x1.3077700000000p+40f,0x1.6beef20000000p+52f,0x1.44c3b40000000p+36f,0x1.437ef00000000p+48f,0x1.7328d20000000p+32f,0x1.3077700000000p+44f,0x1.b02bbe0000000p+56f,0x1.3077700000000p+40f,0x1.6beef20000000p+52f,0x1.44c3b40000000p+36f,0x1.437ef00000000p+48f,0x1.0e1b460000000p+61f,0x1.3077700000000p+44f,0x1.b02bbe0000000p+56f,0x1.3077700000000p+40f,0x1.6beef20000000p+52f,0x1.6283b20000000p+65f,0x1.437ef00000000p+48f,0x1.0e1b460000000p+61f};
__device__ __constant__ float tc_cce[75] = {0x1.0000000000000p+1f,0x1.4000000000000p+2f,0x1.0000000000000p+3f,0x1.0000000000000p+2f,0x1.c000000000000p+2f,0x1.8000000000000p+1f,0x1.8000000000000p+2f,0x1.2000000000000p+3f,0x1.4000000000000p+2f,0x1.0000000000000p+3f,0x1.0000000000000p+2f,0x1.c000000000000p+2f,0x1.4000000000000p+3f,0x1.8000000000000p+2f,0x1.2000000000000p+3f,0x1.4000000000000p+2f,0x1.0000000000000p+3f,0x1.6000000000000p+3f,0x1.c000000000000p+2f,0x1.4000000000000p+3f,0x1.8000000000000p+2f,0x1.2000000000000p+3f,0x1.8000000000000p+3f,0x1.0000000000000p+3f,0x1.6000000000000p+3f,0x1.c000000000000p+2f,0x1.4000000000000p+3f,0x1.a000000000000p+3f,0x1.2000000000000p+3f,0x1.8000000000000p+3f,0x1.0000000000000p+3f,0x1.6000000000000p+3f,0x1.c000000000000p+3f,0x1.4000000000000p+3f,0x1.a000000000000p+3f,0x1.2000000000000p+3f,0x1.8000000000000p+3f,0x1.e000000000000p+3f,0x1.6000000000000p+3f,0x1.c000000000000p+3f,0x1.4000000000000p+3f,0x1.a000000000000p+3f,0x1.0000000000000p+4f,0x1.8000000000000p+3f,0x1.e000000000000p+3f,0x1.6000000000000p+3f,0x1.c000000000000p+3f,0x1.1000000000000p+4f,0x1.a000000000000p+3f,0x1.0000000000000p+4f,0x1.8000000000000p+3f,0x1.e000000000000p+3f,0x1.2000000000000p+4f,0x1.c000000000000p+3f,0x1.1000000000000p+4f,0x1.a000000000000p+3f,0x1.0000000000000p+4f,0x1.3000000000000p+4f,0x1.e000000000000p+3f,0x1.2000000000000p+4f,0x1.c000000000000p+3f,0x1.1000000000000p+4f,0x1.4000000000000p+4f,0x1.0000000000000p+4f,0x1.3000000000000p+4f,0x1.e000000000000p+3f,0x1.2000000000000p+4f,0x1.5000000000000p+4f,0x1.1000000000000p+4f,0x1.4000000000000p+4f,0x1.0000000000000p+4f,0x1.3000000000000p+4f,0x1.6000000000000p+4f,0x1.2000000000000p+4f,0x1.5000000000000p+4f};
struct ThompsonColumn {
int nz;
float mu_c;
static constexpr float am_g = 0x1.05cca60000000p+8f;
static constexpr float am_i = 0x1.d200c00000000p+8f;
static constexpr float am_r = 0x1.05cca60000000p+9f;
static constexpr float am_s = 0x1.1a9fbe0000000p-4f;
static constexpr float ar_volume = 0x1.2dd5880000000p-54f;
static constexpr float ato = 0x1.374bc60000000p-2f;
static constexpr float av_c = 0x1.e39f080000000p+24f;
static constexpr float av_g = 0x1.ba00000000000p+8f;
static constexpr float av_i = 0x1.cde0000000000p+10f;
static constexpr float av_r = 0x1.2f60000000000p+12f;
static constexpr float av_s = 0x1.4000000000000p+5f;
static constexpr float bm_g = 0x1.8000000000000p+1f;
static constexpr float bm_i = 0x1.8000000000000p+1f;
static constexpr float bm_r = 0x1.8000000000000p+1f;
static constexpr float bm_s = 0x1.0000000000000p+1f;
static constexpr float bv_c = 0x1.0000000000000p+1f;
static constexpr float bv_g = 0x1.c7ae140000000p-1f;
static constexpr float bv_i = 0x1.0000000000000p+0f;
static constexpr float bv_s = 0x1.19999a0000000p-1f;
static constexpr float c_cube = 0x1.0000000000000p-1f;
static constexpr float c_sqrd = 0x1.3333340000000p-3f;
const float* cce = tc_cce;
const float* ccg = tc_ccg;
const float* cge = tc_cge;
const float* cgg = tc_cgg;
const float* cie = tc_cie;
const float* cig = tc_cig;
static constexpr float cp = 0x1.f600000000000p+9f;
const float* cre = tc_cre;
const float* crg = tc_crg;
const float* cse = tc_cse;
const float* csg = tc_csg;
static constexpr float d0c = 0x1.0c6f7a0000000p-20f;
static constexpr float d0g = 0x1.0624de0000000p-12f;
static constexpr float d0i = 0x1.b0ccb40000000p-17f;
static constexpr float d0r = 0x1.a36e2e0000000p-15f;
static constexpr float d0s = 0x1.a36e2e0000000p-13f;
const double* dr = tc_dr;
const double* ds = tc_ds;
static constexpr bool dustyice = true;
static constexpr float ef_ri = 0x1.e666660000000p-1f;
static constexpr float ef_si = 0x1.99999a0000000p-5f;
static constexpr float eps = 0x1.203afa0000000p-50f;
static constexpr float fv_r = 0x1.8600000000000p+7f;
static constexpr float fv_s = 0x1.9000000000000p+6f;
static constexpr float gonv_max = 0x1.6e36000000000p+21f;
static constexpr float gonv_min = 0x1.3880000000000p+13f;
static constexpr float hgfr = 0x1.d651ec0000000p+7f;
static constexpr bool homogice = true;
static constexpr int hostinclusionstring_g = 0;
static constexpr int hostinclusionstring_s = 0;
static constexpr int hostmatrixstring_g = 0;
static constexpr int hostmatrixstring_s = 0;
static constexpr int hoststring_g = 0;
static constexpr int hoststring_s = 0;
static constexpr int ifdry = 0;
static constexpr bool iiwarm = false;
static constexpr int inclusionstring_g = 0;
static constexpr int inclusionstring_s = 0;
static constexpr bool is_aerosol_aware = false;
static constexpr double k_w = 0x1.de4b5a79d9de5p-1;
static constexpr float kap0 = 0x1.ea999a0000000p+8f;
static constexpr float kap1 = 0x1.175c280000000p+4f;
static constexpr float lam0 = 0x1.4c7ae20000000p+4f;
static constexpr float lam1 = 0x1.a51eb80000000p+1f;
static constexpr double lamda4 = 0x1.a36e305532621p-14;
static constexpr double lamda_radar = 0x1.99999a0000000p-4;
static constexpr float lfus = 0x1.462c000000000p+18f;
static constexpr float lsub = 0x1.59f2800000000p+21f;
static constexpr float lvap0 = 0x1.312d000000000p+21f;
cd m_i_0 = {0x1.c91dae74d3b24p+0,-0x1.1acb622ae704dp-13};
cd m_w_0 = {0x1.213d1a0000000p+3,-0x1.634d6c0000000p+0};
static constexpr int matrixstring_g = 0;
static constexpr int matrixstring_s = 0;
static constexpr double melt_outside_g = 0x1.ccccccccccccdp-1;
static constexpr double melt_outside_s = 0x1.ccccccccccccdp-1;
static constexpr int mixingrulestring_g = 0;
static constexpr int mixingrulestring_s = 0;
static constexpr float mu_g = 0x0.0p+0f;
static constexpr float mu_i = 0x0.0p+0f;
static constexpr float mu_r = 0x0.0p+0f;
static constexpr float mu_s = 0x1.457a780000000p-1f;
static constexpr float nain1 = 0x1.e848000000000p+18f;
static constexpr int nbc = 100;
static constexpr int nbins = 100;
static constexpr int nbr = 100;
static constexpr int nbs = 100;
static constexpr int nic1 = 7;
static constexpr int nic2 = -6;
static constexpr int nig2 = -5;
static constexpr int nig3 = 4;
static constexpr int nii2 = -10;
static constexpr int nii3 = 0;
static constexpr int niin2 = 0;
static constexpr int nir2 = -6;
static constexpr int nir3 = 6;
static constexpr int nis2 = -5;
static constexpr int nrbins = 50;
float nt_c;
static constexpr float nt_c_max = 0x1.dc99480000000p+30f;
const float* nt_i = tc_nt_i;
const float* nt_in = tc_nt_in;
static constexpr int ntb_arc = 7;
static constexpr int ntb_art = 7;
static constexpr int ntb_arw = 9;
static constexpr int ntb_c = 37;
static constexpr int ntb_g = 28;
static constexpr int ntb_g1 = 28;
static constexpr int ntb_i = 64;
static constexpr int ntb_i1 = 55;
static constexpr int ntb_in = 55;
static constexpr int ntb_r = 37;
static constexpr int ntb_r1 = 37;
static constexpr int ntb_s = 28;
static constexpr int ntb_t = 9;
static constexpr float oams = 0x1.cfc4a40000000p+3f;
static constexpr float obmg = 0x1.5555560000000p-2f;
static constexpr float obmi = 0x1.5555560000000p-2f;
static constexpr float obmr = 0x1.5555560000000p-2f;
static constexpr float obms = 0x1.0000000000000p-1f;
const float* ocg1 = tc_ocg1;
const float* ocg2 = tc_ocg2;
static constexpr float ocmg = 0x1.4023ec0000000p-3f;
static constexpr float ocms = 0x1.e749a60000000p+1f;
static constexpr float oge1 = 0x1.0000000000000p-2f;
static constexpr float ogg1 = 0x1.5555560000000p-3f;
static constexpr float ogg2 = 0x1.0000000000000p+0f;
static constexpr float ogg3 = 0x1.5555560000000p-3f;
static constexpr float oig1 = 0x1.0000000000000p+0f;
static constexpr float oig2 = 0x1.5555560000000p-3f;
static constexpr float olfus = 0x1.91d9780000000p-19f;
static constexpr float org1 = 0x1.5555560000000p-3f;
static constexpr float org2 = 0x1.0000000000000p+0f;
static constexpr float org3 = 0x1.5555560000000p-3f;
static constexpr float orv = 0x1.1c03540000000p-9f;
static constexpr float pi = 0x1.921fb60000000p+1f;
static constexpr double pi5 = 0x1.3204ba0000000p+8;
static constexpr float r = 0x1.1f0a3e0000000p+8f;
static constexpr float r1 = 0x1.1979980000000p-40f;
static constexpr float r2 = 0x1.0c6f7a0000000p-20f;
const float* r_c = tc_r_c;
const float* r_g = tc_r_g;
const float* r_i = tc_r_i;
const float* r_r = tc_r_r;
const float* r_s = tc_r_s;
static constexpr float r_uni = 0x1.0a0c4a0000000p+3f;
static constexpr float rho_g = 0x1.f400000000000p+8f;
static constexpr float rho_i = 0x1.bd00000000000p+9f;
static constexpr float rho_not = 0x1.2f3cc60000000p+0f;
static constexpr float rho_w = 0x1.f400000000000p+9f;
static constexpr float rv = 0x1.cd80000000000p+8f;
const float* sa = tc_sa;
const float* sb = tc_sb;
static constexpr float sc = 0x1.4395820000000p-1f;
const double* simpson = tc_simpson;
static constexpr float t1_qg_me = 0x1.0f6d1a0000000p-16f;
static constexpr float t1_qg_qc = 0x1.c64e860000000p+10f;
static constexpr float t1_qg_sd = 0x1.b851ec0000000p-1f;
static constexpr float t1_qr_ev = 0x1.8f5c280000000p-1f;
static constexpr float t1_qr_qc = 0x1.6567c00000000p+14f;
static constexpr float t1_qr_qi = 0x1.6567c00000000p+14f;
static constexpr float t1_qs_me = 0x1.45b6220000000p-18f;
static constexpr float t1_qs_qc = 0x1.f6a7a40000000p+4f;
static constexpr float t1_qs_qi = 0x1.f6a7a40000000p+4f;
static constexpr float t1_qs_sd = 0x1.b851ec0000000p-1f;
static constexpr float t2_qg_me = 0x1.7b19b60000000p-13f;
static constexpr float t2_qg_sd = 0x1.337f400000000p+3f;
static constexpr float t2_qr_ev = 0x1.26a40e0000000p+5f;
static constexpr float t2_qr_qi = 0x1.56a8760000000p+30f;
static constexpr float t2_qs_me = 0x1.1fc86c0000000p-17f;
static constexpr float t2_qs_sd = 0x1.850ba40000000p+0f;
static constexpr float t_0 = 0x1.1126660000000p+8f;
const double* t_efrw;
const double* t_efsw;
const double* t_nc = tc_t_nc;
const float* ta_na = tc_ta_na;
const float* ta_tk = tc_ta_tk;
const float* ta_ww = tc_ta_ww;
const double* tcg_racg;
const double* tcr_gacr;
const double* tcr_sacr1;
const double* tcr_sacr2;
const double* tcs_racs1;
const double* tcs_racs2;
const double* tmg_gacr;
const double* tmr_racg;
const double* tmr_racs1;
const double* tmr_racs2;
const double* tms_sacr1;
const double* tms_sacr2;
const double* tnc_wev;
const float* tnccn_act;
const double* tni_iaus;
const double* tni_qcfz;
const double* tni_qrfz;
static constexpr float tno = 0x1.4000000000000p+2f;
const double* tnr_gacr;
const double* tnr_qrfz;
const double* tnr_racg;
const double* tnr_racs1;
const double* tnr_racs2;
const double* tnr_rev;
const double* tnr_sacr1;
const double* tnr_sacr2;
const double* tpc_wev;
const double* tpg_qrfz;
const double* tpi_ide;
const double* tpi_qcfz;
const double* tpi_qrfz;
const double* tps_iaus;
const double* xdtg = tc_xdtg;
const double* xdts = tc_xdts;
static constexpr float xm0i = 0x1.1979980000000p-40f;
const double* xxdg = tc_xxdg;
const double* xxds = tc_xxds;
// MPAS physics_mmm/mp_radar.F90:192-292; imported module interface wrapper.
__device__ void rayleigh_soak_wetgraupel(double x,double a,double b,double fm,double mo,cd mw,cd mi,double lambda,double& cback,int,int,int,int,int,int) { cback = refl_rayleigh_soak_wetgraupel(x,a,b,fm,mo,mw,mi,pi5,lamda4); }
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:156 rslf
__device__ __forceinline__ float rslf(float p,float t) {
float result;
float esl;
float x;
float c0 = .611583699e03f;
float c1 = .444606896e02f;
float c2 = .143177157e01f;
float c3 = .264224321e-1f;
float c4 = .299291081e-3f;
float c5 = .203154182e-5f;
float c6 = .702620698e-8f;
float c7 = .379534310e-11f;
float c8 = (-.321582393e-13f);
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:171
x = tc_max((-80.f),tc_sub(t,273.16f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:174
esl = tc_add(c0,tc_mul(x,(tc_add(c1,tc_mul(x,(tc_add(c2,tc_mul(x,(tc_add(c3,tc_mul(x,(tc_add(c4,tc_mul(x,(tc_add(c5,tc_mul(x,(tc_add(c6,tc_mul(x,(tc_add(c7,tc_mul(x,c8)))))))))))))))))))))));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:175
esl = tc_min(esl,tc_mul(p,0.15f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:176
result = tc_div(tc_mul(.622f,esl),(tc_sub(p,esl)));
return result;
}
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:191 rsif
__device__ __forceinline__ float rsif(float p,float t) {
float result;
float esi;
float x;
float c0 = .609868993e03f;
float c1 = .499320233e02f;
float c2 = .184672631e01f;
float c3 = .402737184e-1f;
float c4 = .565392987e-3f;
float c5 = .521693933e-5f;
float c6 = .307839583e-7f;
float c7 = .105785160e-9f;
float c8 = .161444444e-12f;
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:206
x = tc_max((-80.f),tc_sub(t,273.16f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:207
esi = tc_add(c0,tc_mul(x,(tc_add(c1,tc_mul(x,(tc_add(c2,tc_mul(x,(tc_add(c3,tc_mul(x,(tc_add(c4,tc_mul(x,(tc_add(c5,tc_mul(x,(tc_add(c6,tc_mul(x,(tc_add(c7,tc_mul(x,c8)))))))))))))))))))))));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:208
esi = tc_min(esi,tc_mul(p,0.15f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:209
result = tc_div(tc_mul(.622f,esi),(tc_sub(p,esi)));
return result;
}
// MPAS module_mp_thompson.F:4662 icedemott
__device__ __forceinline__ float icedemott(float tempc,float qv,float qvs,float qvsi,float rho,float nifa) {
float result;
float satw;
float sati;
float siw;
float p_x;
float si0x;
float dtt;
float dsi;
float dsw;
float dab;
float fc;
float hx;
float ntilde;
float n_in;
float nmax;
float nhat;
float mux;
float xni;
float nifa_cc;
float p_c1 = 1000.f;
float p_rho_c = 0.76f;
float p_alpha = 1.0f;
float p_gam = 2.f;
float delt = 5.f;
float t0x = (-40.f);
float sw0x = 0.97f;
float delsi = 0.1f;
float hdm = 0.15f;
float p_psi = tc_div(tc_mul(0.058707f,p_gam),p_rho_c);
float aap = 1.f;
float bbp = 0.f;
float y1p = (-35.f);
float y2p = (-25.f);
float rho_not0 = tc_div(101325.f,(tc_mul(287.05f,273.15f)));
// MPAS module_mp_thompson.F:4688
xni = 0.0f;
// MPAS module_mp_thompson.F:4720
nifa_cc = tc_max(0.5f,tc_div(tc_mul(tc_mul(nifa,rho_not0),1.e-6f),rho));
// MPAS module_mp_thompson.F:4722
xni = tc_mul((tc_mul(5.94e-5f,tc_pow(((-tempc)),3.33f))),(tc_pow(nifa_cc,(tc_add((tc_mul((-0.0264f),(tempc))),0.0033f)))));
// MPAS module_mp_thompson.F:4724
xni = tc_mul(tc_div(tc_mul(xni,rho),rho_not0),1000.f);
// MPAS module_mp_thompson.F:4727
result = tc_max(0.f,xni);
return result;
}
// MPAS module_mp_thompson.F:4736 icekoop
__device__ __forceinline__ float icekoop(float temp,float qv,float qvs,float naero,float dt) {
float result;
float mu_diff;
float a_w_i;
float delta_aw;
float log_j_rate;
float j_rate;
float prob_h;
float satw;
float xni;
// MPAS module_mp_thompson.F:4743
xni = 0.0f;
// MPAS module_mp_thompson.F:4744
satw = tc_div(qv,qvs);
// MPAS module_mp_thompson.F:4745
mu_diff = tc_sub(tc_sub(tc_add(210368.0f,(tc_mul(131.438f,temp))),(tc_div(3.32373e6f,temp))),(tc_mul(41729.1f,tc_log(temp))));
// MPAS module_mp_thompson.F:4747
a_w_i = tc_exp(tc_div(mu_diff,(tc_mul(r_uni,temp))));
// MPAS module_mp_thompson.F:4748
delta_aw = tc_sub(satw,a_w_i);
// MPAS module_mp_thompson.F:4749
log_j_rate = tc_add(tc_sub(tc_add((-906.7f),(tc_mul(8502.0f,delta_aw))),(tc_mul(tc_mul(26924.0f,delta_aw),delta_aw))),(tc_mul(tc_mul(tc_mul(29180.0f,delta_aw),delta_aw),delta_aw)));
// MPAS module_mp_thompson.F:4752
log_j_rate = tc_min(20.0f,log_j_rate);
// MPAS module_mp_thompson.F:4753
j_rate = tc_pow(10.f,log_j_rate);
// MPAS module_mp_thompson.F:4754
prob_h = tc_min(tc_sub(1.f,tc_exp(tc_mul(tc_mul((-j_rate),ar_volume),dt))),1.f);
// MPAS module_mp_thompson.F:4755
if((prob_h>0.f)) {
// MPAS module_mp_thompson.F:4756
xni = tc_min(tc_mul(prob_h,naero),1000.e3f);
// MPAS module_mp_thompson.F:4757
}
// MPAS module_mp_thompson.F:4759
result = tc_max(0.0f,xni);
return result;
}
// MPAS module_mp_thompson.F:4766 delta_p
__device__ __forceinline__ float delta_p(float yy,float y1,float y2,float aa,float bb) {
float result;
float dab;
float a;
float b;
float a0;
float a1;
float a2;
float a3;
// MPAS module_mp_thompson.F:4772
a = tc_div(tc_mul(6.f,(tc_sub(aa,bb))),(tc_mul(tc_mul((tc_sub(y2,y1)),(tc_sub(y2,y1))),(tc_sub(y2,y1)))));
// MPAS module_mp_thompson.F:4773
b = tc_sub(tc_add(aa,(tc_div(tc_mul(tc_mul(tc_mul(a,y1),y1),y1),6.f))),(tc_mul(tc_mul(tc_mul(tc_mul(a,y1),y1),y2),0.5f)));
// MPAS module_mp_thompson.F:4774
a0 = b;
// MPAS module_mp_thompson.F:4775
a1 = tc_mul(tc_mul(a,y1),y2);
// MPAS module_mp_thompson.F:4776
a2 = tc_mul(tc_mul((-a),(tc_add(y1,y2))),0.5f);
// MPAS module_mp_thompson.F:4777
a3 = tc_div(a,3.f);
// MPAS module_mp_thompson.F:4779
if((yy<=y1)) {
// MPAS module_mp_thompson.F:4780
dab = aa;
// MPAS module_mp_thompson.F:4781
} else if((yy>=y2)) {
// MPAS module_mp_thompson.F:4782
dab = bb;
// MPAS module_mp_thompson.F:4783
} else {
// MPAS module_mp_thompson.F:4784
dab = tc_add(tc_add(tc_add(a0,(tc_mul(a1,yy))),(tc_mul(tc_mul(a2,yy),yy))),(tc_mul(tc_mul(tc_mul(a3,yy),yy),yy)));
// MPAS module_mp_thompson.F:4785
}
// MPAS module_mp_thompson.F:4787
if((dab<aa)) {
// MPAS module_mp_thompson.F:4788
dab = aa;
// MPAS module_mp_thompson.F:4789
}
// MPAS module_mp_thompson.F:4790
if((dab>bb)) {
// MPAS module_mp_thompson.F:4791
dab = bb;
// MPAS module_mp_thompson.F:4792
}
// MPAS module_mp_thompson.F:4793
result = dab;
return result;
}
// MPAS module_mp_thompson.F:4176 eff_aero
__device__ __forceinline__ float eff_aero(float d,float da,float visc,float rhoa,float temp,int species) {
float result;
float aval;
float cc;
float diff;
float re;
float sc;
float st;
float st2;
float vt;
float eff;
float boltzman = 1.3806503e-23f;
float meanpath = 0.0256e-6f;
// MPAS module_mp_thompson.F:4185
vt = 1.f;
// MPAS module_mp_thompson.F:4186
if((species==114)) {
// MPAS module_mp_thompson.F:4187
vt = tc_sub(tc_add(tc_sub(tc_add((-0.1021f),tc_mul(4.932e3f,d)),tc_mul(tc_mul(0.9551e6f,d),d)),tc_mul(tc_mul(tc_mul(0.07934e9f,d),d),d)),tc_mul(tc_mul(tc_mul(tc_mul(0.002362e12f,d),d),d),d));
// MPAS module_mp_thompson.F:4189
} else if((species==115)) {
// MPAS module_mp_thompson.F:4190
vt = tc_mul(av_s,tc_pow(d,bv_s));
// MPAS module_mp_thompson.F:4191
} else if((species==103)) {
// MPAS module_mp_thompson.F:4192
vt = tc_mul(av_g,tc_pow(d,bv_g));
// MPAS module_mp_thompson.F:4193
}
// MPAS module_mp_thompson.F:4195
cc = tc_add(1.f,tc_mul(tc_div(tc_mul(2.f,meanpath),da),(tc_add(1.257f,tc_mul(0.4f,tc_exp(tc_div(tc_mul((-0.55f),da),meanpath)))))));
// MPAS module_mp_thompson.F:4196
diff = tc_div(tc_mul(tc_mul(boltzman,temp),cc),(tc_mul(tc_mul(tc_mul(3.f,pi),visc),da)));
// MPAS module_mp_thompson.F:4198
re = tc_div(tc_mul(tc_mul(tc_mul(0.5f,rhoa),d),vt),visc);
// MPAS module_mp_thompson.F:4199
sc = tc_div(visc,(tc_mul(rhoa,diff)));
// MPAS module_mp_thompson.F:4201
st = tc_div(tc_mul(tc_mul(tc_mul(da,da),vt),1000.f),(tc_mul(tc_mul(9.f,visc),d)));
// MPAS module_mp_thompson.F:4202
aval = tc_add(1.f,tc_log(tc_add(1.f,re)));
// MPAS module_mp_thompson.F:4203
st2 = tc_div((tc_add(1.2f,tc_mul(tc_div(1.f,12.f),aval))),(tc_add(1.f,aval)));
// MPAS module_mp_thompson.F:4205
eff = tc_add(tc_mul(tc_div(4.f,(tc_mul(re,sc))),(tc_add(tc_add(1.f,tc_mul(tc_mul(0.4f,tc_sqrt(re)),tc_pow(sc,0.3333f))),tc_mul(tc_mul(0.16f,tc_sqrt(re)),tc_sqrt(sc))))),tc_mul(tc_div(tc_mul(4.f,da),d),(tc_add(0.02f,tc_mul(tc_div(da,d),(tc_add(1.f,tc_mul(2.f,tc_sqrt(re)))))))));
// MPAS module_mp_thompson.F:4209
if((st>st2)) { eff = tc_add(eff,tc_pow((tc_div((tc_sub(st,st2)),(tc_add(tc_sub(st,st2),0.666667f)))),1.5f)); }
// MPAS module_mp_thompson.F:4210
result = tc_max(1.e-5f,tc_min(eff,1.0f));
return result;
}
// MPAS module_mp_thompson.F:4391 activ_ncloud
__device__ __forceinline__ float activ_ncloud(float tt,float ww,float nccn) {
float result;
float n_local;
float w_local;
int i;
int j;
int k;
int l;
int m;
int n;
float a;
float b;
float c;
float d;
float t;
float u;
float x1;
float x2;
float y1;
float y2;
float nx;
float wy;
float fraction;
// MPAS module_mp_thompson.F:4406
n_local = tc_mul(nccn,1.e-6f);
// MPAS module_mp_thompson.F:4407
w_local = ww;
// MPAS module_mp_thompson.F:4409
if((n_local>=ta_na[0+tc_sub(ntb_arc,(1))*(1)])) {
// MPAS module_mp_thompson.F:4410
n_local = tc_sub(ta_na[0+tc_sub(ntb_arc,(1))*(1)],1.0f);
// MPAS module_mp_thompson.F:4411
} else if((n_local<=ta_na[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:4412
n_local = tc_add(ta_na[0+tc_sub(1,(1))*(1)],1.0f);
// MPAS module_mp_thompson.F:4413
}
// MPAS module_mp_thompson.F:4414
for(n=2; n<=ntb_arc; n+=1) {
// MPAS module_mp_thompson.F:4415
if(((n_local>=ta_na[0+tc_sub(tc_sub(n,1),(1))*(1)])&&(n_local<ta_na[0+tc_sub(n,(1))*(1)]))) { goto label_8003; }
// MPAS module_mp_thompson.F:4416
}
// MPAS module_mp_thompson.F:4417
label_8003: ;

// MPAS module_mp_thompson.F:4418
i = n;
// MPAS module_mp_thompson.F:4419
x1 = tc_log(ta_na[0+tc_sub(tc_sub(i,1),(1))*(1)]);
// MPAS module_mp_thompson.F:4420
x2 = tc_log(ta_na[0+tc_sub(i,(1))*(1)]);
// MPAS module_mp_thompson.F:4422
if((w_local>=ta_ww[0+tc_sub(ntb_arw,(1))*(1)])) {
// MPAS module_mp_thompson.F:4423
w_local = tc_sub(ta_ww[0+tc_sub(ntb_arw,(1))*(1)],1.0f);
// MPAS module_mp_thompson.F:4424
} else if((w_local<=ta_ww[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:4425
w_local = tc_add(ta_ww[0+tc_sub(1,(1))*(1)],0.001f);
// MPAS module_mp_thompson.F:4426
}
// MPAS module_mp_thompson.F:4427
for(n=2; n<=ntb_arw; n+=1) {
// MPAS module_mp_thompson.F:4428
if(((w_local>=ta_ww[0+tc_sub(tc_sub(n,1),(1))*(1)])&&(w_local<ta_ww[0+tc_sub(n,(1))*(1)]))) { goto label_8005; }
// MPAS module_mp_thompson.F:4429
}
// MPAS module_mp_thompson.F:4430
label_8005: ;

// MPAS module_mp_thompson.F:4431
j = n;
// MPAS module_mp_thompson.F:4432
y1 = tc_log(ta_ww[0+tc_sub(tc_sub(j,1),(1))*(1)]);
// MPAS module_mp_thompson.F:4433
y2 = tc_log(ta_ww[0+tc_sub(j,(1))*(1)]);
// MPAS module_mp_thompson.F:4435
k = tc_max(1,tc_min(tc_add(((int)round(tc_mul((tc_sub(tt,ta_tk[0+tc_sub(1,(1))*(1)])),0.1f))),1),ntb_art));
// MPAS module_mp_thompson.F:4442
l = 3;
// MPAS module_mp_thompson.F:4443
m = 2;
// MPAS module_mp_thompson.F:4445
a = tnccn_act[0+tc_sub(tc_sub(i,1),(1))*(1)+tc_sub(tc_sub(j,1),(1))*tc_mul(1,((7)-(1)+1))+tc_sub(k,(1))*(1*((7)-(1)+1)*((9)-(1)+1))+tc_sub(l,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1))+tc_sub(m,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1)*((5)-(1)+1))];
// MPAS module_mp_thompson.F:4446
b = tnccn_act[0+tc_sub(i,(1))*(1)+tc_sub(tc_sub(j,1),(1))*tc_mul(1,((7)-(1)+1))+tc_sub(k,(1))*(1*((7)-(1)+1)*((9)-(1)+1))+tc_sub(l,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1))+tc_sub(m,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1)*((5)-(1)+1))];
// MPAS module_mp_thompson.F:4447
c = tnccn_act[0+tc_sub(i,(1))*(1)+tc_sub(j,(1))*tc_mul(1,((7)-(1)+1))+tc_sub(k,(1))*(1*((7)-(1)+1)*((9)-(1)+1))+tc_sub(l,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1))+tc_sub(m,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1)*((5)-(1)+1))];
// MPAS module_mp_thompson.F:4448
d = tnccn_act[0+tc_sub(tc_sub(i,1),(1))*(1)+tc_sub(j,(1))*tc_mul(1,((7)-(1)+1))+tc_sub(k,(1))*(1*((7)-(1)+1)*((9)-(1)+1))+tc_sub(l,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1))+tc_sub(m,(1))*(1*((7)-(1)+1)*((9)-(1)+1)*((7)-(1)+1)*((5)-(1)+1))];
// MPAS module_mp_thompson.F:4449
nx = tc_log(n_local);
// MPAS module_mp_thompson.F:4450
wy = tc_log(w_local);
// MPAS module_mp_thompson.F:4452
t = tc_div((tc_sub(nx,x1)),(tc_sub(x2,x1)));
// MPAS module_mp_thompson.F:4453
u = tc_div((tc_sub(wy,y1)),(tc_sub(y2,y1)));
// MPAS module_mp_thompson.F:4458
fraction = tc_add(tc_add(tc_add(tc_mul(tc_mul((tc_sub(1.0f,t)),(tc_sub(1.0f,u))),a),tc_mul(tc_mul(t,(tc_sub(1.0f,u))),b)),tc_mul(tc_mul(t,u),c)),tc_mul(tc_mul((tc_sub(1.0f,t)),u),d));
// MPAS module_mp_thompson.F:4464
result = tc_mul(nccn,fraction);
return result;
}
// MPAS module_mp_thompson.F:1400 mp_thompson
__device__ __forceinline__ void mp_thompson(float* qv1d,float* qc1d,float* qi1d,float* qr1d,float* qs1d,float* qg1d,float* ni1d,float* nr1d,float* nc1d,float* nwfa1d,float* nifa1d,float* t1d,float* p1d,float* w1d,float* dzq,float& pptrain,float& pptsnow,float& pptgraul,float& pptice,float* rainprod,float* evapprod,int kts,int kte,float dt,int ii,int jj) {
float tten[THOMPSON_COLUMN_KMAX];
float qvten[THOMPSON_COLUMN_KMAX];
float qcten[THOMPSON_COLUMN_KMAX];
float qiten[THOMPSON_COLUMN_KMAX];
float qrten[THOMPSON_COLUMN_KMAX];
float qsten[THOMPSON_COLUMN_KMAX];
float qgten[THOMPSON_COLUMN_KMAX];
float niten[THOMPSON_COLUMN_KMAX];
float nrten[THOMPSON_COLUMN_KMAX];
float ncten[THOMPSON_COLUMN_KMAX];
float nwfaten[THOMPSON_COLUMN_KMAX];
float nifaten[THOMPSON_COLUMN_KMAX];
double prw_vcd[THOMPSON_COLUMN_KMAX];
double pnc_wcd[THOMPSON_COLUMN_KMAX];
double pnc_wau[THOMPSON_COLUMN_KMAX];
double pnc_rcw[THOMPSON_COLUMN_KMAX];
double pnc_scw[THOMPSON_COLUMN_KMAX];
double pnc_gcw[THOMPSON_COLUMN_KMAX];
double pna_rca[THOMPSON_COLUMN_KMAX];
double pna_sca[THOMPSON_COLUMN_KMAX];
double pna_gca[THOMPSON_COLUMN_KMAX];
double pnd_rcd[THOMPSON_COLUMN_KMAX];
double pnd_scd[THOMPSON_COLUMN_KMAX];
double pnd_gcd[THOMPSON_COLUMN_KMAX];
double prr_wau[THOMPSON_COLUMN_KMAX];
double prr_rcw[THOMPSON_COLUMN_KMAX];
double prr_rcs[THOMPSON_COLUMN_KMAX];
double prr_rcg[THOMPSON_COLUMN_KMAX];
double prr_sml[THOMPSON_COLUMN_KMAX];
double prr_gml[THOMPSON_COLUMN_KMAX];
double prr_rci[THOMPSON_COLUMN_KMAX];
double prv_rev[THOMPSON_COLUMN_KMAX];
double pnr_wau[THOMPSON_COLUMN_KMAX];
double pnr_rcs[THOMPSON_COLUMN_KMAX];
double pnr_rcg[THOMPSON_COLUMN_KMAX];
double pnr_rci[THOMPSON_COLUMN_KMAX];
double pnr_sml[THOMPSON_COLUMN_KMAX];
double pnr_gml[THOMPSON_COLUMN_KMAX];
double pnr_rev[THOMPSON_COLUMN_KMAX];
double pnr_rcr[THOMPSON_COLUMN_KMAX];
double pnr_rfz[THOMPSON_COLUMN_KMAX];
double pri_inu[THOMPSON_COLUMN_KMAX];
double pni_inu[THOMPSON_COLUMN_KMAX];
double pri_ihm[THOMPSON_COLUMN_KMAX];
double pni_ihm[THOMPSON_COLUMN_KMAX];
double pri_wfz[THOMPSON_COLUMN_KMAX];
double pni_wfz[THOMPSON_COLUMN_KMAX];
double pri_rfz[THOMPSON_COLUMN_KMAX];
double pni_rfz[THOMPSON_COLUMN_KMAX];
double pri_ide[THOMPSON_COLUMN_KMAX];
double pni_ide[THOMPSON_COLUMN_KMAX];
double pri_rci[THOMPSON_COLUMN_KMAX];
double pni_rci[THOMPSON_COLUMN_KMAX];
double pni_sci[THOMPSON_COLUMN_KMAX];
double pni_iau[THOMPSON_COLUMN_KMAX];
double pri_iha[THOMPSON_COLUMN_KMAX];
double pni_iha[THOMPSON_COLUMN_KMAX];
double prs_iau[THOMPSON_COLUMN_KMAX];
double prs_sci[THOMPSON_COLUMN_KMAX];
double prs_rcs[THOMPSON_COLUMN_KMAX];
double prs_scw[THOMPSON_COLUMN_KMAX];
double prs_sde[THOMPSON_COLUMN_KMAX];
double prs_ihm[THOMPSON_COLUMN_KMAX];
double prs_ide[THOMPSON_COLUMN_KMAX];
double prg_scw[THOMPSON_COLUMN_KMAX];
double prg_rfz[THOMPSON_COLUMN_KMAX];
double prg_gde[THOMPSON_COLUMN_KMAX];
double prg_gcw[THOMPSON_COLUMN_KMAX];
double prg_rci[THOMPSON_COLUMN_KMAX];
double prg_rcs[THOMPSON_COLUMN_KMAX];
double prg_rcg[THOMPSON_COLUMN_KMAX];
double prg_ihm[THOMPSON_COLUMN_KMAX];
double zerod0 = 0.0e0;
float temp[THOMPSON_COLUMN_KMAX];
float pres[THOMPSON_COLUMN_KMAX];
float qv[THOMPSON_COLUMN_KMAX];
float rc[THOMPSON_COLUMN_KMAX];
float ri[THOMPSON_COLUMN_KMAX];
float rr[THOMPSON_COLUMN_KMAX];
float rs[THOMPSON_COLUMN_KMAX];
float rg[THOMPSON_COLUMN_KMAX];
float ni[THOMPSON_COLUMN_KMAX];
float nr[THOMPSON_COLUMN_KMAX];
float nc[THOMPSON_COLUMN_KMAX];
float nwfa[THOMPSON_COLUMN_KMAX];
float nifa[THOMPSON_COLUMN_KMAX];
float rho[THOMPSON_COLUMN_KMAX];
float rhof[THOMPSON_COLUMN_KMAX];
float rhof2[THOMPSON_COLUMN_KMAX];
float qvs[THOMPSON_COLUMN_KMAX];
float qvsi[THOMPSON_COLUMN_KMAX];
float delqvs[THOMPSON_COLUMN_KMAX];
float satw[THOMPSON_COLUMN_KMAX];
float sati[THOMPSON_COLUMN_KMAX];
float ssatw[THOMPSON_COLUMN_KMAX];
float ssati[THOMPSON_COLUMN_KMAX];
float diffu[THOMPSON_COLUMN_KMAX];
float visco[THOMPSON_COLUMN_KMAX];
float vsc2[THOMPSON_COLUMN_KMAX];
float tcond[THOMPSON_COLUMN_KMAX];
float lvap[THOMPSON_COLUMN_KMAX];
float ocp[THOMPSON_COLUMN_KMAX];
float lvt2[THOMPSON_COLUMN_KMAX];
double ilamr[THOMPSON_COLUMN_KMAX];
double ilamg[THOMPSON_COLUMN_KMAX];
double n0_r[THOMPSON_COLUMN_KMAX];
double n0_g[THOMPSON_COLUMN_KMAX];
float mvd_r[THOMPSON_COLUMN_KMAX];
float mvd_c[THOMPSON_COLUMN_KMAX];
float smob[THOMPSON_COLUMN_KMAX];
float smo2[THOMPSON_COLUMN_KMAX];
float smo1[THOMPSON_COLUMN_KMAX];
float smo0[THOMPSON_COLUMN_KMAX];
float smoc[THOMPSON_COLUMN_KMAX];
float smod[THOMPSON_COLUMN_KMAX];
float smoe[THOMPSON_COLUMN_KMAX];
float smof[THOMPSON_COLUMN_KMAX];
float sed_r[THOMPSON_COLUMN_KMAX];
float sed_s[THOMPSON_COLUMN_KMAX];
float sed_g[THOMPSON_COLUMN_KMAX];
float sed_i[THOMPSON_COLUMN_KMAX];
float sed_n[THOMPSON_COLUMN_KMAX];
float sed_c[THOMPSON_COLUMN_KMAX];
float rgvm;
float delta_tp;
float orho;
float lfus2;
float onstep[5];
double n0_exp;
double n0_min;
double lam_exp;
double lamc;
double lamr;
double lamg;
double lami;
double ilami;
double ilamc;
float xdc;
float dc_b;
float dc_g;
float xdi;
float xdr;
float xds;
float xdg;
float ds_m;
float dg_m;
double dr_star;
double dc_star;
float zeta1;
float zeta;
float taud;
float tau;
float stoke_r;
float stoke_s;
float stoke_g;
float stoke_i;
float vti;
float vtr;
float vts;
float vtg;
float vtc;
float vtik[THOMPSON_COLUMN_KMAX+1];
float vtnik[THOMPSON_COLUMN_KMAX+1];
float vtrk[THOMPSON_COLUMN_KMAX+1];
float vtnrk[THOMPSON_COLUMN_KMAX+1];
float vtsk[THOMPSON_COLUMN_KMAX+1];
float vtgk[THOMPSON_COLUMN_KMAX+1];
float vtck[THOMPSON_COLUMN_KMAX+1];
float vtnck[THOMPSON_COLUMN_KMAX+1];
float vts_boost[THOMPSON_COLUMN_KMAX];
float mrat;
float ils1;
float ils2;
float t1_vts;
float t2_vts;
float t3_vts;
float t4_vts;
float c_snow;
float a_;
float b_;
float loga_;
float a1;
float a2;
float tf;
float tempc;
float tc0;
float r_mvd1;
float r_mvd2;
float xkrat;
float xnc;
float xri;
float xni;
float xmi;
float oxmi;
float xrc;
float xrr;
float xnr;
float xsat;
float rate_max;
float sump;
float ratio;
float clap;
float fcd;
float dfcd;
float otemp;
float rvs;
float rvs_p;
float rvs_pp;
float gamsc;
float alphsc;
float t1_evap;
float t1_subl;
float r_frac;
float g_frac;
float ef_rw;
float ef_sw;
float ef_gw;
float ef_rr;
float ef_ra;
float ef_sa;
float ef_ga;
float dtsave;
float odts;
float odt;
float odzq;
float hgt_agl;
float sr;
float xslw1;
float ygra1;
float zans1;
float eva_factor;
int i;
int k;
int k2;
int n;
int nn;
int nstep;
int k_0;
int kbot;
int it;
int iexfrq;
int ksed1[5];
int nir;
int nis;
int nig;
int nii;
int nic;
int niin;
int idx_tc;
int idx_t;
int idx_s;
int idx_g1;
int idx_g;
int idx_r1;
int idx_r;
int idx_i1;
int idx_i;
int idx_c;
int idx;
int idx_d;
int idx_n;
int idx_in;
bool melti;
bool no_micro;
bool l_qc[THOMPSON_COLUMN_KMAX];
bool l_qi[THOMPSON_COLUMN_KMAX];
bool l_qr[THOMPSON_COLUMN_KMAX];
bool l_qs[THOMPSON_COLUMN_KMAX];
bool l_qg[THOMPSON_COLUMN_KMAX];
bool debug_flag;
int mp_debug;
int nu_c;
bool l_nifa[THOMPSON_COLUMN_KMAX];
bool l_nwfa[THOMPSON_COLUMN_KMAX];
float tem;
// MPAS module_mp_thompson.F:1513
debug_flag = false;
// MPAS module_mp_thompson.F:1515
if(debug_flag) {
// MPAS module_mp_thompson.F:1516
// diagnostic only
// MPAS module_mp_thompson.F:1518
}
// MPAS module_mp_thompson.F:1520
no_micro = true;
// MPAS module_mp_thompson.F:1521
dtsave = dt;
// MPAS module_mp_thompson.F:1522
odt = tc_div(1.f,dt);
// MPAS module_mp_thompson.F:1523
odts = tc_div(1.f,dtsave);
// MPAS module_mp_thompson.F:1524
iexfrq = 1;
// MPAS module_mp_thompson.F:1541
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1542
tten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1543
qvten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1544
qcten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1545
qiten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1546
qrten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1547
qsten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1548
qgten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1549
niten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1550
nrten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1551
ncten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1552
nwfaten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1553
nifaten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1555
prw_vcd[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1557
pnc_wcd[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1558
pnc_wau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1559
pnc_rcw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1560
pnc_scw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1561
pnc_gcw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1563
prv_rev[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1564
prr_wau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1565
prr_rcw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1566
prr_rcs[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1567
prr_rcg[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1568
prr_sml[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1569
prr_gml[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1570
prr_rci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1571
pnr_wau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1572
pnr_rcs[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1573
pnr_rcg[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1574
pnr_rci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1575
pnr_sml[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1576
pnr_gml[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1577
pnr_rev[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1578
pnr_rcr[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1579
pnr_rfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1581
pri_inu[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1582
pni_inu[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1583
pri_ihm[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1584
pni_ihm[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1585
pri_wfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1586
pni_wfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1587
pri_rfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1588
pni_rfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1589
pri_ide[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1590
pni_ide[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1591
pri_rci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1592
pni_rci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1593
pni_sci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1594
pni_iau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1595
pri_iha[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1596
pni_iha[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1598
prs_iau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1599
prs_sci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1600
prs_rcs[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1601
prs_scw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1602
prs_sde[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1603
prs_ihm[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1604
prs_ide[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1606
prg_scw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1607
prg_rfz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1608
prg_gde[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1609
prg_gcw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1610
prg_rci[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1611
prg_rcs[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1612
prg_rcg[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1613
prg_ihm[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1615
pna_rca[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1616
pna_sca[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1617
pna_gca[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1619
pnd_rcd[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1620
pnd_scd[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1621
pnd_gcd[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1622
}
// MPAS module_mp_thompson.F:1623
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1624
rainprod[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1625
evapprod[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1626
}
// MPAS module_mp_thompson.F:1630
if(is_aerosol_aware) {
// MPAS module_mp_thompson.F:1631
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1632
l_nifa[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1633
l_nwfa[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1634
}
// MPAS module_mp_thompson.F:1635
}
// MPAS module_mp_thompson.F:1638
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1639
smo0[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1640
smo1[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1641
smo2[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1642
smob[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1643
smoc[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1644
smod[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1645
smoe[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1646
smof[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1647
mvd_r[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1648
mvd_c[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:1649
}
// MPAS module_mp_thompson.F:1654
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1655
temp[0+tc_sub(k,(kts))*(1)] = t1d[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:1656
qv[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,qv1d[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1657
pres[0+tc_sub(k,(kts))*(1)] = p1d[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:1658
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,pres[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,temp[0+tc_sub(k,(kts))*(1)]),(tc_add(qv[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:1659
nwfa[0+tc_sub(k,(kts))*(1)] = tc_max(11.1e6f,tc_min(9999.e6f,tc_mul(nwfa1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:1660
nifa[0+tc_sub(k,(kts))*(1)] = tc_max(tc_mul(nain1,0.01f),tc_min(9999.e6f,tc_mul(nifa1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:1662
if((qc1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:1663
no_micro = false;
// MPAS module_mp_thompson.F:1664
rc[0+tc_sub(k,(kts))*(1)] = tc_mul(qc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1665
nc[0+tc_sub(k,(kts))*(1)] = tc_max(2.f,tc_min(tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]),nt_c_max));
// MPAS module_mp_thompson.F:1666
l_qc[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:1668
if((is_aerosol_aware&&(nc[0+tc_sub(k,(kts))*(1)]<=2.f))) { l_nwfa[0+tc_sub(k,(kts))*(1)] = true; }
// MPAS module_mp_thompson.F:1669
nu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,nc[0+tc_sub(k,(kts))*(1)]))),2));
// MPAS module_mp_thompson.F:1670
lamc = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(nc[0+tc_sub(k,(kts))*(1)],am_r),ccg[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:1671
xdc = tc_div((tc_add(tc_add(bm_r,nu_c),1.f)),lamc);
// MPAS module_mp_thompson.F:1672
if((xdc<d0c)) {
// MPAS module_mp_thompson.F:1673
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],d0c);
// MPAS module_mp_thompson.F:1674
} else if((xdc>tc_mul(d0r,2.f))) {
// MPAS module_mp_thompson.F:1675
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],(tc_mul(d0r,2.f)));
// MPAS module_mp_thompson.F:1676
}
// MPAS module_mp_thompson.F:1677
nc[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(nt_c_max)),tc_mul(tc_div(tc_mul(tc_mul(ccg[0+tc_sub(1,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],ocg2[0+tc_sub(nu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)]),am_r),tc_pow(lamc,bm_r)));
// MPAS module_mp_thompson.F:1679
if((!is_aerosol_aware)) { nc[0+tc_sub(k,(kts))*(1)] = nt_c; }
// MPAS module_mp_thompson.F:1680
} else {
// MPAS module_mp_thompson.F:1681
qc1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1682
nc1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1683
rc[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:1684
nc[0+tc_sub(k,(kts))*(1)] = 2.f;
// MPAS module_mp_thompson.F:1685
l_qc[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1686
}
// MPAS module_mp_thompson.F:1688
if((qi1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:1689
no_micro = false;
// MPAS module_mp_thompson.F:1690
ri[0+tc_sub(k,(kts))*(1)] = tc_mul(qi1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1691
ni[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1692
l_qi[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:1694
if((is_aerosol_aware&&(ni[0+tc_sub(k,(kts))*(1)]<=r2))) { l_nifa[0+tc_sub(k,(kts))*(1)] = true; }
// MPAS module_mp_thompson.F:1695
if((ni[0+tc_sub(k,(kts))*(1)]<=r2)) {
// MPAS module_mp_thompson.F:1696
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],5.e-6f);
// MPAS module_mp_thompson.F:1697
ni[0+tc_sub(k,(kts))*(1)] = tc_min(9999.e3,tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),ri[0+tc_sub(k,(kts))*(1)]),am_i),tc_pow(lami,bm_i)));
// MPAS module_mp_thompson.F:1698
}
// MPAS module_mp_thompson.F:1700
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni[0+tc_sub(k,(kts))*(1)]),ri[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:1701
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:1702
xdi = tc_mul((tc_add(tc_add(bm_i,mu_i),1.f)),ilami);
// MPAS module_mp_thompson.F:1703
if((xdi<5.e-6f)) {
// MPAS module_mp_thompson.F:1704
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],5.e-6f);
// MPAS module_mp_thompson.F:1705
ni[0+tc_sub(k,(kts))*(1)] = tc_min(9999.e3,tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),ri[0+tc_sub(k,(kts))*(1)]),am_i),tc_pow(lami,bm_i)));
// MPAS module_mp_thompson.F:1706
} else if((xdi>300.e-6f)) {
// MPAS module_mp_thompson.F:1707
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],300.e-6f);
// MPAS module_mp_thompson.F:1708
ni[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),ri[0+tc_sub(k,(kts))*(1)]),am_i),tc_pow(lami,bm_i));
// MPAS module_mp_thompson.F:1709
}
// MPAS module_mp_thompson.F:1710
} else {
// MPAS module_mp_thompson.F:1711
qi1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1712
ni1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1713
ri[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:1714
ni[0+tc_sub(k,(kts))*(1)] = r2;
// MPAS module_mp_thompson.F:1715
l_qi[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1716
}
// MPAS module_mp_thompson.F:1718
if((qr1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:1719
no_micro = false;
// MPAS module_mp_thompson.F:1720
rr[0+tc_sub(k,(kts))*(1)] = tc_mul(qr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1721
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul(nr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1722
if((nr[0+tc_sub(k,(kts))*(1)]<=r2)) {
// MPAS module_mp_thompson.F:1723
mvd_r[0+tc_sub(k,(kts))*(1)] = 1.0e-3f;
// MPAS module_mp_thompson.F:1724
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1725
nr[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),rr[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:1726
}
// MPAS module_mp_thompson.F:1727
l_qr[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:1728
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:1729
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:1730
if((mvd_r[0+tc_sub(k,(kts))*(1)]>2.5e-3f)) {
// MPAS module_mp_thompson.F:1731
mvd_r[0+tc_sub(k,(kts))*(1)] = 2.5e-3f;
// MPAS module_mp_thompson.F:1732
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1733
nr[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),rr[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:1734
} else if((mvd_r[0+tc_sub(k,(kts))*(1)]<tc_mul(d0r,0.75f))) {
// MPAS module_mp_thompson.F:1735
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_mul(d0r,0.75f);
// MPAS module_mp_thompson.F:1736
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1737
nr[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),rr[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:1738
}
// MPAS module_mp_thompson.F:1739
} else {
// MPAS module_mp_thompson.F:1740
qr1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1741
nr1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1742
rr[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:1743
nr[0+tc_sub(k,(kts))*(1)] = r2;
// MPAS module_mp_thompson.F:1744
l_qr[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1745
}
// MPAS module_mp_thompson.F:1746
if((qs1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:1747
no_micro = false;
// MPAS module_mp_thompson.F:1748
rs[0+tc_sub(k,(kts))*(1)] = tc_mul(qs1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1749
l_qs[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:1750
} else {
// MPAS module_mp_thompson.F:1751
qs1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1752
rs[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:1753
l_qs[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1754
}
// MPAS module_mp_thompson.F:1755
if((qg1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:1756
no_micro = false;
// MPAS module_mp_thompson.F:1757
rg[0+tc_sub(k,(kts))*(1)] = tc_mul(qg1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1758
l_qg[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:1759
} else {
// MPAS module_mp_thompson.F:1760
qg1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:1761
rg[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:1762
l_qg[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:1763
}
// MPAS module_mp_thompson.F:1764
}
// MPAS module_mp_thompson.F:1784
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1785
tempc = tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f);
// MPAS module_mp_thompson.F:1786
rhof[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho_not,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1787
rhof2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(rhof[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1788
qvs[0+tc_sub(k,(kts))*(1)] = rslf(pres[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1789
delqvs[0+tc_sub(k,(kts))*(1)] = tc_max(0.0f,tc_sub(rslf(pres[0+tc_sub(k,(kts))*(1)],273.15f),qv[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1790
if((tempc<=0.0f)) {
// MPAS module_mp_thompson.F:1791
qvsi[0+tc_sub(k,(kts))*(1)] = rsif(pres[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1792
} else {
// MPAS module_mp_thompson.F:1793
qvsi[0+tc_sub(k,(kts))*(1)] = qvs[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:1794
}
// MPAS module_mp_thompson.F:1795
satw[0+tc_sub(k,(kts))*(1)] = tc_div(qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1796
sati[0+tc_sub(k,(kts))*(1)] = tc_div(qv[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1797
ssatw[0+tc_sub(k,(kts))*(1)] = tc_sub(satw[0+tc_sub(k,(kts))*(1)],1.f);
// MPAS module_mp_thompson.F:1798
ssati[0+tc_sub(k,(kts))*(1)] = tc_sub(sati[0+tc_sub(k,(kts))*(1)],1.f);
// MPAS module_mp_thompson.F:1799
if((tc_abs(ssatw[0+tc_sub(k,(kts))*(1)])<eps)) { ssatw[0+tc_sub(k,(kts))*(1)] = 0.0f; }
// MPAS module_mp_thompson.F:1800
if((tc_abs(ssati[0+tc_sub(k,(kts))*(1)])<eps)) { ssati[0+tc_sub(k,(kts))*(1)] = 0.0f; }
// MPAS module_mp_thompson.F:1801
if((no_micro&&(ssati[0+tc_sub(k,(kts))*(1)]>0.0f))) { no_micro = false; }
// MPAS module_mp_thompson.F:1802
diffu[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(2.11e-5f,tc_pow((tc_div(temp[0+tc_sub(k,(kts))*(1)],273.15f)),1.94f)),(tc_div(101325.f,pres[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:1803
if((tempc>=0.0f)) {
// MPAS module_mp_thompson.F:1804
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(1.718f,tc_mul(0.0049f,tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:1805
} else {
// MPAS module_mp_thompson.F:1806
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(tc_add(1.718f,tc_mul(0.0049f,tempc)),tc_mul(tc_mul(1.2e-5f,tempc),tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:1807
}
// MPAS module_mp_thompson.F:1808
ocp[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,(tc_mul(cp,(tc_add(1.f,tc_mul(0.887f,qv[0+tc_sub(k,(kts))*(1)]))))));
// MPAS module_mp_thompson.F:1809
vsc2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho[0+tc_sub(k,(kts))*(1)],visco[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1810
lvap[0+tc_sub(k,(kts))*(1)] = tc_add(lvap0,tc_mul((tc_sub(2106.0f,4218.0f)),tempc));
// MPAS module_mp_thompson.F:1811
tcond[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_add(5.69f,tc_mul(0.0168f,tempc))),1.0e-5f),418.936f);
// MPAS module_mp_thompson.F:1812
}
// MPAS module_mp_thompson.F:1819
if(no_micro) { return; }
// MPAS module_mp_thompson.F:1824
if((!iiwarm)) {
// MPAS module_mp_thompson.F:1825
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1826
if((!l_qs[0+tc_sub(k,(kts))*(1)])) { continue; }
// MPAS module_mp_thompson.F:1827
tc0 = tc_min((-0.1f),tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f));
// MPAS module_mp_thompson.F:1828
smob[0+tc_sub(k,(kts))*(1)] = tc_mul(rs[0+tc_sub(k,(kts))*(1)],oams);
// MPAS module_mp_thompson.F:1832
if(((bm_s>(tc_sub(2.0f,1.e-3f)))&&(bm_s<(tc_add(2.0f,1.e-3f))))) {
// MPAS module_mp_thompson.F:1833
smo2[0+tc_sub(k,(kts))*(1)] = smob[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:1834
} else {
// MPAS module_mp_thompson.F:1835
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:1840
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1841
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:1846
smo2[0+tc_sub(k,(kts))*(1)] = tc_pow((tc_div(smob[0+tc_sub(k,(kts))*(1)],a_)),(tc_div(1.f,b_)));
// MPAS module_mp_thompson.F:1847
}
// MPAS module_mp_thompson.F:1850
loga_ = tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0));
// MPAS module_mp_thompson.F:1851
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1852
b_ = tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0));
// MPAS module_mp_thompson.F:1853
smo0[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:1856
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),sa[0+tc_sub(3,(1))*(1)]),tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),sa[0+tc_sub(6,(1))*(1)]),tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0)),tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),sa[0+tc_sub(10,(1))*(1)]);
// MPAS module_mp_thompson.F:1861
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1862
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),sb[0+tc_sub(3,(1))*(1)]),tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),sb[0+tc_sub(6,(1))*(1)]),tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0)),tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),sb[0+tc_sub(10,(1))*(1)]);
// MPAS module_mp_thompson.F:1866
smo1[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:1869
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:1874
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1875
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:1879
smoc[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:1882
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)]));
// MPAS module_mp_thompson.F:1887
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1888
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)]),cse[0+tc_sub(13,(1))*(1)]));
// MPAS module_mp_thompson.F:1892
smoe[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:1895
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)]));
// MPAS module_mp_thompson.F:1900
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:1901
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)]),cse[0+tc_sub(16,(1))*(1)]));
// MPAS module_mp_thompson.F:1905
smof[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:1907
}
// MPAS module_mp_thompson.F:1912
n0_min = gonv_max;
// MPAS module_mp_thompson.F:1913
k_0 = kts;
// MPAS module_mp_thompson.F:1914
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:1915
if((temp[0+tc_sub(k,(kts))*(1)]>=270.65f)) { k_0 = tc_max(k_0,k); }
// MPAS module_mp_thompson.F:1916
}
// MPAS module_mp_thompson.F:1917
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:1918
if((((k>k_0)&&l_qr[0+tc_sub(k,(kts))*(1)])&&(mvd_r[0+tc_sub(k,(kts))*(1)]>100.e-6f))) {
// MPAS module_mp_thompson.F:1919
xslw1 = tc_add(4.01f,tc_log10(mvd_r[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:1920
} else {
// MPAS module_mp_thompson.F:1921
xslw1 = 0.01f;
// MPAS module_mp_thompson.F:1922
}
// MPAS module_mp_thompson.F:1923
ygra1 = tc_add(4.31f,tc_log10(tc_max(5.e-5f,rg[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:1924
zans1 = tc_add(3.1f,(tc_div(100.f,(tc_add(tc_add(tc_div(tc_mul(tc_mul(300.f,xslw1),ygra1),(tc_add(tc_add(tc_div(10.f,xslw1),1.f),tc_mul(0.25f,ygra1)))),30.f),tc_mul(10.f,ygra1))))));
// MPAS module_mp_thompson.F:1925
n0_exp = tc_pow(10.f,(zans1));
// MPAS module_mp_thompson.F:1926
n0_exp = tc_max(((double)(gonv_min)),tc_min(n0_exp,((double)(gonv_max))));
// MPAS module_mp_thompson.F:1927
n0_min = tc_min(n0_exp,n0_min);
// MPAS module_mp_thompson.F:1928
n0_exp = n0_min;
// MPAS module_mp_thompson.F:1929
lam_exp = tc_pow((tc_div(tc_mul(tc_mul(n0_exp,am_g),cgg[0+tc_sub(1,(1))*(1)]),rg[0+tc_sub(k,(kts))*(1)])),oge1);
// MPAS module_mp_thompson.F:1930
lamg = tc_mul(lam_exp,tc_pow((tc_mul(tc_mul(cgg[0+tc_sub(3,(1))*(1)],ogg2),ogg1)),obmg));
// MPAS module_mp_thompson.F:1931
ilamg[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamg);
// MPAS module_mp_thompson.F:1932
n0_g[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(n0_exp,(tc_mul(cgg[0+tc_sub(2,(1))*(1)],lam_exp))),tc_pow(lamg,cge[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:1933
}
// MPAS module_mp_thompson.F:1935
}
// MPAS module_mp_thompson.F:1940
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:1941
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:1942
ilamr[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamr);
// MPAS module_mp_thompson.F:1943
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:1944
n0_r[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(nr[0+tc_sub(k,(kts))*(1)],org2),tc_pow(lamr,cre[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:1945
}
// MPAS module_mp_thompson.F:1951
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:1955
if((l_qr[0+tc_sub(k,(kts))*(1)]&&(mvd_r[0+tc_sub(k,(kts))*(1)]>d0r))) {
// MPAS module_mp_thompson.F:1958
ef_rr = tc_sub(1.0f,tc_exp(tc_mul(2300.0f,(tc_sub(mvd_r[0+tc_sub(k,(kts))*(1)],1950.0e-6f)))));
// MPAS module_mp_thompson.F:1960
pnr_rcr[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(ef_rr,2.0f),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1961
}
// MPAS module_mp_thompson.F:1963
mvd_c[0+tc_sub(k,(kts))*(1)] = d0c;
// MPAS module_mp_thompson.F:1964
if(l_qc[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:1965
nu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,nc[0+tc_sub(k,(kts))*(1)]))),2));
// MPAS module_mp_thompson.F:1966
xdc = tc_max(tc_mul(d0c,1.e6f),tc_mul((tc_pow((tc_div(rc[0+tc_sub(k,(kts))*(1)],(tc_mul(am_r,nc[0+tc_sub(k,(kts))*(1)])))),obmr)),1.e6f));
// MPAS module_mp_thompson.F:1967
lamc = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(nc[0+tc_sub(k,(kts))*(1)],am_r),ccg[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:1968
mvd_c[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,nu_c),0.672f)),lamc);
// MPAS module_mp_thompson.F:1969
}
// MPAS module_mp_thompson.F:1973
if((rc[0+tc_sub(k,(kts))*(1)]>0.01e-3f)) {
// MPAS module_mp_thompson.F:1974
dc_g = tc_mul((tc_div(tc_pow((tc_mul(ccg[0+tc_sub(3,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],ocg2[0+tc_sub(nu_c,(1))*(1)])),obmr),lamc)),1.e6f);
// MPAS module_mp_thompson.F:1975
dc_b = tc_pow((tc_sub(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(xdc,xdc),xdc),dc_g),dc_g),dc_g),tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(xdc,xdc),xdc),xdc),xdc),xdc))),(tc_div(1.f,6.f)));
// MPAS module_mp_thompson.F:1977
zeta1 = tc_mul(0.5f,(tc_add((tc_sub(tc_mul(tc_mul(tc_mul(tc_mul(6.25e-6f,xdc),dc_b),dc_b),dc_b),0.4f)),tc_abs(tc_sub(tc_mul(tc_mul(tc_mul(tc_mul(6.25e-6f,xdc),dc_b),dc_b),dc_b),0.4f)))));
// MPAS module_mp_thompson.F:1979
zeta = tc_mul(tc_mul(0.027f,rc[0+tc_sub(k,(kts))*(1)]),zeta1);
// MPAS module_mp_thompson.F:1980
taud = tc_add(tc_mul(0.5f,(tc_add((tc_sub(tc_mul(0.5f,dc_b),7.5f)),tc_abs(tc_sub(tc_mul(0.5f,dc_b),7.5f))))),r1);
// MPAS module_mp_thompson.F:1981
tau = tc_div(3.72f,(tc_mul(rc[0+tc_sub(k,(kts))*(1)],taud)));
// MPAS module_mp_thompson.F:1982
prr_wau[0+tc_sub(k,(kts))*(1)] = tc_div(zeta,tau);
// MPAS module_mp_thompson.F:1983
prr_wau[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rc[0+tc_sub(k,(kts))*(1)],odts))),prr_wau[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1984
pnr_wau[0+tc_sub(k,(kts))*(1)] = tc_div(prr_wau[0+tc_sub(k,(kts))*(1)],(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(am_r,nu_c),200.f),d0r),d0r),d0r)));
// MPAS module_mp_thompson.F:1985
pnc_wau[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts))),tc_div(prr_wau[0+tc_sub(k,(kts))*(1)],(tc_mul(tc_mul(tc_mul(am_r,mvd_c[0+tc_sub(k,(kts))*(1)]),mvd_c[0+tc_sub(k,(kts))*(1)]),mvd_c[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:1987
}
// MPAS module_mp_thompson.F:1990
if(((l_qr[0+tc_sub(k,(kts))*(1)]&&(mvd_r[0+tc_sub(k,(kts))*(1)]>d0r))&&(mvd_c[0+tc_sub(k,(kts))*(1)]>d0c))) {
// MPAS module_mp_thompson.F:1991
lamr = tc_div(1.f,ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1992
idx = tc_add(1,((int)(tc_div(tc_mul(nbr,tc_log(tc_div(mvd_r[0+tc_sub(k,(kts))*(1)],dr[0+tc_sub(1,(1))*(1)]))),tc_log(tc_div(dr[0+tc_sub(nbr,(1))*(1)],dr[0+tc_sub(1,(1))*(1)]))))));
// MPAS module_mp_thompson.F:1993
idx = tc_min(idx,nbr);
// MPAS module_mp_thompson.F:1994
ef_rw = t_efrw[0+tc_sub(idx,(1))*(1)+tc_sub(((int)(tc_mul(mvd_c[0+tc_sub(k,(kts))*(1)],1.e6f))),(1))*tc_mul(1,((100)-(1)+1))];
// MPAS module_mp_thompson.F:1995
prr_rcw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qc),ef_rw),rc[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:1997
prr_rcw[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rc[0+tc_sub(k,(kts))*(1)],odts))),prr_rcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:1998
pnc_rcw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qc),ef_rw),nc[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:2000
pnc_rcw[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts))),pnc_rcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2001
}
// MPAS module_mp_thompson.F:2004
if((l_qr[0+tc_sub(k,(kts))*(1)]&&(mvd_r[0+tc_sub(k,(kts))*(1)]>d0r))) {
// MPAS module_mp_thompson.F:2005
ef_ra = eff_aero(mvd_r[0+tc_sub(k,(kts))*(1)],0.04e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],114);
// MPAS module_mp_thompson.F:2006
lamr = tc_div(1.f,ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2007
pna_rca[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qc),ef_ra),nwfa[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:2009
pna_rca[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nwfa[0+tc_sub(k,(kts))*(1)],odts))),pna_rca[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2011
ef_ra = eff_aero(mvd_r[0+tc_sub(k,(kts))*(1)],0.8e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],114);
// MPAS module_mp_thompson.F:2012
pnd_rcd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qc),ef_ra),nifa[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:2014
pnd_rcd[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nifa[0+tc_sub(k,(kts))*(1)],odts))),pnd_rcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2015
}
// MPAS module_mp_thompson.F:2017
}
// MPAS module_mp_thompson.F:2022
if((!iiwarm)) {
// MPAS module_mp_thompson.F:2025
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2026
vts_boost[0+tc_sub(k,(kts))*(1)] = 1.0f;
// MPAS module_mp_thompson.F:2027
xds = 0.0f;
// MPAS module_mp_thompson.F:2028
if(l_qs[0+tc_sub(k,(kts))*(1)]) { xds = tc_div(smoc[0+tc_sub(k,(kts))*(1)],smob[0+tc_sub(k,(kts))*(1)]); }
// MPAS module_mp_thompson.F:2031
tempc = tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f);
// MPAS module_mp_thompson.F:2032
idx_tc = tc_max(1,tc_min(((int)round((-tempc))),45));
// MPAS module_mp_thompson.F:2033
idx_t = tc_sub(((int)(tc_div((tc_sub(tempc,2.5f)),5.f))),1);
// MPAS module_mp_thompson.F:2034
idx_t = tc_max(1,(-idx_t));
// MPAS module_mp_thompson.F:2035
idx_t = tc_min(idx_t,ntb_t);
// MPAS module_mp_thompson.F:2036
it = tc_max(1,tc_min(((int)round((-tempc))),31));
// MPAS module_mp_thompson.F:2039
if((rc[0+tc_sub(k,(kts))*(1)]>r_c[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2040
nic = ((int)round(tc_log10(rc[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2041
for(nn=tc_sub(nic,1); nn<=tc_add(nic,1); nn+=1) {
// MPAS module_mp_thompson.F:2042
n = nn;
// MPAS module_mp_thompson.F:2043
if((((tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_141; }
// MPAS module_mp_thompson.F:2045
}
// MPAS module_mp_thompson.F:2046
label_141: ;

// MPAS module_mp_thompson.F:2047
idx_c = tc_sub(tc_add(((int)(tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nic2)))),(tc_sub(n,nic2)));
// MPAS module_mp_thompson.F:2048
idx_c = tc_max(1,tc_min(idx_c,ntb_c));
// MPAS module_mp_thompson.F:2049
} else {
// MPAS module_mp_thompson.F:2050
idx_c = 1;
// MPAS module_mp_thompson.F:2051
}
// MPAS module_mp_thompson.F:2054
idx_n = ((int)round(tc_add(1.0f,tc_div(tc_mul(float(nbc),tc_log(tc_div(nc[0+tc_sub(k,(kts))*(1)],t_nc[0+tc_sub(1,(1))*(1)]))),nic1))));
// MPAS module_mp_thompson.F:2055
idx_n = tc_max(1,tc_min(idx_n,nbc));
// MPAS module_mp_thompson.F:2058
if((ri[0+tc_sub(k,(kts))*(1)]>r_i[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2059
nii = ((int)round(tc_log10(ri[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2060
for(nn=tc_sub(nii,1); nn<=tc_add(nii,1); nn+=1) {
// MPAS module_mp_thompson.F:2061
n = nn;
// MPAS module_mp_thompson.F:2062
if((((tc_div(ri[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(ri[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_142; }
// MPAS module_mp_thompson.F:2064
}
// MPAS module_mp_thompson.F:2065
label_142: ;

// MPAS module_mp_thompson.F:2066
idx_i = tc_sub(tc_add(((int)(tc_div(ri[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nii2)))),(tc_sub(n,nii2)));
// MPAS module_mp_thompson.F:2067
idx_i = tc_max(1,tc_min(idx_i,ntb_i));
// MPAS module_mp_thompson.F:2068
} else {
// MPAS module_mp_thompson.F:2069
idx_i = 1;
// MPAS module_mp_thompson.F:2070
}
// MPAS module_mp_thompson.F:2072
if((ni[0+tc_sub(k,(kts))*(1)]>nt_i[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2073
nii = ((int)round(tc_log10(ni[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2074
for(nn=tc_sub(nii,1); nn<=tc_add(nii,1); nn+=1) {
// MPAS module_mp_thompson.F:2075
n = nn;
// MPAS module_mp_thompson.F:2076
if((((tc_div(ni[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(ni[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_143; }
// MPAS module_mp_thompson.F:2078
}
// MPAS module_mp_thompson.F:2079
label_143: ;

// MPAS module_mp_thompson.F:2080
idx_i1 = tc_sub(tc_add(((int)(tc_div(ni[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nii3)))),(tc_sub(n,nii3)));
// MPAS module_mp_thompson.F:2081
idx_i1 = tc_max(1,tc_min(idx_i1,ntb_i1));
// MPAS module_mp_thompson.F:2082
} else {
// MPAS module_mp_thompson.F:2083
idx_i1 = 1;
// MPAS module_mp_thompson.F:2084
}
// MPAS module_mp_thompson.F:2087
if((rr[0+tc_sub(k,(kts))*(1)]>r_r[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2088
nir = ((int)round(tc_log10(rr[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2089
for(nn=tc_sub(nir,1); nn<=tc_add(nir,1); nn+=1) {
// MPAS module_mp_thompson.F:2090
n = nn;
// MPAS module_mp_thompson.F:2091
if((((tc_div(rr[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(rr[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_144; }
// MPAS module_mp_thompson.F:2093
}
// MPAS module_mp_thompson.F:2094
label_144: ;

// MPAS module_mp_thompson.F:2095
idx_r = tc_sub(tc_add(((int)(tc_div(rr[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nir2)))),(tc_sub(n,nir2)));
// MPAS module_mp_thompson.F:2096
idx_r = tc_max(1,tc_min(idx_r,ntb_r));
// MPAS module_mp_thompson.F:2098
lamr = tc_div(1.f,ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2099
lam_exp = tc_mul(lamr,tc_pow((tc_mul(tc_mul(crg[0+tc_sub(3,(1))*(1)],org2),org1)),bm_r));
// MPAS module_mp_thompson.F:2100
n0_exp = tc_mul(tc_div(tc_mul(org1,rr[0+tc_sub(k,(kts))*(1)]),am_r),tc_pow(lam_exp,cre[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:2101
nir = ((int)round(tc_log10(n0_exp)));
// MPAS module_mp_thompson.F:2102
for(nn=tc_sub(nir,1); nn<=tc_add(nir,1); nn+=1) {
// MPAS module_mp_thompson.F:2103
n = nn;
// MPAS module_mp_thompson.F:2104
if((((tc_div(n0_exp,tc_pow(10.f,nn)))>=1.0f)&&((tc_div(n0_exp,tc_pow(10.f,nn)))<10.0f))) { goto label_145; }
// MPAS module_mp_thompson.F:2106
}
// MPAS module_mp_thompson.F:2107
label_145: ;

// MPAS module_mp_thompson.F:2108
idx_r1 = tc_sub(tc_add(((int)(tc_div(n0_exp,tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nir3)))),(tc_sub(n,nir3)));
// MPAS module_mp_thompson.F:2109
idx_r1 = tc_max(1,tc_min(idx_r1,ntb_r1));
// MPAS module_mp_thompson.F:2110
} else {
// MPAS module_mp_thompson.F:2111
idx_r = 1;
// MPAS module_mp_thompson.F:2112
idx_r1 = ntb_r1;
// MPAS module_mp_thompson.F:2113
}
// MPAS module_mp_thompson.F:2116
if((rs[0+tc_sub(k,(kts))*(1)]>r_s[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2117
nis = ((int)round(tc_log10(rs[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2118
for(nn=tc_sub(nis,1); nn<=tc_add(nis,1); nn+=1) {
// MPAS module_mp_thompson.F:2119
n = nn;
// MPAS module_mp_thompson.F:2120
if((((tc_div(rs[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(rs[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_146; }
// MPAS module_mp_thompson.F:2122
}
// MPAS module_mp_thompson.F:2123
label_146: ;

// MPAS module_mp_thompson.F:2124
idx_s = tc_sub(tc_add(((int)(tc_div(rs[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nis2)))),(tc_sub(n,nis2)));
// MPAS module_mp_thompson.F:2125
idx_s = tc_max(1,tc_min(idx_s,ntb_s));
// MPAS module_mp_thompson.F:2126
} else {
// MPAS module_mp_thompson.F:2127
idx_s = 1;
// MPAS module_mp_thompson.F:2128
}
// MPAS module_mp_thompson.F:2131
if((rg[0+tc_sub(k,(kts))*(1)]>r_g[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2132
nig = ((int)round(tc_log10(rg[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2133
for(nn=tc_sub(nig,1); nn<=tc_add(nig,1); nn+=1) {
// MPAS module_mp_thompson.F:2134
n = nn;
// MPAS module_mp_thompson.F:2135
if((((tc_div(rg[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(rg[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_147; }
// MPAS module_mp_thompson.F:2137
}
// MPAS module_mp_thompson.F:2138
label_147: ;

// MPAS module_mp_thompson.F:2139
idx_g = tc_sub(tc_add(((int)(tc_div(rg[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nig2)))),(tc_sub(n,nig2)));
// MPAS module_mp_thompson.F:2140
idx_g = tc_max(1,tc_min(idx_g,ntb_g));
// MPAS module_mp_thompson.F:2142
lamg = tc_div(1.f,ilamg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2143
lam_exp = tc_mul(lamg,tc_pow((tc_mul(tc_mul(cgg[0+tc_sub(3,(1))*(1)],ogg2),ogg1)),bm_g));
// MPAS module_mp_thompson.F:2144
n0_exp = tc_mul(tc_div(tc_mul(ogg1,rg[0+tc_sub(k,(kts))*(1)]),am_g),tc_pow(lam_exp,cge[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:2145
nig = ((int)round(tc_log10(n0_exp)));
// MPAS module_mp_thompson.F:2146
for(nn=tc_sub(nig,1); nn<=tc_add(nig,1); nn+=1) {
// MPAS module_mp_thompson.F:2147
n = nn;
// MPAS module_mp_thompson.F:2148
if((((tc_div(n0_exp,tc_pow(10.f,nn)))>=1.0f)&&((tc_div(n0_exp,tc_pow(10.f,nn)))<10.0f))) { goto label_148; }
// MPAS module_mp_thompson.F:2150
}
// MPAS module_mp_thompson.F:2151
label_148: ;

// MPAS module_mp_thompson.F:2152
idx_g1 = tc_sub(tc_add(((int)(tc_div(n0_exp,tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nig3)))),(tc_sub(n,nig3)));
// MPAS module_mp_thompson.F:2153
idx_g1 = tc_max(1,tc_min(idx_g1,ntb_g1));
// MPAS module_mp_thompson.F:2154
} else {
// MPAS module_mp_thompson.F:2155
idx_g = 1;
// MPAS module_mp_thompson.F:2156
idx_g1 = ntb_g1;
// MPAS module_mp_thompson.F:2157
}
// MPAS module_mp_thompson.F:2160
otemp = tc_div(1.f,temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2161
rvs = tc_mul(rho[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2162
rvs_p = tc_mul(tc_mul(rvs,otemp),(tc_sub(tc_mul(tc_mul(lsub,otemp),orv),1.f)));
// MPAS module_mp_thompson.F:2163
rvs_pp = tc_mul(rvs,(tc_add(tc_add(tc_mul(tc_mul(tc_mul(otemp,(tc_sub(tc_mul(tc_mul(lsub,otemp),orv),1.f))),otemp),(tc_sub(tc_mul(tc_mul(lsub,otemp),orv),1.f))),(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul((-2.f),lsub),otemp),otemp),otemp),orv))),tc_mul(otemp,otemp))));
// MPAS module_mp_thompson.F:2167
gamsc = tc_mul(tc_div(tc_mul(lsub,diffu[0+tc_sub(k,(kts))*(1)]),tcond[0+tc_sub(k,(kts))*(1)]),rvs_p);
// MPAS module_mp_thompson.F:2168
alphsc = tc_div(tc_mul(tc_div(tc_mul(tc_mul(tc_mul(0.5f,(tc_div(gamsc,(tc_add(1.f,gamsc))))),(tc_div(gamsc,(tc_add(1.f,gamsc))))),rvs_pp),rvs_p),rvs),rvs_p);
// MPAS module_mp_thompson.F:2170
alphsc = tc_max(1.e-9f,alphsc);
// MPAS module_mp_thompson.F:2171
xsat = ssati[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:2172
if((tc_abs(xsat)<1.e-9f)) { xsat = 0.f; }
// MPAS module_mp_thompson.F:2173
t1_subl = tc_div(tc_mul(tc_mul(4.f,pi),(tc_sub(tc_add(tc_sub(1.0f,tc_mul(alphsc,xsat)),tc_mul(tc_mul(tc_mul(tc_mul(2.f,alphsc),alphsc),xsat),xsat)),tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(5.f,alphsc),alphsc),alphsc),xsat),xsat),xsat)))),(tc_add(1.f,gamsc)));
// MPAS module_mp_thompson.F:2179
if((l_qc[0+tc_sub(k,(kts))*(1)]&&(mvd_c[0+tc_sub(k,(kts))*(1)]>d0c))) {
// MPAS module_mp_thompson.F:2180
if((xds>d0s)) {
// MPAS module_mp_thompson.F:2181
idx = tc_add(1,((int)(tc_div(tc_mul(nbs,tc_log(tc_div(xds,ds[0+tc_sub(1,(1))*(1)]))),tc_log(tc_div(ds[0+tc_sub(nbs,(1))*(1)],ds[0+tc_sub(1,(1))*(1)]))))));
// MPAS module_mp_thompson.F:2182
idx = tc_min(idx,nbs);
// MPAS module_mp_thompson.F:2183
ef_sw = t_efsw[0+tc_sub(idx,(1))*(1)+tc_sub(((int)(tc_mul(mvd_c[0+tc_sub(k,(kts))*(1)],1.e6f))),(1))*tc_mul(1,((100)-(1)+1))];
// MPAS module_mp_thompson.F:2184
prs_scw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qs_qc),ef_sw),rc[0+tc_sub(k,(kts))*(1)]),smoe[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2185
prs_scw[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rc[0+tc_sub(k,(kts))*(1)],odts))),prs_scw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2186
pnc_scw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qs_qc),ef_sw),nc[0+tc_sub(k,(kts))*(1)]),smoe[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2187
pnc_scw[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts))),pnc_scw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2188
}
// MPAS module_mp_thompson.F:2191
if(((rg[0+tc_sub(k,(kts))*(1)]>=r_g[0+tc_sub(1,(1))*(1)])&&(mvd_c[0+tc_sub(k,(kts))*(1)]>d0c))) {
// MPAS module_mp_thompson.F:2192
xdg = tc_mul((tc_add(tc_add(bm_g,mu_g),1.f)),ilamg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2193
vtg = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_g),cgg[0+tc_sub(6,(1))*(1)]),ogg3),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],bv_g));
// MPAS module_mp_thompson.F:2194
stoke_g = tc_div(tc_mul(tc_mul(tc_mul(mvd_c[0+tc_sub(k,(kts))*(1)],mvd_c[0+tc_sub(k,(kts))*(1)]),vtg),rho_w),(tc_mul(tc_mul(9.f,visco[0+tc_sub(k,(kts))*(1)]),xdg)));
// MPAS module_mp_thompson.F:2195
if((xdg>d0g)) {
// MPAS module_mp_thompson.F:2196
if(((stoke_g>=0.4f)&&(stoke_g<=10.f))) {
// MPAS module_mp_thompson.F:2197
ef_gw = tc_mul(0.55f,tc_log10(tc_mul(2.51f,stoke_g)));
// MPAS module_mp_thompson.F:2198
} else if((stoke_g<0.4f)) {
// MPAS module_mp_thompson.F:2199
ef_gw = 0.0f;
// MPAS module_mp_thompson.F:2200
} else if((stoke_g>10)) {
// MPAS module_mp_thompson.F:2201
ef_gw = 0.77f;
// MPAS module_mp_thompson.F:2202
}
// MPAS module_mp_thompson.F:2203
prg_gcw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qg_qc),ef_gw),rc[0+tc_sub(k,(kts))*(1)]),n0_g[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(9,(1))*(1)]));
// MPAS module_mp_thompson.F:2205
pnc_gcw[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qg_qc),ef_gw),nc[0+tc_sub(k,(kts))*(1)]),n0_g[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(9,(1))*(1)]));
// MPAS module_mp_thompson.F:2207
pnc_gcw[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts))),pnc_gcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2208
}
// MPAS module_mp_thompson.F:2209
}
// MPAS module_mp_thompson.F:2210
}
// MPAS module_mp_thompson.F:2213
if((rs[0+tc_sub(k,(kts))*(1)]>r_s[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2214
ef_sa = eff_aero(xds,0.04e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],115);
// MPAS module_mp_thompson.F:2215
pna_sca[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qs_qc),ef_sa),nwfa[0+tc_sub(k,(kts))*(1)]),smoe[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2216
pna_sca[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nwfa[0+tc_sub(k,(kts))*(1)],odts))),pna_sca[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2218
ef_sa = eff_aero(xds,0.8e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],115);
// MPAS module_mp_thompson.F:2219
pnd_scd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qs_qc),ef_sa),nifa[0+tc_sub(k,(kts))*(1)]),smoe[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2220
pnd_scd[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nifa[0+tc_sub(k,(kts))*(1)],odts))),pnd_scd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2221
}
// MPAS module_mp_thompson.F:2222
if((rg[0+tc_sub(k,(kts))*(1)]>r_g[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2223
xdg = tc_mul((tc_add(tc_add(bm_g,mu_g),1.f)),ilamg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2224
ef_ga = eff_aero(xdg,0.04e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],103);
// MPAS module_mp_thompson.F:2225
pna_gca[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qg_qc),ef_ga),nwfa[0+tc_sub(k,(kts))*(1)]),n0_g[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(9,(1))*(1)]));
// MPAS module_mp_thompson.F:2227
pna_gca[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nwfa[0+tc_sub(k,(kts))*(1)],odts))),pna_gca[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2229
ef_ga = eff_aero(xdg,0.8e-6f,visco[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)],103);
// MPAS module_mp_thompson.F:2230
pnd_gcd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qg_qc),ef_ga),nifa[0+tc_sub(k,(kts))*(1)]),n0_g[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(9,(1))*(1)]));
// MPAS module_mp_thompson.F:2232
pnd_gcd[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nifa[0+tc_sub(k,(kts))*(1)],odts))),pnd_gcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2233
}
// MPAS module_mp_thompson.F:2238
if((rr[0+tc_sub(k,(kts))*(1)]>=r_r[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2239
if((rs[0+tc_sub(k,(kts))*(1)]>=r_s[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2240
if((temp[0+tc_sub(k,(kts))*(1)]<t_0)) {
// MPAS module_mp_thompson.F:2241
prr_rcs[0+tc_sub(k,(kts))*(1)] = (-(tc_add(tc_add(tc_add(tmr_racs2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))],tcr_sacr2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tmr_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tcr_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))])));
// MPAS module_mp_thompson.F:2245
prs_rcs[0+tc_sub(k,(kts))*(1)] = tc_sub(tc_sub(tc_add(tmr_racs2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))],tcr_sacr2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tcs_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tms_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2249
prg_rcs[0+tc_sub(k,(kts))*(1)] = tc_add(tc_add(tc_add(tmr_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))],tcr_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tcs_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tms_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2253
prr_rcs[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rr[0+tc_sub(k,(kts))*(1)]),odts))),prr_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2254
prs_rcs[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rs[0+tc_sub(k,(kts))*(1)]),odts))),prs_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2255
prg_rcs[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul((tc_add(rr[0+tc_sub(k,(kts))*(1)],rs[0+tc_sub(k,(kts))*(1)])),odts))),prg_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2256
pnr_rcs[0+tc_sub(k,(kts))*(1)] = tc_add(tc_add(tc_add(tnr_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))],tnr_racs2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tnr_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tnr_sacr2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2260
pnr_rcs[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nr[0+tc_sub(k,(kts))*(1)],odts))),pnr_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2261
} else {
// MPAS module_mp_thompson.F:2262
prs_rcs[0+tc_sub(k,(kts))*(1)] = tc_add(tc_add(tc_sub((-tcs_racs1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tms_sacr1[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tmr_racs2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]),tcr_sacr2[0+tc_sub(idx_s,(1))*(1)+tc_sub(idx_t,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((9)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((9)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2266
prs_rcs[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rs[0+tc_sub(k,(kts))*(1)]),odts))),prs_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2267
prr_rcs[0+tc_sub(k,(kts))*(1)] = (-prs_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2268
}
// MPAS module_mp_thompson.F:2269
}
// MPAS module_mp_thompson.F:2274
if((rg[0+tc_sub(k,(kts))*(1)]>=r_g[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2275
if((temp[0+tc_sub(k,(kts))*(1)]<t_0)) {
// MPAS module_mp_thompson.F:2276
prg_rcg[0+tc_sub(k,(kts))*(1)] = tc_add(tmr_racg[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))],tcr_gacr[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2278
prg_rcg[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rr[0+tc_sub(k,(kts))*(1)],odts))),prg_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2279
prr_rcg[0+tc_sub(k,(kts))*(1)] = (-prg_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2280
pnr_rcg[0+tc_sub(k,(kts))*(1)] = tc_add(tnr_racg[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))],tnr_gacr[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2282
pnr_rcg[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nr[0+tc_sub(k,(kts))*(1)],odts))),pnr_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2283
} else {
// MPAS module_mp_thompson.F:2284
prr_rcg[0+tc_sub(k,(kts))*(1)] = tcg_racg[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))];
// MPAS module_mp_thompson.F:2285
prr_rcg[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rg[0+tc_sub(k,(kts))*(1)],odts))),prr_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2286
prg_rcg[0+tc_sub(k,(kts))*(1)] = (-prr_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2293
pnr_rcg[0+tc_sub(k,(kts))*(1)] = tc_mul((-5.0f),tnr_gacr[0+tc_sub(idx_g1,(1))*(1)+tc_sub(idx_g,(1))*tc_mul(1,((28)-(1)+1))+tc_sub(idx_r1,(1))*(1*((28)-(1)+1)*((28)-(1)+1))+tc_sub(idx_r,(1))*(1*((28)-(1)+1)*((28)-(1)+1)*((37)-(1)+1))]);
// MPAS module_mp_thompson.F:2297
}
// MPAS module_mp_thompson.F:2298
}
// MPAS module_mp_thompson.F:2299
}
// MPAS module_mp_thompson.F:2305
if((temp[0+tc_sub(k,(kts))*(1)]<t_0)) {
// MPAS module_mp_thompson.F:2307
vts_boost[0+tc_sub(k,(kts))*(1)] = 1.0f;
// MPAS module_mp_thompson.F:2308
rate_max = tc_mul(tc_mul(tc_mul((tc_sub(qv[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)])),rho[0+tc_sub(k,(kts))*(1)]),odts),0.999f);
// MPAS module_mp_thompson.F:2324
if((dustyice&&is_aerosol_aware)) {
// MPAS module_mp_thompson.F:2325
xni = icedemott(tempc,qvs[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],nifa[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2326
} else {
// MPAS module_mp_thompson.F:2327
xni = tc_mul(1.0f,1000.f);
// MPAS module_mp_thompson.F:2328
}
// MPAS module_mp_thompson.F:2331
if((xni>nt_in[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2332
niin = ((int)round(tc_log10(xni)));
// MPAS module_mp_thompson.F:2333
for(nn=tc_sub(niin,1); nn<=tc_add(niin,1); nn+=1) {
// MPAS module_mp_thompson.F:2334
n = nn;
// MPAS module_mp_thompson.F:2335
if((((tc_div(xni,tc_pow(10.f,nn)))>=1.0f)&&((tc_div(xni,tc_pow(10.f,nn)))<10.0f))) { goto label_149; }
// MPAS module_mp_thompson.F:2337
}
// MPAS module_mp_thompson.F:2338
label_149: ;

// MPAS module_mp_thompson.F:2339
idx_in = tc_sub(tc_add(((int)(tc_div(xni,tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,niin2)))),(tc_sub(n,niin2)));
// MPAS module_mp_thompson.F:2340
idx_in = tc_max(1,tc_min(idx_in,ntb_in));
// MPAS module_mp_thompson.F:2341
} else {
// MPAS module_mp_thompson.F:2342
idx_in = 1;
// MPAS module_mp_thompson.F:2343
}
// MPAS module_mp_thompson.F:2346
if((rr[0+tc_sub(k,(kts))*(1)]>r_r[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2347
prg_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tpg_qrfz[0+tc_sub(idx_r,(1))*(1)+tc_sub(idx_r1,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((37)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((37)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2348
pri_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tpi_qrfz[0+tc_sub(idx_r,(1))*(1)+tc_sub(idx_r1,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((37)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((37)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2349
pni_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tni_qrfz[0+tc_sub(idx_r,(1))*(1)+tc_sub(idx_r1,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((37)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((37)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2350
pnr_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tnr_qrfz[0+tc_sub(idx_r,(1))*(1)+tc_sub(idx_r1,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((37)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((37)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2351
pnr_rfz[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nr[0+tc_sub(k,(kts))*(1)],odts))),pnr_rfz[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2352
} else if(((rr[0+tc_sub(k,(kts))*(1)]>r1)&&(temp[0+tc_sub(k,(kts))*(1)]<hgfr))) {
// MPAS module_mp_thompson.F:2353
pri_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(rr[0+tc_sub(k,(kts))*(1)],odts);
// MPAS module_mp_thompson.F:2354
pni_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(nr[0+tc_sub(k,(kts))*(1)],odts);
// MPAS module_mp_thompson.F:2355
}
// MPAS module_mp_thompson.F:2357
if((rc[0+tc_sub(k,(kts))*(1)]>r_c[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2358
pri_wfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tpi_qcfz[0+tc_sub(idx_c,(1))*(1)+tc_sub(idx_n,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((100)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((100)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2359
pri_wfz[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rc[0+tc_sub(k,(kts))*(1)],odts))),pri_wfz[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2360
pni_wfz[0+tc_sub(k,(kts))*(1)] = tc_mul(tni_qcfz[0+tc_sub(idx_c,(1))*(1)+tc_sub(idx_n,(1))*tc_mul(1,((37)-(1)+1))+tc_sub(idx_tc,(1))*(1*((37)-(1)+1)*((100)-(1)+1))+tc_sub(idx_in,(1))*(1*((37)-(1)+1)*((100)-(1)+1)*((45)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2361
pni_wfz[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts))),tc_div(pri_wfz[0+tc_sub(k,(kts))*(1)],(tc_mul(2.f,xm0i))),pni_wfz[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2363
} else if(((rc[0+tc_sub(k,(kts))*(1)]>r1)&&(temp[0+tc_sub(k,(kts))*(1)]<hgfr))) {
// MPAS module_mp_thompson.F:2364
pri_wfz[0+tc_sub(k,(kts))*(1)] = tc_mul(rc[0+tc_sub(k,(kts))*(1)],odts);
// MPAS module_mp_thompson.F:2365
pni_wfz[0+tc_sub(k,(kts))*(1)] = tc_mul(nc[0+tc_sub(k,(kts))*(1)],odts);
// MPAS module_mp_thompson.F:2366
}
// MPAS module_mp_thompson.F:2370
if((((ssati[0+tc_sub(k,(kts))*(1)]>=0.25f))||(((ssatw[0+tc_sub(k,(kts))*(1)]>eps)&&(temp[0+tc_sub(k,(kts))*(1)]<253.15f))))) {
// MPAS module_mp_thompson.F:2372
if((dustyice&&is_aerosol_aware)) {
// MPAS module_mp_thompson.F:2373
xnc = icedemott(tempc,qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)],nifa[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2374
} else {
// MPAS module_mp_thompson.F:2375
xnc = tc_min(250.e3f,tc_mul(tno,tc_exp(tc_mul(ato,(tc_sub(t_0,temp[0+tc_sub(k,(kts))*(1)]))))));
// MPAS module_mp_thompson.F:2376
}
// MPAS module_mp_thompson.F:2377
xni = tc_add(ni[0+tc_sub(k,(kts))*(1)],tc_mul((tc_add(pni_rfz[0+tc_sub(k,(kts))*(1)],pni_wfz[0+tc_sub(k,(kts))*(1)])),dtsave));
// MPAS module_mp_thompson.F:2378
pni_inu[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(0.5f,(tc_add(tc_sub(xnc,xni),tc_abs(tc_sub(xnc,xni))))),odts);
// MPAS module_mp_thompson.F:2379
pri_inu[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(rate_max)),tc_mul(xm0i,pni_inu[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2380
pni_inu[0+tc_sub(k,(kts))*(1)] = tc_div(pri_inu[0+tc_sub(k,(kts))*(1)],xm0i);
// MPAS module_mp_thompson.F:2381
}
// MPAS module_mp_thompson.F:2384
xni = tc_add(tc_add(smo0[0+tc_sub(k,(kts))*(1)],ni[0+tc_sub(k,(kts))*(1)]),tc_mul((tc_add(tc_add(pni_rfz[0+tc_sub(k,(kts))*(1)],pni_wfz[0+tc_sub(k,(kts))*(1)]),pni_inu[0+tc_sub(k,(kts))*(1)])),dtsave));
// MPAS module_mp_thompson.F:2385
if(((((is_aerosol_aware&&homogice)&&((xni<=999.e3f)))&&((temp[0+tc_sub(k,(kts))*(1)]<238)))&&((ssati[0+tc_sub(k,(kts))*(1)]>=0.4f)))) {
// MPAS module_mp_thompson.F:2387
xnc = icekoop(temp[0+tc_sub(k,(kts))*(1)],qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)],nwfa[0+tc_sub(k,(kts))*(1)],dtsave);
// MPAS module_mp_thompson.F:2388
pni_iha[0+tc_sub(k,(kts))*(1)] = tc_mul(xnc,odts);
// MPAS module_mp_thompson.F:2389
pri_iha[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(rate_max)),tc_mul(tc_mul(xm0i,0.1f),pni_iha[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2390
pni_iha[0+tc_sub(k,(kts))*(1)] = tc_div(pri_iha[0+tc_sub(k,(kts))*(1)],(tc_mul(xm0i,0.1f)));
// MPAS module_mp_thompson.F:2391
}
// MPAS module_mp_thompson.F:2396
if(l_qi[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:2397
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni[0+tc_sub(k,(kts))*(1)]),ri[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:2398
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:2399
xdi = tc_max(((double)(d0i)),tc_mul((tc_add(tc_add(bm_i,mu_i),1.f)),ilami));
// MPAS module_mp_thompson.F:2400
xmi = tc_mul(am_i,tc_pow(xdi,bm_i));
// MPAS module_mp_thompson.F:2401
oxmi = tc_div(1.f,xmi);
// MPAS module_mp_thompson.F:2402
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(c_cube,t1_subl),diffu[0+tc_sub(k,(kts))*(1)]),ssati[0+tc_sub(k,(kts))*(1)]),rvs),oig1),cig[0+tc_sub(5,(1))*(1)]),ni[0+tc_sub(k,(kts))*(1)]),ilami);
// MPAS module_mp_thompson.F:2405
if((pri_ide[0+tc_sub(k,(kts))*(1)]<0.0f)) {
// MPAS module_mp_thompson.F:2406
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-ri[0+tc_sub(k,(kts))*(1)]),odts))),pri_ide[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2407
pni_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_ide[0+tc_sub(k,(kts))*(1)],oxmi);
// MPAS module_mp_thompson.F:2408
pni_ide[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-ni[0+tc_sub(k,(kts))*(1)]),odts))),pni_ide[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2409
} else {
// MPAS module_mp_thompson.F:2410
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_min(pri_ide[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2411
prs_ide[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(1.0e0,tpi_ide[0+tc_sub(idx_i,(1))*(1)+tc_sub(idx_i1,(1))*tc_mul(1,((64)-(1)+1))])),pri_ide[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2412
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(tpi_ide[0+tc_sub(idx_i,(1))*(1)+tc_sub(idx_i1,(1))*tc_mul(1,((64)-(1)+1))],pri_ide[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2413
}
// MPAS module_mp_thompson.F:2417
if((((idx_i==ntb_i))||((xdi>tc_mul(5.0f,d0s))))) {
// MPAS module_mp_thompson.F:2418
prs_iau[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(ri[0+tc_sub(k,(kts))*(1)],.99f),odts);
// MPAS module_mp_thompson.F:2419
pni_iau[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(ni[0+tc_sub(k,(kts))*(1)],.95f),odts);
// MPAS module_mp_thompson.F:2420
} else if((xdi<tc_mul(0.1f,d0s))) {
// MPAS module_mp_thompson.F:2421
prs_iau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2422
pni_iau[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2423
} else {
// MPAS module_mp_thompson.F:2424
prs_iau[0+tc_sub(k,(kts))*(1)] = tc_mul(tps_iaus[0+tc_sub(idx_i,(1))*(1)+tc_sub(idx_i1,(1))*tc_mul(1,((64)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2425
prs_iau[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(tc_mul(ri[0+tc_sub(k,(kts))*(1)],.99f),odts))),prs_iau[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2426
pni_iau[0+tc_sub(k,(kts))*(1)] = tc_mul(tni_iaus[0+tc_sub(idx_i,(1))*(1)+tc_sub(idx_i1,(1))*tc_mul(1,((64)-(1)+1))],odts);
// MPAS module_mp_thompson.F:2427
pni_iau[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(tc_mul(ni[0+tc_sub(k,(kts))*(1)],.95f),odts))),pni_iau[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2428
}
// MPAS module_mp_thompson.F:2429
}
// MPAS module_mp_thompson.F:2433
if(l_qs[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:2434
c_snow = tc_add(c_sqrd,tc_div(tc_mul((tc_add(tempc,1.5f)),(tc_sub(c_cube,c_sqrd))),(tc_add((-30.f),1.5f))));
// MPAS module_mp_thompson.F:2435
c_snow = tc_max(c_sqrd,tc_min(c_snow,c_cube));
// MPAS module_mp_thompson.F:2436
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(c_snow,t1_subl),diffu[0+tc_sub(k,(kts))*(1)]),ssati[0+tc_sub(k,(kts))*(1)]),rvs),(tc_add(tc_mul(t1_qs_sd,smo1[0+tc_sub(k,(kts))*(1)]),tc_mul(tc_mul(tc_mul(t2_qs_sd,rhof2[0+tc_sub(k,(kts))*(1)]),vsc2[0+tc_sub(k,(kts))*(1)]),smof[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:2439
if((prs_sde[0+tc_sub(k,(kts))*(1)]<0.f)) {
// MPAS module_mp_thompson.F:2440
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rs[0+tc_sub(k,(kts))*(1)]),odts))),prs_sde[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2441
} else {
// MPAS module_mp_thompson.F:2442
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_min(prs_sde[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2443
}
// MPAS module_mp_thompson.F:2444
}
// MPAS module_mp_thompson.F:2446
if((l_qg[0+tc_sub(k,(kts))*(1)]&&(ssati[0+tc_sub(k,(kts))*(1)]<(-eps)))) {
// MPAS module_mp_thompson.F:2447
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(c_cube,t1_subl),diffu[0+tc_sub(k,(kts))*(1)]),ssati[0+tc_sub(k,(kts))*(1)]),rvs),n0_g[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_mul(t1_qg_sd,tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(10,(1))*(1)])),tc_mul(tc_mul(tc_mul(t2_qg_sd,vsc2[0+tc_sub(k,(kts))*(1)]),rhof2[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(11,(1))*(1)])))));
// MPAS module_mp_thompson.F:2450
if((prg_gde[0+tc_sub(k,(kts))*(1)]<0.f)) {
// MPAS module_mp_thompson.F:2451
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rg[0+tc_sub(k,(kts))*(1)]),odts))),prg_gde[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2452
} else {
// MPAS module_mp_thompson.F:2453
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_min(prg_gde[0+tc_sub(k,(kts))*(1)],((double)(rate_max)));
// MPAS module_mp_thompson.F:2454
}
// MPAS module_mp_thompson.F:2455
}
// MPAS module_mp_thompson.F:2458
if(l_qi[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:2459
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni[0+tc_sub(k,(kts))*(1)]),ri[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:2460
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:2461
xdi = tc_max(((double)(d0i)),tc_mul((tc_add(tc_add(bm_i,mu_i),1.f)),ilami));
// MPAS module_mp_thompson.F:2462
xmi = tc_mul(am_i,tc_pow(xdi,bm_i));
// MPAS module_mp_thompson.F:2463
oxmi = tc_div(1.f,xmi);
// MPAS module_mp_thompson.F:2464
if((rs[0+tc_sub(k,(kts))*(1)]>=r_s[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:2465
prs_sci[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(t1_qs_qi,rhof[0+tc_sub(k,(kts))*(1)]),ef_si),ri[0+tc_sub(k,(kts))*(1)]),smoe[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2466
pni_sci[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_sci[0+tc_sub(k,(kts))*(1)],oxmi);
// MPAS module_mp_thompson.F:2467
}
// MPAS module_mp_thompson.F:2470
if(((rr[0+tc_sub(k,(kts))*(1)]>=r_r[0+tc_sub(1,(1))*(1)])&&(mvd_r[0+tc_sub(k,(kts))*(1)]>tc_mul(4.f,xdi)))) {
// MPAS module_mp_thompson.F:2471
lamr = tc_div(1.f,ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2472
pri_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qi),ef_ri),ri[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:2474
pnr_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t1_qr_qi),ef_ri),ni[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(9,(1))*(1)])))));
// MPAS module_mp_thompson.F:2476
pni_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_rci[0+tc_sub(k,(kts))*(1)],oxmi);
// MPAS module_mp_thompson.F:2477
prr_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],t2_qr_qi),ef_ri),ni[0+tc_sub(k,(kts))*(1)]),n0_r[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(8,(1))*(1)])))));
// MPAS module_mp_thompson.F:2479
prr_rci[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rr[0+tc_sub(k,(kts))*(1)],odts))),prr_rci[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2480
prg_rci[0+tc_sub(k,(kts))*(1)] = tc_add(pri_rci[0+tc_sub(k,(kts))*(1)],prr_rci[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2481
}
// MPAS module_mp_thompson.F:2482
}
// MPAS module_mp_thompson.F:2485
if(((prg_gcw[0+tc_sub(k,(kts))*(1)]>eps)&&(tempc>(-8.0f)))) {
// MPAS module_mp_thompson.F:2486
tf = 0.f;
// MPAS module_mp_thompson.F:2487
if(((tempc>=(-5.0f))&&(tempc<(-3.0f)))) {
// MPAS module_mp_thompson.F:2488
tf = tc_mul(0.5f,(tc_sub((-3.0f),tempc)));
// MPAS module_mp_thompson.F:2489
} else if(((tempc>(-8.0f))&&(tempc<(-5.0f)))) {
// MPAS module_mp_thompson.F:2490
tf = tc_mul(0.33333333f,(tc_add(8.0f,tempc)));
// MPAS module_mp_thompson.F:2491
}
// MPAS module_mp_thompson.F:2492
pni_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(3.5e8f,tf),prg_gcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2493
pri_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(xm0i,pni_ihm[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2494
prs_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(prs_scw[0+tc_sub(k,(kts))*(1)],(tc_add(prs_scw[0+tc_sub(k,(kts))*(1)],prg_gcw[0+tc_sub(k,(kts))*(1)]))),pri_ihm[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2496
prg_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(prg_gcw[0+tc_sub(k,(kts))*(1)],(tc_add(prs_scw[0+tc_sub(k,(kts))*(1)],prg_gcw[0+tc_sub(k,(kts))*(1)]))),pri_ihm[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2498
}
// MPAS module_mp_thompson.F:2504
if(((prs_scw[0+tc_sub(k,(kts))*(1)]>tc_mul(2.0f,prs_sde[0+tc_sub(k,(kts))*(1)]))&&(prs_sde[0+tc_sub(k,(kts))*(1)]>eps))) {
// MPAS module_mp_thompson.F:2506
r_frac = tc_min(30.0e0,tc_div(prs_scw[0+tc_sub(k,(kts))*(1)],prs_sde[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2507
g_frac = tc_min(0.95f,tc_add(0.15f,tc_mul((tc_sub(r_frac,2.f)),.028f)));
// MPAS module_mp_thompson.F:2508
vts_boost[0+tc_sub(k,(kts))*(1)] = tc_min(1.5f,tc_add(1.1f,tc_mul((tc_sub(r_frac,2.f)),.014f)));
// MPAS module_mp_thompson.F:2509
prg_scw[0+tc_sub(k,(kts))*(1)] = tc_mul(g_frac,prs_scw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2510
prs_scw[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(1.f,g_frac)),prs_scw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2511
}
// MPAS module_mp_thompson.F:2513
} else {
// MPAS module_mp_thompson.F:2517
if(l_qs[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:2518
prr_sml[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(tc_mul(tempc,tcond[0+tc_sub(k,(kts))*(1)]),tc_mul(tc_mul(lvap0,diffu[0+tc_sub(k,(kts))*(1)]),delqvs[0+tc_sub(k,(kts))*(1)]))),(tc_add(tc_mul(t1_qs_me,smo1[0+tc_sub(k,(kts))*(1)]),tc_mul(tc_mul(tc_mul(t2_qs_me,rhof2[0+tc_sub(k,(kts))*(1)]),vsc2[0+tc_sub(k,(kts))*(1)]),smof[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:2520
if((prr_sml[0+tc_sub(k,(kts))*(1)]>0.f)) {
// MPAS module_mp_thompson.F:2521
prr_sml[0+tc_sub(k,(kts))*(1)] = tc_add(prr_sml[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(4218.f,olfus),tempc),(tc_add(prr_rcs[0+tc_sub(k,(kts))*(1)],prs_scw[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:2523
}
// MPAS module_mp_thompson.F:2524
prr_sml[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rs[0+tc_sub(k,(kts))*(1)],odts))),tc_max(0.e0,prr_sml[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2525
pnr_sml[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_div(smo0[0+tc_sub(k,(kts))*(1)],rs[0+tc_sub(k,(kts))*(1)]),prr_sml[0+tc_sub(k,(kts))*(1)]),tc_pow(10.0f,(tc_mul((-0.25f),tempc))));
// MPAS module_mp_thompson.F:2526
pnr_sml[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(smo0[0+tc_sub(k,(kts))*(1)],odts))),pnr_sml[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2528
if((ssati[0+tc_sub(k,(kts))*(1)]<0.f)) {
// MPAS module_mp_thompson.F:2529
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(c_cube,t1_subl),diffu[0+tc_sub(k,(kts))*(1)]),ssati[0+tc_sub(k,(kts))*(1)]),rvs),(tc_add(tc_mul(t1_qs_sd,smo1[0+tc_sub(k,(kts))*(1)]),tc_mul(tc_mul(tc_mul(t2_qs_sd,rhof2[0+tc_sub(k,(kts))*(1)]),vsc2[0+tc_sub(k,(kts))*(1)]),smof[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:2532
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rs[0+tc_sub(k,(kts))*(1)]),odts))),prs_sde[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2533
}
// MPAS module_mp_thompson.F:2534
}
// MPAS module_mp_thompson.F:2536
if(l_qg[0+tc_sub(k,(kts))*(1)]) {
// MPAS module_mp_thompson.F:2537
prr_gml[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(tc_mul(tempc,tcond[0+tc_sub(k,(kts))*(1)]),tc_mul(tc_mul(lvap0,diffu[0+tc_sub(k,(kts))*(1)]),delqvs[0+tc_sub(k,(kts))*(1)]))),n0_g[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_mul(t1_qg_me,tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(10,(1))*(1)])),tc_mul(tc_mul(tc_mul(t2_qg_me,rhof2[0+tc_sub(k,(kts))*(1)]),vsc2[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(11,(1))*(1)])))));
// MPAS module_mp_thompson.F:2542
prr_gml[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(rg[0+tc_sub(k,(kts))*(1)],odts))),tc_max(0.e0,prr_gml[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2543
pnr_gml[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_div(tc_mul(tc_mul(n0_g[0+tc_sub(k,(kts))*(1)],cgg[0+tc_sub(2,(1))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(2,(1))*(1)])),rg[0+tc_sub(k,(kts))*(1)]),prr_gml[0+tc_sub(k,(kts))*(1)]),tc_pow(10.0f,(tc_mul((-0.5f),tempc))));
// MPAS module_mp_thompson.F:2546
if((ssati[0+tc_sub(k,(kts))*(1)]<0.f)) {
// MPAS module_mp_thompson.F:2547
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(c_cube,t1_subl),diffu[0+tc_sub(k,(kts))*(1)]),ssati[0+tc_sub(k,(kts))*(1)]),rvs),n0_g[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_mul(t1_qg_sd,tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(10,(1))*(1)])),tc_mul(tc_mul(tc_mul(t2_qg_sd,vsc2[0+tc_sub(k,(kts))*(1)]),rhof2[0+tc_sub(k,(kts))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(11,(1))*(1)])))));
// MPAS module_mp_thompson.F:2550
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul((-rg[0+tc_sub(k,(kts))*(1)]),odts))),prg_gde[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2551
}
// MPAS module_mp_thompson.F:2552
}
// MPAS module_mp_thompson.F:2558
if((dt>120.f)) {
// MPAS module_mp_thompson.F:2559
prr_rcw[0+tc_sub(k,(kts))*(1)] = tc_add(tc_add(prr_rcw[0+tc_sub(k,(kts))*(1)],prs_scw[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2560
prs_scw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2561
prg_gcw[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2562
}
// MPAS module_mp_thompson.F:2564
}
// MPAS module_mp_thompson.F:2566
}
// MPAS module_mp_thompson.F:2567
}
// MPAS module_mp_thompson.F:2572
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2578
sump = tc_add(tc_add(tc_add(tc_add(tc_add(pri_inu[0+tc_sub(k,(kts))*(1)],pri_ide[0+tc_sub(k,(kts))*(1)]),prs_ide[0+tc_sub(k,(kts))*(1)]),prs_sde[0+tc_sub(k,(kts))*(1)]),prg_gde[0+tc_sub(k,(kts))*(1)]),pri_iha[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2580
rate_max = tc_mul(tc_mul(tc_mul((tc_sub(qv[0+tc_sub(k,(kts))*(1)],qvsi[0+tc_sub(k,(kts))*(1)])),rho[0+tc_sub(k,(kts))*(1)]),odts),0.999f);
// MPAS module_mp_thompson.F:2581
if(((((sump>eps)&&(sump>rate_max)))||(((sump<(-eps))&&(sump<rate_max))))) {
// MPAS module_mp_thompson.F:2583
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2584
pri_inu[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_inu[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2585
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_ide[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2586
pni_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(pni_ide[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2587
prs_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_ide[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2588
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_sde[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2589
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_gde[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2590
pri_iha[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_iha[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2591
}
// MPAS module_mp_thompson.F:2594
sump = tc_sub(tc_sub(tc_sub(tc_sub(tc_sub((-prr_wau[0+tc_sub(k,(kts))*(1)]),pri_wfz[0+tc_sub(k,(kts))*(1)]),prr_rcw[0+tc_sub(k,(kts))*(1)]),prs_scw[0+tc_sub(k,(kts))*(1)]),prg_scw[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2596
rate_max = tc_mul((-rc[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2597
if(((sump<rate_max)&&l_qc[0+tc_sub(k,(kts))*(1)])) {
// MPAS module_mp_thompson.F:2598
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2599
prr_wau[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_wau[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2600
pri_wfz[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_wfz[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2601
prr_rcw[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_rcw[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2602
prs_scw[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_scw[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2603
prg_scw[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_scw[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2604
prg_gcw[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_gcw[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2605
}
// MPAS module_mp_thompson.F:2608
sump = tc_sub(tc_sub(tc_sub(pri_ide[0+tc_sub(k,(kts))*(1)],prs_iau[0+tc_sub(k,(kts))*(1)]),prs_sci[0+tc_sub(k,(kts))*(1)]),pri_rci[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2610
rate_max = tc_mul((-ri[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2611
if(((sump<rate_max)&&l_qi[0+tc_sub(k,(kts))*(1)])) {
// MPAS module_mp_thompson.F:2612
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2613
pri_ide[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_ide[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2614
prs_iau[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_iau[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2615
prs_sci[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_sci[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2616
pri_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_rci[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2617
}
// MPAS module_mp_thompson.F:2620
sump = tc_add(tc_add(tc_sub(tc_sub((-prg_rfz[0+tc_sub(k,(kts))*(1)]),pri_rfz[0+tc_sub(k,(kts))*(1)]),prr_rci[0+tc_sub(k,(kts))*(1)]),prr_rcs[0+tc_sub(k,(kts))*(1)]),prr_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2622
rate_max = tc_mul((-rr[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2623
if(((sump<rate_max)&&l_qr[0+tc_sub(k,(kts))*(1)])) {
// MPAS module_mp_thompson.F:2624
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2625
prg_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_rfz[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2626
pri_rfz[0+tc_sub(k,(kts))*(1)] = tc_mul(pri_rfz[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2627
prr_rci[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_rci[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2628
prr_rcs[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_rcs[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2629
prr_rcg[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_rcg[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2630
}
// MPAS module_mp_thompson.F:2633
sump = tc_add(tc_sub(tc_sub(prs_sde[0+tc_sub(k,(kts))*(1)],prs_ihm[0+tc_sub(k,(kts))*(1)]),prr_sml[0+tc_sub(k,(kts))*(1)]),prs_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2635
rate_max = tc_mul((-rs[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2636
if(((sump<rate_max)&&l_qs[0+tc_sub(k,(kts))*(1)])) {
// MPAS module_mp_thompson.F:2637
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2638
prs_sde[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_sde[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2639
prs_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_ihm[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2640
prr_sml[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_sml[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2641
prs_rcs[0+tc_sub(k,(kts))*(1)] = tc_mul(prs_rcs[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2642
}
// MPAS module_mp_thompson.F:2645
sump = tc_add(tc_sub(tc_sub(prg_gde[0+tc_sub(k,(kts))*(1)],prg_ihm[0+tc_sub(k,(kts))*(1)]),prr_gml[0+tc_sub(k,(kts))*(1)]),prg_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2647
rate_max = tc_mul((-rg[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2648
if(((sump<rate_max)&&l_qg[0+tc_sub(k,(kts))*(1)])) {
// MPAS module_mp_thompson.F:2649
ratio = tc_div(rate_max,sump);
// MPAS module_mp_thompson.F:2650
prg_gde[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_gde[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2651
prg_ihm[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_ihm[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2652
prr_gml[0+tc_sub(k,(kts))*(1)] = tc_mul(prr_gml[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2653
prg_rcg[0+tc_sub(k,(kts))*(1)] = tc_mul(prg_rcg[0+tc_sub(k,(kts))*(1)],ratio);
// MPAS module_mp_thompson.F:2654
}
// MPAS module_mp_thompson.F:2658
pri_ihm[0+tc_sub(k,(kts))*(1)] = tc_add(prs_ihm[0+tc_sub(k,(kts))*(1)],prg_ihm[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2659
ratio = tc_min(tc_abs(prr_rcg[0+tc_sub(k,(kts))*(1)]),tc_abs(prg_rcg[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2660
prr_rcg[0+tc_sub(k,(kts))*(1)] = tc_mul(ratio,tc_sign(1.0f,tc_sngl(prr_rcg[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2661
prg_rcg[0+tc_sub(k,(kts))*(1)] = (-prr_rcg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2662
if((temp[0+tc_sub(k,(kts))*(1)]>t_0)) {
// MPAS module_mp_thompson.F:2663
ratio = tc_min(tc_abs(prr_rcs[0+tc_sub(k,(kts))*(1)]),tc_abs(prs_rcs[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2664
prr_rcs[0+tc_sub(k,(kts))*(1)] = tc_mul(ratio,tc_sign(1.0f,tc_sngl(prr_rcs[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2665
prs_rcs[0+tc_sub(k,(kts))*(1)] = (-prr_rcs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2666
}
// MPAS module_mp_thompson.F:2668
}
// MPAS module_mp_thompson.F:2674
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2675
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2676
lfus2 = tc_sub(lsub,lvap[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2679
if(is_aerosol_aware) {
// MPAS module_mp_thompson.F:2680
nwfaten[0+tc_sub(k,(kts))*(1)] = tc_sub(nwfaten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_add(tc_add(tc_add(pna_rca[0+tc_sub(k,(kts))*(1)],pna_sca[0+tc_sub(k,(kts))*(1)]),pna_gca[0+tc_sub(k,(kts))*(1)]),pni_iha[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2682
nifaten[0+tc_sub(k,(kts))*(1)] = tc_sub(nifaten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_add(tc_add(pnd_rcd[0+tc_sub(k,(kts))*(1)],pnd_scd[0+tc_sub(k,(kts))*(1)]),pnd_gcd[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2684
if(dustyice) {
// MPAS module_mp_thompson.F:2685
nifaten[0+tc_sub(k,(kts))*(1)] = tc_sub(nifaten[0+tc_sub(k,(kts))*(1)],tc_mul(pni_inu[0+tc_sub(k,(kts))*(1)],orho));
// MPAS module_mp_thompson.F:2686
} else {
// MPAS module_mp_thompson.F:2687
nifaten[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2688
}
// MPAS module_mp_thompson.F:2689
}
// MPAS module_mp_thompson.F:2692
qvten[0+tc_sub(k,(kts))*(1)] = tc_add(qvten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_sub(tc_sub((-pri_inu[0+tc_sub(k,(kts))*(1)]),pri_iha[0+tc_sub(k,(kts))*(1)]),pri_ide[0+tc_sub(k,(kts))*(1)]),prs_ide[0+tc_sub(k,(kts))*(1)]),prs_sde[0+tc_sub(k,(kts))*(1)]),prg_gde[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2697
qcten[0+tc_sub(k,(kts))*(1)] = tc_add(qcten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_sub(tc_sub((-prr_wau[0+tc_sub(k,(kts))*(1)]),pri_wfz[0+tc_sub(k,(kts))*(1)]),prr_rcw[0+tc_sub(k,(kts))*(1)]),prs_scw[0+tc_sub(k,(kts))*(1)]),prg_scw[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2703
ncten[0+tc_sub(k,(kts))*(1)] = tc_add(ncten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_sub((-pnc_wau[0+tc_sub(k,(kts))*(1)]),pnc_rcw[0+tc_sub(k,(kts))*(1)]),pni_wfz[0+tc_sub(k,(kts))*(1)]),pnc_scw[0+tc_sub(k,(kts))*(1)]),pnc_gcw[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2709
xrc = tc_max(r1,tc_mul((tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(qcten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2710
xnc = tc_max(2.f,tc_mul((tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2711
if((xrc>r1)) {
// MPAS module_mp_thompson.F:2712
nu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,xnc))),2));
// MPAS module_mp_thompson.F:2713
lamc = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(xnc,am_r),ccg[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:2714
xdc = tc_div((tc_add(tc_add(bm_r,nu_c),1.f)),lamc);
// MPAS module_mp_thompson.F:2715
if((xdc<d0c)) {
// MPAS module_mp_thompson.F:2716
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],d0c);
// MPAS module_mp_thompson.F:2717
xnc = tc_mul(tc_div(tc_mul(tc_mul(ccg[0+tc_sub(1,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],ocg2[0+tc_sub(nu_c,(1))*(1)]),xrc),am_r),tc_pow(lamc,bm_r));
// MPAS module_mp_thompson.F:2718
ncten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xnc,tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2719
} else if((xdc>tc_mul(d0r,2.f))) {
// MPAS module_mp_thompson.F:2720
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],(tc_mul(d0r,2.f)));
// MPAS module_mp_thompson.F:2721
xnc = tc_mul(tc_div(tc_mul(tc_mul(ccg[0+tc_sub(1,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],ocg2[0+tc_sub(nu_c,(1))*(1)]),xrc),am_r),tc_pow(lamc,bm_r));
// MPAS module_mp_thompson.F:2722
ncten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xnc,tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2723
}
// MPAS module_mp_thompson.F:2724
} else {
// MPAS module_mp_thompson.F:2725
ncten[0+tc_sub(k,(kts))*(1)] = tc_mul((-nc1d[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2726
}
// MPAS module_mp_thompson.F:2727
xnc = tc_max(0.f,tc_mul((tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2728
if((xnc>nt_c_max)) { ncten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(nt_c_max,tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho); }
// MPAS module_mp_thompson.F:2732
qiten[0+tc_sub(k,(kts))*(1)] = tc_add(qiten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(tc_add(pri_inu[0+tc_sub(k,(kts))*(1)],pri_iha[0+tc_sub(k,(kts))*(1)]),pri_ihm[0+tc_sub(k,(kts))*(1)]),pri_wfz[0+tc_sub(k,(kts))*(1)]),pri_rfz[0+tc_sub(k,(kts))*(1)]),pri_ide[0+tc_sub(k,(kts))*(1)]),prs_iau[0+tc_sub(k,(kts))*(1)]),prs_sci[0+tc_sub(k,(kts))*(1)]),pri_rci[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2738
niten[0+tc_sub(k,(kts))*(1)] = tc_add(niten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(tc_add(pni_inu[0+tc_sub(k,(kts))*(1)],pni_iha[0+tc_sub(k,(kts))*(1)]),pni_ihm[0+tc_sub(k,(kts))*(1)]),pni_wfz[0+tc_sub(k,(kts))*(1)]),pni_rfz[0+tc_sub(k,(kts))*(1)]),pni_ide[0+tc_sub(k,(kts))*(1)]),pni_iau[0+tc_sub(k,(kts))*(1)]),pni_sci[0+tc_sub(k,(kts))*(1)]),pni_rci[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2745
xri = tc_max(r1,tc_mul((tc_add(qi1d[0+tc_sub(k,(kts))*(1)],tc_mul(qiten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2746
xni = tc_max(r2,tc_mul((tc_add(ni1d[0+tc_sub(k,(kts))*(1)],tc_mul(niten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2747
if((xri>r1)) {
// MPAS module_mp_thompson.F:2748
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),xni),xri)),obmi);
// MPAS module_mp_thompson.F:2749
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:2750
xdi = tc_mul((tc_add(tc_add(bm_i,mu_i),1.f)),ilami);
// MPAS module_mp_thompson.F:2751
if((xdi<5.e-6f)) {
// MPAS module_mp_thompson.F:2752
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],5.e-6f);
// MPAS module_mp_thompson.F:2753
xni = tc_min(9999.e3,tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),xri),am_i),tc_pow(lami,bm_i)));
// MPAS module_mp_thompson.F:2754
niten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xni,tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2755
} else if((xdi>300.e-6f)) {
// MPAS module_mp_thompson.F:2756
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],300.e-6f);
// MPAS module_mp_thompson.F:2757
xni = tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),xri),am_i),tc_pow(lami,bm_i));
// MPAS module_mp_thompson.F:2758
niten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xni,tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2759
}
// MPAS module_mp_thompson.F:2760
} else {
// MPAS module_mp_thompson.F:2761
niten[0+tc_sub(k,(kts))*(1)] = tc_mul((-ni1d[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2762
}
// MPAS module_mp_thompson.F:2763
xni = tc_max(0.f,tc_mul((tc_add(ni1d[0+tc_sub(k,(kts))*(1)],tc_mul(niten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2764
if((xni>9999.e3f)) { niten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(9999.e3f,tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho); }
// MPAS module_mp_thompson.F:2768
qrten[0+tc_sub(k,(kts))*(1)] = tc_add(qrten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(tc_add(prr_wau[0+tc_sub(k,(kts))*(1)],prr_rcw[0+tc_sub(k,(kts))*(1)]),prr_sml[0+tc_sub(k,(kts))*(1)]),prr_gml[0+tc_sub(k,(kts))*(1)]),prr_rcs[0+tc_sub(k,(kts))*(1)]),prr_rcg[0+tc_sub(k,(kts))*(1)]),prg_rfz[0+tc_sub(k,(kts))*(1)]),pri_rfz[0+tc_sub(k,(kts))*(1)]),prr_rci[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2775
nrten[0+tc_sub(k,(kts))*(1)] = tc_add(nrten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_add(tc_add(pnr_wau[0+tc_sub(k,(kts))*(1)],pnr_sml[0+tc_sub(k,(kts))*(1)]),pnr_gml[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_add(tc_add(tc_add(tc_add(pnr_rfz[0+tc_sub(k,(kts))*(1)],pnr_rcr[0+tc_sub(k,(kts))*(1)]),pnr_rcg[0+tc_sub(k,(kts))*(1)]),pnr_rcs[0+tc_sub(k,(kts))*(1)]),pnr_rci[0+tc_sub(k,(kts))*(1)]),pni_rfz[0+tc_sub(k,(kts))*(1)])))),orho));
// MPAS module_mp_thompson.F:2782
xrr = tc_max(r1,tc_mul((tc_add(qr1d[0+tc_sub(k,(kts))*(1)],tc_mul(qrten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2783
xnr = tc_max(r2,tc_mul((tc_add(nr1d[0+tc_sub(k,(kts))*(1)],tc_mul(nrten[0+tc_sub(k,(kts))*(1)],dtsave))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2784
if((xrr>r1)) {
// MPAS module_mp_thompson.F:2785
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),xnr),xrr)),obmr);
// MPAS module_mp_thompson.F:2786
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:2787
if((mvd_r[0+tc_sub(k,(kts))*(1)]>2.5e-3f)) {
// MPAS module_mp_thompson.F:2788
mvd_r[0+tc_sub(k,(kts))*(1)] = 2.5e-3f;
// MPAS module_mp_thompson.F:2789
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2790
xnr = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),xrr),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:2791
nrten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xnr,tc_mul(nr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2792
} else if((mvd_r[0+tc_sub(k,(kts))*(1)]<tc_mul(d0r,0.75f))) {
// MPAS module_mp_thompson.F:2793
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_mul(d0r,0.75f);
// MPAS module_mp_thompson.F:2794
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2795
xnr = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),xrr),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:2796
nrten[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_sub(xnr,tc_mul(nr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))),odts),orho);
// MPAS module_mp_thompson.F:2797
}
// MPAS module_mp_thompson.F:2798
} else {
// MPAS module_mp_thompson.F:2799
qrten[0+tc_sub(k,(kts))*(1)] = tc_mul((-qr1d[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2800
nrten[0+tc_sub(k,(kts))*(1)] = tc_mul((-nr1d[0+tc_sub(k,(kts))*(1)]),odts);
// MPAS module_mp_thompson.F:2801
}
// MPAS module_mp_thompson.F:2804
qsten[0+tc_sub(k,(kts))*(1)] = tc_add(qsten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(tc_add(prs_iau[0+tc_sub(k,(kts))*(1)],prs_sde[0+tc_sub(k,(kts))*(1)]),prs_sci[0+tc_sub(k,(kts))*(1)]),prs_scw[0+tc_sub(k,(kts))*(1)]),prs_rcs[0+tc_sub(k,(kts))*(1)]),prs_ide[0+tc_sub(k,(kts))*(1)]),prs_ihm[0+tc_sub(k,(kts))*(1)]),prr_sml[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2810
qgten[0+tc_sub(k,(kts))*(1)] = tc_add(qgten[0+tc_sub(k,(kts))*(1)],tc_mul((tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(prg_scw[0+tc_sub(k,(kts))*(1)],prg_rfz[0+tc_sub(k,(kts))*(1)]),prg_gde[0+tc_sub(k,(kts))*(1)]),prg_rcg[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)]),prg_rci[0+tc_sub(k,(kts))*(1)]),prg_rcs[0+tc_sub(k,(kts))*(1)]),prg_ihm[0+tc_sub(k,(kts))*(1)]),prr_gml[0+tc_sub(k,(kts))*(1)])),orho));
// MPAS module_mp_thompson.F:2817
if((temp[0+tc_sub(k,(kts))*(1)]<t_0)) {
// MPAS module_mp_thompson.F:2818
tten[0+tc_sub(k,(kts))*(1)] = tc_add(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_add(tc_mul(tc_mul(lsub,ocp[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_add(tc_add(tc_add(tc_add(pri_inu[0+tc_sub(k,(kts))*(1)],pri_ide[0+tc_sub(k,(kts))*(1)]),prs_ide[0+tc_sub(k,(kts))*(1)]),prs_sde[0+tc_sub(k,(kts))*(1)]),prg_gde[0+tc_sub(k,(kts))*(1)]),pri_iha[0+tc_sub(k,(kts))*(1)]))),tc_mul(tc_mul(lfus2,ocp[0+tc_sub(k,(kts))*(1)]),(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(pri_wfz[0+tc_sub(k,(kts))*(1)],pri_rfz[0+tc_sub(k,(kts))*(1)]),prg_rfz[0+tc_sub(k,(kts))*(1)]),prs_scw[0+tc_sub(k,(kts))*(1)]),prg_scw[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)]),prg_rcs[0+tc_sub(k,(kts))*(1)]),prs_rcs[0+tc_sub(k,(kts))*(1)]),prr_rci[0+tc_sub(k,(kts))*(1)]),prg_rcg[0+tc_sub(k,(kts))*(1)]))))),orho),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:2828
} else {
// MPAS module_mp_thompson.F:2829
tten[0+tc_sub(k,(kts))*(1)] = tc_add(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_add(tc_mul(tc_mul(lfus,ocp[0+tc_sub(k,(kts))*(1)]),(tc_sub(tc_sub(tc_sub((-prr_sml[0+tc_sub(k,(kts))*(1)]),prr_gml[0+tc_sub(k,(kts))*(1)]),prr_rcg[0+tc_sub(k,(kts))*(1)]),prr_rcs[0+tc_sub(k,(kts))*(1)]))),tc_mul(tc_mul(lsub,ocp[0+tc_sub(k,(kts))*(1)]),(tc_add(prs_sde[0+tc_sub(k,(kts))*(1)],prg_gde[0+tc_sub(k,(kts))*(1)]))))),orho),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:2834
}
// MPAS module_mp_thompson.F:2836
}
// MPAS module_mp_thompson.F:2841
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2842
temp[0+tc_sub(k,(kts))*(1)] = tc_add(t1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,tten[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2843
otemp = tc_div(1.f,temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2844
tempc = tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f);
// MPAS module_mp_thompson.F:2845
qv[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,tc_add(qv1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,qvten[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2846
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,pres[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,temp[0+tc_sub(k,(kts))*(1)]),(tc_add(qv[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:2847
rhof[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho_not,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2848
rhof2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(rhof[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2849
qvs[0+tc_sub(k,(kts))*(1)] = rslf(pres[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2850
ssatw[0+tc_sub(k,(kts))*(1)] = tc_sub(tc_div(qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]),1.f);
// MPAS module_mp_thompson.F:2851
if((tc_abs(ssatw[0+tc_sub(k,(kts))*(1)])<eps)) { ssatw[0+tc_sub(k,(kts))*(1)] = 0.0f; }
// MPAS module_mp_thompson.F:2852
diffu[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(2.11e-5f,tc_pow((tc_div(temp[0+tc_sub(k,(kts))*(1)],273.15f)),1.94f)),(tc_div(101325.f,pres[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:2853
if((tempc>=0.0f)) {
// MPAS module_mp_thompson.F:2854
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(1.718f,tc_mul(0.0049f,tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:2855
} else {
// MPAS module_mp_thompson.F:2856
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(tc_add(1.718f,tc_mul(0.0049f,tempc)),tc_mul(tc_mul(1.2e-5f,tempc),tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:2857
}
// MPAS module_mp_thompson.F:2858
vsc2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho[0+tc_sub(k,(kts))*(1)],visco[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2859
lvap[0+tc_sub(k,(kts))*(1)] = tc_add(lvap0,tc_mul((tc_sub(2106.0f,4218.0f)),tempc));
// MPAS module_mp_thompson.F:2860
tcond[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_add(5.69f,tc_mul(0.0168f,tempc))),1.0e-5f),418.936f);
// MPAS module_mp_thompson.F:2861
ocp[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,(tc_mul(cp,(tc_add(1.f,tc_mul(0.887f,qv[0+tc_sub(k,(kts))*(1)]))))));
// MPAS module_mp_thompson.F:2862
lvt2[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],lvap[0+tc_sub(k,(kts))*(1)]),ocp[0+tc_sub(k,(kts))*(1)]),orv),otemp),otemp);
// MPAS module_mp_thompson.F:2864
nwfa[0+tc_sub(k,(kts))*(1)] = tc_max(11.1e6f,tc_mul((tc_add(nwfa1d[0+tc_sub(k,(kts))*(1)],tc_mul(nwfaten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2865
}
// MPAS module_mp_thompson.F:2867
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2868
if(((tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(qcten[0+tc_sub(k,(kts))*(1)],dt)))>r1)) {
// MPAS module_mp_thompson.F:2869
rc[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(qcten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2870
nc[0+tc_sub(k,(kts))*(1)] = tc_max(2.f,tc_min(tc_mul((tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]),nt_c_max));
// MPAS module_mp_thompson.F:2871
if((!is_aerosol_aware)) { nc[0+tc_sub(k,(kts))*(1)] = nt_c; }
// MPAS module_mp_thompson.F:2872
l_qc[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:2873
} else {
// MPAS module_mp_thompson.F:2874
rc[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:2875
nc[0+tc_sub(k,(kts))*(1)] = 2.f;
// MPAS module_mp_thompson.F:2876
l_qc[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:2877
}
// MPAS module_mp_thompson.F:2879
if(((tc_add(qi1d[0+tc_sub(k,(kts))*(1)],tc_mul(qiten[0+tc_sub(k,(kts))*(1)],dt)))>r1)) {
// MPAS module_mp_thompson.F:2880
ri[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(qi1d[0+tc_sub(k,(kts))*(1)],tc_mul(qiten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2881
ni[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul((tc_add(ni1d[0+tc_sub(k,(kts))*(1)],tc_mul(niten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2882
l_qi[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:2883
} else {
// MPAS module_mp_thompson.F:2884
ri[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:2885
ni[0+tc_sub(k,(kts))*(1)] = r2;
// MPAS module_mp_thompson.F:2886
l_qi[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:2887
}
// MPAS module_mp_thompson.F:2889
if(((tc_add(qr1d[0+tc_sub(k,(kts))*(1)],tc_mul(qrten[0+tc_sub(k,(kts))*(1)],dt)))>r1)) {
// MPAS module_mp_thompson.F:2890
rr[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(qr1d[0+tc_sub(k,(kts))*(1)],tc_mul(qrten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2891
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul((tc_add(nr1d[0+tc_sub(k,(kts))*(1)],tc_mul(nrten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:2892
l_qr[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:2893
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:2894
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:2895
if((mvd_r[0+tc_sub(k,(kts))*(1)]>2.5e-3f)) {
// MPAS module_mp_thompson.F:2896
mvd_r[0+tc_sub(k,(kts))*(1)] = 2.5e-3f;
// MPAS module_mp_thompson.F:2897
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2898
nr[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),rr[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:2899
} else if((mvd_r[0+tc_sub(k,(kts))*(1)]<tc_mul(d0r,0.75f))) {
// MPAS module_mp_thompson.F:2900
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_mul(d0r,0.75f);
// MPAS module_mp_thompson.F:2901
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2902
nr[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),rr[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:2903
}
// MPAS module_mp_thompson.F:2904
} else {
// MPAS module_mp_thompson.F:2905
rr[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:2906
nr[0+tc_sub(k,(kts))*(1)] = r2;
// MPAS module_mp_thompson.F:2907
l_qr[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:2908
}
// MPAS module_mp_thompson.F:2910
if(((tc_add(qs1d[0+tc_sub(k,(kts))*(1)],tc_mul(qsten[0+tc_sub(k,(kts))*(1)],dt)))>r1)) {
// MPAS module_mp_thompson.F:2911
rs[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(qs1d[0+tc_sub(k,(kts))*(1)],tc_mul(qsten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2912
l_qs[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:2913
} else {
// MPAS module_mp_thompson.F:2914
rs[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:2915
l_qs[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:2916
}
// MPAS module_mp_thompson.F:2918
if(((tc_add(qg1d[0+tc_sub(k,(kts))*(1)],tc_mul(qgten[0+tc_sub(k,(kts))*(1)],dt)))>r1)) {
// MPAS module_mp_thompson.F:2919
rg[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(qg1d[0+tc_sub(k,(kts))*(1)],tc_mul(qgten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:2920
l_qg[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:2921
} else {
// MPAS module_mp_thompson.F:2922
rg[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:2923
l_qg[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:2924
}
// MPAS module_mp_thompson.F:2925
}
// MPAS module_mp_thompson.F:2931
if((!iiwarm)) {
// MPAS module_mp_thompson.F:2932
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2933
smo2[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2934
smob[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2935
smoc[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2936
smod[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:2937
}
// MPAS module_mp_thompson.F:2938
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:2939
if((!l_qs[0+tc_sub(k,(kts))*(1)])) { continue; }
// MPAS module_mp_thompson.F:2940
tc0 = tc_min((-0.1f),tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f));
// MPAS module_mp_thompson.F:2941
smob[0+tc_sub(k,(kts))*(1)] = tc_mul(rs[0+tc_sub(k,(kts))*(1)],oams);
// MPAS module_mp_thompson.F:2945
if(((bm_s>(tc_sub(2.0f,1.e-3f)))&&(bm_s<(tc_add(2.0f,1.e-3f))))) {
// MPAS module_mp_thompson.F:2946
smo2[0+tc_sub(k,(kts))*(1)] = smob[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:2947
} else {
// MPAS module_mp_thompson.F:2948
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:2953
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:2954
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:2959
smo2[0+tc_sub(k,(kts))*(1)] = tc_pow((tc_div(smob[0+tc_sub(k,(kts))*(1)],a_)),(tc_div(1.f,b_)));
// MPAS module_mp_thompson.F:2960
}
// MPAS module_mp_thompson.F:2963
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:2968
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:2969
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:2973
smoc[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:2976
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)]));
// MPAS module_mp_thompson.F:2981
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:2982
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)]),cse[0+tc_sub(14,(1))*(1)]));
// MPAS module_mp_thompson.F:2986
smod[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:2987
}
// MPAS module_mp_thompson.F:2992
n0_min = gonv_max;
// MPAS module_mp_thompson.F:2993
k_0 = kts;
// MPAS module_mp_thompson.F:2994
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:2995
if((temp[0+tc_sub(k,(kts))*(1)]>=270.65f)) { k_0 = tc_max(k_0,k); }
// MPAS module_mp_thompson.F:2996
}
// MPAS module_mp_thompson.F:2997
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:2998
if((((k>k_0)&&l_qr[0+tc_sub(k,(kts))*(1)])&&(mvd_r[0+tc_sub(k,(kts))*(1)]>100.e-6f))) {
// MPAS module_mp_thompson.F:2999
xslw1 = tc_add(4.01f,tc_log10(mvd_r[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3000
} else {
// MPAS module_mp_thompson.F:3001
xslw1 = 0.01f;
// MPAS module_mp_thompson.F:3002
}
// MPAS module_mp_thompson.F:3003
ygra1 = tc_add(4.31f,tc_log10(tc_max(5.e-5f,rg[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3004
zans1 = tc_add(3.1f,(tc_div(100.f,(tc_add(tc_add(tc_div(tc_mul(tc_mul(300.f,xslw1),ygra1),(tc_add(tc_add(tc_div(10.f,xslw1),1.f),tc_mul(0.25f,ygra1)))),30.f),tc_mul(10.f,ygra1))))));
// MPAS module_mp_thompson.F:3005
n0_exp = tc_pow(10.f,(zans1));
// MPAS module_mp_thompson.F:3006
n0_exp = tc_max(((double)(gonv_min)),tc_min(n0_exp,((double)(gonv_max))));
// MPAS module_mp_thompson.F:3007
n0_min = tc_min(n0_exp,n0_min);
// MPAS module_mp_thompson.F:3008
n0_exp = n0_min;
// MPAS module_mp_thompson.F:3009
lam_exp = tc_pow((tc_div(tc_mul(tc_mul(n0_exp,am_g),cgg[0+tc_sub(1,(1))*(1)]),rg[0+tc_sub(k,(kts))*(1)])),oge1);
// MPAS module_mp_thompson.F:3010
lamg = tc_mul(lam_exp,tc_pow((tc_mul(tc_mul(cgg[0+tc_sub(3,(1))*(1)],ogg2),ogg1)),obmg));
// MPAS module_mp_thompson.F:3011
ilamg[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamg);
// MPAS module_mp_thompson.F:3012
n0_g[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(n0_exp,(tc_mul(cgg[0+tc_sub(2,(1))*(1)],lam_exp))),tc_pow(lamg,cge[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:3013
}
// MPAS module_mp_thompson.F:3015
}
// MPAS module_mp_thompson.F:3020
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3021
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:3022
ilamr[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamr);
// MPAS module_mp_thompson.F:3023
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:3024
n0_r[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(nr[0+tc_sub(k,(kts))*(1)],org2),tc_pow(lamr,cre[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:3025
}
// MPAS module_mp_thompson.F:3035
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:3036
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3037
if((((ssatw[0+tc_sub(k,(kts))*(1)]>eps))||(((ssatw[0+tc_sub(k,(kts))*(1)]<(-eps))&&l_qc[0+tc_sub(k,(kts))*(1)])))) {
// MPAS module_mp_thompson.F:3039
clap = tc_div((tc_sub(qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)])),(tc_add(1.f,tc_mul(lvt2[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:3040
for(n=1; n<=3; n+=1) {
// MPAS module_mp_thompson.F:3041
fcd = tc_add(tc_sub(tc_mul(qvs[0+tc_sub(k,(kts))*(1)],tc_exp(tc_mul(lvt2[0+tc_sub(k,(kts))*(1)],clap))),qv[0+tc_sub(k,(kts))*(1)]),clap);
// MPAS module_mp_thompson.F:3042
dfcd = tc_add(tc_mul(tc_mul(qvs[0+tc_sub(k,(kts))*(1)],lvt2[0+tc_sub(k,(kts))*(1)]),tc_exp(tc_mul(lvt2[0+tc_sub(k,(kts))*(1)],clap))),1.f);
// MPAS module_mp_thompson.F:3043
clap = tc_sub(clap,tc_div(fcd,dfcd));
// MPAS module_mp_thompson.F:3044
}
// MPAS module_mp_thompson.F:3045
xrc = tc_add(rc[0+tc_sub(k,(kts))*(1)],tc_mul(clap,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3046
xnc = 0.f;
// MPAS module_mp_thompson.F:3047
if((xrc>r1)) {
// MPAS module_mp_thompson.F:3048
prw_vcd[0+tc_sub(k,(kts))*(1)] = tc_mul(clap,odt);
// MPAS module_mp_thompson.F:3050
if((clap>eps)) {
// MPAS module_mp_thompson.F:3051
if(is_aerosol_aware) {
// MPAS module_mp_thompson.F:3052
xnc = tc_max(2.f,activ_ncloud(temp[0+tc_sub(k,(kts))*(1)],w1d[0+tc_sub(k,(kts))*(1)],nwfa[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3053
} else {
// MPAS module_mp_thompson.F:3054
xnc = nt_c;
// MPAS module_mp_thompson.F:3055
}
// MPAS module_mp_thompson.F:3056
pnc_wcd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(0.5f,(tc_add(tc_sub(xnc,nc[0+tc_sub(k,(kts))*(1)]),tc_abs(tc_sub(xnc,nc[0+tc_sub(k,(kts))*(1)]))))),odts),orho);
// MPAS module_mp_thompson.F:3059
} else if((((clap<(-eps))&&(ssatw[0+tc_sub(k,(kts))*(1)]<(-1.e-6f)))&&is_aerosol_aware)) {
// MPAS module_mp_thompson.F:3060
tempc = tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f);
// MPAS module_mp_thompson.F:3061
otemp = tc_div(1.f,temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3062
rvs = tc_mul(rho[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3063
rvs_p = tc_mul(tc_mul(rvs,otemp),(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f)));
// MPAS module_mp_thompson.F:3064
rvs_pp = tc_mul(rvs,(tc_add(tc_add(tc_mul(tc_mul(tc_mul(otemp,(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f))),otemp),(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f))),(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul((-2.f),lvap[0+tc_sub(k,(kts))*(1)]),otemp),otemp),otemp),orv))),tc_mul(otemp,otemp))));
// MPAS module_mp_thompson.F:3068
gamsc = tc_mul(tc_div(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],diffu[0+tc_sub(k,(kts))*(1)]),tcond[0+tc_sub(k,(kts))*(1)]),rvs_p);
// MPAS module_mp_thompson.F:3069
alphsc = tc_div(tc_mul(tc_div(tc_mul(tc_mul(tc_mul(0.5f,(tc_div(gamsc,(tc_add(1.f,gamsc))))),(tc_div(gamsc,(tc_add(1.f,gamsc))))),rvs_pp),rvs_p),rvs),rvs_p);
// MPAS module_mp_thompson.F:3071
alphsc = tc_max(1.e-9f,alphsc);
// MPAS module_mp_thompson.F:3072
xsat = ssatw[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:3073
if((tc_abs(xsat)<1.e-9f)) { xsat = 0.f; }
// MPAS module_mp_thompson.F:3074
t1_evap = tc_div(tc_mul(tc_mul(2.f,pi),(tc_sub(tc_add(tc_sub(1.0f,tc_mul(alphsc,xsat)),tc_mul(tc_mul(tc_mul(tc_mul(2.f,alphsc),alphsc),xsat),xsat)),tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(5.f,alphsc),alphsc),alphsc),xsat),xsat),xsat)))),(tc_add(1.f,gamsc)));
// MPAS module_mp_thompson.F:3079
dc_star = tc_sqrt(tc_div(tc_mul(tc_mul(tc_mul(tc_mul(tc_div(tc_mul(tc_mul((-2.e0),dt),t1_evap),(tc_mul(2.f,pi))),4.f),diffu[0+tc_sub(k,(kts))*(1)]),ssatw[0+tc_sub(k,(kts))*(1)]),rvs),rho_w));
// MPAS module_mp_thompson.F:3081
idx_d = tc_max(1,tc_min(((int)(tc_mul(1.e6f,dc_star))),nbc));
// MPAS module_mp_thompson.F:3083
idx_n = ((int)round(tc_add(1.0f,tc_div(tc_mul(float(nbc),tc_log(tc_div(nc[0+tc_sub(k,(kts))*(1)],t_nc[0+tc_sub(1,(1))*(1)]))),nic1))));
// MPAS module_mp_thompson.F:3084
idx_n = tc_max(1,tc_min(idx_n,nbc));
// MPAS module_mp_thompson.F:3087
if((rc[0+tc_sub(k,(kts))*(1)]>r_c[0+tc_sub(1,(1))*(1)])) {
// MPAS module_mp_thompson.F:3088
nic = ((int)round(tc_log10(rc[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3089
for(nn=tc_sub(nic,1); nn<=tc_add(nic,1); nn+=1) {
// MPAS module_mp_thompson.F:3090
n = nn;
// MPAS module_mp_thompson.F:3091
if((((tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))>=1.0f)&&((tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,nn)))<10.0f))) { goto label_159; }
// MPAS module_mp_thompson.F:3093
}
// MPAS module_mp_thompson.F:3094
label_159: ;

// MPAS module_mp_thompson.F:3095
idx_c = tc_sub(tc_add(((int)(tc_div(rc[0+tc_sub(k,(kts))*(1)],tc_pow(10.f,n)))),tc_mul(10,(tc_sub(n,nic2)))),(tc_sub(n,nic2)));
// MPAS module_mp_thompson.F:3096
idx_c = tc_max(1,tc_min(idx_c,ntb_c));
// MPAS module_mp_thompson.F:3097
} else {
// MPAS module_mp_thompson.F:3098
idx_c = 1;
// MPAS module_mp_thompson.F:3099
}
// MPAS module_mp_thompson.F:3103
prw_vcd[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul(tc_mul(tc_mul((-rc[0+tc_sub(k,(kts))*(1)]),0.99f),orho),odt))),prw_vcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3104
pnc_wcd[0+tc_sub(k,(kts))*(1)] = tc_max(((double)(tc_mul(tc_mul(tc_mul((-nc[0+tc_sub(k,(kts))*(1)]),0.99f),orho),odt))),tc_mul(tc_mul((-tnc_wev[0+tc_sub(idx_d,(1))*(1)+tc_sub(idx_c,(1))*tc_mul(1,((100)-(1)+1))+tc_sub(idx_n,(1))*(1*((100)-(1)+1)*((37)-(1)+1))]),orho),odt));
// MPAS module_mp_thompson.F:3107
}
// MPAS module_mp_thompson.F:3108
if((is_aerosol_aware&&l_nwfa[0+tc_sub(k,(kts))*(1)])) { l_nwfa[0+tc_sub(k,(kts))*(1)] = false; }
// MPAS module_mp_thompson.F:3109
} else {
// MPAS module_mp_thompson.F:3110
prw_vcd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((-rc[0+tc_sub(k,(kts))*(1)]),orho),odt);
// MPAS module_mp_thompson.F:3111
pnc_wcd[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((-nc[0+tc_sub(k,(kts))*(1)]),orho),odt);
// MPAS module_mp_thompson.F:3112
}
// MPAS module_mp_thompson.F:3116
qvten[0+tc_sub(k,(kts))*(1)] = tc_sub(qvten[0+tc_sub(k,(kts))*(1)],prw_vcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3117
qcten[0+tc_sub(k,(kts))*(1)] = tc_add(qcten[0+tc_sub(k,(kts))*(1)],prw_vcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3118
ncten[0+tc_sub(k,(kts))*(1)] = tc_add(ncten[0+tc_sub(k,(kts))*(1)],pnc_wcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3119
nwfaten[0+tc_sub(k,(kts))*(1)] = tc_sub(nwfaten[0+tc_sub(k,(kts))*(1)],pnc_wcd[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3120
tten[0+tc_sub(k,(kts))*(1)] = tc_add(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],ocp[0+tc_sub(k,(kts))*(1)]),prw_vcd[0+tc_sub(k,(kts))*(1)]),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:3121
rc[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_mul((tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,qcten[0+tc_sub(k,(kts))*(1)]))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3122
if((rc[0+tc_sub(k,(kts))*(1)]==r1)) { l_qc[0+tc_sub(k,(kts))*(1)] = false; }
// MPAS module_mp_thompson.F:3123
nc[0+tc_sub(k,(kts))*(1)] = tc_max(2.f,tc_min(tc_mul((tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dt))),rho[0+tc_sub(k,(kts))*(1)]),nt_c_max));
// MPAS module_mp_thompson.F:3124
if((!is_aerosol_aware)) { nc[0+tc_sub(k,(kts))*(1)] = nt_c; }
// MPAS module_mp_thompson.F:3125
qv[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,tc_add(qv1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,qvten[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3126
temp[0+tc_sub(k,(kts))*(1)] = tc_add(t1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,tten[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3127
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,pres[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,temp[0+tc_sub(k,(kts))*(1)]),(tc_add(qv[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:3128
qvs[0+tc_sub(k,(kts))*(1)] = rslf(pres[0+tc_sub(k,(kts))*(1)],temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3129
ssatw[0+tc_sub(k,(kts))*(1)] = tc_sub(tc_div(qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]),1.f);
// MPAS module_mp_thompson.F:3130
}
// MPAS module_mp_thompson.F:3131
}
// MPAS module_mp_thompson.F:3137
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:3138
if(((((ssatw[0+tc_sub(k,(kts))*(1)]<(-eps)))&&l_qr[0+tc_sub(k,(kts))*(1)])&&((!((prw_vcd[0+tc_sub(k,(kts))*(1)]>0.f)))))) {
// MPAS module_mp_thompson.F:3140
tempc = tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f);
// MPAS module_mp_thompson.F:3141
otemp = tc_div(1.f,temp[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3142
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3143
rhof[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_mul(rho_not,orho));
// MPAS module_mp_thompson.F:3144
rhof2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(rhof[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3145
diffu[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(2.11e-5f,tc_pow((tc_div(temp[0+tc_sub(k,(kts))*(1)],273.15f)),1.94f)),(tc_div(101325.f,pres[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3146
if((tempc>=0.0f)) {
// MPAS module_mp_thompson.F:3147
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(1.718f,tc_mul(0.0049f,tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:3148
} else {
// MPAS module_mp_thompson.F:3149
visco[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_sub(tc_add(1.718f,tc_mul(0.0049f,tempc)),tc_mul(tc_mul(1.2e-5f,tempc),tempc))),1.0e-5f);
// MPAS module_mp_thompson.F:3150
}
// MPAS module_mp_thompson.F:3151
vsc2[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho[0+tc_sub(k,(kts))*(1)],visco[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3152
lvap[0+tc_sub(k,(kts))*(1)] = tc_add(lvap0,tc_mul((tc_sub(2106.0f,4218.0f)),tempc));
// MPAS module_mp_thompson.F:3153
tcond[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul((tc_add(5.69f,tc_mul(0.0168f,tempc))),1.0e-5f),418.936f);
// MPAS module_mp_thompson.F:3154
ocp[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,(tc_mul(cp,(tc_add(1.f,tc_mul(0.887f,qv[0+tc_sub(k,(kts))*(1)]))))));
// MPAS module_mp_thompson.F:3156
rvs = tc_mul(rho[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3157
rvs_p = tc_mul(tc_mul(rvs,otemp),(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f)));
// MPAS module_mp_thompson.F:3158
rvs_pp = tc_mul(rvs,(tc_add(tc_add(tc_mul(tc_mul(tc_mul(otemp,(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f))),otemp),(tc_sub(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],otemp),orv),1.f))),(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul((-2.f),lvap[0+tc_sub(k,(kts))*(1)]),otemp),otemp),otemp),orv))),tc_mul(otemp,otemp))));
// MPAS module_mp_thompson.F:3162
gamsc = tc_mul(tc_div(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],diffu[0+tc_sub(k,(kts))*(1)]),tcond[0+tc_sub(k,(kts))*(1)]),rvs_p);
// MPAS module_mp_thompson.F:3163
alphsc = tc_div(tc_mul(tc_div(tc_mul(tc_mul(tc_mul(0.5f,(tc_div(gamsc,(tc_add(1.f,gamsc))))),(tc_div(gamsc,(tc_add(1.f,gamsc))))),rvs_pp),rvs_p),rvs),rvs_p);
// MPAS module_mp_thompson.F:3165
alphsc = tc_max(1.e-9f,alphsc);
// MPAS module_mp_thompson.F:3166
xsat = tc_min((-1.e-9f),ssatw[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3167
t1_evap = tc_div(tc_mul(tc_mul(2.f,pi),(tc_sub(tc_add(tc_sub(1.0f,tc_mul(alphsc,xsat)),tc_mul(tc_mul(tc_mul(tc_mul(2.f,alphsc),alphsc),xsat),xsat)),tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(5.f,alphsc),alphsc),alphsc),xsat),xsat),xsat)))),(tc_add(1.f,gamsc)));
// MPAS module_mp_thompson.F:3172
lamr = tc_div(1.f,ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3174
if(((tc_div(qv[0+tc_sub(k,(kts))*(1)],qvs[0+tc_sub(k,(kts))*(1)])<0.95f)&&(tc_mul(rr[0+tc_sub(k,(kts))*(1)],orho)<=1.e-8f))) {
// MPAS module_mp_thompson.F:3175
prv_rev[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(rr[0+tc_sub(k,(kts))*(1)],orho),odts);
// MPAS module_mp_thompson.F:3176
} else {
// MPAS module_mp_thompson.F:3177
prv_rev[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(t1_evap,diffu[0+tc_sub(k,(kts))*(1)]),((-ssatw[0+tc_sub(k,(kts))*(1)]))),n0_r[0+tc_sub(k,(kts))*(1)]),rvs),(tc_add(tc_mul(t1_qr_ev,tc_pow(ilamr[0+tc_sub(k,(kts))*(1)],cre[0+tc_sub(10,(1))*(1)])),tc_mul(tc_mul(tc_mul(t2_qr_ev,vsc2[0+tc_sub(k,(kts))*(1)]),rhof2[0+tc_sub(k,(kts))*(1)]),(tc_pow((tc_add(lamr,tc_mul(0.5f,fv_r))),((-cre[0+tc_sub(11,(1))*(1)]))))))));
// MPAS module_mp_thompson.F:3180
rate_max = tc_min((tc_mul(tc_mul(rr[0+tc_sub(k,(kts))*(1)],orho),odts)),tc_mul((tc_sub(qvs[0+tc_sub(k,(kts))*(1)],qv[0+tc_sub(k,(kts))*(1)])),odts));
// MPAS module_mp_thompson.F:3181
prv_rev[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(rate_max)),tc_mul(prv_rev[0+tc_sub(k,(kts))*(1)],orho));
// MPAS module_mp_thompson.F:3190
if((prr_gml[0+tc_sub(k,(kts))*(1)]>0.0f)) {
// MPAS module_mp_thompson.F:3191
eva_factor = tc_min(1.0f,tc_add(0.01f,tc_mul((tc_sub(0.99f,0.01f)),(tc_div(tempc,20.0f)))));
// MPAS module_mp_thompson.F:3192
prv_rev[0+tc_sub(k,(kts))*(1)] = tc_mul(prv_rev[0+tc_sub(k,(kts))*(1)],eva_factor);
// MPAS module_mp_thompson.F:3193
}
// MPAS module_mp_thompson.F:3194
}
// MPAS module_mp_thompson.F:3196
pnr_rev[0+tc_sub(k,(kts))*(1)] = tc_min(((double)(tc_mul(tc_mul(tc_mul(nr[0+tc_sub(k,(kts))*(1)],0.99f),orho),odts))),tc_div(tc_mul(prv_rev[0+tc_sub(k,(kts))*(1)],nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3199
qrten[0+tc_sub(k,(kts))*(1)] = tc_sub(qrten[0+tc_sub(k,(kts))*(1)],prv_rev[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3200
qvten[0+tc_sub(k,(kts))*(1)] = tc_add(qvten[0+tc_sub(k,(kts))*(1)],prv_rev[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3201
nrten[0+tc_sub(k,(kts))*(1)] = tc_sub(nrten[0+tc_sub(k,(kts))*(1)],pnr_rev[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3202
nwfaten[0+tc_sub(k,(kts))*(1)] = tc_add(nwfaten[0+tc_sub(k,(kts))*(1)],pnr_rev[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3203
tten[0+tc_sub(k,(kts))*(1)] = tc_sub(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(lvap[0+tc_sub(k,(kts))*(1)],ocp[0+tc_sub(k,(kts))*(1)]),prv_rev[0+tc_sub(k,(kts))*(1)]),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:3205
rr[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_mul((tc_add(qr1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,qrten[0+tc_sub(k,(kts))*(1)]))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3206
qv[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,tc_add(qv1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,qvten[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3207
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul((tc_add(nr1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,nrten[0+tc_sub(k,(kts))*(1)]))),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3208
temp[0+tc_sub(k,(kts))*(1)] = tc_add(t1d[0+tc_sub(k,(kts))*(1)],tc_mul(dt,tten[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3209
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,pres[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,temp[0+tc_sub(k,(kts))*(1)]),(tc_add(qv[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:3210
}
// MPAS module_mp_thompson.F:3211
}
// MPAS module_mp_thompson.F:3212
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:3213
evapprod[0+tc_sub(k,(kts))*(1)] = tc_sub(prv_rev[0+tc_sub(k,(kts))*(1)],(tc_add(tc_min(zerod0,prs_sde[0+tc_sub(k,(kts))*(1)]),tc_min(zerod0,prg_gde[0+tc_sub(k,(kts))*(1)]))));
// MPAS module_mp_thompson.F:3215
rainprod[0+tc_sub(k,(kts))*(1)] = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(prr_wau[0+tc_sub(k,(kts))*(1)],prr_rcw[0+tc_sub(k,(kts))*(1)]),prs_scw[0+tc_sub(k,(kts))*(1)]),prg_scw[0+tc_sub(k,(kts))*(1)]),prs_iau[0+tc_sub(k,(kts))*(1)]),prg_gcw[0+tc_sub(k,(kts))*(1)]),prs_sci[0+tc_sub(k,(kts))*(1)]),pri_rci[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3219
}
// MPAS module_mp_thompson.F:3229
nstep = 0;
// MPAS module_mp_thompson.F:3230
for(int _slice=1;_slice<=5;++_slice) onstep[0+tc_sub(_slice,(1))*(1)] = 1.0f;
// MPAS module_mp_thompson.F:3231
for(int _slice=1;_slice<=5;++_slice) ksed1[0+tc_sub(_slice,(1))*(1)] = 1;
// MPAS module_mp_thompson.F:3232
for(k=tc_add(kte,1); k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3233
vtrk[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3234
vtnrk[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3235
vtik[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3236
vtnik[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3237
vtsk[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3238
vtgk[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3239
vtck[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3240
vtnck[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:3241
}
// MPAS module_mp_thompson.F:3243
if(tc_any(l_qr,kts,kte)) {
// MPAS module_mp_thompson.F:3244
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3245
vtr = 0.f;
// MPAS module_mp_thompson.F:3246
rhof[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho_not,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3248
if((rr[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:3249
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:3250
vtr = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_r),crg[0+tc_sub(6,(1))*(1)]),org3),tc_pow(lamr,cre[0+tc_sub(3,(1))*(1)])),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(6,(1))*(1)])))));
// MPAS module_mp_thompson.F:3252
vtrk[0+tc_sub(k,(kts))*(1)] = vtr;
// MPAS module_mp_thompson.F:3258
vtr = tc_mul(tc_mul(tc_div(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_r),crg[0+tc_sub(7,(1))*(1)]),crg[0+tc_sub(12,(1))*(1)]),tc_pow(lamr,cre[0+tc_sub(12,(1))*(1)])),(tc_pow((tc_add(lamr,fv_r)),((-cre[0+tc_sub(7,(1))*(1)])))));
// MPAS module_mp_thompson.F:3260
vtnrk[0+tc_sub(k,(kts))*(1)] = vtr;
// MPAS module_mp_thompson.F:3261
} else {
// MPAS module_mp_thompson.F:3262
vtrk[0+tc_sub(k,(kts))*(1)] = vtrk[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3263
vtnrk[0+tc_sub(k,(kts))*(1)] = vtnrk[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3264
}
// MPAS module_mp_thompson.F:3266
if((tc_max(vtrk[0+tc_sub(k,(kts))*(1)],vtnrk[0+tc_sub(k,(kts))*(1)])>1.e-3f)) {
// MPAS module_mp_thompson.F:3267
ksed1[0+tc_sub(1,(1))*(1)] = tc_max(ksed1[0+tc_sub(1,(1))*(1)],k);
// MPAS module_mp_thompson.F:3268
delta_tp = tc_div(dzq[0+tc_sub(k,(kts))*(1)],(tc_max(vtrk[0+tc_sub(k,(kts))*(1)],vtnrk[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3269
nstep = tc_max(nstep,((int)(tc_add(tc_div(dt,delta_tp),1.f))));
// MPAS module_mp_thompson.F:3270
}
// MPAS module_mp_thompson.F:3271
}
// MPAS module_mp_thompson.F:3272
if((ksed1[0+tc_sub(1,(1))*(1)]==kte)) { ksed1[0+tc_sub(1,(1))*(1)] = tc_sub(kte,1); }
// MPAS module_mp_thompson.F:3273
if((nstep>0)) { onstep[0+tc_sub(1,(1))*(1)] = tc_div(1.f,((float)(nstep))); }
// MPAS module_mp_thompson.F:3274
}
// MPAS module_mp_thompson.F:3278
if(tc_any(l_qc,kts,kte)) {
// MPAS module_mp_thompson.F:3279
hgt_agl = 0.f;
// MPAS module_mp_thompson.F:3280
for(k=kts; k<=tc_sub(kte,1); k+=1) {
// MPAS module_mp_thompson.F:3281
if((rc[0+tc_sub(k,(kts))*(1)]>r2)) { ksed1[0+tc_sub(5,(1))*(1)] = k; }
// MPAS module_mp_thompson.F:3282
hgt_agl = tc_add(hgt_agl,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3283
if((hgt_agl>500.0f)) { goto label_151; }
// MPAS module_mp_thompson.F:3284
}
// MPAS module_mp_thompson.F:3285
label_151: ;

// MPAS module_mp_thompson.F:3287
for(k=ksed1[0+tc_sub(5,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3288
vtc = 0.f;
// MPAS module_mp_thompson.F:3289
if(((rc[0+tc_sub(k,(kts))*(1)]>r1)&&(w1d[0+tc_sub(k,(kts))*(1)]<1.e-1f))) {
// MPAS module_mp_thompson.F:3290
nu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,nc[0+tc_sub(k,(kts))*(1)]))),2));
// MPAS module_mp_thompson.F:3291
lamc = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(nc[0+tc_sub(k,(kts))*(1)],am_r),ccg[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:3292
ilamc = tc_div(1.f,lamc);
// MPAS module_mp_thompson.F:3293
vtc = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_c),ccg[0+tc_sub(5,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg2[0+tc_sub(nu_c,(1))*(1)]),tc_pow(ilamc,bv_c));
// MPAS module_mp_thompson.F:3294
vtck[0+tc_sub(k,(kts))*(1)] = vtc;
// MPAS module_mp_thompson.F:3295
vtc = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_c),ccg[0+tc_sub(4,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),tc_pow(ilamc,bv_c));
// MPAS module_mp_thompson.F:3296
vtnck[0+tc_sub(k,(kts))*(1)] = vtc;
// MPAS module_mp_thompson.F:3297
}
// MPAS module_mp_thompson.F:3298
}
// MPAS module_mp_thompson.F:3299
}
// MPAS module_mp_thompson.F:3303
if((!iiwarm)) {
// MPAS module_mp_thompson.F:3305
if(tc_any(l_qi,kts,kte)) {
// MPAS module_mp_thompson.F:3306
nstep = 0;
// MPAS module_mp_thompson.F:3307
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3308
vti = 0.f;
// MPAS module_mp_thompson.F:3310
if((ri[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:3311
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni[0+tc_sub(k,(kts))*(1)]),ri[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:3312
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:3313
vti = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_i),cig[0+tc_sub(3,(1))*(1)]),oig2),tc_pow(ilami,bv_i));
// MPAS module_mp_thompson.F:3314
vtik[0+tc_sub(k,(kts))*(1)] = vti;
// MPAS module_mp_thompson.F:3318
vti = tc_mul(tc_div(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_i),cig[0+tc_sub(6,(1))*(1)]),cig[0+tc_sub(7,(1))*(1)]),tc_pow(ilami,bv_i));
// MPAS module_mp_thompson.F:3319
vtnik[0+tc_sub(k,(kts))*(1)] = vti;
// MPAS module_mp_thompson.F:3320
} else {
// MPAS module_mp_thompson.F:3321
vtik[0+tc_sub(k,(kts))*(1)] = vtik[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3322
vtnik[0+tc_sub(k,(kts))*(1)] = vtnik[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3323
}
// MPAS module_mp_thompson.F:3325
if((vtik[0+tc_sub(k,(kts))*(1)]>1.e-3f)) {
// MPAS module_mp_thompson.F:3326
ksed1[0+tc_sub(2,(1))*(1)] = tc_max(ksed1[0+tc_sub(2,(1))*(1)],k);
// MPAS module_mp_thompson.F:3327
delta_tp = tc_div(dzq[0+tc_sub(k,(kts))*(1)],vtik[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3328
nstep = tc_max(nstep,((int)(tc_add(tc_div(dt,delta_tp),1.f))));
// MPAS module_mp_thompson.F:3329
}
// MPAS module_mp_thompson.F:3330
}
// MPAS module_mp_thompson.F:3331
if((ksed1[0+tc_sub(2,(1))*(1)]==kte)) { ksed1[0+tc_sub(2,(1))*(1)] = tc_sub(kte,1); }
// MPAS module_mp_thompson.F:3332
if((nstep>0)) { onstep[0+tc_sub(2,(1))*(1)] = tc_div(1.f,((float)(nstep))); }
// MPAS module_mp_thompson.F:3333
}
// MPAS module_mp_thompson.F:3337
if(tc_any(l_qs,kts,kte)) {
// MPAS module_mp_thompson.F:3338
nstep = 0;
// MPAS module_mp_thompson.F:3339
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3340
vts = 0.f;
// MPAS module_mp_thompson.F:3342
if((rs[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:3343
xds = tc_div(smoc[0+tc_sub(k,(kts))*(1)],smob[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3344
mrat = tc_div(1.f,xds);
// MPAS module_mp_thompson.F:3345
ils1 = tc_div(1.f,(tc_add(tc_mul(mrat,lam0),fv_s)));
// MPAS module_mp_thompson.F:3346
ils2 = tc_div(1.f,(tc_add(tc_mul(mrat,lam1),fv_s)));
// MPAS module_mp_thompson.F:3347
t1_vts = tc_mul(tc_mul(kap0,csg[0+tc_sub(4,(1))*(1)]),tc_pow(ils1,cse[0+tc_sub(4,(1))*(1)]));
// MPAS module_mp_thompson.F:3348
t2_vts = tc_mul(tc_mul(tc_mul(kap1,tc_pow(mrat,mu_s)),csg[0+tc_sub(10,(1))*(1)]),tc_pow(ils2,cse[0+tc_sub(10,(1))*(1)]));
// MPAS module_mp_thompson.F:3349
ils1 = tc_div(1.f,(tc_mul(mrat,lam0)));
// MPAS module_mp_thompson.F:3350
ils2 = tc_div(1.f,(tc_mul(mrat,lam1)));
// MPAS module_mp_thompson.F:3351
t3_vts = tc_mul(tc_mul(kap0,csg[0+tc_sub(1,(1))*(1)]),tc_pow(ils1,cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:3352
t4_vts = tc_mul(tc_mul(tc_mul(kap1,tc_pow(mrat,mu_s)),csg[0+tc_sub(7,(1))*(1)]),tc_pow(ils2,cse[0+tc_sub(7,(1))*(1)]));
// MPAS module_mp_thompson.F:3353
vts = tc_div(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_s),(tc_add(t1_vts,t2_vts))),(tc_add(t3_vts,t4_vts)));
// MPAS module_mp_thompson.F:3354
if((temp[0+tc_sub(k,(kts))*(1)]>(tc_add(t_0,0.1f)))) {
// MPAS module_mp_thompson.F:3355
sr = tc_div(rs[0+tc_sub(k,(kts))*(1)],(tc_add(rs[0+tc_sub(k,(kts))*(1)],rr[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:3356
vtsk[0+tc_sub(k,(kts))*(1)] = tc_add(tc_mul(vts,sr),tc_mul((tc_sub(1.f,sr)),vtrk[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3357
} else {
// MPAS module_mp_thompson.F:3358
vtsk[0+tc_sub(k,(kts))*(1)] = tc_mul(vts,vts_boost[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3359
}
// MPAS module_mp_thompson.F:3360
} else {
// MPAS module_mp_thompson.F:3361
vtsk[0+tc_sub(k,(kts))*(1)] = vtsk[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3362
}
// MPAS module_mp_thompson.F:3364
if((vtsk[0+tc_sub(k,(kts))*(1)]>1.e-3f)) {
// MPAS module_mp_thompson.F:3365
ksed1[0+tc_sub(3,(1))*(1)] = tc_max(ksed1[0+tc_sub(3,(1))*(1)],k);
// MPAS module_mp_thompson.F:3366
delta_tp = tc_div(dzq[0+tc_sub(k,(kts))*(1)],vtsk[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3367
nstep = tc_max(nstep,((int)(tc_add(tc_div(dt,delta_tp),1.f))));
// MPAS module_mp_thompson.F:3368
}
// MPAS module_mp_thompson.F:3369
}
// MPAS module_mp_thompson.F:3370
if((ksed1[0+tc_sub(3,(1))*(1)]==kte)) { ksed1[0+tc_sub(3,(1))*(1)] = tc_sub(kte,1); }
// MPAS module_mp_thompson.F:3371
if((nstep>0)) { onstep[0+tc_sub(3,(1))*(1)] = tc_div(1.f,((float)(nstep))); }
// MPAS module_mp_thompson.F:3372
}
// MPAS module_mp_thompson.F:3376
if(tc_any(l_qg,kts,kte)) {
// MPAS module_mp_thompson.F:3377
nstep = 0;
// MPAS module_mp_thompson.F:3378
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3379
vtg = 0.f;
// MPAS module_mp_thompson.F:3381
if((rg[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:3382
vtg = tc_mul(tc_mul(tc_mul(tc_mul(rhof[0+tc_sub(k,(kts))*(1)],av_g),cgg[0+tc_sub(6,(1))*(1)]),ogg3),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],bv_g));
// MPAS module_mp_thompson.F:3383
if((temp[0+tc_sub(k,(kts))*(1)]>t_0)) {
// MPAS module_mp_thompson.F:3384
vtgk[0+tc_sub(k,(kts))*(1)] = tc_max(vtg,vtrk[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3385
} else {
// MPAS module_mp_thompson.F:3386
vtgk[0+tc_sub(k,(kts))*(1)] = vtg;
// MPAS module_mp_thompson.F:3387
}
// MPAS module_mp_thompson.F:3388
} else {
// MPAS module_mp_thompson.F:3389
vtgk[0+tc_sub(k,(kts))*(1)] = vtgk[0+tc_sub(tc_add(k,1),(kts))*(1)];
// MPAS module_mp_thompson.F:3390
}
// MPAS module_mp_thompson.F:3392
if((vtgk[0+tc_sub(k,(kts))*(1)]>1.e-3f)) {
// MPAS module_mp_thompson.F:3393
ksed1[0+tc_sub(4,(1))*(1)] = tc_max(ksed1[0+tc_sub(4,(1))*(1)],k);
// MPAS module_mp_thompson.F:3394
delta_tp = tc_div(dzq[0+tc_sub(k,(kts))*(1)],vtgk[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3395
nstep = tc_max(nstep,((int)(tc_add(tc_div(dt,delta_tp),1.f))));
// MPAS module_mp_thompson.F:3396
}
// MPAS module_mp_thompson.F:3397
}
// MPAS module_mp_thompson.F:3398
if((ksed1[0+tc_sub(4,(1))*(1)]==kte)) { ksed1[0+tc_sub(4,(1))*(1)] = tc_sub(kte,1); }
// MPAS module_mp_thompson.F:3399
if((nstep>0)) { onstep[0+tc_sub(4,(1))*(1)] = tc_div(1.f,((float)(nstep))); }
// MPAS module_mp_thompson.F:3400
}
// MPAS module_mp_thompson.F:3401
}
// MPAS module_mp_thompson.F:3409
if(tc_any(l_qr,kts,kte)) {
// MPAS module_mp_thompson.F:3410
nstep = ((int)round(tc_div(1.f,onstep[0+tc_sub(1,(1))*(1)])));
// MPAS module_mp_thompson.F:3411
for(n=1; n<=nstep; n+=1) {
// MPAS module_mp_thompson.F:3412
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3413
sed_r[0+tc_sub(k,(kts))*(1)] = tc_mul(vtrk[0+tc_sub(k,(kts))*(1)],rr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3414
sed_n[0+tc_sub(k,(kts))*(1)] = tc_mul(vtnrk[0+tc_sub(k,(kts))*(1)],nr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3415
}
// MPAS module_mp_thompson.F:3416
k = kte;
// MPAS module_mp_thompson.F:3417
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3418
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3419
qrten[0+tc_sub(k,(kts))*(1)] = tc_sub(qrten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_r[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(1,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3420
nrten[0+tc_sub(k,(kts))*(1)] = tc_sub(nrten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_n[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(1,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3421
rr[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_sub(rr[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_r[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(1,(1))*(1)])));
// MPAS module_mp_thompson.F:3422
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_sub(nr[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_n[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(1,(1))*(1)])));
// MPAS module_mp_thompson.F:3423
for(k=ksed1[0+tc_sub(1,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3424
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3425
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3426
qrten[0+tc_sub(k,(kts))*(1)] = tc_add(qrten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_r[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_r[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(1,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3428
nrten[0+tc_sub(k,(kts))*(1)] = tc_add(nrten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(1,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3430
rr[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_add(rr[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_r[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_r[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(1,(1))*(1)])));
// MPAS module_mp_thompson.F:3432
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_add(nr[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(1,(1))*(1)])));
// MPAS module_mp_thompson.F:3434
}
// MPAS module_mp_thompson.F:3436
if((rr[0+tc_sub(kts,(kts))*(1)]>tc_mul(r1,1000.f))) { pptrain = tc_add(pptrain,tc_mul(tc_mul(sed_r[0+tc_sub(kts,(kts))*(1)],dt),onstep[0+tc_sub(1,(1))*(1)])); }
// MPAS module_mp_thompson.F:3438
}
// MPAS module_mp_thompson.F:3439
}
// MPAS module_mp_thompson.F:3443
if(tc_any(l_qc,kts,kte)) {
// MPAS module_mp_thompson.F:3444
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3445
sed_c[0+tc_sub(k,(kts))*(1)] = tc_mul(vtck[0+tc_sub(k,(kts))*(1)],rc[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3446
sed_n[0+tc_sub(k,(kts))*(1)] = tc_mul(vtnck[0+tc_sub(k,(kts))*(1)],nc[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3447
}
// MPAS module_mp_thompson.F:3448
for(k=ksed1[0+tc_sub(5,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3449
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3450
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3451
qcten[0+tc_sub(k,(kts))*(1)] = tc_add(qcten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_sub(sed_c[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_c[0+tc_sub(k,(kts))*(1)])),odzq),orho));
// MPAS module_mp_thompson.F:3452
ncten[0+tc_sub(k,(kts))*(1)] = tc_add(ncten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),orho));
// MPAS module_mp_thompson.F:3453
rc[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_add(rc[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_sub(sed_c[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_c[0+tc_sub(k,(kts))*(1)])),odzq),dt)));
// MPAS module_mp_thompson.F:3454
nc[0+tc_sub(k,(kts))*(1)] = tc_max(10.f,tc_add(nc[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),dt)));
// MPAS module_mp_thompson.F:3455
}
// MPAS module_mp_thompson.F:3456
}
// MPAS module_mp_thompson.F:3460
if(tc_any(l_qi,kts,kte)) {
// MPAS module_mp_thompson.F:3461
nstep = ((int)round(tc_div(1.f,onstep[0+tc_sub(2,(1))*(1)])));
// MPAS module_mp_thompson.F:3462
for(n=1; n<=nstep; n+=1) {
// MPAS module_mp_thompson.F:3463
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3464
sed_i[0+tc_sub(k,(kts))*(1)] = tc_mul(vtik[0+tc_sub(k,(kts))*(1)],ri[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3465
sed_n[0+tc_sub(k,(kts))*(1)] = tc_mul(vtnik[0+tc_sub(k,(kts))*(1)],ni[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3466
}
// MPAS module_mp_thompson.F:3467
k = kte;
// MPAS module_mp_thompson.F:3468
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3469
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3470
qiten[0+tc_sub(k,(kts))*(1)] = tc_sub(qiten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_i[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(2,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3471
niten[0+tc_sub(k,(kts))*(1)] = tc_sub(niten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_n[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(2,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3472
ri[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_sub(ri[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_i[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(2,(1))*(1)])));
// MPAS module_mp_thompson.F:3473
ni[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_sub(ni[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_n[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(2,(1))*(1)])));
// MPAS module_mp_thompson.F:3474
for(k=ksed1[0+tc_sub(2,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3475
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3476
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3477
qiten[0+tc_sub(k,(kts))*(1)] = tc_add(qiten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_i[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_i[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(2,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3479
niten[0+tc_sub(k,(kts))*(1)] = tc_add(niten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(2,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3481
ri[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_add(ri[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_i[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_i[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(2,(1))*(1)])));
// MPAS module_mp_thompson.F:3483
ni[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_add(ni[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_n[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_n[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(2,(1))*(1)])));
// MPAS module_mp_thompson.F:3485
}
// MPAS module_mp_thompson.F:3487
if((ri[0+tc_sub(kts,(kts))*(1)]>tc_mul(r1,1000.f))) { pptice = tc_add(pptice,tc_mul(tc_mul(sed_i[0+tc_sub(kts,(kts))*(1)],dt),onstep[0+tc_sub(2,(1))*(1)])); }
// MPAS module_mp_thompson.F:3489
}
// MPAS module_mp_thompson.F:3490
}
// MPAS module_mp_thompson.F:3494
if(tc_any(l_qs,kts,kte)) {
// MPAS module_mp_thompson.F:3495
nstep = ((int)round(tc_div(1.f,onstep[0+tc_sub(3,(1))*(1)])));
// MPAS module_mp_thompson.F:3496
for(n=1; n<=nstep; n+=1) {
// MPAS module_mp_thompson.F:3497
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3498
sed_s[0+tc_sub(k,(kts))*(1)] = tc_mul(vtsk[0+tc_sub(k,(kts))*(1)],rs[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3499
}
// MPAS module_mp_thompson.F:3500
k = kte;
// MPAS module_mp_thompson.F:3501
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3502
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3503
qsten[0+tc_sub(k,(kts))*(1)] = tc_sub(qsten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_s[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(3,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3504
rs[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_sub(rs[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_s[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(3,(1))*(1)])));
// MPAS module_mp_thompson.F:3505
for(k=ksed1[0+tc_sub(3,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3506
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3507
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3508
qsten[0+tc_sub(k,(kts))*(1)] = tc_add(qsten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_s[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_s[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(3,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3510
rs[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_add(rs[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_s[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_s[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(3,(1))*(1)])));
// MPAS module_mp_thompson.F:3512
}
// MPAS module_mp_thompson.F:3514
if((rs[0+tc_sub(kts,(kts))*(1)]>tc_mul(r1,1000.f))) { pptsnow = tc_add(pptsnow,tc_mul(tc_mul(sed_s[0+tc_sub(kts,(kts))*(1)],dt),onstep[0+tc_sub(3,(1))*(1)])); }
// MPAS module_mp_thompson.F:3516
}
// MPAS module_mp_thompson.F:3517
}
// MPAS module_mp_thompson.F:3521
if(tc_any(l_qg,kts,kte)) {
// MPAS module_mp_thompson.F:3522
nstep = ((int)round(tc_div(1.f,onstep[0+tc_sub(4,(1))*(1)])));
// MPAS module_mp_thompson.F:3523
for(n=1; n<=nstep; n+=1) {
// MPAS module_mp_thompson.F:3524
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3525
sed_g[0+tc_sub(k,(kts))*(1)] = tc_mul(vtgk[0+tc_sub(k,(kts))*(1)],rg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3526
}
// MPAS module_mp_thompson.F:3527
k = kte;
// MPAS module_mp_thompson.F:3528
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3529
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3530
qgten[0+tc_sub(k,(kts))*(1)] = tc_sub(qgten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_g[0+tc_sub(k,(kts))*(1)],odzq),onstep[0+tc_sub(4,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3531
rg[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_sub(rg[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(sed_g[0+tc_sub(k,(kts))*(1)],odzq),dt),onstep[0+tc_sub(4,(1))*(1)])));
// MPAS module_mp_thompson.F:3532
for(k=ksed1[0+tc_sub(4,(1))*(1)]; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:3533
odzq = tc_div(1.f,dzq[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3534
orho = tc_div(1.f,rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3535
qgten[0+tc_sub(k,(kts))*(1)] = tc_add(qgten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_g[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_g[0+tc_sub(k,(kts))*(1)])),odzq),onstep[0+tc_sub(4,(1))*(1)]),orho));
// MPAS module_mp_thompson.F:3537
rg[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_add(rg[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul((tc_sub(sed_g[0+tc_sub(tc_add(k,1),(kts))*(1)],sed_g[0+tc_sub(k,(kts))*(1)])),odzq),dt),onstep[0+tc_sub(4,(1))*(1)])));
// MPAS module_mp_thompson.F:3539
}
// MPAS module_mp_thompson.F:3541
if((rg[0+tc_sub(kts,(kts))*(1)]>tc_mul(r1,1000.f))) { pptgraul = tc_add(pptgraul,tc_mul(tc_mul(sed_g[0+tc_sub(kts,(kts))*(1)],dt),onstep[0+tc_sub(4,(1))*(1)])); }
// MPAS module_mp_thompson.F:3543
}
// MPAS module_mp_thompson.F:3544
}
// MPAS module_mp_thompson.F:3550
if((!iiwarm)) {
// MPAS module_mp_thompson.F:3551
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:3552
xri = tc_max(0.0f,tc_add(qi1d[0+tc_sub(k,(kts))*(1)],tc_mul(qiten[0+tc_sub(k,(kts))*(1)],dt)));
// MPAS module_mp_thompson.F:3553
if((((temp[0+tc_sub(k,(kts))*(1)]>t_0))&&((xri>0.0f)))) {
// MPAS module_mp_thompson.F:3554
qcten[0+tc_sub(k,(kts))*(1)] = tc_add(qcten[0+tc_sub(k,(kts))*(1)],tc_mul(xri,odt));
// MPAS module_mp_thompson.F:3555
ncten[0+tc_sub(k,(kts))*(1)] = tc_add(ncten[0+tc_sub(k,(kts))*(1)],tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],odt));
// MPAS module_mp_thompson.F:3556
qiten[0+tc_sub(k,(kts))*(1)] = tc_sub(qiten[0+tc_sub(k,(kts))*(1)],tc_mul(xri,odt));
// MPAS module_mp_thompson.F:3557
niten[0+tc_sub(k,(kts))*(1)] = tc_mul((-ni1d[0+tc_sub(k,(kts))*(1)]),odt);
// MPAS module_mp_thompson.F:3558
tten[0+tc_sub(k,(kts))*(1)] = tc_sub(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(tc_mul(lfus,ocp[0+tc_sub(k,(kts))*(1)]),xri),odt),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:3559
}
// MPAS module_mp_thompson.F:3561
xrc = tc_max(0.0f,tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(qcten[0+tc_sub(k,(kts))*(1)],dt)));
// MPAS module_mp_thompson.F:3562
if((((temp[0+tc_sub(k,(kts))*(1)]<hgfr))&&((xrc>0.0f)))) {
// MPAS module_mp_thompson.F:3563
lfus2 = tc_sub(lsub,lvap[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3564
xnc = tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3565
qiten[0+tc_sub(k,(kts))*(1)] = tc_add(qiten[0+tc_sub(k,(kts))*(1)],tc_mul(xrc,odt));
// MPAS module_mp_thompson.F:3566
niten[0+tc_sub(k,(kts))*(1)] = tc_add(niten[0+tc_sub(k,(kts))*(1)],tc_mul(xnc,odt));
// MPAS module_mp_thompson.F:3567
qcten[0+tc_sub(k,(kts))*(1)] = tc_sub(qcten[0+tc_sub(k,(kts))*(1)],tc_mul(xrc,odt));
// MPAS module_mp_thompson.F:3568
ncten[0+tc_sub(k,(kts))*(1)] = tc_sub(ncten[0+tc_sub(k,(kts))*(1)],tc_mul(xnc,odt));
// MPAS module_mp_thompson.F:3569
tten[0+tc_sub(k,(kts))*(1)] = tc_add(tten[0+tc_sub(k,(kts))*(1)],tc_mul(tc_mul(tc_mul(tc_mul(lfus2,ocp[0+tc_sub(k,(kts))*(1)]),xrc),odt),(tc_sub(1,ifdry))));
// MPAS module_mp_thompson.F:3570
}
// MPAS module_mp_thompson.F:3571
}
// MPAS module_mp_thompson.F:3572
}
// MPAS module_mp_thompson.F:3577
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:3578
t1d[0+tc_sub(k,(kts))*(1)] = tc_add(t1d[0+tc_sub(k,(kts))*(1)],tc_mul(tten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3579
qv1d[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,tc_add(qv1d[0+tc_sub(k,(kts))*(1)],tc_mul(qvten[0+tc_sub(k,(kts))*(1)],dt)));
// MPAS module_mp_thompson.F:3580
qc1d[0+tc_sub(k,(kts))*(1)] = tc_add(qc1d[0+tc_sub(k,(kts))*(1)],tc_mul(qcten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3581
nc1d[0+tc_sub(k,(kts))*(1)] = tc_max(tc_div(2.f,rho[0+tc_sub(k,(kts))*(1)]),tc_min(tc_add(nc1d[0+tc_sub(k,(kts))*(1)],tc_mul(ncten[0+tc_sub(k,(kts))*(1)],dt)),nt_c_max));
// MPAS module_mp_thompson.F:3582
nwfa1d[0+tc_sub(k,(kts))*(1)] = tc_max(11.1e6f,tc_min(9999.e6f,(tc_add(nwfa1d[0+tc_sub(k,(kts))*(1)],tc_mul(nwfaten[0+tc_sub(k,(kts))*(1)],dt)))));
// MPAS module_mp_thompson.F:3584
nifa1d[0+tc_sub(k,(kts))*(1)] = tc_max(tc_mul(nain1,0.01f),tc_min(9999.e6f,(tc_add(nifa1d[0+tc_sub(k,(kts))*(1)],tc_mul(nifaten[0+tc_sub(k,(kts))*(1)],dt)))));
// MPAS module_mp_thompson.F:3587
if((qc1d[0+tc_sub(k,(kts))*(1)]<=r1)) {
// MPAS module_mp_thompson.F:3588
qc1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3589
nc1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3590
} else {
// MPAS module_mp_thompson.F:3591
nu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,(tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]))))),2));
// MPAS module_mp_thompson.F:3592
lamc = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,ccg[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))]),ocg1[0+tc_sub(nu_c,(1))*(1)]),nc1d[0+tc_sub(k,(kts))*(1)]),qc1d[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:3593
xdc = tc_div((tc_add(tc_add(bm_r,nu_c),1.f)),lamc);
// MPAS module_mp_thompson.F:3594
if((xdc<d0c)) {
// MPAS module_mp_thompson.F:3595
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],d0c);
// MPAS module_mp_thompson.F:3596
} else if((xdc>tc_mul(d0r,2.f))) {
// MPAS module_mp_thompson.F:3597
lamc = tc_div(cce[0+tc_sub(2,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],(tc_mul(d0r,2.f)));
// MPAS module_mp_thompson.F:3598
}
// MPAS module_mp_thompson.F:3599
nc1d[0+tc_sub(k,(kts))*(1)] = tc_min(tc_mul(tc_div(tc_mul(tc_mul(ccg[0+tc_sub(1,(1))*(1)+tc_sub(nu_c,(1))*tc_mul(1,((5)-(1)+1))],ocg2[0+tc_sub(nu_c,(1))*(1)]),qc1d[0+tc_sub(k,(kts))*(1)]),am_r),tc_pow(lamc,bm_r)),tc_div(((double)(nt_c_max)),rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3601
}
// MPAS module_mp_thompson.F:3603
qi1d[0+tc_sub(k,(kts))*(1)] = tc_add(qi1d[0+tc_sub(k,(kts))*(1)],tc_mul(qiten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3604
ni1d[0+tc_sub(k,(kts))*(1)] = tc_max(tc_div(r2,rho[0+tc_sub(k,(kts))*(1)]),tc_add(ni1d[0+tc_sub(k,(kts))*(1)],tc_mul(niten[0+tc_sub(k,(kts))*(1)],dt)));
// MPAS module_mp_thompson.F:3605
if((qi1d[0+tc_sub(k,(kts))*(1)]<=r1)) {
// MPAS module_mp_thompson.F:3606
qi1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3607
ni1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3608
} else {
// MPAS module_mp_thompson.F:3609
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni1d[0+tc_sub(k,(kts))*(1)]),qi1d[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:3610
ilami = tc_div(1.f,lami);
// MPAS module_mp_thompson.F:3611
xdi = tc_mul((tc_add(tc_add(bm_i,mu_i),1.f)),ilami);
// MPAS module_mp_thompson.F:3612
if((xdi<5.e-6f)) {
// MPAS module_mp_thompson.F:3613
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],5.e-6f);
// MPAS module_mp_thompson.F:3614
} else if((xdi>300.e-6f)) {
// MPAS module_mp_thompson.F:3615
lami = tc_div(cie[0+tc_sub(2,(1))*(1)],300.e-6f);
// MPAS module_mp_thompson.F:3616
}
// MPAS module_mp_thompson.F:3617
ni1d[0+tc_sub(k,(kts))*(1)] = tc_min(tc_mul(tc_div(tc_mul(tc_mul(cig[0+tc_sub(1,(1))*(1)],oig2),qi1d[0+tc_sub(k,(kts))*(1)]),am_i),tc_pow(lami,bm_i)),tc_div(9999.e3,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:3619
}
// MPAS module_mp_thompson.F:3620
qr1d[0+tc_sub(k,(kts))*(1)] = tc_add(qr1d[0+tc_sub(k,(kts))*(1)],tc_mul(qrten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3621
nr1d[0+tc_sub(k,(kts))*(1)] = tc_max(tc_div(r2,rho[0+tc_sub(k,(kts))*(1)]),tc_add(nr1d[0+tc_sub(k,(kts))*(1)],tc_mul(nrten[0+tc_sub(k,(kts))*(1)],dt)));
// MPAS module_mp_thompson.F:3622
if((qr1d[0+tc_sub(k,(kts))*(1)]<=r1)) {
// MPAS module_mp_thompson.F:3623
qr1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3624
nr1d[0+tc_sub(k,(kts))*(1)] = 0.0f;
// MPAS module_mp_thompson.F:3625
} else {
// MPAS module_mp_thompson.F:3626
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr1d[0+tc_sub(k,(kts))*(1)]),qr1d[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:3627
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),lamr);
// MPAS module_mp_thompson.F:3628
if((mvd_r[0+tc_sub(k,(kts))*(1)]>2.5e-3f)) {
// MPAS module_mp_thompson.F:3629
mvd_r[0+tc_sub(k,(kts))*(1)] = 2.5e-3f;
// MPAS module_mp_thompson.F:3630
} else if((mvd_r[0+tc_sub(k,(kts))*(1)]<tc_mul(d0r,0.75f))) {
// MPAS module_mp_thompson.F:3631
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_mul(d0r,0.75f);
// MPAS module_mp_thompson.F:3632
}
// MPAS module_mp_thompson.F:3633
lamr = tc_div((tc_add(tc_add(3.0f,mu_r),0.672f)),mvd_r[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:3634
nr1d[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(tc_mul(tc_mul(crg[0+tc_sub(2,(1))*(1)],org3),qr1d[0+tc_sub(k,(kts))*(1)]),tc_pow(lamr,bm_r)),am_r);
// MPAS module_mp_thompson.F:3635
}
// MPAS module_mp_thompson.F:3636
qs1d[0+tc_sub(k,(kts))*(1)] = tc_add(qs1d[0+tc_sub(k,(kts))*(1)],tc_mul(qsten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3637
if((qs1d[0+tc_sub(k,(kts))*(1)]<=r1)) { qs1d[0+tc_sub(k,(kts))*(1)] = 0.0f; }
// MPAS module_mp_thompson.F:3638
qg1d[0+tc_sub(k,(kts))*(1)] = tc_add(qg1d[0+tc_sub(k,(kts))*(1)],tc_mul(qgten[0+tc_sub(k,(kts))*(1)],dt));
// MPAS module_mp_thompson.F:3639
if((qg1d[0+tc_sub(k,(kts))*(1)]<=r1)) { qg1d[0+tc_sub(k,(kts))*(1)] = 0.0f; }
// MPAS module_mp_thompson.F:3640
}
}
// MPAS module_mp_thompson.F:4809 calc_effectrad
__device__ __forceinline__ void calc_effectrad(float* t1d,float* p1d,float* qv1d,float* qc1d,float* nc1d,float* qi1d,float* ni1d,float* qs1d,float* re_qc1d,float* re_qi1d,float* re_qs1d,int kts,int kte) {
int k;
float rho[THOMPSON_COLUMN_KMAX];
float rc[THOMPSON_COLUMN_KMAX];
float nc[THOMPSON_COLUMN_KMAX];
float ri[THOMPSON_COLUMN_KMAX];
float ni[THOMPSON_COLUMN_KMAX];
float rs[THOMPSON_COLUMN_KMAX];
float smo2;
float smob;
float smoc;
float tc0;
float loga_;
float a_;
float b_;
double lamc;
double lami;
bool has_qc;
bool has_qi;
bool has_qs;
int inu_c;
float g_ratio[15] = {24,60,120,210,336,504,720,990,1320,1716,2184,2730,3360,4080,4896};
// MPAS module_mp_thompson.F:4830
has_qc = false;
// MPAS module_mp_thompson.F:4831
has_qi = false;
// MPAS module_mp_thompson.F:4832
has_qs = false;
// MPAS module_mp_thompson.F:4834
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4835
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,p1d[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,t1d[0+tc_sub(k,(kts))*(1)]),(tc_add(qv1d[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:4836
// Strict GNU MAX keeps the lower bound when this product is NaN;
// radius-native-nan-before.json witnesses the incorrect upper-radius cap.
rc[0+tc_sub(k,(kts))*(1)] = (tc_mul(qc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) > r1) ? tc_mul(qc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) : r1;
// MPAS module_mp_thompson.F:4837
nc[0+tc_sub(k,(kts))*(1)] = tc_max(2.f,tc_min(tc_mul(nc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]),nt_c_max));
// MPAS module_mp_thompson.F:4838
if((!is_aerosol_aware)) { nc[0+tc_sub(k,(kts))*(1)] = nt_c; }
// MPAS module_mp_thompson.F:4839
if(((rc[0+tc_sub(k,(kts))*(1)]>r1)&&(nc[0+tc_sub(k,(kts))*(1)]>r2))) { has_qc = true; }
// MPAS module_mp_thompson.F:4840
ri[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_mul(qi1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:4841
ni[0+tc_sub(k,(kts))*(1)] = (tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) > r2) ? tc_mul(ni1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) : r2;
// MPAS module_mp_thompson.F:4842
if(((ri[0+tc_sub(k,(kts))*(1)]>r1)&&(ni[0+tc_sub(k,(kts))*(1)]>r2))) { has_qi = true; }
// MPAS module_mp_thompson.F:4843
rs[0+tc_sub(k,(kts))*(1)] = (tc_mul(qs1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) > r1) ? tc_mul(qs1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]) : r1;
// MPAS module_mp_thompson.F:4844
if((rs[0+tc_sub(k,(kts))*(1)]>r1)) { has_qs = true; }
// MPAS module_mp_thompson.F:4845
}
// MPAS module_mp_thompson.F:4847
if(has_qc) {
// MPAS module_mp_thompson.F:4848
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4849
re_qc1d[0+tc_sub(k,(kts))*(1)] = 2.49e-6f;
// MPAS module_mp_thompson.F:4850
if(((rc[0+tc_sub(k,(kts))*(1)]<=r1)||(nc[0+tc_sub(k,(kts))*(1)]<=r2))) { continue; }
// MPAS module_mp_thompson.F:4851
if((nc[0+tc_sub(k,(kts))*(1)]<100)) {
// MPAS module_mp_thompson.F:4852
inu_c = 15;
// MPAS module_mp_thompson.F:4853
} else if((nc[0+tc_sub(k,(kts))*(1)]>1.e10f)) {
// MPAS module_mp_thompson.F:4854
inu_c = 2;
// MPAS module_mp_thompson.F:4855
} else {
// MPAS module_mp_thompson.F:4856
inu_c = tc_min(15,tc_add(((int)round(tc_div(1000.e6f,nc[0+tc_sub(k,(kts))*(1)]))),2));
// MPAS module_mp_thompson.F:4857
}
// MPAS module_mp_thompson.F:4858
lamc = tc_pow((tc_div(tc_mul(tc_mul(nc[0+tc_sub(k,(kts))*(1)],am_r),g_ratio[0+tc_sub(inu_c,(1))*(1)]),rc[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:4859
re_qc1d[0+tc_sub(k,(kts))*(1)] = tc_max(2.51e-6f,tc_min(tc_sngl(tc_div(tc_mul(0.5e0,((double)(tc_add(3.f,inu_c)))),lamc)),50.e-6f));
// MPAS module_mp_thompson.F:4860
}
// MPAS module_mp_thompson.F:4861
}
// MPAS module_mp_thompson.F:4863
if(has_qi) {
// MPAS module_mp_thompson.F:4864
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4865
re_qi1d[0+tc_sub(k,(kts))*(1)] = 2.49e-6f;
// MPAS module_mp_thompson.F:4866
if(((ri[0+tc_sub(k,(kts))*(1)]<=r1)||(ni[0+tc_sub(k,(kts))*(1)]<=r2))) { continue; }
// MPAS module_mp_thompson.F:4867
lami = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_i,cig[0+tc_sub(2,(1))*(1)]),oig1),ni[0+tc_sub(k,(kts))*(1)]),ri[0+tc_sub(k,(kts))*(1)])),obmi);
// MPAS module_mp_thompson.F:4868
re_qi1d[0+tc_sub(k,(kts))*(1)] = tc_max(2.51e-6f,tc_min(tc_sngl(tc_div(tc_mul(0.5e0,((double)(tc_add(3.f,mu_i)))),lami)),125.e-6f));
// MPAS module_mp_thompson.F:4869
}
// MPAS module_mp_thompson.F:4870
}
// MPAS module_mp_thompson.F:4872
if(has_qs) {
// MPAS module_mp_thompson.F:4873
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4874
re_qs1d[0+tc_sub(k,(kts))*(1)] = 4.99e-6f;
// MPAS module_mp_thompson.F:4875
if((rs[0+tc_sub(k,(kts))*(1)]<=r1)) { continue; }
// MPAS module_mp_thompson.F:4876
tc0 = tc_min((-0.1f),tc_sub(t1d[0+tc_sub(k,(kts))*(1)],273.15f));
// MPAS module_mp_thompson.F:4877
smob = tc_mul(rs[0+tc_sub(k,(kts))*(1)],oams);
// MPAS module_mp_thompson.F:4881
if(((bm_s>(tc_sub(2.0f,1.e-3f)))&&(bm_s<(tc_add(2.0f,1.e-3f))))) {
// MPAS module_mp_thompson.F:4882
smo2 = smob;
// MPAS module_mp_thompson.F:4883
} else {
// MPAS module_mp_thompson.F:4884
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:4889
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:4890
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:4895
smo2 = tc_pow((tc_div(smob,a_)),(tc_div(1.f,b_)));
// MPAS module_mp_thompson.F:4896
}
// MPAS module_mp_thompson.F:4898
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:4903
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:4904
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:4908
smoc = tc_mul(a_,tc_pow(smo2,b_));
// MPAS module_mp_thompson.F:4909
re_qs1d[0+tc_sub(k,(kts))*(1)] = tc_max(5.01e-6f,tc_min(tc_mul(0.5f,(tc_div(smoc,smob))),999.e-6f));
// MPAS module_mp_thompson.F:4910
}
// MPAS module_mp_thompson.F:4911
}
}
// MPAS module_mp_thompson.F:4924 calc_refl10cm
__device__ __forceinline__ void calc_refl10cm(float* qv1d,float* qc1d,float* qr1d,float* nr1d,float* qs1d,float* qg1d,float* t1d,float* p1d,float* dbz,int kts,int kte,int ii,int jj) {
float temp[THOMPSON_COLUMN_KMAX];
float pres[THOMPSON_COLUMN_KMAX];
float qv[THOMPSON_COLUMN_KMAX];
float rho[THOMPSON_COLUMN_KMAX];
float rhof[THOMPSON_COLUMN_KMAX];
float rc[THOMPSON_COLUMN_KMAX];
float rr[THOMPSON_COLUMN_KMAX];
float nr[THOMPSON_COLUMN_KMAX];
float rs[THOMPSON_COLUMN_KMAX];
float rg[THOMPSON_COLUMN_KMAX];
double ilamr[THOMPSON_COLUMN_KMAX];
double ilamg[THOMPSON_COLUMN_KMAX];
double n0_r[THOMPSON_COLUMN_KMAX];
double n0_g[THOMPSON_COLUMN_KMAX];
float mvd_r[THOMPSON_COLUMN_KMAX];
float smob[THOMPSON_COLUMN_KMAX];
float smo2[THOMPSON_COLUMN_KMAX];
float smoc[THOMPSON_COLUMN_KMAX];
float smoz[THOMPSON_COLUMN_KMAX];
float om3;
float m0;
float mrat;
float slam1;
float slam2;
float xds;
float ils1;
float ils2;
float t1_vts;
float t2_vts;
float t3_vts;
float t4_vts;
float vtr_dbz_wt;
float vts_dbz_wt;
float vtg_dbz_wt;
float ze_rain[THOMPSON_COLUMN_KMAX];
float ze_snow[THOMPSON_COLUMN_KMAX];
float ze_graupel[THOMPSON_COLUMN_KMAX];
double n0_exp;
double n0_min;
double lam_exp;
double lamr;
double lamg;
float a_;
float b_;
float loga_;
float tc0;
double fmelt_s;
double fmelt_g;
int i;
int k;
int k_0;
int kbot;
int n;
bool melti;
bool l_qr[THOMPSON_COLUMN_KMAX];
bool l_qs[THOMPSON_COLUMN_KMAX];
bool l_qg[THOMPSON_COLUMN_KMAX];
double cback;
double x;
double eta;
double f_d;
float xslw1;
float ygra1;
float zans1;
// MPAS module_mp_thompson.F:4962
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4963
dbz[0+tc_sub(k,(kts))*(1)] = (-35.0f);
// MPAS module_mp_thompson.F:4964
}
// MPAS module_mp_thompson.F:4969
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:4970
temp[0+tc_sub(k,(kts))*(1)] = t1d[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:4971
qv[0+tc_sub(k,(kts))*(1)] = tc_max(1.e-10f,qv1d[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:4972
pres[0+tc_sub(k,(kts))*(1)] = p1d[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:4973
rho[0+tc_sub(k,(kts))*(1)] = tc_div(tc_mul(0.622f,pres[0+tc_sub(k,(kts))*(1)]),(tc_mul(tc_mul(r,temp[0+tc_sub(k,(kts))*(1)]),(tc_add(qv[0+tc_sub(k,(kts))*(1)],0.622f)))));
// MPAS module_mp_thompson.F:4974
rhof[0+tc_sub(k,(kts))*(1)] = tc_sqrt(tc_div(rho_not,rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:4975
rc[0+tc_sub(k,(kts))*(1)] = tc_max(r1,tc_mul(qc1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:4976
if((qr1d[0+tc_sub(k,(kts))*(1)]>r1)) {
// MPAS module_mp_thompson.F:4977
rr[0+tc_sub(k,(kts))*(1)] = tc_mul(qr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:4978
nr[0+tc_sub(k,(kts))*(1)] = tc_max(r2,tc_mul(nr1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:4979
lamr = tc_pow((tc_div(tc_mul(tc_mul(tc_mul(am_r,crg[0+tc_sub(3,(1))*(1)]),org2),nr[0+tc_sub(k,(kts))*(1)]),rr[0+tc_sub(k,(kts))*(1)])),obmr);
// MPAS module_mp_thompson.F:4980
ilamr[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamr);
// MPAS module_mp_thompson.F:4981
n0_r[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(nr[0+tc_sub(k,(kts))*(1)],org2),tc_pow(lamr,cre[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:4982
mvd_r[0+tc_sub(k,(kts))*(1)] = tc_mul((tc_add(tc_add(3.0f,mu_r),0.672f)),ilamr[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:4983
l_qr[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:4984
} else {
// MPAS module_mp_thompson.F:4985
rr[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:4986
nr[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:4987
mvd_r[0+tc_sub(k,(kts))*(1)] = 50.e-6f;
// MPAS module_mp_thompson.F:4988
l_qr[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:4989
}
// MPAS module_mp_thompson.F:4990
if((qs1d[0+tc_sub(k,(kts))*(1)]>r2)) {
// MPAS module_mp_thompson.F:4991
rs[0+tc_sub(k,(kts))*(1)] = tc_mul(qs1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:4992
l_qs[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:4993
} else {
// MPAS module_mp_thompson.F:4994
rs[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:4995
l_qs[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:4996
}
// MPAS module_mp_thompson.F:4997
if((qg1d[0+tc_sub(k,(kts))*(1)]>r2)) {
// MPAS module_mp_thompson.F:4998
rg[0+tc_sub(k,(kts))*(1)] = tc_mul(qg1d[0+tc_sub(k,(kts))*(1)],rho[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:4999
l_qg[0+tc_sub(k,(kts))*(1)] = true;
// MPAS module_mp_thompson.F:5000
} else {
// MPAS module_mp_thompson.F:5001
rg[0+tc_sub(k,(kts))*(1)] = r1;
// MPAS module_mp_thompson.F:5002
l_qg[0+tc_sub(k,(kts))*(1)] = false;
// MPAS module_mp_thompson.F:5003
}
// MPAS module_mp_thompson.F:5004
}
// MPAS module_mp_thompson.F:5009
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:5010
smo2[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:5011
smob[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:5012
smoc[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:5013
smoz[0+tc_sub(k,(kts))*(1)] = 0.f;
// MPAS module_mp_thompson.F:5014
}
// MPAS module_mp_thompson.F:5015
if(tc_any(l_qs,kts,kte)) {
// MPAS module_mp_thompson.F:5016
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:5017
if((!l_qs[0+tc_sub(k,(kts))*(1)])) { continue; }
// MPAS module_mp_thompson.F:5018
tc0 = tc_min((-0.1f),tc_sub(temp[0+tc_sub(k,(kts))*(1)],273.15f));
// MPAS module_mp_thompson.F:5019
smob[0+tc_sub(k,(kts))*(1)] = tc_mul(rs[0+tc_sub(k,(kts))*(1)],oams);
// MPAS module_mp_thompson.F:5023
if(((bm_s>(tc_sub(2.0f,1.e-3f)))&&(bm_s<(tc_add(2.0f,1.e-3f))))) {
// MPAS module_mp_thompson.F:5024
smo2[0+tc_sub(k,(kts))*(1)] = smob[0+tc_sub(k,(kts))*(1)];
// MPAS module_mp_thompson.F:5025
} else {
// MPAS module_mp_thompson.F:5026
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:5031
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:5032
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],bm_s)),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),bm_s)),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),bm_s),bm_s)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],bm_s),bm_s),bm_s));
// MPAS module_mp_thompson.F:5037
smo2[0+tc_sub(k,(kts))*(1)] = tc_pow((tc_div(smob[0+tc_sub(k,(kts))*(1)],a_)),(tc_div(1.f,b_)));
// MPAS module_mp_thompson.F:5038
}
// MPAS module_mp_thompson.F:5041
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:5046
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:5047
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]),cse[0+tc_sub(1,(1))*(1)]));
// MPAS module_mp_thompson.F:5051
smoc[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:5054
loga_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sa[0+tc_sub(1,(1))*(1)],tc_mul(sa[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sa[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(sa[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sa[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sa[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)]));
// MPAS module_mp_thompson.F:5059
a_ = tc_pow(10.0f,loga_);
// MPAS module_mp_thompson.F:5060
b_ = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(sb[0+tc_sub(1,(1))*(1)],tc_mul(sb[0+tc_sub(2,(1))*(1)],tc0)),tc_mul(sb[0+tc_sub(3,(1))*(1)],cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(4,(1))*(1)],tc0),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(sb[0+tc_sub(5,(1))*(1)],tc0),tc0)),tc_mul(tc_mul(sb[0+tc_sub(6,(1))*(1)],cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(7,(1))*(1)],tc0),tc0),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(8,(1))*(1)],tc0),cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)])),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(9,(1))*(1)],tc0),tc0),tc0)),tc_mul(tc_mul(tc_mul(sb[0+tc_sub(10,(1))*(1)],cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)]),cse[0+tc_sub(3,(1))*(1)]));
// MPAS module_mp_thompson.F:5064
smoz[0+tc_sub(k,(kts))*(1)] = tc_mul(a_,tc_pow(smo2[0+tc_sub(k,(kts))*(1)],b_));
// MPAS module_mp_thompson.F:5065
}
// MPAS module_mp_thompson.F:5066
}
// MPAS module_mp_thompson.F:5072
if(tc_any(l_qg,kts,kte)) {
// MPAS module_mp_thompson.F:5073
n0_min = gonv_max;
// MPAS module_mp_thompson.F:5074
k_0 = kts;
// MPAS module_mp_thompson.F:5075
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:5076
if((temp[0+tc_sub(k,(kts))*(1)]>=270.65f)) { k_0 = tc_max(k_0,k); }
// MPAS module_mp_thompson.F:5077
}
// MPAS module_mp_thompson.F:5078
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:5079
if((((k>k_0)&&l_qr[0+tc_sub(k,(kts))*(1)])&&(mvd_r[0+tc_sub(k,(kts))*(1)]>100.e-6f))) {
// MPAS module_mp_thompson.F:5080
xslw1 = tc_add(4.01f,tc_log10(mvd_r[0+tc_sub(k,(kts))*(1)]));
// MPAS module_mp_thompson.F:5081
} else {
// MPAS module_mp_thompson.F:5082
xslw1 = 0.01f;
// MPAS module_mp_thompson.F:5083
}
// MPAS module_mp_thompson.F:5084
ygra1 = tc_add(4.31f,tc_log10(tc_max(5.e-5f,rg[0+tc_sub(k,(kts))*(1)])));
// MPAS module_mp_thompson.F:5085
zans1 = tc_add(3.1f,(tc_div(100.f,(tc_add(tc_add(tc_div(tc_mul(tc_mul(300.f,xslw1),ygra1),(tc_add(tc_add(tc_div(10.f,xslw1),1.f),tc_mul(0.25f,ygra1)))),30.f),tc_mul(10.f,ygra1))))));
// MPAS module_mp_thompson.F:5086
n0_exp = tc_pow(10.f,(zans1));
// MPAS module_mp_thompson.F:5087
n0_exp = tc_max(((double)(gonv_min)),tc_min(n0_exp,((double)(gonv_max))));
// MPAS module_mp_thompson.F:5088
n0_min = tc_min(n0_exp,n0_min);
// MPAS module_mp_thompson.F:5089
n0_exp = n0_min;
// MPAS module_mp_thompson.F:5090
lam_exp = tc_pow((tc_div(tc_mul(tc_mul(n0_exp,am_g),cgg[0+tc_sub(1,(1))*(1)]),rg[0+tc_sub(k,(kts))*(1)])),oge1);
// MPAS module_mp_thompson.F:5091
lamg = tc_mul(lam_exp,tc_pow((tc_mul(tc_mul(cgg[0+tc_sub(3,(1))*(1)],ogg2),ogg1)),obmg));
// MPAS module_mp_thompson.F:5092
ilamg[0+tc_sub(k,(kts))*(1)] = tc_div(1.f,lamg);
// MPAS module_mp_thompson.F:5093
n0_g[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_div(n0_exp,(tc_mul(cgg[0+tc_sub(2,(1))*(1)],lam_exp))),tc_pow(lamg,cge[0+tc_sub(2,(1))*(1)]));
// MPAS module_mp_thompson.F:5094
}
// MPAS module_mp_thompson.F:5095
}
// MPAS module_mp_thompson.F:5100
melti = false;
// MPAS module_mp_thompson.F:5101
k_0 = kts;
// MPAS module_mp_thompson.F:5102
for(k=tc_sub(kte,1); k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:5103
if(((((temp[0+tc_sub(k,(kts))*(1)]>273.15f))&&l_qr[0+tc_sub(k,(kts))*(1)])&&((l_qs[0+tc_sub(tc_add(k,1),(kts))*(1)]||l_qg[0+tc_sub(tc_add(k,1),(kts))*(1)])))) {
// MPAS module_mp_thompson.F:5105
k_0 = tc_max(tc_add(k,1),k_0);
// MPAS module_mp_thompson.F:5106
melti = true;
// MPAS module_mp_thompson.F:5107
goto label_195;
// MPAS module_mp_thompson.F:5108
}
// MPAS module_mp_thompson.F:5109
}
// MPAS module_mp_thompson.F:5110
label_195: ;

// MPAS module_mp_thompson.F:5118
for(k=kts; k<=kte; k+=1) {
// MPAS module_mp_thompson.F:5119
ze_rain[0+tc_sub(k,(kts))*(1)] = 1.e-22f;
// MPAS module_mp_thompson.F:5120
ze_snow[0+tc_sub(k,(kts))*(1)] = 1.e-22f;
// MPAS module_mp_thompson.F:5121
ze_graupel[0+tc_sub(k,(kts))*(1)] = 1.e-22f;
// MPAS module_mp_thompson.F:5122
if(l_qr[0+tc_sub(k,(kts))*(1)]) { ze_rain[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(n0_r[0+tc_sub(k,(kts))*(1)],crg[0+tc_sub(4,(1))*(1)]),tc_pow(ilamr[0+tc_sub(k,(kts))*(1)],cre[0+tc_sub(4,(1))*(1)])); }
// MPAS module_mp_thompson.F:5123
if(l_qs[0+tc_sub(k,(kts))*(1)]) { ze_snow[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul((tc_div(0.176f,0.93f)),(tc_div(6.0f,pi))),(tc_div(6.0f,pi))),(tc_div(am_s,900.0f))),(tc_div(am_s,900.0f))),smoz[0+tc_sub(k,(kts))*(1)]); }
// MPAS module_mp_thompson.F:5125
if(l_qg[0+tc_sub(k,(kts))*(1)]) { ze_graupel[0+tc_sub(k,(kts))*(1)] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul((tc_div(0.176f,0.93f)),(tc_div(6.0f,pi))),(tc_div(6.0f,pi))),(tc_div(am_g,900.0f))),(tc_div(am_g,900.0f))),n0_g[0+tc_sub(k,(kts))*(1)]),cgg[0+tc_sub(4,(1))*(1)]),tc_pow(ilamg[0+tc_sub(k,(kts))*(1)],cge[0+tc_sub(4,(1))*(1)])); }
// MPAS module_mp_thompson.F:5128
}
// MPAS module_mp_thompson.F:5138
if((((!iiwarm)&&melti)&&(k_0>=2))) {
// MPAS module_mp_thompson.F:5139
for(k=tc_sub(k_0,1); k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:5142
if((l_qs[0+tc_sub(k,(kts))*(1)]&&l_qs[0+tc_sub(k_0,(kts))*(1)])) {
// MPAS module_mp_thompson.F:5143
fmelt_s = tc_max(0.05e0,tc_min(tc_sub(1.0e0,tc_div(rs[0+tc_sub(k,(kts))*(1)],rs[0+tc_sub(k_0,(kts))*(1)])),0.99e0));
// MPAS module_mp_thompson.F:5144
eta = 0.e0;
// MPAS module_mp_thompson.F:5145
om3 = tc_div(1.f,smoc[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:5146
m0 = (tc_mul(smob[0+tc_sub(k,(kts))*(1)],om3));
// MPAS module_mp_thompson.F:5147
mrat = tc_mul(tc_mul(tc_mul(smob[0+tc_sub(k,(kts))*(1)],m0),m0),m0);
// MPAS module_mp_thompson.F:5148
slam1 = tc_mul(m0,lam0);
// MPAS module_mp_thompson.F:5149
slam2 = tc_mul(m0,lam1);
// MPAS module_mp_thompson.F:5150
for(n=1; n<=nrbins; n+=1) {
// MPAS module_mp_thompson.F:5151
x = tc_mul(am_s,tc_pow(xxds[0+tc_sub(n,(1))*(1)],bm_s));
// MPAS module_mp_thompson.F:5152
rayleigh_soak_wetgraupel(x,((double)(ocms)),((double)(obms)),fmelt_s,melt_outside_s,m_w_0,m_i_0,lamda_radar,cback,mixingrulestring_s,matrixstring_s,inclusionstring_s,hoststring_s,hostmatrixstring_s,hostinclusionstring_s);
// MPAS module_mp_thompson.F:5157
f_d = tc_mul(mrat,(tc_add(tc_mul(kap0,tc_exp(tc_mul((-slam1),xxds[0+tc_sub(n,(1))*(1)]))),tc_mul(tc_mul(kap1,tc_pow((tc_mul(m0,xxds[0+tc_sub(n,(1))*(1)])),mu_s)),tc_exp(tc_mul((-slam2),xxds[0+tc_sub(n,(1))*(1)]))))));
// MPAS module_mp_thompson.F:5159
eta = tc_add(eta,tc_mul(tc_mul(tc_mul(f_d,cback),simpson[0+tc_sub(n,(1))*(1)]),xdts[0+tc_sub(n,(1))*(1)]));
// MPAS module_mp_thompson.F:5160
}
// MPAS module_mp_thompson.F:5161
ze_snow[0+tc_sub(k,(kts))*(1)] = tc_sngl(tc_mul(tc_div(lamda4,(tc_mul(pi5,k_w))),eta));
// MPAS module_mp_thompson.F:5162
}
// MPAS module_mp_thompson.F:5166
if((l_qg[0+tc_sub(k,(kts))*(1)]&&l_qg[0+tc_sub(k_0,(kts))*(1)])) {
// MPAS module_mp_thompson.F:5167
fmelt_g = tc_max(0.05e0,tc_min(tc_sub(1.0e0,tc_div(rg[0+tc_sub(k,(kts))*(1)],rg[0+tc_sub(k_0,(kts))*(1)])),0.99e0));
// MPAS module_mp_thompson.F:5168
eta = 0.e0;
// MPAS module_mp_thompson.F:5169
lamg = tc_div(1.f,ilamg[0+tc_sub(k,(kts))*(1)]);
// MPAS module_mp_thompson.F:5170
for(n=1; n<=nrbins; n+=1) {
// MPAS module_mp_thompson.F:5171
x = tc_mul(am_g,tc_pow(xxdg[0+tc_sub(n,(1))*(1)],bm_g));
// MPAS module_mp_thompson.F:5172
rayleigh_soak_wetgraupel(x,((double)(ocmg)),((double)(obmg)),fmelt_g,melt_outside_g,m_w_0,m_i_0,lamda_radar,cback,mixingrulestring_g,matrixstring_g,inclusionstring_g,hoststring_g,hostmatrixstring_g,hostinclusionstring_g);
// MPAS module_mp_thompson.F:5177
f_d = tc_mul(tc_mul(n0_g[0+tc_sub(k,(kts))*(1)],tc_pow(xxdg[0+tc_sub(n,(1))*(1)],mu_g)),tc_exp(tc_mul((-lamg),xxdg[0+tc_sub(n,(1))*(1)])));
// MPAS module_mp_thompson.F:5178
eta = tc_add(eta,tc_mul(tc_mul(tc_mul(f_d,cback),simpson[0+tc_sub(n,(1))*(1)]),xdtg[0+tc_sub(n,(1))*(1)]));
// MPAS module_mp_thompson.F:5179
}
// MPAS module_mp_thompson.F:5180
ze_graupel[0+tc_sub(k,(kts))*(1)] = tc_sngl(tc_mul(tc_div(lamda4,(tc_mul(pi5,k_w))),eta));
// MPAS module_mp_thompson.F:5181
}
// MPAS module_mp_thompson.F:5183
}
// MPAS module_mp_thompson.F:5184
}
// MPAS module_mp_thompson.F:5186
for(k=kte; k>=kts; k+=(-1)) {
// MPAS module_mp_thompson.F:5187
dbz[0+tc_sub(k,(kts))*(1)] = tc_mul(10.f,tc_log10(tc_mul((tc_add(tc_add(ze_rain[0+tc_sub(k,(kts))*(1)],ze_snow[0+tc_sub(k,(kts))*(1)]),ze_graupel[0+tc_sub(k,(kts))*(1)])),1.e18)));
// MPAS module_mp_thompson.F:5188
}
}
};

'''

TABLE_SHAPES = {
    **dict.fromkeys(("tcg_racg", "tmr_racg", "tcr_gacr", "tmg_gacr", "tnr_racg", "tnr_gacr"), (28,28,37,37)),
    **dict.fromkeys(("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2"), (28,9,37,37)),
    **dict.fromkeys(("tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz"), (37,37,45,55)),
    **dict.fromkeys(("tpi_qcfz", "tni_qcfz"), (37,100,45,55)),
    **dict.fromkeys(("tpi_ide", "tps_iaus", "tni_iaus"), (64,55)),
    **dict.fromkeys(("t_efrw", "t_efsw"), (100,100)),
    **dict.fromkeys(("tpc_wev", "tnc_wev"), (100,37,100)),
    "tnr_rev": (100,37,37),  # MPAS module_mp_thompson.F:492-494.
    "tnccn_act": (7,9,7,5,4),
}
TABLE_ARGUMENTS = tuple(TABLE_SHAPES)
FIELD_ARGUMENTS = ("qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz", "rainprod", "evapprod", "re_cloud", "re_ice", "re_snow")
SURFACE_ARGUMENTS = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr", "ntc", "muc")

# MPAS module_mp_thompson.F:1001-1387 mp_gt_driver.
_DRIVER = r'''
// MPAS module_mp_thompson.F:1001-1387; device driver boundary.
extern "C" __global__ void mpas_thompson_step(
    float* qv,float* qc,float* qi,float* qr,float* qs,float* qg,
    float* ni,float* nr,float* th,const float* pii,const float* p,
    const float* w,const float* dz,float* rainprod,float* evapprod,
    float* re_cloud,float* re_ice,float* re_snow,
    float* rainnc,float* rainncv,float* snownc,float* snowncv,
    float* graupelnc,float* graupelncv,float* sr,
    const float* ntc,const float* muc,
    int nz,int columns,float dt,int has_reqc,int has_reqi,int has_reqs,
    TABLE_DECLARATIONS)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    ThompsonColumn m;
    TABLE_ASSIGNMENTS
    m.nt_c=ntc[col];m.mu_c=muc[col];
    float qv1d[THOMPSON_COLUMN_KMAX],qc1d[THOMPSON_COLUMN_KMAX],qi1d[THOMPSON_COLUMN_KMAX],qr1d[THOMPSON_COLUMN_KMAX],qs1d[THOMPSON_COLUMN_KMAX],qg1d[THOMPSON_COLUMN_KMAX];
    float ni1d[THOMPSON_COLUMN_KMAX],nr1d[THOMPSON_COLUMN_KMAX],nc1d[THOMPSON_COLUMN_KMAX],nwfa1d[THOMPSON_COLUMN_KMAX],nifa1d[THOMPSON_COLUMN_KMAX];
    float t1d[THOMPSON_COLUMN_KMAX],p1d[THOMPSON_COLUMN_KMAX],w1d[THOMPSON_COLUMN_KMAX],dz1d[THOMPSON_COLUMN_KMAX];
    float rainprod1d[THOMPSON_COLUMN_KMAX],evapprod1d[THOMPSON_COLUMN_KMAX];
    float re_qc1d[THOMPSON_COLUMN_KMAX],re_qi1d[THOMPSON_COLUMN_KMAX],re_qs1d[THOMPSON_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        qv1d[k]=qv[idx];qc1d[k]=qc[idx];qi1d[k]=qi[idx];qr1d[k]=qr[idx];qs1d[k]=qs[idx];qg1d[k]=qg[idx];
        ni1d[k]=ni[idx];nr1d[k]=nr[idx];
        t1d[k]=tc_mul(th[idx],pii[idx]);p1d[k]=p[idx];w1d[k]=w[idx];dz1d[k]=dz[idx];
        float rho=tc_div(tc_mul(0.622f,p1d[k]),(tc_mul(tc_mul(m.r,t1d[k]),(tc_add(qv1d[k],0.622f)))));
        nc1d[k]=tc_div(m.nt_c,rho);nwfa1d[k]=tc_div(11.1e6f,rho);nifa1d[k]=tc_div((tc_mul(m.nain1,0.01f)),rho);
        rainprod1d[k]=0.0f;evapprod1d[k]=0.0f;
    }
    float rain=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;
    m.mp_thompson(qv1d,qc1d,qi1d,qr1d,qs1d,qg1d,ni1d,nr1d,nc1d,nwfa1d,nifa1d,t1d,p1d,w1d,dz1d,
                  rain,snow,graupel,ice,rainprod1d,evapprod1d,1,nz,dt,1,1);
    rainncv[col]=tc_add(tc_add(tc_add(rain,snow),graupel),ice);
    rainnc[col]=tc_add(tc_add(tc_add(tc_add(rainnc[col],rain),snow),graupel),ice);
    snowncv[col]=tc_add(snow,ice);snownc[col]=tc_add(tc_add(snownc[col],snow),ice);
    graupelncv[col]=graupel;graupelnc[col]=tc_add(graupelnc[col],graupel);
    sr[col]=tc_div((tc_add(tc_add(snow,graupel),ice)),(tc_add(rainncv[col],1.0e-12f)));
    // Driver copies state and performs its negative-vapor cleanup before radii.
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        qv[idx]=qv1d[k];qc[idx]=qc1d[k];qi[idx]=qi1d[k];qr[idx]=qr1d[k];qs[idx]=qs1d[k];qg[idx]=qg1d[k];
        ni[idx]=ni1d[k];nr[idx]=nr1d[k];th[idx]=tc_div(t1d[k],pii[idx]);
        rainprod[idx]=rainprod1d[k];evapprod[idx]=evapprod1d[k];
        // MPAS :1321-1333: other negative species only emit warnings.
        // Neighbor below is already copied; neighbor above retains incoming qv.
        if(qv1d[k]<0.0f) {
            if(k+1 < nz-2 && k+1 > 2)
                qv[idx]=tc_max(1.0e-7f,tc_mul(0.5f,(tc_add(qv[idx-columns],qv[idx+columns]))));
            else qv[idx]=1.0e-7f;
        }
        re_qc1d[k]=2.49e-6f;re_qi1d[k]=4.99e-6f;re_qs1d[k]=9.99e-6f;
    }
    if(has_reqc && has_reqi && has_reqs) {
        m.calc_effectrad(t1d,p1d,qv1d,qc1d,nc1d,qi1d,ni1d,qs1d,re_qc1d,re_qi1d,re_qs1d,1,nz);
        for(int k=0;k<nz;k++) {
            int idx=k*columns+col;
            re_cloud[idx]=tc_max(2.49e-6f,tc_min(re_qc1d[k],50.0e-6f));
            re_ice[idx]=tc_max(4.99e-6f,tc_min(re_qi1d[k],125.0e-6f));
            re_snow[idx]=tc_max(9.99e-6f,tc_min(re_qs1d[k],999.0e-6f));
        }
    }
}
'''

# MPAS mpas_atmphys_driver_microphysics.F:738-792 compute_radar_reflectivity.
_RADAR_DRIVER = r'''
// MPAS mpas_atmphys_driver_microphysics.F:738-792; device driver boundary.
extern "C" __global__ void mpas_thompson_radar(
    const float* qv,const float* qc,const float* qr,const float* nr,
    const float* qs,const float* qg,const float* th,const float* pii,
    const float* p,const float* z,const float* dz,float* refl,
    float* refl_max,float* refl_1km,float* refl_1km_max,int nz,int columns)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    ThompsonColumn m;
    float qv1[THOMPSON_COLUMN_KMAX],qc1[THOMPSON_COLUMN_KMAX],qr1[THOMPSON_COLUMN_KMAX];
    float nr1[THOMPSON_COLUMN_KMAX],qs1[THOMPSON_COLUMN_KMAX],qg1[THOMPSON_COLUMN_KMAX];
    float t1[THOMPSON_COLUMN_KMAX],p1[THOMPSON_COLUMN_KMAX],dbz[THOMPSON_COLUMN_KMAX],zp[THOMPSON_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        qv1[k]=qv[idx];qc1[k]=qc[idx];qr1[k]=qr[idx];nr1[k]=nr[idx];
        qs1[k]=qs[idx];qg1[k]=qg[idx];t1[k]=tc_mul(th[idx],pii[idx]);p1[k]=p[idx];
        dbz[k]=-35.0f;zp[k]=tc_add(tc_sub(z[idx],z[col]),tc_mul(0.5f,dz[idx]));
    }
    m.calc_refl10cm(qv1,qc1,qr1,nr1,qs1,qg1,t1,p1,dbz,1,nz,col+1,1);
    int kp=0;float peak=-35.0f;
    for(int k=0;k<nz;k++) {
        dbz[k]=tc_max(-35.0f,dbz[k]);refl[k*columns+col]=dbz[k];
        peak=tc_max(peak,dbz[k]);if(zp[k]<1000.0f)kp=k;
    }
    refl_max[col]=peak;
    // MPAS assumes the column brackets 1 km, including kp+1.
    float w1=tc_div((tc_sub(zp[kp+1],1000.0f)),(tc_sub(zp[kp+1],zp[kp])));float w2=tc_sub(1.0f,w1);
    refl_1km[col]=tc_add(tc_mul(w1,dbz[kp]),tc_mul(w2,dbz[kp+1]));
    refl_1km_max[col]=tc_max(refl_1km_max[col],refl_1km[col]);
}
'''

MODEL_ARGUMENTS = ("qv","qc","qr","qi","qs","qg","ni","nr","theta_m","rho_zz","w")
MESH_ARGUMENTS = ("zz","zgrid")
DIAG_ARGUMENTS = ("exner","pressure_base","pressure_p","rtheta_base","rtheta_p","exner_base","rt_diabatic_tend","dtheta_dt_mp")
TENDENCY_ARGUMENTS = ("rthmpten","rqvmpten","rqcmpten","rqrmpten","rqimpten","rqsmpten","rqgmpten","rnimpten","rnrmpten")
CONVERSION_CONSTANTS = ("R_d","R_v","P0","rcv","rvord","gravity","re_qc_bg","re_qi_bg","re_qs_bg")
_CONVERSION_ARRAYS = (tuple("m_"+n for n in MODEL_ARGUMENTS)+tuple("g_"+n for n in MESH_ARGUMENTS)
                      +tuple("d_"+n for n in DIAG_ARGUMENTS)+tuple("t_"+n for n in TENDENCY_ARGUMENTS)
                      +tuple("f_"+n for n in FIELD_ARGUMENTS)+("f_rho","f_z","d_surface_pressure","d_tend_sfc_pressure"))

# MPAS mpas_atmphys_interface.F:598-823,826-1138. MPAS constants are inputs
# from the seam, preserving the independent R_v/R_d and rvord expressions.
_CONVERSIONS = r'''
// MPAS mpas_atmphys_interface.F:598-823; device driver boundary.
extern "C" __global__ void mpas_thompson_from(float** arrays,int nz,int columns,
    CONSTANT_DECLARATIONS)
{
    ARRAY_ALIASES
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=nz*columns)return;
    f_qv[idx]=m_qv[idx];f_qc[idx]=m_qc[idx];f_qr[idx]=m_qr[idx];
    f_qi[idx]=m_qi[idx];f_qs[idx]=m_qs[idx];f_qg[idx]=m_qg[idx];
    f_ni[idx]=m_ni[idx];f_nr[idx]=m_nr[idx];
    f_rho[idx]=tc_mul(g_zz[idx],m_rho_zz[idx]);
    f_th[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,f_qv[idx])))));
    f_pii[idx]=d_exner[idx];f_p[idx]=tc_add(d_pressure_base[idx],d_pressure_p[idx]);
    f_z[idx]=g_zgrid[idx];f_dz[idx]=tc_sub(g_zgrid[idx+columns],g_zgrid[idx]);f_w[idx]=m_w[idx];
    t_rthmpten[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,m_qv[idx])))));
    t_rqvmpten[idx]=m_qv[idx];t_rqcmpten[idx]=m_qc[idx];t_rqrmpten[idx]=m_qr[idx];
    t_rqimpten[idx]=m_qi[idx];t_rqsmpten[idx]=m_qs[idx];t_rqgmpten[idx]=m_qg[idx];
    t_rnimpten[idx]=m_ni[idx];t_rnrmpten[idx]=m_nr[idx];
}
// MPAS mpas_atmphys_interface.F:826-1138; device driver boundary.
extern "C" __global__ void mpas_thompson_to(float** arrays,int nz,int columns,
    int top,float dt_dyn,CONSTANT_DECLARATIONS)
{
    ARRAY_ALIASES
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=columns)return;
    for(int k=0;k<top;k++) {
        int idx=k*columns+col;
        d_rt_diabatic_tend[idx]=m_theta_m[idx];
        d_dtheta_dt_mp[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(rvord,m_qv[idx]))));
        m_qv[idx]=f_qv[idx];m_qc[idx]=f_qc[idx];m_qr[idx]=f_qr[idx];
        m_theta_m[idx]=tc_mul((tc_add(1.0f,tc_mul(rvord,f_qv[idx]))),f_th[idx]);
        d_rt_diabatic_tend[idx]=tc_div((tc_sub(m_theta_m[idx],d_rt_diabatic_tend[idx])),dt_dyn);
        d_dtheta_dt_mp[idx]=tc_div((tc_sub(tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(rvord,m_qv[idx])))),d_dtheta_dt_mp[idx])),dt_dyn);
        d_rtheta_p[idx]=tc_sub(tc_mul(m_rho_zz[idx],m_theta_m[idx]),d_rtheta_base[idx]);
        d_exner[idx]=tc_pow(tc_mul(tc_mul(g_zz[idx],(tc_div(R_d,P0))),(tc_add(d_rtheta_p[idx],d_rtheta_base[idx]))),rcv);
        d_pressure_p[idx]=tc_mul(tc_mul(g_zz[idx],R_d),(tc_add(tc_mul(d_exner[idx],d_rtheta_p[idx]),tc_mul((tc_sub(d_exner[idx],d_exner_base[idx])),d_rtheta_base[idx]))));
    }
    for(int k=top;k<nz;k++) {
        int idx=k*columns+col;m_qc[idx]=0.0f;m_qr[idx]=0.0f;d_rt_diabatic_tend[idx]=0.0f;
    }
    float tem1=tc_sub(g_zgrid[columns+col],g_zgrid[col]);
    float tem2=tc_sub(g_zgrid[2*columns+col],g_zgrid[columns+col]);
    float rho1=tc_mul(tc_mul(m_rho_zz[col],g_zz[col]),(tc_add(1.0f,f_qv[col])));
    float rho2=tc_mul(tc_mul(m_rho_zz[columns+col],g_zz[columns+col]),(tc_add(1.0f,f_qv[columns+col])));
    d_tend_sfc_pressure[col]=d_surface_pressure[col];
    d_surface_pressure[col]=tc_mul((tc_sub(rho1,tc_div(tc_mul(tc_mul(0.5f,(tc_sub(rho2,rho1))),tem1),(tc_add(tem1,tem2))))),tc_mul(tc_mul(0.5f,gravity),(tc_sub(g_zgrid[columns+col],g_zgrid[col]))));
    d_surface_pressure[col]=tc_add(tc_add(d_surface_pressure[col],d_pressure_p[col]),d_pressure_base[col]);
    d_tend_sfc_pressure[col]=tc_div((tc_sub(d_surface_pressure[col],d_tend_sfc_pressure[col])),dt_dyn);
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        if(k<top) {
            m_qi[idx]=f_qi[idx];m_qs[idx]=f_qs[idx];m_qg[idx]=f_qg[idx];
            t_rthmpten[idx]=tc_div((tc_sub(tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,m_qv[idx]))))),t_rthmpten[idx])),dt_dyn);
            t_rqvmpten[idx]=tc_div((tc_sub(m_qv[idx],t_rqvmpten[idx])),dt_dyn);
            t_rqcmpten[idx]=tc_div((tc_sub(m_qc[idx],t_rqcmpten[idx])),dt_dyn);
            t_rqrmpten[idx]=tc_div((tc_sub(m_qr[idx],t_rqrmpten[idx])),dt_dyn);
            t_rqimpten[idx]=tc_div((tc_sub(m_qi[idx],t_rqimpten[idx])),dt_dyn);
            t_rqsmpten[idx]=tc_div((tc_sub(m_qs[idx],t_rqsmpten[idx])),dt_dyn);
            t_rqgmpten[idx]=tc_div((tc_sub(m_qg[idx],t_rqgmpten[idx])),dt_dyn);
        } else {
            m_qi[idx]=0.0f;m_qs[idx]=0.0f;m_qg[idx]=0.0f;
            f_rainprod[idx]=0.0f;f_evapprod[idx]=0.0f;
            f_re_cloud[idx]=re_qc_bg;f_re_ice[idx]=re_qi_bg;f_re_snow[idx]=re_qs_bg;
            t_rthmpten[idx]=0.0f;t_rqvmpten[idx]=0.0f;t_rqcmpten[idx]=0.0f;t_rqrmpten[idx]=0.0f;
            t_rqimpten[idx]=0.0f;t_rqsmpten[idx]=0.0f;t_rqgmpten[idx]=0.0f;
        }
        // MPAS updates number concentrations over kte, independently of top.
        m_ni[idx]=f_ni[idx];m_nr[idx]=f_nr[idx];
        t_rnimpten[idx]=tc_div((tc_sub(m_ni[idx],t_rnimpten[idx])),dt_dyn);
        t_rnrmpten[idx]=tc_div((tc_sub(m_nr[idx],t_rnrmpten[idx])),dt_dyn);
    }
}
'''

PRECIP_ARGUMENTS=("precipw","rainnc","rainncv","snownc","snowncv","graupelnc","graupelncv","sr")
# MPAS mpas_atmphys_driver_microphysics.F:503-653 precip_from_MPAS/to_MPAS.
_PRECIP = r'''
// MPAS mpas_atmphys_driver_microphysics.F:503-653; device driver boundary.
extern "C" __global__ void mpas_thompson_precip(
    const float* qv,const float* rho,const float* dz,float** work,float** diag,
    int* i_rainnc,int nz,int columns,int phase,int l_acrain,float bucket)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=columns)return;
    if(phase==0) {
        for(int n=0;n<7;n++)work[n][col]=0.0f;
        diag[2][col]=0.0f;diag[4][col]=0.0f;diag[6][col]=0.0f;diag[7][col]=0.0f;
        return;
    }
    diag[0][col]=0.0f;
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;float rho_a=tc_div(rho[idx],(tc_add(1.0f,qv[idx])));
        diag[0][col]=tc_add(diag[0][col],tc_mul(tc_mul(qv[idx],rho_a),dz[idx]));
    }
    diag[2][col]=work[0][col];diag[1][col]=tc_add(diag[1][col],diag[2][col]);
    if(l_acrain && bucket>0.0f && diag[1][col]>bucket) {
        i_rainnc[col]=tc_add(i_rainnc[col],1);diag[1][col]=tc_sub(diag[1][col],bucket);
    }
    diag[4][col]=work[2][col];diag[6][col]=work[4][col];
    diag[7][col]=tc_div((tc_add(work[2][col],work[4][col])),(tc_add(work[0][col],1.0e-12f)));
    diag[3][col]=tc_add(diag[3][col],diag[4][col]);diag[5][col]=tc_add(diag[5][col],diag[6][col]);
}
// MPAS mpas_atmphys_driver_microphysics.F:798-850 compute_relhum.
extern "C" __global__ void mpas_thompson_relhum(const float* qv,const float* th,
    const float* pii,const float* p,float* relhum,int count)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=count)return;
    ThompsonColumn m;
    float t=tc_mul(th[idx],pii[idx]);float tempc=tc_sub(t,273.16f);
    float qvs=m.rslf(p[idx],t);if(tempc<=0.0f)qvs=m.rsif(p[idx],t);
    relhum[idx]=tc_div(qv[idx],qvs);relhum[idx]=tc_mul(relhum[idx],100.0f);
}
'''


def _conversion_source():
    """Compose MPAS mpas_atmphys_interface.F:598-823,826-1138 array views."""
    aliases="\n".join(f"float* {name}=arrays[{i}];" for i,name in enumerate(_CONVERSION_ARRAYS))
    constants=",".join("float "+n for n in CONVERSION_CONSTANTS)
    return _CONVERSIONS.replace("ARRAY_ALIASES",aliases).replace("CONSTANT_DECLARATIONS",constants)


#: Correct mode (CORRECTIONS_V842): exact text, corrected text, expected count.  WB3:
#: mpas_atmphys_driver_microphysics.F:614-617 divides the dry rho_p by (1+qv) a second time.
CORRECT_MODE_SOURCE = {
    "WB3": (('float rho_a=tc_div(rho[idx],(tc_add(1.0f,qv[idx])));', 'float rho_a=rho[idx];', 1),),
}


#: Column workspace (memory lane, 2026-10-14). The driver sizes CUDA local memory
#: as the widest per-thread frame launched in the context times every thread the
#: card can hold (RTX 5090: 170 SMs x 1536 threads) and never shrinks it. The
#: per-level column arrays below made mpas_thompson_step's frame 43,440 bytes,
#: so the store was 10.6 GiB on every card at every mesh size, and the radar
#: kernel's 8 KiB frame alone would keep 2 GiB. Workspace mode stores those same
#: arrays in one pooled global buffer, one block per warp with lanes fastest (the
#: interleaving local memory already used, at a compile-time lane stride), sized
#: to one launch tile of columns and returned to the pool after the call. Only storage moves: every statement
#: and every operand order is the unchanged text. The default is on;
#: WOOF_HEX_THOMPSON_WORKSPACE=0 restores the per-thread arrays for A/B.
#: A launch tile is by default the columns the card runs at once for that kernel
#: (SMs x resident 32-thread blocks, from the occupancy API), so no column waits
#: for a later launch and the step costs no speed; meshes smaller than that run
#: in one launch. WOOF_HEX_THOMPSON_TILE_COLUMNS=N caps the tile at N columns,
#: trading launches for a smaller workspace.
#: The resident width alone priced the workspace at ~33 KB per column: 5.4 GB for a
#: 164k-column rank (a 30 km globe over 4 cards, or the whole 60 km globe), which ran a
#: 32 GB RTX 5090 out of memory at the first microphysics call (hex-global-fit,
#: 2026-10-27, g01.log). The tile is therefore also capped by a workspace byte budget,
#: WOOF_HEX_THOMPSON_WORKSPACE_MIB (default 1536). Columns are independent, so the
#: tile changes only how many launches a call takes, never a value.
WORKSPACE_FUNCTIONS = ("mp_thompson", "calc_effectrad", "calc_refl10cm")
WORKSPACE_TILE_COLUMNS = None
WORKSPACE_BUDGET_MIB = 1536
_WS_TYPES = (("double", "dd", 8), ("float", "ff", 4), ("bool", "bb", 1))
_WS_SUPPORT = r'''
// Column workspace views (memory lane). The workspace is cut into one block per warp of 32
// columns; inside a block, array i holds element k of lane l at [(i*LEVELS+k)*32+l], the
// interleaving CUDA local memory uses. The lane stride is a compile-time constant, so a
// constant level index is an immediate offset from the array's base, as in a thread frame.
#define TC_WS_LANES 32
template<class T> struct TcCol {
    T* p;
    __device__ __forceinline__ TcCol():p(nullptr) {}
    __device__ __forceinline__ explicit TcCol(T* q):p(q) {}
    __device__ __forceinline__ T& operator[](int k) const {return p[k*TC_WS_LANES];}
};
#define TC_WS_LEVELS (THOMPSON_COLUMN_KMAX+1)
struct TcWs {
    double* d; float* f; bool* b;
    __device__ __forceinline__ TcCol<double> dd(int i) const {return TcCol<double>(d+i*(TC_WS_LEVELS*TC_WS_LANES));}
    __device__ __forceinline__ TcCol<float> ff(int i) const {return TcCol<float>(f+i*(TC_WS_LEVELS*TC_WS_LANES));}
    __device__ __forceinline__ TcCol<bool> bb(int i) const {return TcCol<bool>(b+i*(TC_WS_LEVELS*TC_WS_LANES));}
};
__device__ __forceinline__ TcWs tc_ws_make(unsigned char* base,int local,int nd,int nf,int nb) {
    TcWs w;
    const size_t plane=(size_t)TC_WS_LEVELS*TC_WS_LANES;
    unsigned char* block=base+(size_t)(local/TC_WS_LANES)*plane*((size_t)nd*8+(size_t)nf*4+(size_t)nb);
    const int lane=local%TC_WS_LANES;
    w.d=(double*)block+lane;
    w.f=(float*)(block+plane*nd*8)+lane;
    w.b=(bool*)(block+plane*nd*8+plane*nf*4)+lane;
    return w;
}
__device__ bool tc_any(TcCol<bool> a,int lo,int hi) {for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}
'''


def _body_end(text: str, open_brace: int) -> int:
    """Index just past the brace closing the one at open_brace (// comments skipped)."""
    depth, i, n = 0, open_brace, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced Thompson CUDA source")


def _workspace_declarations(body: str, slots: dict, owner: str, prefix: str = "ws"):
    """Replace per-level column array declarations with workspace views.

    Unreferenced arrays are dropped (they own no storage either way). Returns
    the new body. ``slots`` holds the next free slot per element type.
    """
    import re
    single = re.compile(r"^(float|double|bool) (\w+)\[THOMPSON_COLUMN_KMAX(?:\+1)?\];$", re.M)
    multi = re.compile(r"^([ \t]*)float ((?:\w+\[THOMPSON_COLUMN_KMAX\],)+\w+\[THOMPSON_COLUMN_KMAX\]);$", re.M)

    def view(kind, name, rest):
        if not re.search(r"\b" + name + r"\b", rest):
            return None
        method = dict((t, m) for t, m, _ in _WS_TYPES)[kind]
        slot = slots[kind]; slots[kind] += 1
        return f"TcCol<{kind}> {name}={prefix}.{method}({slot});"

    out, pos = [], 0
    for match in sorted(list(single.finditer(body)) + list(multi.finditer(body)), key=lambda m: m.start()):
        out.append(body[pos:match.start()])
        rest = body[match.end():]
        if match.re is single:
            text = view(match.group(1), match.group(2), rest)
            out.append(text or "")
        else:
            names = re.findall(r"(\w+)\[THOMPSON_COLUMN_KMAX\]", match.group(2))
            views = [v for v in (view("float", n, rest) for n in names) if v]
            out.append(match.group(1) + "".join(views))
        pos = match.end()
    out.append(body[pos:])
    new = "".join(out)
    if "THOMPSON_COLUMN_KMAX]" in new or "THOMPSON_COLUMN_KMAX+1]" in new:
        raise ValueError(f"{owner}: a per-level column array escaped the workspace transformation")
    return new


def _workspace_source(text: str, kernels=None):
    """Move the per-level column arrays of the step and radar kernels to the workspace.

    Returns (text, layout); layout maps kernel -> slots per element type. The
    member functions take TcCol views where they took float*; tc_any gains a
    TcCol overload. Kernel bodies are otherwise the unchanged statements.
    """
    import re
    marker = "struct ThompsonColumn {"
    if text.count(marker) != 1:
        raise ValueError("Thompson column struct not found exactly once")
    text = text.replace(marker, _WS_SUPPORT + marker + "\nTcWs ws;", 1)
    layout = {}
    if kernels is None:
        kernels = (("mpas_thompson_step", ("mp_thompson", "calc_effectrad")),
                   ("mpas_thompson_radar", ("calc_refl10cm",)))
    for kernel, functions in kernels:
        slots = {"double": 0, "float": 0, "bool": 0}
        # kernel-scope arrays first, then each member function it calls
        head = text.index(f'extern "C" __global__ void {kernel}(')
        brace = text.index("{", text.index(")", head))
        end = _body_end(text, brace)
        body = _workspace_declarations(text[brace:end], slots, kernel, "m.ws")
        # the launcher passes the workspace, the tile origin, its width and its stride
        sig = text[head:brace].rstrip()
        if not sig.endswith(")"):
            raise ValueError(f"{kernel}: unexpected signature")
        sig = sig[:-1] + ",\n    unsigned char* tc_ws,int tc_col0,int tc_n)\n"
        first = "int col=blockIdx.x*blockDim.x+threadIdx.x;"
        if body.count(first) != 1 or body.count("ThompsonColumn m;") != 1:
            raise ValueError(f"{kernel}: unexpected column prologue")
        body = body.replace(first, "int tc_local=blockIdx.x*blockDim.x+threadIdx.x;"
                            "if(tc_local>=tc_n)return;int col=tc_col0+tc_local;", 1)
        body = body.replace("ThompsonColumn m;", "ThompsonColumn m;"
                            f"m.ws=tc_ws_make(tc_ws,tc_local,TC_WS_{kernel.upper()}_ND,TC_WS_{kernel.upper()}_NF,TC_WS_{kernel.upper()}_NB);", 1)
        text = text[:head] + sig + body + text[end:]
        for function in functions:
            start = text.index(f"__device__ __forceinline__ void {function}(")
            close = text.index(")", start)
            params = text[start:close].replace("float* ", "TcCol<float> ")
            brace = text.index("{", close)
            end = _body_end(text, brace)
            body = _workspace_declarations(text[brace:end], slots, function)
            text = text[:start] + params + text[close:brace] + body + text[end:]
        layout[kernel] = dict(slots)
    defines = "".join(f"#define TC_WS_{k.upper()}_N{t[0].upper()} {n}\n"
                      for k, s in layout.items() for t, n in s.items())
    return defines + text, layout


def workspace_bytes(layout: Mapping, kernel: str, nz: int, stride: int) -> int:
    """Bytes of one tile of the column workspace for ``kernel`` (``stride`` columns, whole warps)."""
    width = 128 if _hex_numerics.mode() == "fast" else 32
    lanes = -(-stride // width) * width
    size = (nz + 1) * lanes * sum(layout[kernel][t] * size for t, _, size in _WS_TYPES)
    if _hex_numerics.mode() == "fast":
        from .thompson_phases_v842 import CARRY_BYTES_PER_COLUMN
        size += lanes * CARRY_BYTES_PER_COLUMN
    return size


def workspace_enabled() -> bool:
    import os
    value = os.environ.get("WOOF_HEX_THOMPSON_WORKSPACE", "1").strip().lower()
    if value in ("1", "on", "true", "yes", ""):
        return True
    if value in ("0", "off", "false", "no", "local"):
        return False
    raise ValueError("WOOF_HEX_THOMPSON_WORKSPACE must be 1/on or 0/off")


def workspace_tile_columns():
    """Columns per workspace tile from the environment, or None for the card's resident width."""
    import os
    value = os.environ.get("WOOF_HEX_THOMPSON_TILE_COLUMNS", "").strip().lower()
    if value in ("", "auto"):
        return WORKSPACE_TILE_COLUMNS
    tile = int(value)
    if tile < 32:
        raise ValueError("WOOF_HEX_THOMPSON_TILE_COLUMNS must be at least 32")
    return tile


def workspace_budget_bytes() -> int:
    """Upper bound on one call's Thompson workspace tile, from the environment or the default."""
    import os
    value = os.environ.get("WOOF_HEX_THOMPSON_WORKSPACE_MIB", "").strip().lower()
    default_mib = 512 if shared_card_memory() else WORKSPACE_BUDGET_MIB
    mib = default_mib if value in ("", "auto") else int(value)
    if mib < 64:
        raise ValueError("WOOF_HEX_THOMPSON_WORKSPACE_MIB must be at least 64")
    return mib << 20


def budget_tile_columns(kernel_name: str, nz: int, *, layout=None) -> int:
    """Most columns (whole 128-column blocks) whose workspace tile fits the byte budget."""
    per_block = workspace_bytes(workspace_layout() if layout is None else layout, kernel_name, nz, 128)
    return max(128, (workspace_budget_bytes() // per_block) * 128)


_RESIDENT_COLUMNS = {}


def resident_columns(kernel, block_threads: int = 32) -> int:
    """Threads of actual-sized blocks the current card runs at once."""
    from cupy.cuda import driver, runtime
    device = runtime.getDevice()
    key = (device, kernel.ptr, block_threads)
    if key not in _RESIDENT_COLUMNS:
        sms = runtime.deviceGetAttribute(runtime.cudaDevAttrMultiProcessorCount, device)
        blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.ptr, block_threads, 0)
        _RESIDENT_COLUMNS[key] = max(1, sms * blocks) * block_threads
    return _RESIDENT_COLUMNS[key]


@lru_cache(maxsize=1)
def _fast_support() -> str:
    """Let fast-mode arithmetic contract, without the strict payload adapters.

    Keep precise division/sqrt and the shared elementary maths. The strict
    source stays byte-for-byte unchanged, including its kernel-pack key.
    This runs once during module construction, never in the column loop.
    """
    import re
    source = _SUPPORT
    for scalar, sqrt in (("float", "sqrtf"), ("double", "sqrt")):
        for name, expression, arguments in (
            ("add", "a+b", f"{scalar} a,{scalar} b"),
            ("sub", "a-b", f"{scalar} a,{scalar} b"),
            ("mul", "a*b", f"{scalar} a,{scalar} b"),
            ("div", "a/b", f"{scalar} a,{scalar} b"),
            ("sqrt", f"{sqrt}(a)", f"{scalar} a"),
        ):
            signature = f"__device__ {scalar} tc_{name}({arguments})"
            source, count = re.subn(
                re.escape(signature) + r" \{[^\n]*\}",
                f"__device__ __forceinline__ {scalar} tc_{name}({arguments}) {{return {expression};}}",
                source,
            )
            if count != 1:
                raise ValueError(f"Thompson fast arithmetic: expected one {signature}, found {count}")
    # Elementary functions for fast numerics (strict keeps every glibc replica).
    # binary32 EXP/LOG/LOG10/** come from CUDA's libm: the replicas of glibc's float functions
    # evaluate in binary64, which RTX-class cards run at 1/64 rate, and CUDA's are binary32
    # throughout (within 2-4 ulp). binary64 EXP/LOG/** (the DOUBLE PRECISION slopes and
    # intercepts) evaluate in binary32 when argument and result are normal binary32 numbers
    # (their arguments mostly are binary32 values widened, so the narrowing is exact); outside
    # that range the argument is split exactly in binary64 and the core is still binary32, so
    # the whole binary64 range is kept; zero, negative, infinite and NaN arguments take the
    # exact replica. About 1e-7 relative, the class of the fast arithmetic. Measured on the
    # 60 -> 15 km zoom globe at dt 90 (216,187 columns, a 3 h state, quiet RTX PRO 6000):
    # 85 -> 62 ms for the binary32 functions, then 68 -> 41 ms for the binary64 ones. CUDA's
    # own binary64 pow was slower than the replica (74 ms).
    narrow = "fabsf(r)>1.2e-38f&&fabsf(r)<3.0e38f"
    for old, new in (
        ("__device__ double tc_exp(double a) {return mpas_exp(a);}",
         "__device__ __forceinline__ double tc_exp(double a) {"
         f"const float r=expf((float)a);if({narrow})return (double)r;"
         "return isfinite(a)?tc_fast_exp2(a*1.4426950408889634):mpas_exp(a);}"),
        ("__device__ double tc_log(double a) {return mpas_log(a);}",
         "__device__ __forceinline__ double tc_log(double a) {"
         "if(a>1.2e-38&&a<3.0e38)return (double)logf((float)a);"
         "return (a>0.0&&isfinite(a))?tc_fast_log2(a)*0.6931471805599453:mpas_log(a);}"),
        ("__device__ double tc_pow(double a,double b) {return mpas_pow(a,b);}",
         "__device__ __forceinline__ double tc_pow(double a,double b) {"
         f"if(a>1.2e-38&&a<3.0e38){{const float r=powf((float)a,(float)b);if({narrow})return (double)r;}}"
         "return (a>0.0&&isfinite(a)&&isfinite(b))?tc_fast_exp2(b*tc_fast_log2(a)):mpas_pow(a,b);}"),
    ):
        if source.count(old) != 1:
            raise ValueError(f"Thompson fast maths: expected one {old}")
        source = source.replace(old, new)
    core = ("__device__ __forceinline__ double tc_fast_log2(double a) "
            "{int e;const double m=frexp(a,&e);return (double)e+(double)log2f((float)m);}\n"
            "__device__ __forceinline__ double tc_fast_exp2(double y) "
            "{const double f=floor(fmin(fmax(y,-2000.0),2000.0));"
            "return ldexp((double)exp2f((float)(y-f)),(int)f);}\n")
    anchor = "__device__ __forceinline__ double tc_exp(double a) {"
    source = source.replace(anchor, core + anchor, 1)
    for old, new in (
        ("__device__ float tc_exp(float a) {return mpas_expf(a);}",
         "__device__ __forceinline__ float tc_exp(float a) {return expf(a);}"),
        ("__device__ float tc_log(float a) {return mpas_logf(a);}",
         "__device__ __forceinline__ float tc_log(float a) {return logf(a);}"),
        ("__device__ float tc_log10(float a) {return mpas_log10f(a);}",
         "__device__ __forceinline__ float tc_log10(float a) {return log10f(a);}"),
    ):
        if source.count(old) != 1:
            raise ValueError(f"Thompson fast maths: expected one {old}")
        source = source.replace(old, new)
    source, count = re.subn(r"__device__ float tc_pow\(float a,float b\) \{[^\n]*\}",
                            "__device__ __forceinline__ float tc_pow(float a,float b) {return powf(a,b);}", source)
    if count != 1:
        raise ValueError("Thompson fast maths: expected one binary32 tc_pow")
    return source


def cuda_source(*, device_math_source: str, max_levels: int = MAX_LEVELS, corrections=(),
                workspace: bool = False, numerics: str | None = None, driver: str | None = None) -> str:
    """Compose with the shared mathematics header; no local libm fallback.

    MPAS module_mp_thompson.F:1400-3642,4176-4212,4391-4466,
    4662-4803,4809-4913,4924-5235; src/core_atmosphere/physics/mpas_atmphys_functions.F:156-217.
    ``workspace`` selects the column-workspace storage (see WORKSPACE_FUNCTIONS).
    ``numerics`` lets the offline pack compose both modes in one process.
    ``driver`` replaces the mpas_thompson_step driver text with another whose kernel has the
    same name, prologue, call and commit shape (thompson_aero_v842's aerosol driver), so the
    workspace storage and the fast phase split apply to it unchanged.
    """
    if not device_math_source.strip():
        raise ValueError("shared MPAS device maths source is required to preserve the reference libm")
    if not 1 <= max_levels <= MAX_LEVELS:
        raise ValueError("column capacity must be between 1 and 256 levels")
    declarations = ",".join(f"const {'float' if n == 'tnccn_act' else 'double'}* input_{n}" for n in TABLE_ARGUMENTS)
    assignments = "\n".join(f"m.{n}=input_{n};" for n in TABLE_ARGUMENTS)
    driver = (_DRIVER if driver is None else driver).replace(
        "TABLE_DECLARATIONS", declarations).replace("TABLE_ASSIGNMENTS", assignments)
    mode = _hex_numerics.mode() if numerics is None else _hex_numerics.normalise(numerics)
    support = _SUPPORT if mode == "strict" else _fast_support()
    text = f"#define THOMPSON_COLUMN_KMAX {max_levels}\n" + device_math_source + "\n" + support + _COLUMN + driver + _RADAR_DRIVER + _conversion_source() + _PRECIP
    if workspace:
        text = _workspace_source(text)[0]
        if mode == "fast":
            from .thompson_phases_v842 import split_source
            text = split_source(text, _body_end)
    if not corrections:
        return text
    from .driver_v842 import correct_source
    return correct_source(text, CORRECT_MODE_SOURCE, corrections, "thompson_v842")


@lru_cache(maxsize=1)
def workspace_layout():
    """Workspace slots per kernel (independent of levels, maths and corrections)."""
    return _workspace_source(f"#define THOMPSON_COLUMN_KMAX {MAX_LEVELS}\n" + _SUPPORT + _COLUMN
                             + _DRIVER + _RADAR_DRIVER)[1]


# One entry per (card, maths, levels, corrections, workspace), unbounded on purpose. The correct-mode door asks
# for two corrections variants per card, so maxsize=8 thrashed on 8 cards: half the ranks reloaded two Thompson
# modules every step (measured 2026-10-28, 286,530-cell zoom globe, 8x RTX 5090: 1.28 s/step on 8 cards against
# 0.27 s on 4). The entries are bounded by cards x variants.
@lru_cache(maxsize=None)
def _module(device: int, device_math_source: str, max_levels: int, corrections=(), workspace: bool = False):
    """Cache device implementations of MPAS module_mp_thompson.F:1001-1387."""
    import cupy as cp
    with cp.cuda.Device(device):
        return compile_module(cuda_source(device_math_source=device_math_source, max_levels=max_levels,
                                          corrections=corrections, workspace=workspace))


def _launch_columns(kernel_name: str, module, args: tuple, nz: int, columns: int, workspace: bool,
                    layout: Mapping | None = None):
    """Launch a column kernel, in workspace tiles when the workspace is on.

    ``layout`` is the module's workspace slot count per kernel (default: this module's own source).
    """
    layout = layout or workspace_layout()
    import cupy as cp
    kernel = module.get_function(kernel_name)
    if not workspace:
        kernel(((columns+31)//32,),(32,),args)
        return
    phases = None
    if kernel_name == "mpas_thompson_step" and _hex_numerics.mode() == "fast":
        from .thompson_phases_v842 import CELL_PHASES, KERNELS, WARP_GROUPS
        phases = tuple(module.get_function(name) for name in KERNELS)
    tile = workspace_tile_columns()
    # A tile is the columns the column-serial phases run at once; the warp phase launches
    # 32 threads per column of the same tile and simply takes more waves.
    resident = (min(resident_columns(phase, 128) for i, phase in enumerate(phases) if i not in WARP_GROUPS)
                if phases else resident_columns(kernel))
    stride = min(columns, tile if tile is not None else min(resident, budget_tile_columns(kernel_name, nz, layout=layout)))
    nbytes = workspace_bytes(layout, kernel_name, nz, stride)
    if (nz + 1) * stride >= 2**31:
        raise ValueError("Thompson workspace tile too wide for 32-bit level offsets")
    ws = cp.empty(nbytes, dtype=cp.uint8)
    for col0 in range(0, columns, stride):
        n = min(stride, columns - col0)
        tile_args = args+(ws,np.int32(col0),np.int32(n))
        if phases:
            for phase, phase_kernel in enumerate(phases):
                count = n * nz if phase in CELL_PHASES else n * 32 if phase in WARP_GROUPS else n
                threads = 128
                phase_kernel(((count + threads - 1) // threads,), (threads,), tile_args)
        else:
            kernel(((n+31)//32,),(32,),tile_args)
    del ws


def compile_module(source: str):
    """Compile the RawKernel sources without CuPy's appended FTZ override.

    Device implementation of MPAS module_mp_thompson.F:1001-1387.

    CuPy 14.2 compiler._compile_with_cache_cuda appends -ftz=true after
    caller options, flushing native binary32 subnormals despite --ftz=false.
    receipts/log10-shared-cuda-scan.json witnesses this concrete defect.
    Compile directly through NVRTC and retain the authenticated image in the
    strict persistent cache for subsequent forecast processes.
    The existing per-device _module cache retains the six exported kernels.
    """
    from cupy.cuda import compiler, function
    from ..cuda_backend.aot import direct_image
    image = direct_image(source, _hex_numerics.options(COMPILE_OPTIONS), compiler._get_arch())
    module = function.Module()
    module.load(image)
    return module


def launch_driver(fields: Mapping, surface: Mapping, tables: Mapping, *, dt: float,
                  device_math_source: str, has_reqc: int = 1, has_reqi: int = 1, has_reqs: int = 1):
    """Launch the MPAS mp_gt_driver boundary, with column-fastest arrays.

    MPAS module_mp_thompson.F:1001-1387. The f4 driver owns invocation count,
    pool preparation. run_slot owns the MPAS from/to conversion and forecast
    precipitation accumulation once per microphysics call.
    """
    import cupy as cp
    if tables.get("_file_sha256") != TABLE_FILE_SHA256:
        raise ValueError("Rust table preparation must certify the f0 strict MPAS lookup bytes; WRF table values change collision rates")
    shape = fields["qv"].shape
    if len(shape) != 2 or not 1 <= shape[0] <= MAX_LEVELS or shape[1] < 1:
        raise ValueError("Thompson needs (levels, columns), 1..256 levels, and a nonempty tile")
    nz, columns = shape
    device = fields["qv"].device.id
    for name in FIELD_ARGUMENTS:
        array=fields[name]
        if array.shape != shape or array.dtype != cp.float32 or not array.flags.c_contiguous or array.device.id != device:
            raise ValueError(f"{name} must be contiguous float32 on the tile device with shape {shape}")
    for name in SURFACE_ARGUMENTS:
        array=surface[name]
        if array.shape != (columns,) or array.dtype != cp.float32 or not array.flags.c_contiguous or array.device.id != device:
            raise ValueError(f"{name} must be contiguous float32 with one entry per column on the tile device")
    for name, expected in TABLE_SHAPES.items():
        array=tables[name]
        dtype=cp.float32 if name == "tnccn_act" else cp.float64
        if array.shape != expected or array.dtype != dtype or not array.flags.f_contiguous or array.device.id != device:
            raise ValueError(f"{name}: wrong MPAS table shape, precision, Fortran ordering or device; lookup indices would change")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite dt required by MPAS reciprocal timestep")
    workspace=workspace_enabled()
    module=_module(device,device_math_source,nz,(),workspace)
    args=tuple(fields[n] for n in FIELD_ARGUMENTS)+tuple(surface[n] for n in SURFACE_ARGUMENTS)
    args+=(np.int32(nz),np.int32(columns),np.float32(dt),np.int32(has_reqc),np.int32(has_reqi),np.int32(has_reqs))
    args+=tuple(tables[n] for n in TABLE_ARGUMENTS)
    _launch_columns("mpas_thompson_step",module,args,nz,columns,workspace)


def launch_radar(fields: Mapping, diagnostics: Mapping, *, device_math_source: str):
    """MPAS reflectivity, maximum, 1 km interpolation, and running 1 km maximum.

    MPAS mpas_atmphys_driver_microphysics.F:738-792. The MPAS caller must
    provide a vertical grid bracketing 1 km, as required by its kp+1 access.
    """
    import cupy as cp
    nz, columns = fields["qv"].shape
    if not 2 <= nz <= MAX_LEVELS:
        raise ValueError("MPAS radar interpolation needs at least two levels and capacity <=256")
    names=("qv","qc","qr","nr","qs","qg","th","pii","p","z","dz")
    device=fields["qv"].device.id
    for name in names:
        a=fields[name]
        if a.shape != (nz,columns) or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
            raise ValueError(f"{name}: radar requires column-fastest float32 arrays on the same device")
    for name in ("refl10cm","refl10cm_max","refl10cm_1km","refl10cm_1km_max"):
        a=diagnostics[name];shape=(nz,columns) if name=="refl10cm" else (columns,)
        if a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
            raise ValueError(f"{name}: wrong radar output shape, precision, ordering or device")
    args=tuple(fields[n] for n in names)+tuple(diagnostics[n] for n in ("refl10cm","refl10cm_max","refl10cm_1km","refl10cm_1km_max"))
    args+=(np.int32(nz),np.int32(columns))
    workspace=workspace_enabled()
    _launch_columns("mpas_thompson_radar",_module(device,device_math_source,nz,(),workspace),args,nz,columns,workspace)


_POINTER_TABLES = {}


def _pointer_table(arrays):
    """Device table of the arrays' addresses, uploaded once per distinct set of views.

    The pools are resident, so the same views come back every step; rebuilding the table each
    call put a blocking host-to-device copy on the step path four times a step. The key is the
    addresses themselves, so a cached table always holds exactly the addresses asked for.
    """
    import cupy as cp
    key = (cp.cuda.runtime.getDevice(), tuple(a.data.ptr for a in arrays))
    table = _POINTER_TABLES.get(key)
    if table is None:
        if len(_POINTER_TABLES) >= 64:
            _POINTER_TABLES.clear()
        table = _POINTER_TABLES[key] = cp.asarray(np.array(key[1], dtype=np.uint64))
    return table


def launch_conversion(phase: str, model: Mapping, mesh: Mapping, diagnostics: Mapping,
                      tendencies: Mapping, fields: Mapping, constants: Mapping, *,
                      device_math_source: str, mp_top_level: int | None = None,
                      dt_dyn: float | None = None):
    """MPAS microphysics_from_MPAS/to_MPAS, without pool lookup or I/O.

    MPAS mpas_atmphys_interface.F:598-823,826-1138. Tile arrays are shaped
    (levels,columns). zgrid has one extra interface level; w may too.
    rainprod, evapprod and effective radii in fields must be the diagnostic
    device views. Surface ntc/muc passed to launch_driver must likewise be
    views of diag_physics nt_c/mu_c. The f4 seam owns these pool bindings.
    """
    import cupy as cp
    if phase not in ("from", "to"):
        raise ValueError("MPAS conversion phase must be 'from' or 'to'")
    nz, columns=fields["qv"].shape;device=fields["qv"].device.id
    if not 2 <= nz <= MAX_LEVELS or columns < 1:
        raise ValueError("MPAS surface-pressure conversion requires >=2 levels and <=256 capacity")
    arrays=tuple(model[n] for n in MODEL_ARGUMENTS)+tuple(mesh[n] for n in MESH_ARGUMENTS)
    arrays+=tuple(diagnostics[n] for n in DIAG_ARGUMENTS)+tuple(tendencies[n] for n in TENDENCY_ARGUMENTS)
    arrays+=tuple(fields[n] for n in FIELD_ARGUMENTS)+tuple(fields[n] for n in ("rho","z"))
    arrays+=tuple(diagnostics[n] for n in ("surface_pressure","tend_sfc_pressure"))
    for name,a in zip(_CONVERSION_ARRAYS,arrays):
        expected=(columns,) if name in ("d_surface_pressure","d_tend_sfc_pressure") else (nz,columns)
        if name == "g_zgrid":expected=(nz+1,columns)
        if name == "m_w" and a.shape==(nz+1,columns):expected=a.shape
        if a.shape != expected or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
            raise ValueError(f"{name}: invalid MPAS conversion tile shape, precision, ordering or device")
    args=(_pointer_table(arrays),np.int32(nz),np.int32(columns))
    if phase == "to":
        if mp_top_level is None or not 1 <= mp_top_level <= nz:
            raise ValueError("MPAS top level must be a 1-based inclusive level inside the tile")
        if dt_dyn is None or not np.isfinite(dt_dyn) or dt_dyn <= 0:
            raise ValueError("positive finite dynamical timestep required by MPAS tendencies")
        args+=(np.int32(mp_top_level),np.float32(dt_dyn))
    args+=tuple(np.float32(constants[n]) for n in CONVERSION_CONSTANTS)
    threads=nz*columns if phase=="from" else columns
    kernel=_module(device,device_math_source,nz,(),workspace_enabled()).get_function("mpas_thompson_"+phase)
    kernel(((threads+31)//32,),(32,),args)


def launch_precip(phase: str, fields: Mapping, surface: Mapping, diagnostics: Mapping, *,
                  device_math_source: str, l_acrain: bool, bucket: float, corrections=()):
    """Reset working precipitation or accumulate it after all substeps.

    MPAS mpas_atmphys_driver_microphysics.F:503-653. The MPAS bucket test is
    strict > and subtracts a single bucket even when the total exceeds two.
    """
    import cupy as cp
    if phase not in ("from","to"):
        raise ValueError("precipitation conversion phase must be 'from' or 'to'")
    nz,columns=fields["qv"].shape;device=fields["qv"].device.id
    inputs=tuple(fields[n] for n in ("qv","rho","dz"))
    work=tuple(surface[n] for n in SURFACE_ARGUMENTS[:7]);diag=tuple(diagnostics[n] for n in PRECIP_ARGUMENTS)
    for a in inputs+work+diag:
        expected=(nz,columns) if any(a is x for x in inputs) else (columns,)
        if a.shape != expected or a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
            raise ValueError("precipitation arrays must be contiguous float32 on the tile device")
    counter=diagnostics["i_rainnc"]
    if counter.shape!=(columns,) or counter.dtype!=cp.int32 or not counter.flags.c_contiguous or counter.device.id!=device:
        raise ValueError("MPAS precipitation bucket counter requires contiguous int32 per column")
    args=inputs+(_pointer_table(work),_pointer_table(diag),counter,np.int32(nz),np.int32(columns),np.int32(phase=="to"),np.int32(l_acrain),np.float32(bucket))
    # Correct mode (CORRECT_MODE_SOURCE): the precipitation kernel comes from the corrected text.
    module=_module(device,device_math_source,nz,tuple(corrections),workspace_enabled())
    module.get_function("mpas_thompson_precip")(((columns+31)//32,),(32,),args)


def prepare_native_tables(provider, *, file_sha256: Mapping):
    """Copy initialized native tables without decoding or recomputing them.

    MPAS module_mp_thompson.F:433-993,4068-4168,4222-4310. The native
    session owner supplies the certificate for the four files actually read
    during initialization. Retain this mapping for the lifetime of that
    session/device; all 33 arrays are read-only inputs to the CUDA column.
    """
    if dict(file_sha256) != TABLE_FILE_SHA256:
        raise ValueError("native initialization must certify the f0 strict Thompson table files")
    cp = provider.cp
    tables = {"_file_sha256": dict(file_sha256)}
    for name, shape in TABLE_SHAPES.items():
        value = provider.native.get_table("module_mp_thompson." + name)
        if value is None:
            raise RuntimeError(f"native initialized Thompson table {name} is not allocated")
        host, _ = value
        dtype = np.float32 if name == "tnccn_act" else np.float64
        if host.shape != shape or host.dtype != dtype or not host.flags.f_contiguous:
            raise ValueError(f"native Thompson table {name} has incorrect extent, precision or ordering")
        tables[name] = cp.asarray(host, order="F")
    return tables


def bind_native_pools(provider, call, *, writeback_levels=None):
    """Own missing native workspace until the committed callback completes.

    MPAS mpas_atmphys_driver_microphysics.F:98-257,359-366,491-496.
    Existing native allocation belongs to the caller. Otherwise this driver
    slice allocates through the native ABI and commits before deallocation.
    """
    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("native Thompson binder requires its registered microphysics slot")
    owned = (not getattr(provider, 'resident', False)
             and provider.native.get_variable("qv_p") is None)
    if owned:
        provider.native.call("allocate_microphysics", configs="configs")
    try:
        bound = _bind_native_views(provider, call, writeback_levels=writeback_levels)
    except BaseException:
        if owned:
            provider.native.call("deallocate_microphysics", configs="configs")
        raise
    if owned:
        bound["thompson_v842"]["_native_workspace"] = provider
    return bound


def _finish_native_workspace(p, *, commit):
    """Return pool words before releasing workspace owned by this callback.

    MPAS mpas_atmphys_driver_microphysics.F:484-496;
    mpas_atmphys_interface.F:826-1138.
    The shared adapter's subsequent commit has no remaining buffered words.
    """
    provider = p.pop("_native_workspace", None)
    if provider is not None:
        try:
            if commit:
                provider.commit()
        finally:
            provider.native.call("deallocate_microphysics", configs="configs")


def _bind_native_views(provider, call, *, writeback_levels=None):
    """Bind f4's native solve-cell views to the complete Thompson callback.

    MPAS mpas_atmphys_interface.F:598-823,826-1138;
    mpas_atmphys_driver_microphysics.F:317-500,503-653,738-850;
    mpas_atmphys_vars.F:149-162,185-218,266-294.

    The session owner installs provider.thompson_v842_resources containing
    certified initialized ``tables`` and the approved ``device_math_source``.
    This binder does no data-path arithmetic, file decoding or substep rounding.
    NativeDevicePoolViews.begin selects state time level 2; commit preserves
    halos and returns the changed solve cells and prepared module arrays.
    """
    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("native Thompson binder requires its registered microphysics slot")
    if provider.level != 2:
        raise ValueError("MPAS microphysics updates state time level 2; level 1 would overwrite the previous state")
    resources = getattr(provider, "thompson_v842_resources", None)
    if resources is None:
        raise RuntimeError("native session must supply certified Thompson tables and approved shared device maths")
    tables = resources["tables"]
    if tables.get("_file_sha256") != TABLE_FILE_SHA256:
        raise ValueError("native Thompson resources must certify the actual strict lookup files")
    maths = resources["device_math_source"]
    if not isinstance(maths, str) or not maths.strip():
        raise ValueError("native Thompson resources require the approved shared device maths source")

    state = provider["state"]
    scalars = state["scalars"]
    model = {}
    for name in MODEL_ARGUMENTS[:8]:
        index = provider.session.layout.indices["state", "index_" + name]
        if not 1 <= index <= scalars.shape[0]:
            raise ValueError(f"native index_{name} must address its 1-based scalar component")
        model[name] = scalars[index - 1]
    model.update({name: state[name] for name in MODEL_ARGUMENTS[8:]})
    mesh = {name: provider["mesh"][name] for name in MESH_ARGUMENTS}
    diagnostics = {name: provider["diag"][name] for name in DIAG_ARGUMENTS
                   if name != "rt_diabatic_tend"}
    diagnostics["rt_diabatic_tend"] = provider["tend"]["rt_diabatic_tend"]
    diagnostics["surface_pressure"] = provider["diag"]["surface_pressure"]
    diagnostics["tend_sfc_pressure"] = provider["tend"]["tend_sfc_pressure"]
    if call.flags.diagnostics:
        diagnostics["relhum"] = provider["diag"]["relhum"]
        diagnostics.update({name: provider["diag_physics"][name] for name in
                            ("refl10cm", "refl10cm_max", "refl10cm_1km", "refl10cm_1km_max")})
    tendencies = {name: provider["tend_physics"][name] for name in TENDENCY_ARGUMENTS}
    constants = {name: provider["constants"][name] for name in CONVERSION_CONSTANTS}

    prepared = provider["prepared"]
    nz = model["qv"].shape[0]
    native_names = {"pii": "pi_p", "p": "pres_p"}
    diagnostic_fields = ("rainprod", "evapprod", "re_cloud", "re_ice", "re_snow")
    fields = {name: prepared[native_names.get(name, name + "_p")][:nz]
              for name in (*FIELD_ARGUMENTS, "rho", "z") if name not in diagnostic_fields}
    fields.update({name: provider["diag_physics"][name] for name in diagnostic_fields})
    surface = {name: prepared[name + "_p"] for name in SURFACE_ARGUMENTS[:7]}
    surface["ntc"] = provider["diag_physics"]["nt_c"]
    surface["muc"] = provider["diag_physics"]["mu_c"]
    precip = {name: provider["diag_physics"][name] for name in (*PRECIP_ARGUMENTS, "i_rainnc")}
    native_scalars = provider["scalar_state"]
    p = dict(model=model, mesh=mesh, diagnostics=diagnostics, tendencies=tendencies,
             constants=constants, fields=fields, surface=surface, precip=precip,
             tables=tables, device_math_source=maths,
             dt_microp=native_scalars["dt_microp"], n_microp=int(native_scalars["n_microp"]),
             mp_top_level=(_mp_top_level(provider, resources) if writeback_levels is None
                           else writeback_levels),
             l_acrain=bool(native_scalars["l_acrain"]))
    p.update({name: bool(native_scalars[name]) for name in ("has_reqc", "has_reqi", "has_reqs")})
    # A [hex_nests] trigger look armed on this run (mpasphys/nest_look_v842): None on every other run.
    p["nest_look"] = getattr(provider.session, "nest_look", None)
    return {"thompson_v842": p}


def _mp_top_level(provider, resources):
    """mp_top_level, read from the device once per session (set at init, constant after).

    Reading it with .item() every call was a device-to-host wait on the step path.
    """
    value = provider["tend_physics"]["mp_top_level"]
    data = getattr(value, "data", None)
    key = getattr(data, "ptr", None)
    if key is None:
        return int(value.item())
    cache = resources.setdefault("_mp_top_level", {})
    if key not in cache:
        cache[key] = int(value.item())
    return cache[key]


def register(registry):
    """Register the published f4 SlotRegistry protocol for MPAS Thompson.

    MPAS mpas_atmphys_driver_microphysics.F:389-411. Registration changes
    neither suite selection nor defaults; the real-door owner invokes this.
    """
    registry.register("microphysics",SCHEME,BACKEND,run_slot)


def run_slot(call):
    """Complete the native workspace transaction around the CUDA driver slice.

    MPAS mpas_atmphys_driver_microphysics.F:359-496. Synthetic replay pools
    and workspace supplied by a native owner retain their existing lifecycle.
    """
    if call.slot != "microphysics" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("Thompson callback requires its exact registered MPAS slot")
    try:
        _run_bound_slot(call)
    except BaseException:
        _finish_native_workspace(call.pools["thompson_v842"], commit=False)
        raise
    _finish_native_workspace(call.pools["thompson_v842"], commit=True)


def _run_bound_slot(call):
    """Execute the full MPAS Thompson driver slice using adapter-bound pools.

    MPAS mpas_atmphys_driver_microphysics.F:317-500. Pool key
    thompson_v842 contains model, mesh, diagnostics, tendencies, fields,
    surface, precip, tables, constants, device_math_source, dt_microp,
    n_microp, mp_top_level, l_acrain, and has_reqc/has_reqi/has_reqs. Numerical
    fields are explicit views of restart-owned pools. Native manager glue
    prepares dt_microp and n_microp; no Python rounding recreates them.
    """
    import cupy as cp
    if call.slot!="microphysics" or call.scheme!=SCHEME or str(call.backend)!=BACKEND:
        raise ValueError("Thompson callback requires its exact registered MPAS slot")
    p=call.pools["thompson_v842"]
    if type(p["n_microp"]) is not int or p["n_microp"]<1:
        raise ValueError("native manager must provide a positive integer microphysics substep count")
    maths=p["device_math_source"];fields=p["fields"]
    conversion=(p["model"],p["mesh"],p["diagnostics"],p["tendencies"],fields,p["constants"])
    launch_precip("from",fields,p["surface"],p["precip"],device_math_source=maths,
                  l_acrain=p["l_acrain"],bucket=call.options["config_bucket_rainnc"])
    launch_conversion("from",*conversion,device_math_source=maths)
    for _ in range(p["n_microp"]):
        launch_driver(fields,p["surface"],p["tables"],dt=p["dt_microp"],device_math_source=maths,
                      **{n:p[n] for n in ("has_reqc","has_reqi","has_reqs")})
    if call.flags.diagnostics:
        launch_radar(fields,p["diagnostics"],device_math_source=maths)
    if p.get("nest_look") is not None:
        # A trigger look's reflectivity: the diagnostics' own on a diagnostics step, else into scratch only.
        p["nest_look"].form(fields,p["diagnostics"],diagnostics_formed=bool(call.flags.diagnostics),
                            launch_radar=launch_radar,device_math_source=maths)
    if call.flags.diagnostics:
        nz,columns=fields["qv"].shape;a=p["diagnostics"]["relhum"]
        if a.shape!=(nz,columns) or a.dtype!=cp.float32 or not a.flags.c_contiguous or a.device.id!=fields["qv"].device.id:
            raise ValueError("MPAS relative humidity requires the float32 mass-grid diagnostic view")
        args=tuple(fields[n] for n in ("qv","th","pii","p"))+(a,np.int32(nz*columns))
        _module(a.device.id,maths,nz,(),workspace_enabled()).get_function("mpas_thompson_relhum")(((nz*columns+31)//32,),(32,),args)
    from .driver_v842 import slot_corrections
    corrections=slot_corrections(call,CORRECT_MODE_SOURCE)
    launch_precip("to",fields,p["surface"],p["precip"],device_math_source=maths,
                  l_acrain=p["l_acrain"],bucket=call.options["config_bucket_rainnc"],
                  **({"corrections":corrections} if corrections else {}))
    launch_conversion("to",*conversion,device_math_source=maths,mp_top_level=p["mp_top_level"],dt_dyn=call.dt)
