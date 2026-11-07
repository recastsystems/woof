// THIRD-PARTY NOTICE. The log10f, expm1f and tanhf reductions below
// retain the permissive FDLIBM grant, also in licenses/LICENSE-FDLIBM-SunPro.txt.
// Source checked: glibc-2.39 sysdeps/ieee754/flt-32/{e_log10f,s_expm1f,s_tanhf}.c
// https://github.com/bminor/glibc/tree/glibc-2.39/sysdeps/ieee754/flt-32
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
//
// ======================================================================
// WOOF float32 libm, continued: the log10 and tanh words.
//
// glibc_flt32.cuh carries WOOF's shared float32 logf/expf/powf/tgamma
// (gfk_*).  A real-argument LOG10 and TANH also reach gfortran's libm, and
// until now their WOOF words lived only inside single translation units:
// log10 as noahmp_leaves.cu::r_log10 (verified by the Noah-MP lane,
// gpuwm/core/noahmp_libm.py log10f) and tanh with its expm1 as
// mynn_pbl.cu::mynn_tanhf/mynn_expm1f.  This header gives them the gfk_
// names beside the rest, SAME ALGORITHMS, the same FP32 statement order,
// with every operation pinned to a round-to-nearest intrinsic (FADD/FSUB/
// FMUL/FDIV from glibc_flt32.cuh) so contraction cannot move a word.
// gfk_log10 calls gfk_log exactly where r_log10 calls r_log; the two log
// words are the same function.
//
// It must be listed AFTER glibc_flt32.cuh (it uses gfk_log and the FADD
// family).  No existing unit includes it, so nothing that compiled before
// assembles a different source.
// ======================================================================
#ifndef WOOF_FLT32_LOG10_TANH_CUH
#define WOOF_FLT32_LOG10_TANH_CUH

// log10 on logf: k*log10(2) in two parts plus log(x')/ln(10).
__device__ float gfk_log10(float x)
{
    const float ivln10 = __int_as_float(0x3ede5bd9);
    const float log10_2hi = __int_as_float(0x3e9a2080);
    const float log10_2lo = __int_as_float(0x355427db);
    const float two25 = __int_as_float(0x4c000000);
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if ((int)hx < 0x00800000) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if ((int)hx < 0) return __int_as_float(0x7fc00000);
        k -= 25;
        x = FMUL(x, two25);
        hx = __float_as_uint(x);
    }
    if (hx >= 0x7f800000u) return FADD(x, x);
    k += (int)(hx >> 23) - 127;
    int i = (k < 0) ? 1 : 0;
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    float y = (float)(k + i);
    x = __uint_as_float(hx);
    float z = FADD(FMUL(y, log10_2lo), FMUL(ivln10, gfk_log(x)));
    return FADD(z, FMUL(y, log10_2hi));
}

__device__ __forceinline__ float gfk_scale_exponent(float y, int k)
{
    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));
}

// expm1 in FP32, the word tanh is built on.
__device__ float gfk_expm1(float x)
{
    const float ln2_hi = __uint_as_float(0x3F317180u);
    const float ln2_lo = __uint_as_float(0x3717F7D1u);
    const float invln2 = __uint_as_float(0x3FB8AA3Bu);
    const float q1 = __uint_as_float(0xBD088889u);
    const float q2 = __uint_as_float(0x3AD00D01u);
    const float q3 = __uint_as_float(0xB8A670CDu);
    const float q4 = __uint_as_float(0x36867E54u);
    const float q5 = __uint_as_float(0xB457EDBBu);
    const float tiny = 1.0e-30f;

    unsigned word = __float_as_uint(x);
    unsigned sign = word & 0x80000000u;
    unsigned magnitude = word & 0x7FFFFFFFu;
    if (magnitude >= 0x4195B844u) {
        if (magnitude >= 0x42B17218u) {
            if (magnitude > 0x7F800000u) return FADD(x, x);
            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);
        }
        if (sign != 0u) return FSUB(tiny, 1.0f);
    }
    int k;
    float correction;
    if (magnitude > 0x3EB17218u) {
        float hi, lo;
        if (magnitude < 0x3F851592u) {
            if (sign == 0u) {
                hi = FSUB(x, ln2_hi); lo = ln2_lo; k = 1;
            } else {
                hi = FADD(x, ln2_hi); lo = -ln2_lo; k = -1;
            }
        } else {
            k = (int)FADD(FMUL(invln2, x), sign == 0u ? 0.5f : -0.5f);
            float scale = (float)k;
            hi = FSUB(x, FMUL(scale, ln2_hi));
            lo = FMUL(scale, ln2_lo);
        }
        x = FSUB(hi, lo);
        correction = FSUB(FSUB(hi, x), lo);
    } else if (magnitude < 0x33000000u) {
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    float hfx = FMUL(0.5f, x);
    float hxs = FMUL(x, hfx);
    float r1 = FADD(1.0f, FMUL(hxs, FADD(q1, FMUL(hxs,
        FADD(q2, FMUL(hxs, FADD(q3, FMUL(hxs,
            FADD(q4, FMUL(hxs, q5))))))))));
    float t = FSUB(3.0f, FMUL(r1, hfx));
    float e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));
    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));
    e = FSUB(FMUL(x, FSUB(e, correction)), correction);
    e = FSUB(e, hxs);
    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));
        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));
    }
    float y;
    if (k <= -2 || k > 56) {
        y = FSUB(1.0f, FSUB(e, x));
        y = gfk_scale_exponent(y, k);
        return FSUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));
        y = FSUB(t, FSUB(e, x));
    } else {
        t = __uint_as_float((unsigned)(0x7F - k) << 23);
        y = FSUB(x, FADD(e, t));
        y = FADD(y, 1.0f);
    }
    return gfk_scale_exponent(y, k);
}

// tanh: 1 - 2/(expm1(2|x|)+2) at |x| >= 1, -t/(t+2) below, saturated at 22.
__device__ float gfk_tanh(float x)
{
    const float tiny = 1.0e-30f;
    unsigned word = __float_as_uint(x);
    unsigned magnitude = word & 0x7FFFFFFFu;
    float z;
    if (magnitude >= 0x7F800000u)                // inf -> +-1, NaN -> NaN
        return (word & 0x80000000u) == 0u ? FADD(FDIV(1.0f, x), 1.0f)
                                          : FSUB(FDIV(1.0f, x), 1.0f);
    if (magnitude < 0x41B00000u) {
        if (magnitude < 0x24000000u)
            return FMUL(x, FADD(1.0f, x));
        float ax = __uint_as_float(magnitude);
        if (magnitude >= 0x3F800000u) {
            float t = gfk_expm1(FMUL(2.0f, ax));
            z = FSUB(1.0f, FDIV(2.0f, FADD(t, 2.0f)));
        } else {
            float t = gfk_expm1(FMUL(-2.0f, ax));
            z = FDIV(-t, FADD(t, 2.0f));
        }
    } else {
        z = FSUB(1.0f, tiny);
    }
    return (word & 0x80000000u) == 0u ? z : -z;
}

#endif
