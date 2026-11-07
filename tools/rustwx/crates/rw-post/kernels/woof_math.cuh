// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor: the one maths library, device side (D30).
//
// Statement-for-statement twin of src/math.rs: IEEE + - * /, sqrt and exact
// bit operations only, compiled with --fmad=false, --prec-div=true,
// --prec-sqrt=true and --ftz=false, so every function returns the same bits
// as its Rust twin.  No libdevice transcendental is called.  See
// src/math.rs for the method.

#pragma once

#define WM_SQRT2 1.4142135623730951
#define WM_LN2_HI 6.93147180369123816490e-01
#define WM_LN2_LO 1.90821492927058770002e-10
#define WM_INV_LN2 1.44269504088896338700e+00
#define WM_TWO_OVER_PI 6.36619772367581382433e-01
#define WM_PIO2_1 1.57079632673412561417e+00
#define WM_PIO2_2 6.07710050650619224932e-11
#define WM_PIO2_3 2.02226624871116645580e-21

__device__ __forceinline__ double wm_nan() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ double wm_inf() { return __longlong_as_double(0x7ff0000000000000LL); }
__device__ __forceinline__ float wm_nanf() { return __int_as_float(0x7fc00000); }

#if defined(__HIP_PLATFORM_AMD__)
// Rust's `as i64` saturates, including when range reduction sees a huge
// argument. A C++ floating-to-integer cast outside the range is undefined
// and HIP and NVCC choose different results. Keep the canonical twins equal.
__device__ __forceinline__ long long wm_to_i64(double v) {
    if (v != v) return 0;
    if (v >= 0x1p63) return 0x7fffffffffffffffLL;
    if (v <= -0x1p63) return (-0x7fffffffffffffffLL - 1LL);
    return (long long)v;
}

#endif

__device__ __forceinline__ double wm_floor(double v) {
#if defined(__HIP_PLATFORM_AMD__)
    double t = (double)wm_to_i64(v);
#else
    double t = (double)(long long)v;
#endif
    return t > v ? t - 1.0 : t;
}

__device__ double wm_ln(double x) {
    if (x != x || x < 0.0) return wm_nan();
    if (x == 0.0) return -wm_inf();
    if (x > 1.7976931348623157e308) return x;
    unsigned long long bits = (unsigned long long)__double_as_longlong(x);
    long long e = (long long)((bits >> 52) & 0x7ffULL);
    if (e == 0) {
        double scaled = x * __longlong_as_double((long long)(1023 + 54) << 52);
        bits = (unsigned long long)__double_as_longlong(scaled);
        e = (long long)((bits >> 52) & 0x7ffULL) - 54;
    }
    e -= 1023;
    double m = __longlong_as_double((long long)((bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL));
    if (m > WM_SQRT2) {
        m = m * 0.5;
        e += 1;
    }
    double s = (m - 1.0) / (m + 1.0);
    double s2 = s * s;
    double p = 1.0 / 23.0;
    for (int k = 10; k >= 0; --k) {
        p = p * s2 + 1.0 / (double)(2 * k + 1);
    }
    double lnm = 2.0 * s * p;
    double de = (double)e;
    return de * WM_LN2_HI + (de * WM_LN2_LO + lnm);
}

__device__ double wm_exp(double y) {
    if (y != y) return wm_nan();
    if (y > 709.78) return wm_inf();
    if (y < -745.2) return 0.0;
    double n = wm_floor(y * WM_INV_LN2 + 0.5);
    double r = (y - n * WM_LN2_HI) - n * WM_LN2_LO;
    double p = 1.0;
    for (int k = 13; k >= 1; --k) {
        p = 1.0 + r * p / (double)k;
    }
    long long ni = (long long)n;
    if (ni > 1023) {
        p = p * __longlong_as_double((long long)(1023 + 1023) << 52);
        ni -= 1023;
    } else if (ni < -1022) {
        p = p * __longlong_as_double((long long)(1023 - 1022) << 52);
        ni += 1022;
        if (ni < -1022) {
            p = p * __longlong_as_double((long long)(1023 - 1022) << 52);
            ni += 1022;
        }
    }
    return p * __longlong_as_double((ni + 1023) << 52);
}

__device__ double wm_pow(double x, double y) {
    if (x != x || y != y || x < 0.0) return wm_nan();
    if (x == 0.0) {
        if (y > 0.0) return 0.0;
        if (y == 0.0) return 1.0;
        return wm_inf();
    }
    return wm_exp(y * wm_ln(x));
}

__device__ double wm_cbrt(double x) {
    if (x != x) return wm_nan();
    if (x == 0.0) return 0.0;
    if (x < 0.0) return -wm_exp(wm_ln(-x) / 3.0);
    return wm_exp(wm_ln(x) / 3.0);
}

__device__ __forceinline__ double wm_sqrt(double x) { return __dsqrt_rn(x); }

__device__ __forceinline__ void wm_sin_cos_reduced(double r, double* so, double* co) {
    double r2 = r * r;
    double s = 1.0;
    double c = 1.0;
    for (int k = 9; k >= 1; --k) {
        s = 1.0 - r2 * s / (double)((2 * k) * (2 * k + 1));
        c = 1.0 - r2 * c / (double)((2 * k - 1) * (2 * k));
    }
    *so = r * s;
    *co = c;
}

__device__ __forceinline__ long long wm_reduce_pio2(double x, double* r) {
    double n = wm_floor(x * WM_TWO_OVER_PI + 0.5);
    *r = ((x - n * WM_PIO2_1) - n * WM_PIO2_2) - n * WM_PIO2_3;
#if defined(__HIP_PLATFORM_AMD__)
    return wm_to_i64(n) & 3LL;
#else
    return ((long long)n) & 3LL;
#endif
}

__device__ double wm_sin(double x) {
    if (x != x || x - x != 0.0) return wm_nan();
    double r, s, c;
    long long q = wm_reduce_pio2(x, &r);
    wm_sin_cos_reduced(r, &s, &c);
    if (q == 0) return s;
    if (q == 1) return c;
    if (q == 2) return -s;
    return -c;
}

__device__ double wm_cos(double x) {
    if (x != x || x - x != 0.0) return wm_nan();
    double r, s, c;
    long long q = wm_reduce_pio2(x, &r);
    wm_sin_cos_reduced(r, &s, &c);
    if (q == 0) return c;
    if (q == 1) return -s;
    if (q == 2) return -c;
    return s;
}

__device__ __forceinline__ float wm_to_f32(double v) {
    return (v != v) ? wm_nanf() : __double2float_rn(v);
}

// Binary32 entry points: widen, one binary64 evaluation, one rounding.
__device__ __forceinline__ float woof_expf(float x) { return (x != x) ? x : wm_to_f32(wm_exp((double)x)); }
__device__ __forceinline__ float woof_logf(float x) { return (x != x) ? x : wm_to_f32(wm_ln((double)x)); }
__device__ __forceinline__ float woof_sinf(float x) { return (x != x) ? x : wm_to_f32(wm_sin((double)x)); }
__device__ __forceinline__ float woof_cosf(float x) { return (x != x) ? x : wm_to_f32(wm_cos((double)x)); }
__device__ __forceinline__ float woof_powpos(float x, float y) {
    if (x != x) return x;
    if (y != y) return y;
    return wm_to_f32(wm_pow((double)x, (double)y));
}
