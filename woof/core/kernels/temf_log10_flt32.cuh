// ======================================================================
// woof_log10f: WOOF's own correctly rounded binary32 log10.
//
// Original work, distributed with WOOF under the Apache License 2.0.  It is
// not a transcription of any C library: the method is the textbook
// argument reduction x = 2^k * m, m in [sqrt(1/2), sqrt(2)), and the
// atanh series ln m = 2 (s + s^3/3 + s^5/5 + ...), s = (m-1)/(m+1),
// carried in double-double arithmetic where it matters:
//
//   * s, s^2, s^3 and the s + s^3/3 head are double-double (about 106
//     bits); the s^5 tail is a plain binary64 Horner sum, and its weight is
//     below s^4/5 <= 1.8e-4 of the total, so its rounding error sits near
//     2^-64 of the result;
//   * 1/ln(10) and log10(2) enter as double-double constants;
//   * the double-double result is rounded ONCE to binary32, with the one
//     case binary64 cannot settle by itself -- a high word exactly on a
//     binary32 midpoint -- decided by the sign of the low word.
//
// Correct rounding is a MEASUREMENT, not an argument: the C compilation of
// this very header (it is written in the C subset CUDA and C share) is
// compared with MPFR's correctly rounded log10 on every positive finite
// binary32 word by tools/temf_sfclay_wrf461_oracle/log10_exhaustive.c, and
// the CUDA compilation is compared with the C one in
// tests/test_temf_sfclay_wrf461_parity.py.
//
// The arithmetic is IEEE binary64 add/sub/mul/div plus fma(), pinned with
// the round-to-nearest intrinsics on the device (never contracted, under
// any --fmad setting) and with -ffp-contract=off in the C harness.  No
// binary32 arithmetic touches the argument, so a flush-to-zero build
// changes nothing: a subnormal argument is read from its bits.
//
// Why WOOF needs its own: the TEMF surface layer (sf_sfclay_physics=10)
// takes log10 of zm/z0.  The C library the gfortran WRF build links is not
// correctly rounded for log10f (it misses on 5.7% of the words in [1,10),
// measured in the oracle's log10-probe.txt), and WOOF does not reproduce a
// library's rounding misses; it declares them.
// ======================================================================
#ifndef WOOF_LOG10_FLT32_CUH
#define WOOF_LOG10_FLT32_CUH

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define WOOF_L10_FN __device__ static inline
#define WOOF_L10_ADD(a, b) __dadd_rn((a), (b))
#define WOOF_L10_SUB(a, b) __dsub_rn((a), (b))
#define WOOF_L10_MUL(a, b) __dmul_rn((a), (b))
#define WOOF_L10_DIV(a, b) __ddiv_rn((a), (b))
#define WOOF_L10_FMA(a, b, c) __fma_rn((a), (b), (c))
#else
#include <math.h>
#define WOOF_L10_FN static inline
#define WOOF_L10_ADD(a, b) ((a) + (b))
#define WOOF_L10_SUB(a, b) ((a) - (b))
#define WOOF_L10_MUL(a, b) ((a) * (b))
#define WOOF_L10_DIV(a, b) ((a) / (b))
#define WOOF_L10_FMA(a, b, c) fma((a), (b), (c))
#endif

typedef struct { double hi, lo; } woof_l10_dd;

WOOF_L10_FN unsigned int woof_l10_f2u(float x)
{ union { float f; unsigned int u; } v; v.f = x; return v.u; }

WOOF_L10_FN float woof_l10_u2f(unsigned int u)
{ union { unsigned int u; float f; } v; v.u = u; return v.f; }

WOOF_L10_FN unsigned long long woof_l10_d2u(double x)
{ union { double d; unsigned long long u; } v; v.d = x; return v.u; }

WOOF_L10_FN double woof_l10_u2d(unsigned long long u)
{ union { unsigned long long u; double d; } v; v.u = u; return v.d; }

// |a| >= |b|, or a == 0
WOOF_L10_FN woof_l10_dd woof_l10_fast_two_sum(double a, double b)
{
    woof_l10_dd r;
    r.hi = WOOF_L10_ADD(a, b);
    r.lo = WOOF_L10_SUB(b, WOOF_L10_SUB(r.hi, a));
    return r;
}

WOOF_L10_FN woof_l10_dd woof_l10_two_sum(double a, double b)
{
    woof_l10_dd r;
    double bb;
    r.hi = WOOF_L10_ADD(a, b);
    bb = WOOF_L10_SUB(r.hi, a);
    r.lo = WOOF_L10_ADD(WOOF_L10_SUB(a, WOOF_L10_SUB(r.hi, bb)),
                        WOOF_L10_SUB(b, bb));
    return r;
}

WOOF_L10_FN woof_l10_dd woof_l10_add(woof_l10_dd x, woof_l10_dd y)
{
    woof_l10_dd s = woof_l10_two_sum(x.hi, y.hi);
    woof_l10_dd t = woof_l10_two_sum(x.lo, y.lo);
    s.lo = WOOF_L10_ADD(s.lo, t.hi);
    s = woof_l10_fast_two_sum(s.hi, s.lo);
    s.lo = WOOF_L10_ADD(s.lo, t.lo);
    return woof_l10_fast_two_sum(s.hi, s.lo);
}

WOOF_L10_FN woof_l10_dd woof_l10_mul(woof_l10_dd x, woof_l10_dd y)
{
    double p = WOOF_L10_MUL(x.hi, y.hi);
    double e = WOOF_L10_FMA(x.hi, y.hi, -p);
    e = WOOF_L10_ADD(e, WOOF_L10_ADD(WOOF_L10_MUL(x.hi, y.lo),
                                     WOOF_L10_MUL(x.lo, y.hi)));
    return woof_l10_fast_two_sum(p, e);
}

WOOF_L10_FN woof_l10_dd woof_l10_mul_d(woof_l10_dd x, double y)
{
    double p = WOOF_L10_MUL(x.hi, y);
    double e = WOOF_L10_FMA(x.hi, y, -p);
    e = WOOF_L10_ADD(e, WOOF_L10_MUL(x.lo, y));
    return woof_l10_fast_two_sum(p, e);
}

// n / d for binary64 n, d, to double-double
WOOF_L10_FN woof_l10_dd woof_l10_div_d(double n, double d)
{
    double q = WOOF_L10_DIV(n, d);
    double r = WOOF_L10_FMA(-q, d, n);
    return woof_l10_fast_two_sum(q, WOOF_L10_DIV(r, d));
}

// Round a double-double once to binary32 (round to nearest, ties to even).
WOOF_L10_FN float woof_l10_round(woof_l10_dd r)
{
    float f = (float)r.hi;
    double fd = (double)f;
    if (fd != r.hi && r.lo != 0.0) {
        // r.hi rounded to f.  The only mistake possible is a tie: r.hi
        // exactly halfway between f and its neighbour g on r.hi's side,
        // where the low word decides.
        unsigned int fu = woof_l10_f2u(f);
        int up = r.hi > fd;              // g lies above f
        unsigned int gu = ((f > 0.0f) == (up != 0)) ? fu + 1u : fu - 1u;
        double gd = (double)woof_l10_u2f(gu);
        if (r.hi - fd == gd - r.hi) {    // exact midpoint
            if ((r.lo > 0.0) == (up != 0)) f = woof_l10_u2f(gu);
        }
    }
    return f;
}

WOOF_L10_FN float woof_log10f(float x)
{
    // log10(2) and 1/ln(10) as double-double: hi = the binary64 value,
    // lo = the remainder, each to binary64.
    const woof_l10_dd log10_2 = { 0x1.34413509f79ffp-2, -0x1.9dc1da994fd21p-59 };
    const woof_l10_dd inv_ln10 = { 0x1.bcb7b1526e50ep-2, 0x1.95355baaafad3p-57 };
    const woof_l10_dd third = { 0x1.5555555555555p-2, 0x1.5555555555555p-56 };
    unsigned int ix = woof_l10_f2u(x);
    int k = 0;
    double m;
    woof_l10_dd s, s2, s3, head, ln_half, r;
    double z, tail;

    if (ix == 0x7f800000u) return x;                       // +inf
    if ((ix & 0x7fffffffu) > 0x7f800000u) return x + x;    // NaN
    if ((ix & 0x7fffffffu) == 0u) return woof_l10_u2f(0xff800000u);  // -inf
    if (ix & 0x80000000u) return woof_l10_u2f(0x7fc00000u);         // x < 0
    {
        // x = 2^k * m, m in [1, 2), read from the bits (no binary32
        // arithmetic, so a flush-to-zero build reads a subnormal exactly).
        double xd = (ix < 0x00800000u)
            ? WOOF_L10_MUL((double)ix, 0x1p-149)            // subnormal, exact
            : (double)x;
        unsigned long long bits = woof_l10_d2u(xd);
        k = (int)((bits >> 52) & 0x7ffu) - 1023;
        m = woof_l10_u2d((bits & 0x000fffffffffffffULL)
                         | 0x3ff0000000000000ULL);
    }
    if (m > 0x1.6a09e667f3bcdp+0) {                        // sqrt(2)
        m = WOOF_L10_MUL(m, 0.5);
        k += 1;
    }
    if (m == 1.0) {
        r = woof_l10_mul_d(log10_2, (double)k);
        return woof_l10_round(r);
    }
    // m - 1 and m + 1 are exact in binary64 (m carries 24 bits).
    s = woof_l10_div_d(WOOF_L10_SUB(m, 1.0), WOOF_L10_ADD(m, 1.0));
    s2 = woof_l10_mul(s, s);
    s3 = woof_l10_mul(s2, s);
    head = woof_l10_add(s, woof_l10_mul(s3, third));
    z = s2.hi;
    // P(z) = sum_{j>=0} z^j / (2j+5); |z| <= 0.0295, 13 terms reach 2^-70.
    tail = 1.0 / 29.0;
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 27.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 25.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 23.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 21.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 19.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 17.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 15.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 13.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 11.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 9.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 7.0);
    tail = WOOF_L10_ADD(WOOF_L10_MUL(tail, z), 1.0 / 5.0);
    tail = WOOF_L10_MUL(tail, WOOF_L10_MUL(s3.hi, z));    // s^5 P(s^2)
    {
        woof_l10_dd t;
        t.hi = tail;
        t.lo = 0.0;
        ln_half = woof_l10_add(head, t);                   // ln(m) / 2
    }
    ln_half.hi = WOOF_L10_MUL(ln_half.hi, 2.0);           // ln(m), exact
    ln_half.lo = WOOF_L10_MUL(ln_half.lo, 2.0);
    r = woof_l10_mul(ln_half, inv_ln10);
    if (k != 0) r = woof_l10_add(woof_l10_mul_d(log10_2, (double)k), r);
    return woof_l10_round(r);
}

#endif  // WOOF_LOG10_FLT32_CUH
