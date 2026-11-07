/* Correctly rounded float32 atanf, atan2f and log10f for the CR oracle arms.
 *
 * The TEMF pair calls eight libm entries from REAL arithmetic (nm -u on the
 * objects): expf, logf, powf, log10f, atanf, atan2f, sinf and cosf.  The C
 * library on the sprint box is glibc 2.39, whose atanf, atan2f and log10f
 * are not correctly rounded; current glibc ships correctly rounded forms of
 * all three, and so does WOOF's own float32 library, which the port calls.
 * Linking this object ahead of libm makes the oracle compute what WRF
 * 4.6.1 computes on a C library that rounds the three correctly.  expf,
 * logf, powf, sinf and cosf are left to the C library.  The plain arm
 * (no shim) is built and reported beside it.
 *
 * Method (the P3 oracle's): the double result, whose error is far below
 * 2^-40 relative, is rounded to float32 directly unless it lies within
 * 2^-40 relative of a float32 rounding boundary or of a representable
 * float; only then is the 113-bit libquadmath value computed and rounded
 * once.  Either way the result is the correctly rounded float32.  This file
 * is the project's own; it consults no libm implementation.
 */
#include <math.h>
#include <quadmath.h>
#include <stdint.h>
#include <string.h>

static int near_boundary(double d)
{
    if (!isfinite(d) || d == 0.0) return 1;
    uint64_t u; memcpy(&u, &d, 8);
    uint64_t low = u & ((1ULL << 29) - 1);
    int64_t dist = (int64_t)low - (int64_t)(1ULL << 28);
    if (dist < 0) dist = -dist;
    if (dist < (1 << 13)) return 1;
    if (low < (1 << 13) || low > (1ULL << 29) - (1 << 13)) return 1;
    return 0;
}

float atanf(float x)
{
    double d = atan((double)x);
    if (!near_boundary(d) || isnan(x))
        return (float)d;
    return (float)atanq((__float128)x);
}

float atan2f(float y, float x)
{
    double d = atan2((double)y, (double)x);
    if (!near_boundary(d) || isnan(x) || isnan(y))
        return (float)d;
    return (float)atan2q((__float128)y, (__float128)x);
}

float log10f(float x)
{
    double d = log10((double)x);
    if (!near_boundary(d) || isnan(x) || x <= 0.0f)
        return (float)d;
    return (float)log10q((__float128)x);
}
