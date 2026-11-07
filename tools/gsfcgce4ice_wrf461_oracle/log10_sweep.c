// Sweep every non-negative float32: does the box's log10f equal the
// decomposition WOOF's kernel uses on top of logf?
//   k = exponent, m = mantissa scaled to [1,2) or [0.5,1) by sign of k,
//   z = y*lo + ivln10*logf(m);  result = z + y*hi
// Build: gcc -O2 -ffp-contract=off -fno-builtin log10_sweep.c -lm -fopenmp
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static float own_log10f(float x)
{
    const float two25 = bits2f(0x4c000000u);
    const float ivln10 = bits2f(0x3ede5bd9u);
    const float log10_2hi = bits2f(0x3e9a2080u);
    const float log10_2lo = bits2f(0x355427dbu);
    int32_t hx = (int32_t)f2bits(x);
    int32_t k = 0;
    if (hx < 0x00800000) {
        if ((hx & 0x7fffffff) == 0) return -two25 / fabsf(x);
        if (hx < 0) return (x - x) / (x - x);
        k -= 25;
        x *= two25;
        hx = (int32_t)f2bits(x);
    }
    if (hx >= 0x7f800000) return x + x;
    k += (hx >> 23) - 127;
    int32_t i = (int32_t)(((uint32_t)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffff) | ((0x7f - i) << 23);
    float y = (float)(k + i);
    x = bits2f((uint32_t)hx);
    volatile float lg = logf(x);
    float a = y * log10_2lo;
    float b = ivln10 * lg;
    float z = a + b;
    float c = y * log10_2hi;
    return z + c;
}

int main(void)
{
    long long bad = 0;
    uint32_t first = 0;
#pragma omp parallel for reduction(+:bad) schedule(static, 65536)
    for (long long u = 0; u <= 0x7f800000LL; ++u) {
        float x = bits2f((uint32_t)u);
        volatile float ref = log10f(x);
        float got = own_log10f(x);
        if (f2bits(ref) != f2bits(got)) {
            ++bad;
            if (first == 0) first = (uint32_t)u;
        }
    }
    printf("non-negative float32 words: %lld mismatches (first 0x%08x)\n",
           bad, first);
    return bad != 0;
}
