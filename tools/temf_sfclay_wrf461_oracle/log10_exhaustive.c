/* Exhaustive check of WOOF's own binary32 log10 (gpuwm/core/kernels/
   log10_flt32.cuh, compiled here as C) against MPFR's correctly rounded
   log10, on every positive finite binary32 word plus the special words.

   MPFR is reached through the shared library gfortran itself depends on
   (libmpfr.so.6); the box has no MPFR headers, so the five entry points
   used are declared below with MPFR 4's ABI.  A 24-bit target with
   round-to-nearest makes mpfr_log10 return the correctly rounded binary32
   value directly.

   Build: gcc -O2 -ffp-contract=off -fopenmp -I <kernels> log10_exhaustive.c
          -l:libmpfr.so.6 -lm
   Usage: log10_exhaustive [threads]                                       */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "log10_flt32.cuh"

typedef struct { long prec; int sign; long exp; void *d; } woof_mpfr_struct;
typedef woof_mpfr_struct woof_mpfr_t[1];
extern void mpfr_init2(woof_mpfr_struct *, long);
extern void mpfr_clear(woof_mpfr_struct *);
extern int mpfr_set_flt(woof_mpfr_struct *, float, int);
extern int mpfr_log10(woof_mpfr_struct *, const woof_mpfr_struct *, int);
extern float mpfr_get_flt(const woof_mpfr_struct *, int);
#define WOOF_MPFR_RNDN 0

static unsigned int f2u(float x) { unsigned int u; memcpy(&u, &x, 4); return u; }
static float u2f(unsigned int u) { float x; memcpy(&x, &u, 4); return x; }

int main(int argc, char **argv)
{
    unsigned long checked = 0, bad = 0;
    int shown = 0;
    if (argc > 1) omp_set_num_threads(atoi(argv[1]));
    /* special words: +0, -0, +inf, a NaN, a negative */
    {
        float s0 = woof_log10f(0.0f), s1 = woof_log10f(-0.0f);
        float s2 = woof_log10f(INFINITY), s3 = woof_log10f(NAN);
        float s4 = woof_log10f(-2.0f), s5 = woof_log10f(1.0f);
        int ok = isinf(s0) && s0 < 0 && isinf(s1) && s1 < 0 && isinf(s2)
                 && s2 > 0 && isnan(s3) && isnan(s4) && f2u(s5) == 0u;
        printf("special words: %s\n", ok ? "ok" : "WRONG");
        if (!ok) bad++;
    }
#pragma omp parallel reduction(+:checked,bad)
    {
        woof_mpfr_t x, y;
        mpfr_init2(x, 24);
        mpfr_init2(y, 24);
#pragma omp for schedule(dynamic, 1 << 16)
        for (long long u = 1; u < 0x7f800000LL; ++u) {
            float xf = u2f((unsigned int)u);
            float got = woof_log10f(xf);
            float cr;
            mpfr_set_flt(x, xf, WOOF_MPFR_RNDN);
            mpfr_log10(y, x, WOOF_MPFR_RNDN);
            cr = mpfr_get_flt(y, WOOF_MPFR_RNDN);
            ++checked;
            if (f2u(got) != f2u(cr)) {
                ++bad;
#pragma omp critical
                if (shown < 20) {
                    printf("x=%a woof=%a cr=%a\n", xf, got, cr);
                    ++shown;
                }
            }
        }
        mpfr_clear(x);
        mpfr_clear(y);
    }
    printf("positive finite binary32 words: %lu checked, %lu differ from "
           "MPFR's correctly rounded log10\n", checked, bad);
    return bad != 0;
}
