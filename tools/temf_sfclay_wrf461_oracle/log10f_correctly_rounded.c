/* Oracle arm C only: a correctly rounded binary32 log10, interposed for the
   C library's log10f when the WRF object is linked, so the arm measures
   WRF's arithmetic with the one libm word WOOF deliberately does not share
   replaced by its correctly rounded value.  The extended-precision value is
   rounded once; a result that lands too close to a binary32 rounding
   boundary to decide is refused rather than guessed. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
float log10f(float x) {
    long double r = log10l((long double)x);
    float f = (float)r;
    if (isfinite(f) && f != 0.0f) {
        long double up = (long double)nextafterf(f, INFINITY);
        long double dn = (long double)nextafterf(f, -INFINITY);
        long double mid_hi = ((long double)f + up) / 2.0L;
        long double mid_lo = ((long double)f + dn) / 2.0L;
        long double tol = fabsl(r) * 0x1p-60L;
        if (fabsl(r - mid_hi) < tol || fabsl(r - mid_lo) < tol) {
            fprintf(stderr, "log10f(%a): too close to a rounding boundary\n", x);
            abort();
        }
    }
    return f;
}
