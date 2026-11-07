/* Correctly rounded float32 gamma, interposed over the C library's tgammaf
 * for the PINNED-GAMMA arm of the BMJ column oracle.
 *
 * Why it exists.  BMJDRV's convective cloud fraction (bmj_rad_feedback
 * arm, module_cu_bmj.F:308) is PAVG**JPR / GAMMA(JPR+1.), and gfortran binds
 * the GAMMA intrinsic to the C library's tgammaf, which is not correctly
 * rounded.  WOOF's own gamma (gfk_tgamma) is correctly rounded on purpose --
 * a standing owner ruling, the same declared divergence Grell-Freitas
 * carries (docs/gf_gamma_known_delta.md).  So the stock oracle and the port
 * are expected to differ on CCLDFRA/QCCONV/QICONV/CONVCLD exactly where the
 * library's gamma is off by an ULP or more, and nowhere else.
 *
 * Linking this file into a second executable replaces tgammaf by symbol
 * interposition (the module source is untouched) with the binary64 gamma
 * rounded once to float32.  The binary64 result is within a few binary64
 * ULP of the true value, so its float32 rounding is the correctly rounded
 * answer unless the true value sits within that distance of a float32
 * rounding midpoint; the guard below aborts the oracle in that case rather
 * than guess.  The port must match THIS arm at 0 ULP on every field.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float tgammaf(float x)
{
    double d = tgamma((double)x);
    float f = (float)d;
    /* midpoint guard: distance from d to the nearest float32 midpoint */
    float lo = nextafterf(f, -INFINITY), hi = nextafterf(f, INFINITY);
    double m1 = 0.5 * ((double)f + (double)lo), m2 = 0.5 * ((double)f + (double)hi);
    double near = fmin(fabs(d - m1), fabs(d - m2));
    if (isfinite(d) && near < 64.0 * fabs(nextafter(d, INFINITY) - d)) {
        fprintf(stderr, "cr_tgammaf: %a is within 64 binary64 ULP of a float32 midpoint; refusing\n", x);
        abort();
    }
    return f;
}
