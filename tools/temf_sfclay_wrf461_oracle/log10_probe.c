/* Measures the float32 log10 the gfortran oracle links against, word for word
   against the correctly rounded value (long double log10l, rounded once).
   Range: every float in [lo, hi). Prints counts of words that differ. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
int main(int argc, char **argv) {
    float lo = argc > 1 ? strtof(argv[1], 0) : 1.0f;
    float hi = argc > 2 ? strtof(argv[2], 0) : 1.0e7f;
    uint32_t a, b; memcpy(&a, &lo, 4); memcpy(&b, &hi, 4);
    unsigned long n = 0, bad = 0; int shown = 0;
    for (uint32_t u = a; u < b; ++u) {
        float x; memcpy(&x, &u, 4);
        volatile float got = log10f(x);
        float cr = (float)log10l((long double)x);
        ++n;
        if (memcmp((const void*)&got, &cr, 4) != 0) {
            ++bad;
            if (shown < 10) { printf("x=%a got=%a cr=%a\n", x, got, cr); ++shown; }
        }
    }
    printf("range [%g,%g): %lu words, %lu differ from correctly rounded\n", lo, hi, n, bad);
    return 0;
}
