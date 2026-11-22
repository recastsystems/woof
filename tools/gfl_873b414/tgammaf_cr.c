/* Witness only: a correctly rounded tgammaf (libquadmath's 113-bit tgammaq, rounded once to
   binary32) linked ahead of glibc's, to show that the GFL replay's only difference from the
   glibc 2.39 referee is that function. Never part of any product path. */
#include <quadmath.h>
float tgammaf(float x) { return (float)tgammaq((__float128)x); }
