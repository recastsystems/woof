// WOOF helpers for the pinned Purdue Lin operation order.
#ifndef LIN_KMAX
#define LIN_KMAX 128
#endif
#define LIN_XNOR 8.0e6f
// gfortran MIN/MAX choose the second operand on equality. Keep signed zeros.
template<class T> __device__ T lin_min(T a, T b) { return a < b ? a : b; }
template<class T> __device__ T lin_max(T a, T b) { return a > b ? a : b; }
// Integer powers use the same square-and-multiply chain as the Fortran runtime.
__device__ float lin_powi(float x, int n) {
    unsigned int m = n < 0 ? -n : n;
    float y = (m & 1) ? x : 1.0f;
    while (m >>= 1) {
        x = __fmul_rn(x, x);
        if (m & 1) y = __fmul_rn(y, x);
    }
    return n < 0 ? 1.0f / y : y;
}

#ifdef GPUWM_WRF_EXACT
__device__ float lin_add(float a,float b) { return __fadd_rn(a,b); }
__device__ float lin_sub(float a,float b) { return __fsub_rn(a,b); }
__device__ float lin_mul(float a,float b) { return __fmul_rn(a,b); }
__device__ float lin_div(float a,float b) { return __fdiv_rn(a,b); }
#else
__device__ float lin_add(float a,float b) { return a+b; }
__device__ float lin_sub(float a,float b) { return a-b; }
__device__ float lin_mul(float a,float b) { return a*b; }
// A146: the callers pass literal divisors, which NVVM rewrites into a
// reciprocal multiply on Blackwell under -ftz=true (lin compiles on that
// route); __fdiv_rn is the IEEE quotient there and identical to div.rn for a
// variable divisor.  2.9.0 round 2, device division census.
__device__ float lin_div(float a,float b) { return __fdiv_rn(a,b); }
#endif
