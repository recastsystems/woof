// Binary64 column vocabulary, derived from ArWen uwpbl_common.cuh.
// ArWen contribution, Apache-2.0. Fast arithmetic is the production default.
// Native table generation and strict replay opt into separately rounded words.
#if defined(OSAS_TABLE_BUILD) || defined(GPUWM_COLUMN_R8_STRICT)
#define COLUMN_R8_ADD(a,b) __dadd_rn(a,b)
#define COLUMN_R8_SUB(a,b) __dsub_rn(a,b)
#define COLUMN_R8_MUL(a,b) __dmul_rn(a,b)
#define COLUMN_R8_DIV(a,b) __ddiv_rn(a,b)
#define COLUMN_R8_EXP glibc_exp
#define COLUMN_R8_LOG glibc_log
#define COLUMN_R8_POW glibc_pow
#else
#define COLUMN_R8_ADD(a,b) ((a)+(b))
#define COLUMN_R8_SUB(a,b) ((a)-(b))
#define COLUMN_R8_MUL(a,b) ((a)*(b))
#define COLUMN_R8_DIV(a,b) ((a)/(b))
#define COLUMN_R8_EXP exp
#define COLUMN_R8_LOG log
#define COLUMN_R8_POW pow
#endif
#ifndef COLUMN_R8_CUH
#define COLUMN_R8_CUH

struct R8 {
    double v;
    __device__ __forceinline__ R8() {}
    __device__ __forceinline__ R8(double x) : v(x) {}
};

__device__ __forceinline__ R8 operator+(R8 a, R8 b) { return R8(COLUMN_R8_ADD(a.v, b.v)); }
__device__ __forceinline__ R8 operator-(R8 a, R8 b) { return R8(COLUMN_R8_SUB(a.v, b.v)); }
__device__ __forceinline__ R8 operator*(R8 a, R8 b) { return R8(COLUMN_R8_MUL(a.v, b.v)); }
__device__ __forceinline__ R8 operator/(R8 a, R8 b) { return R8(COLUMN_R8_DIV(a.v, b.v)); }
__device__ __forceinline__ R8 operator-(R8 a) { return R8(-a.v); }
__device__ __forceinline__ R8& operator+=(R8& a, R8 b) { a = a + b; return a; }
__device__ __forceinline__ R8& operator-=(R8& a, R8 b) { a = a - b; return a; }
__device__ __forceinline__ R8& operator*=(R8& a, R8 b) { a = a * b; return a; }
__device__ __forceinline__ R8& operator/=(R8& a, R8 b) { a = a / b; return a; }
__device__ __forceinline__ bool operator<(R8 a, R8 b) { return a.v < b.v; }
__device__ __forceinline__ bool operator>(R8 a, R8 b) { return a.v > b.v; }
__device__ __forceinline__ bool operator<=(R8 a, R8 b) { return a.v <= b.v; }
__device__ __forceinline__ bool operator>=(R8 a, R8 b) { return a.v >= b.v; }
__device__ __forceinline__ bool operator==(R8 a, R8 b) { return a.v == b.v; }
__device__ __forceinline__ bool operator!=(R8 a, R8 b) { return a.v != b.v; }

__device__ __forceinline__ bool uw_isnan(R8 a) { return a.v != a.v; }
// gfortran 15.2 -O0 MAX/MIN, measured (see header).  Multi-argument forms
// fold left to right: MAX(a,b,c) = uw_max(uw_max(a,b),c).
__device__ __forceinline__ R8 uw_max(R8 a, R8 b) { return (a.v > b.v || a.v != a.v) ? a : b; }
__device__ __forceinline__ R8 uw_min(R8 a, R8 b) { return (a.v < b.v || a.v != a.v) ? a : b; }
__device__ __forceinline__ R8 uw_max(R8 a, R8 b, R8 c) { return uw_max(uw_max(a, b), c); }
__device__ __forceinline__ R8 uw_min(R8 a, R8 b, R8 c) { return uw_min(uw_min(a, b), c); }
__device__ __forceinline__ int uw_imax(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int uw_imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ R8 uw_abs(R8 a) { return R8(fabs(a.v)); }
__device__ __forceinline__ R8 uw_sqrt(R8 a) { return R8(__dsqrt_rn(a.v)); }
// Fortran SIGN(a,b): |a| with the sign bit of b (gfortran copysign).
__device__ __forceinline__ R8 uw_sign(R8 a, R8 b) { return R8(copysign(a.v, b.v)); }
// INT(x): truncation toward zero; AINT(x): the same, kept in binary64.
__device__ __forceinline__ int uw_int(R8 a) { return (int)a.v; }
__device__ __forceinline__ R8 uw_aint(R8 a) { return R8(trunc(a.v)); }
// REAL(i, r8) and float32 -> binary64 widening are exact.
__device__ __forceinline__ R8 uw_real(int i) { return R8((double)i); }
__device__ __forceinline__ R8 uw_widen(float x) { return R8((double)x); }
// binary64 -> default REAL: round to nearest even, as gfortran's cvtsd2ss.
__device__ __forceinline__ float uw_narrow(R8 a) { return __double2float_rn(a.v); }

// x**n for an integer n other than the inline x**2: libgcc __powidf2.
__device__ __forceinline__ R8 uw_powi(R8 x, int m) {
    unsigned int n = m < 0 ? (unsigned int)(-m) : (unsigned int)m;
    R8 y = (n % 2u) ? x : R8(1.0);
    while (n >>= 1) {
        x = x * x;
        if (n % 2u) y = y * x;
    }
    return m < 0 ? R8(1.0) / y : y;
}
__device__ __forceinline__ R8 uw_sq(R8 x) { return x * x; }   // x**2 (integer 2)

// Transcendentals.  glibc_exp/glibc_log/glibc_pow/uw_cos/uw_acos come from
// glibc_flt64.cuh, which the kernel loader places before this header.
__device__ __forceinline__ R8 uw_exp(R8 x) { return R8(COLUMN_R8_EXP(x.v)); }
__device__ __forceinline__ R8 uw_log(R8 x) { return R8(COLUMN_R8_LOG(x.v)); }
__device__ __forceinline__ R8 uw_pow(R8 x, R8 y) { return R8(COLUMN_R8_POW(x.v, y.v)); }
__device__ __forceinline__ R8 uw_cos(R8 x) { return R8(uw_cos(x.v)); }
__device__ __forceinline__ R8 uw_acos(R8 x) { return R8(uw_acos(x.v)); }

// ---- column views and the per-thread workspace ------------------------------
struct V {                        // real(r8) column array, 1-based
    double* p;                    // address of element 1 for this column
    int s;                        // stride between consecutive elements
    __device__ __forceinline__ R8& operator()(int k) const {
        return *reinterpret_cast<R8*>(p + (long long)(k - 1) * s);
    }
};
struct VI {                       // integer column array, 1-based
    int* p;
    int s;
    __device__ __forceinline__ int& operator()(int k) const {
        return p[(long long)(k - 1) * s];
    }
};
typedef VI VL;                    // logical arrays are stored as int 0/1

struct Ws {
    double* r8base;               // this column's slot 0 in the r8 pool
    int* i4base;                  // this column's slot 0 in the i4 pool
    int stride;                   // number of columns sharing the pools
    int r8top, r8cap, i4top, i4cap;
    int* err;                     // set to 1 on pool overflow
    __device__ __forceinline__ V r8(int n) {
        if (r8top + n > r8cap) { *err = 1; r8top = 0; }
        V v{r8base + (long long)r8top * stride, stride};
        r8top += n;
        return v;
    }
    __device__ __forceinline__ VI i4(int n) {
        if (i4top + n > i4cap) { *err = 1; i4top = 0; }
        VI v{i4base + (long long)i4top * stride, stride};
        i4top += n;
        return v;
    }
};
// Automatic arrays: `WsMark m(ws);` at the top of a routine releases every
// view the routine took when it returns.
struct WsMark {
    Ws& w;
    int r8, i4;
    __device__ __forceinline__ explicit WsMark(Ws& ws) : w(ws), r8(ws.r8top), i4(ws.i4top) {}
    __device__ __forceinline__ ~WsMark() { w.r8top = r8; w.i4top = i4; }
};


#endif
