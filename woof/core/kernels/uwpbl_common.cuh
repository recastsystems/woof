// uwpbl_common.cuh -- shared device vocabulary of the UW moist-turbulence PBL
// port (WRF v4.7.1 bl_pbl_physics = 9, module_bl_camuwpbl_driver.F and the
// CAM modules it calls).  ArWen's own work, Apache-2.0.
//
// The CAM code computes in binary64 (real(r8)).  Three things make the port
// graded bit for bit against the gfortran -O0 oracle
// (tools/uwpbl_wrf471_oracle):
//
//  1. R8.  Every binary64 operation goes through a rounding-pinned intrinsic
//     (__dadd_rn, __dsub_rn, __dmul_rn, __ddiv_rn, __dsqrt_rn).  NVRTC compiles
//     this project with its default --fmad=true, so a plain `a*b + c` on
//     doubles WOULD be contracted into an FMA that the SSE2 gfortran build
//     never emits; the intrinsics are guaranteed never merged.  R8 wraps a
//     double so that ordinary infix code (`a*b + c`) calls them, which keeps
//     the transcription readable next to the Fortran.
//  2. Fortran's own semantics, measured on the oracle's toolchain (gfortran
//     15.2.0 -O0, glibc 2.43, x86-64; tools/uwpbl_wrf471_oracle/README.md):
//       MAX(a,b) = (a > b || isnan(a)) ? a : b   (ties return b: max(+0,-0)=-0)
//       MIN(a,b) = (a < b || isnan(a)) ? a : b
//       x**n, integer constant n: n = 2 is x*x inline, any other n is libgcc's
//         __powidf2 (square-and-multiply, 1/y for n < 0) -- uw_powi below.
//       x**y, real y (including 2._r8, 3._r8, 0.5_r8): glibc pow, never folded
//         at -O0 -- uw_pow below.
//       A literal without _r8 is a default (single-precision) REAL: its value
//         is the float32 rounding of the decimal, then widened exactly.
//         Write it as R8(1.e-6f), never R8(1.e-6).
//  3. The transcendentals are glibc's words, not CUDA's: exp/log/pow from
//     glibc_flt64.cuh.  cos/acos are correctly rounded (glibc's own binary64
//     cos/acos are LGPL IBM code and are not transcribed); where glibc does
//     not round correctly the port differs, and that residual is measured,
//     not absorbed.
//
// Column storage.  One thread owns one column.  Every per-column array lives
// in a global workspace laid out slot-major with the column index fastest
// (element k of a view at slot s is base[(s + k - 1) * stride + col]), so a
// warp touching the same k of 32 columns reads 32 consecutive doubles.
// Views are 1-based like the Fortran they transcribe: a(1) .. a(n).
#ifndef UWPBL_COMMON_CUH
#define UWPBL_COMMON_CUH

struct R8 {
    double v;
    __device__ __forceinline__ R8() {}
    __device__ __forceinline__ R8(double x) : v(x) {}
};

// CAM columns alone opt into native arithmetic in the default policy.
#if defined(GPUWM_CAM_FAST) && GPUWM_CAM_FAST
__device__ __forceinline__ R8 operator+(R8 a, R8 b) { return R8(a.v + b.v); }
__device__ __forceinline__ R8 operator-(R8 a, R8 b) { return R8(a.v - b.v); }
__device__ __forceinline__ R8 operator*(R8 a, R8 b) { return R8(a.v * b.v); }
__device__ __forceinline__ R8 operator/(R8 a, R8 b) { return R8(a.v / b.v); }
#else
__device__ __forceinline__ R8 operator+(R8 a, R8 b) { return R8(__dadd_rn(a.v, b.v)); }
__device__ __forceinline__ R8 operator-(R8 a, R8 b) { return R8(__dsub_rn(a.v, b.v)); }
__device__ __forceinline__ R8 operator*(R8 a, R8 b) { return R8(__dmul_rn(a.v, b.v)); }
__device__ __forceinline__ R8 operator/(R8 a, R8 b) { return R8(__ddiv_rn(a.v, b.v)); }
#endif
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
#if defined(GPUWM_CAM_FAST) && GPUWM_CAM_FAST
__device__ __forceinline__ R8 uw_sqrt(R8 a) { return R8(sqrt(a.v)); }
#else
__device__ __forceinline__ R8 uw_sqrt(R8 a) { return R8(__dsqrt_rn(a.v)); }
#endif
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
#if defined(GPUWM_CAM_FAST) && GPUWM_CAM_FAST
__device__ __forceinline__ R8 uw_exp(R8 x) { return R8(exp(x.v)); }
__device__ __forceinline__ R8 uw_log(R8 x) { return R8(log(x.v)); }
__device__ __forceinline__ R8 uw_pow(R8 x, R8 y) { return R8(pow(x.v, y.v)); }
#else
__device__ __forceinline__ R8 uw_exp(R8 x) { return R8(glibc_exp(x.v)); }
__device__ __forceinline__ R8 uw_log(R8 x) { return R8(glibc_log(x.v)); }
__device__ __forceinline__ R8 uw_pow(R8 x, R8 y) { return R8(glibc_pow(x.v, y.v)); }
#endif
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

// ---- constants: the oracle's own run-time words (gpuwm/core/uwpbl_constants.py
// documents each; hex so no decimal conversion intervenes) --------------------
#define UW_CPAIR   0x1.f651eb851eb85p+9    // physconst cpair  = 1004.64
#define UW_GRAVIT  0x1.39cc100e6afcdp+3    // physconst gravit = 9.80616
#define UW_RAIR    0x1.1f0ad4eae9222p+8    // physconst rair   = RGAS/MWDAIR
#define UW_ZVIR    0x1.3730a75507baep-1    // physconst zvir   = RWV/RDAIR - 1
#define UW_LATVAP  0x1.314c400000000p+21   // physconst latvap = 2.501e6
#define UW_LATICE  0x1.45e1000000000p+18   // physconst latice = 3.337e5
#define UW_KARMAN  0x1.999999999999ap-2    // physconst karman = 0.4
#define UW_EPSILO  0x1.3e72edbda50a2p-1    // physconst epsilo = MWWV/MWDAIR
#define UW_RH2O    0x1.cd81301343d8ap+8    // physconst rh2o   = RGAS/MWWV
#define UW_TMELT   0x1.1126666666666p+8    // physconst tmelt  = 273.15
#define UW_QMIN_Q  0x1.19799812dea11p-40   // constituents qmin(1) = 1.E-12_r8
#define UW_B123    0x1.9d339a3d1e83dp+1    // eddy_diff b123 = b1**(2/3), glibc pow

// wv_saturation's estbl(1..250), uploaded from uwpbl_constants.ESTBL by the
// launcher before the first launch (UW_ESTBL[i-1] is Fortran estbl(i)).
__constant__ double UW_ESTBL[250];

#endif  // UWPBL_COMMON_CUH
