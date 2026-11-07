// cam5_support.cuh -- the device half of WOOF's shared CAM5 support layer
// (gpuwm/core/cam5_support.py is the host half).  WOOF's own work,
// Apache-2.0.
//
// The loader places this header after glibc_flt64.cuh, uwpbl_common.cuh and
// uwpbl_wvsat.cuh, which already carry the CAM5 layer's binary64
// vocabulary (R8: every binary64 operation through a rounding-pinned
// intrinsic; uw_max/uw_min with gfortran's MAX/MIN semantics; WOOF's own
// exp/log/pow words), the physconst words the UW PBL uses and
// wv_saturation's table lookups (uw_estblf, uw_aqsat).  What follows is the
// rest of the common layer the other CAM5 schemes need:
//
//   * the physconst words the UW PBL does not read (cpwv, cpliq);
//   * cldwat_fice (module_cam_cldwat.F:190-258);
//   * the WRF-facing float32 vocabulary: CAM's drivers in WRF mix default
//     REAL (float32) WRF arrays with real(r8) CAM state, and each
//     statement keeps its own kinds.  F32 operations go through __f*_rn so
//     no contraction and no flush-to-zero can touch them, and widening /
//     narrowing are the exact conversions Fortran performs.
#ifndef CAM5_SUPPORT_CUH
#define CAM5_SUPPORT_CUH

#define CAM_CPWV   0x1.c480000000000p+10   // physconst cpwv  = 1810.
#define CAM_CPLIQ  0x1.05c0000000000p+12   // physconst cpliq = 4188.

// ---- float32 vocabulary --------------------------------------------------
#if defined(GPUWM_CAM_FAST) && GPUWM_CAM_FAST
__device__ __forceinline__ float cam_fadd(float a, float b) { return a + b; }
__device__ __forceinline__ float cam_fsub(float a, float b) { return a - b; }
__device__ __forceinline__ float cam_fmul(float a, float b) { return a * b; }
__device__ __forceinline__ float cam_fdiv(float a, float b) { return a / b; }
__device__ __forceinline__ float cam_fsqrt(float a) { return sqrtf(a); }
#else
__device__ __forceinline__ float cam_fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float cam_fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float cam_fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float cam_fdiv(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float cam_fsqrt(float a) { return __fsqrt_rn(a); }
#endif
// gfortran MAX/MIN on default REAL: the first argument when it compares
// greater (less), or is NaN; else the second (same rule as uw_max/uw_min).
__device__ __forceinline__ float cam_fmax(float a, float b) { return (a > b || a != a) ? a : b; }
__device__ __forceinline__ float cam_fmin(float a, float b) { return (a < b || a != a) ? a : b; }
// default REAL -> real(r8) (exact) and real(r8) -> default REAL (round to
// nearest even, as gfortran's cvtsd2ss at an assignment).
__device__ __forceinline__ R8 cam_widen(float x) { return R8((double)x); }
__device__ __forceinline__ float cam_narrow(R8 a) { return __double2float_rn(a.v); }

// ---- cldwat_fice (module_cam_cldwat.F:190-258) ---------------------------
// tmax_fice = tmelt - 10, tmin_fice = tmax_fice - 30, tmax_fsnow = tmelt,
// tmin_fsnow = tmelt - 5, all evaluated in binary64 as the WRF port sets
// them on every call (:216-219).
__device__ __forceinline__ void cam_cldwat_fice(R8 t, R8& fice, R8& fsnow) {
    const R8 tmax_fice = R8(UW_TMELT) - R8(10.0);
    const R8 tmin_fice = tmax_fice - R8(30.0);
    const R8 tmax_fsnow = R8(UW_TMELT);
    const R8 tmin_fsnow = R8(UW_TMELT) - R8(5.0);
    if (t > tmax_fice) fice = R8(0.0);
    else if (t < tmin_fice) fice = R8(1.0);
    else fice = (tmax_fice - t) / (tmax_fice - tmin_fice);
    if (t > tmax_fsnow) fsnow = R8(0.0);
    else if (t < tmin_fsnow) fsnow = R8(1.0);
    else fsnow = (tmax_fsnow - t) / (tmax_fsnow - tmin_fsnow);
}

#endif  // CAM5_SUPPORT_CUH
