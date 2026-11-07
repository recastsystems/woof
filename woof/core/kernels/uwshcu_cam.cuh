// uwshcu_cam.cuh -- the pieces of WRF's CAM support layer the UW shallow
// cumulus (shcu_physics = 2) calls that the UW PBL port (uwpbl_common.cuh,
// uwpbl_wvsat.cuh) did not already carry.  ArWen's own work, Apache-2.0,
// transcribed from WRF v4.6.1 (public domain):
//
//   * erfc: module_cam_error_function.F CALERF_r8 with JINT = 1 (the r8
//     DERFC the scheme reaches through the generic erfc), lines 168-319 of
//     the module, operation for operation.  It is WRF's own Fortran, not a
//     libm routine; its two EXP calls go to the binary64 exp the UW PBL
//     port already calls (uw_exp).
//   * the wv_saturation module words findsp reads (gestbl's assignments
//     from esinti's arguments, module_cam_wv_saturation.F:150-208).
//
// Every binary64 operation goes through uwpbl_common.cuh's rounding-pinned
// R8 vocabulary, so NVRTC's --fmad choice cannot contract any of it.
#ifndef UWSHCU_CAM_CUH
#define UWSHCU_CAM_CUH

// ---- wv_saturation words (gestbl, from esinti(epsilo, latvap, latice,
// rh2o, cpair, tmelt) and its literals) -------------------------------------
#define UWSC_TMIN    173.16      // tmn
#define UWSC_TTRICE  20.0        // trice
#define UWSC_PCF1    5.04469588506e-01
#define UWSC_PCF2   -5.47288442819e+00
#define UWSC_PCF3   -3.67471858735e-01
#define UWSC_PCF4   -8.95963532403e-03
#define UWSC_PCF5   -7.78053686625e-05

// CALERF_r8, JINT=1. The common CAM module also supplies MP11's erf.
__device__ R8 uwsc_erfc(R8 arg) { return cam5_calerf_r8(arg, true); }

#endif  // UWSHCU_CAM_CUH
