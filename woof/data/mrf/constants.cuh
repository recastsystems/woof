// WRF share/module_model_constants.F values as module_pbl_driver.F passes
// them to MRF (cp, g, rcp, r_d, xlv, r_v, ep_1, karman, p1000mb), as the
// exact float32 words gfortran folds them to.  Gated against a NumPy
// float32 evaluation of WRF's own expressions in tests/test_mrf_port.py.
#define MRF_G       __int_as_float(0x411cf5c3)   // 9.81
#define MRF_RD      __int_as_float(0x438f8000)   // r_d = 287.
#define MRF_CP      __int_as_float(0x447b2000)   // cp = 7.*r_d/2.
#define MRF_RV      __int_as_float(0x43e6cccd)   // r_v = 461.6
#define MRF_ROVCP   __int_as_float(0x3e924925)   // rcp = r_d/cp
#define MRF_EP1     __int_as_float(0x3f1bbda4)   // ep_1 = r_v/r_d-1.
#define MRF_XLV     __int_as_float(0x4a189680)   // xlv = 2.5e6
#define MRF_KARMAN  __int_as_float(0x3ecccccd)   // karman = 0.4
#define MRF_P1000MB 100000.0f

// MRF2D PARAMETERs (module_bl_mrf.F:305-311).  PFAC = 2.0 survives only as
// the folded square; XKA and PRT feed dead stores.
#define MRF_RLAM    150.0f
#define MRF_PRMIN   0.5f
#define MRF_PRMAX   4.0f
#define MRF_XKZMIN  0.01f
#define MRF_XKZMAX  1000.0f
#define MRF_RIMIN   (-100.0f)
#define MRF_BRCR    0.5f
#define MRF_CFAC    7.8f
#define MRF_SFCFRAC 0.1f
#define MRF_CKZ     0.001f
#define MRF_ZFMIN   1.e-8f
#define MRF_APHI5   5.0f
#define MRF_APHI16  16.0f
#define MRF_GAMCRT  3.0f
#define MRF_GAMCRQ  2.e-3f
// The in-cloud test literal 0.01E-3 (:999-1000).
#define MRF_QCLOUD  0.01e-3f
