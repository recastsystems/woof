// module_model_constants.F (float32 PARAMETERs, folded in float32 here as
// gfortran folds them).
#define GB_G      9.81f
#define GB_R_D    287.0f
#define GB_CP     (7.0f * GB_R_D / 2.0f)
#define GB_RCP    (GB_R_D / GB_CP)
#define GB_R_V    461.6f
#define GB_SVP1   0.6112f
#define GB_SVP2   17.67f
#define GB_SVP3   29.65f
#define GB_SVPT0  273.15f
#define GB_EP_1   (GB_R_V / GB_R_D - 1.0f)
#define GB_EP_2   (GB_R_D / GB_R_V)
#define GB_XLV    2.5e6f
#define GB_KARMAN 0.4f
// GBM2D PARAMETERs (:425-432).
#define GB_XFR    0.1f
#define GB_AONE   (1.9f * GB_XFR)
#define GB_ATWO   10.0f
#define GB_CZERO  5.869f
#define GB_RSTBL  1.5f
#define GB_VK     0.4f
#define GB_PREF   100000.0f
// my() PARAMETERs (:1148-1152).
#define GB_A1     0.92f
#define GB_B1     16.6f
#define GB_C1     0.08f
#define GB_A2     0.74f
#define GB_B2     10.1f
#define GB_NUK    5.0f
#define GB_KMAX   1000.0f
#define GB_N2MIN  1.e-7f

// roots failure sentinel (module_bl_gbmpbl.F:1064).
#define GB_ROOT_FAILURE (-9.99e33f)
// Remaining float32 source literals, retained without changing their words.
#define GB_LITERAL_0 0.001f
#define GB_LITERAL_1 0.0233f
#define GB_LITERAL_2 0.0f
#define GB_LITERAL_3 0.1f
#define GB_LITERAL_4 0.28f
#define GB_LITERAL_5 0.5f
#define GB_LITERAL_6 0.8f
#define GB_LITERAL_7 1.0f
#define GB_LITERAL_8 1.5f
#define GB_LITERAL_9 1.e-4f
#define GB_LITERAL_10 10.0f
#define GB_LITERAL_11 100.0f
#define GB_LITERAL_12 1e-8f
#define GB_LITERAL_13 1e-9f
#define GB_LITERAL_14 2.0f
#define GB_LITERAL_15 3.0f
#define GB_LITERAL_16 3.25f
#define GB_LITERAL_17 4.0f
#define GB_LITERAL_18 6.0f
#define GB_LITERAL_19 9.0f
