// gpuwm/core/kernels/pxlsm.cu
//
// The Pleim-Xiu land-surface model, sf_surface_physics = 7, transcribed from
// WRF v4.6.1 phys/module_sf_pxlsm.F (2,003 lines; pinned with its land-use
// tables module_sf_pxlsm_data.F by tools/pxlsm_wrf461_oracle/build.sh).
// One thread is one column, and its scope is exactly the body of pxlsm's
// column loop, with the per-column half of VEGELAND in front of it:
//
//   VEGELAND (land-use weighting, crop seasonality)   :680-1055
//   pxlsm column body                                 :429-667
//     SOILPROP (texture, Noilhan-Mahfouf parameters)  :1730-1923
//     PXSNOW (fractional snow cover, snow albedo)     :1929-1998
//     sub-step loop of SURFPX                         :1059-1503
//       QFLUX (soil, canopy, transpiration fluxes)    :1509-1671
//       SMASS (soil nudging coefficients)             :1677-1723
//
// The reference is WRF AS BUILT: gfortran 13.3 at the scalar wrf.exe's
// flags (-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), graded
// bitwise by tests/test_pxlsm_wrf461_parity.py on the oracle's 13 cases
// x 96 columns x 6 steps.  Under GPUWM_WRF_EXACT=1:
//
//  * every float operation is a rounding intrinsic (PA/PS/PM/PD below), in
//    Fortran's left-to-right order;
//  * EXP, ALOG and REAL**REAL are WOOF's own float32 routines gfk_exp,
//    gfk_log and gfk_pow; SQRT is correctly rounded on both sides; COS is
//    never needed (BETAP, its only use, is read by nothing);
//  * REAL**INTEGER is GCC's -O2 addition chain (px_p2..px_p5);
//  * every constant -- including the ones gfortran folds at compile time,
//    2.0*PI*TAUINV, PR0/KARMAN, EXP(-SALP) and the rest -- is a word of the
//    pxk_bits table, read through a load, so NVRTC never sees an operation
//    on two constants (it folds some such adds with the wrong rounding;
//    urban_ucm.cu's header has the measurement).  The table is generated
//    by tools/pxlsm_wrf461_oracle/constants.py from gfortran's own words.
//
// DECLARED DIVERGENCES (gpuwm/core/pxlsm.py PXLSM_DIVERGENCES):
//  (a) RSTMIN on a column with no leaf area in any category is WRF's 0/0
//      (:997); it takes the dominant category's RSMIN here.
//  (b) WRF reads soil parameters a water column never set (the previous
//      column's); they are defined here, and no output can tell.

__constant__ unsigned int pxk_bits[] = {
    0x411CF5C3u,  //   0 G = g
    0x438F8000u,  //   1 R_D = r_d
    0x3F1F2B09u,  //   2 EP_2 = EP_2
    0x3F1C779Au,  //   3 SVP1 = SVP1
    0x418D5C29u,  //   4 SVP2 = SVP2
    0x41ED3333u,  //   5 SVP3 = SVP3
    0x43889333u,  //   6 SVPT0 = SVPT0
    0x3ECCCCCDu,  //   7 KARMAN = KARMAN
    0x33738BEBu,  //   8 STBOLT = STBOLT
    0x447B2AE1u,  //   9 CPD = CPD
    0x3E924807u,  //  10 ROVCP = ROVCP
    0x3F000000u,  //  11 CRANKP = CRANKP
    0x3F000000u,  //  12 ONE_MINUS_CRANKP = 1.0 - CRANKP
    0x447A0000u,  //  13 DENW = DENW
    0x37422E45u,  //  14 TAUINV = TAUINV
    0x3DCCCCCDu,  //  15 T2TFAC = T2TFAC
    0x40490FDAu,  //  16 PI = PI
    0x3F733333u,  //  17 PR0 = PR0
    0x3D03126Fu,  //  18 CZO = CZO
    0x38D1B717u,  //  19 OZO = OZO
    0x3898825Au,  //  20 TWO_PI_TAUINV = 2.0 * PI * TAUINV
    0x40180000u,  //  21 PR0_OVER_KARMAN = PR0 / KARMAN
    0x40135D8Eu,  //  22 ALN10 = ALOG(10.0)
    0x33738BEBu,  //  23 STBOLT_X1 = STBOLT * (4.0 * CRANKP - 1.0)
    0x42200000u,  //  24 DTPBLX_PLUS = DTPBLX + 0.000001
    0x00000000u,  //  25 C_0 = 0.0
    0x3F800000u,  //  26 C_1 = 1.0
    0x3F000000u,  //  27 C_0P5 = 0.5
    0x447A0000u,  //  28 C_1000 = 1000.0
    0x3A83126Fu,  //  29 C_1EM3 = 1.0E-3
    0x3F570A3Du,  //  30 C_0P84 = 0.84
    0x3A83126Fu,  //  31 C_0P001 = 0.001
    0x3F333333u,  //  32 C_0P7 = 0.7
    0x406AE148u,  //  33 C_3P670 = 3.670
    0x43910000u,  //  34 C_290 = 290.0
    0x3C800000u,  //  35 C_0P015625 = 0.015625
    0x42C80000u,  //  36 C_100 = 100.0
    0x41000000u,  //  37 C_8 = 8.0
    0x38D1B717u,  //  38 C_1EM4 = 0.0001
    0x41B41CACu,  //  39 C_22P514 = 22.514
    0x45C03000u,  //  40 C_6P15E3 = 6.15E3
    0x3F1F3B64u,  //  41 C_0P622 = 0.622
    0x40901893u,  //  42 C_4P503 = 4.503
    0x3BA3D70Au,  //  43 C_0P005 = 0.005
    0x447A0000u,  //  44 C_1000_INT = 1000
    0x3C23D70Au,  //  45 C_0P01 = 0.01
    0x40A00000u,  //  46 ZNOTCMN = ZNOTCMN
    0x41700000u,  //  47 ZNOTCMX = ZNOTCMX
    0x461C3C00u,  //  48 C_9999 = 9999.0
    0x3DCCCCCDu,  //  49 C_0P10 = 0.10
    0x3FC00000u,  //  50 ZOBS = ZOBS
    0x41800000u,  //  51 GAMAH = GAMAH
    0x40A00000u,  //  52 BETAH = BETAH
    0xC0800000u,  //  53 ONE_MINUS_BETAH = 1.-BETAH
    0x3F000000u,  //  54 SIGF = SIGF
    0x37B6B432u,  //  55 DWAT_E4 = DWAT*1.E-4
    0x3749539Cu,  //  56 CV = CV
    0x37A7C5ACu,  //  57 CT_SNOW = CT_SNOW
    0x365B4FD0u,  //  58 CT_IMPERV = CT_IMPERV
    0x40000000u,  //  59 C_2 = 2.0
    0x40A00000u,  //  60 C_5 = 5.0
    0x3FE00000u,  //  61 C_1P75 = 1.75
    0x3FDBE76Du,  //  62 C_1P718 = 1.718
    0x40400000u,  //  63 C_3 = 3.
    0x41600000u,  //  64 C_14 = 14.0
    0x41200000u,  //  65 C_10 = 10.0
    0x4A2CBAC0u,  //  66 C_2P83E6 = 2.83E6
    0x40201062u,  //  67 C_2P501 = 2.501
    0x3B1B5200u,  //  68 C_0P00237 = 0.00237
    0x43889333u,  //  69 C_273P15 = 273.15
    0x49742400u,  //  70 C_1E6 = 1.E6
    0xC37A0000u,  //  71 C_M250 = -250.0
    0x358637BDu,  //  72 C_1EM6 = 1.0E-6
    0x3951B717u,  //  73 C_0P2EM3 = 0.2E-3
    0x2D2FEBFFu,  //  74 C_1EM11 = 1.E-11
    0x3F9851ECu,  //  75 C_1P19 = 1.19
    0x40A2E148u,  //  76 C_5P09 = 5.09
    0x43120000u,  //  77 C_146 = 146.
    0x44DF4000u,  //  78 C_1786 = 1786.
    0x41200000u,  //  79 C_10P = 10.
    0x40800000u,  //  80 C_4 = 4.
    0x41000000u,  //  81 C_8P = 8.0
    0x459C4000u,  //  82 RSMAX = RSMAX
    0x3E800000u,  //  83 F3MIN = F3MIN
    0x43020000u,  //  84 C_130 = 130.0
    0x3CA3D70Au,  //  85 C_0P02 = 0.02
    0x3D8F5C29u,  //  86 C_0P07 = 0.07
    0x3EE66666u,  //  87 C_0P45 = 0.45
    0x40921CACu,  //  88 C_4P566 = 4.566
    0xBADED289u,  //  89 C_M0P0017 = -0.0017
    0xC0A00000u,  //  90 C_M5 = -5.0
    0x40400000u,  //  91 C_3P0 = 3.0
    0x43971333u,  //  92 C_302P15 = 302.15
    0xBED1EB85u,  //  93 C_M0P41 = -0.41
    0x438D0666u,  //  94 C_282P05 = 282.05
    0x439D0000u,  //  95 C_314 = 314.0
    0x41F00000u,  //  96 C_30 = 30.0
    0x3727C5ACu,  //  97 C_1EM5 = 0.00001
    0x40800000u,  //  98 C_4P0 = 4.0
    0xB8D1B717u,  //  99 A1MAX = A1MAX
    0x3727C5ACu,  // 100 A2MAX = A2MAX
    0xBC23D70Au,  // 101 B1MAX = B1MAX
    0x3A83126Fu,  // 102 B2MAX = B2MAX
    0x38422DF4u,  // 103 TASSI = TASSI
    0x41200000u,  // 104 RAMIN = RAMIN
    0x3ED2F1AAu,  // 105 WWLTX_PLUS_WFCX = WWLTX + WFCX
    0x44AB4000u,  // 106 C_1370 = 1370.0
    0x3727C5ACu,  // 107 C_1EM5B = 1.0E-5
    0x42AA0000u,  // 108 C_85 = 85.
    0x428C0000u,  // 109 C_70 = 70.
    0xBF8A3D71u,  // 110 C_M1P08 = -1.08
    0x43F7270Au,  // 111 C_494P305 = 494.305
    0x3D181A08u,  // 112 C_37P1342EM3 = 37.1342E-3
    0x3DB65E1Eu,  // 113 C_89P0467EM3 = 89.0467E-3
    0x3EB2FEC5u,  // 114 C_0P3496 = 0.3496
    0x3E0C49BAu,  // 115 C_0P137 = 0.137
    0x40601062u,  // 116 C_3P501 = 3.501
    0xBC7F1950u,  // 117 C_M1P557EM2 = -1.557E-2
    0x3C6C17ECu,  // 118 C_1P441EM2 = 1.441E-2
    0x4096779Au,  // 119 C_4P7021 = 4.7021
    0x40B28F5Cu,  // 120 C_5P58 = 5.58
    0x42A9C28Fu,  // 121 C_84P88 = 84.88
    0x3C23D70Au,  // 122 C_1EM2 = 1.0E-2
    0x415D0A3Du,  // 123 C_13P815 = 13.815
    0xBF743958u,  // 124 C_M0P954 = -0.954
    0x40AA76C9u,  // 125 C_5P327 = 5.327
    0xBF858106u,  // 126 C_M1P043 = -1.043
    0x3F3B7FE1u,  // 127 C_732P42EM3 = 732.42E-3
    0xBF09FBE7u,  // 128 C_M0P539 = -0.539
    0x3E09374Cu,  // 129 C_0P134 = 0.134
    0x4059999Au,  // 130 C_3P4 = 3.4
    0x3AA137F4u,  // 131 C_0P00123 = 0.00123
    0x3A2D03DAu,  // 132 C_0P00066 = 0.00066
    0x3D25E354u,  // 133 C_0P0405 = 0.0405
    0x3C23D70Au,  // 134 C_0P01B = 0.01
    0x400E38E4u,  // 135 LIQSN_RATIO = DENW / 450.0
    0x3B11A2B4u,  // 136 CONV_WAT2SNOW = (DENW / 450.0) / 1000
    0x4965D580u,  // 137 RHO_CS_SNOWPACK = 450.0 * CS_SNOWPACK
    0xC0266666u,  // 138 C_M2P6 = -SALP
    0x3D981CC0u,  // 139 EXP_MINUS_SALP = EXP(-SALP)
};
#define K_G (__uint_as_float(pxk_bits[0]))
#define K_R_D (__uint_as_float(pxk_bits[1]))
#define K_EP_2 (__uint_as_float(pxk_bits[2]))
#define K_SVP1 (__uint_as_float(pxk_bits[3]))
#define K_SVP2 (__uint_as_float(pxk_bits[4]))
#define K_SVP3 (__uint_as_float(pxk_bits[5]))
#define K_SVPT0 (__uint_as_float(pxk_bits[6]))
#define K_KARMAN (__uint_as_float(pxk_bits[7]))
#define K_STBOLT (__uint_as_float(pxk_bits[8]))
#define K_CPD (__uint_as_float(pxk_bits[9]))
#define K_ROVCP (__uint_as_float(pxk_bits[10]))
#define K_CRANKP (__uint_as_float(pxk_bits[11]))
#define K_ONE_MINUS_CRANKP (__uint_as_float(pxk_bits[12]))
#define K_DENW (__uint_as_float(pxk_bits[13]))
#define K_TAUINV (__uint_as_float(pxk_bits[14]))
#define K_T2TFAC (__uint_as_float(pxk_bits[15]))
#define K_PI (__uint_as_float(pxk_bits[16]))
#define K_PR0 (__uint_as_float(pxk_bits[17]))
#define K_CZO (__uint_as_float(pxk_bits[18]))
#define K_OZO (__uint_as_float(pxk_bits[19]))
#define K_TWO_PI_TAUINV (__uint_as_float(pxk_bits[20]))
#define K_PR0_OVER_KARMAN (__uint_as_float(pxk_bits[21]))
#define K_ALN10 (__uint_as_float(pxk_bits[22]))
#define K_STBOLT_X1 (__uint_as_float(pxk_bits[23]))
#define K_DTPBLX_PLUS (__uint_as_float(pxk_bits[24]))
#define K_C_0 (__uint_as_float(pxk_bits[25]))
#define K_C_1 (__uint_as_float(pxk_bits[26]))
#define K_C_0P5 (__uint_as_float(pxk_bits[27]))
#define K_C_1000 (__uint_as_float(pxk_bits[28]))
#define K_C_1EM3 (__uint_as_float(pxk_bits[29]))
#define K_C_0P84 (__uint_as_float(pxk_bits[30]))
#define K_C_0P001 (__uint_as_float(pxk_bits[31]))
#define K_C_0P7 (__uint_as_float(pxk_bits[32]))
#define K_C_3P670 (__uint_as_float(pxk_bits[33]))
#define K_C_290 (__uint_as_float(pxk_bits[34]))
#define K_C_0P015625 (__uint_as_float(pxk_bits[35]))
#define K_C_100 (__uint_as_float(pxk_bits[36]))
#define K_C_8 (__uint_as_float(pxk_bits[37]))
#define K_C_1EM4 (__uint_as_float(pxk_bits[38]))
#define K_C_22P514 (__uint_as_float(pxk_bits[39]))
#define K_C_6P15E3 (__uint_as_float(pxk_bits[40]))
#define K_C_0P622 (__uint_as_float(pxk_bits[41]))
#define K_C_4P503 (__uint_as_float(pxk_bits[42]))
#define K_C_0P005 (__uint_as_float(pxk_bits[43]))
#define K_C_1000_INT (__uint_as_float(pxk_bits[44]))
#define K_C_0P01 (__uint_as_float(pxk_bits[45]))
#define K_ZNOTCMN (__uint_as_float(pxk_bits[46]))
#define K_ZNOTCMX (__uint_as_float(pxk_bits[47]))
#define K_C_9999 (__uint_as_float(pxk_bits[48]))
#define K_C_0P10 (__uint_as_float(pxk_bits[49]))
#define K_ZOBS (__uint_as_float(pxk_bits[50]))
#define K_GAMAH (__uint_as_float(pxk_bits[51]))
#define K_BETAH (__uint_as_float(pxk_bits[52]))
#define K_ONE_MINUS_BETAH (__uint_as_float(pxk_bits[53]))
#define K_SIGF (__uint_as_float(pxk_bits[54]))
#define K_DWAT_E4 (__uint_as_float(pxk_bits[55]))
#define K_CV (__uint_as_float(pxk_bits[56]))
#define K_CT_SNOW (__uint_as_float(pxk_bits[57]))
#define K_CT_IMPERV (__uint_as_float(pxk_bits[58]))
#define K_C_2 (__uint_as_float(pxk_bits[59]))
#define K_C_5 (__uint_as_float(pxk_bits[60]))
#define K_C_1P75 (__uint_as_float(pxk_bits[61]))
#define K_C_1P718 (__uint_as_float(pxk_bits[62]))
#define K_C_3 (__uint_as_float(pxk_bits[63]))
#define K_C_14 (__uint_as_float(pxk_bits[64]))
#define K_C_10 (__uint_as_float(pxk_bits[65]))
#define K_C_2P83E6 (__uint_as_float(pxk_bits[66]))
#define K_C_2P501 (__uint_as_float(pxk_bits[67]))
#define K_C_0P00237 (__uint_as_float(pxk_bits[68]))
#define K_C_273P15 (__uint_as_float(pxk_bits[69]))
#define K_C_1E6 (__uint_as_float(pxk_bits[70]))
#define K_C_M250 (__uint_as_float(pxk_bits[71]))
#define K_C_1EM6 (__uint_as_float(pxk_bits[72]))
#define K_C_0P2EM3 (__uint_as_float(pxk_bits[73]))
#define K_C_1EM11 (__uint_as_float(pxk_bits[74]))
#define K_C_1P19 (__uint_as_float(pxk_bits[75]))
#define K_C_5P09 (__uint_as_float(pxk_bits[76]))
#define K_C_146 (__uint_as_float(pxk_bits[77]))
#define K_C_1786 (__uint_as_float(pxk_bits[78]))
#define K_C_10P (__uint_as_float(pxk_bits[79]))
#define K_C_4 (__uint_as_float(pxk_bits[80]))
#define K_C_8P (__uint_as_float(pxk_bits[81]))
#define K_RSMAX (__uint_as_float(pxk_bits[82]))
#define K_F3MIN (__uint_as_float(pxk_bits[83]))
#define K_C_130 (__uint_as_float(pxk_bits[84]))
#define K_C_0P02 (__uint_as_float(pxk_bits[85]))
#define K_C_0P07 (__uint_as_float(pxk_bits[86]))
#define K_C_0P45 (__uint_as_float(pxk_bits[87]))
#define K_C_4P566 (__uint_as_float(pxk_bits[88]))
#define K_C_M0P0017 (__uint_as_float(pxk_bits[89]))
#define K_C_M5 (__uint_as_float(pxk_bits[90]))
#define K_C_3P0 (__uint_as_float(pxk_bits[91]))
#define K_C_302P15 (__uint_as_float(pxk_bits[92]))
#define K_C_M0P41 (__uint_as_float(pxk_bits[93]))
#define K_C_282P05 (__uint_as_float(pxk_bits[94]))
#define K_C_314 (__uint_as_float(pxk_bits[95]))
#define K_C_30 (__uint_as_float(pxk_bits[96]))
#define K_C_1EM5 (__uint_as_float(pxk_bits[97]))
#define K_C_4P0 (__uint_as_float(pxk_bits[98]))
#define K_A1MAX (__uint_as_float(pxk_bits[99]))
#define K_A2MAX (__uint_as_float(pxk_bits[100]))
#define K_B1MAX (__uint_as_float(pxk_bits[101]))
#define K_B2MAX (__uint_as_float(pxk_bits[102]))
#define K_TASSI (__uint_as_float(pxk_bits[103]))
#define K_RAMIN (__uint_as_float(pxk_bits[104]))
#define K_WWLTX_PLUS_WFCX (__uint_as_float(pxk_bits[105]))
#define K_C_1370 (__uint_as_float(pxk_bits[106]))
#define K_C_1EM5B (__uint_as_float(pxk_bits[107]))
#define K_C_85 (__uint_as_float(pxk_bits[108]))
#define K_C_70 (__uint_as_float(pxk_bits[109]))
#define K_C_M1P08 (__uint_as_float(pxk_bits[110]))
#define K_C_494P305 (__uint_as_float(pxk_bits[111]))
#define K_C_37P1342EM3 (__uint_as_float(pxk_bits[112]))
#define K_C_89P0467EM3 (__uint_as_float(pxk_bits[113]))
#define K_C_0P3496 (__uint_as_float(pxk_bits[114]))
#define K_C_0P137 (__uint_as_float(pxk_bits[115]))
#define K_C_3P501 (__uint_as_float(pxk_bits[116]))
#define K_C_M1P557EM2 (__uint_as_float(pxk_bits[117]))
#define K_C_1P441EM2 (__uint_as_float(pxk_bits[118]))
#define K_C_4P7021 (__uint_as_float(pxk_bits[119]))
#define K_C_5P58 (__uint_as_float(pxk_bits[120]))
#define K_C_84P88 (__uint_as_float(pxk_bits[121]))
#define K_C_1EM2 (__uint_as_float(pxk_bits[122]))
#define K_C_13P815 (__uint_as_float(pxk_bits[123]))
#define K_C_M0P954 (__uint_as_float(pxk_bits[124]))
#define K_C_5P327 (__uint_as_float(pxk_bits[125]))
#define K_C_M1P043 (__uint_as_float(pxk_bits[126]))
#define K_C_732P42EM3 (__uint_as_float(pxk_bits[127]))
#define K_C_M0P539 (__uint_as_float(pxk_bits[128]))
#define K_C_0P134 (__uint_as_float(pxk_bits[129]))
#define K_C_3P4 (__uint_as_float(pxk_bits[130]))
#define K_C_0P00123 (__uint_as_float(pxk_bits[131]))
#define K_C_0P00066 (__uint_as_float(pxk_bits[132]))
#define K_C_0P0405 (__uint_as_float(pxk_bits[133]))
#define K_C_0P01B (__uint_as_float(pxk_bits[134]))
#define K_LIQSN_RATIO (__uint_as_float(pxk_bits[135]))
#define K_CONV_WAT2SNOW (__uint_as_float(pxk_bits[136]))
#define K_RHO_CS_SNOWPACK (__uint_as_float(pxk_bits[137]))
#define K_C_M2P6 (__uint_as_float(pxk_bits[138]))
#define K_EXP_MINUS_SALP (__uint_as_float(pxk_bits[139]))

// ---------------------------------------------------------------------------
// arithmetic arms
// ---------------------------------------------------------------------------
// STRICT (GPUWM_WRF_EXACT=1): every float operation is a rounding intrinsic,
// so NVRTC cannot contract a*b+c into an FMA gfortran never forms, and the
// order is Fortran's left to right with gfortran's own constant folds (the
// K_* words).  EXP, ALOG and REAL**REAL are WOOF's own float32 routines
// gfk_exp, gfk_log and gfk_pow; SQRT is correctly rounded on both sides.
// DEFAULT uses the same explicit rounding and WOOF float32 math.
// Both modes are held to zero ULP by tests/test_pxlsm_wrf461_parity.py.
// Both modes keep the scalar operation order. The default's unpinned
// arithmetic amplified cancellation in QFLUX's F3 to 16777216 ULP in RS.
// Explicit rounding retires that defect without requiring an opt-in flag.
#define PA(a, b) __fadd_rn((a), (b))
#define PS(a, b) __fsub_rn((a), (b))
#define PM(a, b) __fmul_rn((a), (b))
#define PD(a, b) __fdiv_rn((a), (b))
#define PSQRT(a) __fsqrt_rn(a)
#define PEXP(x) gfk_exp(x)
#define PLOG(x) gfk_log(x)
#define PPOW(x, y) gfk_pow((x), (y))

// gfortran's MAX/MIN/AMAX1/AMIN1 on two finite reals.
__device__ __forceinline__ float px_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float px_min(float a, float b) { return (a < b) ? a : b; }

// REAL**INTEGER at WRF's as-built -O2: gfortran emits __builtin_powif and
// GCC expands it inline as an addition chain (x2 = x*x; x3 = x*x2;
// x4 = x2*x2; x5 = x2*x3).  -O0 would call libgcc's __powisf2 instead,
// which differs for x**5 only (tools/pxlsm_wrf461_oracle/build.sh).
__device__ __forceinline__ float px_p2(float x) { return PM(x, x); }
__device__ __forceinline__ float px_p3(float x) { return PM(x, PM(x, x)); }
__device__ __forceinline__ float px_p4(float x) { float x2 = PM(x, x); return PM(x2, x2); }
__device__ __forceinline__ float px_p5(float x) {
    float x2 = PM(x, x);
    return PM(x2, PM(x, x2));
}

// Land-use dataset of VEGELAND's LAND_USE_TYPE (module_sf_pxlsm.F:365-381).
#define PX_LU_USGS   0
#define PX_LU_USGS28 1
#define PX_LU_NLCD50 2
#define PX_LU_NLCD40 3
#define PX_LU_MODIS  4

// Rows of the packed land-use table (gpuwm/core/pxlsm.py PX_TABLE_ROWS).
#define PXT_RSMIN  0
#define PXT_Z00    1
#define PXT_VEG0   2
#define PXT_VEGMN0 3
#define PXT_LAI0   4
#define PXT_LAIMN0 5
#define PXT_SNUP0  6
#define PXT_ALBF   7
#define PXT_SNOALB 8

// SOILPROP's per-category texture (module_sf_pxlsm.F:1792-1804).
__constant__ unsigned int px_csand_bits[16] = {
    0x42380000u, 0x42200000u, 0x41E80000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x41E80000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x42380000u,
    0x00000000u};
__constant__ unsigned int px_fmsand_bits[16] = {
    0x42380000u, 0x42200000u, 0x41E80000u, 0x41880000u, 0x41200000u,
    0x422C0000u, 0x41E80000u, 0x41200000u, 0x42000000u, 0x42500000u,
    0x40C00000u, 0x41B00000u, 0x422C0000u, 0x422C0000u, 0x42380000u,
    0x42000000u};
__constant__ unsigned int px_clay_bits[16] = {
    0x40400000u, 0x40800000u, 0x41200000u, 0x41500000u, 0x40A00000u,
    0x41900000u, 0x41D80000u, 0x42080000u, 0x42080000u, 0x42280000u,
    0x423C0000u, 0x42680000u, 0x41900000u, 0x41900000u, 0x40400000u,
    0x42080000u};
#define PX_CSAND(s)  (__uint_as_float(px_csand_bits[(s)]))
#define PX_FMSAND(s) (__uint_as_float(px_fmsand_bits[(s)]))
#define PX_CLAY(s)   (__uint_as_float(px_clay_bits[(s)]))

// One category's VEGELAND parameters after the crop seasonality
// (module_sf_pxlsm.F:842-958).  k is 1-based, as in WRF.
struct PxVegK {
    float laimx, laimn, z0, veg, vegmn;
};

__device__ __forceinline__ float px_tab(const float* __restrict__ tab,
                                         int nlcat, int row, int k)
{
    return tab[row * nlcat + (k - 1)];
}

__device__ PxVegK px_veg_k(const float* __restrict__ tab, int nlcat, int k,
                           int lu_kind, bool vegcell, float vseas,
                           float znotc)
{
    PxVegK v;
    v.laimx = px_tab(tab, nlcat, PXT_LAI0, k);
    v.laimn = px_tab(tab, nlcat, PXT_LAIMN0, k);
    v.z0 = px_tab(tab, nlcat, PXT_Z00, k);
    v.veg = px_tab(tab, nlcat, PXT_VEG0, k);
    v.vegmn = px_tab(tab, nlcat, PXT_VEGMN0, k);
    if (!vegcell) return v;
    bool crop = false, z0crop = false, z0mix = false;
    switch (lu_kind) {
    case PX_LU_MODIS:
        crop = (k == 12 || k == 14); z0crop = (k == 12); z0mix = (k == 14);
        break;
    case PX_LU_NLCD50:
        crop = (k == 20 || k == 43 || k == 45);
        z0crop = (k == 20 || k == 43); z0mix = (k == 45);
        break;
    case PX_LU_NLCD40:
        crop = (k == 12 || k == 14 || k == 38);
        z0crop = (k == 12 || k == 38); z0mix = (k == 14);
        break;
    default:   // USGS, USGS28
        crop = (k >= 2 && k <= 6);
        z0crop = (k >= 2 && k <= 4); z0mix = (k >= 5 && k <= 6);
        break;
    }
    if (!crop) return v;
    float one_m = PS(K_C_1, vseas);
    // LAIMX(K) = LAIMN0(K) * (1-VSEAS) + LAI0(K) * VSEAS
    v.laimx = PA(PM(px_tab(tab, nlcat, PXT_LAIMN0, k), one_m),
                 PM(px_tab(tab, nlcat, PXT_LAI0, k), vseas));
    v.laimn = v.laimx;
    v.veg = PA(PM(px_tab(tab, nlcat, PXT_VEGMN0, k), one_m),
               PM(px_tab(tab, nlcat, PXT_VEG0, k), vseas));
    v.vegmn = v.veg;
    if (z0crop) {
        v.z0 = znotc;
    } else if (z0mix) {
        // Z0(K) = 0.5 * (ZNOTC + Z00(K))
        v.z0 = PM(K_C_0P5, PA(znotc, px_tab(tab, nlcat, PXT_Z00, k)));
    }
    return v;
}

// ---------------------------------------------------------------------------
// QFLUX (module_sf_pxlsm.F:1509-1671)
// ---------------------------------------------------------------------------
struct PxQflux {
    float eg, er, etr, cq4, fass, sigg;
};

__device__ PxQflux px_qflux(float dens1, float qv1, float ta1, float rg,
                            float raw, float qss, float vegfrc, float ifland,
                            float lai, float w2, float wr, float rstmin,
                            float wwlt, float wfc, float& rsoil, float rinc,
                            float& rs)
{
    PxQflux q;
    q.er = K_C_0;
    q.etr = K_C_0;
    q.cq4 = K_C_0;
    q.fass = K_C_0;   // read by SMASS only where VEGFRC > 0 wrote it
    q.sigg = K_C_0;   // read by SURFPX only where VEGFRC > 0 wrote it
    if (qss < qv1) rsoil = K_C_0;
    // EG = DENS1*(QSS-QV1)*((1.0-VEGFRC)/(RAW+RSOIL) + VEGFRC/(RAW+RINC+RSOIL))
    q.eg = PM(PM(dens1, PS(qss, qv1)),
              PA(PD(PS(K_C_1, vegfrc), PA(raw, rsoil)),
                 PD(vegfrc, PA(PA(raw, rinc), rsoil))));
    if (ifland < 1.5f && vegfrc > 0.0f) {
        float wrmax = PM(PM(K_C_0P2EM3, vegfrc), lai);
        float delta = (wr <= 0.0f) ? K_C_0 : PD(wr, wrmax);
        q.sigg = (qss >= qv1) ? delta : K_C_1;
        // ER = DENS1 * VEGFRC * SIGG * (QSS - QV1) / RAW
        q.er = PD(PM(PM(PM(dens1, vegfrc), q.sigg), PS(qss, qv1)), raw);

        float f1max = (rstmin > 130.0f) ? PS(K_C_1, PM(K_C_0P02, lai))
                                        : PS(K_C_1, PM(K_C_0P07, lai));
        float par = PM(PM(K_C_0P45, rg), K_C_4P566);
        float f1 = PM(f1max, PS(K_C_1, PEXP(PM(K_C_M0P0017, par))));
        f1 = px_max(f1, PD(rstmin, K_RSMAX));
        float w2avail = PS(w2, wwlt);
        float w2mxav = PS(wfc, wwlt);
        // F2 = 1.0/(1.0+EXP(-5.0*(W2AVAIL/W2MXAV - (W2MXAV/3.0+WWLT))))
        float f2 = PD(K_C_1, PA(K_C_1, PEXP(PM(K_C_M5,
                      PS(PD(w2avail, w2mxav), PA(PD(w2mxav, K_C_3P0), wwlt))))));
        float f4;
        if (ta1 <= 302.15f)
            f4 = PD(K_C_1, PA(K_C_1, PEXP(PM(K_C_M0P41, PS(ta1, K_C_282P05)))));
        else
            f4 = PD(K_C_1, PA(K_C_1, PEXP(PM(K_C_0P5, PS(ta1, K_C_314)))));
        float ftot = PM(PM(PM(lai, f1), f2), f4);

        // FSHELT = 1.0: RSTMIN*FSHELT is RSTMIN.
        float gs = PD(ftot, rstmin);
        float ga = PD(K_C_1, raw);
        // F3 = 0.5*(GS-GA+SQRT(GA*GA+GA*GS*(4.0*QV1/QSS-2.0)+GS*GS))/GS
        float disc = PA(PA(PM(ga, ga),
                           PM(PM(ga, gs), PS(PD(PM(K_C_4P0, qv1), qss), K_C_2))),
                        PM(gs, gs));
        float f3 = PD(PM(K_C_0P5, PA(PS(gs, ga), PSQRT(disc))), gs);
        f3 = px_min(px_max(f3, K_F3MIN), K_C_1);
        rs = PD(K_C_1, PM(gs, f3));
        float fx;
        if (rg < 0.00001f)
            fx = K_C_0;
        else
            fx = PD(PM(PM(PM(K_C_30, f1), f4), lai), rstmin);
        q.fass = fx;
        // ETR = DENS1*VEGFRC*(1.0-SIGG)*(QSS-QV1)/(RAW+RS)
        q.etr = PD(PM(PM(PM(dens1, vegfrc), PS(K_C_1, q.sigg)), PS(qss, qv1)),
                   PA(raw, rs));
        // CQ4 = DENS1*VEGFRC*((1.0-SIGG)/(RAW+RS) + SIGG/RAW)
        q.cq4 = PM(PM(dens1, vegfrc),
                   PA(PD(PS(K_C_1, q.sigg), PA(raw, rs)), PD(q.sigg, raw)));
    }
    return q;
}

// ---------------------------------------------------------------------------
// SMASS (module_sf_pxlsm.F:1677-1723)
// ---------------------------------------------------------------------------
struct PxSmass {
    float alph1, alph2, bet1, bet2, t2nudf;
};

__device__ PxSmass px_smass(float fass, float rg, float vegfrc, float ra,
                            float wwlt, float wfc)
{
    PxSmass m;
    float fbet = fass;
    float falph = PD(rg, K_C_1370);
    float fra = PD(K_RAMIN, ra);
    // FTEXT = TASSI * (WWLT + WFC) / (WWLTX + WFCX) * FRA
    float ftext = PM(PD(PM(K_TASSI, PA(wwlt, wfc)), K_WWLTX_PLUS_WFCX), fra);
    float one_mv = PS(K_C_1, vegfrc);
    m.alph1 = PM(PM(PM(K_A1MAX, falph), one_mv), ftext);
    m.alph2 = PM(PM(PM(K_A2MAX, falph), one_mv), ftext);
    m.bet1 = PM(PM(PM(K_B1MAX, fbet), vegfrc), ftext);
    m.bet2 = PM(PM(PM(K_B2MAX, fbet), vegfrc), ftext);
    // T2NUDF = 1.0E-5*(VEGFRC*MAX((1.0-5.0*FALPH),0.0) + (1-VEGFRC))
    m.t2nudf = PM(K_C_1EM5B,
                  PA(PM(vegfrc, px_max(PS(K_C_1, PM(K_C_5, falph)), K_C_0)),
                     one_mv));
    return m;
}

// ---------------------------------------------------------------------------
// SOILPROP (module_sf_pxlsm.F:1730-1923)
// ---------------------------------------------------------------------------
struct PxSoil {
    float wsat, wfc, wwlt, clay, csand, fmsand, b, cgsat, jp, as, c2r,
          c1sat, wres, c3;
    int isti;
};

__device__ PxSoil px_soilprop(const float* __restrict__ soilcbot, int ncol,
                              int c, int itimestep, float mavail,
                              int smois_init, float& wg, float& w2)
{
    PxSoil s;
    float cfrac = K_C_0, sumcsnd = K_C_0, sumfmsnd = K_C_0, sumcly = K_C_0;
    float tfracbot = K_C_0;
    int isti = 1;
    for (int k = 0; k < 16; ++k) {
        float f = soilcbot[k * ncol + c];
        tfracbot = PA(tfracbot, f);
        sumcsnd = PA(sumcsnd, PM(PX_CSAND(k), f));
        sumfmsnd = PA(sumfmsnd, PM(PX_FMSAND(k), f));
        sumcly = PA(sumcly, PM(PX_CLAY(k), f));
        if (f >= cfrac) {
            isti = k + 1;
            cfrac = f;
        }
    }
    float avcs, avfms, avs, avc, avslt;
    if (tfracbot > 0.001f) {
        avcs = PD(sumcsnd, tfracbot);
        avfms = PD(sumfmsnd, tfracbot);
        avs = PA(avcs, avfms);
        avc = PD(sumcly, tfracbot);
        avslt = PS(PS(K_C_100, avs), avc);
        if (avs > PA(K_C_85, PM(K_C_0P5, avc)))                isti = 1;
        else if (avs > PA(K_C_70, avc))                         isti = 2;
        else if ((avc < 20.0f && avs > 52.0f) ||
                 (avc <= 7.5f && avslt < 50.0f))                isti = 3;
        else if (avc < 35.0f && avs > 45.0f && avslt < 28.0f)   isti = 7;
        else if (avc >= 35.0f && avs > 45.0f)                   isti = 10;
        else if (avc < 27.0f && avslt < 50.0f)                  isti = 6;
        else if (avc < 12.0f && avslt > 80.0f)                  isti = 5;
        else if (avc < 27.0f)                                   isti = 4;
        else if (avc < 40.0f && avs > 20.0f)                    isti = 9;
        else if (avc < 40.0f)                                   isti = 8;
        else if (avslt >= 40.0f)                                isti = 11;
        else                                                    isti = 12;
    } else {
        // no soil fraction at all: clay loam (ISTI = 9)
        isti = 9;
        avcs = PX_CSAND(8);
        avfms = PX_FMSAND(8);
        avs = PA(avcs, avfms);
        avc = PX_CLAY(8);
        avslt = PS(PS(K_C_100, avs), avc);
    }
    s.isti = isti;
    s.csand = avcs;
    s.fmsand = avfms;
    s.clay = avc;
    // Noilhan and Mahfouf (1996) continuous formulation (:1904-1915)
    s.wsat = PM(PA(PM(K_C_M1P08, avs), K_C_494P305), K_C_1EM3);
    s.wwlt = PM(K_C_37P1342EM3, PSQRT(avc));
    s.wfc = PM(K_C_89P0467EM3, PPOW(avc, K_C_0P3496));
    s.b = PA(PM(K_C_0P137, avc), K_C_3P501);
    s.cgsat = PA(PS(PM(K_C_M1P557EM2, avs), PM(K_C_1P441EM2, avc)),
                 K_C_4P7021);
    s.c1sat = PM(PA(PM(K_C_5P58, avc), K_C_84P88), K_C_1EM2);
    s.c2r = PM(K_C_13P815, PPOW(avc, K_C_M0P954));
    s.c3 = PM(K_C_5P327, PPOW(avc, K_C_M1P043));
    s.as = PM(K_C_732P42EM3, PPOW(avc, K_C_M0P539));
    s.jp = PA(PM(K_C_0P134, avc), K_C_3P4);
    float wres = PA(PS(PM(K_C_0P00123, avc), PM(K_C_0P00066, avslt)),
                    K_C_0P0405);
    s.wres = px_max(wres, K_C_0P01B);
    if (itimestep == 1 && smois_init > 0) {
        // WG = FWWLT + (0.5*(FWSAT+FWFC) - FWWLT) * MAVAIL
        float v = PA(s.wwlt, PM(PS(PM(K_C_0P5, PA(s.wsat, s.wfc)), s.wwlt),
                                mavail));
        wg = v;
        w2 = v;
    }
    return s;
}

// ---------------------------------------------------------------------------
// The pxlsm column (module_sf_pxlsm.F:27-673)
// ---------------------------------------------------------------------------
extern "C" __global__ void pxlsm_column(
    const int ncol, const int nx, const int ny,
    const int i_global0, const int j_global0, const int ide, const int jde,
    const int nlcat, const int kwat, const int lu_kind,
    const int itimestep_arg, const int smois_init, const int soil_nudge,
    const int modis_veg, const float dt, const float dtbl,
    const float curr_secs_arg, const int anal_interval,
    const float dzs1, const float dzs2,
    const float* __restrict__ tab,
    const float* __restrict__ landusef, const float* __restrict__ soilcbot,
    // atmosphere: lowest two mass levels
    const float* __restrict__ dz8w1, const float* __restrict__ dz8w2,
    const float* __restrict__ qv3d1, const float* __restrict__ qv3d2,
    const float* __restrict__ t3d1, const float* __restrict__ th3d1,
    const float* __restrict__ rho1,
    // inputs
    const float* __restrict__ psfc, const float* __restrict__ gsw,
    const float* __restrict__ glw, const float* __restrict__ rainbl,
    const float* __restrict__ emiss, const float* __restrict__ shdmin,
    const float* __restrict__ shdmax, const float* __restrict__ pblh,
    const float* __restrict__ rmol, const float* __restrict__ ust,
    const float* __restrict__ mavail, const float* __restrict__ sst,
    const float* __restrict__ t2_ndg_old, const float* __restrict__ t2_ndg_new,
    const float* __restrict__ q2_ndg_old, const float* __restrict__ q2_ndg_new,
    const float* __restrict__ sn_ndg_old, const float* __restrict__ sn_ndg_new,
    const float* __restrict__ vegfra, const float* __restrict__ imperv,
    const float* __restrict__ canfra,
    // carried
    float* __restrict__ ra, float* __restrict__ rs, float* __restrict__ lai,
    float* __restrict__ znt, float* __restrict__ qsfc,
    float* __restrict__ capg, float* __restrict__ canwat,
    float* __restrict__ qfx, float* __restrict__ hfx, float* __restrict__ lh,
    float* __restrict__ psih, float* __restrict__ vegf_px,
    float* __restrict__ snow, float* __restrict__ snoalb,
    float* __restrict__ snowh, float* __restrict__ snowc,
    float* __restrict__ albedo, float* __restrict__ xland,
    float* __restrict__ xice,
    // published
    float* __restrict__ grdflx, float* __restrict__ tsk,
    float* __restrict__ ta2, float* __restrict__ qa2,
    float* __restrict__ t2obs, float* __restrict__ q2obs,
    float* __restrict__ lai_px, float* __restrict__ wwlt_px,
    float* __restrict__ wfc_px, float* __restrict__ wsat_px,
    float* __restrict__ clay_px, float* __restrict__ csand_px,
    float* __restrict__ fmsand_px, int* __restrict__ isltyp,
    // soil, (2, ny, nx)
    float* __restrict__ smois, float* __restrict__ tslb,
    const double* __restrict__ device_clock)
{
    const float curr_secs = device_clock ? (float)device_clock[0] : curr_secs_arg;
    const int itimestep = device_clock ? (int)llround(device_clock[0]/(double)dt)+1 : itimestep_arg;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const int i = c % nx;
    const int j = c / nx;
    const int I = i_global0 + i + 1;     // WRF's one-based I
    const int J = j_global0 + j + 1;

    // ---- VEGELAND for this column (:818-1051) ----------------------------
    float xlai, xlaimn, rstmin, xveg, xvegmn, xsnup, xalb, xsnoalb, wetfra;
    float znt_c, xland_c;
    {
        float sumlai = K_C_0, sumlmn = K_C_0, sumrsi = K_C_0, sumlz0 = K_C_0;
        float sumveg = K_C_0, sumvmn = K_C_0, alai = K_C_0, sumsnup = K_C_0;
        float sumalb = K_C_0, sumsnoalb = K_C_0;
        float vfmx = shdmax[c];
        float vegf = vegfra[c];
        float lwat = landusef[(kwat - 1) * ncol + c];
        bool vegcell = (vfmx > 0.0f && lwat < 1.0f);
        float vseas = K_C_0, znotc = K_C_0;
        if (vegcell) {
            vseas = PD(vegf, vfmx);
            if (vseas > 1.0f || vseas < 0.0f) {
                vseas = px_min(vseas, K_C_1);
                vseas = px_max(vseas, K_C_0);
            }
            // ZNOTC = ZNOTCMN * (1-VSEAS) + ZNOTCMX * VSEAS
            znotc = PA(PM(K_ZNOTCMN, PS(K_C_1, vseas)), PM(K_ZNOTCMX, vseas));
        }
        // The largest category fraction, for the RSTMIN definition below.
        int kdom = 1;
        float fdom = -1.0f;
        for (int k = 1; k <= nlcat; ++k) {
            float farea = landusef[(k - 1) * ncol + c];
            PxVegK v = px_veg_k(tab, nlcat, k, lu_kind, vegcell, vseas, znotc);
            sumlai = PA(sumlai, PM(v.laimx, farea));
            sumlmn = PA(sumlmn, PM(v.laimn, farea));
            alai = PA(alai, farea);
            sumrsi = PA(sumrsi, PD(PM(farea, v.laimx),
                                   px_tab(tab, nlcat, PXT_RSMIN, k)));
            sumlz0 = PA(sumlz0, PM(farea, PLOG(v.z0)));
            sumveg = PA(sumveg, PM(farea, v.veg));
            sumvmn = PA(sumvmn, PM(farea, v.vegmn));
            sumsnup = PA(sumsnup, PM(farea, px_tab(tab, nlcat, PXT_SNUP0, k)));
            sumalb = PA(sumalb, PM(farea, px_tab(tab, nlcat, PXT_ALBF, k)));
            sumsnoalb = PA(sumsnoalb,
                           PM(farea, px_tab(tab, nlcat, PXT_SNOALB, k)));
            if (farea > fdom) { fdom = farea; kdom = k; }
        }
        float fwat = lwat;
        PxVegK w = px_veg_k(tab, nlcat, kwat, lu_kind, vegcell, vseas, znotc);
        if (fwat >= 0.50f) {
            xlai = w.laimx;
            xlaimn = w.laimn;
            rstmin = px_tab(tab, nlcat, PXT_RSMIN, kwat);
            znt_c = w.z0;
            xveg = w.veg;
            xvegmn = w.vegmn;
            xsnup = px_tab(tab, nlcat, PXT_SNUP0, kwat);
            xalb = px_tab(tab, nlcat, PXT_ALBF, kwat);
            xsnoalb = px_tab(tab, nlcat, PXT_SNOALB, kwat);
        } else {
            if (fwat > 0.10f) {
                alai = PS(alai, fwat);
                sumlz0 = PS(sumlz0, PM(fwat, PLOG(w.z0)));
            }
            xlai = PD(sumlai, alai);
            xlaimn = PD(sumlmn, alai);
            // DECLARED DIVERGENCE (gpuwm/core/pxlsm.py PXLSM_DIVERGENCES):
            // WRF forms RSTMIN = SUMLAI/SUMRSI (:997), which is 0/0 = NaN
            // on a column whose every category has zero leaf area (a land
            // column covered by a water category that is not KWAT), and
            // the NaN reaches TSK.  Such a column takes its dominant
            // category's own RSMIN instead.  Every column with leaf area
            // is WRF's word.
            rstmin = (sumrsi == 0.0f) ? px_tab(tab, nlcat, PXT_RSMIN, kdom)
                                      : PD(sumlai, sumrsi);
            znt_c = PEXP(PD(sumlz0, alai));
            xveg = PD(sumveg, alai);
            xvegmn = PD(sumvmn, alai);
            xsnup = sumsnup;
            xalb = sumalb;
            xsnoalb = sumsnoalb;
        }
        if (fwat > 0.50f) {
            znt_c = w.z0;
            xalb = px_tab(tab, nlcat, PXT_ALBF, kwat);
            xsnoalb = px_tab(tab, nlcat, PXT_SNOALB, kwat);
        }
#define LUF(k) (landusef[((k) - 1) * ncol + c])
        switch (lu_kind) {
        case PX_LU_USGS:
        case PX_LU_USGS28:
            wetfra = PA(LUF(17), LUF(18));
            break;
        case PX_LU_NLCD50:
            wetfra = PA(PA(PA(PA(LUF(22), LUF(23)), LUF(27)), LUF(28)),
                        LUF(42));
            break;
        case PX_LU_NLCD40:
            wetfra = PA(PA(LUF(39), LUF(40)), LUF(11));
            break;
        default:
            wetfra = LUF(11);
            break;
        }
        znt_c = PM(znt_c, K_C_0P01);
        xveg = PM(xveg, K_C_0P01);
        xvegmn = PM(xvegmn, K_C_0P01);
        xland_c = PA(K_C_1, fwat);
        xalb = PM(xalb, K_C_0P01);
        xsnoalb = PM(xsnoalb, K_C_0P01);
        float fimp = PM(imperv[c], K_C_0P01);
        float fcan = PM(canfra[c], K_C_0P01);
        if (lu_kind == PX_LU_NLCD40) {
            xveg = px_min(xveg, PS(K_C_1, fimp));
            xvegmn = px_min(xvegmn, PS(K_C_1, fimp));
            xveg = px_max(xveg, fcan);
            xvegmn = px_max(xvegmn, fcan);
            float forfra = PA(PA(PA(LUF(39), LUF(30)), LUF(29)), LUF(28));
            float extfor = PS(fcan, forfra);
            if (extfor >= 0.01f) {
                PxVegK f30 = px_veg_k(tab, nlcat, 30, lu_kind, vegcell, vseas,
                                      znotc);
                float one_mx = PS(K_C_1, extfor);
                xlai = PA(PM(f30.laimx, extfor), PM(xlai, one_mx));
                xlaimn = PA(PM(f30.laimn, extfor), PM(xlaimn, one_mx));
            }
        }
#undef LUF
    }
    znt[c] = znt_c;
    xland[c] = xland_c;

    // ---- main column body (:429-667) -------------------------------------
    float ifland = xland_c;
    float wg = smois[c];
    float w2 = smois[ncol + c];
    float tg = tslb[c];
    float t2s = tslb[ncol + c];
    PxSoil sp;
    if (ifland < 1.5f) {
        sp = px_soilprop(soilcbot, ncol, c, itimestep, mavail[c], smois_init,
                         wg, w2);
        isltyp[c] = sp.isti;
    } else {
        isltyp[c] = 14;
        sp.wwlt = 0.1f;
        sp.wfc = K_C_1;
        sp.wsat = K_C_1;
        sp.clay = K_C_0;
        sp.csand = K_C_0;
        sp.fmsand = K_C_0;
        // WRF leaves these unset on a water column and passes whatever the
        // previous column held.  Over open water no statement reads them;
        // on a water column turned to ice below they reach only
        // CG = CGSAT*1E-6*(WSAT/W2CG)**(0.5*B/ALN10) with WSAT = W2CG = 1,
        // whose value is CGSAT*1E-6 for every B.  Defined here.
        sp.b = K_C_0; sp.cgsat = K_C_0; sp.jp = K_C_0; sp.as = K_C_0;
        sp.c2r = K_C_0; sp.c1sat = K_C_0; sp.wres = K_C_0; sp.c3 = K_C_0;
        sp.isti = 14;
    }
    wwlt_px[c] = sp.wwlt;
    wfc_px[c] = sp.wfc;
    wsat_px[c] = sp.wsat;
    clay_px[c] = PM(sp.clay, K_C_0P01);
    csand_px[c] = PM(sp.csand, K_C_0P01);
    fmsand_px[c] = PM(sp.fmsand, K_C_0P01);

    const float sfcprs = PD(psfc[c], K_C_1000);
    const float ta1 = t3d1[c];
    const float dens1 = rho1[c];
    const float qv1 = qv3d1[c];
    const float qv2 = qv3d2[c];
    const float zlvl = PM(K_C_0P5, dz8w1[c]);
    const float zf1 = dz8w1[c];
    const float za2 = PA(zf1, PM(K_C_0P5, dz8w2[c]));
    const float lwdn = glw[c];
    const float emissi = emiss[c];
    const float precip = px_max(PD(PM(K_C_1EM3, rainbl[c]), dtbl), K_C_0);
    float wr = PM(K_C_1EM3, canwat[c]);
    const float theta1 = th3d1[c];
    float snobs = snow[c];
    float rh2obs = K_C_0;
    float t2obs_c = t2obs[c];

    if (soil_nudge == 1) {
        // CORB/CORE (:399-409)
        float dt_fdda = (float)anal_interval;
        float tba = fmodf(curr_secs, dt_fdda);
        float corb, core;
        if (tba == 0.0f) {
            corb = K_C_1;
            core = K_C_0;
        } else {
            core = PD(tba, dt_fdda);
            corb = PS(K_C_1, core);
        }
        float t2_old = t2_ndg_old[c];
        float t2_new = t2_ndg_new[c];
        float vapprs = PM(K_SVP1, PEXP(PD(PM(K_SVP2, PS(t2_old, K_SVPT0)),
                                         PS(t2_old, K_SVP3))));
        float qsbt = PD(PM(K_EP_2, vapprs), PS(sfcprs, vapprs));
        float rh2_old = PD(q2_ndg_old[c], qsbt);
        vapprs = PM(K_SVP1, PEXP(PD(PM(K_SVP2, PS(t2_new, K_SVPT0)),
                                   PS(t2_new, K_SVP3))));
        qsbt = PD(PM(K_EP_2, vapprs), PS(sfcprs, vapprs));
        float rh2_new = PD(q2_ndg_new[c], qsbt);
        rh2obs = PA(PM(corb, rh2_old), PM(core, rh2_new));
        t2obs_c = PA(PM(corb, t2_old), PM(core, t2_new));
        t2obs[c] = t2obs_c;
        q2obs[c] = PA(PM(corb, q2_ndg_old[c]), PM(core, q2_ndg_new[c]));
        snobs = PA(PM(corb, sn_ndg_old[c]), PM(core, sn_ndg_new[c]));
    }

    const float ust_c = ust[c];
    const float ustar = px_max(ust_c, K_C_0P005);
    float znt_now = znt_c;
    if (ifland >= 1.5f) {
        // ZNT = CZO * UST * UST / G + OZO
        znt_now = PA(PD(PM(PM(K_CZO, ust_c), ust_c), K_G), K_OZO);
        znt[c] = znt_now;
    }
    const float z0 = znt_now;
    const float cpair = PM(K_CPD, PA(K_C_1, PM(K_C_0P84, qv1)));

    // ---- PXSNOW (:1929-1998) ---------------------------------------------
    float snoalb_c = xsnoalb;
    snoalb[c] = snoalb_c;
    float hc_snow, snow_fra, snowc_c, albedo_c;
    {
        float sneqv = PM(snobs, K_C_0P001);
        float snow_c = snobs;
        float snowh_c = PM(snow_c, K_CONV_WAT2SNOW);
        snowc_c = (snowh_c > 0.005f) ? K_C_1 : K_C_0;
        hc_snow = PM(K_RHO_CS_SNOWPACK, snowh_c);
        if (sneqv < xsnup) {
            float rsnow = PD(sneqv, xsnup);
            // SNOW_FRA = 1. - (EXP(-SALP*RSNOW) - RSNOW*EXP(-SALP))
            snow_fra = PS(K_C_1, PS(PEXP(PM(K_C_M2P6, rsnow)),
                                    PM(rsnow, K_EXP_MINUS_SALP)));
        } else {
            snow_fra = K_C_1;
        }
        snowc_c = snow_fra;
        albedo_c = PA(xalb, PM(snowc_c, PS(snoalb_c, xalb)));
        snow[c] = snow_c;
        snowh[c] = snowh_c;
    }

    // ---- sea ice and cold water (:529-544) --------------------------------
    float xice_c = xice[c];
    float lai_c = lai[c];
    float vegf_c = vegf_px[c];
    float laipx_c = lai_px[c];
    if (xice_c >= 0.5f || (sst[c] <= 270.0f && xland_c >= 1.50f)) {
        xland[c] = K_C_1;
        ifland = K_C_1;
        znt[c] = K_C_0P001;
        wg = K_C_1;
        w2 = K_C_1;
        xice_c = K_C_1;
        albedo_c = K_C_0P7;
        snowc_c = K_C_1;
        snow_fra = K_C_1;
        vegf_c = K_C_0;
        laipx_c = K_C_0;
        lai_c = K_C_0;
        sp.cgsat = K_C_3P670;
    }

    // ---- seasonal LAI and vegetation fraction (:550-592) ------------------
    {
        float t2i = t2s;
        float d = PS(K_C_290, t2i);
        float fseas = px_max(PS(K_C_1, PM(K_C_0P015625, px_p2(d))), K_C_0);
        if (t2i >= 290.0f) fseas = K_C_1;
        laipx_c = PA(xlaimn, PM(fseas, PS(xlai, xlaimn)));
        vegf_c = PA(xvegmn, PM(fseas, PS(xveg, xvegmn)));
        if (modis_veg == 1) {
            float vf = vegfra[c];
            if (vf > 0.0f)
                laipx_c = PD(lai_c, PD(vf, K_C_100));
            else
                laipx_c = K_C_0;
            vegf_c = PD(vf, K_C_100);
            float lw = landusef[(kwat - 1) * ncol + c];
            if (lw < 1.0f)
                vegf_c = PD(vegf_c, PS(K_C_1, lw));
            else
                vegf_c = K_C_0;
        }
        laipx_c = px_min(laipx_c, K_C_8);
        laipx_c = px_max(laipx_c, K_C_1EM4);
        vegf_c = px_min(vegf_c, K_C_1);
        vegf_c = px_max(vegf_c, K_C_1EM4);
        if (ifland >= 1.5f) {
            vegf_c = K_C_0;
            laipx_c = K_C_0;
        }
    }
    lai[c] = lai_c;
    xice[c] = xice_c;

    const float gsw_c = gsw[c];
    const float soldn = PD(gsw_c, PS(K_C_1, albedo_c));
    const float isnow = snowc_c;

    int nudge = soil_nudge;
    if (J <= 2 || J >= (jde - 1)) nudge = 0;
    if (I <= 2 || I >= (ide - 1)) nudge = 0;

    float molx;
    const float rmol_c = rmol[c];
    if (rmol_c > 0.0f)
        molx = px_min(PD(K_C_1, rmol_c), K_C_1000);
    else if (rmol_c < 0.0f)
        molx = px_max(PD(K_C_1, rmol_c), -K_C_1000);
    else
        molx = K_C_1000_INT;

    // ZFUNC = ZF1 * (1.0 - ZF1 / MAX(100.,PBLH(I,J))) ** 2
    const float zfunc = PM(zf1, px_p2(PS(K_C_1, PD(zf1, px_max(K_C_100, pblh[c])))));
    // QST12 = KARMAN * ZFUNC*(QV2-QV1) / (ZA2-ZLVL)
    const float qst12 = PD(PM(PM(K_KARMAN, zfunc), PS(qv2, qv1)), PS(za2, zlvl));

    const int ntsps = (int)(PA(PD(dt, K_DTPBLX_PLUS), K_C_1));
    const float dtpbl = PD(dt, (float)ntsps);

    // SURFPX's carried values
    float psih_c = psih[c], ra_c = ra[c], rs_c = rs[c];
    float grdflx_c = grdflx[c], hfx_c = hfx[c], qfx_c = qfx[c], lh_c = lh[c];
    float capg_c = capg[c], ta2_c = ta2[c], qa2_c = qa2[c];
    const float vegfrc = vegf_c;
    const float laiv = laipx_c;
    const float impv = imperv[c];
    const float ds1 = dzs1, ds2 = dzs2;

    for (int it = 1; it <= ntsps; ++it) {
        float es;
        if (tg <= 273.15f)
            es = PM(K_SVP1, PEXP(PS(K_C_22P514, PD(K_C_6P15E3, tg))));
        else
            es = PM(K_SVP1, PEXP(PD(PM(K_SVP2, PS(tg, K_SVPT0)),
                                    PS(tg, K_SVP3))));
        const float qss = PD(PM(es, K_C_0P622), PS(sfcprs, es));
        // BETAP (:632-635) reaches QFLUX and nothing in QFLUX reads it.

        // ---- SURFPX (:1059-1503) ------------------------------------------
        // RADNET (:1228) is written to a pxlsm local no statement reads.
        const float cpot = PPOW(PD(K_C_100, sfcprs), K_ROVCP);
        const float thetag = PM(tg, cpot);
        const float zol = PD(zlvl, molx);
        const float zobol = PD(K_ZOBS, molx);
        const float zntol = PD(z0, molx);
        float psih15, psiob, phih;
        if (molx < 0.0f) {
            float y = PPOW(PS(K_C_1, PM(K_GAMAH, zol)), K_C_0P5);
            float y0 = PPOW(PS(K_C_1, PM(K_GAMAH, zobol)), K_C_0P5);
            float ynt = PPOW(PS(K_C_1, PM(K_GAMAH, zntol)), K_C_0P5);
            psih15 = PM(K_C_2, PLOG(PD(PA(y, K_C_1), PA(y0, K_C_1))));
            psih_c = PM(K_C_2, PLOG(PD(PA(y, K_C_1), PA(ynt, K_C_1))));
            psiob = PM(K_C_2, PLOG(PD(PA(y0, K_C_1), PA(ynt, K_C_1))));
            phih = PD(K_C_1, y);
        } else {
            float d = PS(zol, zntol);
            if (d <= 1.0f)
                psih_c = PM(-K_BETAH, d);
            else
                psih_c = PS(K_ONE_MINUS_BETAH, d);
            float d0 = PS(zobol, zntol);
            if (d0 <= 1.0f)
                psiob = PM(-K_BETAH, d0);
            else
                psiob = PS(K_ONE_MINUS_BETAH, d0);
            psih15 = PS(psih_c, psiob);
            if (zol <= 1.0f)
                phih = PA(K_C_1, PM(K_BETAH, zol));
            else
                phih = PA(K_BETAH, zol);
        }
        // RA=PR0*(ALOG(Z1/ZNT)-PSIH)/(KARMAN*UST)
        ra_c = PD(PM(K_PR0, PS(PLOG(PD(zlvl, z0)), psih_c)),
                  PM(K_KARMAN, ustar));
        const float rah = PA(ra_c, PD(K_C_5, ustar));
        const float raw = PA(ra_c, PD(K_C_4P503, ustar));
        float rsoil;
        if (ifland < 1.5f && xice_c < 0.5f) {
            // LDRY = 1.75*DS1*(EXP((1.-WG/WSAT)**5)-1.)/1.718
            float x = PS(K_C_1, PD(wg, sp.wsat));
            float ldry = PD(PM(PM(K_C_1P75, ds1), PS(PEXP(px_p5(x)), K_C_1)),
                            K_C_1P718);
            // DP = DWAT*1.E-4 * WSAT**2 * (1.-WRES/WSAT)**(2.+3./B)
            float dp = PM(PM(K_DWAT_E4, px_p2(sp.wsat)),
                          PPOW(PS(K_C_1, PD(sp.wres, sp.wsat)),
                               PA(K_C_2, PD(K_C_3, sp.b))));
            rsoil = PD(ldry, dp);
        } else {
            rsoil = K_C_0;
        }
        if (ifland < 1.5f) {
            // WETSAT = 1.00*WSAT; SM2 = WETFRA*WETSAT; W2 = AMAX1(SM2, W2)
            float sm2 = PM(wetfra, sp.wsat);
            w2 = px_max(sm2, w2);
        }
        const float hcan = PM(z0, K_C_10);
        const float rinc = PD(PM(PM(K_C_14, laiv), hcan), ustar);

        PxQflux q = px_qflux(dens1, qv1, ta1, soldn, raw, qss, vegfrc, ifland,
                             laiv, w2, wr, rstmin, sp.wwlt, sp.wfc, rsoil,
                             rinc, rs_c);
        const float et = PA(PA(q.eg, q.er), q.etr);
        const float qst = PD(-et, PM(dens1, ustar));
        float lv = K_C_2P83E6;
        if (isnow < 0.5f && tg > 273.15f)
            lv = PM(PS(K_C_2P501, PM(K_C_0P00237, PS(tg, K_C_273P15))), K_C_1E6);
        qfx_c = et;
        lh_c = PM(lv, qfx_c);

        const float tst = PD(PS(theta1, thetag), PM(ustar, rah));
        const float hf = PM(ustar, tst);
        hfx_c = px_max(PM(PM(-dens1, cpair), hf), K_C_M250);

        const float qst1 = PM(K_C_0P5, PA(qst, PD(qst12, phih)));
        // TA2 = (THETAG + TST*(PR0/KARMAN*(ALOG(ZOBS/ZNT)-PSIOB)+5.))/CPOT
        ta2_c = PD(PA(thetag, PM(tst, PA(PM(K_PR0_OVER_KARMAN,
                                            PS(PLOG(PD(K_ZOBS, z0)), psiob)),
                                         K_C_5))),
                   cpot);
        // QA2 = QV1 - QST1*PR0/KARMAN*(ALOG(Z1/ZOBS)-PSIH15)
        qa2_c = PS(qv1, PM(PD(PM(qst1, K_PR0), K_KARMAN),
                           PS(PLOG(PD(zlvl, K_ZOBS)), psih15)));
        if (qa2_c <= 0.0f) qa2_c = qv1;
        const float vapprs = PM(K_SVP1, PEXP(PD(PM(K_SVP2, PS(ta2_c, K_SVPT0)),
                                               PS(ta2_c, K_SVP3))));
        const float qsbt = PD(PM(K_EP_2, vapprs), PS(sfcprs, vapprs));
        const float rh2mod = PD(qa2_c, qsbt);

        float ct = K_C_0;
        if (ifland < 1.5f) {
            float w2cg = px_max(w2, sp.wwlt);
            // CG = CGSAT*1.0E-6*(WSAT/W2CG)**(0.5*B/ALN10)
            float cg = PM(PM(sp.cgsat, K_C_1EM6),
                          PPOW(PD(sp.wsat, w2cg), PD(PM(K_C_0P5, sp.b), K_ALN10)));
            float imf = px_max(K_C_0, PD(impv, K_C_100));
            float vegf = PM(PS(K_C_1, imf), vegfrc);
            float soilf = PM(PS(K_C_1, imf), PS(K_C_1, vegfrc));
            ct = PD(K_C_1, PA(PA(PD(imf, K_CT_IMPERV), PD(vegf, K_CV)),
                              PD(soilf, cg)));
            ct = PD(K_C_1, PA(PD(snow_fra, K_CT_SNOW),
                              PD(PS(K_C_1, snow_fra), ct)));
            capg_c = PD(K_C_1, ct);
            float soilflx = PM(K_TWO_PI_TAUINV, PS(tg, t2s));
            grdflx_c = PD(soilflx, ct);
        }

        float wgnudg = K_C_0, w2nudg = K_C_0, t2nud = K_C_0;
        if (ifland < 1.5f && nudge != 0) {
            PxSmass m = px_smass(q.fass, soldn, vegfrc, ra_c, sp.wwlt, sp.wfc);
            float dtn = PS(t2obs_c, ta2_c);
            float drh = PS(rh2obs, rh2mod);
            wgnudg = PA(PM(m.alph1, dtn), PM(PM(m.alph2, drh), K_C_100));
            w2nudg = PA(PM(m.bet1, dtn), PM(PM(m.bet2, drh), K_C_100));
            if (w2 >= sp.wfc) w2nudg = px_min(w2nudg, K_C_0);
            if (w2 <= sp.wwlt) w2nudg = px_max(w2nudg, K_C_0);
            if (w2 >= sp.wfc) wgnudg = px_min(wgnudg, K_C_0);
            if (w2 <= sp.wwlt) wgnudg = px_max(wgnudg, K_C_0);
            t2nud = PM(m.t2nudf, dtn);
        }

        if (ifland < 1.5f) {
            const float cq1 = PM(PS(K_C_1, PD(PM(PM(K_C_0P622, lv), K_CRANKP),
                                             PM(K_R_D, tg))),
                                 qss);
            const float cq2 = PD(PM(PM(PM(K_C_0P622, lv), qss), K_CRANKP),
                                 PM(PM(K_R_D, tg), tg));
            const float cq3bg = PD(PM(dens1, PS(K_C_1, vegfrc)), PA(raw, rsoil));
            const float cq3vw = PD(PM(PM(dens1, vegfrc), q.sigg), raw);
            const float cq3vg = PD(PM(dens1, vegfrc), PA(PA(raw, rsoil), rinc));
            const float cq3 = PA(PA(cq3bg, cq3vw), cq3vg);
            const float dc = PM(PD(PM(dens1, cpair), rah), cpot);   // DENS1*CPAIR/RAH*CPOT
            const float coeffnp1 = PA(
                PA(K_C_1, PM(PM(dtpbl, K_CRANKP),
                             PA(PA(PM(PM(PM(PM(K_C_4, emissi), K_STBOLT),
                                         px_p3(tg)), ct),
                                   PM(dc, ct)),
                                K_TWO_PI_TAUINV))),
                PM(dtpbl, PM(PM(PM(ct, lv), cq2), PA(cq3, q.cq4))));
            // STBOLT*(4.0*CRANKP-1.0)*TG*TG*TG*TG: the leading product folds
            // to STBOLT and the powers are left to right.
            const float rad4 = PM(PM(PM(PM(K_STBOLT_X1, tg), tg), tg), tg);
            const float cq1m = PS(cq1, qv1);
            const float coeffn = PS(
                PM(ct, PS(PA(PA(gsw_c, PM(emissi, PA(rad4, lwdn))),
                             PM(PD(PM(dens1, cpair), rah),
                                PS(theta1, PM(K_ONE_MINUS_CRANKP, thetag)))),
                          PM(lv, PA(PM(cq3, cq1m), PM(q.cq4, cq1m))))),
                PM(K_TWO_PI_TAUINV, PS(PM(K_ONE_MINUS_CRANKP, tg), t2s)));
            float tsnew = PD(PA(tg, PM(dtpbl, coeffn)), coeffnp1);
            if (xice_c > 0.5f) tsnew = px_min(tsnew, K_C_273P15);
            const float tshlf = PM(K_C_0P5, PA(tsnew, tg));
            const float dtt = PM(PM(dtpbl, K_TAUINV), K_T2TFAC);
            const float t2new = PD(
                PA(PA(t2s, PM(dtt, PS(tshlf, PM(K_ONE_MINUS_CRANKP, t2s)))),
                   PM(dtpbl, t2nud)),
                PA(K_C_1, PM(dtt, K_CRANKP)));
            tg = tsnew;
            t2s = t2new;
        }

        float pg = K_C_0, w2hlf = K_C_0;
        if (ifland < 1.5f && xice_c < 0.5f) {
            float wrmax = PM(PM(K_C_0P2EM3, vegfrc), laiv);
            float pc, wrnew;
            if (wrmax > 0.0f) {
                pc = PM(PM(vegfrc, K_SIGF), precip);
                float dwr = PD(PS(wrmax, wr), dtpbl);
                float pnet = PS(pc, PD(q.er, K_DENW));
                if (pnet > dwr) {
                    float roff = PS(pnet, dwr);
                    pc = PS(pc, roff);
                }
                if (qss < qv1) {
                    float tendwr = PS(pc, PD(q.er, K_DENW));
                    wrnew = PA(wr, PM(dtpbl, tendwr));
                } else {
                    float cof1 = PD(PM(PM(PD(dens1, K_DENW), vegfrc),
                                       PS(qss, qv1)), raw);
                    float cfnp1wr = PA(K_C_1, PD(PM(PM(dtpbl, cof1), K_CRANKP),
                                                 wrmax));
                    float cfnwr = PS(pc, PD(PM(PM(cof1, K_ONE_MINUS_CRANKP), wr),
                                            wrmax));
                    wrnew = PD(PA(wr, PM(dtpbl, cfnwr)), cfnp1wr);
                }
            } else {
                pc = K_C_0;
                wrnew = K_C_0;
            }
            pg = PM(K_DENW, PS(precip, pc));
            // TENDW2 = 1.0/(DENW*DS2)*(PG-EG-ETR) - C3/DS2*TAUINV*AMAX1(0.0,(W2-WFC))
            //          + (W2NUDG+WGNUDG)/DS2
            float tendw2 = PA(
                PS(PM(PD(K_C_1, PM(K_DENW, ds2)), PS(PS(pg, q.eg), q.etr)),
                   PM(PM(PD(sp.c3, ds2), K_TAUINV),
                      px_max(K_C_0, PS(w2, sp.wfc)))),
                PD(PA(w2nudg, wgnudg), ds2));
            float w2new = PA(w2, PM(dtpbl, tendw2));
            w2new = px_min(w2new, sp.wsat);
            w2new = px_max(w2new, sp.wres);
            w2hlf = PM(K_C_0P5, PA(w2, w2new));
            w2 = w2new;
            wr = px_min(wrmax, wrnew);
        }

        if (ifland < 1.5f && xice_c < 0.5f) {
            if (isnow > 0.5f) {
                wg = sp.wsat;
            } else {
                float w2rel = PD(w2hlf, sp.wsat);
                float c1;
                if (wg > sp.wwlt) {
                    c1 = PM(PM(ds1, sp.c1sat),
                            PPOW(PD(sp.wsat, wg), PA(PM(K_C_0P5, sp.b), K_C_1)));
                } else {
                    float zy2 = PM(sp.c1sat,
                                   PPOW(PD(sp.wsat, sp.wwlt),
                                        PA(PM(K_C_0P5, sp.b), K_C_1)));
                    float c1max = PA(PS(PM(PS(PM(K_C_1P19, sp.wwlt), K_C_5P09), tg),
                                        PM(K_C_146, sp.wwlt)),
                                     K_C_1786);
                    c1max = px_max(px_max(c1max, zy2), K_C_10P);
                    float zly = PLOG(PD(c1max, K_C_10P));
                    float zza = -PLOG(PD(zy2, K_C_10P));
                    float zzb = PM(PM(K_C_2, sp.wwlt), zly);
                    float zdel = PM(PM(PM(K_C_4, PA(zly, zza)), zly),
                                    px_p2(sp.wwlt));
                    float za = PD(PA(-zzb, PSQRT(zdel)), PM(K_C_2, zza));
                    float zb = PD(px_p2(za), zly);
                    c1 = PM(PM(ds1, c1max),
                            PEXP(-PD(px_p2(PS(wg, za)), zb)));
                }
                float c2 = PD(PM(sp.c2r, w2hlf), PA(PS(sp.wsat, w2hlf), K_C_1EM11));
                float weq;
                if (w2hlf >= sp.wsat) {
                    weq = sp.wsat;
                } else {
                    weq = PS(w2hlf, PM(PM(PM(sp.as, sp.wsat), PPOW(w2rel, sp.jp)),
                                      PS(K_C_1, PPOW(w2rel, PM(K_C_8P, sp.jp)))));
                }
                float cfnp1 = PA(K_C_1, PM(PM(PM(dtpbl, c2), K_TAUINV), K_CRANKP));
                float cfn = PA(PS(PM(PD(c1, PM(K_DENW, ds1)), PS(pg, q.eg)),
                                  PM(PM(c2, K_TAUINV),
                                     PS(PM(K_ONE_MINUS_CRANKP, wg), weq))),
                               PD(wgnudg, ds1));
                float wgnew = px_max(PD(PA(wg, PM(dtpbl, cfn)), cfnp1), sp.wres);
                wg = px_min(wgnew, sp.wsat);
            }
        }
    }

    // ---- closing updates (:659-667) ----------------------------------------
    if (ifland >= 1.5f) {
        tsk[c] = sst[c];
        grdflx_c = K_C_0;
    } else {
        tsk[c] = tg;
    }
    canwat[c] = PM(wr, K_C_1000);
    const float raw_out = PA(ra_c, PD(K_C_4P503, ustar));
    qsfc[c] = PA(PD(PM(qfx_c, raw_out), dens1), qv1);

    psih[c] = psih_c;
    ra[c] = ra_c;
    rs[c] = rs_c;
    grdflx[c] = grdflx_c;
    hfx[c] = hfx_c;
    qfx[c] = qfx_c;
    lh[c] = lh_c;
    capg[c] = capg_c;
    ta2[c] = ta2_c;
    qa2[c] = qa2_c;
    lai_px[c] = laipx_c;
    vegf_px[c] = vegf_c;
    snowc[c] = snowc_c;
    albedo[c] = albedo_c;
    smois[c] = wg;
    smois[ncol + c] = w2;
    tslb[c] = tg;
    tslb[ncol + c] = t2s;
}
