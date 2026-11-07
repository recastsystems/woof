#define KE_TEMIN   1.0e-4f
#define KE_DMIN    1.0e-7f
#define KE_TPEMIN  1.0e-7f
#define KE_TEMAX   50.0f
#define KE_DMAX    50.0f
#define KE_TPEMAX  5.0f
#define KE_CTHETA  0.09f
#define KE_VK      0.4f
#define KE_CMU     0.09f
#define KE_C1      1.44f
#define KE_C2      1.92f
#define KE_C3      1.44f
#define KE_C4      0.27f
#define KE_C5      0.08f
#define KE_SIGEPS  1.3f
#define KE_TH0     300.0f
// surface_bl_pra_ri parameters (module_bl_keps.F:1036-1037)
#define KE_PRMIN   0.25f
#define KE_PRMAX   4.0f
#define KE_SFCFRAC 0.1f
#define KE_RIMIN   (-100.0f)
#define KE_ZFMIN   1.0e-8f
#define KE_APHI5   5.0f
#define KE_APHI16  16.0f
// Folded constants: the words gfortran's constant folding produces.
#define KE_C2M1     0x1.d70a3cp-1f    // c2-1.
#define KE_SQ_C2M1  0x1.eb17d8p-1f    // (c2-1.)**(1./2.)
#define KE_C1M1     0x1.c28f60p-2f    // c1-1.
#define KE_C3M1     0x1.c28f60p-2f    // c3-1.
#define KE_BETA     (-0x1.0aaaacp-1f) // -1./c2
#define KE_INV_AB   0x1.0b2166p+1f    // 1./(alpha+beta), alpha = 1
#define KE_1O1P4    0x1.6db6dcp-1f    // 1/1.4
#define KE_SQRT_CMU 0x1.333334p-2f    // sqrt(c_mu)
#define KE_RVRD_M1  0x1.366940p-1f    // 461./287.-1
#define KE_CONPR    0x1.16872cp-2f    // bfac*vk*sfcfrac
#define KE_4VK      0x1.99999ap+0f    // 4.*vk
#define KE_TKEEPS_LO 0x1.99999ap-7f   // 0.025/2.
#define KE_TKEEPS_HI 0.125f           // 0.25/2.

