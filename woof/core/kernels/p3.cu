// P3 (Predicted Particle Properties) microphysics, WRF mp_physics=50/51.
//
// Transcription authority: WRF phys/module_mp_p3.F, sha256
// 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6
// (7,391 lines, version_p3 = 4.5.2; byte-identical in WRF v4.6.1, v4.7.1
// and v4.8.0).  Every ":NNNN" is a line in that file.  The CPU float32
// authority gpuwm/core/p3.py carries the prose for each transcribed quirk;
// this file is the same statement order on the device and does not repeat it.
//
// LAYOUT -- decided before a line of this file was written.
//   * One THREAD per COLUMN.  k is a loop index inside the thread, exactly
//     as the authority's kloop runs inside its i-loop.  Sedimentation's
//     substep while-loop, its Courant reduction, the top/bottom searches
//     and the two column-scope logical flags are then plain sequential
//     code rather than warp reductions, so the arithmetic ORDER is the
//     authority's by construction and not by care.
//   * Every field is LEVEL-MAJOR: element (k, i) sits at k*ncol + i, which
//     is gpuwm's native (nz, ny, nx) storage read as (nk, ncol).  At a
//     fixed k a warp touches 32 consecutive floats, so every access is
//     coalesced AND no transpose is needed anywhere.  The opposite choice
//     (column-contiguous, the shape the host code used) gives every thread
//     its own cache line per load and forces a transpose per step.
//   * itab keeps the authority (densize, rimsize, isize, tabsize) shape
//     with tabsize FASTEST, so the 7..14 quantities one cell needs sit in
//     one 56-byte run.  itabcoll keeps (.., rcollsize, 2) with the two
//     collection quantities adjacent, for the same reason.
//
// PRECISION.  float32 throughout, with the authority two explicit double
// excursions kept double: the dexp saturation-relaxation factor (:3202,
// :3204, :3231) and the rain-freezing log-space products (:3025-3029).
// Contraction is OFF (-fmad=false), matching the Fortran reference arm
// -ffp-contract=off, so plain infix operators ARE the _rn intrinsics;
// tests/test_p3_cuda.py proves that equivalence instead of assuming it.
//
// TRANSCENDENTALS. The replay routines are shared through
// noahmp_leaves.cu. Fast two-category runs use CUDA's REAL32 leaves for
// ordinary REAL expressions; strict builds retain every replay routine.
// The explicit native double excursions above are unchanged in both.
// All builds request --ftz=false. Strict keeps --fmad=false; fast uses
// the normal numerics boundary's contraction policy.
//
// INTEGER POWERS.  Where the authority writes x**2, x**3 or x**6 with an
// INTEGER exponent, gfortran expands to multiplications and never calls
// powf.  Measured on gfortran 15.2.0 -O0 over 200,000 samples:
// x**2 == x*x, x**3 == (x*x)*x, x**6 == ((x*x)*(x*x))*(x*x), all exactly,
// 0 differing.  A REAL exponent (x**0.54, x**thrd, lamc**bcn where bcn is
// a REAL variable) does go through powf.  Both forms appear below and they
// are not interchangeable: glibc powf(x,6.0f) differs from the
// multiplication chain on 66.6% of arguments, up to 3 ULP.
//
// SCOPE. One or two ice categories, with native predicted-Nc and
// three-moment-ice flags. WRF selects three-moment ice only with one
// category. NCAT templates preserve the category loops and collection
// pass; log_3momentIce preserves the native closure, Z update and
// sedimentation branches in this same kernel family.

// ---------------------------------------------------------------------
// p3_init constants (:177-294), emitted as exact float32 hex literals from
// gpuwm.core.p3 -- the CPU authority whose 52 comparable p3_init constants
// the p3-fortref lane measured bit-identical to the running Fortran.  They
// are NOT recomputed here, so no compile-time folding difference can move
// them.  tests/test_p3_cuda.py re-derives this block and asserts equality.
// ---------------------------------------------------------------------
#define P3_PI                  0x1.921fb60000000p+1f  // 3.1415927410125732       :178
#define P3_THRD                0x1.5555560000000p-2f  // 0.3333333432674408       :180
#define P3_SXTH                0x1.5555560000000p-3f  // 0.1666666716337204       :181
#define P3_PIOV3               0x1.0c15240000000p+0f  // 1.0471975803375244       :182
#define P3_PIOV6               0x1.0c15240000000p-1f  // 0.5235987901687622       :183
#define P3_MAX_TOTAL_NI        0x1.e848000000000p+20f // 2000000.0                :186
#define P3_NCCNST              0x1.7d78400000000p+27f // 200000000.0              :196
#define P3_CP                  0x1.f680000000000p+9f  // 1005.0                   :203
#define P3_INV_CP              0x1.04d6fe0000000p-10f // 0.0009950249223038554    :204
#define P3_G                   0x1.3a1cac0000000p+3f  // 9.815999984741211        :205
#define P3_RD                  0x1.1f26660000000p+8f  // 287.1499938964844        :206
#define P3_RV                  0x1.cd82900000000p+8f  // 461.510009765625         :207
#define P3_EP_2                0x1.3e76c80000000p-1f  // 0.621999979019165        :208
#define P3_RHOSUR              0x1.4662840000000p+0f  // 1.2749407291412354       :209
#define P3_RHOSUI              0x1.a69ad80000000p-1f  // 0.8254001140594482       :210
#define P3_F1R                 0x1.8f5c280000000p-1f  // 0.7799999713897705       :213
#define P3_F2R                 0x1.47ae140000000p-2f  // 0.3199999928474426       :214
#define P3_RHOW                0x1.f400000000000p+9f  // 1000.0                   :216
#define P3_CPW                 0x1.07a0000000000p+12f // 4218.0                   :217
#define P3_INV_RHOW            0x1.0624de0000000p-10f // 0.0010000000474974513    :218
#define P3_MU_R_CONSTANT       0.0f                   // 0.0                      :219
#define P3_INV_DRMAX           0x1.f3fffe0000000p+8f  // 499.9999694824219        :222
#define P3_RHO_RIMEMIN         0x1.9000000000000p+5f  // 50.0                     :225
#define P3_RHO_RIMEMAX         0x1.c200000000000p+9f  // 900.0                    :226
#define P3_INV_RHO_RIMEMAX     0x1.2345680000000p-10f // 0.0011111111380159855    :227
#define P3_QSMALL              0x1.6849b80000000p-47f // 9.9999998245167e-15      :230
#define P3_NSMALL              0x1.cd2b2a0000000p-54f // 1.0000000168623835e-16   :231
#define P3_BSMALL              0x1.99ed7a0000000p-57f // 1.1111111022744057e-17   :232
#define P3_BIMM                0x1.0000000000000p+1f  // 2.0                      :239
#define P3_AIMM                0x1.4ccccc0000000p-1f  // 0.6499999761581421       :240
#define P3_MI0                 0x1.0fa6980000000p-48f // 3.769911561201343e-15    :242
#define P3_ECI                 0x1.0000000000000p-1f  // 0.5                      :244
#define P3_ERI                 0x1.0000000000000p+0f  // 1.0                      :245
#define P3_BCN                 0x1.0000000000000p+1f  // 2.0                      :246
#define P3_NMLTRATIO           0x1.0000000000000p+0f  // 1.0                      :251
#define P3_CONS1               0x1.05cca60000000p+9f  // 523.5988159179688        :259
#define P3_CONS2               0x1.05cca60000000p+12f // 4188.79052734375         :260
#define P3_CONS3               0x1.c758900000000p+33f // 15278874624.0            :261
#define P3_CONS5               0x1.0c15240000000p+0f  // 1.0471975803375244       :263
#define P3_CONS6               0x1.1227dc0000000p+9f  // 548.3114013671875        :264
#define P3_CONS7               0x1.2dd58a0000000p-48f // 4.188790105925802e-15    :265
#define P3_E0                  0x1.319eb60000000p+9f  // 611.2399291992188        :257 polysvp1(273.15,0)
#define P3_XXLV                0x1.314d540000000p+21f // 2501034.5                :2325
#define P3_XXLS                0x1.5a09740000000p+21f // 2834734.5                :2326
#define P3_XLF                 0x1.45e1000000000p+18f // 333700.0                 :2327 xxls-xxlv
// --- SEAM 2: aerosol activation, the two PRESCRIBED lognormal modes as
// --- DATA (:269-294).  A prognostic aerosol species replaces these values
// --- and p3_activate_droplets below, and touches nothing else.
#define P3_MW                  0x1.26e9780000000p-6f  // 0.017999999225139618     :269
#define P3_RR                  0x1.0a32ca0000000p+3f  // 8.318699836730957        :275
#define P3_BACT                0x1.4efb0a0000000p-1f  // 0.6542590260505676       :276
#define P3_INV_RM1             0x1.312d000000000p+24f // 20000000.0               :281
#define P3_SIG1                0x1.0000000000000p+1f  // 2.0                      :282
#define P3_NANEW1              0x1.1e1a300000000p+28f // 300000000.0              :283
#define P3_INV_RM2             0x1.7799d80000000p+19f // 769230.75                :290
#define P3_SIG2                0x1.4000000000000p+1f  // 2.5                      :291
#define P3_NANEW2              0.0f                   // 0.0                      :292

// Lookup-table dimensions (:48-57).
#define P3_ISIZE     50
#define P3_IISIZE    25
#define P3_DENSIZE    5
#define P3_RIMSIZE    4
#define P3_RCOLLSIZE 30
#define P3_TABSIZE   14
// Three-moment ice (:50, :55, :90, :233, :254, :2155).  zsmall is the
// float32 rounding of 1.e-35 (a normal number), mu_i_max and
// mu_i_initial are exact.
#define P3_ZSIZE         11
#define P3_TABSIZE_3MOM  15
#define P3_ZSMALL        1.0e-35f
#define P3_MU_I_MAX      20.0f
#define P3_MU_I_INITIAL  10.0f
#define P3_NITER_MUI     5

// nCat > 1 constants.  deltaD_init for nCat = 2 (:2235-2236); the splinter
// mass (10.e-6)**3, which gfortran folds at compile time to the correctly
// rounded float32 cube of 1e-5f -- MEASURED (gfortran 13.3.0): 0x26901D7C,
// equal to (x*x)*x for this argument; and the 1.e-14 qsmall_loc of
// icecat_destination (:6250).
#define P3_DELTAD_INIT_2CAT    0x1.0624de0000000p-11f // 500.e-6             :2236
#define P3_SPLINTER_D3         0x1.203af80000000p-50f // (10.e-6)**3         :3092
#define P3_QSMALL_LOC          P3_QSMALL              // 1.e-14              :6250

// Slots in the pointer arrays the launcher uploads.  Mirrored by the
// F_*/S_*/W_*/D_*/P_*/T_* tables in gpuwm/core/p3_device.py; a test pins
// the two lists against each other so a reorder cannot go unnoticed.
#define F_QC 0
#define F_NC 1
#define F_QR 2
#define F_NR 3
#define F_QI 4
#define F_QIR 5
#define F_NI 6
#define F_QIB 7
#define F_TH 8
#define F_QV 9
#define F_THOLD 10
#define F_QVOLD 11
#define F_SSAT 12
#define F_PRES 13
#define F_DZ 14
#define F_QZI 15
#define F_QI2 16
#define F_QIR2 17
#define F_NI2 18
#define F_QIB2 19

#define S_RHO 0
#define S_INVRHO 1
#define S_QVS 2
#define S_QVI 3
#define S_SUP 4
#define S_SUPI 5
#define S_RHOFACR 6
#define S_RHOFACI 7
#define S_ACN 8
#define S_T 9
#define S_TMPARR1 10
#define S_QVCLD 11

#define W_VQ 0
#define W_VN 1
#define W_FQ 2
#define W_FN 3
#define W_FQIR 4
#define W_FBIR 5
#define W_VZ 6
#define W_FZ 7

#define D_ZDBZ 0
#define D_EFFC 1
#define D_EFFI 2
#define D_VMI 3
#define D_DI 4
#define D_RHOPO 5
// mp_p3_wrapper_wrf_2cat's category-2 outputs diag_vmi2_3d, diag_di2_3d,
// diag_rhopo2_3d (:1137-1139); D_EFFI is then the wrapper's mass- and
// area-weighted combination of the two categories (:1155-1169).
#define D_VMI2 6
#define D_DI2 7
#define D_RHOPO2 8

#define P_PRTLIQ 0
#define P_PRTSOL 1
#define P_RAINNC 2
#define P_RAINNCV 3
#define P_SR 4
#define P_SNOWNC 5
#define P_SNOWNCV 6

#define T_ITAB 0
#define T_ICOLL 1
#define T_VN 2
#define T_VM 3
#define T_REVAP 4
#define T_ITAB3 5
#define T_ICOLL3 6
#define T_ICOLLI1 7
#define T_ICOLLI2 8

// Level-major addressing: element (k, column i) at k*ncol + i.
#define AT(a, kk) (a)[(size_t)(kk) * (size_t)ncol + (size_t)i]

// Per-category slots.  c is a compile-time constant at every call site
// (unrolled category loops), so these fold to the literal slot.
__device__ __forceinline__ int p3_fqi(int c)  { return c == 0 ? F_QI  : F_QI2; }
__device__ __forceinline__ int p3_fqir(int c) { return c == 0 ? F_QIR : F_QIR2; }
__device__ __forceinline__ int p3_fni(int c)  { return c == 0 ? F_NI  : F_NI2; }
__device__ __forceinline__ int p3_fqib(int c) { return c == 0 ? F_QIB : F_QIB2; }

// itabcolli1/2(i,jjj,jjjj,ii,jjj2,jjjj2), zero-based, Fortran column-major
// (first index fastest): (collectee size, rime, density, collector size,
// rime, density).  gpuwm/core/p3_tables.py:load_lookup_table_2 uploads the
// array in exactly this byte order (pinned to WRF's own p3_init arrays).
#define ICOLLI(tb, a, b, c, d, e, f) \
    (tb)[(((((size_t)(f) * P3_RIMSIZE + (size_t)(e)) * P3_IISIZE \
            + (size_t)(d)) * P3_DENSIZE + (size_t)(c)) * P3_RIMSIZE \
          + (size_t)(b)) * P3_IISIZE + (size_t)(a)]

// Fortran SUM over a category array: gfortran accumulates from +0.0 in
// index order (0. + a(1) + a(2)), which is kept, signed zeros included.
template <int NCAT>
__device__ __forceinline__ float p3_sum(const float* a)
{
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < NCAT; ++c) s = s + a[c];
    return s;
}

// itab(jj, ii, i2, idx), zero-based, tabsize fastest.
#define ITAB(jj, ii, i2, idx) \
    itab[((((size_t)(jj) * P3_RIMSIZE + (size_t)(ii)) * P3_ISIZE \
          + (size_t)(i2)) * P3_TABSIZE) + (size_t)(idx)]
// itabcoll(jj, ii, i2, j2, idx), zero-based, the 2 quantities fastest.
#define ICOLL(jj, ii, i2, j2, idx) \
    icoll[(((((size_t)(jj) * P3_RIMSIZE + (size_t)(ii)) * P3_ISIZE \
           + (size_t)(i2)) * P3_RCOLLSIZE + (size_t)(j2)) * 2) + (size_t)(idx)]
// itab_3mom(zz, jj, ii, i2, idx) (:63), zero-based, tabsize_3mom fastest:
// the 2-moment layout with the mu_i axis outermost.
#define ITAB3(zz, jj, ii, i2, idx) \
    itab3[(((((size_t)(zz) * P3_DENSIZE + (size_t)(jj)) * P3_RIMSIZE \
            + (size_t)(ii)) * P3_ISIZE + (size_t)(i2)) * P3_TABSIZE_3MOM) \
          + (size_t)(idx)]
// One mu_i slice of itabcoll_3mom (:67) is exactly an itabcoll, so the
// 2-moment accessor reads it through this offset.
#define P3_ICOLL3_SLICE \
    ((size_t)P3_DENSIZE * P3_RIMSIZE * P3_ISIZE * P3_RCOLLSIZE * 2)
// the three generated rain tables are (300, 10), mu_r column fastest.
#define RTAB(tb, a, b) (tb)[(size_t)(a) * 10 + (size_t)(b)]

// ---------------------------------------------------------------------
// Transcendental leaves.  r_exp / r_log / r_pow come from noahmp_leaves.cu,
// which is prepended to this translation unit by p3_device.py.
// ---------------------------------------------------------------------
#if defined(P3_FAST_2CAT) && !defined(GPUWM_WRF_EXACT)
// These aliases start after the shared leaves: no other scheme's math
// changes. WRF's ordinary REAL pow/log/exp/gamma expressions are REAL32;
// only the explicitly spelled double exp() calls below stay binary64.
#define r_exp expf
#define r_log logf
#define r_pow powf
#endif

__device__ __forceinline__ float p3_log10(float x)
{
#if defined(P3_FAST_2CAT) && !defined(GPUWM_WRF_EXACT)
    return log10f(x);
#elif defined(P3_NATIVE_GNU239)
    return r_log10(x);
#else
    return (float)log10((double)x);
#endif
}

__device__ __forceinline__ float p3_gam(float x)
{
#if defined(P3_FAST_2CAT) && !defined(GPUWM_WRF_EXACT)
    return tgammaf(x);
#elif defined(P3_NATIVE_GNU239)
    return p3_native_tgamma(x);
#else
    return (float)tgamma((double)x);
#endif
}

// Fortran int(): truncation toward zero, which is what a C float->int cast
// does.  Named so the transcription reads like the authority.
__device__ __forceinline__ int p3_int(float x) { return (int)x; }

// ---------------------------------------------------------------------
// polysvp1 (:6017-6086) -- saturation vapour pressure [Pa].
// ---------------------------------------------------------------------
__device__ float p3_polysvp1(float t, int i_type)
{
    if (i_type == 1 && t < 273.15f) {
        if (t >= 195.8f) {
            float dt = t - 273.15f;
            float a0i = 6.11147274f,      a1i = 0.503160820f,
                  a2i = 0.188439774e-1f,  a3i = 0.420895665e-3f,
                  a4i = 0.615021634e-5f,  a5i = 0.602588177e-7f,
                  a6i = 0.385852041e-9f,  a7i = 0.146898966e-11f,
                  a8i = 0.252751365e-14f;
            float out = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i
                        + dt * (a5i + dt * (a6i + dt * (a7i + a8i * dt)))))));
            return out * 100.0f;
        }
        return r_pow(10.0f, -9.09718f * (273.16f / t - 1.0f)
                            - 3.56654f * p3_log10(273.16f / t)
                            + 0.876793f * (1.0f - __fdiv_rn(t, 273.16f))
                            + p3_log10(6.1071f)) * 100.0f;
    }
    if (t >= 202.0f) {
        float dt = t - 273.15f;
        float a0 = 6.11239921f,       a1 = 0.443987641f,
              a2 = 0.142986287e-1f,   a3 = 0.264847430e-3f,
              a4 = 0.302950461e-5f,   a5 = 0.206739458e-7f,
              a6 = 0.640689451e-10f,  a7 = -0.952447341e-13f,
              a8 = -0.976195544e-15f;
        float out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4
                    + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
        return out * 100.0f;
    }
    return r_pow(10.0f, -7.90298f * (373.16f / t - 1.0f)
                        + 5.02808f * p3_log10(373.16f / t)
                        - 1.3816e-7f * (r_pow(10.0f, 11.344f
                                        * (1.0f - __fdiv_rn(t, 373.16f))) - 1.0f)
                        + 8.1328e-3f * (r_pow(10.0f, -3.49149f
                                        * (373.16f / t - 1.0f)) - 1.0f)
                        + p3_log10(1013.246f)) * 100.0f;
}

// qv_sat (:6859-6889).
__device__ __forceinline__ float p3_qv_sat(float t_atm, float p_atm, int i_wrt)
{
    float e_pres = p3_polysvp1(t_atm, i_wrt);
    return P3_EP_2 * e_pres / fmaxf(1.0e-3f, p_atm - e_pres);
}

// ---------------------------------------------------------------------
// Lookup-table access (:5205-5313) and index helpers (:6316-6410, :6589-6632).
// The Fortran 1-based index VALUES are kept; the -1 shift happens at the
// array access only, exactly as gpuwm/core/p3.py does it.
// ---------------------------------------------------------------------
__device__ float p3_access_lookup_table(const float* __restrict__ itab,
                                        int dumjj, int dumii, int dumi,
                                        int index, float dum1, float dum4,
                                        float dum5)
{
    float iproc1, gproc1, tmp1, tmp2;
    iproc1 = ITAB(dumjj-1, dumii-1, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj-1, dumii-1, dumi, index-1)
                                  - ITAB(dumjj-1, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj-1, dumii, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj-1, dumii, dumi, index-1)
                                  - ITAB(dumjj-1, dumii, dumi-1, index-1));
    tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    iproc1 = ITAB(dumjj, dumii-1, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj, dumii-1, dumi, index-1)
                                  - ITAB(dumjj, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj, dumii, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj, dumii, dumi, index-1)
                                  - ITAB(dumjj, dumii, dumi-1, index-1));
    tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

__device__ __forceinline__ float p3_coll_pair(const float* __restrict__ icoll,
                                              int jj, int ii, int dumi,
                                              int dumj, int index,
                                              float dum1, float dum3)
{
    float dproc1 = ICOLL(jj, ii, dumi-1, dumj-1, index-1)
        + (dum1 - (float)dumi) * (ICOLL(jj, ii, dumi, dumj-1, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj-1, index-1));
    float dproc2 = ICOLL(jj, ii, dumi-1, dumj, index-1)
        + (dum1 - (float)dumi) * (ICOLL(jj, ii, dumi, dumj, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj, index-1));
    return dproc1 + (dum3 - (float)dumj) * (dproc2 - dproc1);
}

__device__ float p3_access_lookup_table_coll(const float* __restrict__ icoll,
                                             int dumjj, int dumii, int dumj,
                                             int dumi, int index, float dum1,
                                             float dum3, float dum4, float dum5)
{
    float iproc1 = p3_coll_pair(icoll, dumjj-1, dumii-1, dumi, dumj, index,
                                dum1, dum3);
    float gproc1 = p3_coll_pair(icoll, dumjj-1, dumii, dumi, dumj, index,
                                dum1, dum3);
    float tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    iproc1 = p3_coll_pair(icoll, dumjj, dumii-1, dumi, dumj, index,
                          dum1, dum3);
    gproc1 = p3_coll_pair(icoll, dumjj, dumii, dumi, dumj, index, dum1, dum3);
    float tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

// find_lookupTable_indices_1a (:6316-6370).  Note the authority takes the
// INTEGER before clamping the real, then clamps both: the interpolation
// weight saturates to exactly 0 below the axis and 1 above it, which is
// constant extrapolation at both ends, never linear.
__device__ void p3_find_lt_1a(float qitot, float nitot, float qirim, float rhop,
                              int* dumi, int* dumjj, int* dumii,
                              float* dum1, float* dum4, float* dum5)
{
    float d1 = (p3_log10(qitot / nitot) + 18.0f) * 3.444606f - 10.0f;
    int di = p3_int(d1);
    d1 = fminf(d1, (float)P3_ISIZE);
    d1 = fmaxf(d1, 1.0f);
    di = max(1, di);
    di = min(P3_ISIZE - 1, di);

    float d4 = (qirim / qitot) * 3.0f + 1.0f;
    int dii = p3_int(d4);
    d4 = fminf(d4, (float)P3_RIMSIZE);
    d4 = fmaxf(d4, 1.0f);
    dii = max(1, dii);
    dii = min(P3_RIMSIZE - 1, dii);

    float d5;
    if (rhop <= 650.0f) d5 = (rhop - 50.0f) * 0.005f + 1.0f;
    else                d5 = (rhop - 650.0f) * 0.004f + 4.0f;
    int djj = p3_int(d5);
    d5 = fminf(d5, (float)P3_DENSIZE);
    d5 = fmaxf(d5, 1.0f);
    djj = max(1, djj);
    djj = min(P3_DENSIZE - 1, djj);

    *dumi = di; *dumjj = djj; *dumii = dii;
    *dum1 = d1; *dum4 = d4; *dum5 = d5;
}

// find_lookupTable_indices_1b (:6374-6410).
__device__ void p3_find_lt_1b(float qr, float nr, int* dumj, float* dum3)
{
    if (qr >= P3_QSMALL && nr > 0.0f) {
        float dumlr = r_pow(qr / (P3_PI * P3_RHOW * nr), P3_THRD);
        float d3 = (p3_log10(1.0f * dumlr) + 5.0f) * 10.70415f;
        int dj = p3_int(d3);
        d3 = fminf(d3, (float)P3_RCOLLSIZE);
        d3 = fmaxf(d3, 1.0f);
        dj = max(1, dj);
        dj = min(P3_RCOLLSIZE - 1, dj);
        *dumj = dj; *dum3 = d3;
    } else {
        *dumj = 1; *dum3 = 1.0f;
    }
}

// find_lookupTable_indices_3 (:6589-6632).
__device__ void p3_find_lt_3(float mu_r, float lamr, int* dumii, int* dumjj,
                             float* rdumii, float* rdumjj)
{
    float dum1 = (mu_r + 1.0f) / lamr;
    float rii, inv_dum3;
    int dii;
    if (dum1 <= 195.0e-6f) {
        inv_dum3 = 0.1f;
        rii = (dum1 * 1.0e6f + 5.0f) * inv_dum3;
        rii = fmaxf(rii, 1.0f);
        rii = fminf(rii, 20.0f);
        dii = p3_int(rii);
        dii = max(dii, 1);
        dii = min(dii, 20);
    } else {
        inv_dum3 = P3_THRD * 0.1f;
        rii = (dum1 * 1.0e6f - 195.0f) * inv_dum3 + 20.0f;
        rii = fmaxf(rii, 20.0f);
        rii = fminf(rii, 300.0f);
        dii = p3_int(rii);
        dii = max(dii, 20);
        dii = min(dii, 299);
    }
    (void)inv_dum3;
    float rjj = mu_r + 1.0f;
    rjj = fmaxf(rjj, 1.0f);
    rjj = fminf(rjj, 10.0f);
    int djj = p3_int(rjj);
    djj = max(djj, 1);
    djj = min(djj, 9);
    *dumii = dii; *dumjj = djj; *rdumii = rii; *rdumjj = rjj;
}

// Bilinear read of one generated (300, 10) rain table -- the shape the
// authority repeats at :3117-3122, :4150-4160 and :4166-4176.
__device__ __forceinline__ float p3_rain_table(const float* __restrict__ tb,
                                               int dumii, int dumjj,
                                               float rdumii, float rdumjj)
{
    float dum1 = RTAB(tb, dumii-1, dumjj-1)
        + (rdumii - (float)dumii) * (RTAB(tb, dumii, dumjj-1)
                                     - RTAB(tb, dumii-1, dumjj-1));
    float dum2 = RTAB(tb, dumii-1, dumjj)
        + (rdumii - (float)dumii) * (RTAB(tb, dumii, dumjj)
                                     - RTAB(tb, dumii-1, dumjj));
    return dum1 + (rdumjj - (float)dumjj) * (dum2 - dum1);
}

// get_cloud_dsd2 (:6635-6701).  nc_grd is INOUT in the authority.
__device__ void p3_get_cloud_dsd2(float* nc_grd, float qc_grd, float rho,
                                  float iscf, float* mu_c_o, float* lamc_o,
                                  float* cdist_o, float* cdist1_o)
{
    float qc = qc_grd * iscf;
    if (qc >= P3_QSMALL) {
        float nc = *nc_grd * iscf;
        nc = fmaxf(nc, P3_NSMALL);
        float mu_c = 0.0005714f * (nc * 1.0e-6f * rho) + 0.2714f;
        mu_c = 1.0f / (mu_c * mu_c) - 1.0f;            // :6662, integer **2
        mu_c = fmaxf(mu_c, 2.0f);
        mu_c = fminf(mu_c, 15.0f);
        // iparam = 3 (:193), so nu is 0 and dnu is unread (:6666-6670).
        float lamc = r_pow(P3_CONS1 * nc * (mu_c + 3.0f) * (mu_c + 2.0f)
                           * (mu_c + 1.0f) / qc, P3_THRD);
        float lammin = (mu_c + 1.0f) * 2.5e4f;
        float lammax = (mu_c + 1.0f) * 1.0e6f;
        if (lamc < lammin) {
            lamc = lammin;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6681, integer **3
        } else if (lamc > lammax) {
            lamc = lammax;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6684, integer **3
        }
        *cdist_o = nc * (mu_c + 1.0f) / lamc;
        *cdist1_o = nc / p3_gam(mu_c + 1.0f);
        *nc_grd = nc / iscf;
        *mu_c_o = mu_c; *lamc_o = lamc;
    } else {
        *mu_c_o = 0.0f; *lamc_o = 0.0f; *cdist_o = 0.0f; *cdist1_o = 0.0f;
    }
}

// get_rain_dsd2 (:6705-6780).  nr_grd is INOUT in the authority.
__device__ void p3_get_rain_dsd2(float* nr_grd, float qr_grd, float ispf,
                                 float* mu_r_o, float* lamr_o,
                                 float* cdistr_o, float* logn0r_o)
{
    float qr = qr_grd * ispf;
    if (qr >= P3_QSMALL) {
        float nr = *nr_grd * ispf;
        nr = fmaxf(nr, P3_NSMALL);
        float mu_r = P3_MU_R_CONSTANT;
        float lamr = r_pow(P3_CONS1 * nr * (mu_r + 3.0f) * (mu_r + 2.0f)
                           * (mu_r + 1.0f) / qr, P3_THRD);
        float lammax = (mu_r + 1.0f) * 1.0e5f;
        float lammin = (mu_r + 1.0f) * P3_INV_DRMAX;
        if (lamr < lammin) {
            lamr = lammin;
            nr = __fdiv_rn(r_exp(3.0f * r_log(lamr) + r_log(qr)
                       + r_log(p3_gam(mu_r + 1.0f))
                       - r_log(p3_gam(mu_r + 4.0f))), P3_CONS1);
        } else if (lamr > lammax) {
            lamr = lammax;
            nr = __fdiv_rn(r_exp(3.0f * r_log(lamr) + r_log(qr)
                       + r_log(p3_gam(mu_r + 1.0f))
                       - r_log(p3_gam(mu_r + 4.0f))), P3_CONS1);
        }
        *logn0r_o = p3_log10(nr) + (mu_r + 1.0f) * p3_log10(lamr)
                    - p3_log10(p3_gam(mu_r + 1.0f));
        *cdistr_o = nr / p3_gam(mu_r + 1.0f);
        *nr_grd = nr / ispf;
        *mu_r_o = mu_r; *lamr_o = lamr;
    } else {
        *mu_r_o = 0.0f; *lamr_o = 0.0f; *cdistr_o = 0.0f; *logn0r_o = 0.0f;
    }
}

// calc_bulkRhoRime (:6784-6830).
__device__ void p3_calc_bulk_rho_rime(float qi_tot, float* qi_rim,
                                      float* bi_rim, float* rho_rime_o)
{
    float qir = *qi_rim, bir = *bi_rim, rho_rime;
    if (bir >= 1.0e-15f) {
        rho_rime = qir / bir;
        if (rho_rime < P3_RHO_RIMEMIN) {
            rho_rime = P3_RHO_RIMEMIN;
            bir = __fdiv_rn(qir, rho_rime);
        } else if (rho_rime > P3_RHO_RIMEMAX) {
            rho_rime = P3_RHO_RIMEMAX;
            bir = __fdiv_rn(qir, rho_rime);
        }
    } else {
        qir = 0.0f; bir = 0.0f; rho_rime = 0.0f;
    }
    if (qir > qi_tot && rho_rime > 0.0f) {
        qir = qi_tot;
        bir = qir / rho_rime;
    }
    if (qir < P3_QSMALL) { qir = 0.0f; bir = 0.0f; }
    *qi_rim = qir; *bi_rim = bir; *rho_rime_o = rho_rime;
}

// impose_max_total_Ni, nCat = 1 form (:6833-6855).  Kept for the
// reflectivity operator (p3_zdiag.cu), whose state is one category.
__device__ __forceinline__ float p3_impose_max_ni(float nitot, float inv_rho)
{
    if (nitot >= 1.0e-20f) {
        float dum = P3_MAX_TOTAL_NI * inv_rho / nitot;
        nitot = nitot * fminf(dum, 1.0f);
    }
    return nitot;
}

// impose_max_total_Ni (:6833-6855) over the categories' numbers at one
// level, in place: the cap applies to sum(nitot(:)) and every category is
// scaled by the same factor, so the ratio between categories is kept.
template <int NCAT>
__device__ __forceinline__ void p3_impose_max_total_ni(float* const* ni, size_t at,
                                                       float inv_rho)
{
    float n[NCAT];
#pragma unroll
    for (int c = 0; c < NCAT; ++c) n[c] = ni[c][at];
    float s = p3_sum<NCAT>(n);
    if (s >= 1.0e-20f) {
        float dum = P3_MAX_TOTAL_NI * inv_rho / p3_sum<NCAT>(n);
#pragma unroll
        for (int c = 0; c < NCAT; ++c) ni[c][at] = n[c] * fminf(dum, 1.0f);
    }
}

// find_lookupTable_indices_2 (:6463-6585): table-2 indices for the
// collector (…_1) and the collectee (…_2).  The density test is bsmall for
// the collector and the literal 1.e-15 for the collectee, as written.
__device__ void p3_find_lt_2(float qitot_1, float qitot_2, float nitot_1,
                             float nitot_2, float qirim_1, float qirim_2,
                             float birim_1, float birim_2,
                             int* dumi, int* dumii, int* dumjj,
                             int* dumic, int* dumiic, int* dumjjc,
                             float* dum1, float* dum4, float* dum5,
                             float* dum1c, float* dum4c, float* dum5c)
{
    float d1 = (p3_log10(qitot_1 / nitot_1) + 18.0f) * 1.722303f - 5.0f;
    int di = p3_int(d1);
    d1 = fminf(d1, (float)P3_IISIZE);
    d1 = fmaxf(d1, 1.0f);
    di = max(1, di);
    di = min(P3_IISIZE - 1, di);

    float d4 = qirim_1 / qitot_1 * 3.0f + 1.0f;
    int dii = p3_int(d4);
    d4 = fminf(d4, (float)P3_RIMSIZE);
    d4 = fmaxf(d4, 1.0f);
    dii = max(1, dii);
    dii = min(P3_RIMSIZE - 1, dii);

    float drhop = (birim_1 >= P3_BSMALL) ? qirim_1 / birim_1 : 0.0f;
    float d5;
    if (drhop <= 650.0f) d5 = (drhop - 50.0f) * 0.005f + 1.0f;
    else                 d5 = (drhop - 650.0f) * 0.004f + 4.0f;
    int djj = p3_int(d5);
    d5 = fminf(d5, (float)P3_DENSIZE);
    d5 = fmaxf(d5, 1.0f);
    djj = max(1, djj);
    djj = min(P3_DENSIZE - 1, djj);

    float d1c = (p3_log10(qitot_2 / nitot_2) + 18.0f) * 1.722303f - 5.0f;
    int dic = p3_int(d1c);
    d1c = fminf(d1c, (float)P3_IISIZE);
    d1c = fmaxf(d1c, 1.0f);
    dic = max(1, dic);
    dic = min(P3_IISIZE - 1, dic);

    float d4c = qirim_2 / qitot_2 * 3.0f + 1.0f;
    int diic = p3_int(d4c);
    d4c = fminf(d4c, (float)P3_RIMSIZE);
    d4c = fmaxf(d4c, 1.0f);
    diic = max(1, diic);
    diic = min(P3_RIMSIZE - 1, diic);

    drhop = (birim_2 >= 1.0e-15f) ? qirim_2 / birim_2 : 0.0f;
    float d5c;
    if (drhop <= 650.0f) d5c = (drhop - 50.0f) * 0.005f + 1.0f;
    else                 d5c = (drhop - 650.0f) * 0.004f + 4.0f;
    int djjc = p3_int(d5c);
    d5c = fminf(d5c, (float)P3_DENSIZE);
    d5c = fmaxf(d5c, 1.0f);
    djjc = max(1, djjc);
    djjc = min(P3_DENSIZE - 1, djjc);

    *dumi = di; *dumii = dii; *dumjj = djj;
    *dumic = dic; *dumiic = diic; *dumjjc = djjc;
    *dum1 = d1; *dum4 = d4; *dum5 = d5;
    *dum1c = d1c; *dum4c = d4c; *dum5c = d5c;
}

// access_lookup_table_colli (:5317-5796).  Six-linear interpolation, in the
// authority's nesting: collector size innermost, then collector rime,
// collector density, collectee rime, collectee density; the collectee size
// is the inner difference of every pair.  The Fortran 1-based index VALUES
// are kept; the -1 shift happens at the array access.
__device__ __forceinline__ float p3_colli_iproc(const float* __restrict__ tb,
    int dumic, int dumiic, int dumjjc, int dumi, int dumii, int dumjj,
    float dum1c, float dum1)
{
    float dproc11 = ICOLLI(tb, dumic-1, dumiic-1, dumjjc-1, dumi-1, dumii-1, dumjj-1)
        + (dum1c - (float)dumic)
          * (ICOLLI(tb, dumic, dumiic-1, dumjjc-1, dumi-1, dumii-1, dumjj-1)
             - ICOLLI(tb, dumic-1, dumiic-1, dumjjc-1, dumi-1, dumii-1, dumjj-1));
    float dproc12 = ICOLLI(tb, dumic-1, dumiic-1, dumjjc-1, dumi, dumii-1, dumjj-1)
        + (dum1c - (float)dumic)
          * (ICOLLI(tb, dumic, dumiic-1, dumjjc-1, dumi, dumii-1, dumjj-1)
             - ICOLLI(tb, dumic-1, dumiic-1, dumjjc-1, dumi, dumii-1, dumjj-1));
    return dproc11 + (dum1 - (float)dumi) * (dproc12 - dproc11);
}

__device__ __forceinline__ float p3_colli_gproc(const float* __restrict__ tb,
    int dumic, int dumiic, int dumjjc, int dumi, int dumii, int dumjj,
    float dum1c, float dum1, float dum4, float dum5)
{
    float iproc1 = p3_colli_iproc(tb, dumic, dumiic, dumjjc, dumi, dumii,
                                  dumjj, dum1c, dum1);
    float iproc2 = p3_colli_iproc(tb, dumic, dumiic, dumjjc, dumi, dumii + 1,
                                  dumjj, dum1c, dum1);
    float tmp1 = iproc1 + (dum4 - (float)dumii) * (iproc2 - iproc1);
    iproc1 = p3_colli_iproc(tb, dumic, dumiic, dumjjc, dumi, dumii,
                            dumjj + 1, dum1c, dum1);
    iproc2 = p3_colli_iproc(tb, dumic, dumiic, dumjjc, dumi, dumii + 1,
                            dumjj + 1, dum1c, dum1);
    float tmp2 = iproc1 + (dum4 - (float)dumii) * (iproc2 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

__device__ float p3_access_lookup_table_colli(const float* __restrict__ tb,
    int dumjjc, int dumiic, int dumic, int dumjj, int dumii, int dumi,
    float dum1c, float dum4c, float dum5c, float dum1, float dum4, float dum5)
{
    float gproc1 = p3_colli_gproc(tb, dumic, dumiic, dumjjc, dumi, dumii,
                                  dumjj, dum1c, dum1, dum4, dum5);
    float gproc2 = p3_colli_gproc(tb, dumic, dumiic + 1, dumjjc, dumi, dumii,
                                  dumjj, dum1c, dum1, dum4, dum5);
    float rproc1 = gproc1 + (dum4c - (float)dumiic) * (gproc2 - gproc1);
    gproc1 = p3_colli_gproc(tb, dumic, dumiic, dumjjc + 1, dumi, dumii,
                            dumjj, dum1c, dum1, dum4, dum5);
    gproc2 = p3_colli_gproc(tb, dumic, dumiic + 1, dumjjc + 1, dumi, dumii,
                            dumjj, dum1c, dum1, dum4, dum5);
    float rproc2 = gproc1 + (dum4c - (float)dumiic) * (gproc2 - gproc1);
    return rproc1 + (dum5c - (float)dumjjc) * (rproc2 - rproc1);
}

// icecat_destination (:6214-6311): the category new ice goes to.  Qi is
// qitot(i,k,:)*iSCF(k) and Di the mean-mass diameters diam_ice(i,k,:).
// Returns a ZERO-based category.  The authority leaves i_mindiff unset when
// every non-empty category's |Di - D_nuc| is NaN or not below 9.e+9 and all
// categories are full (an uninitialised integer); that case returns 0.
template <int NCAT>
__device__ int p3_icecat_destination(const float* qi, const float* di,
                                     float d_nuc, float deltaD_init)
{
    if (p3_sum<NCAT>(qi) < P3_QSMALL_LOC) return 0;           // case 1
    int all_full = 1, i_first_empty = -1, i_mindiff = 0;
    float mindiff = 9.0e9f;
#pragma unroll
    for (int c = 0; c < NCAT; ++c) {
        if (qi[c] >= P3_QSMALL_LOC) {
            float diff = fabsf(di[c] - d_nuc);
            if (diff < mindiff) { mindiff = diff; i_mindiff = c; }
        } else {
            all_full = 0;
            if (i_first_empty < 0) i_first_empty = c;
        }
    }
    if (all_full) return i_mindiff;                           // case 2
    if (mindiff < deltaD_init) return i_mindiff;              // case 3a
    return i_first_empty;                                     // case 3b
}

// ---------------------------------------------------------------------
// DERF (:6090-6204) -- P3's OWN error function, which is what the
// activation's derf(uu1)/derf(uu2) (:3324-3325) call: the module procedure
// shadows the compiler's intrinsic of the same name, so the authority never
// reaches a C library here (nm on the oracle executable lists no erf; the
// routine is __microphy_p3_MOD_derf).  It is a REAL (binary32) function:
// Ooura's piecewise polynomial, thirteen coefficients per unit interval,
// two tables.  Transcribed in the authority's statement order with its
// mixed-kind details kept: W is compared against the DOUBLE literals 2.2D0
// and 6.9D0 (so the float is widened, never the literal narrowed; written
// as exact hex doubles because no unsuffixed decimal may appear here), INT()
// truncates, and T - K converts the integer K to REAL before subtracting.
// The 130 coefficients are the REAL(4) values gfortran makes of the
// decimal DATA statements, bit for bit (generated from those statements;
// tests/test_p3_cuda.py holds them to gpuwm/core/p3.py's table).
// ---------------------------------------------------------------------
__constant__ float P3_DERF_A[65] = {
       0x1.061398p-34f,  // 0.00000000005958930743E0
      -0x1.38a4b8p-30f,  // -0.00000000113739022964E0
       0x1.f7b72ep-27f,  // 0.00000001466005199839E0
      -0x1.5f1f06p-23f,  // -0.00000016350354461960E0
       0x1.b9df2ap-20f,  // 0.00000164610044809620E0
      -0x1.f4d1e4p-17f,  // -0.00001492559551950604E0
       0x1.f9a322p-14f,  // 0.00012055331122299265E0
      -0x1.c02db4p-11f,  // -0.00085483269811296660E0
        0x1.565bcep-8f,  // 0.00522397762482322257E0
       -0x1.b82ce4p-6f,  // -0.02686617064507733420E0
        0x1.ce2f22p-4f,  // 0.11283791670954881569E0
       -0x1.812746p-2f,  // -0.37612638903183748117E0
        0x1.20dd76p+0f,  // 1.12837916709551257377E0
       0x1.a16062p-36f,  // 0.00000000002372510631E0
       -0x1.f4342p-32f,  // -0.00000000045493253732E0
       0x1.95b1b8p-28f,  // 0.00000000590362766598E0
       -0x1.1d469p-24f,  // -0.00000006642090827576E0
       0x1.6ae6bcp-21f,  // 0.00000067595634268133E0
       -0x1.a0df6p-18f,  // -0.00000621188515924000E0
       0x1.ac250ep-15f,  // 0.00005103883009709690E0
      -0x1.84227cp-12f,  // -0.00037015410692956173E0
        0x1.31cd0cp-9f,  // 0.00233307631218880978E0
        -0x1.9b3c1p-7f,  // -0.01254988477182192210E0
        0x1.cf6d2cp-5f,  // 0.05657061146827041994E0
       -0x1.b5db04p-3f,  // -0.21379664776456006580E0
        0x1.af767ap-1f,  // 0.84270079294971486929E0
       0x1.4e37d6p-37f,  // 0.00000000000949905026E0
      -0x1.92a56ep-33f,  // -0.00000000018310229805E0
       0x1.491d94p-29f,  // 0.00000000239463074000E0
      -0x1.d38a64p-26f,  // -0.00000002721444369609E0
       0x1.2d22f2p-22f,  // 0.00000028045522331686E0
      -0x1.5f6c1cp-19f,  // -0.00000261830022482897E0
       0x1.705614p-16f,  // 0.00002195455056768781E0
      -0x1.5712a4p-13f,  // -0.00016358986921372656E0
       0x1.18a17cp-10f,  // 0.00107052153564110318E0
       -0x1.8ea53ep-8f,  // -0.00608284718113590151E0
        0x1.e962f6p-6f,  // 0.02986978465246258244E0
       -0x1.0b60e8p-3f,  // -0.13055593046562267625E0
        0x1.5990dap-1f,  // 0.67493323603965504676E0
       0x1.0d5114p-38f,  // 0.00000000000382722073E0
      -0x1.4667c6p-34f,  // -0.00000000007421598602E0
       0x1.0d3084p-30f,  // 0.00000000097930574080E0
      -0x1.82e4cap-27f,  // -0.00000001126008898854E0
       0x1.f9bcfap-24f,  // 0.00000011775134830784E0
      -0x1.2ca0d6p-20f,  // -0.00000111992758382650E0
        0x1.42cd3p-17f,  // 0.00000962023443095201E0
      -0x1.36902ap-14f,  // -0.00007404402135070773E0
       0x1.09c2f6p-11f,  // 0.00050689993654144881E0
       -0x1.931daep-9f,  // -0.00307553051439272889E0
        0x1.117202p-6f,  // 0.01668977892553165586E0
       -0x1.5e25e2p-4f,  // -0.08548534594781312114E0
        0x1.235fdep-1f,  // 0.56909076642393639985E0
       0x1.b51efcp-40f,  // 0.00000000000155296588E0
      -0x1.0ab734p-35f,  // -0.00000000003032205868E0
       0x1.bc79c8p-32f,  // 0.00000000040424830707E0
      -0x1.43c2f8p-28f,  // -0.00000000471135111493E0
       0x1.ae8532p-25f,  // 0.00000005011915876293E0
      -0x1.0593b8p-21f,  // -0.00000048722516178974E0
       0x1.2106d4p-18f,  // 0.00000430683284629395E0
       -0x1.20fd6p-15f,  // -0.00003445026145385764E0
       0x1.04e0ccp-12f,  // 0.00024879276133931664E0
       -0x1.ab23dp-10f,  // -0.00162940941748079288E0
        0x1.44016ap-7f,  // 0.00988786373932350462E0
       -0x1.e87128p-5f,  // -0.05962426839442303805E0
        0x1.fd9ae2p-2f,  // 0.49766113250947636708E0
};
__constant__ float P3_DERF_B[65] = {
      -0x1.46eedcp-32f,  // -0.00000000029734388465E0
       0x1.72c71cp-29f,  // 0.00000000269776334046E0
      -0x1.b858c4p-28f,  // -0.00000000640788827665E0
      -0x1.1e8782p-26f,  // -0.00000001667820132100E0
      -0x1.d551c4p-23f,  // -0.00000021854388148686E0
        0x1.65597p-19f,  // 0.00000266246030457984E0
       0x1.0e91e4p-16f,  // 0.00001612722157047886E0
      -0x1.0c9b66p-12f,  // -0.00025616361025506629E0
       0x1.428f44p-13f,  // 0.00015380842432375365E0
        0x1.0b3bdep-7f,  // 0.00815533022524927908E0
       -0x1.cb8014p-7f,  // -0.01402283663896319337E0
       -0x1.946a96p-3f,  // -0.19746892495383021487E0
        0x1.6e23d8p-1f,  // 0.71511720328842845913E0
      -0x1.573c8ep-36f,  // -0.00000000001951073787E0
      -0x1.632bfep-32f,  // -0.00000000032302692214E0
       0x1.670876p-28f,  // 0.00000000522461866919E0
       0x1.d7559cp-29f,  // 0.00000000342940918551E0
      -0x1.801bbap-22f,  // -0.00000035772874310272E0
        0x1.ad7edp-23f,  // 0.00000019999935792654E0
        0x1.c2cfbp-16f,  // 0.00002687044575042908E0
      -0x1.f0bdd2p-14f,  // -0.00011843240273775776E0
      -0x1.a8a142p-11f,  // -0.00080991728956032271E0
        0x1.b13bf6p-8f,  // 0.00661062970502241174E0
        0x1.2a08fcp-7f,  // 0.00909530922354827295E0
       -0x1.9ce0d8p-3f,  // -0.20160072778491013140E0
        0x1.05fd26p-1f,  // 0.51169696718727644908E0
        0x1.14df8p-35f,  // 0.00000000003147682272E0
      -0x1.0a71cap-31f,  // -0.00000000048465972408E0
       0x1.5e0fa4p-31f,  // 0.00000000063675740242E0
       0x1.2222b8p-25f,  // 0.00000003377623323271E0
      -0x1.4bcf8ap-23f,  // -0.00000015451139637086E0
      -0x1.10eb4ep-19f,  // -0.00000203340624738438E0
       0x1.46afccp-16f,  // 0.00001947204525295057E0
        0x1.ded8bp-16f,  // 0.00002854147231653228E0
      -0x1.0a3f28p-10f,  // -0.00101565063152200272E0
        0x1.637342p-9f,  // 0.00271187003520095655E0
        0x1.7d6f62p-6f,  // 0.02328095035422810727E0
       -0x1.568748p-3f,  // -0.16725021123116877197E0
        0x1.4cb2bap-2f,  // 0.32490054966649436974E0
       0x1.9806d8p-36f,  // 0.00000000002319363370E0
      -0x1.1537ccp-34f,  // -0.00000000006303206648E0
      -0x1.6c0f46p-29f,  // -0.00000000264888267434E0
       0x1.604f18p-26f,  // 0.00000002050708040581E0
       0x1.e86ae4p-24f,  // 0.00000011371857327578E0
      -0x1.1b7baap-19f,  // -0.00000211211337219663E0
       0x1.eefdccp-19f,  // 0.00000368797328322935E0
       0x1.9c0908p-14f,  // 0.00009823686253424796E0
       -0x1.594c2p-11f,  // -0.00065860243990455368E0
      -0x1.8ab6eap-11f,  // -0.00075285814895230877E0
        0x1.a798fap-6f,  // 0.02585434424202960464E0
       -0x1.dca7c4p-4f,  // -0.11637092784486193258E0
        0x1.761d74p-3f,  // 0.18267336775296612024E0
      -0x1.02cf0ap-38f,  // -0.00000000000367789363E0
       0x1.cb11b2p-33f,  // 0.00000000020876046746E0
      -0x1.09b216p-29f,  // -0.00000000193319027226E0
      -0x1.2b95bcp-28f,  // -0.00000000435953392472E0
       0x1.82b27cp-23f,  // 0.00000018006992266137E0
      -0x1.a520ccp-21f,  // -0.00000078441223763969E0
      -0x1.c54226p-18f,  // -0.00000675407647949153E0
       0x1.618374p-14f,  // 0.00008428418334440096E0
      -0x1.7130d8p-13f,  // -0.00017604388937031815E0
       -0x1.3a37e8p-9f,  // -0.00239729611435071610E0
        0x1.522fd8p-6f,  // 0.02064129023876022970E0
       -0x1.1ada14p-4f,  // -0.06905562880005864105E0
        0x1.741a2ap-4f,  // 0.09084526782065478489E0
};

__device__ float p3_derf(float x)
{
    float w = fabsf(x);
    float t, y;
    int k;
    if ((double)w < 0x1.199999999999ap+1) {             // :6175, 2.2D0
        t = w * w;
        k = p3_int(t);
        t = t - (float)k;
        k = k * 13;
        y = ((((((((((((P3_DERF_A[k] * t + P3_DERF_A[k + 1]) * t
              + P3_DERF_A[k + 2]) * t + P3_DERF_A[k + 3]) * t
              + P3_DERF_A[k + 4]) * t + P3_DERF_A[k + 5]) * t
              + P3_DERF_A[k + 6]) * t + P3_DERF_A[k + 7]) * t
              + P3_DERF_A[k + 8]) * t + P3_DERF_A[k + 9]) * t
              + P3_DERF_A[k + 10]) * t + P3_DERF_A[k + 11]) * t
              + P3_DERF_A[k + 12]) * w;
    } else if ((double)w < 0x1.b99999999999ap+2) {      // :6185, 6.9D0
        k = p3_int(w);
        t = w - (float)k;
        k = 13 * (k - 2);
        y = (((((((((((P3_DERF_B[k] * t + P3_DERF_B[k + 1]) * t
              + P3_DERF_B[k + 2]) * t + P3_DERF_B[k + 3]) * t
              + P3_DERF_B[k + 4]) * t + P3_DERF_B[k + 5]) * t
              + P3_DERF_B[k + 6]) * t + P3_DERF_B[k + 7]) * t
              + P3_DERF_B[k + 8]) * t + P3_DERF_B[k + 9]) * t
              + P3_DERF_B[k + 10]) * t + P3_DERF_B[k + 11]) * t
              + P3_DERF_B[k + 12];
        y = y * y;
        y = y * y;
        y = y * y;
        y = 1.0f - y * y;
    } else {
        y = 1.0f;
    }
    if (x < 0.0f) y = -y;                                            // :6201
    return y;
}

// ---------------------------------------------------------------------
// Three-moment ice (log_3momentIce).  Each routine is the authority's,
// statement for statement, in float32 with contraction off.
// ---------------------------------------------------------------------

// G_of_mu (:7134-7141): M0*M6/M3**2 of a gamma PSD of shape mu.
__device__ __forceinline__ float p3_g_of_mu(float mu)
{
    return ((6.0f + mu) * (5.0f + mu) * (4.0f + mu))
           / ((3.0f + mu) * (2.0f + mu) * (1.0f + mu));
}

// compute_mu_3moment (:7047-7131).  The authority STOPs when mom3 <=
// eps_m3 = 1e-20 (:7121-7127); every caller passes mom3 = 6 qitot /
// (rho_bulk pi) with qitot >= qsmall and rho_bulk at most the table's
// 900 kg m-3, so mom3 >= 2.1e-18 and that branch is unreachable here.
// It returns 0 rather than stopping a device kernel, and mu is
// initialised so a NaN G (which the authority leaves undefined) is
// deterministic.
__device__ float p3_compute_mu_3moment(float mom0, float mom3, float mom6,
                                       float mu_max)
{
    float mu = 0.0f;
    if (mom3 > 1.0e-20f) {
        float Gm = (mom0 / mom3) * (mom6 / mom3);                    // :7079 G
        if (Gm >= 20.0f) {
            mu = 0.0f;
        } else {
            float g2 = Gm * Gm;                                        // :7099 int **2
            if (Gm < 20.0f && Gm >= 13.31f) {
                mu = 3.3638e-3f * g2 - 1.7152e-1f * Gm + 2.0857e+0f;
            } else if (Gm < 13.31f && Gm >= 7.123f) {
                mu = 1.5900e-2f * g2 - 4.8202e-1f * Gm + 4.0108e+0f;
            } else if (Gm < 7.123f && Gm >= 4.200f) {
                mu = 1.0730e-1f * g2 - 1.7481e+0f * Gm + 8.4246e+0f;
            } else if (Gm < 4.200f && Gm >= 2.946f) {
                mu = 5.9070e-1f * g2 - 5.7918e+0f * Gm + 1.6919e+1f;
            } else if (Gm < 2.946f && Gm >= 1.793f) {
                mu = 4.3966e+0f * g2 - 2.6659e+1f * Gm + 4.5477e+1f;
            } else if (Gm < 1.793f && Gm >= 1.405f) {
                mu = 4.7552e+1f * g2 - 1.7958e+2f * Gm + 1.8126e+2f;
            } else if (Gm < 1.405f && Gm >= 1.230f) {
                mu = 3.0889e+2f * g2 - 9.0854e+2f * Gm + 6.8995e+2f;
            } else if (Gm < 1.230f) {
                mu = mu_max;
            }
        }
    }
    return mu;
}

// find_lookupTable_indices_1c (:6414-6460): the mu_i axis of the 3momI
// table.  Integer taken before the real is clamped, as in 1a.
__device__ void p3_find_lt_1c(float qitot, float mu_i, int* dumzz,
                              float* dum6)
{
    if (qitot >= P3_QSMALL) {
        float d6 = mu_i * 0.5f + 1.0f;                               // :6442
        int dz = p3_int(d6);
        d6 = fminf(d6, (float)P3_ZSIZE);
        d6 = fmaxf(d6, 1.0f);
        dz = max(1, dz);
        dz = min(P3_ZSIZE - 1, dz);
        *dumzz = dz; *dum6 = d6;
    } else {
        *dumzz = 1; *dum6 = 1.0f;
    }
}

// access_lookup_table_3mom (:5800-5879): the 2-moment trilinear read at
// mu_i index dumzz and dumzz+1, then linear in dum6.
__device__ float p3_access_lookup_table_3mom(const float* __restrict__ itab3,
                                             int dumzz, int dumjj, int dumii,
                                             int dumi, int index, float dum1,
                                             float dum4, float dum5,
                                             float dum6)
{
    float rproc[2];
    for (int z = 0; z < 2; ++z) {
        int zz = dumzz - 1 + z;
        float iproc1, gproc1, tmp1, tmp2;
        iproc1 = ITAB3(zz, dumjj-1, dumii-1, dumi-1, index-1)
            + (dum1 - (float)dumi) * (ITAB3(zz, dumjj-1, dumii-1, dumi, index-1)
                                      - ITAB3(zz, dumjj-1, dumii-1, dumi-1, index-1));
        gproc1 = ITAB3(zz, dumjj-1, dumii, dumi-1, index-1)
            + (dum1 - (float)dumi) * (ITAB3(zz, dumjj-1, dumii, dumi, index-1)
                                      - ITAB3(zz, dumjj-1, dumii, dumi-1, index-1));
        tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
        iproc1 = ITAB3(zz, dumjj, dumii-1, dumi-1, index-1)
            + (dum1 - (float)dumi) * (ITAB3(zz, dumjj, dumii-1, dumi, index-1)
                                      - ITAB3(zz, dumjj, dumii-1, dumi-1, index-1));
        gproc1 = ITAB3(zz, dumjj, dumii, dumi-1, index-1)
            + (dum1 - (float)dumi) * (ITAB3(zz, dumjj, dumii, dumi, index-1)
                                      - ITAB3(zz, dumjj, dumii, dumi-1, index-1));
        tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
        rproc[z] = tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
    }
    return rproc[0] + (dum6 - (float)dumzz) * (rproc[1] - rproc[0]);  // :5877
}

// access_lookup_table_coll_3mom (:5882-6013): per mu_i slice the
// 2-moment collection read (p3_access_lookup_table_coll above, the same
// statements on the same (densize, rimsize, isize, rcollsize, 2) slice),
// then linear in dum6.
__device__ float p3_access_lookup_table_coll_3mom(
    const float* __restrict__ icoll3, int dumzz, int dumjj, int dumii,
    int dumj, int dumi, int index, float dum1, float dum3, float dum4,
    float dum5, float dum6)
{
    float rproc1 = p3_access_lookup_table_coll(
        icoll3 + (size_t)(dumzz - 1) * P3_ICOLL3_SLICE, dumjj, dumii, dumj,
        dumi, index, dum1, dum3, dum4, dum5);
    float rproc2 = p3_access_lookup_table_coll(
        icoll3 + (size_t)dumzz * P3_ICOLL3_SLICE, dumjj, dumii, dumj,
        dumi, index, dum1, dum3, dum4, dum5);
    return rproc1 + (dum6 - (float)dumzz) * (rproc2 - rproc1);       // :6011
}

// The authority's five-pass mu_i / bulk-density iteration, written out
// identically at :2614-2621, :4396-4403, :4639-4645 and :4806-4813:
// start from a 200 kg m-3 moment-3 estimate, solve mu_i from (N, M3, Z),
// read the bulk density (quantity 12) at that mu_i, re-estimate M3.  The
// indices and dum6 of the LAST pass are the ones every later read uses.
__device__ void p3_mu_i_iterate(const float* __restrict__ itab3, float qitot,
                                float nitot, float zitot, int dumjj,
                                int dumii, int dumi, float dum1, float dum4,
                                float dum5, int* dumzz, float* dum6,
                                float* f1pr16)
{
    float dum1z = 6.0f / (200.0f * P3_PI) * qitot;
    for (int imu = 1; imu <= P3_NITER_MUI; ++imu) {
        float mu_i = p3_compute_mu_3moment(nitot, dum1z, zitot, P3_MU_I_MAX);
        p3_find_lt_1c(qitot, mu_i, dumzz, dum6);
        *f1pr16 = p3_access_lookup_table_3mom(itab3, *dumzz, dumjj, dumii,
                                              dumi, 12, dum1, dum4, dum5,
                                              *dum6);
        dum1z = 6.0f / (*f1pr16 * P3_PI) * qitot;
    }
}

// The "adjust Zitot to make sure mu is in bounds" clamp (:2654-2660,
// :4651-4657, :4838-4846): Z between the mu_i = 20 and mu_i = 0 values
// for the moment-3 estimate from the bulk density.
__device__ __forceinline__ float p3_clamp_zitot(float zitot, float qitot,
                                                float nitot, float f1pr16)
{
    float dum1 = 6.0f / (f1pr16 * P3_PI) * qitot;
    float tmp1 = p3_g_of_mu(0.0f);
    float tmp2 = p3_g_of_mu(20.0f);
    zitot = fminf(zitot, tmp1 * (dum1 * dum1) / nitot);              // int **2
    zitot = fmaxf(zitot, tmp2 * (dum1 * dum1) / nitot);
    return zitot;
}

// ---------------------------------------------------------------------
// SEAM 2 -- droplet activation as a SUBSTITUTABLE STEP.
//
// The whole of the authority activation section (:3303-3339), BOTH
// branches, lifted into one callable device function whose aerosol inputs
// are the P3_NANEW*/P3_INV_RM*/P3_SIG* DATA above.  A prognostic aerosol
// species replaces this function and those constants and touches nothing
// else in the port.  It stays a call rather than arithmetic inlined into a
// fused kernel for exactly that reason; the measured cost is in the
// receipt, not assumed to be zero.
// ---------------------------------------------------------------------
__device__ void p3_activate_droplets(int log_predictNc, float sup_cld, int it,
                                     float t, float pres, float qc, float nc,
                                     float qv_cld, float inv_rho, float iscf,
                                     float scf, float odt,
                                     float* qcnuc, float* ncnuc)
{
    if (!log_predictNc && sup_cld > 1.0e-6f && it > 1) {
        float dum = P3_NCCNST * inv_rho * P3_CONS7 - qc;
        dum = fmaxf(0.0f, dum * iscf);
        float dumqvs = p3_qv_sat(t, pres, 0);
        float dqsdT = P3_XXLV * dumqvs / (P3_RV * t * t);
        float ab = 1.0f + dqsdT * P3_XXLV * P3_INV_CP;
        dum = fmaxf(0.0f, fminf(dum, (qv_cld - dumqvs) / ab));
        *qcnuc = dum * odt * scf;
    }
    if (log_predictNc) {
        if (sup_cld > 1.0e-6f) {
            float dum1 = 1.0f / r_pow(P3_BACT, 0.5f);
            float sigvl = 0.0761f - 1.55e-4f * (t - 273.15f);
            float aact = 2.0f * P3_MW / (P3_RHOW * P3_RR * t) * sigvl;
            float sm1 = 2.0f * dum1 * r_pow(aact * P3_THRD * P3_INV_RM1, 1.5f);
            float sm2 = 2.0f * dum1 * r_pow(aact * P3_THRD * P3_INV_RM2, 1.5f);
            float uu1 = 2.0f * r_log(sm1 / sup_cld) / (4.242f * r_log(P3_SIG1));
            float uu2 = 2.0f * r_log(sm2 / sup_cld) / (4.242f * r_log(P3_SIG2));
            // derf is P3's own REAL DERF (:6090-6204), not a C library
            // erf: see p3_derf above.
            dum1 = P3_NANEW1 * 0.5f * (1.0f - p3_derf(uu1));         // :3324
            float dum2 = P3_NANEW2 * 0.5f * (1.0f - p3_derf(uu2));   // :3325
            dum2 = fminf(P3_NANEW1 + P3_NANEW2, dum1 + dum2);
            dum2 = (dum2 - nc * iscf) * odt * scf;
            dum2 = fmaxf(0.0f, dum2);
            *ncnuc = dum2;
            if (it <= 1) *qcnuc = 0.0f;
            else         *qcnuc = *ncnuc * P3_CONS7;
        }
    }
}

// =====================================================================
// The column steps.  Each is the authority's corresponding loop, run by
// ONE thread for ONE column.  The __global__ wrappers below compose them
// two ways -- one step per launch (the UNFUSED reference arm) and several
// per launch (the FUSED arm) -- from these same function bodies, so a
// difference between the arms can only come from the compiler, which is
// exactly what the byte gate in tests/test_p3_cuda.py is there to catch.
//
// Every step that touches ice is a template on NCAT, the authority's nCat.
// Category c of a field is F[p3_fqi(c)] etc.; per-category process rates
// are NCAT-long register arrays, as they are dimension(nCat) locals in
// p3_main (:2046-2079).
// =====================================================================

#define P3_ARGS \
    float* const* __restrict__ F, float* const* __restrict__ S, \
    float* const* __restrict__ W, float* const* __restrict__ D, \
    float* const* __restrict__ P, const float* const* __restrict__ T, \
    float* __restrict__ FLG, int ncol, int nk, float dt, int it, \
    int log_predictNc, int log_3momentIce, float clbfact_dep, \
    float clbfact_sub, int i
#define P3_PASS \
    F, S, W, D, P, T, FLG, ncol, nk, dt, it, log_predictNc, \
    log_3momentIce, clbfact_dep, clbfact_sub, i

// ---------------------------------------------------------------------
// Slab diagnostics at entry (:2293-2297) plus the diagnostic pre-sets
// (:2276-2288) and the surface accumulators.  Level-local, so this is the
// authority's whole-array section written as a k-loop.  For NCAT = 2 the
// category-2 outputs take p3_main's own pre-sets too (diag_vmi/di/rhoi = 0
// for every category, :2282-2284); D_EFFI's 25.e-6 is both p3_main's
// per-category pre-set and the wrapper's combined default (:1166).
// ---------------------------------------------------------------------
template <int NCAT>
__device__ void p3_step_prep_levels(P3_ARGS, int k_start, int k_stride)
{
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ pres = F[F_PRES];
    for (int k = k_start; k < nk; k += k_stride) {
        float tm = r_pow(AT(pres, k) * 1.0e-5f, P3_RD * P3_INV_CP);  // :2293
        AT(S[S_TMPARR1], k) = tm;
        AT(S[S_T], k) = AT(th, k) * tm;                              // :2295
        AT(qv, k) = fmaxf(AT(qv, k), 0.0f);                          // :2297
        AT(D[D_ZDBZ], k) = -99.0f;                                   // :2278
        AT(D[D_EFFC], k) = 10.0e-6f;                                 // :2279
        AT(D[D_EFFI], k) = 25.0e-6f;                                 // :2280
        AT(D[D_VMI], k) = 0.0f;                                      // :2282
        AT(D[D_DI], k) = 0.0f;                                       // :2283
        AT(D[D_RHOPO], k) = 0.0f;                                    // :2284
        if (NCAT > 1) {
            AT(D[D_VMI2], k) = 0.0f;
            AT(D[D_DI2], k) = 0.0f;
            AT(D[D_RHOPO2], k) = 0.0f;
        }
        if (log_3momentIce) {
            // The wrapper's "convert advected (N*Z)^0.5 to Z for P3 main"
            // (:826-838), on the Ni the caller passed in.
            float* __restrict__ qzi = F[F_QZI];
            float ni_in = AT(F[F_NI], k);
            if (ni_in >= P3_QSMALL) {
                AT(qzi, k) = (AT(qzi, k) * AT(qzi, k)) / ni_in;      // int **2
            } else {
                AT(qzi, k) = 0.0f;
            }
        }
    }
}

// The end of p3_main and of the wrapper for Z (three-moment ice only):
// "zero out zitot if there is no qitot" after label 333 (:4916-4918),
// which every column reaches whether or not it had anything to do, and
// the wrapper's "convert Z from P3 to (N*Z)^0.5 for advection"
// (:911-924) on the Ni p3_main returns.
__device__ __forceinline__ void p3_zi_finish_level(P3_ARGS, int k)
{
    if (!log_3momentIce) return;
    float* __restrict__ qzi = F[F_QZI];
    if (AT(F[F_QI], k) < P3_QSMALL) AT(qzi, k) = 0.0f;
    float ni = AT(F[F_NI], k);
    float zi = AT(qzi, k);
    if (ni >= P3_QSMALL && zi >= 1.0e-30f) {
        AT(qzi, k) = r_pow(zi * ni, 0.5f);                           // :917 real **0.5
    } else {
        AT(qzi, k) = 0.0f;
    }
}

template <int NCAT>
__device__ void p3_step_prep_col(P3_ARGS)
{
    p3_step_prep_levels<NCAT>(P3_PASS, 0, 1);
    P[P_PRTLIQ][i] = 0.0f;                                           // :2270
    P[P_PRTSOL][i] = 0.0f;                                           // :2271
    // The two column-scope logical flags.  Float 0/1 rather than int so
    // the allocation gate prices them alongside every other companion
    // array; 0.0f and 1.0f are exact and compare exactly.
    FLG[i] = 0.0f;                                                   // nucleation
    FLG[ncol + i] = 0.0f;                                            // hydrometeors
}

// ---------------------------------------------------------------------
// k_loop_1 (:2320-2411): atmospheric variables, the two column-scope
// logical flags, and mass clipping in dry air (per category, :2383-2407).
// ---------------------------------------------------------------------
template <int NCAT>
__device__ int p3_step_kloop1_levels(P3_ARGS, int k_start, int k_stride)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ th_old = F[F_THOLD];
    const float* __restrict__ qv_old = F[F_QVOLD];
    float* __restrict__ ssat = F[F_SSAT];      // SEAM 1: live, threaded
    const float* __restrict__ pres = F[F_PRES];

    int nucleation_possible = 0;
    int hydrometeors_present = 0;

    for (int k = k_start; k < nk; k += k_stride) {
        float t = AT(S[S_T], k);
        float tm = AT(S[S_TMPARR1], k);
        float invexn = 1.0f / tm;                                    // :2294
        float rho = AT(pres, k) / (P3_RD * t);                       // :2322
        float inv_rho = 1.0f / rho;                                  // :2323
        AT(S[S_RHO], k) = rho;
        AT(S[S_INVRHO], k) = inv_rho;
        // The first-call guard (:2329-2330): th_old/qv_old are the zero the
        // allocation left, and max(t_old,1.) is what keeps polysvp1 finite.
        float t_old = AT(th_old, k) * tm;
        float qvs = p3_qv_sat(fmaxf(t_old, 1.0f), AT(pres, k), 0);
        float qvi = p3_qv_sat(fmaxf(t_old, 1.0f), AT(pres, k), 1);
        // Cold-start qvs/qvi floor, DEFAULT-ON in all three arms
        // (2026-08-31): polysvp1(1 K) underflows, so the stock step-1
        // sup/supi diagnoses below were 0/0 NaN; the floor pins them at
        // exactly -1 and is inert from step 2 on.  The full rationale,
        // the declared step-1 clip delta and the flipped pin test live
        // at the CPU authority's floor site (gpuwm/core/p3.py); the
        // three arms move together or not at all.
        qvs = fmaxf(qvs, 1.0e-20f);
        qvi = fmaxf(qvi, 1.0e-20f);
        AT(S[S_QVS], k) = qvs;
        AT(S[S_QVI], k) = qvi;
        // SEAM 1 -- log_predictSsat is .false. (:2252) so ssat is DIAGNOSED
        // here, but it stays a real threaded array with both branches
        // present.  The authority's own note is that the prediction code is
        // absent; this is the documented attach point and it is not
        // optimised away just because it_le_1 makes the second branch
        // unreachable today.
        float sup, supi;
        if (1 /* .not. log_predictSsat .or. it <= 1 */) {
            AT(ssat, k) = AT(qv_old, k) - qvs;                       // :2334
            sup = AT(qv_old, k) / qvs - 1.0f;                        // :2335
            supi = AT(qv_old, k) / qvi - 1.0f;                       // :2336
        } else {
            sup = AT(ssat, k) / qvs;                                 // :2339
            supi = (AT(ssat, k) + qvs - qvi) / qvi;                  // :2340
        }
        AT(S[S_SUP], k) = sup;
        AT(S[S_SUPI], k) = supi;
        AT(S[S_RHOFACR], k) = r_pow(P3_RHOSUR * inv_rho, 0.54f);     // :2343
        AT(S[S_RHOFACI], k) = r_pow(P3_RHOSUI * inv_rho, 0.54f);     // :2344
        float dum = 1.496e-6f * r_pow(t, 1.5f) / (t + 120.0f);       // :2345
        AT(S[S_ACN], k) = P3_G * P3_RHOW / (18.0f * dum);            // :2346
        if (!log_predictNc) AT(nc, k) = P3_NCCNST * inv_rho;         // :2349-2351

        // Fortran precedence preserved: A .or. (B .and. .not.SCPF_on)
        if ((t < 273.15f && supi >= -0.05f)
            || (t >= 273.15f && sup >= -0.05f)) {
            nucleation_possible = 1;                                 // :2358-2360
        }

        if (AT(qc, k) < P3_QSMALL
            || (AT(qc, k) < 1.0e-8f && sup < -0.1f)) {               // :2365
            AT(qv, k) = AT(qv, k) + AT(qc, k);
            AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
            AT(qc, k) = 0.0f;
            AT(nc, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qr, k) < P3_QSMALL
            || (AT(qr, k) < 1.0e-8f && sup < -0.1f)) {               // :2375
            AT(qv, k) = AT(qv, k) + AT(qr, k);
            AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
            AT(qr, k) = 0.0f;
            AT(nr, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

#pragma unroll
        for (int c = 0; c < NCAT; ++c) {                             // :2383
            float* __restrict__ qi = F[p3_fqi(c)];
            float* __restrict__ qir = F[p3_fqir(c)];
            float* __restrict__ ni = F[p3_fni(c)];
            float* __restrict__ qib = F[p3_fqib(c)];
            if (AT(qi, k) < P3_QSMALL
                || (AT(qi, k) < 1.0e-8f && supi < -0.1f)) {          // :2384
                AT(qv, k) = AT(qv, k) + AT(qi, k);
                AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
                AT(qi, k) = 0.0f;
                AT(ni, k) = 0.0f;
                AT(qir, k) = 0.0f;
                AT(qib, k) = 0.0f;
            } else {
                hydrometeors_present = 1;
            }

            if (AT(qi, k) >= P3_QSMALL && AT(qi, k) < 1.0e-8f
                && t >= 273.15f) {                                   // :2396
                AT(qr, k) = AT(qr, k) + AT(qi, k);
                AT(nr, k) = AT(nr, k) + AT(ni, k);
                AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XLF * P3_INV_CP;
                AT(qi, k) = 0.0f;
                AT(ni, k) = 0.0f;
                AT(qir, k) = 0.0f;
                AT(qib, k) = 0.0f;
            }
        }
        // "zero out zitot if there is no qitot for triple moment"
        // (:2413-2414), a whole-column WHERE after k_loop_1 and therefore
        // level-local: it reads this level's clipped qitot only.
        if (log_3momentIce && AT(F[F_QI], k) < P3_QSMALL) AT(F[F_QZI], k) = 0.0f;
    }
    // first compute_SCPF (:2432-2434) with SCPF off (:1889-1899): the qv
    // snapshot is the only product the WRF call shape reads.
    for (int k = k_start; k < nk; k += k_stride) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    return nucleation_possible | (hydrometeors_present << 1);
}

template <int NCAT>
__device__ void p3_step_kloop1_col(P3_ARGS)
{
    int flags = p3_step_kloop1_levels<NCAT>(P3_PASS, 0, 1);
    FLG[i] = (flags & 1) ? 1.0f : 0.0f;
    FLG[ncol + i] = (flags & 2) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------
// k_loop_main (:2449-3927): process rates, conservation, prognostic
// update, clipping.  The single largest step; run by one thread over the
// whole column (or by a level group over its levels) so the statement
// order is the authority's.  Every level is independent of every other.
// ---------------------------------------------------------------------
template <int NCAT>
__device__ int p3_step_kloopmain_levels(P3_ARGS, int k_start, int k_stride)
{
    // goto 333 (:2439) -- nothing to do in this column.
    if (FLG[i] == 0.0f && FLG[ncol + i] == 0.0f) return 0;

    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ qv_old = F[F_QVOLD];
    const float* __restrict__ th_old = F[F_THOLD];
    float* __restrict__ ssat = F[F_SSAT];
    const float* __restrict__ pres = F[F_PRES];
    const float* __restrict__ itab = T[T_ITAB];
    const float* __restrict__ icoll = T[T_ICOLL];
    const float* __restrict__ revap = T[T_REVAP];
    float* qi_[NCAT];
    float* qir_[NCAT];
    float* ni_[NCAT];
    float* qib_[NCAT];
#pragma unroll
    for (int c = 0; c < NCAT; ++c) {
        qi_[c] = F[p3_fqi(c)];
        qir_[c] = F[p3_fqir(c)];
        ni_[c] = F[p3_fni(c)];
        qib_[c] = F[p3_fqib(c)];
    }
    // Three-moment ice only: Z (the authority's zitot, already converted
    // from the advected (N*Z)**0.5 by the prep step) and the 3momI tables.
    float* __restrict__ qzi = F[F_QZI];
    const float* __restrict__ itab3 = T[T_ITAB3];
    const float* __restrict__ icoll3 = T[T_ICOLL3];

    const float odt = 1.0f / dt;                                     // :2261
    // SCPF off-branch (:1889-1899) -- constant 1 / 0, kept named so the
    // Sundqvist branch has somewhere to attach.
    const float iscf = 1.0f, scf = 1.0f, spf = 1.0f, ispf = 1.0f;
    const float spf_clr = 0.0f;
    int hydrometeors_present = 0;

    for (int k = k_start; k < nk; k += k_stride) {
        const size_t at = (size_t)k * (size_t)ncol + (size_t)i;
        const float t = AT(S[S_T], k);
        const float rho = AT(S[S_RHO], k);
        const float inv_rho = AT(S[S_INVRHO], k);
        const float qvs = AT(S[S_QVS], k);
        const float qvi = AT(S[S_QVI], k);
        const float sup = AT(S[S_SUP], k);
        const float supi = AT(S[S_SUPI], k);
        const float rhofaci = AT(S[S_RHOFACI], k);
        const float acn = AT(S[S_ACN], k);
        const float qv_cld = AT(S[S_QVCLD], k);
        const float invexn = 1.0f / AT(S[S_TMPARR1], k);
        const float t_old = AT(th_old, k) * AT(S[S_TMPARR1], k);

        // goto 555 (:2452-2464): dry, subsaturated, no hydrometeors.
        int any_ice = 0;
#pragma unroll
        for (int c = 0; c < NCAT; ++c)
            if (qi_[c][at] >= P3_QSMALL) any_ice = 1;
        int log_exitlevel = !(AT(qc, k) >= P3_QSMALL || AT(qr, k) >= P3_QSMALL
                              || any_ice);
        if (log_exitlevel && ((t < 273.15f && supi < -0.05f)
                              || (t >= 273.15f && sup < -0.05f))) continue;

        // process rates (:2467-2485): warm-phase scalars, ice-phase arrays
        // of dimension(nCat), the inter-category pair (nCat,nCat).
        float qcacc = 0.0f, qrevp = 0.0f, qccon = 0.0f;
        float qcaut = 0.0f, qcevp = 0.0f, qrcon = 0.0f;
        float ncacc = 0.0f, ncnuc = 0.0f, ncslf = 0.0f;
        float ncautc = 0.0f, qcnuc = 0.0f, nrslf = 0.0f;
        float nrevp = 0.0f, ncautr = 0.0f;
        float qchetc[NCAT], qisub[NCAT], nrshdr[NCAT], qcheti[NCAT];
        float qrcol[NCAT], qcshd[NCAT], qrhetc[NCAT], qimlt[NCAT];
        float qccol[NCAT], qrheti[NCAT], qinuc[NCAT], nimlt[NCAT];
        float nchetc[NCAT], nccol[NCAT], ncshdc[NCAT], ncheti[NCAT];
        float nrcol[NCAT], nislf[NCAT], nrhetc[NCAT], ninuc[NCAT];
        float qidep[NCAT], nrheti[NCAT], nisub[NCAT], qwgrth[NCAT];
        float qrmul[NCAT], nimul[NCAT];
        float qicol[NCAT][NCAT], nicol[NCAT][NCAT];
        int log_wetgrowth[NCAT];
        float epsi[NCAT], eii_fact[NCAT], diam_ice[NCAT], rhorime_c[NCAT];
#pragma unroll
        for (int c = 0; c < NCAT; ++c) {
            qchetc[c] = 0.0f; qisub[c] = 0.0f; nrshdr[c] = 0.0f;
            qcheti[c] = 0.0f; qrcol[c] = 0.0f; qcshd[c] = 0.0f;
            qrhetc[c] = 0.0f; qimlt[c] = 0.0f; qccol[c] = 0.0f;
            qrheti[c] = 0.0f; qinuc[c] = 0.0f; nimlt[c] = 0.0f;
            nchetc[c] = 0.0f; nccol[c] = 0.0f; ncshdc[c] = 0.0f;
            ncheti[c] = 0.0f; nrcol[c] = 0.0f; nislf[c] = 0.0f;
            nrhetc[c] = 0.0f; ninuc[c] = 0.0f; qidep[c] = 0.0f;
            nrheti[c] = 0.0f; nisub[c] = 0.0f; qwgrth[c] = 0.0f;
            qrmul[c] = 0.0f; nimul[c] = 0.0f;
#pragma unroll
            for (int cc = 0; cc < NCAT; ++cc) {
                qicol[c][cc] = 0.0f;
                nicol[c][cc] = 0.0f;
            }
            log_wetgrowth[c] = 0;
            epsi[c] = 0.0f;
            eii_fact[c] = 1.0f;
            // diam_ice(i,k,:) is 0 at p3_main entry (:2274) and only
            // written in iice_loop1 below (:2578); a level that takes the
            // 444 skip keeps the zero.
            diam_ice[c] = 0.0f;
            // rhorime_c is a dimension(nCat) local set to 400 once (:2290);
            // it is only ever READ when qccol(iice) > 0, which implies the
            // if/else at :2925-2969 just wrote it at this level, so the
            // authority's carry across levels and columns is unobservable
            // and this per-level initialisation is equivalent.
            rhorime_c[c] = 400.0f;
        }

        // goto 444 (:2523-2528)
        any_ice = 0;
#pragma unroll
        for (int c = 0; c < NCAT; ++c)
            if (qi_[c][at] >= P3_QSMALL) any_ice = 1;
        log_exitlevel = !(AT(qc, k) >= P3_QSMALL || AT(qr, k) >= P3_QSMALL
                          || any_ice);

        float dqsdT = 0.0f, epsi_tot = 0.0f;
        float f1pr16 = 0.0f;
        float mu_c = 0.0f, lamc = 0.0f, cdist = 0.0f, cdist1 = 0.0f;
        float mu_r = 0.0f, lamr = 0.0f, cdistr = 0.0f, logn0r = 0.0f;
        float abi = 1.0f, ab = 1.0f, oabi = 1.0f, oxx = 0.0f, aaa = 0.0f;
        float dum, dum1, dum2, tmp1, tmp2;
        int iice_dest;

        if (!log_exitlevel) {
            float mu = 1.496e-6f * r_pow(t, 1.5f) / (t + 120.0f);    // :2531
            float dv = 8.794e-5f * r_pow(t, 1.81f) / AT(pres, k);    // :2532
            float sc = mu / (rho * dv);                              // :2533
            dum = 1.0f / (P3_RV * (t * t));                          // :2534 int **2
            dqsdT = P3_XXLV * qvs * dum;                             // :2535
            float dqsidT = P3_XXLS * qvi * dum;                      // :2536
            ab = 1.0f + dqsdT * P3_XXLV * P3_INV_CP;                 // :2537
            abi = 1.0f + dqsidT * P3_XXLS * P3_INV_CP;               // :2538
            float kap = 1.414e3f * mu;                               // :2539
            float eii;
            if (t < 253.15f)      eii = 0.001f;                      // :2548
            else if (t < 273.15f) eii = 0.001f + __fdiv_rn((t - 253.15f)
                                         * (0.3f - 0.001f), 20.0f);
            else                  eii = 0.3f;

            {   float ncg = AT(nc, k);
                p3_get_cloud_dsd2(&ncg, AT(qc, k), rho, iscf,
                                  &mu_c, &lamc, &cdist, &cdist1);
                AT(nc, k) = ncg; }
            {   float nrg = AT(nr, k);
                p3_get_rain_dsd2(&nrg, AT(qr, k), ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r);
                AT(nr, k) = nrg; }

            epsi_tot = 0.0f;                                         // :2564
            p3_impose_max_total_ni<NCAT>(ni_, at, inv_rho);          // :2566

#pragma unroll
            for (int c = 0; c < NCAT; ++c) {                         // iice_loop1 :2568
                float* __restrict__ qi = qi_[c];
                float* __restrict__ qir = qir_[c];
                float* __restrict__ ni = ni_[c];
                float* __restrict__ qib = qib_[c];
                float f1pr02 = 0.0f, f1pr03 = 0.0f, f1pr04 = 0.0f;
                float f1pr05 = 0.0f, f1pr09 = 0.0f, f1pr10 = 0.0f;
                float f1pr14 = 0.0f, f1pr07 = -99.0f, f1pr08 = -99.0f;

                if (AT(qi, k) >= P3_QSMALL) {                        // :2570
                    AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
                    AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
                    if (NCAT > 1) {
                        // mean-mass diameter, ice density 500 (:2577-2578);
                        // read only by the nCat > 1 destination and
                        // splintering tests.
                        diam_ice[c] = r_pow((AT(qi, k) * 6.0f)
                                            / (AT(ni, k) * 500.0f * P3_PI),
                                            P3_THRD);
                    }
                    float rhop;
                    {   float qq = AT(qir, k), bb = AT(qib, k);
                        p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                        AT(qir, k) = qq; AT(qib, k) = bb; }
                    int dumi, dumjj, dumii, dumj;
                    float d1, d4, d5, d3;
                    p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                                  &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                    p3_find_lt_1b(AT(qr, k), AT(nr, k), &dumj, &d3);
                    if (!log_3momentIce) {
                    f1pr02 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                                    d1, d4, d5);
                    f1pr03 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 3,
                                                    d1, d4, d5);
                    f1pr04 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 4,
                                                    d1, d4, d5);
                    f1pr05 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 5,
                                                    d1, d4, d5);
                    f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                    d1, d4, d5);
                    f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                    d1, d4, d5);
                    f1pr14 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 10,
                                                    d1, d4, d5);
                    if (AT(qr, k) >= P3_QSMALL) {
                        f1pr07 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                             dumj, dumi, 1,
                                                             d1, d3, d4, d5);
                        f1pr08 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                             dumj, dumi, 2,
                                                             d1, d3, d4, d5);
                    } else {
                        f1pr07 = -99.0f;
                        f1pr08 = -99.0f;
                    }
                    } else {
                        // three-moment ice (:2607-2644)
                        AT(qzi, k) = fmaxf(AT(qzi, k), P3_ZSMALL);       // :2612
                        int dumzz;
                        float d6;
                        p3_mu_i_iterate(itab3, AT(qi, k), AT(ni, k), AT(qzi, k),
                                        dumjj, dumii, dumi, d1, d4, d5,
                                        &dumzz, &d6, &f1pr16);           // :2614-2621
                        f1pr02 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 2, d1, d4, d5, d6);
                        f1pr03 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 3, d1, d4, d5, d6);
                        f1pr04 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 4, d1, d4, d5, d6);
                        f1pr05 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 5, d1, d4, d5, d6);
                        f1pr09 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 7, d1, d4, d5, d6);
                        f1pr10 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 8, d1, d4, d5, d6);
                        f1pr14 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj,
                                     dumii, dumi, 10, d1, d4, d5, d6);
                        if (AT(qr, k) >= P3_QSMALL) {
                            f1pr07 = p3_access_lookup_table_coll_3mom(icoll3,
                                         dumzz, dumjj, dumii, dumj, dumi, 1,
                                         d1, d3, d4, d5, d6);
                            f1pr08 = p3_access_lookup_table_coll_3mom(icoll3,
                                         dumzz, dumjj, dumii, dumj, dumi, 2,
                                         d1, d3, d4, d5, d6);
                        } else {
                            f1pr07 = -99.0f;
                            f1pr08 = -99.0f;
                        }
                    }
                    AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));     // :2649
                    AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));     // :2650
                    if (log_3momentIce) {                                 // :2654-2660
                        AT(qzi, k) = p3_clamp_zitot(AT(qzi, k), AT(qi, k),
                                                    AT(ni, k), f1pr16);
                    }
                    if (AT(qir, k) > 0.0f) {                          // :2665
                        tmp1 = AT(qir, k) / AT(qi, k);
                        if (tmp1 < 0.6f)      eii_fact[c] = 1.0f;
                        else if (tmp1 < 0.9f) eii_fact[c] = 1.0f
                                              - __fdiv_rn((tmp1 - 0.6f), 0.3f);
                        else                  eii_fact[c] = 0.0f;
                    } else {
                        eii_fact[c] = 1.0f;
                    }
                }

                // collection of droplets (:2697-2710)
                if (AT(qi, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL
                    && t <= 273.15f) {
                    qccol[c] = rhofaci * f1pr04 * AT(qc, k) * P3_ECI * rho
                               * AT(ni, k) * iscf;
                    nccol[c] = rhofaci * f1pr04 * AT(nc, k) * P3_ECI * rho
                               * AT(ni, k) * iscf;
                }
                if (AT(qi, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL
                    && t > 273.15f) {
                    qcshd[c] = rhofaci * f1pr04 * AT(qc, k) * P3_ECI * rho
                               * AT(ni, k) * iscf;
                    nccol[c] = rhofaci * f1pr04 * AT(nc, k) * P3_ECI * rho
                               * AT(ni, k) * iscf;
                    ncshdc[c] = qcshd[c] * 1.923e6f;
                }

                // collection of rain (:2725-2748)
                if (AT(qi, k) >= P3_QSMALL && AT(qr, k) >= P3_QSMALL
                    && t <= 273.15f) {
                    qrcol[c] = r_pow(10.0f, f1pr08 + logn0r) * rho * rhofaci
                               * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
                    nrcol[c] = r_pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                               * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
                }
                if (AT(qi, k) >= P3_QSMALL && AT(qr, k) >= P3_QSMALL
                    && t > 273.15f) {
                    nrcol[c] = r_pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                               * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
                }

                // collection between ice categories (:2754-2829): only for
                // iice >= 2, against every lower category.  First the
                // collection of catcoll by iice, then of iice by catcoll;
                // Eii_fact is the COLLECTOR's (:2791, :2820), and the
                // second direction re-forces catcoll's qirim/birim
                // consistency first (:2799) -- a call that WRITES catcoll's
                // rime fields, kept.
                if (NCAT > 1 && c >= 1) {
                    if (AT(qi, k) >= P3_QSMALL) {
                        const float* __restrict__ colli1 = T[T_ICOLLI1];
                        const float* __restrict__ colli2 = T[T_ICOLLI2];
#pragma unroll
                        for (int cc = 0; cc < c; ++cc) {
                            float* __restrict__ qi2 = qi_[cc];
                            float* __restrict__ qir2 = qir_[cc];
                            float* __restrict__ ni2 = ni_[cc];
                            float* __restrict__ qib2 = qib_[cc];
                            if (AT(qi2, k) >= P3_QSMALL) {
                                int di, dii, djj, dic, diic, djjc;
                                float e1, e4, e5, e1c, e4c, e5c;
                                p3_find_lt_2(AT(qi, k), AT(qi2, k), AT(ni, k),
                                             AT(ni2, k), AT(qir, k), AT(qir2, k),
                                             AT(qib, k), AT(qib2, k),
                                             &di, &dii, &djj, &dic, &diic, &djjc,
                                             &e1, &e4, &e5, &e1c, &e4c, &e5c);
                                float f1pr17 = p3_access_lookup_table_colli(
                                    colli1, djjc, diic, dic, djj, dii, di,
                                    e1c, e4c, e5c, e1, e4, e5);
                                float f1pr18 = p3_access_lookup_table_colli(
                                    colli2, djjc, diic, dic, djj, dii, di,
                                    e1c, e4c, e5c, e1, e4, e5);
                                nicol[cc][c] = f1pr17 * rhofaci * rho
                                               * AT(ni2, k) * AT(ni, k) * iscf;
                                qicol[cc][c] = f1pr18 * rhofaci * rho
                                               * AT(ni2, k) * AT(ni, k) * iscf;
                                nicol[cc][c] = eii * eii_fact[c] * nicol[cc][c];
                                qicol[cc][c] = eii * eii_fact[c] * qicol[cc][c];
                                nicol[cc][c] = fminf(nicol[cc][c], AT(ni2, k) * odt);
                                qicol[cc][c] = fminf(qicol[cc][c], AT(qi2, k) * odt);

                                float rhop2;
                                {   float qq = AT(qir2, k), bb = AT(qib2, k);
                                    p3_calc_bulk_rho_rime(AT(qi2, k), &qq, &bb,
                                                          &rhop2);
                                    AT(qir2, k) = qq; AT(qib2, k) = bb; }
                                p3_find_lt_2(AT(qi2, k), AT(qi, k), AT(ni2, k),
                                             AT(ni, k), AT(qir2, k), AT(qir, k),
                                             AT(qib2, k), AT(qib, k),
                                             &di, &dii, &djj, &dic, &diic, &djjc,
                                             &e1, &e4, &e5, &e1c, &e4c, &e5c);
                                f1pr17 = p3_access_lookup_table_colli(
                                    colli1, djjc, diic, dic, djj, dii, di,
                                    e1c, e4c, e5c, e1, e4, e5);
                                f1pr18 = p3_access_lookup_table_colli(
                                    colli2, djjc, diic, dic, djj, dii, di,
                                    e1c, e4c, e5c, e1, e4, e5);
                                nicol[c][cc] = f1pr17 * rhofaci * rho
                                               * AT(ni, k) * AT(ni2, k) * iscf;
                                qicol[c][cc] = f1pr18 * rhofaci * rho
                                               * AT(ni, k) * AT(ni2, k) * iscf;
                                nicol[c][cc] = eii * eii_fact[cc] * nicol[c][cc];
                                qicol[c][cc] = eii * eii_fact[cc] * qicol[c][cc];
                                nicol[c][cc] = fminf(nicol[c][cc], AT(ni, k) * odt);
                                qicol[c][cc] = fminf(qicol[c][cc], AT(qi, k) * odt);
                            }
                        }
                    }
                }

                // self-collection of ice (:2840-2842)
                if (AT(qi, k) >= P3_QSMALL) {
                    nislf[c] = f1pr03 * rho * eii * eii_fact[c] * rhofaci
                               * AT(ni, k) * AT(ni, k) * iscf;
                }

                // melting (:2851-2865)
                if (AT(qi, k) >= P3_QSMALL && t > 273.15f) {
                    float qsat0 = 0.622f * P3_E0 / (AT(pres, k) - P3_E0);
                    dum = 0.0f;
                    qimlt[c] = (__fdiv_rn((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                                 * r_pow(rhofaci * rho / mu, 0.5f))
                                * ((t - 273.15f) * kap
                                   - rho * P3_XXLV * dv * (qsat0 - qv_cld))
                                * 2.0f * P3_PI, P3_XLF) + dum) * AT(ni, k);
                    qimlt[c] = fmaxf(qimlt[c], 0.0f);
                    nimlt[c] = qimlt[c] * (AT(ni, k) / AT(qi, k));
                }

                // wet growth (:2873-2894)
                if (AT(qi, k) >= P3_QSMALL
                    && (AT(qc, k) + AT(qr, k)) >= 1.0e-6f && t < 273.15f) {
                    float qsat0 = 0.622f * P3_E0 / (AT(pres, k) - P3_E0);
                    qwgrth[c] = ((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                                  * r_pow(rhofaci * rho / mu, 0.5f))
                                 * 2.0f * P3_PI
                                 * (rho * P3_XXLV * dv * (qsat0 - qv_cld)
                                    - (t - 273.15f) * kap)
                                 / (P3_XLF + P3_CPW * (t - 273.15f))) * AT(ni, k);
                    qwgrth[c] = fmaxf(qwgrth[c], 0.0f);
                    dum = fmaxf(0.0f, (qccol[c] + qrcol[c]) - qwgrth[c]);
                    if (dum >= 1.0e-10f) {
                        nrshdr[c] = nrshdr[c] + dum * 1.923e6f;
                        if ((qccol[c] + qrcol[c]) >= 1.0e-10f) {
                            dum1 = 1.0f / (qccol[c] + qrcol[c]);
                            qcshd[c] = qcshd[c] + dum * qccol[c] * dum1;
                            qccol[c] = qccol[c] - dum * qccol[c] * dum1;
                            qrcol[c] = qrcol[c] - dum * qrcol[c] * dum1;
                        }
                        log_wetgrowth[c] = 1;
                    }
                }

                // total inverse ice relaxation timescale (:2900-2906)
                if (AT(qi, k) >= P3_QSMALL && t < 273.15f) {
                    epsi[c] = ((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                                * r_pow(rhofaci * rho / mu, 0.5f))
                               * 2.0f * P3_PI * rho * dv) * AT(ni, k);
                    epsi_tot = epsi_tot + epsi[c];
                } else {
                    epsi[c] = 0.0f;
                }

                // rime density, Cober and List 1993 (:2925-2969)
                if (qccol[c] >= P3_QSMALL && t < 273.15f) {
                    float vtrmi1 = f1pr02 * rhofaci;
                    float iTc = 1.0f / fminf(-0.001f, t - 273.15f);
                    if (AT(qc, k) >= P3_QSMALL) {
                        float Vt_qc = acn * p3_gam(4.0f + P3_BCN + mu_c)
                                      / (r_pow(lamc, P3_BCN)
                                         * p3_gam(mu_c + 4.0f));
                        float D_c = (mu_c + 4.0f) / lamc;
                        float V_impact = fabsf(vtrmi1 - Vt_qc);
                        float Ri = -(0.5e6f * D_c) * V_impact * iTc;
                        Ri = fmaxf(1.0f, fminf(Ri, 12.0f));
                        if (Ri <= 8.0f) {
                            rhorime_c[c] = (0.051f + 0.114f * Ri
                                            - 0.0055f * (Ri * Ri)) * 1000.0f;
                        } else {
                            rhorime_c[c] = 611.0f + 72.25f * (Ri - 8.0f);
                        }
                    }
                } else {
                    rhorime_c[c] = 400.0f;
                }
            }   // iice_loop1

            // the qitot(i,k,:)*iSCF(k) every icecat_destination call takes
            // (iSCF = 1 with SCPF off), read at the moment of the call.
            float qd[NCAT];

            // immersion freezing of droplets (:2984-3015)
            //
            // OVERFLOW RESCUE, and the divergence is exactly the overflowing
            // case.  WRF's left-to-right single-precision chain
            // cons6*gamma*tmp1*dum**2 reaches 4.8e45 on the partial product
            // at t = 198.99 K / cdist1 = 3.394e5 / mu_c = 12.576 /
            // lamc = 1.6846e4 -- MEASURED, in the cell that took a real 6 h
            // forecast non-finite at step 284 -- while the mathematical
            // result 4.4e19 is an ordinary float.  The +Inf then meets the
            // conservation limiter at :3571-3583, ratio = sources/sinks is
            // 0, and Inf*0 is the NaN that reached theta and vapour.
            // Fortran does not fix the association of a*b*c*d, so WRF is
            // UNDEFINED here rather than authoritative; P3's own authors
            // left the double form commented out at :2999-3002.  The float
            // chain runs first and unchanged, so every representable value
            // is still WRF's bit for bit.  Mirrors
            // gpuwm/core/p3.py _rescue_overflowed_product exactly.
            if (AT(qc, k) >= P3_QSMALL && t <= 269.15f) {
                float d = 1.0f / lamc;
                dum = (d * d) * d;                                   // :2988 int **3
                tmp1 = cdist1 * r_exp(P3_AIMM * (273.15f - t));
                float gam_q = p3_gam(7.0f + mu_c);
                float gam_n = p3_gam(mu_c + 4.0f);
                float Q_nuc = P3_CONS6 * gam_q * tmp1
                              * (dum * dum);                         // :2995 int **2
                float N_nuc = P3_CONS5 * gam_n * tmp1 * dum;
                if (!isfinite(Q_nuc) || !isfinite(N_nuc)) {
                    // The argument is built in FLOAT32 and only the exp is
                    // double -- exactly the authority's own commented-out
                    // form, dexp(dble(aimm*(273.15-t(i,k)))) at :2999, and
                    // exactly what the rain branch below already does.  A
                    // bare `273.15` here would also be an unsuffixed double
                    // literal, which tests/test_p3_cuda.py refuses in this
                    // file on purpose.
                    double exp_aimm =
                        exp((double)(P3_AIMM * (273.15f - t)));
                    double dcdist1 = (double)cdist1;
                    double ddum = (double)dum;
                    if (!isfinite(Q_nuc)) {
                        Q_nuc = (float)((double)P3_CONS6 * (double)gam_q
                                        * dcdist1 * exp_aimm * ddum * ddum);
                    }
                    if (!isfinite(N_nuc)) {
                        N_nuc = (float)((double)P3_CONS5 * (double)gam_n
                                        * dcdist1 * exp_aimm * ddum);
                    }
                }
                iice_dest = 0;
                if (NCAT > 1) {                                      // :3003-3012
                    dum1 = 900.0f;
                    float D_new = r_pow((Q_nuc * 6.0f) / (P3_PI * dum1 * N_nuc),
                                        P3_THRD);
#pragma unroll
                    for (int c = 0; c < NCAT; ++c) qd[c] = qi_[c][at] * iscf;
                    iice_dest = p3_icecat_destination<NCAT>(
                        qd, diam_ice, D_new, P3_DELTAD_INIT_2CAT);
                }
                qcheti[iice_dest] = Q_nuc;
                ncheti[iice_dest] = N_nuc;
            }

            // immersion freezing of rain (:3021-3043) -- the authority's
            // own double excursion, kept double.
            if (AT(qr, k) * ispf >= P3_QSMALL && t <= 269.15f) {
                double tmpdbl1 = exp((double)(r_log(cdistr)
                                     + r_log(p3_gam(7.0f + mu_r))
                                     - 6.0f * r_log(lamr)));
                double tmpdbl2 = exp((double)(r_log(cdistr)
                                     + r_log(p3_gam(mu_r + 4.0f))
                                     - 3.0f * r_log(lamr)));
                double tmpdbl3 = exp((double)(P3_AIMM * (273.15f - t)));
                float Q_nuc = P3_CONS6 * (float)(tmpdbl1 * tmpdbl3) * spf;
                float N_nuc = P3_CONS5 * (float)(tmpdbl2 * tmpdbl3) * spf;
                // Same class as the droplet branch above: WRF's
                // sngl(tmpdbl1*tmpdbl3) (:3037-3038) can round a double to
                // +Inf for the same cold-cloud reason and reach the same
                // limiter.  The double product is in hand, so the rescue is
                // one more multiply.
                if (!isfinite(Q_nuc)) {
                    Q_nuc = (float)((double)P3_CONS6 * tmpdbl1 * tmpdbl3
                                    * (double)spf);
                }
                if (!isfinite(N_nuc)) {
                    N_nuc = (float)((double)P3_CONS5 * tmpdbl2 * tmpdbl3
                                    * (double)spf);
                }
                iice_dest = 0;
                if (NCAT > 1) {                                      // :3031-3040
                    dum1 = 900.0f;
                    float D_new = r_pow((Q_nuc * 6.0f) / (P3_PI * dum1 * N_nuc),
                                        P3_THRD);
#pragma unroll
                    for (int c = 0; c < NCAT; ++c) qd[c] = qi_[c][at] * iscf;
                    iice_dest = p3_icecat_destination<NCAT>(
                        qd, diam_ice, D_new, P3_DELTAD_INIT_2CAT);
                }
                qrheti[iice_dest] = Q_nuc;
                nrheti[iice_dest] = N_nuc;
            }

            // rime splintering, Hallet-Mossop 1974 (:3049-3106).
            // log_hmossopOn = (nCat > 1) (:2256): off for mp_physics = 50,
            // ON for mp_physics = 52.  Splinters are 10 um and go to the
            // category icecat_destination picks for them, chosen once
            // before the loop (:3051-3058).
            if (NCAT > 1) {
#pragma unroll
                for (int c = 0; c < NCAT; ++c) qd[c] = qi_[c][at] * iscf;
                iice_dest = p3_icecat_destination<NCAT>(
                    qd, diam_ice, 10.0e-6f, P3_DELTAD_INIT_2CAT);
#pragma unroll
                for (int c = 0; c < NCAT; ++c) {                     // iice_loop_HM
                    if (qi_[c][at] >= P3_QSMALL && diam_ice[c] >= 4000.0e-6f
                        && (qccol[c] > 0.0f || qrcol[c] > 0.0f)) {
                        float hm = 0.0f;
                        if (t > 270.15f) {
                            hm = 0.0f;
                        } else if (t <= 270.15f && t > 268.15f) {
                            hm = (270.15f - t) * 0.5f;
                        } else if (t <= 268.15f && t >= 265.15f) {
                            hm = (t - 265.15f) * P3_THRD;
                        } else if (t < 265.15f) {
                            hm = 0.0f;
                        }
                        dum1 = 35.0e4f * qrcol[c] * hm * 1000.0f;    // :3091
                        dum2 = dum1 * P3_PIOV6 * 900.0f * P3_SPLINTER_D3;
                        qrcol[c] = qrcol[c] - dum2;
                        if (qrcol[c] < 0.0f) {
                            dum2 = qrcol[c] + dum2;
                            qrcol[c] = 0.0f;
                        }
                        qrmul[iice_dest] = qrmul[iice_dest] + dum2;
                        nimul[iice_dest] = nimul[iice_dest]
                            + dum2 / (P3_PIOV6 * 900.0f * P3_SPLINTER_D3);
                    }
                }
            }

            // condensation / evaporation / deposition / sublimation
            float epsr;
            if (AT(qr, k) * ispf >= P3_QSMALL) {                     // :3114
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj);
                dum = p3_rain_table(revap, dumii_r, dumjj_r, rdumii, rdumjj);
                epsr = 2.0f * P3_PI * cdistr * rho * dv
                       * (P3_F1R * p3_gam(mu_r + 2.0f) / lamr
                          + P3_F2R * r_pow(rho / mu, 0.5f)
                            * r_pow(sc, P3_THRD) * dum);
            } else {
                epsr = 0.0f;
            }
            float epsc = (AT(qc, k) >= P3_QSMALL)
                         ? 2.0f * P3_PI * rho * dv * cdist : 0.0f;

            float xx;
            if (t < 273.15f) {
                oabi = 1.0f / abi;
                xx = epsc + epsr + epsi_tot
                     * (1.0f + P3_XXLS * P3_INV_CP * dqsdT) * oabi;
            } else {
                oabi = 1.0f / abi;
                xx = epsc + epsr;
            }

            float dumqvi = qvi;                                      // :3144

            dum = -P3_CP / P3_G * (t - t_old) * odt;                 // :3171
            if (t < 273.15f) {
                aaa = (AT(qv, k) - AT(qv_old, k)) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP)
                      - (qvs - dumqvi) * (1.0f + P3_XXLS * P3_INV_CP * dqsdT)
                        * oabi * epsi_tot;
            } else {
                aaa = (AT(qv, k) - AT(qv_old, k)) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP);
            }

            xx = fmaxf(1.0e-20f, xx);                                // :3182
            oxx = 1.0f / xx;

            float ssat_cld = AT(ssat, k);                            // :3186
            float ssat_r = AT(ssat, k);
            float sup_cld = sup, sup_r = sup, supi_cld = supi;

            if (AT(qc, k) >= P3_QSMALL) {                            // :3202
                qccon = (aaa * epsc * oxx
                         + (ssat_cld * scf - aaa * oxx) * odt * epsc * oxx
                           * (1.0f - (float)exp(-(double)(xx * dt)))) / ab;
            }
            if (AT(qr, k) >= P3_QSMALL) {                            // :3204
                qrcon = (aaa * epsr * oxx
                         + (ssat_r * spf - aaa * oxx) * odt * epsr * oxx
                           * (1.0f - (float)exp(-(double)(xx * dt)))) / ab;
            }

            if (sup_cld < -0.001f && AT(qc, k) < 1.0e-12f)
                qccon = -AT(qc, k) * odt;
            if (sup_r < -0.001f && AT(qr, k) < 1.0e-12f)
                qrcon = -AT(qr, k) * odt;

            if (qccon < 0.0f) { qcevp = -qccon; qccon = 0.0f; }
            else              { qccon = fminf(qccon, AT(qv, k) * odt); }

            if (qrcon < 0.0f) {
                qrevp = -qrcon;
                nrevp = qrevp * (AT(nr, k) / AT(qr, k));
                qrcon = 0.0f;
            } else {
                qrcon = fminf(qrcon, AT(qv, k) * odt);
            }

#pragma unroll
            for (int c = 0; c < NCAT; ++c) {                         // iice_loop_depsub :3226
                const float qic = qi_[c][at];
                if (qic >= P3_QSMALL && t < 273.15f) {               // :3228
                    qidep[c] = (aaa * epsi[c] * oxx
                                + (ssat_cld * scf - aaa * oxx) * odt * epsi[c] * oxx
                                  * (1.0f - (float)exp(-(double)(xx * dt)))) * oabi
                               + (qvs - dumqvi) * epsi[c] * oabi;
                }
                if (supi_cld < -0.001f && qic < 1.0e-12f)
                    qidep[c] = -qic * odt;
                if (qidep[c] < 0.0f) {
                    qisub[c] = -qidep[c];
                    qisub[c] = qisub[c] * clbfact_sub;
                    qisub[c] = fminf(qisub[c], qic * odt);
                    nisub[c] = qisub[c] * (ni_[c][at] / qic);
                    qidep[c] = 0.0f;
                } else {
                    qidep[c] = qidep[c] * clbfact_dep;
                    qidep[c] = fminf(qidep[c], AT(qv, k) * odt);
                }
            }
        }
        // ---- 444 continue (:3257) ----------------------------------

        float sup_cld = sup;
        float supi_cld = supi;

        // deposition/condensation-freezing nucleation, Cooper 1986 (:3272)
        if (t < 258.15f && supi_cld >= 0.05f) {
            float nsum[NCAT];
#pragma unroll
            for (int c = 0; c < NCAT; ++c) nsum[c] = ni_[c][at];
            dum = 0.005f * r_exp(0.304f * (273.15f - t)) * 1000.0f * inv_rho;
            dum = fminf(dum, 100.0e3f * inv_rho * scf);
            float N_nuc = fmaxf(0.0f, (dum - p3_sum<NCAT>(nsum)) * odt);
            if (N_nuc >= 1.0e-20f) {
                float Q_nuc = fmaxf(0.0f, (dum - p3_sum<NCAT>(nsum)) * P3_MI0
                                          * odt);
                iice_dest = 0;
                if (NCAT > 1) {                                      // :3281-3289
                    dum1 = 900.0f;
                    float D_new = r_pow((Q_nuc * 6.0f) / (P3_PI * dum1 * N_nuc),
                                        P3_THRD);
                    float qd[NCAT];
#pragma unroll
                    for (int c = 0; c < NCAT; ++c) qd[c] = qi_[c][at] * iscf;
                    iice_dest = p3_icecat_destination<NCAT>(
                        qd, diam_ice, D_new, P3_DELTAD_INIT_2CAT);
                }
                qinuc[iice_dest] = Q_nuc;
                ninuc[iice_dest] = N_nuc;
            }
        }

        // SEAM 2 -- droplet activation, both log_predictNc branches.
        p3_activate_droplets(log_predictNc, sup_cld, it, t, AT(pres, k),
                             AT(qc, k), AT(nc, k), qv_cld, inv_rho, iscf, scf,
                             odt, &qcnuc, &ncnuc);

        // first-step saturation adjustment (:3348-3356)
        if (it <= 1) {
            float dumt = AT(th, k) * r_pow(AT(pres, k) * 1.0e-5f,
                                           P3_RD * P3_INV_CP);
            float dumqv = qv_cld;
            float dumqvs = p3_qv_sat(dumt, AT(pres, k), 0);
            float dums = dumqv - dumqvs;
            qccon = dums / (1.0f + (P3_XXLV * P3_XXLV) * dumqvs
                            / (P3_CP * P3_RV * (dumt * dumt)))
                    * odt * scf;                         // :3353, integer **2
            qccon = fmaxf(0.0f, qccon);
            if (qccon <= 1.0e-7f) qccon = 0.0f;
        }

        // autoconversion, iparam = 3, KK2000 (:3396-3402)
        if (AT(qc, k) * iscf >= 1.0e-8f) {
            dum = AT(qc, k) * iscf;
            qcaut = 1350.0f * r_pow(dum, 2.47f)
                    * r_pow(AT(nc, k) * iscf * 1.0e-6f * rho, -1.79f) * scf;
            ncautr = qcaut * P3_CONS3;
            ncautc = qcaut * AT(nc, k) / AT(qc, k);
            if (qcaut == 0.0f) ncautc = 0.0f;
            if (ncautc == 0.0f) qcaut = 0.0f;
        }

        // self-collection of droplets, iparam = 3 (:3421-3435)
        if (AT(qc, k) >= P3_QSMALL) ncslf = 0.0f;

        // accretion, iparam = 3 (:3458-3462)
        if (AT(qr, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL) {
            dum2 = spf - spf_clr;
            qcacc = 67.0f * r_pow(AT(qc, k) * iscf * AT(qr, k) * ispf, 1.15f)
                    * dum2;
            ncacc = qcacc * AT(nc, k) / AT(qc, k);
            if (qcacc == 0.0f) ncacc = 0.0f;
            if (ncacc == 0.0f) qcacc = 0.0f;
        }

        // rain self-collection / breakup, iparam = 3 (:3478-3504)
        if (AT(qr, k) >= P3_QSMALL) {
            dum1 = 280.0e-6f;
            AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
            dum2 = r_pow(AT(qr, k) / (P3_PI * P3_RHOW * AT(nr, k)), P3_THRD);
            if (dum2 < dum1) dum = 1.0f;
            else             dum = 2.0f - r_exp(2300.0f * (dum2 - dum1));
            nrslf = dum * 5.78f * AT(nr, k) * ispf * AT(qr, k) * ispf
                    * rho * spf;
        }

        // ---- conservation of mass (:3515-3644) ---------------------
        float ratio;
        {
            float dumqvs = p3_qv_sat(t, AT(pres, k), 0);
            float qcon_satadj = (qv_cld - dumqvs)
                / (1.0f + (P3_XXLV * P3_XXLV) * dumqvs
                   / (P3_CP * P3_RV * (t * t))) * odt * scf;  // :3516 int **2
            tmp1 = qccon + qrcon + qcnuc;
            if (tmp1 > 0.0f && qcon_satadj < 0.0f) {
                qccon = 0.0f; qrcon = 0.0f; qcnuc = 0.0f; ncnuc = 0.0f;
            } else {
                if (tmp1 > 0.0f && tmp1 > qcon_satadj) {
                    ratio = fmaxf(0.0f, qcon_satadj) / tmp1;
                    ratio = fminf(1.0f, ratio);
                    qccon = qccon * ratio;
                    qrcon = qrcon * ratio;
                    qcnuc = qcnuc * ratio;
                    ncnuc = ncnuc * ratio;
                } else if (qcevp + qrevp > 0.0f) {
                    ratio = fmaxf(0.0f, -qcon_satadj) / (qcevp + qrevp);
                    ratio = fminf(1.0f, ratio);
                    qcevp = qcevp * ratio;
                    qrevp = qrevp * ratio;
                    nrevp = nrevp * ratio;
                }
            }

            float qv_tmp = qv_cld + (-qcnuc - qccon - qrcon + qcevp + qrevp)
                           * dt;
            float t_tmp = t + (qcnuc + qccon + qrcon - qcevp - qrevp)
                          * P3_XXLV * P3_INV_CP * dt;
            float dumqvi = p3_qv_sat(t_tmp, AT(pres, k), 1);
            float qdep_satadj = (qv_tmp - dumqvi)
                / (1.0f + (P3_XXLS * P3_XXLS) * dumqvi
                   / (P3_CP * P3_RV * (t_tmp * t_tmp))) * odt * scf;
            tmp1 = p3_sum<NCAT>(qidep) + p3_sum<NCAT>(qinuc);       // :3547
            if (tmp1 > 0.0f && qdep_satadj < 0.0f) {
#pragma unroll
                for (int c = 0; c < NCAT; ++c) {
                    qidep[c] = 0.0f; qinuc[c] = 0.0f; ninuc[c] = 0.0f;
                }
            } else {
                if (tmp1 > 0.0f && tmp1 > qdep_satadj) {
                    ratio = fmaxf(0.0f, qdep_satadj) / tmp1;
                    ratio = fminf(1.0f, ratio);
#pragma unroll
                    for (int c = 0; c < NCAT; ++c) {
                        qidep[c] = qidep[c] * ratio;
                        qinuc[c] = qinuc[c] * ratio;
                        ninuc[c] = ninuc[c] * ratio;
                    }
                }
                // sum(qisub) is re-evaluated each pass, after the earlier
                // categories were already rescaled (:3560-3564).
#pragma unroll
                for (int c = 0; c < NCAT; ++c) {
                    dum = fmaxf(qisub[c], 1.0e-20f);
                    qisub[c] = qisub[c] * fminf(1.0f, fmaxf(0.0f, -qdep_satadj)
                                                / fmaxf(p3_sum<NCAT>(qisub),
                                                        1.0e-20f));
                    nisub[c] = nisub[c] * fminf(1.0f, qisub[c] / dum);
                }
            }
        }

        float sinks, sources;
        sinks = (qcaut + qcacc + p3_sum<NCAT>(qccol) + qcevp
                 + p3_sum<NCAT>(qchetc) + p3_sum<NCAT>(qcheti)
                 + p3_sum<NCAT>(qcshd)) * dt;
        sources = AT(qc, k) + (qccon + qcnuc) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3573
            ratio = sources / sinks;
            qcaut *= ratio; qcacc *= ratio; qcevp *= ratio;
#pragma unroll
            for (int c = 0; c < NCAT; ++c) {
                qccol[c] *= ratio; qcheti[c] *= ratio; qcshd[c] *= ratio;
            }
            ncautc *= ratio; ncacc *= ratio;
#pragma unroll
            for (int c = 0; c < NCAT; ++c) {
                nccol[c] *= ratio; ncheti[c] *= ratio;
            }
        }

        sinks = (qrevp + p3_sum<NCAT>(qrcol) + p3_sum<NCAT>(qrhetc)
                 + p3_sum<NCAT>(qrheti) + p3_sum<NCAT>(qrmul)) * dt;
        sources = AT(qr, k) + (qrcon + qcaut + qcacc + p3_sum<NCAT>(qimlt)
                               + p3_sum<NCAT>(qcshd)) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3595
            ratio = sources / sinks;
            qrevp *= ratio;
#pragma unroll
            for (int c = 0; c < NCAT; ++c) {
                qrcol[c] *= ratio; qrheti[c] *= ratio; qrmul[c] *= ratio;
            }
            nrevp *= ratio;
#pragma unroll
            for (int c = 0; c < NCAT; ++c) {
                nrcol[c] *= ratio; nrheti[c] *= ratio;
            }
        }

#pragma unroll
        for (int c = 0; c < NCAT; ++c) {                             // :3609
            sinks = (qisub[c] + qimlt[c]) * dt;
            sources = qi_[c][at] + (qidep[c] + qinuc[c] + qrcol[c] + qccol[c]
                                    + qrhetc[c] + qrheti[c] + qchetc[c]
                                    + qcheti[c] + qrmul[c]) * dt;
#pragma unroll
            for (int cc = 0; cc < NCAT; ++cc) {                      // :3613
                sources = sources + qicol[cc][c] * dt;
                sinks = sinks + qicol[c][cc] * dt;
            }
            if (sinks > sources && sinks >= 1.0e-20f) {
                ratio = sources / sinks;
                qisub[c] *= ratio; qimlt[c] *= ratio;
                nisub[c] *= ratio; nimlt[c] *= ratio;
#pragma unroll
                for (int cc = 0; cc < NCAT; ++cc) {
                    qicol[c][cc] *= ratio;
                    nicol[c][cc] *= ratio;
                }
            }
        }

        sinks = (qccon + qrcon + qcnuc + p3_sum<NCAT>(qidep)
                 + p3_sum<NCAT>(qinuc)) * dt;
        sources = AT(qv, k) + (qcevp + qrevp + p3_sum<NCAT>(qisub)) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3635
            ratio = sources / sinks;
            qccon *= ratio; qrcon *= ratio; qcnuc *= ratio;
#pragma unroll
            for (int c = 0; c < NCAT; ++c) {
                qidep[c] *= ratio; qinuc[c] *= ratio; ninuc[c] *= ratio;
            }
            ncnuc *= ratio;
        }

        // ---- update ice reflectivity (:3652-3743), three-moment ice ---
        // Before the prognostic update, from the start-of-level moments
        // and the final (limited) process rates.  nCat = 1, so the
        // category-interaction terms qicol/nicol are the zeros the
        // authority initialises them to; they are kept so the arithmetic
        // is the authority's term for term.
        if (log_3momentIce) {
            float* __restrict__ qi = qi_[0];
            float* __restrict__ ni = ni_[0];
            const float qicol = 0.0f, nicol = 0.0f;
            float dumm3 = AT(qi, k) + (qidep[0] + qrcol[0] + qccol[0] + qrmul[0]
                                       - qisub[0] - qimlt[0]) * dt;        // :3662
            float dumm0 = AT(ni, k) + (-nimlt[0] - nisub[0] + nimul[0] - nislf[0])
                                      * dt;                          // :3665
            dumm3 = dumm3 - qicol * dt;                              // :3673
            dumm3 = dumm3 + qicol * dt;                              // :3674
            dumm0 = dumm0 - nicol * dt;                              // :3675
            if (dumm3 >= P3_QSMALL) {
                if (AT(qi, k) >= P3_QSMALL) {
                    dumm3 = 6.0f / (f1pr16 * P3_PI) * dumm3;         // :3683
                } else {
                    dumm3 = 6.0f / (900.0f * P3_PI) * dumm3;         // :3686
                }
                float mu_i;
                if (AT(qi, k) >= P3_QSMALL) {
                    dum1 = 6.0f / (f1pr16 * P3_PI) * AT(qi, k);      // :3692
                    mu_i = p3_compute_mu_3moment(AT(ni, k), dum1, AT(qzi, k),
                                                 P3_MU_I_MAX);
                } else {
                    mu_i = P3_MU_I_INITIAL;                          // :3696
                }
                AT(qzi, k) = p3_g_of_mu(mu_i) * (dumm3 * dumm3)
                             / fmaxf(dumm0, P3_NSMALL);              // :3700 int **2
            } else {
                AT(qzi, k) = 0.0f;
            }
            // Group 2, initiation processes (:3711-3733).
            tmp2 = nrhetc[0] + nrheti[0];
            if (tmp2 >= P3_QSMALL) {
                tmp1 = (qrhetc[0] + qrheti[0]) * 6.0f / (900.0f * P3_PI);
                AT(qzi, k) = AT(qzi, k)
                             + p3_g_of_mu(mu_r) * (tmp1 * tmp1) / tmp2 * dt;
            }
            tmp2 = nchetc[0] + ncheti[0];
            if (tmp2 >= P3_QSMALL) {
                tmp1 = (qchetc[0] + qcheti[0]) * 6.0f / (900.0f * P3_PI);
                AT(qzi, k) = AT(qzi, k)
                             + p3_g_of_mu(mu_c) * (tmp1 * tmp1) / tmp2 * dt;
            }
            tmp2 = ninuc[0];
            if (tmp2 >= P3_QSMALL) {
                tmp1 = qinuc[0] * 6.0f / (900.0f * P3_PI);
                AT(qzi, k) = AT(qzi, k)
                             + p3_g_of_mu(P3_MU_I_INITIAL) * (tmp1 * tmp1)
                               / tmp2 * dt;
            }
        }

        // ---- update prognostic variables (:3756-3921) ---------------
        float rimevolume[NCAT], rimefraction[NCAT];
#pragma unroll
        for (int c = 0; c < NCAT; ++c) {                             // iice_loop2 :3756
            rimevolume[c] = 0.0f;
            rimefraction[c] = 0.0f;
            if (qi_[c][at] >= P3_QSMALL) {
                tmp1 = 1.0f / qi_[c][at];
                rimevolume[c] = qib_[c][at] * tmp1;
                rimefraction[c] = qir_[c][at] * tmp1;
            }
        }

#pragma unroll
        for (int c = 0; c < NCAT; ++c) {                             // iice_loop3 :3767
            float* __restrict__ qi = qi_[c];
            float* __restrict__ qir = qir_[c];
            float* __restrict__ ni = ni_[c];
            float* __restrict__ qib = qib_[c];

            AT(qc, k) = AT(qc, k) + (-qchetc[c] - qcheti[c] - qccol[c]
                                     - qcshd[c]) * dt;
            if (log_predictNc) {                                     // :3770
                AT(nc, k) = AT(nc, k) + (-nccol[c] - nchetc[c] - ncheti[c]) * dt;
            }
            AT(qr, k) = AT(qr, k) + (-qrcol[c] + qimlt[c] - qrhetc[c] - qrheti[c]
                                     + qcshd[c] - qrmul[c]) * dt;
            AT(nr, k) = AT(nr, k) + (-nrcol[c] - nrhetc[c] - nrheti[c]
                                     + P3_NMLTRATIO * nimlt[c] + nrshdr[c]
                                     + ncshdc[c]) * dt;

            AT(qib, k) = AT(qib, k) - (qisub[c] + qimlt[c]) * dt * rimevolume[c];
            AT(qir, k) = AT(qir, k) - (qisub[c] + qimlt[c]) * dt * rimefraction[c];
            AT(qi, k) = AT(qi, k) - (qisub[c] + qimlt[c]) * dt;

            dum = (qrcol[c] + qccol[c] + qrhetc[c] + qrheti[c] + qchetc[c]
                   + qcheti[c] + qrmul[c]) * dt;
            AT(qi, k) = AT(qi, k) + (qidep[c] + qinuc[c]) * dt + dum;
            AT(qir, k) = AT(qir, k) + dum;
            AT(qib, k) = AT(qib, k) + (qrcol[c] * P3_INV_RHO_RIMEMAX
                                       + qccol[c] / rhorime_c[c]
                                       + (qrhetc[c] + qrheti[c] + qchetc[c]
                                          + qcheti[c] + qrmul[c])
                                         * P3_INV_RHO_RIMEMAX) * dt;
            AT(ni, k) = AT(ni, k) + (ninuc[c] - nimlt[c] - nisub[c] - nislf[c]
                                     + nrhetc[c] + nrheti[c] + nchetc[c]
                                     + ncheti[c] + nimul[c]) * dt;

            if (NCAT > 1) {                                          // :3801-3827
#pragma unroll
                for (int cc = 0; cc < NCAT; ++cc) {
                    if (cc == c) continue;
                    // nicol is a sink for the collectee, NOT a source for
                    // the collector; rime moves with the collected mass at
                    // the collectee's start-of-update fractions.
                    if (qi_[cc][at] >= P3_QSMALL) {
                        AT(qir, k) = AT(qir, k) + qicol[cc][c] * dt
                                                  * rimefraction[cc];
                        AT(qib, k) = AT(qib, k) + qicol[cc][c] * dt
                                                  * rimevolume[cc];
                        qir_[cc][at] = qir_[cc][at] - qicol[cc][c] * dt
                                                      * rimefraction[cc];
                        qib_[cc][at] = qib_[cc][at] - qicol[cc][c] * dt
                                                      * rimevolume[cc];
                    }
                    qi_[cc][at] = qi_[cc][at] - qicol[cc][c] * dt;
                    ni_[cc][at] = ni_[cc][at] - nicol[cc][c] * dt;
                    AT(qi, k) = AT(qi, k) + qicol[cc][c] * dt;
                }
            }

            if (AT(qir, k) < 0.0f) { AT(qir, k) = 0.0f; AT(qib, k) = 0.0f; }

            if (log_wetgrowth[c]) {                                  // :3835
                AT(qir, k) = AT(qi, k);
                AT(qib, k) = AT(qir, k) * P3_INV_RHO_RIMEMAX;
            }

            if (AT(qi, k) >= P3_QSMALL && AT(qib, k) >= P3_BSMALL
                && qimlt[c] > 0.0f) {                                // :3841
                tmp1 = AT(qir, k) / AT(qib, k);
                tmp2 = AT(qi, k) + qimlt[c] * dt;
                AT(qib, k) = AT(qir, k) / (tmp1 + (917.0f - tmp1) * qimlt[c]
                                           * dt / tmp2);
            }

            AT(qv, k) = AT(qv, k) + (-qidep[c] + qisub[c] - qinuc[c]) * dt;
            AT(th, k) = AT(th, k) + invexn * ((qidep[c] - qisub[c] + qinuc[c])
                                              * P3_XXLS * P3_INV_CP
                                              + (qrcol[c] + qccol[c] + qchetc[c]
                                                 + qcheti[c] + qrhetc[c]
                                                 + qrheti[c] + qrmul[c]
                                                 - qimlt[c])
                                                * P3_XLF * P3_INV_CP) * dt;
        }

        // warm-phase updates (:3869-3885)
        AT(qc, k) = AT(qc, k) + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt;
        AT(qr, k) = AT(qr, k) + (qcacc + qcaut + qrcon - qrevp) * dt;
        if (log_predictNc) {                                         // :3872
            AT(nc, k) = AT(nc, k) + (-ncacc - ncautc + ncslf + ncnuc) * dt;
        } else {
            AT(nc, k) = P3_NCCNST * inv_rho;                         // :3875
        }
        AT(nr, k) = AT(nr, k) + (ncautr - nrslf - nrevp) * dt;       // iparam=3
        AT(qv, k) = AT(qv, k) + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt;
        AT(th, k) = AT(th, k) + invexn * ((qcnuc + qccon + qrcon - qcevp
                                           - qrevp) * P3_XXLV * P3_INV_CP)
                                * dt;

        // clipping (:3889-3918)
        if (AT(qc, k) < P3_QSMALL) {
            AT(qv, k) = AT(qv, k) + AT(qc, k);
            AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
            AT(qc, k) = 0.0f;
            AT(nc, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qr, k) < P3_QSMALL) {
            AT(qv, k) = AT(qv, k) + AT(qr, k);
            AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
            AT(qr, k) = 0.0f;
            AT(nr, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

#pragma unroll
        for (int c = 0; c < NCAT; ++c) {
            float* __restrict__ qi = qi_[c];
            if (AT(qi, k) < P3_QSMALL) {
                AT(qv, k) = AT(qv, k) + AT(qi, k);
                AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
                AT(qi, k) = 0.0f;
                AT(ni_[c], k) = 0.0f;
                AT(qir_[c], k) = 0.0f;
                AT(qib_[c], k) = 0.0f;
            } else {
                hydrometeors_present = 1;
            }
        }

        AT(qv, k) = fmaxf(0.0f, AT(qv, k));
        p3_impose_max_total_ni<NCAT>(ni_, at, inv_rho);              // :3921
        // 555 continue
    }

    // "zero out zitot if there is no qitot for triple moment" after
    // k_loop_main (:3935-3942), every level including the skipped ones.
    if (log_3momentIce) {
        for (int k = k_start; k < nk; k += k_stride) {
            if (AT(F[F_QI], k) < P3_QSMALL) AT(qzi, k) = 0.0f;
        }
    }

    // second compute_SCPF (:3968-3970): refresh the qv snapshot.
    for (int k = k_start; k < nk; k += k_stride) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    return hydrometeors_present;
}

template <int NCAT>
__device__ void p3_step_kloopmain_col(P3_ARGS)
{
    if (FLG[i] == 0.0f && FLG[ncol + i] == 0.0f) return;
    int present = p3_step_kloopmain_levels<NCAT>(P3_PASS, 0, 1);
    FLG[ncol + i] = present ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------
// Sedimentation (:3984-4495).  Three species, each with the authority's
// adaptive Courant substepping.  kbot = 0, ktop = nk-1, kdir = +1 (the
// WRF orientation, :2211-2215).  inv_dzq is 1/dz taken where the
// authority takes it (:2260); it is the same value every time, so it is
// recomputed rather than stored.
// ---------------------------------------------------------------------
__device__ void p3_step_sed_cloud_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;                                      // goto 333
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    const float* __restrict__ dz = F[F_DZ];
    const float iscf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {                             // :4064
        if (AT(qc, k) * iscf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {                          // :4002
        if (AT(qc, k) * iscf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qc, k) * iscf > P3_QSMALL) {
                float mu_c, lamc, t1, t2, ncg = AT(nc, k);
                p3_get_cloud_dsd2(&ncg, AT(qc, k), AT(S[S_RHO], k), iscf,
                                  &mu_c, &lamc, &t1, &t2);
                AT(nc, k) = ncg;
                float dum = 1.0f / r_pow(lamc, P3_BCN);              // :4022
                AT(W[W_VQ], k) = AT(S[S_ACN], k)
                                 * p3_gam(4.0f + P3_BCN + mu_c) * dum
                                 / p3_gam(mu_c + 4.0f);
                if (log_predictNc) {                                 // :4025
                    AT(W[W_VN], k) = AT(S[S_ACN], k)
                                     * p3_gam(1.0f + P3_BCN + mu_c) * dum
                                     / p3_gam(mu_c + 1.0f);
                }
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qc, k) * AT(S[S_RHO], k);
            if (log_predictNc) {
                AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nc, k) * AT(S[S_RHO], k);
            }
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = -AT(W[W_FN], k) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTLIQ][i] = prt_accum * P3_INV_RHOW * odt;
}

__device__ void p3_step_sed_rain_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ vn_tab = T[T_VN];
    const float* __restrict__ vm_tab = T[T_VM];
    const float ispf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qr, k) * ispf > P3_QSMALL) {
                AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
                float mu_r, lamr, cdistr, logn0r, nrg = AT(nr, k);
                p3_get_rain_dsd2(&nrg, AT(qr, k), ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r);
                AT(nr, k) = nrg;
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj);
                float vq = p3_rain_table(vm_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VQ], k) = vq * AT(S[S_RHOFACR], k);
                float vn = p3_rain_table(vn_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VN], k) = vn * AT(S[S_RHOFACR], k);
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qr, k) * AT(S[S_RHO], k);
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nr, k) * AT(S[S_RHO], k);
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            float fdn = -AT(W[W_FN], k) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTLIQ][i] = P[P_PRTLIQ][i] + prt_accum * P3_INV_RHOW * odt;
}

// Ice sedimentation for ONE category (:4251-4495, iice_loop_sedi_ice).  The
// categories run in index order through the same workspace, each adding
// its own surface rate to prt_sol (:4491); a flux the previous category
// left in the workspace is never read, because every substep recomputes
// the fluxes of exactly the levels it then differences.
__device__ void p3_step_sed_ice_col(P3_ARGS, int cat)
{
    if (FLG[ncol + i] == 0.0f) return;
    float* __restrict__ qi = F[p3_fqi(cat)];
    float* __restrict__ qir = F[p3_fqir(cat)];
    float* __restrict__ ni = F[p3_fni(cat)];
    float* __restrict__ qib = F[p3_fqib(cat)];
    float* __restrict__ qzi = F[F_QZI];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ itab = T[T_ITAB];
    const float* __restrict__ itab3 = T[T_ITAB3];
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {
        if (AT(qi, k) >= P3_QSMALL) { log_qxpresent = 1; k_qxtop = k; break; }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {
        if (AT(qi, k) >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
        AT(W[W_FQIR], k) = 0.0f; AT(W[W_FBIR], k) = 0.0f;
        if (log_3momentIce) { AT(W[W_VZ], k) = 0.0f; AT(W[W_FZ], k) = 0.0f; }
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
            if (log_3momentIce) AT(W[W_VZ], k) = 0.0f;               // :4376
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qi, k) >= P3_QSMALL) {
                AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
                float rhop;
                {   float qq = AT(qir, k), bb = AT(qib, k);
                    p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                    AT(qir, k) = qq; AT(qib, k) = bb; }
                int dumi, dumjj, dumii;
                float d1, d4, d5;
                p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                              &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                if (!log_3momentIce) {
                float f1pr01 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 1, d1, d4, d5);
                float f1pr02 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 2, d1, d4, d5);
                float f1pr09 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 7, d1, d4, d5);
                float f1pr10 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 8, d1, d4, d5);
                AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
                AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
                AT(W[W_VQ], k) = f1pr02 * AT(S[S_RHOFACI], k);
                AT(W[W_VN], k) = f1pr01 * AT(S[S_RHOFACI], k);
                } else {
                    // three-moment ice (:4384-4419)
                    AT(qzi, k) = fmaxf(AT(qzi, k), P3_ZSMALL);
                    int dumzz;
                    float d6, f1pr16;
                    p3_mu_i_iterate(itab3, AT(qi, k), AT(ni, k), AT(qzi, k),
                                    dumjj, dumii, dumi, d1, d4, d5,
                                    &dumzz, &d6, &f1pr16);
                    float f1pr01 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 1, d1, d4, d5, d6);
                    float f1pr02 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 2, d1, d4, d5, d6);
                    float f1pr09 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 7, d1, d4, d5, d6);
                    float f1pr10 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 8, d1, d4, d5, d6);
                    float f1pr19 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 13, d1, d4, d5, d6);
                    AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
                    AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
                    AT(W[W_VQ], k) = f1pr02 * AT(S[S_RHOFACI], k);
                    AT(W[W_VN], k) = f1pr01 * AT(S[S_RHOFACI], k);
                    AT(W[W_VZ], k) = f1pr19 * AT(S[S_RHOFACI], k);
                }
            }
            // three-moment ice takes the substep from V_zit, which is
            // larger than V_qit (:4423-4424)
            Co_max = fmaxf(Co_max, AT(W[log_3momentIce ? W_VZ : W_VQ], k)
                                   * dt_left * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            float vq = AT(W[W_VQ], k), rho = AT(S[S_RHO], k);
            AT(W[W_FQ], k) = vq * AT(qi, k) * rho;
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(ni, k) * rho;
            AT(W[W_FQIR], k) = vq * AT(qir, k) * rho;
            AT(W[W_FBIR], k) = vq * AT(qib, k) * rho;
            if (log_3momentIce)
                AT(W[W_FZ], k) = AT(W[W_VZ], k) * AT(qzi, k) * rho;  // :4444
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += (-AT(W[W_FQ], k) * idz) * dt_sub * irho;
            AT(qir, k) += (-AT(W[W_FQIR], k) * idz) * dt_sub * irho;
            AT(qib, k) += (-AT(W[W_FBIR], k) * idz) * dt_sub * irho;
            AT(ni, k) += (-AT(W[W_FN], k) * idz) * dt_sub * irho;
            if (log_3momentIce)
                AT(qzi, k) += (-AT(W[W_FZ], k) * idz) * dt_sub * irho;
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += ((AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz)
                         * dt_sub * irho;
            AT(qir, k) += ((AT(W[W_FQIR], k + 1) - AT(W[W_FQIR], k)) * idz)
                          * dt_sub * irho;
            AT(qib, k) += ((AT(W[W_FBIR], k + 1) - AT(W[W_FBIR], k)) * idz)
                          * dt_sub * irho;
            AT(ni, k) += ((AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz)
                         * dt_sub * irho;
            if (log_3momentIce)
                AT(qzi, k) += ((AT(W[W_FZ], k + 1) - AT(W[W_FZ], k)) * idz)
                              * dt_sub * irho;
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTSOL][i] = P[P_PRTSOL][i] + prt_accum * P3_INV_RHOW * odt;
}

// ---------------------------------------------------------------------
// Level bodies for the last loops.  All are level-local, so the unfused arm
// runs them as separate k-loops and the fused arm as one; these functions
// are what makes those arms the same arithmetic.  The authority's order is
// homogeneous freezing (k_loop_fz :4528-4610), then for nCat > 1 the
// category merge (:4615-4697, step 1 over all k then step 2 over all k),
// then the final diagnostics (:4722-4895); each reads and writes level k
// only, so taking them level by level is the same computation.
// ---------------------------------------------------------------------
template <int NCAT>
__device__ __forceinline__ void p3_homofreeze_level(P3_ARGS, int k)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ th = F[F_TH];
    const float t = AT(S[S_T], k);
    const float invexn = 1.0f / AT(S[S_TMPARR1], k);
    const float iscf = 1.0f;

    float* __restrict__ qzi = F[F_QZI];
    // mean-mass ice diameters at this level (:4531-4538), used only by the
    // nCat > 1 destination below.
    float diam_ice[NCAT];
    if (NCAT > 1) {
#pragma unroll
        for (int c = 0; c < NCAT; ++c) {
            diam_ice[c] = 0.0f;
            float* __restrict__ qi = F[p3_fqi(c)];
            if (AT(qi, k) >= P3_QSMALL) {
                float dum1 = fmaxf(AT(F[p3_fni(c)], k), P3_NSMALL);
                diam_ice[c] = r_pow((AT(qi, k) * 6.0f) / (dum1 * 500.0f * P3_PI),
                                    P3_THRD);
            }
        }
    }

    if (AT(qc, k) >= P3_QSMALL && t < 233.15f) {                     // :4540
        float Q_nuc = AT(qc, k);
        AT(nc, k) = fmaxf(AT(nc, k), P3_NSMALL);
        float N_nuc = AT(nc, k);
        int dest = 0;
        if (NCAT > 1) {                                              // :4546-4555
            float D_new = r_pow((Q_nuc * 6.0f) / (P3_PI * 900.0f * N_nuc),
                                P3_THRD);
            float qd[NCAT];
#pragma unroll
            for (int c = 0; c < NCAT; ++c) qd[c] = AT(F[p3_fqi(c)], k) * iscf;
            dest = p3_icecat_destination<NCAT>(qd, diam_ice, D_new,
                                               P3_DELTAD_INIT_2CAT);
        }
#pragma unroll
        for (int c = 0; c < NCAT; ++c) {
            if (c != dest) continue;
            float* __restrict__ qi = F[p3_fqi(c)];
            float* __restrict__ qir = F[p3_fqir(c)];
            float* __restrict__ ni = F[p3_fni(c)];
            float* __restrict__ qib = F[p3_fqib(c)];
            AT(qir, k) = AT(qir, k) + Q_nuc;
            AT(qi, k) = AT(qi, k) + Q_nuc;
            AT(qib, k) = AT(qib, k) + Q_nuc * P3_INV_RHO_RIMEMAX;
            AT(ni, k) = AT(ni, k) + N_nuc;
        if (log_3momentIce && N_nuc >= P3_NSMALL) {                 // :4564-4570
            float tmp1 = Q_nuc * 6.0f / (900.0f * P3_PI);
            // get_cloud_dsd2 on the not-yet-zeroed qc/nc, for mu_c; its
            // nc write-back is overwritten by nc = 0 below, as in WRF.
            float mu_c, lamc, cdist, cdist1, ncg = AT(nc, k);
            p3_get_cloud_dsd2(&ncg, AT(qc, k), AT(S[S_RHO], k), 1.0f,
                              &mu_c, &lamc, &cdist, &cdist1);
            AT(nc, k) = ncg;
            AT(qzi, k) = AT(qzi, k) + p3_g_of_mu(mu_c) * (tmp1 * tmp1) / N_nuc;
        }
        }
        AT(th, k) = AT(th, k) + invexn * Q_nuc * P3_XLF * P3_INV_CP;
        AT(qc, k) = 0.0f;
        AT(nc, k) = 0.0f;
    }
    if (AT(qr, k) >= P3_QSMALL && t < 233.15f) {                     // :4578
        float Q_nuc = AT(qr, k);
        AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
        float N_nuc = AT(nr, k);
        int dest = 0;
        if (NCAT > 1) {                                              // :4583-4591
            float D_new = r_pow((Q_nuc * 6.0f) / (P3_PI * 900.0f * N_nuc),
                                P3_THRD);
            float qd[NCAT];
#pragma unroll
            for (int c = 0; c < NCAT; ++c) qd[c] = AT(F[p3_fqi(c)], k) * iscf;
            dest = p3_icecat_destination<NCAT>(qd, diam_ice, D_new,
                                               P3_DELTAD_INIT_2CAT);
        }
#pragma unroll
        for (int c = 0; c < NCAT; ++c) {
            if (c != dest) continue;
            float* __restrict__ qi = F[p3_fqi(c)];
            float* __restrict__ qir = F[p3_fqir(c)];
            float* __restrict__ ni = F[p3_fni(c)];
            float* __restrict__ qib = F[p3_fqib(c)];
            AT(qir, k) = AT(qir, k) + Q_nuc;
            AT(qi, k) = AT(qi, k) + Q_nuc;
            AT(qib, k) = AT(qib, k) + Q_nuc * P3_INV_RHO_RIMEMAX;
            AT(ni, k) = AT(ni, k) + N_nuc;
        if (log_3momentIce && N_nuc >= P3_QSMALL) {
            float tmp1 = Q_nuc * 6.0f / (900.0f * P3_PI);
            AT(qzi, k) = AT(qzi, k)
                         + p3_g_of_mu(P3_MU_R_CONSTANT) * (tmp1 * tmp1) / N_nuc;
        }
        }
        AT(th, k) = AT(th, k) + invexn * Q_nuc * P3_XLF * P3_INV_CP;
        AT(qr, k) = 0.0f;
        AT(nr, k) = 0.0f;
    }
}

// The category merge (:4615-4697), nCat > 1 only.  Step 1 clips empty
// categories into vapour and takes every non-empty category's mass-weighted
// mean diameter from the table (f1pr15); step 2 folds category iice into
// iice-1 when the two diameters are within deltaD_init.  The step-1 diag_di
// is overwritten by the final diagnostics for every category, so it is a
// local here.
template <int NCAT>
__device__ __forceinline__ void p3_merge_level(P3_ARGS, int k)
{
    if (NCAT < 2) return;
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ itab = T[T_ITAB];
    const float invexn = 1.0f / AT(S[S_TMPARR1], k);
    float di[NCAT];
#pragma unroll
    for (int c = 0; c < NCAT; ++c) {                                 // :4620
        float* __restrict__ qi = F[p3_fqi(c)];
        float* __restrict__ qir = F[p3_fqir(c)];
        float* __restrict__ ni = F[p3_fni(c)];
        float* __restrict__ qib = F[p3_fqib(c)];
        if (AT(qi, k) >= P3_QSMALL) {
            AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
            float rhop;
            {   float qq = AT(qir, k), bb = AT(qib, k);
                p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                AT(qir, k) = qq; AT(qib, k) = bb; }
            int dumi, dumjj, dumii;
            float d1, d4, d5;
            p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                          &dumi, &dumjj, &dumii, &d1, &d4, &d5);
            di[c] = p3_access_lookup_table(itab, dumjj, dumii, dumi, 11,
                                           d1, d4, d5);
        } else {
            AT(qv, k) = AT(qv, k) + AT(qi, k);
            AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
            di[c] = 0.0f;
        }
    }
#pragma unroll
    for (int c = NCAT - 1; c >= 1; --c) {                            // :4678
        float* __restrict__ qi = F[p3_fqi(c)];
        float* __restrict__ qir = F[p3_fqir(c)];
        float* __restrict__ ni = F[p3_fni(c)];
        float* __restrict__ qib = F[p3_fqib(c)];
        float* __restrict__ qi0 = F[p3_fqi(c - 1)];
        float* __restrict__ qir0 = F[p3_fqir(c - 1)];
        float* __restrict__ ni0 = F[p3_fni(c - 1)];
        float* __restrict__ qib0 = F[p3_fqib(c - 1)];
        float tmp1 = fabsf(di[c] - di[c - 1]);
        if (tmp1 <= P3_DELTAD_INIT_2CAT && AT(qi, k) > 0.0f
            && AT(qi0, k) > 0.0f) {
            AT(qi0, k) = AT(qi0, k) + AT(qi, k);
            AT(ni0, k) = AT(ni0, k) + AT(ni, k);
            AT(qir0, k) = AT(qir0, k) + AT(qir, k);
            AT(qib0, k) = AT(qib0, k) + AT(qib, k);
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
        }
    }
}

// final checks + diagnostics (:4722-4895).  ze_ice/ze_rain are the
// authority's (i,k) arrays but are written and read only at this level, so
// they live in registers instead of a scratch field.  For NCAT = 2 the
// level also does mp_p3_wrapper_wrf_2cat's per-level output step
// (:1134-1169): category 1 and 2 fall speed / size / density to their own
// slots, and the effective radius combined by mass and projected area.
template <int NCAT>
__device__ __forceinline__ void p3_final_level(P3_ARGS, int k)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ itab = T[T_ITAB];
    const float iscf = 1.0f;
    const float rho = AT(S[S_RHO], k);
    const float inv_rho = AT(S[S_INVRHO], k);
    const float invexn = 1.0f / AT(S[S_TMPARR1], k);
    const size_t at = (size_t)k * (size_t)ncol + (size_t)i;
    float ze_ice = 1.0e-22f, ze_rain = 1.0e-22f;                     // :2277-2278

    if (AT(qc, k) * iscf >= P3_QSMALL) {
        float mu_c, lamc, t1, t2, ncg = AT(nc, k);
        p3_get_cloud_dsd2(&ncg, AT(qc, k), rho, iscf, &mu_c, &lamc, &t1, &t2);
        AT(nc, k) = ncg;
        AT(D[D_EFFC], k) = 0.5f * (mu_c + 3.0f) / lamc;              // :4728
    } else {
        AT(qv, k) = AT(qv, k) + AT(qc, k);
        AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
        AT(qc, k) = 0.0f;
        AT(nc, k) = 0.0f;
    }

    if (AT(qr, k) >= P3_QSMALL) {
        float mu_r, lamr, cdistr, logn0r, nrg = AT(nr, k);
        // the authority passes the LITERAL iSPF = 1. here (:4739)
        p3_get_rain_dsd2(&nrg, AT(qr, k), 1.0f,
                         &mu_r, &lamr, &cdistr, &logn0r);
        AT(nr, k) = nrg;
        float l2 = lamr * lamr;
#if defined(P3_NATIVE_GNU239)
        // GNU 13 -O2 lowers integer **6 to (x*x*x)*(x*x*x).
        float l3 = l2 * lamr;
        float lam6 = l3 * l3;
#else
        float lam6 = (l2 * l2) * l2;                     // :4756, integer **6
#endif
        ze_rain = rho * AT(nr, k) * (mu_r + 6.0f) * (mu_r + 5.0f)
                  * (mu_r + 4.0f) * (mu_r + 3.0f) * (mu_r + 2.0f)
                  * (mu_r + 1.0f) / lam6;
        ze_rain = fmaxf(ze_rain, 1.0e-22f);
    } else {
        AT(qv, k) = AT(qv, k) + AT(qr, k);
        AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
        AT(qr, k) = 0.0f;
        AT(nr, k) = 0.0f;
    }

    float* ni_[NCAT];
#pragma unroll
    for (int c = 0; c < NCAT; ++c) ni_[c] = F[p3_fni(c)];
    p3_impose_max_total_ni<NCAT>(ni_, at, inv_rho);                  // :4767

    // p3_main's per-category diag_effi, pre-set 25.e-6 (:2281) and written
    // only for a category holding ice.
    float effi[NCAT];
#pragma unroll
    for (int c = 0; c < NCAT; ++c) {                                 // :4769
        float* __restrict__ qi = F[p3_fqi(c)];
        float* __restrict__ qir = F[p3_fqir(c)];
        float* __restrict__ ni = F[p3_fni(c)];
        float* __restrict__ qib = F[p3_fqib(c)];
        const int d_vmi = (c == 0) ? D_VMI : D_VMI2;
        const int d_di = (c == 0) ? D_DI : D_DI2;
        const int d_rhopo = (c == 0) ? D_RHOPO : D_RHOPO2;
        effi[c] = 25.0e-6f;
        if (AT(qi, k) >= P3_QSMALL) {
            AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
            AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
            float rhop;
            {   float qq = AT(qir, k), bb = AT(qib, k);
                p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                AT(qir, k) = qq; AT(qib, k) = bb; }
            int dumi, dumjj, dumii;
            float d1, d4, d5;
            p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                          &dumi, &dumjj, &dumii, &d1, &d4, &d5);
            float f1pr02, f1pr06, f1pr09, f1pr10, f1pr13, f1pr15, f1pr16;
            if (!log_3momentIce) {
            f1pr02 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                            d1, d4, d5);
            f1pr06 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 6,
                                            d1, d4, d5);
            f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                            d1, d4, d5);
            f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                            d1, d4, d5);
            f1pr13 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                            d1, d4, d5);
            f1pr15 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 11,
                                            d1, d4, d5);
            f1pr16 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 12,
                                            d1, d4, d5);
            } else {
                // three-moment ice (:4795-4829).  lambda_i and mu_i (14, 15)
                // feed only diag_lami/diag_mui, which WRF's call does not pass.
                const float* __restrict__ itab3 = T[T_ITAB3];
                AT(F[F_QZI], k) = fmaxf(AT(F[F_QZI], k), P3_ZSMALL);    // :4804
                int dumzz;
                float d6;
                p3_mu_i_iterate(itab3, AT(qi, k), AT(ni, k), AT(F[F_QZI], k),
                                dumjj, dumii, dumi, d1, d4, d5, &dumzz, &d6,
                                &f1pr16);
                f1pr02 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 2, d1, d4, d5, d6);
                f1pr06 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 6, d1, d4, d5, d6);
                f1pr09 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 7, d1, d4, d5, d6);
                f1pr10 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 8, d1, d4, d5, d6);
                f1pr13 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 9, d1, d4, d5, d6);
                f1pr15 = p3_access_lookup_table_3mom(itab3, dumzz, dumjj, dumii,
                                                     dumi, 11, d1, d4, d5, d6);
            }
            AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
            AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
            if (log_3momentIce) {                                        // :4838-4846
                AT(F[F_QZI], k) = p3_clamp_zitot(AT(F[F_QZI], k), AT(qi, k),
                                                 AT(ni, k), f1pr16);
            }
            if (AT(qir, k) < P3_QSMALL) { AT(qir, k) = 0.0f; AT(qib, k) = 0.0f; }
            AT(D[d_vmi], k) = f1pr02 * AT(S[S_RHOFACI], k);
            effi[c] = f1pr06;
            if (NCAT == 1) AT(D[D_EFFI], k) = f1pr06;
            AT(D[d_di], k) = f1pr15;
            AT(D[d_rhopo], k) = f1pr16;
            ze_ice = ze_ice + 0.1892f * f1pr13 * AT(ni, k) * rho;
            ze_ice = fmaxf(ze_ice, 1.0e-22f);
        } else {
            AT(qv, k) = AT(qv, k) + AT(qi, k);
            AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
            if (log_3momentIce) AT(F[F_QZI], k) = 0.0f;
            AT(D[d_di], k) = 0.0f;
        }
    }

    AT(D[D_ZDBZ], k) = 10.0f * p3_log10((ze_rain + ze_ice) * 1.0e18f);

    if (AT(qr, k) < P3_QSMALL) AT(nr, k) = 0.0f;

    if (NCAT > 1) {
        // mp_p3_wrapper_wrf_2cat (:1155-1169): the combined ice effective
        // radius, weighting each category by mass over its own radius.
        float* __restrict__ qi1 = F[p3_fqi(0)];
        float* __restrict__ qi2 = F[p3_fqi(1)];
        float dum1, dum2;
        if (AT(qi1, k) >= P3_QSMALL) dum1 = AT(qi1, k) / effi[0];
        else                         dum1 = 0.0f;
        if (AT(qi2, k) >= P3_QSMALL) dum2 = AT(qi2, k) / effi[NCAT - 1];
        else                         dum2 = 0.0f;
        AT(D[D_EFFI], k) = 25.0e-6f;
        if (AT(qi1, k) >= P3_QSMALL || AT(qi2, k) >= P3_QSMALL) {
            AT(D[D_EFFI], k) = (AT(qi1, k) + AT(qi2, k)) / (dum1 + dum2);
        }
    }
}

template <int NCAT>
__device__ void p3_step_homofreeze_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    // third compute_SCPF (:4521-4523): refresh the qv snapshot.  Nothing in
    // the ported call shape reads it again, but it is where the Sundqvist
    // branch attaches, so the refresh stays.
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    for (int k = 0; k < nk; ++k) p3_homofreeze_level<NCAT>(P3_PASS, k);
}

template <int NCAT>
__device__ void p3_step_final_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    for (int k = 0; k < nk; ++k) p3_merge_level<NCAT>(P3_PASS, k);
    for (int k = 0; k < nk; ++k) p3_final_level<NCAT>(P3_PASS, k);
}

// The fused form of the two above: one k-loop instead of three.  Valid
// because every body reads and writes level k only; the byte gate is what
// proves the compiler agrees.
template <int NCAT>
__device__ void p3_step_homofreeze_final_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    for (int k = 0; k < nk; ++k) {
        p3_homofreeze_level<NCAT>(P3_PASS, k);
        p3_merge_level<NCAT>(P3_PASS, k);
        p3_final_level<NCAT>(P3_PASS, k);
    }
}

// Save end-of-microphysics theta/qv for the next step (:5018-5021).  These
// are the cross-step carriers the diagnosed-ssat branch depends on, and
// they are why gpuwm/io/restart.py serializes th_old/qv_old.
__device__ void p3_step_saveold_col(P3_ARGS)
{
    for (int k = 0; k < nk; ++k) {
        AT(F[F_THOLD], k) = AT(F[F_TH], k);
        AT(F[F_QVOLD], k) = AT(F[F_QV], k);
        p3_zi_finish_level(P3_PASS, k);
    }
}

// mp_p3_wrapper_wrf's precipitation conversions (:892-898; the same lines
// in mp_p3_wrapper_wrf_2cat, :1107-1112).
__device__ void p3_step_precip_col(P3_ARGS)
{
    float dum1 = 1000.0f * dt;
    float prt_liq = P[P_PRTLIQ][i];
    float prt_sol = P[P_PRTSOL][i];
    float total = prt_liq + prt_sol;
    P[P_RAINNC][i] = P[P_RAINNC][i] + total * dum1;
    P[P_RAINNCV][i] = total * dum1;
    P[P_SNOWNC][i] = P[P_SNOWNC][i] + prt_sol * dum1;
    P[P_SNOWNCV][i] = prt_sol * dum1;
    P[P_SR][i] = prt_sol / (prt_liq + prt_sol + 1.0e-12f);
}

// =====================================================================
// The arms.
//
// UNFUSED (9 launches) is the REFERENCE.  One step per kernel, in the
// authority's order.  It is what the per-field agreement against the
// Fortran is measured on, and it is what any fused variant must reproduce
// byte for byte.
//
// FUSED (3 launches) composes the SAME __device__ step functions.  Two of
// its three kernels are pure launch elimination; the third additionally
// merges the homogeneous-freezing, merge and final-diagnostics k-loops into
// one.  Because the bodies are level-local that is arithmetically the same
// computation -- and "arithmetically the same" is a claim the byte gate in
// tests/test_p3_cuda.py checks rather than a claim this comment makes.
//
// Every kernel exists twice: the mp_physics = 50 name (NCAT = 1) and the
// same name with "_c2" (NCAT = 2, mp_physics = 52).
// =====================================================================

#define P3_TID \
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ncol) return

#define P3_KERNEL_ARGS \
    float* const* __restrict__ F, float* const* __restrict__ S, \
    float* const* __restrict__ W, float* const* __restrict__ D, \
    float* const* __restrict__ P, const float* const* __restrict__ T, \
    float* __restrict__ FLG, int ncol, int nk, float dt, int it, \
    int log_predictNc, int log_3momentIce, float clbfact_dep,     float clbfact_sub
#define P3_KERNEL_PASS     F, S, W, D, P, T, FLG, ncol, nk, dt, it, log_predictNc,     log_3momentIce, clbfact_dep, clbfact_sub

// One body template, two extern "C" entry points.
#define P3_BOTH_CATEGORY_COUNTS(name) \
    extern "C" __global__ void name(P3_KERNEL_ARGS) \
    { name##_body<1>(P3_KERNEL_PASS); } \
    extern "C" __global__ void name##_c2(P3_KERNEL_ARGS) \
    { name##_body<2>(P3_KERNEL_PASS); }

template <int NCAT> __device__ void p3k_prep_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_prep_col<NCAT>(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_prep)

template <int NCAT> __device__ void p3k_kloop1_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_kloop1_col<NCAT>(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_kloop1)

template <int NCAT> __device__ void p3k_kloopmain_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_kloopmain_col<NCAT>(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_kloopmain)

template <int NCAT> __device__ void p3k_sed_cloud_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_sed_cloud_col(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_sed_cloud)

template <int NCAT> __device__ void p3k_sed_rain_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_sed_rain_col(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_sed_rain)

template <int NCAT> __device__ void p3k_sed_ice_body(P3_KERNEL_ARGS)
{
    P3_TID;
#pragma unroll
    for (int c = 0; c < NCAT; ++c) p3_step_sed_ice_col(P3_PASS, c);
}
P3_BOTH_CATEGORY_COUNTS(p3k_sed_ice)

template <int NCAT> __device__ void p3k_homofreeze_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_homofreeze_col<NCAT>(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_homofreeze)

template <int NCAT> __device__ void p3k_final_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_final_col<NCAT>(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_final)

template <int NCAT> __device__ void p3k_saveold_precip_body(P3_KERNEL_ARGS)
{ P3_TID; p3_step_saveold_col(P3_PASS); p3_step_precip_col(P3_PASS); }
P3_BOTH_CATEGORY_COUNTS(p3k_saveold_precip)

// Fluxes are completed before any level updates in each substep.
// The only cross-level reduction is the exact Courant maximum.
__device__ void p3_step_sed_cloud_levels(P3_ARGS, float (*co)[4])
{
    int enabled = i < ncol && FLG[ncol + i] != 0.0f;
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    const float* __restrict__ dz = F[F_DZ];
    const float iscf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; enabled && k >= kbot; --k) {                             // :4064
        if (AT(qc, k) * iscf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    enabled = enabled && log_qxpresent;

    float dt_left = enabled ? dt : 0.0f, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; enabled && k <= k_qxtop; ++k) {                          // :4002
        if (AT(qc, k) * iscf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = threadIdx.y; enabled && k < nk; k += 16) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (__syncthreads_or(dt_left > 1.0e-4f)) {
        float Co_max = 0.0f;
        for (int k = k_qxtop - threadIdx.y; dt_left > 1.0e-4f && k >= k_qxbot; k -= 16) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
            if (AT(qc, k) * iscf > P3_QSMALL) {
                float mu_c, lamc, t1, t2, ncg = AT(nc, k);
                p3_get_cloud_dsd2(&ncg, AT(qc, k), AT(S[S_RHO], k), iscf,
                                  &mu_c, &lamc, &t1, &t2);
                AT(nc, k) = ncg;
                float dum = 1.0f / r_pow(lamc, P3_BCN);              // :4022
                AT(W[W_VQ], k) = AT(S[S_ACN], k)
                                 * p3_gam(4.0f + P3_BCN + mu_c) * dum
                                 / p3_gam(mu_c + 4.0f);
                if (log_predictNc) {                                 // :4025
                    AT(W[W_VN], k) = AT(S[S_ACN], k)
                                     * p3_gam(1.0f + P3_BCN + mu_c) * dum
                                     / p3_gam(mu_c + 1.0f);
                }
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        co[threadIdx.y][threadIdx.x] = Co_max;
        __syncthreads();
        Co_max = co[0][threadIdx.x];
        for (int group = 1; group < 16; ++group)
            Co_max = fmaxf(Co_max, co[group][threadIdx.x]);
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp + threadIdx.y; dt_left > 1.0e-4f && k <= k_qxtop; k += 16) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qc, k) * AT(S[S_RHO], k);
            if (log_predictNc) {
                AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nc, k) * AT(S[S_RHO], k);
            }
        }
        __syncthreads();
        if (dt_left > 1.0e-4f) {
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        if (threadIdx.y == 0) {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = -AT(W[W_FN], k) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        for (int k = k_qxtop - 1 - threadIdx.y; k >= k_temp; k -= 16) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
        }
    }
    if (enabled && threadIdx.y == 0) P[P_PRTLIQ][i] = prt_accum * P3_INV_RHOW * odt;
}


__device__ void p3_step_sed_rain_levels(P3_ARGS, float (*co)[4])
{
    int enabled = i < ncol && FLG[ncol + i] != 0.0f;
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ vn_tab = T[T_VN];
    const float* __restrict__ vm_tab = T[T_VM];
    const float ispf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; enabled && k >= kbot; --k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    enabled = enabled && log_qxpresent;

    float dt_left = enabled ? dt : 0.0f, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; enabled && k <= k_qxtop; ++k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = threadIdx.y; enabled && k < nk; k += 16) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (__syncthreads_or(dt_left > 1.0e-4f)) {
        float Co_max = 0.0f;
        for (int k = k_qxtop - threadIdx.y; dt_left > 1.0e-4f && k >= k_qxbot; k -= 16) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
            if (AT(qr, k) * ispf > P3_QSMALL) {
                AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
                float mu_r, lamr, cdistr, logn0r, nrg = AT(nr, k);
                p3_get_rain_dsd2(&nrg, AT(qr, k), ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r);
                AT(nr, k) = nrg;
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj);
                float vq = p3_rain_table(vm_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VQ], k) = vq * AT(S[S_RHOFACR], k);
                float vn = p3_rain_table(vn_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VN], k) = vn * AT(S[S_RHOFACR], k);
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        co[threadIdx.y][threadIdx.x] = Co_max;
        __syncthreads();
        Co_max = co[0][threadIdx.x];
        for (int group = 1; group < 16; ++group)
            Co_max = fmaxf(Co_max, co[group][threadIdx.x]);
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp + threadIdx.y; dt_left > 1.0e-4f && k <= k_qxtop; k += 16) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qr, k) * AT(S[S_RHO], k);
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nr, k) * AT(S[S_RHO], k);
        }
        __syncthreads();
        if (dt_left > 1.0e-4f) {
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        if (threadIdx.y == 0) {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            float fdn = -AT(W[W_FN], k) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        for (int k = k_qxtop - 1 - threadIdx.y; k >= k_temp; k -= 16) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
        }
    }
    if (enabled && threadIdx.y == 0) P[P_PRTLIQ][i] = P[P_PRTLIQ][i] + prt_accum * P3_INV_RHOW * odt;
}


// One ice category's sedimentation by a level group (see
// p3_step_sed_ice_col for the category order and the shared workspace).
__device__ void p3_step_sed_ice_levels(P3_ARGS, float (*co)[4], int cat)
{
    int enabled = i < ncol && FLG[ncol + i] != 0.0f;
    float* __restrict__ qi = F[p3_fqi(cat)];
    float* __restrict__ qir = F[p3_fqir(cat)];
    float* __restrict__ ni = F[p3_fni(cat)];
    float* __restrict__ qib = F[p3_fqib(cat)];
    float* __restrict__ qzi = F[F_QZI];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ itab = T[T_ITAB];
    const float* __restrict__ itab3 = T[T_ITAB3];
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; enabled && k >= kbot; --k) {
        if (AT(qi, k) >= P3_QSMALL) { log_qxpresent = 1; k_qxtop = k; break; }
    }
    enabled = enabled && log_qxpresent;

    float dt_left = enabled ? dt : 0.0f, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; enabled && k <= k_qxtop; ++k) {
        if (AT(qi, k) >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = threadIdx.y; enabled && k < nk; k += 16) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
        AT(W[W_FQIR], k) = 0.0f; AT(W[W_FBIR], k) = 0.0f;
        if (log_3momentIce) { AT(W[W_VZ], k) = 0.0f; AT(W[W_FZ], k) = 0.0f; }
    }

    while (__syncthreads_or(dt_left > 1.0e-4f)) {
        float Co_max = 0.0f;
        for (int k = k_qxtop - threadIdx.y; dt_left > 1.0e-4f && k >= k_qxbot; k -= 16) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
            if (log_3momentIce) AT(W[W_VZ], k) = 0.0f;
            if (AT(qi, k) >= P3_QSMALL) {
                AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
                float rhop;
                {   float qq = AT(qir, k), bb = AT(qib, k);
                    p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                    AT(qir, k) = qq; AT(qib, k) = bb; }
                int dumi, dumjj, dumii;
                float d1, d4, d5;
                p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                              &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                if (!log_3momentIce) {
                float f1pr01 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 1, d1, d4, d5);
                float f1pr02 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 2, d1, d4, d5);
                float f1pr09 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 7, d1, d4, d5);
                float f1pr10 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 8, d1, d4, d5);
                AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
                AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
                AT(W[W_VQ], k) = f1pr02 * AT(S[S_RHOFACI], k);
                AT(W[W_VN], k) = f1pr01 * AT(S[S_RHOFACI], k);
                } else {
                    AT(qzi, k) = fmaxf(AT(qzi, k), P3_ZSMALL);
                    int dumzz;
                    float d6, f1pr16;
                    p3_mu_i_iterate(itab3, AT(qi, k), AT(ni, k), AT(qzi, k),
                                    dumjj, dumii, dumi, d1, d4, d5,
                                    &dumzz, &d6, &f1pr16);
                    float f1pr01 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 1, d1, d4, d5, d6);
                    float f1pr02 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 2, d1, d4, d5, d6);
                    float f1pr09 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 7, d1, d4, d5, d6);
                    float f1pr10 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 8, d1, d4, d5, d6);
                    float f1pr19 = p3_access_lookup_table_3mom(itab3, dumzz,
                                       dumjj, dumii, dumi, 13, d1, d4, d5, d6);
                    AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
                    AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
                    AT(W[W_VQ], k) = f1pr02 * AT(S[S_RHOFACI], k);
                    AT(W[W_VN], k) = f1pr01 * AT(S[S_RHOFACI], k);
                    AT(W[W_VZ], k) = f1pr19 * AT(S[S_RHOFACI], k);
                }
            }
            Co_max = fmaxf(Co_max, AT(W[log_3momentIce ? W_VZ : W_VQ], k)
                                   * dt_left * (1.0f / AT(dz, k)));
        }
        co[threadIdx.y][threadIdx.x] = Co_max;
        __syncthreads();
        Co_max = co[0][threadIdx.x];
        for (int group = 1; group < 16; ++group)
            Co_max = fmaxf(Co_max, co[group][threadIdx.x]);
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp + threadIdx.y; dt_left > 1.0e-4f && k <= k_qxtop; k += 16) {
            float vq = AT(W[W_VQ], k), rho = AT(S[S_RHO], k);
            AT(W[W_FQ], k) = vq * AT(qi, k) * rho;
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(ni, k) * rho;
            AT(W[W_FQIR], k) = vq * AT(qir, k) * rho;
            AT(W[W_FBIR], k) = vq * AT(qib, k) * rho;
            if (log_3momentIce)
                AT(W[W_FZ], k) = AT(W[W_VZ], k) * AT(qzi, k) * rho;
        }
        __syncthreads();
        if (dt_left > 1.0e-4f) {
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        if (threadIdx.y == 0) {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += (-AT(W[W_FQ], k) * idz) * dt_sub * irho;
            AT(qir, k) += (-AT(W[W_FQIR], k) * idz) * dt_sub * irho;
            AT(qib, k) += (-AT(W[W_FBIR], k) * idz) * dt_sub * irho;
            AT(ni, k) += (-AT(W[W_FN], k) * idz) * dt_sub * irho;
            if (log_3momentIce)
                AT(qzi, k) += (-AT(W[W_FZ], k) * idz) * dt_sub * irho;
        }
        for (int k = k_qxtop - 1 - threadIdx.y; k >= k_temp; k -= 16) {
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += ((AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz)
                         * dt_sub * irho;
            AT(qir, k) += ((AT(W[W_FQIR], k + 1) - AT(W[W_FQIR], k)) * idz)
                          * dt_sub * irho;
            AT(qib, k) += ((AT(W[W_FBIR], k + 1) - AT(W[W_FBIR], k)) * idz)
                          * dt_sub * irho;
            AT(ni, k) += ((AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz)
                         * dt_sub * irho;
            if (log_3momentIce)
                AT(qzi, k) += ((AT(W[W_FZ], k + 1) - AT(W[W_FZ], k)) * idz)
                              * dt_sub * irho;
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
        }
    }
    if (enabled && threadIdx.y == 0) P[P_PRTSOL][i] = P[P_PRTSOL][i] + prt_accum * P3_INV_RHOW * odt;
}



// Level groups share each column. The reduction combines only flags.
// Each level retains the original update expressions and statement order.
template <int NCAT> __device__ void p3k_levels_prepare_body(P3_KERNEL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int flags[8][16];
    int f = 0;
    if (i < ncol) {
        // Complete both level-local stages before advancing k.
        for (int k = threadIdx.y; k < nk; k += 8) {
            p3_step_prep_levels<NCAT>(P3_PASS, k, nk);
            f |= p3_step_kloop1_levels<NCAT>(P3_PASS, k, nk);
        }
    }
    flags[threadIdx.y][threadIdx.x] = f;
    __syncthreads();
    if (i < ncol && threadIdx.y == 0) {
        f = flags[0][threadIdx.x];
        for (int group = 1; group < 8; ++group)
            f |= flags[group][threadIdx.x];
        FLG[i] = (f & 1) ? 1.0f : 0.0f;
        FLG[ncol + i] = (f & 2) ? 1.0f : 0.0f;
        P[P_PRTLIQ][i] = 0.0f;
        P[P_PRTSOL][i] = 0.0f;
    }
}
P3_BOTH_CATEGORY_COUNTS(p3k_levels_prepare)

template <int NCAT> __device__ void p3k_levels_sed_body(P3_KERNEL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float co[16][4];
    p3_step_sed_cloud_levels(P3_PASS, co);
    p3_step_sed_rain_levels(P3_PASS, co);
#pragma unroll
    for (int c = 0; c < NCAT; ++c) p3_step_sed_ice_levels(P3_PASS, co, c);
}
P3_BOTH_CATEGORY_COUNTS(p3k_levels_sed)

template <int NCAT> __device__ void p3k_levels_finish_body(P3_KERNEL_ARGS)
{
    P3_TID;
    int k = blockIdx.y;
    if (FLG[ncol + i] != 0.0f) {
        AT(S[S_QVCLD], k) = AT(F[F_QV], k);
        p3_homofreeze_level<NCAT>(P3_PASS, k);
        p3_merge_level<NCAT>(P3_PASS, k);
        p3_final_level<NCAT>(P3_PASS, k);
    }
    AT(F[F_THOLD], k) = AT(F[F_TH], k);
    AT(F[F_QVOLD], k) = AT(F[F_QV], k);
    p3_zi_finish_level(P3_PASS, k);
    if (k == 0) p3_step_precip_col(P3_PASS);
}
P3_BOTH_CATEGORY_COUNTS(p3k_levels_finish)

template <int NCAT> __device__ void p3k_levels_process_body(P3_KERNEL_ARGS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int flags[16][8];
    int f = 0;
    if (i < ncol) f = p3_step_kloopmain_levels<NCAT>(P3_PASS, threadIdx.y, 16);
    flags[threadIdx.y][threadIdx.x] = f;
    __syncthreads();
    if (i < ncol && threadIdx.y == 0) {
        f = flags[0][threadIdx.x];
        for (int group = 1; group < 16; ++group)
            f |= flags[group][threadIdx.x];
        FLG[ncol + i] = f ? 1.0f : 0.0f;
    }
}
P3_BOTH_CATEGORY_COUNTS(p3k_levels_process)

template <int NCAT> __device__ void p3k_fused_process_body(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_prep_col<NCAT>(P3_PASS);
    p3_step_kloop1_col<NCAT>(P3_PASS);
    p3_step_kloopmain_col<NCAT>(P3_PASS);
}
P3_BOTH_CATEGORY_COUNTS(p3k_fused_process)

template <int NCAT> __device__ void p3k_fused_sed_body(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_sed_cloud_col(P3_PASS);
    p3_step_sed_rain_col(P3_PASS);
#pragma unroll
    for (int c = 0; c < NCAT; ++c) p3_step_sed_ice_col(P3_PASS, c);
}
P3_BOTH_CATEGORY_COUNTS(p3k_fused_sed)

template <int NCAT> __device__ void p3k_fused_finish_body(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_homofreeze_final_col<NCAT>(P3_PASS);
    p3_step_saveold_col(P3_PASS);
    p3_step_precip_col(P3_PASS);
}
P3_BOTH_CATEGORY_COUNTS(p3k_fused_finish)

// ---------------------------------------------------------------------
// Contraction control-probe.  tests/test_p3_cuda.py compiles the module
// twice and compares these two kernels: under -fmad=false the infix form
// this file is written in must equal the explicit _rn intrinsics bit for
// bit, and under -fmad=true it must NOT.  That is what makes "plain
// operators are safe here" a measurement instead of an assumption.
// ---------------------------------------------------------------------
extern "C" __global__ void p3k_probe_infix(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           const float* __restrict__ c,
                                           float* __restrict__ o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] = a[i] * b[i] + c[i];
}

extern "C" __global__ void p3k_probe_rn(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        const float* __restrict__ c,
                                        float* __restrict__ o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] = __fadd_rn(__fmul_rn(a[i], b[i]), c[i]);
}
