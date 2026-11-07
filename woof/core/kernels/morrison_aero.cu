// gpuwm/core/kernels/morrison_aero.cu
//
// WRF v4.6.1 Morrison two-moment microphysics with CESM-NCSU prescribed
// aerosol, mp_physics = 40.  Transcription authority:
//   phys/module_mp_morr_two_moment_aero.F (6182 lines, commit d66e442)
// in WRF statement order.  Line numbers below are that file's.
//
// Layout.  The WRF column subroutine MORR_TWO_MOMENT_MICRO has three loops
// over K: the process loop (1570-4278), sedimentation (4302-4611) and the
// final loop (4613-5002).  Every iteration of the process and final loops
// reads only its own level (IBASE = 2 is fixed at :387, so the IBASE = 1
// cloud-base test that reads K-1 is unreachable), so they run one thread per
// cell.  Sedimentation runs one thread per (column, category), ten category
// lanes per column inside one warp: a category's split-step sweep reads only
// its own profile, the step count NSTEP is the column maximum (a warp
// reduction), and the surface precipitation sums of :4604-4609 are formed
// from the five mass lanes in WRF's order through warp shuffles.
//
// Arithmetic.  Default REAL is binary32.  Each Fortran statement is written
// with its own association: A*B/C is (A*B)/C, sums run left to right, unary
// minus binds below * and /.  Divisions go through __fdiv_rn (NVRTC turns a
// division by a literal into a reciprocal multiply on compute_120).  The
// gfortran -O0 lowering the oracle was compiled with (and which the -O2
// build matches word for word on the oracle fixture):
//   REAL**REAL     -> powf, every constant exponent included (2.0, 0.5, 1.5)
//   REAL**(-1.0)   -> 1/x
//   REAL**INTEGER  -> __powisf2 (square-and-multiply, ma_powi below)
//   MAX/MIN        -> maxss/minss with a constant operand in the
//                     destination: a tie or NaN returns the variable
// EXP/LOG/LOG10/** call WOOF's own float32 routines (r_exp, r_log, r_log10,
// r_pow, prepended to this unit), SQRT is the correctly rounded square
// root, and GAMMA/DERF1 are WRF's own REAL FUNCTIONs transcribed below.
// Under GPUWM_WRF_EXACT=1 the unit compiles with --fmad=false --ftz=false
// and is graded word for word against the WRF column oracle
// (tools/morrison_aero_wrf461_oracle). Default mode allows contraction
// and flushes subnormals; strict arithmetic is opt-in.
//
// Constants.  Every value MORR_TWO_MOMENT_INIT_AERO computes (CONS1-41, the
// slope limits, MI0, MMULT, the Abdul-Razzak-Ghan tables) is the FP32 word
// WRF's own init holds, read from gpuwm/data/morrison/constants.toml and
// handed to the kernels as one array (MA_K below).
//
// Declared divergences from the pinned WRF source (each one a WRF defect):
//   D1  The activation thermodynamics (:643, :5725-5727) read T0, the
//       dynamics base-state temperature (300 K), where the Abdul-Razzak-Ghan
//       formulas use a melting-point convention in the correction arm.
//       An explicit diagnostic arm uses SVPT0 = 273.15 K. Default and strict
//       use pinned T0 = 300 K; the table carries both sets of init words.
//   mdm_prescribed_nucleati preserves WRF's fatal [0,9999] cm-3 range
//   check through an error plane; the launcher raises before sedimentation.
//   D3  An immersion-freezing rate whose binary32 evaluation overflows
//       (MNUCCR :3383, MNUCCC/NNUCCC :2985-2991) becomes Inf and then NaN
//       through the conservation ratio; WOOF bounds an overflowed rate by
//       the donor's whole content over the step before conservation.  The
//       guard reads only non-finite values, so every finite WRF result is
//       unchanged.

#define MA_KMAX_SHALLOW 64
#ifndef MA_KMAX
#define MA_KMAX 256
#endif

// ---------------------------------------------------------------------
// Constant array layout (gpuwm/core/morrison_aero.py MA_CONSTANT_NAMES)
// ---------------------------------------------------------------------
#define MA_NDCNST 0
#define MA_AI 1
#define MA_AC 2
#define MA_AS 3
#define MA_AR 4
#define MA_AG 5
#define MA_BI 6
#define MA_BC 7
#define MA_BS 8
#define MA_BR 9
#define MA_BG 10
#define MA_RHOSU 11
#define MA_RHOW 12
#define MA_RHOI 13
#define MA_RHOSN 14
#define MA_RHOG 15
#define MA_AIMM 16
#define MA_BIMM 17
#define MA_ECR 18
#define MA_DCS 19
#define MA_MI0 20
#define MA_MG0 21
#define MA_F1S 22
#define MA_F2S 23
#define MA_F1R 24
#define MA_F2R 25
#define MA_QSMALL 26
#define MA_CI 27
#define MA_DI 28
#define MA_CS 29
#define MA_DS 30
#define MA_CG 31
#define MA_DG 32
#define MA_EII 33
#define MA_ECI 34
#define MA_RIN 35
#define MA_CPW 36
#define MA_MW 37
#define MA_MA 38
#define MA_RR 39
#define MA_MMULT 40
#define MA_LAMMAXI 41
#define MA_LAMMINI 42
#define MA_LAMMAXR 43
#define MA_LAMMINR 44
#define MA_LAMMAXS 45
#define MA_LAMMINS 46
#define MA_LAMMAXG 47
#define MA_LAMMING 48
#define MA_CONS1 49           /* CONS1..CONS41 occupy 49..89 */
#define MA_ATEN 90
#define MA_ALOGATEN 91
#define MA_ALOG2 92
#define MA_ALOG3 93
#define MA_THIRD 94
#define MA_SQ2 95
#define MA_ALOGSIG 96         /* 10 */
#define MA_EXP45LOGSIG 106    /* 10 */
#define MA_SMCRIT 116         /* 10 */
#define MA_F1AER 126          /* 10 */
#define MA_F2AER 136          /* 10 */
#define MA_HYGRO 146          /* 10 */
#define MA_DENSITY 156        /* 10 */
#define MA_NUM_TO_MASS 166    /* 10 */
#define MA_CCNFACT 176        /* 7 x 10, supersaturation fastest */
#define MA_NCONST 246

#define K_(name) (kc[MA_##name])
#define CONS(n) (kc[MA_CONS1 + (n) - 1])

// WRF module_model_constants PARAMETERs (binary32 words)
#define MA_CP __int_as_float(0x447B2000)      /* 7.*r_d/2. = 1004.5 */
#define MA_G __int_as_float(0x411CF5C3)       /* 9.81 */
#define MA_R __int_as_float(0x438F8000)       /* r_d = 287. */
#define MA_RV __int_as_float(0x43E6CCCD)      /* r_v = 461.6 */
#define MA_EP_2 __int_as_float(0x3F1F2B09)    /* r_d/r_v */
#if MA_T0_MILLIKELVIN == 273150
#define MA_SVPT0 273.15f  /* explicit diagnostic correction arm */
#else
#define MA_SVPT0 300.0f  /* pinned WRF T0 */
#endif
#define MA_SVP1 __int_as_float(0x3F1C779A)    /* 0.6112 */
#define MA_SVP2 __int_as_float(0x418D5C29)    /* 17.67 */
#define MA_SVP3 __int_as_float(0x41ED3333)    /* 29.65 */
#define MA_PI __int_as_float(0x40490FDB)      /* module PARAMETER PI */
#define MA_XXX __int_as_float(0x3F6B3F8E)     /* GAMMA's 0.9189385... */
#define MA_REAL_TINY __int_as_float(0x00800000)

#define DIV(a, b) __fdiv_rn((a), (b))

// Fortran MAX/MIN as gfortran -O0 emits them against a constant: maxss/
// minss with the constant in the destination, so a tie (+0 against -0) or
// a NaN returns the variable.
__device__ __forceinline__ real ma_maxc(real c, real v) { return (c > v) ? c : v; }
__device__ __forceinline__ real ma_minc(real c, real v) { return (c < v) ? c : v; }
// MAX(a, b) / MIN(a, b) of two variables: maxss with a in the destination.
__device__ __forceinline__ real ma_max2(real a, real b) { return (a > b) ? a : b; }
__device__ __forceinline__ real ma_min2(real a, real b) { return (a < b) ? a : b; }

// REAL**INTEGER: libgcc __powisf2, square and multiply.
__device__ __forceinline__ real ma_powi(real x, int m)
{
    unsigned int n = (m < 0) ? (unsigned int)(-m) : (unsigned int)m;
    real y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = x * x;
        if (n % 2u) y = y * x;
    }
    return (m < 0) ? DIV(1.0f, y) : y;
}

#define POW(a, b) r_pow((a), (b))
#define EXPF(a) r_exp(a)
#define LOGF(a) r_log(a)
#define SQRTF(a) __fsqrt_rn(a)

// ---------------------------------------------------------------------
// REAL FUNCTION POLYSVP (:5012-5096)
// ---------------------------------------------------------------------
__device__ real ma_polysvp(real t, int type)
{
    real polysvp = 0.0f;
    if (type == 1) {
        if (t >= 195.8f) {
            real dt = t - 273.15f;
            polysvp = 6.11147274f + dt * (0.503160820f + dt * (0.188439774e-1f
                    + dt * (0.420895665e-3f + dt * (0.615021634e-5f
                    + dt * (0.602588177e-7f + dt * (0.385852041e-9f
                    + dt * (0.146898966e-11f + 0.252751365e-14f * dt)))))));
            polysvp = polysvp * 100.0f;
        } else {
            // alog10(6.1071) is a constant expression gfortran folds.
            polysvp = POW(10.0f, -9.09718f * (DIV(273.16f, t) - 1.0f)
                      - 3.56654f * r_log10(DIV(273.16f, t))
                      + 0.876793f * (1.0f - DIV(t, 273.16f))
                      + __int_as_float(0x3F492C7C)) * 100.0f;
        }
    }
    if (type == 0) {
        if (t >= 202.0f) {
            real dt = t - 273.15f;
            polysvp = 6.11239921f + dt * (0.443987641f + dt * (0.142986287e-1f
                    + dt * (0.264847430e-3f + dt * (0.302950461e-5f
                    + dt * (0.206739458e-7f + dt * (0.640689451e-10f
                    + dt * (-0.952447341e-13f + -0.976195544e-15f * dt)))))));
            polysvp = polysvp * 100.0f;
        } else {
            // 10**(...) with an INTEGER base is real**real after promotion;
            // alog10(1013.246) is folded.
            polysvp = POW(10.0f, -7.90298f * (DIV(373.16f, t) - 1.0f)
                      + 5.02808f * r_log10(DIV(373.16f, t))
                      - 1.3816e-7f * (POW(10.0f, 11.344f * (1.0f - DIV(t, 373.16f))) - 1.0f)
                      + 8.1328e-3f * (POW(10.0f, -3.49149f * (DIV(373.16f, t) - 1.0f)) - 1.0f)
                      + __int_as_float(0x40405DA2)) * 100.0f;
        }
    }
    return polysvp;
}

// ---------------------------------------------------------------------
// REAL FUNCTION GAMMA (:5100-5331), W. J. Cody's rational approximation
// ---------------------------------------------------------------------
__device__ const float MA_GAM_P[8] = {
    -1.71618513886549492533811E+0f, 2.47656508055759199108314E+1f,
    -3.79804256470945635097577E+2f, 6.29331155312818442661052E+2f,
    8.66966202790413211295064E+2f, -3.14512729688483675254357E+4f,
    -3.61444134186911729807069E+4f, 6.64561438202405440627855E+4f};
__device__ const float MA_GAM_Q[8] = {
    -3.08402300119738975254353E+1f, 3.15350626979604161529144E+2f,
    -1.01515636749021914166146E+3f, -3.10777167157231109440444E+3f,
    2.25381184209801510330112E+4f, 4.75584627752788110767815E+3f,
    -1.34659959864969306392456E+5f, -1.15132259675553483497211E+5f};
__device__ const float MA_GAM_C[7] = {
    -1.910444077728E-03f, 8.4171387781295E-04f,
    -5.952379913043012E-04f, 7.93650793500350248E-04f,
    -2.777777777777681622553E-03f, 8.333333333333333331554247E-02f,
    5.7083835261E-03f};

__device__ real ma_gamma(real x)
{
    const real one = 1.0f, half = 0.5f, twelve = 12.0f, two = 2.0f, zero = 0.0f;
    const real xbig = 35.040f, xminin = 1.18E-38f, eps = 1.19E-7f, xinf = 3.4E38f;
    bool parity = false;
    real fact = one, res, y, y1, z, xnum, xden, ysq, sum;
    int n = 0;
    y = x;
    if (y <= zero) {
        // Negative arguments (:5236-5251).  Morrison only ever evaluates
        // GAMMA at a positive PGAM+n or a positive constant, but keep the
        // statement sequence.
        y = -x;
        y1 = truncf(y);
        res = y - y1;
        if (res != zero) {
            if (y1 != truncf(y1 * half) * two) parity = true;
            fact = DIV(-MA_PI, sinf(MA_PI * res));
            y = y + one;
        } else {
            return xinf;
        }
    }
    if (y < eps) {
        if (y >= xminin) {
            res = DIV(one, y);
        } else {
            return xinf;
        }
    } else if (y < twelve) {
        y1 = y;
        if (y < one) {
            z = y;
            y = y + one;
        } else {
            n = (int)y - 1;
            y = y - (real)n;
            z = y - one;
        }
        xnum = zero;
        xden = one;
        for (int i = 0; i < 8; ++i) {
            xnum = (xnum + MA_GAM_P[i]) * z;
            xden = xden * z + MA_GAM_Q[i];
        }
        res = DIV(xnum, xden) + one;
        if (y1 < y) {
            res = DIV(res, y1);
        } else if (y1 > y) {
            for (int i = 1; i <= n; ++i) {
                res = res * y;
                y = y + one;
            }
        }
    } else {
        if (y <= xbig) {
            ysq = y * y;
            sum = MA_GAM_C[6];
            for (int i = 0; i < 6; ++i) sum = DIV(sum, ysq) + MA_GAM_C[i];
            sum = DIV(sum, y) - y + MA_XXX;
            sum = sum + (y - half) * LOGF(y);
            res = EXPF(sum);
        } else {
            return xinf;
        }
    }
    if (parity) res = -res;
    if (fact != one) res = DIV(fact, res);
    return res;
}

// ---------------------------------------------------------------------
// REAL FUNCTION DERF1 (:5334-5445)
// ---------------------------------------------------------------------
__device__ const float MA_ERF_A[65] = {
    0.00000000005958930743E0f, -0.00000000113739022964E0f,
    0.00000001466005199839E0f, -0.00000016350354461960E0f,
    0.00000164610044809620E0f, -0.00001492559551950604E0f,
    0.00012055331122299265E0f, -0.00085483269811296660E0f,
    0.00522397762482322257E0f, -0.02686617064507733420E0f,
    0.11283791670954881569E0f, -0.37612638903183748117E0f,
    1.12837916709551257377E0f,
    0.00000000002372510631E0f, -0.00000000045493253732E0f,
    0.00000000590362766598E0f, -0.00000006642090827576E0f,
    0.00000067595634268133E0f, -0.00000621188515924000E0f,
    0.00005103883009709690E0f, -0.00037015410692956173E0f,
    0.00233307631218880978E0f, -0.01254988477182192210E0f,
    0.05657061146827041994E0f, -0.21379664776456006580E0f,
    0.84270079294971486929E0f,
    0.00000000000949905026E0f, -0.00000000018310229805E0f,
    0.00000000239463074000E0f, -0.00000002721444369609E0f,
    0.00000028045522331686E0f, -0.00000261830022482897E0f,
    0.00002195455056768781E0f, -0.00016358986921372656E0f,
    0.00107052153564110318E0f, -0.00608284718113590151E0f,
    0.02986978465246258244E0f, -0.13055593046562267625E0f,
    0.67493323603965504676E0f,
    0.00000000000382722073E0f, -0.00000000007421598602E0f,
    0.00000000097930574080E0f, -0.00000001126008898854E0f,
    0.00000011775134830784E0f, -0.00000111992758382650E0f,
    0.00000962023443095201E0f, -0.00007404402135070773E0f,
    0.00050689993654144881E0f, -0.00307553051439272889E0f,
    0.01668977892553165586E0f, -0.08548534594781312114E0f,
    0.56909076642393639985E0f,
    0.00000000000155296588E0f, -0.00000000003032205868E0f,
    0.00000000040424830707E0f, -0.00000000471135111493E0f,
    0.00000005011915876293E0f, -0.00000048722516178974E0f,
    0.00000430683284629395E0f, -0.00003445026145385764E0f,
    0.00024879276133931664E0f, -0.00162940941748079288E0f,
    0.00988786373932350462E0f, -0.05962426839442303805E0f,
    0.49766113250947636708E0f};
__device__ const float MA_ERF_B[65] = {
    -0.00000000029734388465E0f, 0.00000000269776334046E0f,
    -0.00000000640788827665E0f, -0.00000001667820132100E0f,
    -0.00000021854388148686E0f, 0.00000266246030457984E0f,
    0.00001612722157047886E0f, -0.00025616361025506629E0f,
    0.00015380842432375365E0f, 0.00815533022524927908E0f,
    -0.01402283663896319337E0f, -0.19746892495383021487E0f,
    0.71511720328842845913E0f,
    -0.00000000001951073787E0f, -0.00000000032302692214E0f,
    0.00000000522461866919E0f, 0.00000000342940918551E0f,
    -0.00000035772874310272E0f, 0.00000019999935792654E0f,
    0.00002687044575042908E0f, -0.00011843240273775776E0f,
    -0.00080991728956032271E0f, 0.00661062970502241174E0f,
    0.00909530922354827295E0f, -0.20160072778491013140E0f,
    0.51169696718727644908E0f,
    0.00000000003147682272E0f, -0.00000000048465972408E0f,
    0.00000000063675740242E0f, 0.00000003377623323271E0f,
    -0.00000015451139637086E0f, -0.00000203340624738438E0f,
    0.00001947204525295057E0f, 0.00002854147231653228E0f,
    -0.00101565063152200272E0f, 0.00271187003520095655E0f,
    0.02328095035422810727E0f, -0.16725021123116877197E0f,
    0.32490054966649436974E0f,
    0.00000000002319363370E0f, -0.00000000006303206648E0f,
    -0.00000000264888267434E0f, 0.00000002050708040581E0f,
    0.00000011371857327578E0f, -0.00000211211337219663E0f,
    0.00000368797328322935E0f, 0.00009823686253424796E0f,
    -0.00065860243990455368E0f, -0.00075285814895230877E0f,
    0.02585434424202960464E0f, -0.11637092784486193258E0f,
    0.18267336775296612024E0f,
    -0.00000000000367789363E0f, 0.00000000020876046746E0f,
    -0.00000000193319027226E0f, -0.00000000435953392472E0f,
    0.00000018006992266137E0f, -0.00000078441223763969E0f,
    -0.00000675407647949153E0f, 0.00008428418334440096E0f,
    -0.00017604388937031815E0f, -0.00239729611435071610E0f,
    0.02064129023876022970E0f, -0.06905562880005864105E0f,
    0.09084526782065478489E0f};

__device__ real ma_derf1(real x)
{
    real w = fabsf(x), t, y;
    int k;
    if ((double)w < 2.2) {
        t = w * w;
        k = (int)t;
        t = t - (real)k;
        k = k * 13;
        y = ((((((((((((MA_ERF_A[k] * t + MA_ERF_A[k + 1]) * t
            + MA_ERF_A[k + 2]) * t + MA_ERF_A[k + 3]) * t + MA_ERF_A[k + 4]) * t
            + MA_ERF_A[k + 5]) * t + MA_ERF_A[k + 6]) * t + MA_ERF_A[k + 7]) * t
            + MA_ERF_A[k + 8]) * t + MA_ERF_A[k + 9]) * t + MA_ERF_A[k + 10]) * t
            + MA_ERF_A[k + 11]) * t + MA_ERF_A[k + 12]) * w;
    } else if ((double)w < 6.9) {
        k = (int)w;
        t = w - (real)k;
        k = 13 * (k - 2);
        y = (((((((((((MA_ERF_B[k] * t + MA_ERF_B[k + 1]) * t
            + MA_ERF_B[k + 2]) * t + MA_ERF_B[k + 3]) * t + MA_ERF_B[k + 4]) * t
            + MA_ERF_B[k + 5]) * t + MA_ERF_B[k + 6]) * t + MA_ERF_B[k + 7]) * t
            + MA_ERF_B[k + 8]) * t + MA_ERF_B[k + 9]) * t + MA_ERF_B[k + 10]) * t
            + MA_ERF_B[k + 11]) * t + MA_ERF_B[k + 12];
        y = y * y;
        y = y * y;
        y = y * y;
        y = 1.0f - y * y;
    } else {
        y = 1.0f;
    }
    if (x < 0.0f) y = -y;
    return y;
}

// ---------------------------------------------------------------------
// mdm_prescribed_maxsat (:5793-5844)
// ---------------------------------------------------------------------
__device__ real ma_maxsat(const real* __restrict__ kc, real zeta,
                          const real* eta, const real* smc)
{
    bool significant = false;
    for (int m = 0; m < 10; ++m) {
        if (zeta > 1.e5f * eta[m] || smc[m] * smc[m] > 1.e5f * eta[m]) {
            // weak forcing for this mode
        } else {
            significant = true;
            break;
        }
    }
    if (!significant) return 1.e-20f;
    real sum = 0.0f;
    for (int m = 0; m < 10; ++m) {
        if (eta[m] > 1.e-20f) {
            real g1 = SQRTF(DIV(zeta, eta[m]));
            g1 = g1 * g1 * g1;
            real g2 = DIV(smc[m], SQRTF(eta[m] + 3.0f * zeta));
            g2 = SQRTF(g2);
            g2 = g2 * g2 * g2;
            sum = sum + DIV(kc[MA_F1AER + m] * g1 + kc[MA_F2AER + m] * g2,
                            smc[m] * smc[m]);
        } else {
            sum = 1.e20f;
        }
    }
    return DIV(1.0f, SQRTF(sum));
}

// ---------------------------------------------------------------------
// mdm_prescribed_activate (:5628-5791): activated number (# kg-1)
// ---------------------------------------------------------------------
__device__ real ma_activate(const real* __restrict__ kc, real wbar,
                            real tair, real rhoair, const real* naero,
                            const real* maero, real latvap)
{
    real nact = 0.0f;
    // nmode = 10, so the single-mode early return (:5720) never fires.
    if (wbar <= 0.0f) return nact;
    const real p0 = 1013.25e2f;
    real pres = MA_R * rhoair * tair;
    // D1: SVPT0 where WRF reads the base-state T0 (:5725-5727).
    real diff0 = 0.211e-4f * DIV(p0, pres) * POW(DIV(tair, MA_SVPT0), 1.94f);
    real conduct0 = (5.69f + 0.017f * (tair - MA_SVPT0)) * 4.186e2f * 1.e-5f;
    real es = 1000.0f * MA_SVP1 * EXPF(DIV(MA_SVP2 * (tair - MA_SVPT0),
                                           tair - MA_SVP3));
    real qs = DIV(MA_EP_2 * es, pres - es);
    real dqsdt = DIV(latvap, MA_RV * tair * tair) * qs;
    real alpha_cesm = MA_G * (DIV(latvap, MA_CP * MA_RV * tair * tair)
                              - DIV(1.0f, MA_R * tair));
    real gammaloc = DIV(1.0f + DIV(latvap, MA_CP) * dqsdt, rhoair * qs);
    real gloc = DIV(1.0f, DIV(K_(RHOW), diff0 * rhoair * qs)
                    + DIV(latvap * K_(RHOW), conduct0 * tair)
                      * (DIV(latvap, MA_RV * tair) - 1.0f));
    real sqrtg = SQRTF(gloc);
    real beta = 4.0f * MA_PI * K_(RHOW) * gloc * gammaloc;
    real etafactor2max = DIV(1.e10f, POW(alpha_cesm * wbar, 1.5f));
    real etafactor2[10], smc[10], lnsmloc[10], eta[10];
    for (int m = 0; m < 10; ++m) {
        real volc = DIV(maero[m], kc[MA_DENSITY + m]);
        if (volc > MA_REAL_TINY && naero[m] > MA_REAL_TINY) {
            etafactor2[m] = DIV(1.0f, naero[m] * beta * sqrtg);
            real amcubeloc = DIV(3.0f * volc, 4.0f * MA_PI
                                 * kc[MA_EXP45LOGSIG + m] * naero[m]);
            smc[m] = kc[MA_SMCRIT + m];
            if (kc[MA_HYGRO + m] > 1.e-10f) {
                smc[m] = 2.0f * K_(ATEN) * SQRTF(DIV(K_(ATEN),
                         27.0f * kc[MA_HYGRO + m] * amcubeloc));
            } else {
                smc[m] = 100.0f;
            }
        } else {
            smc[m] = 1.0f;
            etafactor2[m] = etafactor2max;
        }
        lnsmloc[m] = LOGF(smc[m]);
    }
    real wnuc = wbar;
    real alw = alpha_cesm * wnuc;
    real sqrtalw = SQRTF(alw);
    real zeta = DIV(2.0f * sqrtalw * K_(ATEN), 3.0f * sqrtg);
    real etafactor1 = 2.0f * alw * sqrtalw;
    for (int m = 0; m < 10; ++m) eta[m] = etafactor1 * etafactor2[m];
    real smax = ma_maxsat(kc, zeta, eta, smc);
    real lnsmax = LOGF(smax);
    (void)lnsmax;
    nact = 0.0f;
    for (int m = 0; m < 10; ++m) {
        // 2*(...)/(3*sq2*alogsig): the integer literals promote to REAL.
        real x = DIV(2.0f * (lnsmloc[m] - lnsmax),
                     3.0f * K_(SQ2) * kc[MA_ALOGSIG + m]);
        nact = nact + 0.5f * (1.0f - ma_derf1(x)) * naero[m];
    }
    nact = DIV(nact, rhoair);
    return nact;
}

// ---------------------------------------------------------------------
// mdm_prescribed_hetero (:6025-6055) and mdm_prescribed_hf (:6057-6136)
// ---------------------------------------------------------------------
__device__ void ma_hetero(real t, real ww, real ns, real* nis, real* nid)
{
    const real a11 = 0.0263f, a12 = -0.0185f, a21 = 2.758f, a22 = 1.3221f;
    const real b11 = -0.008f, b12 = -0.0468f, b21 = -0.2667f, b22 = -1.4588f;
    real b = (a11 + b11 * LOGF(ns)) * LOGF(ww) + (a12 + b12 * LOGF(ns));
    real c = a21 + b21 * LOGF(ns);
    *nis = EXPF(a22) * POW(ns, b22) * EXPF(b * t) * POW(ww, c);
    *nis = ma_min2(*nis, ns);
    *nid = 0.0f;
}

__device__ real ma_hf(real t, real ww, real rh, real subgrid, real na)
{
    const real a1_fast = 0.0231f, a21_fast = -1.6387f, a22_fast = -6.045f;
    const real b1_fast = -0.008f, b21_fast = -0.042f, b22_fast = -0.112f;
    const real c1_fast = 0.0739f, c2_fast = 1.2372f;
    const real a1_slow = -0.3949f, a2_slow = 1.282f, b1_slow = -0.0156f;
    const real b2_slow = 0.0111f, b3_slow = 0.0217f, c1_slow = 0.120f;
    const real c2_slow = 2.312f;
    real ni = 0.0f;
    real a = 6.0e-4f * LOGF(ww) + 6.6e-3f;
    real b = 6.0e-2f * LOGF(ww) + 1.052f;
    real c = 1.68f * LOGF(ww) + 129.35f;
    real rhw = (a * t * t + b * t + c) * 0.01f;
    if (t <= -37.0f && rh * subgrid >= rhw) {
        real regm = 6.07f * LOGF(ww) - 55.0f;
        if (t >= regm) {
            real a2_fast, b2_fast;
            if (t > -64.0f) {
                a2_fast = a21_fast;
                b2_fast = b21_fast;
            } else {
                a2_fast = a22_fast;
                b2_fast = b22_fast;
            }
            real k1_fast = EXPF(a2_fast + b2_fast * t + c2_fast * LOGF(ww));
            real k2_fast = a1_fast + b1_fast * t + c1_fast * LOGF(ww);
            ni = k1_fast * POW(na, k2_fast);
            ni = ma_min2(ni, na);
        } else {
            real k1_slow = EXPF(a2_slow + (b2_slow + b3_slow * LOGF(ww)) * t
                                + c2_slow * LOGF(ww));
            real k2_slow = a1_slow + b1_slow * t + c1_slow * LOGF(ww);
            ni = k1_slow * POW(na, k2_slow);
            ni = ma_min2(ni, na);
        }
    }
    return ni;
}

// ---------------------------------------------------------------------
// mdm_prescribed_nucleati (:5846-6023): nucleated ice number (# kg-1)
// ---------------------------------------------------------------------
__device__ real ma_nucleati(real wbar, real tair, real relhum, real relhumi,
                            real qc, real rhoair, const real* naero)
{
    real so4_num = naero[0] * 1.0e-6f;
    real soot_num = naero[9] * 1.0e-6f;
    real dst1_num = naero[2] * 1.0e-6f;
    real dst2_num = naero[3] * 1.0e-6f;
    real dst3_num = naero[4] * 1.0e-6f;
    real dst4_num = naero[5] * 1.0e-6f;
    real dst_num = dst1_num + dst2_num + dst3_num + dst4_num;
    real ni = 0.0f;
    real tc = tair - 273.15f;
    real niimm = 0.0f, nidep = 0.0f, nihf = 0.0f, n1;
    if (so4_num >= 1.0e-10f && (soot_num + dst_num) >= 1.0e-10f) {
        const real subgrid = 1.2f;
        if (tc <= -35.0f && (relhumi * subgrid) >= 1.2f) {
            real a = -1.4938f * LOGF(soot_num + dst_num) + 12.884f;
            real b = -10.41f * LOGF(soot_num + dst_num) - 67.69f;
            real regm = a * LOGF(wbar) + b;
            if (tc > regm) {
                if (tc < -40.0f && wbar > 1.0f) {
                    nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                    niimm = 0.0f;
                    nidep = 0.0f;
                    n1 = nihf;
                } else {
                    ma_hetero(tc, wbar, soot_num + dst_num, &niimm, &nidep);
                    nihf = 0.0f;
                    n1 = niimm + nidep;
                }
            } else if (tc < regm - 5.0f) {
                nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                niimm = 0.0f;
                nidep = 0.0f;
                n1 = nihf;
            } else {
                if (tc < -40.0f && wbar > 1.0f) {
                    nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                    niimm = 0.0f;
                    nidep = 0.0f;
                    n1 = nihf;
                } else {
                    nihf = ma_hf(regm - 5.0f, wbar, relhum, subgrid, so4_num);
                    ma_hetero(regm, wbar, soot_num + dst_num, &niimm, &nidep);
                    if (nihf <= (niimm + nidep)) {
                        n1 = nihf;
                    } else {
                        n1 = (niimm + nidep) * POW(DIV(niimm + nidep, nihf),
                                                   DIV(tc - regm, 5.0f));
                    }
                }
            }
            ni = n1;
        }
    }
    real nimey;
    if (tc < 0.0f && tc > -37.0f && qc > 1.e-12f) {
        real deles;
        if (relhumi > 1.5f) {
            deles = 1.5f;
        } else {
            deles = relhumi;
        }
        nimey = 1.e-3f * EXPF(12.96f * (deles - 1.0f) - 0.639f);
    } else {
        nimey = 0.0f;
    }
    real nuci = ni + nimey;
    if (nuci > 9999.0f || nuci < 0.0f) {
        // Preserve WRF's STOP: a negative sentinel reaches the host flag.
        return -1.0f;
    }
    nuci = DIV(nuci * 1.e+6f, rhoair);
    return nuci;
}

// ---------------------------------------------------------------------
// Prescribed aerosol state of one cell (wrapper :1031-1069)
// ---------------------------------------------------------------------
__device__ void ma_aerosol_cell(const real* __restrict__ kc, real fct,
                                const real* __restrict__ aerocu, size_t cell,
                                size_t ncell, real* maero, real* naero)
{
    // aerocu species planes in Registry order: 1 dust1, 2 dust2, 3 dust3,
    // 4 dust4, 5 seasalt, 6 sulfate, 7 phobcar, 8 phibcar, 9 phoocar,
    // 10 phiocar.
#define AERO(s) aerocu[(size_t)((s) - 1) * ncell + cell]
    maero[0] = fct * AERO(6) * 1E-9f;
    naero[0] = 5.64259e13f * POW(maero[0], 0.58f);
    maero[1] = fct * AERO(5) * 1E-9f;
    naero[1] = maero[1] * kc[MA_NUM_TO_MASS + 1];
    maero[2] = fct * 1.44f * AERO(1) * 1E-9f;
    naero[2] = maero[2] * kc[MA_NUM_TO_MASS + 2];
    maero[3] = fct * 1.44f * AERO(2) * 1E-9f;
    naero[3] = maero[3] * kc[MA_NUM_TO_MASS + 3];
    maero[4] = fct * 1.44f * AERO(3) * 1E-9f;
    naero[4] = maero[4] * kc[MA_NUM_TO_MASS + 4];
    maero[5] = fct * 1.44f * AERO(4) * 1E-9f;
    naero[5] = maero[5] * kc[MA_NUM_TO_MASS + 5];
    maero[6] = fct * 1.54f * AERO(9) * 1E-9f;
    naero[6] = maero[6] * kc[MA_NUM_TO_MASS + 6];
    maero[7] = fct * 1.37f * AERO(7) * 1E-9f;
    naero[7] = maero[7] * kc[MA_NUM_TO_MASS + 7];
    maero[8] = fct * 1.25f * AERO(10) * 1E-9f;
    naero[8] = maero[8] * kc[MA_NUM_TO_MASS + 8];
    maero[9] = fct * 1.37f * AERO(8) * 1E-9f;
    naero[9] = maero[9] * kc[MA_NUM_TO_MASS + 9];
#undef AERO
}

// ---------------------------------------------------------------------
// Scratch planes shared by the three kernels (one float per cell each)
// ---------------------------------------------------------------------
#define S_T 0          // T3D
#define S_QC3DTEN 1
#define S_QI3DTEN 2
#define S_QNI3DTEN 3
#define S_QR3DTEN 4
#define S_NI3DTEN 5
#define S_NS3DTEN 6
#define S_NR3DTEN 7
#define S_T3DTEN 8
#define S_QV3DTEN 9
#define S_NC3DTEN 10
#define S_QG3DTEN 11
#define S_NG3DTEN 12
#define S_RHO 13
#define S_MU 14
#define S_XXLV 15
#define S_XXLS 16
#define S_CPM 17
#define S_LAMI 18
#define S_ACTIVE 19    // 1 when the level was not skipped (sets LTRUE)
#define S_COLUMN 20    // per-column LTRUE in the k = 0 plane
#define S_ERROR 21
#define S_NPLANES 22
#define SCR(p) scratch[(size_t)(p) * ncell + cell]

// ---------------------------------------------------------------------
// Process loop, one thread per cell (:1570-4278 plus wrapper :919-1078)
// ---------------------------------------------------------------------
extern "C" __global__ void morrison_aero_process(
    real* __restrict__ th, real* __restrict__ qv, real* __restrict__ qc,
    real* __restrict__ qr, real* __restrict__ qi, real* __restrict__ qs,
    real* __restrict__ qg, real* __restrict__ ni, real* __restrict__ ns,
    real* __restrict__ nr, real* __restrict__ ng, real* __restrict__ nc,
    const real* __restrict__ pii, const real* __restrict__ pres,
    const real* __restrict__ w, const real* __restrict__ kzh_up,
    const real* __restrict__ qrcu, const real* __restrict__ qscu,
    const real* __restrict__ qicu, const real* __restrict__ aerocu,
    real* __restrict__ ccn, real* __restrict__ scratch,
    const real* __restrict__ kc, real dt, real aercu_fct,
    int aercu_opt, int has_cu, int ncell)
{
    const size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= (size_t)ncell) return;
    SCR(S_ERROR) = 0.0f;

    // wrapper :919, :931-933
    real T3D = th[cell] * pii[cell];
    real WVAR = 0.0f;
    if (aercu_opt == 2) {
        WVAR = DIV(kzh_up[cell], 20.0f);
        WVAR = ma_maxc(0.10f, WVAR);
        WVAR = ma_minc(50.0f, WVAR);
    }
    const bool aero = (aercu_opt == 2);
    const int iinum = aero ? 0 : 1;
    // module switches as the wrapper leaves them (:1018-1075)
    const int INUM = aero ? 0 : 1;
    const int INUC = aero ? 2 : 0;
    const real DT = dt;

    real QC3D = qc[cell], QI3D = qi[cell], QNI3D = qs[cell], QR3D = qr[cell];
    real NI3D = ni[cell], NS3D = ns[cell], NR3D = nr[cell];
    real QG3D = qg[cell], NG3D = ng[cell];
    real QV3D = qv[cell];
    const real PRES = pres[cell];
    const real W3D = aero ? w[cell] : 0.0f;
    real QRCU1D = 0.0f, QSCU1D = 0.0f, QICU1D = 0.0f;
    if (has_cu) { QRCU1D = qrcu[cell]; QSCU1D = qscu[cell]; QICU1D = qicu[cell]; }

    real NC3D;
    real maero[10], naero[10];
    if (aero) {
        NC3D = nc[cell];
        ma_aerosol_cell(kc, aercu_fct, aerocu, cell, (size_t)ncell, maero, naero);
        // :1061-1069, CCN at the seven supersaturations (# cm-3)
        for (int s = 0; s < 7; ++s) {
            real sum = 0.0f;
            for (int l = 0; l < 10; ++l)
                sum = sum + naero[l] * kc[MA_CCNFACT + l * 7 + s];
            ccn[(size_t)s * ncell + cell] = sum;
        }
    } else {
        NC3D = 0.0f;
        for (int l = 0; l < 10; ++l) { maero[l] = 0.0f; naero[l] = 0.0f; }
        if (aercu_opt > 0) {
            for (int s = 0; s < 7; ++s) ccn[(size_t)s * ncell + cell] = 0.0f;
        }
    }

    real QC3DTEN = 0.0f, QI3DTEN = 0.0f, QNI3DTEN = 0.0f, QR3DTEN = 0.0f;
    real NI3DTEN = 0.0f, NS3DTEN = 0.0f, NR3DTEN = 0.0f, T3DTEN = 0.0f;
    real QV3DTEN = 0.0f, NC3DTEN = 0.0f, QG3DTEN = 0.0f, NG3DTEN = 0.0f;

    // :1603-1629
    const real XXLV = 3.1484E6f - 2370.0f * T3D;
    const real XXLS = 3.15E6f - 2370.0f * T3D + 0.3337E6f;
    const real CPM = MA_CP * (1.0f + 0.887f * QV3D);
    real EVS = ma_min2(0.99f * PRES, ma_polysvp(T3D, 0));
    real EIS = ma_min2(0.99f * PRES, ma_polysvp(T3D, 1));
    if (EIS > EVS) EIS = EVS;
    const real QVS = DIV(MA_EP_2 * EVS, PRES - EVS);
    const real QVI = DIV(MA_EP_2 * EIS, PRES - EIS);
    const real QVQVS = DIV(QV3D, QVS);
    const real QVQVSI = DIV(QV3D, QVI);
    const real RHO = DIV(PRES, MA_R * T3D);

    // :1636-1647 cumulus number seeding
    if (QRCU1D >= 1.E-10f) {
        real DUM = 1.8e5f * POW(DIV(QRCU1D * DT, MA_PI * K_(RHOW) * ma_powi(RHO, 3)), 0.25f);
        NR3D = NR3D + DUM;
    }
    if (QSCU1D >= 1.E-10f) {
        real DUM = 3.e5f * POW(DIV(QSCU1D * DT, CONS(1) * ma_powi(RHO, 3)),
                               DIV(1.0f, K_(DS) + 1.0f));
        NS3D = NS3D + DUM;
    }
    if (QICU1D >= 1.E-10f) {
        real DUM = DIV(QICU1D * DT, K_(CI) * POW(80.E-6f, K_(DI)));
        NI3D = NI3D + DUM;
    }

    // :1652-1681 subsaturated removal of small amounts
    if (QVQVS < 0.9f) {
        if (QR3D < 1.E-8f) {
            QV3D = QV3D + QR3D;
            T3D = T3D - DIV(QR3D * XXLV, CPM);
            QR3D = 0.0f;
        }
        if (QC3D < 1.E-8f) {
            QV3D = QV3D + QC3D;
            T3D = T3D - DIV(QC3D * XXLV, CPM);
            QC3D = 0.0f;
        }
    }
    if (QVQVSI < 0.9f) {
        if (QI3D < 1.E-8f) {
            QV3D = QV3D + QI3D;
            T3D = T3D - DIV(QI3D * XXLS, CPM);
            QI3D = 0.0f;
        }
        if (QNI3D < 1.E-8f) {
            QV3D = QV3D + QNI3D;
            T3D = T3D - DIV(QNI3D * XXLS, CPM);
            QNI3D = 0.0f;
        }
        if (QG3D < 1.E-8f) {
            QV3D = QV3D + QG3D;
            T3D = T3D - DIV(QG3D * XXLS, CPM);
            QG3D = 0.0f;
        }
    }
    const real XLF = XXLS - XXLV;
    const real QSMALL = K_(QSMALL);

    // :1690-1714 (EFFx are set by the final loop or stay 0 in a skipped column)
    if (QC3D < QSMALL) { QC3D = 0.0f; NC3D = 0.0f; }
    if (QR3D < QSMALL) { QR3D = 0.0f; NR3D = 0.0f; }
    if (QI3D < QSMALL) { QI3D = 0.0f; NI3D = 0.0f; }
    if (QNI3D < QSMALL) { QNI3D = 0.0f; NS3D = 0.0f; }
    if (QG3D < QSMALL) { QG3D = 0.0f; NG3D = 0.0f; }

    // :1728-1747
    const real MU = DIV(1.496E-6f * POW(T3D, 1.5f), T3D + 120.0f);
    real DUM = POW(DIV(K_(RHOSU), RHO), 0.54f);
    const real AIN = POW(DIV(K_(RHOSU), RHO), 0.35f) * K_(AI);
    const real ARN = DUM * K_(AR);
    const real ASN = DUM * K_(AS);
    const real AGN = DUM * K_(AG);
    real LAMI = 0.0f;

    bool active = true;
    // :1753-1757
    if (QC3D < QSMALL && QI3D < QSMALL && QNI3D < QSMALL
        && QR3D < QSMALL && QG3D < QSMALL) {
        if (T3D < 273.15f && QVQVSI < 0.999f) active = false;
        if (T3D >= 273.15f && QVQVS < 0.999f) active = false;
    }

    if (active) {
        // :1762-1783
        const real KAP = 1.414E3f * MU;
        const real DV = DIV(8.794E-5f * POW(T3D, 1.81f), PRES);
        const real SC = DIV(MU, RHO * DV);
        DUM = MA_RV * ma_powi(T3D, 2);
        const real DQSDT = DIV(XXLV * QVS, DUM);
        const real DQSIDT = DIV(XXLS * QVI, DUM);
        const real ABI = 1.0f + DIV(DQSIDT * XXLS, CPM);
        const real AB = 1.0f + DIV(DQSDT * XXLV, CPM);
        (void)ABI;

        real LAMC = 0.0f, LAMS = 0.0f, LAMR = 0.0f, LAMG = 0.0f;
        real CDIST1 = 0.0f, N0I = 0.0f, N0S = 0.0f, N0RR = 0.0f, N0G = 0.0f;
        real PGAM = 0.0f, LAMMIN, LAMMAX;
        real EPSR, EPSS, EPSG, EPSI;
        real DUM1, DUM2, RATIO;

        if (T3D >= 273.15f) {
            // ===================== T >= 273.15 (:1790-2700) =====================
            if (iinum == 1) NC3D = DIV(K_(NDCNST) * 1.E6f, RHO);

            if (QNI3D < 1.E-6f) {
                QR3D = QR3D + QNI3D;
                NR3D = NR3D + NS3D;
                T3D = T3D - DIV(QNI3D * XLF, CPM);
                QNI3D = 0.0f;
                NS3D = 0.0f;
            }
            if (QG3D < 1.E-6f) {
                QR3D = QR3D + QG3D;
                NR3D = NR3D + NG3D;
                T3D = T3D - DIV(QG3D * XLF, CPM);
                QG3D = 0.0f;
                NG3D = 0.0f;
            }

            if (QC3D < QSMALL && QNI3D < 1.E-8f && QR3D < QSMALL && QG3D < 1.E-8f)
                goto label300;

            NS3D = ma_maxc(0.0f, NS3D);
            NC3D = ma_maxc(0.0f, NC3D);
            NR3D = ma_maxc(0.0f, NR3D);
            NG3D = ma_maxc(0.0f, NG3D);

            // rain :1832-1853
            if (QR3D >= QSMALL) {
                LAMR = POW(DIV(MA_PI * K_(RHOW) * NR3D, QR3D), DIV(1.0f, 3.0f));
                N0RR = NR3D * LAMR;
                if (LAMR < K_(LAMMINR)) {
                    LAMR = K_(LAMMINR);
                    N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                } else if (LAMR > K_(LAMMAXR)) {
                    LAMR = K_(LAMMAXR);
                    N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                }
            }
            // cloud droplets :1860-1892
            if (QC3D >= QSMALL) {
                DUM = DIV(PRES, 287.15f * T3D);
                PGAM = 0.0005714f * (DIV(NC3D, 1.E6f) * DUM) + 0.2714f;
                PGAM = DIV(1.0f, ma_powi(PGAM, 2)) - 1.0f;
                PGAM = ma_maxc(2.0f, PGAM);
                PGAM = ma_minc(10.0f, PGAM);
                LAMC = POW(DIV(CONS(26) * NC3D * ma_gamma(PGAM + 4.0f),
                               QC3D * ma_gamma(PGAM + 1.0f)), DIV(1.0f, 3.0f));
                LAMMIN = DIV(PGAM + 1.0f, 60.E-6f);
                LAMMAX = DIV(PGAM + 1.0f, 1.E-6f);
                if (LAMC < LAMMIN) {
                    LAMC = LAMMIN;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                } else if (LAMC > LAMMAX) {
                    LAMC = LAMMAX;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                }
            }
            // snow :1897-1918
            if (QNI3D >= QSMALL) {
                LAMS = POW(DIV(CONS(1) * NS3D, QNI3D), DIV(1.0f, K_(DS)));
                N0S = NS3D * LAMS;
                if (LAMS < K_(LAMMINS)) {
                    LAMS = K_(LAMMINS);
                    N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                } else if (LAMS > K_(LAMMAXS)) {
                    LAMS = K_(LAMMAXS);
                    N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                }
            }
            // graupel :1923-1942
            if (QG3D >= QSMALL) {
                LAMG = POW(DIV(CONS(2) * NG3D, QG3D), DIV(1.0f, K_(DG)));
                N0G = NG3D * LAMG;
                if (LAMG < K_(LAMMING)) {
                    LAMG = K_(LAMMING);
                    N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                } else if (LAMG > K_(LAMMAXG)) {
                    LAMG = K_(LAMMAXG);
                    N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                }
            }

            {
            real PRC = 0.0f, NPRC = 0.0f, NPRC1 = 0.0f, PRA = 0.0f, NPRA = 0.0f;
            real NRAGG = 0.0f, NSMLTS = 0.0f, NSMLTR = 0.0f, EVPMS = 0.0f;
            real PRE = 0.0f, NSUBR = 0.0f, PRACG = 0.0f;
            real NPRACG = 0.0f, PSMLT = 0.0f, PGMLT = 0.0f, EVPMG = 0.0f;
            real PRACS = 0.0f, NGMLTG = 0.0f, NGMLTR = 0.0f;
            real UMS, UMR, UNS, UNR, UMG, UNG;

            // autoconversion :1983-2001
            if (QC3D >= 1.E-6f) {
                PRC = 1350.0f * POW(QC3D, 2.47f)
                      * POW(DIV(NC3D, 1.e6f) * RHO, -1.79f);
                NPRC1 = DIV(PRC, CONS(29));
                NPRC = DIV(PRC, DIV(QC3D, NC3D));
                NPRC = ma_min2(NPRC, DIV(NC3D, DT));
                NPRC1 = ma_min2(NPRC1, NPRC);
            }
            // snow collected by rain above freezing :2007-2047
            if (QR3D >= 1.E-8f && QNI3D >= 1.E-8f) {
                UMS = DIV(ASN * CONS(3), POW(LAMS, K_(BS)));
                UMR = DIV(ARN * CONS(4), POW(LAMR, K_(BR)));
                UNS = DIV(ASN * CONS(5), POW(LAMS, K_(BS)));
                UNR = DIV(ARN * CONS(6), POW(LAMR, K_(BR)));
                DUM = POW(DIV(K_(RHOSU), RHO), 0.54f);
                UMS = ma_minc(1.2f * DUM, UMS);
                UNS = ma_minc(1.2f * DUM, UNS);
                UMR = ma_minc(9.1f * DUM, UMR);
                UNR = ma_minc(9.1f * DUM, UNR);
                PRACS = CONS(41) * (DIV(POW(ma_powi(1.2f * UMR - 0.95f * UMS, 2)
                        + 0.08f * UMS * UMR, 0.5f) * RHO * N0RR * N0S,
                        ma_powi(LAMR, 3))
                        * (DIV(5.0f, ma_powi(LAMR, 3) * LAMS)
                           + DIV(2.0f, ma_powi(LAMR, 2) * ma_powi(LAMS, 2))
                           + DIV(0.5f, LAMR * ma_powi(LAMS, 3))));
            }
            // graupel collected by rain above freezing :2053-2091
            if (QR3D >= 1.E-8f && QG3D >= 1.E-8f) {
                UMG = DIV(AGN * CONS(7), POW(LAMG, K_(BG)));
                UMR = DIV(ARN * CONS(4), POW(LAMR, K_(BR)));
                UNG = DIV(AGN * CONS(8), POW(LAMG, K_(BG)));
                UNR = DIV(ARN * CONS(6), POW(LAMR, K_(BR)));
                DUM = POW(DIV(K_(RHOSU), RHO), 0.54f);
                UMG = ma_minc(20.0f * DUM, UMG);
                UNG = ma_minc(20.0f * DUM, UNG);
                UMR = ma_minc(9.1f * DUM, UMR);
                UNR = ma_minc(9.1f * DUM, UNR);
                PRACG = CONS(41) * (DIV(POW(ma_powi(1.2f * UMR - 0.95f * UMG, 2)
                        + 0.08f * UMG * UMR, 0.5f) * RHO * N0RR * N0G,
                        ma_powi(LAMR, 3))
                        * (DIV(5.0f, ma_powi(LAMR, 3) * LAMG)
                           + DIV(2.0f, ma_powi(LAMR, 2) * ma_powi(LAMG, 2))
                           + DIV(0.5f, LAMR * ma_powi(LAMG, 3))));
                DUM = DIV(PRACG, 5.2E-7f);
                NPRACG = CONS(32) * RHO * POW(1.7f * ma_powi(UNR - UNG, 2)
                         + 0.3f * UNR * UNG, 0.5f) * N0RR * N0G
                         * (DIV(1.0f, ma_powi(LAMR, 3) * LAMG)
                            + DIV(1.0f, ma_powi(LAMR, 2) * ma_powi(LAMG, 2))
                            + DIV(1.0f, LAMR * ma_powi(LAMG, 3)));
                NPRACG = NPRACG - DUM;
            }
            // accretion :2098-2107
            if (QR3D >= 1.E-8f && QC3D >= 1.E-8f) {
                DUM = (QC3D * QR3D);
                PRA = 67.0f * POW(DUM, 1.15f);
                NPRA = DIV(PRA, DIV(QC3D, NC3D));
            }
            // rain self-collection/breakup :2114-2124
            if (QR3D >= 1.E-8f) {
                DUM1 = 300.e-6f;
                if (DIV(1.0f, LAMR) < DUM1) {
                    DUM = 1.0f;
                } else if (DIV(1.0f, LAMR) >= DUM1) {
                    DUM = 2.0f - EXPF(2300.0f * (DIV(1.0f, LAMR) - DUM1));
                }
                NRAGG = -5.78f * DUM * NR3D * QR3D * RHO;
            }
            // rain evaporation :2129-2146
            if (QR3D >= QSMALL) {
                EPSR = 2.0f * MA_PI * N0RR * RHO * DV
                       * (DIV(K_(F1R), LAMR * LAMR)
                          + DIV(K_(F2R) * POW(DIV(ARN * RHO, MU), 0.5f)
                                * POW(SC, DIV(1.0f, 3.0f)) * CONS(9),
                                POW(LAMR, CONS(34))));
            } else {
                EPSR = 0.0f;
            }
            if (QV3D < QVS) {
                PRE = DIV(EPSR * (QV3D - QVS), AB);
                PRE = ma_minc(0.0f, PRE);
            } else {
                PRE = 0.0f;
            }
            // snow melting :2154-2186
            if (QNI3D >= 1.E-8f) {
                DUM = -(DIV(K_(CPW), XLF) * (T3D - 273.15f) * PRACS);
                PSMLT = DIV(2.0f * MA_PI * N0S * KAP * (273.15f - T3D), XLF)
                        * (DIV(K_(F1S), LAMS * LAMS)
                           + DIV(K_(F2S) * POW(DIV(ASN * RHO, MU), 0.5f)
                                 * POW(SC, DIV(1.0f, 3.0f)) * CONS(10),
                                 POW(LAMS, CONS(35)))) + DUM;
                if (QVQVS < 1.0f) {
                    EPSS = 2.0f * MA_PI * N0S * RHO * DV
                           * (DIV(K_(F1S), LAMS * LAMS)
                              + DIV(K_(F2S) * POW(DIV(ASN * RHO, MU), 0.5f)
                                    * POW(SC, DIV(1.0f, 3.0f)) * CONS(10),
                                    POW(LAMS, CONS(35))));
                    EVPMS = DIV((QV3D - QVS) * EPSS, AB);
                    EVPMS = ma_max2(EVPMS, PSMLT);
                    PSMLT = PSMLT - EVPMS;
                }
            }
            // graupel melting :2194-2226
            if (QG3D >= 1.E-8f) {
                DUM = -(DIV(K_(CPW), XLF) * (T3D - 273.15f) * PRACG);
                PGMLT = DIV(2.0f * MA_PI * N0G * KAP * (273.15f - T3D), XLF)
                        * (DIV(K_(F1S), LAMG * LAMG)
                           + DIV(K_(F2S) * POW(DIV(AGN * RHO, MU), 0.5f)
                                 * POW(SC, DIV(1.0f, 3.0f)) * CONS(11),
                                 POW(LAMG, CONS(36)))) + DUM;
                if (QVQVS < 1.0f) {
                    EPSG = 2.0f * MA_PI * N0G * RHO * DV
                           * (DIV(K_(F1S), LAMG * LAMG)
                              + DIV(K_(F2S) * POW(DIV(AGN * RHO, MU), 0.5f)
                                    * POW(SC, DIV(1.0f, 3.0f)) * CONS(11),
                                    POW(LAMG, CONS(36))));
                    EVPMG = DIV((QV3D - QVS) * EPSG, AB);
                    EVPMG = ma_max2(EVPMG, PGMLT);
                    PGMLT = PGMLT - EVPMG;
                }
            }
            PRACG = 0.0f;
            PRACS = 0.0f;

            // conservation :2244-2296
            DUM = (PRC + PRA) * DT;
            if (DUM > QC3D && QC3D >= QSMALL) {
                RATIO = DIV(QC3D, DUM);
                PRC = PRC * RATIO;
                PRA = PRA * RATIO;
            }
            DUM = (-PSMLT - EVPMS + PRACS) * DT;
            if (DUM > QNI3D && QNI3D >= QSMALL) {
                RATIO = DIV(QNI3D, DUM);
                PSMLT = PSMLT * RATIO;
                EVPMS = EVPMS * RATIO;
                PRACS = PRACS * RATIO;
            }
            DUM = (-PGMLT - EVPMG + PRACG) * DT;
            if (DUM > QG3D && QG3D >= QSMALL) {
                RATIO = DIV(QG3D, DUM);
                PGMLT = PGMLT * RATIO;
                EVPMG = EVPMG * RATIO;
                PRACG = PRACG * RATIO;
            }
            DUM = (-PRACS - PRACG - PRE - PRA - PRC + PSMLT + PGMLT) * DT;
            if (DUM > QR3D && QR3D >= QSMALL) {
                RATIO = DIV(DIV(QR3D, DT) + PRACS + PRACG + PRA + PRC - PSMLT - PGMLT,
                            -PRE);
                PRE = PRE * RATIO;
            }

            // tendencies :2300-2314
            QV3DTEN = QV3DTEN + (-PRE - EVPMS - EVPMG);
            T3DTEN = T3DTEN + DIV(PRE * XXLV + (EVPMS + EVPMG) * XXLS
                                  + (PSMLT + PGMLT - PRACS - PRACG) * XLF, CPM);
            QC3DTEN = QC3DTEN + (-PRA - PRC);
            QR3DTEN = QR3DTEN + (PRE + PRA + PRC - PSMLT - PGMLT + PRACS + PRACG);
            QNI3DTEN = QNI3DTEN + (PSMLT + EVPMS - PRACS);
            QG3DTEN = QG3DTEN + (PGMLT + EVPMG - PRACG);
            NC3DTEN = NC3DTEN + (-NPRA - NPRC);
            NR3DTEN = NR3DTEN + (NPRC1 + NRAGG - NPRACG);

            if (PRE < 0.0f) {
                DUM = DIV(PRE * DT, QR3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSUBR = DIV(DUM * NR3D, DT);
            }
            if (EVPMS + PSMLT < 0.0f) {
                DUM = DIV((EVPMS + PSMLT) * DT, QNI3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSMLTS = DIV(DUM * NS3D, DT);
            }
            if (PSMLT < 0.0f) {
                DUM = DIV(PSMLT * DT, QNI3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSMLTR = DIV(DUM * NS3D, DT);
            }
            if (EVPMG + PGMLT < 0.0f) {
                DUM = DIV((EVPMG + PGMLT) * DT, QG3D);
                DUM = ma_maxc(-1.0f, DUM);
                NGMLTG = DIV(DUM * NG3D, DT);
            }
            if (PGMLT < 0.0f) {
                DUM = DIV(PGMLT * DT, QG3D);
                DUM = ma_maxc(-1.0f, DUM);
                NGMLTR = DIV(DUM * NG3D, DT);
            }
            NS3DTEN = NS3DTEN + (NSMLTS);
            NG3DTEN = NG3DTEN + (NGMLTG);
            NR3DTEN = NR3DTEN + (NSUBR - NSMLTR - NGMLTR);
            }

label300:
            {
            // saturation adjustment :2356-2374
            real DUMT = T3D + DT * T3DTEN;
            real DUMQV = QV3D + DT * QV3DTEN;
            real dum = ma_min2(0.99f * PRES, ma_polysvp(DUMT, 0));
            real DUMQSS = DIV(MA_EP_2 * dum, PRES - dum);
            real DUMQC = QC3D + DT * QC3DTEN;
            DUMQC = ma_maxc(0.0f, DUMQC);
            real DUMS = DUMQV - DUMQSS;
            real PCC = DIV(DIV(DUMS, 1.0f + DIV(ma_powi(XXLV, 2) * DUMQSS,
                                               CPM * MA_RV * ma_powi(DUMT, 2))), DT);
            if (PCC * DT + DUMQC < 0.0f) PCC = DIV(-DUMQC, DT);
            QV3DTEN = QV3DTEN - PCC;
            T3DTEN = T3DTEN + DIV(PCC * XXLV, CPM);
            QC3DTEN = QC3DTEN + PCC;

            // droplet activation :2389-2683 (INUM = 0 only: ISUB = 0,
            // IBASE = 2, IACT = 4)
            if (INUM == 0) {
                if (QC3D + QC3DTEN * DT >= QSMALL) {
                    real dumw = W3D + WVAR;
                    dumw = ma_maxc(0.10f, dumw);
                    if (dumw >= 0.001f) {
                        dumw = W3D + WVAR;
                        real nact = ma_activate(kc, dumw, T3D, RHO, naero, maero, XXLV);
                        real d2 = DIV(nact - NC3D, DT);
                        d2 = ma_maxc(0.0f, d2);
                        NC3DTEN = NC3DTEN + d2;
                    }
                }
            }
            }
        } else {
            // ===================== T < 273.15 (:2704-4271) =====================
            if (iinum == 1) NC3D = DIV(K_(NDCNST) * 1.E6f, RHO);

            NI3D = ma_maxc(0.0f, NI3D);
            NS3D = ma_maxc(0.0f, NS3D);
            NC3D = ma_maxc(0.0f, NC3D);
            NR3D = ma_maxc(0.0f, NR3D);
            NG3D = ma_maxc(0.0f, NG3D);

            // cloud ice :2728-2750
            if (QI3D >= QSMALL) {
                LAMI = POW(DIV(CONS(12) * NI3D, QI3D), DIV(1.0f, K_(DI)));
                N0I = NI3D * LAMI;
                if (LAMI < K_(LAMMINI)) {
                    LAMI = K_(LAMMINI);
                    N0I = DIV(ma_powi(LAMI, 4) * QI3D, CONS(12));
                    NI3D = DIV(N0I, LAMI);
                } else if (LAMI > K_(LAMMAXI)) {
                    LAMI = K_(LAMMAXI);
                    N0I = DIV(ma_powi(LAMI, 4) * QI3D, CONS(12));
                    NI3D = DIV(N0I, LAMI);
                }
            }
            // rain :2755-2776
            if (QR3D >= QSMALL) {
                LAMR = POW(DIV(MA_PI * K_(RHOW) * NR3D, QR3D), DIV(1.0f, 3.0f));
                N0RR = NR3D * LAMR;
                if (LAMR < K_(LAMMINR)) {
                    LAMR = K_(LAMMINR);
                    N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                } else if (LAMR > K_(LAMMAXR)) {
                    LAMR = K_(LAMMAXR);
                    N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                }
            }
            // cloud droplets :2783-2818
            if (QC3D >= QSMALL) {
                DUM = DIV(PRES, 287.15f * T3D);
                PGAM = 0.0005714f * (DIV(NC3D, 1.E6f) * DUM) + 0.2714f;
                PGAM = DIV(1.0f, ma_powi(PGAM, 2)) - 1.0f;
                PGAM = ma_maxc(2.0f, PGAM);
                PGAM = ma_minc(10.0f, PGAM);
                LAMC = POW(DIV(CONS(26) * NC3D * ma_gamma(PGAM + 4.0f),
                               QC3D * ma_gamma(PGAM + 1.0f)), DIV(1.0f, 3.0f));
                LAMMIN = DIV(PGAM + 1.0f, 60.E-6f);
                LAMMAX = DIV(PGAM + 1.0f, 1.E-6f);
                if (LAMC < LAMMIN) {
                    LAMC = LAMMIN;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                } else if (LAMC > LAMMAX) {
                    LAMC = LAMMAX;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                }
                CDIST1 = DIV(NC3D, ma_gamma(PGAM + 1.0f));
            }
            // snow :2823-2844
            if (QNI3D >= QSMALL) {
                LAMS = POW(DIV(CONS(1) * NS3D, QNI3D), DIV(1.0f, K_(DS)));
                N0S = NS3D * LAMS;
                if (LAMS < K_(LAMMINS)) {
                    LAMS = K_(LAMMINS);
                    N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                } else if (LAMS > K_(LAMMAXS)) {
                    LAMS = K_(LAMMAXS);
                    N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                }
            }
            // graupel :2849-2870
            if (QG3D >= QSMALL) {
                LAMG = POW(DIV(CONS(2) * NG3D, QG3D), DIV(1.0f, K_(DG)));
                N0G = NG3D * LAMG;
                if (LAMG < K_(LAMMING)) {
                    LAMG = K_(LAMMING);
                    N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                } else if (LAMG > K_(LAMMAXG)) {
                    LAMG = K_(LAMMAXG);
                    N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                }
            }

            real MNUCCC = 0.0f, NNUCCC = 0.0f, PRC = 0.0f, NPRC = 0.0f, NPRC1 = 0.0f;
            real NSAGG = 0.0f, PSACWS = 0.0f, NPSACWS = 0.0f, PSACWI = 0.0f;
            real NPSACWI = 0.0f, PRACS = 0.0f, NPRACS = 0.0f, NMULTS = 0.0f;
            real QMULTS = 0.0f, NMULTR = 0.0f, QMULTR = 0.0f, NMULTG = 0.0f;
            real QMULTG = 0.0f, NMULTRG = 0.0f, QMULTRG = 0.0f, MNUCCR = 0.0f;
            real NNUCCR = 0.0f, PRA = 0.0f, NPRA = 0.0f, NRAGG = 0.0f, PRCI = 0.0f;
            real NPRCI = 0.0f, PRAI = 0.0f, NPRAI = 0.0f, NNUCCD = 0.0f;
            real MNUCCD = 0.0f, PCC = 0.0f, PRE = 0.0f, PRD = 0.0f, PRDS = 0.0f;
            real EPRD = 0.0f, EPRDS = 0.0f, NSUBI = 0.0f, NSUBS = 0.0f;
            real NSUBR = 0.0f, PIACR = 0.0f, NIACR = 0.0f, PRACI = 0.0f;
            real PIACRS = 0.0f, NIACRS = 0.0f, PRACIS = 0.0f, PRACG = 0.0f;
            real PSACR = 0.0f, PSACWG = 0.0f, PGSACW = 0.0f, PGRACS = 0.0f;
            real PRDG = 0.0f, EPRDG = 0.0f, NPRACG = 0.0f, NPSACWG = 0.0f;
            real NSCNG = 0.0f, NGRACS = 0.0f, NSUBG = 0.0f;
            real UMS, UMR, UNS, UNR, UMG, UNG, FMULT, KC2;
            (void)PCC;

            // droplet freezing :2942-2998
            if (QC3D >= QSMALL && T3D < 269.15f) {
                real NACNT = EXPF(-2.80f + 0.262f * (273.15f - T3D)) * 1000.0f;
                DUM = DIV(DIV(7.37f * T3D, 288.0f * 10.0f * PRES), 100.0f);
                real DAP = DIV(CONS(37) * T3D * (1.0f + DIV(DUM, K_(RIN))), MU);
                MNUCCC = CONS(38) * DAP * NACNT * EXPF(LOGF(CDIST1)
                         + LOGF(ma_gamma(PGAM + 5.0f)) - 4.0f * LOGF(LAMC));
                NNUCCC = DIV(2.0f * MA_PI * DAP * NACNT * CDIST1
                             * ma_gamma(PGAM + 2.0f), LAMC);
                MNUCCC = MNUCCC + CONS(39) * EXPF(LOGF(CDIST1)
                         + LOGF(ma_gamma(7.0f + PGAM)) - 6.0f * LOGF(LAMC))
                         * (EXPF(K_(AIMM) * (273.15f - T3D)) - 1.0f);
                NNUCCC = NNUCCC + CONS(40) * EXPF(LOGF(CDIST1)
                         + LOGF(ma_gamma(PGAM + 4.0f)) - 3.0f * LOGF(LAMC))
                         * (EXPF(K_(AIMM) * (273.15f - T3D)) - 1.0f);
                NNUCCC = ma_min2(NNUCCC, DIV(NC3D, DT));
                // D3: an overflowed freezing rate is bounded by the cloud
                // water present (it would otherwise reach NaN through the
                // conservation ratio below).
                if (!isfinite(MNUCCC)) MNUCCC = DIV(QC3D, DT);
            }
            // autoconversion :3010-3028
            if (QC3D >= 1.E-6f) {
                PRC = 1350.0f * POW(QC3D, 2.47f)
                      * POW(DIV(NC3D, 1.e6f) * RHO, -1.79f);
                NPRC1 = DIV(PRC, CONS(29));
                NPRC = DIV(PRC, DIV(QC3D, NC3D));
                NPRC = ma_min2(NPRC, DIV(NC3D, DT));
                NPRC1 = ma_min2(NPRC1, NPRC);
            }
            // snow aggregation :3036-3041
            if (QNI3D >= 1.E-8f) {
                NSAGG = DIV(CONS(15) * ASN * POW(RHO, DIV(2.0f + K_(BS), 3.0f))
                            * POW(QNI3D, DIV(2.0f + K_(BS), 3.0f))
                            * POW(NS3D * RHO, DIV(4.0f - K_(BS), 3.0f)), (RHO));
            }
            // riming of snow :3050-3059
            if (QNI3D >= 1.E-8f && QC3D >= QSMALL) {
                PSACWS = DIV(CONS(13) * ASN * QC3D * RHO * N0S,
                             POW(LAMS, K_(BS) + 3.0f));
                NPSACWS = DIV(CONS(13) * ASN * NC3D * RHO * N0S,
                              POW(LAMS, K_(BS) + 3.0f));
            }
            // riming of graupel :3064-3072
            if (QG3D >= 1.E-8f && QC3D >= QSMALL) {
                PSACWG = DIV(CONS(14) * AGN * QC3D * RHO * N0G,
                             POW(LAMG, K_(BG) + 3.0f));
                NPSACWG = DIV(CONS(14) * AGN * NC3D * RHO * N0G,
                              POW(LAMG, K_(BG) + 3.0f));
            }
            // riming of cloud ice :3081-3095
            if (QI3D >= 1.E-8f && QC3D >= QSMALL) {
                if (DIV(1.0f, LAMI) >= 100.E-6f) {
                    PSACWI = DIV(CONS(16) * AIN * QC3D * RHO * N0I,
                                 POW(LAMI, K_(BI) + 3.0f));
                    NPSACWI = DIV(CONS(16) * AIN * NC3D * RHO * N0I,
                                  POW(LAMI, K_(BI) + 3.0f));
                }
            }
            // rain accreted by snow :3101-3151
            if (QR3D >= 1.E-8f && QNI3D >= 1.E-8f) {
                UMS = DIV(ASN * CONS(3), POW(LAMS, K_(BS)));
                UMR = DIV(ARN * CONS(4), POW(LAMR, K_(BR)));
                UNS = DIV(ASN * CONS(5), POW(LAMS, K_(BS)));
                UNR = DIV(ARN * CONS(6), POW(LAMR, K_(BR)));
                DUM = POW(DIV(K_(RHOSU), RHO), 0.54f);
                UMS = ma_minc(1.2f * DUM, UMS);
                UNS = ma_minc(1.2f * DUM, UNS);
                UMR = ma_minc(9.1f * DUM, UMR);
                UNR = ma_minc(9.1f * DUM, UNR);
                PRACS = CONS(41) * (DIV(POW(ma_powi(1.2f * UMR - 0.95f * UMS, 2)
                        + 0.08f * UMS * UMR, 0.5f) * RHO * N0RR * N0S,
                        ma_powi(LAMR, 3))
                        * (DIV(5.0f, ma_powi(LAMR, 3) * LAMS)
                           + DIV(2.0f, ma_powi(LAMR, 2) * ma_powi(LAMS, 2))
                           + DIV(0.5f, LAMR * ma_powi(LAMS, 3))));
                NPRACS = CONS(32) * RHO * POW(1.7f * ma_powi(UNR - UNS, 2)
                         + 0.3f * UNR * UNS, 0.5f) * N0RR * N0S
                         * (DIV(1.0f, ma_powi(LAMR, 3) * LAMS)
                            + DIV(1.0f, ma_powi(LAMR, 2) * ma_powi(LAMS, 2))
                            + DIV(1.0f, LAMR * ma_powi(LAMS, 3)));
                PRACS = ma_min2(PRACS, DIV(QR3D, DT));
                if (QNI3D >= 0.1E-3f && QR3D >= 0.1E-3f) {
                    PSACR = CONS(31) * (DIV(POW(ma_powi(1.2f * UMR - 0.95f * UMS, 2)
                            + 0.08f * UMS * UMR, 0.5f) * RHO * N0RR * N0S,
                            ma_powi(LAMS, 3))
                            * (DIV(5.0f, ma_powi(LAMS, 3) * LAMR)
                               + DIV(2.0f, ma_powi(LAMS, 2) * ma_powi(LAMR, 2))
                               + DIV(0.5f, LAMS * ma_powi(LAMR, 3))));
                }
            }
            // rain collected by graupel :3157-3191
            if (QR3D >= 1.E-8f && QG3D >= 1.E-8f) {
                UMG = DIV(AGN * CONS(7), POW(LAMG, K_(BG)));
                UMR = DIV(ARN * CONS(4), POW(LAMR, K_(BR)));
                UNG = DIV(AGN * CONS(8), POW(LAMG, K_(BG)));
                UNR = DIV(ARN * CONS(6), POW(LAMR, K_(BR)));
                DUM = POW(DIV(K_(RHOSU), RHO), 0.54f);
                UMG = ma_minc(20.0f * DUM, UMG);
                UNG = ma_minc(20.0f * DUM, UNG);
                UMR = ma_minc(9.1f * DUM, UMR);
                UNR = ma_minc(9.1f * DUM, UNR);
                PRACG = CONS(41) * (DIV(POW(ma_powi(1.2f * UMR - 0.95f * UMG, 2)
                        + 0.08f * UMG * UMR, 0.5f) * RHO * N0RR * N0G,
                        ma_powi(LAMR, 3))
                        * (DIV(5.0f, ma_powi(LAMR, 3) * LAMG)
                           + DIV(2.0f, ma_powi(LAMR, 2) * ma_powi(LAMG, 2))
                           + DIV(0.5f, LAMR * ma_powi(LAMG, 3))));
                NPRACG = CONS(32) * RHO * POW(1.7f * ma_powi(UNR - UNG, 2)
                         + 0.3f * UNR * UNG, 0.5f) * N0RR * N0G
                         * (DIV(1.0f, ma_powi(LAMR, 3) * LAMG)
                            + DIV(1.0f, ma_powi(LAMR, 2) * ma_powi(LAMG, 2))
                            + DIV(1.0f, LAMR * ma_powi(LAMG, 3)));
                PRACG = ma_min2(PRACG, DIV(QR3D, DT));
            }
            // rime splintering, snow :3205-3254
            if (QNI3D >= 0.1E-3f) {
                if (QC3D >= 0.5E-3f || QR3D >= 0.1E-3f) {
                    if (PSACWS > 0.0f || PRACS > 0.0f) {
                        if (T3D < 270.16f && T3D > 265.16f) {
                            if (T3D > 270.16f) {
                                FMULT = 0.0f;
                            } else if (T3D <= 270.16f && T3D > 268.16f) {
                                FMULT = DIV(270.16f - T3D, 2.0f);
                            } else if (T3D >= 265.16f && T3D <= 268.16f) {
                                FMULT = DIV(T3D - 265.16f, 3.0f);
                            } else if (T3D < 265.16f) {
                                FMULT = 0.0f;
                            }
                            if (PSACWS > 0.0f) {
                                NMULTS = 35.E4f * PSACWS * FMULT * 1000.0f;
                                QMULTS = NMULTS * K_(MMULT);
                                QMULTS = ma_min2(QMULTS, PSACWS);
                                PSACWS = PSACWS - QMULTS;
                            }
                            if (PRACS > 0.0f) {
                                NMULTR = 35.E4f * PRACS * FMULT * 1000.0f;
                                QMULTR = NMULTR * K_(MMULT);
                                QMULTR = ma_min2(QMULTR, PRACS);
                                PRACS = PRACS - QMULTR;
                            }
                        }
                    }
                }
            }
            // rime splintering, graupel :3268-3315
            if (QG3D >= 0.1E-3f) {
                if (QC3D >= 0.5E-3f || QR3D >= 0.1E-3f) {
                    if (PSACWG > 0.0f || PRACG > 0.0f) {
                        if (T3D < 270.16f && T3D > 265.16f) {
                            if (T3D > 270.16f) {
                                FMULT = 0.0f;
                            } else if (T3D <= 270.16f && T3D > 268.16f) {
                                FMULT = DIV(270.16f - T3D, 2.0f);
                            } else if (T3D >= 265.16f && T3D <= 268.16f) {
                                FMULT = DIV(T3D - 265.16f, 3.0f);
                            } else if (T3D < 265.16f) {
                                FMULT = 0.0f;
                            }
                            if (PSACWG > 0.0f) {
                                NMULTG = 35.E4f * PSACWG * FMULT * 1000.0f;
                                QMULTG = NMULTG * K_(MMULT);
                                QMULTG = ma_min2(QMULTG, PSACWG);
                                PSACWG = PSACWG - QMULTG;
                            }
                            if (PRACG > 0.0f) {
                                NMULTRG = 35.E4f * PRACG * FMULT * 1000.0f;
                                QMULTRG = NMULTRG * K_(MMULT);
                                QMULTRG = ma_min2(QMULTRG, PRACG);
                                PRACG = PRACG - QMULTRG;
                            }
                        }
                    }
                }
            }
            // rimed cloud water on snow to graupel :3322-3342
            if (PSACWS > 0.0f) {
                if (QNI3D >= 0.1E-3f && QC3D >= 0.5E-3f) {
                    PGSACW = ma_min2(PSACWS, DIV(CONS(17) * DT * N0S * QC3D * QC3D
                                     * ASN * ASN, RHO * POW(LAMS, 2.0f * K_(BS) + 2.0f)));
                    DUM = ma_maxc(0.0f, DIV(K_(RHOSN), K_(RHOG) - K_(RHOSN)) * PGSACW);
                    NSCNG = DIV(DUM, K_(MG0)) * RHO;
                    NSCNG = ma_min2(NSCNG, DIV(NS3D, DT));
                    PSACWS = PSACWS - PGSACW;
                }
            }
            // rimed rain on snow to graupel :3346-3367
            if (PRACS > 0.0f) {
                if (QNI3D >= 0.1E-3f && QR3D >= 0.1E-3f) {
                    DUM = DIV(CONS(18) * ma_powi(DIV(4.0f, LAMS), 3)
                              * ma_powi(DIV(4.0f, LAMS), 3),
                              CONS(18) * ma_powi(DIV(4.0f, LAMS), 3)
                              * ma_powi(DIV(4.0f, LAMS), 3)
                              + CONS(19) * ma_powi(DIV(4.0f, LAMR), 3)
                              * ma_powi(DIV(4.0f, LAMR), 3));
                    DUM = ma_minc(1.0f, DUM);
                    DUM = ma_maxc(0.0f, DUM);
                    PGRACS = (1.0f - DUM) * PRACS;
                    NGRACS = (1.0f - DUM) * NPRACS;
                    NGRACS = ma_min2(NGRACS, DIV(NR3D, DT));
                    NGRACS = ma_min2(NGRACS, DIV(NS3D, DT));
                    PRACS = PRACS - PGRACS;
                    NPRACS = NPRACS - NGRACS;
                    PSACR = PSACR * (1.0f - DUM);
                }
            }
            // rain freezing :3374-3391
            if (T3D < 269.15f && QR3D >= QSMALL) {
                MNUCCR = DIV(DIV(CONS(20) * NR3D * (EXPF(K_(AIMM) * (273.15f - T3D)) - 1.0f),
                                 ma_powi(LAMR, 3)), ma_powi(LAMR, 3));
                NNUCCR = DIV(MA_PI * NR3D * K_(BIMM) * (EXPF(K_(AIMM) * (273.15f - T3D)) - 1.0f),
                             ma_powi(LAMR, 3));
                NNUCCR = ma_min2(NNUCCR, DIV(NR3D, DT));
                // D3, as for MNUCCC above.
                if (!isfinite(MNUCCR)) MNUCCR = DIV(QR3D, DT);
            }
            // accretion :3398-3407
            if (QR3D >= 1.E-8f && QC3D >= 1.E-8f) {
                DUM = (QC3D * QR3D);
                PRA = 67.0f * POW(DUM, 1.15f);
                NPRA = DIV(PRA, DIV(QC3D, NC3D));
            }
            // rain self-collection :3414-3424
            if (QR3D >= 1.E-8f) {
                DUM1 = 300.e-6f;
                if (DIV(1.0f, LAMR) < DUM1) {
                    DUM = 1.0f;
                } else if (DIV(1.0f, LAMR) >= DUM1) {
                    DUM = 2.0f - EXPF(2300.0f * (DIV(1.0f, LAMR) - DUM1));
                }
                NRAGG = -5.78f * DUM * NR3D * QR3D * RHO;
            }
            // ice autoconversion :3432-3442
            if (QI3D >= 1.E-8f && QVQVSI >= 1.0f) {
                NPRCI = DIV(CONS(21) * (QV3D - QVI) * RHO * N0I
                            * EXPF(-(LAMI * K_(DCS))) * DV, ABI);
                PRCI = CONS(22) * NPRCI;
                NPRCI = ma_min2(NPRCI, DIV(NI3D, DT));
            }
            // ice accreted by snow :3449-3456
            if (QNI3D >= 1.E-8f && QI3D >= QSMALL) {
                PRAI = DIV(CONS(23) * ASN * QI3D * RHO * N0S, POW(LAMS, K_(BS) + 3.0f));
                NPRAI = DIV(CONS(23) * ASN * NI3D * RHO * N0S, POW(LAMS, K_(BS) + 3.0f));
                NPRAI = ma_min2(NPRAI, DIV(NI3D, DT));
            }
            // rain-ice collisions :3463-3487
            if (QR3D >= 1.E-8f && QI3D >= 1.E-8f && T3D <= 273.15f) {
                if (QR3D >= 0.1E-3f) {
                    NIACR = DIV(CONS(24) * NI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)) * RHO;
                    PIACR = DIV(DIV(CONS(25) * NI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)),
                                ma_powi(LAMR, 3)) * RHO;
                    PRACI = DIV(CONS(24) * QI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)) * RHO;
                    NIACR = ma_min2(NIACR, DIV(NR3D, DT));
                    NIACR = ma_min2(NIACR, DIV(NI3D, DT));
                } else {
                    NIACRS = DIV(CONS(24) * NI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)) * RHO;
                    PIACRS = DIV(DIV(CONS(25) * NI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)),
                                 ma_powi(LAMR, 3)) * RHO;
                    PRACIS = DIV(CONS(24) * QI3D * N0RR * ARN, POW(LAMR, K_(BR) + 3.0f)) * RHO;
                    NIACRS = ma_min2(NIACRS, DIV(NR3D, DT));
                    NIACRS = ma_min2(NIACRS, DIV(NI3D, DT));
                }
            }
            // ice nucleation :3492-3548
            if (INUC == 0) {
                if ((QVQVS >= 0.999f && T3D <= 265.15f) || QVQVSI >= 1.08f) {
                    KC2 = 0.005f * EXPF(0.304f * (273.15f - T3D)) * 1000.0f;
                    KC2 = ma_minc(500.e3f, KC2);
                    KC2 = ma_maxc(0.0f, DIV(KC2, RHO));
                    if (KC2 > NI3D + NS3D + NG3D) {
                        NNUCCD = DIV(KC2 - NI3D - NS3D - NG3D, DT);
                        MNUCCD = NNUCCD * K_(MI0);
                    }
                }
            } else if (INUC == 2) {
                if ((QVQVS >= 0.999f && T3D <= 265.15f) || QVQVSI >= 1.08f) {
                    real dumw = W3D + WVAR;
                    KC2 = ma_nucleati(dumw, T3D, QVQVS, QVQVSI, QC3D, RHO, naero);
                    if (KC2 < 0.0f) { SCR(S_ERROR) = 1.0f; KC2 = 0.0f; }
                    if (KC2 > NI3D + NS3D + NG3D) {
                        NNUCCD = DIV(KC2 - NI3D - NS3D - NG3D, DT);
                        NNUCCD = ma_maxc(0.0f, NNUCCD);
                        MNUCCD = NNUCCD * K_(MI0);
                    }
                }
            }

            // :3559-3597
            if (QI3D >= QSMALL) {
                EPSI = DIV(2.0f * MA_PI * N0I * RHO * DV, LAMI * LAMI);
            } else {
                EPSI = 0.0f;
            }
            if (QNI3D >= QSMALL) {
                EPSS = 2.0f * MA_PI * N0S * RHO * DV
                       * (DIV(K_(F1S), LAMS * LAMS)
                          + DIV(K_(F2S) * POW(DIV(ASN * RHO, MU), 0.5f)
                                * POW(SC, DIV(1.0f, 3.0f)) * CONS(10),
                                POW(LAMS, CONS(35))));
            } else {
                EPSS = 0.0f;
            }
            if (QG3D >= QSMALL) {
                EPSG = 2.0f * MA_PI * N0G * RHO * DV
                       * (DIV(K_(F1S), LAMG * LAMG)
                          + DIV(K_(F2S) * POW(DIV(AGN * RHO, MU), 0.5f)
                                * POW(SC, DIV(1.0f, 3.0f)) * CONS(11),
                                POW(LAMG, CONS(36))));
            } else {
                EPSG = 0.0f;
            }
            if (QR3D >= QSMALL) {
                EPSR = 2.0f * MA_PI * N0RR * RHO * DV
                       * (DIV(K_(F1R), LAMR * LAMR)
                          + DIV(K_(F2R) * POW(DIV(ARN * RHO, MU), 0.5f)
                                * POW(SC, DIV(1.0f, 3.0f)) * CONS(9),
                                POW(LAMR, CONS(34))));
            } else {
                EPSR = 0.0f;
            }
            // deposition :3603-3618
            if (QI3D >= QSMALL) {
                DUM = (1.0f - EXPF(-(LAMI * K_(DCS))) * (1.0f + LAMI * K_(DCS)));
                PRD = DIV(EPSI * (QV3D - QVI), ABI) * DUM;
            } else {
                DUM = 0.0f;
            }
            if (QNI3D >= QSMALL) {
                PRDS = DIV(EPSS * (QV3D - QVI), ABI)
                       + DIV(EPSI * (QV3D - QVI), ABI) * (1.0f - DUM);
            } else {
                PRD = PRD + DIV(EPSI * (QV3D - QVI), ABI) * (1.0f - DUM);
            }
            PRDG = DIV(EPSG * (QV3D - QVI), ABI);
            // rain evaporation :3622-3627
            if (QV3D < QVS) {
                PRE = DIV(EPSR * (QV3D - QVS), AB);
                PRE = ma_minc(0.0f, PRE);
            } else {
                PRE = 0.0f;
            }
            // :3632-3658
            DUM = DIV(QV3D - QVI, DT);
            const real FUDGEF = 0.9999f;
            real SUM_DEP = PRD + PRDS + MNUCCD + PRDG;
            if ((DUM > 0.0f && SUM_DEP > DUM * FUDGEF)
                || (DUM < 0.0f && SUM_DEP < DUM * FUDGEF)) {
                MNUCCD = DIV(FUDGEF * MNUCCD * DUM, SUM_DEP);
                PRD = DIV(FUDGEF * PRD * DUM, SUM_DEP);
                PRDS = DIV(FUDGEF * PRDS * DUM, SUM_DEP);
                PRDG = DIV(FUDGEF * PRDG * DUM, SUM_DEP);
            }
            if (PRD < 0.0f) { EPRD = PRD; PRD = 0.0f; }
            if (PRDS < 0.0f) { EPRDS = PRDS; PRDS = 0.0f; }
            if (PRDG < 0.0f) { EPRDG = PRDG; PRDG = 0.0f; }

            // ILIQ = 0 and IGRAUP = 0 (:404, :416): the switch blocks of
            // :3678-3713 are dead.

            // conservation of QC :3717-3731
            DUM = (PRC + PRA + MNUCCC + PSACWS + PSACWI + QMULTS + PSACWG + PGSACW + QMULTG) * DT;
            if (DUM > QC3D && QC3D >= QSMALL) {
                RATIO = DIV(QC3D, DUM);
                PRC = PRC * RATIO;
                PRA = PRA * RATIO;
                MNUCCC = MNUCCC * RATIO;
                PSACWS = PSACWS * RATIO;
                PSACWI = PSACWI * RATIO;
                QMULTS = QMULTS * RATIO;
                QMULTG = QMULTG * RATIO;
                PSACWG = PSACWG * RATIO;
                PGSACW = PGSACW * RATIO;
            }
            // conservation of QI :3735-3750
            DUM = (-PRD - MNUCCC + PRCI + PRAI - QMULTS - QMULTG - QMULTR - QMULTRG
                   - MNUCCD + PRACI + PRACIS - EPRD - PSACWI) * DT;
            if (DUM > QI3D && QI3D >= QSMALL) {
                RATIO = DIV(DIV(QI3D, DT) + PRD + MNUCCC + QMULTS + QMULTG + QMULTR + QMULTRG
                            + MNUCCD + PSACWI,
                            PRCI + PRAI + PRACI + PRACIS - EPRD);
                PRCI = PRCI * RATIO;
                PRAI = PRAI * RATIO;
                PRACI = PRACI * RATIO;
                PRACIS = PRACIS * RATIO;
                EPRD = EPRD * RATIO;
            }
            // conservation of QR :3754-3772
            DUM = ((PRACS - PRE) + (QMULTR + QMULTRG - PRC) + (MNUCCR - PRA)
                   + PIACR + PIACRS + PGRACS + PRACG) * DT;
            if (DUM > QR3D && QR3D >= QSMALL) {
                RATIO = DIV(DIV(QR3D, DT) + PRC + PRA,
                            -PRE + QMULTR + QMULTRG + PRACS + MNUCCR + PIACR + PIACRS
                            + PGRACS + PRACG);
                PRE = PRE * RATIO;
                PRACS = PRACS * RATIO;
                QMULTR = QMULTR * RATIO;
                QMULTRG = QMULTRG * RATIO;
                MNUCCR = MNUCCR * RATIO;
                PIACR = PIACR * RATIO;
                PIACRS = PIACRS * RATIO;
                PGRACS = PGRACS * RATIO;
                PRACG = PRACG * RATIO;
            }
            // conservation of QNI (IGRAUP = 0) :3779-3788
            DUM = (-PRDS - PSACWS - PRAI - PRCI - PRACS - EPRDS + PSACR - PIACRS - PRACIS) * DT;
            if (DUM > QNI3D && QNI3D >= QSMALL) {
                RATIO = DIV(DIV(QNI3D, DT) + PRDS + PSACWS + PRAI + PRCI + PRACS + PIACRS + PRACIS,
                            -EPRDS + PSACR);
                EPRDS = EPRDS * RATIO;
                PSACR = PSACR * RATIO;
            }
            // conservation of QG :3808-3817
            DUM = (-PSACWG - PRACG - PGSACW - PGRACS - PRDG - MNUCCR - EPRDG - PIACR
                   - PRACI - PSACR) * DT;
            if (DUM > QG3D && QG3D >= QSMALL) {
                RATIO = DIV(DIV(QG3D, DT) + PSACWG + PRACG + PGSACW + PGRACS + PRDG + MNUCCR
                            + PSACR + PIACR + PRACI, -EPRDG);
                EPRDG = EPRDG * RATIO;
            }

            // tendencies :3821-3872
            QV3DTEN = QV3DTEN + (-PRE - PRD - PRDS - MNUCCD - EPRD - EPRDS - PRDG - EPRDG);
            T3DTEN = T3DTEN + DIV(PRE * XXLV
                     + (PRD + PRDS + MNUCCD + EPRD + EPRDS + PRDG + EPRDG) * XXLS
                     + (PSACWS + PSACWI + MNUCCC + MNUCCR + QMULTS + QMULTG + QMULTR
                        + QMULTRG + PRACS + PSACWG + PRACG + PGSACW + PGRACS + PIACR
                        + PIACRS) * XLF, CPM);
            QC3DTEN = QC3DTEN + (-PRA - PRC - MNUCCC + PCC - PSACWS - PSACWI - QMULTS
                                 - QMULTG - PSACWG - PGSACW);
            QI3DTEN = QI3DTEN + (PRD + EPRD + PSACWI + MNUCCC - PRCI - PRAI + QMULTS
                                 + QMULTG + QMULTR + QMULTRG + MNUCCD - PRACI - PRACIS);
            QR3DTEN = QR3DTEN + (PRE + PRA + PRC - PRACS - MNUCCR - QMULTR - QMULTRG
                                 - PIACR - PIACRS - PRACG - PGRACS);
            QNI3DTEN = QNI3DTEN + (PRAI + PSACWS + PRDS + PRACS + PRCI + EPRDS - PSACR
                                   + PIACRS + PRACIS);
            NS3DTEN = NS3DTEN + (NSAGG + NPRCI - NSCNG - NGRACS + NIACRS);
            QG3DTEN = QG3DTEN + (PRACG + PSACWG + PGSACW + PGRACS + PRDG + EPRDG + MNUCCR
                                 + PIACR + PRACI + PSACR);
            NG3DTEN = NG3DTEN + (NSCNG + NGRACS + NNUCCR + NIACR);
            NC3DTEN = NC3DTEN + (-NNUCCC - NPSACWS - NPRA - NPRC - NPSACWI - NPSACWG);
            NI3DTEN = NI3DTEN + (NNUCCC - NPRCI - NPRAI + NMULTS + NMULTG + NMULTR + NMULTRG
                                 + NNUCCD - NIACR - NIACRS);
            NR3DTEN = NR3DTEN + (NPRC1 - NPRACS - NNUCCR + NRAGG - NIACR - NIACRS - NPRACG
                                 - NGRACS);

            {
            // saturation adjustment :3877-3895
            real DUMT = T3D + DT * T3DTEN;
            real DUMQV = QV3D + DT * QV3DTEN;
            real dum = ma_min2(0.99f * PRES, ma_polysvp(DUMT, 0));
            real DUMQSS = DIV(MA_EP_2 * dum, PRES - dum);
            real DUMQC = QC3D + DT * QC3DTEN;
            DUMQC = ma_maxc(0.0f, DUMQC);
            real DUMS = DUMQV - DUMQSS;
            real PCCc = DIV(DIV(DUMS, 1.0f + DIV(ma_powi(XXLV, 2) * DUMQSS,
                                                CPM * MA_RV * ma_powi(DUMT, 2))), DT);
            if (PCCc * DT + DUMQC < 0.0f) PCCc = DIV(-DUMQC, DT);
            QV3DTEN = QV3DTEN - PCCc;
            T3DTEN = T3DTEN + DIV(PCCc * XXLV, CPM);
            QC3DTEN = QC3DTEN + PCCc;
            }

            // droplet activation :3904-4215 (INUM = 0: IBASE = 2, IACT = 4)
            if (INUM == 0) {
                if (QC3D + QC3DTEN * DT >= QSMALL) {
                    real dumw = W3D + WVAR;
                    dumw = ma_maxc(0.10f, dumw);
                    if (dumw >= 0.001f) {
                        dumw = W3D + WVAR;
                        real nact = ma_activate(kc, dumw, T3D, RHO, naero, maero, XXLV);
                        real d2 = DIV(nact - NC3D, DT);
                        d2 = ma_maxc(0.0f, d2);
                        NC3DTEN = NC3DTEN + d2;
                    }
                }
            }

            // :4229-4260
            if (EPRD < 0.0f) {
                DUM = DIV(EPRD * DT, QI3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSUBI = DIV(DUM * NI3D, DT);
            }
            if (EPRDS < 0.0f) {
                DUM = DIV(EPRDS * DT, QNI3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSUBS = DIV(DUM * NS3D, DT);
            }
            if (PRE < 0.0f) {
                DUM = DIV(PRE * DT, QR3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSUBR = DIV(DUM * NR3D, DT);
            }
            if (EPRDG < 0.0f) {
                DUM = DIV(EPRDG * DT, QG3D);
                DUM = ma_maxc(-1.0f, DUM);
                NSUBG = DIV(DUM * NG3D, DT);
            }
            NI3DTEN = NI3DTEN + NSUBI;
            NS3DTEN = NS3DTEN + NSUBS;
            NG3DTEN = NG3DTEN + NSUBG;
            NR3DTEN = NR3DTEN + NSUBR;
        }
    }

    // write back the level state and the tendencies
    qv[cell] = QV3D; qc[cell] = QC3D; qr[cell] = QR3D; qi[cell] = QI3D;
    qs[cell] = QNI3D; qg[cell] = QG3D; ni[cell] = NI3D; ns[cell] = NS3D;
    nr[cell] = NR3D; ng[cell] = NG3D; nc[cell] = NC3D;
    SCR(S_T) = T3D;
    SCR(S_QC3DTEN) = QC3DTEN; SCR(S_QI3DTEN) = QI3DTEN;
    SCR(S_QNI3DTEN) = QNI3DTEN; SCR(S_QR3DTEN) = QR3DTEN;
    SCR(S_NI3DTEN) = NI3DTEN; SCR(S_NS3DTEN) = NS3DTEN;
    SCR(S_NR3DTEN) = NR3DTEN; SCR(S_T3DTEN) = T3DTEN;
    SCR(S_QV3DTEN) = QV3DTEN; SCR(S_NC3DTEN) = NC3DTEN;
    SCR(S_QG3DTEN) = QG3DTEN; SCR(S_NG3DTEN) = NG3DTEN;
    SCR(S_RHO) = RHO; SCR(S_MU) = MU; SCR(S_XXLV) = XXLV;
    SCR(S_XXLS) = XXLS; SCR(S_CPM) = CPM; SCR(S_LAMI) = LAMI;
    SCR(S_ACTIVE) = active ? 1.0f : 0.0f;
    (void)th;
}

// ---------------------------------------------------------------------
// Sedimentation (:4280-4611), ten category lanes per column.
//
// Lane c of a column owns WRF's c-th fallout pair, in the order of the
// FR..FNG assignments at :4438-4447:
//   0 R  (FR,  DUMR,   QRSTEN)     5 NR (FNR, DUMFNR, NR3DTEN)
//   1 I  (FI,  DUMI,   QISTEN)     6 C  (FC,  DUMC,   QCSTEN)
//   2 NI (FNI, DUMFNI, NI3DTEN)    7 NC (FNC, DUMFNC, NC3DTEN)
//   3 S  (FS,  DUMQS,  QNISTEN)    8 G  (FG,  DUMG,   QGSTEN)
//   4 NS (FNS, DUMFNS, NS3DTEN)    9 NG (FNG, DUMFNG, NG3DTEN)
// Every category's split-step sweep reads only its own speed and content
// profiles, so the lanes are independent except for NSTEP (the column
// maximum of :4486-4488, a maximum of per-lane maxima because
// x -> INT(x*DT/DZ+1.) is monotone) and the surface sums of :4604-4609.
// ---------------------------------------------------------------------
#define MA_LANES 10
#define MA_COLS_PER_WARP 3

__device__ __forceinline__ bool ma_mass_lane(int cat)
{
    return cat == 0 || cat == 1 || cat == 3 || cat == 6 || cat == 8;
}

__device__ __forceinline__ real ma_species_speed(
    const real* __restrict__ kc, int cat, real q, real n, real nc_state,
    real temp, real pres_k, real rho, real mu)
{
    const real QSMALL = K_(QSMALL);
    real dum = POW(DIV(K_(RHOSU), rho), 0.54f);
    real um = 0.0f, un = 0.0f;
    if (cat == 0 || cat == 5) {                         // rain
        if (q >= QSMALL) {
            real ARN = dum * K_(AR);
            real DLAMR = POW(DIV(MA_PI * K_(RHOW) * n, q), DIV(1.0f, 3.0f));
            DLAMR = ma_max2(DLAMR, K_(LAMMINR));
            DLAMR = ma_min2(DLAMR, K_(LAMMAXR));
            un = DIV(ARN * CONS(6), POW(DLAMR, K_(BR)));
            um = DIV(ARN * CONS(4), POW(DLAMR, K_(BR)));
        }
        um = ma_min2(um, 9.1f * dum);
        un = ma_min2(un, 9.1f * dum);
    } else if (cat == 1 || cat == 2) {                  // cloud ice
        if (q >= QSMALL) {
            real AIN = POW(DIV(K_(RHOSU), rho), 0.35f) * K_(AI);
            real DLAMI = POW(DIV(CONS(12) * n, q), DIV(1.0f, K_(DI)));
            DLAMI = ma_max2(DLAMI, K_(LAMMINI));
            DLAMI = ma_min2(DLAMI, K_(LAMMAXI));
            un = DIV(AIN * CONS(27), POW(DLAMI, K_(BI)));
            um = DIV(AIN * CONS(28), POW(DLAMI, K_(BI)));
        }
        um = ma_min2(um, 1.2f * POW(DIV(K_(RHOSU), rho), 0.35f));
        un = ma_min2(un, 1.2f * POW(DIV(K_(RHOSU), rho), 0.35f));
    } else if (cat == 3 || cat == 4) {                  // snow
        if (q >= QSMALL) {
            real ASN = dum * K_(AS);
            real DLAMS = POW(DIV(CONS(1) * n, q), DIV(1.0f, K_(DS)));
            DLAMS = ma_max2(DLAMS, K_(LAMMINS));
            DLAMS = ma_min2(DLAMS, K_(LAMMAXS));
            um = DIV(ASN * CONS(3), POW(DLAMS, K_(BS)));
            un = DIV(ASN * CONS(5), POW(DLAMS, K_(BS)));
        }
        um = ma_min2(um, 1.2f * dum);
        un = ma_min2(un, 1.2f * dum);
    } else if (cat == 6 || cat == 7) {                  // cloud droplets
        if (q >= QSMALL) {
            real d = DIV(pres_k, 287.15f * temp);
            real PGAM = 0.0005714f * (DIV(nc_state, 1.E6f) * d) + 0.2714f;
            PGAM = DIV(1.0f, ma_powi(PGAM, 2)) - 1.0f;
            PGAM = ma_maxc(2.0f, PGAM);
            PGAM = ma_minc(10.0f, PGAM);
            real DLAMC = POW(DIV(CONS(26) * n * ma_gamma(PGAM + 4.0f),
                                 q * ma_gamma(PGAM + 1.0f)), DIV(1.0f, 3.0f));
            real LAMMIN = DIV(PGAM + 1.0f, 60.E-6f);
            real LAMMAX = DIV(PGAM + 1.0f, 1.E-6f);
            DLAMC = ma_max2(DLAMC, LAMMIN);
            DLAMC = ma_min2(DLAMC, LAMMAX);
            real ACN = DIV(MA_G * K_(RHOW), 18.0f * mu);
            un = DIV(ACN * ma_gamma(1.0f + K_(BC) + PGAM),
                     POW(DLAMC, K_(BC)) * ma_gamma(PGAM + 1.0f));
            um = DIV(ACN * ma_gamma(4.0f + K_(BC) + PGAM),
                     POW(DLAMC, K_(BC)) * ma_gamma(PGAM + 4.0f));
        }
        // cloud water speeds carry no limit (:4423-4436)
    } else {                                            // graupel
        if (q >= QSMALL) {
            real AGN = dum * K_(AG);
            real DLAMG = POW(DIV(CONS(2) * n, q), DIV(1.0f, K_(DG)));
            DLAMG = ma_max2(DLAMG, K_(LAMMING));
            DLAMG = ma_min2(DLAMG, K_(LAMMAXG));
            um = DIV(AGN * CONS(7), POW(DLAMG, K_(BG)));
            un = DIV(AGN * CONS(8), POW(DLAMG, K_(BG)));
        }
        um = ma_min2(um, 20.0f * dum);
        un = ma_min2(un, 20.0f * dum);
    }
    return ma_mass_lane(cat) ? um : un;
}

extern "C" __global__ void morrison_aero_sediment(
    const real* __restrict__ qc, const real* __restrict__ qr,
    const real* __restrict__ qi, const real* __restrict__ qs,
    const real* __restrict__ qg, const real* __restrict__ nc,
    const real* __restrict__ nr, const real* __restrict__ ni,
    const real* __restrict__ ns, const real* __restrict__ ng,
    const real* __restrict__ pres, const real* __restrict__ dzq,
    real* __restrict__ scratch, const real* __restrict__ kc,
    real* __restrict__ rainnc, real* __restrict__ rainncv,
    real* __restrict__ snownc, real* __restrict__ snowncv,
    real* __restrict__ graupelnc, real* __restrict__ graupelncv,
    real* __restrict__ sr, real dt, int aercu_opt, int nz, int ncol)
{
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int slot = lane / MA_LANES;
    const int cat = lane - slot * MA_LANES;
    const int col = warp * MA_COLS_PER_WARP + slot;
    const bool lane_ok = (slot < MA_COLS_PER_WARP) && (col < ncol);
    const size_t ncell = (size_t)nz * (size_t)ncol;
    const real DT = dt;
    const int iinum = (aercu_opt == 2) ? 0 : 1;
    const bool mass = ma_mass_lane(cat);

    // LTRUE (:1567, :4274, :4289)
    int ltrue = 0;
    if (lane_ok) {
        for (int k = 0; k < nz; ++k) {
            size_t cell = (size_t)k * ncol + col;
            if (SCR(S_ACTIVE) != 0.0f) { ltrue = 1; break; }
        }
    }
    const bool run = lane_ok && ltrue;

    real F[MA_KMAX], D[MA_KMAX], TEN[MA_KMAX];
    int nstep = 1;
    if (run) {
        // :4302-4502, top down
        for (int k = nz - 1; k >= 0; --k) {
            size_t cell = (size_t)k * ncol + col;
            real rho = SCR(S_RHO);
            real q, n, ten_q, ten_n;
            if (cat == 0 || cat == 5) {
                q = qr[cell]; ten_q = SCR(S_QR3DTEN); n = nr[cell]; ten_n = SCR(S_NR3DTEN);
            } else if (cat == 1 || cat == 2) {
                q = qi[cell]; ten_q = SCR(S_QI3DTEN); n = ni[cell]; ten_n = SCR(S_NI3DTEN);
            } else if (cat == 3 || cat == 4) {
                q = qs[cell]; ten_q = SCR(S_QNI3DTEN); n = ns[cell]; ten_n = SCR(S_NS3DTEN);
            } else if (cat == 6 || cat == 7) {
                q = qc[cell]; ten_q = SCR(S_QC3DTEN); n = nc[cell]; ten_n = SCR(S_NC3DTEN);
            } else {
                q = qg[cell]; ten_q = SCR(S_QG3DTEN); n = ng[cell]; ten_n = SCR(S_NG3DTEN);
            }
            real dq = q + ten_q * DT;
#ifdef MA_POSITIVE_SEDIMENTATION
            // Forecasts transport nonnegative particle mass. A cancelled
            // process tendency can leave a negative FP32 residue, whose
            // inherited fall speed otherwise exports negative precipitation.
            // Same cause and remedy as Morrison mo_sediment_impl; the raw
            // native column replay retains the original signed arithmetic.
            dq = ma_maxc(0.0f, dq);
#endif
            real dn = n + ten_n * DT;
            if ((cat == 6 || cat == 7) && iinum == 1) dn = n;      // :4316-4318
            dn = ma_maxc(0.0f, dn);                                // :4323-4327
            real f = ma_species_speed(kc, cat, dq, dn, nc[cell], SCR(S_T),
                                      pres[cell], rho, SCR(S_MU));
            if (k <= nz - 2) {                                     // :4451-4482
                if (f < 1.E-10f) f = F[k + 1];
            }
            F[k] = f;
            int steps = (int)(DIV(f * DT, dzq[cell]) + 1.0f);      // :4488
            nstep = (steps > nstep) ? steps : nstep;
            D[k] = (mass ? dq : dn) * rho;                         // :4491-4500
            TEN[k] = mass ? 0.0f : ten_n;
        }
    }
    // column NSTEP: the maximum over the column's ten lanes
    {
        int m = run ? nstep : 0;
        int base = slot * MA_LANES;
        int best = 0;
        for (int c = 0; c < MA_LANES; ++c) {
            int v = __shfl_sync(0xffffffffu, m, (base + c) & 31);
            best = (v > best) ? v : best;
        }
        nstep = (best > 0) ? best : 1;
    }
    int nmax = run ? nstep : 0;
    for (int off = 16; off > 0; off >>= 1) {
        int v = __shfl_xor_sync(0xffffffffu, nmax, off);
        nmax = (v > nmax) ? v : nmax;
    }

    real PRECRT = 0.0f, SNOWRT = 0.0f, SNOWPRT = 0.0f, GRPLPRT = 0.0f;
    const real rnstep = (real)nstep;
    for (int step = 1; step <= nmax; ++step) {
        real fal_sfc = 0.0f;
        if (run && step <= nstep) {
            // :4506-4598; FALOUT(K+1) is the value before this substep's
            // update of level K+1, carried down in fal_above.
            real fal_above = 0.0f;
            for (int k = nz - 1; k >= 0; --k) {
                size_t cell = (size_t)k * ncol + col;
                real rho = SCR(S_RHO);
                real fal = F[k] * D[k];
                real faltnd;
                if (k == nz - 1) {
                    faltnd = DIV(fal, dzq[cell]);
                    TEN[k] = TEN[k] - DIV(DIV(faltnd, rnstep), rho);
                    D[k] = D[k] - DIV(faltnd * DT, rnstep);
                } else {
                    faltnd = DIV(fal_above - fal, dzq[cell]);
                    TEN[k] = TEN[k] + DIV(DIV(faltnd, rnstep), rho);
                    D[k] = D[k] + DIV(faltnd * DT, rnstep);
                }
                fal_above = fal;
            }
            fal_sfc = fal_above;
        }
        int base = slot * MA_LANES;
        real fR = __shfl_sync(0xffffffffu, fal_sfc, (base + 0) & 31);
        real fI = __shfl_sync(0xffffffffu, fal_sfc, (base + 1) & 31);
        real fS = __shfl_sync(0xffffffffu, fal_sfc, (base + 3) & 31);
        real fC = __shfl_sync(0xffffffffu, fal_sfc, (base + 6) & 31);
        real fG = __shfl_sync(0xffffffffu, fal_sfc, (base + 8) & 31);
        if (run && step <= nstep && cat == 0) {
            // :4604-4609
            PRECRT = PRECRT + DIV((fR + fC + fS + fI + fG) * DT, rnstep);
            SNOWRT = SNOWRT + DIV((fS + fI + fG) * DT, rnstep);
            SNOWPRT = SNOWPRT + DIV((fI + fS) * DT, rnstep);
            GRPLPRT = GRPLPRT + DIV((fG) * DT, rnstep);
        }
    }

    if (run) {
        // :4617-4621 for the mass lanes; the number tendencies as updated
        for (int k = 0; k < nz; ++k) {
            size_t cell = (size_t)k * ncol + col;
            switch (cat) {
            case 0: SCR(S_QR3DTEN) = SCR(S_QR3DTEN) + TEN[k]; break;
            case 1: SCR(S_QI3DTEN) = SCR(S_QI3DTEN) + TEN[k]; break;
            case 3: SCR(S_QNI3DTEN) = SCR(S_QNI3DTEN) + TEN[k]; break;
            case 6: SCR(S_QC3DTEN) = SCR(S_QC3DTEN) + TEN[k]; break;
            case 8: SCR(S_QG3DTEN) = SCR(S_QG3DTEN) + TEN[k]; break;
            case 2: SCR(S_NI3DTEN) = TEN[k]; break;
            case 4: SCR(S_NS3DTEN) = TEN[k]; break;
            case 5: SCR(S_NR3DTEN) = TEN[k]; break;
            case 7: SCR(S_NC3DTEN) = TEN[k]; break;
            default: SCR(S_NG3DTEN) = TEN[k]; break;
            }
        }
    }
    if (lane_ok && cat == 0) {
        // wrapper :1175-1182
        rainnc[col] = rainnc[col] + PRECRT;
        rainncv[col] = PRECRT;
        snownc[col] = snownc[col] + SNOWPRT;
        snowncv[col] = SNOWPRT;
        graupelnc[col] = graupelnc[col] + GRPLPRT;
        graupelncv[col] = GRPLPRT;
        sr[col] = DIV(SNOWRT, PRECRT + 1.E-12f);
        scratch[(size_t)S_COLUMN * ncell + col] = ltrue ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------------
// Final loop (:4613-5000) and the wrapper's write-back (:1104-1137)
// ---------------------------------------------------------------------
extern "C" __global__ void morrison_aero_finalize(
    real* __restrict__ th, real* __restrict__ qv, real* __restrict__ qc,
    real* __restrict__ qr, real* __restrict__ qi, real* __restrict__ qs,
    real* __restrict__ qg, real* __restrict__ ni, real* __restrict__ ns,
    real* __restrict__ nr, real* __restrict__ ng, real* __restrict__ nc,
    const real* __restrict__ pii, const real* __restrict__ pres,
    const real* __restrict__ w, const real* __restrict__ kzh_up,
    real* __restrict__ effc, real* __restrict__ effi,
    real* __restrict__ effs, real* __restrict__ effr,
    real* __restrict__ efcg, real* __restrict__ efig,
    real* __restrict__ efsg, real* __restrict__ wact,
    real* __restrict__ scratch, const real* __restrict__ kc,
    real dt, int aercu_opt, int ncell, int ncol)
{
    const size_t cell = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= (size_t)ncell) return;
    const size_t col = cell % (size_t)ncol;
    const bool ltrue = scratch[(size_t)S_COLUMN * ncell + col] != 0.0f;
    const real DT = dt;
    const real QSMALL = K_(QSMALL);
    const int iinum = (aercu_opt == 2) ? 0 : 1;

    real T3D = SCR(S_T);
    real QV3D = qv[cell], QC3D = qc[cell], QR3D = qr[cell], QI3D = qi[cell];
    real QNI3D = qs[cell], QG3D = qg[cell];
    real NI3D = ni[cell], NS3D = ns[cell], NR3D = nr[cell], NG3D = ng[cell];
    real NC3D = nc[cell];
    real EFFC = 0.0f, EFFI = 0.0f, EFFS = 0.0f, EFFR = 0.0f;

    if (ltrue) {
        const real PRES = pres[cell];
        const real RHO = SCR(S_RHO);
        const real XXLV = SCR(S_XXLV), XXLS = SCR(S_XXLS), CPM = SCR(S_CPM);
        const real XLF = XXLS - XXLV;
        const real LAMI0 = SCR(S_LAMI);
        real QC3DTEN = SCR(S_QC3DTEN), QI3DTEN = SCR(S_QI3DTEN);
        real QNI3DTEN = SCR(S_QNI3DTEN), QR3DTEN = SCR(S_QR3DTEN);
        real NI3DTEN = SCR(S_NI3DTEN), NS3DTEN = SCR(S_NS3DTEN);
        real NR3DTEN = SCR(S_NR3DTEN), NC3DTEN = SCR(S_NC3DTEN);
        real QG3DTEN = SCR(S_QG3DTEN), NG3DTEN = SCR(S_NG3DTEN);
        real T3DTEN = SCR(S_T3DTEN), QV3DTEN = SCR(S_QV3DTEN);

        // :4627-4634 cloud ice to snow above a mean size of 2*DCS
        if (QI3D >= QSMALL && T3D < 273.15f && LAMI0 >= 1.E-10f) {
            if (DIV(1.0f, LAMI0) >= 2.0f * K_(DCS)) {
                QNI3DTEN = QNI3DTEN + DIV(QI3D, DT) + QI3DTEN;
                NS3DTEN = NS3DTEN + DIV(NI3D, DT) + NI3DTEN;
                QI3DTEN = DIV(-QI3D, DT);
                NI3DTEN = DIV(-NI3D, DT);
            }
        }
        // :4639-4655
        QC3D = QC3D + QC3DTEN * DT;
        QI3D = QI3D + QI3DTEN * DT;
        QNI3D = QNI3D + QNI3DTEN * DT;
        QR3D = QR3D + QR3DTEN * DT;
        NC3D = NC3D + NC3DTEN * DT;
        NI3D = NI3D + NI3DTEN * DT;
        NS3D = NS3D + NS3DTEN * DT;
        NR3D = NR3D + NR3DTEN * DT;
        QG3D = QG3D + QG3DTEN * DT;
        NG3D = NG3D + NG3DTEN * DT;
        T3D = T3D + T3DTEN * DT;
        QV3D = QV3D + QV3DTEN * DT;

        // :4660-4671
        real EVS = ma_min2(0.99f * PRES, ma_polysvp(T3D, 0));
        real EIS = ma_min2(0.99f * PRES, ma_polysvp(T3D, 1));
        if (EIS > EVS) EIS = EVS;
        real QVS = DIV(MA_EP_2 * EVS, PRES - EVS);
        real QVI = DIV(MA_EP_2 * EIS, PRES - EIS);
        real QVQVS = DIV(QV3D, QVS);
        real QVQVSI = DIV(QV3D, QVI);

        // :4676-4705
        if (QVQVS < 0.9f) {
            if (QR3D < 1.E-8f) {
                QV3D = QV3D + QR3D;
                T3D = T3D - DIV(QR3D * XXLV, CPM);
                QR3D = 0.0f;
            }
            if (QC3D < 1.E-8f) {
                QV3D = QV3D + QC3D;
                T3D = T3D - DIV(QC3D * XXLV, CPM);
                QC3D = 0.0f;
            }
        }
        if (QVQVSI < 0.9f) {
            if (QI3D < 1.E-8f) {
                QV3D = QV3D + QI3D;
                T3D = T3D - DIV(QI3D * XXLS, CPM);
                QI3D = 0.0f;
            }
            if (QNI3D < 1.E-8f) {
                QV3D = QV3D + QNI3D;
                T3D = T3D - DIV(QNI3D * XXLS, CPM);
                QNI3D = 0.0f;
            }
            if (QG3D < 1.E-8f) {
                QV3D = QV3D + QG3D;
                T3D = T3D - DIV(QG3D * XXLS, CPM);
                QG3D = 0.0f;
            }
        }
        // :4710-4734
        if (QC3D < QSMALL) { QC3D = 0.0f; NC3D = 0.0f; }
        if (QR3D < QSMALL) { QR3D = 0.0f; NR3D = 0.0f; }
        if (QI3D < QSMALL) { QI3D = 0.0f; NI3D = 0.0f; }
        if (QNI3D < QSMALL) { QNI3D = 0.0f; NS3D = 0.0f; }
        if (QG3D < QSMALL) { QG3D = 0.0f; NG3D = 0.0f; }

        real LAMI = 0.0f, LAMS = 0.0f, LAMR = 0.0f, LAMC = 0.0f, PGAM = 0.0f;
        // :4739-4740 GOTO 500 when every species is below QSMALL
        if (!(QC3D < QSMALL && QI3D < QSMALL && QNI3D < QSMALL
              && QR3D < QSMALL && QG3D < QSMALL)) {
            // :4747-4756 cloud ice melts above freezing
            if (QI3D >= QSMALL && T3D >= 273.15f) {
                QR3D = QR3D + QI3D;
                T3D = T3D - DIV(QI3D * XLF, CPM);
                QI3D = 0.0f;
                NR3D = NR3D + NI3D;
                NI3D = 0.0f;
            }
            // :4763-4781 homogeneous freezing (ILIQ = 0, IGRAUP = 0)
            if (T3D <= 233.15f && QC3D >= QSMALL) {
                QI3D = QI3D + QC3D;
                T3D = T3D + DIV(QC3D * XLF, CPM);
                QC3D = 0.0f;
                NI3D = NI3D + NC3D;
                NC3D = 0.0f;
            }
            if (T3D <= 233.15f && QR3D >= QSMALL) {
                QG3D = QG3D + QR3D;
                T3D = T3D + DIV(QR3D * XLF, CPM);
                QR3D = 0.0f;
                NG3D = NG3D + NR3D;
                NR3D = 0.0f;
            }
            // :4799-4803
            NI3D = ma_maxc(0.0f, NI3D);
            NS3D = ma_maxc(0.0f, NS3D);
            NC3D = ma_maxc(0.0f, NC3D);
            NR3D = ma_maxc(0.0f, NR3D);
            NG3D = ma_maxc(0.0f, NG3D);
            // :4808-4829
            if (QI3D >= QSMALL) {
                LAMI = POW(DIV(CONS(12) * NI3D, QI3D), DIV(1.0f, K_(DI)));
                if (LAMI < K_(LAMMINI)) {
                    LAMI = K_(LAMMINI);
                    real N0I = DIV(ma_powi(LAMI, 4) * QI3D, CONS(12));
                    NI3D = DIV(N0I, LAMI);
                } else if (LAMI > K_(LAMMAXI)) {
                    LAMI = K_(LAMMAXI);
                    real N0I = DIV(ma_powi(LAMI, 4) * QI3D, CONS(12));
                    NI3D = DIV(N0I, LAMI);
                }
            }
            // :4834-4855
            if (QR3D >= QSMALL) {
                LAMR = POW(DIV(MA_PI * K_(RHOW) * NR3D, QR3D), DIV(1.0f, 3.0f));
                if (LAMR < K_(LAMMINR)) {
                    LAMR = K_(LAMMINR);
                    real N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                } else if (LAMR > K_(LAMMAXR)) {
                    LAMR = K_(LAMMAXR);
                    real N0RR = DIV(ma_powi(LAMR, 4) * QR3D, MA_PI * K_(RHOW));
                    NR3D = DIV(N0RR, LAMR);
                }
            }
            // :4862-4893
            if (QC3D >= QSMALL) {
                real DUM = DIV(PRES, 287.15f * T3D);
                PGAM = 0.0005714f * (DIV(NC3D, 1.E6f) * DUM) + 0.2714f;
                PGAM = DIV(1.0f, ma_powi(PGAM, 2)) - 1.0f;
                PGAM = ma_maxc(2.0f, PGAM);
                PGAM = ma_minc(10.0f, PGAM);
                LAMC = POW(DIV(CONS(26) * NC3D * ma_gamma(PGAM + 4.0f),
                               QC3D * ma_gamma(PGAM + 1.0f)), DIV(1.0f, 3.0f));
                real LAMMIN = DIV(PGAM + 1.0f, 60.E-6f);
                real LAMMAX = DIV(PGAM + 1.0f, 1.E-6f);
                if (LAMC < LAMMIN) {
                    LAMC = LAMMIN;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                } else if (LAMC > LAMMAX) {
                    LAMC = LAMMAX;
                    NC3D = DIV(EXPF(3.0f * LOGF(LAMC) + LOGF(QC3D)
                                    + LOGF(ma_gamma(PGAM + 1.0f))
                                    - LOGF(ma_gamma(PGAM + 4.0f))), CONS(26));
                }
            }
            // :4898-4918
            if (QNI3D >= QSMALL) {
                LAMS = POW(DIV(CONS(1) * NS3D, QNI3D), DIV(1.0f, K_(DS)));
                if (LAMS < K_(LAMMINS)) {
                    LAMS = K_(LAMMINS);
                    real N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                } else if (LAMS > K_(LAMMAXS)) {
                    LAMS = K_(LAMMAXS);
                    real N0S = DIV(ma_powi(LAMS, 4) * QNI3D, CONS(1));
                    NS3D = DIV(N0S, LAMS);
                }
            }
            // :4923-4944
            if (QG3D >= QSMALL) {
                real LAMG = POW(DIV(CONS(2) * NG3D, QG3D), DIV(1.0f, K_(DG)));
                if (LAMG < K_(LAMMING)) {
                    LAMG = K_(LAMMING);
                    real N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                } else if (LAMG > K_(LAMMAXG)) {
                    LAMG = K_(LAMMAXG);
                    real N0G = DIV(ma_powi(LAMG, 4) * QG3D, CONS(2));
                    NG3D = DIV(N0G, LAMG);
                }
            }
        }
        // :4950-4975 effective radii (graupel's EFFG is never read)
        if (QI3D >= QSMALL) EFFI = DIV(DIV(3.0f, LAMI), 2.0f) * 1.E6f;
        else EFFI = 4.99f;
        if (QNI3D >= QSMALL) EFFS = DIV(DIV(3.0f, LAMS), 2.0f) * 1.E6f;
        else EFFS = 25.0f;
        if (QR3D >= QSMALL) EFFR = DIV(DIV(3.0f, LAMR), 2.0f) * 1.E6f;
        else EFFR = 25.0f;
        if (QC3D >= QSMALL) {
            EFFC = DIV(DIV(DIV(ma_gamma(PGAM + 4.0f), ma_gamma(PGAM + 3.0f)), LAMC),
                       2.0f) * 1.E6f;
        } else {
            EFFC = 2.49f;
        }
        // :4990-5000
        NI3D = ma_min2(NI3D, DIV(0.3E6f, RHO));
        if (iinum == 1) NC3D = DIV(K_(NDCNST) * 1.E6f, RHO);
    }

    // wrapper write-back :1110-1146
    qc[cell] = QC3D; qi[cell] = QI3D; qs[cell] = QNI3D; qr[cell] = QR3D;
    ni[cell] = NI3D; ns[cell] = NS3D; nr[cell] = NR3D; qg[cell] = QG3D;
    ng[cell] = NG3D; qv[cell] = QV3D; nc[cell] = NC3D;
    SCR(S_T) = T3D;
    th[cell] = DIV(T3D, pii[cell]);
    effc[cell] = EFFC; effi[cell] = EFFI; effs[cell] = EFFS; effr[cell] = EFFR;
    if (aercu_opt > 0) {
        efcg[cell] = ma_maxc(2.49f, ma_minc(50.0f, EFFC));
        efig[cell] = ma_maxc(4.99f, ma_minc(120.0f, EFFI));
        efsg[cell] = ma_maxc(9.99f, ma_minc(999.0f, EFFS));
        real WVAR = DIV(kzh_up[cell], 20.0f);
        WVAR = ma_maxc(0.10f, WVAR);
        WVAR = ma_minc(50.0f, WVAR);
        wact[cell] = WVAR + w[cell];
    }
}
