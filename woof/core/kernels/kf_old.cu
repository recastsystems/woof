/* Source: WRF v4.6.1 and v4.7.1 phys/module_cu_kf.F (byte-identical).
 * WRF is public domain under the UCAR notice retained in
 * tools/kf_old_wrf461_oracle/PIN.json and WRF-LICENSE.txt.
 */
extern "C" __global__ void kf_old_fatal_status(
    const int *fatal, unsigned int *status, int count) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < count && fatal[col] > 0)
        atomicOr(status, 1u << (fatal[col] - 1));
}

/* The previous Kain-Fritsch scheme (WRF cu_physics = 99, KFSCHEME), one
 * CUDA thread per mass-grid column.
 *
 * Numerical authority: the WRF v4.6.1 column oracle,
 * tools/kf_old_wrf461_oracle (phys/module_cu_kf.F, byte-unmodified, sha256
 * pinned there), graded word for word by tests/test_kf_old_wrf461_parity.py.
 *
 * TRANSCRIPTION ANCHORS (phys/module_cu_kf.F)
 *   13-331     KFCPS       the per-column driver: W0AVG running mean, the
 *                          NCA hold test, the species fold of the outputs
 *   334-2082   KFPARA      the column: source-layer search, trigger,
 *                          updraft, downdraft, the CAPE-removal closure,
 *                          compensating subsidence, feedback
 *   2085-2148  CONDLOAD    precipitation fallout
 *   2151-2261  DTFRZNEW    linear glaciation
 *   2275-2305  PROF5       Gaussian mixing profile
 *   2308-2462  TPMIX       wet-bulb parcel temperature by secant iteration
 *   2464-2508  ENVIRTHT    environmental equivalent potential temperature
 *   2518-2580  TPDD        downdraft parcel temperature
 *
 * NOTHING HERE IS SHARED WITH THE cu_physics = 1 KERNEL (kf.cu), AND THAT
 * WAS MEASURED, NOT ASSUMED.  The old scheme solves its parcel
 * thermodynamics in TPMIX/DTFRZNEW/ENVIRTHT/TPDD by iteration; KF-eta
 * replaced all four with the KF_LUTAB lookup (TPMIX2, TPMIX2DD, its own
 * ENVIRTHT and DTFRZNEW).  CONDLOAD differs (1.E-10 vs 1.E-8 guards, the
 * WAVG spelling, KF-eta's |WTW| >= 1.E-4 floor).  PROF5 is the one routine
 * that is textually identical between the two Fortran files -- but kf.cu's
 * kf_prof5 is not a WRF-order transcription of it (it evaluates
 * expf(-0.5f*y*y) with CUDA's expf where the Fortran is EXP(Y*Y/(-2)) on
 * the reference libm), so reusing it would trade a 0 ULP proof for a
 * shared function.  The port and its record of the diff live in
 * tools/kf_old_wrf461_oracle/README.md.
 *
 * ARITHMETIC.  Every basic operation is spelled with its rounded intrinsic
 * (FADD/FSUB/FMUL/FDIV from glibc_flt32.cuh, __fsqrt_rn). Fortran
 * left-to-right association is kept
 * literally: A*B*C/D is ((A*B)*C)/D. Strict numerics uses WOOF's float32
 * powf (gfk_pow), EXP and ALOG (gfk_exp, gfk_log); the
 * oracle object calls the scalar libm (its `nm -u` receipt lists expf,
 * logf, powf, lroundf and nothing vector). NINT is lroundf. The
 * fast default uses CUDA's ordinary expf/logf/powf for these same operations.
 * It keeps the full native source-layer search and parcel iterations;
 * strict numerics is the word-for-word proof opt-in. The one
 * intrinsic gfortran folds at compile time, EXP(-4.5) in PROF5, is the
 * same word on both sides (0x3C360282, receipt in the oracle README).
 *
 * MIN AND MAX.  gfortran lowers AMIN1/AMAX1 to minss/maxss with the FIRST
 * written operand as the destination, so AMAX1(A,B) is (A > B) ? A : B and
 * a signed zero or a NaN in the comparison resolves to B.  When one operand
 * is a literal it is the constant that lands in the destination, whichever
 * side it was written on (probe in the oracle README); ko_max/ko_min take
 * their arguments in that order.
 *
 * LAYOUT.  The 78 column arrays live in a caller-provided global workspace,
 * lane-interleaved per warp exactly like kf.cu's (see its header for why a
 * per-thread local frame is priced at the card's whole resident capacity).
 * Each slot spans nz + 1 levels because OMG does.  Inputs are read and
 * outputs written in place at a[k * ny * nx + column].
 *
 * WRF FATALS.  KFPARA calls wrf_error_fatal for a negative surface vapour
 * (:1586), a mass imbalance (:1617/:1930) and a column water error over
 * 0.05 % (:1951).  WRF stops the model there.  The kernel records the
 * reason in `fatal` (1, 2, 3) and leaves that column's tendencies at zero;
 * gpuwm/core/kf.py raises with WRF's own message when any column carries
 * one, so the run stops where WRF's would.
 */

/* This unit spells WRF's own names.  The shared preamble defines six of
 * them as gpuwm constants (G, CP, EP2, RD, P0 = 1000 hPa, T0 = 300 K);
 * here P0 and T0 are KFPARA's pressure and temperature COLUMNS and RD is
 * TPDD's downdraft mixing ratio, and the scheme takes G, CP and EP2 from
 * module_model_constants through KoConst below.  The preamble's spellings
 * are withdrawn for this translation unit only. */
#undef G
#undef CP
#undef EP2
#undef RD
#undef P0
#undef T0

#define KO_SLOTS 78
#define KO_LANES 32

struct KoCol {
    float* p;
    __device__ __forceinline__ float& operator[](int k) const {
        return p[(size_t)k * (size_t)KO_LANES];
    }
};

#ifdef GPUWM_KF_OLD_STRICT
#define KO_EXP(x) gfk_exp(x)
#define KO_LOG(x) gfk_log(x)
#define KO_POW(x, y) gfk_pow((x), (y))
#else
#define KO_EXP(x) expf(x)
#define KO_LOG(x) logf(x)
#define KO_POW(x, y) powf((x), (y))
#endif
#define KO_SQRT(x) __fsqrt_rn(x)

/* AMAX1 / AMIN1 as gfortran -O0 lowers them (see header). */
__device__ __forceinline__ float ko_max(float a, float b) {
    return a > b ? a : b;
}
__device__ __forceinline__ float ko_min(float a, float b) {
    return a < b ? a : b;
}
/* NINT on a default REAL: lroundf, half away from zero. */
__device__ __forceinline__ int ko_nint(float x) {
    return (int)lroundf(x);
}

/* WRF constants the scheme receives from module_model_constants
 * (share/module_model_constants.F:17-81) and its own DATA statements.
 * The parameter expressions are evaluated in float by gfortran and here
 * at run time with the same single rounding. */
#define KO_G 9.81f
#define KO_R_D 287.f
#define KO_R_V 461.6f
#define KO_XLV0 3.15E6f
#define KO_XLV1 2370.f
#define KO_XLS0 2.905E6f
#define KO_XLS1 259.532f
#define KO_SVP1 0.6112f
#define KO_SVP2 17.67f
#define KO_SVP3 29.65f
#define KO_SVPT0 273.15f
#define KO_RAD 1500.f

struct KoConst {
    float cp, ep2, aliq, bliq, cliq, dliq, aice, bice, cice, dice;
    float xlv0, xlv1, xls0, xls1, r, g;
};

/* ------------------------------------------------------------------------
 * CONDLOAD (module_cu_kf.F:2085-2148)
 * ---------------------------------------------------------------------- */
__device__ void ko_condload(float* QLIQ, float* QICE, float* WTW, float DZ,
                            float BOTERM, float ENTERM, float RATE,
                            float* QNEWLQ, float* QNEWIC, float* QLQOUT,
                            float* QICOUT, float G) {
    float QTOT = FADD(*QLIQ, *QICE);
    float QNEW = FADD(*QNEWLQ, *QNEWIC);
    float QEST = FMUL(0.5f, FADD(QTOT, QNEW));
    float G1 = FSUB(FSUB(FADD(*WTW, BOTERM), ENTERM),
                    FDIV(FMUL(FMUL(FMUL(2.f, G), DZ), QEST), 1.5f));
    if (G1 < 0.0f) G1 = 0.f;
    float WAVG = FDIV(FADD(KO_SQRT(*WTW), KO_SQRT(G1)), 2.f);
    float CONV = FDIV(FMUL(RATE, DZ), WAVG);
    float RATIO3 = FDIV(*QNEWLQ, FADD(QNEW, 1.E-10f));
    QTOT = FADD(QTOT, FMUL(0.6f, QNEW));
    float OLDQ = QTOT;
    float RATIO4 = FDIV(FADD(FMUL(0.6f, *QNEWLQ), *QLIQ),
                        FADD(QTOT, 1.E-10f));
    QTOT = FMUL(QTOT, KO_EXP(-CONV));
    float DQ = FSUB(OLDQ, QTOT);
    *QLQOUT = FMUL(RATIO4, DQ);
    *QICOUT = FMUL(FSUB(1.f, RATIO4), DQ);
    float PPTDRG = FMUL(0.5f, FSUB(FADD(OLDQ, QTOT), FMUL(0.2f, QNEW)));
    *WTW = FSUB(FSUB(FADD(*WTW, BOTERM), ENTERM),
                FDIV(FMUL(FMUL(FMUL(2.f, G), DZ), PPTDRG), 1.5f));
    *QLIQ = FADD(FMUL(RATIO4, QTOT), FMUL(FMUL(RATIO3, 0.4f), QNEW));
    *QICE = FADD(FMUL(FSUB(1.f, RATIO4), QTOT),
                 FMUL(FMUL(FSUB(1.f, RATIO3), 0.4f), QNEW));
    *QNEWLQ = 0.f;
    *QNEWIC = 0.f;
}

/* ------------------------------------------------------------------------
 * DTFRZNEW (module_cu_kf.F:2151-2261)
 * ---------------------------------------------------------------------- */
__device__ void ko_dtfrznew(float* TU, float P, float* THTEU, float* QVAP,
                            float* QLIQ, float* QICE, float* RATIO2,
                            float TTFRZ, float TBFRZ, float* QNWFRZ,
                            float* RL, float* FRC1, float EFFQ, int* IFLAG,
                            const KoConst& k) {
    const float C5 = 1.0723E-3f;
    float QLQFRZ = FMUL(*QLIQ, EFFQ);
    float QNEW = FMUL(FMUL(*QNWFRZ, EFFQ), 0.5f);
    float ESLIQ = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, *TU), k.cliq),
                                           FSUB(*TU, k.dliq))));
    float ESICE = FMUL(k.aice, KO_EXP(FDIV(FSUB(FMUL(k.bice, *TU), k.cice),
                                           FSUB(*TU, k.dice))));
    float RLC = FSUB(2.5E6f, FMUL(2369.276f, FSUB(*TU, 273.16f)));
    float RLS = FSUB(2833922.f, FMUL(259.532f, FSUB(*TU, 273.16f)));
    float RLF = FSUB(RLS, RLC);
    float CCP = FMUL(1005.7f, FADD(1.f, FMUL(0.89f, *QVAP)));
    float A = FDIV(FSUB(k.cice, FMUL(k.bice, k.dice)),
                   FMUL(FSUB(*TU, k.dice), FSUB(*TU, k.dice)));
    float B = FDIV(FMUL(RLS, k.ep2), P);
    float C = FDIV(FMUL(FMUL(A, B), ESICE), CCP);
    float DQVAP = FSUB(FDIV(FMUL(B, FSUB(ESLIQ, ESICE)),
                            FADD(RLS, FMUL(RLS, C))),
                       FDIV(FMUL(RLF, FADD(QLQFRZ, QNEW)),
                            FADD(RLS, FDIV(RLS, C))));
    float DTFRZ = FDIV(FADD(FMUL(RLF, FADD(QLQFRZ, QNEW)),
                            FMUL(B, FSUB(ESLIQ, ESICE))),
                       FADD(CCP, FMUL(FMUL(A, B), ESICE)));
    float TU1 = *TU;
    float QVAP1 = *QVAP;
    *TU = FADD(*TU, FMUL(*FRC1, DTFRZ));
    *QVAP = FSUB(*QVAP, FMUL(*FRC1, DQVAP));
    float ES = FDIV(FMUL(*QVAP, P), FADD(k.ep2, *QVAP));
    ESLIQ = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, *TU), k.cliq),
                                     FSUB(*TU, k.dliq))));
    ESICE = FMUL(k.aice, KO_EXP(FDIV(FSUB(FMUL(k.bice, *TU), k.cice),
                                     FSUB(*TU, k.dice))));
    *RATIO2 = FDIV(FSUB(ESLIQ, ES), FSUB(ESLIQ, ESICE));
    if (*IFLAG > 0 && *RATIO2 < 1.f) {
        *FRC1 = FADD(*FRC1, FSUB(1.f, *RATIO2));
        *TU = FADD(TU1, FMUL(*FRC1, DTFRZ));
        *QVAP = FSUB(QVAP1, FMUL(*FRC1, DQVAP));
        *RATIO2 = 1.f;
        *IFLAG = 1;
        goto L20;
    }
    if (*RATIO2 > 1.f) {
        *FRC1 = FSUB(*FRC1, FSUB(*RATIO2, 1.f));
        *FRC1 = ko_max(0.0f, *FRC1);
        *TU = FADD(TU1, FMUL(*FRC1, DTFRZ));
        *QVAP = FSUB(QVAP1, FMUL(*FRC1, DQVAP));
        *RATIO2 = 1.f;
        *IFLAG = 1;
    }
L20:
    RLC = FSUB(k.xlv0, FMUL(k.xlv1, *TU));
    RLS = FSUB(k.xls0, FMUL(k.xls1, *TU));
    *RL = FADD(FMUL(*RATIO2, RLS), FMUL(FSUB(1.f, *RATIO2), RLC));
    float PI = KO_POW(FDIV(1.E5f, P),
                      FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, *QVAP))));
    *THTEU = FMUL(FMUL(*TU, PI),
                  KO_EXP(FMUL(FDIV(FMUL(FMUL(*RL, *QVAP), C5), *TU),
                              FADD(1.f, FMUL(0.81f, *QVAP)))));
    if (*IFLAG == 1) {
        *QICE = FADD(FADD(*QICE, FMUL(*FRC1, DQVAP)), *QLIQ);
        *QLIQ = 0.f;
    } else {
        *QICE = FADD(*QICE, FMUL(*FRC1, FADD(DQVAP, QLQFRZ)));
        *QLIQ = FSUB(*QLIQ, FMUL(*FRC1, QLQFRZ));
    }
    *QNWFRZ = 0.f;
}

/* ------------------------------------------------------------------------
 * PROF5 (module_cu_kf.F:2275-2305)
 * ---------------------------------------------------------------------- */
__device__ void ko_prof5(float EQ, float* EE, float* UD) {
    const float SQRT2P = 2.506628f, A1 = 0.4361836f, A2 = -0.1201676f,
                A3 = 0.9372980f, P = 0.33267f, SIGMA = 0.166666667f,
                FE = 0.202765151f;
    float Y = FSUB(FMUL(6.f, EQ), 3.f);
    float EY = KO_EXP(FDIV(FMUL(Y, Y), -2.0f));
    /* EXP(-4.5): gfortran folds the constant; the folded word equals the
     * reference expf(-4.5) (oracle README), so the runtime call is the
     * same word and keeps one code path. */
    float E45 = KO_EXP(-4.5f);
    float T2 = FDIV(1.f, FADD(1.f, FMUL(P, fabsf(Y))));
    float T1 = 0.500498f;
    float C1 = FADD(FADD(FMUL(A1, T1), FMUL(FMUL(A2, T1), T1)),
                    FMUL(FMUL(FMUL(A3, T1), T1), T1));
    float C2 = FADD(FADD(FMUL(A1, T2), FMUL(FMUL(A2, T2), T2)),
                    FMUL(FMUL(FMUL(A3, T2), T2), T2));
    if (Y >= 0.f) {
        *EE = FSUB(FMUL(SIGMA, FADD(FMUL(0.5f, FSUB(FSUB(SQRT2P,
                                                         FMUL(E45, C1)),
                                                    FMUL(EY, C2))),
                                    FMUL(SIGMA, FSUB(E45, EY)))),
                   FDIV(FMUL(FMUL(E45, EQ), EQ), 2.f));
        *UD = FSUB(FMUL(SIGMA, FADD(FMUL(0.5f, FSUB(FMUL(EY, C2),
                                                    FMUL(E45, C1))),
                                    FMUL(SIGMA, FSUB(E45, EY)))),
                   FMUL(E45, FSUB(FADD(0.5f, FDIV(FMUL(EQ, EQ), 2.f)),
                                  EQ)));
    } else {
        *EE = FSUB(FMUL(SIGMA, FADD(FMUL(0.5f, FSUB(FMUL(EY, C2),
                                                    FMUL(E45, C1))),
                                    FMUL(SIGMA, FSUB(E45, EY)))),
                   FDIV(FMUL(FMUL(E45, EQ), EQ), 2.f));
        *UD = FSUB(FMUL(SIGMA, FADD(FMUL(0.5f, FSUB(FSUB(SQRT2P,
                                                         FMUL(E45, C1)),
                                                    FMUL(EY, C2))),
                                    FMUL(SIGMA, FSUB(E45, EY)))),
                   FMUL(E45, FSUB(FADD(0.5f, FDIV(FMUL(EQ, EQ), 2.f)),
                                  EQ)));
    }
    *EE = FDIV(*EE, FE);
    *UD = FDIV(*UD, FE);
}

/* ------------------------------------------------------------------------
 * TPMIX (module_cu_kf.F:2308-2462).  The three saturation branches are
 * spelled once in ko_thtgs; RATIO2 selects among them exactly as the
 * Fortran's IF/ELSEIF/ELSE does, with the same expressions.
 * ---------------------------------------------------------------------- */
__device__ __forceinline__ float ko_thtgs(float T, float P, float RATIO2,
                                          float RL, const KoConst& k,
                                          float* QS_out) {
    const float C5 = 1.0723E-3f;
    float ES, QS, PI, THTGS;
    if (RATIO2 < 1.E-6f) {
        ES = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, T), k.cliq),
                                      FSUB(T, k.dliq))));
        QS = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
        PI = KO_POW(FDIV(1.E5f, P),
                    FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QS))));
        THTGS = FMUL(FMUL(T, PI),
                     KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, T), 2.5403f),
                                      QS),
                                 FADD(1.f, FMUL(0.81f, QS)))));
    } else if (fabsf(FSUB(RATIO2, 1.f)) < 1.E-6f) {
        ES = FMUL(k.aice, KO_EXP(FDIV(FSUB(FMUL(k.bice, T), k.cice),
                                      FSUB(T, k.dice))));
        QS = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
        PI = KO_POW(FDIV(1.E5f, P),
                    FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QS))));
        THTGS = FMUL(FMUL(T, PI),
                     KO_EXP(FMUL(FMUL(FSUB(FDIV(3114.834f, T), 0.278296f),
                                      QS),
                                 FADD(1.f, FMUL(0.81f, QS)))));
    } else {
        float ESLIQ = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, T), k.cliq),
                                               FSUB(T, k.dliq))));
        float ESICE = FMUL(k.aice, KO_EXP(FDIV(FSUB(FMUL(k.bice, T), k.cice),
                                               FSUB(T, k.dice))));
        ES = FADD(FMUL(FSUB(1.f, RATIO2), ESLIQ), FMUL(RATIO2, ESICE));
        QS = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
        PI = KO_POW(FDIV(1.E5f, P),
                    FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QS))));
        THTGS = FMUL(FMUL(T, PI),
                     KO_EXP(FMUL(FDIV(FMUL(FMUL(RL, QS), C5), T),
                                 FADD(1.f, FMUL(0.81f, QS)))));
    }
    *QS_out = QS;
    return THTGS;
}

__device__ void ko_tpmix(float P, float THTU, float* TU, float* QU,
                         float* QLIQ, float* QICE, float* QNEWLQ,
                         float* QNEWIC, float RATIO2, float RL,
                         const KoConst& k) {
    float QS, THTGS, F0, T1, T0, F1, DT, QNEW, DQ, QTOT, DQICE, DQLIQ,
          RLL, CCP;
    int ITCNT;
    THTGS = ko_thtgs(*TU, P, RATIO2, RL, k, &QS);
    F0 = FSUB(THTGS, THTU);
    T1 = FSUB(*TU, FMUL(0.5f, F0));
    T0 = *TU;
    ITCNT = 0;
L90:
    THTGS = ko_thtgs(T1, P, RATIO2, RL, k, &QS);
    F1 = FSUB(THTGS, THTU);
    if (fabsf(F1) < 0.01f) goto L50;
    ITCNT = ITCNT + 1;
    if (ITCNT > 10) goto L50;
    DT = FDIV(FMUL(F1, FSUB(T1, T0)), FSUB(F1, F0));
    T0 = T1;
    F0 = F1;
    T1 = FSUB(T1, DT);
    goto L90;
L50:
    if (QS <= *QU) {
        QNEW = FSUB(*QU, QS);
        *QU = QS;
        goto L96;
    }
    QNEW = 0.f;
    DQ = FSUB(QS, *QU);
    QTOT = FADD(*QLIQ, *QICE);
    if (QTOT >= DQ) {
        DQICE = 0.0f;
        DQLIQ = 0.0f;
        *QLIQ = FSUB(*QLIQ, FMUL(FSUB(1.f, RATIO2), DQ));
        if (*QLIQ < 0.f) {
            DQICE = FSUB(0.0f, *QLIQ);
            *QLIQ = 0.0f;
        }
        *QICE = FADD(FSUB(*QICE, FMUL(RATIO2, DQ)), DQICE);
        if (*QICE < 0.f) {
            DQLIQ = FSUB(0.0f, *QICE);
            *QICE = 0.0f;
        }
        *QLIQ = FADD(*QLIQ, DQLIQ);
        *QU = QS;
        goto L96;
    } else {
        if (RATIO2 < 1.E-6f) {
            RLL = FSUB(k.xlv0, FMUL(k.xlv1, T1));
        } else if (fabsf(FSUB(RATIO2, 1.f)) < 1.E-6f) {
            RLL = FSUB(k.xls0, FMUL(k.xls1, T1));
        } else {
            RLL = RL;
        }
        CCP = FMUL(1005.7f, FADD(1.f, FMUL(0.89f, *QU)));
        if (QTOT < 1.E-10f) {
            T1 = FADD(T1, FDIV(FMUL(RLL, FDIV(DQ, FADD(1.f, DQ))), CCP));
            goto L96;
        } else {
            T1 = FADD(T1, FDIV(FMUL(RLL, FDIV(FSUB(DQ, QTOT),
                                              FSUB(FADD(1.f, DQ), QTOT))),
                               CCP));
            *QU = FADD(*QU, QTOT);
            QTOT = 0.f;
        }
        *QLIQ = 0.f;
        *QICE = 0.f;
    }
L96:
    *TU = T1;
    *QNEWLQ = FMUL(FSUB(1.f, RATIO2), QNEW);
    *QNEWIC = FMUL(RATIO2, QNEW);
}

/* ------------------------------------------------------------------------
 * ENVIRTHT (module_cu_kf.F:2464-2508)
 * ---------------------------------------------------------------------- */
__device__ float ko_envirtht(float P1, float T1, float Q1, float R1,
                             float RL, const KoConst& k) {
    const float T00 = 273.16f, P00 = 1.E5f, C1 = 3374.6525f, C2 = 2.5403f,
                C3 = 3114.834f, C4 = 0.278296f, C5 = 1.0723E-3f;
    float EE, TLOG, TDPT, TSAT, THT, TFPT, TLOGIC, TSATLQ, TSATIC;
    if (R1 < 1.E-6f) {
        EE = FDIV(FMUL(Q1, P1), FADD(k.ep2, Q1));
        TLOG = KO_LOG(FDIV(EE, k.aliq));
        TDPT = FDIV(FSUB(k.cliq, FMUL(k.dliq, TLOG)), FSUB(k.bliq, TLOG));
        TSAT = FSUB(TDPT, FMUL(FSUB(FADD(0.212f, FMUL(1.571E-3f,
                                                      FSUB(TDPT, T00))),
                                    FMUL(4.36E-4f, FSUB(T1, T00))),
                               FSUB(T1, TDPT)));
        THT = FMUL(T1, KO_POW(FDIV(P00, P1),
                              FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, Q1)))));
        return FMUL(THT, KO_EXP(FMUL(FMUL(FSUB(FDIV(C1, TSAT), C2), Q1),
                                     FADD(1.f, FMUL(0.81f, Q1)))));
    } else if (fabsf(FSUB(R1, 1.f)) < 1.E-6f) {
        EE = FDIV(FMUL(Q1, P1), FADD(k.ep2, Q1));
        TLOG = KO_LOG(FDIV(EE, k.aice));
        TFPT = FDIV(FSUB(k.cice, FMUL(k.dice, TLOG)), FSUB(k.bice, TLOG));
        THT = FMUL(T1, KO_POW(FDIV(P00, P1),
                              FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, Q1)))));
        TSAT = FSUB(TFPT, FMUL(FSUB(FADD(0.182f, FMUL(1.13E-3f,
                                                      FSUB(TFPT, T00))),
                                    FMUL(3.58E-4f, FSUB(T1, T00))),
                               FSUB(T1, TFPT)));
        return FMUL(THT, KO_EXP(FMUL(FMUL(FSUB(FDIV(C3, TSAT), C4), Q1),
                                     FADD(1.f, FMUL(0.81f, Q1)))));
    } else {
        EE = FDIV(FMUL(Q1, P1), FADD(k.ep2, Q1));
        TLOG = KO_LOG(FDIV(EE, k.aliq));
        TDPT = FDIV(FSUB(k.cliq, FMUL(k.dliq, TLOG)), FSUB(k.bliq, TLOG));
        TLOGIC = KO_LOG(FDIV(EE, k.aice));
        TFPT = FDIV(FSUB(k.cice, FMUL(k.dice, TLOGIC)),
                    FSUB(k.bice, TLOGIC));
        THT = FMUL(T1, KO_POW(FDIV(P00, P1),
                              FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, Q1)))));
        TSATLQ = FSUB(TDPT, FMUL(FSUB(FADD(0.212f, FMUL(1.571E-3f,
                                                        FSUB(TDPT, T00))),
                                      FMUL(4.36E-4f, FSUB(T1, T00))),
                                 FSUB(T1, TDPT)));
        TSATIC = FSUB(TFPT, FMUL(FSUB(FADD(0.182f, FMUL(1.13E-3f,
                                                        FSUB(TFPT, T00))),
                                      FMUL(3.58E-4f, FSUB(T1, T00))),
                                 FSUB(T1, TFPT)));
        TSAT = FADD(FMUL(R1, TSATIC), FMUL(FSUB(1.f, R1), TSATLQ));
        return FMUL(THT, KO_EXP(FMUL(FDIV(FMUL(FMUL(RL, Q1), C5), TSAT),
                                     FADD(1.f, FMUL(0.81f, Q1)))));
    }
}

/* ------------------------------------------------------------------------
 * TPDD (module_cu_kf.F:2518-2580).  KFPARA's only call passes RH = 1.0,
 * so the RH < 1 tail (:2561-2575) is unreachable; it is transcribed anyway
 * so the function is the Fortran function and not a specialisation of it.
 * ---------------------------------------------------------------------- */
__device__ float ko_tpdd(float P, float THTED, float TGS, float* RS,
                         float RD, float RH, const KoConst& k) {
    float ES, PI, THTGS, F0, T1, T0, CCP, F1, DT, RL, DSSDT, T1RH, RSRH;
    int ITCNT;
    ES = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, TGS), k.cliq),
                                  FSUB(TGS, k.dliq))));
    *RS = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
    PI = KO_POW(FDIV(1.E5f, P), FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, *RS))));
    THTGS = FMUL(FMUL(TGS, PI),
                 KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TGS), 2.5403f), *RS),
                             FADD(1.f, FMUL(0.81f, *RS)))));
    F0 = FSUB(THTGS, THTED);
    T1 = FSUB(TGS, FMUL(0.5f, F0));
    T0 = TGS;
    CCP = 1005.7f;
    ITCNT = 0;
L90:
    ES = FMUL(k.aliq, KO_EXP(FDIV(FSUB(FMUL(k.bliq, T1), k.cliq),
                                  FSUB(T1, k.dliq))));
    *RS = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
    PI = KO_POW(FDIV(1.E5f, P), FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, *RS))));
    THTGS = FMUL(FMUL(T1, PI),
                 KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, T1), 2.5403f), *RS),
                             FADD(1.f, FMUL(0.81f, *RS)))));
    F1 = FSUB(THTGS, THTED);
    if (fabsf(F1) < 0.05f) goto L50;
    ITCNT = ITCNT + 1;
    if (ITCNT > 10) goto L50;
    DT = FDIV(FMUL(F1, FSUB(T1, T0)), FSUB(F1, F0));
    T0 = T1;
    F0 = F1;
    T1 = FSUB(T1, DT);
    goto L90;
L50:
    RL = FSUB(k.xlv0, FMUL(k.xlv1, T1));
    if (RH == 1.f) goto L110;
    DSSDT = FDIV(FSUB(k.cliq, FMUL(k.bliq, k.dliq)),
                 FMUL(FSUB(T1, k.dliq), FSUB(T1, k.dliq)));
    DT = FDIV(FMUL(FMUL(RL, *RS), FSUB(1.f, RH)),
              FADD(CCP, FMUL(FMUL(FMUL(RL, RH), *RS), DSSDT)));
    T1RH = FADD(T1, DT);
    ES = FMUL(FMUL(RH, k.aliq), KO_EXP(FDIV(FSUB(FMUL(k.bliq, T1RH), k.cliq),
                                            FSUB(T1RH, k.dliq))));
    RSRH = FDIV(FMUL(k.ep2, ES), FSUB(P, ES));
    if (RSRH < RD) {
        RSRH = RD;
        T1RH = FADD(T1, FDIV(FMUL(FSUB(*RS, RSRH), RL), CCP));
    }
    T1 = T1RH;
    *RS = RSRH;
L110:
    return T1;
}

/* ==========================================================================
 * KFPARA (module_cu_kf.F:334-2082), one column.
 *
 * Indices are Fortran's: every column array is addressed 1..KX (OMG to
 * KX+1) through the macros below, so each statement reads as its source
 * line.  The labels are the Fortran labels.
 * ========================================================================== */
#define KO_IN(arr, kk) ((arr)[(size_t)((kk) - 1) * plane + col])

__device__ void ko_kfpara(
        const float* __restrict__ u_in, const float* __restrict__ v_in,
        const float* __restrict__ t_in, const float* __restrict__ qv_in,
        const float* __restrict__ p_in, const float* __restrict__ dz_in,
        const float* __restrict__ rho_in, const float* __restrict__ w0avg,
        size_t plane, size_t col, int KX,
        float DT, float DX, float DXSQ, const KoConst& kc,
        int warm_rain, int qi_flag, int qs_flag,
        float* __restrict__ dtdt_out, float* __restrict__ dqdt_out,
        float* __restrict__ dqcdt_out, float* __restrict__ dqrdt_out,
        float* __restrict__ dqidt_out, float* __restrict__ dqsdt_out,
        float* RAINCV, float* PRATEC, float* NCA, int* FATAL,
        int KF_EDRATES,
        float* __restrict__ udr_kf, float* __restrict__ ddr_kf,
        float* __restrict__ uer_kf, float* __restrict__ der_kf,
        float* TIMEC_KF, float* base, int kp)
{
#define KO_SLOT(i) (KoCol{base + (size_t)(i) * (size_t)kp * KO_LANES})
    const KoCol q0 = KO_SLOT(0), z0 = KO_SLOT(1), tv0 = KO_SLOT(2),
        tu = KO_SLOT(3), tvu = KO_SLOT(4), qu = KO_SLOT(5), tz = KO_SLOT(6),
        qd = KO_SLOT(7), qes = KO_SLOT(8), thtes = KO_SLOT(9),
        tg = KO_SLOT(10), tvg = KO_SLOT(11), qg = KO_SLOT(12),
        ems = KO_SLOT(13), emsd = KO_SLOT(14), umf = KO_SLOT(15),
        uer = KO_SLOT(16), udr = KO_SLOT(17), dmf = KO_SLOT(18),
        der = KO_SLOT(19), ddr = KO_SLOT(20), umf2 = KO_SLOT(21),
        uer2 = KO_SLOT(22), udr2 = KO_SLOT(23), dmf2 = KO_SLOT(24),
        der2 = KO_SLOT(25), ddr2 = KO_SLOT(26), dza = KO_SLOT(27),
        thta0 = KO_SLOT(28), thetee = KO_SLOT(29), thtau = KO_SLOT(30),
        theteu = KO_SLOT(31), thtad = KO_SLOT(32), theted = KO_SLOT(33),
        qliq = KO_SLOT(34), qice = KO_SLOT(35), qlqout = KO_SLOT(36),
        qicout = KO_SLOT(37), pptliq = KO_SLOT(38), pptice = KO_SLOT(39),
        detlq = KO_SLOT(40), detic = KO_SLOT(41), detlq2 = KO_SLOT(42),
        detic2 = KO_SLOT(43), ratio2 = KO_SLOT(44), domgdp = KO_SLOT(45),
        exn = KO_SLOT(46), tvqu = KO_SLOT(47), dp = KO_SLOT(48),
        eqfrc = KO_SLOT(49), qdt = KO_SLOT(50), fxm = KO_SLOT(51),
        thtesg = KO_SLOT(52), thpa = KO_SLOT(53), thfxtop = KO_SLOT(54),
        thfxbot = KO_SLOT(55), qpa = KO_SLOT(56), qfxtop = KO_SLOT(57),
        qfxbot = KO_SLOT(58), qlpa = KO_SLOT(59), qlfxin = KO_SLOT(60),
        qlfxout = KO_SLOT(61), qipa = KO_SLOT(62), qifxin = KO_SLOT(63),
        qifxout = KO_SLOT(64), qrpa = KO_SLOT(65), qrfxin = KO_SLOT(66),
        qrfxout = KO_SLOT(67), qspa = KO_SLOT(68), qsfxin = KO_SLOT(69),
        qsfxout = KO_SLOT(70), qlg = KO_SLOT(71), qig = KO_SLOT(72),
        qrg = KO_SLOT(73), qsg = KO_SLOT(74), omg = KO_SLOT(75),
        rainfb = KO_SLOT(76), snowfb = KO_SLOT(77);
#undef KO_SLOT
    /* KFPARA's local arrays, one-based like the Fortran. */
#define Q0(k) q0[(k) - 1]
#define Z0(k) z0[(k) - 1]
#define TV0(k) tv0[(k) - 1]
#define TU(k) tu[(k) - 1]
#define TVU(k) tvu[(k) - 1]
#define QU(k) qu[(k) - 1]
#define TZ(k) tz[(k) - 1]
#define QD(k) qd[(k) - 1]
#define QES(k) qes[(k) - 1]
#define THTES(k) thtes[(k) - 1]
#define TG(k) tg[(k) - 1]
#define TVG(k) tvg[(k) - 1]
#define QG(k) qg[(k) - 1]
#define EMS(k) ems[(k) - 1]
#define EMSD(k) emsd[(k) - 1]
#define UMF(k) umf[(k) - 1]
#define UER(k) uer[(k) - 1]
#define UDR(k) udr[(k) - 1]
#define DMF(k) dmf[(k) - 1]
#define DER(k) der[(k) - 1]
#define DDR(k) ddr[(k) - 1]
#define UMF2(k) umf2[(k) - 1]
#define UER2(k) uer2[(k) - 1]
#define UDR2(k) udr2[(k) - 1]
#define DMF2(k) dmf2[(k) - 1]
#define DER2(k) der2[(k) - 1]
#define DDR2(k) ddr2[(k) - 1]
#define DZA(k) dza[(k) - 1]
#define THTA0(k) thta0[(k) - 1]
#define THETEE(k) thetee[(k) - 1]
#define THTAU(k) thtau[(k) - 1]
#define THETEU(k) theteu[(k) - 1]
#define THTAD(k) thtad[(k) - 1]
#define THETED(k) theted[(k) - 1]
#define QLIQ(k) qliq[(k) - 1]
#define QICE(k) qice[(k) - 1]
#define QLQOUT(k) qlqout[(k) - 1]
#define QICOUT(k) qicout[(k) - 1]
#define PPTLIQ(k) pptliq[(k) - 1]
#define PPTICE(k) pptice[(k) - 1]
#define DETLQ(k) detlq[(k) - 1]
#define DETIC(k) detic[(k) - 1]
#define DETLQ2(k) detlq2[(k) - 1]
#define DETIC2(k) detic2[(k) - 1]
#define RATIO2(k) ratio2[(k) - 1]
#define DOMGDP(k) domgdp[(k) - 1]
#define EXN(k) exn[(k) - 1]
#define TVQU(k) tvqu[(k) - 1]
#define DP(k) dp[(k) - 1]
#define EQFRC(k) eqfrc[(k) - 1]
#define QDT(k) qdt[(k) - 1]
#define FXM(k) fxm[(k) - 1]
#define THTESG(k) thtesg[(k) - 1]
#define THPA(k) thpa[(k) - 1]
#define THFXTOP(k) thfxtop[(k) - 1]
#define THFXBOT(k) thfxbot[(k) - 1]
#define QPA(k) qpa[(k) - 1]
#define QFXTOP(k) qfxtop[(k) - 1]
#define QFXBOT(k) qfxbot[(k) - 1]
#define QLPA(k) qlpa[(k) - 1]
#define QLFXIN(k) qlfxin[(k) - 1]
#define QLFXOUT(k) qlfxout[(k) - 1]
#define QIPA(k) qipa[(k) - 1]
#define QIFXIN(k) qifxin[(k) - 1]
#define QIFXOUT(k) qifxout[(k) - 1]
#define QRPA(k) qrpa[(k) - 1]
#define QRFXIN(k) qrfxin[(k) - 1]
#define QRFXOUT(k) qrfxout[(k) - 1]
#define QSPA(k) qspa[(k) - 1]
#define QSFXIN(k) qsfxin[(k) - 1]
#define QSFXOUT(k) qsfxout[(k) - 1]
#define QLG(k) qlg[(k) - 1]
#define QIG(k) qig[(k) - 1]
#define QRG(k) qrg[(k) - 1]
#define QSG(k) qsg[(k) - 1]
#define OMG(k) omg[(k) - 1]
#define RAINFB(k) rainfb[(k) - 1]
#define SNOWFB(k) snowfb[(k) - 1]
    /* The arguments KFCPS hands KFPARA. */
#define U0(k) KO_IN(u_in, k)
#define V0(k) KO_IN(v_in, k)
#define T0(k) KO_IN(t_in, k)
#define QV0(k) KO_IN(qv_in, k)
#define P0(k) KO_IN(p_in, k)
#define DZQ(k) KO_IN(dz_in, k)
#define RHO(k) KO_IN(rho_in, k)
#define W0AVG1D(k) KO_IN(w0avg, k)
#define DQDT(k) KO_IN(dqdt_out, k)
#define DQIDT(k) KO_IN(dqidt_out, k)
#define DQCDT(k) KO_IN(dqcdt_out, k)
#define DQRDT(k) KO_IN(dqrdt_out, k)
#define DQSDT(k) KO_IN(dqsdt_out, k)
#define DTDT(k) KO_IN(dtdt_out, k)
    /* QL0, QI0, QR0, QS0 are zero at every level (:536-539) and never
     * written again; they stay literal zeros so every expression that
     * reads them keeps its operation and its rounding. */
#define QL0(k) 0.f
#define QI0(k) 0.f
#define QR0(k) 0.f
#define QS0(k) 0.f

    /* DATA (:464-468) */
    const float P00 = 1.E5f, T00 = 273.16f, B61 = 0.608f, RLF = 3.339E5f;
    const float RHBC = 0.90f, PIE = 3.141592654f;
    const float TTFRZ = 268.16f, TBFRZ = 248.16f, RATE = 0.01f;
    const float CP = kc.cp, R = kc.r, G = kc.g, EP2 = kc.ep2;
    const float XLV0 = kc.xlv0, XLV1 = kc.xlv1;
    const float ALIQ = kc.aliq, BLIQ = kc.bliq, CLIQ = kc.cliq,
                DLIQ = kc.dliq;

    /* Every scalar zero-initialised.  The Fortran leaves them to the
     * stack; the oracle's poisoned arm (build.sh arm N) proves no graded
     * output depends on that. */
    float GDRY = 0.f, FBFRC = 0.f, P300 = 0.f, DPTHMX = 0.f, THMIX = 0.f,
          QMIX = 0.f, ZMIX = 0.f, PMIX = 0.f, ROCPQ = 0.f, TMIX = 0.f,
          EMIX = 0.f, TLOG = 0.f, TDPT = 0.f, TLCL = 0.f, TVLCL = 0.f,
          CPORQ = 0.f, PLCL = 0.f, ES = 0.f, DLP = 0.f, TENV = 0.f,
          QENV = 0.f, TVEN = 0.f, ZLCL = 0.f, WKL = 0.f, WABS = 0.f,
          TRPPT = 0.f, WSIGNE = 0.f, DTLCL = 0.f, GDT = 0.f, WLCL = 0.f,
          TVAVG = 0.f, QESE = 0.f, WTW = 0.f, RHOLCL = 0.f, AU0 = 0.f,
          VMFLCL = 0.f, UPOLD = 0.f, UPNEW = 0.f, ABE = 0.f, WKLCL = 0.f,
          THTUDL = 0.f, TTEMP = 0.f, FRC1 = 0.f, QNEWIC = 0.f, RL = 0.f,
          R1 = 0.f, QNWFRZ = 0.f, EFFQ = 0.f, BE = 0.f, BOTERM = 0.f,
          ENTERM = 0.f, DZZ = 0.f, UDLBE = 0.f, REI = 0.f, EE2 = 0.f,
          UD2 = 0.f, TTMP = 0.f, F1 = 0.f, F2 = 0.f, THTTMP = 0.f,
          QTMP = 0.f, TMPLIQ = 0.f, TMPICE = 0.f, TU95 = 0.f, TU10 = 0.f,
          EE1 = 0.f, UD1 = 0.f, CLDHGT = 0.f, DPTT = 0.f, QNEWLQ = 0.f,
          DUMFDP = 0.f, EE = 0.f, TSAT = 0.f, THTA = 0.f, P150 = 0.f,
          USR = 0.f, VCONV = 0.f, TIMEC = 0.f, SHSIGN = 0.f, VWS = 0.f,
          PEF = 0.f, CBH = 0.f, RCBH = 0.f, PEFCBH = 0.f, PEFF = 0.f,
          PEFF2 = 0.f, TDER = 0.f, THTMIN = 0.f, DTMLTD = 0.f, QS = 0.f,
          TADVEC = 0.f, DPDD = 0.f, FRC = 0.f, RDD = 0.f, A1 = 0.f,
          DSSDT = 0.f, DTMP = 0.f, T1RH = 0.f, QSRH = 0.f, PPTFLX = 0.f,
          CPR = 0.f, CNDTNF = 0.f, UPDINC = 0.f, DEVDMF = 0.f, PPR = 0.f,
          RCED = 0.f, DPPTDF = 0.f, DMFLFS = 0.f, DMFLFS2 = 0.f,
          RCED2 = 0.f, DDINC = 0.f, AINCMX = 0.f, AINCM1 = 0.f,
          AINC = 0.f, TDER2 = 0.f, PPTFL2 = 0.f, FABE = 0.f, STAB = 0.f,
          DTT = 0.f, DTT1 = 0.f, DTIME = 0.f, TMA = 0.f, TMB = 0.f,
          TMM = 0.f, BCOEFF = 0.f, ACOEFF = 0.f, TOPOMG = 0.f, CPM = 0.f,
          DQ = 0.f, ABEG = 0.f, DABE = 0.f, DFDA = 0.f, FRC2 = 0.f,
          QINIT = 0.f, QFNL = 0.f, ERR2 = 0.f, FABEOLD = 0.f,
          AINCOLD = 0.f, TVD_LFS = 0.f, WSPD_KLCL = 0.f, WSPD_L5 = 0.f,
          WSPD_LTOP = 0.f;
    int KL, K = 0, ISTOP = 0, ML = 0, L5 = 0, KMIX = 0, LOW = 0, LC = 0,
        LLFC = 0, NLAYRS = 0, NK = 0, KPBL = 0, KLCL = 0, LCL = 0, LET = 0,
        IFLAG = 0, NK1 = 0, LTOP = 0, LTOP1 = 0, LTOPM1 = 0, LVF = 0,
        KSTART = 0, KMIN = 0, LFS = 0, ND = 0, NIC = 0, LDB = 0, LDT = 0,
        ND1 = 0, NDK = 0, NM = 0, LMAX = 0, NCOUNT = 0, NOITR = 0,
        NSTEP = 0, NTC = 0, NJ = 0;

    GDRY = -FDIV(G, CP);
    KL = KX;

    /* ALIQ..DICE are set by the caller (:479-486) */
    FBFRC = 0.0f;
    P300 = FSUB(P0(1), 30000.f);
    ML = 0;
    for (int Kk = 1; Kk <= KX; ++Kk) {                         /* DO 15 */
        ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(BLIQ, T0(Kk)), CLIQ),
                                    FSUB(T0(Kk), DLIQ))));
        QES(Kk) = FDIV(FMUL(EP2, ES), FSUB(P0(Kk), ES));
        Q0(Kk) = ko_min(QES(Kk), QV0(Kk));
        Q0(Kk) = ko_max(0.000001f, Q0(Kk));
        TV0(Kk) = FMUL(T0(Kk), FADD(1.f, FMUL(B61, Q0(Kk))));
        DP(Kk) = FMUL(FMUL(RHO(Kk), G), DZQ(Kk));
        if (P0(Kk) >= 500E2f) L5 = Kk;
        if (P0(Kk) >= P300) LLFC = Kk;
        if (T0(Kk) > T00) ML = Kk;
    }
    Z0(1) = FMUL(.5f, DZQ(1));
    for (int Kk = 2; Kk <= KL; ++Kk) {                         /* DO 20 */
        Z0(Kk) = FADD(Z0(Kk - 1), FMUL(.5f, FADD(DZQ(Kk), DZQ(Kk - 1))));
        DZA(Kk - 1) = FSUB(Z0(Kk), Z0(Kk - 1));
    }
    DZA(KL) = 0.f;
    KMIX = 1;
L25:
    LOW = KMIX;
    if (LOW > LLFC) goto L325;
    LC = LOW;
    NLAYRS = 0;
    DPTHMX = 0.f;
    for (NK = LC; NK <= KX; ++NK) {                            /* DO 63 */
        DPTHMX = FADD(DPTHMX, DP(NK));
        NLAYRS = NLAYRS + 1;
        if (DPTHMX > 6.E3f) goto L64;
    }
    goto L325;
L64:
    KPBL = LC + NLAYRS - 1;
    KMIX = LC + 1;
    THMIX = 0.f;
    QMIX = 0.f;
    ZMIX = 0.f;
    PMIX = 0.f;
    DPTHMX = 0.f;
    for (NK = LC; NK <= KPBL; ++NK) {                          /* DO 17 */
        DPTHMX = FADD(DPTHMX, DP(NK));
        ROCPQ = FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, Q0(NK))));
        THMIX = FADD(THMIX, FMUL(FMUL(DP(NK), T0(NK)),
                                 KO_POW(FDIV(P00, P0(NK)), ROCPQ)));
        QMIX = FADD(QMIX, FMUL(DP(NK), Q0(NK)));
        ZMIX = FADD(ZMIX, FMUL(DP(NK), Z0(NK)));
        PMIX = FADD(PMIX, FMUL(DP(NK), P0(NK)));
    }
    THMIX = FDIV(THMIX, DPTHMX);
    QMIX = FDIV(QMIX, DPTHMX);
    ZMIX = FDIV(ZMIX, DPTHMX);
    PMIX = FDIV(PMIX, DPTHMX);
    ROCPQ = FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QMIX)));
    TMIX = FMUL(THMIX, KO_POW(FDIV(PMIX, P00), ROCPQ));
    EMIX = FDIV(FMUL(QMIX, PMIX), FADD(EP2, QMIX));
    TLOG = KO_LOG(FDIV(EMIX, ALIQ));
    TDPT = FDIV(FSUB(CLIQ, FMUL(DLIQ, TLOG)), FSUB(BLIQ, TLOG));
    TLCL = FSUB(TDPT, FMUL(FSUB(FADD(.212f, FMUL(1.571E-3f, FSUB(TDPT, T00))),
                                FMUL(4.36E-4f, FSUB(TMIX, T00))),
                           FSUB(TMIX, TDPT)));
    TLCL = ko_min(TLCL, TMIX);
    TVLCL = FMUL(TLCL, FADD(1.f, FMUL(0.608f, QMIX)));
    CPORQ = FDIV(1.f, ROCPQ);
    PLCL = FMUL(P00, KO_POW(FDIV(TLCL, THMIX), CPORQ));
    for (NK = LC; NK <= KL; ++NK) {                            /* DO 29 */
        KLCL = NK;
        if (PLCL >= P0(NK)) goto L35;
    }
    goto L325;
L35:
    K = KLCL - 1;
    DLP = FDIV(KO_LOG(FDIV(PLCL, P0(K))), KO_LOG(FDIV(P0(KLCL), P0(K))));
    TENV = FADD(T0(K), FMUL(FSUB(T0(KLCL), T0(K)), DLP));
    QENV = FADD(Q0(K), FMUL(FSUB(Q0(KLCL), Q0(K)), DLP));
    TVEN = FMUL(TENV, FADD(1.f, FMUL(0.608f, QENV)));
    ZLCL = FADD(Z0(K), FMUL(FSUB(Z0(KLCL), Z0(K)), DLP));
    WKLCL = FDIV(FMUL(0.02f, ZLCL), 2.5E3f);
    WKL = FSUB(FDIV(FMUL(FADD(W0AVG1D(K),
                              FMUL(FSUB(W0AVG1D(KLCL), W0AVG1D(K)), DLP)),
                         DX), 25.E3f), WKLCL);
    WABS = FADD(fabsf(WKL), 1.E-10f);
    WSIGNE = FDIV(WKL, WABS);
    DTLCL = FMUL(FMUL(4.64f, WSIGNE), KO_POW(WABS, 0.33f));
    GDT = FDIV(FMUL(FMUL(G, DTLCL), FSUB(ZLCL, Z0(LC))),
               FADD(TV0(LC), TVEN));
    WLCL = FADD(1.f, FMUL(FMUL(.5f, WSIGNE),
                          KO_SQRT(FADD(fabsf(GDT), 1.E-10f))));
    if (FADD(TLCL, DTLCL) > TENV) goto L45;
    if (KPBL >= LLFC) goto L325;
    goto L25;
L45:
    THETEU(K) = FMUL(FMUL(TMIX, KO_POW(FDIV(1.E5f, PMIX),
                                       FMUL(0.2854f, FSUB(1.f,
                                                          FMUL(0.28f, QMIX))))),
                     KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TLCL), 2.5403f),
                                      QMIX),
                                 FADD(1.f, FMUL(0.81f, QMIX)))));
    ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TENV, BLIQ), CLIQ),
                                FSUB(TENV, DLIQ))));
    TVAVG = FMUL(0.5f, FADD(TV0(KLCL),
                            FMUL(TENV, FADD(1.f, FMUL(0.608f, QENV)))));
    PLCL = FMUL(P0(KLCL), KO_EXP(FMUL(FDIV(G, FMUL(R, TVAVG)),
                                      FSUB(Z0(KLCL), ZLCL))));
    QESE = FDIV(FMUL(EP2, ES), FSUB(PLCL, ES));
    GDT = FDIV(FMUL(FMUL(G, DTLCL), FSUB(ZLCL, Z0(LC))),
               FADD(TV0(LC), TVEN));
    WLCL = FADD(1.f, FMUL(FMUL(.5f, WSIGNE),
                          KO_SQRT(FADD(fabsf(GDT), 1.E-10f))));
    THTES(K) = FMUL(FMUL(TENV, KO_POW(FDIV(1.E5f, PLCL),
                                      FMUL(0.2854f, FSUB(1.f,
                                                         FMUL(0.28f, QESE))))),
                    KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TENV), 2.5403f),
                                     QESE),
                                FADD(1.f, FMUL(0.81f, QESE)))));
    WTW = FMUL(WLCL, WLCL);
    if (WLCL < 0.f) goto L25;
    TVLCL = FMUL(TLCL, FADD(1.f, FMUL(0.608f, QMIX)));
    RHOLCL = FDIV(PLCL, FMUL(R, TVLCL));
    LCL = KLCL;
    LET = LCL;
    /* UPDRAFT (:689-957) */
    AU0 = FMUL(FMUL(PIE, KO_RAD), KO_RAD);
    UMF(K) = FMUL(RHOLCL, AU0);
    VMFLCL = UMF(K);
    UPOLD = VMFLCL;
    UPNEW = UPOLD;
    RATIO2(K) = 0.f;
    UER(K) = 0.f;
    ABE = 0.f;
    TRPPT = 0.f;
    TU(K) = TLCL;
    TVU(K) = TVLCL;
    QU(K) = QMIX;
    EQFRC(K) = 1.f;
    QLIQ(K) = 0.f;
    QICE(K) = 0.f;
    QLQOUT(K) = 0.f;
    QICOUT(K) = 0.f;
    DETLQ(K) = 0.f;
    DETIC(K) = 0.f;
    PPTLIQ(K) = 0.f;
    PPTICE(K) = 0.f;
    IFLAG = 0;
    THTUDL = THETEU(K);
    TTEMP = TTFRZ;
    for (NK = K; NK <= KL - 1; ++NK) {                         /* DO 60 */
        NK1 = NK + 1;
        RATIO2(NK1) = RATIO2(NK);
        FRC1 = 0.f;
        TU(NK1) = T0(NK1);
        THETEU(NK1) = THETEU(NK);
        QU(NK1) = QU(NK);
        QLIQ(NK1) = QLIQ(NK);
        QICE(NK1) = QICE(NK);
        {
            float tu_ = TU(NK1), qu_ = QU(NK1), ql_ = QLIQ(NK1),
                  qi_ = QICE(NK1);
            ko_tpmix(P0(NK1), THETEU(NK1), &tu_, &qu_, &ql_, &qi_, &QNEWLQ,
                     &QNEWIC, RATIO2(NK1), RL, kc);
            TU(NK1) = tu_; QU(NK1) = qu_; QLIQ(NK1) = ql_; QICE(NK1) = qi_;
        }
        TVU(NK1) = FMUL(TU(NK1), FADD(1.f, FMUL(0.608f, QU(NK1))));
        if (TU(NK1) <= TTFRZ && IFLAG < 1) {
            if (TU(NK1) > TBFRZ) {
                if (TTEMP > TTFRZ) TTEMP = TTFRZ;
                FRC1 = FDIV(FSUB(TTEMP, TU(NK1)), FSUB(TTFRZ, TBFRZ));
                R1 = FDIV(FSUB(TTEMP, TU(NK1)), FSUB(TTEMP, TBFRZ));
            } else {
                FRC1 = FDIV(FSUB(TTEMP, TBFRZ), FSUB(TTFRZ, TBFRZ));
                R1 = 1.f;
                IFLAG = 1;
            }
            QNWFRZ = QNEWLQ;
            QNEWIC = FADD(QNEWIC, FMUL(FMUL(QNEWLQ, R1), 0.5f));
            QNEWLQ = FSUB(QNEWLQ, FMUL(FMUL(QNEWLQ, R1), 0.5f));
            EFFQ = FDIV(FSUB(TTFRZ, TBFRZ), FSUB(TTEMP, TBFRZ));
            TTEMP = TU(NK1);
        }
        if (NK == K) {
            BE = FSUB(FDIV(FADD(TVLCL, TVU(NK1)), FADD(TVEN, TV0(NK1))), 1.f);
            BOTERM = FDIV(FMUL(FMUL(FMUL(2.f, FSUB(Z0(NK1), ZLCL)), G), BE),
                          1.5f);
            ENTERM = 0.f;
            DZZ = FSUB(Z0(NK1), ZLCL);
        } else {
            BE = FSUB(FDIV(FADD(TVU(NK), TVU(NK1)), FADD(TV0(NK), TV0(NK1))),
                      1.f);
            BOTERM = FDIV(FMUL(FMUL(FMUL(2.f, DZA(NK)), G), BE), 1.5f);
            ENTERM = FDIV(FMUL(FMUL(2.f, UER(NK)), WTW), UPOLD);
            DZZ = DZA(NK);
        }
        {
            float ql_ = QLIQ(NK1), qi_ = QICE(NK1), qlo_ = QLQOUT(NK1),
                  qio_ = QICOUT(NK1);
            ko_condload(&ql_, &qi_, &WTW, DZZ, BOTERM, ENTERM, RATE,
                        &QNEWLQ, &QNEWIC, &qlo_, &qio_, G);
            QLIQ(NK1) = ql_; QICE(NK1) = qi_; QLQOUT(NK1) = qlo_;
            QICOUT(NK1) = qio_;
        }
        if (WTW <= 0.f) goto L65;
        WABS = KO_SQRT(fabsf(WTW));
        /* WU(NK1) = WTW/WABS is never read (:804) */
        THTES(NK1) = FMUL(FMUL(T0(NK1),
                               KO_POW(FDIV(1.E5f, P0(NK1)),
                                      FMUL(0.2854f,
                                           FSUB(1.f, FMUL(0.28f, QES(NK1)))))),
                          KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, T0(NK1)),
                                                2.5403f), QES(NK1)),
                                      FADD(1.f, FMUL(0.81f, QES(NK1))))));
        UDLBE = FMUL(FSUB(FDIV(FMUL(2.f, THTUDL),
                               FADD(THTES(NK), THTES(NK1))), 1.f), DZZ);
        if (UDLBE > 0.f) ABE = FADD(ABE, FMUL(UDLBE, G));
        if (FRC1 > 1.E-6f) {
            float tu_ = TU(NK1), th_ = THETEU(NK1), qu_ = QU(NK1),
                  ql_ = QLIQ(NK1), qi_ = QICE(NK1), r2_ = RATIO2(NK1);
            ko_dtfrznew(&tu_, P0(NK1), &th_, &qu_, &ql_, &qi_, &r2_, TTFRZ,
                        TBFRZ, &QNWFRZ, &RL, &FRC1, EFFQ, &IFLAG, kc);
            TU(NK1) = tu_; THETEU(NK1) = th_; QU(NK1) = qu_;
            QLIQ(NK1) = ql_; QICE(NK1) = qi_; RATIO2(NK1) = r2_;
        }
        THETEE(NK1) = ko_envirtht(P0(NK1), T0(NK1), Q0(NK1), RATIO2(NK1), RL,
                                  kc);
        REI = FDIV(FMUL(FMUL(VMFLCL, DP(NK1)), 0.03f), KO_RAD);
        TVQU(NK1) = FMUL(TU(NK1), FSUB(FSUB(FADD(1.f, FMUL(0.608f, QU(NK1))),
                                            QLIQ(NK1)), QICE(NK1)));
        if (TVQU(NK1) <= TV0(NK1)) {
            UER(NK1) = 0.0f;
            UDR(NK1) = REI;
            EE2 = 0.f;
            UD2 = 1.f;
            EQFRC(NK1) = 0.f;
            goto L55;
        }
        LET = NK1;
        TTMP = TVQU(NK1);
        F1 = 0.95f;
        F2 = FSUB(1.f, F1);
        THTTMP = FADD(FMUL(F1, THETEE(NK1)), FMUL(F2, THETEU(NK1)));
        QTMP = FADD(FMUL(F1, Q0(NK1)), FMUL(F2, QU(NK1)));
        TMPLIQ = FMUL(F2, QLIQ(NK1));
        TMPICE = FMUL(F2, QICE(NK1));
        ko_tpmix(P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ, &TMPICE, &QNEWLQ,
                 &QNEWIC, RATIO2(NK1), RL, kc);
        TU95 = FMUL(TTMP, FSUB(FSUB(FADD(1.f, FMUL(0.608f, QTMP)), TMPLIQ),
                               TMPICE));
        if (TU95 > TV0(NK1)) {
            EE2 = 1.f;
            UD2 = 0.f;
            EQFRC(NK1) = 1.0f;
            goto L50;
        }
        F1 = 0.10f;
        F2 = FSUB(1.f, F1);
        THTTMP = FADD(FMUL(F1, THETEE(NK1)), FMUL(F2, THETEU(NK1)));
        QTMP = FADD(FMUL(F1, Q0(NK1)), FMUL(F2, QU(NK1)));
        TMPLIQ = FMUL(F2, QLIQ(NK1));
        TMPICE = FMUL(F2, QICE(NK1));
        ko_tpmix(P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ, &TMPICE, &QNEWLQ,
                 &QNEWIC, RATIO2(NK1), RL, kc);
        TU10 = FMUL(TTMP, FSUB(FSUB(FADD(1.f, FMUL(0.608f, QTMP)), TMPLIQ),
                               TMPICE));
        if (TU10 == TVQU(NK1)) {
            EE2 = 1.f;
            UD2 = 0.f;
            EQFRC(NK1) = 1.0f;
            goto L50;
        }
        EQFRC(NK1) = FDIV(FMUL(FSUB(TV0(NK1), TVQU(NK1)), F1),
                          FSUB(TU10, TVQU(NK1)));
        EQFRC(NK1) = ko_max(0.0f, EQFRC(NK1));
        EQFRC(NK1) = ko_min(1.0f, EQFRC(NK1));
        if (EQFRC(NK1) == 1.f) {
            EE2 = 1.f;
            UD2 = 0.f;
            goto L50;
        } else if (EQFRC(NK1) == 0.f) {
            EE2 = 0.f;
            UD2 = 1.f;
            goto L50;
        } else {
            ko_prof5(EQFRC(NK1), &EE2, &UD2);
        }
L50:
        if (NK == K) {
            EE1 = 1.f;
            UD1 = 0.f;
        }
        UER(NK1) = FMUL(FMUL(0.5f, REI), FADD(EE1, EE2));
        UDR(NK1) = FMUL(FMUL(0.5f, REI), FADD(UD1, UD2));
L55:
        if (FSUB(UMF(NK), UDR(NK1)) < 10.f) {
            if (UDLBE > 0.f) ABE = FSUB(ABE, FMUL(UDLBE, G));
            LET = NK;
            goto L65;
        }
        EE1 = EE2;
        UD1 = UD2;
        UPOLD = FSUB(UMF(NK), UDR(NK1));
        UPNEW = FADD(UPOLD, UER(NK1));
        UMF(NK1) = UPNEW;
        DETLQ(NK1) = FMUL(QLIQ(NK1), UDR(NK1));
        DETIC(NK1) = FMUL(QICE(NK1), UDR(NK1));
        QDT(NK1) = QU(NK1);
        QU(NK1) = FDIV(FADD(FMUL(UPOLD, QU(NK1)), FMUL(UER(NK1), Q0(NK1))),
                       UPNEW);
        THETEU(NK1) = FDIV(FADD(FMUL(THETEU(NK1), UPOLD),
                                FMUL(THETEE(NK1), UER(NK1))), UPNEW);
        QLIQ(NK1) = FDIV(FMUL(QLIQ(NK1), UPOLD), UPNEW);
        QICE(NK1) = FDIV(FMUL(QICE(NK1), UPOLD), UPNEW);
        /* KFRZ (:952) is never read */
        PPTLIQ(NK1) = FMUL(QLQOUT(NK1), FSUB(UMF(NK), UDR(NK1)));
        PPTICE(NK1) = FMUL(QICOUT(NK1), FSUB(UMF(NK), UDR(NK1)));
        TRPPT = FADD(FADD(TRPPT, PPTLIQ(NK1)), PPTICE(NK1));
        if (NK1 <= KPBL)
            UER(NK1) = FADD(UER(NK1), FDIV(FMUL(VMFLCL, DP(NK1)), DPTHMX));
    }
    /* normal termination of DO 60 leaves NK = KL */
L65:
    LTOP = NK;
    CLDHGT = FSUB(Z0(LTOP), ZLCL);
    if (CLDHGT < 3.E3f || ABE < 1.f) {
        for (NK = K; NK <= LTOP; ++NK) {                       /* DO 70 */
            UMF(NK) = 0.f;
            UDR(NK) = 0.f;
            UER(NK) = 0.f;
            DETLQ(NK) = 0.f;
            DETIC(NK) = 0.f;
            PPTLIQ(NK) = 0.f;
            PPTICE(NK) = 0.f;
        }
        goto L25;
    }
    if (LET == LTOP) {
        UDR(LTOP) = FSUB(FADD(UMF(LTOP), UDR(LTOP)), UER(LTOP));
        DETLQ(LTOP) = FDIV(FMUL(FMUL(QLIQ(LTOP), UDR(LTOP)), UPNEW), UPOLD);
        DETIC(LTOP) = FDIV(FMUL(FMUL(QICE(LTOP), UDR(LTOP)), UPNEW), UPOLD);
        TRPPT = FSUB(TRPPT, FADD(PPTLIQ(LTOP), PPTICE(LTOP)));
        UER(LTOP) = 0.f;
        UMF(LTOP) = 0.f;
        goto L85;
    }
    DPTT = 0.f;
    for (NJ = LET + 1; NJ <= LTOP; ++NJ) DPTT = FADD(DPTT, DP(NJ));
    DUMFDP = FDIV(UMF(LET), DPTT);
    for (NK = LET + 1; NK <= LTOP; ++NK) {                     /* DO 75 */
        UDR(NK) = FMUL(DP(NK), DUMFDP);
        UMF(NK) = FSUB(UMF(NK - 1), UDR(NK));
        DETLQ(NK) = FMUL(QLIQ(NK), UDR(NK));
        DETIC(NK) = FMUL(QICE(NK), UDR(NK));
        TRPPT = FSUB(FSUB(TRPPT, PPTLIQ(NK)), PPTICE(NK));
        PPTLIQ(NK) = FMUL(FSUB(UMF(NK - 1), UDR(NK)), QLQOUT(NK));
        PPTICE(NK) = FMUL(FSUB(UMF(NK - 1), UDR(NK)), QICOUT(NK));
        TRPPT = FADD(FADD(TRPPT, PPTLIQ(NK)), PPTICE(NK));
    }
L85:
    for (NK = 1; NK <= K; ++NK) {                              /* DO 90 */
        if (NK >= LC) {
            if (NK == LC) {
                UMF(NK) = FDIV(FMUL(VMFLCL, DP(NK)), DPTHMX);
                UER(NK) = FDIV(FMUL(VMFLCL, DP(NK)), DPTHMX);
            } else if (NK <= KPBL) {
                UER(NK) = FDIV(FMUL(VMFLCL, DP(NK)), DPTHMX);
                UMF(NK) = FADD(UMF(NK - 1), UER(NK));
            } else {
                UMF(NK) = VMFLCL;
                UER(NK) = 0.f;
            }
            TU(NK) = FADD(TMIX, FMUL(FSUB(Z0(NK), ZMIX), GDRY));
            QU(NK) = QMIX;
        } else {
            TU(NK) = 0.f;
            QU(NK) = 0.f;
            UMF(NK) = 0.f;
            UER(NK) = 0.f;
        }
        UDR(NK) = 0.f;
        QDT(NK) = 0.f;
        QLIQ(NK) = 0.f;
        QICE(NK) = 0.f;
        QLQOUT(NK) = 0.f;
        QICOUT(NK) = 0.f;
        PPTLIQ(NK) = 0.f;
        PPTICE(NK) = 0.f;
        DETLQ(NK) = 0.f;
        DETIC(NK) = 0.f;
        RATIO2(NK) = 0.f;
        EE = FDIV(FMUL(Q0(NK), P0(NK)), FADD(EP2, Q0(NK)));
        TLOG = KO_LOG(FDIV(EE, ALIQ));
        TDPT = FDIV(FSUB(CLIQ, FMUL(DLIQ, TLOG)), FSUB(BLIQ, TLOG));
        TSAT = FSUB(TDPT, FMUL(FSUB(FADD(.212f, FMUL(1.571E-3f,
                                                     FSUB(TDPT, T00))),
                                    FMUL(4.36E-4f, FSUB(T0(NK), T00))),
                               FSUB(T0(NK), TDPT)));
        THTA = FMUL(T0(NK), KO_POW(FDIV(1.E5f, P0(NK)),
                                   FMUL(0.2854f,
                                        FSUB(1.f, FMUL(0.28f, Q0(NK))))));
        THETEE(NK) = FMUL(THTA,
                          KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TSAT),
                                                2.5403f), Q0(NK)),
                                      FADD(1.f, FMUL(0.81f, Q0(NK))))));
        THTES(NK) = FMUL(THTA,
                         KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, T0(NK)),
                                               2.5403f), QES(NK)),
                                     FADD(1.f, FMUL(0.81f, QES(NK))))));
        EQFRC(NK) = 1.0f;
    }
    LTOP1 = LTOP + 1;
    LTOPM1 = LTOP - 1;
    for (NK = LTOP1; NK <= KX; ++NK) {                         /* DO 95 */
        UMF(NK) = 0.f;
        UDR(NK) = 0.f;
        UER(NK) = 0.f;
        QDT(NK) = 0.f;
        QLIQ(NK) = 0.f;
        QICE(NK) = 0.f;
        QLQOUT(NK) = 0.f;
        QICOUT(NK) = 0.f;
        DETLQ(NK) = 0.f;
        DETIC(NK) = 0.f;
        PPTLIQ(NK) = 0.f;
        PPTICE(NK) = 0.f;
        if (NK > LTOP1) {
            TU(NK) = 0.f;
            QU(NK) = 0.f;
        }
        THTA0(NK) = 0.f;
        THTAU(NK) = 0.f;
        EMS(NK) = FDIV(FMUL(DP(NK), DXSQ), G);
        EMSD(NK) = FDIV(1.f, EMS(NK));
        TG(NK) = T0(NK);
        QG(NK) = Q0(NK);
        QLG(NK) = 0.f;
        QIG(NK) = 0.f;
        QRG(NK) = 0.f;
        QSG(NK) = 0.f;
        OMG(NK) = 0.f;
    }
    OMG(KL + 1) = 0.f;
    P150 = FSUB(P0(KLCL), 1.50E4f);
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 100 */
        THTAD(NK) = 0.f;
        EMS(NK) = FDIV(FMUL(DP(NK), DXSQ), G);
        EMSD(NK) = FDIV(1.f, EMS(NK));
        EXN(NK) = KO_POW(FDIV(P00, P0(NK)),
                         FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QDT(NK)))));
        THTAU(NK) = FMUL(TU(NK), EXN(NK));
        EXN(NK) = KO_POW(FDIV(P00, P0(NK)),
                         FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, Q0(NK)))));
        THTA0(NK) = FMUL(T0(NK), EXN(NK));
        if (P0(NK) > P150) LVF = NK;
        OMG(NK) = 0.f;
    }
    LVF = min(LVF, LET);
    USR = FMUL(UMF(LVF + 1), FADD(FADD(QU(LVF + 1), QLIQ(LVF + 1)),
                                  QICE(LVF + 1)));
    USR = ko_min(USR, TRPPT);
    if (USR < 1.E-8f) USR = TRPPT;
    /* convective time scale (:1144-1158) */
    WSPD_KLCL = KO_SQRT(FADD(FMUL(U0(KLCL), U0(KLCL)),
                             FMUL(V0(KLCL), V0(KLCL))));
    WSPD_L5 = KO_SQRT(FADD(FMUL(U0(L5), U0(L5)), FMUL(V0(L5), V0(L5))));
    WSPD_LTOP = KO_SQRT(FADD(FMUL(U0(LTOP), U0(LTOP)),
                             FMUL(V0(LTOP), V0(LTOP))));
    /* WSPD is one array in the Fortran: a later store to a shared index
     * overwrites an earlier one, so read back in that light.  All three
     * entries are the same formula of the same inputs, so an aliased
     * index holds the same word either way. */
    VCONV = FMUL(.5f, FADD(WSPD_KLCL, WSPD_L5));
    if (VCONV > 0.f) {
        TIMEC = FDIV(DX, VCONV);
    } else {
        TIMEC = 3600.f;
    }
    TADVEC = TIMEC;
    TIMEC = ko_max(1800.f, TIMEC);
    TIMEC = ko_min(3600.f, TIMEC);
    NIC = ko_nint(FDIV(TIMEC, DT));
    TIMEC = FMUL((float)NIC, DT);
    if (WSPD_LTOP > WSPD_KLCL) {
        SHSIGN = 1.f;
    } else {
        SHSIGN = -1.f;
    }
    VWS = FADD(FMUL(FSUB(U0(LTOP), U0(KLCL)), FSUB(U0(LTOP), U0(KLCL))),
               FMUL(FSUB(V0(LTOP), V0(KLCL)), FSUB(V0(LTOP), V0(KLCL))));
    VWS = FDIV(FMUL(FMUL(1.E3f, SHSIGN), KO_SQRT(VWS)),
               FSUB(Z0(LTOP), Z0(LCL)));
    PEF = FADD(1.591f, FMUL(VWS, FADD(-.639f,
                                      FMUL(VWS, FSUB(9.53E-2f,
                                                     FMUL(VWS, 4.96E-3f))))));
    PEF = ko_max(.2f, PEF);
    PEF = ko_min(.9f, PEF);
    CBH = FMUL(FSUB(ZLCL, Z0(1)), 3.281E-3f);
    if (CBH < 3.f) {
        RCBH = .02f;
    } else {
        RCBH = FADD(.96729352f,
               FMUL(CBH, FADD(-.70034167f,
               FMUL(CBH, FADD(.162179896f,
               FMUL(CBH, FADD(-1.2569798E-2f,
               FMUL(CBH, FSUB(4.2772E-4f, FMUL(CBH, 5.44E-6f))))))))));
    }
    if (CBH > 25.f) RCBH = 2.4f;
    PEFCBH = FDIV(1.f, FADD(1.f, RCBH));
    PEFCBH = ko_min(.9f, PEFCBH);
    PEFF = FMUL(.5f, FADD(PEF, PEFCBH));
    PEFF2 = PEFF;
    /* DOWNDRAFT (:1206-1467) */
    TDER = 0.f;
    KSTART = max(KPBL, KLCL);
    THTMIN = THTES(KSTART + 1);
    KMIN = KSTART + 1;
    for (NK = KSTART + 2; NK <= LTOP - 1; ++NK) {              /* DO 104 */
        THTMIN = ko_min(THTMIN, THTES(NK));
        if (THTMIN == THTES(NK)) KMIN = NK;
    }
    LFS = KMIN;
    if (RATIO2(LFS) > 0.f)
        THETEE(LFS) = ko_envirtht(P0(LFS), T0(LFS), Q0(LFS), 0.f, RL, kc);
    EQFRC(LFS) = FDIV(FSUB(THTES(LFS), THETEU(LFS)),
                      FSUB(THETEE(LFS), THETEU(LFS)));
    EQFRC(LFS) = ko_max(0.f, EQFRC(LFS));
    EQFRC(LFS) = ko_min(1.f, EQFRC(LFS));
    THETED(LFS) = THTES(LFS);
    if (ML > 0) {
        DTMLTD = FDIV(FMUL(FMUL(0.5f, FSUB(QU(KLCL), QU(LTOP))), RLF), CP);
    } else {
        DTMLTD = 0.f;
    }
    TZ(LFS) = FSUB(T0(LFS), DTMLTD);
    ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TZ(LFS), BLIQ), CLIQ),
                                FSUB(TZ(LFS), DLIQ))));
    QS = FDIV(FMUL(EP2, ES), FSUB(P0(LFS), ES));
    QD(LFS) = FADD(FMUL(EQFRC(LFS), Q0(LFS)),
                   FMUL(FSUB(1.f, EQFRC(LFS)), QU(LFS)));
    THTAD(LFS) = FMUL(TZ(LFS), KO_POW(FDIV(P00, P0(LFS)),
                                      FMUL(0.2854f, FSUB(1.f,
                                                         FMUL(0.28f, QD(LFS))))));
    if (QD(LFS) >= QS) {
        THETED(LFS) = FMUL(THTAD(LFS),
                           KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TZ(LFS)),
                                                 2.5403f), QS),
                                       FADD(1.f, FMUL(0.81f, QS)))));
    } else {
        THETED(LFS) = ko_envirtht(P0(LFS), TZ(LFS), QD(LFS), 0.f, RL, kc);
    }
    for (NK = 1; NK <= LFS; ++NK) {                            /* DO 107 */
        ND = LFS - NK;
        if (THETED(LFS) > THTES(ND) || ND == 1) {
            LDB = ND;
            if (NK == 1 || FSUB(P0(LDB), P0(LFS)) < 50.E2f) goto L141;
            goto L110;
        }
    }
L110:
    DPDD = DP(LDB);
    LDT = LDB;
    FRC = 1.f;
    /* :1278 label 120 */
    TVD_LFS = FMUL(T0(LFS), FADD(1.f, FMUL(0.608f, QES(LFS))));
    RDD = FDIV(P0(LFS), FMUL(R, TVD_LFS));
    A1 = FMUL(FSUB(1.f, PEFF), AU0);
    DMF(LFS) = -FMUL(A1, RDD);
    DER(LFS) = FMUL(EQFRC(LFS), DMF(LFS));
    DDR(LFS) = 0.f;
    for (ND = LFS - 1; ND >= LDB; --ND) {                      /* DO 140 */
        ND1 = ND + 1;
        if (ND <= LDT) {
            DER(ND) = 0.f;
            DDR(ND) = -FDIV(FMUL(FMUL(DMF(LDT + 1), DP(ND)), FRC), DPDD);
            DMF(ND) = FADD(DMF(ND1), DDR(ND));
            FRC = 1.f;
            THETED(ND) = THETED(ND1);
            QD(ND) = QD(ND1);
        } else {
            DER(ND) = FDIV(FMUL(FMUL(DMF(LFS), 0.03f), DP(ND)), KO_RAD);
            DDR(ND) = 0.f;
            DMF(ND) = FADD(DMF(ND1), DER(ND));
            if (RATIO2(ND) > 0.f)
                THETEE(ND) = ko_envirtht(P0(ND), T0(ND), Q0(ND), 0.f, RL, kc);
            THETED(ND) = FDIV(FADD(FMUL(THETED(ND1), DMF(ND1)),
                                   FMUL(THETEE(ND), DER(ND))), DMF(ND));
            QD(ND) = FDIV(FADD(FMUL(QD(ND1), DMF(ND1)),
                               FMUL(Q0(ND), DER(ND))), DMF(ND));
        }
    }
    TDER = 0.f;
    for (ND = LDB; ND <= LDT; ++ND) {                          /* DO 135 */
        TZ(ND) = ko_tpdd(P0(ND), THETED(LDT), T0(ND), &QS, QD(ND), 1.0f, kc);
        ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TZ(ND), BLIQ), CLIQ),
                                    FSUB(TZ(ND), DLIQ))));
        QS = FDIV(FMUL(EP2, ES), FSUB(P0(ND), ES));
        DSSDT = FDIV(FSUB(CLIQ, FMUL(BLIQ, DLIQ)),
                     FMUL(FSUB(TZ(ND), DLIQ), FSUB(TZ(ND), DLIQ)));
        RL = FSUB(XLV0, FMUL(XLV1, TZ(ND)));
        DTMP = FDIV(FMUL(FMUL(RL, QS), FSUB(1.f, RHBC)),
                    FADD(CP, FMUL(FMUL(FMUL(RL, RHBC), QS), DSSDT)));
        T1RH = FADD(TZ(ND), DTMP);
        ES = FMUL(FMUL(RHBC, ALIQ), KO_EXP(FDIV(FSUB(FMUL(BLIQ, T1RH), CLIQ),
                                               FSUB(T1RH, DLIQ))));
        QSRH = FDIV(FMUL(EP2, ES), FSUB(P0(ND), ES));
        if (QSRH < QD(ND)) {
            QSRH = QD(ND);
            T1RH = TZ(ND);
        }
        TZ(ND) = T1RH;
        QS = QSRH;
        TDER = FADD(TDER, FMUL(FSUB(QS, QD(ND)), DDR(ND)));
        QD(ND) = QS;
        THTAD(ND) = FMUL(TZ(ND), KO_POW(FDIV(P00, P0(ND)),
                                        FMUL(0.2854f, FSUB(1.f,
                                                           FMUL(0.28f, QD(ND))))));
    }
L141:
    if (TDER < 1.f) {
        PPTFLX = TRPPT;
        CPR = TRPPT;
        TDER = 0.f;
        CNDTNF = 0.f;
        UPDINC = 1.f;
        LDB = LFS;
        for (NDK = 1; NDK <= LTOP; ++NDK) {                    /* DO 117 */
            DMF(NDK) = 0.f;
            DER(NDK) = 0.f;
            DDR(NDK) = 0.f;
            THTAD(NDK) = 0.f;
            TZ(NDK) = 0.f;
            QD(NDK) = 0.f;
        }
        /* AINCM2 (:1358) is never read */
        goto L165;
    }
    DEVDMF = FDIV(TDER, DMF(LFS));
    PPR = 0.f;
    PPTFLX = FMUL(PEFF, USR);
    RCED = FSUB(TRPPT, PPTFLX);
    for (NM = KLCL; NM <= LFS; ++NM)                           /* DO 132 */
        PPR = FADD(FADD(PPR, PPTLIQ(NM)), PPTICE(NM));
    if (LFS >= KLCL) {
        DPPTDF = FDIV(FMUL(FMUL(FSUB(1.f, PEFF), PPR),
                           FSUB(1.f, EQFRC(LFS))), UMF(LFS));
    } else {
        DPPTDF = 0.f;
    }
    CNDTNF = FMUL(FADD(QLIQ(LFS), QICE(LFS)), FSUB(1.f, EQFRC(LFS)));
    DMFLFS = FDIV(RCED, FADD(FADD(DEVDMF, DPPTDF), CNDTNF));
    if (DMFLFS > 0.f) {
        TDER = 0.f;
        goto L141;
    }
    if (LFS >= KLCL) {
        UPDINC = FDIV(FSUB(UMF(LFS), FMUL(FSUB(1.f, EQFRC(LFS)), DMFLFS)),
                      UMF(LFS));
        if (UPDINC > 1.5f) {
            UPDINC = 1.5f;
            DMFLFS2 = FDIV(FMUL(UMF(LFS), FSUB(UPDINC, 1.f)),
                           FSUB(EQFRC(LFS), 1.f));
            RCED2 = FMUL(DMFLFS2, FADD(FADD(DEVDMF, DPPTDF), CNDTNF));
            PPTFLX = FADD(PPTFLX, FSUB(RCED, RCED2));
            PEFF2 = FDIV(PPTFLX, USR);
            RCED = RCED2;
            DMFLFS = DMFLFS2;
        }
    } else {
        UPDINC = 1.f;
    }
    DDINC = FDIV(DMFLFS, DMF(LFS));
    for (NK = LDB; NK <= LFS; ++NK) {                          /* DO 149 */
        DMF(NK) = FMUL(DMF(NK), DDINC);
        DER(NK) = FMUL(DER(NK), DDINC);
        DDR(NK) = FMUL(DDR(NK), DDINC);
    }
    CPR = FADD(TRPPT, FMUL(PPR, FSUB(UPDINC, 1.f)));
    PPTFLX = FADD(PPTFLX, FMUL(FMUL(PEFF, PPR), FSUB(UPDINC, 1.f)));
    PEFF = PEFF2;
    TDER = FMUL(TDER, DDINC);
    for (NK = LC; NK <= LFS; ++NK) {                           /* DO 155 */
        UMF(NK) = FMUL(UMF(NK), UPDINC);
        UDR(NK) = FMUL(UDR(NK), UPDINC);
        UER(NK) = FMUL(UER(NK), UPDINC);
        PPTLIQ(NK) = FMUL(PPTLIQ(NK), UPDINC);
        PPTICE(NK) = FMUL(PPTICE(NK), UPDINC);
        DETLQ(NK) = FMUL(DETLQ(NK), UPDINC);
        DETIC(NK) = FMUL(DETIC(NK), UPDINC);
    }
    if (LDB > 1) {
        for (NK = 1; NK <= LDB - 1; ++NK) {                    /* DO 156 */
            DMF(NK) = 0.f;
            DER(NK) = 0.f;
            DDR(NK) = 0.f;
            TZ(NK) = 0.f;
            QD(NK) = 0.f;
            THTAD(NK) = 0.f;
        }
    }
    for (NK = LFS + 1; NK <= KX; ++NK) {                       /* DO 157 */
        DMF(NK) = 0.f;
        DER(NK) = 0.f;
        DDR(NK) = 0.f;
        TZ(NK) = 0.f;
        QD(NK) = 0.f;
        THTAD(NK) = 0.f;
    }
    for (NK = LDT + 1; NK <= LFS - 1; ++NK) {                  /* DO 158 */
        TZ(NK) = 0.f;
        QD(NK) = 0.f;
    }
L165:
    AINCMX = 1000.f;
    LMAX = max(KLCL, LFS);
    for (NK = LC; NK <= LMAX; ++NK) {                          /* DO 166 */
        if (FSUB(UER(NK), DER(NK)) > 0.f)
            AINCM1 = FDIV(EMS(NK), FMUL(FSUB(UER(NK), DER(NK)), TIMEC));
        AINCMX = ko_min(AINCMX, AINCM1);
    }
    AINC = 1.f;
    if (AINCMX < AINC) AINC = AINCMX;
    NCOUNT = 0;
    TDER2 = TDER;
    PPTFL2 = PPTFLX;
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 170 */
        DETLQ2(NK) = DETLQ(NK);
        DETIC2(NK) = DETIC(NK);
        UDR2(NK) = UDR(NK);
        UER2(NK) = UER(NK);
        DDR2(NK) = DDR(NK);
        DER2(NK) = DER(NK);
        UMF2(NK) = UMF(NK);
        DMF2(NK) = DMF(NK);
    }
    FABE = 1.f;
    STAB = 0.95f;
    NOITR = 0;
    if (FDIV(AINC, AINCMX) > 0.999f) {
        NCOUNT = 0;
        goto L255;
    }
    ISTOP = 0;
L175:
    NCOUNT = NCOUNT + 1;
    DTT = TIMEC;
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 200 */
        DOMGDP(NK) = -FMUL(FSUB(FSUB(FSUB(UER(NK), DER(NK)), UDR(NK)),
                                DDR(NK)), EMSD(NK));
        if (NK > 1) {
            OMG(NK) = FSUB(OMG(NK - 1), FMUL(DP(NK - 1), DOMGDP(NK - 1)));
            DTT1 = FDIV(FMUL(0.75f, DP(NK - 1)),
                        FADD(fabsf(OMG(NK)), 1.E-10f));
            DTT = ko_min(DTT, DTT1);
        }
    }
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 488 */
        THPA(NK) = THTA0(NK);
        QPA(NK) = Q0(NK);
        NSTEP = ko_nint(FADD(FDIV(TIMEC, DTT), 1.f));
        DTIME = FDIV(TIMEC, (float)NSTEP);
        FXM(NK) = FDIV(FMUL(OMG(NK), DXSQ), G);
    }
    for (NTC = 1; NTC <= NSTEP; ++NTC) {                       /* DO 495 */
        for (NK = 1; NK <= LTOP; ++NK) {
            THFXTOP(NK) = 0.f;
            THFXBOT(NK) = 0.f;
            QFXTOP(NK) = 0.f;
            QFXBOT(NK) = 0.f;
        }
        for (NK = 2; NK <= LTOP; ++NK) {                       /* DO 494 */
            if (OMG(NK) <= 0.f) {
                THFXBOT(NK) = -FMUL(FXM(NK), THPA(NK - 1));
                QFXBOT(NK) = -FMUL(FXM(NK), QPA(NK - 1));
                THFXTOP(NK - 1) = FSUB(THFXTOP(NK - 1), THFXBOT(NK));
                QFXTOP(NK - 1) = FSUB(QFXTOP(NK - 1), QFXBOT(NK));
            } else {
                THFXBOT(NK) = -FMUL(FXM(NK), THPA(NK));
                QFXBOT(NK) = -FMUL(FXM(NK), QPA(NK));
                THFXTOP(NK - 1) = FSUB(THFXTOP(NK - 1), THFXBOT(NK));
                QFXTOP(NK - 1) = FSUB(QFXTOP(NK - 1), QFXBOT(NK));
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {                       /* DO 492 */
            THPA(NK) = FADD(THPA(NK),
                FMUL(FMUL(FSUB(FADD(FADD(FADD(THFXBOT(NK),
                                              FMUL(UDR(NK), THTAU(NK))),
                                         FMUL(DDR(NK), THTAD(NK))),
                                    THFXTOP(NK)),
                               FMUL(FSUB(UER(NK), DER(NK)), THTA0(NK))),
                          DTIME), EMSD(NK)));
            QPA(NK) = FADD(QPA(NK),
                FMUL(FMUL(FSUB(FADD(FADD(FADD(QFXBOT(NK),
                                              FMUL(UDR(NK), QDT(NK))),
                                         FMUL(DDR(NK), QD(NK))),
                                    QFXTOP(NK)),
                               FMUL(FSUB(UER(NK), DER(NK)), Q0(NK))),
                          DTIME), EMSD(NK)));
        }
    }
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 498 */
        /* THTAG(NK) = THPA(NK): THTAG is read only by DO 230 below */
        QG(NK) = QPA(NK);
    }
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 499 */
        if (QG(NK) < 0.f) {
            if (NK == 1) {
                *FATAL = 1;          /* :1586 qg = 0 at the surface */
                return;
            }
            NK1 = NK + 1;
            if (NK == LTOP) NK1 = KLCL;
            TMA = FMUL(QG(NK1), EMS(NK1));
            TMB = FMUL(QG(NK - 1), EMS(NK - 1));
            TMM = FMUL(FSUB(QG(NK), 1.E-9f), EMS(NK));
            BCOEFF = -FDIV(TMM, FADD(FDIV(FMUL(TMA, TMA), TMB), TMB));
            ACOEFF = FDIV(FMUL(BCOEFF, TMA), TMB);
            TMB = FMUL(TMB, FSUB(1.f, BCOEFF));
            TMA = FMUL(TMA, FSUB(1.f, ACOEFF));
            /* :1597-1605 QVDIFF is a printed warning only */
            QG(NK) = 1.E-9f;
            QG(NK1) = FMUL(TMA, EMSD(NK1));
            QG(NK - 1) = FMUL(TMB, EMSD(NK - 1));
        }
    }
    TOPOMG = FMUL(FMUL(FSUB(UDR(LTOP), UER(LTOP)), DP(LTOP)), EMSD(LTOP));
    if (fabsf(FSUB(TOPOMG, OMG(LTOP))) > 1.E-3f) {
        ISTOP = 1;
        goto L265;
    }
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 230 */
        EXN(NK) = KO_POW(FDIV(P00, P0(NK)),
                         FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QG(NK)))));
        TG(NK) = FDIV(THPA(NK), EXN(NK));
        TVG(NK) = FMUL(TG(NK), FADD(1.f, FMUL(0.608f, QG(NK))));
    }
    THMIX = 0.f;
    QMIX = 0.f;
    PMIX = 0.f;
    for (NK = LC; NK <= KPBL; ++NK) {                          /* DO 217 */
        ROCPQ = FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QG(NK))));
        THMIX = FADD(THMIX, FMUL(FMUL(DP(NK), TG(NK)),
                                 KO_POW(FDIV(P00, P0(NK)), ROCPQ)));
        QMIX = FADD(QMIX, FMUL(DP(NK), QG(NK)));
        PMIX = FADD(PMIX, FMUL(DP(NK), P0(NK)));
    }
    THMIX = FDIV(THMIX, DPTHMX);
    QMIX = FDIV(QMIX, DPTHMX);
    PMIX = FDIV(PMIX, DPTHMX);
    ROCPQ = FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QMIX)));
    TMIX = FMUL(THMIX, KO_POW(FDIV(PMIX, P00), ROCPQ));
    ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TMIX, BLIQ), CLIQ),
                                FSUB(TMIX, DLIQ))));
    QS = FDIV(FMUL(EP2, ES), FSUB(PMIX, ES));
    if (QMIX > QS) {
        RL = FSUB(XLV0, FMUL(XLV1, TMIX));
        CPM = FMUL(CP, FADD(1.f, FMUL(0.887f, QMIX)));
        DSSDT = FDIV(FMUL(QS, FSUB(CLIQ, FMUL(BLIQ, DLIQ))),
                     FMUL(FSUB(TMIX, DLIQ), FSUB(TMIX, DLIQ)));
        DQ = FDIV(FSUB(QMIX, QS), FADD(1.f, FDIV(FMUL(RL, DSSDT), CPM)));
        TMIX = FADD(TMIX, FMUL(FDIV(RL, CP), DQ));
        QMIX = FSUB(QMIX, DQ);
        ROCPQ = FMUL(0.2854f, FSUB(1.f, FMUL(0.28f, QMIX)));
        THMIX = FMUL(TMIX, KO_POW(FDIV(P00, PMIX), ROCPQ));
        TLCL = TMIX;
        PLCL = PMIX;
    } else {
        QMIX = ko_max(0.f, QMIX);
        EMIX = FDIV(FMUL(QMIX, PMIX), FADD(EP2, QMIX));
        TLOG = KO_LOG(FDIV(EMIX, ALIQ));
        TDPT = FDIV(FSUB(CLIQ, FMUL(DLIQ, TLOG)), FSUB(BLIQ, TLOG));
        TLCL = FSUB(TDPT, FMUL(FSUB(FADD(.212f, FMUL(1.571E-3f,
                                                     FSUB(TDPT, T00))),
                                    FMUL(4.36E-4f, FSUB(TMIX, T00))),
                               FSUB(TMIX, TDPT)));
        TLCL = ko_min(TLCL, TMIX);
        CPORQ = FDIV(1.f, ROCPQ);
        PLCL = FMUL(P00, KO_POW(FDIV(TLCL, THMIX), CPORQ));
    }
    TVLCL = FMUL(TLCL, FADD(1.f, FMUL(0.608f, QMIX)));
    for (NK = LC; NK <= KL; ++NK) {                            /* DO 235 */
        KLCL = NK;
        if (PLCL >= P0(NK)) break;      /* GOTO 240; else falls into 240 */
    }
    K = KLCL - 1;                                              /* 240 */
    DLP = FDIV(KO_LOG(FDIV(PLCL, P0(K))), KO_LOG(FDIV(P0(KLCL), P0(K))));
    TENV = FADD(TG(K), FMUL(FSUB(TG(KLCL), TG(K)), DLP));
    QENV = FADD(QG(K), FMUL(FSUB(QG(KLCL), QG(K)), DLP));
    TVEN = FMUL(TENV, FADD(1.f, FMUL(0.608f, QENV)));
    /* TVBAR (:1691) is never read */
    ZLCL = FADD(Z0(K), FMUL(FSUB(Z0(KLCL), Z0(K)), DLP));
    TVAVG = FMUL(0.5f, FADD(TVEN, FMUL(TG(KLCL),
                                       FADD(1.f, FMUL(0.608f, QG(KLCL))))));
    PLCL = FMUL(P0(KLCL), KO_EXP(FMUL(FDIV(G, FMUL(R, TVAVG)),
                                      FSUB(Z0(KLCL), ZLCL))));
    THETEU(K) = FMUL(FMUL(TMIX, KO_POW(FDIV(1.E5f, PMIX),
                                       FMUL(0.2854f, FSUB(1.f,
                                                          FMUL(0.28f, QMIX))))),
                     KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TLCL), 2.5403f),
                                      QMIX),
                                 FADD(1.f, FMUL(0.81f, QMIX)))));
    ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TENV, BLIQ), CLIQ),
                                FSUB(TENV, DLIQ))));
    QESE = FDIV(FMUL(EP2, ES), FSUB(PLCL, ES));
    THTESG(K) = FMUL(FMUL(TENV, KO_POW(FDIV(1.E5f, PLCL),
                                       FMUL(0.2854f, FSUB(1.f,
                                                          FMUL(0.28f, QESE))))),
                     KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TENV), 2.5403f),
                                      QESE),
                                 FADD(1.f, FMUL(0.81f, QESE)))));
    ABEG = 0.f;
    THTUDL = THETEU(K);
    for (NK = K; NK <= LTOPM1; ++NK) {                         /* DO 245 */
        NK1 = NK + 1;
        ES = FMUL(ALIQ, KO_EXP(FDIV(FSUB(FMUL(TG(NK1), BLIQ), CLIQ),
                                    FSUB(TG(NK1), DLIQ))));
        QESE = FDIV(FMUL(EP2, ES), FSUB(P0(NK1), ES));
        THTESG(NK1) = FMUL(FMUL(TG(NK1),
                                KO_POW(FDIV(1.E5f, P0(NK1)),
                                       FMUL(0.2854f, FSUB(1.f,
                                                          FMUL(0.28f, QESE))))),
                           KO_EXP(FMUL(FMUL(FSUB(FDIV(3374.6525f, TG(NK1)),
                                                 2.5403f), QESE),
                                       FADD(1.f, FMUL(0.81f, QESE)))));
        if (NK == K) {
            DZZ = FSUB(Z0(KLCL), ZLCL);
        } else {
            DZZ = DZA(NK);
        }
        BE = FMUL(FSUB(FDIV(FMUL(2.f, THTUDL), FADD(THTESG(NK1), THTESG(NK))),
                       1.f), DZZ);
        if (BE > 0.f) ABEG = FADD(ABEG, FMUL(BE, G));
    }
    if (NOITR == 1) goto L265;
    DABE = ko_max(FSUB(ABE, ABEG), FMUL(0.1f, ABE));
    FABE = FDIV(ABEG, FADD(ABE, 1.E-8f));
    if (FABE > 1.f) goto L325;
    if (NCOUNT != 1) {
        DFDA = FDIV(FSUB(FABE, FABEOLD), FSUB(AINC, AINCOLD));
        if (DFDA > 0.f) {
            NOITR = 1;
            AINC = AINCOLD;
            goto L255;
        }
    }
    AINCOLD = AINC;
    FABEOLD = FABE;
    if (FDIV(AINC, AINCMX) > 0.999f && FABE > FSUB(1.05f, STAB)) goto L265;
    if (FABE <= FSUB(1.05f, STAB) && FABE >= FSUB(0.95f, STAB)) goto L265;
    if (NCOUNT > 10) goto L265;
    if (FABE == 0.f) {
        AINC = FMUL(AINC, 0.5f);
    } else {
        AINC = FDIV(FMUL(FMUL(AINC, STAB), ABE), FADD(DABE, 1.E-8f));
    }
L255:
    AINC = ko_min(AINCMX, AINC);
    if (AINC < 0.05f) goto L325;
    TDER = FMUL(TDER2, AINC);
    PPTFLX = FMUL(PPTFL2, AINC);
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 260 */
        UMF(NK) = FMUL(UMF2(NK), AINC);
        DMF(NK) = FMUL(DMF2(NK), AINC);
        DETLQ(NK) = FMUL(DETLQ2(NK), AINC);
        DETIC(NK) = FMUL(DETIC2(NK), AINC);
        UDR(NK) = FMUL(UDR2(NK), AINC);
        UER(NK) = FMUL(UER2(NK), AINC);
        DER(NK) = FMUL(DER2(NK), AINC);
        DDR(NK) = FMUL(DDR2(NK), AINC);
    }
    goto L175;
L265:
    if (KF_EDRATES == 1) {
        for (NK = 1; NK <= LTOP; ++NK) {
            KO_IN(udr_kf, NK) = UDR(NK);
            KO_IN(ddr_kf, NK) = DDR(NK);
            KO_IN(uer_kf, NK) = UER(NK);
            KO_IN(der_kf, NK) = DER(NK);
        }
    }
    if (ISTOP == 1) {
        /* :1884-1931 print the column and stop the model */
        *FATAL = 2;
        return;
    }
    FRC2 = FDIV(PPTFLX, FMUL(CPR, AINC));
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 270 */
        QLPA(NK) = QL0(NK);
        QIPA(NK) = QI0(NK);
        QRPA(NK) = QR0(NK);
        QSPA(NK) = QS0(NK);
        RAINFB(NK) = FMUL(FMUL(FMUL(PPTLIQ(NK), AINC), FBFRC), FRC2);
        SNOWFB(NK) = FMUL(FMUL(FMUL(PPTICE(NK), AINC), FBFRC), FRC2);
    }
    for (NTC = 1; NTC <= NSTEP; ++NTC) {                       /* DO 290 */
        for (NK = 1; NK <= LTOP; ++NK) {                       /* DO 275 */
            QLFXIN(NK) = 0.f;
            QLFXOUT(NK) = 0.f;
            QIFXIN(NK) = 0.f;
            QIFXOUT(NK) = 0.f;
            QRFXIN(NK) = 0.f;
            QRFXOUT(NK) = 0.f;
            QSFXIN(NK) = 0.f;
            QSFXOUT(NK) = 0.f;
        }
        for (NK = 2; NK <= LTOP; ++NK) {                       /* DO 280 */
            if (OMG(NK) <= 0.f) {
                QLFXIN(NK) = -FMUL(FXM(NK), QLPA(NK - 1));
                QIFXIN(NK) = -FMUL(FXM(NK), QIPA(NK - 1));
                QRFXIN(NK) = -FMUL(FXM(NK), QRPA(NK - 1));
                QSFXIN(NK) = -FMUL(FXM(NK), QSPA(NK - 1));
                QLFXOUT(NK - 1) = FADD(QLFXOUT(NK - 1), QLFXIN(NK));
                QIFXOUT(NK - 1) = FADD(QIFXOUT(NK - 1), QIFXIN(NK));
                QRFXOUT(NK - 1) = FADD(QRFXOUT(NK - 1), QRFXIN(NK));
                QSFXOUT(NK - 1) = FADD(QSFXOUT(NK - 1), QSFXIN(NK));
            } else {
                QLFXOUT(NK) = FMUL(FXM(NK), QLPA(NK));
                QIFXOUT(NK) = FMUL(FXM(NK), QIPA(NK));
                QRFXOUT(NK) = FMUL(FXM(NK), QRPA(NK));
                QSFXOUT(NK) = FMUL(FXM(NK), QSPA(NK));
                QLFXIN(NK - 1) = FADD(QLFXIN(NK - 1), QLFXOUT(NK));
                QIFXIN(NK - 1) = FADD(QIFXIN(NK - 1), QIFXOUT(NK));
                QRFXIN(NK - 1) = FADD(QRFXIN(NK - 1), QRFXOUT(NK));
                QSFXIN(NK - 1) = FADD(QSFXIN(NK - 1), QSFXOUT(NK));
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {                       /* DO 285 */
            QLPA(NK) = FADD(QLPA(NK),
                FMUL(FMUL(FSUB(FADD(QLFXIN(NK), DETLQ(NK)), QLFXOUT(NK)),
                          DTIME), EMSD(NK)));
            QIPA(NK) = FADD(QIPA(NK),
                FMUL(FMUL(FSUB(FADD(QIFXIN(NK), DETIC(NK)), QIFXOUT(NK)),
                          DTIME), EMSD(NK)));
            QRPA(NK) = FADD(QRPA(NK),
                FMUL(FMUL(FADD(FSUB(FADD(QRFXIN(NK),
                                         FMUL(QLQOUT(NK), UDR(NK))),
                                    QRFXOUT(NK)), RAINFB(NK)),
                          DTIME), EMSD(NK)));
            QSPA(NK) = FADD(QSPA(NK),
                FMUL(FMUL(FADD(FSUB(FADD(QSFXIN(NK),
                                         FMUL(QICOUT(NK), UDR(NK))),
                                    QSFXOUT(NK)), SNOWFB(NK)),
                          DTIME), EMSD(NK)));
        }
    }
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 295 */
        QLG(NK) = QLPA(NK);
        QIG(NK) = QIPA(NK);
        QRG(NK) = QRPA(NK);
        QSG(NK) = QSPA(NK);
    }
    if (KF_EDRATES == 1) *TIMEC_KF = TIMEC;
    /* CNDTNF (:1933) is never read again */
    QINIT = 0.f;
    QFNL = 0.f;
    for (NK = 1; NK <= LTOP; ++NK) {                           /* DO 315 */
        QINIT = FADD(QINIT, FMUL(Q0(NK), EMS(NK)));
        QFNL = FADD(QFNL, FMUL(QG(NK), EMS(NK)));
        QFNL = FADD(QFNL, FMUL(FADD(FADD(FADD(QLG(NK), QIG(NK)), QRG(NK)),
                                    QSG(NK)), EMS(NK)));
    }
    QFNL = FADD(QFNL, FMUL(FMUL(PPTFLX, TIMEC), FSUB(1.f, FBFRC)));
    ERR2 = FDIV(FMUL(FSUB(QFNL, QINIT), 100.f), QINIT);
    if (fabsf(ERR2) > 0.05f) {
        *FATAL = 3;                  /* :1951 QVERR */
        return;
    }
    if (TADVEC < TIMEC) NIC = ko_nint(FDIV(TADVEC, DT));
    *NCA = FMUL((float)NIC, DT);
    for (int Kk = 1; Kk <= KX; ++Kk) {                         /* DO 320 */
        if (!qi_flag && warm_rain) {
            CPM = FMUL(CP, FADD(1.f, FMUL(0.887f, QG(Kk))));
            TG(Kk) = FSUB(TG(Kk), FDIV(FMUL(FADD(QIG(Kk), QSG(Kk)), RLF), CPM));
            DQCDT(Kk) = FDIV(FSUB(FSUB(FADD(QLG(Kk), QIG(Kk)), QL0(Kk)),
                                  QI0(Kk)), TIMEC);
            DQIDT(Kk) = 0.f;
            DQRDT(Kk) = FDIV(FSUB(FSUB(FADD(QRG(Kk), QSG(Kk)), QR0(Kk)),
                                  QS0(Kk)), TIMEC);
            DQSDT(Kk) = 0.f;
        } else if (!qi_flag && !warm_rain) {
            CPM = FMUL(CP, FADD(1.f, FMUL(0.887f, QG(Kk))));
            if (Kk <= ML) {
                TG(Kk) = FSUB(TG(Kk), FDIV(FMUL(FADD(QIG(Kk), QSG(Kk)), RLF),
                                           CPM));
            } else if (Kk > ML) {
                TG(Kk) = FADD(TG(Kk), FDIV(FMUL(FADD(QLG(Kk), QRG(Kk)), RLF),
                                           CPM));
            }
            DQCDT(Kk) = FDIV(FSUB(FSUB(FADD(QLG(Kk), QIG(Kk)), QL0(Kk)),
                                  QI0(Kk)), TIMEC);
            DQIDT(Kk) = 0.f;
            DQRDT(Kk) = FDIV(FSUB(FSUB(FADD(QRG(Kk), QSG(Kk)), QR0(Kk)),
                                  QS0(Kk)), TIMEC);
            DQSDT(Kk) = 0.f;
        } else {
            /* qi_flag; the Fortran's final ELSE (:2021-2022) cannot be
             * reached, the three tests above cover every flag pair */
            DQCDT(Kk) = FDIV(FSUB(QLG(Kk), QL0(Kk)), TIMEC);
            DQIDT(Kk) = FDIV(FSUB(QIG(Kk), QI0(Kk)), TIMEC);
            DQRDT(Kk) = FDIV(FSUB(QRG(Kk), QR0(Kk)), TIMEC);
            if (qs_flag) {
                DQSDT(Kk) = FDIV(FSUB(QSG(Kk), QS0(Kk)), TIMEC);
            } else {
                DQIDT(Kk) = FADD(DQIDT(Kk), FDIV(FSUB(QSG(Kk), QS0(Kk)),
                                                 TIMEC));
            }
        }
        DTDT(Kk) = FDIV(FSUB(TG(Kk), T0(Kk)), TIMEC);
        DQDT(Kk) = FDIV(FSUB(QG(Kk), Q0(Kk)), TIMEC);
    }
    *PRATEC = FDIV(FMUL(PPTFLX, FSUB(1.f, FBFRC)), DXSQ);
    *RAINCV = FMUL(DT, *PRATEC);
L325:
    return;
#undef Q0
#undef Z0
#undef TV0
#undef TU
#undef TVU
#undef QU
#undef TZ
#undef QD
#undef QES
#undef THTES
#undef TG
#undef TVG
#undef QG
#undef EMS
#undef EMSD
#undef UMF
#undef UER
#undef UDR
#undef DMF
#undef DER
#undef DDR
#undef UMF2
#undef UER2
#undef UDR2
#undef DMF2
#undef DER2
#undef DDR2
#undef DZA
#undef THTA0
#undef THETEE
#undef THTAU
#undef THETEU
#undef THTAD
#undef THETED
#undef QLIQ
#undef QICE
#undef QLQOUT
#undef QICOUT
#undef PPTLIQ
#undef PPTICE
#undef DETLQ
#undef DETIC
#undef DETLQ2
#undef DETIC2
#undef RATIO2
#undef DOMGDP
#undef EXN
#undef TVQU
#undef DP
#undef EQFRC
#undef QDT
#undef FXM
#undef THTESG
#undef THPA
#undef THFXTOP
#undef THFXBOT
#undef QPA
#undef QFXTOP
#undef QFXBOT
#undef QLPA
#undef QLFXIN
#undef QLFXOUT
#undef QIPA
#undef QIFXIN
#undef QIFXOUT
#undef QRPA
#undef QRFXIN
#undef QRFXOUT
#undef QSPA
#undef QSFXIN
#undef QSFXOUT
#undef QLG
#undef QIG
#undef QRG
#undef QSG
#undef OMG
#undef RAINFB
#undef SNOWFB
#undef U0
#undef V0
#undef T0
#undef QV0
#undef P0
#undef DZQ
#undef RHO
#undef W0AVG1D
#undef DQDT
#undef DQIDT
#undef DQCDT
#undef DQRDT
#undef DQSDT
#undef DTDT
#undef QL0
#undef QI0
#undef QR0
#undef QS0
}

/* ==========================================================================
 * KFCPS (module_cu_kf.F:13-331), one column per thread.
 * ========================================================================== */
extern "C" __global__ __launch_bounds__(256)
void kf_old_column(
        const float* __restrict__ u,
        const float* __restrict__ v,
        const float* __restrict__ w,
        const float* __restrict__ t,
        const float* __restrict__ qv,
        const float* __restrict__ pressure,
        const float* __restrict__ rho,
        const float* __restrict__ dz8w,
        const float* __restrict__ pii,
        float* __restrict__ w0avg,
        const float* __restrict__ nca_in,
        float* __restrict__ rthcuten,
        float* __restrict__ rqvcuten,
        float* __restrict__ rqccuten,
        float* __restrict__ rqrcuten,
        float* __restrict__ rqicuten,
        float* __restrict__ rqscuten,
        float* __restrict__ raincv,
        float* __restrict__ pratec,
        float* __restrict__ nca_out,
        int* __restrict__ cu_act_flag,
        int* __restrict__ fatal,
        float* __restrict__ udr_kf,
        float* __restrict__ ddr_kf,
        float* __restrict__ uer_kf,
        float* __restrict__ der_kf,
        float* __restrict__ timec_kf,
        float* __restrict__ ws,
        const float dt, const float dx, const float cudt,
        const int adapt_step_flag, const int stepcu,
        const int warm_rain, const int qi_flag, const int qs_flag,
        const int f_qr, const int kf_edrates,
        const int nz, const int ny, const int nx, const int col0)
{
    const size_t plane = (size_t)ny * (size_t)nx;
    const int kp = nz + 1;
    float* base = ws + ((size_t)blockIdx.x * (blockDim.x / KO_LANES)
                        + threadIdx.x / KO_LANES) * KO_SLOTS * (size_t)kp
                       * KO_LANES + threadIdx.x % KO_LANES;
    const long long cl = (long long)col0
                         + (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cl >= (long long)plane) return;
    const size_t col = (size_t)cl;

    KoConst kc;
    kc.g = KO_G;
    kc.r = KO_R_D;
    kc.cp = FDIV(FMUL(7.f, KO_R_D), 2.f);      /* cp = 7.*r_d/2. */
    kc.ep2 = FDIV(KO_R_D, KO_R_V);              /* EP_2 = R_d/R_v */
    kc.xlv0 = KO_XLV0;
    kc.xlv1 = KO_XLV1;
    kc.xls0 = KO_XLS0;
    kc.xls1 = KO_XLS1;
    kc.aliq = FMUL(KO_SVP1, 1000.f);            /* :479-486 */
    kc.bliq = KO_SVP2;
    kc.cliq = FMUL(KO_SVP2, KO_SVPT0);
    kc.dliq = KO_SVP3;
    kc.aice = 613.2f;
    kc.bice = 22.452f;
    kc.cice = 6133.0f;
    kc.dice = 0.61f;

    const float DXSQ = FMUL(dx, dx);
    const int NTST = stepcu;
    const float TST = (float)(NTST * 2);
    float W0AVGfctr, W0fctr, W0den;
    if (adapt_step_flag) {
        const float big = ko_max(FMUL(cudt, 60.f), dt);
        W0AVGfctr = FSUB(FMUL(2.f, big), dt);
        W0fctr = dt;
        W0den = FMUL(2.f, big);
    } else {
        W0AVGfctr = FSUB(TST, 1.f);
        W0fctr = 1.f;
        W0den = TST;
    }
    for (int k = 0; k < nz; ++k) {
        const size_t i = (size_t)k * plane + col;
        const float W0 = FMUL(0.5f, FADD(w[i], w[i + plane]));
        w0avg[i] = FDIV(FADD(FMUL(w0avg[i], W0AVGfctr), FMUL(W0, W0fctr)),
                        W0den);
    }

    /* Every output is defined for every column.  A held column keeps
     * WRF's stored tendencies in the driver (gpuwm/core/physics.py masks
     * these by the same hold test), so its zeros here are never applied. */
    for (int k = 0; k < nz; ++k) {
        const size_t i = (size_t)k * plane + col;
        rthcuten[i] = 0.f; rqvcuten[i] = 0.f; rqccuten[i] = 0.f;
        rqrcuten[i] = 0.f; rqicuten[i] = 0.f; rqscuten[i] = 0.f;
    }
    /* The kf_edrates diagnostics exist only when asked for (:248-256);
     * otherwise the launcher passes one-element placeholders. */
    if (kf_edrates == 1) {
        for (int k = 0; k < nz; ++k) {
            const size_t i = (size_t)k * plane + col;
            udr_kf[i] = 0.f; ddr_kf[i] = 0.f; uer_kf[i] = 0.f;
            der_kf[i] = 0.f;
        }
        timec_kf[col] = 0.f;
    }
    raincv[col] = 0.f;
    pratec[col] = 0.f;
    fatal[col] = 0;
    float nca = nca_in[col];
    cu_act_flag[col] = 1;
    if (nca > FMUL(0.5f, dt)) {
        cu_act_flag[col] = 0;
        nca_out[col] = nca;
        return;
    }
    float rcv = 0.f, prc = 0.f, tkf = 0.f;
    int fat = 0;
    /* DTDT, DQDT, DQCDT, DQRDT, DQIDT, DQSDT are formed in the six rate
     * arrays and folded into WRF's outputs below. */
    ko_kfpara(u, v, t, qv, pressure, dz8w, rho, w0avg, plane, col, nz,
              dt, dx, DXSQ, kc, warm_rain, qi_flag, qs_flag,
              rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten,
              &rcv, &prc, &nca, &fat, kf_edrates,
              udr_kf, ddr_kf, uer_kf, der_kf, &tkf, base, kp);
    nca_out[col] = nca;
    fatal[col] = fat;
    if (fat != 0) {
        for (int k = 0; k < nz; ++k) {
            const size_t i = (size_t)k * plane + col;
            rthcuten[i] = 0.f; rqvcuten[i] = 0.f; rqccuten[i] = 0.f;
            rqrcuten[i] = 0.f; rqicuten[i] = 0.f; rqscuten[i] = 0.f;
        }
        return;
    }
    raincv[col] = rcv;
    pratec[col] = prc;
    if (kf_edrates == 1) timec_kf[col] = tkf;
    for (int k = 0; k < nz; ++k) {                       /* :290-325 */
        const size_t i = (size_t)k * plane + col;
        rthcuten[i] = FDIV(rthcuten[i], pii[i]);
        if (!f_qr) {
            /* Eta microphysics without a 3-D rain field (:305-309) */
            rqccuten[i] = FADD(rqrcuten[i], rqccuten[i]);
            rqrcuten[i] = 0.f;
        }
        if (!qi_flag) rqicuten[i] = 0.f;
        if (!qs_flag) rqscuten[i] = 0.f;
    }
}

// On-card evidence that the column's shared compile route preserves tiny
// finite values. It uses the same rounded intrinsics as the column.
extern "C" __global__ void kf_old_float_probe(const float* x, float* y) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        y[0] = __fmul_rn(x[0], x[1]);
        y[1] = __fdiv_rn(x[2], x[3]);
    }
}
