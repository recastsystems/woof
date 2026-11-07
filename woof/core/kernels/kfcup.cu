/* Kain-Fritsch Cumulus Potential (KF-CuP), WRF v4.6.1 cu_physics = 10.
 *
 * A statement-order transcription of phys/module_cu_kfcup.F (sha256
 * d5773c66c9e353b7838bc14e59d66905091425ce449e84cea58f14e7725b074c), one
 * CUDA thread per mass column, graded word for word against WRF's own
 * compiled object by tests/test_kfcup_column_oracle.py over the capture in
 * tests/data/oracles/kfcup (tools/kfcup_wrf461_oracle).
 *
 * WHAT IS HERE, in WRF's call order:
 *   kc_slope_sigma    cupSlopeSigma   :4614-4912  trigger potential (cupflag),
 *                                                 PDF slopes, sigmas, wStar
 *   kc_jfd            cup_jfd         :4470-4610  the joint frequency
 *                                                 distribution of (theta, r)
 *                                                 perturbations
 *   kc_para           KF_cup_PARA     :1252-3658  the KF column: USL search,
 *                                                 CuP or standard trigger,
 *                                                 updraft, downdraft, closure,
 *                                                 feedback, CuP diagnostics
 *   kc_adjust_mf      adjust_mfentdet_kfcup :5133-5284
 *   kc_cloud_fraction cupCloudFraction :4297-4466
 *   kfcup_column      KF_cup_CPS      :92-1249    the driver: hold test,
 *                                                 PERTLOOPS, deep/shallow
 *                                                 selection, outputs
 * and the helpers TPMIX2, DTFRZNEW, CONDLOAD, PROF5, TPMIX2DD, ENVIRTHT,
 * findCp, findIndex, findRs.
 *
 * ARITHMETIC.  Every operation is written in WRF's operation order on
 * binary32.  Fortran EXP, ALOG/LOG and REAL**REAL call WOOF's own float32
 * routines gfk_exp, gfk_log and gfk_pow (glibc_flt32.cuh) in strict mode;
 * ordinary fast forecasts bind these to CUDA float functions. SQRT is the
 * correctly rounded sqrtf.  Integer powers are multiply chains (x**2 is
 * x*x on gfortran).  Under GPUWM_WRF_EXACT=1 the module compiles with
 * --fmad=false --ftz=false, and every column output is graded at 0 ULP;
 * the default image uses ordinary float operators and permits FMA.
 *
 * MAX/MIN follow gfortran's lowering, which returns the second operand on a
 * tie after putting a literal constant second -- kc_max/kc_maxc below.  NINT is lroundf (round
 * half away from zero), INT truncates, AINT is truncf, SIGN(a,b) is
 * copysignf(|a|, b).
 *
 * NOT TRANSCRIBED, each for its reason:
 *   - activate_cldbase_kfcup and the qndrop chemistry it feeds: reached only
 *     under flag_chem, which is .false. whenever WRF_CHEM /= 1 (:415-430).
 *     qndrop_ic_cup is therefore identically zero, as it is in WRF.
 *   - cupSlopeSigma's sf_sfclay_physics SELECT CASE (:4680-4737): it sets
 *     UnstableOrNeutral, which nothing reads (the test that used it,
 *     :4743, is commented out).  The capture proves the answer does not
 *     move with sf_sfclay_physics.
 *   - lnterms: WRF does not define it outside shallow columns (the
 *     non-contiguous INTENT(OUT) section at :1075 copies back an
 *     uninitialised temporary) and it is identically zero where it is
 *     defined.
 *   - every WRITE(98,...)/wrf_debug diagnostic.
 *
 * DIVERGENCE, declared (never bit-exact to a WRF bug): cup_jfd's overflow
 * guard (:4547-4580) sets an over-range term to sign(bigcheck, term)
 * UNSQUARED, so a NEGATIVE over-range c/sigmaEZ or m/sigmaSfc becomes
 * -1.8e19 and exp(-0.5*(...)) overflows to +Inf, which then poisons the
 * whole normalised PDF.  Here the guard uses the magnitude, giving that bin
 * the zero frequency the guard exists to give it.  Reachable only when
 * |c/sigmaEZ| or |m/sigmaSfc| exceeds 1.8e19; no capture column reaches it.
 *
 * FATAL CONDITIONS.  WRF stops the model in three places this kernel can
 * reach; here the column sets a status bit and the launcher refuses:
 *   KFC_FATAL_QG_SURFACE   QG(1) < 0 after compensating subsidence (:2940)
 *   KFC_FATAL_MASS_BUDGET  ISTOP = 1, mass flux does not balance (:2968, :3398)
 *   KFC_TABLE_OUT_OF_RANGE TPMIX2 indexed the KF_LUTAB outside its 250 x 220
 *                          grid (:3712) -- WRF prints and reads past the
 *                          table; this kernel clamps the index and flags it
 */

/* The loader's preamble defines G, CP, P0 and T0 as object-like macros
 * (gpuwm/core/constants.py CUDA_DEFINES).  This unit spells WRF's own
 * variables and arrays with exactly those names, so it retires the four
 * macros for itself; nothing in this unit or its header reads them. */
#undef G
#undef CP
#undef P0
#undef T0

#define KC_LANES 32
#define KFC_FATAL_QG_SURFACE   1
#define KFC_FATAL_MASS_BUDGET  2
#define KFC_TABLE_OUT_OF_RANGE 4

/* module_model_constants (share/module_model_constants.F), as the words
 * gfortran folds them: probed on gfortran 13.3 against WRF's own object. */
#define KC_G      9.81f
#define KC_RD     287.0f
#define KC_RV     461.6f
#define KC_CP     1004.5f                       /* 7.*r_d/2. = 0x447B2000 */
#define KC_XLV0   3.15e6f
#define KC_XLV1   2370.0f
#define KC_XLS0   2.905e6f
#define KC_XLS1   259.532f
#define KC_XLV    2.5e6f
#define KC_SVP1   0.6112f
#define KC_SVP2   17.67f
#define KC_SVP3   29.65f
#define KC_SVPT0  273.15f
#define KC_EP1    __int_as_float(0x3F1BBDA4)    /* r_v/r_d - 1. */
#define KC_EP2    __int_as_float(0x3F1F2B09)    /* r_d/r_v */
#define KC_INVXLV __int_as_float(0x34D6BF95)    /* 1./xlv */
#define KC_BIGCHECK __int_as_float(0x5F7FFFFF)  /* sqrt(huge(c)) */
#define KC_REALLYSMALL 5e-4f
#define KC_EPS 0.622f

/* gfortran's MAX/MIN lower to MAX_EXPR/MIN_EXPR (maxss/minss), which return
 * the SECOND operand on a tie -- and the tie that matters is +0 against -0.
 * GCC canonicalises a commutative operation to put a literal constant
 * second, so MAX(0.0, x) is evaluated as MAX(x, 0.0) and a tie returns the
 * CONSTANT.  kc_max/kc_min are the two-variable form; kc_maxc/kc_minc take
 * the literal first, exactly as WRF spells the call, and return it on a tie.
 * MEASURED: spelling MAX(0.0, x) as the two-variable form turns +0 into -0
 * on 488 words of mfdn_ent_cup across the capture. */
__device__ __forceinline__ float kc_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float kc_min(float a, float b) { return (a < b) ? a : b; }
__device__ __forceinline__ float kc_maxc(float c, float x) { return (x > c) ? x : c; }
__device__ __forceinline__ float kc_minc(float c, float x) { return (x < c) ? x : c; }
__device__ __forceinline__ int kc_imax(int a, int b) { return (a > b) ? a : b; }
__device__ __forceinline__ int kc_imin(int a, int b) { return (a < b) ? a : b; }
__device__ __forceinline__ float kc_sign(float a, float b) { return copysignf(fabsf(a), b); }
__device__ __forceinline__ int kc_nint(float x) { return (int)lroundf(x); }

/* A 1-based column array, A(k) for k = 1..n: a workspace slot (stride
 * KC_LANES, lane-interleaved) or a column of an (nz, ncol) engine field
 * (stride ncol). */
struct KcCol {
    float* p;
    size_t stride;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(size_t)(k - 1) * stride];
    }
};
struct KcColI {
    int* p;
    __device__ __forceinline__ int& operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)KC_LANES];
    }
};

/* The KF_LUTAB, the same words cu_physics = 1 loads (gpuwm/data/kf_lutab):
 * KF-CuP's KF_LUTAB (:4164-4291) is the same text as KF-eta's, measured. */
struct KcTable {
    const float* ttab;    /* (250, 220): ttab(it, kp) at (it-1) + (kp-1)*250 */
    const float* qstab;
    const float* the0k;   /* (220) */
    const float* alu;     /* (200) */
    float plutop, rdpr, rdthk;
};

__device__ __forceinline__ float kc_ttab(const KcTable& t, int it, int kp) {
    return t.ttab[(it - 1) + (kp - 1) * 250];
}
__device__ __forceinline__ float kc_qstab(const KcTable& t, int it, int kp) {
    return t.qstab[(it - 1) + (kp - 1) * 250];
}

/* ------------------------------------------------------------------------
 * TPMIX2 (:3663-3787)
 * ---------------------------------------------------------------------- */
__device__ void kc_tpmix2(const KcTable& tb, float p, float thes, float* tu, float* qu,
                          float* qliq, float* qice, float* qnewlq, float* qnewic,
                          float xlv1, float xlv0, int* status)
{
    float tp = (p - tb.plutop) * tb.rdpr;
    float qq = tp - truncf(tp);
    int iptb = (int)tp + 1;
    if (iptb < 1 || iptb > 219) {
        *status |= KFC_TABLE_OUT_OF_RANGE;
        iptb = kc_imin(kc_imax(iptb, 1), 219);
    }
    float bth = (tb.the0k[iptb] - tb.the0k[iptb - 1]) * qq + tb.the0k[iptb - 1];
    float tth = (thes - bth) * tb.rdthk;
    float pp = tth - truncf(tth);
    int ithtb = (int)tth + 1;
    /* WRF prints '**** OUT OF BOUNDS' when IPTB or ITHTB reaches 1 or the
     * top cell (:3712) and reads on; index 1 and the top cell are inside the
     * table, so only an index that would leave it is clamped and flagged. */
    if (iptb < 1 || iptb > 219 || ithtb < 1 || ithtb > 249) {
        *status |= KFC_TABLE_OUT_OF_RANGE;
        iptb = kc_imin(kc_imax(iptb, 1), 219);
        ithtb = kc_imin(kc_imax(ithtb, 1), 249);
    }
    float t00 = kc_ttab(tb, ithtb, iptb);
    float t10 = kc_ttab(tb, ithtb + 1, iptb);
    float t01 = kc_ttab(tb, ithtb, iptb + 1);
    float t11 = kc_ttab(tb, ithtb + 1, iptb + 1);
    float q00 = kc_qstab(tb, ithtb, iptb);
    float q10 = kc_qstab(tb, ithtb + 1, iptb);
    float q01 = kc_qstab(tb, ithtb, iptb + 1);
    float q11 = kc_qstab(tb, ithtb + 1, iptb + 1);
    float temp = (t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq);
    float qs = (q00 + (q10 - q00) * pp + (q01 - q00) * qq + (q00 - q10 - q01 + q11) * pp * qq);
    float dq = qs - *qu;
    float qnew;
    if (dq <= 0.f) {
        qnew = *qu - qs;
        *qu = qs;
    } else {
        qnew = 0.f;
        float qtot = *qliq + *qice;
        if (qtot >= dq) {
            *qliq = *qliq - dq * *qliq / (qtot + 1.e-10f);
            *qice = *qice - dq * *qice / (qtot + 1.e-10f);
            *qu = qs;
        } else {
            float rll = xlv0 - xlv1 * temp;
            float cpp = 1004.5f * (1.f + 0.89f * *qu);
            if (qtot < 1.e-10f) {
                temp = temp + rll * (dq / (1.f + dq)) / cpp;
            } else {
                temp = temp + rll * ((dq - qtot) / (1.f + dq - qtot)) / cpp;
                *qu = *qu + qtot;
                qtot = 0.f;
                *qliq = 0.f;
                *qice = 0.f;
            }
        }
    }
    *tu = temp;
    *qnewlq = qnew;
    *qnewic = 0.f;
}

/* ------------------------------------------------------------------------
 * DTFRZNEW (:3789-3832)
 * ---------------------------------------------------------------------- */
__device__ void kc_dtfrznew(float* tu, float p, float* thteu, float* qu, float qfrz,
                            float* qice, float aliq, float bliq, float cliq, float dliq)
{
    float rlc = 2.5e6f - 2369.276f * (*tu - 273.16f);
    float rls = 2833922.f - 259.532f * (*tu - 273.16f);
    float rlf = rls - rlc;
    float cpp = 1004.5f * (1.f + 0.89f * *qu);
    float a = (cliq - bliq * dliq) / ((*tu - dliq) * (*tu - dliq));
    float dtfrz = rlf * qfrz / (cpp + rls * *qu * a);
    *tu = *tu + dtfrz;
    float es = aliq * gfk_exp((bliq * *tu - cliq) / (*tu - dliq));
    float qs = es * 0.622f / (p - es);
    float dqevap = qs - *qu;
    *qice = *qice - dqevap;
    *qu = *qu + dqevap;
    float pii = gfk_pow(1.e5f / p, 0.2854f * (1.f - 0.28f * *qu));
    *thteu = *tu * pii * gfk_exp((3374.6525f / *tu - 2.5403f) * *qu * (1.f + 0.81f * *qu));
}

/* ------------------------------------------------------------------------
 * CONDLOAD (:3835-3905) -- KF-CuP's own copy, which guards the WAVG divide
 * ---------------------------------------------------------------------- */
__device__ void kc_condload(float* qliq, float* qice, float* wtw, float dz, float boterm,
                            float enterm, float rate, float* qnewlq, float* qnewic,
                            float* qlqout, float* qicout, float g)
{
    float qtot = *qliq + *qice;
    float qnew = *qnewlq + *qnewic;
    float qest = 0.5f * (qtot + qnew);
    float g1 = *wtw + boterm - enterm - 2.f * g * dz * qest / 1.5f;
    if (g1 < 0.0f) g1 = 0.f;
    float wavg = 0.5f * (sqrtf(*wtw) + sqrtf(g1));
    float conv = rate * dz / kc_max(wavg, 1e-7f);
    float ratio3 = *qnewlq / (qnew + 1.e-8f);
    qtot = qtot + 0.6f * qnew;
    float oldq = qtot;
    float ratio4 = (0.6f * *qnewlq + *qliq) / (qtot + 1.e-8f);
    qtot = qtot * gfk_exp(-conv);
    float dq = oldq - qtot;
    *qlqout = ratio4 * dq;
    *qicout = (1.f - ratio4) * dq;
    float pptdrg = 0.5f * (oldq + qtot - 0.2f * qnew);
    *wtw = *wtw + boterm - enterm - 2.f * g * dz * pptdrg / 1.5f;
    if (fabsf(*wtw) < 1.e-4f) *wtw = 1.e-4f;
    *qliq = ratio4 * qtot + ratio3 * 0.4f * qnew;
    *qice = (1.f - ratio4) * qtot + (1.f - ratio3) * 0.4f * qnew;
    *qnewlq = 0.f;
    *qnewic = 0.f;
}

/* ------------------------------------------------------------------------
 * PROF5 (:3908-3950)
 * ---------------------------------------------------------------------- */
__device__ void kc_prof5(float eq, float* ee, float* ud)
{
    const float sqrt2p = 2.506628f, a1 = 0.4361836f, a2 = -0.1201676f, a3 = 0.9372980f;
    const float p = 0.33267f, sigma = 0.166666667f, fe = 0.202765151f;
    float x = (eq - 0.5f) / sigma;
    (void)x;
    float y = 6.f * eq - 3.f;
    float ey = gfk_exp(y * y / (-2.f));
    float e45 = gfk_exp(-4.5f);
    float t2 = 1.f / (1.f + p * fabsf(y));
    float t1 = 0.500498f;
    float c1 = a1 * t1 + a2 * t1 * t1 + a3 * t1 * t1 * t1;
    float c2 = a1 * t2 + a2 * t2 * t2 + a3 * t2 * t2 * t2;
    if (y >= 0.f) {
        *ee = sigma * (0.5f * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * eq * eq / 2.f;
        *ud = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * (0.5f + eq * eq / 2.f - eq);
    } else {
        *ee = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * eq * eq / 2.f;
        *ud = sigma * (0.5f * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * (0.5f + eq * eq / 2.f - eq);
    }
    *ee = *ee / fe;
    *ud = *ud / fe;
}

/* ------------------------------------------------------------------------
 * TPMIX2DD (:3953-4017)
 * ---------------------------------------------------------------------- */
__device__ void kc_tpmix2dd(const KcTable& tb, float p, float thes, float* ts, float* qs,
                            int* status)
{
    float tp = (p - tb.plutop) * tb.rdpr;
    float qq = tp - truncf(tp);
    int iptb = (int)tp + 1;
    if (iptb < 1 || iptb > 219) {
        *status |= KFC_TABLE_OUT_OF_RANGE;
        iptb = kc_imin(kc_imax(iptb, 1), 219);
    }
    float bth = (tb.the0k[iptb] - tb.the0k[iptb - 1]) * qq + tb.the0k[iptb - 1];
    float tth = (thes - bth) * tb.rdthk;
    float pp = tth - truncf(tth);
    int ithtb = (int)tth + 1;
    if (ithtb < 1 || ithtb > 249) {
        *status |= KFC_TABLE_OUT_OF_RANGE;
        ithtb = kc_imin(kc_imax(ithtb, 1), 249);
    }
    float t00 = kc_ttab(tb, ithtb, iptb);
    float t10 = kc_ttab(tb, ithtb + 1, iptb);
    float t01 = kc_ttab(tb, ithtb, iptb + 1);
    float t11 = kc_ttab(tb, ithtb + 1, iptb + 1);
    float q00 = kc_qstab(tb, ithtb, iptb);
    float q10 = kc_qstab(tb, ithtb + 1, iptb);
    float q01 = kc_qstab(tb, ithtb, iptb + 1);
    float q11 = kc_qstab(tb, ithtb + 1, iptb + 1);
    *ts = (t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq);
    *qs = (q00 + (q10 - q00) * pp + (q01 - q00) * qq + (q00 - q10 - q01 + q11) * pp * qq);
}

/* ------------------------------------------------------------------------
 * ENVIRTHT (:4020-4056)
 * ---------------------------------------------------------------------- */
__device__ float kc_envirtht(const KcTable& tb, float p1, float t1, float q1,
                             float aliq, float bliq, float cliq, float dliq)
{
    const float t00 = 273.16f, p00 = 1.e5f, c1 = 3374.6525f, c2 = 2.5403f;
    float ee = q1 * p1 / (0.622f + q1);
    float astrt = 1.e-3f;
    float ainc = 0.075f;
    float a1 = ee / aliq;
    float tp = (a1 - astrt) / ainc;
    int indlu = (int)tp + 1;
    float value = (float)(indlu - 1) * ainc + astrt;
    float aintrp = (a1 - value) / ainc;
    float tlog = aintrp * tb.alu[indlu] + (1.f - aintrp) * tb.alu[indlu - 1];
    float tdpt = (cliq - dliq * tlog) / (bliq - tlog);
    float tsat = tdpt - (.212f + 1.571e-3f * (tdpt - t00) - 4.36e-4f * (t1 - t00)) * (t1 - tdpt);
    float tht = t1 * gfk_pow(p00 / p1, 0.2854f * (1.f - 0.28f * q1));
    return tht * gfk_exp((c1 / tsat - c2) * q1 * (1.f + 0.81f * q1));
}

/* ------------------------------------------------------------------------
 * findCp, findIndex, findRs (:4918-4958)
 * ---------------------------------------------------------------------- */
__device__ __forceinline__ float kc_find_cp(float r) { return 1004.67f * (1.0f + 0.84f * r); }

__device__ __forceinline__ float kc_find_rs(float t, float p)
{
    float es = 610.78f * gfk_exp(17.67f * (t - 273.16f) / (t - 29.66f));
    return KC_EPS * es / (p - es);
}

/* ========================================================================
 * The per-thread column workspace.
 *
 * KF_cup_PARA alone declares ~90 column arrays; with the driver's
 * accumulators a column needs ~150.  They live in a GLOBAL workspace the
 * launcher sizes to the threads in flight (the reason is kf.cu's: CUDA
 * prices a local frame at the card's resident-thread capacity), laid out
 * lane-interleaved per warp so arr(k) across a warp is 32 consecutive
 * floats.  Every slot has extent nz + 1 (OMG needs kte + 1).  The joint
 * frequency distribution follows the slots: numBins**2 + 2*numBins floats.
 * Slot ids are literal; tests/test_kfcup_column_oracle.py re-derives KFCUP_WS_SLOTS
 * from this source and fails on a duplicate or an id past the cap.
 * ====================================================================== */
#define KFCUP_WS_SLOTS 148

struct KcWs {
    float* base;   /* this thread's lane in its warp's region */
    int kp;        /* per-slot extent, nz + 1 */
    __device__ __forceinline__ KcCol col(int slot) const {
        return KcCol{base + (size_t)slot * (size_t)kp * (size_t)KC_LANES, (size_t)KC_LANES};
    }
    __device__ __forceinline__ KcColI coli(int slot) const {
        return KcColI{(int*)(base + (size_t)slot * (size_t)kp * (size_t)KC_LANES)};
    }
    /* the PDF region, after the slots: element e (0-based) */
    __device__ __forceinline__ float& pdf(int e) const {
        return base[((size_t)KFCUP_WS_SLOTS * (size_t)kp + (size_t)e) * (size_t)KC_LANES];
    }
};

/* The KF_cup_PARA argument block that is not a column array. */
struct KcParaIn {
    float dt, dx, dxsq, pblh, th_perturb, r_perturb, freq;
    int nz, ntst;
    bool cupflag, f_qi, f_qs, warm_rain;
};

/* Arrays the driver hands KF_cup_PARA (inputs and in/out accumulators). */
struct KcParaArrays {
    KcCol u0, v0, t0, qv0, p0, dzq, w0avg1d, rhoe, z_at_w1d;
    KcCol dqdt, dqidt, dqcdt, dqrdt, dqsdt, dtdt, qlg, qig;
    KcCol umfout, uerout, udrout, dmfout, derout, ddrout, wup;
    KcCol qndrop1d, qc1d, qi1d, fcvt_qc_to_qi, fcvt_qc_to_pr, fcvt_qi_to_pr;
};

/* KF_cup_PARA's 2-D INOUT/OUT scalars for this column, and its OUT scalars. */
struct KcParaOut {
    float raincv, nca, cutop, cubot, wlcl, updfra, wulcl, wact;
    int ishall;
};

/* ------------------------------------------------------------------------
 * KF_cup_PARA (:1252-3658)
 *
 * The local arrays are workspace slots 0..89; the slot table is the
 * declaration list at :1402-1486 in its own order, less the arrays WRF
 * declares and never reads (THTES, W0, RATIO, TKE, WD, RATIO2 -- each is
 * written and never read, or neither).
 * ---------------------------------------------------------------------- */
__device__ void kc_para(const KcTable& tb, const KcWs& ws, const KcParaIn& in,
                        const KcParaArrays& a, KcParaOut& o, int* status)
{
    const int kts = 1, kte = in.nz;
    KcCol Q0 = ws.col(0), Z0 = ws.col(1), TV0 = ws.col(2), TU = ws.col(3), TVU = ws.col(4);
    KcCol QU = ws.col(5), TZ = ws.col(6), TVD = ws.col(7), QD = ws.col(8), QES = ws.col(9);
    KcCol TG = ws.col(10), TVG = ws.col(11), QG = ws.col(12), WU = ws.col(13), EMS = ws.col(14);
    KcCol EMSD = ws.col(15), UMF = ws.col(16), UER = ws.col(17), UDR = ws.col(18), DMF = ws.col(19);
    KcCol DER = ws.col(20), DDR = ws.col(21), UMF2 = ws.col(22), UER2 = ws.col(23), UDR2 = ws.col(24);
    KcCol DMF2 = ws.col(25), DER2 = ws.col(26), DDR2 = ws.col(27), DZA = ws.col(28), THTA0 = ws.col(29);
    KcCol THETEE = ws.col(30), THTAU = ws.col(31), THETEU = ws.col(32), THTAD = ws.col(33), THETED = ws.col(34);
    KcCol QLIQ = ws.col(35), QICE = ws.col(36), QLQOUT = ws.col(37), QICOUT = ws.col(38), PPTLIQ = ws.col(39);
    KcCol PPTICE = ws.col(40), DETLQ = ws.col(41), DETIC = ws.col(42), DETLQ2 = ws.col(43), DETIC2 = ws.col(44);
    KcCol DOMGDP = ws.col(45), EXN = ws.col(46), TVQU = ws.col(47), DP = ws.col(48), RH = ws.col(49);
    KcCol EQFRC = ws.col(50), WSPD = ws.col(51), QDT = ws.col(52), FXM = ws.col(53), THTAG = ws.col(54);
    KcCol THPA = ws.col(55), THFXOUT = ws.col(56), THFXIN = ws.col(57), QPA = ws.col(58), QFXOUT = ws.col(59);
    KcCol QFXIN = ws.col(60), QLPA = ws.col(61), QLFXIN = ws.col(62), QLFXOUT = ws.col(63), QIPA = ws.col(64);
    KcCol QIFXIN = ws.col(65), QIFXOUT = ws.col(66), QRPA = ws.col(67), QRFXIN = ws.col(68), QRFXOUT = ws.col(69);
    KcCol QSPA = ws.col(70), QSFXIN = ws.col(71), QSFXOUT = ws.col(72), QRG = ws.col(73), QSG = ws.col(74);
    KcCol OMG = ws.col(75), RAINFB = ws.col(76), SNOWFB = ws.col(77), CLDHGT = ws.col(78), QSD = ws.col(79);
    KcCol DILFRC = ws.col(80), DDILFRC = ws.col(81), TGU = ws.col(82), QGU = ws.col(83), THTEEG = ws.col(84);
    KcCol qndropbb = ws.col(85);
    KcColI KCHECK = ws.coli(86);
    /* QL0, QI0, QR0, QS0 are zero at every level (:1571-1574) and only ever
     * read; they are the literal 0.f below, in the operations WRF does
     * with them. */
    const float QL0 = 0.f, QI0 = 0.f, QR0 = 0.f, QS0 = 0.f;

    const KcCol& U0 = a.u0; const KcCol& V0 = a.v0; const KcCol& T0 = a.t0;
    const KcCol& QV0 = a.qv0; const KcCol& P0 = a.p0; const KcCol& DZQ = a.dzq;
    const KcCol& W0AVG1D = a.w0avg1d; const KcCol& rhoe = a.rhoe; const KcCol& z_at_w1d = a.z_at_w1d;

    const float DT = in.dt, DX = in.dx, DXSQ = in.dxsq;
    const float XLV0 = KC_XLV0, XLV1 = KC_XLV1, CP = KC_CP, R = KC_RD, G = KC_G;
    const float P00 = 1.e5f, T00 = 273.16f, RLF = 3.339e5f;
    const float TTFRZ = 268.16f, TBFRZ = 248.16f, RATE = 0.03f;

    float GDRY = -G / CP;
    float ROCP = R / CP;
    int NSHALL = 0;
    (void)NSHALL;
    const int KL = kte, KX = kte;

    o.updfra = 0.0f;
    for (int k = kts; k <= kte; ++k) a.wup(k) = 0.0f;
    o.wulcl = 0.0f;
    o.wact = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        a.qndrop1d(k) = 0.0f; a.qc1d(k) = 0.0f; a.qi1d(k) = 0.0f;
        a.fcvt_qc_to_qi(k) = 0.0f; a.fcvt_qc_to_pr(k) = 0.0f; a.fcvt_qi_to_pr(k) = 0.0f;
        a.umfout(k) = 0.0f; a.uerout(k) = 0.0f; a.udrout(k) = 0.0f;
        a.dmfout(k) = 0.0f; a.derout(k) = 0.0f; a.ddrout(k) = 0.0f;
    }

    const float ALIQ = KC_SVP1 * 1000.f;
    const float BLIQ = KC_SVP2;
    const float CLIQ = KC_SVP2 * KC_SVPT0;
    const float DLIQ = KC_SVP3;

    float FBFRC = 0.0f;
    int NCHM = 0;
    int ISHALL = 0;
    const float DPMIN = 5.e3f;
    float P300 = P0(1) - 30000.f;
    const float TIMEC_SHALL = 1800.0f;

    int ML = 0, L5 = 0, LLFC = 0;
    for (int K = 1; K <= KX; ++K) {
        float ES = ALIQ * gfk_exp((BLIQ * T0(K) - CLIQ) / (T0(K) - DLIQ));
        QES(K) = 0.622f * ES / (P0(K) - ES);
        Q0(K) = kc_min(QES(K), QV0(K));
        Q0(K) = kc_maxc(0.000001f, Q0(K));
        RH(K) = Q0(K) / QES(K);
        DILFRC(K) = 1.f;
        TV0(K) = T0(K) * (1.f + 0.608f * Q0(K));
        DP(K) = rhoe(K) * G * DZQ(K);
        CLDHGT(K) = 0.f;
        if (P0(K) >= 0.5f * P0(1)) L5 = K;
        if (P0(K) >= P300) LLFC = K;
        if (T0(K) > T00) ML = K;
    }
    Z0(1) = .5f * DZQ(1);
    for (int K = 2; K <= KL; ++K) {
        Z0(K) = Z0(K - 1) + .5f * (DZQ(K) + DZQ(K - 1));
        DZA(K - 1) = Z0(K) - Z0(K - 1);
    }
    DZA(KL) = 0.f;

    int NCHECK = 1;
    KCHECK(NCHECK) = 1;
    float PM15 = P0(1) - 15.E2f;
    for (int K = 2; K <= LLFC; ++K) {
        if (P0(K) < PM15) {
            NCHECK = NCHECK + 1;
            KCHECK(NCHECK) = K;
            PM15 = PM15 - 15.E2f;
        }
    }

    int NU = 0, NUCHM = 0;
    int KMIX, LOW, LC = 0, NLAYRS, NK, KCLDLAYER = 0, KLCL = 0, LCL = 0, LET = 0, K = 0, LTOP = 0;
    float DPTHMX = 0.f, TMIX = 0.f, QMIX = 0.f, ZMIX = 0.f, PMIX = 0.f, EMIX, TLOG, TDPT, TLCL = 0.f;
    float TVLCL = 0.f, ZLCL = 0.f, DLP, TENV, QENV, TVEN = 0.f, WKLCL = 0.f, WKL = 0.f;
    float DTLCL = 0.f, DTRH = 0.f, PLCL = 0.f, WLCL = o.wlcl, WTW, RHOLCL, RAD = 0.f, AU0 = 0.f, VMFLCL = 0.f;
    float UPOLD = 0.f, UPNEW = 0.f, ABE = 0.f, TRPPT = 0.f, TTEMP, EE1, UD1, REI, DILBE, CHMIN, CHMAX;
    float astrt, ainc, a1, tp, value, aintrp;
    int indlu, ksvaa = 0;
    (void)ksvaa;
    while (true) {   /* usl: DO */
        NU = NU + 1;
        if (NU > NCHECK) {
            if (ISHALL == 1) {
                CHMAX = 0.f;
                NCHM = 0;
                for (int NK2 = 1; NK2 <= NCHECK; ++NK2) {
                    int NNN = KCHECK(NK2);
                    if (CLDHGT(NNN) > CHMAX) {
                        NCHM = NNN;
                        NUCHM = NK2;
                        CHMAX = CLDHGT(NNN);
                    }
                }
                NU = NUCHM - 1;
                FBFRC = 1.f;
                continue;   /* CYCLE usl */
            } else {
                ISHALL = 2;
                o.ishall = ISHALL; o.wlcl = WLCL;
                return;
            }
        }
        KMIX = KCHECK(NU);
        LOW = KMIX;
        LC = LOW;
        NLAYRS = 0;
        DPTHMX = 0.f;
        NK = LC - 1;
        if (NK + 1 < kts) {
            /* WOULD GO OFF BOTTOM: wrf_message only */
        } else {
            while (true) {
                NK = NK + 1;
                if (NK > kte) break;   /* WOULD GO OFF TOP: wrf_message only */
                DPTHMX = DPTHMX + DP(NK);
                NLAYRS = NLAYRS + 1;
                if (DPTHMX > DPMIN) break;
            }
        }
        if (DPTHMX < DPMIN) {
            ISHALL = 2;
            o.ishall = ISHALL; o.wlcl = WLCL;
            return;
        }
        KCLDLAYER = LC + NLAYRS - 1;
        TMIX = 0.f;
        QMIX = 0.f;
        ZMIX = 0.f;
        PMIX = 0.f;
        for (NK = LC; NK <= KCLDLAYER; ++NK) {
            TMIX = TMIX + DP(NK) * T0(NK);
            QMIX = QMIX + DP(NK) * Q0(NK);
            ZMIX = ZMIX + DP(NK) * Z0(NK);
            PMIX = PMIX + DP(NK) * P0(NK);
        }
        TMIX = TMIX / DPTHMX;
        QMIX = QMIX / DPTHMX;
        ZMIX = ZMIX / DPTHMX;
        PMIX = PMIX / DPTHMX;
        EMIX = QMIX * PMIX / (0.622f + QMIX);
        astrt = 1.e-3f;
        ainc = 0.075f;
        a1 = EMIX / ALIQ;
        tp = (a1 - astrt) / ainc;
        indlu = (int)tp + 1;
        value = (float)(indlu - 1) * ainc + astrt;
        aintrp = (a1 - value) / ainc;
        TLOG = aintrp * tb.alu[indlu] + (1 - aintrp) * tb.alu[indlu - 1];
        TDPT = (CLIQ - DLIQ * TLOG) / (BLIQ - TLOG);
        TLCL = TDPT - (.212f + 1.571E-3f * (TDPT - T00) - 4.36E-4f * (TMIX - T00)) * (TMIX - TDPT);
        TLCL = kc_min(TLCL, TMIX);
        TVLCL = TLCL * (1.f + 0.608f * QMIX);
        ZLCL = ZMIX + (TLCL - TMIX) / GDRY;
        NK = LC - 1;
        while (true) {
            NK = NK + 1;
            KLCL = NK;
            if (NK > KL) break;   /* (NK.GT.KL tested after the read below in WRF) */
            if (ZLCL <= Z0(NK)) break;
        }
        if (NK > KL) {
            ISHALL = 2;
            o.ishall = ISHALL; o.wlcl = WLCL;
            return;
        }
        K = KLCL - 1;
        DLP = (ZLCL - Z0(K)) / (Z0(KLCL) - Z0(K));
        TENV = T0(K) + (T0(KLCL) - T0(K)) * DLP;
        QENV = Q0(K) + (Q0(KLCL) - Q0(K)) * DLP;
        TVEN = TENV * (1.f + 0.608f * QENV);
        if (ZLCL < 2.E3f) {
            WKLCL = 0.02f * ZLCL / 2.E3f;
        } else {
            WKLCL = 0.02f;
        }
        WKL = (W0AVG1D(K) + (W0AVG1D(KLCL) - W0AVG1D(K)) * DLP) * DX / 25.E3f - WKLCL;
        if (!in.cupflag) {
            if (WKL < 0.0001f) {
                DTLCL = 0.f;
            } else {
                DTLCL = 4.64f * gfk_pow(WKL, 0.33f);
            }
            DTRH = 0.f;
        } else {
            PLCL = P0(K) + (P0(KLCL) - P0(K)) * DLP;
            DTLCL = in.th_perturb * gfk_pow(P00 / P0(K), ROCP);
            DTRH = 0.608f * in.r_perturb;
        }
        TVLCL = TLCL * (1.f + 0.608f * QMIX);
        if (TVLCL + DTLCL + DTRH < TVEN) {
            continue;   /* CYCLE usl: parcel not buoyant */
        }
        THETEU(K) = kc_envirtht(tb, PMIX, TMIX, QMIX, ALIQ, BLIQ, CLIQ, DLIQ);
        float DTTOT = DTLCL + DTRH;
        if (DTTOT > 1.E-4f) {
            float GDT = 2.f * G * DTTOT * 500.f / TVEN;
            WLCL = 1.f + 0.5f * sqrtf(GDT);
            if (!in.cupflag) WLCL = kc_min(WLCL, 3.f);
        } else {
            if (in.cupflag) {
                WLCL = 0.f;
            } else {
                WLCL = 1.f;
            }
        }
        PLCL = P0(K) + (P0(KLCL) - P0(K)) * DLP;
        WTW = WLCL * WLCL;
        TVLCL = TLCL * (1.f + 0.608f * QMIX);
        RHOLCL = PLCL / (R * TVLCL);
        LCL = KLCL;
        LET = LCL;
        if (WKL < 0.f) {
            RAD = 1000.f;
        } else if (WKL > 0.1f) {
            RAD = 2000.f;
        } else {
            RAD = 1000.f + 1000 * WKL / 0.1f;
        }
        WU(K) = WLCL;
        AU0 = 0.01f * DXSQ;
        UMF(K) = RHOLCL * AU0;
        VMFLCL = UMF(K);
        UPOLD = VMFLCL;
        UPNEW = UPOLD;
        ksvaa = K;
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
        TTEMP = TTFRZ;
        EE1 = 1.f;
        UD1 = 0.f;
        REI = 0.f;
        DILBE = 0.f;
        for (int kk = kts; kk <= kte; ++kk) qndropbb(kk) = 0.0f;

        float qnewlq, qnewic, EE2 = 0.f, UD2 = 0.f;
        for (NK = K; NK <= KL - 1; ++NK) {   /* updraft */
            int NK1 = NK + 1;
            float FRC1 = 0.f;
            TU(NK1) = T0(NK1);
            THETEU(NK1) = THETEU(NK);
            QU(NK1) = QU(NK);
            QLIQ(NK1) = QLIQ(NK);
            QICE(NK1) = QICE(NK);
            {
                float tu = TU(NK1), qu = QU(NK1), ql = QLIQ(NK1), qi = QICE(NK1);
                kc_tpmix2(tb, P0(NK1), THETEU(NK1), &tu, &qu, &ql, &qi, &qnewlq, &qnewic,
                          XLV1, XLV0, status);
                TU(NK1) = tu; QU(NK1) = qu; QLIQ(NK1) = ql; QICE(NK1) = qi;
            }
            if (TU(NK1) <= TTFRZ) {
                if (TU(NK1) > TBFRZ) {
                    if (TTEMP > TTFRZ) TTEMP = TTFRZ;
                    FRC1 = (TTEMP - TU(NK1)) / (TTEMP - TBFRZ);
                } else {
                    FRC1 = 1.f;
                }
                TTEMP = TU(NK1);
                float tmpa = kc_maxc(0.0f, QLIQ(NK1) + qnewlq);
                float QFRZ = (QLIQ(NK1) + qnewlq) * FRC1;
                qnewic = qnewic + qnewlq * FRC1;
                qnewlq = qnewlq - qnewlq * FRC1;
                QICE(NK1) = QICE(NK1) + QLIQ(NK1) * FRC1;
                QLIQ(NK1) = QLIQ(NK1) - QLIQ(NK1) * FRC1;
                float tmpc = kc_maxc(0.0f, QLIQ(NK1) + qnewlq);
                a.fcvt_qc_to_qi(NK1) = kc_maxc(0.0f, tmpa - tmpc) / kc_maxc(1.0e-10f, tmpa);
                {
                    float tu = TU(NK1), th = THETEU(NK1), qu = QU(NK1), qi = QICE(NK1);
                    kc_dtfrznew(&tu, P0(NK1), &th, &qu, QFRZ, &qi, ALIQ, BLIQ, CLIQ, DLIQ);
                    TU(NK1) = tu; THETEU(NK1) = th; QU(NK1) = qu; QICE(NK1) = qi;
                }
            }
            TVU(NK1) = TU(NK1) * (1.f + 0.608f * QU(NK1));
            float BE, BOTERM, DZZ;
            if (NK == K) {
                BE = (TVLCL + TVU(NK1)) / (TVEN + TV0(NK1)) - 1.f;
                BOTERM = 2.f * (Z0(NK1) - ZLCL) * G * BE / 1.5f;
                DZZ = Z0(NK1) - ZLCL;
            } else {
                BE = (TVU(NK) + TVU(NK1)) / (TV0(NK) + TV0(NK1)) - 1.f;
                BOTERM = 2.f * DZA(NK) * G * BE / 1.5f;
                DZZ = DZA(NK);
            }
            float ENTERM = 2.f * REI * WTW / UPOLD;
            {
                float tmpa = kc_maxc(0.0f, QLIQ(NK1) + qnewlq);
                float tmpb = kc_maxc(0.0f, QICE(NK1) + qnewic);
                float ql = QLIQ(NK1), qi = QICE(NK1), qlo, qio;
                kc_condload(&ql, &qi, &WTW, DZZ, BOTERM, ENTERM, RATE, &qnewlq, &qnewic,
                            &qlo, &qio, G);
                QLIQ(NK1) = ql; QICE(NK1) = qi; QLQOUT(NK1) = qlo; QICOUT(NK1) = qio;
                float tmpc = kc_maxc(0.0f, QLIQ(NK1) + qnewlq);
                a.fcvt_qc_to_pr(NK1) = kc_maxc(0.0f, tmpa - tmpc) / kc_maxc(1.0e-10f, tmpa);
                tmpc = kc_maxc(0.0f, QICE(NK1) + qnewic);
                a.fcvt_qi_to_pr(NK1) = kc_maxc(0.0f, tmpb - tmpc) / kc_maxc(1.0e-10f, tmpb);
            }
            if (WTW < 1.E-3f) {
                break;
            } else {
                WU(NK1) = sqrtf(WTW);
            }
            THETEE(NK1) = kc_envirtht(tb, P0(NK1), T0(NK1), Q0(NK1), ALIQ, BLIQ, CLIQ, DLIQ);
            REI = VMFLCL * DP(NK1) * 0.03f / RAD;
            TVQU(NK1) = TU(NK1) * (1.f + 0.608f * QU(NK1) - QLIQ(NK1) - QICE(NK1));
            if (NK == K) {
                DILBE = ((TVLCL + TVQU(NK1)) / (TVEN + TV0(NK1)) - 1.f) * DZZ;
            } else {
                DILBE = ((TVQU(NK) + TVQU(NK1)) / (TV0(NK) + TV0(NK1)) - 1.f) * DZZ;
            }
            if (DILBE > 0.f) ABE = ABE + DILBE * G;
            if (TVQU(NK1) <= TV0(NK1)) {
                EE2 = 0.5f;
                UD2 = 1.f;
                EQFRC(NK1) = 0.f;
            } else {
                LET = NK1;
                float TTMP = TVQU(NK1);
                float F1 = 0.95f;
                float F2 = 1.f - F1;
                float THTTMP = F1 * THETEE(NK1) + F2 * THETEU(NK1);
                float QTMP = F1 * Q0(NK1) + F2 * QU(NK1);
                float TMPLIQ = F2 * QLIQ(NK1);
                float TMPICE = F2 * QICE(NK1);
                kc_tpmix2(tb, P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ, &TMPICE, &qnewlq, &qnewic,
                          XLV1, XLV0, status);
                float TU95 = TTMP * (1.f + 0.608f * QTMP - TMPLIQ - TMPICE);
                if (TU95 > TV0(NK1)) {
                    EE2 = 1.f;
                    UD2 = 0.f;
                    EQFRC(NK1) = 1.0f;
                } else {
                    F1 = 0.10f;
                    F2 = 1.f - F1;
                    THTTMP = F1 * THETEE(NK1) + F2 * THETEU(NK1);
                    QTMP = F1 * Q0(NK1) + F2 * QU(NK1);
                    TMPLIQ = F2 * QLIQ(NK1);
                    TMPICE = F2 * QICE(NK1);
                    kc_tpmix2(tb, P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ, &TMPICE, &qnewlq, &qnewic,
                              XLV1, XLV0, status);
                    float TU10 = TTMP * (1.f + 0.608f * QTMP - TMPLIQ - TMPICE);
                    float TVDIFF = fabsf(TU10 - TVQU(NK1));
                    if (TVDIFF < 1.e-3f) {
                        EE2 = 1.f;
                        UD2 = 0.f;
                        EQFRC(NK1) = 1.0f;
                    } else {
                        EQFRC(NK1) = (TV0(NK1) - TVQU(NK1)) * F1 / (TU10 - TVQU(NK1));
                        EQFRC(NK1) = kc_maxc(0.0f, EQFRC(NK1));
                        EQFRC(NK1) = kc_minc(1.0f, EQFRC(NK1));
                        if (EQFRC(NK1) == 1) {
                            EE2 = 1.f;
                            UD2 = 0.f;
                        } else if (EQFRC(NK1) == 0.f) {
                            EE2 = 0.f;
                            UD2 = 1.f;
                        } else {
                            kc_prof5(EQFRC(NK1), &EE2, &UD2);
                        }
                    }
                }
            }
            EE2 = kc_max(EE2, 0.5f);
            UD2 = 1.5f * UD2;
            UER(NK1) = 0.5f * REI * (EE1 + EE2);
            UDR(NK1) = 0.5f * REI * (UD1 + UD2);
            if (UMF(NK) - UDR(NK1) < 10.f) {
                if (DILBE > 0.f) {
                    ABE = ABE - DILBE * G;
                }
                LET = NK;
                break;
            } else {
                EE1 = EE2;
                UD1 = UD2;
                UPOLD = UMF(NK) - UDR(NK1);
                UPNEW = UPOLD + UER(NK1);
                UMF(NK1) = UPNEW;
                DILFRC(NK1) = UPNEW / UPOLD;
                DETLQ(NK1) = QLIQ(NK1) * UDR(NK1);
                DETIC(NK1) = QICE(NK1) * UDR(NK1);
                QDT(NK1) = QU(NK1);
                QU(NK1) = (UPOLD * QU(NK1) + UER(NK1) * Q0(NK1)) / UPNEW;
                THETEU(NK1) = (THETEU(NK1) * UPOLD + THETEE(NK1) * UER(NK1)) / UPNEW;
                QLIQ(NK1) = QLIQ(NK1) * UPOLD / UPNEW;
                QICE(NK1) = QICE(NK1) * UPOLD / UPNEW;
                PPTLIQ(NK1) = QLQOUT(NK1) * UMF(NK);
                PPTICE(NK1) = QICOUT(NK1) * UMF(NK);
                TRPPT = TRPPT + PPTLIQ(NK1) + PPTICE(NK1);
                if (NK1 <= KCLDLAYER) UER(NK1) = UER(NK1) + VMFLCL * DP(NK1) / DPTHMX;
            }
        }   /* END DO updraft; on normal completion NK = KL */
        LTOP = NK;
        CLDHGT(LC) = Z0(LTOP) - ZLCL;
        if (TLCL > 293.f) {
            CHMIN = 4.E3f;
        } else if (TLCL <= 293.f && TLCL >= 273) {
            CHMIN = 2.E3f + 100.f * (TLCL - 273.f);
        } else /* TLCL < 273. */ {
            CHMIN = 2.E3f;
        }
        if (LTOP <= KLCL || LTOP <= KCLDLAYER || LET + 1 <= KCLDLAYER) {
            CLDHGT(LC) = 0.f;
            for (NK = K; NK <= LTOP; ++NK) {
                UMF(NK) = 0.f; UDR(NK) = 0.f; UER(NK) = 0.f; DETLQ(NK) = 0.f;
                DETIC(NK) = 0.f; PPTLIQ(NK) = 0.f; PPTICE(NK) = 0.f;
            }
        } else if (CLDHGT(LC) > CHMIN && ABE > 1) {
            ISHALL = 0;
            break;   /* EXIT usl */
        } else {
            ISHALL = 1;
            if (NU == NUCHM) {
                break;   /* EXIT usl: shallow convection from this layer */
            } else {
                for (NK = K; NK <= LTOP; ++NK) {
                    UMF(NK) = 0.f; UDR(NK) = 0.f; UER(NK) = 0.f; DETLQ(NK) = 0.f;
                    DETIC(NK) = 0.f; PPTLIQ(NK) = 0.f; PPTICE(NK) = 0.f;
                }
            }
        }
    }   /* END DO usl */
    int KSTART;
    if (ISHALL == 1) {
        KSTART = kc_imax(KCLDLAYER, KLCL);
        LET = KSTART;
    }
    if (LET == LTOP) {
        UDR(LTOP) = UMF(LTOP) + UDR(LTOP) - UER(LTOP);
        DETLQ(LTOP) = QLIQ(LTOP) * UDR(LTOP) * UPNEW / UPOLD;
        DETIC(LTOP) = QICE(LTOP) * UDR(LTOP) * UPNEW / UPOLD;
        UER(LTOP) = 0.f;
        UMF(LTOP) = 0.f;
    } else {
        float DPTT = 0.f;
        for (int NJ = LET + 1; NJ <= LTOP; ++NJ) DPTT = DPTT + DP(NJ);
        float DUMFDP = UMF(LET) / DPTT;
        for (NK = LET + 1; NK <= LTOP; ++NK) {
            if (NK == LTOP) {
                UDR(NK) = UMF(NK - 1);
                UER(NK) = 0.f;
                DETLQ(NK) = UDR(NK) * QLIQ(NK) * DILFRC(NK);
                DETIC(NK) = UDR(NK) * QICE(NK) * DILFRC(NK);
            } else {
                UMF(NK) = UMF(NK - 1) - DP(NK) * DUMFDP;
                UER(NK) = UMF(NK) * (1.f - 1.f / DILFRC(NK));
                UDR(NK) = UMF(NK - 1) - UMF(NK) + UER(NK);
                DETLQ(NK) = UDR(NK) * QLIQ(NK) * DILFRC(NK);
                DETIC(NK) = UDR(NK) * QICE(NK) * DILFRC(NK);
            }
            if (NK >= LET + 2) {
                TRPPT = TRPPT - PPTLIQ(NK) - PPTICE(NK);
                PPTLIQ(NK) = UMF(NK - 1) * QLQOUT(NK);
                PPTICE(NK) = UMF(NK - 1) * QICOUT(NK);
                TRPPT = TRPPT + PPTLIQ(NK) + PPTICE(NK);
            }
        }
    }
    for (NK = 1; NK <= K; ++NK) {
        if (NK >= LC) {
            if (NK == LC) {
                UMF(NK) = VMFLCL * DP(NK) / DPTHMX;
                UER(NK) = VMFLCL * DP(NK) / DPTHMX;
            } else if (NK <= KCLDLAYER) {
                UER(NK) = VMFLCL * DP(NK) / DPTHMX;
                UMF(NK) = UMF(NK - 1) + UER(NK);
            } else {
                UMF(NK) = VMFLCL;
                UER(NK) = 0.f;
            }
            TU(NK) = TMIX + (Z0(NK) - ZMIX) * GDRY;
            QU(NK) = QMIX;
            WU(NK) = WLCL;
        } else {
            TU(NK) = 0.f;
            QU(NK) = 0.f;
            UMF(NK) = 0.f;
            WU(NK) = 0.f;
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
        THETEE(NK) = kc_envirtht(tb, P0(NK), T0(NK), Q0(NK), ALIQ, BLIQ, CLIQ, DLIQ);
        EQFRC(NK) = 1.0f;
    }
    int LTOP1 = LTOP + 1;
    int LTOPM1 = LTOP - 1;
    for (NK = LTOP1; NK <= KX; ++NK) {
        UMF(NK) = 0.f; UDR(NK) = 0.f; UER(NK) = 0.f; QDT(NK) = 0.f; QLIQ(NK) = 0.f;
        QICE(NK) = 0.f; QLQOUT(NK) = 0.f; QICOUT(NK) = 0.f; DETLQ(NK) = 0.f; DETIC(NK) = 0.f;
        PPTLIQ(NK) = 0.f; PPTICE(NK) = 0.f;
        if (NK >= LTOP1) {
            TU(NK) = 0.f;
            QU(NK) = 0.f;
            WU(NK) = 0.f;
        }
        THTA0(NK) = 0.f;
        THTAU(NK) = 0.f;
        EMS(NK) = 0.f;
        EMSD(NK) = 0.f;
        TG(NK) = T0(NK);
        QG(NK) = Q0(NK);
        a.qlg(NK) = 0.f;
        a.qig(NK) = 0.f;
        QRG(NK) = 0.f;
        QSG(NK) = 0.f;
        OMG(NK) = 0.f;
    }
    OMG(KX + 1) = 0.f;
    /* (flag_chem is .false.: the qndropbb block :2372-2454 does not run) */
    for (NK = 1; NK <= LTOP; ++NK) {
        EMS(NK) = DP(NK) * DXSQ / G;
        EMSD(NK) = 1.f / EMS(NK);
        EXN(NK) = gfk_pow(P00 / P0(NK), 0.2854f * (1.f - 0.28f * QDT(NK)));
        THTAU(NK) = TU(NK) * EXN(NK);
        EXN(NK) = gfk_pow(P00 / P0(NK), 0.2854f * (1.f - 0.28f * Q0(NK)));
        THTA0(NK) = T0(NK) * EXN(NK);
        DDILFRC(NK) = 1.f / DILFRC(NK);
        OMG(NK) = 0.f;
    }
    WSPD(KLCL) = sqrtf(U0(KLCL) * U0(KLCL) + V0(KLCL) * V0(KLCL));
    WSPD(L5) = sqrtf(U0(L5) * U0(L5) + V0(L5) * V0(L5));
    WSPD(LTOP) = sqrtf(U0(LTOP) * U0(LTOP) + V0(LTOP) * V0(LTOP));
    float VCONV = .5f * (WSPD(KLCL) + WSPD(L5));
    float TIMEC = DX / VCONV;
    float TADVEC = TIMEC;
    TIMEC = kc_maxc(1800.f, TIMEC);
    TIMEC = kc_minc(3600.f, TIMEC);
    if (ISHALL == 1) TIMEC = TIMEC_SHALL;
    int NIC = kc_nint(TIMEC / DT);
    TIMEC = (float)NIC * DT;
    float SHSIGN;
    if (WSPD(LTOP) > WSPD(KLCL)) {
        SHSIGN = 1.f;
    } else {
        SHSIGN = -1.f;
    }
    float VWS = (U0(LTOP) - U0(KLCL)) * (U0(LTOP) - U0(KLCL)) + (V0(LTOP) - V0(KLCL)) *
                (V0(LTOP) - V0(KLCL));
    VWS = 1.E3f * SHSIGN * sqrtf(VWS) / (Z0(LTOP) - Z0(LCL));
    float PEF = 1.591f + VWS * (-.639f + VWS * (9.53E-2f - VWS * 4.96E-3f));
    PEF = kc_max(PEF, .2f);
    PEF = kc_min(PEF, .9f);
    float CBH = (ZLCL - Z0(1)) * 3.281E-3f;
    float RCBH;
    if (CBH < 3.f) {
        RCBH = .02f;
    } else {
        RCBH = .96729352f + CBH * (-.70034167f + CBH * (.162179896f + CBH * (-
               1.2569798E-2f + CBH * (4.2772E-4f - CBH * 5.44E-6f))));
    }
    if (CBH > 25) RCBH = 2.4f;
    float PEFCBH = 1.f / (1.f + RCBH);
    PEFCBH = kc_min(PEFCBH, .9f);
    float PEFF = .5f * (PEF + PEFCBH);

    /* ---- downdraft ---- */
    float TDER = 0.f;
    int LFS = 0, LDB = 0, LDT = 0;
    float DMFFRC = 0.f, DPDD = 0.f;
    if (ISHALL == 1) {
        LFS = 1;
        for (int k2 = 1; k2 <= KX; ++k2) {
            DMF(k2) = 0.f; DER(k2) = 0.f; DDR(k2) = 0.f;
            TZ(k2) = 0.f; QD(k2) = 0.f; THTAD(k2) = 0.f;
        }
    } else {
        KSTART = KCLDLAYER + 1;
        int KLFS = LET - 1;
        for (NK = KSTART + 1; NK <= KL; ++NK) {
            float DPPP = P0(KSTART) - P0(NK);
            if (DPPP > 150.E2f) {
                KLFS = NK;
                break;
            }
        }
        KLFS = kc_imin(KLFS, LET - 1);
        LFS = KLFS;
        if ((P0(KSTART) - P0(LFS)) > 50.E2f) {
            THETED(LFS) = THETEE(LFS);
            QD(LFS) = Q0(LFS);
            float QSS;
            {
                float ts, qs;
                kc_tpmix2dd(tb, P0(LFS), THETED(LFS), &ts, &qs, status);
                TZ(LFS) = ts; QSS = qs;
            }
            THTAD(LFS) = TZ(LFS) * gfk_pow(P00 / P0(LFS), 0.2854f * (1.f - 0.28f * QSS));
            TVD(LFS) = TZ(LFS) * (1.f + 0.608f * QSS);
            float RDD = P0(LFS) / (R * TVD(LFS));
            float A1 = (1.f - PEFF) * AU0;
            DMF(LFS) = -A1 * RDD;
            DER(LFS) = DMF(LFS);
            DDR(LFS) = 0.f;
            float RHBAR = RH(LFS) * DP(LFS);
            float DPTT = DP(LFS);
            for (int ND = LFS - 1; ND >= KSTART; --ND) {
                int ND1 = ND + 1;
                DER(ND) = DER(LFS) * EMS(ND) / EMS(LFS);
                DDR(ND) = 0.f;
                DMF(ND) = DMF(ND1) + DER(ND);
                THETED(ND) = (THETED(ND1) * DMF(ND1) + THETEE(ND) * DER(ND)) / DMF(ND);
                QD(ND) = (QD(ND1) * DMF(ND1) + Q0(ND) * DER(ND)) / DMF(ND);
                DPTT = DPTT + DP(ND);
                RHBAR = RHBAR + RH(ND) * DP(ND);
            }
            RHBAR = RHBAR / DPTT;
            DMFFRC = 2.f * (1.f - RHBAR);
            DPDD = 0.f;
            float pptmlt = 0.f;
            for (NK = KLCL; NK <= LTOP; ++NK) pptmlt = pptmlt + PPTICE(NK);
            float DTMELT;
            if (LC < ML) {
                DTMELT = RLF * pptmlt / (CP * UMF(KLCL));
            } else {
                DTMELT = 0.f;
            }
            LDT = kc_imin(LFS - 1, KSTART - 1);
            {
                float ts, qs;
                kc_tpmix2dd(tb, P0(KSTART), THETED(KSTART), &ts, &qs, status);
                TZ(KSTART) = ts; QSS = qs;
            }
            TZ(KSTART) = TZ(KSTART) - DTMELT;
            float ES = ALIQ * gfk_exp((BLIQ * TZ(KSTART) - CLIQ) / (TZ(KSTART) - DLIQ));
            QSS = 0.622f * ES / (P0(KSTART) - ES);
            THETED(KSTART) = TZ(KSTART) * gfk_pow(1.E5f / P0(KSTART), 0.2854f * (1.f - 0.28f * QSS)) *
                gfk_exp((3374.6525f / TZ(KSTART) - 2.5403f) * QSS * (1.f + 0.81f * QSS));
            LDT = kc_imin(LFS - 1, KSTART - 1);
            for (int ND = LDT; ND >= 1; --ND) {
                DPDD = DPDD + DP(ND);
                THETED(ND) = THETED(KSTART);
                QD(ND) = QD(KSTART);
                {
                    float ts, qs;
                    kc_tpmix2dd(tb, P0(ND), THETED(ND), &ts, &qs, status);
                    TZ(ND) = ts; QSS = qs;
                }
                QSD(ND) = QSS;
                float RHH = 1.f - 0.2f / 1000.f * (Z0(KSTART) - Z0(ND));
                if (RHH < 1.f) {
                    float DSSDT = (CLIQ - BLIQ * DLIQ) / ((TZ(ND) - DLIQ) * (TZ(ND) - DLIQ));
                    float RL = XLV0 - XLV1 * TZ(ND);
                    float DTMP = RL * QSS * (1.f - RHH) / (CP + RL * RHH * QSS * DSSDT);
                    float T1RH = TZ(ND) + DTMP;
                    ES = RHH * ALIQ * gfk_exp((BLIQ * T1RH - CLIQ) / (T1RH - DLIQ));
                    float QSRH = 0.622f * ES / (P0(ND) - ES);
                    if (QSRH < QD(ND)) {
                        QSRH = QD(ND);
                        T1RH = TZ(ND) + (QSS - QSRH) * RL / CP;
                    }
                    TZ(ND) = T1RH;
                    QSS = QSRH;
                    QSD(ND) = QSS;
                }
                TVD(ND) = TZ(ND) * (1.f + 0.608f * QSD(ND));
                if (TVD(ND) > TV0(ND) || ND == 1) {
                    LDB = ND;
                    break;
                }
            }
            if ((P0(LDB) - P0(LFS)) > 50.E2f) {
                for (int ND = LDT; ND >= LDB; --ND) {
                    int ND1 = ND + 1;
                    DDR(ND) = -DMF(KSTART) * DP(ND) / DPDD;
                    DER(ND) = 0.f;
                    DMF(ND) = DMF(ND1) + DDR(ND);
                    TDER = TDER + (QSD(ND) - QD(ND)) * DDR(ND);
                    QD(ND) = QSD(ND);
                    THTAD(ND) = TZ(ND) * gfk_pow(P00 / P0(ND), 0.2854f * (1.f - 0.28f * QD(ND)));
                }
            }
        }
    }
    float PPTFLX, CPR, UPDINC, AINCM2;
    (void)UPDINC; (void)AINCM2;
    if (TDER < 1.f) {
        PPTFLX = TRPPT;
        CPR = TRPPT;
        TDER = 0.f;
        UPDINC = 1.f;
        LDB = LFS;
        for (int NDK = 1; NDK <= LTOP; ++NDK) {
            DMF(NDK) = 0.f; DER(NDK) = 0.f; DDR(NDK) = 0.f;
            THTAD(NDK) = 0.f; TZ(NDK) = 0.f; QD(NDK) = 0.f;
        }
        AINCM2 = 100.f;
    } else {
        float DDINC = -DMFFRC * UMF(KLCL) / DMF(KSTART);
        UPDINC = 1.f;
        if (TDER * DDINC > TRPPT) {
            DDINC = TRPPT / TDER;
        }
        TDER = TDER * DDINC;
        for (NK = LDB; NK <= LFS; ++NK) {
            DMF(NK) = DMF(NK) * DDINC;
            DER(NK) = DER(NK) * DDINC;
            DDR(NK) = DDR(NK) * DDINC;
        }
        CPR = TRPPT;
        PPTFLX = TRPPT - TDER;
        PEFF = PPTFLX / TRPPT;
        if (LDB > 1) {
            for (NK = 1; NK <= LDB - 1; ++NK) {
                DMF(NK) = 0.f; DER(NK) = 0.f; DDR(NK) = 0.f;
                TZ(NK) = 0.f; QD(NK) = 0.f; THTAD(NK) = 0.f;
            }
        }
        for (NK = LFS + 1; NK <= KX; ++NK) {
            DMF(NK) = 0.f; DER(NK) = 0.f; DDR(NK) = 0.f;
            TZ(NK) = 0.f; QD(NK) = 0.f; THTAD(NK) = 0.f;
        }
        for (NK = LDT + 1; NK <= LFS - 1; ++NK) {
            TZ(NK) = 0.f; QD(NK) = 0.f; THTAD(NK) = 0.f;
        }
    }

    /* ---- closure ---- */
    float AINCMX = 1000.f;
    int LMAX = kc_imax(KLCL, LFS);
    for (NK = LC; NK <= LMAX; ++NK) {
        if ((UER(NK) - DER(NK)) > 1.e-5f) {
            float AINCM1 = EMS(NK) / ((UER(NK) - DER(NK)) * TIMEC);
            AINCMX = kc_min(AINCMX, AINCM1);
        }
    }
    float AINC = 1.f;
    if (AINCMX < AINC) AINC = AINCMX;
    float TDER2 = TDER;
    float PPTFL2 = PPTFLX;
    for (NK = 1; NK <= LTOP; ++NK) {
        DETLQ2(NK) = DETLQ(NK); DETIC2(NK) = DETIC(NK); UDR2(NK) = UDR(NK); UER2(NK) = UER(NK);
        DDR2(NK) = DDR(NK); DER2(NK) = DER(NK); UMF2(NK) = UMF(NK); DMF2(NK) = DMF(NK);
    }
    float FABE = 1.f;
    const float STAB = 0.95f;
    int NOITR = 0;
    int ISTOP = 0;
    float EVAC = 0.f;
    (void)EVAC;
    if (ISHALL == 1) {
        float TKEMAX = 5.f;
        EVAC = 0.5f * TKEMAX * 0.1f;
        AINC = EVAC * DPTHMX * DXSQ / (VMFLCL * G * TIMEC) * in.freq * 2.0f;
        TDER = TDER2 * AINC;
        PPTFLX = PPTFL2 * AINC;
        for (NK = 1; NK <= LTOP; ++NK) {
            UMF(NK) = UMF2(NK) * AINC; DMF(NK) = DMF2(NK) * AINC;
            DETLQ(NK) = DETLQ2(NK) * AINC; DETIC(NK) = DETIC2(NK) * AINC;
            UDR(NK) = UDR2(NK) * AINC; UER(NK) = UER2(NK) * AINC;
            DER(NK) = DER2(NK) * AINC; DDR(NK) = DDR2(NK) * AINC;
        }
    }
    int NSTEP = 0;
    float DTIME = 0.f, ABEG, DABE, DFDA, AINCOLD = 0.f, FABEOLD = 0.f;
    for (int NCOUNT = 1; NCOUNT <= 10; ++NCOUNT) {   /* iter */
        float DTT = TIMEC;
        for (NK = 1; NK <= LTOP; ++NK) {
            DOMGDP(NK) = -(UER(NK) - DER(NK) - UDR(NK) - DDR(NK)) * EMSD(NK);
            if (NK > 1) {
                OMG(NK) = OMG(NK - 1) - DP(NK - 1) * DOMGDP(NK - 1);
                float ABSOMG = fabsf(OMG(NK));
                float ABSOMGTC = ABSOMG * TIMEC;
                float FRDP = 0.75f * DP(NK - 1);
                if (ABSOMGTC > FRDP) {
                    float DTT1 = FRDP / ABSOMG;
                    DTT = kc_min(DTT, DTT1);
                }
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            THPA(NK) = THTA0(NK);
            QPA(NK) = Q0(NK);
            NSTEP = kc_nint(TIMEC / DTT + 1);
            DTIME = TIMEC / (float)NSTEP;
            FXM(NK) = OMG(NK) * DXSQ / G;
        }
        for (int NTC = 1; NTC <= NSTEP; ++NTC) {
            for (NK = 1; NK <= LTOP; ++NK) {
                THFXIN(NK) = 0.f; THFXOUT(NK) = 0.f; QFXIN(NK) = 0.f; QFXOUT(NK) = 0.f;
            }
            for (NK = 2; NK <= LTOP; ++NK) {
                if (OMG(NK) <= 0.f) {
                    THFXIN(NK) = -FXM(NK) * THPA(NK - 1);
                    QFXIN(NK) = -FXM(NK) * QPA(NK - 1);
                    THFXOUT(NK - 1) = THFXOUT(NK - 1) + THFXIN(NK);
                    QFXOUT(NK - 1) = QFXOUT(NK - 1) + QFXIN(NK);
                } else {
                    THFXOUT(NK) = FXM(NK) * THPA(NK);
                    QFXOUT(NK) = FXM(NK) * QPA(NK);
                    THFXIN(NK - 1) = THFXIN(NK - 1) + THFXOUT(NK);
                    QFXIN(NK - 1) = QFXIN(NK - 1) + QFXOUT(NK);
                }
            }
            for (NK = 1; NK <= LTOP; ++NK) {
                THPA(NK) = THPA(NK) + (THFXIN(NK) + UDR(NK) * THTAU(NK) + DDR(NK) *
                           THTAD(NK) - THFXOUT(NK) - (UER(NK) - DER(NK)) * THTA0(NK)) *
                           DTIME * EMSD(NK);
                QPA(NK) = QPA(NK) + (QFXIN(NK) + UDR(NK) * QDT(NK) + DDR(NK) * QD(NK) -
                          QFXOUT(NK) - (UER(NK) - DER(NK)) * Q0(NK)) * DTIME * EMSD(NK);
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            THTAG(NK) = THPA(NK);
            QG(NK) = QPA(NK);
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            if (QG(NK) < 0.f) {
                if (NK == 1) {
                    /* wrf_error_fatal('QG, QG(NK).LT.0') */
                    *status |= KFC_FATAL_QG_SURFACE;
                    o.ishall = 2; o.wlcl = WLCL;
                    return;
                }
                int NK1 = NK + 1;
                if (NK == LTOP) {
                    NK1 = KLCL;
                }
                float TMA = QG(NK1) * EMS(NK1);
                float TMB = QG(NK - 1) * EMS(NK - 1);
                float TMM = (QG(NK) - 1.E-9f) * EMS(NK);
                float BCOEFF = -TMM / ((TMA * TMA) / TMB + TMB);
                float ACOEFF = BCOEFF * TMA / TMB;
                TMB = TMB * (1.f - BCOEFF);
                TMA = TMA * (1.f - ACOEFF);
                QG(NK) = 1.E-9f;
                QG(NK1) = TMA * EMSD(NK1);
                QG(NK - 1) = TMB * EMSD(NK - 1);
            }
        }
        float TOPOMG = (UDR(LTOP) - UER(LTOP)) * DP(LTOP) * EMSD(LTOP);
        if (fabsf(TOPOMG - OMG(LTOP)) > 1.E-3f) {
            ISTOP = 1;
            break;   /* EXIT iter */
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            EXN(NK) = gfk_pow(P00 / P0(NK), 0.2854f * (1.f - 0.28f * QG(NK)));
            TG(NK) = THTAG(NK) / EXN(NK);
            TVG(NK) = TG(NK) * (1.f + 0.608f * QG(NK));
        }
        if (ISHALL == 1) {
            break;   /* EXIT iter */
        }
        TMIX = 0.f;
        QMIX = 0.f;
        for (NK = LC; NK <= KCLDLAYER; ++NK) {
            TMIX = TMIX + DP(NK) * TG(NK);
            QMIX = QMIX + DP(NK) * QG(NK);
        }
        TMIX = TMIX / DPTHMX;
        QMIX = QMIX / DPTHMX;
        float ES = ALIQ * gfk_exp((TMIX * BLIQ - CLIQ) / (TMIX - DLIQ));
        float QSS = 0.622f * ES / (PMIX - ES);
        if (QMIX > QSS) {
            float RL = XLV0 - XLV1 * TMIX;
            float CPM = CP * (1.f + 0.887f * QMIX);
            float DSSDT = QSS * (CLIQ - BLIQ * DLIQ) / ((TMIX - DLIQ) * (TMIX - DLIQ));
            float DQ = (QMIX - QSS) / (1.f + RL * DSSDT / CPM);
            TMIX = TMIX + RL / CP * DQ;
            QMIX = QMIX - DQ;
            TLCL = TMIX;
        } else {
            QMIX = kc_max(QMIX, 0.f);
            EMIX = QMIX * PMIX / (0.622f + QMIX);
            astrt = 1.e-3f;
            float binc = 0.075f;
            a1 = EMIX / ALIQ;
            tp = (a1 - astrt) / binc;
            indlu = (int)tp + 1;
            value = (float)(indlu - 1) * binc + astrt;
            aintrp = (a1 - value) / binc;
            TLOG = aintrp * tb.alu[indlu] + (1 - aintrp) * tb.alu[indlu - 1];
            TDPT = (CLIQ - DLIQ * TLOG) / (BLIQ - TLOG);
            TLCL = TDPT - (.212f + 1.571E-3f * (TDPT - T00) - 4.36E-4f * (TMIX - T00)) * (TMIX - TDPT);
            TLCL = kc_min(TLCL, TMIX);
        }
        TVLCL = TLCL * (1.f + 0.608f * QMIX);
        ZLCL = ZMIX + (TLCL - TMIX) / GDRY;
        for (NK = LC; NK <= KL; ++NK) {
            KLCL = NK;
            if (ZLCL <= Z0(NK)) break;
        }
        K = KLCL - 1;
        DLP = (ZLCL - Z0(K)) / (Z0(KLCL) - Z0(K));
        TENV = TG(K) + (TG(KLCL) - TG(K)) * DLP;
        QENV = QG(K) + (QG(KLCL) - QG(K)) * DLP;
        TVEN = TENV * (1.f + 0.608f * QENV);
        PLCL = P0(K) + (P0(KLCL) - P0(K)) * DLP;
        THETEU(K) = TMIX * gfk_pow(1.E5f / PMIX, 0.2854f * (1.f - 0.28f * QMIX)) *
                    gfk_exp((3374.6525f / TLCL - 2.5403f) * QMIX * (1.f + 0.81f * QMIX));
        ABEG = 0.f;
        for (NK = K; NK <= LTOPM1; ++NK) {
            int NK1 = NK + 1;
            THETEU(NK1) = THETEU(NK);
            {
                float ts, qs;
                kc_tpmix2dd(tb, P0(NK1), THETEU(NK1), &ts, &qs, status);
                TGU(NK1) = ts; QGU(NK1) = qs;
            }
            TVQU(NK1) = TGU(NK1) * (1.f + 0.608f * QGU(NK1) - QLIQ(NK1) - QICE(NK1));
            float DZZ;
            if (NK == K) {
                DZZ = Z0(KLCL) - ZLCL;
                DILBE = ((TVLCL + TVQU(NK1)) / (TVEN + TVG(NK1)) - 1.f) * DZZ;
            } else {
                DZZ = DZA(NK);
                DILBE = ((TVQU(NK) + TVQU(NK1)) / (TVG(NK) + TVG(NK1)) - 1.f) * DZZ;
            }
            if (DILBE > 0.f) ABEG = ABEG + DILBE * G;
            THTEEG(NK1) = kc_envirtht(tb, P0(NK1), TG(NK1), QG(NK1), ALIQ, BLIQ, CLIQ, DLIQ);
            THETEU(NK1) = THETEU(NK1) * DDILFRC(NK1) + THTEEG(NK1) * (1.f - DDILFRC(NK1));
        }
        if (NOITR == 1) {
            break;   /* EXIT iter */
        }
        DABE = kc_max(ABE - ABEG, 0.1f * ABE);
        FABE = ABEG / ABE;
        if (FABE > 1.f && ISHALL == 0) {
            ISHALL = 2;
            o.ishall = ISHALL; o.wlcl = WLCL;
            return;
        }
        if (NCOUNT != 1) {
            if (fabsf(AINC - AINCOLD) < 0.0001f) {
                NOITR = 1;
                AINC = AINCOLD;
                continue;   /* CYCLE iter */
            }
            DFDA = (FABE - FABEOLD) / (AINC - AINCOLD);
            if (DFDA > 0.f) {
                NOITR = 1;
                AINC = AINCOLD;
                continue;
            }
        }
        AINCOLD = AINC;
        FABEOLD = FABE;
        if (AINC / AINCMX > 0.999f && FABE > 1.05f - STAB) {
            break;
        }
        if ((FABE <= 1.05f - STAB && FABE >= 0.95f - STAB) || NCOUNT == 10) {
            break;
        } else {
            if (NCOUNT > 10) {
                break;
            }
            if (FABE == 0.f) {
                AINC = AINC * 0.5f;
            } else {
                if (DABE < 1.e-4f) {
                    NOITR = 1;
                    AINC = AINCOLD;
                    continue;
                } else {
                    AINC = AINC * STAB * ABE / DABE;
                }
            }
            AINC = kc_min(AINCMX, AINC);
            if (AINC < 0.05f) {
                ISHALL = 2;
                o.ishall = ISHALL; o.wlcl = WLCL;
                return;
            }
            TDER = TDER2 * AINC;
            PPTFLX = PPTFL2 * AINC;
            for (NK = 1; NK <= LTOP; ++NK) {
                UMF(NK) = UMF2(NK) * AINC; DMF(NK) = DMF2(NK) * AINC;
                DETLQ(NK) = DETLQ2(NK) * AINC; DETIC(NK) = DETIC2(NK) * AINC;
                UDR(NK) = UDR2(NK) * AINC; UER(NK) = UER2(NK) * AINC;
                DER(NK) = DER2(NK) * AINC; DDR(NK) = DDR2(NK) * AINC;
            }
        }
    }   /* END DO iter */
    if (ISTOP == 1) {
        /* IPRNT = .TRUE.; the diagnostic block ends in
         * wrf_error_fatal('KAIN-FRITSCH, istop=1, diags') (:3397-3399) */
        *status |= KFC_FATAL_MASS_BUDGET;
        o.ishall = 2; o.wlcl = WLCL;
        return;
    }

    /* ---- hydrometeors ---- */
    float FRC2;
    if (CPR > 0.f) {
        FRC2 = PPTFLX / (CPR * AINC);
    } else {
        FRC2 = 0.f;
    }
    for (NK = 1; NK <= LTOP; ++NK) {
        QLPA(NK) = QL0;
        QIPA(NK) = QI0;
        QRPA(NK) = QR0;
        QSPA(NK) = QS0;
        RAINFB(NK) = PPTLIQ(NK) * AINC * FBFRC * FRC2;
        SNOWFB(NK) = PPTICE(NK) * AINC * FBFRC * FRC2;
    }
    for (int NTC = 1; NTC <= NSTEP; ++NTC) {
        for (NK = 1; NK <= LTOP; ++NK) {
            QLFXIN(NK) = 0.f; QLFXOUT(NK) = 0.f; QIFXIN(NK) = 0.f; QIFXOUT(NK) = 0.f;
            QRFXIN(NK) = 0.f; QRFXOUT(NK) = 0.f; QSFXIN(NK) = 0.f; QSFXOUT(NK) = 0.f;
        }
        for (NK = 2; NK <= LTOP; ++NK) {
            if (OMG(NK) <= 0.f) {
                QLFXIN(NK) = -FXM(NK) * QLPA(NK - 1);
                QIFXIN(NK) = -FXM(NK) * QIPA(NK - 1);
                QRFXIN(NK) = -FXM(NK) * QRPA(NK - 1);
                QSFXIN(NK) = -FXM(NK) * QSPA(NK - 1);
                QLFXOUT(NK - 1) = QLFXOUT(NK - 1) + QLFXIN(NK);
                QIFXOUT(NK - 1) = QIFXOUT(NK - 1) + QIFXIN(NK);
                QRFXOUT(NK - 1) = QRFXOUT(NK - 1) + QRFXIN(NK);
                QSFXOUT(NK - 1) = QSFXOUT(NK - 1) + QSFXIN(NK);
            } else {
                QLFXOUT(NK) = FXM(NK) * QLPA(NK);
                QIFXOUT(NK) = FXM(NK) * QIPA(NK);
                QRFXOUT(NK) = FXM(NK) * QRPA(NK);
                QSFXOUT(NK) = FXM(NK) * QSPA(NK);
                QLFXIN(NK - 1) = QLFXIN(NK - 1) + QLFXOUT(NK);
                QIFXIN(NK - 1) = QIFXIN(NK - 1) + QIFXOUT(NK);
                QRFXIN(NK - 1) = QRFXIN(NK - 1) + QRFXOUT(NK);
                QSFXIN(NK - 1) = QSFXIN(NK - 1) + QSFXOUT(NK);
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            QLPA(NK) = QLPA(NK) + (QLFXIN(NK) + DETLQ(NK) - QLFXOUT(NK)) * DTIME * EMSD(NK);
            QIPA(NK) = QIPA(NK) + (QIFXIN(NK) + DETIC(NK) - QIFXOUT(NK)) * DTIME * EMSD(NK);
            QRPA(NK) = QRPA(NK) + (QRFXIN(NK) - QRFXOUT(NK) + RAINFB(NK)) * DTIME * EMSD(NK);
            QSPA(NK) = QSPA(NK) + (QSFXIN(NK) - QSFXOUT(NK) + SNOWFB(NK)) * DTIME * EMSD(NK);
        }
    }
    for (NK = 1; NK <= LTOP; ++NK) {
        a.qlg(NK) = QLPA(NK);
        a.qig(NK) = QIPA(NK);
        QRG(NK) = QRPA(NK);
        QSG(NK) = QSPA(NK);
    }
    /* (the IPRNT diagnostics :3268-3402 write only to unit 98) */
    o.raincv = DT * PPTFLX * (1.f - FBFRC) / DXSQ;
    /* (moisture-budget check :3415-3449: diagnostic writes only; its ISTOP
     * is set after the only fatal test that reads ISTOP) */
    if (TADVEC < TIMEC) NIC = kc_nint(TADVEC / DT);
    o.nca = (float)NIC * DT;
    if (ISHALL == 1) {
        TIMEC = TIMEC_SHALL;
        o.nca = (float)kc_nint(TIMEC_SHALL / DT) * DT;
    }
    for (int k2 = 1; k2 <= KX; ++k2) {
        if (in.warm_rain) {
            float CPM = CP * (1.f + 0.887f * QG(k2));
            TG(k2) = TG(k2) - (a.qig(k2) + QSG(k2)) * RLF / CPM;
            a.dqcdt(k2) = (a.qlg(k2) + a.qig(k2) - QL0 - QI0) / TIMEC;
            a.dqidt(k2) = 0.f;
            a.dqrdt(k2) = (QRG(k2) + QSG(k2) - QR0 - QS0) / TIMEC;
            a.dqsdt(k2) = 0.f;
        } else if (!in.f_qs) {
            float CPM = CP * (1.f + 0.887f * QG(k2));
            if (k2 <= ML) {
                TG(k2) = TG(k2) - (a.qig(k2) + QSG(k2)) * RLF / CPM;
            } else if (k2 > ML) {
                TG(k2) = TG(k2) + (a.qlg(k2) + QRG(k2)) * RLF / CPM;
            }
            a.dqcdt(k2) = (a.qlg(k2) + a.qig(k2) - QL0 - QI0) / TIMEC;
            a.dqidt(k2) = 0.f;
            a.dqrdt(k2) = (QRG(k2) + QSG(k2) - QR0 - QS0) / TIMEC;
            a.dqsdt(k2) = 0.f;
        } else {
            a.dqcdt(k2) = (a.qlg(k2) - QL0) / TIMEC;
            a.dqsdt(k2) = (QSG(k2) - QS0) / TIMEC;
            a.dqrdt(k2) = (QRG(k2) - QR0) / TIMEC;
            if (in.f_qi) {
                a.dqidt(k2) = (a.qig(k2) - QI0) / TIMEC;
            } else {
                a.dqsdt(k2) = a.dqsdt(k2) + (a.qig(k2) - QI0) / TIMEC;
            }
        }
        a.dtdt(k2) = (TG(k2) - T0(k2)) / TIMEC;
        a.dqdt(k2) = (QG(k2) - Q0(k2)) / TIMEC;
    }
    o.raincv = DT * PPTFLX * (1.f - FBFRC) / DXSQ;
    if (ISHALL < 2) {
        o.cutop = (float)LTOP;
        o.cubot = (float)LCL;
        o.updfra = AU0 * AINC / DXSQ;
        o.wulcl = WU(KLCL);
        for (int k2 = kts; k2 <= kte; ++k2) {
            a.wup(k2) = WU(k2);
            a.qc1d(k2) = QLIQ(k2);
            a.qi1d(k2) = QICE(k2);
            a.qndrop1d(k2) = qndropbb(k2);
        }
        for (int k2 = kts; k2 <= LTOP - 1; ++k2) a.umfout(k2) = kc_maxc(0.0f, UMF(k2) / DXSQ);
        for (int k2 = kts; k2 <= LTOP; ++k2) {
            a.uerout(k2) = kc_maxc(0.0f, UER(k2) / DXSQ);
            a.udrout(k2) = kc_maxc(0.0f, UDR(k2) / DXSQ);
        }
        for (int k2 = kts; k2 <= LTOP - 1; ++k2) a.dmfout(k2) = kc_minc(0.0f, DMF(k2 + 1) / DXSQ);
        for (int k2 = kts; k2 <= LTOP; ++k2) {
            a.derout(k2) = kc_maxc(0.0f, -DER(k2) / DXSQ);
            a.ddrout(k2) = kc_maxc(0.0f, DDR(k2) / DXSQ);
        }
    }
    if (ISHALL == 1 && (z_at_w1d(LCL) - in.pblh) > 500.f) ISHALL = 2;
    o.ishall = ISHALL;
    o.wlcl = WLCL;
}

/* ------------------------------------------------------------------------
 * cupSlopeSigma (:4614-4912)
 *
 * The sf_sfclay_physics SELECT CASE (:4680-4737) computes only
 * UnstableOrNeutral, which nothing reads, and is not transcribed; the
 * kpbl == 0 fatal (:4789) is unreachable behind the kpbl <= 2 test.
 * ---------------------------------------------------------------------- */
struct KcSlope {
    float slopesfc, slopeez, sigmasfc, sigmaez, wstar;
    bool cupflag;
};

__device__ void kc_slope_sigma(const KcCol& z, const KcCol& th, const KcCol& qv_curr,
                               const KcCol& rho, float ht, float hfx, float xland, float qfxin,
                               float pblh, int kpbl, int kte, KcSlope& s, float* shall)
{
    float qfx;
    if (fabsf(qfxin) < KC_INVXLV) {
        qfx = kc_sign(KC_INVXLV, qfxin);
    } else {
        qfx = qfxin;
    }
    if (xland == 1.f) {
        if (kpbl <= 2 || hfx < 100.f) {
            s.cupflag = false;
            s.slopesfc = 0.f; s.slopeez = 0.f; s.sigmasfc = 0.f; s.sigmaez = 0.f;
            *shall = 2.f;
            return;
        } else {
            s.cupflag = true;
        }
    } else {
        if (kpbl <= 2 || hfx < 1.f) {
            s.cupflag = false;
            s.slopesfc = 0.f; s.slopeez = 0.f; s.sigmasfc = 0.f; s.sigmaez = 0.f;
            *shall = 2.f;
            return;
        } else {
            s.cupflag = true;
        }
    }
    int kpblmid = kc_imax(1, kpbl / 2);
    float zagl_mid = z(kpblmid) - ht;   /* zagl(k) = z(k) - ht, :4774-4776 */
    float flux = (1.f + KC_EP1 * qv_curr(1)) * hfx / rho(1) / KC_CP +
                 KC_EP1 * th(1) * qfx / rho(1);
    float tvcon = 1.f + KC_EP1 * qv_curr(kpblmid);
    float thv = th(kpblmid) * tvcon;
    s.wstar = gfk_pow(KC_G * pblh * flux / thv, __int_as_float(0x3EAAAAAB));  /* 1./3. */
    /* (:4805-4809 form a surface-layer slope that :4812 replaces with the
     * Bowen ratio; nothing reads them) */
    s.slopesfc = hfx / (KC_XLV * qfx) * KC_XLV / KC_CP;
    int k2 = kc_imin(kpbl + 2, kte);
    tvcon = 1.f + KC_EP1 * qv_curr(k2);
    thv = th(k2) * tvcon;
    tvcon = 1.f + KC_EP1 * qv_curr(kpblmid);
    float thv2 = th(kpblmid) * tvcon;
    float qdiff = qv_curr(k2) - qv_curr(kpblmid);
    if (fabsf(qdiff) < KC_REALLYSMALL) {
        qdiff = kc_sign(KC_REALLYSMALL, qdiff);
    }
    s.slopeez = (thv - thv2) / qdiff;
    s.sigmaez = flux / s.wstar * (2.f + (8.2e-4f) * gfk_pow(zagl_mid / pblh, -1.8f));
    flux = qfx / rho(1);
    s.sigmasfc = flux / s.wstar * (2.3f + 1.1e-2f * gfk_pow(zagl_mid / pblh, -1.6f));
}

/* ------------------------------------------------------------------------
 * cup_jfd (:4470-4610).  jfd(i,j) lives in the workspace PDF region at
 * e = (j-1)*numBins + (i-1); th_perturb and r_perturb follow it.
 * ---------------------------------------------------------------------- */
__device__ __forceinline__ float& kc_jfd_at(const KcWs& ws, int nb, int i, int j) {
    return ws.pdf((j - 1) * nb + (i - 1));
}
__device__ __forceinline__ float& kc_thp_at(const KcWs& ws, int nb, int i) {
    return ws.pdf(nb * nb + (i - 1));
}
__device__ __forceinline__ float& kc_rp_at(const KcWs& ws, int nb, int j) {
    return ws.pdf(nb * nb + nb + (j - 1));
}

__device__ void kc_jfd(KcSlope& s, int numbins, float thbinsize, float rbinsize, const KcWs& ws)
{
    s.slopesfc = kc_sign(kc_max(fabsf(s.slopesfc), 1e-15f), s.slopesfc);
    if (s.slopeez > 2000) {
        s.slopeez = 2000.0f;
    } else if (s.slopeez < -2000) {
        s.slopeez = -2000.0f;
    } else if (s.slopeez < 10 && s.slopeez > 0) {
        s.slopeez = 10.0f;
    } else if (s.slopeez < 0 && s.slopeez > -10.0f) {
        s.slopeez = -10.0f;
    }
    s.sigmasfc = kc_sign(kc_max(fabsf(s.sigmasfc), 1e-15f), s.sigmasfc);
    s.sigmaez = kc_sign(kc_max(fabsf(s.sigmaez), 1e-15f), s.sigmaez);
    int centerbin = numbins / 2 + 1;
    float dslope = kc_sign(kc_max(fabsf(s.slopeez - s.slopesfc), 1e-15f), s.slopeez - s.slopesfc);
    float jfdchecksum = 0.f;
    for (int j = 1; j <= numbins; ++j) {
        kc_rp_at(ws, numbins, j) = rbinsize * (float)(j - centerbin);
        for (int i = 1; i <= numbins; ++i) {
            kc_thp_at(ws, numbins, i) = thbinsize * (float)(i - centerbin);
            float thp = kc_thp_at(ws, numbins, i);
            float rp = kc_rp_at(ws, numbins, j);
            float c = s.slopeez * (thp - s.slopesfc * rp) / dslope;
            float m = (thp - s.slopeez * rp) / dslope;
            float cterm = c / s.sigmaez;
            if (fabsf(cterm) > KC_BIGCHECK) {
                /* DIVERGENCE (see the file header): WRF stores
                 * sign(bigcheck, cterm) unsquared. */
                cterm = KC_BIGCHECK;
            } else {
                cterm = cterm * cterm;
            }
            float mterm = m / s.sigmasfc;
            if (fabsf(mterm) > KC_BIGCHECK) {
                mterm = KC_BIGCHECK;
            } else {
                mterm = mterm * mterm;
            }
            kc_jfd_at(ws, numbins, i, j) = gfk_exp(-0.5f * (mterm + cterm));
            jfdchecksum = jfdchecksum + kc_jfd_at(ws, numbins, i, j);
        }
    }
    if (jfdchecksum != 0.f) {
        for (int j = 1; j <= numbins; ++j)
            for (int i = 1; i <= numbins; ++i)
                kc_jfd_at(ws, numbins, i, j) = kc_jfd_at(ws, numbins, i, j) / jfdchecksum;
    }
}

/* ------------------------------------------------------------------------
 * adjust_mfentdet_kfcup (:5133-5284), less the diagnostics-only tmpg/tmph
 * bookkeeping.
 * ---------------------------------------------------------------------- */
__device__ void kc_adjust_mf(int kts, int kte, int kcutop, int ishall,
                             const KcCol& umfout, const KcCol& uerout, const KcCol& udrout,
                             const KcCol& dmfout, const KcCol& derout, const KcCol& ddrout)
{
    float tmpa, tmpb, tmpc;
    if (ishall == 0) {
        for (int k = kcutop; k <= kte; ++k) dmfout(k) = 0.0f;
        if (kcutop < kte) {
            for (int k = kcutop + 1; k <= kte; ++k) { derout(k) = 0.0f; ddrout(k) = 0.0f; }
        }
        for (int k = kts; k <= kcutop; ++k) {
            if (k > kts) {
                tmpa = dmfout(k) - dmfout(k - 1);
            } else {
                tmpa = dmfout(k);
            }
            tmpb = derout(k) - ddrout(k);
            tmpc = tmpa - tmpb;
            if (tmpc > 0.0f) {
                if (derout(k) < ddrout(k) * 0.05f) {
                    ddrout(k) = kc_maxc(0.0f, ddrout(k) - tmpc);
                    tmpb = derout(k) - ddrout(k);
                    tmpc = tmpa - tmpb;
                    derout(k) = derout(k) + tmpc;
                } else {
                    derout(k) = derout(k) + tmpc;
                }
            } else {
                if (ddrout(k) <= derout(k) * 0.05f) {
                    derout(k) = kc_maxc(0.0f, derout(k) + tmpc);
                    tmpb = derout(k) - ddrout(k);
                    tmpc = tmpa - tmpb;
                    ddrout(k) = ddrout(k) - tmpc;
                } else {
                    ddrout(k) = ddrout(k) - tmpc;
                }
            }
        }
    }
    if (ishall == 0 || ishall == 1) {
        for (int k = kcutop; k <= kte; ++k) umfout(k) = 0.0f;
        if (kcutop < kte) {
            for (int k = kcutop + 1; k <= kte; ++k) { uerout(k) = 0.0f; udrout(k) = 0.0f; }
        }
        for (int k = kts; k <= kcutop; ++k) {
            if (k > kts) {
                tmpa = umfout(k) - umfout(k - 1);
            } else {
                tmpa = umfout(k);
            }
            tmpb = uerout(k) - udrout(k);
            tmpc = tmpa - tmpb;
            if (tmpc > 0.0f) {
                if (uerout(k) < udrout(k) * 0.05f) {
                    udrout(k) = kc_maxc(0.0f, udrout(k) - tmpc);
                    tmpb = uerout(k) - udrout(k);
                    tmpc = tmpa - tmpb;
                    uerout(k) = uerout(k) + tmpc;
                } else {
                    uerout(k) = uerout(k) + tmpc;
                }
            } else {
                if (udrout(k) <= uerout(k) * 0.05f) {
                    uerout(k) = kc_maxc(0.0f, uerout(k) + tmpc);
                    tmpb = uerout(k) - udrout(k);
                    tmpc = tmpa - tmpb;
                    udrout(k) = udrout(k) - tmpc;
                } else {
                    udrout(k) = udrout(k) - tmpc;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * cupCloudFraction (:4297-4466).  lnTerms is not formed (file header).
 * ---------------------------------------------------------------------- */
struct KcCloudOut {
    float taucloud, tactive, tstar, lnint;
};

__device__ void kc_cloud_fraction(const KcCol& qlg, const KcCol& qv1d, const KcCol& t1d,
                                  const KcCol& z1d, const KcCol& p1d, int kcubot, int kcutop,
                                  int ishall, float wstar, float wparcel, float pblh,
                                  float activefrac, const KcCol& cldfra, const KcCol& satdef,
                                  int kts, int kte, const KcCol& mfup_cup, KcCloudOut& co)
{
    if (ishall == 2) {
        for (int k = kts; k <= kte; ++k) cldfra(k) = 0.f;
    } else if (ishall == 0) {
        for (int k = kts; k <= kte; ++k) cldfra(k) = 0.f;
        for (int k = kcubot; k <= kcutop; ++k) {
            cldfra(k) = kc_maxc(0.f, kc_min(0.1f * gfk_log(1.f + 675.f * mfup_cup(k)), 1.f));
        }
        co.tstar = pblh / wstar;
    } else {
        co.tstar = pblh / wstar;
        float intqc = 0.f;
        float intsatdef = 0.f;
        float zsum = 0.f;
        int kcutop_p1 = kc_imin(kcutop + 1, kte);
        for (int k = kcubot; k <= kcutop_p1; ++k) {
            float rs = kc_find_rs(t1d(k), p1d(k));
            satdef(k) = kc_max(rs - qv1d(k), 1.0e-20f);
        }
        for (int k = kcubot; k <= kcutop; ++k) {
            int kp1 = kc_imin(k + 1, kte - 1);
            float deltaz = z1d(kp1) - z1d(k);
            zsum = zsum + deltaz;
            float rs = kc_find_rs(t1d(k), p1d(k));
            satdef(k) = kc_max(rs - qv1d(k), 1e-20f);
            intqc = 0.5f * (qlg(k) + qlg(kp1)) * deltaz + intqc;
            intsatdef = 0.5f * (satdef(k) + satdef(kp1)) * deltaz + intsatdef;
        }
        float cp = kc_find_cp(qv1d(kcubot));
        float rs = kc_find_rs(t1d(kcubot), p1d(kcubot));
        float lv = KC_XLV0 - KC_XLV1 * t1d(kcubot);
        float gamma_ = (lv * lv) * rs / (cp * KC_RV * (t1d(kcubot) * t1d(kcubot)));
        co.lnint = gfk_log(1.0f + (1.0f + gamma_) * intqc / intsatdef);
        co.lnint = kc_max(co.lnint, 1.0f);
        co.taucloud = kc_min(co.tstar * co.lnint, 1800.f);
        if (wparcel > 0) {
            co.tactive = z1d(kcutop) / wparcel;
        } else {
            co.tactive = z1d(kcutop) / wstar;
        }
        for (int k = kts; k <= kte; ++k) cldfra(k) = 0.f;
        for (int k = kcubot; k <= kcutop; ++k) {
            cldfra(k) = activefrac * co.taucloud / co.tactive;
            cldfra(k) = kc_max(cldfra(k), 0.01f);
            cldfra(k) = kc_min(cldfra(k), 1.f);
        }
    }
}

/* findIndex (:4930-4945) over z_at_w1d(kts:kte) */
__device__ __forceinline__ int kc_find_index(float value, const KcCol& list, int n) {
    for (int i = 1; i <= n; ++i) {
        if (value <= list(i)) return i;
    }
    return 0;
}

/* ========================================================================
 * KF_cup_CPS (:92-1249) for one column, plus the driver's output algebra.
 * ====================================================================== */

/* Level fields in, (nz, ncol) each: the scheme's 1-D column inputs. */
#define KCI_U 0
#define KCI_V 1
#define KCI_TH 2
#define KCI_T 3
#define KCI_DZ8W 4
#define KCI_P 5
#define KCI_PI 6
#define KCI_RHO 7
#define KCI_QV 8
#define KCI_Z 9
#define KCI_ZW 10
#define KCI_W0AVG 11
#define KCI_NLEV 12
/* Surface fields in, (ncol) each. */
#define KCS_HT 0
#define KCS_HFX 1
#define KCS_QFX 2
#define KCS_XLAND 3
#define KCS_PBLH 4
#define KCS_NSCA 5
/* Held column state, (ncol) each, read and written in place. */
#define KCH_NCA 0
#define KCH_RAINCV 1
#define KCH_CUTOP 2
#define KCH_CUBOT 3
#define KCH_SHALL 4
#define KCH_SLOPESFC 5
#define KCH_SLOPEEZ 6
#define KCH_SIGMASFC 7
#define KCH_SIGMAEZ 8
#define KCH_TAUCLOUD 9
#define KCH_TACTIVE 10
#define KCH_TSTAR 11
#define KCH_LNINT 12
#define KCH_ACTIVEFRAC 13
#define KCH_WCLOUDBASE 14
#define KCH_WACT 15
#define KCH_WULCL 16
#define KCH_TCLOUD 17
#define KCH_NSCA 18
/* Integer column state (0/1). */
#define KCF_CUPFLAG 0
#define KCF_CU_ACT 1
#define KCF_NI 2
/* Level outputs and held level state, (nz, ncol) each, written in place. */
#define KCO_RTHCUTEN 0
#define KCO_RQVCUTEN 1
#define KCO_RQCCUTEN 2
#define KCO_RQRCUTEN 3
#define KCO_RQICUTEN 4
#define KCO_RQSCUTEN 5
#define KCO_CLDFRA_CUP 6
#define KCO_WUP 7
#define KCO_QC_IC 8
#define KCO_QNDROP_IC 9
#define KCO_QC_IU 10
#define KCO_FCVT_QC_PR 11
#define KCO_FCVT_QC_QI 12
#define KCO_FCVT_QI_PR 13
#define KCO_MFUP 14
#define KCO_MFUP_ENT 15
#define KCO_MFDN 16
#define KCO_MFDN_ENT 17
#define KCO_UPDFRA 18
#define KCO_NLEV 19

/* Launch flags. */
#define KCFLAG_WARM_RAIN 1
#define KCFLAG_F_QR 2
#define KCFLAG_F_QI 4
#define KCFLAG_F_QS 8

/* The per-thread workspace extent, in floats. */
__device__ __forceinline__ size_t kc_ws_floats_per_thread(int nz, int numbins) {
    return (size_t)KFCUP_WS_SLOTS * (size_t)(nz + 1)
        + (size_t)numbins * (size_t)numbins + 2 * (size_t)numbins;
}

__device__ void kc_column(
    int col, int lt,
    float* __restrict__ lev_in, const float* __restrict__ sca_in, const int* __restrict__ kpbl_in,
    float* __restrict__ held, int* __restrict__ iheld, float* __restrict__ lev_out,
    const KcTable& tb, float dt, float dx, int ntst, int numbins, float thbinsize, float rbinsize,
    float mindeepfreq, float minshallowfreq, int flags,
    float* __restrict__ ws_all, int* __restrict__ status_out, int nz, int ncol)
{
    const int kts = 1, kte = nz;
    const size_t nc = (size_t)ncol;
    const size_t nzs = (size_t)nz;
#define LEVIN(f) (KcCol{lev_in + (size_t)(f) * nzs * nc + (size_t)col, nc})
#define LEVOUT(f) (KcCol{lev_out + (size_t)(f) * nzs * nc + (size_t)col, nc})
#define HELD(f) (held[(size_t)(f) * nc + (size_t)col])
#define IHELD(f) (iheld[(size_t)(f) * nc + (size_t)col])

    const KcCol U1D = LEVIN(KCI_U), V1D = LEVIN(KCI_V), TH1D = LEVIN(KCI_TH), T1D = LEVIN(KCI_T);
    const KcCol DZ1D = LEVIN(KCI_DZ8W), P1D = LEVIN(KCI_P), PI1D = LEVIN(KCI_PI);
    const KcCol RHO1D = LEVIN(KCI_RHO), QV1D = LEVIN(KCI_QV), Z1D = LEVIN(KCI_Z);
    const KcCol ZW1D = LEVIN(KCI_ZW), W0AVG1D = LEVIN(KCI_W0AVG);
    const float ht = sca_in[KCS_HT * nc + col], hfx = sca_in[KCS_HFX * nc + col];
    const float qfx = sca_in[KCS_QFX * nc + col], xland = sca_in[KCS_XLAND * nc + col];
    const float pblh = sca_in[KCS_PBLH * nc + col];
    const int kpbl = kpbl_in[col];
    const bool warm_rain = (flags & KCFLAG_WARM_RAIN) != 0;
    const bool f_qr = (flags & KCFLAG_F_QR) != 0;
    const bool f_qi = (flags & KCFLAG_F_QI) != 0;
    const bool f_qs = (flags & KCFLAG_F_QS) != 0;
    int status = 0;

    /* ---- the NCA hold (:516-519) ---- */
    if (HELD(KCH_NCA) >= 0.5f * dt) {
        IHELD(KCF_CU_ACT) = 0;
        return;
    }
    IHELD(KCF_CU_ACT) = 1;

    KcWs ws;
    ws.kp = nz + 1;
    {
        const size_t per_thread = kc_ws_floats_per_thread(nz, numbins);
        const size_t warp = (size_t)lt / KC_LANES;
        ws.base = ws_all + warp * per_thread * KC_LANES + (size_t)(lt % KC_LANES);
    }
    /* driver arrays: slots 87..147 */
    KcCol DQDT = ws.col(87), DQIDT = ws.col(88), DQCDT = ws.col(89), DQRDT = ws.col(90);
    KcCol DQSDT = ws.col(91), DTDT = ws.col(92), QLG = ws.col(93), QIG = ws.col(94);
    KcCol UMFOUT = ws.col(95), UEROUT = ws.col(96), UDROUT = ws.col(97), DMFOUT = ws.col(98);
    KcCol DEROUT = ws.col(99), DDROUT = ws.col(100), WUP = ws.col(101), QNDROP1D = ws.col(102);
    KcCol QC1D = ws.col(103), QI1D = ws.col(104), FCVT_QC_TO_PR = ws.col(105);
    KcCol FCVT_QC_TO_QI = ws.col(106), FCVT_QI_TO_PR = ws.col(107);
    KcCol DQDT_DEEP = ws.col(108), DQIDT_DEEP = ws.col(109), DQCDT_DEEP = ws.col(110);
    KcCol DQRDT_DEEP = ws.col(111), DQSDT_DEEP = ws.col(112), DTDT_DEEP = ws.col(113);
    KcCol DQDT_SHALL = ws.col(114), DQIDT_SHALL = ws.col(115), DQCDT_SHALL = ws.col(116);
    KcCol DQRDT_SHALL = ws.col(117), DQSDT_SHALL = ws.col(118), DTDT_SHALL = ws.col(119);
    KcCol QLG_SHALL = ws.col(120), QIG_SHALL = ws.col(121);
    KcCol QNDROP_IC_DEEP = ws.col(122), QNDROP_IC_SHALL = ws.col(123);
    KcCol QC_IC_DEEP = ws.col(124), QC_IC_SHALL = ws.col(125);
    KcCol QI_IC_DEEP = ws.col(126), QI_IC_SHALL = ws.col(127);
    KcCol FCVT_QC_TO_PR_DEEP = ws.col(128), FCVT_QC_TO_QI_DEEP = ws.col(129);
    KcCol FCVT_QI_TO_PR_DEEP = ws.col(130), FCVT_QC_TO_PR_SHALL = ws.col(131);
    KcCol FCVT_QC_TO_QI_SHALL = ws.col(132), FCVT_QI_TO_PR_SHALL = ws.col(133);
    KcCol CUMSHALLFREQ1D = ws.col(134);
    KcCol UMF_DEEP = ws.col(135), UER_DEEP = ws.col(136), UDR_DEEP = ws.col(137);
    KcCol DMF_DEEP = ws.col(138), DER_DEEP = ws.col(139), DDR_DEEP = ws.col(140);
    KcCol UMF_SHALL = ws.col(141), UER_SHALL = ws.col(142), UDR_SHALL = ws.col(143);
    KcCol WUP_DEEP = ws.col(144), WUP_SHALL = ws.col(145);
    KcCol CLDFRA_CUP1D = ws.col(146), SATDEF = ws.col(147);

    const KcCol RTH = LEVOUT(KCO_RTHCUTEN), RQV = LEVOUT(KCO_RQVCUTEN), RQC = LEVOUT(KCO_RQCCUTEN);
    const KcCol RQR = LEVOUT(KCO_RQRCUTEN), RQI = LEVOUT(KCO_RQICUTEN), RQS = LEVOUT(KCO_RQSCUTEN);
    const KcCol CLDFRA_CUP = LEVOUT(KCO_CLDFRA_CUP), WUP_CUP = LEVOUT(KCO_WUP);
    const KcCol QC_IC_CUP = LEVOUT(KCO_QC_IC), QNDROP_IC_CUP = LEVOUT(KCO_QNDROP_IC);
    const KcCol QC_IU_CUP = LEVOUT(KCO_QC_IU), FCVT_QC_TO_PR_CUP = LEVOUT(KCO_FCVT_QC_PR);
    const KcCol FCVT_QC_TO_QI_CUP = LEVOUT(KCO_FCVT_QC_QI), FCVT_QI_TO_PR_CUP = LEVOUT(KCO_FCVT_QI_PR);
    const KcCol MFUP_CUP = LEVOUT(KCO_MFUP), MFUP_ENT_CUP = LEVOUT(KCO_MFUP_ENT);
    const KcCol MFDN_CUP = LEVOUT(KCO_MFDN), MFDN_ENT_CUP = LEVOUT(KCO_MFDN_ENT);
    const KcCol UPDFRA_CUP = LEVOUT(KCO_UPDFRA);

    for (int k = kts; k <= kte; ++k) {
        DQDT(k) = 0.f; DQIDT(k) = 0.f; DQCDT(k) = 0.f; DQRDT(k) = 0.f; DQSDT(k) = 0.f; DTDT(k) = 0.f;
    }
    float raincv = 0.f;
    float cutop = (float)kts;
    float cubot = (float)(kte + 1);
    float nca = HELD(KCH_NCA);
    for (int k = kts; k <= kte; ++k) {
        QC_IC_CUP(k) = 0.0f; QNDROP_IC_CUP(k) = 0.0f; QC_IU_CUP(k) = 0.0f;
        FCVT_QC_TO_PR_CUP(k) = 0.0f; FCVT_QC_TO_QI_CUP(k) = 0.0f; FCVT_QI_TO_PR_CUP(k) = 0.0f;
        WUP_CUP(k) = 0.0f; UPDFRA_CUP(k) = 0.0f; MFUP_CUP(k) = 0.0f; MFUP_ENT_CUP(k) = 0.0f;
        MFDN_CUP(k) = 0.0f; MFDN_ENT_CUP(k) = 0.0f;
    }
    HELD(KCH_WACT) = 0.0f;
    HELD(KCH_WULCL) = 0.0f;
    HELD(KCH_TCLOUD) = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        QNDROP1D(k) = 0.0f; QC1D(k) = 0.0f; QI1D(k) = 0.0f;
        FCVT_QC_TO_PR(k) = 0.0f; FCVT_QC_TO_QI(k) = 0.0f; FCVT_QI_TO_PR(k) = 0.0f; WUP(k) = 0.0f;
        CLDFRA_CUP1D(k) = CLDFRA_CUP(k);
    }

    KcParaArrays pa{U1D, V1D, T1D, QV1D, P1D, DZ1D, W0AVG1D, RHO1D, ZW1D,
                    DQDT, DQIDT, DQCDT, DQRDT, DQSDT, DTDT, QLG, QIG,
                    UMFOUT, UEROUT, UDROUT, DMFOUT, DEROUT, DDROUT, WUP,
                    QNDROP1D, QC1D, QI1D, FCVT_QC_TO_QI, FCVT_QC_TO_PR, FCVT_QI_TO_PR};

    /* ---- cupSlopeSigma (:603-612) ---- */
    KcSlope sl;
    sl.slopesfc = HELD(KCH_SLOPESFC); sl.slopeez = HELD(KCH_SLOPEEZ);
    sl.sigmasfc = HELD(KCH_SIGMASFC); sl.sigmaez = HELD(KCH_SIGMAEZ);
    sl.wstar = 0.f;
    float shall = HELD(KCH_SHALL);
    kc_slope_sigma(Z1D, TH1D, QV1D, RHO1D, ht, hfx, xland, qfx, pblh, kpbl, kte, sl, &shall);

    KcParaIn pin;
    pin.dt = dt; pin.dx = dx; pin.dxsq = dx * dx; pin.pblh = pblh;
    pin.nz = nz; pin.ntst = ntst;
    pin.f_qi = f_qi; pin.f_qs = f_qs; pin.warm_rain = warm_rain;
    KcParaOut po;
    po.raincv = raincv; po.nca = nca; po.cutop = cutop; po.cubot = cubot;
    po.wlcl = 0.f; po.updfra = 0.f; po.wulcl = 0.f; po.wact = 0.f; po.ishall = 0;

    if (sl.cupflag) {
        const int nb = numbins;
        kc_jfd(sl, nb, thbinsize, rbinsize, ws);
        float minfreq = minshallowfreq * kc_jfd_at(ws, nb, nb / 2 + 1, nb / 2 + 1);
        float biggestdeepfreq = -999.f;
        float cumdeepfreq = 0.f;
        float cumshallfreq = 0.f;
        for (int k = kts; k <= kte; ++k) {
            DQDT_SHALL(k) = 0.f; DQIDT_SHALL(k) = 0.f; DQCDT_SHALL(k) = 0.f;
            DQRDT_SHALL(k) = 0.f; DQSDT_SHALL(k) = 0.f; DTDT_SHALL(k) = 0.f;
            QLG_SHALL(k) = 0.f; QIG_SHALL(k) = 0.f;
            CUMSHALLFREQ1D(k) = 0.f; QNDROP_IC_SHALL(k) = 0.f; QC_IC_SHALL(k) = 0.f; QI_IC_SHALL(k) = 0.f;
            FCVT_QC_TO_PR_SHALL(k) = 0.f; FCVT_QC_TO_QI_SHALL(k) = 0.f; FCVT_QI_TO_PR_SHALL(k) = 0.f;
            UMF_SHALL(k) = 0.f; UER_SHALL(k) = 0.f; UDR_SHALL(k) = 0.f;
            WUP_DEEP(k) = 0.f; WUP_SHALL(k) = 0.f;
        }
        float raincv_shall = 0.f, cubot_shall = 0.f, cutop_shall = 0.f;
        float wcloudbase = 0.f;
        float wact_shall = 0.f, wulcl_shall = 0.f, wcloudbase_shall = 0.f, updfra_shall = 0.f;
        float nca_deep = 0.f, raincv_deep = 0.f, cubot_deep = 0.f, cutop_deep = 0.f;
        float updfra_deep = 0.f, wact_deep = 0.f, wulcl_deep = 0.f, wcloudbase_deep = 0.f;
        float nca_shall = 0.f;
        int kupdrbot_deep = 0;
        pin.cupflag = true;
        for (int jpert = 1; jpert <= nb; ++jpert) {
            for (int ipert = 1; ipert <= nb; ++ipert) {
                const float fj = kc_jfd_at(ws, nb, ipert, jpert);
                if (fj < minfreq || (cumdeepfreq > mindeepfreq && fj < biggestdeepfreq)) continue;
                pin.th_perturb = kc_thp_at(ws, nb, ipert);
                pin.r_perturb = kc_rp_at(ws, nb, jpert);
                pin.freq = fj;
                kc_para(tb, ws, pin, pa, po, &status);
                if (po.ishall == 0) {
                    cumdeepfreq = cumdeepfreq + fj;
                    if (fj > biggestdeepfreq) {
                        biggestdeepfreq = fj;
                        for (int k = kts; k <= kte; ++k) {
                            DQDT_DEEP(k) = DQDT(k); DQIDT_DEEP(k) = DQIDT(k); DQCDT_DEEP(k) = DQCDT(k);
                            DQRDT_DEEP(k) = DQRDT(k); DQSDT_DEEP(k) = DQSDT(k); DTDT_DEEP(k) = DTDT(k);
                        }
                        nca_deep = po.nca;
                        raincv_deep = po.raincv;
                        cubot_deep = po.cubot;
                        cutop_deep = po.cutop;
                        for (int k = kts; k <= kte; ++k) {
                            QNDROP_IC_DEEP(k) = QNDROP1D(k); QC_IC_DEEP(k) = QC1D(k); QI_IC_DEEP(k) = QI1D(k);
                            FCVT_QC_TO_PR_DEEP(k) = FCVT_QC_TO_PR(k);
                            FCVT_QC_TO_QI_DEEP(k) = FCVT_QC_TO_QI(k);
                            FCVT_QI_TO_PR_DEEP(k) = FCVT_QI_TO_PR(k);
                            WUP_DEEP(k) = WUP(k);
                        }
                        updfra_deep = po.updfra;
                        wact_deep = po.wact;
                        wulcl_deep = po.wulcl;
                        wcloudbase_deep = po.wlcl;
                        int kcb = kc_nint(cubot_deep);
                        kupdrbot_deep = kcb;
                        for (int k = kcb - 1; k >= kts; --k) {
                            if ((UMFOUT(k) > 0.0f) || (UEROUT(k) > 0.0f)) kupdrbot_deep = k;
                        }
                        for (int k = kts; k <= kte; ++k) {
                            UMF_DEEP(k) = kc_maxc(0.0f, UMFOUT(k));
                            UER_DEEP(k) = kc_maxc(0.0f, UEROUT(k));
                            UDR_DEEP(k) = kc_maxc(0.0f, UDROUT(k));
                            DMF_DEEP(k) = kc_minc(0.0f, DMFOUT(k));
                            DER_DEEP(k) = kc_maxc(0.0f, DEROUT(k));
                            DDR_DEEP(k) = kc_maxc(0.0f, DDROUT(k));
                        }
                    }
                } else if (po.ishall == 1) {
                    cumshallfreq = cumshallfreq + fj;
                    for (int k = kts; k <= kte; ++k) {
                        DQDT_SHALL(k) = DQDT_SHALL(k) + DQDT(k);
                        DQIDT_SHALL(k) = DQIDT_SHALL(k) + DQIDT(k);
                        DQCDT_SHALL(k) = DQCDT_SHALL(k) + DQCDT(k);
                        DQRDT_SHALL(k) = DQRDT_SHALL(k) + DQRDT(k);
                        DQSDT_SHALL(k) = DQSDT_SHALL(k) + DQSDT(k);
                        DTDT_SHALL(k) = DTDT_SHALL(k) + DTDT(k);
                        UMF_SHALL(k) = UMF_SHALL(k) + kc_maxc(0.0f, UMFOUT(k));
                        UER_SHALL(k) = UER_SHALL(k) + kc_maxc(0.0f, UEROUT(k));
                        UDR_SHALL(k) = UDR_SHALL(k) + kc_maxc(0.0f, UDROUT(k));
                    }
                    nca_shall = po.nca;
                    raincv_shall = raincv_shall + po.raincv * fj;
                    cubot_shall = cubot_shall + Z1D(kc_nint(po.cubot)) * fj;
                    cutop_shall = cutop_shall + Z1D(kc_nint(po.cutop)) * fj;
                    for (int k = kts; k <= kte; ++k) {
                        QLG_SHALL(k) = QLG_SHALL(k) + QLG(k) * fj;
                        QIG_SHALL(k) = QIG_SHALL(k) + QIG(k) * fj;
                    }
                    wcloudbase_shall = po.wlcl * fj + wcloudbase_shall;
                    const int k0 = kc_imax(kts, kc_nint(po.cubot));
                    const int k1 = kc_imin(kte, kc_nint(po.cutop));
                    for (int k = k0; k <= k1; ++k) {
                        CUMSHALLFREQ1D(k) = CUMSHALLFREQ1D(k) + fj;
                        QNDROP_IC_SHALL(k) = QNDROP_IC_SHALL(k) + QNDROP1D(k) * fj;
                        QC_IC_SHALL(k) = QC_IC_SHALL(k) + QC1D(k) * fj;
                        QI_IC_SHALL(k) = QI_IC_SHALL(k) + QI1D(k) * fj;
                        FCVT_QC_TO_PR_SHALL(k) = FCVT_QC_TO_PR_SHALL(k) + FCVT_QC_TO_PR(k) * QC1D(k) * fj;
                        FCVT_QC_TO_QI_SHALL(k) = FCVT_QC_TO_QI_SHALL(k) + FCVT_QC_TO_QI(k) * QC1D(k) * fj;
                        FCVT_QI_TO_PR_SHALL(k) = FCVT_QI_TO_PR_SHALL(k) + FCVT_QI_TO_PR(k) * QI1D(k) * fj;
                    }
                    for (int k = kts; k <= kte; ++k) WUP_SHALL(k) = WUP_SHALL(k) + WUP(k) * fj;
                    wact_shall = wact_shall + po.wact * fj;
                    wulcl_shall = wulcl_shall + po.wulcl * fj;
                    updfra_shall = updfra_shall + po.updfra;
                }
            }
        }
        int ishall;
        float activefrac;
        if (cumdeepfreq > mindeepfreq) {
            ishall = 0;
            activefrac = 1.f;
            for (int k = kts; k <= kte; ++k) {
                DQDT(k) = DQDT_DEEP(k); DQIDT(k) = DQIDT_DEEP(k); DQCDT(k) = DQCDT_DEEP(k);
                DQRDT(k) = DQRDT_DEEP(k); DQSDT(k) = DQSDT_DEEP(k); DTDT(k) = DTDT_DEEP(k);
            }
            nca = nca_deep;
            raincv = raincv_deep;
            cubot = cubot_deep;
            cutop = cutop_deep;
            for (int k = kts; k <= kte; ++k) {
                QC_IU_CUP(k) = QC_IC_DEEP(k);
                QC_IC_CUP(k) = QC_IC_DEEP(k);
                QNDROP_IC_CUP(k) = QNDROP_IC_DEEP(k);
                WUP_CUP(k) = WUP_DEEP(k);
            }
            HELD(KCH_WACT) = wact_deep;
            HELD(KCH_WULCL) = wulcl_deep;
            wcloudbase = wcloudbase_deep;
            int kcutop = kc_nint(cutop_deep);
            for (int k = kts; k <= kcutop; ++k) {
                FCVT_QC_TO_PR_CUP(k) = FCVT_QC_TO_PR_DEEP(k);
                FCVT_QC_TO_QI_CUP(k) = FCVT_QC_TO_QI_DEEP(k);
                FCVT_QI_TO_PR_CUP(k) = FCVT_QI_TO_PR_DEEP(k);
            }
            kc_adjust_mf(kts, kte, kcutop, ishall, UMF_DEEP, UER_DEEP, UDR_DEEP,
                         DMF_DEEP, DER_DEEP, DDR_DEEP);
            for (int k = kts; k <= kcutop; ++k) MFUP_ENT_CUP(k) = UER_DEEP(k);
            for (int k = kts + 1; k <= kcutop; ++k) MFUP_CUP(k) = UMF_DEEP(k - 1);
            for (int k = kts; k <= kcutop; ++k) MFDN_ENT_CUP(k) = DER_DEEP(k);
            for (int k = kts + 1; k <= kcutop; ++k) MFDN_CUP(k) = DMF_DEEP(k - 1);
            for (int k = kupdrbot_deep; k <= kcutop; ++k) UPDFRA_CUP(k) = updfra_deep;
            HELD(KCH_TCLOUD) = nca_deep;
        } else if (cumshallfreq > 0.f) {
            ishall = 1;
            activefrac = cumshallfreq;
            for (int k = kts; k <= kte; ++k) {
                DQDT(k) = DQDT_SHALL(k); DQIDT(k) = DQIDT_SHALL(k); DQCDT(k) = DQCDT_SHALL(k);
                DQRDT(k) = DQRDT_SHALL(k); DQSDT(k) = DQSDT_SHALL(k); DTDT(k) = DTDT_SHALL(k);
            }
            nca = nca_shall;
            raincv = raincv_shall / cumshallfreq;
            cubot_shall = cubot_shall / cumshallfreq;
            cutop_shall = cutop_shall / cumshallfreq;
            cubot = (float)(kc_find_index(cubot_shall, ZW1D, kte) - 1);
            cutop = (float)(kc_find_index(cutop_shall, ZW1D, kte) - 1);
            for (int k = kts; k <= kte; ++k) {
                QLG(k) = QLG_SHALL(k) / cumshallfreq;
                QIG(k) = QIG_SHALL(k) / cumshallfreq;
            }
            wcloudbase_shall = wcloudbase_shall / cumshallfreq;
            wcloudbase = wcloudbase_shall;
            for (int k = kts; k <= kte; ++k) {
                if (CUMSHALLFREQ1D(k) > 0.0f) {
                    FCVT_QC_TO_PR_SHALL(k) = FCVT_QC_TO_PR_SHALL(k) / kc_maxc(1.0e-20f, QC_IC_SHALL(k));
                    FCVT_QC_TO_QI_SHALL(k) = FCVT_QC_TO_QI_SHALL(k) / kc_maxc(1.0e-20f, QC_IC_SHALL(k));
                    FCVT_QI_TO_PR_SHALL(k) = FCVT_QI_TO_PR_SHALL(k) / kc_maxc(1.0e-20f, QI_IC_SHALL(k));
                    QNDROP_IC_SHALL(k) = QNDROP_IC_SHALL(k) / CUMSHALLFREQ1D(k);
                    QC_IC_SHALL(k) = QC_IC_SHALL(k) / CUMSHALLFREQ1D(k);
                    QI_IC_SHALL(k) = QI_IC_SHALL(k) / CUMSHALLFREQ1D(k);
                }
                CUMSHALLFREQ1D(k) = CUMSHALLFREQ1D(k) / cumshallfreq;
            }
            for (int k = kts; k <= kte; ++k) WUP_SHALL(k) = WUP_SHALL(k) / cumshallfreq;
            wact_shall = wact_shall / cumshallfreq;
            wulcl_shall = wulcl_shall / cumshallfreq;
            for (int k = kts; k <= kte; ++k) WUP_CUP(k) = WUP_SHALL(k);
            HELD(KCH_WACT) = wact_shall;
            HELD(KCH_WULCL) = wulcl_shall;
            int kcubot = kc_nint(cubot);
            int kcutop = kc_nint(cutop);
            for (int k = kts; k <= kcutop; ++k) {
                QC_IC_CUP(k) = QC_IC_SHALL(k);
                QNDROP_IC_CUP(k) = QNDROP_IC_SHALL(k);
                QC_IU_CUP(k) = QC_IC_SHALL(k);
            }
            for (int k = kcubot; k <= kcutop; ++k) QC_IC_CUP(k) = 1.0e-3f;
            for (int k = kts; k <= kcutop; ++k) {
                FCVT_QC_TO_PR_CUP(k) = FCVT_QC_TO_PR_SHALL(k);
                FCVT_QC_TO_QI_CUP(k) = FCVT_QC_TO_QI_SHALL(k);
                FCVT_QI_TO_PR_CUP(k) = FCVT_QI_TO_PR_SHALL(k);
            }
            kc_adjust_mf(kts, kte, kcutop, ishall, UMF_SHALL, UER_SHALL, UDR_SHALL,
                         DMFOUT, DEROUT, DDROUT);
            for (int k = kts; k <= kcutop; ++k) MFUP_ENT_CUP(k) = UER_SHALL(k);
            for (int k = kts + 1; k <= kcutop; ++k) MFUP_CUP(k) = UMF_SHALL(k - 1);
            int kupdrbot_shall = kcubot;
            for (int k = kcubot - 1; k >= kts; --k) {
                if ((UMF_SHALL(k) > 0.0f) || (UER_SHALL(k) > 0.0f)) kupdrbot_shall = k;
            }
            for (int k = kupdrbot_shall; k <= kcutop; ++k) UPDFRA_CUP(k) = updfra_shall;
            HELD(KCH_TCLOUD) = nca_shall;
        } else {
            ishall = 2;
            activefrac = 0.f;
            for (int k = kts; k <= kte; ++k) {
                DQDT(k) = 0.f; DQIDT(k) = 0.f; DQCDT(k) = 0.f; DQRDT(k) = 0.f; DQSDT(k) = 0.f; DTDT(k) = 0.f;
            }
            nca = -1.f;
            raincv = 0.f;
            cubot = 1.f;
            cutop = 1.f;
        }
        HELD(KCH_ACTIVEFRAC) = activefrac;
        HELD(KCH_WCLOUDBASE) = wcloudbase;
        shall = (float)ishall;
        int kcubot = (int)cubot;
        int kcutop = (int)cutop;
        KcCloudOut co;
        co.taucloud = HELD(KCH_TAUCLOUD); co.tactive = HELD(KCH_TACTIVE);
        co.tstar = HELD(KCH_TSTAR); co.lnint = HELD(KCH_LNINT);
        kc_cloud_fraction(QLG, QV1D, T1D, Z1D, P1D, kcubot, kcutop, ishall, sl.wstar, wcloudbase,
                          pblh, activefrac, CLDFRA_CUP1D, SATDEF, kts, kte, MFUP_CUP, co);
        HELD(KCH_TAUCLOUD) = co.taucloud; HELD(KCH_TACTIVE) = co.tactive;
        HELD(KCH_TSTAR) = co.tstar; HELD(KCH_LNINT) = co.lnint;
        for (int k = kts; k <= kte; ++k) CLDFRA_CUP(k) = CLDFRA_CUP1D(k);
    } else {
        /* The standard KF scheme for a column CuP does not trigger on
         * (:1132-1168).  th_perturb(1)/r_perturb(1) are read by nothing on
         * this path (:1780-1787); freq is 0.01.  KF_cup_PARA writes RAINCV,
         * NCA, CUTOP and CUBOT only when it reaches its end; po carries the
         * driver's values through an early return. */
        pin.th_perturb = 0.f; pin.r_perturb = 0.f; pin.freq = 0.01f;
        pin.cupflag = false;
        kc_para(tb, ws, pin, pa, po, &status);
        raincv = po.raincv; nca = po.nca; cutop = po.cutop; cubot = po.cubot;
    }
    HELD(KCH_SLOPESFC) = sl.slopesfc; HELD(KCH_SLOPEEZ) = sl.slopeez;
    HELD(KCH_SIGMASFC) = sl.sigmasfc; HELD(KCH_SIGMAEZ) = sl.sigmaez;
    HELD(KCH_SHALL) = shall;
    IHELD(KCF_CUPFLAG) = sl.cupflag ? 1 : 0;
    HELD(KCH_NCA) = nca;
    HELD(KCH_RAINCV) = raincv;
    HELD(KCH_CUTOP) = cutop;
    HELD(KCH_CUBOT) = cubot;

    /* ---- the driver's output algebra (:1183-1222) ---- */
    for (int k = kts; k <= kte; ++k) {
        RTH(k) = DTDT(k) / PI1D(k);
        RQV(k) = DQDT(k);
    }
    if (f_qr) {
        for (int k = kts; k <= kte; ++k) { RQR(k) = DQRDT(k); RQC(k) = DQCDT(k); }
    } else {
        for (int k = kts; k <= kte; ++k) { RQR(k) = 0.f; RQC(k) = DQRDT(k) + DQCDT(k); }
    }
    if (f_qi) {
        for (int k = kts; k <= kte; ++k) RQI(k) = DQIDT(k);
    }
    if (f_qs) {
        for (int k = kts; k <= kte; ++k) RQS(k) = DQSDT(k);
    }
    if (status) atomicOr(status_out, status);
#undef LEVIN
#undef LEVOUT
#undef HELD
#undef IHELD
}

/* One thread per column; a launch covers columns [col0, col0 + gridDim*blockDim). */
extern "C" __global__ void kfcup_column(
    float* __restrict__ lev_in, const float* __restrict__ sca_in,
    const int* __restrict__ kpbl_in, float* __restrict__ held, int* __restrict__ iheld,
    float* __restrict__ lev_out,
    const float* __restrict__ ttab, const float* __restrict__ qstab,
    const float* __restrict__ the0k, const float* __restrict__ alu,
    float plutop, float rdpr, float rdthk,
    float dt, float dx, int ntst, int numbins, float thbinsize, float rbinsize,
    float mindeepfreq, float minshallowfreq, int flags,
    float* __restrict__ ws_all, int* __restrict__ status_out,
    int nz, int ncol, int col0)
{
    const int lt = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = col0 + lt;
    if (col >= ncol) return;
    KcTable tb{ttab, qstab, the0k, alu, plutop, rdpr, rdthk};
    kc_column(col, lt, lev_in, sca_in, kpbl_in, held, iheld, lev_out, tb, dt, dx, ntst,
              numbins, thbinsize, rbinsize, mindeepfreq, minshallowfreq, flags, ws_all,
              status_out, nz, ncol);
}

/* KF_cup_CPS: W0AVG update (:446-447). Explicit rounding preserves the
 * separate WRF operations on the default and strict CUDA routes. */
__device__ __forceinline__ float kc_history(float old, float wlo, float whi, float tst) {
    const float w0 = __fmul_rn(0.5f, __fadd_rn(wlo, whi));
    return __fdiv_rn(__fadd_rn(__fmul_rn(old, __fsub_rn(tst, 1.f)), w0), tst);
}
// One coalesced copy pass replaces 17 separately launched stack copies.
// No arithmetic touches the native input words.
extern "C" __global__ void kfcup_pack(
    float* lev, float* sfc,
    const float* u, const float* v, const float* th, const float* t,
    const float* dz, const float* p, const float* pi, const float* rho,
    const float* qv, const float* z, const float* zw, const float* avg,
    const float* ht, const float* hfx, const float* qfx,
    const float* xland, const float* pblh, int cells, int columns) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<cells) {
        lev[0*cells+i]=u[i]; lev[1*cells+i]=v[i];
        lev[2*cells+i]=th[i]; lev[3*cells+i]=t[i];
        lev[4*cells+i]=dz[i]; lev[5*cells+i]=p[i];
        lev[6*cells+i]=pi[i]; lev[7*cells+i]=rho[i];
        lev[8*cells+i]=qv[i]; lev[9*cells+i]=z[i];
        lev[10*cells+i]=zw[i]; lev[11*cells+i]=avg[i];
    }
    if (i<columns) {
        sfc[i]=ht[i]; sfc[columns+i]=hfx[i];
        sfc[2*columns+i]=qfx[i]; sfc[3*columns+i]=xland[i];
        sfc[4*columns+i]=pblh[i];
    }
}

extern "C" __global__ void kfcup_trigger_history(float* avg, const float* w,
                                               float tst, int ncol, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nz*ncol) avg[i] = kc_history(avg[i], w[i], w[i+ncol], tst);
}

/* WRF module_radiation_driver.F:1510-1557, KF-CuP arm only. This is the
 * single override implementation, including shallowcu_forced_ra. No ice
 * override occurs in that WRF block: QI is preserved by the consumer. */
__device__ __forceinline__ void kc_radiation_merge(
    int icloud, bool forced, int flag, float shall, float cubot, float cutop,
    int k, float& cf, float& qc, float& cup) {
    bool compute = true;
    if (icloud == 1) {
        compute = false;
        if (flag == 1 && shall > 1.f) cf = 0.f;
        else if (flag == 1 && shall <= 1.f) { cf = 0.f; compute = true; }
        else if (flag == 2 && shall > 1.f) cf = 1.f;
        else if (flag == 3) compute = true;
    }
    if (compute) {
        if ((int)shall <= 1 && k >= (int)cubot && k <= (int)cutop) cf = cup;
        else if (shall > 1.f) cup = 0.f;
    }
    if (shall <= 1.f && (float)k >= cubot && (float)k <= cutop) {
        const float modified = __fmul_rn(cup, 1.e-3f);
        qc = kc_max(modified, qc);
        if (forced) cf = kc_maxc(0.36f, cf);
    }
}
extern "C" __global__ void kfcup_radiation_merge(
    float* cf, float* qc, float* cup, const int* flags,
    const float* shall, const float* cubot, const float* cutop,
    int icloud, int forced, int ncol, int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz*ncol) return;
    const int c=i%ncol, k=i/ncol+1;
    kc_radiation_merge(icloud, forced != 0, flags[i], shall[c], cubot[c], cutop[c],
                       k, cf[i], qc[i], cup[i]);
}

/* advance_ppt's KFCUPSCHEME cloud clear occurs AFTER the NCA decrement.
 * The six tendency clears use the pre-decrement predicate instead. */
extern "C" __global__ void kfcup_cloud_clock(
    const float* nca, float* cup, float dt, int ncol, int nz) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=ncol) return;
    const float held=nca[c];
    if (held>0.f && floorf(__fadd_rn(__fdiv_rn(held,dt),0.5f))<=1.f)
        for (int k=0;k<nz;++k) cup[k*ncol+c]=0.f;
}

/* CuP returns RAINCV in mm per model step. Accumulate that word directly:
 * dividing by DT then multiplying by DT would change captured rain words. */
extern "C" __global__ void kfcup_advance_clock(
    float* rainc, float* rainbl, float* surface, float* nca, float* expiring,
    const float* raincv, float dt, int has_surface, int ncol) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if (c>=ncol) return;
    const float increment=raincv[c];
    rainc[c]=__fadd_rn(rainc[c],increment);
    rainbl[c]=__fadd_rn(rainbl[c],increment);
    if (has_surface) surface[c]=__fadd_rn(surface[c],increment);
    const float held=nca[c];
    expiring[c]=(held>0.f && floorf(__fadd_rn(__fdiv_rn(held,dt),0.5f))<=1.f)?1.f:0.f;
    nca[c]=(held>0.f)?__fsub_rn(held,dt):held;
}
