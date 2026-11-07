#ifndef KF_TRIGGER
#define KF_TRIGGER 1
#endif
/* Kain--Fritsch cumulus (WRF cu_physics=1), one CUDA thread per column.
 *
 * AUTHORITY: WRF v4.6.1 phys/module_cu_kfeta.F (sha256 e6376c2d...6e5f, LF),
 * graded word for word by the column oracle in tools/kf_wrf461_oracle: the
 * unmodified Fortran at gfortran -O0, every output of KF_eta_CPS, 0 ULP.
 *
 * This file is a line-for-line port of KF_eta_PARA (module_cu_kfeta.F:533-
 * 2692) and its helpers TPMIX2 (:2695), DTFRZNEW (:2817), CONDLOAD (:2863),
 * PROF5 (:2930), TPMIX2DD (:2977) and ENVIRTHT (:3044), plus the per-column
 * part of KF_eta_CPS (:406-525).  It replaced, on 2026-10-07, a kernel that
 * re-derived the scheme instead of porting it and was never compared with
 * WRF: its own Bolton-form density instead of WRF's RHO input, a
 * per-candidate reset of arrays WRF carries between updraft-source
 * candidates, clipped table lookups WRF does not clip, a floor()-based NINT,
 * shallow-NCA in seconds of the clock instead of CUDT*60, and different
 * operation orders throughout.  Graded by the same oracle, that kernel
 * matched WRF on none of the 1,793 convecting columns it could be run on
 * (it had no F_QR fold), up to 2.0e9 ULP in RTHCUTEN; see
 * tools/kf_wrf461_oracle/README.md.
 *
 * How the transcription is held to the Fortran:
 *   - Arrays keep the Fortran names and the Fortran 1-based index:
 *     ``z0(k)`` is ``Z0(K)``.  Inputs read the global arrays through the
 *     same 1-based accessor (``t0(k)`` is ``T0(K)``).
 *   - Every expression keeps Fortran's left-to-right evaluation order; the
 *     unit compiles with --fmad=false --ftz=false on its own route
 *     (gpuwm/core/kf.py), so nothing is contracted or flushed, and every
 *     division by a non-power-of-two constant is spelled __fdiv_rn (the A146
 *     rule: under -ftz=true NVRTC turns it into a reciprocal multiply on
 *     Blackwell).
 *   - EXP, ALOG and REAL**REAL call WOOF's own float32 libm (gfk_exp,
 *     gfk_log, gfk_pow from glibc_flt32.cuh), not CUDA's builtins, because
 *     the builtins round differently.  SQRT and division are IEEE on both
 *     sides.  NINT is round-half-away (roundf), INT/AINT truncate.
 *   - State WRF carries across updraft-source candidates within one column
 *     (DILFRC above all, read later through DDILFRC in the CAPE closure) is
 *     carried here too: no array is reset that the Fortran does not reset.
 *
 * Inputs are WRF's: T, QV, P (p_hyd), RHO ((1+qv)/alt), DZ8W, PI, U, V and
 * the advanced W0AVG.  Outputs are KF_eta_CPS's: the six tendencies (with
 * the F_QR fold and the four F_QI/F_QS/warm_rain branches), RAINCV, PRATEC,
 * NCA (written only when the column convects, 0 otherwise -- the driver
 * owns the timer), CUTOP/CUBOT, SHALL, and optionally the CLDFRA_DP/SH and
 * QC_KF/QI_KF diagnostics.
 */

/* KF_KMAX is the REFUSAL CEILING and nothing else: nz outside [8, KF_KMAX]
 * is refused here and in gpuwm/core/kf.py's VERTICAL_LEVEL_BOUNDS.  No
 * array in this file is sized by it. */
#ifndef KF_KMAX
#define KF_KMAX 128
#endif

/* ==========================================================================
 * The per-thread column workspace
 * ==========================================================================
 * One thread owns one whole column, so the scheme's nz-sized arrays are
 * function-scope state.  CUDA prices a local frame at the card's
 * RESIDENT-THREAD CAPACITY (one per-context backing store of
 * (frame - 1024) * SMs * maxThreadsPerSM, never returned), so the column
 * arrays live in a caller-provided GLOBAL workspace sized to the threads in
 * flight instead; gpuwm/core/kf.py tiles the columns to match.  MEASURED on
 * an RTX 5070 Ti (sm_120) for the previous kernel: a 9,216 B frame at
 * nz = 49 reserved 842 MiB at first launch.
 *
 * Layout, interleaved across a warp exactly like local memory: element k of
 * slot s for lane t of a warp's region is region + (s * nz + k) * 32 + t.
 * Slot ids are literal (NVRTC has no __COUNTER__); tests/test_kf_workspace.py
 * reads them out of this file and fails on a duplicate or an id past
 * KFWS_SLOTS.  The extent is the RUNTIME nz.
 */
#define KFWS_SLOTS 66
#define KFWS_LANES 32

/* A column array in the workspace, indexed with FORTRAN's 1-based index. */
struct KfCol {
    float* p;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)KFWS_LANES];
    }
    /* 0-based view, for kf_predict_trigger only. */
    __device__ __forceinline__ float& operator[](int k) const {
        return p[(size_t)k * (size_t)KFWS_LANES];
    }
};
struct KfColI {
    int* p;
    __device__ __forceinline__ int& operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)KFWS_LANES];
    }
};
/* A read-only (nz, ny, nx) input column with a 1-based level index. */
struct KfIn {
    const float* __restrict__ p;
    int ncol;
    int column;
    __device__ __forceinline__ float operator()(int k) const {
        return p[(size_t)(k - 1) * (size_t)ncol + column];
    }
};

#define KFWS_AT(base, idx, kp) \
    (KfCol{(base) + (size_t)(idx) * (size_t)(kp) * (size_t)KFWS_LANES})
#define KFWS_AT_I(base, idx, kp) \
    (KfColI{(int*)((base) + (size_t)(idx) * (size_t)(kp) \
                   * (size_t)KFWS_LANES)})
#define KFWS_LANE_BASE(ws, kp) \
    ((ws) + ((size_t)blockIdx.x * (blockDim.x / KFWS_LANES) \
             + threadIdx.x / KFWS_LANES) * KFWS_SLOTS * (kp) \
             * KFWS_LANES + threadIdx.x % KFWS_LANES)

/* KFPhaseMode in gpuwm/core/kf.py: WRF's feedback branch, from
 * (warm_rain, F_QS, F_QI). */
#define KF_PHASE_WARM_RAIN 0
#define KF_PHASE_NO_SEPARATE_SNOW 1
#define KF_PHASE_SEPARATE_SNOW 2
#define KF_PHASE_SEPARATE_ICE_SNOW 3

/* The KF_LUTAB state (module_cu_kfeta.F:18-22).  TTAB/QSTAB are uploaded in
 * FORTRAN (column-major) order so TTAB(IT,IP) is ttab[(IT-1)+(IP-1)*250]
 * and every index the Fortran forms, in bounds or not, reads the same
 * word it reads. */
struct KfLut {
    const float* __restrict__ ttab;
    const float* __restrict__ qstab;
    const float* __restrict__ the0k;
    const float* __restrict__ alu;
    float plutop, rdpr, rdthk;
    /* Every in-bounds index reads exactly the Fortran's word.  An index
     * outside the whole array is an out-of-bounds read in WRF (it reads a
     * neighbouring module variable); here it is clamped to the array so
     * the kernel never leaves its allocation.  No physical column reaches
     * it: it needs p < PLUTOP = 50 hPa at an updraft level, or a dewpoint
     * above ~44 C. */
    __device__ __forceinline__ float tt(int it, int ip) const {
        return ttab[min(max((it - 1) + (ip - 1) * 250, 0), 250 * 220 - 1)];
    }
    __device__ __forceinline__ float qt(int it, int ip) const {
        return qstab[min(max((it - 1) + (ip - 1) * 250, 0), 250 * 220 - 1)];
    }
    __device__ __forceinline__ float th(int ip) const {  /* THE0K(IP) */
        return the0k[min(max(ip - 1, 0), 219)];
    }
    __device__ __forceinline__ float al(int i) const {   /* ALU(I) */
        return alu[min(max(i - 1, 0), 199)];
    }
};

/* The Buck/Bolton constants KF derives from the SVP arguments
 * (module_cu_kfeta.F:730-733): ALIQ = SVP1*1000., BLIQ = SVP2,
 * CLIQ = SVP2*SVPT0, DLIQ = SVP3, each a binary32 operation on the
 * share/module_model_constants.F parameters. */
struct KfSvp {
    float aliq, bliq, cliq, dliq;
};

/* ---------------------------------------------------------------- TPMIX2 */
/* module_cu_kfeta.F:2695-2815 */
__device__ __forceinline__ void kf_tpmix2(
        const KfLut& L, float p, float thes, float* tu, float* qu,
        float* qliq, float* qice, float* qnewlq, float* qnewic,
        float xlv1, float xlv0) {
    float tp = (p - L.plutop) * L.rdpr;
    float qq = tp - truncf(tp);
    int iptb = (int)tp + 1;
    float bth = (L.th(iptb + 1) - L.th(iptb)) * qq + L.th(iptb);
    float tth = (thes - bth) * L.rdthk;
    float pp = tth - truncf(tth);
    int ithtb = (int)tth + 1;
    float t00 = L.tt(ithtb, iptb);
    float t10 = L.tt(ithtb + 1, iptb);
    float t01 = L.tt(ithtb, iptb + 1);
    float t11 = L.tt(ithtb + 1, iptb + 1);
    float q00 = L.qt(ithtb, iptb);
    float q10 = L.qt(ithtb + 1, iptb);
    float q01 = L.qt(ithtb, iptb + 1);
    float q11 = L.qt(ithtb + 1, iptb + 1);
    float temp = (((t00 + (t10 - t00) * pp) + (t01 - t00) * qq)
                  + ((((t00 - t10) - t01) + t11) * pp) * qq);
    float qs = (((q00 + (q10 - q00) * pp) + (q01 - q00) * qq)
                + ((((q00 - q10) - q01) + q11) * pp) * qq);
    float qnew;
    float dq = qs - *qu;
    if (dq <= 0.0f) {
        qnew = *qu - qs;
        *qu = qs;
    } else {
        qnew = 0.0f;
        float qtot = *qliq + *qice;
        if (qtot >= dq) {
            *qliq = *qliq - (dq * *qliq) / (qtot + 1.e-10f);
            *qice = *qice - (dq * *qice) / (qtot + 1.e-10f);
            *qu = qs;
        } else {
            float rll = xlv0 - xlv1 * temp;
            float cpp = 1004.5f * (1.0f + 0.89f * *qu);
            if (qtot < 1.e-10f) {
                temp = temp + (rll * (dq / (1.0f + dq))) / cpp;
            } else {
                temp = temp + (rll * ((dq - qtot) / ((1.0f + dq) - qtot)))
                              / cpp;
                *qu = *qu + qtot;
                qtot = 0.0f;
                *qliq = 0.0f;
                *qice = 0.0f;
            }
        }
    }
    *tu = temp;
    *qnewlq = qnew;
    *qnewic = 0.0f;
}

/* -------------------------------------------------------------- TPMIX2DD */
/* module_cu_kfeta.F:2977-3041 */
__device__ __forceinline__ void kf_tpmix2dd(
        const KfLut& L, float p, float thes, float* ts, float* qs) {
    float tp = (p - L.plutop) * L.rdpr;
    float qq = tp - truncf(tp);
    int iptb = (int)tp + 1;
    float bth = (L.th(iptb + 1) - L.th(iptb)) * qq + L.th(iptb);
    float tth = (thes - bth) * L.rdthk;
    float pp = tth - truncf(tth);
    int ithtb = (int)tth + 1;
    float t00 = L.tt(ithtb, iptb);
    float t10 = L.tt(ithtb + 1, iptb);
    float t01 = L.tt(ithtb, iptb + 1);
    float t11 = L.tt(ithtb + 1, iptb + 1);
    float q00 = L.qt(ithtb, iptb);
    float q10 = L.qt(ithtb + 1, iptb);
    float q01 = L.qt(ithtb, iptb + 1);
    float q11 = L.qt(ithtb + 1, iptb + 1);
    *ts = (((t00 + (t10 - t00) * pp) + (t01 - t00) * qq)
           + ((((t00 - t10) - t01) + t11) * pp) * qq);
    *qs = (((q00 + (q10 - q00) * pp) + (q01 - q00) * qq)
           + ((((q00 - q10) - q01) + q11) * pp) * qq);
}

/* -------------------------------------------------------------- DTFRZNEW */
/* module_cu_kfeta.F:2817-2860 */
__device__ __forceinline__ void kf_dtfrznew(
        const KfSvp& S, float* tu, float p, float* thteu, float* qu,
        float qfrz, float* qice) {
    float rlc = 2.5e6f - 2369.276f * (*tu - 273.16f);
    float rls = 2833922.0f - 259.532f * (*tu - 273.16f);
    float rlf = rls - rlc;
    float cpp = 1004.5f * (1.0f + 0.89f * *qu);
    float a = (S.cliq - S.bliq * S.dliq) / ((*tu - S.dliq) * (*tu - S.dliq));
    float dtfrz = (rlf * qfrz) / (cpp + (rls * *qu) * a);
    *tu = *tu + dtfrz;
    float es = S.aliq * gfk_exp((S.bliq * *tu - S.cliq) / (*tu - S.dliq));
    float qs = (es * 0.622f) / (p - es);
    float dqevap = qs - *qu;
    *qice = *qice - dqevap;
    *qu = *qu + dqevap;
    float pii = gfk_pow(1.e5f / p, 0.2854f * (1.0f - 0.28f * *qu));
    *thteu = (*tu * pii)
        * gfk_exp(((3374.6525f / *tu - 2.5403f) * *qu)
                  * (1.0f + 0.81f * *qu));
}

/* -------------------------------------------------------------- CONDLOAD */
/* module_cu_kfeta.F:2863-2927 */
__device__ __forceinline__ void kf_condload(
        float* qliq, float* qice, float* wtw, float dz, float boterm,
        float enterm, float rate, float* qnewlq, float* qnewic,
        float* qlqout, float* qicout, float g) {
    float qtot = *qliq + *qice;
    float qnew = *qnewlq + *qnewic;
    float qest = 0.5f * (qtot + qnew);
    float g1 = ((*wtw + boterm) - enterm)
               - __fdiv_rn(((2.0f * g) * dz) * qest, 1.5f);
    if (g1 < 0.0f) g1 = 0.0f;
    float wavg = 0.5f * (sqrtf(*wtw) + sqrtf(g1));
    float conv = (rate * dz) / wavg;
    float ratio3 = *qnewlq / (qnew + 1.e-8f);
    qtot = qtot + 0.6f * qnew;
    float oldq = qtot;
    float ratio4 = (0.6f * *qnewlq + *qliq) / (qtot + 1.e-8f);
    qtot = qtot * gfk_exp(-conv);
    float dq = oldq - qtot;
    *qlqout = ratio4 * dq;
    *qicout = (1.0f - ratio4) * dq;
    float pptdrg = 0.5f * ((oldq + qtot) - 0.2f * qnew);
    *wtw = ((*wtw + boterm) - enterm)
           - __fdiv_rn(((2.0f * g) * dz) * pptdrg, 1.5f);
    if (fabsf(*wtw) < 1.e-4f) *wtw = 1.e-4f;
    *qliq = ratio4 * qtot + (ratio3 * 0.4f) * qnew;
    *qice = (1.0f - ratio4) * qtot + ((1.0f - ratio3) * 0.4f) * qnew;
    *qnewlq = 0.0f;
    *qnewic = 0.0f;
}

/* ----------------------------------------------------------------- PROF5 */
/* module_cu_kfeta.F:2930-2974.  E45 = EXP(-4.5) has a constant argument,
 * which gfortran folds at compile time; tools/kf_wrf461_oracle/
 * e45_probe.F90 measured the folded word and glibc's runtime expf(-4.5)
 * on the oracle toolchain: both are 0x3c360282, the word used here. */
#define KF_PROF5_E45 __uint_as_float(0x3c360282u)
__device__ __forceinline__ void kf_prof5(float eq, float* ee, float* ud) {
    const float sqrt2p = 2.506628f, a1 = 0.4361836f, a2 = -0.1201676f;
    const float a3 = 0.9372980f, p = 0.33267f, sigma = 0.166666667f;
    const float fe = 0.202765151f;
    float y = 6.0f * eq - 3.0f;
    float ey = gfk_exp((y * y) / (-2.0f));
    float e45 = KF_PROF5_E45;
    float t2 = 1.0f / (1.0f + p * fabsf(y));
    float t1 = 0.500498f;
    float c1 = (a1 * t1 + (a2 * t1) * t1) + ((a3 * t1) * t1) * t1;
    float c2 = (a1 * t2 + (a2 * t2) * t2) + ((a3 * t2) * t2) * t2;
    if (y >= 0.0f) {
        *ee = sigma * (0.5f * ((sqrt2p - e45 * c1) - ey * c2)
                       + sigma * (e45 - ey))
              - ((e45 * eq) * eq) / 2.0f;
        *ud = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
              - e45 * ((0.5f + (eq * eq) / 2.0f) - eq);
    } else {
        *ee = sigma * (0.5f * (ey * c2 - e45 * c1) + sigma * (e45 - ey))
              - ((e45 * eq) * eq) / 2.0f;
        *ud = sigma * (0.5f * ((sqrt2p - e45 * c1) - ey * c2)
                       + sigma * (e45 - ey))
              - e45 * ((0.5f + (eq * eq) / 2.0f) - eq);
    }
    *ee = __fdiv_rn(*ee, fe);
    *ud = __fdiv_rn(*ud, fe);
}

/* -------------------------------------------------------------- ENVIRTHT */
/* module_cu_kfeta.F:3044-3081 */
__device__ __forceinline__ float kf_envirtht(
        const KfLut& L, const KfSvp& S, float p1, float t1, float q1) {
    const float t00 = 273.16f, p00 = 1.e5f, c1 = 3374.6525f, c2 = 2.5403f;
    float ee = (q1 * p1) / (0.622f + q1);
    const float astrt = 1.e-3f, ainc = 0.075f;
    float a1 = ee / S.aliq;
    float tp = (a1 - astrt) / ainc;
    int indlu = (int)tp + 1;
    float value = (float)(indlu - 1) * ainc + astrt;
    float aintrp = (a1 - value) / ainc;
    float tlog = aintrp * L.al(indlu + 1) + (1.0f - aintrp) * L.al(indlu);
    float tdpt = (S.cliq - S.dliq * tlog) / (S.bliq - tlog);
    float tsat = tdpt - ((0.212f + 1.571e-3f * (tdpt - t00))
                         - 4.36e-4f * (t1 - t00)) * (t1 - tdpt);
    float tht = t1 * gfk_pow(p00 / p1, 0.2854f * (1.0f - 0.28f * q1));
    return tht * gfk_exp(((c1 / tsat - c2) * q1) * (1.0f + 0.81f * q1));
}

/* The LCL temperature from a mixed layer, the ALU-table dewpoint
 * (module_cu_kfeta.F:919-929 and, with BINC for AINC, :2117-2129). */
__device__ __forceinline__ float kf_tlcl(
        const KfLut& L, const KfSvp& S, float emix, float tmix) {
    const float t00 = 273.16f;
    const float astrt = 1.e-3f, ainc = 0.075f;
    float a1 = emix / S.aliq;
    float tp = (a1 - astrt) / ainc;
    int indlu = (int)tp + 1;
    float value = (float)(indlu - 1) * ainc + astrt;
    float aintrp = (a1 - value) / ainc;
    float tlog = aintrp * L.al(indlu + 1) + (1.0f - aintrp) * L.al(indlu);
    float tdpt = (S.cliq - S.dliq * tlog) / (S.bliq - tlog);
    float tlcl = tdpt - ((0.212f + 1.571e-3f * (tdpt - t00))
                         - 4.36e-4f * (tmix - t00)) * (tmix - tdpt);
    return fminf(tlcl, tmix);
}

/* NINT for the positive and negative finite values KF forms: Fortran
 * rounds half away from zero, which is roundf, not rintf. */
__device__ __forceinline__ int kf_nint(float x) { return (int)roundf(x); }

/* W0AVG running mean, KF_eta_CPS (module_cu_kfeta.F:218-251), for every
 * mass level of every column: W0 = 0.5*(w(k)+w(k+1)) and
 * W0AVG = (W0AVG*W0AVGfctr + W0*W0fctr)/W0den.  The factors are formed on
 * the host from TST/CUDT/DT exactly as the Fortran forms them. */
extern "C" __global__ void kf_w0avg_update(
        const float* __restrict__ w, float* __restrict__ w0avg,
        float w0avgfctr, float w0fctr, float w0den, int nz, int ncol) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= (long long)nz * ncol) return;
    float w0 = 0.5f * (w[index] + w[index + ncol]);
    w0avg[index] = (w0avg[index] * w0avgfctr + w0 * w0fctr) / w0den;
}

/* Column ORDER only: a cheap guess at which columns will run the long
 * updraft/closure path, so a block can run them together.  Every column
 * still runs the full scheme; nothing here reaches an output. */
__device__ __noinline__ bool kf_predict_trigger(
        const float* temperature, const float* qv, const float* pressure,
        const float* dz, const float* w0avg, float* workspace, int nz,
        int ncol, int column, float dx) {
    KfCol z = KFWS_AT(workspace, 0, nz);
    KfCol dp = KFWS_AT(workspace, 1, nz);
    KfCol qenv = KFWS_AT(workspace, 3, nz);
    float z_interface = 0.0f;
    for (int k = 0; k < nz; ++k) {
        int index = k * ncol + column;
        float depth = dz[index];
        z[k] = z_interface + 0.5f * depth;
        z_interface += depth;
        float t = temperature[index];
        float es = 611.2f * expf(17.67f * (t - 273.15f) / (t - 29.65f));
        float qsat = 0.622f * es / (pressure[index] - es);
        qenv[k] = fmaxf(fminf(qv[index], qsat), 1.0e-6f);
        dp[k] = pressure[index] / (287.0f * t) * 9.81f * depth;
    }
    float surface_pressure = pressure[column];
    float threshold = surface_pressure - 1500.0f;
    for (int candidate = 0; candidate < nz; ++candidate) {
        if (candidate > 0) {
            float p = pressure[candidate * ncol + column];
            if (p < surface_pressure - 30000.0f) break;
            if (p >= threshold) continue;
            threshold -= 1500.0f;
        }
        float sum_dp = 0.0f;
        int top = candidate;
        while (top < nz && sum_dp <= 5000.0f) sum_dp += dp[top++];
        if (sum_dp <= 5000.0f || top >= nz) continue;
        float sum_t = 0.0f, sum_q = 0.0f, sum_z = 0.0f, sum_p = 0.0f;
        for (int k = candidate; k < top; ++k) {
            int index = k * ncol + column;
            sum_t += dp[k] * temperature[index];
            sum_q += dp[k] * qenv[k];
            sum_z += dp[k] * z[k];
            sum_p += dp[k] * pressure[index];
        }
        float tm = sum_t / sum_dp, qm = sum_q / sum_dp;
        float zm = sum_z / sum_dp, pm = sum_p / sum_dp;
        float emix = fmaxf(qm * pm / (0.622f + qm), 1.0e-3f);
        float tlog = logf(__fdiv_rn(emix, 611.2f));
        float dewpoint = (17.67f * 273.15f - 29.65f * tlog) / (17.67f - tlog);
        float lcl_t = fminf(dewpoint - (0.212f + 1.571e-3f * (dewpoint - 273.16f)
            - 4.36e-4f * (tm - 273.16f)) * (tm - dewpoint), tm);
        float lcl_z = zm + __fdiv_rn(lcl_t - tm, -9.81f / 1004.5f);
        int lk = 0;
        while (lk < nz && z[lk] < lcl_z) ++lk;
        if (lk <= 0 || lk >= nz - 2) continue;
        float fz = (lcl_z - z[lk - 1]) / (z[lk] - z[lk - 1]);
        float env = temperature[(lk - 1) * ncol + column]
            + (temperature[lk * ncol + column]
               - temperature[(lk - 1) * ncol + column]) * fz;
        float wl = w0avg[(lk - 1) * ncol + column]
            + (w0avg[lk * ncol + column]
               - w0avg[(lk - 1) * ncol + column]) * fz;
        float wk = __fdiv_rn(wl * dx, 25000.0f)
            - 0.02f * fminf(__fdiv_rn(lcl_z, 2000.0f), 1.0f);
        float dt_lcl = wk < 1.0e-4f ? 0.0f : 4.64f * powf(wk, 0.33f);
        if (lcl_t + dt_lcl < env) continue;
        return true;
    }
    return false;
}

#if KF_TRIGGER == 3
/* WRF module_cu_kfeta.F:989-1016, kfeta_trigger = 3: the parcel's extra
 * temperature perturbation DTRH from the environmental relative humidity at
 * the LCL (Narita and Ohmori 2007).  U00 = 0.75 < 1, so the branch is always
 * taken.  Every operation is an explicit round-to-nearest intrinsic in
 * Fortran's left-to-right order, so neither arm can contract it:
 *   QSLCL  = QES(K) + (QES(KLCL) - QES(K))*DLP
 *   RHLCL  = QENV/QSLCL
 *   DQSSDT = QMIX*(CLIQ - BLIQ*DLIQ)/((TLCL - DLIQ)*(TLCL - DLIQ))
 *   DTRH   = 0.25*(RHLCL - 0.75)*QMIX/DQSSDT   0.75 <= RHLCL <= 0.95
 *          = (1./RHLCL - 1.)*QMIX/DQSSDT       RHLCL > 0.95
 *          = 0                                 otherwise
 * CLIQ = SVP2*SVPT0 and BLIQ*DLIQ = SVP2*SVP3 are the run-time products of
 * the model constants 17.67, 273.15 and 29.65 (module_cu_kfeta.F:730-733). */
__device__ __forceinline__ float kf_trigger3_dtrh(
        float qes_k, float qes_klcl, float dlp, float qenv, float qmix,
        float tlcl) {
    float qslcl = __fadd_rn(qes_k, __fmul_rn(__fsub_rn(qes_klcl, qes_k), dlp));
    float rhlcl = __fdiv_rn(qenv, qslcl);
    float cliq = __fmul_rn(17.67f, 273.15f);
    float bliq_dliq = __fmul_rn(17.67f, 29.65f);
    float tm = __fsub_rn(tlcl, 29.65f);
    float dqssdt = __fdiv_rn(__fmul_rn(qmix, __fsub_rn(cliq, bliq_dliq)),
                             __fmul_rn(tm, tm));
    if (rhlcl >= 0.75f && rhlcl <= 0.95f)
        return __fdiv_rn(__fmul_rn(__fmul_rn(0.25f, __fsub_rn(rhlcl, 0.75f)),
                                   qmix), dqssdt);
    if (rhlcl > 0.95f)
        return __fdiv_rn(__fmul_rn(__fsub_rn(__fdiv_rn(1.0f, rhlcl), 1.0f),
                                   qmix), dqssdt);
    return 0.0f;
}

/* QENV = Q0(K) + (Q0(KLCL) - Q0(K))*DLP, module_cu_kfeta.F:947, in its own
 * operation order; DTRH reads this form in both arms. */
__device__ __forceinline__ float kf_lcl_interp(float lower, float upper,
                                               float dlp) {
    return __fadd_rn(lower, __fmul_rn(__fsub_rn(upper, lower), dlp));
}

/* The oracle hook: kf_trigger3_dtrh above, the function kf_column calls,
 * evaluated on inputs WRF recorded at its own trigger evaluations
 * (tools/kfeta_wrf461_oracle, gpuwm.core.kf.probe_trigger3_dtrh). */
extern "C" __global__ __launch_bounds__(256)
void kf_trigger3_dtrh_probe(
        const float* __restrict__ qes_k, const float* __restrict__ qes_klcl,
        const float* __restrict__ dlp, const float* __restrict__ qenv,
        const float* __restrict__ qmix, const float* __restrict__ tlcl,
        float* __restrict__ dtrh, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        dtrh[i] = kf_trigger3_dtrh(qes_k[i], qes_klcl[i], dlp[i], qenv[i],
                                   qmix[i], tlcl[i]);
}
#endif

extern "C" __global__ __launch_bounds__(256)
void kf_column(
        const float* __restrict__ u,
        const float* __restrict__ v,
        const float* __restrict__ temperature,
        const float* __restrict__ qv,
        const float* __restrict__ pressure,
        const float* __restrict__ rho,
        const float* __restrict__ exner,
        const float* __restrict__ dz,
        const float* __restrict__ w0avg,
#if KF_TRIGGER == 2
        const float* __restrict__ tpart,
#endif
        const float* __restrict__ ttab_f,
        const float* __restrict__ qstab_f,
        const float* __restrict__ the0k,
        const float* __restrict__ alu,
        float* __restrict__ rthcuten,
        float* __restrict__ rqvcuten,
        float* __restrict__ rqccuten,
        float* __restrict__ rqicuten,
        float* __restrict__ rqrcuten,
        float* __restrict__ rqscuten,
        float* __restrict__ raincv_out,
        float* __restrict__ pratec_out,
        float* __restrict__ nca_out,
        float* __restrict__ cutop_out,
        float* __restrict__ cubot_out,
        float* __restrict__ shall_out,
        float* __restrict__ cape_before,
        float* __restrict__ cape_after,
        float* __restrict__ timec_out,
        float* __restrict__ updraft_out,
        float* __restrict__ downdraft_out,
        float* __restrict__ cldfra_dp,
        float* __restrict__ cldfra_sh,
        float* __restrict__ qc_kf,
        float* __restrict__ qi_kf,
        float* __restrict__ ws,          // (blocks, KFWS_SLOTS, nz, LANES)
        float plutop, float rdpr, float rdthk,
        float dx, float dt, float cudt,
        int phase_mode, int f_qr, int want_cloud,
        int nz, int ny, int nx, int col0) {
    int column = col0 + blockDim.x * blockIdx.x + threadIdx.x;
    const int ncol = ny * nx;
    if (nz < 8 || nz > KF_KMAX) return;

    float* kfws = KFWS_LANE_BASE(ws, nz);
    __shared__ int order[256];
    __shared__ int counts[8];
    bool predicted = column < ncol && kf_predict_trigger(
        temperature, qv, pressure, dz, w0avg, kfws, nz, ncol, column, dx);
    {
        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        unsigned ballot = __ballot_sync(0xffffffffu, predicted);
        int before = __popc(ballot & ((1u << lane) - 1u));
        if (lane == 0) counts[warp] = __popc(ballot);
        __syncthreads();
        int preceding = 0, total = 0;
        for (int j = 0; j < (int)(blockDim.x / 32); ++j) {
            if (j < warp) preceding += counts[j];
            total += counts[j];
        }
        int position = predicted ? preceding + before
            : total + (int)threadIdx.x - preceding - before;
        order[position] = column;
        __syncthreads();
        column = order[threadIdx.x];
    }
    if (column >= ncol) return;

    /* ---- the column state, Fortran names ---------------------------- */
    KfCol z0 = KFWS_AT(kfws, 0, nz);
    KfCol dp = KFWS_AT(kfws, 1, nz);
    KfCol qes = KFWS_AT(kfws, 2, nz);
    KfCol q0 = KFWS_AT(kfws, 3, nz);
    KfCol rh = KFWS_AT(kfws, 4, nz);
    KfCol dilfrc = KFWS_AT(kfws, 5, nz);
    KfCol tv0 = KFWS_AT(kfws, 6, nz);
    KfCol cldhgt = KFWS_AT(kfws, 7, nz);
    KfCol dza = KFWS_AT(kfws, 8, nz);
    KfCol theteu = KFWS_AT(kfws, 9, nz);
    KfCol tu = KFWS_AT(kfws, 10, nz);
    KfCol tvu = KFWS_AT(kfws, 11, nz);
    KfCol qu = KFWS_AT(kfws, 12, nz);
    KfCol eqfrc = KFWS_AT(kfws, 13, nz);
    KfCol qliq = KFWS_AT(kfws, 14, nz);
    KfCol qice = KFWS_AT(kfws, 15, nz);
    KfCol qlqout = KFWS_AT(kfws, 16, nz);
    KfCol qicout = KFWS_AT(kfws, 17, nz);
    KfCol detlq = KFWS_AT(kfws, 18, nz);
    KfCol detic = KFWS_AT(kfws, 19, nz);
    KfCol pptliq = KFWS_AT(kfws, 20, nz);
    KfCol pptice = KFWS_AT(kfws, 21, nz);
    KfCol uer = KFWS_AT(kfws, 22, nz);
    KfCol umf = KFWS_AT(kfws, 23, nz);
    KfCol udr = KFWS_AT(kfws, 24, nz);
    KfCol thetee = KFWS_AT(kfws, 25, nz);
    KfCol tvqu = KFWS_AT(kfws, 26, nz);
    KfCol qdt = KFWS_AT(kfws, 27, nz);
    KfCol ems = KFWS_AT(kfws, 28, nz);
    KfCol emsd = KFWS_AT(kfws, 29, nz);
    KfCol thtau = KFWS_AT(kfws, 30, nz);
    KfCol thta0 = KFWS_AT(kfws, 31, nz);
    KfCol ddilfrc = KFWS_AT(kfws, 32, nz);
    KfCol omg = KFWS_AT(kfws, 33, nz);
    KfCol tz = KFWS_AT(kfws, 34, nz);
    KfCol qd = KFWS_AT(kfws, 35, nz);
    KfCol thtad = KFWS_AT(kfws, 36, nz);
    KfCol theted = KFWS_AT(kfws, 37, nz);
    KfCol dmf = KFWS_AT(kfws, 38, nz);
    KfCol der = KFWS_AT(kfws, 39, nz);
    KfCol ddr = KFWS_AT(kfws, 40, nz);
    KfCol qsd = KFWS_AT(kfws, 41, nz);
    KfCol detlq2 = KFWS_AT(kfws, 42, nz);
    KfCol detic2 = KFWS_AT(kfws, 43, nz);
    KfCol udr2 = KFWS_AT(kfws, 44, nz);
    KfCol uer2 = KFWS_AT(kfws, 45, nz);
    KfCol ddr2 = KFWS_AT(kfws, 46, nz);
    KfCol der2 = KFWS_AT(kfws, 47, nz);
    KfCol umf2 = KFWS_AT(kfws, 48, nz);
    KfCol dmf2 = KFWS_AT(kfws, 49, nz);
    /* THPA/QPA are dead once the closure has copied them into TG/QG, so
     * the hydrometeor arrays QLPA/QIPA reuse their slots; likewise the
     * theta/q fluxes carry the QL/QI fluxes afterwards. */
    KfCol thpa = KFWS_AT(kfws, 50, nz);
    KfCol qpa = KFWS_AT(kfws, 51, nz);
    KfCol fxm = KFWS_AT(kfws, 52, nz);
    KfCol thfxin = KFWS_AT(kfws, 53, nz);
    KfCol thfxout = KFWS_AT(kfws, 54, nz);
    KfCol qfxin = KFWS_AT(kfws, 55, nz);
    KfCol qfxout = KFWS_AT(kfws, 56, nz);
    KfCol qg = KFWS_AT(kfws, 57, nz);
    KfCol tg = KFWS_AT(kfws, 58, nz);
    KfCol qrpa = KFWS_AT(kfws, 59, nz);
    KfCol qspa = KFWS_AT(kfws, 60, nz);
    KfCol qrfxin = KFWS_AT(kfws, 61, nz);
    KfCol qrfxout = KFWS_AT(kfws, 62, nz);
    KfCol qsfxin = KFWS_AT(kfws, 63, nz);
    KfCol qsfxout = KFWS_AT(kfws, 64, nz);
    KfColI kcheck = KFWS_AT_I(kfws, 65, nz);
    KfCol& qlpa = thpa;
    KfCol& qipa = qpa;
    KfCol& qlfxin = thfxin;
    KfCol& qlfxout = thfxout;
    KfCol& qifxin = qfxin;
    KfCol& qifxout = qfxout;

    const KfIn u0{u, ncol, column}, v0{v, ncol, column};
    const KfIn t0{temperature, ncol, column}, qv0{qv, ncol, column};
    const KfIn p0{pressure, ncol, column}, dzq{dz, ncol, column};
    const KfIn rhoe{rho, ncol, column}, pi3{exner, ncol, column};
    const KfIn w0avg1d{w0avg, ncol, column};
    const KfLut L{ttab_f, qstab_f, the0k, alu, plutop, rdpr, rdthk};
    /* share/module_model_constants.F, as module_cumulus_driver.F passes
     * them to KF_eta_CPS. */
    const float g = 9.81f, r = 287.0f;
    const float cp = __fdiv_rn(__fmul_rn(7.0f, 287.0f), 2.0f);
    const float xlv0 = 3.15e6f, xlv1 = 2370.0f;
    const KfSvp S{__fmul_rn(0.6112f, 1000.0f), 17.67f,
                  __fmul_rn(17.67f, 273.15f), 29.65f};
    const bool warm_rain = phase_mode == KF_PHASE_WARM_RAIN;
    const bool f_qs = phase_mode == KF_PHASE_SEPARATE_SNOW
                      || phase_mode == KF_PHASE_SEPARATE_ICE_SNOW;
    const bool f_qi = phase_mode == KF_PHASE_SEPARATE_ICE_SNOW;

    /* KF_eta_CPS per-column initialisation (:414-440).  The tendencies are
     * written at the end of the column on every path. */
    const int kx = nz, kl = nz;
    if (want_cloud) {
        for (int k = 1; k <= kx; ++k) {
            size_t i3 = (size_t)(k - 1) * ncol + column;
            cldfra_dp[i3] = 0.0f;
            cldfra_sh[i3] = 0.0f;
            qc_kf[i3] = 0.0f;
            qi_kf[i3] = 0.0f;
        }
    }
#define KF_CLD(arr, k, value) \
    do { if (want_cloud && (k) >= 1 && (k) <= kx) \
            (arr)[(size_t)((k) - 1) * ncol + column] = (value); } while (0)

    /* KF_eta_PARA's outputs before any RETURN: DTDT.. = 0, RAINCV/PRATEC
     * = 0, CUTOP = KTS, CUBOT = KTE+1, NCA untouched (0 here). */
    float raincv = 0.0f, pratec = 0.0f, nca = 0.0f;
    float cutop = 1.0f, cubot = (float)(kx + 1);
    int ishall = 0;
    float abe = 0.0f, abeg = 0.0f, timec_kf = 0.0f;
    bool convected = false;

    /* ---------------- KF_eta_PARA ----------------------------------- */
    const float p00 = 1.e5f, t00 = 273.16f, rlf = 3.339e5f;
    const float ttfrz = 268.16f, tbfrz = 248.16f, rate = 0.03f;
    const float gdry = -(g / cp);
    float fbfrc = 0.0f;
    ishall = 0;
    const float dpmin = 5.e3f;
    float p300 = p0(1) - 30000.0f;
    int ml = 0, l5 = 0, llfc = 0;
    for (int k = 1; k <= kx; ++k) {
        float es = S.aliq * gfk_exp((S.bliq * t0(k) - S.cliq)
                                    / (t0(k) - S.dliq));
        qes(k) = (0.622f * es) / (p0(k) - es);
        q0(k) = fminf(qes(k), qv0(k));
        q0(k) = fmaxf(0.000001f, q0(k));
        rh(k) = q0(k) / qes(k);
        dilfrc(k) = 1.0f;
        tv0(k) = t0(k) * (1.0f + 0.608f * q0(k));
        dp(k) = (rhoe(k) * g) * dzq(k);
        cldhgt(k) = 0.0f;
        if (p0(k) >= 0.5f * p0(1)) l5 = k;
        if (p0(k) >= p300) llfc = k;
    }
    z0(1) = 0.5f * dzq(1);
    for (int k = 2; k <= kl; ++k) {
        z0(k) = z0(k - 1) + 0.5f * (dzq(k) + dzq(k - 1));
        dza(k - 1) = z0(k) - z0(k - 1);
    }
    dza(kl) = 0.0f;

    int ncheck = 1;
    kcheck(ncheck) = 1;
    float pm15 = p0(1) - 15.e2f;
    for (int k = 2; k <= llfc; ++k) {
        if (p0(k) < pm15) {
            ncheck = ncheck + 1;
            kcheck(ncheck) = k;
            pm15 = pm15 - 15.e2f;
        }
    }

    int nu = 0, nuchm = 0;
    int lc = 0, kpbl = 0, klcl = 0, k = 0, lcl = 0, let = 0, ltop = 0;
    float dpthmx = 0.0f, tmix = 0.0f, qmix = 0.0f, zmix = 0.0f, pmix = 0.0f;
    float tlcl = 0.0f, tvlcl = 0.0f, zlcl = 0.0f, dlp = 0.0f, tenv = 0.0f;
    float qenv = 0.0f, tven = 0.0f, wkl = 0.0f, dtlcl = 0.0f;
    float wlcl = 0.0f, plcl = 0.0f, wtw = 0.0f, rholcl = 0.0f, rad = 0.0f;
    float au0 = 0.0f, vmflcl = 0.0f, upold = 0.0f, upnew = 0.0f;
    float trppt = 0.0f, chmin = 0.0f;
    const float dxsq = dx * dx;
    bool returned = false;

    for (;;) {                                       /* usl: DO */
        nu = nu + 1;
        if (nu > ncheck) {
            if (ishall == 1) {
                float chmax = 0.0f;
                for (int nk = 1; nk <= ncheck; ++nk) {
                    int nnn = kcheck(nk);
                    if (cldhgt(nnn) > chmax) {
                        nuchm = nk;
                        chmax = cldhgt(nnn);
                    }
                }
                nu = nuchm - 1;
                fbfrc = 1.0f;
                continue;
            } else {
                ishall = 2;
                returned = true;
                break;
            }
        }
        int kmix = kcheck(nu);
        lc = kmix;
        int nlayrs = 0;
        dpthmx = 0.0f;
        int nk = lc - 1;
        if (nk + 1 >= 1) {
            for (;;) {
                nk = nk + 1;
                if (nk > nz) break;
                dpthmx = dpthmx + dp(nk);
                nlayrs = nlayrs + 1;
                if (dpthmx > dpmin) break;
            }
        }
        if (dpthmx < dpmin) {
            ishall = 2;
            returned = true;
            break;
        }
        kpbl = lc + nlayrs - 1;
        tmix = 0.0f; qmix = 0.0f; zmix = 0.0f; pmix = 0.0f;
        for (nk = lc; nk <= kpbl; ++nk) {
            tmix = tmix + dp(nk) * t0(nk);
            qmix = qmix + dp(nk) * q0(nk);
            zmix = zmix + dp(nk) * z0(nk);
            pmix = pmix + dp(nk) * p0(nk);
        }
        tmix = tmix / dpthmx;
        qmix = qmix / dpthmx;
        zmix = zmix / dpthmx;
        pmix = pmix / dpthmx;
        float emix = (qmix * pmix) / (0.622f + qmix);
        tlcl = kf_tlcl(L, S, emix, tmix);
        tvlcl = tlcl * (1.0f + 0.608f * qmix);
        zlcl = zmix + (tlcl - tmix) / gdry;
        for (nk = lc; nk <= kl; ++nk) {
            klcl = nk;
            if (zlcl <= z0(nk)) break;
        }
        if (zlcl > z0(kl)) {
            ishall = 2;
            returned = true;
            break;
        }
        k = klcl - 1;
        if (k < 1) {
            /* KLCL = 1 (a saturated source layer at the ground deeper than
             * 50 hPa): WRF goes on to read Z0(0), T0(0) and W0AVG1D(0),
             * outside its arrays -- undefined in the Fortran.  The port
             * does not evaluate that candidate. */
            continue;
        }
        dlp = (zlcl - z0(k)) / (z0(klcl) - z0(k));
        tenv = t0(k) + (t0(klcl) - t0(k)) * dlp;
        qenv = q0(k) + (q0(klcl) - q0(k)) * dlp;
        tven = tenv * (1.0f + 0.608f * qenv);
        float wklcl;
        if (zlcl < 2.e3f) {
            wklcl = __fdiv_rn(0.02f * zlcl, 2.e3f);
        } else {
            wklcl = 0.02f;
        }
        wkl = __fdiv_rn((w0avg1d(k) + (w0avg1d(klcl) - w0avg1d(k)) * dlp)
                        * dx, 25.e3f) - wklcl;
        if (wkl < 0.0001f) {
            dtlcl = 0.0f;
        } else {
            dtlcl = 4.64f * gfk_pow(wkl, 0.33f);
        }
#if KF_TRIGGER == 2
        dtlcl = fmaxf(tpart[(size_t)(klcl - 1) * ncol + column], 0.0f);
#endif
#if KF_TRIGGER == 3
        const float dtrh = kf_trigger3_dtrh(qes(k), qes(klcl), dlp, qenv, qmix, tlcl);
#else
        const float dtrh = 0.0f;
#endif
        if ((tlcl + dtlcl) + dtrh < tenv) {
            continue;                                /* CYCLE usl */
        }
        theteu(k) = kf_envirtht(L, S, pmix, tmix, qmix);
        float dttot = dtlcl + dtrh;
        if (dttot > 1.e-4f) {
            float gdt = (((2.0f * g) * dttot) * 500.0f) / tven;
            wlcl = 1.0f + 0.5f * sqrtf(gdt);
            wlcl = fminf(wlcl, 3.0f);
        } else {
            wlcl = 1.0f;
        }
        plcl = p0(k) + (p0(klcl) - p0(k)) * dlp;
        wtw = wlcl * wlcl;
        tvlcl = tlcl * (1.0f + 0.608f * qmix);
        rholcl = plcl / (r * tvlcl);
        lcl = klcl;
        let = lcl;
        if (wkl < 0.0f) {
            rad = 1000.0f;
        } else if (wkl > 0.1f) {
            rad = 2000.0f;
        } else {
            rad = 1000.0f + __fdiv_rn(1000.0f * wkl, 0.1f);
        }
        au0 = 0.01f * dxsq;
        umf(k) = rholcl * au0;
        vmflcl = umf(k);
        upold = vmflcl;
        upnew = upold;
        uer(k) = 0.0f;
        abe = 0.0f;
        trppt = 0.0f;
        tu(k) = tlcl;
        tvu(k) = tvlcl;
        qu(k) = qmix;
        eqfrc(k) = 1.0f;
        qliq(k) = 0.0f;
        qice(k) = 0.0f;
        qlqout(k) = 0.0f;
        qicout(k) = 0.0f;
        detlq(k) = 0.0f;
        detic(k) = 0.0f;
        pptliq(k) = 0.0f;
        pptice(k) = 0.0f;
        float ttemp = ttfrz;
        float ee1 = 1.0f, ud1 = 0.0f, rei = 0.0f, dilbe = 0.0f;
        for (nk = k; nk <= kl - 1; ++nk) {           /* updraft */
            int nk1 = nk + 1;
            float frc1 = 0.0f;
            tu(nk1) = t0(nk1);
            theteu(nk1) = theteu(nk);
            qu(nk1) = qu(nk);
            qliq(nk1) = qliq(nk);
            qice(nk1) = qice(nk);
            float qnewlq, qnewic;
            {
                float tu_ = tu(nk1), qu_ = qu(nk1);
                float ql_ = qliq(nk1), qi_ = qice(nk1);
                kf_tpmix2(L, p0(nk1), theteu(nk1), &tu_, &qu_, &ql_, &qi_,
                          &qnewlq, &qnewic, xlv1, xlv0);
                tu(nk1) = tu_; qu(nk1) = qu_;
                qliq(nk1) = ql_; qice(nk1) = qi_;
            }
            if (tu(nk1) <= ttfrz) {
                if (tu(nk1) > tbfrz) {
                    if (ttemp > ttfrz) ttemp = ttfrz;
                    frc1 = (ttemp - tu(nk1)) / (ttemp - tbfrz);
                } else {
                    frc1 = 1.0f;
                }
                ttemp = tu(nk1);
                float qfrz = (qliq(nk1) + qnewlq) * frc1;
                qnewic = qnewic + qnewlq * frc1;
                qnewlq = qnewlq - qnewlq * frc1;
                qice(nk1) = qice(nk1) + qliq(nk1) * frc1;
                qliq(nk1) = qliq(nk1) - qliq(nk1) * frc1;
                float tu_ = tu(nk1), th_ = theteu(nk1);
                float qu_ = qu(nk1), qi_ = qice(nk1);
                kf_dtfrznew(S, &tu_, p0(nk1), &th_, &qu_, qfrz, &qi_);
                tu(nk1) = tu_; theteu(nk1) = th_;
                qu(nk1) = qu_; qice(nk1) = qi_;
            }
            tvu(nk1) = tu(nk1) * (1.0f + 0.608f * qu(nk1));
            float be, boterm, dzz;
            if (nk == k) {
                be = (tvlcl + tvu(nk1)) / (tven + tv0(nk1)) - 1.0f;
                boterm = __fdiv_rn(((2.0f * (z0(nk1) - zlcl)) * g) * be, 1.5f);
                dzz = z0(nk1) - zlcl;
            } else {
                be = (tvu(nk) + tvu(nk1)) / (tv0(nk) + tv0(nk1)) - 1.0f;
                boterm = __fdiv_rn(((2.0f * dza(nk)) * g) * be, 1.5f);
                dzz = dza(nk);
            }
            float enterm = ((2.0f * rei) * wtw) / upold;
            {
                float ql_ = qliq(nk1), qi_ = qice(nk1);
                float qlo_, qio_;
                kf_condload(&ql_, &qi_, &wtw, dzz, boterm, enterm, rate,
                            &qnewlq, &qnewic, &qlo_, &qio_, g);
                qliq(nk1) = ql_; qice(nk1) = qi_;
                qlqout(nk1) = qlo_; qicout(nk1) = qio_;
            }
            if (wtw < 1.e-3f) break;
            thetee(nk1) = kf_envirtht(L, S, p0(nk1), t0(nk1), q0(nk1));
            rei = ((vmflcl * dp(nk1)) * 0.03f) / rad;
            tvqu(nk1) = tu(nk1) * (((1.0f + 0.608f * qu(nk1)) - qliq(nk1))
                                   - qice(nk1));
            if (nk == k) {
                dilbe = ((tvlcl + tvqu(nk1)) / (tven + tv0(nk1)) - 1.0f)
                        * dzz;
            } else {
                dilbe = ((tvqu(nk) + tvqu(nk1)) / (tv0(nk) + tv0(nk1))
                         - 1.0f) * dzz;
            }
            if (dilbe > 0.0f) abe = abe + dilbe * g;
            float ee2, ud2;
            if (tvqu(nk1) <= tv0(nk1)) {
                ee2 = 0.5f;
                ud2 = 1.0f;
                eqfrc(nk1) = 0.0f;
            } else {
                let = nk1;
                float ttmp = tvqu(nk1);
                float f1 = 0.95f;
                float f2 = 1.0f - f1;
                float thttmp = f1 * thetee(nk1) + f2 * theteu(nk1);
                float qtmp = f1 * q0(nk1) + f2 * qu(nk1);
                float tmpliq = f2 * qliq(nk1);
                float tmpice = f2 * qice(nk1);
                kf_tpmix2(L, p0(nk1), thttmp, &ttmp, &qtmp, &tmpliq,
                          &tmpice, &qnewlq, &qnewic, xlv1, xlv0);
                float tu95 = ttmp * (((1.0f + 0.608f * qtmp) - tmpliq)
                                     - tmpice);
                if (tu95 > tv0(nk1)) {
                    ee2 = 1.0f;
                    ud2 = 0.0f;
                    eqfrc(nk1) = 1.0f;
                } else {
                    f1 = 0.10f;
                    f2 = 1.0f - f1;
                    thttmp = f1 * thetee(nk1) + f2 * theteu(nk1);
                    qtmp = f1 * q0(nk1) + f2 * qu(nk1);
                    tmpliq = f2 * qliq(nk1);
                    tmpice = f2 * qice(nk1);
                    kf_tpmix2(L, p0(nk1), thttmp, &ttmp, &qtmp, &tmpliq,
                              &tmpice, &qnewlq, &qnewic, xlv1, xlv0);
                    float tu10 = ttmp * (((1.0f + 0.608f * qtmp) - tmpliq)
                                         - tmpice);
                    float tvdiff = fabsf(tu10 - tvqu(nk1));
                    if (tvdiff < 1.e-3f) {
                        ee2 = 1.0f;
                        ud2 = 0.0f;
                        eqfrc(nk1) = 1.0f;
                    } else {
                        eqfrc(nk1) = ((tv0(nk1) - tvqu(nk1)) * f1)
                                     / (tu10 - tvqu(nk1));
                        eqfrc(nk1) = fmaxf(0.0f, eqfrc(nk1));
                        eqfrc(nk1) = fminf(1.0f, eqfrc(nk1));
                        if (eqfrc(nk1) == 1.0f) {
                            ee2 = 1.0f;
                            ud2 = 0.0f;
                        } else if (eqfrc(nk1) == 0.0f) {
                            ee2 = 0.0f;
                            ud2 = 1.0f;
                        } else {
                            kf_prof5(eqfrc(nk1), &ee2, &ud2);
                        }
                    }
                }
            }
            ee2 = fmaxf(ee2, 0.5f);
            ud2 = 1.5f * ud2;
            uer(nk1) = (0.5f * rei) * (ee1 + ee2);
            udr(nk1) = (0.5f * rei) * (ud1 + ud2);
            if (umf(nk) - udr(nk1) < 10.0f) {
                if (dilbe > 0.0f) abe = abe - dilbe * g;
                let = nk;
                break;
            } else {
                ee1 = ee2;
                ud1 = ud2;
                upold = umf(nk) - udr(nk1);
                upnew = upold + uer(nk1);
                umf(nk1) = upnew;
                dilfrc(nk1) = upnew / upold;
                detlq(nk1) = qliq(nk1) * udr(nk1);
                detic(nk1) = qice(nk1) * udr(nk1);
                qdt(nk1) = qu(nk1);
                qu(nk1) = (upold * qu(nk1) + uer(nk1) * q0(nk1)) / upnew;
                theteu(nk1) = (theteu(nk1) * upold + thetee(nk1) * uer(nk1))
                              / upnew;
                qliq(nk1) = (qliq(nk1) * upold) / upnew;
                qice(nk1) = (qice(nk1) * upold) / upnew;
                pptliq(nk1) = qlqout(nk1) * umf(nk);
                pptice(nk1) = qicout(nk1) * umf(nk);
                trppt = (trppt + pptliq(nk1)) + pptice(nk1);
                if (nk1 <= kpbl)
                    uer(nk1) = uer(nk1) + (vmflcl * dp(nk1)) / dpthmx;
            }
        }                                            /* END updraft */
        ltop = nk;           /* = KL after a loop that ran to completion */
        cldhgt(lc) = z0(ltop) - zlcl;
        if (tlcl > 293.0f) {
            chmin = 4.e3f;
        } else if (tlcl <= 293.0f && tlcl >= 273.0f) {
            chmin = 2.e3f + 100.0f * (tlcl - 273.0f);
        } else if (tlcl < 273.0f) {
            chmin = 2.e3f;
        }
        for (int n = k; n <= ltop; ++n) {
            KF_CLD(qc_kf, n, qliq(n));
            KF_CLD(qi_kf, n, qice(n));
        }
        if (ltop <= klcl || ltop <= kpbl || let + 1 <= kpbl) {
            cldhgt(lc) = 0.0f;
            for (int n = k; n <= ltop; ++n) {
                umf(n) = 0.0f; udr(n) = 0.0f; uer(n) = 0.0f;
                detlq(n) = 0.0f; detic(n) = 0.0f;
                pptliq(n) = 0.0f; pptice(n) = 0.0f;
                KF_CLD(cldfra_dp, n, 0.0f); KF_CLD(cldfra_sh, n, 0.0f);
                KF_CLD(qc_kf, n, 0.0f); KF_CLD(qi_kf, n, 0.0f);
            }
        } else if (cldhgt(lc) > chmin && abe > 1.0f) {
            ishall = 0;
            for (int n = k; n <= ltop; ++n) KF_CLD(cldfra_sh, n, 0.0f);
            break;                                   /* EXIT usl */
        } else {
            ishall = 1;
            for (int n = k; n <= ltop; ++n) KF_CLD(cldfra_dp, n, 0.0f);
            if (nu == nuchm) {
                break;                               /* EXIT usl */
            } else {
                for (int n = k; n <= ltop; ++n) {
                    umf(n) = 0.0f; udr(n) = 0.0f; uer(n) = 0.0f;
                    detlq(n) = 0.0f; detic(n) = 0.0f;
                    pptliq(n) = 0.0f; pptice(n) = 0.0f;
                    KF_CLD(cldfra_dp, n, 0.0f); KF_CLD(cldfra_sh, n, 0.0f);
                    KF_CLD(qc_kf, n, 0.0f); KF_CLD(qi_kf, n, 0.0f);
                }
            }
        }
    }                                                /* END usl */

    float pptflx = 0.0f, cpr = 0.0f, ainc = 0.0f, timec = 0.0f;
    float tadvec = 0.0f, dtime = 0.0f;
    int nstep = 0, nic = 0, lfs = 0, ldb = 0, ldt = 0, kstart = 0;
    bool istop = false;

    if (!returned) do {
    if (ishall == 1) {
        kstart = max(kpbl, klcl);
        let = kstart;
    }
    if (let == ltop) {
        udr(ltop) = (umf(ltop) + udr(ltop)) - uer(ltop);
        detlq(ltop) = ((qliq(ltop) * udr(ltop)) * upnew) / upold;
        detic(ltop) = ((qice(ltop) * udr(ltop)) * upnew) / upold;
        uer(ltop) = 0.0f;
        umf(ltop) = 0.0f;
    } else {
        float dptt = 0.0f;
        for (int nj = let + 1; nj <= ltop; ++nj) dptt = dptt + dp(nj);
        float dumfdp = umf(let) / dptt;
        for (int nk = let + 1; nk <= ltop; ++nk) {
            if (nk == ltop) {
                udr(nk) = umf(nk - 1);
                uer(nk) = 0.0f;
                detlq(nk) = (udr(nk) * qliq(nk)) * dilfrc(nk);
                detic(nk) = (udr(nk) * qice(nk)) * dilfrc(nk);
            } else {
                umf(nk) = umf(nk - 1) - dp(nk) * dumfdp;
                uer(nk) = umf(nk) * (1.0f - 1.0f / dilfrc(nk));
                udr(nk) = (umf(nk - 1) - umf(nk)) + uer(nk);
                detlq(nk) = (udr(nk) * qliq(nk)) * dilfrc(nk);
                detic(nk) = (udr(nk) * qice(nk)) * dilfrc(nk);
            }
            if (nk >= let + 2) {
                trppt = (trppt - pptliq(nk)) - pptice(nk);
                pptliq(nk) = umf(nk - 1) * qlqout(nk);
                pptice(nk) = umf(nk - 1) * qicout(nk);
                trppt = (trppt + pptliq(nk)) + pptice(nk);
            }
        }
    }
    for (int nk = 1; nk <= ltop; ++nk)
        if (t0(nk) > t00) ml = nk;
    for (int nk = 1; nk <= k; ++nk) {
        if (nk >= lc) {
            if (nk == lc) {
                umf(nk) = (vmflcl * dp(nk)) / dpthmx;
                uer(nk) = (vmflcl * dp(nk)) / dpthmx;
            } else if (nk <= kpbl) {
                uer(nk) = (vmflcl * dp(nk)) / dpthmx;
                umf(nk) = umf(nk - 1) + uer(nk);
            } else {
                umf(nk) = vmflcl;
                uer(nk) = 0.0f;
            }
            tu(nk) = tmix + (z0(nk) - zmix) * gdry;
            qu(nk) = qmix;
        } else {
            tu(nk) = 0.0f;
            qu(nk) = 0.0f;
            umf(nk) = 0.0f;
            uer(nk) = 0.0f;
            KF_CLD(cldfra_dp, nk, 0.0f); KF_CLD(cldfra_sh, nk, 0.0f);
            KF_CLD(qc_kf, nk, 0.0f); KF_CLD(qi_kf, nk, 0.0f);
        }
        udr(nk) = 0.0f;
        qdt(nk) = 0.0f;
        qliq(nk) = 0.0f;
        qice(nk) = 0.0f;
        qlqout(nk) = 0.0f;
        qicout(nk) = 0.0f;
        pptliq(nk) = 0.0f;
        pptice(nk) = 0.0f;
        detlq(nk) = 0.0f;
        detic(nk) = 0.0f;
        thetee(nk) = kf_envirtht(L, S, p0(nk), t0(nk), q0(nk));
        eqfrc(nk) = 1.0f;
    }
    const int ltop1 = ltop + 1, ltopm1 = ltop - 1;
    for (int nk = ltop1; nk <= kx; ++nk) {
        umf(nk) = 0.0f; udr(nk) = 0.0f; uer(nk) = 0.0f; qdt(nk) = 0.0f;
        qliq(nk) = 0.0f; qice(nk) = 0.0f; qlqout(nk) = 0.0f;
        qicout(nk) = 0.0f; detlq(nk) = 0.0f; detic(nk) = 0.0f;
        pptliq(nk) = 0.0f; pptice(nk) = 0.0f;
        if (nk > ltop1) {
            tu(nk) = 0.0f;
            qu(nk) = 0.0f;
            KF_CLD(cldfra_dp, nk, 0.0f); KF_CLD(cldfra_sh, nk, 0.0f);
            KF_CLD(qc_kf, nk, 0.0f); KF_CLD(qi_kf, nk, 0.0f);
        }
        thta0(nk) = 0.0f;
        thtau(nk) = 0.0f;
        ems(nk) = 0.0f;
        emsd(nk) = 0.0f;
        tg(nk) = t0(nk);
        qg(nk) = q0(nk);
        omg(nk) = 0.0f;
    }
    for (int nk = 1; nk <= ltop; ++nk) {
        ems(nk) = (dp(nk) * dxsq) / g;
        emsd(nk) = 1.0f / ems(nk);
        float exn = gfk_pow(p00 / p0(nk), 0.2854f * (1.0f - 0.28f * qdt(nk)));
        thtau(nk) = tu(nk) * exn;
        exn = gfk_pow(p00 / p0(nk), 0.2854f * (1.0f - 0.28f * q0(nk)));
        thta0(nk) = t0(nk) * exn;
        ddilfrc(nk) = 1.0f / dilfrc(nk);
        omg(nk) = 0.0f;
    }

    /* Convective time scale, shear and cloud-base precipitation
     * efficiency (:1588-1632). */
    float wspd_klcl = sqrtf(u0(klcl) * u0(klcl) + v0(klcl) * v0(klcl));
    float wspd_l5 = sqrtf(u0(l5) * u0(l5) + v0(l5) * v0(l5));
    float wspd_ltop = sqrtf(u0(ltop) * u0(ltop) + v0(ltop) * v0(ltop));
    float vconv = 0.5f * (wspd_klcl + wspd_l5);
    timec = dx / vconv;
    tadvec = timec;
    timec = fmaxf(1800.0f, timec);
    timec = fminf(3600.0f, timec);
    if (ishall == 1) timec = 2400.0f;
    nic = kf_nint(timec / dt);
    timec = (float)nic * dt;
    float shsign = wspd_ltop > wspd_klcl ? 1.0f : -1.0f;
    float vws = (u0(ltop) - u0(klcl)) * (u0(ltop) - u0(klcl))
                + (v0(ltop) - v0(klcl)) * (v0(ltop) - v0(klcl));
    vws = ((1.e3f * shsign) * sqrtf(vws)) / (z0(ltop) - z0(lcl));
    float pef = 1.591f + vws * (-0.639f + vws * (9.53e-2f - vws * 4.96e-3f));
    pef = fmaxf(pef, 0.2f);
    pef = fminf(pef, 0.9f);
    float cbh = (zlcl - z0(1)) * 3.281e-3f;
    float rcbh;
    if (cbh < 3.0f) {
        rcbh = 0.02f;
    } else {
        rcbh = 0.96729352f + cbh * (-0.70034167f + cbh * (0.162179896f
               + cbh * (-1.2569798e-2f + cbh * (4.2772e-4f
               - cbh * 5.44e-6f))));
    }
    if (cbh > 25.0f) rcbh = 2.4f;
    float pefcbh = 1.0f / (1.0f + rcbh);
    pefcbh = fminf(pefcbh, 0.9f);
    float peff = 0.5f * (pef + pefcbh);

    /* Downdraft (:1647-1873). */
    float tder = 0.0f, dmffrc = 0.0f;
    if (ishall == 1) {
        lfs = 1;
    } else {
        kstart = kpbl + 1;
        int klfs = let - 1;
        for (int nk = kstart + 1; nk <= kl; ++nk) {
            float dppp = p0(kstart) - p0(nk);
            if (dppp > 150.e2f) {
                klfs = nk;
                break;
            }
        }
        klfs = min(klfs, let - 1);
        lfs = klfs;
        if ((p0(kstart) - p0(lfs)) > 50.e2f) {
            theted(lfs) = thetee(lfs);
            qd(lfs) = q0(lfs);
            float qss;
            {
                float tz_;
                kf_tpmix2dd(L, p0(lfs), theted(lfs), &tz_, &qss);
                tz(lfs) = tz_;
            }
            thtad(lfs) = tz(lfs) * gfk_pow(p00 / p0(lfs),
                                           0.2854f * (1.0f - 0.28f * qss));
            float tvd_lfs = tz(lfs) * (1.0f + 0.608f * qss);
            float rdd = p0(lfs) / (r * tvd_lfs);
            float a1 = (1.0f - peff) * au0;
            dmf(lfs) = -a1 * rdd;
            der(lfs) = dmf(lfs);
            ddr(lfs) = 0.0f;
            float rhbar = rh(lfs) * dp(lfs);
            float dptt = dp(lfs);
            for (int nd = lfs - 1; nd >= kstart; --nd) {
                int nd1 = nd + 1;
                der(nd) = (der(lfs) * ems(nd)) / ems(lfs);
                ddr(nd) = 0.0f;
                dmf(nd) = dmf(nd1) + der(nd);
                theted(nd) = (theted(nd1) * dmf(nd1) + thetee(nd) * der(nd))
                             / dmf(nd);
                qd(nd) = (qd(nd1) * dmf(nd1) + q0(nd) * der(nd)) / dmf(nd);
                dptt = dptt + dp(nd);
                rhbar = rhbar + rh(nd) * dp(nd);
            }
            rhbar = rhbar / dptt;
            dmffrc = 2.0f * (1.0f - rhbar);
            float dpdd = 0.0f;
            float pptmlt = 0.0f;
            for (int nk = klcl; nk <= ltop; ++nk) pptmlt = pptmlt + pptice(nk);
            float dtmelt;
            if (lc < ml) {
                dtmelt = (rlf * pptmlt) / (cp * umf(klcl));
            } else {
                dtmelt = 0.0f;
            }
            ldt = min(lfs - 1, kstart - 1);
            {
                float tz_;
                kf_tpmix2dd(L, p0(kstart), theted(kstart), &tz_, &qss);
                tz(kstart) = tz_;
            }
            tz(kstart) = tz(kstart) - dtmelt;
            float es = S.aliq * gfk_exp((S.bliq * tz(kstart) - S.cliq)
                                        / (tz(kstart) - S.dliq));
            qss = (0.622f * es) / (p0(kstart) - es);
            theted(kstart) = (tz(kstart)
                * gfk_pow(1.e5f / p0(kstart), 0.2854f * (1.0f - 0.28f * qss)))
                * gfk_exp(((3374.6525f / tz(kstart) - 2.5403f) * qss)
                          * (1.0f + 0.81f * qss));
            ldt = min(lfs - 1, kstart - 1);
            for (int nd = ldt; nd >= 1; --nd) {
                dpdd = dpdd + dp(nd);
                theted(nd) = theted(kstart);
                qd(nd) = qd(kstart);
                {
                    float tz_;
                    kf_tpmix2dd(L, p0(nd), theted(nd), &tz_, &qss);
                    tz(nd) = tz_;
                }
                qsd(nd) = qss;
                float rhh = 1.0f - (0.2f / 1000.0f) * (z0(kstart) - z0(nd));
                if (rhh < 1.0f) {
                    float dssdt = (S.cliq - S.bliq * S.dliq)
                        / ((tz(nd) - S.dliq) * (tz(nd) - S.dliq));
                    float rl = xlv0 - xlv1 * tz(nd);
                    float dtmp = ((rl * qss) * (1.0f - rhh))
                                 / (cp + ((rl * rhh) * qss) * dssdt);
                    float t1rh = tz(nd) + dtmp;
                    es = (rhh * S.aliq) * gfk_exp((S.bliq * t1rh - S.cliq)
                                                  / (t1rh - S.dliq));
                    float qsrh = (0.622f * es) / (p0(nd) - es);
                    if (qsrh < qd(nd)) {
                        qsrh = qd(nd);
                        t1rh = tz(nd) + ((qss - qsrh) * rl) / cp;
                    }
                    tz(nd) = t1rh;
                    qss = qsrh;
                    qsd(nd) = qss;
                }
                float tvd = tz(nd) * (1.0f + 0.608f * qsd(nd));
                if (tvd > tv0(nd) || nd == 1) {
                    ldb = nd;
                    break;
                }
            }
            if ((p0(ldb) - p0(lfs)) > 50.e2f) {
                for (int nd = ldt; nd >= ldb; --nd) {
                    int nd1 = nd + 1;
                    ddr(nd) = -((dmf(kstart) * dp(nd)) / dpdd);
                    der(nd) = 0.0f;
                    dmf(nd) = dmf(nd1) + ddr(nd);
                    tder = tder + (qsd(nd) - qd(nd)) * ddr(nd);
                    qd(nd) = qsd(nd);
                    thtad(nd) = tz(nd) * gfk_pow(p00 / p0(nd),
                        0.2854f * (1.0f - 0.28f * qd(nd)));
                }
            }
        }
    }
    if (tder < 1.0f) {                               /* d_mf */
        pptflx = trppt;
        cpr = trppt;
        tder = 0.0f;
        ldb = lfs;
        for (int ndk = 1; ndk <= ltop; ++ndk) {
            dmf(ndk) = 0.0f; der(ndk) = 0.0f; ddr(ndk) = 0.0f;
            thtad(ndk) = 0.0f; tz(ndk) = 0.0f; qd(ndk) = 0.0f;
        }
    } else {
        float ddinc = -((dmffrc * umf(klcl)) / dmf(kstart));
        if (tder * ddinc > trppt) ddinc = trppt / tder;
        tder = tder * ddinc;
        for (int nk = ldb; nk <= lfs; ++nk) {
            dmf(nk) = dmf(nk) * ddinc;
            der(nk) = der(nk) * ddinc;
            ddr(nk) = ddr(nk) * ddinc;
        }
        cpr = trppt;
        /* Declared divergence from WRF (d547dc8c3, kept on the port): when
         * the cap above sets DDINC = TRPPT/TDER the downdraft evaporates
         * all of the updraft's precipitation and the float64 answer is 0,
         * but in binary32 TDER*DDINC rounds to either side of TRPPT and
         * WRF's bare TRPPT-TDER (module_cu_kfeta.F:1810-1821) leaves minus
         * a few ulp.  RAINC accumulates PRATEC*DT signed, so it went
         * negative on a global 60 km hex forecast (11 columns by hour 1,
         * minimum -3.5e-8 mm) under the pre-port kernel.  WRF's own
         * arithmetic gives exactly 0 on those seven columns
         * (tests/data/oracles/kf/oracle-capped-O0.out), and no column of
         * 37,101 run through the WRF Fortran (the committed fixtures, those
         * seven and two seeded sweeps) has a negative PRATEC, but TDER*DDINC can
         * still round above TRPPT, so the floor stays: zero in both
         * arithmetic modes, the only place the port leaves WRF on purpose. */
        pptflx = fmaxf(trppt - tder, 0.0f);
        peff = pptflx / trppt;
        if (ldb > 1) {
            for (int nk = 1; nk <= ldb - 1; ++nk) {
                dmf(nk) = 0.0f; der(nk) = 0.0f; ddr(nk) = 0.0f;
                tz(nk) = 0.0f; qd(nk) = 0.0f; thtad(nk) = 0.0f;
            }
        }
        for (int nk = lfs + 1; nk <= kx; ++nk) {
            dmf(nk) = 0.0f; der(nk) = 0.0f; ddr(nk) = 0.0f;
            tz(nk) = 0.0f; qd(nk) = 0.0f; thtad(nk) = 0.0f;
        }
        for (int nk = ldt + 1; nk <= lfs - 1; ++nk) {
            tz(nk) = 0.0f; qd(nk) = 0.0f; thtad(nk) = 0.0f;
        }
    }

    /* Closure (:1879-2281). */
    float aincmx = 1000.0f;
    int lmax = max(klcl, lfs);
    for (int nk = lc; nk <= lmax; ++nk) {
        if ((uer(nk) - der(nk)) > 1.e-3f) {
            float aincm1 = ems(nk) / ((uer(nk) - der(nk)) * timec);
            aincmx = fminf(aincmx, aincm1);
        }
    }
    ainc = 1.0f;
    if (aincmx < ainc) ainc = aincmx;
    float tder2 = tder, pptfl2 = pptflx;
    for (int nk = 1; nk <= ltop; ++nk) {
        detlq2(nk) = detlq(nk); detic2(nk) = detic(nk);
        udr2(nk) = udr(nk); uer2(nk) = uer(nk);
        ddr2(nk) = ddr(nk); der2(nk) = der(nk);
        umf2(nk) = umf(nk); dmf2(nk) = dmf(nk);
    }
    float fabe = 1.0f, fabeold = 0.0f, aincold = 0.0f;
    const float stab = 0.95f;
    int noitr = 0;
    if (ishall == 1) {
        const float tkemax = 5.0f;
        float evac = (0.5f * tkemax) * 0.1f;
        ainc = ((evac * dpthmx) * dxsq) / ((vmflcl * g) * timec);
        tder = tder2 * ainc;
        pptflx = pptfl2 * ainc;
        for (int nk = 1; nk <= ltop; ++nk) {
            umf(nk) = umf2(nk) * ainc; dmf(nk) = dmf2(nk) * ainc;
            detlq(nk) = detlq2(nk) * ainc; detic(nk) = detic2(nk) * ainc;
            udr(nk) = udr2(nk) * ainc; uer(nk) = uer2(nk) * ainc;
            der(nk) = der2(nk) * ainc; ddr(nk) = ddr2(nk) * ainc;
        }
    }
    for (int ncount = 1; ncount <= 10; ++ncount) {   /* iter */
        float dtt = timec;
        float domgdp_prev = 0.0f;
        for (int nk = 1; nk <= ltop; ++nk) {
            float domgdp = -((((uer(nk) - der(nk)) - udr(nk)) - ddr(nk))
                             * emsd(nk));
            if (nk > 1) {
                omg(nk) = omg(nk - 1) - dp(nk - 1) * domgdp_prev;
                float absomg = fabsf(omg(nk));
                float absomgtc = absomg * timec;
                float frdp = 0.75f * dp(nk - 1);
                if (absomgtc > frdp) {
                    float dtt1 = frdp / absomg;
                    dtt = fminf(dtt, dtt1);
                }
            }
            domgdp_prev = domgdp;
        }
        for (int nk = 1; nk <= ltop; ++nk) {
            thpa(nk) = thta0(nk);
            qpa(nk) = q0(nk);
            nstep = kf_nint(timec / dtt + 1.0f);
            dtime = timec / (float)nstep;
            fxm(nk) = (omg(nk) * dxsq) / g;
        }
        for (int ntc = 1; ntc <= nstep; ++ntc) {
            for (int nk = 1; nk <= ltop; ++nk) {
                thfxin(nk) = 0.0f; thfxout(nk) = 0.0f;
                qfxin(nk) = 0.0f; qfxout(nk) = 0.0f;
            }
            for (int nk = 2; nk <= ltop; ++nk) {
                if (omg(nk) <= 0.0f) {
                    thfxin(nk) = -fxm(nk) * thpa(nk - 1);
                    qfxin(nk) = -fxm(nk) * qpa(nk - 1);
                    thfxout(nk - 1) = thfxout(nk - 1) + thfxin(nk);
                    qfxout(nk - 1) = qfxout(nk - 1) + qfxin(nk);
                } else {
                    thfxout(nk) = fxm(nk) * thpa(nk);
                    qfxout(nk) = fxm(nk) * qpa(nk);
                    thfxin(nk - 1) = thfxin(nk - 1) + thfxout(nk);
                    qfxin(nk - 1) = qfxin(nk - 1) + qfxout(nk);
                }
            }
            for (int nk = 1; nk <= ltop; ++nk) {
                thpa(nk) = thpa(nk) + ((((((thfxin(nk) + udr(nk) * thtau(nk))
                    + ddr(nk) * thtad(nk)) - thfxout(nk))
                    - (uer(nk) - der(nk)) * thta0(nk)) * dtime) * emsd(nk));
                qpa(nk) = qpa(nk) + ((((((qfxin(nk) + udr(nk) * qdt(nk))
                    + ddr(nk) * qd(nk)) - qfxout(nk))
                    - (uer(nk) - der(nk)) * q0(nk)) * dtime) * emsd(nk));
            }
        }
        for (int nk = 1; nk <= ltop; ++nk) qg(nk) = qpa(nk);
        for (int nk = 1; nk <= ltop; ++nk) {
            if (qg(nk) < 0.0f) {
                if (nk == 1) {
                    /* wrf_error_fatal('QG, QG(NK).LT.0'): WRF stops the
                     * model (and would next read QG(0)).  WOOF gives the
                     * column no convection; the oracle counts any such
                     * column as a WRF abort, never as a match. */
                    istop = true;
                    break;
                }
                int nk1 = nk + 1;
                if (nk == ltop) nk1 = klcl;
                float tma = qg(nk1) * ems(nk1);
                float tmb = qg(nk - 1) * ems(nk - 1);
                float tmm = (qg(nk) - 1.e-9f) * ems(nk);
                float bcoeff = -(tmm / ((tma * tma) / tmb + tmb));
                float acoeff = (bcoeff * tma) / tmb;
                tmb = tmb * (1.0f - bcoeff);
                tma = tma * (1.0f - acoeff);
                qg(nk) = 1.e-9f;
                qg(nk1) = tma * emsd(nk1);
                qg(nk - 1) = tmb * emsd(nk - 1);
            }
        }
        if (istop) break;
        float topomg = ((udr(ltop) - uer(ltop)) * dp(ltop)) * emsd(ltop);
        if (fabsf(topomg - omg(ltop)) > 1.e-3f) {
            istop = true;                    /* wrf_error_fatal in WRF */
            break;
        }
        for (int nk = 1; nk <= ltop; ++nk) {
            float exn = gfk_pow(p00 / p0(nk), 0.2854f * (1.0f - 0.28f * qg(nk)));
            tg(nk) = thpa(nk) / exn;
        }
        if (ishall == 1) break;
        tmix = 0.0f;
        qmix = 0.0f;
        for (int nk = lc; nk <= kpbl; ++nk) {
            tmix = tmix + dp(nk) * tg(nk);
            qmix = qmix + dp(nk) * qg(nk);
        }
        tmix = tmix / dpthmx;
        qmix = qmix / dpthmx;
        float es = S.aliq * gfk_exp((tmix * S.bliq - S.cliq) / (tmix - S.dliq));
        float qss = (0.622f * es) / (pmix - es);
        if (qmix > qss) {
            float rl = xlv0 - xlv1 * tmix;
            float cpm = cp * (1.0f + 0.887f * qmix);
            float dssdt = (qss * (S.cliq - S.bliq * S.dliq))
                          / ((tmix - S.dliq) * (tmix - S.dliq));
            float dq = (qmix - qss) / (1.0f + (rl * dssdt) / cpm);
            tmix = tmix + (rl / cp) * dq;
            qmix = qmix - dq;
            tlcl = tmix;
        } else {
            qmix = fmaxf(qmix, 0.0f);
            float emix = (qmix * pmix) / (0.622f + qmix);
            tlcl = kf_tlcl(L, S, emix, tmix);
        }
        tvlcl = tlcl * (1.0f + 0.608f * qmix);
        zlcl = zmix + (tlcl - tmix) / gdry;
        for (int nk = lc; nk <= kl; ++nk) {
            klcl = nk;
            if (zlcl <= z0(nk)) break;
        }
        k = klcl - 1;
        if (k < 1) {                     /* Z0(0): see the trigger's note */
            ishall = 2;
            returned = true;
            break;
        }
        dlp = (zlcl - z0(k)) / (z0(klcl) - z0(k));
        tenv = tg(k) + (tg(klcl) - tg(k)) * dlp;
        qenv = qg(k) + (qg(klcl) - qg(k)) * dlp;
        tven = tenv * (1.0f + 0.608f * qenv);
        plcl = p0(k) + (p0(klcl) - p0(k)) * dlp;
        theteu(k) = (tmix * gfk_pow(1.e5f / pmix,
                                    0.2854f * (1.0f - 0.28f * qmix)))
            * gfk_exp(((3374.6525f / tlcl - 2.5403f) * qmix)
                      * (1.0f + 0.81f * qmix));
        abeg = 0.0f;
        float tvqu_prev = 0.0f;
        for (int nk = k; nk <= ltopm1; ++nk) {
            int nk1 = nk + 1;
            theteu(nk1) = theteu(nk);
            float tgu, qgu;
            kf_tpmix2dd(L, p0(nk1), theteu(nk1), &tgu, &qgu);
            float tvqu1 = tgu * (((1.0f + 0.608f * qgu) - qliq(nk1))
                                 - qice(nk1));
            float tvg1 = tg(nk1) * (1.0f + 0.608f * qg(nk1));
            float dzz, dilbe;
            if (nk == k) {
                dzz = z0(klcl) - zlcl;
                dilbe = ((tvlcl + tvqu1) / (tven + tvg1) - 1.0f) * dzz;
            } else {
                float tvg0 = tg(nk) * (1.0f + 0.608f * qg(nk));
                dzz = dza(nk);
                dilbe = ((tvqu_prev + tvqu1) / (tvg0 + tvg1) - 1.0f) * dzz;
            }
            if (dilbe > 0.0f) abeg = abeg + dilbe * g;
            float thteeg = kf_envirtht(L, S, p0(nk1), tg(nk1), qg(nk1));
            theteu(nk1) = theteu(nk1) * ddilfrc(nk1)
                          + thteeg * (1.0f - ddilfrc(nk1));
            tvqu_prev = tvqu1;
        }
        if (noitr == 1) break;
        float dabe = fmaxf(abe - abeg, 0.1f * abe);
        fabe = abeg / abe;
        if (fabe > 1.0f && ishall == 0) {
            ishall = 2;
            returned = true;
            break;
        }
        if (ncount != 1) {
            if (fabsf(ainc - aincold) < 0.0001f) {
                noitr = 1;
                ainc = aincold;
                continue;
            }
            float dfda = (fabe - fabeold) / (ainc - aincold);
            if (dfda > 0.0f) {
                noitr = 1;
                ainc = aincold;
                continue;
            }
        }
        aincold = ainc;
        fabeold = fabe;
        if (ainc / aincmx > 0.999f && fabe > 1.05f - stab) break;
        if ((fabe <= 1.05f - stab && fabe >= 0.95f - stab) || ncount == 10) {
            break;
        } else {
            if (fabe == 0.0f) {
                ainc = ainc * 0.5f;
            } else {
                if (dabe < 1.e-4f) {
                    noitr = 1;
                    ainc = aincold;
                    continue;
                } else {
                    ainc = ((ainc * stab) * abe) / dabe;
                }
            }
            ainc = fminf(aincmx, ainc);
            if (ainc < 0.05f) {
                ishall = 2;
                returned = true;
                break;
            }
            tder = tder2 * ainc;
            pptflx = pptfl2 * ainc;
            for (int nk = 1; nk <= ltop; ++nk) {
                umf(nk) = umf2(nk) * ainc; dmf(nk) = dmf2(nk) * ainc;
                detlq(nk) = detlq2(nk) * ainc; detic(nk) = detic2(nk) * ainc;
                udr(nk) = udr2(nk) * ainc; uer(nk) = uer2(nk) * ainc;
                der(nk) = der2(nk) * ainc; ddr(nk) = ddr2(nk) * ainc;
            }
        }
    }                                                /* END iter */
    if (returned) break;
    if (istop) {
        /* WRF stops the model on both istop paths: wrf_error_fatal
         * 'QG, QG(NK).LT.0' (:2033) and 'KAIN-FRITSCH, istop=1, diags'
         * (:2498, the cloud-top mass budget did not close).  WOOF keeps
         * running and gives the column no convection; the oracle counts
         * any such column as a WRF abort, never as a match. */
        ishall = 2;
        returned = true;
        break;
    }

    /* Cloud fraction (:2284-2298), at the closure's KLCL. */
    if (want_cloud) {
        for (int nk = klcl - 1; nk <= ltop1; ++nk) {
            if (nk < 1 || nk > kx) continue;     /* WRF reads past UMF */
            float umf_new = umf(nk) / dxsq;
            if (ishall == 1) {
                float x = 0.07f * gfk_log(1.0f + (500.0f * umf_new));
                x = fmaxf(0.01f, x);
                KF_CLD(cldfra_sh, nk, fminf(0.2f, x));
            } else {
                float x = 0.14f * gfk_log(1.0f + (500.0f * umf_new));
                x = fmaxf(0.01f, x);
                KF_CLD(cldfra_dp, nk, fminf(0.6f, x));
            }
        }
    }

    /* Hydrometeors (:2318-2382). */
    float frc2;
    if (cpr > 0.0f) {
        frc2 = pptflx / (cpr * ainc);
    } else {
        frc2 = 0.0f;
    }
    for (int nk = 1; nk <= ltop; ++nk) {
        qlpa(nk) = 0.0f;
        qipa(nk) = 0.0f;
        qrpa(nk) = 0.0f;
        qspa(nk) = 0.0f;
    }
    for (int ntc = 1; ntc <= nstep; ++ntc) {
        for (int nk = 1; nk <= ltop; ++nk) {
            qlfxin(nk) = 0.0f; qlfxout(nk) = 0.0f;
            qifxin(nk) = 0.0f; qifxout(nk) = 0.0f;
            qrfxin(nk) = 0.0f; qrfxout(nk) = 0.0f;
            qsfxin(nk) = 0.0f; qsfxout(nk) = 0.0f;
        }
        for (int nk = 2; nk <= ltop; ++nk) {
            if (omg(nk) <= 0.0f) {
                qlfxin(nk) = -fxm(nk) * qlpa(nk - 1);
                qifxin(nk) = -fxm(nk) * qipa(nk - 1);
                qrfxin(nk) = -fxm(nk) * qrpa(nk - 1);
                qsfxin(nk) = -fxm(nk) * qspa(nk - 1);
                qlfxout(nk - 1) = qlfxout(nk - 1) + qlfxin(nk);
                qifxout(nk - 1) = qifxout(nk - 1) + qifxin(nk);
                qrfxout(nk - 1) = qrfxout(nk - 1) + qrfxin(nk);
                qsfxout(nk - 1) = qsfxout(nk - 1) + qsfxin(nk);
            } else {
                qlfxout(nk) = fxm(nk) * qlpa(nk);
                qifxout(nk) = fxm(nk) * qipa(nk);
                qrfxout(nk) = fxm(nk) * qrpa(nk);
                qsfxout(nk) = fxm(nk) * qspa(nk);
                qlfxin(nk - 1) = qlfxin(nk - 1) + qlfxout(nk);
                qifxin(nk - 1) = qifxin(nk - 1) + qifxout(nk);
                qrfxin(nk - 1) = qrfxin(nk - 1) + qrfxout(nk);
                qsfxin(nk - 1) = qsfxin(nk - 1) + qsfxout(nk);
            }
        }
        for (int nk = 1; nk <= ltop; ++nk) {
            float rainfb = ((pptliq(nk) * ainc) * fbfrc) * frc2;
            float snowfb = ((pptice(nk) * ainc) * fbfrc) * frc2;
            qlpa(nk) = qlpa(nk) + (((qlfxin(nk) + detlq(nk)) - qlfxout(nk))
                                   * dtime) * emsd(nk);
            qipa(nk) = qipa(nk) + (((qifxin(nk) + detic(nk)) - qifxout(nk))
                                   * dtime) * emsd(nk);
            qrpa(nk) = qrpa(nk) + (((qrfxin(nk) - qrfxout(nk)) + rainfb)
                                   * dtime) * emsd(nk);
            qspa(nk) = qspa(nk) + (((qsfxin(nk) - qsfxout(nk)) + snowfb)
                                   * dtime) * emsd(nk);
        }
    }
    timec_kf = timec;
    pratec = (pptflx * (1.0f - fbfrc)) / dxsq;
    raincv = dt * pratec;
    if (tadvec < timec) nic = kf_nint(tadvec / dt);
    nca = (float)nic * dt;
    if (ishall == 1) {
        timec = 2400.0f;
        nca = cudt * 60.0f;
    }
    convected = true;
    } while (0);

    /* Feedback (:2577-2643) and KF_eta_CPS's mapping (:485-524). */
    for (int kk = 1; kk <= kx; ++kk) {
        size_t i3 = (size_t)(kk - 1) * ncol + column;
        float dtdt = 0.0f, dqdt = 0.0f, dqcdt = 0.0f, dqidt = 0.0f;
        float dqrdt = 0.0f, dqsdt = 0.0f;
        if (convected) {
            float tgk, qgk, qlg = 0.0f, qig = 0.0f, qrg = 0.0f, qsg = 0.0f;
            if (kk <= ltop) {
                tgk = tg(kk);
                qgk = qg(kk);
                qlg = qlpa(kk);
                qig = qipa(kk);
                qrg = qrpa(kk);
                qsg = qspa(kk);
            } else {
                tgk = t0(kk);
                qgk = q0(kk);
            }
            if (warm_rain) {
                float cpm = cp * (1.0f + 0.887f * qgk);
                tgk = tgk - ((qig + qsg) * rlf) / cpm;
                dqcdt = (((qlg + qig) - 0.0f) - 0.0f) / timec;
                dqidt = 0.0f;
                dqrdt = (((qrg + qsg) - 0.0f) - 0.0f) / timec;
                dqsdt = 0.0f;
            } else if (!f_qs) {
                float cpm = cp * (1.0f + 0.887f * qgk);
                if (kk <= ml) {
                    tgk = tgk - ((qig + qsg) * rlf) / cpm;
                } else if (kk > ml) {
                    tgk = tgk + ((qlg + qrg) * rlf) / cpm;
                }
                dqcdt = (((qlg + qig) - 0.0f) - 0.0f) / timec;
                dqidt = 0.0f;
                dqrdt = (((qrg + qsg) - 0.0f) - 0.0f) / timec;
                dqsdt = 0.0f;
            } else {
                dqcdt = (qlg - 0.0f) / timec;
                dqsdt = (qsg - 0.0f) / timec;
                dqrdt = (qrg - 0.0f) / timec;
                if (f_qi) {
                    dqidt = (qig - 0.0f) / timec;
                } else {
                    dqsdt = dqsdt + (qig - 0.0f) / timec;
                }
            }
            dtdt = (tgk - t0(kk)) / timec;
            dqdt = (qgk - q0(kk)) / timec;
        }
        rthcuten[i3] = dtdt / pi3(kk);
        rqvcuten[i3] = dqdt;
        if (f_qr) {
            rqrcuten[i3] = dqrdt;
            rqccuten[i3] = dqcdt;
        } else {
            rqrcuten[i3] = 0.0f;
            rqccuten[i3] = dqrdt + dqcdt;
        }
        rqicuten[i3] = f_qi ? dqidt : 0.0f;
        rqscuten[i3] = f_qs ? dqsdt : 0.0f;
        updraft_out[i3] = convected ? umf(kk) : 0.0f;
        downdraft_out[i3] = (convected && kk <= ltop) ? dmf(kk) : 0.0f;
    }
    if (convected) {
        cutop = (float)ltop;
        cubot = (float)lcl;
    }
    raincv_out[column] = raincv;
    pratec_out[column] = pratec;
    nca_out[column] = convected ? nca : 0.0f;
    cutop_out[column] = cutop;
    cubot_out[column] = cubot;
    shall_out[column] = (float)ishall;
    cape_before[column] = convected ? abe : 0.0f;
    cape_after[column] = convected ? abeg : 0.0f;
    timec_out[column] = convected ? timec_kf : 0.0f;
#undef KF_CLD
}

#if KF_TRIGGER == 2
/* ==========================================================================
 * kfeta_trigger = 2: the moisture-advection stencil pre-pass
 * ==========================================================================
 * WRF module_cu_kfeta.F:253-397, transcribed in operation order.  One
 * thread per mass column of the WHOLE domain (WOOF runs KF on one patch, so
 * its/ite = ids/ide-1 and the halo reads WRF makes past a tile edge are
 * ordinary in-domain reads here).  Inputs are the temperature T and the
 * decoupled, pure-advective qv tendency RQVFTEN WRF's driver passes this arm
 * (module_cumulus_driver.F:1041; module_big_step_utilities_em.F:5301).
 *
 * Horizontal (tpart_h).  aveh_t/aveh_q are the 9-point means
 * (T(i-1,j-1)+T(i-1,j)+...+T(i+1,j+1))/9. summed in that order, defined on
 * i = ids+1..ide-2, j = jds+1..jde-2.  The domain-edge rows and columns COPY
 * their inward neighbour and the four corners their diagonal one
 * (:291-336); that copy rule is exactly "evaluate at the index clamped into
 * [1, n-2]", which is how kf_aveh9 reads an edge.  tpart_h is formed on the
 * interior only (istart = ids+1 .. iend = ide-2, :340-343) and is 0 on the
 * domain's outer ring:
 *   coef_h = (aveh_q - min9)/(max9 - min9), 0 when max9 == min9, then
 *            amin1(coef_h, 1.0) and amax1(coef_h, 0.0)
 *   tpart_h = coef_h*(T - aveh_t)
 *
 * Vertical (tpart_v), every column (:364-396):
 *   Z0(1) = .5*DZ8W(1), Z0(K) = Z0(K-1) + .5*(DZ8W(K) + DZ8W(K-1))
 *   ttop  = T(k) + ((T(k) - T(k+1))/(Z0(k) - Z0(k+1)))*(Z0(k) - Z0(k-1))
 *   avev_t(k) = (T(k-1) + T(k) + ttop)/3.,  avev_q(k) = (Q(k-1)+Q(k)+Q(k+1))/3.
 *   for k = kts+1..kte-1, copied to kts and kte from their neighbour;
 *   coef_v = (avev_q - min3)/(max3 - min3), 0 when equal (no clamp in WRF);
 *   tpart_v(k) = coef_v*(T(k) - avev_t(k)), copied to kts and kte.
 *
 * KF_eta_PARA then reads DTLCL = amax1(TPART_H0(KLCL) + TPART_V0(KLCL), 0.)
 * (:984-987).  With tpart_v == nullptr the kernel stores that sum,
 * __fadd_rn(tpart_h, tpart_v), at every level: one array for kf_column.
 * With both pointers it stores the two parts, which is how the oracle
 * grades them field by field.
 *
 * Every operation is an explicit round-to-nearest intrinsic, so the default
 * arm cannot contract any of it either: the pre-pass is the same bits in
 * both arms.  No local array: the vertical pass streams Z0 through three
 * registers, so the kernel reserves no local-memory backing store.
 */
__device__ __forceinline__ int kf_edge_clamp(int index, int n) {
    return min(max(index, 1), n - 2);
}

__device__ __forceinline__ float kf_aveh9(
        const float* __restrict__ field, int level, int i, int j,
        int ny, int nx) {
    i = kf_edge_clamp(i, nx);
    j = kf_edge_clamp(j, ny);
    const float* f = field + (size_t)level * (size_t)ny * (size_t)nx;
    // WRF's summation order: i-1 (j-1, j, j+1), i (...), i+1 (...).
    float s = f[(j - 1) * nx + (i - 1)];
    s = __fadd_rn(s, f[j * nx + (i - 1)]);
    s = __fadd_rn(s, f[(j + 1) * nx + (i - 1)]);
    s = __fadd_rn(s, f[(j - 1) * nx + i]);
    s = __fadd_rn(s, f[j * nx + i]);
    s = __fadd_rn(s, f[(j + 1) * nx + i]);
    s = __fadd_rn(s, f[(j - 1) * nx + (i + 1)]);
    s = __fadd_rn(s, f[j * nx + (i + 1)]);
    s = __fadd_rn(s, f[(j + 1) * nx + (i + 1)]);
    return __fdiv_rn(s, 9.0f);
}

__device__ __forceinline__ float kf_avev3(
        const float* __restrict__ field, size_t column, size_t ncol, int k,
        int nz) {
    // avev_q(kts) = avev_q(kts+1), avev_q(kte) = avev_q(kte-1).
    k = kf_edge_clamp(k, nz);
    float s = __fadd_rn(field[(size_t)(k - 1) * ncol + column],
                        field[(size_t)k * ncol + column]);
    s = __fadd_rn(s, field[(size_t)(k + 1) * ncol + column]);
    return __fdiv_rn(s, 3.0f);
}

extern "C" __global__ __launch_bounds__(256)
void kf_trigger2_tpart(
        const float* __restrict__ temperature,
        const float* __restrict__ rqvften,
        const float* __restrict__ dz,
        float* __restrict__ tpart_h,
        float* __restrict__ tpart_v,
        int nz, int ny, int nx) {
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int ncol = ny * nx;
    if (column >= ncol || nz < 3) return;
    int i = column % nx, j = column / nx;
    bool interior = nx >= 3 && ny >= 3
        && i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2;
    size_t col = (size_t)column, n = (size_t)ncol;

    // Vertical: Z0 of the levels k-1, k, k+1 rides in z_m, z_c, z_p.
    float z_m = __fmul_rn(0.5f, dz[col]);
    float z_c = __fadd_rn(z_m, __fmul_rn(0.5f, __fadd_rn(dz[n + col], dz[col])));
    float tv_bottom = 0.0f, tv_top = 0.0f;
    for (int k = 0; k < nz; ++k) {
        size_t index = (size_t)k * n + col;
        float h = 0.0f;
        if (interior) {
            float q_c = kf_aveh9(rqvften, k, i, j, ny, nx);
            float q_max = q_c, q_min = q_c;
            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj) {
                    float q = kf_aveh9(rqvften, k, i + ii, j + jj, ny, nx);
                    if (q > q_max) q_max = q;
                    if (q < q_min) q_min = q;
                }
            float coef = q_max > q_min
                ? __fdiv_rn(__fsub_rn(q_c, q_min), __fsub_rn(q_max, q_min))
                : 0.0f;
            coef = fminf(coef, 1.0f);
            coef = fmaxf(coef, 0.0f);
            h = __fmul_rn(coef, __fsub_rn(
                temperature[index], kf_aveh9(temperature, k, i, j, ny, nx)));
        }
        float v = 0.0f;
        if (k >= 1 && k <= nz - 2) {
            float z_p = __fadd_rn(z_c, __fmul_rn(0.5f, __fadd_rn(
                dz[index + n], dz[index])));
            float t_c = temperature[index];
            float t_p = temperature[index + n];
            float ttop = __fadd_rn(t_c, __fmul_rn(
                __fdiv_rn(__fsub_rn(t_c, t_p), __fsub_rn(z_c, z_p)),
                __fsub_rn(z_c, z_m)));
            float avev_t = __fdiv_rn(__fadd_rn(__fadd_rn(
                temperature[index - n], t_c), ttop), 3.0f);
            float q_c = kf_avev3(rqvften, col, n, k, nz);
            float q_max = q_c, q_min = q_c;
            for (int kk = -1; kk <= 1; ++kk) {
                float q = kf_avev3(rqvften, col, n, k + kk, nz);
                if (q > q_max) q_max = q;
                if (q < q_min) q_min = q;
            }
            float coef = q_max > q_min
                ? __fdiv_rn(__fsub_rn(q_c, q_min), __fsub_rn(q_max, q_min))
                : 0.0f;
            v = __fmul_rn(coef, __fsub_rn(t_c, avev_t));
            if (k == 1) tv_bottom = v;
            if (k == nz - 2) tv_top = v;
            z_m = z_c;
            z_c = z_p;
        }
        if (tpart_v == nullptr) {
            // tpart_v(kts) and tpart_v(kte) are copies, known only once
            // k = 1 and k = nz-2 are done: the two ends are stored below.
            if (k >= 1 && k <= nz - 2) tpart_h[index] = __fadd_rn(h, v);
            else tpart_h[index] = h;
        } else {
            tpart_h[index] = h;
            if (k >= 1 && k <= nz - 2) tpart_v[index] = v;
        }
    }
    if (tpart_v == nullptr) {
        tpart_h[col] = __fadd_rn(tpart_h[col], tv_bottom);
        size_t top = (size_t)(nz - 1) * n + col;
        tpart_h[top] = __fadd_rn(tpart_h[top], tv_top);
    } else {
        tpart_v[col] = tv_bottom;
        tpart_v[(size_t)(nz - 1) * n + col] = tv_top;
    }
}
#endif
