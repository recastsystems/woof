/* Multiscale Kain-Fritsch cumulus (cu_physics = 11), one CUDA thread per
 * mass column.
 *
 * STAGED CORE TRANSCRIPTION IN WRF OPERATION ORDER from WRF v4.6.1 phys/module_cu_mskf.F
 * (sha256 4bca54b2...96fcb), graded word for word against the column oracle
 * tools/mskf_wrf461_oracle (built from that file with WRF's own scalar
 * flags; its object code is instruction-identical to WRF's scalar build).
 * Anchors:
 *   3424-3987  MSKF_CPS: W0AVG running mean, the NCA hold, the per-column
 *              zeroing, the F_QR / F_QI / F_QS feedback routing
 *   3990-4392  MSKF_eta_PARA set-up: the dx scale factor (24.999 km), the
 *              environment, the 15 hPa source-layer list
 *   4424-5369  the source-layer search (usl), the Kain (2004) trigger with
 *              the 25 km dx threshold, the LCL-radius updraft with the
 *              scale-dependent Tokioka entrainment and fallout rate, the
 *              saturated-environment ("double counting") refusal
 *   5370-5560  cloud-top detrainment and the arrays outside the cloud
 *   5565-5657  the dynamic adjustment time scale (cloud height, sub-cloud
 *              velocity scale, CAPE) and the precipitation efficiency
 *   5666-5893  the downdraft
 *   5899-6299  the closure: compensating subsidence, the CAPE-removal
 *              iteration with the dx-dependent capeDX floor
 *   6303-6334  the diagnosed deep/shallow cloud fraction
 *   6364-6688  hydrometeor transport and the phase feedback
 *   6691-6867  the momentum-transport hand-off, flipped to CAM order
 *   7555-8038  MSKF_CMT (Zhang & McFarlane 1995), in DOUBLE precision as
 *              WRF runs it (REAL(r8) throughout)
 *   6925-7322  TPMIX2, DTFRZNEW, CONDLOAD, PROF5, TPMIX2DD, ENVIRTHT --
 *              textually identical to module_cu_kfeta.F's (TPMIX2 adds the
 *              QSu output); the lookup table is the same KF_LUTAB and is
 *              read from gpuwm/data/kf_lutab.
 *
 * ARITHMETIC.  Every expression keeps Fortran's left-to-right association.
 * Strict and aerosol-coupled modes preserve operation order. The fast
 * aerosol-off specialization permits contraction and uses CUDA float libm;
 * its thermodynamic intermediates therefore stay in float32. Under
 * GPUWM_WRF_EXACT=1 the CMT and aerosol path also use binary64
 * g64_exp/g64_log/g64_pow
 * (glibc_flt64.cuh).  Lowering facts measured on the oracle's own
 * toolchain (tools/mskf_wrf461_oracle/lowering-probe.txt):
 *   x**2 = x*x, x**3 = (x*x)*x, x**4 = x2*x2, x**5 = x3*x2, x**6 = x3*x3;
 *   real exponents other than the folded ones call pow;
 *   MAX/MIN return the non-NaN operand and order -0 below +0 (fmaxf/fminf);
 *   NINT rounds half away from zero; EXP(-4.5) and the other constant
 *   intrinsics are folded at compile time and are spelled here as words.
 *
 * DECLARED DIVERGENCES (each one a WRF read of undefined memory, fixed):
 *   D1  The saturated-environment average reads T0/Q0/P0 at K-1 and LTOP+1
 *       (:5247).  K-1 = 0 when the LCL sits in the second model level, and
 *       LTOP+1 = KTE+1 when the updraft reaches the model top; both are
 *       out of bounds.  Here only levels that exist are summed and the
 *       divisor counts them.  Unreached on the oracle fixture.
 *   D2  On the aerosol path (aercu_opt > 0) QSATu is never set outside the
 *       updraft and pgam(k) is read where pgam(k-1) is meant; see the
 *       aerosol section.  The oracle measures both (uninit-pristine.txt).
 */

/* The engine preamble (gpuwm.core.constants.CUDA_DEFINES) defines G, CP,
 * P0, T0 and friends as object-like macros.  WRF's own spellings are used
 * below with WRF's own values (module_model_constants.F as MSKF_CPS receives
 * them), so the preamble names are released here first. */
#undef G
#undef RD
#undef RV
#undef CP
#undef CV
#undef P0
#undef T0
#undef GAMMA
#undef RCP
#undef RCV
#undef RVOVRD
#undef RERADIUS
#undef XLV
#undef SVP1
#undef SVP2
#undef SVP3
#undef SVPT0
#undef RHOWATER
#undef EP2

#define MSKF_KMAX 160
#define MSKF_LANES 32
#define MSKF_FSLOTS 101
#define MSKF_DSLOTS 62
#define MSKF_ASLOTS 512

/* status bits: WRF stops the model at each of these */
#define MSKF_FATAL_QG_SURFACE   1   /* 'QG, QG(NK).LT.0'              :6053 */
#define MSKF_FATAL_MASS_BUDGET  2   /* 'KAIN-FRITSCH, istop=1, diags' :6545 */
#define MSKF_FATAL_PHASE        4   /* 'THIS MICROPHYSICS CHOICE ...' :6684 */
#define MSKF_FATAL_CLOUD_BASE   8   /* 'MSKF Cloud Bottom IS ON THE GROUND' */
#define MSKF_FATAL_CLOUD_TOP   16   /* 'MSKF CLOUD TOP WENT OVER MODEL TOP' */
#define MSKF_FATAL_ICE_NUCL    32   /* 'Incorrect Ice Nucleation Number'    */
#define MSKF_FATAL_LEVELS      64   /* nz outside [8, MSKF_KMAX]            */

/* A column array in the lane-interleaved workspace, indexed the Fortran way:
 * a(k) for k = 0 .. nz+1 (index 0 serves ZF_WRF(0:KTE)). */
struct MsF {
    float* p;
    __device__ __forceinline__ float& operator()(int k) const {
        return p[(size_t)k * (size_t)MSKF_LANES];
    }
};
struct MsD {
    double* p;
    __device__ __forceinline__ double& operator()(int k) const {
        return p[(size_t)k * (size_t)MSKF_LANES];
    }
};
struct MsI {
    int* p;
    __device__ __forceinline__ int& operator()(int k) const {
        return p[(size_t)k * (size_t)MSKF_LANES];
    }
};

__device__ __forceinline__ MsF ms_fslot(float* base, int s, int ext) {
    return MsF{base + (size_t)s * (size_t)ext * MSKF_LANES};
}
__device__ __forceinline__ MsD ms_dslot(double* base, int s, int ext) {
    return MsD{base + (size_t)s * (size_t)ext * MSKF_LANES};
}

/* This thread's lane inside its warp's workspace region. */
__device__ __forceinline__ size_t ms_lane_offset(int slots, int ext) {
    size_t warp = (size_t)blockIdx.x * (blockDim.x / MSKF_LANES)
                  + threadIdx.x / MSKF_LANES;
    return warp * (size_t)slots * (size_t)ext * MSKF_LANES
           + threadIdx.x % MSKF_LANES;
}

/* Fortran intrinsics as gfortran lowers them (lowering-probe.txt). */
__device__ __forceinline__ float ms_max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float ms_min(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ double ms_dmax(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ double ms_dmin(double a, double b) { return fmin(a, b); }
__device__ __forceinline__ int ms_nint(float x) { return (int)roundf(x); }
__device__ __forceinline__ float ms_aint(float x) { return truncf(x); }

// Software float libm contains binary64 polynomial arithmetic. That is
// replay work, not MSKF's native REAL thermodynamics. The aerosol-off fast
// specialization uses CUDA float functions. Coupled aerosol thresholds keep
// the previously qualified arithmetic; strict replay keeps every operation.
#if defined(GPUWM_MSKF_FAST) && defined(GPUWM_MSKF_NO_AEROSOL)
#define MS_EXP(x) expf(x)
#define MS_LOG(x) logf(x)
#define MS_POW(x, y) powf((x), (y))
#else
#define MS_EXP(x) gfk_exp(x)
#define MS_LOG(x) gfk_log(x)
#define MS_POW(x, y) gfk_pow((x), (y))
#endif
#ifdef GPUWM_MSKF_FAST
#define MS_DEXP(x) exp(x)
#define MS_DLOG(x) log(x)
#define MS_DPOW(x, y) pow((x), (y))
#else
#define MS_DEXP(x) g64_exp::exp(x)
#define MS_DLOG(x) g64_log::log(x)
#define MS_DPOW(x, y) g64_pow::pow((x), (y))
#endif

/* EXP(-4.5) folded by gfortran (lowering-probe.txt: 3C360282). */
#define MS_E45 __uint_as_float(0x3C360282u)
/* 2./3. folded: 3F2AAAAB */
#define MS_TWO_THIRDS __uint_as_float(0x3F2AAAABu)

/* The KF lookup table: the0k/alu 1-based in Fortran, 0-based here. */
struct MsLut {
    const float* ttab;    /* (250, 220) row-major [it][ip] */
    const float* qstab;
    const float* the0k;   /* 220 */
    const float* alu;     /* 200 */
    float plutop, rdpr, rdthk;
};

__device__ __forceinline__ float ms_the0k(const MsLut& L, int i) {
    return L.the0k[i - 1];
}
__device__ __forceinline__ float ms_alu(const MsLut& L, int i) {
    return L.alu[i - 1];
}
__device__ __forceinline__ float ms_ttab(const MsLut& L, int it, int ip) {
    return L.ttab[(size_t)(it - 1) * 220 + (ip - 1)];
}
__device__ __forceinline__ float ms_qstab(const MsLut& L, int it, int ip) {
    return L.qstab[(size_t)(it - 1) * 220 + (ip - 1)];
}

/* ---- TPMIX2 (6925-7051), with the QSu output -------------------------- */
__device__ void ms_tpmix2(const MsLut& L, float p, float thes, float* tu,
                          float* qu, float* qliq, float* qice,
                          float* qnewlq, float* qnewic, float XLV1,
                          float XLV0, float* qsu) {
    float tp = (p - L.plutop) * L.rdpr;
    float qq = tp - ms_aint(tp);
    int iptb = (int)tp + 1;
    float bth = (ms_the0k(L, iptb + 1) - ms_the0k(L, iptb)) * qq
                + ms_the0k(L, iptb);
    float tth = (thes - bth) * L.rdthk;
    float pp = tth - ms_aint(tth);
    int ithtb = (int)tth + 1;
    float t00 = ms_ttab(L, ithtb, iptb);
    float t10 = ms_ttab(L, ithtb + 1, iptb);
    float t01 = ms_ttab(L, ithtb, iptb + 1);
    float t11 = ms_ttab(L, ithtb + 1, iptb + 1);
    float q00 = ms_qstab(L, ithtb, iptb);
    float q10 = ms_qstab(L, ithtb + 1, iptb);
    float q01 = ms_qstab(L, ithtb, iptb + 1);
    float q11 = ms_qstab(L, ithtb + 1, iptb + 1);
    float temp = (t00 + (t10 - t00) * pp + (t01 - t00) * qq
                  + (t00 - t10 - t01 + t11) * pp * qq);
    float qs = (q00 + (q10 - q00) * pp + (q01 - q00) * qq
                + (q00 - q10 - q01 + q11) * pp * qq);
    *qsu = qs;
    float dq = qs - *qu;
    float qnew;
    if (dq <= 0.0f) {
        qnew = *qu - qs;
        *qu = qs;
    } else {
        qnew = 0.0f;
        float qtot = *qliq + *qice;
        if (qtot >= dq) {
            *qliq = *qliq - dq * *qliq / (qtot + 1.e-10f);
            *qice = *qice - dq * *qice / (qtot + 1.e-10f);
            *qu = qs;
        } else {
            float rll = XLV0 - XLV1 * temp;
            float cpp = 1004.5f * (1.0f + 0.89f * *qu);
            if (qtot < 1.E-10f) {
                temp = temp + rll * (dq / (1.0f + dq)) / cpp;
            } else {
                temp = temp + rll * ((dq - qtot) / (1.0f + dq - qtot)) / cpp;
                *qu = *qu + qtot;
                *qliq = 0.0f;
                *qice = 0.0f;
            }
        }
    }
    *tu = temp;
    *qnewlq = qnew;
    *qnewic = 0.0f;
}

/* ---- DTFRZNEW (7053-7097) --------------------------------------------- */
__device__ void ms_dtfrznew(float* tu, float p, float* thteu, float* qu,
                            float qfrz, float* qice, float ALIQ, float BLIQ,
                            float CLIQ, float DLIQ) {
    float rlc = 2.5E6f - 2369.276f * (*tu - 273.16f);
    float rls = 2833922.0f - 259.532f * (*tu - 273.16f);
    float rlf = rls - rlc;
    float cpp = 1004.5f * (1.0f + 0.89f * *qu);
    float a = (CLIQ - BLIQ * DLIQ) / ((*tu - DLIQ) * (*tu - DLIQ));
    float dtfrz = rlf * qfrz / (cpp + rls * *qu * a);
    *tu = *tu + dtfrz;
    float es = ALIQ * MS_EXP((BLIQ * *tu - CLIQ) / (*tu - DLIQ));
    float qs = es * 0.622f / (p - es);
    float dqevap = qs - *qu;
    *qice = *qice - dqevap;
    *qu = *qu + dqevap;
    float pii = MS_POW(1.E5f / p, 0.2854f * (1.0f - 0.28f * *qu));
    *thteu = *tu * pii * MS_EXP((3374.6525f / *tu - 2.5403f) * *qu
                                * (1.0f + 0.81f * *qu));
}

/* ---- CONDLOAD (7100-7165) --------------------------------------------- */
__device__ void ms_condload(float* qliq, float* qice, float* wtw, float dz,
                            float boterm, float enterm, float rate,
                            float* qnewlq, float* qnewic, float* qlqout,
                            float* qicout, float g) {
    float qtot = *qliq + *qice;
    float qnew = *qnewlq + *qnewic;
    float qest = 0.5f * (qtot + qnew);
    float g1 = *wtw + boterm - enterm - 2.0f * g * dz * qest / 1.5f;
    if (g1 < 0.0f) g1 = 0.0f;
    float wavg = 0.5f * (sqrtf(*wtw) + sqrtf(g1));
    float conv = rate * dz / wavg;
    float ratio3 = *qnewlq / (qnew + 1.E-8f);
    qtot = qtot + 0.6f * qnew;
    float oldq = qtot;
    float ratio4 = (0.6f * *qnewlq + *qliq) / (qtot + 1.E-8f);
    qtot = qtot * MS_EXP(-conv);
    float dq = oldq - qtot;
    *qlqout = ratio4 * dq;
    *qicout = (1.0f - ratio4) * dq;
    float pptdrg = 0.5f * (oldq + qtot - 0.2f * qnew);
    *wtw = *wtw + boterm - enterm - 2.0f * g * dz * pptdrg / 1.5f;
    if (fabsf(*wtw) < 1.E-4f) *wtw = 1.E-4f;
    *qliq = ratio4 * qtot + ratio3 * 0.4f * qnew;
    *qice = (1.0f - ratio4) * qtot + (1.0f - ratio3) * 0.4f * qnew;
    *qnewlq = 0.0f;
    *qnewic = 0.0f;
}

/* ---- PROF5 (7168-7213) ------------------------------------------------ */
__device__ void ms_prof5(float eq, float* ee, float* ud) {
    const float SQRT2P = 2.506628f, A1 = 0.4361836f, A2 = -0.1201676f,
                A3 = 0.9372980f, P = 0.33267f, SIGMA = 0.166666667f,
                FE = 0.202765151f;
    float y = 6.0f * eq - 3.0f;
    float ey = MS_EXP(y * y / (-2.0f));
    float e45 = MS_E45;
    float t2 = 1.0f / (1.0f + P * fabsf(y));
    float t1 = 0.500498f;
    float c1 = A1 * t1 + A2 * t1 * t1 + A3 * t1 * t1 * t1;
    float c2 = A1 * t2 + A2 * t2 * t2 + A3 * t2 * t2 * t2;
    if (y >= 0.0f) {
        *ee = SIGMA * (0.5f * (SQRT2P - e45 * c1 - ey * c2)
                       + SIGMA * (e45 - ey)) - e45 * eq * eq / 2.0f;
        *ud = SIGMA * (0.5f * (ey * c2 - e45 * c1) + SIGMA * (e45 - ey))
              - e45 * (0.5f + eq * eq / 2.0f - eq);
    } else {
        *ee = SIGMA * (0.5f * (ey * c2 - e45 * c1) + SIGMA * (e45 - ey))
              - e45 * eq * eq / 2.0f;
        *ud = SIGMA * (0.5f * (SQRT2P - e45 * c1 - ey * c2)
                       + SIGMA * (e45 - ey)) - e45 * (0.5f + eq * eq / 2.0f
                                                       - eq);
    }
    *ee = *ee / FE;
    *ud = *ud / FE;
}

/* ---- TPMIX2DD (7216-7281) --------------------------------------------- */
__device__ void ms_tpmix2dd(const MsLut& L, float p, float thes, float* ts,
                            float* qs) {
    float tp = (p - L.plutop) * L.rdpr;
    float qq = tp - ms_aint(tp);
    int iptb = (int)tp + 1;
    float bth = (ms_the0k(L, iptb + 1) - ms_the0k(L, iptb)) * qq
                + ms_the0k(L, iptb);
    float tth = (thes - bth) * L.rdthk;
    float pp = tth - ms_aint(tth);
    int ithtb = (int)tth + 1;
    float t00 = ms_ttab(L, ithtb, iptb);
    float t10 = ms_ttab(L, ithtb + 1, iptb);
    float t01 = ms_ttab(L, ithtb, iptb + 1);
    float t11 = ms_ttab(L, ithtb + 1, iptb + 1);
    float q00 = ms_qstab(L, ithtb, iptb);
    float q10 = ms_qstab(L, ithtb + 1, iptb);
    float q01 = ms_qstab(L, ithtb, iptb + 1);
    float q11 = ms_qstab(L, ithtb + 1, iptb + 1);
    *ts = (t00 + (t10 - t00) * pp + (t01 - t00) * qq
           + (t00 - t10 - t01 + t11) * pp * qq);
    *qs = (q00 + (q10 - q00) * pp + (q01 - q00) * qq
           + (q00 - q10 - q01 + q11) * pp * qq);
}

/* ---- ENVIRTHT (7284-7322) --------------------------------------------- */
__device__ float ms_envirtht(const MsLut& L, float p1, float t1, float q1,
                             float ALIQ, float BLIQ, float CLIQ, float DLIQ) {
    const float T00 = 273.16f, P00 = 1.E5f, C1 = 3374.6525f, C2 = 2.5403f;
    float ee = q1 * p1 / (0.622f + q1);
    float astrt = 1.e-3f, ainc = 0.075f;
    float a1 = ee / ALIQ;
    float tp = (a1 - astrt) / ainc;
    int indlu = (int)tp + 1;
    float value = (float)(indlu - 1) * ainc + astrt;
    float aintrp = (a1 - value) / ainc;
    float tlog = aintrp * ms_alu(L, indlu + 1)
                 + (1.0f - aintrp) * ms_alu(L, indlu);
    float tdpt = (CLIQ - DLIQ * tlog) / (BLIQ - tlog);
    float tsat = tdpt - (0.212f + 1.571E-3f * (tdpt - T00)
                         - 4.36E-4f * (t1 - T00)) * (t1 - tdpt);
    float tht = t1 * MS_POW(P00 / p1, 0.2854f * (1.0f - 0.28f * q1));
    return tht * MS_EXP((C1 / tsat - C2) * q1 * (1.0f + 0.81f * q1));
}

/* ======================================================================
 * MSKF_CMT (7555-8038): convective momentum transport, REAL(r8), on the
 * flipped (CAM) index J = 1 at the model top, ILEV at the surface.
 * ====================================================================== */
struct MsCmt {
    MsD DUDT, DVDT, DP, ED, EU, MC, MD, MU, P, QD, QU, QHAT, SD, SU, SHAT,
        T, U, V, Z, ZF, AC, AD, AU, ACFL, ADFL, AUFL, B1, B1FL, BD, BU, D0,
        D0HAT, DELPX, DELPY, DZ, DZF, EC, E, EHAT, RHO, RHOHAT, UHAT, VHAT,
        UC, VC, W0, W1, W0FL, W1FL, FX, FY, AA, BB, CC, DXN, DYN, ZFW,
        LRHO, WW, B1T, BASEBB, BASECC;
};

__device__ void ms_cmt(const MsCmt& C, double DSUBCLD, int JB, int JD,
                       int JT, int MSG, double DT, double GRAV, double CPRES,
                       double RGAS, int ILEV) {
    const double ALFA1[10] = {4.33675e-05, 7.88429e-05, 1.13065e-04,
        1.45836e-04, 1.76823e-04, 2.05660e-04, 2.32101e-04, 2.55829e-04,
        2.76672e-04, 2.94347e-04};
    const double ALFA2[10] = {-8.64731e-05, -1.56092e-04, -2.21315e-04,
        -2.80995e-04, -3.33799e-04, -3.78455e-04, -4.14020e-04, -4.39600e-04,
        -4.54652e-04, -4.58761e-04};
    const double BSJ0[10] = {-4.02759e-01, 3.00116e-01, -2.49705e-01,
        2.18359e-01, -1.96465e-01, 1.80062e-01, -1.67183e-01, 1.56722e-01,
        -1.48011e-01, 1.40605e-01};
    const double BSJ1[10] = {1.14193e-01, 2.05841e-01, 2.91209e-01,
        3.68619e-01, 4.36201e-01, 4.92233e-01, 5.35486e-01, 5.64886e-01,
        5.79879e-01, 5.80195e-01};
    const double BSJ0CHI = -0.402759, FACTOR = 0.179503;
    const double TAU[10] = {3.83170e+00, 7.01560e+00, 1.01735e+01,
        1.33237e+01, 1.64705e+01, 1.96120e+01, 2.27560e+01, 2.58980e+01,
        2.90480e+01, 3.21926e+01};
    const double RC = 3000.0, RMAX = 50000.0;
    const double CHI = 3.8317 / RC;
    const double SHAPE = 0.8;
    const double ALPHA = 0.5;

    for (int J = MSG + 1; J <= ILEV; ++J) {
        C.BU(J) = 0.0;
        C.BD(J) = 0.0;
        C.DELPX(J) = 0.0;
        C.DELPY(J) = 0.0;
        C.EC(J) = C.EU(J) + C.ED(J);
        if (C.T(J) > 0.0) C.RHO(J) = 100.0 * C.P(J) / (RGAS * C.T(J));
    }
    for (int J = MSG + 2; J <= ILEV; ++J) {
        C.BU(J) = (C.SU(J) - C.SHAT(J) + 0.608 * (C.QU(J)
                   * (C.SU(J) - GRAV / CPRES * C.ZF(J)) - C.QHAT(J)
                   * (C.SHAT(J) - GRAV / CPRES * C.ZF(J))))
                  / ((C.SHAT(J) - GRAV / CPRES * C.ZF(J))
                     * (1.0 + .608 * C.QHAT(J))) * GRAV;
        C.BD(J) = (C.SD(J) - C.SHAT(J) + 0.608 * (C.QD(J)
                   * (C.SD(J) - GRAV / CPRES * C.ZF(J)) - C.QHAT(J)
                   * (C.SHAT(J) - GRAV / CPRES * C.ZF(J))))
                  / ((C.SHAT(J) - GRAV / CPRES * C.ZF(J))
                     * (1.0 + .608 * C.QHAT(J))) * GRAV;
    }
    double UMN = 0.0, VMN = 0.0, COSA = 0.0, SINA = 0.0;
    for (int J = 1; J <= ILEV; ++J) {
        if (J >= JT && J < JB) {
            UMN = UMN + C.U(J) * C.P(J) / C.T(J) * (C.ZF(J) - C.ZF(J + 1));
            VMN = VMN + C.V(J) * C.P(J) / C.T(J) * (C.ZF(J) - C.ZF(J + 1));
        }
    }
    double WINDMAG = sqrt(UMN * UMN + VMN * VMN);
    if (WINDMAG != 0.0) {
        COSA = UMN / WINDMAG;
        SINA = VMN / WINDMAG;
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J > MSG + 1) {
            C.RHOHAT(J) = ALPHA * C.RHO(J - 1) + (1.0 - ALPHA) * C.RHO(J);
            C.DZF(J) = C.Z(J - 1) - C.Z(J);
            C.UHAT(J) = ALPHA * C.U(J - 1) + (1.0 - ALPHA) * C.U(J);
            C.VHAT(J) = ALPHA * C.V(J - 1) + (1.0 - ALPHA) * C.V(J);
        } else {
            C.RHOHAT(J) = C.RHO(J);
            C.DZF(J) = C.ZF(J) - C.Z(J);
            C.UHAT(J) = C.U(J);
            C.VHAT(J) = C.V(J);
        }
        C.UC(J) = C.UHAT(J);
        C.VC(J) = C.VHAT(J);
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J < ILEV) C.DZ(J) = C.ZF(J) - C.ZF(J + 1);
        else C.DZ(ILEV) = C.DP(ILEV) * 100.0 / (C.RHO(ILEV) * GRAV);
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J >= JT) C.AU(J) = C.MU(JB) * 100.0 / GRAV / C.RHOHAT(JB);
        else C.AU(J) = 0.0;
        if (J >= JD) C.AD(J) = -C.MD(JD) * 100.0 / GRAV / C.RHOHAT(JD);
        else C.AD(J) = 0.0;
        C.AC(J) = C.AU(J) + C.AD(J);
        if (C.AC(J) > 0.0)
            C.W0(J) = C.MC(J) * 100.0 / GRAV / (C.RHOHAT(J) * C.AC(J));
        else C.W0(J) = 0.0;
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J < ILEV) {
            C.ACFL(J) = ALPHA * C.AC(J) + (1.0 - ALPHA) * C.AC(J + 1);
            C.AUFL(J) = ALPHA * C.AU(J) + (1.0 - ALPHA) * C.AU(J + 1);
            C.ADFL(J) = ALPHA * C.AD(J) + (1.0 - ALPHA) * C.AD(J + 1);
            C.W0FL(J) = ALPHA * C.W0(J) + (1.0 - ALPHA) * C.W0(J + 1);
        } else {
            C.ACFL(J) = ALPHA * C.AC(J);
            C.AUFL(J) = ALPHA * C.AU(J);
            C.ADFL(J) = ALPHA * C.AD(J);
            C.W0FL(J) = ALPHA * C.W0(J);
        }
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J < JB && J >= JT && C.ACFL(J) > 0.0) {
            C.D0(J) = (C.MC(J + 1) - C.MC(J)) * 100.0 / GRAV
                      / (C.RHO(J) * C.DZ(J) * C.ACFL(J));
        } else if (J == JB && C.ACFL(J) > 0.0) {
            C.D0(J) = -C.MC(J) * 100.0 / GRAV
                      / (C.RHO(J) * C.DZ(J) * C.ACFL(J));
        } else {
            C.D0(J) = 0.0;
        }
        if (J < JB && J >= JT && C.ACFL(J) > 0.0 && C.ADFL(J) > 0.0
                && C.AUFL(J) > 0.0) {
            C.E(J) = 1.0 / (CHI * CHI) * C.ADFL(J) / (C.ACFL(J) * C.RHO(J))
                     * (1.0 / (C.AUFL(J) * C.DZ(J)) * (C.MU(J) - C.MU(J + 1))
                        * 100.0 / GRAV
                        - 1.0 / (C.ADFL(J) * C.DZ(J))
                          * (C.MD(J) - C.MD(J + 1)) * 100.0 / GRAV)
                     * SHAPE / FACTOR;
        } else {
            C.E(J) = 0.0;
        }
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J > MSG + 1) {
            C.D0HAT(J) = ALPHA * C.D0(J - 1) + (1.0 - ALPHA) * C.D0(J);
            C.EHAT(J) = ALPHA * C.E(J - 1) + (1.0 - ALPHA) * C.E(J);
        } else {
            C.D0HAT(J) = (1.0 - ALPHA) * C.D0(J);
            C.EHAT(J) = (1.0 - ALPHA) * C.E(J);
        }
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J >= JD && J <= JB && C.AU(J) > 0.0 && C.AD(J) > 0.0
                && C.AC(J) > 0.0) {
            C.W1(J) = SHAPE / FACTOR * (C.MU(J) / C.AU(J) - C.MD(J) / C.AD(J))
                      * 100.0 / GRAV * C.AD(J) / (C.AC(J) * C.RHOHAT(J));
            C.B1(J) = SHAPE / FACTOR * (C.BU(J) - C.BD(J)) * C.AD(J)
                      / C.AC(J);
        } else {
            C.W1(J) = 0.0;
            C.B1(J) = 0.0;
        }
    }
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J < ILEV) {
            C.W1FL(J) = ALPHA * C.W1(J) + (1.0 - ALPHA) * C.W1(J + 1);
            C.B1FL(J) = ALPHA * C.B1(J) + (1.0 - ALPHA) * C.B1(J + 1);
        } else {
            C.W1FL(J) = ALPHA * C.W1(J);
            C.B1FL(J) = ALPHA * C.B1(J);
        }
    }
    // These operands do not change between harmonics. Store their original
    // binary64 expressions once, without reassociation or precision changes.
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J < JB && J >= JT && C.MC(J) > 0.0 && C.MC(J + 1) > 0.0) {
            C.LRHO(J) = (MS_DLOG(C.RHO(J - 1)) / (C.DZ(J) * C.DZF(J))
                         - (1.0 / C.DZF(J) + 1.0 / C.DZF(J + 1))
                           / C.DZ(J) * MS_DLOG(C.RHO(J))
                         + MS_DLOG(C.RHO(J + 1))
                           / (C.DZ(J) * C.DZF(J + 1)));
            C.WW(J) = ((C.W0(J) - C.W0(J + 1)) * (C.W1(J) - C.W1(J + 1))
                       / (C.DZ(J) * C.DZ(J))
                       - C.W0FL(J) * C.W1FL(J) * C.LRHO(J));
            C.B1T(J) = (C.RHOHAT(J) * C.B1(J)
                        - C.RHOHAT(J + 1) * C.B1(J + 1))
                       / (C.DZ(J) * C.RHO(J));
        }
    }
    for (int J = MSG + 2; J <= ILEV - 1; ++J) {
        C.AA(J) = 1.0 / (C.DZ(J) * C.DZF(J));
        C.BASEBB(J) = 1.0 / C.DZ(J) * (1.0 / C.DZF(J) + 1.0 / C.DZF(J + 1));
        C.BASECC(J) = 1.0 / (C.DZ(J) * C.DZF(J + 1));
    }
    /* Loops 500-700 for each harmonic N in turn: FX/FY, the tridiagonal
     * system, and the BSJ1-weighted accumulation into DELPX/DELPY, which
     * WRF also accumulates in N order for every J. */
    for (int N = 0; N < 10; ++N) {
        for (int J = MSG + 1; J <= ILEV; ++J) {
            C.FX(J) = 0.0;
            C.FY(J) = 0.0;
        }
        for (int J = MSG + 1; J <= ILEV; ++J) {
            if (J < JB && J >= JT && C.MC(J) > 0.0 && C.MC(J + 1) > 0.0) {
                double ww = C.WW(J);
                double b1t = C.B1T(J);
                C.FX(J) = 2.0 * C.RHO(J) / (BSJ0[N] * BSJ0[N])
                    * (C.D0(J) * C.E(J) * (CHI * CHI) * COSA * ALFA1[N]
                       + 2.0 * C.W0FL(J) * RC / (RMAX * RMAX) * BSJ1[N]
                         * ((C.EHAT(J) - C.EHAT(J + 1)) / C.DZ(J) * CHI
                            * COSA * BSJ0CHI
                            + (C.UHAT(J) - C.UHAT(J + 1)) / C.DZ(J))
                       - (C.D0HAT(J) - C.D0HAT(J + 1)) / C.DZ(J) * C.W1FL(J)
                         * COSA * (ALFA2[N] - ALFA1[N])
                       - 2.0 * ww * COSA * ALFA1[N]
                       + b1t * COSA * ALFA1[N]);
                C.FY(J) = 2.0 * C.RHO(J) / (BSJ0[N] * BSJ0[N])
                    * (C.D0(J) * C.E(J) * (CHI * CHI) * SINA * ALFA1[N]
                       + 2.0 * C.W0FL(J) * RC / (RMAX * RMAX) * BSJ1[N]
                         * ((C.EHAT(J) - C.EHAT(J + 1)) / C.DZ(J) * CHI
                            * SINA * BSJ0CHI
                            + (C.VHAT(J) - C.VHAT(J + 1)) / C.DZ(J))
                       - (C.D0HAT(J) - C.D0HAT(J + 1)) / C.DZ(J) * C.W1FL(J)
                         * SINA * (ALFA2[N] - ALFA1[N])
                       - 2.0 * ww * SINA * ALFA1[N]
                       + b1t * SINA * ALFA1[N]);
            }
        }
        /* 525 */
        C.AA(MSG + 1) = 0.0;
        C.BB(MSG + 1) = 1.0 / C.DZF(MSG + 2);
        C.CC(MSG + 1) = -1.0 / C.DZF(MSG + 2);
        C.AA(ILEV) = 1.0 / C.DZF(ILEV);
        C.BB(ILEV) = -1.0 / C.DZF(ILEV);
        C.CC(ILEV) = 0.0;
        C.DXN(MSG + 1) = C.B1(MSG + 2) * ALFA1[N] * 2.0 * COSA
                         / (BSJ0[N] * BSJ0[N]);
        C.DXN(ILEV) = C.B1(ILEV) * ALFA1[N] * 2.0 * COSA
                      / (BSJ0[N] * BSJ0[N]);
        C.DYN(MSG + 1) = C.B1(MSG + 2) * ALFA1[N] * 2.0 * SINA
                         / (BSJ0[N] * BSJ0[N]);
        C.DYN(ILEV) = C.B1(ILEV) * ALFA1[N] * 2.0 * SINA
                      / (BSJ0[N] * BSJ0[N]);
        /* 550 */
        for (int J = MSG + 2; J <= ILEV - 1; ++J) {
            C.BB(J) = -(C.BASEBB(J)
                        + (TAU[N] / RMAX) * (TAU[N] / RMAX));
            C.CC(J) = C.BASECC(J);
            C.DXN(J) = C.FX(J);
            C.DYN(J) = C.FY(J);
        }
        /* 575 */
        C.CC(MSG + 1) = C.CC(MSG + 1) / C.BB(MSG + 1);
        C.DXN(MSG + 1) = C.DXN(MSG + 1) / C.BB(MSG + 1);
        C.DYN(MSG + 1) = C.DYN(MSG + 1) / C.BB(MSG + 1);
        /* 600 */
        for (int J = MSG + 2; J <= ILEV; ++J) {
            C.CC(J) = C.CC(J) / (C.BB(J) - C.AA(J) * C.CC(J - 1));
            C.DXN(J) = (C.DXN(J) - C.AA(J) * C.DXN(J - 1))
                       / (C.BB(J) - C.AA(J) * C.CC(J - 1));
            C.DYN(J) = (C.DYN(J) - C.AA(J) * C.DYN(J - 1))
                       / (C.BB(J) - C.AA(J) * C.CC(J - 1));
        }
        /* 650 */
        for (int J = ILEV - 1; J >= MSG + 1; --J) {
            C.DXN(J) = C.DXN(J) - C.CC(J) * C.DXN(J + 1);
            C.DYN(J) = C.DYN(J) - C.CC(J) * C.DYN(J + 1);
        }
        /* 700 */
        for (int J = MSG + 1; J <= ILEV; ++J) {
            C.DELPX(J) = C.DELPX(J) + C.DXN(J) * BSJ1[N];
            C.DELPY(J) = C.DELPY(J) + C.DYN(J) * BSJ1[N];
        }
    }
    /* 850 */
    for (int J = MSG + 1; J <= ILEV; ++J) {
        C.DELPX(J) = C.DELPX(J) / RC;
        C.DELPY(J) = C.DELPY(J) / RC;
        C.DELPX(J) = C.ACFL(J) * C.DELPX(J) / C.RHO(J);
        C.DELPY(J) = C.ACFL(J) * C.DELPY(J) / C.RHO(J);
    }
    /* 875 */
    for (int J = ILEV - 1; J >= MSG + 1; --J) {
        if (C.MC(J) > 0.0 && C.MC(J + 1) > 0.0 && J > JT && J < JB) {
            C.UC(J) = C.UC(J + 1) + C.RHO(J) * C.DZ(J)
                      / ((C.MC(J) + C.MC(J + 1)) * 0.5 * 100.0 / GRAV)
                      * (C.EC(J) * (C.U(J) - C.UC(J + 1)) - C.DELPX(J));
            C.VC(J) = C.VC(J + 1) + C.RHO(J) * C.DZ(J)
                      / ((C.MC(J) + C.MC(J + 1)) * 0.5 * 100.0 / GRAV)
                      * (C.EC(J) * (C.V(J) - C.VC(J + 1)) - C.DELPY(J));
        }
    }
    /* 950, RNU = 0 */
    const double RNU = 0.0;
    for (int J = MSG + 1; J <= ILEV; ++J) {
        if (J >= JT && J <= JB) {
            C.UHAT(J) = C.UHAT(J) + RNU * ALPHA * DT * C.DUDT(J - 1);
            C.VHAT(J) = C.VHAT(J) + RNU * ALPHA * DT * C.DVDT(J - 1);
            if (J == JT) {
                C.DUDT(J) = 1.0 / (1.0 + RNU * ALPHA * C.MC(J + 1) * DT
                                   / C.DP(J))
                            * C.MC(J + 1) * (C.UC(J + 1) - C.UHAT(J + 1))
                            / C.DP(J);
                C.DVDT(J) = 1.0 / (1.0 + RNU * ALPHA * C.MC(J + 1) * DT
                                   / C.DP(J))
                            * C.MC(J + 1) * (C.VC(J + 1) - C.VHAT(J + 1))
                            / C.DP(J);
            } else if (J < JB) {
                C.DUDT(J) = 1.0 / (1.0 + RNU * ALPHA * C.MC(J + 1) * DT
                                   / C.DP(J))
                            * (C.MC(J + 1) * (C.UC(J + 1) - C.UHAT(J + 1))
                               - C.MC(J) * (C.UC(J) - C.UHAT(J))) / C.DP(J);
                C.DVDT(J) = 1.0 / (1.0 + RNU * ALPHA * C.MC(J + 1) * DT
                                   / C.DP(J))
                            * (C.MC(J + 1) * (C.VC(J + 1) - C.VHAT(J + 1))
                               - C.MC(J) * (C.VC(J) - C.VHAT(J))) / C.DP(J);
            } else {
                C.DUDT(J) = -1.0 / DSUBCLD * C.MC(J) * (C.UC(J) - C.UHAT(J));
                C.DVDT(J) = -1.0 / DSUBCLD * C.MC(J) * (C.VC(J) - C.VHAT(J));
            }
        }
    }
}

/* ======================================================================
 * The aerosol-aware updraft microphysics (module_cu_mp, aercu_opt > 0).
 * ====================================================================== */
/* Aerosol microphysics is emitted in mskf_aerosol.cuh. */








/* ======================================================================
 * The column: MSKF_CPS's per-column work (3839-3983) and MSKF_eta_PARA.
 * ====================================================================== */
struct MsIn {
    const float *u, *v, *t, *qv, *p, *rho, *pi, *dz8w, *w0avg;
    const float *hfx, *ust, *pblh, *zol;
    const float *aerocu;      /* (10, nz, ncol) or null */
};
struct MsOut {
    float *rthcuten, *rqvcuten, *rqccuten, *rqrcuten, *rqicuten, *rqscuten;
    float *rucuten, *rvcuten;
    float *cldfra_dp, *cldfra_sh, *qc_kf, *qi_kf;
    float *qr_kf, *qs_kf, *nc_kf, *ni_kf, *nr_kf, *ns_kf, *ccn_kf;
    float *efcs, *efis, *efss;
    float *udr_kf, *ddr_kf, *uer_kf, *der_kf;
    float *raincv, *pratec, *nca, *cutop, *cubot, *ainc_frac, *timec_kf;
    int *cu_act, *status;
};
struct MsCfg {
    float dt, dx, cudt, aercu_fct;
    int warm_rain, f_qr, f_qi, f_qs, aercu_opt, kf_edrates;
};

/* WRF v4.6.1 share/module_model_constants.F, as MSKF_CPS receives them. */
#define MS_G 9.81f
#define MS_R 287.0f
#define MS_CP 1004.5f
#define MS_XLV0 3.15E6f
#define MS_XLV1 2370.0f
#define MS_SVP1 0.6112f
#define MS_SVP2 17.67f
#define MS_SVP3 29.65f
#define MS_SVPT0 273.15f

extern "C" __global__ __launch_bounds__(128)
void mskf_column(
        const float* __restrict__ u, const float* __restrict__ v,
        const float* __restrict__ t, const float* __restrict__ qv,
        const float* __restrict__ p, const float* __restrict__ rho,
        const float* __restrict__ pi, const float* __restrict__ dz8w,
        const float* __restrict__ w0avg, const float* __restrict__ hfx,
        const float* __restrict__ ust, const float* __restrict__ pblh,
        const float* __restrict__ zol, const float* __restrict__ aerocu,
        float* rthcuten, float* rqvcuten, float* rqccuten, float* rqrcuten,
        float* rqicuten, float* rqscuten, float* rucuten, float* rvcuten,
        float* cldfra_dp, float* cldfra_sh, float* qc_kf, float* qi_kf,
        float* qr_kf, float* qs_kf, float* nc_kf, float* ni_kf,
        float* nr_kf, float* ns_kf, float* ccn_kf, float* efcs,
        float* efis, float* efss, float* udr_kf, float* ddr_kf,
        float* uer_kf, float* der_kf, float* raincv, float* pratec,
        float* nca, float* cutop, float* cubot, float* ainc_frac,
        float* timec_kf, int* cu_act, int* status,
        const float* __restrict__ ttab,
        const float* __restrict__ qstab,
        const float* __restrict__ the0k,
        const float* __restrict__ alu,
        float plutop, float rdpr, float rdthk,
        float* __restrict__ wsf, double* __restrict__ wsd,
        double* __restrict__ wsa,
        float dt, float dx, float cudt, float aercu_fct,
        int warm_rain, int f_qr, int f_qi, int f_qs, int aercu_opt,
        int kf_edrates, int nz, int ncol, int col0, int span
#if defined(GPUWM_MSKF_FAST) && !defined(GPUWM_MSKF_NO_AEROSOL)
        , double* aerosol_constants
#endif
        ) {
#ifdef GPUWM_MSKF_NO_AEROSOL
    // The selector is host metadata. Exclude the inactive aerosol workspace
    // and its register demand from this compilation; equations are unchanged.
    aercu_opt = 0;
#endif
    const MsIn in{u, v, t, qv, p, rho, pi, dz8w, w0avg, hfx, ust, pblh,
                  zol, aerocu};
    const MsOut out{rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten,
                    rqscuten, rucuten, rvcuten, cldfra_dp, cldfra_sh, qc_kf,
                    qi_kf, qr_kf, qs_kf, nc_kf, ni_kf, nr_kf, ns_kf, ccn_kf,
                    efcs, efis, efss, udr_kf, ddr_kf, uer_kf, der_kf, raincv,
                    pratec, nca, cutop, cubot, ainc_frac, timec_kf, cu_act,
                    status};
    const MsCfg cfg{dt, dx, cudt, aercu_fct, warm_rain, f_qr, f_qi, f_qs,
                    aercu_opt, kf_edrates};
    int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_col >= span) return;
    int col = col0 + local_col;
    if (col >= ncol) return;
    const int ext = nz + 2;
    if (nz < 8 || nz > MSKF_KMAX) {
        atomicOr(out.status, MSKF_FATAL_LEVELS);
        return;
    }
    MsLut L{ttab, qstab, the0k, alu, plutop, rdpr, rdthk};
    const int KTS = 1, KTE = nz, KL = nz, KX = nz;
    const float DT = cfg.dt, DX = cfg.dx, CUDT = cfg.cudt;
    const float DXSQ = DX * DX;
    const float G = MS_G, R = MS_R, CP = MS_CP;
    const float XLV0 = MS_XLV0, XLV1 = MS_XLV1;
#define IX(k) ((size_t)((k) - 1) * (size_t)ncol + (size_t)col)

    /* the NCA hold (3843): a held column is not touched at all */
    if (out.nca[col] >= 0.5f * DT) {
        out.cu_act[col] = 0;
        return;
    }
    out.cu_act[col] = 1;
    for (int k = KTS; k <= KTE; ++k) {
        out.cldfra_dp[IX(k)] = 0.0f;
        out.cldfra_sh[IX(k)] = 0.0f;
        out.qc_kf[IX(k)] = 0.0f;
        out.qi_kf[IX(k)] = 0.0f;
        if (cfg.aercu_opt > 0) {
            out.qr_kf[IX(k)] = 0.0f; out.qs_kf[IX(k)] = 0.0f;
            out.nc_kf[IX(k)] = 0.0f; out.ni_kf[IX(k)] = 0.0f;
            out.nr_kf[IX(k)] = 0.0f; out.ns_kf[IX(k)] = 0.0f;
            out.ccn_kf[IX(k)] = 0.0f;
            out.efss[IX(k)] = 10.01f;
            out.efcs[IX(k)] = 2.51f;
            out.efis[IX(k)] = 5.01f;
        }
    }
    if (cfg.aercu_opt > 0) out.ainc_frac[col] = 0.0f;
    if (cfg.kf_edrates == 1) {
        for (int k = KTS; k <= KTE; ++k) {
            out.udr_kf[IX(k)] = 0.0f; out.ddr_kf[IX(k)] = 0.0f;
            out.uer_kf[IX(k)] = 0.0f; out.der_kf[IX(k)] = 0.0f;
        }
        out.timec_kf[col] = 0.0f;
    }
    out.raincv[col] = 0.0f;
    out.cutop[col] = (float)KTS;
    out.cubot[col] = (float)(KTE + 1);
    out.pratec[col] = 0.0f;

    /* workspace */
    float* fb = wsf + ms_lane_offset(MSKF_FSLOTS, ext);
    double* db = wsd + ms_lane_offset(MSKF_DSLOTS, ext);
    int s = 0;
    MsF Q0 = ms_fslot(fb, s++, ext), Z0 = ms_fslot(fb, s++, ext),
        TV0 = ms_fslot(fb, s++, ext), TU = ms_fslot(fb, s++, ext),
        TVU = ms_fslot(fb, s++, ext), QU = ms_fslot(fb, s++, ext),
        TZ = ms_fslot(fb, s++, ext), TVD = ms_fslot(fb, s++, ext),
        QD = ms_fslot(fb, s++, ext), QES = ms_fslot(fb, s++, ext),
        TG = ms_fslot(fb, s++, ext), TVG = ms_fslot(fb, s++, ext),
        QG = ms_fslot(fb, s++, ext), WU = ms_fslot(fb, s++, ext),
        EMS = ms_fslot(fb, s++, ext), EMSD = ms_fslot(fb, s++, ext),
        UMF = ms_fslot(fb, s++, ext), UER = ms_fslot(fb, s++, ext),
        UDR = ms_fslot(fb, s++, ext), DMF = ms_fslot(fb, s++, ext),
        DER = ms_fslot(fb, s++, ext), DDR = ms_fslot(fb, s++, ext),
        UMF2 = ms_fslot(fb, s++, ext), UER2 = ms_fslot(fb, s++, ext),
        UDR2 = ms_fslot(fb, s++, ext), DMF2 = ms_fslot(fb, s++, ext),
        DER2 = ms_fslot(fb, s++, ext), DDR2 = ms_fslot(fb, s++, ext),
        DZA = ms_fslot(fb, s++, ext), THTA0 = ms_fslot(fb, s++, ext),
        THETEE = ms_fslot(fb, s++, ext), THTAU = ms_fslot(fb, s++, ext),
        THETEU = ms_fslot(fb, s++, ext), THTAD = ms_fslot(fb, s++, ext),
        THETED = ms_fslot(fb, s++, ext), QLIQ = ms_fslot(fb, s++, ext),
        QICE = ms_fslot(fb, s++, ext), QLQOUT = ms_fslot(fb, s++, ext),
        QICOUT = ms_fslot(fb, s++, ext), PPTLIQ = ms_fslot(fb, s++, ext),
        PPTICE = ms_fslot(fb, s++, ext), DETLQ = ms_fslot(fb, s++, ext),
        DETIC = ms_fslot(fb, s++, ext), DETLQ2 = ms_fslot(fb, s++, ext),
        DETIC2 = ms_fslot(fb, s++, ext), RATIO2 = ms_fslot(fb, s++, ext),
        DOMGDP = ms_fslot(fb, s++, ext), EXN = ms_fslot(fb, s++, ext),
        TVQU = ms_fslot(fb, s++, ext), DP = ms_fslot(fb, s++, ext),
        RH = ms_fslot(fb, s++, ext), EQFRC = ms_fslot(fb, s++, ext),
        QDT = ms_fslot(fb, s++, ext), FXM = ms_fslot(fb, s++, ext),
        THTAG = ms_fslot(fb, s++, ext), THPA = ms_fslot(fb, s++, ext),
        THFXOUT = ms_fslot(fb, s++, ext), THFXIN = ms_fslot(fb, s++, ext),
        QPA = ms_fslot(fb, s++, ext), QFXOUT = ms_fslot(fb, s++, ext),
        QFXIN = ms_fslot(fb, s++, ext), QLPA = ms_fslot(fb, s++, ext),
        QLFXIN = ms_fslot(fb, s++, ext), QLFXOUT = ms_fslot(fb, s++, ext),
        QIPA = ms_fslot(fb, s++, ext), QIFXIN = ms_fslot(fb, s++, ext),
        QIFXOUT = ms_fslot(fb, s++, ext), QRPA = ms_fslot(fb, s++, ext),
        QRFXIN = ms_fslot(fb, s++, ext), QRFXOUT = ms_fslot(fb, s++, ext),
        QSPA = ms_fslot(fb, s++, ext), QSFXIN = ms_fslot(fb, s++, ext),
        QSFXOUT = ms_fslot(fb, s++, ext), QLG = ms_fslot(fb, s++, ext),
        QIG = ms_fslot(fb, s++, ext), QRG = ms_fslot(fb, s++, ext),
        QSG = ms_fslot(fb, s++, ext), OMG = ms_fslot(fb, s++, ext),
        RAINFB = ms_fslot(fb, s++, ext), SNOWFB = ms_fslot(fb, s++, ext),
        CLDHGT = ms_fslot(fb, s++, ext), QSD = ms_fslot(fb, s++, ext),
        DILFRC = ms_fslot(fb, s++, ext), DDILFRC = ms_fslot(fb, s++, ext),
        TGU = ms_fslot(fb, s++, ext), QGU = ms_fslot(fb, s++, ext),
        THTEEG = ms_fslot(fb, s++, ext), TDN = ms_fslot(fb, s++, ext),
        TUP = ms_fslot(fb, s++, ext), STATE = ms_fslot(fb, s++, ext);
    MsI KCHECK{(int*)ms_fslot(fb, s++, ext).p};
    MsF QRAIN = ms_fslot(fb, s++, ext), QSNOW = ms_fslot(fb, s++, ext),
        NLIQ = ms_fslot(fb, s++, ext), NICE = ms_fslot(fb, s++, ext),
        NRAIN = ms_fslot(fb, s++, ext), NSNOW = ms_fslot(fb, s++, ext),
        CCN = ms_fslot(fb, s++, ext), EFFCH = ms_fslot(fb, s++, ext),
        EFFIH = ms_fslot(fb, s++, ext), EFFSH = ms_fslot(fb, s++, ext);

    const float P00 = 1.E5f, T00 = 273.16f, RLF = 3.339E5f;
    const float TTFRZ = 268.16f, TBFRZ = 248.16f, RATE = 0.03f;
    const float GDRY = -G / CP;
    const float ROCP = R / CP;
    const float ALIQ = MS_SVP1 * 1000.0f;
    const float BLIQ = MS_SVP2;
    const float CLIQ = MS_SVP2 * MS_SVPT0;
    const float DLIQ = MS_SVP3;
    (void)ROCP;
    float Scale_Fac, capeDX;
    if (DX >= 24.999E3f) {
        Scale_Fac = 1.0f;
        capeDX = 0.1f;
    } else {
        Scale_Fac = 1.0f + (MS_LOG(25.E3f / DX));
        capeDX = 0.1f * sqrtf(Scale_Fac);
    }
    float FBFRC = 0.0f;
    int NCHM = 0, ISHALL = 0;
    const float DPMIN = 5.E3f;
    const float P300 = in.p[IX(1)] - 30000.0f;
    int ML = 0, L5 = 0, LLFC = 0;
    bool feedback = false;   /* the column reached the feedback block */
    (void)NCHM;
#define T0(k) in.t[IX(k)]
#define P0(k) in.p[IX(k)]
#define U0(k) in.u[IX(k)]
#define V0(k) in.v[IX(k)]
#define DZQ(k) in.dz8w[IX(k)]
#define W0AVG1D(k) in.w0avg[IX(k)]
    for (int K = 1; K <= KX; ++K) {
        float ES = ALIQ * MS_EXP((BLIQ * T0(K) - CLIQ) / (T0(K) - DLIQ));
        QES(K) = 0.622f * ES / (P0(K) - ES);
        Q0(K) = ms_min(QES(K), in.qv[IX(K)]);
        Q0(K) = ms_max(0.000001f, Q0(K));
        RH(K) = Q0(K) / QES(K);
        DILFRC(K) = 1.0f;
        TV0(K) = T0(K) * (1.0f + 0.608f * Q0(K));
        DP(K) = in.rho[IX(K)] * G * DZQ(K);
        CLDHGT(K) = 0.0f;
        if (P0(K) >= 0.5f * P0(1)) L5 = K;
        if (P0(K) >= P300) LLFC = K;
    }
    Z0(1) = 0.5f * DZQ(1);
    for (int K = 2; K <= KL; ++K) {
        Z0(K) = Z0(K - 1) + 0.5f * (DZQ(K) + DZQ(K - 1));
        DZA(K - 1) = Z0(K) - Z0(K - 1);
    }
    DZA(KL) = 0.0f;
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
    int LC = 0, KPBL = 0, KLCL = 0, K = 0, LCL = 0, LET = 0, LTOP = 0, NK;
    float DPTHMX = 0, TMIX = 0, QMIX = 0, ZMIX = 0, PMIX = 0, TLCL = 0,
          TVLCL = 0, ZLCL = 0, TENV = 0, QENV = 0, TVEN = 0, DLP = 0,
          DTLCL = 0, WKL = 0, WKLCL = 0, WLCL = 0, PLCL = 0, WTW = 0,
          RHOLCL = 0, RAD = 0, AU0 = 0, VMFLCL = 0, UPOLD = 0, UPNEW = 0,
          ABE = 0, TRPPT = 0, TTEMP = 0, EE1 = 0, UD1 = 0, REI = 0,
          DILBE = 0, CHMIN = 0, QNEWLQ = 0, QNEWIC = 0, QSu = 0;
    int IFLAG = 0;
    (void)IFLAG;
#if defined(GPUWM_MSKF_FAST) && !defined(GPUWM_MSKF_NO_AEROSOL)
    MaPhysics A(cfg.aercu_opt>0 ? wsa+ms_lane_offset(MSKF_ASLOTS,ext) : nullptr,
                aerosol_constants);
#else
    MaPhysics A(cfg.aercu_opt>0 ? wsa+ms_lane_offset(MSKF_ASLOTS,ext) : nullptr);
#endif
MaArena& arena=A.arena;
const int PCOLS=1,NAER_CU=10;
MaArray<double> MUU;
if(cfg.aercu_opt>0) MUU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> SU;
if(cfg.aercu_opt>0) SU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> QUU;
if(cfg.aercu_opt>0) QUU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> DUU;
if(cfg.aercu_opt>0) DUU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> EUU;
if(cfg.aercu_opt>0) EUU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> CMEL;
if(cfg.aercu_opt>0) CMEL=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> CMEI;
if(cfg.aercu_opt>0) CMEI=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZFU;
if(cfg.aercu_opt>0) ZFU=arena.make<double>(1,PCOLS,KTS,(KTE+1));
MaArray<double> ZF_WRF;
if(cfg.aercu_opt>0) ZF_WRF=arena.make<double>(0,KTE);
MaArray<double> PRU;
if(cfg.aercu_opt>0) PRU=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> TEE;
if(cfg.aercu_opt>0) TEE=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> QEE;
if(cfg.aercu_opt>0) QEE=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> QSATZM;
if(cfg.aercu_opt>0) QSATZM=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> GAMHAT;
if(cfg.aercu_opt>0) GAMHAT=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> AER_MMR;
if(cfg.aercu_opt>0) AER_MMR=arena.make<double>(1,PCOLS,KTS,KTE,1,NAER_CU);
MaArray<double> AQNEWIC;
if(cfg.aercu_opt>0) AQNEWIC=arena.make<double>(KTS,KTE);
MaArray<double> AQNEWLQ;
if(cfg.aercu_opt>0) AQNEWLQ=arena.make<double>(KTS,KTE);
MaArray<double> WU_MSKF_ACT;
if(cfg.aercu_opt>0) WU_MSKF_ACT=arena.make<double>(KTS,KTE);
MaArray<double> QC_MSKF_ACT;
if(cfg.aercu_opt>0) QC_MSKF_ACT=arena.make<double>(KTS,KTE);
MaArray<double> QI_MSKF_ACT;
if(cfg.aercu_opt>0) QI_MSKF_ACT=arena.make<double>(KTS,KTE);
MaArray<double> EFFC;
if(cfg.aercu_opt>0) EFFC=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> EFFI;
if(cfg.aercu_opt>0) EFFI=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> EFFS;
if(cfg.aercu_opt>0) EFFS=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> QSATU;
if(cfg.aercu_opt>0) QSATU=arena.make<double>(KTS,KTE);
MaArray<double> OLDQU;
if(cfg.aercu_opt>0) OLDQU=arena.make<double>(KTS,KTE);
MaArray<double> OLDTU;
if(cfg.aercu_opt>0) OLDTU=arena.make<double>(KTS,KTE);
MaArray<double> EPSI0;
if(cfg.aercu_opt>0) EPSI0=arena.make<double>(1,PCOLS);
MaArray<double> DLFMZMP;
if(cfg.aercu_opt>0) DLFMZMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> DIFMZMP;
if(cfg.aercu_opt>0) DIFMZMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> OLDPPTLIQ;
if(cfg.aercu_opt>0) OLDPPTLIQ=arena.make<double>(KTS,KTE);
MaArray<double> OLDPPTICE;
if(cfg.aercu_opt>0) OLDPPTICE=arena.make<double>(KTS,KTE);
MaArray<double> WUMP;
if(cfg.aercu_opt>0) WUMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZMQLIQ;
if(cfg.aercu_opt>0) ZMQLIQ=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZMQICE;
if(cfg.aercu_opt>0) ZMQICE=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZMQRAIN;
if(cfg.aercu_opt>0) ZMQRAIN=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZMQSNOW;
if(cfg.aercu_opt>0) ZMQSNOW=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> NCMP;
if(cfg.aercu_opt>0) NCMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> NIMP;
if(cfg.aercu_opt>0) NIMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> NRMP;
if(cfg.aercu_opt>0) NRMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> NSMP;
if(cfg.aercu_opt>0) NSMP=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> ZMCCN;
if(cfg.aercu_opt>0) ZMCCN=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> RPRD;
if(cfg.aercu_opt>0) RPRD=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> SPRD;
if(cfg.aercu_opt>0) SPRD=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<double> FRZ;
if(cfg.aercu_opt>0) FRZ=arena.make<double>(1,PCOLS,KTS,KTE);
MaArray<int> JBB;
if(cfg.aercu_opt>0) JBB=arena.make<int>(1,1);
MaArray<int> JTT;
if(cfg.aercu_opt>0) JTT=arena.make<int>(1,1);
MaArray<int> JLCL;
if(cfg.aercu_opt>0) JLCL=arena.make<int>(1,1);
int KQ=0,JK=0,MSG1=0,IL2G=1,NLEVZM=KTE,NLEVZMP1=KTE+1,I=1,J=1;
float EPS1U=0.622f,ALATENT=2.54E6f,DENSPLUME=0,TC_HU10,SF_HU10,A1KAY;
double CPIN=CP,RDRY=R,GRAV=G,DTZMP=DT,PSH_FAC;
const float P1_HU10=7.6725f,P2_HU10=1.0118f,P3_HU10=0.1422f,P4_HU10=0.0106f,P5_HU10=3.39E-4f,P6_HU10=3.95E-6f;
#define AEROCU(I,K,J,S) in.aerocu[(((size_t)(S)-1)*nz+((K)-1))*ncol+col]
    for (;;) {   /* usl */
        NU = NU + 1;
        if (NU > NCHECK) {
            if (ISHALL == 1) {
                float CHMAX = 0.0f;
                NCHM = 0;
                for (int nk = 1; nk <= NCHECK; ++nk) {
                    int NNN = KCHECK(nk);
                    if (CLDHGT(NNN) > CHMAX) {
                        NCHM = NNN;
                        NUCHM = nk;
                        CHMAX = CLDHGT(NNN);
                    }
                }
                NU = NUCHM - 1;
                FBFRC = 1.0f;
                continue;
            } else {
                goto para_done;
            }
        }
        {
        int KMIX = KCHECK(NU);
        int LOW = KMIX;
        LC = LOW;
        int NLAYRS = 0;
        DPTHMX = 0.0f;
        NK = LC - 1;
        if (NK + 1 < KTS) {
            /* 'WOULD GO OFF BOTTOM' message only */
        } else {
            for (;;) {
                NK = NK + 1;
                if (NK > KTE) break;
                DPTHMX = DPTHMX + DP(NK);
                NLAYRS = NLAYRS + 1;
                if (DPTHMX > DPMIN) break;
            }
        }
        if (DPTHMX < DPMIN) goto para_done;
        KPBL = LC + NLAYRS - 1;
        }
        TMIX = 0.0f; QMIX = 0.0f; ZMIX = 0.0f; PMIX = 0.0f;
        for (NK = LC; NK <= KPBL; ++NK) {
            TMIX = TMIX + DP(NK) * T0(NK);
            QMIX = QMIX + DP(NK) * Q0(NK);
            ZMIX = ZMIX + DP(NK) * Z0(NK);
            PMIX = PMIX + DP(NK) * P0(NK);
        }
        TMIX = TMIX / DPTHMX;
        QMIX = QMIX / DPTHMX;
        ZMIX = ZMIX / DPTHMX;
        PMIX = PMIX / DPTHMX;
        {
            float EMIX = QMIX * PMIX / (0.622f + QMIX);
            float astrt = 1.e-3f, ainc = 0.075f;
            float a1 = EMIX / ALIQ;
            float tp = (a1 - astrt) / ainc;
            int indlu = (int)tp + 1;
            float value = (float)(indlu - 1) * ainc + astrt;
            float aintrp = (a1 - value) / ainc;
            float tlog = aintrp * ms_alu(L, indlu + 1)
                         + (1.0f - aintrp) * ms_alu(L, indlu);
            float TDPT = (CLIQ - DLIQ * tlog) / (BLIQ - tlog);
            TLCL = TDPT - (0.212f + 1.571E-3f * (TDPT - T00)
                           - 4.36E-4f * (TMIX - T00)) * (TMIX - TDPT);
        }
        TLCL = ms_min(TLCL, TMIX);
        TVLCL = TLCL * (1.0f + 0.608f * QMIX);
        ZLCL = ZMIX + (TLCL - TMIX) / GDRY;
        for (NK = LC; NK <= KL; ++NK) {
            KLCL = NK;
            if (ZLCL <= Z0(NK)) break;
        }
        if (ZLCL > Z0(KL)) goto para_done;
        K = KLCL - 1;
        DLP = (ZLCL - Z0(K)) / (Z0(KLCL) - Z0(K));
        TENV = T0(K) + (T0(KLCL) - T0(K)) * DLP;
        QENV = Q0(K) + (Q0(KLCL) - Q0(K)) * DLP;
        TVEN = TENV * (1.0f + 0.608f * QENV);
        {
        float DTRH = 0.0f;
        /* 4559-4570: the Bechtold DTLCL is overwritten below on both
         * branches before any read, so it is not formed here. */
        if (ZLCL < 2.E3f) WKLCL = 0.02f * ZLCL / 2.E3f;
        else WKLCL = 0.02f;
        if (DX >= 25.E3f) {
            WKL = (W0AVG1D(K) + (W0AVG1D(KLCL) - W0AVG1D(K)) * DLP) * DX
                  / 25.E3f - WKLCL;
        } else {
            WKL = (W0AVG1D(K) + (W0AVG1D(KLCL) - W0AVG1D(K)) * DLP) - WKLCL;
        }
        if (WKL < 0.0001f) DTLCL = 0.0f;
        else DTLCL = 4.64f * MS_POW(WKL, 0.33f);
        if (TLCL + DTLCL < TENV) continue;   /* CYCLE usl */

        THETEU(K) = ms_envirtht(L, PMIX, TMIX, QMIX, ALIQ, BLIQ, CLIQ, DLIQ);
        float DTTOT = DTLCL + DTRH;
        if (DTTOT > 1.E-4f) {
            float GDT = 2.0f * G * DTTOT * 500.0f / TVEN;
            WLCL = 1.0f + 0.5f * sqrtf(GDT);
            WLCL = ms_min(WLCL, 3.0f);
        } else {
            WLCL = 1.0f;
        }
        }
        PLCL = P0(K) + (P0(KLCL) - P0(K)) * DLP;
        WTW = WLCL * WLCL;
        TVLCL = TLCL * (1.0f + 0.608f * QMIX);
        RHOLCL = PLCL / (R * TVLCL);
        LCL = KLCL;
        LET = LCL;
        RAD = ZLCL;
        {
        float sourceht = Z0(KPBL);
        RAD = ms_max(sourceht, RAD);
        }
        RAD = ms_min(4000.0f, RAD);
        RAD = ms_max(500.0f, RAD);
        WU(K) = WLCL;
        AU0 = 0.01f * DXSQ;
        UMF(K) = RHOLCL * AU0;
        VMFLCL = UMF(K);
        UPOLD = VMFLCL;
        UPNEW = UPOLD;
        RATIO2(K) = 0.0f;
        UER(K) = 0.0f;
        ABE = 0.0f;
        TRPPT = 0.0f;
        TU(K) = TLCL;
        TVU(K) = TVLCL;
        QU(K) = QMIX;
        EQFRC(K) = 1.0f;
        QLIQ(K) = 0.0f;
        QICE(K) = 0.0f;
        if (cfg.aercu_opt > 0) {
            QRAIN(K) = 0.0f; QSNOW(K) = 0.0f; NLIQ(K) = 0.0f;
            NICE(K) = 0.0f; NRAIN(K) = 0.0f; NSNOW(K) = 0.0f; CCN(K) = 0.0f;
            EFFCH(K) = 2.5f; EFFIH(K) = 4.99f; EFFSH(K) = 9.99f;
        }
        QLQOUT(K) = 0.0f;
        QICOUT(K) = 0.0f;
        DETLQ(K) = 0.0f;
        DETIC(K) = 0.0f;
        PPTLIQ(K) = 0.0f;
        PPTICE(K) = 0.0f;
        IFLAG = 0;
        TTEMP = TTFRZ;
        EE1 = 1.0f;
        UD1 = 0.0f;
        REI = 0.0f;
        DILBE = 0.0f;
        // WRF:4717
if((cfg.aercu_opt>0)) {
// WRF:4718
ZF_WRF(0) = 0.0f;
// WRF:4719
for(KQ=KTS; KQ<=KTE; KQ+=1) {
// WRF:4720
ZF_WRF(KQ) = (ZF_WRF((KQ-1))+DZQ(KQ));
// WRF:4721
AQNEWLQ(KQ) = 0.0f;
// WRF:4722
AQNEWIC(KQ) = 0.0f;
// WRF:4723
RPRD(1,KQ) = 0.0f;
// WRF:4724
WUMP(1,KQ) = 0.0f;
// WRF:4725
NCMP(1,KQ) = 0.0f;
// WRF:4726
NIMP(1,KQ) = 0.0f;
// WRF:4727
SPRD(1,KQ) = 0.0f;
// WRF:4728
FRZ(1,KQ) = 0.0f;
// WRF:4729
JK = KQ;
// WRF:4730
MUU(1,JK) = 0.0f;
// WRF:4731
DUU(1,JK) = 0.0f;
// WRF:4732
EUU(1,JK) = 0.0f;
// WRF:4733
CMEL(1,JK) = 0.0f;
// WRF:4734
CMEI(1,JK) = 0.0f;
// WRF:4735
OLDTU(KQ) = T0(KQ);
// WRF:4736
OLDQU(KQ) = Q0(KQ);
// Defined-read reference: QSATu outside the plume.
QSATU(KQ)=0.0;
// WRF:4737
}
// WRF:4739
}
        for (NK = K; NK <= KL - 1; ++NK) {   /* updraft */
            int NK1 = NK + 1;
            float FRC1 = 0.0f;
            RATIO2(NK1) = RATIO2(NK);
            TU(NK1) = T0(NK1);
            THETEU(NK1) = THETEU(NK);
            QU(NK1) = QU(NK);
            // WRF:4749
if((cfg.aercu_opt>0)) {
// WRF:4750
OLDQU(NK) = QU(NK);
// WRF:4751
OLDTU(NK) = TU(NK);
// WRF:4752
}
            QLIQ(NK1) = QLIQ(NK);
            QICE(NK1) = QICE(NK);
            {
                float tu = TU(NK1), qu = QU(NK1), ql = QLIQ(NK1),
                      qi = QICE(NK1);
                ms_tpmix2(L, P0(NK1), THETEU(NK1), &tu, &qu, &ql, &qi,
                          &QNEWLQ, &QNEWIC, XLV1, XLV0, &QSu);
                TU(NK1) = tu; QU(NK1) = qu; QLIQ(NK1) = ql; QICE(NK1) = qi;
            }
            if (TU(NK1) <= TTFRZ) {
                if (TU(NK1) > TBFRZ) {
                    if (TTEMP > TTFRZ) TTEMP = TTFRZ;
                    FRC1 = (TTEMP - TU(NK1)) / (TTEMP - TBFRZ);
                } else {
                    FRC1 = 1.0f;
                    IFLAG = 1;
                }
                TTEMP = TU(NK1);
                // WRF:4801
if((cfg.aercu_opt>0)) {
// WRF:4802
if((TU(NK1)>TBFRZ)) {
// WRF:4803
TC_HU10 = (TU(NK1)-273.15f);
// WRF:4804
SF_HU10 = ((-1.0f)*((((((P1_HU10+((P2_HU10*TC_HU10)))+((P3_HU10*(ma_powi<2>(TC_HU10)))))+((P4_HU10*(ma_powi<3>(TC_HU10)))))+((P5_HU10*(ma_powi<4>(TC_HU10)))))+((P6_HU10*(ma_powi<5>(TC_HU10)))))));
// WRF:4806
FRC1 = (1.0f-((1.0f/((1.0f+ma_exp(SF_HU10))))));
// WRF:4807
} else {
// WRF:4808
FRC1 = 1.f;
// WRF:4809
IFLAG = 1;
// WRF:4810
}
// WRF:4811
}
                float QFRZ = (QLIQ(NK1) + QNEWLQ) * FRC1;
                QNEWIC = QNEWIC + QNEWLQ * FRC1;
                QNEWLQ = QNEWLQ - QNEWLQ * FRC1;
                QICE(NK1) = QICE(NK1) + QLIQ(NK1) * FRC1;
                QLIQ(NK1) = QLIQ(NK1) - QLIQ(NK1) * FRC1;
                {
                    float tu = TU(NK1), th = THETEU(NK1), qu = QU(NK1),
                          qi = QICE(NK1);
                    ms_dtfrznew(&tu, P0(NK1), &th, &qu, QFRZ, &qi, ALIQ, BLIQ,
                                CLIQ, DLIQ);
                    TU(NK1) = tu; THETEU(NK1) = th; QU(NK1) = qu;
                    QICE(NK1) = qi;
                }
            }
            TVU(NK1) = TU(NK1) * (1.0f + 0.608f * QU(NK1));
            // WRF:4827
if((cfg.aercu_opt>0)) {
// WRF:4828
QSATU(NK) = (QSu/((1.f+QSu)));
// WRF:4829
AQNEWLQ(NK) = QNEWLQ;
// WRF:4830
AQNEWIC(NK) = QNEWIC;
// WRF:4831
}
            float BE, BOTERM, DZZ;
            if (NK == K) {
                BE = (TVLCL + TVU(NK1)) / (TVEN + TV0(NK1)) - 1.0f;
                BOTERM = 2.0f * (Z0(NK1) - ZLCL) * G * BE / 1.5f;
                DZZ = Z0(NK1) - ZLCL;
            } else {
                BE = (TVU(NK) + TVU(NK1)) / (TV0(NK) + TV0(NK1)) - 1.0f;
                BOTERM = 2.0f * DZA(NK) * G * BE / 1.5f;
                DZZ = DZA(NK);
            }
            float ENTERM = 2.0f * REI * WTW / UPOLD;
            float RATE_kay;
            if (DX >= 24.999E3f) RATE_kay = RATE;
            else RATE_kay = RATE / Scale_Fac;
            {
                float ql = QLIQ(NK1), qi = QICE(NK1), lo, io;
                ms_condload(&ql, &qi, &WTW, DZZ, BOTERM, ENTERM, RATE_kay,
                            &QNEWLQ, &QNEWIC, &lo, &io, G);
                QLIQ(NK1) = ql; QICE(NK1) = qi; QLQOUT(NK1) = lo;
                QICOUT(NK1) = io;
            }
            if (WTW < 1.E-3f) {
                break;
            } else {
                WU(NK1) = sqrtf(WTW);
            }
            THETEE(NK1) = ms_envirtht(L, P0(NK1), T0(NK1), Q0(NK1), ALIQ,
                                      BLIQ, CLIQ, DLIQ);
            float TOKIOKA = 0.03f;
            TOKIOKA = TOKIOKA * Scale_Fac;
            REI = VMFLCL * DP(NK1) * TOKIOKA / RAD;
            TVQU(NK1) = TU(NK1) * (1.0f + 0.608f * QU(NK1) - QLIQ(NK1)
                                   - QICE(NK1));
            if (NK == K) {
                DILBE = ((TVLCL + TVQU(NK1)) / (TVEN + TV0(NK1)) - 1.0f) * DZZ;
            } else {
                DILBE = ((TVQU(NK) + TVQU(NK1)) / (TV0(NK) + TV0(NK1)) - 1.0f)
                        * DZZ;
            }
            if (DILBE > 0.0f) ABE = ABE + DILBE * G;
            float EE2, UD2;
            if (TVQU(NK1) <= TV0(NK1)) {
                EE2 = 0.5f;
                UD2 = 1.0f;
                EQFRC(NK1) = 0.0f;
            } else {
                LET = NK1;
                float TTMP = TVQU(NK1);
                float F1 = 0.95f;
                float F2 = 1.0f - F1;
                float THTTMP = F1 * THETEE(NK1) + F2 * THETEU(NK1);
                float QTMP = F1 * Q0(NK1) + F2 * QU(NK1);
                float TMPLIQ = F2 * QLIQ(NK1);
                float TMPICE = F2 * QICE(NK1);
                ms_tpmix2(L, P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ, &TMPICE,
                          &QNEWLQ, &QNEWIC, XLV1, XLV0, &QSu);
                float TU95 = TTMP * (1.0f + 0.608f * QTMP - TMPLIQ - TMPICE);
                if (TU95 > TV0(NK1)) {
                    EE2 = 1.0f;
                    UD2 = 0.0f;
                    EQFRC(NK1) = 1.0f;
                } else {
                    F1 = 0.10f;
                    F2 = 1.0f - F1;
                    THTTMP = F1 * THETEE(NK1) + F2 * THETEU(NK1);
                    QTMP = F1 * Q0(NK1) + F2 * QU(NK1);
                    TMPLIQ = F2 * QLIQ(NK1);
                    TMPICE = F2 * QICE(NK1);
                    ms_tpmix2(L, P0(NK1), THTTMP, &TTMP, &QTMP, &TMPLIQ,
                              &TMPICE, &QNEWLQ, &QNEWIC, XLV1, XLV0, &QSu);
                    float TU10 = TTMP * (1.0f + 0.608f * QTMP - TMPLIQ
                                         - TMPICE);
                    float TVDIFF = fabsf(TU10 - TVQU(NK1));
                    if (TVDIFF < 1.e-3f) {
                        EE2 = 1.0f;
                        UD2 = 0.0f;
                        EQFRC(NK1) = 1.0f;
                    } else {
                        EQFRC(NK1) = (TV0(NK1) - TVQU(NK1)) * F1
                                     / (TU10 - TVQU(NK1));
                        EQFRC(NK1) = ms_max(0.0f, EQFRC(NK1));
                        EQFRC(NK1) = ms_min(1.0f, EQFRC(NK1));
                        if (EQFRC(NK1) == 1.0f) {
                            EE2 = 1.0f;
                            UD2 = 0.0f;
                        } else if (EQFRC(NK1) == 0.0f) {
                            EE2 = 0.0f;
                            UD2 = 1.0f;
                        } else {
                            ms_prof5(EQFRC(NK1), &EE2, &UD2);
                        }
                    }
                }
            }
            EE2 = ms_max(EE2, 0.5f);
            UD2 = 1.5f * UD2;
            UER(NK1) = 0.5f * REI * (EE1 + EE2);
            UDR(NK1) = 0.5f * REI * (UD1 + UD2);
            if (UMF(NK) - UDR(NK1) < 10.0f) {
                if (DILBE > 0.0f) ABE = ABE - DILBE * G;
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
                THETEU(NK1) = (THETEU(NK1) * UPOLD + THETEE(NK1) * UER(NK1))
                              / UPNEW;
                QLIQ(NK1) = QLIQ(NK1) * UPOLD / UPNEW;
                QICE(NK1) = QICE(NK1) * UPOLD / UPNEW;
                PPTLIQ(NK1) = QLQOUT(NK1) * UMF(NK);
                PPTICE(NK1) = QICOUT(NK1) * UMF(NK);
                TRPPT = TRPPT + PPTLIQ(NK1) + PPTICE(NK1);
                if (NK1 <= KPBL) UER(NK1) = UER(NK1) + VMFLCL * DP(NK1)
                                                       / DPTHMX;
            }
            // WRF:5006
if((cfg.aercu_opt>0)) {
// WRF:5007
EPS1U = 0.622f;
// WRF:5008
ALATENT = 2.54e6f;
// WRF:5009
KQ = NK;
// WRF:5010
JK = ((KTE-KQ)+1);
// WRF:5011
MUU(1,JK) = (UMF(KQ)/VMFLCL);
// WRF:5012
DUU(1,JK) = ((UDR(KQ)/DZQ(KQ))/VMFLCL);
// WRF:5013
EUU(1,JK) = ((UER(KQ)/DZQ(KQ))/VMFLCL);
// WRF:5014
CMEL(1,JK) = ((MUU(1,JK)*AQNEWLQ(KQ))/DZQ(KQ));
// WRF:5015
CMEI(1,JK) = ((MUU(1,JK)*AQNEWIC(KQ))/DZQ(KQ));
// WRF:5016
GAMHAT(1,JK) = ((((((QSATU(KQ)*((1.f+(QSATU(KQ)/EPS1U))))*EPS1U)*ALATENT)/((R*ma_powi<2>(OLDTU(KQ)))))*ALATENT)/CP);
// WRF:5018
WU_MSKF_ACT(JK) = WU(KQ);
// WRF:5019
QC_MSKF_ACT(JK) = AQNEWLQ(KQ);
// WRF:5020
QI_MSKF_ACT(JK) = AQNEWIC(KQ);
// WRF:5021
}
        }   /* updraft */
        if(cfg.aercu_opt>0) {
// WRF:5030
ZFU(1,(KTE+1)) = 0.0f;
// WRF:5032
CPIN = CP;
// WRF:5034
EPSI0(1) = 2.0e-4f;
// WRF:5035
for(KQ=KTS; KQ<=KTE; KQ+=1) {
// WRF:5036
JK = ((KTE-KQ)+1);
// WRF:5037
ZFU(1,JK) = ZF_WRF(KQ);
// WRF:5038
SU(1,JK) = ((OLDTU(KQ)*((1.0f+(0.622f*QSATU(KQ)))))+(((G*ZF_WRF(KQ)))/CP));
// WRF:5039
QUU(1,JK) = (OLDQU(KQ)/((1.f+OLDQU(KQ))));
// WRF:5040
PRU(1,JK) = (P0(KQ)/100.0f);
// WRF:5041
TEE(1,JK) = T0(KQ);
// WRF:5042
QEE(1,JK) = (Q0(KQ)/((1.f+Q0(KQ))));
// WRF:5044
QEE(1,JK) = (OLDQU(KQ)/((1.f+OLDQU(KQ))));
// WRF:5045
QSATZM(1,JK) = QSATU(KQ);
// WRF:5049
DENSPLUME = (P0(KQ)/((R*OLDTU(KQ))));
// WRF:5050
PSH_FAC = (1.0e-09f/DENSPLUME);
// WRF:5052
AER_MMR(1,JK,1) = ((cfg.aercu_fct*AEROCU(I,KQ,J,6))*PSH_FAC);
// WRF:5053
AER_MMR(1,JK,2) = ((cfg.aercu_fct*AEROCU(I,KQ,J,5))*PSH_FAC);
// WRF:5054
AER_MMR(1,JK,3) = (((cfg.aercu_fct*1.44f)*AEROCU(I,KQ,J,1))*PSH_FAC);
// WRF:5055
AER_MMR(1,JK,4) = (((cfg.aercu_fct*1.44f)*AEROCU(I,KQ,J,2))*PSH_FAC);
// WRF:5056
AER_MMR(1,JK,5) = (((cfg.aercu_fct*1.44f)*AEROCU(I,KQ,J,3))*PSH_FAC);
// WRF:5057
AER_MMR(1,JK,6) = (((cfg.aercu_fct*1.44f)*AEROCU(I,KQ,J,4))*PSH_FAC);
// WRF:5058
AER_MMR(1,JK,7) = (((cfg.aercu_fct*1.54f)*AEROCU(I,KQ,J,9))*PSH_FAC);
// WRF:5059
AER_MMR(1,JK,8) = (((cfg.aercu_fct*1.37f)*AEROCU(I,KQ,J,7))*PSH_FAC);
// WRF:5060
AER_MMR(1,JK,9) = (((cfg.aercu_fct*1.25f)*AEROCU(I,KQ,J,10))*PSH_FAC);
// WRF:5061
AER_MMR(1,JK,10) = (((cfg.aercu_fct*1.37f)*AEROCU(I,KQ,J,8))*PSH_FAC);
// WRF:5064
GAMHAT(1,JK) = ((((((QSATU(KQ)*((1.f+(QSATU(KQ)/EPS1U))))*EPS1U)*ALATENT)/((R*ma_powi<2>(OLDTU(KQ)))))*ALATENT)/CP);
// WRF:5067
}
// WRF:5068
JTT(1) = ((KX-NK)+1);
// WRF:5069
JBB(1) = ((KX-K)+1);
// WRF:5070
if((JTT(1)>JBB(1))) {
// WRF:5071
JTT(1) = JBB(1);
// WRF:5072
}
// WRF:5073
JLCL(1) = (JBB(1)-1);
// WRF:5074
MSG1 = 0;
// WRF:5075
IL2G = 1;
// WRF:5076
GRAV = G;
// WRF:5077
RDRY = R;
// WRF:5078
DTZMP = DT;
// WRF:5082
NLEVZM = ((KTE-KTS)+1);
// WRF:5083
NLEVZMP1 = (NLEVZM+1);
// WRF:5085
if((JTT(1)==1)) {
atomicOr(out.status, MSKF_FATAL_CLOUD_BASE); return;
// WRF:5089
}
// WRF:5090
if((JBB(1)==KTE)) {
atomicOr(out.status, MSKF_FATAL_CLOUD_TOP); return;
// WRF:5094
}
// WRF:5095
if((cfg.aercu_opt>0)) {
// WRF:5101
A.mskf_mphy(SU,QUU,MUU,DUU,CMEL,CMEI,ZFU,PRU,TEE,QEE,EPSI0,JBB,JTT,JLCL,MSG1,IL2G,GRAV,CPIN,RDRY,ZMQLIQ,ZMQICE,ZMQRAIN,ZMQSNOW,RPRD,WUMP,EUU,NCMP,NIMP,NRMP,NSMP,ZMCCN,SPRD,FRZ,AER_MMR,DTZMP,NLEVZM,NLEVZMP1,GAMHAT,QSATZM,WU_MSKF_ACT,QC_MSKF_ACT,QI_MSKF_ACT,EFFC,EFFI,EFFS); if(A.error) {atomicOr(out.status,A.error);return;}
// WRF:5105
}
// WRF:5126
if((cfg.aercu_opt>0)) {
// WRF:5127
for(KQ=KTS; KQ<=KTE; KQ+=1) {
// WRF:5128
QLIQ(KQ) = 0.0f;
// WRF:5129
QICE(KQ) = 0.0f;
// WRF:5130
QRAIN(KQ) = 0.0f;
// WRF:5131
QSNOW(KQ) = 0.0f;
// WRF:5132
NLIQ(KQ) = 0.0f;
// WRF:5133
NICE(KQ) = 0.0f;
// WRF:5134
NRAIN(KQ) = 0.0f;
// WRF:5135
NSNOW(KQ) = 0.0f;
// WRF:5136
CCN(KQ) = 0.0f;
// WRF:5137
EFFCH(KQ) = 2.51f;
// WRF:5138
EFFIH(KQ) = 4.99f;
// WRF:5139
EFFSH(KQ) = 9.99f;
// WRF:5140
PPTLIQ(KQ) = 0.0f;
// WRF:5141
PPTICE(KQ) = 0.0f;
// WRF:5142
QLQOUT(KQ) = 0.0f;
// WRF:5143
QICOUT(KQ) = 0.0f;
// WRF:5144
DETLQ(KQ) = 0.0f;
// WRF:5145
DETIC(KQ) = 0.0f;
// WRF:5146
}
// WRF:5147
TRPPT = 0.0f;
// WRF:5148
for(KQ=KTS; KQ<=KTE; KQ+=1) {
// WRF:5150
JK = ((KX-KQ)+1);
// WRF:5152
QLIQ(KQ) = ma_max(0.,ZMQLIQ(1,JK));
// WRF:5153
QICE(KQ) = ma_max(0.,ZMQICE(1,JK));
// WRF:5155
QRAIN(KQ) = ma_max(0.,ZMQRAIN(1,JK));
// WRF:5156
QSNOW(KQ) = ma_max(0.,ZMQSNOW(1,JK));
// WRF:5157
NLIQ(KQ) = ma_max(0.,NCMP(1,JK));
// WRF:5158
NICE(KQ) = ma_max(0.,NIMP(1,JK));
// WRF:5159
NRAIN(KQ) = ma_max(0.,NRMP(1,JK));
// WRF:5160
NSNOW(KQ) = ma_max(0.,NSMP(1,JK));
// WRF:5161
CCN(KQ) = ma_max(0.,ZMCCN(1,JK));
// WRF:5162
EFFCH(KQ) = ma_max(2.49,ma_min(EFFC(1,JK),50.));
// WRF:5163
EFFIH(KQ) = ma_max(4.99,ma_min(EFFI(1,JK),125.));
// WRF:5164
EFFSH(KQ) = ma_max(9.99,ma_min(EFFS(1,JK),999.));
// WRF:5166
DETLQ(KQ) = (QLIQ(KQ)*UDR(KQ));
// WRF:5167
DETIC(KQ) = (QICE(KQ)*UDR(KQ));
// WRF:5171
if((RPRD(1,JK)<0.0f)) { RPRD(1,JK) = 0.0f; }
// WRF:5172
if((SPRD(1,JK)<0.0f)) { SPRD(1,JK) = 0.0f; }
// WRF:5174
QLQOUT(KQ) = (RPRD(1,JK)*DZQ(KQ));
// WRF:5175
QICOUT(KQ) = (SPRD(1,JK)*DZQ(KQ));
// WRF:5176
PPTLIQ(KQ) = (QLQOUT(KQ)*VMFLCL);
// WRF:5177
PPTICE(KQ) = (QICOUT(KQ)*VMFLCL);
// WRF:5179
TRPPT = ((TRPPT+PPTLIQ(KQ))+PPTICE(KQ));
// WRF:5191
}
// WRF:5193
}
}

        LTOP = NK;
        CLDHGT(LC) = Z0(LTOP) - ZLCL;
        if (TLCL > 293.0f) {
            CHMIN = 4.E3f;
        } else if (TLCL <= 293.0f && TLCL >= 273.0f) {
            CHMIN = 2.E3f + 100.0f * (TLCL - 273.0f);
        } else if (TLCL < 273.0f) {
            CHMIN = 2.E3f;
        }
        for (NK = K; NK <= LTOP; ++NK) {
            out.qc_kf[IX(NK)] = QLIQ(NK);
            out.qi_kf[IX(NK)] = QICE(NK);
            if (cfg.aercu_opt > 0) {
                out.qr_kf[IX(NK)] = QRAIN(NK);
                out.qs_kf[IX(NK)] = QSNOW(NK);
                out.nc_kf[IX(NK)] = NLIQ(NK);
                out.ni_kf[IX(NK)] = NICE(NK);
                out.nr_kf[IX(NK)] = NRAIN(NK);
                out.ns_kf[IX(NK)] = NSNOW(NK);
                out.ccn_kf[IX(NK)] = CCN(NK);
                out.efcs[IX(NK)] = ms_max(2.49f, ms_min(EFFCH(NK), 50.0f));
                out.efis[IX(NK)] = ms_max(4.99f, ms_min(EFFIH(NK), 120.0f));
                out.efss[IX(NK)] = ms_max(9.99f, ms_min(EFFSH(NK), 999.0f));
            }
        }
        /* 5246-5264, the saturated-environment refusal.  D1: only levels
         * that exist are summed (WRF reads K-1 = 0 and LTOP+1 = KTE+1). */
        float envRHavg = 0.0f;
        {
            int lo = K - 1, hi = LTOP + 1, nsum = 0;
            bool early = false;
            for (NK = lo; NK <= hi; ++NK) {
                if (NK < 1 || NK > KL) continue;
                float envEsat;
                if (T0(NK) <= 273.16f) {
                    envEsat = 6.112f * MS_EXP(21.87f * (T0(NK) - 273.16f)
                                              / (T0(NK) - 7.66f));
                } else {
                    envEsat = 6.112f * MS_EXP(17.67f * (T0(NK) - 273.16f)
                                              / (243.5f + T0(NK) - 273.16f));
                }
                envEsat = envEsat * 100.0f;
                float envQsat = 0.622f * envEsat / (P0(NK) - envEsat);
                float envRH = Q0(NK) / envQsat;
                if (NK > K && envRH < 0.99f) {
                    envRHavg = 0.0f;
                    early = true;
                    break;
                }
                envRHavg = envRHavg + envRH;
                nsum = nsum + 1;
            }
            if (!early) {
                int nwrf = LTOP - K + 1 + 2;
                envRHavg = envRHavg / (float)(nsum == (hi - lo + 1)
                                              ? nwrf : nsum);
            }
        }
        if (LTOP <= KLCL || LTOP <= KPBL || LET + 1 <= KPBL
                || envRHavg >= 1.01f) {
            CLDHGT(LC) = 0.0f;
            for (NK = K; NK <= LTOP; ++NK) {
                UMF(NK) = 0.0f; UDR(NK) = 0.0f; UER(NK) = 0.0f;
                DETLQ(NK) = 0.0f; DETIC(NK) = 0.0f;
                PPTLIQ(NK) = 0.0f; PPTICE(NK) = 0.0f;
                out.cldfra_dp[IX(NK)] = 0.0f;
                out.cldfra_sh[IX(NK)] = 0.0f;
                out.qc_kf[IX(NK)] = 0.0f;
                out.qi_kf[IX(NK)] = 0.0f;
                if (cfg.aercu_opt > 0) {
                    out.qr_kf[IX(NK)] = 0.0f; out.qs_kf[IX(NK)] = 0.0f;
                    out.nc_kf[IX(NK)] = 0.0f; out.ni_kf[IX(NK)] = 0.0f;
                    out.nr_kf[IX(NK)] = 0.0f; out.ns_kf[IX(NK)] = 0.0f;
                    out.ccn_kf[IX(NK)] = 0.0f;
                    out.efcs[IX(NK)] = 2.51f;
                    out.efis[IX(NK)] = 5.01f;
                    out.efss[IX(NK)] = 10.01f;
                }
            }
        } else if (CLDHGT(LC) > CHMIN && ABE > 1.0f) {
            ISHALL = 0;
            for (NK = K; NK <= LTOP; ++NK) out.cldfra_sh[IX(NK)] = 0.0f;
            break;   /* EXIT usl */
        } else {
            ISHALL = 1;
            for (NK = K; NK <= LTOP; ++NK) out.cldfra_dp[IX(NK)] = 0.0f;
            if (NU == NUCHM) {
                break;   /* EXIT usl */
            } else {
                for (NK = K; NK <= LTOP; ++NK) {
                    UMF(NK) = 0.0f; UDR(NK) = 0.0f; UER(NK) = 0.0f;
                    DETLQ(NK) = 0.0f; DETIC(NK) = 0.0f;
                    PPTLIQ(NK) = 0.0f; PPTICE(NK) = 0.0f;
                    out.cldfra_dp[IX(NK)] = 0.0f;
                    out.cldfra_sh[IX(NK)] = 0.0f;
                    out.qc_kf[IX(NK)] = 0.0f;
                    out.qi_kf[IX(NK)] = 0.0f;
                    if (cfg.aercu_opt > 0) {
                        out.qr_kf[IX(NK)] = 0.0f; out.qs_kf[IX(NK)] = 0.0f;
                        out.nc_kf[IX(NK)] = 0.0f; out.ni_kf[IX(NK)] = 0.0f;
                        out.nr_kf[IX(NK)] = 0.0f; out.ns_kf[IX(NK)] = 0.0f;
                        out.ccn_kf[IX(NK)] = 0.0f;
                        out.efcs[IX(NK)] = 2.51f;
                        out.efis[IX(NK)] = 5.01f;
                        out.efss[IX(NK)] = 10.01f;
                    }
                }
            }
        }
    }   /* usl */
    {   /* everything after the source-layer search, in its own scope so the
         * early exits above may jump to para_done */
        int KSTART = 0;
        if (ISHALL == 1) {
            KSTART = max(KPBL, KLCL);
            LET = KSTART;
        }
        if (LET == LTOP) {
            UDR(LTOP) = UMF(LTOP) + UDR(LTOP) - UER(LTOP);
            DETLQ(LTOP) = QLIQ(LTOP) * UDR(LTOP) * UPNEW / UPOLD;
            DETIC(LTOP) = QICE(LTOP) * UDR(LTOP) * UPNEW / UPOLD;
            UER(LTOP) = 0.0f;
            UMF(LTOP) = 0.0f;
        } else {
            float DPTT = 0.0f;
            for (int NJ = LET + 1; NJ <= LTOP; ++NJ) DPTT = DPTT + DP(NJ);
            float DUMFDP = UMF(LET) / DPTT;
            for (NK = LET + 1; NK <= LTOP; ++NK) {
                if (NK == LTOP) {
                    UDR(NK) = UMF(NK - 1);
                    UER(NK) = 0.0f;
                    DETLQ(NK) = UDR(NK) * QLIQ(NK) * DILFRC(NK);
                    DETIC(NK) = UDR(NK) * QICE(NK) * DILFRC(NK);
                } else {
                    UMF(NK) = UMF(NK - 1) - DP(NK) * DUMFDP;
                    UER(NK) = UMF(NK) * (1.0f - 1.0f / DILFRC(NK));
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
        for (NK = 1; NK <= LTOP; ++NK) {
            if (T0(NK) > T00) ML = NK;
        }
        for (NK = 1; NK <= K; ++NK) {
            if (NK >= LC) {
                if (NK == LC) {
                    UMF(NK) = VMFLCL * DP(NK) / DPTHMX;
                    UER(NK) = VMFLCL * DP(NK) / DPTHMX;
                } else if (NK <= KPBL) {
                    UER(NK) = VMFLCL * DP(NK) / DPTHMX;
                    UMF(NK) = UMF(NK - 1) + UER(NK);
                } else {
                    UMF(NK) = VMFLCL;
                    UER(NK) = 0.0f;
                }
                TU(NK) = TMIX + (Z0(NK) - ZMIX) * GDRY;
                QU(NK) = QMIX;
                WU(NK) = WLCL;
            } else {
                TU(NK) = 0.0f; QU(NK) = 0.0f; UMF(NK) = 0.0f; WU(NK) = 0.0f;
                UER(NK) = 0.0f;
                out.cldfra_dp[IX(NK)] = 0.0f;
                out.cldfra_sh[IX(NK)] = 0.0f;
                out.qc_kf[IX(NK)] = 0.0f;
                out.qi_kf[IX(NK)] = 0.0f;
                if (cfg.aercu_opt > 0) {
                    out.qr_kf[IX(NK)] = 0.0f; out.qs_kf[IX(NK)] = 0.0f;
                    out.nc_kf[IX(NK)] = 0.0f; out.ni_kf[IX(NK)] = 0.0f;
                    out.nr_kf[IX(NK)] = 0.0f; out.ns_kf[IX(NK)] = 0.0f;
                    out.ccn_kf[IX(NK)] = 0.0f;
                    out.efcs[IX(NK)] = 2.51f;
                    out.efis[IX(NK)] = 5.01f;
                    out.efss[IX(NK)] = 10.01f;
                }
            }
            UDR(NK) = 0.0f; QDT(NK) = 0.0f; QLIQ(NK) = 0.0f; QICE(NK) = 0.0f;
            QLQOUT(NK) = 0.0f; QICOUT(NK) = 0.0f; PPTLIQ(NK) = 0.0f;
            PPTICE(NK) = 0.0f; DETLQ(NK) = 0.0f; DETIC(NK) = 0.0f;
            RATIO2(NK) = 0.0f;
            THETEE(NK) = ms_envirtht(L, P0(NK), T0(NK), Q0(NK), ALIQ, BLIQ,
                                     CLIQ, DLIQ);
            EQFRC(NK) = 1.0f;
        }
        int LTOP1 = LTOP + 1;
        int LTOPM1 = LTOP - 1;
        for (NK = LTOP1; NK <= KX; ++NK) {
            UMF(NK) = 0.0f; UDR(NK) = 0.0f; UER(NK) = 0.0f; QDT(NK) = 0.0f;
            QLIQ(NK) = 0.0f; QICE(NK) = 0.0f; QLQOUT(NK) = 0.0f;
            QICOUT(NK) = 0.0f; DETLQ(NK) = 0.0f; DETIC(NK) = 0.0f;
            PPTLIQ(NK) = 0.0f; PPTICE(NK) = 0.0f;
            if (NK > LTOP1) {
                TU(NK) = 0.0f; QU(NK) = 0.0f; WU(NK) = 0.0f;
                out.cldfra_dp[IX(NK)] = 0.0f;
                out.cldfra_sh[IX(NK)] = 0.0f;
                out.qc_kf[IX(NK)] = 0.0f;
                out.qi_kf[IX(NK)] = 0.0f;
                if (cfg.aercu_opt > 0) {
                    out.qr_kf[IX(NK)] = 0.0f; out.qs_kf[IX(NK)] = 0.0f;
                    out.nc_kf[IX(NK)] = 0.0f; out.ni_kf[IX(NK)] = 0.0f;
                    out.nr_kf[IX(NK)] = 0.0f; out.ns_kf[IX(NK)] = 0.0f;
                    out.ccn_kf[IX(NK)] = 0.0f;
                    out.efss[IX(NK)] = 10.01f;
                    out.efcs[IX(NK)] = 2.51f;
                    out.efis[IX(NK)] = 5.01f;
                }
            }
            THTA0(NK) = 0.0f; THTAU(NK) = 0.0f; EMS(NK) = 0.0f;
            EMSD(NK) = 0.0f;
            TG(NK) = T0(NK);
            QG(NK) = Q0(NK);
            QLG(NK) = 0.0f; QIG(NK) = 0.0f; QRG(NK) = 0.0f; QSG(NK) = 0.0f;
            OMG(NK) = 0.0f;
        }
        OMG(KX + 1) = 0.0f;
        for (NK = 1; NK <= LTOP; ++NK) {
            EMS(NK) = DP(NK) * DXSQ / G;
            EMSD(NK) = 1.0f / EMS(NK);
            EXN(NK) = MS_POW(P00 / P0(NK), 0.2854f * (1.0f - 0.28f * QDT(NK)));
            THTAU(NK) = TU(NK) * EXN(NK);
            EXN(NK) = MS_POW(P00 / P0(NK), 0.2854f * (1.0f - 0.28f * Q0(NK)));
            THTA0(NK) = T0(NK) * EXN(NK);
            DDILFRC(NK) = 1.0f / DILFRC(NK);
            OMG(NK) = 0.0f;
        }
        /* the convective time scale (5565-5619) */
        float WSPD_KLCL = sqrtf(U0(KLCL) * U0(KLCL) + V0(KLCL) * V0(KLCL));
        float WSPD_L5 = sqrtf(U0(L5) * U0(L5) + V0(L5) * V0(L5));
        float WSPD_LTOP = sqrtf(U0(LTOP) * U0(LTOP) + V0(LTOP) * V0(LTOP));
        float VCONV = 0.5f * (WSPD_KLCL + WSPD_L5);
        float TIMEC = DX / VCONV;
        float TADVEC = TIMEC;
        TIMEC = ms_max(CHMIN, CLDHGT(LC));
        TIMEC = TIMEC * Scale_Fac;
        float densPlume = P0(1) / R / T0(1);
        float WST = in.hfx[col] / densPlume / CP;
        WST = ms_max(0.0f, WST);
        WST = G * in.pblh[col] * WST;
        float thetav = MS_POW(1.E5f / P0(1), R / CP);
        thetav = T0(1) * thetav;
        float eps1u = 0.622f;
        thetav = thetav * (1.0f + Q0(1) * eps1u);
        WST = WST / thetav;
        WST = MS_POW(WST, 0.3333f);
        float SCLvel = WST * WST * WST;
        float ZLCL_KAY = ms_max(ZLCL, Z0(KPBL));
        SCLvel = SCLvel / in.pblh[col];
        SCLvel = SCLvel * ZLCL_KAY;
        SCLvel = MS_POW(SCLvel, 0.333f);
        float FRC2;
        float UST = in.ust[col], ZOL = in.zol[col];
        if (ZOL <= 0.0f) {
            FRC2 = 3.8f * UST * UST;
            FRC2 = FRC2 + 0.22f * SCLvel * SCLvel;
            float zz_kay = -1.0f * ZOL;
            ZLCL_KAY = MS_POW(zz_kay, MS_TWO_THIRDS);
            ZLCL_KAY = ZLCL_KAY * (1.9f * UST * UST);
            FRC2 = FRC2 + ZLCL_KAY;
        } else {
            FRC2 = 3.8f * UST * UST;
        }
        FRC2 = sqrtf(FRC2);
        SCLvel = FRC2;
        if (SCLvel < 0.1f) SCLvel = 0.1f;
        if (ABE <= 0.0f) ABE = 1.0f;
        TIMEC = TIMEC / (MS_POW(0.03f * SCLvel * ABE, 0.3333f));
        TIMEC = ms_max(TADVEC, TIMEC);
        int NIC = ms_nint(TIMEC / DT);
        TIMEC = (float)NIC * DT;
        TIMEC = ms_min(TIMEC, 86400.0f);
        /* shear and precipitation efficiency (5623-5651) */
        float SHSIGN = (WSPD_LTOP > WSPD_KLCL) ? 1.0f : -1.0f;
        float VWS = (U0(LTOP) - U0(KLCL)) * (U0(LTOP) - U0(KLCL))
                    + (V0(LTOP) - V0(KLCL)) * (V0(LTOP) - V0(KLCL));
        VWS = 1.E3f * SHSIGN * sqrtf(VWS) / (Z0(LTOP) - Z0(LCL));
        float PEF = 1.591f + VWS * (-0.639f + VWS * (9.53E-2f
                                                     - VWS * 4.96E-3f));
        PEF = ms_max(PEF, 0.2f);
        PEF = ms_min(PEF, 0.9f);
        float CBH = (ZLCL - Z0(1)) * 3.281E-3f;
        float RCBH;
        if (CBH < 3.0f) {
            RCBH = 0.02f;
        } else {
            RCBH = 0.96729352f + CBH * (-0.70034167f + CBH * (0.162179896f
                   + CBH * (-1.2569798E-2f + CBH * (4.2772E-4f
                                                   - CBH * 5.44E-6f))));
        }
        if (CBH > 25.0f) RCBH = 2.4f;
        float PEFCBH = 1.0f / (1.0f + RCBH);
        PEFCBH = ms_min(PEFCBH, 0.9f);
        float PEFF = 0.5f * (PEF + PEFCBH);
        /* the downdraft (5666-5803) */
        float TDER = 0.0f, DMFFRC = 0.0f, DPDD = 0.0f;
        int LFS, LDB = 0, LDT = 0;
        if (ISHALL == 1) {
            LFS = 1;
        } else {
            KSTART = KPBL + 1;
            int KLFS = LET - 1;
            for (NK = KSTART + 1; NK <= KL; ++NK) {
                float DPPP = P0(KSTART) - P0(NK);
                if (DPPP > 150.E2f) {
                    KLFS = NK;
                    break;
                }
            }
            KLFS = min(KLFS, LET - 1);
            LFS = KLFS;
            if ((P0(KSTART) - P0(LFS)) > 50.E2f) {
                THETED(LFS) = THETEE(LFS);
                QD(LFS) = Q0(LFS);
                float QSS;
                {
                    float tz;
                    ms_tpmix2dd(L, P0(LFS), THETED(LFS), &tz, &QSS);
                    TZ(LFS) = tz;
                }
                THTAD(LFS) = TZ(LFS) * MS_POW(P00 / P0(LFS),
                                              0.2854f * (1.0f - 0.28f * QSS));
                TVD(LFS) = TZ(LFS) * (1.0f + 0.608f * QSS);
                float RDD = P0(LFS) / (R * TVD(LFS));
                float A1 = (1.0f - PEFF) * AU0;
                DMF(LFS) = -A1 * RDD;
                DER(LFS) = DMF(LFS);
                DDR(LFS) = 0.0f;
                float RHBAR = RH(LFS) * DP(LFS);
                float DPTT = DP(LFS);
                for (int ND = LFS - 1; ND >= KSTART; --ND) {
                    int ND1 = ND + 1;
                    DER(ND) = DER(LFS) * EMS(ND) / EMS(LFS);
                    DDR(ND) = 0.0f;
                    DMF(ND) = DMF(ND1) + DER(ND);
                    THETED(ND) = (THETED(ND1) * DMF(ND1) + THETEE(ND) * DER(ND))
                                 / DMF(ND);
                    QD(ND) = (QD(ND1) * DMF(ND1) + Q0(ND) * DER(ND)) / DMF(ND);
                    DPTT = DPTT + DP(ND);
                    RHBAR = RHBAR + RH(ND) * DP(ND);
                }
                RHBAR = RHBAR / DPTT;
                DMFFRC = 2.0f * (1.0f - RHBAR);
                DPDD = 0.0f;
                float PPTMLT = 0.0f;
                for (NK = KLCL; NK <= LTOP; ++NK) PPTMLT = PPTMLT + PPTICE(NK);
                float DTMELT;
                if (LC < ML) DTMELT = RLF * PPTMLT / (CP * UMF(KLCL));
                else DTMELT = 0.0f;
                LDT = min(LFS - 1, KSTART - 1);
                {
                    float tz;
                    ms_tpmix2dd(L, P0(KSTART), THETED(KSTART), &tz, &QSS);
                    TZ(KSTART) = tz;
                }
                TZ(KSTART) = TZ(KSTART) - DTMELT;
                float ES = ALIQ * MS_EXP((BLIQ * TZ(KSTART) - CLIQ)
                                         / (TZ(KSTART) - DLIQ));
                QSS = 0.622f * ES / (P0(KSTART) - ES);
                THETED(KSTART) = TZ(KSTART) * MS_POW(1.E5f / P0(KSTART),
                    0.2854f * (1.0f - 0.28f * QSS))
                    * MS_EXP((3374.6525f / TZ(KSTART) - 2.5403f) * QSS
                             * (1.0f + 0.81f * QSS));
                LDT = min(LFS - 1, KSTART - 1);
                for (int ND = LDT; ND >= 1; --ND) {
                    DPDD = DPDD + DP(ND);
                    THETED(ND) = THETED(KSTART);
                    QD(ND) = QD(KSTART);
                    {
                        float tz;
                        ms_tpmix2dd(L, P0(ND), THETED(ND), &tz, &QSS);
                        TZ(ND) = tz;
                    }
                    QSD(ND) = QSS;
                    float RHH = 1.0f - 0.2f / 1000.0f * (Z0(KSTART) - Z0(ND));
                    if (RHH < 1.0f) {
                        float DSSDT = (CLIQ - BLIQ * DLIQ)
                                      / ((TZ(ND) - DLIQ) * (TZ(ND) - DLIQ));
                        float RL = XLV0 - XLV1 * TZ(ND);
                        float DTMP = RL * QSS * (1.0f - RHH)
                                     / (CP + RL * RHH * QSS * DSSDT);
                        float T1RH = TZ(ND) + DTMP;
                        ES = RHH * ALIQ * MS_EXP((BLIQ * T1RH - CLIQ)
                                                 / (T1RH - DLIQ));
                        float QSRH = 0.622f * ES / (P0(ND) - ES);
                        if (QSRH < QD(ND)) {
                            QSRH = QD(ND);
                            T1RH = TZ(ND) + (QSS - QSRH) * RL / CP;
                        }
                        TZ(ND) = T1RH;
                        QSS = QSRH;
                        QSD(ND) = QSS;
                    }
                    TVD(ND) = TZ(ND) * (1.0f + 0.608f * QSD(ND));
                    if (TVD(ND) > TV0(ND) || ND == 1) {
                        LDB = ND;
                        break;
                    }
                }
                if ((P0(LDB) - P0(LFS)) > 50.E2f) {
                    for (int ND = LDT; ND >= LDB; --ND) {
                        int ND1 = ND + 1;
                        DDR(ND) = -DMF(KSTART) * DP(ND) / DPDD;
                        DER(ND) = 0.0f;
                        DMF(ND) = DMF(ND1) + DDR(ND);
                        TDER = TDER + (QSD(ND) - QD(ND)) * DDR(ND);
                        QD(ND) = QSD(ND);
                        THTAD(ND) = TZ(ND) * MS_POW(P00 / P0(ND),
                                    0.2854f * (1.0f - 0.28f * QD(ND)));
                    }
                }
            }
        }
        /* d_mf (5808-5893) */
        float PPTFLX, CPR;
        if (TDER < 1.0f) {
            PPTFLX = TRPPT;
            CPR = TRPPT;
            TDER = 0.0f;
            LDB = LFS;
            for (int NDK = 1; NDK <= LTOP; ++NDK) {
                DMF(NDK) = 0.0f; DER(NDK) = 0.0f; DDR(NDK) = 0.0f;
                THTAD(NDK) = 0.0f; TZ(NDK) = 0.0f; QD(NDK) = 0.0f;
            }
        } else {
            float DDINC = -DMFFRC * UMF(KLCL) / DMF(KSTART);
            if (TDER * DDINC > TRPPT) DDINC = TRPPT / TDER;
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
                    DMF(NK) = 0.0f; DER(NK) = 0.0f; DDR(NK) = 0.0f;
                    TZ(NK) = 0.0f; QD(NK) = 0.0f; THTAD(NK) = 0.0f;
                }
            }
            for (NK = LFS + 1; NK <= KX; ++NK) {
                DMF(NK) = 0.0f; DER(NK) = 0.0f; DDR(NK) = 0.0f;
                TZ(NK) = 0.0f; QD(NK) = 0.0f; THTAD(NK) = 0.0f;
            }
            for (NK = LDT + 1; NK <= LFS - 1; ++NK) {
                TZ(NK) = 0.0f; QD(NK) = 0.0f; THTAD(NK) = 0.0f;
            }
        }
        /* the closure (5899-6299) */
        float AINCMX = 1000.0f;
        int LMAX = max(KLCL, LFS);
        for (NK = LC; NK <= LMAX; ++NK) {
            if ((UER(NK) - DER(NK)) > 1.e-3f) {
                float AINCM1 = EMS(NK) / ((UER(NK) - DER(NK)) * TIMEC);
                AINCMX = ms_min(AINCMX, AINCM1);
            }
        }
        float AINC = 1.0f;
        if (AINCMX < AINC) AINC = AINCMX;
        float TDER2 = TDER, PPTFL2 = PPTFLX;
        for (NK = 1; NK <= LTOP; ++NK) {
            DETLQ2(NK) = DETLQ(NK); DETIC2(NK) = DETIC(NK);
            UDR2(NK) = UDR(NK); UER2(NK) = UER(NK);
            DDR2(NK) = DDR(NK); DER2(NK) = DER(NK);
            UMF2(NK) = UMF(NK); DMF2(NK) = DMF(NK);
        }
        float FABE = 1.0f;
        const float STAB = 0.95f;
        int NOITR = 0, ISTOP = 0;
        if (ISHALL == 1) {
            float TKEMAX = 5.0f;
            float EVAC = 0.5f * TKEMAX * 0.1f;
            AINC = EVAC * DPTHMX * DXSQ / (VMFLCL * G * TIMEC);
            TDER = TDER2 * AINC;
            PPTFLX = PPTFL2 * AINC;
            for (NK = 1; NK <= LTOP; ++NK) {
                UMF(NK) = UMF2(NK) * AINC; DMF(NK) = DMF2(NK) * AINC;
                DETLQ(NK) = DETLQ2(NK) * AINC; DETIC(NK) = DETIC2(NK) * AINC;
                UDR(NK) = UDR2(NK) * AINC; UER(NK) = UER2(NK) * AINC;
                DER(NK) = DER2(NK) * AINC; DDR(NK) = DDR2(NK) * AINC;
            }
        }
        int NSTEP = 1;
        float DTIME = 0.0f, AINCOLD = 0.0f, FABEOLD = 0.0f;
        bool bail = false;   /* a RETURN inside the iteration */
        for (int NCOUNT = 1; NCOUNT <= 10; ++NCOUNT) {   /* iter */
            float DTT = TIMEC;
            for (NK = 1; NK <= LTOP; ++NK) {
                DOMGDP(NK) = -(UER(NK) - DER(NK) - UDR(NK) - DDR(NK))
                             * EMSD(NK);
                if (NK > 1) {
                    OMG(NK) = OMG(NK - 1) - DP(NK - 1) * DOMGDP(NK - 1);
                    float ABSOMG = fabsf(OMG(NK));
                    float ABSOMGTC = ABSOMG * TIMEC;
                    float FRDP = 0.75f * DP(NK - 1);
                    if (ABSOMGTC > FRDP) {
                        float DTT1 = FRDP / ABSOMG;
                        DTT = ms_min(DTT, DTT1);
                    }
                }
            }
            for (NK = 1; NK <= LTOP; ++NK) {
                THPA(NK) = THTA0(NK);
                QPA(NK) = Q0(NK);
                NSTEP = ms_nint(TIMEC / DTT + 1.0f);
                DTIME = TIMEC / (float)NSTEP;
                FXM(NK) = OMG(NK) * DXSQ / G;
            }
            for (int NTC = 1; NTC <= NSTEP; ++NTC) {
                for (NK = 1; NK <= LTOP; ++NK) {
                    THFXIN(NK) = 0.0f; THFXOUT(NK) = 0.0f;
                    QFXIN(NK) = 0.0f; QFXOUT(NK) = 0.0f;
                }
                for (NK = 2; NK <= LTOP; ++NK) {
                    if (OMG(NK) <= 0.0f) {
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
                    THPA(NK) = THPA(NK) + (THFXIN(NK) + UDR(NK) * THTAU(NK)
                               + DDR(NK) * THTAD(NK) - THFXOUT(NK)
                               - (UER(NK) - DER(NK)) * THTA0(NK)) * DTIME
                               * EMSD(NK);
                    QPA(NK) = QPA(NK) + (QFXIN(NK) + UDR(NK) * QDT(NK)
                              + DDR(NK) * QD(NK) - QFXOUT(NK)
                              - (UER(NK) - DER(NK)) * Q0(NK)) * DTIME
                              * EMSD(NK);
                }
            }
            for (NK = 1; NK <= LTOP; ++NK) {
                THTAG(NK) = THPA(NK);
                QG(NK) = QPA(NK);
            }
            for (NK = 1; NK <= LTOP; ++NK) {
                if (QG(NK) < 0.0f) {
                    if (NK == 1) {
                        atomicOr(out.status, MSKF_FATAL_QG_SURFACE);
                        bail = true;
                        break;
                    }
                    int NK1 = NK + 1;
                    if (NK == LTOP) NK1 = KLCL;
                    float TMA = QG(NK1) * EMS(NK1);
                    float TMB = QG(NK - 1) * EMS(NK - 1);
                    float TMM = (QG(NK) - 1.E-9f) * EMS(NK);
                    float BCOEFF = -TMM / ((TMA * TMA) / TMB + TMB);
                    float ACOEFF = BCOEFF * TMA / TMB;
                    TMB = TMB * (1.0f - BCOEFF);
                    TMA = TMA * (1.0f - ACOEFF);
                    QG(NK) = 1.E-9f;
                    QG(NK1) = TMA * EMSD(NK1);
                    QG(NK - 1) = TMB * EMSD(NK - 1);
                }
            }
            if (bail) break;
            float TOPOMG = (UDR(LTOP) - UER(LTOP)) * DP(LTOP) * EMSD(LTOP);
            if (fabsf(TOPOMG - OMG(LTOP)) > 1.E-3f) {
                ISTOP = 1;
                break;
            }
            for (NK = 1; NK <= LTOP; ++NK) {
                EXN(NK) = MS_POW(P00 / P0(NK),
                                 0.2854f * (1.0f - 0.28f * QG(NK)));
                TG(NK) = THTAG(NK) / EXN(NK);
                TVG(NK) = TG(NK) * (1.0f + 0.608f * QG(NK));
            }
            if (ISHALL == 1) break;
            TMIX = 0.0f;
            QMIX = 0.0f;
            for (NK = LC; NK <= KPBL; ++NK) {
                TMIX = TMIX + DP(NK) * TG(NK);
                QMIX = QMIX + DP(NK) * QG(NK);
            }
            TMIX = TMIX / DPTHMX;
            QMIX = QMIX / DPTHMX;
            float ES = ALIQ * MS_EXP((TMIX * BLIQ - CLIQ) / (TMIX - DLIQ));
            float QSS = 0.622f * ES / (PMIX - ES);
            if (QMIX > QSS) {
                float RL = XLV0 - XLV1 * TMIX;
                float CPM = CP * (1.0f + 0.887f * QMIX);
                float DSSDT = QSS * (CLIQ - BLIQ * DLIQ)
                              / ((TMIX - DLIQ) * (TMIX - DLIQ));
                float DQ = (QMIX - QSS) / (1.0f + RL * DSSDT / CPM);
                TMIX = TMIX + RL / CP * DQ;
                QMIX = QMIX - DQ;
                TLCL = TMIX;
            } else {
                QMIX = ms_max(QMIX, 0.0f);
                float EMIX2 = QMIX * PMIX / (0.622f + QMIX);
                float astrt = 1.e-3f, binc = 0.075f;
                float a1 = EMIX2 / ALIQ;
                float tp = (a1 - astrt) / binc;
                int indlu = (int)tp + 1;
                float value = (float)(indlu - 1) * binc + astrt;
                float aintrp = (a1 - value) / binc;
                float tlog = aintrp * ms_alu(L, indlu + 1)
                             + (1.0f - aintrp) * ms_alu(L, indlu);
                float TDPT = (CLIQ - DLIQ * tlog) / (BLIQ - tlog);
                TLCL = TDPT - (0.212f + 1.571E-3f * (TDPT - T00)
                               - 4.36E-4f * (TMIX - T00)) * (TMIX - TDPT);
                TLCL = ms_min(TLCL, TMIX);
            }
            TVLCL = TLCL * (1.0f + 0.608f * QMIX);
            ZLCL = ZMIX + (TLCL - TMIX) / GDRY;
            for (NK = LC; NK <= KL; ++NK) {
                KLCL = NK;
                if (ZLCL <= Z0(NK)) break;
            }
            K = KLCL - 1;
            DLP = (ZLCL - Z0(K)) / (Z0(KLCL) - Z0(K));
            TENV = TG(K) + (TG(KLCL) - TG(K)) * DLP;
            QENV = QG(K) + (QG(KLCL) - QG(K)) * DLP;
            TVEN = TENV * (1.0f + 0.608f * QENV);
            PLCL = P0(K) + (P0(KLCL) - P0(K)) * DLP;
            THETEU(K) = TMIX * MS_POW(1.E5f / PMIX,
                                      0.2854f * (1.0f - 0.28f * QMIX))
                        * MS_EXP((3374.6525f / TLCL - 2.5403f) * QMIX
                                 * (1.0f + 0.81f * QMIX));
            float ABEG = 0.0f;
            for (NK = K; NK <= LTOPM1; ++NK) {
                int NK1 = NK + 1;
                // WRF:6178
if((cfg.aercu_opt>0.0f)) {
// WRF:6179
JK = ((KX-NK)+1);
// WRF:6180
A1KAY = (((FRZ(1,JK)*DZA(NK))*3.337e5f)/CP);
// WRF:6181
A1KAY = (A1KAY*(ma_pow(((1.e5f/P0(NK))),ROCP)));
// WRF:6182
THETEU(NK) = (A1KAY+THETEU(NK));
// WRF:6183
}
                THETEU(NK1) = THETEU(NK);
                {
                    float tg, qg;
                    ms_tpmix2dd(L, P0(NK1), THETEU(NK1), &tg, &qg);
                    TGU(NK1) = tg;
                    QGU(NK1) = qg;
                }
                TVQU(NK1) = TGU(NK1) * (1.0f + 0.608f * QGU(NK1) - QLIQ(NK1)
                                        - QICE(NK1));
                float DZZ;
                if (NK == K) {
                    DZZ = Z0(KLCL) - ZLCL;
                    DILBE = ((TVLCL + TVQU(NK1)) / (TVEN + TVG(NK1)) - 1.0f)
                            * DZZ;
                } else {
                    DZZ = DZA(NK);
                    DILBE = ((TVQU(NK) + TVQU(NK1)) / (TVG(NK) + TVG(NK1))
                             - 1.0f) * DZZ;
                }
                if (DILBE > 0.0f) ABEG = ABEG + DILBE * G;
                THTEEG(NK1) = ms_envirtht(L, P0(NK1), TG(NK1), QG(NK1), ALIQ,
                                          BLIQ, CLIQ, DLIQ);
                THETEU(NK1) = THETEU(NK1) * DDILFRC(NK1)
                              + THTEEG(NK1) * (1.0f - DDILFRC(NK1));
            }
            if (NOITR == 1) break;
            float DABE = ms_max(ABE - ABEG, capeDX * ABE);
            FABE = ABEG / ABE;
            if (FABE > 1.0f && ISHALL == 0) {
                bail = true;
                break;
            }
            if (NCOUNT != 1) {
                if (fabsf(AINC - AINCOLD) < 0.0001f) {
                    NOITR = 1;
                    AINC = AINCOLD;
                    continue;
                }
                float DFDA = (FABE - FABEOLD) / (AINC - AINCOLD);
                if (DFDA > 0.0f) {
                    NOITR = 1;
                    AINC = AINCOLD;
                    continue;
                }
            }
            AINCOLD = AINC;
            FABEOLD = FABE;
            if (AINC / AINCMX > 0.999f && FABE > 1.05f - STAB) break;
            if ((FABE <= 1.05f - STAB && FABE >= 0.95f - STAB)
                    || NCOUNT == 10) {
                break;
            } else {
                if (NCOUNT > 10) break;
                if (FABE == 0.0f) {
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
                AINC = ms_min(AINCMX, AINC);
                if (AINC < 0.05f) {
                    bail = true;
                    break;
                }
                TDER = TDER2 * AINC;
                PPTFLX = PPTFL2 * AINC;
                for (NK = 1; NK <= LTOP; ++NK) {
                    UMF(NK) = UMF2(NK) * AINC; DMF(NK) = DMF2(NK) * AINC;
                    DETLQ(NK) = DETLQ2(NK) * AINC;
                    DETIC(NK) = DETIC2(NK) * AINC;
                    UDR(NK) = UDR2(NK) * AINC; UER(NK) = UER2(NK) * AINC;
                    DER(NK) = DER2(NK) * AINC; DDR(NK) = DDR2(NK) * AINC;
                }
            }
        }   /* iter */
        if (ISTOP == 1) {
            /* 6086 sets IPRNT and 6544 stops the model */
            atomicOr(out.status, MSKF_FATAL_MASS_BUDGET);
            bail = true;
        }
        if (!bail) {
        /* cloud fraction (6303-6334) */
        float updil = (100.0f - AINC);
        updil = updil / 100.0f;
        if (cfg.aercu_opt > 0) out.ainc_frac[col] = 1.0f - updil;
        updil = updil * DXSQ;
        if (ISHALL == 1) {
            for (NK = KLCL; NK <= LTOP; ++NK) {
                float UMF_new = UMF(NK) / updil;
                float xcldfra = 0.07f * MS_LOG(1.0f + (500.0f * UMF_new));
                xcldfra = ms_max(0.01f, xcldfra);
                out.cldfra_sh[IX(NK)] = ms_min(0.2f, xcldfra);
            }
        } else {
            for (NK = KLCL; NK <= LTOP; ++NK) {
                float UMF_new = UMF(NK) / updil;
                float xcldfra = 0.14f * MS_LOG(1.0f + (500.0f * UMF_new));
                xcldfra = ms_max(0.01f, xcldfra);
                out.cldfra_dp[IX(NK)] = ms_min(0.6f, xcldfra);
            }
        }
        if (cfg.kf_edrates == 1) {
            for (NK = 1; NK <= LTOP; ++NK) {
                out.udr_kf[IX(NK)] = UDR(NK);
                out.ddr_kf[IX(NK)] = DDR(NK);
                out.uer_kf[IX(NK)] = UER(NK);
                out.der_kf[IX(NK)] = DER(NK);
            }
        }
        /* hydrometeor transport (6364-6428) */
        float FRC2b;
        if (CPR > 0.0f) FRC2b = PPTFLX / (CPR * AINC);
        else FRC2b = 0.0f;
        for (NK = 1; NK <= LTOP; ++NK) {
            QLPA(NK) = 0.0f; QIPA(NK) = 0.0f; QRPA(NK) = 0.0f;
            QSPA(NK) = 0.0f;
            RAINFB(NK) = PPTLIQ(NK) * AINC * FBFRC * FRC2b;
            SNOWFB(NK) = PPTICE(NK) * AINC * FBFRC * FRC2b;
        }
        for (int NTC = 1; NTC <= NSTEP; ++NTC) {
            for (NK = 1; NK <= LTOP; ++NK) {
                QLFXIN(NK) = 0.0f; QLFXOUT(NK) = 0.0f;
                QIFXIN(NK) = 0.0f; QIFXOUT(NK) = 0.0f;
                QRFXIN(NK) = 0.0f; QRFXOUT(NK) = 0.0f;
                QSFXIN(NK) = 0.0f; QSFXOUT(NK) = 0.0f;
            }
            for (NK = 2; NK <= LTOP; ++NK) {
                if (OMG(NK) <= 0.0f) {
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
                QLPA(NK) = QLPA(NK) + (QLFXIN(NK) + DETLQ(NK) - QLFXOUT(NK))
                           * DTIME * EMSD(NK);
                QIPA(NK) = QIPA(NK) + (QIFXIN(NK) + DETIC(NK) - QIFXOUT(NK))
                           * DTIME * EMSD(NK);
                QRPA(NK) = QRPA(NK) + (QRFXIN(NK) - QRFXOUT(NK) + RAINFB(NK))
                           * DTIME * EMSD(NK);
                QSPA(NK) = QSPA(NK) + (QSFXIN(NK) - QSFXOUT(NK) + SNOWFB(NK))
                           * DTIME * EMSD(NK);
            }
        }
        for (NK = 1; NK <= LTOP; ++NK) {
            QLG(NK) = QLPA(NK); QIG(NK) = QIPA(NK);
            QRG(NK) = QRPA(NK); QSG(NK) = QSPA(NK);
        }
        if (cfg.kf_edrates == 1) out.timec_kf[col] = TIMEC;
        float PRATEC = PPTFLX * (1.0f - FBFRC) / DXSQ;
        out.pratec[col] = PRATEC;
        out.raincv[col] = DT * PRATEC;
        if (TADVEC < TIMEC) NIC = ms_nint(TADVEC / DT);
        out.nca[col] = (float)NIC * DT;
        if (ISHALL == 1) out.nca[col] = CUDT * 60.0f;
        /* the phase feedback (6624-6688) and MSKF_CPS 3951-3981.  QL0,
         * QI0, QR0 and QS0 are identically zero, and x - 0 is x for every
         * float, so the subtractions are not spelled. */
        for (int k = 1; k <= KX; ++k) {
            float dqc, dqi = 0.0f, dqr, dqs = 0.0f;
            if (cfg.warm_rain) {
                float CPM = CP * (1.0f + 0.887f * QG(k));
                TG(k) = TG(k) - (QIG(k) + QSG(k)) * RLF / CPM;
                dqc = (QLG(k) + QIG(k)) / TIMEC;
                dqr = (QRG(k) + QSG(k)) / TIMEC;
            } else if (!cfg.f_qs) {
                float CPM = CP * (1.0f + 0.887f * QG(k));
                if (k <= ML) TG(k) = TG(k) - (QIG(k) + QSG(k)) * RLF / CPM;
                else if (k > ML) TG(k) = TG(k) + (QLG(k) + QRG(k)) * RLF
                                                 / CPM;
                dqc = (QLG(k) + QIG(k)) / TIMEC;
                dqr = (QRG(k) + QSG(k)) / TIMEC;
            } else {
                dqc = QLG(k) / TIMEC;
                dqs = QSG(k) / TIMEC;
                dqr = QRG(k) / TIMEC;
                if (cfg.f_qi) dqi = QIG(k) / TIMEC;
                else dqs = dqs + QIG(k) / TIMEC;
            }
            float dtdt = (TG(k) - T0(k)) / TIMEC;
            float dqdt = (QG(k) - Q0(k)) / TIMEC;
            out.rthcuten[IX(k)] = dtdt / in.pi[IX(k)];
            out.rqvcuten[IX(k)] = dqdt;
            if (cfg.f_qr) {
                out.rqrcuten[IX(k)] = dqr;
                out.rqccuten[IX(k)] = dqc;
            } else {
                out.rqrcuten[IX(k)] = 0.0f;
                out.rqccuten[IX(k)] = dqr + dqc;
            }
            if (cfg.f_qi) out.rqicuten[IX(k)] = dqi;
            if (cfg.f_qs) out.rqscuten[IX(k)] = dqs;
        }
        feedback = true;
        /* the CMT hand-off (6691-6866), flipped to J = KTE-KQ+1 */
        {
            int dsl = 0;
            MsCmt C;
            C.DUDT = ms_dslot(db, dsl++, ext); C.DVDT = ms_dslot(db, dsl++, ext);
            C.DP = ms_dslot(db, dsl++, ext); C.ED = ms_dslot(db, dsl++, ext);
            C.EU = ms_dslot(db, dsl++, ext); C.MC = ms_dslot(db, dsl++, ext);
            C.MD = ms_dslot(db, dsl++, ext); C.MU = ms_dslot(db, dsl++, ext);
            C.P = ms_dslot(db, dsl++, ext); C.QD = ms_dslot(db, dsl++, ext);
            C.QU = ms_dslot(db, dsl++, ext); C.QHAT = ms_dslot(db, dsl++, ext);
            C.SD = ms_dslot(db, dsl++, ext); C.SU = ms_dslot(db, dsl++, ext);
            C.SHAT = ms_dslot(db, dsl++, ext); C.T = ms_dslot(db, dsl++, ext);
            C.U = ms_dslot(db, dsl++, ext); C.V = ms_dslot(db, dsl++, ext);
            C.Z = ms_dslot(db, dsl++, ext); C.ZF = ms_dslot(db, dsl++, ext);
            C.AC = ms_dslot(db, dsl++, ext); C.AD = ms_dslot(db, dsl++, ext);
            C.AU = ms_dslot(db, dsl++, ext); C.ACFL = ms_dslot(db, dsl++, ext);
            C.ADFL = ms_dslot(db, dsl++, ext); C.AUFL = ms_dslot(db, dsl++, ext);
            C.B1 = ms_dslot(db, dsl++, ext); C.B1FL = ms_dslot(db, dsl++, ext);
            C.BD = ms_dslot(db, dsl++, ext); C.BU = ms_dslot(db, dsl++, ext);
            C.D0 = ms_dslot(db, dsl++, ext); C.D0HAT = ms_dslot(db, dsl++, ext);
            C.DELPX = ms_dslot(db, dsl++, ext);
            C.DELPY = ms_dslot(db, dsl++, ext);
            C.DZ = ms_dslot(db, dsl++, ext); C.DZF = ms_dslot(db, dsl++, ext);
            C.EC = ms_dslot(db, dsl++, ext); C.E = ms_dslot(db, dsl++, ext);
            C.EHAT = ms_dslot(db, dsl++, ext); C.RHO = ms_dslot(db, dsl++, ext);
            C.RHOHAT = ms_dslot(db, dsl++, ext);
            C.UHAT = ms_dslot(db, dsl++, ext); C.VHAT = ms_dslot(db, dsl++, ext);
            C.UC = ms_dslot(db, dsl++, ext); C.VC = ms_dslot(db, dsl++, ext);
            C.W0 = ms_dslot(db, dsl++, ext); C.W1 = ms_dslot(db, dsl++, ext);
            C.W0FL = ms_dslot(db, dsl++, ext); C.W1FL = ms_dslot(db, dsl++, ext);
            C.FX = ms_dslot(db, dsl++, ext); C.FY = ms_dslot(db, dsl++, ext);
            C.AA = ms_dslot(db, dsl++, ext); C.BB = ms_dslot(db, dsl++, ext);
            C.CC = ms_dslot(db, dsl++, ext); C.DXN = ms_dslot(db, dsl++, ext);
            C.DYN = ms_dslot(db, dsl++, ext); C.ZFW = ms_dslot(db, dsl++, ext);
            C.LRHO = ms_dslot(db, dsl++, ext); C.WW = ms_dslot(db, dsl++, ext);
            C.B1T = ms_dslot(db, dsl++, ext); C.BASEBB = ms_dslot(db, dsl++, ext);
            C.BASECC = ms_dslot(db, dsl++, ext);
            for (int KQ = KTS; KQ <= KTE; ++KQ) {
                int JK = KTE - KQ + 1;
                C.DP(JK) = 0.0; C.ED(JK) = 0.0; C.EU(JK) = 0.0; C.MC(JK) = 0.0;
                C.MU(JK) = 0.0; C.P(JK) = 0.0; C.QD(JK) = 0.0; C.QU(JK) = 0.0;
                C.QHAT(JK) = 0.0; C.SD(JK) = 0.0; C.SU(JK) = 0.0;
                C.SHAT(JK) = 0.0; C.T(JK) = 0.0; C.U(JK) = 0.0;
                C.V(JK) = 0.0; C.Z(JK) = 0.0; C.ZF(JK) = 0.0;
            }
            C.ZFW(0) = 0.0;
            /* grav = 9.8 is a default-REAL literal stored into REAL(r8) */
            const double grav = (double)9.8f;
            const double cpin = (double)CP;
            const double Rdry = (double)R;
            C.ZF(KTE + 1) = 0.0;
            const double DTnew = (double)DT;
            const double DSUBCLD = (double)ZLCL;
            int JBB = KTE - KLCL + 1;
            int JDD = KTE - LFS + 1;
            int JTT = KTE - LTOP + 1;
            if (JDD < JTT || JDD > JBB) JDD = JBB - 1;
            if (JTT > JBB) JTT = JBB;
            for (int KQ = KTS; KQ <= KTE; ++KQ) {
                TDN(KQ) = T0(KQ);
                TUP(KQ) = T0(KQ);
            }
            for (int KQ = KLCL; KQ <= LTOP; ++KQ) {
                if (TZ(KQ) != 0.0f) TDN(KQ) = TZ(KQ);
                if (TU(KQ) != 0.0f) TUP(KQ) = TU(KQ);
            }
            for (int KQ = KTS; KQ <= KTE; ++KQ) {
                C.ZFW(KQ) = C.ZFW(KQ - 1) + (double)DZQ(KQ);
                STATE(KQ) = (float)((double)(CP * (T0(KQ) * (1.0f + 0.622f
                                                             * Q0(KQ))))
                                    + (double)G * C.ZFW(KQ));
            }
            for (int KQ = KTS; KQ <= KTE; ++KQ) {
                int JK = KTE - KQ + 1;
                C.DUDT(JK) = 0.0;
                C.DVDT(JK) = 0.0;
                C.ZFW(KQ) = C.ZFW(KQ - 1) + (double)DZQ(KQ);
                C.ZF(JK) = C.ZFW(KQ);
                if (KQ == KTE) {
                    C.QHAT(JK) = (double)Q0(KQ);
                    C.SHAT(JK) = (double)STATE(KQ);
                } else {
                    C.QHAT(JK) = (double)(0.5f * (Q0(KQ) / (1.0f + Q0(KQ))
                                          + Q0(KQ + 1) / (1.0f + Q0(KQ + 1))));
                    C.SHAT(JK) = (double)(0.5f * (STATE(KQ) + STATE(KQ + 1)));
                }
                if (KQ < KLCL || KQ > LTOP) {
                    C.QU(JK) = (double)(Q0(KQ) / (1.0f + Q0(KQ)));
                    C.QD(JK) = (double)(Q0(KQ) / (1.0f + Q0(KQ)));
                    double se = (double)(CP * (T0(KQ) * (1.0f + 0.622f
                                                         * Q0(KQ))))
                                + (double)G * C.ZFW(KQ);
                    C.SU(JK) = se;
                    C.SD(JK) = se;
                } else {
                    C.QU(JK) = (double)(QU(KQ) / (1.0f + QU(KQ)));
                    C.QD(JK) = (double)(QD(KQ) / (1.0f + QD(KQ)));
                    C.SU(JK) = (double)CP * ((double)TUP(KQ)
                               * (1.0 + (double)0.622f * C.QU(JK)))
                               + (double)G * C.ZFW(KQ);
                    C.SD(JK) = (double)CP * ((double)TDN(KQ)
                               * (1.0 + (double)0.622f * C.QD(JK)))
                               + (double)G * C.ZFW(KQ);
                }
                C.ED(JK) = (double)((((DER(KQ) / DXSQ) * G) / 100.0f)
                                    / DZQ(KQ));
                C.EU(JK) = (double)((((UER(KQ) / DXSQ) * G) / 100.0f)
                                    / DZQ(KQ));
                C.MD(JK) = (double)(((DMF(KQ) / DXSQ) * G) / 100.0f);
                C.MU(JK) = (double)(((UMF(KQ) / DXSQ) * G) / 100.0f);
                C.MC(JK) = C.MD(JK) + C.MU(JK);
                C.DP(JK) = (double)(DP(KQ) / 100.0f);
                C.P(JK) = (double)(P0(KQ) / 100.0f);
                C.U(JK) = (double)U0(KQ);
                C.V(JK) = (double)V0(KQ);
                C.Z(JK) = (double)Z0(KQ);
                C.T(JK) = (double)T0(KQ);
            }
            ms_cmt(C, DSUBCLD, JBB, JDD, JTT, 0, DTnew, grav, cpin, Rdry,
                   KTE);
            for (int KQ = KTS; KQ <= KTE; ++KQ) {
                int JK = KTE - KQ + 1;
                out.rucuten[IX(KQ)] = (float)C.DUDT(JK);
                out.rvcuten[IX(KQ)] = (float)C.DVDT(JK);
            }
        }
        out.cutop[col] = (float)LTOP;
        out.cubot[col] = (float)LCL;
        }   /* !bail */
    }
para_done:
    if (!feedback) {
        /* MSKF_eta_PARA returned before its feedback: MSKF_CPS writes the
         * zero tendencies it initialised (3944-3981). */
        for (int k = KTS; k <= KTE; ++k) {
            out.rucuten[IX(k)] = 0.0f;
            out.rvcuten[IX(k)] = 0.0f;
            out.rthcuten[IX(k)] = 0.0f / in.pi[IX(k)];
            out.rqvcuten[IX(k)] = 0.0f;
            out.rqrcuten[IX(k)] = 0.0f;
            out.rqccuten[IX(k)] = 0.0f;
            if (cfg.f_qi) out.rqicuten[IX(k)] = 0.0f;
            if (cfg.f_qs) out.rqscuten[IX(k)] = 0.0f;
        }
    }
#undef T0
#undef P0
#undef U0
#undef V0
#undef DZQ
#undef W0AVG1D
#undef AEROCU
#undef IX
}

/* ======================================================================
 * W0AVG running mean, MSKF_CPS 3649-3681, every column of the tile.
 * ====================================================================== */
extern "C" __global__ void mskf_w0avg(
        float* __restrict__ w0avg, const float* __restrict__ w,
        float w0avgfctr, float w0fctr, float w0den, int nz, int ncol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nz * ncol) return;
    int k = idx / ncol, c = idx % ncol;
    float w0 = 0.5f * (w[(size_t)k * ncol + c] + w[(size_t)(k + 1) * ncol + c]);
    w0avg[idx] = (w0avg[idx] * w0avgfctr + w0 * w0fctr) / w0den;
}

/* Upstream embedded error-function notice, retained verbatim.
!    Copyright(C) 1996 Takuya OOURA (email: ooura@mmm.t.u-tokyo.ac.jp).
!    You may use, copy, modify this code for any purpose and
!    without fee. You may distribute this ORIGINAL package.
*/

#if defined(GPUWM_MSKF_FAST) && !defined(GPUWM_MSKF_NO_AEROSOL)
// WRF MSKF_MPHYI (module_cu_mskf.F:241-385; called at :3913) sets
// invariant tables per column. Share its unchanged values per device image.
extern "C" __global__ void mskf_aerosol_init(double* table) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        MaPhysics constants(table);
    }
}
#endif
