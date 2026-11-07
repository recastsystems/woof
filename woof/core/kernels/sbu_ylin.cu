// gpuwm/core/kernels/sbu_ylin.cu
//
// Stony Brook University Y. Lin five-class microphysics (WRF mp_physics = 13),
// transcribed in WRF's own operation order from the public-domain
//   phys/module_mp_sbu_ylin.F   (WRF v4.6.1, d66e442, 1,745 lines)
// sbu_ylin (:61-199) -> clphy1d_ylin (:203-1460) -> satadj (:1468-1601), with
// the in-file parama1/parama2 (:1605-1657) and ggamma (:1713-1740).  No lookup
// table and no file read: the scheme is closed-form, so there is no table path.
//
// One CUDA thread owns one column.  gpuwm stores (k, j, i) with x fastest;
// k = 0 is WRF's kts.  Fortran's left-to-right evaluation is kept term by
// term; every division is __fdiv_rn so NVRTC cannot turn a constant divisor
// into a reciprocal multiply (A146).  Under GPUWM_WRF_EXACT=1 the process-wide
// strict options also forbid multiply-add contraction, which is what the
// 0-ULP column oracle (tools/sbu_ylin_wrf461_oracle) grades.
//
// Transcendentals are WOOF's own float32 routines: gfk_exp for EXP and gfk_pow
// for every REAL**REAL (the shared glibc_flt32.cuh header this module is
// assembled with).  REAL**INTEGER with a run-time exponent (ggamma's
// TEMP**K1) is the repeated-squaring product the referee's gfortran lowers it
// to (an out-of-line __powisf2 call, pinned by the oracle's probe), and
// constant integer powers are the multiply chains gfortran expands.  SQRT is
// correctly rounded on both sides.  The constant powers gfortran folds at
// compile time (0.01**(1-bv_r), 0.01**(1-bv_i), di50**bv_i) are the folded
// words, read off the referee build.
//
// Native WRF 4.6.1 and 4.7.1 formulas are preserved, including terms
// computed but unused in the native module. The pristine oracle is authoritative.
// WRF's other choices are kept as written: pracs is computed then zeroed,
// rain freezing (pgfr) runs only where snow is also present, sedimentation
// clips snow and ice but not rain, and theiz keeps its pre-sedimentation ice.

#ifndef SBU_KMAX
#define SBU_KMAX 64
#endif

// gfortran's AMAX1/AMIN1 at the referee flags, pinned by the oracle probe:
// MAX returns its SECOND argument on a tie (amax1(+0,-0) = -0) and MIN its
// FIRST (amin1(+0,-0) = +0).  fmaxf/fminf are free to pick either zero.
__device__ __forceinline__ float sbu_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float sbu_min(float a, float b) { return (b < a) ? b : a; }

// REAL**INTEGER with a run-time exponent: repeated squaring, each product
// rounded to float, as the out-of-line __powisf2 the referee calls.
__device__ __forceinline__ float sbu_powi(float x, int m)
{
    unsigned int n = m < 0 ? 0u - (unsigned int)m : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = x * x;
        if (n % 2u) y = y * x;
    }
    return m < 0 ? __fdiv_rn(1.0f, y) : y;
}

// ggamma (:1713-1740): the scheme's own 8-term polynomial gamma.
__device__ float sbu_ggamma(float x)
{
    const float b[8] = {-.577191652f, .988205891f, -.897056937f, .918206857f,
                        -.756704078f, .482199394f, -.193527818f, .035868343f};
    float pf = 1.0f;
    float temp = x;
    for (int j = 1; j <= 200; ++j) {
        if (temp <= 2.0f) break;
        temp = temp - 1.0f;
        pf = pf * temp;
    }
    float g1to2 = 1.0f;
    temp = temp - 1.0f;
    for (int k1 = 1; k1 <= 8; ++k1)
        g1to2 = g1to2 + b[k1 - 1] * sbu_powi(temp, k1);
    return pf * g1to2;
}

// parama1/parama2 (:1605-1657): Bergeron growth-rate tables, linear in T.
__device__ float sbu_parama(float temp, bool second)
{
    const float a1[32] = {
        0.100e-10f, 0.7939e-7f, 0.7841e-6f, 0.3369e-5f, 0.4336e-5f,
        0.5285e-5f, 0.3728e-5f, 0.1852e-5f, 0.2991e-6f, 0.4248e-6f,
        0.7434e-6f, 0.1812e-5f, 0.4394e-5f, 0.9145e-5f, 0.1725e-4f,
        0.3348e-4f, 0.1725e-4f, 0.9175e-5f, 0.4412e-5f, 0.2252e-5f,
        0.9115e-6f, 0.4876e-6f, 0.3473e-6f, 0.4758e-6f, 0.6306e-6f,
        0.8573e-6f, 0.7868e-6f, 0.7192e-6f, 0.6513e-6f, 0.5956e-6f,
        0.5333e-6f, 0.4834e-6f};
    const float a2[32] = {
        0.0100f, 0.4006f, 0.4831f, 0.5320f, 0.5307f, 0.5319f, 0.5249f,
        0.4888f, 0.3849f, 0.4047f, 0.4318f, 0.4771f, 0.5183f, 0.5463f,
        0.5651f, 0.5813f, 0.5655f, 0.5478f, 0.5203f, 0.4906f, 0.4447f,
        0.4126f, 0.3960f, 0.4149f, 0.4320f, 0.4506f, 0.4483f, 0.4460f,
        0.4433f, 0.4413f, 0.4382f, 0.4361f};
    const float* a = second ? a2 : a1;
    int i1 = (int)(-temp) + 1;
    int i1p1 = i1 + 1;
    float ratio = -(temp) - (float)(i1 - 1);
    return a[i1 - 1] + ratio * (a[i1p1 - 1] - a[i1 - 1]);
}

// Module parameters (:22-54).
#define SBU_XNOR 8.0e6f
#define SBU_NT_C 10.e6f
#define SBU_RVAPOR 461.5f
#define SBU_RAIR 287.04f
#define SBU_CP 1004.0f
#define SBU_GRAV 9.81f
#define SBU_RHOWATER 1000.0f
#define SBU_RHOSNOW 100.0f
#define SBU_SVP1 0.6112f
#define SBU_SVP2 17.67f
#define SBU_SVP3 29.65f
#define SBU_SVPT0 273.15f
#define SBU_XLS 2.834e6f
#define SBU_XLV 2.5e6f
#define SBU_XLF (SBU_XLS - SBU_XLV)
#define SBU_QI0 1.0e-3f
#define SBU_XMI50 4.8e-10f
#define SBU_XMI40 2.46e-10f
#define SBU_XNI0 1.0e-2f
#define SBU_XMNIN 1.05e-18f
#define SBU_BNI 0.5f
#define SBU_DI50 1.0e-4f
#define SBU_XMI 4.19e-13f
#define SBU_BV_R 0.8f
#define SBU_AVISC 1.49628e-6f
#define SBU_ADIFFWV 8.7602e-5f
#define SBU_AXKA 1.4132e3f
#define SBU_CW 4.187e3f
#define SBU_QVMIN 1.e-20f
// clphy1d_ylin's own parameters (:311-321).
#define SBU_VF1S 0.65f
#define SBU_VF2S 0.44f
#define SBU_VF1R 0.78f
#define SBU_VF2R 0.31f
#define SBU_BEST_A 1.08f
#define SBU_BEST_B 0.499f
// sqrho = 1.0 at every level (:153, note 5: no density factor on fall speed).
#define SBU_SQRHO 1.0f

struct SbuConst {
    float ep2, pi, pio4, pio6, ocp, oxlf, xlvocp, xlfocp, oxmi, cwoxlf;
    float av_r, av_i, di50_bv_i, o6, episp0k, mu_c;
    float gambp4, gambp3, gambp6, gambp5o2, bp3, bp5, bp6;
};

__device__ SbuConst sbu_constants()
{
    SbuConst c;
    c.ep2 = __fdiv_rn(SBU_RAIR, SBU_RVAPOR);
    c.pi = __int_as_float(0x40490FDB);            // acos(-1.), folded
    c.pio4 = __fdiv_rn(c.pi, 4.0f);
    c.pio6 = __fdiv_rn(c.pi, 6.0f);
    c.ocp = __fdiv_rn(1.0f, SBU_CP);
    c.oxlf = __fdiv_rn(1.0f, SBU_XLF);
    c.xlvocp = __fdiv_rn(SBU_XLV, SBU_CP);
    c.xlfocp = __fdiv_rn(SBU_XLF, SBU_CP);
    c.oxmi = __fdiv_rn(1.0f, SBU_XMI);
    c.cwoxlf = __fdiv_rn(SBU_CW, SBU_XLF);
    c.av_r = __int_as_float(0x44527FCA);          // 2115.0*0.01**(1-bv_r)
    c.av_i = __int_as_float(0x409AC117);          // 152.93*0.01**(1-bv_i)
    c.di50_bv_i = __int_as_float(0x3DCCCCCD);     // di50**bv_i
    c.o6 = __fdiv_rn(1.0f, 6.0f);
    c.episp0k = ((1.0f * c.ep2) * 1000.0f) * SBU_SVP1;
    c.mu_c = fminf(15.0f, __fdiv_rn(1000.e6f, SBU_NT_C) + 2.0f);
    c.gambp4 = sbu_ggamma(SBU_BV_R + 4.0f);
    c.gambp3 = sbu_ggamma(SBU_BV_R + 3.0f);
    c.gambp6 = sbu_ggamma(SBU_BV_R + 6.0f);
    c.gambp5o2 = sbu_ggamma(__fdiv_rn(SBU_BV_R + 5.0f, 2.0f));
    c.bp3 = SBU_BV_R + 3.0f;
    c.bp5 = SBU_BV_R + 5.0f;
    c.bp6 = SBU_BV_R + 6.0f;
    return c;
}

// Saturation vapour mixing ratios as clphy1d writes them (:409-417).
__device__ __forceinline__ float sbu_qsw(float tem, float temcc, float prez,
                                         const SbuConst& c)
{
    float es = (1000.0f * SBU_SVP1)
        * gfk_exp(__fdiv_rn(SBU_SVP2 * temcc, tem - SBU_SVP3));
    return __fdiv_rn(c.ep2 * es, prez - es);
}

__device__ __forceinline__ float sbu_qsi_raw(float tem, float prez,
                                             const SbuConst& c)
{
    float es = (1000.0f * SBU_SVP1)
        * gfk_exp(__fdiv_rn(21.8745584f * (tem - 273.16f), tem - 7.66f));
    return __fdiv_rn(c.ep2 * es, prez - es);
}

// satadj (:1468-1601) for one level.
__device__ void sbu_satadj(float& qvz, float& qlz, float& qiz, float prez,
                           float theiz, float& thz, float tothz,
                           const SbuConst& c)
{
    thz = theiz - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
    float tem = tothz * thz;
    float qsat;
    if (tem > 273.15f) {
        float es = (1000.0f * SBU_SVP1) * gfk_exp(
            __fdiv_rn(SBU_SVP2 * (tem - SBU_SVPT0), tem - SBU_SVP3));
        qsat = __fdiv_rn(c.ep2 * es, prez - es);
    } else {
        qsat = __fdiv_rn(c.episp0k, prez) * gfk_exp(
            __fdiv_rn(21.8745584f * (tem - 273.15f), tem - 7.66f));
    }
    float qpz = (qvz + qlz) + qiz;
    if (qpz < qsat) {
        qvz = qpz;
        qiz = 0.0f;
        qlz = 0.0f;
        return;
    }
    float qlpqi = qlz + qiz;
    float ratql, ratqi;
    if (qlpqi >= 1.0e-5f) {
        ratql = __fdiv_rn(qlz, qlpqi);
        ratqi = __fdiv_rn(qiz, qlpqi);
    } else {
        const float t0 = 273.15f;
        const float t1 = 248.15f;
        float tmp1 = __fdiv_rn(t0 - tem, t0 - t1);
        tmp1 = sbu_min(1.0f, tmp1);
        tmp1 = sbu_max(0.0f, tmp1);
        ratqi = tmp1;
        ratql = 1.0f - tmp1;
    }
    float tsat = tem;
    float absft = 1.0f;
    float qswz = 0.0f, qsiz = 0.0f, qvsbar = 0.0f;
    for (int n = 1; n <= 20; ++n) {
        float denom1 = __fdiv_rn(1.0f, tsat - SBU_SVP3);
        float denom2 = __fdiv_rn(1.0f, tsat - 7.66f);
        float es = (1000.0f * SBU_SVP1)
            * gfk_exp((SBU_SVP2 * denom1) * (tsat - SBU_SVPT0));
        qswz = __fdiv_rn(c.ep2 * es, prez - es);
        if (tem < 273.15f) {
            es = (1000.0f * SBU_SVP1)
                * gfk_exp((21.8745584f * denom2) * (tsat - 273.15f));
            qsiz = __fdiv_rn(c.ep2 * es, prez - es);
            if (tem < 233.15f) qswz = qsiz;
        } else {
            qsiz = qswz;
        }
        qvsbar = ratql * qswz + ratqi * qsiz;
        if (absft < 0.01f) break;
        float dqvsbar =
            ((((ratql * qswz) * SBU_SVP2) * 243.5f) * denom1) * denom1
            + ((((ratqi * qsiz) * 21.8745584f) * 265.5f) * denom2) * denom2;
        float lat = c.xlvocp + ratqi * c.xlfocp;
        float ftsat = ((tsat + lat * qvsbar) - tothz * theiz)
            - (c.xlfocp * ratqi) * ((qvz + qlz) + qiz);
        float dftsat = 1.0f + lat * dqvsbar;
        tsat = tsat - __fdiv_rn(ftsat, dftsat);
        absft = fabsf(ftsat);
    }
    if (qpz > qvsbar) {
        qvz = qvsbar;
        qiz = ratqi * (qpz - qvz);
        qlz = ratql * (qpz - qvz);
    } else {
        qvz = qpz;
        qiz = 0.0f;
        qlz = 0.0f;
    }
}

extern "C" __global__ void sbu_ylin_column(
        float* __restrict__ th, float* __restrict__ qv,
        float* __restrict__ ql, float* __restrict__ qr,
        float* __restrict__ qi, float* __restrict__ qs,
        float* __restrict__ rimi,
        const float* __restrict__ rho, const float* __restrict__ pii,
        const float* __restrict__ p, const float* __restrict__ zmass,
        const float* __restrict__ dz8w, const float* __restrict__ ht,
        float* __restrict__ rainnc, float* __restrict__ rainncv,
        float* __restrict__ sr,
        float dt, int nz, int ny, int nx)
{
    const int ncol = ny * nx;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol || nz < 1 || nz > SBU_KMAX) return;
#define AT(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)col]

    const SbuConst c = sbu_constants();
    const float dtb = dt;
    const float odtb = __fdiv_rn(1.0f, dtb);
    const float zsfc = ht[col];
    const int kte = nz - 1;            // WRF kte, zero-based

    // Per-level state WRF keeps in clphy1d's local arrays across the
    // sedimentation and process phases.  theiz must be stored: it is formed
    // from the PRE-sedimentation cloud ice (:426) and read after it.
    float theiz[SBU_KMAX];
    float vt[SBU_KMAX];
    float am_s[SBU_KMAX], n0_s[SBU_KMAX], av_s[SBU_KMAX], bv_s[SBU_KMAX];
    float tmp_ss[SBU_KMAX], tmp_sa[SBU_KMAX], aa_s[SBU_KMAX], cap_s[SBU_KMAX];

    // ---- :400-427 entry clipping, saturation ratios, theiz --------------
    for (int k = 0; k <= kte; ++k) {
        float q;
        // The scalar referee lowers MAX with a constant +0 clamp so its
        // zero ties select +0. The two-variable probe has a different tie.
        q = AT(ql, k) > 0.0f ? AT(ql, k) : 0.0f; AT(ql, k) = q;
        q = AT(qi, k) > 0.0f ? AT(qi, k) : 0.0f; AT(qi, k) = q;
        q = sbu_max(SBU_QVMIN, AT(qv, k)); AT(qv, k) = q;
        q = AT(qs, k) > 0.0f ? AT(qs, k) : 0.0f; AT(qs, k) = q;
        q = AT(qr, k) > 0.0f ? AT(qr, k) : 0.0f; AT(qr, k) = q;
        float tothz = AT(pii, k);
        theiz[k] = AT(th, k)
            + __fdiv_rn(c.xlvocp * AT(qv, k) - c.xlfocp * AT(qi, k), tothz);
    }

    // ---- :485-544 transport properties, riming intensity, snow laws -----
    // Ri (:500-507), forced non-decreasing downward from its column maximum
    // (:511-516: MAXLOC is the FIRST maximum), clamped and stored as RIMI.
    for (int k = 0; k <= kte; ++k) {
        float rhok = AT(rho, k);
        float qlk = AT(ql, k), qsk = AT(qs, k);
        float ri = 0.0f;
        if (rhok * qlk > 1e-5f && rhok * qsk > 1e-5f)
            ri = __fdiv_rn(1.0f, 1.0f + __fdiv_rn(6e-5f,
                 (gfk_pow(rhok, 1.170f) * qlk) * gfk_pow(qsk, 0.170f)));
        AT(rimi, k) = ri;
    }
    {
        int max_ri_k = 0;
        float max_ri = AT(rimi, 0);
        for (int k = 1; k <= kte; ++k) {
            float r = AT(rimi, k);
            if (r > max_ri) { max_ri = r; max_ri_k = k; }
        }
        for (int k = 0; k <= max_ri_k; ++k) AT(rimi, k) = max_ri;
    }
    for (int k = 0; k <= kte; ++k) {
        float ri = sbu_max(0.0f, sbu_min(AT(rimi, k), 1.0f));
        AT(rimi, k) = ri;
        float rhok = AT(rho, k);
        float tem = AT(th, k) * AT(pii, k);
        float viscmu = __fdiv_rn(SBU_AVISC * gfk_pow(tem, 1.5f), tem + 120.0f);
        cap_s[k] = 0.25f * (1.0f + ri);
        float tc0 = sbu_min(-0.1f, tem - 273.15f);
        n0_s[k] = sbu_min(2.0e8f, 2.0e6f * gfk_exp(-(0.12f * tc0)));
        float am = (0.004f + 6e-5f * tc0) + (0.15f * ri) * ri;
        am = sbu_max(0.000023f, am);
        float bm = (1.85f + 0.003f * tc0) + 1.25f * ri;
        bm = sbu_min(bm, 3.0f);
        am = gfk_pow(10.0f, 2.0f * bm - 3.0f) * am;
        float aa = (1.28f + (-0.012f) * tc0) + (-0.6f) * ri;
        float ba = (1.5f + 0.0075f * tc0) + 0.5f * ri;
        aa = gfk_pow(1e-2f, 2.0f - ba) * aa;
        av_s[k] = (SBU_BEST_A * viscmu) * gfk_pow(
            __fdiv_rn(__fdiv_rn(__fdiv_rn((2.0f * SBU_GRAV) * am, rhok), aa),
                      viscmu * viscmu),
            SBU_BEST_B);
        bv_s[k] = SBU_BEST_B * ((bm - ba) + 2.0f) - 1.0f;
        tmp_ss[k] = (bm + 0.0f) + 1.0f;
        tmp_sa[k] = (ba + 0.0f) + 1.0f;
        am_s[k] = am;
        aa_s[k] = aa;
    }

    // ---- :558-741 sedimentation: rain, snow, cloud ice -------------------
    float pptrain = 0.0f, pptsnow = 0.0f, pptice = 0.0f;
    for (int species = 0; species < 3; ++species) {
        float* q = species == 0 ? qr : (species == 1 ? qs : qi);
        float t_del_tv = 0.0f;
        float del_tv = dtb;
        bool notlast = true;
        while (notlast) {
            int min_q = kte;
            int max_q = -1;
            for (int k = 0; k <= kte - 1; ++k) {
                float qk = AT(q, k);
                if (qk > 1.0e-8f) {
                    min_q = min(min_q, k);
                    max_q = max(max_q, k);
                    float rhok = AT(rho, k);
                    float v;
                    if (species == 0) {
                        float tmp1 = __fsqrt_rn(__fdiv_rn(__fdiv_rn(
                            (c.pi * SBU_RHOWATER) * SBU_XNOR, rhok), qk));
                        tmp1 = __fsqrt_rn(tmp1);
                        v = __fdiv_rn(((c.o6 * c.av_r) * c.gambp4) * SBU_SQRHO,
                                      gfk_pow(tmp1, SBU_BV_R));
                    } else if (species == 1) {
                        float orho = __fdiv_rn(1.0f, rhok);
                        float gss = sbu_ggamma(tmp_ss[k]);
                        float tmp1 = gfk_pow(
                            __fdiv_rn(((am_s[k] * n0_s[k]) * gss) * orho, qk),
                            __fdiv_rn(1.0f, tmp_ss[k]));
                        v = __fdiv_rn(__fdiv_rn(
                                (SBU_SQRHO * av_s[k]) * sbu_ggamma(bv_s[k] + tmp_ss[k]),
                                gss),
                            gfk_pow(tmp1, bv_s[k]));
                    } else {
                        v = 3.29f * gfk_pow(rhok * qk, 0.16f);
                    }
                    vt[k] = v;
                    float dzk = (k == 0) ? AT(zmass, k) - zsfc
                                         : AT(zmass, k) - AT(zmass, k - 1);
                    del_tv = sbu_min(del_tv, __fdiv_rn(0.9f * dzk, v));
                } else {
                    vt[k] = 0.0f;
                }
            }
            if (max_q >= min_q) {
                t_del_tv = t_del_tv + del_tv;
                if (t_del_tv >= dtb) {
                    notlast = false;
                    del_tv = (dtb + del_tv) - t_del_tv;
                }
                float fluxin = 0.0f;
                for (int k = max_q; k >= min_q; --k) {
                    float rhok = AT(rho, k);
                    float fluxout = (rhok * vt[k]) * AT(q, k);
                    float flux = __fdiv_rn(__fdiv_rn(fluxin - fluxout, rhok),
                                           AT(dz8w, k));
                    float qn = AT(q, k) + del_tv * flux;
                    if (species != 0) qn = sbu_max(0.0f, qn);
                    AT(q, k) = qn;
                    fluxin = fluxout;
                }
                if (min_q == 0) {
                    float add = fluxin * del_tv;
                    if (species == 0) pptrain = pptrain + add;
                    else if (species == 1) pptsnow = pptsnow + add;
                    else pptice = pptice + add;
                } else {
                    int kb = min_q - 1;
                    AT(q, kb) = AT(q, kb) + __fdiv_rn(__fdiv_rn(
                        del_tv * fluxin, AT(rho, kb)), AT(dz8w, kb));
                }
            } else {
                notlast = false;
            }
        }
    }

    // ---- :747-1444 microphysical processes, level by level ---------------
    for (int k = 0; k <= kte; ++k) {
        float qvz = AT(qv, k), qlz = AT(ql, k), qrz = AT(qr, k);
        float qiz = AT(qi, k), qsz = AT(qs, k);
        float thz = AT(th, k);
        const float tothz = AT(pii, k);
        const float rhok = AT(rho, k);
        const float orho = __fdiv_rn(1.0f, rhok);
        const float prez = AT(p, k);
        const float oprez = __fdiv_rn(1.0f, prez);

        // Quantities clphy1d fixed before sedimentation (:400-427, :485-492);
        // sedimentation moves no theta or vapour, so they recompute exactly.
        float tem = thz * tothz;
        float temcc = tem - 273.15f;
        float qswz = sbu_qsw(tem, temcc, prez, c);
        float qsiz;
        if (tem < 273.15f) {
            qsiz = sbu_qsi_raw(tem, prez, c);
            if (temcc < -40.0f) qswz = qsiz;
        } else {
            qsiz = qswz;
        }
        const float qvoqswz = __fdiv_rn(qvz, qswz);
        const float qvoqsiz = __fdiv_rn(qvz, qsiz);
        const float viscmu = __fdiv_rn(SBU_AVISC * gfk_pow(tem, 1.5f), tem + 120.0f);
        const float visc = viscmu * orho;
        const float diffwv = (SBU_ADIFFWV * gfk_pow(tem, 1.81f)) * oprez;
        const float schmidt = __fdiv_rn(visc, diffwv);
        const float xka = SBU_AXKA * viscmu;
        const float rs0 = __fdiv_rn((c.ep2 * 1000.0f) * SBU_SVP1,
                                    prez - 1000.0f * SBU_SVP1);

        const float qvzodt = sbu_max(0.0f, odtb * qvz);
        const float qlzodt = sbu_max(0.0f, odtb * qlz);
        const float qizodt = sbu_max(0.0f, odtb * qiz);
        const float qszodt = sbu_max(0.0f, odtb * qsz);
        const float qrzodt = sbu_max(0.0f, odtb * qrz);

        float tmp = ((qiz + qlz) + qsz) + qrz;
        if ((qvz + qlz) + qiz < qsiz && tmp == 0.0f) continue;   // :773

        float psaut = 0.f, psfw = 0.f, psfi = 0.f, praci = 0.f, piacr = 0.f;
        float psaci = 0.f, psacw = 0.f, psdep = 0.f, pssub = 0.f, pracs = 0.f;
        float psacr = 0.f, psmlt = 0.f, psmltevp = 0.f, praut = 0.f;
        float pracw = 0.f, prevp = 0.f, pgfr = 0.f;
        float pihom = 0.f, pimlt = 0.f, pidw = 0.f;

        // :778-822 rain and snow slopes and fall speeds.
        float vtr, olambdar;
        if (qrz > 1.0e-8f) {
            float tmp1 = __fsqrt_rn(__fdiv_rn(
                ((c.pi * SBU_RHOWATER) * SBU_XNOR) * orho, qrz));
            float xlambdar = __fsqrt_rn(tmp1);
            olambdar = __fdiv_rn(1.0f, xlambdar);
            vtr = (((c.o6 * c.av_r) * c.gambp4) * SBU_SQRHO)
                * gfk_pow(olambdar, SBU_BV_R);
        } else {
            vtr = 0.0f;
            olambdar = 0.0f;
        }
        float vts, olambdas;
        if (qsz > 1.0e-8f) {
            float gss = sbu_ggamma(tmp_ss[k]);
            float tmp1 = gfk_pow(
                __fdiv_rn(((am_s[k] * n0_s[k]) * gss) * orho, qsz),
                __fdiv_rn(1.0f, tmp_ss[k]));
            olambdas = __fdiv_rn(1.0f, tmp1);
            vts = __fdiv_rn(__fdiv_rn(
                    (SBU_SQRHO * av_s[k]) * sbu_ggamma(bv_s[k] + tmp_ss[k]), gss),
                gfk_pow(tmp1, bv_s[k]));
        } else {
            vts = 0.0f;
            olambdas = 0.0f;
        }

        // The ventilated snow growth factor shared by Psdep, Psmlt and
        // Psmltevp (:952-958, :1053-1056, :1078-1081).
        auto snow_vent = [&]() -> float {
            float tmp1 = __fdiv_rn(
                (av_s[k] * SBU_SQRHO)
                    * gfk_pow(olambdas, (5.0f + bv_s[k]) + 0.0f),
                visc);
            return (SBU_VF1S * olambdas) * olambdas
                + ((SBU_VF2S * gfk_pow(schmidt, 0.33334f))
                   * sbu_ggamma((2.5f + 0.5f * bv_s[k]) + 0.0f))
                  * __fsqrt_rn(tmp1);
        };
        // Native snow collection coefficient (:931-932, :1029-1030).
        auto snow_collect = [&]() -> float {
            float e = bv_s[k] + tmp_sa[k];
            return ((aa_s[k] * SBU_SQRHO) * n0_s[k])
                * sbu_ggamma(e) * gfk_pow(olambdas, e);
        };

        if (tem < 273.15f) {
            // (1) Psaut :837-847
            float alpha1 = 1.0e-3f * gfk_exp(0.025f * temcc);
            float qic;
            if (temcc < -20.0f) {
                float tmp1 = -7.6f + 4.0f * gfk_exp(-(0.2443e-3f
                    * gfk_pow(fabsf(temcc) - 20.0f, 2.455f)));
                qic = (1.0e-3f * gfk_exp(tmp1)) * orho;
            } else {
                qic = SBU_QI0;
            }
            float tmp1 = (odtb * (qiz - qic)) * (1.0f - gfk_exp(-(alpha1 * dtb)));
            psaut = sbu_max(0.0f, tmp1);

            // (2)(3) Bergeron Psfw/Psfi :858-897
            if (qlz > 1.0e-10f) {
                float temc1 = sbu_max(-30.99f, temcc);
                float a1 = sbu_parama(temc1, false);
                float a2 = sbu_parama(temc1, true);
                float t1 = 1.0f - a2;
                a1 = a1 * gfk_pow(0.001f, t1);
                float odtberg = __fdiv_rn(a1 * t1,
                    gfk_pow(SBU_XMI50, t1) - gfk_pow(SBU_XMI40, t1));
                float vti50 = (c.av_i * c.di50_bv_i) * SBU_SQRHO;
                const float eiw = 1.0f;
                float save1 = a1 * gfk_pow(SBU_XMI50, a2);
                float save2 = (((((0.25f * c.pi) * eiw) * rhok) * SBU_DI50)
                               * SBU_DI50) * vti50;
                float tmp2 = save1 + save2 * qlz;
                float xni50mx = __fdiv_rn(qlzodt, tmp2);
                float xni50 = __fdiv_rn(
                    qiz * (1.0f - gfk_exp(-(dtb * odtberg))), SBU_XMI50);
                xni50 = sbu_min(xni50, xni50mx);
                float tmp3 = __fdiv_rn(odtb * tmp2, save2)
                    * (1.0f - gfk_exp(-((save2 * xni50) * dtb)));
                psfw = sbu_min(tmp3, qlzodt);
                float tmp1b = xni50 * SBU_XMI50 - psfw;
                psfi = sbu_min(tmp1b, qizodt);
            }

            if (qrz > 0.0f) {
                // (4) Praci :908-911, (5) Piacr :916-918
                const float eri = 1.0f;
                float save1 = (((c.pio4 * eri) * SBU_XNOR) * c.av_r) * SBU_SQRHO;
                float t1 = (save1 * c.gambp3) * gfk_pow(olambdar, c.bp3);
                praci = qizodt * (1.0f - gfk_exp(-(t1 * dtb)));
                float t2 = ((((((qiz * save1) * rhok) * c.pio6) * SBU_RHOWATER)
                             * c.gambp6) * c.oxmi) * gfk_pow(olambdar, c.bp6);
                piacr = sbu_min(t2, qrzodt);
            }

            if (qsz > 0.0f) {
                // (6) Psaci, (7) Psacw :930-942
                float esi = gfk_exp(0.025f * temcc);
                float save1 = snow_collect();
                float t1 = esi * save1;
                psaci = qizodt * (1.0f - gfk_exp(-(t1 * dtb)));
                const float esw = 1.0f;
                t1 = esw * save1;
                psacw = qlzodt * (1.0f - gfk_exp(-(t1 * dtb)));

                // (8) Psdep/Pssub :948-969
                float tmpa = ((SBU_RVAPOR * xka) * tem) * tem;
                float tmpb = (((SBU_XLS * SBU_XLS) * rhok) * qsiz) * diffwv;
                float tmpc = (tmpa * qsiz) * diffwv;
                float abi = __fdiv_rn(
                    (((4.0f * c.pi) * cap_s[k]) * (qvoqsiz - 1.0f)) * tmpc,
                    tmpa + tmpb);
                float t2 = (abi * n0_s[k]) * snow_vent();
                float t3 = odtb * (qvz - qsiz);
                if (t2 <= 0.0f) {
                    t2 = sbu_max(t2, t3);
                    pssub = sbu_max(t2, -qszodt);
                    psdep = 0.0f;
                } else {
                    psdep = sbu_min(t2, t3);
                    pssub = 0.0f;
                }

                if (qrz > 0.0f) {
                    // (9) Pracs (computed, then zeroed by WRF), (10) Psacr
                    const float esr = 1.0f;
                    float ta = olambdar * olambdar;
                    float tb = olambdas * olambdas;
                    float tc = olambdar * olambdas;
                    float t1b = (((((c.pi * c.pi) * esr) * SBU_XNOR) * n0_s[k])
                                 * fabsf(vtr - vts)) * orho;
                    pracs = 0.0f;
                    float t3b = ((ta * ta) * olambdas)
                        * ((5.0f * ta + 2.0f * tc) + 0.5f * tb);
                    float t4 = (t1b * SBU_RHOWATER) * t3b;
                    psacr = sbu_min(t4, qrzodt);
                    // Pgfr :1002-1011
                    if (qrz > 1.e-8f) {
                        const float bp = 100.0f;
                        const float ap = 0.66f;
                        float o3 = (olambdar * olambdar) * olambdar;
                        float t2b = ((((((((20.0f * c.pi) * c.pi) * bp) * SBU_XNOR)
                                       * SBU_RHOWATER) * orho)
                                      * (gfk_exp(-(ap * temcc)) - 1.0f))
                                     * o3) * o3 * olambdar;
                        pgfr = sbu_min(t2b, qrzodt);
                    } else {
                        pgfr = 0.0f;
                    }
                }
            }
        } else if (qsz > 0.0f) {
            // T >= 0 C.  (1) Psacw :1027-1033
            const float esw = 1.0f;
            float save1 = snow_collect();
            float t1 = esw * save1;
            psacw = qlzodt * (1.0f - gfk_exp(-(t1 * dtb)));
            // (2) Psacr :1038-1045
            const float esr = 1.0f;
            float ta = olambdar * olambdar;
            float tb = olambdas * olambdas;
            float tc = olambdar * olambdas;
            float t1b = (((((c.pi * c.pi) * esr) * SBU_XNOR) * n0_s[k])
                         * fabsf(vtr - vts)) * orho;
            float t2 = ((ta * ta) * olambdas) * ((5.0f * ta + 2.0f * tc) + 0.5f * tb);
            float t3 = (t1b * SBU_RHOWATER) * t2;
            psacr = sbu_min(t3, qrzodt);
            // (3) Psmlt :1050-1059
            float delrs = rs0 - qvz;
            float term1 = ((2.0f * c.pi) * orho)
                * (((SBU_XLV * diffwv) * rhok) * delrs - xka * temcc);
            float vent = snow_vent();
            float t2m = n0_s[k] * vent;
            float t3m = (term1 * c.oxlf) * t2m - (c.cwoxlf * temcc) * (psacw + psacr);
            float t4m = sbu_min(0.0f, t3m);
            psmlt = sbu_max(t4m, -qszodt);
            // (4) Native Psmltevp :1065-1084: abr is computed but unused.
            float tmpa = ((SBU_RVAPOR * xka) * tem) * tem;
            float tmpb = (((SBU_XLV * SBU_XLV) * rhok) * qswz) * diffwv;
            float tmpc = (tmpa * qswz) * diffwv;
            float tmpd = sbu_min(0.0f, ((qvoqswz - 0.90f) * qswz) * odtb);
            float abr = __fdiv_rn(((2.0f * c.pi) * (qvoqswz - 0.90f)) * tmpc,
                                  tmpa + tmpb);
            float t2e = n0_s[k] * vent;
            float t3e = sbu_min(0.0f, t2e);
            t3e = sbu_max(t3e, tmpd);
            psmltevp = sbu_max(t3e, -qszodt);
        }

        // ---- :1102-1146 rain production --------------------------------
        if (qlz > 1e-6f) {
            float lamc = gfk_pow(
                __fdiv_rn(__fdiv_rn(((SBU_NT_C * SBU_RHOWATER) * c.pi)
                                        * sbu_ggamma(4.0f + c.mu_c),
                                    (6.0f * rhok) * qlz),
                          sbu_ggamma(1.0f + c.mu_c)),
                0.3333f);
            float dc_liu = __fdiv_rn(
                gfk_pow(__fdiv_rn(sbu_ggamma(7.0f + c.mu_c),
                                  sbu_ggamma(1.0f + c.mu_c)),
                        __fdiv_rn(1.0f, 6.0f)),
                lamc);
            if (dc_liu > 10.0e-6f) {
                float disp = __fdiv_rn(1.0f, c.mu_c + 1.0f);
                float x = __fdiv_rn(__fdiv_rn(0.75f, c.pi), 1e-3f * SBU_RHOWATER);
                float eta = ((x * x) * 1.9e11f)
                    * __fdiv_rn(__fdiv_rn(((1.0f + 3.0f * disp) * (1.0f + 4.0f * disp))
                                              * (1.0f + 5.0f * disp),
                                          1.0f + disp),
                                1.0f + 2.0f * disp);
                float y = (1e-3f * rhok) * qlz;
                praut = __fdiv_rn(eta * ((y * y) * y), 1e-6f * SBU_NT_C);
                praut = __fdiv_rn(praut, 1e-3f * rhok);
            }
        }
        {
            const float erw = 1.0f;
            float t1 = (((((c.pio4 * erw) * SBU_XNOR) * c.av_r) * SBU_SQRHO)
                        * c.gambp3) * gfk_pow(olambdar, c.bp3);
            pracw = qlzodt * (1.0f - gfk_exp(-(t1 * dtb)));

            float tmpa = ((SBU_RVAPOR * xka) * tem) * tem;
            float tmpb = (((SBU_XLV * SBU_XLV) * rhok) * qswz) * diffwv;
            float tmpc = (tmpa * qswz) * diffwv;
            float tmpd = sbu_min(0.0f, ((qvoqswz - 0.90f) * qswz) * odtb);
            float abr = __fdiv_rn(((2.0f * c.pi) * (qvoqswz - 0.90f)) * tmpc,
                                  tmpa + tmpb);
            float t1v = __fdiv_rn((c.av_r * SBU_SQRHO) * gfk_pow(olambdar, c.bp5),
                                  visc);
            float t2 = (abr * SBU_XNOR) * ((SBU_VF1R * olambdar) * olambdar
                + ((SBU_VF2R * gfk_pow(schmidt, 0.33334f)) * c.gambp5o2)
                  * __fsqrt_rn(t1v));
            float t3 = sbu_min(0.0f, t2);
            t3 = sbu_max(t3, tmpd);
            prevp = sbu_max(t3, -qrzodt);
        }

        // ---- :1156-1305 limit and sum all processes --------------------
        float qvsbar;
        float theiz_k = theiz[k];
        if (temcc < 0.0f) {
            if (psdep > qvzodt) {
                float factor = __fdiv_rn(qvzodt, psdep);
                psdep = psdep * factor;
            }
            tmp = ((praut + psacw) + psfw) + pracw;
            if (tmp > qlzodt) {
                float factor = __fdiv_rn(qlzodt, tmp);
                praut = praut * factor;
                psacw = psacw * factor;
                psfw = psfw * factor;
                pracw = pracw * factor;
            }
            tmp = ((psaut + psaci) + praci) + psfi;
            if (tmp > qizodt) {
                float factor = __fdiv_rn(qizodt, tmp);
                psaut = psaut * factor;
                psaci = psaci * factor;
                praci = praci * factor;
                psfi = psfi * factor;
            }
            float tmp_r = ((((piacr + psacr) - prevp) - praut) - pracw) + pgfr;
            if (tmp_r > qrzodt) {
                float factor = __fdiv_rn(qrzodt, tmp_r);
                piacr = piacr * factor;
                psacr = psacr * factor;
                prevp = prevp * factor;
                pgfr = pgfr * factor;
            }
            float tmp_s = (-pssub) - ((((((((((psaut + psaci) + psacw) + psfw)
                + pgfr) + psfi) + praci) + piacr) + psdep) + psacr) - pracs);
            if (tmp_s > qszodt) {
                float factor = __fdiv_rn(qszodt, tmp_s);
                pssub = pssub * factor;
                pracs = pracs * factor;
            }

            float pvapor = ((-pssub) - psdep) - prevp;
            qvz = sbu_max(SBU_QVMIN, qvz + dtb * pvapor);
            float pclw = (((-praut) - pracw) - psacw) - psfw;
            qlz = sbu_max(0.0f, qlz + dtb * pclw);
            float pcli = (((-psaut) - psfi) - psaci) - praci;
            qiz = sbu_max(0.0f, qiz + dtb * pcli);
            tmp_r = (((((piacr + psacr) - prevp) - praut) - pracw) + pgfr) - pracs;
            float prain = -tmp_r;
            qrz = sbu_max(0.0f, qrz + dtb * prain);
            tmp_s = (-pssub) - ((((((((((psaut + psaci) + psacw) + psfw)
                + pgfr) + psfi) + praci) + piacr) + psdep) + psacr) - pracs);
            float psnow = -tmp_s;
            qsz = sbu_max(0.0f, qsz + dtb * psnow);
            float qschg = psnow;
            tmp = (__fdiv_rn(c.ocp, tothz) * SBU_XLF) * qschg;
            theiz_k = theiz_k + dtb * tmp;
            thz = theiz_k - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
            tem = thz * tothz;
            temcc = tem - 273.15f;
            if (temcc < -40.0f) qswz = qsiz;
            float qlpqi = qlz + qiz;
            if (qlpqi == 0.0f)
                qvsbar = qsiz;
            else
                qvsbar = __fdiv_rn(qiz * qsiz + qlz * qswz, qlpqi);
        } else {
            tmp = (praut + psacw) + pracw;
            if (tmp > qlzodt) {
                float factor = __fdiv_rn(qlzodt, tmp);
                praut = praut * factor;
                psacw = psacw * factor;
                pracw = pracw * factor;
            }
            float tmp_s = -(psmlt + psmltevp);
            if (tmp_s > qszodt) {
                float factor = __fdiv_rn(qszodt, tmp_s);
                psmlt = psmlt * factor;
                psmltevp = psmltevp * factor;
            }
            float tmp_r = (-prevp) - (((praut + pracw) + psacw) - psmlt);
            if (tmp_r > qrzodt) {
                float factor = __fdiv_rn(qrzodt, tmp_r);
                prevp = prevp * factor;
            }
            float pvapor = (-psmltevp) - prevp;
            qvz = sbu_max(SBU_QVMIN, qvz + dtb * pvapor);
            float pclw = ((-praut) - pracw) - psacw;
            qlz = sbu_max(0.0f, qlz + dtb * pclw);
            const float pcli = 0.0f;
            qiz = sbu_max(0.0f, qiz + dtb * pcli);
            tmp_r = (-prevp) - (((praut + pracw) + psacw) - psmlt);
            float prain = -tmp_r;
            qrz = sbu_max(0.0f, qrz + dtb * prain);
            tmp_s = -(psmlt + psmltevp);
            float psnow = -tmp_s;
            qsz = sbu_max(0.0f, qsz + dtb * psnow);
            float qschg = psnow;
            tmp = (__fdiv_rn(c.ocp, tothz) * SBU_XLF) * qschg;
            theiz_k = theiz_k + dtb * tmp;
            thz = theiz_k - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
            tem = thz * tothz;
            temcc = tem - 273.15f;
            qswz = sbu_qsw(tem, temcc, prez, c);
            qsiz = qswz;
            qvsbar = qswz;
        }

        // ---- :1317-1373 saturation adjustment --------------------------
        const float rsat = 1.0f;
        bool done = false;
        if ((qvz + qlz) + qiz < rsat * qvsbar) {
            qvz = (qvz + qlz) + qiz;
            qlz = 0.0f;
            qiz = 0.0f;
            thz = theiz_k - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
            done = true;
        } else {
            sbu_satadj(qvz, qlz, qiz, prez, theiz_k, thz, tothz, c);
            thz = theiz_k - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
            tem = thz * tothz;
            temcc = tem - 273.15f;
        }

        // ---- :1379-1436 melting and freezing of cloud ice and water ----
        if (!done && qlz + qiz > 0.0f) {
            if (temcc < -40.0f) pihom = qlz * odtb;
            if (temcc > 0.0f) pimlt = qiz * odtb;
            if (temcc < 0.0f && temcc > -31.0f) {
                float a1 = sbu_parama(temcc, false);
                float a2 = sbu_parama(temcc, true);
                a1 = a1 * gfk_pow(0.001f, 1.0f - a2);
                float xnin = SBU_XNI0 * gfk_exp(-(SBU_BNI * temcc));
                pidw = (xnin * orho) * (a1 * gfk_pow(SBU_XMNIN, a2));
            }
            qlz = sbu_max(0.0f, qlz + dtb * (((-pihom) + pimlt) - pidw));
            qiz = sbu_max(0.0f, qiz + dtb * ((pihom - pimlt) + pidw));
            sbu_satadj(qvz, qlz, qiz, prez, theiz_k, thz, tothz, c);
            thz = theiz_k - __fdiv_rn(c.xlvocp * qvz - c.xlfocp * qiz, tothz);
        }

        AT(qv, k) = qvz;
        AT(ql, k) = qlz;
        AT(qr, k) = qrz;
        AT(qi, k) = qiz;
        AT(qs, k) = qsz;
        AT(th, k) = thz;
    }

    // ---- :1449-1455 native vapour floor, preserving assignment order
    for (int k = 1; k <= kte; ++k) {
        float qvz = AT(qv, k);
        if (qvz < SBU_QVMIN) {
            AT(ql, k) = 0.0f;
            AT(qi, k) = 0.0f;
            AT(qv, k) = sbu_max(SBU_QVMIN, (qvz + AT(ql, k)) + AT(qi, k));
        }
    }

    // ---- :176-181 surface precipitation, mm per call and accumulated ----
    rainncv[col] = (pptrain + pptsnow) + pptice;
    rainnc[col] = ((rainnc[col] + pptrain) + pptsnow) + pptice;
    // Native driver has no SBU-YLin SR argument; retain its zero value.
    sr[col] = 0.0f;
#undef AT
}


// In-file saturation polynomials, WRF module_mp_sbu_ylin.F:1663-1709.
// These functions are exported by WRF but not called by its column driver.
__device__ float sbu_rsif(float p, float t)
{
    float x = sbu_max(-80.0f, t - 273.16f);
    float e = .609868993e3f + x*(.499320233e2f + x*(.184672631e1f
        + x*(.402737184e-1f + x*(.565392987e-3f + x*(.521693933e-5f
        + x*(.307839583e-7f + x*(.105785160e-9f + x*.161444444e-12f)))))));
    return __fdiv_rn(.622f*e, p-e);
}

__device__ float sbu_rslf(float p, float t)
{
    float x = sbu_max(-80.0f, t - 273.16f);
    float e = .611583699e3f + x*(.444606896e2f + x*(.143177157e1f
        + x*(.264224321e-1f + x*(.299291081e-3f + x*(.203154182e-5f
        + x*(.702620698e-8f + x*(.379534310e-11f - x*.321582393e-13f)))))));
    return __fdiv_rn(.622f*e, p-e);
}

extern "C" __global__ void sbu_ylin_helpers(const float* inputs, float* outputs, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = inputs[3*i], t = inputs[3*i+1], x = inputs[3*i+2];
    outputs[3*i] = sbu_rsif(p,t);
    outputs[3*i+1] = sbu_rslf(p,t);
    outputs[3*i+2] = sbu_ggamma(x);
}
