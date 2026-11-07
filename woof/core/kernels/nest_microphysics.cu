// Nondefault NSSL receiving-domain distributions. Defaults keep their literals.
#ifdef NSSL2_ALPHA_G_WORD
#define NSSL_EDGE_ALPHA_G (__uint_as_float((unsigned int)(NSSL2_ALPHA_G_WORD-1ULL)))
#define NSSL_EDGE_ALPHA_H (__uint_as_float((unsigned int)(NSSL2_ALPHA_H_WORD-1ULL)))
__device__ __forceinline__ float nssl_edge_zfactor(float a) {
    return ((6.f+a)*(5.f+a)*(4.f+a))/((3.f+a)*(2.f+a)*(1.f+a));
}
#endif
// GPUWM-specific Thompson MP8 -> NSSL-2 MP18 nest translation.
//
// This is not stock-WRF equivalence: WRF v4.6.1 normalizes every domain to
// the innermost microphysics selector.  The diagnosed values reproduce the
// admitted mass-only NSSL calcnfromq closure in nssl2.cu, but emit one field at
// a time so nest forcing reuses the existing full-parent scratch allocation.

// Preserve the established default entry points and constant arithmetic.
// A nondefault target compiles a separate variant carrying its REAL CCN seed.
#ifdef NSSL2_VARIABLE_CCCN
#define NSSL_CCN_PARAMETER , float nssl_background_ccn
#define NSSL_CCN_ARGUMENT , nssl_background_ccn
#else
#define NSSL_CCN_PARAMETER
#define NSSL_CCN_ARGUMENT
#endif

struct NsslTransitionState {
    float qv, qc, qr, qi, qs, qg;
    float qh, qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh;
};

static __device__ __forceinline__
NsslTransitionState diagnose_mp8_mass_as_mp18(
    float rho, float qv, float qc, float qr, float qi, float qs, float qg
    NSSL_CCN_PARAMETER)
{
    NsslTransitionState result;
    const double density_inverse = 1.0 / (double)rho;
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;

    float vapor = qv;
    float cloud = qc;
    float rain = qr;
    float ice = qi;
    float snow = qs;
    float graupel = qg;
    float cloud_number = 0.0f;
    float rain_number = 0.0f;
    float ice_number = 0.0f;
    float snow_number = 0.0f;
    float graupel_number = 0.0f;
#ifdef NSSL2_VARIABLE_CCCN
    const float qccn = nssl_background_ccn;
#else
    const float qccn = 408163264.0f;
#endif
    float ccn_number = (float)((double)qccn * (double)rho);
    float graupel_volume = 0.0f;

    if (cloud_number <= cxmin && cloud > qxmin_init) {
        const float cwmas_inverse = 327479132160.0f;
        cloud_number = fminf(qccn, cloud * cwmas_inverse) * rho;
        ccn_number -= cloud_number;
    } else if (cloud <= qxmin_cloud
               || (cloud_number <= cxmin && cloud <= qxmin_init)) {
        vapor += cloud;
        cloud_number = 0.0f;
        cloud = 0.0f;
    }

    if (ice_number <= cxmin && ice > qxmin_init) {
        const float xims = 4.7123910329460728e-10f;
        ice_number = __fdiv_rn(rho * ice, xims);
    } else if (ice <= qxmin_cloud
               || (ice_number <= cxmin && ice <= qxmin_init)) {
        vapor += ice;
        ice_number = 0.0f;
        ice = 0.0f;
    }

    if (rain_number <= 0.1f * cxmin && rain > qxmin_init) {
        const float zrfac = 3.9788734806922577e-11f;
        const double lambda_inverse = pow(
            (double)rho * (double)rain * (double)zrfac, 0.25);
        const double n1 = lambda_inverse * (double)8000000.0f;
        rain_number = (float)(n1 * (double)20.0f / (double)20.0f);
    } else if (rain <= qxmin_rain
               || (rain_number <= cxmin && rain <= qxmin_init)) {
        vapor += rain;
        rain_number = 0.0f;
        rain = 0.0f;
    }

    if (snow_number <= 0.1f * cxmin && snow > qxmin_init) {
        const float zsfac = 1.0610329281846020e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)snow * (double)zsfac, 0.25);
        const double n1 = lambda_inverse * (double)3000000.0f;
        snow_number = (float)(
            n1 * (double)6.0000004768371582f / (double)20.0f);
    } else if (snow <= qxmin_cloud
               || (snow_number <= cxmin && snow <= qxmin_init)) {
        vapor += snow;
        snow_number = 0.0f;
        snow = 0.0f;
    }

    if (graupel_number <= 0.1f * cxmin && graupel > qxmin_init) {
        graupel_volume = __fdiv_rn(graupel, 700.0f);
        const float zhfac = 2.2736419413860176e-9f;
        const float xgms = 9.8960235561662557e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)graupel * (double)zhfac, 0.25);
        const double intercept_number =
            lambda_inverse * (double)200000.0f
#ifdef NSSL2_ALPHA_G_WORD
            * (double)nssl_edge_zfactor(NSSL_EDGE_ALPHA_G) / (double)20.0f
#endif
            ;
        const double maximum_number =
            (double)rho * (double)graupel / (double)xgms;
        const double diagnosed = fmin(intercept_number, maximum_number);
        if (diagnosed > (double)cxmin) {
            graupel_number = (float)diagnosed;
        } else {
            graupel = 0.0f;
            graupel_number = 0.0f;
            graupel_volume = 0.0f;
        }
    } else if (graupel <= qxmin_rain
               || (graupel_number <= cxmin && graupel <= qxmin_init)) {
        vapor += graupel;
        graupel = 0.0f;
    }

    result.qv = vapor;
    result.qc = cloud;
    result.qr = rain;
    result.qi = ice;
    result.qs = snow;
    result.qg = graupel;
    result.qh = 0.0f;
    result.qndrop = (float)((double)cloud_number * density_inverse);
    result.qnr = (float)((double)rain_number * density_inverse);
    result.qni = (float)((double)ice_number * density_inverse);
    result.qns = (float)((double)snow_number * density_inverse);
    result.qng = (float)((double)graupel_number * density_inverse);
    result.qnh = 0.0f;
    result.qnn = (float)((double)ccn_number * density_inverse);
    result.qvolg = graupel_volume;
    result.qvolh = 0.0f;
    return result;
}

static __device__ __forceinline__
float transition_field(const NsslTransitionState& value, int field)
{
    if (field == 0) return value.qv;
    if (field == 1) return value.qc;
    if (field == 2) return value.qr;
    if (field == 3) return value.qi;
    if (field == 4) return value.qs;
    if (field == 5) return value.qg;
    if (field == 6) return value.qh;
    if (field == 7) return value.qndrop;
    if (field == 8) return value.qnr;
    if (field == 9) return value.qni;
    if (field == 10) return value.qns;
    if (field == 11) return value.qng;
    if (field == 12) return value.qnh;
    if (field == 13) return value.qnn;
    if (field == 14) return value.qvolg;
    return value.qvolh;
}

extern "C" __global__
void mp8_to_mp18_mass_diagnosed_field(
    const float* __restrict__ alt,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ mub2d,
    const float* __restrict__ mup,
    const float* __restrict__ c1h,
    const float* __restrict__ c2h,
    float* __restrict__ out,
    int field, int coupled, int nz, int ny, int nx NSSL_CCN_PARAMETER)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nz * ny * nx) return;
    const int k = idx / (ny * nx);
    const int rem = idx - k * ny * nx;
    const int j = rem / nx;
    const int i = rem - j * nx;
    const NsslTransitionState diagnosed = diagnose_mp8_mass_as_mp18(
        1.0f / alt[idx], qv[idx], qc[idx], qr[idx], qi[idx], qs[idx], qg[idx]
        NSSL_CCN_ARGUMENT);
    float value = transition_field(diagnosed, field);
    if (coupled != 0) {
        const size_t q = (size_t)j * nx + i;
        const float base = __fadd_rn(__fmul_rn(c1h[k], mub2d[q]), c2h[k]);
        const float perturbation = __fmul_rn(c1h[k], mup[q]);
        const float hybrid_mass = __fadd_rn(base, perturbation);
        value = __fmul_rn(hybrid_mass, value);
    }
    out[idx] = value;
}

// General ArWen MP edge matrix.  This entry point is never used by the
// bit-specified MP8->MP18 edge above.  Source NUMBER moments are
// deliberately absent from the API: every destination moment is
// reconstructed from destination mass using the destination scheme's own
// entry closure.  The one deliberate exception is P3's rime pair
// (source_qir/source_qib): those are MASS/VOLUME prognostics, not numbers,
// and leaving mp=50 they are consumed by the ice split -- dropping them
// would drop the only record of where the single category's snow/graupel
// boundary lies.  They are read only when source_mp == 50; every other
// source passes a placeholder the code never dereferences into meaning.

static __device__ __forceinline__
float edge_source_mass(
    int code, int idx,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh)
{
    if (code == 0) return qv[idx];
    if (code == 1) return qc[idx];
    if (code == 2) return qr[idx];
    if (code == 3) return qi[idx];
    if (code == 4) return qs[idx];
    if (code == 5) return qg[idx];
    if (code == 6) return qh[idx];
    return 0.0f;
}

static __device__ __forceinline__
NsslTransitionState diagnose_edge_mass_as_mp18(
    float rho, float qv, float qc, float qr, float qi, float qs,
    float qg, float qh NSSL_CCN_PARAMETER)
{
    NsslTransitionState result = diagnose_mp8_mass_as_mp18(
        rho, qv, qc, qr, qi, qs, qg NSSL_CCN_ARGUMENT);
    const double density_inverse = 1.0 / (double)rho;
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_rain = 1.0e-12f;
    float vapor = result.qv;
    float hail = qh;
    float hail_number = 0.0f;
    float hail_volume = 0.0f;

    // NSSL calcnfromq, module_mp_nssl_2mom.F:5428-5467.  The 900 kg/m3
    // volume default and alphahl=1 number diagnosis exactly match
    // nssl2_initial_state in nssl2.cu.
    if (hail_number <= 0.1f * cxmin && hail > qxmin_init) {
        hail_volume = __fdiv_rn(hail, 900.0f);
        const float zhlfac = 8.8419414012719244e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)hail * (double)zhlfac, 0.25);
        const double n1 = lambda_inverse * (double)40000.0f;
        hail_number = (float)(
            n1 *
#ifdef NSSL2_ALPHA_G_WORD
            (double)nssl_edge_zfactor(NSSL_EDGE_ALPHA_H)
#else
            (double)8.75f
#endif
            / (double)20.0f);
    } else if (hail <= qxmin_rain
               || (hail_number <= cxmin && hail <= qxmin_init)) {
        vapor += hail;
        hail = 0.0f;
    }
    result.qv = vapor;
    result.qh = hail;
    result.qnh = (float)((double)hail_number * density_inverse);
    result.qvolh = hail_volume;
    return result;
}

static __device__ __forceinline__
float thompson_edge_rain_number(float qr)
{
    if (qr <= 1.0e-12f) return 0.0f;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float lambda = 3.672f / 1.0e-3f;
    return __fdiv_rn((1.0f / 6.0f) * qr, am_r)
        * lambda * lambda * lambda;
}

static __device__ __forceinline__
float thompson_edge_ice_number(float qi, float rho)
{
    if (qi <= 1.0e-12f) return 0.0f;
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    const float ice_mass = qi * rho;
    float lambda = 4.0f / 5.0e-6f;
    float number = fminf(
        999.0e3f,
        __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
            * lambda * lambda * lambda);
    lambda = cbrtf(am_i * 6.0f * number / ice_mass);
    const float diameter = 4.0f / lambda;
    if (diameter < 5.0e-6f) {
        lambda = 4.0f / 5.0e-6f;
        number = fminf(
            999.0e3f,
            __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
                * lambda * lambda * lambda);
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0f / 300.0e-6f;
        number = __fdiv_rn((1.0f / 6.0f) * ice_mass, am_i)
            * lambda * lambda * lambda;
    }
    return number / rho;
}

static __device__ __forceinline__
float morrison_edge_number(float q, int kind, float rimed_density)
{
    // A source-absent N=0 enters Morrison's PSD slope limiter at its lower
    // lambda bound (module_mp_morr_two_moment.F:1525-1638 and
    // morr_bound_one in morrison.cu).  Reconstruct the resulting prognostic
    // number mixing ratio directly.
    if (q < 1.0e-14f) return 0.0f;
    const float pi = 3.14159265358979323846f;
    float six_c;
    float lambda;
    if (kind == 0) {
        six_c = pi * 997.0f;
        lambda = 1.0f / 2800.0e-6f;
    } else if (kind == 1) {
        six_c = 500.0f * pi;
        lambda = 1.0f / 350.0e-6f;
    } else if (kind == 2) {
        six_c = 100.0f * pi;
        lambda = 1.0f / 2000.0e-6f;
    } else {
        six_c = rimed_density * pi;
        lambda = 1.0f / 2000.0e-6f;
    }
    return __fdiv_rn(q * lambda * lambda * lambda, six_c);
}

// P3 (mp_physics=50) edge closure.  WRF defines neither direction of a P3
// mixed nest edge, so both arms below execute the DEFINED, documented ArWen
// closure whose constants live beside their citations in
// microphysics_transition.py (the P3_EDGE_* constants).  Every float
// operation uses a round-to-nearest intrinsic so the arithmetic mirrors the
// CPU references p3_edge_entry_reference/p3_edge_exit_reference bitwise --
// plain expressions would let NVRTC contract mul+add into FMA and the GPU
// shard's equivalence tests compare exact.

// calc_bulkRhoRime, module_mp_p3.F:6784-6830: P3's own admission function
// for a (qitot, qirim, birim) triple.  Running it on every edge output is
// what makes qirim <= qitot and the [50, 900] kg/m3 density bound hold by
// construction.
static __device__ __forceinline__
float p3_edge_bulk_rho_rime(float qi_tot, float* qi_rim, float* bi_rim)
{
    float rho_rime = 0.0f;
    if (*bi_rim >= 1.0e-15f) {
        rho_rime = __fdiv_rn(*qi_rim, *bi_rim);
        if (rho_rime < 50.0f) {
            rho_rime = 50.0f;
            *bi_rim = __fdiv_rn(*qi_rim, rho_rime);
        } else if (rho_rime > 900.0f) {
            rho_rime = 900.0f;
            *bi_rim = __fdiv_rn(*qi_rim, rho_rime);
        }
    } else {
        *qi_rim = 0.0f;
        *bi_rim = 0.0f;
        rho_rime = 0.0f;
    }
    if (*qi_rim > qi_tot && rho_rime > 0.0f) {
        *qi_rim = qi_tot;
        *bi_rim = __fdiv_rn(*qi_rim, rho_rime);
    }
    if (*qi_rim < 1.0e-14f) {          // qsmall, module_mp_p3.F:230
        *qi_rim = 0.0f;
        *bi_rim = 0.0f;
    }
    return rho_rime;
}

// Entering P3: merge every frozen source species into the single ice
// category (mass-conserving; sub-qsmall ice returns to vapor exactly as
// p3_main's own end-of-step canonicalization does), diagnose the rime pair
// at the two named densities -- fresh snow 100 kg/m3
// (module_mp_morr_two_moment.F:377), dense rime 400 kg/m3 (:378-382, the
// IHAIL=0 graupel arm; hail merges at the same dense constant, a
// documented divergence) -- and diagnose nr/ni with P3's own entry-closure
// constants (get_rain_dsd2's lammin reconstruction :6705-6780; mi0 :242
// capped by max_total_Ni :186 per impose_max_total_Ni :6833-6855).
static __device__ __forceinline__
float p3_edge_field(
    int field, float inv_rho,
    float qv, float qc, float qr, float qi, float qs, float qg, float qh)
{
    const float qsmall = 1.0e-14f;
    float vapor = qv;
    const float frozen_dense = __fadd_rn(qg, qh);
    float qitot = __fadd_rn(__fadd_rn(qi, qs), frozen_dense);
    float qirim = __fadd_rn(qs, frozen_dense);
    float birim = __fadd_rn(__fdiv_rn(qs, 100.0f),
                            __fdiv_rn(frozen_dense, 400.0f));
    if (qitot < qsmall) {
        vapor = __fadd_rn(vapor, qitot);
        qitot = 0.0f;
        qirim = 0.0f;
        birim = 0.0f;
    }
    p3_edge_bulk_rho_rime(qitot, &qirim, &birim);
    if (field == 0) return vapor;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qitot;
    if (field == 6) {
        if (qr < qsmall) return 0.0f;
        // P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS: lammin^3/(6*cons1) in the
        // float32 chain p3_init uses (lammin = 1/0.002 rounds to
        // 499.99997f, so the value is 39788.727f, not 500^3/(6*cons1)).
        return __fmul_rn(qr, 39788.727f);
    }
    if (field == 7) {
        if (qitot < qsmall) return 0.0f;
        const float mi0 = 3.7699116e-15f;        // module_mp_p3.F:242
        const float max_total_ni = 2000.0e3f;    // module_mp_p3.F:186
        return fminf(__fdiv_rn(qitot, mi0),
                     __fmul_rn(max_total_ni, inv_rho));
    }
    if (field == 20) return qirim;
    if (field == 21) return birim;
    return 0.0f;
}

// Milbrandt-Yau (mp_physics=9) edge closure.  The target moments are
// diagnosed with MY2's OWN mass-to-number closure -- the consistency block
// mp_milbrandt2mom_main runs on entry, module_mp_milbrandt2mom.F:1459-1528,
// transcribed in gpuwm/core/kernels/milbrandt2.cu:547-600 -- so an MY2
// child entered across a scheme boundary holds exactly the moments the
// scheme itself would have built from those masses at its first call, and
// the block then finds nothing left to fix.  That is the same criterion
// mp=18's arm meets with NSSL's calcnfromq and mp=8's and mp=10's meet with
// their own PSD closures; nothing here is a new intercept.
//
// The closure needs the scheme's own de = pres/(Rd*T) (:3400) and, for ice
// and snow, temperature -- Cooper's N(T) and Thompson's Nos(T) -- which is
// why microphysics_edge_field carries a temperature and a pressure plane
// and the constant vector ck.  Every ck index below is the one
// milbrandt2.cu names, read from the same host-built vector
// (gpuwm.core.milbrandt2_constants.ck_vector), so the two translation
// units cannot disagree about a constant.
#define MY2E_icmr  ck[16]
#define MY2E_icmg  ck[20]
#define MY2E_icmh  ck[22]
#define MY2E_dms   ck[25]
#define MY2E_icms  ck[26]
#define MY2E_N_c_SM ck[50]
#define MY2E_GR31  ck[54]
#define MY2E_iGR34 ck[59]
#define MY2E_icexs2 ck[93]
#define MY2E_GS31  ck[98]
#define MY2E_iGS40 ck[107]
#define MY2E_GG31  ck[118]
#define MY2E_iGG34 ck[123]
#define MY2E_GH31  ck[136]
#define MY2E_iGH34 ck[140]

#define MY2E_epsQ      1.0e-14f
#define MY2E_epsN      1.0e-3f
#define MY2E_TRPL      0.27316e+3f
#define MY2E_RGASD     0.28705e+3f
#define MY2E_alpha_r   0.0f
#define MY2E_alpha_s   0.0f
#define MY2E_alpha_g   0.0f
#define MY2E_alpha_h   0.0f
#define MY2E_No_r_SM   1.0e+7f
#define MY2E_No_g_SM   4.0e+6f
#define MY2E_No_h_SM   1.0e+5f

// :3513-3518 and :3520-3525, IEEE expf as in milbrandt2.cu.
static __device__ __forceinline__ float my2_edge_N_Cooper(float T)
{
    return 5.0f * expf(0.304f * (MY2E_TRPL - fmaxf(233.0f, T)));
}

static __device__ __forceinline__ float my2_edge_Nos_Thompson(float T)
{
    return fminf(2.0e+8f,
                 2.0e+6f * expf(-0.12f * fminf(-0.001f, T - MY2E_TRPL)));
}

static __device__ __forceinline__
float my2_edge_field(
    int field, float t, float pr, const float* __restrict__ ck,
    float qv, float qc, float qr, float qi, float qs, float qg, float qh)
{
    if (field == 0) return qv;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qi;
    if (field == 4) return qs;
    if (field == 5) return qg;
    if (field == 10) return qh;

    // de and the #/m3 <-> #/kg conversion are the scheme's own (:1224-1233,
    // :3400): the closure below produces a per-VOLUME number and the state
    // carries per-MASS, so every branch divides by de at the end.
    const float de = pr / (MY2E_RGASD * t);
    float number = 0.0f;
    if (field == 22) {                      // nc, MY2's droplet number
        // :1459-1461.  N_c_SM is the CCNtype=2 continental value the WRF
        // wrapper pins (:3615), read from the same constant vector.
        if (qc > MY2E_epsQ) number = MY2E_N_c_SM;
    } else if (field == 6) {                // nr
        if (qr > MY2E_epsQ) {               // :1470-1473
            number = powf(MY2E_No_r_SM * MY2E_GR31,
                          3.0f / (4.0f + MY2E_alpha_r))
                     * powf(MY2E_GR31 * MY2E_iGR34 * de * qr * MY2E_icmr,
                            (1.0f + MY2E_alpha_r) / (4.0f + MY2E_alpha_r));
        }
    } else if (field == 7) {                // ni
        if (qi > MY2E_epsQ) {               // :1482-1484
            number = fmaxf(2.0f * MY2E_epsN, my2_edge_N_Cooper(t));
        }
    } else if (field == 8) {                // ns
        if (qs > MY2E_epsQ) {               // :1494-1498
            const float No_s = my2_edge_Nos_Thompson(t);
            number = powf(No_s * MY2E_GS31, MY2E_dms * MY2E_icexs2)
                     * powf(MY2E_GS31 * MY2E_iGS40 * MY2E_icms * de * qs,
                            (1.0f + MY2E_alpha_s) * MY2E_icexs2);
        }
    } else if (field == 9) {                // ng
        if (qg > MY2E_epsQ) {               // :1507-1510
            number = powf(MY2E_No_g_SM * MY2E_GG31,
                          3.0f / (4.0f + MY2E_alpha_g))
                     * powf(MY2E_GG31 * MY2E_iGG34 * de * qg * MY2E_icmg,
                            (1.0f + MY2E_alpha_g) / (4.0f + MY2E_alpha_g));
        }
    } else if (field == 23) {               // nh
        if (qh > MY2E_epsQ) {               // :1519-1522
            number = powf(MY2E_No_h_SM * MY2E_GH31,
                          3.0f / (4.0f + MY2E_alpha_h))
                     * powf(MY2E_GH31 * MY2E_iGH34 * de * qh * MY2E_icmh,
                            (1.0f + MY2E_alpha_h) / (4.0f + MY2E_alpha_h));
        }
    }
    return number / de;
}

static __device__ __forceinline__
float edge_plain_or_moment_field(
    int field, int target_mp, float rho, float morr_rhog, float wdm6_ccn,
    float qv, float qc, float qr, float qi, float qs, float qg, float qh)
{
    // Stable host field codes (decoded in full beside _EDGE_FIELD_CODES in
    // gpuwm/core/microphysics_transition.py -- the two must agree):
    // qv/qc/qr/qi/qs/qg=0..5, nr/ni/ns/ng=6..9, qh=10,
    // qndrop/qnr/qni/qns/qng/qnh/qnn/qvolg/qvolh=11..19,
    // qir/qib=20/21 (mp_physics=50), nc=22, nh=23 (mp_physics=9),
    // nn=24 (mp_physics=16), nwfa=25, nifa=26 (mp_physics=28).
    if (field == 0) return qv;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qi;
    if (field == 4) return qs;
    if (field == 5) return qg;
    if (field == 10) return qh;
    if (target_mp == 8) {
        if (field == 6) return thompson_edge_rain_number(qr);
        return thompson_edge_ice_number(qi, rho);
    }
    if (target_mp == 10) {
        if (field == 6) return morrison_edge_number(qr, 0, morr_rhog);
        if (field == 7) return morrison_edge_number(qi, 1, morr_rhog);
        if (field == 8) return morrison_edge_number(qs, 2, morr_rhog);
        return morrison_edge_number(qg, 3, morr_rhog);
    }
    if (target_mp == 16) {
        // WDM6 entry: the reservoir takes the CHILD domain's ccn_conc --
        // the value WRF's flow_dep_bdy_qnn pushes through an inflow face
        // and the value ingest/microphysics_cold_start.py writes for a
        // fresh WDM6 domain -- and the two warm-rain numbers enter at
        // zero, which is where WRF's own allocator leaves them.  No mass
        // is consulted: a number diagnosed from mass here would be an
        // ArWen invention, while these three values are ported.
        if (field == 24) return wdm6_ccn;
        return 0.0f;                       // nc (22) and nr (6)
    }
    if (target_mp == 28) {
        // Thompson aerosol-aware entry.  nr/ni run the SAME two closures
        // the ratified mp=8 edge runs, on the same codes.  The three
        // species classic Thompson does not carry take WRF's own
        // non-aerosol-aware fallbacks, module_mp_thompson.F:1248-1255 --
        // the ELSE branch mp_gt_driver takes when is_aerosol_aware is
        // FALSE.  Each constant is per m3 and the state is per kg, so
        // each divides by rho exactly as WRF writes it.
        if (field == 6) return thompson_edge_rain_number(qr);
        if (field == 7) return thompson_edge_ice_number(qi, rho);
        if (field == 22) return __fdiv_rn(100.0e6f, rho);   // Nt_c, :88
        if (field == 25) return __fdiv_rn(11.1e6f, rho);    // :1805
        if (field == 26) return __fdiv_rn(5.0e3f, rho);     // :1806
        return 0.0f;
    }
    return 0.0f;
}

extern "C" __global__
void microphysics_edge_field(
    const float* __restrict__ alt,
    const float* __restrict__ source_qv,
    const float* __restrict__ source_qc,
    const float* __restrict__ source_qr,
    const float* __restrict__ source_qi,
    const float* __restrict__ source_qs,
    const float* __restrict__ source_qg,
    const float* __restrict__ source_qh,
    const float* __restrict__ source_qir,
    const float* __restrict__ source_qib,
    // Milbrandt-Yau's entry closure only: the base and perturbation
    // potential temperature it forms absolute temperature from, the
    // pressure that forms both the Exner factor and the scheme's own
    // density, and the scheme's constant vector.  Every other target reads
    // none of the four and the host passes a placeholder plane, exactly as
    // it does for source_qir/source_qib outside a P3 edge.
    //
    // The temperature is formed HERE, not on the host, because the host has
    // no array to form it into on every route this kernel is launched from:
    // a tile-streamed nest hands the launcher the bounded donor namespace
    // transition_parent_window builds, which owns no scratch arena and whose
    // planes are window-shaped copies.  Reading thb/thp/p through the same
    // index the masses use is the one form that works from both.
    const float* __restrict__ target_thb,
    const float* __restrict__ target_thp,
    const float* __restrict__ target_pres,
    const float* __restrict__ my2_ck,
    const float* __restrict__ mub2d,
    const float* __restrict__ mup,
    const float* __restrict__ c1h,
    const float* __restrict__ c2h,
    float* __restrict__ out,
    int qv_source, int qc_source, int qr_source, int qi_source,
    int qs_source, int qg_source, int qh_source,
    int field, int source_mp, int target_mp, float morr_rhog,
    // MY2 only: the reference pressure and R/cp the Exner factor above is
    // built from, passed rather than spelled here so the edge and
    // gpuwm.core.constants can never disagree by a rounding, and whether
    // target_thb is the (nz,) base profile or a full mass-shaped field.
    float my2_p0, float my2_rcp, int thb_columnar,
    // WDM6 entry only: the CHILD domain's ccn_conc, so the reservoir the
    // edge seeds is the domain's own value and not a constant restated
    // here.  Every other target ignores it.
    float wdm6_ccn,
    int coupled, int nz, int ny, int nx NSSL_CCN_PARAMETER)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nz * ny * nx) return;
    const int k = idx / (ny * nx);
    const int rem = idx - k * ny * nx;
    const int j = rem / nx;
    const int i = rem - j * nx;
    const float rho = 1.0f / alt[idx];
    const float qv = edge_source_mass(
        qv_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qc = edge_source_mass(
        qc_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qr = edge_source_mass(
        qr_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    float qi = edge_source_mass(
        qi_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    float qs = edge_source_mass(
        qs_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    float qg = edge_source_mass(
        qg_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    float qh = edge_source_mass(
        qh_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);

    if (source_mp == 50) {
        // Leaving P3: split the single ice category back into qi/qs/qg by
        // rime state, mass-conserving, before the target scheme's own
        // closure runs on the split masses.  Mirrors
        // p3_edge_exit_reference operation for operation; zero ice and
        // zero rime are exact, adversarial negatives floor at zero, and
        // calc_bulkRhoRime enforces qirim <= qitot and the density bound.
        float qitot = fmaxf(qi, 0.0f);
        float qirim = fmaxf(source_qir[idx], 0.0f);
        float birim = fmaxf(source_qib[idx], 0.0f);
        p3_edge_bulk_rho_rime(qitot, &qirim, &birim);
        float graupel = 0.0f;
        if (qirim > 0.0f) {
            // Invert the entry diagnosis: conserve rime mass AND rime
            // volume against the same two densities (100/400 kg/m3), so
            // qg = (qirim - 100*qib) * 400/300, clamped to [0, qirim].
            const float scale = __fdiv_rn(400.0f, 300.0f);
            graupel = __fmul_rn(
                __fsub_rn(qirim, __fmul_rn(100.0f, birim)), scale);
            graupel = fminf(fmaxf(graupel, 0.0f), qirim);
        }
        qg = graupel;
        qs = __fsub_rn(qirim, graupel);
        qi = __fsub_rn(qitot, qirim);
        qh = 0.0f;
    }

    float value;
    if (target_mp == 18) {
        const NsslTransitionState diagnosed =
            diagnose_edge_mass_as_mp18(rho, qv, qc, qr, qi, qs, qg, qh
                                      NSSL_CCN_ARGUMENT);
        // MP18's local field order differs from the stable matrix codes.
        int nssl_field = field;
        if (field == 10) nssl_field = 6;
        else if (field >= 11) nssl_field = field - 4;
        value = transition_field(diagnosed, nssl_field);
    } else if (target_mp == 50) {
        value = p3_edge_field(
            field, alt[idx], qv, qc, qr, qi, qs, qg, qh);
    } else if (target_mp == 9) {
        // T = (thb + thp) * (p/p0)^(R/cp), the state's own Exner form.
        const float pr = target_pres[idx];
        const float thb = target_thb[thb_columnar != 0 ? k : idx];
        float theta = __fadd_rn(thb, target_thp[idx]);
#ifdef GPUWM_USE_THETA_M
        theta = __fdiv_rn(theta, __fadd_rn(1.0f,
            __fmul_rn(__fdiv_rn(461.6f, 287.0f), source_qv[idx])));
#endif
        const float t = __fmul_rn(
            theta,
            powf(__fdiv_rn(pr, my2_p0), my2_rcp));
        value = my2_edge_field(
            field, t, pr, my2_ck, qv, qc, qr, qi, qs, qg, qh);
    } else {
        value = edge_plain_or_moment_field(
            field, target_mp, rho, morr_rhog, wdm6_ccn,
            qv, qc, qr, qi, qs, qg, qh);
    }
    if (coupled != 0) {
        const size_t q = (size_t)j * nx + i;
        const float base = __fadd_rn(__fmul_rn(c1h[k], mub2d[q]), c2h[k]);
        const float perturbation = __fmul_rn(c1h[k], mup[q]);
        const float hybrid_mass = __fadd_rn(base, perturbation);
        value = __fmul_rn(hybrid_mass, value);
    }
    out[idx] = value;
}
