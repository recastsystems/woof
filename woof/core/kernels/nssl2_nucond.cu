// WRF v4.6.1 NSSL option-18 NUCOND (and QVEXCESS), transliterated.
//
// Numerical authority: phys/module_mp_nssl_2mom.F:6052-6199 (QVEXCESS) and
// :9611-12215 (NUCOND), commit d66e442f, for the resolved option-18 call:
// ipconc=5, rcond=2, irenuc=2, iqcinit=2, imaxsupopt=4, imurain=1 with
// alphar=0, iferwisventr=2, eqtset=1, lss=lccna=lccnuf=lzr=0, no WRF-Chem
// droplet override, every other module switch at its WRF default (values
// read back from WRF's own initialised module by
// tools/nssl2_wrf461_column_oracle/probes).  lccn > 1 with ipconc = 5 whether
// or not CCN is predicted (module_mp_nssl_2mom.F:1657): with nssl_ccn_on=0
// the driver fills the slot with the diagnosed CCN (:2734) and raises
// renucfrac to 1 (:2555-2557), which is all the predicted_ccn = 0 variant
// changes here (the driver skips the Registry store).  Branches the
// switches disable are not carried.
//
// Arithmetic contract, the same as nssl2_fused_gs.cu (the column oracle
// grades it bit for bit against gfortran):
//   * REAL operations are float operations in WRF's evaluation order;
//     every quotient is __fdiv_rn so NVRTC cannot turn a constant divisor
//     into a reciprocal multiply (A146).
//   * REAL**REAL and EXP are WOOF's own float32 words gfk_pow / gfk_exp
//     (glibc_flt32.cuh, prepended by the kernel loader); Sqrt is sqrtf.
//   * MAX/MIN are gfortran's: MAX(a,b) = (b > a) ? b : a, with a constant
//     operand canonicalised to the second place (nssl2_max / nssl2_min).
//   * Folded REAL parameters carry WRF's bits (cwmas20, the 4-micron
//     droplet volume, nssl2_wrf_math.cuh).
//
// One thread per grid point.  NUCOND's gather/scatter is pointwise: the only
// cross-point read is the vertical neighbours of the immutable SSFILT field,
// which nssl2_nucond_supersaturation builds for the whole slab first, as
// WRF does at :9872-9887.  The moment tail (:11732-12206) visits every
// point, gathered or not.

namespace {

constexpr float PI = 3.141592653589793f;          // 0x40490FDB
constexpr float PIINV = 1.0f / 3.141592653589793f; // 0x3EA2F983
constexpr float TFR = 273.15f;
constexpr float TFRH = 233.15f;
constexpr float NCP = 1004.0f;
constexpr float RW = 461.5f;
constexpr float RHO00 = 1.225f;
constexpr float ADVISC0 = 1.832e-05f;
constexpr float ADVISC1 = 1.718e-05f;
constexpr float TKA0 = 2.43e-2f;
constexpr float CAW = 17.2693882f;
constexpr float CBW = 35.86f;
constexpr float FQSAT = 0.002f;
constexpr int NQSAT = 1000001;
constexpr float CXMIN = 1.e-8f;
constexpr float AR = 841.99666f;
constexpr float BR = 0.8f;
#ifdef NSSL2_VARIABLE_CCCN
#else
constexpr float CWCCN = 5.0e8f;
constexpr float QCCN = 408163264.0f;               // 0x4DC2A09E
#endif
constexpr float CCK = 0.6f;
constexpr float MAXLOWTEMPSS = 1.08f;
constexpr float MAXSUPERSAT = 1.9f;
constexpr float SSMXINIT = 0.4f;
constexpr float RESTORECCNFRAC = 1.0f;
constexpr float CCNTIMECONST = 3600.0f;
constexpr float SUPCB = 0.5f;
constexpr float SUPMX = 238.0f;
constexpr float QXMIN_C = 1.0e-13f;
constexpr float QXMIN_R = 1.0e-12f;
constexpr float QXMIN_I = 1.0e-13f;
constexpr float QXMIN_S = 1.0e-13f;
constexpr float QXMIN_H = 1.0e-12f;
constexpr float QXMIN_HL = 1.0e-12f;
constexpr float XDN_C = 1000.0f;
constexpr float XDN_R = 1000.0f;
constexpr float CNOHMN = 1.e-2f;

__device__ __forceinline__ float ventc() { return __int_as_float(0x3F649A4E); }
__device__ __forceinline__ float ventrn() { return __int_as_float(0x3FE9E6C7); }
__device__ __forceinline__ float cpi() { return __int_as_float(0x3A828CC0); }
__device__ __forceinline__ float cwc1() { return __int_as_float(0x3AFA543E); }
__device__ __forceinline__ float c1f3() { return __int_as_float(0x3EAAAAAB); }
__device__ __forceinline__ float ccne0() { return __int_as_float(0x41BFE0CE); }
__device__ __forceinline__ float cnexp() { return __int_as_float(0x3EB13B15); }
__device__ __forceinline__ float cwmasn5() { return __int_as_float(0x2B136148); }
__device__ __forceinline__ float cwmas20() { return __int_as_float(0x2E136148); }
// (4.*pi/3.) and (4.e-6)**3, folded REAL parameters.
__device__ __forceinline__ float four_pi_third() { return __int_as_float(0x40860A92); }
__device__ __forceinline__ float four_micron_cube() { return __int_as_float(0x249392EE); }
__device__ __forceinline__ float xvmn_c() { return __int_as_float(0x241A8A06); }
__device__ __forceinline__ float xvmx_c() { return __int_as_float(0x2B7EAC34); }

__device__ __forceinline__ float nssl2_max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float nssl2_min(float a, float b) { return (b < a) ? b : a; }

__device__ __forceinline__ float fqsati() { return __fdiv_rn(1.0f, FQSAT); }

// ltemq = (temg-163.15)/fqsat+1.5, clamped (:9927).
__device__ __forceinline__ int ltemq_of(float temperature)
{
    int l = (int)(__fdiv_rn(temperature - 163.15f, FQSAT) + 1.5f);
    return min(NQSAT, max(1, l));
}

// ltemq1 = (temp1+dtemp-163.15)*fqsati+1.5, clamped (:10772).
__device__ __forceinline__ int ltemq_fqsati(float temperature)
{
    int l = (int)((temperature - 163.15f) * fqsati() + 1.5f);
    return min(NQSAT, max(1, l));
}

// tabqvs / dtabqvs (module_mp_nssl_2mom.F:1466-1471), evaluated on demand.
__device__ __forceinline__ float temq_of(int l)
{
    return __fadd_rn(163.15f, __fmul_rn((float)(l - 1), FQSAT));
}

__device__ __forceinline__ float tabqvs(int l)
{
    const float temq = temq_of(l);
    return gfk_exp(__fdiv_rn(__fmul_rn(CAW, temq - 273.15f), temq - CBW));
}

__device__ __forceinline__ float dtabqvs(int l)
{
    const float temq = temq_of(l);
    const float t = temq - CBW;
    const float a = __fdiv_rn(__fmul_rn(-CAW, __fadd_rn(-273.15f, temq)), __fmul_rn(t, t));
    const float b = __fdiv_rn(CAW, t);
    return __fmul_rn(__fadd_rn(a, b), tabqvs(l));
}

// QVEXCESS (:6052-6199) for one gathered point.  qcw1 is the caller's
// qx(mgs,lc); only qvex leaves the routine.
__device__ float qvexcess(float qwvp0, float qv0, float qcw1, float pres,
                          float thetap0, float theta0, float pi0, float fcqv1,
                          float felvcp, float ss1, float pk)
{
    const float pqs = __fdiv_rn(380.0f, pres);
    float thetap = thetap0;
    float theta = thetap + theta0;
    float qwvp = qwvp0;
    float qvap = nssl2_max(qwvp0 + qv0, 0.0f);
    float temg = theta * pk;
    float qwv = nssl2_max(qvap, 0.0f);
    float qcw = nssl2_max(qcw1, 0.0f);
    int ltemq = ltemq_of(temg);
    float qvs = pqs * tabqvs(ltemq);
    float qss = (0.01f * ss1 + 1.0f) * qvs;
    for (int itertd = 1; itertd <= 2; ++itertd) {
        float dqcw = 0.0f;
        float dqwv = qwv - qss;
        if (dqwv < 0.0f) {
            if (qcw > -dqwv) {
                dqcw = dqwv;
                dqwv = 0.0f;
            } else {
                dqcw = -qcw;
                dqwv = dqwv + qcw;
            }
            qwvp = qwvp - dqcw;
            qcw = qcw + dqcw;
            thetap = thetap + __fdiv_rn(1.0f, pi0) * (felvcp * dqcw);
        }
        if (dqwv >= 0.0f) {
            const float t = temg - CBW;
            const float dqvcnd = __fdiv_rn(
                dqwv, 1.0f + __fdiv_rn(fcqv1 * qss, t * t));
            dqcw = dqvcnd;
            thetap = thetap + __fdiv_rn(felvcp * dqcw, pi0);
            qwvp = qwvp - dqvcnd;
            qcw = qcw + dqcw;
        }
        theta = thetap + theta0;
        temg = theta * pk;
        qvap = nssl2_max(qwvp + qv0, 0.0f);
        ltemq = ltemq_of(temg);
        qvs = pqs * tabqvs(ltemq);
        qcw = nssl2_max(qcw, 0.0f);
        qwv = nssl2_max(qvap, 0.0f);
        qss = (0.01f * ss1 + 1.0f) * qvs;
    }
    return nssl2_max(qcw - qcw1, 0.0f);
}

// Restoration of evaporated droplets to CCN (:10536-10546, irenuc <= 2,
// lccna = 0, invertccn false).
#ifdef NSSL2_VARIABLE_CCCN
__device__ __forceinline__ float restore_ccn(float ccnc, float rho0, float add, float background_ccn_per_kg)
#else
__device__ __forceinline__ float restore_ccn(float ccnc, float rho0, float add)
#endif
{
#ifdef NSSL2_VARIABLE_CCCN
    return nssl2_max(ccnc, nssl2_min(background_ccn_per_kg * rho0, ccnc + RESTORECCNFRAC * add));
#else
    return nssl2_max(ccnc, nssl2_min(QCCN * rho0, ccnc + RESTORECCNFRAC * add));
#endif
}

}  // namespace

// SSFILT (:9872-9887): t0 = an(lt)*t77 and the percent supersaturation of
// the raw vapour, for every point before NUCOND changes any of them.
extern "C" __global__ void nssl2_nucond_supersaturation(
    const float* __restrict__ full_theta,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ qv,
    float* __restrict__ supersaturation_percent,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    const float temp1 = full_theta[idx] * exner[idx];
    const int ltemq = ltemq_of(temp1);
    const float t00 = __fdiv_rn(380.0f, pressure_pa[idx]);
    const float c1 = t00 * tabqvs(ltemq);
    float ss = 0.0f;
    if (c1 > 0.0f) ss = 100.0f * (__fdiv_rn(qv[idx], c1) - 1.0f);
    supersaturation_percent[idx] = ss;
}

extern "C" __global__ void nssl2_nucond_default(
    float* __restrict__ full_theta,
    const float* __restrict__ air_density,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    const float* __restrict__ w_interface,
    const float* __restrict__ ssfilt,
    float* __restrict__ qv_field,
    float* __restrict__ qc_field,
    float* __restrict__ qr_field,
    float* __restrict__ qi_field,
    float* __restrict__ qs_field,
    float* __restrict__ qndrop,
    float* __restrict__ qnr,
    float* __restrict__ qni,
    float* __restrict__ qns,
    float* __restrict__ qnn,
    float* __restrict__ qg,
    float* __restrict__ qh,
    float* __restrict__ qng,
    float* __restrict__ qnh,
    float* __restrict__ qvolg,
    float* __restrict__ qvolh,
    int has_frozen,
    float dtp,
    int concentration_space,
    int n,
    int nz,
    int horizontal_size,
#ifdef NSSL2_VARIABLE_CCCN
    int predicted_ccn,
    float background_ccn_per_kg,
    float nssl_cccn)
#else
    int predicted_ccn)
#endif
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    const int k0 = idx / horizontal_size;
    const int col = idx - k0 * horizontal_size;
    const int kgs = k0 + 1;                       // WRF's 1-based level

    const float dn = air_density[idx];
    const float pn = pressure_pa[idx];
    const float t77 = exner[idx];
    const float in_scale = concentration_space ? 1.0f : dn;

    // an(...) for this point, NSSL internal units (#/m3 for numbers).
    float an_lt = full_theta[idx];
    float an_lv = qv_field[idx];
    float an_lc = qc_field[idx];
    float an_lr = qr_field[idx];
    float an_li = qi_field[idx];
    float an_ls = qs_field[idx];
    float an_lnc = qndrop[idx] * in_scale;
    float an_lnr = qnr[idx] * in_scale;
    // Ice and snow numbers enter only the tail's sign tests; they are not
    // rescaled, so a Registry-unit caller gets them back untouched.
    float an_lni = qni[idx];
    float an_lns = qns[idx];
    float an_lccn = qnn[idx] * in_scale;

    // ---- gather test (:9922-9947) ----
    const float t0 = an_lt * t77;
    {
        const float pqs1 = __fdiv_rn(380.0f, pn + 0.0f);
        const float temg1 = t0;
        const int ltemq = ltemq_of(temg1);
        const float qvs1 = pqs1 * tabqvs(ltemq);
        const float qss1 = qvs1;
        const bool gathered =
            (temg1 > TFRH || __fdiv_rn(an_lv, qvs1) > MAXLOWTEMPSS)
            && (an_lv > qss1 || an_lc > QXMIN_C || an_lr > QXMIN_R);
        if (gathered) {
            // ---- gather (:9963-10058) ----
            const float qx_lv = an_lv;
            float qc = nssl2_max(an_lc, 0.0f);
            float qr = nssl2_max(an_lr, 0.0f);
            const float theta0 = an_lt;
            float thetap = 0.0f;
            float theta;
            const float qv0 = qx_lv;
            float qwvp = qx_lv - qv0;
            const float pres = pn + 0.0f;
            const float rho0 = dn;
            const float rhoinv = __fdiv_rn(1.0f, rho0);
            const float rhovt = sqrtf(__fdiv_rn(RHO00, rho0));
            const float pi0 = t77 + 0.0f;
            float temg = t0;
            const float pk = t77 + 0.0f;
            const float pqs = __fdiv_rn(380.0f, pres);
            int ltemq = ltemq_of(temg);
            float qvs = pqs * tabqvs(ltemq);
            const float qvap = nssl2_max(qwvp + qv0, 0.0f);
            const float es = 6.1078e2f * tabqvs(ltemq);
            const float qss = qvs;
            float temgx = nssl2_min(temg, 313.15f);
            temgx = nssl2_max(temgx, 233.15f);
            const float felv = 2500837.367f * gfk_pow(
                __fdiv_rn(273.15f, temgx), 0.167f + 3.67e-4f * temgx);
            const float felvcp = felv * cpi();
            const float fcqv1 = (4098.0258f * felv) * cpi();
            const float wvdf = (2.11e-05f * gfk_pow(__fdiv_rn(temg, TFR), 1.94f))
                * __fdiv_rn(101325.0f, 0.0f + pn);
            const float advisc = (ADVISC0 * __fdiv_rn(416.16f, temg + 120.0f))
                * gfk_pow(__fdiv_rn(temg, 296.0f), 1.5f);
            const float tka = __fdiv_rn(TKA0 * advisc, ADVISC1);

            // numbers (:10065-10110); lccn > 1, lccna = lccnuf = 0.
            float cx_lc = nssl2_max(an_lnc, 0.0f);
#ifdef NSSL2_VARIABLE_CCCN
            const float cwnccn = __fdiv_rn(nssl_cccn * rho0, RHO00);
#else
            const float cwnccn = __fdiv_rn(CWCCN * rho0, RHO00);
#endif
            float ccnc = an_lccn;
            float ccna = cwnccn - ccnc;
            float cx_lr = nssl2_max(an_lnr, 0.0f);
            float cn = 0.0f;

            // cnuc (:10113-10129); renucfrac = 0, or 1 with nssl_ccn_on=0.
            const float renucfrac = predicted_ccn ? 0.0f : 1.0f;
            float cnuc = nssl2_max(ccnc, cwnccn) * (1.0f - renucfrac)
                + ccnc * renucfrac;
            if (renucfrac >= 0.999f) {
                if (temg < 265.0f) {
                    if (qc > 10.0f * QXMIN_C && w_interface[idx] > 2.0f) {
                        cnuc = 0.0f;
                    } else {
                        cnuc = 0.1f * cnuc;
                    }
                }
            }

            // w and ssfilt neighbours (:10378-10397).
            const int kp1 = min(nz, kgs + 1);
            const float wvel = 0.5f * (w_interface[(kp1 - 1) * horizontal_size + col]
                                       + w_interface[idx]);
            const float ssat0 = ssfilt[idx];
            const float ssf = ssfilt[idx];
            const float ssfkp1 = ssfilt[(min(nz - 1, kgs + 1) - 1) * horizontal_size + col];
            const float ssfkm1 = ssfilt[(max(1, kgs - 1) - 1) * horizontal_size + col];

            // cloud water (:10407-10436).
            float xmas_lc;
            if (cx_lc > 1.0e6f) {
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              NSSL2_WRF_CWMASN), NSSL2_WRF_CWMASX);
            } else if (qc > QXMIN_C && cx_lc > CXMIN) {
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              XDN_C * xvmn_c()), XDN_C * xvmx_c());
                cx_lc = __fdiv_rn(qc * rho0, xmas_lc);
            } else if (qc > QXMIN_C && cx_lc <= CXMIN) {
                cx_lc = nssl2_max(CXMIN, __fdiv_rn(rho0 * qc, NSSL2_WRF_CWMASX));
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              NSSL2_WRF_CWMASN), NSSL2_WRF_CWMASX);
            } else {
                xmas_lc = NSSL2_WRF_CWMASN;
            }
            float xdia_lc = gfk_pow(xmas_lc * cwc1(), c1f3());

            // rain (:10440-10477); imurain = 1, alpha(lr) = alphar = 0.
            const float alpha_lr = 0.0f;
            float xdia_lr = 1.e-9f;
            if (qr > QXMIN_R) {
                float xv = __fdiv_rn(rho0 * qr, XDN_R * nssl2_max(cx_lr, 1.0e-9f));
                if (xv > NSSL2_WRF_XVRMX0) {
                    xv = NSSL2_WRF_XVRMX0;
                    cx_lr = __fdiv_rn(rho0 * qr, NSSL2_WRF_XVRMX0 * XDN_R);
                } else if (xv < NSSL2_WRF_XVRMN) {
                    xv = NSSL2_WRF_XVRMN;
                    cx_lr = __fdiv_rn(rho0 * qr, NSSL2_WRF_XVRMN * XDN_R);
                }
                xdia_lr = gfk_pow(__fdiv_rn((6.0f * PIINV) * xv,
                                            ((alpha_lr + 3.0f) * (alpha_lr + 2.0f))
                                                * (alpha_lr + 1.0f)),
                                  c1f3());
            }

            // (:10483-10498)
            const float fadvisc = (ADVISC0 * __fdiv_rn(416.16f, temg + 120.0f))
                * gfk_pow(__fdiv_rn(temg, 296.0f), 1.5f);
            const float fakvisc = fadvisc * rhoinv;
            const float fwvdf = (2.11e-05f * gfk_pow(__fdiv_rn(temg, TFR), 1.94f))
                * __fdiv_rn(101325.0f, pres);
            const float fschm = __fdiv_rn(fakvisc, fwvdf);
            const float fvent = gfk_pow(fschm, 1.0f / 3.0f) * gfk_pow(fakvisc, -0.5f);

            // ---- condensation / nucleation (:10507-11637) ----
            float dcloud = 0.0f;
            float dqc = 0.0f;
            bool cycle = (temg <= TFRH && __fdiv_rn(qx_lv, qvs) < MAXLOWTEMPSS);
            if (!cycle) {
                // label selector: 0 = fall to 631, 1 = renucleation block.
                bool renuc_block = false;
                if (ssat0 > 0.0f || ssf > 0.0f) {
                    // 620
                    if (qc > QXMIN_C && cx_lc >= 1.0f) {
                        const float ac1 = __fdiv_rn(felv * felv,
                                                    (tka * RW) * (temg * temg));
                        const float bc = __fdiv_rn(RW * temg, wvdf * es);
                        float d1 = 0.0f, d1r = 0.0f;
                        if (ssf > 0.0f || ssat0 > 0.0f) {
                            if (qc > QXMIN_C) {
                                if (xdia_lc <= 0.0f) {
                                    xmas_lc = NSSL2_WRF_CWMASN;
                                    xdia_lc = gfk_pow(xmas_lc * cwc1(), c1f3());
                                }
                                d1 = __fdiv_rn(1.0f, ac1 + bc) * 4.0f * PI * ventc()
                                    * 0.5f * xdia_lc * cx_lc * rhoinv;
                            } else {
                                d1 = 0.0f;
                            }
                            if (qr > QXMIN_R && cx_lr > 1.e-9f) {
                                const float x = 1.0f + alpha_lr;
                                const float rwvent = 0.78f * x
                                    + 0.308f * ventrn() * fvent * sqrtf(AR * rhovt)
                                        * gfk_pow(xdia_lr, __fdiv_rn(1.0f + BR, 2.0f));
                                d1r = __fdiv_rn(1.0f, ac1 + bc) * 4.0f * PI * rwvent
                                    * 0.5f * xdia_lr * cx_lr * rhoinv;
                            } else {
                                d1r = 0.0f;
                            }
                            const float e1 = __fdiv_rn(felvcp, pi0);
                            const float f1 = pk;
                            float temp1 = temg;
                            const float p380 = __fdiv_rn(380.0f, pres);
                            float ss1 = __fdiv_rn(qx_lv, qvs);
                            float ss2 = ss1;
                            float temp2 = temp1;
                            float qv1 = qx_lv;
                            float qvs1 = qvs;
                            float delta;
                            if (fabsf(ss1 - 1.0f) > 1.e-5f) {
                                delta = __fdiv_rn(0.5f * (qv1 - qvs1), d1 * (ss1 - 1.0f));
                            } else {
                                delta = 0.1f * dtp;
                            }
                            const float dtcon1 = nssl2_min(0.2f * delta, 0.05f);
                            int nc = 2 * (int)lroundf(__fdiv_rn(dtp - 4.0f * dtcon1, delta));
                            nc = max(nc, 5);
                            const float dtcon2 = __fdiv_rn(dtp - 4.0f * dtcon1, (float)nc);
                            int nstep = 1;
                            float dt1 = 0.0f;
                            float dqr = 0.0f;
                            while (dt1 < dtp) {
                                float dtcon = (nstep <= 4) ? dtcon1 : dtcon2;
                                float ss1m;
                                bool exit_loop = false;
                                while (true) {
                                    // 609
                                    const float dqv = -(ss1 - 1.0f) * d1 * dtcon;
                                    const float dqvr = -(ss1 - 1.0f) * d1r * dtcon;
                                    const float dtemp = -0.5f * e1 * f1 * (dqv + dqvr);
                                    const int ltemq1m = ltemq_fqsati(temp1 + dtemp);
                                    const float dqvs = dtemp * p380 * dtabqvs(ltemq1m);
                                    const float qv1m = qv1 + dqv + dqvr;
                                    const float qvs1m = qvs1 + dqvs;
                                    ss1m = __fdiv_rn(qv1m, qvs1m);
                                    // (dqvii + dqvis) == 0 always: no ice here.
                                    if (ss1m < 1.0f) {
                                        dtcon = 0.5f * dtcon;
                                        if (dtcon >= dtcon1) continue;
                                        exit_loop = true;
                                    }
                                    break;
                                }
                                if (exit_loop) break;
                                const float dqv = -(ss1m - 1.0f) * d1 * dtcon;
                                const float dqvr = -(ss1m - 1.0f) * d1r * dtcon;
                                const float dtemp = -e1 * f1 * (dqv + dqvr);
                                const int ltemq1 = ltemq_fqsati(temp1 + dtemp);
                                const float dqvs = dtemp * p380 * dtabqvs(ltemq1);
                                qv1 = qv1 + dqv + dqvr;
                                dqc = dqc - dqv;
                                dqr = dqr - dqvr;
                                qvs1 = qvs1 + dqvs;
                                ss1 = __fdiv_rn(qv1, qvs1);
                                temp1 = temp1 + dtemp;
                                if (temp2 == temp1 || ss2 == ss1 || ss1 == 1.0f
                                        || (nstep > 10 && ss1 < 1.0005f)) {
                                    break;
                                }
                                ss2 = ss1;
                                temp2 = temp1;
                                dt1 = dt1 + dtcon;
                                nstep = nstep + 1;
                            }
                            dcloud = dqc;
                            thetap = thetap + e1 * (dcloud + dqr);
                            qwvp = qwvp - (dcloud + dqr);
                            qc = qc + dcloud;
                            qr = qr + dqr;
                            theta = thetap + theta0;
                            temg = theta * f1;
                            ltemq = ltemq_of(temg);
                            qvs = pqs * tabqvs(ltemq);
                        }
                        // :11023-11033
                        bool go631 = false, go613 = false, at616 = false;
                        if (wvel <= 0.0f) {
                            at616 = true;
                        } else if (cx_lc <= 0.0f) {
                            go613 = true;
                        } else if (kgs > 1 && qc <= QXMIN_C) {
                            go613 = true;
                        } else if (kgs == 1 && wvel > 0.0f) {
                            go613 = true;
                        } else {
                            at616 = true;
                        }
                        if (at616 && !go613) {
                            if (ssf <= SUPCB && wvel > 0.0f) {
                                go631 = true;
                            } else if (kgs > 1 && kgs < nz - 1
                                       && (ssfkp1 >= SUPMX || ssf >= SUPMX
                                           || ssfkm1 >= SUPMX)) {
                                go631 = true;
                            } else if (ssf < 1.e-10f || ssf >= SUPMX) {
                                go631 = true;
                            } else {
                                renuc_block = true;
                            }
                        }
                        (void)go631;
                    } else {
                        // qc .le. qxmin(lc) (:10891-11012)
                        if (ssf > 0.0f) {
                            const float ssmx = SSMXINIT;
                            if (ssf > ssmx && ssf < 20.0f && ccnc > 0.05f * cwnccn) {
                                dcloud = qvexcess(qwvp, qv0, qc, pres, thetap, theta0,
                                                  pi0, fcqv1, felvcp, ssmx, pk);
                            } else {
                                dcloud = 0.0f;
                            }
                        } else {
                            dcloud = 0.0f;
                        }
                        thetap = thetap + __fdiv_rn(felvcp * dcloud, pi0);
                        qwvp = qwvp - dcloud;
                        qc = qc + dcloud;
                        theta = thetap + theta0;
                        temg = theta * pk;
                        ltemq = ltemq_of(temg);
                        qvs = pqs * tabqvs(ltemq);
                        cn = 0.0f;
                        const float cnuctmp = cnuc;
                        if (dcloud > QXMIN_C && wvel > 0.0f) {
                            cn = ccne0() * gfk_pow(cnuctmp, __fdiv_rn(2.0f, 2.0f + CCK))
                                * gfk_pow(wvel, cnexp());
                            // iccwflg = 1
#ifdef NSSL2_VARIABLE_CCCN
                            cn = nssl2_min(__fdiv_rn(nssl_cccn * rho0, RHO00),
#else
                            cn = nssl2_min(__fdiv_rn(CWCCN * rho0, RHO00),
#endif
                                           nssl2_max(cn, __fdiv_rn(rho0 * qc,
                                               XDN_C * four_pi_third() * four_micron_cube())));
                        } else {
                            cn = 0.0f;
                            dcloud = 0.0f;
                        }
                        if (cn > 0.0f) {
                            if (cn > ccnc) cn = ccnc;
                            ccnc = nssl2_max(ccnc - cn, 0.0f);
                            ccna = ccna + cn;
                        }
                        if (cn > cx_lc) cx_lc = cn;
                        if (cx_lc > 0.0f && qc <= QXMIN_C) {
                            cx_lc = 0.0f;
                        } else {
                            cx_lc = nssl2_min(cx_lc, __fdiv_rn(rho0 * nssl2_max(qc, 0.0f),
                                                               NSSL2_WRF_CWMASN));
                        }
                    }
                } else if (qc > 0.0f) {
                    // cloud evaporation (:10524-10591)
                    const float tc = temg - CBW;
                    const float r1 = __fdiv_rn(1.0f, 1.0f + __fdiv_rn(
                        CAW * (273.15f - CBW) * qss * felv, NCP * (tc * tc)));
                    const float qevap = nssl2_min(qc, r1 * (qss - qvap));
                    if (qc <= qevap) {
                        qwvp = qwvp + qc;
                        thetap = thetap - __fdiv_rn(felvcp * qc, pi0);
                        qc = 0.0f;
#ifdef NSSL2_VARIABLE_CCCN
                        ccnc = restore_ccn(ccnc, rho0, cx_lc, background_ccn_per_kg);
#else
                        ccnc = restore_ccn(ccnc, rho0, cx_lc);
#endif
                        cx_lc = 0.0f;
                    } else {
                        const float qctmp = qc;
                        qwvp = qwvp + qevap;
                        qc = qc - qevap;
                        if (qc <= 0.0f) {
#ifdef NSSL2_VARIABLE_CCCN
                            ccnc = restore_ccn(ccnc, rho0, cx_lc, background_ccn_per_kg);
#else
                            ccnc = restore_ccn(ccnc, rho0, cx_lc);
#endif
                            cx_lc = 0.0f;
                        } else {
                            const float tmp = __fdiv_rn(0.9f * qevap * cx_lc, qctmp);
#ifdef NSSL2_VARIABLE_CCCN
                            ccnc = restore_ccn(ccnc, rho0, tmp, background_ccn_per_kg);
#else
                            ccnc = restore_ccn(ccnc, rho0, tmp);
#endif
                            cx_lc = cx_lc - tmp;
                        }
                        thetap = thetap - __fdiv_rn(felvcp * qevap, pi0);
                    }
                }

                if (renuc_block) {
                    // irenuc = 2 (:11111-11149), then :11508-11513.
                    cn = ccne0() * gfk_pow(cnuc, __fdiv_rn(2.0f, 2.0f + CCK))
                        * gfk_pow(nssl2_max(wvel, 0.0f), cnexp());
                    cn = nssl2_min(cn, ccnc);
                    cn = nssl2_min(cn, __fdiv_rn(0.5f * dqc, NSSL2_WRF_CWMASN));
                    cn = nssl2_min(cn, nssl2_max(cnuc - ccna, 0.0f));
                    cx_lc = cx_lc + cn;
                    ccnc = nssl2_max(ccnc - cn, 0.0f);
                    ccna = ccna + cn;
                    if (cx_lc > 0.0f && qc <= QXMIN_C) cx_lc = 0.0f;
                }

                // 613 / 631: maximum supersaturation (:11523-11577).
                {
                    float ssmx = MAXSUPERSAT;
                    const float qv1 = qv0 + qwvp;
                    const float qvs1 = qvs;
                    if (qv1 > ssmx * qvs1) {
                        ssmx = 100.0f * (ssmx - 1.0f);
                        const float qvex = qvexcess(qwvp, qv0, qc, pres, thetap, theta0,
                                                    pi0, fcqv1, felvcp, ssmx, pk);
                        if (qvex > 0.0f) {
                            thetap = thetap + __fdiv_rn(felvcp * qvex, pi0);
                            qwvp = qwvp - qvex;
                            qc = qc + qvex;
                            cn = nssl2_min(nssl2_max(ccnc, cwnccn),
                                           __fdiv_rn(rho0 * qvex,
                                                     nssl2_max(nssl2_max(xmas_lc, cwmas20()),
                                                               cwmasn5())));
                            ccnc = nssl2_max(ccnc - cn, 0.0f);
                            cx_lc = cx_lc + cn;
                        }
                    }
                }
                // :11587-11600
                if (cx_lc > CXMIN && qc > QXMIN_C) {
                    xmas_lc = __fdiv_rn(rho0 * qc, cx_lc);
                    if (xmas_lc < NSSL2_WRF_CWMASN || xmas_lc > NSSL2_WRF_CWMASX) {
                        xmas_lc = nssl2_min(xmas_lc, NSSL2_WRF_CWMASX);
                        xmas_lc = nssl2_max(xmas_lc, NSSL2_WRF_CWMASN);
                        cx_lc = __fdiv_rn(rho0 * qc, xmas_lc);
                    }
                }
            }

            // ---- scatter (:11649-11695) ----
            an_lt = theta0 + thetap;
            an_lv = qv0 + qwvp;
            an_lc = qc + nssl2_min(an_lc, 0.0f);
            an_lr = qr + nssl2_min(an_lr, 0.0f);
            an_lnc = nssl2_max(cx_lc, 0.0f);
            an_lccn = nssl2_max(ccnc, 0.0f);
            an_lnr = nssl2_max(cx_lr, 0.0f);
        }
    }

    // ---- moment tail (:11732-12206), every point; frac = 1 ----
    const bool zero_lc = an_lnc <= 0.0f;
    const bool zero_lr = an_lnr <= 0.0f;
    const bool zero_li = an_lni <= 0.0f;
    const bool zero_ls = an_lns <= 0.0f;
    if (has_frozen) {
        float an_lh = qg[idx];
        float an_lhl = qh[idx];
        // The frozen arm requires concentration space (launcher-enforced).
        float an_lnh = qng[idx];
        float an_lnhl = qnh[idx];
        float an_lvh = qvolg[idx];
        float an_lvhl = qvolh[idx];
        const bool zero_lh = an_lnh <= 0.0f;
        const bool zero_lhl = an_lnhl <= 0.0f;
        // hail (:11799-11892): lhlw = 0, alphahl = 1 > 0.1.
        if (an_lhl < QXMIN_HL || zero_lhl) {
            an_lv = an_lv + an_lhl;
            an_lhl = 0.0f;
            an_lnhl = 0.0f;
            an_lvhl = 0.0f;
        } else {
            float tmp;
            if (an_lvhl > 0.0f) {
                tmp = __fdiv_rn(dn * an_lhl, an_lvhl);
            } else {
                tmp = 900.0f;                     // rho_qhl
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
            if (tmp < 500.0f) {                   // xdnmn(lhl)
                tmp = nssl2_max(500.0f, tmp);
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
            if (tmp > 900.0f) {                   // xdnmx(lhl)
                tmp = nssl2_min(900.0f, tmp);
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
        }
        // graupel (:11935-12033): lhw = 0; alphah = 0 arms the intercept.
        if (an_lh < QXMIN_H || zero_lh) {
            an_lv = an_lv + an_lh;
            an_lh = 0.0f;
            an_lnh = 0.0f;
            an_lvh = 0.0f;
        } else {
            float tmp;
            if (an_lvh > 0.0f) {
                tmp = __fdiv_rn(dn * an_lh, an_lvh);
            } else {
                tmp = 500.0f;                     // rho_qh
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (tmp < 170.0f) {                   // xdnmn(lh)
                tmp = nssl2_max(170.0f, tmp);
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (tmp > 900.0f) {                   // xdnmx(lh)
                tmp = nssl2_min(900.0f, tmp);
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (an_lh > QXMIN_H) {
                float hwdn;
                if (an_lvh > 0.0f) {
                    hwdn = __fdiv_rn(dn * an_lh, an_lvh);
                } else {
                    hwdn = 500.0f;                // xdn0(lh)
                }
                hwdn = nssl2_max(170.0f, hwdn);
                tmp = __fdiv_rn(hwdn * an_lnh, dn * an_lh);
                const float tmpg = an_lnh * gfk_pow(tmp * 3.14159f, 1.0f / 3.0f);
                if (tmpg < CNOHMN) {
                    tmp = gfk_pow(__fdiv_rn(hwdn, dn * an_lh) * 3.14159f, 1.0f / 3.0f);
                    an_lnh = gfk_pow(__fdiv_rn(CNOHMN, tmp), 0.75f);
                }
            }
        }
        qg[idx] = an_lh;
        qh[idx] = an_lhl;
        qng[idx] = an_lnh;
        qnh[idx] = an_lnhl;
        qvolg[idx] = an_lvh;
        qvolh[idx] = an_lvhl;
    }
    // snow (:12036-12092): lvs = lsw = 0, both temperature arms alike.
    if (an_ls < QXMIN_S || zero_ls) {
        an_lv = an_lv + an_ls;
        an_ls = 0.0f;
        an_lns = 0.0f;
    }
    // rain (:12098-12111)
    if (an_lr < QXMIN_R || zero_lr) {
        an_lv = an_lv + an_lr;
        an_lr = 0.0f;
        an_lnr = 0.0f;
    }
    // cloud ice (:12116-12123)
    if (an_li <= QXMIN_I || zero_li) {
        an_lv = an_lv + an_li;
        an_li = 0.0f;
        an_lni = 0.0f;
    }
    // cloud water (:12158-12200)
    if (an_lc <= QXMIN_C || zero_lc) {
        an_lv = an_lv + an_lc;
        an_lc = 0.0f;
        an_lccn = an_lccn + nssl2_max(an_lnc, 0.0f);
        an_lnc = 0.0f;
        an_lccn = nssl2_max(an_lccn, 0.0f);
        const float tmp = an_li + an_ls;
        if (an_lccn > 1.0f && tmp < QXMIN_I) {
#ifdef NSSL2_VARIABLE_CCCN
            an_lccn = dn * background_ccn_per_kg - nssl2_max(dn * background_ccn_per_kg - an_lccn, 0.0f)
#else
            an_lccn = dn * QCCN - nssl2_max(dn * QCCN - an_lccn, 0.0f)
#endif
                * gfk_exp(-__fdiv_rn(dtp, CCNTIMECONST));
        }
    }

    const float out_scale = concentration_space ? 1.0f : __fdiv_rn(1.0f, dn);
    full_theta[idx] = an_lt;
    qv_field[idx] = an_lv;
    qc_field[idx] = an_lc;
    qr_field[idx] = an_lr;
    qi_field[idx] = an_li;
    qs_field[idx] = an_ls;
    qndrop[idx] = an_lnc * out_scale;
    qnr[idx] = an_lnr * out_scale;
    qni[idx] = an_lni;
    qns[idx] = an_lns;
    qnn[idx] = an_lccn * out_scale;
}
