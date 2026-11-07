// WRF v4.6.1 WDM6 double-moment 6-class microphysics (mp_physics = 16).
//
// Transcribed line by line from the byte-frozen reference bundle's
//   phys/module_mp_wdm6.F  (wdm6 driver :121-330, wdm62D :334-2035,
//   wdm6init :2085-2211, slope_wdm6 :2213-2294, slope_rain :2296-2341,
//   slope_snow :2343-2389, slope_graup :2391-2435,
//   nislfv_rain_plmr :2438-2685, nislfv_rain_plm6 :2687-2954,
//   effectRad_wdm6 :3135-3234).
// Line citations below are into that file.  The cold (ice/snow/graupel)
// half is WSM6's physics with WDM6's own copies of the machinery; this
// translation unit is DELIBERATELY self-contained rather than sharing
// device code with kernels/wsm6.cu: module_mp_wdm6.F carries its own
// slope/fall routines (WRF's layout), and the house rule is that a new
// scheme gets a new .cu file.  That rule is enforced, not merely stated --
// tests/test_mp8_frozen.py::test_every_frozen_kernel_module_is_unchanged
// pins every pre-existing translation unit's file AND compiled-source
// sha256, so refactoring wsm6.cu into a shared header would change the
// mp=6 assembled source and therefore its compiled numerics.  WDM6's
// reflectivity obeys the same rule in kernels/wdm6_refl.cu rather than
// being appended to refl.cu.  The genuinely shared pieces are shared on
// the HOST instead (gpuwm/core/wdm6_constants.py imports WSM6's
// _rgmma/rimed_ice_constants).
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.  Rain mass AND rain number sediment with WDM6's
// explicit upstream flux scheme with per-column sub-stepping (:795-854);
// snow+graupel use the paired semi-Lagrangian monotone-PLM remap with one
// mean-velocity iteration (nislfv_rain_plm6, iter=1, :871-872); cloud ice
// uses the single-species remap with the departure velocity
// (nislfv_rain_plmr, iter=0, :974-975).
//
// Derived scheme constants are the FP32 words WRF's own wdm6init leaves in
// its default-REAL saved variables (WDM6_* below), read back from the
// Fortran by tools/wdm6_wrf461_oracle; gpuwm/core/wdm6_constants.py carries
// the same words and tests/test_wdm6.py pins the correspondence.  Every
// expression keeps the Fortran's operation order, the transcendentals are
// WOOF's own float32 exp/log/pow (gfk_exp, gfk_log, gfk_pow), and the unit
// compiles without multiply-add contraction (gpuwm/core/kernels/__init__.py
// _NO_FMAD_MODULES): the column oracle tools/wdm6_wrf461_oracle grades it
// word for word against WRF v4.6.1's module_mp_wdm6.F.
// Rain sedimentation deliberately corrects the inherited :810-839 update:
// each interface transfers one donor-owned mass and number amount. Capping
// the arrival by the donor's already-updated state discarded real rain.

#ifdef WDM6_CPU_MIRROR
#include <cmath>
using std::isfinite;
#define __device__
#define __forceinline__ inline
#define __global__
// A146: the device divides by constants through __fdiv_rn (NVRTC
// rewrites a plain constant division on Blackwell); on the host it
// is the IEEE quotient the compiler already gives.
static inline float __fdiv_rn(float a, float b) { return a / b; }
// The host mirror has no device libm block; the C library's float32
// routines stand in for gfk_exp/gfk_log/gfk_pow there.
#define gfk_exp(x) expf(x)
#define gfk_log(x) logf(x)
#define gfk_pow(x, y) powf((x), (y))
#endif

// wdm6init (:2085-2211) as WRF v4.6.1 computes it: default REAL, so every
// product below is the float32 word the Fortran's saved variable holds
// (gfortran 13.3, the oracle's own dump; float64 products of the same
// formulas differ from these in the last 1-70 ULP, enough to move every
// fall speed and collection rate off WRF's word).
#define WDM6_PI      3.14159274f      // 4.*atan(1.)          0x40490FDB
#define WDM6_QC0     8.3775805e-05f   // :2113                0x38AFB0CB
#define WDM6_QC1     0.00083775807f   // :2114                0x3A5B9CFE
#define WDM6_QCK1    4534.669f        // :2115                0x458DB55A
#define WDM6_PIDNC   523.5988f        // :2116                0x4402E653
#define WDM6_G4PBR   17.817297f       // :2130 rgmma(4.8)     0x418E89D3
#define WDM6_G7PBR   3366.664f        // :2133 rgmma(7.8)     0x45526AA0
#define WDM6_PVTR    2998.4763f       // :2136                0x453B679F
#define WDM6_PVTRN   1410.88f         // :2137                0x44B05C29
#define WDM6_PRECR1  9.801769f        // :2140                0x411CD40C
#define WDM6_PRECR2  299.26874f       // :2141                0x4395A266
#define WDM6_PIDNR   12566.371f       // :2143                0x4644597C
#define WDM6_ROQIMAX 8.125004e-05f    // :2146                0x38AA64C8
#define WDM6_PVTS    20.051638f       // :2156                0x41A069C1
#define WDM6_PRECS1  5200000.0f       // :2158                0x4A9EB100
#define WDM6_PRECS2  18681944.0f      // :2159                0x4B8E882C
#define WDM6_PIDN0S  628318528.0f     // :2160                0x4E15CD85
#define WDM6_PACRC   55442112.0f      // :2162                0x4C537EB0
#define WDM6_RSLOPECMAX  2.0e-06f        // :2178            0x360637BD
#define WDM6_RSLOPEC2MAX 4.0e-12f        // :2185            0x2C8CBCCC
#define WDM6_RSLOPEC3MAX 7.9999995e-18f  // :2189            0x231392EE
#define WDM6_RSLOPERMAX  2.0e-05f        // :2179            0x37A7C5AC
#define WDM6_RSLOPERBMAX 0.00017411009f  // :2182            0x39369152
#define WDM6_RSLOPER2MAX 3.9999998e-10f  // :2186            0x2FDBE6FE
#define WDM6_RSLOPER3MAX 7.999999e-15f   // :2190            0x28101D7C
#define WDM6_RSLOPESMAX  1.0e-05f        // :2180            0x3727C5AC
#define WDM6_RSLOPESBMAX 0.008912509f    // :2183            0x3C1205C6
#define WDM6_RSLOPES2MAX 9.9999994e-11f  // :2187            0x2EDBE6FE
#define WDM6_RSLOPES3MAX 9.999999e-16f   // :2191            0x26901D7C
// effectRad_wdm6's rgmma(5./3.) (:3189), the same truncated product.
#define WDM6_CDM2    0.90261394f      //                      0x3F6711B5
// wdm62D :1962 (actr*1.E-6)**3, which gfortran folds at compile time.
#define WDM6_ACTR3   3.3750003e-18f   //                      0x227907F4
// The two transcendental forms the module spells exp(log(x)*c).
#define WDM6_EXPLOG(x, c) gfk_exp(gfk_log(x) * (c))

#ifndef WDM6_KMAX
#define WDM6_KMAX 64
#endif

// The limit is the signed counter's first unrepresentable positive value.
// Check the exact FP32 schedule expression before converting it to int.
__device__ __forceinline__ int wdm6_checked_steps(float value) {
    float count = floorf(value);
    if (!isfinite(count) || count < 0.0f || count >= 2147483648.0f) return -1;
    return count < 1.0f ? 1 : (int)count;
}

__device__ __forceinline__ int wdm6_rain_steps(float den, float dz, float dtcld) {
    // The existing inverse-slope cap bounds future vt, including dry receivers.
    // Its coupled wave speed is <1.25*vt; wave Courant 0.5 gives terminal
    // Courant 0.4. This is the production schedule itself, not a looser estimate.
    float df = sqrtf(1.28f / den);
    return wdm6_checked_steps(2.5f * 2.99849272e3f * gfk_pow(1.0e-3f, 0.8f)
                             * df / dz * dtcld + 1.0f);
}

__device__ __forceinline__ float wdm6_max0(float x) { return fmaxf(x, 0.0f); }
__device__ __forceinline__ float wdm6_eff01(float x) {
    // The "reduce collection efficiency" factor min(max(0.,x),1.)**2
    // (B. Wilt, e.g. :1312).
    float c = fminf(fmaxf(x, 0.0f), 1.0f);
    return c * c;
}

// wdm6init :2096-2108 (hail_opt) products: the branch sets the same five
// base constants (n0g/deng/avtg/bvtg/lamdagmax) WSM6's does; the derived
// words are wdm6init's own float32 results, as for WDM6_* above.
struct WdmRimed {
    float n0g, deng, bvtg, pidn0g, pvtg, pacrg, precg1, precg2;
    float rslopegmax, rslopegbmax, rslopeg2max, rslopeg3max;
};

__device__ __forceinline__ WdmRimed wdm6_rimed_constants(int hail_opt) {
    WdmRimed r;
    if (hail_opt == 1) {
        r.n0g = 4.0e4f; r.deng = 700.0f; r.bvtg = 0.8f;
        r.pidn0g = 87964600.0f;          // 0x4CA7C777
        r.pvtg = 846.3216f;              // 0x44539495
        r.pacrg = 41999168.0f;           // 0x4C2036D0
        r.precg1 = 196035.375f;          // 0x483F70D8
        r.precg2 = 2402510.0f;           // 0x4A12A338
        r.rslopegmax = 5.0e-05f;         // 0x3851B717
        r.rslopegbmax = 0.00036238978f;  // 0x39BDFF22
        r.rslopeg2max = 2.4999998e-09f;  // 0x312BCC76
        r.rslopeg3max = 1.2499999e-13f;  // 0x2A0CBCCB
    } else {
        r.n0g = 4.0e6f; r.deng = 500.0f; r.bvtg = 0.8f;
        r.pidn0g = 6283185664.0f;        // 0x4FBB40E7
        r.pvtg = 979.95135f;             // 0x4474FCE3
        r.pacrg = 4863062016.0f;         // 0x4F90EE38
        r.precg1 = 19603538.0f;          // 0x4B959029
        r.precg2 = 258523328.0f;         // 0x4D768C0C
        r.rslopegmax = 1.6666667e-05f;   // 0x378BCF65
        r.rslopegbmax = 0.00015048006f;  // 0x391DCA2F
        r.rslopeg2max = 2.7777777e-10f;  // 0x2F98B5BF
        r.rslopeg3max = 4.6296297e-15f;  // 0x27A6CCCC
    }
    return r;
}

struct WdmSlope {
    float r, rb, r2, r3, vt;
};

// slope_wdm6 rain arm (:2240, :2251-2261, :2284-2288): gamma mu=1 PSD with
// the PROGNOSTIC intercept -- lamdar = ((pidnr*nr)/(q*den))**(1/3),
// pidnr = 4*pi*denr, rslope capped at 1e-3, and the pair of fall speeds
// pvtr (mass, avtr*Gamma(5.8)/24) / pvtrn (number, avtr*Gamma(2.8)).
// ``denfac`` is always the wdm62D array (:653-659), sqrt((1/den)*den0).
struct WdmRainSlope {
    float r, rb, r2, r3, vt, vtn;
};

struct WdmRainFallout {
    float mass;       // kg m-2 leaving the bottom interface this sub-step
    float number;     // m-2 leaving the bottom interface this sub-step
};

__device__ __forceinline__ WdmRainFallout wdm6_rain_substep(
    float *qr, float *nr, const float *den, const float *dz,
    const float *mass_rate, const float *number_rate, int nz, float dt)
{
    float incoming_mass = 0.0f, incoming_number = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const float air_mass = den[k] * dz[k];
        const float mass = qr[k] * air_mass;
        const float number = nr[k] * dz[k];
        // The donor bounds its outgoing transfer once. Both adjacent
        // cells use that same amount, including unequal density/depth.
        // This is a conservative flux limit, never a repaired state.
        const float outgoing_mass = fminf(mass, mass * mass_rate[k] * dt);
        const float outgoing_number = fminf(number, number * number_rate[k] * dt);
        qr[k] = (mass - outgoing_mass + incoming_mass) / air_mass;
        nr[k] = (number - outgoing_number + incoming_number) / dz[k];
        incoming_mass = outgoing_mass;
        incoming_number = outgoing_number;
    }
    return {incoming_mass, incoming_number};
}

__device__ __forceinline__ WdmRainSlope wdm6_rain_slope(
    float q, float nr, float den, float denfac) {
    WdmRainSlope s;
    if (q <= 1.0e-9f || nr <= 1.0e-2f) {
        s.r = WDM6_RSLOPERMAX; s.rb = WDM6_RSLOPERBMAX;
        s.r2 = WDM6_RSLOPER2MAX; s.r3 = WDM6_RSLOPER3MAX;
    } else {
        float lamdar = WDM6_EXPLOG((WDM6_PIDNR * nr) / (q * den),
                                   0.33333333f);
        s.r = fminf(1.0f / lamdar, 1.0e-3f);
        s.rb = gfk_pow(s.r, 0.8f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = WDM6_PVTR * s.rb * denfac;
    s.vtn = WDM6_PVTRN * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    if (nr <= 0.0f) s.vtn = 0.0f;
    return s;
}

// slope_wdm6 snow arm (:2241, :2262-2272, :2285/:2289) == slope_snow
// (:2343-2389): fixed n0s with the supercooling intercept factor.
__device__ __forceinline__ WdmSlope wdm6_snow_slope(
    float q, float den, float denfac, float temp, float *n0sfac_out) {
    WdmSlope s;
    float n0sfac = fmaxf(fminf(gfk_exp(0.12f * (273.15f - temp)), 5.0e4f),
                         1.0f);
    *n0sfac_out = n0sfac;
    if (q <= 1.0e-9f) {
        s.r = WDM6_RSLOPESMAX; s.rb = WDM6_RSLOPESBMAX;
        s.r2 = WDM6_RSLOPES2MAX; s.r3 = WDM6_RSLOPES3MAX;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(WDM6_PIDN0S * n0sfac / (q * den)));
        s.rb = gfk_pow(s.r, 0.41f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = WDM6_PVTS * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// slope_wdm6 graupel arm (:2242, :2273-2283, :2286/:2290) == slope_graup
// (:2391-2435).
__device__ __forceinline__ WdmSlope wdm6_graupel_slope(
    float q, float den, float denfac, const WdmRimed &g) {
    WdmSlope s;
    if (q <= 1.0e-9f) {
        s.r = g.rslopegmax; s.rb = g.rslopegbmax;
        s.r2 = g.rslopeg2max; s.r3 = g.rslopeg3max;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(g.pidn0g / (q * den)));
        s.rb = gfk_pow(s.r, g.bvtg); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = g.pvtg * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// denfac (:653-659): VREC then *den0 then VSQRT, i.e. sqrt((1/den)*den0)
// -- two roundings before the root, not den0/den.  A pure function of den,
// so it is evaluated where it is read rather than stored per level.
__device__ __forceinline__ float wdm6_denfac(float den) {
    return sqrtf((1.0f / den) * 1.28f);
}

// Statement functions (:562-567).  diffus and venfac are spelled
// exp(log(x)*c) in the module (:555 "A**B => exp(log(A)*(B))"), which is
// not pow's word, so they are evaluated as written.
__device__ __forceinline__ float wdm6_diffus(float t, float p) {
    return 8.794e-5f * WDM6_EXPLOG(t, 1.81f) / p;
}
__device__ __forceinline__ float wdm6_viscos(float t, float den) {
    return 1.496e-6f * (t * sqrtf(t)) / (t + 120.0f) / den;
}
__device__ __forceinline__ float wdm6_xka(float t, float den) {
    return 1.414e3f * wdm6_viscos(t, den) * den;
}
__device__ __forceinline__ float wdm6_diffac(float latent, float p, float t,
                                             float den, float qsat) {
    return den * latent * latent / (wdm6_xka(t, den) * 461.6f * t * t)
         + 1.0f / (qsat * wdm6_diffus(t, p));
}
__device__ __forceinline__ float wdm6_venfac(float p, float t, float den) {
    return WDM6_EXPLOG(wdm6_viscos(t, den) / wdm6_diffus(t, p), 0.3333333f)
         / sqrtf(wdm6_viscos(t, den)) * sqrtf(sqrtf(1.28f / den));
}
// Inline fpvs expansion (:661-695): saturation mixing ratio over water
// (ice=false) / ice (ice=true), with the 0.99p cap, ep2 conversion, and
// the qmin floor applied by the caller exactly as the Fortran does.
__device__ __forceinline__ float wdm6_fpvs(float t, bool ice) {
    const float ttp = 273.15f + 0.01f;             // t0c + 0.01 (:668)
    const float xa = -(1846.4f - 4190.0f) / 461.6f;
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    const float xai = -(1846.4f - 2106.0f) / 461.6f;
    const float xbi = xai + 2.85e6f / (461.6f * ttp);
    float tr = ttp / t;
    float aa = (ice && t < ttp) ? xai : xa;
    float bb = (ice && t < ttp) ? xbi : xb;
    return 610.78f * WDM6_EXPLOG(tr, aa) * gfk_exp(bb * (1.0f - tr));
}
__device__ __forceinline__ float wdm6_qsat(float t, float p, bool ice) {
    float es = fminf(wdm6_fpvs(t, ice), 0.99f * p);
    float qs = (287.0f / 461.6f) * es / (p - es);
    return fmaxf(qs, 1.0e-15f);
}

// nislfv_rain_plm6/plmr arrival-point construction (:2749-2807 pair /
// :2499-2556 single): third-order interface velocity interpolation, the
// raingroup-top and diffusivity CFL limits, arrival coordinates, and the
// deformed density concentrations.  Arrays with nz+1 entries are
// interfaces; qq2 may be null for the single-species (ice) call.
__device__ __forceinline__ void wdm6_plm_arrival(
    const float *qq, const float *qq2, const float *ww, const float *dz,
    int nz, float dt, float *zi, float *za, float *dza,
    float *qa, float *qa2)
{
    float wi[WDM6_KMAX + 1];
    zi[0] = 0.0f;
    for (int k = 0; k < nz; ++k) zi[k + 1] = zi[k] + dz[k];

    // 2nd-order interface interpolation, then overwritten by the active
    // 3rd-order one (:2760-2774), exactly as the Fortran computes both.
    wi[0] = ww[0];
    wi[nz] = ww[nz - 1];
    for (int k = 1; k < nz; ++k)
        wi[k] = (ww[k] * dz[k - 1] + ww[k - 1] * dz[k])
              / (dz[k - 1] + dz[k]);
    wi[0] = ww[0];
    if (nz > 1) wi[1] = 0.5f * (ww[1] + ww[0]);
    for (int k = 2; k < nz - 1; ++k)
        wi[k] = (9.0f / 16.0f) * (ww[k] + ww[k - 1])
              - (1.0f / 16.0f) * (ww[k + 1] + ww[k - 2]);
    if (nz > 1) wi[nz - 1] = 0.5f * (ww[nz - 1] + ww[nz - 2]);
    wi[nz] = ww[nz - 1];

    for (int k = 1; k < nz; ++k)                   // :2777-2779
        if (ww[k] == 0.0f) wi[k] = ww[k - 1];
    for (int k = nz - 1; k >= 0; --k) {            // :2782-2788
        float decfl = (wi[k + 1] - wi[k]) * dt / dz[k];
        if (decfl > 0.05f) wi[k] = wi[k + 1] - 0.05f * dz[k] / dt;
    }
    for (int k = 0; k <= nz; ++k) za[k] = zi[k] - wi[k] * dt;
    for (int k = 0; k < nz; ++k) dza[k] = za[k + 1] - za[k];
    dza[nz] = zi[nz] - za[nz];
    for (int k = 0; k < nz; ++k) {
        qa[k] = qq[k] * dz[k] / dza[k];
        if (qq2) qa2[k] = qq2[k] * dz[k] / dza[k];
    }
    qa[nz] = 0.0f;
    if (qq2) qa2[nz] = 0.0f;
}

// Monotone PLM interface reconstruction, remap to the regular grid, and
// the rain-out integral (:2843-2939 / :2582-2677).  Returns the fallen
// mass per area (the Fortran's precip).
__device__ __forceinline__ float wdm6_plm_remap(
    const float *qa, const float *zi, const float *za, const float *dza,
    int nz, float *qn)
{
    float qmi[WDM6_KMAX + 1], qpi[WDM6_KMAX + 1];
    for (int k = 1; k < nz; ++k) {
        float dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
        float dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
        if (dip * dim <= 0.0f) {
            qmi[k] = qa[k]; qpi[k] = qa[k];
        } else {
            qpi[k] = qa[k] + 0.5f * (dip + dim) * dza[k];
            qmi[k] = 2.0f * qa[k] - qpi[k];
            if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                qpi[k] = qa[k]; qmi[k] = qa[k];
            }
        }
    }
    qpi[0] = qa[0]; qmi[0] = qa[0];
    qpi[nz] = qa[nz]; qmi[nz] = qa[nz];
    for (int k = 0; k < nz; ++k) qn[k] = 0.0f;

    int kb = 0, kt = 0;
    for (int k = 0; k < nz; ++k) {
        kb = (kb > 0) ? kb - 1 : 0;
        kt = (kt > 0) ? kt - 1 : 0;
        if (zi[k] >= za[nz]) break;
        for (int kk = kb; kk < nz; ++kk) {
            if (zi[k] <= za[kk + 1]) { kb = kk; break; }
        }
        for (int kk = kt; kk < nz; ++kk) {
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        }
        // DELIBERATE, DOCUMENTED PLM DIVERGENCE from the Fortran.
        // The separate rain-interface correction is documented above.
        // nislfv_rain_plmr:2629 and
        // nislfv_rain_plm6:2891 both do an UNCONDITIONAL `kt = kt - 1`
        // here; this clamps at the first index instead.  The two differ on
        // exactly one input: when find_kt leaves kt at its first index
        // (Fortran kt==1, i.e. zi(k+1) <= za(1)).  There the Fortran's kt
        // falls to 0, which is below kb's floor of 1, so NEITHER the
        // kt==kb nor the kt>kb branch runs and qn(k) keeps its qn=0
        // initialisation -- the layer's mass is dropped on the floor.  The
        // clamp takes the kt==kb piecewise-constant branch and conserves
        // it.  ArWen's standing rule is to implement the DEFINED behaviour
        // where WRF is undefined, and losing mass out of an index guard is
        // not a modelled process, so the clamp is the choice.
        //
        // NOT REACHABLE as far as we can construct, for nz >= 2: the
        // con1 = 0.05 diffusivity limiter above forces dza >= 0.95*dz, so
        // the arrival column cannot compress enough for zi(k+1) to fall to
        // or below za(0), and nz = 1 is below the adapter's
        // VERTICAL_LEVEL_BOUNDS floor of 2 (gpuwm/core/wdm6.py).  It is
        // written down anyway because "unreachable" is an argument, not a
        // measurement, and an oracle campaign that ever does fire it will
        // otherwise be hunting a silent 1-layer mass difference.
        kt = (kt > 0) ? kt - 1 : 0;
        if (kt == kb) {
            // tl2 = tl*tl and th2 = th*th are their own roundings in the
            // Fortran (:2634-2638, :2643-2644, :2655-2657), multiplied into
            // qqd afterwards.
            float tl = (zi[k] - za[kb]) / dza[kb];
            float th = (zi[k + 1] - za[kb]) / dza[kb];
            float tl2 = tl * tl;
            float th2 = th * th;
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qqh = qqd * th2 + qmi[kb] * th;
            float qql = qqd * tl2 + qmi[kb] * tl;
            qn[k] = (qqh - qql) / (th - tl);
        } else if (kt > kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float tl2 = tl * tl;
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qql = qqd * tl2 + qmi[kb] * tl;
            float zsum = (1.0f - tl) * dza[kb];
            float qsum = (qa[kb] - qql) * dza[kb];
            for (int m = kb + 1; m < kt; ++m) {
                zsum += dza[m]; qsum += qa[m] * dza[m];
            }
            float th = (zi[k + 1] - za[kt]) / dza[kt];
            float th2 = th * th;
            float qqdt = 0.5f * (qpi[kt] - qmi[kt]);
            float dqh = qqdt * th2 + qmi[kt] * th;
            zsum += th * dza[kt]; qsum += dqh * dza[kt];
            qn[k] = qsum / zsum;
        }
    }

    float precip = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (za[k] < 0.0f && za[k + 1] < 0.0f) {
            precip += qa[k] * dza[k];
        } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
#ifdef WDM6_PLM_SURFACE
            // Correct mode, finding WB5 (physics.mode = "correct"; the WSM6
            // defect of kernels/wsm6.cu, shared by mp_wdm6.F's
            // nislfv_rain_plm6/plmr): book the below-ground part of the
            // straddling source cell with the same piecewise-linear profile
            // (qmi/qpi) the remap above integrates for the part that stays
            // aloft, so column water + surface precipitation is conserved.
            // The Fortran books it at the flat cell mean (the #else line,
            // exact mode).
            const float th = (0.0f - za[k]) / dza[k];
            precip += dza[k] * (qmi[k] * th
                      + 0.5f * (qpi[k] - qmi[k]) * th * th);
#else
            precip += qa[k] * (0.0f - za[k]);
#endif
            break;
        } else {
            break;
        }
    }
    return precip;
}

// Caller block :858-884 + nislfv_rain_plm6 with iter=1 (:2812-2835):
// snow and graupel share the mass-weighted combined fall speed, one
// arrival-velocity iteration averages departure and arrival winds.
__device__ void wdm6_sediment_snow_graupel(
    float *qs, float *qg, const float *den,
    const float *dz, const float *temp, int nz, float dt, const WdmRimed &g,
    float *precip_s, float *precip_g)
{
    float qq[WDM6_KMAX], qq2[WDM6_KMAX], wd[WDM6_KMAX], ww[WDM6_KMAX];
    float qn[WDM6_KMAX], qn2[WDM6_KMAX];
    float zi[WDM6_KMAX + 1], za[WDM6_KMAX + 1];
    float dza[WDM6_KMAX + 1], qa[WDM6_KMAX + 1], qa2[WDM6_KMAX + 1];
    *precip_s = 0.0f; *precip_g = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        float q1 = qs[k], q2 = qg[k];
        float n0s;
        // worka (:858-869): work1(:,2)/(:,3) from slope_wdm6 of the
        // unchanged qs/qg, mass-weighted with the same 1e-15 guard.
        float df = wdm6_denfac(den[k]);
        float vs = wdm6_snow_slope(q1, den[k], df, temp[k], &n0s).vt;
        float vg = wdm6_graupel_slope(q2, den[k], df, g).vt;
        float total = fmaxf(q1 + q2, 1.0e-15f);
        ww[k] = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        wd[k] = ww[k];
        qq[k] = den[k] * q1; qq2[k] = den[k] * q2;
        allold = allold + qq[k] + qq2[k];
    }
    if (allold <= 0.0f) return;                    // :2740-2746

    wdm6_plm_arrival(qq, qq2, ww, dz, nz, dt, zi, za, dza, qa, qa2);
    // iter = 1 (:2812-2835): one arrival-velocity estimate, mean wind.
    for (int k = 0; k < nz; ++k) {
        float q1 = qa[k] / den[k];
        float q2 = qa2[k] / den[k];
        float df = wdm6_denfac(den[k]);
        float n0s;
        float vs = wdm6_snow_slope(q1, den[k], df, temp[k], &n0s).vt;
        float vg = wdm6_graupel_slope(q2, den[k], df, g).vt;
        float total = fmaxf(q1 + q2, 1.0e-15f);
        float wa = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        ww[k] = 0.5f * (wd[k] + wa);
    }
    wdm6_plm_arrival(qq, qq2, ww, dz, nz, dt, zi, za, dza, qa, qa2);

    *precip_s = wdm6_plm_remap(qa, zi, za, dza, nz, qn);
    *precip_g = wdm6_plm_remap(qa2, zi, za, dza, nz, qn2);
    for (int k = 0; k < nz; ++k) {
        qs[k] = fmaxf(qn[k] / den[k], 0.0f);       // :873-876
        qg[k] = fmaxf(qn2[k] / den[k], 0.0f);
    }
}

// Caller block :955-983 + nislfv_rain_plmr with iter=0 (departure wind,
// no velocity iteration): cloud-ice fallout on its own HDC 5a fall speed.
__device__ float wdm6_sediment_ice(
    float *qi, const float *ww, const float *den, const float *dz, int nz,
    float dt)
{
    float qq[WDM6_KMAX], qn[WDM6_KMAX];
    float zi[WDM6_KMAX + 1], za[WDM6_KMAX + 1];
    float dza[WDM6_KMAX + 1], qa[WDM6_KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        qq[k] = den[k] * qi[k];
        allold = allold + qq[k];
    }
    if (allold <= 0.0f) return 0.0f;
    wdm6_plm_arrival(qq, nullptr, ww, dz, nz, dt, zi, za, dza, qa,
                     nullptr);
    float precip = wdm6_plm_remap(qa, zi, za, dza, nz, qn);
    for (int k = 0; k < nz; ++k)
        qi[k] = fmaxf(qn[k] / den[k], 0.0f);       // :976-980
    return precip;
}

__device__ unsigned int wdm6_column_impl(
    float *theta, float *qv_g, float *qc_g, float *qi_g, float *qr_g,
    float *qs_g, float *qg_g, float *nn_g, float *nc_g, float *nr_g,
    const float *den_g, const float *p, const float *pii,
    const float *dz_g, const float *xland,
    float *rainnc, float *rainncv, float *snownc, float *snowncv,
    float *graupelnc, float *graupelncv, float *sr,
    float *effc, float *effi, float *effs,
    float delt, int hail_opt, int nz, int stride, int col)
{
    if (!isfinite(delt) || delt <= 0.0f) return 1u;
    // loops = max(nint(delt/dtcldcr),1) (:639): nint is round-half-away,
    // which roundf is exactly; adding 0.5 first can round up below a half.
    int loops = wdm6_checked_steps(roundf(__fdiv_rn(delt, 120.0f)));
    if (loops < 0) return 2u;
    float dtcld = delt <= 120.0f ? delt : delt / (float)loops;
    int mstep = 1;
    unsigned int count_flags = 0u;
    const WdmRimed gcn = wdm6_rimed_constants(hail_opt);
    float t[WDM6_KMAX], qv[WDM6_KMAX], qc[WDM6_KMAX], qi[WDM6_KMAX];
    float qr[WDM6_KMAX], qs[WDM6_KMAX], qg[WDM6_KMAX];
    float nn[WDM6_KMAX], nc[WDM6_KMAX], nr[WDM6_KMAX];
    float den[WDM6_KMAX], delz[WDM6_KMAX], cpm[WDM6_KMAX], xl[WDM6_KMAX];
    float qsw[WDM6_KMAX], qsi[WDM6_KMAX], rhw[WDM6_KMAX], rhi[WDM6_KMAX];
    float xni_pre[WDM6_KMAX], work1r[WDM6_KMAX], workn[WDM6_KMAX];
    float rslope3r[WDM6_KMAX];

    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        den[k] = den_g[id]; delz[k] = dz_g[id];
        // This exact schedule depends only on the density, thickness and
        // minor interval, which are constant throughout this scheme call.
        // Check it while loading the column, before any output is written.
        if (!isfinite(den[k]) || den[k] <= 0.0f || !isfinite(delz[k]) || delz[k] <= 0.0f)
            count_flags |= 1u;
        else {
            int steps = wdm6_rain_steps(den[k], delz[k], dtcld);
            if (steps < 0) count_flags |= 2u;
            else if (steps > mstep) mstep = steps;
        }
        // padding for negative values generated by dynamics (:577-588),
        // including the CCN floor/cap of the Oct-2017 KIAPS revision.
        qv[k] = qv_g[id];
        qc[k] = wdm6_max0(qc_g[id]); qi[k] = wdm6_max0(qi_g[id]);
        qr[k] = wdm6_max0(qr_g[id]); qs[k] = wdm6_max0(qs_g[id]);
        qg[k] = wdm6_max0(qg_g[id]);
        nn[k] = fminf(fmaxf(nn_g[id], 1.0e8f), 2.0e10f);
        nc[k] = wdm6_max0(nc_g[id]); nr[k] = wdm6_max0(nr_g[id]);
        // cpm/xl once at scheme entry (:600-605), never refreshed.
        float vap = fmaxf(qv[k], 1.0e-15f);
        cpm[k] = 1004.5f * (1.0f - vap) + vap * 1846.4f;
        xl[k] = 2.5e6f - 2343.6f * (t[k] - 273.15f);
    }
    if (count_flags) return count_flags;
    // Land/sea autoconversion threshold (:607-614): xland == 2 is water
    // (module_microphysics_driver.F:2495 "land mask, 1: land, 2: water"),
    // maritime qc0 (xncr0=5e7) against continental qc1 (xncr1=5e8).
    float qcr_col = (xland[col] == 2.0f) ? WDM6_QC0 : WDM6_QC1;

    rainncv[col] = 0.0f; snowncv[col] = 0.0f; graupelncv[col] = 0.0f;
    sr[col] = 0.0f;                                // :625-635

    for (int loop = 0; loop < loops; ++loop) {
        // Saturation state at minor-loop start (:665-695); consumed
        // unrefreshed by the process block exactly as the Fortran's qs/rh
        // arrays are.
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = wdm6_qsat(t[k], p[id], false);
            qsi[k] = wdm6_qsat(t[k], p[id], true);
            rhw[k] = fmaxf(qv[k] / qsw[k], 1.0e-15f);
            rhi[k] = fmaxf(qv[k] / qsi[k], 1.0e-15f);
        }
        // Pre-sedimentation ice number column (:771-777): consumed by the
        // ice fall speed (:960) and by nimlt (:1034) AFTER ice has fallen,
        // so it must be the loop-start value, not a live recompute.
        for (int k = 0; k < nz; ++k) {
            float temp = den[k] * fmaxf(qi[k], 1.0e-15f);
            temp = sqrtf(sqrtf(temp * temp * temp));
            xni_pre[k] = fminf(fmaxf(5.38e7f * temp, 1.0e3f), 1.0e6f);
        }

        // ---- rain mass+number fallout: explicit upstream flux with
        // per-column sub-stepping (:790-854) --------------------------------
        for (int k = nz - 1; k >= 0; --k) {
            WdmRainSlope rsl = wdm6_rain_slope(qr[k], nr[k], den[k],
                                               wdm6_denfac(den[k]));
            work1r[k] = rsl.vt / delz[k];
            workn[k] = rsl.vtn / delz[k];
        }
        float fall_r_sfc = 0.0f;                   // fall(i,kts,1)
        for (int n = 0; n < mstep; ++n) {
            const WdmRainFallout fallout = wdm6_rain_substep(
                qr, nr, den, delz, work1r, workn, nz, dtcld / (float)mstep);
            // Retain the existing precipitation accumulator's units.
            fall_r_sfc += fallout.mass / delz[0] / dtcld;
            // slope_rain refresh for the next sub-step (:840-853).
            for (int k = nz - 1; k >= 0; --k) {
                WdmRainSlope rsl = wdm6_rain_slope(qr[k], nr[k], den[k],
                                                   wdm6_denfac(den[k]));
                work1r[k] = rsl.vt / delz[k];
                workn[k] = rsl.vtn / delz[k];
            }
        }

        // ---- snow + graupel paired semi-Lagrangian fallout (:858-884) -----
        float ps_sfc_mass, pg_sfc_mass;
        wdm6_sediment_snow_graupel(qs, qg, den, delz, t, nz, dtcld,
                                   gcn, &ps_sfc_mass, &pg_sfc_mass);
        float fall2_sfc = ps_sfc_mass / delz[0] / dtcld;   // :881-884
        float fall3_sfc = pg_sfc_mass / delz[0] / dtcld;

        // ---- psmlt/pgmlt with their nr transfers (:885-951) ---------------
        // Consumes the :893 slope_wdm6 refresh; rain rslope3 from that
        // refresh (post-sediment qr/nr, PRE-melt) must survive to pgfrz
        // (:1081/:1089), so it is captured here.
        for (int k = nz - 1; k >= 0; --k) {
            int id = col + k * stride;
            float df = wdm6_denfac(den[k]);
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k], df);
            rslope3r[k] = rr.r3;
            float n0sfac;
            WdmSlope ss = wdm6_snow_slope(qs[k], den[k], df, t[k], &n0sfac);
            WdmSlope gg = wdm6_graupel_slope(qg[k], den[k], df, gcn);
            if (t[k] > 273.15f) {
                float xlf = 3.5e5f;
                float work2 = wdm6_venfac(p[id], t[k], den[k]);
                if (qs[k] > 0.0f) {
                    float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                    float psmlt = __fdiv_rn(wdm6_xka(t[k], den[k]), xlf)
                        * (273.15f - t[k]) * WDM6_PI / 2.0f
                        * n0sfac * (WDM6_PRECS1 * ss.r2
                                    + WDM6_PRECS2 * work2 * coeres)
                        / den[k];
                    psmlt = fminf(fmaxf(psmlt * dtcld, -qs[k]), 0.0f);
                    if (qs[k] > 1.0e-9f) {         // nsmlt (:917-920)
                        float sfac = ss.r * 2.0e6f * n0sfac / qs[k];
                        nr[k] = nr[k] - sfac * psmlt;
                    }
                    qs[k] += psmlt; qr[k] -= psmlt;
                    t[k] += xlf / cpm[k] * psmlt;
                }
                if (qg[k] > 0.0f) {
                    // pgmlt reads t AFTER the psmlt update (:932), with
                    // work2 from the block-entry t (:906).
                    float coeres = gg.r2 * sqrtf(gg.r * gg.rb);
                    float pgmlt = __fdiv_rn(wdm6_xka(t[k], den[k]), xlf)
                        * (273.15f - t[k])
                        * (gcn.precg1 * gg.r2
                           + gcn.precg2 * work2 * coeres) / den[k];
                    pgmlt = fminf(fmaxf(pgmlt * dtcld, -qg[k]), 0.0f);
                    if (qg[k] > 1.0e-9f) {         // ngmlt (:940-943)
                        float gfac = gg.r * gcn.n0g / qg[k];
                        nr[k] = nr[k] - gfac * pgmlt;
                    }
                    qg[k] += pgmlt; qr[k] -= pgmlt;
                    t[k] += xlf / cpm[k] * pgmlt;
                }
            }
        }

        // ---- cloud-ice fallout, HDC 5a speed on the PRE-sediment xni
        // (:955-983) --------------------------------------------------------
        {
            float wic[WDM6_KMAX];
            for (int k = nz - 1; k >= 0; --k) {
                if (qi[k] <= 0.0f) {
                    wic[k] = 0.0f;
                } else {
                    float xmi = den[k] * qi[k] / xni_pre[k];
                    float diameter = fmaxf(
                        fminf(11.9f * sqrtf(xmi), 500.0e-6f), 1.0e-25f);
                    wic[k] = 1.49e4f * WDM6_EXPLOG(diameter, 1.31f);
                }
            }
            float pi_ice = wdm6_sediment_ice(qi, wic, den, delz, nz, dtcld);
            float fallc_sfc = pi_ice / delz[0] / dtcld;    // :981-983

            // ---- surface precipitation accounting (:988-1017) -------------
            float fallsum = fall_r_sfc + fall2_sfc + fall3_sfc + fallc_sfc;
            float fallsum_qsi = fall2_sfc + fallc_sfc;
            float fallsum_qg = fall3_sfc;
            if (fallsum > 0.0f) {
                float step = __fdiv_rn(fallsum * delz[0], 1000.0f) * dtcld * 1000.0f;
                rainncv[col] += step;
                rainnc[col] += step;
            }
            if (fallsum_qsi > 0.0f) {
                float step = __fdiv_rn(fallsum_qsi * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                snowncv[col] += step;
                snownc[col] += step;
            }
            if (fallsum_qg > 0.0f) {
                float step = __fdiv_rn(fallsum_qg * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                graupelncv[col] += step;
                graupelnc[col] += step;
            }
            if (fallsum > 0.0f)                    // :1012-1013
                sr[col] = (snowncv[col] + graupelncv[col])
                        / (rainncv[col] + 1.0e-12f);
        }

        // ---- instantaneous melting/freezing (:1023-1104) -------------------
        for (int k = 0; k < nz; ++k) {
            // rslopec from the loop-start qc/nc (:759-769): nothing has
            // touched qc/nc yet this minor loop, so the live values equal
            // the Fortran's precomputed array as long as they are read
            // BEFORE this iteration's own updates.
            float rslopec3;
            if (qc[k] <= 1.0e-15f || nc[k] <= 1.0e1f) {
                rslopec3 = WDM6_RSLOPEC3MAX;
            } else {
                float lamdac = WDM6_EXPLOG((WDM6_PIDNC * nc[k])
                                           / (qc[k] * den[k]), 0.33333333f);
                float rc1 = 1.0f / lamdac;
                rslopec3 = rc1 * rc1 * rc1;
            }
            float supcol = 273.15f - t[k];
            float xlf = 2.85e6f - xl[k];
            if (supcol < 0.0f) xlf = 3.5e5f;
            if (supcol < 0.0f && qi[k] > 0.0f) {   // pimlt/nimlt
                qc[k] += qi[k];
                nc[k] += xni_pre[k];
                t[k] -= xlf / cpm[k] * qi[k];
                qi[k] = 0.0f;
            }
            if (supcol > 40.0f && qc[k] > 0.0f) {  // pihmf/nihmf
                qi[k] += qc[k];
                if (nc[k] > 0.0f) nc[k] = 0.0f;
                t[k] += xlf / cpm[k] * qc[k];
                qc[k] = 0.0f;
            }
            if (supcol > 0.0f && qc[k] > 1.0e-15f) {   // pihtf/nihtf
                float supcolt = fminf(supcol, 70.0f);
                float pfrzdtc = fminf(
                    __fdiv_rn(WDM6_PI * WDM6_PI * 100.0f
                    * (gfk_exp(0.66f * supcolt) - 1.0f)
                    * 1000.0f / den[k] * nc[k] * rslopec3 * rslopec3,
                    18.0f) * dtcld, qc[k]);
                if (nc[k] > 1.0e1f) {
                    float nfrzdtc = fminf(
                        __fdiv_rn(WDM6_PI * 100.0f
                        * (gfk_exp(0.66f * supcolt) - 1.0f) * nc[k]
                        * rslopec3, 6.0f) * dtcld, nc[k]);
                    nc[k] -= nfrzdtc;
                }
                qi[k] += pfrzdtc;
                t[k] += xlf / cpm[k] * pfrzdtc;
                qc[k] -= pfrzdtc;
            }
            if (supcol > 0.0f && qr[k] > 0.0f) {   // pgfrz/ngfrz
                float supcolt = fminf(supcol, 70.0f);
                // rslope3(i,k,1) is the :893 refresh captured above.
                float pfrzdtr = fminf(
                    140.0f * (WDM6_PI * WDM6_PI) * 100.0f * nr[k]
                    * 1000.0f / den[k] * (gfk_exp(0.66f * supcolt) - 1.0f)
                    * rslope3r[k] * rslope3r[k] * dtcld, qr[k]);
                if (nr[k] > 1.0e-2f) {
                    float nfrzdtr = fminf(
                        4.0f * WDM6_PI * 100.0f * nr[k]
                        * (gfk_exp(0.66f * supcolt) - 1.0f) * rslope3r[k]
                        * dtcld, nr[k]);
                    nr[k] -= nfrzdtr;
                }
                qg[k] += pfrzdtr;
                t[k] += xlf / cpm[k] * pfrzdtr;
                qr[k] -= pfrzdtr;
            }
            nc[k] = fmaxf(nc[k], 0.0f);            // :1099-1104
            nr[k] = fmaxf(nr[k], 0.0f);
        }

        // ---- warm + cold process rates, conservation, and update
        // (:1109-1881).  Every Fortran loop in this span is pointwise, so
        // the fusion preserves each cell's exact statement order. ----------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            const float dfac = wdm6_denfac(den[k]);
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k], dfac);
            float n0sfac;
            WdmSlope ss = wdm6_snow_slope(qs[k], den[k], dfac, t[k],
                                          &n0sfac);
            WdmSlope gg = wdm6_graupel_slope(qg[k], den[k], dfac, gcn);
            // mean-volume diameters (:1125/:1140) and cloud slope refresh.
            // (24.)**(.3333333) is folded by gfortran: 0x40389BA2.
            float avedia2 = rr.r * 2.88449907f;
            float rslopec, rslopec2, rslopec3;
            if (qc[k] <= 1.0e-15f || nc[k] <= 1.0e1f) {
                rslopec = WDM6_RSLOPECMAX; rslopec2 = WDM6_RSLOPEC2MAX;
                rslopec3 = WDM6_RSLOPEC3MAX;
            } else {
                float lamdac = WDM6_EXPLOG((WDM6_PIDNC * nc[k])
                                           / (qc[k] * den[k]), 0.33333333f);
                rslopec = 1.0f / lamdac;
                rslopec2 = rslopec * rslopec;
                rslopec3 = rslopec2 * rslopec;
            }
            float avedia1 = rslopec;
            float workw = wdm6_diffac(xl[k], p[id], t[k], den[k], qsw[k]);
            float worki = wdm6_diffac(2.85e6f, p[id], t[k], den[k], qsi[k]);
            float vf = wdm6_venfac(p[id], t[k], den[k]);

            float prevp=0, psdep=0, pgdep=0, praut=0, psaut=0, pgaut=0;
            float pracw=0, praci=0, piacr=0, psaci=0, psacw=0, pracs=0;
            float psacr=0, pgacw=0, paacw=0, pgaci=0, pgacr=0, pgacs=0;
            float pigen=0, pidep=0, pseml=0, pgeml=0, psevp=0, pgevp=0;
            float nraut=0, nracw=0, nccol=0, nrcol=0, nsacw=0, ngacw=0;
            float niacr=0, nsacr=0, ngacr=0, naacw=0, nseml=0, ngeml=0;

            // ==== warm rain (:1160-1259) ====
            float supsat = fmaxf(qv[k], 1.0e-15f) - qsw[k];
            float satdt = supsat / dtcld;
            float lencon = 2.7e-2f * den[k] * qc[k]
                * (1.0e20f / 16.0f * rslopec2 * rslopec2 - 0.4f);
            float lenconcr = fmaxf(1.2f * lencon, 1.0e-9f);
            if (qc[k] > qcr_col && nc[k] > 1.0e1f) {       // praut/nraut
                praut = WDM6_QCK1 * gfk_pow(qc[k], 7.0f / 3.0f)
                      * gfk_pow(nc[k], -1.0f / 3.0f);
                praut = fminf(praut, qc[k] / dtcld);
                nraut = 3.5e9f * den[k] * praut;
                if (qr[k] > lenconcr)
                    nraut = nr[k] / qr[k] * praut;
                nraut = fminf(nraut, nc[k] / dtcld);
            }
            if (qr[k] >= lenconcr) {                       // pracw/nracw
                if (avedia2 >= 1.0e-4f) {
                    nracw = fminf(3.03e3f * nc[k] * nr[k]
                                  * (rslopec3 + 24.0f * rr.r3),
                                  nc[k] / dtcld);
                    pracw = fminf(WDM6_PI / 6.0f
                                  * (1000.0f / den[k]) * 3.03e3f
                                  * nc[k] * nr[k] * rslopec3
                                  * (2.0f * rslopec3 + 24.0f * rr.r3),
                                  qc[k] / dtcld);
                } else {
                    nracw = fminf(2.59e15f * nc[k] * nr[k]
                                  * (2.0f * rslopec3 * rslopec3
                                     + 5040.0f * rr.r3 * rr.r3),
                                  nc[k] / dtcld);
                    pracw = fminf(WDM6_PI / 6.0f
                                  * (1000.0f / den[k]) * 2.59e15f
                                  * nc[k] * nr[k] * rslopec3
                                  * (6.0f * rslopec3 * rslopec3
                                     + 5040.0f * rr.r3 * rr.r3),
                                  qc[k] / dtcld);
                }
            }
            if (avedia1 >= 1.0e-4f) {                      // nccol
                nccol = 3.03e3f * nc[k] * nc[k] * rslopec3;
            } else {
                nccol = 2.0f * 2.59e15f * nc[k] * nc[k]
                      * rslopec3 * rslopec3;
            }
            if (qr[k] >= lenconcr) {                       // nrcol
                if (avedia2 < 1.0e-4f) {
                    nrcol = 5040.0f * 2.59e15f * nr[k] * nr[k]
                          * rr.r3 * rr.r3;
                } else if (avedia2 < 6.0e-4f) {
                    nrcol = 24.0f * 3.03e3f * nr[k] * nr[k] * rr.r3;
                } else if (avedia2 < 2000.0e-6f) {
                    float coecol = -2.5e3f * (avedia2 - 6.0e-4f);
                    nrcol = 24.0f * gfk_exp(coecol) * 3.03e3f
                          * nr[k] * nr[k] * rr.r3;
                } else {
                    nrcol = 0.0f;
                }
            }
            if (qr[k] > 0.0f) {                            // prevp/nrevp
                float coeres = rr.r * sqrtf(rr.r * rr.rb);
                prevp = (rhw[k] - 1.0f) * nr[k]
                      * (WDM6_PRECR1 * rr.r
                         + WDM6_PRECR2 * vf * coeres) / workw;
                if (prevp < 0.0f) {
                    prevp = fmaxf(prevp, -qr[k] / dtcld);
                    prevp = fmaxf(prevp, satdt / 2.0f);
                    if (prevp == -qr[k] / dtcld) {         // :1249-1252
                        nn[k] += nr[k];
                        nr[k] = 0.0f;
                    }
                } else {
                    // DELIBERATE, DOCUMENTED DIVERGENCE from :1255 (A144).
                    // This branch caps CONDENSATION at half the water
                    // supersaturation.  WRF also enters it with a rate of
                    // exactly zero -- rain carrying no number (nr = 0) makes
                    // the product -0, and -0 < 0 is false -- and in
                    // subsaturated air min(0, satdt/2) then turns "no drops"
                    // into evaporation of half the saturation deficit,
                    // never bounded by qr (that bound, :1243, sits in the
                    // other branch).  pidep, psdep, pgdep and pigen count
                    // that vapour source in supice (:1527 on) and deposit
                    // against it; the rain limiter (:1670-1680) shrinks
                    // prevp back to the tiny qr but never revisits the
                    // deposition, so vapour goes negative.  WRF's own
                    // Fortran does exactly that on the captured column
                    // (docs/wdm6_oracle_known_deltas.md, section 6).  The
                    // cap keeps its meaning and never changes the sign: a
                    // zero rate stays zero, and the numberless rain joins
                    // the cloud water in the di82 collapse at :1938.
                    prevp = fmaxf(fminf(prevp, satdt / 2.0f), 0.0f);
                }
            }

            // ==== cold rain (:1274-1625) ====
            float supcol = 273.15f - t[k];
            n0sfac = fmaxf(fminf(gfk_exp(0.12f * supcol), 5.0e4f), 1.0f);
            supsat = fmaxf(qv[k], 1.0e-15f) - qsi[k];
            satdt = supsat / dtcld;
            int ifsat = 0;
            float temp = den[k] * fmaxf(qi[k], 1.0e-15f);
            temp = sqrtf(sqrtf(temp * temp * temp));
            float xni = fminf(fmaxf(5.38e7f * temp, 1.0e3f), 1.0e6f);
            float eacrs = gfk_exp(0.07f * (-supcol));
            float xmi = den[k] * qi[k] / xni;
            float diameter = fminf(11.9f * sqrtf(xmi), 500.0e-6f);
            float vt2i = 1.49e4f * gfk_pow(diameter, 1.31f);
            float vt2r = WDM6_PVTR * rr.rb * dfac;
            float vt2s = WDM6_PVTS * ss.rb * dfac;
            float vt2g = gcn.pvtg * gg.rb * dfac;
            float qsum = fmaxf(qs[k] + qg[k], 1.0e-15f);
            float vt2ave = qsum > 1.0e-15f
                         ? (vt2s * qs[k] + vt2g * qg[k]) / qsum : 0.0f;
            if (supcol > 0.0f && qi[k] > 1.0e-15f) {
                if (qr[k] > 1.0e-9f) {             // praci/piacr
                    float acrfac = 6.0f * rr.r2 + 4.0f * diameter * rr.r
                                 + diameter * diameter;
                    praci = WDM6_PI * qi[k] * nr[k]
                          * fabsf(vt2r - vt2i) * acrfac / 4.0f;
                    praci *= wdm6_eff01(qr[k] / qi[k]);
                    praci = fminf(praci, qi[k] / dtcld);
                    piacr = __fdiv_rn(WDM6_PI * WDM6_PI * 841.9f * nr[k]
                          * 1000.0f * xni * dfac * WDM6_G7PBR * rr.r3 * rr.r2
                          * rr.rb, 24.0f) / den[k];
                    piacr *= wdm6_eff01(qi[k] / qr[k]);
                    piacr = fminf(piacr, qr[k] / dtcld);
                }
                if (nr[k] > 1.0e-2f) {             // niacr (:1329-1335)
                    niacr = WDM6_PI * 841.9f * nr[k] * xni * dfac
                          * WDM6_G4PBR * rr.r2 * rr.rb / 4.0f;
                    niacr *= wdm6_eff01(qi[k] / qr[k]);
                    niacr = fminf(niacr, nr[k] / dtcld);
                }
                if (qs[k] > 1.0e-9f) {             // psaci
                    float acrfac = 2.0f * ss.r3 + 2.0f * diameter * ss.r2
                                 + diameter * diameter * ss.r;
                    psaci = WDM6_PI * qi[k] * eacrs * 2.0e6f * n0sfac
                          * fabsf(vt2ave - vt2i) * acrfac / 4.0f;
                    psaci = fminf(psaci, qi[k] / dtcld);
                }
                if (qg[k] > 1.0e-9f) {             // pgaci
                    float egi = gfk_exp(0.07f * (-supcol));
                    float acrfac = 2.0f * gg.r3 + 2.0f * diameter * gg.r2
                                 + diameter * diameter * gg.r;
                    pgaci = WDM6_PI * egi * qi[k] * gcn.n0g
                          * fabsf(vt2ave - vt2i) * acrfac / 4.0f;
                    pgaci = fminf(pgaci, qi[k] / dtcld);
                }
            }
            if (qs[k] > 1.0e-9f && qc[k] > 1.0e-15f)       // psacw
                psacw = fminf(WDM6_PACRC * n0sfac * ss.r3 * ss.rb
                              * wdm6_eff01(qs[k] / qc[k])
                              * qc[k] * dfac, qc[k] / dtcld);
            if (qs[k] > 1.0e-9f && nc[k] > 1.0e1f)         // nsacw
                nsacw = fminf(WDM6_PACRC * n0sfac * ss.r3 * ss.rb
                              * wdm6_eff01(qs[k] / qc[k])
                              * nc[k] * dfac, nc[k] / dtcld);
            if (qg[k] > 1.0e-9f && qc[k] > 1.0e-15f)       // pgacw
                pgacw = fminf(gcn.pacrg * gg.r3 * gg.rb * qc[k]
                              * wdm6_eff01(qg[k] / qc[k])
                              * dfac, qc[k] / dtcld);
            if (qg[k] > 1.0e-9f && nc[k] > 1.0e1f)         // ngacw
                ngacw = fminf(gcn.pacrg * gg.r3 * gg.rb * nc[k]
                              * wdm6_eff01(qg[k] / qc[k])
                              * dfac, nc[k] / dtcld);
            if (qsum > 1.0e-15f) {                         // paacw/naacw
                paacw = (qs[k] * psacw + qg[k] * pgacw) / qsum;
                naacw = (qs[k] * nsacw + qg[k] * ngacw) / qsum;
            }
            if (qs[k] > 1.0e-9f && qr[k] > 1.0e-9f) {
                if (supcol > 0.0f) {               // pracs
                    float acrfac = 5.0f * ss.r3 * ss.r3
                                 + 4.0f * ss.r3 * ss.r2 * rr.r
                                 + 1.5f * ss.r2 * ss.r2 * rr.r2;
                    pracs = WDM6_PI * WDM6_PI * nr[k] * 2.0e6f * n0sfac
                          * fabsf(vt2r - vt2ave) * (100.0f / den[k])
                          * acrfac;
                    pracs *= wdm6_eff01(qr[k] / qs[k]);
                    pracs = fminf(pracs, qs[k] / dtcld);
                }
                // psacr
                float acrfac = 30.0f * rr.r3 * rr.r2 * ss.r
                             + 10.0f * rr.r2 * rr.r2 * ss.r2
                             + 2.0f * rr.r3 * ss.r3;
                psacr = WDM6_PI * WDM6_PI * nr[k] * 2.0e6f * n0sfac
                      * fabsf(vt2ave - vt2r) * (1000.0f / den[k])
                      * acrfac;
                psacr *= wdm6_eff01(qs[k] / qr[k]);
                psacr = fminf(psacr, qr[k] / dtcld);
            }
            if (qs[k] > 1.0e-9f && nr[k] > 1.0e-2f) {      // nsacr
                float acrfac = 1.5f * rr.r2 * ss.r + 1.0f * rr.r * ss.r2
                             + 0.5f * ss.r3;
                nsacr = WDM6_PI * nr[k] * 2.0e6f * n0sfac
                      * fabsf(vt2ave - vt2r) * acrfac;
                nsacr *= wdm6_eff01(qs[k] / qr[k]);
                nsacr = fminf(nsacr, nr[k] / dtcld);
            }
            if (qg[k] > 1.0e-9f && qr[k] > 1.0e-9f) {      // pgacr
                float acrfac = 30.0f * rr.r3 * rr.r2 * gg.r
                             + 10.0f * rr.r2 * rr.r2 * gg.r2
                             + 2.0f * rr.r3 * gg.r3;
                pgacr = WDM6_PI * WDM6_PI * nr[k] * gcn.n0g
                      * fabsf(vt2ave - vt2r) * (1000.0f / den[k])
                      * acrfac;
                pgacr *= wdm6_eff01(qg[k] / qr[k]);
                pgacr = fminf(pgacr, qr[k] / dtcld);
            }
            if (qg[k] > 1.0e-9f && nr[k] > 1.0e-2f) {      // ngacr
                float acrfac = 1.5f * rr.r2 * gg.r + 1.0f * rr.r * gg.r2
                             + 0.5f * gg.r3;
                ngacr = WDM6_PI * nr[k] * gcn.n0g
                      * fabsf(vt2ave - vt2r) * acrfac;
                ngacr *= wdm6_eff01(qg[k] / qr[k]);
                ngacr = fminf(ngacr, nr[k] / dtcld);
            }
            pgacs = 0.0f;                                  // :1484-1486
            if (supcol <= 0.0f) {                          // pseml/pgeml
                float xlf = 3.5e5f;
                if (qs[k] > 0.0f) {
                    pseml = fminf(fmaxf(__fdiv_rn(4190.0f * supcol
                                        * (paacw + psacr), xlf),
                                        -qs[k] / dtcld), 0.0f);
                    if (qs[k] > 1.0e-9f) {         // nseml (:1500-1503)
                        float sfac = ss.r * 2.0e6f * n0sfac / qs[k];
                        nseml = -sfac * pseml;
                    }
                }
                if (qg[k] > 0.0f) {
                    pgeml = fminf(fmaxf(__fdiv_rn(4190.0f * supcol
                                        * (paacw + pgacr), xlf),
                                        -qg[k] / dtcld), 0.0f);
                    if (qg[k] > 1.0e-9f) {         // ngeml (:1515-1518)
                        float gfac = gg.r * gcn.n0g / qg[k];
                        ngeml = -gfac * pgeml;
                    }
                }
            }
            if (supcol > 0.0f) {
                if (qi[k] > 0.0f && ifsat != 1) {          // pidep
                    pidep = 4.0f * diameter * xni * (rhi[k] - 1.0f)
                          / worki;
                    float supice = satdt - prevp;
                    if (pidep < 0.0f) {
                        pidep = fmaxf(fmaxf(pidep, satdt / 2.0f), supice);
                        pidep = fmaxf(pidep, -qi[k] / dtcld);
                    } else {
                        pidep = fminf(fminf(pidep, satdt / 2.0f), supice);
                    }
                    if (fabsf(prevp + pidep) >= fabsf(satdt)) ifsat = 1;
                }
                if (qs[k] > 0.0f && ifsat != 1) {          // psdep
                    float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                    psdep = (rhi[k] - 1.0f) * n0sfac
                          * (WDM6_PRECS1 * ss.r2
                             + WDM6_PRECS2 * vf * coeres) / worki;
                    float supice = satdt - prevp - pidep;
                    if (psdep < 0.0f) {
                        psdep = fmaxf(psdep, -qs[k] / dtcld);
                        psdep = fmaxf(fmaxf(psdep, satdt / 2.0f), supice);
                    } else {
                        psdep = fminf(fminf(psdep, satdt / 2.0f), supice);
                    }
                    if (fabsf(prevp + pidep + psdep) >= fabsf(satdt))
                        ifsat = 1;
                }
                if (qg[k] > 0.0f && ifsat != 1) {          // pgdep
                    float coeres = gg.r2 * sqrtf(gg.r * gg.rb);
                    pgdep = (rhi[k] - 1.0f)
                          * (gcn.precg1 * gg.r2
                             + gcn.precg2 * vf * coeres) / worki;
                    float supice = satdt - prevp - pidep - psdep;
                    if (pgdep < 0.0f) {
                        pgdep = fmaxf(pgdep, -qg[k] / dtcld);
                        pgdep = fmaxf(fmaxf(pgdep, satdt / 2.0f), supice);
                    } else {
                        pgdep = fminf(fminf(pgdep, satdt / 2.0f), supice);
                    }
                    if (fabsf(prevp + pidep + psdep + pgdep)
                        >= fabsf(satdt)) ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {         // pigen
                    float supice = satdt - prevp - pidep - psdep - pgdep;
                    float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
                    float roqi0 = 4.92e-11f * gfk_pow(xni0, 1.33f);
                    pigen = fmaxf(0.0f, (roqi0 / den[k]
                                         - fmaxf(qi[k], 0.0f)) / dtcld);
                    pigen = fminf(fminf(pigen, satdt), supice);
                }
                if (qi[k] > 0.0f) {                        // psaut
                    float qimax = WDM6_ROQIMAX / den[k];
                    psaut = fmaxf(0.0f, (qi[k] - qimax) / dtcld);
                }
                if (qs[k] > 0.0f) {                        // pgaut
                    float alpha2 = 1.0e-3f * gfk_exp(0.09f * (-supcol));
                    pgaut = fminf(fmaxf(0.0f, alpha2 * (qs[k] - 6.0e-4f)),
                                  qs[k] / dtcld);
                }
            }
            if (supcol < 0.0f) {                           // psevp/pgevp
                if (qs[k] > 0.0f && rhw[k] < 1.0f) {
                    float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                    psevp = (rhw[k] - 1.0f) * n0sfac
                          * (WDM6_PRECS1 * ss.r2
                             + WDM6_PRECS2 * vf * coeres) / workw;
                    psevp = fminf(fmaxf(psevp, -qs[k] / dtcld), 0.0f);
                }
                if (qg[k] > 0.0f && rhw[k] < 1.0f) {
                    float coeres = gg.r2 * sqrtf(gg.r * gg.rb);
                    pgevp = (rhw[k] - 1.0f)
                          * (gcn.precg1 * gg.r2
                             + gcn.precg2 * vf * coeres) / workw;
                    pgevp = fminf(fmaxf(pgevp, -qg[k] / dtcld), 0.0f);
                }
            }

            // ==== conservation limiting and the coupled update
            // (:1632-1881) ====
            float delta2 = (qr[k] < 1.0e-4f && qs[k] < 1.0e-4f)
                         ? 1.0f : 0.0f;
            float delta3 = (qr[k] < 1.0e-4f) ? 1.0f : 0.0f;
#define WDM6_LIMIT(value_, source_, ...) do { \
    float _src = (source_) * dtcld; \
    if (_src > (value_)) { float factor = (value_) / _src; __VA_ARGS__; } \
} while (0)
            if (t[k] <= 273.15f) {
                WDM6_LIMIT(fmaxf(1.0e-15f, qc[k]),
                           praut + pracw + paacw + paacw,
                           praut *= factor; pracw *= factor;
                           paacw *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-15f, qi[k]),
                           psaut - pigen - pidep + praci + psaci + pgaci,
                           psaut *= factor; pigen *= factor;
                           pidep *= factor; praci *= factor;
                           psaci *= factor; pgaci *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-15f, qr[k]),
                           -praut - prevp - pracw + piacr + psacr + pgacr,
                           praut *= factor; prevp *= factor;
                           pracw *= factor; piacr *= factor;
                           psacr *= factor; pgacr *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-15f, qs[k]),
                           -(psdep + psaut - pgaut + paacw
                             + piacr * delta3 + praci * delta3
                             - pracs * (1.0f - delta2) + psacr * delta2
                             + psaci - pgacs),
                           psdep *= factor; psaut *= factor;
                           pgaut *= factor; paacw *= factor;
                           piacr *= factor; praci *= factor;
                           psaci *= factor; pracs *= factor;
                           psacr *= factor; pgacs *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-15f, qg[k]),
                           -(pgdep + pgaut
                             + piacr * (1.0f - delta3)
                             + praci * (1.0f - delta3)
                             + psacr * (1.0f - delta2)
                             + pracs * (1.0f - delta2)
                             + pgaci + paacw + pgacr + pgacs),
                           pgdep *= factor; pgaut *= factor;
                           piacr *= factor; praci *= factor;
                           psacr *= factor; pracs *= factor;
                           paacw *= factor; pgaci *= factor;
                           pgacr *= factor; pgacs *= factor;);
                WDM6_LIMIT(fmaxf(1.0e1f, nc[k]),
                           nraut + nccol + nracw + naacw + naacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; naacw *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-2f, nr[k]),
                           -nraut + nrcol + niacr + nsacr + ngacr,
                           nraut *= factor; nrcol *= factor;
                           niacr *= factor; nsacr *= factor;
                           ngacr *= factor;);
                float work2k = -(prevp + psdep + pgdep + pigen + pidep);
                qv[k] += work2k * dtcld;
                qc[k] = fmaxf(qc[k] - (praut + pracw + paacw + paacw)
                              * dtcld, 0.0f);
                qr[k] = fmaxf(qr[k] + (praut + pracw + prevp - piacr
                                       - pgacr - psacr) * dtcld, 0.0f);
                qi[k] = fmaxf(qi[k] - (psaut + praci + psaci + pgaci
                                       - pigen - pidep) * dtcld, 0.0f);
                qs[k] = fmaxf(qs[k] + (psdep + psaut + paacw - pgaut
                                       + piacr * delta3 + praci * delta3
                                       + psaci - pgacs
                                       - pracs * (1.0f - delta2)
                                       + psacr * delta2) * dtcld, 0.0f);
                qg[k] = fmaxf(qg[k] + (pgdep + pgaut
                                       + piacr * (1.0f - delta3)
                                       + praci * (1.0f - delta3)
                                       + psacr * (1.0f - delta2)
                                       + pracs * (1.0f - delta2)
                                       + pgaci + paacw + pgacr + pgacs)
                              * dtcld, 0.0f);
                nc[k] = fmaxf(nc[k] + (-nraut - nccol - nracw - naacw
                                       - naacw) * dtcld, 0.0f);
                nr[k] = fmaxf(nr[k] + (nraut - nrcol - niacr - nsacr
                                       - ngacr) * dtcld, 0.0f);
                float xlf = 2.85e6f - xl[k];
                float xlwork2 = -2.85e6f * (psdep + pgdep + pidep + pigen)
                              - xl[k] * prevp
                              - xlf * (piacr + paacw + paacw + pgacr
                                       + psacr);
                t[k] -= xlwork2 / cpm[k] * dtcld;
            } else {
                WDM6_LIMIT(fmaxf(1.0e-15f, qc[k]),
                           praut + pracw + paacw + paacw,
                           praut *= factor; pracw *= factor;
                           paacw *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-15f, qr[k]),
                           -paacw - praut + pseml + pgeml - pracw
                           - paacw - prevp,
                           praut *= factor; prevp *= factor;
                           pracw *= factor; paacw *= factor;
                           pseml *= factor; pgeml *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-9f, qs[k]),
                           pgacs - pseml - psevp,
                           pgacs *= factor; psevp *= factor;
                           pseml *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-9f, qg[k]),
                           -(pgacs + pgevp + pgeml),
                           pgacs *= factor; pgevp *= factor;
                           pgeml *= factor;);
                WDM6_LIMIT(fmaxf(1.0e1f, nc[k]),
                           nraut + nccol + nracw + naacw + naacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; naacw *= factor;);
                WDM6_LIMIT(fmaxf(1.0e-2f, nr[k]),
                           -nraut + nrcol - nseml - ngeml,
                           nraut *= factor; nrcol *= factor;
                           nseml *= factor; ngeml *= factor;);
                float work2k = -(prevp + psevp + pgevp);
                qv[k] += work2k * dtcld;
                qc[k] = fmaxf(qc[k] - (praut + pracw + paacw + paacw)
                              * dtcld, 0.0f);
                qr[k] = fmaxf(qr[k] + (praut + pracw + prevp + paacw
                                       + paacw - pseml - pgeml) * dtcld,
                              0.0f);
                qs[k] = fmaxf(qs[k] + (psevp - pgacs + pseml) * dtcld,
                              0.0f);
                qg[k] = fmaxf(qg[k] + (pgacs + pgevp + pgeml) * dtcld,
                              0.0f);
                nc[k] = fmaxf(nc[k] + (-nraut - nccol - nracw - naacw
                                       - naacw) * dtcld, 0.0f);
                nr[k] = fmaxf(nr[k] + (nraut - nrcol + nseml + ngeml)
                              * dtcld, 0.0f);
                float xlf = 2.85e6f - xl[k];
                float xlwork2 = -xl[k] * (prevp + psevp + pgevp)
                              - xlf * (pseml + pgeml);
                t[k] -= xlwork2 / cpm[k] * dtcld;
            }
#undef WDM6_LIMIT
        }

        // ---- post-update saturation refresh (:1886-1914): qs1/qs2 and
        // rh1 only; rh2 keeps its loop-start value exactly as in WRF. ------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = wdm6_qsat(t[k], p[id], false);
            qsi[k] = wdm6_qsat(t[k], p[id], true);
            rhw[k] = fmaxf(qv[k] / qsw[k], 1.0e-15f);
        }

        // ---- small-drop collapse, CCN activation, cloud condensation,
        // and the padding clamps (:1916-2032; all pointwise) ---------------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            // Nrevp_s / Prevp_s (:1925-1948): all rain collapses to cloud
            // when the mean-volume raindrop shrinks below di82.
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k],
                                              wdm6_denfac(den[k]));
            float avedia2 = rr.r * 2.88449907f;
            if (avedia2 <= 82.0e-6f) {
                nc[k] += nr[k];
                nr[k] = 0.0f;
                qc[k] += qr[k];
                qr[k] = 0.0f;
            }
            // CCN activation (:1951-1969).
            if (rhw[k] > 1.0f) {
                float ncact = fmaxf(0.0f,
                    (nn[k] + nc[k])
                    * fminf(1.0f, gfk_pow(__fdiv_rn(rhw[k], 1.0048f), 0.6f))
                    - nc[k]) / dtcld;
                ncact = fminf(ncact, fmaxf(nn[k], 0.0f) / dtcld);
                float pcact = fminf(
                    4.0f * WDM6_PI * 1000.0f * WDM6_ACTR3 * ncact
                    / (3.0f * den[k]),
                    fmaxf(qv[k], 0.0f) / dtcld);
                qv[k] = fmaxf(qv[k] - pcact * dtcld, 0.0f);
                qc[k] = fmaxf(qc[k] + pcact * dtcld, 0.0f);
                nn[k] = fmaxf(nn[k] - ncact * dtcld, 0.0f);
                nc[k] = fmaxf(nc[k] + ncact * dtcld, 0.0f);
                t[k] += pcact * xl[k] / cpm[k] * dtcld;
            }
            // pcond/ncevp (:1976-1998).
            float qs1 = wdm6_qsat(t[k], p[id], false);
            float work1c = (fmaxf(qv[k], 1.0e-15f) - qs1)
                / (1.0f + xl[k] * xl[k] / (461.6f * cpm[k]) * qs1
                   / (t[k] * t[k]));
            float pcond = fminf(fmaxf(work1c / dtcld, 0.0f),
                                fmaxf(qv[k], 0.0f) / dtcld);
            if (qc[k] > 0.0f && work1c < 0.0f)
                pcond = fmaxf(work1c, -qc[k]) / dtcld;
            if (pcond == -qc[k] / dtcld) {         // ncevp (:1990-1994)
                nn[k] += nc[k];
                nc[k] = 0.0f;
            }
            qv[k] -= pcond * dtcld;
            qc[k] = fmaxf(qc[k] + pcond * dtcld, 0.0f);
            t[k] += pcond * xl[k] / cpm[k] * dtcld;

            // padding for small values (:2005-2030), with WDM6's
            // number-adjusting rain/cloud slope clamps.
            if (qc[k] <= 1.0e-15f) qc[k] = 0.0f;
            if (qi[k] <= 1.0e-15f) qi[k] = 0.0f;
            if (qr[k] >= 1.0e-9f && nr[k] >= 1.0e-2f) {
                // lamdr_tmp**3 is one integer power, (l*l)*l, multiplied in
                // after den*qr (:2014/:2017).
                float lamdr = WDM6_EXPLOG((WDM6_PIDNR * nr[k])
                                          / (den[k] * qr[k]), 0.33333333f);
                if (lamdr <= 2.0e3f) {
                    lamdr = 2.0e3f;
                    nr[k] = __fdiv_rn(den[k] * qr[k] * (lamdr * lamdr * lamdr),
                          WDM6_PIDNR);
                } else if (lamdr >= 5.0e4f) {
                    lamdr = 5.0e4f;
                    nr[k] = __fdiv_rn(den[k] * qr[k] * (lamdr * lamdr * lamdr),
                          WDM6_PIDNR);
                }
            }
            if (qc[k] >= 1.0e-15f && nc[k] >= 1.0e1f) {
                float lamdc = WDM6_EXPLOG((WDM6_PIDNC * nc[k])
                                          / (den[k] * qc[k]), 0.33333333f);
                if (lamdc <= 2.0e4f) {
                    lamdc = 2.0e4f;
                    nc[k] = __fdiv_rn(den[k] * qc[k] * (lamdc * lamdc * lamdc),
                          WDM6_PIDNC);
                } else if (lamdc >= 5.0e5f) {
                    lamdc = 5.0e5f;
                    nc[k] = __fdiv_rn(den[k] * qc[k] * (lamdc * lamdc * lamdc),
                          WDM6_PIDNC);
                }
            }
        }
    }

    // ---- writeback + effectRad_wdm6 (:3135-3234) with the wdm6 driver's
    // clamp (:321-325).  The radii are formed in WRF's metres exactly as
    // the Fortran forms them and only then scaled to gpuwm's
    // radiation-facing microns by the one float32 multiply WRF's RRTMG
    // applies itself (re_cloud*1.E6, module_ra_rrtmg_lw.F/_sw.F), so the
    // micron word is WRF's radiation input word. -----------------------------
    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        theta[id] = t[k] / pii[id];
        qv_g[id] = qv[k]; qc_g[id] = qc[k]; qi_g[id] = qi[k];
        qr_g[id] = qr[k]; qs_g[id] = qs[k]; qg_g[id] = qg[k];
        nn_g[id] = nn[k]; nc_g[id] = nc[k]; nr_g[id] = nr[k];
        // The driver hands effectRad t1d = th*pii (:309) from the theta it
        // just stored, which is not always t to the last bit.
        const float tk = theta[id] * pii[id];
        const float rho = den[k];
        // cloud: gamma PSD with the prognostic nc (:3193-3195, :3208-3214);
        // lamc is DOUBLE PRECISION in WRF, assigned from a REAL product,
        // and 1/lamc is taken in double before sngl.
        float re_qc = 2.49e-6f;                           // RE_QC_BG
        float rqc = fmaxf(1.0e-12f, qc[k] * rho);
        float rnc = fmaxf(1.0e-6f, nc[k] * rho);
        if (rqc > 1.0e-12f && rnc > 1.0e-6f) {
            float lamc = 2.0f * WDM6_CDM2
                * gfk_pow(WDM6_PIDNC * nc[k] / rqc, 1.0f / 3.0f);
            re_qc = fmaxf(2.51e-6f,
                          fminf((float)(1.0 / (double)lamc), 50.0e-6f));
        }
        // ice: WSM6's diagnostic number (:3196-3202, :3216-3222).
        float re_qi = 4.99e-6f;                           // RE_QI_BG
        float rqi = fmaxf(1.0e-12f, qi[k] * rho);
        float tempi = rho * fmaxf(qi[k], 1.0e-15f);
        tempi = sqrtf(sqrtf(tempi * tempi * tempi));
        float ni = fminf(fmaxf(5.38e7f * tempi, 1.0e3f), 1.0e6f);
        float rni = fmaxf(1.0e-6f, ni * rho);
        if (rqi > 1.0e-12f && rni > 1.0e-6f) {
            float diai = 11.9f * sqrtf(rqi / ni);
            re_qi = fmaxf(10.01e-6f, fminf(0.75f * 0.163f * diai, 125.0e-6f));
        }
        // snow (:3204-3205, :3224-3232).
        float re_qs = 9.99e-6f;                           // RE_QS_BG
        float rqs = fmaxf(1.0e-12f, qs[k] * rho);
        if (rqs > 1.0e-12f) {
            float supcol = 273.15f - tk;
            float nsfac = fmaxf(fminf(gfk_exp(0.12f * supcol), 5.0e4f), 1.0f);
            float lamdas = sqrtf(sqrtf(WDM6_PIDN0S * nsfac / rqs));
            re_qs = fmaxf(25.0e-6f, fminf(0.5f * (1.0f / lamdas), 999.0e-6f));
        }
        // wdm6 driver clamp (:321-325), then RRTMG's micron multiply.
        effc[id] = fmaxf(2.49e-6f, fminf(re_qc, 50.0e-6f)) * 1.0e6f;
        effi[id] = fmaxf(4.99e-6f, fminf(re_qi, 125.0e-6f)) * 1.0e6f;
        effs[id] = fmaxf(9.99e-6f, fminf(re_qs, 999.0e-6f)) * 1.0e6f;
    }
    return 0u;
}

#ifndef WDM6_CPU_MIRROR
extern "C" __global__ void wdm6_column(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, float *nn, float *nc, float *nr,
    const float *den, const float *p, const float *pii, const float *dz,
    const float *xland,
    float *rainnc, float *rainncv, float *snownc, float *snowncv,
    float *graupelnc, float *graupelncv, float *sr,
    float *effc, float *effi, float *effs,
    float delt, int hail_opt, int nz, int ny, int nx,
    unsigned int *count_status)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WDM6_KMAX) return;
    unsigned int flags = wdm6_column_impl(theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
                     den, p, pii, dz, xland,
                     rainnc, rainncv, snownc, snowncv,
                     graupelnc, graupelncv, sr, effc, effi, effs,
                     delt, hail_opt, nz, stride, col);
    if (flags) atomicOr(count_status, flags);
}
#endif
