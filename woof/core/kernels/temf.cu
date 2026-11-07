// gpuwm/core/kernels/temf.cu
//
// The Total Energy - Mass Flux pair of WRF v4.6.1 ,
// one CUDA thread per column:
//
//   temf_sfclay_column   oracle-only stock surface ABI
//   Production surface:  gpuwm/core/kernels/temf_sfclay.cu, TEMFSFCLAY1D
//                        (sf_sfclay_physics = 10)
//   temf_pbl_column      phys/module_bl_temf.F  temf2d with
//                        thlqt2thqvqc, rsat, solve_implicit_temf, trid
//                        (bl_pbl_physics = 10)
//
// WRF requires the two together, in both directions
// (share/module_check_a_mundo.F:2287-2304); gpuwm.config refuses any other
// pairing for the same reason.
//
// ARITHMETIC. Statements follow native WRF operation order. The shared
// loader enables contraction and FTZ by default. GPUWM_WRF_EXACT=1 selects
// -fmad=false --ftz=false for bitwise replay. Constants are source-derived
// binary32 words checked against the pinned native compiler's folding.
// Native TVAL truncation, bottom solver denominator, temperature/Exner
// indexing, top spacing and upper mass-flux recurrence are retained.
// The older corrected-source oracle is historical evidence only.
//
// LAYOUT.  Fields are gpuwm's (nz, ny, nx), x fastest; the kernel sees the
// flattened column index.  The scheme's own vertical index runs 1..kte with
// kte = nz: TEMF level 1 is the surface (z0) and TEMF level k >= 2 is WRF
// mass level k-1, so the carried *_temf fields are on TEMF's grid exactly
// as WRF stores them.  Column work arrays live in a global workspace,
// TEMF_WS_SLOTS slots of (kte+2) words per lane, interleaved across the
// TEMF_LANES threads of a block so neighbouring lanes read neighbouring
// words.

#define TEMF_LANES 64
#define TEMF_WS_SLOTS 49

// BEGIN TEMF CONSTANTS -- written by gpuwm/core/temf_constants.py (cuda_table); do not edit
#define TEMF_NCONST 63
__constant__ unsigned int TEMFC[TEMF_NCONST] = {
    0x411CF5C3u,  // [ 0] G = 9.81
    0x438F8000u,  // [ 1] R_D = 287.
    0x447B2000u,  // [ 2] CP = 7.*R_D/2.
    0x43E6CCCDu,  // [ 3] R_V = 461.6
    0x44E6CCCDu,  // [ 4] CPV = 4.*R_V
    0x3E924925u,  // [ 5] RCP = R_D/CP
    0x4A189680u,  // [ 6] XLV = 2.5E6
    0x3F1F2B09u,  // [ 7] EP2 = R_D/R_V
    0x3ECCCCCDu,  // [ 8] KARMAN = 0.4
    0x3F1C779Au,  // [ 9] SVP1 = 0.6112
    0x418D5C29u,  // [10] SVP2 = 17.67
    0x41ED3333u,  // [11] SVP3 = 29.65
    0x43889333u,  // [12] SVPT0 = 273.15
    0x3924A05Eu,  // [13] VISC_TEMF = 1.57e-4
    0x396097BAu,  // [14] CONDUC_TEMF = 1.57e-4 / 0.733
    0x3A83126Fu,  // [15] TEMIN = 1e-3
    0x3E2E147Bu,  // [16] FTAU0 = 0.17
    0x3E147AE1u,  // [17] FTH0 = 0.145
    0x3E3D70A4u,  // [18] CF = 0.185
    0x3D8F5C29u,  // [19] CEPS = 0.070
    0x3F2FF165u,  // [20] PRT0 = CEPS/CEPS * FTAU0**2 / 2. / FTH0**2
    0x3CF5C28Fu,  // [21] CM = 0.03
    0x3BC49BA6u,  // [22] CDELT = 0.006
    0x3F2FF165u,  // [23] RATIO_POS = CEPS**2.*FTAU0**2./(2.*CEPS**2.*FTH0**2.)
    0xBF2FF165u,  // [24] RATIO_NEG = CEPS**2.*FTAU0**2./(-2.*CEPS**2.*FTH0**2.)
    0x3F400000u,  // [25] THREE_QUARTERS = 3./4.
    0x3E800000u,  // [26] ONE_QUARTER = 1./4.
    0x3BA3D70Au,  // [27] INV_LASYMP = 1./200.0
    0x3F19C868u,  // [28] KHCONV = FTAU0**2. / CEPS / PRT0
    0x3EAAAAABu,  // [29] ONE_THIRD = 1./3.
    0x3F2AAAABu,  // [30] TWO_THIRDS = 2./3.
    0x40490FDAu,  // [31] PI_LIT = 3.1415926
    0x3FC90FDAu,  // [32] HALF_PI_LIT = 3.1415926/2
    0x3F1BA5E3u,  // [33] K0P608 = 0.608
    0x26901D7Du,  // [34] TINY15 = 1e-15
    0x3B03126Fu,  // [35] K0P002 = 0.002
    0x3EA8F5C3u,  // [36] K0P33 = 0.33
    0x3F1F3B64u,  // [37] K0P622 = 0.622
    0x3F666666u,  // [38] K0P9 = 0.9
    0x3D4CCCCDu,  // [39] K0P05 = 0.05
    0x3EB851ECu,  // [40] K0P36 = 0.36
    0x3FC66666u,  // [41] K1P55 = 1.55
    0x3F1C28F6u,  // [42] K0P61 = 0.61
    0x3E926E98u,  // [43] K0P286 = 0.286
    0x36246AAEu,  // [44] K2P45EM6 = 2.45e-6
    0x447B2AE1u,  // [45] K1004P67 = 1004.67
    0x3FCE147Bu,  // [46] K1P61 = 1.61
    0x3F333333u,  // [47] K0P7 = 0.7
    0x3E99999Au,  // [48] K0P3 = 0.3
    0x3A83126Fu,  // [49] K0P001 = 0.001
    0x4418A572u,  // [50] RSAT_C0 = 0.6105851e+3
    0x42319CD6u,  // [51] RSAT_C1 = 0.4440316e+2
    0x3FB7156Au,  // [52] RSAT_C2 = 0.1430341e+1
    0x3CD8626Du,  // [53] RSAT_C3 = 0.2641412e-1
    0x399D06F9u,  // [54] RSAT_C4 = 0.2995057e-3
    0x36085D76u,  // [55] RSAT_C5 = 0.2031998e-5
    0x31EE52B2u,  // [56] RSAT_C6 = 0.6936113e-8
    0x2C347C6Eu,  // [57] RSAT_C7 = 0.2564861e-11
    0xA926D4DFu,  // [58] RSAT_C8 = -0.3704404e-13
    0x37B3AC94u,  // [59] CONDUC_SFC = 1.57e-5 / 0.733
    0x3E4CCCCDu,  // [60] K0P2 = 0.2
    0x3C23D70Au,  // [61] K1EM2 = 1e-2
    0x3F4CCCCDu,  // [62] K0P8 = 0.8
};
#define TC_G __uint_as_float(TEMFC[0])
#define TC_R_D __uint_as_float(TEMFC[1])
#define TC_CP __uint_as_float(TEMFC[2])
#define TC_R_V __uint_as_float(TEMFC[3])
#define TC_CPV __uint_as_float(TEMFC[4])
#define TC_RCP __uint_as_float(TEMFC[5])
#define TC_XLV __uint_as_float(TEMFC[6])
#define TC_EP2 __uint_as_float(TEMFC[7])
#define TC_KARMAN __uint_as_float(TEMFC[8])
#define TC_SVP1 __uint_as_float(TEMFC[9])
#define TC_SVP2 __uint_as_float(TEMFC[10])
#define TC_SVP3 __uint_as_float(TEMFC[11])
#define TC_SVPT0 __uint_as_float(TEMFC[12])
#define TC_VISC_TEMF __uint_as_float(TEMFC[13])
#define TC_CONDUC_TEMF __uint_as_float(TEMFC[14])
#define TC_TEMIN __uint_as_float(TEMFC[15])
#define TC_FTAU0 __uint_as_float(TEMFC[16])
#define TC_FTH0 __uint_as_float(TEMFC[17])
#define TC_CF __uint_as_float(TEMFC[18])
#define TC_CEPS __uint_as_float(TEMFC[19])
#define TC_PRT0 __uint_as_float(TEMFC[20])
#define TC_CM __uint_as_float(TEMFC[21])
#define TC_CDELT __uint_as_float(TEMFC[22])
#define TC_RATIO_POS __uint_as_float(TEMFC[23])
#define TC_RATIO_NEG __uint_as_float(TEMFC[24])
#define TC_THREE_QUARTERS __uint_as_float(TEMFC[25])
#define TC_ONE_QUARTER __uint_as_float(TEMFC[26])
#define TC_INV_LASYMP __uint_as_float(TEMFC[27])
#define TC_KHCONV __uint_as_float(TEMFC[28])
#define TC_ONE_THIRD __uint_as_float(TEMFC[29])
#define TC_TWO_THIRDS __uint_as_float(TEMFC[30])
#define TC_PI_LIT __uint_as_float(TEMFC[31])
#define TC_HALF_PI_LIT __uint_as_float(TEMFC[32])
#define TC_K0P608 __uint_as_float(TEMFC[33])
#define TC_TINY15 __uint_as_float(TEMFC[34])
#define TC_K0P002 __uint_as_float(TEMFC[35])
#define TC_K0P33 __uint_as_float(TEMFC[36])
#define TC_K0P622 __uint_as_float(TEMFC[37])
#define TC_K0P9 __uint_as_float(TEMFC[38])
#define TC_K0P05 __uint_as_float(TEMFC[39])
#define TC_K0P36 __uint_as_float(TEMFC[40])
#define TC_K1P55 __uint_as_float(TEMFC[41])
#define TC_K0P61 __uint_as_float(TEMFC[42])
#define TC_K0P286 __uint_as_float(TEMFC[43])
#define TC_K2P45EM6 __uint_as_float(TEMFC[44])
#define TC_K1004P67 __uint_as_float(TEMFC[45])
#define TC_K1P61 __uint_as_float(TEMFC[46])
#define TC_K0P7 __uint_as_float(TEMFC[47])
#define TC_K0P3 __uint_as_float(TEMFC[48])
#define TC_K0P001 __uint_as_float(TEMFC[49])
#define TC_RSAT_C0 __uint_as_float(TEMFC[50])
#define TC_RSAT_C1 __uint_as_float(TEMFC[51])
#define TC_RSAT_C2 __uint_as_float(TEMFC[52])
#define TC_RSAT_C3 __uint_as_float(TEMFC[53])
#define TC_RSAT_C4 __uint_as_float(TEMFC[54])
#define TC_RSAT_C5 __uint_as_float(TEMFC[55])
#define TC_RSAT_C6 __uint_as_float(TEMFC[56])
#define TC_RSAT_C7 __uint_as_float(TEMFC[57])
#define TC_RSAT_C8 __uint_as_float(TEMFC[58])
#define TC_CONDUC_SFC __uint_as_float(TEMFC[59])
#define TC_K0P2 __uint_as_float(TEMFC[60])
#define TC_K1EM2 __uint_as_float(TEMFC[61])
#define TC_K0P8 __uint_as_float(TEMFC[62])
// END TEMF CONSTANTS

// ------------------------------------------------------------------------
// WOOF's own float32 library, under the scheme's names.
// ------------------------------------------------------------------------
__device__ __forceinline__ float temf_exp(float x) { return gfk_exp(x); }
__device__ __forceinline__ float temf_log(float x) { return gfk_log(x); }
__device__ __forceinline__ float temf_pow(float x, float y) { return gfk_pow(x, y); }
__device__ __forceinline__ float temf_atan(float x) { return glibc_atanf(x); }
__device__ __forceinline__ float temf_sin(float x) { return glibc_sinf(x); }
__device__ __forceinline__ float temf_cos(float x) { return glibc_cosf(x); }
__device__ __forceinline__ float temf_log10(float x)
{ return __double2float_rn(log10((double)x)); }
__device__ __forceinline__ float temf_atan2(float y, float x)
{ return __double2float_rn(atan2((double)y, (double)x)); }

// gfortran's MAX for REAL arguments: the first argument unless a later one
// compares greater (or the running value is a NaN).
__device__ __forceinline__ float temf_max(float a, float b)
{ return (b > a || a != a) ? b : a; }

// ------------------------------------------------------------------------
// Column workspace.
// ------------------------------------------------------------------------
struct TfCol {
    float *p;
    __device__ __forceinline__ float &operator[](int k) const {
        return p[(long long)k * TEMF_LANES];
    }
    // A view whose index 1 is this column's index 1 + off.
    __device__ __forceinline__ TfCol shift(int off) const {
        return TfCol{p + (long long)off * TEMF_LANES};
    }
};
#define TF_SLOT(base, idx, kp) \
    (TfCol{(base) + (long long)(idx) * (long long)(kp) * TEMF_LANES})
#define TF_LANE_BASE(ws, kp) \
    ((ws) + (long long)blockIdx.x * (long long)TEMF_WS_SLOTS * (long long)(kp) \
     * TEMF_LANES + threadIdx.x)

// 1-based WRF level k of a (nz, ncol) field.
#define TG(a, k) a[(long long)((k) - 1) * ncol + col]

// ------------------------------------------------------------------------
// rsat (module_bl_temf.F:1283-1309): saturation mixing ratio over water.
// ------------------------------------------------------------------------
__device__ float temf_rsat(float p, float T, float ep2)
{
    float temp = T - 273.15f;
    // 273.15 is SVPT0's word (0x43889333); the literal is the same decimal.
    float x = TC_RSAT_C0 + temp * (TC_RSAT_C1 + temp * (TC_RSAT_C2 + temp
              * (TC_RSAT_C3 + temp * (TC_RSAT_C4 + temp * (TC_RSAT_C5 + temp
              * (TC_RSAT_C6 + temp * (TC_RSAT_C7 + temp * TC_RSAT_C8)))))));
    return (ep2 * x) / (p - x);
}

// ------------------------------------------------------------------------
// trid and solve_implicit_temf (module_bl_temf.F:1313-1394), on
// kbot = 1 .. ktop.  psi is updated in place.
// ------------------------------------------------------------------------
__device__ void temf_solve_implicit(TfCol khlf, TfCol psi, float srf_flux,
                                    TfCol dzm, TfCol dzt, int ktop, float dt,
                                    TfCol au, TfCol bu, TfCol cu, TfCol yu,
                                    TfCol gam, TfCol usol)
{
    const int kbot = 1;
    au[kbot] = khlf[kbot] / (dzm[kbot] * dzt[kbot]);
    bu[kbot] = -(1.0f / dt) - khlf[kbot + 1] / (dzm[kbot + 1] * dzt[kbot + 1]);
    cu[kbot] = khlf[kbot + 1] / (dzm[kbot] * dzt[kbot + 1]);
    yu[kbot] = -(psi[kbot] / dt) - srf_flux / dzm[kbot];
    for (int k = kbot + 1; k <= ktop - 1; ++k) {
        au[k] = khlf[k] / (dzm[k] * dzt[k]);
        bu[k] = -(1.0f / dt) - (khlf[k] / dzt[k] + khlf[k + 1] / dzt[k + 1])
                / dzm[k];
        cu[k] = khlf[k + 1] / (dzm[k] * dzt[k + 1]);
        yu[k] = -(psi[k] / dt);
    }
    au[ktop] = 0.0f;
    bu[ktop] = -(1.0f / dt);
    yu[ktop] = -(psi[ktop] / dt);
    // trid
    float bet = bu[kbot];
    usol[kbot] = yu[kbot] / bet;
    for (int k = kbot + 1; k <= ktop; ++k) {
        gam[k] = cu[k - 1] / bet;
        bet = bu[k] - au[k] * gam[k];
        usol[k] = (yu[k] - au[k] * usol[k - 1]) / bet;
    }
    for (int k = ktop - 1; k >= kbot; --k) {
        usol[k] = usol[k] - gam[k + 1] * usol[k + 1];
    }
    for (int k = kbot; k <= ktop; ++k) psi[k] = usol[k];
}

// ------------------------------------------------------------------------
// temfsfclay1d (module_sf_temfsfclay.F:154-384).
// ------------------------------------------------------------------------
extern "C" __global__
void temf_sfclay_column(
        // lowest model level and surface inputs
        const float *u1d, const float *v1d, const float *th1d,
        const float *qv1d, const float *p1d, const float *pi1d,
        const float *rho1, const float *z1, const float *zsrf,
        const float *psfc, const float *znt, const float *mavail,
        const float *tsk, const float *fcor, const float *hd_temf,
        // inout
        float *hfx, float *qfx, float *lh, float *ust, float *te1,
        float *wm_temf, float *qsfc,
        // out
        float *flhc, float *flqc, float *exch_temf, float *chs,
        float *chs2, float *cqs2, float *u10, float *v10, float *th2,
        float *t2, float *q2, float *qgh, float *cpm,
        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = TC_G, cp = TC_CP, xlv = TC_XLV, ep2 = TC_EP2;
    const float karman = TC_KARMAN;
    const float svp1 = TC_SVP1, svp2 = TC_SVP2, svp3 = TC_SVP3;
    const float svpt0 = TC_SVPT0;
    const float conduc_temf = TC_CONDUC_SFC, visc_temf = 1.57e-5f;
    const float TEmin = TC_TEMIN, ftau0 = TC_FTAU0, fth0 = TC_FTH0;
    const float Cf = TC_CF, CN = 2.0f, Ceps = TC_CEPS;
    const float Cphi = Ceps, PrT0 = TC_PRT0;
    (void)visc_temf;

    float tski = tsk[i], pii = pi1d[i], th1 = th1d[i];
    float u1 = u1d[i], v1 = v1d[i], qv1 = qv1d[i];

    // :235-238
    float e1 = svp1 * temf_exp((svp2 * (tski - svpt0)) / (tski - svp3));
    float qsfc_i = (ep2 * e1) / ((psfc[i] / 1000.0f) - e1);
    float qsfc_air = qsfc_i * mavail[i];
    float thetav = (tski / pii) * (1.0f + TC_K0P608 * qsfc_air);
    float z0t = znt[i];
    // :244-251
    float zt = ((z1[i] - zsrf[i]) - znt[i]) / 2.0f;
    float zm = z1[i] - zsrf[i];
    float dthdz = (th1 - (tski / pii)) / (zt * temf_log10(zm / z0t));
    float dqtdz = (qv1 - qsfc_air) / (zt * temf_log10(zm / z0t));
    (void)dqtdz;
    float dudz = u1 / (zt * temf_log10(zm / znt[i]));
    float dvdz = v1 / (zt * temf_log10(zm / znt[i]));
    // :256
    float te = te1[i];
    if (te < TEmin) te = TEmin;
    te1[i] = te;
    // :258-262
    float wstr;
    if (hfx[i] > 0.0f) {
        wstr = temf_pow(((g * hd_temf[i]) / thetav)
                        * (hfx[i] / (rho1[i] * cp)), TC_ONE_THIRD);
    } else {
        wstr = 0.0f;
    }
    // :272-287
    float N2 = (g / thetav) * dthdz;
    float S = sqrtf(dudz * dudz + dvdz * dvdz);
    float Ri = N2 / (S * S);
    if (S < TC_TINY15) {
        Ri = (N2 >= 0.0f) ? TC_K0P2 : -1.0f;
    }
    if (Ri > TC_K0P2) Ri = TC_K0P2;
    float beta = g / thetav;
    float ratio, ftau, fth, TE2;
    if (Ri > 0.0f) {
        ratio = Ri / (TC_RATIO_POS + 3.0f * Ri);
        ftau = ftau0 * ((TC_THREE_QUARTERS / (1.0f + 4.0f * Ri))
                        + TC_ONE_QUARTER);
        fth = fth0 / (1.0f + 4.0f * Ri);
        TE2 = (((2.0f * te) * ratio) * N2) / (beta * beta);
    } else {
        ratio = Ri / (TC_RATIO_NEG + 2.0f * Ri);
        ftau = ftau0;
        fth = fth0;
        TE2 = 0.0f;
    }
    // :302-314
    float TKE = te * (1.0f - ratio);
    float ustrtilde = sqrtf(ftau * TKE);
    float linv;
    if (N2 > 0.0f) {
        linv = (((1.0f / karman) / zt) + (fabsf(fcor[i]) / (Cf * ustrtilde)))
               + (sqrtf(N2) / (CN * ustrtilde));
    } else {
        linv = ((1.0f / karman) / zt) + (fabsf(fcor[i]) / (Cf * ustrtilde));
    }
    float leps = 1.0f / linv;
    leps = temf_max(leps, 0.0f);
    // :320-323.  km is computed and floored but read by nothing below.
    float km = (temf_pow(TKE, 1.5f) * (ftau * ftau))
               / (-((beta * fth) * sqrtf(TE2))
                  + ((Ceps * sqrtf(TKE * te)) / leps));
    float kh = ((((2.0f * leps) * (fth * fth)) * TKE) / sqrtf(te)) / Cphi;
    km = temf_max(km, 1.57e-5f);
    kh = temf_max(kh, conduc_temf);
    (void)km; (void)kh;
    // :329-340
    float ust_i = ((sqrtf(ftau / ftau0)
                    * sqrtf((u1 * u1 + v1 * v1) + (0.5f * wstr) * (0.5f * wstr)))
                   * leps / temf_log(zm / znt[i])) / zt;
    ust[i] = ust_i;
    float wm = ust_i;
    wm = (wm + wm_temf[i]) / 2.0f;
    wm_temf[i] = temf_max(wm, TC_K1EM2);
    // :349-352
    float lzm_z0t = temf_log(zm / z0t);
    float flhc_i = (((((rho1[i] * cp) * fth) / fth0) * wm) * leps / PrT0)
                   / lzm_z0t / zt;
    float flqc_i = ((((((rho1[i] * fth) / fth0) * wm) * leps) / PrT0)
                    / lzm_z0t / zt) * mavail[i];
    flhc[i] = flhc_i;
    flqc[i] = flqc_i;
    // Oracle-only: WRF's scalar invalid division has the negative quiet
    // NaN word. CUDA's invalid-result payload differs on this hardware.
    exch_temf[i] = mavail[i] == 0.0f ? __uint_as_float(0xffc00000u)
                                     : flqc_i / mavail[i];
    chs[i] = mavail[i] == 0.0f ? __uint_as_float(0xffc00000u)
                              : (flqc_i / rho1[i]) / mavail[i];
    // :358-362
    float hfx_i = flhc_i * (tski - th1 * pii);
    hfx[i] = hfx_i;
    float qfx_i = flqc_i * (qsfc_i - qv1);
    qfx_i = temf_max(qfx_i, 0.0f);
    qfx[i] = qfx_i;
    lh[i] = xlv * qfx_i;
    // :367-375
    float lzm_znt = temf_log(zm / znt[i]);
    u10[i] = (u1 * temf_log(10.0f / znt[i])) / lzm_znt;
    v10[i] = (v1 * temf_log(10.0f / znt[i])) / lzm_znt;
    float t2_i = ((tski / pii) + ((th1 - (tski / pii)) * temf_log(2.0f / z0t))
                  / lzm_z0t) * pii;
    t2[i] = t2_i;
    th2[i] = t2_i / pii;
    q2[i] = qsfc_air + ((qv1 - qsfc_air) * temf_log(2.0f / znt[i])) / lzm_znt;
    chs2[i] = ((((fth / fth0) * wm) * leps) / PrT0) / temf_log(2.0f / z0t) / zt;
    cqs2[i] = ((((fth / fth0) * wm) * leps) / PrT0) / temf_log(2.0f / znt[i])
              / zt;
    // :378-380
    float t1k = th1 * pii;
    e1 = svp1 * temf_exp((svp2 * (t1k - svpt0)) / (t1k - svp3));
    qgh[i] = (ep2 * e1) / ((p1d[i] / 1000.0f) - e1);
    cpm[i] = cp * (1.0f + TC_K0P8 * qv1);
    qsfc[i] = qsfc_i;
}

// ------------------------------------------------------------------------
// thlqt2thqvqc (module_bl_temf.F:1181-1218) for one level.
// ------------------------------------------------------------------------
__device__ void temf_thlqt2thqvqc(float thetal, float qt, float p, float piex,
                                  float ep2, float Lv, float Cp,
                                  float *theta, float *qv, float *qc)
{
    float T1 = thetal * piex;
    float Tair = T1;
    float rt = qt / (1.0f - qt);
    float rc = 0.0f;
    for (int iterate = 1; iterate <= 20; ++iterate) {
        float rst = temf_rsat(p, Tair, ep2);
        rc = temf_max(rt - rst, 0.0f);
        float Tt = TC_K0P7 * Tair
                   + (TC_K0P3 * T1) * (1.0f + (Lv * rc)
                                       / (Cp * temf_max(Tair, 253.0f)));
        if (fabsf(Tt - Tair) < TC_K0P001) break;
        Tair = Tt;
    }
    float rv = rt - rc;
    *qv = rv / (1.0f + rv);
    *qc = rc / (1.0f + rc);
    *theta = Tair / piex;
}

// ------------------------------------------------------------------------
// temf2d (module_bl_temf.F:256-1177) for one column.
// ------------------------------------------------------------------------
extern "C" __global__
void temf_pbl_column(
        // (nz, ncol) WRF mass-level inputs
        const float *ux, const float *vx, const float *thx, const float *tx,
        const float *qvx, const float *qcx, const float *p2d,
        const float *pi2d, const float *rho, const float *z2d,
        // (nz+1, ncol) interface pressure
        const float *p2di,
        // (ncol) surface inputs
        const float *znt, const float *zsrf, const float *hfx,
        const float *qfx, const float *tsk, const float *fcor,
        // (ncol) inout
        float *ust, float *hpbl, int *kpbl, float *u10, float *v10,
        float *t2,
        // (nz, ncol) carried TEMF state, on TEMF's grid
        float *te, float *kh_temf, float *km_temf, float *shf, float *qf,
        float *uw, float *vw, float *wupd, float *mf, float *thup,
        float *qtup, float *qlup, float *cf3d,
        // (ncol) carried TEMF diagnostics
        float *hd_temf, float *lcl_temf, float *hct_temf, float *cfm_temf,
        // (nz, ncol) tendencies
        float *rubl, float *rvbl, float *rthbl, float *rqvbl, float *rqcbl,
        float dt, int nz, int ncol, float *ws, int wskp, int col0)
{
    int col = col0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float *wsb = TF_LANE_BASE(ws, wskp);
    const int kte = nz;
    const int kts = 1;

    const float g = TC_G, cp = TC_CP, rcp = TC_RCP, r_d = TC_R_D;
    const float r_v = TC_R_V, cpv = TC_CPV, xlv = TC_XLV, ep2 = TC_EP2;
    const float karman = TC_KARMAN;
    const float visc_temf = TC_VISC_TEMF, conduc_temf = TC_CONDUC_TEMF;
    const float TEmin = TC_TEMIN, ftau0 = TC_FTAU0, fth0 = TC_FTH0;
    const float Cf = TC_CF, CN = 2.0f, Ceps = TC_CEPS;
    const float Cgamma = Ceps, Cphi = Ceps, PrT0 = TC_PRT0;
    const float CM = TC_CM, Cdelt = TC_CDELT, Cw = 0.5f, Cc = 3.0f;
    const float hmax = 4000.0f;
    (void)r_d;

    TfCol thetal = TF_SLOT(wsb, 0, wskp);
    TfCol qt = TF_SLOT(wsb, 1, wskp);
    TfCol u_temf = TF_SLOT(wsb, 2, wskp);
    TfCol v_temf = TF_SLOT(wsb, 3, wskp);
    TfCol thetav = TF_SLOT(wsb, 4, wskp);
    TfCol zm = TF_SLOT(wsb, 5, wskp);
    TfCol zt = TF_SLOT(wsb, 6, wskp);
    TfCol dzm = TF_SLOT(wsb, 7, wskp);
    TfCol dzt = TF_SLOT(wsb, 8, wskp);
    TfCol dthdz = TF_SLOT(wsb, 9, wskp);
    TfCol dqtdz = TF_SLOT(wsb, 10, wskp);
    TfCol dudz = TF_SLOT(wsb, 11, wskp);
    TfCol dvdz = TF_SLOT(wsb, 12, wskp);
    TfCol ftau = TF_SLOT(wsb, 13, wskp);
    TfCol fth = TF_SLOT(wsb, 14, wskp);
    TfCol TKE = TF_SLOT(wsb, 15, wskp);
    TfCol linv = TF_SLOT(wsb, 16, wskp);
    TfCol leps = TF_SLOT(wsb, 17, wskp);
    TfCol km = TF_SLOT(wsb, 18, wskp);
    TfCol kh = TF_SLOT(wsb, 19, wskp);
    TfCol Fz = TF_SLOT(wsb, 20, wskp);
    TfCol wupd_dry = TF_SLOT(wsb, 21, wskp);
    TfCol UUPD = TF_SLOT(wsb, 22, wskp);
    TfCol VUPD = TF_SLOT(wsb, 23, wskp);
    TfCol TEUPD = TF_SLOT(wsb, 24, wskp);
    TfCol qlUPD = TF_SLOT(wsb, 25, wskp);
    TfCol rUPD = TF_SLOT(wsb, 26, wskp);
    TfCol rstUPD = TF_SLOT(wsb, 27, wskp);
    TfCol deltmf = TF_SLOT(wsb, 28, wskp);
    TfCol MFCth = TF_SLOT(wsb, 29, wskp);
    TfCol MFCq = TF_SLOT(wsb, 30, wskp);
    TfCol MFCu = TF_SLOT(wsb, 31, wskp);
    TfCol MFCv = TF_SLOT(wsb, 32, wskp);
    TfCol MFCql = TF_SLOT(wsb, 33, wskp);
    TfCol MFCTE = TF_SLOT(wsb, 34, wskp);
    TfCol QFK = TF_SLOT(wsb, 35, wskp);
    TfCol alpha2 = TF_SLOT(wsb, 36, wskp);
    TfCol beta2 = TF_SLOT(wsb, 37, wskp);
    TfCol THVF = TF_SLOT(wsb, 38, wskp);
    TfCol srcs = TF_SLOT(wsb, 39, wskp);
    TfCol u_new = TF_SLOT(wsb, 40, wskp);
    TfCol v_new = TF_SLOT(wsb, 41, wskp);
    TfCol khavg = TF_SLOT(wsb, 42, wskp);
    TfCol s_au = TF_SLOT(wsb, 43, wskp);
    TfCol s_bu = TF_SLOT(wsb, 44, wskp);
    TfCol s_cu = TF_SLOT(wsb, 45, wskp);
    TfCol s_yu = TF_SLOT(wsb, 46, wskp);
    TfCol s_gam = TF_SLOT(wsb, 47, wskp);
    TfCol s_usol = TF_SLOT(wsb, 48, wskp);

    const float zsrf_i = zsrf[col], znt_i = znt[col], hfx_i = hfx[col];
    const float rho1 = TG(rho, 1), fcor_i = fcor[col];

    // ---- :467-476  the surface (TEMF level 1) ----
    thetal[1] = tsk[col] / TG(pi2d, 1);
    qt[1] = TG(qvx, 1);
    thetav[1] = thetal[1] * ((1.0f + TC_K0P608 * qt[1]) - TG(qcx, 1));
    const float z0t = znt_i;

    // ---- :480-493  WRF levels to TEMF levels ----
    for (int k = kts + 1; k <= kte; ++k) {
        float qvk = TG(qvx, k - 1), qck = TG(qcx, k - 1);
        float rv = qvk / (1.0f - qvk);
        float rl = qck / (1.0f - qck);
        float rt = rv + rl;
        float chi = (rcp * (1.0f + rv / ep2)) / (1.0f + (rv * cpv) / cp);
        float gam = (rt * r_v) / (cp + rt * cpv);
        float tlev = TG(tx, k);
        thetal[k] = ((TG(thx, k - 1)
                      * temf_pow((ep2 + rv) / (ep2 + rt), chi))
                     * temf_pow(rv / rt, -gam))
                    * temf_exp((-(xlv * rl)) / ((cp + rt * cpv) * tlev));
        qt[k] = qvk + qck;
        thetav[k] = thetal[k] * ((1.0f + TC_K0P608 * qt[k]) - qck);
    }

    // ---- :497-502 ----
    u_temf[1] = 0.0f;
    v_temf[1] = 0.0f;
    for (int k = kts + 1; k <= kte; ++k) {
        u_temf[k] = TG(ux, k - 1);
        v_temf[k] = TG(vx, k - 1);
    }

    // ---- :505-516  heights ----
    zm[1] = znt_i;
    dzt[1] = (TG(z2d, 1) - zsrf_i) - zm[1];
    zt[1] = ((TG(z2d, 1) - zsrf_i) - znt_i) / 2.0f;
    for (int kt = kts + 1; kt <= kte; ++kt) {
        zm[kt] = TG(z2d, kt - 1) - zsrf_i;
        zt[kt] = ((zm[kt] + TG(z2d, kt)) - zsrf_i) / 2.0f;
        dzm[kt] = zt[kt] - zt[kt - 1];
        // Z2D(kte+1) is read past the column here and the value is
        // overwritten below; the kernel does not read it.
        if (kt < kte) dzt[kt] = TG(z2d, kt + 1) - TG(z2d, kt);
    }
    dzm[1] = dzm[2];
    dzt[kte] = dzt[kte - 1];

    // ---- :519-522  first-level gradients ----
    {
        float l10t = temf_log10(zm[2] / z0t);
        float l10m = temf_log10(zm[2] / znt_i);
        dthdz[1] = (thetal[2] - thetal[1]) / (zt[1] * l10t);
        dqtdz[1] = (qt[2] - qt[1]) / (zt[1] * l10t);
        dudz[1] = (u_temf[2] - u_temf[1]) / (zt[1] * l10m);
        dvdz[1] = (v_temf[2] - v_temf[1]) / (zt[1] * l10m);
    }

    // ---- :525 ----
    const float sfcTHVF = ((hfx_i / (rho1 * cp))
                           * (1.0f + TC_K0P608 * (TG(qvx, 1) + TG(qcx, 1))))
                          + (TC_K0P608 * thetav[1]) * TG(qf, 1);

    // ---- :529-561 ----
    int h0idx = 1;
    float h0 = zm[1];
    int hmax_idx = kte - 1;
    for (int k = kts + 1; k <= kte - 1; ++k) {
        dthdz[k] = (thetal[k + 1] - thetal[k]) / dzt[k];
        dqtdz[k] = (qt[k + 1] - qt[k]) / dzt[k];
        dudz[k] = (u_temf[k + 1] - u_temf[k]) / dzt[k];
        dvdz[k] = (v_temf[k + 1] - v_temf[k]) / dzt[k];
        if (thetav[k] > thetav[1] && h0idx == 1) {
            if (zm[k] < hmax) {
                h0idx = k;
                h0 = zm[k];
            } else {
                h0idx = k;
                h0 = hmax;
            }
        }
        if (zm[k] > hmax) {
            hmax_idx = min(hmax_idx, k);
        }
    }
    dthdz[kte] = dthdz[kte - 1];
    dqtdz[kte] = dqtdz[kte - 1];
    dudz[kte] = dudz[kte - 1];
    dvdz[kte] = dvdz[kte - 1];

    // ---- :570-579 ----
    float wstr;
    if (hfx_i > 0.0f) {
        wstr = temf_pow((((g * h0) / thetav[2]) * hfx_i) / (rho1 * cp),
                        TC_ONE_THIRD);
    } else {
        wstr = 0.0f;
    }
    const bool is_convective = wstr > 0.0f && dthdz[1] < 0.0f
                               && dthdz[2] < 0.0f;

    // ---- :582-622  stability and length scale on turbulence levels ----
    float S1 = 0.0f, beta1 = 0.0f, ratio1 = 0.0f;
    for (int kt = 1; kt <= kte - 1; ++kt) {
        float N2 = ((2.0f * g) / (thetav[kt] + thetav[kt + 1])) * dthdz[kt];
        float S = sqrtf(dudz[kt] * dudz[kt] + dvdz[kt] * dvdz[kt]);
        float Ri = N2 / (S * S);
        if (S < TC_TINY15) {
            Ri = (N2 >= 0.0f) ? 10.0f : -1.0f;
        }
        float beta = (2.0f * g) / (thetav[kt] + thetav[kt + 1]);
        float ratio;
        if (Ri > 0.0f) {
            ratio = Ri / (TC_RATIO_POS + 3.0f * Ri);
            ftau[kt] = ftau0 * ((TC_THREE_QUARTERS / (1.0f + 4.0f * Ri))
                                + TC_ONE_QUARTER);
            fth[kt] = fth0 / (1.0f + 4.0f * Ri);
            // TE2 (:598) is read by nothing; not formed.
        } else {
            ratio = Ri / (TC_RATIO_NEG + 2.0f * Ri);
            ftau[kt] = ftau0;
            fth[kt] = fth0;
        }
        float tek = TG(te, kt);
        TKE[kt] = tek * (1.0f - ratio);
        float ustrtilde = sqrtf(ftau[kt] * TKE[kt]);
        if (N2 > 0.0f) {
            linv[kt] = ((((1.0f / karman) / zt[kt])
                         + (fabsf(fcor_i) / (Cf * ustrtilde)))
                        + (sqrtf(N2) / (CN * ustrtilde))) + TC_INV_LASYMP;
        } else {
            linv[kt] = (((1.0f / karman) / zt[kt])
                        + (fabsf(fcor_i) / (Cf * ustrtilde))) + TC_INV_LASYMP;
        }
        leps[kt] = 1.0f / linv[kt];
        leps[kt] = temf_max(leps[kt], 0.0f);   // lepsmin = 0 (:532, :538)
        if (kt == 1) { S1 = S; beta1 = beta; ratio1 = ratio; }
    }
    TKE[kte] = 0.0f;
    linv[kte] = linv[kte - 1];
    leps[kte] = leps[kte - 1];

    // ---- :632-667  diffusivities ----
    for (int kt = 1; kt <= kte - 1; ++kt) {
        float tek = TG(te, kt);
        float kmk = (temf_pow(TKE[kt], 1.5f) * (ftau[kt] * ftau[kt]))
                    / ((Ceps * sqrtf(TKE[kt] * tek)) / leps[kt]);
        float khk = ((((2.0f * leps[kt]) * (fth[kt] * fth[kt])) * TKE[kt])
                     / sqrtf(tek)) / Cphi;
        if (is_convective) {
            float lconv;
            if (kt <= h0idx && (h0 - zt[kt]) > TC_TINY15) {
                lconv = 1.0f / ((1.0f / (karman * zt[kt]))
                                + (Cc / (karman * (h0 - zt[kt]))));
            } else {
                lconv = 0.0f;
            }
            float kh_conv = (TC_KHCONV * sqrtf(TKE[kt])) * lconv;
            if (kh_conv < 0.0f) kh_conv = 0.0f;
            float km_conv = PrT0 * kh_conv;
            if (zt[kt] <= h0 / 2.0f) {
                kmk = km_conv;
                khk = kh_conv;
            }
            if (zt[kt] > h0 / 2.0f && kt <= h0idx) {
                kmk = temf_max(temf_max(kmk, km_conv), visc_temf);
                khk = temf_max(temf_max(khk, kh_conv), conduc_temf);
            }
        }
        kmk = temf_max(kmk, visc_temf);
        khk = temf_max(khk, conduc_temf);
        km[kt] = kmk;
        kh[kt] = khk;
        Fz[kt] = -(khk * dthdz[kt]);
    }
    km[kte] = km[kte - 1];
    kh[kte] = kh[kte - 1];
    Fz[kte] = 0.0f;

    // ---- :672-975  mass flux ----
    float hd, lcl, hct;
    int hdidx = 1, hctidx = 1;
    if (is_convective) {
        float Cepsmf = 2.0f / temf_max(200.0f, h0);
        Cepsmf = temf_max(Cepsmf, TC_K0P002);
        const float epsmf = Cepsmf;
        // :683-696
        TG(thup, 1) = thetal[1];
        TG(qtup, 1) = qt[1];
        rUPD[1] = TG(qtup, 1) / (1.0f - TG(qtup, 1));
        TG(wupd, 1) = Cw * wstr;
        wupd_dry[1] = Cw * wstr;
        UUPD[1] = u_temf[1];
        VUPD[1] = v_temf[1];
        TEUPD[1] = TG(te, 1) + (g / thetav[1]) * sfcTHVF;
        qlUPD[1] = TG(qcx, 1);
        {
            float TUPD1 = TG(thup, 1) * TG(pi2d, 1);
            rstUPD[1] = temf_rsat(TG(p2d, 1), TUPD1, ep2);
        }
        // :699-758
        for (int k = 2; k <= kte; ++k) {
            if (k < hmax_idx) {
                float dth = -(epsmf * (TG(thup, k - 1) - thetal[k - 1]));
                TG(thup, k) = TG(thup, k - 1) + dth * dzm[k - 1];
                float dqt = -(epsmf * (TG(qtup, k - 1) - qt[k - 1]));
                TG(qtup, k) = TG(qtup, k - 1) + dqt * dzm[k - 1];
                float thup_k = TG(thup, k), qtup_k = TG(qtup, k);
                float thetavUPD = thup_k * (1.0f + TC_K0P608 * qtup_k);
                float B = (g * (thetavUPD - thetav[k])) / thetav[k];
                if (wupd_dry[k - 1] < TC_TINY15) {
                    wupd_dry[k] = 0.0f;
                } else {
                    float dw = -((2.0f * epsmf) * wupd_dry[k - 1])
                               + (TC_K0P33 * B) / wupd_dry[k - 1];
                    wupd_dry[k] = wupd_dry[k - 1] + dw * dzm[k - 1];
                }
                float dU = -(epsmf * (UUPD[k - 1] - u_temf[k - 1]));
                UUPD[k] = UUPD[k - 1] + dU * dzm[k - 1];
                float dV = -(epsmf * (VUPD[k - 1] - v_temf[k - 1]));
                VUPD[k] = VUPD[k - 1] + dV * dzm[k - 1];
                float dTE = -(epsmf * (TEUPD[k - 1] - TG(te, k - 1)));
                TEUPD[k] = TEUPD[k - 1] + dTE * dzm[k - 1];
                rUPD[k] = qtup_k / (1.0f - qtup_k);
                const float pi_up = TG(pi2d, k);
                float TUPD = thup_k * pi_up;
                float rst = temf_rsat(TG(p2d, k - 1), TUPD, ep2);
                float b1 = (TC_K0P622 * (xlv / (r_d * TUPD)))
                           * (xlv / (cp * TUPD));
                rst = (rst * (1.0f + b1 * rUPD[k])) / (1.0f + b1 * rst);
                rstUPD[k] = rst;
                float qstUPD = rst / (1.0f + rst);
                float thetavUPDmoist;
                if (rUPD[k] > rstUPD[k]) {
                    float rlUPD = rUPD[k] - rstUPD[k];
                    qlUPD[k] = rlUPD / (1.0f + rlUPD);
                    thetavUPDmoist = (thup_k + (((xlv / cp) * qlUPD[k])
                                                / pi_up))
                                     * ((1.0f + TC_K0P608 * qstUPD) - qlUPD[k]);
                } else {
                    qlUPD[k] = TG(qcx, k - 1);
                    thetavUPDmoist = thup_k * (1.0f + TC_K0P608 * qtup_k);
                }
                float Bmoist = (g * (thetavUPDmoist - thetav[k])) / thetav[k];
                if (TG(wupd, k - 1) < TC_TINY15) {
                    TG(wupd, k) = 0.0f;
                } else {
                    float wprev = TG(wupd, k - 1);
                    float dwm = -((2.0f * epsmf) * wprev)
                                + (TC_K0P33 * Bmoist) / wprev;
                    TG(wupd, k) = wprev + dwm * dzm[k - 1];
                }
            } else {
                TG(thup, k) = thetal[k];
                TG(qtup, k) = qt[k];
                wupd_dry[k] = 0.0f;
                UUPD[k] = u_temf[k];
                VUPD[k] = v_temf[k];
                TEUPD[k] = TG(te, k);
                qlUPD[k] = TG(qcx, k - 1);
                TG(wupd, k) = 0.0f;
            }
        }
        // :761-775  hd from the dry updraft
        if (wupd_dry[1] == 0.0f) {
            hdidx = 1;
        } else {
            hdidx = kte;
            for (int k = 2; k <= kte; ++k) {
                if (wupd_dry[k] <= 0.0f || zm[k] > hmax) {
                    hdidx = k;
                    break;
                }
            }
        }
        hd = zm[hdidx];
        kpbl[col] = hdidx;
        hpbl[col] = hd;
        // :778-785  LCL
        int lclidx = kte;
        for (int k = kts; k <= kte; ++k) {
            if (k < hmax_idx && rUPD[k] > rstUPD[k]) {
                lclidx = k;
                break;
            }
        }
        lcl = zm[lclidx];
        // :787-811  hct
        if (hd > lcl) {
            if (TG(wupd, 1) == 0.0f) {
                hctidx = 1;
            } else {
                hctidx = kte;
                for (int k = 2; k <= kte; ++k) {
                    if (TG(wupd, k) <= 0.0f || zm[k] > hmax) {
                        hctidx = k;
                        break;
                    }
                }
            }
            hct = zm[hctidx];
            if (hctidx <= hdidx + 1) {
                hct = hd;
                hctidx = hdidx;
            }
        } else {
            hct = hd;
            hctidx = hdidx;
        }
        const float ht = temf_max(hd, hct);
        // :814-839  taper the updraft at ht
        for (int k = 1; k <= kte; ++k) {
            float zmk = zm[k];
            int tval;   // WRF's INTEGER: the band weight truncates to 0
            if (zmk < TC_K0P9 * ht) {
                tval = 1;
            } else if (zmk >= TC_K0P9 * ht && zmk <= 1.0f * ht) {
                tval = (int)(1.0f - ((zmk - TC_K0P9 * ht)
                                     / (1.0f * ht - TC_K0P9 * ht)));
            } else {
                tval = 0;
            }
            const float tw = (float)tval, tc = (float)(1 - tval);
            TG(thup, k) = tw * TG(thup, k) + tc * thetal[k];
            // THETAVUPD (:824) is tapered and then read by nothing.
            TG(qtup, k) = tw * TG(qtup, k) + tc * qt[k];
            if (k > 1) {
                qlUPD[k] = tw * qlUPD[k] + tc * TG(qcx, k - 1);
            }
            UUPD[k] = tw * UUPD[k] + tc * u_temf[k];
            VUPD[k] = tw * VUPD[k] + tc * v_temf[k];
            TEUPD[k] = tw * TEUPD[k] + tc * TG(te, k);
            if (zmk > ht) {
                TG(wupd, k) = 0.0f;
                wupd_dry[k] = 0.0f;
            }
        }
        // :842-852  detrainment
        deltmf[1] = Cepsmf;
        for (int k = 2; k <= kte - 1; ++k) {
            if (hctidx > hdidx + 1) {
                deltmf[k] = TC_K0P9 * Cepsmf
                            + (Cdelt * (temf_atan((zm[k] - (lcl + (hct - lcl)
                                                             / 1.5f))
                                                  / ((hct - lcl) / 8.0f))
                                        + TC_HALF_PI_LIT)) / TC_PI_LIT;
            } else if (k < hdidx) {
                deltmf[k] = Cepsmf + (TC_K0P05 * 1.0f) / (hd - zm[k]);
            } else {
                deltmf[k] = deltmf[k - 1];
            }
        }
        // :855-859  mass flux on turbulence levels
        TG(mf, 1) = CM * wstr;
        for (int kt = 2; kt <= kte - 1; ++kt) {
            float dMdz = ((epsmf - deltmf[kt]) * TG(mf, kt - 1)) * dzt[kt];
            TG(mf, kt) = TG(mf, kt - 1) + dMdz;
        }
        // :864-870  keep the second level from turning stable
        // Fortran's a-b + c-d is ((a - b) + c) - d.
        MFCth[2] = (TG(mf, 2) * (((TG(thup, 2) - thetal[2]) + TG(thup, 3))
                                 - thetal[3])) / 2.0f;
        if (MFCth[2] > Fz[2]) {
            float red_fact = Fz[2] / MFCth[2];
            for (int kt = 1; kt <= kte; ++kt) {
                TG(mf, kt) = TG(mf, kt) * red_fact;
            }
        }
        // :874-905  mass-flux contributions
        {
            // m*(a1-b1 + (a2-b2 - (a1-b1)) * log(zt1/z0)/log(zm2/z0))
            const float la = temf_log(zt[1] / znt_i);
            const float lb = temf_log(zm[2] / znt_i);
            const float m1 = TG(mf, 1);
#define TEMF_MFC1(a1, b1, a2, b2)             (m1 * (((a1) - (b1)) + ((((a2) - (b2)) - ((a1) - (b1))) * la) / lb))
            MFCth[1] = TEMF_MFC1(TG(thup, 1), thetal[1], TG(thup, 2), thetal[2]);
            MFCq[1] = TEMF_MFC1(TG(qtup, 1), qt[1], TG(qtup, 2), qt[2]);
            MFCu[1] = TEMF_MFC1(UUPD[1], u_temf[1], UUPD[2], u_temf[2]);
            MFCv[1] = TEMF_MFC1(VUPD[1], v_temf[1], VUPD[2], v_temf[2]);
            // :886-888 index QCX at WRF levels 1 and 2 as WRF does; the
            // value reaches only THVF(1), which nothing reads.
            MFCql[1] = TEMF_MFC1(qlUPD[1], TG(qcx, 1), qlUPD[2], TG(qcx, 2));
            MFCTE[1] = TEMF_MFC1(TEUPD[1], TG(te, 1), TEUPD[2], TG(te, 2));
#undef TEMF_MFC1
        }
        for (int kt = 2; kt <= kte - 1; ++kt) {
            float m = TG(mf, kt);
            MFCth[kt] = (m * (((TG(thup, kt) - thetal[kt]) + TG(thup, kt + 1))
                              - thetal[kt + 1])) / 2.0f;
            MFCq[kt] = (m * (((TG(qtup, kt) - qt[kt]) + TG(qtup, kt + 1))
                             - qt[kt + 1])) / 2.0f;
            MFCu[kt] = (m * (((UUPD[kt] - u_temf[kt]) + UUPD[kt + 1])
                             - u_temf[kt + 1])) / 2.0f;
            MFCv[kt] = (m * (((VUPD[kt] - v_temf[kt]) + VUPD[kt + 1])
                             - v_temf[kt + 1])) / 2.0f;
            MFCql[kt] = (m * (((qlUPD[kt] - TG(qcx, kt - 1)) + qlUPD[kt + 1])
                              - TG(qcx, kt))) / 2.0f;
            MFCTE[kt] = m * (TEUPD[kt] - TG(te, kt));
        }
        MFCth[kte] = 0.0f; MFCq[kte] = 0.0f; MFCu[kte] = 0.0f;
        MFCv[kte] = 0.0f; MFCql[kte] = 0.0f; MFCTE[kte] = 0.0f;
        // :908-943  cloud fraction on mass levels
        TG(cf3d, 1) = 0.0f;
        float cfm = 0.0f;
        for (int k = 2; k <= kte; ++k) {
            float au;
            if (TG(wupd, k - 1) >= TC_TINY15 && TG(wupd, k) >= TC_TINY15) {
                au = ((TG(mf, k - 1) + TG(mf, k)) / 2.0f)
                     / ((TG(wupd, k - 1) + TG(wupd, k)) / 2.0f);
            } else {
                au = 0.0f;
            }
            float sigq2 = au * (TG(qtup, k) - qt[k]);
            float sigq = (sigq2 > 0.0f) ? sqrtf(sigq2) : 0.0f;
            float rst = temf_rsat(TG(p2d, k - 1),
                                  TG(thx, k - 1) * TG(pi2d, k - 1), ep2);
            float qst = rst / (1.0f + rst);
            float satdef = qt[k] - qst;
            float cf;
            if (satdef <= 0.0f) {
                if (sigq > TC_TINY15) {
                    cf = temf_max(0.5f + TC_K0P36
                                  * temf_atan(TC_K1P55 * (satdef / sigq)),
                                  0.0f);
                } else {
                    cf = 0.0f;
                }
            } else {
                cf = 1.0f;
            }
            if (zm[k] < lcl) cf = 0.0f;
            TG(cf3d, k) = cf;
            if (zt[k] <= hmax) cfm = temf_max(cf, cfm);
        }
        cfm_temf[col] = cfm;
    } else {
        // :946-973
        for (int kt = 1; kt <= kte; ++kt) {
            MFCth[kt] = 0.0f; MFCq[kt] = 0.0f; MFCu[kt] = 0.0f;
            MFCv[kt] = 0.0f; MFCql[kt] = 0.0f; MFCTE[kt] = 0.0f;
        }
        lcl = zm[kte - 1];
        hct = zm[1];
        hctidx = 1;
        hd = zm[1];
        hdidx = 1;
        TG(cf3d, 1) = 0.0f;
        float cfm = 0.0f;
        for (int k = 2; k <= kte; ++k) {
            float cf = (TG(qcx, k - 1) > TC_TINY15) ? 1.0f : 0.0f;
            TG(cf3d, k) = cf;
            if (zt[k] <= hmax) cfm = temf_max(cf, cfm);
        }
        cfm_temf[col] = cfm;
    }
    TG(cf3d, kte) = 0.0f;

    // ---- :980-989  flux profiles ----
    for (int kt = 2; kt <= kte; ++kt) {
        TG(shf, kt) = Fz[kt] + MFCth[kt];
        QFK[kt] = -(kh[kt] * dqtdz[kt]);
        TG(qf, kt) = QFK[kt] + MFCq[kt];
        TG(uw, kt) = -(km[kt] * dudz[kt]) + MFCu[kt];
        TG(vw, kt) = -(km[kt] * dvdz[kt]) + MFCv[kt];
    }

    // ---- :995-999  surface momentum flux ----
    float ust_i = (((sqrtf(ftau[1] / ftau0)
                     * sqrtf((u_temf[2] * u_temf[2] + v_temf[2] * v_temf[2])
                             + (0.5f * wstr) * (0.5f * wstr)))
                    * leps[1]) / temf_log(zm[2] / znt_i)) / zt[1];
    ust[col] = ust_i;
    {
        float ang = temf_atan2(v_temf[2], u_temf[2]);
        TG(uw, 1) = -temf_cos(ang) * (ust_i * ust_i);
        TG(vw, 1) = -temf_sin(ang) * (ust_i * ust_i);
    }

    // ---- :1010-1013 ----
    TG(shf, 1) = hfx_i / (rho1 * cp);
    TG(qf, 1) = qfx[col] / rho1;
    Fz[1] = TG(shf, 1) - MFCth[1];
    QFK[1] = TG(qf, 1) - MFCq[1];

    // ---- :1020-1029  thetav flux coefficients ----
    for (int kt = 2; kt <= kte - 1; ++kt) {
        alpha2[kt] = (TC_K0P61 * (thetal[kt] + thetal[kt + 1])) / 2.0f;
        beta2[kt] = ((temf_pow(100000.0f / TG(p2di, kt), TC_K0P286)
                      * TC_K2P45EM6) / TC_K1004P67)
                    - (TC_K1P61 * (thetal[kt] + thetal[kt + 1])) / 2.0f;
    }
    {
        float th_zt2 = thetal[1] + ((thetal[2] - thetal[1]) * (zt[2] - znt_i))
                       / (zm[2] - znt_i);
        alpha2[1] = TC_K0P61 * th_zt2;
        alpha2[kte] = TC_K0P61 * thetal[kte];
        beta2[1] = ((temf_pow(100000.0f / TG(p2di, 1), TC_K0P286)
                     * TC_K2P45EM6) / TC_K1004P67) - TC_K1P61 * th_zt2;
        beta2[kte] = ((temf_pow(100000.0f / TG(p2di, kte), TC_K0P286)
                       * TC_K2P45EM6) / TC_K1004P67) - TC_K1P61 * thetal[kte];
    }
    // ---- :1030-1044 ----
    for (int kt = 1; kt <= kte; ++kt) {
        float MFCthv = 0.0f;
        if (is_convective && kt <= kte - 1) {
            MFCthv = ((((1.0f + TC_K0P61 * (TG(qtup, kt) + TG(qtup, kt + 1)))
                        / 2.0f) * MFCth[kt]) + alpha2[kt] * MFCq[kt])
                     + beta2[kt] * MFCql[kt];
        }
        THVF[kt] = (((1.0f + TC_K0P61 * qt[kt]) * Fz[kt])
                    + alpha2[kt] * QFK[kt]) + MFCthv;
    }

    // ---- :1052-1076  implicit diffusion, then the explicit mass flux ----
    for (int k = 1; k <= kte; ++k) {
        u_new[k] = u_temf[k];
        v_new[k] = v_temf[k];
    }
    temf_solve_implicit(km, u_new.shift(1), TG(uw, 1), dzm, dzt, kte - 1, dt,
                        s_au, s_bu, s_cu, s_yu, s_gam, s_usol);
    for (int k = 2; k <= kte - 1; ++k) {
        u_new[k] = u_new[k] + (dt * (-(MFCu[k] - MFCu[k - 1]))) / dzm[k];
    }
    temf_solve_implicit(km, v_new.shift(1), TG(vw, 1), dzm, dzt, kte - 1, dt,
                        s_au, s_bu, s_cu, s_yu, s_gam, s_usol);
    for (int k = 2; k <= kte - 1; ++k) {
        v_new[k] = v_new[k] + (dt * (-(MFCv[k] - MFCv[k - 1]))) / dzm[k];
    }
    temf_solve_implicit(kh, thetal.shift(1), Fz[1], dzm, dzt, kte - 1, dt,
                        s_au, s_bu, s_cu, s_yu, s_gam, s_usol);
    for (int k = 2; k <= kte - 1; ++k) {
        thetal[k] = thetal[k] + (dt * (-(MFCth[k] - MFCth[k - 1]))) / dzm[k];
    }
    temf_solve_implicit(kh, qt.shift(1), QFK[1], dzm, dzt, kte - 1, dt,
                        s_au, s_bu, s_cu, s_yu, s_gam, s_usol);
    for (int k = 2; k <= kte - 1; ++k) {
        qt[k] = qt[k] + (dt * (-(MFCq[k] - MFCq[k - 1]))) / dzm[k];
    }

    // ---- :1079-1114  total energy ----
    {
        float te1 = ((ust_i * ust_i) / ftau[1]) * (1.0f + ratio1);
        float src = is_convective ? (2.0f * beta1) * TG(shf, 1) : 0.0f;
        te1 = temf_max(te1, temf_pow((leps[1] / Cgamma)
                                     * ((ust_i * ust_i) * S1 + src),
                                     TC_TWO_THIRDS));
        if (te1 > 20.0f) te1 = 20.0f;
        TG(te, 1) = te1;
    }
    const float sfcFTE = (-((km[1] + km[2]) / 2.0f)
                          * (TG(te, 2) - TG(te, 1))) / dzm[2];
    for (int kt = 1; kt <= kte; ++kt) {
        float buoy_src = (THVF[kt] >= 0.0f)
                         ? ((2.0f * g) / thetav[kt]) * THVF[kt] : 0.0f;
        float tek = TG(te, kt);
        srcs[kt] = ((-(TG(uw, kt) * dudz[kt]) - TG(vw, kt) * dvdz[kt])
                    - (Cgamma * temf_pow(tek, 1.5f)) * linv[kt]) + buoy_src;
    }
    for (int k = 1; k <= kte - 1; ++k) {
        khavg[k] = (km[k] + km[k + 1]) / 2.0f;
    }
    // TE_TEMFX(KTS+1:KTE) is solved as an array section: stage it in the
    // UUPD slot (read by nothing after the mass-flux block) and copy back.
    TfCol tes = UUPD;
    for (int k = 2; k <= kte; ++k) tes[k] = TG(te, k);
    temf_solve_implicit(khavg, tes.shift(1), sfcFTE, dzt.shift(1), dzt,
                        kte - 1, dt, s_au, s_bu, s_cu, s_yu, s_gam, s_usol);
    for (int kt = 2; kt <= kte - 1; ++kt) {
        float tek = tes[kt];
        tek = tek + dt * srcs[kt];
        tek = tek + (dt * (-(MFCTE[kt] - MFCTE[kt - 1]))) / dzt[kt];
        if (tek < TEmin) tek = TEmin;
        tes[kt] = tek;
    }
    tes[kte] = 0.0f;
    for (int kt = 2; kt <= kte - 1; ++kt) {
        if (tes[kt] > 20.0f) tes[kt] = 20.0f;
    }
    for (int k = 2; k <= kte; ++k) TG(te, k) = tes[k];

    // ---- :1117-1121 ----
    for (int k = kts; k <= kte; ++k) {
        TG(kh_temf, k) = kh[k];
        TG(km_temf, k) = km[k];
    }

    // ---- :1127-1145  back to theta, qv, qc and the tendencies ----
    float thx_new1 = 0.0f;
    for (int k = kts; k <= kte - 1; ++k) {
        float thn, qvn, qcn;
        temf_thlqt2thqvqc(thetal[k + 1], qt[k + 1], TG(p2d, k), TG(pi2d, k),
                          ep2, xlv, cp, &thn, &qvn, &qcn);
        if (k == 1) thx_new1 = thn;
        TG(rubl, k) = (u_new[k + 1] - u_temf[k + 1]) / dt;
        TG(rvbl, k) = (v_new[k + 1] - v_temf[k + 1]) / dt;
        TG(rthbl, k) = (thn - TG(thx, k)) / dt;
        TG(rqvbl, k) = (qvn - TG(qvx, k)) / dt;
        TG(rqcbl, k) = (qcn - TG(qcx, k)) / dt;
    }
    TG(rubl, kte) = 0.0f;
    TG(rvbl, kte) = 0.0f;
    TG(rthbl, kte) = 0.0f;
    TG(rqvbl, kte) = 0.0f;
    TG(rqcbl, kte) = 0.0f;

    // ---- :1153-1155  10 m wind and 2 m temperature ----
    {
        float lzm = temf_log(zm[2] / znt_i);
        u10[col] = (u_new[2] * temf_log(10.0f / znt_i)) / lzm;
        v10[col] = (v_new[2] * temf_log(10.0f / znt_i)) / lzm;
        float pi1 = TG(pi2d, 1);
        float ts = tsk[col] / pi1;
        t2[col] = (ts + ((thx_new1 - ts) * temf_log(2.0f / z0t))
                        / temf_log(zm[2] / z0t)) * pi1;
    }

    // ---- :1158-1173  diagnostics ----
    hd_temf[col] = hd;
    lcl_temf[col] = lcl;
    hct_temf[col] = hct;
    if (is_convective) {
        for (int k = kts; k <= kte - 1; ++k) TG(qlup, k) = qlUPD[k];
    } else {
        TG(qlup, 1) = TG(qcx, 1);
        for (int k = kts + 1; k <= kte - 1; ++k) TG(qlup, k) = TG(qcx, k - 1);
    }
    TG(qlup, kte) = TG(qcx, kte);
}
