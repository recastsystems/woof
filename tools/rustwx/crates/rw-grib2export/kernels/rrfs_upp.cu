// SPDX-License-Identifier: Apache-2.0
//
// RRFS UPP-set extras: the derived fields of NOAA's rrfs_mpas_postcntrl_L60
// field list that the WOOF post (rw-post Groups A to E) does not already
// compute.  One thread per column (or per grid point for the stencil), no
// atomics, no shared-memory reductions, fixed iteration counts.
//
// Clean room.  NOAA's UPP (NOAA-EMC/UPP) is LGPL-3.0 and is NOT read or
// ported here; only the field list (names, levels, GRIB2 identities) of the
// rrfs-workflow control file is used, and every method below is written from
// the published source it cites:
//
// - Parcel thermodynamics (LCL, theta-e, pseudoadiabat, virtual buoyancy):
//   the rw-post Group C helpers (severe_c.cu, included below), from Bolton,
//   Mon. Wea. Rev. 108 (1980) 1046; Davies-Jones, Mon. Wea. Rev. 136 (2008)
//   2764; Doswell and Rasmussen, Wea. Forecasting 9 (1994) 625.
// - Lifted index: Galway, Bull. Amer. Meteor. Soc. 37 (1956) 528; best
//   (four-layer) lifted index: the lowest of the 30 hPa layer parcels in
//   the 180 hPa above ground (Fujita et al. 1970, NWS usage).
// - Most-unstable parcel in the lowest 255 hPa, its LFC and EL heights and
//   its lifting level pressure: AMS Glossary of Meteorology.
// - 0-3 km CAPE: positive buoyancy of the mixed-layer parcel below 3 km
//   AGL (Rasmussen, Wea. Forecasting 18 (2003) 530).
// - Effective inflow layer and effective storm-relative helicity:
//   Thompson et al., Wea. Forecasting 22 (2007) 102 (CAPE >= 100 J kg-1,
//   CIN >= -250 J kg-1, Bunkers right-mover).
// - Enhanced stretching potential: SPC mesoanalysis definition (Davies):
//   (0-3 km MLCAPE / 50) ((0-2 km lapse rate - 7) / 1), 0 where the lapse
//   rate is under 7 K km-1 or MLCAPE under 250 J kg-1.
// - Critical angle: Esterheld and Giuliano, Electronic J. Severe Storms
//   Meteor. 3(2) (2008): angle between the 10-500 m shear vector and the
//   storm-relative 10 m wind.
// - VIL: Greene and Clark, J. Appl. Meteor. 11 (1972) 1346,
//   3.44e-6 Z^(4/7) dh with Z in mm6 m-3, reflectivity capped at 56 dBZ
//   (NWS WSR-88D VIL algorithm hail cap).
// - Echo top: highest height of the 18.3 dBZ reflectivity (NWS WSR-88D
//   echo-top threshold), linear between levels.
// - Precipitable water and RH with respect to it: AMS Glossary
//   (column water / saturated column water).
// - Column integrated condensate: (1/g) integral of q dp.
// - Dendritic growth layer depth: the -12 C to -17 C layer (Bailey and
//   Hallett, J. Atmos. Sci. 66 (2009) 2888).
// - Ventilation rate: PBL height times the PBL mean wind (EPA/NWS fire
//   weather usage).
// - Total cloud cover from the low, middle and high layer covers with
//   random overlap between layers (Geleyn and Hollingsworth, Beitr. Phys.
//   Atmos. 52 (1979) 1).
// - Absolute vorticity on a conformal grid: Haltiner and Williams,
//   Numerical Prediction and Dynamic Meteorology (1980) eq. 1-24, with the
//   Lambert conformal map factor.
//
// Built with --fmad=false --prec-div=true --prec-sqrt=true --ftz=false into
// rrfs_upp.ptx (compute_60, NVRTC 12.0) by examples/rebuild_rrfs_kernels.rs.

#include "woof_math.cuh"

namespace rrfs_c {
#include "severe_c.cu"
}
using namespace rrfs_c;

// ---- packed inputs ---------------------------------------------------------
// 3-D mass-level slots, each nz * n words, level-major [k][cell].
#define V_P     0   // full pressure, Pa
#define V_TK    1   // temperature, K
#define V_R     2   // water-vapour mixing ratio, kg kg-1
#define V_ZM    3   // mass-level height, m MSL
#define V_U     4   // mass-level wind components (any frame; only differences
#define V_V     5   //   and magnitudes are used, plus the PBL mean, rotated after)
#define V_QC    6
#define V_QI    7
#define V_QR    8
#define V_QS    9
#define V_QG   10
#define V_REFL 11   // reflectivity, dBZ (NaN where the history has none)
#define V_CLDBL 12  // boundary-layer cloud fraction, 0-1 (NaN where absent)
#define V_COUNT 13

// Interface slots, each (nz + 1) * n words.
#define I_PINT 0
#define I_ZINT 1

// 2-D slots, each n words.
#define S_ZSFC   0
#define S_PSFC   1
#define S_T2     2
#define S_Q2     3
#define S_P2     4
#define S_U10    5
#define S_V10    6
#define S_PBLH   7
#define S_SMU    8   // Bunkers right-mover (Group C), same frame as V_U/V_V
#define S_SMV    9
#define S_MLCAPE 10
#define S_FRZH   11  // lowest 0 C height, m MSL (Group E)
#define S_HFRZH  12  // highest tropospheric freezing height, m MSL (Group E)
#define S_CBASE  13  // cloud base height, m MSL (Group D)
#define S_CTOP   14  // cloud top height, m MSL (Group D)
#define S_LCLD   15  // low / middle / high cloud cover, percent (Group D)
#define S_MCLD   16
#define S_HCLD   17
#define S_RAINV  18  // last-step precipitation, mm: all phases (RAINNCV)
#define S_SNOWV  19  //   snow and ice (SNOWNCV)
#define S_GRAUPV 20  //   graupel (GRAUPELNCV)
#define S_COUNT  21

// Output planes.
#define O_LFTX     0
#define O_4LFTX    1
#define O_MUCAPE   2
#define O_MUCIN    3
#define O_PLPL     4
#define O_CAPE03   5
#define O_LFCH     6
#define O_ELH      7
#define O_EFHL     8
#define O_ESP      9
#define O_CANGLE  10
#define O_LAYTH   11
#define O_RHPW    12
#define O_TCOLW   13
#define O_TCOLI   14
#define O_TCOLP   15
#define O_VIL     16
#define O_RETOP   17
#define O_REFM10  18
#define O_REF1KM  19
#define O_REF4KM  20
#define O_UPBL    21
#define O_VPBL    22
#define O_VRATE   23
#define O_TCDCBL  24
#define O_RHFRZ   25
#define O_RHHFRZ  26
#define O_PCBASE  27
#define O_PCTOP   28
#define O_TCDC    29
#define O_CPOFP   30
#define O_CRAIN   31
#define O_CSNOW   32
#define O_CICEP   33
#define O_CFRZR   34
#define O_COUNT   35

#define NANV __longlong_as_double(0x7ff8000000000000LL)
#define ECHO_TOP_DBZ 18.3
#define VIL_CAP_DBZ 56.0

struct Ext {
    const float* v3;
    const float* vi;
    long long n, vol, ivol;
    int nz, c;
    __device__ double V(int s, int k) const { return (double)v3[s * vol + (long long)k * n + c]; }
    __device__ double I(int s, int k) const { return (double)vi[s * ivol + (long long)k * n + c]; }
};

// Interpolate mass slot `s` to height z (m MSL), linear in height; NaN outside.
__device__ double at_height(const Ext& e, int s, double z) {
    for (int k = 0; k + 1 < e.nz; ++k) {
        double za = e.V(V_ZM, k), zb = e.V(V_ZM, k + 1);
        if (z >= za && z <= zb) {
            double w = (z - za) / (zb - za);
            return e.V(s, k) + (e.V(s, k + 1) - e.V(s, k)) * w;
        }
    }
    if (e.nz > 0 && z < e.V(V_ZM, 0) && z >= e.I(I_ZINT, 0)) return e.V(s, 0);
    return NANV;
}

// Pressure at height z, log-linear between interfaces.
__device__ double p_at_height(const Ext& e, double z) {
    for (int k = 0; k < e.nz; ++k) {
        double za = e.I(I_ZINT, k), zb = e.I(I_ZINT, k + 1);
        if (z >= za && z <= zb) {
            double w = (z - za) / (zb - za);
            return wm_exp(wm_ln(e.I(I_PINT, k)) + (wm_ln(e.I(I_PINT, k + 1)) - wm_ln(e.I(I_PINT, k))) * w);
        }
    }
    return NANV;
}

// Height (m MSL) at pressure p, log-linear between interfaces.
__device__ double z_at_pressure(const Ext& e, double p) {
    for (int k = 0; k < e.nz; ++k) {
        double pa = e.I(I_PINT, k), pb = e.I(I_PINT, k + 1);
        if (p <= pa && p >= pb) {
            double w = wm_ln(pa / p) / wm_ln(pa / pb);
            return e.I(I_ZINT, k) + (e.I(I_ZINT, k + 1) - e.I(I_ZINT, k)) * w;
        }
    }
    return NANV;
}

// Environmental temperature at pressure p (log-linear between mass levels).
__device__ double t_env_at_p(const Ext& e, double p) {
    for (int k = 0; k + 1 < e.nz; ++k) {
        double pa = e.V(V_P, k), pb = e.V(V_P, k + 1);
        if (p <= pa && p >= pb) {
            double w = wm_ln(pa / p) / wm_ln(pa / pb);
            return e.V(V_TK, k) + (e.V(V_TK, k + 1) - e.V(V_TK, k)) * w;
        }
    }
    return NANV;
}

// Temperature of a parcel from (p_o, t_o, r_o) lifted to p_t: dry adiabat to
// the LCL, then the pseudoadiabat in 10 hPa Newton steps.
__device__ double parcel_t_at(double p_o, double t_o, double r_o, double p_t) {
    r_o = fmin_(fmax_(r_o, R_FLOOR_), rsat(t_o, p_o));
    double t_l, p_l;
    lcl_tp(p_o, t_o, r_o, &t_l, &p_l);
    if (p_t >= p_l) return t_from_theta(theta_(t_o, p_o), p_t);
    double ln_te = wm_ln(theta_e(t_o, p_o, r_o, t_l));
    double t = t_l, p = p_l;
    while (p - 1000.0 > p_t) {
        p -= 1000.0;
        t = pseudoadiabat_t(ln_te, p, t);
    }
    return pseudoadiabat_t(ln_te, p_t, t);
}

// A lift that also reports the LFC and EL pressures and the positive area
// below p_cap (CAPE 0-3 km when p_cap is the 3 km AGL pressure).  Buoyancy
// is the virtual temperature difference, integrated in ln p (as `lift`).
__device__ void lift_ext(const Col& col, double p_o, double t_o, double r_o, double tv_env_o, double p_cap,
                         double* cape, double* cin, double* p_lfc, double* p_el, double* cape_cap) {
    r_o = fmin_(fmax_(r_o, R_FLOOR_), rsat(t_o, p_o));
    double t_l, p_lcl;
    lcl_tp(p_o, t_o, r_o, &t_l, &p_lcl);
    double th_o = theta_(t_o, p_o);
    double ln_te = wm_ln(theta_e(t_o, p_o, r_o, t_l));
    double x_prev = wm_ln(p_o), p_prev = p_o;
    double b_prev = tv_(t_o, r_o) - tv_env_o;
    double tve_prev = tv_env_o;
    double t_moist = t_l;
    double pos = 0.0, neg_below = 0.0, best_cin = 0.0;
    bool have_lfc = false;
    double lfc = NANV, el = NANV, capped = 0.0;
    for (int k = 0; k < col.nz; ++k) {
        double pk = col.P(k);
        if (pk >= p_prev) continue;
        double xk = wm_ln(pk);
        double tve_k = col.tv_env(k);
        double tvp;
        if (pk >= p_lcl) {
            tvp = tv_(t_from_theta(th_o, pk), r_o);
        } else {
            t_moist = pseudoadiabat_t(ln_te, pk, t_moist);
            tvp = tv_(t_moist, rsat(t_moist, pk));
        }
        double b_k = tvp - tve_k;
        double dx = x_prev - xk;
        // zero crossing inside the segment
        double xz = NANV;
        if ((b_prev > 0.0) != (b_k > 0.0)) xz = x_prev - dx * (b_prev / (b_prev - b_k));
        double seg_pos, seg_neg;
        if (b_prev > 0.0 && b_k > 0.0) { seg_pos = 0.5 * (b_prev + b_k) * dx; seg_neg = 0.0; }
        else if (b_prev <= 0.0 && b_k <= 0.0) { seg_pos = 0.0; seg_neg = 0.5 * (b_prev + b_k) * dx; }
        else if (b_prev > 0.0) { double f = b_prev / (b_prev - b_k); seg_pos = 0.5 * b_prev * f * dx; seg_neg = 0.5 * b_k * (1.0 - f) * dx; }
        else { double f = b_prev / (b_prev - b_k); seg_pos = 0.5 * b_k * (1.0 - f) * dx; seg_neg = 0.5 * b_prev * f * dx; }
        // positive area below the cap pressure (clip the segment in ln p)
        if (isfinite(p_cap) && p_prev > p_cap && seg_pos > 0.0) {
            double frac = pk >= p_cap ? 1.0 : (x_prev - wm_ln(p_cap)) / dx;
            capped += seg_pos * fmin_(fmax_(frac, 0.0), 1.0);
        }
        if (!have_lfc) {
            if (b_k > 0.0 && pk < p_lcl) {
                have_lfc = true;
                best_cin = neg_below + seg_neg;
                lfc = (b_prev > 0.0) ? p_prev : wm_exp(xz);
                if (p_prev > p_lcl && b_prev > 0.0) lfc = p_lcl;
                pos = seg_pos;
            } else {
                neg_below += seg_neg;
            }
        } else {
            pos += seg_pos;
            if (b_prev > 0.0 && b_k <= 0.0) el = wm_exp(xz);
            if (b_k > 0.0) el = NANV;
        }
        x_prev = xk; p_prev = pk; b_prev = b_k; tve_prev = tve_k;
    }
    if (have_lfc && !isfinite(el)) el = p_prev;
    *cape = have_lfc ? RD_ * pos : 0.0;
    *cin = have_lfc ? RD_ * best_cin : 0.0;
    *p_lfc = lfc;
    *p_el = have_lfc ? el : NANV;
    *cape_cap = RD_ * capped;
}

extern "C" __global__ void rrfs_upp_column_v1(
    int nz, int n,
    const float* __restrict__ v3, const float* __restrict__ vi, const float* __restrict__ s2,
    float* __restrict__ out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    long long N = n;
    Ext e;
    e.v3 = v3; e.vi = vi; e.n = N; e.vol = (long long)nz * N; e.ivol = (long long)(nz + 1) * N; e.nz = nz; e.c = c;
    double o[O_COUNT];
    for (int i = 0; i < O_COUNT; ++i) o[i] = NANV;
    #define S(slot) ((double)s2[(long long)(slot) * N + c])
    double zsfc = S(S_ZSFC), psfc = S(S_PSFC);
    bool ok = isfinite(zsfc) && isfinite(psfc);
    for (int k = 0; k < nz && ok; ++k)
        ok = isfinite(e.V(V_P, k)) && isfinite(e.V(V_TK, k)) && isfinite(e.V(V_R, k)) && isfinite(e.V(V_ZM, k));
    for (int k = 0; k <= nz && ok; ++k) ok = isfinite(e.I(I_PINT, k)) && isfinite(e.I(I_ZINT, k));
    if (!ok) {
        for (int i = 0; i < O_COUNT; ++i) out[(long long)i * N + c] = (float)NANV;
        return;
    }
    Col col;
    col.nz = nz; col.n = n; col.c = c;
    col.p = v3 + V_P * e.vol; col.tk = v3 + V_TK * e.vol; col.r = v3 + V_R * e.vol;
    col.p_int = vi + I_PINT * e.ivol; col.z_int = vi + I_ZINT * e.ivol;
    col.zsfc = zsfc; col.psfc = psfc; col.t2 = S(S_T2); col.q2 = S(S_Q2); col.p2 = S(S_P2);
    bool shelter = isfinite(col.t2) && isfinite(col.q2) && isfinite(col.p2);

    // ---- lifted indices (500 hPa)
    double t500 = t_env_at_p(e, 50000.0);
    if (isfinite(t500) && shelter && col.p2 > 50000.0) {
        o[O_LFTX] = t500 - parcel_t_at(col.p2, col.t2, fmax_(col.q2, R_FLOOR_), 50000.0);
    }
    if (isfinite(t500)) {
        double pg = col.PI(0), best = 1.0e30;
        for (int i = 0; i < 6; ++i) {
            double p_bot = pg - 3000.0 * (double)i;
            double th, rr;
            layer_mean(col, p_bot - 3000.0, p_bot, &th, &rr);
            double pm = p_bot - 1500.0;
            if (!(isfinite(th) && isfinite(rr)) || pm <= 50000.0) continue;
            double li = t500 - parcel_t_at(pm, t_from_theta(th, pm), fmax_(rr, R_FLOOR_), 50000.0);
            if (li < best) best = li;
        }
        if (best < 1.0e29) o[O_4LFTX] = best;
    }

    // ---- most-unstable parcel in the lowest 255 hPa
    {
        int bk = -1;
        double bte = -1.0e30;
        for (int k = 0; k < nz; ++k) {
            if (col.P(k) < psfc - 25500.0) break;
            double te = theta_e_parcel(col.T(k), col.P(k), fmax_(col.R(k), R_FLOOR_));
            if (te > bte) { bte = te; bk = k; }
        }
        if (bk >= 0) {
            double cape, cin, plfc, pel, capped;
            lift_ext(col, col.P(bk), col.T(bk), col.R(bk), col.tv_env(bk), NANV, &cape, &cin, &plfc, &pel, &capped);
            o[O_MUCAPE] = cape;
            o[O_MUCIN] = cin;
            o[O_PLPL] = col.P(bk);
            o[O_LFCH] = isfinite(plfc) ? z_at_pressure(e, plfc) : NANV;
            o[O_ELH] = isfinite(pel) ? z_at_pressure(e, pel) : NANV;
        }
    }

    // ---- 0-3 km CAPE of the 90 hPa mixed-layer parcel
    double p3km = p_at_height(e, zsfc + 3000.0);
    double cape03 = NANV;
    {
        double pg = col.PI(0), th, rr;
        layer_mean(col, pg - 9000.0, pg, &th, &rr);
        if (isfinite(th) && isfinite(rr) && isfinite(p3km)) {
            double t_ml = t_from_theta(th, psfc);
            double cape, cin, plfc, pel;
            lift_ext(col, psfc, t_ml, rr, tv_env_at(col, psfc), p3km, &cape, &cin, &plfc, &pel, &cape03);
            o[O_CAPE03] = cape03;
        }
    }

    // ---- winds: effective SRH, critical angle, PBL mean wind
    Wind w;
    w.nz = nz; w.n = n; w.c = c; w.zm = v3 + V_ZM * e.vol; w.u = v3 + V_U * e.vol; w.v = v3 + V_V * e.vol;
    w.zsfc = zsfc; w.u10 = S(S_U10); w.v10 = S(S_V10);
    bool winds = isfinite(w.u10) && isfinite(w.v10);
    for (int k = 0; k < nz && winds; ++k) winds = isfinite(e.V(V_U, k)) && isfinite(e.V(V_V, k));
    double cu = S(S_SMU), cv = S(S_SMV);
    if (winds && isfinite(cu) && isfinite(cv)) {
        // effective inflow layer (Thompson et al. 2007), levels within 300 hPa of the ground
        int kb = -1, kt = -1;
        for (int k = 0; k < nz; ++k) {
            if (col.P(k) < psfc - 30000.0) break;
            double cape, cin;
            lift(col, col.P(k), col.T(k), col.R(k), col.tv_env(k), &cape, &cin);
            bool good = cape >= 100.0 && cin >= -250.0;
            if (kb < 0) {
                if (good) { kb = k; kt = k; }
            } else if (good) {
                kt = k;
            } else {
                break;
            }
        }
        if (kb < 0) {
            o[O_EFHL] = 0.0;
        } else {
            double hb = e.V(V_ZM, kb) - zsfc;
            double ht = e.V(V_ZM, kt) - zsfc;
            if (kb == 0) hb = 0.0;
            double s_top, s_bot = 0.0;
            bool okt = wind_srh(w, fmax_(ht, Z10_ + 1.0), cu, cv, &s_top);
            bool okb = hb > Z10_ ? wind_srh(w, hb, cu, cv, &s_bot) : true;
            if (okt && okb) o[O_EFHL] = s_top - s_bot;
        }
        double u5, v5;
        if (wind_at(w, 500.0, &u5, &v5)) {
            double sx = u5 - w.u10, sy = v5 - w.v10;
            double ix = w.u10 - cu, iy = w.v10 - cv;
            double a = wm_sqrt(sx * sx + sy * sy), b = wm_sqrt(ix * ix + iy * iy);
            if (a > 0.0 && b > 0.0) {
                double cosang = (sx * ix + sy * iy) / (a * b);
                cosang = fmin_(fmax_(cosang, -1.0), 1.0);
                // acos(x) = 2 atan(sqrt((1 - x) / (1 + x))); atan by two half-angle
                // reductions (tan(a/2) = t / (1 + sqrt(1 + t^2))) and its series.
                double t = wm_sqrt((1.0 - cosang) / (1.0 + cosang + 1.0e-300));
                bool inv = t > 1.0;
                double x = inv ? 1.0 / t : t;
                double h = x / (1.0 + wm_sqrt(1.0 + x * x));
                h = h / (1.0 + wm_sqrt(1.0 + h * h));
                double h2 = h * h, term = h, at = 0.0;
                for (int i = 0; i < 24; ++i) { at += term / (double)(2 * i + 1); term *= -h2; }
                at *= 4.0;
                if (inv) at = 1.5707963267948966 - at;
                o[O_CANGLE] = 2.0 * at * 57.29577951308232;
            }
        }
    }
    double pblh = S(S_PBLH);
    if (winds && isfinite(pblh) && pblh > Z10_) {
        double up, vp;
        if (wind_layer_mean(w, 0.0, pblh, &up, &vp)) {
            o[O_UPBL] = up;
            o[O_VPBL] = vp;
            o[O_VRATE] = pblh * wm_sqrt(up * up + vp * vp);
        }
    }

    // ---- ESP: 0-2 km lapse rate and 0-3 km MLCAPE
    {
        double t0 = e.V(V_TK, 0);
        double t2k = at_height(e, V_TK, zsfc + 2000.0);
        double z0 = e.V(V_ZM, 0) - zsfc;
        double mlcape = S(S_MLCAPE);
        if (isfinite(t0) && isfinite(t2k) && isfinite(cape03) && isfinite(mlcape) && 2000.0 > z0) {
            double lapse = (t0 - t2k) / ((2000.0 - z0) / 1000.0);
            o[O_ESP] = (lapse < 7.0 || mlcape < 250.0) ? 0.0 : (cape03 / 50.0) * ((lapse - 7.0) / 1.0);
        }
    }

    // ---- dendritic layer depth (-12 C to -17 C)
    {
        double z261 = NANV, z256 = NANV;
        for (int k = 0; k + 1 < nz; ++k) {
            double ta = e.V(V_TK, k), tb = e.V(V_TK, k + 1);
            double za = e.V(V_ZM, k), zb = e.V(V_ZM, k + 1);
            if (!isfinite(z261) && ta >= 261.15 && tb < 261.15) z261 = za + (zb - za) * (ta - 261.15) / (ta - tb);
            if (!isfinite(z256) && ta >= 256.15 && tb < 256.15) z256 = za + (zb - za) * (ta - 256.15) / (ta - tb);
        }
        if (e.V(V_TK, 0) < 261.15 && !isfinite(z261)) z261 = zsfc;
        if (e.V(V_TK, 0) < 256.15 && !isfinite(z256)) z256 = zsfc;
        o[O_LAYTH] = (isfinite(z261) && isfinite(z256)) ? fmax_(z256 - z261, 0.0) : 0.0;
    }

    // ---- column integrals
    {
        double pw = 0.0, pws = 0.0, cw = 0.0, ci = 0.0, cp = 0.0, vil = 0.0;
        bool have_q = true, have_r = true;
        for (int k = 0; k < nz; ++k) {
            double dp = e.I(I_PINT, k) - e.I(I_PINT, k + 1);
            double r = fmax_(e.V(V_R, k), 0.0);
            double qc = e.V(V_QC, k), qi = e.V(V_QI, k), qr = e.V(V_QR, k), qs = e.V(V_QS, k), qg = e.V(V_QG, k);
            if (!(isfinite(qc) && isfinite(qi) && isfinite(qr) && isfinite(qs) && isfinite(qg))) have_q = false;
            double qt = r + (isfinite(qc) ? fmax_(qc, 0.0) : 0.0) + (isfinite(qi) ? fmax_(qi, 0.0) : 0.0)
                      + (isfinite(qr) ? fmax_(qr, 0.0) : 0.0) + (isfinite(qs) ? fmax_(qs, 0.0) : 0.0)
                      + (isfinite(qg) ? fmax_(qg, 0.0) : 0.0);
            double m = dp / G_ / (1.0 + qt);  // dry-air mass of the layer, kg m-2
            pw += r * m;
            double rs = rsat(e.V(V_TK, k), e.V(V_P, k));
            pws += rs * m;
            if (have_q) {
                cw += fmax_(qc, 0.0) * m;
                ci += fmax_(qi, 0.0) * m;
                cp += (fmax_(qr, 0.0) + fmax_(qs, 0.0) + fmax_(qg, 0.0)) * m;
            }
            double dbz = e.V(V_REFL, k);
            if (!isfinite(dbz)) { have_r = false; continue; }
            double zl = wm_pow(10.0, fmin_(dbz, VIL_CAP_DBZ) / 10.0);
            if (dbz > 0.0 || zl > 0.0) vil += 3.44e-6 * wm_pow(zl, 4.0 / 7.0) * (e.I(I_ZINT, k + 1) - e.I(I_ZINT, k));
        }
        o[O_RHPW] = pws > 0.0 ? 100.0 * pw / pws : NANV;
        if (have_q) { o[O_TCOLW] = cw; o[O_TCOLI] = ci; o[O_TCOLP] = cp; }
        if (have_r) {
            o[O_VIL] = vil;
            double top = NANV;
            for (int k = nz - 1; k >= 0; --k) {
                double r0 = e.V(V_REFL, k);
                if (r0 >= ECHO_TOP_DBZ) {
                    if (k + 1 < nz) {
                        double r1 = e.V(V_REFL, k + 1);
                        double za = e.V(V_ZM, k), zb = e.V(V_ZM, k + 1);
                        top = za + (zb - za) * (r0 - ECHO_TOP_DBZ) / (r0 - r1);
                    } else {
                        top = e.V(V_ZM, k);
                    }
                    break;
                }
            }
            o[O_RETOP] = top;
            // reflectivity at the -10 C isotherm (lowest crossing), 1 km and 4 km AGL:
            // linear in Z (mm6 m-3), then dBZ
            double zm10 = NANV;
            for (int k = 0; k + 1 < nz; ++k) {
                double ta = e.V(V_TK, k), tb = e.V(V_TK, k + 1);
                if (ta >= 263.15 && tb < 263.15) {
                    zm10 = e.V(V_ZM, k) + (e.V(V_ZM, k + 1) - e.V(V_ZM, k)) * (ta - 263.15) / (ta - tb);
                    break;
                }
            }
            if (!isfinite(zm10) && e.V(V_TK, 0) < 263.15) zm10 = e.V(V_ZM, 0);
            double hs[3] = {zm10, zsfc + 1000.0, zsfc + 4000.0};
            int os[3] = {O_REFM10, O_REF1KM, O_REF4KM};
            for (int j = 0; j < 3; ++j) {
                double z = hs[j];
                if (!isfinite(z)) continue;
                for (int k = 0; k + 1 < nz; ++k) {
                    double za = e.V(V_ZM, k), zb = e.V(V_ZM, k + 1);
                    if (z >= za && z <= zb) {
                        double wgt = (z - za) / (zb - za);
                        double za_l = wm_pow(10.0, e.V(V_REFL, k) / 10.0), zb_l = wm_pow(10.0, e.V(V_REFL, k + 1) / 10.0);
                        double zl = za_l + (zb_l - za_l) * wgt;
                        o[os[j]] = zl > 0.0 ? 10.0 * wm_ln(zl) / wm_ln(10.0) : e.V(V_REFL, k);
                        break;
                    }
                }
                if (!isfinite(o[os[j]]) && z < e.V(V_ZM, 0)) o[os[j]] = e.V(V_REFL, 0);
            }
        }
    }

    // ---- boundary-layer cloud cover (maximum in the PBL), percent
    if (isfinite(pblh)) {
        double mx = NANV;
        for (int k = 0; k < nz; ++k) {
            double f = e.V(V_CLDBL, k);
            if (!isfinite(f)) { mx = NANV; break; }
            if (e.V(V_ZM, k) - zsfc <= pblh) mx = isfinite(mx) ? fmax_(mx, f) : f;
        }
        if (isfinite(mx)) o[O_TCDCBL] = 100.0 * fmin_(fmax_(mx, 0.0), 1.0);
    }

    // ---- RH at the freezing levels; pressure at cloud base and top
    {
        double hz[2] = {S(S_FRZH), S(S_HFRZH)};
        int oz[2] = {O_RHFRZ, O_RHHFRZ};
        for (int j = 0; j < 2; ++j) {
            double z = hz[j];
            if (!isfinite(z)) continue;
            double t = at_height(e, V_TK, z), r = at_height(e, V_R, z), p = p_at_height(e, z);
            if (isfinite(t) && isfinite(r) && isfinite(p)) o[oz[j]] = fmin_(100.0 * fmax_(r, 0.0) / rsat(t, p), 100.0);
        }
        double cb = S(S_CBASE), ct = S(S_CTOP);
        if (isfinite(cb)) o[O_PCBASE] = p_at_height(e, cb);
        if (isfinite(ct)) o[O_PCTOP] = p_at_height(e, ct);
    }

    // ---- total cloud cover from the layer covers (random overlap between layers)
    {
        double l = S(S_LCLD), m = S(S_MCLD), h = S(S_HCLD);
        if (isfinite(l) && isfinite(m) && isfinite(h))
            o[O_TCDC] = 100.0 * (1.0 - (1.0 - l / 100.0) * (1.0 - m / 100.0) * (1.0 - h / 100.0));
    }

    // ---- precipitation type and percent frozen from the last step's phases
    {
        double tot = S(S_RAINV), sn = S(S_SNOWV), gr = S(S_GRAUPV);
        if (isfinite(tot) && isfinite(sn) && isfinite(gr) && shelter) {
            double frozen = fmax_(sn, 0.0) + fmax_(gr, 0.0);
            double liquid = fmax_(tot - frozen, 0.0);
            double all = liquid + frozen;
            for (int j = O_CRAIN; j <= O_CFRZR; ++j) o[j] = 0.0;
            if (all > 0.0) {
                double ff = frozen / all;
                o[O_CPOFP] = 100.0 * ff;
                if (ff >= 0.5) {
                    if (fmax_(gr, 0.0) > fmax_(sn, 0.0)) o[O_CICEP] = 1.0; else o[O_CSNOW] = 1.0;
                } else if (col.t2 <= 273.15) {
                    o[O_CFRZR] = 1.0;
                } else {
                    o[O_CRAIN] = 1.0;
                }
            } else {
                o[O_CPOFP] = -50.0;
            }
        }
    }
    #undef S
    for (int i = 0; i < O_COUNT; ++i) out[(long long)i * N + c] = (float)o[i];
}

// Interpolate `nf` mass-level fields to `nl` pressure levels (Pa), linear in
// ln p between mass levels; the lowest level is held below it; NaN above the
// top mass level.  in: nf * nz * n; out: [field][level][cell].
extern "C" __global__ void rrfs_upp_isobaric_v1(
    int nz, int n, int nf, int nl,
    const float* __restrict__ p, const float* __restrict__ fields, const double* __restrict__ levels,
    float* __restrict__ out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    long long N = n, vol = (long long)nz * N;
    for (int l = 0; l < nl; ++l) {
        double pl = levels[l];
        int k0 = -1;
        double w = 0.0;
        bool below = false, above = false;
        double p0 = (double)p[c];
        if (pl >= p0) {
            below = true;
        } else if (pl < (double)p[(long long)(nz - 1) * N + c]) {
            above = true;
        } else {
            for (int k = 0; k + 1 < nz; ++k) {
                double pa = (double)p[(long long)k * N + c], pb = (double)p[(long long)(k + 1) * N + c];
                if (pl <= pa && pl >= pb) {
                    k0 = k;
                    w = wm_ln(pa / pl) / wm_ln(pa / pb);
                    break;
                }
            }
        }
        for (int f = 0; f < nf; ++f) {
            const float* fv = fields + (long long)f * vol;
            double v;
            if (!isfinite(p0)) v = NANV;
            else if (below) v = (double)fv[c];
            else if (above || k0 < 0) v = NANV;
            else {
                double a = (double)fv[(long long)k0 * N + c], b = (double)fv[(long long)(k0 + 1) * N + c];
                v = a + (b - a) * w;
            }
            out[((long long)f * nl + l) * N + c] = (float)v;
        }
    }
}

// Absolute vorticity of grid-relative winds on a conformal grid:
// zeta = m^2 [d(v/m)/dx - d(u/m)/dy] + f, centred differences, one-sided at
// the edges; NaN next to a missing point.
extern "C" __global__ void rrfs_upp_absv_v1(
    int nx, int ny, int nlev, double dx,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ mapf, const float* __restrict__ fcor,
    float* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)nx * ny;
    if (idx >= n * nlev) return;
    long long l = idx / n, c = idx % n;
    int i = (int)(c % nx), j = (int)(c / nx);
    const float* U = u + l * n;
    const float* V = v + l * n;
    int ia = i > 0 ? i - 1 : i, ib = i + 1 < nx ? i + 1 : i;
    int ja = j > 0 ? j - 1 : j, jb = j + 1 < ny ? j + 1 : j;
    long long cA = (long long)j * nx + ia, cB = (long long)j * nx + ib;
    long long cC = (long long)ja * nx + i, cD = (long long)jb * nx + i;
    double m = (double)mapf[c];
    double dvdx = ((double)V[cB] / (double)mapf[cB] - (double)V[cA] / (double)mapf[cA]) / ((double)(ib - ia) * dx);
    double dudy = ((double)U[cD] / (double)mapf[cD] - (double)U[cC] / (double)mapf[cC]) / ((double)(jb - ja) * dx);
    double z = m * m * (dvdx - dudy) + (double)fcor[c];
    out[idx] = (ib == ia || jb == ja) ? (float)NANV : (float)z;
}
