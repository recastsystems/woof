// TEMPO microphysics port for WOOF: diagnostics.
// Line-by-line transliteration of NCAR/TEMPO c62699f (tag tempo_v3.2.5)
//   src/module_mp_tempo_diags.F90 (whole module)
// Apache-2.0, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
// Conventions: see tempo_common.cuh.  Column arrays are 1-based (a[1..nz]).
#pragma once
#include "tempo_common.cuh"

// ---------------------------------------------------------------------------
// Minimal complex(dp) arithmetic for the melting-reflectivity path (device
// safe, no std::complex).  Division uses Smith's algorithm; sqrt and log are
// the principal branches, as Fortran's intrinsics.  Results can differ from
// libgcc's __divdc3 / csqrt in the last bits.
// ---------------------------------------------------------------------------
struct tcplx {
  double re, im;
};
TEMPO_HD tcplx tc_make(double re, double im) { tcplx z; z.re = re; z.im = im; return z; }
TEMPO_HD tcplx tc_add(tcplx a, tcplx b) { return tc_make(a.re + b.re, a.im + b.im); }
TEMPO_HD tcplx tc_sub(tcplx a, tcplx b) { return tc_make(a.re - b.re, a.im - b.im); }
TEMPO_HD tcplx tc_mul(tcplx a, tcplx b) {
  return tc_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
TEMPO_HD tcplx tc_scale(double s, tcplx a) { return tc_make(s * a.re, s * a.im); }
TEMPO_HD tcplx tc_addr(tcplx a, double r) { return tc_make(a.re + r, a.im); }
TEMPO_HD tcplx tc_div(tcplx a, tcplx b) {
  if (fabs(b.re) >= fabs(b.im)) {
    double r = b.im / b.re;
    double den = b.re + b.im * r;
    return tc_make((a.re + a.im * r) / den, (a.im - a.re * r) / den);
  }
  double r = b.re / b.im;
  double den = b.re * r + b.im;
  return tc_make((a.re * r + a.im) / den, (a.im * r - a.re) / den);
}
TEMPO_HD double tc_abs(tcplx a) { return hypot(a.re, a.im); }
TEMPO_HD tcplx tc_conj(tcplx a) { return tc_make(a.re, -a.im); }
TEMPO_HD tcplx tc_sqrt(tcplx a) {
  double r = hypot(a.re, a.im);
  if (r == 0.0) return tc_make(0.0, 0.0);
  if (a.re >= 0.0) {
    double t = sqrt(0.5 * (r + a.re));
    return tc_make(t, a.im / (2.0 * t));
  }
  double t = sqrt(0.5 * (r - a.re));
  return tc_make(fabs(a.im) / (2.0 * t), copysign(t, a.im));
}
TEMPO_HD tcplx tc_log(tcplx a) { return tc_make(tempo_log(hypot(a.re, a.im)), atan2(a.im, a.re)); }

// diags.F90:13-56
TEMPO_HDN inline void effective_radius(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const bool* l_qc, const double* nc, const double* ilamc,
    const bool* l_qi, const double* ilami, const bool* l_qs, const double* rs,
    double* re_qc, double* re_qi, double* re_qs) {
  double smob, smoc;
  double tc0;
  int nu_c;
  (void)C;
  for (int k = 1; k <= nz; ++k) {
    re_qc[k] = 0.0;
    re_qi[k] = 0.0;
    re_qs[k] = 0.0;
    if (l_qc[k]) {
      nu_c = get_nuc(P, nc[k]);
      re_qc[k] = tmax(2.51e-6, tmin(0.5 * (3.0 + (double)nu_c) * ilamc[k], 50.e-6));
    }
    if (l_qi[k]) {
      re_qi[k] = tmax(2.51e-6, tmin(0.5 * (3.0 + PS(mu_i)) * ilami[k], 125.e-6));
    }
    if (l_qs[k]) {
      tc0 = tmin(-0.1, temp[k] - PS(t0));
      snow_moments(P, rs[k], tc0, &smob, &smoc, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr);
      re_qs[k] = tmax(5.01e-6, tmin(0.5 * (smoc / smob), 999.e-6));
    }
  }
}

// diags.F90:124-141
TEMPO_HDN inline int find_melting_level(const TempoParams& P, int nz, const double* temp,
                                        const bool* l_qr, const bool* l_qs, const bool* l_qg) {
  int k_melt = 1;
  for (int k = nz - 1; k >= 1; --k) {
    if ((temp[k] > PS(t0)) && l_qr[k] && (l_qs[k + 1] || l_qg[k + 1])) {
      k_melt = imax(k + 1, k_melt);
      break;
    }
  }
  return k_melt;
}

// diags.F90:144-172
TEMPO_HDN inline tcplx complex_water_ray(const TempoParams& P, double lambda, double t) {
  double epsinf, epss, epsr, epsi, alpha, lambdas, nenner;
  const double pi = PS(pi);
  epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t;
  epss = 78.54 * (1.0 - 4.579e-3 * (t - 25.0) + 1.190e-5 * (t - 25.0) * (t - 25.0)
                  - 2.800e-8 * (t - 25.0) * (t - 25.0) * (t - 25.0));
  alpha = -16.8129 / (t + 273.16) + 0.0609265;
  lambdas = 0.00033836 * tempo_exp(2513.98 / (t + 273.16)) * 1e-2;

  nenner = 1.0 + 2.0 * tempo_pow((lambdas / lambda), (1.0 - alpha)) * sin(alpha * pi * 0.5)
         + tempo_pow((lambdas / lambda), (2.0 - 2.0 * alpha));
  epsr = epsinf + ((epss - epsinf) * (tempo_pow((lambdas / lambda), (1.0 - alpha)) * sin(alpha * pi * 0.5)
                                      + 1.0)) / nenner;
  epsi = ((epss - epsinf) * (tempo_pow((lambdas / lambda), (1.0 - alpha)) * cos(alpha * pi * 0.5) + 0.0))
       / nenner + lambda * 1.25664 / 1.88496;

  return tc_sqrt(tc_make(epsr, -epsi));
}

// diags.F90:175-201
TEMPO_HDN inline tcplx complex_ice_maetzler(double lambda, double t) {
  double f, c, tk, b1, b2, b, deltabeta, betam, beta, theta, alfa, e;
  tcplx refractive_index;
  c = 2.99e8;
  tk = t + 273.16;
  f = c / lambda * 1e-9;
  b1 = 0.0207;
  b2 = 1.16e-11;
  b = 335.0;
  deltabeta = tempo_exp(-10.02 + 0.0364 * (tk - 273.16));
  e = tempo_exp(b / tk) - 1.0;
  betam = (b1 / tk) * (tempo_exp(b / tk) / (e * e)) + b2 * f * f;
  beta = betam + deltabeta;
  theta = 300.0 / tk - 1.0;
  alfa = (0.00504 + 0.0062 * theta) * tempo_exp(-22.1 * theta);

  refractive_index = tc_make(3.1884 + 9.1e-4 * (tk - 273.16), 0.0);
  refractive_index = tc_add(refractive_index, tc_make(0.0, (alfa / f + beta * f)));
  return tc_sqrt(tc_conj(refractive_index));
}

// Maxwell-Garnett beta term: 2*m1t/(m2t-m1t) * (m2t/(m2t-m1t)*log(m2t/m1t)-1)
TEMPO_HD tcplx tempo_mg_beta(tcplx m1t, tcplx mxt) {
  tcplx dm = tc_sub(mxt, m1t);
  tcplx inner = tc_addr(tc_mul(tc_div(mxt, dm), tc_log(tc_div(mxt, m1t))), -1.0);
  return tc_mul(tc_div(tc_scale(2.0, m1t), dm), inner);
}

// diags.F90:309-395
TEMPO_HDN inline void rayleigh_soak_wetgraupel(const TempoParams& P, double x_g, double a_geo,
    double b_geo, double fmelt, double lambda_radar, double meltratio_outside, tcplx m_w,
    tcplx m_i, double* backscatter) {
  double fm, mra, x_w, d_g, vg, rhog, meltratio_outside_grenz, volg, fmgrenz, d_large,
      volwater, volice, volair, volmix1, volmix2, vol1, vol2, vol3;
  tcplx m_core, m_air, m_tmp, m1t, m2t, m3t, beta2, beta3, num, den;
  const double pi = PS(pi);

  m_air = tc_make(1.0, 0.0);
  fm = tmax(tmin(fmelt, 1.0), 0.0);
  mra = tmax(tmin(meltratio_outside, 1.0), 0.0);
  mra = mra + (1.0 - mra) * fm;

  x_w = x_g * fm;
  d_g = a_geo * tempo_pow(x_g, b_geo);
  if (d_g >= PS(r1)) {
    vg = pi / 6.0 * tpowi(d_g, 3);
    rhog = tmax(tmin(x_g / vg, 900.0), 10.0);
    vg = x_g / rhog;
    meltratio_outside_grenz = 1.0 - rhog / 1000.0;

    if (mra <= meltratio_outside_grenz) {
      volg = vg * (1.0 - mra * fm);
    } else {
      fmgrenz = (900.0 - rhog) / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
      if (fm <= fmgrenz) {
        volg = (1.0 - mra * fm) * vg;
      } else {
        volg = (x_g - x_w) / 900.0 + x_w / 1000.0;
      }
    }

    d_large = tempo_pow((6.0 / pi * volg), (1.0 / 3.0));
    volice = (x_g - x_w) / (volg * 900.0);
    volwater = x_w / (1000.0 * volg);
    volair = 1.0 - volice - volwater;

    volmix1 = volice / tmax(volice + volwater, 1e-10);
    volmix2 = 1.0 - volmix1;

    m1t = tc_mul(m_w, m_w);
    m2t = tc_mul(m_air, m_air);
    m3t = tc_mul(m_i, m_i);
    vol1 = volmix2;
    vol2 = 0.0;
    vol3 = volmix1;
    (void)vol1;

    beta2 = tempo_mg_beta(m1t, m2t);
    beta3 = tempo_mg_beta(m1t, m3t);

    num = tc_add(tc_add(tc_scale((1.0 - vol2 - vol3), m1t), tc_mul(tc_scale(vol2, beta2), m2t)),
                 tc_mul(tc_scale(vol3, beta3), m3t));
    den = tc_add(tc_addr(tc_scale(vol2, beta2), 1.0 - vol2 - vol3), tc_scale(vol3, beta3));
    m_tmp = tc_sqrt(tc_div(num, den));

    m1t = tc_mul(m_tmp, m_tmp);
    m2t = tc_mul(m_air, m_air);
    m3t = tc_mul(tc_scale(2.0, m_air), tc_scale(2.0, m_air));
    vol1 = (1.0 - volair);
    vol2 = volair;
    vol3 = 0.0;

    beta2 = tempo_mg_beta(m1t, m2t);
    beta3 = tempo_mg_beta(m1t, m3t);

    num = tc_add(tc_add(tc_scale((1.0 - vol2 - vol3), m1t), tc_mul(tc_scale(vol2, beta2), m2t)),
                 tc_mul(tc_scale(vol3, beta3), m3t));
    den = tc_add(tc_addr(tc_scale(vol2, beta2), 1.0 - vol2 - vol3), tc_scale(vol3, beta3));
    m_core = tc_sqrt(tc_div(num, den));

    tcplx mc2 = tc_mul(m_core, m_core);
    double kk = tc_abs(tc_div(tc_addr(mc2, -1.0), tc_addr(mc2, 2.0)));
    *backscatter = tpowi(kk, 2) * pi * pi * pi * pi * pi * tpowi(d_large, 6)
                 / (lambda_radar * lambda_radar * lambda_radar * lambda_radar);
  } else {
    *backscatter = 0.0;
  }
}

// Simpson weights shared by the two binned melting routines (diags.F90:223-230, 273-280).
TEMPO_HD void tempo_simpson(int radar_bins, double* simpson) {
  const double basis1 = 1.0 / 3.0, basis2 = 4.0 / 3.0, basis3 = 1.0 / 3.0;
  for (int n = 1; n <= radar_bins + 1; ++n) simpson[n] = 0.0;
  for (int n = 1; n <= radar_bins - 1; n += 2) {
    simpson[n] = simpson[n] + basis1;
    simpson[n + 1] = simpson[n + 1] + basis2;
    simpson[n + 2] = simpson[n + 2] + basis3;
  }
}

#define TEMPO_RADAR_BINS_MAX 50

// diags.F90:204-252
TEMPO_HDN inline double reflectivity_from_melting_graupel(const TempoParams& P, double rg,
    double ng, double ilamg, int idx, double rr) {
  const double melt_outside = 0.9;
  const double lambda_radar = 0.10;
  const int radar_bins = PI(radar_bins);
  tcplx m_w_0, m_i_0;
  double simpson[TEMPO_RADAR_BINS_MAX + 2];
  double sr, fmelt, eta, lamg, backscatter, f_d, n0_g, mass, k_w;
  const double pi = PS(pi);

  tempo_simpson(radar_bins, simpson);
  m_w_0 = complex_water_ray(P, lambda_radar, 0.0);
  m_i_0 = complex_ice_maetzler(lambda_radar, 0.0);
  tcplx mw2 = tc_mul(m_w_0, m_w_0);
  k_w = tpowi(tc_abs(tc_div(tc_addr(mw2, -1.0), tc_addr(mw2, 2.0))), 2);

  // v3.2.x: guard the sum (diags.F90:179)
  sr = tmax(0.01, tmin(1.0 - rg / tmax(rg + rr, PS(r1)), 0.99));
  fmelt = sr * sr;
  eta = 0.0;
  lamg = 1.0 / ilamg;
  n0_g = ng * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));

  for (int n = 1; n <= radar_bins; ++n) {
    mass = PA1(am_g, idx) * tempo_pow(PA1(gbins_radar, n), PS(bm_g));
    rayleigh_soak_wetgraupel(P, mass, PA1(ocmg, idx), PS(obmg), fmelt, lambda_radar,
                             melt_outside, m_w_0, m_i_0, &backscatter);
    f_d = n0_g * tempo_pow(PA1(gbins_radar, n), PS(mu_g)) * tempo_exp(-lamg * PA1(gbins_radar, n));
    eta = eta + f_d * backscatter * simpson[n] * PA1(dgbins_radar, n);
  }
  return lambda_radar * lambda_radar * lambda_radar * lambda_radar / (pi * pi * pi * pi * pi * k_w)
       * eta;
}

// diags.F90:255-306
TEMPO_HDN inline double reflectivity_from_melting_snow(const TempoParams& P, double rs,
    double smob, double smoc, double rr) {
  const double melt_outside = 0.9;
  const double lambda_radar = 0.10;
  const int radar_bins = PI(radar_bins);
  tcplx m_w_0, m_i_0;
  double simpson[TEMPO_RADAR_BINS_MAX + 2];
  double sr, fmelt, eta, backscatter, f_d, mass, k_w, om3, m0, mrat, slam1, slam2;
  const double pi = PS(pi);

  tempo_simpson(radar_bins, simpson);
  m_w_0 = complex_water_ray(P, lambda_radar, 0.0);
  m_i_0 = complex_ice_maetzler(lambda_radar, 0.0);
  tcplx mw2 = tc_mul(m_w_0, m_w_0);
  k_w = tpowi(tc_abs(tc_div(tc_addr(mw2, -1.0), tc_addr(mw2, 2.0))), 2);

  sr = tmax(0.01, tmin(1.0 - rs / (rs + rr), 0.99));
  fmelt = sr * sr;
  eta = 0.0;

  om3 = 1.0 / smoc;
  m0 = (smob * om3);
  mrat = smob * m0 * m0 * m0;
  slam1 = m0 * PS(lam0);
  slam2 = m0 * PS(lam1);
  for (int n = 1; n <= radar_bins; ++n) {
    mass = PS(am_s) * tempo_pow(PA1(sbins_radar, n), PS(bm_s));
    rayleigh_soak_wetgraupel(P, mass, PS(ocms), PS(obms), fmelt, lambda_radar, melt_outside,
                             m_w_0, m_i_0, &backscatter);
    f_d = mrat * (PS(kap0) * tempo_exp(-slam1 * PA1(sbins_radar, n))
                  + PS(kap1) * tempo_pow((m0 * PA1(sbins_radar, n)), PS(mu_s)) * tempo_exp(-slam2 * PA1(sbins_radar, n)));
    eta = eta + f_d * backscatter * simpson[n] * PA1(dsbins_radar, n);
  }
  return lambda_radar * lambda_radar * lambda_radar * lambda_radar / (pi * pi * pi * pi * pi * k_w)
       * eta;
}

// diags.F90:59-121
TEMPO_HDN inline void reflectivity_10cm(const TempoParams& P, const TempoCtx& C, int nz,
    int refl10cm_from_melting_flag, const double* temp, const bool* l_qr, const double* rr,
    const double* nr, const double* ilamr, const bool* l_qs, const double* rs,
    const double* smoc, const double* smob, const double* smoz, const bool* l_qg,
    const double* rg, const double* ng, const int* idx, const double* ilamg, const bool* l_qh,
    const double* nh, const double* ilamh, double* dbz) {
  double ze_rain, ze_snow, ze_graupel, ze_hail;
  double n0_r, lamr, n0_g, n0_h;
  const int nrhg = PI(nrhg);
  int k_melt = 1;
  const double pi = PS(pi), am_s = PS(am_s);
  (void)C;

  if (refl10cm_from_melting_flag) {
    k_melt = find_melting_level(P, nz, temp, l_qr, l_qs, l_qg);
  }

  for (int k = nz; k >= 1; --k) {
    // v3.2.0 #48: floor 0.75e-22 per species (four species)
    ze_rain = 0.75e-22;
    ze_snow = 0.75e-22;
    ze_graupel = 0.75e-22;
    ze_hail = 0.75e-22;
    dbz[k] = -35.0;

    if (l_qr[k]) {
      lamr = 1.0 / ilamr[k];
      n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
      ze_rain = n0_r * PA1(crg, 4) * tempo_pow(ilamr[k], PA1(cre, 4));
    }
    if (l_qs[k]) {
      ze_snow = (0.176 / 0.93) * (6.0 / pi) * (6.0 / pi) * (am_s / 900.0) * (am_s / 900.0) * smoz[k];
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_snow = reflectivity_from_melting_snow(P, rs[k], smob[k], smoc[k], rr[k]);
        }
      }
    }
    if (l_qg[k]) {
      n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
      ze_graupel = (0.176 / 0.93) * (6.0 / pi) * (6.0 / pi) * (PA1(am_g, idx[k]) / 900.0)
                 * (PA1(am_g, idx[k]) / 900.0) * n0_g * PA2(cgg, 4, 1) * tempo_pow(ilamg[k], PA2(cge, 4, 1));
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_graupel = reflectivity_from_melting_graupel(P, rg[k], ng[k], ilamg[k], idx[k], rr[k]);
        }
      }
    }
    if (l_qh[k]) {
      n0_h = nh[k] / ilamh[k];
      ze_hail = (0.176 / 0.93) * (6.0 / pi) * (6.0 / pi) * (PA1(am_g, nrhg) / 900.0)
              * (PA1(am_g, nrhg) / 900.0) * n0_h * PA2(cgg, 4, 1) * tempo_pow(ilamh[k], PA2(cge, 4, 1));
    }
    dbz[k] = tmax(-35.0, 10.0 * tempo_log10((ze_rain + ze_snow + ze_graupel + ze_hail) * 1.e18));
  }
}

// diags.F90:398-438
TEMPO_HDN inline void max_hail_diam(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* rg, const double* ng, const double* ilamg,
    const int* idx, double* max_hail_diameter) {
  double lamg, n0_g, sum_nh, sum_t, f_d, hail_max;
  int n;
  const double threshold_conc = 0.0005;
  const int nhbins = PI(nhbins);
  (void)C;
  for (int k = 1; k <= nz; ++k) {
    max_hail_diameter[k] = 0.0;
    if (rg[k] / rho[k] >= 1.e-6) {
      if (PA1(rho_g, idx[k]) < 350.0) continue;
      lamg = 1.0 / ilamg[k];
      n0_g = ng[k] * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));

      sum_nh = 0.0;
      sum_t = 0.0;
      // Fortran DO: without an exit n ends at 0.
      for (n = nhbins; n >= 1; --n) {
        f_d = n0_g * tempo_pow(PA1(hbins, n), PS(mu_g)) * tempo_exp(-lamg * PA1(hbins, n)) * PA1(dhbins, n);
        sum_nh = sum_nh + f_d;
        if (sum_nh > threshold_conc) break;
        sum_t = sum_nh;
      }
      if (n >= nhbins) {
        hail_max = PA1(hbins, nhbins);
      } else if (PA1(hbins, n + 1) > 1.e-3) {
        // Fortran quirk: n = 0 here would read hbins(0); unreachable because
        // hbins(1) is about 0.53 mm, below the 1 mm test.
        hail_max = PA1(hbins, n) - (sum_nh - threshold_conc) / (sum_nh - sum_t)
                 * (PA1(hbins, n) - PA1(hbins, n + 1));
      } else {
        hail_max = 1.e-4;
      }
      max_hail_diameter[k] = 1000.0 * hail_max;
    }
  }
}

// diags.F90:441-456.  cloud_precip is optional in the Fortran; tempo_main always
// passes it, so it is taken by value here.
TEMPO_HDN inline void freezing_rain(const TempoParams& P, double temp, double rain_precip,
                                    double cloud_precip, double* frz_rain) {
  *frz_rain = 0.0;
  if ((temp - PS(t0)) < -0.5) {
    *frz_rain = rain_precip;
    *frz_rain = *frz_rain + cloud_precip;
  }
}
