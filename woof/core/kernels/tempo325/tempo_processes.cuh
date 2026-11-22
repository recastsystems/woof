// TEMPO microphysics port for WOOF: process-rate routines.
// Line-by-line transliteration of NCAR/TEMPO c62699f (tag tempo_v3.2.5)
//   src/module_mp_tempo_main.F90 lines 2284-3500 (the v3.0.3 port is
//   ../tempo/tempo_processes.cuh; line notes below cite v3.0.3 where unchanged)
//   src/module_mp_tempo_aerosols.F90 (whole module)
// Apache-2.0, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
// Conventions: see tempo_common.cuh.  Column arrays are 1-based (a[1..nz]).
// Fortran quirks are preserved bug-for-bug; each is marked "Fortran quirk".
#pragma once
#include "tempo_common.cuh"

// ---------------------------------------------------------------------------
// module_mp_tempo_aerosols.F90
// ---------------------------------------------------------------------------

// aerosols.F90:14-42
TEMPO_HDN inline void init_water_friendly_aerosols(const TempoParams& P, int nz,
                                                   const double* dz1d, double* nwfa) {
  double hgt[TEMPO_NZP];
  double h_01, niccn3;
  for (int k = 0; k < TEMPO_NZP; ++k) hgt[k] = 0.0;
  // Fortran quirk: hgt(k) accumulates dz1d(k), not dz1d(k-1); hgt(1) = 0.
  for (int k = 2; k <= nz; ++k) hgt[k] = hgt[k - 1] + dz1d[k];
  if (hgt[1] <= 1000.0) {
    h_01 = 0.8;
  } else if (hgt[1] >= 2500.0) {
    h_01 = 0.01;
  } else {
    h_01 = 0.8 * cos(hgt[1] * 0.001 - 1.0);
  }
  niccn3 = -1.0 * tempo_log(PS(naccn1) / PS(naccn0)) / h_01;
  // Fortran quirk: level 1 uses the level-2 height.
  nwfa[1] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[2] - hgt[1]) / 1000.0) * niccn3);
  for (int k = 2; k <= nz; ++k) {
    nwfa[k] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[k] - hgt[1]) / 1000.0) * niccn3);
  }
}

// aerosols.F90:45-73
TEMPO_HDN inline void init_ice_friendly_aerosols(const TempoParams& P, int nz,
                                                 const double* dz1d, double* nifa) {
  double hgt[TEMPO_NZP];
  double h_01, niin3;
  for (int k = 0; k < TEMPO_NZP; ++k) hgt[k] = 0.0;
  for (int k = 2; k <= nz; ++k) hgt[k] = hgt[k - 1] + dz1d[k];
  if (hgt[1] <= 1000.0) {
    h_01 = 0.8;
  } else if (hgt[1] >= 2500.0) {
    h_01 = 0.01;
  } else {
    h_01 = 0.8 * cos(hgt[1] * 0.001 - 1.0);
  }
  niin3 = -1.0 * tempo_log(PS(nain1) / PS(nain0)) / h_01;
  nifa[1] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[2] - hgt[1]) / 1000.0) * niin3);
  for (int k = 2; k <= nz; ++k) {
    nifa[k] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[k] - hgt[1]) / 1000.0) * niin3);
  }
}

// aerosols.F90:76-118
TEMPO_HDN inline double aerosol_collection_efficiency(const TempoParams& P, double d, double da,
                                                      double visc, double rhoa, double temp,
                                                      char species) {
  double aval, cc, diff, re, sc, st, st2, vt, eff, rho_p;
  const double boltzman = 1.3806503e-23;
  const double meanpath = 0.0256e-6;
  vt = 1.0;
  rho_p = PS(rho_w);
  if (species == 'r') {
    vt = -0.1021 + 4.932e3 * d - 0.9551e6 * d * d + 0.07934e9 * d * d * d
       - 0.002362e12 * d * d * d * d;
    rho_p = PS(rho_w);
  } else if (species == 's') {
    vt = PS(av_s) * tempo_pow(d, PS(bv_s));
    rho_p = PS(rho_s);
  } else if (species == 'g') {
    vt = PA1(av_g, PI(idx_bg1)) * tempo_pow(d, PA1(bv_g, PI(idx_bg1)));
    rho_p = PA1(rho_g, PI(idx_bg1));
  }
  cc = 1.0 + 2.0 * meanpath / da * (1.257 + 0.4 * tempo_exp(-0.55 * da / meanpath));
  diff = boltzman * temp * cc / (3.0 * PS(pi) * visc * da);
  re = 0.5 * rhoa * d * vt / visc;
  sc = visc / (rhoa * diff);
  st = (rho_p - rhoa) * da * da * vt * cc / (9.0 * visc * d);
  aval = tempo_log(1.0 + re);
  st2 = (1.2 + 1.0 / 12.0 * aval) / (1.0 + aval);
  eff = 4.0 / (re * sc) * (1.0 + 0.4 * sqrt(re) * tempo_pow(sc, 0.3333) + 0.16 * sqrt(re) * sqrt(sc))
      + 4.0 * da / d * (0.02 + da / d * (1.0 + 2.0 * sqrt(re)));
  if (st > st2) eff = eff + tempo_pow(((st - st2) / (st - st2 + 0.666667)), 1.5);
  return tmax(1.e-5, tmin(eff, 1.0));
}

// ---------------------------------------------------------------------------
// table index helpers (main.F90:2459-2601, 2802-2821)
// ---------------------------------------------------------------------------

// Fortran pattern: do nn = m-1, m+1; n = nn; if (x/10**nn in [1,10)) exit
// enddo.  Without an exit n keeps the last value m+1.
TEMPO_HD int tempo_decade(double x, int m) {
  int n = m - 1;
  for (int nn = m - 1; nn <= m + 1; ++nn) {
    n = nn;
    if ((x / tpowi(10.0, nn)) >= 1.0 && (x / tpowi(10.0, nn)) < 10.0) break;
  }
  return n;
}

// main.F90:2459-2474
TEMPO_HDN inline void get_snow_table_index(const TempoParams& P, double rs, int* idx_s) {
  int nis = tnint(tempo_log10(rs));
  int n = tempo_decade(rs, nis);
  int v = (int)(rs / tpowi(10.0, n)) + 10 * (n - PI(nis2)) - (n - PI(nis2));
  *idx_s = imax(1, imin(v, PI(ntb_s)));
}

// main.F90:2477-2489
TEMPO_HDN inline void get_temperature_table_index(const TempoParams& P, double tempk, int* idx_t) {
  double tempc = tempk - PS(t0);
  int v = (int)((tempc - 2.5) / 5.0) - 1;
  v = imax(1, -v);
  *idx_t = imin(v, PI(ntb_t));
}

// main.F90:2492-2521
TEMPO_HDN inline void get_rain_table_index(const TempoParams& P, double rr, double ilamr,
                                           int* idx_r, int* idx_r1) {
  double lamr, lam_exp, n0_exp;
  int nir = tnint(tempo_log10(rr));
  int n = tempo_decade(rr, nir);
  int v = (int)(rr / tpowi(10.0, n)) + 10 * (n - PI(nir2)) - (n - PI(nir2));
  *idx_r = imax(1, imin(v, PI(ntb_r)));

  lamr = 1. / ilamr;
  lam_exp = lamr * tempo_pow((PA1(crg, 3) * PS(org2) * PS(org1)), PS(bm_r));
  n0_exp = PS(org1) * rr / PS(am_r) * tempo_pow(lam_exp, PA1(cre, 1));
  nir = tnint(tempo_log10(n0_exp));
  n = tempo_decade(n0_exp, nir);
  v = (int)(n0_exp / tpowi(10.0, n)) + 10 * (n - PI(nir3)) - (n - PI(nir3));
  *idx_r1 = imax(1, imin(v, PI(ntb_r1)));
}

// main.F90:2524-2554
TEMPO_HDN inline void get_graupel_table_index(const TempoParams& P, double rg, double ilamg,
                                              int idx, int* idx_g, int* idx_g1) {
  double lamg, lam_exp, n0_exp;
  int nig = tnint(tempo_log10(rg));
  int n = tempo_decade(rg, nig);
  int v = (int)(rg / tpowi(10.0, n)) + 10 * (n - PI(nig2)) - (n - PI(nig2));
  *idx_g = imax(1, imin(v, PI(ntb_g)));

  lamg = 1. / ilamg;
  lam_exp = lamg * tempo_pow((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), PS(bm_g));
  n0_exp = PS(ogg1) * rg / PA1(am_g, idx) * tempo_pow(lam_exp, PA2(cge, 1, 1));
  nig = tnint(tempo_log10(n0_exp));
  n = tempo_decade(n0_exp, nig);
  v = (int)(n0_exp / tpowi(10.0, n)) + 10 * (n - PI(nig3)) - (n - PI(nig3));
  *idx_g1 = imax(1, imin(v, PI(ntb_g1)));
}

// main.F90:2557-2575
TEMPO_HDN inline void get_cloud_table_index(const TempoParams& P, double rc, double nc,
                                            int* idx_c, int* idx_n) {
  int nic = tnint(tempo_log10(rc));
  int n = tempo_decade(rc, nic);
  int v = (int)(rc / tpowi(10.0, n)) + 10 * (n - PI(nic2)) - (n - PI(nic2));
  *idx_c = imax(1, imin(v, PI(ntb_c)));

  v = tnint(1.0 + (double)PI(nbc) * tempo_log(nc / PA1(t_nc, 1)) / PS(nic1));
  *idx_n = imax(1, imin(v, PI(nbc)));
}

// main.F90:2578-2601
TEMPO_HDN inline void get_ice_table_index(const TempoParams& P, double ri, double ni,
                                          int* idx_i, int* idx_i1) {
  int nii = tnint(tempo_log10(ri));
  int n = tempo_decade(ri, nii);
  int v = (int)(ri / tpowi(10.0, n)) + 10 * (n - PI(nii2)) - (n - PI(nii2));
  *idx_i = imax(1, imin(v, PI(ntb_i)));

  nii = tnint(tempo_log10(ni));
  n = tempo_decade(ni, nii);
  v = (int)(ni / tpowi(10.0, n)) + 10 * (n - PI(nii3)) - (n - PI(nii3));
  *idx_i1 = imax(1, imin(v, PI(ntb_i1)));
}

// main.F90:2802-2821
TEMPO_HDN inline void get_in_table_index(const TempoParams& P, double xni, int* idx_in) {
  if (xni > PA1(nt_in, 1)) {
    int niin = tnint(tempo_log10(xni));
    int n = tempo_decade(xni, niin);
    int v = (int)(xni / tpowi(10.0, n)) + 10 * (n - PI(niin2)) - (n - PI(niin2));
    *idx_in = imax(1, imin(v, PI(ntb_in)));
  } else {
    *idx_in = 1;
  }
}

// ---------------------------------------------------------------------------
// nucleation / activation functions
// ---------------------------------------------------------------------------

// v3.2.5 main.F90:2472-2499.  The caller passes supersaturation, so
// delta_aw = ssatw + 1 - a_w_i (v3.2.0 #48 Koop bug fix).
TEMPO_HDN inline double koop_nucleation(const TempoParams& P, const TempoCtx& C, double temp,
                                        double ssatw, double naero) {
  double xni, mu_diff, a_w_i, delta_aw, log_j_rate, j_rate, prob_h;
  xni = 0.0;
  mu_diff = 210368.0 + (131.438 * temp) - (3.32373e6 / temp) - (41729.1 * tempo_log(temp));
  a_w_i = tempo_exp(mu_diff / (PS(r_uni) * temp));
  delta_aw = ssatw + 1.0 - a_w_i;
  log_j_rate = -906.7 + (8502.0 * delta_aw) - (26924.0 * delta_aw * delta_aw)
             + (29180.0 * delta_aw * delta_aw * delta_aw);
  log_j_rate = tmin(20.0, log_j_rate);
  j_rate = tempo_pow(10.0, log_j_rate);
  prob_h = tmin(1.0 - tempo_exp(-j_rate * PS(ar_volume) * C.global_dt), 1.0);
  if (prob_h > 0.0) {
    xni = tmin(prob_h * naero, 1000.e3);
  }
  return tmax(0.0, xni);
}

// main.F90:2166-2244.  land is the optional Fortran dummy (nullptr = absent).
TEMPO_HDN inline double activate_cloud_number(const TempoParams& P, double temp, double w1d,
                                              double nwfa, const int* land) {
  double n_local, w_local;
  double a, b, c, d, t, u, x1, x2, y1, y2, nx, wy, fraction;
  double lower_lim_nuc_frac;
  int i, j, k, l, m, n;
  const int ntb_arc = PI(ntb_arc), ntb_arw = PI(ntb_arw), ntb_art = PI(ntb_art);

  n_local = nwfa * 1.e-6;
  if (n_local >= PA1(ta_na, ntb_arc)) {
    n_local = PA1(ta_na, ntb_arc) - 1.0;
  } else if (n_local <= PA1(ta_na, 1)) {
    n_local = PA1(ta_na, 1) + 1.0;
  }
  // Fortran DO: without an exit n ends at ntb_arc+1.
  for (n = 2; n <= ntb_arc; ++n) {
    if (n_local >= PA1(ta_na, n - 1) && n_local < PA1(ta_na, n)) break;
  }
  i = n;
  x1 = tempo_log(PA1(ta_na, i - 1));
  x2 = tempo_log(PA1(ta_na, i));

  w_local = w1d;
  if (w_local >= PA1(ta_ww, ntb_arw)) {
    w_local = PA1(ta_ww, ntb_arw) - 1.0;
  } else if (w_local <= PA1(ta_ww, 1)) {
    w_local = PA1(ta_ww, 1) + 0.001;
  }
  for (n = 2; n <= ntb_arw; ++n) {
    if (w_local >= PA1(ta_ww, n - 1) && w_local < PA1(ta_ww, n)) break;
  }
  j = n;
  y1 = tempo_log(PA1(ta_ww, j - 1));
  y2 = tempo_log(PA1(ta_ww, j));

  k = imax(1, imin(tnint((temp - PA1(ta_tk, 1)) * 0.1) + 1, ntb_art));

  l = 3;
  m = 2;

  lower_lim_nuc_frac = 0.;
  if (land != nullptr) {
    if (*land == 1) {
      lower_lim_nuc_frac = 0.;
    } else if (*land == 0) {
      lower_lim_nuc_frac = 0.15;
    } else {
      lower_lim_nuc_frac = 0.15;
    }
  }

  a = PA5(tnccn_act, i - 1, j - 1, k, l, m);
  b = PA5(tnccn_act, i, j - 1, k, l, m);
  c = PA5(tnccn_act, i, j, k, l, m);
  d = PA5(tnccn_act, i - 1, j, k, l, m);
  nx = tempo_log(n_local);
  wy = tempo_log(w_local);
  t = (nx - x1) / (x2 - x1);
  u = (wy - y1) / (y2 - y1);

  fraction = (1.0 - t) * (1.0 - u) * a + t * (1.0 - u) * b + t * u * c + (1.0 - t) * u * d;
  fraction = tmax(fraction, lower_lim_nuc_frac);

  return nwfa * fraction;
}

// main.F90:2786-2799
TEMPO_HDN inline double demott_nucleation(const TempoParams& P, double tempc, double rho,
                                          double nifa) {
  double xni, nifa_cc;
  xni = 0.0;
  nifa_cc = tmax(0.5, nifa * PS(rho_not0) * 1.e-6 / rho);
  xni = (5.94e-5 * tempo_pow((-tempc), 3.33)) * tempo_pow(nifa_cc, ((-0.0264 * (tempc)) + 0.0033));
  xni = xni * rho / PS(rho_not0) * 1000.0;
  return tmax(0.0, xni);
}

// ---------------------------------------------------------------------------
// condensation / evaporation
// ---------------------------------------------------------------------------

// v3.2.5 main.F90:2284-2361.  nwfa1d and ncsave are optional (nullptr = absent).
TEMPO_HDN inline void cloud_condensation(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* temp, const double* w1d, const double* ssatw,
    const double* lvap, const double* tcond, const double* diffu, const double* lvt2,
    const double* nwfa1d, const double* nwfa, const double* ncsave, const double* qv,
    const double* qvs, const bool* l_qc, const double* rc, const double* nc, TempoTend& T) {
  double clap, fcd, dfcd, xrc, xnc, orho, tempc, otemp, rvs, rvs_p, rvs_pp, gamsc, alphsc,
      xsat, t1_evap;
  double dc_star;
  int idx_d, idx_n, idx_c;
  const double orv = PS(orv), pi = PS(pi);
  for (int k = 1; k <= nz; ++k) {
    // v3.0.5 #29: CYCLE (v3.0.3 returned, leaving the rest of the column).
    if (fabs(ssatw[k]) < PS(eps)) continue;

    orho = 1.0 / rho[k];
    clap = (qv[k] - qvs[k]) / (1.0 + lvt2[k] * qvs[k]);
    for (int n = 1; n <= 3; ++n) {
      fcd = qvs[k] * tempo_exp(lvt2[k] * clap) - qv[k] + clap;
      dfcd = qvs[k] * lvt2[k] * tempo_exp(lvt2[k] * clap) + 1.0;
      clap = clap - fcd / dfcd;
    }
    xrc = rc[k] + clap * rho[k];
    xnc = 0.0;

    if (xrc > PS(r1)) {
      T.prw_vcd[k] = clap * C.global_inverse_dt;

      if (clap > PS(eps)) {
        if (nwfa1d != nullptr) {
          xnc = tmax(PS(nt_c_min), activate_cloud_number(P, temp[k], w1d[k], nwfa[k], nullptr));
        } else if (ncsave != nullptr) {
          xnc = ncsave[k];
        } else {
          xnc = PS(nt_c_l);
        }
        T.pnc_wcd[k] = 0.5 * (xnc - nc[k] + fabs(xnc - nc[k])) * C.global_inverse_dt * orho;
      } else if (l_qc[k] && ssatw[k] < -1.e-6 && clap < -PS(eps)) {
        tempc = temp[k] - PS(t0);
        otemp = 1.0 / temp[k];
        rvs = rho[k] * qvs[k];
        rvs_p = rvs * otemp * (lvap[k] * otemp * orv - 1.0);
        rvs_pp = rvs * (otemp * (lvap[k] * otemp * orv - 1.0) * otemp * (lvap[k] * otemp * orv - 1.0)
               + (-2.0 * lvap[k] * otemp * otemp * otemp * orv) + otemp * otemp);
        gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
        alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = tmax(1.e-9, alphsc);
        xsat = ssatw[k];
        if (fabs(xsat) < 1.e-9) xsat = 0.0;
        t1_evap = 2.0 * pi * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
                - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc);

        dc_star = sqrt(-2.0 * C.global_dt * t1_evap / (2.0 * pi) * 4.0 * diffu[k] * ssatw[k] * rvs
                       / PS(rho_w));
        idx_d = imax(1, imin((int)(1.e6 * dc_star), PI(nbc)));
        get_cloud_table_index(P, rc[k], nc[k], &idx_c, &idx_n);

        T.prw_vcd[k] = tmax(-rc[k] * 0.99 * orho * C.global_inverse_dt, T.prw_vcd[k]);
        T.pnc_wcd[k] = tmax(-nc[k] * 0.99 * orho * C.global_inverse_dt,
                            -PA3(tnc_wev, idx_d, idx_c, idx_n) * orho * C.global_inverse_dt);
      }
    } else {
      T.prw_vcd[k] = -rc[k] * orho * C.global_inverse_dt;
      T.pnc_wcd[k] = -nc[k] * orho * C.global_inverse_dt;
    }
  }
}

// main.F90:2029-2090
TEMPO_HDN inline void rain_evaporation(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* temp, const double* ssatw, const double* lvap,
    const double* tcond, const double* diffu, const double* vsc2, const double* rhof2,
    const double* qv, const double* qvs, const bool* l_qr, const double* rr,
    const double* nr, const double* ilamr, TempoTend& T) {
  double orho, tempc, otemp, rvs, rvs_p, rvs_pp, gamsc, alphsc, xsat, t1_evap, rate_max,
      eva_factor;
  double lamr, n0_r;
  const double orv = PS(orv), pi = PS(pi);
  for (int k = 1; k <= nz; ++k) {
    if (l_qr[k]) {
      if ((ssatw[k] < -PS(eps)) && T.prw_vcd[k] <= 0.0) {
        orho = 1.0 / rho[k];
        tempc = temp[k] - PS(t0);
        otemp = 1.0 / temp[k];
        rvs = rho[k] * qvs[k];
        rvs_p = rvs * otemp * (lvap[k] * otemp * orv - 1.0);
        rvs_pp = rvs * (otemp * (lvap[k] * otemp * orv - 1.0) * otemp * (lvap[k] * otemp * orv - 1.0)
               + (-2.0 * lvap[k] * otemp * otemp * otemp * orv) + otemp * otemp);
        gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
        alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = tmax(1.e-9, alphsc);
        xsat = tmin(-1.e-9, ssatw[k]);
        t1_evap = 2.0 * pi * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
                - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc);

        if (qv[k] / qvs[k] < 0.95 && rr[k] * orho <= 1.e-8) {
          T.prv_rev[k] = rr[k] * orho * C.global_inverse_dt;
        } else {
          lamr = 1.0 / ilamr[k];
          n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
          T.prv_rev[k] = t1_evap * diffu[k] * (-ssatw[k]) * n0_r * rvs
                       * (PS(t1_qr_ev) * tempo_pow(ilamr[k], PA1(cre, 10))
                          + PS(t2_qr_ev) * vsc2[k] * rhof2[k]
                            * (tempo_pow((lamr + 0.5 * PS(fv_r)), (-PA1(cre, 11)))));
          rate_max = tmin((rr[k] * orho * C.global_inverse_dt), (qvs[k] - qv[k]) * C.global_inverse_dt);
          T.prv_rev[k] = tmin(rate_max, T.prv_rev[k] * orho);
          if (T.prr_gml[k] > 0.0) {
            eva_factor = tmin(1.0, 0.01 + (0.99 - 0.01) * (tempc / 20.0));
            T.prv_rev[k] = T.prv_rev[k] * eva_factor;
          }
        }
        T.pnr_rev[k] = tmin(nr[k] * 0.99 * orho * C.global_inverse_dt, T.prv_rev[k] * nr[k] / rr[k]);
      }
    }
  }
}

// main.F90:2093-2133.  nc1d and ncsave are optional (nullptr = absent).
TEMPO_HDN inline void freeze_cloud_melt_ice(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const double* rho, const double* ocp, const double* lvap,
    const double* qi1d, const double* ni1d, double* qiten, double* niten,
    const double* qc1d, const double* nc1d, double* qcten, double* ncten, double* tten,
    const double* ncsave) {
  double xri, xrc, lfus2, xnc;
  for (int k = 1; k <= nz; ++k) {
    xri = tmax(0.0, qi1d[k] + qiten[k] * C.global_dt);
    if ((temp[k] > PS(t0)) && (xri > 0.0)) {
      qcten[k] = qcten[k] + xri * C.global_inverse_dt;
      ncten[k] = ncten[k] + ni1d[k] * C.global_inverse_dt;
      qiten[k] = qiten[k] - xri * C.global_inverse_dt;
      niten[k] = -ni1d[k] * C.global_inverse_dt;
      tten[k] = tten[k] - PS(lfus) * ocp[k] * xri * C.global_inverse_dt;
    }
    xrc = tmax(0.0, qc1d[k] + qcten[k] * C.global_dt);
    if ((temp[k] < PS(hgfrz)) && (xrc > 0.0)) {
      lfus2 = PS(lsub) - lvap[k];
      if (nc1d != nullptr) {
        xnc = nc1d[k] + ncten[k] * C.global_dt;
      } else if (ncsave != nullptr) {
        xnc = ncsave[k] / rho[k] + ncten[k] * C.global_dt;
      } else {
        xnc = PS(nt_c_l) / rho[k] + ncten[k] * C.global_dt;
      }
      qiten[k] = qiten[k] + xrc * C.global_inverse_dt;
      niten[k] = niten[k] + xnc * C.global_inverse_dt;
      qcten[k] = qcten[k] - xrc * C.global_inverse_dt;
      ncten[k] = ncten[k] - xnc * C.global_inverse_dt;
      tten[k] = tten[k] + lfus2 * ocp[k] * xrc * C.global_inverse_dt;
    }
  }
}

// ---------------------------------------------------------------------------
// warm rain, riming, collection
// ---------------------------------------------------------------------------

// main.F90:2247-2322
TEMPO_HDN inline void warm_rain(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const bool* l_qc, const double* rc, const double* nc,
    const double* ilamc, const double* mvd_c, const bool* l_qr, const double* rr,
    const double* nr, const double* mvd_r, TempoTend& T) {
  double lamr, lamc, n0_r;
  double ef_rr, dc_g, dc_b, xdc, zeta1, zeta, taud, tau, ef_rw;
  int nu_c, idx;
  const double d0r = PS(d0r), d0c = PS(d0c), am_r = PS(am_r);
  for (int k = 1; k <= nz; ++k) {
    if (l_qr[k]) {
      if (mvd_r[k] > d0r) {
        ef_rr = tmax(-0.1, 1.0 - tempo_exp(2300.0 * (mvd_r[k] - 1950.0e-6)));
        T.pnr_rcr[k] = ef_rr * 2.0 * nr[k] * rr[k];
      }
    }

    if (l_qc[k]) {
      if (rc[k] > 0.01e-3) {
        nu_c = get_nuc(P, nc[k]);
        lamc = 1.0 / ilamc[k];
        xdc = tmax(d0c * 1.e6, (tempo_pow((rc[k] / (am_r * nc[k])), PS(obmr))) * 1.e6);
        dc_g = (tempo_pow((PA2(ccg, 3, nu_c) * PA1(ocg2, nu_c)), PS(obmr)) / lamc) * 1.e6;
        dc_b = tempo_pow((xdc * xdc * xdc * dc_g * dc_g * dc_g - xdc * xdc * xdc * xdc * xdc * xdc),
                   (1.0 / 6.0));
        zeta1 = 0.5 * ((6.25e-6 * xdc * dc_b * dc_b * dc_b - 0.4)
                       + fabs(6.25e-6 * xdc * dc_b * dc_b * dc_b - 0.4));
        zeta = 0.027 * rc[k] * zeta1;
        taud = 0.5 * ((0.5 * dc_b - 7.5) + fabs(0.5 * dc_b - 7.5)) + PS(r1);
        tau = 3.72 / (rc[k] * taud);
        T.prr_wau[k] = zeta / tau;
        T.prr_wau[k] = tmin(rc[k] * C.global_inverse_dt, T.prr_wau[k]);
        T.pnr_wau[k] = T.prr_wau[k] / (am_r * nu_c * PS(autocon_nr_factor) * d0r * d0r * d0r);
        T.pnc_wau[k] = tmin(nc[k] * C.global_inverse_dt,
                            T.prr_wau[k] / (am_r * mvd_c[k] * mvd_c[k] * mvd_c[k]));
      }
    }

    if (l_qr[k] && l_qc[k]) {
      if (mvd_r[k] > d0r && mvd_c[k] > d0c) {
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        idx = 1 + (int)(PI(nbr) * tempo_log(mvd_r[k] / PA1(dr, 1)) / tempo_log(PA1(dr, PI(nbr)) / PA1(dr, 1)));
        idx = imin(idx, PI(nbr));
        ef_rw = PA2(t_efrw, idx, (int)(mvd_c[k] * 1.e6));
        n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
        T.prr_rcw[k] = rhof[k] * PS(t1_qr_qc) * ef_rw * rc[k] * n0_r
                     * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
        T.prr_rcw[k] = tmin(rc[k] * C.global_inverse_dt, T.prr_rcw[k]);
        T.pnc_rcw[k] = rhof[k] * PS(t1_qr_qc) * ef_rw * nc[k] * n0_r
                     * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
        T.pnc_rcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_rcw[k]);
      }
    }
  }
}

// v3.2.5 main.F90:2662-2810.  vtboost is intent(out).
TEMPO_HDN inline void riming(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const double* rhof, const double* visco, const bool* l_qc,
    const double* rc, const double* nc, const double* ilamc, const double* mvd_c,
    const bool* l_qs, const double* rs, const double* smo0, const double* smob,
    const double* smoc, const double* smoe, double* vtboost, const bool* l_qg,
    const double* rg, const double* ng, const double* ilamg, const int* idx,
    const bool* l_qh, const double* rh, const double* nh, const double* ilamh, TempoTend& T) {
  double xds, xdg, n0_g, lamc, xdh, n0_h;
  double ef_sw, vtg, stoke_g, const_ri, tempc, rime_dens, ef_gw, ef_hw;
  double t1_qg_qc, r_frac, g_frac, vts, tf, snow_dens_frac, t1_qh_qc;
  const int nrhg = PI(nrhg);
  int idxs, nu_c;
  const double t0 = PS(t0), eps = PS(eps), mtl = PS(meters3_to_liters);
  for (int k = 1; k <= nz; ++k) {
    tempc = temp[k] - t0;
    vtboost[k] = 1.0;
    if (l_qc[k] && l_qs[k]) {
      nu_c = get_nuc(P, nc[k]);
      lamc = 1.0 / ilamc[k];
      (void)nu_c;
      (void)lamc;
      xds = smoc[k] / smob[k];
      if ((mvd_c[k] > PS(d0c)) && (xds > PS(d0s))) {
        idxs = 1 + (int)(PI(nbs) * tempo_log(xds / PA1(ds, 1)) / tempo_log(PA1(ds, PI(nbs)) / PA1(ds, 1)));
        idxs = imin(idxs, PI(nbs));
        ef_sw = PA2(t_efsw, idxs, (int)(mvd_c[k] * 1.e6));
        T.prs_scw[k] = rhof[k] * PS(t1_qs_qc) * ef_sw * rc[k] * smoe[k];
        T.prs_scw[k] = tmin(rc[k] * C.global_inverse_dt, T.prs_scw[k]);
        T.pnc_scw[k] = rhof[k] * PS(t1_qs_qc) * ef_sw * nc[k] * smoe[k];
        T.pnc_scw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_scw[k]);

        if (temp[k] < t0) {
          if (T.prs_scw[k] > PS(rime_threshold) * T.prs_sde[k] && T.prs_sde[k] > eps) {
            r_frac = tmin(30.0, T.prs_scw[k] / T.prs_sde[k]);
            g_frac = tmin(PS(rime_conversion), 0.15 + (r_frac - 2.0) * .028);
            vtboost[k] = tmin(1.5, 1.1 + (r_frac - 2.) * .014);
            T.prg_scw[k] = g_frac * T.prs_scw[k];
            T.png_scw[k] = T.prg_scw[k] * smo0[k] / rs[k];
            vts = PS(av_s) * tempo_pow(xds, PS(bv_s)) * tempo_exp(-PS(fv_s) * xds);
            const_ri = -1.0 * (mvd_c[k] * 0.5e6) * vts / tmin(-0.1, tempc);
            const_ri = tmax(0.1, tmin(const_ri, 10.0));
            rime_dens = (0.051 + 0.114 * const_ri - 0.0055 * const_ri * const_ri) * 1000.0;
            if (rime_dens < 150.0) {
              g_frac = 0.0;
              T.prg_scw[k] = 0.0;
              T.png_scw[k] = 0.0;
            }
            snow_dens_frac = tmin(1.0, tmax(0.0, rs[k] * C.global_inverse_dt
                                  / (rs[k] * C.global_inverse_dt + T.prg_scw[k])));
            T.pbg_scw[k] = mtl * T.prg_scw[k]
                         / (PS(rho_s) * snow_dens_frac + rime_dens * (1.0 - snow_dens_frac));
            T.prs_scw[k] = (1.0 - g_frac) * T.prs_scw[k];
          }
        }
      }
    }

    // hail collecting cloud water (v3.2.0 #48)
    if (l_qh[k] && l_qc[k]) {
      if (rh[k] >= PA1(r_g, 1) && mvd_c[k] > PS(d0c)) {
        xdh = (PS(bm_g) + PS(mu_g) + 1.0) * ilamh[k];
        if (xdh > PS(d0g)) {
          ef_hw = 0.95;
          t1_qh_qc = PS(pi) * .25 * PA1(av_g, nrhg) * PA2(cgg, 9, nrhg);
          n0_h = nh[k] / ilamh[k];
          T.prh_hcw[k] = rhof[k] * t1_qh_qc * ef_hw * rc[k] * n0_h * tempo_pow(ilamh[k], PA2(cge, 9, nrhg));
          T.pnc_hcw[k] = rhof[k] * t1_qh_qc * ef_hw * nc[k] * n0_h * tempo_pow(ilamh[k], PA2(cge, 9, nrhg));
          T.pnc_hcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_hcw[k]);
        }
      }
    }

    if (l_qc[k] && l_qg[k]) {
      if (rg[k] >= PA1(r_g, 1) && mvd_c[k] > PS(d0c)) {
        xdg = (PS(bm_g) + PS(mu_g) + 1.0) * ilamg[k];
        vtg = rhof[k] * PA1(av_g, idx[k]) * PA2(cgg, 6, idx[k]) * PS(ogg3)
            * tempo_pow(ilamg[k], PA1(bv_g, idx[k]));
        stoke_g = mvd_c[k] * mvd_c[k] * vtg * PS(rho_w) / (9.0 * visco[k] * xdg);
        const_ri = -1.0 * (mvd_c[k] * 0.5e6) * vtg / tmin(-0.1, tempc);
        const_ri = tmax(0.1, tmin(const_ri, 10.0));
        rime_dens = (0.051 + 0.114 * const_ri - 0.0055 * const_ri * const_ri) * 1000.0;
        if (xdg > PS(d0g)) {
          // Fortran quirk: ef_gw is undefined when stoke_g is NaN; 0 here.
          ef_gw = 0.0;
          if (stoke_g >= 0.4 && stoke_g <= 10.0) {
            ef_gw = 0.55 * tempo_log10(2.51 * stoke_g);
          } else if (stoke_g < 0.4) {
            ef_gw = 0.0;
          } else if (stoke_g > 10.0) {
            ef_gw = 0.77;
          }
          if (temp[k] > t0)
            ef_gw = ef_gw * tmin(tmax(tempo_pow(10.0, (-0.33 * (temp[k] - t0))), 0.1), 1.0);
          t1_qg_qc = PS(pi) * .25 * PA1(av_g, idx[k]) * PA2(cgg, 9, idx[k]);
          n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
          T.prg_gcw[k] = rhof[k] * t1_qg_qc * ef_gw * rc[k] * n0_g * tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));
          T.pnc_gcw[k] = rhof[k] * t1_qg_qc * ef_gw * nc[k] * n0_g * tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));
          T.pnc_gcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_gcw[k]);
          if (temp[k] < t0) T.pbg_gcw[k] = mtl * T.prg_gcw[k] / rime_dens;

          if (temp[k] < t0) {
            if (T.prg_gcw[k] > eps && tempc > -8.0) {
              tf = 0.0;
              if (tempc >= -5.0 && tempc < -3.0) {
                tf = 0.5 * (-3.0 - tempc);
              } else if (tempc > -8.0 && tempc < -5.0) {
                tf = 0.33333333 * (8.0 + tempc);
              }
              T.pni_ihm[k] = 3.5e8 * tf * T.prg_gcw[k];
              T.pri_ihm[k] = PS(xm0i) * T.pni_ihm[k];
              T.prs_ihm[k] = T.prs_scw[k] / (T.prs_scw[k] + T.prg_gcw[k]) * T.pri_ihm[k];
              T.prg_ihm[k] = T.prg_gcw[k] / (T.prs_scw[k] + T.prg_gcw[k]) * T.pri_ihm[k];
            }
          }
        }
      }
    }
  }
}

// main.F90:2604-2692
TEMPO_HDN inline void rain_snow_rain_graupel(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const bool* l_qr, const double* rr, const double* nr,
    const double* ilamr, const bool* l_qs, const double* rs, const bool* l_qg,
    const double* rg, const double* ng, const double* ilamg, const int* idx,
    const bool* l_qh, const double* rh, const double* ilamh, TempoTend& T) {
  int idx_r, idx_r1, idx_s, idx_t, idx_g, idx_g1, idx_h, idx_h1;
  const double t0 = PS(t0), mtl = PS(meters3_to_liters);
  const int nrhg = PI(nrhg);
  for (int k = 1; k <= nz; ++k) {
    if (l_qr[k] && l_qs[k]) {
      if (rr[k] >= PA1(r_r, 1) && rs[k] >= PA1(r_s, 1)) {
        get_temperature_table_index(P, temp[k], &idx_t);
        get_rain_table_index(P, rr[k], ilamr[k], &idx_r, &idx_r1);
        get_snow_table_index(P, rs[k], &idx_s);
#define TQS(tab) PA4(tab, idx_s, idx_t, idx_r1, idx_r)
        if (temp[k] < t0) {
          T.prr_rcs[k] = -(TQS(tmr_racs2) + TQS(tcr_sacr2) + TQS(tmr_racs1) + TQS(tcr_sacr1));
          T.prs_rcs[k] = TQS(tmr_racs2) + TQS(tcr_sacr2) - TQS(tcs_racs1) - TQS(tms_sacr1);
          T.prg_rcs[k] = TQS(tmr_racs1) + TQS(tcr_sacr1) + TQS(tcs_racs1) + TQS(tms_sacr1);
          T.prr_rcs[k] = tmax(-rr[k] * C.global_inverse_dt, T.prr_rcs[k]);
          T.prs_rcs[k] = tmax(-rs[k] * C.global_inverse_dt, T.prs_rcs[k]);
          T.prg_rcs[k] = tmin((rr[k] + rs[k]) * C.global_inverse_dt, T.prg_rcs[k]);
          T.pnr_rcs[k] = TQS(tnr_racs1) + TQS(tnr_racs2) + TQS(tnr_sacr1) + TQS(tnr_sacr2);
          T.pnr_rcs[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rcs[k]);
          // v3.2.0 #48: graupel number from the racs1 + sacr1 parts only, and
          // frozen-drop volume at rho_g(nrhg) instead of rho_i.
          T.png_rcs[k] = TQS(tnr_racs1) + TQS(tnr_sacr1);
          T.png_rcs[k] = tmin(nr[k] * C.global_inverse_dt, T.png_rcs[k]);
          T.pbg_rcs[k] = mtl * T.prg_rcs[k] / PA1(rho_g, nrhg);
        } else {
          T.prs_rcs[k] = -TQS(tcs_racs1) - TQS(tms_sacr1) + TQS(tmr_racs2) + TQS(tcr_sacr2);
          T.prs_rcs[k] = tmax(-rs[k] * C.global_inverse_dt, T.prs_rcs[k]);
          T.prr_rcs[k] = -T.prs_rcs[k];
        }
#undef TQS
      }
    }

    if (l_qr[k] && l_qg[k]) {
      if (rr[k] >= PA1(r_r, 1) && rg[k] >= PA1(r_g, 1)) {
        get_temperature_table_index(P, temp[k], &idx_t);
        get_rain_table_index(P, rr[k], ilamr[k], &idx_r, &idx_r1);
        get_graupel_table_index(P, rg[k], ilamg[k], idx[k], &idx_g, &idx_g1);
#define TQG(tab) PA5(tab, idx_g1, idx_g, idx[k], idx_r1, idx_r)
        if (temp[k] < t0) {
          T.prg_rcg[k] = TQG(tmr_racg) + TQG(tcr_gacr);
          T.prg_rcg[k] = tmin(rr[k] * C.global_inverse_dt, T.prg_rcg[k]);
          T.prr_rcg[k] = -T.prg_rcg[k];
          T.pnr_rcg[k] = TQG(tnr_racg) + TQG(tnr_gacr);
          T.pnr_rcg[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rcg[k]);
          T.pbg_rcg[k] = mtl * T.prg_rcg[k] / PA1(rho_g, nrhg);
        } else {
          T.prr_rcg[k] = TQG(tcg_racg);
          T.prr_rcg[k] = tmin(rg[k] * C.global_inverse_dt, T.prr_rcg[k]);
          T.prg_rcg[k] = -T.prr_rcg[k];
          T.png_rcg[k] = TQG(tnr_racg);
          T.png_rcg[k] = tmin(ng[k] * C.global_inverse_dt, T.png_rcg[k]);
          T.pbg_rcg[k] = mtl * T.prg_rcg[k] / PA1(rho_g, idx[k]);
          T.pnr_rcg[k] = -1.5 * TQG(tnr_gacr);
        }
#undef TQG
      }
    }

    // rain-hail collection (v3.2.0 #48): graupel tables at the densest class
    if (l_qh[k] && l_qr[k]) {
      if (rh[k] >= PA1(r_g, 1)) {
        if (temp[k] < t0) {
          get_temperature_table_index(P, temp[k], &idx_t);
          get_rain_table_index(P, rr[k], ilamr[k], &idx_r, &idx_r1);
          get_graupel_table_index(P, rh[k], ilamh[k], nrhg, &idx_h, &idx_h1);
#define TQH(tab) PA5(tab, idx_h1, idx_h, nrhg, idx_r1, idx_r)
          T.prh_rch[k] = TQH(tmr_racg) + TQH(tcr_gacr);
          T.prh_rch[k] = tmin(rr[k] * C.global_inverse_dt, T.prh_rch[k]);
          T.prr_rch[k] = -T.prh_rch[k];
          T.pnr_rch[k] = TQH(tnr_racg) + TQH(tnr_gacr);
          T.pnr_rch[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rch[k]);
#undef TQH
        }
      }
    }
  }
}

// v3.2.5 main.F90:3070-3168.  nwfa1d and nifa1d are optional (nullptr =
// absent); nwfa and nifa are tempo_main's local number concentrations.
TEMPO_HDN inline void ice_nucleation(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const double* rho, const double* w1d, const double* qv,
    const double* qvsi, const double* ssati, const double* ssatw, const double* nwfa1d,
    const double* nifa1d, const double* nwfa, const double* nifa, const double* ni,
    const double* ns, const double* rc, const double* nc, const double* rr,
    const double* nr, const double* ilamr, TempoTend& T) {
  double rate_max, tempc, xni, xnc, rime_autoconv_ratio;
  (void)w1d;
  int idx_in, idx_r, idx_r1, idx_tc, idx_c, idx_n;
  const double t0 = PS(t0), xm0i = PS(xm0i);
  for (int k = 1; k <= nz; ++k) {
    if (temp[k] < t0) {
      tempc = temp[k] - t0;
      idx_tc = imax(1, imin(tnint(-tempc), 45));
      rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999;
      if (nifa1d != nullptr) {
        xni = demott_nucleation(P, tempc, rho[k], nifa[k]);
      } else {
        xni = 1.0 * 1000.0;
      }
      get_in_table_index(P, xni, &idx_in);

      if (rr[k] > PA1(r_r, 1)) {
        get_rain_table_index(P, rr[k], ilamr[k], &idx_r, &idx_r1);
        T.prg_rfz[k] = PA4(tpg_qrfz, idx_r, idx_r1, idx_tc, idx_in) * C.global_inverse_dt;
        T.pri_rfz[k] = PA4(tpi_qrfz, idx_r, idx_r1, idx_tc, idx_in) * C.global_inverse_dt;
        T.pni_rfz[k] = PA4(tni_qrfz, idx_r, idx_r1, idx_tc, idx_in) * C.global_inverse_dt;
        T.pnr_rfz[k] = PA4(tnr_qrfz, idx_r, idx_r1, idx_tc, idx_in) * C.global_inverse_dt;
        T.prg_rfz[k] = tmin(rr[k] * C.global_inverse_dt, T.prg_rfz[k]);
        T.pnr_rfz[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rfz[k]);
        T.png_rfz[k] = T.pnr_rfz[k];
        // v3.2.0 #48: fewer frozen drops become graupel the larger the
        // collection rate is against cloud-water autoconversion.
        if (T.prr_wau[k] < PS(eps)) {
          if (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k] > PS(eps)) {
            T.png_rfz[k] = 0.1 * T.pnr_rfz[k];
          }
        } else {
          rime_autoconv_ratio = (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k]) / T.prr_wau[k];
          rime_autoconv_ratio = tmax(tmin(rime_autoconv_ratio, 1.), 0.);
          T.png_rfz[k] = T.pnr_rfz[k] * tmax(tempo_pow(10., (-2.0 * rime_autoconv_ratio)), 0.1);
        }
      } else if (rr[k] > PS(r1) && temp[k] < PS(hgfrz)) {
        T.pri_rfz[k] = rr[k] * C.global_inverse_dt;
        T.pni_rfz[k] = nr[k] * C.global_inverse_dt;
      }
      T.pbg_rfz[k] = PS(meters3_to_liters) * T.prg_rfz[k] / PA1(rho_g, PI(nrhg));

      if (rc[k] > PA1(r_c, 1)) {
        get_cloud_table_index(P, rc[k], nc[k], &idx_c, &idx_n);
        T.pri_wfz[k] = PA4(tpi_qcfz, idx_c, idx_n, idx_tc, idx_in) * C.global_inverse_dt;
        T.pri_wfz[k] = tmin(rc[k] * C.global_inverse_dt, T.pri_wfz[k]);
        T.pni_wfz[k] = PA4(tni_qcfz, idx_c, idx_n, idx_tc, idx_in) * C.global_inverse_dt;
        T.pni_wfz[k] = tmin3(nc[k] * C.global_inverse_dt, T.pri_wfz[k] / (2.0 * xm0i), T.pni_wfz[k]);
      } else if (rc[k] > PS(r1) && temp[k] < PS(hgfrz)) {
        T.pri_wfz[k] = rc[k] * C.global_inverse_dt;
        T.pni_wfz[k] = nc[k] * C.global_inverse_dt;
      }

      // demott_nuc_tempc = -20 (params.F90:115, v3.2.0)
      if ((ssati[k] >= PS(demott_nuc_ssati)) || (ssatw[k] > PS(eps) && tempc < -20.0)) {
        if (nifa1d != nullptr) {
          xnc = demott_nucleation(P, tempc, rho[k], nifa[k]);
        } else {
          xnc = tmin(PS(icenuc_max), PS(tno) * tempo_exp(PS(ato) * (t0 - temp[k])));
        }
        xni = ni[k] + (T.pni_rfz[k] + T.pni_wfz[k]) * C.global_dt;
        T.pni_inu[k] = 0.5 * (xnc - xni + fabs(xnc - xni)) * C.global_inverse_dt;
        T.pri_inu[k] = tmin(rate_max, xm0i * T.pni_inu[k]);
        T.pni_inu[k] = T.pri_inu[k] / xm0i;
      }

      // v3.2.3 #51: snow number ns (v3.0.3 used smo0)
      xni = ns[k] + ni[k] + (T.pni_rfz[k] + T.pni_wfz[k] + T.pni_inu[k]) * C.global_dt;
      if (nwfa1d != nullptr) {
        if ((xni <= PS(max_ni)) && (temp[k] < 238.0) && (ssati[k] >= 0.4)) {
          xnc = koop_nucleation(P, C, temp[k], ssatw[k], nwfa[k]);
          T.pni_iha[k] = xnc * C.global_inverse_dt;
          T.pri_iha[k] = tmin(rate_max, xm0i * 0.1 * T.pni_iha[k]);
          T.pni_iha[k] = T.pri_iha[k] / (xm0i * 0.1);
        }
      }
    }
  }
}

// main.F90:2824-2851
TEMPO_HDN inline void get_t1_subl(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* temp, const double* qvsi, const double* tcond,
    const double* diffu, const double* ssati, double* t1_subl) {
  double otemp, rvs, rvs_p, rvs_pp, gamsc, alphsc, xsat;
  const double lsub = PS(lsub), orv = PS(orv);
  (void)C;
  for (int k = 1; k <= nz; ++k) {
    otemp = 1.0 / temp[k];
    rvs = rho[k] * qvsi[k];
    rvs_p = rvs * otemp * (lsub * otemp * orv - 1.0);
    rvs_pp = rvs * (otemp * (lsub * otemp * orv - 1.0) * otemp * (lsub * otemp * orv - 1.0)
           + (-2. * lsub * otemp * otemp * otemp * orv) + otemp * otemp);
    gamsc = lsub * diffu[k] / tcond[k] * rvs_p;
    alphsc = 0.5 * (gamsc / (1.0 + gamsc)) * (gamsc / (1.0 + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
    alphsc = tmax(1.e-9, alphsc);
    xsat = ssati[k];
    if (fabs(xsat) < 1.e-9) xsat = 0.0;
    t1_subl[k] = 4.0 * PS(pi) * (1.0 - alphsc * xsat + 2.0 * alphsc * alphsc * xsat * xsat
               - 5.0 * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0 + gamsc);
  }
}

// main.F90:2854-2980
TEMPO_HDN inline void ice_processes(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const double* rhof2, const double* rho, const double* w1d,
    const double* temp, const double* qv, const double* qvsi, const double* tcond,
    const double* diffu, const double* vsc2, const double* ssati, const bool* l_qi,
    const double* ri, const double* ni, const double* ilami, const bool* l_qs,
    const double* rs, const double* smoe, const double* smof, const double* smo1,
    const double* rr, const double* nr, const double* ilamr, const double* mvd_r,
    const bool* l_qg, const double* rg, const double* ng, const double* ilamg,
    const int* idx, const bool* l_qh, const double* rh, const double* nh,
    const double* ilamh, TempoTend& T) {
  double xdi, xmi, oxmi, c_snow, rate_max, otemp, rvs, t2_qg_sd, rime_autoconv_ratio;
  double lamr, n0_r, n0_g, n0_h;
  const int nrhg = PI(nrhg);
  int idx_i, idx_i1;
  double t1_subl[TEMPO_NZP];
  const double t0 = PS(t0), c_cube = PS(c_cube), c_sqrd = PS(c_sqrd);
  const double bm_i = PS(bm_i), mu_i = PS(mu_i), am_i = PS(am_i), d0i = PS(d0i);
  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0;
  get_t1_subl(P, C, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl);

  for (int k = 1; k <= nz; ++k) {
    otemp = 1.0 / temp[k];
    rvs = rho[k] * qvsi[k];
    rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999;

    if (temp[k] < t0) {
      if (l_qi[k]) {
        get_ice_table_index(P, ri[k], ni[k], &idx_i, &idx_i1);
        xdi = tmax(d0i, (bm_i + mu_i + 1.) * ilami[k]);
        xmi = am_i * tempo_pow(xdi, bm_i);
        oxmi = 1.0 / xmi;
        T.pri_ide[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * PS(oig1) * PA1(cig, 5)
                     * ni[k] * ilami[k];
        if (T.pri_ide[k] < 0.0) {
          T.pri_ide[k] = tmax3(-ri[k] * C.global_inverse_dt, T.pri_ide[k], rate_max);
          T.pni_ide[k] = T.pri_ide[k] * oxmi;
          T.pni_ide[k] = tmax(-ni[k] * C.global_inverse_dt, T.pni_ide[k]);
        } else {
          T.pri_ide[k] = tmin(T.pri_ide[k], rate_max);
          T.prs_ide[k] = (1.0 - PA2(tpi_ide, idx_i, idx_i1)) * T.pri_ide[k];
          T.pri_ide[k] = PA2(tpi_ide, idx_i, idx_i1) * T.pri_ide[k];
        }

        if ((idx_i == PI(ntb_i)) || (xdi > 5.0 * PS(d0s))) {
          T.prs_iau[k] = ri[k] * .99 * C.global_inverse_dt;
          T.pni_iau[k] = ni[k] * .95 * C.global_inverse_dt;
        } else if (xdi < 0.1 * PS(d0s)) {
          T.prs_iau[k] = 0.0;
          T.pni_iau[k] = 0.0;
        } else {
          T.prs_iau[k] = PA2(tps_iaus, idx_i, idx_i1) * C.global_inverse_dt;
          T.prs_iau[k] = tmin(ri[k] * .99 * C.global_inverse_dt, T.prs_iau[k]);
          T.pni_iau[k] = PA2(tni_iaus, idx_i, idx_i1) * C.global_inverse_dt;
          T.pni_iau[k] = tmin(ni[k] * .95 * C.global_inverse_dt, T.pni_iau[k]);
        }

        xdi = tmax(d0i, (bm_i + mu_i + 1.) * ilami[k]);
        xmi = am_i * tempo_pow(xdi, bm_i);
        oxmi = 1. / xmi;
        if (rs[k] >= PA1(r_s, 1)) {
          T.prs_sci[k] = PS(t1_qs_qi) * rhof[k] * PS(ef_si) * ri[k] * smoe[k];
          T.pni_sci[k] = T.prs_sci[k] * oxmi;
        }

        if (rr[k] >= PA1(r_r, 1) && mvd_r[k] > 4.0 * xdi) {
          lamr = 1.0 / ilamr[k];
          n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
          T.pri_rci[k] = rhof[k] * PS(t1_qr_qi) * PS(ef_ri) * ri[k] * n0_r
                       * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
          T.pnr_rci[k] = rhof[k] * PS(t1_qr_qi) * PS(ef_ri) * ni[k] * n0_r
                       * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
          T.pnr_rci[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rci[k]);
          T.png_rci[k] = T.pnr_rci[k];
          if (T.prr_wau[k] < PS(eps)) {
            if (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k] > PS(eps)) {
              T.png_rci[k] = 0.1 * T.pnr_rci[k];
            }
          } else {
            rime_autoconv_ratio = (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k]) / T.prr_wau[k];
            rime_autoconv_ratio = tmax(tmin(rime_autoconv_ratio, 1.), 0.);
            T.png_rci[k] = T.pnr_rci[k] * tmax(tempo_pow(10., (-2.0 * rime_autoconv_ratio)), 0.1);
          }
          T.pni_rci[k] = T.pri_rci[k] * oxmi;
          T.prr_rci[k] = rhof[k] * PS(t2_qr_qi) * PS(ef_ri) * ni[k] * n0_r
                       * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 8))));
          T.prr_rci[k] = tmin(rr[k] * C.global_inverse_dt, T.prr_rci[k]);
          T.prg_rci[k] = T.pri_rci[k] + T.prr_rci[k];
          // v3.2.0 #48 graupel-volume fix: meters3_to_liters, rho_g(nrhg)
          T.pbg_rci[k] = PS(meters3_to_liters) * T.prg_rci[k] / PA1(rho_g, nrhg);
        }
      }

      if (l_qs[k]) {
        c_snow = c_sqrd + (temp[k] - t0 + 1.5) * (c_cube - c_sqrd) / (-30.0 + 1.5);
        c_snow = tmax(c_sqrd, tmin(c_snow, c_cube));
        T.prs_sde[k] = c_snow * t1_subl[k] * diffu[k] * ssati[k] * rvs
                     * (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);
        if (T.prs_sde[k] < 0.0) {
          T.prs_sde[k] = tmax3(-rs[k] * C.global_inverse_dt, T.prs_sde[k], rate_max);
        } else {
          T.prs_sde[k] = tmin(T.prs_sde[k], rate_max);
        }
      }
      if (l_qg[k]) {
        if (ssati[k] < -PS(eps)) {
          n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
          t2_qg_sd = 0.28 * PS(sc3) * sqrt(PA1(av_g, idx[k])) * PA2(cgg, 11, idx[k]);
          T.prg_gde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_g
                       * (PS(t1_qg_sd) * tempo_pow(ilamg[k], PA2(cge, 10, 1))
                          + t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow(ilamg[k], PA2(cge, 11, idx[k])));
          if (T.prg_gde[k] < 0.0) {
            T.prg_gde[k] = tmax3(-rg[k] * C.global_inverse_dt, T.prg_gde[k], rate_max);
            T.png_gde[k] = T.prg_gde[k] * ng[k] / rg[k];
          } else {
            T.prg_gde[k] = tmin(T.prg_gde[k], rate_max);
          }
        }
      }
      if (l_qh[k]) {
        if (ssati[k] < -PS(eps)) {
          t2_qg_sd = 0.28 * PS(sc3) * sqrt(PA1(av_g, nrhg)) * PA2(cgg, 11, nrhg);
          n0_h = nh[k] / ilamh[k];
          T.prh_hde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_h
                       * (PS(t1_qg_sd) * tempo_pow(ilamh[k], PA2(cge, 10, 1))
                          + t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow(ilamh[k], PA2(cge, 11, nrhg)));
          if (T.prh_hde[k] < 0.0) {
            T.prh_hde[k] = tmax3(-rh[k] * C.global_inverse_dt, T.prh_hde[k], rate_max);
          } else {
            T.prh_hde[k] = tmin(T.prh_hde[k], rate_max);
          }
        }
      }
    }
  }
  (void)otemp;
}

// main.F90:2983-3080
TEMPO_HDN inline void melting(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof2, const double* rho, const double* temp, const double* qvsi,
    const double* tcond, const double* diffu, const double* vsc2, const double* ssati,
    const double* delqvs, const bool* l_qs, const double* rs, const double* smof,
    const double* smo0, const double* smo1, const bool* l_qg, const double* rg,
    const double* ng, const double* ilamg, const int* idx, const bool* l_qh,
    const double* rh, const double* nh, const double* ilamh, const double* hail_fraction,
    bool hail_fraction_flag, TempoTend& T) {
  double tempc, otemp, rvs, melt_f, t2_qg_me, t2_qg_sd, t2_qh_me, melt_constant_1,
      melt_constant_2;
  double n0_g, n0_melt, lamg, n0_h;
  const int nrhg = PI(nrhg);
  double t1_subl[TEMPO_NZP];
  const double t0 = PS(t0), c_cube = PS(c_cube), olfus = PS(olfus), lvap0 = PS(lvap0);
  const double pi = PS(pi), mtl = PS(meters3_to_liters);
  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0;
  get_t1_subl(P, C, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl);

  for (int k = 1; k <= nz; ++k) {
    otemp = 1.0 / temp[k];
    tempc = temp[k] - t0;
    rvs = rho[k] * qvsi[k];

    if (temp[k] > t0) {
      if (l_qs[k]) {
        T.prr_sml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k])
                     * (PS(t1_qs_me) * smo1[k] + PS(t2_qs_me) * rhof2[k] * vsc2[k] * smof[k]);

        if (T.prr_sml[k] > 0.0) {
          T.prr_sml[k] = T.prr_sml[k] + 4218.0 * olfus * tempc * (T.prr_rcs[k] + T.prs_scw[k]);
          T.prr_sml[k] = tmin(rs[k] * C.global_inverse_dt, tmax(0.0, T.prr_sml[k]));
          T.pnr_sml[k] = smo0[k] / rs[k] * T.prr_sml[k] * tempo_pow(10.0, (-0.25 * tempc));
          T.pnr_sml[k] = tmin(smo0[k] * C.global_inverse_dt, T.pnr_sml[k]);
        } else {
          T.prr_sml[k] = 0.0;
          T.pnr_sml[k] = 0.0;
          if (ssati[k] < 0.0) {
            // v3.2.3 #51: c_sqrd (plate/aggregate capacitance), as WRF
            T.prs_sde[k] = PS(c_sqrd) * t1_subl[k] * diffu[k] * ssati[k] * rvs
                         * (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);
            T.prs_sde[k] = tmax(-rs[k] * C.global_inverse_dt, T.prs_sde[k]);
          }
        }
      }

      if (l_qg[k]) {
        n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
        n0_melt = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
        if ((rg[k] * ng[k]) < 1.e-4) {
          lamg = 1. / ilamg[k];
          n0_melt = (1.e-4 / rg[k]) * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));
        }
        // v3.0.5 #29: ventilation coefficient 0.28 (v3.0.3 used 0.2)
        t2_qg_me = pi * 4.0 * c_cube * olfus * 0.28 * PS(sc3) * sqrt(PA1(av_g, idx[k]))
                 * PA2(cgg, 11, idx[k]);
        T.prr_gml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k]) * n0_melt
                     * (PS(t1_qg_me) * tempo_pow(ilamg[k], PA2(cge, 10, 1))
                        + t2_qg_me * rhof2[k] * vsc2[k] * tempo_pow(ilamg[k], PA2(cge, 11, idx[k])));
        T.prr_gml[k] = tmin(rg[k] * C.global_inverse_dt, tmax(0.0, T.prr_gml[k]));
        if (T.prr_gml[k] > 0.0) {
          melt_f = tmax(0.05, tmin(T.prr_gml[k] * C.global_dt / rg[k], 1.0));
          T.pbg_gml[k] = mtl * T.prr_gml[k]
                       / tmax(tmin(melt_f * PA1(rho_g, idx[k]), PA1(rho_g, nrhg)), PA1(rho_g, 1));
          // v3.2.0 #48 logistic number change from melting
          melt_constant_1 = tmax(tmin(5.0 * tempo_pow((hail_fraction[k] / 0.25), .33), 5.0), 0.0);
          if (!hail_fraction_flag) melt_constant_1 = 0.0;
          melt_constant_2 = (1.0 + tempo_exp(-1.215 * melt_constant_1))
                          / (1.0 + tempo_exp(1.215 * (temp[k] - t0 - melt_constant_1)));
          T.pnr_gml[k] = T.prr_gml[k] * ng[k] / rg[k] * tmax(tmin(melt_constant_2, 1.0), 0.0);
        } else {
          T.prr_gml[k] = 0.0;
          T.pnr_gml[k] = 0.0;
          T.pbg_gml[k] = 0.0;
          if (ssati[k] < 0.0) {
            t2_qg_sd = 0.28 * PS(sc3) * sqrt(PA1(av_g, idx[k])) * PA2(cgg, 11, idx[k]);
            T.prg_gde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_g
                         * (PS(t1_qg_sd) * tempo_pow(ilamg[k], PA2(cge, 10, 1))
                            + t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow(ilamg[k], PA2(cge, 11, idx[k])));
            T.prg_gde[k] = tmax(-rg[k] * C.global_inverse_dt, T.prg_gde[k]);
            T.png_gde[k] = T.prg_gde[k] * ng[k] / rg[k];
          }
        }
      }
      if (C.global_dt > PS(timestep_conversion_rime_to_rain)) {
        T.prr_rcw[k] = T.prr_rcw[k] + T.prs_scw[k] + T.prg_gcw[k];
        T.prs_scw[k] = 0.0;
        T.prg_gcw[k] = 0.0;
      }

      if (l_qh[k]) {
        t2_qh_me = pi * 4.0 * c_cube * olfus * 0.28 * PS(sc3) * sqrt(PA1(av_g, nrhg)) * PA2(cgg, 11, nrhg);
        n0_h = nh[k] / ilamh[k];
        T.prr_hml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k]) * n0_h
                     * (PS(t1_qg_me) * tempo_pow(ilamh[k], PA2(cge, 10, 1))
                        + t2_qh_me * rhof2[k] * vsc2[k] * tempo_pow(ilamh[k], PA2(cge, 11, nrhg)));
        T.prr_hml[k] = tmin(rh[k] * C.global_inverse_dt, tmax(0.0, T.prr_hml[k]));
        if (T.prr_hml[k] > 0.0) {
          T.pnr_hml[k] = T.prr_hml[k] * nh[k] / rh[k];
        } else {
          T.prr_hml[k] = 0.0;
          T.pnr_hml[k] = 0.0;
        }
      }
    }
  }
  (void)otemp;
}

// main.F90:3083-3153
TEMPO_HDN inline void aerosol_scavenging(const TempoParams& P, const TempoCtx& C, int nz,
    const double* temp, const double* rho, const double* rhof, const double* visco,
    const double* nwfa, const double* nifa, const bool* l_qr, const double* nr,
    const double* ilamr, const double* mvd_r, const bool* l_qs, const double* rs,
    const double* smob, const double* smoc, const double* smoe, const bool* l_qg,
    const double* rg, const double* ng, const double* ilamg, const int* idx, TempoTend& T) {
  double ef_ra, ef_sa, ef_ga, t1_qg_qc;
  double n0_r, xds, xdg, n0_g, lamr;
  const double wf_aerosol_size = 0.04e-6;
  const double if_aerosol_size = 0.8e-6;
  for (int k = 1; k <= nz; ++k) {
    if (l_qr[k] && mvd_r[k] > PS(d0r)) {
      ef_ra = aerosol_collection_efficiency(P, mvd_r[k], wf_aerosol_size, visco[k], rho[k], temp[k], 'r');
      lamr = 1.0 / ilamr[k];
      n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
      T.pna_rca[k] = rhof[k] * PS(t1_qr_qc) * ef_ra * nwfa[k] * n0_r
                   * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
      T.pna_rca[k] = tmin(nwfa[k] * C.global_inverse_dt, T.pna_rca[k]);
      ef_ra = aerosol_collection_efficiency(P, mvd_r[k], if_aerosol_size, visco[k], rho[k], temp[k], 'r');
      T.pnd_rcd[k] = rhof[k] * PS(t1_qr_qc) * ef_ra * nifa[k] * n0_r
                   * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9))));
      T.pnd_rcd[k] = tmin(nifa[k] * C.global_inverse_dt, T.pnd_rcd[k]);
    }

    if (l_qs[k] && rs[k] > PA1(r_s, 1)) {
      xds = smoc[k] / smob[k];
      ef_sa = aerosol_collection_efficiency(P, xds, wf_aerosol_size, visco[k], rho[k], temp[k], 's');
      T.pna_sca[k] = rhof[k] * PS(t1_qs_qc) * ef_sa * nwfa[k] * smoe[k];
      T.pna_sca[k] = tmin(nwfa[k] * C.global_inverse_dt, T.pna_sca[k]);
      ef_sa = aerosol_collection_efficiency(P, xds, if_aerosol_size, visco[k], rho[k], temp[k], 's');
      T.pnd_scd[k] = rhof[k] * PS(t1_qs_qc) * ef_sa * nifa[k] * smoe[k];
      T.pnd_scd[k] = tmin(nifa[k] * C.global_inverse_dt, T.pnd_scd[k]);
    }

    if (l_qg[k] && rg[k] > PA1(r_g, 1)) {
      xdg = (PS(bm_g) + PS(mu_g) + 1.0) * ilamg[k];
      ef_ga = aerosol_collection_efficiency(P, xdg, wf_aerosol_size, visco[k], rho[k], temp[k], 'g');
      t1_qg_qc = PS(pi) * .25 * PA1(av_g, idx[k]) * PA2(cgg, 9, idx[k]);
      n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0 / ilamg[k]), PA2(cge, 2, 1));
      T.pna_gca[k] = rhof[k] * t1_qg_qc * ef_ga * nwfa[k] * n0_g * tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));
      T.pna_gca[k] = tmin(nwfa[k] * C.global_inverse_dt, T.pna_gca[k]);
      ef_ga = aerosol_collection_efficiency(P, xdg, if_aerosol_size, visco[k], rho[k], temp[k], 'g');
      T.pnd_gcd[k] = rhof[k] * t1_qg_qc * ef_ga * nifa[k] * n0_g * tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));
      T.pnd_gcd[k] = tmin(nifa[k] * C.global_inverse_dt, T.pnd_gcd[k]);
    }
  }
}
