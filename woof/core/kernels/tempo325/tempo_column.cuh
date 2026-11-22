// TEMPO column driver: port of module_mp_tempo_main.F90 lines 75-2281
// (tempo_main, compute_hail_fraction and the check/update, thermodynamic,
// depletion, tendency-sum, sedimentation and fallspeed routines) at TEMPO
// c62699f (tempo_v3.2.5).  The v3.0.3 port is ../tempo/tempo_column.cuh.
// Conventions: see tempo_common.cuh.
#pragma once
#include "tempo_common.cuh"
#include "tempo_processes.cuh"
#include "tempo_diags.cuh"

// All locals of tempo_main plus diagnostics.  One per column, in global
// memory on the GPU (too large for registers or stack).
struct TempoWork {
#define TEMPO_WORK_ARRAYS(WA) \
  WA(tten) WA(qvten) WA(qcten) WA(qiten) WA(qrten) WA(qsten) WA(qgten) WA(qbten) \
  WA(niten) WA(nrten) WA(ncten) WA(ngten) WA(nwfaten) WA(nifaten) \
  WA(temp) WA(pres) WA(qv) WA(rho) WA(rhof) WA(rhof2) WA(qvs) WA(qvsi) WA(delqvs) \
  WA(satw) WA(sati) WA(ssatw) WA(ssati) WA(diffu) WA(visco) WA(vsc2) WA(tcond) \
  WA(lvap) WA(ocp) WA(lvt2) \
  WA(rc) WA(ri) WA(rr) WA(rs) WA(rg) WA(rb) WA(ni) WA(nr) WA(nc) WA(ng) WA(nwfa) WA(nifa) \
  WA(ilamc) WA(ilami) WA(ilamr) WA(ilamg) WA(mvd_r) WA(mvd_c) WA(mvd_g)   WA(qhten) WA(rh) WA(nh) WA(qh1d) WA(ilamh) WA(vtrh) WA(hail_fraction) \
  WA(smob) WA(smo2) WA(smo1) WA(smo0) WA(smoc) WA(smoe) WA(smof) WA(smog) WA(ns) WA(smoz) \
  WA(xrx) WA(xnx) WA(xncx) WA(xngx) WA(xqbx) WA(ncsave) \
  WA(vtrr) WA(vtnr) WA(vtrs) WA(vtri) WA(vtni) WA(vtrg) WA(vtng) WA(vtrc) WA(vtnc) WA(vtboost) \
  WA(refl10cm) WA(re_cloud) WA(re_ice) WA(re_snow) WA(max_hail_diameter) \
  WA(rain_med_vol_diam) WA(graupel_med_vol_diam) WA(cloud_number_mixing_ratio) \
  WA(sl_zi) WA(sl_wi) WA(sl_za) WA(sl_dza) WA(sl_qa) WA(sl_qmi) WA(sl_qpi) \
  WA(sl_net_flx) WA(sl_precip_flx) WA(sl_rr_save) WA(sed_r)
#define TEMPO_WA_MEMBER(name) double name[TEMPO_NZP];
  TEMPO_WORK_ARRAYS(TEMPO_WA_MEMBER)
#undef TEMPO_WA_MEMBER
  bool l_qc[TEMPO_NZP], l_qi[TEMPO_NZP], l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP],
      l_qh[TEMPO_NZP];
  int idx_bg[TEMPO_NZP];
  TempoTend T;
  // ty_tempo_main_diags scalars
  double rain_precip, cloud_precip, ice_liquid_equiv_precip, snow_liquid_equiv_precip,
      graupel_liquid_equiv_precip, hail_liquid_equiv_precip, frozen_fraction, frz_rain_precip;
  int did_micro;  // 0 when tempo_main returned early (no hydrometeors, no supersaturation)
};

TEMPO_HD bool tempo_any(const bool* l, int nz) {
  for (int k = 1; k <= nz; ++k)
    if (l[k]) return true;
  return false;
}

// 828-848
TEMPO_HDN inline void aerosol_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* nwfa1d, const double* nifa1d, double* nwfa, double* nifa,
    double* nwfaten, double* nifaten) {
  for (int k = 1; k <= nz; ++k) {
    if (nwfa1d != nullptr) nwfa[k] = (nwfa1d[k] + nwfaten[k] * C.global_dt) * rho[k];
    if (nifa1d != nullptr) nifa[k] = (nifa1d[k] + nifaten[k] * C.global_dt) * rho[k];
    nwfa[k] = tmax(PS(nwfa_default) * rho[k], tmin(PS(aero_max) * rho[k], nwfa[k]));
    nifa[k] = tmax(PS(nifa_default) * rho[k], tmin(PS(aero_max) * rho[k], nifa[k]));
  }
}

// 851-941 (optional dt never passed by tempo_main: dt_ = global_dt)
TEMPO_HDN inline void cloud_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qc, double* qc1d, double* nc1d, const double* ncsave,
    double* rc, double* nc, double* qcten, double* ncten, double* ilamc, double* mvd_c) {
  const double dt_ = C.global_dt;
  const double odt_ = 1.0 / dt_;
  int nu_c;
  double lamc, xdc;
  for (int k = 1; k <= nz; ++k) {
    bool hit_limit = false;
    if (qc1d[k] + qcten[k] * dt_ > PS(r1)) {
      l_qc[k] = true;
      rc[k] = (qc1d[k] + qcten[k] * dt_) * rho[k];
      qc1d[k] = qc1d[k] + qcten[k] * dt_;
      if (nc1d != nullptr) {
        nc[k] = tmax(PS(nt_c_min), (nc1d[k] + ncten[k] * dt_) * rho[k]);
        if (nc[k] <= PS(nt_c_min)) {
          hit_limit = true;
          nc[k] = PS(nt_c_min);
        }
        if (nc[k] > PS(nt_c_max)) {
          hit_limit = true;
          nc[k] = PS(nt_c_max);
        }
        nu_c = get_nuc(P, nc[k]);
        lamc = tempo_pow((nc[k] * PS(am_r) * PA2(ccg, 2, nu_c) * PA1(ocg1, nu_c) / rc[k]), PS(obmr));
        xdc = (PS(bm_r) + nu_c + 1.0) / lamc;
        if (xdc < PS(d0c)) {
          lamc = PA2(cce, 2, nu_c) / PS(d0c);
          hit_limit = true;
        } else if (xdc > PS(d0r) * 2.0) {
          lamc = PA2(cce, 2, nu_c) / (PS(d0r) * 2.0);
          hit_limit = true;
        }
        nc[k] = PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * rc[k] / PS(am_r) * tempo_pow(lamc, PS(bm_r));
        if (hit_limit) ncten[k] = (nc[k] / rho[k] - nc1d[k]) * odt_;
        nc1d[k] = tmax(PS(nt_c_min) / rho[k],
            tmin(PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * qc1d[k] / PS(am_r) * tempo_pow(lamc, PS(bm_r)),
                 PS(nt_c_max) / rho[k]));
      } else {
        if (ncsave != nullptr) {
          nc[k] = ncsave[k];
        } else {
          nc[k] = PS(nt_c_l);
        }
      }
      nu_c = get_nuc(P, nc[k]);
      lamc = tempo_pow((nc[k] * PS(am_r) * PA2(ccg, 2, nu_c) * PA1(ocg1, nu_c) / rc[k]), PS(obmr));
      ilamc[k] = 1.0 / lamc;
      mvd_c[k] = tmax(tmin((3.0 + nu_c + 0.672) * ilamc[k], PS(d0r)), PS(d0c));
    } else {
      l_qc[k] = false;
      rc[k] = PS(r1);
      nc[k] = PS(nt_c_min);
      mvd_c[k] = PS(d0c);
      ilamc[k] = 0.0;
      qcten[k] = -qc1d[k] * odt_;
      qc1d[k] = 0.0;
      if (nc1d != nullptr) {
        ncten[k] = -nc1d[k] * odt_;
        nc1d[k] = 0.0;
      }
    }
  }
}

// 944-1009
TEMPO_HDN inline void rain_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qr, double* qr1d, double* nr1d, double* rr, double* nr,
    double* qrten, double* nrten, double* ilamr, double* mvd_r) {
  double lamr = 0.0;
  for (int k = 1; k <= nz; ++k) {
    bool hit_limit = false;
    if (qr1d[k] + qrten[k] * C.global_dt > PS(r1)) {
      l_qr[k] = true;
      rr[k] = (qr1d[k] + qrten[k] * C.global_dt) * rho[k];
      qr1d[k] = qr1d[k] + qrten[k] * C.global_dt;
      nr[k] = tmax(PS(r2), (nr1d[k] + nrten[k] * C.global_dt) * rho[k]);
      if (nr[k] <= PS(r2)) {
        hit_limit = true;
        mvd_r[k] = 1.0e-3;
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tempo_pow(lamr, PS(bm_r)) / PS(am_r);
      }
      lamr = tempo_pow((PS(am_r) * PA1(crg, 3) * PS(org2) * nr[k] / rr[k]), PS(obmr));
      mvd_r[k] = (3.0 + PS(mu_r) + 0.672) / lamr;
      if (mvd_r[k] > PS(d0r_max)) {
        hit_limit = true;
        mvd_r[k] = PS(d0r_max);
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tempo_pow(lamr, PS(bm_r)) / PS(am_r);
      } else if (mvd_r[k] < PS(d0r) * 0.75) {
        hit_limit = true;
        mvd_r[k] = PS(d0r) * 0.75;
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tempo_pow(lamr, PS(bm_r)) / PS(am_r);
      }
      if (hit_limit) nrten[k] = (nr[k] / rho[k] - nr1d[k]) * C.global_inverse_dt;
      nr1d[k] = PA1(crg, 2) * PS(org3) * qr1d[k] * tempo_pow(lamr, PS(bm_r)) / PS(am_r);
      ilamr[k] = 1.0 / lamr;
    } else {
      l_qr[k] = false;
      rr[k] = PS(r1);
      nr[k] = PS(r2);
      mvd_r[k] = PS(d0r);
      ilamr[k] = 0.0;
      qrten[k] = -qr1d[k] * C.global_inverse_dt;
      nrten[k] = -nr1d[k] * C.global_inverse_dt;
      qr1d[k] = 0.0;
      nr1d[k] = 0.0;
    }
  }
}

// 1012-1082 (optional dt never passed by tempo_main)
TEMPO_HDN inline void ice_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qi, double* qi1d, double* ni1d, double* ri, double* ni,
    double* qiten, double* niten, double* ilami) {
  const double dt_ = C.global_dt;
  const double odt_ = 1.0 / dt_;
  double lami, xdi;
  for (int k = 1; k <= nz; ++k) {
    bool hit_limit = false;
    if (qi1d[k] + qiten[k] * dt_ > PS(r1)) {
      l_qi[k] = true;
      ri[k] = (qi1d[k] + qiten[k] * dt_) * rho[k];
      qi1d[k] = qi1d[k] + qiten[k] * dt_;
      ni[k] = tmax(PS(r2), (ni1d[k] + niten[k] * dt_) * rho[k]);
      if (ni[k] <= PS(r2)) {
        hit_limit = true;
        lami = PA1(cie, 2) / 5.e-6;
        ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tempo_pow(lami, PS(bm_i)));
      }
      lami = tempo_pow((PS(am_i) * PA1(cig, 2) * PS(oig1) * ni[k] / ri[k]), PS(obmi));
      xdi = (PS(bm_i) + PS(mu_i) + 1.0) / lami;
      if (xdi < 5.e-6) {
        hit_limit = true;
        lami = PA1(cie, 2) / 5.e-6;
        ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tempo_pow(lami, PS(bm_i)));
      } else if (xdi > PS(d0s)) {
        hit_limit = true;
        lami = PA1(cie, 2) / PS(d0s);
        ni[k] = PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tempo_pow(lami, PS(bm_i));
      }
      if (hit_limit) niten[k] = (ni[k] / rho[k] - ni1d[k]) * odt_;
      ni1d[k] = tmax(PS(r2) / rho[k],
          tmin(PA1(cig, 1) * PS(oig2) * qi1d[k] / PS(am_i) * tempo_pow(lami, PS(bm_i)), PS(max_ni) / rho[k]));
      ilami[k] = 1.0 / lami;
    } else {
      l_qi[k] = false;
      ri[k] = PS(r1);
      ni[k] = PS(r2);
      ilami[k] = 0.0;
      qiten[k] = -qi1d[k] * odt_;
      niten[k] = -ni1d[k] * odt_;
      qi1d[k] = 0.0;
      ni1d[k] = 0.0;
    }
  }
}

// 1085-1117
TEMPO_HDN inline void snow_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qs, double* qs1d, double* rs, double* qsten) {
  const double dt_ = C.global_dt;
  const double odt_ = 1.0 / dt_;
  for (int k = 1; k <= nz; ++k) {
    if (qs1d[k] + qsten[k] * dt_ > PS(r1)) {
      l_qs[k] = true;
      rs[k] = (qs1d[k] + qsten[k] * dt_) * rho[k];
      qs1d[k] = qs1d[k] + qsten[k] * dt_;
    } else {
      l_qs[k] = false;
      rs[k] = PS(r1);
      qsten[k] = -qs1d[k] * odt_;
      qs1d[k] = 0.0;
    }
  }
}

// 1120-1213
TEMPO_HDN inline void graupel_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qg, double* qg1d, double* ng1d, double* qb1d, double* rg,
    double* ng, double* rb, int* idx, double* qgten, double* ngten, double* qbten,
    double* ilamg, double* mvd_g) {
  const int nrhg = PI(nrhg);
  double lamg = 0.0, ygra1, zans1, n0_exp, lam_exp;
  for (int k = 1; k <= nz; ++k) {
    bool hit_limit = false;
    if (qg1d[k] + qgten[k] * C.global_dt > PS(r1)) {
      l_qg[k] = true;
      rg[k] = (qg1d[k] + qgten[k] * C.global_dt) * rho[k];
      qg1d[k] = qg1d[k] + qgten[k] * C.global_dt;
      if (ng1d != nullptr && qb1d != nullptr) {
        ng[k] = tmax(PS(r2), (ng1d[k] + ngten[k] * C.global_dt) * rho[k]);
        rb[k] = tmin(tmax(rg[k] * PS(meters3_to_liters) / PA1(rho_g, nrhg),
                          (qb1d[k] + qbten[k] * C.global_dt) * rho[k]),
                     rg[k] * PS(meters3_to_liters) / PA1(rho_g, 1));
        idx[k] = imax(1, imin(tnint(10.0 * rg[k] / rb[k]) + 1, nrhg));
        if (ng[k] <= PS(r2)) {
          hit_limit = true;
          mvd_g[k] = 1.5e-3;
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx[k]);
        }
        lamg = tempo_pow((PA1(am_g, idx[k]) * PA2(cgg, 3, 1) * PS(ogg2) * ng[k] / rg[k]), PS(obmg));
        mvd_g[k] = (3.0 + PS(mu_g) + 0.672) / lamg;
        if (mvd_g[k] > 25.4e-3) {
          hit_limit = true;
          mvd_g[k] = 25.4e-3;
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx[k]);
        } else if (mvd_g[k] < PS(d0r)) {
          hit_limit = true;
          mvd_g[k] = PS(d0r);
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx[k]);
        }
        if (hit_limit) ngten[k] = (ng[k] / rho[k] - ng1d[k]) * C.global_inverse_dt;
        ng1d[k] = PA2(cgg, 2, 1) * PS(ogg3) * qg1d[k] * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx[k]);
        qb1d[k] = tmin(tmax(qg1d[k] * PS(meters3_to_liters) / PA1(rho_g, nrhg),
                            qb1d[k] + qbten[k] * C.global_dt),
                       PS(meters3_to_liters) * qg1d[k] / PA1(rho_g, 1));
        idx[k] = imax(1, imin(tnint(10.0 * qg1d[k] / qb1d[k]) + 1, nrhg));
      } else {
        idx[k] = PI(idx_bg1);
        ygra1 = tempo_log10(tmax(1.e-9, rg[k]));
        zans1 = 3.4 + 2.0 / 7.0 * (ygra1 + 8.0);
        n0_exp = tmax(PS(gonv_min), tmin(tempo_pow(10.0, zans1), PS(gonv_max)));
        lam_exp = tempo_pow((n0_exp * PA1(am_g, idx[k]) * PA2(cgg, 1, 1) / rg[k]), PS(oge1));
        lamg = lam_exp * tempo_pow((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), PS(obmg));
        ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx[k]);
        rb[k] = PS(meters3_to_liters) * rg[k] / PA1(rho_g, idx[k]);
      }
      ilamg[k] = 1.0 / lamg;
      mvd_g[k] = (3.0 + PS(mu_g) + 0.672) * ilamg[k];
    } else {
      l_qg[k] = false;
      rg[k] = PS(r1);
      ng[k] = PS(r2);
      mvd_g[k] = PS(d0g);
      ilamg[k] = 0.0;
      idx[k] = PI(idx_bg1);
      rb[k] = PS(meters3_to_liters) * PS(r1) / PA1(rho_g, idx[k]);
      qgten[k] = -qg1d[k] * C.global_inverse_dt;
      qg1d[k] = 0.0;
      if (ng1d != nullptr && qb1d != nullptr) {
        ngten[k] = -ng1d[k] * C.global_inverse_dt;
        qbten[k] = -qb1d[k] * C.global_inverse_dt;
        ng1d[k] = 0.0;
        qb1d[k] = 0.0;
      }
    }
  }
}

// v3.2.5 main.F90:1410-1448 (v3.2.0 #48 hail-hyperaware)
TEMPO_HDN inline void hail_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, bool* l_qh, double* qh1d, double* rh, double* nh, double* qhten,
    double* ilamh) {
  const int nrhg = PI(nrhg);
  double lamh, ygra1, zans1, n0_exp, lam_exp;
  for (int k = 1; k <= nz; ++k) {
    if (qh1d[k] + qhten[k] * C.global_dt > PS(r1)) {
      l_qh[k] = true;
      rh[k] = (qh1d[k] + qhten[k] * C.global_dt) * rho[k];
      ygra1 = tempo_log10(tmax(1.e-9, rh[k]));
      zans1 = 3.0 + 2.0 / 7.0 * (ygra1 + 8.0);
      n0_exp = tmax(PS(gonv_min), tmin(tempo_pow(10.0, zans1), PS(gonv_max)));
      lam_exp = tempo_pow((n0_exp * PA1(am_g, nrhg) * PA2(cgg, 1, 1) / rh[k]), PS(oge1));
      lamh = lam_exp * tempo_pow((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), PS(obmg));
      ilamh[k] = 1.0 / lamh;
      nh[k] = PA2(cgg, 2, 1) * PS(ogg3) * rh[k] * tempo_pow(lamh, PS(bm_g)) / PA1(am_g, nrhg);
      qh1d[k] = qh1d[k] + qhten[k] * C.global_dt;
    } else {
      l_qh[k] = false;
      rh[k] = PS(r1);
      nh[k] = PS(r2);
      ilamh[k] = 0.0;
      qhten[k] = -qh1d[k] * C.global_inverse_dt;
      qh1d[k] = 0.0;
    }
  }
}

// v3.2.5 main.F90:980-1040
TEMPO_HDN inline void compute_hail_fraction(const TempoParams& P, int nz, const bool* l_qg,
    const double* qg1d, const double* mvd_g, const int* idx_bg, const double* qr1d,
    const double* qc1d, const double* temp, const double* tcond, const double* diffu,
    const double* delqvs, double* qh1d, double* hail_fraction) {
  const double t0 = PS(t0);
  double tempc, melt_prefactor, rho_g_val;
  for (int k = 1; k <= nz; ++k) {
    hail_fraction[k] = 0.0;
    qh1d[k] = 0.0;
    tempc = temp[k] - t0;
    melt_prefactor = tempc * tcond[k] - PS(lvap0) * diffu[k] * delqvs[k];
    rho_g_val = PA1(rho_g, idx_bg[k]);
    if (l_qg[k]) {
      if (rho_g_val >= 350.0 && qg1d[k] > 1.e-6) {
        if (mvd_g[k] > 2.e-3 && tempc <= 0.0) hail_fraction[k] = 0.1;
        if (qg1d[k] > 5.e-4) {
          hail_fraction[k] = tmax(tmin(tempo_exp(12.0 * (rho_g_val / 1070.0) - 9.9) + 0.1, 1.0), 0.1);
          if (mvd_g[k] > 2.e-3) hail_fraction[k] = hail_fraction[k] + (mvd_g[k] * 1000.0 - 2.0) * 0.1;
          hail_fraction[k] = tmax(tmin(hail_fraction[k], 0.5), 0.2);
          if (qr1d[k] + qc1d[k] > 1.e-3) {
            if (qg1d[k] > 1.e-3 && rho_g_val > 699.0) hail_fraction[k] = hail_fraction[k] + 0.2;
          } else {
            if (rho_g_val < 699.0 && qg1d[k] > 5.e-3) {
              hail_fraction[k] = tmin(hail_fraction[k], tmin(0.001 / qg1d[k], 1.0) * 0.5);
            }
          }
          qh1d[k] = hail_fraction[k] * qg1d[k];
          if (rho_g_val >= 749.0 && melt_prefactor > PS(eps) && tempc > 0.0) {
            if (mvd_g[k] > 2.e-3) {
              hail_fraction[k] = hail_fraction[k]
                  + (1.0 - tmax(tmin((1.0 / (1.0 + tempo_exp(temp[k] - t0 - 5.0))), 1.0), 0.0));
              hail_fraction[k] = tmin(hail_fraction[k], 0.8 + (mvd_g[k] * 1000.0 - 6.0) * 0.1);
              hail_fraction[k] = tmax(tmin(hail_fraction[k], 0.8), 0.0);
            } else {
              hail_fraction[k] = 0.0;
            }
            qh1d[k] = hail_fraction[k] * qg1d[k];
          }
        }
      }
    }
  }
}

// 1216-1242
TEMPO_HDN inline void graupel_init(const TempoParams& P, int nz, const double* rho,
    const double* qg1d, double* ng1d, double* qb1d) {
  for (int k = 1; k <= nz; ++k) {
    if (qg1d[k] > PS(r1)) {
      double rg = qg1d[k] * rho[k];
      double rb = rg * PS(meters3_to_liters) / PA1(rho_g, PI(idx_bg1));
      int idx = imax(1, imin(tnint(10.0 * rg / rb) + 1, PI(nrhg)));
      double mvd_g = 5.e-3;
      double lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g;
      double ng = PA2(cgg, 2, 1) * PS(ogg3) * rg * tempo_pow(lamg, PS(bm_g)) / PA1(am_g, idx);
      ng1d[k] = ng / rho[k];
      qb1d[k] = rb / rho[k];
    }
  }
}

// 1245-1292
TEMPO_HDN inline void thermo_vars(const TempoParams& P, int nz, const double* qv,
    const double* temp, const double* pres, const double* rho, double* rhof, double* rhof2,
    double* qvs, double* delqvs, double* qvsi, double* satw, double* sati, double* ssatw,
    double* ssati, double* diffu, double* visco, double* vsc2, double* ocp, double* lvap,
    double* tcond, double* lvt2, bool* supersaturated) {
  const double t0 = PS(t0);
  *supersaturated = false;
  for (int k = 1; k <= nz; ++k) {
    double otemp = 1.0 / temp[k];
    double tempc = temp[k] - t0;
    rhof[k] = sqrt(PS(rho_not) / rho[k]);
    rhof2[k] = sqrt(rhof[k]);
    qvs[k] = calc_rslf(pres[k], temp[k]);
    delqvs[k] = tmax(0.0, calc_rslf(pres[k], t0) - qv[k]);
    if (tempc <= 0.0) {
      qvsi[k] = calc_rsif(pres[k], temp[k]);
    } else {
      qvsi[k] = qvs[k];
    }
    satw[k] = qv[k] / qvs[k];
    sati[k] = qv[k] / qvsi[k];
    ssatw[k] = satw[k] - 1.0;
    ssati[k] = sati[k] - 1.0;
    if (fabs(ssatw[k]) < PS(eps)) ssatw[k] = 0.0;
    if (fabs(ssati[k]) < PS(eps)) ssati[k] = 0.0;
    if (ssati[k] > 0.0) *supersaturated = true;
    diffu[k] = 2.11e-5 * tempo_pow((temp[k] / t0), 1.94) * (101325.0 / pres[k]);
    if (tempc >= 0.0) {
      visco[k] = (1.718 + 0.0049 * tempc) * 1.0e-5;
    } else {
      visco[k] = (1.718 + 0.0049 * tempc - 1.2e-5 * tempc * tempc) * 1.0e-5;
    }
    ocp[k] = 1.0 / (PS(cp) * (1.0 + 0.887 * qv[k]));
    vsc2[k] = sqrt(rho[k] / visco[k]);
    lvap[k] = PS(lvap0) + (2106.0 - 4218.0) * tempc;
    tcond[k] = (5.69 + 0.0168 * tempc) * 1.0e-5 * 418.936;
    lvt2[k] = lvap[k] * lvap[k] * ocp[k] * PS(orv) * otemp * otemp;
  }
}

// 1295-1419
TEMPO_HDN inline void check_over_depletion(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rho, const double* temp, const double* qvsi, const double* qv,
    const bool* l_qc, const double* rc, const bool* l_qi, const double* ri, const bool* l_qr,
    const double* rr, const bool* l_qs, const double* rs, const bool* l_qg, const double* rg,
    const bool* l_qh, const double* rh, TempoTend& T) {
  double sump, rate_max, ratio;
  const double eps = PS(eps);
  for (int k = 1; k <= nz; ++k) {
    sump = T.pri_inu[k] + T.pri_ide[k] + T.prs_ide[k] + T.prs_sde[k] + T.prg_gde[k] + T.pri_iha[k];
    rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999;
    if ((sump > eps && sump > rate_max) || (sump < -eps && sump < rate_max)) {
      ratio = rate_max / sump;
      T.pri_inu[k] = T.pri_inu[k] * ratio;
      T.pri_ide[k] = T.pri_ide[k] * ratio;
      T.pni_ide[k] = T.pni_ide[k] * ratio;
      T.prs_ide[k] = T.prs_ide[k] * ratio;
      T.prs_sde[k] = T.prs_sde[k] * ratio;
      T.prg_gde[k] = T.prg_gde[k] * ratio;
      T.pri_iha[k] = T.pri_iha[k] * ratio;
    }

    sump = -T.prr_wau[k] - T.pri_wfz[k] - T.prr_rcw[k] - T.prs_scw[k] - T.prg_scw[k] - T.prg_gcw[k]
         - T.prh_hcw[k];
    rate_max = -rc[k] * C.global_inverse_dt;
    if (l_qc[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.prr_wau[k] = T.prr_wau[k] * ratio;
        T.pri_wfz[k] = T.pri_wfz[k] * ratio;
        T.prr_rcw[k] = T.prr_rcw[k] * ratio;
        T.prs_scw[k] = T.prs_scw[k] * ratio;
        T.prg_scw[k] = T.prg_scw[k] * ratio;
        T.prg_gcw[k] = T.prg_gcw[k] * ratio;
        T.prh_hcw[k] = T.prh_hcw[k] * ratio;
      }
    }

    sump = T.pri_ide[k] - T.prs_iau[k] - T.prs_sci[k] - T.pri_rci[k];
    rate_max = -ri[k] * C.global_inverse_dt;
    if (l_qi[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.pri_ide[k] = T.pri_ide[k] * ratio;
        T.prs_iau[k] = T.prs_iau[k] * ratio;
        T.prs_sci[k] = T.prs_sci[k] * ratio;
        T.pri_rci[k] = T.pri_rci[k] * ratio;
      }
    }

    sump = -T.prg_rfz[k] - T.pri_rfz[k] - T.prr_rci[k] + T.prr_rcs[k] + T.prr_rcg[k] + T.prr_rch[k];
    rate_max = -rr[k] * C.global_inverse_dt;
    if (l_qr[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.prg_rfz[k] = T.prg_rfz[k] * ratio;
        T.pbg_rfz[k] = T.pbg_rfz[k] * ratio;
        T.pri_rfz[k] = T.pri_rfz[k] * ratio;
        T.prr_rci[k] = T.prr_rci[k] * ratio;
        T.prr_rcs[k] = T.prr_rcs[k] * ratio;
        T.prr_rcg[k] = T.prr_rcg[k] * ratio;
        T.prr_rch[k] = T.prr_rch[k] * ratio;
      }
    }

    sump = T.prs_sde[k] - T.prs_ihm[k] - T.prr_sml[k] + T.prs_rcs[k];
    rate_max = -rs[k] * C.global_inverse_dt;
    if (l_qs[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.prs_sde[k] = T.prs_sde[k] * ratio;
        T.prs_ihm[k] = T.prs_ihm[k] * ratio;
        T.prr_sml[k] = T.prr_sml[k] * ratio;
        T.prs_rcs[k] = T.prs_rcs[k] * ratio;
      }
    }

    sump = T.prg_gde[k] - T.prg_ihm[k] - T.prr_gml[k] + T.prg_rcg[k];
    rate_max = -rg[k] * C.global_inverse_dt;
    if (l_qg[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.prg_gde[k] = T.prg_gde[k] * ratio;
        T.prg_ihm[k] = T.prg_ihm[k] * ratio;
        T.prr_gml[k] = T.prr_gml[k] * ratio;
        T.prg_rcg[k] = T.prg_rcg[k] * ratio;
        T.pbg_rcg[k] = T.pbg_rcg[k] * ratio;
      }
    }

    sump = T.prh_hde[k] - T.prr_hml[k];
    rate_max = -rh[k] * C.global_inverse_dt;
    if (l_qh[k]) {
      if (sump < rate_max) {
        ratio = rate_max / sump;
        T.prr_hml[k] = T.prr_hml[k] * ratio;
        T.prh_hde[k] = T.prh_hde[k] * ratio;
      }
    }

    T.pri_ihm[k] = T.prs_ihm[k] + T.prg_ihm[k];
    ratio = tmin(fabs(T.prr_rcg[k]), fabs(T.prg_rcg[k]));
    T.prr_rcg[k] = ratio * tsign(1.0, T.prr_rcg[k]);
    T.prg_rcg[k] = -T.prr_rcg[k];
    T.pbg_rcg[k] = PS(meters3_to_liters) * T.prg_rcg[k] / PA1(rho_g, PI(nrhg));
    if (temp[k] >= PS(t0)) {
      ratio = tmin(fabs(T.prr_rcs[k]), fabs(T.prs_rcs[k]));
      T.prr_rcs[k] = ratio * tsign(1.0, T.prr_rcs[k]);
      T.prs_rcs[k] = -T.prr_rcs[k];
    }
    ratio = tmin(fabs(T.prr_rch[k]), fabs(T.prh_rch[k]));
    T.prr_rch[k] = ratio * tsign(1.0, T.prr_rch[k]);
    T.prh_rch[k] = ratio * tsign(1.0, T.prh_rch[k]);
  }
}

// 1422-1494
TEMPO_HDN inline void sum_tendencies(const TempoParams& P, int nz, const double* rho,
    const double* temp, const int* idx, const double* lvap, const double* ocp,
    const TempoTend& T, double* tten, double* qvten, double* qcten, double* ncten,
    double* qiten, double* niten, double* qsten, double* qrten, double* nrten, double* qgten,
    double* ngten, double* qbten, double* qhten) {
  double massg_sources, volg_sources, dens_weight, final_weight, frozen_terms, riming_terms;
  for (int k = 1; k <= nz; ++k) {
    double orho = 1. / rho[k];
    double lfus2 = PS(lsub) - lvap[k];
    qvten[k] = qvten[k] + (-T.pri_inu[k] - T.pri_iha[k] - T.pri_ide[k] - T.prs_ide[k]
        - T.prs_sde[k] - T.prg_gde[k] - T.prh_hde[k]) * orho;
    qcten[k] = qcten[k] + (-T.prr_wau[k] - T.pri_wfz[k] - T.prr_rcw[k] - T.prs_scw[k]
        - T.prg_scw[k] - T.prg_gcw[k] - T.prh_hcw[k]) * orho;
    ncten[k] = ncten[k] + (-T.pnc_wau[k] - T.pnc_rcw[k] - T.pni_wfz[k] - T.pnc_scw[k]
        - T.pnc_gcw[k] - T.pnc_hcw[k]) * orho;
    qiten[k] = qiten[k] + (T.pri_inu[k] + T.pri_iha[k] + T.pri_ihm[k] + T.pri_wfz[k]
        + T.pri_rfz[k] + T.pri_ide[k] - T.prs_iau[k] - T.prs_sci[k] - T.pri_rci[k]) * orho;
    niten[k] = niten[k] + (T.pni_inu[k] + T.pni_iha[k] + T.pni_ihm[k] + T.pni_wfz[k]
        + T.pni_rfz[k] + T.pni_ide[k] - T.pni_iau[k] - T.pni_sci[k] - T.pni_rci[k]) * orho;
    qrten[k] = qrten[k] + (T.prr_wau[k] + T.prr_rcw[k] + T.prr_sml[k] + T.prr_gml[k]
        + T.prr_rcs[k] + T.prr_rcg[k] + T.prr_rch[k] - T.prg_rfz[k] - T.pri_rfz[k]
        - T.prr_rci[k] + T.prr_hml[k]) * orho;
    nrten[k] = nrten[k] + (T.pnr_wau[k] + T.pnr_sml[k] + T.pnr_gml[k] + T.pnr_hml[k]
        - (T.pnr_rfz[k] + T.pnr_rcr[k] + T.pnr_rcg[k] + T.pnr_rch[k] + T.pnr_rcs[k]
        + T.pnr_rci[k] + T.pni_rfz[k])) * orho;
    qsten[k] = qsten[k] + (T.prs_iau[k] + T.prs_sde[k] + T.prs_sci[k] + T.prs_scw[k]
        + T.prs_rcs[k] + T.prs_ide[k] - T.prs_ihm[k] - T.prr_sml[k]) * orho;
    qgten[k] = qgten[k] + (T.prg_scw[k] + T.prg_rfz[k] + T.prg_gde[k] + T.prg_rcg[k]
        + T.prg_gcw[k] + T.prg_rci[k] + T.prg_rcs[k] - T.prg_ihm[k] - T.prr_gml[k]) * orho;
    ngten[k] = ngten[k] + (T.png_scw[k] + T.png_rfz[k] - T.png_rcg[k] + T.png_rci[k]
        + T.png_rcs[k] + T.png_gde[k] - T.pnr_gml[k]) * orho;
    qbten[k] = qbten[k] + (T.pbg_scw[k] + T.pbg_rfz[k] + T.pbg_gcw[k] + T.pbg_rci[k]
        + T.pbg_rcs[k] + T.pbg_rcg[k] + T.pbg_sml[k] - T.pbg_gml[k]
        + PS(meters3_to_liters) * (T.prg_gde[k] - T.prg_ihm[k]) / PA1(rho_g, 1)) * orho;
    // v3.2.4 #53: raise the volume tendency density when frozen drops
    // dominate graupel growth.
    if (temp[k] < PS(t0)) {
      frozen_terms = T.prg_rfz[k] + T.prg_rcg[k] + T.prg_rci[k] + T.prg_rcs[k];
      riming_terms = T.prg_scw[k] + T.prg_gcw[k];
      massg_sources = frozen_terms + riming_terms;
      volg_sources = T.pbg_scw[k] + T.pbg_gcw[k] + T.pbg_rfz[k] + T.pbg_rcg[k] + T.pbg_rci[k]
                   + T.pbg_rcs[k];
      if (frozen_terms > PS(r1) && frozen_terms > riming_terms) {
        // Upstream defect: v3.2.5 reads final_weight unset when the test
        // below fails (main.F90:1752-1755).  Defined here as 1, no scaling.
        final_weight = 1.0;
        dens_weight = PA1(rho_g, PI(nrhg)) * frozen_terms / massg_sources;
        if (T.pbg_scw[k] > PS(eps))
          dens_weight = dens_weight + T.prg_scw[k] / T.pbg_scw[k] * (T.prg_scw[k] / massg_sources);
        if (T.pbg_gcw[k] > PS(eps))
          dens_weight = dens_weight + T.prg_gcw[k] / T.pbg_gcw[k] * (T.prg_gcw[k] / massg_sources);
        if (dens_weight > PS(r1) && volg_sources > PS(r1)) {
          final_weight = (1.0 / dens_weight) * (massg_sources / volg_sources);
        }
        if (final_weight <= 1.0) qbten[k] = qbten[k] * final_weight;
      }
    }
    qhten[k] = qhten[k] + (T.prh_rch[k] + T.prh_hcw[k] + T.prh_hde[k] - T.prr_hml[k]) * orho;
    // v3.2.0 #48 / v3.2.3 #51: freezing heat counts the rain frozen in
    // rain-snow and rain-graupel collection.
    if (temp[k] < PS(t0)) {
      tten[k] = tten[k]
          + (PS(lsub) * ocp[k] * (T.pri_inu[k] + T.pri_ide[k] + T.prs_ide[k] + T.prs_sde[k]
             + T.prg_gde[k] + T.prh_hde[k] + T.pri_iha[k])
             + lfus2 * ocp[k] * (T.pri_wfz[k] + T.pri_rfz[k] + T.prg_rfz[k] + T.prs_scw[k]
             + T.prg_scw[k] + T.prg_gcw[k] + T.prh_hcw[k] - T.prr_rcs[k] + T.prr_rci[k]
             - T.prr_rcg[k] - T.prr_rch[k])) * orho;
    } else {
      tten[k] = tten[k]
          + (PS(lfus) * ocp[k] * (-T.prr_sml[k] - T.prr_gml[k] - T.prr_hml[k] - T.prr_rcg[k]
             - T.prr_rcs[k])
             + PS(lsub) * ocp[k] * (T.prs_sde[k] + T.prg_gde[k] + T.prh_hde[k])) * orho;
    }
  }
}

// 1497-1542.  precip may be nullptr.  ktop_sedi always passed by tempo_main.
TEMPO_HDN inline void sedimentation(const TempoParams& P, const TempoCtx& C, int nz,
    double* xr, const double* vt, const double* dz1d, const double* rho, double* xten,
    double limit, int steps, int ktop_sedi, double* precip, double* sed_r) {
  double odz, orho;
  int ktop = ktop_sedi;
  for (int k = nz; k >= 1; --k) sed_r[k] = vt[k] * xr[k];
  int k = nz;
  odz = 1.0 / dz1d[k];
  orho = 1.0 / rho[k];
  xten[k] = xten[k] - sed_r[k] * odz * (1.0 / (double)steps) * orho;
  xr[k] = tmax(limit, xr[k] - sed_r[k] * odz * C.global_dt * (1.0 / (double)steps));
  for (k = ktop; k >= 1; --k) {
    odz = 1.0 / dz1d[k];
    orho = 1.0 / rho[k];
    xten[k] = xten[k] + (sed_r[k + 1] - sed_r[k]) * odz * (1.0 / (double)steps) * orho;
    xr[k] = tmax(limit, xr[k] + (sed_r[k + 1] - sed_r[k]) * odz * C.global_dt * (1.0 / (double)steps));
  }
  if (precip != nullptr) {
    if (xr[1] > PS(low_limit_mass_for_precip)) {
      *precip = *precip + sed_r[1] * C.global_dt * (1.0 / (double)steps);
    }
  }
}

// 1545-1731 semi-Lagrangian sedimentation (Juang and Hong 2010).  W supplies scratch.
TEMPO_HDN inline void semilagrangian_sedimentation(const TempoParams& P, const TempoCtx& C,
    int nz, TempoWork& W, const double* dz1d, const double* rho, double* xr, double* xten,
    const double* vt, int steps, double limit, double* precip) {
  double fa1, fa2, con1, decfl, dip, dim, tl, th, tl2, th2, qqd, qqh, qql, zsum, qsum, orho, dql, dqh;
  double* zi = W.sl_zi;
  double* wi = W.sl_wi;
  double* za = W.sl_za;
  double* dza = W.sl_dza;
  double* qa = W.sl_qa;
  double* qmi = W.sl_qmi;
  double* qpi = W.sl_qpi;
  double* net_flx = W.sl_net_flx;
  double* precip_flx = W.sl_precip_flx;
  double* rr_save = W.sl_rr_save;
  int k, kk, kb, kt, m;
  for (k = 1; k <= nz + 2; ++k) {
    zi[k] = 0.0; wi[k] = 0.0; dza[k] = 0.0; za[k] = 0.0; qa[k] = 0.0; qmi[k] = 0.0; qpi[k] = 0.0;
  }
  for (k = 1; k <= nz; ++k) {
    net_flx[k] = 0.0;
    precip_flx[k] = 0.0;
    rr_save[k] = xr[k];
  }
  zi[1] = 0.0;
  for (k = 1; k <= nz; ++k) zi[k + 1] = zi[k] + dz1d[k];
  fa1 = 9.0 / 16.0;
  fa2 = 1.0 / 16.0;
  wi[1] = vt[1];
  wi[2] = 0.5 * (vt[2] + vt[1]);
  for (k = 3; k <= nz - 1; ++k) wi[k] = fa1 * (vt[k] + vt[k - 1]) - fa2 * (vt[k + 1] + vt[k - 2]);
  wi[nz] = 0.5 * (vt[nz] + vt[nz - 1]);
  wi[nz + 1] = vt[nz + 1];
  for (k = 2; k <= nz; ++k)
    if (vt[k] == 0.0) wi[k] = vt[k - 1];
  con1 = 0.05;
  for (k = nz; k >= 1; --k) {
    decfl = (wi[k + 1] - wi[k]) * C.global_dt * (1.0 / (double)steps) / dz1d[k];
    if (decfl > con1) wi[k] = wi[k + 1] - con1 * dz1d[k] * C.global_inverse_dt * (double)steps;
  }
  for (k = 1; k <= nz + 1; ++k) za[k] = zi[k] - wi[k] * C.global_dt * (1.0 / (double)steps);
  za[nz + 2] = zi[nz + 1];
  for (k = 1; k <= nz + 1; ++k) dza[k] = za[k + 1] - za[k];
  for (k = 1; k <= nz; ++k) qa[k] = xr[k] * dz1d[k] / dza[k];
  qa[nz + 1] = 0.0;
  for (k = 2; k <= nz; ++k) {
    dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
    dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
    if (dip * dim <= 0.0) {
      qmi[k] = qa[k];
      qpi[k] = qa[k];
    } else {
      qpi[k] = qa[k] + 0.5 * (dip + dim) * dza[k];
      qmi[k] = 2.0 * qa[k] - qpi[k];
      if (qpi[k] > 0.0 || qmi[k] < 0.0) {
        qpi[k] = qa[k];
        qmi[k] = qa[k];
      }
    }
  }
  qpi[1] = qa[1];
  qmi[1] = qa[1];
  qmi[nz + 1] = qa[nz + 1];
  qpi[nz + 1] = qa[nz + 1];

  kb = 1;
  kt = 1;
  for (k = 1; k <= nz; ++k) {
    kb = imax(kb - 1, 1);
    kt = imax(kt - 1, 1);
    if (zi[k] >= za[nz + 1]) {
      break;
    } else {
      for (kk = kb; kk <= nz; ++kk) {
        if (zi[k] <= za[kk + 1]) {
          kb = kk;
          break;
        }
      }
      for (kk = kt; kk <= nz + 2; ++kk) {
        if (zi[k + 1] <= za[kk]) {
          kt = kk;
          break;
        }
      }
      kt = kt - 1;
      if (kt == kb) {
        tl = (zi[k] - za[kb]) / dza[kb];
        th = (zi[k + 1] - za[kb]) / dza[kb];
        tl2 = tl * tl;
        th2 = th * th;
        qqd = 0.5 * (qpi[kb] - qmi[kb]);
        qqh = qqd * th2 + qmi[kb] * th;
        qql = qqd * tl2 + qmi[kb] * tl;
        xr[k] = (qqh - qql) / (th - tl);
      } else if (kt > kb) {
        tl = (zi[k] - za[kb]) / dza[kb];
        tl2 = tl * tl;
        qqd = 0.5 * (qpi[kb] - qmi[kb]);
        qql = qqd * tl2 + qmi[kb] * tl;
        dql = qa[kb] - qql;
        zsum = (1.0 - tl) * dza[kb];
        qsum = dql * dza[kb];
        if (kt - kb > 1) {
          for (m = kb + 1; m <= kt - 1; ++m) {
            zsum = zsum + dza[m];
            qsum = qsum + qa[m] * dza[m];
          }
        }
        th = (zi[k + 1] - za[kt]) / dza[kt];
        th2 = th * th;
        qqd = 0.5 * (qpi[kt] - qmi[kt]);
        dqh = qqd * th2 + qmi[kt] * th;
        zsum = zsum + th * dza[kt];
        qsum = qsum + dqh * dza[kt];
        xr[k] = qsum / zsum;
      }
    }
    orho = 1.0 / rho[k];
    xr[k] = tmax(xr[k], limit);
    xten[k] = xten[k] + (xr[k] - rr_save[k]) * orho * C.global_inverse_dt;
  }

  for (k = 1; k <= nz; ++k) {
    if (za[k] < 0.0 && za[k + 1] <= 0.0) {
      if (precip != nullptr) *precip = *precip + qa[k] * dza[k];
      net_flx[k] = qa[k] * dza[k];
    } else if (za[k] < 0.0 && za[k + 1] > 0.0) {
      th = (0.0 - za[k]) / dza[k];
      th2 = th * th;
      qqd = 0.5 * (qpi[k] - qmi[k]);
      qqh = qqd * th2 + qmi[k] * th;
      if (precip != nullptr) *precip = *precip + qqh * dza[k];
      net_flx[k] = qqh * dza[k];
      break;
    }
  }
  for (k = nz; k >= 1; --k) {
    if (k == nz) precip_flx[k] = net_flx[k];
    else precip_flx[k] = precip_flx[k + 1] + net_flx[k];
  }
}

// 1734-1771
TEMPO_HDN inline void rain_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const bool* l_qr, const double* rr, const double* ilamr,
    const double* dz1d, double* vt, double* vtn, int* substeps_sedi, int* ktop_sedi) {
  double dz_by_vt, lamr;
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rr[k] > PS(r1)) {
      lamr = 1.0 / ilamr[k];
      vt[k] = rhof[k] * PS(av_r) * PA1(crg, 6) * PS(org3) * tempo_pow(lamr, PA1(cre, 3))
            * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 6))));
      vtn[k] = rhof[k] * PS(av_r) * PA1(crg, 7) / PA1(crg, 12) * tempo_pow(lamr, PA1(cre, 12))
             * (tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 7))));
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (tmax(vt[k], vtn[k]) > 1.e-3) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / (tmax(vt[k], vtn[k]));
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// v3.2.5 main.F90:2056-2093
TEMPO_HDN inline void hail_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const double* rh, const double* ilamh, const double* dz1d, double* vt,
    int* substeps_sedi, int* ktop_sedi) {
  double dz_by_vt, vth;
  const int nrhg = PI(nrhg);
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rh[k] > PS(r1)) {
      vth = rhof[k] * PA1(av_g, nrhg) * PA2(cgg, 6, nrhg) * PS(ogg3) * tempo_pow(ilamh[k], PA1(bv_g, nrhg));
      vt[k] = vth;
    } else {
      vt[k] = vt[k + 1];
    }
    if (vt[k] > 1.e-3) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1774-1828.  has_qb1d = present(qb1d).
TEMPO_HDN inline void graupel_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const double* rho, const double* visco, const bool* l_qg,
    const double* rg, const double* rb, bool has_qb1d, const int* idx, const double* ilamg,
    const double* dz1d, double* vt, double* vtn, int* substeps_sedi, int* ktop_sedi) {
  double dz_by_vt, dens_g, afall, bfall;
  const int nrhg = PI(nrhg);
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rg[k] > PS(r1)) {
      if (has_qb1d) {
        dens_g = tmax(PA1(rho_g, 1), tmin(PS(meters3_to_liters) * rg[k] / rb[k], PA1(rho_g, nrhg)));
        // v3.2.0 #48: constant reference density (earth_gravity = 9.8)
        afall = tempo_pow((4.0 * dens_g * 9.8 / (3.0 * 0.504843467198652 * PS(rho_not))), PS(b_coeff));
        bfall = 3.0 * PS(b_coeff) - 1.0;
      } else {
        afall = PS(av_g_old);
        bfall = PS(bv_g_old);
      }
      vt[k] = rhof[k] * afall * PA2(cgg, 6, idx[k]) * PS(ogg3) * tempo_pow(ilamg[k], bfall);
      if (PS(mu_g) == 0) {
        vtn[k] = rhof[k] * afall * PA2(cgg, 7, idx[k]) / PA2(cgg, 12, idx[k]) * tempo_pow(ilamg[k], bfall);
      } else {
        vtn[k] = rhof[k] * afall * PA2(cgg, 8, idx[k]) * PS(ogg2) * tempo_pow(ilamg[k], bfall);
      }
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (vt[k] > 1.e-3) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1831-1885
TEMPO_HDN inline void snow_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const bool* l_qs, const double* rs, const double* prr_sml,
    const double* smob, const double* smoc, const double* rr, const double* vtrr,
    const double* dz1d, double* vt, const double* vtboost, int* substeps_sedi, int* ktop_sedi) {
  double dz_by_vt, vts, sr, xds, mrat, ils1, ils2, t1_vts, t2_vts, t3_vts, t4_vts;
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rs[k] > PS(r1)) {
      xds = smoc[k] / smob[k];
      mrat = 1.0 / xds;
      ils1 = 1.0 / (mrat * PS(lam0) + PS(fv_s));
      ils2 = 1.0 / (mrat * PS(lam1) + PS(fv_s));
      t1_vts = PS(kap0) * PA1(csg, 4) * tempo_pow(ils1, PA1(cse, 4));
      t2_vts = PS(kap1) * tempo_pow(mrat, PS(mu_s)) * PA1(csg, 10) * tempo_pow(ils2, PA1(cse, 10));
      ils1 = 1.0 / (mrat * PS(lam0));
      ils2 = 1.0 / (mrat * PS(lam1));
      t3_vts = PS(kap0) * PA1(csg, 1) * tempo_pow(ils1, PA1(cse, 1));
      t4_vts = PS(kap1) * tempo_pow(mrat, PS(mu_s)) * PA1(csg, 7) * tempo_pow(ils2, PA1(cse, 7));
      vts = rhof[k] * PS(av_s) * (t1_vts + t2_vts) / (t3_vts + t4_vts);
      if (prr_sml[k] > 0.0) {
        sr = rs[k] / (rs[k] + rr[k]);
        vt[k] = vts * sr + (1.0 - sr) * vtrr[k];
      } else {
        vt[k] = vts * vtboost[k];
      }
    } else {
      vt[k] = vt[k + 1];
    }
    if (vt[k] > 1.e-3) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1888-1920
TEMPO_HDN inline void ice_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const bool* l_qi, const double* ri, const double* ilami,
    const double* dz1d, double* vt, double* vtn, int* substeps_sedi, int* ktop_sedi) {
  double dz_by_vt;
  *ktop_sedi = 1;
  *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (ri[k] > PS(r1)) {
      vt[k] = rhof[k] * PS(av_i) * PA1(cig, 3) * PS(oig2) * tempo_pow(ilami[k], PS(bv_i));
      vtn[k] = rhof[k] * PS(av_i) * PA1(cig, 6) / PA1(cig, 7) * tempo_pow(ilami[k], PS(bv_i));
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (vt[k] > 1.e-3) {
      *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0));
    }
  }
  if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
}

// 1923-1955
TEMPO_HDN inline void cloud_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const double* rhof, const double* w1d, const bool* l_qc, const double* rc,
    const double* nc, const double* ilamc, const double* dz1d, double* vt, double* vtn,
    int* ktop_sedi) {
  double hgt;
  int nu_c;
  *ktop_sedi = 1;
  hgt = 0.0;
  for (int k = 1; k <= nz - 1; ++k) {
    if (rc[k] > PS(r2)) *ktop_sedi = k;
    hgt = hgt + dz1d[k];
    if (hgt > 500.0) break;
  }
  for (int k = *ktop_sedi; k >= 1; --k) {
    if (rc[k] > PS(r1) && w1d[k] < 0.1) {
      nu_c = get_nuc(P, nc[k]);
      vt[k] = rhof[k] * PS(av_c) * PA2(ccg, 5, nu_c) * PA1(ocg2, nu_c) * tempo_pow(ilamc[k], PS(bv_c));
      vtn[k] = rhof[k] * PS(av_c) * PA2(ccg, 4, nu_c) * PA1(ocg1, nu_c) * tempo_pow(ilamc[k], PS(bv_c));
    }
  }
}

TEMPO_HD void tempo_copy(double* dst, const double* src, int n) {
  for (int k = 1; k <= n; ++k) dst[k] = src[k];
}

TEMPO_HD void tempo_fill(double* dst, double v, int n) {
  for (int k = 1; k <= n; ++k) dst[k] = v;
}

// ---------------------------------------------------------------------------
// tempo_main, v3.2.5 main.F90 lines 75-977.  Optional Fortran arguments are
// nullable pointers (nc1d, nwfa1d, nifa1d, ng1d, qb1d).  The land1d, ML
// cloud-number, boundary-layer, radiation and cloud-fraction optional arrays
// are not passed by the MPAS/WOOF drivers and are not ported (the qc_bl1d
// effective-radius block is the separate tempo_subgrid_cloud_radii).
// C.first_call_main is tempo_run's is_first_step (itimestep == 1), applied
// to every column (v3.1.1 #38; v3.0.3's SAVE flag seeded the first column).
// ---------------------------------------------------------------------------
TEMPO_HDN inline void tempo_main_column(const TempoParams& P, const TempoCtx& C, int nz,
    TempoWork& W, double* qv1d, double* qc1d, double* qi1d, double* qr1d, double* qs1d,
    double* qg1d, double* qb1d, double* ni1d, double* nr1d, double* nc1d, double* ng1d,
    double* nwfa1d, double* nifa1d, double* t1d, const double* p1d, const double* w1d,
    const double* dz1d) {
  TempoTend& T = W.T;
  const int nz1 = nz + 1;
  const int nrhg = PI(nrhg);
  int substeps_sedi, ktop_sedi, n;
  double semi_sedi_factor, tempc, tc0;
  bool do_micro = false, supersaturated = false;
  const double t0 = PS(t0);
  const bool local_first_step = C.first_call_main != 0;
  W.did_micro = 0;

#define TEMPO_TT_ZERO(name) for (int k = 0; k < TEMPO_NZP; ++k) T.name[k] = 0.0;
  TEMPO_TEND_FIELDS(TEMPO_TT_ZERO)
#undef TEMPO_TT_ZERO

  for (int k = 1; k <= nz; ++k) {
    W.tten[k] = 0.0; W.qvten[k] = 0.0; W.qcten[k] = 0.0; W.qiten[k] = 0.0; W.qrten[k] = 0.0;
    W.qsten[k] = 0.0; W.qgten[k] = 0.0; W.qhten[k] = 0.0; W.ngten[k] = 0.0; W.qbten[k] = 0.0;
    W.niten[k] = 0.0; W.nrten[k] = 0.0; W.ncten[k] = 0.0; W.nwfaten[k] = 0.0; W.nifaten[k] = 0.0;
    W.smo0[k] = 0.0; W.smo1[k] = 0.0; W.smo2[k] = 0.0; W.smob[k] = 0.0; W.smoc[k] = 0.0;
    W.smoe[k] = 0.0; W.smof[k] = 0.0; W.smog[k] = 0.0; W.smoz[k] = 0.0; W.ns[k] = 0.0;
    W.vtboost[k] = 1.0;
  }
  for (int k = 1; k <= nz1; ++k) {
    W.vtrr[k] = 0.0; W.vtnr[k] = 0.0; W.vtrs[k] = 0.0; W.vtri[k] = 0.0; W.vtni[k] = 0.0;
    W.vtrc[k] = 0.0; W.vtnc[k] = 0.0; W.vtrg[k] = 0.0; W.vtng[k] = 0.0; W.vtrh[k] = 0.0;
  }
  W.rain_precip = 0.0;
  W.cloud_precip = 0.0;
  W.ice_liquid_equiv_precip = 0.0;
  W.snow_liquid_equiv_precip = 0.0;
  W.graupel_liquid_equiv_precip = 0.0;
  W.hail_liquid_equiv_precip = 0.0;
  W.frz_rain_precip = 0.0;
  W.frozen_fraction = 0.0;
  for (int k = 1; k <= nz; ++k) {
    W.refl10cm[k] = -35.0; W.re_cloud[k] = 0.0; W.re_ice[k] = 0.0; W.re_snow[k] = 0.0;
    W.max_hail_diameter[k] = 0.0; W.rain_med_vol_diam[k] = 0.0; W.graupel_med_vol_diam[k] = 0.0;
    W.cloud_number_mixing_ratio[k] = 0.0;
  }

  // initialization
  for (int k = 1; k <= nz; ++k) {
    W.temp[k] = t1d[k];
    W.qv[k] = tmax(PS(min_qv), qv1d[k]);
    W.pres[k] = p1d[k];
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
  }

  if (nwfa1d != nullptr) {
    if (local_first_step) {
      double s = 0.0;
      for (int k = 1; k <= nz; ++k) s += nwfa1d[k];
      if (s < PS(eps)) init_water_friendly_aerosols(P, nz, dz1d, W.nwfa);
    }
  } else {
    init_water_friendly_aerosols(P, nz, dz1d, W.nwfa);
  }
  if (nifa1d != nullptr) {
    if (local_first_step) {
      double s = 0.0;
      for (int k = 1; k <= nz; ++k) s += nifa1d[k];
      if (s < PS(eps)) init_ice_friendly_aerosols(P, nz, dz1d, W.nifa);
    }
  } else {
    init_ice_friendly_aerosols(P, nz, dz1d, W.nifa);
  }

  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, qr1d, nr1d, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, qi1d, ni1d, W.ri, W.ni, W.qiten, W.niten, W.ilami);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, qs1d, W.rs, W.qsten);
  for (int k = 1; k <= nz; ++k) {
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], &W.ns[k], &W.smo0[k], &W.smo1[k],
                   &W.smo2[k], &W.smoe[k], &W.smof[k], &W.smog[k], nullptr);
    }
  }

  // one-moment cloud number: single_moment_nc_opt = 'land' (land1d absent,
  // ml_for_nc off), main.F90:375-391
  const double* ncsave = nullptr;
  if (nc1d == nullptr) {
    for (int k = 1; k <= nz; ++k) W.nc[k] = PS(nt_c_l);
    tempo_copy(W.ncsave, W.nc, nz);
    ncsave = W.ncsave;
  }

  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);

  if (local_first_step) {
    if (ng1d != nullptr && qb1d != nullptr) {
      double sq = 0.0, sn = 0.0, sb = 0.0;
      for (int k = 1; k <= nz; ++k) { sq += qg1d[k]; sn += ng1d[k]; sb += qb1d[k]; }
      if (sq > PS(r1) && sn < PS(eps) && sb < PS(eps)) graupel_init(P, nz, W.rho, qg1d, ng1d, qb1d);
    }
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);

#define TEMPO_THERMO()                                                                      \
  thermo_vars(P, nz, W.qv, W.temp, W.pres, W.rho, W.rhof, W.rhof2, W.qvs, W.delqvs, W.qvsi, \
              W.satw, W.sati, W.ssatw, W.ssati, W.diffu, W.visco, W.vsc2, W.ocp, W.lvap,    \
              W.tcond, W.lvt2, &supersaturated)
  TEMPO_THERMO();

  // hail split from graupel (v3.2.0 #48), main.F90:418-445
  for (int k = 1; k <= nz; ++k) {
    W.qh1d[k] = 0.0;
    W.hail_fraction[k] = 0.0;
    W.l_qh[k] = false;
  }
  if (C.hailhyperaware_flag) {
    compute_hail_fraction(P, nz, W.l_qg, qg1d, W.mvd_g, W.idx_bg, qr1d, qc1d, W.temp, W.tcond,
                          W.diffu, W.delqvs, W.qh1d, W.hail_fraction);
  }
  hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.qh1d, W.rh, W.nh, W.qhten, W.ilamh);
  bool any_qh = false;
  for (int k = 1; k <= nz; ++k) {
    if (W.l_qh[k]) {
      any_qh = true;
      qg1d[k] = qg1d[k] - W.qh1d[k];
      // hyperaware is forced off without hail-aware (tempo_init), so ng1d
      // and qb1d are present whenever l_qh is set
      ng1d[k] = ng1d[k] - W.nh[k] / W.rho[k];
      qb1d[k] = qb1d[k] - W.qh1d[k] * PS(meters3_to_liters) / PA1(rho_g, nrhg);
      W.qgten[k] = 0.0;
      W.ngten[k] = 0.0;
      W.qbten[k] = 0.0;
    }
  }
  if (any_qh) {
    graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                             W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  }

  for (int k = 1; k <= nz; ++k) {
    W.qcten[k] = 0.0; W.qiten[k] = 0.0; W.qrten[k] = 0.0; W.qsten[k] = 0.0; W.qgten[k] = 0.0;
    W.qhten[k] = 0.0; W.ngten[k] = 0.0; W.qbten[k] = 0.0; W.niten[k] = 0.0; W.nrten[k] = 0.0;
    W.ncten[k] = 0.0;
  }

  do_micro = tempo_any(W.l_qc, nz) || tempo_any(W.l_qr, nz) || tempo_any(W.l_qi, nz) ||
             tempo_any(W.l_qs, nz) || tempo_any(W.l_qg, nz) || tempo_any(W.l_qh, nz);
  if (!do_micro && !supersaturated) return;
  W.did_micro = 1;

  if (!C.turn_off_micro_flag) {
    warm_rain(P, C, nz, W.rhof, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qr, W.rr, W.nr, W.mvd_r, T);
    rain_snow_rain_graupel(P, C, nz, W.temp, W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.l_qg,
                           W.rg, W.ng, W.ilamg, W.idx_bg, W.l_qh, W.rh, W.ilamh, T);
    ice_nucleation(P, C, nz, W.temp, W.rho, w1d, W.qv, W.qvsi, W.ssati, W.ssatw, nwfa1d, nifa1d,
                   W.nwfa, W.nifa, W.ni, W.ns, W.rc, W.nc, W.rr, W.nr, W.ilamr, T);
    ice_processes(P, C, nz, W.rhof, W.rhof2, W.rho, w1d, W.temp, W.qv, W.qvsi, W.tcond, W.diffu,
                  W.vsc2, W.ssati, W.l_qi, W.ri, W.ni, W.ilami, W.l_qs, W.rs, W.smoe, W.smof,
                  W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg,
                  W.l_qh, W.rh, W.nh, W.ilamh, T);
    riming(P, C, nz, W.temp, W.rhof, W.visco, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qs, W.rs,
           W.smo0, W.smob, W.smoc, W.smoe, W.vtboost, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg,
           W.l_qh, W.rh, W.nh, W.ilamh, T);
    melting(P, C, nz, W.rhof2, W.rho, W.temp, W.qvsi, W.tcond, W.diffu, W.vsc2, W.ssati, W.delqvs,
            W.l_qs, W.rs, W.smof, W.smo0, W.smo1, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg,
            W.l_qh, W.rh, W.nh, W.ilamh, W.hail_fraction, C.hailhyperaware_flag != 0, T);
    if (nwfa1d != nullptr || nifa1d != nullptr) {
      aerosol_scavenging(P, C, nz, W.temp, W.rho, W.rhof, W.visco, W.nwfa, W.nifa, W.l_qr, W.nr,
                         W.ilamr, W.mvd_r, W.l_qs, W.rs, W.smob, W.smoc, W.smoe, W.l_qg, W.rg, W.ng,
                         W.ilamg, W.idx_bg, T);
    }
  }

  check_over_depletion(P, C, nz, W.rho, W.temp, W.qvsi, W.qv, W.l_qc, W.rc, W.l_qi, W.ri, W.l_qr,
                       W.rr, W.l_qs, W.rs, W.l_qg, W.rg, W.l_qh, W.rh, T);
  sum_tendencies(P, nz, W.rho, W.temp, W.idx_bg, W.lvap, W.ocp, T, W.tten, W.qvten, W.qcten,
                 W.ncten, W.qiten, W.niten, W.qsten, W.qrten, W.nrten, W.qgten, W.ngten, W.qbten,
                 W.qhten);

  for (int k = 1; k <= nz; ++k) {
    W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
    tempc = W.temp[k] - t0;
    W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.nwfaten[k] = W.nwfaten[k] - (T.pna_rca[k] + T.pna_sca[k] + T.pna_gca[k] + T.pni_iha[k]) / W.rho[k];
    // v3.2.4 #53: the nucleated ice leaves the ice-friendly aerosol
    W.nifaten[k] = W.nifaten[k] - (T.pnd_rcd[k] + T.pnd_scd[k] + T.pnd_gcd[k] + T.pni_inu[k]) / W.rho[k];
  }
  (void)tempc;

  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);

  // temporaries so the 1d variables are not updated here
  double* xncx = nullptr;
  double* xngx = nullptr;
  double* xqbx = nullptr;
  tempo_copy(W.xrx, qc1d, nz);
  if (nc1d != nullptr) {
    tempo_copy(W.xncx, nc1d, nz);
    xncx = W.xncx;
  }
  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, xncx, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);

  tempo_copy(W.xrx, qr1d, nz);
  tempo_copy(W.xnx, nr1d, nz);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);

  tempo_copy(W.xrx, qi1d, nz);
  tempo_copy(W.xnx, ni1d, nz);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, W.xrx, W.xnx, W.ri, W.ni, W.qiten, W.niten, W.ilami);

  tempo_copy(W.xrx, qs1d, nz);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, W.xrx, W.rs, W.qsten);
  for (int k = 1; k <= nz; ++k) {
    W.smo0[k] = 0.0; W.smo1[k] = 0.0; W.smo2[k] = 0.0; W.smob[k] = 0.0; W.smoc[k] = 0.0;
    W.smoe[k] = 0.0; W.smof[k] = 0.0; W.smog[k] = 0.0; W.ns[k] = 0.0;
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                   &W.smo2[k], nullptr, nullptr, nullptr, nullptr);
    }
  }

  tempo_copy(W.xrx, qg1d, nz);
  if (ng1d != nullptr && qb1d != nullptr) {
    tempo_copy(W.xngx, ng1d, nz);
    tempo_copy(W.xqbx, qb1d, nz);
    xngx = W.xngx;
    xqbx = W.xqbx;
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, W.xrx, xngx, xqbx, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);

  tempo_copy(W.xrx, W.qh1d, nz);
  hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.xrx, W.rh, W.nh, W.qhten, W.ilamh);

  TEMPO_THERMO();

  // cloud condensation
  if (!C.turn_off_micro_flag && C.cloud_condensation_flag) {
    cloud_condensation(P, C, nz, W.rho, W.temp, w1d, W.ssatw, W.lvap, W.tcond, W.diffu, W.lvt2,
                       nwfa1d, W.nwfa, ncsave, W.qv, W.qvs, W.l_qc, W.rc, W.nc, T);
    for (int k = 1; k <= nz; ++k) {
      W.qvten[k] = W.qvten[k] - T.prw_vcd[k];
      W.qcten[k] = W.qcten[k] + T.prw_vcd[k];
      W.ncten[k] = W.ncten[k] + T.pnc_wcd[k];
      W.nwfaten[k] = W.nwfaten[k] - T.pnc_wcd[k];
      W.tten[k] = W.tten[k] + W.lvap[k] * W.ocp[k] * T.prw_vcd[k];
    }
    tempo_copy(W.xrx, qc1d, nz);
    if (nc1d != nullptr) {
      tempo_copy(W.xncx, nc1d, nz);
      xncx = W.xncx;
    }
    cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, xncx, ncsave, W.rc, W.nc, W.qcten,
                           W.ncten, W.ilamc, W.mvd_c);
    for (int k = 1; k <= nz; ++k) {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    }
    TEMPO_THERMO();
  }

  // rain evaporation
  if (!C.turn_off_micro_flag) {
    rain_evaporation(P, C, nz, W.rho, W.temp, W.ssatw, W.lvap, W.tcond, W.diffu, W.vsc2, W.rhof2,
                     W.qv, W.qvs, W.l_qr, W.rr, W.nr, W.ilamr, T);
    for (int k = 1; k <= nz; ++k) {
      W.qrten[k] = W.qrten[k] - T.prv_rev[k];
      W.qvten[k] = W.qvten[k] + T.prv_rev[k];
      W.nrten[k] = W.nrten[k] - T.pnr_rev[k];
      W.nwfaten[k] = W.nwfaten[k] + T.pnr_rev[k];
      W.tten[k] = W.tten[k] - W.lvap[k] * W.ocp[k] * T.prv_rev[k];
    }
    tempo_copy(W.xrx, qr1d, nz);
    tempo_copy(W.xnx, nr1d, nz);
    rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
    for (int k = 1; k <= nz; ++k) {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    }
    TEMPO_THERMO();
  }

  // sedimentation: rain.  v3.0.5 #29: the sub-stepped (not semi-Lagrangian)
  // path no longer recomputes the fall speeds inside the sub-steps.
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0;
  if (tempo_any(W.l_qr, nz)) {
    rain_fallspeed(P, C, nz, W.rhof, W.l_qr, W.rr, W.ilamr, dz1d, W.vtrr, W.vtnr, &substeps_sedi, &ktop_sedi);
    if (C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rr, W.qrten, W.vtrr, substeps_sedi, PS(r1), &W.rain_precip);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.nr, W.nrten, W.vtnr, substeps_sedi, PS(r2), nullptr);
        tempo_fill(W.vtrr, 0.0, nz1);
        tempo_fill(W.vtnr, 0.0, nz1);
        tempo_copy(W.xrx, qr1d, nz);
        tempo_copy(W.xnx, nr1d, nz);
        rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
        rain_fallspeed(P, C, nz, W.rhof, W.l_qr, W.rr, W.ilamr, dz1d, W.vtrr, W.vtnr, nullptr, nullptr);
      }
    } else {
      for (n = 1; n <= substeps_sedi; ++n) {
        sedimentation(P, C, nz, W.rr, W.vtrr, dz1d, W.rho, W.qrten, PS(r1), substeps_sedi, ktop_sedi, &W.rain_precip, W.sed_r);
        sedimentation(P, C, nz, W.nr, W.vtnr, dz1d, W.rho, W.nrten, PS(r2), substeps_sedi, ktop_sedi, nullptr, W.sed_r);
      }
    }
  }

  // graupel
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0;
  const bool has_qb1d = (qb1d != nullptr);
  const double rb_limit = PS(meters3_to_liters) * PS(r1) / PA1(rho_g, nrhg);
  if (tempo_any(W.l_qg, nz)) {
    graupel_fallspeed(P, C, nz, W.rhof, W.rho, W.visco, W.l_qg, W.rg, W.rb, has_qb1d, W.idx_bg,
                      W.ilamg, dz1d, W.vtrg, W.vtng, &substeps_sedi, &ktop_sedi);
    if (C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rg, W.qgten, W.vtrg, substeps_sedi, PS(r1), &W.graupel_liquid_equiv_precip);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.ng, W.ngten, W.vtng, substeps_sedi, PS(r2), nullptr);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rb, W.qbten, W.vtrg, substeps_sedi, rb_limit, nullptr);
        tempo_fill(W.vtrg, 0.0, nz1);
        tempo_fill(W.vtng, 0.0, nz1);
        tempo_copy(W.xrx, qg1d, nz);
        if (ng1d != nullptr && qb1d != nullptr) {
          tempo_copy(W.xngx, ng1d, nz);
          tempo_copy(W.xqbx, qb1d, nz);
          xngx = W.xngx;
          xqbx = W.xqbx;
        }
        graupel_check_and_update(P, C, nz, W.rho, W.l_qg, W.xrx, xngx, xqbx, W.rg, W.ng, W.rb, W.idx_bg,
                                 W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
        graupel_fallspeed(P, C, nz, W.rhof, W.rho, W.visco, W.l_qg, W.rg, W.rb, has_qb1d, W.idx_bg,
                          W.ilamg, dz1d, W.vtrg, W.vtng, nullptr, nullptr);
      }
    } else {
      for (n = 1; n <= substeps_sedi; ++n) {
        sedimentation(P, C, nz, W.rg, W.vtrg, dz1d, W.rho, W.qgten, PS(r1), substeps_sedi, ktop_sedi, &W.graupel_liquid_equiv_precip, W.sed_r);
        sedimentation(P, C, nz, W.ng, W.vtng, dz1d, W.rho, W.ngten, PS(r2), substeps_sedi, ktop_sedi, nullptr, W.sed_r);
        sedimentation(P, C, nz, W.rb, W.vtrg, dz1d, W.rho, W.qbten, rb_limit, substeps_sedi, ktop_sedi, nullptr, W.sed_r);
      }
    }
  }

  // hail (v3.2.0 #48)
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0;
  if (tempo_any(W.l_qh, nz)) {
    hail_fallspeed(P, C, nz, W.rhof, W.rh, W.ilamh, dz1d, W.vtrh, &substeps_sedi, &ktop_sedi);
    if (C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rh, W.qhten, W.vtrh, substeps_sedi, PS(r1), &W.hail_liquid_equiv_precip);
        tempo_fill(W.vtrh, 0.0, nz1);
        tempo_copy(W.xrx, W.qh1d, nz);
        hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.xrx, W.rh, W.nh, W.qhten, W.ilamh);
        hail_fallspeed(P, C, nz, W.rhof, W.rh, W.ilamh, dz1d, W.vtrh, nullptr, nullptr);
      }
    } else {
      for (n = 1; n <= substeps_sedi; ++n) {
        sedimentation(P, C, nz, W.rh, W.vtrh, dz1d, W.rho, W.qhten, PS(r1), substeps_sedi, ktop_sedi, &W.hail_liquid_equiv_precip, W.sed_r);
      }
    }
  }

  // snow
  ktop_sedi = 1;
  substeps_sedi = 1;
  if (tempo_any(W.l_qs, nz)) {
    snow_fallspeed(P, C, nz, W.rhof, W.l_qs, W.rs, T.prr_sml, W.smob, W.smoc, W.rr, W.vtrr, dz1d,
                   W.vtrs, W.vtboost, &substeps_sedi, &ktop_sedi);
    for (n = 1; n <= substeps_sedi; ++n) {
      sedimentation(P, C, nz, W.rs, W.vtrs, dz1d, W.rho, W.qsten, PS(r1), substeps_sedi, ktop_sedi,
                    &W.snow_liquid_equiv_precip, W.sed_r);
    }
  }

  // ice.  v3.2.3 #51: sub-stepped like the other species (v3.0.3 ran one
  // pass with a 1/substeps flux, so cloud ice fell substeps times too slowly).
  ktop_sedi = 1;
  substeps_sedi = 1;
  if (tempo_any(W.l_qi, nz)) {
    ice_fallspeed(P, C, nz, W.rhof, W.l_qi, W.ri, W.ilami, dz1d, W.vtri, W.vtni, &substeps_sedi, &ktop_sedi);
    for (n = 1; n <= substeps_sedi; ++n) {
      sedimentation(P, C, nz, W.ri, W.vtri, dz1d, W.rho, W.qiten, PS(r1), substeps_sedi, ktop_sedi,
                    &W.ice_liquid_equiv_precip, W.sed_r);
      sedimentation(P, C, nz, W.ni, W.vtni, dz1d, W.rho, W.niten, PS(r2), substeps_sedi, ktop_sedi,
                    nullptr, W.sed_r);
    }
  }

  // cloud
  ktop_sedi = 1;
  substeps_sedi = 1;
  if (tempo_any(W.l_qc, nz)) {
    cloud_fallspeed(P, C, nz, W.rhof, w1d, W.l_qc, W.rc, W.nc, W.ilamc, dz1d, W.vtrc, W.vtnc, &ktop_sedi);
    sedimentation(P, C, nz, W.rc, W.vtrc, dz1d, W.rho, W.qcten, PS(r1), substeps_sedi, ktop_sedi,
                  &W.cloud_precip, W.sed_r);
    sedimentation(P, C, nz, W.nc, W.vtnc, dz1d, W.rho, W.ncten, PS(r2), substeps_sedi, ktop_sedi,
                  nullptr, W.sed_r);
  }

  if (!C.turn_off_micro_flag) {
    freeze_cloud_melt_ice(P, C, nz, W.temp, W.rho, W.ocp, W.lvap, qi1d, ni1d, W.qiten, W.niten,
                          qc1d, nc1d, W.qcten, W.ncten, W.tten, ncsave);
  }

  // final update
  for (int k = 1; k <= nz; ++k) {
    t1d[k] = t1d[k] + W.tten[k] * C.global_dt;
    qv1d[k] = tmax(PS(min_qv), (qv1d[k] + W.qvten[k] * C.global_dt));
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    if (nwfa1d != nullptr)
      nwfa1d[k] = tmax(PS(nwfa_default), tmin(PS(aero_max), (nwfa1d[k] + W.nwfaten[k] * C.global_dt)));
    if (nifa1d != nullptr)
      nifa1d[k] = tmax(PS(nifa_default), tmin(PS(aero_max), (nifa1d[k] + W.nifaten[k] * C.global_dt)));
  }

  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, qr1d, nr1d, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, qi1d, ni1d, W.ri, W.ni, W.qiten, W.niten, W.ilami);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, qs1d, W.rs, W.qsten);
  for (int k = 1; k <= nz; ++k) {
    W.smo0[k] = 0.0; W.smo1[k] = 0.0; W.smo2[k] = 0.0; W.smob[k] = 0.0; W.smoc[k] = 0.0;
    W.smoe[k] = 0.0; W.smof[k] = 0.0; W.smog[k] = 0.0; W.smoz[k] = 0.0; W.ns[k] = 0.0;
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                   &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.qh1d, W.rh, W.nh, W.qhten, W.ilamh);

  // diagnostics
  W.frozen_fraction =
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip
       + W.hail_liquid_equiv_precip) /
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip
       + W.hail_liquid_equiv_precip + W.rain_precip + PS(r1));
  W.graupel_liquid_equiv_precip = W.graupel_liquid_equiv_precip + W.hail_liquid_equiv_precip;
  freezing_rain(P, W.temp[1], W.rain_precip, W.cloud_precip, &W.frz_rain_precip);
  for (int k = 1; k <= nz; ++k) {
    W.cloud_number_mixing_ratio[k] = W.nc[k] * W.rho[k];
    W.rain_med_vol_diam[k] = W.mvd_r[k];
    W.graupel_med_vol_diam[k] = W.mvd_g[k];
  }
  if (C.refl10cm_flag) {
    reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp, W.l_qr, W.rr, W.nr, W.ilamr,
                      W.l_qs, W.rs, W.smoc, W.smob, W.smoz, W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg,
                      W.l_qh, W.nh, W.ilamh, W.refl10cm);
  }

  // graupel and hail recombined for the model state and the hail-size
  // diagnostic, main.F90:913-937
  for (int k = 1; k <= nz; ++k) {
    if (W.l_qh[k]) {
      qg1d[k] = qg1d[k] + W.qh1d[k];
      if (W.ngten[k] >= 0.0) ng1d[k] = ng1d[k] + W.nh[k] / W.rho[k];
      qb1d[k] = qb1d[k] + W.qh1d[k] * PS(meters3_to_liters) / PA1(rho_g, nrhg);
    }
  }
  for (int k = 1; k <= nz; ++k) { W.qgten[k] = 0.0; W.ngten[k] = 0.0; W.qbten[k] = 0.0; }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  if (C.max_hail_diameter_flag) {
    max_hail_diam(P, C, nz, W.rho, W.rg, W.ng, W.ilamg, W.idx_bg, W.max_hail_diameter);
  }
  if (C.re_cloud_flag && C.re_ice_flag && C.re_snow_flag) {
    effective_radius(P, C, nz, W.temp, W.l_qc, W.nc, W.ilamc, W.l_qi, W.ilami, W.l_qs, W.rs,
                     W.re_cloud, W.re_ice, W.re_snow);
  }
#undef TEMPO_THERMO
}
