"""MPAS-A v8.4.2 sea-ice surface and MPAS driver conversions.

Column arrays are CUDA float32, level-major (level, cell) where applicable.
The source is MPAS's complete Noah sea-ice module, including its thermal
solver dependency. WOOF's physics and default suite are unchanged.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Mapping

import numpy as np

SOURCE_FILES = (
    "physics_wrf/module_sf_noah_seaice.F:36-1292",
    "physics_wrf/module_sf_noah_seaice_drv.F:22-518",
    "mpas_atmphys_driver_seaice.F:160-561",
    "mpas_atmphys_lsm_shared.F:22-57",
    "physics_wrf/module_sf_noahlsm.F:1857-1905,2541-2600",
)
BACKEND = "hex-cuda"
SCHEME = "seaice"
LSM_SCHEMES = ("sf_noah", "sf_noahmp")

FIELDS = ('t3d', 'qv3d', 'p8w3d', 'dz8w', 'sr', 'glw', 'qgh', 'swdown', 'rainbl', 'snoalb2d', 'xice', 'rib', 'chs', 'tslb', 'emiss', 'albedo', 'albsi', 'z02d', 'snow', 'tsk', 'snowc', 'snowh2d', 'cqs2', 'acsnow', 'acsnom', 'sfcrunoff', 'potevp', 'snopcx', 'hfx', 'lh', 'qfx', 'znt', 'grdflx', 'qsfc', 'chs2', 'noahres', 'snowsi', 'icedepth', 'gsw', 'sr_input', 'rainncv', 'raincv', 'tsk_sea', 'cpm', 'cpm_sea', 'chs_sea', 'chs2_sea', 'cqs2_sea', 'hfx_sea', 'lh_sea', 'qfx_sea', 'qgh_sea', 'qsfc_sea')
OPTIONAL_FIELDS = ("potevp", "snopcx", "noahres")


@dataclass(frozen=True)
class SeaiceOptions:
    """MPAS controls, supplied by the physics seam rather than suite defaults."""

    dt: float
    xice_threshold: float
    seaice_albedo_opt: int
    seaice_albedo_default: float
    seaice_thickness_opt: int
    seaice_thickness_default: float
    seaice_snowdepth_opt: int
    seaice_snowdepth_max: float
    seaice_snowdepth_min: float
    frpcpn: bool = False  # MPAS driver has frpcpn=.false.


_CORE_SOURCE = r"""
// MPAS module_sf_noah_seaice.F:222-1292 and wrapper .F:189-511:
// GNU scalar SSE keeps NaN operand payload/sign; CUDA arithmetic canonicalizes.
// Keep rounded device arithmetic for finite results, and reproduce the
// measured GNU operand words for nonfinite results. Shared libm bodies stay intact.
__device__ inline float si_nan_result(float result,float a,float b) {
  unsigned int r=__float_as_uint(result);
  if ((r & 0x7fffffffu)<=0x7f800000u) return result;
  unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
  if ((x & 0x7fffffffu)>0x7f800000u) return __uint_as_float(x|0x00400000u);
  if ((y & 0x7fffffffu)>0x7f800000u) return __uint_as_float(y|0x00400000u);
  return __uint_as_float(0xffc00000u);
}
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
__device__ inline float si_add(float a,float b) { return si_nan_result(__fadd_rn(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
__device__ inline float si_sub(float a,float b) { return si_nan_result(__fsub_rn(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
__device__ inline float si_mul(float a,float b) { return si_nan_result(__fmul_rn(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
__device__ inline float si_div(float a,float b) { return si_nan_result(__fdiv_rn(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
__device__ inline float si_neg(float a) {
  unsigned int bits=__float_as_uint(a);
#if defined(__CUDA_ARCH__)
  // NVRTC folds a C++ sign-bit XOR into neg.f32, which canonicalizes NaNs.
  // Native XORPS preserves payload/sign; retain that integer operation.
  asm volatile("xor.b32 %0, %1, 0x80000000;" : "=r"(bits) : "r"(bits));
#else
  bits ^= 0x80000000u;
#endif
  return __uint_as_float(bits);
}

constexpr float cp=1004.5f, rd=287.04f, sigma=5.67e-8f, cph2o=4.218e3f, cpice=2.106e3f, lsubf=3.335e5f, tfreez=273.15f;
// Generated transcription reviewed against pinned MPAS-A v8.4.2; one-based private arrays.
// MPAS physics_wrf/module_sf_noahlsm.F:1857-1905 (forward declaration).
__device__ void hstep(float* stcout, float* stcin, float* rhsts, float& dt, int& nsoil, float* ai, float* bi, float* ci, int& status);
// MPAS physics_wrf/module_sf_noahlsm.F:2541-2600 (forward declaration).
__device__ void rosr12(float* p, float* a, float* b, float* c, float* d, float* delta, int& nsoil, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:36-545 (forward declaration).
__device__ void sflx_seaice(int& iiloc, int& jjloc, int& seaice_albedo_opt, float& seaice_albedo_default, int& seaice_snowdepth_opt, float& seaice_snowdepth_max, float& seaice_snowdepth_min, float& ffrozp, float& dt, float& zlvl, int& nsoil, float& sithick, float& lwdn, float& solnet, float& sfcprs, float& prcp, float& sfctmp, float& q2, float& th2, float& q2sat, float& dqsdt2, float& snoalb, float& tbot, float& z0brd, float& z0, float& emissi, float& t1, float* stc, float& snowh, float& sneqv, float& albedo, float& ch, float& albedosi, float& snowonsi, float& eta, float& sheat, float& eta_kinematic, float& fdown, float& esnow, float& dew, float& etp, float& ssoil, float& flx1, float& flx2, float& flx3, float& snomlt, float& sncovr, float& runoff1, float& q1, float& ribb, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:548-587 (forward declaration).
__device__ void csnow(float& sncond, float& dsnow, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:589-707 (forward declaration).
__device__ void hrtice(float* rhsts, float* stc, float& tbot, int& nsoil, float* zsoil, float& yy, float& zz1, float& df1, float* ai, float* bi, float* ci, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:710-782 (forward declaration).
__device__ void penman(float& sfctmp, float& sfcprs, float& ch, float& th2, float& prcp, float& fdown, float& t24, float& ssoil, float& q2, float& q2sat, float& etp, float& rch, float& rr, bool& snowng, bool& frzgra, float& dqsdt2, float& flx2, float& emissi, float& t1, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:785-812 (forward declaration).
__device__ void shflx(float* stc, int& nsoil, float& dt, float& yy, float& zz1, float* zsoil, float& tbot, float& df1, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:815-1079 (forward declaration).
__device__ void snopac(float& etp, float& eta, float& prcp, bool& snowng, int& nsoil, float& dt, float& df1, float& q2, float& t1, float& sfctmp, float& t24, float& th2, float& fdown, float& ssoil, float* stc, float& sfcprs, float& rch, float& rr, float& sncovr, float& esd, float& sndens, float& snowh, float* zsoil, float& tbot, float& snomlt, float& dew, float& flx1, float& flx2, float& flx3, float& esnow, float& emissi, float& ribb, int& seaice_albedo_opt, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1082-1207 (forward declaration).
__device__ void snowpack(float& esd, float& dtsec, float& snowh, float& sndens, float& tsnow, float& tsoil, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1210-1239 (forward declaration).
__device__ void snowz0(float& sncovr, float& z0, float& z0brd, float& snowh, int& status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1243-1292 (forward declaration).
__device__ void snow_new(float& temp, float& newsn, float& snowh, float& sndens, int& status);

// MPAS physics_wrf/module_sf_noahlsm.F:1857-1905
__device__ void hstep(float* stcout, float* stcin, float* rhsts, float& dt, int& nsoil, float* ai, float* bi, float* ci, int& status) {
  int k;
  float rhstsin[NSOLD+1];
  float ciin[NSOLD+1];
  for (k = 1; k <= nsoil; k += 1) {
  rhsts[k] = si_mul(rhsts[k], dt);
  ai[k] = si_mul(ai[k], dt);
  bi[k] = si_add(1.0f, si_mul(bi[k], dt));
  ci[k] = si_mul(ci[k], dt);
  }
  for (k = 1; k <= nsoil; k += 1) {
  rhstsin[k] = rhsts[k];
  }
  for (k = 1; k <= nsoil; k += 1) {
  ciin[k] = ci[k];
  }
  rosr12(ci, ai, bi, ciin, rhstsin, rhsts, nsoil, status);
  for (k = 1; k <= nsoil; k += 1) {
  stcout[k] = si_add(stcin[k], ci[k]);
  }
}

// MPAS physics_wrf/module_sf_noahlsm.F:2541-2600
__device__ void rosr12(float* p, float* a, float* b, float* c, float* d, float* delta, int& nsoil, int& status) {
  int k;
  int kk;
  c[nsoil] = 0.0f;
  // MPAS module_sf_noahlsm.F:2573,2583: unary minus follows each full chain.
  p[1] = si_neg(si_div(c[1], b[1]));
  delta[1] = si_div(d[1], b[1]);
  for (k = 2; k <= nsoil; k += 1) {
  p[k] = si_neg(si_mul(si_div(1.0f, si_add(si_mul(p[(k - 1)], a[k]), b[k])), c[k]));
  delta[k] = si_mul(si_sub(d[k], si_mul(a[k], delta[(k - 1)])), si_div(1.0f, si_add(si_mul(p[(k - 1)], a[k]), b[k])));
  }
  p[nsoil] = delta[nsoil];
  for (k = 2; k <= nsoil; k += 1) {
  kk = ((nsoil - k) + 1);
  p[kk] = si_add(si_mul(p[(kk + 1)], p[kk]), delta[kk]);
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:36-545
__device__ void sflx_seaice(int& iiloc, int& jjloc, int& seaice_albedo_opt, float& seaice_albedo_default, int& seaice_snowdepth_opt, float& seaice_snowdepth_max, float& seaice_snowdepth_min, float& ffrozp, float& dt, float& zlvl, int& nsoil, float& sithick, float& lwdn, float& solnet, float& sfcprs, float& prcp, float& sfctmp, float& q2, float& th2, float& q2sat, float& dqsdt2, float& snoalb, float& tbot, float& z0brd, float& z0, float& emissi, float& t1, float* stc, float& snowh, float& sneqv, float& albedo, float& ch, float& albedosi, float& snowonsi, float& eta, float& sheat, float& eta_kinematic, float& fdown, float& esnow, float& dew, float& etp, float& ssoil, float& flx1, float& flx2, float& flx3, float& snomlt, float& sncovr, float& runoff1, float& q1, float& ribb, int& status) {
  bool frzgra;
  bool snowng;
  float zsoil[NSOLD+1];
  float df1;
  float df1a;
  float dsoil;
  float dtot;
  float frcsno;
  float frcsoi;
  float rch;
  float rr;
  float sndens;
  float sncond;
  float sn_new;
  float t24;
  float t2v;
  float th2v;
  float tsnow;
  float rho;
  int kz;
  int k;
  float alb_snow;
  float alb_ice;
  float z0n;
  float sncovrr;
  const float lvh2o = 2501000.0f;
  const float lsubs = 2830000.0f;
  const float r = 287.04f;
  runoff1 = 0.0f;
  snomlt = 0.0f;
  for (kz = 1; kz <= nsoil; kz += 1) {
  zsoil[kz] = si_neg(si_div(si_mul(sithick, (float)(kz)), (float)(nsoil)));
  }
  z0brd = 0.001f;
  snowng = false;
  frzgra = false;
  switch (seaice_albedo_opt) {
  default:
  if ((sneqv < 0.01f)) {
  sneqv = 0.01f;
  snowh = 0.05f;
  }
  break;
  case 1:
  if ((sneqv < 0.0001f)) {
  sneqv = 0.0001f;
  snowh = 0.0005f;
  }
  break;
  }
  if ((seaice_snowdepth_opt == 0)) {
  sndens = si_div(sneqv, snowh);
  snowh = fmaxf(seaice_snowdepth_min, fminf(snowh, seaice_snowdepth_max));
  sneqv = si_mul(snowh, sndens);
  } else if ((seaice_snowdepth_opt == 1)) {
  sndens = 0.3f;
  snowh = snowonsi;
  sneqv = si_mul(snowh, sndens);
  }
  sndens = si_div(sneqv, snowh);
  if ((sndens > 1.0f)) {
  status = 1; return; // MPAS fatal: snow depth below water equivalent
  }
  csnow(sncond, sndens, status);
  if ((prcp > 0.0f)) {
  if ((ffrozp > 0.5f)) {
  snowng = true;
  } else {
  if ((t1 <= tfreez)) frzgra = true;
  }
  }
  if ((snowng || frzgra)) {
  sn_new = si_mul(si_mul(prcp, dt), 0.001f);
  sneqv = si_add(sneqv, sn_new);
  snow_new(sfctmp, sn_new, snowh, sndens, status);
  if ((sncovr > 0.99f)) {
  if ((stc[1] < si_sub(tfreez, 5.0f))) sndens = 0.2f;
  if ((snowng && (t1 < 273.0f) && (sfctmp < 273.0f))) sndens = 0.2f;
  }
  csnow(sncond, sndens, status);
  }
  switch (seaice_albedo_opt) {
  default:
  sncovr = 1.0f;
  emissi = 0.98f;
  albedo = seaice_albedo_default;
  break;
  case 1:
  if ((t1 < 268.15f)) {
  alb_snow = 0.8f;
  } else if (((t1 >= 268.15f) && (t1 < 273.15f))) {
  alb_snow = si_sub(0.65f, si_mul(0.03f, si_sub(t1, 273.15f)));
  } else {
  alb_snow = 0.65f;
  }
  if ((sfctmp <= 273.15f)) {
  alb_ice = 0.65f;
  } else if (((sfctmp > 273.15f) && (sfctmp < 278.15f))) {
  alb_ice = si_sub(0.65f, si_mul(0.04f, si_sub(sfctmp, 273.15f)));
  } else {
  alb_ice = 0.45f;
  }
  z0n = 0.1f;
  sncovrr = si_div(snowh, si_add(snowh, z0n));
  albedo = si_add(si_mul(sncovrr, alb_snow), si_mul(si_sub(1.0f, sncovrr), alb_ice));
  break;
  case 2:
  sncovr = 1.0f;
  emissi = 0.98f;
  albedo = albedosi;
  break;
  }
  df1 = 2.2f;
  dsoil = (si_neg(si_mul(0.5f, zsoil[1])));
  dtot = si_add(snowh, dsoil);
  frcsno = si_div(snowh, dtot);
  frcsoi = si_div(dsoil, dtot);
  df1a = si_add(si_mul(frcsno, sncond), si_mul(frcsoi, df1));
  df1 = si_add(si_mul(df1a, sncovr), si_mul(df1, si_sub(1.0f, sncovr)));
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  snowz0(sncovr, z0, z0brd, snowh, status);
  fdown = si_add(solnet, lwdn);
  t2v = si_mul(sfctmp, si_add(1.0f, si_mul(0.61f, q2)));
  t24 = si_mul(si_mul(si_mul(sfctmp, sfctmp), sfctmp), sfctmp);
  rho = si_div(sfcprs, si_mul(rd, t2v));
  rch = si_mul(si_mul(rho, 1004.6f), ch);
  penman(sfctmp, sfcprs, ch, th2, prcp, fdown, t24, ssoil, q2, q2sat, etp, rch, rr, snowng, frzgra, dqsdt2, flx2, emissi, t1, status);
  esnow = 0.0f;
  snopac(etp, eta, prcp, snowng, nsoil, dt, df1, q2, t1, sfctmp, t24, th2, fdown, ssoil, stc, sfcprs, rch, rr, sncovr, sneqv, sndens, snowh, zsoil, tbot, snomlt, dew, flx1, flx2, flx3, esnow, emissi, ribb, seaice_albedo_opt, status);
  eta_kinematic = etp;
  if ((seaice_snowdepth_opt == 0)) {
  sndens = si_div(sneqv, snowh);
  snowh = fmaxf(seaice_snowdepth_min, fminf(snowh, seaice_snowdepth_max));
  sneqv = si_mul(snowh, sndens);
  } else if ((seaice_snowdepth_opt == 1)) {
  sndens = 0.3f;
  snowh = snowonsi;
  sneqv = si_mul(snowh, sndens);
  }
  q1 = si_add(q2, si_div(si_mul(eta_kinematic, cp), rch));
  // MPAS module_sf_noah_seaice.F:514: GNU negates after the complete multiply/divide chain.
  sheat = si_neg(si_mul(si_div(si_mul(si_mul(ch, cp), sfcprs), si_mul(r, t2v)), si_sub(th2, t1)));
  esnow = si_mul(esnow, lsubs);
  etp = si_mul(etp, si_add(si_mul(si_sub(1.0f, sncovr), lvh2o), si_mul(sncovr, lsubs)));
  if ((etp > 0.0f)) {
  eta = esnow;
  } else {
  eta = etp;
  }
  // MPAS module_sf_noah_seaice.F:534: native uses sign-bit negation.
  ssoil = si_neg(ssoil);
  runoff1 = si_div(snomlt, dt);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:548-587
__device__ void csnow(float& sncond, float& dsnow, int& status) {
  float c;
  const float unit = 0.11631f;
  c = si_mul(0.328f, si_nan_result(mynn_glibc_powf(10, si_mul(2.25f, dsnow)), 10,  si_mul(2.25f, dsnow)));
  sncond = si_mul(si_mul(2.0f, unit), c);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:589-707
__device__ void hrtice(float* rhsts, float* stc, float& tbot, int& nsoil, float* zsoil, float& yy, float& zz1, float& df1, float* ai, float* bi, float* ci, int& status) {
  int k;
  float ddz;
  float ddz2;
  float denom;
  float dtsdz;
  float dtsdz2;
  float ssoil;
  float zbot;
  float hcpct;
  float df1k;
  float df1n;
  float zmd;
  hcpct = 1723960.0f;
  zbot = zsoil[nsoil];
  ddz = si_div(1.0f, si_mul((-0.5f), zsoil[2]));
  ai[1] = 0.0f;
  ci[1] = si_div(si_mul(df1, ddz), si_mul(zsoil[1], hcpct));
  bi[1] = si_add(si_neg(ci[1]), si_div(df1, si_mul(si_mul(si_mul(si_mul(0.5f, zsoil[1]), zsoil[1]), hcpct), zz1)));
  dtsdz = si_div(si_sub(stc[1], stc[2]), si_mul((-0.5f), zsoil[2]));
  ssoil = si_div(si_mul(df1, si_sub(stc[1], yy)), si_mul(si_mul(0.5f, zsoil[1]), zz1));
  rhsts[1] = si_div(si_sub(si_mul(df1, dtsdz), ssoil), si_mul(zsoil[1], hcpct));
  ddz2 = 0.0f;
  df1k = df1;
  df1n = df1;
  for (k = 2; k <= nsoil; k += 1) {
  if ((k != nsoil)) {
  denom = si_mul(0.5f, si_sub(zsoil[(k - 1)], zsoil[(k + 1)]));
  dtsdz2 = si_div(si_sub(stc[k], stc[(k + 1)]), denom);
  ddz2 = si_div(2.0f, si_sub(zsoil[(k - 1)], zsoil[(k + 1)]));
  ci[k] = si_neg(si_div(si_mul(df1n, ddz2), si_mul(si_sub(zsoil[(k - 1)], zsoil[k]), hcpct)));
  } else {
  dtsdz2 = si_div(si_sub(stc[k], tbot), si_sub(si_mul(0.5f, si_add(zsoil[(k - 1)], zsoil[k])), zbot));
  ci[k] = 0.0f;
  }
  denom = si_mul(si_sub(zsoil[k], zsoil[(k - 1)]), hcpct);
  rhsts[k] = si_div(si_sub(si_mul(df1n, dtsdz2), si_mul(df1k, dtsdz)), denom);
  ai[k] = si_neg(si_div(si_mul(df1k, ddz), si_mul(si_sub(zsoil[(k - 1)], zsoil[k]), hcpct)));
  bi[k] = (si_neg(si_add(ai[k], ci[k])));
  df1k = df1n;
  dtsdz = dtsdz2;
  ddz = ddz2;
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:710-782
__device__ void penman(float& sfctmp, float& sfcprs, float& ch, float& th2, float& prcp, float& fdown, float& t24, float& ssoil, float& q2, float& q2sat, float& etp, float& rch, float& rr, bool& snowng, bool& frzgra, float& dqsdt2, float& flx2, float& emissi, float& t1, int& status) {
  float elcp1;
  float lvs;
  float epsca;
  float a;
  float delta;
  float fnet;
  float rad;
  const float elcp = 2488.8f;
  const float lsubc = 2501000.0f;
  const float cp = 1004.6f;
  const float lsubs = 2830000.0f;
  if ((t1 > 273.15f)) {
  elcp1 = elcp;
  lvs = lsubc;
  } else {
  elcp1 = si_div(si_mul(elcp, lsubs), lsubc);
  lvs = lsubs;
  }
  flx2 = 0.0f;
  delta = si_mul(elcp1, dqsdt2);
  rr = si_add(si_div(si_mul(si_mul(emissi, t24), 6.48e-08f), si_mul(sfcprs, ch)), 1.0f);
  if ((prcp > 0.0f)) {
  if ((!snowng)) {
  rr = si_add(rr, si_div(si_mul(cph2o, prcp), rch));
  } else {
  rr = si_add(rr, si_div(si_mul(cpice, prcp), rch));
  }
  }
  fnet = si_sub(si_sub(fdown, si_mul(si_mul(emissi, sigma), t24)), ssoil);
  if (frzgra) {
  flx2 = si_neg(si_mul(lsubf, prcp));
  fnet = si_sub(fnet, flx2);
  }
  rad = si_sub(si_add(si_div(fnet, rch), th2), sfctmp);
  a = si_mul(elcp1, si_sub(q2sat, q2));
  epsca = si_div(si_add(si_mul(a, rr), si_mul(rad, delta)), si_add(delta, rr));
  etp = si_div(si_mul(epsca, rch), lvs);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:785-812
__device__ void shflx(float* stc, int& nsoil, float& dt, float& yy, float& zz1, float* zsoil, float& tbot, float& df1, int& status) {
  float ai[NSOLD+1];
  float bi[NSOLD+1];
  float ci[NSOLD+1];
  float stcf[NSOLD+1];
  float rhsts[NSOLD+1];
  int i;
  const float t0 = 273.15f;
  hrtice(rhsts, stc, tbot, nsoil, zsoil, yy, zz1, df1, ai, bi, ci, status);
  hstep(stcf, stc, rhsts, dt, nsoil, ai, bi, ci, status);
  for (i = 1; i <= nsoil; i += 1) {
  stc[i] = stcf[i];
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:815-1079
__device__ void snopac(float& etp, float& eta, float& prcp, bool& snowng, int& nsoil, float& dt, float& df1, float& q2, float& t1, float& sfctmp, float& t24, float& th2, float& fdown, float& ssoil, float* stc, float& sfcprs, float& rch, float& rr, float& sncovr, float& esd, float& sndens, float& snowh, float* zsoil, float& tbot, float& snomlt, float& dew, float& flx1, float& flx2, float& flx3, float& esnow, float& emissi, float& ribb, int& seaice_albedo_opt, int& status) {
  int k;
  float denom;
  float dsoil;
  float dtot;
  float esnow1;
  float esnow2;
  float eta1;
  float etp1;
  float etp2;
  float etanrg;
  float ex;
  float seh;
  float sncond;
  float t12;
  float t12a;
  float t12b;
  float t14;
  float yy;
  float zz1;
  const float esdmin = 1e-06f;
  const float lsubc = 2501000.0f;
  const float lsubs = 2830000.0f;
  const float snoexp = 2.0f;
  dew = 0.0f;
  esnow = 0.0f;
  esnow1 = 0.0f;
  esnow2 = 0.0f;
  if ((etp <= 0.0f)) {
  if (((ribb >= 0.1f) && (fdown > 150.0f))) {
  etp = si_div(si_add(si_div(si_mul(fminf(si_mul(etp, si_sub(1.0f, ribb)), 0.0f), sncovr), 0.98f), si_mul(etp, si_sub(0.98f, sncovr))), 0.98f);
  }
  etp1 = si_mul(etp, 0.001f);
  dew = si_neg(etp1);
  esnow2 = si_mul(etp1, dt);
  etanrg = si_mul(etp, si_add(si_mul(si_sub(1.0f, sncovr), lsubc), si_mul(sncovr, lsubs)));
  } else {
  etp1 = si_mul(etp, 0.001f);
  esnow = etp;
  esnow1 = si_mul(esnow, 0.001f);
  esnow2 = si_mul(esnow1, dt);
  etanrg = si_mul(esnow, lsubs);
  esnow = si_mul(etp, sncovr);
  esnow1 = si_mul(esnow, 0.001f);
  esnow2 = si_mul(esnow1, dt);
  etanrg = si_mul(esnow, lsubs);
  }
  flx1 = 0.0f;
  if (snowng) {
  flx1 = si_mul(si_mul(cpice, prcp), si_sub(t1, sfctmp));
  } else {
  if ((prcp > 0.0f)) flx1 = si_mul(si_mul(cph2o, prcp), si_sub(t1, sfctmp));
  }
  dsoil = (si_neg(si_mul(0.5f, zsoil[1])));
  dtot = si_add(snowh, dsoil);
  denom = si_add(1.0f, si_div(df1, si_mul(si_mul(dtot, rr), rch)));
  t12a = si_div(si_sub(si_sub(si_add(si_div(si_sub(si_sub(si_sub(fdown, flx1), flx2), si_mul(si_mul(emissi, sigma), t24)), rch), th2), sfctmp), si_div(etanrg, rch)), rr);
  t12b = si_div(si_mul(df1, stc[1]), si_mul(si_mul(dtot, rr), rch));
  t12 = si_div(si_add(si_add(sfctmp, t12a), t12b), denom);
  if ((t12 <= tfreez)) {
  t1 = t12;
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  esd = fmaxf(0.0f, si_sub(esd, esnow2));
  flx3 = 0.0f;
  ex = 0.0f;
  snomlt = 0.0f;
  } else {
  t1 = tfreez;
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  if ((si_sub(esd, esnow2) <= esdmin)) {
  esd = 0.0f;
  ex = 0.0f;
  snomlt = 0.0f;
  flx3 = 0.0f;
  } else {
  esd = si_sub(esd, esnow2);
  seh = si_mul(rch, si_sub(t1, th2));
  t14 = si_mul(si_mul(t1, t1), si_mul(t1, t1));
  flx3 = si_sub(si_sub(si_sub(si_sub(si_sub(si_sub(fdown, flx1), flx2), si_mul(si_mul(emissi, sigma), t14)), ssoil), seh), etanrg);
  if ((flx3 <= 0.0f)) flx3 = 0.0f;
  ex = si_div(si_mul(flx3, 0.001f), lsubf);
  snomlt = si_mul(ex, dt);
  if ((si_sub(esd, snomlt) >= esdmin)) {
  esd = si_sub(esd, snomlt);
  } else {
  ex = si_div(esd, dt);
  flx3 = si_mul(si_mul(ex, 1000.0f), lsubf);
  snomlt = esd;
  esd = 0.0f;
  }
  }
  }
  zz1 = 1.0f;
  yy = si_sub(stc[1], si_div(si_mul(si_mul(si_mul(0.5f, ssoil), zsoil[1]), zz1), df1));
  shflx(stc, nsoil, dt, yy, zz1, zsoil, tbot, df1, status);
  switch (seaice_albedo_opt) {
  default:
  if ((esd >= 0.01f)) {
  snowpack(esd, dt, snowh, sndens, t1, yy, status);
  } else {
  esd = 0.01f;
  snowh = 0.05f;
  sncovr = 1.0f;
  }
  break;
  case 1:
  if ((esd >= 0.0001f)) {
  snowpack(esd, dt, snowh, sndens, t1, yy, status);
  } else {
  esd = 0.0001f;
  snowh = 0.0005f;
  sncovr = 0.005f;
  }
  break;
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1082-1207
__device__ void snowpack(float& esd, float& dtsec, float& snowh, float& sndens, float& tsnow, float& tsoil, int& status) {
  int ipol;
  int j;
  float bfac;
  float dsx;
  float dthr;
  float dw;
  float snowhc;
  float pexp;
  float tavgc;
  float tsnowc;
  float tsoilc;
  float esdc;
  float esdcx;
  const float c1 = 0.01f;
  const float c2 = 21.0f;
  const float g = 9.81f;
  const float kn = 4000.0f;
  snowhc = si_mul(snowh, 100.0f);
  esdc = si_mul(esd, 100.0f);
  dthr = si_div(dtsec, 3600.0f);
  tsnowc = si_sub(tsnow, 273.15f);
  tsoilc = si_sub(tsoil, 273.15f);
  tavgc = si_mul(0.5f, si_add(tsnowc, tsoilc));
  if ((esdc > 0.01f)) {
  esdcx = esdc;
  } else {
  esdcx = 0.01f;
  }
  bfac = si_mul(si_mul(dthr, c1), si_nan_result(mynn_glibc_expf(si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens))), si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens)), si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens))));
  ipol = 4;
  pexp = 0.0f;
  for (j = ipol; j >= 1; j += (-1)) {
  pexp = si_div(si_mul(si_mul(si_add(1.0f, pexp), bfac), esdcx), (float)((j + 1)));
  }
  pexp = si_add(pexp, 1.0f);
  dsx = si_mul(sndens, pexp);
  if ((dsx > 0.4f)) dsx = 0.4f;
  if ((dsx < 0.05f)) dsx = 0.05f;
  sndens = dsx;
  if ((tsnowc >= 0.0f)) {
  dw = si_div(si_mul(0.13f, dthr), 24.0f);
  sndens = si_add(si_mul(sndens, si_sub(1.0f, dw)), dw);
  if ((sndens >= 0.4f)) sndens = 0.4f;
  }
  snowhc = si_div(esdc, sndens);
  snowh = si_mul(snowhc, 0.01f);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1210-1239
__device__ void snowz0(float& sncovr, float& z0, float& z0brd, float& snowh, int& status) {
  const float z0s = 0.001f;
  float burial;
  float z0eff;
  burial = si_sub(si_mul(7.0f, z0brd), snowh);
  if ((burial <= 0.0007f)) {
  z0eff = z0s;
  } else {
  z0eff = si_div(burial, 7.0f);
  }
  z0 = si_add(si_mul(si_sub(1.0f, sncovr), z0brd), si_mul(sncovr, z0eff));
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1243-1292
__device__ void snow_new(float& temp, float& newsn, float& snowh, float& sndens, int& status) {
  float dsnew;
  float hnewc;
  float snowhc;
  float newsnc;
  float tempc;
  snowhc = si_mul(snowh, 100.0f);
  newsnc = si_mul(newsn, 100.0f);
  tempc = si_sub(temp, 273.15f);
  if ((tempc <= (-15.0f))) {
  dsnew = 0.05f;
  } else {
  dsnew = si_add(0.05f, si_mul(0.0017f, si_nan_result(mynn_glibc_powf(si_add(tempc, 15.0f), 1.5f), si_add(tempc, 15.0f),  1.5f)));
  }
  hnewc = si_div(newsnc, dsnew);
  if ((si_add(snowhc, hnewc) < 0.001f)) {
  sndens = fmaxf(dsnew, sndens);
  } else {
  sndens = si_div(si_add(si_mul(snowhc, sndens), si_mul(hnewc, dsnew)), si_add(snowhc, hnewc));
  }
  snowhc = si_add(snowhc, hnewc);
  snowh = si_mul(snowhc, 0.01f);
}
#define P_t3d 0
#define P_qv3d 1
#define P_p8w3d 2
#define P_dz8w 3
#define P_sr 4
#define P_glw 5
#define P_qgh 6
#define P_swdown 7
#define P_rainbl 8
#define P_snoalb2d 9
#define P_xice 10
#define P_rib 11
#define P_chs 12
#define P_tslb 13
#define P_emiss 14
#define P_albedo 15
#define P_albsi 16
#define P_z02d 17
#define P_snow 18
#define P_tsk 19
#define P_snowc 20
#define P_snowh2d 21
#define P_cqs2 22
#define P_acsnow 23
#define P_acsnom 24
#define P_sfcrunoff 25
#define P_potevp 26
#define P_snopcx 27
#define P_hfx 28
#define P_lh 29
#define P_qfx 30
#define P_znt 31
#define P_grdflx 32
#define P_qsfc 33
#define P_chs2 34
#define P_noahres 35
#define P_snowsi 36
#define P_icedepth 37
#define P_gsw 38
#define P_sr_input 39
#define P_rainncv 40
#define P_raincv 41
#define P_tsk_sea 42
#define P_cpm 43
#define P_cpm_sea 44
#define P_chs_sea 45
#define P_chs2_sea 46
#define P_cqs2_sea 47
#define P_hfx_sea 48
#define P_lh_sea 49
#define P_qfx_sea 50
#define P_qgh_sea 51
#define P_qsfc_sea 52
#define F(name) a[P_##name][i]
// MPAS physics_wrf/module_sf_noah_seaice_drv.F:22-518
extern "C" __global__ void seaice_noah(float** a, int n, int nsoil,
    float dt, int albedo_opt, float albedo_default, int thickness_opt,
    float thickness_default, int snowdepth_opt, float snowdepth_max,
    float snowdepth_min, float threshold, int frpcpn, int optional_noah,
    int* errors, int* last_sflx_column) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i >= n) return;
  int status = 0;
  if (F(xice) < threshold) {
    if (thickness_opt == 1) F(icedepth) = 0.0f;
    if (snowdepth_opt == 1) F(snowsi) = 0.0f;
    return;
  }
  float sithick;
  if (thickness_opt == 0) sithick = thickness_default;
  else if (thickness_opt == 1) {
    if (F(icedepth) < -1.0e6f) { errors[i]=2; return; }
    sithick = fminf(fmaxf(0.10f,F(icedepth)),10.0f);
    F(icedepth)=sithick;
  } else { errors[i]=3; return; }
  float sfctmp=F(t3d), t1=F(tsk), snoalb, albedosi=0.0f;
  if (albedo_opt == 2) {
    if (F(albsi) < -1.0e6f) { errors[i]=4; return; }
    snoalb=F(albsi); F(albedo)=F(albsi); albedosi=F(albsi);
  } else snoalb=F(snoalb2d);
  float zlvl=si_mul(0.5f,F(dz8w)), emissi=F(emiss), lwdn=si_mul(F(glw),emissi);
  float sneqv=si_mul(F(snow),0.001f), snowh=F(snowh2d), sncovr=F(snowc);
  float solnet=si_mul(F(swdown),si_sub(1.0f,F(albedo)));
  float sfcprs=si_mul(si_add(a[P_p8w3d][n+i],F(p8w3d)),0.5f);
  float psfc=F(p8w3d), q2=si_div(F(qv3d),si_add(1.0f,F(qv3d)));
  const float capa=287.0f/1004.5f;
  const float a23m4=17.67f*(273.15f-29.65f);
  float apes=si_nan_result(mynn_glibc_powf(si_div(1.0e5f,psfc),capa), si_div(1.0e5f,psfc), capa);
  float apelm=si_nan_result(mynn_glibc_powf(si_div(1.0e5f,sfcprs),capa), si_div(1.0e5f,sfcprs), capa);
  float th2=si_div(si_mul(sfctmp,apelm),apes);
  float q2sat=si_div(F(qgh),si_add(1.0f,F(qgh)));
  float tempdiff=si_sub(sfctmp,29.65f);
  float dqsdt2=si_div(si_mul(q2sat,a23m4),si_mul(tempdiff,tempdiff));
  float snowonsi=0.0f;
  if (snowdepth_opt == 1) {
    snowonsi=fmaxf(snowdepth_min,fminf(F(snowsi),snowdepth_max));
    sneqv=si_mul(snowonsi,0.3f);
    F(snowh2d)=snowonsi;
    // MPAS deliberately retains the previously loaded local SNOWH here.
  } else if (snowdepth_opt != 0) { errors[i]=5; return; }
  if (F(snow) > 0.0f) {
    float sfctsno=sfctmp;
    float e2sat=si_mul(611.2f,si_nan_result(mynn_glibc_expf(si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno)))), si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno))), si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno)))));
    float q2sati=si_div(si_mul(0.622f,e2sat),si_sub(sfcprs,e2sat));
    q2sati=si_div(q2sati,si_add(1.0f,q2sati));
    if (t1 > 273.14f) {
      q2sat=si_add(si_mul(q2sat,si_sub(1.0f,F(snowc))),si_mul(q2sati,F(snowc)));
      dqsdt2=si_add(si_mul(dqsdt2,si_sub(1.0f,F(snowc))),si_mul(si_div(si_mul(q2sati,6174.0f),si_mul(sfctsno,sfctsno)),F(snowc)));
    } else {
      q2sat=q2sati;
      dqsdt2=si_div(si_mul(q2sati,6174.0f),si_mul(sfctsno,sfctsno));
    }
    if (t1 > 273.0f && F(snowc) > 0.0f) dqsdt2=si_mul(dqsdt2,si_sub(1.0f,F(snowc)));
  }
  float prcp=si_div(F(rainbl),dt);
  float ffrozp=frpcpn ? F(sr) : (sfctmp <= 273.15f ? 1.0f : 0.0f);
  float tbot=271.36f, z0brd=F(z02d), stc[NSOLD+1];
  for(int k=1;k<=nsoil;k++) stc[k]=a[P_tslb][(k-1)*n+i];
  float ch=F(chs),ribb=F(rib);
  float z0=-1.e36f,eta=-1.e36f,sheat=-1.e36f,eta_kinematic=-1.e36f;
  float fdown=-1.e36f,esnow=-1.e36f,dew=-1.e36f,etp=-1.e36f,ssoil=-1.e36f;
  float flx1=-1.e36f,flx2=-1.e36f,flx3=-1.e36f,snomlt=-1.e36f,runoff1=-1.e36f,q1=-1.e36f;
  float albedok=albedo_opt==2 ? F(albsi) : 0.0f;
  int iloc=i+1,jloc=1;
  // MPAS physics_wrf/module_sf_noah_seaice.F:208-209 records the call indices.
  // Reduce in wrapper loop order, independently of device scheduling.
  atomicMax(last_sflx_column, i);
  sflx_seaice(iloc,jloc,albedo_opt,albedo_default,snowdepth_opt,snowdepth_max,
      snowdepth_min,ffrozp,dt,zlvl,nsoil,sithick,lwdn,solnet,sfcprs,prcp,sfctmp,
      q2,th2,q2sat,dqsdt2,snoalb,tbot,z0brd,z0,emissi,t1,stc,snowh,sneqv,
      albedok,ch,albedosi,snowonsi,eta,sheat,eta_kinematic,fdown,esnow,dew,
      etp,ssoil,flx1,flx2,flx3,snomlt,sncovr,runoff1,q1,ribb,status);
  if(status) { errors[i]=status; return; }
  F(albedo)=albedok; F(emiss)=emissi; F(tsk)=t1; F(z02d)=z0brd;
  F(snowh2d)=snowh; F(snowc)=sncovr; F(snow)=si_mul(sneqv,1000.0f);
  for(int k=1;k<=nsoil;k++) a[P_tslb][(k-1)*n+i]=stc[k];
  F(znt)=z0; F(lh)=eta; F(hfx)=sheat; F(qfx)=eta_kinematic; F(grdflx)=ssoil;
  F(chs2)=F(cqs2);
  if(q1 > F(qsfc)) F(cqs2)=F(chs);
  F(qsfc)=si_div(q1,si_sub(1.0f,q1));
  if(snowdepth_opt==1) F(snowsi)=snowonsi;
  const float rowliw=1000.0f*350000.0f;
  // MPAS module_sf_noah_seaice_drv.F:207-208: retain each rounded divide.
  // NVRTC can replace division by a constant with reciprocal multiplication.
  float fdtliw=si_div(dt,rowliw);
  float fdtw=si_div(dt,(2500000.0f*1000.0f));
  if(optional_noah & 1) F(potevp)=si_add(F(potevp),si_mul(etp,fdtw));
  if(ffrozp > 0.5f) F(acsnow)=si_add(F(acsnow),si_mul(prcp,dt));
  F(acsnom)=si_add(F(acsnom),si_mul(snomlt,1000.0f));
  // MPAS module_sf_noah_seaice_drv.F:483-484 (optional melt accumulator).
  if(optional_noah & 2) F(snopcx)=si_sub(F(snopcx),si_div(snomlt,fdtliw));
  F(sfcrunoff)=si_add(F(sfcrunoff),si_mul(si_mul(runoff1,dt),1000.0f));
  if(optional_noah & 4) {
    float t1sq=si_mul(t1,t1);
    F(noahres)=si_sub(si_sub(si_sub(si_sub(si_sub(si_add(si_sub(si_add(solnet,lwdn),sheat),ssoil),eta),si_mul(si_mul(emissi,5.67051e-8f),si_mul(t1sq,t1sq))),flx1),flx2),flx3);
  }
}
// MPAS physics_wrf/module_sf_noah_seaice.F:208-209;
// physics_wrf/module_sf_noah_seaice_drv.F:211-216,417-431
extern "C" __global__ void seaice_module_state(const int* last_sflx_column,
    int* state, int i_start, int j_index) {
  if (*last_sflx_column >= 0) {
    state[0] = i_start + *last_sflx_column;
    state[1] = j_index;
  }
}
// MPAS mpas_atmphys_driver_seaice.F:160-327 (seaice_from_MPAS)
// MPAS mpas_atmphys_lsm_shared.F:22-57 (correct_tsk_over_seaice)
extern "C" __global__ void seaice_from_MPAS(float** a,int n,float threshold,
    float albedo_default,float snowdepth_min,float thickness_default,
    int microp,int convection) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i>=n) return;
  F(swdown)=si_div(F(gsw),si_sub(1.0f,F(albedo)));
  F(albsi)=albedo_default; F(snowsi)=snowdepth_min; F(icedepth)=thickness_default;
  F(tsk_sea)=F(tsk);
  if(F(xice)>=threshold && F(xice)<=1.0f) {
    F(albedo)=si_div(si_sub(F(albedo),si_mul(0.08f,si_sub(1.0f,F(xice)))),F(xice));
    F(emiss)=si_div(si_sub(F(emiss),si_mul(0.98f,si_sub(1.0f,F(xice)))),F(xice));
    F(tsk_sea)=fmaxf(F(tsk_sea),271.4f);
    if(F(xice)<0.2f && F(tsk)<253.15f) F(tsk)=253.15f;
    if(F(xice)<0.1f && F(tsk)<263.15f) F(tsk)=263.15f;
  }
  F(sr)=0.0f; F(rainbl)=0.0f;
  if(microp) { F(sr)=F(sr_input); F(rainbl)=si_add(F(rainbl),F(rainncv)); }
  if(convection) F(rainbl)=si_add(F(rainbl),F(raincv));
}
// MPAS mpas_atmphys_driver_seaice.F:330-458 (seaice_to_MPAS)
extern "C" __global__ void seaice_to_MPAS(float** a,int n,float threshold) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i>=n) return;
  if(F(xice)>=threshold && F(xice)<=1.0f) {
    float x=F(xice),s=si_sub(1.0f,x);
    F(cpm)=si_add(si_mul(x,F(cpm)),si_mul(s,F(cpm_sea)));
    F(chs)=si_add(si_mul(x,F(chs)),si_mul(s,F(chs_sea)));
    F(chs2)=si_add(si_mul(x,F(chs2)),si_mul(s,F(chs2_sea)));
    F(cqs2)=si_add(si_mul(x,F(cqs2)),si_mul(s,F(cqs2_sea)));
    F(hfx)=si_add(si_mul(x,F(hfx)),si_mul(s,F(hfx_sea)));
    F(lh)=si_add(si_mul(x,F(lh)),si_mul(s,F(lh_sea)));
    F(qfx)=si_add(si_mul(x,F(qfx)),si_mul(s,F(qfx_sea)));
    F(qgh)=si_add(si_mul(x,F(qgh)),si_mul(s,F(qgh_sea)));
    F(qsfc)=si_add(si_mul(x,F(qsfc)),si_mul(s,F(qsfc_sea)));
    F(tsk)=si_add(si_mul(x,F(tsk)),si_mul(s,F(tsk_sea)));
    F(albedo)=si_add(si_mul(x,F(albedo)),si_mul(s,0.08f));
    F(emiss)=si_add(si_mul(x,F(emiss)),si_mul(s,0.98f));
  }
}
"""


def _module(nsoil: int):
    """The sea-ice module loaded on the calling card."""
    import cupy as cp
    # A CUDA module belongs to the card it was loaded on. Same-host ranks drive
    # several cards from one process, so a cache keyed without the card hands
    # the second card the first card's handle (CUDA_ERROR_INVALID_HANDLE at step 1).
    return _module_on(int(cp.cuda.Device().id), nsoil)


def cuda_source(nsoil):
    shared = files("woof.core.kernels").joinpath("mynn_libm.cuh").read_text()
    return f"#define NSOLD {nsoil}\ntypedef float real;\n" + shared + _CORE_SOURCE


@lru_cache(maxsize=None)
def _module_on(device: int, nsoil: int):
    """Compile MPAS physics_wrf/module_sf_noah_seaice.F:36-1292 on the device."""
    # Shared pinned engine glibc float routines; no private exp/pow implementation.
    import cupy as cp
    from ..cuda_backend.aot import direct_image

    source = cuda_source(nsoil)
    # RawModule's cache compiler appends forced FTZ after caller options.
    # Direct NVRTC preserves strict native subnormals (pool proof-bar ruling).
    binary = direct_image(source,
        options=_hex_numerics.options(("--std=c++11", "--fmad=false", "--ftz=false",
                 "--prec-div=true", "--prec-sqrt=true"), keep_ftz=True),
        arch=cp.cuda.Device(device).compute_capability)
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(binary)
    return module


def _validate(arrays, required, n, nsoil):
    """Validate MPAS module_sf_noah_seaice_drv.F:22-115 array dummy storage."""
    import cupy as cp

    for name in required:
        if name not in arrays:
            raise ValueError(f"MPAS seaice requires {name}")
        array = arrays[name]
        if not isinstance(array, cp.ndarray) or array.dtype != cp.float32 or not array.flags.c_contiguous:
            raise ValueError(f"MPAS seaice {name} must be contiguous CUDA float32")
        if name == "tslb":
            shape = (nsoil, n)
        elif name in ("t3d", "qv3d", "p8w3d", "dz8w"):
            if array.ndim != 2 or array.shape[1] != n or array.shape[0] < (2 if name == "p8w3d" else 1):
                raise ValueError(f"MPAS seaice {name} must have levels by cells, with two pressure interfaces")
            continue
        else:
            shape = (n,)
        if array.shape != shape:
            raise ValueError(f"MPAS seaice {name} must have shape {shape}")


def _pointers(arrays, n):
    """Bind MPAS module_sf_noah_seaice_drv.F:22-115 argument pointers."""
    import cupy as cp

    # Host work here consists solely of pointer metadata; all field math stays on CUDA.
    dummy = cp.zeros(n, dtype=cp.float32)
    pointers = cp.asarray(np.asarray([arrays[name].data.ptr if name in arrays else
                                     (0 if name in _SEA_FLUX_FIELDS else dummy.data.ptr)
                                     for name in FIELDS], dtype=np.uint64))
    return pointers, dummy


def launch_seaice_noah(arrays: Mapping, options: SeaiceOptions, *,
                       module_state=None, i_start=1, j_index=1):
    """MPAS physics_wrf/module_sf_noah_seaice_drv.F:22-518; update columns in place."""
    import cupy as cp

    n = arrays["xice"].size
    nsoil = arrays["tslb"].shape[0]
    # HRTICE reads STC(2); prevent the concrete out-of-bounds defect for <2 layers.
    if nsoil < 2:
        raise ValueError("MPAS HRTICE reads layer 2; seaice needs at least two layers")
    required = FIELDS[:38]
    _validate(arrays, [name for name in required if name not in OPTIONAL_FIELDS], n, nsoil)
    _validate(arrays, [name for name in OPTIONAL_FIELDS if name in arrays], n, nsoil)
    pointers, dummy = _pointers(arrays, n)
    errors = cp.zeros(n, dtype=cp.int32)
    # MPAS module_sf_noah_seaice.F:28-30,208-209: caller-owned diagnostic
    # thread state survives skipped columns. It never enters weather math.
    if module_state is None:
        module_state = cp.empty(2, dtype=cp.int32)
    if (not isinstance(module_state, cp.ndarray) or module_state.dtype != cp.int32
            or module_state.shape != (2,) or not module_state.flags.c_contiguous):
        raise ValueError("MPAS seaice module state must be two contiguous CUDA int32 words")
    last_sflx_column = cp.full(1, -1, dtype=cp.int32)
    optional = sum(1 << i for i, name in enumerate(OPTIONAL_FIELDS) if name in arrays)
    args = (pointers, np.int32(n), np.int32(nsoil), np.float32(options.dt),
            np.int32(options.seaice_albedo_opt), np.float32(options.seaice_albedo_default),
            np.int32(options.seaice_thickness_opt), np.float32(options.seaice_thickness_default),
            np.int32(options.seaice_snowdepth_opt), np.float32(options.seaice_snowdepth_max),
            np.float32(options.seaice_snowdepth_min), np.float32(options.xice_threshold),
            np.int32(options.frpcpn), np.int32(optional), errors, last_sflx_column)
    if n:
        _module(nsoil).get_function("seaice_noah")(((n+127)//128,), (128,), args)
        _module(nsoil).get_function("seaice_module_state")((1,), (1,),
            (last_sflx_column, module_state, np.int32(i_start), np.int32(j_index)))
    # Preserve the first-column fatal message. Inside a forecast the first nonzero word is read
    # off the step path (deferred_checks_v842, U-MPASPHYS-1); elsewhere it is read at once.
    if n:
        from .deferred_checks_v842 import defer
        messages = {1: "Physical snow depth is less than snow water equiv.",
                    2: "SEAICE_THICKNESS_OPT=1 requires ICEDEPTH field",
                    3: "SEAICE_THICKNESS_OPT not recognized",
                    4: "SEAICE_ALBEDO_OPT=2 requires ALBSI field",
                    5: "SEAICE_SNOWDEPTH_OPT not recognized"}
        defer(errors, lambda code: "MPAS seaice fatal: " + messages.get(code, f"error code {code}"),
              reduce="first", error=ValueError)
    return arrays


_DIAG_FIELDS = {"acsnom": "acsnom", "acsnow": "acsnow", "br": "rib", "chs": "chs",
                "chs2": "chs2", "cpm": "cpm", "cqs2": "cqs2", "qgh": "qgh", "qsfc": "qsfc",
                "glw": "glw", "gsw": "gsw", "grdflx": "grdflx", "hfx": "hfx", "qfx": "qfx",
                "lh": "lh", "sfc_albedo": "albedo", "sfc_emiss": "emiss", "sfcrunoff": "sfcrunoff",
                "z0": "z02d", "znt": "znt"}
_SFC_FIELDS = {"snoalb": "snoalb2d", "snow": "snow", "snowc": "snowc", "snowh": "snowh2d",
               "skintemp": "tsk", "tslb": "tslb", "xice": "xice"}
_SEA_FLUX_FIELDS = ("cpm_sea", "chs_sea", "chs2_sea", "cqs2_sea", "hfx_sea",
                    "lh_sea", "qfx_sea", "qgh_sea", "qsfc_sea")


def seaice_from_MPAS(diag_physics, sfc_input, shared, options: SeaiceOptions,
                     *, lsm_scheme: str, microp_scheme: str, convection_scheme: str, driver_module=None):
    """MPAS mpas_atmphys_driver_seaice.F:160-327; allocate and fill driver arrays."""
    import cupy as cp

    if lsm_scheme not in LSM_SCHEMES:
        raise ValueError(f"MPAS seaice requires sf_noah or sf_noahmp, got {lsm_scheme}")
    n = sfc_input["xice"].size
    nsoil = sfc_input["tslb"].shape[0]
    work = {dest: diag_physics[src].copy() for src, dest in _DIAG_FIELDS.items()}
    work.update({dest: sfc_input[src].copy() for src, dest in _SFC_FIELDS.items()})
    work.update({name: shared[name] for name in ("t3d", "qv3d", "p8w3d", "dz8w")})
    # MPAS driver_sfclayer.F:150-175 allocates ocean fluxes only for fractional
    # ice. from_MPAS never reads them; defer lazy native views to to_MPAS.
    for name in ("swdown", "albsi", "snowsi", "icedepth", "tsk_sea", "sr", "rainbl"):
        work[name] = cp.empty(n, dtype=cp.float32)
    if microp_scheme != "off":
        work["sr_input"] = diag_physics["sr"]
        work["rainncv"] = diag_physics["rainncv"]
    if convection_scheme != "off":
        work["raincv"] = diag_physics["raincv"]
    if lsm_scheme == "sf_noah":
        work.update({name: diag_physics[name].copy() for name in OPTIONAL_FIELDS})
    _validate(work, list(work), n, nsoil)
    pointers, dummy = _pointers(work, n)
    if n:
        (driver_module or _module(nsoil)).get_function("seaice_from_MPAS")(((n+127)//128,), (128,),
            (pointers, np.int32(n), np.float32(options.xice_threshold),
             np.float32(options.seaice_albedo_default), np.float32(options.seaice_snowdepth_min),
             np.float32(options.seaice_thickness_default), np.int32(microp_scheme != "off"),
             np.int32(convection_scheme != "off")))
    return work


def seaice_to_MPAS(work, diag_physics, sfc_input, options: SeaiceOptions, *, lsm_scheme: str,
                   shared=None, driver_module=None, blend=True):
    """MPAS mpas_atmphys_driver_seaice.F:330-458; reconstruct fractional fluxes."""
    import cupy as cp

    n = work["xice"].size
    nsoil = work["tslb"].shape[0]
    missing = [name for name in _SEA_FLUX_FIELDS if name not in work]
    # Bind the shared ocean flux storage first: a name lookup, no device read. The kernel reads it
    # only in iced cells, so binding it on an ice-free call changes no word. Only when storage is
    # truly absent does the any-ice test (a device sync) decide whether to refuse. Breakage it
    # prevents: U-MPASPHYS-1, that test synchronized the step on every call.
    if missing and shared is not None:
        for name in missing:
            try:
                work[name] = shared[name]
            except KeyError:
                break
        if all(name in work for name in _SEA_FLUX_FIELDS):
            _validate(work, _SEA_FLUX_FIELDS, n, nsoil)
        missing = [name for name in _SEA_FLUX_FIELDS if name not in work]
    if blend and missing and bool(cp.any((work["xice"] >= options.xice_threshold) &
                              (work["xice"] <= 1.0)).item()):
        # Pinned native crashes at driver_seaice.F:390 on the first cpm_sea
        # read when frac=false. Prevent that same unallocated device read,
        # without refusing no-ice calls or supplying invented ocean values.
        raise RuntimeError("MPAS seaice_to_MPAS reads unallocated native ocean flux storage "
                           "at mpas_atmphys_driver_seaice.F:390-398: " + ", ".join(missing))
    pointers, dummy = _pointers(work, n)
    if n and blend:
        (driver_module or _module(nsoil)).get_function("seaice_to_MPAS")(((n+127)//128,), (128,),
            (pointers, np.int32(n), np.float32(options.xice_threshold)))
    diag_out = ("acsnom", "acsnow", "chs", "chs2", "cpm", "cqs2", "qsfc", "qgh",
                "hfx", "qfx", "lh", "sfc_albedo", "sfc_emiss", "sfcrunoff", "z0", "znt", "grdflx")
    for name in diag_out:
        cp.copyto(diag_physics[name], work[_DIAG_FIELDS[name]])
    for name in ("snow", "snowc", "snowh", "skintemp", "tslb"):
        cp.copyto(sfc_input[name], work[_SFC_FIELDS[name]])
    if lsm_scheme == "sf_noah":
        for name in OPTIONAL_FIELDS:
            cp.copyto(diag_physics[name], work[name])


def driver_seaice(diag_physics, sfc_input, shared, options: SeaiceOptions,
                   *, lsm_scheme: str, microp_scheme: str, convection_scheme: str,
                   module_state=None, sea_temperature=None):
    """MPAS mpas_atmphys_driver_seaice.F:461-558; both Noah land-surface choices."""
    work = seaice_from_MPAS(diag_physics, sfc_input, shared, options, lsm_scheme=lsm_scheme,
                            microp_scheme=microp_scheme, convection_scheme=convection_scheme)
    if sea_temperature is not None:
        # ArWen carries the actual open-water SST separately from blended TSK.
        _validate({"tsk_sea": sea_temperature}, ["tsk_sea"], work["xice"].size,
                  work["tslb"].shape[0])
        work["tsk_sea"][...] = sea_temperature
    launch_seaice_noah(work, options, module_state=module_state)
    seaice_to_MPAS(work, diag_physics, sfc_input, options, lsm_scheme=lsm_scheme, shared=shared)
    return work


def options_from_MPAS(dt: float, *, config_frac_seaice: bool) -> SeaiceOptions:
    """MPAS mpas_atmphys_vars.F:637-654 and mpas_atmphys_manager.F:739-744."""
    return SeaiceOptions(dt=dt, xice_threshold=0.02 if config_frac_seaice else 0.5,
                         seaice_albedo_opt=0, seaice_albedo_default=0.65,
                         seaice_thickness_opt=0, seaice_thickness_default=3.0,
                         seaice_snowdepth_opt=0, seaice_snowdepth_max=1.e10,
                         seaice_snowdepth_min=0.001, frpcpn=False)


def launch_slot(call):
    """MPAS mpas_atmphys_driver.F:282-321; f4 immutable SlotCall adapter.

    ``seaice_shared`` supplies the already prepared MPAS shared arrays under the
    wrapper names t3d/qv3d/p8w3d/dz8w plus the *_sea surface-layer fluxes. It has
    no hidden numerical state. ``call.dt`` must be native dt_pbl.
    """
    options = options_from_MPAS(call.dt, config_frac_seaice=call.options["config_frac_seaice"])
    return driver_seaice(call.pools["diag_physics"], call.pools["sfc_input"],
                         call.pools["seaice_shared"], options, lsm_scheme=call.scheme,
                         microp_scheme=call.options["config_microp_scheme"],
                         convection_scheme=call.options["config_convection_scheme"],
                         module_state=call.pools.get("seaice_module_state"))


def register_backend(registry):
    """MPAS mpas_atmphys_driver.F:285-321; register both sea-ice choices."""
    for scheme in LSM_SCHEMES:
        registry.register(SCHEME, scheme, BACKEND, launch_slot)
