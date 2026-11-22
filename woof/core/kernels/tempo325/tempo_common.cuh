// TEMPO microphysics (Thompson-Eidhammer Microphysics Parameterization for
// Operations), port for WOOF.  Shared definitions for the CPU float64
// reference and the GPU kernel.  The same headers compile with nvcc / NVRTC
// (device) and with g++ (host reference).
//
// Source of record (this directory, tempo_version="v3.2.5"):
// https://github.com/NCAR/TEMPO tag tempo_v3.2.5,
// commit c62699fb660ebbb14e4388f7b59a8b38decac1ab (2026-09-25), the latest
// TEMPO release and the WOOF default.  The pinned v3.0.3 port (RRFS v2,
// 9adf9ef) lives unchanged in ../tempo/ and stays selectable by name.
// One upstream defect is given a defined value here: sum_tendencies reads
// final_weight where v3.2.5 leaves it unset (main.F90:1752-1755); this port
// sets final_weight = 1 (no scaling) per level, and the native referee is
// built with the same one-line initialisation, labelled patched.
// Licence: Apache-2.0, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.  See
// tempo325/NOTICE-TEMPO.txt.
//
// Port conventions (keep these when editing):
//  * Everything is double precision, matching TEMPO built with
//    -fdefault-real-8 (wp = dp = 8 bytes), which is the reference build.
//  * Column arrays are 1-based like the Fortran: a[k] for k = 1..nz, element
//    0 unused, storage TEMPO_NZP doubles.  Fallspeed arrays use k = nz+1.
//  * Every scalar, parameter array and lookup table lives in one flat
//    double blob laid out exactly like struct TempoBlob below.  The blob is
//    filled from TEMPO's own tempo_init() state (tools/tempo_ref dump) so the
//    kernel port is checked separately from the table builders.
//  * Access: PS(name) scalar, PI(name) integer, PA1(name,i),
//    PA2(name,i,j), PA3, PA4, PA5 with Fortran 1-based indices.
//  * Fortran ** with an integer literal exponent -> tpowi (libgcc
//    __powidf2 order); real exponent -> pow.  nint -> tnint, int() -> (int).
#pragma once

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define TEMPO_HD __host__ __device__ __forceinline__
#define TEMPO_HDN __host__ __device__
#else
#include <cmath>
#define TEMPO_HD inline
#define TEMPO_HDN
#endif

// ---------------------------------------------------------------------------
// Elementary functions.  TEMPO's Fortran calls the C library for EXP, LOG,
// LOG10 and real ** real.  The host build of these headers does the same.
// The device build calls WOOF's own float64 exp, log and pow
// (kernels/glibc_flt64.cuh) and the log10 below, which give the same words
// as the x86-64 C library the reference Fortran links; CUDA's own exp, log,
// pow and log10 differ from it in the last place on some arguments, and
// those differences grew to 1.9e-10 relative over 90 steps
// (tools/tempo_ref/oracle, docs/TEMPO-PORT.md).  Every TEMPO kernel calls
// tempo_exp / tempo_log / tempo_pow / tempo_log10, never the bare names
// (tests/test_tempo_device_libm.py holds that).
// ---------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include "../glibc_flt64.cuh"
#endif

#ifdef __CUDA_ARCH__
// log10(x) = k*log10(2) + log(m)/ln(10) for x = 2^k * m, with log10(2) split
// into a high part exact in 33 bits and a low part (Sun fdlibm e_log10.c
// algorithm).  The reduced argument m lies in [1, 2) for k >= 0 and in
// [0.5, 1) for k < 0, so k*log10_2hi is exact and cancellation is avoided.
__device__ __forceinline__ double tempo_device_log10(double x) {
  const double two54 = 0x1p54;
  const double ivln10 = 0x1.bcb7b1526e50ep-2;
  const double log10_2hi = 0x1.34413509f6p-2;
  const double log10_2lo = 0x1.9fef311f12b36p-42;
  unsigned long long u = (unsigned long long)__double_as_longlong(x);
  int hx = (int)(u >> 32);
  int k = 0;
  if (hx < 0x00100000) {  // zero, negative, or below 2^-1022 (scaled up by 2^54)
    if (((hx & 0x7fffffff) | (unsigned int)u) == 0) return -two54 / fabs(x);
    if (hx < 0) return (x - x) / (x - x);
    k -= 54;
    x *= two54;
    u = (unsigned long long)__double_as_longlong(x);
    hx = (int)(u >> 32);
  }
  if (hx >= 0x7ff00000) return x + x;
  k += (hx >> 20) - 1023;
  int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
  hx = (hx & 0x000fffff) | ((0x3ff - i) << 20);
  double y = (double)(k + i);
  x = __longlong_as_double((long long)(((unsigned long long)(unsigned int)hx << 32) |
                                       (u & 0xffffffffULL)));
  double z = __dadd_rn(__dmul_rn(y, log10_2lo), __dmul_rn(ivln10, glibc_log(x)));
  return __dadd_rn(z, __dmul_rn(y, log10_2hi));
}
#endif

TEMPO_HD double tempo_exp(double x) {
#ifdef __CUDA_ARCH__
  return glibc_exp(x);
#else
  return exp(x);
#endif
}
TEMPO_HD double tempo_log(double x) {
#ifdef __CUDA_ARCH__
  return glibc_log(x);
#else
  return log(x);
#endif
}
TEMPO_HD double tempo_pow(double x, double y) {
#ifdef __CUDA_ARCH__
  return glibc_pow(x, y);
#else
  return pow(x, y);
#endif
}
TEMPO_HD double tempo_log10(double x) {
#ifdef __CUDA_ARCH__
  return tempo_device_log10(x);
#else
  return log10(x);
#endif
}

#ifndef TEMPO_NZ_MAX
#define TEMPO_NZ_MAX 100
#endif
#define TEMPO_NZP (TEMPO_NZ_MAX + 3)

// ---------------------------------------------------------------------------
// Blob layout.  TX(name, d1, d2, d3, d4, d5); count = d1*d2*d3*d4*d5.
// Integers are stored as doubles.
// ---------------------------------------------------------------------------
#define TEMPO_FIELDS(TX) \
  TX(av_r,1,1,1,1,1) TX(bv_r,1,1,1,1,1) TX(av_s,1,1,1,1,1) TX(bv_s,1,1,1,1,1) \
  TX(fv_s,1,1,1,1,1) TX(bv_c,1,1,1,1,1) TX(bv_i,1,1,1,1,1) TX(av_g_old,1,1,1,1,1) \
  TX(bv_g_old,1,1,1,1,1) TX(av_g_new,1,1,1,1,1) TX(bv_g_new,1,1,1,1,1) TX(fv_r,1,1,1,1,1) \
  TX(av_c,1,1,1,1,1) TX(a_coeff,1,1,1,1,1) TX(b_coeff,1,1,1,1,1) TX(av_i,1,1,1,1,1) \
  TX(am_s,1,1,1,1,1) TX(bm_s,1,1,1,1,1) TX(bm_g,1,1,1,1,1) TX(bm_i,1,1,1,1,1) \
  TX(bm_r,1,1,1,1,1) TX(rho_i,1,1,1,1,1) TX(xm0i,1,1,1,1,1) TX(d0c,1,1,1,1,1) \
  TX(d0r,1,1,1,1,1) TX(d0s,1,1,1,1,1) TX(d0g,1,1,1,1,1) TX(d0r_max,1,1,1,1,1) \
  TX(c_cube,1,1,1,1,1) TX(c_sqrd,1,1,1,1,1) TX(mu_r,1,1,1,1,1) TX(mu_s,1,1,1,1,1) \
  TX(mu_g,1,1,1,1,1) TX(mu_i,1,1,1,1,1) TX(nu_c_scale,1,1,1,1,1) TX(nu_c_max,1,1,1,1,1) \
  TX(nu_c_min,1,1,1,1,1) TX(naccn0,1,1,1,1,1) TX(naccn1,1,1,1,1,1) TX(nain0,1,1,1,1,1) \
  TX(nain1,1,1,1,1,1) TX(nwfa_default,1,1,1,1,1) TX(nifa_default,1,1,1,1,1) TX(aero_max,1,1,1,1,1) \
  TX(hgfrz,1,1,1,1,1) TX(nt_c_o,1,1,1,1,1) TX(nt_c_l,1,1,1,1,1) TX(nt_c_max,1,1,1,1,1) \
  TX(nt_c_min,1,1,1,1,1) TX(tno,1,1,1,1,1) TX(ato,1,1,1,1,1) TX(rho_s,1,1,1,1,1) \
  TX(demott_nuc_ssati,1,1,1,1,1) TX(max_ni,1,1,1,1,1) TX(icenuc_max,1,1,1,1,1) \
  TX(rime_threshold,1,1,1,1,1) TX(rime_conversion,1,1,1,1,1) TX(ef_si,1,1,1,1,1) \
  TX(ef_rs,1,1,1,1,1) TX(ef_rg,1,1,1,1,1) TX(ef_ri,1,1,1,1,1) TX(autocon_nr_factor,1,1,1,1,1) \
  TX(timestep_conversion_rime_to_rain,1,1,1,1,1) TX(nrhg,1,1,1,1,1) TX(nrhg1,1,1,1,1,1) \
  TX(min_qv,1,1,1,1,1) TX(r1,1,1,1,1,1) TX(r2,1,1,1,1,1) TX(low_limit_mass_for_precip,1,1,1,1,1) \
  TX(eps,1,1,1,1,1) TX(meters3_to_liters,1,1,1,1,1) TX(gonv_min,1,1,1,1,1) TX(gonv_max,1,1,1,1,1) \
  TX(t0,1,1,1,1,1) TX(rho_w,1,1,1,1,1) TX(sc,1,1,1,1,1) TX(pi,1,1,1,1,1) TX(lsub,1,1,1,1,1) \
  TX(lvap0,1,1,1,1,1) TX(rv,1,1,1,1,1) TX(rdry,1,1,1,1,1) TX(roverrv,1,1,1,1,1) TX(r,1,1,1,1,1) \
  TX(rho_not,1,1,1,1,1) TX(rho_not0,1,1,1,1,1) TX(cp,1,1,1,1,1) TX(r_uni,1,1,1,1,1) \
  TX(kap0,1,1,1,1,1) TX(kap1,1,1,1,1,1) TX(lam0,1,1,1,1,1) TX(lam1,1,1,1,1,1) \
  TX(nbins,1,1,1,1,1) TX(nbc,1,1,1,1,1) TX(nbr,1,1,1,1,1) TX(nbs,1,1,1,1,1) TX(nbi,1,1,1,1,1) \
  TX(nbg,1,1,1,1,1) TX(ntb_i,1,1,1,1,1) TX(ntb_i1,1,1,1,1,1) TX(ntb_c,1,1,1,1,1) \
  TX(ntb_t,1,1,1,1,1) TX(ntb_g1,1,1,1,1,1) TX(ntb_s,1,1,1,1,1) TX(ntb_g,1,1,1,1,1) \
  TX(ntb_r,1,1,1,1,1) TX(ntb_r1,1,1,1,1,1) TX(ntb_t1,1,1,1,1,1) TX(ntb_in,1,1,1,1,1) \
  TX(ntb_arc,1,1,1,1,1) TX(ntb_arw,1,1,1,1,1) TX(ntb_art,1,1,1,1,1) TX(ntb_arr,1,1,1,1,1) \
  TX(ntb_ark,1,1,1,1,1) TX(idx_bg1,1,1,1,1,1) TX(nhbins,1,1,1,1,1) TX(radar_bins,1,1,1,1,1) \
  TX(rho_g,9,1,1,1,1) TX(r_c,37,1,1,1,1) TX(r_i,64,1,1,1,1) TX(r_r,37,1,1,1,1) \
  TX(r_s,37,1,1,1,1) TX(r_g,37,1,1,1,1) TX(n0r_exp,37,1,1,1,1) TX(n0g_exp,37,1,1,1,1) \
  TX(nt_i,55,1,1,1,1) TX(nt_in,55,1,1,1,1) TX(ta_na,7,1,1,1,1) TX(ta_ww,9,1,1,1,1) \
  TX(ta_tk,7,1,1,1,1) TX(ta_ra,5,1,1,1,1) TX(ta_ka,4,1,1,1,1) TX(sa,10,1,1,1,1) \
  TX(sb,10,1,1,1,1) TX(tc,9,1,1,1,1) \
  TX(dim_nrhg,1,1,1,1,1) TX(av_g,9,1,1,1,1) TX(bv_g,9,1,1,1,1) TX(am_i,1,1,1,1,1) \
  TX(am_r,1,1,1,1,1) TX(am_g,9,1,1,1,1) TX(lfus,1,1,1,1,1) TX(olfus,1,1,1,1,1) \
  TX(orv,1,1,1,1,1) TX(ar_volume,1,1,1,1,1) TX(sc3,1,1,1,1,1) TX(d0i,1,1,1,1,1) \
  TX(xm0s,1,1,1,1,1) TX(xm0g,1,1,1,1,1) TX(obmi,1,1,1,1,1) TX(obmr,1,1,1,1,1) \
  TX(oams,1,1,1,1,1) TX(obms,1,1,1,1,1) TX(ocms,1,1,1,1,1) TX(oamg,9,1,1,1,1) \
  TX(ocmg,9,1,1,1,1) TX(obmg,1,1,1,1,1) TX(cce,5,15,1,1,1) TX(ccg,5,15,1,1,1) \
  TX(ocg1,15,1,1,1,1) TX(ocg2,15,1,1,1,1) TX(cie,7,1,1,1,1) TX(cig,7,1,1,1,1) \
  TX(oig1,1,1,1,1,1) TX(oig2,1,1,1,1,1) TX(cre,13,1,1,1,1) TX(crg,13,1,1,1,1) \
  TX(ore1,1,1,1,1,1) TX(org1,1,1,1,1,1) TX(org2,1,1,1,1,1) TX(org3,1,1,1,1,1) \
  TX(cse,17,1,1,1,1) TX(csg,17,1,1,1,1) TX(cge,12,9,1,1,1) TX(cgg,12,9,1,1,1) \
  TX(oge1,1,1,1,1,1) TX(ogg1,1,1,1,1,1) TX(ogg2,1,1,1,1,1) TX(ogg3,1,1,1,1,1) \
  TX(t1_qr_qc,1,1,1,1,1) TX(t1_qr_qi,1,1,1,1,1) TX(t2_qr_qi,1,1,1,1,1) \
  TX(t1_qs_qc,1,1,1,1,1) TX(t1_qs_qi,1,1,1,1,1) TX(t1_qr_ev,1,1,1,1,1) TX(t2_qr_ev,1,1,1,1,1) \
  TX(t1_qs_sd,1,1,1,1,1) TX(t2_qs_sd,1,1,1,1,1) TX(t1_qs_me,1,1,1,1,1) TX(t2_qs_me,1,1,1,1,1) \
  TX(t1_qg_sd,1,1,1,1,1) TX(t1_qg_me,1,1,1,1,1) \
  TX(nic2,1,1,1,1,1) TX(nii2,1,1,1,1,1) TX(nii3,1,1,1,1,1) TX(nir2,1,1,1,1,1) \
  TX(nir3,1,1,1,1,1) TX(nis2,1,1,1,1,1) TX(nig2,1,1,1,1,1) TX(nig3,1,1,1,1,1) \
  TX(niin2,1,1,1,1,1) TX(nic1,1,1,1,1,1) \
  TX(dc,100,1,1,1,1) TX(dtc,100,1,1,1,1) TX(di,100,1,1,1,1) TX(dti,100,1,1,1,1) \
  TX(dr,100,1,1,1,1) TX(dtr,100,1,1,1,1) TX(ds,100,1,1,1,1) TX(dts,100,1,1,1,1) \
  TX(dg,100,1,1,1,1) TX(dtg,100,1,1,1,1) TX(t_nc,100,1,1,1,1) \
  TX(hbins,50,1,1,1,1) TX(dhbins,50,1,1,1,1) TX(sbins_radar,50,1,1,1,1) \
  TX(dsbins_radar,50,1,1,1,1) TX(gbins_radar,50,1,1,1,1) TX(dgbins_radar,50,1,1,1,1) \
  TX(t_efrw,100,100,1,1,1) TX(t_efsw,100,100,1,1,1) \
  TX(tpc_wev,100,37,100,1,1) TX(tnc_wev,100,37,100,1,1) \
  TX(tnccn_act,7,9,7,5,4) \
  TX(tps_iaus,64,55,1,1,1) TX(tni_iaus,64,55,1,1,1) TX(tpi_ide,64,55,1,1,1) \
  TX(tpi_qcfz,37,100,45,55,1) TX(tni_qcfz,37,100,45,55,1) \
  TX(tpi_qrfz,37,37,45,55,1) TX(tpg_qrfz,37,37,45,55,1) \
  TX(tni_qrfz,37,37,45,55,1) TX(tnr_qrfz,37,37,45,55,1) \
  TX(tcs_racs1,37,9,37,37,1) TX(tmr_racs1,37,9,37,37,1) TX(tcs_racs2,37,9,37,37,1) \
  TX(tmr_racs2,37,9,37,37,1) TX(tcr_sacr1,37,9,37,37,1) TX(tms_sacr1,37,9,37,37,1) \
  TX(tcr_sacr2,37,9,37,37,1) TX(tms_sacr2,37,9,37,37,1) TX(tnr_racs1,37,9,37,37,1) \
  TX(tnr_racs2,37,9,37,37,1) TX(tnr_sacr1,37,9,37,37,1) TX(tnr_sacr2,37,9,37,37,1) \
  TX(tcg_racg,37,37,9,37,37) TX(tmr_racg,37,37,9,37,37) TX(tcr_gacr,37,37,9,37,37) \
  TX(tnr_racg,37,37,9,37,37) TX(tnr_gacr,37,37,9,37,37)

struct TempoBlob {
#define TEMPO_TX_MEMBER(name, d1, d2, d3, d4, d5) double name[(d1) * (d2) * (d3) * (d4) * (d5)];
  TEMPO_FIELDS(TEMPO_TX_MEMBER)
#undef TEMPO_TX_MEMBER
};

// Leading dimensions per field, for the multi-dimensional accessors.
#define TEMPO_TX_DIMS(name, d1, d2, d3, d4, d5) \
  enum : long long { TD1_##name = (d1), TD2_##name = (d2), TD3_##name = (d3), TD4_##name = (d4) };
TEMPO_FIELDS(TEMPO_TX_DIMS)
#undef TEMPO_TX_DIMS

struct TempoParams {
  const TempoBlob* L;
};

#define PS(name) (P.L->name[0])
#define PI(name) ((int)(P.L->name[0]))
#define PA1(name, i) (P.L->name[(long long)(i) - 1])
#define PA2(name, i, j) (P.L->name[((long long)(i) - 1) + TD1_##name * ((long long)(j) - 1)])
#define PA3(name, i, j, k) (P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * ((long long)(k) - 1))])
#define PA4(name, i, j, k, l) (P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * ((long long)(l) - 1)))])
#define PA5(name, i, j, k, l, m) (P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * (((long long)(l) - 1) + \
    TD4_##name * ((long long)(m) - 1))))])

// Per-call context: the Fortran module variables global_dt / global_inverse_dt
// and the configuration flags (module_mp_tempo_cfgs::ty_tempo_cfgs).
struct TempoCtx {
  double global_dt;
  double global_inverse_dt;
  int aerosolaware_flag;
  int hailaware_flag;
  int semi_sedi_flag;
  int refl10cm_from_melting_flag;
  int turn_off_micro_flag;
  int cloud_condensation_flag;
  int refl10cm_flag;
  int re_cloud_flag;
  int re_ice_flag;
  int re_snow_flag;
  int max_hail_diameter_flag;
  int first_call_main;  // v3.2.5: tempo_run is_first_step (itimestep == 1), every column
  int hailhyperaware_flag;       // ty_tempo_cfgs%hailhyperaware_flag (v3.2.x, default on)
  int diagnostic_aerosols_flag;  // ty_tempo_cfgs%diagnostic_aerosols_flag (default off)
};

// ty_tend: process rates, real(dp).
#define TEMPO_TEND_FIELDS(TT) \
  TT(prr_wau) TT(pnr_wau) TT(pnc_wau) TT(prr_rcw) TT(pnc_rcw) TT(pnr_rcr) \
  TT(prs_scw) TT(pnc_scw) TT(png_scw) TT(pbg_scw) TT(prg_gcw) TT(pnc_gcw) TT(pbg_gcw) \
  TT(pri_ihm) TT(pni_ihm) TT(prs_ihm) TT(prg_ihm) TT(prg_scw) \
  TT(prr_rcs) TT(pnr_rcs) TT(prg_rcs) TT(png_rcs) TT(prs_rcs) TT(pbg_rcs) \
  TT(prr_rcg) TT(pnr_rcg) TT(prg_rcg) TT(png_rcg) TT(pbg_rcg) \
  TT(pri_inu) TT(pni_inu) TT(pri_iha) TT(pni_iha) \
  TT(pri_wfz) TT(pni_wfz) \
  TT(prg_rfz) TT(png_rfz) TT(pnr_rfz) TT(pri_rfz) TT(pni_rfz) TT(pbg_rfz) \
  TT(prs_sde) TT(pri_ide) TT(pni_ide) TT(prs_ide) TT(prg_gde) TT(png_gde) \
  TT(pni_iau) TT(prs_iau) \
  TT(prr_sml) TT(prr_gml) TT(pbg_sml) TT(pbg_gml) TT(pnr_sml) TT(pnr_gml) \
  TT(prr_rci) TT(pnr_rci) TT(pri_rci) TT(pni_rci) TT(prg_rci) TT(png_rci) TT(pbg_rci) \
  TT(pni_sci) TT(prs_sci) \
  TT(prw_vcd) TT(pnc_wcd) TT(prv_rev) TT(pnr_rev) \
  TT(pna_rca) TT(pna_sca) TT(pna_gca) TT(pnd_rcd) TT(pnd_scd) TT(pnd_gcd) \
  TT(prr_hml) TT(pnr_hml) TT(prh_hcw) TT(pnc_hcw) TT(prh_rch) TT(prr_rch) TT(pnr_rch) TT(prh_hde)

struct TempoTend {
#define TEMPO_TT_MEMBER(name) double name[TEMPO_NZP];
  TEMPO_TEND_FIELDS(TEMPO_TT_MEMBER)
#undef TEMPO_TT_MEMBER
};

// ---------------------------------------------------------------------------
// Fortran intrinsic helpers
// ---------------------------------------------------------------------------
TEMPO_HD double tmax(double a, double b) { return a > b ? a : b; }
TEMPO_HD double tmin(double a, double b) { return a < b ? a : b; }
TEMPO_HD double tmax3(double a, double b, double c) { return tmax(tmax(a, b), c); }
TEMPO_HD double tmin3(double a, double b, double c) { return tmin(tmin(a, b), c); }
TEMPO_HD int imax(int a, int b) { return a > b ? a : b; }
TEMPO_HD int imin(int a, int b) { return a < b ? a : b; }
// Fortran NINT: nearest integer, halfway cases away from zero (gfortran calls
// lround).  x - trunc(x) is exact, so the halfway test sees x itself.  The
// earlier floor(x + 0.5) rounded 0.49999999999999994 up to 1 because x + 0.5
// rounds to 1.0 before the floor; graupel at the 50 kg m-3 density bound
// computes exactly that ratio in the density index nint(10 rg / rb) and took
// the next density class (tools/tempo_ref/oracle, tests/test_tempo_nint.py).
TEMPO_HD int tnint(double x) {
  double t = trunc(x);
  double f = x - t;
  if (f >= 0.5) t += 1.0;
  else if (f <= -0.5) t -= 1.0;
  return (int)t;
}
// Fortran SIGN(a, b)
TEMPO_HD double tsign(double a, double b) { return copysign(fabs(a), b); }
// libgcc __powidf2 order for real ** integer
TEMPO_HD double tpowi(double x, int m) {
  unsigned int n = m < 0 ? (unsigned int)(-m) : (unsigned int)m;
  double y = (n % 2) ? x : 1.0;
  while (n >>= 1) {
    x = x * x;
    if (n % 2) y = y * x;
  }
  return m < 0 ? 1.0 / y : y;
}

// ---------------------------------------------------------------------------
// module_mp_tempo_utils (column-callable parts)
// ---------------------------------------------------------------------------
TEMPO_HD double calc_rslf(double p, double t) {
  const double c0 = .611583699E03, c1 = .444606896E02, c2 = .143177157e01;
  const double c3 = .264224321e-1, c4 = .299291081e-3, c5 = .203154182e-5;
  const double c6 = .702620698e-8, c7 = .379534310e-11, c8 = -.321582393e-13;
  double x = tmax(-80.0, t - 273.16);
  double esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
  esl = tmin(esl, p * 0.15);
  return .622 * esl / (p - esl);
}

TEMPO_HD double calc_rsif(double p, double t) {
  const double c0 = .609868993e03, c1 = .499320233e02, c2 = .184672631e01;
  const double c3 = .402737184e-1, c4 = .565392987e-3, c5 = .521693933e-5;
  const double c6 = .307839583e-7, c7 = .105785160e-9, c8 = .161444444e-12;
  double x = tmax(-80.0, t - 273.16);
  double esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
  esi = tmin(esi, p * 0.15);
  return .622 * esi / tmax(1.e-4, (p - esi));
}

TEMPO_HD int get_nuc(const TempoParams& P, double nc) {
  if ((PS(nu_c_scale) / nc) >= 12.5) return 15;
  return imin(15, tnint(PS(nu_c_scale) / nc) + 2);
}

TEMPO_HD double calc_gamma_series(double a, double x) {
  const int it_max = 100;
  const double smallvalue = 1.e-7;
  double ap1 = a;
  double sum_term = 1.0 / tgamma(ap1 + 1.0);
  double sum = sum_term;
  for (int k = 1; k <= it_max; ++k) {
    ap1 = ap1 + 1.0;
    sum_term = sum_term * x / ap1;
    sum = sum + sum_term;
    if (fabs(sum_term) < (fabs(sum) * smallvalue)) break;
  }
  return sum * tempo_pow(x, a) * tempo_exp(-x);
}

TEMPO_HD double calc_gamma_cf(double a, double x) {
  const int it_max = 100;
  const double smallvalue = 1.e-7;
  const double offset = 1.e-30;
  double b = 1.0 - a + x;
  double d = 1.0 / b;
  double h0 = offset;
  double c = b + (1.0 / offset);
  double delta = c * d;
  double h = h0 * delta;
  for (int k = 1; k <= it_max; ++k) {
    double aj = k * (a - k);
    b = b + 2.0;
    d = b + aj * d;
    if (fabs(d) < offset) d = offset;
    c = b + aj / c;
    if (fabs(c) < offset) c = offset;
    d = 1.0 / d;
    delta = c * d;
    h = h * delta;
    if (fabs(delta - 1.0) < smallvalue) break;
  }
  return tempo_exp(-x + a * tempo_log(x)) * h / tgamma(a);
}

TEMPO_HD double calc_gamma_p(double a, double x) {
  if (x < (a + 1.0)) return calc_gamma_series(a, x);
  return 1.0 - calc_gamma_cf(a, x);
}

// snow_moments: optional outputs are nullptr when absent.  As in the
// Fortran, smoz reads the smo2 OUTPUT (callers always pass smo2 with smoz).
TEMPO_HD void snow_moments(const TempoParams& P, double rs, double tc,
                           double* smob, double* smoc, double* ns, double* smo0,
                           double* smo1, double* smo2, double* smoe, double* smof,
                           double* smog, double* smoz) {
  const double bm_s = PS(bm_s);
  double loga_, a_, b_, smo2_, m0, mrat, slam1, slam2, cc;
#define SA(i) PA1(sa, i)
#define SB(i) PA1(sb, i)
  *smob = rs * PS(oams);
  if (bm_s > 2.0 - 1.e-3 && bm_s < 2.0 + 1.e-3) {
    loga_ = SA(1) + SA(2) * tc + SA(3) * bm_s + SA(4) * tc * bm_s + SA(5) * tc * tc
          + SA(6) * bm_s * bm_s + SA(7) * tc * tc * bm_s + SA(8) * tc * bm_s * bm_s
          + SA(9) * tc * tc * tc + SA(10) * bm_s * bm_s * bm_s;
    a_ = tempo_pow(10.0, loga_);
    b_ = SB(1) + SB(2) * tc + SB(3) * bm_s + SB(4) * tc * bm_s + SB(5) * tc * tc
       + SB(6) * bm_s * bm_s + SB(7) * tc * tc * bm_s + SB(8) * tc * bm_s * bm_s
       + SB(9) * tc * tc * tc + SB(10) * bm_s * bm_s * bm_s;
    smo2_ = tempo_pow((*smob / a_), (1.0 / b_));
  } else {
    smo2_ = *smob;
  }
  if (smo2) *smo2 = smo2_;

#define TEMPO_SNOW_AB(cc)                                                              \
  loga_ = SA(1) + SA(2) * tc + SA(3) * (cc) + SA(4) * tc * (cc) + SA(5) * tc * tc        \
        + SA(6) * (cc) * (cc) + SA(7) * tc * tc * (cc) + SA(8) * tc * (cc) * (cc)        \
        + SA(9) * tc * tc * tc + SA(10) * (cc) * (cc) * (cc);                            \
  a_ = tempo_pow(10.0, loga_);                                                                 \
  b_ = SB(1) + SB(2) * tc + SB(3) * (cc) + SB(4) * tc * (cc) + SB(5) * tc * tc           \
     + SB(6) * (cc) * (cc) + SB(7) * tc * tc * (cc) + SB(8) * tc * (cc) * (cc)           \
     + SB(9) * tc * tc * tc + SB(10) * (cc) * (cc) * (cc);

  cc = PA1(cse, 1);
  TEMPO_SNOW_AB(cc)
  *smoc = a_ * tempo_pow(smo2_, b_);

  loga_ = SA(1) + SA(2) * tc + SA(5) * tc * tc + SA(9) * tc * tc * tc;
  a_ = tempo_pow(10.0, loga_);
  b_ = SB(1) + SB(2) * tc + SB(5) * tc * tc + SB(9) * tc * tc * tc;
  if (smo0) *smo0 = a_ * tempo_pow(smo2_, b_);

  loga_ = SA(1) + SA(2) * tc + SA(3) + SA(4) * tc + SA(5) * tc * tc + SA(6)
        + SA(7) * tc * tc + SA(8) * tc + SA(9) * tc * tc * tc + SA(10);
  a_ = tempo_pow(10.0, loga_);
  b_ = SB(1) + SB(2) * tc + SB(3) + SB(4) * tc + SB(5) * tc * tc + SB(6)
     + SB(7) * tc * tc + SB(8) * tc + SB(9) * tc * tc * tc + SB(10);
  if (smo1) *smo1 = a_ * tempo_pow(smo2_, b_);

  m0 = *smob / *smoc;
  mrat = *smob * m0 * m0 * m0;
  slam1 = m0 * PS(lam0);
  slam2 = m0 * PS(lam1);
  if (ns) {
    *ns = mrat * PS(kap0) / slam1
        + mrat * PS(kap1) * tempo_pow(m0, PS(mu_s)) * PA1(csg, 15) / tempo_pow(slam2, PA1(cse, 15));
  }

  cc = PA1(cse, 13);
  TEMPO_SNOW_AB(cc)
  if (smoe) *smoe = a_ * tempo_pow(smo2_, b_);

  cc = PA1(cse, 16);
  TEMPO_SNOW_AB(cc)
  if (smof) *smof = a_ * tempo_pow(smo2_, b_);

  cc = PA1(cse, 17);
  TEMPO_SNOW_AB(cc)
  if (smog) *smog = a_ * tempo_pow(smo2_, b_);

  cc = PA1(cse, 3);
  TEMPO_SNOW_AB(cc)
  if (smoz) *smoz = a_ * tempo_pow(*smo2, b_);
#undef TEMPO_SNOW_AB
#undef SA
#undef SB
}
