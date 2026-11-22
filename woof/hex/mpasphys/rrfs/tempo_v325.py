"""TEMPO v3.2.5 mixed-precision CUDA column backend: the default TEMPO of the RRFS microphysics slot.

Source authority: NCAR/TEMPO tag tempo_v3.2.5, commit c62699fb660e (Apache-2.0), the latest TEMPO release
(2026-10-06); the RRFS fork moved to v3.2.4 at ufs-community/MPAS-Model de016e0ff4 and v3.2.5 differs from it
only by the default-initialised precipitation accumulators (#56). Selected by ``config_tempo_version``
(&physics_mp_tempo, default ``v3.2.5``). ``v3.0.3`` keeps ``tempo_v303`` (RRFS 44c0849, TEMPO 9adf9ef), the
pinned referee every bitwise RRFS claim and the landing gate are proven against; with ``v3.0.3`` no word of
this module is compiled or run.

This module shares the slot's pools, carriers, binding and native-initialized tables with tempo_v303: the
TEMPO lookup tables and every tempo_init word are the same at both tags (the table builder changed only
verbosity, configuration plumbing and messages), except ``roverrv``, which v3.2.5 derives in tempo_init
(params:451, rdry / rv); the column computes it from rdry and rv and never reads the v3.0.3 blob slot.

What the column changes from v3.0.3 (every line cites tempo_v3.2.5 src/module_mp_tempo_main.F90 unless named):
fall speeds fixed through the rain and graupel sedimentation substeps (:672-745); cloud condensation CYCLEs at
RH = 100 % (:2303); the snow riming boost slope 0.014 (:2712); graupel melting ventilation 0.28 (:3455); one
net-flux sedimentation update (:1806-1813); roverrv = rdry / rv; without aerosols present, the condensation
number from ncsave or nt_c_l (:2316-2325) and the DeMott/Koop paths off; the first-step initialisation for every
column (is_first_step, driver:404, :345-410); Koop's delta_aw from supersaturation (:2488); the graupel volume
of rain-ice collection with meters3_to_liters (:3340); frozen-drop volumes at rho_g(nrhg) (:1661,3032,3127);
graupel deposition and splintering volume at rho_g(1) (:1735); melting-graupel volume bounds and a logistic
number loss (:3464-3473); the rain-snow graupel number from racs1 + sacr1 (:3004-3006); the frozen-drop graupel
number cut by the rime-to-autoconversion ratio (:3112-3122, 3326-3334); depletion rain-snow reset at T >= T0
(:1664); hail-hyperaware (``config_tempo_hailhyperaware``, default on; off with hail-aware off, driver:85): a
diagnosed hail fraction splits graupel into hail with its own size distribution, fall speed, sedimentation,
riming, rain collection, sublimation, melting, depletion, reflectivity and precipitation, merged back into
graupel at the end of the call (:425-445, 913-937, 979-1040, 1410-1448, 2056-2093); hail in the reflectivity
and a 0.75e-22 Ze floor (diags:80-125); ice and ice-number sedimentation every substep (:797-802); the
nucleation max_ni test on snow number ns (:3157); sublimation of melting snow with c_sqrd (:3440); freezing
heat of rain-snow and rain-graupel collection from the rain frozen (:1763-1768); the ice-friendly aerosol sink
of nucleated ice (:515); frozen-drop density weighting of the graupel volume (:1737-1757); the graupel fall
speed coefficient from rho_not (:2121-2123); graupel cloud-collection efficiency above freezing
10^(-0.33 (T - T0)) clipped to [0.1, 1] (:2776); max_ni in binary32 (params:116).

The native code reads ``final_weight`` (:1688) at :1755 even where :1752 did not set it (undefined or stale in
native). Here it is 1 there (no scaling); the native referee this module is replayed against carries the same
definition (``final_weight = 1._wp`` before :1752, labelled patched). WP = MPAS RKIND (binary32), DP binary64.
"""
# Derived from NCAR/TEMPO tempo_v3.2.5 c62699f, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Provided AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
from __future__ import annotations
from functools import lru_cache
import numpy as np
from woof.hex import numerics as _hex_numerics
from woof.hex.mpasphys.rrfs import tempo_v303 as _base

VERSION = "v3.2.5"
TEMPO_TAG = "tempo_v3.2.5"
TEMPO_COMMIT = "c62699fb660e"
SCHEME = _base.SCHEME
BACKEND = _base.BACKEND
COMPILE_OPTIONS = _base.COMPILE_OPTIONS
STATE_FIELDS = _base.STATE_FIELDS
TABLE_LAYOUT = _base.TABLE_LAYOUT
#: aerosolaware, hailaware, ml_for_bl_nc, refl10cm_from_melting, hailhyperaware.
FLAG_COUNT = 5
_mpas_source = _base._mpas_source

_COLUMN = r'''
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_3F6E147B_DEFINED
#define HEX_DIVISOR_3F6E147B_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_3f6e147b = 0.93f;
#else
static const float hex_divisor_3f6e147b = 0.93f;
#endif
#endif
#ifndef HEX_DIVISOR_4485C000_DEFINED
#define HEX_DIVISOR_4485C000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_4485c000 = 1070.0f;
#else
static const float hex_divisor_4485c000 = 1070.0f;
#endif
#endif
#ifndef HEX_DIVISOR_3E800000_DEFINED
#define HEX_DIVISOR_3E800000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_3e800000 = 0.25f;
#else
static const float hex_divisor_3e800000 = 0.25f;
#endif
#endif
#ifndef HEX_DIVISOR_40A00000_DEFINED
#define HEX_DIVISOR_40A00000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40a00000 = 5.0f;
#else
static const float hex_divisor_40a00000 = 5.0f;
#endif
#endif
#ifndef HEX_DIVISOR_40C00000_DEFINED
#define HEX_DIVISOR_40C00000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40c00000 = 6.0f;
#else
static const float hex_divisor_40c00000 = 6.0f;
#endif
#endif
#ifndef HEX_DIVISOR_41400000_DEFINED
#define HEX_DIVISOR_41400000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_41400000 = 12.0f;
#else
static const float hex_divisor_41400000 = 12.0f;
#endif
#endif
#ifndef HEX_DIVISOR_41800000_DEFINED
#define HEX_DIVISOR_41800000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_41800000 = 16.0f;
#else
static const float hex_divisor_41800000 = 16.0f;
#endif
#endif
#ifndef HEX_DIVISOR_41A00000_DEFINED
#define HEX_DIVISOR_41A00000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_41a00000 = 20.0f;
#else
static const float hex_divisor_41a00000 = 20.0f;
#endif
#endif
#ifndef HEX_DIVISOR_44610000_DEFINED
#define HEX_DIVISOR_44610000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_44610000 = 900.0f;
#else
static const float hex_divisor_44610000 = 900.0f;
#endif
#endif
#ifndef HEX_DIVISOR_447A0000_DEFINED
#define HEX_DIVISOR_447A0000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_447a0000 = 1000.0f;
#else
static const float hex_divisor_447a0000 = 1000.0f;
#endif
#endif

#define TEMPO_HD __device__ __forceinline__
#define TEMPO_HDN __device__
#ifndef TEMPO_NZ_MAX
#define TEMPO_NZ_MAX 100
#endif
#define TEMPO_NZP (TEMPO_NZ_MAX + 3)

// ---------------------------------------------------------------------------
// Execution shape.  TEMPO_WARP 0 (default): one thread runs one column, every
// loop below is the plain serial transcription.  TEMPO_WARP 1: one 32-lane
// warp runs one column and level k always lives on lane (k-1)%32, so the
// level-local loops stride the lanes with no synchronisation.  Every step that
// reads another level (fall-speed copy chains, the ktop/substep maxima, the
// cloud-condensation RETURN, sedimentation fluxes from above, the aerosol and
// cloud-fall prefix heights, the melting level, the first-call sums) is
// marked "warp:" at its site, synchronises the warp and is resolved exactly,
// so both shapes evaluate the same statements on the same operands.
// ---------------------------------------------------------------------------
#ifndef TEMPO_WARP
#define TEMPO_WARP 0
#endif
#if TEMPO_WARP
#define TEMPO_LANE ((int)(threadIdx.x & 31u))
#define TEMPO_STRIDE 32
#define TEMPO_SYNC() __syncwarp()
#define TEMPO_FULL 0xffffffffu
#else
#define TEMPO_LANE 0
#define TEMPO_STRIDE 1
#define TEMPO_SYNC() ((void)0)
#endif
#define TEMPO_LEAD (TEMPO_LANE == 0)
// Levels 1..n, this lane's share (serial order in the column shape).
#define TEMPO_LEVELS(k, n) for (int k = 1 + TEMPO_LANE; k <= (n); k += TEMPO_STRIDE)
// A top-down level loop whose iterations are independent of each other.
#if TEMPO_WARP
#define TEMPO_LEVELS_DOWN(k, n) TEMPO_LEVELS(k, n)
#else
#define TEMPO_LEVELS_DOWN(k, n) for (int k = (n); k >= 1; --k)
#endif
// Workspace slots lo..hi-1 (any lane; a TEMPO_SYNC follows before levels are used).
#define TEMPO_SLOTS(k, lo, hi) for (int k = (lo) + TEMPO_LANE; k < (hi); k += TEMPO_STRIDE)

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

using TK_av_r = float;
using TK_bv_r = float;
using TK_av_s = float;
using TK_bv_s = float;
using TK_fv_s = float;
using TK_bv_c = float;
using TK_bv_i = float;
using TK_av_g_old = float;
using TK_bv_g_old = float;
using TK_av_g_new = float;
using TK_bv_g_new = float;
using TK_fv_r = float;
using TK_av_c = float;
using TK_a_coeff = float;
using TK_b_coeff = float;
using TK_av_i = float;
using TK_am_s = float;
using TK_bm_s = float;
using TK_bm_g = float;
using TK_bm_i = float;
using TK_bm_r = float;
using TK_rho_i = float;
using TK_xm0i = float;
using TK_d0c = float;
using TK_d0r = float;
using TK_d0s = float;
using TK_d0g = float;
using TK_d0r_max = float;
using TK_c_cube = float;
using TK_c_sqrd = float;
using TK_mu_r = float;
using TK_mu_s = float;
using TK_mu_g = float;
using TK_mu_i = float;
using TK_nu_c_scale = float;
using TK_nu_c_max = float;
using TK_nu_c_min = float;
using TK_naccn0 = float;
using TK_naccn1 = float;
using TK_nain0 = float;
using TK_nain1 = float;
using TK_nwfa_default = float;
using TK_nifa_default = float;
using TK_aero_max = float;
using TK_hgfrz = float;
using TK_nt_c_o = float;
using TK_nt_c_l = float;
using TK_nt_c_max = float;
using TK_nt_c_min = float;
using TK_tno = float;
using TK_ato = float;
using TK_rho_s = float;
using TK_demott_nuc_ssati = float;
using TK_max_ni = float;
using TK_icenuc_max = float;
using TK_rime_threshold = float;
using TK_rime_conversion = float;
using TK_ef_si = float;
using TK_ef_rs = float;
using TK_ef_rg = float;
using TK_ef_ri = float;
using TK_autocon_nr_factor = float;
using TK_timestep_conversion_rime_to_rain = float;
using TK_nrhg = float;
using TK_nrhg1 = float;
using TK_min_qv = float;
using TK_r1 = float;
using TK_r2 = float;
using TK_low_limit_mass_for_precip = float;
using TK_eps = float;
using TK_meters3_to_liters = float;
using TK_gonv_min = double;
using TK_gonv_max = double;
using TK_t0 = float;
using TK_rho_w = float;
using TK_sc = float;
using TK_pi = float;
using TK_lsub = float;
using TK_lvap0 = float;
using TK_rv = float;
using TK_rdry = float;
using TK_roverrv = float;
using TK_r = float;
using TK_rho_not = float;
using TK_rho_not0 = float;
using TK_cp = float;
using TK_r_uni = float;
using TK_kap0 = float;
using TK_kap1 = float;
using TK_lam0 = float;
using TK_lam1 = float;
using TK_nbins = float;
using TK_nbc = float;
using TK_nbr = float;
using TK_nbs = float;
using TK_nbi = float;
using TK_nbg = float;
using TK_ntb_i = float;
using TK_ntb_i1 = float;
using TK_ntb_c = float;
using TK_ntb_t = float;
using TK_ntb_g1 = float;
using TK_ntb_s = float;
using TK_ntb_g = float;
using TK_ntb_r = float;
using TK_ntb_r1 = float;
using TK_ntb_t1 = float;
using TK_ntb_in = float;
using TK_ntb_arc = float;
using TK_ntb_arw = float;
using TK_ntb_art = float;
using TK_ntb_arr = float;
using TK_ntb_ark = float;
using TK_idx_bg1 = float;
using TK_nhbins = float;
using TK_radar_bins = float;
using TK_rho_g = float;
using TK_r_c = float;
using TK_r_i = float;
using TK_r_r = float;
using TK_r_s = float;
using TK_r_g = float;
using TK_n0r_exp = float;
using TK_n0g_exp = float;
using TK_nt_i = float;
using TK_nt_in = float;
using TK_ta_na = float;
using TK_ta_ww = float;
using TK_ta_tk = float;
using TK_ta_ra = float;
using TK_ta_ka = float;
using TK_sa = float;
using TK_sb = float;
using TK_tc = float;
using TK_dim_nrhg = float;
using TK_av_g = float;
using TK_bv_g = float;
using TK_am_i = float;
using TK_am_r = float;
using TK_am_g = float;
using TK_lfus = float;
using TK_olfus = float;
using TK_orv = float;
using TK_ar_volume = float;
using TK_sc3 = float;
using TK_d0i = float;
using TK_xm0s = float;
using TK_xm0g = float;
using TK_obmi = float;
using TK_obmr = float;
using TK_oams = float;
using TK_obms = float;
using TK_ocms = float;
using TK_oamg = float;
using TK_ocmg = float;
using TK_obmg = float;
using TK_cce = float;
using TK_ccg = float;
using TK_ocg1 = float;
using TK_ocg2 = float;
using TK_cie = float;
using TK_cig = float;
using TK_oig1 = float;
using TK_oig2 = float;
using TK_cre = float;
using TK_crg = float;
using TK_ore1 = float;
using TK_org1 = float;
using TK_org2 = float;
using TK_org3 = float;
using TK_cse = float;
using TK_csg = float;
using TK_cge = float;
using TK_cgg = float;
using TK_oge1 = float;
using TK_ogg1 = float;
using TK_ogg2 = float;
using TK_ogg3 = float;
using TK_t1_qr_qc = float;
using TK_t1_qr_qi = float;
using TK_t2_qr_qi = float;
using TK_t1_qs_qc = float;
using TK_t1_qs_qi = float;
using TK_t1_qr_ev = float;
using TK_t2_qr_ev = float;
using TK_t1_qs_sd = float;
using TK_t2_qs_sd = float;
using TK_t1_qs_me = float;
using TK_t2_qs_me = float;
using TK_t1_qg_sd = float;
using TK_t1_qg_me = float;
using TK_nic2 = float;
using TK_nii2 = float;
using TK_nii3 = float;
using TK_nir2 = float;
using TK_nir3 = float;
using TK_nis2 = float;
using TK_nig2 = float;
using TK_nig3 = float;
using TK_niin2 = float;
using TK_nic1 = double;
using TK_dc = double;
using TK_dtc = double;
using TK_di = double;
using TK_dti = double;
using TK_dr = double;
using TK_dtr = double;
using TK_ds = double;
using TK_dts = double;
using TK_dg = double;
using TK_dtg = double;
using TK_t_nc = double;
using TK_hbins = double;
using TK_dhbins = double;
using TK_sbins_radar = double;
using TK_dsbins_radar = double;
using TK_gbins_radar = double;
using TK_dgbins_radar = double;
using TK_t_efrw = double;
using TK_t_efsw = double;
using TK_tpc_wev = double;
using TK_tnc_wev = double;
using TK_tnccn_act = float;
using TK_tps_iaus = double;
using TK_tni_iaus = double;
using TK_tpi_ide = double;
using TK_tpi_qcfz = double;
using TK_tni_qcfz = double;
using TK_tpi_qrfz = double;
using TK_tpg_qrfz = double;
using TK_tni_qrfz = double;
using TK_tnr_qrfz = double;
using TK_tcs_racs1 = double;
using TK_tmr_racs1 = double;
using TK_tcs_racs2 = double;
using TK_tmr_racs2 = double;
using TK_tcr_sacr1 = double;
using TK_tms_sacr1 = double;
using TK_tcr_sacr2 = double;
using TK_tms_sacr2 = double;
using TK_tnr_racs1 = double;
using TK_tnr_racs2 = double;
using TK_tnr_sacr1 = double;
using TK_tnr_sacr2 = double;
using TK_tcg_racg = double;
using TK_tmr_racg = double;
using TK_tcr_gacr = double;
using TK_tnr_racg = double;
using TK_tnr_gacr = double;
struct TempoParams {
  const TempoBlob* L;
};

#define PS(name) ((TK_##name)P.L->name[0])
#define PI(name) ((int)(P.L->name[0]))
#define PA1(name, i) ((TK_##name)P.L->name[(long long)(i) - 1])
#define PA2(name, i, j) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * ((long long)(j) - 1)])
#define PA3(name, i, j, k) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * ((long long)(k) - 1))])
#define PA4(name, i, j, k, l) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * ((long long)(l) - 1)))])
#define PA5(name, i, j, k, l, m) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * (((long long)(l) - 1) + \
    TD4_##name * ((long long)(m) - 1))))])

// Per-call context: the Fortran module variables global_dt / global_inverse_dt
// and the configuration flags (module_mp_tempo_cfgs::ty_tempo_cfgs).
struct TempoCtx {
  float global_dt;
  float global_inverse_dt;
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
  int first_call_main;  // v3.2.5: tempo_main's is_first_step (driver:404 itimestep == 1), every column
  int ml_for_nc_flag;
  int hailhyperaware_flag;  // cfgs:18 (forced off with hail-aware off, driver:85)
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
  TT(prh_hcw) TT(pnc_hcw) TT(prh_rch) TT(prr_rch) TT(pnr_rch) TT(prh_hde) TT(prr_hml) TT(pnr_hml)

struct TempoTend {
#define TEMPO_TT_MEMBER(name) double name[TEMPO_NZP];
  TEMPO_TEND_FIELDS(TEMPO_TT_MEMBER)
#undef TEMPO_TT_MEMBER
};

// ---------------------------------------------------------------------------
// Fortran intrinsic helpers
// ---------------------------------------------------------------------------

template<class A,class B> TEMPO_HD auto tmax(A a,B b)->decltype(a+b) {return a>b?a:b;}
template<class A,class B> TEMPO_HD auto tmin(A a,B b)->decltype(a+b) {return a<b?a:b;}
template<class A,class B,class C> TEMPO_HD auto tmax3(A a,B b,C c)->decltype(a+b+c) {return tmax(tmax(a,b),c);}
template<class A,class B,class C> TEMPO_HD auto tmin3(A a,B b,C c)->decltype(a+b+c) {return tmin(tmin(a,b),c);}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD int imax(int a,int b) {return a>b?a:b;}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD int imin(int a,int b) {return a<b?a:b;}
#if TEMPO_WARP
// warp: integer column maxima/minima are order-free, so a butterfly gives the serial result.
TEMPO_HD int tempo_warp_max(int v) {
  for (int o = 16; o > 0; o >>= 1) v = imax(v, __shfl_xor_sync(TEMPO_FULL, v, o));
  return v;
}
TEMPO_HD int tempo_warp_min(int v) {
  for (int o = 16; o > 0; o >>= 1) v = imin(v, __shfl_xor_sync(TEMPO_FULL, v, o));
  return v;
}
#define TEMPO_ROWS ((TEMPO_NZ_MAX + 31) / 32)
// warp: rows[r] bit i marks level 32r+i+1 active.  The nearest active level
// above k, or nz+1 when none: where a serial "x(k) = x(k+1)" chain ends.
TEMPO_HD int tempo_next_active(const unsigned* rows, int k, int nz) {
  for (int b = k; b < nz;) {
    const unsigned m = rows[b >> 5] >> (b & 31);
    if (m) return b + __ffs(m);
    b = ((b >> 5) + 1) << 5;
  }
  return nz + 1;
}
// Map a compact active-level rank to the original level. Integer selection
// only: each active level still evaluates its original FP expressions once.
TEMPO_HD int tempo_active_count(const unsigned* rows, int nz) {
  int total=0; for(int r=0;r<(nz+31)/32;++r)total+=__popc(rows[r]);
  return total;
}
TEMPO_HD int tempo_active_level(const unsigned* rows, int rank, int nz) {
  for(int r=0;r<(nz+31)/32;++r){
    unsigned bits=rows[r]; const int count=__popc(bits);
    if(rank<count){while(rank-- > 0)bits&=bits-1u;return r*32+__ffs(bits);}
    rank-=count;
  }
  return nz+1;
}
#define TEMPO_BALLOT_ROWS(rows, cond)                                          \
  for (int r_ = 0; r_ < (nz + 31) / 32; ++r_) {                                \
    const int k = (r_ << 5) + TEMPO_LANE + 1;                                  \
    rows[r_] = __ballot_sync(TEMPO_FULL, k <= nz && (cond));                   \
  }
#endif
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1161,1196: GNU NINT lowers to lroundf/lround. Adding 0.5
// in the input kind can round a value below the tie onto the tie.
TEMPO_HD int tnint(float x) {return (int)lroundf(x);}
TEMPO_HD int tnint(double x) {return (int)lround(x);}
template<class A,class B> TEMPO_HD auto tsign(A a,B b)->decltype(a+b) {using T=decltype(a+b);return copysign(fabs(T(a)),T(b));}
template<class T> TEMPO_HD T tpowi(T x,int m) {
    bool neg=m<0;unsigned int n=neg?0U-(unsigned int)m:(unsigned int)m;
    T y=(n&1U)?x:T(1);while(n>>=1U){x=x*x;if(n&1U)y=y*x;}
    return neg?T(1)/y:y;
}
TEMPO_HD float tempo_exp(float x) {return mpas_expf(x);}
TEMPO_HD double tempo_exp(double x) {return mpas_exp(x);}
TEMPO_HD float tempo_log(float x) {return mpas_logf(x);}
TEMPO_HD double tempo_log(double x) {return mpas_log(x);}
TEMPO_HD float tempo_log10(float x) {return mpas_log10f(x);}
TEMPO_HD double tempo_log10(double x) {return mpas_log10(x);}
TEMPO_HD float tempo_pow(float x,float y) {return mpas_powf(x,y);}
TEMPO_HD double tempo_pow(double x,double y) {return mpas_pow(x,y);}
template<class A,class B> TEMPO_HD auto tempo_pow(A x,B y)->decltype(x+y) {using T=decltype(x+y);return tempo_pow(T(x),T(y));}
TEMPO_HD float tempo_gamma(float x) {return mpas_tgammaf(x);}
TEMPO_HD double tempo_gamma(double x) {return mpas_tgamma(x);}
// Several binary64 pows of ONE base (snow moments, rain fall speeds): the base's logarithm is
// evaluated once and each exponent finishes glibc pow's own fast path (log_inline, the ehi/elo
// split, exp_inline; every step an explicit _rn intrinsic, so no contraction choice can differ).
// Any operand outside that fast path (zero, subnormal, negative, inf or NaN base; |y| outside
// [2^-65, 2^63) or zero/inf/NaN) goes to mpas_pow itself.  So tempo_pow_with(tempo_pow_base(x), y)
// == mpas_pow(x, y) word for word.  TEMPO_SHARED_POW is defined only when the supplied maths
// header's mpas_pow is that glibc pow (g64_pow); otherwise every call is plain mpas_pow.
struct TempoPowBase { double x, hi, lo; bool fast; };
#if defined(TEMPO_SHARED_POW)
TEMPO_HD TempoPowBase tempo_pow_base(double x) {
  TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0;
  const uint32_t topx = g64_pow::top12(x);
  b.fast = !(topx - 0x001 >= 0x7ff - 0x001);
  if (b.fast) b.hi = g64_pow::log_inline(g64_pow::asuint64(x), &b.lo);
  return b;
}
TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {
  const uint32_t topy = g64_pow::top12(y);
  if (!b.fast || (topy & 0x7ff) - 0x3be >= 0x43e - 0x3be) return mpas_pow(b.x, y);
  const double ehi = __dmul_rn(y, b.hi);
  const double elo = __fma_rn(y, b.lo, __fma_rn(y, b.hi, -ehi));
  return g64_pow::exp_inline(ehi, elo, 0);
}
#else
TEMPO_HD TempoPowBase tempo_pow_base(double x) {TempoPowBase b; b.x = x; b.hi = 0.0; b.lo = 0.0; b.fast = false; return b;}
TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {return mpas_pow(b.x, y);}
#endif
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:256-282 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float calc_rslf(float p, float t) {
  const float c0 = .611583699E03f; const float  c1 = .444606896E02f; const float  c2 = .143177157e01f;
  const float c3 = .264224321e-1f; const float  c4 = .299291081e-3f; const float  c5 = .203154182e-5f;
  const float c6 = .702620698e-8f; const float  c7 = .379534310e-11f; const float  c8 = -.321582393e-13f;
  float x = tmax(-80.0f, t - 273.16f);
  float esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
  esl = tmin(esl, p * 0.15f);
  return .622f * esl / (p - esl);
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:283-304 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float calc_rsif(float p, float t) {
  const float c0 = .609868993e03f; const float  c1 = .499320233e02f; const float  c2 = .184672631e01f;
  const float c3 = .402737184e-1f; const float  c4 = .565392987e-3f; const float  c5 = .521693933e-5f;
  const float c6 = .307839583e-7f; const float  c7 = .105785160e-9f; const float  c8 = .161444444e-12f;
  float x = tmax(-80.0f, t - 273.16f);
  float esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));
  esi = tmin(esi, p * 0.15f);
  return .622f * esi / tmax(1.e-4f, (p - esi));
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:305-319 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD int get_nuc(const TempoParams& P, float nc) {
  if ((PS(nu_c_scale) / nc) >= 12.5f) return 15;
  return imin(15, tnint(PS(nu_c_scale) / nc) + 2);
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:45-78 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float calc_gamma_series(float a, float x) {
  const int it_max = 100;
  const float smallvalue = 1.e-7f;
  float ap1 = a;
  float sum_term = 1.0f / tempo_gamma(ap1 + 1.0f);
  float sum = sum_term;
  for (int k = 1; k <= it_max; ++k) {
    ap1 = ap1 + 1.0f;
    sum_term = sum_term * x / ap1;
    sum = sum + sum_term;
    if (fabs(sum_term) < (fabs(sum) * smallvalue)) break;
  }
  return sum * tempo_pow(x, a) * tempo_exp(-x);
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:79-118 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float calc_gamma_cf(float a, float x) {
  const int it_max = 100;
  const float smallvalue = 1.e-7f;
  const float offset = 1.e-30f;
  float b = 1.0f - a + x;
  float d = 1.0f / b;
  float h0 = offset;
  float c = b + (1.0f / offset);
  float delta = c * d;
  float h = h0 * delta;
  for (int k = 1; k <= it_max; ++k) {
    float aj = k * (a - k);
    b = b + 2.0f;
    d = b + aj * d;
    if (fabs(d) < offset) d = offset;
    c = b + aj / c;
    if (fabs(c) < offset) c = offset;
    d = 1.0f / d;
    delta = c * d;
    h = h * delta;
    if (fabs(delta - 1.0f) < smallvalue) break;
  }
  return tempo_exp(-x + a * tempo_log(x)) * h / tempo_gamma(a);
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:25-44 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float calc_gamma_p(float a, float x) {
  if (x < (a + 1.0f)) return calc_gamma_series(a, x);
  return 1.0f - calc_gamma_cf(a, x);
}

// snow_moments: optional outputs are nullptr when absent.  As in the
// Fortran, smoz reads the smo2 OUTPUT (callers always pass smo2 with smoz).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:119-255 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD void snow_moments(const TempoParams& P, float rs, float tc,
                           double* smob, double* smoc, double* ns, double* smo0,
                           double* smo1, double* smo2, double* smoe, double* smof,
                           double* smog, double* smoz) {
  const float bm_s = PS(bm_s);
  double loga_; double  a_; double  b_; double  smo2_; double  m0; double  mrat; double  slam1; double  slam2; float  cc;
#define SA(i) PA1(sa, i)
#define SB(i) PA1(sb, i)
  // pow(10, loga) shares log 10; pow(smo2, b) shares log smo2 (tempo_pow_base: exact).
  const TempoPowBase b10 = tempo_pow_base(10.0);
  *smob = rs * PS(oams);
  if (bm_s > 2.0f - 1.e-3f && bm_s < 2.0f + 1.e-3f) {
    loga_ = SA(1) + SA(2) * tc + SA(3) * bm_s + SA(4) * tc * bm_s + SA(5) * tc * tc
          + SA(6) * bm_s * bm_s + SA(7) * tc * tc * bm_s + SA(8) * tc * bm_s * bm_s
          + SA(9) * tc * tc * tc + SA(10) * bm_s * bm_s * bm_s;
    a_ = tempo_pow_with(b10, loga_);
    b_ = SB(1) + SB(2) * tc + SB(3) * bm_s + SB(4) * tc * bm_s + SB(5) * tc * tc
       + SB(6) * bm_s * bm_s + SB(7) * tc * tc * bm_s + SB(8) * tc * bm_s * bm_s
       + SB(9) * tc * tc * tc + SB(10) * bm_s * bm_s * bm_s;
    smo2_ = tempo_pow((*smob / a_), (1.0f / b_));
  } else {
    smo2_ = *smob;
  }
  if (smo2) *smo2 = smo2_;
  const TempoPowBase bs = tempo_pow_base(smo2_);

#define TEMPO_SNOW_AB(cc)                                                              \
  loga_ = SA(1) + SA(2) * tc + SA(3) * (cc) + SA(4) * tc * (cc) + SA(5) * tc * tc        \
        + SA(6) * (cc) * (cc) + SA(7) * tc * tc * (cc) + SA(8) * tc * (cc) * (cc)        \
        + SA(9) * tc * tc * tc + SA(10) * (cc) * (cc) * (cc);                            \
  a_ = tempo_pow_with(b10, loga_);                                                                 \
  b_ = SB(1) + SB(2) * tc + SB(3) * (cc) + SB(4) * tc * (cc) + SB(5) * tc * tc           \
     + SB(6) * (cc) * (cc) + SB(7) * tc * tc * (cc) + SB(8) * tc * (cc) * (cc)           \
     + SB(9) * tc * tc * tc + SB(10) * (cc) * (cc) * (cc);

  cc = PA1(cse, 1);
  TEMPO_SNOW_AB(cc)
  *smoc = a_ * tempo_pow_with(bs, b_);

  loga_ = SA(1) + SA(2) * tc + SA(5) * tc * tc + SA(9) * tc * tc * tc;
  a_ = tempo_pow_with(b10, loga_);
  b_ = SB(1) + SB(2) * tc + SB(5) * tc * tc + SB(9) * tc * tc * tc;
  if (smo0) *smo0 = a_ * tempo_pow_with(bs, b_);

  loga_ = SA(1) + SA(2) * tc + SA(3) + SA(4) * tc + SA(5) * tc * tc + SA(6)
        + SA(7) * tc * tc + SA(8) * tc + SA(9) * tc * tc * tc + SA(10);
  a_ = tempo_pow_with(b10, loga_);
  b_ = SB(1) + SB(2) * tc + SB(3) + SB(4) * tc + SB(5) * tc * tc + SB(6)
     + SB(7) * tc * tc + SB(8) * tc + SB(9) * tc * tc * tc + SB(10);
  if (smo1) *smo1 = a_ * tempo_pow_with(bs, b_);

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
  if (smoe) *smoe = a_ * tempo_pow_with(bs, b_);

  cc = PA1(cse, 16);
  TEMPO_SNOW_AB(cc)
  if (smof) *smof = a_ * tempo_pow_with(bs, b_);

  cc = PA1(cse, 17);
  TEMPO_SNOW_AB(cc)
  if (smog) *smog = a_ * tempo_pow_with(bs, b_);

  cc = PA1(cse, 3);
  TEMPO_SNOW_AB(cc)
  if (smoz) *smoz = a_ * tempo_pow_with(bs, b_);
#undef TEMPO_SNOW_AB
#undef SA
#undef SB
}

// TEMPO microphysics port for WOOF: process-rate routines.
// Line-by-line transliteration of NCAR/TEMPO 9adf9ef (tag tempo_v3.0.3)
//   src/module_mp_tempo_main.F90 lines 1958-3153
//   src/module_mp_tempo_aerosols.F90 (whole module)
// Apache-2.0, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
// Conventions: see tempo_common.cuh.  Column arrays are 1-based (a[1..nz]).
// Fortran quirks are preserved bug-for-bug; each is marked "Fortran quirk".

// ---------------------------------------------------------------------------
// module_mp_tempo_aerosols.F90
// ---------------------------------------------------------------------------

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:14-42
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:13-42 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void init_water_friendly_aerosols(const TempoParams& P, int nz,
                                                   const float* dz1d, float* nwfa) {
  float hgt[TEMPO_NZP];
  float h_01; float  niccn3;
  for (int k = 0; k < TEMPO_NZP; ++k) hgt[k] = 0.0f;
  // Fortran quirk: hgt(k) accumulates dz1d(k), not dz1d(k-1); hgt(1) = 0.
  for (int k = 2; k <= nz; ++k) hgt[k] = hgt[k - 1] + dz1d[k];
  if (hgt[1] <= 1000.0f) {
    h_01 = 0.8f;
  } else if (hgt[1] >= 2500.0f) {
    h_01 = 0.01f;
  } else {
    h_01 = 0.8f * mpas_cosf(hgt[1] * 0.001f - 1.0f);
  }
  niccn3 = -1.0f * tempo_log(PS(naccn1) / PS(naccn0)) / h_01;
  // Fortran quirk: level 1 uses the level-2 height.
  nwfa[1] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[2] - hgt[1]) / hex_divisor_447a0000) * niccn3);
  for (int k = 2; k <= nz; ++k) {
    nwfa[k] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[k] - hgt[1]) / hex_divisor_447a0000) * niccn3);
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:45-73
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:43-73 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void init_ice_friendly_aerosols(const TempoParams& P, int nz,
                                                 const float* dz1d, float* nifa) {
  float hgt[TEMPO_NZP];
  float h_01; float  niin3;
  for (int k = 0; k < TEMPO_NZP; ++k) hgt[k] = 0.0f;
  for (int k = 2; k <= nz; ++k) hgt[k] = hgt[k - 1] + dz1d[k];
  if (hgt[1] <= 1000.0f) {
    h_01 = 0.8f;
  } else if (hgt[1] >= 2500.0f) {
    h_01 = 0.01f;
  } else {
    h_01 = 0.8f * mpas_cosf(hgt[1] * 0.001f - 1.0f);
  }
  niin3 = -1.0f * tempo_log(PS(nain1) / PS(nain0)) / h_01;
  nifa[1] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[2] - hgt[1]) / hex_divisor_447a0000) * niin3);
  for (int k = 2; k <= nz; ++k) {
    nifa[k] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[k] - hgt[1]) / hex_divisor_447a0000) * niin3);
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:76-118
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:74-118 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline float aerosol_collection_efficiency(const TempoParams& P, double d, float da,
                                                      float visc, float rhoa, float temp,
                                                      char species) {
  float aval; float  cc; float  diff; float  re; float  sc; float  st; float  st2; float  vt; float  eff; float  rho_p;
  const float boltzman = 1.3806503e-23f;
  const float meanpath = 0.0256e-6f;
  vt = 1.0f;
  rho_p = PS(rho_w);
  if (species == 'r') {
    vt = -0.1021f + 4.932e3f * d - 0.9551e6f * d * d + 0.07934e9f * d * d * d
       - 0.002362e12f * d * d * d * d;
    rho_p = PS(rho_w);
  } else if (species == 's') {
    vt = PS(av_s) * tempo_pow(d, PS(bv_s));
    rho_p = PS(rho_s);
  } else if (species == 'g') {
    vt = PA1(av_g, PI(idx_bg1)) * tempo_pow(d, PA1(bv_g, PI(idx_bg1)));
    rho_p = PA1(rho_g, PI(idx_bg1));
  }
  cc = 1.0f + 2.0f * meanpath / da * (1.257f + 0.4f * tempo_exp(-0.55f * da / meanpath));
  diff = boltzman * temp * cc / (3.0f * PS(pi) * visc * da);
  re = 0.5f * rhoa * d * vt / visc;
  sc = visc / (rhoa * diff);
  st = (rho_p - rhoa) * da * da * vt * cc / (9.0f * visc * d);
  aval = tempo_log(1.0f + re);
  st2 = (1.2f + 1.0f / hex_divisor_41400000 * aval) / (1.0f + aval);
  eff = 4.0f / (re * sc) * (1.0f + 0.4f * sqrt(re) * tempo_pow(sc, 0.3333f) + 0.16f * sqrt(re) * sqrt(sc))
      + 4.0f * da / d * (0.02f + da / d * (1.0f + 2.0f * sqrt(re)));
  if (st > st2) eff = eff + tempo_pow(((st - st2) / (st - st2 + 0.666667f)), 1.5f);
  return tmax(1.e-5f, tmin(eff, 1.0f));
}

// ---------------------------------------------------------------------------
// table index helpers (MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2459-2601, 2802-2821)
// ---------------------------------------------------------------------------

// Fortran pattern: do nn = m-1, m+1; n = nn; if (x/10**nn in [1,10)) exit
// enddo.  Without an exit n keeps the last value m+1.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD int tempo_decade(float x, int m) {
  int n = m - 1;
  for (int nn = m - 1; nn <= m + 1; ++nn) {
    n = nn;
    if ((x / tpowi(10.0f, nn)) >= 1.0f && (x / tpowi(10.0f, nn)) < 10.0f) break;
  }
  return n;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2459-2474
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2457-2474 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_snow_table_index(const TempoParams& P, float rs, int* idx_s) {
  int nis = tnint(tempo_log10(rs));
  int n = tempo_decade(rs, nis);
  int v = (int)(rs / tpowi(10.0f, n)) + 10 * (n - PI(nis2)) - (n - PI(nis2));
  *idx_s = imax(1, imin(v, PI(ntb_s)));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2477-2489
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2475-2489 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_temperature_table_index(const TempoParams& P, float tempk, int* idx_t) {
  float tempc = tempk - PS(t0);
  int v = (int)((tempc - 2.5f) / hex_divisor_40a00000) - 1;
  v = imax(1, -v);
  *idx_t = imin(v, PI(ntb_t));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2492-2521
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2490-2521 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_rain_table_index(const TempoParams& P, float rr, double ilamr,
                                           int* idx_r, int* idx_r1) {
  double lamr; double  lam_exp; double  n0_exp;
  int nir = tnint(tempo_log10(rr));
  int n = tempo_decade(rr, nir);
  int v = (int)(rr / tpowi(10.0f, n)) + 10 * (n - PI(nir2)) - (n - PI(nir2));
  *idx_r = imax(1, imin(v, PI(ntb_r)));

  lamr = 1.f / ilamr;
  lam_exp = lamr * tpowi((PA1(crg, 3) * PS(org2) * PS(org1)), 3);
  n0_exp = PS(org1) * rr / PS(am_r) * tempo_pow(lam_exp, PA1(cre, 1));
  nir = tnint(tempo_log10(n0_exp));
  n = tempo_decade(n0_exp, nir);
  v = (int)(n0_exp / tpowi(10.0f, n)) + 10 * (n - PI(nir3)) - (n - PI(nir3));
  *idx_r1 = imax(1, imin(v, PI(ntb_r1)));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2524-2554
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2522-2554 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_graupel_table_index(const TempoParams& P, float rg, double ilamg,
                                              int idx, int* idx_g, int* idx_g1) {
  double lamg; double  lam_exp; double  n0_exp;
  int nig = tnint(tempo_log10(rg));
  int n = tempo_decade(rg, nig);
  int v = (int)(rg / tpowi(10.0f, n)) + 10 * (n - PI(nig2)) - (n - PI(nig2));
  *idx_g = imax(1, imin(v, PI(ntb_g)));

  lamg = 1.f / ilamg;
  lam_exp = lamg * tpowi((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), 3);
  n0_exp = PS(ogg1) * rg / PA1(am_g, idx) * tempo_pow(lam_exp, PA2(cge, 1, 1));
  nig = tnint(tempo_log10(n0_exp));
  n = tempo_decade(n0_exp, nig);
  v = (int)(n0_exp / tpowi(10.0f, n)) + 10 * (n - PI(nig3)) - (n - PI(nig3));
  *idx_g1 = imax(1, imin(v, PI(ntb_g1)));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2557-2575
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2555-2575 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_cloud_table_index(const TempoParams& P, float rc, float nc,
                                            int* idx_c, int* idx_n) {
  int nic = tnint(tempo_log10(rc));
  int n = tempo_decade(rc, nic);
  int v = (int)(rc / tpowi(10.0f, n)) + 10 * (n - PI(nic2)) - (n - PI(nic2));
  *idx_c = imax(1, imin(v, PI(ntb_c)));

  v = tnint(1.0f + (float )PI(nbc) * tempo_log(nc / PA1(t_nc, 1)) / PS(nic1));
  *idx_n = imax(1, imin(v, PI(nbc)));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2578-2601
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2576-2601 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_ice_table_index(const TempoParams& P, float ri, float ni,
                                          int* idx_i, int* idx_i1) {
  int nii = tnint(tempo_log10(ri));
  int n = tempo_decade(ri, nii);
  int v = (int)(ri / tpowi(10.0f, n)) + 10 * (n - PI(nii2)) - (n - PI(nii2));
  *idx_i = imax(1, imin(v, PI(ntb_i)));

  nii = tnint(tempo_log10(ni));
  n = tempo_decade(ni, nii);
  v = (int)(ni / tpowi(10.0f, n)) + 10 * (n - PI(nii3)) - (n - PI(nii3));
  *idx_i1 = imax(1, imin(v, PI(ntb_i1)));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2802-2821
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2800-2821 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_in_table_index(const TempoParams& P, float xni, int* idx_in) {
  if (xni > PA1(nt_in, 1)) {
    int niin = tnint(tempo_log10(xni));
    int n = tempo_decade(xni, niin);
    int v = (int)(xni / tpowi(10.0f, n)) + 10 * (n - PI(niin2)) - (n - PI(niin2));
    *idx_in = imax(1, imin(v, PI(ntb_in)));
  } else {
    *idx_in = 1;
  }
}

// ---------------------------------------------------------------------------
// nucleation / activation functions
// ---------------------------------------------------------------------------

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2136-2163
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2134-2163 @ 1d6e370e; TEMPO 9adf9ef.
// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:2475-2500 (koop_nucleation(temp, ssatw, naero, dt)).
TEMPO_HDN inline float koop_nucleation(const TempoParams& P, const TempoCtx& C, float temp,
                                        float ssatw, float naero) {
  float xni; float  mu_diff; float  a_w_i; float  delta_aw; float  log_j_rate; float  j_rate; float  prob_h;
  xni = 0.0f;
  mu_diff = 210368.0f + (131.438f * temp) - (3.32373e6f / temp) - (41729.1f * tempo_log(temp));
  a_w_i = tempo_exp(mu_diff / (PS(r_uni) * temp));
  delta_aw = ssatw + 1.0f - a_w_i;
  log_j_rate = -906.7f + (8502.0f * delta_aw) - (26924.0f * delta_aw * delta_aw)
             + (29180.0f * delta_aw * delta_aw * delta_aw);
  log_j_rate = tmin(20.0f, log_j_rate);
  j_rate = tempo_pow(10.0f, log_j_rate);
  prob_h = tmin(1.0f - tempo_exp(-j_rate * PS(ar_volume) * C.global_dt), 1.0f);
  if (prob_h > 0.0f) {
    xni = tmin(prob_h * naero, 1000.e3f);
  }
  return tmax(0.0f, xni);
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2166-2244.  land is the optional Fortran dummy (nullptr = absent).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2164-2244 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline float activate_cloud_number(const TempoParams& P, float temp, float w1d,
                                              float nwfa, const int* land) {
  float n_local; float  w_local;
  float a; float  b; float  c; float  d; float  t; float  u; float  x1; float  x2; float  y1; float  y2; float  nx; float  wy; float  fraction;
  float lower_lim_nuc_frac;
  int i, j, k, l, m, n;
  const int ntb_arc = PI(ntb_arc), ntb_arw = PI(ntb_arw), ntb_art = PI(ntb_art);

  n_local = nwfa * 1.e-6f;
  if (n_local >= PA1(ta_na, ntb_arc)) {
    n_local = PA1(ta_na, ntb_arc) - 1.0f;
  } else if (n_local <= PA1(ta_na, 1)) {
    n_local = PA1(ta_na, 1) + 1.0f;
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
    w_local = PA1(ta_ww, ntb_arw) - 1.0f;
  } else if (w_local <= PA1(ta_ww, 1)) {
    w_local = PA1(ta_ww, 1) + 0.001f;
  }
  for (n = 2; n <= ntb_arw; ++n) {
    if (w_local >= PA1(ta_ww, n - 1) && w_local < PA1(ta_ww, n)) break;
  }
  j = n;
  y1 = tempo_log(PA1(ta_ww, j - 1));
  y2 = tempo_log(PA1(ta_ww, j));

  k = imax(1, imin(tnint((temp - PA1(ta_tk, 1)) * 0.1f) + 1, ntb_art));

  l = 3;
  m = 2;

  lower_lim_nuc_frac = 0.f;
  if (land != nullptr) {
    if (*land == 1) {
      lower_lim_nuc_frac = 0.f;
    } else if (*land == 0) {
      lower_lim_nuc_frac = 0.15f;
    } else {
      lower_lim_nuc_frac = 0.15f;
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

  fraction = (1.0f - t) * (1.0f - u) * a + t * (1.0f - u) * b + t * u * c + (1.0f - t) * u * d;
  fraction = tmax(fraction, lower_lim_nuc_frac);

  return nwfa * fraction;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2786-2799
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2784-2799 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline float demott_nucleation(const TempoParams& P, float tempc, float rho,
                                          float nifa) {
  float xni; float  nifa_cc;
  xni = 0.0f;
  nifa_cc = tmax(0.5f, nifa * PS(rho_not0) * 1.e-6f / rho);
  xni = (5.94e-5f * tempo_pow((-tempc), 3.33f)) * tempo_pow(nifa_cc, ((-0.0264f * (tempc)) + 0.0033f));
  xni = xni * rho / PS(rho_not0) * 1000.0f;
  return tmax(0.0f, xni);
}

// ---------------------------------------------------------------------------
// condensation / evaporation
// ---------------------------------------------------------------------------

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1958-2026
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1956-2026 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void cloud_condensation(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* temp, const float* w1d, const float* ssatw,
    const float* lvap, const float* tcond, const float* diffu, const float* lvt2,
    const float* nwfa1d, const float* nwfa, const float* ncsave, const float* qv,
    const float* qvs, const bool* l_qc, const float* rc, const float* nc, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:2284-2361
  float clap; float  fcd; float  dfcd; float  xrc; float  xnc; float  orho; float  tempc; float  otemp; float  rvs; float  rvs_p; float  rvs_pp; float  gamsc; float  alphsc; float 
      xsat; float  t1_evap;
  double dc_star;
  int idx_d, idx_n, idx_c;
  const float orv = PS(orv); const float  pi = PS(pi);
  TEMPO_LEVELS(k, nz) {
    if (fabs(ssatw[k]) < PS(eps)) continue;  // v3.2.5 main:2303: CYCLE (v3.0.3 returned)

    orho = 1.0f / rho[k];
    clap = (qv[k] - qvs[k]) / (1.0f + lvt2[k] * qvs[k]);
    for (int n = 1; n <= 3; ++n) {
      fcd = qvs[k] * tempo_exp(lvt2[k] * clap) - qv[k] + clap;
      dfcd = qvs[k] * lvt2[k] * tempo_exp(lvt2[k] * clap) + 1.0f;
      clap = clap - fcd / dfcd;
    }
    xrc = rc[k] + clap * rho[k];
    xnc = 0.0f;

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
        T.pnc_wcd[k] = 0.5f * (xnc - nc[k] + fabs(xnc - nc[k])) * C.global_inverse_dt * orho;
      } else if (l_qc[k] && ssatw[k] < -1.e-6f && clap < -PS(eps)) {
        tempc = temp[k] - PS(t0);
        otemp = 1.0f / temp[k];
        rvs = rho[k] * qvs[k];
        rvs_p = rvs * otemp * (lvap[k] * otemp * orv - 1.0f);
        rvs_pp = rvs * (otemp * (lvap[k] * otemp * orv - 1.0f) * otemp * (lvap[k] * otemp * orv - 1.0f)
               + (-2.0f * lvap[k] * otemp * otemp * otemp * orv) + otemp * otemp);
        gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
        alphsc = 0.5f * (gamsc / (1.0f + gamsc)) * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = tmax(1.e-9f, alphsc);
        xsat = ssatw[k];
        if (fabs(xsat) < 1.e-9f) xsat = 0.0f;
        t1_evap = 2.0f * pi * (1.0f - alphsc * xsat + 2.0f * alphsc * alphsc * xsat * xsat
                - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);

        dc_star = sqrt(-2.0 * C.global_dt * t1_evap / (2.0 * pi) * 4.0 * diffu[k] * ssatw[k] * rvs
                       / PS(rho_w));
        idx_d = imax(1, imin((int)(1.e6f * dc_star), PI(nbc)));
        get_cloud_table_index(P, rc[k], nc[k], &idx_c, &idx_n);

        T.prw_vcd[k] = tmax(-rc[k] * 0.99f * orho * C.global_inverse_dt, T.prw_vcd[k]);
        T.pnc_wcd[k] = tmax(-nc[k] * 0.99f * orho * C.global_inverse_dt,
                            -PA3(tnc_wev, idx_d, idx_c, idx_n) * orho * C.global_inverse_dt);
      }
    } else {
      T.prw_vcd[k] = -rc[k] * orho * C.global_inverse_dt;
      T.pnc_wcd[k] = -nc[k] * orho * C.global_inverse_dt;
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2029-2090
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2027-2090 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void rain_evaporation(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* temp, const float* ssatw, const float* lvap,
    const float* tcond, const float* diffu, const float* vsc2, const float* rhof2,
    const float* qv, const float* qvs, const bool* l_qr, const float* rr,
    const float* nr, const double* ilamr, TempoTend& T) {
  float orho; float  tempc; float  otemp; float  rvs; float  rvs_p; float  rvs_pp; float  gamsc; float  alphsc; float  xsat; float  t1_evap; float  rate_max; float 
      eva_factor;
  double lamr; double  n0_r;
  const float orv = PS(orv); const float  pi = PS(pi);
  TEMPO_LEVELS(k, nz) {
    if (l_qr[k]) {
      if ((ssatw[k] < -PS(eps)) && T.prw_vcd[k] <= 0.0f) {
        orho = 1.0f / rho[k];
        tempc = temp[k] - PS(t0);
        otemp = 1.0f / temp[k];
        rvs = rho[k] * qvs[k];
        rvs_p = rvs * otemp * (lvap[k] * otemp * orv - 1.0f);
        rvs_pp = rvs * (otemp * (lvap[k] * otemp * orv - 1.0f) * otemp * (lvap[k] * otemp * orv - 1.0f)
               + (-2.0f * lvap[k] * otemp * otemp * otemp * orv) + otemp * otemp);
        gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
        alphsc = 0.5f * (gamsc / (1.0f + gamsc)) * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = tmax(1.e-9f, alphsc);
        xsat = tmin(-1.e-9f, ssatw[k]);
        t1_evap = 2.0f * pi * (1.0f - alphsc * xsat + 2.0f * alphsc * alphsc * xsat * xsat
                - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);

        if (qv[k] / qvs[k] < 0.95f && rr[k] * orho <= 1.e-8f) {
          T.prv_rev[k] = rr[k] * orho * C.global_inverse_dt;
        } else {
          lamr = 1.0f / ilamr[k];
          n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
          T.prv_rev[k] = t1_evap * diffu[k] * (-ssatw[k]) * n0_r * rvs
                       * (PS(t1_qr_ev) * tempo_pow(ilamr[k], PA1(cre, 10))
                          + PS(t2_qr_ev) * vsc2[k] * rhof2[k]
                            * (tempo_pow((lamr + 0.5f * PS(fv_r)), (-PA1(cre, 11)))));
          rate_max = tmin((rr[k] * orho * C.global_inverse_dt), (qvs[k] - qv[k]) * C.global_inverse_dt);
          T.prv_rev[k] = tmin(rate_max, T.prv_rev[k] * orho);
          if (T.prr_gml[k] > 0.0f) {
            eva_factor = tmin(1.0f, 0.01f + (0.99f - 0.01f) * (tempc / hex_divisor_41a00000));
            T.prv_rev[k] = T.prv_rev[k] * eva_factor;
          }
        }
        T.pnr_rev[k] = tmin(nr[k] * 0.99f * orho * C.global_inverse_dt, T.prv_rev[k] * nr[k] / rr[k]);
      }
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2093-2133.  nc1d and ncsave are optional (nullptr = absent).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2091-2133 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void freeze_cloud_melt_ice(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const float* rho, const float* ocp, const float* lvap,
    const float* qi1d, const float* ni1d, float* qiten, float* niten,
    const float* qc1d, const float* nc1d, float* qcten, float* ncten, float* tten,
    const float* ncsave) {
  float xri; float  xrc; float  lfus2; float  xnc;
  TEMPO_LEVELS(k, nz) {
    xri = tmax(0.0f, qi1d[k] + qiten[k] * C.global_dt);
    if ((temp[k] > PS(t0)) && (xri > 0.0f)) {
      qcten[k] = qcten[k] + xri * C.global_inverse_dt;
      ncten[k] = ncten[k] + ni1d[k] * C.global_inverse_dt;
      qiten[k] = qiten[k] - xri * C.global_inverse_dt;
      niten[k] = -ni1d[k] * C.global_inverse_dt;
      tten[k] = tten[k] - PS(lfus) * ocp[k] * xri * C.global_inverse_dt;
    }
    xrc = tmax(0.0f, qc1d[k] + qcten[k] * C.global_dt);
    if ((temp[k] < PS(hgfrz)) && (xrc > 0.0f)) {
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2247-2322
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2245-2322 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void warm_rain(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const bool* l_qc, const float* rc, const float* nc,
    const double* ilamc, const float* mvd_c, const bool* l_qr, const float* rr,
    const float* nr, const float* mvd_r, TempoTend& T) {
  double lamr; double  lamc; double  n0_r;
  float ef_rr; float  dc_g; float  dc_b; float  xdc; float  zeta1; float  zeta; float  taud; float  tau; float  ef_rw;
  int nu_c, idx;
  const float d0r = PS(d0r); const float  d0c = PS(d0c); const float  am_r = PS(am_r);
  TEMPO_LEVELS(k, nz) {
    if (l_qr[k]) {
      if (mvd_r[k] > d0r) {
        ef_rr = tmax(-0.1f, 1.0f - tempo_exp(2300.0f * (mvd_r[k] - 1950.0e-6f)));
        T.pnr_rcr[k] = ef_rr * 2.0f * nr[k] * rr[k];
      }
    }

    if (l_qc[k]) {
      if (rc[k] > 0.01e-3f) {
        nu_c = get_nuc(P, nc[k]);
        lamc = 1.0f / ilamc[k];
        xdc = tmax(d0c * 1.e6f, (tempo_pow((rc[k] / (am_r * nc[k])), PS(obmr))) * 1.e6f);
        dc_g = (tempo_pow((PA2(ccg, 3, nu_c) * PA1(ocg2, nu_c)), PS(obmr)) / lamc) * 1.e6f;
        dc_b = tempo_pow((xdc * xdc * xdc * dc_g * dc_g * dc_g - xdc * xdc * xdc * xdc * xdc * xdc),
                   (1.0f / hex_divisor_40c00000));
        zeta1 = 0.5f * ((6.25e-6f * xdc * dc_b * dc_b * dc_b - 0.4f)
                       + fabs(6.25e-6f * xdc * dc_b * dc_b * dc_b - 0.4f));
        zeta = 0.027f * rc[k] * zeta1;
        taud = 0.5f * ((0.5f * dc_b - 7.5f) + fabs(0.5f * dc_b - 7.5f)) + PS(r1);
        tau = 3.72f / (rc[k] * taud);
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
        ef_rw = PA2(t_efrw, idx, (int)(mvd_c[k] * 1.e6f));
        n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
        const double p9 = tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9)));  // one evaluation, two uses
        T.prr_rcw[k] = rhof[k] * PS(t1_qr_qc) * ef_rw * rc[k] * n0_r
                     * (p9);
        T.prr_rcw[k] = tmin(rc[k] * C.global_inverse_dt, T.prr_rcw[k]);
        T.pnc_rcw[k] = rhof[k] * PS(t1_qr_qc) * ef_rw * nc[k] * n0_r
                     * (p9);
        T.pnc_rcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_rcw[k]);
      }
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2325-2456.  vtboost is intent(out).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2323-2456 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void riming(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const float* rhof, const float* visco, const bool* l_qc,
    const float* rc, const float* nc, const double* ilamc, const float* mvd_c,
    const bool* l_qs, const float* rs, const double* smo0, const double* smob,
    const double* smoc, const double* smoe, float* vtboost, const bool* l_qg,
    const float* rg, const float* ng, const double* ilamg, const int* idx,
    const bool* l_qh, const float* rh, const float* nh, const double* ilamh, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:2662-2840 (hail collecting cloud water :2736-2750)
  double xds; double  xdg; double  n0_g; double  lamc; double  xdh; double  n0_h;
  float ef_hw; float  t1_qh_qc;
  float ef_sw; float  vtg; float  stoke_g; float  const_ri; float  tempc; float  rime_dens; float  ef_gw;
  float t1_qg_qc; float  r_frac; float  g_frac; float  vts; float  tf; float  snow_dens_frac;
  int idxs, nu_c;
  const float t0 = PS(t0); const float  eps = PS(eps); const float  mtl = PS(meters3_to_liters);
  TEMPO_LEVELS(k, nz) {
    tempc = temp[k] - t0;
    vtboost[k] = 1.0f;
    if (l_qc[k] && l_qs[k]) {
      nu_c = get_nuc(P, nc[k]);
      lamc = 1.0f / ilamc[k];
      (void)nu_c;
      (void)lamc;
      xds = smoc[k] / smob[k];
      if ((mvd_c[k] > PS(d0c)) && (xds > PS(d0s))) {
        idxs = 1 + (int)(PI(nbs) * tempo_log(xds / PA1(ds, 1)) / tempo_log(PA1(ds, PI(nbs)) / PA1(ds, 1)));
        idxs = imin(idxs, PI(nbs));
        ef_sw = PA2(t_efsw, idxs, (int)(mvd_c[k] * 1.e6f));
        T.prs_scw[k] = rhof[k] * PS(t1_qs_qc) * ef_sw * rc[k] * smoe[k];
        T.prs_scw[k] = tmin(rc[k] * C.global_inverse_dt, T.prs_scw[k]);
        T.pnc_scw[k] = rhof[k] * PS(t1_qs_qc) * ef_sw * nc[k] * smoe[k];
        T.pnc_scw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_scw[k]);

        if (temp[k] < t0) {
          if (T.prs_scw[k] > PS(rime_threshold) * T.prs_sde[k] && T.prs_sde[k] > eps) {
            r_frac = tmin(30.0f, T.prs_scw[k] / T.prs_sde[k]);
            g_frac = tmin(PS(rime_conversion), 0.15f + (r_frac - 2.0f) * .028f);
            vtboost[k] = tmin(1.5f, 1.1f + (r_frac - 2.f) * .014f);
            T.prg_scw[k] = g_frac * T.prs_scw[k];
            T.png_scw[k] = T.prg_scw[k] * smo0[k] / rs[k];
            vts = PS(av_s) * tempo_pow(xds, PS(bv_s)) * tempo_exp(-PS(fv_s) * xds);
            const_ri = -1.0f * (mvd_c[k] * 0.5e6f) * vts / tmin(-0.1f, tempc);
            const_ri = tmax(0.1f, tmin(const_ri, 10.0f));
            rime_dens = (0.051f + 0.114f * const_ri - 0.0055f * const_ri * const_ri) * 1000.0f;
            if (rime_dens < 150.0f) {
              g_frac = 0.0f;
              T.prg_scw[k] = 0.0f;
              T.png_scw[k] = 0.0f;
            }
            snow_dens_frac = tmin(1.0f, tmax(0.0f, rs[k] * C.global_inverse_dt
                                  / (rs[k] * C.global_inverse_dt + T.prg_scw[k])));
            T.pbg_scw[k] = mtl * T.prg_scw[k]
                         / (PS(rho_s) * snow_dens_frac + rime_dens * (1.0f - snow_dens_frac));
            T.prs_scw[k] = (1.0f - g_frac) * T.prs_scw[k];
          }
        }
      }
    }

    // hail collecting cloud water - assume dc << dh and vtc ~ 0 (main:2736-2750)
    if (l_qh[k] && l_qc[k]) {
      if (rh[k] >= PA1(r_g, 1) && mvd_c[k] > PS(d0c)) {
        xdh = (PS(bm_g) + PS(mu_g) + 1.0f) * ilamh[k];
        if (xdh > PS(d0g)) {
          ef_hw = 0.95f;
          t1_qh_qc = PS(pi) * .25f * PA1(av_g, PI(nrhg)) * PA2(cgg, 9, PI(nrhg));
          n0_h = nh[k] / ilamh[k];
          const double p9 = tempo_pow(ilamh[k], PA2(cge, 9, PI(nrhg)));  // one evaluation, two uses
          T.prh_hcw[k] = rhof[k] * t1_qh_qc * ef_hw * rc[k] * n0_h * p9;
          T.pnc_hcw[k] = rhof[k] * t1_qh_qc * ef_hw * nc[k] * n0_h * p9;
          T.pnc_hcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_hcw[k]);
        }
      }
    }

    if (l_qc[k] && l_qg[k]) {
      if (rg[k] >= PA1(r_g, 1) && mvd_c[k] > PS(d0c)) {
        xdg = (PS(bm_g) + PS(mu_g) + 1.0f) * ilamg[k];
        vtg = rhof[k] * PA1(av_g, idx[k]) * PA2(cgg, 6, idx[k]) * PS(ogg3)
            * tempo_pow(ilamg[k], PA1(bv_g, idx[k]));
        stoke_g = mvd_c[k] * mvd_c[k] * vtg * PS(rho_w) / (9.0f * visco[k] * xdg);
        const_ri = -1.0f * (mvd_c[k] * 0.5e6f) * vtg / tmin(-0.1f, tempc);
        const_ri = tmax(0.1f, tmin(const_ri, 10.0f));
        rime_dens = (0.051f + 0.114f * const_ri - 0.0055f * const_ri * const_ri) * 1000.0f;
        if (xdg > PS(d0g)) {
          // Fortran quirk: ef_gw is undefined when stoke_g is NaN; 0 here.
          ef_gw = 0.0f;
          if (stoke_g >= 0.4f && stoke_g <= 10.0f) {
            ef_gw = 0.55f * tempo_log10(2.51f * stoke_g);
          } else if (stoke_g < 0.4f) {
            ef_gw = 0.0f;
          } else if (stoke_g > 10.0f) {
            ef_gw = 0.77f;
          }
          if (temp[k] > t0) ef_gw = ef_gw * tmin(tmax(tempo_pow(10.0f, (-0.33f * (temp[k] - t0))), 0.1f), 1.0f);
          t1_qg_qc = PS(pi) * .25f * PA1(av_g, idx[k]) * PA2(cgg, 9, idx[k]);
          n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));
          const double p9 = tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));  // one evaluation, two uses
          T.prg_gcw[k] = rhof[k] * t1_qg_qc * ef_gw * rc[k] * n0_g * p9;
          T.pnc_gcw[k] = rhof[k] * t1_qg_qc * ef_gw * nc[k] * n0_g * p9;
          T.pnc_gcw[k] = tmin(nc[k] * C.global_inverse_dt, T.pnc_gcw[k]);
          if (temp[k] < t0) T.pbg_gcw[k] = mtl * T.prg_gcw[k] / rime_dens;

          if (temp[k] < t0) {
            if (T.prg_gcw[k] > eps && tempc > -8.0f) {
              tf = 0.0f;
              if (tempc >= -5.0f && tempc < -3.0f) {
                tf = 0.5f * (-3.0f - tempc);
              } else if (tempc > -8.0f && tempc < -5.0f) {
                tf = 0.33333333f * (8.0f + tempc);
              }
              T.pni_ihm[k] = 3.5e8f * tf * T.prg_gcw[k];
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2604-2692
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2602-2692 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void rain_snow_rain_graupel(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const bool* l_qr, const float* rr, const float* nr,
    const double* ilamr, const bool* l_qs, const float* rs, const bool* l_qg,
    const float* rg, const float* ng, const double* ilamg, const int* idx,
    const bool* l_qh, const float* rh, const double* ilamh, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:2958-3067
  int idx_r, idx_r1, idx_s, idx_t, idx_g, idx_g1, idx_h, idx_h1;
  const float t0 = PS(t0); const float  mtl = PS(meters3_to_liters);
  TEMPO_LEVELS(k, nz) {
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
          T.png_rcs[k] = TQS(tnr_racs1) + TQS(tnr_sacr1);
          T.png_rcs[k] = tmin(nr[k] * C.global_inverse_dt, T.png_rcs[k]);
          T.pbg_rcs[k] = mtl * T.prg_rcs[k] / PA1(rho_g, PI(nrhg));
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
          T.pbg_rcg[k] = mtl * T.prg_rcg[k] / PA1(rho_g, PI(nrhg));
        } else {
          T.prr_rcg[k] = TQG(tcg_racg);
          T.prr_rcg[k] = tmin(rg[k] * C.global_inverse_dt, T.prr_rcg[k]);
          T.prg_rcg[k] = -T.prr_rcg[k];
          T.png_rcg[k] = TQG(tnr_racg);
          T.png_rcg[k] = tmin(ng[k] * C.global_inverse_dt, T.png_rcg[k]);
          T.pbg_rcg[k] = mtl * T.prg_rcg[k] / PA1(rho_g, idx[k]);
          T.pnr_rcg[k] = -1.5f * TQG(tnr_gacr);
        }
#undef TQG
      }
    }

    // rain collecting hail (main:3049-3065)
    if (l_qh[k] && l_qr[k]) {
      if (rh[k] >= PA1(r_g, 1)) {
        if (temp[k] < t0) {
          get_temperature_table_index(P, temp[k], &idx_t);
          get_rain_table_index(P, rr[k], ilamr[k], &idx_r, &idx_r1);
          get_graupel_table_index(P, rh[k], ilamh[k], PI(nrhg), &idx_h, &idx_h1);
#define TQH(tab) PA5(tab, idx_h1, idx_h, PI(nrhg), idx_r1, idx_r)
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2695-2783.  nifa and nwfa are optional (nullptr = absent).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2693-2783 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void ice_nucleation(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const float* rho, const float* w1d, const float* qv,
    const float* qvsi, const float* ssati, const float* ssatw, const float* nwfa1d,
    const float* nifa1d, const float* nwfa, const float* nifa, const float* ni,
    const double* ns, const float* rc, const float* nc, const float* rr, const float* nr,
    const double* ilamr, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:3070-3236 (nifa1d/nwfa1d presence selects the aerosol paths)
  float rate_max; float  tempc; float  xni; float  xnc; float  rime_autoconv_ratio;
  int idx_in, idx_r, idx_r1, idx_tc, idx_c, idx_n;
  const float t0 = PS(t0); const float  xm0i = PS(xm0i);
  TEMPO_LEVELS(k, nz) {
    if (temp[k] < t0) {
      tempc = temp[k] - t0;
      idx_tc = imax(1, imin(tnint(-tempc), 45));
      rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999f;
      if (nifa1d != nullptr) {
        xni = demott_nucleation(P, tempc, rho[k], nifa[k]);
      } else {
        xni = 1.0f * 1000.0f;
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
        // main:3112-3122: fewer frozen drops become graupel as collection outpaces autoconversion
        if (T.prr_wau[k] < PS(eps)) {
          if (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k] > PS(eps)) {
            T.png_rfz[k] = 0.1f * T.pnr_rfz[k];
          }
        } else {
          rime_autoconv_ratio = (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k]) / T.prr_wau[k];
          rime_autoconv_ratio = tmax(tmin(rime_autoconv_ratio, 1.f), 0.f);
          T.png_rfz[k] = T.pnr_rfz[k] * tmax(tempo_pow(10.f, (-2.0f * rime_autoconv_ratio)), 0.1f);
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
        T.pni_wfz[k] = tmin3(nc[k] * C.global_inverse_dt, T.pri_wfz[k] / (2.0f * xm0i), T.pni_wfz[k]);
      } else if (rc[k] > PS(r1) && temp[k] < PS(hgfrz)) {
        T.pri_wfz[k] = rc[k] * C.global_inverse_dt;
        T.pni_wfz[k] = nc[k] * C.global_inverse_dt;
      }

      if ((ssati[k] >= PS(demott_nuc_ssati)) || (ssatw[k] > PS(eps) && tempc < -20.0f)) {  // demott_nuc_tempc
        if (nifa1d != nullptr) {
          xnc = demott_nucleation(P, tempc, rho[k], nifa[k]);
        } else {
          xnc = tmin(PS(icenuc_max), PS(tno) * tempo_exp(PS(ato) * (t0 - temp[k])));
        }
        xni = ni[k] + (T.pni_rfz[k] + T.pni_wfz[k]) * C.global_dt;
        T.pni_inu[k] = 0.5f * (xnc - xni + fabs(xnc - xni)) * C.global_inverse_dt;
        T.pri_inu[k] = tmin(rate_max, xm0i * T.pni_inu[k]);
        T.pni_inu[k] = T.pri_inu[k] / xm0i;
      }

      xni = ns[k] + ni[k] + (T.pni_rfz[k] + T.pni_wfz[k] + T.pni_inu[k]) * C.global_dt;
      if (nwfa1d != nullptr) {
        if ((xni <= PS(max_ni)) && (temp[k] < 238.0f) && (ssati[k] >= 0.4f)) {
          xnc = koop_nucleation(P, C, temp[k], ssatw[k], nwfa[k]);
          T.pni_iha[k] = xnc * C.global_inverse_dt;
          T.pri_iha[k] = tmin(rate_max, xm0i * 0.1f * T.pni_iha[k]);
          T.pni_iha[k] = T.pri_iha[k] / (xm0i * 0.1f);
        }
      }
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2824-2851
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2822-2851 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void get_t1_subl(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* temp, const float* qvsi, const float* tcond,
    const float* diffu, const float* ssati, float* t1_subl) {
  float otemp; float  rvs; float  rvs_p; float  rvs_pp; float  gamsc; float  alphsc; float  xsat;
  const float lsub = PS(lsub); const float  orv = PS(orv);
  (void)C;
  TEMPO_LEVELS(k, nz) {
    otemp = 1.0f / temp[k];
    rvs = rho[k] * qvsi[k];
    rvs_p = rvs * otemp * (lsub * otemp * orv - 1.0f);
    rvs_pp = rvs * (otemp * (lsub * otemp * orv - 1.0f) * otemp * (lsub * otemp * orv - 1.0f)
           + (-2.f * lsub * otemp * otemp * otemp * orv) + otemp * otemp);
    gamsc = lsub * diffu[k] / tcond[k] * rvs_p;
    alphsc = 0.5f * (gamsc / (1.0f + gamsc)) * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
    alphsc = tmax(1.e-9f, alphsc);
    xsat = ssati[k];
    if (fabs(xsat) < 1.e-9f) xsat = 0.0f;
    t1_subl[k] = 4.0f * PS(pi) * (1.0f - alphsc * xsat + 2.0f * alphsc * alphsc * xsat * xsat
               - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2854-2980
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2852-2980 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void ice_processes(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const float* rhof2, const float* rho, const float* w1d,
    const float* temp, const float* qv, const float* qvsi, const float* tcond,
    const float* diffu, const float* vsc2, const float* ssati, const bool* l_qi,
    const float* ri, const float* ni, const double* ilami, const bool* l_qs,
    const float* rs, const double* smoe, const double* smof, const double* smo1,
    const float* rr, const float* nr, const double* ilamr, const float* mvd_r,
    const bool* l_qg, const float* rg, const float* ng, const double* ilamg,
    const int* idx, const bool* l_qh, const float* rh, const float* nh, const double* ilamh,
    TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:3239-3390
  float xdi; float  xmi; float  oxmi; float  c_snow; float  rate_max; float  otemp; float  rvs; float  t2_qg_sd;
  float rime_autoconv_ratio;
  double lamr; double  n0_r; double  n0_g; double  n0_h;
  int idx_i, idx_i1;
  float t1_subl[TEMPO_NZP];
  const float t0 = PS(t0); const float  c_cube = PS(c_cube); const float  c_sqrd = PS(c_sqrd);
  const float bm_i = PS(bm_i); const float  mu_i = PS(mu_i); const float  am_i = PS(am_i); const float  d0i = PS(d0i);
  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0f;
  get_t1_subl(P, C, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl);

  TEMPO_LEVELS(k, nz) {
    otemp = 1.0f / temp[k];
    rvs = rho[k] * qvsi[k];
    rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999f;

    if (temp[k] < t0) {
      if (l_qi[k]) {
        get_ice_table_index(P, ri[k], ni[k], &idx_i, &idx_i1);
        xdi = tmax(d0i, (bm_i + mu_i + 1.f) * ilami[k]);
        xmi = am_i * tempo_pow(xdi, bm_i);
        oxmi = 1.0f / xmi;
        T.pri_ide[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * PS(oig1) * PA1(cig, 5)
                     * ni[k] * ilami[k];
        if (T.pri_ide[k] < 0.0f) {
          T.pri_ide[k] = tmax3(-ri[k] * C.global_inverse_dt, T.pri_ide[k], rate_max);
          T.pni_ide[k] = T.pri_ide[k] * oxmi;
          T.pni_ide[k] = tmax(-ni[k] * C.global_inverse_dt, T.pni_ide[k]);
        } else {
          T.pri_ide[k] = tmin(T.pri_ide[k], rate_max);
          T.prs_ide[k] = (1.0f - PA2(tpi_ide, idx_i, idx_i1)) * T.pri_ide[k];
          T.pri_ide[k] = PA2(tpi_ide, idx_i, idx_i1) * T.pri_ide[k];
        }

        if ((idx_i == PI(ntb_i)) || (xdi > 5.0f * PS(d0s))) {
          T.prs_iau[k] = ri[k] * .99f * C.global_inverse_dt;
          T.pni_iau[k] = ni[k] * .95f * C.global_inverse_dt;
        } else if (xdi < 0.1f * PS(d0s)) {
          T.prs_iau[k] = 0.0f;
          T.pni_iau[k] = 0.0f;
        } else {
          T.prs_iau[k] = PA2(tps_iaus, idx_i, idx_i1) * C.global_inverse_dt;
          T.prs_iau[k] = tmin(ri[k] * .99f * C.global_inverse_dt, T.prs_iau[k]);
          T.pni_iau[k] = PA2(tni_iaus, idx_i, idx_i1) * C.global_inverse_dt;
          T.pni_iau[k] = tmin(ni[k] * .95f * C.global_inverse_dt, T.pni_iau[k]);
        }

        xdi = tmax(d0i, (bm_i + mu_i + 1.f) * ilami[k]);
        xmi = am_i * tempo_pow(xdi, bm_i);
        oxmi = 1.f / xmi;
        if (rs[k] >= PA1(r_s, 1)) {
          T.prs_sci[k] = PS(t1_qs_qi) * rhof[k] * PS(ef_si) * ri[k] * smoe[k];
          T.pni_sci[k] = T.prs_sci[k] * oxmi;
        }

        if (rr[k] >= PA1(r_r, 1) && mvd_r[k] > 4.0f * xdi) {
          lamr = 1.0f / ilamr[k];
          n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
          const TempoPowBase bf = tempo_pow_base(lamr + PS(fv_r));  // shared log: -cre9 (twice), -cre8
          const double p9 = tempo_pow_with(bf, (-PA1(cre, 9)));
          T.pri_rci[k] = rhof[k] * PS(t1_qr_qi) * PS(ef_ri) * ri[k] * n0_r
                       * (p9);
          T.pnr_rci[k] = rhof[k] * PS(t1_qr_qi) * PS(ef_ri) * ni[k] * n0_r
                       * (p9);
          T.pnr_rci[k] = tmin(nr[k] * C.global_inverse_dt, T.pnr_rci[k]);
          T.png_rci[k] = T.pnr_rci[k];
          // main:3324-3334
          if (T.prr_wau[k] < PS(eps)) {
            if (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k] > PS(eps)) {
              T.png_rci[k] = 0.1f * T.pnr_rci[k];
            }
          } else {
            rime_autoconv_ratio = (T.prh_rch[k] + T.prg_rcg[k] + T.prg_rcs[k]) / T.prr_wau[k];
            rime_autoconv_ratio = tmax(tmin(rime_autoconv_ratio, 1.f), 0.f);
            T.png_rci[k] = T.pnr_rci[k] * tmax(tempo_pow(10.f, (-2.0f * rime_autoconv_ratio)), 0.1f);
          }
          T.pni_rci[k] = T.pri_rci[k] * oxmi;
          T.prr_rci[k] = rhof[k] * PS(t2_qr_qi) * PS(ef_ri) * ni[k] * n0_r
                       * (tempo_pow_with(bf, (-PA1(cre, 8))));
          T.prr_rci[k] = tmin(rr[k] * C.global_inverse_dt, T.prr_rci[k]);
          T.prg_rci[k] = T.pri_rci[k] + T.prr_rci[k];
          T.pbg_rci[k] = PS(meters3_to_liters) * T.prg_rci[k] / PA1(rho_g, PI(nrhg));  // main:3340
        }
      }

      if (l_qs[k]) {
        c_snow = c_sqrd + (temp[k] - t0 + 1.5f) * (c_cube - c_sqrd) / (-30.0f + 1.5f);
        c_snow = tmax(c_sqrd, tmin(c_snow, c_cube));
        T.prs_sde[k] = c_snow * t1_subl[k] * diffu[k] * ssati[k] * rvs
                     * (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);
        if (T.prs_sde[k] < 0.0f) {
          T.prs_sde[k] = tmax3(-rs[k] * C.global_inverse_dt, T.prs_sde[k], rate_max);
        } else {
          T.prs_sde[k] = tmin(T.prs_sde[k], rate_max);
        }
      }
      if (l_qg[k]) {
        if (ssati[k] < -PS(eps)) {
          n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));
          t2_qg_sd = 0.28f * PS(sc3) * sqrt(PA1(av_g, idx[k])) * PA2(cgg, 11, idx[k]);
          const TempoPowBase bg = tempo_pow_base(ilamg[k]);  // shared log: cge(10,1), cge(11,idx)
          T.prg_gde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_g
                       * (PS(t1_qg_sd) * tempo_pow_with(bg, PA2(cge, 10, 1))
                          + t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow_with(bg, PA2(cge, 11, idx[k])));
          if (T.prg_gde[k] < 0.0f) {
            T.prg_gde[k] = tmax3(-rg[k] * C.global_inverse_dt, T.prg_gde[k], rate_max);
            T.png_gde[k] = T.prg_gde[k] * ng[k] / rg[k];
          } else {
            T.prg_gde[k] = tmin(T.prg_gde[k], rate_max);
          }
        }
      }
      // hail sublimation (main:3372-3386)
      if (l_qh[k]) {
        if (ssati[k] < -PS(eps)) {
          t2_qg_sd = 0.28f * PS(sc3) * sqrt(PA1(av_g, PI(nrhg))) * PA2(cgg, 11, PI(nrhg));
          n0_h = nh[k] / ilamh[k];
          const TempoPowBase bh = tempo_pow_base(ilamh[k]);  // shared log: cge(10,1), cge(11,nrhg)
          T.prh_hde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_h
                       * (PS(t1_qg_sd) * tempo_pow_with(bh, PA2(cge, 10, 1))
                          + t2_qg_sd * vsc2[k] * rhof2[k] * tempo_pow_with(bh, PA2(cge, 11, PI(nrhg))));
          if (T.prh_hde[k] < 0.0f) {
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2983-3080
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2981-3080 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void melting(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof2, const float* rho, const float* temp, const float* qvsi,
    const float* tcond, const float* diffu, const float* vsc2, const float* ssati,
    const float* delqvs, const bool* l_qs, const float* rs, const double* smof,
    const double* smo0, const double* smo1, const bool* l_qg, const float* rg,
    const float* ng, const double* ilamg, const int* idx, const bool* l_qh,
    const float* rh, const float* nh, const double* ilamh, const float* hail_fraction,
    int hail_fraction_flag, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:3393-3516
  float tempc; float  otemp; float  rvs; float  melt_f; float  t2_qg_me; float  t2_qg_sd; float  t2_qh_me;
  float melt_constant_1; float  melt_constant_2;
  double n0_g; double  n0_melt; double  lamg; double  n0_h;
  float t1_subl[TEMPO_NZP];
  const float t0 = PS(t0); const float  c_cube = PS(c_cube); const float  olfus = PS(olfus); const float  lvap0 = PS(lvap0);
  const float pi = PS(pi); const float  mtl = PS(meters3_to_liters);
  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0f;
  get_t1_subl(P, C, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl);

  TEMPO_LEVELS(k, nz) {
    otemp = 1.0f / temp[k];
    tempc = temp[k] - t0;
    rvs = rho[k] * qvsi[k];

    if (temp[k] > t0) {
      if (l_qs[k]) {
        T.prr_sml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k])
                     * (PS(t1_qs_me) * smo1[k] + PS(t2_qs_me) * rhof2[k] * vsc2[k] * smof[k]);

        if (T.prr_sml[k] > 0.0f) {
          T.prr_sml[k] = T.prr_sml[k] + 4218.0f * olfus * tempc * (T.prr_rcs[k] + T.prs_scw[k]);
          T.prr_sml[k] = tmin(rs[k] * C.global_inverse_dt, tmax(0.0f, T.prr_sml[k]));
          T.pnr_sml[k] = smo0[k] / rs[k] * T.prr_sml[k] * tempo_pow(10.0f, (-0.25f * tempc));
          T.pnr_sml[k] = tmin(smo0[k] * C.global_inverse_dt, T.pnr_sml[k]);
        } else {
          T.prr_sml[k] = 0.0f;
          T.pnr_sml[k] = 0.0f;
          if (ssati[k] < 0.0f) {
            T.prs_sde[k] = PS(c_sqrd) * t1_subl[k] * diffu[k] * ssati[k] * rvs
                         * (PS(t1_qs_sd) * smo1[k] + PS(t2_qs_sd) * rhof2[k] * vsc2[k] * smof[k]);
            T.prs_sde[k] = tmax(-rs[k] * C.global_inverse_dt, T.prs_sde[k]);
          }
        }
      }

      if (l_qg[k]) {
        const double p2 = tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));  // one evaluation, two uses
        n0_g = ng[k] * PS(ogg2) * p2;
        n0_melt = ng[k] * PS(ogg2) * p2;
        // The (ilamg; cge(10,1)) and (ilamg; cge(11,idx)) pows of both branches below: one shared log, each once.
        const TempoPowBase bg = tempo_pow_base(ilamg[k]);
        const double p10 = tempo_pow_with(bg, PA2(cge, 10, 1)), p11 = tempo_pow_with(bg, PA2(cge, 11, idx[k]));
        if ((rg[k] * ng[k]) < 1.e-4f) {
          lamg = 1.f / ilamg[k];
          n0_melt = (1.e-4f / rg[k]) * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));
        }
        t2_qg_me = pi * 4.0f * c_cube * olfus * 0.28f * PS(sc3) * sqrt(PA1(av_g, idx[k]))
                 * PA2(cgg, 11, idx[k]);
        T.prr_gml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k]) * n0_melt
                     * (PS(t1_qg_me) * p10
                        + t2_qg_me * rhof2[k] * vsc2[k] * p11);
        T.prr_gml[k] = tmin(rg[k] * C.global_inverse_dt, tmax(0.0f, T.prr_gml[k]));
        if (T.prr_gml[k] > 0.0f) {
          melt_f = tmax(0.05f, tmin(T.prr_gml[k] * C.global_dt / rg[k], 1.0f));
          T.pbg_gml[k] = mtl * T.prr_gml[k]
                       / tmax(tmin(melt_f * PA1(rho_g, idx[k]), PA1(rho_g, PI(nrhg))), PA1(rho_g, 1));
          // main:3467-3473: logistic number loss (the old curve's form when hail_fraction = 0)
          melt_constant_1 = tmax(tmin(5.0f * tempo_pow((hail_fraction[k] / hex_divisor_3e800000), .33f), 5.0f), 0.0f);
          if (!hail_fraction_flag) melt_constant_1 = 0.0f;
          melt_constant_2 = (1.0f + tempo_exp(-1.215f * melt_constant_1))
                          / (1.0f + tempo_exp(1.215f * (temp[k] - t0 - melt_constant_1)));
          T.pnr_gml[k] = T.prr_gml[k] * ng[k] / rg[k] * tmax(tmin(melt_constant_2, 1.0f), 0.0f);
        } else {
          T.prr_gml[k] = 0.0f;
          T.pnr_gml[k] = 0.0f;
          T.pbg_gml[k] = 0.0f;
          if (ssati[k] < 0.0f) {
            t2_qg_sd = 0.28f * PS(sc3) * sqrt(PA1(av_g, idx[k])) * PA2(cgg, 11, idx[k]);
            T.prg_gde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs * n0_g
                         * (PS(t1_qg_sd) * p10
                            + t2_qg_sd * vsc2[k] * rhof2[k] * p11);
            T.prg_gde[k] = tmax(-rg[k] * C.global_inverse_dt, T.prg_gde[k]);
            T.png_gde[k] = T.prg_gde[k] * ng[k] / rg[k];
          }
        }
      }
      if (C.global_dt > PS(timestep_conversion_rime_to_rain)) {
        T.prr_rcw[k] = T.prr_rcw[k] + T.prs_scw[k] + T.prg_gcw[k];
        T.prs_scw[k] = 0.0f;
        T.prg_gcw[k] = 0.0f;
      }

      // hail melting (main:3500-3513)
      if (l_qh[k]) {
        t2_qh_me = pi * 4.0f * c_cube * olfus * 0.28f * PS(sc3) * sqrt(PA1(av_g, PI(nrhg))) * PA2(cgg, 11, PI(nrhg));
        n0_h = nh[k] / ilamh[k];
        const TempoPowBase bh = tempo_pow_base(ilamh[k]);  // shared log: cge(10,1), cge(11,nrhg)
        T.prr_hml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k])
                     * n0_h * (PS(t1_qg_me) * tempo_pow_with(bh, PA2(cge, 10, 1))
                               + t2_qh_me * rhof2[k] * vsc2[k] * tempo_pow_with(bh, PA2(cge, 11, PI(nrhg))));
        T.prr_hml[k] = tmin(rh[k] * C.global_inverse_dt, tmax(0.0f, T.prr_hml[k]));
        if (T.prr_hml[k] > 0.0f) {
          T.pnr_hml[k] = T.prr_hml[k] * nh[k] / rh[k];
        } else {
          T.prr_hml[k] = 0.0f;
          T.pnr_hml[k] = 0.0f;
        }
      }
    }
  }
  (void)otemp;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:3083-3153
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:3081-3153 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void aerosol_scavenging(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const float* rho, const float* rhof, const float* visco,
    const float* nwfa, const float* nifa, const bool* l_qr, const float* nr,
    const double* ilamr, const float* mvd_r, const bool* l_qs, const float* rs,
    const double* smob, const double* smoc, const double* smoe, const bool* l_qg,
    const float* rg, const float* ng, const double* ilamg, const int* idx, TempoTend& T) {
  float ef_ra; float  ef_sa; float  ef_ga; float  t1_qg_qc;
  double n0_r; double  xds; double  xdg; double  n0_g; double  lamr;
  const float wf_aerosol_size = 0.04e-6f;
  const float if_aerosol_size = 0.8e-6f;
  TEMPO_LEVELS(k, nz) {
    if (l_qr[k] && mvd_r[k] > PS(d0r)) {
      ef_ra = aerosol_collection_efficiency(P, mvd_r[k], wf_aerosol_size, visco[k], rho[k], temp[k], 'r');
      lamr = 1.0f / ilamr[k];
      n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
      const double p9 = tempo_pow((lamr + PS(fv_r)), (-PA1(cre, 9)));  // one evaluation, two uses
      T.pna_rca[k] = rhof[k] * PS(t1_qr_qc) * ef_ra * nwfa[k] * n0_r
                   * (p9);
      T.pna_rca[k] = tmin(nwfa[k] * C.global_inverse_dt, T.pna_rca[k]);
      ef_ra = aerosol_collection_efficiency(P, mvd_r[k], if_aerosol_size, visco[k], rho[k], temp[k], 'r');
      T.pnd_rcd[k] = rhof[k] * PS(t1_qr_qc) * ef_ra * nifa[k] * n0_r
                   * (p9);
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
      xdg = (PS(bm_g) + PS(mu_g) + 1.0f) * ilamg[k];
      ef_ga = aerosol_collection_efficiency(P, xdg, wf_aerosol_size, visco[k], rho[k], temp[k], 'g');
      t1_qg_qc = PS(pi) * .25f * PA1(av_g, idx[k]) * PA2(cgg, 9, idx[k]);
      n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));
      const double pg9 = tempo_pow(ilamg[k], PA2(cge, 9, idx[k]));  // one evaluation, two uses
      T.pna_gca[k] = rhof[k] * t1_qg_qc * ef_ga * nwfa[k] * n0_g * pg9;
      T.pna_gca[k] = tmin(nwfa[k] * C.global_inverse_dt, T.pna_gca[k]);
      ef_ga = aerosol_collection_efficiency(P, xdg, if_aerosol_size, visco[k], rho[k], temp[k], 'g');
      T.pnd_gcd[k] = rhof[k] * t1_qg_qc * ef_ga * nifa[k] * n0_g * pg9;
      T.pnd_gcd[k] = tmin(nifa[k] * C.global_inverse_dt, T.pnd_gcd[k]);
    }
  }
}

// TEMPO microphysics port for WOOF: diagnostics.
// Line-by-line transliteration of NCAR/TEMPO 9adf9ef (tag tempo_v3.0.3)
//   src/module_mp_tempo_diags.F90 (whole module)
// Apache-2.0, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
// Conventions: see tempo_common.cuh.  Column arrays are 1-based (a[1..nz]).

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
TEMPO_HD double tc_abs(tcplx a) { return mpas_hypot(a.re, a.im); }
TEMPO_HD tcplx tc_conj(tcplx a) { return tc_make(a.re, -a.im); }
TEMPO_HD tcplx tc_sqrt(tcplx a) {
  double r = mpas_hypot(a.re, a.im);
  if (r == 0.0) return tc_make(0.0, 0.0);
  if (a.re >= 0.0) {
    double t = sqrt(0.5 * (r + a.re));
    return tc_make(t, a.im / (2.0 * t));
  }
  double t = sqrt(0.5 * (r - a.re));
  return tc_make(fabs(a.im) / (2.0 * t), copysign(t, a.im));
}
TEMPO_HD tcplx tc_log(tcplx a) { return tc_make(tempo_log(mpas_hypot(a.re, a.im)), mpas_atan2(a.im, a.re)); }

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:13-56
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:12-56 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void effective_radius(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const bool* l_qc, const float* nc, const double* ilamc,
    const bool* l_qi, const double* ilami, const bool* l_qs, const float* rs,
    float* re_qc, float* re_qi, float* re_qs) {
  double smob; double  smoc;
  float tc0;
  int nu_c;
  (void)C;
  TEMPO_LEVELS(k, nz) {
    re_qc[k] = 0.0f;
    re_qi[k] = 0.0f;
    re_qs[k] = 0.0f;
    if (l_qc[k]) {
      nu_c = get_nuc(P, nc[k]);
      re_qc[k] = tmax(2.51e-6f, tmin(float(0.5 * (3.0 + double(nu_c)) * ilamc[k]), 50.e-6f));
    }
    if (l_qi[k]) {
      re_qi[k] = tmax(2.51e-6f, tmin(float(0.5 * (3.0 + double(PS(mu_i))) * ilami[k]), 125.e-6f));
    }
    if (l_qs[k]) {
      tc0 = tmin(-0.1f, temp[k] - PS(t0));
      snow_moments(P, rs[k], tc0, &smob, &smoc, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr);
      re_qs[k] = tmax(5.01e-6f, tmin(0.5f * (smoc / smob), 999.e-6f));
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:124-141
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:122-141 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline int find_melting_level(const TempoParams& P, int nz, const float* temp,
                                        const bool* l_qr, const bool* l_qs, const bool* l_qg) {
  int k_melt = 1;
#if TEMPO_WARP
  // warp: the serial scan stops at the highest qualifying level; reads k+1.
  TEMPO_SYNC();
  int found = 0;
  TEMPO_LEVELS(k, nz - 1) {
    if ((temp[k] > PS(t0)) && l_qr[k] && (l_qs[k + 1] || l_qg[k + 1])) found = k;
  }
  found = tempo_warp_max(found);
  if (found > 0) k_melt = imax(found + 1, k_melt);
  TEMPO_SYNC();
#else
  for (int k = nz - 1; k >= 1; --k) {
    if ((temp[k] > PS(t0)) && l_qr[k] && (l_qs[k + 1] || l_qg[k + 1])) {
      k_melt = imax(k + 1, k_melt);
      break;
    }
  }
#endif
  return k_melt;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:144-172
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:142-172 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline tcplx complex_water_ray(const TempoParams& P, double lambda, double t) {
  double epsinf; double  epss; double  epsr; double  epsi; double  alpha; double  lambdas; double  nenner;
  const float pi = PS(pi);
  epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t;
  epss = 78.54 * (1.0 - 4.579e-3 * (t - 25.0) + 1.190e-5 * (t - 25.0) * (t - 25.0)
                  - 2.800e-8 * (t - 25.0) * (t - 25.0) * (t - 25.0));
  alpha = -16.8129 / (t + 273.16) + 0.0609265;
  lambdas = 0.00033836 * tempo_exp(2513.98 / (t + 273.16)) * 1e-2;

  nenner = 1.0 + 2.0 * tempo_pow((lambdas / lambda), (1.0 - alpha)) * mpas_sin(alpha * pi * 0.5)
         + tempo_pow((lambdas / lambda), (2.0 - 2.0 * alpha));
  epsr = epsinf + ((epss - epsinf) * (tempo_pow((lambdas / lambda), (1.0 - alpha)) * mpas_sin(alpha * pi * 0.5)
                                      + 1.0)) / nenner;
  epsi = ((epss - epsinf) * (tempo_pow((lambdas / lambda), (1.0 - alpha)) * mpas_cos(alpha * pi * 0.5) + 0.0))
       / nenner + lambda * 1.25664 / 1.88496;

  return tc_sqrt(tc_make(epsr, -epsi));
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:175-201
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:173-201 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline tcplx complex_ice_maetzler(double lambda, double t) {
  double f; double  c; double  tk; double  b1; double  b2; double  b; double  deltabeta; double  betam; double  beta; double  theta; double  alfa; double  e;
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:309-395
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:307-395 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void rayleigh_soak_wetgraupel(const TempoParams& P, double x_g, double a_geo,
    double b_geo, double fmelt, double lambda_radar, double meltratio_outside, tcplx m_w,
    tcplx m_i, double* backscatter) {
  double fm; double  mra; double  x_w; double  d_g; double  vg; double  rhog; double  meltratio_outside_grenz; double  volg; double  fmgrenz; double  d_large; double 
      volwater; double  volice; double  volair; double  volmix1; double  volmix2; double  vol1; double  vol2; double  vol3;
  tcplx m_core, m_air, m_tmp, m1t, m2t, m3t, beta2, beta3, num, den;
  const float pi = PS(pi);

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

// Simpson weights shared by the two binned melting routines (MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:223-230, 273-280).
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

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:204-252
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:202-252 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline float reflectivity_from_melting_graupel(const TempoParams& P, float rg,
    float ng, double ilamg, int idx, float rr) {
  const double melt_outside = 0.9;
  const double lambda_radar = 0.10;
  const int radar_bins = PI(radar_bins);
  tcplx m_w_0, m_i_0;
  double simpson[TEMPO_RADAR_BINS_MAX + 2];
  double sr; double  fmelt; double  eta; double  lamg; double  backscatter; double  f_d; double  n0_g; double  mass; double  k_w;
  const float pi = PS(pi);

  tempo_simpson(radar_bins, simpson);
  m_w_0 = complex_water_ray(P, lambda_radar, 0.0);
  m_i_0 = complex_ice_maetzler(lambda_radar, 0.0);
  tcplx mw2 = tc_mul(m_w_0, m_w_0);
  k_w = tpowi(tc_abs(tc_div(tc_addr(mw2, -1.0), tc_addr(mw2, 2.0))), 2);

  sr = tmax(0.01, tmin(1.0 - rg / tmax(rg + rr, PS(r1)), 0.99));  // diags:242
  fmelt = sr * sr;
  eta = 0.0;
  lamg = 1.0 / ilamg;
  n0_g = ng * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));

  for (int n = 1; n <= radar_bins; ++n) {
    mass = PA1(am_g, idx) * tpowi(PA1(gbins_radar, n), 3);
    rayleigh_soak_wetgraupel(P, mass, PA1(ocmg, idx), PS(obmg), fmelt, lambda_radar,
                             melt_outside, m_w_0, m_i_0, &backscatter);
    f_d = n0_g * tpowi(PA1(gbins_radar, n), 0) * tempo_exp(-lamg * PA1(gbins_radar, n));
    eta = eta + f_d * backscatter * simpson[n] * PA1(dgbins_radar, n);
  }
  return lambda_radar * lambda_radar * lambda_radar * lambda_radar / (pi * pi * pi * pi * pi * k_w)
       * eta;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:255-306
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:253-306 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline float reflectivity_from_melting_snow(const TempoParams& P, float rs,
    double smob, double smoc, float rr) {
  const double melt_outside = 0.9;
  const double lambda_radar = 0.10;
  const int radar_bins = PI(radar_bins);
  tcplx m_w_0, m_i_0;
  double simpson[TEMPO_RADAR_BINS_MAX + 2];
  double sr; double  fmelt; double  eta; double  backscatter; double  f_d; double  mass; double  k_w; double  om3; double  m0; double  mrat; double  slam1; double  slam2;
  const float pi = PS(pi);

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
    mass = PS(am_s) * tpowi(PA1(sbins_radar, n), 2);
    rayleigh_soak_wetgraupel(P, mass, PS(ocms), PS(obms), fmelt, lambda_radar, melt_outside,
                             m_w_0, m_i_0, &backscatter);
    f_d = mrat * (PS(kap0) * tempo_exp(-slam1 * PA1(sbins_radar, n))
                  + PS(kap1) * tempo_pow((m0 * PA1(sbins_radar, n)), PS(mu_s)) * tempo_exp(-slam2 * PA1(sbins_radar, n)));
    eta = eta + f_d * backscatter * simpson[n] * PA1(dsbins_radar, n);
  }
  return lambda_radar * lambda_radar * lambda_radar * lambda_radar / (pi * pi * pi * pi * pi * k_w)
       * eta;
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:59-121
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:57-121 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void reflectivity_10cm(const TempoParams& P, const TempoCtx& C, int nz,
    int refl10cm_from_melting_flag, const float* temp, const bool* l_qr, const float* rr,
    const float* nr, const double* ilamr, const bool* l_qs, const float* rs,
    const double* smoc, const double* smob, const double* smoz, const bool* l_qg,
    const float* rg, const float* ng, const int* idx, const double* ilamg, const bool* l_qh,
    const float* nh, const double* ilamh, float* dbz) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_diags.F90:59-127
  float ze_rain; float  ze_snow; float  ze_graupel; float  ze_hail;
  double n0_r; double  lamr; double  n0_g; double  n0_h;
  int k_melt = 1;
  const float pi = PS(pi); const float  am_s = PS(am_s);
  (void)C;

  if (refl10cm_from_melting_flag) {
    k_melt = find_melting_level(P, nz, temp, l_qr, l_qs, l_qg);
  }

  TEMPO_LEVELS_DOWN(k, nz) {
    ze_rain = 0.75e-22f;
    ze_snow = 0.75e-22f;
    ze_graupel = 0.75e-22f;
    ze_hail = 0.75e-22f;
    dbz[k] = -35.0f;

    if (l_qr[k]) {
      lamr = 1.0f / ilamr[k];
      n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
      ze_rain = n0_r * PA1(crg, 4) * tempo_pow(ilamr[k], PA1(cre, 4));
    }
    if (l_qs[k]) {
      ze_snow = (0.176f / hex_divisor_3f6e147b) * (6.0f / pi) * (6.0f / pi) * (am_s / hex_divisor_44610000) * (am_s / hex_divisor_44610000) * smoz[k];
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_snow = reflectivity_from_melting_snow(P, rs[k], smob[k], smoc[k], rr[k]);
        }
      }
    }
    if (l_qg[k]) {
      n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));
      ze_graupel = (0.176f / hex_divisor_3f6e147b) * (6.0f / pi) * (6.0f / pi) * (PA1(am_g, idx[k]) / hex_divisor_44610000)
                 * (PA1(am_g, idx[k]) / hex_divisor_44610000) * n0_g * PA2(cgg, 4, 1) * tempo_pow(ilamg[k], PA2(cge, 4, 1));
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_graupel = reflectivity_from_melting_graupel(P, rg[k], ng[k], ilamg[k], idx[k], rr[k]);
        }
      }
    }
    if (l_qh[k]) {
      n0_h = nh[k] / ilamh[k];
      ze_hail = (0.176f / hex_divisor_3f6e147b) * (6.0f / pi) * (6.0f / pi) * (PA1(am_g, PI(nrhg)) / hex_divisor_44610000)
              * (PA1(am_g, PI(nrhg)) / hex_divisor_44610000) * n0_h * PA2(cgg, 4, 1) * tempo_pow(ilamh[k], PA2(cge, 4, 1));
    }
    dbz[k] = tmax(-35.0f, 10.0f * (float)tempo_log10((ze_rain + ze_snow + ze_graupel + ze_hail) * 1.e18));
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:398-438
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:396-438 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void max_hail_diam(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* rg, const float* ng, const double* ilamg,
    const int* idx, float* max_hail_diameter) {
  double lamg; double  n0_g; double  sum_nh; double  sum_t; double  f_d; double  hail_max;
  int n;
  const double threshold_conc = double(0.0005f);
  const int nhbins = PI(nhbins);
  (void)C;
  TEMPO_LEVELS(k, nz) {
    max_hail_diameter[k] = 0.0f;
    if (rg[k] / rho[k] >= 1.e-6f) {
      if (PA1(rho_g, idx[k]) < 350.0f) continue;
      lamg = 1.0f / ilamg[k];
      n0_g = ng[k] * PS(ogg2) * tempo_pow(lamg, PA2(cge, 2, 1));

      sum_nh = 0.0f;
      sum_t = 0.0f;
      // Fortran DO: without an exit n ends at 0.
      for (n = nhbins; n >= 1; --n) {
        f_d = n0_g * tpowi(PA1(hbins, n), 0) * tempo_exp(-lamg * PA1(hbins, n)) * PA1(dhbins, n);
        sum_nh = sum_nh + f_d;
        if (sum_nh > threshold_conc) break;
        sum_t = sum_nh;
      }
      if (n >= nhbins) {
        hail_max = PA1(hbins, nhbins);
      } else if (PA1(hbins, n + 1) > 1.e-3f) {
        // Fortran quirk: n = 0 here would read hbins(0); unreachable because
        // hbins(1) is about 0.53 mm, below the 1 mm test.
        hail_max = PA1(hbins, n) - (sum_nh - threshold_conc) / (sum_nh - sum_t)
                 * (PA1(hbins, n) - PA1(hbins, n + 1));
      } else {
        hail_max = 1.e-4f;
      }
      max_hail_diameter[k] = 1000.0f * hail_max;
    }
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:441-456.  cloud_precip is optional in the Fortran; tempo_main always
// passes it, so it is taken by value here.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:439-456 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void freezing_rain(const TempoParams& P, float temp, float rain_precip,
                                    float cloud_precip, float* frz_rain) {
  *frz_rain = 0.0f;
  if ((temp - PS(t0)) < -0.5f) {
    *frz_rain = rain_precip;
    *frz_rain = *frz_rain + cloud_precip;
  }
}

// TEMPO column driver: port of module_mp_tempo_main.F90 lines 74-1955
// (tempo_main and the check/update, thermodynamic, depletion, tendency-sum,
// sedimentation and fallspeed routines) at TEMPO 9adf9ef (tempo_v3.0.3).
// Conventions: see tempo_common.cuh.

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
  WA(ilamc) WA(ilami) WA(ilamr) WA(ilamg) WA(mvd_r) WA(mvd_c) WA(mvd_g) \
  WA(smob) WA(smo2) WA(smo1) WA(smo0) WA(smoc) WA(smoe) WA(smof) WA(smog) WA(ns) WA(smoz) \
  WA(xrx) WA(xnx) WA(xncx) WA(xngx) WA(xqbx) WA(ncsave) \
  WA(vtrr) WA(vtnr) WA(vtrs) WA(vtri) WA(vtni) WA(vtrg) WA(vtng) WA(vtrc) WA(vtnc) WA(vtboost) \
  WA(refl10cm) WA(re_cloud) WA(re_ice) WA(re_snow) WA(max_hail_diameter) \
  WA(rain_med_vol_diam) WA(graupel_med_vol_diam) WA(cloud_number_mixing_ratio) \
  WA(sl_zi) WA(sl_wi) WA(sl_za) WA(sl_dza) WA(sl_qa) WA(sl_qmi) WA(sl_qpi) \
  WA(sl_net_flx) WA(sl_precip_flx) WA(sl_rr_save) WA(sed_r) \
  WA(qhten) WA(rh) WA(nh) WA(qh1d) WA(ilamh) WA(vtrh) WA(hail_fraction)
float tten[TEMPO_NZP];
float qvten[TEMPO_NZP];
float qcten[TEMPO_NZP];
float qiten[TEMPO_NZP];
float qrten[TEMPO_NZP];
float qsten[TEMPO_NZP];
float qgten[TEMPO_NZP];
float qbten[TEMPO_NZP];
float niten[TEMPO_NZP];
float nrten[TEMPO_NZP];
float ncten[TEMPO_NZP];
float ngten[TEMPO_NZP];
float nwfaten[TEMPO_NZP];
float nifaten[TEMPO_NZP];
float temp[TEMPO_NZP];
float pres[TEMPO_NZP];
float qv[TEMPO_NZP];
float rho[TEMPO_NZP];
float rhof[TEMPO_NZP];
float rhof2[TEMPO_NZP];
float qvs[TEMPO_NZP];
float qvsi[TEMPO_NZP];
float delqvs[TEMPO_NZP];
float satw[TEMPO_NZP];
float sati[TEMPO_NZP];
float ssatw[TEMPO_NZP];
float ssati[TEMPO_NZP];
float diffu[TEMPO_NZP];
float visco[TEMPO_NZP];
float vsc2[TEMPO_NZP];
float tcond[TEMPO_NZP];
float lvap[TEMPO_NZP];
float ocp[TEMPO_NZP];
float lvt2[TEMPO_NZP];
float rc[TEMPO_NZP];
float ri[TEMPO_NZP];
float rr[TEMPO_NZP];
float rs[TEMPO_NZP];
float rg[TEMPO_NZP];
float rb[TEMPO_NZP];
float ni[TEMPO_NZP];
float nr[TEMPO_NZP];
float nc[TEMPO_NZP];
float ng[TEMPO_NZP];
float nwfa[TEMPO_NZP];
float nifa[TEMPO_NZP];
double ilamc[TEMPO_NZP];
double ilami[TEMPO_NZP];
double ilamr[TEMPO_NZP];
double ilamg[TEMPO_NZP];
float mvd_r[TEMPO_NZP];
float mvd_c[TEMPO_NZP];
float mvd_g[TEMPO_NZP];
double smob[TEMPO_NZP];
double smo2[TEMPO_NZP];
double smo1[TEMPO_NZP];
double smo0[TEMPO_NZP];
double smoc[TEMPO_NZP];
double smoe[TEMPO_NZP];
double smof[TEMPO_NZP];
double smog[TEMPO_NZP];
double ns[TEMPO_NZP];
double smoz[TEMPO_NZP];
float xrx[TEMPO_NZP];
float xnx[TEMPO_NZP];
float xncx[TEMPO_NZP];
float xngx[TEMPO_NZP];
float xqbx[TEMPO_NZP];
float ncsave[TEMPO_NZP];
float vtrr[TEMPO_NZP];
float vtnr[TEMPO_NZP];
float vtrs[TEMPO_NZP];
float vtri[TEMPO_NZP];
float vtni[TEMPO_NZP];
float vtrg[TEMPO_NZP];
float vtng[TEMPO_NZP];
float vtrc[TEMPO_NZP];
float vtnc[TEMPO_NZP];
float vtboost[TEMPO_NZP];
float refl10cm[TEMPO_NZP];
float re_cloud[TEMPO_NZP];
float re_ice[TEMPO_NZP];
float re_snow[TEMPO_NZP];
float max_hail_diameter[TEMPO_NZP];
float rain_med_vol_diam[TEMPO_NZP];
float graupel_med_vol_diam[TEMPO_NZP];
float cloud_number_mixing_ratio[TEMPO_NZP];
float sl_zi[TEMPO_NZP];
float sl_wi[TEMPO_NZP];
float sl_za[TEMPO_NZP];
float sl_dza[TEMPO_NZP];
float sl_qa[TEMPO_NZP];
float sl_qmi[TEMPO_NZP];
float sl_qpi[TEMPO_NZP];
float sl_net_flx[TEMPO_NZP];
float sl_precip_flx[TEMPO_NZP];
float sl_rr_save[TEMPO_NZP];
float sed_r[TEMPO_NZP];
float qhten[TEMPO_NZP];
float rh[TEMPO_NZP];
float nh[TEMPO_NZP];
float qh1d[TEMPO_NZP];
double ilamh[TEMPO_NZP];
float vtrh[TEMPO_NZP];
float hail_fraction[TEMPO_NZP];
#define TEMPO_WA_MEMBER(name)
  TEMPO_WORK_ARRAYS(TEMPO_WA_MEMBER)
#undef TEMPO_WA_MEMBER
  bool l_qc[TEMPO_NZP], l_qi[TEMPO_NZP], l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP];
  bool l_qh[TEMPO_NZP];
  int idx_bg[TEMPO_NZP];
  TempoTend T;
  // ty_tempo_main_diags scalars
  float rain_precip, cloud_precip, ice_liquid_equiv_precip, snow_liquid_equiv_precip,
      graupel_liquid_equiv_precip, frozen_fraction, frz_rain_precip, hail_liquid_equiv_precip;
  int did_micro;  // 0 when tempo_main returned early (no hydrometeors, no supersaturation)
};

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD bool tempo_any(const bool* l, int nz) {
#if TEMPO_WARP
  // warp: OR over this lane's levels, then over the warp.
  bool hit = false;
  TEMPO_LEVELS(k, nz) hit = hit || l[k];
  return __any_sync(TEMPO_FULL, hit);
#else
  for (int k = 1; k <= nz; ++k)
    if (l[k]) return true;
  return false;
#endif
}

// 828-848
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:826-848 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void aerosol_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* nwfa1d, const float* nifa1d, float* nwfa, float* nifa,
    float* nwfaten, float* nifaten) {
  TEMPO_LEVELS(k, nz) {
    if (nwfa1d != nullptr) nwfa[k] = (nwfa1d[k] + nwfaten[k] * C.global_dt) * rho[k];
    if (nifa1d != nullptr) nifa[k] = (nifa1d[k] + nifaten[k] * C.global_dt) * rho[k];
    nwfa[k] = tmax(PS(nwfa_default) * rho[k], tmin(PS(aero_max) * rho[k], nwfa[k]));
    nifa[k] = tmax(PS(nifa_default) * rho[k], tmin(PS(aero_max) * rho[k], nifa[k]));
  }
}

// 851-941 (optional dt never passed by tempo_main: dt_ = global_dt)
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:849-941 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void cloud_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qc, float* qc1d, float* nc1d, const float* ncsave,
    float* rc, float* nc, float* qcten, float* ncten, double* ilamc, float* mvd_c) {
  const float dt_ = C.global_dt;
  const float odt_ = 1.0f / dt_;
  int nu_c;
  double lamc; double  xdc;
  TEMPO_LEVELS(k, nz) {
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
        xdc = (PS(bm_r) + nu_c + 1.0f) / lamc;
        if (xdc < PS(d0c)) {
          lamc = PA2(cce, 2, nu_c) / PS(d0c);
          hit_limit = true;
        } else if (xdc > PS(d0r) * 2.0f) {
          lamc = PA2(cce, 2, nu_c) / (PS(d0r) * 2.0);
          hit_limit = true;
        }
        nc[k] = PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * rc[k] / PS(am_r) * tpowi(lamc, 3);
        if (hit_limit) ncten[k] = (nc[k] / rho[k] - nc1d[k]) * odt_;
        nc1d[k] = tmax(PS(nt_c_min) / rho[k],
            tmin(PA2(ccg, 1, nu_c) * PA1(ocg2, nu_c) * qc1d[k] / PS(am_r) * tpowi(lamc, 3),
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
      ilamc[k] = 1.0f / lamc;
      mvd_c[k] = tmax(tmin((3.0f + nu_c + 0.672f) * ilamc[k], PS(d0r)), PS(d0c));
    } else {
      l_qc[k] = false;
      rc[k] = PS(r1);
      nc[k] = PS(nt_c_min);
      mvd_c[k] = PS(d0c);
      ilamc[k] = 0.0f;
      qcten[k] = -qc1d[k] * odt_;
      qc1d[k] = 0.0f;
      if (nc1d != nullptr) {
        ncten[k] = -nc1d[k] * odt_;
        nc1d[k] = 0.0f;
      }
    }
  }
}

// 944-1009
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:942-1009 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void rain_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qr, float* qr1d, float* nr1d, float* rr, float* nr,
    float* qrten, float* nrten, double* ilamr, float* mvd_r) {
  double lamr = 0.0f;
  TEMPO_LEVELS(k, nz) {
    bool hit_limit = false;
    if (qr1d[k] + qrten[k] * C.global_dt > PS(r1)) {
      l_qr[k] = true;
      rr[k] = (qr1d[k] + qrten[k] * C.global_dt) * rho[k];
      qr1d[k] = qr1d[k] + qrten[k] * C.global_dt;
      nr[k] = tmax(PS(r2), (nr1d[k] + nrten[k] * C.global_dt) * rho[k]);
      if (nr[k] <= PS(r2)) {
        hit_limit = true;
        mvd_r[k] = 1.0e-3f;
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);
      }
      lamr = tempo_pow((PS(am_r) * PA1(crg, 3) * PS(org2) * nr[k] / rr[k]), PS(obmr));
      mvd_r[k] = (3.0f + PS(mu_r) + 0.672f) / lamr;
      if (mvd_r[k] > PS(d0r_max)) {
        hit_limit = true;
        mvd_r[k] = PS(d0r_max);
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);
      } else if (mvd_r[k] < PS(d0r) * 0.75f) {
        hit_limit = true;
        mvd_r[k] = PS(d0r) * 0.75f;
        lamr = (3.0 + PS(mu_r) + 0.672) / mvd_r[k];
        nr[k] = PA1(crg, 2) * PS(org3) * rr[k] * tpowi(lamr, 3) / PS(am_r);
      }
      if (hit_limit) nrten[k] = (nr[k] / rho[k] - nr1d[k]) * C.global_inverse_dt;
      nr1d[k] = PA1(crg, 2) * PS(org3) * qr1d[k] * tpowi(lamr, 3) / PS(am_r);
      ilamr[k] = 1.0f / lamr;
    } else {
      l_qr[k] = false;
      rr[k] = PS(r1);
      nr[k] = PS(r2);
      mvd_r[k] = PS(d0r);
      ilamr[k] = 0.0f;
      qrten[k] = -qr1d[k] * C.global_inverse_dt;
      nrten[k] = -nr1d[k] * C.global_inverse_dt;
      qr1d[k] = 0.0f;
      nr1d[k] = 0.0f;
    }
  }
}

// 1012-1082 (optional dt never passed by tempo_main)
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1010-1082 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void ice_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qi, float* qi1d, float* ni1d, float* ri, float* ni,
    float* qiten, float* niten, double* ilami) {
  const float dt_ = C.global_dt;
  const float odt_ = 1.0f / dt_;
  double lami; double  xdi;
  TEMPO_LEVELS(k, nz) {
    bool hit_limit = false;
    if (qi1d[k] + qiten[k] * dt_ > PS(r1)) {
      l_qi[k] = true;
      ri[k] = (qi1d[k] + qiten[k] * dt_) * rho[k];
      qi1d[k] = qi1d[k] + qiten[k] * dt_;
      ni[k] = tmax(PS(r2), (ni1d[k] + niten[k] * dt_) * rho[k]);
      if (ni[k] <= PS(r2)) {
        hit_limit = true;
        lami = PA1(cie, 2) / 5.e-6;
        ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tpowi(lami, 3));
      }
      lami = tempo_pow((PS(am_i) * PA1(cig, 2) * PS(oig1) * ni[k] / ri[k]), PS(obmi));
      xdi = (PS(bm_i) + PS(mu_i) + 1.0f) / lami;
      if (xdi < 5.e-6f) {
        hit_limit = true;
        lami = PA1(cie, 2) / 5.e-6;
        ni[k] = tmin(PS(max_ni), PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tpowi(lami, 3));
      } else if (xdi > PS(d0s)) {
        hit_limit = true;
        lami = PA1(cie, 2) / PS(d0s);
        ni[k] = PA1(cig, 1) * PS(oig2) * ri[k] / PS(am_i) * tpowi(lami, 3);
      }
      if (hit_limit) niten[k] = (ni[k] / rho[k] - ni1d[k]) * odt_;
      ni1d[k] = tmax(PS(r2) / rho[k],
          tmin(PA1(cig, 1) * PS(oig2) * qi1d[k] / PS(am_i) * tpowi(lami, 3), PS(max_ni) / rho[k]));
      ilami[k] = 1.0f / lami;
    } else {
      l_qi[k] = false;
      ri[k] = PS(r1);
      ni[k] = PS(r2);
      ilami[k] = 0.0f;
      qiten[k] = -qi1d[k] * odt_;
      niten[k] = -ni1d[k] * odt_;
      qi1d[k] = 0.0f;
      ni1d[k] = 0.0f;
    }
  }
}

// 1085-1117
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1083-1117 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void snow_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qs, float* qs1d, float* rs, float* qsten) {
  const float dt_ = C.global_dt;
  const float odt_ = 1.0f / dt_;
  TEMPO_LEVELS(k, nz) {
    if (qs1d[k] + qsten[k] * dt_ > PS(r1)) {
      l_qs[k] = true;
      rs[k] = (qs1d[k] + qsten[k] * dt_) * rho[k];
      qs1d[k] = qs1d[k] + qsten[k] * dt_;
    } else {
      l_qs[k] = false;
      rs[k] = PS(r1);
      qsten[k] = -qs1d[k] * odt_;
      qs1d[k] = 0.0f;
    }
  }
}

// 1120-1213
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1118-1213 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void graupel_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qg, float* qg1d, float* ng1d, float* qb1d, float* rg,
    float* ng, float* rb, int* idx, float* qgten, float* ngten, float* qbten,
    double* ilamg, float* mvd_g) {
  const int nrhg = PI(nrhg);
  double lamg = 0.0f; double  ygra1; double  zans1; double  n0_exp; double  lam_exp;
  TEMPO_LEVELS(k, nz) {
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
        idx[k] = imax(1, imin(tnint(10.0f * rg[k] / rb[k]) + 1, nrhg));
        if (ng[k] <= PS(r2)) {
          hit_limit = true;
          mvd_g[k] = 1.5e-3f;
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3) / PA1(am_g, idx[k]);
        }
        lamg = tempo_pow((PA1(am_g, idx[k]) * PA2(cgg, 3, 1) * PS(ogg2) * ng[k] / rg[k]), PS(obmg));
        mvd_g[k] = (3.0f + PS(mu_g) + 0.672f) / lamg;
        if (mvd_g[k] > 25.4e-3f) {
          hit_limit = true;
          mvd_g[k] = 25.4e-3f;
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3) / PA1(am_g, idx[k]);
        } else if (mvd_g[k] < PS(d0r)) {
          hit_limit = true;
          mvd_g[k] = PS(d0r);
          lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g[k];
          ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3) / PA1(am_g, idx[k]);
        }
        if (hit_limit) ngten[k] = (ng[k] / rho[k] - ng1d[k]) * C.global_inverse_dt;
        ng1d[k] = PA2(cgg, 2, 1) * PS(ogg3) * qg1d[k] * tpowi(lamg, 3) / PA1(am_g, idx[k]);
        qb1d[k] = tmin(tmax(qg1d[k] * PS(meters3_to_liters) / PA1(rho_g, nrhg),
                            qb1d[k] + qbten[k] * C.global_dt),
                       PS(meters3_to_liters) * qg1d[k] / PA1(rho_g, 1));
        idx[k] = imax(1, imin(tnint(10.0f * qg1d[k] / qb1d[k]) + 1, nrhg));
      } else {
        idx[k] = PI(idx_bg1);
        ygra1 = tempo_log10(tmax(1.e-9, double(rg[k])));
        zans1 = 3.4 + 2.0 / 7.0 * (ygra1 + 8.0);
        n0_exp = tmax(PS(gonv_min), tmin(tempo_pow(10.0, zans1), PS(gonv_max)));
        lam_exp = tempo_pow((n0_exp * PA1(am_g, idx[k]) * PA2(cgg, 1, 1) / rg[k]), PS(oge1));
        lamg = lam_exp * tempo_pow((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), PS(obmg));
        ng[k] = PA2(cgg, 2, 1) * PS(ogg3) * rg[k] * tpowi(lamg, 3) / PA1(am_g, idx[k]);
        rb[k] = PS(meters3_to_liters) * rg[k] / PA1(rho_g, idx[k]);
      }
      ilamg[k] = 1.0f / lamg;
      mvd_g[k] = (3.0f + PS(mu_g) + 0.672f) * ilamg[k];
    } else {
      l_qg[k] = false;
      rg[k] = PS(r1);
      ng[k] = PS(r2);
      mvd_g[k] = PS(d0g);
      ilamg[k] = 0.0f;
      idx[k] = PI(idx_bg1);
      rb[k] = PS(meters3_to_liters) * PS(r1) / PA1(rho_g, idx[k]);
      qgten[k] = -qg1d[k] * C.global_inverse_dt;
      qg1d[k] = 0.0f;
      if (ng1d != nullptr && qb1d != nullptr) {
        ngten[k] = -ng1d[k] * C.global_inverse_dt;
        qbten[k] = -qb1d[k] * C.global_inverse_dt;
        ng1d[k] = 0.0f;
        qb1d[k] = 0.0f;
      }
    }
  }
}

// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:1410-1448
TEMPO_HDN inline void hail_check_and_update(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, bool* l_qh, float* qh1d, float* rh, float* nh, float* qhten, double* ilamh) {
  const int nrhg = PI(nrhg);
  double lamh; double  ygra1; double  zans1; double  n0_exp; double  lam_exp;
  TEMPO_LEVELS(k, nz) {
    if (qh1d[k] + qhten[k] * C.global_dt > PS(r1)) {
      l_qh[k] = true;
      rh[k] = (qh1d[k] + qhten[k] * C.global_dt) * rho[k];
      // Intercept parameter from Jensen et al. (2023)
      ygra1 = tempo_log10(tmax(1.e-9, double(rh[k])));
      zans1 = 3.0 + 2.0 / 7.0 * (ygra1 + 8.0);
      n0_exp = tmax(PS(gonv_min), tmin(tempo_pow(10.0, zans1), PS(gonv_max)));
      lam_exp = tempo_pow((n0_exp * PA1(am_g, nrhg) * PA2(cgg, 1, 1) / rh[k]), PS(oge1));
      lamh = lam_exp * tempo_pow((PA2(cgg, 3, 1) * PS(ogg2) * PS(ogg1)), PS(obmg));
      ilamh[k] = 1.0 / lamh;
      nh[k] = PA2(cgg, 2, 1) * PS(ogg3) * rh[k] * tpowi(lamh, 3) / PA1(am_g, nrhg);
      qh1d[k] = qh1d[k] + qhten[k] * C.global_dt;
    } else {
      l_qh[k] = false;
      rh[k] = PS(r1);
      nh[k] = PS(r2);
      ilamh[k] = 0.0;
      qhten[k] = -qh1d[k] * C.global_inverse_dt;
      qh1d[k] = 0.0f;
    }
  }
}

// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:979-1040 (only with hail-hyperaware)
TEMPO_HDN inline void compute_hail_fraction(const TempoParams& P, int nz, const bool* l_qg,
    const float* qg1d, const float* mvd_g, const int* idx_bg, const float* qr1d, const float* qc1d,
    const float* temp, const float* tcond, const float* diffu, const float* delqvs, float* qh1d,
    float* hail_fraction) {
  float tempc; float  melt_prefactor; float  rho_g_val;
  const float t0 = PS(t0);
  TEMPO_LEVELS(k, nz) {
    hail_fraction[k] = 0.0f;
    qh1d[k] = 0.0f;
    tempc = temp[k] - t0;
    melt_prefactor = tempc * tcond[k] - PS(lvap0) * diffu[k] * delqvs[k];
    rho_g_val = PA1(rho_g, idx_bg[k]);
    if (l_qg[k]) {
      if (rho_g_val >= 350.0f && qg1d[k] > 1.e-6f) {
        // set hail fraction to slightly affect melting
        if (mvd_g[k] > 2.e-3f && tempc <= 0.0f) hail_fraction[k] = 0.1f;
        // main if for qg1d > 0.5 g/kg
        if (qg1d[k] > 5.e-4f) {
          hail_fraction[k] = tmax(tmin(tempo_exp(12.0f * (rho_g_val / hex_divisor_4485c000) - 9.9f) + 0.1f, 1.0f), 0.1f);
          // size boost
          if (mvd_g[k] > 2.e-3f) hail_fraction[k] = hail_fraction[k] + (mvd_g[k] * 1000.0f - 2.0f) * 0.1f;
          hail_fraction[k] = tmax(tmin(hail_fraction[k], 0.5f), 0.2f);
          // more hail with liquid present
          if (qr1d[k] + qc1d[k] > 1.e-3f) {
            if (qg1d[k] > 1.e-3f && rho_g_val > 699.0f) hail_fraction[k] = hail_fraction[k] + 0.2f;
          } else {
            // little liquid: limit the fraction at large mixing ratios (many small graupel/hail)
            if (rho_g_val < 699.0f && qg1d[k] > 5.e-3f) {
              hail_fraction[k] = tmin(hail_fraction[k], tmin(0.001f / qg1d[k], 1.0f) * 0.5f);
            }
          }
          qh1d[k] = hail_fraction[k] * qg1d[k];
          // melting
          if (rho_g_val >= 749.0f && melt_prefactor > PS(eps) && tempc > 0.0f) {
            if (mvd_g[k] > 2.e-3f) {
              hail_fraction[k] = hail_fraction[k]
                  + (1.0f - tmax(tmin((1.0f / (1.0f + tempo_exp(temp[k] - t0 - 5.0f))), 1.0f), 0.0f));
              hail_fraction[k] = tmin(hail_fraction[k], 0.8f + (mvd_g[k] * 1000.0f - 6.0f) * 0.1f);
              hail_fraction[k] = tmax(tmin(hail_fraction[k], 0.8f), 0.0f);
            } else {
              hail_fraction[k] = 0.0f;
            }
            qh1d[k] = hail_fraction[k] * qg1d[k];
          }
        }
      }
    }
  }
}

// 1216-1242
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1214-1242 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void graupel_init(const TempoParams& P, int nz, const float* rho,
    const float* qg1d, float* ng1d, float* qb1d) {
  for (int k = 1; k <= nz; ++k) {
    if (qg1d[k] > PS(r1)) {
      float rg = qg1d[k] * rho[k];
      float rb = rg * PS(meters3_to_liters) / PA1(rho_g, PI(idx_bg1));
      int idx = imax(1, imin(tnint(10.0f * rg / rb) + 1, PI(nrhg)));
      float mvd_g = 5.e-3f;
      double lamg = (3.0 + PS(mu_g) + 0.672) / mvd_g;
      float ng = PA2(cgg, 2, 1) * PS(ogg3) * rg * tpowi(lamg, 3) / PA1(am_g, idx);
      ng1d[k] = ng / rho[k];
      qb1d[k] = rb / rho[k];
    }
  }
}

// 1245-1292
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1243-1292 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void thermo_vars(const TempoParams& P, int nz, const float* qv,
    const float* temp, const float* pres, const float* rho, float* rhof, float* rhof2,
    float* qvs, float* delqvs, float* qvsi, float* satw, float* sati, float* ssatw,
    float* ssati, float* diffu, float* visco, float* vsc2, float* ocp, float* lvap,
    float* tcond, float* lvt2, bool* supersaturated) {
  const float t0 = PS(t0);
  *supersaturated = false;
  TEMPO_LEVELS(k, nz) {
    float otemp = 1.0f / temp[k];
    float tempc = temp[k] - t0;
    rhof[k] = sqrt(PS(rho_not) / rho[k]);
    rhof2[k] = sqrt(rhof[k]);
    qvs[k] = calc_rslf(pres[k], temp[k]);
    delqvs[k] = tmax(0.0f, calc_rslf(pres[k], t0) - qv[k]);
    if (tempc <= 0.0f) {
      qvsi[k] = calc_rsif(pres[k], temp[k]);
    } else {
      qvsi[k] = qvs[k];
    }
    satw[k] = qv[k] / qvs[k];
    sati[k] = qv[k] / qvsi[k];
    ssatw[k] = satw[k] - 1.0f;
    ssati[k] = sati[k] - 1.0f;
    if (fabs(ssatw[k]) < PS(eps)) ssatw[k] = 0.0f;
    if (fabs(ssati[k]) < PS(eps)) ssati[k] = 0.0f;
    if (ssati[k] > 0.0f) *supersaturated = true;
    diffu[k] = 2.11e-5f * tempo_pow((temp[k] / t0), 1.94f) * (101325.0f / pres[k]);
    if (tempc >= 0.0f) {
      visco[k] = (1.718f + 0.0049f * tempc) * 1.0e-5f;
    } else {
      visco[k] = (1.718f + 0.0049f * tempc - 1.2e-5f * tempc * tempc) * 1.0e-5f;
    }
    ocp[k] = 1.0f / (PS(cp) * (1.0f + 0.887f * qv[k]));
    vsc2[k] = sqrt(rho[k] / visco[k]);
    lvap[k] = PS(lvap0) + (2106.0f - 4218.0f) * tempc;
    tcond[k] = (5.69f + 0.0168f * tempc) * 1.0e-5f * 418.936f;
    lvt2[k] = lvap[k] * lvap[k] * ocp[k] * PS(orv) * otemp * otemp;
  }
#if TEMPO_WARP
  // warp: the column flag is the OR over every lane's levels.
  *supersaturated = __any_sync(TEMPO_FULL, *supersaturated);
#endif
}

// 1295-1419
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1293-1419 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void check_over_depletion(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rho, const float* temp, const float* qvsi, const float* qv,
    const bool* l_qc, const float* rc, const bool* l_qi, const float* ri, const bool* l_qr,
    const float* rr, const bool* l_qs, const float* rs, const bool* l_qg, const float* rg,
    const bool* l_qh, const float* rh, TempoTend& T) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:1530-1675
  float sump; float  rate_max; float  ratio;
  const float eps = PS(eps);
  TEMPO_LEVELS(k, nz) {
    sump = T.pri_inu[k] + T.pri_ide[k] + T.prs_ide[k] + T.prs_sde[k] + T.prg_gde[k] + T.pri_iha[k];
    rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999f;
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

    // losses to hail: sublimation, melting
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
    T.prr_rcg[k] = ratio * tsign(1.0f, T.prr_rcg[k]);
    T.prg_rcg[k] = -T.prr_rcg[k];
    T.pbg_rcg[k] = PS(meters3_to_liters) * T.prg_rcg[k] / PA1(rho_g, PI(nrhg));
    if (temp[k] >= PS(t0)) {
      ratio = tmin(fabs(T.prr_rcs[k]), fabs(T.prs_rcs[k]));
      T.prr_rcs[k] = ratio * tsign(1.0f, T.prr_rcs[k]);
      T.prs_rcs[k] = -T.prr_rcs[k];
    }
    // rain-hail
    ratio = tmin(fabs(T.prr_rch[k]), fabs(T.prh_rch[k]));
    T.prr_rch[k] = ratio * tsign(1.0f, T.prr_rch[k]);
    T.prh_rch[k] = ratio * tsign(1.0f, T.prh_rch[k]);
  }
}

// 1422-1494
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1420-1494 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void sum_tendencies(const TempoParams& P, int nz, const float* rho,
    const float* temp, const int* idx, const float* lvap, const float* ocp,
    const TempoTend& T, float* tten, float* qvten, float* qcten, float* ncten,
    float* qiten, float* niten, float* qsten, float* qrten, float* nrten, float* qgten,
    float* ngten, float* qbten, float* qhten) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:1678-1777.  final_weight (main:1688) is read at main:1755 even when main:1752 did not set it
  // (an undefined or stale value in native); here it is 1 there (no scaling).  The native referee carries the
  // same definition (final_weight = 1._wp, labelled patched).
  TEMPO_LEVELS(k, nz) {
    float orho = 1.f / rho[k];
    float lfus2 = PS(lsub) - lvap[k];
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
        + T.prr_rcs[k] + T.prr_rcg[k] + T.prr_rch[k] - T.prg_rfz[k] - T.pri_rfz[k] - T.prr_rci[k]
        + T.prr_hml[k]) * orho;
    nrten[k] = nrten[k] + (T.pnr_wau[k] + T.pnr_sml[k] + T.pnr_gml[k] + T.pnr_hml[k]
        - (T.pnr_rfz[k] + T.pnr_rcr[k] + T.pnr_rcg[k] + T.pnr_rch[k] + T.pnr_rcs[k] + T.pnr_rci[k]
        + T.pni_rfz[k])) * orho;
    qsten[k] = qsten[k] + (T.prs_iau[k] + T.prs_sde[k] + T.prs_sci[k] + T.prs_scw[k]
        + T.prs_rcs[k] + T.prs_ide[k] - T.prs_ihm[k] - T.prr_sml[k]) * orho;
    qgten[k] = qgten[k] + (T.prg_scw[k] + T.prg_rfz[k] + T.prg_gde[k] + T.prg_rcg[k]
        + T.prg_gcw[k] + T.prg_rci[k] + T.prg_rcs[k] - T.prg_ihm[k] - T.prr_gml[k]) * orho;
    ngten[k] = ngten[k] + (T.png_scw[k] + T.png_rfz[k] - T.png_rcg[k] + T.png_rci[k]
        + T.png_rcs[k] + T.png_gde[k] - T.pnr_gml[k]) * orho;
    qbten[k] = qbten[k] + (T.pbg_scw[k] + T.pbg_rfz[k] + T.pbg_gcw[k] + T.pbg_rci[k]
        + T.pbg_rcs[k] + T.pbg_rcg[k] + T.pbg_sml[k] - T.pbg_gml[k]
        + PS(meters3_to_liters) * (T.prg_gde[k] - T.prg_ihm[k]) / PA1(rho_g, 1)) * orho;
    // density weighting: frozen drops raise the graupel density (main:1737-1757)
    if (temp[k] < PS(t0)) {
      float frozen_terms = T.prg_rfz[k] + T.prg_rcg[k] + T.prg_rci[k] + T.prg_rcs[k];
      float riming_terms = T.prg_scw[k] + T.prg_gcw[k];
      float massg_sources = frozen_terms + riming_terms;
      float volg_sources = T.pbg_scw[k] + T.pbg_gcw[k] + T.pbg_rfz[k] + T.pbg_rcg[k] + T.pbg_rci[k] + T.pbg_rcs[k];
      if (frozen_terms > PS(r1) && frozen_terms > riming_terms) {
        float dens_weight = PA1(rho_g, PI(nrhg)) * frozen_terms / massg_sources;
        if (T.pbg_scw[k] > PS(eps)) dens_weight = dens_weight + T.prg_scw[k] / T.pbg_scw[k] * (T.prg_scw[k] / massg_sources);
        if (T.pbg_gcw[k] > PS(eps)) dens_weight = dens_weight + T.prg_gcw[k] / T.pbg_gcw[k] * (T.prg_gcw[k] / massg_sources);
        float final_weight = 1.0f;
        if (dens_weight > PS(r1) && volg_sources > PS(r1)) {
          final_weight = (1.0f / dens_weight) * (massg_sources / volg_sources);
        }
        if (final_weight <= 1.0f) qbten[k] = qbten[k] * final_weight;
      }
    }
    qhten[k] = qhten[k] + (T.prh_rch[k] + T.prh_hcw[k] + T.prh_hde[k] - T.prr_hml[k]) * orho;
    if (temp[k] < PS(t0)) {
      tten[k] = tten[k]
          + (PS(lsub) * ocp[k] * (T.pri_inu[k] + T.pri_ide[k] + T.prs_ide[k] + T.prs_sde[k]
             + T.prg_gde[k] + T.prh_hde[k] + T.pri_iha[k])
             + lfus2 * ocp[k] * (T.pri_wfz[k] + T.pri_rfz[k] + T.prg_rfz[k] + T.prs_scw[k]
             + T.prg_scw[k] + T.prg_gcw[k] + T.prh_hcw[k] - T.prr_rcs[k] + T.prr_rci[k]
             - T.prr_rcg[k] - T.prr_rch[k])) * orho;
    } else {
      tten[k] = tten[k]
          + (PS(lfus) * ocp[k] * (-T.prr_sml[k] - T.prr_gml[k] - T.prr_hml[k] - T.prr_rcg[k] - T.prr_rcs[k])
             + PS(lsub) * ocp[k] * (T.prs_sde[k] + T.prg_gde[k] + T.prh_hde[k])) * orho;
    }
  }
}

// 1497-1542.  precip may be nullptr.  ktop_sedi always passed by tempo_main.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1495-1542 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void sedimentation(const TempoParams& P, const TempoCtx& C, int nz,
    float* xr, const float* vt, const float* dz1d, const float* rho, float* xten,
    float limit, int steps, int ktop_sedi, float* precip, float* sed_r) {
  float odz; float  orho;
  int ktop = ktop_sedi;
#if TEMPO_WARP
  // warp: every flux is taken from the pre-update profile, so the level
  // updates are independent once all fluxes exist; level nz keeps its
  // top update first, as the serial text orders it.
  TEMPO_SYNC();
  TEMPO_LEVELS(k, nz) sed_r[k] = vt[k] * xr[k];
  TEMPO_SYNC();
  TEMPO_LEVELS(k, nz) {
    if (k == nz) {
      odz = 1.0f / dz1d[k];
      orho = 1.0f / rho[k];
      xten[k] = xten[k] - sed_r[k] * odz * (1.0f / (float )steps) * orho;
      xr[k] = tmax(limit, xr[k] - sed_r[k] * odz * C.global_dt * (1.0f / (float )steps));
    }
    if (k <= ktop) {
      odz = 1.0f / dz1d[k];
      orho = 1.0f / rho[k];
      xten[k] = xten[k] + (sed_r[k + 1] - sed_r[k]) * odz * (1.0f / (float )steps) * orho;
      xr[k] = tmax(limit, xr[k] + (sed_r[k + 1] - sed_r[k]) * odz * C.global_dt * (1.0f / (float )steps));
    }
  }
  // Level 1 and the precipitation scalars belong to the lead lane.
  if (TEMPO_LEAD && precip != nullptr) {
    if (xr[1] > PS(low_limit_mass_for_precip)) {
      *precip = *precip + sed_r[1] * C.global_dt * (1.0f / (float )steps);
    }
  }
  TEMPO_SYNC();
#else
  for (int k = nz; k >= 1; --k) sed_r[k] = vt[k] * xr[k];
  int k = nz;
  odz = 1.0f / dz1d[k];
  orho = 1.0f / rho[k];
  xten[k] = xten[k] - sed_r[k] * odz * (1.0f / (float )steps) * orho;
  xr[k] = tmax(limit, xr[k] - sed_r[k] * odz * C.global_dt * (1.0f / (float )steps));
  for (k = ktop; k >= 1; --k) {
    odz = 1.0f / dz1d[k];
    orho = 1.0f / rho[k];
    xten[k] = xten[k] + (sed_r[k + 1] - sed_r[k]) * odz * (1.0f / (float )steps) * orho;
    xr[k] = tmax(limit, xr[k] + (sed_r[k + 1] - sed_r[k]) * odz * C.global_dt * (1.0f / (float )steps));
  }
  if (precip != nullptr) {
    if (xr[1] > PS(low_limit_mass_for_precip)) {
      *precip = *precip + sed_r[1] * C.global_dt * (1.0f / (float )steps);
    }
  }
#endif
}

// 1545-1731 semi-Lagrangian sedimentation (Juang and Hong 2010).  W supplies scratch.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1543-1731 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void semilagrangian_sedimentation(const TempoParams& P, const TempoCtx& C,
    int nz, TempoWork& W, const float* dz1d, const float* rho, float* xr, float* xten,
    const float* vt, int steps, float limit, float* precip) {
  float fa1; float  fa2; float  con1; float  decfl; float  dip; float  dim; float  tl; float  th; float  tl2; float  th2; float  qqd; float  qqh; float  qql; float  zsum; float  qsum; float  orho; float  dql; float  dqh;
  float * zi = W.sl_zi;
  float * wi = W.sl_wi;
  float * za = W.sl_za;
  float * dza = W.sl_dza;
  float * qa = W.sl_qa;
  float * qmi = W.sl_qmi;
  float * qpi = W.sl_qpi;
  float * net_flx = W.sl_net_flx;
  float * precip_flx = W.sl_precip_flx;
  float * rr_save = W.sl_rr_save;
  int k, kk, kb, kt, m;
  for (k = 1; k <= nz + 2; ++k) {
    zi[k] = 0.0f; wi[k] = 0.0f; dza[k] = 0.0f; za[k] = 0.0f; qa[k] = 0.0f; qmi[k] = 0.0f; qpi[k] = 0.0f;
  }
  for (k = 1; k <= nz; ++k) {
    net_flx[k] = 0.0f;
    precip_flx[k] = 0.0f;
    rr_save[k] = xr[k];
  }
  zi[1] = 0.0f;
  for (k = 1; k <= nz; ++k) zi[k + 1] = zi[k] + dz1d[k];
  fa1 = 9.0f / hex_divisor_41800000;
  fa2 = 1.0f / hex_divisor_41800000;
  wi[1] = vt[1];
  wi[2] = 0.5f * (vt[2] + vt[1]);
  for (k = 3; k <= nz - 1; ++k) wi[k] = fa1 * (vt[k] + vt[k - 1]) - fa2 * (vt[k + 1] + vt[k - 2]);
  wi[nz] = 0.5f * (vt[nz] + vt[nz - 1]);
  wi[nz + 1] = vt[nz + 1];
  for (k = 2; k <= nz; ++k)
    if (vt[k] == 0.0f) wi[k] = vt[k - 1];
  con1 = 0.05f;
  for (k = nz; k >= 1; --k) {
    decfl = (wi[k + 1] - wi[k]) * C.global_dt * (1.0f / (float )steps) / dz1d[k];
    if (decfl > con1) wi[k] = wi[k + 1] - con1 * dz1d[k] * C.global_inverse_dt * (float )steps;
  }
  for (k = 1; k <= nz + 1; ++k) za[k] = zi[k] - wi[k] * C.global_dt * (1.0f / (float )steps);
  za[nz + 2] = zi[nz + 1];
  for (k = 1; k <= nz + 1; ++k) dza[k] = za[k + 1] - za[k];
  for (k = 1; k <= nz; ++k) qa[k] = xr[k] * dz1d[k] / dza[k];
  qa[nz + 1] = 0.0f;
  for (k = 2; k <= nz; ++k) {
    dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
    dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
    if (dip * dim <= 0.0f) {
      qmi[k] = qa[k];
      qpi[k] = qa[k];
    } else {
      qpi[k] = qa[k] + 0.5f * (dip + dim) * dza[k];
      qmi[k] = 2.0f * qa[k] - qpi[k];
      if (qpi[k] > 0.0f || qmi[k] < 0.0f) {
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
        qqd = 0.5f * (qpi[kb] - qmi[kb]);
        qqh = qqd * th2 + qmi[kb] * th;
        qql = qqd * tl2 + qmi[kb] * tl;
        xr[k] = (qqh - qql) / (th - tl);
      } else if (kt > kb) {
        tl = (zi[k] - za[kb]) / dza[kb];
        tl2 = tl * tl;
        qqd = 0.5f * (qpi[kb] - qmi[kb]);
        qql = qqd * tl2 + qmi[kb] * tl;
        dql = qa[kb] - qql;
        zsum = (1.0f - tl) * dza[kb];
        qsum = dql * dza[kb];
        if (kt - kb > 1) {
          for (m = kb + 1; m <= kt - 1; ++m) {
            zsum = zsum + dza[m];
            qsum = qsum + qa[m] * dza[m];
          }
        }
        th = (zi[k + 1] - za[kt]) / dza[kt];
        th2 = th * th;
        qqd = 0.5f * (qpi[kt] - qmi[kt]);
        dqh = qqd * th2 + qmi[kt] * th;
        zsum = zsum + th * dza[kt];
        qsum = qsum + dqh * dza[kt];
        xr[k] = qsum / zsum;
      }
    }
    orho = 1.0f / rho[k];
    xr[k] = tmax(xr[k], limit);
    xten[k] = xten[k] + (xr[k] - rr_save[k]) * orho * C.global_inverse_dt;
  }

  for (k = 1; k <= nz; ++k) {
    if (za[k] < 0.0f && za[k + 1] <= 0.0f) {
      if (precip != nullptr) *precip = *precip + qa[k] * dza[k];
      net_flx[k] = qa[k] * dza[k];
    } else if (za[k] < 0.0f && za[k + 1] > 0.0f) {
      th = (0.0f - za[k]) / dza[k];
      th2 = th * th;
      qqd = 0.5f * (qpi[k] - qmi[k]);
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
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1732-1771 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void rain_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const bool* l_qr, const float* rr, const double* ilamr,
    const float* dz1d, float* vt, float* vtn, int* substeps_sedi, int* ktop_sedi) {
  float dz_by_vt; double  lamr;
#if TEMPO_WARP
  // warp: active levels evaluate in parallel; each "vt(k) = vt(k+1)" chain
  // ends at the nearest active level above (or slot nz+1), read after a sync;
  // ktop and the substep count are integer maxima.
  unsigned rows[TEMPO_ROWS];
  TEMPO_SYNC();
  TEMPO_BALLOT_ROWS(rows, rr[k] > PS(r1))
  const int active_count=tempo_active_count(rows,nz);
  for(int rank=TEMPO_LANE;rank<active_count;rank+=32) {
    const int k=tempo_active_level(rows,rank,nz);
    if (rr[k] > PS(r1)) {
      lamr = 1.0f / ilamr[k];
      // Two pows per base share the base's log (tempo_pow_base: same words as tempo_pow).
      const TempoPowBase bl = tempo_pow_base(lamr), bf = tempo_pow_base(lamr + PS(fv_r));
      vt[k] = rhof[k] * PS(av_r) * PA1(crg, 6) * PS(org3) * tempo_pow_with(bl, PA1(cre, 3))
            * (tempo_pow_with(bf, (-PA1(cre, 6))));
      vtn[k] = rhof[k] * PS(av_r) * PA1(crg, 7) / PA1(crg, 12) * tempo_pow_with(bl, PA1(cre, 12))
             * (tempo_pow_with(bf, (-PA1(cre, 7))));
    }
  }
  TEMPO_SYNC();
  int ktop = 1, substeps = 1;
  TEMPO_LEVELS(k, nz) {
    if (!(rr[k] > PS(r1))) {
      const int j = tempo_next_active(rows, k, nz);
      vt[k] = vt[j];
      vtn[k] = vtn[j];
    }
    if (tmax(vt[k], vtn[k]) > 1.e-3f) {
      ktop = imax(ktop, k);
      dz_by_vt = dz1d[k] / (tmax(vt[k], vtn[k]));
      substeps = imax(substeps, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  ktop = tempo_warp_max(ktop);
  substeps = tempo_warp_max(substeps);
  if (ktop_sedi) *ktop_sedi = (ktop == nz) ? nz - 1 : ktop;
  if (substeps_sedi) *substeps_sedi = substeps;
  TEMPO_SYNC();
  return;
#endif
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rr[k] > PS(r1)) {
      lamr = 1.0f / ilamr[k];
      // Two pows per base share the base's log (tempo_pow_base: same words as tempo_pow).
      const TempoPowBase bl = tempo_pow_base(lamr), bf = tempo_pow_base(lamr + PS(fv_r));
      vt[k] = rhof[k] * PS(av_r) * PA1(crg, 6) * PS(org3) * tempo_pow_with(bl, PA1(cre, 3))
            * (tempo_pow_with(bf, (-PA1(cre, 6))));
      vtn[k] = rhof[k] * PS(av_r) * PA1(crg, 7) / PA1(crg, 12) * tempo_pow_with(bl, PA1(cre, 12))
             * (tempo_pow_with(bf, (-PA1(cre, 7))));
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (tmax(vt[k], vtn[k]) > 1.e-3f) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / (tmax(vt[k], vtn[k]));
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1774-1828.  has_qb1d = present(qb1d).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1772-1828 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void graupel_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const float* rho, const float* visco, const bool* l_qg,
    const float* rg, const float* rb, bool has_qb1d, const int* idx, const double* ilamg,
    const float* dz1d, float* vt, float* vtn, int* substeps_sedi, int* ktop_sedi) {
  float dz_by_vt; float  dens_g; float  afall; float  bfall;
  const int nrhg = PI(nrhg);
#if TEMPO_WARP
  // warp: as rain_fallspeed.
  unsigned rows[TEMPO_ROWS];
  TEMPO_SYNC();
  TEMPO_BALLOT_ROWS(rows, rg[k] > PS(r1))
  const int active_count=tempo_active_count(rows,nz);
  for(int rank=TEMPO_LANE;rank<active_count;rank+=32) {
    const int k=tempo_active_level(rows,rank,nz);
    if (rg[k] > PS(r1)) {
      if (has_qb1d) {
        dens_g = tmax(PA1(rho_g, 1), tmin(PS(meters3_to_liters) * rg[k] / rb[k], PA1(rho_g, nrhg)));
        afall = tempo_pow((4.0f * dens_g * 9.8f / (3.0f * 0.504843467198652f * PS(rho_not))), PS(b_coeff));  // main:2123
        bfall = 3.0f * PS(b_coeff) - 1.0f;
      } else {
        afall = PS(av_g_old);
        bfall = PS(bv_g_old);
      }
      const double pg = tempo_pow(ilamg[k], bfall);  // one evaluation for vt and vtn (same operands)
      vt[k] = rhof[k] * afall * PA2(cgg, 6, idx[k]) * PS(ogg3) * pg;
      if (PS(mu_g) == 0) {
        vtn[k] = rhof[k] * afall * PA2(cgg, 7, idx[k]) / PA2(cgg, 12, idx[k]) * pg;
      } else {
        vtn[k] = rhof[k] * afall * PA2(cgg, 8, idx[k]) * PS(ogg2) * pg;
      }
    }
  }
  TEMPO_SYNC();
  int ktop = 1, substeps = 1;
  TEMPO_LEVELS(k, nz) {
    if (!(rg[k] > PS(r1))) {
      const int j = tempo_next_active(rows, k, nz);
      vt[k] = vt[j];
      vtn[k] = vtn[j];
    }
    if (vt[k] > 1.e-3f) {
      ktop = imax(ktop, k);
      dz_by_vt = dz1d[k] / vt[k];
      substeps = imax(substeps, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  ktop = tempo_warp_max(ktop);
  substeps = tempo_warp_max(substeps);
  if (ktop_sedi) *ktop_sedi = (ktop == nz) ? nz - 1 : ktop;
  if (substeps_sedi) *substeps_sedi = substeps;
  TEMPO_SYNC();
  return;
#endif
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rg[k] > PS(r1)) {
      if (has_qb1d) {
        dens_g = tmax(PA1(rho_g, 1), tmin(PS(meters3_to_liters) * rg[k] / rb[k], PA1(rho_g, nrhg)));
        afall = tempo_pow((4.0f * dens_g * 9.8f / (3.0f * 0.504843467198652f * PS(rho_not))), PS(b_coeff));  // main:2123
        bfall = 3.0f * PS(b_coeff) - 1.0f;
      } else {
        afall = PS(av_g_old);
        bfall = PS(bv_g_old);
      }
      const double pg = tempo_pow(ilamg[k], bfall);  // one evaluation for vt and vtn (same operands)
      vt[k] = rhof[k] * afall * PA2(cgg, 6, idx[k]) * PS(ogg3) * pg;
      if (PS(mu_g) == 0) {
        vtn[k] = rhof[k] * afall * PA2(cgg, 7, idx[k]) / PA2(cgg, 12, idx[k]) * pg;
      } else {
        vtn[k] = rhof[k] * afall * PA2(cgg, 8, idx[k]) * PS(ogg2) * pg;
      }
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (vt[k] > 1.e-3f) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}


// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:2056-2093 (mass-weighted hail fall speed)
TEMPO_HDN inline void hail_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const bool* l_qh, const float* rh, const double* ilamh,
    const float* dz1d, float* vt, int* substeps_sedi, int* ktop_sedi) {
  float dz_by_vt; float  vth;
  const int nrhg = PI(nrhg);
  (void)l_qh;
#if TEMPO_WARP
  // warp: as graupel_fallspeed (one speed).
  unsigned rows[TEMPO_ROWS];
  TEMPO_SYNC();
  TEMPO_BALLOT_ROWS(rows, rh[k] > PS(r1))
  const int active_count=tempo_active_count(rows,nz);
  for(int rank=TEMPO_LANE;rank<active_count;rank+=32) {
    const int k=tempo_active_level(rows,rank,nz);
    if (rh[k] > PS(r1)) {
      vth = rhof[k] * PA1(av_g, nrhg) * PA2(cgg, 6, nrhg) * PS(ogg3) * tempo_pow(ilamh[k], PA1(bv_g, nrhg));
      vt[k] = vth;
    }
  }
  TEMPO_SYNC();
  int ktop = 1, substeps = 1;
  TEMPO_LEVELS(k, nz) {
    if (!(rh[k] > PS(r1))) {
      const int j = tempo_next_active(rows, k, nz);
      vt[k] = vt[j];
    }
    if (vt[k] > 1.e-3f) {
      ktop = imax(ktop, k);
      dz_by_vt = dz1d[k] / vt[k];
      substeps = imax(substeps, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  ktop = tempo_warp_max(ktop);
  substeps = tempo_warp_max(substeps);
  if (ktop_sedi) *ktop_sedi = (ktop == nz) ? nz - 1 : ktop;
  if (substeps_sedi) *substeps_sedi = substeps;
  TEMPO_SYNC();
  return;
#endif
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rh[k] > PS(r1)) {
      vth = rhof[k] * PA1(av_g, nrhg) * PA2(cgg, 6, nrhg) * PS(ogg3) * tempo_pow(ilamh[k], PA1(bv_g, nrhg));
      vt[k] = vth;
    } else {
      vt[k] = vt[k + 1];
    }
    if (vt[k] > 1.e-3f) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1831-1885
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1829-1885 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void snow_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const bool* l_qs, const float* rs, const double* prr_sml,
    const double* smob, const double* smoc, const float* rr, const float* vtrr,
    const float* dz1d, float* vt, const float* vtboost, int* substeps_sedi, int* ktop_sedi) {
  float dz_by_vt; float  vts; float  sr; double  xds; double  mrat; double  ils1; double  ils2; double  t1_vts; double  t2_vts; double  t3_vts; double  t4_vts;
#if TEMPO_WARP
  // warp: as rain_fallspeed (one speed, no number-weighted twin).
  unsigned rows[TEMPO_ROWS];
  TEMPO_SYNC();
  TEMPO_BALLOT_ROWS(rows, rs[k] > PS(r1))
  const int active_count=tempo_active_count(rows,nz);
  for(int rank=TEMPO_LANE;rank<active_count;rank+=32) {
    const int k=tempo_active_level(rows,rank,nz);
    if (rs[k] > PS(r1)) {
      xds = smoc[k] / smob[k];
      mrat = 1.0f / xds;
      ils1 = 1.0f / (mrat * PS(lam0) + PS(fv_s));
      ils2 = 1.0f / (mrat * PS(lam1) + PS(fv_s));
      t1_vts = PS(kap0) * PA1(csg, 4) * tempo_pow(ils1, PA1(cse, 4));
      const double pm = tempo_pow(mrat, PS(mu_s));  // one evaluation for t2 and t4 (same operands)
      t2_vts = PS(kap1) * pm * PA1(csg, 10) * tempo_pow(ils2, PA1(cse, 10));
      ils1 = 1.0f / (mrat * PS(lam0));
      ils2 = 1.0f / (mrat * PS(lam1));
      t3_vts = PS(kap0) * PA1(csg, 1) * tempo_pow(ils1, PA1(cse, 1));
      t4_vts = PS(kap1) * pm * PA1(csg, 7) * tempo_pow(ils2, PA1(cse, 7));
      vts = rhof[k] * PS(av_s) * (t1_vts + t2_vts) / (t3_vts + t4_vts);
      if (prr_sml[k] > 0.0f) {
        sr = rs[k] / (rs[k] + rr[k]);
        vt[k] = vts * sr + (1.0f - sr) * vtrr[k];
      } else {
        vt[k] = vts * vtboost[k];
      }
    }
  }
  TEMPO_SYNC();
  int ktop = 1, substeps = 1;
  TEMPO_LEVELS(k, nz) {
    if (!(rs[k] > PS(r1))) vt[k] = vt[tempo_next_active(rows, k, nz)];
    if (vt[k] > 1.e-3f) {
      ktop = imax(ktop, k);
      dz_by_vt = dz1d[k] / vt[k];
      substeps = imax(substeps, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  ktop = tempo_warp_max(ktop);
  substeps = tempo_warp_max(substeps);
  if (ktop_sedi) *ktop_sedi = (ktop == nz) ? nz - 1 : ktop;
  if (substeps_sedi) *substeps_sedi = substeps;
  TEMPO_SYNC();
  return;
#endif
  if (ktop_sedi) *ktop_sedi = 1;
  if (substeps_sedi) *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (rs[k] > PS(r1)) {
      xds = smoc[k] / smob[k];
      mrat = 1.0f / xds;
      ils1 = 1.0f / (mrat * PS(lam0) + PS(fv_s));
      ils2 = 1.0f / (mrat * PS(lam1) + PS(fv_s));
      t1_vts = PS(kap0) * PA1(csg, 4) * tempo_pow(ils1, PA1(cse, 4));
      const double pm = tempo_pow(mrat, PS(mu_s));  // one evaluation for t2 and t4 (same operands)
      t2_vts = PS(kap1) * pm * PA1(csg, 10) * tempo_pow(ils2, PA1(cse, 10));
      ils1 = 1.0f / (mrat * PS(lam0));
      ils2 = 1.0f / (mrat * PS(lam1));
      t3_vts = PS(kap0) * PA1(csg, 1) * tempo_pow(ils1, PA1(cse, 1));
      t4_vts = PS(kap1) * pm * PA1(csg, 7) * tempo_pow(ils2, PA1(cse, 7));
      vts = rhof[k] * PS(av_s) * (t1_vts + t2_vts) / (t3_vts + t4_vts);
      if (prr_sml[k] > 0.0f) {
        sr = rs[k] / (rs[k] + rr[k]);
        vt[k] = vts * sr + (1.0f - sr) * vtrr[k];
      } else {
        vt[k] = vts * vtboost[k];
      }
    } else {
      vt[k] = vt[k + 1];
    }
    if (vt[k] > 1.e-3f) {
      if (ktop_sedi) *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      if (substeps_sedi) *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  if (ktop_sedi) {
    if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
  }
}

// 1888-1920
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1886-1920 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void ice_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const bool* l_qi, const float* ri, const double* ilami,
    const float* dz1d, float* vt, float* vtn, int* substeps_sedi, int* ktop_sedi) {
  float dz_by_vt;
#if TEMPO_WARP
  // warp: as rain_fallspeed.
  unsigned rows[TEMPO_ROWS];
  TEMPO_SYNC();
  TEMPO_BALLOT_ROWS(rows, ri[k] > PS(r1))
  const int active_count=tempo_active_count(rows,nz);
  for(int rank=TEMPO_LANE;rank<active_count;rank+=32) {
    const int k=tempo_active_level(rows,rank,nz);
    if (ri[k] > PS(r1)) {
      const double pi_ = tempo_pow(ilami[k], PS(bv_i));  // one evaluation for vt and vtn
      vt[k] = rhof[k] * PS(av_i) * PA1(cig, 3) * PS(oig2) * pi_;
      vtn[k] = rhof[k] * PS(av_i) * PA1(cig, 6) / PA1(cig, 7) * pi_;
    }
  }
  TEMPO_SYNC();
  int ktop = 1, substeps = 1;
  TEMPO_LEVELS(k, nz) {
    if (!(ri[k] > PS(r1))) {
      const int j = tempo_next_active(rows, k, nz);
      vt[k] = vt[j];
      vtn[k] = vtn[j];
    }
    if (vt[k] > 1.e-3f) {
      ktop = imax(ktop, k);
      dz_by_vt = dz1d[k] / vt[k];
      substeps = imax(substeps, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  ktop = tempo_warp_max(ktop);
  *substeps_sedi = tempo_warp_max(substeps);
  *ktop_sedi = (ktop == nz) ? nz - 1 : ktop;
  TEMPO_SYNC();
  return;
#endif
  *ktop_sedi = 1;
  *substeps_sedi = 1;
  for (int k = nz; k >= 1; --k) {
    if (ri[k] > PS(r1)) {
      const double pi_ = tempo_pow(ilami[k], PS(bv_i));  // one evaluation for vt and vtn
      vt[k] = rhof[k] * PS(av_i) * PA1(cig, 3) * PS(oig2) * pi_;
      vtn[k] = rhof[k] * PS(av_i) * PA1(cig, 6) / PA1(cig, 7) * pi_;
    } else {
      vt[k] = vt[k + 1];
      vtn[k] = vtn[k + 1];
    }
    if (vt[k] > 1.e-3f) {
      *ktop_sedi = imax(*ktop_sedi, k);
      dz_by_vt = dz1d[k] / vt[k];
      *substeps_sedi = imax(*substeps_sedi, (int)(C.global_dt / dz_by_vt + 1.0f));
    }
  }
  if (*ktop_sedi == nz) *ktop_sedi = nz - 1;
}

// 1923-1955
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1921-1955 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void cloud_fallspeed(const TempoParams& P, const TempoCtx& C, int nz,
    const float* rhof, const float* w1d, const bool* l_qc, const float* rc,
    const float* nc, const double* ilamc, const float* dz1d, float* vt, float* vtn,
    int* ktop_sedi) {
  float hgt;
  int nu_c;
#if TEMPO_WARP
  // warp: the 500 m height sum is a serial prefix (lead lane, then shared);
  // the levels at or below ktop then evaluate independently.
  TEMPO_SYNC();
  int ktop = 1;
  if (TEMPO_LEAD) {
    hgt = 0.0f;
    for (int k = 1; k <= nz - 1; ++k) {
      if (rc[k] > PS(r2)) ktop = k;
      hgt = hgt + dz1d[k];
      if (hgt > 500.0f) break;
    }
  }
  *ktop_sedi = ktop = __shfl_sync(TEMPO_FULL, ktop, 0);
  TEMPO_LEVELS(k, ktop) {
    if (rc[k] > PS(r1) && w1d[k] < 0.1f) {
      nu_c = get_nuc(P, nc[k]);
      const double pc = tempo_pow(ilamc[k], PS(bv_c));  // one evaluation for vt and vtn
      vt[k] = rhof[k] * PS(av_c) * PA2(ccg, 5, nu_c) * PA1(ocg2, nu_c) * pc;
      vtn[k] = rhof[k] * PS(av_c) * PA2(ccg, 4, nu_c) * PA1(ocg1, nu_c) * pc;
    }
  }
  TEMPO_SYNC();
  return;
#endif
  *ktop_sedi = 1;
  hgt = 0.0f;
  for (int k = 1; k <= nz - 1; ++k) {
    if (rc[k] > PS(r2)) *ktop_sedi = k;
    hgt = hgt + dz1d[k];
    if (hgt > 500.0f) break;
  }
  for (int k = *ktop_sedi; k >= 1; --k) {
    if (rc[k] > PS(r1) && w1d[k] < 0.1f) {
      nu_c = get_nuc(P, nc[k]);
      const double pc = tempo_pow(ilamc[k], PS(bv_c));  // one evaluation for vt and vtn
      vt[k] = rhof[k] * PS(av_c) * PA2(ccg, 5, nu_c) * PA1(ocg2, nu_c) * pc;
      vtn[k] = rhof[k] * PS(av_c) * PA2(ccg, 4, nu_c) * PA1(ocg1, nu_c) * pc;
    }
  }
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
template<class T> TEMPO_HD void tempo_copy(T* dst,const T* src,int n) {TEMPO_LEVELS(k,n)dst[k]=src[k];}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
template<class T,class V> TEMPO_HD void tempo_fill(T* dst,V v,int n) {TEMPO_LEVELS(k,n)dst[k]=T(v);}

// ---------------------------------------------------------------------------
// tempo_main, lines 74-825.  Optional Fortran arguments are nullable
// pointers (nc1d, nwfa1d, nifa1d, ng1d, qb1d).  The boundary-layer, radiation
// and cloud-fraction optional arrays follow the MPAS driver. The MYNN cloud
// arrays are consumed by tempo_subgrid_cloud_radii after the main processes.
// ---------------------------------------------------------------------------
TEMPO_HD float tempo_cloud_number(float,float,float,float,float,float,float);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:73-825 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void tempo_main_column(const TempoParams& P, const TempoCtx& C, int nz,
    TempoWork& W, float* qv1d, float* qc1d, float* qi1d, float* qr1d, float* qs1d,
    float* qg1d, float* qb1d, float* ni1d, float* nr1d, float* nc1d, float* ng1d,
    float* nwfa1d, float* nifa1d, float* t1d, const float* p1d, const float* w1d,
    const float* dz1d) {
  // TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:75-975 (tempo_main).
  TempoTend& T = W.T;
  const int nz1 = nz + 1;
  int substeps_sedi, ktop_sedi, n;
  float semi_sedi_factor; float  tempc; float  tc0;
  bool do_micro = false, supersaturated = false;
  const float t0 = PS(t0);
  // params:451: roverrv = rdry / rv (tempo_init); the blob's v3.0.3 slot (0.622) is not read here.
  const float roverrv = PS(rdry) / PS(rv);
  const float mtl = PS(meters3_to_liters);
  const float rho_gh = PA1(rho_g, PI(nrhg));
  // warp: the column scalars of W are written by the lead lane only.
  if (TEMPO_LEAD) W.did_micro = 0;

#define TEMPO_TT_ZERO(name) TEMPO_SLOTS(k, 0, TEMPO_NZP) T.name[k] = 0.0f;
  TEMPO_TEND_FIELDS(TEMPO_TT_ZERO)
#undef TEMPO_TT_ZERO

  TEMPO_LEVELS(k, nz) {
    W.tten[k] = 0.0f; W.qvten[k] = 0.0f; W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f;
    W.qsten[k] = 0.0f; W.qgten[k] = 0.0f; W.qhten[k] = 0.0f; W.ngten[k] = 0.0f; W.qbten[k] = 0.0f;
    W.niten[k] = 0.0f; W.nrten[k] = 0.0f; W.ncten[k] = 0.0f; W.nwfaten[k] = 0.0f; W.nifaten[k] = 0.0f;
    W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
    W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.smoz[k] = 0.0f; W.ns[k] = 0.0f;
    W.vtboost[k] = 1.0f;
  }
  TEMPO_LEVELS(k, nz1) {
    W.vtrr[k] = 0.0f; W.vtnr[k] = 0.0f; W.vtrs[k] = 0.0f; W.vtri[k] = 0.0f; W.vtni[k] = 0.0f;
    W.vtrc[k] = 0.0f; W.vtnc[k] = 0.0f; W.vtrg[k] = 0.0f; W.vtng[k] = 0.0f; W.vtrh[k] = 0.0f;
  }
  if (TEMPO_LEAD) {
    W.rain_precip = 0.0f;
    W.cloud_precip = 0.0f;
    W.ice_liquid_equiv_precip = 0.0f;
    W.snow_liquid_equiv_precip = 0.0f;
    W.graupel_liquid_equiv_precip = 0.0f;
    W.hail_liquid_equiv_precip = 0.0f;
    W.frz_rain_precip = 0.0f;
    W.frozen_fraction = 0.0f;
  }
  TEMPO_LEVELS(k, nz) {
    W.refl10cm[k] = -35.0f; W.re_cloud[k] = 0.0f; W.re_ice[k] = 0.0f; W.re_snow[k] = 0.0f;
    W.max_hail_diameter[k] = 0.0f; W.rain_med_vol_diam[k] = 0.0f; W.graupel_med_vol_diam[k] = 0.0f;
    W.cloud_number_mixing_ratio[k] = 0.0f;
  }

  // warp: the tendency slots above were zeroed lane-strided, not by level owner.
  TEMPO_SYNC();

  // initialization
  TEMPO_LEVELS(k, nz) {
    W.temp[k] = t1d[k];
    W.qv[k] = tmax(PS(min_qv), qv1d[k]);
    W.pres[k] = p1d[k];
    W.rho[k] = roverrv * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + roverrv));
  }

  // warp: the first-step column sums and the aerosol height profiles are
  // serial prefixes; the lead lane evaluates them for the whole column.
  TEMPO_SYNC();
  if (TEMPO_LEAD) {
  if (nwfa1d != nullptr) {
    if (C.first_call_main) {
      float s = 0.0f;
      for (int k = 1; k <= nz; ++k) s += nwfa1d[k];
      if (s < PS(eps)) init_water_friendly_aerosols(P, nz, dz1d, W.nwfa);
    }
  } else {
    init_water_friendly_aerosols(P, nz, dz1d, W.nwfa);
  }
  if (nifa1d != nullptr) {
    if (C.first_call_main) {
      float s = 0.0f;
      for (int k = 1; k <= nz; ++k) s += nifa1d[k];
      if (s < PS(eps)) init_ice_friendly_aerosols(P, nz, dz1d, W.nifa);
    }
  } else {
    init_ice_friendly_aerosols(P, nz, dz1d, W.nifa);
  }
  }
  TEMPO_SYNC();

  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, qr1d, nr1d, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, qi1d, ni1d, W.ri, W.ni, W.qiten, W.niten, W.ilami);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, qs1d, W.rs, W.qsten);
  TEMPO_LEVELS(k, nz) {
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1f, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], &W.ns[k], &W.smo0[k], &W.smo1[k],
                   &W.smo2[k], &W.smoe[k], &W.smof[k], &W.smog[k], nullptr);
    }
  }

  // main:383-397. MPAS leaves the internal ml_for_nc flag at false; land is absent in the MPAS driver, so the
  // single-moment value is tempo_cfgs%get_nc_val (single_moment_nc_opt 'land': nt_c_l).
  const float * ncsave = nullptr;
  if (nc1d == nullptr) {
    TEMPO_LEVELS(k, nz) {
      if(C.ml_for_nc_flag) {
        float x=qc1d[k]<=1.e-12f?0.0f:qc1d[k];
        W.nc[k]=tempo_cloud_number(x,qr1d[k],qi1d[k],qs1d[k],W.pres[k],W.temp[k],w1d[k])*W.rho[k];
      } else W.nc[k] = PS(nt_c_l);
    }
    tempo_copy(W.ncsave, W.nc, nz);
    ncsave = W.ncsave;
  }

  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);

  // warp: first-step sums and graupel_init on the lead lane (serial prefix).
  TEMPO_SYNC();
  if (C.first_call_main && TEMPO_LEAD) {
    if (ng1d != nullptr && qb1d != nullptr) {
      float sq = 0.0f; float  sn = 0.0f; float  sb = 0.0f;
      for (int k = 1; k <= nz; ++k) { sq += qg1d[k]; sn += ng1d[k]; sb += qb1d[k]; }
      if (sq > PS(r1) && sn < PS(eps) && sb < PS(eps)) graupel_init(P, nz, W.rho, qg1d, ng1d, qb1d);
    }
  }
  TEMPO_SYNC();
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);

#define TEMPO_THERMO()                                                                      \
  thermo_vars(P, nz, W.qv, W.temp, W.pres, W.rho, W.rhof, W.rhof2, W.qvs, W.delqvs, W.qvsi, \
              W.satw, W.sati, W.ssatw, W.ssati, W.diffu, W.visco, W.vsc2, W.ocp, W.lvap,    \
              W.tcond, W.lvt2, &supersaturated)
  TEMPO_THERMO();

  // main:425-445: zero hail, optionally split graupel into hail
  TEMPO_LEVELS(k, nz) {
    W.qh1d[k] = 0.0f;
    W.hail_fraction[k] = 0.0f;
    W.l_qh[k] = false;
  }
  if (C.hailhyperaware_flag) {
    compute_hail_fraction(P, nz, W.l_qg, qg1d, W.mvd_g, W.idx_bg, qr1d, qc1d, W.temp, W.tcond, W.diffu,
                          W.delqvs, W.qh1d, W.hail_fraction);
  }
  hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.qh1d, W.rh, W.nh, W.qhten, W.ilamh);
  TEMPO_LEVELS(k, nz) {
    if (W.l_qh[k]) {
      qg1d[k] = qg1d[k] - W.qh1d[k];
      if (ng1d != nullptr) ng1d[k] = ng1d[k] - W.nh[k] / W.rho[k];
      if (qb1d != nullptr) qb1d[k] = qb1d[k] - W.qh1d[k] * mtl / rho_gh;
      W.qgten[k] = 0.0f;
      W.ngten[k] = 0.0f;
      W.qbten[k] = 0.0f;
    }
  }
  // if hail is split from graupel, rerun the graupel checks
  if (tempo_any(W.l_qh, nz)) {
    graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                             W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  }

  TEMPO_LEVELS(k, nz) {
    W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f; W.qsten[k] = 0.0f; W.qgten[k] = 0.0f;
    W.qhten[k] = 0.0f; W.ngten[k] = 0.0f; W.qbten[k] = 0.0f; W.niten[k] = 0.0f; W.nrten[k] = 0.0f;
    W.ncten[k] = 0.0f;
  }

  do_micro = tempo_any(W.l_qc, nz) || tempo_any(W.l_qr, nz) || tempo_any(W.l_qi, nz) ||
             tempo_any(W.l_qs, nz) || tempo_any(W.l_qg, nz) || tempo_any(W.l_qh, nz);
  if (!do_micro && !supersaturated) return;
  if (TEMPO_LEAD) W.did_micro = 1;

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
            W.l_qh, W.rh, W.nh, W.ilamh, W.hail_fraction, C.hailhyperaware_flag, T);
    if (nwfa1d != nullptr || nifa1d != nullptr) {
      aerosol_scavenging(P, C, nz, W.temp, W.rho, W.rhof, W.visco, W.nwfa, W.nifa, W.l_qr, W.nr,
                         W.ilamr, W.mvd_r, W.l_qs, W.rs, W.smob, W.smoc, W.smoe, W.l_qg, W.rg, W.ng,
                         W.ilamg, W.idx_bg, T);
    }
  }

  check_over_depletion(P, C, nz, W.rho, W.temp, W.qvsi, W.qv, W.l_qc, W.rc, W.l_qi, W.ri, W.l_qr,
                       W.rr, W.l_qs, W.rs, W.l_qg, W.rg, W.l_qh, W.rh, T);
  sum_tendencies(P, nz, W.rho, W.temp, W.idx_bg, W.lvap, W.ocp, T, W.tten, W.qvten, W.qcten,
                 W.ncten, W.qiten, W.niten, W.qsten, W.qrten, W.nrten, W.qgten, W.ngten, W.qbten, W.qhten);

  TEMPO_LEVELS(k, nz) {
    W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
    tempc = W.temp[k] - t0;
    W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
    W.rho[k] = roverrv * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + roverrv));
    W.nwfaten[k] = W.nwfaten[k] - (T.pna_rca[k] + T.pna_sca[k] + T.pna_gca[k] + T.pni_iha[k]) / W.rho[k];
    W.nifaten[k] = W.nifaten[k] - (T.pnd_rcd[k] + T.pnd_scd[k] + T.pnd_gcd[k] + T.pni_inu[k]) / W.rho[k];
  }
  (void)tempc;

  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);

  // temporaries so the 1d variables are not updated here
  float * xncx = nullptr;
  float * xngx = nullptr;
  float * xqbx = nullptr;
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
  TEMPO_LEVELS(k, nz) {
    W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
    W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.ns[k] = 0.0f;
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1f, W.temp[k] - t0);
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
    TEMPO_LEVELS(k, nz) {
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
    TEMPO_LEVELS(k, nz) {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = roverrv * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + roverrv));
    }
    TEMPO_THERMO();
  }

  // rain evaporation
  if (!C.turn_off_micro_flag) {
    rain_evaporation(P, C, nz, W.rho, W.temp, W.ssatw, W.lvap, W.tcond, W.diffu, W.vsc2, W.rhof2,
                     W.qv, W.qvs, W.l_qr, W.rr, W.nr, W.ilamr, T);
    TEMPO_LEVELS(k, nz) {
      W.qrten[k] = W.qrten[k] - T.prv_rev[k];
      W.qvten[k] = W.qvten[k] + T.prv_rev[k];
      W.nrten[k] = W.nrten[k] - T.pnr_rev[k];
      W.nwfaten[k] = W.nwfaten[k] + T.pnr_rev[k];
      W.tten[k] = W.tten[k] - W.lvap[k] * W.ocp[k] * T.prv_rev[k];
    }
    tempo_copy(W.xrx, qr1d, nz);
    tempo_copy(W.xnx, nr1d, nz);
    rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
    TEMPO_LEVELS(k, nz) {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = roverrv * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + roverrv));
    }
    TEMPO_THERMO();
  }

  // sedimentation: rain (main:644-683; the substeps no longer recompute the fall speeds)
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0f;
  if (tempo_any(W.l_qr, nz)) {
    rain_fallspeed(P, C, nz, W.rhof, W.l_qr, W.rr, W.ilamr, dz1d, W.vtrr, W.vtnr, &substeps_sedi, &ktop_sedi);
    if (!TEMPO_WARP && C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rr, W.qrten, W.vtrr, substeps_sedi, PS(r1), &W.rain_precip);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.nr, W.nrten, W.vtnr, substeps_sedi, PS(r2), nullptr);
        tempo_fill(W.vtrr, 0.0f, nz1);
        tempo_fill(W.vtnr, 0.0f, nz1);
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

  // graupel (main:685-743)
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0f;
  const bool has_qb1d = (qb1d != nullptr);
  const float rb_limit = PS(meters3_to_liters) * PS(r1) / PA1(rho_g, PI(nrhg));
  if (tempo_any(W.l_qg, nz)) {
    graupel_fallspeed(P, C, nz, W.rhof, W.rho, W.visco, W.l_qg, W.rg, W.rb, has_qb1d, W.idx_bg,
                      W.ilamg, dz1d, W.vtrg, W.vtng, &substeps_sedi, &ktop_sedi);
    if (!TEMPO_WARP && C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rg, W.qgten, W.vtrg, substeps_sedi, PS(r1), &W.graupel_liquid_equiv_precip);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.ng, W.ngten, W.vtng, substeps_sedi, PS(r2), nullptr);
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rb, W.qbten, W.vtrg, substeps_sedi, rb_limit, nullptr);
        tempo_fill(W.vtrg, 0.0f, nz1);
        tempo_fill(W.vtng, 0.0f, nz1);
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

  // hail (main:745-776)
  ktop_sedi = 1;
  substeps_sedi = 1;
  semi_sedi_factor = 10.0f;
  if (tempo_any(W.l_qh, nz)) {
    hail_fallspeed(P, C, nz, W.rhof, W.l_qh, W.rh, W.ilamh, dz1d, W.vtrh, &substeps_sedi, &ktop_sedi);
    if (!TEMPO_WARP && C.semi_sedi_flag) {
      substeps_sedi = imax((int)(substeps_sedi / semi_sedi_factor) + 1, 1);
      for (n = 1; n <= substeps_sedi; ++n) {
        semilagrangian_sedimentation(P, C, nz, W, dz1d, W.rho, W.rh, W.qhten, W.vtrh, substeps_sedi, PS(r1), &W.hail_liquid_equiv_precip);
        tempo_fill(W.vtrh, 0.0f, nz1);
        tempo_copy(W.xrx, W.qh1d, nz);
        hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.xrx, W.rh, W.nh, W.qhten, W.ilamh);
        hail_fallspeed(P, C, nz, W.rhof, W.l_qh, W.rh, W.ilamh, dz1d, W.vtrh, nullptr, nullptr);
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

  // ice (main:791-803: every substep, v3.0.3 ran one steps-divided pass)
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
  TEMPO_LEVELS(k, nz) {
    t1d[k] = t1d[k] + W.tten[k] * C.global_dt;
    qv1d[k] = tmax(PS(min_qv), (qv1d[k] + W.qvten[k] * C.global_dt));
    W.rho[k] = roverrv * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + roverrv));
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
  TEMPO_LEVELS(k, nz) {
    W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
    W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.smoz[k] = 0.0f; W.ns[k] = 0.0f;
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1f, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                   &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  hail_check_and_update(P, C, nz, W.rho, W.l_qh, W.qh1d, W.rh, W.nh, W.qhten, W.ilamh);

  // diagnostics (warp: the lead lane owns level 1 and the precipitation scalars)
  TEMPO_SYNC();
  if (TEMPO_LEAD) {
  W.frozen_fraction =
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip
       + W.hail_liquid_equiv_precip) /
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip
       + W.hail_liquid_equiv_precip + W.rain_precip + PS(r1));
  // combine graupel and hail precipitation
  W.graupel_liquid_equiv_precip = W.graupel_liquid_equiv_precip + W.hail_liquid_equiv_precip;
  freezing_rain(P, W.temp[1], W.rain_precip, W.cloud_precip, &W.frz_rain_precip);
  }
  TEMPO_LEVELS(k, nz) {
    W.cloud_number_mixing_ratio[k] = W.nc[k] * W.rho[k];
    W.rain_med_vol_diam[k] = W.mvd_r[k];
    W.graupel_med_vol_diam[k] = W.mvd_g[k];
  }
  if (C.refl10cm_flag) {
    reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp, W.l_qr, W.rr, W.nr, W.ilamr,
                      W.l_qs, W.rs, W.smoc, W.smob, W.smoz, W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg,
                      W.l_qh, W.nh, W.ilamh, W.refl10cm);
  }
  // main:913-937: graupel and hail combined again for the hail-size diagnostic and the returned state
  TEMPO_LEVELS(k, nz) {
    if (W.l_qh[k]) {
      qg1d[k] = qg1d[k] + W.qh1d[k];
      if (ng1d != nullptr && W.ngten[k] >= 0.0f) ng1d[k] = ng1d[k] + W.nh[k] / W.rho[k];
      if (qb1d != nullptr) qb1d[k] = qb1d[k] + W.qh1d[k] * mtl / rho_gh;
    }
    W.qgten[k] = 0.0f;
    W.ngten[k] = 0.0f;
    W.qbten[k] = 0.0f;
  }
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

// TEMPO 9adf9ef MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:28-81,104-203.
// Public TEMPO constants and network; covered by NOTICE-TEMPO.txt.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_trans_mean(int i) {
  const float values[7] = {0.000191556196486247f, 3.58145042772654e-05f,     3.12611085359273e-07f, 5.74303078738579e-05f, 84191.2092225319f,     279.070551773565f, 0.123679354084004f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_trans_var(int i) {
  const float values[7] = {5.78143564171777e-08f, 3.22834309750552e-08f,     6.45745893307455e-11f, 4.16625579383794e-08f, 215694631.771185f,     94.6576255386858f, 0.384841247662964f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_w00(int i) {
  const float values[168] = {-2.006957f, -0.2812008f, -0.339073f, 1.596426f, 2.395225f, 1.76315f,     -0.0626798f, 1.267002f, -0.02234177f, -6.522605e-33f, 0.4792154f, -0.1253034f,     -3.217191f, -3.092887f, -0.0863651f, 1.071625f, 0.09741028f, 0.2255831f,     -0.6929023f, -0.02693799f, -3.432344e-33f, -0.8791879f, -0.9359049f, 1.083484f,     -0.07909214f, -0.0122418f, -0.02815927f, 0.1676407f, 0.08252326f, 0.6697816f,     -0.4019359f, 0.4687141f, 0.001813132f, -9.792186e-33f, 0.0409322f, 0.0113192f,     -0.01354596f, 0.00307771f, -0.4635534f, 0.03835761f, -0.1015553f, 0.7316446f,     -0.05791711f, -0.0002690362f, -7.920147e-33f, -0.1216918f, -0.3190572f, 0.09809405f,     -0.16476f, -0.03387314f, 0.005422261f, 0.04043967f, 0.03901243f, 0.07444729f,     0.01954299f, 0.06918761f, 0.04823543f, -8.637957e-33f, 0.06371575f, -0.09250915f,     -1.109653f, -1.373999f, -0.2412623f, -0.04482195f, 0.1584691f, 0.06353725f,     0.0006248798f, 0.04593191f, -8.878673e-33f, -0.4988684f, 0.01110262f, 0.04623203f,     0.006581791f, 0.03536217f, -0.1890567f, -0.08839592f, 0.1327181f, 0.03478973f,     -0.1565902f, -0.100401f, -0.1179777f, -8.879818e-33f, -0.1383738f, 0.02847495f,     -0.005902881f, 0.005615512f, -0.6308192f, -0.02431803f, -0.141971f, -0.3490018f,     -0.9850957f, -0.1449479f, -8.059166e-33f, -0.1186465f, -1.165381f, 0.069015f,     0.003388841f, -0.04041302f, 0.1638467f, 0.1147008f, -0.04833491f, -0.07755993f,     -0.5137688f, 0.04546477f, 0.04101883f, 6.752353e-33f, 0.3541977f, 0.04880851f,     -0.00102834f, -0.01280629f, -0.1116254f, -0.02204754f, 0.07100908f, 0.2354002f,     0.07129629f, 0.2489657f, 8.080785e-33f, -0.03449865f, 0.06037927f, -0.02023619f,     0.7779589f, 0.04680278f, 0.7492616f, 0.6545208f, -1.09497f, -1.176524f,     -0.451585f, 0.881124f, -0.4551499f, -8.708624e-33f, 1.006558f, -0.04979523f,     -0.0006915367f, -0.002993054f, 0.01654614f, -0.07141764f, -0.2216591f, 0.8637336f,     0.8358089f, -0.7576646f, 8.186339e-33f, 0.1161914f, 0.7121871f, -1.146734f,     0.03925339f, 0.9975697f, -0.06953461f, -0.07598846f, -0.06418022f, -0.01897495f,     -0.03612464f, -0.06703389f, -0.103049f, 9.364858e-33f, -0.03949085f, 1.005938f,     0.0001625419f, -0.01027657f, 0.03823901f, 0.3197246f, -0.1160718f, 0.04449431f,     0.009831783f, 0.6551342f, -8.470687e-33f, 0.1374123f, -0.01782138f, 0.01719949f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_w01(int i) {
  const float values[24] = {-4.045869f, 0.558111f, -1.567351f, -1.64972f, 2.748608f,     -1.909901f, 0.3955558f, 1.507247f, 0.3599722f, -0.0001173223f, -0.9398569f,     -0.6867028f, -61.72853f, -63.13766f, 1.165811f, -0.6848684f, 0.1931683f,     1.1208f, 2.63087f, 0.740169f, -21.62499f, 1.545568f, 3.575141f, -1.299604f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_b00(int i) {
  const float values[24] = {-0.9842531f, 0.3064759f, -0.4500185f, 1.28336f, 1.384105f, 0.528031f,     0.8453538f, 1.579872f, 2.245679f, -0.008679952f, 0.4549862f, -0.136581f,     -2.576741f, -2.483647f, -0.2089484f, 0.7607977f, 1.847745f, 0.7316047f,     -0.287945f, 2.227298f, -2.314714f, -0.2561245f, -0.6993448f, -0.1359731f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_nc_ml_b01(int i) {
  const float values[1] = {1.572826f};
  return values[i];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:102-195 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HD float tempo_cloud_number(float qc, float qr, float qi, float qs,
                                  float pressure, float temperature, float w) {
  const float input[7] = {qc, qr, qi, qs, pressure, temperature, w};
  float transformed[7]; float  activated[24];
  for (int i = 0; i < 7; ++i)
    transformed[i] = (input[i] - tempo_nc_ml_trans_mean(i)) / sqrt(tempo_nc_ml_trans_var(i));
  for (int node = 0; node < 24; ++node) {
    float value = 0.0f;
    for (int i = 0; i < 7; ++i) value = value + tempo_nc_ml_w00(node + 24*i) * transformed[i];
    activated[node] = tmax(value + tempo_nc_ml_b00(node), 0.0f);
  }
  float value = 0.0f;
  for (int node = 0; node < 24; ++node) value = value + tempo_nc_ml_w01(node) * activated[node];
  value = tmax(value + tempo_nc_ml_b01(0), 0.0f);
  const float p = tmin(9.3010299957f, tmax(-6.0f, value));
  float bias = 1.0f;
  if (p >= 0.0f && p < 3.0f) {
    const float p2 = p*p; const float  p3 = p2*p; const float  p4 = p2*p2; const float  p5 = p2*p3;
    bias = -0.2704f*p5 + 1.838f*p4 - 5.127f*p3 + 8.547f*p2 - 8.439f*p + 4.297f;
  }
  return bias * tempo_pow(10.0f, p);
}

// Native TEMPO main:801-823. Update internal cloud moments for radiation
// only; the prognostic cloud and droplet number have already been committed.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:104-196 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void tempo_subgrid_cloud_radii(const TempoParams& P, const TempoCtx& C,
    int nz, TempoWork& W, const float* qc, const float* qr, const float* qi,
    const float* qs, const float* nc, const float* w,
    const float* qc_bl, const float* fraction_bl, int nt, int c0) {
  if (!W.did_micro || !C.re_cloud_flag || !C.re_ice_flag || !C.re_snow_flag) return;
  TEMPO_LEVELS(k, nz) {
    const long long index = (long long)(k-1)*nt + c0;
    W.xrx[k] = qc[k];
    if (qc[k] <= PS(r1) && qc_bl[index] > 1.0e-9f && fraction_bl[index] > 0.0f)
      W.xrx[k] += qc_bl[index] / fraction_bl[index];
    if (W.xrx[k] <= 1.0e-12f) W.xrx[k] = 0.0f;
    W.xncx[k] = tempo_cloud_number(W.xrx[k], qr[k], qi[k], qs[k], W.pres[k], W.temp[k], w[k]);
    W.qcten[k] = 0.0f;
    W.ncten[k] = 0.0f;
  }
  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, W.xncx, nullptr,
                        W.rc, W.nc, W.qcten, W.ncten, W.ilamc, W.mvd_c);
  effective_radius(P, C, nz, W.temp, W.l_qc, W.nc, W.ilamc, W.l_qi, W.ilami,
                   W.l_qs, W.rs, W.re_cloud, W.re_ice, W.re_snow);
}

'''

_DRIVER = r"""
// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_driver.F90:191-521 (flags[4] = hailhyperaware).
struct TempoState {float f[17][TEMPO_NZP];};
TEMPO_HD void tempo_call(int nz, float dt, TempoState& s, TempoWork& work,
        const TempoBlob* blob, const int* flags, int first, const float* bl,
        float* diag, int presence) {
    TempoParams P={blob};
    TempoCtx ctx={dt, 1.0f/dt, flags[0],flags[1],0,flags[3],0,1,1,1,1,1,1,first,0,flags[4]};
    tempo_main_column(P,ctx,nz,work,s.f[4],s.f[5],s.f[7],s.f[6],s.f[8],s.f[9],
                      (presence&16)?s.f[16]:nullptr,s.f[10],s.f[11],
                      (presence&1)?s.f[12]:nullptr,(presence&8)?s.f[15]:nullptr,
                      (presence&2)?s.f[13]:nullptr,(presence&4)?s.f[14]:nullptr,
                      s.f[0],s.f[1],s.f[2],s.f[3]);
    if(flags[2] && bl) tempo_subgrid_cloud_radii(P,ctx,nz,work,s.f[5],s.f[6],s.f[7],
                                 s.f[8],(presence&1)?s.f[12]:nullptr,s.f[2],bl,bl+nz,1,0);
    for(int k=0;k<nz*15;++k)diag[k]=0.0f;
    float maxhail=work.max_hail_diameter[1];
    for(int k=2;k<=nz;++k)maxhail=tmax(maxhail,work.max_hail_diameter[k]);
    const float sd[8]={work.rain_precip,work.ice_liquid_equiv_precip,
        work.snow_liquid_equiv_precip,work.graupel_liquid_equiv_precip,
        work.frz_rain_precip,work.frozen_fraction,work.max_hail_diameter[1],maxhail};
    for(int f=0;f<8;++f)diag[f*nz]=sd[f];
    const float* cd[7]={work.refl10cm,work.re_cloud,work.re_ice,work.re_snow,
                       work.rain_med_vol_diam,work.graupel_med_vol_diam,
                       work.cloud_number_mixing_ratio};
    for(int f=0;f<4;++f)for(int k=1;k<=nz;++k)diag[(8+f)*nz+k-1]=cd[f][k];
}
// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_main.F90:75-975.
extern "C" __global__ void tempo_sizes(unsigned long long* out) {
    out[0]=sizeof(TempoWork);out[1]=sizeof(TempoState);out[2]=sizeof(TempoBlob);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438.
extern "C" __global__ void tempo_columns(int nz,int columns,float dt,float* fields,
    const float* bl,float* diag,const TempoBlob* blob,const int* flags,int first,
    TempoState* states,TempoWork* work,float* blwork,float* diagwork,int presence) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=columns)return;
    TempoState& s=states[c];
    for(int f=0;f<17;++f)for(int k=1;k<=nz;++k)
        s.f[f][k]=fields[((long long)f*nz+k-1)*columns+c];
    float* b=blwork+c*nz*2;float* d=diagwork+c*nz*15;
    if(bl)for(int f=0;f<2;++f)for(int k=0;k<nz;++k)
        b[f*nz+k]=bl[((long long)f*nz+k)*columns+c];
    tempo_call(nz,dt,s,work[c],blob,flags,first,bl?b:nullptr,d,presence);
    for(int f=0;f<17;++f)for(int k=1;k<=nz;++k)
        fields[((long long)f*nz+k-1)*columns+c]=s.f[f][k];
    for(int f=0;f<15;++f)for(int k=0;k<nz;++k)
        diag[((long long)f*nz+k)*columns+c]=d[f*nz+k];
}
"""

_WARP = r"""
// RRFS ufs-community/MPAS-Model de016e0ff4 mpas_atmphys_driver_microphysics.F:592-646;
// TEMPO tempo_v3.2.5 c62699f src/module_mp_tempo_driver.F90:191-521.
struct TempoState {float f[17][TEMPO_NZP];};
extern "C" __global__ void tempo_warp_sizes(unsigned long long* out) {
    out[0]=sizeof(TempoWork);out[1]=sizeof(TempoState);out[2]=sizeof(TempoBlob);
}
// Column order (scheduling only).  Columns are independent (own state, work zeroed every substep), so the
// order they are taken in changes no word.  The slot's cost is its slowest columns (heavy rain and graupel
// sedimentation: up to ~3x the median); taking them first keeps them out of the tail.  The order comes from
// the previous call's per-column cycle counts (cost), bucketed heaviest first: 128 buckets, four per octave.
TEMPO_HD int tempo_order_bucket(unsigned c) {
    if(c==0u)return 127;
    const int e=31-__clz(c);
    const int key=(e<<2)|(int)((e>=2?(c>>(e-2)):(c<<(2-e)))&3u);
    return 127-imin(key,127);
}
extern "C" __global__ void tempo_order_hist(const unsigned* cost,int n,unsigned* hist) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&hist[tempo_order_bucket(cost[i])],1u);
}
extern "C" __global__ void tempo_order_scan(unsigned* hist) {
    if(blockIdx.x==0 && threadIdx.x==0){unsigned run=0u;for(int b=0;b<128;++b){const unsigned h=hist[b];hist[b]=run;run+=h;}}
}
extern "C" __global__ void tempo_order_scatter(const unsigned* cost,int n,unsigned* start,int* order) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)order[atomicAdd(&start[tempo_order_bucket(cost[i])],1u)]=i;
}
extern "C" __global__ void tempo_warp_steps(int nz,int columns,int steps,float dt,
    float** f,const float* bl,float** s,float** profiles,const TempoBlob* blob,
    int aero,int hail,int ml_bl,int melt,int hyper,int first,int presence,
    TempoState* states,TempoWork* work,float* blwork,const int* order,unsigned* cost,int radii) {
    const int lane=threadIdx.x&31;
    const int warp=(int)((blockIdx.x*blockDim.x+threadIdx.x)>>5);
    const int warps=(int)((gridDim.x*blockDim.x)>>5);
    TempoState& st=states[warp];TempoWork& W=work[warp];float* b=blwork+(long long)warp*nz*2;
    TempoParams P={blob};
    const bool has_bl=bl!=nullptr;
    for(int i=warp;i<columns;i+=warps){
        const int c=order?order[i]:i;
        const long long started=clock64();
        // tempo_pack: theta*exner into slot 0, absent carriers zero.
        TEMPO_LEVELS(k,nz){
            const long long x=(long long)(k-1)*columns+c;
            st.f[0][k]=f[0][x]*f[17][x];
            for(int j=1;j<17;++j)st.f[j][k]=f[j]?f[j][x]:0.0f;
            if(has_bl){b[k-1]=bl[x];b[nz+k-1]=bl[(long long)nz*columns+x];}
        }
        for(int sub=0;sub<steps;++sub){
            // tempo_columns ran on freshly zeroed work every substep.
            unsigned long long* w8=reinterpret_cast<unsigned long long*>(&W);
            // tempo_main_column clears every TempoTend element before use.
            const int tb=(int)((char*)&W.T-(char*)&W)/8;
            const int te=tb+(int)(sizeof(TempoTend)/8);
            for(int i=lane;i<(int)(sizeof(TempoWork)/8);i+=32)
                if(i<tb || i>=te)w8[i]=0ULL;
            TEMPO_SYNC();
            // The reflectivity, effective-radius and hail-size diagnostics (and the PBL subgrid radii) write only
            // work-area profiles that the next substep's zeroing discards; the slot keeps the LAST substep's
            // (profiles, s[8], s[9]). They feed nothing else, so earlier substeps skip them: same words.
            // radii=0 (launch_mpas: no reader of this call's effective radii, see radii_due) also skips the
            // effective and PBL subgrid radii of the last substep and leaves re_* untouched; they write only
            // W.re_* and W's subgrid scratch, which nothing else reads: the state words are the same.
            const int last=(sub+1==steps);
            const int re=last && radii;
            TempoCtx ctx={dt,1.0f/dt,aero,hail,0,melt,0,1,last,re,re,re,last,first,0,hyper};
            tempo_main_column(P,ctx,nz,W,st.f[4],st.f[5],st.f[7],st.f[6],st.f[8],st.f[9],
                              (presence&16)?st.f[16]:nullptr,st.f[10],st.f[11],
                              (presence&1)?st.f[12]:nullptr,(presence&8)?st.f[15]:nullptr,
                              (presence&2)?st.f[13]:nullptr,(presence&4)?st.f[14]:nullptr,
                              st.f[0],st.f[1],st.f[2],st.f[3]);
            TEMPO_SYNC();
            if(ml_bl && has_bl && re){
                tempo_subgrid_cloud_radii(P,ctx,nz,W,st.f[5],st.f[6],st.f[7],st.f[8],
                    (presence&1)?st.f[12]:nullptr,st.f[2],b,b+nz,1,0);
                TEMPO_SYNC();
            }
            // tempo_step_diags, in its order of operations.
            if(TEMPO_LEAD){
                float maxhail=W.max_hail_diameter[1];
                for(int k=2;k<=nz;++k)maxhail=tmax(maxhail,W.max_hail_diameter[k]);
                const float r=W.rain_precip,i=W.ice_liquid_equiv_precip,
                            sn=W.snow_liquid_equiv_precip,g=W.graupel_liquid_equiv_precip;
                s[2][c]=(s[2][c]+i)+sn;s[3][c]=(s[3][c]+i)+sn;
                s[4][c]+=g;s[5][c]+=g;s[0][c]=(((s[0][c]+r)+i)+sn)+g;s[1][c]=(((s[1][c]+r)+i)+sn)+g;
                s[6][c]=W.frozen_fraction;s[7][c]+=W.frz_rain_precip;
                s[8][c]=W.max_hail_diameter[1];s[9][c]=maxhail;
            }
            // tempo_pack's unpack then the next pack: theta = T / exner, T = theta * exner.
            TEMPO_LEVELS(k,nz){
                const long long x=(long long)(k-1)*columns+c;
                const float theta=st.f[0][k]/f[17][x];
                if(sub+1<steps)st.f[0][k]=theta*f[17][x];
                else{
                    f[0][x]=theta;
                    for(int j=4;j<17;++j)if(f[j])f[j][x]=st.f[j][k];
                    profiles[0][x]=W.refl10cm[k];
                    if(radii){profiles[1][x]=W.re_cloud[k];profiles[2][x]=W.re_ice[k];profiles[3][x]=W.re_snow[k];}
                }
            }
            TEMPO_SYNC();
        }
        if(cost && lane==0){const long long d=(clock64()-started)>>4;cost[c]=(unsigned)(d>0xffffffffLL?0xffffffffLL:d);}
    }
}
"""


def _column_source(numerics: str | None = None, corrections=()) -> str:
    """The column support for this numerics mode: strict is ``_COLUMN`` byte for byte; fast applies the
    tempo_v303 fast transforms that hold for this column (tempo_fast_v325.fast_column).
    ``corrections``: correct mode's fork ids this column takes (corrected_column); none is ``_COLUMN`` itself."""
    column = corrected_column(corrections)
    if not _base._fast(numerics):
        return column
    from woof.hex.mpasphys.rrfs.tempo_fast_v325 import fast_column
    return fast_column(column)


def cuda_source(device_math_source: str, *, max_levels: int = 100, numerics: str | None = None,
                corrections=()) -> str:
    """The column-per-thread module (replays and tests), as tempo_v303.cuda_source."""
    if not device_math_source.strip():
        raise ValueError("TEMPO requires the shared device mathematics header")
    if type(max_levels) is not int or max_levels < 1:
        raise ValueError("TEMPO column capacity must be a positive integer")
    return (f"#define TEMPO_NZ_MAX {max_levels}\n" + _base._shared_pow_define(device_math_source)
            + device_math_source + _column_source(numerics, corrections) + _DRIVER + _base._mpas_source(numerics))


@lru_cache(maxsize=None)
def _compiled(device: int, device_math_source: str, max_levels: int, corrections: tuple = ()):
    """Compile without CuPy RawModule's appended FTZ=true override (as tempo_v303._compiled)."""
    import cupy as cp
    from cupy.cuda import compiler, function
    with cp.cuda.Device(device):
        image, _ = compiler.compile_using_nvrtc(
            cuda_source(device_math_source, max_levels=max_levels, corrections=corrections),
            options=_hex_numerics.options(COMPILE_OPTIONS), cache_in_memory=True)
        module = function.Module()
        module.load(image)
        sizes = cp.empty(3, dtype=cp.uint64)
        module.get_function("tempo_sizes")((1,), (1,), (sizes,))
        return module, tuple(map(int, sizes.get()))


def warp_source(device_math_source: str, *, max_levels: int = 100, numerics: str | None = None,
                corrections=()) -> str:
    """The warp-per-column module the forecast runs (as tempo_v303.warp_source)."""
    if not device_math_source.strip():
        raise ValueError("TEMPO requires the shared device mathematics header")
    if type(max_levels) is not int or max_levels < 1:
        raise ValueError("TEMPO column capacity must be a positive integer")
    return (f"#define TEMPO_NZ_MAX {max_levels}\n#define TEMPO_WARP 1\n" + _base._shared_pow_define(device_math_source)
            + device_math_source + _column_source(numerics, corrections) + corrected_warp(corrections)
            + _base._mpas_source(numerics))


@lru_cache(maxsize=None)
def _compiled_warp(device: int, device_math_source: str, max_levels: int, corrections: tuple = ()):
    """The warp module, from the AOT kernel pack when it holds this source, else NVRTC (as tempo_v303)."""
    import cupy as cp
    from cupy.cuda import compiler, function
    from woof.hex.cuda_backend.aot import direct_image
    with cp.cuda.Device(device):
        image = direct_image(warp_source(device_math_source, max_levels=max_levels, corrections=corrections),
                             _hex_numerics.options(COMPILE_OPTIONS), compiler._get_arch())
        module = function.Module()
        module.load(image)
        sizes = cp.empty(3, dtype=cp.uint64)
        module.get_function("tempo_warp_sizes")((1,), (1,), (sizes,))
        kernel = module.get_function("tempo_warp_steps")
        from woof.hip_runtime import device_attribute
        sms = device_attribute(cp, "MultiProcessorCount", device)
        from woof.backend import active_blocks_per_multiprocessor
        blocks = active_blocks_per_multiprocessor(kernel.ptr, _base.WARP_BLOCK, 0)
        resident = max(1, sms * blocks) * (_base.WARP_BLOCK // 32)
        return module, tuple(map(int, sizes.get())), resident


def launch_columns(fields, tables, *, dt, flags, device_math_source,
                   first_call_main=False, pbl=None, presence):
    """Run prepared columns in place (one thread per column), as tempo_v303.launch_columns.

    ``flags``: aerosolaware, hailaware, ml_for_bl_nc, refl10cm_from_melting, hailhyperaware.
    ``first_call_main`` is v3.2.5's is_first_step: it applies to every column.
    """
    import cupy as cp
    if fields.ndim != 3 or fields.shape[0] != len(STATE_FIELDS):
        raise ValueError("TEMPO needs 17 prepared carriers, levels, cells")
    if fields.dtype != cp.float32 or not fields.flags.c_contiguous:
        raise ValueError("TEMPO MPAS carriers must be contiguous float32")
    _, nz, cells = fields.shape
    if nz < 1 or cells < 1 or len(flags) != FLAG_COUNT:
        raise ValueError("TEMPO v3.2.5 needs positive extents and five configuration flags")
    module, (work_bytes, state_bytes, blob_bytes) = _compiled(
        cp.cuda.runtime.getDevice(), device_math_source, nz)
    tables = _base._device_tables(module, tables, blob_bytes)
    if pbl is not None and (pbl.shape != (2, nz, cells) or pbl.dtype != cp.float32
                            or not pbl.flags.c_contiguous):
        raise ValueError("TEMPO PBL cloud carriers must be contiguous float32 (2, levels, cells)")
    state = cp.empty(cells*state_bytes, dtype=cp.uint8)
    work = cp.zeros(cells*work_bytes, dtype=cp.uint8)
    blwork = cp.empty(cells*nz*2, dtype=cp.float32)
    diagwork = cp.empty(cells*nz*15, dtype=cp.float32)
    diagnostics = cp.empty((15,nz,cells), dtype=cp.float32)
    config = cp.asarray(flags, dtype=cp.int32)
    module.get_function("tempo_columns")(((cells+31)//32,), (32,), (
        np.int32(nz), np.int32(cells), np.float32(dt), fields,
        pbl if pbl is not None else np.uint64(0), diagnostics, tables, config,
        np.int32(first_call_main), state, work, blwork, diagwork, np.int32(presence)))
    return diagnostics


def launch_mpas(p, *, dt_dyn, flags, bucket, diagnostics, hourly):
    """The fork driver slice with the v3.2.5 column (tempo_v303.launch_mpas with five flags): the warp kernel in
    strict numerics, the phase kernels (tempo_phase_v325) in fast numerics unless WOOF_HEX_TEMPO_PHASE=0.

    RRFS ufs-community/MPAS-Model de016e0ff4 mpas_atmphys_driver_microphysics.F:592-646 (the v3.2.x call:
    is_first_step = itimestep == 1 on every column and substep of the run's first step).
    """
    import cupy as cp
    if len(flags) != FLAG_COUNT:
        raise ValueError("TEMPO v3.2.5 needs five configuration flags")
    fields=p["fields"];nz,cells=fields["qv"].shape;device=fields["qv"].device.id
    if type(p["n_microp"]) is not int or p["n_microp"]<1:
        raise ValueError("native manager must supply a positive microphysics substep count")
    phase=_base._phase_mode()
    # Correct mode's fork ids this column takes (tempo_v303.run_slot sets them); none in exact mode.
    corrections=_corrections_key(p.get("corrections",()))
    if phase:
        # Fast numerics: the (level, cell) phase kernels and the warp tail (tempo_phase_v325), as tempo_v303.
        from woof.hex.mpasphys.rrfs import tempo_phase_v325 as _phase
        pmod,module,sizes,_,_=_phase._modules(device,p["device_math_source"],nz,_phase.inline_mode(),corrections)
        tables=_phase.float_blob(p["tables"])
        if not tables.flags.c_contiguous or tables.nbytes!=sizes[2]:
            raise ValueError("TEMPO table blob has a different layout than the native module")
        _base._set_scalars(pmod,tables);_base._set_scalars(module,tables)
    else:
        module,sizes,resident=_compiled_warp(device,p["device_math_source"],nz,corrections)
        tables=_base._device_tables(module,p["tables"],sizes[2])
    FA,MA,TA,SA=_base.FIELD_ARGUMENTS,_base.MODEL_ARGUMENTS,_base.TEND_ARGUMENTS,_base.SURFACE_ARGUMENTS
    for names,values in ((FA[:12]+FA[17:]+(FA[12:15] if flags[0] else ())+(FA[15:17] if flags[1] else ()),fields),
                         (MA[:8]+MA[13:]+(MA[8:11] if flags[0] else ())+(MA[11:13] if flags[1] else ()),p["model"]),
                         (TA if flags[0] else TA[:9],p["tendencies"]),
                         (SA[:10]+(SA[10:] if flags[0] else ()),p["surface"])):
        absent=[name for name in names if values.get(name) is None]
        if absent:raise ValueError(f"allocated native TEMPO carriers missing: {absent}")
    pa=_base._pointer_array
    f=pa([fields.get(n) for n in FA],shape=(nz,cells),device=device,optional=True)
    if "TP6" in corrections:
        # Correct mode TP6: pointer 20 is the solve cells' xland, which the warp driver and the phase kernels read.
        land=_base.tp6_land(p,cells)
        pa([land],shape=(cells,),device=device)
        f=pa([fields.get(n) for n in FA]+[land],shape=None,device=device,optional=True)
    m=pa([p["model"].get(n) for n in MA],shape=None,device=device,optional=True)
    mesh=pa([p["mesh"][n] for n in ("zz","zgrid")],shape=None,device=device)
    d=pa([p["diagnostics"][n] for n in _base.DIAG_ARGUMENTS],shape=None,device=device)
    t=pa([p["tendencies"].get(n) for n in TA],shape=(nz,cells),device=device,optional=True)
    s=pa([p["surface"].get(n) for n in SA],shape=(cells,),device=device,optional=True)
    pr=pa([p["precip"][n] for n in _base.PRECIP_ARGUMENTS],shape=(cells,),device=device)
    profiles=pa([p["profiles"][n] for n in _base.PROFILE_ARGUMENTS],shape=(nz,cells),device=device)
    c=p["constants"];extent=(np.int32(nz),np.int32(cells));af=(np.int32(flags[0]),np.int32(flags[1]))
    def run(name,count,args):
        """CUDA orchestration for mpas_atmphys_driver_microphysics.F:568-825."""
        module.get_function(name)(((count+31)//32,),(32,),args)
    counter=p["precip"]["i_rainnc"]
    if counter.dtype!=cp.int32 or counter.shape!=(cells,):raise ValueError("native rain bucket counter must be int32 per solve cell")
    precip_args=(f,s,pr,counter)+extent
    run("tempo_precip",cells,precip_args+(np.int32(0),np.int32(p["l_acrain"]),np.float32(bucket)))
    run("tempo_from",nz*cells,(m,mesh,d,t,f)+extent+af+(np.float32(c["R_v"]),np.float32(c["R_d"])))
    for name,value in p.get("surface_inputs",{}).items():cp.copyto(p["surface"][name],value)
    if flags[0]:
        run("tempo_emissions",cells,(fields["nwfa"],p["surface"]["nwfa2d"],np.float32(p["dt_microp"]),np.int32(cells)))
    presence=sum(1<<i for i,n in enumerate(FA[12:17]) if fields.get(n) is not None)
    pbl=p.get("pbl")
    if pbl is not None and (pbl.shape!=(2,nz,cells) or pbl.dtype!=cp.float32 or not pbl.flags.c_contiguous):
        raise ValueError("TEMPO PBL cloud carriers must be contiguous float32 (2, levels, cells)")
    if phase:
        _phase.launch_steps(device=device,nz=nz,cells=cells,steps=p["n_microp"],dt=p["dt_microp"],f=f,bl=pbl,s=s,
                            profiles=profiles,tables=tables,flags=flags,first=bool(p["first_call_main"]),
                            presence=presence,radii=bool(p.get("radii",True)),device_math_source=p["device_math_source"],
                            force_warp=(phase=="warp"),corrections=corrections)
    else:
        _launch_warp(module,resident,sizes,device,nz,cells,f,pbl,s,profiles,p,flags,presence,tables)
    p["first_call_main"]=False
    if "resources" in p:p["resources"]["first_call_main"]=False
    _base._launch_tail(module,p,f,s,profiles,pr,counter,precip_args,m,mesh,d,t,extent,af,c,nz,cells,device,
                       flags,diagnostics,hourly,dt_dyn,bucket,run)


def _launch_warp(module,resident,sizes,device,nz,cells,f,pbl,s,profiles,p,flags,presence,tables):
    """The warp-per-column kernel: all n_microp substeps in one launch (as tempo_v303._launch_warp)."""
    def run(name,count,args):
        module.get_function(name)(((count+31)//32,),(32,),args)
    blocks=-(-min(cells,_base.WARP_WAVES*resident)*32//_base.WARP_BLOCK)
    warps=blocks*(_base.WARP_BLOCK//32)
    states,work,blwork=_base._warp_work(device,nz,warps,sizes)
    cost,order,hist=_base._order_work(device,cells)
    hist.fill(0)
    run("tempo_order_hist",cells,(cost,np.int32(cells),hist))
    module.get_function("tempo_order_scan")((1,),(1,),(hist,))
    run("tempo_order_scatter",cells,(cost,np.int32(cells),hist,order))
    module.get_function("tempo_warp_steps")((blocks,),(_base.WARP_BLOCK,),(
        np.int32(nz),np.int32(cells),np.int32(p["n_microp"]),np.float32(p["dt_microp"]),f,
        pbl if pbl is not None else np.uint64(0),s,profiles,tables,
        *(np.int32(bool(v)) for v in flags),np.int32(bool(p["first_call_main"])),np.int32(presence),
        states,work,blwork,order,cost,np.int32(bool(p.get("radii",True)))))


# ---------------------------------------------------------------------------------------------------
# Correct mode: the fork corrections this column takes (woof.hex.mpasphys.rrfs.fork_corrections).
# v3.2.5 already carries TP1, TP2 and TP3 in its own text (main:2303, :3340, driver:404); TP4 and TP5 are
# later than the tag.  Exact mode and every caller that passes no id get _COLUMN itself.
# ---------------------------------------------------------------------------------------------------
#: The fork correction ids this column applies as a text pair.
CORRECTIONS = ("TP4", "TP5", "TP6")
#: TP4 (NCAR/TEMPO 71588d1, #62, check_over_depletion): the riming graupel volume terms follow their mass
#: terms through the cloud-water limiter, in upstream's order (after prg_gcw, before prh_hcw).
_TP4_OLD = """        T.prg_gcw[k] = T.prg_gcw[k] * ratio;
        T.prh_hcw[k] = T.prh_hcw[k] * ratio;
"""
_TP4_NEW = """        T.prg_gcw[k] = T.prg_gcw[k] * ratio;
        // Correct mode TP4 (TEMPO 71588d1): the volume of the riming terms scales with their mass.
        T.pbg_scw[k] = T.pbg_scw[k] * ratio;
        T.pbg_gcw[k] = T.pbg_gcw[k] * ratio;
        T.prh_hcw[k] = T.prh_hcw[k] * ratio;
"""
#: TP5 (NCAR/TEMPO ace0eae, #59): snow_moments' two branches the right way round.  With bm_s = 2 (TEMPO's value) the
#: mass moment smob IS the second moment; the temperature fit turns another bm_s moment into the second.  The pinned
#: text ran the fit for bm_s = 2 and took smob otherwise.  The rain-snow collection table was built with the same
#: flip and is NOT rebuilt here (fork_corrections TP5 says so).
_TP5_HEAD = ("""  if (bm_s > 2.0f - 1.e-3f && bm_s < 2.0f + 1.e-3f) {
    loga_ = SA(1) + SA(2) * tc + SA(3) * bm_s + SA(4) * tc * bm_s + SA(5) * tc * tc
""", """  if (bm_s > 2.0f - 1.e-3f && bm_s < 2.0f + 1.e-3f) {
    // Correct mode TP5 (TEMPO ace0eae): with bm_s = 2 the mass moment is the second moment itself.
    smo2_ = *smob;
  } else {
    loga_ = SA(1) + SA(2) * tc + SA(3) * bm_s + SA(4) * tc * bm_s + SA(5) * tc * tc
""")
_TP5_TAIL = ("""    smo2_ = tempo_pow((*smob / a_), (1.0f / b_));
  } else {
    smo2_ = *smob;
  }
""", """    smo2_ = tempo_pow((*smob / a_), (1.0f / b_));
  }
""")


def _corrections_key(corrections) -> tuple:
    """The ids of ``corrections`` this column applies, in a fixed order (the module cache key)."""
    return tuple(sorted(set(corrections or ()) & set(CORRECTIONS)))


def corrected_column(corrections=()) -> str:
    """``_COLUMN`` with this column's corrections in ``corrections`` applied; ``_COLUMN`` itself with none."""
    column = _COLUMN
    if "TP4" in _corrections_key(corrections):
        if column.count(_TP4_OLD) != 1:
            raise ValueError(f"TEMPO v3.2.5 correct mode: TP4 anchor found {column.count(_TP4_OLD)} times")
        column = column.replace(_TP4_OLD, _TP4_NEW)
    if "TP5" in _corrections_key(corrections):
        for old, new in (_TP5_HEAD, _TP5_TAIL):
            if column.count(old) != 1:
                raise ValueError(f"TEMPO v3.2.5 correct mode: TP5 anchor found {column.count(old)} times")
            column = column.replace(old, new)
    if "TP6" in _corrections_key(corrections):
        # TP6 (fork_corrections): the activation takes the cell's land mask (tempo_v303.tp6_warp says how it arrives).
        for old, new in (_TP6_CTX, _base._TP6_CALL):
            if column.count(old) != 1:
                raise ValueError(f"TEMPO v3.2.5 correct mode: TP6 anchor found {column.count(old)} times")
            column = column.replace(old, new)
    return column


#: TP6: the context's land word (0 = not given), after v3.2.5's last flag.
_TP6_CTX = ("  int hailhyperaware_flag;  // cfgs:18 (forced off with hail-aware off, driver:85)\n};\n",
            "  int hailhyperaware_flag;  // cfgs:18 (forced off with hail-aware off, driver:85)\n" + _base._TP6_CTX_V303[1].split("\n")[1] + "\n};\n")


def corrected_warp(corrections=()) -> str:
    """``_WARP`` with this column's corrections applied (TP6 sets the cell's land word); ``_WARP`` itself without."""
    return _base.tp6_warp(_WARP) if "TP6" in _corrections_key(corrections) else _WARP
