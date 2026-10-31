"""Pinned RRFS TEMPO v3.0.3 mixed-precision CUDA column backend.

Source authority: RRFS fork 1d6e370e7b43a332074809f170532a216c2212fb,
TEMPO submodule 9adf9ef2827104640125f0c322e1218fce74e94c.
WP = MPAS RKIND (binary32), DP and table_dp = binary64. Table values must
come from pristine strict native initialization and Rust-owned decoding.
The shared device mathematics header is supplied by the physics driver.
"""
# Derived from NCAR/TEMPO 9adf9ef, Copyright 2026 NOAA, UCAR/NCAR, CU/CIRES.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Provided AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
from functools import lru_cache
import numpy as np
SCHEME = "mp_tempo"
BACKEND = "hex-cuda"
# Both fork rows pin the identical TEMPO 9adf9ef source and table layout.
# Each forecast still supplies tables from its own authenticated initialization.
TEMPO_FORK_PINS = (
    "1d6e370e7b43a332074809f170532a216c2212fb",
    "44c084944e17f7098668bef6d1e353d12e3c2d1a",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
STATE_FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "qb")

TABLE_LAYOUT = [('av_r', (1, 1, 1, 1, 1)), ('bv_r', (1, 1, 1, 1, 1)), ('av_s', (1, 1, 1, 1, 1)), ('bv_s', (1, 1, 1, 1, 1)), ('fv_s', (1, 1, 1, 1, 1)), ('bv_c', (1, 1, 1, 1, 1)), ('bv_i', (1, 1, 1, 1, 1)), ('av_g_old', (1, 1, 1, 1, 1)), ('bv_g_old', (1, 1, 1, 1, 1)), ('av_g_new', (1, 1, 1, 1, 1)), ('bv_g_new', (1, 1, 1, 1, 1)), ('fv_r', (1, 1, 1, 1, 1)), ('av_c', (1, 1, 1, 1, 1)), ('a_coeff', (1, 1, 1, 1, 1)), ('b_coeff', (1, 1, 1, 1, 1)), ('av_i', (1, 1, 1, 1, 1)), ('am_s', (1, 1, 1, 1, 1)), ('bm_s', (1, 1, 1, 1, 1)), ('bm_g', (1, 1, 1, 1, 1)), ('bm_i', (1, 1, 1, 1, 1)), ('bm_r', (1, 1, 1, 1, 1)), ('rho_i', (1, 1, 1, 1, 1)), ('xm0i', (1, 1, 1, 1, 1)), ('d0c', (1, 1, 1, 1, 1)), ('d0r', (1, 1, 1, 1, 1)), ('d0s', (1, 1, 1, 1, 1)), ('d0g', (1, 1, 1, 1, 1)), ('d0r_max', (1, 1, 1, 1, 1)), ('c_cube', (1, 1, 1, 1, 1)), ('c_sqrd', (1, 1, 1, 1, 1)), ('mu_r', (1, 1, 1, 1, 1)), ('mu_s', (1, 1, 1, 1, 1)), ('mu_g', (1, 1, 1, 1, 1)), ('mu_i', (1, 1, 1, 1, 1)), ('nu_c_scale', (1, 1, 1, 1, 1)), ('nu_c_max', (1, 1, 1, 1, 1)), ('nu_c_min', (1, 1, 1, 1, 1)), ('naccn0', (1, 1, 1, 1, 1)), ('naccn1', (1, 1, 1, 1, 1)), ('nain0', (1, 1, 1, 1, 1)), ('nain1', (1, 1, 1, 1, 1)), ('nwfa_default', (1, 1, 1, 1, 1)), ('nifa_default', (1, 1, 1, 1, 1)), ('aero_max', (1, 1, 1, 1, 1)), ('hgfrz', (1, 1, 1, 1, 1)), ('nt_c_o', (1, 1, 1, 1, 1)), ('nt_c_l', (1, 1, 1, 1, 1)), ('nt_c_max', (1, 1, 1, 1, 1)), ('nt_c_min', (1, 1, 1, 1, 1)), ('tno', (1, 1, 1, 1, 1)), ('ato', (1, 1, 1, 1, 1)), ('rho_s', (1, 1, 1, 1, 1)), ('demott_nuc_ssati', (1, 1, 1, 1, 1)), ('max_ni', (1, 1, 1, 1, 1)), ('icenuc_max', (1, 1, 1, 1, 1)), ('rime_threshold', (1, 1, 1, 1, 1)), ('rime_conversion', (1, 1, 1, 1, 1)), ('ef_si', (1, 1, 1, 1, 1)), ('ef_rs', (1, 1, 1, 1, 1)), ('ef_rg', (1, 1, 1, 1, 1)), ('ef_ri', (1, 1, 1, 1, 1)), ('autocon_nr_factor', (1, 1, 1, 1, 1)), ('timestep_conversion_rime_to_rain', (1, 1, 1, 1, 1)), ('nrhg', (1, 1, 1, 1, 1)), ('nrhg1', (1, 1, 1, 1, 1)), ('min_qv', (1, 1, 1, 1, 1)), ('r1', (1, 1, 1, 1, 1)), ('r2', (1, 1, 1, 1, 1)), ('low_limit_mass_for_precip', (1, 1, 1, 1, 1)), ('eps', (1, 1, 1, 1, 1)), ('meters3_to_liters', (1, 1, 1, 1, 1)), ('gonv_min', (1, 1, 1, 1, 1)), ('gonv_max', (1, 1, 1, 1, 1)), ('t0', (1, 1, 1, 1, 1)), ('rho_w', (1, 1, 1, 1, 1)), ('sc', (1, 1, 1, 1, 1)), ('pi', (1, 1, 1, 1, 1)), ('lsub', (1, 1, 1, 1, 1)), ('lvap0', (1, 1, 1, 1, 1)), ('rv', (1, 1, 1, 1, 1)), ('rdry', (1, 1, 1, 1, 1)), ('roverrv', (1, 1, 1, 1, 1)), ('r', (1, 1, 1, 1, 1)), ('rho_not', (1, 1, 1, 1, 1)), ('rho_not0', (1, 1, 1, 1, 1)), ('cp', (1, 1, 1, 1, 1)), ('r_uni', (1, 1, 1, 1, 1)), ('kap0', (1, 1, 1, 1, 1)), ('kap1', (1, 1, 1, 1, 1)), ('lam0', (1, 1, 1, 1, 1)), ('lam1', (1, 1, 1, 1, 1)), ('nbins', (1, 1, 1, 1, 1)), ('nbc', (1, 1, 1, 1, 1)), ('nbr', (1, 1, 1, 1, 1)), ('nbs', (1, 1, 1, 1, 1)), ('nbi', (1, 1, 1, 1, 1)), ('nbg', (1, 1, 1, 1, 1)), ('ntb_i', (1, 1, 1, 1, 1)), ('ntb_i1', (1, 1, 1, 1, 1)), ('ntb_c', (1, 1, 1, 1, 1)), ('ntb_t', (1, 1, 1, 1, 1)), ('ntb_g1', (1, 1, 1, 1, 1)), ('ntb_s', (1, 1, 1, 1, 1)), ('ntb_g', (1, 1, 1, 1, 1)), ('ntb_r', (1, 1, 1, 1, 1)), ('ntb_r1', (1, 1, 1, 1, 1)), ('ntb_t1', (1, 1, 1, 1, 1)), ('ntb_in', (1, 1, 1, 1, 1)), ('ntb_arc', (1, 1, 1, 1, 1)), ('ntb_arw', (1, 1, 1, 1, 1)), ('ntb_art', (1, 1, 1, 1, 1)), ('ntb_arr', (1, 1, 1, 1, 1)), ('ntb_ark', (1, 1, 1, 1, 1)), ('idx_bg1', (1, 1, 1, 1, 1)), ('nhbins', (1, 1, 1, 1, 1)), ('radar_bins', (1, 1, 1, 1, 1)), ('rho_g', (9, 1, 1, 1, 1)), ('r_c', (37, 1, 1, 1, 1)), ('r_i', (64, 1, 1, 1, 1)), ('r_r', (37, 1, 1, 1, 1)), ('r_s', (37, 1, 1, 1, 1)), ('r_g', (37, 1, 1, 1, 1)), ('n0r_exp', (37, 1, 1, 1, 1)), ('n0g_exp', (37, 1, 1, 1, 1)), ('nt_i', (55, 1, 1, 1, 1)), ('nt_in', (55, 1, 1, 1, 1)), ('ta_na', (7, 1, 1, 1, 1)), ('ta_ww', (9, 1, 1, 1, 1)), ('ta_tk', (7, 1, 1, 1, 1)), ('ta_ra', (5, 1, 1, 1, 1)), ('ta_ka', (4, 1, 1, 1, 1)), ('sa', (10, 1, 1, 1, 1)), ('sb', (10, 1, 1, 1, 1)), ('tc', (9, 1, 1, 1, 1)), ('dim_nrhg', (1, 1, 1, 1, 1)), ('av_g', (9, 1, 1, 1, 1)), ('bv_g', (9, 1, 1, 1, 1)), ('am_i', (1, 1, 1, 1, 1)), ('am_r', (1, 1, 1, 1, 1)), ('am_g', (9, 1, 1, 1, 1)), ('lfus', (1, 1, 1, 1, 1)), ('olfus', (1, 1, 1, 1, 1)), ('orv', (1, 1, 1, 1, 1)), ('ar_volume', (1, 1, 1, 1, 1)), ('sc3', (1, 1, 1, 1, 1)), ('d0i', (1, 1, 1, 1, 1)), ('xm0s', (1, 1, 1, 1, 1)), ('xm0g', (1, 1, 1, 1, 1)), ('obmi', (1, 1, 1, 1, 1)), ('obmr', (1, 1, 1, 1, 1)), ('oams', (1, 1, 1, 1, 1)), ('obms', (1, 1, 1, 1, 1)), ('ocms', (1, 1, 1, 1, 1)), ('oamg', (9, 1, 1, 1, 1)), ('ocmg', (9, 1, 1, 1, 1)), ('obmg', (1, 1, 1, 1, 1)), ('cce', (5, 15, 1, 1, 1)), ('ccg', (5, 15, 1, 1, 1)), ('ocg1', (15, 1, 1, 1, 1)), ('ocg2', (15, 1, 1, 1, 1)), ('cie', (7, 1, 1, 1, 1)), ('cig', (7, 1, 1, 1, 1)), ('oig1', (1, 1, 1, 1, 1)), ('oig2', (1, 1, 1, 1, 1)), ('cre', (13, 1, 1, 1, 1)), ('crg', (13, 1, 1, 1, 1)), ('ore1', (1, 1, 1, 1, 1)), ('org1', (1, 1, 1, 1, 1)), ('org2', (1, 1, 1, 1, 1)), ('org3', (1, 1, 1, 1, 1)), ('cse', (17, 1, 1, 1, 1)), ('csg', (17, 1, 1, 1, 1)), ('cge', (12, 9, 1, 1, 1)), ('cgg', (12, 9, 1, 1, 1)), ('oge1', (1, 1, 1, 1, 1)), ('ogg1', (1, 1, 1, 1, 1)), ('ogg2', (1, 1, 1, 1, 1)), ('ogg3', (1, 1, 1, 1, 1)), ('t1_qr_qc', (1, 1, 1, 1, 1)), ('t1_qr_qi', (1, 1, 1, 1, 1)), ('t2_qr_qi', (1, 1, 1, 1, 1)), ('t1_qs_qc', (1, 1, 1, 1, 1)), ('t1_qs_qi', (1, 1, 1, 1, 1)), ('t1_qr_ev', (1, 1, 1, 1, 1)), ('t2_qr_ev', (1, 1, 1, 1, 1)), ('t1_qs_sd', (1, 1, 1, 1, 1)), ('t2_qs_sd', (1, 1, 1, 1, 1)), ('t1_qs_me', (1, 1, 1, 1, 1)), ('t2_qs_me', (1, 1, 1, 1, 1)), ('t1_qg_sd', (1, 1, 1, 1, 1)), ('t1_qg_me', (1, 1, 1, 1, 1)), ('nic2', (1, 1, 1, 1, 1)), ('nii2', (1, 1, 1, 1, 1)), ('nii3', (1, 1, 1, 1, 1)), ('nir2', (1, 1, 1, 1, 1)), ('nir3', (1, 1, 1, 1, 1)), ('nis2', (1, 1, 1, 1, 1)), ('nig2', (1, 1, 1, 1, 1)), ('nig3', (1, 1, 1, 1, 1)), ('niin2', (1, 1, 1, 1, 1)), ('nic1', (1, 1, 1, 1, 1)), ('dc', (100, 1, 1, 1, 1)), ('dtc', (100, 1, 1, 1, 1)), ('di', (100, 1, 1, 1, 1)), ('dti', (100, 1, 1, 1, 1)), ('dr', (100, 1, 1, 1, 1)), ('dtr', (100, 1, 1, 1, 1)), ('ds', (100, 1, 1, 1, 1)), ('dts', (100, 1, 1, 1, 1)), ('dg', (100, 1, 1, 1, 1)), ('dtg', (100, 1, 1, 1, 1)), ('t_nc', (100, 1, 1, 1, 1)), ('hbins', (50, 1, 1, 1, 1)), ('dhbins', (50, 1, 1, 1, 1)), ('sbins_radar', (50, 1, 1, 1, 1)), ('dsbins_radar', (50, 1, 1, 1, 1)), ('gbins_radar', (50, 1, 1, 1, 1)), ('dgbins_radar', (50, 1, 1, 1, 1)), ('t_efrw', (100, 100, 1, 1, 1)), ('t_efsw', (100, 100, 1, 1, 1)), ('tpc_wev', (100, 37, 100, 1, 1)), ('tnc_wev', (100, 37, 100, 1, 1)), ('tnccn_act', (7, 9, 7, 5, 4)), ('tps_iaus', (64, 55, 1, 1, 1)), ('tni_iaus', (64, 55, 1, 1, 1)), ('tpi_ide', (64, 55, 1, 1, 1)), ('tpi_qcfz', (37, 100, 45, 55, 1)), ('tni_qcfz', (37, 100, 45, 55, 1)), ('tpi_qrfz', (37, 37, 45, 55, 1)), ('tpg_qrfz', (37, 37, 45, 55, 1)), ('tni_qrfz', (37, 37, 45, 55, 1)), ('tnr_qrfz', (37, 37, 45, 55, 1)), ('tcs_racs1', (37, 9, 37, 37, 1)), ('tmr_racs1', (37, 9, 37, 37, 1)), ('tcs_racs2', (37, 9, 37, 37, 1)), ('tmr_racs2', (37, 9, 37, 37, 1)), ('tcr_sacr1', (37, 9, 37, 37, 1)), ('tms_sacr1', (37, 9, 37, 37, 1)), ('tcr_sacr2', (37, 9, 37, 37, 1)), ('tms_sacr2', (37, 9, 37, 37, 1)), ('tnr_racs1', (37, 9, 37, 37, 1)), ('tnr_racs2', (37, 9, 37, 37, 1)), ('tnr_sacr1', (37, 9, 37, 37, 1)), ('tnr_sacr2', (37, 9, 37, 37, 1)), ('tcg_racg', (37, 37, 9, 37, 37)), ('tmr_racg', (37, 37, 9, 37, 37)), ('tcr_gacr', (37, 37, 9, 37, 37)), ('tnr_racg', (37, 37, 9, 37, 37)), ('tnr_gacr', (37, 37, 9, 37, 37))]
SOURCE_CITATIONS = {'tempo_build_tables': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:21-77', 'build_table_freezewater': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:78-91', 'write_table_freezewater': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:92-120', 'build_table_qr_acr_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:121-259', 'write_table_qr_acr_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:260-295', 'build_table_qr_acr_qg': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:296-393', 'write_table_qr_acr_qg': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:394-421', 'gather': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:424-438', 'get_index_for_rank': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:440-467', 'qr_acr_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:468-598', 'qr_acr_qg': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:599-691', 'freezewater': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_tables.F90:692-782', 'effective_radius': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:12-56', 'reflectivity_10cm': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:57-121', 'find_melting_level': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:122-141', 'complex_water_ray': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:142-172', 'complex_ice_maetzler': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:173-201', 'reflectivity_from_melting_graupel': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:202-252', 'reflectivity_from_melting_snow': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:253-306', 'rayleigh_soak_wetgraupel': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:307-395', 'max_hail_diam': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:396-438', 'freezing_rain': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_diags.F90:439-456', 'init_water_friendly_aerosols': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:13-42', 'init_ice_friendly_aerosols': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:43-73', 'aerosol_collection_efficiency': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_aerosols.F90:74-118', 'tempo_main': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:73-825', 'aerosol_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:826-848', 'cloud_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:849-941', 'rain_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:942-1009', 'ice_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1010-1082', 'snow_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1083-1117', 'graupel_check_and_update': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1118-1213', 'graupel_init': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1214-1242', 'thermo_vars': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1243-1292', 'check_over_depletion': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1293-1419', 'sum_tendencies': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1420-1494', 'sedimentation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1495-1542', 'semilagrangian_sedimentation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1543-1731', 'rain_fallspeed': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1732-1771', 'graupel_fallspeed': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1772-1828', 'snow_fallspeed': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1829-1885', 'ice_fallspeed': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1886-1920', 'cloud_fallspeed': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1921-1955', 'cloud_condensation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1956-2026', 'rain_evaporation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2027-2090', 'freeze_cloud_melt_ice': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2091-2133', 'koop_nucleation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2134-2163', 'activate_cloud_number': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2164-2244', 'warm_rain': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2245-2322', 'riming': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2323-2456', 'get_snow_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2457-2474', 'get_temperature_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2475-2489', 'get_rain_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2490-2521', 'get_graupel_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2522-2554', 'get_cloud_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2555-2575', 'get_ice_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2576-2601', 'rain_snow_rain_graupel': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2602-2692', 'ice_nucleation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2693-2783', 'demott_nucleation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2784-2799', 'get_in_table_index': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2800-2821', 'get_t1_subl': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2822-2851', 'ice_processes': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2852-2980', 'melting': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2981-3080', 'aerosol_scavenging': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:3081-3153', 'get_constant_cloud_number': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:11-24', 'calc_gamma_p': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:25-44', 'calc_gamma_series': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:45-78', 'calc_gamma_cf': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:79-118', 'snow_moments': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:119-255', 'calc_rslf': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:256-282', 'calc_rsif': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:283-304', 'get_nuc': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:305-319', 'compute_efrw': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:320-373', 'compute_efsw': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:374-413', 'qi_aut_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:414-463', 'compute_drop_evap': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_utils.F90:464-500', 'get_version': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:382-407', 'initialize_graupel_vars': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:408-425', 'initialize_parameters': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:426-608', 'initialize_bins_for_tables': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:609-656', 'initialize_bins_for_hail_size': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:657-669', 'initialize_bins_for_radar': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:670-690', 'create_bins': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:691-720', 'initialize_arrays_freezewater': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:721-746', 'initialize_arrays_qr_acr_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:747-774', 'initialize_arrays_qr_acr_qg': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:775-796', 'initialize_arrays_ccn': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:797-814', 'initialize_arrays_drop_evap': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:815-825', 'initialize_array_efsw': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:826-835', 'initialize_array_efrw': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:836-845', 'initialize_arrays_qi_aut_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:846-857', 'save_or_read_ml_data': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:84-101', 'tempo_ml_predict_cloud_number': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:102-195', 'standard_scaler_transform': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:196-211', 'relu_activation': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:212-228', 'tempo_init': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182', 'tempo_run': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438', 'tempo_aerosol_surface_emissions': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:439-454', 'read_table_freezewater': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:455-478', 'read_table_qr_acr_qs': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:479-509', 'read_table_qr_acr_qg': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:510-532', 'read_table_ccn': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:533-554', 'check_before_table_read': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:555-603', 'init_ml_data': 'MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:604-633'}
_COLUMN = r'''
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
using TK_max_ni = double;
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
  int first_call_main;  // Fortran SAVE variable first_call_main, host-managed
  int ml_for_nc_flag;
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
  TT(pna_rca) TT(pna_sca) TT(pna_gca) TT(pnd_rcd) TT(pnd_scd) TT(pnd_gcd)

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
  nwfa[1] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[2] - hgt[1]) / 1000.0f) * niccn3);
  for (int k = 2; k <= nz; ++k) {
    nwfa[k] = PS(naccn1) + PS(naccn0) * tempo_exp(-((hgt[k] - hgt[1]) / 1000.0f) * niccn3);
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
  nifa[1] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[2] - hgt[1]) / 1000.0f) * niin3);
  for (int k = 2; k <= nz; ++k) {
    nifa[k] = PS(nain1) + PS(nain0) * tempo_exp(-((hgt[k] - hgt[1]) / 1000.0f) * niin3);
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
  st2 = (1.2f + 1.0f / 12.0f * aval) / (1.0f + aval);
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
  int v = (int)((tempc - 2.5f) / 5.0f) - 1;
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
TEMPO_HDN inline float koop_nucleation(const TempoParams& P, const TempoCtx& C, float temp,
                                        float satw, float naero) {
  float xni; float  mu_diff; float  a_w_i; float  delta_aw; float  log_j_rate; float  j_rate; float  prob_h;
  xni = 0.0f;
  mu_diff = 210368.0f + (131.438f * temp) - (3.32373e6f / temp) - (41729.1f * tempo_log(temp));
  a_w_i = tempo_exp(mu_diff / (PS(r_uni) * temp));
  delta_aw = satw - a_w_i;
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
    const float* nwfa, const float* qv, const float* qvs, const bool* l_qc,
    const float* rc, const float* nc, TempoTend& T) {
  float clap; float  fcd; float  dfcd; float  xrc; float  xnc; float  orho; float  tempc; float  otemp; float  rvs; float  rvs_p; float  rvs_pp; float  gamsc; float  alphsc; float 
      xsat; float  t1_evap;
  double dc_star;
  int idx_d, idx_n, idx_c;
  const float orv = PS(orv); const float  pi = PS(pi);
#if TEMPO_WARP
  // warp: the serial RETURN at the first level with |ssatw| < eps leaves every
  // level from there up untouched; find that level (own levels ascend), take
  // the warp minimum, and process only the levels below it.
  int kstop = nz + 1;
  TEMPO_LEVELS(k, nz) if (fabs(ssatw[k]) < PS(eps)) { kstop = k; break; }
  kstop = tempo_warp_min(kstop);
#endif
  TEMPO_LEVELS(k, nz) {
    // Fortran quirk: RETURN (not CYCLE) leaves the remaining levels untouched.
#if TEMPO_WARP
    if (k >= kstop) break;
#else
    if (fabs(ssatw[k]) < PS(eps)) return;
#endif

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
        xnc = tmax(PS(nt_c_min), activate_cloud_number(P, temp[k], w1d[k], nwfa[k], nullptr));
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
            eva_factor = tmin(1.0f, 0.01f + (0.99f - 0.01f) * (tempc / 20.0f));
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
                   (1.0f / 6.0f));
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
    const float* rg, const float* ng, const double* ilamg, const int* idx, TempoTend& T) {
  double xds; double  xdg; double  n0_g; double  lamc;
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
            vtboost[k] = tmin(1.5f, 1.1f + (r_frac - 2.f) * .016f);
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
          if (temp[k] > t0) ef_gw = ef_gw * 0.1f;
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
    const float* rg, const float* ng, const double* ilamg, const int* idx, TempoTend& T) {
  int idx_r, idx_r1, idx_s, idx_t, idx_g, idx_g1;
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
          T.png_rcs[k] = T.pnr_rcs[k];
          T.pbg_rcs[k] = mtl * T.prg_rcs[k] / PS(rho_i);
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
          T.pbg_rcg[k] = mtl * T.prg_rcg[k] / PS(rho_i);
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
  }
}

// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2695-2783.  nifa and nwfa are optional (nullptr = absent).
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:2693-2783 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void ice_nucleation(const TempoParams& P, const TempoCtx& C, int nz,
    const float* temp, const float* rho, const float* w1d, const float* qv,
    const float* qvsi, const float* ssati, const float* ssatw, const float* nifa,
    const float* nwfa, const float* ni, const double* smo0, const float* rc,
    const float* nc, const float* rr, const float* nr, const double* ilamr, TempoTend& T) {
  float rate_max; float  tempc; float  xni; float  xnc;
  int idx_in, idx_r, idx_r1, idx_tc, idx_c, idx_n;
  const float t0 = PS(t0); const float  xm0i = PS(xm0i);
  TEMPO_LEVELS(k, nz) {
    if (temp[k] < t0) {
      tempc = temp[k] - t0;
      idx_tc = imax(1, imin(tnint(-tempc), 45));
      rate_max = (qv[k] - qvsi[k]) * rho[k] * C.global_inverse_dt * 0.999f;
      if (nifa != nullptr) {
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
        T.png_rfz[k] = T.pnr_rfz[k] * tmax(tmin((tempo_pow(10.0f, (-0.1f * w1d[k])) + 0.1f), 1.0f), 0.1f);
      } else if (rr[k] > PS(r1) && temp[k] < PS(hgfrz)) {
        T.pri_rfz[k] = rr[k] * C.global_inverse_dt;
        T.pni_rfz[k] = nr[k] * C.global_inverse_dt;
      }
      T.pbg_rfz[k] = PS(meters3_to_liters) * T.prg_rfz[k] / PS(rho_i);

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

      if ((ssati[k] >= PS(demott_nuc_ssati)) || (ssatw[k] > PS(eps) && tempc < -20.0f)) {
        if (nifa != nullptr) {
          xnc = demott_nucleation(P, tempc, rho[k], nifa[k]);
        } else {
          xnc = tmin(PS(icenuc_max), PS(tno) * tempo_exp(PS(ato) * (t0 - temp[k])));
        }
        xni = ni[k] + (T.pni_rfz[k] + T.pni_wfz[k]) * C.global_dt;
        T.pni_inu[k] = 0.5f * (xnc - xni + fabs(xnc - xni)) * C.global_inverse_dt;
        T.pri_inu[k] = tmin(rate_max, xm0i * T.pni_inu[k]);
        T.pni_inu[k] = T.pri_inu[k] / xm0i;
      }

      xni = smo0[k] + ni[k] + (T.pni_rfz[k] + T.pni_wfz[k] + T.pni_inu[k]) * C.global_dt;
      if (nwfa != nullptr) {
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
    const int* idx, TempoTend& T) {
  float xdi; float  xmi; float  oxmi; float  c_snow; float  rate_max; float  otemp; float  rvs; float  t2_qg_sd;
  double lami; double  lamr; double  n0_r; double  n0_g;
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
        lami = 1.0f / ilami[k];
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

        // Fortran quirk: lami is recomputed but xdi still uses ilami(k).
        lami = tempo_pow((am_i * PA1(cig, 2) * PS(oig1) * ni[k] / ri[k]), PS(obmi));
        (void)lami;
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
          T.png_rci[k] = T.pnr_rci[k] * tmax(tmin((tempo_pow(10.0f, (-0.1f * w1d[k])) + 0.1f), 1.0f), 0.1f);
          T.pni_rci[k] = T.pri_rci[k] * oxmi;
          T.prr_rci[k] = rhof[k] * PS(t2_qr_qi) * PS(ef_ri) * ni[k] * n0_r
                       * (tempo_pow_with(bf, (-PA1(cre, 8))));
          T.prr_rci[k] = tmin(rr[k] * C.global_inverse_dt, T.prr_rci[k]);
          T.prg_rci[k] = T.pri_rci[k] + T.prr_rci[k];
          // Fortran quirk: no meters3_to_liters factor here (unlike other pbg terms).
          T.pbg_rci[k] = T.prg_rci[k] / PS(rho_i);
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
    const float* ng, const double* ilamg, const int* idx, TempoTend& T) {
  float tempc; float  otemp; float  rvs; float  melt_f; float  t2_qg_me; float  t2_qg_sd;
  double n0_g; double  n0_melt; double  lamg;
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
            T.prs_sde[k] = c_cube * t1_subl[k] * diffu[k] * ssati[k] * rvs
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
        // Fortran quirk: 0.2 here vs 0.28 in the sublimation term.
        t2_qg_me = pi * 4.0f * c_cube * olfus * 0.2f * PS(sc3) * sqrt(PA1(av_g, idx[k]))
                 * PA2(cgg, 11, idx[k]);
        T.prr_gml[k] = (tempc * tcond[k] - lvap0 * diffu[k] * delqvs[k]) * n0_melt
                     * (PS(t1_qg_me) * p10
                        + t2_qg_me * rhof2[k] * vsc2[k] * p11);
        T.prr_gml[k] = tmin(rg[k] * C.global_inverse_dt, tmax(0.0f, T.prr_gml[k]));
        if (T.prr_gml[k] > 0.0f) {
          melt_f = tmax(0.05f, tmin(T.prr_gml[k] * C.global_dt / rg[k], 1.0f));
          T.pbg_gml[k] = mtl * T.prr_gml[k]
                       / tmax(tmin(melt_f * PA1(rho_g, idx[k]), PS(rho_w)), 50.0f);
          T.pnr_gml[k] = T.prr_gml[k] * ng[k] / rg[k] * tempo_pow(10.0f, (-0.33f * (temp[k] - t0)));
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

  sr = tmax(0.01, tmin(1.0 - rg / (rg + rr), 0.99));
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
    const float* rg, const float* ng, const int* idx, const double* ilamg, float* dbz) {
  float ze_rain; float  ze_snow; float  ze_graupel;
  double n0_r; double  lamr; double  n0_g;
  int k_melt = 1;
  const float pi = PS(pi); const float  am_s = PS(am_s);
  (void)C;

  if (refl10cm_from_melting_flag) {
    k_melt = find_melting_level(P, nz, temp, l_qr, l_qs, l_qg);
  }

  TEMPO_LEVELS_DOWN(k, nz) {
    ze_rain = 1.e-22f;
    ze_snow = 1.e-22f;
    ze_graupel = 1.e-22f;
    dbz[k] = -35.0f;

    if (l_qr[k]) {
      lamr = 1.0f / ilamr[k];
      n0_r = nr[k] * PS(org2) * tempo_pow(lamr, PA1(cre, 2));
      ze_rain = n0_r * PA1(crg, 4) * tempo_pow(ilamr[k], PA1(cre, 4));
    }
    if (l_qs[k]) {
      ze_snow = (0.176f / 0.93f) * (6.0f / pi) * (6.0f / pi) * (am_s / 900.0f) * (am_s / 900.0f) * smoz[k];
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_snow = reflectivity_from_melting_snow(P, rs[k], smob[k], smoc[k], rr[k]);
        }
      }
    }
    if (l_qg[k]) {
      n0_g = ng[k] * PS(ogg2) * tempo_pow((1.0f / ilamg[k]), PA2(cge, 2, 1));
      ze_graupel = (0.176f / 0.93f) * (6.0f / pi) * (6.0f / pi) * (PA1(am_g, idx[k]) / 900.0f)
                 * (PA1(am_g, idx[k]) / 900.0f) * n0_g * PA2(cgg, 4, 1) * tempo_pow(ilamg[k], PA2(cge, 4, 1));
      if (refl10cm_from_melting_flag) {
        if (k_melt > 2 && k < k_melt - 1) {
          ze_graupel = reflectivity_from_melting_graupel(P, rg[k], ng[k], ilamg[k], idx[k], rr[k]);
        }
      }
    }
    dbz[k] = tmax(-35.0f, 10.0f * tempo_log10((ze_rain + ze_snow + ze_graupel) * 1.e18));
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
  WA(sl_net_flx) WA(sl_precip_flx) WA(sl_rr_save) WA(sed_r)
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
#define TEMPO_WA_MEMBER(name)
  TEMPO_WORK_ARRAYS(TEMPO_WA_MEMBER)
#undef TEMPO_WA_MEMBER
  bool l_qc[TEMPO_NZP], l_qi[TEMPO_NZP], l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP];
  int idx_bg[TEMPO_NZP];
  TempoTend T;
  // ty_tempo_main_diags scalars
  float rain_precip, cloud_precip, ice_liquid_equiv_precip, snow_liquid_equiv_precip,
      graupel_liquid_equiv_precip, frozen_fraction, frz_rain_precip;
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
    TempoTend& T) {
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

    sump = -T.prr_wau[k] - T.pri_wfz[k] - T.prr_rcw[k] - T.prs_scw[k] - T.prg_scw[k] - T.prg_gcw[k];
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

    sump = -T.prg_rfz[k] - T.pri_rfz[k] - T.prr_rci[k] + T.prr_rcs[k] + T.prr_rcg[k];
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

    T.pri_ihm[k] = T.prs_ihm[k] + T.prg_ihm[k];
    ratio = tmin(fabs(T.prr_rcg[k]), fabs(T.prg_rcg[k]));
    T.prr_rcg[k] = ratio * tsign(1.0f, T.prr_rcg[k]);
    T.prg_rcg[k] = -T.prr_rcg[k];
    T.pbg_rcg[k] = PS(meters3_to_liters) * T.prg_rcg[k] / PS(rho_i);
    if (temp[k] > PS(t0)) {
      ratio = tmin(fabs(T.prr_rcs[k]), fabs(T.prs_rcs[k]));
      T.prr_rcs[k] = ratio * tsign(1.0f, T.prr_rcs[k]);
      T.prs_rcs[k] = -T.prr_rcs[k];
    }
  }
}

// 1422-1494
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:1420-1494 @ 1d6e370e; TEMPO 9adf9ef.
TEMPO_HDN inline void sum_tendencies(const TempoParams& P, int nz, const float* rho,
    const float* temp, const int* idx, const float* lvap, const float* ocp,
    const TempoTend& T, float* tten, float* qvten, float* qcten, float* ncten,
    float* qiten, float* niten, float* qsten, float* qrten, float* nrten, float* qgten,
    float* ngten, float* qbten) {
  TEMPO_LEVELS(k, nz) {
    float orho = 1.f / rho[k];
    float lfus2 = PS(lsub) - lvap[k];
    qvten[k] = qvten[k] + (-T.pri_inu[k] - T.pri_iha[k] - T.pri_ide[k] - T.prs_ide[k]
        - T.prs_sde[k] - T.prg_gde[k]) * orho;
    qcten[k] = qcten[k] + (-T.prr_wau[k] - T.pri_wfz[k] - T.prr_rcw[k] - T.prs_scw[k]
        - T.prg_scw[k] - T.prg_gcw[k]) * orho;
    ncten[k] = ncten[k] + (-T.pnc_wau[k] - T.pnc_rcw[k] - T.pni_wfz[k] - T.pnc_scw[k]
        - T.pnc_gcw[k]) * orho;
    qiten[k] = qiten[k] + (T.pri_inu[k] + T.pri_iha[k] + T.pri_ihm[k] + T.pri_wfz[k]
        + T.pri_rfz[k] + T.pri_ide[k] - T.prs_iau[k] - T.prs_sci[k] - T.pri_rci[k]) * orho;
    niten[k] = niten[k] + (T.pni_inu[k] + T.pni_iha[k] + T.pni_ihm[k] + T.pni_wfz[k]
        + T.pni_rfz[k] + T.pni_ide[k] - T.pni_iau[k] - T.pni_sci[k] - T.pni_rci[k]) * orho;
    qrten[k] = qrten[k] + (T.prr_wau[k] + T.prr_rcw[k] + T.prr_sml[k] + T.prr_gml[k]
        + T.prr_rcs[k] + T.prr_rcg[k] - T.prg_rfz[k] - T.pri_rfz[k] - T.prr_rci[k]) * orho;
    nrten[k] = nrten[k] + (T.pnr_wau[k] + T.pnr_sml[k] + T.pnr_gml[k]
        - (T.pnr_rfz[k] + T.pnr_rcr[k] + T.pnr_rcg[k] + T.pnr_rcs[k] + T.pnr_rci[k]
        + T.pni_rfz[k])) * orho;
    qsten[k] = qsten[k] + (T.prs_iau[k] + T.prs_sde[k] + T.prs_sci[k] + T.prs_scw[k]
        + T.prs_rcs[k] + T.prs_ide[k] - T.prs_ihm[k] - T.prr_sml[k]) * orho;
    qgten[k] = qgten[k] + (T.prg_scw[k] + T.prg_rfz[k] + T.prg_gde[k] + T.prg_rcg[k]
        + T.prg_gcw[k] + T.prg_rci[k] + T.prg_rcs[k] - T.prg_ihm[k] - T.prr_gml[k]) * orho;
    ngten[k] = ngten[k] + (T.png_scw[k] + T.png_rfz[k] - T.png_rcg[k] + T.png_rci[k]
        + T.png_rcs[k] + T.png_gde[k] - T.pnr_gml[k]) * orho;
    qbten[k] = qbten[k] + (T.pbg_scw[k] + T.pbg_rfz[k] + T.pbg_gcw[k] + T.pbg_rci[k]
        + T.pbg_rcs[k] + T.pbg_rcg[k] + T.pbg_sml[k] - T.pbg_gml[k]
        + PS(meters3_to_liters) * (T.prg_gde[k] - T.prg_ihm[k]) / PA1(rho_g, idx[k])) * orho;
    if (temp[k] < PS(t0)) {
      tten[k] = tten[k]
          + (PS(lsub) * ocp[k] * (T.pri_inu[k] + T.pri_ide[k] + T.prs_ide[k] + T.prs_sde[k]
             + T.prg_gde[k] + T.pri_iha[k])
             + lfus2 * ocp[k] * (T.pri_wfz[k] + T.pri_rfz[k] + T.prg_rfz[k] + T.prs_scw[k]
             + T.prg_scw[k] + T.prg_gcw[k] + T.prg_rcs[k] + T.prs_rcs[k] + T.prr_rci[k]
             + T.prg_rcg[k])) * orho;
    } else {
      tten[k] = tten[k]
          + (PS(lfus) * ocp[k] * (-T.prr_sml[k] - T.prr_gml[k] - T.prr_rcg[k] - T.prr_rcs[k])
             + PS(lsub) * ocp[k] * (T.prs_sde[k] + T.prg_gde[k])) * orho;
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
      xten[k] = xten[k] + (sed_r[k + 1]) * (1.0f / dz1d[k]) * (1.0f / (float )steps) / rho[k];
      xten[k] = xten[k] - (sed_r[k]) * odz * (1.0f / (float )steps) * orho;
      xr[k] = tmax(limit, xr[k] + (sed_r[k + 1]) * (1.0f / dz1d[k]) * C.global_dt * (1.0f / (float )steps));
      xr[k] = tmax(limit, xr[k] - (sed_r[k]) * odz * C.global_dt * (1.0f / (float )steps));
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
    xten[k] = xten[k] + (sed_r[k + 1]) * (1.0f / dz1d[k]) * (1.0f / (float )steps) / rho[k];
    xten[k] = xten[k] - (sed_r[k]) * odz * (1.0f / (float )steps) * orho;
    xr[k] = tmax(limit, xr[k] + (sed_r[k + 1]) * (1.0f / dz1d[k]) * C.global_dt * (1.0f / (float )steps));
    xr[k] = tmax(limit, xr[k] - (sed_r[k]) * odz * C.global_dt * (1.0f / (float )steps));
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
  fa1 = 9.0f / 16.0f;
  fa2 = 1.0f / 16.0f;
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
        afall = PS(a_coeff) * tempo_pow(((4.0f * dens_g * 9.8f) / (3.0f * rho[k])), PS(b_coeff));
        afall = afall * tempo_pow(visco[k], (1.0f - 2.0f * PS(b_coeff)));
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
        afall = PS(a_coeff) * tempo_pow(((4.0f * dens_g * 9.8f) / (3.0f * rho[k])), PS(b_coeff));
        afall = afall * tempo_pow(visco[k], (1.0f - 2.0f * PS(b_coeff)));
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
  TempoTend& T = W.T;
  const int nz1 = nz + 1;
  int substeps_sedi, ktop_sedi, n;
  float semi_sedi_factor; float  tempc; float  tc0;
  bool do_micro = false, supersaturated = false;
  const float t0 = PS(t0);
  // warp: the column scalars of W are written by the lead lane only.
  if (TEMPO_LEAD) W.did_micro = 0;

#define TEMPO_TT_ZERO(name) TEMPO_SLOTS(k, 0, TEMPO_NZP) T.name[k] = 0.0f;
  TEMPO_TEND_FIELDS(TEMPO_TT_ZERO)
#undef TEMPO_TT_ZERO

  TEMPO_LEVELS(k, nz) {
    W.tten[k] = 0.0f; W.qvten[k] = 0.0f; W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f;
    W.qsten[k] = 0.0f; W.qgten[k] = 0.0f; W.ngten[k] = 0.0f; W.qbten[k] = 0.0f; W.niten[k] = 0.0f;
    W.nrten[k] = 0.0f; W.ncten[k] = 0.0f; W.nwfaten[k] = 0.0f; W.nifaten[k] = 0.0f;
    W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
    W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.smoz[k] = 0.0f; W.ns[k] = 0.0f;
    W.vtboost[k] = 1.0f;
  }
  TEMPO_LEVELS(k, nz1) {
    W.vtrr[k] = 0.0f; W.vtnr[k] = 0.0f; W.vtrs[k] = 0.0f; W.vtri[k] = 0.0f; W.vtni[k] = 0.0f;
    W.vtrc[k] = 0.0f; W.vtnc[k] = 0.0f; W.vtrg[k] = 0.0f; W.vtng[k] = 0.0f;
  }
  if (TEMPO_LEAD) {
    W.rain_precip = 0.0f;
    W.cloud_precip = 0.0f;
    W.ice_liquid_equiv_precip = 0.0f;
    W.snow_liquid_equiv_precip = 0.0f;
    W.graupel_liquid_equiv_precip = 0.0f;
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
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
  }

  // warp: the first-call column sums and the aerosol height profiles are
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

  // Native main:334-348. MPAS leaves the internal ml_for_nc flag at false;
  // the Registry's config_tempo_ml_nc has no executable read at this pin.
  const float * ncsave = nullptr;
  if (nc1d == nullptr) {
    TEMPO_LEVELS(k, nz) {
      if(C.ml_for_nc_flag) {
        float x=qc1d[k]<=1.e-12f?0.0f:qc1d[k];
        W.nc[k]=tempo_cloud_number(x,qr1d[k],qi1d[k],qs1d[k],W.pres[k],W.temp[k],w1d[k])*W.rho[k];
      } else W.nc[k] = PS(nt_c_l); // get_constant_cloud_number, land absent
    }
    tempo_copy(W.ncsave, W.nc, nz);
    ncsave = W.ncsave;
  }

  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);

  // warp: first-call sums and graupel_init on the lead lane (serial prefix).
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

  TEMPO_LEVELS(k, nz) {
    W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f; W.qsten[k] = 0.0f; W.qgten[k] = 0.0f;
    W.ngten[k] = 0.0f; W.qbten[k] = 0.0f; W.niten[k] = 0.0f; W.nrten[k] = 0.0f; W.ncten[k] = 0.0f;
  }

#define TEMPO_THERMO()                                                                      \
  thermo_vars(P, nz, W.qv, W.temp, W.pres, W.rho, W.rhof, W.rhof2, W.qvs, W.delqvs, W.qvsi, \
              W.satw, W.sati, W.ssatw, W.ssati, W.diffu, W.visco, W.vsc2, W.ocp, W.lvap,    \
              W.tcond, W.lvt2, &supersaturated)
  TEMPO_THERMO();

  do_micro = tempo_any(W.l_qc, nz) || tempo_any(W.l_qr, nz) || tempo_any(W.l_qi, nz) ||
             tempo_any(W.l_qs, nz) || tempo_any(W.l_qg, nz);
  if (!do_micro && !supersaturated) return;
  if (TEMPO_LEAD) W.did_micro = 1;

  if (!C.turn_off_micro_flag) {
    warm_rain(P, C, nz, W.rhof, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qr, W.rr, W.nr, W.mvd_r, T);
    rain_snow_rain_graupel(P, C, nz, W.temp, W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.l_qg,
                           W.rg, W.ng, W.ilamg, W.idx_bg, T);
    ice_nucleation(P, C, nz, W.temp, W.rho, w1d, W.qv, W.qvsi, W.ssati, W.ssatw, W.nifa, W.nwfa,
                   W.ni, W.smo0, W.rc, W.nc, W.rr, W.nr, W.ilamr, T);
    ice_processes(P, C, nz, W.rhof, W.rhof2, W.rho, w1d, W.temp, W.qv, W.qvsi, W.tcond, W.diffu,
                  W.vsc2, W.ssati, W.l_qi, W.ri, W.ni, W.ilami, W.l_qs, W.rs, W.smoe, W.smof,
                  W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    riming(P, C, nz, W.temp, W.rhof, W.visco, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qs, W.rs,
           W.smo0, W.smob, W.smoc, W.smoe, W.vtboost, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    melting(P, C, nz, W.rhof2, W.rho, W.temp, W.qvsi, W.tcond, W.diffu, W.vsc2, W.ssati, W.delqvs,
            W.l_qs, W.rs, W.smof, W.smo0, W.smo1, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    aerosol_scavenging(P, C, nz, W.temp, W.rho, W.rhof, W.visco, W.nwfa, W.nifa, W.l_qr, W.nr,
                       W.ilamr, W.mvd_r, W.l_qs, W.rs, W.smob, W.smoc, W.smoe, W.l_qg, W.rg, W.ng,
                       W.ilamg, W.idx_bg, T);
  }

  check_over_depletion(P, C, nz, W.rho, W.temp, W.qvsi, W.qv, W.l_qc, W.rc, W.l_qi, W.ri, W.l_qr,
                       W.rr, W.l_qs, W.rs, W.l_qg, W.rg, T);
  sum_tendencies(P, nz, W.rho, W.temp, W.idx_bg, W.lvap, W.ocp, T, W.tten, W.qvten, W.qcten,
                 W.ncten, W.qiten, W.niten, W.qsten, W.qrten, W.nrten, W.qgten, W.ngten, W.qbten);

  TEMPO_LEVELS(k, nz) {
    W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
    tempc = W.temp[k] - t0;
    W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.nwfaten[k] = W.nwfaten[k] - (T.pna_rca[k] + T.pna_sca[k] + T.pna_gca[k] + T.pni_iha[k]) / W.rho[k];
    W.nifaten[k] = W.nifaten[k] - (T.pnd_rcd[k] + T.pnd_scd[k] + T.pnd_gcd[k]) / W.rho[k];
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

  TEMPO_THERMO();

  // cloud condensation
  if (!C.turn_off_micro_flag && C.cloud_condensation_flag) {
    cloud_condensation(P, C, nz, W.rho, W.temp, w1d, W.ssatw, W.lvap, W.tcond, W.diffu, W.lvt2,
                       W.nwfa, W.qv, W.qvs, W.l_qc, W.rc, W.nc, T);
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
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
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
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    }
    TEMPO_THERMO();
  }

  // sedimentation: rain
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
        tempo_fill(W.vtrr, 0.0f, nz1);
        tempo_fill(W.vtnr, 0.0f, nz1);
        tempo_copy(W.xrx, qr1d, nz);
        tempo_copy(W.xnx, nr1d, nz);
        rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
        rain_fallspeed(P, C, nz, W.rhof, W.l_qr, W.rr, W.ilamr, dz1d, W.vtrr, W.vtnr, nullptr, nullptr);
      }
    }
  }

  // graupel
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

  // ice
  ktop_sedi = 1;
  substeps_sedi = 1;
  if (tempo_any(W.l_qi, nz)) {
    ice_fallspeed(P, C, nz, W.rhof, W.l_qi, W.ri, W.ilami, dz1d, W.vtri, W.vtni, &substeps_sedi, &ktop_sedi);
    sedimentation(P, C, nz, W.ri, W.vtri, dz1d, W.rho, W.qiten, PS(r1), substeps_sedi, ktop_sedi,
                  &W.ice_liquid_equiv_precip, W.sed_r);
    sedimentation(P, C, nz, W.ni, W.vtni, dz1d, W.rho, W.niten, PS(r2), substeps_sedi, ktop_sedi,
                  nullptr, W.sed_r);
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

  // diagnostics (warp: the lead lane owns level 1 and the precipitation scalars)
  if (TEMPO_LEAD) {
  W.frozen_fraction =
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip) /
      (W.ice_liquid_equiv_precip + W.snow_liquid_equiv_precip + W.graupel_liquid_equiv_precip +
       W.rain_precip + PS(r1));
  freezing_rain(P, W.temp[1], W.rain_precip, W.cloud_precip, &W.frz_rain_precip);
  }
  TEMPO_LEVELS(k, nz) {
    W.cloud_number_mixing_ratio[k] = W.nc[k] * W.rho[k];
    W.rain_med_vol_diam[k] = W.mvd_r[k];
    W.graupel_med_vol_diam[k] = W.mvd_g[k];
  }
  if (C.max_hail_diameter_flag) {
    max_hail_diam(P, C, nz, W.rho, W.rg, W.ng, W.ilamg, W.idx_bg, W.max_hail_diameter);
  }
  if (C.refl10cm_flag) {
    reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp, W.l_qr, W.rr, W.nr, W.ilamr,
                      W.l_qs, W.rs, W.smoc, W.smob, W.smoz, W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg,
                      W.refl10cm);
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
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438.
struct TempoState {float f[17][TEMPO_NZP];};
TEMPO_HD void tempo_call(int nz, float dt, TempoState& s, TempoWork& work,
        const TempoBlob* blob, const int* flags, int first, const float* bl,
        float* diag, int presence) {
    TempoParams P={blob};
    TempoCtx ctx={dt, 1.0f/dt, flags[0],flags[1],0,flags[3],0,1,1,1,1,1,1,first};
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
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:74-825.
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
    tempo_call(nz,dt,s,work[c],blob,flags,first && c==0,bl?b:nullptr,d,presence);
    for(int f=0;f<17;++f)for(int k=1;k<=nz;++k)
        fields[((long long)f*nz+k-1)*columns+c]=s.f[f][k];
    for(int f=0;f<15;++f)for(int k=0;k<nz;++k)
        diag[((long long)f*nz+k)*columns+c]=d[f*nz+k];
}
"""

# One warp per column, lanes across levels, the n_microp substeps inside the
# kernel.  Per substep it evaluates exactly what launch_mpas's former host loop
# did: tempo_pack (theta * exner, absent carriers zero), tempo_columns on zeroed
# work, tempo_pack's unpack (theta = T / exner) and tempo_step_diags.  The
# carriers stay in the warp's state between substeps (the unpack/pack copies
# are identities), theta goes through the same divide and multiply, and the
# pools are written once at the end.  Columns are striped over the resident
# warps, so the work area is per warp, not per column.
_WARP = r"""
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:614-646;
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438.
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
    int aero,int hail,int ml_bl,int melt,int first,int presence,
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
            TempoCtx ctx={dt,1.0f/dt,aero,hail,0,melt,0,1,last,re,re,re,last,first && sub==0 && c==0};
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


#: What tempo_pow_base/tempo_pow_with call into: the header's mpas_pow must be glibc's binary64 pow
#: (g64_pow::pow behind glibc_pow's NaN prelude), whose fast path they finish.
_SHARED_POW_MARKERS = ("namespace g64_pow {", "__device__ __forceinline__ double pow (double x, double y)",
                       "log_inline (uint64_t ix, double *tail)", "exp_inline (double x, double xtail, uint32_t sign_bias)",
                       "return g64_pow::pow(x,y);", "return glibc_pow(x, y);", "#define mpas_pow thompson_aa_pow")


def _shared_pow_define(device_math_source: str) -> str:
    """``#define TEMPO_SHARED_POW`` when the supplied header's mpas_pow is glibc pow (same words either way)."""
    return "#define TEMPO_SHARED_POW 1\n" if all(m in device_math_source for m in _SHARED_POW_MARKERS) else ""


def _fast(numerics: str | None = None) -> bool:
    """Whether this numerics mode is fast (the process mode when none is named)."""
    return (_hex_numerics.mode() if numerics is None else _hex_numerics.normalise(numerics)) == "fast"


def _column_source(numerics: str | None = None) -> str:
    """The column support for this numerics mode: strict is ``_COLUMN`` byte for byte; fast (rrfs/tempo-yolo)
    fixes the fall speeds for the call and runs the sedimentation substeps in registers
    (tempo_fastsed_v303.fast_sed_source, warp shape), narrows TEMPO's own binary64 arithmetic to binary32
    (tempo_fastdp_v303.fast_dp_source), removes every remaining binary64 (tempo_allfloat_v303.all_float_source:
    binary32 blob, constant-bank parameters, binary32 intrinsics; in place of tempo_fastmath_v303.fast_source),
    then treats graupel traces below 1e-9 kg/kg as absent (tempo_tinymass_v303.tiny_mass_source)."""
    if not _fast(numerics):
        return _COLUMN
    from woof.hex.mpasphys.rrfs.tempo_fastdp_v303 import fast_dp_source
    from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import all_float_source
    from woof.hex.mpasphys.rrfs.tempo_fastsed_v303 import fast_sed_source
    from woof.hex.mpasphys.rrfs.tempo_tinymass_v303 import tiny_mass_source
    return tiny_mass_source(all_float_source(fast_dp_source(fast_sed_source(_COLUMN))))


def _mpas_source(numerics: str | None = None) -> str:
    """The MPAS slice kernels: strict is ``_MPAS`` byte for byte; fast swaps their glibc replicas for CUDA
    binary32 functions (tempo_allfloat_v303.all_float_mpas)."""
    if not _fast(numerics):
        return _MPAS
    from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import all_float_mpas
    return all_float_mpas(_MPAS)


@lru_cache(maxsize=None)
def _scalar_offsets() -> tuple:
    """Blob word offsets of the extent-1 fields, in the order of the fast module's constant-bank block."""
    import math
    from woof.hex.mpasphys.rrfs.tempo_allfloat_v303 import scalar_names
    names = scalar_names(_COLUMN)
    offsets, offset = {}, 0
    for name, shape in TABLE_LAYOUT:
        offsets[name] = offset
        offset += math.prod(shape)
    if names != [n for n, shape in TABLE_LAYOUT if math.prod(shape) == 1]:
        raise ValueError("TEMPO column fields differ from TABLE_LAYOUT")
    return tuple(offsets[n] for n in names)


def fast_tables(tables):
    """The binary32 blob of the fast module: float64 rounded to nearest, binary32 subnormals flushed to signed
    zero (the value every fast read of the float64 blob produced: cvt.rn.ftz.f32.f64). A float32 blob is
    returned as it is."""
    import cupy as cp
    if tables.dtype == cp.float32:
        return tables
    if tables.dtype != cp.float64 or not tables.flags.c_contiguous:
        raise ValueError("TEMPO tables must be the Rust-prepared contiguous native blob")
    t = tables.astype(cp.float32)
    return cp.where(cp.abs(t) < np.float32(np.finfo(np.float32).tiny), cp.copysign(cp.float32(0), t), t)


_SCALAR_INDEX = {}


def _set_scalars(module, tables):
    """Write the fast module's constant-bank parameter block (tempo_k) from the binary32 blob, on the current
    stream, before the launch that reads it."""
    import cupy as cp
    device = tables.device.id
    index = _SCALAR_INDEX.get(device)
    if index is None:
        index = _SCALAR_INDEX[device] = cp.asarray(np.asarray(_scalar_offsets(), dtype=np.int64))
    words = cp.take(tables, index)
    symbol = module.get_global_var("tempo_k")  # the symbol's device address (CuPy returns it as an int)
    cp.cuda.runtime.memcpyAsync(int(getattr(symbol, "ptr", symbol)), words.data.ptr, words.nbytes,
                                cp.cuda.runtime.memcpyDeviceToDevice, cp.cuda.get_current_stream().ptr)


def _device_tables(module, tables, blob_bytes):
    """The blob a launch passes: binary64 in strict; binary32 in fast, with the parameter block written."""
    import cupy as cp
    if _fast():
        tables = fast_tables(tables)
        if not tables.flags.c_contiguous or tables.nbytes != blob_bytes:
            raise ValueError("TEMPO table blob has a different layout than the native module")
        _set_scalars(module, tables)
        return tables
    if tables.dtype != cp.float64 or not tables.flags.c_contiguous or tables.nbytes != blob_bytes:
        raise ValueError("TEMPO table blob has a different layout than the native module")
    return tables


def cuda_source(device_math_source: str, *, max_levels: int = 100, numerics: str | None = None) -> str:
    """Compose the full module with the shared, host-identified math header.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:74-3153.
    """
    if not device_math_source.strip():
        raise ValueError("TEMPO requires the shared device mathematics header")
    if type(max_levels) is not int or max_levels < 1:
        raise ValueError("TEMPO column capacity must be a positive integer")
    return (f"#define TEMPO_NZ_MAX {max_levels}\n" + _shared_pow_define(device_math_source) + device_math_source
            + _column_source(numerics) + _DRIVER + _mpas_source(numerics))


# Unbounded on purpose (one entry per card and variant): a bounded per-card cache thrashes once ranks x
# variants exceed it, and every rank then rebuilds its module each step (thompson_v842._module, 8 cards,
# 2026-10-28: 1.28 s/step against 0.25 s with the cache fixed).
@lru_cache(maxsize=None)
def _compiled(device: int, device_math_source: str, max_levels: int):
    """Compile without CuPy RawModule's appended FTZ=true override.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438.
    """
    import cupy as cp
    from cupy.cuda import compiler, function
    with cp.cuda.Device(device):
        image, _ = compiler.compile_using_nvrtc(
            cuda_source(device_math_source, max_levels=max_levels),
            options=_hex_numerics.options(COMPILE_OPTIONS), cache_in_memory=True)
        module = function.Module()
        module.load(image)
        sizes = cp.empty(3, dtype=cp.uint64)
        module.get_function("tempo_sizes")((1,), (1,), (sizes,))
        return module, tuple(map(int, sizes.get()))


#: Threads per block of the warp kernel: one column per block, so a finished column frees its slot at once
#: (with two per block the slot waited for the slower partner).
WARP_BLOCK = 32
#: Warps launched per resident warp slot: scheduling slack for uneven columns,
#: while the work area stays bounded by the card, not by the mesh.
WARP_WAVES = 4


def warp_source(device_math_source: str, *, max_levels: int = 100, numerics: str | None = None) -> str:
    """The warp-per-column module: the same column routines and maths with
    TEMPO_WARP 1, the warp driver and the MPAS slice kernels.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_main.F90:74-3153.
    """
    if not device_math_source.strip():
        raise ValueError("TEMPO requires the shared device mathematics header")
    if type(max_levels) is not int or max_levels < 1:
        raise ValueError("TEMPO column capacity must be a positive integer")
    return (f"#define TEMPO_NZ_MAX {max_levels}\n#define TEMPO_WARP 1\n" + _shared_pow_define(device_math_source)
            + device_math_source + _column_source(numerics) + _WARP + _mpas_source(numerics))


# Per-card module cache, unbounded on purpose: entries are cards x variants. A bounded LRU thrashes once
# ranks x variants exceed it, and the evicted card reloads its module on every call (thompson_v842._module
# at maxsize=8 on 8 cards, measured 2026-10-28: 1.28 s/step against 0.25 s unbounded).
@lru_cache(maxsize=None)
def _compiled_warp(device: int, device_math_source: str, max_levels: int):
    """Compile the warp module (same options as the column module); sizes and resident warps.

    Served from the shipped AOT kernel pack when it holds this source (aot_build.physics_inventory's
    tempo_rrfs rows); otherwise NVRTC compiles it exactly as before. THE BREAKAGE THIS ENDS: the
    module was absent from every pack, so a fresh box compiled it on one core at the first
    microphysics step (27-29 s on an RTX 5090 host, both numerics).
    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:568-825.
    """
    import cupy as cp
    from cupy.cuda import compiler, driver, function, runtime
    from woof.hex.cuda_backend.aot import direct_image
    with cp.cuda.Device(device):
        image = direct_image(warp_source(device_math_source, max_levels=max_levels),
                             _hex_numerics.options(COMPILE_OPTIONS), compiler._get_arch())
        module = function.Module()
        module.load(image)
        sizes = cp.empty(3, dtype=cp.uint64)
        module.get_function("tempo_warp_sizes")((1,), (1,), (sizes,))
        kernel = module.get_function("tempo_warp_steps")
        sms = runtime.deviceGetAttribute(runtime.cudaDevAttrMultiProcessorCount, device)
        blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.ptr, WARP_BLOCK, 0)
        resident = max(1, sms * blocks) * (WARP_BLOCK // 32)
        return module, tuple(map(int, sizes.get())), resident


import threading as _threading

_WARP_WORK = {}
_WARP_WORK_LOCK = _threading.Lock()
_ORDER_WORK = {}


def _order_work(device: int, cells: int):
    """Per-column cost (previous call's cycles/16), column order and bucket counts, kept across calls.

    One entry per (device, stream), as _warp_work: same-host rank threads share this module, may share a card, and
    own different cell counts (1,283 vs 1,279 on 2 ranks of x1.2562). THE BREAKAGE THIS PREVENTS: a key of
    (device, cells) cleared on any new key let every rank's TEMPO call free the other ranks' cost/order/hist
    buffers while their launches could still read them (the bug class that stalled 4-rank runs, 2026-10-28) and
    reset the cost history the column ordering reads. A rank evicts only its own stream's entries.
    """
    import cupy as cp
    key = (device, cp.cuda.get_current_stream().ptr, cells)
    with _WARP_WORK_LOCK:
        if key not in _ORDER_WORK:
            for old in [k for k in _ORDER_WORK if k[:2] == key[:2]]:
                del _ORDER_WORK[old]
            _ORDER_WORK[key] = (cp.zeros(cells, dtype=cp.uint32), cp.empty(cells, dtype=cp.int32),
                                cp.empty(128, dtype=cp.uint32))
        return _ORDER_WORK[key]


def _warp_work(device: int, nz: int, warps: int, sizes):
    """Per-warp state, work and PBL scratch, kept across calls (the kernel zeroes work itself).

    One scratch per (device, stream): same-host rank threads (samehost_v842) share this module and may share a
    card, each on its own stream. THE BREAKAGE THIS PREVENTS: one entry per device, cleared on any new key,
    let ranks free and reallocate each other's scratch every microphysics call; two 4-rank hrrrv5 runs stalled
    with a rank inside the TEMPO launch (2026-10-28). A rank evicts only its own stream's entries.
    """
    import cupy as cp
    key = (device, cp.cuda.get_current_stream().ptr, nz, warps, sizes)
    with _WARP_WORK_LOCK:
        if key not in _WARP_WORK:
            for old in [k for k in _WARP_WORK if k[:2] == key[:2]]:
                del _WARP_WORK[old]
            work_bytes, state_bytes, _ = sizes
            _WARP_WORK[key] = (cp.empty(warps * state_bytes, dtype=cp.uint8),
                               cp.empty(warps * work_bytes, dtype=cp.uint8),
                               cp.empty(warps * nz * 2, dtype=cp.float32))
        return _WARP_WORK[key]


def launch_columns(fields, tables, *, dt, flags, device_math_source,
                   first_call_main=False, pbl=None, presence):
    """Run prepared MPAS columns in place, preserving optional carrier presence.

    `fields` has shape (17, levels, cells), float32, column-fastest. `tables`
    is the promoted float64 blob from pristine native initialization, prepared
    by Rust. Flags are aerosolaware, hailaware, ml_for_bl_nc, melting reflection.
    Presence is the five-bit NC/NWFA/NIFA/NG/QB argument mask. MPAS can allocate
    the first three through MYNN independently of the TEMPO aerosol flag.
    The caller owns first_call_main and its restart serialization.
    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:187-438.
    """
    import cupy as cp
    if fields.ndim != 3 or fields.shape[0] != len(STATE_FIELDS):
        raise ValueError("TEMPO needs 17 prepared carriers, levels, cells")
    if fields.dtype != cp.float32 or not fields.flags.c_contiguous:
        raise ValueError("TEMPO MPAS carriers must be contiguous float32")
    _, nz, cells = fields.shape
    if nz < 1 or cells < 1 or len(flags) != 4:
        raise ValueError("TEMPO needs positive extents and four configuration flags")
    if tables.dtype not in ((cp.float64, cp.float32) if _fast() else (cp.float64,)) or not tables.flags.c_contiguous:
        raise ValueError("TEMPO tables must be the Rust-prepared contiguous native blob")
    module, (work_bytes, state_bytes, blob_bytes) = _compiled(
        cp.cuda.runtime.getDevice(), device_math_source, nz)
    tables = _device_tables(module, tables, blob_bytes)
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
        np.int32(first_call_main), state, work, blwork, diagwork,np.int32(presence)))
    return diagnostics

# The carrier list preserves MPAS's names and one-based source-level order.
FIELD_ARGUMENTS = ("th","p","w","dz","qv","qc","qr","qi","qs","qg","ni","nr","nc","nwfa","nifa","ng","qb","pii","rho","z")
MODEL_ARGUMENTS = ("qv","qc","qr","qi","qs","qg","ni","nr","nc","nwfa","nifa","ng","volg","theta_m","rho_zz","w")
DIAG_ARGUMENTS = ("exner","pressure_base","pressure_p","exner_base","rtheta_base","rtheta_p","dtheta_dt_mp","rt_diabatic_tend","surface_pressure","tend_sfc_pressure")
TEND_ARGUMENTS = ("rthmpten","rqvmpten","rqcmpten","rqrmpten","rqimpten","rqsmpten","rqgmpten","rnimpten","rnrmpten","rncmpten","rnifampten","rnwfampten")
SURFACE_ARGUMENTS = ("rainncv","rainnc","snowncv","snownc","graupelncv","graupelnc","sr","frainnc","max_hail_diameter_sfc","max_hail_diameter_column","nwfa2d","nifa2d")
PRECIP_ARGUMENTS = ("precipw","rainncv","rainnc","snowncv","snownc","graupelncv","graupelnc","sr","frainnc","max_hail_diameter_sfc","max_hail_diameter_column","max_hail_diameter_sfc_acc","max_hail_diameter_column_acc")
PROFILE_ARGUMENTS = ("refl10cm","re_cloud","re_ice","re_snow")
RADAR_ARGUMENTS = ("refl10cm_max","refl10cm_1km","refl10cm_1km_max")
CONFIG_ARGUMENTS = ("config_tempo_aerosolaware","config_tempo_hailaware","config_tempo_ml_for_bl_nc","config_tempo_refl10cm_from_melting")
CONSTANT_ARGUMENTS = ("R_v","R_d","rvord","P0","rcv","gravity")

_MPAS = r"""
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:439-454.
extern "C" __global__ void tempo_emissions(float* nwfa,const float* surface,
                                         float dt,int columns) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<columns)
        nwfa[c]=__fadd_rn(nwfa[c],__fmul_rn(surface[c],dt));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:321-336,409-436.
extern "C" __global__ void tempo_temperature(float* th,const float* exner,
                                           int n,int phase) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)
        th[i]=phase==0?__fmul_rn(th[i],exner[i]):__fdiv_rn(th[i],exner[i]);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:699-748,828-950,1067-1109.
extern "C" __global__ void tempo_from(float** m,float** mesh,float** d,float** t,
 float** f,int nz,int nc,int aero,int hail,float rv,float rd) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nz*nc)return;
 int maps[13]={4,5,6,7,8,9,10,11,12,13,14,15,16};
 for(int j=0;j<13;++j)if(j<8 || (j<11?aero:hail)) f[maps[j]][x]=m[j][x];
 f[18][x]=mesh[0][x]*m[14][x];
 f[0][x]=m[13][x]/(1.0f+(rv/rd)*tmax(0.0f,f[4][x]));
 f[17][x]=d[0][x];f[1][x]=d[1][x]+d[2][x];
 f[19][x]=mesh[1][x];f[3][x]=mesh[1][x+nc]-mesh[1][x];f[2][x]=m[15][x];
 t[0][x]=m[13][x]/(1.0f+(rv/rd)*tmax(0.0f,m[0][x]));
 for(int j=0;j<8;++j)t[j+1][x]=m[j][x];
 if(aero){t[9][x]=m[8][x];t[10][x]=m[10][x];t[11][x]=m[9][x];}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:321-438.
extern "C" __global__ void tempo_pack(float** f,float* a,int n,int unpack) {
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;
 for(int j=0;j<17;++j){if(!f[j]){if(!unpack)a[(long long)j*n+x]=0.0f;continue;}if(!unpack)a[(long long)j*n+x]=j==0?f[0][x]*f[17][x]:f[j][x];
 else if(j==0 || j>=4)f[j][x]=j==0?a[x]/f[17][x]:a[(long long)j*n+x];}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:614-646.
extern "C" __global__ void tempo_step_diags(const float* out,float** s,float** profiles,int nz,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 float r=out[c],i=out[(long long)nz*nc+c],sn=out[2LL*nz*nc+c],g=out[3LL*nz*nc+c];
 s[2][c]=(s[2][c]+i)+sn;s[3][c]=(s[3][c]+i)+sn;
 s[4][c]+=g;s[5][c]+=g;s[0][c]=(((s[0][c]+r)+i)+sn)+g;s[1][c]=(((s[1][c]+r)+i)+sn)+g;
 s[6][c]=out[5LL*nz*nc+c];s[7][c]+=out[4LL*nz*nc+c];
 s[8][c]=out[6LL*nz*nc+c];s[9][c]=out[7LL*nz*nc+c];
 for(int f=0;f<4;++f)for(int k=0;k<nz;++k)profiles[f][k*nc+c]=out[((long long)(f+8)*nz+k)*nc+c];
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:852-1042,1350-1380.
extern "C" __global__ void tempo_precip(float** f,float** s,float** d,int* counter,
 int nz,int nc,int finish,int accrue,float bucket) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(!finish){for(int j=0;j<8;++j)s[j][c]=0.0f;d[1][c]=0.0f;d[3][c]=0.0f;d[5][c]=0.0f;d[7][c]=0.0f;return;}
 d[11][c]=tmax(d[11][c],s[8][c]);d[12][c]=tmax(d[12][c],s[9][c]);
 d[0][c]=0.0f;
 for(int k=0;k<nz;++k){int x=k*nc+c;float rho=f[18][x]/(1.0f+f[4][x]);d[0][c]+=f[4][x]*rho*f[3][x];}
 d[1][c]=s[1][c];d[2][c]+=d[1][c];
 if(accrue && bucket>0.0f && d[2][c]>bucket){counter[c]+=1;d[2][c]-=bucket;}
 d[3][c]=s[3][c];d[5][c]=s[5][c];d[7][c]=(s[3][c]+s[5][c])/(s[1][c]+1.e-12f);
 d[4][c]+=d[3][c];d[6][c]+=d[5][c];d[8][c]+=s[7][c];
}
// RRFS 44c0849 mpas_atmphys_driver_microphysics.F:1038-1060.
extern "C" __global__ void tempo_prate(int nc,int longest,float dt,
 const float* step,const float* total,float* peak,float* rolling,
 float* ave1,float* ave5,float* ave10,int n1,int n5,int n10,
 float time1,float time5,float time10) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 peak[c]=tmax(__fdiv_rn(step[c],dt),peak[c]);
 if(!rolling)return;
 for(int k=longest-1;k>0;--k)rolling[k*nc+c]=rolling[(k-1)*nc+c];
 rolling[c]=total[c];
 ave1[c]=tmax(__fdiv_rn(__fsub_rn(rolling[c],rolling[(n1-1)*nc+c]),time1),ave1[c]);
 ave5[c]=tmax(__fdiv_rn(__fsub_rn(rolling[c],rolling[(n5-1)*nc+c]),time5),ave5[c]);
 ave10[c]=tmax(__fdiv_rn(__fsub_rn(rolling[c],rolling[(n10-1)*nc+c]),time10),ave10[c]);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:1195-1270,1354-1456,1572-1616.
extern "C" __global__ void tempo_to(float** m,float** mesh,float** d,float** t,
 float** f,int nz,int nc,int aero,int hail,float dt,float rv,float rd,float rvord,float p0,float rcv,float grav,
 int mp_top_level,float* compmax,const float* comp) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 int maps[13]={4,5,6,7,8,9,10,11,12,13,14,15,16};
 for(int k=0;k<nz;++k){int x=k*nc+c;
 if(k<mp_top_level){
 d[7][x]=m[13][x];d[6][x]=m[13][x]/(1.0f+rvord*m[0][x]);
 for(int j=0;j<3;++j)m[j][x]=f[maps[j]][x];
 m[13][x]=f[0][x]*(1.0f+rvord*f[4][x]);
 d[7][x]=(m[13][x]-d[7][x])/dt;d[6][x]=(m[13][x]/(1.0f+rvord*m[0][x])-d[6][x])/dt;
 d[5][x]=m[14][x]*m[13][x]-d[4][x];d[0][x]=mpas_powf(mesh[0][x]*(rd/p0)*(d[5][x]+d[4][x]),rcv);
 d[2][x]=mesh[0][x]*rd*(d[0][x]*d[5][x]+(d[0][x]-d[3][x])*d[4][x]);
 }else{m[1][x]=0.f;m[2][x]=0.f;d[7][x]=0.f;}
 // TEMPO ice and number carriers retain the full native vertical extent.
 for(int j=3;j<13;++j)if(j<8 || (j<11?aero:hail))m[j][x]=f[maps[j]][x];
 if(k<mp_top_level){
 t[0][x]=(m[13][x]/(1.0f+(rv/rd)*tmax(0.0f,m[0][x]))-t[0][x])/dt;
 for(int j=0;j<6;++j)t[j+1][x]=(m[j][x]-t[j+1][x])/dt;
 }else{for(int j=0;j<7;++j)t[j][x]=0.f;}
 for(int j=6;j<8;++j)t[j+1][x]=(m[j][x]-t[j+1][x])/dt;
 if(aero){t[9][x]=(m[8][x]-t[9][x])/dt;t[10][x]=(m[10][x]-t[10][x])/dt;t[11][x]=(m[9][x]-t[11][x])/dt;}}
 if(compmax)compmax[c]=tmax(compmax[c],comp[c]);
 float dz1=mesh[1][nc+c]-mesh[1][c],dz2=mesh[1][2*nc+c]-mesh[1][nc+c];
 float rho1=m[14][c]*mesh[0][c]*(1.0f+f[4][c]);float rho2=m[14][nc+c]*mesh[0][nc+c]*(1.0f+f[4][nc+c]);
 d[9][c]=d[8][c];d[8][c]=0.5f*grav*dz1*(rho1-0.5f*(rho2-rho1)*dz1/(dz1+dz2));
 d[8][c]=(d[8][c]+d[2][c])+d[1][c];d[9][c]=(d[8][c]-d[9][c])/dt;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_functions.F:191-217.
TEMPO_HD float tempo_mpas_rsif(float p,float t) {
 const float c0=.609868993E03f,c1=.499320233E02f,c2=.184672631E01f,c3=.402737184E-1f,c4=.565392987E-3f,c5=.521693933E-5f,c6=.307839583E-7f,c7=.105785160E-9f,c8=.161444444E-12f;
 float x=tmax(-80.0f,t-273.16f);
 float esi=c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+x*(c6+x*(c7+x*c8)))))));
 esi=tmin(esi,p*0.15f);return .622f*esi/(p-esi);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:1130-1165,1336-1343,1383-1435.
extern "C" __global__ void tempo_radar(float** f,float** profile,float** radar,float* rh,
 int nz,int nc,int hourly,int full,const int* hourly_word) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 // do_diag_dbz read here, not on the host: the step queues without waiting for the GPU.
 if(hourly_word)hourly=(*hourly_word==1);
 if(full){int kp=0;float maxz=-35.0f;
 for(int k=0;k<nz;++k){int x=k*nc+c;maxz=tmax(maxz,tmax(-35.0f,profile[0][x]));
 float zp=f[19][x]-f[19][c]+0.5f*f[3][x];if(zp<1000.0f)kp=k;
 float temp=f[0][x]*f[17][x];float qs=temp-273.16f<=0.0f?tempo_mpas_rsif(f[1][x],temp):calc_rslf(f[1][x],temp);
 rh[x]=(f[4][x]/qs)*100.0f;}
 float z1=f[19][kp*nc+c]-f[19][c]+0.5f*f[3][kp*nc+c];
 float z2=f[19][(kp+1)*nc+c]-f[19][c]+0.5f*f[3][(kp+1)*nc+c];
 float w1=(z2-1000.0f)/(z2-z1),w2=1.0f-w1;radar[0][c]=maxz;
 radar[1][c]=w1*tmax(-35.0f,profile[0][kp*nc+c])+w2*tmax(-35.0f,profile[0][(kp+1)*nc+c]);}
 if(hourly)radar[2][c]=tmax(radar[2][c],radar[1][c]);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init_tempo.F:30-187.
extern "C" __global__ void tempo_init_aerosols(float* nwfa,float* nifa,float* nwfa2d,float* nifa2d,
 const float* z,const float* area,int nz,int nc,int initw,int initi,const TempoBlob* blob) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;TempoParams P={blob};
 float h1=0.5f*(z[c]+z[nc+c]);
 float h01=h1<=1000.0f?0.8f:(h1>=2500.0f?0.01f:0.8f*mpas_cosf(h1*0.001f-1.0f));
 float cc=-1.0f*mpas_logf(PS(naccn1)/PS(naccn0))/h01;
 float in=-1.0f*mpas_logf(PS(nain1)/PS(nain0))/h01;
 for(int k=0;k<nz;++k){int j=k==0?1:k;float h=0.5f*(z[j*nc+c]+z[(j+1)*nc+c]);
 if(initw)nwfa[k*nc+c]=PS(naccn1)+PS(naccn0)*mpas_expf(-((h-h1)/1000.0f)*cc);
 if(initi)nifa[k*nc+c]=PS(nain1)+PS(nain0)*mpas_expf(-((h-h1)/1000.0f)*in);}
 float dz=z[nc+c]-z[c];if(dz<9.0f)dz=9.0f;
 nwfa2d[c]=tmax(PS(nwfa_default),tmin(PS(aero_max),nwfa[c]))*0.000196f*(5.0f/dz)*(area[c]/9.e6f);nifa2d[c]=0.0f;
}
"""


def _pointer_array(arrays, *, shape, device, optional=False):
    """Validate and bind CUDA pool views; RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:699-748."""
    import cupy as cp
    pointers=[]
    for a in arrays:
        if a is None and optional:pointers.append(0);continue
        if a.dtype != cp.float32 or not a.flags.c_contiguous or a.device.id != device:
            raise ValueError("TEMPO carriers must be contiguous native RKIND on the tile device")
        if shape is not None and a.shape != shape:
            raise ValueError("TEMPO carrier extent differs from the native solve tile")
        pointers.append(a.data.ptr)
    # The pool views live for the run: upload each pointer table once, not every step.
    key=(device,tuple(pointers))
    table=_POINTER_TABLES.get(key)
    if table is None:
        if len(_POINTER_TABLES)>=64:_POINTER_TABLES.clear()
        table=_POINTER_TABLES[key]=cp.asarray(pointers,dtype=cp.uint64)
    return table


_POINTER_TABLES={}


def launch_mpas(p, *, dt_dyn, flags, bucket, diagnostics, hourly):
    """Full fork driver slice; RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:568-825."""
    import cupy as cp
    fields=p["fields"];nz,cells=fields["qv"].shape;device=fields["qv"].device.id
    if type(p["n_microp"]) is not int or p["n_microp"]<1:
        raise ValueError("native manager must supply a positive microphysics substep count")
    phase=_phase_mode()
    if phase:
        # Fast numerics: (level, cell) phase kernels + warp tail (tempo_phase_v303); its module S also
        # carries the driver slice kernels below. Both modules read the binary32 blob (converted once per
        # blob) and their own constant-bank parameter block, written here for this launch.
        from woof.hex.mpasphys.rrfs import tempo_phase_v303 as _phase
        pmod,module,sizes,_,_=_phase._modules(device,p["device_math_source"],nz,_phase.inline_mode())
        tables=_phase.float_blob(p["tables"])
        if not tables.flags.c_contiguous or tables.nbytes!=sizes[2]:
            raise ValueError("TEMPO table blob has a different layout than the native module")
        _set_scalars(pmod,tables);_set_scalars(module,tables)
    else:
        module,sizes,resident=_compiled_warp(device,p["device_math_source"],nz)
        # Fast: the binary32 blob (the bind converts it once) and the constant-bank parameter block for this launch.
        tables=_device_tables(module,p["tables"],sizes[2])
    for names,values in ((FIELD_ARGUMENTS[:12]+FIELD_ARGUMENTS[17:]+(FIELD_ARGUMENTS[12:15] if flags[0] else ())+(FIELD_ARGUMENTS[15:17] if flags[1] else ()),fields),
                         (MODEL_ARGUMENTS[:8]+MODEL_ARGUMENTS[13:]+(MODEL_ARGUMENTS[8:11] if flags[0] else ())+(MODEL_ARGUMENTS[11:13] if flags[1] else ()),p["model"]),
                         (TEND_ARGUMENTS if flags[0] else TEND_ARGUMENTS[:9],p["tendencies"]),
                         (SURFACE_ARGUMENTS[:10]+(SURFACE_ARGUMENTS[10:] if flags[0] else ()),p["surface"])):
        absent=[name for name in names if values.get(name) is None]
        if absent:raise ValueError(f"allocated native TEMPO carriers missing: {absent}")
    f=_pointer_array([fields.get(n) for n in FIELD_ARGUMENTS],shape=(nz,cells),device=device,optional=True)
    m=_pointer_array([p["model"].get(n) for n in MODEL_ARGUMENTS],shape=None,device=device,optional=True)
    mesh=_pointer_array([p["mesh"][n] for n in ("zz","zgrid")],shape=None,device=device)
    d=_pointer_array([p["diagnostics"][n] for n in DIAG_ARGUMENTS],shape=None,device=device)
    t=_pointer_array([p["tendencies"].get(n) for n in TEND_ARGUMENTS],shape=(nz,cells),device=device,optional=True)
    s=_pointer_array([p["surface"].get(n) for n in SURFACE_ARGUMENTS],shape=(cells,),device=device,optional=True)
    pr=_pointer_array([p["precip"][n] for n in PRECIP_ARGUMENTS],shape=(cells,),device=device)
    profiles=_pointer_array([p["profiles"][n] for n in PROFILE_ARGUMENTS],shape=(nz,cells),device=device)
    c=p["constants"];extent=(np.int32(nz),np.int32(cells));af=(np.int32(flags[0]),np.int32(flags[1]))
    def run(name,count,args):
        """CUDA orchestration for MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:568-825."""
        module.get_function(name)(((count+31)//32,),(32,),args)
    counter=p["precip"]["i_rainnc"]
    if counter.dtype!=cp.int32 or counter.shape!=(cells,):raise ValueError("native rain bucket counter must be int32 per solve cell")
    precip_args=(f,s,pr,counter)+extent
    run("tempo_precip",cells,precip_args+(np.int32(0),np.int32(p["l_acrain"]),np.float32(bucket)))
    run("tempo_from",nz*cells,(m,mesh,d,t,f)+extent+af+(np.float32(c["R_v"]),np.float32(c["R_d"])))
    for name,value in p.get("surface_inputs",{}).items():cp.copyto(p["surface"][name],value)
    if flags[0]:
        run("tempo_emissions",cells,(fields["nwfa"],p["surface"]["nwfa2d"],np.float32(p["dt_microp"]),np.int32(cells)))
    presence=sum(1<<i for i,n in enumerate(FIELD_ARGUMENTS[12:17]) if fields.get(n) is not None)
    pbl=p.get("pbl")
    if pbl is not None and (pbl.shape!=(2,nz,cells) or pbl.dtype!=cp.float32 or not pbl.flags.c_contiguous):
        raise ValueError("TEMPO PBL cloud carriers must be contiguous float32 (2, levels, cells)")
    if phase:
        _phase.launch_steps(device=device,nz=nz,cells=cells,steps=p["n_microp"],dt=p["dt_microp"],f=f,bl=pbl,s=s,
                            profiles=profiles,tables=tables,flags=flags,first=bool(p["first_call_main"]),
                            presence=presence,radii=bool(p.get("radii",True)),device_math_source=p["device_math_source"],
                            force_warp=(phase=="warp"))
    else:
        _launch_warp(module,resident,sizes,device,nz,cells,f,pbl,s,profiles,p,flags,presence,tables)
    p["first_call_main"]=False
    if "resources" in p:p["resources"]["first_call_main"]=False
    _launch_tail(module,p,f,s,profiles,pr,counter,precip_args,m,mesh,d,t,extent,af,c,nz,cells,device,flags,diagnostics,hourly,dt_dyn,bucket,run)


def _phase_mode():
    """Fast numerics runs TEMPO as phase kernels (tempo_phase_v303) unless WOOF_HEX_TEMPO_PHASE=0;
    WOOF_HEX_TEMPO_PHASE=warp keeps the warp kernel with the fast binary32 parameter storage. Strict never."""
    import os
    if _hex_numerics.mode()!="fast":return None
    v=os.environ.get("WOOF_HEX_TEMPO_PHASE","1").strip().lower()
    if v in ("0","off","false","no"):return None
    return "warp" if v=="warp" else "phase"


def _launch_warp(module,resident,sizes,device,nz,cells,f,pbl,s,profiles,p,flags,presence,tables):
    """The warp-per-column kernel (strict, and fast with WOOF_HEX_TEMPO_PHASE=0)."""
    import cupy as cp
    def run(name,count,args):
        module.get_function(name)(((count+31)//32,),(32,),args)
    # All n_microp substeps in one launch: one warp per column, lanes across levels
    # (the former host loop of pack, columns, unpack and step diagnostics, in order).
    blocks=-(-min(cells,WARP_WAVES*resident)*32//WARP_BLOCK)
    warps=blocks*(WARP_BLOCK//32)
    states,work,blwork=_warp_work(device,nz,warps,sizes)
    # Heaviest columns first (previous call's per-column cost): scheduling only, no word changes.
    cost,order,hist=_order_work(device,cells)
    hist.fill(0)
    run("tempo_order_hist",cells,(cost,np.int32(cells),hist))
    module.get_function("tempo_order_scan")((1,),(1,),(hist,))
    run("tempo_order_scatter",cells,(cost,np.int32(cells),hist,order))
    module.get_function("tempo_warp_steps")((blocks,),(WARP_BLOCK,),(
        np.int32(nz),np.int32(cells),np.int32(p["n_microp"]),np.float32(p["dt_microp"]),f,
        pbl if pbl is not None else np.uint64(0),s,profiles,tables,
        *(np.int32(bool(v)) for v in flags),np.int32(bool(p["first_call_main"])),np.int32(presence),
        states,work,blwork,order,cost,np.int32(bool(p.get("radii",True)))))


def _launch_tail(module,p,f,s,profiles,pr,counter,precip_args,m,mesh,d,t,extent,af,c,nz,cells,device,flags,diagnostics,hourly,dt_dyn,bucket,run):
    """Driver slice after the microphysics (radar, precip, MPAS tendencies); mpas_atmphys_driver_microphysics.F:780-825."""
    import cupy as cp
    # hourly: a host bool, or the device do_diag_dbz word (int32), read by tempo_radar itself.
    word=hourly if isinstance(hourly,cp.ndarray) else None
    if word is not None and (word.dtype!=cp.int32 or word.size!=1 or word.device.id!=device):
        raise ValueError("TEMPO hourly switch must be the device do_diag_dbz int32 word")
    if diagnostics or word is not None or hourly:
        radar=_pointer_array([p["diagnostics"][n] for n in RADAR_ARGUMENTS],shape=(cells,),device=device)
        run("tempo_radar",cells,(f,profiles,radar,p["diagnostics"]["relhum"])+extent+(np.int32(bool(hourly) if word is None else 0),np.int32(diagnostics),
            word if word is not None else np.uint64(0)))
    run("tempo_precip",cells,precip_args+(np.int32(1),np.int32(p["l_acrain"]),np.float32(bucket)))
    if p.get("prate") is not None:
        rate=p["prate"]
        run("tempo_prate",cells,(np.int32(cells),np.int32(rate["longest"]),np.float32(p["dt_microp"]),
            p["surface"]["rainnc"],p["precip"]["rainnc"],rate["prate_max"],
            rate["rolling"] if rate["rolling"] is not None else np.uint64(0),
            *(rate[n] for n in ("ave_prate_1min","ave_prate_5min","ave_prate_10min")),
            *(np.int32(n) for n in rate["limits"]),*(np.float32(t) for t in rate["times"])))
    # Without radii (radii_due) the kernel left re_* alone: the pool keeps the radii its readers last needed.
    for name,value in p.get("profile_targets",{}).items():
        if name=="refl10cm" or p.get("radii",True):cp.copyto(value,p["profiles"][name])
    run("tempo_to",cells,(m,mesh,d,t,f)+extent+af+(np.float32(dt_dyn),)+tuple(np.float32(c[n]) for n in CONSTANT_ARGUMENTS)+(
        np.int32(p.get("mp_top_level",nz)),p.get("compref_max",np.uint64(0)),
        p["diagnostics"]["refl10cm_max"]))
    # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:1400-1410: diagnostics copied on return.
    for name in SURFACE_ARGUMENTS[8:10]:cp.copyto(p["precip"][name],p["surface"][name])


def register(registry):
    """Bind slot only; RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:592-646."""
    registry.register("microphysics",SCHEME,BACKEND,run_slot)


def pack_initialized_resources(native, rust_transport, blob_path, *, table_files,
                               device_math, flags, first_call_main):
    """Snapshot pristine initialized tables using Rust kind promotion/encoding.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:23-857;
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182. The session supplies the resolved flags
    and native SAVE-state witness. This helper alone certifies no forecast.
    """
    import ctypes as ct
    import hashlib
    import json
    import math
    from pathlib import Path
    def seal(filename):
        """Hash inputs to MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182."""
        filename=Path(filename).resolve()
        with filename.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        return {"path":str(filename),"sha256":digest}
    if native.build.get("source_pin") not in TEMPO_FORK_PINS:
        raise ValueError("TEMPO table snapshot requires the pinned fork native session")
    if len(flags)!=4 or any(type(flag) not in (int,bool) or flag not in (0,1) for flag in flags):
        raise ValueError("four resolved native TEMPO flags required")
    blob_path=Path(blob_path).resolve()
    if blob_path.exists():raise FileExistsError(blob_path)
    transport=ct.CDLL(str(Path(rust_transport).resolve()))
    transport.tempo_append_table.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_int]
    transport.tempo_append_table.restype=ct.c_int
    kinds={np.dtype("float32"):1,np.dtype("float64"):2,np.dtype("int32"):3,np.dtype("int64"):4}
    bounds={};unallocated=[]
    for index,(name,shape) in enumerate(TABLE_LAYOUT):
        item=native.get_table("module_mp_tempo_params."+name)
        if item is None:
            # MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:153-161 leaves these absent when melting is disabled.
            # Reserve their unused blob slots; no native value is substituted.
            if flags[3] or name not in ("sbins_radar","dsbins_radar","gbins_radar","dgbins_radar"):
                raise ValueError("native TEMPO table is not initialized: "+name)
            status=transport.tempo_append_table(str(blob_path).encode(),None,
                        math.prod(shape),0,int(index==0))
            if status:raise OSError("Rust TEMPO inactive table padding failed: "+name)
            bounds[name]=None;unallocated.append(name)
            continue
        value,lower=item
        if value.dtype not in kinds or not value.flags.f_contiguous or value.shape+(1,)*(5-value.ndim)!=shape:
            raise ValueError("native TEMPO table kind/extent differs: "+name)
        status=transport.tempo_append_table(str(blob_path).encode(),value.ctypes.data,
                    math.prod(shape),kinds[value.dtype],int(index==0))
        if status:raise OSError("Rust TEMPO table encoding failed: "+name)
        bounds[name]=list(lower)
    packet={"fork":native.build["source_pin"],"tempo":"9adf9ef2827104640125f0c322e1218fce74e94c",
        "rkind":4,"layout_sha256":hashlib.sha256(json.dumps(TABLE_LAYOUT,separators=(",",":")).encode()).hexdigest(),
        "native_library":seal(native.path),"table_files":[seal(path) for path in table_files],
        "native_build_recipe":seal(Path(native.path).with_name("build-receipt.json")),
        "native_abi":seal(Path(native.path).with_name("abi.json")),
        "blob":seal(blob_path),"rust_transport":seal(rust_transport),"device_math":seal(device_math),
        "flags":list(flags),"first_call_main":bool(first_call_main),"native_table_lower_bounds":bounds,
        "native_unallocated_tables":unallocated,
        "native_pin_set_sha256":native.build["pin_set_sha256"],"forecast_proven":False}
    return packet


def load_resources(packet, device):
    """Load hashed pristine initialization through a Rust binary transport.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182,455-633 and
    MPAS-Model-rrfs/src/core_atmosphere/utils/build_tables_tempo.F:1-22.
    The packet records the native library and the table files that produced
    its promoted blob. It must be regenerated when the native flags change.
    This loader performs no table generation or numerical decoding in Python.
    """
    import ctypes
    import hashlib
    import json
    import math
    from pathlib import Path
    def verify(row):
        """Authenticate inputs to MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:455-633."""
        path=Path(row["path"])
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
        if digest.hexdigest()!=row["sha256"]:
            raise ValueError(f"native TEMPO resource bytes changed: {path}")
        return path
    if packet.get("session_tables"):
        # Install packet (woof hex physics-resources build): the certified
        # device maths only. The tables come from this forecast session's own
        # native TEMPO initialization at the first bind (_session_tables).
        maths=verify(packet["device_math"]).read_text()
        return {"tables":None,"device_math_source":maths,"flags":None,
                "first_call_main":True,"provenance":packet}
    if packet["fork"] not in TEMPO_FORK_PINS or packet["tempo"]!="9adf9ef2827104640125f0c322e1218fce74e94c" or packet["rkind"]!=4:
        raise ValueError("TEMPO resource pin/precision differs from this module")
    layout=hashlib.sha256(json.dumps(TABLE_LAYOUT,separators=(",",":")).encode()).hexdigest()
    if packet["layout_sha256"]!=layout:
        raise ValueError("native TEMPO resource layout differs from this module")
    native=verify(packet["native_library"])
    for name in ("native_build_recipe","native_abi"):
        if name in packet:verify(packet[name])
    for row in packet["table_files"]:verify(row)
    blob=verify(packet["blob"])
    transport_path=verify(packet["rust_transport"])
    maths=verify(packet["device_math"]).read_text()
    count=sum(math.prod(shape) for _,shape in TABLE_LAYOUT)
    transport=ctypes.CDLL(str(transport_path))
    transport.tempo_read_f64.argtypes=[ctypes.c_char_p,ctypes.c_size_t]
    transport.tempo_read_f64.restype=ctypes.POINTER(ctypes.c_double)
    transport.tempo_free_f64.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
    ptr=transport.tempo_read_f64(str(blob).encode(),count)
    if not ptr:raise ValueError("Rust TEMPO transport rejected the native blob extent")
    try:
        tables=device.asarray(np.ctypeslib.as_array(ptr,shape=(count,)))
        device.cuda.get_current_stream().synchronize()
    finally:transport.tempo_free_f64(ptr,count)
    return {"tables":tables,"device_math_source":maths,"flags":tuple(packet["flags"]),
            "first_call_main":bool(packet["first_call_main"]),"provenance":packet}


def _session_tables(native,flags,cp):
    """The session's native-initialized TEMPO tables as the device binary64 blob.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_params.F90:23-857;
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:42-182.
    The same native library already ran tempo_init for this run. Each public
    table is read in its native kind and memory order and promoted to binary64
    on the device: the layout pack_initialized_resources writes. Tables native
    leaves unallocated when melting is off keep reserved zero slots.
    """
    import math
    if native.build.get("source_pin") not in TEMPO_FORK_PINS:
        raise ValueError("TEMPO session tables require the pinned fork native session")
    kinds=(np.dtype("float32"),np.dtype("float64"),np.dtype("int32"),np.dtype("int64"))
    total=sum(math.prod(shape) for _,shape in TABLE_LAYOUT)
    tables=cp.zeros(total,dtype=cp.float64);offset=0
    for name,shape in TABLE_LAYOUT:
        count=math.prod(shape)
        item=native.get_table("module_mp_tempo_params."+name)
        if item is None:
            # MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:153-161 leaves these absent when melting is disabled.
            if flags[3] or name not in ("sbins_radar","dsbins_radar","gbins_radar","dgbins_radar"):
                raise ValueError("native TEMPO table is not initialized: "+name)
        else:
            value,_=item
            if value.dtype not in kinds or not value.flags.f_contiguous or value.shape+(1,)*(5-value.ndim)!=shape:
                raise ValueError("native TEMPO table kind/extent differs: "+name)
            tables[offset:offset+count]=cp.asarray(value.reshape(-1,order="F")).astype(cp.float64)
        offset+=count
    cp.cuda.get_current_stream().synchronize()
    return tables


def _hourly_switch(value):
    """The do_diag_dbz switch as launch_mpas takes it: the device int32 word stays on the device (tempo_radar reads
    it), anything else is a host value.  RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:553-559."""
    return value if hasattr(value,"__cuda_array_interface__") else bool(value)


def run_slot(call):
    """Fork driver callback; RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:568-825."""
    if call.slot!="microphysics" or call.scheme!=SCHEME or str(call.backend)!=BACKEND:
        raise ValueError("TEMPO requires its registered microphysics slot")
    p=call.pools["tempo_v303"]
    try:
        flags=tuple(bool(call.options[n]) for n in CONFIG_ARGUMENTS)
        result=launch_mpas(p,dt_dyn=call.dt,flags=flags,
            bucket=call.options["config_bucket_rainnc"],diagnostics=call.flags.diagnostics,
            hourly=_hourly_switch(p["do_diag_dbz_flag"]))
    except BaseException:
        _finish_native_workspace(p,commit=False)
        raise
    _finish_native_workspace(p,commit=True)
    return result


def _allocated(native,name):
    """Whether native allocated module array ``name``, read from its storage record without copying it.

    THE COST THIS REMOVES: get_variable returns a copy of the whole array (on the resident facade a device
    allocation and two copy kernels), and the bind asked it six times per call only to test for None.
    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:124-146.
    """
    storage=getattr(native,"variable_storage",None)
    return (storage(name) if storage is not None else native.get_variable(name)) is not None


def bind_native_pools(provider,call):
    """Own absent workspace for the complete fork microphysics transaction.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:561-565,821-825.
    Allocation supplied by a native caller retains that caller's lifecycle.
    """
    if call.slot!="microphysics" or call.scheme!=SCHEME or str(call.backend)!=BACKEND:
        raise ValueError("TEMPO requires its registered microphysics slot")
    owned=not _allocated(provider.native,"qv_p")
    if owned:provider.native.call("allocate_microphysics",configs="configs")
    try:
        bound=_bind_native_views(provider,call)
    except BaseException:
        if owned:provider.native.call("deallocate_microphysics",configs="configs")
        raise
    if owned:bound["tempo_v303"]["_native_workspace"]=provider
    return bound


def _finish_native_workspace(p,*,commit):
    """Commit pool words before releasing callback-owned native storage.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:810-825.
    The shared adapter's subsequent commit has no remaining buffered words.
    """
    provider=p.pop("_native_workspace",None)
    if provider is not None:
        try:
            if commit:provider.commit()
        finally:
            provider.native.call("deallocate_microphysics",configs="configs")


def _device_dbz_word(provider):
    """The resident device word of diag_physics.do_diag_dbz (int32, one word), or None off the resident route.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:553-559."""
    cp=provider.cp
    try:
        word=provider["diag_physics"]["do_diag_dbz"]
    except (KeyError,TypeError,ValueError):
        return None
    if not isinstance(word,cp.ndarray) or word.dtype!=cp.int32 or word.size!=1:
        return None
    return word.reshape(())


def radii_due(provider,call):
    """Whether this call's effective radii (re_cloud, re_ice, re_snow, the PBL subgrid radii included) have a reader.

    Their readers: radiation, which reads re_* only on its own steps, at the start of the step, so after this
    call only when the NEXT step is a radiation step (RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/
    mpas_atmphys_driver.F:234-260; mpas_atmphys_driver_radiation_lw.F radiation_lw_from_MPAS), and whatever
    reads the pools at the end of THIS step (a due output stream: restart and da_state carry re_*; the last
    step; moving and cycling publications), which the runtime decides before the step
    (runtime_v842.physics_pool_readers_due, as session.pool_readers_after_step). Any other call skips them: the
    radii are computed afresh from the state each time and no process reads them, so every reader sees the
    words native computed (native computes them every call, RRFS mp_tempo 9adf9ef main:801-823).
    THE COST THIS REMOVES: at dt 20 s and 15-min radiation the radii were computed on 45 calls for 1 read,
    6.4 % of the TEMPO slot at 3 km (effective 2.1 %, PBL subgrid 4.3 %; TEMPO-3KM-PROFILE.md).
    Without the runtime's word for this very step (another runner, a replay) every call computes them.
    """
    session=getattr(provider,"session",None)
    readers=getattr(session,"pool_readers_after_step",None)
    if readers is None or readers[0]!=call.itimestep or readers[1]:
        return True
    driver=session.driver;manager,config=driver.manager,driver.config
    if manager.last_time is None:
        return True
    for slot in ("radiation_lw","radiation_sw"):
        if config.active(slot):
            alarm=manager.alarms.get(slot)
            # No alarm: radiation runs every step.
            if alarm is None or alarm.ringing(manager.last_time,manager.dt):
                return True
    return False


def _bind_native_views(provider,call):
    """Pool binder; RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:699-1110,1195-1616.

    The session supplies native initialized, Rust-packed ``tempo_v303_resources``.
    All persistent module and pool storage stays owned by the forecast session.
    """
    if provider.level!=2:raise ValueError("MPAS microphysics updates state time level 2")
    resource=provider.tempo_v303_resources;flags=tuple(bool(call.options[n]) for n in CONFIG_ARGUMENTS)
    if resource["tables"] is None:
        resource["tables"]=_session_tables(provider.native,flags,provider.cp)
        resource["flags"]=flags
    if tuple(map(bool,resource["flags"]))!=flags:
        raise ValueError("native initialized TEMPO parameters differ from the live flags")
    # Fast numerics: the binary32 blob, converted once for the run; it replaces the binary64 one (half the VRAM).
    if _fast() and resource["tables"].dtype!=provider.cp.float32:
        resource["tables"]=fast_tables(resource["tables"])
    state=provider["state"];scalars=state["scalars"];model={}
    names=MODEL_ARGUMENTS[:8]+(MODEL_ARGUMENTS[8:11] if flags[0] else ())+ (MODEL_ARGUMENTS[11:13] if flags[1] else ())
    for name in names:
        index=provider.session.layout.indices["state","index_"+name]
        if not 1<=index<=scalars.shape[0]:raise ValueError("native TEMPO scalar index out of range")
        model[name]=scalars[index-1]
    model.update({n:state[n] for n in MODEL_ARGUMENTS[13:]})
    nz=model["qv"].shape[0];prepared=provider["prepared"]
    aliases={"p":"pres_p","pii":"pi_p","qb":"volg_p"}
    enabled=FIELD_ARGUMENTS[:12]+FIELD_ARGUMENTS[17:]
    # Native allocation controls OPTIONAL presence; the TEMPO flag does not.
    # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:124-146; TEMPO MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/module_mp_tempo_driver.F90:284-303,337-348.
    enabled+=tuple(n for n in FIELD_ARGUMENTS[12:17]
                   if _allocated(provider.native,aliases.get(n,n+"_p")))
    fields={n:prepared[aliases.get(n,n+"_p")][:nz] for n in enabled}
    surface={n:prepared[n+"_p"] for n in SURFACE_ARGUMENTS[:10]}
    if flags[0]:surface.update({n:prepared[n+"_p"] for n in SURFACE_ARGUMENTS[10:]})
    diag={n:provider["diag"][n] for n in DIAG_ARGUMENTS if n not in ("rt_diabatic_tend","tend_sfc_pressure")}
    diag.update({n:provider["tend"][n] for n in ("rt_diabatic_tend","tend_sfc_pressure")})
    diag.update({n:provider["diag_physics"][n] for n in RADAR_ARGUMENTS});diag["relhum"]=provider["diag"]["relhum"]
    tendency_names=TEND_ARGUMENTS if flags[0] else TEND_ARGUMENTS[:9]
    # This logical is local to the native driver, not a module global.
    # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:553-559.
    # On the resident route the device pool word is the authority: hand it to tempo_radar as is. THE BREAKAGE
    # THIS ENDS: bool(get_field(...)[()]==1) copied the word to the host every step, so the host waited for the
    # GPU to drain before queueing the rest of the step (2.8 ms host wait per step, hrrrv5 x1.2562 profile p2).
    word=_device_dbz_word(provider)
    do_diag_dbz_flag=word if word is not None else bool(provider.native.get_field("diag_physics","do_diag_dbz")[()]==1)
    p=dict(model=model,fields=fields,surface=surface,mesh={n:provider["mesh"][n] for n in ("zz","zgrid")},diagnostics=diag,
        tendencies={n:provider["tend_physics"][n] for n in tendency_names},constants={n:provider["constants"][n] for n in CONSTANT_ARGUMENTS},
        precip={n:provider["diag_physics"][n] for n in (*PRECIP_ARGUMENTS,"i_rainnc")},
        profiles={n:prepared[{"re_cloud":"recloud","re_ice":"reice","re_snow":"resnow"}.get(n,n)+"_p"][:nz] for n in PROFILE_ARGUMENTS},
        tables=resource["tables"],device_math_source=resource["device_math_source"],first_call_main=resource["first_call_main"],
        dt_microp=provider["scalar_state"]["dt_microp"],n_microp=int(provider["scalar_state"]["n_microp"]),
        l_acrain=bool(provider["scalar_state"]["l_acrain"]),do_diag_dbz_flag=do_diag_dbz_flag)
    p["radii"]=radii_due(provider,call)
    # Pack the two PBL carriers with a CUDA copy, with no host numerical processing. Only the PBL subgrid
    # radii read them, so a call without radii does not pack them.
    if flags[2] and p["radii"]:
        p["pbl"]=provider.cp.stack((provider["diag_physics"]["qc_bl"],provider["diag_physics"]["cldfrac_bl"]))
    p["surface_inputs"]={n:provider["diag_physics"][n] for n in SURFACE_ARGUMENTS[8:10]+(SURFACE_ARGUMENTS[10:] if flags[0] else ())}
    p["profile_targets"]={n:provider["diag_physics"][n] for n in PROFILE_ARGUMENTS}
    if "config_microphysics_top" in call.options:
        # Scalar prepared at physics_init; host framework storage is authoritative
        # for this model-wide metadata, so no per-step device scalar read occurs.
        if "mp_top_level" not in resource:
            host=getattr(provider.native,"_host",provider.native)
            resource["mp_top_level"]=int(host.get_field("tend_physics","mp_top_level"))
        p["mp_top_level"]=resource["mp_top_level"]
        p["compref_max"]=provider["diag_physics"]["refl10cm_compref_max"]
    if call.options.get("calc_prate_max",False):
        if "rolling_prate" not in resource:
            dt=np.float32(p["dt_microp"])
            counts=np.asarray((60,300,600),dtype=np.float32)/dt
            # Native FRACTION is the binary significand, not the fractional part.
            fractional=np.abs(np.frexp(counts)[0])>np.float32(1.e-6)
            limits=np.where(fractional,np.floor(counts+np.float32(.5)),np.trunc(counts)).astype(np.int32)+1
            times=np.where(fractional,dt*(limits-1).astype(np.float32),np.asarray((60,300,600),dtype=np.float32))
            longest=int(limits.max())
            rolling=None if np.any(limits==1) else provider.cp.zeros((longest,model["qv"].shape[1]),dtype=provider.cp.float32)
            resource["rolling_prate"]=dict(rolling=rolling,longest=longest,limits=limits,times=times)
        p["prate"]={**resource["rolling_prate"],**{n:provider["diag_physics"][n] for n in
            ("prate_max","ave_prate_1min","ave_prate_5min","ave_prate_10min")}}
    # first_call_main is shared SAVE state; update it through the same resource mapping.
    p["resources"]=resource
    return {"tempo_v303":p}


def initialize_aerosols(nwfa,nifa,nwfa2d,nifa2d,zgrid,area_cell,tables,*,
                        global_nwfa_extrema,global_nifa_extrema,do_restart,
                        device_math_source):
    """Native initialization using framework MPI extrema; no numerical host reduction.

    RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_init_tempo.F:30-187.
    The native framework supplies global MIN/MAX over solve cells, excluding halos.
    Existing GOCART distributions remain intact. Restart skips initialization.
    """
    if do_restart:return
    import cupy as cp
    nz,cells=nwfa.shape;device=nwfa.device.id
    _pointer_array((nwfa,nifa),shape=(nz,cells),device=device)
    _pointer_array((nwfa2d,nifa2d,area_cell),shape=(cells,),device=device)
    _pointer_array((zgrid,),shape=(nz+1,cells),device=device)
    module,(_,_,blob_bytes)=_compiled(device,device_math_source,nz)
    tables=_device_tables(module,tables,blob_bytes)
    initw=all(np.float32(v)==np.float32(0) for v in global_nwfa_extrema)
    initi=all(np.float32(v)==np.float32(0) for v in global_nifa_extrema)
    if len(global_nwfa_extrema)!=2 or len(global_nifa_extrema)!=2:
        raise ValueError("native MPI min/max pair required")
    module.get_function("tempo_init_aerosols")(((cells+31)//32,),(32,),(
        nwfa,nifa,nwfa2d,nifa2d,zgrid,area_cell,np.int32(nz),np.int32(cells),
        np.int32(initw),np.int32(initi),tables))


def reset_output_diagnostics(diagnostics):
    """Reset after stream output; RRFS MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:975-1003.

    The common output owner invokes this after writing the native diagnostic
    stream. Association is represented by the presence of a device pool view.
    """
    for name in ("refl10cm_1km_max", "max_hail_diameter_sfc_acc",
                 "max_hail_diameter_column_acc"):
        array = diagnostics.get(name)
        if array is not None:
            array.fill(np.float32(0.0))
