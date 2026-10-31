"""Pinned RRFS-MPAS MYNN-SFC v6.0.0, complete MPAS-linked CUDA scheme.

RRFS 1d6e370e, MYNN-SFC b66187f. Each emitted routine cites its source.
The MPAS Makefile links the modular land/water/ice implementation.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
from importlib.resources import files
import numpy as np

SCHEME = 'sf_mynnsfclay'
BACKEND = 'hex-cuda'
FIELDS = ('u3d2', 'v3d2', 'dz8w2', 'u3d', 'v3d', 't3d', 'qv3d', 'p3d', 'dz8w', 'th3d', 'rho3d', 'sigmaf', 'shdmax', 'z0pert', 'ztpert', 'psfcpa', 'chs', 'chs2', 'cqs', 'cqs2', 'cpm', 'znt', 'ust', 'ustm', 'pblh', 'mavail', 'zol', 'mol', 'rmol', 'psim', 'psih', 'xland', 'qgh', 'hfx', 'qfx', 'lh', 'tsk', 'flhc', 'flqc', 'qsfc', 'u10', 'v10', 'th2', 't2', 'lakemask', 'q2', 'snowh', 'gz1oz0', 'wspd', 'br', 'dx', 'ch', 'ck', 'cka', 'cd', 'cda', 'stress', 'hflx', 'qflx', 'cm', 'fm', 'fh', 'fm10', 'fh2', 'tsurf', 'water_depth', 'xice', 'pattern_spp_pbl')
OPTIONS = ('ivegsrc', 'redrag', 'sfc_z0_type', 'spp_pbl', 'sf_mynn_sfcflux_water', 'sf_mynn_sfcflux_land', 'shalwater_z0', 'isfflx', 'restart', 'cycling', 'initflag', 'flag_iter', 'flagc_lsm', 'itimestep')
OUTPUTS = ('chs', 'chs2', 'cqs', 'cqs2', 'cpm', 'znt', 'ust', 'ustm', 'pblh', 'zol', 'mol', 'rmol', 'psim', 'psih', 'qgh', 'hfx', 'qfx', 'lh', 'flhc', 'flqc', 'qsfc', 'u10', 'v10', 'th2', 't2', 'q2', 'gz1oz0', 'wspd', 'br', 'ch', 'ck', 'cka', 'cd', 'cda', 'stress', 'hflx', 'qflx', 'cm', 'fm', 'fh', 'fm10', 'fh2')
OPTIONAL = ('restart', 'cycling', 'initflag', 'sf_mynn_sfcflux_water', 'sf_mynn_sfcflux_land', 'shalwater_z0', 'flagc_lsm', 'spp_pbl', 'ivegsrc', 'sfc_z0_type', 'redrag', 'flag_iter', 'vegtype', 'sigmaf', 'shdmax', 'z0pert', 'ztpert', 'pattern_spp_pbl', 'tsurf', 'lakemask', 'water_depth', 'cm', 'ck', 'cka', 'cd', 'cda', 'ustm', 'fm', 'fh', 'fm10', 'fh2', 'hflx', 'qflx', 'stress')
MPAS_PRESENT = 130023647
SOURCE = r"""
__device__ float psim_stab[1001];
__device__ float psim_unstab[1001];
__device__ float psih_stab[1001];
__device__ float psih_unstab[1001];
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:120-903 @1d6e370e prototype land_mynnsfc_land
__device__ __noinline__ void m_land_mynnsfc_land(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool redrag, int sfc_z0_type, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &qgh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &qsfc, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int sf_mynn_sfcflux_land, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_sf_mynn_sfcflux_land, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:919-954 @1d6e370e prototype land_zilitinkevich_1995
__device__ __noinline__ void m_land_zilitinkevich_1995(float z_0, float &zt, float &zq, float restar, float ustar, float karman, int zt_opt, int lsm, int lsm_ruc, bool has_zt_opt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:963-984 @1d6e370e prototype land_garratt_1992
__device__ __noinline__ void m_land_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1010-1041 @1d6e370e prototype land_yang_2008
__device__ __noinline__ void m_land_yang_2008(float z_0, float &zt, float &zq, float ustar, float tstar, float qst, float ren, float visc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1045-1096 @1d6e370e prototype land_gfs_z0_land
__device__ __noinline__ void m_land_gfs_z0_land(float &z0max, float shdmax, float z1, int vegtype, int ivegsrc, float z0pert);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1100-1124 @1d6e370e prototype land_gfs_zt_land
__device__ __noinline__ void m_land_gfs_zt_land(float &ztmax, float z0max, float sigmaf, float ztpert, float ustar);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1129-1158 @1d6e370e prototype land_psi_hogstrom_1996
__device__ __noinline__ void m_land_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1165-1195 @1d6e370e prototype land_psi_dyerhicks
__device__ __noinline__ void m_land_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1201-1224 @1d6e370e prototype land_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_land_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1231-1253 @1d6e370e prototype land_psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_land_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1258-1283 @1d6e370e prototype land_psi_businger_1971
__device__ __noinline__ void m_land_psi_businger_1971(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1289-1300 @1d6e370e prototype land_psi_cb2005
__device__ __noinline__ void m_land_psi_cb2005(float &psim1, float &psih1, float zl, float z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1308-1364 @1d6e370e prototype land_li_etal_2010
__device__ __noinline__ void m_land_li_etal_2010(float &zl, float rib, float zaz0, float z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1367-1420 @1d6e370e prototype land_zolri
__device__ __noinline__ float m_land_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1422-1462 @1d6e370e prototype land_zolri2
__device__ __noinline__ float m_land_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1465-1539 @1d6e370e prototype land_zolrib
__device__ __noinline__ float m_land_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1544-1551 @1d6e370e prototype land_psim_stable_full
__device__ __noinline__ float m_land_psim_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1554-1561 @1d6e370e prototype land_psih_stable_full
__device__ __noinline__ float m_land_psih_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1564-1579 @1d6e370e prototype land_psim_unstable_full
__device__ __noinline__ float m_land_psim_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1582-1596 @1d6e370e prototype land_psih_unstable_full
__device__ __noinline__ float m_land_psih_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1603-1612 @1d6e370e prototype land_psim_stable_full_gfs
__device__ __noinline__ float m_land_psim_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1616-1625 @1d6e370e prototype land_psih_stable_full_gfs
__device__ __noinline__ float m_land_psih_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1629-1645 @1d6e370e prototype land_psim_unstable_full_gfs
__device__ __noinline__ float m_land_psim_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1649-1665 @1d6e370e prototype land_psih_unstable_full_gfs
__device__ __noinline__ float m_land_psih_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1669-1686 @1d6e370e prototype land_psim_stable
__device__ __noinline__ float m_land_psim_stable(float zolf, int psi_opt, const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1689-1706 @1d6e370e prototype land_psih_stable
__device__ __noinline__ float m_land_psih_stable(float zolf, int psi_opt, const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1709-1726 @1d6e370e prototype land_psim_unstable
__device__ __noinline__ float m_land_psim_unstable(float zolf, int psi_opt, const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1729-1746 @1d6e370e prototype land_psih_unstable
__device__ __noinline__ float m_land_psih_unstable(float zolf, int psi_opt, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:122-919 @1d6e370e prototype water_mynnsfc_water
__device__ __noinline__ void m_water_mynnsfc_water(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float lakemask, float wat_depth, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool redrag, int sfc_z0_type, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &qsfc, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &qgh, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int sf_mynn_sfcflux_water, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int shalwater_z0, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_sf_mynn_sfcflux_water, bool has_shalwater_z0, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:924-950 @1d6e370e prototype water_davis_etal_2008
__device__ __noinline__ void m_water_davis_etal_2008(float &z_0, float ustar);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:955-974 @1d6e370e prototype water_taylor_yelland_2001
__device__ __noinline__ void m_water_taylor_yelland_2001(float &z_0, float ustar, float wsp10);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:979-995 @1d6e370e prototype water_coare30_znt
__device__ __noinline__ void m_water_coare30_znt(float &z_0, float ustar, float wsp10, float visc, float zu);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1002-1020 @1d6e370e prototype water_edson_etal_2013
__device__ __noinline__ void m_water_edson_etal_2013(float &z_0, float ustar, float wsp10, float visc, float zu);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1027-1051 @1d6e370e prototype water_depth_dependent_z0
__device__ __noinline__ float m_water_depth_dependent_z0(float water_depth, float z0, float ust);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1061-1082 @1d6e370e prototype water_garratt_1992
__device__ __noinline__ void m_water_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1093-1127 @1d6e370e prototype water_fairall_etal_2003
__device__ __noinline__ void m_water_fairall_etal_2003(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, int spp_sfc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1135-1157 @1d6e370e prototype water_fairall_etal_2014
__device__ __noinline__ void m_water_fairall_etal_2014(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, int spp_sfc);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1160-1209 @1d6e370e prototype water_gfs_z0_wat
__device__ __noinline__ void m_water_gfs_z0_wat(float &z0rl, float &ustar, float wspd, float z1, int sfc_z0_type, bool redrag);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1212-1257 @1d6e370e prototype water_gfs_zt_wat
__device__ __noinline__ void m_water_gfs_zt_wat(float &ztmax, float z0rl, float restar, float wspd, float z1, int sfc_z0_type, int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1262-1304 @1d6e370e prototype water_znot_m_v6
__device__ __noinline__ void m_water_znot_m_v6(float uref, float &znotm);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1315-1368 @1d6e370e prototype water_znot_t_v6
__device__ __noinline__ void m_water_znot_t_v6(float uref, float &znott);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1378-1415 @1d6e370e prototype water_znot_m_v7
__device__ __noinline__ void m_water_znot_m_v7(float uref, float &znotm);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1424-1478 @1d6e370e prototype water_znot_t_v7
__device__ __noinline__ void m_water_znot_t_v7(float uref, float &znott);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1484-1513 @1d6e370e prototype water_psi_hogstrom_1996
__device__ __noinline__ void m_water_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1520-1550 @1d6e370e prototype water_psi_dyerhicks
__device__ __noinline__ void m_water_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1556-1579 @1d6e370e prototype water_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_water_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1584-1609 @1d6e370e prototype water_psi_businger_1971
__device__ __noinline__ void m_water_psi_businger_1971(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1618-1640 @1d6e370e prototype water_psi_suselj_sood_2010
__device__ __noinline__ void m_water_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1646-1657 @1d6e370e prototype water_psi_cb2005
__device__ __noinline__ void m_water_psi_cb2005(float &psim1, float &psih1, float zl, float z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1663-1719 @1d6e370e prototype water_li_etal_2010
__device__ __noinline__ void m_water_li_etal_2010(float &zl, float rib, float zaz0, float z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1722-1775 @1d6e370e prototype water_zolri
__device__ __noinline__ float m_water_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1777-1817 @1d6e370e prototype water_zolri2
__device__ __noinline__ float m_water_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1820-1894 @1d6e370e prototype water_zolrib
__device__ __noinline__ float m_water_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1899-1906 @1d6e370e prototype water_psim_stable_full
__device__ __noinline__ float m_water_psim_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1909-1916 @1d6e370e prototype water_psih_stable_full
__device__ __noinline__ float m_water_psih_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1919-1934 @1d6e370e prototype water_psim_unstable_full
__device__ __noinline__ float m_water_psim_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1937-1951 @1d6e370e prototype water_psih_unstable_full
__device__ __noinline__ float m_water_psih_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1958-1967 @1d6e370e prototype water_psim_stable_full_gfs
__device__ __noinline__ float m_water_psim_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1971-1980 @1d6e370e prototype water_psih_stable_full_gfs
__device__ __noinline__ float m_water_psih_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1984-2000 @1d6e370e prototype water_psim_unstable_full_gfs
__device__ __noinline__ float m_water_psim_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2004-2020 @1d6e370e prototype water_psih_unstable_full_gfs
__device__ __noinline__ float m_water_psih_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2024-2041 @1d6e370e prototype water_psim_stable
__device__ __noinline__ float m_water_psim_stable(float zolf, int psi_opt, const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2044-2061 @1d6e370e prototype water_psih_stable
__device__ __noinline__ float m_water_psih_stable(float zolf, int psi_opt, const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2064-2081 @1d6e370e prototype water_psim_unstable
__device__ __noinline__ float m_water_psim_unstable(float zolf, int psi_opt, const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2084-2101 @1d6e370e prototype water_psih_unstable
__device__ __noinline__ float m_water_psih_unstable(float zolf, int psi_opt, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:116-836 @1d6e370e prototype ice_mynnsfc_ice
__device__ __noinline__ void m_ice_mynnsfc_ice(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &qgh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &qsfc, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_ck, bool has_cka, bool has_cd, bool has_cda);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:846-898 @1d6e370e prototype ice_andreas_2002
__device__ __noinline__ void m_ice_andreas_2002(float z_0, float bvisc, float ustar, float &zt, float &zq, int spp_sfc, float rstoch);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:903-932 @1d6e370e prototype ice_psi_hogstrom_1996
__device__ __noinline__ void m_ice_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:939-969 @1d6e370e prototype ice_psi_dyerhicks
__device__ __noinline__ void m_ice_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:975-998 @1d6e370e prototype ice_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_ice_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1005-1027 @1d6e370e prototype ice_psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_ice_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1032-1057 @1d6e370e prototype ice_psi_businger_1971
__device__ __noinline__ void m_ice_psi_businger_1971(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1066-1088 @1d6e370e prototype ice_psi_suselj_sood_2010
__device__ __noinline__ void m_ice_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1094-1105 @1d6e370e prototype ice_psi_cb2005
__device__ __noinline__ void m_ice_psi_cb2005(float &psim1, float &psih1, float zl, float z0l);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1111-1167 @1d6e370e prototype ice_li_etal_2010
__device__ __noinline__ void m_ice_li_etal_2010(float &zl, float rib, float zaz0, float z0zt);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1170-1223 @1d6e370e prototype ice_zolri
__device__ __noinline__ float m_ice_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1225-1265 @1d6e370e prototype ice_zolri2
__device__ __noinline__ float m_ice_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1268-1342 @1d6e370e prototype ice_zolrib
__device__ __noinline__ float m_ice_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1347-1354 @1d6e370e prototype ice_psim_stable_full
__device__ __noinline__ float m_ice_psim_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1357-1364 @1d6e370e prototype ice_psih_stable_full
__device__ __noinline__ float m_ice_psih_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1367-1382 @1d6e370e prototype ice_psim_unstable_full
__device__ __noinline__ float m_ice_psim_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1385-1399 @1d6e370e prototype ice_psih_unstable_full
__device__ __noinline__ float m_ice_psih_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1406-1415 @1d6e370e prototype ice_psim_stable_full_gfs
__device__ __noinline__ float m_ice_psim_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1419-1428 @1d6e370e prototype ice_psih_stable_full_gfs
__device__ __noinline__ float m_ice_psih_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1432-1448 @1d6e370e prototype ice_psim_unstable_full_gfs
__device__ __noinline__ float m_ice_psim_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1452-1468 @1d6e370e prototype ice_psih_unstable_full_gfs
__device__ __noinline__ float m_ice_psih_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1472-1489 @1d6e370e prototype ice_psim_stable
__device__ __noinline__ float m_ice_psim_stable(float zolf, int psi_opt, const float *psim_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1492-1509 @1d6e370e prototype ice_psih_stable
__device__ __noinline__ float m_ice_psih_stable(float zolf, int psi_opt, const float *psih_stab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1512-1529 @1d6e370e prototype ice_psim_unstable
__device__ __noinline__ float m_ice_psim_unstable(float zolf, int psi_opt, const float *psim_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1532-1549 @1d6e370e prototype ice_psih_unstable
__device__ __noinline__ float m_ice_psih_unstable(float zolf, int psi_opt, const float *psih_unstab);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87 @1d6e370e prototype drv_mynnsfc_init
__device__ __noinline__ void m_drv_mynnsfc_init(bool allowed_to_read, int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:91-101 @1d6e370e prototype drv_mynnsfclay_finalize
__device__ __noinline__ void m_drv_mynnsfclay_finalize(int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740 @1d6e370e prototype drv_mynnsfc_driver
__device__ __noinline__ void m_drv_mynnsfc_driver(float u3d2, float v3d2, float dz8w2, float u3d, float v3d, float t3d, float qv3d, float p3d, float dz8w, float th3d, float rho3d, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool &redrag, int &sfc_z0_type, float psfcpa, float &chs, float &chs2, float &cqs, float &cqs2, float &cpm, float &znt, float &ust, float &ustm, float &pblh, float mavail, float &zol, float &mol, float &rmol, float &psim, float &psih, float xland, float &qgh, float &hfx, float &qfx, float &lh, float tsk, float &flhc, float &flqc, float &qsfc, float &u10, float &v10, float &th2, float &t2, float lakemask, float &q2, float snowh, float &gz1oz0, float &wspd, float &br, float dx, float &ch, float &ck, float &cka, float &cd, float &cda, float &stress, float &hflx, float &qflx, float &cm, float &fm, float &fh, float &fm10, float &fh2, float tsurf, float water_depth, float xice, float xice_threshold, int spp_pbl, float pattern_spp_pbl, int sf_mynn_sfcflux_water, int sf_mynn_sfcflux_land, int shalwater_z0, int isfflx, bool restart, bool cycling, int initflag, bool &flag_iter, int flagc_lsm, int itimestep, int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, int &errflg, bool has_restart, bool has_cycling, bool has_initflag, bool has_sf_mynn_sfcflux_water, bool has_sf_mynn_sfcflux_land, bool has_shalwater_z0, bool has_flagc_lsm, bool has_spp_pbl, bool has_ivegsrc, bool has_sfc_z0_type, bool has_redrag, bool has_flag_iter, bool has_vegtype, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_pattern_spp_pbl, bool has_tsurf, bool has_lakemask, bool has_water_depth, bool has_cm, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, bool has_fm, bool has_fh, bool has_fm10, bool has_fh2, bool has_hflx, bool has_qflx, bool has_stress);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:745-788 @1d6e370e prototype drv_psi_init
__device__ __noinline__ void m_drv_psi_init(int psi_opt, int &errflg);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:794-801 @1d6e370e prototype drv_psim_stable_full
__device__ __noinline__ float m_drv_psim_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:804-811 @1d6e370e prototype drv_psih_stable_full
__device__ __noinline__ float m_drv_psih_stable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:814-829 @1d6e370e prototype drv_psim_unstable_full
__device__ __noinline__ float m_drv_psim_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:832-846 @1d6e370e prototype drv_psih_unstable_full
__device__ __noinline__ float m_drv_psih_unstable_full(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:853-862 @1d6e370e prototype drv_psim_stable_full_gfs
__device__ __noinline__ float m_drv_psim_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:866-875 @1d6e370e prototype drv_psih_stable_full_gfs
__device__ __noinline__ float m_drv_psih_stable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:879-895 @1d6e370e prototype drv_psim_unstable_full_gfs
__device__ __noinline__ float m_drv_psim_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:899-915 @1d6e370e prototype drv_psih_unstable_full_gfs
__device__ __noinline__ float m_drv_psih_unstable_full_gfs(float zolf);
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:120-903 @1d6e370e land_mynnsfc_land
__device__ __noinline__ void m_land_mynnsfc_land(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool redrag, int sfc_z0_type, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &qgh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &qsfc, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int sf_mynn_sfcflux_land, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_sf_mynn_sfcflux_land, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
float xka = 0x1.92a7380000000p-16f;
float prt = 0x1.0000000000000p+0f;
float za;
float za2;
float thv_1;
float th_1;
float tc_1;
float tv_1;
float qvmr;
float psih2;
float psim10;
float psih10;
float wspdi;
float govrth;
float psfc;
float qsfcmr;
float thcon;
float zratio;
float tsk;
float thsk;
float thvsk;
float gz1ozt;
float gz2oz0;
float gz2ozt;
float gz10oz0;
float gz10ozt;
float zntstoch;
float zt;
float zq;
float psiq;
float psiq2;
int n;
int k;
int l;
int yesno;
float pl;
float e1;
float tabs;
float dthvdz;
float dthvm;
float vconv;
float zol2;
float zol10;
float zolza;
float zolz0;
float zolzt;
float dtg;
float dtthx;
float psiq10;
float fluxc;
float vsgd;
float restar;
float visc;
float dqg;
float oldust;
float oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
psfc = __fdiv_rn(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
if ((itimestep==1)) {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
} else {
tabs = __fadd_rn(__fmul_rn(0x1.fae1480000000p-1f,tskin),__fmul_rn(0x1.47ae140000000p-7f,t_1));
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tabs,0x1.1126660000000p+8f)),__fsub_rn(tabs,0x1.da66660000000p+4f))));
qsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
}
} else {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
} else {
tabs = tskin;
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tabs,0x1.1126660000000p+8f)),__fsub_rn(tabs,0x1.da66660000000p+4f))));
qsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
}
}
}
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(t_1,0x1.1126660000000p+8f)),__fsub_rn(t_1,0x1.da66660000000p+4f))));
pl = __fdiv_rn(p_1,0x1.f400000000000p+9f);
qgh = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(pl,e1));
qvmr = __fdiv_rn(qv_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
thcon = SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = __fmul_rn(tsk,thcon);
thvsk = __fmul_rn(thsk,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qsfc)));
}
th_1 = __fmul_rn(t_1,SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = __fsub_rn(t_1,0x1.1126660000000p+8f);
thv_1 = __fmul_rn(th_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
tv_1 = __fmul_rn(t_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
rho_1 = __fdiv_rn(p_1,__fmul_rn(0x1.1f00000000000p+8f,tv_1));
za = __fmul_rn(0x1.0000000000000p-1f,dz8w_1);
za2 = __fadd_rn(dz8w_1,__fmul_rn(0x1.0000000000000p-1f,dz8w_2));
govrth = __fdiv_rn(0x1.39cc100000000p+3f,th_1);
cpm = __fmul_rn(0x1.f640000000000p+9f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1)));
dthvdz = __fsub_rn(thv_1,thvsk);
fluxc = fmaxf(__fadd_rn(__fdiv_rn(__fdiv_rn(hfx,rho_1),0x1.f640000000000p+9f),__fdiv_rn(__fmul_rn(__fmul_rn(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = __fmul_rn(0x1.4000000000000p+0f,SL_POW(__fmul_rn(__fmul_rn(__fdiv_rn(0x1.39cc100000000p+3f,tsk),fminf(__fmul_rn(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.5555560000000p-2f));
vsgd = fminf(__fmul_rn(0x1.47ae140000000p-2f,SL_POW(fmaxf(__fsub_rn(__fdiv_rn(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = SL_SQRT(__fadd_rn(__fadd_rn(__fmul_rn(wspd,wspd),__fmul_rn(wstar,wstar)),__fmul_rn(vsgd,vsgd)));
wspd = fmaxf(wspd,0x1.99999a0000000p-4f);
rb = __fdiv_rn(__fmul_rn(__fmul_rn(govrth,za),dthvdz),__fmul_rn(wspd,wspd));
rb = fmaxf(rb,-0x1.0000000000000p+1f);
rb = fminf(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
}
if (flag_iter) {
visc = __fmul_rn(0x1.bcee880000000p-17f,__fsub_rn(__fadd_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.acbc8c0000000p-8f,tc_1)),__fmul_rn(__fmul_rn(0x1.16890c0000000p-17f,tc_1),tc_1)),__fmul_rn(__fmul_rn(__fmul_rn(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((sf_mynn_sfcflux_land==4)) {
m_land_gfs_z0_land(znt,shdmax,za,vegtype,ivegsrc,z0pert);
}
if ((spp_sfc==1)) {
zntstoch = fmaxf(__fadd_rn(znt,__fmul_rn(__fmul_rn(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
zntstoch = fminf(zntstoch,__fmul_rn(dz8w_1,0x1.10cb2a0000000p-4f));
restar = fmaxf(__fdiv_rn(__fmul_rn(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (has_sf_mynn_sfcflux_land) {
if ((sf_mynn_sfcflux_land<=1)) {
m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,sf_mynn_sfcflux_land,lsm,lsm_ruc,has_sf_mynn_sfcflux_land);
} else if ((sf_mynn_sfcflux_land==2)) {
m_land_yang_2008(zntstoch,zt,zq,ust,mol,qstar,restar,visc);
} else if ((sf_mynn_sfcflux_land==3)) {
m_land_garratt_1992(zt,zq,zntstoch,restar,0x1.0000000000000p+0f);
} else if ((sf_mynn_sfcflux_land==4)) {
m_land_gfs_zt_land(zt,zntstoch,sigmaf,ztpert,ust);
zq = zt;
}
} else {
m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,0,lsm,lsm_ruc,true);
}
if ((spp_sfc==1)) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.8000000000000p-1f),rstoch_1));
zq = __fadd_rn(zq,__fmul_rn(__fmul_rn(zq,0x1.8000000000000p-1f),rstoch_1));
zt = fminf(zt,__fmul_rn(0x1.cccccc0000000p-1f,zntstoch));
zt = fmaxf(zt,0x1.a36e2e0000000p-14f);
zq = fminf(zq,__fmul_rn(0x1.cccccc0000000p-1f,zntstoch));
zq = fmaxf(zq,0x1.a36e2e0000000p-14f);
}
gz1oz0 = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zntstoch));
gz1ozt = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zt));
gz2oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zt));
zratio = __fdiv_rn(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_land_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_land_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_land_psim_stable(zolza,psi_opt,psim_stab),m_land_psim_stable(zolz0,psi_opt,psim_stab));
psih = __fsub_rn(m_land_psih_stable(zolza,psi_opt,psih_stab),m_land_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = __fsub_rn(m_land_psim_stable(zol10,psi_opt,psim_stab),m_land_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = __fsub_rn(m_land_psih_stable(zol10,psi_opt,psih_stab),m_land_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = __fsub_rn(m_land_psih_stable(zol2,psi_opt,psih_stab),m_land_psih_stable(zolzt,psi_opt,psih_stab));
rmol = __fdiv_rn(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_land_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_land_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_land_psim_unstable(zolza,psi_opt,psim_unstab),m_land_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = __fsub_rn(m_land_psih_unstable(zolza,psi_opt,psih_unstab),m_land_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = __fsub_rn(m_land_psim_unstable(zol10,psi_opt,psim_unstab),m_land_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = __fsub_rn(m_land_psih_unstable(zol10,psi_opt,psih_unstab),m_land_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = __fsub_rn(m_land_psih_unstable(zol2,psi_opt,psih_unstab),m_land_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = fminf(psih,__fmul_rn(0x1.cccccc0000000p-1f,gz1ozt));
psim = fminf(psim,__fmul_rn(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = fminf(psih2,__fmul_rn(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = fminf(psim10,__fmul_rn(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = fminf(psih10,__fmul_rn(0x1.cccccc0000000p-1f,gz10ozt));
rmol = __fdiv_rn(zol,za);
}
psix = fmaxf(__fsub_rn(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = fmaxf(__fsub_rn(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = fmaxf(__fsub_rn(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmaxf(__fsub_rn(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(ust,__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,wspd),psix)));
ust = fmaxf(ust,0x1.47ae140000000p-8f);
stress = __fmul_rn(ust,ust);
ustm = ust;
dtg = __fsub_rn(thv_1,thvsk);
oldtst = mol;
mol = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = __fmul_rn(__fsub_rn(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = __fmul_rn(flqc,__fsub_rn(qsfc,qv_1));
qfx = fmaxf(qfx,-0x1.47ae140000000p-6f);
lh = __fmul_rn(0x1.312d000000000p+21f,qfx);
hfx = __fmul_rn(__fdiv_rn(__fmul_rn(__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),__fsub_rn(thsk,th_1));
hfx = fmaxf(hfx,-0x1.f400000000000p+7f);
}
chs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit);
cqs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psiq10));
cd = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psix10));
cka = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psiq));
cda = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
}
}
ch = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psit));
cm = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,psix10),psix);
v10 = __fdiv_rn(__fmul_rn(v_1,psix10),psix);
}
dtg = __fsub_rn(th_1,thsk);
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(0x1.0000000000000p+1f,__fsub_rn(th_1,thsk)),za));
}
t2 = __fmul_rn(th2,SL_POW(__fdiv_rn(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = __fadd_rn(qsfc,__fdiv_rn(__fmul_rn(__fsub_rn(qv_1,qsfc),psiq2),psiq));
q2 = fmaxf(q2,fminf(qsfc,qv_1));
q2 = fminf(q2,__fmul_rn(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==96))&&(j==108)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((__fmul_rn(qsfc,0x1.f400000000000p+9f)<0x0.0p+0f)||(__fmul_rn(qsfc,0x1.f400000000000p+9f)>0x1.4000000000000p+5f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:919-954 @1d6e370e land_zilitinkevich_1995
__device__ __noinline__ void m_land_zilitinkevich_1995(float z_0, float &zt, float &zq, float restar, float ustar, float karman, int zt_opt, int lsm, int lsm_ruc, bool has_zt_opt) {
float czil;
float znt;
if ((zt_opt==1)) {
if ((lsm!=lsm_ruc)) {
czil = SL_POW(0x1.4000000000000p+3f,__fmul_rn(-0x1.99999a0000000p-2f,__fdiv_rn(z_0,0x1.1eb8520000000p-4f)));
znt = z_0;
} else {
czil = __fadd_rn(0x1.47ae140000000p-4f,SL_POW(0x1.4000000000000p+3f,__fmul_rn(-0x1.3333340000000p-1f,__fdiv_rn(__fadd_rn(z_0,0x1.c28f5c0000000p-4f),0x1.eb851e0000000p-5f))));
znt = fminf(0x1.3333340000000p-1f,z_0);
}
} else {
czil = 0x1.851eb80000000p-4f;
znt = z_0;
}
zt = __fmul_rn(znt,SL_EXP(__fmul_rn(__fmul_rn((-karman),czil),SL_SQRT(restar))));
zt = fminf(zt,__fmul_rn(0x1.8000000000000p-1f,znt));
zt = fmaxf(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:963-984 @1d6e370e land_garratt_1992
__device__ __noinline__ void m_land_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea) {
float rq;
float e = 0x1.5bf0a80000000p+1f;
if ((__fsub_rn(landsea,0x1.8000000000000p+0f)>0x0.0p+0f)) {
zt = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.3d70a40000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.23d70a0000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = fminf(zq,0x1.cd5f9a0000000p-15f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
zt = fminf(zt,0x1.cd5f9a0000000p-15f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
} else {
zq = __fdiv_rn(z_0,0x1.d9999a0000000p+2f);
zt = zq;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1010-1041 @1d6e370e land_yang_2008
__device__ __noinline__ void m_land_yang_2008(float z_0, float &zt, float &zq, float ustar, float tstar, float qst, float ren, float visc) {
float ht;
float tstar2;
float qstar2;
float z_02;
float renc2;
float renc = 0x1.2c00000000000p+8f;
float beta = 0x1.8000000000000p+0f;
float m = 0x1.5400000000000p+7f;
float b = 0x1.5980000000000p+9f;
z_02 = fminf(z_0,0x1.0000000000000p-1f);
z_02 = fmaxf(z_02,0x1.47ae140000000p-5f);
renc2 = __fadd_rn(0x1.5980000000000p+9f,__fmul_rn(0x1.5400000000000p+7f,SL_LOG(z_02)));
ht = __fdiv_rn(__fmul_rn(renc2,visc),fmaxf(ustar,0x1.47ae140000000p-7f));
tstar2 = fminf(tstar,-0x1.47ae140000000p-7f);
qstar2 = fminf(qst,-0x1.47ae140000000p-7f);
zt = __fmul_rn(ht,SL_EXP(__fmul_rn(__fmul_rn(-0x1.8000000000000p+0f,SL_SQRT(ustar)),fabsf(tstar2))));
zt = fminf(zt,__fmul_rn(0x1.8000000000000p-1f,z_0));
zt = fmaxf(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1045-1096 @1d6e370e land_gfs_z0_land
__device__ __noinline__ void m_land_gfs_z0_land(float &z0max, float shdmax, float z1, int vegtype, int ivegsrc, float z0pert) {
float tem1;
float tem2;
z0max = fmaxf(0x1.0c6f7a0000000p-20f,fminf(z0max,z1));
tem1 = __fsub_rn(0x1.0000000000000p+0f,shdmax);
tem2 = __fmul_rn(tem1,tem1);
tem1 = __fsub_rn(0x1.0000000000000p+0f,tem2);
if ((ivegsrc==1)) {
if ((vegtype==10)) {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,-0x1.5462a20000000p+1f)));
} else if ((vegtype==6)) {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,-0x1.7f74280000000p+1f)));
} else if ((vegtype==7)) {
z0max = 0x1.47ae140000000p-7f;
} else if ((vegtype==16)) {
z0max = 0x1.47ae140000000p-7f;
} else {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,SL_LOG(z0max))));
}
} else if ((ivegsrc==2)) {
if ((vegtype==7)) {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,-0x1.5462a20000000p+1f)));
} else if ((vegtype==8)) {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,-0x1.7f74280000000p+1f)));
} else if ((vegtype==9)) {
z0max = 0x1.47ae140000000p-7f;
} else if ((vegtype==11)) {
z0max = 0x1.47ae140000000p-7f;
} else {
z0max = SL_EXP(__fadd_rn(__fmul_rn(tem2,-0x1.26bb1c0000000p+2f),__fmul_rn(tem1,SL_LOG(z0max))));
}
}
if ((z0pert!=0x0.0p+0f)) {
z0max = __fmul_rn(z0max,SL_POW(0x1.4000000000000p+3f,z0pert));
}
z0max = fmaxf(z0max,0x1.0c6f7a0000000p-20f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1100-1124 @1d6e370e land_gfs_zt_land
__device__ __noinline__ void m_land_gfs_zt_land(float &ztmax, float z0max, float sigmaf, float ztpert, float ustar) {
float czilc;
float tem1;
float tem2;
float ca = 0x1.99999a0000000p-2f;
czilc = 0x1.99999a0000000p-1f;
tem1 = __fsub_rn(0x1.0000000000000p+0f,sigmaf);
ztmax = __fmul_rn(z0max,SL_EXP(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-tem1),tem1),czilc),0x1.99999a0000000p-2f),SL_SQRT(__fmul_rn(ustar,0x1.4d55560000000p+9f)))));
if ((ztpert!=0x0.0p+0f)) {
ztmax = __fmul_rn(ztmax,SL_POW(0x1.4000000000000p+3f,ztpert));
}
ztmax = fmaxf(ztmax,0x1.0c6f7a0000000p-20f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1129-1158 @1d6e370e land_psi_hogstrom_1996
__device__ __noinline__ void m_land_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.5333340000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.0000000000000p+3f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1165-1195 @1d6e370e land_psi_dyerhicks
__device__ __noinline__ void m_land_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1201-1224 @1d6e370e land_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_land_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl) {
float a = 0x1.0000000000000p+0f;
float b = 0x1.54fdf40000000p-1f;
float c = 0x1.4000000000000p+2f;
float d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-__fadd_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+0f,zl),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-__fsub_rn(__fadd_rn(__fadd_rn(SL_POW(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1231-1253 @1d6e370e land_psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_land_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl) {
float cm = 0x1.8000000000000p+1f;
float ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = __fmul_rn(-0x1.8000000000000p+1f,SL_POW(zl,0x1.aaaaaa0000000p-1f));
psi_h = __fmul_rn(-0x1.4000000000000p+1f,SL_POW(zl,0x1.99999a0000000p-1f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1258-1283 @1d6e370e land_psi_businger_1971
__device__ __noinline__ void m_land_psi_businger_1971(float &psi_m, float &psi_h, float zl) {
float x;
float y;
float pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(SL_LOG(__fmul_rn(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),0x1.0000000000000p+1f))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = __fmul_rn(-0x1.2ccccc0000000p+2f,zl);
psi_h = __fmul_rn(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1289-1300 @1d6e370e land_psi_cb2005
__device__ __noinline__ void m_land_psi_cb2005(float &psim1, float &psih1, float zl, float z0l) {
psim1 = __fsub_rn(__fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),__fmul_rn(0x1.8666660000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = __fsub_rn(__fmul_rn(-0x1.6000000000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),__fmul_rn(0x1.6000000000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1308-1364 @1d6e370e land_li_etal_2010
__device__ __noinline__ void m_land_li_etal_2010(float &zl, float rib, float zaz0, float z0zt) {
float alfa;
float beta;
float zaz02;
float z0zt2;
float au11 = 0x1.70a3d80000000p-5f;
float bu11 = 0x1.89374c0000000p-9f;
float bu12 = 0x1.82a9940000000p-8f;
float bu21 = -0x1.5326180000000p-4f;
float bu22 = 0x1.c4dd300000000p-1f;
float bu31 = 0x1.6425ae0000000p-3f;
float bu32 = -0x1.d7b4a20000000p-1f;
float bu33 = -0x1.b0f27c0000000p-4f;
float aw11 = 0x1.25c91e0000000p-1f;
float aw12 = -0x1.c275260000000p-2f;
float aw21 = -0x1.39a9fc0000000p+2f;
float aw22 = 0x1.a400000000000p+5f;
float bw11 = -0x1.b98c7e0000000p-5f;
float bw12 = 0x1.8a3d700000000p+0f;
float bw21 = -0x1.56872c0000000p-1f;
float bw22 = -0x1.a418940000000p+1f;
float as11 = 0x1.817c1c0000000p-1f;
float as21 = 0x1.de147a0000000p+3f;
float bs11 = 0x1.4154ca0000000p-3f;
float bs21 = -0x1.3c84b60000000p-2f;
float bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = SL_LOG(zaz02);
beta = SL_LOG(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = __fadd_rn(__fmul_rn(__fmul_rn(0x1.70a3d80000000p-5f,alfa),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),__fmul_rn(alfa,alfa)),__fmul_rn(__fadd_rn(__fmul_rn(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),__fadd_rn(__fadd_rn(__fmul_rn(0x1.6425ae0000000p-3f,__fmul_rn(beta,beta)),__fmul_rn(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = fmaxf(zl,-0x1.e000000000000p+3f);
zl = fminf(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),__fadd_rn(__fmul_rn(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),__fadd_rn(__fmul_rn(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = fminf(zl,0x1.0000000000000p+2f);
zl = fmaxf(zl,0x0.0p+0f);
} else {
zl = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),__fmul_rn(0x1.4154ca0000000p-3f,alfa)),__fmul_rn(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = fminf(zl,0x1.4000000000000p+4f);
zl = fmaxf(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1367-1420 @1d6e370e land_zolri
__device__ __noinline__ float m_land_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float x1;
float x2;
float fx1;
float fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = __fsub_rn(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = __fadd_rn(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_land_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_land_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((fabsf(__fsub_rn(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((fabsf(fx2)<fabsf(fx1))) {
x1 = __fsub_rn(x1,__fmul_rn(__fdiv_rn(fx1,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx1 = m_land_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = __fsub_rn(x2,__fmul_rn(__fdiv_rn(fx2,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx2 = m_land_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(x1,x2))>=0x1.47ae140000000p-7f))) {
m_land_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1422-1462 @1d6e370e land_zolri2
__device__ __noinline__ float m_land_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float psim1;
float psih1;
float psix2;
float psit2;
float zolt;
if ((__fmul_rn(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = __fdiv_rn(__fmul_rn(zol2,z0),za);
zol3 = __fadd_rn(zol2,zol20);
zolt = __fdiv_rn(__fmul_rn(zol2,zt),za);
if ((ri2<0)) {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_land_psih_unstable(zol3,psi_opt,psih_unstab),m_land_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_land_psim_unstable(zol3,psi_opt,psim_unstab),m_land_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_land_psih_stable(zol3,psi_opt,psih_stab),m_land_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_land_psim_stable(zol3,psi_opt,psim_stab),m_land_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fsub_rn(__fdiv_rn(__fmul_rn(zol2,psit2),__fmul_rn(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1465-1539 @1d6e370e land_zolrib
__device__ __noinline__ float m_land_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float zolt;
float zolold;
int n;
int nmax = 20;
float psit2;
float psix2;
if ((__fmul_rn(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = __fdiv_rn(__fmul_rn(zolold,z0),za);
zol3 = __fadd_rn(zolold,zol20);
zolt = __fdiv_rn(__fmul_rn(zolold,zt),za);
if ((ri<0)) {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_land_psih_unstable(zol3,psi_opt,psih_unstab),m_land_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_land_psim_unstable(zol3,psi_opt,psim_unstab),m_land_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_land_psih_stable(zol3,psi_opt,psih_stab),m_land_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_land_psim_stable(zol3,psi_opt,psim_stab),m_land_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fdiv_rn(__fmul_rn(ri,__fmul_rn(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f))) {
m_land_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1544-1551 @1d6e370e land_psim_stable_full
__device__ __noinline__ float m_land_psim_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1554-1561 @1d6e370e land_psih_stable_full
__device__ __noinline__ float m_land_psih_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.5333340000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1564-1579 @1d6e370e land_psim_unstable_full
__device__ __noinline__ float m_land_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,x)))),SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
ym = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psimk,__fmul_rn(__fmul_rn(zolf,zolf),psimc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1582-1596 @1d6e370e land_psih_unstable_full
__device__ __noinline__ float m_land_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psihk,__fmul_rn(__fmul_rn(zolf,zolf),psihc)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1603-1612 @1d6e370e land_psim_stable_full_gfs
__device__ __noinline__ float m_land_psim_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float aa;
aa = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,aa),SL_LOG(__fadd_rn(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1616-1625 @1d6e370e land_psih_stable_full_gfs
__device__ __noinline__ float m_land_psih_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float bb;
bb = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,bb),SL_LOG(__fadd_rn(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1629-1645 @1d6e370e land_psim_unstable_full_gfs
__device__ __noinline__ float m_land_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fccccc0000000p+1f,__fmul_rn(0x1.8a3d700000000p+3f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.f051ec0000000p+2f,__fmul_rn(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fsub_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p+1f,SL_SQRT(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1649-1665 @1d6e370e land_psih_unstable_full_gfs
__device__ __noinline__ float m_land_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fc39580000000p+2f,__fmul_rn(0x1.8c00000000000p+4f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.168f5c0000000p+3f,__fmul_rn(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fadd_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1669-1686 @1d6e370e land_psim_stable
__device__ __noinline__ float m_land_psim_stable(float zolf, int psi_opt, const float *psim_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_stab[nzol],__fmul_rn(rzol,__fsub_rn(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psim_stable_full(zolf);
} else {
result = m_land_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1689-1706 @1d6e370e land_psih_stable
__device__ __noinline__ float m_land_psih_stable(float zolf, int psi_opt, const float *psih_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_stab[nzol],__fmul_rn(rzol,__fsub_rn(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psih_stable_full(zolf);
} else {
result = m_land_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1709-1726 @1d6e370e land_psim_unstable
__device__ __noinline__ float m_land_psim_unstable(float zolf, int psi_opt, const float *psim_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psim_unstable_full(zolf);
} else {
result = m_land_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_land.F90:1729-1746 @1d6e370e land_psih_unstable
__device__ __noinline__ float m_land_psih_unstable(float zolf, int psi_opt, const float *psih_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_land_psih_unstable_full(zolf);
} else {
result = m_land_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:122-919 @1d6e370e water_mynnsfc_water
__device__ __noinline__ void m_water_mynnsfc_water(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float lakemask, float wat_depth, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool redrag, int sfc_z0_type, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &qsfc, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &qgh, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int sf_mynn_sfcflux_water, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int shalwater_z0, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_sf_mynn_sfcflux_water, bool has_shalwater_z0, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
float xka = 0x1.92a7380000000p-16f;
float prt = 0x1.0000000000000p+0f;
float za;
float za2;
float thv_1;
float th_1;
float tc_1;
float tv_1;
float qvmr;
float psih2;
float psim10;
float psih10;
float wspdi;
float govrth;
float psfc;
float qsfcmr;
float thcon;
float zratio;
float tsk;
float thsk;
float thvsk;
float gz1ozt;
float gz2oz0;
float gz2ozt;
float gz10oz0;
float gz10ozt;
float zntstoch;
float zt;
float zq;
float psiq;
float psiq2;
int n;
int k;
int l;
int yesno;
float pl;
float e1;
float tabs;
float dthvdz;
float dthvm;
float vconv;
float zol2;
float zol10;
float zolza;
float zolz0;
float zolzt;
float dtg;
float dtthx;
float psiq10;
float fluxc;
float vsgd;
float restar;
float visc;
float dqg;
float oldust;
float oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
psfc = __fdiv_rn(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
tsk = tskin;
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(tsk,0x1.1126660000000p+8f)),__fsub_rn(tsk,0x1.da66660000000p+4f))));
if ((lakemask<0x1.0000000000000p-1f)) e1 = __fmul_rn(e1,0x1.f65fd80000000p-1f);
qsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));
qsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));
}
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(t_1,0x1.1126660000000p+8f)),__fsub_rn(t_1,0x1.da66660000000p+4f))));
pl = __fdiv_rn(p_1,0x1.f400000000000p+9f);
qgh = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(pl,e1));
qvmr = __fdiv_rn(qv_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
thcon = SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = __fmul_rn(tsk,thcon);
thvsk = __fmul_rn(thsk,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qsfc)));
if (((thvsk<0x1.4000000000000p+7f)||(thvsk>0x1.8600000000000p+8f))) {
}
}
th_1 = __fmul_rn(t_1,SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = __fsub_rn(t_1,0x1.1126660000000p+8f);
thv_1 = __fmul_rn(th_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
tv_1 = __fmul_rn(t_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
rho_1 = __fdiv_rn(p_1,__fmul_rn(0x1.1f00000000000p+8f,tv_1));
za = __fmul_rn(0x1.0000000000000p-1f,dz8w_1);
za2 = __fadd_rn(dz8w_1,__fmul_rn(0x1.0000000000000p-1f,dz8w_2));
govrth = __fdiv_rn(0x1.39cc100000000p+3f,th_1);
cpm = __fmul_rn(0x1.f640000000000p+9f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1)));
dthvdz = __fsub_rn(thv_1,thvsk);
fluxc = fmaxf(__fadd_rn(__fdiv_rn(__fdiv_rn(hfx,rho_1),0x1.f640000000000p+9f),__fdiv_rn(__fmul_rn(__fmul_rn(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = __fmul_rn(0x1.4000000000000p+0f,SL_POW(__fmul_rn(__fmul_rn(__fdiv_rn(0x1.39cc100000000p+3f,tsk),pblh),fluxc),0x1.5555560000000p-2f));
vsgd = fminf(__fmul_rn(0x1.0000000000000p-2f,SL_POW(fmaxf(__fsub_rn(__fdiv_rn(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = SL_SQRT(__fadd_rn(__fadd_rn(__fmul_rn(wspd,wspd),__fmul_rn(wstar,wstar)),__fmul_rn(vsgd,vsgd)));
wspd = fmaxf(wspd,0x1.99999a0000000p-4f);
rb = __fdiv_rn(__fmul_rn(__fmul_rn(govrth,za),dthvdz),__fmul_rn(wspd,wspd));
rb = fmaxf(rb,-0x1.0000000000000p+1f);
rb = fminf(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
}
if (flag_iter) {
visc = __fmul_rn(0x1.bcee880000000p-17f,__fsub_rn(__fadd_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.acbc8c0000000p-8f,tc_1)),__fmul_rn(__fmul_rn(0x1.16890c0000000p-17f,tc_1),tc_1)),__fmul_rn(__fmul_rn(__fmul_rn(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((sfc_z0_type>=0)) {
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water==0)) {
m_water_coare30_znt(znt,ust,wspd,visc,za);
} else if ((sf_mynn_sfcflux_water==1)) {
m_water_edson_etal_2013(znt,ust,wspd,visc,za);
} else if (((sf_mynn_sfcflux_water==2)||(sf_mynn_sfcflux_water==3))) {
m_water_davis_etal_2008(znt,ust);
} else if ((sf_mynn_sfcflux_water==4)) {
m_water_taylor_yelland_2001(znt,ust,wspd);
} else if ((sf_mynn_sfcflux_water==5)) {
m_water_gfs_z0_wat(znt,ust,wspd,za,sfc_z0_type,redrag);
}
} else {
m_water_edson_etal_2013(znt,ust,wspd,visc,za);
}
if (has_shalwater_z0) {
if ((shalwater_z0==1)) {
znt = m_water_depth_dependent_z0(wat_depth,znt,ust);
}
}
}
if ((spp_sfc==1)) {
zntstoch = fmaxf(__fadd_rn(znt,__fmul_rn(__fmul_rn(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmaxf(__fdiv_rn(__fmul_rn(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water==0)) {
m_water_fairall_etal_2003(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==1)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==2)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==3)) {
m_water_garratt_1992(zt,zq,zntstoch,restar,0x1.0000000000000p+1f);
} else if ((sf_mynn_sfcflux_water==4)) {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
} else if ((sf_mynn_sfcflux_water==5)) {
m_water_gfs_zt_wat(zt,zntstoch,restar,wspd,za,sfc_z0_type,errflg);
if ((errflg!=0)) return;
zq = zt;
}
} else {
m_water_fairall_etal_2014(zt,zq,restar,ust,visc,rstoch_1,spp_sfc);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
gz1oz0 = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zntstoch));
gz1ozt = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zt));
gz2oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zt));
zratio = __fdiv_rn(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_water_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_water_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_water_psim_stable(zolza,psi_opt,psim_stab),m_water_psim_stable(zolz0,psi_opt,psim_stab));
psih = __fsub_rn(m_water_psih_stable(zolza,psi_opt,psih_stab),m_water_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = __fsub_rn(m_water_psim_stable(zol10,psi_opt,psim_stab),m_water_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = __fsub_rn(m_water_psih_stable(zol10,psi_opt,psih_stab),m_water_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = __fsub_rn(m_water_psih_stable(zol2,psi_opt,psih_stab),m_water_psih_stable(zolzt,psi_opt,psih_stab));
rmol = __fdiv_rn(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_water_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_water_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_water_psim_unstable(zolza,psi_opt,psim_unstab),m_water_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = __fsub_rn(m_water_psih_unstable(zolza,psi_opt,psih_unstab),m_water_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = __fsub_rn(m_water_psim_unstable(zol10,psi_opt,psim_unstab),m_water_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = __fsub_rn(m_water_psih_unstable(zol10,psi_opt,psih_unstab),m_water_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = __fsub_rn(m_water_psih_unstable(zol2,psi_opt,psih_unstab),m_water_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = fminf(psih,__fmul_rn(0x1.cccccc0000000p-1f,gz1ozt));
psim = fminf(psim,__fmul_rn(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = fminf(psih2,__fmul_rn(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = fminf(psim10,__fmul_rn(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = fminf(psih10,__fmul_rn(0x1.cccccc0000000p-1f,gz10ozt));
rmol = __fdiv_rn(zol,za);
}
psix = fmaxf(__fsub_rn(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = fmaxf(__fsub_rn(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = fmaxf(__fsub_rn(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmaxf(__fsub_rn(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = __fadd_rn(__fmul_rn(0x1.0000000000000p-1f,ust),__fdiv_rn(__fmul_rn(0x1.99999a0000000p-3f,wspd),psix));
stress = __fmul_rn(ust,ust);
wspdi = fmaxf(SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1))),0x1.99999a0000000p-4f);
ustm = __fadd_rn(__fmul_rn(0x1.0000000000000p-1f,ustm),__fdiv_rn(__fmul_rn(0x1.99999a0000000p-3f,wspdi),psix));
dtg = __fsub_rn(thv_1,thvsk);
oldtst = mol;
mol = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = __fmul_rn(__fsub_rn(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = __fmul_rn(flqc,__fsub_rn(qsfc,qv_1));
qfx = fmaxf(qfx,-0x1.47ae140000000p-6f);
lh = __fmul_rn(0x1.312d000000000p+21f,qfx);
hfx = __fmul_rn(__fdiv_rn(__fmul_rn(__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),__fsub_rn(thsk,th_1));
if (has_sf_mynn_sfcflux_water) {
if ((sf_mynn_sfcflux_water>1)) {
hfx = __fadd_rn(hfx,__fmul_rn(__fmul_rn(__fmul_rn(rho_1,ustm),ustm),wspdi));
}
}
}
chs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit);
cqs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psiq10));
cd = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psix10));
cka = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psiq));
cda = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
ch = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psit));
cm = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,psix10),psix);
v10 = __fdiv_rn(__fmul_rn(v_1,psix10),psix);
}
dtg = __fsub_rn(th_1,thsk);
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(0x1.0000000000000p+1f,__fsub_rn(th_1,thsk)),za));
}
t2 = __fmul_rn(th2,SL_POW(__fdiv_rn(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = __fadd_rn(qsfc,__fdiv_rn(__fmul_rn(__fsub_rn(qv_1,qsfc),psiq2),psiq));
q2 = fmaxf(q2,fminf(qsfc,qv_1));
q2 = fminf(q2,__fmul_rn(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:924-950 @1d6e370e water_davis_etal_2008
__device__ __noinline__ void m_water_davis_etal_2008(float &z_0, float ustar) {
float zw;
float zn1;
float zn2;
float ozo = 0x1.0ac1fa0000000p-16f;
zw = fminf(SL_POW(__fdiv_rn(ustar,0x1.0f5c280000000p+0f),0x1.3333340000000p-2f),0x1.0000000000000p+0f);
zn1 = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(0x1.6872b00000000p-7f,ustar),ustar),0x1.a1b2560000000p-4f),0x1.0ac1fa0000000p-16f);
zn2 = __fadd_rn(__fmul_rn(0x1.4000000000000p+3f,SL_EXP(__fmul_rn(-0x1.3000000000000p+3f,SL_POW(ustar,-0x1.5555560000000p-2f)))),__fdiv_rn(0x1.baeb220000000p-20f,fmaxf(ustar,0x1.47ae140000000p-7f)));
z_0 = __fadd_rn(__fmul_rn(__fsub_rn(0x1.0000000000000p+0f,zw),zn1),__fmul_rn(zw,zn2));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:955-974 @1d6e370e water_taylor_yelland_2001
__device__ __noinline__ void m_water_taylor_yelland_2001(float &z_0, float ustar, float wsp10) {
float pi = 0x1.921fb60000000p+1f;
float hs;
float tp;
float lp;
hs = __fmul_rn(0x1.9652be0000000p-6f,__fmul_rn(wsp10,wsp10));
tp = __fmul_rn(0x1.753f7c0000000p-1f,fmaxf(wsp10,0x1.99999a0000000p-4f));
lp = __fdiv_rn(__fmul_rn(0x1.39cc100000000p+3f,__fmul_rn(tp,tp)),0x1.921fb60000000p+2f);
z_0 = __fmul_rn(__fmul_rn(0x1.2c00000000000p+10f,hs),SL_POW(__fdiv_rn(hs,lp),0x1.2000000000000p+2f));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:979-995 @1d6e370e water_coare30_znt
__device__ __noinline__ void m_water_coare30_znt(float &z_0, float ustar, float wsp10, float visc, float zu) {
float czo2 = 0x1.6872b00000000p-7f;
float czc;
float wsp10m;
wsp10m = __fdiv_rn(__fmul_rn(wsp10,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(zu,0x1.a36e2e0000000p-14f)));
czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f,fminf(fmaxf(__fdiv_rn(__fsub_rn(wsp10m,0x1.4000000000000p+3f),0x1.0000000000000p+3f),0x0.0p+0f),0x1.0000000000000p+0f)));
z_0 = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(czc,ustar),ustar),0x1.a1b2560000000p-4f),__fdiv_rn(__fmul_rn(0x1.c28f5c0000000p-4f,visc),fmaxf(ustar,0x1.99999a0000000p-5f)));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1002-1020 @1d6e370e water_edson_etal_2013
__device__ __noinline__ void m_water_edson_etal_2013(float &z_0, float ustar, float wsp10, float visc, float zu) {
float m = 0x1.bda5120000000p-10f;
float b = -0x1.47ae140000000p-8f;
float czc;
float wsp10m;
wsp10m = __fdiv_rn(__fmul_rn(wsp10,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(zu,0x1.a36e2e0000000p-14f)));
wsp10m = fminf(0x1.3000000000000p+4f,wsp10m);
czc = __fadd_rn(__fmul_rn(0x1.bda5120000000p-10f,wsp10m),-0x1.47ae140000000p-8f);
czc = fmaxf(czc,0x0.0p+0f);
z_0 = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(czc,ustar),ustar),0x1.a1b2560000000p-4f),__fdiv_rn(__fmul_rn(0x1.c28f5c0000000p-4f,visc),fmaxf(ustar,0x1.99999a0000000p-5f)));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1027-1051 @1d6e370e water_depth_dependent_z0
__device__ __noinline__ float m_water_depth_dependent_z0(float water_depth, float z0, float ust) {
float result;
float depth_b;
float wt;
float effective_depth;
if ((water_depth<0x1.4000000000000p+3f)) {
effective_depth = 0x1.4000000000000p+3f;
} else if ((water_depth>0x1.9000000000000p+6f)) {
effective_depth = 0x1.9000000000000p+6f;
} else {
effective_depth = water_depth;
}
depth_b = __fmul_rn(0x1.1111120000000p-5f,SL_LOG(__fdiv_rn(0x1.3b00000000000p+10f,effective_depth)));
result = SL_EXP(__fdiv_rn(__fsub_rn(__fmul_rn(0x1.59999a0000000p+1f,ust),__fdiv_rn(0x1.cccccc0000000p+0f,depth_b)),__fadd_rn(ust,__fdiv_rn(0x1.5c28f60000000p-3f,depth_b))));
result = fminf(result,0x1.99999a0000000p-4f);
wt = fminf(0x1.0000000000000p+0f,fmaxf(0x0.0p+0f,__fdiv_rn(__fsub_rn(water_depth,0x1.2c00000000000p+6f),0x1.2c00000000000p+6f)));
result = __fadd_rn(__fmul_rn(__fsub_rn(0x1.0000000000000p+0f,wt),result),__fmul_rn(wt,z0));
return result;
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1061-1082 @1d6e370e water_garratt_1992
__device__ __noinline__ void m_water_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea) {
float rq;
float e = 0x1.5bf0a80000000p+1f;
if ((__fsub_rn(landsea,0x1.8000000000000p+0f)>0)) {
zt = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.3d70a40000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.23d70a0000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = fminf(zq,0x1.cd5f9a0000000p-15f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
zt = fminf(zt,0x1.cd5f9a0000000p-15f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
} else {
zq = __fdiv_rn(z_0,0x1.d8e64a0000000p+2f);
zt = zq;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1093-1127 @1d6e370e water_fairall_etal_2003
__device__ __noinline__ void m_water_fairall_etal_2003(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, int spp_sfc) {
if ((ren<=0x1.0000000000000p+1f)) {
zt = __fmul_rn(0x1.cd5f9a0000000p-15f,SL_POW(ren,-0x1.3333340000000p-1f));
zq = zt;
} else {
zt = __fmul_rn(0x1.cd5f9a0000000p-15f,SL_POW(ren,-0x1.3333340000000p-1f));
zq = zt;
}
if ((spp_sfc==1)) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch));
zq = zt;
}
zt = fminf(zt,0x1.a36e2e0000000p-14f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = fminf(zq,0x1.a36e2e0000000p-14f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1135-1157 @1d6e370e water_fairall_etal_2014
__device__ __noinline__ void m_water_fairall_etal_2014(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, int spp_sfc) {
zt = fminf(0x1.4f8b580000000p-13f,__fdiv_rn(0x1.e68a0e0000000p-15f,SL_POW(ren,0x1.70a3d80000000p-1f)));
zq = zt;
if ((spp_sfc==1)) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch));
zq = __fadd_rn(zq,__fmul_rn(__fmul_rn(zq,0x1.0000000000000p-1f),rstoch));
}
zt = fminf(zt,0x1.a36e2e0000000p-14f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = fminf(zq,0x1.a36e2e0000000p-14f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1160-1209 @1d6e370e water_gfs_z0_wat
__device__ __noinline__ void m_water_gfs_z0_wat(float &z0rl, float &ustar, float wspd, float z1, int sfc_z0_type, bool redrag) {
float z0;
float z0max;
float wind10m;
float charnock = 0x1.cac0840000000p-7f;
float z0s_max = 0x1.9f7f8c0000000p-9f;
z0 = z0rl;
z0max = fmaxf(0x1.0c6f7a0000000p-20f,fminf(z0,z1));
ustar = SL_SQRT(__fdiv_rn(__fmul_rn(0x1.39cc100000000p+3f,z0),0x1.cac0840000000p-7f));
wind10m = __fdiv_rn(__fmul_rn(wspd,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(z1,0x1.a36e2e0000000p-14f)));
if ((sfc_z0_type>=0)) {
if ((sfc_z0_type==0)) {
z0 = __fmul_rn(__fmul_rn(0x1.7641960000000p-10f,ustar),ustar);
if (redrag) {
z0rl = fmaxf(fminf(z0,0x1.9f7f8c0000000p-9f),0x1.ad7f2a0000000p-24f);
} else {
z0rl = fmaxf(fminf(z0,0x1.99999a0000000p-4f),0x1.ad7f2a0000000p-24f);
}
} else if ((sfc_z0_type==6)) {
m_water_znot_m_v6(wind10m,z0);
} else if ((sfc_z0_type==7)) {
m_water_znot_m_v7(wind10m,z0);
} else {
z0rl = 0x1.0c6f7a0000000p-20f;
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1212-1257 @1d6e370e water_gfs_zt_wat
__device__ __noinline__ void m_water_gfs_zt_wat(float &ztmax, float z0rl, float restar, float wspd, float z1, int sfc_z0_type, int &errflg) {
float z0;
float z0max;
float wind10m;
float rat;
float ustar;
float charnock = 0x1.cac0840000000p-7f;
float z0s_max = 0x1.9f7f8c0000000p-9f;
errflg = 0;
z0 = z0rl;
z0max = fmaxf(0x1.0c6f7a0000000p-20f,fminf(z0,z1));
ustar = SL_SQRT(__fdiv_rn(__fmul_rn(0x1.39cc100000000p+3f,z0),0x1.cac0840000000p-7f));
wind10m = __fdiv_rn(__fmul_rn(wspd,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(z1,0x1.a36e2e0000000p-14f)));
rat = fminf(0x1.c000000000000p+2f,__fsub_rn(__fmul_rn(0x1.55c2900000000p+1f,SL_SQRT(SL_SQRT(restar))),0x1.48f5c20000000p+1f));
ztmax = fmaxf(__fmul_rn(z0max,SL_EXP((-rat))),0x1.0c6f7a0000000p-20f);
if ((sfc_z0_type==6)) {
m_water_znot_t_v6(wind10m,ztmax);
} else if ((sfc_z0_type==7)) {
m_water_znot_t_v7(wind10m,ztmax);
} else if ((sfc_z0_type>0)) {
errflg = 1;
return;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1262-1304 @1d6e370e water_znot_m_v6
__device__ __noinline__ void m_water_znot_m_v6(float uref, float &znotm) {
float p13 = -0x1.a8d8240000000p-7f;
float p12 = 0x1.246e940000000p-2f;
float p11 = -0x1.990fe00000000p+0f;
float p10 = -0x1.0cb4060000000p+3f;
float p25 = 0x1.a0ceda0000000p-32f;
float p24 = 0x1.c311da0000000p-29f;
float p23 = 0x1.a5659e0000000p-23f;
float p22 = -0x1.4cec960000000p-20f;
float p21 = 0x1.759c440000000p-23f;
float p20 = 0x1.68404a0000000p-16f;
float p35 = 0x1.8b3ab80000000p-23f;
float p34 = -0x1.d4baee0000000p-16f;
float p33 = 0x1.c6e6900000000p-10f;
float p32 = -0x1.f6eec80000000p-5f;
float p31 = 0x1.4165b00000000p+0f;
float p30 = -0x1.0a3d2e0000000p+4f;
float p40 = 0x1.e02e820000000p-12f;
if (((uref>=0x0.0p+0f)&&(uref<=0x1.a000000000000p+2f))) {
znotm = SL_EXP(__fadd_rn(-0x1.0cb4060000000p+3f,__fmul_rn(uref,__fadd_rn(-0x1.990fe00000000p+0f,__fmul_rn(uref,__fadd_rn(0x1.246e940000000p-2f,__fmul_rn(uref,-0x1.a8d8240000000p-7f)))))));
} else if (((uref>0x1.a000000000000p+2f)&&(uref<=0x1.f666660000000p+3f))) {
znotm = __fadd_rn(0x1.68404a0000000p-16f,__fmul_rn(uref,__fadd_rn(0x1.759c440000000p-23f,__fmul_rn(uref,__fadd_rn(-0x1.4cec960000000p-20f,__fmul_rn(uref,__fadd_rn(0x1.a5659e0000000p-23f,__fmul_rn(uref,__fadd_rn(0x1.c311da0000000p-29f,__fmul_rn(uref,0x1.a0ceda0000000p-32f))))))))));
} else if (((uref>0x1.f666660000000p+3f)&&(uref<=0x1.a800000000000p+5f))) {
znotm = SL_EXP(__fadd_rn(-0x1.0a3d2e0000000p+4f,__fmul_rn(uref,__fadd_rn(0x1.4165b00000000p+0f,__fmul_rn(uref,__fadd_rn(-0x1.f6eec80000000p-5f,__fmul_rn(uref,__fadd_rn(0x1.c6e6900000000p-10f,__fmul_rn(uref,__fadd_rn(-0x1.d4baee0000000p-16f,__fmul_rn(uref,0x1.8b3ab80000000p-23f)))))))))));
} else if ((uref>0x1.a800000000000p+5f)) {
znotm = 0x1.e02e820000000p-12f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1315-1368 @1d6e370e water_znot_t_v6
__device__ __noinline__ void m_water_znot_t_v6(float uref, float &znott) {
float p00 = 0x1.cd5f9a0000000p-14f;
float p15 = -0x1.f6ba8c0000000p-31f;
float p14 = 0x1.2d85880000000p-24f;
float p13 = -0x1.2151ec0000000p-19f;
float p12 = 0x1.17a9dc0000000p-15f;
float p11 = -0x1.139e4c0000000p-12f;
float p10 = 0x1.c4ae900000000p-11f;
float p25 = -0x1.3080ec0000000p-37f;
float p24 = 0x1.52c10a0000000p-30f;
float p23 = -0x1.25afc20000000p-24f;
float p22 = 0x1.f65a2a0000000p-20f;
float p21 = -0x1.ac322c0000000p-16f;
float p20 = 0x1.2bad260000000p-13f;
float p35 = 0x1.c36aea0000000p-39f;
float p34 = -0x1.43e3d40000000p-32f;
float p33 = 0x1.2dfc9e0000000p-27f;
float p32 = -0x1.47a8ce0000000p-25f;
float p31 = -0x1.48b2700000000p-19f;
float p30 = 0x1.204b0a0000000p-15f;
float p45 = -0x1.3690060000000p-35f;
float p44 = 0x1.0ebee40000000p-28f;
float p43 = -0x1.a5d2da0000000p-27f;
float p42 = -0x1.cad8c40000000p-17f;
float p41 = 0x1.39f4ce0000000p-11f;
float p40 = -0x1.fba9280000000p-8f;
float p56 = -0x1.0b82860000000p-43f;
float p55 = 0x1.a663120000000p-35f;
float p54 = -0x1.1491840000000p-27f;
float p53 = 0x1.80edfa0000000p-21f;
float p52 = -0x1.2c744e0000000p-15f;
float p51 = 0x1.f247720000000p-11f;
float p50 = -0x1.53b2fa0000000p-7f;
float p60 = 0x1.8e90740000000p-15f;
if (((uref>=0x0.0p+0f)&&(uref<0x1.79999a0000000p+2f))) {
znott = 0x1.cd5f9a0000000p-14f;
} else if (((uref>=0x1.79999a0000000p+2f)&&(uref<=0x1.eccccc0000000p+3f))) {
znott = __fadd_rn(0x1.c4ae900000000p-11f,__fmul_rn(uref,__fadd_rn(-0x1.139e4c0000000p-12f,__fmul_rn(uref,__fadd_rn(0x1.17a9dc0000000p-15f,__fmul_rn(uref,__fadd_rn(-0x1.2151ec0000000p-19f,__fmul_rn(uref,__fadd_rn(0x1.2d85880000000p-24f,__fmul_rn(uref,-0x1.f6ba8c0000000p-31f))))))))));
} else if (((uref>0x1.eccccc0000000p+3f)&&(uref<=0x1.59999a0000000p+4f))) {
znott = __fadd_rn(0x1.2bad260000000p-13f,__fmul_rn(uref,__fadd_rn(-0x1.ac322c0000000p-16f,__fmul_rn(uref,__fadd_rn(0x1.f65a2a0000000p-20f,__fmul_rn(uref,__fadd_rn(-0x1.25afc20000000p-24f,__fmul_rn(uref,__fadd_rn(0x1.52c10a0000000p-30f,__fmul_rn(uref,-0x1.3080ec0000000p-37f))))))))));
} else if (((uref>0x1.59999a0000000p+4f)&&(uref<=0x1.51999a0000000p+5f))) {
znott = __fadd_rn(0x1.204b0a0000000p-15f,__fmul_rn(uref,__fadd_rn(-0x1.48b2700000000p-19f,__fmul_rn(uref,__fadd_rn(-0x1.47a8ce0000000p-25f,__fmul_rn(uref,__fadd_rn(0x1.2dfc9e0000000p-27f,__fmul_rn(uref,__fadd_rn(-0x1.43e3d40000000p-32f,__fmul_rn(uref,0x1.c36aea0000000p-39f))))))))));
} else if (((uref>0x1.51999a0000000p+5f)&&(uref<=0x1.aa66660000000p+5f))) {
znott = __fadd_rn(-0x1.fba9280000000p-8f,__fmul_rn(uref,__fadd_rn(0x1.39f4ce0000000p-11f,__fmul_rn(uref,__fadd_rn(-0x1.cad8c40000000p-17f,__fmul_rn(uref,__fadd_rn(-0x1.a5d2da0000000p-27f,__fmul_rn(uref,__fadd_rn(0x1.0ebee40000000p-28f,__fmul_rn(uref,-0x1.3690060000000p-35f))))))))));
} else if (((uref>0x1.aa66660000000p+5f)&&(uref<=0x1.4000000000000p+6f))) {
znott = __fadd_rn(-0x1.53b2fa0000000p-7f,__fmul_rn(uref,__fadd_rn(0x1.f247720000000p-11f,__fmul_rn(uref,__fadd_rn(-0x1.2c744e0000000p-15f,__fmul_rn(uref,__fadd_rn(0x1.80edfa0000000p-21f,__fmul_rn(uref,__fadd_rn(-0x1.1491840000000p-27f,__fmul_rn(uref,__fadd_rn(0x1.a663120000000p-35f,__fmul_rn(uref,-0x1.0b82860000000p-43f))))))))))));
} else if ((uref>0x1.4000000000000p+6f)) {
znott = 0x1.8e90740000000p-15f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1378-1415 @1d6e370e water_znot_m_v7
__device__ __noinline__ void m_water_znot_m_v7(float uref, float &znotm) {
float p13 = -0x1.a8d8240000000p-7f;
float p12 = 0x1.246e940000000p-2f;
float p11 = -0x1.990fe00000000p+0f;
float p10 = -0x1.0cb4060000000p+3f;
float p25 = 0x1.a0ceda0000000p-32f;
float p24 = 0x1.c311da0000000p-29f;
float p23 = 0x1.a5659e0000000p-23f;
float p22 = -0x1.4cec960000000p-20f;
float p21 = 0x1.759c440000000p-23f;
float p20 = 0x1.68404a0000000p-16f;
float p35 = 0x1.977e100000000p-23f;
float p34 = -0x1.fa965e0000000p-16f;
float p33 = 0x1.fa4d960000000p-10f;
float p32 = -0x1.166ac80000000p-4f;
float p31 = 0x1.58c51e0000000p+0f;
float p30 = -0x1.11416a0000000p+4f;
float p40 = 0x1.6185120000000p-12f;
if (((uref>=0x0.0p+0f)&&(uref<=0x1.a000000000000p+2f))) {
znotm = SL_EXP(__fadd_rn(-0x1.0cb4060000000p+3f,__fmul_rn(uref,__fadd_rn(-0x1.990fe00000000p+0f,__fmul_rn(uref,__fadd_rn(0x1.246e940000000p-2f,__fmul_rn(uref,-0x1.a8d8240000000p-7f)))))));
} else if (((uref>0x1.a000000000000p+2f)&&(uref<=0x1.f666660000000p+3f))) {
znotm = __fadd_rn(0x1.68404a0000000p-16f,__fmul_rn(uref,__fadd_rn(0x1.759c440000000p-23f,__fmul_rn(uref,__fadd_rn(-0x1.4cec960000000p-20f,__fmul_rn(uref,__fadd_rn(0x1.a5659e0000000p-23f,__fmul_rn(uref,__fadd_rn(0x1.c311da0000000p-29f,__fmul_rn(uref,0x1.a0ceda0000000p-32f))))))))));
} else if (((uref>0x1.f666660000000p+3f)&&(uref<=0x1.a800000000000p+5f))) {
znotm = SL_EXP(__fadd_rn(-0x1.11416a0000000p+4f,__fmul_rn(uref,__fadd_rn(0x1.58c51e0000000p+0f,__fmul_rn(uref,__fadd_rn(-0x1.166ac80000000p-4f,__fmul_rn(uref,__fadd_rn(0x1.fa4d960000000p-10f,__fmul_rn(uref,__fadd_rn(-0x1.fa965e0000000p-16f,__fmul_rn(uref,0x1.977e100000000p-23f)))))))))));
} else if ((uref>0x1.a800000000000p+5f)) {
znotm = 0x1.6185120000000p-12f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1424-1478 @1d6e370e water_znot_t_v7
__device__ __noinline__ void m_water_znot_t_v7(float uref, float &znott) {
float p00 = 0x1.cd5f9a0000000p-14f;
float p15 = -0x1.f96ebc0000000p-31f;
float p14 = 0x1.2ee3f40000000p-24f;
float p13 = -0x1.225e840000000p-19f;
float p12 = 0x1.186d960000000p-15f;
float p11 = -0x1.1426440000000p-12f;
float p10 = 0x1.c53ef60000000p-11f;
float p25 = -0x1.4ad4320000000p-37f;
float p24 = 0x1.6c527e0000000p-30f;
float p23 = -0x1.397efc0000000p-24f;
float p22 = 0x1.0a23fa0000000p-19f;
float p21 = -0x1.c1adba0000000p-16f;
float p20 = 0x1.3749fc0000000p-13f;
float p35 = 0x1.16b60c0000000p-37f;
float p34 = -0x1.181bba0000000p-30f;
float p33 = 0x1.c3246a0000000p-25f;
float p32 = -0x1.671fcc0000000p-20f;
float p31 = 0x1.1669020000000p-16f;
float p30 = -0x1.3d0b320000000p-14f;
float p45 = -0x1.283fca0000000p-32f;
float p44 = 0x1.f3b52a0000000p-25f;
float p43 = -0x1.4f99520000000p-18f;
float p42 = 0x1.c196940000000p-13f;
float p41 = -0x1.2caf0e0000000p-8f;
float p40 = 0x1.417bba0000000p-5f;
float p56 = -0x1.76ae640000000p-43f;
float p55 = 0x1.27c3220000000p-34f;
float p54 = -0x1.833ea00000000p-27f;
float p53 = 0x1.0d6c960000000p-20f;
float p52 = -0x1.a47d280000000p-15f;
float p51 = 0x1.5c96d20000000p-10f;
float p50 = -0x1.db28080000000p-7f;
float p60 = 0x1.1eec900000000p-14f;
if (((uref>=0x0.0p+0f)&&(uref<0x1.79999a0000000p+2f))) {
znott = 0x1.cd5f9a0000000p-14f;
} else if (((uref>=0x1.79999a0000000p+2f)&&(uref<=0x1.eccccc0000000p+3f))) {
znott = __fadd_rn(0x1.c53ef60000000p-11f,__fmul_rn(uref,__fadd_rn(-0x1.1426440000000p-12f,__fmul_rn(uref,__fadd_rn(0x1.186d960000000p-15f,__fmul_rn(uref,__fadd_rn(-0x1.225e840000000p-19f,__fmul_rn(uref,__fadd_rn(0x1.2ee3f40000000p-24f,__fmul_rn(uref,-0x1.f96ebc0000000p-31f))))))))));
} else if (((uref>0x1.eccccc0000000p+3f)&&(uref<=0x1.59999a0000000p+4f))) {
znott = __fadd_rn(0x1.3749fc0000000p-13f,__fmul_rn(uref,__fadd_rn(-0x1.c1adba0000000p-16f,__fmul_rn(uref,__fadd_rn(0x1.0a23fa0000000p-19f,__fmul_rn(uref,__fadd_rn(-0x1.397efc0000000p-24f,__fmul_rn(uref,__fadd_rn(0x1.6c527e0000000p-30f,__fmul_rn(uref,-0x1.4ad4320000000p-37f))))))))));
} else if (((uref>0x1.59999a0000000p+4f)&&(uref<=0x1.54cccc0000000p+5f))) {
znott = __fadd_rn(-0x1.3d0b320000000p-14f,__fmul_rn(uref,__fadd_rn(0x1.1669020000000p-16f,__fmul_rn(uref,__fadd_rn(-0x1.671fcc0000000p-20f,__fmul_rn(uref,__fadd_rn(0x1.c3246a0000000p-25f,__fmul_rn(uref,__fadd_rn(-0x1.181bba0000000p-30f,__fmul_rn(uref,0x1.16b60c0000000p-37f))))))))));
} else if (((uref>0x1.54cccc0000000p+5f)&&(uref<=0x1.a800000000000p+5f))) {
znott = __fadd_rn(0x1.417bba0000000p-5f,__fmul_rn(uref,__fadd_rn(-0x1.2caf0e0000000p-8f,__fmul_rn(uref,__fadd_rn(0x1.c196940000000p-13f,__fmul_rn(uref,__fadd_rn(-0x1.4f99520000000p-18f,__fmul_rn(uref,__fadd_rn(0x1.f3b52a0000000p-25f,__fmul_rn(uref,-0x1.283fca0000000p-32f))))))))));
} else if (((uref>0x1.a800000000000p+5f)&&(uref<=0x1.4000000000000p+6f))) {
znott = __fadd_rn(-0x1.db28080000000p-7f,__fmul_rn(uref,__fadd_rn(0x1.5c96d20000000p-10f,__fmul_rn(uref,__fadd_rn(-0x1.a47d280000000p-15f,__fmul_rn(uref,__fadd_rn(0x1.0d6c960000000p-20f,__fmul_rn(uref,__fadd_rn(-0x1.833ea00000000p-27f,__fmul_rn(uref,__fadd_rn(0x1.27c3220000000p-34f,__fmul_rn(uref,-0x1.76ae640000000p-43f))))))))))));
} else if ((uref>0x1.4000000000000p+6f)) {
znott = 0x1.1eec900000000p-14f;
} else {
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1484-1513 @1d6e370e water_psi_hogstrom_1996
__device__ __noinline__ void m_water_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.5333340000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.0000000000000p+3f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1520-1550 @1d6e370e water_psi_dyerhicks
__device__ __noinline__ void m_water_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1556-1579 @1d6e370e water_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_water_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl) {
float a = 0x1.0000000000000p+0f;
float b = 0x1.54fdf40000000p-1f;
float c = 0x1.4000000000000p+2f;
float d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-__fadd_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+0f,zl),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-__fsub_rn(__fadd_rn(__fadd_rn(SL_POW(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1584-1609 @1d6e370e water_psi_businger_1971
__device__ __noinline__ void m_water_psi_businger_1971(float &psi_m, float &psi_h, float zl) {
float x;
float y;
float pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(SL_LOG(__fmul_rn(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),0x1.0000000000000p+1f))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = __fmul_rn(-0x1.2ccccc0000000p+2f,zl);
psi_h = __fmul_rn(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1618-1640 @1d6e370e water_psi_suselj_sood_2010
__device__ __noinline__ void m_water_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl) {
float rfc = 0x1.851eb80000000p-3f;
float ric = 0x1.76c8b40000000p-3f;
float phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-__fadd_rn(__fdiv_rn(zl,0x1.851eb80000000p-3f),__fmul_rn(0x1.1f4f0e0000000p+0f,SL_EXP(__fsub_rn(0x1.0000000000000p+0f,__fdiv_rn(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-__fadd_rn(__fdiv_rn(__fmul_rn(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),__fmul_rn(0x1.c5c2900000000p+2f,SL_POW(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = __fmul_rn(0x1.fb15b60000000p-1f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.c872b00000000p+3f,zl))));
psi_h = __fmul_rn(0x1.02a3060000000p+0f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.04e7d60000000p+4f,zl))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1646-1657 @1d6e370e water_psi_cb2005
__device__ __noinline__ void m_water_psi_cb2005(float &psim1, float &psih1, float zl, float z0l) {
psim1 = __fsub_rn(__fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),__fmul_rn(0x1.8666660000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = __fsub_rn(__fmul_rn(-0x1.6000000000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),__fmul_rn(0x1.6000000000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1663-1719 @1d6e370e water_li_etal_2010
__device__ __noinline__ void m_water_li_etal_2010(float &zl, float rib, float zaz0, float z0zt) {
float alfa;
float beta;
float zaz02;
float z0zt2;
float au11 = 0x1.70a3d80000000p-5f;
float bu11 = 0x1.89374c0000000p-9f;
float bu12 = 0x1.82a9940000000p-8f;
float bu21 = -0x1.5326180000000p-4f;
float bu22 = 0x1.c4dd300000000p-1f;
float bu31 = 0x1.6425ae0000000p-3f;
float bu32 = -0x1.d7b4a20000000p-1f;
float bu33 = -0x1.b0f27c0000000p-4f;
float aw11 = 0x1.25c91e0000000p-1f;
float aw12 = -0x1.c275260000000p-2f;
float aw21 = -0x1.39a9fc0000000p+2f;
float aw22 = 0x1.a400000000000p+5f;
float bw11 = -0x1.b98c7e0000000p-5f;
float bw12 = 0x1.8a3d700000000p+0f;
float bw21 = -0x1.56872c0000000p-1f;
float bw22 = -0x1.a418940000000p+1f;
float as11 = 0x1.817c1c0000000p-1f;
float as21 = 0x1.de147a0000000p+3f;
float bs11 = 0x1.4154ca0000000p-3f;
float bs21 = -0x1.3c84b60000000p-2f;
float bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = SL_LOG(zaz02);
beta = SL_LOG(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = __fadd_rn(__fmul_rn(__fmul_rn(0x1.70a3d80000000p-5f,alfa),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),__fmul_rn(alfa,alfa)),__fmul_rn(__fadd_rn(__fmul_rn(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),__fadd_rn(__fadd_rn(__fmul_rn(0x1.6425ae0000000p-3f,__fmul_rn(beta,beta)),__fmul_rn(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = fmaxf(zl,-0x1.e000000000000p+3f);
zl = fminf(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),__fadd_rn(__fmul_rn(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),__fadd_rn(__fmul_rn(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = fminf(zl,0x1.0000000000000p+2f);
zl = fmaxf(zl,0x0.0p+0f);
} else {
zl = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),__fmul_rn(0x1.4154ca0000000p-3f,alfa)),__fmul_rn(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = fminf(zl,0x1.4000000000000p+4f);
zl = fmaxf(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1722-1775 @1d6e370e water_zolri
__device__ __noinline__ float m_water_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float x1;
float x2;
float fx1;
float fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = __fsub_rn(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = __fadd_rn(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_water_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_water_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((fabsf(__fsub_rn(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((fabsf(fx2)<fabsf(fx1))) {
x1 = __fsub_rn(x1,__fmul_rn(__fdiv_rn(fx1,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx1 = m_water_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = __fsub_rn(x2,__fmul_rn(__fdiv_rn(fx2,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx2 = m_water_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(x1,x2))>=0x1.47ae140000000p-7f))) {
m_water_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1777-1817 @1d6e370e water_zolri2
__device__ __noinline__ float m_water_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float psim1;
float psih1;
float psix2;
float psit2;
float zolt;
if ((__fmul_rn(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = __fdiv_rn(__fmul_rn(zol2,z0),za);
zol3 = __fadd_rn(zol2,zol20);
zolt = __fdiv_rn(__fmul_rn(zol2,zt),za);
if ((ri2<0)) {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_water_psih_unstable(zol3,psi_opt,psih_unstab),m_water_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_water_psim_unstable(zol3,psi_opt,psim_unstab),m_water_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_water_psih_stable(zol3,psi_opt,psih_stab),m_water_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_water_psim_stable(zol3,psi_opt,psim_stab),m_water_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fsub_rn(__fdiv_rn(__fmul_rn(zol2,psit2),__fmul_rn(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1820-1894 @1d6e370e water_zolrib
__device__ __noinline__ float m_water_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float zolt;
float zolold;
int n;
int nmax = 20;
float psit2;
float psix2;
if ((__fmul_rn(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = __fdiv_rn(__fmul_rn(zolold,z0),za);
zol3 = __fadd_rn(zolold,zol20);
zolt = __fdiv_rn(__fmul_rn(zolold,zt),za);
if ((ri<0)) {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_water_psih_unstable(zol3,psi_opt,psih_unstab),m_water_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_water_psim_unstable(zol3,psi_opt,psim_unstab),m_water_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_water_psih_stable(zol3,psi_opt,psih_stab),m_water_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_water_psim_stable(zol3,psi_opt,psim_stab),m_water_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fdiv_rn(__fmul_rn(ri,__fmul_rn(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f))) {
m_water_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1899-1906 @1d6e370e water_psim_stable_full
__device__ __noinline__ float m_water_psim_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1909-1916 @1d6e370e water_psih_stable_full
__device__ __noinline__ float m_water_psih_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.5333340000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1919-1934 @1d6e370e water_psim_unstable_full
__device__ __noinline__ float m_water_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,x)))),SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
ym = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psimk,__fmul_rn(__fmul_rn(zolf,zolf),psimc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1937-1951 @1d6e370e water_psih_unstable_full
__device__ __noinline__ float m_water_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psihk,__fmul_rn(__fmul_rn(zolf,zolf),psihc)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1958-1967 @1d6e370e water_psim_stable_full_gfs
__device__ __noinline__ float m_water_psim_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float aa;
aa = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,aa),SL_LOG(__fadd_rn(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1971-1980 @1d6e370e water_psih_stable_full_gfs
__device__ __noinline__ float m_water_psih_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float bb;
bb = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,bb),SL_LOG(__fadd_rn(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:1984-2000 @1d6e370e water_psim_unstable_full_gfs
__device__ __noinline__ float m_water_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fccccc0000000p+1f,__fmul_rn(0x1.8a3d700000000p+3f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.f051ec0000000p+2f,__fmul_rn(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fsub_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p+1f,SL_SQRT(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2004-2020 @1d6e370e water_psih_unstable_full_gfs
__device__ __noinline__ float m_water_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fc39580000000p+2f,__fmul_rn(0x1.8c00000000000p+4f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.168f5c0000000p+3f,__fmul_rn(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fadd_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2024-2041 @1d6e370e water_psim_stable
__device__ __noinline__ float m_water_psim_stable(float zolf, int psi_opt, const float *psim_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_stab[nzol],__fmul_rn(rzol,__fsub_rn(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psim_stable_full(zolf);
} else {
result = m_water_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2044-2061 @1d6e370e water_psih_stable
__device__ __noinline__ float m_water_psih_stable(float zolf, int psi_opt, const float *psih_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_stab[nzol],__fmul_rn(rzol,__fsub_rn(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psih_stable_full(zolf);
} else {
result = m_water_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2064-2081 @1d6e370e water_psim_unstable
__device__ __noinline__ float m_water_psim_unstable(float zolf, int psi_opt, const float *psim_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psim_unstable_full(zolf);
} else {
result = m_water_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_water.F90:2084-2101 @1d6e370e water_psih_unstable
__device__ __noinline__ float m_water_psih_unstable(float zolf, int psi_opt, const float *psih_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_water_psih_unstable_full(zolf);
} else {
result = m_water_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:116-836 @1d6e370e ice_mynnsfc_ice
__device__ __noinline__ void m_ice_mynnsfc_ice(bool flag_iter, int itimestep, int i, int j, float dx, float xland, float u_1, float v_1, float t_1, float qv_1, float p_1, float dz8w_1, float &rho_1, float u_2, float v_2, float dz8w_2, float pblh, float &znt, float psfcpa, float mavail, float tskin, float tsurf, float snowh, float &qgh, float &chs, float &chs2, float &cqs2, float &cqs, float &ust, float &ustm, float &stress, float &rmol, float &zol, float &mol, float &psim, float &psih, float &hfx, float &qfx, float &u10, float &v10, float &th2, float &qsfc, float &t2, float &q2, float &flhc, float &flqc, float &lh, float &gz1oz0, float &wspd, float &rb, float &cpm, float &ch, float &cm, float rstoch_1, float &wstar, float &qstar, float &ck, float &cka, float &cd, float &cda, float &psix, float &psit, float &psix10, float &psit2, int spp_sfc, int isfflx, bool flag_restart, bool flag_cycle, int psi_opt, bool compute_flux, bool compute_diag, int iter, int lsm, int lsm_ruc, const float *psim_stab, const float *psim_unstab, const float *psih_stab, const float *psih_unstab, int &errflg, bool has_ck, bool has_cka, bool has_cd, bool has_cda) {
float xka = 0x1.92a7380000000p-16f;
float prt = 0x1.0000000000000p+0f;
float za;
float za2;
float thv_1;
float th_1;
float tc_1;
float tv_1;
float qvmr;
float psih2;
float psim10;
float psih10;
float wspdi;
float govrth;
float psfc;
float qsfcmr;
float thcon;
float zratio;
float tsk;
float thsk;
float thvsk;
float gz1ozt;
float gz2oz0;
float gz2ozt;
float gz10oz0;
float gz10ozt;
float zntstoch;
float zt;
float zq;
float psiq;
float psiq2;
int n;
int k;
int l;
int yesno;
float pl;
float e1;
float tabs;
float dthvdz;
float dthvm;
float vconv;
float zol2;
float zol10;
float zolza;
float zolz0;
float zolzt;
float dtg;
float dtthx;
float psiq10;
float fluxc;
float vsgd;
float restar;
float visc;
float dqg;
float oldust;
float oldtst;
errflg = 0;
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
psfc = __fdiv_rn(psfcpa,0x1.f400000000000p+9f);
if (flag_iter) {
if ((itimestep==1)) {
tsk = tskin;
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
} else {
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tsk))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tsk)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tsk)))));
qsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));
qsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));
}
} else {
if (((lsm==lsm_ruc)&&(qsfc>0x0.0p+0f))) {
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
} else {
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tskin))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tskin)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tskin)))));
qsfc = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,__fmul_rn(0x1.8353dc0000000p-2f,e1)));
qsfcmr = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(psfc,e1));
}
}
}
e1 = __fmul_rn(0x1.38ef340000000p-1f,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(0x1.2280000000000p+12f,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,t_1))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,t_1)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,t_1)))));
pl = __fdiv_rn(p_1,0x1.f400000000000p+9f);
qgh = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,e1),__fsub_rn(pl,e1));
qvmr = __fdiv_rn(qv_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
thcon = SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,psfcpa),0x1.24924a0000000p-2f);
if (flag_iter) {
tsk = tskin;
thsk = __fmul_rn(tsk,thcon);
thvsk = __fmul_rn(thsk,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qsfc)));
if (((thvsk<0x1.4000000000000p+7f)||(thvsk>0x1.8600000000000p+8f))) {
}
}
th_1 = __fmul_rn(t_1,SL_POW(__fdiv_rn(0x1.86a0000000000p+16f,p_1),0x1.24924a0000000p-2f));
tc_1 = __fsub_rn(t_1,0x1.1126660000000p+8f);
thv_1 = __fmul_rn(th_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
tv_1 = __fmul_rn(t_1,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.377b480000000p-1f,qv_1)));
rho_1 = __fdiv_rn(p_1,__fmul_rn(0x1.1f00000000000p+8f,tv_1));
za = __fmul_rn(0x1.0000000000000p-1f,dz8w_1);
za2 = __fadd_rn(dz8w_1,__fmul_rn(0x1.0000000000000p-1f,dz8w_2));
govrth = __fdiv_rn(0x1.39cc100000000p+3f,th_1);
cpm = __fmul_rn(0x1.f640000000000p+9f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.ae147a0000000p-1f,qv_1)));
if (flag_iter) {
wspd = SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1)));
dthvdz = __fsub_rn(thv_1,thvsk);
fluxc = fmaxf(__fadd_rn(__fdiv_rn(__fdiv_rn(hfx,rho_1),0x1.f640000000000p+9f),__fdiv_rn(__fmul_rn(__fmul_rn(0x1.377b480000000p-1f,thvsk),qfx),rho_1)),0x0.0p+0f);
wstar = __fmul_rn(0x1.4000000000000p+0f,SL_POW(__fmul_rn(__fmul_rn(__fdiv_rn(0x1.39cc100000000p+3f,tsk),fminf(__fmul_rn(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.5555560000000p-2f));
vsgd = fminf(__fmul_rn(0x1.47ae140000000p-2f,SL_POW(fmaxf(__fsub_rn(__fdiv_rn(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.5555560000000p-2f)),0x1.0000000000000p-1f);
wspd = SL_SQRT(__fadd_rn(__fadd_rn(__fmul_rn(wspd,wspd),__fmul_rn(wstar,wstar)),__fmul_rn(vsgd,vsgd)));
wspd = fmaxf(wspd,0x1.99999a0000000p-4f);
rb = __fdiv_rn(__fmul_rn(__fmul_rn(govrth,za),dthvdz),__fmul_rn(wspd,wspd));
rb = fmaxf(rb,-0x1.0000000000000p+1f);
rb = fminf(rb,0x1.0000000000000p+1f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
}
if (flag_iter) {
visc = __fmul_rn(0x1.bcee880000000p-17f,__fsub_rn(__fadd_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.acbc8c0000000p-8f,tc_1)),__fmul_rn(__fmul_rn(0x1.16890c0000000p-17f,tc_1),tc_1)),__fmul_rn(__fmul_rn(__fmul_rn(0x1.4c9a2e0000000p-28f,tc_1),tc_1),tc_1)));
if ((spp_sfc==1)) {
zntstoch = fmaxf(__fadd_rn(znt,__fmul_rn(__fmul_rn(znt,0x1.0000000000000p+0f),rstoch_1)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmaxf(__fdiv_rn(__fmul_rn(ust,zntstoch),visc),0x1.99999a0000000p-4f);
m_ice_andreas_2002(zntstoch,visc,ust,zt,zq,spp_sfc,rstoch_1);
gz1oz0 = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zntstoch));
gz1ozt = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zt));
gz2oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zt));
gz10oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zt));
zratio = __fdiv_rn(zntstoch,zt);
if ((rb>0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_ice_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_ice_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_ice_psim_stable(zolza,psi_opt,psim_stab),m_ice_psim_stable(zolz0,psi_opt,psim_stab));
psih = __fsub_rn(m_ice_psih_stable(zolza,psi_opt,psih_stab),m_ice_psih_stable(zolzt,psi_opt,psih_stab));
psim10 = __fsub_rn(m_ice_psim_stable(zol10,psi_opt,psim_stab),m_ice_psim_stable(zolz0,psi_opt,psim_stab));
psih10 = __fsub_rn(m_ice_psih_stable(zol10,psi_opt,psih_stab),m_ice_psih_stable(zolz0,psi_opt,psih_stab));
psih2 = __fsub_rn(m_ice_psih_stable(zol2,psi_opt,psih_stab),m_ice_psih_stable(zolzt,psi_opt,psih_stab));
rmol = __fdiv_rn(zol,za);
} else if ((rb==0x0.0p+0f)) {
psim = 0x0.0p+0f;
psih = 0x0.0p+0f;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = 0x0.0p+0f;
} else if ((rb<0x0.0p+0f)) {
if (((!flag_restart)||(flag_restart&&(itimestep>1)))) {
m_ice_li_etal_2010(zol,rb,__fdiv_rn(za,zntstoch),zratio);
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
if (((zntstoch<0x1.5798ee0000000p-27f)||(zt<0x1.b7cdfe0000000p-34f))) {
}
}
}
zol = m_ice_zolrib(rb,za,zntstoch,zt,gz1oz0,gz1ozt,zol,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
}
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
zolzt = __fdiv_rn(__fmul_rn(zol,zt),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
psim = __fsub_rn(m_ice_psim_unstable(zolza,psi_opt,psim_unstab),m_ice_psim_unstable(zolz0,psi_opt,psim_unstab));
psih = __fsub_rn(m_ice_psih_unstable(zolza,psi_opt,psih_unstab),m_ice_psih_unstable(zolzt,psi_opt,psih_unstab));
psim10 = __fsub_rn(m_ice_psim_unstable(zol10,psi_opt,psim_unstab),m_ice_psim_unstable(zolz0,psi_opt,psim_unstab));
psih10 = __fsub_rn(m_ice_psih_unstable(zol10,psi_opt,psih_unstab),m_ice_psih_unstable(zolz0,psi_opt,psih_unstab));
psih2 = __fsub_rn(m_ice_psih_unstable(zol2,psi_opt,psih_unstab),m_ice_psih_unstable(zolzt,psi_opt,psih_unstab));
psih = fminf(psih,__fmul_rn(0x1.cccccc0000000p-1f,gz1ozt));
psim = fminf(psim,__fmul_rn(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = fminf(psih2,__fmul_rn(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = fminf(psim10,__fmul_rn(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = fminf(psih10,__fmul_rn(0x1.cccccc0000000p-1f,gz10ozt));
rmol = __fdiv_rn(zol,za);
}
psix = fmaxf(__fsub_rn(gz1oz0,psim),0x1.0000000000000p+0f);
psix10 = fmaxf(__fsub_rn(gz10oz0,psim10),0x1.0000000000000p+0f);
psit = fmaxf(__fsub_rn(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmaxf(__fsub_rn(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,zq),zq)),psih),0x1.0000000000000p+0f);
psiq2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zq),zq)),psih2),0x1.0000000000000p+0f);
psiq10 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zq),zq)),psih10),0x1.0000000000000p+0f);
oldust = ust;
ust = __fadd_rn(__fmul_rn(0x1.0000000000000p-1f,ust),__fdiv_rn(__fmul_rn(0x1.99999a0000000p-3f,wspd),psix));
ust = fmaxf(ust,0x1.47ae140000000p-8f);
stress = __fmul_rn(ust,ust);
ustm = ust;
dtg = __fsub_rn(thv_1,thvsk);
oldtst = mol;
mol = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dtg),psit),0x1.0000000000000p+0f);
dqg = __fmul_rn(__fsub_rn(qv_1,qsfc),0x1.f400000000000p+9f);
qstar = __fdiv_rn(__fdiv_rn(__fmul_rn(0x1.99999a0000000p-2f,dqg),psiq),0x1.0000000000000p+0f);
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
if (flag_iter) {
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
cqs = 0x0.0p+0f;
cm = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,mavail),ust),0x1.99999a0000000p-2f),psiq);
flhc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),ust),0x1.99999a0000000p-2f),psit);
if (compute_flux) {
qfx = __fmul_rn(flqc,__fsub_rn(qsfc,qv_1));
qfx = fmaxf(qfx,-0x1.47ae140000000p-6f);
lh = __fmul_rn(0x1.55cc000000000p+18f,qfx);
hfx = __fmul_rn(__fdiv_rn(__fmul_rn(__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho_1,cpm),0x1.99999a0000000p-2f),wspd),psix),0x1.99999a0000000p-2f),psit),__fsub_rn(thsk,th_1));
hfx = fmaxf(hfx,-0x1.f400000000000p+7f);
}
chs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit);
cqs = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq);
cqs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psiq2);
chs2 = __fdiv_rn(__fmul_rn(ust,0x1.99999a0000000p-2f),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psiq10));
cd = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix10),__fdiv_rn(0x1.99999a0000000p-2f,psix10));
cka = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psiq));
cda = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
}
}
ch = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psit));
cm = __fmul_rn(__fdiv_rn(0x1.99999a0000000p-2f,psix),__fdiv_rn(0x1.99999a0000000p-2f,psix));
}
}
if (compute_diag) {
if (flag_iter) {
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u_2;
v10 = v_2;
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = __fdiv_rn(__fmul_rn(u_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v_1,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
} else {
u10 = __fdiv_rn(__fmul_rn(u_1,psix10),psix);
v10 = __fdiv_rn(__fmul_rn(v_1,psix10),psix);
}
dtg = __fsub_rn(th_1,thsk);
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(dtg,psit2),psit));
if ((((th_1>thsk)&&((th2<thsk)||(th2>th_1)))||((th_1<thsk)&&((th2>thsk)||(th2<th_1))))) {
th2 = __fadd_rn(thsk,__fdiv_rn(__fmul_rn(0x1.0000000000000p+1f,__fsub_rn(th_1,thsk)),za));
}
t2 = __fmul_rn(th2,SL_POW(__fdiv_rn(psfcpa,0x1.86a0000000000p+16f),0x1.24924a0000000p-2f));
q2 = __fadd_rn(qsfc,__fdiv_rn(__fmul_rn(__fsub_rn(qv_1,qsfc),psiq2),psiq));
q2 = fmaxf(q2,fminf(qsfc,qv_1));
q2 = fminf(q2,__fmul_rn(0x1.0ccccc0000000p+0f,qv_1));
}
}
if (false) {
if ((true||((false&&(i==31))&&(j==1)))) {
yesno = 0;
if (compute_flux) {
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho_1<0x0.0p+0f)||(rho_1>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:846-898 @1d6e370e ice_andreas_2002
__device__ __noinline__ void m_ice_andreas_2002(float z_0, float bvisc, float ustar, float &zt, float &zq, int spp_sfc, float rstoch) {
float ren2;
float zntsno;
float bt0_s = 0x1.4000000000000p+0f;
float bt0_t = 0x1.3126ea0000000p-3f;
float bt0_r = 0x1.449ba60000000p-2f;
float bt1_s = 0x0.0p+0f;
float bt1_t = -0x1.19999a0000000p-1f;
float bt1_r = -0x1.2147ae0000000p-1f;
float bt2_s = 0x0.0p+0f;
float bt2_t = 0x0.0p+0f;
float bt2_r = -0x1.76c8b40000000p-3f;
float bq0_s = 0x1.9c28f60000000p+0f;
float bq0_t = 0x1.676c8c0000000p-2f;
float bq0_r = 0x1.9581060000000p-2f;
float bq1_s = 0x0.0p+0f;
float bq1_t = -0x1.4189380000000p-1f;
float bq1_r = -0x1.0624de0000000p-1f;
float bq2_s = 0x0.0p+0f;
float bq2_t = 0x0.0p+0f;
float bq2_r = -0x1.70a3d80000000p-3f;
zntsno = __fadd_rn(__fdiv_rn(__fmul_rn(0x1.147ae20000000p-3f,bvisc),ustar),__fmul_rn(__fmul_rn(__fmul_rn(0x1.1eb8520000000p-5f,__fmul_rn(ustar,ustar)),0x1.a1b2560000000p-4f),__fadd_rn(__fmul_rn(0x1.4000000000000p+2f,SL_EXP(__fmul_rn(-0x1.0000000000000p+0f,__fmul_rn(__fdiv_rn(__fsub_rn(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f),__fdiv_rn(__fsub_rn(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f))))),0x1.0000000000000p+0f)));
ren2 = __fdiv_rn(__fmul_rn(ustar,zntsno),bvisc);
if ((ren2>0x1.f400000000000p+9f)) ren2 = 0x1.f400000000000p+9f;
if ((ren2<=0x1.147ae20000000p-3f)) {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.4000000000000p+0f,__fmul_rn(0x0.0p+0f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.9c28f60000000p+0f,__fmul_rn(0x0.0p+0f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
} else if (((ren2>0x1.147ae20000000p-3f)&&(ren2<0x1.4000000000000p+1f))) {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.3126ea0000000p-3f,__fmul_rn(-0x1.19999a0000000p-1f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.676c8c0000000p-2f,__fmul_rn(-0x1.4189380000000p-1f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
} else {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.449ba60000000p-2f,__fmul_rn(-0x1.2147ae0000000p-1f,SL_LOG(ren2))),__fmul_rn(-0x1.76c8b40000000p-3f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.9581060000000p-2f,__fmul_rn(-0x1.0624de0000000p-1f,SL_LOG(ren2))),__fmul_rn(-0x1.70a3d80000000p-3f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
}
if ((spp_sfc==1)) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch));
zt = fmaxf(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:903-932 @1d6e370e ice_psi_hogstrom_1996
__device__ __noinline__ void m_ice_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.5333340000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.0000000000000p+3f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:939-969 @1d6e370e ice_psi_dyerhicks
__device__ __noinline__ void m_ice_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:975-998 @1d6e370e ice_psi_beljaars_holtslag_1991
__device__ __noinline__ void m_ice_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl) {
float a = 0x1.0000000000000p+0f;
float b = 0x1.54fdf40000000p-1f;
float c = 0x1.4000000000000p+2f;
float d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-__fadd_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+0f,zl),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-__fsub_rn(__fadd_rn(__fadd_rn(SL_POW(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1005-1027 @1d6e370e ice_psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_ice_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl) {
float cm = 0x1.8000000000000p+1f;
float ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = __fmul_rn(-0x1.8000000000000p+1f,SL_POW(zl,0x1.aaaaaa0000000p-1f));
psi_h = __fmul_rn(-0x1.4000000000000p+1f,SL_POW(zl,0x1.99999a0000000p-1f));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1032-1057 @1d6e370e ice_psi_businger_1971
__device__ __noinline__ void m_ice_psi_businger_1971(float &psi_m, float &psi_h, float zl) {
float x;
float y;
float pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(SL_LOG(__fmul_rn(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),0x1.0000000000000p+1f))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = __fmul_rn(-0x1.2ccccc0000000p+2f,zl);
psi_h = __fmul_rn(-0x1.967c880000000p+2f,zl);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1066-1088 @1d6e370e ice_psi_suselj_sood_2010
__device__ __noinline__ void m_ice_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl) {
float rfc = 0x1.851eb80000000p-3f;
float ric = 0x1.76c8b40000000p-3f;
float phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-__fadd_rn(__fdiv_rn(zl,0x1.851eb80000000p-3f),__fmul_rn(0x1.1f4f0e0000000p+0f,SL_EXP(__fsub_rn(0x1.0000000000000p+0f,__fdiv_rn(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-__fadd_rn(__fdiv_rn(__fmul_rn(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),__fmul_rn(0x1.c5c2900000000p+2f,SL_POW(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = __fmul_rn(0x1.fb15b60000000p-1f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.c872b00000000p+3f,zl))));
psi_h = __fmul_rn(0x1.02a3060000000p+0f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.04e7d60000000p+4f,zl))));
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1094-1105 @1d6e370e ice_psi_cb2005
__device__ __noinline__ void m_ice_psi_cb2005(float &psim1, float &psih1, float zl, float z0l) {
psim1 = __fsub_rn(__fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),__fmul_rn(0x1.8666660000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = __fsub_rn(__fmul_rn(-0x1.6000000000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),__fmul_rn(0x1.6000000000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1111-1167 @1d6e370e ice_li_etal_2010
__device__ __noinline__ void m_ice_li_etal_2010(float &zl, float rib, float zaz0, float z0zt) {
float alfa;
float beta;
float zaz02;
float z0zt2;
float au11 = 0x1.70a3d80000000p-5f;
float bu11 = 0x1.89374c0000000p-9f;
float bu12 = 0x1.82a9940000000p-8f;
float bu21 = -0x1.5326180000000p-4f;
float bu22 = 0x1.c4dd300000000p-1f;
float bu31 = 0x1.6425ae0000000p-3f;
float bu32 = -0x1.d7b4a20000000p-1f;
float bu33 = -0x1.b0f27c0000000p-4f;
float aw11 = 0x1.25c91e0000000p-1f;
float aw12 = -0x1.c275260000000p-2f;
float aw21 = -0x1.39a9fc0000000p+2f;
float aw22 = 0x1.a400000000000p+5f;
float bw11 = -0x1.b98c7e0000000p-5f;
float bw12 = 0x1.8a3d700000000p+0f;
float bw21 = -0x1.56872c0000000p-1f;
float bw22 = -0x1.a418940000000p+1f;
float as11 = 0x1.817c1c0000000p-1f;
float as21 = 0x1.de147a0000000p+3f;
float bs11 = 0x1.4154ca0000000p-3f;
float bs21 = -0x1.3c84b60000000p-2f;
float bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = SL_LOG(zaz02);
beta = SL_LOG(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = __fadd_rn(__fmul_rn(__fmul_rn(0x1.70a3d80000000p-5f,alfa),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),__fmul_rn(alfa,alfa)),__fmul_rn(__fadd_rn(__fmul_rn(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),__fadd_rn(__fadd_rn(__fmul_rn(0x1.6425ae0000000p-3f,__fmul_rn(beta,beta)),__fmul_rn(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = fmaxf(zl,-0x1.e000000000000p+3f);
zl = fminf(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),__fadd_rn(__fmul_rn(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),__fadd_rn(__fmul_rn(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = fminf(zl,0x1.0000000000000p+2f);
zl = fmaxf(zl,0x0.0p+0f);
} else {
zl = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),__fmul_rn(0x1.4154ca0000000p-3f,alfa)),__fmul_rn(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = fminf(zl,0x1.4000000000000p+4f);
zl = fmaxf(zl,0x1.0000000000000p+0f);
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1170-1223 @1d6e370e ice_zolri
__device__ __noinline__ float m_ice_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float x1;
float x2;
float fx1;
float fx2;
int n;
int nmax = 20;
if ((ri<0x0.0p+0f)) {
x1 = __fsub_rn(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = __fadd_rn(zol1,0x1.47ae140000000p-6f);
}
n = 1;
fx1 = m_ice_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
fx2 = m_ice_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
while (((fabsf(__fsub_rn(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((fabsf(fx2)<fabsf(fx1))) {
x1 = __fsub_rn(x1,__fmul_rn(__fdiv_rn(fx1,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx1 = m_ice_zolri2(x1,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x1;
} else {
x2 = __fsub_rn(x2,__fmul_rn(__fdiv_rn(fx2,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx2 = m_ice_zolri2(x2,ri,za,z0,zt,psi_opt,psim_stab,psih_stab,psim_unstab,psih_unstab);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(x1,x2))>=0x1.47ae140000000p-7f))) {
m_ice_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1225-1265 @1d6e370e ice_zolri2
__device__ __noinline__ float m_ice_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float psim1;
float psih1;
float psix2;
float psit2;
float zolt;
if ((__fmul_rn(zol2,ri2)<0x0.0p+0f)) zol2 = 0x0.0p+0f;
zol20 = __fdiv_rn(__fmul_rn(zol2,z0),za);
zol3 = __fadd_rn(zol2,zol20);
zolt = __fdiv_rn(__fmul_rn(zol2,zt),za);
if ((ri2<0)) {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_ice_psih_unstable(zol3,psi_opt,psih_unstab),m_ice_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_ice_psim_unstable(zol3,psi_opt,psim_unstab),m_ice_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_ice_psih_stable(zol3,psi_opt,psih_stab),m_ice_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_ice_psim_stable(zol3,psi_opt,psim_stab),m_ice_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fsub_rn(__fdiv_rn(__fmul_rn(zol2,psit2),__fmul_rn(psix2,psix2)),ri2);
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1268-1342 @1d6e370e ice_zolrib
__device__ __noinline__ float m_ice_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt, const float *psim_stab, const float *psih_stab, const float *psim_unstab, const float *psih_unstab) {
float result;
float zol20;
float zol3;
float zolt;
float zolold;
int n;
int nmax = 20;
float psit2;
float psix2;
if ((__fmul_rn(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = __fdiv_rn(__fmul_rn(zolold,z0),za);
zol3 = __fadd_rn(zolold,zol20);
zolt = __fdiv_rn(__fmul_rn(zolold,zt),za);
if ((ri<0)) {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_ice_psih_unstable(zol3,psi_opt,psih_unstab),m_ice_psih_unstable(zolt,psi_opt,psih_unstab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_ice_psim_unstable(zol3,psi_opt,psim_unstab),m_ice_psim_unstable(zol20,psi_opt,psim_unstab))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_ice_psih_stable(zol3,psi_opt,psih_stab),m_ice_psih_stable(zolt,psi_opt,psih_stab))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_ice_psim_stable(zol3,psi_opt,psim_stab),m_ice_psim_stable(zol20,psi_opt,psim_stab))),0x1.0000000000000p+0f);
}
result = __fdiv_rn(__fmul_rn(ri,__fmul_rn(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f))) {
m_ice_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1347-1354 @1d6e370e ice_psim_stable_full
__device__ __noinline__ float m_ice_psim_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1357-1364 @1d6e370e ice_psih_stable_full
__device__ __noinline__ float m_ice_psih_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.5333340000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1367-1382 @1d6e370e ice_psim_unstable_full
__device__ __noinline__ float m_ice_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,x)))),SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
ym = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psimk,__fmul_rn(__fmul_rn(zolf,zolf),psimc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1385-1399 @1d6e370e ice_psih_unstable_full
__device__ __noinline__ float m_ice_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psihk,__fmul_rn(__fmul_rn(zolf,zolf),psihc)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1406-1415 @1d6e370e ice_psim_stable_full_gfs
__device__ __noinline__ float m_ice_psim_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float aa;
aa = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,aa),SL_LOG(__fadd_rn(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1419-1428 @1d6e370e ice_psih_stable_full_gfs
__device__ __noinline__ float m_ice_psih_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float bb;
bb = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,bb),SL_LOG(__fadd_rn(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1432-1448 @1d6e370e ice_psim_unstable_full_gfs
__device__ __noinline__ float m_ice_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fccccc0000000p+1f,__fmul_rn(0x1.8a3d700000000p+3f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.f051ec0000000p+2f,__fmul_rn(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fsub_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p+1f,SL_SQRT(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1452-1468 @1d6e370e ice_psih_unstable_full_gfs
__device__ __noinline__ float m_ice_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fc39580000000p+2f,__fmul_rn(0x1.8c00000000000p+4f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.168f5c0000000p+3f,__fmul_rn(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fadd_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1472-1489 @1d6e370e ice_psim_stable
__device__ __noinline__ float m_ice_psim_stable(float zolf, int psi_opt, const float *psim_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_stab[nzol],__fmul_rn(rzol,__fsub_rn(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psim_stable_full(zolf);
} else {
result = m_ice_psim_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1492-1509 @1d6e370e ice_psih_stable
__device__ __noinline__ float m_ice_psih_stable(float zolf, int psi_opt, const float *psih_stab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_stab[nzol],__fmul_rn(rzol,__fsub_rn(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psih_stable_full(zolf);
} else {
result = m_ice_psih_stable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1512-1529 @1d6e370e ice_psim_unstable
__device__ __noinline__ float m_ice_psim_unstable(float zolf, int psi_opt, const float *psim_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psim_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psim_unstable_full(zolf);
} else {
result = m_ice_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/module_sf_mynnsfc_ice.F90:1532-1549 @1d6e370e ice_psih_unstable
__device__ __noinline__ float m_ice_psih_unstable(float zolf, int psi_opt, const float *psih_unstab) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<1000)) {
result = __fadd_rn(psih_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_ice_psih_unstable_full(zolf);
} else {
result = m_ice_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87 @1d6e370e drv_mynnsfc_init
__device__ __noinline__ void m_drv_mynnsfc_init(bool allowed_to_read, int &errflg) {
m_drv_psi_init(0,errflg);
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:91-101 @1d6e370e drv_mynnsfclay_finalize
__device__ __noinline__ void m_drv_mynnsfclay_finalize(int &errflg) {
errflg = 0;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740 @1d6e370e drv_mynnsfc_driver
__device__ __noinline__ void m_drv_mynnsfc_driver(float u3d2, float v3d2, float dz8w2, float u3d, float v3d, float t3d, float qv3d, float p3d, float dz8w, float th3d, float rho3d, float sigmaf, int vegtype, float shdmax, int ivegsrc, float z0pert, float ztpert, bool &redrag, int &sfc_z0_type, float psfcpa, float &chs, float &chs2, float &cqs, float &cqs2, float &cpm, float &znt, float &ust, float &ustm, float &pblh, float mavail, float &zol, float &mol, float &rmol, float &psim, float &psih, float xland, float &qgh, float &hfx, float &qfx, float &lh, float tsk, float &flhc, float &flqc, float &qsfc, float &u10, float &v10, float &th2, float &t2, float lakemask, float &q2, float snowh, float &gz1oz0, float &wspd, float &br, float dx, float &ch, float &ck, float &cka, float &cd, float &cda, float &stress, float &hflx, float &qflx, float &cm, float &fm, float &fh, float &fm10, float &fh2, float tsurf, float water_depth, float xice, float xice_threshold, int spp_pbl, float pattern_spp_pbl, int sf_mynn_sfcflux_water, int sf_mynn_sfcflux_land, int shalwater_z0, int isfflx, bool restart, bool cycling, int initflag, bool &flag_iter, int flagc_lsm, int itimestep, int ids, int ide, int jds, int jde, int kds, int kde, int ims, int ime, int jms, int jme, int kms, int kme, int its, int ite, int jts, int jte, int kts, int kte, int &errflg, bool has_restart, bool has_cycling, bool has_initflag, bool has_sf_mynn_sfcflux_water, bool has_sf_mynn_sfcflux_land, bool has_shalwater_z0, bool has_flagc_lsm, bool has_spp_pbl, bool has_ivegsrc, bool has_sfc_z0_type, bool has_redrag, bool has_flag_iter, bool has_vegtype, bool has_sigmaf, bool has_shdmax, bool has_z0pert, bool has_ztpert, bool has_pattern_spp_pbl, bool has_tsurf, bool has_lakemask, bool has_water_depth, bool has_cm, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, bool has_fm, bool has_fh, bool has_fm10, bool has_fh2, bool has_hflx, bool has_qflx, bool has_stress) {
int flag_lsm;
int lsm_ruc = 3;
float snow_thresh = 0x1.99999a0000000p-5f;
int i;
int j;
int k;
int vegtype_1;
int iter;
int loc_z0_type;
int ncalls;
bool loc_redrag;
bool loc_iter;
bool loc_cycle;
float wstar;
float qstar;
float mavail_1;
float pblh_1;
float xland_1;
float tsk_1;
float psfcpa_1;
float snowh_1;
float dx_1;
float lakemask_1;
float wat_depth_1;
float xice_1;
float u_1;
float v_1;
float u_2;
float v_2;
float qv_1;
float p_1;
float t_1;
float rho_1;
float dz8w_1;
float dz8w_2;
float rstoch_1;
float qvmr;
float hfx_1;
float qfx_1;
float lh_1;
float mol_1;
float rmol_1;
float qsfc_1;
float qgh_1;
float znt_1;
float zol_1;
float ust_1;
float cpm_1;
float chs2_1;
float cqs_1;
float cqs2_1;
float chs_1;
float ch_1;
float flhc_1;
float flqc_1;
float gz1oz0_1;
float wspd_1;
float br_1;
float psim_1;
float psih_1;
float u10_1;
float v10_1;
float th2_1;
float t2_1;
float q2_1;
float cd_1;
float cda_1;
float ck_1;
float cka_1;
float ustm_1;
float wstar_1;
float qstar_1;
float tsurf_1;
float cm_1;
float stress_1;
float fm_1;
float fh_1;
float fm10_1;
float fh2_1;
float hflx_1;
float qflx_1;
float sigmaf_1;
float shdmax_1;
float z0pert_1;
float ztpert_1;
int debug_driver = 0;
errflg = 0;
iter = 1;
if ((flagc_lsm==1)) {
flag_lsm = 1;
} else if ((flagc_lsm==2)) {
flag_lsm = 2;
} else if ((flagc_lsm==3)) {
flag_lsm = 3;
} else {
flag_lsm = 4;
}
if (has_cycling) {
loc_cycle = cycling;
} else {
loc_cycle = false;
}
if (false) {
}
if (((itimestep==1)&&((!restart)||(!loc_cycle)))) {
{ i=its; j=jts; // independent MPAS column loop
{ i=its; j=jts; // independent MPAS column loop
if (((ust<=0x1.a36e2e0000000p-14f)||(ust>0x1.0000000000000p+2f))) {
u_1 = u3d;
v_1 = v3d;
ust = fmaxf(__fmul_rn(0x1.47ae140000000p-5f,SL_SQRT(__fadd_rn(__fmul_rn(u_1,u_1),__fmul_rn(v_1,v_1)))),0x1.0624de0000000p-10f);
}
pblh = fmaxf(0x1.4000000000000p+3f,__fmul_rn(0x1.5e00000000000p+9f,ust));
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
mol = 0x0.0p+0f;
}
}
}
if ((itimestep==1)) {
qstar = 0x0.0p+0f;
}
{ i=its; j=jts; // independent MPAS column loop
{ i=its; j=jts; // independent MPAS column loop
qvmr = qv3d;
dz8w_1 = dz8w;
qv_1 = fmaxf(0x1.5798ee0000000p-27f,__fdiv_rn(qvmr,__fadd_rn(0x1.0000000000000p+0f,qvmr)));
p_1 = p3d;
t_1 = t3d;
rho_1 = rho3d;
dz8w_2 = dz8w2;
u_2 = u3d2;
v_2 = v3d2;
u_1 = u3d;
v_1 = v3d;
ust_1 = ust;
stress_1 = __fmul_rn(ust_1,ust_1);
mol_1 = mol;
qsfc_1 = __fdiv_rn(qsfc,__fadd_rn(0x1.0000000000000p+0f,qvmr));
qgh_1 = __fdiv_rn(qgh,__fadd_rn(0x1.0000000000000p+0f,qvmr));
qstar_1 = qstar;
if ((spp_pbl==1)) {
rstoch_1 = pattern_spp_pbl;
} else {
rstoch_1 = 0x0.0p+0f;
}
mavail_1 = mavail;
pblh_1 = pblh;
xland_1 = xland;
xice_1 = xice;
tsk_1 = tsk;
psfcpa_1 = psfcpa;
snowh_1 = snowh;
dx_1 = dx;
if (has_lakemask) {
lakemask_1 = lakemask;
} else {
lakemask_1 = 0x0.0p+0f;
}
if ((has_water_depth&&has_shalwater_z0)) {
wat_depth_1 = water_depth;
} else {
wat_depth_1 = 0x1.f400000000000p+9f;
}
hfx_1 = hfx;
qfx_1 = __fdiv_rn(qfx,__fadd_rn(0x1.0000000000000p+0f,qvmr));
hflx_1 = __fdiv_rn(hfx_1,__fmul_rn(0x1.f640000000000p+9f,rho_1));
qflx_1 = __fdiv_rn(qfx_1,rho_1);
lh_1 = lh;
rmol_1 = rmol;
znt_1 = znt;
zol_1 = zol;
cpm_1 = cpm;
chs2_1 = chs2;
cqs2_1 = cqs2;
cqs_1 = cqs;
chs_1 = chs;
ch_1 = ch;
flhc_1 = flhc;
flqc_1 = flqc;
gz1oz0_1 = gz1oz0;
wspd_1 = wspd;
br_1 = br;
psim_1 = psim;
psih_1 = psih;
u10_1 = 0x0.0p+0f;
v10_1 = 0x0.0p+0f;
th2_1 = 0x0.0p+0f;
t2_1 = 0x0.0p+0f;
q2_1 = 0x0.0p+0f;
wstar_1 = 0x0.0p+0f;
loc_redrag = false;
if (has_redrag) loc_redrag = redrag;
loc_iter = true;
if (has_flag_iter) loc_iter = flag_iter;
loc_z0_type = 0;
if (has_sfc_z0_type) loc_z0_type = sfc_z0_type;
if (has_tsurf) {
tsurf_1 = tsurf;
} else {
tsurf_1 = tsk_1;
}
if (has_sigmaf) {
sigmaf_1 = sigmaf;
} else {
sigmaf_1 = 0x0.0p+0f;
}
if (has_shdmax) {
shdmax_1 = shdmax;
} else {
shdmax_1 = 0x0.0p+0f;
}
if (has_z0pert) {
z0pert_1 = z0pert;
} else {
z0pert_1 = 0x0.0p+0f;
}
if (has_ztpert) {
ztpert_1 = ztpert;
} else {
ztpert_1 = 0x0.0p+0f;
}
if (has_ztpert) {
vegtype_1 = vegtype;
} else {
vegtype_1 = 1;
}
if ((((has_ck&&has_cka)&&has_cd)&&has_cda)) {
ck_1 = 0x0.0p+0f;
cka_1 = 0x0.0p+0f;
cd_1 = 0x0.0p+0f;
cda_1 = 0x0.0p+0f;
}
if (has_ustm) {
ustm_1 = ustm;
} else {
ustm_1 = 0x0.0p+0f;
}
if ((((has_fm&&has_fh)&&has_fm10)&&has_fh2)) {
fm_1 = 0x0.0p+0f;
fh_1 = 0x0.0p+0f;
fm10_1 = 0x0.0p+0f;
fh2_1 = 0x0.0p+0f;
}
if (false) {
ncalls = 0;
}
if (((__fsub_rn(xland_1,0x1.8000000000000p+0f)<0x0.0p+0f)&&(snowh_1<0x1.99999a0000000p-5f))) {
if (false) {
ncalls = (ncalls+1);
}
m_land_mynnsfc_land(loc_iter,itimestep,i,j,dx_1,xland_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,sigmaf_1,vegtype_1,shdmax_1,ivegsrc,z0pert_1,ztpert_1,loc_redrag,loc_z0_type,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,qgh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,qsfc_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,sf_mynn_sfcflux_land,isfflx,restart,cycling,0,true,true,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,has_sf_mynn_sfcflux_land,true,true,true,true,true,true,true,true);
}
if ((__fsub_rn(xland_1,0x1.8000000000000p+0f)>=0x0.0p+0f)) {
if (false) {
ncalls = (ncalls+1);
}
m_water_mynnsfc_water(loc_iter,itimestep,i,j,dx_1,xland_1,lakemask_1,wat_depth_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,sigmaf_1,vegtype_1,shdmax_1,ivegsrc,z0pert_1,ztpert_1,loc_redrag,loc_z0_type,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,qsfc_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,qgh_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,sf_mynn_sfcflux_water,isfflx,restart,cycling,0,true,true,shalwater_z0,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,has_sf_mynn_sfcflux_water,has_shalwater_z0,true,true,true,true,true,true,true,true);
}
if ((((__fsub_rn(xland_1,0x1.8000000000000p+0f)<0x0.0p+0f)&&(snowh_1>=0x1.99999a0000000p-5f))||((xice_1>=xice_threshold)&&(xice_1<0x1.0000000000000p+0f)))) {
if (false) {
ncalls = (ncalls+1);
}
m_ice_mynnsfc_ice(loc_iter,itimestep,i,j,dx_1,xland_1,u_1,v_1,t_1,qv_1,p_1,dz8w_1,rho_1,u_2,v_2,dz8w_2,pblh_1,znt_1,psfcpa_1,mavail_1,tsk_1,tsurf_1,snowh_1,qgh_1,chs_1,chs2_1,cqs2_1,cqs_1,ust_1,ustm_1,stress_1,rmol_1,zol_1,mol_1,psim_1,psih_1,hfx_1,qfx_1,u10_1,v10_1,th2_1,qsfc_1,t2_1,q2_1,flhc_1,flqc_1,lh_1,gz1oz0_1,wspd_1,br_1,cpm_1,ch_1,cm_1,rstoch_1,wstar_1,qstar_1,ck_1,cka_1,cd_1,cda_1,fm_1,fh_1,fm10_1,fh2_1,spp_pbl,isfflx,restart,cycling,0,true,true,iter,flag_lsm,3,psim_stab,psim_unstab,psih_stab,psih_unstab,errflg,true,true,true,true);
}
hfx = hfx_1;
qfx = __fdiv_rn(qfx_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
lh = __fdiv_rn(lh_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
mol = mol_1;
rmol = rmol_1;
qsfc = __fdiv_rn(qsfc_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
qgh = __fdiv_rn(qgh_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
znt = znt_1;
zol = zol_1;
ust = ust_1;
cpm = cpm_1;
chs2 = chs2_1;
cqs2 = cqs2_1;
cqs = cqs_1;
chs = chs_1;
ch = ch_1;
flhc = flhc_1;
flqc = flqc_1;
gz1oz0 = gz1oz0_1;
wspd = wspd_1;
br = br_1;
psim = psim_1;
psih = psih_1;
u10 = u10_1;
v10 = v10_1;
th2 = th2_1;
t2 = t2_1;
q2 = __fdiv_rn(q2_1,__fsub_rn(0x1.0000000000000p+0f,qv_1));
wstar = wstar_1;
qstar = qstar_1;
if (has_hflx) hflx = __fdiv_rn(hfx_1,__fmul_rn(0x1.f640000000000p+9f,rho_1));
if (has_qflx) qflx = __fdiv_rn(qfx_1,rho_1);
if ((((has_ck&&has_cka)&&has_cd)&&has_cda)) {
ck = ck_1;
cka = cka_1;
cd = cd_1;
cda = cda_1;
}
if ((((has_fm&&has_fh)&&has_fm10)&&has_fh2)) {
fm = fm_1;
fh = fh_1;
fm10 = fm10_1;
fh2 = fh2_1;
}
if (has_ustm) ustm = ustm_1;
if (has_stress) stress = stress_1;
if (has_cm) cm = cm_1;
if (false) {
}
}
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:745-788 @1d6e370e drv_psi_init
__device__ __noinline__ void m_drv_psi_init(int psi_opt, int &errflg) {
int n;
float zolf;
if ((psi_opt==0)) {
for (n=0; n<=1000; ++n) {
zolf = __fmul_rn((float)(n),0x1.47ae140000000p-7f);
psim_stab[n] = m_drv_psim_stable_full(zolf);
psih_stab[n] = m_drv_psih_stable_full(zolf);
zolf = __fmul_rn((-(float)(n)),0x1.47ae140000000p-7f);
psim_unstab[n] = m_drv_psim_unstable_full(zolf);
psih_unstab[n] = m_drv_psih_unstable_full(zolf);
}
} else {
for (n=0; n<=1000; ++n) {
zolf = __fmul_rn((float)(n),0x1.47ae140000000p-7f);
psim_stab[n] = m_drv_psim_stable_full_gfs(zolf);
psih_stab[n] = m_drv_psih_stable_full_gfs(zolf);
zolf = __fmul_rn((-(float)(n)),0x1.47ae140000000p-7f);
psim_unstab[n] = m_drv_psim_unstable_full_gfs(zolf);
psih_unstab[n] = m_drv_psih_unstable_full_gfs(zolf);
}
}
if (((((psim_stab[1]<0x0.0p+0f)&&(psih_stab[1]<0x0.0p+0f))&&(psim_unstab[1]>0x0.0p+0f))&&(psih_unstab[1]>0x0.0p+0f))) {
errflg = 0;
} else {
errflg = 1;
}
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:794-801 @1d6e370e drv_psim_stable_full
__device__ __noinline__ float m_drv_psim_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:804-811 @1d6e370e drv_psih_stable_full
__device__ __noinline__ float m_drv_psih_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.5333340000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zolf,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f))));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:814-829 @1d6e370e drv_psim_unstable_full
__device__ __noinline__ float m_drv_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,x)))),SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
ym = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+3f,zolf)),0x1.5555560000000p-2f);
psimc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(ym,ym),ym),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psimk,__fmul_rn(__fmul_rn(zolf,zolf),psimc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:832-846 @1d6e370e drv_psih_unstable_full
__device__ __noinline__ float m_drv_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fmul_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p-1f)));
yh = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.1000000000000p+5f,zolf)),0x1.5555560000000p-2f);
psihc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(yh,yh),yh),0x1.0000000000000p+0f),0x1.5555560000000p-2f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,yh),0x1.0000000000000p+0f),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psihk,__fmul_rn(__fmul_rn(zolf,zolf),psihc)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(zolf,zolf)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:853-862 @1d6e370e drv_psim_stable_full_gfs
__device__ __noinline__ float m_drv_psim_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float aa;
aa = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,aa),SL_LOG(__fadd_rn(aa,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:866-875 @1d6e370e drv_psih_stable_full_gfs
__device__ __noinline__ float m_drv_psih_stable_full_gfs(float zolf) {
float result;
float alpha4 = 0x1.4000000000000p+4f;
float bb;
bb = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,bb),SL_LOG(__fadd_rn(bb,0x1.0000000000000p+0f)));
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:879-895 @1d6e370e drv_psim_unstable_full_gfs
__device__ __noinline__ float m_drv_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fccccc0000000p+1f,__fmul_rn(0x1.8a3d700000000p+3f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.f051ec0000000p+2f,__fmul_rn(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fsub_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p+1f,SL_SQRT(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:899-915 @1d6e370e drv_psih_unstable_full_gfs
__device__ __noinline__ float m_drv_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fc39580000000p+2f,__fmul_rn(0x1.8c00000000000p+4f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.168f5c0000000p+3f,__fmul_rn(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fadd_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740 @1d6e370e
extern "C" __global__ void mynnsfc_columns(float *f, const int *options,
    const int *vegtype, int *errors, float xice_threshold, unsigned long long present, int n,
    int step_override=-1, int init_override=-1) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n) return;
    int error=0;
    int o_ivegsrc=options[0];
    bool o_redrag=options[1];
    int o_sfc_z0_type=options[2];
    int o_spp_pbl=options[3];
    int o_sf_mynn_sfcflux_water=options[4];
    int o_sf_mynn_sfcflux_land=options[5];
    int o_shalwater_z0=options[6];
    int o_isfflx=options[7];
    bool o_restart=options[8];
    bool o_cycling=options[9];
    int o_initflag=init_override<0 ? options[10] : init_override;
    bool o_flag_iter=options[11];
    int o_flagc_lsm=options[12];
    int o_itimestep=step_override<0 ? options[13] : step_override;
    m_drv_mynnsfc_driver(f[0*n+c],f[1*n+c],f[2*n+c],f[3*n+c],f[4*n+c],f[5*n+c],f[6*n+c],f[7*n+c],f[8*n+c],f[9*n+c],f[10*n+c],f[11*n+c],vegtype[c],f[12*n+c],o_ivegsrc,f[13*n+c],f[14*n+c],o_redrag,o_sfc_z0_type,f[15*n+c],f[16*n+c],f[17*n+c],f[18*n+c],f[19*n+c],f[20*n+c],f[21*n+c],f[22*n+c],f[23*n+c],f[24*n+c],f[25*n+c],f[26*n+c],f[27*n+c],f[28*n+c],f[29*n+c],f[30*n+c],f[31*n+c],f[32*n+c],f[33*n+c],f[34*n+c],f[35*n+c],f[36*n+c],f[37*n+c],f[38*n+c],f[39*n+c],f[40*n+c],f[41*n+c],f[42*n+c],f[43*n+c],f[44*n+c],f[45*n+c],f[46*n+c],f[47*n+c],f[48*n+c],f[49*n+c],f[50*n+c],f[51*n+c],f[52*n+c],f[53*n+c],f[54*n+c],f[55*n+c],f[56*n+c],f[57*n+c],f[58*n+c],f[59*n+c],f[60*n+c],f[61*n+c],f[62*n+c],f[63*n+c],f[64*n+c],f[65*n+c],f[66*n+c],xice_threshold,o_spp_pbl,f[67*n+c],o_sf_mynn_sfcflux_water,o_sf_mynn_sfcflux_land,o_shalwater_z0,o_isfflx,o_restart,o_cycling,o_initflag,o_flag_iter,o_flagc_lsm,o_itimestep,1,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,error,(present & (1ull<<0))!=0,(present & (1ull<<1))!=0,(present & (1ull<<2))!=0,(present & (1ull<<3))!=0,(present & (1ull<<4))!=0,(present & (1ull<<5))!=0,(present & (1ull<<6))!=0,(present & (1ull<<7))!=0,(present & (1ull<<8))!=0,(present & (1ull<<9))!=0,(present & (1ull<<10))!=0,(present & (1ull<<11))!=0,(present & (1ull<<12))!=0,(present & (1ull<<13))!=0,(present & (1ull<<14))!=0,(present & (1ull<<15))!=0,(present & (1ull<<16))!=0,(present & (1ull<<17))!=0,(present & (1ull<<18))!=0,(present & (1ull<<19))!=0,(present & (1ull<<20))!=0,(present & (1ull<<21))!=0,(present & (1ull<<22))!=0,(present & (1ull<<23))!=0,(present & (1ull<<24))!=0,(present & (1ull<<25))!=0,(present & (1ull<<26))!=0,(present & (1ull<<27))!=0,(present & (1ull<<28))!=0,(present & (1ull<<29))!=0,(present & (1ull<<30))!=0,(present & (1ull<<31))!=0,(present & (1ull<<32))!=0,(present & (1ull<<33))!=0);
    errors[c]=error;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87,747-788 @1d6e370e
extern "C" __global__ void mynnsfc_tables(int *errors) {
    if(blockIdx.x || threadIdx.x) return;
    int error=0; m_drv_psi_init(0,error); errors[0]=error;
}
// Verify direct NVRTC subnormal behavior at the actual RKIND arithmetic boundary.
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_common.F90:14-24 @1d6e370e
extern "C" __global__ void mynnsfc_subnormal(float *x) {
    int i=threadIdx.x; if(i<4) x[i]=__fmul_rn(x[i],0.5f);
}
"""

_MODULES = {}


def cuda_source():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87 @1d6e370e."""
    kernels=files('woof').joinpath('core/kernels')
    return ("typedef float real;\n#define SL_LOG mynn_glibc_logf\n"
            "#define SL_EXP mynn_glibc_expf\n#define SL_POW mynn_glibc_powf\n"
            "#define SL_ATAN glibc_atanf\n#define SL_SQRT __fsqrt_rn\n"
            + kernels.joinpath('mynn_libm.cuh').read_text()
            + kernels.joinpath('glibc_trig_flt32.cuh').read_text() + SOURCE + _driver_source())


def module():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:73-87,747-788."""
    import cupy as cp
    from cupy.cuda import compiler, function
    dev=cp.cuda.runtime.getDevice()
    if dev not in _MODULES:
        binary,_=compiler.compile_using_nvrtc(cuda_source(),
            options=_hex_numerics.options(('--std=c++17','--fmad=false','--ftz=false',
                     '--prec-div=true','--prec-sqrt=true')),
            arch=cp.cuda.Device(dev).compute_capability,cache_in_memory=True)
        obj=function.Module(); obj.load(binary)
        error=cp.empty(1,dtype=cp.int32)
        obj.get_function('mynnsfc_tables')((1,),(1,),(error,))
        _MODULES[dev]=obj
    return _MODULES[dev]


def run_packed(fields, options, *, vegtype=None, errors=None,
               xice_threshold=0.02, present=MPAS_PRESENT,
               itimestep=None, initflag=None):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90:103-740.

    Two lowest profile levels use FIELDS u3d/u3d2, v3d/v3d2, dz8w/dz8w2.
    Other fields use the MPAS driver argument names; no host field arithmetic.
    Optional presence is a uint64 mask (34 arguments). Errors stay on device.
    """
    import cupy as cp
    if (not isinstance(fields,cp.ndarray) or fields.dtype!=cp.float32
            or fields.ndim!=2 or fields.shape[0]!=len(FIELDS)
            or not fields.flags.c_contiguous):
        raise TypeError('MYNN-SFC requires contiguous float32 (len(FIELDS), nCells) device storage')
    if (not isinstance(options,cp.ndarray) or options.dtype!=cp.int32
            or options.shape!=(len(OPTIONS),) or not options.flags.c_contiguous):
        raise TypeError('MYNN-SFC options require contiguous int32 device storage in OPTIONS order')
    n=fields.shape[1]
    if vegtype is None: vegtype=cp.ones(n,dtype=cp.int32)
    if errors is None: errors=cp.empty(n,dtype=cp.int32)
    for value in (vegtype,errors):
        if value.dtype!=cp.int32 or value.shape!=(n,) or not value.flags.c_contiguous:
            raise TypeError('MYNN-SFC integer carriers require contiguous int32 (nCells,) storage')
    if n:
        module().get_function('mynnsfc_columns')(((n+127)//128,),(128,),
            (fields,options,vegtype,errors,np.float32(xice_threshold),np.uint64(present),np.int32(n),
             np.int32(-1 if itimestep is None else itimestep),
             np.int32(-1 if initflag is None else initflag)))
    return errors

# Appended verbatim to the generated runtime module by assemble.py.
# RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-1052,1148-1463 @1d6e370e
_DIAG_INPUT = ('br','cpm','chs','chs2','cqs','cqs2','flhc','flqc','gz1oz0',
               'hfx','qfx','qgh','qsfc','lh','mol','psim','psih','rmol',
               'ust','ustm','wspd','znt','zol','mavail','ch')
_SURFACE = {'tsk':'skintemp','xland':'xland','xice':'xice','snowh':'snowh'}
_WRITE = tuple(n for n in OUTPUTS if n not in ('pblh','stress','hflx','qflx','cm','fm','fh','fm10','fh2'))
_DIAG_NAMES = {'t2':'t2m','th2':'th2m'}
_BLEND = ('br','gz1oz0','mol','psih','psim','rmol','ust','wspd','zol','q2',
          't2','th2','u10','v10','cd','cda','ck','cka','ustm','ch')
# RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:574-597 @1d6e370e
_SEA_HANDOFF = ('chs','chs2','cqs','cqs2','cpm','flhc','flqc',
                'hfx','qfx','qgh','qsfc','lh')
DRIVER_SOURCE = r'''
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:517-622 @1d6e370e
extern "C" __global__ void mynnsfc_from_mpas(float *f,float *sea,
    const float *density,const float *sst,const float *ice,
    float len_disp,float threshold,int fraction,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    // Strict GNU 15.2 emits powf(x,0.25); see native power probe.
    F(dx)=__fdiv_rn(len_disp,SL_POW(density[i],0.25f));
    if(fraction) {
        for(int j=0;j<FIELD_COUNT;++j) sea[j*n+i]=f[j*n+i];
        sea[IDX_qsfc*n+i]=0.0f;
        if(ice[i]>=threshold && ice[i]<1.0f) {
            sea[IDX_xland*n+i]=2.0f;
            F(xland)=1.0f;
            sea[IDX_mavail*n+i]=1.0f;
            sea[IDX_znt*n+i]=0.0001f;
            sea[IDX_tsk*n+i]=fmaxf(sst[i],271.4f);
        }
    }
}
// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:903-927,973-991 @1d6e370e
extern "C" __global__ void mynnsfc_to_mpas(float *f,const float *sea,
    const float *ice,float threshold,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || ice[i]<threshold || ice[i]>=1.0f) return;
    float x=ice[i],y=__fsub_rn(1.0f,x);
    for(int j=0;j<FIELD_COUNT;++j)
        if(BLEND_FIELD(j)) f[j*n+i]=__fadd_rn(__fmul_rn(f[j*n+i],x),
                                            __fmul_rn(y,sea[j*n+i]));
}
'''


def _driver_source():
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:517-622,903-991 @1d6e370e."""
    macros='\n'.join('#define IDX_'+name+' '+str(i) for i,name in enumerate(FIELDS))
    macros+='\n#define FIELD_COUNT '+str(len(FIELDS))
    macros+='\n#define F(name) f[IDX_##name*n+i]\n'
    macros+='#define BLEND_FIELD(j) ('+' || '.join('(j)==IDX_'+k for k in _BLEND)+')\n'
    return macros+DRIVER_SOURCE


def from_mpas(mesh, diagnostics, surface, profiles, *, len_disp,
              xice_threshold, config_frac_seaice=False):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:434-698 @1d6e370e.

    Profiles hold prepared MPAS atmosphere arrays, float32 (level, cell).
    psfcpa is the prepared surface-pressure view from MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:602.
    The enclosing adapter owns atmosphere preparation and solve-cell views.
    """
    import cupy as cp
    n=surface['xland'].size
    for name in ('u3d','v3d','t3d','qv3d','p3d','dz8w','th3d','rho3d'):
        value=profiles[name]
        if (not isinstance(value,cp.ndarray) or value.dtype!=cp.float32
                or value.ndim!=2 or value.shape[0]<2 or value.shape[1]!=n
                or not value.flags.c_contiguous):
            raise ValueError('MYNN-SFC profile '+name+' requires contiguous float32 (levels>=2, nCells)')
    packed=cp.zeros((len(FIELDS),n),cp.float32)
    for name in _DIAG_INPUT: packed[FIELDS.index(name)]=diagnostics[name]
    packed[FIELDS.index('pblh')]=diagnostics['hpbl']
    for name,key in _SURFACE.items(): packed[FIELDS.index(name)]=surface[key]
    packed[FIELDS.index('psfcpa')]=profiles['psfcpa']
    for name in ('u3d','v3d','t3d','qv3d','p3d','dz8w','th3d','rho3d'):
        packed[FIELDS.index(name)]=profiles[name][0]
    for name in ('u3d','v3d','dz8w'): packed[FIELDS.index(name+'2')]=profiles[name][1]
    sea=cp.empty_like(packed) if config_frac_seaice else packed
    sst=surface['sst'] if config_frac_seaice else packed[0]
    if n:
        module().get_function('mynnsfc_from_mpas')(((n+127)//128,),(128,),
            (packed,sea,mesh['meshDensity'],sst,surface['xice'],np.float32(len_disp),
             np.float32(xice_threshold),np.int32(config_frac_seaice),np.int32(n)))
    return packed,sea


def to_mpas(packed, diagnostics, surface, *, sea=None,xice_threshold):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:772-991 @1d6e370e.

    hpbl and regime are unchanged in these native copy-back branches.
    cqs, flhc, flqc and the fluxes retain the first call's values.
    """
    n=packed.shape[1]
    if sea is not None and n:
        module().get_function('mynnsfc_to_mpas')(((n+127)//128,),(128,),
            (packed,sea,surface['xice'],np.float32(xice_threshold),np.int32(n)))
    for name in _WRITE: diagnostics[_DIAG_NAMES.get(name,name)][...]=packed[FIELDS.index(name)]
    return diagnostics


def driver(mesh, diagnostics, surface, profiles, *, itimestep,len_disp,
           xice_threshold,config_frac_seaice=False,config_do_restart=False,
           config_do_DAcycling=False,config_lsm_scheme='sf_ruc',
           config_mynn_sfcflux_land=0,config_mynn_sfcflux_water=0,spp_pbl=0,
           sea_state=None):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1148-1209,1390-1463 @1d6e370e."""
    import cupy as cp
    initflag=int(not (config_do_restart or config_do_DAcycling or itimestep>1))
    lsm={'sf_noaa':1,'sf_noahmp':2,'sf_ruc':3}.get(config_lsm_scheme,4)
    values=dict(ivegsrc=0,redrag=0,sfc_z0_type=0,spp_pbl=spp_pbl,
        sf_mynn_sfcflux_water=config_mynn_sfcflux_water,sf_mynn_sfcflux_land=config_mynn_sfcflux_land,
        shalwater_z0=0,isfflx=1,restart=int(config_do_restart),cycling=int(config_do_DAcycling),
        initflag=initflag,flag_iter=1,flagc_lsm=lsm,itimestep=itimestep)
    options=cp.asarray([values[name] for name in OPTIONS],cp.int32)
    packed,sea=from_mpas(mesh,diagnostics,surface,profiles,len_disp=len_disp,
                         xice_threshold=xice_threshold,config_frac_seaice=config_frac_seaice)
    errors=run_packed(packed,options,xice_threshold=xice_threshold)
    if config_frac_seaice:
        # hpbl_p is shared by both native calls; the first cold-start update
        # is visible to the sea call, without modifying diag_physics%hpbl.
        sea[FIELDS.index('pblh')]=packed[FIELDS.index('pblh')]
        errors=run_packed(sea,options,xice_threshold=xice_threshold,errors=errors)
        # RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:335-361,1434-1445 @1d6e370e
        # These globals survive sfclayer deallocation for native sea-ice/RUC.
        if sea_state is not None:
            for name in _SEA_HANDOFF:
                sea_state[name+'_sea'][...]=sea[FIELDS.index(name)]
    result=to_mpas(packed,diagnostics,surface,sea=sea if config_frac_seaice else None,
                   xice_threshold=xice_threshold)
    return result


def launch_slot(call):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1148-1463 @1d6e370e."""
    if (call.slot,call.scheme,str(call.backend))!=('surface_layer',SCHEME,BACKEND):
        raise ValueError('MYNN-SFC launcher requires surface_layer/sf_mynnsfclay/hex-cuda')
    from .resident_sfclay_rrfs import resident_slot
    slot = resident_slot(call.pools, call)
    if slot is not None:
        return slot.run(call)
    pools=call.pools
    mesh,diag,sfc,atm,configs,state=(pools[k] for k in
        ('mesh','diag_physics','sfc_input','atm_input','configs','scalar_state'))
    return driver(mesh,diag,sfc,atm,itimestep=call.itimestep,
        len_disp=configs['config_len_disp'],xice_threshold=state['xice_threshold'],
        config_do_restart=configs['config_do_restart'],config_do_DAcycling=configs['config_do_DAcycling'],
        config_frac_seaice=call.options['config_frac_seaice'],
        config_lsm_scheme=call.options['config_lsm_scheme'],
        config_mynn_sfcflux_land=call.options['config_mynn_sfcflux_land'],
        config_mynn_sfcflux_water=call.options['config_mynn_sfcflux_water'],
        sea_state=pools['prepared'])


def register_backend(registry):
    """RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1390-1463 @1d6e370e."""
    registry.register('surface_layer',SCHEME,BACKEND,launch_slot)
