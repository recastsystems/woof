from __future__ import annotations
from functools import lru_cache
from woof.core.device_cache import cuda_cache
from pathlib import Path
from types import MappingProxyType

from woof.core.constants import CUDA_DEFINES

_KDIR = Path(__file__).parent

# Every read of a .cu/.cuh in this directory is UTF-8, named explicitly.
#
# These sources are the input to nvrtc AND to the pinned hash the kernel
# manifest records (certify/kernel_manifest.py::record_module digests exactly
# the string module_source returns).  Path.read_text() with no encoding
# decodes with the host's locale, which is cp1252 on a stock Windows box, and
# cp1252 does not fail on a UTF-8 em dash -- it silently turns the three bytes
# into three characters.  acoustic.cu, advection.cu and coriolis_map.cu all
# carry U+2014 in comments, so the same checkout produced two different
# source_sha256 values depending on the host, with nothing raising to say so.
# The Thompson translation units happen to be pure ASCII today, which is luck,
# not a property anyone is maintaining.
_ENCODING = "utf-8"

# Explicit, closed allow-list of modules that receive an extra device header
# prepended between the preamble and their own source.  There is no #include
# path under cupy.RawModule, so this is how the six aerosol-aware Thompson
# (mp_physics=28) translation units share one set of __device__ helpers.
#
# The mechanism is deliberately inert for everything else: a module absent
# from this dict contributes the empty string and therefore assembles a
# BYTE-IDENTICAL source to what it assembled before this hook existed.  That
# is what keeps woof/core/kernels/thompson.cu's compiled source string -- and
# so its PTX, register allocation and FP contraction -- unchanged by
# construction rather than by measurement.  tests/test_kernel_loader_inert.py
# proves it for every .cu file in this directory.
#
# This table must stay a literal name -> filenames mapping.  Do not give it
# filesystem probing, globbing, or any implicit fallback.
_EXTRA_HEADERS: dict[str, tuple[str, ...]] = {
    "morrison_aero": ("noahmp_leaves.cu",),
    "morrison_aero_refl": ("noahmp_leaves.cu", "refl.cu"),
    "udm": ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh", "thompson_column_support.cuh", "thompson_column_radar.cuh", "udm_math.cuh", "udm_generated.cuh"),
    "rcon": ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_aerosol_libm.cuh", "thompson_column_support.cuh", "thompson_column_radar.cuh", "rcon_math.cuh", "rcon_generated.cuh"),
    "fogdes": ("glibc_flt32.cuh",),
    "madwrf": ("glibc_flt32.cuh",),
    "deng": ("glibc_flt32.cuh", "deng_support.cuh", "deng_column.cuh"),
    "grims": ("glibc_flt32.cuh",),
    "shcu_nscv": ("glibc_flt32.cuh",),
    "tc_mixing": ("glibc_flt32.cuh", "smag2d.cu"),
    "km5": ("glibc_flt32.cuh", "km5_math.cuh"),
    "km5_geometry": ("glibc_flt32.cuh", "smag2d.cu"),
    "sfs_output": ("glibc_flt32.cuh", "smag2d.cu"),

    "flg_driver": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "flg_column": ("glibc_flt32.cuh", "flg_workspace_generated.cuh"),
    "flg": ("glibc_flt32.cuh",),
    "flg_gases": ("glibc_flt32.cuh", "flg_gases_generated.cuh"),
    "flg_streams": ("glibc_flt32.cuh", "flg_streams_generated.cuh"),
    "sas": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
            "atan_scalar_flt32.cuh", "wrf_atan_f32.cuh"),
    "gfdleta_init": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "gfdleta_lw_helpers": ("portable_libm64.cuh",),
    "gfdleta_tables": ("glibc_flt32.cuh", "gfdleta_table_coefficients.cuh"),
    "gfdleta_math_probe": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_trig_flt32.cuh", "portable_libm64.cuh", "gfdleta_support.cuh"),
    "gfdleta_co2_init": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_trig_flt32.cuh", "portable_libm64.cuh", "gfdleta_support.cuh", "gfdleta_co2_init.cuh"),
    "gfdleta_table_runtime": ("glibc_flt32.cuh",),
    "gfdleta_column": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_trig_flt32.cuh", "portable_libm64.cuh", "gfdleta_support.cuh", "gfdleta_column.cuh"),
    "cam3_prepare": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "swint.cu"),
    "cam3_paths": ("glibc_flt64.cuh",),
    "cam3_emissivity": ("glibc_flt64.cuh",),
    "gsfcsw": ("glibc_flt32.cuh",),
    "boulac": ("glibc_flt32.cuh",),


    "keps": ("glibc_flt32.cuh", "mynn_pbl.cu"),

    "level_diagnostics": ("glibc_flt32.cuh",),
    "afwa_buoyancy": ("glibc_flt32.cuh",),
    "afwa_severe": ("glibc_flt32.cuh",),
    "afwa_diagnostics": ("glibc_flt32.cuh", "log10_flt32.cuh"),
    "nwp_windows": ("glibc_flt32.cuh",),
    "native_surface_diagnostics": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "nwp_hail": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_flt64.cuh", "nwp_hail_bins.cuh"),
    "dfi_reset": ("glibc_flt32.cuh",),

    "pxsfclay": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "scalar_atan.cuh"),

    "solar_eclipse": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "wrf_stock_aer3": ("glibc_flt32.cuh", "rrtmg_aer3.cu"),
    "cldfra3": ("glibc_flt32.cuh", "wrf471_log10f.cuh"),
    "farms": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "portable_libm64.cuh", "wrf471_log10f.cuh"),
    "global_ideal_init": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "global_polar": ("glibc_trig_flt32.cuh",),
    "global_hydrostatic": ("glibc_flt32.cuh",),
    "global_rotation": ("glibc_trig_flt32.cuh",),
    "global_geometry": ("glibc_flt32.cuh",),
    "idealized_forcing": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "trajectory": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_trig_flt32.cuh"),
    "tracer_pblmix": ("mynn_scalar_mix.cu",),
    "wind_mav": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "portable_libm64.cuh", "glibc_flt64.cuh"),
    "wind_legacy": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "gsd_helicity": ("gsd_helicity_helpers.cuh",),
    "gsd_diagnostics": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_flt64.cuh", "gsd_hail_bins.cuh"),
    "hail": ("hailcast_support.cuh", "hailcast_wrf.cuh", "hailcast_wrf39.cuh"),
    "thompson_column": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                        "thompson_aerosol_libm.cuh", "thompson_column_support.cuh", "thompson_column_radar.cuh",
                        "thompson_column_generated.cuh"),
    "temf": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "temf_sfclay": ("glibc_flt32.cuh", "temf_log10_flt32.cuh"),
    "acoustic": ("glibc_trig_flt32.cuh",),
    "lin_refl": ("refl.cu",),
    "lin": ("glibc_flt32.cuh", "lin_support.cuh", "lin_wrf.cuh"),
    "goddard": ("glibc_flt32.cuh", "portable_libm64.cuh", "goddard_bands.cuh", "goddard_sounding.cuh"),
    # WOOF float32 math for the WRF aerosol column arithmetic.
    "aer_opt2": ("glibc_flt32.cuh",),
    "satmedmf": ("glibc_flt32.cuh", "glibc_flt64.cuh", "gfs_physcons.cuh", "satmedmf_common.cuh", "satmedmf_column.cuh"),
    "sstskin": ("glibc_flt32.cuh",),
    "upper_wind_limiter": ("glibc_flt32.cuh",),
    # WRF's ocean mixed layer (module_sf_oml.F) takes FSQRT from here.
    "oml": ("glibc_flt32.cuh",),
    # Reuse the scalar high-order helpers without moving the order-3 unit.
    "pd_vertical_sl": ("pd_advection.cu",),
    "sfire_coupling": ("glibc_trig_flt32.cuh",),
    "sfire_ideal": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "sfire_libm.cuh"),
    "sfire_ideal_atmos": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    # Reuse the existing FRH2O device function for cold-start soil water.
    # noah.cu needs the glibc header for its sea-ice column (below).
    "noah_init": ("glibc_flt32.cuh", "noah.cu"),
    "pxlsm_init": ("glibc_flt32.cuh", "noah.cu"),
    "pxlsm_driver": ("glibc_flt32.cuh",),
    # noah.cu carries noah_seaice_column (WRF's seaice_noah), which calls
    # glibc's float32 expf/powf (gfk_exp, gfk_pow) as gfortran's build
    # does.  The ordinary land column calls none of them.
    "noah": ("glibc_flt32.cuh",),
    "noah_seaice_options": ("glibc_flt32.cuh", "noah_seaice_options.cuh", "noah.cu"),
    "noah_fractional": ("glibc_flt32.cuh",),

    "noah_ua_sfcdiags": ("glibc_flt32.cuh",),
    "noah_ua": ("glibc_flt32.cuh", "noah_ua_plain.cuh", "noah_mosaic.cu", "noah_ua.cuh", "noah.cu"),
    "noah_mosaic_ua": ("glibc_flt32.cuh", "noah_ua.cuh", "noah_mosaic.cu"),
    "horizontal": ("portable_libm64.cuh",),
    "portable_libm64_grade": ("portable_libm64.cuh",),
    "vert_interp": ("glibc_flt32.cuh",),
    "thompson_collision": ("glibc_flt32.cuh", "glibc_flt64.cuh", "thompson_classic_constants.cuh", "thompson_collision_constants.cuh"),
    "thompson_cold_start": ("glibc_flt32.cuh", "portable_libm64.cuh"),
    "chem_prep": ("glibc_flt32.cuh",),
    "chem_dust": ("glibc_flt32.cuh",),
    "chem_seasalt": ("glibc_flt32.cuh",),
    "chem_rrtmgp_aerosol": ("glibc_flt32.cuh",),
    "chem_sulfur": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "chem_ageing": ("glibc_flt32.cuh",),
    "chem_settling": ("glibc_flt32.cuh",),
    "chem_drydep_gocart": ("glibc_flt32.cuh",),
    "chem_optics": ("glibc_flt32.cuh",),
    # The mp=28 units evaluate WRF's EXP/LOG/LOG10/** through WOOF's own
    # float32 and binary64 libm words (thompson_aerosol_libm.cuh),
    # measured equal to the gfortran oracle host's words; CUDA's
    # builtins are different functions and left no column of the 0 ULP
    # column oracle bit-identical (tools/thompson_aerosol_column_oracle).
    "thompson_aerosol_probe": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_state": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_sat": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_cold": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_warm": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_sed": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    # The LW solver derives the Planck sources itself instead of loading
    # what rrtmgp_planck_sources wrote; the helpers it needs live in a
    # header whose every FP op is pinned (HOWTO 13.6j).  rrtmgp_gas is
    # deliberately NOT listed -- it keeps its own helpers verbatim so its
    # assembled source and PTX stay byte-identical.
    # The strict process adds the exact chain's float32 libm header; a default
    # process composes rrtmgp_rte from rrtmgp_planck_common_default.cuh
    # (_extra_header_text), byte-identical to 2.9.1's helpers.
    "rrtmgp_rte": ("rrtmgp_libm.cuh", "rrtmgp_planck_common.cuh"),
    # glibc 2.39's own float32 expf/logf/powf words, which every kernel
    # graded bitwise against a gfortran oracle must call instead of CUDA's
    # builtins.  gf.cu owned the only copy until New Tiedtke (cu_physics=16)
    # needed the same three functions; the alternative was a THIRD
    # transcription beside gf's gfk_* and noahmp_leaves.cu's r_log/r_exp/
    # r_pow.  Listing gf here costs it the by-construction inertness the
    # unlisted modules keep, so it is bought with a measurement instead:
    # the lift leaves all seven gf entry points at byte-identical
    # local_size_bytes/num_regs/const_size_bytes, and the gf parity suites
    # still grade at max_ulp 0.  See glibc_flt32.cuh's header.
    # Noah mosaic uses scalar glibc float32 words for the WRF column oracle.
    "noah_mosaic": ("glibc_flt32.cuh",),
    # RUC: every EXP, LOG, LOG10, TANH and REAL**REAL of module_sf_ruclsm.F
    # takes WOOF's float32 words (gfk_* here, the log10/expm1/tanh routines in
    # ruc.cu) and the soil-resistance COS takes glibc_cosf, graded bitwise
    # against WRF v4.6.1 by tools/ruc_lsm_gpu_oracle.
    "ruc": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "lake": ("glibc_flt32.cuh", "f64cr.cuh", "lake_support.cuh", "lake_wrf.cuh"),
    # Strict GF adds the WRF normalization header; a default process keeps
    # glibc_flt32.cuh alone (_extra_header_text).
    "gf": ("glibc_flt32.cuh", "gf_wrf_normalization.cuh"),
    # Ferrier-Aligo (mp_physics=5): EXP, ALOG and REAL**REAL in
    # module_mp_fer_hires.F are the float32 libm calls its WRF v4.6.1 column
    # oracle makes, graded at 0 ULP in the strict build.
    "fer_hires": ("glibc_flt32.cuh",),
    # WOOF float32 powers for WRF lightning diagnostics.
    "lightning": ("glibc_flt32.cuh",),
    # Grell-Devenyi (cu_physics=93): EXP and ALOG are WOOF's own float32
    # exp/log (gfk_exp/gfk_log), graded word for word against the gfortran
    # WRF object by tests/test_gd_cu93.py.  A new module, so no existing
    # unit moves; the shared load_module_noftz compiles it.
    "gd": ("glibc_flt32.cuh",),
    # WRF-Chem Wesely gas dry deposition: rc/depvel read exp, log and pow,
    # and glibc's float forms are what the gfortran column oracle ran.
    "chem_drydep_wesely": ("glibc_flt32.cuh",),
    # New Tiedtke: scale_fac reads log(dxref/dx), and glibc's logf is not
    # CUDA's.  Prep stage only so far; cumastrn will add exp and pow.
    "ntiedtke": ("glibc_flt32.cuh",),
    "kfcup": ("glibc_flt32.cuh",),
    "pxlsm": ("glibc_flt32.cuh",),
    # Original cu6 uses WOOF's own scalar math routines.
    "tiedtke": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                "tiedtke_support.cuh", "tiedtke_routines.cuh"),
    # Grell 3D (cu_physics=5): EXP and ALOG in cup_env/cup_axx are WOOF's own
    # float32 routines, graded word for word against the gfortran column
    # oracle tools/g3_wrf461_oracle.  A new module, so no existing unit moves.
    "g3": ("glibc_flt32.cuh",),
    # Betts-Miller-Janjic (cu_physics=2): EXP, LOG, real ** real and GAMMA
    # in module_cu_bmj.F are WOOF's own float32 routines, graded at 0 ULP
    # against the gfortran WRF v4.6.1 column oracle
    # (tools/bmj_wrf461_oracle).  A new module, so no existing unit moves.
    "bmj": ("glibc_flt32.cuh",),

    # Previous KF (cu_physics=99) uses these EXP/ALOG/REAL**REAL helpers
    # for the strict WRF column proof. The fast default selects CUDA's
    # standard functions; the strict define is part of the image key.
    "kf_old": ("glibc_flt32.cuh",),
    # The single-layer urban canopy model (sf_urban_physics=1): EXP, ALOG
    # and every REAL**REAL in module_sf_urban.F are glibc calls in its WRF
    # v4.7.1 column oracle, graded bitwise.
    "urban_ucm": ("glibc_flt32.cuh",),
    "urban_ucm480": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                     "urban_ucm.cu", "urban_ucm480_support.cuh", "urban_ucm480.cuh"),
    # Urban BEP (sf_urban_physics 2/3), graded bitwise against gfortran/glibc
    # WRF v4.7.1 column oracles: the column uses logf/powf and six trig
    # functions (glibc_trig_flt32.cuh), the surface coupling uses powf.
    "urban_bep": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "urban_bep_couple": ("glibc_flt32.cuh",),
    # MYJ under BEP (module_bl_myjurb.F), graded against gfortran/glibc:
    # EXP and REAL powers are glibc's expf/powf.
    "myjurb": ("glibc_flt32.cuh",),
    # MYJ PBL (module_bl_myjpbl.F): PRODQ2's two EXPs and the sea-QSFC EXP
    # are gfk_exp_fma (the contracted expf, flt32_expf_fma.cuh), THSK/
    # EXNSFC's (1.E5/PSFC)**CAPA is gfk_pow, graded bitwise against the
    # gfortran WRF v4.6.1 column oracle by tests/test_myjpbl_wrf461_parity.py
    # and over every float32 input by tools/myjpbl_wrf461_oracle/libm_sweep.py.
    "myjpbl": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    # QNSE PBL and its EDMF mass flux (bl_pbl_physics=4, mfshconv=1),
    # graded word for word against the gfortran WRF v4.6.1 column oracle
    # (tools/qnse_wrf461_oracle): EXP, LOG and every REAL**REAL but 2.0 are
    # WOOF's own gfk_exp/gfk_log/gfk_pow.  A new module.
    "qnsepbl": ("glibc_flt32.cuh",),
    # The QNSE surface layer (module_sf_qnsesfc.F, sf_sfclay_physics=4):
    # LOG is gfk_log, EXP gfk_exp_fma (the contracted form an FMA host's
    # expf evaluates) and REAL**REAL gfk_pow, graded word for word against
    # the WRF v4.6.1 column oracle by tests/test_qnsesfc_wrf461_parity.py.
    # A new module, so no existing unit moves.
    "qnsesfc": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    # topo_wind arm (ysu_column_topo): glibc powf for the paj TKE profile
    # and the Beljaars convective velocity, as bl_ysu.F90 evaluates them,
    # and the arm's own pieces (get_pblh, the 10 m blend), kept out of
    # ysu_default.cu so its cited line numbers stand.
    "ysu": ("glibc_flt32.cuh", "ysu_topo_default.cuh"),
    # Shin-Hong takes glibc's powf/expf (gfk_pow/gfk_exp) since
    # lane/parity-pbl-libm, graded by tests/test_shinhong_wrf461_parity.py.
    "shinhong": ("glibc_flt32.cuh",),
    # Corrected generation reuses WOOF's own float32 math and the qualified
    # YSU driver arithmetic. The legacy module retains its original source.
    "ysu_480": ("glibc_flt32.cuh", "ysu_480_topo.cuh"),
    # The UW moist-turbulence PBL (bl_pbl_physics=9) computes in binary64
    # like the CAM code it transcribes: glibc's own binary64 exp/log/pow,
    # the rounding-pinned R8 vocabulary, then the CAM modules in call order
    # (saturation lookups, the implicit diffusion solver, exacol/zisocl/
    # compute_cubic, caleddy, compute_eddy_diff with trbintd/sfdiag and the
    # camuwpbl column driver).  A new module, so no existing unit moves.
    "uwpbl": ("glibc_flt64.cuh", "uwpbl_common.cuh", "uwpbl_wvsat.cuh",
              "uwpbl_vdiff.cuh", "uwpbl_zisocl.cuh", "uwpbl_caleddy.cuh",
              "uwpbl_eddy.cuh", "uwpbl_driver.cuh"),
    # GFS SAS (cu_physics = 94, module_cu_sas.F) computes in binary64
    # (kind_phys) and is graded word for word against the built WRF trees
    # (tools/sas_wrf461_oracle): its funcphys tables and the scheme's EXP and
    # REAL powers call WOOF's own binary64 exp and pow from this header.  A
    # new module, so no existing unit moves.
    "gfs_sas": ("glibc_flt64.cuh",),

    # CAM5 Zhang-McFarlane deep convection (cu_physics = 7), graded word for
    # word against its WRF v4.6.1 gfortran column oracle
    # (tools/camzm_wrf461_oracle): binary64 like the CAM code, so it lists
    # the shared CAM5 layer -- WOOF's own binary64 exp/log/pow, the
    # rounding-pinned R8 vocabulary and wv_saturation lookups the UW PBL
    # established, and cam5_support.cuh (cldwat_fice, the WRF-facing float32
    # vocabulary) -- plus WOOF's own float32 expf/powf for get_tpert, which
    # works in WRF's default REAL.  A new module, so no existing unit moves.
    "camzm": ("glibc_flt32.cuh", "glibc_flt64.cuh", "uwpbl_common.cuh",
              "uwpbl_wvsat.cuh", "cam5_support.cuh"),
    # The UW shallow cumulus (shcu_physics=2) is the same CAM5 suite: it
    # reuses the UW PBL port's binary64 exp/log/pow, its rounding-pinned
    # R8 vocabulary, workspace pool and CAM constants, and its saturation
    # lookups (estblf, fqsatd), then adds CAM's erfc and the scheme's own
    # column.  A new module, so no existing unit moves.
    "uwshcu": ("glibc_flt64.cuh", "uwpbl_common.cuh", "uwpbl_wvsat.cuh",
               "cam5_error_function.cuh", "uwshcu_cam.cuh", "uwshcu_column.cuh"),
    # WRF swint_opt = 1 (module_radiation_driver.F radconst/calc_coszen,
    # update_swinterp_parameters, interp_sw_radiation of the operational
    # HRRR fork): LOG and ** are glibc's logf/powf, SIN/COS/ASIN glibc's
    # sinf/cosf/asinf, graded bitwise against the fork's gfortran/glibc
    # Fortran by tests/test_swint_interpolation.py.  A new module, so no
    # existing unit moves.
    "rrtmg_legacy_solar": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "swint": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    # WRF aer_opt = 3 shortwave optics (gt_aod, calc_aerosol_rrtmg_sw of
    # the operational HRRR fork): EXP is glibc's expf, graded bitwise
    # against the fork's gfortran/glibc Fortran by
    # tests/test_rrtmg_aerosol_optics.py.  A new module.
    "rrtmg_aer3": ("glibc_flt32.cuh",),
    # Kessler warm rain (mp_physics=1): its six REAL**REAL powers and its
    # one EXP take WOOF's own float32 gfk_pow/gfk_exp, graded bitwise
    # against the WRF v4.6.1 column oracle by
    # tests/test_kessler_wrf461_parity.py.  With CUDA's powf/expf (and the
    # old qr shortcut) the strict build missed it by up to 42 ULP in qv and
    # 1 ULP in theta (lane/verify-kessler-mp1).
    "kessler": ("glibc_flt32.cuh",),
    # Dudhia shortwave (ra_sw_physics=1): REAL**0.635 is WOOF's own gfk_pow
    # and ALOG10 builds on gfk_log, graded bitwise against the WRF v4.6.1
    # column oracle by tests/test_dudhia_sw_wrf461_parity.py.  A new
    # module, so no existing unit moves.
    "dudhia_sw": ("glibc_flt32.cuh",),
    # Milbrandt-Yau (mp_physics = 9), graded bitwise against the WRF v4.6.1
    # gfortran column oracle (tools/milbrandt2_wrf461_oracle,
    # tests/test_milbrandt2_wrf461_exact_gpu.py): every EXP, LOG, LOG10 and
    # REAL**REAL in module_mp_milbrandt2mom.F is WOOF's own gfk_exp/gfk_log/
    # gfk_log10/gfk_pow, not CUDA's.  The radar operator's Z block is the
    # scheme's own and takes the same log10.
    "milbrandt2": ("glibc_flt32.cuh", "log10_flt32.cuh"),
    "milbrandt2_zet": ("glibc_flt32.cuh", "log10_flt32.cuh"),
    "rrtmg_aer1": (),
    # The GFS surface layer (sf_sfclay_physics=3, WRF v4.6.1
    # module_sf_gfs.F) computes in binary64 like the kind_phys column it
    # transcribes: EXP, LOG and every REAL power (PROGTM, FPVSX) are the
    # binary64 routines graded bitwise against its gfortran oracle
    # (tools/sfgfs_wrf461_oracle).  A new module, so no existing unit moves.
    "sf_gfs": ("glibc_flt64.cuh",),
    "real_init": ("real_init_common.cuh",),
    # The GFS nonlocal PBL (bl_pbl_physics=3): WOOF's own binary32 powf and
    # binary64 pow (the two real powers in BL_GFS), then the constants
    # generated from the pinned WRF source (tools/gfs_pbl_wrf461_oracle/
    # gen_constants.py).  A new module, so no existing unit moves.
    "gfs_pbl": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                "../../data/gfs_pbl/gfs_pbl_constants.cuh"),
    # REAL's float64 thermodynamics use the CPU portable library's bits.
    "real_init_math": ("real_init_common.cuh", "portable_libm64.cuh"),
    # Kain-Fritsch (cu_physics=1): EXP, ALOG and REAL**REAL in
    # module_cu_kfeta.F are glibc's expf/logf/powf in its WRF 4.6.1 column
    # oracle (tools/kf_wrf461_oracle), graded at 0 ULP by
    # tests/test_kf_wrf461_oracle.py.
    "kf": ("glibc_flt32.cuh",),
    "mrf": ("glibc_flt32.cuh",),
    # SBU-YLin (mp_physics=13), graded bitwise against its gfortran WRF
    # v4.6.1 column oracle (tools/sbu_ylin_wrf461_oracle): every EXP and
    # REAL**REAL in module_mp_sbu_ylin.F is a float32 libm call there, so the
    # column calls WOOF's own gfk_exp/gfk_pow words.  A new module.
    "sbu_ylin": ("glibc_flt32.cuh",),
    "sbu_ylin_edge": ("glibc_flt32.cuh",),
    # ISHMAEL uses WOOF math and pinned compile-time constants.
    "ishmael": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "glibc_flt64.cuh",
                "flt32_log10_tanh.cuh", "ishmael_folded.cuh"),
    # WRF cu_physics = 95 (module_cu_osas.F) computes in binary64
    # (kind_phys): WOOF's own binary64 exp/log/pow and the rounding-pinned
    # R8 vocabulary, graded word for word against its gfortran WRF oracle
    # (tools/osas_wrf461_oracle).  A new module, so no existing unit moves.
    "cu_osas": ("glibc_flt64.cuh", "column_r8.cuh", "cu_osas_table_seeds.cuh"),
    # SSiB land surface (sf_surface_physics=8), graded bitwise against the
    # gfortran WRF v4.6.1 column oracle: EXP/ALOG/REAL**REAL are the float32
    # words, and the handful of DOUBLE PRECISION literals module_sf_ssib.F
    # mixes in (COMPACT, GETMET, SDSOL) make binary64 exp and pow calls.
    "ssib": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    "chem_fire": ("glibc_flt32.cuh",),
    # The slab land surface (sf_surface_physics=1): SLAB1D's EXP and
    # REAL**ROVCP are WOOF's own gfk_exp/gfk_pow words, graded word for
    # word against the gfortran -O0 module (tools/slab_wrf461_oracle).  A
    # new module, so no existing unit moves.
    "slab": ("glibc_flt32.cuh",),
    # NTU native WRF 4.7.1 arithmetic; strict first-call replay is qualified.
    "ntu": ("glibc_flt32.cuh", "glibc_flt64.cuh", "glibc_trig_flt32.cuh"),
    "chem_plumerise": ("glibc_flt32.cuh",),
    "wsm3": ("glibc_flt32.cuh",),
    # The km_opt=2/3 path (tke_km, calc_l_scale, tke_dissip, calculate_N2 and
    # phy_prep's p8w/t8w) takes REAL**REAL, EXP and LOG as gfk_pow/gfk_exp/
    # gfk_log, graded bitwise by tools/tke_km2_wrf461_oracle against the WRF
    # v4.6.1 Fortran.  CUDA's powf/expf/logf flipped calculate_N2's saturated
    # predicate there and moved the TKE coefficients by up to 34,746 ULP.
    "smag2d": ("glibc_flt32.cuh",),
    "myjsfc": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    "mynn_surface": ("mynn_libm.cuh", "surface_subnormal.cuh"),
    "sfclay": ("glibc_flt32.cuh", "sfclay_classic.cuh"),
    "sfclay_dx2d": ("glibc_flt32.cuh", "sfclay_classic.cuh", "sfclay_dx2d.cuh", "sfclay.cu"),
    # WSM6 (mp_physics = 6), graded word for word against its gfortran
    # column oracle by tests/test_wsm6_wrf461_column_exact.py: EXP, LOG and
    # REAL**REAL of mp_wsm6.F90 and mp_wsm6_effectRad.F90 are WOOF's own
    # float32 words (gfk_exp/gfk_log/gfk_pow), and refl10cm_wsm6's binary64
    # EXP, LOG and ** are WOOF's own binary64 words (glibc_flt64.cuh).
    "wsm6": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # WDM6 (mp_physics=16): WOOF's own float32 exp/log/pow (gfk_exp,
    # gfk_log, gfk_pow), graded word for word against WRF v4.6.1's
    # module_mp_wdm6.F by the column oracle tools/wdm6_wrf461_oracle
    # (tests/test_wdm6_wrf461_oracle.py).
    "wdm6": ("glibc_flt32.cuh",),
    "wdm5": ("glibc_flt32.cuh", "wdm_shared.cuh", "wdm_common.cuh", "wdm_family_refl.cuh"),
    "wdm7": ("glibc_flt32.cuh", "wdm_shared.cuh"),
    "wdm7_refl": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # WDM6's 10 cm reflectivity (refl10cm_wdm6 with the radar module's
    # melting-particle routines): WOOF's own float32 exp/pow and binary64
    # exp/log/pow (g64_exp, g64_log, g64_pow), graded against the same
    # oracle.
    "wdm6_refl": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    'nsas': ('glibc_flt32.cuh', 'nsas_constants.cuh'),
    # WSM5 (mp_physics = 4), graded word for word against the WRF v4.6.1
    # build trees' own objects by tests/test_wsm5.py: EXP, LOG and
    # REAL**REAL of module_mp_wsm5.F are WOOF's own float32 words (gfk_exp/
    # gfk_log/gfk_pow), and refl10cm_wsm5's binary64 EXP, LOG and ** are
    # WOOF's own binary64 words (glibc_flt64.cuh).  A new module.
    "wsm5": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # WSM6R (mp_physics = 106), graded word for word against its gfortran
    # column oracle by tests/test_wsm6r_wrf461_column_exact.py: EXP, LOG and
    # REAL**REAL of module_mp_wsm6r.F are WOOF's own float32 words
    # (gfk_exp/gfk_log/gfk_pow), and smoothif's binary64 EXP is WOOF's own
    # binary64 word (glibc_exp).  A new module, so no existing unit moves.
    "wsm6r": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # WSM7 (mp_physics = 24), graded word for word against its gfortran
    # column oracle by tests/test_wsm7_wrf461_column_exact.py: EXP, LOG and
    # REAL**REAL of module_mp_wsm7.F are WOOF's own float32 words
    # (gfk_exp/gfk_log/gfk_pow), and refl10cm_wsm7's binary64 EXP, LOG and
    # ** are WOOF's own binary64 words (glibc_flt64.cuh).  A new module, so
    # no existing unit moves; it compiles through woof/core/wsm_family.py.
    "wsm7": ("glibc_flt32.cuh", "glibc_flt64.cuh"),
    # The exact chain's units (intake 2.9.1).  physics_exner is the strict
    # moist_physics_prep_em Exner; thompson_classic and thompson_fork are the
    # strict mp=8 and mp=28 wrf_39_noaa transcriptions; rrtmgp_gas and the
    # nssl2 units take these headers only in a strict process.
    "physics_exner": ("glibc_flt32.cuh",),
    "topo_radiation": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh",
                       "atan_scalar_flt32.cuh"),
    "thompson_classic": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh",
                         "glibc_flt64.cuh", "thompson_classic_constants.cuh"),
    "morrison": ("glibc_flt32.cuh", "glibc_flt32_extra.cuh", "glibc_flt64.cuh",
                 "glibc_trig_flt32.cuh", "morrison_constants.cuh"),
    "gsd_initial_reflectivity": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                                 "thompson_fork_constants.cuh", "thompson_fork.cu"),
    "thompson_fork": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                      "thompson_fork_constants.cuh"),
    "nssl2_nucond": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_fused_gs": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_driver_support": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_diagnostics": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_shape_nucond": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_shape_gs": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_shape_driver": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),
    "nssl2_shape_diagnostics": ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"),

    "rrtmgp_gas": ("rrtmgp_libm.cuh", "rrtmgp_planck_common.cuh"),
    # Goddard GCE (mp_physics=97): every REAL power and EXP in
    # module_mp_gsfcgce.F is WOOF's own float32 gfk_pow/gfk_exp, graded
    # bitwise against the WRF 4.6.1 column oracle
    # (tools/gsfcgce_wrf461_oracle).  A new module, so no existing unit moves.
    "gsfcgce": ("glibc_flt32.cuh",),
    # Its REFL_10CM reuses refl.cu's Blahak melting-particle machinery as
    # written instead of a third copy; refl.cu itself is not edited.
    "gsfcgce_refl": ("refl.cu",),
    "eeps": ("glibc_flt32.cuh",),
    # Goddard 4-ice (mp_physics=7), graded word for word against the scheme
    # strict object WRF's build produced: EXP/REAL**REAL use pinned host
    # libm; ALOG10 follows the referee's glibc generation. Production uses
    # CUDA float math; gammagce retains the pinned binary64 log.
    "gsfcgce4ice": ("glibc_flt32.cuh", "glibc_flt64.cuh", "gsfcgce4ice_log10.cuh"),
    "gsfcgce4ice_aux": ("glibc_flt32.cuh", "glibc_flt64.cuh", "gsfcgce4ice_log10.cuh",
                        "gsfcgce4ice.cu", "radar_backscatter_common.cuh"),


    # CAM5.1 Morrison-Gettelman (mp_physics=11) computes in binary64 like
    # the CAM code it transcribes, so it reuses the UW PBL's binary64
    # vocabulary unchanged: WOOF's float64 exp/log/pow/cos/acos, the
    # rounding-pinned R8 type with its column workspace, and wv_saturation's
    # estblf; then the scheme's own words, support functions, aerosol
    # activation and MG1 microphysics.  A new module, so no unit moves.
    "cammgmp": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                "uwpbl_common.cuh", "uwpbl_wvsat.cuh", "cam5_error_function.cuh",
                "cammgmp_words.cuh", "cammgmp_support.cuh",
                "cammgmp_aero.cuh", "cammgmp_micro.cuh"),
    "mskf": ("glibc_flt32.cuh", "glibc_flt64.cuh", "mskf_aerosol.cuh"),
    "mskf_aerosol_replay": ("glibc_flt32.cuh", "glibc_flt64.cuh", "mskf_aerosol.cuh"),
}

# Content-addressed scalar coefficient headers in the packaged table tree.
# Kept outside CUDA units so each numerical setup carries its own asset pin.
_TABLE_HEADERS = {
    "mrf": ("mrf/constants.cuh", 1439, "c15be2603047c7eb2c9831afc3cd5b1b83e7bd5e5cb212823bb6cee3e72ddd1e"),
}


def _table_header_text(name: str, *, data_root=None) -> str:
    """Read the pinned table bytes once before assembling a CUDA module."""
    row = _TABLE_HEADERS.get(name)
    if row is None:
        return ""
    from woof.data_assets import package_data_root
    from woof.core.thompson_contract import TableAsset, read_validated_table_asset

    relative, size, digest = row
    path = Path(relative)
    root = package_data_root() if data_root is None else Path(data_root)
    asset = TableAsset(path.name, size, digest)
    try:
        return read_validated_table_asset(root / path.parent, asset).decode("utf-8")
    except (FileNotFoundError, ValueError) as error:
        raise type(error)(
            f"{name} coefficient table is unavailable or changed; using other "
            f"bytes would change the kernel's numerical setup: {error}") from error


#: Read-only view for tests and freeze receipts.
EXTRA_HEADERS = MappingProxyType(_EXTRA_HEADERS)


def _preamble(kernel_dir: Path = _KDIR) -> str:
    lines = [f"#define {k} {float(v)!r}f" for k, v in CUDA_DEFINES.items()]
    lines.append((Path(kernel_dir) / "common.cuh").read_text(encoding=_ENCODING))
    return "\n".join(lines) + "\n"


def _sas_strict_numerics() -> bool:
    from woof.numerics import mode
    from woof.wrf_exact import ENABLED
    return ENABLED or mode() == "strict"


def _flg_fast_math_aliases(*, rounded_operators: bool = False) -> str:
    """Fast float intrinsics; the column solve retains rounded operators."""
    elementary = ("\n#define gfk_exp expf\n#define gfk_log logf\n#define gfk_pow powf\n#define gfk_sqrt sqrtf\n"
                  "#define glibc_sinf sinf\n#define glibc_cosf cosf\n")
    if rounded_operators:
        return elementary
    return (elementary + "#define __fadd_rn(a,b) ((a)+(b))\n#define __fsub_rn(a,b) ((a)-(b))\n"
                         "#define __fmul_rn(a,b) ((a)*(b))\n#define __fdiv_rn(a,b) ((a)/(b))\n")


def _flg_column_noinline(text: str) -> str:
    return text.replace("__device__ void flg", "__device__ __noinline__ void flg")


def _sfs_smag_geometry_source(text: str) -> str:
    """The SFS diagnostic's exact, shared geometry-only header composition."""
    start = text.index("struct WrfSmagGrid {")
    end = text.index('extern "C" __global__', start)
    shear = text.index("real wrf_w_xavg(")
    shear = text.rfind("__device__", 0, shear)
    shear_end = text.index("real wrf_tau13(", shear)
    shear_end = text.rfind("__device__", 0, shear_end)
    return "typedef size_t SmagIndex;\n" + text[start:end] + text[shear:shear_end]


def _extra_header_text(name: str, kernel_dir: Path = _KDIR) -> str:
    """Return the allow-listed headers for ``name``, or ``''`` for any other.

    The empty-string return for an unlisted module is the whole point: it
    makes the assembled source byte-identical to the pre-hook string.
    """
    headers = _EXTRA_HEADERS.get(name, ())
    from woof.wrf_exact import ENABLED, DIAGNOSTICS_ENABLED, DIFFUSION_ENABLED
    if name in ("level_diagnostics", "nwp_windows", "afwa_diagnostics") and not ENABLED:
        headers = ()
    if name == "nwp_hail" and not ENABLED:
        headers = ("nwp_hail_bins.cuh",)
    # A default process composes these units from their *_default sources
    # (see _DEFAULT_SOURCE_MODULES), the 2.9.1 default arithmetic; the exact
    # chain's strict sources and headers are a GPUWM_WRF_EXACT=1 process's.
    if not ENABLED and name in ("rrtmgp_gas", "rrtmgp_rte", "rrtmgp_cloud"):
        headers = {'rrtmgp_gas': (), 'rrtmgp_rte': ('rrtmgp_planck_common_default.cuh', 'rrtmgp_planck_staged_default.cuh'), 'rrtmgp_cloud': ()}[name]
    if name == "sas" and not _sas_strict_numerics():
        headers = ("sas_fast.cuh",)
    if not ENABLED and name == "gf":
        headers = ('glibc_flt32.cuh',)
    if not ENABLED and name in ('nssl2_nucond', 'nssl2', 'nssl2_fused_gs', 'nssl2_driver_support', 'nssl2_diagnostics'):
        headers = ()
    if DIAGNOSTICS_ENABLED and name == "diagnostics":
        headers += ("glibc_flt32.cuh",)
    if ENABLED and name == "tslist":
        headers += ("glibc_flt32.cuh",)
    parts = []
    if ENABLED and name in ("cldfra3", "farms"):
        from woof.wrf_exact import referee_glibc
        if referee_glibc() < (2, 41):
            # The strict GNU 13/Ubuntu 24.04 referee uses the older
            # binary32 LOG10, which differs from correctly rounded
            # CORE-MATH. Keep ordinary forecasts' source unchanged.
            headers = tuple("log10_flt32.cuh" if h == "wrf471_log10f.cuh"
                            else h for h in headers)
            parts.append("#define wrf471_log10f gfk_log10\n")
    if "gfdleta_support.cuh" in headers:
        from woof.wrf_exact import referee_glibc
        # The checked synthetic fixtures use 2.39; the real-input proof
        # uses 2.43. Keep their measured LOG10 bindings distinct.
        parts.append(f"#define GPUWM_GFDLETA_GLIBC243 {int(referee_glibc() >= (2, 43))}\n")
    for header in headers:
        path = Path(kernel_dir) / header
        if header.startswith("../../data/"):
            from woof.table_assets import read_table_header
            parts.append(read_table_header(path))
        else:
            header_text = path.read_text(encoding=_ENCODING)
            if name == "sfs_output" and header == "smag2d.cu":
                # Reuse the production geometry verbatim without compiling
                # every diffusion entry point for a diagnostic-only module.
                # The complete header still binds the retained-image key.
                header_text = _sfs_smag_geometry_source(header_text)
            if not ENABLED and name == "flg_column" and header == "flg_workspace_generated.cuh":
                # The IR instability belongs to the stream solve. Keep
                # its rounded operators, while allowing ordinary fast
                # algebra in gas/cloud optics and column bookkeeping.
                stream = "// Generated from pristine WRF FLG stream solver statements."
                cloud = "// Generated from WRF module_ra_flg.F SHA256"
                if header_text.count(stream) != 1 or header_text.count(cloud) != 1:
                    raise ValueError("FLG stream rounding boundary changed")
                rounded = "\n#undef __fadd_rn\n#undef __fsub_rn\n#undef __fmul_rn\n#undef __fdiv_rn\n"
                header_text = ( _flg_fast_math_aliases() + header_text.replace(
                    stream, rounded + stream, 1).replace(
                    cloud, _flg_fast_math_aliases() + cloud, 1))
            parts.append(header_text)
        if not ENABLED and name == "flg_driver" and header == headers[-1]:
            # Packing retains its ordinary arithmetic.
            parts.append(_flg_fast_math_aliases())
        if not ENABLED and name == "flg_column" and header == "glibc_flt32.cuh":
            # Insert before the generated functions, not after their bodies.
            # Fast float elementary functions are safe with the stream
            # solver's separately rounded operators. Removing those
            # roundings destabilizes the real 49-level clear-sky IR solve.
            parts.append(_flg_fast_math_aliases(rounded_operators=True))
    text = "".join(parts)
    if name == "flg_column":
        # Keep the complete column call tree from being inlined into each
        # stream arm. This affects only this unit and retains arithmetic.
        text = _flg_column_noinline(text)
    if name == "sas" and _sas_strict_numerics():
        from woof.wrf_exact import referee_correctly_rounded_trig
        if referee_correctly_rounded_trig():
            text += "\n#undef wrf_atan_f32\n#define wrf_atan_f32 glibc_atanf\n"
    if ENABLED and name == "cu_osas":
        text = "#define GPUWM_COLUMN_R8_STRICT 1\n" + text
    if not ENABLED and name in ("wind_mav", "wind_legacy"):
        text += "#define GPUWM_WIND_FAST 1\n#define glibc_sinf sinf\n#define glibc_cosf cosf\n#define glibc_asinf asinf\n#define glibc_acosf acosf\n#define gfk_exp expf\n#define plm_exp exp\n#define uw_acos acos\n"
    if not ENABLED and name == "kfcup":
        # Default forecasts use REAL32 CUDA arithmetic, including constant
        # divisor folding and contraction. Only strict native replay needs
        # the explicitly rounded operators and software elementary functions.
        text += ("#define gfk_exp expf\n#define gfk_log logf\n#define gfk_pow powf\n"
                 "#define __fadd_rn(a,b) ((a)+(b))\n#define __fsub_rn(a,b) ((a)-(b))\n"
                 "#define __fmul_rn(a,b) ((a)*(b))\n#define __fdiv_rn(a,b) ((a)/(b))\n")
    if not ENABLED and name == "smag2d":
        text = _smag2d_default_gfk_exp(text)
    if not ENABLED and name in ("ssib", "nssl2", "nssl2_shape_nucond", "nssl2_shape_gs", "nssl2_shape_driver", "nssl2_shape_diagnostics"):
        text += "#define gfk_pow powf\n#define gfk_exp expf\n#define gfk_log logf\n"
    # The NSSL shape units keep their __fdiv_rn spellings in every mode, as
    # nssl2.cu does: a fast alias to "/" let NVRTC turn constant divisors
    # into inexact reciprocal multiplies on compute_120 (A146).
    if name == "temf_sfclay":
        from woof.core.temf_constants import (cuda_table, table_sha256,
                                               TEMF_CONSTANTS_SHA256)
        if table_sha256() != TEMF_CONSTANTS_SHA256:
            raise ValueError("TEMF constant table hash mismatch")
        text += cuda_table()
    if name == "gfs_sas":
        from woof.numerics import is_strict
        text = f"#define SAS_STRICT {int(is_strict())}\n" + text
    if name in ("gsfcgce4ice", "gsfcgce4ice_aux"):
        from woof.wrf_exact import referee_glibc
        if referee_glibc() >= (2, 41):
            text = "#define GPUWM_G4_CORE_LOG10 1\n" + text
    if name == "keps":
        from woof.table_assets import kernel_parameter_source
        # Explicit kernel trees own their parameter sources as well.
        data_root = (None if Path(kernel_dir).resolve() == _KDIR.resolve()
                     else Path(kernel_dir).resolve().parents[1] / "data")
        text += kernel_parameter_source(name, data_root=data_root)
    data_root = (None if Path(kernel_dir).resolve() == _KDIR.resolve()
                 else Path(kernel_dir).resolve().parents[1] / "data")
    if ENABLED and name == "tslist":
        text += "#define TSLIST_LOG gfk_log\n#define TSLIST_EXP gfk_exp\n"
    return text + _table_header_text(name, data_root=data_root)


#: The exact chain FMA-fused glibc_flt32.cuh's gfk_exp (its comment block
#: starts with the first line below and the fused body ends with the second).
_GFK_EXP_FMA_FIRST = "    // The reference expf on an x86-64 host with FMA is the FMA build of this\n"
_GFK_EXP_FMA_LAST = "    return gfk_d2f_rn(DMUL(y, s));\n"
#: 2.9.1's separately rounded gfk_exp body, which smag2d's default
#: composition keeps.
_GFK_EXP_291 = (
    "    double xd = (double)x;\n"
    "    double z = DMUL(0x1.71547652b82fep+0 * 32.0, xd);\n"
    "    return gfk_d2f_rn(gfk_exp2_core(\n"
    "        z, GFK_EXP2F_SHIFT,\n"
    "        GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0,\n"
    "        GFK_EXP2F_P1 / 32.0 / 32.0,\n"
    "        GFK_EXP2F_P2 / 32.0, 0ULL));\n")


def _smag2d_default_gfk_exp(text: str) -> str:
    """smag2d's default headers with 2.9.1's gfk_exp body, byte for byte.

    THE BREAKAGE THIS PREVENTS (2.9.1 integration): the exact chain
    FMA-fused gfk_exp in glibc_flt32.cuh, which moved smag2d's composed
    source; the GPU-measured assembled coordinate and driver proofs
    (tests/assembled_legacy_proofs.py) seal that source by hash on a card
    class the fleet no longer has.  A default process composes smag2d with
    the 2.9.1 body (the proofs' bytes); a strict process takes the chain's.
    The rewrite refuses rather than guess if the header no longer carries
    exactly one FMA block.
    """
    if text.count(_GFK_EXP_FMA_FIRST) != 1 or text.count(_GFK_EXP_FMA_LAST) != 1:
        raise RuntimeError("glibc_flt32.cuh no longer carries exactly one fused gfk_exp "
                           "block; smag2d's default composition cannot keep its sealed bytes")
    start = text.index(_GFK_EXP_FMA_FIRST)
    end = text.index(_GFK_EXP_FMA_LAST, start) + len(_GFK_EXP_FMA_LAST)
    return text[:start] + _GFK_EXP_291 + text[end:]


#: Units whose strict (GPUWM_WRF_EXACT=1) source is ``<name>.cu`` and whose
#: default source is ``<name>_default.cu``: the exact chain rewrote these for
#: WRF's scalar arithmetic, and a default process keeps the 2.9.1 unit (for
#: ysu, the default wrf_461 generation's retained kernel).
_DEFAULT_SOURCE_MODULES = frozenset({
    "rrtmgp_gas", "rrtmgp_rte", "ysu", "nssl2_nucond",
    "nssl2_fused_gs", "nssl2_driver_support", "nssl2_diagnostics"})


def _unit_text(name: str, kernel_dir: Path = _KDIR) -> str:
    """Read a CUDA unit and apply only the active set's registered literals."""
    from woof.physics_params import edit_kernel_source
    from woof.wrf_exact import ENABLED
    source_name = "mynn_pbl" if name == "mynn_dmp_sibling" else name
    unit_path = Path(kernel_dir) / f"{source_name}.cu"
    if name == "gsfcsw":
        from woof.numerics import is_strict
        if not is_strict():
            # Same column physics with known horizontal dimensions and
            # fused independent flux-level passes; strict keeps its replay image.
            unit_path = Path(kernel_dir) / "gsfcsw_fast.cuh"
    if name == "ysu" or (not ENABLED and name in _DEFAULT_SOURCE_MODULES):
        default_path = Path(kernel_dir) / f"{name}_default.cu"
        if default_path.is_file():
            unit_path = default_path
    source = edit_kernel_source(
        name, unit_path.read_text(encoding=_ENCODING))
    if name == "slab":
        from woof.numerics import is_strict
        source = f"#define GPUWM_SLAB_STRICT {int(is_strict())}\n" + source
    if name == "ishmael":
        if ENABLED:
            from woof.wrf_exact import referee_glibc
            if referee_glibc() >= (2, 41):
                # Newer referees use CORE-MATH LOG10. The pinned GNU13 /
                # glibc2.39 referee uses the existing Sun reduction from
                # flt32_log10_tanh.cuh; forcing CORE-MATH changes radar words.
                log10 = Path(kernel_dir).parents[1] / "hydro/noahowp/log10.cu"
                source = log10.read_text(encoding=_ENCODING) + "\n#define gfk_log10 owp_log10\n" + source
        else:
            source = "#define ISHMAEL_FAST_NUMERICS 1\n" + source
    if name == "trajectory" and ENABLED:
        source = "#define WRF_EXACT 1\n" + source
    if name == "mynn_dmp_sibling":
        source = "#define MYNN_EXPORT_PLUME_EDGE 1\n" + source
    if name == "noah":
        from woof.wrf_exact import ENABLED
        if ENABLED:
            # The strict referee is built at -O2, where GNU Fortran folds
            # CANRES's **2.0 to x*x. An -O0 native build calls powf there
            # and is not the referee recipe (tools/fdda_sfc_wrf461_oracle).
            source = "#define NOAH_WRF_SCALAR_RCT 1\n" + source
            from woof.wrf_exact import referee_glibc
            if referee_glibc() >= (2, 43):
                source = "#define NOAH_WRF_MODERN_LOG10 1\n" + source
    return source


def module_source(name: str, *, kernel_dir: Path = _KDIR) -> str:
    """The exact source string :func:`load_module` hands to nvrtc.

    ``kernel_dir`` composes the same unit from another tree's kernel files:
    tools/literal_division_census.py gates the tree it scans, which need not
    be the imported package (A193).
    """
    from woof.cuda_subnormals import MARKER
    prefix = MARKER if name in _URBAN_NO_FTZ_MODULES else ""
    if name == "pxsfclay":
        from woof.core.pxsfclay import source_constant_prefix
        # Another tree's kernels take that tree's constants (A193).
        prefix += source_constant_prefix(
            None if Path(kernel_dir).resolve() == _KDIR.resolve()
            else Path(kernel_dir).resolve().parents[1] / "data" / "pxsfclay")

    if name == "deng":
        from woof.numerics import is_strict
        if not is_strict():
            prefix += "#define GPUWM_DENG_FAST 1\n"
    if name in {"mskf", "mskf_aerosol_replay"}:
        from woof.numerics import is_strict
        if not is_strict():
            prefix += "#define GPUWM_MSKF_FAST 1\n"
    return (prefix + _preamble(kernel_dir) + _extra_header_text(name, kernel_dir)
            + _unit_text(name, kernel_dir))


#: Units graded word for word against a WRF column oracle, which needs NVRTC
#: not to contract multiply-adds (WRF's reference is gfortran -O0, no FMA).
#: lake: the CLM lake oracle.  ysu and shinhong: the WRF 4.6.1 YSU and
#: Shin-Hong oracles (tests/test_ysu_wrf461_parity.py,
#: tests/test_shinhong_wrf461_parity.py), together with glibc's powf/expf
#: (gfk_pow/gfk_exp) in place of CUDA's.  mynn_pbl and its DMP sibling: the
#: WRF 4.6.1 MYNN oracles (tests/test_mynn_wrf461_exact_gpu.py), through
#: both loaders (the gsd_41 generation compiles mynn_pbl with an integer
#: define, :func:`load_module_int_defines`).  (Kain-Fritsch is graded the
#: same way but compiles on its own unflushed route, woof/core/kf.py.)
#: wsm6: the WRF 4.6.1 WSM6 column oracle
#: (tests/test_wsm6_wrf461_column_exact.py), at both WSM6_KMAX tiers.
#: kessler: the WRF 4.6.1
#: Kessler column oracle (tests/test_kessler_wrf461_parity.py); with
#: contraction on, a single call misses it in 49,927 of 130,000 qv words
#: (max 23 ULP), and in theta, qc, qr and RAINNCV.
#: myjpbl: the WRF 4.6.1 MYJ PBL
#: column oracle (tests/test_myjpbl_wrf461_parity.py).
#: milbrandt2 and its radar
#: operator milbrandt2_zet: the WRF 4.6.1 Milbrandt-Yau column oracle
#: (tests/test_milbrandt2_wrf461_exact_gpu.py), whose WRF object has no FMA
#: (baseline x86-64); with contraction the default build missed WRF in about
#: 15% of output words on the oracle columns, without it none.
#: wdm6 and wdm6_refl: the WRF
#: 4.6.1 WDM6 column oracle (tests/test_wdm6_wrf461_oracle.py), through both
#: loaders (nz above 64 compiles the WDM6_KMAX=80 tier, and reflectivity
#: compiles its REFL_KMAX tier).  noah: the exact chain's Noah words (glibc
#: transcendentals, integer powers, float32 NOAHRES) are WRF's only without
#: contraction; with it a default column read hfx 802 and noahres 606208 ULP
#: from WRF 4.6.1 and failed the WRF 4.7.1 urban hand-over oracle
#: (tests/test_noah_wrf461_parity.py, tests/test_urban_noah_hook_wrf471_parity.py).
#: climate_diagnostics: all 48 climate fields match the pinned gfortran
#: oracle only with separate rounding in strict mode; fast strips this option.
_NO_FMAD_MODULES = frozenset({"deng", "thompson_collision", "soil_points", "gsfcsw", "gsfcgce", "gsfcgce_refl", "climate_diagnostics", "hail", "lake", "g3", "ysu", "shinhong", "mynn_pbl",
                              "mynn_dmp_sibling", "wsm6", "kessler", "myjpbl",
                              "milbrandt2", "milbrandt2_zet", "wdm6",
                              "wdm6_refl", "wsm3", "wsm6r", "noah", "pxlsm", "mskf", "mskf_aerosol_replay", "cumulus_feedback", "cumulus_aerosol"})


#: The solar geometry units (WRF 4.6.1 / 4.8.0 radconst + calc_coszen and the
#: swint_opt=1 interpolation) keep subnormals but not the no-FMA request: their
#: arithmetic is written with explicit round-to-nearest intrinsics, and they
#: were graded word for word against both releases' Fortran under exactly
#: these options (tests/test_solar_generation_gpu.py,
#: tools/solar_geometry_oracle/compare_versions.py).
_SOLAR_NO_FTZ_MODULES = frozenset({"rrtmg_legacy_solar", "swint", "cldfra3", "farms"})
#: ysu_480: the WRF 4.8 YSU generation (capped top-down entrainment), graded
#: word for word against WRF 4.8 MMM 550b5b4 bl_ysu.F90 by
#: tools/ysu_generations_oracle/gate.py in default and strict arithmetic.
#: The SLUCM/BEP units (WRF 4.7.1 water and canopy, WRF 4.8 full canopy):
#: urban rain clamps and small momentum terms need IEEE subnormals.  They keep
#: the urban lanes' own route (woof.cuda_subnormals marker on RawModule) and
#: options, under which their column oracles were graded
#: (tests/test_urban_water_balance_gpu.py, tests/test_urban480_gpu_oracle.py).
_URBAN_NO_FTZ_MODULES = frozenset({"urban_ucm", "urban_ucm480", "urban_bep",
                                   "urban_bep_couple"})
# aer_opt2: WRF keeps the aerosol optics' subnormal operands (port289-final 550349628).
# oml: graded word for word against WRF 4.6.1 OML1D (tools/oml_wrf461_oracle).
_NO_FTZ_MODULES = frozenset({"deng", "grims", "shcu_nscv", "wind_mav", "wind_legacy", "gfs_pbl", "sf_gfs", "bmj", "gsfcsw", "wdm5", "wdm7", "wdm7_refl", "sbu_ylin", "sbu_ylin_edge", "kfcup", "gsfcgce", "gsfcgce_refl", "g3", "gd", "nsas", "sfclay", "myjsfc", "mynn_surface", "mynn_pbl",
                             "thompson_column", "rcon", "udm", "ysu_480", "wsm6r",
                             "aer_opt2", "oml", "kf_old", "clwrf", "fogdes"}) | _SOLAR_NO_FTZ_MODULES

# The exact chain's strict intake contracts (GPUWM_WRF_EXACT=1): these units
# also compile without contraction in a strict process.  A default process
# keeps the grants above, so its options are 2.9.1's.
_STRICT_NO_FMAD_MODULES = frozenset({"udm", "rcon","ssib", "wdm5", "wdm7", "wdm7_refl", "morrison", "rrtmgp_rte", "noah_init", "pxlsm_init", "pxlsm_driver",
    "wdm6", "milbrandt2_zet", "wdm6_refl", "rrtmgp_gas", "milbrandt2",
    "rrtmgp_cloud", "noah", "myjsfc", "kessler", "thompson_classic",
    "myjpbl", "wsm6", "wind_farm"})


def _needs_no_fmad(name):
    from woof.wrf_exact import ENABLED
    return name in _NO_FMAD_MODULES or (ENABLED and name in _STRICT_NO_FMAD_MODULES)


_GFDLETA_MODULES = frozenset(('gfdleta_init', 'gfdleta_lw_helpers', 'gfdleta_tables', 'gfdleta_column', 'gfdleta_math_probe', 'gfdleta_co2_init', 'gfdleta_table_runtime'))

# A146: these port units contain constant divisions that NVVM rewrites to
# rounded reciprocal multiplies under CuPy's appended -ftz=true. Use direct
# NVRTC in both ordinary and integer-tier loads. Preserve each unit's FMA
# policy; disabling FTZ alone keeps IEEE division on Blackwell. CAM's
# emissivity header shares this route so its binary64 expressions and dead
# float branch need no source-scan exception based on a particular compiler.
_PORT_IEEE_DIVISION_MODULES = _GFDLETA_MODULES | frozenset({
    "afwa_buoyancy", "afwa_severe", "misc_diag_print", "snow_papers",
    "cam3_emissivity", "eeps", "gsfcgce4ice_aux", "qnsepbl", "qnsesfc",
    "sas", "sstskin", "temf", "temf_sfclay", "idealscmsfclay",
    'pxsfclay', 'ssib', 'trajectory', 'tslist', 'wrf_stock_aer3',
    # New port quotients retain IEEE division through direct NVRTC.
    'afwa_diagnostics', 'dfi_reset', 'global_advection', 'global_geometry', 'global_hydrostatic', 'global_polar', 'level_diagnostics', 'nwp_windows', 'nssl2_shape_gs',
})



def module_options(name: str) -> tuple[str, ...]:
    """Compile options shared by runtime, oracle and division census."""
    if name == "deng":
        from woof.numerics import is_strict
        return (("-std=c++17", "--fmad=false", "--ftz=false") if is_strict()
                else ("-std=c++17", "--fmad=true", "--ftz=false"))
    if name in ("noah_ua", "noah_mosaic_ua", "noah_ua_sfcdiags", "noah_seaice_options", "noah_fractional", "sfclay_dx2d"):
        return ("-std=c++17", "--fmad=false", "--ftz=false")
    if name == "eeps":
        from woof.wrf_exact import ENABLED
        if not ENABLED:
            # Use float32 fast division in runtime, as the other PBLs do.
            # Native word replay keeps the precise division contract.
            return ("-std=c++17", "--ftz=false", "--prec-div=false")
    if name == "keps":
        from woof.wrf_exact import ENABLED
        if not ENABLED:
            # Limit live register ranges without adding a local frame.
            return ("-std=c++17", "--maxrregcount=112")

    if name in {"temf", "temf_sfclay"}:
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--ftz=false", "--fmad=false") if ENABLED
                else ("-std=c++17", "--ftz=false", "--fmad=true"))

    if name in ("flg", "flg_gases", "flg_streams", "flg_column", "flg_driver"):
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--fmad=true", "--ftz=false")
                if name in ("flg_driver", "flg_column") and not ENABLED else UNFLUSHED_OPTIONS)

    if name in {"wdm5", "wdm7", "wdm7_refl"}:
        from woof.wrf_exact import ENABLED
        from woof.numerics import is_strict
        return (("-std=c++17", "--fmad=false", "--ftz=false") if ENABLED or is_strict()
                else ("-std=c++17", "--ftz=false"))
    if name == "sas":
        return (("-std=c++17", "--ftz=false", "--fmad=false")
                if _sas_strict_numerics() else ("-std=c++17", "--ftz=false"))
    if name in _GFDLETA_MODULES:
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--ftz=false", "--fmad=false", "-DGPUWM_GFDLETA_STRICT=1")
                if ENABLED else ("-std=c++17", "--ftz=false"))
    if name == "ishmael":
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--ftz=false", "--fmad=false") if ENABLED
                else ("-std=c++17", "--ftz=false"))
    if name in ("gfs_pbl", "sf_gfs", "rcon", "udm"):
        from woof.numerics import mode
        options = ("-std=c++17", "--ftz=false")
        # These columns switch their scalar helpers as well as contraction.
        # This define is local to the unit, not whole-model WRF replay.
        if mode() == "strict":
            options += ("--fmad=false", "-DGPUWM_WRF_EXACT=1")
        return options
    if name in {"qnsepbl", "qnsesfc"}:
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--fmad=false", "--ftz=false",
                 "--prec-div=true", "--prec-sqrt=true", "-DGPUWM_QNSE_STRICT=1") if ENABLED
                else ("-std=c++17", "--ftz=false"))

    if name == "pxsfclay":
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--fmad=false", "--ftz=false")
                if ENABLED else ("-std=c++17", "--ftz=false"))

    if name == "boulac":
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--fmad=false", "--ftz=false")
                if ENABLED else ("-std=c++17",))

    if name in ("camzm", "cammgmp", "uwshcu"):
        from woof.numerics import is_strict
        return (("-std=c++17", "--fmad=false", "--ftz=false") if is_strict()
                else ("-std=c++17", "--ftz=false", "-DGPUWM_CAM_FAST=1"))

    if name in {"mskf", "mskf_aerosol_replay"}:
        from woof.wrf_exact import ENABLED
        # Native operation order protects threshold-sensitive aerosol numbers.
        # Default CUDA libm remains distinct from strict software libm.
        return ("-std=c++17", "--fmad=false", "--ftz=false", "-DGPUWM_MSKF_OPERATION_ORDER=1")

    if name == "kf_old":
        from woof.numerics import is_strict
        options = ("-std=c++17", "--fmad=false", "--ftz=false")
        return options + (("-DGPUWM_KF_OLD_STRICT=1",) if is_strict() else ())
    from woof.wrf_exact import ENABLED
    if name == "trajectory" and ENABLED:
        return ("-std=c++17", "--ftz=false", "--fmad=false")
    if name == "tslist":
        from woof.wrf_exact import ENABLED
        if ENABLED:
            return ("-std=c++17", "--ftz=false", "--fmad=false")
    if name in ("wind_mav", "wind_legacy"):
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
                if ENABLED else ("-std=c++17", "--ftz=false"))
    if name == 'nsas':
        return ('-std=c++17', '--ftz=false', '--fmad=false')
    if name == "urban_ucm480":
        return ("-std=c++17", "--ftz=false", "--fmad=false")
    if name in _URBAN_NO_FTZ_MODULES:
        return ("-std=c++17", "--ftz=false")
    if name in _SOLAR_NO_FTZ_MODULES:
        return ("-std=c++17", "--ftz=false")
    if name in ("lightning", "native_surface_diagnostics"):
        from woof.wrf_exact import ENABLED
        return (("-std=c++17", "--ftz=false", "--fmad=false") if ENABLED
                else ("-std=c++17", "--ftz=false"))
    if name == "kfcup":
        return (("-std=c++17", "--fmad=false", "--ftz=false") if ENABLED
                else ("-std=c++17", "--fmad=true", "--ftz=false"))
    if name in _NO_FTZ_MODULES:
        return ("-std=c++17", "--fmad=false", "--ftz=false")
    if name in _DIFFUSION_MODULES:
        return DIFFUSION_OPTIONS
    if name in _PORT_IEEE_DIVISION_MODULES:
        return (("-std=c++17", "--fmad=false", "--ftz=false")
                if _needs_no_fmad(name) else ("-std=c++17", "--ftz=false"))
    return (("-std=c++17", "--fmad=false") if _needs_no_fmad(name)
            else ("-std=c++17",))


_DIFFUSION_MODULES = frozenset(("smag2d", "sfs_output", "km5", "km5_geometry", "diffusion", "diff_opt1", "diff6", "diff6_seam"))

DIFFUSION_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false",
                     "--prec-div=true", "--prec-sqrt=true",
                     "-DGPUWM_WRF_EXACT_C_DIFFUSION=1")
# Only the coefficient and w solve use this route in the acoustic unit.
# Other acoustic symbols keep their existing compiled module.
_DIFFUSION_FUNCTIONS = {
    "acoustic": frozenset(("calc_coefs", "advance_w_phi", "advance_w_phi_msf")),
    "dycore": frozenset(("w_damp", "w_cfl_stat")),
}


def diffusion_kernel(name: str, function: str) -> bool:
    # The default acoustic arm failed the real-column exactness gate.
    # Retain its existing strict opt-in compiler and the qualified diffusion
    # modules; an ordinary acoustic launch keeps its published arithmetic.
    if name == "acoustic":
        from woof.wrf_exact import ENABLED
        if not ENABLED:
            return False
    return name in _DIFFUSION_MODULES or function in _DIFFUSION_FUNCTIONS.get(name, ())


def function_options(name: str, function: str, options) -> tuple[str, ...]:
    """Keep scalar and batch diffusion on the same arithmetic route."""
    if not diffusion_kernel(name, function):
        return tuple(options)
    overridden = {"std", "fmad", "ftz", "prec-div", "prec-sqrt", "use_fast_math",
                  "DGPUWM_WRF_EXACT_C_DIFFUSION"}
    return tuple(o for o in options if o.lstrip("-").split("=", 1)[0] not in overridden) + DIFFUSION_OPTIONS


@cuda_cache(maxsize=None)
def compile_diffusion_source(source: str, key: str, options):
    """Load and record an IEEE diffusion image, including batch witnesses."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.kernel_compile_notice import observe_module_compile
    from woof.certify.kernel_manifest import record_module
    from types import SimpleNamespace
    from woof.wrf_exact import ENABLED, effective_options
    if ENABLED:
        options = effective_options(options)
    with observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(source, options=options)
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    image = binary.encode() if isinstance(binary, str) else binary
    kind = "cubin" if image.startswith(b"\x7fELF") else "ptx"
    record_module(key, source=source, options=options,
                  module=SimpleNamespace(**{kind: image}))
    return _DriverModule(module)


@cuda_cache(maxsize=None)
def _load_diffusion_module(name: str, defines=()):
    """Compile with IEEE arithmetic at NVRTC's final boundary.

    RawModule appends FTZ after caller options. Direct NVRTC keeps the
    requested --ftz=false and records the actual image and option tuple.
    The ordinary and integer-tier loaders share this one compile site.
    """
    from woof.wrf_exact import ENABLED, DIFFUSION_ENABLED, effective_options
    from woof.physics_params import note_compiled
    source = module_source_int_defines(name, defines) if defines else module_source(name)
    options = DIFFUSION_OPTIONS
    if not DIFFUSION_ENABLED:
        options = tuple(o for o in options if "DGPUWM_WRF_EXACT_C_DIFFUSION" not in o)
    if ENABLED:
        options = effective_options(options)
    tier = ",".join(f"{key}={value}" for key, value in defines)
    key = f"{MODULE_KEY_ROOT}:{name}" + (f"[{tier}]" if tier else "")
    module = compile_diffusion_source(source, key, options)
    note_compiled(name, module)
    return module


class _DriverKernel:
    """A direct CUDA function with the loader's common kernel interface."""

    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._function, name)

    @property
    def attributes(self):
        from cupy.cuda import driver
        names = ("max_threads_per_block", "shared_size_bytes", "const_size_bytes",
                 "local_size_bytes", "num_regs", "ptx_version", "binary_version",
                 "cache_mode_ca", "max_dynamic_shared_size_bytes",
                 "preferred_shared_memory_carveout")
        return {name: driver.funcGetAttribute(
            getattr(driver, "CU_FUNC_ATTRIBUTE_" + name.upper()), self._function.ptr)
            for name in names}


class _DriverModule:
    def __init__(self, module):
        self._module = module

    def get_function(self, name):
        return _DriverKernel(self._module.get_function(name))

    def __getattr__(self, name):
        return getattr(self._module, name)


def compile_noftz_module(name: str, src: str, key: str):
    """Compile the requested image without CuPy's appended FTZ option."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.kernel_compile_notice import observe_module_compile
    from woof.certify.kernel_manifest import record_module
    from woof.physics_params import note_compiled
    options = module_options(name)
    # Bind strict macros into the direct cache key as well as the compiler
    # request. Otherwise a default cached image could enter a strict run.
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    with observe_module_compile(key):
        image, _ = compiler.compile_using_nvrtc(src, options, None, name + ".cu")
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    note_compiled(name, module)
    record_module(key, source=src, options=options, module=None)
    return _DriverModule(module)


def _load_urban_module_without_ftz(name: str, src: str):
    """Urban rain clamps and small momentum terms require IEEE subnormals."""
    import cupy as cp
    from woof.cuda_subnormals import install
    install()
    options = module_options(name)
    mod = cp.RawModule(code=src, options=options,
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                  options=options, module=mod)
    return mod


def _load_keps_register_limited(name: str, src: str):
    """Keep the BL17 speed lane's ordinary load and integer tiers alike."""
    import cupy as cp
    mod = cp.RawModule(code=src, options=("-std=c++17", "--maxrregcount=112"),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                  options=("-std=c++17", "--maxrregcount=112"), module=mod)
    return mod


def _load_module_without_fmad(name: str, src: str, options: tuple[str, ...]):
    """The shared compile site for operation-order-pinned units.

    Its WRF column oracle is graded word for word, which needs NVRTC not to
    contract multiply-adds.  A separate site keeps load_module's call a
    literal tuple, so the FTZ route inventory still reads route R1's options.

    The site records what it compiled here, beside the compile, with the
    same literal tuple.  The breakage this prevents: the kernel manifest's
    audit pairs each compile with a record in the same function and
    compares their arguments as written; a record left in load_module named
    a variable, so the audit could no longer tell that the lake's manifest
    row states the options NVRTC was given.
    """
    import cupy as cp
    if options != ("-std=c++17", "--fmad=false"):
        raise ValueError(f"no kernel compile site takes options {options!r}")
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    # The physics-parameter guard must see these units compile too:
    # mynn_pbl and mynn_dmp_sibling carry registered literals
    # (woof/physics_params_registry_v1.json), and a set declared after they
    # compiled would otherwise run under the set's name with the default
    # constants, the breakage physics_params.declare refuses.
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                  options=("-std=c++17", "--fmad=false"), module=mod)
    return mod


def compile_ishmael_no_ftz(name: str, src: str, options: tuple[str, ...]):
    """Use the shared direct-NVRTC cache for kernels that retain subnormals.

    RawModule appends FTZ and changes closure decisions in G3. The existing
    exact-image cache preserves the explicitly requested arithmetic options.
    This entry also serves the oracle's planted-source control.
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache
    from woof.kernel_compile_notice import observe_module_compile
    if "--ftz=false" not in options:
        raise ValueError("the no-FTZ route requires --ftz=false to retain subnormals")
    with observe_module_compile(f"{MODULE_KEY_ROOT}:{name}"):
        image, _mapping = nvrtc_ptx_cache.compile_using_nvrtc(
            src, options, None, f"{name}.cu")
        mod = cp.cuda.function.Module()
        mod.load(image.encode() if isinstance(image, str) else image)
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                  options=options, module=mod)
    return mod



def compile_cam_module_no_ftz(name: str, src: str, options: tuple[str, ...] | None = None,
                          *, suffix: str = ""):
    """Compile a CAM column or instrument without changing other loaders."""
    import cupy as cp
    from woof import nvrtc_ptx_cache
    from woof.kernel_compile_notice import observe_module_compile
    options = module_options(name) if options is None else options
    key = f"{MODULE_KEY_ROOT}:{name}{suffix}"
    if "--ftz=false" not in options:
        raise ValueError("the no-FTZ route requires --ftz=false to retain subnormals")
    with observe_module_compile(key):
        image, _mapping = nvrtc_ptx_cache.compile_using_nvrtc(
            src, options, None, f"{name}.cu")
        mod = cp.cuda.function.Module()
        mod.load(image.encode() if isinstance(image, str) else image)
    from woof.physics_params import note_compiled
    note_compiled(name)
    from woof.certify.kernel_manifest import record_module
    record_module(key, source=src,
                  options=options, module=mod)
    return mod



def _load_module_preserving_subnormals(name: str, src: str, options):
    """Use the shared direct NVRTC cache so CuPy cannot append FTZ=true."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    from woof.physics_params import note_compiled

    key = f"{MODULE_KEY_ROOT}:{name}"
    with observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(src, options, None, f"{name}.cu")
        mod = cp.cuda.function.Module()
        mod.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(key, source=src, options=options, module=None)
    note_compiled(name, mod)
    return mod
class _DirectNvrtcModule:
    """Shared-loader facade for the direct no-FTZ route and image receipt."""
    def __init__(self, native, image):
        self.module = native
        payload = image.encode() if isinstance(image, str) else image
        if payload.startswith(b"\x7fELF"):
            self.cubin = payload
        else:
            self.ptx = payload

    def get_function(self, name):
        return self.module.get_function(name)


def compile_no_ftz(source: str, *, module_key: str, filename: str,
                   device: int):
    """Load a column unit without FTZ, honoring the selected numerics mode.

    Direct NVRTC compilation keeps these options last. RawModule adds
    FTZ after its caller's options, which loses subnormal tendencies.
    The manifest records both the exact source and the compiled image.
    """
    import cupy as cp
    from types import SimpleNamespace
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    from woof.numerics import effective_options
    options = effective_options(("-std=c++17", "--fmad=false", "--ftz=false"))
    with cp.cuda.Device(device), observe_module_compile(module_key):
        image, _ = compiler.compile_using_nvrtc(
            source, options, None, filename)
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    record_module(module_key, source=source, options=options,
                  module=SimpleNamespace(ptx=image))
    return module


@cuda_cache(maxsize=None)
def load_module(name: str):
    import cupy as cp
    if name in _DIFFUSION_MODULES:
        return _load_diffusion_module(name)
    from woof.wrf_exact import ENABLED
    if name in _UNFLUSHED_MODULES:
        return load_module_unflushed(name)
    if name.startswith("noahmp_"):
        from woof.core.noahmp_kernel_sources import (
            NOAHMP_TRANSLATION_UNITS, compile_runtime_unit)
        # A standalone Noah-MP unit compiles through the one Noah-MP
        # RawModule site, so the source string a forecast hands NVRTC is
        # the one its frame recording was read from.  Fragments (which
        # fail alone, and must keep failing alone) and the generic C++17
        # VEGE_FLUX census stay on the plain route below: the runtime
        # VEGE_FLUX unit is C++14 without the preamble and has its own
        # factory.  tests/test_kernel_loader_inert.py asserts the two
        # routes assemble byte-identical source for every unit this
        # branch takes.
        if (name in NOAHMP_TRANSLATION_UNITS
                and len(NOAHMP_TRANSLATION_UNITS[name]) == 1
                and name != "noahmp_vegeflux"):
            return compile_runtime_unit(name, module_key=f"{MODULE_KEY_ROOT}:{name}")
    if name == "ishmael":
        return compile_ishmael_no_ftz(name, module_source(name), module_options(name))
    if name == "fast_sbm":
        # FAST_SBM compiles through its own direct NVRTC loader (its own
        # options and include path), the one the forecast launches and
        # the G2 frame recording was read from; the plain route below
        # cannot build it, so a census through this function measured
        # nothing for the module the preflight table prices.
        from woof.core.fast_sbm_cuda import module as _fast_sbm_module
        return _fast_sbm_module(cp.cuda.Device().id)
    src = module_source(name)
    if name in _PORT_IEEE_DIVISION_MODULES:
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")

    if name == "boulac":
        from woof.wrf_exact import ENABLED
        if ENABLED:
            return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")

    if name == "fogdes":
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")
    if name == "pxsfclay":
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")
    if name == "tslist" and "--ftz=false" in module_options(name):
        return _load_module_preserving_subnormals(name, src, module_options(name))
    if name in _URBAN_NO_FTZ_MODULES:
        return _load_urban_module_without_ftz(name, src)
    if name in _NO_FTZ_MODULES or name in {"mskf", "mskf_aerosol_replay"}:
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")
    options = module_options(name)
    if name in ("noah_ua", "noah_mosaic_ua", "noah_ua_sfcdiags", "noah_seaice_options", "noah_fractional", "sfclay_dx2d"):
        return _load_module_preserving_subnormals(name, src, options)

    if name in ("camzm", "cammgmp", "uwshcu"):
        return compile_cam_module_no_ftz(name, src, options)
    if name in _NO_FTZ_MODULES:
        return _load_module_preserving_subnormals(name, src, options)
    if name in ("lightning", "native_surface_diagnostics") or (name == "trajectory" and ENABLED):
        from woof import nvrtc_ptx_cache as compiler
        from woof.kernel_compile_notice import observe_module_compile
        from woof.certify.kernel_manifest import record_module
        with observe_module_compile(f"{MODULE_KEY_ROOT}:{name}"):
            image, _ = compiler.compile_using_nvrtc(
                src, options,
                None, f"{name}.cu")
            native = cp.cuda.function.Module()
            native.load(image.encode() if isinstance(image, str) else image)
        mod = _DirectNvrtcModule(native, image)
        record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                      options=options,
                      module=mod)
        return mod
    if name == "keps" and options == ("-std=c++17", "--maxrregcount=112"):
        return _load_keps_register_limited(name, src)
    if options != ("-std=c++17",):
        return _load_module_without_fmad(name, src, options)
    # The loader's literal tuple: tools/ftz_receipt reads it from this
    # call (route R1) and probes NVRTC's float behaviour under it.
    mod = cp.RawModule(code=src, options=("-std=c++17",), name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}",
                  source=src, options=("-std=c++17",), module=mod)
    return mod


@cuda_cache(maxsize=None)
def load_module_int_defines(
        name: str, defines: tuple[tuple[str, int], ...]):
    """Compile one kernel source with a small, identity-bound integer tier.

    This keeps compile-time local-array bounds specialized without enlarging
    the common kernel.  Only uppercase C-preprocessor identifiers and positive
    integer values are accepted; callers cannot inject arbitrary source text.
    """
    import re
    import cupy as cp

    if name in _UNFLUSHED_MODULES:
        return load_module_unflushed(name, defines)

    normalized = tuple((str(key), int(value)) for key, value in defines)
    if normalized != defines:
        raise TypeError("kernel integer defines must be canonical (str, int) pairs")
    for key, value in normalized:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            raise ValueError(f"invalid CUDA preprocessor identifier {key!r}")
        if isinstance(value, bool) or value < 1:
            raise ValueError(
                f"CUDA integer define {key} must be a positive integer")
    prefix = "\n".join(f"#define {key} {value}" for key, value in normalized)
    src = module_source_int_defines(name, normalized, prefix=prefix)
    if name in _NO_FTZ_MODULES | _PORT_IEEE_DIVISION_MODULES:
        tier = ",".join(f"{key}={value}" for key, value in normalized)
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}[{tier}]")
    if name in _DIFFUSION_MODULES:
        return _load_diffusion_module(name, normalized)
    if name in ("camzm", "cammgmp", "uwshcu"):
        tier = ",".join(f"{key}={value}" for key, value in normalized)
        return compile_cam_module_no_ftz(name, src, suffix=f"[{tier}]")
    if _needs_no_fmad(name):
        return _load_module_int_defines_without_fmad(name, src, normalized)
    options = module_options(name) if name == "keps" else ("-std=c++17",)
    mod = cp.RawModule(code=src, options=options, name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    record_module(f"{MODULE_KEY_ROOT}:{name}[{tier}]",
                  source=src, options=options, module=mod)
    return mod


def _load_module_int_defines_without_fmad(name: str, src: str, normalized):
    """An integer-define tier of a no-FMA unit: the same literal tuple as
    :func:`_load_module_without_fmad`.

    The breakage this prevents: the gsd_41 MYNN generation compiles
    mynn_pbl through this loader; with the plain tuple NVRTC would contract
    its multiply-adds while the default build (load_module) does not, and
    the two builds of one source would round differently.  The record sits
    beside the compile with the same literal tuple, as the lake's does.
    """
    import cupy as cp
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from woof.physics_params import note_compiled
    note_compiled(name, mod)
    from woof.certify.kernel_manifest import record_module
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    record_module(f"{MODULE_KEY_ROOT}:{name}[{tier}]",
                  source=src, options=("-std=c++17", "--fmad=false"),
                  module=mod)
    return mod
#: The one option tuple of :func:`load_module_unflushed`: no multiply-add
#: contraction and no flush to zero.
UNFLUSHED_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")

#: Units compiled by :func:`load_module_unflushed`, closed and literal.
#: wsm5: the WRF v4.6.1 WSM5 column scheme, graded word for word against
#: WRF's own build (tests/test_wsm5.py), whose trajectories carry subnormal
#: hydrometeor words in the default arithmetic users run, not only under
#: GPUWM_WRF_EXACT=1.
#: gsd_initial_reflectivity: the pinned HRRR helper shares the fork's
#: radar transcription; retain IEEE subnormals in its PSD powers. Strict
#: tests disable contraction; the default numerics policy still permits FMA.
_UNFLUSHED_MODULES = frozenset({"wsm5", "gsd_initial_reflectivity", "flg", "flg_gases", "flg_streams", "flg_column", "flg_driver"})


@cuda_cache(maxsize=None)
def load_module_unflushed(name: str, defines: tuple[tuple[str, int], ...] = ()):
    """Compile a forecast unit unflushed and uncontracted, default-on.

    The breakage this prevents: ``cp.RawModule`` appends ``-ftz=true`` after
    any caller option (and NVRTC 13 refuses the duplicate an explicit
    ``--ftz=false`` would make), so a unit loaded by :func:`load_module` or
    :func:`load_module_int_defines` flushes every subnormal it computes --
    hydrometeor mixing ratios of 1e-40 and the surface increments they
    sediment -- in the default build, and only GPUWM_WRF_EXACT=1 would keep
    WRF's words.  This route hands NVRTC :data:`UNFLUSHED_OPTIONS` directly
    (CuPy injects no ``-ftz`` on it), so the default build and the strict
    build compile the same arithmetic.  ``defines`` is the same validated
    integer tier :func:`load_module_int_defines` takes; the module key
    carries it, and the record sits beside the compile with the same
    expressions.
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache
    from woof.kernel_compile_notice import observe_module_compile

    if name not in _UNFLUSHED_MODULES:
        raise ValueError(
            f"{name!r} is not an unflushed unit; load_module_unflushed "
            f"compiles {sorted(_UNFLUSHED_MODULES)} only")
    normalized = _validated_int_defines(defines)
    source = (module_source_int_defines(name, normalized) if normalized
              else module_source(name))
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    key = f"{MODULE_KEY_ROOT}:{name}" + (f"[{tier}]" if tier else "")
    options = module_options(name) if name.startswith("flg") else UNFLUSHED_OPTIONS
    with observe_module_compile(key):
        image, _mapping = nvrtc_ptx_cache.compile_using_nvrtc(
            source, options, None, f"{name}.cu")
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    from woof.physics_params import note_compiled
    note_compiled(name, module)
    from woof.certify.kernel_manifest import record_module
    # Record the exact image returned by NVRTC and loaded above.
    from types import SimpleNamespace
    blob = image.encode() if isinstance(image, str) else image
    identity = (SimpleNamespace(cubin=blob) if blob.startswith(b"\x7fELF")
                else SimpleNamespace(ptx=blob))
    record_module(key, source=source, options=options, module=identity)
    return module


@cuda_cache(maxsize=None)
def get_kernel_unflushed(name: str, func: str,
                         defines: tuple[tuple[str, int], ...] = ()):
    """One stable function handle per unflushed-unit symbol and tier."""
    return load_module_unflushed(name, defines).get_function(func)


def _validated_int_defines(defines) -> tuple[tuple[str, int], ...]:
    """The integer-tier validation of :func:`load_module_int_defines`."""
    import re
    normalized = tuple((str(key), int(value)) for key, value in defines)
    if normalized != tuple(defines):
        raise TypeError("kernel integer defines must be canonical (str, int) pairs")
    for key, value in normalized:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            raise ValueError(f"invalid CUDA preprocessor identifier {key!r}")
        if isinstance(value, bool) or value < 1:
            raise ValueError(
                f"CUDA integer define {key} must be a positive integer")
    return normalized


#: Manifest namespace for the translation units the loaders above compile.
#: Declared below them because the FTZ receipt pins their RawModule lines.
MODULE_KEY_ROOT = "woof.core.kernels"


# Headers that consume the loader integer tier.
_HEADERS_NEED_INT_DEFINES = frozenset({'goddard', 'cu_osas', 'km5_geometry', 'flg_column', 'cammgmp'})

def module_source_int_defines(
        name: str, defines: tuple[tuple[str, int], ...],
        *, prefix: str | None = None, kernel_dir: Path = _KDIR) -> str:
    """The exact source :func:`load_module_int_defines` hands to nvrtc.

    The allow-listed header, when present, goes immediately after the
    preamble, exactly as in :func:`module_source`; for every module absent
    from ``_EXTRA_HEADERS`` the inserted text is empty and the string is
    byte-identical to the pre-hook assembly.
    """
    if prefix is None:
        prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
    if name == "cam3_emissivity" and {'CAM_LONGWAVE_MODE','CAM_SHORTWAVE_MODE'} <= dict(defines).keys():
        from woof.numerics import is_strict
        if not is_strict():
            from woof.core.cam3_fast import fast_source
            # Fast CAM calls CUDA float math. The unused strict binary64
            # header also contains a host-only include, which makes the
            # direct NVRTC cache bypass every lookup.
            return (_preamble(kernel_dir) + prefix + "\n"
                    + fast_source(_unit_text(name, kernel_dir)))
    if name in ("thompson_column", "rcon", "udm"):
        # This tier sizes the generated header's arrays as well as the unit.
        return (_preamble(kernel_dir) + prefix + "\n" + _extra_header_text(name,kernel_dir)
                + _unit_text(name,kernel_dir))
    if name in _HEADERS_NEED_INT_DEFINES:
        return (_preamble(kernel_dir) + prefix + "\n"
                + _extra_header_text(name, kernel_dir) + _unit_text(name, kernel_dir))
    source = (_preamble(kernel_dir) + _extra_header_text(name, kernel_dir)
              + prefix + "\n" + _unit_text(name, kernel_dir))
    from woof.core.kernel_shape import specialize_source
    return specialize_source(name, source, tuple(defines))


@cuda_cache(maxsize=None)
def get_kernel(name: str, func: str, shape: tuple[int, int, int] = ()):
    """Return one stable function per symbol and optional (nz, ny, nx)."""
    if shape:
        from woof.core.kernel_shape import shape_defines
        defines = shape_defines(name, func, shape)
        if defines:
            return get_kernel_int_defines(name, func, defines)
    if diffusion_kernel(name, func) and func in _DIFFUSION_FUNCTIONS.get(name, ()):
        return _load_diffusion_module(name).get_function(func)
    return load_module(name).get_function(func)


@cuda_cache(maxsize=None)
def get_kernel_int_defines(
        name: str, func: str, defines: tuple[tuple[str, int], ...],
        shape: tuple[int, int, int] = ()):
    """Return a cached kernel compiled with validated integer definitions."""
    if shape:
        from woof.core.kernel_shape import shape_defines
        defines = defines + shape_defines(name, func, shape)
    if diffusion_kernel(name, func) and func in _DIFFUSION_FUNCTIONS.get(name, ()):
        # Validate through the same public loader before selecting a tier.
        # The tier loader's validation remains the sole definition contract.
        if defines != tuple((str(key), int(value)) for key, value in defines):
            raise TypeError("kernel integer defines must be canonical (str, int) pairs")
        import re
        for key, value in defines:
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None or isinstance(value, bool) or value < 1:
                raise ValueError(f"invalid CUDA integer define {key}={value!r}")
        return _load_diffusion_module(name, defines).get_function(func)
    return load_module_int_defines(name, defines).get_function(func)


def _compile_observed(module, module_key: str) -> None:
    """Compile ``module``, telling a watching run when it really compiled.

    :func:`woof.kernel_compile_notice.observe_module_compile` costs nothing
    when no run is watching; when one is, a module that wrote to the kernel
    cache (a compile, not a cache load) is published as progress.
    """
    from woof.kernel_compile_notice import observe_module_compile
    with observe_module_compile(module_key):
        module.compile()


def compile_module_no_ftz(name: str, src: str, options=None):
    """Column probes share E's direct cache and compiler receipt site."""
    if options is not None and set(options) != set(module_options("nsas")):
        raise ValueError("NSAS probes require the scheme compiler options to preserve subnormals and operation order")
    return compile_noftz_module(
        "nsas", src, f"{MODULE_KEY_ROOT}:{name}")
def _load_module_no_ftz(name: str, src: str):
    """Strict KSAS oracle probes use the same direct compiler as the scheme."""
    return compile_noftz_module("sas", src, f"{MODULE_KEY_ROOT}:{name}")


def kernel_local_size_bytes(function) -> int:
    """Read the actual driver frame from RawKernel or direct-NVRTC Function."""
    attributes = getattr(function, "attributes", None)
    if attributes is not None:
        return int(attributes["local_size_bytes"])
    from cupy.cuda import driver
    return int(driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function.ptr))


def compile_wdm5_no_ftz(source: str, module_key: str):
    """WDM5 column mutation control through the shared direct compiler.

    Named apart from :func:`compile_no_ftz`: sharing that name, this later
    definition replaced ACM2's keyword loader and every ACM2 compile raised
    TypeError ('filename').
    """
    return compile_noftz_module("wdm5", source, module_key)
@cuda_cache(maxsize=None)
def load_module_no_ftz(name: str, *, no_fmad: bool = False):
    """Shared direct NVRTC route retaining subnormal operands and results."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    from woof.wrf_exact import ENABLED, effective_options
    source = module_source(name)
    options = ("-std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if no_fmad:
        options += ("--fmad=false",)
    if ENABLED:
        options = effective_options(options)
    with observe_module_compile(f"{MODULE_KEY_ROOT}:{name}"):
        binary, _ = compiler.compile_using_nvrtc(source, options, None, name + ".cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=source,
                  options=options, module=None)
    return module


# DFI shares the same direct NVRTC cache and subnormal-preserving image.
compile_module_preserving_subnormals = compile_ishmael_no_ftz
