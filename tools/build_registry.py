"""Rewrite physics_registry_v2.json from the verified knob tables.

Importing this module is side-effect free: nothing is read and nothing is
written until :func:`main` runs.  It used to write the registry at module
scope, so any ``import tools.build_registry`` -- a test collecting the tools
tree, a REPL, an editor's autocomplete -- silently rewrote a tracked file.

``tests/test_build_registry.py`` is the enforcement this file lacked: it runs
the builder into a temporary path and requires the bytes to equal the tracked
registry.  Without that gate the builder and the file it generates drift, and
they did: replaying the pre-fix builder reverted every citation commit 611396b
had just corrected.

Usage
-----
    python tools/build_registry.py [--out <path>]
"""
from __future__ import annotations

import argparse
import ast
import copy
from concurrent.futures import ProcessPoolExecutor
import os
import json
import pathlib
import re
import sys

# Repo-relative: this script lives in tools/, so the model root is its parent.
# The knob survey it consumes is committed beside it rather than read from a
# session-temp workflow journal, which would not have survived the session.
MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

JOURNAL = MODEL / "tools" / "data" / "knob_survey_lanes.jsonl"
REGISTRY_PATH = MODEL / "woof" / "physics_registry_v2.json"

from woof.config import (  # noqa: E402
    GF_CLOSURE_MEMBERS,
    MOSAIC_URBAN_CANOPY_DEFAULT,
    MOSAIC_URBAN_CANOPY_RULES,
    URBAN_WATER_BALANCE_DEFAULT,
    URBAN_WATER_BALANCE_GENERATIONS,
    MP28_AEROSOL_SOURCES as MP28_AEROSOL_SOURCES,
)
from woof.physics_registry import (  # noqa: E402
    TEMPLATE_ID_ALIASES, canonical_json, component_override_declaration,
)
from woof import physics_compat  # noqa: E402

#: The value ``mp28_aerosol_source`` takes when nobody chooses, which is
#: the one value a non-mp=28 option may carry: the enumeration hands every
#: plan the registry's own default, and refusing that would refuse every
#: plan that never mentioned the key.
MP28_AEROSOL_SOURCE_DEFAULT_VALUE = "auto"
from woof.wrf461_compatibility import (  # noqa: E402
    CUMULUS_OPTIONS,
    LAND_SURFACE_OPTIONS,
    MATRIX_CELL_COUNT,
    MP_OPTIONS,
    PBL_OPTIONS,
    PBL_SURFACE_LAYER_AUTHORITY,
    RADIATION_OPTIONS,
    SURFACE_LAYER_OPTIONS,
    WRF_COMMIT,
    WRF_VERSION,
    compatibility_cell,
    iter_compatibility_matrix,
)
# The prose-stripping rule belongs to the gate, so it is imported rather than
# reimplemented.  A second copy is how the citations drifted: this builder
# stripped only ``#`` comments while tools/check_parameter_claims.py stripped
# docstrings too, so the builder kept proposing citations the gate rejected.
# Imported INSIDE _source() below rather than here: tools.check_parameter_claims
# reaches woof.config and, through it, every consumer module whose
# import-time agreement check compares against the registry on disk -- the
# registry this builder is about to replace.  main() sets the rebuild flag
# before any of those imports happen; a module-scope import here would run
# them first and refuse the rebuild that fixes a stale registry.

# ---------------------------------------------------------------- implemented
# type / enum / minimum / default taken from woof's own accepted sets
# (woof/config.py validate_run_config), not from WRF's wider sets.
IMPLEMENTED: dict[str, dict] = {
    # The Noah-MP admitted-value table includes these paper-rewrite keys.
    # Register their actual defaults too: an absent parameter otherwise
    # resolves to None and incorrectly refuses even opt_alb=2 plans.
    "snicar_snowshape_opt": {
        "type": "integer", "enum": [1, 2, 3, 4], "default": 3,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 4, "opt_alb": 3},
        "consuming_read": "woof/core/snow_papers.py",
        "description": "Paper snow optics grain habit; written from papers, not bitwise to WRF."},
    "snicar_snowbc_intmix": {
        "type": "boolean", "default": True,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 4, "opt_alb": 3},
        "consuming_read": "woof/core/snow_papers.py",
        "description": "Paper snow optics internal black-carbon mixing; written from papers, not bitwise to WRF."},
    "snicar_snowdust_intmix": {
        "type": "boolean", "default": False,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 4, "opt_alb": 3},
        "consuming_read": "woof/core/snow_papers.py",
        "description": "Paper snow optics internal dust mixing; written from papers, not bitwise to WRF."},
    "use_theta_m": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "consuming_read": "woof/core/state.py",
        "description": "Thermal prognostic: dry potential temperature (0) or moist potential temperature (1); history T stays dry and THM carries moist theta."},
    "do_avgflx_em": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "consuming_read": "woof/core/dycore.py",
        "description": "WRF history-interval means of mass-coupled wind fluxes."},
    "do_avgflx_cugd": {
        "type": "integer", "enum": [0], "default": 0,
        "consuming_read": "woof/config.py",
        "description": "Convective flux averaging is unavailable: WRF's Grell-Devenyi cu_physics=93 flux producer is missing."},
    "co2tf": {"type": "integer", "enum": [0, 1], "default": 1,
              "component_id": "radiation",
              "warnings": ["Read only when either radiation selector is 99. CONRAD (0) requires a transmission table matched to the vertical grid; CO2O3 (1) builds it on the GPU."]},
    'ntu_ccnty': {"type": "integer", "enum": [1, 2, 3, 4], "default": 2},
    "windturbines_spec": {"type":"string", "default":"none", "per_domain":False,
        "component_id":"pbl", "consuming_read":"woof/namelist_import.py",
        "description":"WRF V3.5 analytic turbine drag: none, ideal or a nine-field windspec file."},
    **{key: {"type":"integer" if type(value) is int else "number", "default":value,
             "per_domain":False, "component_id":"pbl", "consuming_read":"woof/core/wind_legacy.py"}
       for key,value in dict(td_turbgridid=-1,td_hubheight=100.,td_diameter=60.,td_stdthrcoef=.158,
           td_cutinspeed=4.,td_cutoutspeed=27.,td_power=2.,td_turbpercell=1.,td_ewfx=0,
           td_ewfy=0,td_pwfx=1,td_pwfy=1).items()},
    "windfarm_opt": {"type":"integer", "enum":[0,1,2], "default":0,
        "per_domain":True,"component_id":"pbl","consuming_read":"woof/namelist_import.py",
        "description":"Native WRF wind farm: off, Fitch or MAV. MAV is a named opt-in: measured forecast hook cost 2.044114x Fitch on the retained eight-turbine case; added native wake quadratures are profiled. See docs/wind-farms.md."},
    "windfarm_induction": {"type":"integer", "enum":[0,1], "default":0,
        "per_domain":False,"component_id":"pbl","consuming_read":"woof/core/wind_farm.py",
        "description":"WRF Fitch two-pass axial induction correction; explicit opt-in, measured 1.28x Fitch cost; zero is the default."},
    "windfarm_wake_model": {"type":"integer", "enum":list(range(1,8)), "default":2,
        "per_domain":True,"component_id":"pbl","consuming_read":"woof/core/wind_mav.py",
        "description":"WRF MAV JS, XA, GM and the four native wake-model ensembles."},
    "windfarm_overlap_method": {"type":"integer", "enum":[1,2,3,4], "default":4,
        "per_domain":True,"component_id":"pbl","consuming_read":"woof/core/wind_mav.py",
        "description":"WRF MAV native wake overlap method."},
    "windfarm_deg": {"type":"number", "default":0.0,
        "per_domain":True,"component_id":"pbl","consuming_read":"woof/core/wind_mav.py",
        "description":"WRF MAV xy layout rotation in degrees."},
    "windfarm_ij": {
        "type": "integer", "enum": [0, 1, 2], "default": 0,
        "per_domain": True, "component_id": "pbl",
        "consuming_read": "woof/wind_table.py",
        "description": "WRF turbine coordinates: Fitch 0 geographic / 1 domain indices; MAV 1 xy / 2 geographic."},
    "solar_generation": {
        "type": "string", "enum": ["wrf_480", "wrf_461"], "default": "wrf_480",
        "component_id": "radiation",
        "warnings": ["wrf_480 uses WRF 4.8 calc_coszen: julian minus 0.5, coefficient 0.040849. wrf_461 retains julian minus 1 and coefficient 0.04089 for reproduction. A change refuses an incompatible restart."]},
    "ysu_version": {
        "type": "string", "enum": ["wrf_480", "wrf_461"], "default": "wrf_480",
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 1},
        "warnings": ["wrf_480 is the corrected default: WRF 4.8 pinned MMM "
                     "550b5b4 removes the 0.2 offset and caps top-down entrainment "
                     "efficiency at 0.4. wrf_461 under default arithmetic "
                     "reproduces 2.8.9's YSU kernel (kernels/ysu_default.cu) under its "
                     "restart identity; under GPUWM_WRF_EXACT=1 it is unmodified "
                     "WRF 4.6.1 bl_ysu_run word for word, under its own restart "
                     "identity. Changing generation changes tendencies and is "
                     "refused on restart."]},
    "noahmp_snow_age": {
        "type": "string", "enum": ["wrf_470", "wrf_461"],
        "default": "wrf_470", "component_id": "land_surface",
        "read_when": {"sf_surface_physics": 4},
        "warnings": ["Snow-aging call placement only. wrf_470 moves SNOW_AGE "
                     "and SNOW_AGE_GLACIER before the solar branch, as WRF "
                     "v4.7.0 and v4.7.1 do. wrf_461 preserves daylight-only "
                     "aging. Neither selects the complete Noah-MP 5.1.2 "
                     "model. A flip is refused on restart because TAUSS and "
                     "the following dawn's BATS albedo would change."]},
    "scalar_adv_opt": {
        "type": "integer", "enum": [1], "default": 1,
        "consuming_read": "woof/core/moist.py"},
    "traj_opt": {
        "type":"integer", "enum":[0,1], "default":0,
        "consuming_read":"woof/core/trajectory.py"},
    "dm_has_traj": {
        "type":"boolean", "default":False, "per_domain":True,
        "consuming_read":"woof/core/trajectory.py"},
    "num_traj": {
        "type":"integer", "minimum":1, "maximum":1000, "default":1000,
        "consuming_read":"woof/core/state.py"},
    "trajectory_input": {
        "type":"string", "default":"", "per_domain":True,
        "consuming_read":"woof/core/trajectory.py"},
    "tracer_opt": {
        "type": "integer", "enum": [0, 2], "default": 0,
        "per_domain": True, "consuming_read": "woof/core/moist.py"},
    "tracer_adv_opt": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "per_domain": True, "consuming_read": "woof/core/moist.py"},
    "tracer_mix2_off": {
        "type": "boolean", "default": False, "per_domain": True,
        "consuming_read": "woof/core/dycore.py"},
    "tracer_mix6_off": {
        "type": "boolean", "default": False, "per_domain": True,
        "consuming_read": "woof/core/dycore.py"},
    "tracer_pblmix": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "per_domain": True, "consuming_read": "woof/core/physics.py"},
    "bl_mynn_tkeadvect": {
        "type": "boolean", "enum": [False, True], "default": False,
        "component_id": "pbl",
        "read_when": {"bl_pbl_physics": 5, "bl_mynn_tkeadvect": True},
        "warnings": ["Advects QKE_ADV through the scalar RK transport and copies it into and out of MYNN."]},
    "tmn_update": {"type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface", "consuming_read": "woof/core/physics.py",
        "state_fields": ["tmn", "tlag", "tyr", "tyra", "tdly", "nday", "nyear"],
        "restart_fields": ["tmn", "tlag", "tyr", "tyra", "tdly", "nday", "nyear"],
        "constraints": {"requires_when_enabled": {"lagday": 150}},
        "wrf_source": "WRF-4.6.1/phys/module_sf_tmnupdate.F"},
    "lagday": {"type": "integer", "minimum": 1, "default": 150,
        "component_id": "land_surface", "read_when": {"tmn_update": 1},
        "consuming_read": "woof/core/physics.py",
        "warnings": ["tmn_update=1 requires lagday=150, WRF share/module_check_a_mundo.F:2306-2312"]},
    "mfshconv": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 4},
        "warnings": ["WRF v4.6.1 resets mfshconv to 0 outside QNSE (share/module_check_a_mundo.F:1715-1730)."]},
    "nssl_icdx": {"type": "integer", "default": 6,
        "component_id": "microphysics", "read_when": {"mp_physics": 18},
        "wrf_reference": "module_mp_nssl_2mom.F:1487-1521,7135-7514",
        "warnings": ["Native fixed/density-dependent, MM and nonpositive power-law CUDA branches. Coefficient identity persists through continuation; default 6 preserves original kernels. Graupel >6 and hail 2 or >6 preserve native gathered wet-64 CD carry in parallel CUDA, with whole-column carry before fallout."]},
    "nssl_icdxhl": {"type": "integer", "default": 6,
        "component_id": "microphysics", "read_when": {"mp_physics": 18},
        "wrf_reference": "module_mp_nssl_2mom.F:1509-1521,7249-7514",
        "warnings": ["Native hail mass/number drag distinctions include selector-5 separate number law. Coefficient identity persists through continuation; default 6 preserves original kernels. Graupel >6 and hail 2 or >6 preserve native gathered wet-64 CD carry in parallel CUDA, with whole-column carry before fallout."]},
    "nssl_alphah": {"type": "number", "default": 0.0,
        "component_id": "microphysics", "read_when": {"mp_physics": 18},
        "wrf_reference": "module_mp_nssl_2mom.F:1284-1285,1845-1900,11874-12030",
        "warnings": ["Native gamma distribution and gmoi(0:3001) address bounds are checked; "
                     "the alpha=15 incomplete-gamma lookup clamp is not a namelist bound. "
                     "Nondefault shapes specialize CUDA collection, ventilation, sedimentation, "
                     "moment initialization/tails and radar. Qualified at graupel 2.5 / hail 0: real native replay, three-hour forecast with native/observation maps, whole nests, checkpoint and ensemble continuation."]},
    "nssl_alphahl": {"type": "number", "default": 1.0,
        "component_id": "microphysics", "read_when": {"mp_physics": 18},
        "wrf_reference": "module_mp_nssl_2mom.F:1284-1285,1845-1900,11874-12030",
        "warnings": ["Native gamma distribution and gmoi(0:3001) address bounds are checked; "
                     "the alpha=15 incomplete-gamma lookup clamp is not a namelist bound. "
                     "Nondefault shapes specialize CUDA collection, ventilation, sedimentation, "
                     "moment initialization/tails and radar. Qualified at graupel 2.5 / hail 0: real native replay, three-hour forecast with native/observation maps, whole nests, checkpoint and ensemble continuation."]},
    "nssl_cccn": {"type": "number", "minimum": -3.4028234663852886e38, "maximum": 3.4028234663852886e38, "default": 0.5e9,
                  "component_id": "microphysics", "read_when": {"mp_physics": 18},
                  "wrf_reference": "module_mp_nssl_2mom.F:1283,1997,10073",
                  "warnings": ["Finite float32 domain; stock WRF uses ABS(nssl_params(1)), with no coefficient bound. "
                               "Nests use the receiving domain's concentration in both coupling directions.",
                               "Qualified at 1e7 in the two-moment hail/predicted-CCN mode: two real native-call "
                               "strict replays (6228 words each), a three-hour forecast beside native and observations, "
                               "whole feedback-0/1 nests, exact nested checkpoint continuation, and two-member forecasts. "
                               "The mixed-phase strict fixture matches 10614 words using the native glibc referee ABI. "
                               "One-moment and coefficient qualifications have separate receipts."]},
    "bl_mynn_mixlength": {
        "type": "integer", "enum": [0, 1, 2], "default": 1,
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 5}},
    "tke_budget": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "pbl", "consuming_read": "woof/core/physics.py",
        "description": "WRF physics TKE budget terms for MYNN; also supported by the km_opt=2 LES owner."},
    "scalar_pblmix": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "pbl", "consuming_read": "woof/core/physics.py",
        "description": "WRF post-PBL diff4d of the mp=28 nc/ni/nwfa/nifa "
                       "scalars on the selected scheme's lower-interface "
                       "EXCH_H (consumer row scalar_diffusion_exchange). No "
                       "PBL and SASE are declared no-ops."},
    "bl_mynn_version": {
        "type": "string", "enum": ["wrf_461", "wrf_471", "gsd_41"], "default": "wrf_461",
        "component_id": "pbl", "read_when": {"bl_pbl_physics": 5},
        "warnings": ["Selects the MYNN generation. wrf_461 is WRF v4.6.1 "
                     "module_bl_mynn.F. wrf_471 selects the pinned WRF 4.7.1 sub-options. gsd_41 is the GSD MYNN v4.1 of the "
                     "NOAA-EMC WRF 3.9 branch, ported row by row: a downward "
                     "surface vapour flux (dew, frost) reaches the vapour "
                     "equation instead of being deleted, and mixing length "
                     "option 2 takes that generation's constants, caps and "
                     "blend. A namelist that spells the budget switch "
                     "bl_mynn_tkebudget (WRF 3.x) imports as gsd_41."]},
    "bl_mynn_cloud_tendency_form": {
        "type": "string", "enum": ["wrf_461", "gsd_41"], "default": "wrf_461",
        "component_id": "pbl",
        "read_when": {"bl_pbl_physics": 5, "bl_mynn_version": "gsd_41"},
        "warnings": ["wrf_461 conserves water and reconstructs heat with mixed condensate. "
                     "gsd_41 reproduces the source's pre-mixing condensate heat and "
                     "tendency-only negative-condensate clipping together. This defect "
                     "form is never an importer or recipe default."]},
    "bl_mynn_gsd41_unsquared_qtke": {
        "type": "boolean", "default": False,
        "component_id": "pbl",
        "read_when": {"bl_pbl_physics": 5, "bl_mynn_version": "gsd_41",
                      "bl_mynn_mixlength": 2},
        "warnings": ["true takes the gsd_41 option-2 mixing length's TKE "
                     "conversion 0.5*q as written (no square); false, the "
                     "default, takes 0.5*q**2 as its option 1 and every "
                     "later generation do."]},
    "cycling": {
        "type": "boolean", "default": False,
        "component_id": "pbl",
        "read_when": {"bl_pbl_physics": 5, "bl_mynn_version": "gsd_41"},
        "warnings": ["WRF &time_control cycling. true keeps the start's "
                     "QKE (unless its lowest-level maximum is below "
                     "0.0002), QC_BL and CLDFRA_BL at the first MYNN call, "
                     "as the operational WRF 3.9 fork does. Refused with "
                     "the v4.6.1 MYNN, whose cycled start zeroes them."]},
    "sst_update": {"type": "integer", "enum": [0, 1], "default": 0,
                   "component_id": "land_surface",
                   "warnings": ["1 reads Rust-prepared stream-4 SST, SEAICE, VEGFRA and ALBBCK by valid time; CUDA applies WRF surface and soil resets on water/ice transitions."]},
    "surface_analysis": {"type": "string", "default": "",
                         "component_id": "land_surface", "read_when": {"sst_update": 1},
                         "warnings": ["Path to the Rust surface-analysis manifest. Missing source cells preserve the initial or previously valid analysis."]},
    "mynn_sfclay_variant": {
        "type": "string", "enum": ["wrf_461", "gsl_wrf39"],
        "default": "wrf_461",
        "component_id": "surface_layer", "read_when": {"sf_sfclay_physics": 5},
        "warnings": ["Selects the generation of the MYNN surface layer. "
                     "wrf_461 is WRF v4.6.1 module_sf_mynn.F. gsl_wrf39 is "
                     "the same module in the GSL WRF 3.9 fork (NOAA-EMC/HRRR "
                     "v4.1.21): z/L from a 5-pass secant search that gives up "
                     "to 5 Ri (unstable) or 8 Ri (stable), z/L capped at 50 "
                     "instead of 20, the bulk Richardson number clamped at 50 "
                     "instead of 4 after the first step, zt instead of z0 in "
                     "the heat log numerators, and z0/L instead of zt/L as "
                     "psih's lower limit at the first level. Over rough land "
                     "(forest, urban) on stable nights the two give "
                     "different u* and exchange coefficients."]},
    "urban_ucm_generation": {
        "type": "string", "enum": ["wrf_471", "wrf_480"], "default": "wrf_471",
        "component_id": "urban", "read_when": {"sf_urban_physics": 1},
        "warnings": ["wrf_480 selects WRF v4.8.0 ground vegetation, tree shading and canopy flux feedback with declared defined entry memory; changing generation refuses restart."]},
    "urban_treeoption": {"type": "integer", "enum": [-1, 0, 1], "default": -1,
        "component_id": "urban", "read_when": {"sf_urban_physics": 1, "urban_ucm_generation": "wrf_480"}},
    "urban_ttscheme": {"type": "integer", "enum": [-1, 0, 1], "default": -1,
        "component_id": "urban", "read_when": {"sf_urban_physics": 1, "urban_ucm_generation": "wrf_480"}},
    "urban_fvg": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": -1.0,
        "component_id": "urban", "read_when": {"sf_urban_physics": 1, "urban_ucm_generation": "wrf_480"}},
    "urban_table_dir": {"type": "string", "default": "", "component_id": "urban",
        "read_when": {"sf_urban_physics": 1},
        "warnings": ["Explicit URBPARM entries replace the pinned release table and their decoded coefficient words bind restart. Empty retains the generation's pinned bundled tables."]},
    # acoustic / small step
    "time_step_sound": {"type": "integer", "minimum": 2, "default": 4},
    "smdiv": {"type": "number", "minimum": 0.0, "default": 0.1},
    "emdiv": {"type": "number", "minimum": 0.0, "default": 0.0},
    # explicit / constant-K mixing
    "khdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "kvdif": {"type": "number", "minimum": 0.0, "default": 0.0},
    "c_s": {"type": "number", "minimum": 0.0, "default": 0.25},
    "diff_opt": {"type": "integer", "enum": [1, 2], "default": 2,
                 "description": "1: coordinate-surface diffusion for km_opt=2/4; 2: metric stress/scalar diffusion."},
    "mix_full_fields": {"type": "boolean", "default": True,
                        # Left as it was by lane 286-mix-full-fields on purpose: it is part of
                        # the parameter's registry physics identity, and a selection
                        # receipt bound before lane 286-mix-full-fields must keep
                        # resolving.  Both values are admitted under either operator
                        # (woof/config.py, validate_km_opt; docs/public/CONFIGURATION.md).
                        "description": "WRF logical retained under coordinate diffusion; that operator always mixes theta relative to its initial field."},
    "diff_6th_thresh": {"type": "number", "minimum": 0.0, "default": 0.10},
    # upper-level damping
    "damp_opt": {"type": "integer", "enum": [0, 3], "default": 0},
    "zdamp": {"type": "number", "minimum": 0.0, "default": 5000.0},
    "dampcoef": {"type": "number", "minimum": 0.0, "default": 0.2},
    "w_damping": {"type": "integer", "enum": [0, 1], "default": 0},
    # vertical coordinate and base state
    "hybrid_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "etac": {"type": "number", "minimum": 0.0, "default": 0.2},
    "base_temp": {"type": "number", "minimum": 0.0, "default": 290.0},
    "hypsometric_opt": {"type": "integer", "enum": [1, 2], "default": 1},
    # transport
    "h_sca_adv_order": {"type": "integer", "enum": [2, 3, 4, 5, 6], "default": 2,
        "description": "Horizontal scalar, W and geopotential transport. Explicit orders 2, 3, 4 and 6 have retained native real-input replays and two-hour forecasts. Order 2 excludes exactly 14504 native undefined words with a retained unchanged-routine initialization witness. See docs/dynamics-options.md."},
    # Only replay-qualified added orders are advertised here. Real-input
    # native captures and rerun commands are listed in docs/dynamics-options.md.
    "v_sca_adv_order": {"type": "integer", "enum": [2, 3, 4, 5, 6], "default": 3,
        "description": "Vertical scalar and W transport. Order 2 has a retained nonzero strict WRF 4.7.1 replay; see docs/dynamics-options.md."},
    "v_mom_adv_order": {"type": "integer", "enum": [2, 3, 4, 5, 6], "default": 3,
        "description": "Vertical u/v transport. Orders 2, 4 and 6 have a retained strict WRF 4.7.1 replay; see docs/dynamics-options.md."},
    "h_mom_adv_order": {"type": "integer", "enum": [0, 2, 3, 4, 5, 6], "default": 5,
        "description": "Horizontal u/v transport. Orders 0, 2, 3 and 4 have a retained strict WRF 4.7.1 replay; see docs/dynamics-options.md."},
    "rk_ord": {"type": "integer", "enum": [2, 3], "default": 3,
        "description": "WRF Runge-Kutta stage schedule; RK2 has a retained native real-input stage and scalar-update replay plus a two-hour forecast. See docs/dynamics-options.md."},
    "momentum_adv_opt": {"type": "integer", "enum": [1, 3], "default": 1,
        "description": "1: ordinary momentum transport; 3: WRF WENO momentum transport on the final RK stage. Retained strict replay and measured forecast cost: docs/dynamics-options.md."},
    "moist_adv_opt": {"type": "integer", "enum": [0, 1], "default": 1},
    "tke_adv_opt": {"type": "integer", "enum": [1], "default": 1},
    "sfs_opt": {"type": "integer", "enum": [0], "default": 0},
    "m_opt": {"type": "integer", "enum": [0, 1], "default": 0,
        "description": "WRF subfilter stress history output. Retained strict raw and production geometry replay; measured 1.119862x default mixing stage; see docs/dynamics-options.md."},
    "tke_mix2_off": {"type": "boolean", "default": False},
    "tke_mix6_off": {"type": "boolean", "default": False},
    # WRF 4.3+ implicit-explicit vertical advection (woof.core.ieva), off
    # by default.  Its warning is the declared divergence of its w solve
    # (A179), which the option's own documentation also records.
    "zadvect_implicit": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "For zadvect_implicit_variant='wrf_471', DIVERGENCE from WRF "
            "v4.7.1, declared: the implicit w solve's "
            "two boundary terms take the units of the w terms beside them. "
            "WRF's lower boundary (dyn_em/module_ieva_em.F:1231-1244) builds "
            "the surface w increment from the mass-coupled u/v tendencies, "
            "about one column mass too large, and a steep-ridge run went "
            "NaN in three steps; woof uncouples each tendency by its map "
            "factor and face column mass first. WRF's upper boundary "
            "(:1248-1253) leaves the geopotential change over dt undivided "
            "by g; woof divides it. Every other IEVA routine matches WRF "
            "v4.7.1 word for word, and the w solve matches WRF's routine "
            "with those two corrections (tools/ieva_wrf_oracle)."]},
    "zadvect_implicit_variant": {
        "type": "string", "enum": ["wrf_471", "wrf_legacy"],
        "default": "wrf_471",
        "warnings": [
            "Selects the WRF numerical generation when zadvect_implicit=1. "
            "wrf_471 preserves module_ieva_em. wrf_legacy uses "
            "module_advect_em's current-mass coefficients and one-sided "
            "horizontal Courant allowance. Both uncouple the lower w "
            "boundary's momentum tendencies, a declared unit correction."]},
    # microphysics heating controls
    "no_mp_heating": {"type": "integer", "enum": [0, 1], "default": 0},
    "mp_tend_lim": {"type": "number", "minimum": 0.0, "default": 10.0},
    # lateral boundaries
    "specified": {"type": "boolean", "default": False},
    "open_x": {"type": "boolean", "default": False},
    "open_y": {"type": "boolean", "default": False},
    "spec_bdy_width": {"type": "integer", "minimum": 1, "default": 5},
    "spec_zone": {"type": "integer", "minimum": 1, "default": 1},
    "relax_zone": {"type": "integer", "minimum": 2, "default": 4},
    # projection
    "map_proj": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0},
    # boundary layer
    "ysu_topdown_pblmix": {"type": "integer", "enum": [0, 1], "default": 1},
    # radiation
    "cam_abs_freq_s": {
        "type": "number", "minimum": 0.0, "default": 21600.0,
        "component_id": "radiation",
        "read_when": {"ra_lw_physics": 3, "ra_sw_physics": 3},
        "consuming_read": "woof/core/cam3.py",
        "description": "WRF CAM absorption/emissivity refresh interval in seconds. Zero refreshes at every radiation call; retained absorption arrays and their clock belong to the radiation checkpoint."},
    "swrad_scat": {"type": "number", "minimum": 0.0, "default": 1.0},
    # WRF v4.7.1 slope-dependent surface shortwave and terrain shadowing
    # (woof.core.topo_radiation, lane 281-namelist-gaps), off by default.
    "slope_rad": {"type": "integer", "enum": [0, 1], "default": 0},
    "topo_shading": {"type": "integer", "enum": [0, 1], "default": 0},
    # WRF radiation-driver options as the operational HRRR fork runs them
    # (lane 286-aer-swint), both off by default.  swint_opt 1 refits the
    # surface shortwave at every radiation call and evaluates it at the
    # current sun on every step (woof.core.swint); aer_opt 3 hands the
    # legacy RRTMG shortwave per-band aerosol optics from the Thompson
    # aerosol numbers (woof.core.rrtmg_aerosol_optics).  aer_opt 1 is the
    # root-domain six-type climatology and aer_opt 2 the user AOD/Angstrom
    # producer (woof.core.aer_opt2), both on legacy RRTMG shortwave only
    # (woof.config.validate_radiation_driver_options).
    "madwrf_opt": {"type":"integer","enum":[0,1,2],"default":0,
        "description":"MAD-WRF: 1 transports clouds with empty MP96 (1.0030x final scalar transport plus hook versus MAD-off MP96); 2 transports and nudges three cloud tracers (named opt-in, never default; 1.7384x that stage on RTX PRO 6000 Blackwell). See docs/public/wrf-comparison/madwrf-port.md",
        "consuming_read":"woof/core/madwrf_runtime.py:nudge"},
    "madwrf_cldinit": {"type":"integer","enum":[0,1],"default":0,
        "consuming_read":"woof/core/madwrf_runtime.py:initialize"},
    "madwrf_dt_relax": {"type":"number","default":60.0,"units":"s",
        "consuming_read":"woof/core/madwrf_runtime.py:nudge"},
    "madwrf_dt_nudge": {"type":"number","default":60.0,"units":"min",
        "consuming_read":"woof/core/madwrf_runtime.py:nudge"},
    "ra_call_offset": {"type": "integer", "enum": [-1, 0], "default": 0,
        "description": "WRF fixed-step radiation call phase: -1 old, 0 new. Adaptive radiation timing ignores this offset.",
        "consuming_read": "woof/core/physics.py:_radiation_step_due"},
    "swint_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0,
        "implementation_rows": {"farms": {
            "value": 2,
            "state": ["swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2"],
            "outputs": ["SWDOWN2", "SWDDIR2", "SWDDNI2", "SWDDIF2", "SWDOWNC2", "SWDDNIC2"],
            "authority": "WRF v4.7.1 phys/module_ra_farms.F and module_radiation_driver.F",
            "maturity": "wrf-matched-run",
            "native_replay": "tools/cloud_aerosol_wrf471_oracle: 768 output words, zero differences",
            "integration_proof": "stock-namelist woof go; both nest feedback modes; byte-exact checkpoint continuation; two distinct ensemble members; disabled forecast byte identity",
            "limits": ["Full FARMS forecast hook costs 1.194390x Dudhia SW per call (1.080132 ms); 59 versus 11 warm calls give 6.406275x accumulated hook time on 4380x39 columns. See docs/evidence/retime-20261102/farms-fused-summary.json",
                       "native FARMS diagnostics remain undimmed during an eclipse"]}}},
    "couple_farms": {"type": "boolean", "default": False},
    "ra_sw_eclipse": {"type": "integer", "enum": [0, 1], "default": 1,
        "component_id": "radiation", "consuming_read": "woof/eclipse_windows.py",
        "implementation_rows": {"wrf_eclipse": {
            "authority": "WRF 4.6.1 module_ra_eclipse.F and distributed eclipse_besselian_elements.dat",
            "maturity": "wrf-matched-run",
            "state": ["ecobsc", "ecmask", "elat_track", "elon_track", "eclipse_initialized"],
            "outputs": ["ECOBSC", "ECMASK", "ELAT_TRACK", "ELON_TRACK"],
            "native_replay": "368896 words, zero differences; gfortran 12/glibc 2.43 native build",
            "integration_proof": "2024-04-08 15-21 UTC woof go; Rust maps beside native and station observations; active nests in both feedback modes; byte-exact checkpoint continuation; cloud1/cloud3 ensemble",
            "limits": ["whole-forecast proof uses legacy RRTMG", "native FARMS surface diagnostics remain undimmed"]}}},
    "cu_rad_feedback": {"type": "boolean", "default": False,
        "implementation_rows": {"convective_radiation_clouds": {
            "authority": "WRF module_physics_init.F ICLOUD_CU; module_radiation_driver.F:1264-1345",
            "schemes": [1, 3, 5],
            "derived_icloud_cu": {"1": 2, "3": 1, "5": 1},
            "state": ["qc_cu", "qi_cu", "cldfra_dp", "cldfra_sh", "cldfra_cu"],
            "maturity": "wrf-matched-run",
            "native_replay": "29952 words per derived merge mode, zero differences on real CU1/CU3 inputs",
            "integration_proof": "stock-namelist woof go; both nest feedback modes; byte-exact checkpoint continuation; CU1/CU3 ensemble",
            "limits": ["CU11 and CU93 retain their parent-scheme admission requirements"]}}},
    "aer_opt": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0,
                "implementation_rows": {"aeropt1": {
                    "value": 1, "state": ["aerodm"], "outputs": ["AODTOT"],
                    "doors": {"ra_sw_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy", "nested": False},
                    "authority": "WRF 4.6.1 Registry.EM_COMMON:3240 and phys/module_radiation_driver.F:1826",
                    "rrtmgp": "refused: no checked six-type mapping"},
                    "aeropt2": {
                    "value": 2, "state": [], "outputs": ["AOD5503D", "AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D"],
                    "doors": {"ra_sw_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy"},
                    "authority": "WRF 4.6.1 phys/module_radiation_driver.F, the aer_opt = 2 aerosol_in branch",
                    "rrtmgp": "refused: no checked user-aerosol spectral mapping"},
                    "aeropt3_stock": {
                    "value": 3, "source": "wrf_461",
                    "state": ["nwfa", "nifa"],
                    "outputs": ["AOD5502D", "ANGEXP2D", "AERSSA2D", "AERASY2D"],
                    "doors": {"ra_sw_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy", "mp_physics": 28},
                    "authority": "WRF 4.6.1 module_ra_aerosol.F; module_radiation_driver.F aer_opt=3",
                    "maturity": "wrf-matched-run",
                    "native_replay": "430336 words, zero differences on 256 real columns",
                    "integration_proof": "stock-namelist woof go; both nest feedback modes; byte-exact checkpoint continuation; distinct CU1/CU3 ensemble",
                    "limits": ["defined zero pressure halo matches the captured native producer-unfilled zero words",
                               "legacy configured forecasts retain the existing hrrr source default"]}}},
    "alb_sol": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "alb_sol=1 updates sun-angle-dependent ALBSOL/ALBBCKSOL on "
            "radiation steps, supplies ALBSOL to shortwave radiation and "
            "the RUC land surface, and uses the corrected background "
            "albedo in RUC's snow albedo. It needs active shortwave and "
            "MODIS21 land-use categories. Other configurations retain "
            "the unchanged albedo carrier at 0."]},
    "rrtmg_smoke_manifest": {
        "type": "string", "default": "", "component_id": "radiation",
        "read_when": {"ra_sw_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy"},
        "consuming_read": "woof/core/rrtmg_smoke_identity.py",
        "description": "Explicit timestamped three-dimensional prescribed smoke forcing. Empty selects no prescribed smoke; enabled requires legacy shortwave, aer_opt=3 and aerosol-aware Thompson. Manifest and member content identities bind forecasts and restarts."},
    "rrtmg_cloud_optics_form": {
        "type": "string", "enum": ["wrf_461", "noaa_wrf39"],
        "default": "wrf_461", "component_id": "radiation",
        "read_when": {"ra_rrtmg_variant": "rrtmg_legacy"},
        "consuming_read": "woof/core/rrtmg_legacy.py",
        "description": "Source form of the legacy RRTMG cloud wrapper: WRF v4.6.1 or the NOAA WRFV3.9 fork's shortwave radius defaults, snow ice fraction and Thompson cold-start radii."},
    "rrtmg_aer3_source": {
        "type": "string", "enum": ["operational_fork", "wrf_461"], "default": "operational_fork",
        "component_id": "radiation",
        "read_when": {"aer_opt": 3},
        "consuming_read": "woof/core/wrf_stock_aer3.py",
        "description": "Stock WRF namelist aer_opt=3 imports wrf_461; hrrr preserves existing configured forecasts. Stock optics use the native layer floor and rural aerosol diagnostics."},
    "rrtmg_aer3_type_source": {
        "type": "string", "enum": ["rural", "landmask"],
        "default": "rural", "component_id": "radiation",
        "read_when": {"aer_opt": 3, "ra_sw_physics": 4,
                      "ra_rrtmg_variant": "rrtmg_legacy"},
        "consuming_read": "woof/core/rrtmg_legacy.py",
        "description": "Aerosol type of the aer_opt=3 shortwave optics: rural type 1 everywhere (the NOAA WRFV3.9 HRRR fork) or the NOAA MPAS fork's land/water type, rural over land and maritime over water."},
    # WRF v4.7.1 sub-grid terrain drag (woof.core.terrain_drag, lane
    # 282-terrain-drag), off by default: topo_wind under YSU, gwd_opt under
    # every PBL scheme.
    "topo_wind": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "gwd_opt": {"type": "integer", "enum": [0, 1, 3], "default": 0},
    "o3input": {
        "type": "integer", "enum": [0, 2], "default": 2,
        "warnings": [
            "o3input=0 is implemented only by ra_rrtmg_variant="
            "'rrtmg_legacy', where the WRF wrapper constructs O3DATA. "
            "RTE+RRTMGP admits only 2."]},
    "use_mp_re": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "warnings": [
            "use_mp_re=0 is implemented only by ra_rrtmg_variant="
            "'rrtmg_legacy'; it disables the WRF microphysics effective-"
            "radius scheme table and makes the wrapper calculate radii."]},
    "ra_rrtmg_variant": {
        "type": "string",
        "enum": ["rte-rrtmgp", "rrtmg_legacy"],
        "default": "rte-rrtmgp"},
    # experiment feedback already executes in the native multi-domain runner.
    "feedback": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "DIVERGENCE from WRF's default 1: woof defaults feedback to 0 "
            "to preserve every assembled one-way trajectory. feedback=1 is "
            "experimental; the native woof run multi-domain executor and "
            "the prepared-tree executor, the native HRRR route's included, "
            "both run it."]},
    # surface layer -- newly ported this pass
    "sst_skin": {"type": "integer", "enum": [0, 1], "default": 0,
                 "component_id": "land_surface",
                 "description": "WRF 4.6.1 skin SST adjustment; carries SSTSK (K) and DTW (C), writes TSK over water",
                 "runtime_read_site": "woof/core/physics.py"},
    "isfflx": {"type": "integer", "enum": [0, 1], "default": 1},
    "isftcflx": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "iz0tlnd": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    # Land-surface vegetation and albedo switches.  Noah's kernel branches
    # are at noah.cu:1036/:1111/:1112/:1116/:1117 (usemonalb, rdlai2d) and
    # :181 (opt_thcnd).  RUC reads rdlai2d in SOILVEGIN (ruc.cu:171, :178,
    # :292 and the fused driver prologue) and usemonalb through
    # landuse_init's background albedo (woof/core/landuse.py
    # initialize_landuse, WRF module_physics_init.F:1611); opt_thcnd is
    # Noah only.
    "usemonalb": {
        "type": "boolean", "default": False,
        "warnings": [
            "usemonalb=true is implemented by Noah (sf_surface_physics=2) "
            "and RUC (sf_surface_physics=3): landuse_init keeps real.exe's "
            "ALBEDO12M background albedo interpolated to the start date "
            "instead of LANDUSE.TBL's seasonal row, and a snow-covered "
            "cell takes SNOALB. A surface source without the monthly "
            "albedo field is refused by name."]},
    "rdlai2d": {
        "type": "boolean", "default": False,
        "warnings": [
            "rdlai2d=true is implemented by Noah (sf_surface_physics=2) "
            "and RUC (sf_surface_physics=3): the LSM keeps the LAI12M field "
            "interpolated to the start date instead of the VEGPARM table "
            "value. Under RUC a road that seeds no LAI field is refused at "
            "the first land-surface step rather than running the "
            "allocation default."]},
    "opt_thcnd": {"type": "integer", "enum": [1, 2], "default": 1},
    "ua_phys": {"type": "boolean", "default": False,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 2},
        "warnings": ["UA snow and canopy physics use a separate Noah kernel image. Strict WRF 4.7.1 real-input replay: 0/42,977,064 words; default-fast forecast and Rust maps qualified. Reproduction: tools/sfc_irr_seaice_wrf471/README.md."]},
    # WRF &physics fractional_seaice: which of WRF's two sea-ice
    # thresholds the surface runs (module_surface_driver.F:1365-1368 of the
    # HRRR v4.1.21 fork).  Read by the RUC seam, the RUC land-use and
    # mosaic initialisation and the CLM lake beside it.
    "fractional_seaice": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface",
        "read_when": {"sf_surface_physics": [2, 3, 4, 7, 8]},
        "warnings": ["Named opt-in: WRF 4.7.1 fractional ice uses the 0.02 threshold, separate ice/open-water surface calls and the native post-LSM blend. Measured shared-card cost: Noah surface plus land 1.526x; Noah-MP surface layer 1.958x versus the same land model with option off. Native two-call profile 1.966x. Strict retained replay and default-fast forecast qualified with sfclay=1; only native run-to-run undefined USTM words excluded. Reproduction: tools/sfc_irr_seaice_wrf471/README.md."]},
    "seaice_albedo_opt": {"type":"integer", "enum":[0,1,2], "default":0,
        "component_id":"land_surface", "read_when":{"sf_surface_physics":[2,4]},
        "warnings":["WRF prescribed ice: 1 Mills temperature-dependent albedo; 2 input ALBSI. Each has strict WRF 4.7.1 real-input replay 0/15,593,448 ice words and a default-fast whole forecast. Reproduction: tools/sfc_irr_seaice_wrf471/README.md."]},
    "seaice_snowdepth_opt": {"type":"integer", "enum":[0,1], "default":0,
        "component_id":"land_surface", "read_when":{"sf_surface_physics":[2,4]},
        "warnings":["Option 1 requires SNOWSI and retains WRF's fixed 0.3 snow density. Strict native replay and whole forecast: tools/sfc_irr_seaice_wrf471/README.md."]},
    "seaice_thickness_opt": {"type":"integer", "enum":[0,1], "default":0,
        "component_id":"land_surface", "read_when":{"sf_surface_physics":[2,4]},
        "warnings":["Option 1 requires ICEDEPTH and uses WRF's 0.1..10 m clamp. Strict native replay and whole forecast: tools/sfc_irr_seaice_wrf471/README.md."]},
    "seaice_thickness_default": {"type":"number", "default":3.0, "exclusiveMinimum":0},
    "seaice_snowdepth_min": {"type":"number", "default":0.001, "exclusiveMinimum":0},
    "seaice_snowdepth_max": {"type":"number", "default":1.e10, "exclusiveMinimum":0},
    "sf_lake_physics": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "per_domain": True, "component_id": "land_surface",
        "warnings": ["Option 1 runs WRF's CLM lake column on LAKEMASK "
                     "cells, with ten water and ten sediment layers and "
                     "up to five snow layers. Default 0 retains prescribed "
                     "water temperatures."]},
    "use_lakedepth": {
        "type": "integer", "enum": [0, 1], "default": 1,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["WRF default 1 requires input LAKE_DEPTH. Explicit 0 "
                     "uses lakedepth_default for every lake column."]},
    "lakedepth_default": {
        "type": "number", "default": 50.0,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["Depth in metres used by WRF lakeini when a supplied "
                     "depth is nonpositive or use_lakedepth=0. A nonpositive "
                     "default uses WRF's reference layer geometry depth."]},
    "lake_min_elev": {
        "type": "number", "default": 5.0,
        "per_domain": True, "component_id": "land_surface",
        "read_when": {"sf_lake_physics": 1},
        "warnings": ["Minimum water-cell elevation in metres used by WRF "
                     "when the input does not provide a lake mask."]},
    # WRF &physics sf_ocean_physics = 1: the 1-D ocean mixed layer
    # (phys/module_sf_oml.F, called by module_sf_ocean_driver.F after the
    # land surface and before the lake).  2 is the 3-D PWP ocean, run on
    # the GPU by woof/ocean/mixed_layer (pwp.cu).
    "sf_ocean_physics": {
        "type": "integer", "enum": [0, 1, 2], "default": 0,
        "component_id": "land_surface", "consuming_read": "woof/core/physics.py",
        "warnings": ["Option 1 runs WRF Pollard OML1D; option 2 runs CUDA "
                     "3DPWP circulation and vertical mixing with prognostic "
                     "temperature, salinity and currents. Both couple SST to "
                     "TSK every surface step. An absent ocean analysis is a "
                     "loudly labelled configured uniform MLD fallback. Full "
                     "TC and observation qualification is pending."]},
    "seaice_prognostic": {'type': 'boolean', 'default': True, 'per_domain': False, 'component_id': 'land_surface', 'consuming_read': 'woof/ocean/seaice/__init__.py', 'warnings': ['Prognostic thermodynamic ice and snow on non-lake ocean cells, activated when initial ocean ice is present. False selects prescribed sea ice.']},
    "seaice_initial_thickness": {'type': 'number', 'default': 1.0, 'minimum': 0.0, 'per_domain': False, 'component_id': 'land_surface', 'consuming_read': 'woof/ocean/seaice/__init__.py', 'warnings': ['Uniform initial ice thickness prior in metres unless an input thickness field is supplied; not an observation.']},
    "seaice_new_ice_thickness": {'type': 'number', 'default': 0.5, 'minimum': 0.0, 'per_domain': False, 'component_id': 'land_surface', 'consuming_read': 'woof/ocean/seaice/__init__.py', 'warnings': ['New-ice thickness in metres used for thermodynamic concentration growth.']},
    "seaice_ocean_heat_flux": {'type': 'number', 'default': 2.0, 'per_domain': False, 'component_id': 'land_surface', 'consuming_read': 'woof/ocean/seaice/__init__.py', 'warnings': ['Prescribed basal ocean-to-ice heat flux in W m-2; no ocean circulation or seasonal skill is implied.']},
    "oml_hml0": {
        "type": "number", "default": 50.0,
        "component_id": "land_surface", "consuming_read": "woof/core/surface_ocean.py",
        "read_when": {"sf_ocean_physics": 1},
        "warnings": ["Initial mixed-layer depth in metres (WRF default 50). "
                     "0 or less reads a 2-D H0ML from the input file."]},
    "oml_gamma": {
        "type": "number", "default": 0.14,
        "component_id": "land_surface", "consuming_read": "woof/core/physics.py",
        "read_when": {"sf_ocean_physics": 1},
        "warnings": ["Deep-water lapse rate below the mixed layer in K/m "
                     "(WRF default 0.14); must be positive."]},
    "oml_relaxation_time": {
        "type": "number", "default": 0.0,
        "component_id": "land_surface", "consuming_read": "woof/core/physics.py",
        "read_when": {"sf_ocean_physics": 1},
        "warnings": ["Relaxation time in seconds back to the initial "
                     "mixed-layer state; 0 (WRF default) means none."]},
    "mosaic_lu": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["RUC weighted vegetation parameters and irrigation use "
                     "LANDUSEF. WRF's default is 0; 1 requires category fractions."]},
    "mosaic_soil": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["RUC weighted soil parameters use SOILCTOP. WRF's default "
                     "is 0; 1 requires category fractions."]},
    "ruc_soilprop": {
        "type": "string", "enum": ["wrf_45", "wrf_461"], "default": "wrf_45",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["Selects which WRF lineage's LSMRUC SOILPROP sets soil-water "
                     "diffusivity and hydraulic conductivity. wrf_45 (WRF v4.0-4.5, also "
                     "the operational RAP/HRRR branch) normalises both by the moisture "
                     "above the residual, (theta - qmin)/(theta_sat - qmin), and uses "
                     "mineral conductivity 2.0 at every quartz fraction. wrf_461 (WRF "
                     "v4.6.1) uses total moisture over porosity and 3.0 below 20 percent "
                     "quartz; in dry soil its water diffusivity is 2.5 to 8 times "
                     "larger, measured to raise a 3 km afternoon top soil level from "
                     "0.161 to 0.187 m3/m3 in one hour from the levels below, where the "
                     "operational model's own top level fell to 0.157. The default "
                     "changed from the v4.6.1 form to wrf_45: every RUC configuration "
                     "changes answers."]},
    "thompson_version": {
        "type": "string", "enum": ["wrf_461", "wrf_39_noaa"],
        "default": "wrf_461", "component_id": "microphysics",
        "read_when": {"mp_physics": 28}, "per_domain": False,
        "consuming_read": "woof/core/microphysics_aerosol.py",
        "warnings": ["wrf_39_noaa uses the NOAA WRF 3.9 fork's size distributions "
                     "and its own pinned tables. Fork column comparisons are in "
                     "tests/test_thompson_wrf39.py; the v4.6.1 qualification "
                     "receipts do not qualify this generation."]},
    "thompson_fork_snow_fall": {
        "type": "string", "enum": ["blend", "wrf_39_noaa"], "default": "blend",
        "component_id": "microphysics", "per_domain": False,
        "read_when": {"mp_physics": 28, "thompson_version": "wrf_39_noaa"},
        "consuming_read": "woof/core/microphysics_aerosol.py",
        "warnings": ["wrf_39_noaa selects the fork's singular melting-snow "
                     "fall speed. It is a defined fork defect and defaults off. "
                     "blend retains the later rain-share blend."]},
    "ruc_snow": {
        "type": "string", "enum": ["wrf_45", "wrf_461"], "default": "wrf_461",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["Which WRF lineage's RUC snow scheme runs. wrf_45 (WRF v4.0-4.5, "
                     "which the operational RAP/HRRR branch carries) takes a constant "
                     "snow conductivity, snow cover from depth over a critical depth "
                     "taken after compaction, fresh-snow albedo from the depth on the "
                     "ground, a melt cap independent of the step, melt bookkeeping "
                     "scaled by cover under the snow mosaic, and SNOWFALLAC grown by "
                     "new snow less its melt. wrf_461 (WRF v4.6.1) takes the Sturm "
                     "conductivity, the blended depth and roughness cover rebuilt "
                     "after the snow column, and density-gated melt limits. The "
                     "generic default remains wrf_461. The operational namelist importer "
                     "and HRRR configuration recipe explicitly select wrf_45, "
                     "which changes answers where there is snow."]},
    "ruc_2m_diagnostic": {
        "type": "string", "enum": ["flux", "log_profile"], "default": "flux",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["How SFCDIAGS_RUCLSM writes T2, TH2 and Q2. flux (public WRF) is "
                     "the flux form. log_profile adds the block the operational RAP/HRRR "
                     "branch carries and no public WRF has: where the air is warmer or "
                     "moister than the surface, T2 and Q2 follow a logarithmic profile "
                     "between the surface and half the lowest layer. Measured on a 3 km "
                     "cut of an operational-HRRR start at night: T2 0.33 to 0.36 K "
                     "lower, 2 m dewpoint 0.01 K lower, against the operational model's "
                     "own files, which match the flux form's T2 within 0.04 K. Default "
                     "flux."]},
    "ruc_qvg_cold_start": {
        "type": "string", "enum": ["air", "wrf"], "default": "wrf",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["LSMRUC's cold start of the ground vapour and condensate when the "
                     "run starts without them. wrf (public WRF, the default) starts QCG "
                     "from the lowest-level condensate and QVG from saturation at the "
                     "skin times moisture availability. air (the operational RAP/HRRR "
                     "branch's fallback; that branch cycles QVG) starts an invalid QVG "
                     "from the lowest-level vapour with no ground condensate. Because "
                     "SOILTEMP carries the old QVG as vapour storage, air pulls the skin "
                     "toward the air's dewpoint on the first steps: measured 2.7 K "
                     "colder after 20 steps on a moist test column, and -0.013 to +0.018 "
                     "K of 2 m dewpoint on a 3 km cut of an operational-HRRR start. Read "
                     "only on a cold start."]},
    "ruc_irrigation": {
        "type": "string", "enum": ["wrf_45", "wrf_461"], "default": "wrf_461",
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 3},
        "warnings": ["Selects which WRF lineage's LSMRUC irrigation holds root-layer "
                     "soil moisture up after SFCTMP. wrf_45 (WRF v4.0-4.5, also the "
                     "operational RAP/HRRR branch) is a hard floor scaled by the cell's "
                     "cropland fraction and gated on leaf area index above 1.1 "
                     "(cropland) or 0.7 (dominant crop/natural mosaic), applied whatever "
                     "mosaic_lu says: with mosaic_lu=1 it reads the LANDUSEF fractions, "
                     "with mosaic_lu=0 the dominant category counts as the whole cell. "
                     "wrf_461 (WRF v4.6.1, mosaic_lu=1 only) relaxes every root layer "
                     "toward the full 1.1 x wilting point each step for any cell with "
                     "any crop or crop/natural fraction and a greenness factor above "
                     "0.75. The generic default remains wrf_461. The operational "
                     "namelist importer and HRRR configuration recipe explicitly "
                     "select wrf_45, which changes the added irrigation water."]},
    "sf_surf_irr_scheme": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0,
        "warnings": ["WRF 4.7.1 Noah packages: channel=1, drip=2, sprinkler=3, separate from Noah-MP opt_irr. IRRIGATION is percent (0..100); requires IRRIGATION and IRR_PH_FIELD static fields. All three have strict native real-input replay and default-fast whole forecasts; tools/sfc_irr_seaice_wrf471/README.md."]},
    "irr_daily_amount": {"type": "number", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_start_hour": {"type": "integer", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_num_hours": {"type": "integer", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_start_julianday": {"type": "integer", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_end_julianday": {"type": "integer", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_freq": {"type": "integer", "default": 1, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "irr_ph": {"type": "integer", "default": 0, "component_id": "land_surface", "read_when": {"sf_surf_irr_scheme": [1, 2, 3]}},
    "sf_surface_mosaic": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": ["Noah only: WRF v4.7.1 lsm_mosaic tile state. "
                     "Stock-WRF export of mosaic land state and moving-nest "
                     "tile interpolation are not reproduced. Urban option 1 "
                     "runs inside the tile loop, by default only where a "
                     "cell's dominant category is urban (mosaic_urban_canopy); "
                     "mosaic option cannot work with urban options 2 and 3 "
                     "(WRF share/module_check_a_mundo.F:505-518)."]},
    # read_when: the settings a knob is read under at all
    # (woof.physics_registry.registry_knob_is_read), so a WRF namelist or a
    # TOML that carries a tile count with mosaic off runs what a 2.8.0
    # preparation ran and is not refused as new physics (A153).
    "mosaic_cat": {"type": "integer", "minimum": 1, "default": 3,
                   "component_id": "land_surface",
                   "read_when": {"sf_surface_mosaic": 1}},
    "mosaic_urban_canopy": {
        "type": "string", "enum": list(MOSAIC_URBAN_CANOPY_RULES),
        "default": MOSAIC_URBAN_CANOPY_DEFAULT,
        "read_when": {"sf_surface_mosaic": 1, "sf_urban_physics": 1},
        "warnings": ["Read only with sf_surface_mosaic = 1 and "
                     "sf_urban_physics = 1. dominant is WRF v4.7.1: the urban "
                     "canopy runs only where a cell's dominant category is "
                     "urban. every_tile also runs the urban tiles of a mostly "
                     "rural cell at their own land-use weights, sharing the "
                     "largest urban tile type's URBPARM urban fraction, a "
                     "woof option WRF does not have; the 10 m wind override "
                     "stays on dominant-urban cells."]},
    "urban_water_balance": {
        "type": "string", "enum": list(URBAN_WATER_BALANCE_GENERATIONS),
        "default": URBAN_WATER_BALANCE_DEFAULT, "component_id": "urban",
        "read_when": {"sf_urban_physics": 1},
        "consuming_read": "woof/core/urban_ucm.py",
        "warnings": ["wrf_480 removes the three rain-positive impervious-store "
                     "clamps so evaporation may exceed weak rainfall. wrf_471 "
                     "reproduces the older clamps. Both retain rho_air/1000. "
                     "This selects water balance only, not WRF 4.8 vegetation "
                     "or trees. A generation change refuses an SLUCM restart."]},
    "rdmaxalb": {
        "type": "boolean", "default": True,
        "warnings": [
            "rdmaxalb=false is implemented only by Noah LSM "
            "(sf_surface_physics=2), whose LSMINIT replaces supplied SNOALB "
            "with VEGPARM MAXALB by vegetation category."]},
    # The slab scheme's two WRF knobs (sf_surface_physics=1).
    "ifsnow": {
        "type": "integer", "default": 1,
        "component_id": "land_surface", "read_when": {"sf_surface_physics": 1},
        "warnings": [
            "Read only by the slab scheme (sf_surface_physics=1): 1 caps the "
            "skin and the top soil layer at 273.16 K under snow cover, 0 "
            "does not. For every nonzero value the default caps the top "
            "layer and skin together. GPUWM_WRF_EXACT=1 retains WRF's "
            "IFSNOW.EQ.1 top-layer cap, which leaves a measured 16 K "
            "skin/soil split at IFSNOW=2."]},
    "seaice_threshold": {
        "type": "number", "default": 100.0,
        "component_id": "land_surface",
        "warnings": [
            "Read at initialization: real.exe's adjust_for_seaice_pre "
            "(slab) or adjust_for_seaice_post (Noah, RUC, Noah-MP) converts "
            "open water below this skin temperature to ice. The default "
            "100 K converts nothing."]},
    "seaice_albedo_default": {
        "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.65,
        "warnings": [
            "A nondefault seaice_albedo_default is consumed by Noah, Noah-MP and RUC "
            "LSM (sf_surface_physics=2/3/4), in the live sea-ice ALBBCK "
            "override before LSMRUC."]},
    # experiment-scope knobs (whole tree, never per-domain)
    "p_top": {"type": "number", "minimum": 0.0},
    "blend_width": {"type": "integer", "minimum": 0, "default": 5},
    "co2_vmr": {"type": "number", "minimum": 0.0},
}

# Existing declarations that were looser than woof's real accepted set.
# User aerosol controls (aer_opt = 2) are owned here, so a rebuild from the
# release registry cannot lose the namelist rows carried by the optics port.
IMPLEMENTED.update({'aer_type': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1, 'enum': [1, 2, 3], 'per_domain': False, 'type': 'integer'}, 'aer_aod550_opt': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1, 'enum': [1, 2], 'per_domain': False, 'type': 'integer'}, 'aer_aod550_val': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 0.12, 'per_domain': False, 'type': 'number'}, 'aer_angexp_opt': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1, 'enum': [1, 2, 3], 'per_domain': False, 'type': 'integer'}, 'aer_angexp_val': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1.3, 'per_domain': False, 'type': 'number'}, 'aer_ssa_opt': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1, 'enum': [1, 2, 3], 'per_domain': False, 'type': 'integer'}, 'aer_ssa_val': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 0.85, 'per_domain': False, 'type': 'number'}, 'aer_asy_opt': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 1, 'enum': [1, 2, 3], 'per_domain': False, 'type': 'integer'}, 'aer_asy_val': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 0.9, 'per_domain': False, 'type': 'number'}, 'aer_auxinput15': {'consuming_read': 'woof/core/aer_opt2.py', 'default': '', 'per_domain': False, 'type': 'string'}, 'aer_auxinput15_interval': {'consuming_read': 'woof/core/aer_opt2.py', 'default': 0, 'per_domain': False, 'type': 'integer'}})

from woof.dfi_control import registry_parameters as _dfi_registry_parameters
IMPLEMENTED.update(_dfi_registry_parameters())


TIGHTEN: dict[str, dict] = {
    "ra_physics": {"type": "integer", "enum": [0, 4, 90], "default": 0},
    # -v2 is the assembly resolution of 2026-07-28 (woof/physics_compat.py
    # WRF_RRTMG_TO_RTE_RRTMGP): the tracked registry was rebound to -v2 in
    # that pass but this table still said -v1, so the builder no longer
    # reproduced the tracked bytes and regenerating would have silently
    # reverted the ratified token.  -v1 stays accepted by the CODE for
    # historical receipts (woof/config.py); the registry enum governs what
    # a NEW plan may set, which is the current token only.
    "wrf_rrtmg_compatibility": {
        "type": "string",
        "enum": [
            "none",
            "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
            "wrf-rrtmg-4-4-legacy-v1",
        ],
        "default": "none"},
    # km_opt is NOT here any more.  It used to be a parameter with
    # ``enum: [1, 4]``, which was wrong twice over: 2 (1.5-order
    # prognostic TKE) and 3 (3-D Smagorinsky) are executable -- the
    # dycore's fail-closed admission takes 1/2/3/4 (woof/config.py:1345,
    # decided at woof/core/dycore.py:2226), the kernels are transcribed
    # from WRF v4.6.1 with dry-CBL oracle receipts, and
    # configs/les_nest_250m_km3.toml has shipped km_opt=3 on a nest child
    # since 1.5.1 -- and the enum said neither could be named.  Widening
    # it in place would have put km_opt on both channels at once, which
    # this document's own authority note forbids ("Component
    # selector_keys are declared on their component and are deliberately
    # absent from parameters") and tests/test_physics_registry_
    # declarations.py::test_selector_keys_are_not_duplicated_as_parameters
    # enforces.  So the selector moved to where every other scheme
    # selector already lives: components.turbulence.selector_keys, whose
    # four option rows carry km_opt 1/2/3/4 with their own maturity,
    # reachability and evidence.  Per the les-completion spec's
    # registry-accuracy item (8.1.5) and AC-P6.4.
    "diff_6th_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0},
    "diff_6th_slopeopt": {"type": "integer", "enum": [0, 1], "default": 0},
    # AN ENUM, not minimum 1.  ``minimum: 1`` advertised 1, 2, 3, 5, 6,
    # 7, 8 as type-legal, and woof/config.py:376 accepts none of them -- the
    # declaration was looser than the code it declares.  The enum is the set
    # of soil-layer counts this registry's own land-surface options carry,
    # read from those options' schemes and assigned near the end of
    # ``build`` (the merge below deep-copies this spec over the parameters
    # dict, so the literal here is only a placeholder).
    #
    # It is NOT the single count each scheme defines.  Every count any
    # scheme defines has to stay type-legal for the COMPONENT layer to be
    # the thing that speaks: tests/test_physics_registry.py requires a
    # domain asking for num_soil_layers=9 under NOAH to raise
    # component-admitted-setting, and a parameter-value rejection would
    # pre-empt that -- the value never reaches ``settings``, so the
    # option's own admitted set never sees it.
    #
    # The warning was written when RUC was refused and said three things that
    # are no longer true: that only 4 is accepted, that no physics component
    # reads the knob, and that the nine-layer identity was unported.  RUC is
    # admitted at nine layers and runs a forecast, so the text now describes
    # the resolver that actually decides.
    #
    # AND, since development branch admitted six: this warning is the text a
    # user reads to decide what they may set, so a stale sentence here is a
    # capability the product denies having.  Two clauses were left behind by
    # that lane's rewrite and contradicted the sentence after them --
    # 'RUC resolves 9' (woof/config.py soil_layer_count returns the
    # requested 6 when RUC is asked for 6; measured) and 'Both 4 and 9 are
    # selectable ... and no other value is' (six is selectable, and the very
    # next sentence says how).  Both are corrected.
    #
    # A THIRD clause survived that pass and was caught in review: 'This ENUM
    # declares 4 and 9 ... the enum is a statement about EVIDENCE, not the
    # whole set of selectable values', sitting in the MIDDLE of the same
    # warning string whose tail had just been rewritten to say six is
    # selectable, and contradicting the enum this module now derives.  One
    # user-visible string is one statement: it is corrected here, and the
    # evidence claim it was carrying is made by the maturity paragraph
    # further down, which is where it can be read without denying what the
    # field admits.
    #
    # THE ENUM NOW CARRIES SIX, and the enum is not where evidence is
    # stated.  Holding it at [4, 9] made the plan-review door NARROWER than
    # the run door: a six-level RUC plan was refused here with "must be one
    # of [4, 9]" -- a refusal naming no breakage and no way out -- while
    # woof/config.py admitted it, the kernel sized itself for it and a
    # forecast completed on it.  That is the second-table drift this audit
    # exists for, pointed the other way.  The enum states which counts are
    # SELECTABLE and is derived below from the land-surface schemes' own
    # tables, so a scheme that gains a geometry gains the enum entry in the
    # same edit; the EVIDENCE claim (six is internal-consistency-only) stays
    # exactly where a user reads it, in this warning and in the runtime
    # receipt, which is what a warning is for.
    "num_soil_layers": {
        # Placeholder only: the enum is DERIVED from the land-surface
        # schemes' own tables and reassigned after the parameter merge
        # below.  Held equal to what that derivation currently produces so
        # a reader of this module is never told a different set than the
        # artifact carries.
        "type": "integer", "enum": [4, 6, 9], "default": 4,
        "warnings": [
            "The resolved layer count comes from the SCHEME, not from this "
            "knob: woof/config.py soil_layer_count consults "
            "LAND_SURFACE_SOIL_LAYERS for the selected sf_surface_physics, so "
            "Noah and Noah-MP resolve 4 and RUC resolves the 6 or 9 it was "
            "asked for, and every soil allocation, VRAM count, output dimension "
            "and restart shape reads that resolver. DIVERGENCE from WRF, "
            "deliberate: set_physics_rconfigs OVERWRITES a namelist request "
            "that disagrees with the scheme and only logs it at debug level, so "
            "a namelist asking Noah for nine layers runs on four with no error; "
            "woof refuses instead. This ENUM declares every count a "
            "registered land-surface scheme defines -- 4 with Noah or "
            "Noah-MP, 6 or 9 with RUC -- and is derived from those schemes' "
            "own tables; it states what is SELECTABLE, and the separate "
            "question of what an oracle has judged is answered by this "
            "warning and by the run receipt, never by the enum. WRF's "
            "six-level RUC grid "
            "(share/module_soil_pre.F:init_soil_depth_3) is one of those "
            "counts, and it COMPILES and RUNS: woof/core/kernels/ruc.cu "
            "sizes every "
            "soil scratch from RUC_NZS and selects its level table with it, "
            "gpuwm.core.ruc/.ruc_gpu resolve the count from the profile, and a "
            "1-hour HRRR-initialised forecast completes at six levels with "
            "soil_layers_stag=6 in its wrfout. What six LACKS is a WRF FORECAST "
            "ORACLE -- every lsmruc/sfctmp/soilmoist/snowtemp fixture in the "
            "tree is nine-level -- so it is a MATURITY statement, not an "
            "admission question: it is selected through the hash-bound "
            "experiment config (sf_surface_physics = 3 with num_soil_layers = "
            "6) rather than a named --physics-profile, warns at runtime, and "
            "carries soil_geometry_evidence=internal-consistency-only in its "
            "run receipt. See docs/wrf_ruc_runtime_admission.md."]},
    "terrain_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    # AUDIT R-060.  These three were roadmap rows that had outlived their
    # gap, and the gate that keeps such a row true was one-directional --
    # it proved an implemented row's citation and FORBADE an unimplemented
    # row from carrying one, so a lane that landed the read could not move
    # the row even if it wanted to.  tools/check_parameter_claims.py now
    # proves the negative as well.
    #
    # smooth_option: woof/core/nest_interp.py smoother() dispatches WRF's
    # sm121 and smdsm through the nest_smooth_i/nest_smooth_j kernels, and
    # woof/core/model.py wires ExperimentConfig.smooth_option into it.
    "smooth_option": {"type": "integer", "enum": [0, 1, 2], "default": 0,
                      "warnings": [
                          "0 is no smoothing, 1 is WRF's sm121 and 2 is "
                          "smdsm. Prepared hierarchy artifacts are the "
                          "same at every value: this smooths the parent "
                          "after feedback while the tree runs and does "
                          "nothing to a prepared child."]},
    # wif_input_opt / aer_init_opt: woof/ingest/wif_climatology.py IS the
    # WIF ingest, woof/ingest/real.py branches on the (aer_init_opt,
    # wif_input_opt) == (1, 1) pair, and woof/config.py's own error text
    # instructs the user to "use aer_init_opt=1 with wif_input_opt=1" --
    # a pair the registry made unspellable, since one row refused it and
    # the other did not exist.
    "wif_input_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "aer_init_opt": {"type": "integer", "enum": [0, 1, 2], "default": 0,
                     "compatible_previous_enums": [[0, 1]]},
    "use_rap_aero_icbc": {
        "type": "boolean", "default": False,
        "component_id": "microphysics",
        "consuming_read": "woof/ingest/real.py",
        "description": "Analyzed QNWFA/QNIFA initial and lateral values with operational monthly surface emissions; mp_physics=28 only.",
    },
    "epssm": {"type": "number", "minimum": 0.0, "maximum": 1.0,
              "default": 0.1},
    "diff_6th_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                        "default": 0.12},
    # The sixth-order filter's source form (woof.core.dycore.DIFF6_FORMS)
    # and the NOAA WRFV3.9 fork's second factor (fork
    # Registry.EM_COMMON:2629, &dynamics, max_domains, default 0.04).
    "diff_6th_form": {
        "type": "string", "enum": ["wrf_461", "noaa_wrf39"],
        "default": "wrf_461",
        "consuming_read": "woof/core/dycore.py",
        "warnings": [
            "Selects which source's sixth-order filter runs when "
            "diff_6th_opt > 0. wrf_461 (WRF v4.6.1) filters moisture and "
            "scalars with diff_6th_factor on the RK3 first-stage step dt/3 "
            "and stops three points short of a specified or nested edge. "
            "noaa_wrf39 (the NOAA-EMC WRFV3.9 fork, operational HRRR "
            "v4.1.21) filters the moist and scalar arrays with "
            "diff_6th_factor2 on the full step and runs the filter to the "
            "domain edge on zero-gradient halo copies; at HRRR's 0.12/0.04 "
            "that is nine times weaker on moisture. An imported namelist "
            "that names diff_6th_factor2 selects noaa_wrf39."]},
    "upper_wind_limiter_form": {
        "type": "string", "enum": ["wrf_461", "noaa_wrf39"],
        "default": "wrf_461", "consuming_read": "woof/core/acoustic.py",
        "description": "WRF v4.6.1 has no upper-wind limiter. The NOAA WRFV3.9 form damps saved stage winds above 110 m/s inside zdamp on each acoustic substep when damp_opt=3. A namelist naming diff_6th_factor2 selects the fork form."},
    "diff_6th_factor2": {
        "type": "number", "minimum": 0.0, "maximum": 1.0,
        "default": None,
        "warnings": [
            "Read only under diff_6th_form = noaa_wrf39 (null takes the "
            "fork's 0.04); refused under wrf_461, which has no second "
            "factor."]},
    # WRF mp_zero_out (Registry.EM_COMMON:2553, both sources) and v4.6.1's
    # mp_zero_out_all (:2554), consumed by woof.core.microphysics.
    "mp_zero_out": {
        "type": "integer", "enum": [0, 1, 2], "default": 0,
        "warnings": [
            "After microphysics: 1 sets non-vapour species below "
            "mp_zero_out_thresh to zero, 2 also floors vapour at zero; the "
            "outermost ring is floored at zero in both (WRF "
            "module_microphysics_zero_out)."]},
    "mp_zero_out_thresh": {"type": "number", "default": 1.0e-8},
    "mp_zero_out_all": {
        "type": "integer", "enum": [0, 1], "default": 0,
        "warnings": [
            "1 also applies mp_zero_out to the scalar number arrays, which "
            "the NOAA WRFV3.9 fork always does; WRF v4.6.1's default 0 "
            "applies it to the moist array only."]},
    "moist": {"type": "boolean", "default": False},
    "moist_cq": {"type": "boolean", "default": True},
    # WRF's own &dynamics switch (Registry.EM_COMMON:2889, max_domains,
    # default .false.), consumed by woof.core.dycore.diff6_exempt_slots.
    # Declared here because it is divergence-ledger entry L4: an ArWen
    # DEFAULT candidate whose promotion is decided by the observation
    # battery, not by argument.  The warning is what a plan author needs to
    # know before setting it by hand instead of through the axis.
    "moist_mix6_off": {
        "type": "boolean", "default": False,
        "warnings": [
            "moist_mix6_off = true removes the 6th-order horizontal filter "
            "from the WRF moist array only (dyn_em/module_em.F:1421 under "
            "config_flags%moist_mix6_off); theta keeps its filter and the "
            "number/volume tracers keep theirs, which is WRF's own per-array "
            "scoping, not a woof simplification.",
            "This is divergence-ledger entry L4 and it is UNDECIDED: it is "
            "a candidate WOOF default with no obs-skill receipt yet. "
            "Selecting it through the [experiment] physics_mode axis "
            "(woof/physics_mode.py) records the arm in the run receipt; "
            "setting it by hand here does not, and a hand-set value beside "
            "physics_mode is refused rather than merged."]},
    "grav_settling": {
        "type": "integer", "enum": [0, 1, 2], "default": 0,
        "component_id": "pbl", "consuming_read": "woof/core/fog_settling.py",
        "maturity": "supported",
        "verification_scope": "retained-native-call-replay-and-real-forecast",
        "verification_scope_note": "Strict WRF 4.7.1 real-input replay: 0/1170000 words for each setting, no exclusions. Kept captures, command and limits: woof/data/fog_kfeta/verification.json. Real forecasts and viewed Rust maps do not establish observation skill.",
        "description": "Shared fog settling after every native PBL. Explicit opt-ins: 1 Duynkerke costs 1.2717x, 2 FogDES costs 1.3431x YSU without settling in the retained interleaved real forecast (complete YSU plus added surface pass). Default 0. Native added surface/PBL work is profiled in woof/data/fog_kfeta/native-profile.json; GPU timing is in stage-ratios.json beside it. Native WRF 4.7.1 disables it under mp=28/29. Its surface pass requires MMINLU USGS or MODIFIED_IGBP_MODIS_NOAH; with sf_sfclay_physics=0 only the land-use-independent PBL pass runs.",
        "refused_when": [{"settings": {"mp_physics": [28, 29]}, "reason": "WRF disables separate fog settling because these microphysics already settle cloud water (share/module_check_a_mundo.F:2474-2487)"}]},
    "kfeta_trigger": {
        "type": "integer", "enum": [1, 2, 3], "default": 1,
        "component_id": "cumulus", "read_when": {"cu_physics": 1},
        "inactive_value_policy": "require-default",
        "consuming_read": "woof/core/kf.py",
        "maturity": "supported",
        "verification_scope": "retained-native-call-replay-and-real-forecast",
        "verification_scope_note": "Strict WRF 4.7.1 real-input replay: trigger2 0/8955000 words, trigger3 0/6740100, no exclusions. Kept captures, command and limits: woof/data/fog_kfeta/verification.json. Real forecasts and viewed Rust maps do not establish observation skill.",
        "description": "KF trigger: 1 vertical velocity (default), 2 moisture-advection perturbations, 3 relative humidity. Triggers 2/3 require cu_physics=1 (WRF 4.7.1 README.namelist:868). Retained interleaved real-forecast complete CU1 costs: trigger2 0.8193x, trigger3 1.0417x trigger1; woof/data/fog_kfeta/stage-ratios.json."},
    "top_lid": {"type": "boolean", "default": True},
    "morr_rimed_ice": {"type": "integer", "enum": [0, 1], "default": 1},
    "aercu_opt": {"type":"integer", "minimum":0, "default":0,
                  "component_id":"microphysics", "per_domain":False,
                  "consuming_read":"woof/core/morrison_aero.py",
                  "description":"0 uses fixed droplets. 1/2 require cu_physics=11 and mp_physics=40. NASA MERRA-2 GMI 2001-2010 monthly forcing is prepared by default: written from papers, not bitwise to WRF."},
    "wsm6_hail_opt": {"type": "integer", "enum": [0, 1], "default": 0},
    "gsfcgce_hail": {"type": "integer", "enum": [0, 1], "default": 0},
    "gsfcgce_2ice": {"type": "integer", "enum": [0, 1, 2, 3], "default": 0},
    # mp=28's aerosol source (woof.config.MP28_AEROSOL_SOURCES).  It is
    # REGISTERED, not just a RunConfig field, because the refusal it
    # answers is now a registry refusal too: audit R-044's
    # consumers.lateral_forcing_dataset row names 'synthetic' as the
    # deliberate way out, and a way out a plan cannot express is not a way
    # out.  'auto' resolves WRF's monthly WIF climatology and announces the
    # synthetic fallback by name; 'climatology' refuses rather than
    # degrading; 'synthetic' selects thompson_init's profile on purpose.
    "mp28_aerosol_source": {
        "type": "string",
        "enum": list(MP28_AEROSOL_SOURCES),
        "compatible_previous_enums": [["auto", "climatology", "synthetic"]],
        "default": "auto",
        "consuming_read": "woof/ingest/real.py",
    },
    # WRF v4.6.1's NSSL variant selectors.  There is one NSSL scheme
    # (mp_physics=18) and these four flags on top of it; the deprecated
    # IDs 17/19/21/22 are spellings share/module_check_a_mundo.F
    # rewrites onto exactly this set.  woof/core/nssl2_contract.py
    # resolves them and refuses the combinations with no ported path.
    'nssl_2moment_on': {'type': 'integer', 'compatible_previous_enums': [[-1, 1]], 'enum': [-1, 0, 1], 'default': -1, 'note': 'WRF -1 resolves to two moments; 0 uses the native one-moment NSSL path (legacy 19/21). Legacy aliases are canonicalized before numerical dispatch.', 'evidence': 'Native WRF v4.6.1 public-domain source. One-moment option 19/21 production CUDA strict replay: MP19 matches 2994 active words and MP21 2406; 882 unwritten radius words are byte-preserved (native has_req*=0); each mixed-phase fixture matches 10614 words. Whole-forecast and lifecycle qualification is recorded separately in the lane evidence. Fresh strict WRF4.7.1 MP21 replay differs in 123 of 59348 words (theta/qv/qc), despite stock-input admission and a complete 600-step forecast. The one-moment numerical path remains pinned to WRF4.6.1 and is not WRF4.7.1-qualified; see docs/microphysics-retiming.md.'},
    'nssl_ccn_on': {'type': 'integer', 'enum': [-1, 0, 1], 'default': -1, 'note': "WRF's nssl_ccn_on (Registry.EM_COMMON:2421). -1 resolves to 1 (share/module_check_a_mundo.F:3433-3435). 0 is the deprecated mp_physics=17/22 semantics: qnn is not allocated, the unactivated CCN is diagnosed from the base concentration every step (module_mp_nssl_2mom.F:2734), never stored back (:3283), and renucfrac rises to 1.0 (:2555-2557), which changes the nucleation pool at :10116 and arms the low-temperature limiter at :10120-10127.", 'evidence': 'Native WRF v4.6.1 public-domain source. One-moment option 19/21 production CUDA strict replay: MP19 matches 2994 active words and MP21 2406; 882 unwritten radius words are byte-preserved (native has_req*=0); each mixed-phase fixture matches 10614 words. Whole-forecast and lifecycle qualification is recorded separately in the lane evidence. Fresh strict WRF4.7.1 MP21 replay differs in 123 of 59348 words (theta/qv/qc), despite stock-input admission and a complete 600-step forecast. The one-moment numerical path remains pinned to WRF4.6.1 and is not WRF4.7.1-qualified; see docs/microphysics-retiming.md.'},
    'nssl_density_on': {'type': 'integer', 'compatible_previous_enums': [[-1, 1, 2]], 'enum': [-1, 0, 1, 2], 'default': -1, 'note': 'Native density packages: 0 fixed density is admitted for one-moment/no-hail; 1 predicts graupel volume in one-moment hail. Two-moment hail requires 2 so qvolh is allocated.', 'evidence': 'Native WRF v4.6.1 public-domain source. One-moment option 19/21 production CUDA strict replay: MP19 matches 2994 active words and MP21 2406; 882 unwritten radius words are byte-preserved (native has_req*=0); each mixed-phase fixture matches 10614 words. Whole-forecast and lifecycle qualification is recorded separately in the lane evidence. Fresh strict WRF4.7.1 MP21 replay differs in 123 of 59348 words (theta/qv/qc), despite stock-input admission and a complete 600-step forecast. The one-moment numerical path remains pinned to WRF4.6.1 and is not WRF4.7.1-qualified; see docs/microphysics-retiming.md.'},
    'nssl_hail_on': {'type': 'integer', 'compatible_previous_enums': [[-1, 0, 1]], 'enum': [-1, 0, 1, 2], 'default': -1, 'note': 'Native hail packages. 2 is one-moment hail and requires nssl_2moment_on=0. Two-moment hail-number reads through an unallocated slot remain refused.', 'evidence': 'Native WRF v4.6.1 public-domain source. One-moment option 19/21 production CUDA strict replay: MP19 matches 2994 active words and MP21 2406; 882 unwritten radius words are byte-preserved (native has_req*=0); each mixed-phase fixture matches 10614 words. Whole-forecast and lifecycle qualification is recorded separately in the lane evidence. Fresh strict WRF4.7.1 MP21 replay differs in 123 of 59348 words (theta/qv/qc), despite stock-input admission and a complete 600-step forecast. The one-moment numerical path remains pinned to WRF4.6.1 and is not WRF4.7.1-qualified; see docs/microphysics-retiming.md.'},
    'icloud': {'type': 'integer', 'enum': [0, 1, 2, 3], 'default': 1, 'implementation_rows': {'cldfra3': {'value': 3, 'state': ['cldfra'], 'outputs': ['CLDFRA'], 'authority': 'WRF v4.7.1 module_radiation_driver.F:4003-4473', 'maturity': 'wrf-matched-run', 'native_replay': 'tools/cloud_aerosol_wrf471_oracle: 19968 output words, zero differences', 'integration_proof': 'stock-namelist woof go; both nest feedback modes; byte-exact checkpoint continuation; two distinct ensemble members; disabled forecast byte identity', 'limits': ['QC and QI required']}}},
    "radt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "radt_minutes": {"type": "number", "minimum": 0.0, "default": 12.0},
    "bldt": {"type": "number", "minimum": 0.0, "default": 0.0},
    "kf_edrates": {"type": "integer", "enum": [0, 1], "default": 0,
                   "read_when_any": [{"cu_physics": 11}, {"cu_physics": 99}], "requires_reader": True,
                   "description": "The selected cumulus kernel cannot publish KF rate profiles; set kf_edrates=0 or select its KF reader."},
    "cudt_minutes": {"type": "number", "minimum": 0.0, "default": 5.0},
    "bmj_rad_feedback": {"type": "boolean", "default": False,
                         "per_domain": True, "component_id": "cumulus"},
    # Grell-family keys, WRF v4.6.1 Registry defaults
    # (Registry.EM_COMMON:2544,2546); read only where cu_physics = 3.
    # 0 is the ensemble mean, 1..16 one closure member alone; anything
    # else has no meaning in cup_forcing_ens_3d (woof.config
    # gf_clos_choice_refusal names the breakage).
    "clos_choice": {
        "type": "integer",
        "enum": list(range(GF_CLOSURE_MEMBERS + 1)), "default": 0,
        "warnings": [
            "0 (the Registry default) is the 16-member ensemble mean, "
            "compared bitwise against WRF v4.6.1. 1..16 run one closure "
            "member of cup_forcing_ens_3d alone: WRF's own code path, "
            "implemented but not verified against a WRF run."]},
    "ishallow": {"type": "integer", "enum": [0, 1], "default": 0},
    # Grell 3D's subsidence-spreading width (Registry.EM_COMMON:2543,
    # default 1); read only where cu_physics = 5.  1 is no spreading; > 1
    # selects G3's high-resolution arm.  WRF defines every integer.
    "cugd_avedx": {"type": "integer", "default": 1, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "maxiens": {"type": "integer", "default": 1, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "maxens": {"type": "integer", "default": 3, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "maxens2": {"type": "integer", "default": 3, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "maxens3": {"type": "integer", "default": 16, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "ensdim": {"type": "integer", "default": 144, "component_id": "cumulus", "read_when": {"cu_physics": 5}, "inactive_value_policy": "require-default"},
    "cu_diag": {"type": "integer", "default": 0, "enum": [0, 1], "component_id": "cumulus", "read_when_any": [{"cu_physics": 5}, {"cu_physics": 93}], "inactive_value_policy": "require-default"},
    "convtrans_avglen_m": {"type": "number", "default": 30.0, "component_id": "cumulus", "read_when_any": [{"cu_physics": 5}, {"cu_physics": 93}, {"cu_physics": 3, "cu_rad_feedback": True}], "inactive_value_policy": "require-default"},
}

WRF_TYPE = {"integer": "integer", "real": "number",
            "logical": "boolean", "character": "string"}

# Lane K's final ledger for declarations that remain unavailable.  The class
# is kept beside the blocker so the generated registry and the handoff cannot
# drift into calling a bounded option branch a new subsystem.
UNIMPLEMENTED_LEDGER: dict[str, tuple[str, str]] = {
    "brcr_ub": (
        "c",
        "Not a WRF v4.6.1 namelist option. BRCR_UB is a YSU scheme-internal "
        "constant compiled into the existing kernel, so there is no legal "
        "WRF user setting to expose."),
    "cu_used": (
        "c",
        "Not a WRF v4.6.1 namelist option. CU_USED is derived by WRF from "
        "the selected cumulus scheme and domain state, not legally set by a "
        "user."),
    "dust_emis": (
        "c",
        "Dust emission outside WRF-Chem feeds the ice-friendly aerosol "
        "surface source. woof allocates nifa2d and leaves it exactly zero, "
        "matching thompson_init, and carries no dust inventory, emission "
        "operator, or surface-flux coupling for it."),

    "icloud_cu": (
        "c",
        "Derived internal selector, implemented by woof.core.cloud_fraction.icloud_cu "
        "and the CUDA cu_radiation merge for the ported CU schemes. Stock "
        "namelists select cu_rad_feedback; ICLOUD_CU is not a user knob. "
        "CU11 retains its named parent-scheme requirement."),

    "naer": (
        "c",
        "NAER seeds WRF's naer-based droplet mode inside classic Thompson "
        "(thompson-mp8), which is a fixed-droplet port. The prognostic "
        "aerosol path lives in components.microphysics.options."
        "thompson-aerosol-mp28, whose nc/nwfa/nifa come from CCN activation "
        "rather than from this scalar, so honouring it here would be a "
        "different scheme."),
    "num_wif_levels": (
        "c",
        "This sizes the WIF (water/ice-friendly aerosol) metgrid input "
        "stream. The WRF namelist importer accepts the monthly WIF dataset's "
        "30 levels with use_aero_icbc=true and wif_input_opt=1; the native "
        "configuration does not expose an independent level-count override."),
    "nssl_3moment": (
        "c",
        "The NSSL three-moment extension (nssl_3moment=1, which WRF rewrites "
        "to 2 with hail on at share/module_check_a_mundo.F:3457-3463) adds the "
        "qzr/qzg/qzh reflectivity moments and the ipconc>=6 branch "
        "throughout the scheme. The ported family is the two-moment ipconc=5 "
        "branch only; woof/core/nssl2_contract.py refuses the combination "
        "at validation."),
    "nssl_alphar": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_ALPHAR is a scheme-internal "
        "rain-shape constant and has no legal user setting."),
    "nssl_ehlw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHLW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
    "nssl_ehw0": (
        "c",
        "Not a WRF v4.6.1 namelist option. NSSL_EHW0 is a scheme-internal "
        "collection-efficiency constant and has no legal user setting."),
    "num_land_cat": (
        "c",
        "WRF derives this count from the selected land-use dataset/table. "
        "Supporting another value requires alternate static categories, "
        "parameter tables, field validation, and state dimensions; it is not "
        "an independent runtime tune in woof."),
    "num_soil_cat": (
        "c",
        "WRF derives this count from the selected soil-category dataset/table. "
        "Arbitrary values require alternate static categories, parameter "
        "tables, validation, and allocations rather than a scalar branch."),
    "progn": (
        "c",
        "PROGN switches classic Thompson (thompson-mp8) onto WRF's "
        "chemistry-driven droplet source, which needs a QNDROPSOURCE carrier "
        "woof has no writer for. Prognostic droplet number itself IS ported, "
        "as components.microphysics.options.thompson-aerosol-mp28 "
        "(mp_physics=28), where it is driven by CCN activation instead."),
    "qna_update": (
        "c",
        "Updating aerosol number from a wrfqnainp auxiliary stream needs an "
        "auxiliary input subsystem, an update cadence, restart position and "
        "field ownership; woof writes one fixed wrfout frame per domain and "
        "reads no auxiliary input streams at all."),
    "scm_force_flux": (
        "c",
        "This belongs to WRF's single-column-model forcing subsystem; woof "
        "has no SCM runner, forcing time series, column boundary contract, or "
        "restart semantics."),
    "shalwater_depth": (
        "c",
        "The shallow-water surface branch needs a bathymetry/depth static "
        "field and its surface coupling; woof ingests neither and has no "
        "runtime branch to consume the scalar."),
    "shalwater_z0": (
        "c",
        "The shallow-water roughness branch needs the missing shallow-water/"
        "bathymetry classification and surface-driver coupling; setting a "
        "scalar alone would be inert."),
    "surface_input_source": (
        "c",
        "Alternate surface-input sources require source-specific static/"
        "met-field selection and provenance semantics. woof's ingest routes "
        "own those choices explicitly and implement no interchangeable "
        "runtime source selector."),
    "tice2tsk_if2cold": (
        "b",
        "This is a bounded branch in the existing fractional-sea-ice surface "
        "wrapper, but woof currently transcribes only the false arithmetic "
        "(the operational HRRR v4.1.21 value). The true get_local_ice_tsk "
        "branch, TSK_ICE = MIN(TSK, 273.15), is not ported and has no WRF "
        "oracle fixture."),
    "use_aero_icbc": (
        "c",
        "For components.microphysics.options.thompson-aerosol-mp28, this "
        "namelist-only key imports through the aer_init_opt=1 / "
        "wif_input_opt=1 pair rather than a RunConfig field of its own. "
        "Monthly WIF and analyzed QNWFA/QNIFA initial and lateral carriers "
        "are implemented. An unrestricted GOCART species reader is not."),
    "wif_fire_emit": (
        "c",
        "Biomass-burning aerosol emissions for Thompson-MP-Aero need a fire "
        "inventory, an emission cadence and the derived aer_fire_emit_opt "
        "state WRF computes from this flag; woof carries none of them."),
    "wif_fire_inj": (
        "c",
        "This selects the vertical injection profile for the biomass-burning "
        "aerosol emissions above. It is a branch inside a subsystem woof "
        "does not have, so there is nothing for it to distribute."),
}

NOISE = re.compile(
    r"^(MISSED BY THE CANDIDATE LIST[^.]*\.|MISFILED IN THE CANDIDATE LIST[^.]*\.|"
    r"Adjudication:\s*|DERIVED, not namelist[^.]*\.)\s*", re.I)
CASE_TOKEN = re.compile(r"real74|hrrr|ohio|oklahoma|may1999|20cr|1974", re.I)

# Rows this pass claims. Every other declarer's rows are left exactly as
# written: with no citation they assert nothing, and their owner turns one on
# by citing the read that makes it true.
OWNED = list(IMPLEMENTED) + list(TIGHTEN) + [
    "spec_exp", "nest_microphysics_transition"]


def clean(reason: str) -> str:
    text = " ".join((reason or "").split())
    text = NOISE.sub("", text)
    text = CASE_TOKEN.sub("the reference configuration", text)
    if len(text) > 220:
        cut = text[:220]
        stop = max(cut.rfind(". "), cut.rfind("; "))
        text = (cut[: stop + 1] if stop > 80 else cut.rstrip() + "...")
    return text.strip()


# ------------------------------------------------------ cited consuming reads
# A knob counts as implemented because a citation proves GPUWM reads it, never
# because someone asserted a flag.  Resolving the citation against the same
# rule the gate applies means the claim and its evidence cannot drift apart,
# and it leaves every other declarer's rows untouched: a row with no citation
# makes no claim, so an owner flips their own knob on by citing the read that
# makes it true.
SEARCH_ROOT = "woof"
#: The registry loader names every knob as a dict key, so it matches all of
#: them and proves nothing.  woof/config.py is deliberately NOT skipped: it
#: declares and validates the knobs, which is the only read some of them have.
SKIP = {"woof/physics_registry.py"}
CONFIG = "woof/config.py"

_files: list[str] | None = None
_sources: dict[str, tuple[str, frozenset[str], frozenset[str]] | None] = {}


def _searchable_files() -> list[str]:
    """Repo-relative Python files a citation may name, in a stable order.

    Sorted on the POSIX relative path rather than on ``Path`` objects, whose
    ordering depends on the platform separator and on case folding, so the
    citation this picks does not depend on which machine ran the builder.
    """
    global _files
    if _files is None:
        found = []
        for path in (MODEL / SEARCH_ROOT).rglob("*.py"):
            rel = path.relative_to(MODEL).as_posix()
            if rel in SKIP or "__pycache__" in rel:
                continue
            # A generic knob must cite a generic reader.  Citing a
            # source-specific adapter would say the knob only exists for one
            # data source, which is the specialization the case-token gate
            # exists to prevent.
            if CASE_TOKEN.search(path.name):
                continue
            found.append(rel)
        _files = sorted(found)
    return _files


def _identifiers(tree: ast.AST) -> frozenset[str]:
    """Names the module binds or reads as identifiers, not as text.

    ``cfg.isftcflx``, ``num_soil_layers: int`` and ``radt=radt`` are reads the
    interpreter performs.  ``{"num_soil_layers": 9}`` is a string that happens
    to spell a knob; it can be a read (``getattr(cfg, "moist_cq", True)``) but
    it can equally be an unrelated namelist table, so it ranks lower.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.add(node.arg)
    return frozenset(names)


def _source(rel: str) -> tuple[str, frozenset[str], frozenset[str]] | None:
    """Prose-stripped code and identifier set, or None if it proves nothing."""
    if rel not in _sources:
        path = MODEL / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _sources[rel] = None
        else:
            try:
                from tools.check_parameter_claims import (
                    UnparseableCitation, _code_without_prose)
                code = _code_without_prose(text, path.suffix)
                tree = ast.parse(text)
            except (UnparseableCitation, SyntaxError, ValueError):
                # The gate fails an unparseable citation, so the builder must
                # never propose one.
                _sources[rel] = None
            else:
                _sources[rel] = (code, _identifiers(tree),
                                 frozenset(re.findall(r"\b\w+\b", code)))
    return _sources[rel]



def available_cpus() -> int:
    """Count allowed CPUs before sizing citation workers, including Python 3.12."""
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        return process_count() or 1
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return len(affinity(0)) or 1
        except OSError:
            pass
    return os.cpu_count() or 1


def preload_sources(jobs: int) -> None:
    """Parse independent citation sources on the available CPUs.

    Workers return facts only; the parent preserves the sorted path order
    and the existing citation ranking. One worker is the serial reference.
    """
    files = [rel for rel in _searchable_files() if rel not in _sources]
    if not files:
        return
    if jobs == 1:
        for rel in files:
            _source(rel)
        return
    with ProcessPoolExecutor(max_workers=min(jobs, len(files))) as pool:
        _sources.update(zip(files, pool.map(_source, files, chunksize=8)))


def reads_knob(rel: str, name: str) -> bool:
    """Whether ``rel`` satisfies the shipped gate as a citation of ``name``.

    This is tools/check_parameter_claims' own test: the file must parse, and
    the knob must survive prose stripping.  A docstring mention does not
    count.
    """
    if not (MODEL / rel).is_file():
        return False
    source = _source(rel)
    if source is None:
        return False
    return name in source[2]


def _rank(rel: str, name: str) -> int:
    """Lower is a better citation for ``name``.

    A runtime component that executes on the knob is the strongest evidence,
    so ``woof/core`` identifier reads come first and other packages next.
    ``woof/config.py`` is where the knob is declared and validated, so it is
    cited only when nothing consumes the value -- which is the whole story for
    ``num_soil_layers``.  A name that appears only inside string literals
    ranks below every identifier read, because a namelist key spells a knob
    without reading it.
    """
    source = _source(rel)
    if source is None:  # reads_knob() already excluded these
        return 5
    if name in source[1]:
        if rel.startswith(SEARCH_ROOT + "/core/"):
            return 0
        return 2 if rel == CONFIG else 1
    return 3 if rel.startswith(SEARCH_ROOT + "/core/") else 4


def consuming_read_candidates(name: str) -> list[str]:
    """Every file that satisfies the gate for ``name``, best citation first."""
    return sorted(
        (rel for rel in _searchable_files() if reads_knob(rel, name)),
        key=lambda rel: (_rank(rel, name), rel))


def find_consuming_read(name: str, current: str | None = None) -> str | None:
    """Return a repo-relative file whose code reads ``name``.

    ``current`` -- the citation already in the registry -- wins whenever it is
    still a file this builder would accept and still reads the knob.  Which
    file *consumes* a knob is a judgement someone made by reading the code,
    and a search cannot re-derive it: several files read the same knob and only
    one of them is the component that acts on it.  So the search proposes
    rather than overrules, and a citation is replaced only once it stops being
    true, which is exactly when the claim it backs has outlived its evidence.
    """
    candidates = consuming_read_candidates(name)
    if current and current in candidates:
        return current
    return candidates[0] if candidates else None


# --------------------------------------------------- verified per-domain data
# One-way four-domain chains transcribed from the verified nested
# configurations; the values are NOT derived from a grid-spacing scaling law.
#
# Each template gets its own list, because the two chains are not the same
# chain.  Both descend 12 km -> 3 km -> 1 km on parent grid ratios 1/4/3, and
# then they diverge: the reference chain closes on parent_grid_ratio 3
# (1000/3 m, published as the nominal 333 m) and the wrf-matched-run-candidate
# chain closes on parent_grid_ratio 2 (500 m).  Those ratios were read from
# the two four-domain experiment configurations, not inferred.
#
# One shared list used to serve both, so index 3 published 333 m for a
# template whose fourth domain is 500 m.  nominal_dx_m is provenance for
# display and never resolves into a setting, so the wrong value mislabelled a
# run rather than mis-integrating it -- which is exactly why no numerical gate
# could catch it, and why the two lists are kept apart rather than shared with
# one entry overridden.
NEST_COLUMNS: dict[str, list[dict]] = {
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        # parent_grid_ratio 3 below the 1 km nest.
        {"nominal_dx_m": 333.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        # parent_grid_ratio 2 below the 1 km nest, not 3.
        {"nominal_dx_m": 500.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
    # The default full-physics suite (`woof domain` emits it): columns
    # transcribed from the Thompson matched-run campaign geometry
    # (configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml), whose d04
    # refines the 1 km nest by 2 exactly as the NSSL-2 ladder does.
    "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1": [
        {"nominal_dx_m": 12000.0, "diff_6th_factor": 0.12, "radt": 12.0,
         "epssm": 0.5},
        {"nominal_dx_m": 3000.0, "diff_6th_factor": 0.10, "radt": 3.0,
         "epssm": 0.1},
        {"nominal_dx_m": 1000.0, "diff_6th_factor": 0.08, "radt": 1.0,
         "epssm": 0.1},
        {"nominal_dx_m": 500.0, "diff_6th_factor": 0.06, "radt": 1.0,
         "epssm": 0.1},
    ],
}


def _surface_coupling_warnings(registry: dict) -> None:
    """Replace the v1.1 surface-seam restrictions retired in v1.2."""
    land = registry["components"]["land_surface"]["options"]
    noahmp = land["noah-mp"]["warnings"]
    noahmp[4] = (
        "WRF's SIX-RATE precipitation seam is active. The surface driver "
        "carries convective, nonconvective, shallow-convective, snow, "
        "graupel and hail accumulations from their real writers, and "
        "noahmplsm takes module_sf_noahmpdrv.F:776-789's PRESENT(MP_*) "
        "branch including PRCPOTHR.")
    noahmp[5] = (
        "COSZEN is a radiation-driver carrier. Noah-MP consumes the last "
        "radiation call's value unchanged between calls, including radconst's "
        "half-radiation-interval hour-angle offset. A radiation-free run "
        "still binds start time and latitude/longitude and seeds the same "
        "offset value once.")
    noahmp[8] = (
        "When the MYNN 5/5 pairing is selected, WRF v4.6.1 runs the MYNN "
        "surface layer first, then NOAHMP_SFLX overwrites "
        "TSK/HFX/QFX/LH on its active land columns while leaving "
        "UST/CHS/CHS2/CQS2/FLHC/FLQC with MYNN. The surface-driver post-pass "
        "unconditionally replaces MYNN's T2/Q2/TH2 with Noah-MP's "
        "water/urban/vegetated diagnostics before MYNN PBL consumes them. "
        "The source transcription and writer-order tests are in "
        "tools/mynn_surface_pairing_wrf461_oracle and "
        "tests/test_mynn_surface_pairing_ownership.py.")
    noahmp_option = land["noah-mp"]
    noahmp_option["constraints"]["requires_components"]["surface_layer"] = [
        "revised-mm5", "classic-mm5", "mynn"]
    noahmp_option["extensions"]["mynn_surface_ownership"] = {
        "surface_layer_writes_first": [
            "ust", "hfx", "qfx", "chs", "chs2", "cqs2", "flhc", "flqc",
            "t2", "q2", "th2"],
        "lsm_overwrites": ["tsk", "hfx", "qfx", "lh"],
        "lsm_preserves": [
            "ust", "chs", "chs2", "cqs2", "flhc", "flqc"],
        "post_lsm_overwrites": ["t2", "q2", "th2"],
        "wrf_source": (
            "phys/module_surface_driver.F:3127-3181,3324-3370; "
            "phys/module_sf_noahmpdrv.F:1206-1207,1223-1285"),
    }

    ruc = land["ruc-lsm"]["warnings"]
    ruc[3] = (
        "WRF-ARW's EM_CORE==1 surface couplings are active: LSMRUC consumes "
        "RAINNCV/SNOWNCV/GRAUPELNCV, LAKEMASK bypasses the column core, "
        "fractional sea ice is deblended before and reblended after the call, "
        "and GSW is carried from radiation cadence. The historical EM_CORE=0 "
        "oracle replay remains explicit-only; independent source "
        "transcription probes cover the ARW seam.")
    ruc[9] = (
        "Mosaic land-use and soil accept 0/1, default 0, and read LANDUSEF "
        "and SOILCTOP. spp_lsm accepts 0/1 and needs a member-owned SPP "
        "pattern; flag_sm_adj remains at 0. "
        "The sea-ice threshold follows fractional_seaice: 0.5 at 0 (the "
        "default), 0.02 at 1, read by the seam, the fused driver, land-use "
        "initialisation and the CLM lake. rdlai2d and usemonalb are read "
        "by RUC: rdlai2d keeps the monthly LAI12M field and usemonalb the "
        "monthly ALBEDO12M background albedo. Pinned rather than "
        "configurable: isncovr_opt=2, c1sn=0.026, c2sn=21.0 and myj=False. "
        "seaice_albedo_default is configurable over [0,1] and defaults to "
        "the former literal 0.65. The fractional sea-ice deblend and "
        "reblend run under both thresholds.")
    ruc[8] = (
        "Admitted at num_soil_layers=9 with revised MM5, classic MM5 or "
        "MYNN surface. Under the MYNN 5/5 pairing, WRF v4.6.1 runs the surface "
        "layer first; LSMRUC then overwrites TSK/HFX/QFX/LH, preserves "
        "UST/FLHC/FLQC/CHS2/CQS2, and the driver recomputes CHS from FLHC. "
        "SFCDIAGS_RUCLSM unconditionally replaces MYNN's T2/Q2/TH2 before "
        "MYNN PBL consumes the post-LSM fields. This is HRRR's operational "
        "MYNN/MYNN/RUC pairing class. The source transcription and "
        "writer-order tests are in tools/mynn_surface_pairing_wrf461_oracle "
        "and tests/test_mynn_surface_pairing_ownership.py.")
    ruc_option = land["ruc-lsm"]
    for key in ("mosaic_lu", "mosaic_soil"):
        ruc_option["constraints"]["required_settings"].pop(key, None)
        ruc_option["constraints"].setdefault("admitted_setting_values", {})[key] = [0, 1]
        ruc_option["constraints"].setdefault("admitted_setting_values_reasons", {})[key] = (
            "WRF RUC SOILVEGIN selects dominant parameters at 0 and weighted "
            "category parameters at 1; other values select neither branch")
    ruc_option["constraints"]["required_settings"].pop("ruc_soilprop", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_soilprop"] = ["wrf_45", "wrf_461"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_soilprop"] = (
        "wrf_45 is the WRF v4.0-4.5 soil-water diffusivity over the moisture "
        "above the residual and wrf_461 the WRF v4.6.1 form over total "
        "porosity; the two move different water between soil levels, so no "
        "other name can select either")
    ruc_option["constraints"]["required_settings"].pop("ruc_irrigation", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_irrigation"] = ["wrf_45", "wrf_461"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_irrigation"] = (
        "wrf_45 is the WRF v4.0-4.5 crop-fraction-scaled soil moisture floor "
        "and wrf_461 the WRF v4.6.1 per-step relaxation to 1.1 x wilting "
        "point; the two add different soil water, so no other name can "
        "select either")
    ruc_option["constraints"]["required_settings"].pop("ruc_qvg_cold_start", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_qvg_cold_start"] = ["air", "wrf"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_qvg_cold_start"] = (
        "air starts the ground vapour from the lowest-level air and wrf from "
        "saturation at the skin times moisture availability; the two start "
        "different surface humidity, so no other name can select either")
    ruc_option["constraints"]["required_settings"].pop("ruc_snow", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_snow"] = ["wrf_45", "wrf_461"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_snow"] = (
        "wrf_45 is the WRF v4.0-4.5 snow scheme the operational RAP/HRRR "
        "branch carries and wrf_461 the WRF v4.6.1 rewrite; the two differ "
        "in snow conductivity, cover, melt and albedo, so no other name can "
        "select either")
    ruc_option["constraints"]["required_settings"].pop("ruc_2m_diagnostic", None)
    ruc_option["constraints"]["admitted_setting_values"]["ruc_2m_diagnostic"] = ["flux", "log_profile"]
    ruc_option["constraints"]["admitted_setting_values_reasons"]["ruc_2m_diagnostic"] = (
        "flux is public WRF's 2 m flux form and log_profile adds the "
        "operational RAP/HRRR branch's logarithmic profile; the two write "
        "different 2 m values, so no other name can select either")
    ruc_option["constraints"]["requires_components"]["surface_layer"] = [
        "revised-mm5", "classic-mm5", "mynn"]
    ruc_option["extensions"]["mynn_surface_ownership"] = {
        "surface_layer_writes_first": [
            "ust", "hfx", "qfx", "chs", "chs2", "cqs2", "flhc", "flqc",
            "t2", "q2", "th2"],
        "lsm_overwrites": ["tsk", "hfx", "qfx", "lh"],
        "lsm_preserves": ["ust", "flhc", "flqc", "chs2", "cqs2"],
        "post_lsm_overwrites": ["chs", "t2", "q2", "th2"],
        "wrf_source": (
            "phys/module_surface_driver.F:3500-3528,3579-3592; "
            "phys/module_sf_ruclsm.F:219-230,284-303"),
    }

    registry["parameters"]["spp_lsm"]["warnings"][0] = (
        "spp_lsm=1 consumes a member-owned stochastic pattern in RUC's "
        "historical WRF 3.9.1 hydraulic-conductivity operator. Current WRF "
        "4.6.1 and 4.7.1 retain the arguments but omit this operator. The "
        "ensemble provider owns its spectral restart state; ordinary "
        "physics refuses an enabled consumer without a bound pattern.")
    ruc_option["constraints"].setdefault("required_settings", {}).pop("spp_lsm", None)
    ruc_option["constraints"].setdefault("admitted_setting_values", {})["spp_lsm"] = [0, 1]
    for name, consumer, reader in (
            ("spp_conv", "GF closure", "woof/core/gf.py"),
            ("spp_pbl", "MYNN PBL and surface", "woof/core/physics.py")):
        registry["parameters"][name] = {
            "type": "integer", "enum": [0, 1], "per_domain": False,
            "consuming_read": reader,
            "warnings": [f"{name}=1 requires {consumer} and a member-owned stochastic "
                         "pattern bound by the ensemble timestep provider. The "
                         "provider owns the spectral restart state."]}

    route_text = (
        "The Noah-MP glacier refusal and sea-ice skip still apply. Its "
        "six-rate precipitation seam and radiation-cadence COSZEN carrier "
        "now follow WRF v4.6.1.")
    ceiling_text = (
        "Noah-MP's measured column ceiling is 360,000 columns, the widest "
        "configuration it has been timed at (2026-07-27). A wider grid is "
        "NOT refused: plan review WARNS and names the measured cost and its "
        "linear projection to the requested width, and "
        "WOOF_NOAHMP_EXPERT_COLUMN_BUDGET records consent to a larger "
        "budget when a receipt wants one. The number is measurement "
        "coverage, not a physical or memory limit.")
    for route in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast",
            "tools.prepared_single_domain_forecast"):
        registry["runner_routes"][route]["expert_warnings"][1] = ceiling_text
        registry["runner_routes"][route]["expert_warnings"][2] = route_text

    template = registry["templates"][
        "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"]
    template["warnings"][0] = (
        "EXPERT ONLY. The throughput reason this template was originally "
        "gated on is retired: the whole column runs on the device and is "
        "bitwise against the scalar authority. It stays expert-only because "
        "no woof/WRF forecast trajectory comparison exists. Dudhia supplies "
        "the carried COSZEN at radiation cadence; a radiation-free Noah-MP "
        "run instead seeds that carrier once from explicit geometry.")

    templates = registry["templates"]
    mynn_noah = templates[
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"]
    mynn_ruc_id = (
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1")
    mynn_ruc = copy.deepcopy(templates[
        "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"])
    mynn_ruc["components"]["pbl"] = "mynn"
    mynn_ruc["components"]["surface_layer"] = "mynn"
    mynn_ruc["label"] = (
        "WSM6 + MYNN PBL + MYNN surface layer + RUC LSM + Dudhia SW "
        "(HRRR pairing)")
    inherited_ruc_warnings = [
        warning for warning in mynn_ruc["warnings"]
        if not warning.startswith("This template differs from ")
    ]
    mynn_ruc["warnings"] = [
        mynn_noah["warnings"][0],
        (
            "This is the HRRR operational pairing class. WRF owns its "
            "write-back sequence explicitly: MYNN surface first, RUC "
            "flux/state write-back second, CHS and SFCDIAGS_RUCLSM last. "
            "It is offered only on routes where the established RUC template "
            "is already reachable; its nine-layer ingest and source "
            "restrictions are unchanged."),
        *inherited_ruc_warnings,
    ]
    templates[mynn_ruc_id] = mynn_ruc

    mynn_noahmp_id = (
        "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1")
    mynn_noahmp = copy.deepcopy(template)
    mynn_noahmp["components"]["pbl"] = "mynn"
    mynn_noahmp["components"]["surface_layer"] = "mynn"
    mynn_noahmp["label"] = (
        "WSM6 + MYNN PBL + MYNN surface layer + Noah-MP + Dudhia SW "
        "(expert only)")
    mynn_noahmp["warnings"] = [
        mynn_noah["warnings"][0],
        (
            "EXPERT ONLY. WRF owns the write-back sequence explicitly: MYNN "
            "surface first, Noah-MP flux/state write-back second, and the "
            "Noah-MP category/fraction 2-m diagnostic post-pass last. The "
            "existing Noah-MP glacier, sea-ice and verification-status warnings "
            "still apply."),
        *mynn_noahmp["warnings"],
    ]
    templates[mynn_noahmp_id] = mynn_noahmp

    # The RUC templates reach the experiment-per-domain route on exactly the
    # sources whose initializers run RUC's own nine-level soil ingest -- the
    # single-domain route's era5 list and the benchmark's hrrr list already
    # declare them.  A tree whose every domain carries the nine-layer soil
    # is one land surface throughout (land_surface is not a per-domain
    # override on this route), so nothing is mixed; before this the second
    # user-report tuple (new-tiedtke / RUC / Milbrandt-Yau / YSU) was refused
    # on era5 by route declaration alone.  gfs stays withdrawn (v1.1.1).
    tree_route_ruc = registry["runner_routes"][
        "tools.prepared_domain_tree_forecast"].setdefault(
            "source_template_ids", {})
    for source_id in ("era5", "hrrr"):
        declared = tree_route_ruc.setdefault(source_id, [])
        for template_id in (
                "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
                "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"):
            if template_id not in declared:
                declared.append(template_id)

    # Pairing reachability follows each LSM's established source discipline:
    # neither land-surface model becomes a broad component override.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if (
                "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
                in declared
                and mynn_ruc_id not in declared
            ):
                declared.append(mynn_ruc_id)
        for declared in route.get("expert_template_ids", {}).values():
            if (
                "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
                in declared
                and mynn_noahmp_id not in declared
            ):
                declared.append(mynn_noahmp_id)


# ------------------------------------------ aerosol-aware Thompson (mp=28)
#: G3 -- the whole aerosol column deck driven end to end through the shipped
#: adapter -- measured on this tree, one RTX 5090, FP32, gate 2.0e-6 relative
#: on every one of the 16 compared fields (15 column + rainnc).
#:
#: THE DECK IS TWENTY-TWO COLUMNS, NOT NINETEEN, and the registry now says so.
#: ``_FIXTURES`` in the gate is a glob over
#: ``woof/data/thompson/oracle-aero/*-column.csv``: the nineteen ``aero-*``
#: scenarios MP28_PORT_SPEC.md specifies (ids 101-119) PLUS three ``wp08-*``
#: columns (ids 120-122) that the same ``build_aero.sh`` invocation produced
#: in the same format and that pin every reachable ``nu_c`` and both branches
#: of the terminal phase cleanup.  Waves 1-4 published "nineteen" while the
#: gate drove twenty-two, which understated the deck AND hid two residuals
#: (``wp08-freeze``, ``wp08-nusweep``) in no published class at all.
#:
#: These are RE-MEASURED, not transcribed: ``tests/test_physics_registry.py::
#: test_mp28_published_residuals_still_equal_a_live_adapter_measurement``
#: reruns every fixture through the shipped adapter on the device and
#: rebuilds this partition.  Regenerate the numbers, never round them.
#:
#: WHAT MOVED SINCE THE LAST PUBLISHED SET, all of it re-measured on
#: 2026-08-01 through the shipped adapter.  TWO production changes did it,
#: both in mp=28-owned kernels, and the attribution below is the gate's own
#: (tests/test_thompson_aerosol_adapter.py::_G3_RESIDUALS records the
#: per-change deltas) rather than a summary invented here:
#:
#:   WP-13a, THE SEDIMENTATION DENSITY.  WRF builds the working rain mass and
#:   number sedimentation consumes in two places: at
#:   module_mp_thompson.F:3237-3238 from the :3193 TAU+1 density, for every
#:   level with L_qr, and again at :3568/:3570 from the :3490 POST-
#:   condensation density -- but only inside the :3501-3502 gate
#:   (``ssatw < -eps .and. L_qr .and. .not. prw_vcd > 0``).
#:   woof/core/kernels/thompson_aerosol_sat.cu wrote the post-condensation
#:   density into its ``reference_density`` output unconditionally, so every
#:   level got the :3568 answer including the levels WRF never rewrote; it now
#:   defaults to the :3237 density and overwrites it level by level from its
#:   own transcription of those three gates.
#:
#:   WP-13b, CONTRACTION PINNING OF THE SOURCE-NETWORK APPLY.  WRF's
#:   :3973-4023 terminal apply is ``q1d(k) = q1d(k) + qten(k)*DT`` and the
#:   gfortran -O2 baseline-x86-64 oracle has no FMA instruction, so qten*DT is
#:   rounded to REAL(4) before the add; nvrtc contracted the same expression
#:   in thompson_aerosol_cold.cu and thompson_aerosol_warm.cu and never
#:   rounded it.  Both now round it, as thompson_aerosol_sat.cu's rain-
#:   evaporation apply already did.
#:
#: WHAT THAT DID TO THE PUBLISHED PARTITION:
#:   * ``aero-drop-evap`` LEFT the residual list entirely and is published
#:     CLEAN -- WP-13a alone: rainnc 5.165e-04 -> 0.000e+00 (BIT-EXACT),
#:     qr 3.533e-05 -> 7.346e-08, nr 2.258e-05 -> 3.919e-07.
#:   * ``aero-ice-demott-idxin`` LEFT it too, and needed BOTH changes: WP-13a
#:     took sr and rainnc/rainncv 1.279e-04 -> 3.5e-07 and qr 2.894e-05 ->
#:     1.35e-06, then WP-13b took rainnc/rainncv and sr to BITWISE 0.000e+00
#:     and qr to 6.416e-07.  Its nr now measures 3.243e-07.
#:   * ``aero-cloud-freeze-nc`` lost three of the four rows this table
#:     published -- qr 2.800e-05 -> 8.973e-08, nr 1.797e-05 -> 2.594e-07,
#:     rainnc 1.162e-05 -> 0.000e+00, all WP-13a -- and is published with
#:     ``qc`` 4.926e-06 alone.
#:   * ``aero-reduces-to-classic``'s carve-out lost its ``qr`` FIELD:
#:     7.813e-05 -> 1.788e-07, inside the FLAT 2.0e-6 gate, so the gate's
#:     ``_END_TO_END_BOUNDS`` now names ``nr_per_kg`` only and the bound on it
#:     went 1.0e-04 -> 1.0e-05 (ten times stricter) on a measurement of
#:     5.700e-06.  The registry published {qr, nr_per_kg} against a gate that
#:     applies {nr_per_kg}; that is corrected here.
#:   * ``_REFL_DB_BOUNDS`` was RETIRED, not widened: the reflectivity residual
#:     it covered went 5.283e-04 dB -> 3.242e-05 dB, inside the flat 2.0e-4 dB
#:     gate.  The published allowance list is therefore TWO entries where it
#:     was three.
#:   * ``aero-cold-overlap`` GOT WORSE ON ONE FIELD and is published that way:
#:     qr 3.667e-05 -> 4.443e-05 at 0-based level 6.  That growth is WP-13b's,
#:     bisected to a single line -- the cold network's qr apply -- and it was
#:     KEPT rather than reverted because reverting it also loses all four of
#:     aero-ice-demott-idxin's improvements above, two of which are bitwise.
#:     In ulps of the entry value, the scale this cell is recorded at because
#:     99.5% of the level's rain is consumed in the step, the move is 1.477 ->
#:     1.789 ulp.  The same change IMPROVED this fixture's nr, 1.340e-04 ->
#:     1.261e-04, and its level-4 rows are unchanged.  The wave-5 registry
#:     published 3.667e-05, which UNDERSTATED the port's own error by 21%;
#:     publishing the larger number is the point of re-measuring.
#:
#: WHAT MOVED IN THE WAVE BEFORE THAT, kept because the number it retires is
#: still quoted in the public documents:
#:   * ``aero-ice-koop`` LEFT the residual list.  Its published qi 1.612e-03 /
#:     ni 1.764e-03 / effi 5.093e-05 -- called "the largest genuine gap" by
#:     the registry, PHYSICS.md, PROVENANCE.md D9k and the evidence page --
#:     now measure 1.534e-07 / 3.396e-07 / 1.886e-07 and the fixture is
#:     CLEAN.  IT WAS NOT CLOSED BY A KERNEL.  It was closed by correcting
#:     the ORACLE HARNESS: tools/thompson_wrf461_oracle/run_column_aero.F90
#:     built the Exner function with rd_over_cp = 287.0/1004.0 where WRF's
#:     own rcp is r_d/cp = 287./(7.*287./2.) = 2/7
#:     (share/module_model_constants.F:19,:20,:31), 4774 float32 ulps away,
#:     so the recorded (p, theta) pair could not be inverted exactly on the
#:     woof side and the adapter drove dozens of the deck's 528 entry
#:     levels (47 as re-pinned 2026-08-03; first published as 40, the
#:     original environment's libm) from a perturbed pressure.  The deck was regenerated with WRF's own
#:     constant, with no change to any .cu or .cuh file.  The registry was
#:     10,500x pessimistic about its own worst number, and about a residual
#:     that its own reference harness had manufactured.  See
#:     docs/public/wrf-comparison/mp28-column-evidence.md section 3.4.
#:   * ``aero-cloud-freeze-nc`` lost its ``effc_m`` row (5.018e-06 ->
#:     1.619e-06, inside the gate) and its ``qc`` fell 1.478e-05 -> 4.926e-06.
#:   * ``aero-ice-demott-idxin`` lost its ``qc`` row (6.031e-06 -> 7.556e-08)
#:     and its ``qr`` fell 3.895e-05 -> 2.894e-05.
#:   * ``aero-cold-overlap`` got WORSE and is published that way: a new
#:     ``qc``/``nc_per_kg`` pair at 1.000e+00 and ``effc_m`` at 8.102e-01.
#:     See the note on that entry -- it is one mechanism at one level, not
#:     three failures, and it is a full-scale relative number on a ONE-ULP
#:     absolute difference.
#:   * ``wp08-freeze`` and ``wp08-nusweep`` were published for the first time.
#:
#: THE UNEXCEPTIONED CLEAN SET.  22 of 22, since the 2.8.6 accumulator
#: rework: every committed fixture clears the flat 2.0e-6 / 2.0e-4 dB gate on
#: all 23 compared quantities with nothing held out.  It was 18 of 22 before
#: it (sixteen of the nineteen spec'd ``aero-*`` fixtures plus
#: ``wp08-freeze`` and ``wp08-melt``); the four that joined are
#: ``aero-cloud-freeze-nc``, ``aero-cold-overlap``, ``aero-reduces-to-classic``
#: and ``wp08-nusweep``, and the note on MP28_G3_RESIDUALS says what closed
#: each.  This tuple is asserted EQUAL to the gate's own
#: ``_G3_UNEXCEPTIONED_CLEAN`` by
#: ``tests/test_physics_md_aerosol_claims.py::
#: test_the_published_clean_counts_are_the_gates_own_counts``, so it cannot be
#: a transcription that drifts.
#:
#: AND NOW 21 OF 22 (2.8.8).  ``aero-cold-overlap`` LEFT: it is the declared
#: rain-graupel divergence (the gate's ``_G3_DECLARED_DIVERGENCE``).  WRF
#: v4.6.1 reads its rain-graupel collision tables out of bounds when the
#: scheme is not hail aware (dimNRHG = 1, idx_bg = 5;
#: module_mp_thompson.F:465, :607-615, :2527-2545) and WOOF reads the one
#: slab the tables hold.  This registry kept publishing the fixture as clean
#: after the gate moved it; re-measured through the shipped adapter on an
#: RTX 4090 (sm_89) and an RTX 5070 Ti (sm_120), the unexceptioned table
#: reads 21 of 22 with that fixture the one miss, at exactly the values
#: MP28_G3_RESIDUALS publishes.
MP28_G3_CLEAN = (
    "aero-ccn-activate", "aero-ccn-sweep", "aero-cloud-freeze-nc",
    "aero-drop-evap",
    "aero-ice-demott-dep", "aero-ice-demott-idxin", "aero-ice-koop",
    "aero-init-profile", "aero-nc-accrete", "aero-nc-auto", "aero-nc-cap",
    "aero-nc-effrad", "aero-nc-sed", "aero-reduces-to-classic",
    "aero-scav-frozen", "aero-scav-rain",
    "aero-sfc-emit", "aero-warm-overlap", "wp08-freeze", "wp08-melt",
    "wp08-nusweep",
)

#: Fixtures that do NOT clear 2e-6 on every field, with every field that
#: misses and its measured maximum relative difference.  NONE of twenty-two
#: from the 2.8.6 accumulator rework until 2.8.8 declared the rain-graupel
#: divergence; ONE since (the note at the end of this comment).
#:
#: WHAT CLOSED THEM.  WRF's mp_thompson never writes a hydrometeor during the
#: call: every process adds to a running tendency (qcten, qrten, nrten,
#: qiten, niten), every stage reads X1d + Xten*DT, and the terminal apply
#: (module_mp_thompson.F:3972-4053) rounds each species once, with the size
#: bounds after the :3943-3966 phase cleanup.  The port applied each stage to
#: the state in turn, so a species was rounded once per stage, and the rain
#: and ice fallout it shared with mp=8 folded the terminal size bounds in
#: before the freeze.  mp=28 (v4.6.1 generation) now carries those five
#: tendencies, runs the :3033-3055 / :3070-3091 balances in tendency form,
#: has its own tendency-form rain and ice fallout, applies once in a terminal
#: kernel, and takes the cold network's rain-conservation ratio as REAL as
#: :1615 declares it.  Measured on the shipped adapter, RTX 5090 and RTX 4090
#: identical: aero-cloud-freeze-nc qc 4.926e-06 -> 0 (bit-exact);
#: aero-cold-overlap qc / nc / effc at level 4 and qr / nr at level 6 -> 0;
#: wp08-nusweep qr 4.642e-06 -> 5.532e-07; aero-reduces-to-classic qr / nr at
#: level 6 bit-exact, retiring the near-cancellation allowance.  The frozen
#: mp=8 kernel file thompson.cu is unchanged.
#:
#: THE TEXT BELOW IS THE RECORD OF WHAT CLOSED, kept because the numbers are
#: still quoted as history in the public documents.
#:
#: ``aero-cold-overlap``'s 1.000e+00 rows are the accurate publication of a
#: sub-ulp disagreement and are recorded rather than allowanced.  MEASURED
#: at 0-based level 4: the level enters with qc = 2.3252160e-04 kg/kg and
#: nc = 9.1306704e+07 per kg; WRF ends the step with qc =
#: 1.4551915228366852e-11 kg/kg -- which is EXACTLY 2**-36, exactly 1.000
#: float32 ulp of the entry value -- and nc = 1.8333361 per kg, while woof
#: ends at exactly 0.0.  module_mp_thompson.F:4007-4009 is
#: ``if (qc1d(k) .le. R1) then qc1d = 0.0 ; nc1d = 0.0`` with R1 = 1.E-12
#: (:183), so WRF takes the ELSE branch and its :4011-4020 nu_c/lamc/xDc size
#: bound hands back nc = 1.833336 per kg where woof takes the THEN branch and
#: zeroes it.  A relative metric therefore reports 1.0 (qc, nc) on
#: an absolute difference of one ulp and 0.229 ulp respectively, and effc
#: reports 8.102e-01 because :5638 does the same thing again: a level with
#: rc <= R1 CYCLEs and keeps RE_QC_BG = 2.49 um
#: (share/module_model_constants.F:62, installed at :5619) while WRF's
#: 1.455e-11 kg/kg remainder gives 1.31176e-05 m.  One branch flip, three
#: views.  The fixture's OTHER residual is separate and real: nr 1.261e-04 /
#: qr 4.443e-05 at level 6, where the rain number falls 255.407 -> 0.0739 per
#: kg (99.97% consumed) and the difference is 0.611 ulp (nr) / 1.789 ulp (qr)
#: of the entry value.
#:
#: ``wp08-freeze`` nr and ``wp08-nusweep`` qr are both fields CREATED FROM
#: EXACTLY ZERO inside the step, at 1.4x and 2.3x the gate.  ONE of the two
#: has a traced mechanism and it is in a file mp=28 may not touch:
#: woof/core/kernels/thompson.cu:438 gates rain sedimentation on
#: ``qr > 1.0e-12``, a MIXING RATIO, where module_mp_thompson.F:3616 tests
#: ``rr(k) > R1``, a MASS CONCENTRATION, so at wp08-freeze level 1 -- qr =
#: 8.5265e-13 kg/kg but rr = 1.1748e-12 kg/m3 -- WRF gives the level a real
#: fall speed and ArWen treats it as rain-free.  That kernel is the frozen,
#: wrf-matched-run mp=8 one, so the residual is recorded and filed as an
#: integration request rather than fixed here.
#:
#: THE TWO wp08 CELLS SWAPPED PLACES AT THE 1.4.1 MERGE.
#:
#: ``wp08-nusweep`` qr level 12 -- the cell this comment used to say NO
#: MECHANISM IS CLAIMED for -- is now explained, by CONDITIONING.  Perturbing
#: the level's entry state by exactly one float32 ulp and re-running the
#: adapter moves the exit qr by 128 ulp (entry qc +1), 32 ulp (entry qc -1)
#: or 256 ulp (entry nc, either direction).  The disagreement is 60 ulp,
#: smaller than a single-ulp input change produces, and the 2.0e-06 gate is
#: about 26 ulp there -- below the cell's own condition number.  No FP32
#: implementation can hold it, and |got - want| is 1.04e-16 kg/kg, still the
#: smallest absolute disagreement in this table by nine decades.
#:
#: ``wp08-freeze`` nr level 0 is EXPLAINED and the explanation is measured,
#: not argued.  29 of its 34 ulp are the frozen mp=8 kernel gating rain
#: presence on a mixing ratio (thompson.cu:438, qr > 1.0e-12f) where WRF
#: gates on a mass concentration (:3616, rr .gt. R1).  Forcing ArWen's gate
#: open at the one level where the two disagree moves level 0's nr from 34
#: ulp away from WRF to 5, and a same-sized mass change that does not flip
#: the gate leaves the output bit-identical.
#:
#: It was published as falsified for part of 2026-08-01 and that was wrong:
#: the falsification read qr1d + qrten*DT at the END of the step and took it
#: for the value :3236 tested, while :3501's evaporation block subtracts
#: from qrten in between.  Instrumented WRF records L_qr = .true. there.
#: cb765336 did not move the residual because it reconciled the
#: sedimentation DENSITY, not the gate's UNITS.
#:
#: CLOSED 2026-09-23: the mp=28 rain evaporation writes WRF's L_qr into
#: its reference density (zero where :3236 failed) and the adapter launches
#: the fallout's ``_with_presence`` entry points, which read it; the plain
#: entry points keep the mixing-ratio stand-in, and the classic rain
#: evaporation and the mp=8 adapter carry the same hand-off since
#: 7727fda3c.
#: Level 0 nr went 2.724e-06 (34 ulp) to 4.006e-07 (5 ulp) on a card (RTX
#: 4090 and RTX 5090 alike; 8.012e-08, 1 ulp, on the host build of the
#: kernels) and the fixture left this table.
#:
#: ONE ROW SINCE 2.8.8, AND IT IS THE DECLARED DIVERGENCE, not a rounding
#: residual: ``aero-cold-overlap``, the one fixture carrying rain and graupel
#: together, where WOOF reads the rain-graupel slab WRF's tables hold and WRF
#: reads past it (see MP28_G3_CLEAN).  The values are the gate's own
#: ``_G3_RESIDUALS`` in the 16-field contract shape, re-measured through the
#: shipped adapter on an RTX 4090 (sm_89) and an RTX 5070 Ti (sm_120) and
#: identical on both; its reflectivity misses by 1.585e-03 dB.  With WRF's
#: read emulated in a measurement copy of the tree (never shipped) the
#: fixture clears all 23 quantities, so nothing else in it differs.
MP28_G3_RESIDUALS: dict[str, dict[str, float]] = {
    "aero-cold-overlap": {
        "qr": 5.070e-04, "qi": 1.618e-04,
        "qg": 8.776e-05, "ni_per_kg": 1.464e-05, "nr_per_kg": 5.066e-04,
        "effi_m": 5.394e-05, "effs_m": 6.574e-06,
    },
}

#: The one fixture that clears the gate only through a carved-out bound, and
#: the ONE FIELD that bound still covers.
#:
#: THE FIELD SET IS PART OF THE PUBLICATION, not decoration.  ``carved_out_
#: bound`` is what a reader is told the port bought itself, so publishing
#: {qr, nr_per_kg} against a gate whose ``_END_TO_END_BOUNDS`` names
#: {nr_per_kg} overstates the relaxation by a whole quantity.  It is bound to
#: the gate's own dict by ``tests/test_physics_registry.py::
#: test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_applies``,
#: which compares the SETS and not only the values.
#:
#: THE HISTORY, because two successive tightenings are easy to mistake for a
#: widening.  The bound was 2.5e-03 on {qr, nr_per_kg}, accommodating a
#: 1.915e-03 / 1.922e-03 residual at 0-based level 5.  WP-12a found the cause
#: -- module_mp_thompson.F:3490 overwrites rho(k) inside the condensation
#: loop and :3505-3520's orho/rhof/vsc2/rvs read THAT one, so prv_rev scales
#: with the PRE-condensation density and the adapter was not passing it --
#: and the bound went 2.5e-03 -> 1.0e-04, twenty-five times stricter, on a
#: re-measured 7.813e-05 / 4.832e-05.  WP-13a then restored WRF's level-wise
#: :3237-vs-:3568 sedimentation density, ``qr`` fell to 1.788e-07 and LEFT
#: the dict entirely (it clears the FLAT 2.0e-06 gate), and the surviving
#: ``nr_per_kg`` bound went 1.0e-04 -> 1.0e-05, ten times stricter again, on
#: a measurement of 5.700e-06.  The sequence is 2.5e-03 -> 1.0e-04 -> (qr
#: deleted, nr 1.0e-05).  Nothing was widened at any point.
#:
#: WHAT THE SURVIVING NUMBER IS.  0-based level 5 is the one level of this
#: column where the step removes a large fraction of the rain number without
#: emptying it (49.75%: 3.000000e+05 -> 1.507546e+05 per kg), and 5.700e-06
#: is 27.5 ulps of the entry value; every other unexcluded level is 0-3 ulps.
#: RE-MEASURED on 2026-08-01 through the shipped adapter: 5.7005e-06.
MP28_G3_CARVED_OUT: dict[str, dict[str, float]] = {
    # EMPTY.  RETIRED AT THE 1.4.1 MERGE, not narrowed again.
    #
    # It read {"aero-reduces-to-classic": {"nr_per_kg": 5.700e-06}} and its
    # gate-side bound was _END_TO_END_BOUNDS = 1.0e-5.  Merging
    # integration/release-1.4.1 inherited the mp=8 lane's two rain
    # sedimentation reconciliations -- 5e4af4e3 ("the rain MVD bound belongs
    # to TAU+1, not to sedimentation") and cb765336 ("the rain-presence gate
    # is a mass concentration, floor included") -- into the byte-frozen
    # thompson.cu mp=28 shares for rain fallout.  No mp=28 file changed.
    # RE-MEASURED through the shipped adapter on the merged tree, 0-based
    # level 5: nr_per_kg 4.146e-07, inside the FLAT 2.0e-06 gate by a factor
    # of 4.8.  The bound had nothing left to buy and was deleted.
    #
    # The full sequence, none of it a widening: 2.5e-03 on {qr, nr_per_kg}
    # -> 1.0e-04 -> (qr deleted, nr 1.0e-05) -> GONE.
    #
    # aero-reduces-to-classic joined MP28_G3_CLEAN at the 2.8.6 accumulator
    # rework: _NEAR_CANCELLATION_LEVELS, which held 0-based level 6 to 32 ULP
    # of the entry value, is retired because that level is now bit-exact
    # against WRF in qr and nr.  No departure from the flat gate remains.
}

MP28_OPTION_ID = "thompson-aerosol-mp28"


def _thompson_hail_mp38(registry: dict) -> None:
    """Keep the hail selector in the tracked registry declarations."""
    options = registry["components"]["microphysics"]["options"]
    options[MP28_OPTION_ID]["constraints"].setdefault(
        "admitted_setting_values", {})["thompson_version"] = ["wrf_461", "wrf_39_noaa"]
    row = copy.deepcopy(options[MP28_OPTION_ID])
    row["parameters"]["thompson_version"] = "wrf_461"
    row["parameters"]["thompson_fork_snow_fall"] = "blend"
    row["parameters"].update(force_read_thompson=False,
                             write_thompson_tables=True,
                             write_thompson_mp38table=False)
    row.update({
        "label": "Thompson graupel-hail / MP38",
        "selectors": {"mp_physics": 38},
        "extensions": {"density_bins": 9, "wrf_package": "thompsongh"},
        "maturity": "implemented-unverified",
        "scientific_evidence": "none",
        "warnings": [
            "Strict replay of four real native WRF columns, including two with "
            "active graupel mass, number and volume, has zero differing words "
            "(3752 state, 2744 literal tendencies). Three-hour real forecasts, "
            "live-moment restart, both nest "
            "feedback modes and an ordinary two-member ensemble were exercised. "
            "Synthetic strict residue remains declared. Forecast maps show broad "
            "weak precipitation east of the observed line; no skill qualification "
            "or whole-model trajectory parity is claimed."],
    })
    from woof.core.thompson_contract import HAIL_TABLE_ASSETS, HAIL_TABLE_SET_ID
    row["asset_requirements"].append({
        "id": HAIL_TABLE_SET_ID,
        "kind": "operator-supplied-table-set",
        "source": "WRF v4.6.1 phys/module_mp_thompson.F initialization; byte-identical in v4.7.1",
        "regenerable": True,
        "redistributed_by_gpuwm": False,
        "root_environment_override": "WOOF_THOMPSON_HAIL_TABLE_ROOT",
        "resolution": dict(_ASSET_RESOLUTION_LADDERS[HAIL_TABLE_SET_ID]),
        "assets": [{"filename": asset.filename, "bytes": asset.bytes,
                    "sha256": asset.sha256} for asset in HAIL_TABLE_ASSETS],
        "note": "Nine graupel-density slabs. The shipped hail-build harness generates the pinned cache outside the forecast step; the loader validates every asset.",
    })
    row["constraints"].setdefault("admitted_setting_values", {})["thompson_version"] = ["wrf_461"]
    row["constraints"].setdefault("required_settings", {})["thompson_version"] = "wrf_461"
    row["constraints"].setdefault("required_settings_reasons", {})["thompson_version"] = (
        "the WRF 3.9 fork has no prognostic graupel number, volume or density bins")
    row.pop("consumers", None)
    options["thompson-graupel-hail-mp38"] = row


def _thompson_aerosol_mp28(registry: dict) -> None:
    """Register aerosol-aware Thompson at the maturity its evidence earns.

    Three declarations carry the whole claim and are worth stating together,
    because a reader who takes any one of them alone gets a wrong answer:

    ``implemented: true`` -- woof has the component.  The scheme runs on the
    device through ``woof/core/microphysics_aerosol.py`` and is dispatched by
    ``woof/core/microphysics.py`` on ``mp_physics == 28``.

    ``maturity: implemented-unverified`` -- and no higher.  PHYSICS.md's
    published definition of that label is exactly this option's state: column
    -oracle-measured against unmodified WRF Fortran, with no forecast
    -trajectory comparison.  It may not claim ``wrf-matched-run-candidate`` (no
    ratified reference comparison exists) and certainly not
    ``wrf-matched-run`` (no matched multi-hour run, no decay tables).  What it
    also may not do is claim the column evidence is CLEAN, so the measured G3
    residuals are published on the option itself rather than left in a test
    file: four of twenty-two fixtures miss the 2e-6 gate and a fifth clears
    it only under a carved-out bound.

    ``reachability: component-override`` -- computed, not chosen.  The tree
    route already lists ``microphysics`` in ``allowed_component_overrides``,
    so an implemented microphysics option carrying no template is reachable
    exactly one way: as a per-domain experiment override.  Registering no
    template and leaving ``DEFAULT_TEMPLATE_ID`` alone is what keeps it out of
    every default suite; ``tests/test_registry_reachability.py`` recomputes
    the state from the routes and would fail on any other declaration.
    """

    options = registry["components"]["microphysics"]["options"]
    options[MP28_OPTION_ID] = {
        "asset_requirements": [
            {
                # ``kind`` is ``packaged-table-set`` (audit R-045).  It said
                # ``operator-supplied-table-set`` from before woof shipped
                # the file, and every reader of the row -- including the
                # refusal text mp=28 printed when the table was missing --
                # inherited that stale word and told the operator to supply
                # a table a default install already has.  The row's own
                # note said the opposite in the same object, and
                # ``redistributed_by_gpuwm`` said it in a field.  ``kind``
                # now agrees with both.
                #
                # It remains deliberately NOT a member of the packaged
                # CLASSIC table set mp=8 resolves through TABLE_SET_ID: a
                # separate set with its own id, its own root and its own
                # environment overrides, so no mp_physics=8 launch acquires
                # a dependency on it.  ``kind`` names how the bytes ARRIVE,
                # which set they belong to is the ``id``.
                "id": "wrf-v4.6.1-aerosol-thompson-mp28-v1",
                "kind": "packaged-table-set",
                "assets": [
                    {
                        "filename": "CCN_ACTIVATE.BIN",
                        "bytes": 35288,
                        "sha256": (
                            "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75"
                            "301e422f147c82a3dbd"),
                    },
                ],
                "redistributed_by_gpuwm": True,
                "source": (
                    "WRF v4.6.1 (git tag v4.6.1, commit "
                    "d66e442fccc04111067e29274c9f9eaccc3cef28), file "
                    "run/CCN_ACTIVATE.BIN"),
                "search_root": "woof_data/data/thompson/tables",
                "root_environment_override": "WOOF_THOMPSON_TABLE_ROOT",
                "path_environment_override": "WOOF_THOMPSON_CCN_ACTIVATE",
                "regenerable": False,
                "note": (
                    "table_ccnAct (phys/module_mp_thompson.F:5110-5166) READS "
                    "this file; it computes nothing. The numbers are offline "
                    "parcel-model output (Feingold & Heymsfield as modified "
                    "by Eidhammer and Kreidenweis, WRF's own comment at "
                    ":5102-5108), so no recompilation of WRF, no re-run of "
                    "thompson_init and no woof code path regenerates it. "
                    "woof redistributes WRF's file verbatim -- it is "
                    "committed under search_root, listed in that directory's "
                    "MANIFEST.sha256 and shipped in the recast-woof-data companion "
                    "wheel that `pip install recast-woof` pulls, under WRF's "
                    "public-domain dedication whose notice travels in "
                    "woof/data/wrf_radiation/LICENSE-WRF.txt -- so a default "
                    "install satisfies this requirement and the environment "
                    "overrides exist to bind a run to another copy instead. "
                    "It "
                    "is not in thompson_contract.CLASSIC_TABLE_ASSETS and "
                    "TABLE_SET_ID is unchanged, so no mp_physics=8 launch "
                    "acquires a dependency on it. Size and SHA-256 are pinned "
                    "in woof/core/thompson_aerosol_contract.py and checked "
                    "on every load; absence is fatal and never defaulted, and "
                    "a byte-different table is refused rather than used."),
            },
        ],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3036 binds mp_physics==28 to the "
                "thompsonaero package: moist qv,qc,qr,qi,qs,qg and scalar "
                "qni,qnr,qnc,qnwfa,qnifa,qnbca"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "nr", "ni", "nc",
                                "nwfa", "nifa"],
                "surface_emission_2d": ["nwfa2d", "nifa2d"],
                "not_ported": ["qnbca", "taod5502d", "taod5503d"],
                "note": (
                    "qnbca (black-carbon aerosol number) is out of scope for "
                    "v1 and refused rather than zero-filled; taod5502d/"
                    "taod5503d are radiation-side aerosol optical-depth "
                    "diagnostics that mp_gt_driver does not produce."),
            },
            "column_oracle_evidence": {
                "authority": (
                    "unmodified WRF v4.6.1 phys/module_mp_thompson.F at "
                    "commit d66e442fccc04111067e29274c9f9eaccc3cef28, "
                    "compiled by gfortran 13.3.0 -O2"),
                # 22 columns, not 19.  ``spec_fixtures`` is the count
                # MP28_PORT_SPEC.md names (the aero-* ids 101-119);
                # ``fixtures`` is what the gate actually drives, which is
                # that set plus three wp08-* columns (ids 120-122) from the
                # same build_aero.sh run.  Publishing only the smaller
                # number is how wp08-freeze and wp08-nusweep sat above the
                # gate in no published class for four waves.
                "fixtures": 22,
                "spec_fixtures": 19,
                "compared_fields": 16,
                # The 16 above are the contract shape the residual table is
                # published in (15 prognostic column fields + rainnc_mm).
                # The GATE compares 23 and asserts that width so it cannot be
                # narrowed back: the 15, six more surface diagnostics
                # (rainncv, snownc, snowncv, graupelnc, graupelncv, sr) plus
                # rainnc, and REFL_10CM against WRF's own calc_refl10cm at a
                # 2.0e-4 dB gate.
                "compared_quantities": 23,
                "compared_quantities_breakdown": {
                    "column_prognostic": 15,
                    "surface_accumulation": 7,
                    "reflectivity_db": 1,
                },
                "gate_relative": 2.0e-6,
                "gate_reflectivity_db": 2.0e-4,
                "clean_fixtures": list(MP28_G3_CLEAN),
                "residual_fixtures": {
                    name: dict(fields)
                    for name, fields in sorted(MP28_G3_RESIDUALS.items())},
                "carved_out_bound": {
                    name: dict(fields)
                    for name, fields in sorted(MP28_G3_CARVED_OUT.items())},
                # The gate's SECOND relaxation, RETIRED by the 2.8.6
                # accumulator rework.  Kept as a record with no fixtures so a
                # reader of an older registry finds where it went; it is
                # bound to the gate's own constants by
                # tests/test_physics_registry.py::
                # test_mp28_publishes_the_near_cancellation_relaxation_too.
                "near_cancellation_bound": {
                    "fixtures": {},
                    "ulps_of_entry_value": 32.0,
                    "measured": {},
                    "retired": (
                        "aero-reduces-to-classic level 6 used to be held to "
                        "32 ulps of the entry value (measured 0.585 qr / "
                        "0.159 nr) because 99.958% of the level's rain "
                        "evaporates in one 10 s step. The 2.8.6 accumulator "
                        "rework applies qrten/nrten once, as WRF does, and "
                        "the level is now bit-exact against WRF in qr and "
                        "nr, so the bound buys nothing."),
                },
                # EVERY DEPARTURE FROM THE FLAT GATE, NAMED.  NONE since the
                # 2.8.6 accumulator rework; every one that ever existed is in
                # retired_allowances, and every one moved STRICTER before it
                # went.  The gate's own ``_G3_ALLOWANCES`` is the authority
                # and tests/test_physics_registry.py::
                # test_mp28_evidence_publishes_the_allowances_the_gate_
                # actually_has reads it back.
                "allowances": [],
                "retired_allowances": [
                    {"name": "near_cancellation_bound",
                     "gate_constant": "_NEAR_CANCELLATION_LEVELS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": "level 6 held to 32 ulps of the entry value",
                     "is": None,
                     "direction": "retired by the 2.8.6 accumulator rework; "
                                  "the level is bit-exact against WRF in qr "
                                  "and nr"},
                    {"name": "carved_out_bound",
                     "gate_constant": "_END_TO_END_BOUNDS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": 1.0e-05,
                     "is": None,
                     "direction": "retired at the 1.4.1 merge; the residual "
                                  "it covered is now 4.146e-07, inside the "
                                  "flat 2.0e-06 gate"},
                    {"name": "reflectivity_bound_db",
                     "gate_constant": "_REFL_DB_BOUNDS",
                     "fixtures": ["aero-reduces-to-classic"],
                     "was": 1.0e-03,
                     "is": None,
                     "direction": "retired; the residual it covered is now "
                                  "3.242e-05 dB, inside the flat 2.0e-4 dB "
                                  "gate"},
                ],
                # The two counts, stated separately, because conflating them
                # is how a port claims a clean number it did not earn.
                "clean_unexceptioned": len(MP28_G3_CLEAN),
                "clean_as_gated": len(MP28_G3_CLEAN),
                "clean_counts_note": (
                    "21 of 22 clear a FLAT 2.0e-6 relative / 2.0e-4 dB gate "
                    "on all 23 quantities with no bounds dict, no excluded "
                    "level and no per-fixture carve-out -- 18 of the 19 "
                    "spec'd aero-* fixtures plus wp08-freeze, wp08-melt and "
                    "wp08-nusweep. The 22nd, aero-cold-overlap, misses by "
                    "the declared rain-graupel divergence alone (WRF v4.6.1 "
                    "reads its rain-graupel collision tables out of bounds "
                    "when the scheme is not hail aware; WOOF reads the one "
                    "slab the tables hold) and is published in "
                    "residual_fixtures; with WRF's read emulated in a "
                    "measurement copy of the tree it clears all 23 "
                    "quantities. The gated count is the same 21 of 22: there "
                    "is no allowance left, and the three that ever existed "
                    "are in retired_allowances. A clean column deck is not a "
                    "forecast validation; the maturity stays "
                    "implemented-unverified on the forecast evidence."),
                # No longer None.  docs/public/wrf-comparison/
                # mp28-matched-trajectory.md is a matched IDEALIZED forecast
                # against unmodified WRF v4.6.1, and it publishes its own
                # FAILED gate rather than a summary of the parts that passed.
                "forecast_trajectory_comparison": {
                    "document": ("docs/public/wrf-comparison/"
                                 "mp28-matched-trajectory.md"),
                    "kind": "idealized single-domain doubly-periodic forecast",
                    "not_nested_not_real_data": (
                        "This retained comparison is idealized, single-domain "
                        "and periodic. A matched nested or real-data WRF "
                        "comparison was not attempted. WRF's real.exe refuses "
                        "wif_input_opt=0 with mp_physics=28 "
                        "(dyn_em/module_initialize_real.F:2734-2736), but that "
                        "does not make a comparison on the current WIF route "
                        "impossible. The current route couples nwfa/nifa from "
                        "the WIF climatology and refuses specified domains "
                        "without the required dataset. Matching initial and "
                        "boundary inputs and retaining the comparison results "
                        "remain evidence gaps, not absent runtime capability."),
                    "case": (
                        "WRF's own em_quarter_ss initializer with the "
                        "hodograph removed: a 3 K cos^2 thermal, 10 km "
                        "radius, centred at z = 1500 m, in an unsheared WK82 "
                        "sounding. 120 x 120 x 40, dx = dy = 2000 m, "
                        "ztop = 20000 m, dt = 12 s, time_step_sound = 6, 600 "
                        "steps = 7200 s, periodic_x = periodic_y = .true., "
                        "microphysics the only physics."),
                    "initial_condition": (
                        "woof is initialised FROM WRF's own wrfinput_d01, "
                        "not from a transcription of WRF's initializer, so a "
                        "transcription difference cannot enter the "
                        "trajectory disguised as a microphysics difference. "
                        "MEASURED: the t = 0 normalised RMS field difference "
                        "is exactly 0.0 in ten of the thirteen compared "
                        "fields, in both configurations. The three that are "
                        "not are the three woof derives rather than copies: "
                        "T at 2.510e-09 (WRF stores theta - 300 and woof "
                        "stores an absolute base plus a perturbation, so the "
                        "field is split and recombined once each way), and "
                        "QNWFA at 2.176e-08 / QNIFA at 1.731e-08, which both "
                        "models overwrite with thompson_init's synthetic "
                        "profile -- so what is compared there is CUDA's "
                        "evaluation of WRF's analytic CCN/IN profile against "
                        "gfortran's. One float32 rounding each."),
                    "wrf_reference": (
                        "WRF v4.6.1 release tarball sha256 b8ec11b2..., "
                        "gfortran 13.3.0, glibc 2.39, netCDF 4.9.2, "
                        "configure option 32 (serial) / nesting 0, built "
                        "TWICE from identical source: build A with WRF's own "
                        "default -O2 -ftree-vectorize -funroll-loops and "
                        "build B with -O2 -fno-tree-vectorize. Build B is "
                        "the control: it measures how far WRF's own answer "
                        "moves under one optimization flag."),
                    "design": (
                        "difference-in-differences. The tested quantity is "
                        "the aerosol SIGNATURE M(mp28) - M(mp8) measured "
                        "inside each model, read against two published "
                        "floors: the mp=8 model-pair disagreement (woof is "
                        "wrf-matched-run there) and WRF's own flag "
                        "sensitivity. Six runs; every woof configuration "
                        "run twice and byte-compared, the 5090 having no "
                        "ECC."),
                    "declared_verdict": "HOLD",
                    "declared_verdict_detail": (
                        "V1 sign agreement PASS 43/47 = 91.5% (threshold "
                        "90%); V2 floor-calibrated magnitude PASS 9/9; V3 no "
                        "scheme-level amplification FAIL, 7 of 197 rows over "
                        "3x (V3 as declared spans M1-M4 and M8: 111 scalar "
                        "rows and 86 field-difference rows; all seven "
                        "failures are scalar); V4 "
                        "bounded/finite/no-depletion PASS. The rule "
                        "declared before the runs was that any failure holds "
                        "mp=28 out of 1.5. It is not being rewritten."),
                    "control_result": (
                        "THE SAME MACHINERY APPLIED TO WRF AGAINST ITSELF "
                        "ALSO FAILS V3 -- build B vs build A, identical "
                        "source, one optimization flag, 17 of 195 rows over "
                        "3x, worst ratio 808. A gate unmodified WRF cannot "
                        "pass against its own recompilation is not measuring "
                        "the port; past t ~ 2400 s this case is chaotic and "
                        "the scalar metrics compare two samples rather than "
                        "two answers. V3 as written was mis-specified."),
                    "diagnostic_results": {
                        "aerosol_budget_agreement_2h_relative": 1.530e-04,
                        "aerosol_budget_agreement_2h_relative_nifa": 2.451e-04,
                        "dual_run_byte_identical": True,
                        "dual_run_frames_byte_compared": {
                            "long_run": 13, "short_window": 11},
                        "long_run_median_mp28_over_mp8_disagreement": 0.691,
                        "long_run_median_scalar_rows_only": 0.628,
                        "short_window_v3_pass": True,
                        "short_window_v3_worst_ratio": 2.195,
                        "short_window_w_rms_5_steps_mp08": 1.800e-02,
                        "short_window_w_rms_5_steps_mp28": 1.797e-02,
                        "short_window_v3_second_worst_ratio": 1.501,
                        "t0_field_rms_difference_exact_zero_fields": 10,
                        "t0_field_rms_difference_max": 2.176e-08,
                    },
                    "short_window_note": (
                        "ADDED AFTER the long runs and labelled as such in "
                        "the document: both models restarted from the SAME "
                        "mature WRF state at t = 1800 s and run 50 steps, "
                        "which is the only regime in which a matched "
                        "trajectory can mean what it says. The statistic "
                        "(M8) and the condition (V3) are the pre-declared "
                        "ones; only the window is new."),
                    "what_it_establishes": (
                        "mp=28's per-step disagreement with unmodified WRF "
                        "is the disagreement mp=8 already has -- 1.797e-02 "
                        "vs 1.800e-02 RMS in w after five steps from a "
                        "mature state -- so the aerosol-aware scheme adds no "
                        "error of its own that this test can see; nwfa and "
                        "nifa are the best-agreeing 3-D fields measured; and "
                        "the domain aerosol budget matches WRF's to "
                        "1.530e-04 over two hours with no depletion trend."),
                    "what_it_does_not_establish": (
                        "nothing about a real-data or nested forecast, "
                        "nothing about aerosol crossing a lateral boundary "
                        "(there is none), nothing past t ~ 2400 s where the "
                        "trajectories have decorrelated, nothing about "
                        "radiative feedback, PBL interaction, heterogeneous "
                        "surface emission or sheared convection (all physics "
                        "but microphysics is off), and nothing about "
                        "correctness against observations. One case, one "
                        "resolution, one sounding, one bubble."),
                },
                # Named so the "UNVERIFIED" warning cannot be read as "never
                # integrated": it HAS been, against itself, and the row says
                # which gate did it and what that gate can and cannot see.
                "self_forecast_gate": {
                    "test": (
                        "tests/test_mp28_forecast_smoke.py::"
                        "test_g4_multistep_specified_bc_forecast_is_finite"
                        "_and_bounded"),
                    "steps": 150,
                    "timestep_s": 12.0,
                    "checks": (
                        "every prognostic and every radiation-facing "
                        "effective radius finite each step, WRF's terminal "
                        "apply bounds holding in the microphysics-updated "
                        "interior, spec-zone ring bit-restored"),
                    "does_not_check": (
                        "agreement with WRF. There is no reference "
                        "trajectory to compare against, so a finite, bounded "
                        "and wrong forecast passes it."),
                    "longest_integration": {
                        "test": (
                            "tests/test_mp28_forecast_smoke.py::"
                            "test_g4_a_two_hour_forecast_stays_finite"
                            "_bounded_and_ring_clean"),
                        "steps": 600,
                        "seconds": 7200.0,
                        "measured": (
                            "0 non-finite values, 0 bound violations and 0 "
                            "spec-zone ring violations across all 600 "
                            "microphysics calls; peak |w| 56.10 m/s; "
                            "domain-total RAINNC 1.9055 mm; final "
                            "domain-interior mean nwfa 1.3395e+07 kg^-1 "
                            "against a floor of 1.110e+07 and nifa "
                            "5.9041e+03 against a floor of 5.000e+03 -- i.e. "
                            "entirely inflow air after 2.6 domain "
                            "ventilation times, exactly as the lateral-"
                            "boundary deviation below predicts"),
                    },
                },
                "test": (
                    "tests/test_thompson_aerosol_adapter.py::"
                    "test_g3_end_to_end_against_all_nineteen_oracle_fixtures"),
            },
            # WHAT AN mp=28 RUN ACTUALLY STARTS FROM.  Published as structure
            # rather than prose, and DERIVED rather than transcribed:
            # tests/test_physics_registry.py scans gpuwm/ for callers of
            # microphysics_init and fails if this row disagrees with the scan
            # in EITHER direction.
            #
            # THIS ROW FLIPPED ON 2026-08-01.  For four waves
            # ``production_call_site`` was null and the option carried a "NO
            # AEROSOL INITIALISATION" warning: microphysics_init implemented
            # WRF's fill, was oracle-gated against it, and nothing called it,
            # so every mp=28 forecast integrated from nwfa = nifa = 0 under
            # WRF's floors.  woof/core/physics.py::initialize_physics now
            # calls it once per domain, exactly where WRF's phy_init calls
            # mp_init.  The measured numbers below did NOT change -- they are
            # the same two forecasts -- but what they mean did: they were the
            # measured COST of the missing call and they are now the measured
            # VALUE of the profile, i.e. how much of an mp=28 forecast the
            # CCN/IN loading owns.
            "aerosol_initialisation": {
                "wrf_source": (
                    "phys/module_mp_thompson.F:493-558 fills a synthetic "
                    "CCN/IN profile at domain construction whenever "
                    "MAXVAL(nwfa) / MAXVAL(nifa) come in below eps -- two "
                    "independent tests, at :493 and :530 -- and :509-510 "
                    "derives the surface emission nwfa2d from the filled "
                    "lowest level. Nothing inside mp_gt_driver ever refills "
                    "it."),
                "wrf_call_site": (
                    "phys/module_physics_init.F::microphysics_init, once per "
                    "domain, before the first step"),
                "gpuwm_implementation": (
                    "woof/core/microphysics.py::microphysics_init"),
                "gpuwm_implementation_evidence": (
                    "tests/test_mp28_runnable.py::"
                    "test_microphysics_init_fills_wrfs_synthetic_ccn_profile"
                    "_for_mp28 -- the fill itself is measured against WRF and "
                    "is NOT what is missing"),
                "production_call_site": (
                    "woof/core/physics.py::initialize_physics"),
                "production_call_site_note": (
                    "once per domain, unconditionally, at the seam WRF uses. "
                    "woof reaches it through microphysics_cold_start, which "
                    "returns an empty receipt for every scheme without a "
                    "domain-construction step. Presence-gated exactly as WRF "
                    "is: the fill runs only where the domain-wide MAXVAL "
                    "tests open, so a nest that inherited its parent's "
                    "aerosol and a restart that is about to be overwritten by "
                    "a checkpoint are both left alone. WRF: "
                    "phys/module_physics_init.F:1635 calls mp_init from "
                    "phy_init, phys/module_physics_init.F:4522-4538 is the "
                    "CASE (THOMPSONAERO) arm that calls thompson_init, and "
                    "the presence tests are "
                    "phys/module_mp_thompson.F:490-493 (CCN) and "
                    "phys/module_mp_thompson.F:528-531 (IN)."),
                "shipped_profile": (
                    "WRF's synthetic thompson_init CCN/IN profile, installed "
                    "at domain construction: on the aero-init-profile "
                    "fixture's grid nwfa runs 1.478987e+08 kg^-1 at the "
                    "lowest level to 5.000000e+07 aloft and nifa 1.254902e+06 "
                    "to 5.000002e+05, with nwfa2d derived from the filled "
                    "surface value at :509-510"),
                "shipped_consequence": (
                    "a freshly initialised mp=28 domain starts strictly ABOVE "
                    "the terminal apply's clamps "
                    "(phys/module_mp_thompson.F:3972-4021, nwfa >= 11.1e6 and "
                    "nifa >= 5.0e3 per m3) rather than pinned at them, so CCN "
                    "activation sees a continental aerosol loading that "
                    "decays with height instead of a maritime-clean floor "
                    "everywhere. The installed-state pin below asserts the "
                    "floors of WRF's own profile constants (naCCN1 = 50.0e6 "
                    "at phys/module_mp_thompson.F:96-97, naIN1 = 0.5e6 at "
                    "phys/module_mp_thompson.F:94-95) -- 4.5x and 100x above "
                    "the clamp floors, so a run that started at zero and got "
                    "clamped cannot satisfy it."),
                "operator_workaround": (
                    "none needed: the call is wired. An embedder that builds "
                    "a DomainState WITHOUT going through initialize_physics "
                    "gets no profile, and woof.core.microphysics."
                    "microphysics_init is the entry point to call once per "
                    "domain before the first step; it is presence-gated, so "
                    "calling it twice is a no-op, and it returns an empty "
                    "receipt for every scheme other than 28. Do NOT call it "
                    "per step: nothing in mp_gt_driver refills the profile, "
                    "and a per-step call would overwrite an advected, "
                    "activated and scavenged aerosol field with the synthetic "
                    "one while leaving every bound intact."),
                "measured_forecast_sensitivity": {
                    "test": (
                        "tests/test_mp28_forecast_smoke.py::"
                        "test_the_aerosol_profile_changes_the_forecast"
                        "_measurably"),
                    "steps": 150,
                    "timestep_s": 12.0,
                    "domain": (
                        "28 x 16 x 24 at dx = 2 km, specified BC, warm "
                        "bubble, two runs identical but for the init call"),
                    "initial_mean_nwfa_per_kg_with_profile": 6.6532e7,
                    "initial_mean_nwfa_per_kg_without_profile": 0.0,
                    "final_interior_nwfa_per_kg_with_profile": 2.1737e7,
                    "final_interior_nwfa_per_kg_without_profile": 4.2877e6,
                    "peak_nc_per_kg_with_profile": 1.5980e8,
                    "peak_nc_per_kg_without_profile": 2.9451e7,
                    "droplet_ratio_with_over_without": 5.43,
                    "domain_total_rainnc_mm_with_profile": 1.957357,
                    "domain_total_rainnc_mm_without_profile": 3.207102,
                    "domain_total_rainnc_relative_excess": 0.6385,
                    "peak_rainnc_mm_with_profile": 0.794230,
                    "peak_rainnc_mm_without_profile": 1.029196,
                    "reading": (
                        "the LEFT column is what a run does today; the RIGHT "
                        "column is the counterfactual with the profile "
                        "removed. Removing WRF's CCN/IN loading gives 5.4x "
                        "fewer cloud droplets and 63.8% MORE domain-total "
                        "surface rain over 30 minutes. Until 2026-08-01 the "
                        "right column WAS the shipped behaviour and this was "
                        "published as the port's largest measured error; it "
                        "is now the measured sensitivity of an mp=28 forecast "
                        "to its aerosol initial condition, which is also the "
                        "magnitude the lateral-boundary deviation below "
                        "converges to after L/U."),
                    "note": (
                        "a SNAPSHOT on this tree, not a physics pin: any "
                        "legitimate mp=28 numerics change moves these digits. "
                        "What is published is the sign and the order of "
                        "magnitude, and tests/test_physics_registry.py::"
                        "test_the_published_aerosol_initialisation_cost_is"
                        "_still_the_measured_one re-runs both forecasts to "
                        "check them."),
                },
                "call_site_pin": (
                    "tests/test_mp28_forecast_smoke.py::"
                    "test_microphysics_init_has_a_production_call_site"),
                "installed_state_pin": (
                    "tests/test_mp28_forecast_smoke.py::"
                    "test_a_freshly_initialised_mp28_domain_carries_the"
                    "_profile_not_zero"),
            },
            "activation_bin_edge_policy": (
                "activ_ncloud selects a NEAREST 10 K temperature bin and "
                "truncates idx_d/idx_c/idx_n with INT(), so nc is a STEP "
                "function of state. Within one ulp of a bin edge an FP32 GPU "
                "port and the Fortran reference can select different bins and "
                "differ by tens of percent in nc while every mass field "
                "agrees. Fixture states are chosen away from bin edges and "
                "the behaviour is documented here rather than absorbed into a "
                "loose tolerance."),
        },
        "implemented": True,
        "label": "Thompson aerosol-aware / MP28",
        "maturity": "implemented-unverified",
        # Exactly thompson-mp8's pins.  aer_init_opt and aer_fire_emit_opt are
        # DERIVED in WRF (Registry.EM_COMMON:2656/:2658), not namelist knobs,
        # so they are not woof settings and are not pinned here; wif_input_opt
        # and the rest of the WIF family are published as implemented=false
        # roadmap rows, and an implemented option may not pin one of those.
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "selectors": {"mp_physics": 28},
        "warnings": [
            "No matched REAL-DATA or NESTED WRF trajectory and no decay table "
            "is recorded here. The WIF lateral-boundary carrier now exists, "
            "so missing matched inputs and a retained comparison are the "
            "evidence gap. There IS a "
            "matched IDEALIZED trajectory -- a doubly periodic "
            "single-domain warm-bubble forecast against unmodified WRF "
            "v4.6.1, recorded in extensions.column_oracle_evidence."
            "forecast_trajectory_comparison -- and its pre-declared gate "
            "FAILED on one of its four conditions, V3. Read that entry "
            "before relying on this scheme: it also records that WRF fails "
            "the same condition against its own recompilation, which is why "
            "the failure is published rather than acted on. The single-call "
            "evidence is 22 committed WRF v4.6.1 column fixtures plus "
            "per-kernel and device-helper oracles. mp_physics=28 has also "
            "been integrated multi-step against ITSELF: "
            "tests/test_mp28_forecast_smoke.py::"
            "test_g4_multistep_specified_bc_forecast_is_finite_and_bounded "
            "runs 150 steps x 12 s on a specified-BC convective domain and "
            "checks that every prognostic and every radiation-facing "
            "effective radius stays finite, that WRF's own terminal bounds "
            "hold, and that the spec-zone ring is bit-restored; "
            "test_g4_a_two_hour_forecast_stays_finite_bounded_and_ring_clean "
            "carries the same domain to 600 steps (7200 s) with 0 non-finite "
            "values, 0 bound violations and 0 ring violations. Finite and "
            "bounded is NOT correct: a scheme with a systematically wrong "
            "activation rate passes every one of those checks for two hours. "
            "The bounds are WRF's, but they are clamps, not answers.",
            "THE COLUMN EVIDENCE IS CLEAN BUT FOR ONE DECLARED DIVERGENCE; "
            "THE FORECAST EVIDENCE IS NOT. "
            "Driven end to end through the shipped "
            "adapter, 22 fixtures x 23 quantities, at a flat 2.0e-6 relative "
            "/ 2.0e-4 dB gate with nothing held out: 21 of 22 clear every "
            "quantity (18 of the 19 spec'd aero-* fixtures, plus "
            "wp08-freeze, wp08-melt and wp08-nusweep), with no allowance "
            "anywhere, on an RTX 4090 and an RTX 5070 Ti identically. The "
            "22nd, aero-cold-overlap, the one fixture carrying rain and "
            "graupel together, misses by the declared rain-graupel "
            "divergence alone: WRF v4.6.1 reads its rain-graupel collision "
            "tables out of bounds when the scheme is not hail aware "
            "(dimNRHG = 1, idx_bg = 5; phys/module_mp_thompson.F:465, "
            ":607-615, :2527-2545) and WOOF reads the one slab the tables "
            "hold, so it reads qr 5.070e-04, nr 5.066e-04, qi 1.618e-04, qg "
            "8.776e-05, effi 5.394e-05, ni 1.464e-05, effs 6.574e-06 and "
            "1.585e-03 dB of reflectivity from WRF; with WRF's read emulated "
            "in a measurement copy of the tree it clears all 23 quantities. "
            "The 2.8.6 accumulator "
            "rework closed the last four: WRF's mp_thompson never writes a "
            "hydrometeor during the call -- every process adds to qcten, "
            "qrten, nrten, qiten or niten and the terminal apply "
            "(phys/module_mp_thompson.F:3972-4053) rounds each species once, "
            "with the size bounds after the :3943-3966 phase cleanup -- "
            "while the port applied each stage in place and so rounded a "
            "species once per stage. mp=28 now carries the five tendencies "
            "and applies them once, and takes the cold network's "
            "rain-conservation ratio as REAL as :1615 declares it. "
            "aero-cloud-freeze-nc qc 4.926e-06 -> bit-exact; "
            "aero-cold-overlap qc / nc / effc at level 4 (one float32 ulp "
            "of cloud flipping the qc1d <= R1 test at :4007) and nr "
            "1.261e-04 / qr 4.443e-05 at level 6 -> bit-exact in 2.8.6, "
            "before 2.8.8 moved rain collecting graupel onto the one slab "
            "WRF's tables hold (the declared divergence above, which is "
            "what the fixture misses by now); "
            "wp08-nusweep qr 4.642e-06 -> 5.532e-07; "
            "aero-reduces-to-classic level 6 bit-exact in qr and nr, which "
            "retired the port's last named allowance (32 ulps of the entry "
            "value). The frozen mp=8 kernel file is unchanged. A clean "
            "single-call deck is not a forecast validation: the maturity "
            "stays implemented-unverified. BEFORE THAT, TWO REAL PORT "
            "FIXES, both in mp=28-owned kernels. WP-13a restored WRF's "
            "LEVEL-WISE sedimentation density: WRF forms the "
            "working rain mass and number twice -- "
            "phys/module_mp_thompson.F:3237-3238 from the :3193 TAU+1 "
            "density at every L_qr level, and :3568/:3570 from the :3490 "
            "post-condensation density but ONLY inside the :3501-3502 gate "
            "-- and woof/core/kernels/thompson_aerosol_sat.cu was exporting "
            "the post-condensation density unconditionally. WP-13b pinned "
            "the source-network apply against contraction: WRF's terminal "
            "apply at phys/module_mp_thompson.F:3973-4023 "
            "is q1d(k) = q1d(k) + qten(k)*DT and the gfortran -O2 oracle has "
            "no FMA, so qten*DT is rounded to REAL(4) before the add, while "
            "nvrtc was fusing it in thompson_aerosol_cold.cu and "
            "thompson_aerosol_warm.cu. Together they "
            "took aero-drop-evap (rainnc 5.165e-04, qr 3.533e-05, nr "
            "2.258e-05; WP-13a alone) and aero-ice-demott-idxin (rainnc "
            "1.279e-04, qr 2.894e-05, nr 4.594e-06; both changes, RAINNC and "
            "sr bitwise 0 after WP-13b) to CLEAN, took "
            "aero-cloud-freeze-nc's qr 2.800e-05 / nr 1.797e-05 / rainnc "
            "1.162e-05 rows to 8.973e-08 / 2.594e-07 / 0.000e+00, deleted "
            "qr from the carve-out (7.813e-05 -> 1.788e-07, inside the flat "
            "gate), tightened the surviving nr bound 1.0e-04 -> 1.0e-05 and "
            "RETIRED the reflectivity carve-out (5.283e-04 dB -> 3.242e-05 "
            "dB). One number went the OTHER way and is published as "
            "measured: aero-cold-overlap qr 3.667e-05 -> 4.443e-05, WP-13b's "
            "cost, bisected to the cold network's qr apply and KEPT because "
            "reverting that one line also loses all four of "
            "aero-ice-demott-idxin's improvements, two of which are bitwise; "
            "in ulps of the entry value the move is 1.477 -> 1.789. "
            "AND WHAT MOVED BEFORE THOSE, WHICH WAS NOT "
            "A PHYSICS FIX: aero-ice-koop, published by four waves of this "
            "registry as the port's largest genuine physics gap at qi "
            "1.612e-03 / ni 1.764e-03 / effi 5.093e-05, now measures "
            "1.534e-07 / 3.396e-07 / 1.886e-07 and is CLEAN -- and NO KERNEL "
            "CHANGED. The cause was this port's own oracle harness: "
            "tools/thompson_wrf461_oracle/run_column_aero.F90 built the Exner "
            "function with rd_over_cp = 287.0/1004.0, while WRF's own rcp is "
            "r_d/cp with r_d=287. and cp=7.*r_d/2.=1004.5 (declared in "
            "share/module_model_constants.F at lines 19, 20 and 31) -- "
            "exactly 2/7, and "
            "4774 float32 ulps from what the harness used. 47 of the deck's "
            "528 entry levels (first published as 40; re-pinned 2026-08-03, "
            "owner-ratified -- the count is the host libm's, and "
            "mp28-column-evidence.md section 3.4 carries the correction "
            "note) then had no exact float32 theta, the adapter "
            "perturbed the entry pressure by up to 15 ulps to recover the "
            "recorded temperature, and that perturbed pressure drove "
            "different microphysics on 7 fixtures including this one. The "
            "deck was regenerated with WRF's own constant and the residual "
            "went with it; the same regeneration made aero-cold-overlap "
            "WORSE, which is why it can be trusted. See "
            "extensions.column_oracle_evidence for the full "
            "table, the allowance list and the two counts, and "
            "docs/public/wrf-comparison/mp28-column-evidence.md section 3.4 for "
            "the re-derivation from the committed fixtures.",
            "AEROSOL INPUT LIMITS: native met_em preparation accepts a complete "
            "analyzed QNWFA/QNIFA pair and the monthly WIF climatology reader "
            "is available. Auto uses the analyzed pair when present; an explicit "
            "source selector is retained and its receipt names the input used. "
            "The imported use_aero_icbc=true / wif_input_opt=1 / "
            "num_wif_levels=30 route selects the monthly dataset. There is no "
            "black-carbon (nbca) species, generic GOCART reader or qna_update "
            "auxiliary stream. SYNTHETIC INITIALISATION: "
            "WRF's synthetic fallback is thompson_init's "
            "SYNTHETIC CCN/IN profile (phys/module_mp_thompson.F:493-558), "
            "woof implements it in "
            "woof/core/microphysics.py::microphysics_init, measures it "
            "against WRF, and -- since 2026-08-01 -- CALLS it, once per "
            "domain, from woof/core/physics.py::initialize_physics, at the "
            "seam WRF calls mp_init from phy_init. A freshly initialised "
            "mp=28 domain selecting this fallback starts on WRF's decaying continental "
            "profile, strictly above the terminal apply's clamps "
            "(phys/module_mp_thompson.F:3972-4021), not pinned at them. A "
            "HISTORICAL SYNTHETIC-PROFILE SENSITIVITY run measured over "
            "150 steps x 12 s on a "
            "28 x 16 x 24 2 km specified-BC convective domain against an "
            "otherwise identical run with the profile removed: initial mean "
            "nwfa 6.6532e+07 vs 0.0 kg^-1, peak nc 1.5980e+08 vs 2.9451e+07 "
            "kg^-1 (5.4x fewer droplets without it), domain-total RAINNC "
            "1.957357 vs 3.207102 mm -- the aerosol-free run rains 63.8% "
            "MORE. This describes the retained synthetic-profile experiment, "
            "not a current WIF-initialized run or a score against "
            "observations. See extensions.aerosol_initialisation.",
            "SYNTHETIC-PROFILE ADMISSION DIFFERENCE: WRF's initializer "
            "refuses this fallback configuration. dyn_em/module_initialize_real.F:"
            "2734-2736 calls wrf_error_fatal('wif_input_opt=0 but "
            "mp_physics=28'), so real.exe will not build a wrfinput for the "
            "synthetic-profile case at all. The PHYSICS is WRF's; the "
            "admission decision is not.",
            "AEROSOL LATERAL BOUNDARIES: the current WIF-climatology route "
            "couples nwfa/nifa through the specified boundary carrier. A "
            "specified domain without its required aerosol dataset is "
            "refused before step 0. The formerly measured zero-inflow "
            "depletion describes the historical uncoupled route, not the "
            "current WIF route. Runtime coupling is not evidence of a "
            "matched real-data or nested WRF forecast.",
            "MYNN aerosol-number mixing is optional. bl_mynn_mixscalars=0 "
            "is the default (WRF v4.6.1 Registry.EM_COMMON:2479). Setting "
            "it to 1 mixes nc/ni/nwfa/nifa using the WRF qn solves under "
            "bl_pbl_physics=5 with mp_physics=28 (then bldt=0 is required); "
            "under a microphysics with no number species (mp_physics 0, 1, "
            "6) the key is WRF's no-op and the key-0 path runs; under a "
            "scheme WRF mixes only in part (8, 9, 10, 16, 18, 50) it is "
            "refused by name (config.MYNN_QN_FLAG_SPECIES). This is "
            "component code verification, not validation against "
            "observations.",
            "PBL number mixing is selectable for mp_physics=28: "
            "scalar_pblmix=1 applies WRF post-PBL diffusion through the "
            "selected scheme's exch_h (YSU, MYJ, MYNN, UW, Shin-Hong) "
            "to nc/ni/nwfa/nifa and mixes nothing with no PBL scheme or "
            "under SASE; bl_mynn_mixscalars=1 instead selects "
            "MYNN scalar plume transport. Both default to 0 and, when they "
            "mix, require bldt=0. WRF disables the former when the latter "
            "is active, so selecting both is refused.",
            "MIXED NESTING uses the registered transition policy. Entry "
            "into mp_physics=28 from a non-aerosol parent requires a declared "
            "aerosol source; same-scheme nesting carries the parent state. "
            "See the transition registry for admitted edges and provenance. "
            "Runtime admission is not a matched WRF trajectory comparison.",
            "DELIBERATE THERMODYNAMIC DIVERGENCE FROM mp_physics=8, and it is "
            "not a defect on either side. mp=28's RSLF/RSIF saturation Horner "
            "chains are contraction-pinned while mp=8's stay FMA-contracted, "
            "so the two schemes' saturation vapour pressures differ by one "
            "ulp. That matters because phys/module_mp_thompson.F:3401 opens the "
            "whole condensation/CCN-activation block on ssatw > 1.E-15 (:185) "
            "-- one ulp flips a branch. mp=28 matches WRF's own gfortran "
            "-O2 arithmetic; mp=8 retains its FMA-contracted saturation "
            "chains. Its kernels changed after the 2026-07-28 matched "
            "run, so that run describes an earlier build. The two "
            "schemes are deliberately not bit-identical.",
            "Reachable through the registered aerosol-aware template and "
            "per-domain microphysics overrides where the route admits "
            "them. Read each route's declared templates and constraints; "
            "reachability is not forecast verification evidence.",
            "Launch must byte-validate CCN_ACTIVATE.BIN (35,288 bytes, "
            "sha256 f2b8d391...) plus the four classic Thompson tables. The "
            "activation table IS distributed with woof as of 2026-08-01 -- "
            "WRF v4.6.1's own run/CCN_ACTIVATE.BIN, bit for bit -- so a "
            "default install has it; WOOF_THOMPSON_CCN_ACTIVATE and "
            "WOOF_THOMPSON_TABLE_ROOT still redirect a run to another copy. "
            "Absence is fatal, never defaulted, and a byte-different table is "
            "refused: a different parcel-model table would silently be a "
            "different activation scheme. The consequence for EVIDENCE, not "
            "just for launch: every device gate for the scheme, including all "
            "22 column fixtures, skips by name if the file is ever missing. "
            "The skip names the one file rather than swallowing a load "
            "failure, so it can never be mistaken for a pass.",
        ],
    }


def _unimplemented_specs(known: set[str]) -> dict[str, dict]:
    """Knobs the survey found in WRF that no GPUWM component honors."""
    lanes = []
    with JOURNAL.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "result":
                lanes.append(record["result"])

    unimplemented: dict[str, dict] = {}
    for lane in lanes:
        for knob in lane["knobs"]:
            name = knob["name"]
            if name in known:
                continue
            if knob.get("gpuwm_implemented"):
                continue
            reason = clean(knob.get("not_implemented_reason", ""))
            if not knob.get("wrf_namelist_knob"):
                reason = ("Not a WRF v4.6.1 namelist option. "
                          + reason).strip()
            if not reason:
                reason = "Not implemented by any GPUWM runtime component."
            spec = {
                "type": (knob.get("proposed_spec") or {}).get("type")
                or WRF_TYPE.get((knob.get("wrf_type") or "").lower(),
                                "integer"),
                "implemented": False,
                "unimplemented_reason": reason}
            prior = unimplemented.get(name)
            if (prior is None
                    or len(prior["unimplemented_reason"]) < len(reason)):
                unimplemented[name] = spec
    return unimplemented


def _wrf_compatibility_authority() -> dict:
    """JSON form of the normalized, fully cited WRF compatibility matrix."""

    def citation(value) -> dict[str, str]:
        return {
            "source": value.anchor,
            "law": value.law,
        }

    def cell(**updates):
        values = {
            "mp_physics": 1,
            "bl_pbl_physics": 0,
            "sf_sfclay_physics": 1,
            "sf_surface_physics": 2,
            "radiation": "off",
            "cu_physics": 0,
        }
        values.update(updates)
        return compatibility_cell(**values)

    counts: dict[str, int] = {}
    for matrix_cell in iter_compatibility_matrix():
        verdict = matrix_cell.verdict.value
        counts[verdict] = counts.get(verdict, 0) + 1

    return {
        "wrf_version": WRF_VERSION,
        "wrf_commit": WRF_COMMIT,
        "implementation": "woof.wrf461_compatibility",
        "cell_count": MATRIX_CELL_COUNT,
        "dimensions": {
            "mp_physics": list(MP_OPTIONS),
            "bl_pbl_physics": list(PBL_OPTIONS),
            "sf_sfclay_physics": list(SURFACE_LAYER_OPTIONS),
            "sf_surface_physics": list(LAND_SURFACE_OPTIONS),
            "radiation": list(RADIATION_OPTIONS),
            "cu_physics": list(CUMULUS_OPTIONS),
        },
        "independent_axis_citations": {
            "mp_physics": {
                str(value): citation(cell(mp_physics=value).citations[0])
                for value in MP_OPTIONS
            },
            "sf_surface_physics": {
                str(value): citation(
                    cell(sf_surface_physics=value).citations[2])
                for value in LAND_SURFACE_OPTIONS
            },
            "radiation": {
                value: citation(cell(radiation=value).citations[3])
                for value in RADIATION_OPTIONS
            },
            "cu_physics": {
                str(value): citation(cell(cu_physics=value).citations[4])
                for value in CUMULUS_OPTIONS
            },
            "soil_layer_reconfiguration": citation(
                cell().citations[5]),
        },
        "pbl_surface_layer_cells": [
            {
                "bl_pbl_physics": pbl,
                "sf_sfclay_physics": surface,
                "verdict": verdict.value,
                "citation": citation(source),
            }
            for (pbl, surface), (verdict, source)
            in sorted(PBL_SURFACE_LAYER_AUTHORITY.items())
        ],
        "precedence": [
            "a fatal PBL/surface-layer cell is fatal for every other axis",
            "otherwise analytic radiation is not expressible in WRF v4.6.1",
            "otherwise sf_surface_physics=0 is legal with WRF silently "
            "setting num_soil_layers=5",
            "all remaining cells are legal",
        ],
        "verdict_counts": dict(sorted(counts.items())),
        # The count is computed, not typed: a typed "2,400" survived two
        # axis widenings (mp=28, bl_pbl=11) as stale prose.
        "test": (
            f"tests/test_wrf461_compatibility.py sweeps all "
            f"{MATRIX_CELL_COUNT:,} cells and "
            "requires every cell to carry all six WRF citations"),
    }


#: Path A of the owner-ratified vocabulary decision (D-1/D-2).  The old
#: spellings claimed validation the labels never carried: agreement with a
#: matched WRF run is agreement with another model, and "validated" is read
#: by everyone as skill against the atmosphere.  The new spellings say what
#: the evidence is.  Applied to maturity VALUES at every surface; template
#: ids migrate through an explicit alias table, with old inputs accepted
#: and preparation-receipt physics identities kept unchanged.
MATURITY_RENAMES = {
    "model-validated": "wrf-matched-run",
    "validation-candidate": "wrf-matched-run-candidate",
}

#: The conformance axis, lowest evidence first.  This tuple is the ONLY
#: ordering of maturity names in the repository: woof/physics_registry.py
#: derives MATURITY_RANK, both warning tiers and the composition ceiling
#: from the block this builds, so a rung cannot be reordered in one place
#: and not the other.
_MATURITY_RUNGS = (
    ("planned", "unimplemented-only",
     "Registered so the roadmap is readable. No GPUWM runtime component "
     "exists, implemented is false, and the option cannot be selected."),
    # 'port-in-progress' lived here while the WRF RRTM longwave row was
    # the one option using it.  That port landed (ra_lw_physics=1 +
    # ra_sw_physics=1, woof/core/rrtm_lw.py) and moved to
    # implemented-unverified, leaving the rung with no occupant.  What
    # forces the deletion is tests/test_evidence_axes.py's AC3 case
    # test_the_rung_set_equals_the_set_in_use ("no decorative rungs"),
    # which requires the rung set to equal the set of maturity values
    # actually in use.  NOT the loader: woof/physics_registry.py's D-22
    # check (_enforce_evidence_axes) enforces order-vs-rungs agreement
    # and value membership, and an unoccupied rung passes it.  Either
    # way the ladder is meant to describe states the registry is in, so
    # a future in-tree-but-unreachable port re-adds it here.
    ("implemented-unverified", "warn",
     "A GPUWM runtime component executes this option and no matched "
     "WOOF-versus-WRF forecast trajectory has been run with it. "
     "Selecting it warns and does not block."),
    ("experimental-runtime", "warn",
     "Executable, and carrying a documented runtime restriction or an "
     "unratified composition -- a table-bound runtime, or a nest edge "
     "between two microphysics schemes. Selecting it warns and does not "
     "block."),
    ("supported", "nonwarning",
     "Agreement with the WRF reference implementation is settled for this "
     "option by committed oracle parity or an equivalent comparison, and "
     "it carries no runtime restriction. Selecting it does not warn."),
    ("wrf-matched-run-candidate", "warn",
     "Executable and gated, with a ratified reference comparison, and "
     "deliberately not the default: the next candidate for a full matched "
     "run. Selecting it warns and does not block."),
    ("wrf-matched-run", "nonwarning",
     "A historical matched-run label. Read verification_scope and its "
     "note: the current code or exact suite may not be covered, and "
     "some template labels are composition exemptions. "
     "This is code verification against WRF. It is not validation "
     "against observations."),
)

#: The independent-science axis (D-26: options only).  ``none`` is the
#: accurate default and the only value this pass assigns: the value set above
#: it is the ratified catalogue's, and an option is promoted off ``none``
#: only by an entry in ``scientific_evidence_catalogue``.  Assigning a
#: category here without that entry would be exactly the unbacked claim the
#: two-axis split exists to prevent.
_SCIENTIFIC_ENUM = (
    ("none",
     "No independent scientific evidence is claimed for this option. Its "
     "evidence is conformance with the WRF reference implementation, which "
     "lives on the other axis."),
    ("idealized-analytic",
     "Compared against a closed-form or analytically constrained solution "
     "of an idealized problem."),
    ("converged-numerical-reference",
     "Compared against a converged high-resolution numerical reference "
     "solution of an idealized problem."),
    ("conservation-gated",
     "Carries a committed conservation residual gate over an idealized "
     "problem."),
    ("obs-evaluated",
     "Evaluated against observations. No option in this registry carries "
     "this value; the rung exists so the absence is visible rather than "
     "unrepresentable."),
)


def _rename_maturities(registry: dict) -> None:
    """Rewrite every maturity VALUE at every surface under Path A."""

    for component in registry.get("components", {}).values():
        for option in component.get("options", {}).values():
            if option.get("maturity") in MATURITY_RENAMES:
                option["maturity"] = MATURITY_RENAMES[option["maturity"]]
    for template in registry.get("templates", {}).values():
        if template.get("maturity") in MATURITY_RENAMES:
            template["maturity"] = MATURITY_RENAMES[template["maturity"]]
    for transition in registry.get("transitions", {}).values():
        for rule in transition.get("cross_options", []):
            if rule.get("maturity") in MATURITY_RENAMES:
                rule["maturity"] = MATURITY_RENAMES[rule["maturity"]]
        same = transition.get("same_option")
        if isinstance(same, dict) and same.get("maturity") in MATURITY_RENAMES:
            same["maturity"] = MATURITY_RENAMES[same["maturity"]]


def _evidence_architecture(registry: dict) -> None:
    """Author the two axes, the ladder and the composition rule."""

    rungs = {
        name: {
            "rank": rank,
            "warning_tier": tier,
            "definition": definition,
        }
        for rank, (name, tier, definition) in enumerate(_MATURITY_RUNGS)
    }
    order = [name for name, _tier, _definition in _MATURITY_RUNGS]

    registry["maturity_ladder"] = {
        "axis": "conformance",
        "order": order,
        "rungs": rungs,
        "aliases": dict(MATURITY_RENAMES),
        "meaning": (
            "How far agreement with the WRF reference implementation has "
            "been demonstrated for this component, template or nest edge. "
            "Every rung is a statement about agreement with another model "
            "and none of them is a statement about skill against "
            "observations."),
        "composition_rule": _composition_rule(),
    }
    registry["evidence_axes"] = {
        "conformance_implies_scientific_validation": False,
        "conformance_implies_scientific_validation_meaning": (
            "Agreement with WRF is agreement with a model. No rung of the "
            "conformance ladder implies any value on the scientific axis, "
            "and the two are reported separately everywhere."),
        "maturity": {
            "axis": "conformance",
            "ladder": "maturity_ladder",
            "rungs": rungs,
            "surfaces": [
                "components.<component_id>.options.<option_id>.maturity",
                "templates.<template_id>.maturity",
                "transitions.<transition_id>.cross_options[].maturity",
            ],
            "absent_surfaces": [
                {
                    "surface": (
                        "transitions.<transition_id>.cross_options[] whose "
                        "status is 'ratified'"),
                    "owner_decision_id": "D-26",
                    "contract": (
                        "A ratified nest edge carries no maturity key. The "
                        "absence is the contract, not an omission: the edge "
                        "was ratified as a whole against its per-species "
                        "receipt, so there is no separate conformance rung "
                        "to report for it. An unratified edge carries "
                        "'experimental-runtime' and warns."),
                },
            ],
        },
        "scientific": {
            "axis": "independent-scientific-evidence",
            "default": "none",
            "enum": {
                name: {"definition": definition}
                for name, definition in _SCIENTIFIC_ENUM
            },
            "surfaces": [
                "components.<component_id>.options.<option_id>"
                ".scientific_evidence",
            ],
            "absent_surfaces": [
                {
                    "surface": "templates.<template_id>",
                    "owner_decision_id": "D-26",
                    "contract": (
                        "Templates carry no scientific_evidence. A template "
                        "is a composition of options and inherits no "
                        "independent evidence by being composed; read the "
                        "axis on the options it selects."),
                },
                {
                    "surface": "transitions.<transition_id>.cross_options[]",
                    "owner_decision_id": "D-26",
                    "contract": (
                        "A nest edge carries no scientific_evidence for the "
                        "same reason."),
                },
            ],
            "catalogue_contract": (
                "An option is promoted off 'none' only by an entry in "
                "scientific_evidence_catalogue naming the artifact and "
                "quoting its category basis. Every option in this registry "
                "reads 'none': their evidence is conformance, which lives "
                "on the other axis."),
        },
    }

    for component in registry.get("components", {}).values():
        for option in component.get("options", {}).values():
            option.setdefault("scientific_evidence", "none")

    tier_of = {name: tier for name, tier, _definition in _MATURITY_RUNGS}
    policy = registry["warning_policy"]
    policy["nonwarning_maturities"] = [
        name for name in order if tier_of[name] == "nonwarning"]
    policy["warn_maturities"] = [
        name for name in order if tier_of[name] == "warn"]
    policy["tier_authority"] = (
        "Both lists are computed from maturity_ladder.rungs[].warning_tier "
        "in ladder order. There is no second ordering of maturity names.")


def _composition_rule() -> dict:
    """The owner-ratified two-clause rule (D-16), axis A, as an invariant.

    Enforcement point is the registry document, not the loader.  A blocking
    load-time ERROR on clause C2 would take WSM6 and both NSSL-2 templates
    out of service, so the rule is enforced by
    ``tests/test_physics_registry_composition.py`` over the shipped
    document while the loader enforces only the two-axis membership that
    cannot take a working template out of service (D-22).
    """

    return {
        "id": "template-composition-ceiling-v1",
        "axis": "A-strict-min",
        "owner_decision_id": "D-16",
        "enforcement_point": "registry-document-invariant",
        "severity": "invariant-test",
        "enforcement_ref": "tests/test_physics_registry_composition.py",
        "clauses": {
            "C1": {
                "name": "trajectory pointer",
                "statement": (
                    "A template whose maturity is at or above "
                    "'wrf-matched-run-candidate' carries an evidence_pointer "
                    "that resolves to a matched-run manifest under "
                    "woof/authorities/matched_runs/."),
            },
            "C2": {
                "name": "composition ceiling",
                "statement": (
                    "A template's maturity rank does not exceed the lowest "
                    "maturity rank among the component options it selects. "
                    "A composed suite is only as conformant as its weakest "
                    "member."),
            },
        },
        "discharge": (
            "A clause is discharged for a template either by a resolvable "
            "evidence_pointer -- a whole-suite matched run outranks a "
            "component-wise minimum, because the suite itself was compared "
            "-- or by an entry in composition_exemptions naming the "
            "owner-decision id that granted it. Nothing is discharged "
            "silently."),
        "composition_exemptions": _composition_exemptions(),
    }


def _composition_exemptions() -> dict:
    """Every template the ratified axis flags without a matched-run pointer.

    Each entry names the decision that granted it and states, in the terms
    of the rule, what is missing.  These are not waivers of the finding;
    they are the finding, written down where the checker reads it.
    """

    unverified_land_pbl = (
        "Noah and YSU are 'implemented-unverified': neither has its own "
        "matched WOOF-versus-WRF forecast trajectory, so the strict-min "
        "ceiling for every suite selecting them is 'implemented-unverified'."
    )
    return {
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The Morrison reference campaign matched the whole suite "
                "against WRF, and no matched-run manifest for it has been "
                "assembled yet. " + unverified_land_pbl),
        },
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The default template selects the substitution radiation "
                "engine, and the matched run of record was produced with "
                "the exact legacy engine, so its manifest does not cover "
                "this template's tuple. " + unverified_land_pbl),
        },
        "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "A table-bound experimental runtime carried above its "
                "component floor. " + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-wrf-comparison-candidate-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "NSSL-2 has fused-process oracles and a ratified 500 m "
                "comparison, and no matched-run manifest. "
                + unverified_land_pbl),
        },
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-wrf-comparison-candidate-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The legacy-RRTMG sibling of the entry above, granted on "
                "the same basis."),
        },
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The observation battery's registered composition (lead "
                "ruling, obs-battery integration wave 2026-08-04): "
                "Thompson mp8 is wrf-matched-run and the legacy RRTMG "
                "engine is the certified WRF v4.6.1 port, but no receipt "
                "covers the composed suite yet -- the battery shakedown "
                "case's stock-WRF-paired t0/case receipt is the named "
                "payer. " + unverified_land_pbl),
        },
        "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1": {
            "owner_decision_id": "D-16",
            "clause": "C1+C2",
            "basis": (
                "The gray-zone sibling of the entry above -- the same "
                "composition with Shin-Hong 2015 in place of YSU -- "
                "granted on the same basis, with the same payer: this "
                "composition's first stock-WRF-paired t0/case receipt. "
                "Shin-Hong's own port is measured bitwise against the "
                "byte-frozen WRF v4.6.1 module on both halves (max ULP 0 "
                "on the float32 CPU authority; 0 ULP on the CUDA heat "
                "tendency), which is conformance evidence and not a "
                "matched forecast trajectory. Noah and Shin-Hong are "
                "'implemented-unverified': neither has its own matched "
                "WOOF-versus-WRF forecast trajectory, so the strict-min "
                "ceiling for this suite is 'implemented-unverified'."),
        },
        "wsm6-ysu-mm5-noah-no-radiation-v1": {
            "owner_decision_id": "D-16",
            "clause": "C2",
            "basis": (
                "The reference no-radiation suite is 'supported' on its own "
                "oracle parity. " + unverified_land_pbl),
        },
    }


MP9_OPTION_ID = "milbrandt2mom-mp9"


def _composition_ceiling(registry: dict, components: dict) -> str:
    """Clause C2 as a function: the highest rung a composition may carry.

    A composed suite is only as conformant as its weakest member, so the
    ceiling is the lowest-ranked maturity among the option rows the
    composition selects.  Nothing here reads a declared TEMPLATE maturity:
    the rung order is :data:`_MATURITY_RUNGS`, the one ordering of maturity
    names in this repository, and the option rows are the ones being built,
    so a template minted at this value states what the tree holds and
    cannot outrank it.

    Called from inside :func:`build`, before :func:`_rename_maturities`
    runs, so the raw option values are mapped through
    :data:`MATURITY_RENAMES` here rather than read as written.
    """

    order = [name for name, _tier, _definition in _MATURITY_RUNGS]
    selected = []
    for component_id, option_id in sorted(components.items()):
        option = registry["components"][component_id]["options"][option_id]
        maturity = option.get("maturity")
        maturity = MATURITY_RENAMES.get(maturity, maturity)
        if maturity not in order:
            raise SystemExit(
                f"component {component_id}.{option_id} carries maturity "
                f"{maturity!r}, which is not a rung of the ladder "
                f"({order}): a composition ceiling cannot be computed from "
                "a value the ladder does not define, and a template minted "
                "from it would carry an unreadable rank")
        selected.append(maturity)
    return min(selected, key=order.index)


def _milbrandt2mom_mp9(registry: dict) -> None:
    """Register Milbrandt-Yau two-moment at the maturity its evidence earns.

    ``implemented: true`` -- the scheme runs on the device through
    ``woof/core/milbrandt2.py`` and ``woof/core/kernels/milbrandt2.cu``
    and is dispatched by ``woof/core/microphysics.py`` on
    ``mp_physics == 9``.

    ``maturity: implemented-unverified`` -- and no higher.  The column port
    is bitwise against WRF v4.6.1's own mp_milbrandt2mom_driver on 603
    oracle columns (tools/milbrandt2_wrf461_oracle,
    tests/test_milbrandt2_wrf461_exact_gpu.py), under the default build and
    the strict one, but no forecast-trajectory comparison against WRF
    exists, which is the same line TEMPO's row draws.

    ``reachability: component-override`` -- computed, not chosen.  No
    template selects mp=9 and ``DEFAULT_TEMPLATE_ID`` is untouched, so the
    only way in is a per-domain experiment override on the tree route,
    which already lists ``microphysics`` in
    ``allowed_component_overrides``.
    """
    options = registry["components"]["microphysics"]["options"]
    options[MP9_OPTION_ID] = {
        "asset_requirements": [],
        # The RTE+RRTMGP cloud-optics refusal is NOT written here.  It is
        # attached by _rte_rrtmgp_cloud_optics_constraints below, which
        # derives WHICH schemes need it from
        # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME rather than from a
        # list maintained by hand -- mp=9 was not the only scheme missing
        # a row, and the second one (P3) shipped into 1.9 undetected.
        "constraints": {
            "required_settings": {"moist": True},
        },
        "extensions": {
            "fixed_mode": (
                "two-moment in all six hydrometeors, separate graupel AND "
                "hail, continental CCN (CCNtype=2), Meyers+contact primary "
                "ice nucleation, non-spherical snow"),
            "column_oracle_evidence": {
                "authority": (
                    "WRF v4.6.1 d66e442 phys/module_mp_milbrandt2mom.F, "
                    "mp_milbrandt2mom_driver, compiled with WRF's GNU flags "
                    "(-O2 -ftree-vectorize -funroll-loops), object "
                    "byte-identical to the one a WRF build links"),
                "columns": 603,
                "strict_build": "bitwise on every output word",
                "default_build": "bitwise on every output word",
                "harness": "tools/milbrandt2_wrf461_oracle",
                "test": "tests/test_milbrandt2_wrf461_exact_gpu.py",
            },
            # ``arwen_radiation_constraint`` RETIRED with the refusal it
            # described: the RTE+RRTMGP adapter now carries the scheme's
            # own cloud-optics row (woof.core.rrtmgp ``9: "milbrandt2"``),
            # so there is no pairing to refuse and the consumers block's
            # cloud_optics row, derived from that table, is the record.
        },
        "implemented": True,
        "label": "Milbrandt-Yau two-moment / MP9",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 9},
        "warnings": [
            "COLUMN ORACLE: BITWISE. WRF v4.6.1's own "
            "mp_milbrandt2mom_driver (the object WRF's GNU build links, "
            "gfortran 13.3, glibc 2.39, no FMA) and this port, through its "
            "shipped launcher, return bit-identical output on every field "
            "(qv, the six masses, the six numbers, theta, Zet, the eight "
            "precipitation accumulators and SR) for 603 distinct columns "
            "(91 named-regime and edge-case columns at dt 20 s and 60 s, "
            "512 randomised columns at 50 and 80 levels), 5 to 10 "
            "successive calls each, on an RTX PRO 6000, under the default "
            "build and under GPUWM_WRF_EXACT=1. Getting there replaced "
            "CUDA's expf/logf/log10f/powf with WOOF's own gfk_* functions, "
            "followed gfortran's evaluation of powers (REAL**2.0 is a "
            "multiply, integer powers by the power tree, powers before "
            "products), multiplied by WRF's idt where the port divided by "
            "dt, spelled NINT as round-half-away, and compiled the unit "
            "without FMA contraction. There is NO forecast-trajectory "
            "comparison against WRF. The earlier column smoke (finiteness, "
            "boundedness, moment-vs-mass consistency, a water budget on "
            "three seeding layouts) is in tests/test_milbrandt2.py. This "
            "option is a line-by-line "
            "transcription of the byte-frozen WRF v4.6.1 "
            "phys/module_mp_milbrandt2mom.F:841-3485 "
            "(mp_milbrandt2mom_main), :564-836 "
            "(sedi_wrapper_2/sedi_1D/count_columns), :31-433 and :3489-3525 "
            "(the helper functions) and :3559-3703 (the 3-D wrapper) -- and "
            "what has been TESTED is a column smoke "
            "driving the shipped seams (initialize_physics + "
            "microphysics.apply) for finiteness, boundedness and "
            "moment-vs-mass consistency, plus a water budget that closes "
            "to the reported precipitation on three seeding layouts. Read "
            "the budget narrowly: it detects a source/sink term only where "
            "that term is non-zero on the column, so it resolves the "
            "deposition, riming and melting families and NOT the terms "
            "that are zero on all three columns, and the number-only "
            "source/sink lines at "
            "phys/module_mp_milbrandt2mom.F:2721-2728 carry no mass and "
            "cannot appear in a water budget at all. The layouts and the "
            "measured per-term margins are in tests/test_milbrandt2.py.",
            "PINNED IDENTITY, not a subset. WRF exposes no namelist for any "
            "Milbrandt-Yau switch: mp_milbrandt2mom_driver hard-codes "
            "CCNtype=2 (continental, "
            "phys/module_mp_milbrandt2mom.F:3615), precipDiag/sedi/"
            "warmphase/autoconv/icephase/snow all .true. (:3618-3623) and "
            "nk_BOTTOM=.false. (:3591), and the scheme body hard-codes "
            "snowSpherical=.false. (:1174), primIceNucl=1 (:1175) and "
            "grpl/hail/rainAccr/iceDep ON (:1170-1173). mp=9 therefore has "
            "exactly ONE identity in WRF v4.6.1 and WOOF ships that one; "
            "woof/config.py::validate_milbrandt2_options refuses by name "
            "any request to move one, because the 154-entry constant table "
            "in woof/core/milbrandt2_constants.py is derived under these "
            "settings (CCNtype=2 fixes N_c_SM=2e8, snowSpherical=.false. "
            "selects the Brandes m(D) pair).",
            "DELIBERATE DIVERGENCE, WRF is undefined or unreachable, four "
            "of them, each commented at its site in the ported "
            "translation unit woof/core/kernels/milbrandt2.cu. (1) "
            "phys/module_mp_milbrandt2mom.F:2819-2823 is a LIVE "
            "print-and-STOP on T<173 K or T>323 K -- a device kernel cannot "
            "abort a run and WOOF's step-level health gate owns that "
            "decision, so the value is left as computed. (2) count_columns "
            "at phys/module_mp_milbrandt2mom.F:821 would read QX(i,0) when "
            "ktop_sedi==kbot, which needs the lowest two layers to span "
            "20 km, so the walk is clamped to 'inactive'. (3) "
            "phys/module_mp_milbrandt2mom.F:1733, :1754 and :1771 each zero "
            "iLAMxB1 twice and never zero iLAMxB2 -- every iLAMxB2 read "
            "sits inside the Qx>epsQ branch that also writes it, so the "
            "stale value is unreachable and the defined behaviour (zeroing) "
            "is implemented. (4) RT_snd at "
            "phys/module_mp_milbrandt2mom.F:3224-3262 and RT_peL at "
            ":3321-3327 are computed by the scheme and DISCARDED by "
            "mp_milbrandt2mom_driver, so they are not computed at all.",
            "TRANSCRIBED WRF QUIRKS, deliberately NOT corrected because "
            "they are defined behaviour: the surface precipitation "
            "diagnostic reads N_r, DE and T at the model TOP while reading "
            "QR at the bottom (phys/module_mp_milbrandt2mom.F:3298-3313, "
            "nk vs kbot); the raindrop-breakup efficiency tests iLAMr "
            "rather than Dr and goes negative for large iLAMr (:2951); the "
            "hail size-sorting ratio omits the air-density factor its Dm_x "
            "sibling carries (:698); "
            "and with T>0 C the pristine-ice mass is zeroed at :2042 and "
            "the resulting QMLir is then cancelled by the ice "
            "overdepletion guard, so that mass leaves the system rather "
            "than becoming rain. The scheme's own 4-term-truncated Lanczos "
            "gamma (:160-195, 'do j=1,4') is used throughout -- gamma(3) "
            "comes out 1.9999542 and that is the number every WRF mp=9 run "
            "integrates with.",
            "DECLARED DIVERGENCE, RTE+RRTMGP cloud optics (WOOF's own; WRF "
            "has no RRTMGP and hands RRTMG no Milbrandt-Yau radii, "
            "has_reqc=has_reqi=has_reqs=0 at phys/module_physics_init.F:"
            "1004-1023). The adapter radiates the scheme's OWN effective "
            "radii: the block WRF ships commented out at "
            "phys/module_mp_milbrandt2mom.F:3351-3378 -- r_eff = "
            "M_D(3)/(2 M_D(2)) over MY2005a eqn (2), 0.664639/lambda_c for "
            "the alpha_c=1, mu_c=3 cloud distribution and 1.5/lambda_i for "
            "exponential ice, with the scheme's own iLAMDA_x constants and "
            "iLAMmin2 floor -- evaluated from the transported nc/ni on "
            "every radiation call, and extended to snow as 1.5/lambda_s "
            "over the alpha_s=0 exponential with the Brandes m(D) pair "
            "(cms=0.1597, dms=2.078) that snowSpherical=.false. selects. "
            "Ice and snow merge into RRTMGP's one ice species by number "
            "and clip to the table domain exactly as Morrison's row does. "
            "The legacy RRTMG arm is unchanged: it computes its own radii "
            "as WRF does under has_reqc=0, so the two arms radiate "
            "DIFFERENT cloud radii for mp=9 by design; the verification of "
            "record is obs skill, not agreement between them.",
        ],
    }

WSM7_OPTION_ID = "wsm7-mp24"


def _wsm7_mp24(registry: dict) -> None:
    """Register WSM7 (mp_physics=24) from its WSM-family table row.

    ``implemented: true`` -- the column kernel is
    ``woof/core/kernels/wsm7.cu`` (WRF v4.6.1 phys/module_mp_wsm7.F in
    WRF's operation order), run by the table-driven
    ``woof/core/wsm_family.py`` from its row in
    ``woof/data/wsm_family/schemes.json``; no per-scheme adapter exists.

    ``maturity: implemented-unverified`` -- the column is graded word for
    word against WRF's own Fortran (tools/wsm7_wrf461_column_oracle), but
    no matched ArWen-versus-WRF forecast trajectory exists, and the ladder
    reserves the higher rungs for that.

    Reachability is computed from the registered WSM7 suite and runner
    overrides. The suite retains the component evidence ceiling.
    """
    from woof.core.wsm_family_constants import scheme_metadata

    wsm7 = scheme_metadata(24)
    options = registry["components"]["microphysics"]["options"]
    options[WSM7_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON binds mp_physics==24 to the "
                "wsm7scheme package: moist qv,qc,qr,qi,qs,qg,qh and state "
                "re_cloud,re_ice,re_snow"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "qh"],
                "not_ported": [],
                "note": (
                    "WSM7 is WSM6's six-class mass set plus a separate hail "
                    "class qh. Nothing in the WRF package is left out."),
            },
            "wsm_family_row": {
                "table": "woof/data/wsm_family/schemes.json",
                "unit": wsm7["unit"],
                "kernel_species": list(wsm7["kernel_species"]),
                "precipitating_classes": list(wsm7["precip"]),
                "level_tiers": list(wsm7["level_tiers"]),
                "words": "woof/data/wsm_family/" + wsm7["words_file"],
            },
            "declared_divergences": [dict(d) for d in wsm7["declared_divergences"]],
        },
        "implemented": True,
        "label": "WSM7 single-moment with hail / MP24",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 24},
        "warnings": [
            "COLUMN ORACLE, NOT A FORECAST MATCH. The CUDA kernel is graded "
            "word for word against WRF v4.6.1 phys/module_mp_wsm7.F built "
            "with gfortran -O0 (tools/wsm7_wrf461_column_oracle, "
            "tests/test_wsm7_wrf461_column_exact.py): 108 columns over 45 "
            "and 72 levels, five time-step configurations, every output "
            "field of every step, in the strict and the default build. No "
            "matched WOOF-versus-WRF forecast trajectory exists for WSM7.",
            "TWO DECLARED DIVERGENCES FROM WRF, default-on. D1: the PLM "
            "sedimentation remap searches every arrival interface and the "
            "column top, so no layer drops its arriving mass; WRF's search "
            "stops one interface short and deletes that mass (the same "
            "search WSM6 and WDM6 carry). D2: REFL_10CM includes hail's "
            "dry-ice Rayleigh term with WSM7's own hail size distribution; "
            "WRF's refl10cm_wsm7 leaves the hail class out. The kernel is "
            "graded against WRF's Fortran with exactly these two "
            "substitutions, and differs from unmodified WRF only in the "
            "columns the oracle's probe arm identifies (D1) and in "
            "hail-bearing reflectivity (D2).",
        ],
    }


    options[WSM7_OPTION_ID]["physics_identity_extension"] = {"transitions": True}
    # RETIRED: the lane's named suite "wsm7-ysu-revised-mm5-noah-rrtmg-
    # legacy-v1" (Thompson legacy with WSM7, revised MM5, Grell-Freitas).
    # It was declared only on the native HRRR benchmark route, whose runner
    # replays a transcribed native namelist per profile and refused it
    # ("unsupported native HRRR physics profile"; no _PROFILE_SWITCH_HOMES
    # for clos_choice/ishallow).  WSM7 is reached, like WSM3, WSM5, WSM6R
    # and Ferrier-Aligo, as mp_physics=24 (namelist import, component
    # override).  The builder is seeded from the tracked registry, so the
    # id is removed explicitly here and from every route below.
    registry["templates"].pop(WSM7_RETIRED_TEMPLATE_ID, None)


#: The retired WSM7 suite id (see _wsm7_mp24); kept so the seeded registry
#: and every route declaration lose it on each build.
WSM7_RETIRED_TEMPLATE_ID = "wsm7-ysu-revised-mm5-noah-rrtmg-legacy-v1"


WSM3_OPTION_ID = "wsm3-mp3"


def _wsm3_mp3(registry: dict) -> None:
    """Register WSM3 (mp_physics=3) at exactly the maturity it has earned.

    ``implemented: true`` -- the column kernel is
    ``woof/core/kernels/wsm3.cu``, a word-for-word port of the byte-
    unmodified WRF v4.6.1 ``phys/module_mp_wsm3.F``; the adapter is
    ``woof/core/wsm3.py``; ``woof/core/microphysics.py`` dispatches on
    ``mp_physics == 3``; the namelist importer maps 3 natively.

    ``maturity: implemented-unverified`` -- the COLUMN scheme is graded word
    for word against its gfortran oracle (the warning below gives the
    numbers); no whole-model WRF trajectory comparison has been run, which
    is what the label withholds.

    ``reachability: component-override`` -- computed, not chosen: no
    template registers mp_physics=3, so the option is reached as a
    per-domain override, which is what an imported namelist produces.
    """

    options = registry["components"]["microphysics"]["options"]
    options[WSM3_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3017 binds mp_physics==3 to "
                "the wsm3scheme package: moist qv,qc,qr plus state "
                "re_cloud,re_ice,re_snow"),
            "prognostic_species": {
                "transported": [],
                "not_ported": [],
                "note": (
                    "WSM3 is the simple-ice scheme: qc is cloud water at or "
                    "above 0 C and cloud ice below it, qr is rain at or "
                    "above 0 C and snow below it. It carries no qi, qs or "
                    "qg, so the Registry's F_QI, F_QS and F_QG are false, "
                    "and module_physics_init.F leaves warm_rain false."),
            },
        },
        "implemented": True,
        "label": "WSM3 simple ice / MP3",
        "maturity": "implemented-unverified",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 3},
        "warnings": [
            "COLUMN ORACLE, NOT A TRAJECTORY. The CUDA kernel is graded word "
            "for word against WRF v4.6.1 module_mp_wsm3.F compiled by "
            "gfortran 13.3.0 at -O0 over 77 columns x 45 levels and five "
            "time steps (tools/wsm3_wrf461_column_oracle, "
            "tests/test_wsm3_wrf461_column_exact.py). No whole-model "
            "comparison with a WRF forecast has been run.",
            "DECLARED DIVERGENCE D1, a WRF water-loss defect fixed default-"
            "on: WRF adds the cloud-ice fallout to the surface only when "
            "the lowest level is below 0 C (module_mp_wsm3.F:604-607), but "
            "the semi-Lagrangian CFL limiter lets ice fall out of a warm "
            "lowest level beneath a cold one, and WRF then counts that ice "
            "nowhere. woof counts it as rain. Only RAINNC, RAINNCV and SR "
            "move; the oracle's second reference build carries exactly this "
            "one-line change. GPUWM_WRF_EXACT=1 retains stock WRF surface "
            "accounting and is graded against unmodified WRF on every word; "
            "default arithmetic differences are reported separately.",
            "DECLARED DIVERGENCE D2, undefined behaviour fixed: with the "
            "melting level at the model top and rising air there, WRF moves "
            "the freezing level to kte+1 and reads its temperature array out "
            "of bounds (module_mp_wsm3.F:572-592); woof freezes at the top "
            "level.",
            "DECLARED DIVERGENCE D3: default sedimentation includes the "
            "last arrival interface, zero-density extension and kt >= kb, "
            "using the delivered WSM6/WDM6 upper-cell search repair. "
            "GPUWM_WRF_EXACT=1 retains the stock search for comparison "
            "against unmodified WRF.",
            "RADIATION: WSM3 hands its three radii to RRTMG "
            "(WSM3SCHEME is in module_physics_init.F's use_mp_re list), and "
            "WRF's radiation wrappers move cold qc into qi and cold qr into "
            "qs for a scheme without F_QI and without warm_rain before "
            "reading them (module_ra_sw.F:193-203). The species flags woof "
            "publishes for mp=3 are WRF's; the Dudhia, RRTM and RTE+RRTMGP "
            "conversion keyed on them is the radiation side's work.",
            "REFL_10CM: WRF computes none for WSM3. woof's is its own "
            "diagnostic: refl10cm_wsm6 on WSM3's phase split of qr (rain at "
            "or above 0 C, snow below, no graupel), whose intercepts and "
            "densities are WSM3's own (woof.core.refl."
            "wsm3_precipitation_phases).",
            "NEST EDGES: a same-scheme WSM3 hierarchy resolves; a mixed edge "
            "between WSM3 and another scheme is refused by name "
            "(woof.core.microphysics_transition, its phase split of qc/qr "
            "has no closure).",
        ],
    }



NTU_OPTION_ID = "ntu-mp56"

def _ntu_mp56(registry: dict) -> None:
    """Register NTU multi-moment microphysics (mp_physics=56).

    ``implemented: true`` -- ``woof/core/ntu.py`` and
    ``woof/core/kernels/ntu.cu`` run MP_NTU on the device and
    ``woof/core/microphysics.py`` dispatches ``mp_physics == 56``.

    Qualification evidence is recorded by tools/ntu_wrf471_oracle.
    """
    options = registry["components"]["microphysics"]["options"]
    options[NTU_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3044 binds mp_physics==56 to "
                "the ntu package: moist qv,qc,qr,qi,qs,qg,qh and scalar "
                "qnc,qnr,qni,qns,qng,qnh,qdcn,qtcn,qccn,qrcn,qnin,fi,fs,vi,"
                "vs,vg,ai,as,ag,ah,i3m"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "qh"] + ['nc', 'nr', 'ni', 'ns', 'ng', 'nh', 'qdcn', 'qtcn', 'qccn', 'qrcn', 'qnin', 'fi', 'fs', 'vi', 'vs', 'vg', 'ai', 'as_', 'ag', 'ah', 'i3m'],
                "not_ported": [],
                "note": (
                    "Every field of the WRF package is carried. The snow "
                    "second moment 'as' lives at state.as_ ('as' is a "
                    "Python keyword); its wrfout name stays AS."),
            },
            "namelist": {
                "ccnty": ("ntu_ccnty: AERO_CONST aerosol background, "
                          "1 marine, 2 clean continental (WRF default), 3 "
                          "average continental, 4 urban"),
            },
            "source_manifest": "tools/ntu_wrf471_oracle/source.json",
            "column_oracle": "Real native capture: strict GPU replay 0/2,048,705 differing words; tools/ntu_wrf471_oracle/README.md",
        },
        "implemented": True,
        "label": "NTU multi-moment / MP56",
        "maturity": "implemented-unverified",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 56},
        "warnings": [
            "Pinned WRF v4.7.1 NTU module. No defined source correction is applied.",
            "Native undefined initialization boundary and rimed-density reads are documented in tools/ntu_wrf471_oracle/README.md.",
            "One three-hour stock-namelist forecast has hourly Rust maps beside native WRF and ASOS; this is one-case evidence, not general forecast skill.",
            "Opt-in ntu-mp56 only (mp_physics=56), never a default: real-input fast stage on RTX PRO 6000 Blackwell is 81.03 ms versus MY2 0.366 ms (221.54x); prior RTX 4090 cost is 352.76x. The 1.2x bar is exceeded.",
            "An actual native REAL32 profile records 274,625 adaptive iterations and 80,568 SMALL_DT calls on this tile; this is extra NTU work. GPU register pressure and low active-lane use also remain; the measured cost is not claimed to be an optimal bound. See tools/ntu_wrf471_oracle/SPEED.md.",
            "WRF's NTU driver provides no REFL_10CM. The engine derives Rayleigh reflectivity from the predicted distributions.",
            "RTE+RRTMGP optics are refused until moment-derived effective radii are qualified; legacy RRTMG follows WRF's coupling.",
        ],
    }

WDM6_OPTION_ID = "wdm6-mp16"
#: WRF cu_physics=99 (KFSCHEME), the previous Kain-Fritsch.
KF_OLD_OPTION_ID = "kain-fritsch-cu99"

WDM5_OPTION_ID = "wdm5-mp14"


def _wdm_family_row(registry: dict, selector: int, scheme: int) -> None:
    """Register the native WRF WDM column and its transported carriers."""
    import copy
    row = copy.deepcopy(registry["components"]["microphysics"]["options"][WDM6_OPTION_ID])
    hail = selector == 26
    transported = ["qi", "qs"] + (["qg", "qh"] if hail else []) + ["nn", "nc", "nr"]
    row.update(label=f"WDM{scheme} double-moment warm rain" + (" with hail" if hail else "") + f" / MP{selector}",
               selectors={"mp_physics": selector},
               parameters={"moist": True, "moist_cq": True, "wdm6_ccn_conc": 1.0e8,
                           **({"wdm6_hail_opt": 0} if hail else {})},
               maturity="wrf-matched-run-candidate", scientific_evidence="none")
    row["extensions"] = {
        "wrf_package": f"Registry/Registry.EM_COMMON: WDM{scheme}; mass qv,qc,qr,qi,qs" + (",qg,qh" if hail else "") + "; number qnn,qnc,qnr; re_cloud,re_ice,re_snow",
        "prognostic_species": {"transported": transported, "not_ported": [],
                                "note": "Native WRF number carriers are CCN, cloud and rain only."},
        "kernel": {"unit": f"woof/core/kernels/wdm{scheme}.cu",
                   "launcher": f"woof.core.wdm6.launch_wdm{scheme}"},
        "native_replay": {
            "source_pin": "d66e442fccc04111067e29274c9f9eaccc3cef28",
            "provenance": f"tests/data/oracles/wdm{scheme}/native_step2.provenance.json",
            "words": 95658 if selector == 14 else 110856,
            "differing_words": 0,
            "scope": "149 real native columns, 49 levels, timestep two, dt=15 s; strict numerics; reflectivity included",
        },
    }
    if hail:
        row["extensions"]["surface_field_dependency"] = {
            "field": "XLAND", "authority": "phys/module_mp_wdm7.F:584-590",
            "note": "WRF uses separate maritime and continental autoconversion thresholds."}
    row["warnings"] = [
        "Pinned to stock WRF v4.6.1, d66e442fccc04111067e29274c9f9eaccc3cef28. Strict numerics select native arithmetic for bitwise replay; default numerics permit FMA. See docs/wdm" + str(scheme) + "_oracle_known_deltas.md for evidence limits.",
        "Forecast, restart, ensemble and same-scheme nest qualification must be checked separately from column conformance. Cross-scheme number and hail transitions remain refused until validated.",
    ]
    if hail:
        row["warnings"].append("Native WRF refl10cm_wdm7 reads rain, snow and graupel, and has no hail contribution.")
    registry["components"]["microphysics"]["options"][f"wdm{scheme}-mp{selector}"] = row


def _wdm5_mp14(registry: dict) -> None:
    _wdm_family_row(registry, 14, 5)


WDM7_OPTION_ID = "wdm7-mp26"


def _wdm7_mp26(registry: dict) -> None:
    _wdm_family_row(registry, 26, 7)


TEMPO_OPTION_ID = "tempo"


P3_MP53_OPTION_ID = "p3-mp53"


def _p3_three_moment_mp53(registry: dict) -> None:
    """Register P3 three-moment ice (mp_physics=53) as a P3_VARIANTS row.

    The option is mp=50's option with the variant's differences and
    nothing else: one adapter (woof/core/p3.py) and one translation unit
    (woof/core/kernels/p3.cu) run every row of woof.config.P3_VARIANTS,
    so the copy is the faithful spelling of "same scheme, three switches
    set".  Built from the p3-mp50 row on every pass, so the build stays a
    fixed point.

    The real-input module replay and one three-hour forecast are recorded
    in docs/public/receipts/p3-mp53-port-20261101.json. The named mixed-nest
    restriction keeps this row at experimental-runtime. Independent
    scientific validation is not claimed. No template selects it.
    """
    from woof.config import P3_VARIANTS
    from woof.core.p3_tables import (
        TABLE_1_3MOM_ASSET, TABLE_1_3MOM_PACKAGED_SHA256,
        TABLE_1_3MOM_PACKAGED_SIZE)

    options = registry["components"]["microphysics"]["options"]
    base = options["p3-mp50"]
    # mp=50's own row stops calling 53 unported, and its asset note stops
    # saying the 3momI table is not packaged.
    base.setdefault("extensions", {}).setdefault(
        "unported_wrf_siblings", {}).pop("53", None)
    base["extensions"].setdefault("ported_wrf_siblings", {}).update({
        "53": "three-moment ice with prognostic droplet number "
              "(p3_1cat_3mom), option " + P3_MP53_OPTION_ID})
    for requirement in base.get("asset_requirements", []):
        if requirement.get("id") == "wrf-v4.6.1-p3-lookuptable1-2momi-v1":
            requirement["note"] = (
                "p3_init READS this file (module_mp_p3.F:337-410); it "
                "computes nothing. No woof code path regenerates it, so a "
                "missing or byte-modified table is a hard refusal rather "
                "than any fallback, raised by the SHA-256 validator in the "
                "loader module woof.core.p3_tables. P3's OTHER tables, the "
                "rain fallspeed and ventilation arrays, are generated "
                "in-process from module_mp_p3.F:599-670 and need no asset. "
                "Lookup table 2 (mp=52) is deliberately not packaged: that "
                "option is refused by name in woof.config. The 3momI "
                "variant of table 1 belongs to p3-mp53.")
    row = P3_VARIANTS[53]
    option = copy.deepcopy(base)
    option.pop("consumers", None)
    option["label"] = "P3 one-category three-moment ice / MP53"
    option["maturity"] = "experimental-runtime"
    option["selectors"] = {"mp_physics": 53}
    option["reachability"] = {"state": "component-override"}
    option["verification_scope"] = "real-input-module-replay-and-single-real-forecast"
    option["verification_scope_note"] = (
        "2026-11-02: 92,442 computed words from four retained native WRF forecast "
        "slabs match bitwise under strict numerics. One 73x60x34, 30 km, "
        "three-hour stock-namelist forecast completed through woof go "
        "under default numerics and was reviewed beside native WRF and "
        "ASOS surface observations at three valid times. The whole production "
        "microphysics stage took 1.544192 ms versus MP50's 1.408000 ms "
        "(1.096727x) in interleaved RTX 5090 forecasts. This is a "
        "limited integration check, not whole-model bitwise parity or "
        "a general forecast-skill result. Receipt: "
        "docs/public/receipts/p3-mp53-port-20261101.json.")
    option["asset_requirements"] = [{
        # The PACKAGED file, which is what is on disk and what plan review
        # resolves; the text digest p3_init's loader pins is in the note.
        "assets": [{
            "bytes": TABLE_1_3MOM_PACKAGED_SIZE,
            "filename": TABLE_1_3MOM_ASSET.packaged_filename,
            "sha256": TABLE_1_3MOM_PACKAGED_SHA256,
        }],
        "id": "wrf-v4.6.1-p3-lookuptable1-3momi-v1",
        "installed_root": "woof/data/p3/tables",
        "kind": "bundled-package-data",
        "note": (
            "p3_init(trplMomI=.true.) READS this file "
            "(module_mp_p3.F:412-470); it computes nothing. It is WRF's own "
            "run/ " + TABLE_1_3MOM_ASSET.filename + " text ("
            + str(TABLE_1_3MOM_ASSET.size) + " B, sha256 "
            + TABLE_1_3MOM_ASSET.sha256 + "), packaged xz-compressed as the "
            "file above. The Rust rw-p3 bridge decompresses and parses it "
            "in parallel, with version, size and digest "
            "checks; a missing, unreadable or byte-modified table is a hard "
            "refusal. The parsed itab_3mom/itabcoll_3mom are bit-identical "
            "to WRF's own read loop (FORTRAN_ITAB3_SHA256, "
            "FORTRAN_ITABCOLL3_SHA256)."),
        "redistributed_by_gpuwm": True,
        "resolution": {
            "data_relative": "p3/tables",
            "root_environment_override": "WOOF_P3_TABLE_ROOT",
        },
        "root_environment_override": "WOOF_P3_TABLE_ROOT",
        "source": (
            "WRF v4.6.1 (git tag v4.6.1, commit "
            "d66e442fccc04111067e29274c9f9eaccc3cef28), file "
            "run/p3_lookupTable_1.dat-v5.4_3momI"),
    }]
    option["extensions"] = {
        "execution_model": (
            "Runs on the card: p3_backend selects 'cuda' (the default) or "
            "'fused'. 'reference', the CPU float32 transcription, carries "
            "two-moment ice only and is refused for this option by name."),
        "fixed_mode": (
            "one ice category, three-moment ice (prognostic Z, carried as "
            "the advected (N*Z)**0.5 qzi), predicted droplet number with "
            "CCN activation, SCPF cloud fraction off"),
        "wrf_package": (
            "Registry/Registry.EM_COMMON:%d binds mp_physics==53 to the %s "
            "package: moist qv,qc,qr,qi and scalar qnc,qni,qnr,qir,qib,qzi"
            % (row.registry_line, row.package)),
        "ported_wrf_siblings": {
            "51": "prognostic droplet number with two-moment ice "
                  "(p3_1category_nc), option p3-mp51",
            "52": "two ice categories (p3_2category), option p3-mp52",
        },
        "not_ported": {
            "mixed_nest_edges": (
                "a nest edge between mp=53 and another scheme: refused by "
                "name in woof.core.microphysics_transition; same-scheme "
                "nests carry nc and qzi"),
        },
    }
    option["warnings"] = [
        "Measured against WRF's own Fortran, not against itself, and "
        "bit-identical to it. The oracle (tools/p3_wrf461_oracle, variant "
        "53) is the unmodified phys/module_mp_p3.F (sha256 716950a3...f6a3c6) "
        "driven through mp_p3_wrapper_wrf in WRF's own P3_1CAT_3MOM call "
        "shape -- nc_3d and qzi1_3d present, N_ICECAT=1, "
        "p3_init(trplMomI=.true.) on run/p3_lookupTable_1.dat-v5.4_3momI -- "
        "at -O0 -ffp-contract=off and linked with correctly rounded tgammaf "
        "and log10f. Over 88 columns (the mp=50 regimes plus ten "
        "three-moment edges: qzi without ice, ice without qzi, mu_i pinned "
        "at both bounds, Ni below qsmall, droplets with zero and with "
        "5e9/kg number, activation in clear supersaturated air, every "
        "freezing path, dense hail), 1 to 30 steps at dt 20 s and 60 s, "
        "every output word of every step equals WRF's, all 24 fields "
        "including nc and qzi, in the default build and under "
        "GPUWM_WRF_EXACT=1 alike. A planted one-bit change in "
        "compute_mu_3moment fails that comparison. The one documented "
        "divergence is mp=50's first-call guard: with th_old and qv_old "
        "still zero WRF evaluates 0/0 for sup/supi and WOOF floors "
        "qvs/qvi at 1e-20 (fixture X01, outside the claim). A separate "
        "real-input replay matches all 92,442 computed words; the actual "
        "native inputs and reference outputs are retained. One "
        "three-hour real forecast and ASOS surface comparison are "
        "recorded in verification_scope_note; broader forecast skill "
        "has not been established.",
        "P3 has ONE ice category: an mp=53 run produces no QSNOW and no "
        "QGRAUP field and no GRAUPELNC accumulator. It carries QNCLOUD "
        "(prognostic droplet number) and QZI, the advected square root of "
        "the ice number times its sixth moment, beside mp=50's QIR/QIB.",
        "Mixed nest edges into or out of mp=53 are refused by name; the CPU reference arm "
        "(p3_backend='reference') carries two-moment ice only and is "
        "refused for this option.",
    ]
    options[P3_MP53_OPTION_ID] = option


def _tempo_named(registry: dict) -> None:
    """Register TEMPO, the first NAMED microphysics scheme.

    Selected by name (TOML ``mp_physics = "tempo"``); its selector value 900
    is the WOOF-side id from woof.microphysics_schemes, outside WRF's
    mp_physics range, so a WRF namelist number never means something else in
    WOOF.  Everything a consumer needs is the scheme's capability row there.

    ``maturity: implemented-unverified``: the column port is bitwise against
    TEMPO's own Fortran on the CPU and within 1.9e-10 on the GPU, but no
    forecast-trajectory comparison against a reference model exists.
    ``reachability: component-override``: no template selects it.
    """
    from woof.microphysics_schemes import TEMPO
    from woof.tempo_assets import TEMPO_RELEASE_URL_BASE, TEMPO_TABLE_ASSETS

    options = registry["components"]["microphysics"]["options"]
    options[TEMPO_OPTION_ID] = {
        "asset_requirements": [{
            "id": "ncar-tempo-v3-tables-v1",
            "kind": "downloaded-table-set",
            "assets": [{"filename": a.filename, "bytes": a.bytes, "sha256": a.sha256}
                       for a in TEMPO_TABLE_ASSETS],
            "redistributed_by_gpuwm": False,
            "source": TEMPO_RELEASE_URL_BASE,
            "search_root": "~/.woof/tables/tempo-v3",
            "root_environment_override": "WOOF_TEMPO_TABLE_ROOT",
            "path_environment_override": "WOOF_TEMPO_TABLE_SOURCE_ROOT",
            "regenerable": True,
            "note": (
                "TEMPO's published tempo_v3.1.0 table assets (Apache-2.0), "
                "pinned by size and SHA-256 in woof/tempo_assets.py and "
                "fetched on first use. Tables built from the pinned TEMPO "
                "source 9adf9ef agree with them to 2.2e-12 relative."),
        }],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "woof_named_scheme": {
                "name": TEMPO.name, "woof_id": TEMPO.mp_id,
                "capability_row": "woof.microphysics_schemes.TEMPO"},
            "source": {
                "repository": "https://github.com/NCAR/TEMPO",
                "commit": "9adf9ef2827104640125f0c322e1218fce74e94c",
                "tag": "tempo_v3.0.3",
                "pinned_by": ("RRFS v2 (rrfs-workflow rrfs-mpas-jedi, "
                              "RRFSx/MPAS-Model 1d6e370e)"),
                "licence": "Apache-2.0"},
            "prognostic_species": {
                "transported": list(TEMPO.ice_mass_species + TEMPO.moment_species
                                    + TEMPO.aerosol_species),
                "surface_emission_2d": list(TEMPO.surface_fields),
                "not_ported": ["ml_for_nc", "ml_for_bl_nc", "refl10cm_from_melting"],
                "note": (
                    "Aerosol-aware and hail-aware (prognostic graupel number "
                    "and volume, 9 density classes). The machine-learning "
                    "droplet number and melting reflectivity options are off "
                    "in RRFS and not wired."),
            },
            "column_oracle_evidence": {
                "authority": ("unmodified TEMPO 9adf9ef, gfortran 13.3 "
                              "-fdefault-real-8, through tempo_run"),
                "fixtures": 6,
                "cpu_float64": "bitwise on the five default-sedimentation fixtures",
                "gpu": "bitwise reproducible; worst relative difference 1.9e-10",
                "init": "Rust tempo_init bitwise on all 249 blob fields",
                "record": "docs/TEMPO-PORT.md",
            },
        },
        "implemented": True,
        "label": 'TEMPO (Thompson-Eidhammer, RRFS v2) / named scheme "tempo"',
        "maturity": "implemented-unverified",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": TEMPO.mp_id},
        "warnings": [
            'NAMED SCHEME. mp_physics = "tempo" resolves to the WOOF-side '
            "id 900; WRF has no TEMPO option, so no WRF namelist selects it "
            "and WRF-matched comparisons do not apply.",
            "TEMPO's optional semi-Lagrangian sedimentation amplifies a "
            "1e-15 perturbation to order one in TEMPO itself; it is off "
            "here, as in RRFS v2.",
        ],
    }


def _gsfcgce4ice_mp7(registry: dict) -> None:
    """NU-WRF four-ice package and its shared column-driver bindings."""
    from woof.core.gsfcgce4ice import (SCHEME_LEVEL_FIELDS, HEATING_FIELDS,
        ACCUMULATED_HEATING_FIELDS, RADIUS_FIELDS, FLUX_FIELDS, SURFACE_SLOTS)
    masses = ['qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh']
    volume = ['qi', 'qs', 'qg', 'qh', 'effc', 'effi', 'effs',
              *SCHEME_LEVEL_FIELDS, 'phys_tot', 'acphys_tot']
    binding = {
        'module': 'woof.core.gsfcgce4ice', 'launcher': 'launch_gsfcgce4ice',
        'surface_validator': 'refuse_unknown_land_mask',
        'volume_fields': volume,
        'transported_extras': ['qi', 'qs', 'qg', 'qh'],
        'sr_sum_additions': [3, 4],
        'vertical_bounds': [2, None],
        'mixed_edge_refusal': 'The hail mass folding into and out of this four-ice package has not been validated; a mixed edge could drop hail water or retain an inactive category. Use the same scheme on both domains.',
        'initial_values': {'effc': 2.49, 'effi': 4.99, 'effs': 9.99},
        'species': {name: name for name in masses},
        'groups': {'heating': list(HEATING_FIELDS),
                   'accumulated': list(ACCUMULATED_HEATING_FIELDS),
                   'radii': list(RADIUS_FIELDS), 'fluxes': list(FLUX_FIELDS)},
        'surface_slots': {slot.removeprefix('mp_'): slot for slot in SURFACE_SLOTS},
        'surface_history_fields': {'HAILNCV': 'hailncv'},
        'radiation_radii': {'effc': 're_cloud_gsfc', 'effi': 're_ice_gsfc',
                           'effs': 're_snow_gsfc'},
        'sum_outputs': {'phys_tot': list(HEATING_FIELDS),
                        'acphys_tot': list(ACCUMULATED_HEATING_FIELDS)},
        'history_fields': {name.upper(): name for name in
                           (*SCHEME_LEVEL_FIELDS, 'phys_tot', 'acphys_tot')},
    }
    registry['components']['microphysics']['options']['goddard4ice-mp7'] = {
        'asset_requirements': [], 'implemented': True,
        'label': 'Goddard four-ice with hail / MP7 (opt-in; measured 1.221x WSM6 stage)',
        'maturity': 'supported', 'scientific_evidence': 'none',
        'selectors': {'mp_physics': 7},
        'parameters': {'moist': True, 'moist_cq': True},
        'constraints': {'required_settings': {'moist': True}},
        'extensions': {'column_kernel': binding,
            'standalone_helpers': {
                'negcor': {'launcher': 'launch_negcor', 'default_active': False,
                           'reason': 'The WRF entry comments out this tile correction.'},
                'auto_conversion': {'launcher': 'launch_auto_conversion', 'default_active': False,
                                    'reason': 'The WRF caller is in chemistry coupling; the fixed no-threshold helper is available separately.'},
                'refl10cm_gsfc': {'launcher': 'launch_refl10cm_gsfc', 'default_active': False,
                                 'reason': 'The active entry writes hail-inclusive saticel_s reflectivity; this helper omits hail.'}},
            'fixed_mode': 'WRF 4.6.1 NU-WRF four-ice, separate graupel and hail, WRF_CHEM=0'},
        'reachability': {'state': 'component-override'},
        'warnings': [
            'Named opt-in; never selected by a default. In the retained two-hour 3km convective forecast the complete MP stage costs 1.221x WSM6 on the same card and commit with interleaved steps. Native WRF 4.7.1 process CPU profiles show 2.763x WSM6 stage work. Cost is case-dependent; see docs/microphysics-retiming.md. Fresh native471 strict replay is 0/8,635,652 words; retained captures are under paper/wrf-ports/spd-retime-mp/mp7/.',
            'Real native strict replay, six-hour stock-input GPU forecast, '
            'Rust maps beside native WRF and observations, checkpoint and '
            'option-off byte comparisons are recorded in '
            'tools/gsfcgce4ice_wrf461_oracle/qualification.json. '
            'Implementation qualification does not establish observed forecast skill.',
            'WRF_CHEM=0 scope. WRF comments out the negcor calls and never calls '
            'refl10cm_gsfc from this entry. auto_conversion is chemistry-only. '
            'The active reflectivity is saticel_s DBZ, including hail.'],
    }
    _VERTICAL_BOUND_SOURCES['microphysics']['goddard4ice-mp7'] = (
        'Goddard four-ice microphysics', 'GSFCGCE4ICE_VERTICAL_LEVEL_BOUNDS')
    _MOMENT_ROWS[7] = {'name': 'Goddard four-ice', 'mass_only': masses,
        'pairs': [], 'unpaired': [], 'repair_authority': None, 'q_threshold': 1e-20}
    _OFFLINE_CHILD_ROWS[7] = {'same_scheme': True, 'refusal': None}
    _RING_STATE_ROWS[7] = ['thp', 'qv', 'qc', 'qr', *volume]
    _RING_SURFACE_ROWS[7] = list(SURFACE_SLOTS)

def _wdm6_mp16(registry: dict) -> None:
    """Register WDM6 (mp_physics=16) at exactly the maturity it has earned.

    ``implemented: true`` -- the component exists and runs.  The column
    kernel is ``woof/core/kernels/wdm6.cu``, transcribed line by line from
    the byte-frozen WRF v4.6.1 ``phys/module_mp_wdm6.F``; the adapter is
    ``woof/core/wdm6.py``; ``woof/core/microphysics.py`` dispatches on
    ``mp_physics == 16``; the state carries qnn/qnc/qnr; the namelist
    importer maps 16 natively.

    ``maturity: implemented-unverified`` -- for the reason it carries on
    ``shinhong``: the column oracle (tools/wdm6_wrf461_oracle) grades the
    kernel word for word against WRF v4.6.1's Fortran, with its three
    documented defined behaviours, and the first warning states exactly
    that and no more.  Conformance is not validation: no woof/WRF forecast
    trajectory comparison exists for mp_physics=16.

    ``reachability: component-override`` -- computed from the routes, not
    chosen.  No template registers mp_physics=16 and no runner route lists
    one, so an implemented microphysics option with no template is reachable
    exactly one way: as a per-domain experiment override.
    ``tests/test_registry_reachability.py`` recomputes the state and would
    fail on any other declaration.
    """

    options = registry["components"]["microphysics"]["options"]
    options[WDM6_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3031 binds mp_physics==16 to "
                "the wdm6scheme package: moist qv,qc,qr,qi,qs,qg and scalar "
                "qnn,qnc,qnr, plus state re_cloud,re_ice,re_snow"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg", "nn", "nc", "nr"],
                "not_ported": [],
                "note": (
                    "WDM6 is WSM6's six-class mass set plus three number "
                    "moments: nn is the CCN reservoir, nc the cloud droplet "
                    "number and nr the rain number. Nothing in the WRF "
                    "package is left out."),
            },
            "surface_field_dependency": {
                "field": "XLAND",
                "authority": "phys/module_mp_wdm6.F:607-614",
                "note": (
                    "WDM6 is the first woof microphysics scheme whose "
                    "PROCESS RATES read a surface field: the autoconversion "
                    "threshold is qc0 (maritime, xncr0=5e7) where xland==2 "
                    "and qc1 (continental, xncr1=5e8) elsewhere, a factor of "
                    "ten. The adapter reads XLAND from the physics driver "
                    "and REFUSES when none is present rather than assuming "
                    "a mask."),
            },
        },
        "implemented": True,
        "label": "WDM6 double-moment warm rain / MP16",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
            "wdm6_hail_opt": 0,
            "wdm6_ccn_conc": 1.0e8,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 16},
        "warnings": [
            "WDM6's distance from WRF v4.6.1 is MEASURED by a column oracle. "
            "tools/wdm6_wrf461_oracle drives WRF's own wdm6 entry of the "
            "byte-frozen phys/module_mp_wdm6.F at gfortran -O0 (no FMA, "
            "scalar libm) over 84 columns built to reach regimes and "
            "branches (convective land and ocean, stable night, winter snow "
            "and ice, a melting layer, high terrain, a cold anvil, "
            "evaporating rain, numberless rain and cloud, edge cases) in "
            "seven configurations (one to eight minor loops, up to ten "
            "consecutive calls, both hail_opt arms, the cold-start CCN "
            "fill, and a 72-level copy on the second kernel tier). Every "
            "output word -- theta, the six masses, nn/nc/nr, the three "
            "effective radii, refl_10cm and the surface accumulators -- of "
            "the CUDA kernel equals the oracle's DEFINED arm, which is the "
            "unmodified file with exactly the three documented defined "
            "behaviours of docs/wdm6_oracle_known_deltas.md substituted, "
            "under the strict build (tests/test_wdm6_wrf461_oracle.py, RTX "
            "PRO 6000, NVRTC 12.9). Against the unmodified file the kernel "
            "is bitwise wherever those behaviours do not fire and differs "
            "exactly where they do: the conservative rain transfer (section "
            "5, every column carrying rain) and the zero-rate condensation "
            "cap (section 6, numberless rain); the kt clamp (section 2) "
            "never fires on the fixture. What made it exact: wdm6init's own "
            "float32 coefficient words instead of float64 products, WOOF's "
            "own float32 exp/log/pow, the Fortran's operation order, and no "
            "multiply-add contraction. The default build keeps the loader's "
            "flush-to-zero route: its state differs only in columns into "
            "which a subnormal moisture value enters a call, and a surface "
            "accumulator elsewhere by under 5e-34 mm. That is conformance "
            "evidence, not scientific "
            "validation: no woof/WRF forecast trajectory comparison exists "
            "for this scheme, which is why this option is "
            "'implemented-unverified' and not 'supported'.",
            "DELIBERATE DIVERGENCE, WRF is inconsistent with itself (both "
            "halves are defined behaviour, so this is transcribed rather "
            "than repaired, and recorded here so it is not mistaken for a "
            "port error): wdm62D consumes ncr as a VOLUMETRIC number, "
            "dividing by (q*den) at module_mp_wdm6.F:2251, while "
            "refl10cm_wdm6 forms nr = nr1d*rho against rr = qr1d*rho so the "
            "densities cancel and nr1d enters as a PER-MASS number (:3005). "
            "The two rain slope definitions therefore differ by a factor "
            "rho inside a cube root, about 3 per cent in lamr near the "
            "surface. woof reproduces both as written.",
            "ENGINE SEAM, recorded deviations of the mp_physics=16 adapter: "
            "(1) WRF fills the whole CCN array with the namelist ccn_conc "
            "on the first time step (module_mp_wdm6.F:220-227) and ccn0 "
            "reaches nothing else in the module; woof performs that fill "
            "once at state allocation from cfg.wdm6_ccn_conc (the NSSL qnn "
            "precedent) and the wrfinput restore refloods an ingested QNCCN "
            "with it, as WRF's unconditional flood discards one; the column "
            "oracle's cold-start configuration grades the equivalence. A "
            "future ingest that is meant to KEEP an initial CCN field must "
            "advance the restart algorithm identity rather than resume onto "
            "this one. (2) WDM6 is not in "
            "the registry's cross-scheme nest transition table, so a domain "
            "hierarchy mixing mp=16 with another scheme is refused rather "
            "than diagnosed across the edge.",
        ],
    }

SBU_YLIN_OPTION_ID = "sbu-ylin-mp13"

def _sbu_ylin_mp13(registry: dict) -> None:
    """Register SBU-YLin (mp_physics=13) at the maturity its evidence earns.

    ``implemented: true``: ``woof/core/kernels/sbu_ylin.cu`` is WRF v4.6.1
    ``phys/module_mp_sbu_ylin.F`` transcribed in its own operation order,
    launched by the table-admitted column door
    (``woof/core/column_microphysics.py``) from the ``column_kernel``
    consumer row; there is no adapter module.

    ``maturity: supported``: committed native synthetic oracle parity,
    strict replay of real native inputs, GPU state-consumer checks and one
    three-hour native/WOOF comparison are recorded in the public receipt.
    The observation review is visual and supplies no general skill score.

    ``reachability: component-override``: no template registers it.
    """

    options = registry["components"]["microphysics"]["options"]
    options[SBU_YLIN_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3029 binds mp_physics==13 to "
                "the sbu_ylinscheme package: moist qv,qc,qr,qi,qs and state "
                "rimi (Registry.EM_COMMON:491, IO irh: input, restart, "
                "history; not advected)"),
            "prognostic_species": {
                "transported": ["qi", "qs"],
                "not_ported": [],
                "note": (
                    "Five mass classes, no graupel: the snow class carries "
                    "partially rimed snow and graupel, and the riming "
                    "intensity rimi, diagnosed from cloud water and snow, "
                    "sets its mass, area and fall-speed laws. rimi is "
                    "written on every call and never read by the scheme. "
                    "It remains unadvected state, with input, restart and history IO."),
            },
            "native_source": {"release": "WRF v4.7.1 (identical to v4.6.1)",
                "sha256": "d06a0b8fa01465c09ac218f94e2afc409b4ef3a65cf5782fc79cbe8a9d25e06f",
                "license": "WRF public-domain notice; no external tables"},
            "verification_receipt": "docs/public/receipts/sbu-ylin-mp13-20261101.json",
            "column_oracle": {
                "tool": "tools/sbu_ylin_wrf461_oracle",
                "fixture": "woof/data/sbu_ylin/oracle",
                "strict_arm": ("GPUWM_WRF_EXACT=1: 0 ULP on every output "
                               "word against the unmodified native build"),
            },
        },
        "implemented": True,
        "physics_identity_extension": {"transitions": True},
        "label": "SBU-YLin five-class with riming intensity / MP13",
        "maturity": "supported",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 13},
        "warnings": [
            "SINGLE-CASE EVIDENCE: strict real-input replay matches all "
            "49,680 native output words from 144 columns; one three-hour "
            "real forecast was compared with native WRF and temperature "
            "observations by eye. This is no general forecast-skill claim. "
            "See docs/public/receipts/sbu-ylin-mp13-20261101.json.",
            "Native process formulas and the native zero SR are preserved. "
            "Historical declared-fix fixtures are negative controls only.",
            "RADIATION: SBU-YLin supplies no effective radii, so it runs "
            "with the legacy RRTMG arm (ra_rrtmg_variant='rrtmg_legacy'), "
            "which computes radii itself as WRF does; RTE+RRTMGP is refused "
            "by name (woof.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING[13]).",
        ],
    }


GSFCGCE_OPTION_ID = "gsfcgce-mp97"

#: The registry warning that states mp=97's evidence.  Held as a constant so
#: tests/test_gsfcgce_admission.py can hold it to the oracle receipt.
GSFCGCE_EVIDENCE_WARNING = (
    "REAL-INPUT MODULE REPLAY, 2026-11-01. Strict CUDA matches all 69,646 "
    "output words, including melting reflectivity, from four real WRF "
    "4.7.1 calls at forecast step 60 with gsfcgce_hail=0 and "
    "gsfcgce_2ice=0. A three-hour real forecast through woof go was "
    "viewed at hours 1, 2 and 3 beside native WRF and observed radar. "
    "This is a qualitative forecast comparison, not a bitwise whole-model "
    "trajectory or an observation skill score. An additional native hail "
    "capture matches 118,104 words. All eight switch combinations match "
    "944,832 words from native module calls on those real captured inputs, "
    "with only two witnessed undefined-read repairs (ice2=1 radar graupel "
    "and ice2=3 transport coefficients). Deterministic ice2=2 fallout "
    "follows unmodified WRF. See tools/gsfcgce_wrf461_oracle/REAL-INPUT.md and "
    "SOURCE.md for commands, scope and limitations.")


#: The two witnessed native undefined reads repaired, as registry warnings.  The text
#: is woof.core.gsfcgce_constants.DECLARED_DIVERGENCES, read rather than
#: respelled.
def _gsfcgce_divergence_warnings() -> list[str]:
    from woof.core.gsfcgce_constants import DECLARED_DIVERGENCES
    return [f"DECLARED DIVERGENCE, {name}: {why}."
            for name, why in DECLARED_DIVERGENCES]


def _gsfcgce_mp97(registry: dict) -> None:
    """Register Goddard GCE three-ice (mp_physics=97).

    ``implemented: true`` -- the column kernel is
    ``woof/core/kernels/gsfcgce.cu`` (fall_flux and saticel_s of WRF
    v4.6.1 ``phys/module_mp_gsfcgce.F``, transcribed in operation order),
    its constants ``woof/core/gsfcgce_constants.py``, and it is admitted
    as ONE ROW of ``woof.core.mp_column_schemes.MP_COLUMN_SCHEMES`` --
    there is no per-scheme adapter module; ``microphysics.py``'s table arm
    dispatches every row.

    ``maturity: implemented-unverified`` -- the column is measured against
    WRF's own Fortran (the warning says how), but no matched ArWen-versus-
    WRF forecast trajectory exists, which is what the next rung asks.

    ``reachability: component-override`` -- no template selects it; a bare
    configuration reaches it by setting mp_physics = 97.
    """

    options = registry["components"]["microphysics"]["options"]
    options[GSFCGCE_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3046 binds mp_physics==97 to "
                "the gsfcgcescheme package: moist qv,qc,qr,qi,qs,qg"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg"],
                "not_ported": [],
                "note": (
                    "Single-moment: cloud water, rain, cloud ice, snow and "
                    "one dense-ice class whose meaning gsfcgce_hail selects "
                    "(graupel or hail). Nothing in the WRF package is left "
                    "out."),
            },
            "table_admission": (
                "woof.core.mp_column_schemes.MP_COLUMN_SCHEMES[97]: unit, "
                "species, switches, constant table, reflectivity and "
                "restart identity in one row; no adapter module"),
            "switches": {
                "gsfcgce_hail": "0 graupel (WRF default), 1 hail",
                "gsfcgce_2ice": ("0 ice+snow+graupel/hail (WRF default), "
                                 "1 ice+snow, 2 ice+graupel, 3 warm rain "
                                 "only"),
            },
        },
        "implemented": True,
        "label": "Goddard GCE three-ice / MP97",
        "maturity": "implemented-unverified",
        "parameters": {
            "moist": True,
            "moist_cq": True,
            "gsfcgce_hail": 0,
            "gsfcgce_2ice": 0,
        },
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 97},
        "warnings": [GSFCGCE_EVIDENCE_WARNING,
                     *_gsfcgce_divergence_warnings(),
                     "CLOUD OPTICS: WRF hands radiation no GCE radii "
                     "(GSFCGCESCHEME is absent from the use_mp_re "
                     "disjunction, module_physics_init.F:1004-1023), so the "
                     "legacy RRTMG arm computes its own as WRF does; the "
                     "RTE+RRTMGP arm, which has no WRF referent, radiates "
                     "the scheme's ice, snow and cloud water at the fixed "
                     "10 um liquid and 25 um ice radii of its 'gsfcgce' "
                     "coupling with ice-active cloud fraction. A mixed "
                     "nest edge touching mp=97 is refused (no cross-scheme "
                     "entry closure is ratified); a same-scheme nest runs."],
    }


WSM6R_OPTION_ID = "wsm6r-mp106"


def _wsm6r_mp106(registry: dict) -> None:
    """Register WSM6R (mp_physics=106) at exactly the maturity it has earned.

    ``implemented: true`` -- the column kernel is
    ``woof/core/kernels/wsm6r.cu``, a word-for-word port of WRF v4.6.1
    ``phys/module_mp_wsm6r.F``; the WSM6 family module
    ``woof/core/wsm6.py`` launches it as its second row (no adapter file of
    its own); ``woof/core/microphysics.py`` dispatches ``mp_physics in
    (6, 106)`` there; the namelist importer maps 106 natively.

    ``maturity: implemented-unverified`` -- the ladder's rung for a runtime
    with no matched ArWen-versus-WRF FORECAST trajectory.  The column-level
    evidence is stronger than that rung's floor: a 66-column, five-time-step
    oracle against WRF's own module, 0 differing finite words under strict
    arithmetic (tests/test_wsm6r_wrf461_column_exact.py), which is
    conformance evidence and not a matched forecast trajectory.

    ``reachability: component-override`` -- no template selects 106.
    """

    options = registry["components"]["microphysics"]["options"]
    options[WSM6R_OPTION_ID] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3022 binds mp_physics==106 to "
                "the wsm6rscheme package: moist qv,qc,qr,qi,qs,qg"),
            "prognostic_species": {
                "transported": ["qi", "qs", "qg"],
                "not_ported": [],
                "note": (
                    "WSM6R is the revised WSM6 with WSM6's six-class mass "
                    "set and no number moment. Nothing in the WRF package "
                    "is left out."),
            },
            "family": (
                "WSM6 family: woof/core/wsm6.py WSM6_FAMILY row 106 and "
                "woof/core/wsm6_constants.py WSM6R_INIT_F32 (wsm6rinit's "
                "float32 words, copied from WRF's own init, never "
                "recomputed)"),
        },
        "implemented": True,
        "label": "WSM6R revised WSM6 / MP106",
        "maturity": "implemented-unverified",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 106},
        "warnings": [
            "CONFORMANCE EVIDENCE, COLUMN LEVEL: the CUDA kernel is graded "
            "word for word against WRF v4.6.1's unmodified "
            "phys/module_mp_wsm6r.F (gfortran 13.3.0 -O0, glibc 2.39) over "
            "66 columns x 45 levels and five time steps (30, 60, 150, 240 "
            "and 300 s, one to three minor loops) by "
            "tests/test_wsm6r_wrf461_column_exact.py. No matched forecast "
            "trajectory against WRF exists, which is why the rung is "
            "implemented-unverified.",
            "DECLARED DIVERGENCE (NaN payload): WRF itself becomes "
            "nonfinite in the thin-layer dt150 edge. NaN sign and payload "
            "may differ; raw hashes are retained and every finite word and "
            "nonfinite location is still graded in strict and default mode.",
            "DECLARED DIVERGENCE (RAINNCV): WRF's wsm6r overwrites RAINNCV "
            "with the current minor loop's fallout and only when it exceeds "
            "qmin, and the driver never zeroes it, so a multi-loop step "
            "reports its last loop and a dry step repeats the previous "
            "step's rain. woof zeroes it per call and adds every loop's "
            "word, the word WRF adds to RAINNC; RAINNC is WRF's word for "
            "word.",
            "GPUWM ADDITIONS, NOT WRF: WRF's wsm6r produces no REFL_10CM, no "
            "SR, no snow or graupel accumulation and no effective radius. "
            "woof diagnoses REFL_10CM with refl10cm_wsm6 and RTE+RRTMGP "
            "radii with WSM6's mp_wsm6_effectRad_run (graded word for word "
            "against WRF's routine on the WSM6R state by the same oracle), "
            "because WSM6R's size-distribution PARAMETERs are WSM6's; the "
            "legacy RRTMG arm computes its own radii as WRF does "
            "(has_reqc = 0). With no scheme SR the land surface takes "
            "WRF's no-SR temperature proxy, and all fallout reaches the "
            "surface as RAINNC, as in WRF.",
        ],
    }

#: Why a microphysics scheme has no RTE+RRTMGP cloud-optics row, one
#: entry per selector that lacks one.  MEMBERSHIP is derived from
#: ``woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME``; only the WRF reason is
#: written here, because the reasons genuinely differ and a generic one
#: would be worse than none.  A selector with no row and no entry here
#: STOPS THE BUILD -- that is the whole point, and it is what mp=50 needed
#: and did not get.
_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON: dict[int, str] = {
    96: "MAD-WRF declares no effective radii or particle moments; use native legacy RRTMG calculated radii",
    # EMPTY: every implemented scheme has a cloud-optics row.  The table
    # stays because the pass below still needs it the day a scheme is
    # implemented without one, and because tests/test_rrtmgp_coupling.py
    # pins it equal to the runtime's own exclusion set.
    # mp=9's entry RETIRED with the defect it described.  Milbrandt-Yau
    # now HAS a cloud-optics row (``9: "milbrandt2"`` in
    # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME): the scheme's own radii,
    # the block WRF ships commented out (module_mp_milbrandt2mom.F:
    # 3351-3378) evaluated over the transported number moments.  The
    # entry's own words -- Morrison's row "would derive the radii from a
    # gamma distribution that is not this scheme's" -- argued for a row of
    # the scheme's own, not for a refusal, and with ra_rrtmg_variant
    # defaulting to rte-rrtmgp the refusal fired on every bare mp=9 run.
    # mp=50's entry RETIRED 2026-08-31 with the defect it described.  P3
    # now HAS a cloud-optics row (``50: "p3"`` in
    # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME), transcribed from WRF's
    # own coupling: has_reqc=1, has_reqi=1, has_reqs=0, and the wrappers'
    # P3 species remap that hands the single ice category to the snow slot
    # at P3's own ice radius rather than consuming a snow radius that does
    # not exist.  Membership below is DERIVED from that dict, so the
    # selector no longer reaches this table and an entry left here would
    # be a reason for a refusal nothing emits.
}

#: The other way out of this pairing, stated in the REASON rather than in
#: the remedy label.
#:
#: A remedy label is printed as the title of a repair a front end can
#: APPLY, above a button that applies ``remedy_settings`` -- so a label
#: naming a second edit the remedy does not perform tells the reader the
#: button does something it does not.  Measured at
#: woof/companion_physics.py: the offered repair sets
#: ra_rrtmg_variant='rrtmg_legacy' and nothing else, while its title also
#: offered the Dudhia pair.  The alternative is real and belongs in the
#: refusal, so it stays -- one clause earlier.


FERRIER_OPTION_ID = "ferrier-aligo-mp5"


def _ferrier_mp5(registry: dict) -> None:
    """Register Ferrier-Aligo (mp_physics=5) at the maturity it has earned.

    ``implemented: true``: ``woof/core/kernels/fer_hires.cu`` carries WRF
    v4.6.1 ``phys/module_mp_fer_hires.F`` statement by statement, the
    adapter is ``woof/core/fer_hires.py``, ``woof/core/microphysics.py``
    dispatches on ``mp_physics == 5``, the state carries qi/qt and the
    three f_* fields, and the namelist importer maps 5 natively.

    ``maturity: implemented-unverified``: the column kernel is graded at 0
    ULP against the stock WRF column oracle when verified. This maturity
    does not assert a matched forecast trajectory.

    ``reachability: component-override``: no template registers it.
    """
    from woof.core import fer_hires

    blob = (MODEL / "woof" / "data" / "ferrier" / "tables"
            / fer_hires.TABLE_NAME).read_bytes()
    options = registry["components"]["microphysics"]["options"]
    options[FERRIER_OPTION_ID] = {
        "asset_requirements": [{
            "assets": [{"bytes": len(blob),
                        "filename": fer_hires.TABLE_NAME,
                        "sha256": fer_hires.TABLE_SHA256}],
            "id": "wrf-v4.6.1-ferrier-etampnew-expanded-rain-v1",
            "kind": "bundled-package-data",
            "note": (
                "fer_hires_init READS this file "
                "(module_mp_fer_hires.F:2969-2983) and every rain and ice "
                "rate is a lookup into it; nothing regenerates it, so a "
                "missing or byte-modified table is refused by the SHA-256 "
                "check in woof.core.fer_hires.load_table."),
            "redistributed_by_gpuwm": True,
            "root_environment_override": "WOOF_FERRIER_TABLE_ROOT",
            "source": (
                "WRF v4.6.1 (git tag v4.6.1, commit "
                "d66e442fccc04111067e29274c9f9eaccc3cef28), file "
                "run/ETAMPNEW_DATA.expanded_rain"),
        }],
        "constraints": {"required_settings": {"moist": True}},
        "extensions": {
            "wrf_package": (
                "Registry/Registry.EM_COMMON:3019 binds mp_physics==5 to the "
                "fer_mp_hires package: moist qv,qc,qr,qi, scalar qt, state "
                "f_ice_phy,f_rain_phy,f_rimef_phy"),
            "prognostic_species": {
                "transported": ["qi"],
                "not_ported": [],
                "note": (
                    "qi is the total ice (cloud ice and precipitating ice). "
                    "qt is rebuilt from qc+qr+qi on every FER_HIRES entry "
                    "and is carried, not transported; f_ice/f_rain/f_rimef "
                    "are WRF state, never advected, and f_rimef crosses "
                    "steps and checkpoints."),
            },
            "family_siblings_refused": {
                "eta_ferrier_mp95": (
                    "not ported; the old Eta scheme's qs holds the total "
                    "ice and its RRTMG and cal_cldfra1 arms are weighted by "
                    "F_ICE_PHY"),
            },
        },
        "implemented": True,
        "label": "Ferrier-Aligo / MP5",
        "maturity": "implemented-unverified",
        "parameters": {"moist": True, "moist_cq": True},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"mp_physics": 5},
        "warnings": [
            "The strict column check is tools/ferrier_hires_wrf461_oracle/"
            "check_port.py. No forecast skill claim is made. Stock WRF's "
            "Rcw arithmetic order is retained.",
            "REFLECTIVITY: WRF v4.6.1 computes Ferrier's DBZ_col and "
            "discards it, leaving REFL_10CM unwritten under mp=5.  woof "
            "writes that column (clear air -20 dBZ) as REFL_10CM; its "
            "10*ALOG10 is CUDA's log10f, so the diagnostic alone is outside "
            "the 0 ULP claim.",
            "NEST AND OFFLINE CHILD: no cross-scheme nest edge is ported "
            "for mp=5 and the offline child refuses a Ferrier parent (no "
            "F_RIMEF_PHY wrfout row); same-scheme online nests run.",
        ],
    }


def _ferrier_mp15(registry: dict) -> None:
    """One shared Ferrier kernel; selector 15 transports mass-weighted rime."""
    options = registry["components"]["microphysics"]["options"]
    base = options[FERRIER_OPTION_ID]
    package5 = {"variant": "state_fractions",
                "state_fields": ["qi", "qt", "f_ice", "f_rain", "f_rimef"],
                "transported": ["qi"], "initial_values": {"f_rimef": 1.0}}
    package15 = {"variant": "advected_rime", "no_fmad": False,
                 "state_fields": ["qi", "qrimef"], "transported": ["qi", "qrimef"],
                 "initial_values": {"qrimef": 0.0}}
    base["extensions"]["ferrier_hires"] = package5
    row = copy.deepcopy(base)
    row.update(label="Ferrier-Aligo advected rime / MP15",
               selectors={"mp_physics": 15}, warnings=[
        "Strict real-native replay: 3,212,040 FP32 words, zero differences. "
        "The WRF v4.7.1 reference needs three caller corrections; its column "
        "and table equal v4.6.1. See tools/ferrier_hires_wrf461_oracle/advect/"
        "REAL-REPLAY.md for the patches and three-hour forecast review. "
        "This is not a general forecast-skill certification.",
        "Both selectors use the qualified mp=5 dependency's stock WRF "
        "DUM2 operation order for Rcw and invalid-state budget operands.",
        "QRIMEF is qi times rime factor, transported as a scalar and excluded "
        "from total-water mass. QT and F_* are local to each physics call.",
        "Reflectivity is the shared kernel diagnostic and is outside the "
        "prognostic 0 ULP claim."])
    row["extensions"]["ferrier_hires"] = package15
    row["extensions"]["wrf_package"] = (
        "Registry/Registry.EM_COMMON:3024: fer_mp_hires_advect, "
        "moist:qv,qc,qr,qi;scalar:qrimef")
    row["extensions"]["prognostic_species"] = {
        "transported": ["qi", "qrimef"], "not_ported": [],
        "note": "QRIMEF/QI derives rime factor at entry; QI*F_RIMEF_PHY "
                "writes QRIMEF after qi/qc/qr repartition (:615, :690)."}
    options["ferrier-aligo-advect-mp15"] = row
    _MOMENT_ROWS[15] = copy.deepcopy(_MOMENT_ROWS[5])
    _MOMENT_ROWS[15]["name"] = "Ferrier-Aligo advected rime"
    _MOMENT_ROWS[15]["unpaired"] = ["qrimef"]
    _OFFLINE_CHILD_ROWS[15] = {"same_scheme": True, "refusal": None}
    _RING_STATE_ROWS[15] = ["thp", "qv", "qc", "qr", "qi", "qrimef"]
    _RING_SURFACE_ROWS[15] = list(_RING_SURFACE_ROWS[5])
    _VERTICAL_BOUND_SOURCES["microphysics"]["ferrier-aligo-advect-mp15"] = (
        "Ferrier-Aligo advected rime", "FERRIER_VERTICAL_LEVEL_BOUNDS")


#: Why a microphysics scheme has no RTE+RRTMGP cloud-optics row, one
#: entry per selector that lacks one.  MEMBERSHIP is derived from
#: ``woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME``; only the WRF reason is
#: written here, because the reasons genuinely differ and a generic one
#: would be worse than none.  A selector with no row and no entry here
#: STOPS THE BUILD -- that is the whole point, and it is what mp=50 needed
#: and did not get.
_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON: dict[int, str] = {
    96: "MAD-WRF declares no effective radii or particle moments; use native legacy RRTMG calculated radii",
    56: "NTU multi-moment hands radiation no radii (absent from WRF's "
         "use_mp_re disjunction, module_physics_init.F:1004-1023) and no "
         "existing RTE+RRTMGP row is the scheme's own; the legacy RRTMG "
         "arm computes its own radii exactly as WRF does",
    2: "Lin supplies no effective radii to this adapter; borrowing another scheme would change its cloud distribution",
    30: "the bin effective-radius integrals are not qualified; a bulk radius closure would replace the spectral distribution optics",
    # SBU-YLin: no radii (has_req = 0, absent from module_physics_init.F's
    # use_mp_re list) and no number moments, so no coupling row is its
    # own; woof.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING[13] says the same.
    13: ("SBU-YLin (mp_physics=13) hands radiation no effective radii "
         "and carries no number moments, so the RTE+RRTMGP adapter has "
         "no cloud-optics row that is this scheme's own and would "
         "otherwise size its ice and snow with another scheme's radii."),
    # EMPTY: every implemented scheme has a cloud-optics row.  The table
    # stays because the pass below still needs it the day a scheme is
    # implemented without one, and because tests/test_rrtmgp_coupling.py
    # pins it equal to the runtime's own exclusion set.
    # mp=9's entry RETIRED with the defect it described.  Milbrandt-Yau
    # now HAS a cloud-optics row (``9: "milbrandt2"`` in
    # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME): the scheme's own radii,
    # the block WRF ships commented out (module_mp_milbrandt2mom.F:
    # 3351-3378) evaluated over the transported number moments.  The
    # entry's own words -- Morrison's row "would derive the radii from a
    # gamma distribution that is not this scheme's" -- argued for a row of
    # the scheme's own, not for a refusal, and with ra_rrtmg_variant
    # defaulting to rte-rrtmgp the refusal fired on every bare mp=9 run.
    # mp=50's entry RETIRED 2026-08-31 with the defect it described.  P3
    # now HAS a cloud-optics row (``50: "p3"`` in
    # woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME), transcribed from WRF's
    # own coupling: has_reqc=1, has_reqi=1, has_reqs=0, and the wrappers'
    # P3 species remap that hands the single ice category to the snow slot
    # at P3's own ice radius rather than consuming a snow radius that does
    # not exist.  Membership below is DERIVED from that dict, so the
    # selector no longer reaches this table and an entry left here would
    # be a reason for a refusal nothing emits.
}

#: The other way out of this pairing, stated in the REASON rather than in
#: the remedy label.
#:
#: A remedy label is printed as the title of a repair a front end can
#: APPLY, above a button that applies ``remedy_settings`` -- so a label
#: naming a second edit the remedy does not perform tells the reader the
#: button does something it does not.  Measured at
#: woof/companion_physics.py: the offered repair sets
#: ra_rrtmg_variant='rrtmg_legacy' and nothing else, while its title also
#: offered the Dudhia pair.  The alternative is real and belongs in the
#: refusal, so it stays -- one clause earlier.
_RTE_RRTMGP_CLOUD_OPTICS_ALTERNATIVE = (
    "The Dudhia pair ra_lw_physics=0 / ra_sw_physics=1 is the other way "
    "out of this pairing.")

#: The remedy sentence both refusals end on, and the one edit
#: ``_RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS`` performs, in words.  One
#: copy, because it is the same two doors and a user who is told different
#: things about the same door stops trusting either.
_RTE_RRTMGP_CLOUD_OPTICS_REMEDY = (
    "Set ra_rrtmg_variant='rrtmg_legacy', which computes its own radii "
    "the way WRF does.")

#: The remedy's MACHINE half: the smallest edit that clears the refusal.
#:
#: A rule that carries only prose makes every front end re-derive the way
#: out from the sentence.  woof/companion_physics.py did exactly that --
#: it matched "mp_physics=9" and "cloud-optics" in the refusal text to
#: decide both the summary it printed and the one repair it offered first
#: -- so a scheme that gains this refusal gets no tailored repair, and a
#: reworded sentence silently loses the one that exists.  Written here
#: beside the prose it belongs to, the rule answers "what do I change"
#: for every scheme this pass covers, including ones added later.
_RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS = {"ra_rrtmg_variant": "rrtmg_legacy"}


def registry_setting_names(registry: dict) -> set[str]:
    """Every setting name this registry can talk about.

    The parameter table alone is NOT that set, and assuming it was made a
    guard that could not see its own defect: ``ra_lw_physics`` is a
    component SELECTOR, not a parameter, so a remedy label naming the
    Dudhia pair passed a check written to catch exactly that label.  The
    selector keys each component declares, and the selectors and
    parameters its options carry, are the rest of the vocabulary.
    """

    names = set(registry.get("parameters", {}))
    for component in registry.get("components", {}).values():
        names.update(component.get("selector_keys", []) or [])
        for option in component.get("options", {}).values():
            names.update(option.get("selectors", {}) or {})
            names.update(option.get("parameters", {}) or {})
    return names


def _check_remedy_label_describes_its_edit(
        rule: dict, registry: dict, option_id: str) -> None:
    """A remedy's words and a remedy's edit must be the same remedy.

    ``remedy_label`` is printed as the TITLE of an applicable repair --
    woof/companion_physics.py offers it above a button that applies
    ``remedy_settings``, and the desktop panel renders that title -- so a
    label naming a parameter the edit does not touch offers a way out the
    button does not take.  That is what shipped: the label named both
    ra_rrtmg_variant and the Dudhia pair while the edit set only the
    variant.

    Both directions are checked, over the registry's OWN parameter names,
    so this holds for every rule any later pass writes rather than for the
    one it was found on.
    """

    label = str(rule.get("remedy_label") or "")
    settings = rule.get("remedy_settings") or {}
    if not label or not settings:
        return
    missing = sorted(name for name in settings if name not in label)
    if missing:
        raise RuntimeError(
            f"microphysics/{option_id}: the remedy label does not name "
            f"{', '.join(missing)}, which its remedy_settings edits. A "
            "reader offered this repair cannot tell what it changes.")
    extra = sorted(
        name for name in registry_setting_names(registry)
        if name in label and name not in settings)
    if extra:
        raise RuntimeError(
            f"microphysics/{option_id}: the remedy label names "
            f"{', '.join(extra)}, which its remedy_settings does not set, "
            "so a front end that applies the remedy performs a smaller "
            "edit than the title it printed. State the other way out in "
            "the reason instead.")


def _rte_rrtmgp_cloud_optics_constraints(registry: dict) -> None:
    """Refuse, in the registry, every scheme RTE+RRTMGP cannot couple to.

    THE FAILURE THIS CLOSES.  woof/core/rrtmgp.py resolves a cloud-optics
    row per ``mp_physics`` and RAISES on a selector with no row -- correct,
    and it raises at the FIRST RADIATION CALL, deep inside a forecast.  Two
    schemes reached 1.9 admitted by the loader with no row: mp=9, whose
    refusal existed in woof/config.py but only as prose in the registry
    (tests/test_authority_agreement.py measured 320 combinations the two
    authorities decided differently), and mp=50, which had no refusal
    anywhere and would simply die mid-run.

    IT IS A CONJUNCTION, so it needs ``constraints.refused_when``.  A plain
    ``requires_components`` on radiation is WRONG here and was tried first:
    the three shipped ``*-rrtmg-legacy-*`` templates select the SAME
    ``rte-rrtmgp`` component option and switch adapters with the
    ``ra_rrtmg_variant`` PARAMETER, so excluding the option refuses the
    very configuration the refusal message recommends -- measured, 16
    disagreements in the other direction.

    DERIVED, not transcribed, in all three of its parts: which schemes need
    the rule comes from ``_MP_CLOUD_OPTICS_SCHEME`` itself, the radiation
    clause from the options' resolved selectors, and the variant clause
    from the shipped
    ``ra_rrtmg_variant`` enum minus the legacy value.  A new scheme, a new
    radiation option resolving to the same adapter, or a third adapter each
    force a decision here instead of silently widening the admission.
    """

    from types import SimpleNamespace

    from woof.config import radiation_scheme_ids
    from woof.core.rrtmgp import _MP_CLOUD_OPTICS_SCHEME

    radiation_options = registry["components"]["radiation"]["options"]
    rte_rrtmgp_4_4 = sorted(
        option_id for option_id, option in radiation_options.items()
        if radiation_scheme_ids(SimpleNamespace(
            **(option.get("parameters", {}) | option.get("selectors", {}))
        )) == (4, 4))
    if not rte_rrtmgp_4_4:
        raise RuntimeError(
            "no radiation option projects the RTE+RRTMGP 4/4 selector pair; "
            "the cloud-optics constraints would be vacuous.  If the adapter "
            "moved, re-derive from its new spelling rather than deleting "
            "the constraints.")
    variants = registry["parameters"]["ra_rrtmg_variant"]["enum"]
    rte_rrtmgp_variants = sorted(
        value for value in variants if value != "rrtmg_legacy")
    if not rte_rrtmgp_variants:
        raise RuntimeError(
            "ra_rrtmg_variant declares no non-legacy adapter; the "
            "cloud-optics constraints would be vacuous")
    if _RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS["ra_rrtmg_variant"] not in (
            variants):
        raise RuntimeError(
            "the cloud-optics remedy names an ra_rrtmg_variant value this "
            "registry does not declare, so every front end that applies it "
            "would offer a way out the parser refuses.  Re-derive the "
            "remedy from the adapter's new spelling rather than shipping "
            "one nothing accepts.")

    for option_id, option in sorted(
            registry["components"]["microphysics"]["options"].items()):
        if option.get("implemented") is not True:
            continue
        selector = option.get("selectors", {}).get("mp_physics")
        if selector is None:
            continue
        if selector in _MP_CLOUD_OPTICS_SCHEME:
            # DERIVED MEANS DERIVED IN BOTH DIRECTIONS.  This builder is a
            # transform over the TRACKED registry, not a render from
            # scratch: everything these passes do not touch is carried
            # through from the file on disk.  So a scheme that GAINS a
            # cloud-optics row -- which is what retiring the defect looks
            # like -- kept publishing the refusal it was fixed out of,
            # because the pass only ever wrote rows and never removed
            # them.  MEASURED at mp=50: the row ``50: "p3"`` landed, the
            # config-door and namelist-report guards were retired with it,
            # and the registry still told every reader P3 "has no
            # RTE+RRTMGP cloud-optics coupling" through a full rebuild.
            # A refusal the engine no longer makes is a refusal that
            # refuses working configurations, so this pass drops the
            # entries it owns once their reason is gone.
            constraints = option.get("constraints")
            if not constraints:
                continue
            kept = [entry for entry in constraints.get("refused_when", [])
                    if "ra_rrtmg_variant" not in entry.get("settings", {})]
            if kept:
                constraints["refused_when"] = kept
            else:
                constraints.pop("refused_when", None)
                if not constraints:
                    option.pop("constraints", None)
            continue
        reason = _NO_RTE_RRTMGP_CLOUD_OPTICS_REASON.get(selector)
        if reason is None:
            raise RuntimeError(
                f"microphysics/{option_id} (mp_physics={selector}) has no "
                "row in woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME, so an "
                "RTE+RRTMGP run of it dies at the first radiation call. "
                "Either add the cloud-optics row with its WRF authority, "
                "or add the scheme's reason to "
                "_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON so the registry can "
                "refuse the pairing up front.")
        rule = {
            "components": {"radiation": rte_rrtmgp_4_4},
            "reason": f"{reason} {_RTE_RRTMGP_CLOUD_OPTICS_ALTERNATIVE}",
            "remedy_label": _RTE_RRTMGP_CLOUD_OPTICS_REMEDY,
            "remedy_settings": dict(
                _RTE_RRTMGP_CLOUD_OPTICS_REMEDY_SETTINGS),
            "settings": {"ra_rrtmg_variant": rte_rrtmgp_variants},
        }
        _check_remedy_label_describes_its_edit(rule, registry, option_id)
        option.setdefault("constraints", {})["refused_when"] = [rule]


# ------------------------------------------------------------ consumer rows
#: Which PBL option publishes the lower-interface heat diffusivity that WRF's
#: post-PBL ``diff4d`` (scalar_pblmix=1, module_pbl_driver.F:2251-2259) reads,
#: and where.  The config validator, the physics coupling and VRAM admission
#: read this row instead of a hand-kept scheme pair.  ``mixes_in_scheme``
#: names the one producer whose wrapper already runs the solve itself (MYNN,
#: woof/core/mynn_pbl_runtime.py), so the post-PBL owner never runs it twice.
#: Ported from the scalar-pblmix-any-pbl lane (2026-10-07), whose MYJ
#: EXCH_H column check matched stock diff4d at 0 ULP.
_SCALAR_DIFFUSION_EXCHANGE: dict[str, dict[str, object]] = {
    "temf": {"writes_exch_h": False, "levels": None, "field": None,
             "mixes_in_scheme": False,
             "reason": "WRF module_pbl_driver.F:1173-1204 passes KH_TEMF "
                       "and KM_TEMF to TEMF, without an EXCH_H argument; "
                       "the scalar_pblmix seam has no TEMF EXCH_H writer."},
    "keps": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
             "mixes_in_scheme": False,
             "reason": "KEPS publishes heat diffusivity at lower interfaces."},
    "off": {"writes_exch_h": False, "levels": None, "field": None,
            "mixes_in_scheme": False,
            "reason": "WRF skips the PBL driver with no PBL, so scalar_pblmix "
                      "mixes nothing."},
    "ysu": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
            "mixes_in_scheme": False,
            "reason": "YSU publishes heat diffusivity at lower interfaces."},
    "mynn": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
             "mixes_in_scheme": True,
             "reason": "MYNN publishes heat diffusivity at lower interfaces; "
                       "its wrapper runs the diff4d solve after the column."},
    "boulac": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
               "mixes_in_scheme": False,
               "reason": "BouLac publishes heat diffusivity at lower interfaces."},
    "myj": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
            "mixes_in_scheme": False,
            "reason": "MYJ publishes heat diffusivity at lower interfaces."},
    "gfs": {"writes_exch_h": False, "levels": None, "field": None,
            "mixes_in_scheme": False,
            "reason": "WRF BL_GFS does not publish EXCH_H. Its internal "
                      "thermal diffusivity is not a PBL-driver output; "
                      "scalar_pblmix therefore has no exchange field."},
    "qnse": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
             "mixes_in_scheme": False,
             "reason": "QNSE publishes heat diffusivity at lower interfaces."},

    "acm2": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
             "mixes_in_scheme": False,
             "reason": "ACMPBL publishes EXCH_H at levels 1..kte-1 and "
                       "preserves the top slot (phys/module_bl_acm.F:539-544)."},
    "shinhong": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
                 "mixes_in_scheme": False,
                 "reason": "Shin-Hong publishes heat diffusivity at lower "
                           "interfaces."},
    "uw": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
           "mixes_in_scheme": False,
           "reason": "UW publishes interface KVH (nz+1); exch_h carries its "
                     "first nz interfaces, the levels diff4d reads."},
    "gbm": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
            "mixes_in_scheme": False,
            "reason": "GBM publishes KTH_GBM at lower interfaces as EXCH_H."},
    "eeps": {"writes_exch_h": True, "levels": "nz", "field": "exch_h",
             "mixes_in_scheme": False,
             "reason": "EEPS publishes heat diffusivity at lower interfaces; "
                       "the PBL driver mixes scalar-array species with diff4d."},
    "mrf": {"writes_exch_h": False, "levels": None, "field": None,
            "mixes_in_scheme": False,
            "reason": "WRF's MRF driver arm has no EXCH_H argument "
                      "(module_pbl_driver.F:1352-1382). Its private XKZH "
                      "does not leave the column; the initialized EXCH_H "
                      "remains zero and post-PBL diff4d mixes nothing."},
    "ccpp-tke-edmf": {
        "writes_exch_h": False, "levels": None, "field": None,
        "mixes_in_scheme": False,
        "reason": "CCPP TKE-EDMF has no WRF PBL-driver arm, and its own "
                  "column solve already mixes every transported scalar "
                  "(nc/ni/nwfa/nifa included, physics._run_satmedmf), so "
                  "scalar_pblmix adds no second diff4d."},
    "sase": {"writes_exch_h": False, "levels": None, "field": None,
             "mixes_in_scheme": False,
             "reason": "SASE has no WRF counterpart and publishes a layer "
                       "diffusivity for its own split solve, not lower-"
                       "interface EXCH_H; reading it as EXCH_H would shift "
                       "the vertical fluxes, so scalar_pblmix mixes nothing."},
}

#: Stock adapter classes the checkpoint writer recognises, by component and
#: option id.  These used to live as two literal dicts INSIDE
#: woof/io/restart.py::physics_setup_identity, where a scheme with no row
#: was sent down the custom-callable path with a message about declaring a
#: restart_identity -- true of nothing the tree ships.  The 4/4 radiation
#: options carry one class per ``ra_rrtmg_variant`` because one selector
#: pair names two adapters.
_STOCK_CALLABLE_CLASSES: dict[str, dict[str, object]] = {
    "cumulus": {
        "cam-deep": "woof.core.cam5_runtime.CAMDeepConvection",
        "off": None,
        "kain-fritsch": "woof.core.kf.KainFritsch",
        "kf-cup": "woof.core.kf.KFCumulusPotential",

        "multiscale-kf": "woof.core.column_cumulus.MskfColumnCumulus",

        KF_OLD_OPTION_ID: "woof.core.kf.KainFritsch",
        "grell-freitas": "woof.core.gf.GrellFreitas",
        "new-tiedtke": "woof.core.ntiedtke.NewTiedtke",
        "tiedtke": "woof.core.tiedtke.Tiedtke",
        "grell-3d": "woof.core.g3.Grell3D",
        "gfs-sas": "woof.core.sas.GfsSas",
        "new-sas": "woof.core.column_cumulus.ColumnCumulus",
        "kiaps-sas": "woof.core.sas.SasCumulus",
        # The table-driven column-kernel route: one class for every scheme
        # that publishes a ``column_kernel`` row (below).  The checkpoint
        # binds the scheme through restart_algorithm_identity and the
        # scheme id, so sharing the class names nothing ambiguous.
        "betts-miller-janjic": "woof.core.column_cumulus.RegistryColumnCumulus",
        "old-gfs-sas": "woof.core.column_cumulus_r8.ColumnCumulus",
        "grell-devenyi": "woof.core.gf.GrellDevenyi",
    },
    "radiation": {
        "relaxation-31": "woof.core.radiation_composition.NewtonianRelaxationRadiation",
        "off": None,
        "wrf-cam3": "woof.core.cam3.Cam3Radiation",
        "dudhia-shortwave": "woof.core.dudhia.DudhiaShortwaveRadiation",
        "gsfc-shortwave": "woof.core.gsfcsw.GSFCSWShortwaveRadiation",
        "wrf-rrtm-dudhia": "woof.core.rrtm_lw.RRTMDudhiaRadiation",
        "new-goddard": "woof.core.radiation_composition.ColumnTableRadiation",
        "fu-liou-gu": "woof.core.flg_radiation.FLGRadiation",
        "wrf-gfdleta": "woof.core.gfdleta.GFDLEtaRadiation",
        "analytic-clear-sky":
            "woof.core.analytic_radiation.AnalyticClearSkyRadiation",
        "rte-rrtmgp": {
            "rte-rrtmgp": "woof.core.rrtmgp.RRTMGPRadiation",
            "rrtmg_legacy": "woof.core.rrtmg_legacy.RRTMGLegacyRadiation",
        },
    },
}

#: Component-owned first-call level windows, by option id: the receipt
#: label the vertical preflight reports and the NAME of the contract
#: constant in woof/physics_vertical_contract.py that holds the pair.  The
#: value is read off the constant at build time so the registry cannot
#: carry a number the launcher does not enforce.  ``off`` options carry no
#: window (null).  The four rows the preflight's old if/elif chain never
#: had -- milbrandt2mom-mp9, wdm6-mp16, p3-mp50, new-tiedtke -- are why
#: a 100-level WDM6 run passed ``woof check`` and died on its first
#: microphysics call.
_VERTICAL_BOUND_SOURCES: dict[str, dict[str, tuple[str, str]]] = {
    "radiation": {"wrf-gfdleta": ("GFDL Eta radiation", "GFDLETA_VERTICAL_LEVEL_BOUNDS")},
    "microphysics": {
        "cam-mg-mp11": ("CAMMGMP microphysics", "CAMMGMP_VERTICAL_LEVEL_BOUNDS"),
        'ntu-mp56': ("NTU microphysics", "NTU_VERTICAL_LEVEL_BOUNDS"),
        "kessler-mp1": ("Kessler microphysics", "KESSLER_VERTICAL_LEVEL_BOUNDS"),
        "lin-mp2": ("Lin microphysics", "LIN_VERTICAL_LEVEL_BOUNDS"),
        "ferrier-aligo-mp5": ("Ferrier-Aligo microphysics",
                              "FERRIER_VERTICAL_LEVEL_BOUNDS"),
        "wsm5-mp4": ("WSM5 microphysics", "WSM5_VERTICAL_LEVEL_BOUNDS"),
        "wsm6-mp6": ("WSM6 microphysics", "WSM6_VERTICAL_LEVEL_BOUNDS"),
        "thompson-mp8": ("Thompson microphysics",
                         "THOMPSON_VERTICAL_LEVEL_BOUNDS"),
        "thompson-aerosol-mp28": ("Thompson aerosol-aware microphysics",
                                  "THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS"),
        "morrison-mp10": ("Morrison microphysics",
                          "MORRISON_VERTICAL_LEVEL_BOUNDS"),
        "nssl2-mp18": ("NSSL-2 microphysics", "NSSL2_VERTICAL_LEVEL_BOUNDS"),
        "milbrandt2mom-mp9": ("Milbrandt-Yau microphysics",
                              "MILBRANDT2_VERTICAL_LEVEL_BOUNDS"),
        "wdm5-mp14": ("WDM5 microphysics", "WDM5_VERTICAL_LEVEL_BOUNDS"),
        "wdm6-mp16": ("WDM6 microphysics", "WDM6_VERTICAL_LEVEL_BOUNDS"),
        "wsm3-mp3": ("WSM3 microphysics", "WSM3_VERTICAL_LEVEL_BOUNDS"),
        "wsm6r-mp106": ("WSM6R microphysics", "WSM6R_VERTICAL_LEVEL_BOUNDS"),
        "wsm7-mp24": ("WSM7 microphysics", "WSM7_VERTICAL_LEVEL_BOUNDS"),
        # WDM7's independent unit uses the WDM6_KMAX level tiers.
        "wdm7-mp26": ("WDM7 microphysics", "WDM6_VERTICAL_LEVEL_BOUNDS"),
        "p3-mp50": ("P3 microphysics", "P3_VERTICAL_LEVEL_BOUNDS"),
        "gsfcgce-mp97": ("Goddard GCE microphysics", "GSFCGCE_VERTICAL_LEVEL_BOUNDS"),
        "p3-mp51": ("P3 microphysics", "P3_VERTICAL_LEVEL_BOUNDS"),
        "p3-mp53": ("P3 microphysics", "P3_VERTICAL_LEVEL_BOUNDS"),
        "tempo": ("TEMPO microphysics", "TEMPO_VERTICAL_LEVEL_BOUNDS"),
        "sbu-ylin-mp13": ("SBU-YLin microphysics",
                          "SBU_YLIN_VERTICAL_LEVEL_BOUNDS"),
    },
    "pbl": {
        "mynn": ("MYNN PBL", "MYNN_VERTICAL_LEVEL_BOUNDS"),
        "ysu": ("YSU PBL", "YSU_VERTICAL_LEVEL_BOUNDS"),
        "myj": ("MYJ PBL", "MYJ_VERTICAL_LEVEL_BOUNDS"),
        "qnse": ("QNSE PBL", "QNSE_VERTICAL_LEVEL_BOUNDS"),
        "uw": ("UW moist-turbulence PBL", "UWPBL_VERTICAL_LEVEL_BOUNDS"),
        "ccpp-tke-edmf": ("CCPP TKE-EDMF PBL", "SATMEDMF_VERTICAL_LEVEL_BOUNDS"),
        "temf": ("TEMF PBL", "TEMF_VERTICAL_LEVEL_BOUNDS"),
        "gbm": ("GBM moist TKE PBL", "GBMPBL_VERTICAL_LEVEL_BOUNDS"),
        "eeps": ("EEPS PBL", "EEPS_VERTICAL_LEVEL_BOUNDS"),

        "mrf": ("MRF PBL", "MRF_VERTICAL_LEVEL_BOUNDS"),
        "gfs": ("GFS PBL", "GFS_PBL_VERTICAL_LEVEL_BOUNDS"),
        "boulac": ("BouLac PBL", "BOULAC_VERTICAL_LEVEL_BOUNDS"),
        "keps": ("K-epsilon PBL", "KEPS_VERTICAL_LEVEL_BOUNDS"),

        "acm2": ("ACM2 PBL", "ACM2_VERTICAL_LEVEL_BOUNDS"),
        "shinhong": ("Shin-Hong PBL", "SHINHONG_VERTICAL_LEVEL_BOUNDS"),
        "sase": ("SASE PBL", "SASE_VERTICAL_LEVEL_BOUNDS"),
    },
    "cumulus": {
        "cam-deep": ("CAM ZM deep convection", "CAMZM_VERTICAL_LEVEL_BOUNDS"),
        "kain-fritsch": ("Kain-Fritsch cumulus", "KF_VERTICAL_LEVEL_BOUNDS"),
        "kf-cup": ("KF-CuP cumulus", "KFCUP_VERTICAL_LEVEL_BOUNDS"),

        "multiscale-kf": ("Multiscale KF cumulus", "MSKF_VERTICAL_LEVEL_BOUNDS"),

        KF_OLD_OPTION_ID: ("previous Kain-Fritsch cumulus",
                           "KF_VERTICAL_LEVEL_BOUNDS"),
        "grell-freitas": ("Grell-Freitas cumulus", "GF_VERTICAL_LEVEL_BOUNDS"),
        "grell-3d": ("Grell 3D cumulus", "GRELL_3D_VERTICAL_LEVEL_BOUNDS"),
        "new-tiedtke": ("New Tiedtke cumulus",
                        "NEW_TIEDTKE_VERTICAL_LEVEL_BOUNDS"),
        "tiedtke": ("Original cu6 cumulus", "TIEDTKE_VERTICAL_LEVEL_BOUNDS"),
        "gfs-sas": ("GFS SAS cumulus", "SAS_VERTICAL_LEVEL_BOUNDS"),
        "new-sas": ("New SAS cumulus", "NEW_SAS_VERTICAL_LEVEL_BOUNDS"),
        "kiaps-sas": ("KIAPS SAS cumulus", "KSAS_VERTICAL_LEVEL_BOUNDS"),
        "betts-miller-janjic": ("Betts-Miller-Janjic cumulus",
                                "BMJ_VERTICAL_LEVEL_BOUNDS"),
        "old-gfs-sas": ("Old GFS SAS cumulus", "OSAS_VERTICAL_LEVEL_BOUNDS"),
        "grell-devenyi": ("Grell-Devenyi cumulus",
                          "GD_VERTICAL_LEVEL_BOUNDS"),
    },
}

#: ``consumers.column_kernel`` rows: a cumulus scheme whose WRF driver is a
#: per-column flip-call-flip is admitted as THIS metadata and launched by
#: the one table-driven route, woof/core/column_cumulus.py.  The order of
#: each list is the kernel's argument order.  ``tables.sha256`` pins the
#: float32 words the table entry point must produce (the oracle's own
#: BMJINIT dump, tools/bmj_wrf461_oracle); the route refuses to run the
#: scheme on any other words.  ``carried`` names driver fields the scheme
#: reads and rewrites every call, with the cold-start word WRF's init
#: routine writes (BMJINIT: CLDEFI = AVGEFI = (EFIMN+1.)*0.5).
_CUMULUS_COLUMN_KERNEL_ROWS: dict[str, dict] = {
    "betts-miller-janjic": {
        "identity": "betts-miller-janjic-wrf461-bmjdrv-v1",
        "module": "bmj", "entry": "bmj_column",
        "workspace_slots": 20, "block": 128, "vertical_minimum": 4,
        "tables": {
            "entry": "bmj_init_tables", "floats": 87972,
            "grid": 3, "block": 160, "scratch_floats_per_thread": 3080,
            "constants": ["cp", "r_d"],
            "sha256": ("2a4b5858d635b0727a1b399757a67dde"
                       "eae393dc958d464db905cd87f4d198f8"),
        },
        "level_inputs": [["t", "temperature"], ["qv", "qv"],
                         ["pmid", "pressure"], ["pi", "exner"],
                         ["rho", "rho"], ["dz8w", "dz"]],
        "interface_inputs": [["pint", "p_interface"]],
        "column_inputs": [["xland", "xland"]],
        "carried": [{"field": "cldefi", "cold_start": 0.6000000238418579}],
        "level_outputs": ["rthcuten", "rqvcuten", "ccldfra", "qcconv",
                          "qiconv"],
        "column_outputs": ["raincv", "pratec", "htop", "hbot", "convcld"],
        "int_column_outputs": ["bmj_path"],
        "persist_outputs": ["qcconv", "qiconv", "ccldfra", "convcld",
                            "htop", "hbot", "raincv", "pratec"],
        "feedback": {"parameter": "bmj_rad_feedback",
                     "entry": "column_cumulus_cloud_merge",
                     "liquid": "qcconv", "ice": "qiconv",
                     "fraction": "ccldfra"},

        "result": {"rthcuten": "rthcuten", "rqvcuten": "rqvcuten",
                   "rainc": "pratec"},
    },
}

#: Prognostic moment structure per microphysics selector, the row
#: woof/da/moments.py builds its SchemeMoments from.  Names are the
#: DomainState attributes each scheme's allocator gives it
#: (woof/core/state.py, mp arm by mp arm), so the analysis state vector is
#: derived from the scheme and never typed at a call site.
#: ``repair_authority`` names the scheme's OWN q>0/N=0 repair where it is
#: ported (Morrison's PSD limiter); ``None`` means the analysis detects and
#: refuses instead of inventing an intercept.  mp=18 resolves its set from
#: WRF's option-18 consistency pass at call time (woof.core.nssl2_contract)
#: and is marked so rather than enumerated twice.
#: Repair authorities are TOKENS the moments module resolves to its own
#: prose (woof.da.moments._REPAIR_AUTHORITIES), because the prose carries a
#: WRF citation the registry's citation checker cannot resolve.
_MORRISON_REPAIR_AUTHORITY = "morrison-psd-limiter"
#: Thompson's own entry moment-consistency block
#: (module_mp_thompson.F:1827-1899), which sets a number moment from the
#: mass under the scheme's assumed distribution and zeroes both moments
#: below its activity gate R1.  ``zero_number_below_threshold`` records
#: that second half: a scheme that does not state it does not get it.
_THOMPSON_REPAIR_AUTHORITY = "thompson-entry-block"
_MOMENT_ROWS: dict[int, dict | None] = {
    11: {"name":"CAMMGMP", "mass_only":["qv"], "pairs":[
        {"species":n,"mass":m,"number":v} for n,m,v in
        (("cloud","qc","nc"),("ice","qi","ni"),("rain","qr","nr"),("snow","qs","ns"))],
        "unpaired":[],"repair_authority":None,"q_threshold":1.e-14},
    56: {"name": "NTU multi-moment", "mass_only": ["qv"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"},
                   {"species": "snow", "mass": "qs", "number": "ns"},
                   {"species": "graupel", "mass": "qg", "number": "ng"},
                   {"species": "hail", "mass": "qh", "number": "nh"}],
         "unpaired": ["qdcn", "qtcn", "qccn", "qrcn", "qnin", "fi",
                      "fs", "vi", "vs", "vg", "ai", "as_", "ag", "ah",
                      "i3m"],
         "repair_authority": None, "q_threshold": 1.0e-14},
    2: {"name":"Lin", "mass_only":["qv","qc","qr","qi","qs","qg"], "pairs":[], "unpaired":[], "repair_authority":None, "q_threshold":1.0e-14},
    0: None,
    1: {"name": "Kessler", "mass_only": ["qv", "qc", "qr"], "pairs": [],
        "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    # Ferrier-Aligo: four masses and no number moment; the threshold is the
    # scheme's own EPSQ (module_mp_fer_hires.F:168).  qt and the f_* state
    # are not analysis variables: FER_HIRES rebuilds qt and F_ICE/F_RAIN at
    # entry, and F_RIMEF only matters where qi exceeds EPSQ.
    5: {"name": "Ferrier-Aligo", "mass_only": ["qv", "qc", "qr", "qi"],
        "pairs": [], "unpaired": [], "repair_authority": None,
        "q_threshold": 1.0e-12},
    # WSM3: three masses, no moments (Registry.EM_COMMON:3017).
    3: {"name": "WSM3", "mass_only": ["qv", "qc", "qr"], "pairs": [],
        "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    4: {"name": "WSM5", "mass_only": ["qv", "qc", "qr", "qi", "qs"],
        "pairs": [], "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    6: {"name": "WSM6", "mass_only": ["qv", "qc", "qr", "qi", "qs", "qg"],
        "pairs": [], "unpaired": [], "repair_authority": None,
        "q_threshold": 1.0e-14},
    # WSM6R: WSM6's six masses, no number moment (wsm6rscheme,
    # Registry.EM_COMMON:3022).
    106: {"name": "WSM6R", "mass_only": ["qv", "qc", "qr", "qi", "qs", "qg"],
          "pairs": [], "unpaired": [], "repair_authority": None,
          "q_threshold": 1.0e-14},
    # Thompson's entry block is its repair authority, and R1
    # (module_mp_thompson.F:183) is the activity gate that block compares
    # every mass against -- two orders of magnitude above the table's own
    # default, which stood here until the scheme's threshold was read.
    8: {"name": "Thompson", "mass_only": ["qv", "qc", "qs", "qg"],
        "pairs": [{"species": "rain", "mass": "qr", "number": "nr"},
                  {"species": "ice", "mass": "qi", "number": "ni"}],
        "unpaired": [], "repair_authority": _THOMPSON_REPAIR_AUTHORITY,
        "q_threshold": 1.0e-12, "zero_number_below_threshold": True},
    # Milbrandt-Yau: every one of the six hydrometeors carries a number
    # moment (module_microphysics_driver.F:1857-1862 binds
    # qnc/qnr/qni/qns/qng/qnh INOUT); woof/core/state.py's mp=9 arm
    # allocates qh, nc, nr, ni, ns, ng, nh.  Its number initialisation is
    # not ported, so no repair authority.
    9: {"name": "Milbrandt-Yau two-moment", "mass_only": ["qv"],
        "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                  {"species": "rain", "mass": "qr", "number": "nr"},
                  {"species": "ice", "mass": "qi", "number": "ni"},
                  {"species": "snow", "mass": "qs", "number": "ns"},
                  {"species": "graupel", "mass": "qg", "number": "ng"},
                  {"species": "hail", "mass": "qh", "number": "nh"}],
        "unpaired": [], "repair_authority": None, "q_threshold": 1.0e-14},
    10: {"name": "Morrison two-moment", "mass_only": ["qv"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"},
                   {"species": "snow", "mass": "qs", "number": "ns"},
                   {"species": "graupel", "mass": "qg", "number": "ng"}],
         "unpaired": [], "repair_authority": _MORRISON_REPAIR_AUTHORITY,
         "q_threshold": 1.0e-14},
    # WDM6: double-moment warm rain (nc, nr) over WSM6's ice; the CCN
    # reservoir nn is prognostic and has no mass partner
    # (woof/core/wdm6_constants.py WDM6_NUMBER_SPECIES).
    24: {"name": "WSM7",
         "mass_only": ["qv", "qc", "qr", "qi", "qs", "qg", "qh"],
         "pairs": [], "unpaired": [], "repair_authority": None,
         "q_threshold": 1.0e-14},
    14: {"name": "WDM5 double-moment warm rain",
         "mass_only": ["qv", "qi", "qs"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"}],
         "unpaired": ["nn"], "repair_authority": None, "q_threshold": 1.0e-14},
    16: {"name": "WDM6 double-moment warm rain",
         "mass_only": ["qv", "qi", "qs", "qg"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"}],
         "unpaired": ["nn"], "repair_authority": None, "q_threshold": 1.0e-14},
    # WDM7: WDM6's moments plus the hail mass, which has no number.
    26: {"name": "WDM7 double-moment warm rain with hail",
         "mass_only": ["qv", "qi", "qs", "qg", "qh"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"}],
         "unpaired": ["nn"], "repair_authority": None, "q_threshold": 1.0e-14},
    18: {"resolved_by": "woof.core.nssl2_contract.resolve_nssl2_mode"},
    # Aerosol-aware Thompson: mp=8's pairs plus prognostic droplet number
    # and the two aerosol tracers (woof/core/state.py mp=28 arm).
    28: {"name": "Thompson aerosol-aware", "mass_only": ["qv", "qs", "qg"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["nwfa", "nifa"],
         "repair_authority": _THOMPSON_REPAIR_AUTHORITY,
         "q_threshold": 1.0e-12, "zero_number_below_threshold": True},
    # TEMPO: two-moment cloud, rain, ice and graupel with graupel volume and
    # the two aerosol tracers; its own entry block repairs number moments
    # inside the scheme, so the analysis detects and refuses (no token).
    900: {"name": "TEMPO", "mass_only": ["qv", "qs"],
          "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                    {"species": "rain", "mass": "qr", "number": "nr"},
                    {"species": "ice", "mass": "qi", "number": "ni"},
                    {"species": "graupel", "mass": "qg", "number": "ng"}],
          "unpaired": ["volg", "nwfa", "nifa"],
          "repair_authority": None, "q_threshold": 1.0e-12},
    # SBU-YLin: five single-moment masses.  rimi is not a moment and not
    # transported (WRF ``state`` package): the scheme rewrites it from
    # cloud water and snow on every call, so the analysis has nothing to
    # keep consistent with it.
    13: {"name": "SBU-YLin", "mass_only": ["qv", "qc", "qr", "qi", "qs"],
         "pairs": [], "unpaired": [], "repair_authority": None,
         "q_threshold": 1.0e-14},
    # P3 one-category: rain and the single ice category are two-moment; the
    # rime pair rides with the ice mass and has no number partner.
    50: {"name": "P3 one-category", "mass_only": ["qv", "qc"],
         "pairs": [{"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["qir", "qib"], "repair_authority": None,
         "q_threshold": 1.0e-14},
    97: {"name": "Goddard GCE three-ice",
         "mass_only": ["qv", "qc", "qr", "qi", "qs", "qg"],
         "pairs": [], "unpaired": [], "repair_authority": None,
         "q_threshold": 1.0e-14},
    # P3 with predicted droplet number: cloud water gains its number
    # partner (the transported qnc, Registry.EM_COMMON:3039); the rest is
    # mp=50's row.
    51: {"name": "P3 one-category, predicted droplet number",
         "mass_only": ["qv"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["qir", "qib"], "repair_authority": None,
         "q_threshold": 1.0e-14},
    # P3 three-moment ice: cloud becomes two-moment (prognostic nc), and
    # the advected ice moment qzi rides with the ice beside the rime pair.
    53: {"name": "P3 one-category three-moment ice", "mass_only": ["qv"],
         "pairs": [{"species": "cloud", "mass": "qc", "number": "nc"},
                   {"species": "rain", "mass": "qr", "number": "nr"},
                   {"species": "ice", "mass": "qi", "number": "ni"}],
         "unpaired": ["qir", "qib", "qzi"], "repair_authority": None,
         "q_threshold": 1.0e-14},
}

#: Offline downscale admission per microphysics selector: whether the
#: offline child lane (woof/offline_child.py) can read a same-scheme parent
#: of this scheme, and where it cannot, the breakage that keeps it out.
#: Cross-scheme admission is a different question, derived there from the
#: nest-edge closures.  Every ``false`` names its defect so the retirement
#: sweep is a grep.
_OFFLINE_CHILD_ROWS: dict[int, dict] = {
    11: {"same_scheme":False,"refusal":"Offline WRF-output child initialization lacks CAM cloud fraction, sedimentation and deep/shallow restart carriers; resident same-scheme nests retain them."},
    56: {"same_scheme": True, "refusal": None},
    2: {"same_scheme":True,"refusal":None},
    # passiveqv (Registry.EM_COMMON:3014) transports qv and nothing else,
    # which is exactly what the lane's transported-field helper now asks
    # for; the generic wrfout field map already carries QVAPOR.
    0: {"same_scheme": True, "refusal": None},
    # Kessler (Registry.EM_COMMON:3015): the qv/qc/qr prefix the helper
    # builds for every scheme IS its whole transported set, and the online
    # nest lane has carried mp=1 in PORTED_MP_PHYSICS throughout.
    1: {"same_scheme": True, "refusal": None},
    # Ferrier-Aligo.  Breakage this prevents: the lane maps a parent's
    # wrfout by scheme, and F_RIMEF_PHY is WRF state with no wrfout input
    # row there; a same-scheme child would start every ice cell unrimed
    # (F_RIMEF = 1) under rimed parent ice, changing fall speeds and
    # riming from the first step.  Admission is a wrfout map row, not a
    # code path.
    5: {"same_scheme": False,
        "refusal": (
            "the offline child lane has no wrfout map row for F_RIMEF_PHY, "
            "the Ferrier-Aligo rime factor FER_HIRES carries between steps "
            "; a child started without it "
            "would integrate the parent's rimed ice as unrimed")},
    # WSM3 (Registry.EM_COMMON:3017): Kessler's qv/qc/qr prefix is its whole
    # transported set too.
    3: {"same_scheme": True, "refusal": None},
    4: {"same_scheme": True, "refusal": None},
    6: {"same_scheme": True, "refusal": None},
    # WSM6R transports WSM6's six masses and nothing else, which the lane's
    # transported-field helper builds for it beside WSM6.
    106: {"same_scheme": True, "refusal": None},
    8: {"same_scheme": True, "refusal": None},
    # Milbrandt-Yau: the lane carries a third scheme-qualified wrfout map
    # (_MY2_WRF_TO_STATE) beside the NSSL one, because QHAIL/QNHAIL are
    # declared by both packages and bind to different state fields, and
    # woof's own history writer publishes QNHAIL -> nh for mp=9.
    9: {"same_scheme": True, "refusal": None},
    10: {"same_scheme": True, "refusal": None},
    # WDM6: the lane carries a fourth scheme-qualified wrfout map
    # (_WDM6_WRF_TO_STATE) with the QNCCN -> nn row, so the CCN reservoir
    # is read rather than zero-filled.
    14: {"same_scheme": True, "refusal": None},
    16: {"same_scheme": True, "refusal": None},
    24: {"same_scheme": True, "refusal": None},
    # WDM7: the lane's WDM6 map plus QHAIL -> qh (_WDM7_WRF_TO_STATE).
    26: {"same_scheme": True, "refusal": None},
    18: {"same_scheme": True, "refusal": None},
    28: {"same_scheme": True, "refusal": None},
    50: {"same_scheme": True, "refusal": None},
    # GCE carries the standard QVAPOR through QGRAUP mass fields.
    97: {"same_scheme": True, "refusal": None},
    # mp=51: the lane's transported-field helper carries the mp=50 set plus
    # nc (woof/offline_child.py), and the generic wrfout map already reads
    # QNCLOUD -> nc.
    51: {"same_scheme": True, "refusal": None},
    # P3 three-moment ice: the lane's map carries QZI -> qzi and its
    # transported-field helper names nc and qzi (woof/offline_child.py).
    53: {"same_scheme": True, "refusal": None},
    # TEMPO (named scheme, WOOF id 900): the offline-child lane has no
    # wrfout map for its graupel volume yet, so a same-scheme child is
    # refused by name rather than zero-filling volg.
    900: {"same_scheme": False, "refusal": (
        "the offline-child lane has no wrfout-to-state map for TEMPO's "
        "graupel volume (QVGRAUPEL -> volg) yet")},
    # SBU-YLin: the lane's transported-field helper carries the five-class
    # arm (qi, qs) and the generic wrfout map reads QICE/QSNOW; RIMI is a
    # scheme diagnostic the child's own first call rewrites.
    13: {"same_scheme": True, "refusal": None},
}

#: The specified-zone ring guard's snapshot family per microphysics
#: selector: the 3-D state fields it captures around every microphysics
#: call and the (ny, nx) accumulator/diagnostic slots the scheme writes.
#: woof/core/physics_inventory.py prices exactly these slots and
#: woof/core/microphysics.py captures the union of the state fields, so the
#: two cannot drift from each other or from this row.  Built in the pass
#: below because mp=18's set is read off woof.core.nssl2_contract.
_RING_SURFACE_STANDARD = ["mp_rainnc", "mp_rainncv", "mp_snownc",
                          "mp_snowncv", "mp_graupelnc", "mp_graupelncv",
                          "mp_sr"]
_RING_STATE_ROWS: dict[int, list[str]] = {
    11: ["thp","qv","qc","qr","qi","qs","nc","ni","nr","ns","qndrop",
         "cldfra_old_mp","cldfra_mp","cldfra_mp_all","lradius","iradius","cldfrai",
         "cldfral","cldfra_conv","wsedl3d","rh_old_mp","lcd_old_mp","effc","effi","effs"],
    56: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "qh"]
        + ['nc', 'nr', 'ni', 'ns', 'ng', 'nh', 'qdcn', 'qtcn', 'qccn', 'qrcn', 'qnin', 'fi', 'fs', 'vi', 'vs', 'vg', 'ai', 'as_', 'ag', 'ah', 'i3m']
        + ["effc", "effi", "effs"],
    2: ["thp","qv","qc","qr","qi","qs","qg"],
    1: ["thp", "qv", "qc", "qr"],
    # Ferrier-Aligo writes the total ice, qt and all three f_* fields.
    5: ["thp", "qv", "qc", "qr", "qi", "qt", "f_ice", "f_rain", "f_rimef"],
    # WSM3 writes its three radii in place every call (module_mp_wsm3.F:
    # 185-189).
    3: ["thp", "qv", "qc", "qr", "effc", "effi", "effs"],
    4: ["thp", "qv", "qc", "qr", "qi", "qs", "effc", "effi", "effs"],
    6: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"],
    # WSM6R advances the six masses and woof writes WSM6's radii on its
    # state every call (woof/core/wsm6.py), so the radii are in the set.
    106: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"],
    8: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
        "effc", "effi", "effs"],
    # Milbrandt-Yau: hail mass beside graupel plus all six numbers.
    9: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
        "nc", "nr", "ni", "ns", "ng", "nh", "effc", "effi", "effs"],
    10: ["thp", "qv", "qc", "qr", "qi", "qs", "qg",
         "nc", "nr", "ni", "ns", "ng", "effc", "effi", "effs", "effr"],
    # WDM6's three transported moments; nn is in the set because the scheme
    # WRITES it (fully evaporated rain and cloud return their number to the
    # reservoir, module_mp_wdm6.F:1249-1252, :1990-1994).
    14: ["thp", "qv", "qc", "qr", "qi", "qs", "nn", "nc", "nr",
         "effc", "effi", "effs"],
    16: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr",
         "effc", "effi", "effs"],
    26: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "nn", "nc", "nr",
         "effc", "effi", "effs"],
    24: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
         "effc", "effi", "effs"],
    # mp=18 is completed in the pass from nssl2_contract.DEFAULT_RESTART_FIELDS.
    18: ["thp"],
    # The exact mp=28 moment set; nwfa2d/nifa2d are INTENT(IN) in WRF and
    # no kernel writes them, so the guard has nothing to restore.
    28: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "nc",
         "nwfa", "nifa", "effc", "effi", "effs"],
    # P3: one ice mass with its rime pair, two numbers, the two
    # previous-step carriers, and the cloud/ice radii pair only.
    50: ["thp", "qv", "qc", "qr", "qi", "qir", "qib", "ni", "nr",
         "th_old", "qv_old", "effc", "effi"],
    # mp=51: mp=50's family plus the prognostic droplet number it writes.
    51: ["thp", "qv", "qc", "qr", "qi", "qir", "qib", "nc", "ni", "nr",
         "th_old", "qv_old", "effc", "effi"],
    # P3 three-moment ice: mp=50's family plus the droplet number and the
    # advected ice moment, both written in place by the kernels.
    53: ["thp", "qv", "qc", "qr", "qi", "qir", "qib", "ni", "nr", "nc",
         "qzi", "th_old", "qv_old", "effc", "effi"],
    # TEMPO: the six masses, cloud/rain/ice/graupel numbers, graupel
    # volume, the two aerosol tracers and the three radii (the capability
    # row in woof.microphysics_schemes).
    97: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "effc", "effi", "effs"],
    900: ["thp", "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ng",
          "volg", "nwfa", "nifa", "effc", "effi", "effs"],
    # SBU-YLin: five masses and the riming intensity it rewrites.
    13: ["thp", "qv", "qc", "qr", "qi", "qs", "rimi"],
}
_RING_SURFACE_ROWS: dict[int, list[str]] = {
    11: ["mp_rainnc","mp_rainncv","mp_snownc","mp_snowncv","mp_sr"],
    56: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    2: list(_RING_SURFACE_STANDARD),
    1: ["mp_rainnc", "mp_rainncv", "mp_kessler_sr"],
    # FER_HIRES writes RAINNC/RAINNCV/SR and nothing else
    # (module_mp_fer_hires.F:449-454, :970-974).
    5: ["mp_rainnc", "mp_rainncv", "mp_sr"],
    # WSM3 writes RAINNC/RAINNCV/SNOWNC/SNOWNCV/SR and no graupel
    # (module_mp_wsm3.F:597-625).
    3: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    4: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    6: list(_RING_SURFACE_STANDARD),
    # WSM6R's driver arm binds RAINNC/RAINNCV only; SR is the contract's
    # zero (module_microphysics_driver.F:2362-2372).
    106: ["mp_rainnc", "mp_rainncv", "mp_sr"],
    8: list(_RING_SURFACE_STANDARD),
    9: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    10: list(_RING_SURFACE_STANDARD),
    14: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    16: list(_RING_SURFACE_STANDARD),
    18: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    24: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    26: _RING_SURFACE_STANDARD + ["mp_hailnc", "mp_hailncv"],
    28: list(_RING_SURFACE_STANDARD),
    # P3's wrapper writes RAINNC/RAINNCV/SNOWNC/SNOWNCV/SR and nothing else
    # (module_mp_p3.F:894-898); no graupel category.
    50: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    97: list(_RING_SURFACE_STANDARD),
    51: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    53: ["mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv", "mp_sr"],
    900: list(_RING_SURFACE_STANDARD),
    # SBU-YLin's driver arm binds RAINNC/RAINNCV only
    # (module_microphysics_driver.F:2210); SR remains zero as in native WRF.
    13: ["mp_rainnc", "mp_rainncv", "mp_sr"],
}

#: Table-admitted column schemes (woof/core/column_microphysics.py): the
#: CUDA unit and entry point, the moist species and scheme-declared state
#: fields it updates in kernel argument order, the surface slots it
#: writes and the level tiers it is compiled at.  One launcher reads the
#: row, so admitting such a scheme is this row plus its kernel.
_COLUMN_KERNEL_ROWS: dict[int, dict] = {
    13: {
        "module": "sbu_ylin",
        "entry": "sbu_ylin_column",
        "level_define": "SBU_KMAX",
        "level_tiers": [64, 128, 256],
        "species": ["qv", "qc", "qr", "qi", "qs"],
        "state_fields": ["rimi"],
        "surface_slots": ["mp_rainnc", "mp_rainncv", "mp_sr"],
        "inputs": ["rho", "pii", "p", "z", "dz8w", "ht"],
        "authority": ("WRF v4.6.1 phys/module_mp_sbu_ylin.F, driver call "
                      "module_microphysics_driver.F:2182-2222"),
        "state_field_io": {"rimi": {
            "wrf_name": "RIMI", "units": "fraction",
            "description": "riming intensity", "advected": False,
            "restart": True, "history": True,
            "authority": "Registry.EM_COMMON:491 (IO irh)"}},
    },
}

#: Which wrfout precipitation accumulator each ring surface slot fills, for
#: the consumer export the Rust renderers read.
_RING_SLOT_ACCUMULATOR = {
    "mp_rainnc": "RAINNC", "mp_snownc": "SNOWNC",
    "mp_graupelnc": "GRAUPELNC", "mp_hailnc": "HAILNC",
}

#: The reflectivity producer set woof/core/refl.py dispatches, the
#: scheme-diagnostic route woof/da/obsop.py takes for NSSL, and the
#: selector ids these two facts leave without any route.
_REFLECTIVITY_SCHEME_DIAGNOSTIC = {
    56: "woof.core.ntu.reflectivity (WOOF's 10 cm operator over the "
        "scheme's own predicted gamma distributions; WRF's NTU arm "
        "writes no REFL_10CM; kernel ntu_refl_cell in "
        "woof/core/kernels/ntu.cu)",
    18: "woof.core.nssl2_diagnostics.diagnose_radardd02_if_due (WRF's "
        "radardd02, a separate pure diagnostic)",
    9: "woof.core.milbrandt2.reflectivity (the scheme's own Zet block, "
       "lifted into woof/core/kernels/milbrandt2_zet.cu because it "
       "updates nothing; that file carries the WRF citation)",
    50: "woof.core.p3_device.reflectivity (the Z half of the scheme's own "
        "final diagnostics, replayed on local copies in "
        "woof/core/kernels/p3_zdiag.cu because the forecast loop also "
        "updates the state; that file carries the WRF citation)",
    97: "woof.core.mp_column_schemes.reflectivity (the scheme's own "
        "refl10cm_gsfc in woof/core/kernels/gsfcgce_refl.cu, which updates "
        "nothing; that file carries the WRF citation)",
    51: "woof.core.p3_device.reflectivity (the Z half of the scheme's own "
        "final diagnostics, replayed on local copies in "
        "woof/core/kernels/p3_zdiag.cu because the forecast loop also "
        "updates the state; that file carries the WRF citation)",
    53: "woof.core.p3_device.reflectivity (the three-moment Z closure on "
        "local values in woof/core/kernels/p3_zdiag.cu, including the "
        "Registry sqrt(N*Z) conversion; WRF module_mp_p3.F:826-838,4795-4887)",
}


_MOMENT_ROWS[38] = copy.deepcopy(_MOMENT_ROWS[28])
_MOMENT_ROWS[38]["name"] = "Thompson graupel-hail"
_MOMENT_ROWS[38]["mass_only"].remove("qg")
_MOMENT_ROWS[38]["pairs"].append({"mass": "qg", "number": "ng", "species": "graupel"})
_MOMENT_ROWS[38]["unpaired"].append("qvolg")
_OFFLINE_CHILD_ROWS[38] = {
    "same_scheme": False,
    "refusal": "the offline archive reader has no Thompson graupel-hail "
               "number and volume mapping; reading this parent would "
               "discard the required ng and qvolg moments",
}
_RING_STATE_ROWS[38] = _RING_STATE_ROWS[28] + ["ng", "qvolg"]
_RING_SURFACE_ROWS[38] = list(_RING_SURFACE_ROWS[28])
_VERTICAL_BOUND_SOURCES["microphysics"]["thompson-graupel-hail-mp38"] = (
    "Thompson graupel-hail microphysics", "THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS")



_MOMENT_ROWS[27] = copy.deepcopy(_MOMENT_ROWS[26])
_MOMENT_ROWS[27]["name"] = "UDM"
_OFFLINE_CHILD_ROWS[27] = {"same_scheme": False, "refusal": "UDM offline parent initialization has no qualified CCN/cloud/rain-number closure."}
_RING_STATE_ROWS[27] = list(_RING_STATE_ROWS[26])
_RING_SURFACE_ROWS[27] = list(_RING_SURFACE_ROWS[26])
_VERTICAL_BOUND_SOURCES["microphysics"]["udm-mp27"] = ("UDM microphysics", "UDM_VERTICAL_LEVEL_BOUNDS")


_MOMENT_ROWS[29] = copy.deepcopy(_MOMENT_ROWS[28])
_MOMENT_ROWS[29]["name"] = "RCON"
_OFFLINE_CHILD_ROWS[29] = {"same_scheme": False, "refusal":
    "The offline parent reader has no qualified RCON aerosol/number initialization contract."}
_RING_STATE_ROWS[29] = list(_RING_STATE_ROWS[28])
_RING_SURFACE_ROWS[29] = list(_RING_SURFACE_ROWS[28]) + ["mp_cloudnc"]
_VERTICAL_BOUND_SOURCES["microphysics"]["rcon-mp29"] = (
    "RCON microphysics", "RCON_VERTICAL_LEVEL_BOUNDS")


def _fast_sbm_admission(registry):
    """Scheme admission and explicit boundaries for its spectral consumers."""
    from woof.core.fast_sbm_contract import BIN_SPECIES,NUMBER_SPECIES,MASS_SPECIES
    _MOMENT_ROWS[30]={"name":"FAST_SBM spectral distributions","mass_only":list(MASS_SPECIES),
                     "pairs":[],"unpaired":list(BIN_SPECIES)+list(NUMBER_SPECIES),
                     "repair_authority":None,"q_threshold":1.e-13}
    _OFFLINE_CHILD_ROWS[30]={"same_scheme":False,"refusal":"the offline forcing writer does not export the 142 spectral distributions; bulk fields cannot reconstruct a bin state"}
    _RING_STATE_ROWS[30]=["thp",*MASS_SPECIES,*BIN_SPECIES,*NUMBER_SPECIES,"th_old","qv_old","effc","effi","effs"]
    _RING_SURFACE_ROWS[30]=list(_RING_SURFACE_STANDARD)
    _VERTICAL_BOUND_SOURCES["microphysics"]["fast-sbm-mp30"]=("FAST_SBM microphysics","FAST_SBM_VERTICAL_LEVEL_BOUNDS")
    registry["components"]["microphysics"]["options"]["fast-sbm-mp30"]={
        "label":"FAST_SBM spectral bin / MP30","selectors":{"mp_physics":30},
        "implemented":True,"maturity":"implemented-unverified","scientific_evidence":"none",
        "parameters":{"moist":True,"moist_cq":True},"asset_requirements":[],
        "constraints":{"required_settings":{"moist":True},"required_settings_reasons":{"moist":"the scheme advances vapor and 142 spectral distributions and needs a moist state"}},
        "warnings":["The corrected default fixes declared WRF defects, including cell-depth sedimentation and collision mass conservation. Column oracle and water-budget gates are implementation checks; a matched forecast and observation-based skill remain separate qualifications."]}
    # A direct component override makes mp30 normally selectable without a case template.
    for route in registry["runner_routes"].values():
        allowed=route.get("allowed_component_options",{})
        if isinstance(allowed,dict) and "microphysics" in allowed and isinstance(allowed["microphysics"],list):
            if "fast-sbm-mp30" not in allowed["microphysics"]:
                allowed["microphysics"].append("fast-sbm-mp30")


def _consumer_rows(registry: dict) -> None:
    """Publish, per implemented option, what every consumer of it reads.

    THE DEFECT THIS CLOSES is the one the 2026-09-10 trigger was an
    instance of: forty-four hand-kept scheme lists restating part of the
    registry's inventory, none of them checked at plan review, each read
    for the first time inside the run.  This pass makes the registry carry
    the rows those lists need -- PULLED from the module that owns the fact
    where one does (checkpoint identities, vertical-bound constants, the
    nest resolver's ported set, the radar operator's floors, the export
    inventory), and OWNED here as data where the consumer will derive its
    table from the registry (moments, offline-child admission, ring guard,
    stock adapter classes).  Every consumer then either derives from these
    rows at import or asserts equality with them at import
    (woof.physics_registry.require_registry_agreement), and
    validate_run_config asks consumer_row_gaps() before step 0.

    Builds with the rebuild flag set so the consumers it imports do not
    refuse against the registry it is about to replace.
    """

    # P3 variants extend one scheme by package metadata, not adapter code.
    options = registry["components"]["microphysics"]["options"]
    row = copy.deepcopy(options["p3-mp50"])
    row.update(label="P3 two-category / MP52", selectors={"mp_physics": 52},
               reachability={"state": "component-override"},
               warnings=["Strict replay of 16 real native WRF captures: 1,456,320 words, zero differences. "
                         "A three-hour real forecast was inspected beside native WRF and MRMS; "
                         "both models underpredict the observed convection. No observation skill is claimed. "
                         "Two-category GPU stage time is 1.448x MP50 on RTX 5090; the second ice "
                         "population adds process, sedimentation and diagnostic work. Mixed-scheme "
                         "nest edges and offline-parent ingestion retain named category-partition refusals."],
               extensions={"fixed_mode": "two free ice categories, two-moment ice, prognostic droplet number, SCPF off",
                           "execution_model": "shared P3 CUDA category templates; CPU reference is one-category only",
                           "qualification": {"date": "2026-11-01", "replay_words": 1456320,
                                             "replay_differing_words": 0,
                                             "forecast_hours": 3, "same_scheme_nest_feedback": [0, 1],
                                             "oracle_scope": "tools/p3_wrf461_oracle/README.md#candidate-qualification-2026-11-01"}})
    row["asset_requirements"][0]["note"] = (
        "p3_init reads table 1 (module_mp_p3.F:337-410); the SHA-256 validator refuses "
        "missing or modified assets. Table 2 is packaged separately for MP52 ice-ice "
        "collection. The MP53 three-moment table and implementation remain unavailable.")
    from woof.core.p3_tables import TABLE_2_ASSET
    asset = copy.deepcopy(row["asset_requirements"][0])
    asset.update(id="wrf-v4.6.1-p3-lookuptable2-v5.3-v1",
                 assets=[dict(filename=TABLE_2_ASSET.filename, bytes=TABLE_2_ASSET.size,
                              sha256=TABLE_2_ASSET.sha256)],
                 source="WRF v4.6.1 run/p3_lookupTable_2.dat-v5.3",
                 note="p3_init reads table 2 for ice-ice collection when nCat > 1 (module_mp_p3.F:474-519)")
    row["asset_requirements"].append(asset)
    options["p3-mp52"] = row
    options["p3-mp50"].get("extensions", {}).get("unported_wrf_siblings", {}).pop("52", None)
    _VERTICAL_BOUND_SOURCES["microphysics"]["p3-mp52"] = ("P3 microphysics", "P3_VERTICAL_LEVEL_BOUNDS")
    _MOMENT_ROWS[52] = dict(name="P3 two-category", mass_only=["qv"],
                          pairs=[dict(species=s, mass=m, number=n) for s,m,n in
                                 (("cloud","qc","nc"),("rain","qr","nr"),("ice","qi","ni"),("ice2","qi2","ni2"))],
                          unpaired=["qir","qib","qir2","qib2"], repair_authority=None, q_threshold=1e-14)
    _OFFLINE_CHILD_ROWS[52] = dict(same_scheme=False, refusal="the offline-child route has no verified two-category parent completeness and partition contract")
    _RING_STATE_ROWS[52] = _RING_STATE_ROWS[50] + ["nc", "qi2", "ni2", "qir2", "qib2"]
    _RING_SURFACE_ROWS[52] = list(_RING_SURFACE_ROWS[50])
    _REFLECTIVITY_SCHEME_DIAGNOSTIC[52] = _REFLECTIVITY_SCHEME_DIAGNOSTIC[50]

    import dataclasses

    from woof import checkpoint_identity as ci
    from woof import physics_vertical_contract as vertical
    from woof.core import microphysics_transition as transition
    from woof.core import refl
    from woof.core import rrtmg_legacy
    from woof.core import rrtmgp
    from woof.core.nssl2_contract import DEFAULT_RESTART_FIELDS
    from woof.da import obsop
    from woof.core.radiation_carriers import CONSUMER_CARRIERS
    from woof.physics_registry import (
        CONSUMER_ROWS_KEY, CONSUMER_ROW_CONTRACT)
    from woof.wrf_physics_inventory import _INVENTORIES

    components = registry["components"]

    def selector(option: dict, key: str):
        return option.get("selectors", {}).get(key)

    def bounds_row(component_id: str, option_id: str):
        source = _VERTICAL_BOUND_SOURCES.get(component_id, {}).get(option_id)
        if source is None:
            return None
        label, constant = source
        minimum, maximum = getattr(vertical, constant)
        return {"label": label, "minimum": minimum, "maximum": maximum,
                "contract_constant": constant}

    # -- microphysics ------------------------------------------------------
    for option_id, option in components["microphysics"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        mp = int(selector(option, "mp_physics"))
        if (option.get("consumers", {}).get("native_column")
                or option.get("consumers", {}).get("transport", {}).get("empty_microphysics_step")):
            continue
        rows: dict[str, object] = {}
        rows["restart_algorithm_identity"] = (
            ci.MICROPHYSICS_ALGORITHM_IDENTITIES[mp])
        rows["vertical_level_bounds"] = bounds_row("microphysics", option_id)
        if mp in transition.PORTED_MP_PHYSICS:
            rows["nest_transition"] = {
                "mixed_edge_ported": True,
                "mass_fields": list(transition._MASS_FIELDS[mp]),
                "moment_fields": list(transition._MOMENT_FIELDS[mp]),
                "refusal": None,
            }
        elif mp in transition.UNVALIDATED_MIXED_EDGE_SELECTORS:
            scheme, _reason = transition._UNVALIDATED_MIXED_EDGE_REASONS[mp]
            rows["nest_transition"] = {
                "mixed_edge_ported": False,
                "mass_fields": None,
                "moment_fields": None,
                # The paragraph itself stays in the resolver (it carries WRF
                # citations the registry's citation checker cannot resolve);
                # the registry names the decision and where the reason is.
                "refusal": (
                    f"MP{mp} ({scheme}) is ported and runs, but it has no "
                    "validated cross-scheme entry closure for its moments ("
                    + (", ".join(transition.UNVALIDATED_MIXED_EDGE_MOMENTS[mp])
                       or ("none: gsfcgce_hail changes qg's meaning and "
                           "gsfcgce_2ice removes frozen classes" if mp == 97
                           else "none: its two condensate classes change phase at 0 C"))
                    + "); the reason is recorded at woof.core."
                    f"microphysics_transition._UNVALIDATED_MIXED_EDGE_REASONS[{mp}]"),
            }
        elif option.get('extensions', {}).get('column_kernel') is not None:
            rows['nest_transition'] = {
                'mixed_edge_ported': False, 'mass_fields': None,
                'moment_fields': None,
                'refusal': option['extensions']['column_kernel']['mixed_edge_refusal'],
            }
        else:
            rows["nest_transition"] = {
                "mixed_edge_ported": False,
                "mass_fields": None,
                "moment_fields": None,
                "refusal": (
                    f"mp_physics={mp} is in neither PORTED_MP_PHYSICS nor "
                    "UNVALIDATED_MIXED_EDGE_SELECTORS "
                    "(woof/core/microphysics_transition.py), so a mixed "
                    "edge touching it falls through to the generic "
                    "'ported selectors are ...' refusal; a same-scheme edge "
                    "resolves before either tuple is consulted"),
            }
        if mp == 0:
            radar = {"clear_air_floor_dbz": None,
                     "clear_air_floor_status": "no-microphysics",
                     "clear_air_floor_reason": None,
                     "reflectivity_route": "none",
                     "reflectivity_route_reason": None}
        else:
            if mp in obsop.CLEAR_AIR_FLOOR_DBZ:
                floor = {"clear_air_floor_dbz": obsop.CLEAR_AIR_FLOOR_DBZ[mp],
                         "clear_air_floor_status": "one-number",
                         "clear_air_floor_reason": None}
            elif mp in obsop.CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER:
                floor = {"clear_air_floor_dbz": None,
                         "clear_air_floor_status": "not-one-number",
                         "clear_air_floor_reason": (
                             "the operator reports more than one clear-air "
                             "value; the measured reason is recorded at "
                             "woof.da.obsop.CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER"
                             f"[{mp}]")}
            else:
                floor = {"clear_air_floor_dbz": None,
                         "clear_air_floor_status": "unread",
                         "clear_air_floor_reason": (
                             "nobody has read this scheme's clear-air floor "
                             "off its kernel yet")}
            if mp in refl.REFL_10CM_INPUT_SPECIES:
                route = {"reflectivity_route": "operator",
                         "reflectivity_route_reason":
                             "woof.core.refl.compute_refl_10cm"}
            elif mp in _REFLECTIVITY_SCHEME_DIAGNOSTIC:
                route = {"reflectivity_route": "scheme-diagnostic",
                         "reflectivity_route_reason":
                             _REFLECTIVITY_SCHEME_DIAGNOSTIC[mp]}
            elif mp == 11:
                route = {"reflectivity_route": "unrouted",
                         "reflectivity_route_reason": (
                             "CAMMGMP never writes REFL_10CM; its initialized "
                             "zero field is retained by woof.core.refl. "
                             "No radar reflectivity H(x) is bound for mp_physics=11")}
            elif mp in obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP:
                route = {"reflectivity_route": "native-not-separable",
                         "reflectivity_route_reason": (
                             "the scheme's Z is fused with its state update, "
                             "so there is no pure H(x); recorded at woof.da."
                             "obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP"
                             f"[{mp}]")}
            else:
                route = {"reflectivity_route": "unrouted",
                         "reflectivity_route_reason": (
                             "the scheme produces REFL_10CM natively "
                             "(woof.core.refl.SCHEME_NATIVE_REFL_10CM) but "
                             "woof/da/obsop.py names no H(x) route for it. "
                             "No shipped scheme is in this state: "
                             "simulated_reflectivity refuses the class by "
                             "name from this row (audit R-050), so a scheme "
                             "that reaches it is one added ahead of its "
                             "operator row rather than one whose refusal "
                             "reads like a typo")}
            radar = {**floor, **route}
        rows["radar_da"] = radar
        rows["moments"] = _MOMENT_ROWS[mp]
        if mp in _COLUMN_KERNEL_ROWS:
            row = copy.deepcopy(_COLUMN_KERNEL_ROWS[mp])
            maximum = getattr(vertical, _VERTICAL_BOUND_SOURCES[
                "microphysics"][option_id][1])[1]
            if row["level_tiers"][-1] != maximum:
                raise RuntimeError(
                    f"column_kernel row {mp}: largest tier "
                    f"{row['level_tiers'][-1]} != vertical bound {maximum}")
            rows["column_kernel"] = row
        rows["offline_child"] = dict(_OFFLINE_CHILD_ROWS[mp])
        if mp == 0:
            rows["ring_guard"] = None
        else:
            state_fields = list(_RING_STATE_ROWS[mp])
            if mp == 18:
                # WRF's option-18 defaults resolve to the full two-moment,
                # hail, CCN and volume set; every one is advanced in place.
                state_fields += [name for name in DEFAULT_RESTART_FIELDS
                                 if name not in state_fields]
                state_fields += ["effc", "effi", "effs"]
            rows["ring_guard"] = {"state_fields": state_fields,
                                  "surface_slots": list(_RING_SURFACE_ROWS[mp])}
        if mp in refl.REFL_10CM_INPUT_SPECIES:
            rows["reflectivity_input_species"] = list(
                refl.REFL_10CM_INPUT_SPECIES[mp])
            rows["reflectivity_native_reason"] = None
        else:
            rows["reflectivity_input_species"] = None
            rows["reflectivity_native_reason"] = (
                ("CAMMGMP never writes REFL_10CM in WRF; retain initialized zeros"
                 if mp == 11 else
                 "ETAMP_NEW never writes REFL_10CM in WRF; retain initialized zeros"
                 if mp == 95 else "the scheme produces REFL_10CM itself; recorded at "
                 f"woof.core.refl.SCHEME_NATIVE_REFL_10CM[{mp}]")
                if mp else "no microphysics, no reflectivity")
        # Cloud optics: the RTE+RRTMGP coupling name (None where the
        # adapter records a judged refusal) and the legacy adapter's
        # use_mp_re declaration, both PULLED from the adapters.  An
        # implemented scheme in neither RTE+RRTMGP table is a gap the
        # build refuses, because the adapter would refuse it at the first
        # radiation call with "add a row".
        coupling = rrtmgp._MP_CLOUD_OPTICS_SCHEME.get(mp)
        if coupling is None and mp not in rrtmgp._NO_CLOUD_OPTICS_COUPLING:
            raise RuntimeError(
                f"mp_physics={mp} is implemented and woof.core.rrtmgp "
                "neither couples it (_MP_CLOUD_OPTICS_SCHEME) nor records "
                "why not (_NO_CLOUD_OPTICS_COUPLING)")
        rows["cloud_optics"] = {
            "rte_rrtmgp_coupling": coupling,
            "rte_rrtmgp_refusal": (
                None if coupling is not None else
                rrtmgp._NO_CLOUD_OPTICS_COUPLING[mp] + ". "
                + rrtmgp._CLOUD_OPTICS_REMEDY),
            "legacy_declares_radii": rrtmg_legacy._MP_DECLARES_RADII[mp],
        }
        inventory = _INVENTORIES.get(mp)
        if inventory is None:
            rows["stock_wrf_export"] = {
                "inventoried": False,
                "reason": (
                    "no evidenced WRF v4.6.1 Registry.EM_COMMON package "
                    "contract is packaged for an unchanged WRF; export-only "
                    "scope, independent of the runtime verdict"),
                "wrfinput_fields": None,
                "runtime_state_not_wrfinput": None,
            }
        else:
            rows["stock_wrf_export"] = {
                "inventoried": True,
                "reason": None,
                "scheme": inventory.scheme,
                "registry_package": inventory.registry_package,
                "wrfinput_fields": [
                    {**dataclasses.asdict(field),
                     "dimensions": list(field.dimensions)}
                    for field in inventory.wrfinput_fields],
                "runtime_state_not_wrfinput": [
                    {**dataclasses.asdict(field),
                     "dimensions": list(field.dimensions)}
                    for field in inventory.runtime_state_not_wrfinput],
            }
        # Null for every scheme whose laterally forced form needs no
        # dataset beyond the boundary file itself, which is all of them but
        # one: the row exists on every option so a consumer reads a
        # DECISION rather than a missing key.
        rows["lateral_forcing_dataset"] = None
        if mp not in (28, 29, 38):
            # ``mp28_aerosol_source`` is read by ONE scheme and
            # woof.config refuses it on any other ("no other scheme reads
            # it, and woof refuses a stray value instead of silently
            # dropping it").  Every other option therefore forbids its two
            # non-default values here, so the registry decides that tuple
            # the same way the config validator does instead of calling it
            # launchable.  Derived from the enum, not re-typed.
            option.setdefault("constraints", {}).setdefault(
                "forbidden_setting_values", {})["mp28_aerosol_source"] = [
                    value for value in MP28_AEROSOL_SOURCES
                    if value != MP28_AEROSOL_SOURCE_DEFAULT_VALUE]
            option["constraints"]["forbidden_setting_values"]["use_rap_aero_icbc"] = [True]
            previous_forbidden = copy.deepcopy(option["constraints"]["forbidden_setting_values"])
            previous_forbidden["mp28_aerosol_source"] = [
                value for value in registry["parameters"]["mp28_aerosol_source"]["compatible_previous_enums"][0]
                if value != MP28_AEROSOL_SOURCE_DEFAULT_VALUE]
            previous_forbidden.pop("use_rap_aero_icbc", None)
            option["compatible_previous_forbidden_settings"] = [previous_forbidden]
        if mp in (28, 29, 38):
            # THE ONE SPELLING of the dataset precondition, read by both
            # authorities: the run door refuses an externally forced mp=28
            # domain without the dataset
            # (woof.config.mp28_aerosol_lateral_forcing_precondition), and
            # validate_physics_plan reports the same domain from this row.
            # The sentence and the ladder are imported from the modules
            # that own them rather than retyped, so a registry that says
            # LAUNCHABLE where the run door refuses is not expressible.
            from woof.config import MP28_AEROSOL_LATERAL_FORCING_PRECONDITION
            from woof.ingest import wif_climatology

            rows["lateral_forcing_dataset"] = {
                "id": "wrf-wif-monthly-aerosol-climatology-v1",
                "assets": [{"filename": wif_climatology.WIF_CLIMATOLOGY_FILE}],
                # ONE LADDER, named rather than copied.  This row used to
                # carry its own rung list, and plan review walked it with
                # the generic asset resolver -- which has no
                # working-directory rung (WRF's constants_name rule, which
                # the ingest ladder honours) and reads the single-file
                # override as a DIRECTORY.  Two ladders over one dataset is
                # how plan review calls a run launchable that the ingest
                # then cannot feed.  The registry now names the resolver
                # that owns the search, and
                # woof.physics_registry.resolve_lateral_forcing_dataset
                # asks it, with no argument.  There was a
                # ``path_setting`` here naming ``wif_climatology_path``;
                # it is gone because it was never reachable -- that name
                # is not in the registry's ``parameters`` and no runner
                # route admits it, so the plan that would carry it is
                # refused by ``_parameter_error`` before the row is read.
                # A plan is portable and a filesystem path is not: the
                # operator-named path is a RunConfig field the run door
                # resolves, and the ways out this row can offer are the
                # environment override and the staged root.
                "resolver": ("woof.ingest.wif_climatology"
                             ":resolve_wif_climatology"),
                "deliberate_setting": {
                    "name": "mp28_aerosol_source", "value": "synthetic"},
                "refusal": MP28_AEROSOL_LATERAL_FORCING_PRECONDITION,
            }
        option[CONSUMER_ROWS_KEY] = rows

    # -- cumulus / pbl / surface layer / land surface ----------------------
    identity_tables = {
        "cumulus": ("cu_physics", ci.CUMULUS_ALGORITHM_IDENTITIES),
        "pbl": ("bl_pbl_physics", ci.PBL_ALGORITHM_IDENTITIES),
        "surface_layer": ("sf_sfclay_physics",
                          ci.SURFACE_LAYER_ALGORITHM_IDENTITIES),
        "land_surface": ("sf_surface_physics",
                         ci.LAND_SURFACE_ALGORITHM_IDENTITIES),
    }
    for component_id, (key, table) in identity_tables.items():
        for option_id, option in components[component_id]["options"].items():
            if option.get("implemented") is not True:
                option.pop(CONSUMER_ROWS_KEY, None)
                continue
            value = int(selector(option, key))
            rows = {"restart_algorithm_identity": table[value]}
            if component_id == "cumulus" and option_id == "multiscale-kf":
                rows["restart_algorithm_identity"] = ci.MSKF_ALGORITHM_IDENTITIES["fast"]
                option["extensions"]["numerical_generations"] = {
                    key: {"restart_algorithm_identity": identity}
                    for key, identity in ci.MSKF_ALGORITHM_IDENTITIES.items()}
            if component_id == "cumulus" and option_id == "gfs-sas":
                rows["restart_algorithm_identity"] = ci.SAS_ALGORITHM_IDENTITIES["fast"]
                option["extensions"]["numerical_generations"] = {
                    key: {"restart_algorithm_identity": identity}
                    for key, identity in ci.SAS_ALGORITHM_IDENTITIES.items()}
            if component_id == "cumulus" and option_id == "kiaps-sas":
                # This file describes the shipped default and all proof
                # generations. Its bytes must not follow build-time math.
                rows["restart_algorithm_identity"] = (
                    ci.KSAS_ALGORITHM_IDENTITIES["fast"])
                option["extensions"]["numerical_generations"] = {
                    key: {"restart_algorithm_identity": identity}
                    for key, identity in ci.KSAS_ALGORITHM_IDENTITIES.items()}
            if component_id in ("cumulus", "pbl"):
                rows["vertical_level_bounds"] = bounds_row(
                    component_id, option_id)
            if component_id == "pbl":
                # WRF module_pbl_driver.F:2251-2259 (v4.6.1) runs diff4d
                # on the selected scheme's EXCH_H after ANY PBL scheme when
                # scalar_pblmix=1; with no PBL the driver is not called.
                rows["scalar_diffusion_exchange"] = (
                    _SCALAR_DIFFUSION_EXCHANGE[option_id])
            if component_id == "cumulus":
                rows["stock_callable_class"] = (
                    _STOCK_CALLABLE_CLASSES["cumulus"][option_id])
                if option_id in _CUMULUS_COLUMN_KERNEL_ROWS:
                    rows["column_kernel"] = copy.deepcopy(
                        _CUMULUS_COLUMN_KERNEL_ROWS[option_id])
            if component_id == "land_surface":
                bundle = ci.LAND_SURFACE_PARAMETER_SOURCES.get(value)
                rows["restart_parameter_bundle"] = (
                    None if bundle is None else
                    {"driver_attribute": bundle[0],
                     "asset_roles": list(bundle[1])})
                # FROM THE SCHEME'S OWN CARRIER CONTRACT, not from the
                # guard's list.  CONSUMER_CARRIERS states what each
                # land-surface scheme reads every surface step and REFUSES
                # an unlisted scheme rather than defaulting it to "reads
                # nothing", so a newly registered GLW-consuming LSM turns
                # this row True on its own -- and woof.physics_compat's
                # radiation-off guard, whose table is held equal to these
                # rows at import, then fails until it carries the scheme
                # too.  Deriving it from that guard's own literal made the
                # row a restatement and could never have caught the drift.
                rows["reads_glw"] = "glw" in CONSUMER_CARRIERS[value]
            option[CONSUMER_ROWS_KEY] = rows

    # -- radiation ---------------------------------------------------------
    for option_id, option in components["radiation"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        # Every radiation option is keyed on the pair its engine runs.
        # One row was keyed on (-1, -1) instead -- the sentinel for "the
        # split pair is not stated here" -- and had to be translated back
        # to 4/4 right here.  It is retired: a selector tuple that is an
        # absence matched every configuration written in the aggregate
        # spelling, whatever engine that spelling named.
        lw = int(selector(option, "ra_lw_physics"))
        sw = int(selector(option, "ra_sw_physics"))
        rows = {
            "restart_algorithm_identity": {
                "longwave": ci.LONGWAVE_ALGORITHM_IDENTITIES[lw],
                "shortwave": ci.SHORTWAVE_ALGORITHM_IDENTITIES[sw],
                "longwave_above_atmosphere_policy":
                    ci.LONGWAVE_ABOVE_ATMOSPHERE_POLICIES[lw],
                "shortwave_above_atmosphere_policy":
                    ci.SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES[sw],
                "resolved_pair": [lw, sw],
            },
            "stock_callable_class": _STOCK_CALLABLE_CLASSES["radiation"][option_id],
        }
        if (lw, sw) == (4, 4):
            rows["restart_algorithm_identity"]["rrtmg_legacy_variant"] = (
                "identity resolved from woof.core.rrtmg_legacy module "
                "constants at checkpoint time; no table row")
        option[CONSUMER_ROWS_KEY] = rows

    # -- turbulence --------------------------------------------------------
    for option_id, option in components["turbulence"]["options"].items():
        if option.get("implemented") is not True:
            option.pop(CONSUMER_ROWS_KEY, None)
            continue
        option[CONSUMER_ROWS_KEY] = {
            "restart_identity_binding": (
                "km_opt and its constants are bound by the checkpoint's "
                "configuration_sha256; there is deliberately no turbulence "
                "identity table (a row would be a sixth turbulence "
                "authority); the mixing operators' implementation is bound "
                "by the one string woof.checkpoint_identity."
                "DYCORE_MIXING_ALGORITHM_IDENTITY, recorded whenever the "
                "domain mixes"),
        }

    # -- completeness: the build refuses a registry that violates the
    #    contract, so plan review's consumer-row gate is silent by
    #    construction and fires only on a hand edit or a new consumer.
    for component_id, contract in CONSUMER_ROW_CONTRACT.items():
        for option_id, option in components[component_id]["options"].items():
            if option.get("implemented") is not True:
                continue
            rows = option.get(CONSUMER_ROWS_KEY, {})
            missing = sorted(set(contract) - set(rows))
            if missing:
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id} is "
                    f"implemented and lacks consumer rows {missing}")

    registry["authority"]["consumer_rows_declaration"] = (
        "components.<component>.options.<option>.consumers publishes, per "
        "implemented option, the row every downstream consumer of that "
        "option reads: restart_algorithm_identity (and for land surface the "
        "restart_parameter_bundle, for cumulus and radiation the "
        "stock_callable_class), vertical_level_bounds, nest_transition, "
        "radar_da, moments, offline_child, ring_guard, "
        "reflectivity_input_species, stock_wrf_export and cloud_optics.  "
        "Generated by "
        "tools/build_registry.py from the module that owns each fact. "
        "Consumers derive their tables from these rows or assert equality "
        "with them at import (woof.physics_registry."
        "require_registry_agreement); woof.config.validate_run_config "
        "asks woof.physics_registry.consumer_row_gaps before step 0, and "
        "validate_physics_plan mirrors it as consumer-row-missing.  A "
        "scheme is rows here, not a code path.")




def _madwrf_metadata(registry: dict) -> None:
    """WRF 4.7.1 empty MP96 package and optional MAD-WRF tracer driver."""
    import dataclasses
    from woof.wrf_physics_inventory import _INVENTORIES
    from woof.core.refl import SCHEME_ABSENT_REFL_10CM
    from woof.checkpoint_identity import MICROPHYSICS_ALGORITHM_IDENTITIES
    inv = _INVENTORIES[96]
    rows = {
        "restart_algorithm_identity": MICROPHYSICS_ALGORITHM_IDENTITIES[96],
        "vertical_level_bounds": {"label":"MAD-WRF", "minimum":4,
                                  "maximum":None, "contract_constant":None},
        "transport": {"base_fields":["qv","qc"],"extra_fields":["qi","qs"],
                      "empty_microphysics_step":True},
        "moments": {"name":"MAD-WRF", "mass_only":["qv","qc","qi","qs"],
                    "pairs":[], "unpaired":[], "q_threshold":1e-14,
                    "repair_authority":None},
        "ring_guard": {"state_fields":["thp","qv","qc","qi","qs"],
                       "surface_slots":[]},
        "reflectivity_input_species":None,
        "reflectivity_native_reason":SCHEME_ABSENT_REFL_10CM[96],
        "radar_da": {"clear_air_floor_dbz":None,"clear_air_floor_status":"unread",
                     "clear_air_floor_reason":"The empty native MP96 arm has no reflectivity diagnostic",
                     "reflectivity_route":"unrouted",
                     "reflectivity_route_reason":"MADWRF_MP never writes REFL_10CM"},
        "nest_transition": {"mixed_edge_ported":False,"mass_fields":["qv","qc","qi","qs"],
                            "moment_fields":[],"refusal":"No qualified MAD-WRF mixed-scheme nest edge"},
        "offline_child": {"same_scheme":False,"refusal":"No qualified MAD-WRF offline-child route"},
        "cloud_optics": {"legacy_declares_radii":False,"rte_rrtmgp_coupling":None,
                         "rte_rrtmgp_refusal":"MAD-WRF cloud optics are qualified with legacy RRTMG"},
        "lateral_forcing_dataset":None,
        "stock_wrf_export": {"inventoried":True,"reason":None,"scheme":inv.scheme,
            "registry_package":inv.registry_package,
            "wrfinput_fields":[{**dataclasses.asdict(f),"dimensions":list(f.dimensions)} for f in inv.wrfinput_fields],
            "runtime_state_not_wrfinput":[]},
    }
    registry["components"]["microphysics"]["options"]["madwrf-mp96"] = {
        "label":"MAD-WRF cloud transport / MP96", "implemented":True,
        "maturity":"supported", "scientific_evidence":"none",
        "selectors":{"mp_physics":96},"parameters":{"moist":True,"moist_cq":True,"use_mp_re":0},
        "constraints":{"required_settings":{"moist":True},
                       "required_settings_reasons":{"moist":"MAD-WRF transports cloud water, ice and snow"}},
        "warnings":["Native MAD-WRF has an empty microphysics step; no precipitation conversion or sedimentation is added",
                    "madwrf_opt=2 is an explicit opt-in: measured final scalar transport plus nudging costs 1.7384x MAD-off MP96; the native model also transports three extra tracers"],
        "verification_scope":"native-module-oracle-and-single-real-forecast",
        "verification_scope_note":"Strict WRF 4.7.1 real-input replay: mode 1 has 0/703950 differing physical output words; mode 2 has 0/4413936 including production tracer initialization. Captures, native inputs and reference outputs are permanently retained with manifests. Satellite flags are zero; no above-model-top or observation-skill claim. Both modes completed 300-step fast real forecasts with inspected Rust maps. See docs/evidence/madwrf471/qualification.json and tools/madwrf_wrf471_oracle/README.md.",
        "asset_requirements":[], "consumers":rows,
        "reachability":{"state":"component-override"},
        "extensions":{"preset_scope_reason": "MAD-WRF is an explicit cloud initialization and nudging option with no observation-based forecast skill qualification; stock WRF namelists and component overrides select mp_physics=96 directly",
                      "qualification_receipt":"docs/evidence/madwrf471/qualification.json",
                      "selection_policy":"Explicit selector only; mode 2 is FOLD-READY-OPTIN at measured 1.7384x affected-stage cost; MAD defaults to off"},
    }
    menu=registry["parameters"].get("mp_physics", {}).get("enum")
    if isinstance(menu,list) and 96 not in menu:
        menu.append(96);menu.sort()
    for route in registry["runner_routes"].values():
        options=route.get("allowed_component_options",{}).get("microphysics",[])
        if "kessler-mp1" in options and "madwrf-mp96" not in options:
            options.append("madwrf-mp96")


def _ishmael_native_metadata(registry: dict) -> None:
    """WRF Registry.EM_COMMON:3043, native adaptive-habit column metadata."""
    from woof.core.ishmael import ISHMAEL_TABLE_ASSETS, DIAG_SLOTS, SCRATCH_SLOTS
    transport = ["qv", "qc", "qr", "qi", "qi2", "qi3", "nr", "ni",
                 "qvoli", "qaoli", "ni2", "qvoli2", "qaoli2",
                 "ni3", "qvoli3", "qaoli3"]
    roles = {"qv":"qv", "qc":"qc", "qr":"qr", "nr":"nr",
             "qi1":"qi", "qi2":"qi2", "qi3":"qi3",
             "ni1":"ni", "ni2":"ni2", "ni3":"ni3",
             "ai1":"qvoli", "ai2":"qvoli2", "ai3":"qvoli3",
             "ci1":"qaoli", "ci2":"qaoli2", "ci3":"qaoli3"}
    surfaces = {"rainnc":"mp_rainnc", "rainncv":"mp_rainncv",
                "snownc":"mp_snownc", "snowncv":"mp_snowncv", "sr":"mp_sr"}
    native = {"module":"woof.core.ishmael", "function":"run_column_step",
              "transport":transport, "mass_slots":["qi","qi2","qi3"],
              "fields":roles, "state_fields":transport[3:] + ["effc","effi"],
              "diagnostics":{n:"native_column_"+n for n in DIAG_SLOTS},
              "scratch":{n:"native_column_"+n for n in SCRATCH_SLOTS},
              "surfaces":surfaces, "radius_units":"metres",
              "initial_radii_um":{"effc":10.0,"effi":25.0}}
    rows = {
       "native_column":native,
       "restart_algorithm_identity":"ishmael-wrf471-3ice-habit-volume-fast-default-native-strict-v2",
       "vertical_level_bounds":{"label":"ISHMAEL microphysics","minimum":1,"maximum":None,"contract_constant":None},
       "moments":{"name":"ISHMAEL", "mass_only":["qv","qc"],
          "pairs":[{"mass":q,"number":n,"species":q} for q,n in (("qr","nr"),("qi","ni"),("qi2","ni2"),("qi3","ni3"))],
          "unpaired":["qvoli","qaoli","qvoli2","qaoli2","qvoli3","qaoli3"],
          "q_threshold":1e-12,"repair_authority":None},
       "ring_guard":{"state_fields":["thp"]+transport+["effc","effi"],"surface_slots":list(surfaces.values())},
       "reflectivity_input_species":None,
       "reflectivity_native_reason":"WRF module_microphysics_driver.F:1480 binds the scheme diagnostic to REFL_10CM",
       "radar_da":{"clear_air_floor_dbz":None,"clear_air_floor_status":"unread",
          "clear_air_floor_reason":"The forecast and analysis helper has a native PSD diagnostic, but the public radar DA operator does not route MP55 or own a floor for it",
          "reflectivity_route":"native-not-separable","reflectivity_route_reason":"woof.da.obsop.NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP refuses MP55 until its state-reading diagnostic is wired through the public radar operator"},
       "nest_transition":{"mixed_edge_ported":False,"mass_fields":["qv","qc","qr","qi","qi2","qi3"],"moment_fields":transport[6:],
          "refusal":"Adaptive-habit ice cannot cross a mixed-scheme edge without a closure for both volume moments of all three categories"},
       "offline_child":{"same_scheme":True,"refusal":None},
       "cloud_optics":{"legacy_declares_radii":True,"rte_rrtmgp_coupling":"p3",
          "rte_rrtmgp_refusal":None},
       "lateral_forcing_dataset":None}
    from woof.wrf_physics_inventory import _INVENTORIES
    import dataclasses
    inventory = _INVENTORIES[55]
    rows["stock_wrf_export"] = {
        "inventoried": True, "reason": None,
        "scheme": inventory.scheme, "registry_package": inventory.registry_package,
        "wrfinput_fields": [{**dataclasses.asdict(f), "dimensions": list(f.dimensions)}
                            for f in inventory.wrfinput_fields],
        "runtime_state_not_wrfinput": [{**dataclasses.asdict(f), "dimensions": list(f.dimensions)}
                                      for f in inventory.runtime_state_not_wrfinput],
    }
    option={"label":"ISHMAEL adaptive-habit ice / MP55 (opt-in; measured 2.087x MP10 stage)", "implemented":True,
            "maturity":"supported","scientific_evidence":"none",
            "selectors":{"mp_physics":55},"parameters":{"moist":True,"moist_cq":True},
            "warnings":["Named opt-in, never a default: the whole MP55 stage measured 2.086776x MP10 in an interleaved 600-step deep-convective forecast on RTX PRO6000 Blackwell. The same native WRF4.7.1 case costs 2.316258x MP10 in 16-rank process CPU time, with extra three-habit and volume-moment work. Cost depends on the case. See docs/microphysics-retiming.md."],
            "verification_scope":"native-real-input-and-two-hour-forecast",
            "verification_scope_note":"Strict WRF4.7.1 real-input replay matches 254412 output words with zero differences. The retained step300 capture and full two-hour, 600-step deep-convective forecast are recorded by lane spd-retime-mp. Rust reflectivity maps were inspected. This verifies the implementation in the recorded suite, not observation-based forecast skill. See docs/microphysics-retiming.md and tools/ishmael_wrf461_oracle/qualification.json.",
            "constraints":{"required_settings":{"moist":True},"required_settings_reasons":{"moist":"Adaptive-habit ice needs transported water and habit moments"}},
            "asset_requirements":[{"id":"wrf461-ishmael-tables-v1","kind":"operator-supplied-table-set",
                "assets":[{"filename":a.filename,"bytes":a.bytes,"sha256":a.sha256} for a in ISHMAEL_TABLE_ASSETS],
                "root_environment_override":"WOOF_ISHMAEL_TABLE_ROOT","redistributed_by_gpuwm":False,
                "resolution":{"root_environment_override":"WOOF_ISHMAEL_TABLE_ROOT","staged_root":"~/.woof/tables/ishmael"},
                "source":"WRF v4.7.1 run/ishmael tables (identical at v4.6.1)","note":"Pinned WRF lookup tables; fetched on each box"}],
            "consumers":rows,"extensions":{"qualification":"tools/ishmael_wrf461_oracle/qualification.json"},"reachability":{"state":"component-override"}}
    registry["components"]["microphysics"]["options"]["ishmael-mp55"] = option
    menu = registry["parameters"].get("mp_physics", {}).get("enum")
    if isinstance(menu, list) and 55 not in menu:
        menu.append(55)
        menu.sort()


def render_consumer_export(registry: dict) -> bytes:
    """The JSON the Rust crates hold their catalogs against.

    ``tools/rustwx`` (rw-wrfbatch's raw-plane catalog and QPF palette) and
    ``tools/rw_wps`` (the stock-WRF inventory admission set) each keep a
    hand-written table that restates part of this registry.  Neither can
    import Python, so the inventory they need is exported here as plain
    JSON, generated beside the registry and byte-pinned by the same test,
    and each crate's test reads it.
    """

    from woof.io.wrf_output_schema import PRECIPITATION_OUTPUT_FIELDS
    from woof.physics_registry import CONSUMER_ROWS_KEY

    microphysics = {}
    accumulators: dict[str, list[int]] = {
        name: [] for name in PRECIPITATION_OUTPUT_FIELDS}
    for option_id, option in sorted(
            registry["components"]["microphysics"]["options"].items()):
        if option.get("implemented") is not True:
            continue
        mp = int(option["selectors"]["mp_physics"])
        rows = option[CONSUMER_ROWS_KEY]
        ring = rows.get("ring_guard") or {}
        filled = sorted({
            _RING_SLOT_ACCUMULATOR[slot]
            for slot in ring.get("surface_slots", [])
            if slot in _RING_SLOT_ACCUMULATOR})
        for name in filled:
            accumulators[name].append(mp)
        export = rows["stock_wrf_export"]
        microphysics[str(mp)] = {
            "option_id": option_id,
            "label": option["label"],
            "stock_wrf_export_inventoried": bool(export["inventoried"]),
            "wrfinput_netcdf_names": (
                [field["netcdf_name"] for field in export["wrfinput_fields"]]
                if export["inventoried"] else []),
            "wrfinput_dimensions": (
                sorted({"/".join(field["dimensions"])
                        for field in export["wrfinput_fields"]})
                if export["inventoried"] else []),
            "precipitation_accumulators": filled,
        }
    document = {
        "schema": "gpuwm-physics-consumer-export-v1",
        "generated_by": "tools/build_registry.py",
        "registry_version": registry["registry_version"],
        "microphysics": microphysics,
        "precipitation_output_fields": list(PRECIPITATION_OUTPUT_FIELDS),
        "scheme_bound_precipitation_fields": {
            name: sorted(ids) for name, ids in accumulators.items()},
    }
    return (canonical_json(document) + "\n").encode("utf-8")


CONSUMER_EXPORT_PATH = MODEL / "woof" / "physics_consumer_export_v1.json"


#: The one ladder every asset requirement is resolved down, derived here so
#: the registry document carries it and :func:`woof.physics_registry.
#: resolve_asset_requirement` can walk it with nothing but ``pathlib`` and
#: :mod:`woof.data_assets`.  Before this pass each row spelled its root in
#: its own key -- ``relative_root``, ``packaged_root``, ``search_root`` --
#: with no agreement on whether the spelling was the SOURCE tree's or the
#: INSTALL's, and two of them were stale: the RTE+RRTMGP rows still named
#: ``woof/data/rrtmgp``, a directory that has not existed since 2.5.0 moved
#: those bytes into the recast-woof-data companion.  Nothing caught it because
#: nothing read the field: ``validate_physics_plan`` collected asset
#: requirements and never resolved one, so a wheel install missing an asset
#: heard nothing at plan review and refused at load, naming a path inside
#: site-packages.  It is resolved now, and the answer is REPORTED --
#: ``install_state``, with the missing members, the roots walked and the
#: command that stages them -- rather than made the plan's verdict, because
#: what this machine has staged is not a property of the plan.  The load is
#: still where it refuses, and it now refuses something the reader was
#: already told about.
#:
#: ``data_relative`` is the ``woof/data``-relative path
#: :func:`woof.data_assets.data_path` understands, which is the spelling
#: that survived the companion split: the caller states the path it always
#: stated and never which distribution carries it.  ``staged_root`` mirrors
#: the loader's own second rung (``woof fetch-tables`` stages outside every
#: install, so the packaged root can legitimately be short an asset and the
#: run still work).
def _home_relative(parts) -> str:
    """``~``-relative spelling of a home-anchored root, from its owner.

    The document this builder writes must be byte-identical on every
    machine, so an absolute ``Path.home()`` cannot go in it -- which is
    why the roots below were re-typed here.  Importing the SEGMENTS and
    joining them keeps the one spelling in the module that owns the root
    and still writes a portable string.
    """

    return "~/" + "/".join(parts)


_ASSET_RESOLUTION_LADDERS = {
    "wrf-v4.6.1-mrf-constants-v1": {"data_relative": "mrf"},
    "wrf461-gfdleta-tables": {"data_relative": "gfdleta", "root_environment_override": "WOOF_TABLE_ROOT", "root_environment_subdirectory": "gfdleta", "exclusive_root_environment_override": True},
    "wrf461-ishmael-tables-v1": {
        "root_environment_override": "WOOF_ISHMAEL_TABLE_ROOT",
        "staged_root": "~/.woof/tables/ishmael",
    },
    # MP38's nine-slab cache is generated outside the installation. Its
    # loader has no packaged-table fallback, unlike the classic set.
    "wrf-v4.6.1-hail-aware-thompson-mp38-gfortran13-v1": {
        "root_environment_override": "WOOF_THOMPSON_HAIL_TABLE_ROOT",
        "staged_root": _home_relative(
            (*physics_compat.USER_THOMPSON_TABLE_ROOT_PARTS[:-1], "thompson-wrf461-hail")),
    },
    "wrf-cam3-tables-v1": {"data_relative":"cam3", "root_environment_override":"WOOF_CAM_TABLE_ROOT", "staged_root":"~/.woof/tables/cam3"},
    "wrf461-gsfc-shortwave-coefficients": {"data_relative":"wrf_radiation/gsfcsw"},
    # mp=8's classic set: two of its four assets are excluded from the
    # companion wheel by size (recast-woof-data/pyproject.toml) and arrive
    # through `woof fetch-tables`, so the staged root is not a fallback
    # here, it is the normal answer on a fresh install.
    "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1": {
        "data_relative": "thompson/tables",
        "root_environment_override": "WOOF_THOMPSON_TABLE_ROOT",
        "staged_root": _home_relative(
            physics_compat.USER_THOMPSON_TABLE_ROOT_PARTS),
    },
    # mp=28's CCN activation table: redistributed whole, no staging rung.
    "wrf-v4.6.1-aerosol-thompson-mp28-v1": {
        "data_relative": "thompson/tables",
        "root_environment_override": "WOOF_THOMPSON_TABLE_ROOT",
        "path_environment_override": "WOOF_THOMPSON_CCN_ACTIVATE",
    },
    "wrf-v4.6.1-p3-lookuptable1-2momi-v1": {
        "data_relative": "p3/tables",
        "root_environment_override": "WOOF_P3_TABLE_ROOT",
    },
    # mp=53's 3momI table: the same root and override as the 2momI table,
    # one packaged .xz file beside it.
    "wrf-v4.6.1-p3-lookuptable1-3momi-v1": {
        "data_relative": "p3/tables",
        "root_environment_override": "WOOF_P3_TABLE_ROOT",
    },
    "wrf-v4.6.1-ferrier-etampnew-expanded-rain-v1": {
        "data_relative": "ferrier/tables",
        "root_environment_override": "WOOF_FERRIER_TABLE_ROOT",
    },
    # RTE+RRTMGP's NetCDF set.  ``assets`` is filled from
    # ``woof.core.rrtmgp.RRTMGP_TABLE_FILES`` by ``_asset_resolution``:
    # this row declared a root and NO members, so plan review resolved it
    # vacuously -- ``resolve_asset_requirement`` returned
    # ``no-assets-declared`` and the caller skipped it -- and an install
    # short of a table still refused when radiation loaded.  It is one of
    # the two rows audit R-045 names, so it is also the row that must not
    # be allowed to declare nothing again; the guard below refuses that.
    "gpuwm-rte-rrtmgp-tables-v1": {
        "data_relative": "rrtmgp",
    },
    # Kain-Fritsch's lookup table: one file, in this wheel, no override.
    # It declares a relative_path rather than a root, so the ladder names
    # the directory and the row keeps its filename.
    "gpuwm-kf-lutab-v1": {
        "data_relative": "kf_lutab",
    },
    # TEMPO's four tables (NCAR release assets, not redistributed): the
    # adapter fetches them into the staged root on first use
    # (woof.tempo_assets.ensure_tempo_tables), so a fresh install reports
    # them missing at plan review and still runs.
    "ncar-tempo-v3-tables-v1": {
        "data_relative": "tempo/tables",
        "root_environment_override": "WOOF_TEMPO_TABLE_ROOT",
        "staged_root": "~/.woof/tables/tempo-v3",
    },
}
_ASSET_RESOLUTION_LADDERS["wrf-v4.6.1-p3-lookuptable2-v5.3-v1"] = copy.deepcopy(_ASSET_RESOLUTION_LADDERS["wrf-v4.6.1-p3-lookuptable1-2momi-v1"])



def _asset_requirement_members() -> dict:
    """Files a requirement must name, imported from the module that opens them.

    Only for rows that carry no ``assets`` of their own.  A requirement is
    resolved by plan review by CHECKING ITS FILES, so a row that names
    none resolves vacuously; the guard in :func:`_asset_resolution`
    refuses that, and this is where the missing names come from -- the
    module that opens the members, never a second list.
    """

    from woof.core import rrtmgp

    return {
        "gpuwm-rte-rrtmgp-tables-v1": [
            {"filename": name} for name in rrtmgp.RRTMGP_TABLE_FILES],
    }


def _asset_resolution(registry: dict) -> None:
    """Give every asset requirement the one ladder that resolves it."""

    from woof import data_assets

    members = _asset_requirement_members()
    seen = set()
    for component in registry["components"].values():
        options = component.get("options")
        if not isinstance(options, dict):
            continue
        for option in options.values():
            for requirement in option.get("asset_requirements", []) or []:
                ladder = _ASSET_RESOLUTION_LADDERS.get(requirement.get("id"))
                if ladder is None:
                    raise SystemExit(
                        "asset requirement " + repr(requirement.get("id"))
                        + " has no row in _ASSET_RESOLUTION_LADDERS; a "
                        "requirement plan review cannot resolve is a "
                        "requirement that refuses after step 0")
                seen.add(requirement["id"])
                requirement["resolution"] = dict(ladder)
                declared = members.get(requirement["id"])
                if declared is not None and not requirement.get("assets"):
                    requirement["assets"] = copy.deepcopy(declared)
                # A requirement that names no FILE is a requirement plan
                # review cannot check: ``resolve_asset_requirement`` reports
                # ``no-assets-declared`` and an empty ``missing`` list would
                # otherwise read as satisfied, so the install would refuse
                # at load with nothing said earlier.  The reader's half
                # raises ``asset-undeclared`` for it -- a registry defect,
                # the same on every machine, so unlike the install-state
                # codes it does decide ``launchable`` -- and this refuses to
                # emit one at all.  Having a ladder is not enough; the
                # ladder has to be walked LOOKING FOR SOMETHING.
                if not (requirement.get("assets")
                        or requirement.get("relative_path")):
                    raise SystemExit(
                        "asset requirement " + repr(requirement.get("id"))
                        + " declares no assets and no relative_path, so "
                        "plan review would report it satisfied without "
                        "checking a single file; give it its file list "
                        "(from the module that opens them, via "
                        "_asset_requirement_members) or a relative_path")
                # The stale per-row root spellings are replaced by the one
                # the ladder carries, so no reader can pick up a path that
                # has not existed since 2.5.0.
                for stale in ("relative_root", "packaged_root",
                              "search_root"):
                    requirement.pop(stale, None)
                # ``relative_path`` names a FILE and is left alone; only the
                # three ROOT spellings are replaced by the ladder.
                # Where the bytes live on an INSTALL, decided by the one
                # module that owns the companion split rather than by a
                # second list here.  This is the field the two RRTMGP rows
                # got wrong for four releases.
                relative = ladder.get("data_relative")
                if isinstance(relative, str):
                    requirement["installed_root"] = (
                        "woof_data/data/" + relative
                        if data_assets._is_companion(relative)
                        else "woof/data/" + relative)
                else:
                    requirement["installed_root"] = ladder.get("staged_root")
    unused = sorted(set(_ASSET_RESOLUTION_LADDERS) - seen)
    if unused:
        raise SystemExit(
            "_ASSET_RESOLUTION_LADDERS rows with no asset requirement: "
            + ", ".join(unused))


def _lateral_forcing_remedy_is_reachable(registry: dict) -> None:
    """Every route that can select the option can also answer its refusal.

    A refusal names the way out or it does not stand (gate law).  The
    lateral-forcing dataset refusal offers two: stage the dataset, which
    is an environment question every route can answer, and set a
    per-domain parameter, which a route can only answer if its
    ``allowed_parameter_keys`` names it.  This pass derives that from the
    consumers row rather than leaving it to be typed into a route table --
    exactly the drift class this audit is retiring, and a way out only one
    route accepts would be a second per-route physics table.
    """

    from woof.physics_registry import CONSUMER_ROWS_KEY

    remedies: dict[str, set[str]] = {}
    for component_id, component in registry["components"].items():
        for option in (component.get("options") or {}).values():
            row = (option.get(CONSUMER_ROWS_KEY) or {}).get(
                "lateral_forcing_dataset")
            if not isinstance(row, dict):
                continue
            deliberate = row.get("deliberate_setting") or {}
            name = deliberate.get("name")
            if name:
                remedies.setdefault(component_id, set()).add(str(name))
    if not remedies:
        return
    for route in registry["runner_routes"].values():
        reachable = set(route.get("allowed_component_overrides") or ())
        reachable |= set((route.get("allowed_component_options") or {}))
        wanted: set[str] = set()
        for component_id, names in remedies.items():
            if component_id in reachable:
                wanted |= names
        if not wanted:
            continue
        keys = set(route.get("allowed_parameter_keys") or ())
        route["allowed_parameter_keys"] = sorted(keys | wanted)


def _tke_nest_child(registry: dict) -> None:
    """km_opt=2 on a nest child: what the child does, not a refusal.

    The carried-through row refused it "until a nested prognostic-TKE
    domain has been run", a reason that names no breakage.
    woof.experiment admits it under any parent; this row says what the
    child does and what is measured.
    """
    tke = registry["components"]["turbulence"]["options"]["tke-1.5-order"]
    tke.setdefault("extensions", {}).pop("nest_child_restriction", None)
    tke["extensions"]["nest_child"] = {
        "behaviour": (
            "a km_opt=2 nest child cold-starts its own TKE under any parent "
            "and never returns it, as in WRF v4.6.1, whose Registry gives "
            "tke no nest-interpolation (i) and no feedback (f) flag; tke is "
            "not a nest-forced field in woof/core/nest_fields.py"),
        "measured": (
            "a 250 m km_opt=2 child under a km_opt=4 parent, 7 h, status "
            "PASS. A km_opt=2 child under a km_opt=2 parent loads with a "
            "not-yet-verified warning"),
    }
    tke["warnings"] = [
        ("Runs on a nest child under any parent; the child cold-starts its "
         "own TKE. See extensions.nest_child.")
        if warning.startswith("Refused on a nest child") else warning
        for warning in tke.get("warnings", [])]


def _urban_component(registry: dict) -> None:
    """WRF's urban canopy selector (``sf_urban_physics``), per option.

    Implemented: each model's CUDA column is graded against WRF v4.7.1's own
    routine (tests/test_urban_*_wrf471_parity.py).  Maturity stays at the
    implemented-unverified rung until the observation verification of record
    (ASOS, urban against rural stations) is recorded against it.
    """
    # The restart identities are woof.checkpoint_identity's table, the
    # source the checkpoint writer reads, copied here as every component's
    # are.
    from woof.checkpoint_identity import URBAN_ALGORITHM_IDENTITIES

    options = {}
    for value, option_id, module in (
            (0, "none", None), (1, "slucm", "woof.core.urban_ucm"),
            (2, "bep", "woof.core.urban_bep"),
            (3, "bep-bem", "woof.core.urban_bem")):
        row = {
            "label": option_id, "selectors": {"sf_urban_physics": value},
            "implemented": True,
            "maturity": "supported" if value == 0 else "implemented-unverified",
            "scientific_evidence": "none",
            "parameters": ({"use_wudapt_lcz": 0, "num_urban_hi": 15}
                           if value else {}),
            "asset_requirements": [], "warnings": [], "extensions": {},
        }
        if module:
            row["consumers"] = {
                "restart_algorithm_identity": URBAN_ALGORITHM_IDENTITIES[value],
                "vertical_level_bounds": None}
            # Selected in a run's [shared] table.  WRF runs one urban
            # selector on every domain, so the per-domain override lists
            # exclude it (_PER_DOMAIN_EXCLUSION_REASONS) and no template
            # names it yet.
            row["reachability"] = {
                "state": "unreachable",
                "blocker": (
                    "selected by sf_urban_physics in a run's [shared] "
                    "table; WRF runs one urban selector on every domain, "
                    "so no per-domain route override offers it and no "
                    "registered template names it yet")}
            row["constraints"] = {
                "admitted_setting_values": {"use_wudapt_lcz": [0, 1], "num_urban_hi": [15]},
                "admitted_setting_values_reasons": {
                    "use_wudapt_lcz": "urban_param_init opens only URBPARM.TBL or URBPARM_LCZ.TBL",
                    "num_urban_hi": "HI_URB2D uses a 15-bin stride; another count would read past the array"},
                "requires_components": {"land_surface": ["noah", "noah-mp"]},
                "requires_components_reasons": {
                    "land_surface": "RUC and no-LSM never call an urban model; "
                    "the selector would be silently ignored."},
            }
            if value in (2, 3):
                row["constraints"]["requires_components"]["pbl"] = [
                    "ysu", "myj", "boulac", "keps"]
                row["constraints"]["requires_components_reasons"]["pbl"] = (
                    "BEP drag, heat and TKE sources enter the PBL implicit "
                    "solve (YSU, MYJ, BouLac) or, for K-epsilon as WRF codes it, the "
                    "grid-blended surface fluxes; other ported PBL schemes "
                    "would drop them.")
        else:
            row["consumers"] = {
                "restart_algorithm_identity": URBAN_ALGORITHM_IDENTITIES[0],
                "vertical_level_bounds": None}
        if value == 1:
            from woof.checkpoint_identity import URBAN_UCM_GENERATION_IDENTITIES
            row["parameters"].update(urban_ucm_generation="wrf_471", urban_treeoption=-1, urban_ttscheme=-1, urban_fvg=-1.0)
            row["constraints"]["admitted_setting_values"].update(urban_ucm_generation=["wrf_471", "wrf_480"], urban_treeoption=[-1,0,1], urban_ttscheme=[-1,0,1])
            row["extensions"]["numerical_generations"] = {key: {"restart_algorithm_identity": value} for key,value in URBAN_UCM_GENERATION_IDENTITIES.items()}
            row["warnings"] = [
                "DECLARED DIVERGENCE from WRF v4.7.1: under Noah-MP the "
                "UCM's 2 m temperature is blended as the absolute "
                "temperature it is. module_surface_driver.F:3393 divides it "
                "by (1e5/PSFC)**RCP as if it were potential temperature, "
                "but module_sf_urban.F:1686 builds it from TS and TA, both "
                "absolute; the conversion cools every Noah-MP city cell by "
                "FRC x T x (1-(PSFC/1e5)**RCP), about 12 K at 1,900 m. "
                "Bitwise against WRF built with that one line fixed "
                "(tests/test_urban_ucm_noahmp_wrf471_parity.py).",
                "WRF stops the model (module_sf_urban.F:825) on any urban "
                "cell whose ZDC + Z0C + 2 m reaches the first model level; "
                "woof refuses such a configuration when it loads, from the "
                "table's classes and the eta ladder (Local Climate Zones 1 "
                "and 4 need a first level above about 34 and 31 m).",
            ]
        elif value in (2, 3):
            row["warnings"] = [
                "DECLARED DIVERGENCE from WRF v4.7.1 under YSU: the rural "
                "surface drag is applied once. bl_ysu.F90:1313-1314 removes "
                "only the urban fraction of YSU's own drag while the BEP "
                "couple (module_sf_noahdrv.F:1708-1711) already folds the "
                "rural drag into a_u_bep, so WRF drags every column that is "
                "not wholly urban twice, ocean included (1.25 m/s of 10 m "
                "wind over the sea within two hours at 750 m). kernels/"
                "ysu_480.cu removes the whole of YSU's own drag; graded against "
                "WRF built with that one-line fix "
                "(tests/test_ysu_bep_rural_drag.py). MYJ is unaffected.",
            ]
        options[option_id] = row
    registry["components"]["urban"] = {
        "scope": "per-domain", "selector_keys": ["sf_urban_physics"],
        "options": options,
    }
    for name, default in (("use_wudapt_lcz", 0), ("num_urban_hi", 15)):
        registry["parameters"][name] = {
            "type": "integer", "default": default,
            "component_id": "urban", "consuming_read": "woof/config.py",
        }
    for template in registry["templates"].values():
        template["components"]["urban"] = "none"
#: The EEPS option's evidence limits (bl_pbl_physics=16).
_EEPS_WARNINGS = (
    "Recovered CUDA transcription of WRF's public-domain EEPS module. "
    "The v4.6.1 and v4.7.1 module and constants pins are byte-identical. "
    "Four original-call strict replays match 151,552 retained native words; a separate full hour-three replay matches 13,320,000. The reproduced PBLH difference is localized to coupled input state, not same-input EEPS arithmetic. "
    "The historical lost-stream claim remains withdrawn. A new retained "
    "recapture matches all 189,440 words: four May calls (151,552) plus "
    "native Eta (37,888), with stock-zero and stock-snan witnesses. "
    "Inputs, outputs and commands: datasets/paper/wrf-ports/"
    "spd-recapture/bl16/MANIFEST.md, engine bd750d51ae66. "
    "The historical slow implementation remains withdrawn; the optimized "
    "source stays retained per the integration list. Cause audit report: "
    "WRF-PORTS-2026-11-01/port-bl16-eeps.md (2026-11-02 10:15 UTC). "
    "The superseded speed exception is withdrawn. The 3 km 12 s forecast "
    "audit measured 0.730x YSU; on the 12 km 60 s release timing case (three "
    "EEPS substeps, RTX 4090) the earlier kernel cost 1.457x YSU on the "
    "surface-layer plus PBL stage and the fused kernel costs 1.039x "
    "(eeps-port.md); the old 0.811516x was a replay microbenchmark. "
    "See docs/public/receipts/retime-pbl-forecast.json. No full-forecast native parity is claimed.",
    "Fast CUDA numerics is the default. GPUWM_WRF_EXACT=1 selects the "
    "pinned scalar math, no contraction and no flush to zero for proof. "
    "Bitwise replay covers strict mode; fast mode has a separate forecast check.",
    "RADSUM is uninitialized in native module_bl_eepsilon.F:748. The port "
    "initializes it to zero; the synthetic oracle keeps the stock-zero "
    "and stock-snan witness. A native-undefined SZV(KMINRAD-2) at :734 "
    "is reported through the device health ledger, never silently clamped.",
    "PEK_ADV/PEP_ADV are transported and restart-serialized separately "
    "from the PEK_PBL/PEP_PBL history carriers. Eta independently allocates "
    "its zero TKE_PBL column, witnessed in a completed native one-step run. "
    "It is never replaced with the positive EEPS PEK column.",
)


#: The ACM2 PBL option's warnings (bl_pbl_physics=7).  The measured numbers
#: are the parity test's own (tests/test_acm2_wrf461_parity.py).
_ACM2_WARNINGS = (
    "ACM2 is implemented-unverified. The pinned WRF v4.6.1 module is "
    "byte-identical in v4.7.1. Constructed fixtures grade 287120 output "
    "words; real strict native captures grade 96471 sfclay1 words and "
    "32157 PX words, all with zero differences. See "
    "tools/acm2_wrf461_oracle/native-real-receipt.json and "
    "tests/data/oracles/acm2/native-px/. Three-hour real sfclay1 and PX "
    "forecasts completed 900 steps each; PX used woof go --wrfinput. "
    "Rust maps and ASOS scores were compared with native WRF at three "
    "hours. Patterns are close with localized differences and shared "
    "warm/dry station biases; these cases do not establish general "
    "skill or whole-model bit identity. Fast numerics permits "
    "contraction and uses a linear bordered solve of the same implicit "
    "operator, retaining every native substep. Strict replay keeps "
    "native elimination order. Interleaved 4096-column, 40-level binding "
    "medians are 0.290 ms ACM2 / 0.317 ms YSU v4.8.0 (0.913x). All three "
    "legal surface pairs pass device nests (feedback 0/1), restart and "
    "one/four-member ensemble execution. A real GO split/resume also "
    "matches 203 checkpoint arrays and final NetCDF bytes.",
    "PAIRING: WRF v4.6.1 runs ACM2 with surface layers that set isfc 1 or 7 "
    "(phys/module_physics_init.F:3728-3730). Among the ported surface layers "
    "that is revised MM5 (1), classic MM5 (91) and Pleim-Xiu (7). "
    "The MYNN and Eta layers and the off option are refused, as WRF "
    "refuses them.",
    "DECLARED DIVERGENCE, witnessed native undefined accesses: a "
    "model-top free-convection column reads THETAV(kte+1) in the thermal "
    "search; ACM and ACMM write MDWN(kte+1), and their zero top mixing "
    "terms reference DZH/VCI(kte+1). Bounds-checked byte-unmodified WRF "
    "stops at phys/module_bl_acm.F:439, :978 and :1301; see "
    "tools/acm2_wrf461_oracle/native-top-undefined-receipt.json. CUDA "
    "bounds the thermal search and pads the zero top mixing term. "
    "No bitwise claim covers those native undefined accesses.",
    "WRF's own stop is kept: when the bulk Richardson number never reaches "
    "RIC in a column, WRF calls wrf_error_fatal ('RIBX never exceeds RIC', "
    "phys/module_bl_acm.F:464-470), and so does woof "
    "(woof.core.acm2pbl.AcmFatal). Model steps defer GPU health flags "
    "to the existing ledger drain; standalone calls name the first column.",
)


#: The UW moist-turbulence PBL option's warnings (bl_pbl_physics=9).  The
#: measured numbers are the parity tests' own (tests/test_uwpbl_*).
#: The MRF option's warnings (bl_pbl_physics=99).  The measured numbers are
#: the parity tests' own (tests/test_mrf_wrf461_parity.py, test_mrf_port.py).
_MRF_WARNINGS = (
    "MRF's distance from WRF v4.6.1 is MEASURED against WRF's own compiled "
    "module_bl_mrf.F: tools/mrf_wrf461_oracle links the object WRF's "
    "scalar referee build produced, proves a rebuild of the pinned source "
    "writes the same bytes, and records 73 columns x 4 steps (convective "
    "land, stable night, neutral windy, cloud-topped, very shallow, a "
    "thermal filling a short column, water, and single-comparison "
    "probes); the CUDA kernel reproduces every tendency, UST, HOL, PBLH "
    "and KPBL word bit for bit (tests/test_mrf_wrf461_parity.py). A strict "
    "real WRF v4.7.1 forecast capture adds four 49-level calls, 1192 defined "
    "output words, with zero differing words (woof.verify.mrf_real_capture); "
    "the v4.6.1 and v4.7.1 MRF source files are byte-identical. That is "
    "module conformance evidence. One complete three-hour real forecast "
    "through woof go was compared with native WRF and ASOS observations "
    "at three hours (docs/public/receipts/mrf/real-20261016.json), including "
    "local PBL-height differences and station wind errors. One case does "
    "not establish general skill; this option remains implemented-unverified.",
    "DECLARED DIVERGENCE, KPBL: WRF stores MRF's TOP-DOWN level index "
    "KPBL0 in KPBL2D (module_bl_mrf.F:1284), the field every other PBL "
    "scheme fills bottom-up, so a stable WRF column reports the model top. "
    "woof publishes the same level bottom-up (nz + 1 - KPBL0); nothing "
    "inside the scheme reads it, so no tendency moves "
    "(woof/core/kernels/mrf.cu header).",
    "WRF-AS-BUILT: the pinned source compiled at -O0 is not WRF as built "
    "on this toolchain: gfortran keeps ZFAC**PFAC (PFAC = 2.0) as a powf "
    "call there and folds it to a multiply at -O1 and above. The port "
    "follows WRF's own -O2 object (the multiply).",
    "MRF reports a stable column's PBL height, and that of a column whose "
    "thermal reaches the top level, as the first layer depth ZQ(KL) "
    "(module_bl_mrf.F:805, :1285); that is the scheme's own definition "
    "and the port keeps it.",
)


_UWPBL_WARNINGS = (
    "The UW PBL's distance from WRF v4.7.1 is MEASURED against the "
    "byte-unmodified CAMUWPBL sources: tools/uwpbl_wrf471_oracle builds "
    "them at gfortran -O0 and records every step of six regime families "
    "(convective day, stable night, stratocumulus, valley cold pool, "
    "mixed-phase, shallow cumulus) on three vertical grids, and the "
    "binary64 CPU reference woof/verify/uwpbl_ref and the CUDA kernel are "
    "graded against those words bit for bit (tests/test_uwpbl_driver_"
    "wrf471_parity.py). That is conformance evidence, not scientific "
    "validation: no woof/WRF forecast trajectory comparison and no "
    "observation score exists for this scheme yet, which is why this "
    "option is 'implemented-unverified'.",
    "THE ONE LICENCE-BOUND DIFFERENCE: WRF's cos and acos are glibc's "
    "binary64 IBM Accurate Mathematical Library (LGPL), which is not "
    "transcribed; the kernel uses correctly rounded cos/acos instead "
    "(woof/core/kernels/glibc_flt64.cuh). They enter only the "
    "trigonometric root of the entrainment cubic, and where glibc does not "
    "round correctly there a column can differ from WRF in its last bits; "
    "the rate is measured and stated in the kernel header. exp, log and "
    "pow are glibc's own (Arm optimized-routines, MIT) and bitwise.",
    "WRF-AS-BUILT: gfortran's -O2 -ftree-vectorize build of these sources "
    "(WRF's configure default) is not byte-identical to its -O0 build on "
    "this toolchain: the vectoriser routes loops in eddy_diff through "
    "libmvec's cos/pow and folds x**2._r8 to x*x. The port follows the "
    "-O0 build, which calls glibc's scalar functions everywhere.",
)


_GBMPBL_WARNINGS = (
    "The GBM PBL's distance from WRF v4.6.1 is MEASURED: "
    "tools/gbmpbl_wrf461_oracle drives WRF's own compiled "
    "module_bl_gbmpbl.o (the scalar referee build, byte-identical to a -O0 "
    "compile of the pristine source) on 86 columns for 8 steps each, and "
    "tests/test_gbmpbl_wrf461_parity.py requires the CUDA kernel to match "
    "every tendency, TKE, length scale, exchange coefficient, PBLH, UST "
    "and KPBL word for word under strict numerics. A real WRF 4.7.1 "
    "forecast capture also matched all 48,256 output words in two calls "
    "on an RTX 5090; its GBM source matches the 4.6.1 pin. "
    "Default fast numerics uses CUDA float32 exp/log/pow and the common "
    "compiler boundary with one warp per column, live shared workspace and "
    "warp-register tridiagonal factors. Unpublished first-pass ice and scalar "
    "work and second-pass TKE work are eliminated; strict retains its exact "
    "replay image. These module "
    "checks establish conformance, not forecast skill; the option stays "
    "implemented-unverified.",
    "DECLARED DIVERGENCES, both reads of undefined values in WRF: GBM2D's "
    "kpbl2dx is never initialised before pblhgt tests its sign "
    "(module_bl_gbmpbl.F:415, :1028-1050); the port starts it at 0, the arm "
    "WRF takes for every column after the first in a tile. tridag's pivot "
    "array is gam(100) in WRF and overruns past 100 levels; the port's is "
    "sized to the column.",
)

#: The K-epsilon PBL option's warnings (bl_pbl_physics=17).  The measured
#: numbers are the parity test's own (tests/test_keps_wrf471_parity.py).
_KEPS_WARNINGS = (
    "WRF v4.7.1 module_bl_keps.F, source SHA-256 "
    "a79058405affcd293740b15137002e788d168a5eba51586d14cee41342ffc204. "
    "The v4.6.1 synthetic replay is historical and does not qualify this port.",
    "Native undefined residues: GET_PBLH compares never-assigned ttke "
    "(:822-823); cloud-top cooling accumulates never-initialized radsum "
    "(:1065). The device uses computed tke and an initial radiative sum of zero. "
    "The neutral 0/0 clamp follows the scalar -O2 native result. The defined "
    "first-level WV_TUR expression keeps WRF's x wind (:363).",
    "The scheme receives model DT at each bldt cadence. It publishes no KPBL; "
    "consumers of KPBL are refused. BEP affects KEPS through blended surface "
    "fluxes; native bl_keps_adv is zero. Seven real-input captures have "
    "420336 compared words: 15372 differences from the unmodified module, "
    "all witnessed by its two undefined reads; zero differences after only "
    "those reads are defined. One three-hour real forecast and 12 station-hour "
    "samples are recorded in docs/public/receipts/keps-wrf471-real.json. "
    "This single case does not establish general forecast skill.",
)


_GFS_PBL_WARNINGS = (
    "The strict CUDA module replay reproduced all 152,576 output words "
    "captured from two calls on eight native WRF ranks in a real "
    "2026-10-16 09Z forecast, with zero differing words. "
    "GPUWM_WRF_EXACT=1 uses the native scalar module's rounding and libm; "
    "default arithmetic permits contraction and device libm. "
    "tools/gfs_pbl_wrf461_oracle also covers synthetic branch cases. "
    "Module conformance alone is not forecast validation.",
    "The GFS PBL requires its GFS surface layer (sf_sfclay_physics=3). "
    "The archived MM5 pairing exception is retired; WRF requires isfc=2.",
    "THE SCHEME REWRITES ITS SURFACE LAYER'S UST AND WSPD on every PBL "
    "call, as WRF's BL_GFS does (UST = sqrt(stress) from GZ1OZ0 - PSIM, "
    "WSPD the first-level speed plus 1e-9; module_bl_gfs.F:463-466), and "
    "the next surface-layer call reads them.",
)

_QNSE_WARNINGS = (
    "The QNSE PBL and its EDMF shallow mass flux are graded word for word "
    "against WRF v4.6.1: tools/qnse_wrf461_oracle builds the "
    "byte-unmodified module_bl_qnsepbl.F and module_bl_mfshconvpbl.F at "
    "WRF's own -O2 -fno-fast-math -ffp-contract=off flags and drives 72 "
    "columns (convective, stable, stratocumulus and shallow cumulus over "
    "land and sea, plain sea, snow, sea ice, sub-freezing, 1500 and 3000 m "
    "terrain, a jet, weak u*, southern and equatorial Coriolis, the TKE "
    "floor) for four steps with mfshconv 0 and 1, and the CUDA kernel "
    "(woof/core/kernels/qnsepbl.cu) is held to every output word under "
    "GPUWM_WRF_EXACT=1, step by step and chained "
    "(tests/test_qnse_wrf461_parity.py). That is conformance evidence, not "
    "scientific validation. A 2026-10-16 09-12Z, 3 km, 300x300x49 "
    "real forecast also runs through woof go with this pair and mfshconv=1. "
    "Native WRF, station scores and Rust-rendered maps are recorded in "
    "docs/public/receipts/qnse-real-forecast.json. This single case does "
    "not establish general skill, so maturity remains implemented-unverified.",
    "WRF's tile-wide COUNT(GTEST) is retained by a device reduction and "
    "second updraft pass. Runtime treats the full device domain as one "
    "WRF tile, including its decomposition-dependent RV_UP/RC_UP and "
    "DETR_EDKF diagnostics above a finished plume. The historical "
    "synthetic oracle uses one-column native tiles explicitly.",
    "WRF quirks transcribed rather than repaired, each cited in the "
    "kernel. The mass flux's theta and moisture tendencies are the "
    "conserved-variable rates d(theta_l)/dt and d(r_t)/dt, assigned to "
    "RTHBLTEN and RQVBLTEN after the theta reconstruction above them is "
    "overwritten (phys/module_bl_mfshconvpbl.F:458-469); RC_MF is a "
    "diagnostic split of QV, not a cloud-water tendency (RQCBLTEN = 0). "
    "QNSEPBL receives the mass flux's theta-v flux and BL89 mixing length "
    "and reads neither (the WTHV_MF term is commented out at "
    "phys/module_bl_qnsepbl.F:948), so woof does not compute them in a "
    "forecast. GTEST_LOCAL_LCL2 is read before it is set on some paths "
    "(phys/module_bl_mfshconvpbl.F:730, :997); woof starts it false, and "
    "the oracle's -finit-logical=true arm proves the garbage value is "
    "inert there (it gates a (1-ZPART_DRY)=0 increment).",
    "MYJ REUSE: QNSE is MYJ's column with three subroutines rewritten -- "
    "MIXLEN's stable-case length (phys/module_bl_qnsepbl.F:869-879), "
    "DIFCOF's QNSE stability functions (:973-1061) and PRODQ2's explicit "
    "production/dissipation (:886-968) -- and PBLH measured from Z(LPBL) "
    "rather than MYJ's Z(LPBL+1) (:732). VDIFQ, VDIFH and VDIFV are MYJ's "
    "solves; THSK is rebuilt from HFX (:442-451) instead of TSK. The "
    "kernel seeds the interface heights with the terrain height as WRF "
    "does.",
)

_BOULAC_WARNINGS = (
    "BouLac is transcribed from WRF's public-domain module_bl_boulac.F, "
    "identical in v4.6.1 and v4.7.1. Native WRF v4.7.1 real-input replay "
    "compared 81,664 words over 128 columns and 49 levels: zero differing "
    "words and zero native-undefined words under GPUWM_WRF_EXACT=1. "
    "The former capture was lost. The replacement inputs, reference outputs, "
    "four-rank native command and GPU replay are permanently retained at "
    "datasets/paper/wrf-ports/spd-recapture/bl8/MANIFEST.md; engine "
    "c69a56084ad73e4d8ae24350ded8534198b34427, WRF 4.7.1, glibc 2.39. "
    "Replay: python tools/boulac_wrf471_oracle/replay.py <bl8-dataset> "
    "--receipt replay-rerun.json. This establishes module arithmetic "
    "conformance, not forecast skill.",
    "WRF's rural drag is 0/0 at zero wind and stress; the port uses zero "
    "drag there. WRF does not write KPBL for BouLac; the port diagnoses it "
    "from PBLH for consumers requiring a one-based index.",
)


def _rename_template_ids(registry: dict) -> None:
    """Move menu IDs while retaining an explicit map for old inputs."""

    def replace(node):
        if isinstance(node, dict):
            return {TEMPLATE_ID_ALIASES.get(key, key): replace(value)
                    for key, value in node.items()}
        if isinstance(node, list):
            return [replace(value) for value in node]
        if isinstance(node, str):
            return TEMPLATE_ID_ALIASES.get(node, node)
        return node

    normalized = replace({key: value for key, value in registry.items()
                          if key != "template_aliases"})
    registry.clear()
    registry.update(normalized)


def _current_verification_scope(registry: dict) -> None:
    """Keep historical/exempt labels from claiming current exact-suite evidence."""

    mp8 = registry["components"]["microphysics"]["options"]["thompson-mp8"]
    mp8["verification_scope"] = "historical-matched-run"
    mp8["verification_scope_note"] = (
        "The 2026-07-28 WRF v4.6.1 comparison used legacy RRTMG and failed "
        "the t=0 digest on all four domains. Kernels changed in 2.7.4 and "
        "on 2026-09-23; that matched run has not been repeated. Current "
        "component evidence is the documented WRF Fortran column comparisons.")
    for template_id, template in registry["templates"].items():
        if template.get("maturity") == "wrf-matched-run":
            exemption = registry["maturity_ladder"]["composition_rule"][
                "composition_exemptions"].get(template_id)
            if exemption:
                template["verification_scope"] = "composition-exemption"
                template["verification_scope_note"] = (
                    "This label is a composition exemption, not a current "
                    "matched run of this exact suite. " + exemption["basis"])


def _lin_mp2(registry: dict) -> None:
    from copy import deepcopy
    options=registry["components"]["microphysics"]["options"]
    option=deepcopy(options["wsm6-mp6"])
    option.update(label="Purdue Lin six-class / MP2", selectors={"mp_physics":2},
                  parameters={"moist":True,"moist_cq":True},
                  maturity="implemented-unverified", scientific_evidence="none",
                  asset_requirements=[], extensions={},
                  warnings=["The 72-column WRF 4.6.1 oracle is bitwise under strict arithmetic. This is implementation evidence, not observation-based forecast skill."])
    option.pop("compatible_previous_forbidden_settings",None)
    option["constraints"]={"required_settings":{"moist":True},"forbidden_setting_values":{},"required_settings_reasons":{}}
    option["consumers"]={"restart_algorithm_identity":"lin-wrf461-six-class-v1"}
    option["physics_identity_extension"] = {"transitions": True}
    options["lin-mp2"]=option


def _single_moment_column_options(registry: dict) -> None:
    """Register the column table rows; the door is the ordinary mp_physics selector.

    No preset is generated: the radiation-off template the WSM lane generated here
    made radiation.off reachable from a registered suite (spot-check FAIL,
    the 2.8.9 ports spot-check results).  A row is reachable as a
    component override, exactly like WSM3, WSM6R, Lin and Ferrier-Aligo.
    """
    from woof.core.column_schemes import COLUMN_SCHEMES
    options = registry["components"]["microphysics"]["options"]
    for mp, row in COLUMN_SCHEMES.items():
        option_id = f"{row.unit}-mp{mp}"
        option = copy.deepcopy(options["wsm6-mp6"])
        option.update(label=f"{row.label} / MP{mp}", selectors={"mp_physics": mp},
                      maturity="implemented-unverified", scientific_evidence="none",
                      reachability={"state": "component-override"}, parameters={"moist": True, "moist_cq": True},
                      warnings=["Column arithmetic is checked against WRF 4.6.1; observation-based forecast skill is not established."],
                      extensions={"declared_divergences": [
                          {"id": "remap-top-search", "default_on": True,
                           "cause": "WRF nislfv_rain_plm skips the last arrival interface and deletes upper-cell condensate.",
                           "oracle_arm": "defined", "code": "woof/core/kernels/wsm5.cu"},
                          {"id": "melting-snow-evaporation", "default_on": True,
                           "cause": "Stock WSM5 uses psdep, which is zero in the warm subsaturated branch; the port uses the WSM6 evaporation expression.",
                           "oracle_arm": "defined", "code": "tools/wsm5_wrf461_column_oracle/patch_arms.py"}]})
        option.pop("consumers", None)
        options[option_id] = option

def _new_goddard(registry):
    """A paired solver is admitted by its inputs, outputs and selector row."""
    row=copy.deepcopy(registry['components']['radiation']['options']['wrf-rrtm-dudhia'])
    row.update(label='WRF 4.6.1 New Goddard longwave and shortwave',
        selectors={'ra_lw_physics':5,'ra_sw_physics':5},
        parameters={'ra_physics':0,'icloud':1,'wrf_rrtmg_compatibility':'none'},
        warnings=['WRF-Chem aerosol coupling is not implemented. Meteorological aerosols are zero, as in WRF_CHEM=0.',
                  'The SW visible cloud depth is defined from taucl(:,:,8); stock WRF passes unassigned taux scratch.',
                  'Non-7 microphysics uses WRF use_mp_re radii when provided; otherwise initialized zero radii reach the cloud LUT lower-bound clipping.'],
        extensions={'column_contract':{
            'inputs':['T','P','P8W','PI','DZ8W','RHO','QVAPOR','QCLOUD','QICE','QRAIN','QSNOW','QGRAUP','QHAIL','CLDFRA','RE_*_GSFC','ALBEDO','EMISS','TSK','XLAT','XLONG'],
            'outputs':['RTHRATENLW','RTHRATENSW','SWDOWN','GLW','GSW','OLR','SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP','TSWDN','TSWUP','SSWDN','SSWUP'],
            'wrfout_names':['SWDOWN','GLW','OLR','SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP','TSWDN','TSWUP','SSWDN','SSWUP','CLDFRA'],
            'cloud_table_sha256':'ad4bb1dd8446a2b00559ac425e21255a01b79a4cac7b5e3e693a3c98c222a1c8'}})
    registry['components']['radiation']['options']['new-goddard']=row

def _sms_3dtke(registry: dict) -> None:
    """Native SMS closure's selector, state and WRF pairing rules."""
    surface = [key for key, row in registry["components"]["surface_layer"]["options"].items()
               if row.get("selectors", {}).get("sf_sfclay_physics") in (0, 1, 5, 91)]
    registry["components"]["turbulence"]["options"]["sms-3dtke"] = {
        "label": "SMS-3DTKE scale-adaptive turbulence",
        "selectors": {"km_opt": 5}, "implemented": True,
        "maturity": "experimental-runtime",
        "scientific_evidence": "converged-numerical-reference",
        "parameters": {"tke_adv_opt": 1, "sfs_opt": 0, "m_opt": 0,
                       "tke_mix2_off": False, "tke_mix6_off": False,
                       "c_k": 0.15, "c_s": 0.25, "mix_isotropic": 0,
                       "mix_upper_bound": 0.1, "tke_heat_flux": 0.0,
                       "tke_drag_coefficient": 0.0, "tke_upper_bound": 1000.0},
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"diff_opt": 2, "bl_pbl_physics": 0},
            "requires_components": {"pbl": ["off"], "surface_layer": surface},
            "required_settings_reasons": {
                "diff_opt": "WRF SMS-3DTKE requires metric stresses and implicit vertical diffusion",
                "bl_pbl_physics": "SMS-3DTKE supplies the boundary-layer turbulence closure, so a PBL scheme would mix the same boundary layer a second time"},
            "requires_components_reasons": {
                "surface_layer": "WRF SMS-3DTKE supports surface-layer selectors 0, 1, 5 and 91"},
            "refused_when": [{"settings": {"cu_physics": [11]},
                "reason": "WRF scale-aware Kain-Fritsch requires a separate PBL scheme",
                "remedy_label": "Set cu_physics=0.",
                "remedy_settings": {"cu_physics": 0}}]},
        "warnings": ["Strict native replay: zero differing words over 192 real calls at both theta settings. A paired six-hour forecast, restart and speed receipts are retained. Nondefault TKE transport, backscatter and stress diagnostics remain unimplemented. Physical multi-card transport and strict whole-forecast equality are not claimed."],
        "wrf_counterpart": "WRF 4.6.1 dyn_em/module_diffusion_em.F, km_opt=5",
        "extensions": {"prognostic_state": {
            "arrays": ["tke", "tke0", "gamu", "gamv"],
            "note": "WRF retains TKE and the nonlocal momentum terms across steps and restart"}},
    }


def _kfcup_metadata(registry):
    """Scheme-local selector, persistent state and namelist controls."""
    from woof.core.kfcup_contract import KFCUP_PARAMETER_DEFAULTS, KFCUP_STATE_ROWS
    registry["components"]["cumulus"]["options"]["kf-cup"] = {
        "label": "KF-CuP", "implemented": True,
        "maturity": "experimental-runtime", "scientific_evidence": "none",
        "selectors": {"cu_physics": 10}, "parameters": {},
        "asset_requirements": list(registry["components"]["cumulus"]["options"]
                                   ["kain-fritsch"]["asset_requirements"]),
        "extensions": {"held_state": list(KFCUP_STATE_ROWS)},
        "constraints": {"required_settings": {"moist": True}},
        "warnings": ["KF-CuP requires 8 through 128 mass levels for its column workspace. "
                     "Native LNTERMS is undefined outside shallow columns. "
                     "The inherited extreme-PDF overflow correction remains declared. "
                     "Forecast and observation proof is recorded separately."],
    }
    option = registry['components']['cumulus']['options']['kf-cup']
    option['warnings'].append(
        'Explicit opt-in only; never a default. The whole-forecast CuP stage costs '
        '1.377177x CU1 on the same RTX PRO 6000. Native CuP PDF selection and '
        'deep/shallow cloud bookkeeping, after subtracting the nested KF solver, '
        'account for 22.27% of diagnostic column cycles. Default math uses CUDA '
        'REAL32 operators and FMA; strict replay retains native rounding. The '
        'older 1.122x number was a kernel deck. Profiles are attribution, not '
        'lower bounds. See docs/evidence/retime-20261102/cu10-fast-summary.json.')
    option['extensions']['forecast_cost'] = {
        'stage_ratio_to_cu1': 1.3771765611436333,
        'receipt': 'docs/evidence/retime-20261102/cu10-fast-summary.json',
        'native_work_profile': 'docs/evidence/retime-20261102/cu10-native-work-private-profile.json',
        'selection_policy': 'explicit-component-override-only; never a default'}
    for key, default in KFCUP_PARAMETER_DEFAULTS.items():
        row = {"implemented": True, "default": default, "per_domain": True,
               "component_id": "cumulus", "read_when": {"cu_physics": 10},
               "consuming_read": ("woof/core/kfcup_radiation.py" if key == "shallowcu_forced_ra"
                                  else "woof/core/kf.py"),
               "type": "boolean" if isinstance(default, bool) else
                       "integer" if isinstance(default, int) else "number"}
        if key == "numbins": row.update(minimum=1, maximum=46340)
        elif key in ("mindeepfreq", "minshallowfreq"): row.update(minimum=0., maximum=1.)
        elif not isinstance(default, bool): row["exclusiveMinimum"] = 0.
        registry["parameters"][key] = row


def _temf_pair(registry: dict) -> None:
    """The coupled TEMF pair, state and admission as metadata rows."""
    from woof.config import PHYSICS_SELECTOR_COUPLINGS
    from woof.core.physics_inventory import (
        TEMF_PBL_STATE_3D, TEMF_PBL_STATE_2D, TEMF_SFCLAY_STATE_2D)
    from woof.core.temf_constants import TEMF_CONSTANTS_SHA256
    coupling = next(row for row in PHYSICS_SELECTOR_COUPLINGS if row[1] == 10)
    warning = (
        "Native WRF v4.6.1 real-run input replay under strict numerics: "
        "PBL 0/7136 differing words; surface layer 0/160. Native PBL "
        "arithmetic and the upper mass-flux recurrence are retained. "
        "The surface zero-moisture limit is declared separately. "
        "Observation-based forecast skill is not established.")
    for component, selector, other in (
            ("pbl", "bl_pbl_physics", "surface_layer"),
            ("surface_layer", "sf_sfclay_physics", "pbl")):
        carried = (list(TEMF_PBL_STATE_3D + TEMF_PBL_STATE_2D)
                   if component == "pbl" else list(TEMF_SFCLAY_STATE_2D))
        registry["components"][component]["options"]["temf"] = {
            "asset_requirements": [],
            "constraints": {
                "requires_components": {other: ["temf"]},
                "requires_components_reasons": {other: coupling[4]}},
            "extensions": {
                "carried_restart_fields": carried,
                "constant_table": {
                    "generator": "woof.core.temf_constants:cuda_table",
                    "sha256": TEMF_CONSTANTS_SHA256},
                "dfi_refusal": {
                    "dfi_opt": 0,
                    "reason": "DFI not available for bl_pbl_physics=10",
                    "wrf_source": "share/module_check_a_mundo.F:2317-2330"},
                "surface_kernel": "woof/core/kernels/temf_sfclay.cu",
                "surface_math": "WOOF correctly rounded float32 LOG10",
                "coupled_selectors": {coupling[0]: coupling[1],
                                      coupling[2]: coupling[3]},
                "tendencies": (["rublten", "rvblten", "rthblten",
                                "rqvblten", "rqcblten"] if component == "pbl"
                               else []),
                "declared_fixes": ([] if component == "pbl" else ["F3"])},
            "implemented": True, "label": "TEMF " + component.replace("_", " "),
            "maturity": "implemented-unverified", "parameters": {},
            "reachability": {"state": "component-override"},
            "scientific_evidence": "none", "selectors": {selector: 10},
            "warnings": [warning]}
def _temf_pair_routes(registry: dict) -> None:
    # TEMF writes KPBL and the surface fluxes consumed by the GF trigger.
    gf = registry["components"]["cumulus"]["options"].get("grell-freitas")
    if gf is not None:
        allowed_pbl = gf["constraints"]["requires_components"]["pbl"]
        if "temf" not in allowed_pbl:
            allowed_pbl.append("temf")
    for route in registry["runner_routes"].values():
        allowed = route.get("allowed_component_options", {})
        for component in ("pbl", "surface_layer"):
            if component in allowed and "temf" not in allowed[component]:
                allowed[component].append("temf")


def _fu_liou_gu(registry):
    from woof.core.flg import SOURCE_SHA256, TABLE_SHA256
    row=copy.deepcopy(registry['components']['radiation']['options']['wrf-rrtm-dudhia'])
    row.pop('consumers',None)
    row.update(label='WRF Fu-Liou-Gu longwave and shortwave',
        selectors={'ra_lw_physics':7,'ra_sw_physics':7},
        implemented=True,maturity='implemented-unverified',scientific_evidence='none',
        reachability={'state':'component-override'},
        parameters={'ra_physics':0,'icloud':1,'wrf_rrtmg_compatibility':'none'},
        warnings=['Native WRF executes both spectra through LW-7 and its SW-7 case is empty. Mixed selections preserve that ordering, including additional SW heating.',
                  'TSWUP is the computed per-column upward SW flux. Native RAD_FLG assigns its entire USWTOP array on every column.'],
        extensions={'column_contract':{
            'source_sha256':SOURCE_SHA256,'table_sha256':TABLE_SHA256,
            'layers':[4,100],'native_controls':{'ngas':0,'nice':2,'npde':2,'nfract':0,'ninho':0,'nozone':1,'naero':0},
            'inputs':['T','P8W','PI','RHO','QVAPOR','QCLOUD','QICE','QRAIN','QSNOW','QGRAUP','ALBEDO','EMISS','TSK','XLAT','XLONG'],
            'outputs':['RTHRATENLW','RTHRATENSW','SWDOWN','GLW','GSW','OLR','SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP','TSWDN','TSWUP','SSWDN','SSWUP','CLDFRA'],
            'licence':'WRF public domain statement, README.md at v4.7.1'}})
    registry['components']['radiation']['options']['fu-liou-gu']=row

def _retimed_radiation_costs(registry):
    """Real forecast costs; these additional native solvers stay opt-in."""
    options = registry['components']['radiation']['options']
    for key, ratio, warning, receipt, profile, *extra in (
        ('fu-liou-gu', 0.8163255155585885,
         'Explicit opt-in only; never a default. Its radiation stage costs 1.776675x RTE+RRTMGP, the default 4/4 path (160.906 ms against 90.566 ms per call), 0.816326x legacy RRTMG (197.110 ms), and 1.092380x RTE+RRTMGP per whole model step: one RTX 4090, fast numerics, a stock 150 x 150 x 49 WRF input over a 30-minute forecast. Before the 2026-11-02 column storage work it cost 237.612 ms, 2.623639x RTE+RRTMGP. The remaining cost is the native four-stream solar and mixed infrared solves, run one column per thread through a 60 KB per-column workspace; strict replay stays word-identical to native WRF.',
         'flg-speed-4090-summary.json', 'flg-native-work-clock-profile.json',
         {'stage_ratio_to_rte_rrtmgp': 1.7766753515086477,
          'scope': '30-minute real forecast, same card, base and candidate commits back to back'}),
        ('wrf-cam3', 1.8193152090675295,
         'Explicit selection only; never a default. Matched CAM/legacy-RRTMG whole forecasts cost 1.008509x in active wall time, 1.225603x in warm model-step time and 1.819315x in the radiation stage. Native maximum/random cloud-overlap shortwave work accounts for 57.9% of sampled inclusive CAM cycles. Fast column physics uses REAL32; this native work, not a binary64 port path, is the named opt-in cost. These profiles are attribution, not lower bounds.',
         'cam-summary.json', 'cam-native-work-clock-profile.json')):
        row = options[key]
        row['warnings'] = [w for w in row.get('warnings', [])
                           if not w.startswith(('Explicit opt-in only;', 'Explicit selection only;'))] + [warning]
        row.setdefault('extensions', {})['forecast_cost'] = {
            'stage_ratio_to_legacy_rrtmg': ratio,
            'receipt': 'docs/evidence/retime-20261102/' + receipt,
            'native_work_profile': 'docs/evidence/retime-20261102/' + profile,
            'selection_policy': 'explicit-selection-only; never a default',
            'scope': 'three-hour real forecast, same card and commit, AB/BA whole steps',
            **(extra[0] if extra else {})}


def _p3_nc_mp51(registry: dict) -> None:
    """Declare the shared P3 kernel with WRF's predicted-droplet call shape."""
    options = registry["components"]["microphysics"]["options"]
    row = copy.deepcopy(options["p3-mp50"])
    row.update(
        label="P3 one-category, predicted droplet number / MP51",
        selectors={"mp_physics": 51},
        maturity="experimental-runtime", scientific_evidence="none",
        reachability={"state": "component-override"},
        warnings=[
            "MP51 carries prognostic QNCLOUD in addition to MP50's ice and "
            "rain moments and rime mass/volume. The shared CUDA kernels use "
            "WRF's log_predictNc activation, droplet sinks and two-moment "
            "cloud sedimentation. The historical native step31 exact "
            "claim is unqualified: a retained WRF4.7.1 strict O2 recapture "
            "differs in545/87614 words, maximum6945 ULP. Pending arithmetic "
            "repair; packets, every differing word and command are in "
            "datasets/paper/wrf-ports/spd-recapture/mp51/MANIFEST.md. A three-hour "
            "stock-namelist real forecast completed 180 steps; hourly Rust "
            "maps and 13-station observations were reviewed beside native "
            "WRF. This single case establishes operation, not forecast skill. "
            "Fast step timing is 1.088 times MP50 on the same real columns.",
            "Only same-scheme MP51 nests are admitted. Mixed-scheme edges "
            "are refused because the MP50 entry/exit closure does not carry "
            "droplet number. P3 has one ice mass and no QSNOW or QGRAUP.",
        ])
    row["extensions"]["fixed_mode"] = (
        "one ice category, two-moment ice, predicted droplet number, "
        "two prescribed lognormal aerosol modes, SCPF off")
    row["extensions"]["droplet_activation"] = (
        "WRF module_mp_p3.F:819-820, 3313-3339, 3770, 3872-3873, "
        "4009-4066: present(nc_3d) enables activation, droplet sinks "
        "and two-moment cloud sedimentation; DERF is the module's own "
        "routine at 6090-6204.")
    row["extensions"]["unported_wrf_siblings"] = {
        "52": "two ice categories (p3_2category)",
        "53": "three-moment ice (p3_1cat_3mom)"}
    row["extensions"]["runtime_evidence"] = {
        "receipt": "docs/public/wrf-comparison/mp51-real-runtime.md",
        "strict_native_words": 87614, "differing_words": 545,
        "exact_replay_qualification": "withdrawn pending retained recapture repair",
        "retained_capture_manifest": "datasets/paper/wrf-ports/spd-recapture/mp51/MANIFEST.md",
        "forecast_hours": 3, "forecast_steps": 180,
        "nearest_scheme": 50, "fast_step_ratio": 1.087914134399242,
        "startup": "parallel Rust rw-p3; pinned WRF tables",
        "checkpoint": "187 arrays byte-identical after real minute-two resume",
    }
    options["p3-mp51"] = row
def _slab_component(registry: dict) -> None:
    """Own the slab option and its reachable suite, with explicit proof limits."""
    registry["components"]["land_surface"]["options"]["slab"] = {
        "asset_requirements": [],
        "constraints": {
            "admitted_setting_values": {"num_soil_layers": [5]},
            "admitted_setting_values_reasons": {
                "num_soil_layers": "WRF init_soil_depth_1 defines five layers, "
                                   "with thicknesses 1, 2, 4, 8 and 16 cm."},
            "required_settings": {},
            "requires_components": {"surface_layer": [
                "revised-mm5", "classic-mm5", "mynn", "eta-similarity"]},
            "requires_components_reasons": {
                "surface_layer": "The slab consumes UST/CHS/CHS2/CQS2/FLHC/FLQC "
                                 "exchange fields that have no writer when "
                                 "sf_sfclay_physics=0."}},
        "consumers": {
            "reads_glw": True,
            "restart_algorithm_identity": "wrf-v4.7.1-slab-5layer-snow-fast-v3",
            "restart_parameter_bundle": {
                "asset_roles": ["noah_landuse"], "driver_attribute": "slab_params"}},
        "extensions": {"arwen_surface_exchange_structural_requirement": {
            "classification": "WOOF structural constraint; WRF v4.6.1 does not "
                              "prohibit sf_sfclay_physics=0 with an active LSM",
            "reason": "The active LSM consumes exchange fields that have no "
                      "writer when sf_sfclay_physics=0."}},
        "implemented": True,
        "label": "Five-layer thermal diffusion (slab)",
        "maturity": "supported",
        "parameters": {"ifsnow": 1, "num_soil_layers": 5},
        "reachability": {"state": "template"},
        "scientific_evidence": "none",
        "selectors": {"sf_surface_physics": 1},
        "warnings": [
            "Real strict native WRF SLAB1D replay: 73 columns, 1,022 output words, "
            "zero differing words on 2026-11-01. The native v4.7.1 slab source "
            "is byte-identical to v4.6.1. See tools/slab_wrf461_oracle/source_pin.json. "
            "A real three-hour woof go forecast with RRTM/Dudhia, Rust maps beside "
            "native WRF and airport observations, both nest feedback modes, "
            "native warm starts, checkpoint continuation and two independent "
            "ensemble members were exercised. Evidence is case-limited, not "
            "observation skill or whole-model bitwise agreement.",
            "The generated fixture checks 96 columns over 36 calls, all 48,384 "
            "output words at 0 ULP under strict numerics and GPUWM_WRF_EXACT=1. "
            "It is branch coverage, not a real native forecast capture.",
            "The slab integrates five soil heat layers, with no soil water, "
            "vegetation or snow model. Snow cover remains as initialized; "
            "moisture availability remains the land-use table's SLMO.",
            "Declared initialization divergence: sea-ice columns use the land "
            "profile down to 271.4 K. Native real.exe builds slab soil before "
            "converting ice to land, leaving its deepest layer at initial "
            "skin or SST. GPUWM_WRF_EXACT=1 restores native initialization order.",
            "Declared default divergence: every nonzero IFSNOW caps both skin "
            "and top soil under snow. Native WRF caps the top soil only at "
            "IFSNOW=1. GPUWM_WRF_EXACT=1 restores the native switch."]}
    key = "morrison-mp10-ysu-mm5-slab-kf-rte-rrtmgp-v1"
    template = copy.deepcopy(registry["templates"][
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"])
    template["components"]["land_surface"] = "slab"
    template["label"] = "Five-layer thermal diffusion with the standard radiation and boundary-layer suite"
    template["maturity"] = "implemented-unverified"
    template["warnings"] = [
        "The slab column has strict native and generated replay receipts, and "
        "a real RRTM/Dudhia forecast comparison. That evidence does not qualify "
        "this RTE+RRTMGP composition or establish observation skill."]
    registry["templates"][key] = template


def _slab_fixed_template_routes(registry: dict) -> None:
    # Fixed-template menus are derived from route metadata. Offer each slab
    # composition wherever the same composition with another land scheme
    # already has a route, so the registered template has an actual menu row.
    slab_templates = {key: value for key, value in registry["templates"].items()
                      if value.get("components", {}).get("land_surface") == "slab"}
    for template in slab_templates.values():
        template.setdefault("extensions", {})["prepared_land_variant"] = True
    for route_id, route in registry["runner_routes"].items():
        # The native benchmark still binds its fixed Noah input tuple.
        if route_id == _NATIVE_BENCHMARK_ROUTE:
            # Old generated seeds may already offer a slab suite here.
            # The native comparison route has no immutable WRF slab contract.
            for offered in (route.get("source_template_ids") or {}).values():
                offered[:] = [key for key in offered if key not in slab_templates]
            continue
        if route.get("mode") != "fixed-template":
            continue
        for offered in (route.get("source_template_ids") or {}).values():
            for key, template in slab_templates.items():
                wanted = {k: v for k, v in template["components"].items()
                          if k != "land_surface"}
                if key not in offered and any(
                    {k: v for k, v in registry["templates"][old]["components"].items()
                     if k != "land_surface"} == wanted for old in offered):
                    offered.append(key)
                base = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
                if key in offered and base in offered:
                    offered.remove(key)
                    offered.insert(offered.index(base) + 1, key)


def _morrison_aero_mp40(registry):
    options=registry["components"]["microphysics"]["options"]
    row=copy.deepcopy(options["morrison-mp10"])
    row.update(label="Morrison aerosol / MP40",selectors={"mp_physics":40},maturity="experimental-runtime",scientific_evidence="none",parameters={"moist":True,"moist_cq":True,"morr_rimed_ice":1,"aercu_opt":0},reachability={"state":"component-override"})
    row.pop("consumers",None)
    row["warnings"]=["The CESM activation column port follows pinned WRF T0; a melting-point correction is an explicit column-only diagnostic arm. Positive aercu_opt needs the MSKF monthly aerosol provider. Column agreement does not establish forecast skill."]
    row["aerosol_state"]={"prescribed_input":"aerocu","species":["dust1","dust2","dust3","dust4","seasalt","sulfate","phobcar","phibcar","phoocar","phiocar"],"units":"ug m-3","activation_diagnostics":[f"CCN{n}_GS" for n in range(1,8)],"forecast_input_active":False}
    options["morrison-aero-mp40"]=row

_VERTICAL_BOUND_SOURCES["microphysics"]["morrison-aero-mp40"]=("Morrison aerosol microphysics","MORRISON_AERO_VERTICAL_LEVEL_BOUNDS")
_MOMENT_ROWS[40]=copy.deepcopy(_MOMENT_ROWS[10])
_MOMENT_ROWS[40].update(name="Morrison aerosol",mass_only=["qv","qc"],pairs=[p for p in _MOMENT_ROWS[40]["pairs"] if p["species"]!="cloud"],unpaired=["nc"],repair_authority=None)
# MP40 transports QNC through the ordinary history map and nest inventory.
# Its own restart identity binds both rimed-ice selection and aercu_opt=0.
_OFFLINE_CHILD_ROWS[40]={"same_scheme":True,"refusal":None}
_RING_STATE_ROWS[40]=list(_RING_STATE_ROWS[10])
_RING_SURFACE_ROWS[40]=list(_RING_SURFACE_STANDARD)

#: The Pleim-Xiu surface layer's warnings (sf_sfclay_physics=7).  The
#: measured numbers are the parity test's own
#: (tests/test_pxsfclay_wrf461_parity.py).
_PX_SFCLAY_WARNINGS = (
    "WRF v4.6.1 Pleim-Xiu surface layer and fractional sea-ice wrapper. "
    "The module and wrapper are unchanged in WRF v4.7.1. Strict module "
    "replay uses GPUWM_WRF_EXACT=1; default arithmetic uses CUDA math. "
    "The packaged real-forecast captures replay all 43,200 output words "
    "with zero differences under the native referee's glibc 2.43. "
    "Constructed columns separately cover the fractional sea-ice wrapper. "
    "These are conformance checks; forecast and observation evidence "
    "have their own receipts.",
    "WRF sets isfc=7 and refuses YSU, Shin-Hong, MYNN and MYJ with this "
    "surface layer. PBL-off, UW and SASE consume its published fields. "
    "This layer writes no FM, FH, USTM, TH2, T2 or Q2. The land model "
    "forms the 2 m fields from CHS2/CQS2.",
)




def _cam5_components(registry):
    from woof.microphysics_schemes import CAMMGMP
    common={"implemented":True,"maturity":"implemented-unverified",
            "scientific_evidence":"none","asset_requirements":[],
            "reachability":{"state":"component-override"},
            "parameters":{"moist":True,"moist_cq":True},
            "warnings":["Native captured module replay and a real three-hour coupled forecast are documented in docs/public/cam5-suite.md. MP11 retains 71 declared native trig rounding differences, exactly witnessed against correctly rounded MIT CORE-MATH and independent high-precision bounds; no glibc code ships. CAM binary64 stages cost more than the nearest cheaper schemes; the measured native work and optimizations are documented. One forecast does not establish broad forecast skill."]}
    registry["components"]["microphysics"]["options"]["cam-mg-mp11"]={
        **copy.deepcopy(common),"label":CAMMGMP.label,"selectors":{"mp_physics":11},
        "constraints":{"required_settings":{"moist":True,"moist_cq":True}},
        "extensions":{"source":{"repository":"https://github.com/wrf-model/WRF",
            "commit":"f52c197ed39d12e087d02c50f412d90d418f6186","licence":"WRF public domain"},
            "prognostic_species":{"transported":["qi","qs","nc","ni","nr","ns"],
                "diagnostic":list(CAMMGMP.diagnostic_species)},
            "column_oracle_evidence":{"record":"tools/cam5_wrf471_oracle/README.md"}}}
    registry["components"]["cumulus"]["options"]["cam-deep"]={
        **copy.deepcopy(common),"label":"CAM Zhang-McFarlane deep convection",
        "selectors":{"cu_physics":7},"extensions":{},
        "constraints":{"required_settings":{"moist":True},
            "requires_components":{"pbl":["myj","uw"]},
            "requires_components_reasons":{"pbl":"WRF get_tpert reads the MYJ TKE and u* (Eta similarity) or the CAM UW PBL perturbation and stops with wrf_error_fatal under any other PBL (module_cu_camzm_driver.F:978-1018)"}}}
    registry["components"]["microphysics"]["options"]["cam-mg-mp11"]["warnings"].append(
        "Explicit opt-in only, never a default: measured fast stage costs 21.3x MP10.")
    registry["components"]["cumulus"]["options"]["cam-deep"]["warnings"].append(
        "Explicit opt-in only, never a default: measured fast stage costs 6.0x CU96.")
    params=registry["parameters"]
    # Deng requires cloud coupling off independently on each shallow domain.
    params["icloud_bl"]["per_domain"] = True
    params["shcu_physics"]={"implemented":True,"type":"integer","default":0,
        "per_domain":True,"enum":[0,1,2,3,4,5],"consuming_read":"woof/core/shallow_runtime.py",
        "schemes":{"0":{"label":"off"},
            "5":{"label":"Deng shallow convection (opt-in; measured 3.4813x NSCV wall, 34.9493x GPU)","algorithm_identity":"deng-shcu5-wrf471-v1","cadence":"every-model-step","maturity":"supported",
                "extensions":{"qualification":{"opt_in_only":True,"default_eligible":False,
                    "nearest":"NSCV shallow (shcu_physics=4)","stage_wall_ratio":3.481310617315788,
                    "stage_gpu_ratio":34.94929707271424,"card":"RTX 5070 Ti, shared",
                    "record":"docs/shallow-wrf471.md","native_extra_work":"Deng cloud solver, mixing/freezing, condensate loading and time averaging; retained GNU REAL32 gprof",
                    "strict_replay_differing":0,"strict_replay_words":6359760}}},
            "1":{"label":"G3 shallow selector (native no-op)","algorithm_identity":"g3-shcu1-wrf471-noop-v1","cadence":"every-model-step","maturity":"supported"},
            "3":{"label":"GRIMS shallow convection","algorithm_identity":"grims-shcu3-wrf471-v1","cadence":"every-model-step","maturity":"supported"},
            "4":{"label":"NSCV shallow convection","algorithm_identity":"nscv-shcu4-wrf471-v1","cadence":"every-model-step","maturity":"supported"},
            "2":{"label":"CAM UW shallow convection (opt-in, measured 97.8x NSCV shallow cost; never a default)",
            "algorithm_identity":"camuwshcu-wrf471-r8-v1","cadence":"every-model-step"}}}
    for key,default in (("accum_mode",1.e9),("aitken_mode",3.e8),("coarse_mode",2.e5)):
        params[key]={"implemented":True,"type":"number","default":default,"minimum":0.,
            "per_domain":False,"read_when":{"mp_physics":11},
            "consuming_read":"woof/core/cammgmp.py"}
    for route in registry["runner_routes"].values():
        choices=route.get("allowed_component_options",{})
        for component,name in (("microphysics","cam-mg-mp11"),("cumulus","cam-deep")):
            if component in choices and name not in choices[component]:choices[component].append(name)
        keys=route.get("allowed_parameter_keys",[])
        for key in ("shcu_physics","accum_mode","aitken_mode","coarse_mode"):
            if key not in keys:keys.append(key)
        keys.sort()



def _kain_fritsch_cu99(registry: dict) -> None:
    """Register WRF's previous Kain-Fritsch (cu_physics=99, KFSCHEME).

    ``implemented: true`` -- ``woof/core/kernels/kf_old.cu`` is a
    WRF-order transcription of the byte-frozen v4.6.1
    ``phys/module_cu_kf.F`` and ``woof.core.physics.initialize_physics``
    binds ``KainFritsch(scheme=99)`` on the selector.

    ``maturity: implemented-unverified`` -- the column is measured against
    the retained fixture and strict real-run captures. A three-hour default
    forecast reaches the public go door. CU99 scheduling and tile submission
    run in Rust; ordinary and packed member bindings are checked. A short
    forecast and airport comparisons do not establish observation skill.

    ``reachability: component-override`` -- computed, not chosen: no
    template selects cu_physics=99, so the way in is a namelist import or
    a per-domain override.
    """
    options = registry["components"]["cumulus"]["options"]
    option = copy.deepcopy(options["kain-fritsch"])
    option["asset_requirements"] = []
    option["consumers"]["restart_algorithm_identity"] = (
        "kain-fritsch-cu99-wrf461-kfcps-v1")
    option["label"] = "Kain-Fritsch (previous scheme, cu_physics=99)"
    option["maturity"] = "implemented-unverified"
    option["reachability"] = {"state": "component-override"}
    option["selectors"] = {"cu_physics": 99}
    option["parameters"] = {"cudt_minutes": 5.0, "kf_edrates": 0}
    option["warnings"] = [
        "BITWISE COLUMN EVIDENCE. tools/kf_old_wrf461_"
        "oracle drives the byte-unmodified WRF v4.6.1 phys/module_cu_kf.F "
        "through KFCPS over 256 columns (1222 of its 1324 executable lines "
        "reached), and woof/core/kernels/kf_old.cu reproduces every "
        "output -- the six tendencies, RAINCV, PRATEC, NCA, W0AVG and the "
        "kf_edrates profiles -- at zero differing words with "
        "GPUWM_WRF_EXACT=0 and 1 under the test battery's "
        "GPUWM_NUMERICS=strict compiler policy "
        "(tests/test_kf_old_wrf461_parity.py). The real native capture "
        "instrumenter and strict replay are in the same oracle directory. "
        "See docs/cumulus-cu99.md for the forecast evidence and its limits.",
        "WRF's timer rules are this scheme's, not KF-eta's: a column is "
        "held while NCA > 0.5*DT (phys/module_cu_kf.F:235) and its tendencies "
        "are zeroed once NINT(NCA/DT) <= 0 "
        "(phys/module_physics_addtendc.F:2306 in WRF v4.7.1), one step later than "
        "cu_physics=1. The hydrometeor fold follows F_QI, F_QS and F_QR "
        "of the microphysics' moist package (phys/module_cu_kf.F:297-325, "
        ":1983-2023).",
        "WRF's three wrf_error_fatal sites in this scheme (negative "
        "surface vapour after the moisture borrow, a mass-flux imbalance, "
        "a column water error over 0.05 %) stop the run with WRF's own "
        "message, as WRF does.",
        "With mp_physics=0, WRF's passiveqv package has no "
        "qc, so the condensate this scheme detrains has nowhere to go and "
        "is dropped by the coupling scalar gate. Its raw RQCCUTEN remains "
        "available for module replay and held-state restart.",
        "kf_edrates=1 publishes UDR_KF/DDR_KF/UER_KF/DER_KF/TIMEC_KF "
        "from the last due KFCPS call, using WRF Registry metadata. These "
        "fields retain their values between due calls and across restart.",
        "CU99 per-step cadence and compatibility-clock advancement run "
        "in the production Rust planner, with domain-local restored clocks "
        "and adaptive event overrides. Bounded CUDA tile submissions also "
        "run in Rust; Python remains launch orchestration. Resident health "
        "uses asynchronous polling and mandatory publication drains. "
        "Concurrent ordinary ensembles carry this scheme, and the "
        "original-member MYNN/RUC/RRTMGP component joins its columns with "
        "private held carriers. The default Thompson/Noah/YSU/MM5 packed "
        "graph also keeps private CU99 carriers and the stock rain-number "
        "seed. No whole-model bitwise parity or formal observation skill "
        "is claimed.",
    ]
    options[KF_OLD_OPTION_ID] = option


# WRF ETAMPNEW (mp95): retained total-condensate state is distinct from MP5.
_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON[95] = "Eta Ferrier has no qualified RRTMGP optics; use native legacy RRTMG"
_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON[27] = "UDM's optional native radius routine receives uninitialized den1d (WRF 4.7.1 module_mp_udm.F:289,411); use native legacy RRTMG"
_VERTICAL_BOUND_SOURCES["microphysics"]["etanew-mp95"] = ("Eta Ferrier microphysics", "ETANEW_VERTICAL_LEVEL_BOUNDS")
_MOMENT_ROWS[95] = {"name": "Eta Ferrier", "mass_only": ["qv", "qc", "qr", "qs"], "pairs": [], "unpaired": ["qt"], "repair_authority": None, "q_threshold": 1.0e-12}
_OFFLINE_CHILD_ROWS[95] = {"same_scheme": False, "refusal": "No offline mapping of total qt and carried phase fractions is qualified."}
_RING_STATE_ROWS[95] = ["thp", "qv", "qc", "qr", "qs", "qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy"]
_RING_SURFACE_ROWS[95] = ["mp_rainnc", "mp_rainncv", "mp_sr"]


def build(registry: dict) -> dict:
    """Apply this pass's tables to ``registry`` in place and return it."""
    _sms_3dtke(registry)
    from woof.core.physics_inventory import PBL_RQI_MICROPHYSICS
    _slab_component(registry)
    from tools.seed_physics_options import load_seeds, seed_options
    registry = seed_options(registry, load_seeds())
    from woof.spectral_nudging.surface_contract import registry_rows as surface_fdda_rows
    registry["analysis_nudging"] = surface_fdda_rows()
    from woof.knobs.physics import registry_row as physics_knob_registry_row
    registry["authority"]["regional_tuning_controls"] = physics_knob_registry_row()
    _new_goddard(registry)
    _kfcup_metadata(registry)

    _fu_liou_gu(registry)
    _retimed_radiation_costs(registry)
    from woof.lightning_contract import LIGHTNING_DEFAULTS, LIGHTNING_OPTIONS, FIELD_METADATA
    registry["diagnostic_packages"] = {"lightning_option": {str(k): dict(v) for k,v in LIGHTNING_OPTIONS.items()}}
    # Preserve the native Noah-MP output metadata supplied by its owner.
    # The package builders below replace the groups they own; resetting this
    # dictionary erased the newer Noah-MP output fields during regeneration.
    lightning_proofs = registry["diagnostic_packages"]["lightning_option"]
    lightning_proofs["3"]["replay_qualification"] = {
        "state": "retained-native-module-replay",
        "historical_missing_capture_words": 256,
        "historical_claim": "Withdrawn; not restored by this separate recapture.",
        "may_driver_words": 342018,
        "active_october_lpi_words": 4096,
        "active_october_native_nonzero_words": 630,
        "differing_words": 0,
        "limits": "Three complete May driver calls compare all five planes and clock, with zero rates. Separate hour-one October MP6 capture compares LPI on all sixteen 16x16 native patches. October flash rates were not captured and are not qualified. No whole-model identity or flash-skill claim.",
        "source_snapshots": ["4f618604dc54", "bd750d51ae66", "c69a56084ad7"],
        "evidence": "datasets/paper/wrf-ports/spd-recapture/ltng3/MANIFEST.md",
        "old_failed_attempts": "datasets/paper/wrf-ports/port-diag-climate-ltng/provenance/lightning3-recapture-recipe.md",
    }
    lightning_proofs["11"]["replay_qualification"] = {
        "state": "retained-native-module-replay",
        "words_per_abi_and_source_snapshot": 342018,
        "differing_words": 0,
        "capture_abis": [47103, 47104],
        "source_snapshots": ["83aeb0f3603e", "8f0b2cf1d39d"],
        "limits": "All five output planes and clock compared; early zero-rate calls. Historical47104 label corrected to47103; four planted-bit controls caught. No whole-model identity or flash-skill claim.",
        "evidence": "datasets/paper/wrf-ports/port-diag-climate-ltng/MANIFEST.json",
    }
    registry.setdefault("diagnostic_state", {})["lightning"] = FIELD_METADATA
    # The current output owner uses the domain-relative atmospheric clock;
    # an older diagnostic port declared a separate epoch that no longer exists.
    from woof.core.noahmp_output import CARRIERS as noahmp_output_carriers
    from woof.io.wrf_output_schema import NOAHMP_OUTPUT_FIELDS
    diagnostic_state = registry["diagnostic_state"]
    diagnostic_state.pop("noahmp_output_clock", None)
    diagnostic_state["noahmp_output_fields"] = {
        name: {
            "netcdf_name": field.netcdf_name,
            "units": field.units,
            "description": field.description,
            "source": "Registry/" + field.registry + " at WRF 4.7.1",
            "allocation": "sf_surface_physics=4 and noahmp_output=3",
            "shape": "surface", "history": True, "restart": True,
        }
        for name in dict.fromkeys(noahmp_output_carriers.values())
        if (field := NOAHMP_OUTPUT_FIELDS.get(name)) is not None
    }
    from woof.core.clwrf import registry_rows as clwrf_registry_rows
    clwrf_registry_rows(registry)
    from woof.core.rasm import registry_rows as rasm_rows
    rasm_rows(registry)
    from woof.core.solar_diagnostics import registry_rows as solar_rows
    solar_rows(registry)
    from woof.core.hail import registry_rows as hail_registry_rows
    hail_registry_rows(registry)
    from woof.core.nwp_diagnostics import registry_rows as nwp_registry_rows
    nwp_registry_rows(registry)
    from woof.afwa_config import registry_rows as afwa_registry_rows
    afwa_registry_rows(registry)
    from woof.core.output_diagnostics import registry_rows as precipitation_registry_rows
    precipitation_registry_rows(registry)
    from woof.level_output_config import registry_rows as level_registry_rows
    level_registry_rows(registry)

    from woof.fddaobs.registry import registry_rows as fddaobs_registry_rows
    fddaobs_registry_rows(registry)
    for key, value in LIGHTNING_DEFAULTS.items():
        row = {"type": "integer" if isinstance(value, int) else "number",
               "default": value, "per_domain": True,
               # do_radar_ref is read by the lightning admission rule (PR92
               # options 1 and 2 need the reflectivity it turns on).
               "consuming_read": ("woof/lightning_contract.py"
                                  if key == "do_radar_ref"
                                  else "woof/core/lightning.py")}
        if key == "lightning_option": row["enum"] = list(LIGHTNING_OPTIONS)
        elif key == "iccg_method": row["enum"] = list(range(5))
        elif key == "cellcount_method": row["enum"] = list(range(3))
        registry["parameters"][key] = row
    _rename_template_ids(registry)
    _temf_pair(registry)
    # Registered source tables determine which prepared inputs each runner
    # accepts. Carrying yesterday's registry source list omits a newly
    # mapped source even after its decoder and forecast route are usable.
    from woof.prepared_single_domain_forecast import SUPPORTED_SOURCES as single_sources
    from woof.prepared_domain_tree_forecast import SUPPORTED_SOURCES as tree_sources
    registry["runner_routes"]["tools.prepared_single_domain_forecast"]["source_ids"] = sorted(single_sources)
    registry["runner_routes"]["tools.prepared_domain_tree_forecast"]["source_ids"] = sorted(tree_sources)
    registry["components"]["land_surface"]["options"]["ssib"] = (
{   'asset_requirements': [],
    'column_contract': {   'admission': {   'bl_pbl_physics': {   'reason': 'module_physics_init.F:3544: '
                                                                            'use ysu scheme with SSiB',
                                                                  'values': [1]},
                                            'fractional_seaice': {   'reason': 'module_surface_driver.F:1483-1486: '
                                                                               'SSiB requires '
                                                                               'fractional_seaice=1',
                                                                     'values': [1]},
                                            'num_soil_layers': {   'reason': 'share/module_check_a_mundo.F:3571-3572: '
                                                                             'SSiB uses 3 soil layers',
                                                                   'values': [3]},
                                            'ra_lw_physics': {   'reason': 'module_physics_init.F:3547-3548: '
                                                                           'lw_phys=1,3,4',
                                                                 'values': [1, 3, 4]},
                                            'ra_sw_physics': {   'reason': 'module_physics_init.F:3549-3550: '
                                                                           'sw_phys=1,3,4',
                                                                 'values': [1, 3, 4]},
                                            'sf_lake_physics': {   'reason': 'Lake initialization uses '
                                                                             'an ice threshold of 0.5 '
                                                                             'for non-RUC land surfaces '
                                                                             '(woof/core/physics.py '
                                                                             'initialize_physics). SSiB '
                                                                             'requires 0.02, so this '
                                                                             'combination would '
                                                                             'classify fractional ice '
                                                                             'incorrectly. Keep '
                                                                             'sf_lake_physics=0.',
                                                                   'values': [0]},
                                            'sf_sfclay_physics': {   'reason': 'module_physics_init.F:3542: '
                                                                               'use sfclay scheme with '
                                                                               'SSiB',
                                                                     'values': [1]}},
                           'forcing_layout': [   'rainncv',
                                                 'raincv',
                                                 'glw',
                                                 'coszen',
                                                 'swdown',
                                                 'swvisdir',
                                                 'swvisdif',
                                                 'swnirdir',
                                                 'swnirdif',
                                                 'dz8w1',
                                                 'u1',
                                                 'v1',
                                                 'qv1',
                                                 't1',
                                                 'p1',
                                                 'psfc',
                                                 'hfx_sea',
                                                 'qfx_sea',
                                                 'lh_sea',
                                                 'tsk_sea',
                                                 'xice',
                                                 'xice_save',
                                                 'julian',
                                                 't2_sfclay'],
                           'history_schema_overrides': {   'ISNOW': [   'ISNOW',
                                                                        'i4',
                                                                        '',
                                                                        'ssib-snow ISNOW',
                                                                        '',
                                                                        'Registry/registry.ssib:30',
                                                                        False]},
                           'integer_state': ['isnow'],
                           'mminlu_codes': {   'MODIFIED_IGBP_MODIS_NOAH': 4,
                                               'MODIS': 3,
                                               'SSIB': 1,
                                               'USGS': 2},
                           'module': 'woof.core.ssib',
                           'refused_landuse': {   'NLCD40': 'SSIB does not work with NLCD data. Stop. '
                                                            '(module_physics_init.F:3554)'},
                           'state_bindings': {   'albedo': 'albedo',
                                                 'alswnirdif': 'alswnirdif',
                                                 'alswnirdir': 'alswnirdir',
                                                 'alswvisdif': 'alswvisdif',
                                                 'alswvisdir': 'alswvisdir',
                                                 'bio1': 'bio1',
                                                 'bio2': 'bio2',
                                                 'bio3': 'bio3',
                                                 'bio4': 'bio4',
                                                 'blo1': 'blo1',
                                                 'blo2': 'blo2',
                                                 'blo3': 'blo3',
                                                 'blo4': 'blo4',
                                                 'br': 'br',
                                                 'bto1': 'bto1',
                                                 'bto2': 'bto2',
                                                 'bto3': 'bto3',
                                                 'bto4': 'bto4',
                                                 'bwo1': 'bwo1',
                                                 'bwo2': 'bwo2',
                                                 'bwo3': 'bwo3',
                                                 'bwo4': 'bwo4',
                                                 'canwat': 'canwat',
                                                 'cto1': 'cto1',
                                                 'cto2': 'cto2',
                                                 'cto3': 'cto3',
                                                 'cto4': 'cto4',
                                                 'dzo1': 'dzo1',
                                                 'dzo2': 'dzo2',
                                                 'dzo3': 'dzo3',
                                                 'dzo4': 'dzo4',
                                                 'fio1': 'fio1',
                                                 'fio2': 'fio2',
                                                 'fio3': 'fio3',
                                                 'fio4': 'fio4',
                                                 'flo1': 'flo1',
                                                 'flo2': 'flo2',
                                                 'flo3': 'flo3',
                                                 'flo4': 'flo4',
                                                 'grdflx': 'grdflx',
                                                 'gz1oz0': 'gz1oz0',
                                                 'hfx': 'hfx',
                                                 'ho1': 'ho1',
                                                 'ho2': 'ho2',
                                                 'ho3': 'ho3',
                                                 'ho4': 'ho4',
                                                 'isnow': 'isnow',
                                                 'lh': 'lh',
                                                 'q2': 'q2',
                                                 'qfx': 'qfx',
                                                 'sfcevp': 'sfcevp',
                                                 'sfcrunoff': 'sfcrunoff',
                                                 'smois1': 'smois:0',
                                                 'smois2': 'smois:1',
                                                 'smois3': 'smois:2',
                                                 'snow': 'snow',
                                                 'snowden': 'snowden',
                                                 'snowdepth': 'snowdepth',
                                                 'snowh': 'snowh',
                                                 'ssib_br': 'ssib_br',
                                                 'ssib_cm': 'ssib_cm',
                                                 'ssib_eci': 'ssib_eci',
                                                 'ssib_ect': 'ssib_ect',
                                                 'ssib_egi': 'ssib_egi',
                                                 'ssib_egs': 'ssib_egs',
                                                 'ssib_egt': 'ssib_egt',
                                                 'ssib_fh': 'ssib_fh',
                                                 'ssib_fm': 'ssib_fm',
                                                 'ssib_ghf': 'ssib_ghf',
                                                 'ssib_lai': 'ssib_lai',
                                                 'ssib_ldn': 'ssib_ldn',
                                                 'ssib_lhf': 'ssib_lhf',
                                                 'ssib_lup': 'ssib_lup',
                                                 'ssib_sdn': 'ssib_sdn',
                                                 'ssib_shc': 'ssib_shc',
                                                 'ssib_shf': 'ssib_shf',
                                                 'ssib_shg': 'ssib_shg',
                                                 'ssib_sup': 'ssib_sup',
                                                 'ssib_vcf': 'ssib_vcf',
                                                 'ssib_veg': 'ssib_veg',
                                                 'ssib_wat': 'ssib_wat',
                                                 'ssib_z00': 'ssib_z00',
                                                 'ssibxdd': 'ssibxdd',
                                                 'swe': 'swe',
                                                 'swnirdif': 'swnirdif',
                                                 'swnirdir': 'swnirdir',
                                                 'swvisdif': 'swvisdif',
                                                 'swvisdir': 'swvisdir',
                                                 't2': 't2',
                                                 'tkair': 'tkair',
                                                 'tsk': 'tsk',
                                                 'tsk_save': 'tsk_save',
                                                 'tslb1': 'tslb:0',
                                                 'tslb2': 'tslb:1',
                                                 'tslb3': 'tslb:2',
                                                 'tssn1': 'tssn1',
                                                 'tssn2': 'tssn2',
                                                 'tssn3': 'tssn3',
                                                 'tssn4': 'tssn4',
                                                 'tssno1': 'tssno1',
                                                 'tssno2': 'tssno2',
                                                 'tssno3': 'tssno3',
                                                 'tssno4': 'tssno4',
                                                 'u10': 'u10',
                                                 'ust': 'ust',
                                                 'v10': 'v10',
                                                 'wo1': 'wo1',
                                                 'wo2': 'wo2',
                                                 'wo3': 'wo3',
                                                 'wo4': 'wo4',
                                                 'znt': 'znt'},
                           'static_layout': ['xland', 'ivgtyp', 'xlat', 'mminlu', 'ra_sw', 'dtbl'],
                           'step': 'step',
                           'tables': 'load_tables',
                           'xice_threshold': 0.02},
    'constraints': {   'admitted_setting_values': {   'bl_pbl_physics': [1],
                                                      'fractional_seaice': [1],
                                                      'num_soil_layers': [3],
                                                      'ra_lw_physics': [1, 3, 4],
                                                      'ra_sw_physics': [1, 3, 4],
                                                      'sf_lake_physics': [0],
                                                      'sf_sfclay_physics': [1]},
                       'admitted_setting_values_reasons': {   'bl_pbl_physics': 'module_physics_init.F:3544: '
                                                                                'use ysu scheme with '
                                                                                'SSiB',
                                                              'fractional_seaice': 'module_surface_driver.F:1483-1486: '
                                                                                   'SSiB requires '
                                                                                   'fractional_seaice=1',
                                                              'num_soil_layers': 'share/module_check_a_mundo.F:3571-3572: '
                                                                                 'SSiB uses 3 soil '
                                                                                 'layers',
                                                              'ra_lw_physics': 'module_physics_init.F:3547-3548: '
                                                                               'lw_phys=1,3,4',
                                                              'ra_sw_physics': 'module_physics_init.F:3549-3550: '
                                                                               'sw_phys=1,3,4',
                                                              'sf_lake_physics': 'Lake initialization '
                                                                                 'uses an ice threshold '
                                                                                 'of 0.5 for non-RUC '
                                                                                 'land surfaces '
                                                                                 '(woof/core/physics.py '
                                                                                 'initialize_physics). '
                                                                                 'SSiB requires 0.02, '
                                                                                 'so this combination '
                                                                                 'would classify '
                                                                                 'fractional ice '
                                                                                 'incorrectly. Keep '
                                                                                 'sf_lake_physics=0.',
                                                              'sf_sfclay_physics': 'module_physics_init.F:3542: '
                                                                                   'use sfclay scheme '
                                                                                   'with SSiB'},
                       'refused_when': [   {   'reason': 'RTE+RRTMGP lacks the four SSiB spectral '
                                                         'surface flux and albedo carriers.',
                                               'remedy_label': "Set ra_rrtmg_variant='rrtmg_legacy'.",
                                               'remedy_settings': {'ra_rrtmg_variant': 'rrtmg_legacy'},
                                               'settings': {   'ra_rrtmg_variant': ['rte-rrtmgp'],
                                                               'ra_sw_physics': [4]}}]},
    'consumers': {   'reads_glw': True,
                     'restart_algorithm_identity': 'ssib-wrf-v4.6.1-v1-seaice-unit-repair',
                     'restart_parameter_bundle': {   'asset_roles': ['ssib_vegetation', 'noah_landuse'],
                                                     'driver_attribute': 'column_surface_params'}},
    'implemented': True,
    'label': 'SSiB land surface (opt-in; 1.942x Noah lower-tail stage cost)',
    'maturity': 'supported',
    'parameters': {'fractional_seaice': 1, 'num_soil_layers': 3},
    'reachability': {'state': 'component-override'},
    'scientific_evidence': 'none',
    'selectors': {'sf_surface_physics': 8},
    'warnings': [   'Strict replay of real WRF v4.6.1 forecast inputs matches all 13,320 state words. '
                    'A three-hour 150x150x49 forecast, native restart, checkpoint continuation, '
                    'SSiB nest feedback and two ordinary ensemble members are qualified in '
                    'docs/public/receipts/ssib-native-qualification.json. This is one case, '
                    'not general observation-skill certification. Explicit opt-in only: '
                    'same-card interleaved whole forecasts give median/p10/minimum ratios '
                    '0.511/1.942/1.390x Noah. The conservative audit cost is 1.942x, not '
                    'the favorable median alone. Native REAL32 gprof shows extra canopy '
                    'radiation, coupled temperature and repeated aerodynamic solves. '
                    'See docs/public/receipts/retime-pbl-forecast.json and pbl-land-stage.md.',
                    'Strict uses zero for WRF automatic locals read before assignment. Default repairs '
                    'sea-ice snow and canopy-water units.']}
    )
    registry["components"]["land_surface"]["options"]["ssib"]["column_contract"]["soil_ingest"] = {"method": "wrf_init_soil_2_real", "depths_m": [0.0, 0.05, 1.05], "thickness_m": [0.05, 1.0, 0.0], "texture_downscale": False, "wrf_source": "share/module_soil_pre.F:855-864,1269-1285"}
    contract = registry["components"]["land_surface"]["options"]["ssib"]["column_contract"]
    # Use the existing documentation field, preserving dataset selector keys.
    contract["refused_landuse"] = {
        key: {"reason": value} for key, value in contract["refused_landuse"].items()}
    admission = registry["components"]["land_surface"]["options"]["ssib"]["column_contract"]["admission"]
    reasons = {
        "sf_sfclay_physics": "module_physics_init.F:3542: SSiB requires the revised MM5 transfer interface; other surface layers cannot supply it. Select sf_sfclay_physics=1.",
        "bl_pbl_physics": "module_physics_init.F:3544: SSiB requires its WRF YSU coupling; other PBL choices are not admitted. Select bl_pbl_physics=1.",
        "ra_lw_physics": "module_physics_init.F:3547-3548: SSiB requires supported downward longwave; other choices would leave its surface budget without that forcing. Select ra_lw_physics in [1,3,4].",
        "fractional_seaice": 'module_surface_driver.F:1483-1486: SSiB requires fractional_seaice=1; WRF stops with wrf_error_fatal when SSiB runs without fractional sea ice, because its surface driver has no all-or-nothing ice path. Set fractional_seaice=1.',
        "ra_sw_physics": "module_physics_init.F:3549-3550: SSiB requires supported shortwave band forcing; other choices cannot supply it. Select ra_sw_physics in [1,3,4].",
    }
    for key, reason in reasons.items():
        admission[key]["reason"] = reason
    # The plan door reads the same sentence as the run door.
    registry["components"]["land_surface"]["options"]["ssib"]["constraints"][
        "admitted_setting_values_reasons"]["fractional_seaice"] = (
            reasons["fractional_seaice"])

    _urban_component(registry)
    _surface_coupling_warnings(registry)
    _thompson_aerosol_mp28(registry)
    _thompson_hail_mp38(registry)
    _milbrandt2mom_mp9(registry)
    _wdm6_mp16(registry)
    _morrison_aero_mp40(registry)
    _ntu_mp56(registry)
    _tempo_named(registry)
    _cam5_components(registry)
    _lin_mp2(registry)
    _ferrier_mp5(registry)
    _wsm3_mp3(registry)
    _single_moment_column_options(registry)
    _gsfcgce_mp97(registry)
    _wsm6r_mp106(registry)
    _wsm7_mp24(registry)
    _sbu_ylin_mp13(registry)
    _wdm5_mp14(registry)
    _wdm7_mp26(registry)
    registry["components"]["microphysics"]["options"][WDM6_OPTION_ID]["extensions"]["family_siblings_refused"] = {}
    _p3_three_moment_mp53(registry)
    _gsfcgce4ice_mp7(registry)
    _ferrier_mp15(registry)



    _kain_fritsch_cu99(registry)
    from tools.morrison_wrf461_oracle.patch_registry_morrison import (
        MORRISON_WARNINGS)
    registry["components"]["microphysics"]["options"]["morrison-mp10"][
        "warnings"] = list(MORRISON_WARNINGS)
    _fast_sbm_admission(registry)
    _p3_nc_mp51(registry)
    # After every microphysics option is registered, because it walks them.
    _rte_rrtmgp_cloud_optics_constraints(registry)
    _tke_nest_child(registry)
    registry["authority"][
        "wrf_v461_compatibility_matrix"
    ] = _wrf_compatibility_authority()
    registry["authority"]["reachability_declaration"] = (
        "components.<component>.options.<option>.reachability declares how "
        "a user can select the option: 'template' through a registered base "
        "template, 'component-override' through either a route's full "
        "allowed_component_overrides or its option-scoped "
        "allowed_component_options, 'expert-template' only through a route's "
        "expert_template_ids, with expert_acknowledgement_id advisory, and "
        "'unreachable' not normally reachable -- which must name a blocker. "
        "implemented and reachable are independent. "
        "tests/test_registry_reachability.py recomputes every state.")

    # Named native-HRRR Kessler product used by the end-to-end ratification
    # probe.  Its source route is intentionally HRRR-only: no other source
    # inherits evidence from that run.
    kessler_id = "kessler-mp1-ysu-mm5-noah-dudhia-v1"
    wsm6_id = "wsm6-ysu-mm5-noah-no-radiation-v1"
    kessler = copy.deepcopy(registry["templates"][wsm6_id])
    kessler["components"]["microphysics"] = "kessler-mp1"
    kessler["label"] = (
        "Kessler warm rain + YSU + classic MM5 + Noah + Dudhia SW")
    kessler["maturity"] = "implemented-unverified"
    kessler["warnings"] = [
        "Native-HRRR Kessler admission is bound to the Lane C one-hour "
        "end-to-end probe and its frozen-species discard receipt; it is not "
        "evidence for any non-HRRR source route."
    ]
    registry["templates"][kessler_id] = kessler
    registry["components"]["microphysics"]["options"][
        "kessler-mp1"]["reachability"] = {"state": "template"}
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id].setdefault(
            "source_template_ids", {}).setdefault("hrrr", [])
        if kessler_id in declared:
            declared.remove(kessler_id)
        position = declared.index(wsm6_id) + 1 if wsm6_id in declared else 0
        declared.insert(position, kessler_id)

    # WRF v4.6.1's actual PBL/surface-layer law is in
    # phys/module_physics_init.F:3699-3704,3837-3839.  In particular MYNN
    # PBL accepts the revised and classic MM5 surface layers, and the MYNN
    # surface layer is legal with PBL off.  These declarative constraints
    # mirror the same 16-cell table used by runtime admission.
    pbl_options = registry["components"]["pbl"]["options"]
    # Retire only the old ctopo-absent warning after the selected YSU repair.
    # Keep its independent WRF comparison and all other qualification limits.
    pbl_options["ysu"]["warnings"] = [
        ("WRF's default surface-drag arm is implemented: module_bl_ysu.F:404 "
         "passes ctopo and ctopo2 and bl_ysu.F90:1308 applies their drag. "
         "The selected kernel carries the paj, GET_PBLH and Beljaars vconv "
         "calculation; tests/test_ysu_wrf461_parity.py pins zero momentum ULP "
         "against that arm and retains the separate 182 ULP comparison to "
         "the historical ctopo-absent reference. This is column evidence, "
         "not qualification against a WRF forecast.")
        if "bl_ysu.F90:1315" in warning else warning
        for warning in pbl_options["ysu"]["warnings"]]
    from woof.ysu_generations import YSU_GENERATIONS, YSU_DEFAULT_GENERATION
    pbl_options["ysu"]["numerical_generations"] = {
        "selector": "ysu_version", "default": YSU_DEFAULT_GENERATION,
        "options": YSU_GENERATIONS,
    }
    pbl_options["ccpp-tke-edmf"] = {'label': 'CCPP scale-aware TKE moist EDMF', 'implemented': True, 'maturity': 'implemented-unverified', 'scientific_evidence': 'none', 'selectors': {'bl_pbl_physics': 901}, 'parameters': {}, 'asset_requirements': [], 'wrf_counterpart': {'exists': False, 'implementation_origin': 'upstream-port', 'reason': 'CCPP satmedmfvdifq is a GFS and RRFS scheme with no WRF selector counterpart. WRF selector 3 is the older hybrid EDMF.'}, 'constraints': {'required_settings': {'moist': True, 'bldt': 0.0}, 'required_settings_reasons': {'moist': 'moist buoyancy needs vapor and condensate', 'bldt': 'positive cadence would retain a TKE tendency absent from the extra-scalar restart stream'}, 'requires_components': {'surface_layer': ['revised-mm5', 'classic-mm5']}, 'requires_components_reasons': {'surface_layer': 'the CCPP stability closure divides by fm/fh, supplied by these similarity layers'}}, 'reachability': {'state': 'component-override'}, 'warnings': ['New non-WRF selector 901. The pinned CCPP column has 0 ULP GPU agreement on 64 columns over three steps. No observation-based forecast score is claimed.'], 'extensions': {'upstream_commit': '1d9b076503c27cd4cfa8b22a977a71e889cbb149', 'fpvs_manifest': 'woof/data/satmedmf/manifest.json', 'prognostic_scalars': {'tke_edmf': {'units': 'm2 s-2', 'initial_value': 0.2, 'transport': 'existing positive-definite scalar path', 'lateral_boundary': 'existing flow-dependent scalar boundary', 'restart': 'serialized'}}, 'vertical_order': 'bottom-to-top', 'humidity_binding': 'specific humidity from dry-air mixing ratio; finite increments converted back', 'outputs': ['temperature and water tendencies', 'momentum tendencies', 'TKE tendency', 'PBLH', 'KPBL', 'heat and momentum diffusivities'], 'surface_inputs': ['UST', 'HFX', 'QFX', 'FM', 'FH', 'BR', 'ZNT', 'vegetation fraction', '10 m winds'], 'bldt': {'supported_values': [0], 'meaning': 'run on every model step'}}}
    surface_options = registry["components"]["surface_layer"]["options"]
    pbl_options["ysu"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    pbl_options["mynn"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5", "mynn"]
    # Shin-Hong (bl_pbl_physics=11) requires isfc=1 exactly as YSU does,
    # through WRF's own SHINHONGSCHEME arm: phys/module_physics_init.F:
    # 3702-3704 fatals unless sf_sfclay_physics initialized isfc=1, which
    # only the revised and classic MM5 surface layers do.
    pbl_options["shinhong"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5"]
    # WHY, beside WHICH, for the three rows above: the evaluator prints
    # this after the list, so the plan door names the breakage the run
    # door names (woof.config / woof.physics_compat) instead of a bare
    # "requires surface_layer in [...]".
    _fm_fh_reason = (
        "{scheme} binds fm/fh, the full similarity denominators "
        "ln(z/z0)-psi, directly and divides by them (zol = br*fm^2/fh); "
        "only the revised and classic MM5 surface layers publish them, so "
        "any other surface layer leaves them at their allocated zeros and "
        "the PBL runs on finite, plausible, wrong values for the whole "
        "forecast")
    pbl_options["ysu"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": _fm_fh_reason.format(scheme="YSU")}
    pbl_options["shinhong"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": _fm_fh_reason.format(scheme="Shin-Hong")}
    pbl_options["mynn"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": (
            "the MYNN PBL takes its lower boundary from the friction "
            "velocity and the surface heat and moisture fluxes (ust/flt/flq) "
            "that the revised MM5, classic MM5 and MYNN surface layers "
            "publish; the Eta layer publishes MYJ's own exchange set "
            "instead and the off option writes none of them (WRF v4.6.1 "
            "admits the same three at phys/module_physics_init.F:3837-3839)")}
    # These implemented choices no longer have a single-value constraint.
    for name in ("bl_mynn_mixlength", "bl_mynn_mixscalars"):
        pbl_options["mynn"]["constraints"]["required_settings"].pop(name, None)
    # Both MYNN surface-layer generations are implemented.
    mynn_surface = surface_options["mynn"]["constraints"]
    mynn_surface.setdefault("required_settings", {}).pop(
        "mynn_sfclay_variant", None)
    mynn_surface.setdefault("admitted_setting_values", {})[
        "mynn_sfclay_variant"] = ["wrf_461", "gsl_wrf39"]
    mynn_surface.setdefault("admitted_setting_values_reasons", {})[
        "mynn_sfclay_variant"] = (
        "wrf_461 is WRF v4.6.1's MYNN surface layer and gsl_wrf39 the GSL "
        "WRF 3.9 fork's; the two solve different equations for z/L, so no "
        "other name can select either")
    pbl_options["mynn"]["parameters"]["scalar_pblmix"] = 0
    from woof.config import MYNN471_OPTION_VALUES
    mynn = pbl_options["mynn"]
    for name, values in MYNN471_OPTION_VALUES.items():
        mynn["constraints"]["required_settings"].pop(name, None)
        mynn["constraints"].setdefault("admitted_setting_values", {})[name] = list(values)
        # Widen the public knob declaration alongside the planner constraint.
        if name in registry["parameters"]:
            registry["parameters"][name]["enum"] = list(values)
    mynn["constraints"]["required_settings"].pop("tke_budget", None)
    mynn["constraints"].setdefault("admitted_setting_values", {})["tke_budget"] = [0,1]
    mynn["constraints"]["required_settings_reasons"] = {
        name: "Existing MYNN scheme constraint"
        for name in mynn["constraints"]["required_settings"]}
    mynn["extensions"]["wrf_471_qualification"] = json.loads(
        (MODEL / "woof/data/mynn471/qualification.json").read_text())
    mynn["warnings"].append("WRF 4.7.1 sub-options use the source-pinned wrf_471 solver; default forecasts retain wrf_461. All 14 qualified changed settings meet the 1.2x stage-speed bar against interleaved ordinary MYNN defaults on the same RTX 4090 and commit; retained strict single-kernel and graph replays differ in zero output words. Cold rmol follows the retained fresh-process native zero frame; whole-native ABI-context equality is not claimed. Qualification: woof/data/mynn471/qualification.json.")
    # Shin-Hong is now selected by a registered template
    # (thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1, below), which is
    # the easiest path to it, so its recomputed reachability is
    # "template".  It remains a legal per-domain override on the tree
    # route as well (allowed_component_options below); the state names
    # the easiest path, not the only one.  It stays
    # "implemented-unverified": the template that selects it is a
    # composition candidate, not a matched forecast trajectory for this
    # closure.
    pbl_options["shinhong"]["reachability"] = {"state": "template"}
    # ysu, mynn and shinhong declare NO moist requirement, and that is
    # a decision rather than a gap.  Audit R-025 proposed
    # required_settings moist=true on all five closures, reading the
    # rule off the state allocation; as a rule over the slot it is
    # false.  A dry state hands every seam the persistent zero moisture
    # planes, the moisture rows solve to exactly zero tendencies and
    # nothing consumes them, and a dry YSU plan is admitted by the
    # loader (tests/test_config.py::
    # test_km_opt4_admits_pbl_off_vertical_diffusion).  A
    # required_settings row is refused at plan review, so a row on
    # those three would refuse runs that work.
    # myj and sase keep theirs, each for a reason that is the row's own
    # text: WRF's PBL driver fatals a MYJ column without qv_curr/qc_curr
    # (phys/module_pbl_driver.F:1441-1443, pinned by
    # tests/test_myj_port.py::
    # test_a_dry_myj_run_is_refused_the_way_wrf_refuses_it), and SASE
    # forms its stability from the saturated Brunt-Vaisala frequency and
    # mixes condensate rows a dry column cannot give it.
    # SASE is not in the WRF v4.6.1 table above -- WRF has no such scheme,
    # which is why it carries an out-of-namespace selector.  Its
    # surface-layer constraint is therefore NOT a transcription of WRF's
    # 12-cell matrix but a statement of what the closure reads: any
    # surface-layer scheme that produces a friction velocity and the
    # heat/moisture fluxes serves, and only "off" does not.
    #
    # "mynn" is NOT among them, and the reason is the other option's
    # constraint rather than anything SASE needs.  The MYNN surface layer
    # declares requires_components pbl = [off, mynn], transcribed from
    # WRF's 16-cell matrix (phys/module_physics_init.F:3699-3704,
    # 3837-3839) and pinned by tests/test_physics_registry.py::
    # test_mynn_component_dependencies_are_the_wrf_v461_cells.  SASE is
    # neither of those, so the pairing was refused by the registry while
    # woof.config.validate_sase_config admitted it -- a disagreement of
    # 512 combinations that only became visible once the turbulence
    # component gained its km_opt=0 option and SASE plans could resolve
    # at all.  Listing what the closure would ACCEPT while the other half
    # refuses to run is not an admission, so the intersection is what is
    # declared here, and validate_sase_config now refuses the same pair.
    pbl_options["sase"]["constraints"]["requires_components"][
        "surface_layer"
    ] = ["revised-mm5", "classic-mm5", "mynn", "pleim-xiu"]
    pbl_options["sase"]["constraints"]["requires_components_reasons"] = {
        "surface_layer": (
            "SASE's lower boundary condition is the surface layer's "
            "friction velocity, heat and moisture fluxes and "
            "gust-corrected wind speed (ust/hfx/qfx/wspd); the off option "
            "writes none of them"),
    }
    # "mynn" JOINED THAT LIST.  It was excluded by intersecting two
    # tables rather than by a physical reason: the MYNN surface layer's
    # own row transcribes WRF's isfc matrix, which has no cell for SASE
    # at all -- bl_pbl_physics=900 is outside the transcription's PBL
    # axis, and asking it for a verdict RAISES.  What SASE actually reads
    # is ust/hfx/qfx/wspd, and MYNN_SURFACE_OUTPUTS publishes all four,
    # allocated on sf_sfclay_physics=5 alone, independent of the PBL
    # selector.  The pairing is unmeasured, and unmeasured is maturity:
    # this registry's own policy is that maturity warns and never blocks,
    # so it carries the warning below and no constraint.
    _sase_mynn_warning = (
        "The MYNN surface layer is admitted under SASE on the field "
        "contract alone: it publishes the friction velocity, the heat and "
        "moisture fluxes and the gust-corrected wind speed the closure "
        "reads, and nothing else in the closure is surface-layer "
        "specific. No trajectory evidence covers this pairing -- the "
        "measured SASE runs are all revised/classic MM5 -- so it is "
        "experimentable, not evidenced.")
    # This builder is run against the registry it last wrote, so an append
    # has to be conditional or it doubles the warning on the second run.
    if _sase_mynn_warning not in pbl_options["sase"]["warnings"]:
        pbl_options["sase"]["warnings"].append(_sase_mynn_warning)
    # bldt IS NOT PINNED TO 0.  The required_settings row that carried it
    # was refused at plan review and at run start while
    # validate_sase_config admitted any cadence, and the driver is the
    # tiebreaker: it runs SASE at bldt_seconds, holds and recouples the
    # tendencies across skipped calls, and retains the flux diagnostics
    # between calls "at a positive PBL cadence". tests/test_sase.py and
    # tests/test_sase_cadence*.py exercise 0.1 s and 5.0 s cadences.
    pbl_options["sase"]["constraints"]["required_settings"].pop("bldt", None)
    # Stated in this option's own fourth warning as prose since it was
    # written ("it requires moist=true"), and enforced by
    # validate_sase_config, but never declared machine-readably -- so the
    # registry called 72 dry SASE combinations launchable that the loader
    # then refused.  The closure mixes vapour, cloud water and cloud ice
    # beside theta and forms its stability from the SATURATED
    # Brunt-Vaisala frequency; a dry state has nothing for it to integrate.
    pbl_options["sase"]["constraints"]["required_settings"]["moist"] = True
    pbl_options["sase"]["reachability"] = {"state": "component-override"}
    # ---- MYJ (bl_pbl_physics=2) and its Eta similarity surface layer -----
    # The one PAIR in this registry.  WRF v4.6.1 fatals a MYJ PBL whose
    # surface layer did not set isfc=2 (phys/module_physics_init.F:
    # 3770-3772), and among the surface layers woof ports only MYJSFCSCHEME
    # does (:3169).  Both halves therefore declare the other in
    # requires_components, and woof.config.validate_myj_pairing refuses the
    # mismatch in BOTH directions at load.  No template selects either --
    # the shinhong/sase posture: a user asks for the pair explicitly or does
    # not get it.
    _MYJ_PAIR_WARNING = (
        "MYJ requires the Eta similarity surface layer. EEPS also admits Eta "
        "with its independently allocated native-zero TKE column. WRF's "
        "MYJ law is one-directional -- "
        "phys/module_physics_init.F:3770-3772 fatals bl_pbl_physics=2 "
        "unless the surface layer set isfc=2, which only "
        "sf_sfclay_physics=2 does among the layers woof ports (:3169) -- "
        "and WOOF adds the reverse refusal for a stated reason: the Eta "
        "layer publishes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and publishes NO MOL, "
        "ZOL, PSIM/PSIH, REGIME, GZ1OZ0 or WSPD "
        "(phys/module_sf_myjsfc.F:361-1056), while YSU, MYNN, Shin-Hong and "
        "SASE each read at least one of those. Pairing them would hand a "
        "PBL scheme a zero where WRF hands it a similarity function: "
        "finite, plausible and wrong. woof.config.validate_myj_pairing is "
        "that refusal.")
    _MYJ_EVIDENCE_WARNING = (
        "MYJ is implemented-unverified and the evidence string says exactly "
        "what was run: the float32 CPU authority "
        "(woof/verify/myj_ref.py) is a line-by-line transcription of the "
        "byte-frozen phys/module_bl_myjpbl.F and phys/module_sf_myjsfc.F, "
        "and tests/test_myj_port.py drives the SHIPPED seams "
        "(initialize_physics + PhysicsDriver.compute) plus the CUDA "
        "translation units, asserting finiteness, physical bounds, the "
        "conservation the scheme has and CPU-vs-CUDA agreement, with "
        "mutation controls that STUB THE PORTED ROUTINES THEMSELVES and "
        "record which bars each stub turns red (the two bars nothing "
        "breaks -- _vdifq and the similarity-table lookup -- are declared "
        "in the same table rather than left silently green). TKE_MYJ "
        "cold-starts at epsq2=0.2, which is what MYJPBLINIT writes "
        "(phys/module_bl_myjpbl.F:1725, share/module_model_constants.F:92) "
        "and not zero: the seed decides MIXLEN's LPBL scan on step one and "
        "a zero column is a 0/0 in EL0 that WRF never reaches. For the "
        "Eta surface layer (MYJSFC) NO ORACLE COMPARISON AGAINST THE WRF "
        "FORTRAN HAS BEEN RUN: there is no gfortran replay of "
        "module_sf_myjsfc.F and no ULP table, so nothing here claims bit "
        "agreement with WRF for it. The PBL half has one: the MYJ PBL "
        "kernel is bit-identical to WRF v4.6.1's MYJPBL "
        "(tools/myjpbl_wrf461_oracle, tests/test_myjpbl_wrf461_parity.py; "
        "see the MYJ PBL row).")
    _MYJ_PBL_EVIDENCE_WARNING = (
        "MYJ PBL is implemented-unverified because no matched "
        "WOOF-versus-WRF forecast trajectory exists for it, but its CUDA "
        "kernel (woof/core/kernels/myjpbl.cu) is BIT-IDENTICAL to WRF "
        "v4.6.1's own MYJPBL. tools/myjpbl_wrf461_oracle compiles the "
        "byte-unmodified phys/module_bl_myjpbl.F at gfortran 13.3 -O0 "
        "(WRF's own -O2 -ftree-vectorize -funroll-loops build writes the "
        "same bytes) and drives it over 176 columns -- convective, stable "
        "night, neutral, ocean, snow, sea ice, cloud liquid and ice, "
        "terrain from -25 to 4500 m, the QCI-present and QCI-absent arms, "
        "nz 4, 35, 50 and 128, DT 10 to 90 s, and edge columns (CT /= 0, "
        "CHKLOWQ = 0, dew, USTAR = 0, calm, exactly neutral, TKE floor, "
        "LPBL = 1, subnormal condensate) -- for three consecutive calls; "
        "the fixture executes every line of MYJPBL's call tree except "
        "PRODQ2's ELOQN <= EPS1 arm (:1145-1146). Under the strict build "
        "(GPUWM_WRF_EXACT=1) every word of every output (TKE_MYJ, EL_MYJ, "
        "EXCH_H, the six tendencies, PBLH, MIXHT, KPBL, THZ0, QZ0, QSFC, "
        "CT) is identical, replayed and free-running; under default "
        "arithmetic (CuPy's -ftz=true) only RQCBLTEN/RQIBLTEN words differ, "
        "each by less than FLT_MIN (at most 1.2e-38), where diffusion "
        "leaves subnormal condensate (tests/test_myjpbl_wrf461_parity.py). "
        "The kernel seeds "
        "ZINT(KTE+1)=HT as WRF does (phys/module_bl_myjpbl.F:312), calls "
        "WOOF's own float32 gfk_exp_fma/gfk_pow (matched to the oracle "
        "host's C library over all 2**32 inputs by "
        "tools/myjpbl_wrf461_oracle/libm_sweep.py), compiles without FMA "
        "contraction and carries gfortran's folded constant words. That is "
        "conformance evidence, not scientific validation.")
    _MYJ_QUIRK_WARNING = (
        "WRF quirks transcribed rather than repaired, each cited in the "
        "port. CT (the countergradient correction) is identically zero in "
        "ARW: MYJSFC zeroes it every call "
        "(phys/module_sf_myjsfc.F:206-211) and SFCDIF's countergradient "
        "block is commented out (phys/module_sf_myjsfc.F:816-825, CT=0.), "
        "so MIXLEN's DTH+CT fix (phys/module_bl_myjpbl.F:845-850) and "
        "VDIFH's RKCT term add exactly zero -- both seams are still built "
        "so a future WRF that restores them lands in the right place. CZIL "
        "is hard-coded to 0.1 because the Chen-Zhang block is commented "
        "out (phys/module_sf_myjsfc.F:689-697), which is why "
        "IVGTYP/ISURBAN/IZ0TLND are dead arguments and why woof.config "
        "refuses isftcflx/iz0tlnd with this surface layer. The "
        "Zilitinkevich thermal-roughness fix reads Z0BASE, not the working "
        "ZNT (phys/module_sf_myjsfc.F:733). DIFCOF's inversion block is "
        "commented out (phys/module_bl_myjpbl.F:1275-1322), so its T "
        "argument is dead and the port does not take it.")
    _MYJ_DIVERGENCE_WARNING = (
        "DELIBERATE DIVERGENCE in the Eta surface layer, woof goes its "
        "own way (float32 precision): MYJSFC's interface-height column is "
        "GROUND-relative. WRF seeds it with the terrain height, "
        "ZINT(I,KTE+1,J)=HT(I,J) (phys/module_sf_myjsfc.F:162), so every "
        "height it carries is above sea level; woof's surface layer seeds "
        "zero. (The MYJ PBL kernel no longer does: it seeds HT as "
        "phys/module_bl_myjpbl.F:312 does, because the float32 words depend "
        "on it -- up to 51 ULP on PBLH and 231 on EXCH_H over terrain -- "
        "and it is graded bit-identical to WRF.) "
        "Every consumer reads only DIFFERENCES of those "
        "heights, so HT cancels EXACTLY in real arithmetic and only "
        "APPROXIMATELY in float32, where differencing numbers offset by "
        "1-3 km drops bits the ground-relative column keeps. woof's "
        "column is the more accurate one, which is why it ships. The "
        "cancellation is MEASURED rather than assumed: over five columns "
        "(four stretched so no dz and no interface height is an exactly "
        "representable float32) crossed with terrain at 1523.7, 2987.3 "
        "and 4411.1 m, land and water, KPBL is identical in every case, "
        "non-tendency fields move at most 69 ULP -- attained on lh over "
        "water, where the move is 1.645e-05 W m-2 -- and at most "
        "5.75e-06 relative, attained on qfx rather than on lh, and "
        "tendency rows move at most 2.05 "
        "quanta of the source field's ULP over dt "
        "(tests/test_myj_port.py::"
        "test_the_dropped_terrain_height_cancels_in_float32). It will be "
        "a real term in the ULP table when the surface-layer oracle "
        "campaign runs.")
    _MYJ_SCOPE_WARNING = (
        "OUT OF SCOPE, refused rather than approximated. WRF sends the MYJ "
        "PBL through myjurb (phys/module_bl_myjurb.F:130-771) when sf_urban_physics "
        "is 2 or 3, the BEP/BEM multi-layer urban canopies "
        "(phys/module_physics_init.F:3775-3781). woof does the same: "
        "PhysicsDriver._run_myj_pbl routes MYJ to woof.core.myjurb under "
        "the bep and bep-bem urban options. "
        "With sf_surface_physics=0 the land branch stops evolving surface "
        "humidity rather than failing: MYJPBL rebuilds QSFC over land from "
        "ELFLX and CHKLOWQ, both of which an LSM writes, and its "
        "IF(QFC1>0.) guard leaves QSFC untouched when they are zero "
        "(phys/module_bl_myjpbl.F:547-557). That is WRF's own behaviour and "
        "it is admitted, not refused -- but a land run without an LSM is a "
        "land run whose surface moisture is frozen at its initial value. "
        "The MYJPBL species stack is WRF's own three or four rows (theta, "
        "vapour, cloud water and cloud ice when the moist set carries it): "
        "the QCS/QCR/QCG rows the source declares are commented out "
        "upstream (:274-285) and are not ported. The kernel holds the "
        "column in per-thread local memory like YSU's, so it carries the "
        "same 128-level ceiling and refuses a deeper column rather than "
        "truncating it.")
    pbl_options["myj"] = {
        "asset_requirements": [],
        "constraints": {
            # NO required_settings.  ``moist: True`` stood here and was
            # retired with the loader's arm: a dry woof state is not an
            # absent state (every PBL seam gets the persistent zero
            # qv/qc planes), YSU, MYNN and Shin-Hong are all admitted dry
            # through those same planes, and MYJ's dry limit is defined
            # rather than 0/0.  If dry admission is ever denied it is a
            # property of the PBL SLOT and belongs as one column on every
            # PBL option row, not on this one.
            "required_settings": {},
            "requires_components_reasons": {
                "surface_layer": (
                    "the MYJ PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the Eta similarity surface layer produces them; "
                    "WRF v4.6.1 fatals the same pairing "
                    "(phys/module_physics_init.F:3770-3772)")},
            "requires_components": {
                # The surface layer is the ONLY component MYJ constrains.
                # An earlier draft also listed the land-surface options,
                # and that was a gate this port invented: WRF places no
                # LSM restriction on MYJ, and with sf_surface_physics=0
                # the code path is DEFINED rather than broken -- LH and
                # CHKLOWQ are zero, so QFC1 is zero and MYJPBL's
                # IF(QFC1>0.) guard simply leaves QSFC where it was
                # (module_bl_myjpbl.F:547-557).  Degraded, not impossible,
                # and the consequence belongs in a warning rather than in
                # a refusal; the pairing the port drove end to end is
                # named in the evidence warning instead.
                "surface_layer": ["eta-similarity"],
            },
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the MYJ PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the Eta similarity surface layer produces them"),
                "classification": (
                    "WRF v4.6.1 law in the PBL direction "
                    "(phys/module_physics_init.F:3770-3772); WOOF structural "
                    "constraint in the surface-layer direction"),
                "wrf_source": (
                    "phys/module_physics_init.F:3169,3770-3772"),
            },
        },
        "implemented": True,
        "label": "MYJ PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 2},
        "warnings": [_MYJ_PBL_EVIDENCE_WARNING, _MYJ_PAIR_WARNING,
                     _MYJ_QUIRK_WARNING, _MYJ_SCOPE_WARNING],
    }
    # ---- UW moist turbulence (bl_pbl_physics=9) ---------------------------
    # WRF v4.7.1's CAMUWPBLSCHEME, ported with its CAM modules and graded
    # against tools/uwpbl_wrf471_oracle.  It reads UST/HFX/QFX from its
    # surface layer and nothing else of the surface layer's set, so the
    # fm/fh reason that ties YSU and Shin-Hong to the MM5 layers does not
    # reach it and the MYNN layer is admitted beside it; the Eta layer is
    # refused for its own reason (its PBLH scan reads TKE_MYJ).
    pbl_options["uw"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5", "mynn",
                                  "gfs", "pleim-xiu"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "the UW scheme takes its lower boundary from the "
                    "friction velocity and the surface heat and moisture "
                    "fluxes (UST/HFX/QFX), which the revised MM5, classic "
                    "MM5, MYNN, GFS and Pleim-Xiu surface layers publish; the Eta layer "
                    "publishes them too but its own PBLH scan reads the "
                    "TKE_MYJ column only the MYJ PBL advances, and the off "
                    "option writes none of them")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the Eta surface layer's PBLH scan is MYJSFC's TKE scan "
                    "over TKE_MYJ, which only the MYJ PBL allocates"),
                "classification": (
                    "WOOF structural constraint; WRF v4.7.1 places no "
                    "surface-layer law on CAMUWPBLSCHEME and fatals only "
                    "the BEP/BEM urban pairing"),
                "wrf_source": "phys/module_physics_init.F:3825-3832",
            },
        },
        "implemented": True,
        "label": "UW moist-turbulence PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 9},
        "warnings": list(_UWPBL_WARNINGS),
    }
    # The Eta surface layer carries its OWN evidence and has no divergence
    # warning: its WRF column oracle is graded word for word, and the
    # ground-relative height column _MYJ_DIVERGENCE_WARNING describes is
    # gone from it (the kernel seeds ZINT(KTE+1)=HT as WRF does).
    _MYJ_SFC_EVIDENCE_WARNING = (
        "The Eta similarity surface layer is implemented-unverified because "
        "no matched WOOF-versus-WRF forecast trajectory exists for it, but "
        "its CUDA kernel (woof/core/kernels/myjsfc.cu) and its host "
        "similarity tables are BIT-IDENTICAL to WRF v4.6.1's own MYJSFC and "
        "MYJSFCINIT. tools/myjsfc_wrf461_oracle compiles the byte-unmodified "
        "phys/module_sf_myjsfc.F at gfortran 13.3 -O0 (WRF's own -O2 "
        "-ftree-vectorize -funroll-loops build writes the same bytes; "
        "neither has an FMA or a libmvec call) and drives 224 columns -- "
        "convective and stable land, snow and sea ice, warm, cold and "
        "storm-force ocean, terrain to 4.4 km, calm air, cloud water, a "
        "TKE column that never drops below EPSQ2 and a dead one, a "
        "boundary layer above 1000 m, extreme instability and stability, "
        "a saturated cold column, a lowest layer under 4 m and the "
        "two-level minimum column -- through MYJSFCINIT and three "
        "consecutive MYJSFC calls (cold start at ITIMESTEP 1 and warm "
        "seeded state across the three sea viscous regimes); the fixture "
        "executes every line of MYJSFC and SFCDIF. Every word of all 35 "
        "INOUT and output fields is identical, replayed and free-running, "
        "under the strict build (GPUWM_WRF_EXACT=1) and under default "
        "arithmetic, and all 4 x 10,001 PSIM/PSIH table words match "
        "(tests/test_myjsfc_wrf461_parity.py). A further 8,192 columns "
        "with randomly drawn regimes, four calls each (32,768 "
        "column-calls, tools/myjsfc_wrf461_oracle/stress.sh), are "
        "identical too, and so is the float32 CPU authority "
        "(woof/verify/myj_ref.py np_myjsfc_column, graded on CPU). The "
        "kernel seeds "
        "ZINT(KTE+1)=HT as WRF does (phys/module_sf_myjsfc.F:165), calls "
        "WOOF's own float32 gfk_log/gfk_exp_fma/gfk_pow (graded against "
        "the oracle host's C library over all 2**32 inputs by "
        "tools/myjsfc_wrf461_oracle/libm_sweep.py: one input differs, 1 ULP "
        "of POW at x = 0.00825 with exponent 1/CAPA, which needs a surface "
        "pressure below 0.01 Pa) and compiles without FMA "
        "contraction; the host tables use WOOF's own float32 "
        "logf/atanf/expf. That is conformance evidence, not scientific "
        "validation.")
    # Seed the RCON identity before any component derives admitted MP lists.
    _misc_rcon_component(registry)
    _misc_udm_component(registry)
    # ---- GBM moist TKE (bl_pbl_physics=12) --------------------------------
    # WRF v4.6.1's GBMPBLSCHEME, graded word for word against
    # tools/gbmpbl_wrf461_oracle.  WRF's one law for it is the isfc = 1
    # surface-layer fatal (module_physics_init.F:3892-3893), which admits
    # the two MM5 layers only; woof.config.validate_gbmpbl_config is that
    # law.
    pbl_options["gbm"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5"],
                "microphysics": [key for key, option in registry["components"]["microphysics"]["options"].items()
                                 if option.get("implemented") and option.get("selectors", {}).get("mp_physics")
                                 in PBL_RQI_MICROPHYSICS]},
            "requires_components_reasons": {
                "microphysics": ("GBM requires F_QC and F_QI; absent cloud water "
                                 "or ice would discard RQCBLTEN or RQIBLTEN "
                                 "and their condensate budget "
                                 "(module_pbl_driver.F:1992-2018)"),
                "surface_layer": (
                    "the GBM scheme forms its friction velocity and surface "
                    "stress from the MM5 layers' GZ1OZ0 and PSIM "
                    "(module_bl_gbmpbl.F:539); WRF v4.6.1 fatals every "
                    "other surface layer (phys/module_physics_init.F:"
                    "3892-3893, isfc must be 1), and with the slot off "
                    "nothing writes them")},
        },
        "extensions": {
            "wrf_contract": {
                "source": "phys/module_bl_gbmpbl.F",
                "version": "4.6.1",
                "package_state": ["exch_tke", "el_pbl", "tke_pbl"],
                "prognostic_restart_state": ["tke_pbl"],
                "tendencies": ["RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN", "RQIBLTEN"],
                "outputs": ["EXCH_M", "EXCH_H", "EXCH_TKE", "EL_PBL", "PBLH", "KPBL", "UST", "REGIME"],
                "surface_inputs_read": ["PSFC", "GZ1OZ0", "PSIM", "XLAND", "HFX", "QFX", "TSK"],
                "surface_arguments_unused": ["ZNT", "ZOL", "HOL", "PSIH", "WSPD", "BR"],
                "preserved_surface_output": ["REGIME"],
                "radiation_input": "RTHRATEN",
                "cadence": "bldt",
                "required_species_flags": ["F_QC", "F_QI"],
                "constants_asset": "woof/data/gbmpbl/constants.cuh",
            },
            "arwen_pairing_requirement": {
                "reason": (
                    "the scheme's u* is k|V1|/(GZ1OZ0-PSIM), read from the "
                    "MM5 surface layers"),
                "classification": (
                    "WRF v4.6.1 law (phys/module_physics_init.F:3892-3893)"),
                "wrf_source": "phys/module_physics_init.F:3891-3893",
            },
        },
        "implemented": True,
        "label": "GBM moist TKE PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 12},
        "warnings": list(_GBMPBL_WARNINGS),
    }
    # ---- EEPS, TKE + dissipation (bl_pbl_physics=16) ---------------------
    # WRF v4.6.1's EEPSSCHEME, graded against tools/eeps_wrf461_oracle.  It
    # reads UST/RMOL/WSPD/HFX/QFX/QSFC from its surface layer and none of
    # the fm/fh pair, so the MM5 layers and the MYNN layer are all admitted
    # (WRF's own list, module_physics_init.F:3872, is isfc 1/2/5/91); the
    # Eta layer independently activates TKE_PBL through myjsfcscheme.
    pbl_options["eeps"] = {
        "asset_requirements": [],
        "constraints": {
            # EEPS's TKE and dissipation are WRF scalar-array species the
            # dycore advects, and a dry state runs no scalar transport
            # (woof.config.validate_eeps_config).
            "required_settings": {"moist": True},
            "required_settings_reasons": {
                "moist": (
                    "EEPS's TKE and dissipation rate are the advected "
                    "scalars PEK_ADV/PEP_ADV, and a dry state runs no "
                    "scalar transport, so the pair would never move")},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5", "mynn", "eta-similarity"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "EEPS takes its lower boundary from UST, RMOL, WSPD, "
                    "HFX, QFX and QSFC, which the revised MM5, classic MM5, "
                    "MYNN and Eta surface layers publish; WRF v4.6.1 fatals "
                    "every surface layer outside isfc 1/2/5/91 "
                    "(phys/module_physics_init.F:3872-3873), the off "
                    "option writes none of them and the column divides by "
                    "WSPD")},
        },
        "extensions": {
            "eta_tke_ownership": {
                "wrf_source": "Registry/Registry.EM_COMMON:3140",
                "meaning": "Eta independently allocates TKE_PBL. Under EEPS it remains the native allocation zero, separate from PEK_PBL/PEK_ADV.",
            },
            "advected_scalars": {
                "names": ["pek_adv", "pep_adv"],
                "wrf_source": "Registry/Registry.EM_COMMON:1053-1054,3185",
                "io": "i0rusdf=(bdy_interp:dt)",
            },
        },
        "implemented": True,
        "label": "EEPS PBL (TKE-dissipation)",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 16},
        "warnings": list(_EEPS_WARNINGS),
    }
    # ---- MRF (bl_pbl_physics=99) ----------------------------------------
    # WRF v4.6.1's MRFSCHEME, the Hong-Pan nonlocal PBL, graded word for
    # word against WRF's compiled module_bl_mrf.F.  WRF's own law ties it
    # to isfc=1 (phys/module_physics_init.F:3715-3717): the two MM5 layers.
    pbl_options["mrf"] = {
        "asset_requirements": [{
            "id": "wrf-v4.6.1-mrf-constants-v1",
            "kind": "packaged-table-set",
            "search_root": "woof/data/mrf",
            "redistributed_by_gpuwm": True,
            "regenerable": False,
            "assets": [{"filename": "constants.cuh", "bytes": 1439,
                        "sha256": "c15be2603047c7eb2c9831afc3cd5b1b83e7bd5e5cb212823bb6cee3e72ddd1e"}],
            "source": "WRF v4.6.1 module_model_constants.F and module_bl_mrf.F:305-311",
            "note": "Scalar float32 words, checked before CUDA source assembly.",
        }],
        "extensions": {
            "wrf_driver_contract": {
                "registry_package": "mrfscheme bl_pbl_physics==99",
                "source": "phys/module_pbl_driver.F:1352-1382",
                "tendencies": ["RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN", "RQIBLTEN"],
                "outputs": ["PBLH", "KPBL", "HOL"],
                "surface_layer_passthrough": ["ZOL", "REGIME"],
                "surface_layer_inputs": ["UST", "ZNT", "PSIM", "PSIH", "GZ1OZ0", "WSPD", "BR", "HFX", "QFX", "TSK"],
                "unused_inputs": ["ZNT", "TSK"],
                "cadence_parameter": "bldt",
                "initialization": "Cold tendencies are zero; restart preserves them, matching mrfinit.",
                "kpbl_index": "bottom-up nz+1-KPBL0; declared output-index divergence",
            },
        },
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "MRF reads the MM5 surface layer's raw PSIM/PSIH, "
                    "GZ1OZ0, WSPD and BR and divides by GZ1OZ0-PSIM and "
                    "GZ1OZ0-PSIH; WRF v4.6.1 fatals every other surface "
                    "layer with it (phys/module_physics_init.F:3715-3717, "
                    "'Use sf_sfclay_physics= 1 or 91 for this pbl "
                    "option'), and with no surface layer those fields stay "
                    "at their allocated zeros and the first division is by "
                    "zero")},
        },
        "implemented": True,
        "label": "MRF PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 99},
        "warnings": list(_MRF_WARNINGS),
    }

    pbl_options["acm2"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5", "pleim-xiu"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "ACM2 takes its surface forcing from the friction "
                    "velocity, the surface heat and moisture fluxes and the "
                    "first-level wind speed, and WRF v4.6.1 runs it only "
                    "beside surface layers that set isfc 1 or 7 "
                    "(phys/module_physics_init.F:3728-3730: the MM5 layers "
                    "or Pleim-Xiu); the MYNN and Eta layers are refused "
                    "there, the Eta layer additionally publishes no WSPD, "
                    "and the off option writes none of these fields")},
        },
        "extensions": {},
        "implemented": True,
        "label": "ACM2 PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 7},
        "warnings": list(_ACM2_WARNINGS),
    }

    pbl_options["gfs"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["gfs"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "the GFS scheme forms FM = GZ1OZ0 - PSIM and its "
                    "surface stress (karman*wspd/FM)**2 from the surface "
                    "layer's UST/PSIM/PSIH/GZ1OZ0/WSPD/BR. WRF requires "
                    "isfc=2; the GFS surface layer supplies these fields. "
                    "The Eta surface layer leaves this boundary unset.")},
        },
        "extensions": {},
        "implemented": True,
        "label": "GFS PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 3},
        "warnings": list(_GFS_PBL_WARNINGS),
    }


    pbl_options["keps"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": ["revised-mm5", "classic-mm5", "mynn"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "the K-epsilon scheme divides the surface buoyancy "
                    "flux by UST and forms z/L as br*psim*psim/psih "
                    "(module_bl_keps.F:660, :1074); the revised MM5, "
                    "classic MM5 and MYNN layers publish all of them, the "
                    "Eta layer publishes no PSIM/PSIH/BR, and with the "
                    "layer off they are the allocated zeros, a NaN in "
                    "every tendency on the first call")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the scheme reads PSIM/PSIH/BR/UST, which only the MM5 "
                    "and MYNN surface layers publish here"),
                "classification": (
                    "WOOF structural constraint; WRF v4.7.1's bl_init has "
                    "no KEPSSCHEME case and places no pairing law on it"),
                "wrf_source": "phys/module_physics_init.F:3697-3916",
            },
        },
        "implemented": True,
        "label": "K-epsilon PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 17},
        "warnings": list(_KEPS_WARNINGS),
    }



    pbl_options["boulac"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {
                "surface_layer": [
                    "revised-mm5", "classic-mm5", "eta-similarity"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "WRF v4.6.1 runs BouLac only over a surface layer that "
                    "sets isfc 1 or 2 (phys/module_physics_init.F:3814-3816): "
                    "the revised and classic MM5 layers and the Eta layer, "
                    "each of which publishes the friction velocity and the "
                    "surface heat and moisture fluxes (UST/HFX/QFX) that are "
                    "BouLac's whole lower boundary. The MYNN layer sets isfc "
                    "5, which WRF fatals, and with the surface layer off "
                    "nothing writes UST/HFX/QFX at all")},
        },
        "extensions": {
            "arwen_divergences": {
                "rural_drag_zero_over_zero": (
                    "WRF's rural drag -ust**2/dz/|V1| is 0/0 on a calm, "
                    "zero-stress lowest level and writes a NaN U/V column; "
                    "the port writes the drag's limit, -0."),
                "kpbl": (
                    "WRF's BouLac writes no KPBL and leaves the zero "
                    "allocation Grell-Freitas and the GSL drag index; the "
                    "port diagnoses KPBL from BouLac's PBLH"),

            },
        },
        "implemented": True,
        "label": "BouLac PBL",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 8},
        "warnings": list(_BOULAC_WARNINGS),
    }


    pbl_options["qnse"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {},
            "requires_components": {"surface_layer": ["qnse"]},
            "requires_components_reasons": {
                "surface_layer": (
                    "the QNSE PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the QNSE surface layer produces them for it; WRF "
                    "v4.6.1 fatals the same pairing "
                    "(phys/module_physics_init.F:3790-3791)")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the QNSE PBL's every implicit solve takes "
                    "AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary and "
                    "only the QNSE surface layer produces them for it"),
                "classification": (
                    "WRF v4.6.1 law in the PBL direction "
                    "(phys/module_physics_init.F:3790-3791)"),
                "wrf_source": "phys/module_physics_init.F:3171,3790-3791",
            },
        },
        "implemented": True,
        "label": "QNSE PBL (with the EDMF shallow mass flux)",
        "maturity": "implemented-unverified",
        "parameters": {"mfshconv": 1},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"bl_pbl_physics": 4},
        "warnings": list(_QNSE_WARNINGS),
    }

    from woof.core.myjsfc_tables import QNSE_PSI_TABLES_SHA256
    surface_options["ideal-scm"] = {
        "asset_requirements": [],
        "constraints": {
            "requires_components": {"pbl": ["off"], "land_surface": ["off"]},
            "requires_components_reasons": {
                "pbl": "WRF 4.7.1 module_physics_init.F:3140-3237 leaves isfc unassigned for selector 89 before PBL compatibility checks.",
                "land_surface": "The prescribed-flux surface leaves UST/CHS/CHS2/CQS2 unchanged; they have no surface producer for an active LSM."}},
        "extensions": {"wrf_source": "WRF 4.7.1 phys/module_sf_idealscmsfclay.F:121-123,239-252",
            "state": ["HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND"]},
        "implemented": True, "label": "Ideal SCM prescribed flux surface",
        "maturity": "implemented-unverified", "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none", "selectors": {"sf_sfclay_physics": 89},
        "warnings": ["Native prescribed-flux controls are scalar state read from the forcing stream, not namelist parameters. Cold starts without that stream use their native zero initialization, including zero prescribed skin temperature. U10/V10/TH2/T2/Q2 are unassigned by the native routine and retain their incoming storage words. Qualification is pending; no native exactness claim is made."],
    }
    surface_options["qnse"] = {
        "asset_requirements": [],
        "constraints": {"required_settings": {},
            "requires_components": {"pbl": ["qnse"]},
            "requires_components_reasons": {"pbl":
                "The surface layer scans TKE_PBL and publishes AKHS/AKMS instead of FM/FH; only the QNSE PBL consumes this boundary state."}},
        "extensions": {"table_manifest": {
            "builder": "woof.core.myjsfc_tables.build_qnse_psi_tables",
            "source": "WRF v4.6.1 QNSESFCINIT",
            "sha256": QNSE_PSI_TABLES_SHA256,
            "layout": "PSIM1, PSIH1, PSIM2, PSIH2 in little-endian float32 order",
            "bytes": 4 * 10001 * 4}}, "implemented": True,
        "label": "QNSE surface layer", "maturity": "implemented-unverified",
        "parameters": {}, "reachability": {"state": "component-override"},
        "scientific_evidence": "none", "selectors": {"sf_sfclay_physics": 4},
        "warnings": ["Strict real-native replay matches all 1120 surface output words with zero differences (docs/public/receipts/qnse-native-module-replay.json). Production uses CUDA fast intrinsics and frozen QNSESFCINIT assets (phys/module_sf_qnsesfc.F:1125-1279) read through Rust NetCDF. The paired 2026-10-16 09-12Z real forecast, native comparison and station evidence are in docs/public/receipts/qnse-real-forecast.json; one case does not establish general skill."],
    }
    surface_options["eta-similarity"] = {
        "asset_requirements": [],
        "constraints": {
            "requires_components": {"pbl": ["myj", "eeps", "boulac"]},
            "requires_components_reasons": {
                "pbl": (
                    "the Eta surface layer publishes no fm/fh, the full "
                    "similarity denominators ln(z/z0)-psi that YSU and "
                    "Shin-Hong bind directly and divide by, so under it "
                    "they would divide by an allocated zero; and its PBLH "
                    "scan reads its registered TKE column. EEPS preserves the native "
                    "allocation zero; MYJ and BouLac evolve their registered columns with their own closures")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the Eta surface layer's published set is not the MM5 "
                    "layers' published set. YSU and Shin-Hong need FM/FH, "
                    "which it does not publish. EEPS consumes its actual "
                    "flux outputs and preserves the separate Eta TKE column"),
                "classification": (
                    "WOOF structural constraint; WRF v4.6.1 admits "
                    "sf_sfclay_physics=2 with PBL schemes woof does not "
                    "port (phys/module_physics_init.F:3742, phys/module_physics_init.F:3756)"),
                "wrf_source": "phys/module_sf_myjsfc.F:361-1056",
            },
        },
        "implemented": True,
        "label": "Eta similarity surface layer",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"sf_sfclay_physics": 2},
        "warnings": [_MYJ_SFC_EVIDENCE_WARNING, _MYJ_PAIR_WARNING,
                     _MYJ_QUIRK_WARNING, _MYJ_SCOPE_WARNING],
    }
    # ---- GFS surface layer (sf_sfclay_physics=3) ------------------------
    # WRF v4.6.1 phys/module_sf_gfs.F (SF_GFS over PROGTM), ported in
    # binary64 and graded word for word against tools/sfgfs_wrf461_oracle.
    # It publishes the shared exchange set and no fm/fh/MOL/ZOL, so the
    # MM5-bound PBLs (YSU, Shin-Hong) cannot read it, and MYNN/SASE read
    # MYNN-layer fields it does not write; the UW PBL reads UST/HFX/QFX only.
    surface_options["pleim-xiu"] = {
        "asset_requirements": [],
        "constraints": {
            "requires_components": {"pbl": ["off", "uw", "sase", "acm2"]},
            "requires_components_reasons": {
                "pbl": (
                    "the Pleim-Xiu layer publishes no fm/fh, the full "
                    "similarity denominators ln(z/z0)-psi that YSU and "
                    "Shin-Hong bind directly and divide by, so under it "
                    "they would divide by an allocated zero; WRF v4.6.1 "
                    "fatals YSU, Shin-Hong and MYNN with it "
                    "(phys/module_physics_init.F:3699-3704, 3837-3839: "
                    "PXSFCSCHEME sets isfc=7) and MYJ needs the Eta layer. "
                    "ACM2 is its native isfc=7 partner; UW reads UST/HFX/QFX "
                    "and SASE UST/HFX/QFX/WSPD, all of which it publishes")},
        },
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "WRF admits sf_sfclay_physics=7 with PBL-off, ACM2 "
                    "(bl_pbl_physics=7) and CAMUW; woof ports all three, "
                    "and SASE is WOOF's own "
                    "closure reading only UST/HFX/QFX/WSPD"),
                "classification": (
                    "WRF's own isfc law (phys/module_physics_init.F:"
                    "3153-3155, 3699-3704, 3728-3730, 3837-3839) on the "
                    "ported PBL set"),
                "wrf_source": "phys/module_sf_pxsfclay.F:239-529",
            },
        },
        "implemented": True,
        "label": "Pleim-Xiu surface layer",
        "maturity": "supported",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"sf_sfclay_physics": 7},
        "verification_scope": "native-module-oracle",
        "verification_scope_note": (
            "2026-11-01: committed real WRF inputs replay 43,200 float32 "
            "output words bitwise. Constructed fixtures cover both ice "
            "thresholds. Both nest feedback modes, restart continuation, "
            "ensembles and RUC coupling have device checks. One six-hour "
            "real forecast completed through woof go. This label does "
            "not claim whole-trajectory bitwise equality or general "
            "observation skill. See docs/physics/pxsfclay.md."),
        "warnings": list(_PX_SFCLAY_WARNINGS),
    }

    surface_options["gfs"] = {
        "asset_requirements": [],
        "constraints": {
            "requires_components": {"pbl": ["off", "uw", "gfs"]},
            "requires_components_reasons": {
                "pbl": (
                    "the GFS surface layer publishes UST, HFX, QFX, "
                    "PSIM/PSIH, GZ1OZ0, BR, WSPD and the exchange "
                    "coefficients, and no fm/fh, MOL, ZOL or REGIME "
                    "(phys/module_sf_gfs.F:9-18): YSU and Shin-Hong divide "
                    "by fm/fh (and WRF fatals them on isfc=2), the MYJ PBL "
                    "needs AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 that only the Eta "
                    "layer writes, and MYNN and SASE read MYNN-layer fields "
                    "this layer leaves at their allocated values, so each "
                    "would run on zeros -- finite, plausible and wrong. The "
                    "GFS PBL reads this published stability set, the UW PBL reads UST/HFX/QFX only, and PBL-off reads "
                    "nothing")},
        },
        "consumers": {
            "restart_algorithm_identity": (
                "gfs-surface-layer-wrf-v4.6.1-v2-fast-default-xland-sea-tsk-sea")},
        "extensions": {
            "arwen_pairing_requirement": {
                "reason": (
                    "the GFS layer's published set is the exchange set plus "
                    "PSIM/PSIH/GZ1OZ0/BR/WSPD; a PBL that binds fm/fh, MOL, "
                    "ZOL or the Eta layer's AKHS family would read zeros"),
                "classification": (
                    "WOOF structural constraint; WRF v4.6.1 admits "
                    "sf_sfclay_physics=3 (isfc=2) with the GFS, MYJ, MYNN, "
                    "BouLac and EEPS PBLs (phys/module_physics_init.F:"
                    "3742,3771,3815,3838,3872)"),
                "wrf_source": "phys/module_sf_gfs.F:9-282",
            },
        },
        "implemented": True,
        "label": "GFS surface layer",
        "maturity": "implemented-unverified",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"sf_sfclay_physics": 3},
        "warnings": [
            "The strict CUDA module replay reproduced all 10,240 output "
            "words captured from two calls on eight native WRF ranks in "
            "a real 2026-10-16 09Z forecast, with zero differing words. "
            "GPUWM_WRF_EXACT=1 preserves native scalar rounding and libm; "
            "default arithmetic permits contraction and device libm. "
            "tools/sfgfs_wrf461_oracle additionally covers land, sea, "
            "stability branches and fractional ice. Module conformance "
            "alone is not forecast validation.",
            "DECLARED DIVERGENCES, two WRF defects repaired in the "
            "FRACTIONAL_SEAICE wrapper: (1) WRF builds XLAND_SEA=2 for the "
            "open-water component of a sea-ice cell and passes XLAND to the "
            "open-water call (module_surface_driver.F:5751 vs :5772), so "
            "that component runs as land on a cell real.exe marked XLAND=1; "
            "woof passes XLAND_SEA, as every other wrapper in the file "
            "does. (2) WRF keeps the open-water TSK_SEA local to the GFS "
            "wrapper (:5670), so the RUC reblend reads the driver's "
            "automatic TSK_SEA (:1370, :3561) unset; woof publishes it. "
            "Outside sea-ice cells both are inert.",
            "Not ported because WRF never runs it: PROGTM's soil model "
            "(GOTO 1111/2222/5555, module_sf_gfs.F:489-1432), PROGT2, "
            "KTSOIL/FUNCDF/FUNCKT/THSAT/TWLT and module_progtm's tables. "
            "No WRF v4.6.1 path executes them, so they have no output to "
            "grade.",
        ],
    }
    # Grell-Freitas (cu_physics=3), the first cumulus option admitted
    # since KF and the first scale-aware one: sig = (1-frh)^2 is the
    # scheme's own dx taper, so per-domain admission carries no grid gate.
    # No template selects it -- the shinhong/sase posture: a user asks for
    # it explicitly or does not get it.
    _multiscale_column_option(registry)
    cumulus_options = registry["components"]["cumulus"]["options"]
    # Rebuild this option from its source declaration, not an old generated seed.
    cumulus_options["grell-devenyi"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            "required_settings_reasons": {
                "moist": "the cumulus schemes are moist convective schemes; woof/core/physics.py initialize_physics refuses a cumulus scheme on a dry DomainState, whose qv is None"},
            "requires_components": {},
        },
        "extensions": {
            "wrf_cudt": "Grell-Devenyi runs at the namelist's cudt as WRF does: tendencies and PRATEC hold between calls and RAINC accumulates PRATEC*DT every step",
            "source": "module_physics_addtendc.F:2141",
            "native_source": {
                "release": "WRF v4.7.1", "path": "phys/module_cu_gd.F",
                "sha256": "14948e3b6e82ba2d1529c2b31ff89c2e40e1596018d01898123bc2baa8eb81df",
                "license": "public-domain",
                "url": "https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_cu_gd.F",
            },
        },
        "implemented": True, "label": "Grell-Devenyi",
        "maturity": "implemented-unverified",
        # WRF check_a_mundo forces GD's convective transport diagnostics on.
        # The offered profile must carry the same value its namelist exports.
        "parameters": {"cudt_minutes": 0.0, "cu_diag": 1},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none", "selectors": {"cu_physics": 93},
        "warnings": ["The native module and CUDA kernel are covered by column conformance checks. Whole-forecast skill against observations is not established."],
    }
    cumulus_options["grell-freitas"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            # config.py enforces the same law: the trigger's excesses and
            # the shallow arm read KPBL and the PBL-maintained surface
            # fluxes, so a PBL scheme must be active.
            # "myj" joined this list with the MYJ port, and it is a
            # STRUCTURAL statement rather than an evidence one -- the
            # extension below says so.  MYJ writes KPBL
            # (module_bl_myjpbl.F:421) and maintains the surface fluxes
            # through its Eta surface layer, which is exactly what the
            # adapter reads; leaving it out would have invented a
            # prohibition WRF does not make and would have put the
            # registry and validate_run_config into disagreement over 816
            # combinations.
            "requires_components": {
                "pbl": ["ysu", "mynn", "shinhong", "sase", "myj", "acm2", "uw", "gfs", "qnse", "boulac", "gbm", "mrf", "ccpp-tke-edmf", "eeps", "temf"]},
            "requires_components_reasons": {
                "pbl": (
                    "WOOF's Grell-Freitas adapter reads KPBL and the "
                    "PBL-maintained surface fluxes for the trigger's "
                    "temperature and moisture excesses and for the shallow "
                    "arm; with the PBL off nothing writes them, so the "
                    "scheme has no boundary-layer top to hand to it")},
            "required_settings_reasons": {
                "moist": _CUMULUS_MOIST_REASON},
        },
        "extensions": {
            "arwen_pbl_structural_requirement": {
                "reason": (
                    "WOOF's GF adapter reads KPBL and the PBL-maintained "
                    "surface fluxes for the trigger's excesses and the "
                    "shallow arm; with bl_pbl_physics=0 nothing writes "
                    "them (WRF reads KPBL=0 there and indexes below the "
                    "column base, which WOOF refuses rather than "
                    "reproduces)"),
                "classification": (
                    "WOOF structural constraint; WRF v4.6.1 does not "
                    "prohibit cu_physics=3 with bl_pbl_physics=0"),
            },
            "shallow_closure_top_bound": {
                "default": "min(kpbl,ktf-1)",
                "classification": "declared divergence from WRF v4.6.1",
                "reason": "The pressure difference reads po_cup(k+1). A top-level PBL must not read above the column.",
            },
            "vertical_velocity_convention": {
                "default": "lower-face",
                "strict": "lower-face",
                "classification": "declared change from the previous adapter convention",
                "reason": "GFDRV expects lower-face W in its closure. Averaging adjacent faces changes the forcing and rain.",
            },
            "strict_arithmetic": {
                "selector": "GPUWM_WRF_EXACT=1",
                "normalization": "stock-scalar-operation-sequence",
                "publication": "four-cell-nonperiodic-interior",
                "cold_condensate_without_ice": "retain-stock-split-and-ignore-ice-slot",
                "default_unchanged": True,
            },
        },
        "implemented": True,
        "label": "Grell-Freitas",
        "maturity": "implemented-unverified",
        "parameters": {
            "cudt_minutes": 0.0, "clos_choice": 0, "ishallow": 0},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 3},
        "warnings": [
            "Grell-Freitas's distance from WRF v4.6.1 is MEASURED on both "
            "halves of the port and on the whole driver, not a scheme "
            "fragment. tools/gf_wrf461_oracle drives the byte-frozen "
            "module_cu_gf_wrfdrv.F/module_cu_gf_deep.F/module_cu_gf_sh.F "
            "at gfortran -O0 over two column sets: 18 branch-targeted "
            "cases x 6 grid spacings x 2 ishallow arms (216 columns), and "
            "280 realistic regime soundings (tropical, continental severe, "
            "nocturnal, winter snow, lake effect, arctic, high terrain, "
            "shallow cumulus, desert; climatological and active states, a "
            "clos_choice 1..16 sweep and 15 edge cases). The float32 CPU "
            "authority (woof/verify/gf_driver.py) reproduces GFDRV word "
            "for word on all 216 columns, and the CUDA translation unit "
            "(woof/core/kernels/gf.cu) does so on all 216 with the fzu "
            "normalisation PINNED from the capture and on all 280 with "
            "nothing pinned, against WRF linked with the correctly rounded "
            "gamma WOOF ships; strict and default arithmetic both give 0 "
            "differing words. WOOF's own float32 logf/expf/powf give WRF's "
            "words throughout, and gamma is graded against a 113-bit "
            "oracle because it is a deliberate divergence since 2.6.6 -- "
            "see the next warning (tests/test_gf_deep_cuda.py, "
            "tests/test_gf_shallow_cuda.py, tests/test_gf_gfdrv_cuda.py, "
            "tests/test_gf_regime_column_oracle_gpu.py, "
            "tests/test_gf_gamma_correctly_rounded.py). That is "
            "conformance evidence, not scientific validation: no "
            "woof/WRF forecast trajectory comparison exists for this "
            "scheme, which is why it is 'implemented-unverified' and not "
            "'supported'.",
            "DELIBERATE DIVERGENCE, owner ruling (no inherited WRF bugs) "
            "-- GAMMA. Read docs/gf_gamma_known_delta.md before quoting "
            "any GF parity number. gfortran binds WRF's F2008 gamma() to "
            "glibc's tgammaf, which is NOT correctly rounded: MEASURED "
            "against a 113-bit oracle over all 59,768,833 float32 of "
            "[0.25, 36], it is wrong on 23,575,230 (39.4440 per cent), "
            "worst 6 ULP; tgammaf(4.0f) returns 6.00000048, not 6. "
            "Through 2.6.5 WOOF's earlier gamma returned those same "
            "words; it is replaced, and like its replacement it was this "
            "project's own work under the project's licence. "
            "gfk_tgamma is now WOOF's own correctly rounded "
            "gamma (0 of 59,768,833 arguments wrong) and the SHIPPED "
            "forecast path computes it, so fzu is no longer WRF's word: "
            "it changes on 68.1707 per cent of the reachable set, 98.390 "
            "per cent of that within 4 ULP, worst 12. MEASURED forecast "
            "consequence on the committed 216-column WRF v4.6.1 capture: "
            "the deep mass flux xmb moves by up to ~7 per cent, median "
            "1.9 (tests/test_gf_deep_parity.py::"
            "test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_"
            "seven_percent). WOOF is the closer of the two to the value "
            "the scheme's equations define on 49.95 per cent of the "
            "reachable set and further on 9.59; NEITHER is determined to "
            "better than tens of per cent, so this is NOT a skill claim. "
            "To compare against WRF column by column, pin fzu through the "
            "fzu_override slots the three stage kernels expose "
            "(INS_fzu_up/dn, SINS_fzu_sh, DINS_fzu_up/dn/sh) -- a runtime "
            "input, not a build flag.",
            "DELIBERATE DIVERGENCE, owner ruling (no inherited WRF bugs): "
            "WRF's shallow k22 trigger is a MAXLOC over the array section "
            "heo_cup(2:kbmax) whose result module_cu_gf_sh.F uses as an "
            "absolute level index without adding the section offset, "
            "leaving k22 one level below the argmax wherever the argmax "
            "sits above level 2. The SHIPPED kernel uses the corrected "
            "indexing; the WRF-faithful off-by-one lives behind a launch "
            "flag only the parity suites set. Measured over the committed "
            "fixture: k22 moves on 3 of 18 cases (6, 13, 16), all three "
            "rejected under both modes with identical ierr, and ZERO "
            "output words differ at the scheme or driver boundary "
            "(tests/test_gf_shallow_cuda.py, the ledger test).",
            "DELIBERATE DIVERGENCE, WRF is undefined: "
            "the shallow closure pressure-difference loop is capped at "
            "min(kpbl,ktf-1) in strict and default arithmetic. WRF's "
            "kpbl=kte loop reads po_cup(k+1) above the column. The "
            "corrected WRF referee applies this cap and the MAXLOC "
            "section offset; every shipped output word is graded, "
            "including columns whose k22 changes. "
            "tests/test_gf_defects_gpu.py and "
            "tests/test_gf_regime_column_oracle_gpu.py own these checks.",
            "DELIBERATE DIVERGENCE, WRF is undefined: "
            "get_inversion_layers' first-derivative loop reads "
            "t_cup(kend+8) past the array end whenever kend > ktf-8 "
            "(module_cu_gf_deep.F, both live call sites pass "
            "kend = kstabi). The port clamps kend to ktf-8 -- the oracle "
            "capture clamps identically -- and COUNTS the clamps; the "
            "count is zero on the whole committed fixture and the gates "
            "assert it stays zero.",
            "ENGINE SEAM, recorded deviations of the cu_physics=3 "
            "adapter (woof/core/gf.py) -- the kernel behind it is "
            "bitwise; this is what the engine can hand it today, and it "
            "is the plumb-list for any label upgrade: (1) GFDRV's "
            "boundary-layer forcing (RTHBLTEN/RQVBLTEN) is fed the PBL "
            "slot's own raw dry-theta/qv rates, whichever of YSU, MYJ, "
            "MYNN, Shin-Hong or SASE holds the slot, and its radiative "
            "forcing (RTHRATEN) is fed the driver's held rates; the "
            "ADVECTIVE half (RTHFTEN/RQVFTEN) is fed the integrator's "
            "own exported rates -- the ARW dycore captures pure theta "
            "and qv advection at RK stage 1 of every step, uncoupled to "
            "K s-1 and kg kg-1 s-1 with a one-step lag, and the MPAS "
            "seam's caller supplies its own -- so all four forcing "
            "lanes are live.  PURE advection by construction: WRF's "
            "module_cumulus_driver.F:867 pre-folds RTHRATEN+RTHBLTEN "
            "into RTHFTEN for G3/NTiedtke and NOT for GFSCHEME, which "
            "sums the lanes itself; (2) GF's convective momentum "
            "tendencies are computed but not yet coupled (CumulusResult "
            "carries no momentum slots); (3) lower-face w is passed in "
            "strict and default arithmetic, matching GFDRV's physical "
            "interface. This deliberately changes the former adjacent-face "
            "average, which changed closure forcing and rain.",
            "MEASURED regime behaviour (2026-08-17, 12 km single-domain "
            "real-case twins on the tree route, 150x120x49, 6 h, KF "
            "control differing only in cu_physics/cudt): under strong "
            "synoptic forcing (1974-04-03 12-18Z) GF's domain-mean RAINC "
            "is ~40% of the KF control's -- ordinary inter-scheme "
            "spread; under weak forcing (1999-05-03 12-18Z, Ohio "
            "valley) it is 1-2% of KF's, i.e. the scheme is nearly "
            "silent where KF still rains. A column-level kernel probe "
            "of the weak-forcing state (18,000 real columns through "
            "gf_gfdrv_stage) found the deep trigger rejecting every "
            "column under four forcing arms alike -- the then-shipped "
            "zero-forcing seam, HFX/QFX-reconstructed RTHBLTEN/"
            "RQVBLTEN, and radiative forcing of either sign -- so the "
            "weak-forcing silence is the bitwise scheme's own "
            "trigger/closure response to these inputs, not a numeric "
            "defect at the adapter seam. A user expecting KF-like "
            "convective rain from cu_physics=3 on a weakly forced case "
            "will see almost none; that is the measured shape of the "
            "scheme as fed then, recorded here so it is not "
            "re-diagnosed as breakage. NOTE (2026-08-20): those twins "
            "predate the boundary-layer forcing lanes. Re-measured on a "
            "12 km GEM real case, one forecast hour, GF+YSU, feeding "
            "RTHBLTEN/RQVBLTEN moves domain-total RAINC by 12.8% "
            "(3.3% at 3 km, where the scale-aware closure damps "
            "itself), so the KF-relative ratios above are indicative "
            "and not re-verified against the current seam.",
        ],
    }
    # Grell 3D (cu_physics=5), WRF's G3SCHEME.  No PBL requirement: the
    # deep arm reads no KPBL, and the shallow arm guards every KPBL use
    # with `kpbl > 5` (module_cu_g3.F:1238-1241), so with the PBL slot off
    # it takes the moist-static-energy maximum as WRF does.  No template
    # selects it: a user asks for it explicitly or does not get it.
    cumulus_options["grell-3d"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"cudt_minutes": 0.0, "moist": True},
            "required_settings_reasons": {"moist": _CUMULUS_MOIST_REASON},
            "requires_components": {},
        },
        "extensions": {},
        "implemented": True,
        "label": "Grell 3D",
        "maturity": "implemented-unverified",
        "parameters": {"cudt_minutes": 0.0, "clos_choice": 0, "ishallow": 0,
                       "cugd_avedx": 1, "maxiens": 1, "maxens": 3, "maxens2": 3,
                       "maxens3": 16, "ensdim": 144, "cu_diag": 0,
                       "convtrans_avglen_m": 30.0},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 5},
        "warnings": [
            "The strict GPUWM_WRF_EXACT=1 check requires 0 ULP against public WRF v4.6.1 "
            "G3DRV and conv_grell_spread3d. The default arm uses the same source and "
            "operation-order-pinned shared loader; it also requires 0 ULP. "
            "tests/test_g3_cu5_parity.py covers both shallow arms, spreading, all "
            "16 closure selectors and nondefault ensemble dimensions. This is "
            "implementation conformance evidence, not observation-based forecast skill.",
            "With ishallow=0 WRF leaves three shallow level outputs uninitialized "
            "(module_cu_g3.F:487-490). The port writes zero; a poisoned oracle build "
            "isolates this declared divergence. Closure selector 16 keeps WRF's original copy order.",
            "cu_diag=1 carries WRF convtrans_prep time averages, cloud fraction, "
            "first-window behavior and fixed/adaptive window boundaries. These arrays "
            "restart through driver.fields and publish as a history extension of "
            "WRF's restart-only fields. tests/test_g3_diag.py compares full windows.",
            "G3 uses a four-point lateral boundary exclusion and reads a 3x3 forcing "
            "and spreading neighborhood, extended to 5x5 under cugd_avedx>1. It runs "
            "on the model step with the WRF forcing fold and per-call rain amount.",
        ],
    }
    # Retire the earlier inactive-control guards from seeded registries.
    # Config validation already refuses nondefault G3 controls without G3;
    # adding their defaults to other options changes legacy identities.
    from woof.config import G3_RUN_DEFAULTS
    for option in cumulus_options.values():
        if option.get("selectors", {}).get("cu_physics") == 5:
            continue
        constraints = option.get("constraints", {})
        for key in ("required_settings", "required_settings_reasons"):
            settings = constraints.get(key, {})
            for name in G3_RUN_DEFAULTS:
                if name != "cugd_avedx":
                    settings.pop(name, None)
            if not settings:
                constraints.pop(key, None)
        if not constraints:
            option.pop("constraints", None)
    # Betts-Miller-Janjic (cu_physics=2), the first scheme admitted as a
    # consumers.column_kernel row: the kernel (woof/core/kernels/bmj.cu)
    # plus this option and that row are the whole admission; the launcher
    # is the shared table-driven route.  It reads no KPBL (BMJDRV computes
    # LPBL and BMJ never uses it), so no PBL scheme is required; WRF's usual
    # pairing is MYJ, which is ported.  No template selects it.
    cumulus_options["betts-miller-janjic"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            "required_settings_reasons": {
                "moist": _CUMULUS_MOIST_REASON},
        },
        "extensions": {},
        "implemented": True,
        "label": "Betts-Miller-Janjic",
        "maturity": "implemented-unverified",
        "parameters": {"bmj_rad_feedback": False},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 2},
        "warnings": [
            "Betts-Miller-Janjic's distance from WRF v4.6.1 is MEASURED "
            "on the whole driver: tools/bmj_wrf461_oracle drives the "
            "byte-unmodified module_cu_bmj.F (BMJINIT and BMJDRV) at "
            "gfortran -O0 over 96 columns in twelve families (deep land "
            "and sea, shallow-only, stable, moist-adiabat edge, high-top, "
            "low-LCL, elevated terrain, cold season, marginal trigger), "
            "four consecutive calls per arm so the carried CLDEFI is read "
            "back, both bmj_rad_feedback arms; the CUDA kernel matches it "
            "at 0 ULP on every output field and the BMJINIT tables at 0 "
            "ULP on every word (tests/test_bmj_wrf461_parity.py). That is "
            "conformance evidence. A strict native-run replay also matches "
            "251244 output and table words exactly; its initial rows do "
            "not convect. A three-hour real-input GPU forecast and hourly "
            "station/MRMS comparisons are documented in docs/cumulus-bmj.md. "
            "One case is not scientific validation, so this option remains "
            "'implemented-unverified'.",
            "DELIBERATE DIVERGENCE, owner ruling (no inherited WRF bugs) "
            "-- GAMMA, the Grell-Freitas precedent "
            "(docs/gf_gamma_known_delta.md). The bmj_rad_feedback cloud "
            "fraction (module_cu_bmj.F:308) divides by GAMMA(JPR+1.), "
            "which gfortran binds to a gamma that is not correctly "
            "rounded; WOOF's is. The oracle is run with a correctly "
            "rounded gamma interposed and the kernel matches that arm at "
            "0 ULP everywhere; against the stock arm only CCLDFRA, "
            "QCCONV and QICONV move, by at most a few ULP.",
            "The driver holds PRATEC and accumulates PRATEC*DT every model "
            "step. CLDEFI and convective cloud fields survive restarts.",
        ],
    }
    from types import SimpleNamespace
    from woof.config import bmj_feedback_radiation_supported, radiation_scheme_ids
    rad_options = registry["components"]["radiation"]["options"]
    unsupported_feedback = []
    for option_id, option in rad_options.items():
        pair = radiation_scheme_ids(SimpleNamespace(
            **(option.get("parameters", {}) | option.get("selectors", {}))))
        if not bmj_feedback_radiation_supported(pair):
            unsupported_feedback.append(option_id)
    feedback_reason = ("The selected radiation adapter does not merge the "
                       "carried convective cloud; BMJ cloud diagnostics "
                       "would never reach its optical inputs.")
    feedback_rules = [{"components": {"radiation": unsupported_feedback},
                       "settings": {"bmj_rad_feedback": [True]},
                       "reason": feedback_reason}]
    cumulus_options["betts-miller-janjic"]["constraints"][
        "refused_when"] = feedback_rules
    surface_options["mynn"]["constraints"]["requires_components"][
        "pbl"
    ] = ["off", "mynn", "sase", "uw", "eeps", "keps"]
    # WHY, not just WHAT.  "requires pbl in [...]" told a reader which
    # tuples were refused and nothing about what breaks, which is half a
    # refusal under the gate law; the evaluator renders this string after
    # the list.  The reason is ArWen's own field contract, not a WRF
    # citation: WRF's matrix is why the pairing is illegal THERE, and this
    # is what would happen HERE.
    surface_options["mynn"]["constraints"]["requires_components_reasons"] = {
        "pbl": (
            "the MYNN surface layer publishes psim/psih/gz1oz0 and does "
            "NOT publish fm/fh, the full similarity denominators "
            "ln(z/z0)-psi that YSU and Shin-Hong bind directly and divide "
            "by (they reconstruct zol = br*fm^2/fh from them). Those two "
            "would run on the allocated zeros -- finite, plausible and "
            "wrong -- for the whole forecast. Select the revised MM5 (1) "
            "or classic MM5 (91) surface layer for them, or the MYNN PBL, "
            "SASE, the UW PBL, EEPS, the K-epsilon PBL or no PBL scheme for this surface "
            "layer, all of which read only fields it publishes"),
    }
    cumulus_options["tiedtke"] = {
        "label": "Original cu6 cumulus", "implemented": True,
        "maturity": "implemented-unverified", "scientific_evidence": "none",
        "selectors": {"cu_physics": 6}, "parameters": {"cudt_minutes": 0.0},
        "constraints": {
            "required_settings": {"moist": True, "cudt_minutes": 0.0},
            "required_settings_reasons": {
                "moist": "the original cu6 kernel requires qv, qc and qi column arrays",
                "cudt_minutes": "per-call precipitation has no NCA hold; holding it would reapply that rain each step"},
            "requires_components": {}},
        "asset_requirements": [], "extensions": {},
        "reachability": {"state": "component-override"},
        "warnings": [
            "Original WRF v4.6.1 module_cu_tiedtke.F, the cumulus of WRF's "
            "CONUS physics suite. Its distance from WRF is MEASURED at the "
            "driver boundary: tools/tiedtke_wrf461_oracle runs the unmodified "
            "Fortran (gfortran -O0, glibc 2.43 libm) over 72 synthetic columns "
            "x 4 vertical grids x 3 configurations, and the GPU kernel matches "
            "every word of all six tendencies, RAINCV, PRATEC and CU_ACT_FLAG "
            "under GPUWM_WRF_EXACT=1 and without it, also with many j rows per "
            "launch. NAMED RESIDUAL: a WRF linked to glibc 2.40 or older has a "
            "tanf that is not correctly rounded; F:2939's tangent then differs "
            "by 1 ULP on 1,079,656 of the 1,070,141,404 reachable arguments, "
            "moving RTHCUTEN by at most 6.6e-7 K/s on the fixture "
            "(docs/cumulus-cu6.md). That is conformance evidence, not "
            "scientific validation: no matched woof/WRF forecast trajectory "
            "grades this scheme, which is why it is 'implemented-unverified'. "
            "The cu16 ntiedtke_tiedtke_closure option remains an emulation "
            "and does not select this scheme."]}

    # The old GFS simplified Arakawa-Schubert scheme (cu_physics=95, WRF's
    # OSASSCHEME), the binary64 adapter woof.core.column_cumulus_r8: one
    # CUDA translation unit, admitted by table.  No PBL requirement (it reads
    # no KPBL and no surface flux) and no cudt pin (it hands the driver
    # WRF's held PRATEC, so any STEPCU books its rain as WRF does). A named
    # composition selects it, and namelist import admits selector 95.
    cumulus_options["old-gfs-sas"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            "required_settings_reasons": {"moist": _CUMULUS_MOIST_REASON},
            "requires_components": {},
        },
        "extensions": {},
        "implemented": True,
        "label": "old GFS simplified Arakawa-Schubert (OSAS)",
        "maturity": "supported",
        "parameters": {},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 95},
        "warnings": [
            "The old GFS SAS scheme's distance from WRF v4.6.1 is MEASURED "
            "on the whole CU_OSAS call, not a fragment. "
            "tools/osas_wrf461_oracle runs the byte-pinned "
            "module_cu_osas.F through WRF's own preprocessing and the WRF "
            "0 ULP referee's compile flags, with XKT2 pinned, over 96 "
            "analytic soundings (deep tropical, midlatitude severe, "
            "shallow-only, capped, dry, near-saturated; land and sea) on "
            "a 44-level and a 29-level grid, six calls each at STEPCU 1, 3 "
            "and 5; the CUDA translation unit "
            "(woof/core/kernels/cu_osas.cu) reproduces every float32 "
            "output word and every binary64 word of the deep and shallow "
            "stage records, and the gfuncphys lookup tables it builds on "
            "the card hash to the oracle's (tests/"
            "test_cu_osas_wrf461_parity.py). A strict native-run capture "
            "adds 60,000 real-input output words with zero differences "
            "(tools/osas_wrf461_oracle/evidence/native-live-replay.json). "
            "The same module and support sources are byte-identical in "
            "WRF v4.7.1. The lane also records one three-hour native/GPU "
            "forecast comparison with actual ASOS observations. This "
            "settles the conformance label, not independent OSAS skill: "
            "the scientific evidence axis remains none because one "
            "composed suite does not isolate this option. "
            "See docs/cumulus-osas.md and the delivered lane receipts.",
            "DECLARED DIVERGENCE -- XKT2. WRF draws the cloud-top "
            "selector from a clock-seeded random_number on every call and "
            "reseeds per J row from the same clock "
            "(module_cu_osas.F:281-282, :2493-2506), so WRF is not "
            "reproducible and rows share their draws. The strict arm "
            "(GPUWM_WRF_EXACT=1) takes XKT2 as an injected input, WRF's "
            "own REGTEST value 0.1 when none is injected; the default arm "
            "draws it from a counter-based generator keyed on the model "
            "step and the column, reproducible and restart-safe.",
            "DECLARED DIVERGENCE, owner ruling (no inherited WRF bugs) -- "
            "the RN <= 0 back-out. When a column's rain evaporates to "
            "nothing WRF restores T1 and Q1 (module_cu_osas.F:2174-2183) "
            "but keeps the cloud-top condensate it already detrained into "
            "QL (:2044-2058): water created with no vapour removed. The "
            "default arm restores QL with T1 and Q1. The production strict "
            "arm reproduces WRF's back-out through its metadata flags. "
            "With strict arithmetic the tests isolate the correction to "
            "those backed-out columns. Default fast arithmetic uses CUDA "
            "binary64 intrinsics and fused operations; it is not a bitwise "
            "WRF replay. Tables retain their native words in both modes.",
        ],
    }
    # ---- New Tiedtke (cu_physics=16) -------------------------------------
    # THE PBL REQUIREMENT IS RETIRED, and this row is owned here now so
    # the retirement cannot be undone by a carried-through JSON row.  It
    # was cloned from Grell-Freitas, whose reason does not transfer: GF
    # reads fields["kpbl"] as a ONE-BASED column index and divides by
    # t[kpbl], so with the slot off it reads slot 0 of an uninitialised
    # workspace.  New Tiedtke reads no kpbl at all; its hfx/qfx come from
    # the SURFACE stack, which runs independently of bl_pbl_physics, and
    # its gf_rthblten/gf_rqvblten lanes are allocated zeros -- WRF's own
    # RTHBLTEN=0 fold at module_cumulus_driver.F:879-880.  The
    # "cumastrn:509 zdhpbl" integral the old reason cited runs from the
    # CLOUD-BASE index, not from a PBL index.
    ntiedtke = cumulus_options["new-tiedtke"]
    ntiedtke["constraints"]["requires_components"].pop("pbl", None)
    ntiedtke["extensions"].pop("arwen_pbl_structural_requirement", None)
    _ntiedtke_pbl_off_warning = (
        "New Tiedtke runs with the PBL slot off. It reads no KPBL, its "
        "surface fluxes come from the surface layer and the land-surface "
        "model rather than from the PBL scheme, and its advective-forcing "
        "lanes are the zero planes WRF's own cumulus driver folds in when "
        "no PBL tendency exists. The measured runs all carry a PBL "
        "scheme, so the PBL-off configuration is admitted on the field "
        "contract, not on evidence.")
    if _ntiedtke_pbl_off_warning not in ntiedtke["warnings"]:
        ntiedtke["warnings"].append(_ntiedtke_pbl_off_warning)

    # ---- KIAPS SAS (cu_physics=14) ---------------------------------------
    # The first SAS-family member.  Admitted as THIS ROW: the stock class
    # (woof.core.sas.SasCumulus) is the family's one seam and the variant
    # is its cu_physics; there is no per-scheme adapter.  WRF honours cudt
    # for KSAS (STEPCU calls, PRATEC held and accumulated every step), so
    # no cudt pin; WRF's default of 0 is the parameter.  Under an adaptive
    # step WRF sets cudt = 0 itself, and that rule is the extension below,
    # read by woof.config.validate_run_config and the namelist importer.
    cumulus_options["kiaps-sas"] = {
        "asset_requirements": [],
        "constraints": {
            "required_settings": {"moist": True},
            "required_settings_reasons": {"moist": _CUMULUS_MOIST_REASON},
        },
        "extensions": {
            "wrf_adaptive_step_cudt_zero": {
                "wrf_call_site": "share/module_check_a_mundo.F:2141-2160",
                "rule": (
                    "with use_adaptive_time_step WRF sets cudt = 0 for BMJ, "
                    "every SAS scheme and Tiedtke ('CUDT=0 has been done for "
                    "you'): the scheme's closure is formed over "
                    "delt = dt*stepcu, which a varying step cannot hold"),
            },
            "checkpoint_rebuild_arrays": [
                "_hold", "_workspace", "raincv", "hbot", "htop"],
            "sas_family": {
                "kernel": "woof/core/kernels/sas.cu (sas_ksas_column)",
                "variant_table": "woof.core.sas.SAS_VARIANTS",
                "constant_table": "woof.core.sas.WRF_MODEL_CONSTANTS",
                "constant_table_sha256": __import__(
                    "woof.core.sas", fromlist=["constants_sha256"]
                ).constants_sha256(),
                "wrf_source": "phys/module_cu_ksas.F (WRF v4.7.1, identical to v4.6.1)",
                "wrf_source_sha256": "84d88fe79c3aba7c299d0efaee216c596021c9659b445dd8d3365c5799a48854",
                "license": "WRF public domain; licenses/NOTICE-KSAS-WRF.txt",
            },
        },
        "implemented": True,
        "label": "KIAPS SAS",
        "maturity": "supported",
        "parameters": {"cudt_minutes": 0.0},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none",
        "selectors": {"cu_physics": 14},
        "warnings": [
            "KIAPS SAS transcribes the public WRF v4.7.1 module, identical "
            "to v4.6.1. A real native forecast capture at step 21 replayed "
            "45,000 columns and all ten outputs with zero differing "
            "words out of 13,410,000. The old capture was not retained; "
            "replacement inputs and outputs, the two-rank native command "
            "and strict replay are permanently retained at "
            "datasets/paper/wrf-ports/spd-recapture/cu14/MANIFEST.md "
            "(SHA256 909feb33b673720aca501f722067bca327a9d637ffd95a6c881795ff1cbb45de), "
            "engine c69a56084ad73e4d8ae24350ded8534198b34427, WRF 4.7.1, "
            "glibc 2.39. Replay: python tools/ksas_wrf461_oracle/replay_capture.py "
            "<cu14-dataset>/capture-inputs.bin <cu14-dataset>/capture-outputs.bin "
            "--receipt replay-rerun.json. The additional "
            "768-column fixture is constructed and has a planted-bit control. "
            "Default forecasts use CUDA math and contraction; strict "
            "proof selects GPUWM_NUMERICS=strict and the native libm "
            "through GPUWM_WRF_GLIBC. A 300 by 300, 49-level, three-hour "
            "forecast completed 720 steps through woof go in fast mode. "
            "The 10, 11 and 12 UTC frames were reviewed beside native WRF "
            "and ASOS surface observations through woof.rustwx. "
            "tools/ksas_wrf461_oracle/forecast.json records that bounded "
            "comparison and the 3.994 ms versus New SAS 7.846 ms stage "
            "timing. One surface case does not establish convection "
            "skill; radar, rainfall observations and full nested or "
            "ensemble forecasts were not checked.",
            "ENGINE SEAM, recorded deviations of the cu_physics=14 callable "
            "(woof/core/sas.py SasCumulus): (1) HPBL_HOLD is the PBL height "
            "the driver saw before this step's PBL call, through the "
            "hold_pbl_height hook; on a step whose PBL does not run, or with "
            "the PBL slot off, WRF passes an uninitialised stack array "
            "(module_first_rk_step_part1.F:130), and here the hold is the "
            "current PBLH, so the boundary-layer growth term is zero. "
            "PRATEC is held and RAINC accumulates PRATEC*DT each model "
            "step. Surface rain retains the separately rounded native "
            "RAINCV. HBOT/HTOP and raw momentum use persistent driver "
            "fields for restart and relocation; convection levels are "
            "published as an explicit extension of WRF's restart-only "
            "history inventory.",
        ],
    }

    surface_options["mynn"]["warnings"] = [
        warning for warning in surface_options["mynn"]["warnings"]
        if not warning.startswith("MYNN is admitted only as the coupled")
        and not warning.startswith(
            "WRF v4.6.1 admits this surface layer with PBL off")
    ]
    surface_options["mynn"]["warnings"].insert(
        0,
        "WRF v4.6.1 admits this surface layer with PBL off or MYNN PBL. "
        "MYNN PBL also admits revised/classic MM5 surface layers; the exact "
        "16-cell authority is phys/module_physics_init.F:3699-3704,"
        "3837-3839 and is published under "
        "authority.wrf_v461_compatibility_matrix.",
    )

    # PBL-off with km_opt=4 follows WRF's diff_opt=2 vertical_diffusion_2
    # path.  The prior registry rail described an absent vertical operator;
    # that operator is now ported, including USTM/HFX/QFX surface fluxes.
    pbl_off = pbl_options["off"]
    pbl_off.setdefault("constraints", {}).setdefault(
        "forbidden_setting_values", {}).pop("km_opt", None)
    if not pbl_off["constraints"]["forbidden_setting_values"]:
        pbl_off["constraints"].pop("forbidden_setting_values")
    if not pbl_off["constraints"]:
        pbl_off.pop("constraints")
    pbl_off.setdefault("extensions", {})["wrf_vertical_diffusion_2"] = {
        "activation": "bl_pbl_physics=0, diff_opt=2, km_opt=4",
        "wrf_call_site": "dyn_em/module_first_rk_step_part2.F:1008-1074",
        "wrf_coefficient_policy": (
            "dyn_em/module_diffusion_em.F:2018-2023 sets xkmv=xkmh and "
            "xkhv=0 for km_opt=4"),
        "ported_operators": [
            "tau13 u momentum", "tau23 v momentum", "tau33 w momentum",
            "USTM lower-boundary momentum stress",
            "HFX lower-boundary heat flux", "QFX lower-boundary vapor flux",
        ],
    }
    pbl_off["reachability"] = {"state": "component-override"}

    # A source-driven active LSM still needs a surface-layer writer in ArWen.
    # This is not a WRF prohibition: it is the named local structural seam
    # that keeps the sfclay=0/LSM>0 cells fail-closed.
    land_options = registry["components"]["land_surface"]["options"]

    from woof.core.pxlsm import PXLSM_DIVERGENCES
    land_options["pleim-xiu"] = {
        "asset_requirements": [], "constraints": {}, "extensions": {},
        "implemented": True, "label": "Pleim-Xiu land-surface model",
        # PX's soil column is two layers (woof.config.LAND_SURFACE_SOIL_LAYERS
        # row 7); stated here as every other land option states its count,
        # so a plan that selects PX resolves the geometry PX defines instead
        # of inheriting the template's four-layer Noah column and being
        # refused for it.
        "maturity": "experimental-runtime", "parameters": {"num_soil_layers": 2},
        "reachability": {"state": "component-override"},
        "scientific_evidence": "none", "selectors": {"sf_surface_physics": 7},
        "warnings": list(PXLSM_DIVERGENCES),
        "verification_scope": "native-module-oracle-and-single-real-forecast",
        "verification_scope_note": (
            "525600 real native output words replay bitwise, zero exclusions. "
            "A three-hour stock-namelist forecast completed through woof go "
            "with PX land/surface layer and ACM2, with Rust maps beside native "
            "WRF and ASOS at four hours. Both nest feedback modes, restart "
            "continuation, tree checkpoint and ensembles have GPU checks. "
            "This does not claim whole-trajectory bitwise equality or general "
            "observation skill. See woof/data/pxlsm/coupled-proof.json."),
    }
    for key, default in (("pxlsm_smois_init", 1), ("pxlsm_modis_veg", 0),
                         ("pxlsm_soil_nudge", 0), ("sgfdda_interval_m", 180)):
        registry["parameters"][key] = {
            "type": "integer", "default": default, "per_domain": True,
            "component_id": "land_surface", "read_when": {"sf_surface_physics": 7},
            "consuming_read": "woof/core/pxlsm_runtime.py",
        }
    registry["parameters"]["pxlsm_soil_nudge"]["verification"] = (
        "woof/data/pxlsm/soil-nudging-proof.json: live native auxinput9 "
        "and per-domain/member CUDA clock; 525600 physical words exact")

    # SOIL GEOMETRY, FROM THE SCHEMES' OWN TABLES.  RUC's row was
    # ``required_settings.num_soil_layers = 9`` -- a single value, because
    # that kind can say nothing else -- so plan review refused the
    # six-level RUC column that woof/config.py admits, the kernel sizes
    # itself for and a completed forecast has written a wrfout on.  The
    # multi-valued kind says the set instead, and the set is READ from
    # woof.config.LAND_SURFACE_SOIL_LAYERS, which reads it from each
    # scheme's own module (Noah and Noah-MP: woof.core.noah; RUC:
    # woof.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS, the counts
    # WRF's init_soil_depth_3 tabulates).  A scheme that gains a geometry
    # gains the registry row, the parameter enum and the loader's
    # admission in one edit, which is the arbitrary acceptance test
    # applied to a soil table.  The EVIDENCE difference between RUC's two
    # geometries is not stated here: it is the option's warning and the
    # run receipt's soil_geometry_evidence line, because maturity warns
    # and never blocks.
    from woof.config import LAND_SURFACE_SOIL_LAYERS
    # WHICH OPTIONS GET A GEOMETRY IS DERIVED, NOT LISTED.  A hand-typed
    # option-id -> selector map standing beside the registry being built is
    # the second table this whole edit exists to delete: every option
    # already carries its own ``selectors`` row one attribute away, and a
    # map also fixes the membership, so a newly registered land-surface
    # scheme would silently get no admitted geometry and contribute
    # nothing to the enum.  Reading the rows means registering a scheme is
    # the same edit that gives it both, and a scheme whose soil geometry
    # no module publishes fails the BUILD by name below rather than
    # shipping unconstrained.
    #
    # sf_surface_physics=0 is excluded and is the only value named here.
    # It selects no scheme at all: its LAND_SURFACE_SOIL_LAYERS row is the
    # length of a wrfout axis no field is written on
    # (woof.config.NO_LAND_SURFACE_SOIL_LAYERS and the DIVERGENCE note
    # above it), not a soil column, so constraining a knob against it
    # would refuse counts on a run that allocates no soil state.
    (_soil_selector_key,) = (
        registry["components"]["land_surface"]["selector_keys"])
    _NO_LAND_SURFACE_SELECTOR = 0
    _SOIL_SCHEME_SELECTORS = {
        option_id: int(option["selectors"][_soil_selector_key])
        for option_id, option in land_options.items()
        if int(option["selectors"][_soil_selector_key])
        != _NO_LAND_SURFACE_SELECTOR}
    _soil_geometry_reason = (
        "a soil column exists only where its LEVEL DEPTHS do. WRF's "
        "generators tabulate zs for these counts and no others -- "
        "init_soil_depth_2 (Noah, Noah-MP) is fatal at any count but 4, "
        "and init_soil_depth_3 (RUC) tabulates 6 and 9 and leaves zs "
        "uninitialised otherwise -- and dzs is derived from zs, so a "
        "count with no row has no soil column at all rather than a "
        "coarse one. Select a count this scheme defines, or the scheme "
        "that defines the count you want")
    selectable_soil_counts: set[int] = set()
    for option_id, selector in sorted(_SOIL_SCHEME_SELECTORS.items()):
        try:
            defined = LAND_SURFACE_SOIL_LAYERS[selector]
        except KeyError:
            raise KeyError(
                f"land_surface option {option_id!r} selects "
                f"{_soil_selector_key}={selector}, for which no module "
                "publishes a soil geometry: give the scheme a row in "
                "woof.config._LandSurfaceSoilLayers._PROVIDERS, naming the "
                "module and attributes that hold its counts, before "
                "registering it") from None
        counts = [int(value) for value in defined]
        selectable_soil_counts.update(counts)
        constraints = land_options[option_id].setdefault("constraints", {})
        constraints.setdefault("required_settings", {}).pop(
            "num_soil_layers", None)
        constraints.setdefault(
            "admitted_setting_values", {})["num_soil_layers"] = counts
        constraints.setdefault(
            "admitted_setting_values_reasons",
            {})["num_soil_layers"] = _soil_geometry_reason
    # Use the same admitted selectors as config and namelist import.
    from woof.config import (NOAHMP_OPTIONS_WITHOUT_CONSUMER,
                              NOAHMP_OPTION_VALUES)
    _noahmp_required = land_options["noah-mp"].setdefault(
        "constraints", {}).setdefault("required_settings", {})
    for _name in sorted(NOAHMP_OPTIONS_WITHOUT_CONSUMER):
        _noahmp_required.pop(_name, None)
    _noahmp_constraints = land_options["noah-mp"]["constraints"]
    _noahmp_constraints.setdefault("admitted_setting_ranges",{})["wtddt"]={"minimum":0.0,"finite":True,"units":"minutes","evidence":"native complete spatial groundwater; opt_run=5 only"}
    _noahmp_required.pop("soiltstep",None)
    _noahmp_required.pop("noahmp_acc_dt",None)
    _noahmp_constraints.setdefault("admitted_setting_ranges",{})["noahmp_acc_dt"]={"minimum":0.0,"finite":True,"units":"minutes","evidence":"native complete output accumulators"}
    _noahmp_constraints.setdefault("admitted_setting_ranges",{})["soiltstep"]={"minimum":0.0,"finite":True,"units":"seconds","evidence":"native decimated soil cadence with persisted Registry ACC fields"}
    for _name, _values in NOAHMP_OPTION_VALUES.items():
        _noahmp_required.pop(_name, None)
        _noahmp_constraints.setdefault("admitted_setting_values", {})[_name] = list(_values)
        _noahmp_constraints.setdefault("admitted_setting_values_reasons", {})[_name] = (
            "The canonical default is unchanged. Admitted nondefault values "
            "have strict native WRF v4.6.1 whole-module replays: four real "
            "columns per captured configuration, all 153 formal outputs, "
            "980 words, zero differences and zero exclusions. The captured "
            "configurations are BTR2/RSF2, ALB1/BTR3/RSF3, RSF4, SNF2-5 "
            "individually, DVEG1/3/7/8/9 individually, INF2, RUN1/2/4/5/6/7/8, SOIL2/3, RAD1/2, FRZ2, STC2/3, TBOT1, CRS2 and SFC2 individually. "
            "Each completed a real "
            "three-hour forecast examined on hourly Rust maps beside native "
            "WRF and station observations. Selected columns use resident "
            "device classification and parameter gathering; restart, history, "
            "checkpoint, actual two-member ensembles and both nest feedback "
            "modes are covered. See tools/noahmp_options_wrf461_oracle/README.md "
            "for evidence scope; this does not qualify other option values.")
        if _name.startswith("snicar_"):
            # These kw-only RunConfig controls are absent from the old knob
            # survey. An admitted setting without its parameter default
            # resolves to None and makes every Noah-MP plan unlaunchable.
            from woof.config import RunConfig
            _default = RunConfig.__dataclass_fields__[_name].default
            registry["parameters"].setdefault(_name, {
                "type": "boolean" if isinstance(_default, bool) else "integer",
                "default": _default,
                "enum": list(_values),
                "per_domain": False,
                "consuming_read": "woof/core/snow_papers.py:settings",
            })
            _noahmp_constraints["admitted_setting_values_reasons"][_name] = (
                "These settings select the paper-based snow rewrite at opt_alb=3: "
                "written from papers, not bitwise to WRF. Physical response and "
                "lifecycle checks cover snow habits and impurity mixing; the "
                "native v4.6.1 replay claim does not cover these settings. "
                "See docs/SNOW-PAPERS.md and tests/test_snow_papers.py.")
    _noahmp_constraints["admitted_setting_values_reasons"]["noahmp_output"] = (
        "Level 3 runs all 58 native WRF output accumulators on resident GPU arrays. "
        "The pinned diagnostics driver treats levels 0/1/2 as standard output. "
        "See replay_output.py for the complete real-input bitwise comparison.")
    _noahmp_constraints["admitted_setting_values_reasons"]["opt_alb"] += (
        " opt_alb=3 is a separate rewrite: written from papers, not bitwise to WRF. "
        "It uses generated optics, layered snow and glacier ice, four snow habits "
        "and internal or external impurity mixing. "
        "Native SNICAR forecast qualification is recorded separately; the preceding "
        "v4.6.1 replay claim covers only values 1 and 2. See docs/SNOW-PAPERS.md.")
    # The enum is assigned after the parameter tables are merged below,
    # which deep-copies this module's spec over whatever stands here.
    from woof.core.noahmp_soil_options import SOIL4_LABEL
    _noahmp_constraints["admitted_setting_values_reasons"]["opt_soil"] += (
        " Option 4: "+SOIL4_LABEL+". Requires all nine explicit *_3D soil-property "
        "inputs. Category initialization supplies ancillary KDT/FRZX; urban "
        "overrides remain native. No whole-model native parity is claimed.")

    for option_id in ("noah", "ruc-lsm", "noah-mp"):
        option = land_options[option_id]
        # "eta-similarity" joined all three lists with the MYJ port, and
        # the reason it is all three is the reason this list exists at all.
        # It is a STRUCTURAL statement -- "who writes the exchange fields
        # this seam reads" -- not an evidence one, exactly as the extension
        # below says, and the Eta layer writes every one of them
        # (UST/CHS/CHS2/CQS2/FLHC/FLQC plus the RIB the driver carries as
        # BR); none of woof's three LSM runtimes reads a MM5-only surface
        # field by name.  Restricting it to Noah, the pairing the port
        # actually exercised, was tried and is WRONG here: it invents a
        # prohibition WRF does not make, it puts this table into
        # disagreement with validate_run_config over 744 combinations (the
        # thing tests/test_authority_agreement.py exists to catch), and it
        # states evidence in the place reserved for structure.  Where the
        # evidence scope belongs is the OPTION's maturity and warnings,
        # which say implemented-unverified and say what was run -- the same
        # posture "mynn" already has in these three lists.
        accepted = ["revised-mm5", "classic-mm5", "mynn", "eta-similarity", "temf", "gfs", "qnse", "pleim-xiu"]
        option.setdefault("constraints", {}).setdefault(
            "requires_components", {})["surface_layer"] = accepted
        option["constraints"]["requires_components_reasons"] = {
            "surface_layer": (
                "WOOF's active land-surface drivers consume the "
                "UST/CHS/CHS2/CQS2/FLHC/FLQC exchange fields the surface "
                "layer writes; with the surface layer off those fields are "
                "allocated and stay identically zero, and Noah-MP's "
                "write-back then divides by chs2/cqs2 behind a "
                "substitute-1.0 guard, so the run degenerates silently "
                "instead of failing")}
        option.setdefault("extensions", {})[
            "arwen_surface_exchange_structural_requirement"
        ] = {
            "reason": (
                "WOOF's active LSM drivers consume UST/CHS/CHS2/CQS2/"
                "FLHC/FLQC exchange fields that have no writer when "
                "sf_sfclay_physics=0"),
            "classification": (
                "WOOF structural constraint; WRF v4.6.1 does not prohibit "
                "sf_sfclay_physics=0 with an active LSM"),
        }

    # The experiment-per-domain front door now exposes harmless WRF-legal
    # PBL-off and radiation-off choices, plus every WRF-legal implemented
    # surface-layer pairing.  This is option-scoped so ArWen's analytic
    # radiation proxy remains outside normal registry reachability.
    tree_route = registry["runner_routes"][
        "tools.prepared_domain_tree_forecast"]
    tree_route["allowed_component_options"] = {
        # "sase" is listed so the closure is selectable per domain on the
        # tree route.  No template selects it, so its reachability is
        # "component-override": a user asks for it explicitly or does not
        # get it, which is the right posture for an experimental scheme.
        # "shinhong" is listed so the closure is selectable per domain
        # here too.  Since the gray-zone template registered it, its
        # reachability is "template" -- a user can also ask for the whole
        # registered suite -- and this entry keeps the per-domain override
        # legal beside it.
        # "myj" and "eta-similarity" are listed TOGETHER and only
        # together: they are the one pair in this registry, and a plan that
        # names one without the other is refused by their own
        # requires_components and again by
        # woof.config.validate_myj_pairing.
        # "uw" is listed so the UW moist-turbulence PBL is selectable per
        # domain here; no template selects it (component-override).
        "pbl": ["off", "acm2", "ysu", "mynn", "sase", "shinhong", "myj", "uw", "gfs", "qnse", "boulac", "mrf", "ccpp-tke-edmf", "eeps", "temf", "gbm", "keps"],
        "surface_layer": [
            "revised-mm5", "classic-mm5", "mynn", "eta-similarity", "gfs", "qnse", "pleim-xiu"],
        # "wrf-rrtm-dudhia" is WRF's classic 1/1 pair.  No template
        # selects it, so like sase and grell-freitas it is a
        # component-override: a user asks for RRTM longwave explicitly.
        # It is the only registered longwave-on pairing that is not the
        # coupled RRTMG adapter, which is the point of listing it here.
        "radiation": [
            "off", "dudhia-shortwave", "gsfc-shortwave", "wrf-rrtm-dudhia", "rte-rrtmgp", "new-goddard", "fu-liou-gu", "wrf-gfdleta", "wrf-cam3", "relaxation-31"],
    }
    # The per-domain override surface names the WHOLE implemented cumulus
    # set, so it is derived from the options rather than typed: a typed
    # list was not updated when New Tiedtke (cu_physics 16) landed, and
    # the published declaration went on naming three schemes while the
    # engine ran four.  Ordered by cu_physics.  Each option's own
    # required_settings row (New Tiedtke's cudt_minutes 0 and moist)
    # still applies to every plan that names it, and the single-domain
    # route inherits this list below.
    tree_route["allowed_component_options"]["cumulus"] = sorted(
        (option_id for option_id, option in cumulus_options.items()
         if option.get("implemented") is True),
        key=lambda option_id: cumulus_options[option_id]["selectors"][
            "cu_physics"])
    surface_options["revised-mm5"]["reachability"] = {
        "state": "component-override"}
    radiation_options = registry["components"]["radiation"]["options"]
    radiation_options["off"]["reachability"] = {
        "state": "component-override"}
    analytic = radiation_options["analytic-clear-sky"]
    analytic["reachability"] = {
        "state": "unreachable",
        "blocker": (
            "The 90/90 analytic clear-sky proxy is WOOF-specific: WRF "
            "v4.6.1 registers no equivalent package "
            "(Registry/Registry.EM_COMMON:3107-3125). It remains available "
            "to unnamed domain trees only through the authority-level "
            "expert-tuple-v1 acknowledgement and is intentionally excluded "
            "from allowed_component_options."),
    }
    analytic["warnings"] = [
        "WOOF-SPECIFIC, NOT A WRF SCHEME. Analytic 90/90 radiation remains "
        "an expert acknowledgement path because WRF v4.6.1 has no equivalent "
        "Registry package."
    ]
    # THE RRTMG RECEIPT TOKEN BELONGS TO THE RADIATION OPTION.  The RRTMG
    # templates carry wrf_rrtmg_compatibility in their parameters, and a
    # per-domain radiation override on the tree route kept the template's
    # token beside a 0/0, 0/1, 1/1 or 90/90 pair -- plan review accepted
    # the domain and validate_run_config refused it at run start
    # ("requires the resolved 4/4 pair"), the two-door disagreement
    # tests/test_physics_combination_matrix.py measures.  Every radiation
    # option that is not the RRTMG 4/4 pair resolves the token to 'none'
    # through its own parameters, the way the legacy aggregate already
    # did, so the override resolves to the run the door admits.
    from types import SimpleNamespace
    from woof.config import radiation_scheme_ids
    for option_id, option in radiation_options.items():
        if option.get("implemented") is not True:
            continue
        if radiation_scheme_ids(SimpleNamespace(
                **(option.get("parameters", {})
                   | option.get("selectors", {})))) == (4, 4):
            continue
        option.setdefault("parameters", {})["wrf_rrtmg_compatibility"] = "none"

    # WHY a turbulence closure needs the PBL slot it declares, in the
    # words the run door uses (woof.config validate_run_config), so plan
    # review names the breakage and not only the pairing.
    turbulence_options = registry["components"]["turbulence"]["options"]
    _turbulence_reasons = {
        "closure-supplied": (
            "km_opt=0 runs no horizontal mixing operator, so the closure "
            "must supply the mixing itself: SASE computes its own "
            "horizontal mixing from its diffusivities and is the one PBL "
            "option that does, while every other PBL scheme produces none "
            "and the run would carry no explicit horizontal mixing at all "
            "(woof.config admits that deliberately only through the "
            "km_opt_zero_acknowledgement research control)"),
        "smagorinsky-3d": (
            "its vertical exchange pair (kmv/khv) is applied by "
            "vertical_diffusion_2, which runs only with the PBL off, so "
            "with a PBL scheme on only the horizontal half of the closure "
            "would run and the run would not be the 3-D closure it names; "
            "km_opt=4 (2-D Smagorinsky) is the horizontal-only closure for "
            "a PBL-on domain"),
        "tke-1.5-order": (
            "its vertical TKE self-diffusion and surface TKE forcing are "
            "applied by vertical_diffusion_2, which runs only with the PBL "
            "off because a PBL scheme is already the column's vertical "
            "closure and running both would double-count vertical mixing; "
            "with a PBL on, TKE would be produced and dissipated "
            "column-locally with no vertical redistribution, so the run "
            "would not be the prognostic-TKE closure it names"),
    }
    # Citation offsets follow the actual retained source declaration. Keep
    # the claim's text and semantic anchor unchanged across source intake.
    dycore_lines = (MODEL / "woof/core/dycore.py").read_text(
        encoding="utf-8").splitlines()
    for option_id, anchor in {
            "tke-1.5-order": "WRF v4.6.1 km_opt=2:",
            "smagorinsky-3d": "WRF v4.6.1 km_opt=3:"}.items():
        matches = [number for number, line in enumerate(dycore_lines, 1)
                   if anchor in line]
        if len(matches) != 1:
            raise ValueError(f"registry citation needs one {anchor!r} declaration, got {matches}")
        citation = f"woof/core/dycore.py:{matches[0]}"
        turbulence_options[option_id]["warnings"] = [
            re.sub(r"woof/core/dycore\.py:\d+", citation, warning)
            for warning in turbulence_options[option_id]["warnings"]]
    for option_id, reason in _turbulence_reasons.items():
        constraints = turbulence_options[option_id].setdefault("constraints", {})
        if "pbl" in constraints.get("requires_components", {}):
            constraints["requires_components_reasons"] = {"pbl": reason}
        if "bl_pbl_physics" in constraints.get("required_settings", {}):
            constraints["required_settings_reasons"] = {
                "bl_pbl_physics": reason}
    # Coordinate diffusion has no tke_rhs or metric vertical operator.
    # WRF permits its horizontal TKE coefficients alongside a PBL scheme.
    tke_constraints = turbulence_options["tke-1.5-order"]["constraints"]
    for table, key in (("required_settings", "bl_pbl_physics"),
                       ("required_settings_reasons", "bl_pbl_physics"),
                       ("requires_components", "pbl"),
                       ("requires_components_reasons", "pbl")):
        tke_constraints.get(table, {}).pop(key, None)
    tke_constraints["refused_when"] = [{
        "settings": {"diff_opt": [2]},
        "components": {"pbl": sorted(set(pbl_options) - {"off"})},
        "reason": ("diff_opt=2: " + _turbulence_reasons["tke-1.5-order"]
                   + "; diff_opt=1 instead selects WRF coordinate-surface mixing"),
        "remedy_label": "Set bl_pbl_physics=0 for metric TKE closure.",
        "remedy_settings": {"bl_pbl_physics": 0},
    }]
    # The two diffusion selectors' own refusals (woof.config
    # validate_km_opt), stated on every closure so plan review and the
    # front ends refuse what the run door refuses: tests/
    # test_authority_agreement.py measured both as "registry says
    # LAUNCHABLE, validate_run_config REFUSES".  Each closure's list is
    # assigned whole, after its own rules, so a rebuild is a fixpoint.
    from woof.config import TOPO_WIND_PBL_SCHEMES
    non_topo_pbl = sorted(
        option_id for option_id, option in pbl_options.items()
        if option.get("selectors", {}).get("bl_pbl_physics")
        not in TOPO_WIND_PBL_SCHEMES)
    no_pbl = sorted(
        option_id for option_id, option in pbl_options.items()
        if option.get("selectors", {}).get("bl_pbl_physics") == 0)
    for option_id, option in sorted(turbulence_options.items()):
        if option.get("implemented") is not True:
            continue
        km_opt = option["selectors"]["km_opt"]
        rules = [rule for rule in
                 option.get("constraints", {}).get("refused_when", [])
                 if not {"topo_wind", "gwd_opt"}.intersection(
                     rule.get("settings", {}))
                 and ("diff_opt" not in rule.get("settings", {})
                      or "components" in rule)]
        if km_opt not in (2, 4):
            rules.append({
                "settings": {"diff_opt": [1]},
                "reason": (
                    f"diff_opt=1 (WRF coordinate-surface diffusion) is "
                    f"implemented for km_opt=2 and 4, which supply its "
                    f"exchange coefficients; km_opt={km_opt} does not, so "
                    f"the run would carry no coordinate operator"),
                "remedy_label": (
                    "Set diff_opt=2 for the metric operator this closure "
                    "runs."),
                "remedy_settings": {"diff_opt": 2},
            })
        # mix_full_fields=false under diff_opt=2 carries no rule: the run
        # door admits it (woof/config.py, validate_km_opt), because WRF's
        # perturbation branch subtracts base-state profiles real.exe
        # leaves at zero, the same operator for a real-data run.
        # Every plan selects a turbulence option.  Its existing constraint
        # object carries this PBL coupling without creating a new empty
        # physics-identity wrapper for PBL off.  No sources clause: the
        # preflight re-checks this configuration-only admission rule.
        rules.append({
            "settings": {"topo_wind": [1, 2]},
            "components": {"pbl": non_topo_pbl},
            "reason": (
                "topo_wind acts only through the surface drag of the PBL "
                "schemes that read CTOPO/CTOPO2, YSU (bl_pbl_physics=1) and "
                "Shin-Hong (bl_pbl_physics=11); any other PBL does not "
                "consume the coefficients, so the run would name a "
                "terrain-wind correction without applying one"),
            "remedy_label": "Set topo_wind=0.",
            "remedy_settings": {"topo_wind": 0},
        })
        rules.append({
            "settings": {"gwd_opt": [1, 3]},
            "components": {"pbl": no_pbl},
            "reason": (
                "gwd_opt adds its drag inside the PBL driver, which a run "
                "with bl_pbl_physics=0 never calls, and reads the PBL "
                "height and top level an active PBL scheme writes"),
            "remedy_label": "Set gwd_opt=0.",
            "remedy_settings": {"gwd_opt": 0},
        })
        option.setdefault("constraints", {})["refused_when"] = rules
    sase_constraints = pbl_options["sase"]["constraints"]
    sase_constraints["required_settings_reasons"] = {
        "km_opt": (
            "SASE computes its own horizontal mixing from the closure's "
            "own diffusivities, so a km_opt mixing operator would "
            "double-count it"),
        "khdif": "constant-K diffusion may not silently stack on the SASE mixing",
        "kvdif": "constant-K diffusion may not silently stack on the SASE mixing",
        "moist": (
            "the closure mixes water vapour, cloud water and cloud ice "
            "beside theta and forms its stability from the saturated "
            "Brunt-Vaisala frequency; a dry state has nothing for it to "
            "integrate"),
    }
    # Every moist scheme says why it needs a moist state, once per slot.
    for option in registry["components"]["cumulus"]["options"].values():
        constraints = option.get("constraints", {})
        if constraints.get("required_settings", {}).get("moist") is True:
            constraints.setdefault("required_settings_reasons", {})[
                "moist"] = _CUMULUS_MOIST_REASON
    for option in registry["components"]["microphysics"]["options"].values():
        constraints = option.get("constraints", {})
        if constraints.get("required_settings", {}).get("moist") is True:
            constraints.setdefault("required_settings_reasons", {})[
                "moist"] = (
                    "a dry state allocates no water vapour or hydrometeor "
                    "fields, so the scheme would have nothing to integrate")
    # icloud=2 is an admitted metadata value, never a fixed-on parameter.
    for option in radiation_options.values():
        constraints = option.get("constraints", {})
        constraints.get("required_settings", {}).pop("icloud", None)
        constraints.get("required_settings_reasons", {}).pop("icloud", None)

    params = registry["parameters"]
    # Citations are read before the tables overwrite the specs that carry
    # them, so a verified citation survives a spec being tightened.
    prior_citations = {
        name: spec.get("consuming_read")
        for name, spec in params.items() if isinstance(spec, dict)}

    selectors: set[str] = set()
    for component in registry["components"].values():
        selectors |= set(component.get("selector_keys", []))

    known = set(IMPLEMENTED) | set(params) | selectors
    unimplemented = _unimplemented_specs(known)

    # Copied, so the citation pass below cannot write a citation back into
    # this module's tables and leak it into a second build.
    for table in (IMPLEMENTED, TIGHTEN, unimplemented):
        for name, spec in table.items():
            params[name] = copy.deepcopy(spec)

    # SELECTABLE, not evidenced -- see the spec's own comment.  Assigned
    # here because the merge above deep-copies this module's spec table
    # over the parameters dict, so an earlier write would be discarded.
    # The counts come from the land-surface schemes' own modules, gathered
    # where their option rows were written.
    params["num_soil_layers"]["enum"] = sorted(selectable_soil_counts)

    params["aercu_opt"] = {"type": "integer", "default": 0, "enum": [0],
                           "per_domain": True, "consuming_read": "woof/core/morrison_aero.py"}
    for name, (_, reason) in UNIMPLEMENTED_LEDGER.items():
        prior = params.get(name)
        if not isinstance(prior, dict) or "type" not in prior:
            raise KeyError(
                f"Lane K ledger row {name!r} has no typed registry parameter")
        params[name] = {
            "type": prior["type"],
            "implemented": False,
            "unimplemented_reason": reason,
        }
    # Retire MSKF handoff refusals before checking the retained ledger.
    _multiscale_parameters(registry)
    remaining_false = {
        name for name, spec in params.items()
        if isinstance(spec, dict) and spec.get("implemented") is False
    }
    if remaining_false != set(UNIMPLEMENTED_LEDGER):
        raise AssertionError(
            "Lane K ledger no longer covers exactly every implemented=false "
            f"parameter; missing={sorted(remaining_false - set(UNIMPLEMENTED_LEDGER))}, "
            f"retired={sorted(set(UNIMPLEMENTED_LEDGER) - remaining_false)}")

    uncited, replaced = [], []
    for name in OWNED:
        spec = params.get(name)
        if not isinstance(spec, dict):
            continue
        before = prior_citations.get(name)
        citation = find_consuming_read(name, before or spec.get("consuming_read"))
        if citation is None:
            uncited.append(name)
            spec.pop("consuming_read", None)
            continue
        if before and citation != before:
            replaced.append(f"{name}: {before} -> {citation}")
        spec["consuming_read"] = citation
    if uncited:
        print("NO CONSUMING READ FOUND (claim withdrawn):", uncited)
    if replaced:
        print("CITATION NO LONGER RESOLVES (repointed):")
        for line in replaced:
            print("  " + line)

    for template_id, columns in NEST_COLUMNS.items():
        registry["templates"][template_id]["per_domain_overrides"] = [
            dict(column) for column in columns]

    registry["authority"]["parameter_declaration"] = (
        "parameters declare every knob a GPUWM runtime component reads; "
        "implemented=false publishes a knob GPUWM does not yet honor so the "
        "registry doubles as the porting roadmap, and such a knob can never "
        "be set. Component selector_keys are declared on their component and "
        "are deliberately absent from parameters.")
    registry["authority"]["per_domain_override_semantics"] = (
        "templates.per_domain_overrides is indexed by depth below the tree "
        "root and carries values transcribed from verified runs; nominal_dx_m "
        "is provenance for display and is never a resolved setting.")

    # WRF v4.6.1 module_pbl_driver.F:873-878 derives FLAG_QS from Registry
    # F_QS.  module_bl_mynn_wrapper.F:452-475 converts the real snow field to
    # specific units, and module_bl_mynn.F:1104-1106 supplies it to
    # mym_condensation.  Radiation then consumes the previous interval's
    # carried MYNN clouds at module_radiation_driver.F:1403-1429.
    mynn = registry["components"]["pbl"]["options"]["mynn"]
    # The exact-driver and cycling series replace the older rounded-leaf
    # warnings. Keep their component evidence and corrected-referee scope
    # distinct from a matched full WRF forecast.
    mynn["warnings"] = [
        "UNVERIFIED against a WRF forecast. MYNN PBL runs through the exact "
        "production driver, with phim/phih evaluated on the device. Cold-column "
        "fixtures are bitwise WRF v4.6.1 on every oracle in "
        "tests/test_mynn_wrf461_exact_gpu.py; the stock and gsd_41 family "
        "drivers and restart controls are described by the selected MYNN "
        "review receipts. This is component qualification, not a matched "
        "full WRF forecast or observation skill.",
        "Cycled stock MYNN carries QKE, QC_BL and CLDFRA_BL by default. "
        "Its column comparison uses the three-assignment repaired referee: "
        "only the upstream assignments that erase these held arrays are "
        "removed. Cold starts retain their stock behavior. "
        "tests/test_mynn_wrf461_cycled_carry_gpu.py and "
        "tests/test_mynn_restart_live_gpu.py retain the raw-word and "
        "continuation checks; other columns and compiler platforms are "
        "not qualified by this declaration.",
        "An admitted option is limited by the actual MYNN configuration "
        "and consumer tables. Unsupported closure, cloud-PDF and diagnostic "
        "values still refuse before the driver; cycling and restart are "
        "not blanket refusals.",
    ]
    from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
    # ``flag_qs_*_microphysics_selectors`` must partition EVERY implemented
    # microphysics selector -- tests/test_mynn_pbl.py asserts set equality
    # against the registry's own options -- so a new scheme lands here in the
    # same pass that registers it.  mp_physics=28's thompsonaero package
    # carries qs (Registry.EM_COMMON:3036), so F_QS is true for it, and
    # mp_physics=16's wdm6scheme package carries qs at :3031, so F_QS is
    # true for WDM6 too.
    from woof.microphysics_schemes import PORTED_WRF_SNOW_SELECTORS
    from woof.core.mynn_pbl_metadata import MYNN_SNOW_MICROPHYSICS
    _mynn_snow_selectors = set(MYNN_SNOW_MICROPHYSICS) | set(PORTED_WRF_SNOW_SELECTORS)
    _implemented_microphysics_selectors = {
        int(row["selectors"]["mp_physics"])
        for row in registry["components"]["microphysics"]["options"].values()
        if row.get("implemented") is True and "mp_physics" in row["selectors"]}
    mynn["extensions"]["supplied_moisture_species"] = {
        "supplied": ["qv", "qc", "qi", "qs"],
        "withheld": ["qnc", "qni", "qnwfa", "qnifa", "qnbca", "o3"],
        # Named schemes (woof/microphysics_schemes.py) join from their own
        # rows: a scheme whose mass species carry qs is F_QS true.
        "flag_qs_true_microphysics_selectors": sorted(_mynn_snow_selectors),
        "flag_qs_false_microphysics_selectors": sorted(
            _implemented_microphysics_selectors - _mynn_snow_selectors),
        "wrf_flag_source": (
            "phys/module_pbl_driver.F:873-878 derives flag_qs from F_QS; "
            "Registry.EM_COMMON declares qs for mp_physics 6, 8, 9, 10, 14, 16, 26, "
            "18, 28 and 38, Registry/registry.sbm:318 declares it for 30, and "
            "does NOT declare it for 50 (P3 has one ice "
            "category, :3038); a named scheme (woof/microphysics_schemes.py, "
            "when enabled) is true when its own mass species carry qs"),
        "gpuwm_runtime_source": (
            "woof/core/mynn_pbl_runtime.py::MYNN_SNOW_MICROPHYSICS is the "
            "shipped set this list is checked against, selector by selector, "
            "by tests/test_physics_registry.py::"
            "test_the_registry_flag_qs_contract_is_the_one_the_shipped"
            "_runtime_applies. mp_physics=28 was published here before the "
            "runtime honoured it: woof passed flag_qs=False for 28, so "
            "phys/module_bl_mynn.F:734/:876 substituted sqs = 0 and MYNN "
            "never saw snow under the one Thompson variant whose Registry "
            "package declares it. MEASURED on the committed WRF MYNN driver "
            "oracle's snow_anvil column (max sqs 4.08e-05): withholding snow "
            "drove qi_bl from 5.4863e-07 to exactly 0 and moved qc_bl, "
            "cldfra_bl, rqvblten, rthblten and exch_h with it. MEASURED "
            "again as a forecast -- mp_physics=28 + MYNN + SFCLAY + Noah, "
            "8x6x50 at dx = 3 km, 20 steps of dt = 12 s, two runs identical "
            "but for FLAG_QS, snow 4.0e-05 kg/kg seeded at 0-based levels "
            "20-33: max relative difference qr 2.037e-02, qv 7.226e-03, "
            "qc 4.035e-03, qke 5.593e-02, exch_h 8.380e-01, nc 6.915e-04, "
            "nwfa 1.142e-04, with qs and qi bitwise identical because "
            "neither WRF nor woof applies a snow PBL tendency "
            "(phys/module_bl_mynn.F:1240-1242). The same experiment with the "
            "snow seeded in the WARM boundary layer instead is BITWISE "
            "identical, because MYNN's condensation only takes snow into the "
            "ice branch where the liquid fraction is below 1 -- the flag "
            "matters exactly where snow exists."),
        "wrf_live_consumer": (
            "phys/module_bl_mynn.F:1104-1106 passes real sqs to "
            "mym_condensation when FLAG_QS is true; mynn_tendencies still "
            "receives kzero at :1240-1242, matching WRF"),
        "aerosol_number_mixing_note": (
            "MYNN scalar plume transport is selected by bl_mynn_mixscalars=1. "
            "WRF post-PBL local diffusion is selected by scalar_pblmix=1 "
            "(phys/module_pbl_driver.F:2251,2641-2844). Both mix the "
            "mp_physics=28 nc/ni/nwfa/nifa family, default to 0 and, when "
            "they mix, require bldt=0; bl_mynn_mixscalars=1 is WRF's no-op "
            "under a microphysics with no number species and is refused by "
            "name under one WRF mixes only in part "
            "(config.MYNN_QN_FLAG_SPECIES). WRF's check_a_mundo disables "
            "scalar_pblmix when bl_mynn_mixscalars=1, so that joint request "
            "is refused."),
    }
    mynn["extensions"]["radiation_cloud_merge"] = {
        "activation": "bl_pbl_physics=5 and icloud_bl>0",
        "ordering": (
            "radiation precedes PBL and consumes the previous interval's "
            "carried QC_BL/QI_BL/CLDFRA_BL"),
        "wrf_source": "phys/module_radiation_driver.F:1403-1429",
        "implementations": ["dudhia-shortwave", "rte-rrtmgp", "rrtmg-legacy"],
    }
    mynn["warnings"] = [
        warning for warning in mynn["warnings"]
        if not warning.startswith("DEVIATION from WRF, affecting MYNN PBL")
    ]
    template = registry["templates"][
        "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1"]
    template["warnings"] = [
        warning.replace(
            "the CUDA-versus-CPU ULP spread and the withheld snow species",
            "the CUDA-versus-CPU ULP spread; the WRF FLAG_QS snow path and "
            "previous-interval radiation cloud merge are coupled")
        for warning in template["warnings"]
    ]
    registry["authority"]["real_source_moisture_contract"] = (
        "runner_routes.<runner>.requires_moist_real_initialization declares "
        "that every source on the route enters woof.ingest.real and therefore "
        "requires a moist state even when microphysics is off. The plan must "
        "then set moist=true explicitly; a component default cannot make that "
        "source-preparation decision for the user.")
    nssl2_id = (
        "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-"
        "wrf-comparison-candidate-v1")
    nssl2_legacy_id = (
        "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-"
        "wrf-comparison-candidate-v1")
    nssl2_legacy = copy.deepcopy(registry["templates"][nssl2_id])
    nssl2_legacy["label"] = (
        "NSSL-2 + YSU + classic MM5 + Noah + KF + legacy RRTMG")
    nssl2_legacy["maturity"] = "wrf-matched-run-candidate"
    nssl2_legacy["parameters"]["wrf_rrtmg_compatibility"] = (
        "wrf-rrtmg-4-4-legacy-v1")
    nssl2_legacy["parameters"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    nssl2_legacy["warnings"] = [
        "Ratified fixed NSSL-2 plus exact WRF v4.6.1 legacy RRTMG profile; "
        "the NSSL-2 trajectory remains wrf-matched-run-candidate maturity."
    ]
    registry["templates"][nssl2_legacy_id] = nssl2_legacy
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if nssl2_id in declared:
                if nssl2_legacy_id in declared:
                    declared.remove(nssl2_legacy_id)
                declared.insert(declared.index(nssl2_id) + 1, nssl2_legacy_id)

    # The observation battery's registered composition (obs-battery
    # integration wave, lead ruling 2026-08-04): Thompson + YSU + classic
    # MM5 + Noah + cumulus off + the exact WRF v4.6.1 legacy RRTMG.  Built
    # from the Thompson validation template on the nssl2_legacy idiom: the
    # radiation component is the resolved 4/4 pair ("rte-rrtmgp" is the
    # registry's spelling of that pair; the ENGINE is named by
    # ra_rrtmg_variant in parameters, exactly as the NSSL-2 legacy row
    # does).  Parameters and the single per-domain row are TRANSCRIBED
    # from configs/battery/shape_3km_thompson_rrtmg_legacy.toml as
    # registered -- notably radt 12.0 at dx 3000 m, where the KF template
    # family's ladder carries radt 3.0 at 3 km.  That divergence is
    # deliberate: this row names what the battery runs, not the ladder.
    thompson_validation_id = "thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1"
    thompson_legacy_id = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
    thompson_legacy = copy.deepcopy(
        registry["templates"][thompson_validation_id])
    thompson_legacy["components"]["radiation"] = "rte-rrtmgp"
    thompson_legacy["label"] = (
        "Thompson + YSU + classic MM5 + Noah + cumulus off + legacy RRTMG")
    thompson_legacy["maturity"] = "wrf-matched-run-candidate"
    thompson_legacy["parameters"]["diff_6th_factor"] = 0.12
    thompson_legacy["parameters"]["radt"] = 12.0
    thompson_legacy["parameters"]["wrf_rrtmg_compatibility"] = (
        "wrf-rrtmg-4-4-legacy-v1")
    thompson_legacy["parameters"]["ra_rrtmg_variant"] = "rrtmg_legacy"
    thompson_legacy["per_domain_overrides"] = [
        {
            "diff_6th_factor": 0.12,
            "epssm": 0.5,
            "nominal_dx_m": 3000.0,
            "radt": 12.0,
        },
    ]
    thompson_legacy["warnings"] = [
        "Composition candidate: every component is individually verified "
        "(Thompson mp8 wrf-matched-run; the legacy RRTMG engine is the "
        "certified WRF v4.6.1 port) but no receipt covers the composed "
        "suite.  The upgrade payer is named: the composition's first "
        "stock-WRF-paired t0/case receipt (the observation battery's "
        "shakedown case) is what moves this label.",
        "radt 12.0 at dx 3000 m is transcribed from the battery's "
        "registered configuration and deliberately diverges from the KF "
        "template family's per-domain ladder (radt 3.0 at 3 km); it is "
        "not an oversight.",
    ]
    registry["templates"][thompson_legacy_id] = thompson_legacy
    # HRRR-only registration, on the Kessler precedent: the battery runs
    # HRRR, and no other source inherits evidence from that run.  The
    # prepared-single-domain route's per-source lists are the runner's
    # own VERIFICATION-EVIDENCE metadata (its drift check compares them
    # to _SOURCE_PHYSICS_PROFILES), and this composition has no receipt
    # on gfs/era5/20crv3 -- so it is deliberately absent there.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if thompson_legacy_id in declared:
                declared.remove(thompson_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(
            declared.index(thompson_validation_id) + 1, thompson_legacy_id)

    # The gray-zone sibling of the row above: the SAME composition with
    # Shin-Hong 2015 in place of YSU, which is the single edge the
    # divergence ledger's L3 entry moves (woof/physics_mode.py).  It is
    # registered because a physics-fidelity arm that selects L3 resolves
    # to exactly this suite, and an unregistered suite has no root
    # preparation -- so the ledger entry that already carries the
    # strongest scheme-level evidence in the tree had no run route at
    # all.  Built from the sibling by moving ONE component, so a paired
    # run of the two isolates the closure and nothing else; the surface
    # layer stays classic MM5 because WRF v4.6.1's own SHINHONGSCHEME arm
    # (phys/module_physics_init.F:3702-3704) requires isfc=1 exactly as
    # YSU does.
    shinhong_legacy_id = "thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1"
    shinhong_legacy = copy.deepcopy(thompson_legacy)
    shinhong_legacy["components"]["pbl"] = "shinhong"
    shinhong_legacy["label"] = (
        "Thompson + Shin-Hong + classic MM5 + Noah + cumulus off + "
        "legacy RRTMG")
    shinhong_legacy["maturity"] = "wrf-matched-run-candidate"
    shinhong_legacy["warnings"] = [
        "Composition candidate: every component is individually verified "
        "(Thompson mp8 wrf-matched-run; the legacy RRTMG engine is the "
        "certified WRF v4.6.1 port; Shin-Hong is measured bitwise against "
        "the byte-frozen WRF v4.6.1 module on both halves of the port, "
        "max ULP 0 on the float32 CPU authority and 0 ULP on the CUDA "
        "heat tendency) but no receipt covers the composed suite.  The "
        "upgrade payer is named: this composition's first stock-WRF-"
        "paired t0/case receipt -- the first paired case run of the "
        "gray-zone arm -- is what moves this label.",
        "This template differs from "
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 in exactly ONE "
        "component, the PBL closure, so the pair is a controlled "
        "gray-zone comparison rather than two independent suites.  Every "
        "other parameter, including the per-domain row, is transcribed "
        "from that template.",
        "Shin-Hong carries the component-level warnings of its option "
        "(components.pbl.options.shinhong): the entrainment-flux guard "
        "where WRF reads one element past its array, WRF's own 0/0 NaN "
        "column reproduced rather than repaired, and the "
        "subnormal-tendency flush on CuPy's -ftz=true compile route.  "
        "Selecting this template selects those.",
    ]
    registry["templates"][shinhong_legacy_id] = shinhong_legacy
    # HRRR-only, on the same Kessler rule as its sibling: the arm that
    # selects this composition runs the HRRR route, and no other source
    # inherits evidence from that run.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if shinhong_legacy_id in declared:
                declared.remove(shinhong_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(
            declared.index(thompson_legacy_id) + 1, shinhong_legacy_id)

    # P3 one-category, on the thompson_legacy idiom: the SAME composition
    # with one selector moved (microphysics thompson-mp8 -> p3-mp50), so a
    # paired run of the two isolates the scheme.  Registered because the
    # native-HRRR doors admit mp_physics=50 now
    # (woof/hrrr_route_inputs.SUPPORTED_MICROPHYSICS derives from the
    # ported set) and a scheme no template selects has no root
    # preparation -- the ship-only-what-users-can-reach rule.  Legacy
    # RRTMG is a REQUIREMENT for this scheme's 4/4 pair, not a taste:
    # p3-mp50's own refused_when blocks the RTE+RRTMGP variant (WRF sets
    # has_reqs=0 for P3, no snow radius exists to hand it), so this
    # template pins ra_rrtmg_variant=rrtmg_legacy exactly as its sibling
    # does and is the one full-radiation P3 composition the registry can
    # admit.  moist_cq is pinned True here because the option row does
    # not carry it (unlike thompson-mp8/nssl2-mp18): P3's four moist
    # species feed the same device cq path, and the shipped runtime
    # switches (woof/physics_compat.py) pin the identical value --
    # tests/test_physics_registry.py holds the two equal.
    p3_legacy_id = "p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1"
    p3_legacy = copy.deepcopy(thompson_legacy)
    p3_legacy["components"]["microphysics"] = "p3-mp50"
    p3_legacy["label"] = (
        "P3 one-category + YSU + classic MM5 + Noah + cumulus off + "
        "legacy RRTMG")
    # The composition ceiling: p3-mp50 itself is implemented-unverified
    # (measured against the unmodified P3 v4.5.2 oracle, no composed
    # forecast receipt), so the template cannot rank above it.
    p3_legacy["maturity"] = "implemented-unverified"
    p3_legacy["parameters"]["moist_cq"] = True
    p3_legacy["warnings"] = [
        "Composition candidate: the P3 port is measured against WRF's own "
        "Fortran oracle (see components.microphysics.options.p3-mp50) and "
        "the legacy RRTMG engine is the certified WRF v4.6.1 port, but no "
        "receipt covers the composed suite.  The upgrade payer is named: "
        "the composition's first stock-WRF-paired t0/case receipt is what "
        "moves this label.",
        "This template differs from thompson-mp8-ysu-mm5-noah-rrtmg-"
        "legacy-v1 in exactly ONE component, the microphysics, so the "
        "pair is a controlled scheme comparison; every other parameter, "
        "including the per-domain row, is transcribed from that template.",
        "P3 has ONE ice category: this suite's history carries QIR/QIB "
        "and no QSNOW/QGRAUP/GRAUPELNC fields, exactly as stock WRF's "
        "mp=50 Registry package does.  REFL_10CM is not stashed for "
        "mp=50 on the native-HRRR runner (its native-reflectivity set "
        "does not include 50), so history frames omit it.",
    ]
    registry["templates"][p3_legacy_id] = p3_legacy
    registry["components"]["microphysics"]["options"][
        "p3-mp50"]["reachability"] = {"state": "template"}
    # HRRR-only, on the Kessler rule: the doors that admit mp=50 are the
    # native-HRRR routes, and no other source inherits evidence from them.
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if p3_legacy_id in declared:
                declared.remove(p3_legacy_id)
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast"):
        declared = registry["runner_routes"][route_id][
            "source_template_ids"]["hrrr"]
        declared.insert(declared.index(nssl2_legacy_id) + 1, p3_legacy_id)

    # The radiation-arm siblings of the rows above.  Minted here, after
    # every base they copy exists and before the route declarations are
    # audited, from the table beside _SUITELESS_TEMPLATES.
    _radiation_arm_siblings(registry)
    # And the microphysics-arm siblings, at the same point for the same
    # reason: their bases exist and are declared on their final routes.
    _microphysics_arm_siblings(registry)
    _mynn_source_version_templates(registry)
    _thompson_fork_source_template(registry)
    _monthly_surface_template(registry)
    _solar_monthly_surface_template(registry)

    # Owner-ratified declaration: the GFS runner has always advertised this
    # profile and retains the existing Noah-MP route acknowledgement.
    gfs_route = registry["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    gfs_expert = gfs_route.setdefault(
        "expert_template_ids", {}).setdefault("gfs", [])
    noahmp_id = "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1"
    if noahmp_id not in gfs_expert:
        gfs_expert.append(noahmp_id)
    # EVERY SOURCE THIS ROUTE SUPPORTS, not gfs alone.  A route that
    # declares any expert list is exhaustive, so the gfs-only declaration
    # made the three Noah-MP templates undeclared on the other seventeen
    # sources: the launcher offered none of them and a plan naming one was
    # refused as off-route instead of raising the acknowledgement advisory
    # the option exists for.  What gates these templates is the route's
    # expert acknowledgement and Noah-MP's own evidence warnings, and both
    # are source-independent -- the scheme is implemented-unverified on
    # every source, not unverified here and verified there.  Derived from
    # the route's own source_ids so a new source inherits the offer
    # instead of quietly losing it.  "mapped" is excluded because it names
    # no model and declares no per-source list at all.
    # The offer is written ONCE, under the route-wide key
    # (woof.physics_registry.EXPERT_TEMPLATES_ANY_SOURCE), which every
    # reader resolves through expert_template_ids_for_source.  It used to
    # be copied under one key per source id, which put seventeen dataset
    # names into a generic table (tests/test_case_token_leakage.py); the
    # per-source copies a tracked registry still carries are removed here
    # so the route's declaration has one shape.
    from woof.physics_registry import EXPERT_TEMPLATES_ANY_SOURCE

    for source_id in gfs_route.get("source_ids", []):
        if source_id != "gfs":
            gfs_route["expert_template_ids"].pop(source_id, None)
    route_wide = gfs_route["expert_template_ids"].setdefault(
        EXPERT_TEMPLATES_ANY_SOURCE, [])
    for template_id in gfs_expert:
        if template_id not in route_wide:
            route_wide.append(template_id)

    # Every graph setting constraint names what it prevents.  The
    # evaluator refuses a reasonless row outright, so these two are
    # written here rather than carried through from the file: they are the
    # only voice on a question a per-domain config cannot see (which
    # domain is the root), and "requires spec_exp=0.0, got 0.33" told a
    # user nothing about the sponge that would be applied.
    _GRAPH_CONSTRAINT_REASONS = {
        ("non-root", "spec_exp"): (
            "the nested lateral-boundary branch carries no exponential "
            "sponge term (WRF dyn_em/module_bc_em.F:1297-1341 applies "
            "spongeweight on the SPECIFIED branch only), while woof's "
            "nested Davies weights do read spec_exp -- so a nonzero value "
            "on a child applies a sponge the branch it transliterates does "
            "not have. The root of this tree may set spec_exp freely; it "
            "is externally forced and takes the specified branch."),
        ("root", "nest_microphysics_transition"): (
            "this setting names the closure used to translate condensate "
            "ACROSS a parent-to-child edge, and the root has no parent "
            "edge to translate over. Select it on the child domain whose "
            "microphysics differs from its parent, where the transition "
            "matrix owns it."),
    }
    for route in registry["runner_routes"].values():
        for constraint in route.get("graph_setting_constraints", []):
            reason = _GRAPH_CONSTRAINT_REASONS.get(
                (constraint.get("scope"), constraint.get("setting_key")))
            if reason is not None:
                constraint["reason"] = reason

    registry["authority"][
        "unnamed_tree_outside_reachability_acknowledgement_id"
    ] = "expert-tuple-v1"
    registry["authority"]["unnamed_tree_reachability_contract"] = (
        "An unnamed domain tree resolves each domain's complete component "
        "tuple against the union of registry-declared experiment-per-domain "
        "template, whole-component override, option-scoped component "
        "override, and expert-template reachability. "
        "Normal tuples proceed unchanged; expert-template tuples require the "
        "route's expert acknowledgement; a tuple outside that union requires "
        "the authority-level acknowledgement id. This is a tuple capability "
        "check and never uses source identity.")
    registry["authority"]["v1_launch_behavior_changed"] = True
    for route_id in (
            "tools.hrrr_single_domain_benchmark",
            "tools.prepared_domain_tree_forecast",
            "tools.prepared_single_domain_forecast"):
        registry["runner_routes"][route_id][
            "requires_moist_real_initialization"] = True
    mp_options = (
        (1, "kessler-mp1"),
        (2, "lin-mp2"),
        (6, "wsm6-mp6"),
        (8, "thompson-mp8"),
        (9, MP9_OPTION_ID),
        (10, "morrison-mp10"),
        (18, "nssl2-mp18"),
        # mp=50 joined when its rime-pair mixed-edge closure was ratified
        # into microphysics_transition.PORTED_MP_PHYSICS; mp=16 and mp=28
        # joined with theirs.
        (50, "p3-mp50"),
        (16, WDM6_OPTION_ID),
        (28, MP28_OPTION_ID),
        (24, WSM7_OPTION_ID),
        (13, "sbu-ylin-mp13"),
    )
    cross_options = []
    for parent_mp, parent_option in mp_options:
        for child_mp, child_option in mp_options:
            if parent_mp == child_mp:
                continue
            # The p3<->mp9 skip that stood here retired with its defect.
            # It existed because mp=9 was in neither PORTED_MP_PHYSICS nor
            # the named-exclusion tuple, so every published mp9 edge was an
            # over-claim the runtime refused, and the skip only kept the
            # newest pair out of a table that already carried ten of them
            # (audit R-003).  mp=9 is now a ported mixed edge with its own
            # closure, and the assertion below launches every published row
            # through the resolver, so an over-claim fails the BUILD.
            ratified = (parent_mp, child_mp) == (8, 18)
            rule = {
                "parent_option_id": parent_option,
                "child_option_id": child_option,
                "required_parent_settings": {
                    "moist": True,
                    "moist_cq": True,
                },
                "required_child_settings": {
                    "moist": True,
                    "moist_cq": True,
                    "nest_microphysics_transition": (
                        "mp8-to-mp18-mass-diagnosed-v1"
                        if ratified else "mp-edge-mass-diagnosed-v1"
                    ),
                },
                "status": "ratified" if ratified else "experimental",
            }
            if not ratified:
                rule["maturity"] = "experimental-runtime"
            cross_options.append(rule)
    # EVERY published cross edge must resolve through the runtime resolver
    # the run will use.  Two authorities said opposite things about ten
    # mp=9 edges -- the registry admitted them at plan review and
    # woof.core.microphysics_transition refused them at nest construction
    # -- and nothing compared the two (audit R-003).  Resolving them here
    # makes the registry's claim and the runtime's answer one statement,
    # checked when the table is written rather than when a nest is built.
    from woof.core.microphysics_transition import (
        resolve_microphysics_transition)
    from types import SimpleNamespace

    for rule in cross_options:
        parent = next(mp for mp, option in mp_options
                      if option == rule["parent_option_id"])
        child = next(mp for mp, option in mp_options
                     if option == rule["child_option_id"])
        # The namespaces carry exactly the settings the rule REQUIRES, so
        # the check is "a run that satisfies this published row resolves",
        # not "some run resolves".
        parent_cfg = SimpleNamespace(
            mp_physics=parent, **rule["required_parent_settings"])
        child_cfg = SimpleNamespace(
            mp_physics=child, **rule["required_child_settings"])
        try:
            resolve_microphysics_transition(parent_cfg, child_cfg)
        except Exception as error:
            raise RuntimeError(
                f"the registry publishes the microphysics edge mp={parent} "
                f"-> mp={child} and woof.core.microphysics_transition "
                f"refuses it at nest construction: {error}") from None

    registry["transitions"]["microphysics-one-way-v1"] = {
        "component_id": "microphysics",
        "cross_options": cross_options,
        "same_option": {
            "allowed": True,
            "required_child_settings": {
                "nest_microphysics_transition": "same-scheme-only",
            },
        },
        "topology_id": "one-way-nested-v1",
    }
    registry["parameters"]["nest_microphysics_transition"]["enum"] = [
        "same-scheme-only",
        "mp8-to-mp18-mass-diagnosed-v1",
        "mp-edge-mass-diagnosed-v1",
    ]
    # After every option, template and route the passes above created, and
    # before the maturity/evidence/consumer-row passes below, which walk
    # whatever this one leaves behind.
    for template in registry["templates"].values():
        template["components"]["urban"] = "none"
    # Retire the disabled CQ guard in old registry seeds after every
    # microphysics option is registered. WRF applies calc_cq to every
    # active moist package, including passive vapor. The off option keeps
    # moist=False; dry states bypass the driver.
    for option_id in ("off", "kessler-mp1", "wsm6-mp6"):
        registry["components"]["microphysics"]["options"][option_id][
            "parameters"]["moist_cq"] = True
    _multiscale_parameters(registry)
    _phase2c_route_declarations(registry)
    _temf_pair_routes(registry)
    _every_served_source_declares_a_template_list(registry)
    _phase2c_soil_geometry(registry)
    _phase2c_suiteless_templates(registry)
    _kf_trigger_templates(registry)
    _slab_fixed_template_routes(registry)
    # The retired WSM7 suite (see _wsm7_mp24) leaves every route the seeded
    # registry still declares it on.
    for route in registry["runner_routes"].values():
        for ids in (route.get("source_template_ids") or {}).values():
            while WSM7_RETIRED_TEMPLATE_ID in ids:
                ids.remove(WSM7_RETIRED_TEMPLATE_ID)
    _no_radiation_name_warnings(registry)
    # Declare the shortwave option before asset resolution and evidence passes.
    from copy import deepcopy
    shortwave = deepcopy(registry['components']['radiation']['options']['dudhia-shortwave'])
    shortwave.update(label='Longwave off + WRF old GSFC shortwave', maturity='experimental-runtime', scientific_evidence='none', reachability={'state':'component-override'}, warnings=['WRF 4.7.1 old GSFC shortwave: strict column replay and forecast qualification are separate evidence. Retained strict O2 recapture differs in2/25088 words, RTHRATEN column55 levels28/29. Exact qualification is withheld pending repair; datasets/paper/wrf-ports/spd-recapture/sw2/MANIFEST.md retains packets, command and both hex pairs.'])
    shortwave['selectors']={'ra_lw_physics':0,'ra_sw_physics':2}
    shortwave['constructor_settings']={'icloud':'icloud','center_lat':'center_lat'}
    shortwave['state_fields']=['taucldc','taucldi']
    shortwave['outputs']=['RTHRATENSW','GSW','RSWTOA','TAUCLDC','TAUCLDI']
    shortwave['cadence_parameter']='radt'
    shortwave['wrf_package']='gsfcswscheme ra_sw_physics==2 state:taucldc,taucldi'
    shortwave['asset_requirements']=[{'id':'wrf461-gsfc-shortwave-coefficients','kind':'bundled-package-data','installed_root':'woof/data/wrf_radiation/gsfcsw','assets':[{'filename':'coefficients.f32','bytes':9896,'sha256':'a23e21798425039a82cf799f81c638a9f6bb7b0562c586829f05bdff10abc533'}]}]
    registry['components']['radiation']['options']['gsfc-shortwave']=shortwave
    for route in registry['runner_routes'].values():
        options=route.get('allowed_component_options',{}).get('radiation',[])
        if 'dudhia-shortwave' in options and 'gsfc-shortwave' not in options: options.append('gsfc-shortwave')
    # After every option that owns an asset is registered.
    _ishmael_native_metadata(registry)
    _madwrf_metadata(registry)
    _asset_resolution(registry)
    # Last, so both passes see every surface this builder created above --
    # including the legacy NSSL-2 template and the regenerated nest edges.
    _rename_maturities(registry)
    _evidence_architecture(registry)
    _current_verification_scope(registry)
    _misc_qualified_components(registry)
    # LAST: every option is registered, every constraint written and
    # every maturity renamed, so the consumer rows see the final
    # option set.
    _cam5_components(registry)
    _consumer_rows(registry)
    # After the consumer rows exist, because it reads one of them, and
    # BEFORE the reachability pass below, which prices a way out through
    # the allowed_parameter_keys this one widens.
    _lateral_forcing_remedy_is_reachable(registry)
    # LAST of all: every template and every route declaration above is
    # final, so the easiest path to each option is the one this computes.
    _phase2c_recompute_reachability(registry)
    for option_id, old_line, new_line in (("tke-1.5-order", 1424, 1423),
                                          ("smagorinsky-3d", 1518, 1517)):
        row = registry["components"]["turbulence"]["options"][option_id]
        row["warnings"] = [text.replace(f"woof/core/dycore.py:{old_line}",
                                       f"woof/core/dycore.py:{new_line}")
                           for text in row.get("warnings", [])]
    registry["template_aliases"] = dict(TEMPLATE_ID_ALIASES)
    for option in registry["components"]["cumulus"]["options"].values():
        constraints = option.get("constraints", {})
        for kind, names in (("admitted_setting_values", ("nsas_dx_factor", "sas_pgcon")),
                            ("admitted_setting_values_reasons", ("nsas_dx_factor", "sas_pgcon")),
                            ("forbidden_setting_values", ("bmj_rad_feedback",))):
            values = constraints.get(kind, {})
            for name in names:
                values.pop(name, None)
            if kind in constraints and not values:
                constraints.pop(kind)
        if "constraints" in option and not constraints:
            option.pop("constraints")
    # The two New SAS knobs (golden/ports-cu) are rows this pass owns, so a
    # rebuild from a document that predates them (2.8.9's, which
    # tests/test_ports_rad_regeneration.py rebuilds) creates them.
    registry["parameters"].setdefault("nsas_dx_factor", {
        "consuming_read": "woof/core/column_cumulus.py", "default": 0,
        "enum": [0, 1], "per_domain": True, "type": "integer"})
    registry["parameters"].setdefault("sas_pgcon", {
        "consuming_read": "woof/core/column_cumulus.py", "default": 0.55,
        "per_domain": True, "type": "number"})
    registry["parameters"].setdefault("shalconv", {
        "consuming_read": "woof/core/sas.py", "default": 1,
        "enum": [0, 1], "per_domain": True, "type": "integer",
        "component_id": "cumulus",
        "nondefault_requires_components": {"cumulus": ["gfs-sas"]},
        "warnings": ["ARW extension: CU_SAS defaults to 1; 0 disables its shallow branch."]})
    registry["parameters"]["nsas_dx_factor"]["component_id"] = "cumulus"
    registry["parameters"]["nsas_dx_factor"]["nondefault_requires_components"] = {"cumulus": ["new-sas"]}
    registry["parameters"]["sas_pgcon"]["component_id"] = "cumulus"
    registry["parameters"]["sas_pgcon"].pop("nondefault_requires_components", None)
    registry["parameters"]["sas_pgcon"]["read_when_any"] = [
        {"cu_physics": 94}, {"cu_physics": 96}, {"shcu_physics": 4}]
    registry["parameters"]["sas_pgcon"]["inactive_value_policy"] = "require-default"
    # Bounded evidence carried by verification lanes that updated registry
    # metadata without changing a knob table. Keep it reproducible too.
    warning_rows = json.loads((MODEL / "tools/verified_scheme_warning_rows.json")
                              .read_text(encoding="utf-8"))
    if isinstance(warning_rows, dict):
        warning_rows = [warning_rows]
    for row in warning_rows:
        registry["components"][row["component"]]["options"][row["option"]][
            "warnings"] = row["warnings"]
    # The selected default YSU generation owns the runtime guard. Historical snapshots
    # carried both its full account and a shorter duplicate citing the
    # removed strict copy. Keep the complete account once and resolve the
    # surviving source anchor each time the registry is regenerated.
    warnings = registry["components"]["pbl"]["options"]["ysu"]["warnings"]
    guards = [w for w in warnings if "bl_ysu.F90:846" in w]
    if not guards:
        raise ValueError("YSU warning snapshot is missing the top-down bounds account")
    guard = max(guards, key=len)
    from woof.ysu_generations import YSU_DEFAULT_GENERATION, ysu_kernel_selection
    ysu_module, _ = ysu_kernel_selection(YSU_DEFAULT_GENERATION)
    ysu_path = f"kernels/{ysu_module}.cu"
    lines = (MODEL / "woof/core" / ysu_path).read_text().splitlines()
    matches = [i for i, line in enumerate(lines, 1)
               if "if (ysu_topdown_pblmix && kpbl < nz" in line]
    if len(matches) != 1:
        raise ValueError(f"YSU top-down guard resolves to {matches}")
    guard = re.sub(r"kernels/ysu(?:_480|_default)?\.cu:\d+",
                   f"{ysu_path}:{matches[0]}", guard)
    guard = guard.replace("kernels/ysu_topo.cuh", "kernels/ysu_topo_default.cuh")
    warnings[:] = [w for w in warnings if w not in guards] + [guard]
    # The Shin-Hong snapshot retains its claim; resolve the live guard
    # rather than publishing its historical source offset. The former YSU
    # zero-coupling citation has retired with that warning and must not
    # rewrite the bounds guard's independent source location.
    for option_id, filename, anchor in (
            ("shinhong", "shinhong.cu", "if (pblflg && kpbl < kte)"),):
        lines = (MODEL / "woof/core/kernels" / filename).read_text().splitlines()
        matches = [i for i, line in enumerate(lines, 1) if anchor in line]
        if len(matches) != 1:
            raise ValueError(f"warning anchor {anchor!r} resolves to {matches}")
        option = registry["components"]["pbl"]["options"][option_id]
        option["warnings"] = [re.sub(
            r"kernels/" + re.escape(filename) + r":\d+",
            f"kernels/{filename}:{matches[0]}", warning)
            for warning in option["warnings"]]
    # The kept real-input captures qualify these selector rows, not every
    # surface-layer cross-product admitted by the surrounding land model.
    for key in ("ua_phys", "sf_surf_irr_scheme", "fractional_seaice",
                "seaice_albedo_opt", "seaice_snowdepth_opt", "seaice_thickness_opt"):
        registry["parameters"][key]["verification"] = {
            "status": "supported-opt-in" if key == "fractional_seaice" else "supported",
            "source": "WRF 4.7.1 f52c197ed39d12e087d02c50f412d90d418f6186",
            "recipe": "tools/sfc_irr_seaice_wrf471/README.md",
            "kept_captures": "datasets/paper/wrf-ports/port-sfc-irr-seaice/",
            "scope": "documented selector values; Noah/Noah-MP with sf_sfclay_physics=1; UA and channel irrigation Noah only",
            "strict_differing_words": 0,
            "forecast": "real-input woof go, default fast numerics, evolved Rust maps viewed",
            "native_undefined": "fractional surface only: 1272 USTM words witnessed by two native runs" if key == "fractional_seaice" else None,
        }
    return registry


#: Every template whose id says "no-radiation" and whose radiation component
#: is ``dudhia-shortwave``.  Recomputed and cross-checked against the built
#: registry below rather than trusted as a list, so a seventh one cannot be
#: added quietly.
_NO_RADIATION_NAMED_TEMPLATE_IDS = (
    "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1",
    "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1",
    "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
    "wsm6-ysu-mm5-noah-no-radiation-v1",
    "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1",
    "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
)

#: The sentence every one of them now carries, verbatim.
NO_RADIATION_NAME_WARNING = (
    "THE NAME IS WRONG AND THE ID IS FROZEN. This template's id says "
    "'no-radiation'; its radiation component is 'dudhia-shortwave', which "
    "is ra_lw_physics 0 with ra_sw_physics 1 -- WRF's Dudhia SHORTWAVE "
    "scheme RUNS, and no longwave scheme does. Downward longwave is "
    "therefore never computed: GLW stays at zero (WRF's own value for a "
    "longwave-free run, phys/module_physics_init.F:1168-1170 and "
    "phys/module_radiation_driver.F:1719-1722) for the entire forecast, "
    "and every land-surface model reads it every step. That makes this a "
    "DAYTIME-ONLY suite: a real window containing local night "
    "refuses to load unless [experiment] declares acknowledgements = "
    "[\"asymmetric-radiation-nocturnal-window-v1\"]. Read the label, not "
    "the id -- the label has always named Dudhia SW. The id is not being "
    "corrected because it is quoted by shipped configs, evidence receipts "
    "and user plans; this warning, the label and docs/public/PHYSICS.md "
    "are the authority on what the option runs."
)


def _no_radiation_name_warnings(registry: dict) -> None:
    """Make six lying template ids say out loud what they actually run.

    Found 2026-08-09 by execution.  Six shipped templates are named
    ``...-no-radiation-...`` and every one of them selects Dudhia
    shortwave with longwave off -- not radiation off, which is a
    DIFFERENT registered option (``radiation: off``, the (0, 0) pair).
    The gap is not theoretical: ``configs/real74_4dom_mynn_norad.toml``
    reached for ``ra_physics = 0`` and explained itself with "the MYNN
    5/5 registry template ... carries ra_physics 0", which the template
    does not, and ``docs/public/STREAMING.md`` handed
    ``wsm6-ysu-mm5-noah-no-radiation-v1`` to users in a copy-paste
    streaming plan while that template's ``warnings`` list did not exist
    and its maturity, ``supported``, is one of
    ``warning_policy.nonwarning_maturities`` -- so selecting it produced
    no warning of any kind at any door.

    Maturity is deliberately NOT touched.  The ladder's axis is
    ``conformance`` and every component of this suite is measured; saying
    ``experimental-runtime`` would be a false claim about port fidelity
    to buy a warning through the maturity channel.  ``warnings`` is the
    channel that exists for "informed user review"
    (``warning_policy.meaning``) and it is emitted for every template
    regardless of maturity
    (``woof/physics_registry.py`` template-warning pass), so it is the
    accurate lever and it is the one used here.

    Appended last and appended (not inserted at 0) on purpose: this
    builder composes ``mynn_ruc`` and ``mynn_noahmp`` by copying another
    template's ``warnings[0]``, so a leading insert would silently
    reassign which sentence those two inherit.
    """

    templates = registry["templates"]
    named = tuple(sorted(
        template_id for template_id in templates
        if "no-radiation" in template_id))
    if named != _NO_RADIATION_NAMED_TEMPLATE_IDS:
        raise SystemExit(
            "the set of templates whose id says 'no-radiation' moved: "
            f"expected {_NO_RADIATION_NAMED_TEMPLATE_IDS}, built {named}. "
            "Update _NO_RADIATION_NAMED_TEMPLATE_IDS deliberately -- a new "
            "member needs this warning, and a departed one must not be "
            "silently dropped from it.")
    for template_id in named:
        template = templates[template_id]
        radiation = template.get("components", {}).get("radiation")
        if radiation != "dudhia-shortwave":
            raise SystemExit(
                f"template {template_id!r} is named 'no-radiation' and "
                f"selects radiation {radiation!r}; this warning states "
                "'dudhia-shortwave' and would now be the lie it exists to "
                "correct")
        warnings = template.setdefault("warnings", [])
        if NO_RADIATION_NAME_WARNING not in warnings:
            warnings.append(NO_RADIATION_NAME_WARNING)


#: Audit R-021: the one route whose immutable templates are the point.
_BENCHMARK_OVERRIDE_REFUSAL = (
    "its published wall-clock and skill numbers are only "
    "comparable against the immutable template they were measured on, so a "
    "component override would silently retire the comparison the route "
    "exists to publish. Run the same suite on "
    "tools.prepared_single_domain_forecast, which declares the same "
    "component overrides as the domain-tree route.")

#: Audit R-021, per component.  Every implemented option a per-domain
#: route's lists exclude is keyed here with the breakage that exclusion
#: prevents.  Keyed per component because a refusal is delivered about ONE
#: component: a route-wide essay pasted into every refusal told a user
#: naming land_surface what is wrong with an analytic radiation scheme, and
#: restated the admitted lists the surrounding sentence already names.
_PER_DOMAIN_EXCLUSION_REASONS = {
    **{("urban", option): (
        "WRF runs one urban selector on every domain (it resets them all to "
        "the innermost domain's value, share/module_check_a_mundo.F:"
        "1063-1077); set sf_urban_physics in [shared], not per domain.")
       for option in ("none", "slucm", "bep", "bep-bem")},
    ("radiation", "analytic-clear-sky"): (
        "analytic-clear-sky is not a forecast radiation scheme: it computes "
        "a clear-sky flux from solar geometry alone and carries no cloud, "
        "aerosol or gas optics, so every cloudy column's heating is wrong by "
        "the whole cloud effect"),
}

#: Where each route sends the value instead, per component.  The exclusion
#: above is the same physical fact on both routes; the way out is not,
#: because only one of them declares expert selectors.
_TREE_ROUTE_OVERRIDE_WAY_OUT = {
    "radiation": (
        "; this route declares ra_lw_physics and ra_sw_physics in "
        "allowed_expert_selector_keys, which is where a dycore or idealized "
        "experiment writes 90/90"),
}
_SINGLE_ROUTE_OVERRIDE_WAY_OUT = {
    "radiation": (
        "; this route declares no expert selectors, so write it in the "
        "hash-bound experiment config, or run the experiment on "
        "tools.prepared_domain_tree_forecast, which declares ra_lw_physics "
        "and ra_sw_physics as expert selectors"),
}

#: Audit R-059: what each single-domain route refuses an expert SETTING
#: for.  Two sentences, not one, because the two routes stopped refusing
#: the same things when R-021 widened one of them: a component override is
#: accepted on tools.prepared_single_domain_forecast and refused on the
#: benchmark route, so a shared "invalidates the seal" sentence no longer
#: distinguished what is refused from what is allowed.
_BENCHMARK_ROUTE_EXPERT_REFUSAL = (
    "sealed-proof benchmark route: this runner replays ONE immutable "
    "template and verifies it against "
    "--proof-sha256/--prepared-content-sha256, and it varies NOTHING -- an "
    "expert setting and a component override are refused alike, because "
    "either one retires the comparison the published numbers were measured "
    "against. State the value in the hash-bound experiment config and run "
    "it on tools.prepared_domain_tree_forecast, whose declaration lists "
    "the parameters its loader accepts.")
_SINGLE_DOMAIN_ROUTE_EXPERT_REFUSAL = (
    "single-domain route: it runs ONE domain, so there is no per-domain "
    "override table for a plan-level expert setting to be carried in "
    "(woof/experiment.py _DOMAIN_RUN_OVERRIDES is the tree loader's), and "
    "every value this runner uses comes from the experiment config it "
    "hash-binds before step 0. State the value THERE, where "
    "woof.config.validate_run_config checks it, or run the plan on "
    "tools.prepared_domain_tree_forecast, whose declaration lists the "
    "parameters its loader accepts. A COMPONENT choice is a different "
    "question and this route admits the ones it declares: the seal covers "
    "the prepared input and its preparation proof, not the suite, and the "
    "2026-07-31 owner ruling removed this runner's profile whitelist so it "
    "resolves any engine-valid suite from that same hash-bound config.")


def _constraints(option: dict) -> dict:
    """``option['constraints']`` as a dict, created when it is JSON null."""

    current = option.get("constraints")
    if not isinstance(current, dict):
        current = {}
        option["constraints"] = current
    return current


def _reachability(option: dict) -> dict:
    """``option['reachability']`` as a dict, created when it is missing.

    A caller here writes only the BLOCKER.  The state is derived by
    :func:`_phase2c_recompute_reachability` from the declared template and
    override lists, so that "how do I select this" and "why can I not" are
    never two hand-set opinions about the same option.
    """

    current = option.get("reachability")
    if not isinstance(current, dict):
        current = {}
        option["reachability"] = current
    return current


#: Knobs the domain-tree loader accepts per domain and this route does NOT
#: offer at plan review, each with the concrete refusal that keeps it out
#: (audit R-059).  Every one is a coupling that lives only in
#: ``woof.config.validate_run_config``: the registry can say "this option
#: requires that setting" and "this option is refused when that setting
#: holds", and none of these is scoped to a component -- they are
#: parameter-to-parameter.  Offering them here without the rule would make
#: plan review call launchable a plan the loader refuses, which is the
#: exact drift class this audit exists to close, so the omission is
#: DECLARED with its reason instead.
#:
#: FOLLOW-UP, named: encode each as a component-scoped ``refused_when``
#: rule and delete its row here.  The gate that will notice is
#: tests/test_authority_agreement.py::
#: test_every_reachable_plan_the_registry_calls_launchable_actually_validates.
_COUPLINGS_NOT_YET_REGISTRY_ROWS = {
    "diff_6th_opt": (
        "diff_6th_opt=1 is non-monotonic and woof.config refuses it with "
        "moist=true, because the unlimited 6th-order fluxes bypass the "
        "positive-definite transport limiter and can drive negative "
        "moisture. The coupling is to moist, not to any component, so no "
        "option row can carry it yet. Set it in the experiment config, "
        "where validate_run_config checks the pair."),
    "clos_choice": (
        "clos_choice and ishallow are Grell-family keys, read only where "
        "cu_physics is 3 or 5; woof.config refuses a nonzero value beside "
        "any other cumulus scheme. Select Grell-Freitas or Grell 3D and "
        "set them in the experiment config."),
    "ishallow": (
        "clos_choice and ishallow are Grell-family keys, read only where "
        "cu_physics is 3 or 5; woof.config refuses a nonzero value beside "
        "any other cumulus scheme. Select Grell-Freitas or Grell 3D and "
        "set them in the experiment config."),
    "cugd_avedx": (
        "cugd_avedx is Grell 3D's subsidence-spreading width, read only "
        "where cu_physics=5; woof.config refuses a value other than 1 "
        "beside any other cumulus scheme. Select Grell 3D and set it in "
        "the experiment config."),
    "o3input": (
        "o3input and use_mp_re are honoured only by the legacy RRTMG "
        "engine; the modern spectrum does not implement the nondefault "
        "ozone and effective-radius operations, and woof.config refuses "
        "the pair. Select ra_rrtmg_variant='rrtmg_legacy' in the "
        "experiment config and set them there."),
    "use_mp_re": (
        "use_mp_re is honoured only by the legacy RRTMG engine; see "
        "o3input. Set it in the experiment config beside "
        "ra_rrtmg_variant='rrtmg_legacy'."),
    "ra_rrtmg_variant": (
        "the engine and the compatibility token must agree, and "
        "woof.config refuses a variant that contradicts the resolved "
        "wrf_rrtmg_compatibility. A per-domain override of one without the "
        "other is exactly that contradiction, so the pair is set together "
        "in the experiment config."),
    "wrf_rrtmg_compatibility": (
        "the compatibility token records a 4/4 substitution and "
        "woof.config refuses it unless the domain resolves the 4/4 pair. "
        "Selecting the radiation option that resolves that pair is how a "
        "domain asks for it; the token follows the option rather than "
        "varying against it."),
}


def _phase2c_route_declarations(registry: dict) -> None:
    """Audit phase 2c: route declarations stop refusing what the runners run.

    Section 3 of the 2026-09-10 physics-glue audit found every refusal in
    this area names no breakage, which under the gate law means it does
    not exist: a blanket "fixed-template runner accepts only its
    immutable template_id" on a route whose own runner applies registry
    governance as a warning and runs (R-021); a land-surface axis simply
    absent from the tree route's option lists, with no blocker text
    anywhere, while every other axis carries a written reason for each
    option it lists AND for the one it excludes (R-022); three options
    the desktop offers and the engine accepts, declared unreachable for
    parity reasons that were retired (R-023); a uniform-base-template
    check that compares template LABELS while the same route declares
    per-domain variation legal for nearly every component (R-058); an
    expert-override refusal on all three routes with no stated reason and
    42 registered parameters stranded behind it (R-059); and a relocating
    nest tree with no declared topology, which is a sequence of static
    one-way trees and changes nothing plan review validates (R-056).

    Everything here is ROWS.  A scheme is rows, and so is a route
    capability: adding one is table work in this builder, never a code
    path in the validator.
    """

    components = registry["components"]
    routes = registry["runner_routes"]
    tree_route = routes["tools.prepared_domain_tree_forecast"]
    single_route = routes["tools.prepared_single_domain_forecast"]
    benchmark_route = routes["tools.hrrr_single_domain_benchmark"]
    land_options = components["land_surface"]["options"]
    surface_options = components["surface_layer"]["options"]
    radiation_options = components["radiation"]["options"]
    parameters = registry["parameters"]

    # -- R-023 ---------------------------------------------------------
    # Selecting land_surface "off" resolved no num_soil_layers at all, so
    # whatever count the plan was carrying reached
    # woof.config.soil_layer_count, which refuses a count the selected
    # scheme does not define.  Scheme 0's geometry is Noah's four
    # (woof/config.py LAND_SURFACE_SOIL_LAYERS._PROVIDERS[0] ->
    # woof.core.noah.NUM_SOIL_LAYERS, which is also what
    # NO_LAND_SURFACE_SOIL_LAYERS resolves to), so the option states its
    # geometry the way every other land surface does.
    _constraints(land_options["off"]).setdefault(
        "required_settings", {})["num_soil_layers"] = 4
    land_options["off"]["warnings"] = [
        "No land-surface scheme runs: TSK, the soil column and the surface "
        "fluxes below the surface layer stay at their initial values for the "
        "whole forecast.  Every registered source supplies a soil state, so "
        "this is a deliberate degradation -- an idealized or LES composition "
        "-- rather than a saving."
    ]
    surface_options["off"]["warnings"] = [
        "No surface layer runs: nothing computes a friction velocity or the "
        "heat and moisture fluxes, so a land-surface scheme or a PBL closure "
        "selected beside it is refused by woof/core/physics.py -- that "
        "refusal names its own breakage and stands.  What remains legal is a "
        "dry no-physics column."
    ]

    # -- R-059: two couplings that ARE component-scoped, as rows -------
    # cudt is a Kain-Fritsch cadence knob.  Grell-Freitas and New Tiedtke
    # both run on the model step and carry no NCA hold, so woof.config
    # refuses a nonzero cudt beside either -- and each option already
    # PINS cudt_minutes=0 in its own parameter block, which a per-domain
    # override could silently move.  Stating it as a required setting is
    # what makes plan review refuse the override the loader refuses.
    for option_id in ("grell-freitas", "grell-3d", "new-tiedtke"):
        _constraints(components["cumulus"]["options"][option_id]).setdefault(
            "required_settings", {})["cudt_minutes"] = 0.0
    # Every radiation option resolves its spectra through the split
    # ra_lw_physics/ra_sw_physics selectors, and woof.config refuses a
    # plan that also sets the legacy aggregate ra_physics to anything but
    # 0 ("do not mix split and legacy radiation selection").  Each option
    # already carries ra_physics=0 in its parameters; the required setting
    # is what refuses a per-domain override of it.
    for option in components["radiation"]["options"].values():
        if (option.get("parameters") or {}).get("ra_physics") == 0:
            _constraints(option).setdefault(
                "required_settings", {})["ra_physics"] = 0

    # -- R-005: the one per-source refusal, as a row -------------------
    # Microphysics off on a real-source route carried TWO refusals fused
    # into one prose error.  The first is per-source, and it is a row now:
    # native HRRR supplies analyzed QC/QR/QI/QS/QG and MP off cannot
    # faithfully retain it, which is why the run died in
    # woof/ingest/real.py rather than at review.  It is refused before
    # step 0, and a source that gains such an incompatibility is table
    # work.
    #
    # The second is the moist carrier, and it STAYS an error at review.
    # This pass briefly resolved it instead -- plan review defaulted
    # moist=true and warned -- and that was a resolution nothing performs:
    # validate_physics_plan reports, while the RunConfig a runner builds
    # takes moist from the microphysics-off option's own row and from the
    # experiment config, so review would have called launchable a plan
    # woof/ingest/real.py refuses before step 0.  What the attempt got
    # right is that the remedy has to be expressible on the route it is
    # prescribed to: the old text prescribed one two of the three routes
    # could not express.  Each route now declares WHERE its moist value
    # lives, and the refusal reads that declaration out.
    mp_off_constraints = _constraints(components["microphysics"]["options"]["off"])
    # Rebuilt, not appended: this builder transforms the tracked registry in
    # place, so a rule appended on every run would multiply.
    mp_off_constraints["refused_when"] = [
        rule for rule in mp_off_constraints.get("refused_when", [])
        if not isinstance(rule, dict) or "sources" not in rule
    ]
    mp_off_constraints["refused_when"].append({
        "sources": ["hrrr"],
        "reason": (
            "native HRRR supplies analyzed QC/QR/QI/QS/QG, and microphysics "
            "off cannot faithfully retain them: no radiation-only "
            "analyzed-cloud carrier is implemented, so the analyzed "
            "condensate would be dropped at initialization and the forecast "
            "would run clear where the analysis was cloudy. Select a "
            "microphysics scheme, or run microphysics off on a source whose "
            "analysis carries no condensate."),
    })
    registry["authority"]["real_source_moisture_contract"] = (
        "runner_routes.<runner>.requires_moist_real_initialization declares "
        "that every source on the route enters woof.ingest.real and "
        "therefore requires a moist state even when microphysics is off. "
        "Plan review REFUSES such a plan until moist=true is stated, "
        "because nothing between review and the loader rewrites the "
        "microphysics-off option's own moist=false: the refusal names "
        "runner_routes.<runner>.moist_declaration_site, which is where that "
        "route's value lives, so the remedy is expressible on the route it "
        "is prescribed to. An explicit moist=false is refused for the same "
        "reason, naming what a dry column would drop. Per-source "
        "incompatibilities are rows -- "
        "components.<id>.options.<id>.constraints.refused_when entries "
        "carrying a sources clause -- so they are refused at review rather "
        "than inside the preparation.")
    # WHERE each route's moist value lives.  A refusal that prescribes a
    # door the route does not have is the defect this key exists to
    # prevent; it is a declaration, so a new route answers the question by
    # adding a row rather than by teaching the validator about itself.
    for route_id, site in (
            ("tools.prepared_domain_tree_forecast",
             "in the plan's per-domain parameters -- this route declares "
             "moist in allowed_parameter_keys -- or in the hash-bound "
             "experiment config the tree loader reads"),
            ("tools.prepared_single_domain_forecast",
             "in the hash-bound experiment config this route runs from, "
             "whose physics keys carry the same names"),
            ("tools.hrrr_single_domain_benchmark",
             "by selecting one of this route's immutable templates: every "
             "one of them runs a microphysics scheme, so this route has no "
             "microphysics-off composition to carry a moist value"),
    ):
        registry["runner_routes"][route_id]["moist_declaration_site"] = site

    # -- R-022 + R-023: the tree route's option lists -------------------
    # land_surface was the one component axis with NO entry at all: no
    # option list, no blocker text, no physical statement anywhere, while
    # every other axis carries a written reason for each option it lists
    # and for the one it excludes.  RUC is a shipped land surface with a
    # runtime gate, a device-path two-sided gate and three declared ERA5
    # templates; its one named consequence under mp_physics=9 is a
    # fidelity divergence carried as the option's own warning, not a
    # failure.
    #
    # Implemented land options retain their own state and coupling constraints.
    # The former throughput acknowledgement is advisory and cannot exclude one.
    tree_route["allowed_component_options"]["land_surface"] = sorted(
        option_id for option_id, option in land_options.items()
        if option.get("implemented") is True)
    # "off" joins the surface-layer list on the same terms: the engine
    # accepts it, the desktop already offers it, and the combinations that
    # would make it dangerous are refused by woof/core/physics.py with
    # their own named breakage.
    tree_route["allowed_component_options"]["surface_layer"] = [
        "off", "revised-mm5", "classic-mm5", "mynn", "eta-similarity", "temf", "gfs", "qnse", "pleim-xiu", "ideal-scm"]
    # analytic-clear-sky is the third member of R-023's trio, and it is
    # the one this block does NOT add to a per-domain option list.  Its
    # STATE is not hand-set either way: _phase2c_recompute_reachability
    # derives it, and the derivation answers component-override for a
    # reason worth stating, because it is the reason the audit found the
    # old "unreachable" label untrue.  The tree route declares
    # ra_lw_physics and ra_sw_physics in allowed_expert_selector_keys, so
    # writing 90/90 through expert_overrides.selectors reaches this
    # option on that route today.  Calling it unreachable while a
    # declared door opens it is exactly the drift this pass exists to
    # end; the degradation is carried by the warning below instead, which
    # is what a user actually reads.
    #
    # The BLOCKER is rewritten rather than kept, so that if the option
    # ever does become unreachable it names a breakage in ArWen's own
    # terms.  The old text -- "WRF v4.6.1 registers no equivalent
    # package" -- is a PARITY reason, and parity stopped being the
    # referee.
    analytic = radiation_options["analytic-clear-sky"]
    _reachability(analytic)["blocker"] = (
        "The 90/90 proxy is not a forecast radiation scheme: it computes "
        "a clear-sky flux from solar geometry alone and carries no cloud, "
        "aerosol or gas optics, so in any cloudy column the radiative "
        "heating profile is wrong by the entire cloud effect -- surface "
        "shortwave too high by day, longwave cooling too strong at night. "
        "It is a dycore and idealized instrument, which is why no template "
        "selects it and no per-domain option list admits it.")
    analytic["warnings"] = [
        "WOOF-SPECIFIC, NOT A WRF SCHEME, AND NOT A FORECAST PRODUCT. The "
        "90/90 analytic clear-sky proxy carries no cloud, aerosol or gas "
        "optics: every cloudy column's radiative heating is wrong by the "
        "whole cloud effect. WRF v4.6.1 registers no equivalent package "
        "(Registry.EM_COMMON:3107-3125), so a run selecting it also has no "
        "stock-WRF counterpart to be compared against."
    ]

    # -- R-022: the RUC warning sentence the shipped tree contradicts ---
    # woof/ingest/soil_contract.py carries RUC_TARGET_LEVEL_DEPTHS_M (nine
    # levels) with an import-time drift check against
    # woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M, plus RUC_REMAP_POLICIES.
    # The sentence claiming a composition that ships a soil_layer_contract
    # "cannot ask for RUC" was true before that landed and is false now.
    # The GFS withdrawal it also records was root-caused rather than
    # re-gated: a shoreline land column carrying the water soil category
    # reached soilvegin, which has no arm for it, and 0./0. went into
    # MAVAIL. Every ArWen door reconciles that column now
    # (woof/ingest/soil.py door_reconciled_soil_category), and
    # woof/prepared_single_domain_forecast.py records that no route
    # blocker remains and none is enforced.
    #
    # Rewritten IN PLACE rather than dropped: _surface_coupling_warnings
    # above addresses this list by index, so removing a member would move
    # every warning after it.
    ruc = land_options["ruc-lsm"]
    ruc["warnings"] = [
        warning.replace(
            "WHAT IS STILL REFUSED is the DECLARATIVE contract: "
            "woof/ingest/soil_contract.py validate_soil_layer_contract "
            "declares exactly one target, Noah's four layers, so a "
            "composition that ships a soil_layer_contract -- which is the "
            "20crv3/mapped path -- cannot ask for RUC.",
            "THE DECLARATIVE CONTRACT REACHES RUC TOO: "
            "woof/ingest/soil_contract.py carries "
            "RUC_TARGET_LEVEL_DEPTHS_M and RUC_REMAP_POLICIES beside "
            "Noah's four layers, with an import-time drift check against "
            "woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M, so a composition "
            "that ships a soil_layer_contract can target RUC.",
        ).replace(
            "The template is offered through the direct HRRR and ERA5 "
            "preparations; it remains absent from 20crv3/mapped "
            "preparation, and from GFS preparation. The GFS withdrawal is "
            "a v1.1.1 field finding rather than an ingest-contract one: a "
            "GFS-initialised RUC forecast PREPARES cleanly -- proof PASS, "
            "nine soil layers -- and then dies on its first "
            "surface-temperature call with `mavail must be finite` "
            "(woof/core/ruc.py:_horizontal_float_field) having advanced "
            "no model time, so the GFS route's initialised land and soil "
            "state does not reach ruc_cold_start in a condition RUC can "
            "integrate from. Completing that initialisation is a v1.2 "
            "item.",
            "The GFS withdrawal that once stood here is RETIRED, and was "
            "root-caused rather than re-gated: a shoreline land column "
            "carrying the water soil category (SOILTYP 14 under a land "
            "LU_INDEX) reached soilvegin, which has no arm for it, and "
            "0./0. went into MAVAIL. WRF's own real program reconciles "
            "that column at initialization and so does every WOOF door, "
            "through woof/ingest/soil.py door_reconciled_soil_category "
            "(proven both ways by "
            "tests/test_ruc_shoreline_soil_category.py). No route blocker "
            "remains and none is enforced.",
        )
        for warning in ruc.get("warnings", [])
    ]

    # -- R-021: the single-domain route mirrors the tree route ----------
    # The 2026-07-31 owner ruling removed the profile whitelist from this
    # route's runner: woof/prepared_single_domain_forecast.py records the
    # removal, labels its own per-source lists "REPORTED METADATA, NOT A
    # GATE", applies registry governance as a WARNING and runs.  The route
    # DECLARATION was never widened to match, so plan review refused what
    # the runner runs -- and the trigger configuration integrated 59
    # minutes on this exact route with two components the declaration says
    # cannot vary. Mirroring the tree route retains each option's actual
    # field and shared-setting constraints, evaluated by the same authority.
    single_route["allowed_component_overrides"] = list(
        tree_route["allowed_component_overrides"])
    single_route["allowed_component_options"] = {
        component_id: list(option_ids)
        for component_id, option_ids
        in tree_route["allowed_component_options"].items()
    }
    # The benchmark route stays closed, and now says why.  An empty
    # declaration with no reason is a refusal that names no breakage; this
    # is the one route where immutability IS the product.
    benchmark_route["allowed_component_overrides"] = []
    benchmark_route["allowed_component_options"] = {}
    benchmark_route["component_override_refusal_reason"] = (
        _BENCHMARK_OVERRIDE_REFUSAL)

    # The refusal that actually FIRES on the two per-domain routes had no
    # reason at all: only the benchmark route carried one, so the single
    # option the widened lists deliberately exclude came back as "runner
    # route does not allow this component option to vary per domain" and
    # stopped there.  Every option a route excludes is named here with its
    # way out, and the exclusion set is computed rather than trusted, so a
    # list that moves cannot leave this sentence describing the old one.
    excluded = {
        (component_id, option_id)
        for component_id, component in components.items()
        if component_id not in tree_route["allowed_component_overrides"]
        for option_id, option in component["options"].items()
        if option.get("implemented") is True
        and option_id not in tree_route[
            "allowed_component_options"].get(component_id, [])
    }
    if excluded != set(_PER_DOMAIN_EXCLUSION_REASONS):
        raise SystemExit(
            "the per-domain option lists exclude "
            f"{sorted(excluded)}; _PER_DOMAIN_EXCLUSION_REASONS keys "
            f"{sorted(_PER_DOMAIN_EXCLUSION_REASONS)} and nothing else. "
            "Give each new exclusion its own breakage and way out in that "
            "table, or add the option to the list.")
    # One reason per COMPONENT, because one refusal is about one component.
    for route, route_way_out in ((tree_route, _TREE_ROUTE_OVERRIDE_WAY_OUT),
                                 (single_route, _SINGLE_ROUTE_OVERRIDE_WAY_OUT)):
        reasons: dict[str, str] = {}
        for (component_id, _option), reason in sorted(
                _PER_DOMAIN_EXCLUSION_REASONS.items()):
            clause = reason + route_way_out.get(component_id, "")
            if reasons.setdefault(component_id, clause) != clause:
                raise SystemExit(
                    f"component {component_id!r} excludes two implemented "
                    "options with different reasons; one refusal carries one "
                    "component's reason, so merge them into one clause or "
                    "key the table by option as well.")
        route["component_override_refusal_reasons"] = reasons
        # Specific component reasons take precedence. Other components need
        # this route's general declaration, not an inferred benchmark reason.
        # Replace on every build so obsolete wording cannot survive a rebuild.
        route["component_override_refusal_reason"] = (
            component_override_declaration(route))

    # -- R-059: expert overrides ---------------------------------------
    # allowed_parameter_keys and allowed_expert_selector_keys are DERIVED
    # from the loader that actually accepts them, so a parameter the tree
    # loader gains becomes settable at plan review in the same pass.  Hand
    # lists are how 42 registered parameters carrying a named
    # consuming_read came to be settable through no route at all, while
    # the validator's own message pointed at expert_overrides.settings --
    # a door every route nailed shut.
    from woof.experiment import _DOMAIN_RUN_OVERRIDES

    selector_keys = {
        key
        for component in components.values()
        for key in component.get("selector_keys", ())
    }
    # A knob EVERY option of one component pins in ``required_settings`` is
    # that component's own identity spelling, not a free per-domain value: a
    # plan overriding it either restates the option it already selected or
    # contradicts it, and plan review then refuses a configuration
    # woof.config.validate_run_config admits -- two doors disagreeing about
    # one file.  ``ra_physics`` is the live case, the pre-split spelling of
    # the radiation pair, which every radiation option states as 0: the
    # aggregate spelling of a pair resolves to that pair at the capability
    # door (woof.physics_compat._one_radiation_spelling), so no option is
    # selected by the aggregate key.
    # Derived rather than listed, so a second alias is excluded the day it
    # is registered.
    identity_keys: set[str] = set()
    for component in components.values():
        pinned = None
        for option in (component.get("options") or {}).values():
            required = set(
                (option.get("constraints") or {}).get("required_settings")
                or ())
            pinned = required if pinned is None else (pinned & required)
        identity_keys |= set(pinned or ())
    per_domain_loader_keys = frozenset(_DOMAIN_RUN_OVERRIDES)
    tree_route["allowed_parameter_keys"] = sorted(
        name for name, spec in parameters.items()
        if spec.get("implemented") is not False
        and name in per_domain_loader_keys
        and name not in selector_keys
        and name not in identity_keys
        and name not in _COUPLINGS_NOT_YET_REGISTRY_ROWS)
    # Published on the route, so the declaration a door reads carries the
    # reason a loader-accepted knob is not offered here.  A refusal that
    # names no breakage does not exist; neither does an omission.
    tree_route["deferred_parameter_keys"] = {
        name: reason
        for name, reason in sorted(_COUPLINGS_NOT_YET_REGISTRY_ROWS.items())
        if name in per_domain_loader_keys and name in parameters
    }
    for name in sorted(identity_keys & per_domain_loader_keys & set(parameters)):
        component_id = next(
            cid for cid, component in components.items()
            if all(name in ((option.get("constraints") or {}).get(
                       "required_settings") or ())
                   for option in (component.get("options") or {}).values()))
        tree_route["deferred_parameter_keys"][name] = (
            f"every {component_id} option states {name}, so it spells WHICH "
            f"option is selected rather than a value inside one. Select the "
            f"{component_id} option that carries the value you want; "
            f"overriding {name} per domain could only restate or contradict "
            "the option already chosen.")
    tree_route["allowed_expert_selector_keys"] = sorted(
        selector_keys & per_domain_loader_keys)
    # The two sealed routes keep empty lists, and now name what the
    # emptiness protects: the proof they replay.
    for route, refusal in (
            (single_route, _SINGLE_DOMAIN_ROUTE_EXPERT_REFUSAL),
            (benchmark_route, _BENCHMARK_ROUTE_EXPERT_REFUSAL)):
        route["allowed_parameter_keys"] = []
        route["allowed_expert_selector_keys"] = []
        route["expert_override_refusal_reason"] = refusal
    tree_route["expert_override_refusal_reason"] = (
        "the domain-tree loader accepts the parameters and selectors this "
        "route declares; an expert SETTING outside them would reach "
        "woof.config.validate_run_config with no registry row to validate "
        "it against. Register the knob as a parameter -- table work in "
        "tools/build_registry.py -- rather than passing it through.")

    # -- R-058: per-domain declarations, and a value check --------------
    # The uniform-base-template check compared template LABELS, so two
    # templates whose component maps are byte-identical and differ only in
    # a per-domain key were refused for having different names.  The
    # narrow thing it stood in for is real: sf_surface_physics, its soil
    # geometry and the Noah-MP option block are deliberately outside the
    # loader's per-domain table, so ONE land surface runs for a whole
    # tree.  That is now stated per parameter and checked on resolved
    # VALUES.
    for name, spec in parameters.items():
        spec["per_domain"] = name in per_domain_loader_keys
    for component in components.values():
        selectors = {
            key: key in per_domain_loader_keys
            for key in component.get("selector_keys", ())
        }
        if selectors:
            component["per_domain_selectors"] = selectors
    tree_route.pop("template_policy", None)
    registry["authority"]["per_domain_declaration"] = (
        "parameters.<name>.per_domain and components.<id>."
        "per_domain_selectors declare which knobs the domain-tree loader "
        "accepts inside a [[domain]] table (woof/experiment.py "
        "_DOMAIN_RUN_OVERRIDES, which this registry is generated from). "
        "validate_physics_plan refuses a tree whose domains resolve "
        "DIFFERENT values for a knob declared per_domain false, naming both "
        "values and the loader that cannot express them; it no longer "
        "compares template ids, which are labels.")

    # -- R-056: a relocating tree is a sequence of static one-way trees --
    # Nothing plan review validates changes when a nest moves: the
    # topology stays one-way nested, the transition policy stays the
    # microphysics edge policy, and relocation is a runtime schedule.  The
    # route says so rather than leaving a relocating plan with no declared
    # topology at all.  No second topology id is minted -- a receipt naming
    # one would claim a second closure exists.
    tree_route["relocation"] = {
        "supported": True,
        "topology_id": "one-way-nested-v1",
        "declaration": (
            "A relocating child is a SEQUENCE of static one-way nested "
            "trees: at every instant the tree validated is the one this "
            "topology id names, and relocation moves the child's parent "
            "window between instants. Plan review validates the instant; "
            "the schedule is runtime state (woof/core/nest_spawn.py, and "
            "the follow/target itineraries in woof/companion_domains.py)."),
        "unchanged_by_relocation": [
            "topology_ids",
            "transition_policy_id",
            "graph_setting_constraints",
        ],
    }


def _phase2c_soil_geometry(registry: dict) -> None:
    """Audit R-038: one soil geometry, one authority.

    Two authorities described the same geometry and disagreed.  This
    registry's ``num_soil_layers`` enum was ``[4, 9]`` and is enforced as a
    HARD plan error, while ``woof/config.py``'s ``soil_layer_count``,
    built from ``woof.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS``
    (which is itself tabulated from ``woof.ingest.ruc_soil``), accepts six
    and a six-level forecast completes with ``soil_layers_stag = 6`` in its
    wrfout.  The registry's stated reason for the refusal was "a schema
    enum entry would advertise a validated geometry" -- an EVIDENCE
    POLICY, not a physical or numerical incompatibility, so under the gate
    law it is not a refusal at all.

    The enum is now DERIVED from the module that owns the geometry, so a
    count the soil column gains is table work here rather than a second
    hand-typed opinion.  What six actually lacks -- a WRF FORECAST oracle,
    every lsmruc/sfctmp/soilmoist/snowtemp fixture in the tree being
    nine-level -- is said in the warning a user reads, beside the receipt
    field that carries it, which is what an absence of evidence is worth.

    DELIBERATELY NOT DONE HERE, and handed back as a named follow-up: the
    ``ruc-lsm-6level`` OPTION row the audit note proposes.  Land-surface
    options are keyed by ``sf_surface_physics`` throughout the authority
    lane that landed after the audit was written -- ``implemented_selector_
    values``, ``consumer_rows_by_selector`` and ``_option_for_selectors``
    all index by selector value -- so a second option carrying selector 3
    would silently shadow the first in every consumer table derived from
    those.  Admitting six-level RUC as a named option needs an inventory
    keyed by (selector, geometry), which is a design change rather than
    table work, and it rides on the owner decision the audit files as its
    open question 5.  Six-level RUC keeps the door it has: the hash-bound
    experiment config, which ``woof.config.soil_layer_count`` admits and
    ``woof/core/ruc_contract.py`` sizes.
    """

    from woof.core.ruc_contract import WRF_SUPPORTED_NUM_SOIL_LAYERS

    spec = registry["parameters"]["num_soil_layers"]
    # Keep every declared geometry, including slab's five layers,
    # Noah/Noah-MP's four layers and each supported RUC level table.
    from woof.config import LAND_SURFACE_SOIL_LAYERS
    spec["enum"] = sorted({int(count)
                           for selector in LAND_SURFACE_SOIL_LAYERS
                           if selector != 0
                           for count in LAND_SURFACE_SOIL_LAYERS[selector]})
    # Adding slab's five-layer geometry changes no earlier selected soil
    # count. The comparator checks the complete historical declaration,
    # so changed defaults, types or bounds still refuse an old receipt.
    spec["compatible_previous_enums"] = [[4, 9], [4, 6, 9], [4, 5, 6, 9]]
    spec["warnings"] = [
        warning.replace(
            "This ENUM declares 4 and 9 -- 4 with Noah "
            "or Noah-MP, 9 with RUC -- because those are the geometries an "
            "oracle has judged; the enum is a statement about EVIDENCE, not "
            "the whole set of selectable values.",
            "This ENUM declares every geometry a plan may STATE, derived "
            "from woof.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS and "
            "Noah's single count, because a schema that refused a value the "
            "shipped resolver accepts was two authorities disagreeing about "
            "one geometry. Which count each SCHEME resolves is a separate "
            "question, answered by soil_layer_count above; how much evidence "
            "each geometry carries is a third, answered here and in the run "
            "receipt.",
        ).replace(
            "A schema enum entry would advertise a validated geometry; "
            "see docs/wrf_ruc_runtime_admission.md.",
            "Six carries no WRF FORECAST oracle -- every "
            "lsmruc/sfctmp/soilmoist/snowtemp fixture in the tree is "
            "nine-level -- so a run at six carries "
            "soil_geometry_evidence=internal-consistency-only in its "
            "receipt and warns at runtime. What IS measured at six: ZS/DZS "
            "reproduce WRF 4.7.1 real.exe wrfinput_d01 exactly, and the "
            "43-field host/device column comparison agrees at max_ulp 0. "
            "See docs/wrf_ruc_runtime_admission.md.",
        )
        for warning in spec.get("warnings", [])
    ]


#: Radiation-arm siblings: the SAME registered composition with the
#: radiation ENGINE moved and nothing else.  Both 4/4 engines live on ONE
#: registry option (``components.radiation.options.rte-rrtmgp``) and are
#: chosen by a SETTING, ``ra_rrtmg_variant``, so a sibling that swaps
#: engines moves no component at all and the suite-less table below --
#: which is keyed on component moves -- cannot express it.
#:
#: Each row is minted by copying its base and applying the settings
#: written here, so the pair differs in the radiation engine and in
#: nothing else and a paired run isolates it.  The maturity is NOT written
#: here: :func:`_composition_ceiling` derives it from the option rows the
#: composition selects, which is clause C2 of the ratified composition
#: rule.  A row here therefore cannot claim evidence the tree does not
#: hold and needs no entry in ``composition_exemptions``.
_RADIATION_ARM_SIBLINGS = (
    # (new id, base id, settings moved, label, warnings)
    (
        "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {
            "ra_rrtmg_variant": "rte-rrtmgp",
            "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-to-rte-rrtmgp-v2",
        },
        "Thompson + YSU + classic MM5 + Noah + cumulus off + RTE+RRTMGP",
        (
            "Composition candidate: every component is individually "
            "verified (Thompson mp8 is wrf-matched-run; the RTE+RRTMGP "
            "longwave and shortwave engines are the shipped 4/4 radiation "
            "arm) and no receipt covers the composed suite, so this "
            "template sits AT its composition ceiling rather than above "
            "it.  The upgrade payer is named: this composition's first "
            "stock-WRF-paired t0/case receipt is what moves the label.",
            "This template differs from "
            "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1 in exactly ONE "
            "setting, the radiation engine, so the pair is a controlled "
            "engine comparison; every component and every other parameter, "
            "including the per-domain row, is transcribed from that "
            "template.",
            "radt 12.0 at dx 3000 m is transcribed from the base template "
            "and deliberately diverges from the KF template family's "
            "per-domain ladder (radt 3.0 at 3 km); every value in the "
            "per-domain row equals this template's own parameters "
            "(diff_6th_factor 0.12, epssm 0.5, radt 12.0), so the row "
            "changes no resolved setting at any grid spacing.",
        ),
    ),
)


def _radiation_arm_siblings(registry: dict) -> None:
    """Mint every row of :data:`_RADIATION_ARM_SIBLINGS`.

    Idempotent, exactly as the sibling blocks around it are: a minted id is
    removed from every route list before it is inserted, so a second build
    produces the same bytes.

    A sibling is declared on precisely the routes and sources its BASE is
    declared on, immediately after it.  The base's route set is the
    statement of where this composition can run at all, and swapping the
    radiation engine neither widens nor narrows it -- so the sibling
    inherits the set rather than asserting one of its own.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    for template_id, base_id, settings, label, warnings in (
            _RADIATION_ARM_SIBLINGS):
        template = copy.deepcopy(templates[base_id])
        template["parameters"].update(settings)
        template["label"] = label
        template["maturity"] = _composition_ceiling(
            registry, template["components"])
        template["warnings"] = list(warnings)
        templates[template_id] = template
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if template_id in declared:
                    declared.remove(template_id)
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if base_id in declared:
                    declared.insert(
                        declared.index(base_id) + 1, template_id)


#: Microphysics-arm siblings: a registered composition with ONE component
#: moved, the microphysics, and declared on precisely the routes and
#: sources its base is declared on, immediately after it.
#:
#: The suite-less table below cannot carry these: it registers a suite on
#: EVERY declared source of every route, and a RUC suite runs only where
#: RUC's nine-layer soil ingest does, which is what the base's own route
#: set already states.  Moving the microphysics neither widens nor narrows
#: that set, so the sibling inherits it.
#:
#: The maturity is not written here: :func:`_composition_ceiling` derives
#: it from the option rows the composition selects, and the build refuses
#: a row whose id names a maturity the derivation does not reach.
_MICROPHYSICS_ARM_SIBLINGS = (
    ("lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1",
     "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1", "lin-mp2",
     "implemented-unverified", "Purdue Lin + YSU + MM5 + Noah + legacy RRTMG",
     ("Lin has strict 72-column oracle evidence; the coupled suite has no observation qualification.",),
     ("tools.hrrr_single_domain_benchmark",),
     ("tools.prepared_single_domain_forecast",)),
    # (new id, base id, microphysics option, maturity the id names,
    #  label, lead warnings)
    (
        "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "thompson-mp8",
        "implemented-unverified",
        "Thompson + MYNN PBL + MYNN surface layer + RUC LSM + RTE+RRTMGP",
        (
            "Composition candidate: Thompson mp8 is wrf-matched-run and "
            "RTE+RRTMGP is the shipped 4/4 radiation arm, while MYNN and RUC "
            "are implemented-unverified, so this suite sits at the "
            "implemented-unverified ceiling its weakest members set. No "
            "receipt covers the composed suite; its first stock-WRF-paired "
            "t0/case receipt is what moves the label.",
            "This template differs from "
            "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 in "
            "exactly ONE component, the microphysics (WSM6 -> Thompson), so "
            "the pair isolates it; every other component and parameter is "
            "transcribed from that template, and it is offered on exactly "
            "the routes and sources that template is.",
            "Both radiation streams run, so this is the nocturnally valid "
            "member of the Thompson + MYNN + RUC pair and the one to choose "
            "for low cloud, fog and stratus, where the night-time longwave "
            "cooling is the process being forecast.",
            "RADIATION IS THE ONLY DIFFERENCE from "
            "thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1: "
            "ra_lw_physics 0 -> 4, ra_sw_physics 1 -> 4 and radt 1.0 -> 12.0. "
            "This row is nocturnally valid and that one is not.",
        ),
    ),
    (
        "thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1",
        "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
        "thompson-mp8",
        "implemented-unverified",
        "Thompson + MYNN PBL + MYNN surface layer + RUC LSM + Dudhia SW",
        (
            "Composition candidate: Thompson mp8 is wrf-matched-run, while "
            "MYNN and RUC are implemented-unverified, so this suite sits at "
            "the implemented-unverified ceiling its weakest members set. No "
            "receipt covers the composed suite.",
            "This template differs from "
            "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1 in "
            "exactly ONE component, the microphysics (WSM6 -> Thompson), so "
            "the pair isolates it; every other component and parameter is "
            "transcribed from that template, and it is offered on exactly "
            "the routes and sources that template is.",
            "DAYTIME-ONLY SUITE. The radiation component is "
            "'dudhia-shortwave': ra_lw_physics 0 with ra_sw_physics 1, so "
            "Dudhia shortwave runs and no longwave scheme does, and GLW "
            "stays at zero for the whole forecast. A real window containing "
            "local night refuses to load unless [experiment] declares "
            "acknowledgements = [\"asymmetric-radiation-nocturnal-window-v1\"]. "
            "For fog and stratus choose "
            "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1, "
            "which runs both streams.",
        ),
    ),
)


#: Warning sentences that describe a sibling's BASE pairing rather than the
#: minted composition: carried over, they would name the base's partner (a
#: WSM6 row) as this row's only difference.
_BASE_DESCRIBING_WARNING_PREFIXES = (
    "This template differs from ",
    "RADIATION IS THE ONLY DIFFERENCE from ",
)


THOMPSON_FORK_SOURCE_PROFILE_ID = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"


def _thompson_fork_source_template(registry: dict) -> None:
    template = registry["templates"][THOMPSON_FORK_SOURCE_PROFILE_ID]
    microphysics = registry["components"]["microphysics"]["options"][template["components"]["microphysics"]]
    if microphysics["selectors"]["mp_physics"] != 28:
        raise ValueError("Thompson fork source forms require the staged MP28 composition")
    if template["parameters"]["bl_mynn_version"] != "gsd_41":
        raise ValueError("Thompson source composition must retain its staged MYNN generation")
    if registry["parameters"]["bl_mynn_cloud_tendency_form"]["default"] != "wrf_461":
        raise ValueError("Thompson source composition must retain the conservative MYNN cloud default")
    template["parameters"].update(
        thompson_version="wrf_39_noaa", thompson_fork_snow_fall="wrf_39_noaa")


def _mynn_source_version_templates(registry: dict) -> None:
    """A source version is an explicit composition, never a global default."""
    template_id = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
    base_id = "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"
    template = copy.deepcopy(registry["templates"][base_id])
    template["components"]["microphysics"] = "thompson-aerosol-mp28"
    template["parameters"].update({
        "bl_mynn_version": "gsd_41",
        "bl_mynn_gsd41_unsquared_qtke": False,
        "mynn_sfclay_variant": "gsl_wrf39",
        "bl_mynn_mixlength": 2,
        "scalar_pblmix": 1,
        "aer_init_opt": 1,
        "wif_input_opt": 1,
        "ra_rrtmg_variant": "rrtmg_legacy",
        "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-legacy-v1",
        "radt": 15.0,
    })
    # The suite runs HRRR's NOAA WRFV3.9 fork Thompson (thompson_version =
    # "wrf_39_noaa"), not WRF 4.6.1's; its label says so (a user who swapped
    # in YSU, Noah and KF read it as WRF's mp28).
    template["label"] = ("HRRR WRFV3.9 aerosol Thompson + GSD v4.1 MYNN + MYNN surface + RUC + "
                         "legacy RRTMG")
    template["maturity"] = _composition_ceiling(registry, template["components"])
    template["warnings"] = [
        "The GSD v4.1 MYNN source version is selected explicitly. Remaining fork differences are listed in docs/dev/mynn-gsd41.md.",
        "This composition requires the staged WIF aerosol climatology and legacy RRTMG assets. It has no matched operational full-grid trajectory receipt.",
    ]
    registry["templates"][template_id] = template
    for route_id, route in registry["runner_routes"].items():
        for source_id, declared in route.get("source_template_ids", {}).items():
            if template_id in declared:
                declared.remove(template_id)
            if route_id == "tools.prepared_single_domain_forecast":
                from woof.prepared_single_domain_forecast import _SOURCE_PHYSICS_PROFILES
                offered = _SOURCE_PHYSICS_PROFILES.get(source_id, ())
                if template_id in offered:
                    declared.append(template_id)
                    order = {name: index for index, name in enumerate(offered)}
                    declared.sort(key=lambda name: order.get(name, len(order)))
                continue
            if base_id in declared:
                declared.insert(declared.index(base_id) + 1, template_id)


def _monthly_surface_template(registry: dict) -> None:
    """A named surface configuration with its prescribed fields selected."""
    template_id = "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1"
    base_id = "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1"
    template = copy.deepcopy(registry["templates"][base_id])
    template["label"] = (
        "Thompson + MYNN + RUC monthly LAI/albedo and fractional sea ice + legacy RRTMG")
    template["parameters"].update(
        usemonalb=True, rdlai2d=True, fractional_seaice=1,
        ra_rrtmg_variant="rrtmg_legacy",
        wrf_rrtmg_compatibility="wrf-rrtmg-4-4-legacy-v1")
    template["maturity"] = _composition_ceiling(registry, template["components"])
    template["warnings"] = [
        "The monthly surface fields and the 0.02 fractional ice threshold "
        "are selected by this template with no additional flags. A static "
        "catalogue must supply LAI12M, ALBEDO12M and SNOALB. The template "
        "does not qualify a forecast against operational output; solar "
        "angle albedo, native cycled inputs and the complete physics "
        "configuration require separate verification.",
        *[warning for warning in template.get("warnings", [])
          if not warning.startswith(_BASE_DESCRIBING_WARNING_PREFIXES)],
    ]
    registry["templates"][template_id] = template
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if template_id in declared:
                declared.remove(template_id)
            if base_id in declared:
                declared.insert(declared.index(base_id) + 1, template_id)


def _solar_monthly_surface_template(registry: dict) -> None:
    """An explicit solar-albedo sibling; prior templates keep their values."""
    base_id = "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1"
    template_id = "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1"
    template = copy.deepcopy(registry["templates"][base_id])
    template["parameters"]["alb_sol"] = 1
    template["label"] = (
        "Thompson + MYNN + RUC monthly LAI/albedo, solar albedo and "
        "fractional sea ice + legacy RRTMG")
    template["warnings"][0] = (
        "Monthly LAI and albedo, the 0.02 fractional ice threshold and "
        "sun-angle albedo are selected by this template with no additional "
        "flags. A MODIS21 static catalogue must supply LAI12M, ALBEDO12M "
        "and SNOALB. This template does not qualify a forecast against "
        "operational output; native cycled inputs and the complete physics "
        "configuration require separate verification.")
    registry["templates"][template_id] = template
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            if template_id in declared:
                declared.remove(template_id)
            if base_id in declared:
                declared.insert(declared.index(base_id) + 1, template_id)


def _microphysics_arm_siblings(registry: dict) -> None:
    """Mint every row of :data:`_MICROPHYSICS_ARM_SIBLINGS`.

    Idempotent, as :func:`_radiation_arm_siblings` is: a minted id is
    removed from every route list before it is inserted after its base, so
    a second build over this build's own output produces the same bytes.

    The base's warnings are inherited after the row's own, less the two
    kinds of sentence that describe the BASE rather than the composition:
    its "differs from" pairing sentence, and the frozen-name warning a
    'no-radiation' id carries, which would be false on an id that does not
    say 'no-radiation'.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    for row in _MICROPHYSICS_ARM_SIBLINGS:
        template_id, base_id, microphysics, named_maturity, label, warnings = row[:6]
        excluded_routes = set(row[6] if len(row) > 6 else ())
        offered_routes = tuple(row[7] if len(row) > 7 else ())
        base = templates[base_id]
        template = copy.deepcopy(base)
        template["components"]["microphysics"] = microphysics
        template["label"] = label
        maturity = _composition_ceiling(registry, template["components"])
        if MATURITY_RENAMES.get(named_maturity, named_maturity) != maturity:
            raise SystemExit(
                f"template {template_id!r} names maturity {named_maturity!r} "
                f"and its composition ceiling is {maturity!r}; an id that "
                "states a rank its components do not reach is a false claim "
                "every door would repeat")
        template["maturity"] = maturity
        inherited = [
            warning for warning in base.get("warnings", [])
            if not warning.startswith(_BASE_DESCRIBING_WARNING_PREFIXES)
            and warning != NO_RADIATION_NAME_WARNING
        ]
        template["warnings"] = [*warnings, *inherited]
        templates[template_id] = template
        for route in routes.values():
            for declared in route.get("source_template_ids", {}).values():
                if template_id in declared:
                    declared.remove(template_id)
        for route_id, route in routes.items():
            if route_id in excluded_routes:
                continue
            for declared in route.get("source_template_ids", {}).values():
                if base_id in declared:
                    declared.insert(
                        declared.index(base_id) + 1, template_id)

        for route_id in offered_routes:
            route = routes[route_id]
            for declared in route.get("source_template_ids", {}).values():
                # Existing source declarations establish the generic basis.
                # An empty list denotes a caller-supplied composition.
                if declared and template_id not in declared:
                    declared.append(template_id)


#: Audit R-067.  Eleven implemented options had no shipped template at all,
#: so every one of them was a scheme a user could not select from a named
#: suite -- ``implemented: true`` with no front door.  Each row below is
#: built from an existing template by moving the FEWEST components that
#: reach the option, so a paired run against its base isolates the change,
#: and each is registered on every route the composition is valid for.
#:
#: The ids carry no case, site or source token: they name the composition.
_SUITELESS_TEMPLATES = (
    ('ferrier-aligo-advect-mp15-ysu-mm5-noah-rrtmg-legacy-v1', 'thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1', {'microphysics': 'ferrier-aligo-advect-mp15'}, {}, 'Ferrier-Aligo advected rime + YSU + classic MM5 + Noah + legacy RRTMG', ('Composition candidate. The selected scheme has independent native-column checks. No matched native forecast or observation score qualifies this composed suite.',)),
    ('goddard4ice-mp7-ysu-mm5-noah-rrtmg-legacy-v1', 'thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1', {'microphysics': 'goddard4ice-mp7'}, {}, 'Goddard four-ice + YSU + classic MM5 + Noah + legacy RRTMG', ('Composition candidate. The selected scheme has independent native-column checks. No matched native forecast or observation score qualifies this composed suite.',)),
    ('morrison-aero-mp40-ysu-mm5-noah-rrtmg-legacy-v1', 'thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1', {'microphysics': 'morrison-aero-mp40'}, {}, 'Morrison aerosol-aware + YSU + classic MM5 + Noah + legacy RRTMG', ('Composition candidate. The selected scheme has independent native-column checks. No matched native forecast or observation score qualifies this composed suite.',)),
    ('thompson-mp8-ysu-mm5-noah-cam3-v1', 'thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1', {'radiation': 'wrf-cam3'}, {}, 'Thompson + YSU + classic MM5 + Noah + CAM3', ('Composition candidate. The selected scheme has independent native-column checks. No matched native forecast or observation score qualifies this composed suite.',)),

    (
        "thompson-graupel-hail-mp38-ysu-mm5-noah-kf-rrtmg-legacy-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {"microphysics": "thompson-graupel-hail-mp38", "cumulus": "kain-fritsch"},
        {"thompson_version": "wrf_461"},
        "Thompson graupel-hail + YSU + classic MM5 + Noah + KF + legacy RRTMG",
        (
            "Real three-hour forecast, two-member ensemble and live-moment restart "
            "and nest checks completed; strict synthetic residuals remain declared. "
            "This suite has no observation-based skill qualification.",
            "Monthly aerosol forcing is required unless the caller explicitly "
            "selects the synthetic aerosol source.",
        ),
    ),
    (
        "wsm6-ysu-mm5-noah-osas-rrtmg-legacy-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {"microphysics": "wsm6-mp6", "cumulus": "old-gfs-sas"},
        {"cudt_minutes": 0.0},
        "WSM6 + YSU + classic MM5 + Noah + OSAS + legacy RRTMG",
        ("Composition candidate: OSAS has strict real-input column conformance and one three-hour native/GPU forecast with ASOS comparison; that single composed case does not establish independent OSAS skill.",),
    ),
    (
        "wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {"microphysics": "wsm6-mp6", "cumulus": "new-sas"},
        {"cudt_minutes": 0.0, "nsas_dx_factor": 0, "sas_pgcon": 0.55},
        "WSM6 + YSU + classic MM5 + Noah + New SAS + legacy RRTMG",
        (
            "Composition candidate: NSAS deep, shallow and driver calls have WRF column conformance. This suite has no observation-based forecast score.",
        ),
    ),
    (
        "ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"pbl": "ccpp-tke-edmf", "cumulus": "grell-freitas"},
        {"moist": True},
        "CCPP TKE-EDMF + Thompson + classic MM5 + Noah + Grell-Freitas + RTE+RRTMGP",
        (
            "Composition candidate. CCPP column parity is measured against its pinned upstream oracle; this is not a WRF PBL counterpart.",
            "No long integration or observation-based skill receipt covers this composed suite. Positive PBL cadence is refused because extra-scalar tendencies are not serialized.",
        ),
    ),
    (
        "wsm6-eeps-mm5-noah-dudhia-sw-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "eeps"},
        {},
        "WSM6 + EEPS + classic MM5 + Noah + Dudhia shortwave",
        (
            "Composition candidate: EEPS is a recovered WRF transcription "
            "with a RADSUM-zero repair. This WSM6 composition "
            "has no forecast trajectory comparison or observation score.",
        ),
    ),
    (
        "thompson-mp8-ysu-mm5-noah-grell-devenyi-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"cumulus": "grell-devenyi"},
        {"cudt_minutes": 0.0},
        "Thompson + YSU + classic MM5 + Noah + Grell-Devenyi + RTE+RRTMGP",
        ("Composition candidate: cu_physics=93 is column-verified, but no "
         "forecast or observation receipt qualifies this composed suite. "
         "The stock WRF water closure count is used in both arithmetic modes.",),
    ),

    # (new id, base id, component moves, extra parameters, label, warnings)
    (
        "wdm5-mp14-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"microphysics": "wdm5-mp14", "cumulus": "grell-freitas"},
        {},
        "WDM5 + YSU + classic MM5 + Noah + Grell-Freitas + RTE+RRTMGP",
        (
            "Composition candidate. WDM5 has column implementation evidence, "
            "including its declared differences from stock WRF. No observation "
            "score or matched forecast receipt covers this composed suite.",
        ),
    ),
    (
        "wdm7-mp26-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"microphysics": "wdm7-mp26", "cumulus": "grell-freitas"},
        {},
        "WDM7 + YSU + classic MM5 + Noah + Grell-Freitas + RTE+RRTMGP",
        (
            "Composition candidate. WDM7 has column implementation evidence, "
            "including its declared differences from stock WRF. No observation "
            "score or matched forecast receipt covers this composed suite.",
        ),
    ),
    (
        "milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1",
        "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1",
        {"microphysics": "milbrandt2mom-mp9", "cumulus": "new-tiedtke"},
        {},
        "Milbrandt-Yau two-moment + YSU + classic MM5 + Noah + New Tiedtke "
        "+ legacy RRTMG",
        (
            "Composition candidate: every component is individually "
            "implemented and the legacy RRTMG engine is the certified WRF "
            "v4.6.1 port, but no receipt covers the composed suite. This is "
            "the suite that ran 59 minutes and died at its first checkpoint "
            "before the identity row for mp_physics=9 existed; it is "
            "registered so the scheme has a named front door rather than "
            "only an unnamed tuple.",
            "This preset selects the legacy RRTMG engine to retain its "
            "named composition. Milbrandt-Yau also supports the modern "
            "RTE+RRTMGP coupling through its implemented cloud-optics row; "
            "choose that engine in a separate composition when wanted.",
        ),
    ),
    (
        "wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {"microphysics": "wdm6-mp16", "cumulus": "grell-freitas"},
        {},
        "WDM6 + YSU + classic MM5 + Noah + Grell-Freitas + RTE+RRTMGP",
        (
            "Composition candidate: WDM6 and Grell-Freitas are each "
            "implemented and individually measured, and RTE+RRTMGP is the "
            "shipped radiation arm, but no receipt covers the composed "
            "suite. Registered so both schemes have a named front door.",
            "Grell-Freitas requires a PBL closure that supplies its "
            "boundary-layer state (its own requires_components row); YSU is "
            "the arm this suite pairs it with.",
        ),
    ),
    (
        "thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
        "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        {
            "microphysics": "thompson-aerosol-mp28",
            "pbl": "myj",
            "surface_layer": "eta-similarity",
            "cumulus": "off",
        },
        {"cudt_minutes": 0.0},
        "Thompson aerosol-aware + MYJ + Eta similarity + Noah + cumulus off "
        "+ RTE+RRTMGP",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the aerosol-aware Thompson scheme and the "
            "MYJ/Eta pair each have a named front door.",
            "MYJ and the Eta similarity surface layer require EACH OTHER "
            "(both requires_components rows, and woof.config."
            "validate_myj_pairing), so they are selected together or not at "
            "all. This template is the one registered suite that satisfies "
            "both halves.",
            "The aerosol-aware scheme reads the packaged CCN activation "
            "table shipped in the recast-woof-data companion wheel; a default "
            "install satisfies it, and a run with no bound aerosol input "
            "uses the packaged fallback constants and says so.",
        ),
    ),
    (
        "wsm6-sase-revised-mm5-noah-closure-supplied-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {
            "pbl": "sase",
            "surface_layer": "revised-mm5",
            "turbulence": "closure-supplied",
        },
        # NOT restated here.  SASE's own required_settings row pins the
        # quadruple (bldt, khdif, kvdif, km_opt), the derivation reads
        # that row into every runtime product, and a template that
        # mentions ``bldt`` at ANY value is the inheritance vector
        # tests/test_noahmp_surface_interval.py refuses outright: a
        # surface-call interval is a cost mitigation that must stay
        # opt-in per configuration, and a template carries its parameters
        # into every run built from it with no author ever typing them.
        {},
        "WSM6 + SASE + revised MM5 + Noah + closure-supplied mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the SASE closure has a named front door.",
            "SASE supplies its own mixing, so this suite runs with km_opt=0 "
            "and bldt, khdif and kvdif at zero -- the quadruple SASE's own "
            "required_settings row states, which every route resolves and "
            "plan review refuses an override of. The suite inherits that "
            "row rather than restating it. A second mixing operator beside "
            "the closure would double-count its own transport.",
            "SASE is not a WRF v4.6.1 scheme: it carries an "
            "out-of-namespace selector and has no stock-WRF counterpart to "
            "be compared against, so its surface-layer requirement states "
            "what the closure READS rather than transcribing WRF's cell "
            "table.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-tke-1-5-order-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "tke-1.5-order"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + 1.5-order TKE mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the 1.5-order TKE closure has a named front "
            "door.",
            "A three-dimensional mixing operator REPLACES the PBL "
            "parameterization rather than joining it: km_opt=2 requires "
            "bl_pbl_physics=0 (the option's own required_settings row), "
            "which is why this suite selects PBL off. The surface layer "
            "stays on and supplies the fluxes Noah needs.",
            "This is a large-eddy composition. On a grid whose spacing does "
            "not resolve the energy-containing eddies it under-mixes the "
            "boundary layer, because nothing else is parameterizing it.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "smagorinsky-3d"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + 3D Smagorinsky mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the three-dimensional Smagorinsky closure has a "
            "named front door.",
            "km_opt=3 requires bl_pbl_physics=0 (the option's own "
            "required_settings row): the operator replaces the PBL "
            "parameterization rather than joining it. The surface layer "
            "stays on and supplies the fluxes Noah needs.",
            "This is a large-eddy composition, and it is the sibling of the "
            "1.5-order TKE row: the two differ in exactly ONE component, so "
            "a paired run isolates the closure.",
        ),
    ),
    (
        "wsm6-pbl-off-mm5-noah-constant-k-v1",
        "wsm6-ysu-mm5-noah-no-radiation-v1",
        {"pbl": "off", "turbulence": "constant-k"},
        {},
        "WSM6 + PBL off + classic MM5 + Noah + constant-K mixing",
        (
            "Composition candidate: no receipt covers the composed suite. "
            "Registered so the constant-K operator has a named front door, "
            "as the km_opt=1 sibling of the two large-eddy rows above.",
            "khdif and kvdif come from the option's own row and are zero "
            "there, which is no horizontal or vertical mixing at all. A "
            "suite that wants constant-K transport states the two "
            "coefficients in its experiment config; this template does not "
            "invent values an oracle has not judged.",
        ),
    ),
)


#: The ONE runner route id that replays a native comparison rather than
#: building its product from the registry alone.  Named once, here, so the
#: exclusion table below and the gate that checks it read the same string.
_NATIVE_BENCHMARK_ROUTE = "tools.hrrr_single_domain_benchmark"

#: Which suite-less template stays off which route, and the concrete
#: breakage that keeps it off.  A template is registered on every route it
#: is valid for; every departure from "all three" is a row here, and
#: :func:`_phase2c_suiteless_templates` FAILS THE BUILD if a template is
#: absent from a route with no row -- which is the reverse leg the first
#: pass lacked, when six suites were declared on a route whose runner
#: refuses all six.
#:
#: RETIRED, with the fix it waited on: the aerosol-aware Thompson row that
#: kept its suite off BOTH fixed-template routes because
#: source_absent_microphysics had no mp_physics=28 arm (audit R-044).  The
#: arm exists (woof/ingest/microphysics_cold_start.py, nc/nr/ni at exact
#: zero), woof/ingest/real.py seeds nwfa/nifa from the WIF monthly
#: climatology, and a missing dataset is refused by name before the fetch
#: by woof.config.mp28_aerosol_lateral_forcing_precondition, which offers
#: mp28_aerosol_source = 'synthetic' as the way out.  The suite is on the
#: prepared single-domain route now; the native benchmark keeps it off for
#: its own reason, in the composition loop below.
_TEMPLATE_ROUTES_REFUSED: dict[str, dict[str, str]] = {}
#: The composition suites are valid compositions and the prepared
#: single-domain route resolves each of them from the registry alone.  The
#: NATIVE BENCHMARK route cannot: its product is a replay of a native WRF
#: run, gated field for field against a transcribed namelist contract
#: (tools/hrrr_single_domain_benchmark.py _NATIVE_HRRR_NAMELIST_CONTRACTS)
#: and forwarded through a per-switch home map (_PROFILE_SWITCH_HOMES).
#: A composition with no native run behind it has no contract to be
#: replayed against, so declaring it there offered a suite the runner
#: refused with `unsupported native HRRR physics profile`.
for _composition_suite_id, _off_the_benchmark_because in (
        ('ferrier-aligo-advect-mp15-ysu-mm5-noah-rrtmg-legacy-v1', 'no native run of this composition exists, so there is no namelist contract to gate its replay against'),
        ('goddard4ice-mp7-ysu-mm5-noah-rrtmg-legacy-v1', 'no native run of this composition exists, so there is no namelist contract to gate its replay against'),
        ('morrison-aero-mp40-ysu-mm5-noah-rrtmg-legacy-v1', 'no native run of this composition exists, so there is no namelist contract to gate its replay against'),
        ('thompson-mp8-ysu-mm5-noah-cam3-v1', 'no native run of this composition exists, so there is no namelist contract to gate its replay against'),

        ("wsm6-ysu-mm5-noah-osas-rrtmg-legacy-v1",
         "no matched native WRF forecast of this composition exists, so "
         "there is no namelist contract to gate its replay against"),
        ("wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1",
         "no matched native WRF forecast of this composition exists, so "
         "there is no namelist contract to gate its replay against"),
        ("lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1",
         "CCPP is a non-WRF closure, so no native WRF namelist contract can replay this composition"),
        ("wsm6-eeps-mm5-noah-dudhia-sw-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wdm5-mp14-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wdm7-mp26-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("thompson-mp8-ysu-mm5-noah-grell-devenyi-rte-rrtmgp-v1",
         "the native comparison has no recorded namelist contract for "
         "this composition"),
        ("milbrandt2mom-mp9-ysu-mm5-noah-ntiedtke-rrtmg-legacy-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("thompson-graupel-hail-mp38-ysu-mm5-noah-kf-rrtmg-legacy-v1",
         "no native trajectory receipt or immutable namelist contract "
         "covers this hail composition, so a native replay cannot be certified"),
        ("thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1",
         "no native run of this composition exists, so there is no "
         "namelist contract to gate its replay against"),
        ("wsm6-sase-revised-mm5-noah-closure-supplied-v1",
         "SASE carries an out-of-namespace selector and has no stock-WRF "
         "counterpart at all, so a native comparison cannot be stated for "
         "it, let alone measured"),
        ("wsm6-pbl-off-mm5-noah-tke-1-5-order-v1",
         "a large-eddy closure on this route's fixed kilometre-scale "
         "single domain has no native run behind it and no resolved "
         "energy-containing eddies to close over"),
        ("wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
         "a large-eddy closure on this route's fixed kilometre-scale "
         "single domain has no native run behind it and no resolved "
         "energy-containing eddies to close over"),
        ("wsm6-pbl-off-mm5-noah-constant-k-v1",
         "the constant-K operator runs with the option's own zero "
         "coefficients, so there is nothing for a published wall-clock "
         "and skill comparison to be measured against"),
):
    _TEMPLATE_ROUTES_REFUSED.setdefault(_composition_suite_id, {})[
        _NATIVE_BENCHMARK_ROUTE] = (
            "the native benchmark route replays ONE immutable template and "
            "gates the operator's native WRF namelist field for field "
            "against a transcribed contract: "
            + _off_the_benchmark_because
            + ". Run this suite on tools.prepared_single_domain_forecast, "
            "which resolves every switch of it from the registry, or state "
            "it per domain on tools.prepared_domain_tree_forecast.")
del _composition_suite_id, _off_the_benchmark_because


#: A source whose template contract is another declared source's, because the
#: two decode the SAME producer on the same field and soil contract through
#: routes that differ only in transport.  One row per such pair, so a route
#: that gains a second way into a producer it already serves inherits that
#: producer's declaration instead of acquiring an empty one.  This is table
#: work by construction: a new producer is a key, never a branch.
_CONTRACT_TWIN_SOURCES = {
    # The native and packaged pressure-level routes into one producer, each
    # way in naming the other: whichever of the pair a route declares, the
    # other inherits, so a route that reaches a producer twice does not
    # offer its suites on one road and nothing on the other.
    "hrrr": "hrrr-prs",
    "hrrr-prs": "hrrr",
    # The NetCDF profile decodes the same producer on the same
    # pressure-level and soil contract as the GRIB one.
    "20crv3-cf": "20crv3",
}


#: A source a route SERVES and names NO suite for, with the reason it names
#: none.  The completion below prices every other undeclared source from the
#: route's own generic declaration, so an empty list is reachable only by
#: saying here that the source names nothing and why.  Table work by
#: construction: such a source is a key, never a branch.
_SOURCES_THAT_NAME_NO_SUITE = {
    # The caller-supplied composition: no packaged profile stands behind it
    # and the physics is whatever the caller wrote, so there is no suite to
    # name and no route-wide expert list to reach it either (which is what
    # woof.physics_registry.expert_template_ids_for_source reads an empty
    # normal list as meaning).
    "mapped": "the composition is the caller's, so no suite is named for it",
}


def _route_generic_template_declaration(
        stated: dict[str, list[str]]) -> list[str]:
    """The suites a route declares for EVERY source it has declared.

    A template every declared source names is one that reads nothing
    source-specific, by construction -- no source's own verification row
    can survive the intersection -- so this is the most conservative basis
    the route itself has on record, and it is what a source with no
    measurement of its own is priced from.

    Computed from TWO OR MORE declared sources only.  With one, the
    intersection is that source's whole list, evidence rows and all, and
    handing it to a second source would publish one source's evidence as
    another's -- the exact failure tests/test_build_registry.py's Kessler
    pin caught in the first, route-wide version of this completion.
    """

    lists = [templates for templates in stated.values() if templates]
    if len(lists) < 2:
        return []
    common = list(lists[0])
    for other in lists[1:]:
        common = [template_id for template_id in common if template_id in other]
    return common


def _every_served_source_declares_a_template_list(registry: dict) -> None:
    """A fixed-template route that SERVES a source says what it offers there.

    Such a route runs ONLY its registered templates, so a source it serves
    and declares nothing for offers nothing at all, and the declaration has
    to say which of the two it means.

    ``source_ids`` and ``source_template_ids`` were allowed to disagree:
    three sources of the single-domain route were served and undeclared,
    so the runner drift check raised ``KeyError`` on the first of them
    rather than comparing anything, and
    ``expert_template_ids_for_source``'s rule that an empty declared list
    offers nothing could not fire for a source that had no list at all.

    An undeclared source gets, in order: the declaration of the source it
    shares a producer and contract with where one is recorded; an empty
    list where the source is recorded as naming no suite at all, with the
    reason; and otherwise the route's OWN generic declaration, the suites
    every source it has declared names.

    What it may NOT get is an empty list for want of a measurement.  That
    was the first shape of this pass, and an empty list here is not
    silence: it is published as "this source reaches no named suite", and
    ``expert_template_ids_for_source`` withholds the route-wide expert
    list from such a source as well.  On the single-domain route it took
    the six composition suites and the three Noah-MP expert suites away
    from aigfs and era5-l137 -- two sources the runner runs through the
    same generic mapped route as the seven siblings that declare those
    suites -- and warned a plan naming one with
    ``template-route-evidence``, which says the template is off-route for
    this source, instead of handing it the acknowledgement advisory.  The
    plan still launched; what was published about it was wrong.  A source with nothing measured on it is priced from the
    most conservative basis the route has on record, and the route says
    which basis that was.
    """

    for route_id, route in registry["runner_routes"].items():
        if route.get("mode") != "fixed-template":
            # An experiment-per-domain route composes from its declaration
            # rather than from a template, so an absent per-source list is a
            # different shape there, not an incomplete one -- and filling it
            # would hand a source evidence-scoped registrations made for
            # another (the Kessler and legacy-RRTMG rows are one source's own).
            continue
        declared = route.setdefault("source_template_ids", {})
        if not declared:
            continue
        # Twins and the generic basis both resolve against what the route
        # declared BEFORE this pass, so the completion does not depend on
        # the order source_ids happens to list a pair in, and a second
        # build over this build's own output changes nothing.
        stated = {source_id: list(templates)
                  for source_id, templates in declared.items() if templates}
        generic = _route_generic_template_declaration(stated)
        for source_id in route.get("source_ids", []):
            if declared.get(source_id):
                continue
            twin = _CONTRACT_TWIN_SOURCES.get(source_id)
            if twin is not None:
                declared[source_id] = list(stated.get(twin, ()))
                continue
            if source_id in _SOURCES_THAT_NAME_NO_SUITE:
                declared[source_id] = []
                continue
            if not generic:
                raise RuntimeError(
                    f"runner route {route_id} serves source {source_id}, "
                    "declares no template list for it, and its declared "
                    "sources share no suite for one to be priced from, so "
                    "completing it would publish a source as reaching no "
                    "named suite at all. Record the source it shares a "
                    "producer with in _CONTRACT_TWIN_SOURCES, or record "
                    "why it names no suite in _SOURCES_THAT_NAME_NO_SUITE.")
            declared[source_id] = list(generic)



_SUITELESS_TEMPLATES += ((
    "wsm6-ysu-mm5-ssib-rrtm-dudhia-v1", "wsm6-ysu-mm5-noah-no-radiation-v1",
    {"land_surface": "ssib", "radiation": "wrf-rrtm-dudhia", "surface_layer": "revised-mm5"},
    {"num_soil_layers": 3, "fractional_seaice": 1},
    "WSM6, YSU and SSiB (opt-in; 1.942x Noah lower-tail land stage) with RRTM and Dudhia",
    ["Explicit opt-in; never a default. The land stage has a conservative 1.942x Noah forecast qualification cost. The column core has a strict WRF oracle. This composition has no matched WRF forecast or observation verification."],
),)
_TEMPLATE_ROUTES_REFUSED["wsm6-ysu-mm5-ssib-rrtm-dudhia-v1"] = {
    _NATIVE_BENCHMARK_ROUTE: "No native run and namelist replay contract exists for this composition. Use a prepared forecast route."
}

def _phase2c_suiteless_templates(registry: dict) -> None:
    """Audit R-067: every implemented option gets a named suite.

    Eleven implemented options carried ``reachability: component-override``
    and no template at all, which is the ship-only-what-users-can-reach
    rule failing quietly: the option was selectable only by hand-writing a
    tuple, and the front doors that offer NAMED suites offered none of
    them.

    Each template below is registered on every route it is physically
    valid for.  None of these compositions reads anything source-specific,
    so what a route CAN do with them is the only question, and every
    departure from "all three routes" carries a row in
    ``_TEMPLATE_ROUTES_REFUSED`` naming the breakage and the way out.  A
    source that declares no suite AT ALL (the ``mapped`` row, which is a
    caller-supplied composition) does not get its first one here: that is
    the source's gap, not the composition's.

    The build FAILS if a minted template ends up off a route with no row,
    or carries a row for a route it was never kept off.  That reverse leg
    is why this function is the place the exclusion is enforced: the first
    pass declared six of these on the native benchmark route, whose runner
    refuses every one of them, and nothing in the build noticed.
    """

    templates = registry["templates"]
    routes = registry["runner_routes"]
    benchmark = routes["tools.hrrr_single_domain_benchmark"]
    tree = routes["tools.prepared_domain_tree_forecast"]
    single = routes["tools.prepared_single_domain_forecast"]
    minted: list[str] = []

    components = registry["components"]
    for (template_id, base_id, moves, extra_parameters, label,
         warnings) in _SUITELESS_TEMPLATES:
        template = copy.deepcopy(templates[base_id])
        template["components"].update(moves)
        # The base suite's parameters were chosen for the options this row
        # moves AWAY from.  Where the option moved TO declares a value of its
        # own, that value is the option's identity and wins over the base's
        # choice for the replaced scheme.  Breakage prevented: the CAM3 row
        # (radiation rte-rrtmgp -> wrf-cam3) inherited the legacy RRTMG
        # suite's wrf_rrtmg_compatibility='wrf-rrtmg-4-4-legacy-v1', so plan
        # review offered it and woof.config.validate_run_config refused it
        # at run start ("requires the resolved 4/4 pair ... got
        # ra_lw_physics=3"): 117 physics-combination tuples, four door
        # round trips and the kernel-compile walk.
        for component_id, option_id in moves.items():
            option = components[component_id]["options"][option_id]
            for name, value in (option.get("parameters") or {}).items():
                if name in template["parameters"]:
                    template["parameters"][name] = value
        # ra_rrtmg_variant chooses the engine of the RRTMG 4/4 pair and
        # exists only for that pair (woof.physics_compat derives it into a
        # product only when the resolved pair is 4/4), so a row that moves
        # radiation off 4/4 does not carry the base's variant.
        selectors: dict = {}
        for component_id, option_id in template["components"].items():
            selectors.update(
                components[component_id]["options"][option_id].get(
                    "selectors") or {})
        if (selectors.get("ra_lw_physics"),
                selectors.get("ra_sw_physics")) != (4, 4):
            template["parameters"].pop("ra_rrtmg_variant", None)
        template["parameters"].update(extra_parameters)
        template["label"] = label
        # The composition ceiling: a suite with no receipt of its own
        # cannot rank above the weakest thing in it, and none of these has
        # a composed receipt at all.
        template["maturity"] = "implemented-unverified"
        template["warnings"] = list(warnings)
        # per_domain_overrides transcribe values from a verified run of
        # the BASE suite; none of these suites has one, so the row is
        # dropped rather than inherited as a claim about this composition.
        template.pop("per_domain_overrides", None)
        templates[template_id] = template
        minted.append(template_id)

    # Idempotent: remove before inserting, exactly as the Kessler, legacy
    # NSSL-2, Shin-Hong and P3 rows above do, so a second build produces
    # the same bytes.
    for route in routes.values():
        for declared in route.get("source_template_ids", {}).values():
            for template_id in minted:
                if template_id in declared:
                    declared.remove(template_id)
    declared_on: dict[str, set[str]] = {
        template_id: set() for template_id in minted}
    for route_id, route in (
            (_NATIVE_BENCHMARK_ROUTE, benchmark),
            ("tools.prepared_domain_tree_forecast", tree),
            ("tools.prepared_single_domain_forecast", single),
    ):
        refused_here = {
            template_id
            for template_id, routes in _TEMPLATE_ROUTES_REFUSED.items()
            if route_id in routes
        }
        for source_id, declared in route["source_template_ids"].items():
            if source_id not in route["source_ids"]:
                continue
            if not declared:
                # A source with no declared suite at all keeps none.
                continue
            for template_id in minted:
                if template_id in refused_here:
                    continue
                declared.append(template_id)
                declared_on[template_id].add(route_id)

    # The reverse leg.  A template off a route with no written reason is a
    # silent narrowing; a reason for a route the template is on is a stale
    # refusal that will outlive what it describes.
    all_routes = {
        _NATIVE_BENCHMARK_ROUTE,
        "tools.prepared_domain_tree_forecast",
        "tools.prepared_single_domain_forecast",
    }
    for template_id in minted:
        refused = set(_TEMPLATE_ROUTES_REFUSED.get(template_id, {}))
        absent = all_routes - declared_on[template_id]
        if absent != refused:
            raise SystemExit(
                f"template {template_id!r} is declared on "
                f"{sorted(declared_on[template_id])} and refused on "
                f"{sorted(refused)}: every route a suite-less template is "
                "kept off needs a row in _TEMPLATE_ROUTES_REFUSED naming "
                "the breakage and the way out, and a row for a route it "
                "IS on must be retired with the reason it recorded "
                f"(unreconciled: {sorted(absent ^ refused)})")

    _publish_refused_template_ids(registry)


def _publish_refused_template_ids(registry: dict) -> None:
    """Put every route refusal where the user meets it: the registry.

    ``_TEMPLATE_ROUTES_REFUSED`` above states, for each template kept off
    a route, the concrete breakage and the way out.  It lived only in this
    builder, so none of it reached a user: plan review saw a template that
    the route simply did not declare, warned that "the resolved runtime
    settings still apply", and returned launchable -- and then the runner
    refused at the door with a bare ``unsupported ... physics profile``
    naming neither the breakage nor an alternative.  A refusal that fires
    after review, in a sentence that offers nothing, is the shape this
    project refuses to ship.

    Published as ``runner_routes.<runner>.refused_template_ids``, a map of
    template id to that sentence, so
    :func:`woof.physics_registry.validate_physics_plan` can refuse at plan
    review with the reason written here.  Rewritten from scratch on every
    build, so a retired row leaves the registry with the reason it recorded
    rather than outliving it.
    """

    for route_id, route in registry["runner_routes"].items():
        refused = {
            template_id: routes[route_id]
            for template_id, routes in sorted(_TEMPLATE_ROUTES_REFUSED.items())
            if route_id in routes
        }
        if refused:
            route["refused_template_ids"] = refused
        else:
            route.pop("refused_template_ids", None)


def _phase2c_recompute_reachability(registry: dict) -> None:
    """Recompute every option's ``reachability`` from templates and routes.

    ``reachability.state`` names the EASIEST path a user has to an option,
    and it was hand-set beside each row that created a path.  With the
    route declarations widened (R-021, R-022, R-023, R-059) and eleven
    options gaining their first template (R-067), hand-setting forty
    states is how one of them ends up wrong and nobody notices.  It is
    computed here from the same two facts a user actually has -- the
    declared template lists and the declared override lists -- and
    ``tests/test_registry_reachability.py`` recomputes it independently
    and fails on any difference, which is the agreement that makes this a
    derivation rather than a second opinion.

    An option that ends up unreachable keeps the blocker it declared: an
    implemented option declared unreachable must name what blocks it, and
    that sentence is written where the option is registered, not here.
    """

    components = registry["components"]
    templates = registry["templates"]
    # Easiest first, the order a user finds them in.
    state_order = ("template", "component-override", "expert-template")
    reached: dict[tuple[str, str], set[str]] = {
        (component_id, option_id): set()
        for component_id, component in components.items()
        for option_id in component["options"]
    }

    for route in registry["runner_routes"].values():
        if route.get("implemented") is not True:
            continue
        normal = route.get("source_template_ids", {}) or {}
        expert = route.get("expert_template_ids", {}) or {}
        declares = bool(normal) or bool(expert)
        per_domain = route.get("mode") == "experiment-per-domain"
        overridable = set(route.get("allowed_component_overrides", []) or [])
        option_overrides = route.get("allowed_component_options", {}) or {}
        expert_selector_keys = set(
            route.get("allowed_expert_selector_keys", []) or [])
        for source_id in route.get("source_ids", []) or []:
            if declares:
                normal_ids = list(normal.get(source_id, []) or [])
                expert_ids = list(expert.get(source_id, []) or [])
            else:
                normal_ids, expert_ids = list(templates), []
            for template_id in normal_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("template")
            for template_id in expert_ids:
                for component_id, option_id in templates[
                        template_id]["components"].items():
                    reached[(component_id, option_id)].add("expert-template")
            if not (normal_ids or expert_ids):
                # An override still needs a base template to override.
                continue
            for component_id, component in components.items():
                selector_keys = set(component.get("selector_keys", []) or [])
                by_override = per_domain and component_id in overridable
                by_selector = bool(selector_keys) and selector_keys <= (
                    expert_selector_keys)
                admitted = set(option_overrides.get(component_id, []) or [])
                for option_id in component["options"]:
                    if by_override or by_selector or option_id in admitted:
                        reached[(component_id, option_id)].add(
                            "component-override")

    for (component_id, option_id), ways in reached.items():
        option = components[component_id]["options"][option_id]
        declared = option.get("reachability")
        declared = declared if isinstance(declared, dict) else {}
        if option.get("implemented") is not True:
            # Nameable is not reachable: the resolver refuses it from every
            # template, on every route, for every source.
            option["reachability"] = declared
            continue
        state = next((name for name in state_order if name in ways), None)
        if state is None:
            if not declared.get("blocker"):
                raise RuntimeError(
                    f"components.{component_id}.options.{option_id} is "
                    "implemented and no template or route declaration "
                    "reaches it, and it names no blocker; give it a route "
                    "row or write what blocks it")
            option["reachability"] = {
                "state": "unreachable", "blocker": declared["blocker"]}
            continue
        option["reachability"] = {"state": state}


def render(registry: dict) -> bytes:
    """The exact bytes the tracked registry file must contain."""
    return (canonical_json(registry) + "\n").encode("utf-8")


#: Why every cumulus option pins moist=true, in the words of the run door
#: (woof.config validate_run_config): rendered by the plan door after the
#: pinned value so both doors name one breakage.
_CUMULUS_MOIST_REASON = (
    "the cumulus schemes are moist convective schemes; "
    "woof/core/physics.py initialize_physics refuses a cumulus scheme on a "
    "dry DomainState, whose qv is None")


def _kf_trigger_templates(registry):
    """Expose the native global KF switch through selectable shared templates."""
    base_id = "thompson-mp8-ysu-mm5-noah-kf-rte-rrtmgp-v1"
    for trigger in (2, 3):
        template_id = f"thompson-mp8-ysu-mm5-noah-kf-trigger{trigger}-rte-rrtmgp-v1"
        template = copy.deepcopy(registry["templates"][base_id])
        template["parameters"]["kfeta_trigger"] = trigger
        template["label"] += f"; KF trigger {trigger}"
        template["maturity"] = _composition_ceiling(registry, template["components"])
        template.pop("evidence_pointer", None)
        template.pop("per_domain_overrides", None)
        template["warnings"] = [
            "KF trigger is global (WRF 4.7.1 Registry.EM_COMMON:2495). "
            "Every domain in a tree must select the same trigger. "
            "Native-call replay does not establish whole-forecast parity or skill."]
        registry["templates"][template_id] = template
        for route in registry["runner_routes"].values():
            for ids in route.get("source_template_ids", {}).values():
                while template_id in ids:
                    ids.remove(template_id)
                if base_id in ids:
                    ids.insert(ids.index(base_id) + 1, template_id)


def _multiscale_column_option(registry):
    from copy import deepcopy
    option=deepcopy(registry['components']['cumulus']['options']['kain-fritsch'])
    option.update(label='Multiscale KF',selectors={'cu_physics':11},implemented=True,
                  maturity='experimental-runtime',scientific_evidence='none',
                  reachability={'state':'component-override'},
                  warnings=['Fresh native replay and one uninterrupted three-hour forecast are recorded. Resident health status polls are asynchronous; publication and final return drain all ledgers. Explicit opt-in only: the active-input MSKF column-kernel deck costs 6.6714x CU1 on RTX 4090; whole forecast stage costs 3.030382x CU1. Source-bound parallel prewarm limits the cold first step to 19.1382 seconds with a longest sampled zero/one active-process interval of eight seconds. Hardware profiling attributes 55.9% of samples to the native binary64 momentum/pressure routine; this extra work is absent from CU1. Positive aerosol selectors use the public NASA monthly replacement: written from papers, not bitwise to WRF. Positive public aerosol forcing is an explicit measured-cost opt-in at 3.6864x/3.6830x aerosol off; the 1.2x bar remains unmet. Matched forecast receipt: docs/evidence/retime-20261102/cu11-summary.json.'])
    option.pop('consumers',None)
    option['extensions']={'column_contract': {'lookup_table_asset_role':'kf_lutab', 'lookup_table_loader':'woof.core.kf.load_kf_table', 'module': 'woof.core.mskf', 'launcher': 'launch_mskf', 'phase_factory': 'MskfPhase', 'atmosphere_inputs': {'u': 'u', 'v': 'v', 't': 'temperature', 'qv': 'qv', 'p': 'pressure', 'rho': 'rho', 'pi': 'exner', 'dz8w': 'dz'}, 'surface_inputs': {'hfx': 'hfx', 'ust': 'ust', 'pblh': 'pblh', 'zol': 'zol'}, 'outputs_3d': ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten', 'rucuten', 'rvcuten', 'cldfra_dp', 'cldfra_sh', 'qc_cu', 'qi_cu', 'qr_cu', 'qs_cu', 'nc_cu', 'ni_cu', 'nr_cu', 'ns_cu', 'ccn_cu', 'efcs', 'efis', 'efss', 'udr_kf', 'ddr_kf', 'uer_kf', 'der_kf'], 'outputs_2d': ['raincv', 'pratec', 'nca', 'cutop', 'cubot', 'ainc_frac', 'timec_kf'], 'result_mapping': {'rthcuten': 'rthcuten', 'rqvcuten': 'rqvcuten', 'rqccuten': 'rqccuten', 'rqrcuten': 'rqrcuten', 'rqicuten': 'rqicuten', 'rqscuten': 'rqscuten', 'rucuten': 'rucuten', 'rvcuten': 'rvcuten', 'rainc': 'raincv', 'pratec': 'pratec', 'nca_seconds': 'nca'}, 'optional_tendencies': {'rqrcuten': 'qr', 'rqicuten': 'qi', 'rqscuten': 'qs'}, 'radiation_feedback': ['cldfra_dp', 'cldfra_sh', 'qc_cu', 'qi_cu', 'qs_cu', 'efcs', 'efis', 'efss'], 'warm_rain_selectors': [1]}}
    option['extensions']['verification_scope']={'receipt':'docs/evidence/mskf-20261101/verification.json','native_words':107520,'differing_words':0,'aercu_opt':0,'forecast_hours':3,'full_pipeline_qualification':'resident aercu_opt=0 qualified: asynchronous status polling, publication and final-return drains; periodic state checks retained','stage_speed_ratio_to_cu1':3.0303820420637178,'forecast_stage_receipt':'docs/evidence/retime-20261102/cu11-summary.json','column_kernel_speed_ratio_to_cu1':6.671356507587179,'prior_forecast_stage_speed_ratio_to_cu1':7.4346915830324365,'cold_start_receipt':'docs/evidence/mskf-20261102-retime/cold-start.json','speed_receipt':'docs/evidence/mskf-20261102-retime/forecast-stage.json','column_kernel_speed_receipt':'docs/evidence/mskf-20261102-speed/verification.json','speed_report':'spd-retime-rad-cu.md','selection_policy':'explicit-component-override-only; never a default'}
    option['constraints'].setdefault('forbidden_setting_values',{})['km_opt']=[5]
    option['constraints'].setdefault('forbidden_setting_values_reasons',{})['km_opt']='WRF share/module_check_a_mundo.F:1353-1363 forbids MSKF with the incompatible 3D TKE closure.'
    registry['components']['cumulus']['options']['multiscale-kf']=option


def _multiscale_parameters(registry):
    for key in ('aercu_opt','aercu_fct','cu_rad_feedback','kf_edrates'):
        parameter=registry['parameters'][key]
        parameter['implemented']=True
        parameter.pop('unimplemented_reason',None)
        parameter.pop('refusal',None)
        parameter['readers']=['woof.core.mskf.launch_mskf','woof.core.morrison_aero.launch_morrison_aero']
    registry['parameters']['cu_rad_feedback']['readers']=['woof.namelist_import.import_namelists','woof.core.physics.initialize_physics','woof.core.kf.KainFritsch.__init__','woof.core.column_cumulus.MskfColumnCumulus.radiation_feedback']
    registry['parameters']['aercu_fct']['default']=1.0
    registry['parameters']['aercu_opt']['default']=0
    # Public NASA replacement is an explicit opt-in, off by default.
    registry['parameters']['aercu_opt']['enum']=[0,1,2]
    registry['parameters']['aercu_opt']['minimum']=0
    registry['parameters']['kf_edrates'].update(
        type='integer', enum=[0,1], default=0,
        read_when={'cu_physics': [11, 99]},
        readers=['woof.core.column_cumulus.MskfColumnCumulus.__call__', 'woof.core.kf.KainFritsch.__call__'],
        warnings=['Rate-profile carriers are implemented for cu_physics=11 or cu_physics=99.'])
    mp40=registry['components']['microphysics']['options']['morrison-aero-mp40']
    mp40['aerosol_state']['forecast_input_active']=True
    for key in ('required_settings', 'required_settings_reasons'):
        mp40['constraints'].get(key, {}).pop('aercu_opt',None)
    mp40['warnings']=['Positive aercu_opt=1/2 is opt-in and requires cu_physics=11. Public NASA forcing is written from papers, not bitwise to WRF. Fast real-forecast physics stage costs are 3.6864x/3.6830x aercu_opt=0 on RTX PRO 6000 Blackwell (selectors 1/2); see docs/public-cumulus-aerosol.md for the case, timing conditions and approximation limits.']
    registry['parameters']['aercu_opt']['warnings']=list(mp40['warnings'])
    for component, option_id in (('cumulus', 'multiscale-kf'), ('microphysics', 'morrison-aero-mp40')):
        registry['components'][component]['options'][option_id].setdefault('extensions', {})['preset_scope_reason'] = (
            'Column reference evidence is available, but no forecast or observation score '
            'supports recommending a preset; public component overrides select this option.')
    registry['components']['cumulus']['options']['multiscale-kf']['extensions']['preset_scope_reason'] = (
        'One three-hour native/GPU forecast and ASOS temperature comparison is recorded. '
        'General skill is not qualified. Whole forecast stage costs 3.030382x CU1; '
        'the separate column-kernel deck costs 6.6714x. Positive public aerosol '
        'forcing costs 3.6864x/3.6830x aerosol off. Explicit component override '
        'only; never select this option by default.')



def _misc_rcon_component(registry):
    options = registry["components"]["microphysics"]["options"]
    options["rcon-mp29"] = {
        "label": "RCON / MP29", "selectors": {"mp_physics": 29},
        "implemented": True, "maturity": "implemented-unverified", "scientific_evidence": "none",
        "parameters": {"moist": True, "moist_cq": True}, "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True},
                        "required_settings_reasons": {"moist": "RCON advances vapor, five condensate classes and number moments."}},
        "extensions": {"preset_scope_reason": "Strict replay and a three-hour forecast pass. The measured 1.885431x MP28 GPU stage cost exceeds the speed gate; RCON-SPEED-01 refuses forecast admission until a qualified handoff."},
        "warnings": ["Requires the native initialized RCON table stream, validated by size and SHA-256. Set WOOF_RCON_TABLE_ROOT to its directory. Strict real-input replay: 0/25246820 words, no exclusions. A three-hour forecast passes, but measured stage cost is 1.885431x MP28; the native 1.053748x profile does not justify the GPU excess. Forecast admission is refused by RCON-SPEED-01 until speed qualification."]}


def _misc_udm_component(registry):
    registry["components"]["microphysics"]["options"]["udm-mp27"] = {
        "label": "UDM / MP27", "selectors": {"mp_physics": 27},
        "implemented": True, "maturity": "implemented-unverified", "scientific_evidence": "none",
        "parameters": {"moist": True, "moist_cq": True}, "asset_requirements": [],
        "constraints": {"required_settings": {"moist": True},
            "required_settings_reasons": {"moist": "UDM advances vapor, six condensate classes and three number fields."}},
        "extensions": {"preset_scope_reason": "Retained WRF 4.7.1 driver replay, whole forecast and interleaved speed evidence are recorded in misc-wrf471.json."},
        "warnings": ["UDM retains WRF 4.7.1 initialized tables and native REAL kinds. Use legacy RRTMG; the native optional radius routine receives uninitialized density and is not qualified."]}


def _misc_qualified_components(registry):
    """Retained real-input WRF 4.7.1 receipts, without an observation-skill claim."""
    receipt = json.loads((MODEL / "docs/public/receipts/misc-wrf471.json").read_text())
    options = registry["components"]["microphysics"]["options"]
    if "etanew-mp95" not in options:
        options["etanew-mp95"] = json.loads((MODEL / "tools/data/etanew_option.json").read_text())
    _misc_rcon_component(registry)
    _misc_udm_component(registry)
    for component, option_id, key in (("microphysics", "etanew-mp95", "mp95"),
                                       ("microphysics", "udm-mp27", "mp27"),
                                       ("surface_layer", "ideal-scm", "sf89")):
        option = registry["components"][component]["options"][option_id]
        proof = receipt["rows"][key]
        option["maturity"] = "supported"
        option["scientific_evidence"] = "none"
        option["verification_scope"] = "retained-real-native-module-replay-and-real-forecast"
        option.setdefault("extensions", {})["wrf471_qualification"] = {
            **proof, "receipt": "docs/public/receipts/misc-wrf471.json"}
        option["extensions"]["preset_scope_reason"] = (
            "A stock namelist or component override selects this option. "
            "Retained native replay and a real forecast do not establish "
            "observation skill for a recommended preset.")
        option["warnings"] = [
            f"Strict WRF 4.7.1 real-input replay: 0/{proof['words']} words, 32 captured calls, no exclusions. "
            f"Three-hour fast forecast and inspected Rust maps pass. Same-card interleaved stage cost "
            f"{proof['stage_ratio']:.3f}x {proof['nearest']}. Retained capture manifest and recipes are named "
            "in docs/public/misc-wrf471.md. This does not establish forecast skill against observations."]
    options[FERRIER_OPTION_ID]["extensions"].pop("family_siblings_refused", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registry", type=pathlib.Path, default=REGISTRY_PATH,
        help="the registry to transform; defaults to the tracked file. The "
             "tables below own only part of it, so the rest is carried "
             "through from here unchanged")
    parser.add_argument(
        "--out", type=pathlib.Path, default=REGISTRY_PATH,
        help="where to write the registry; defaults to the tracked file, and "
             "a test points it at a temporary path to compare bytes without "
             "touching the tree")
    parser.add_argument(
        "--export-out", type=pathlib.Path, default=None,
        help="where to write the consumer export the Rust crates read; "
             "defaults to woof/physics_consumer_export_v1.json beside the "
             "tracked registry, or beside --out when that names another "
             "directory")
    parser.add_argument("--check", action="store_true",
                        help="compare both generated documents without writing them")
    from woof.ingest.preparation_workers import cpu_budget
    parser.add_argument("--jobs", type=int,
                        default=cpu_budget()["available_cpus"],
                        help="citation parsing workers; defaults to affinity and CPU quota")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    from woof.physics_registry import REGISTRY_REBUILD_ENV
    # The consumer modules this build pulls facts from assert agreement
    # with the registry ON DISK at import, and the disk copy is the one
    # being replaced; the flag tells them so for this process only.
    os.environ[REGISTRY_REBUILD_ENV] = "1"

    preload_sources(args.jobs)
    registry = build(json.loads(args.registry.read_text(encoding="utf-8")))
    if args.check:
        export_path = args.export_out or (
            CONSUMER_EXPORT_PATH if args.out == REGISTRY_PATH
            else args.out.parent / CONSUMER_EXPORT_PATH.name)
        documents = ((args.out, render(registry)),
                     (export_path, render_consumer_export(registry)))
        stale = [str(path) for path, expected in documents
                 if not path.is_file() or path.read_bytes() != expected]
        for path in stale:
            print(f"DRIFT: {path}; regenerate with tools/build_registry.py")
        if not stale:
            print("registry and consumer export reproduce")
        return 1 if stale else 0
    params = registry["parameters"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(registry))
    export_out = args.export_out
    if export_out is None:
        export_out = (CONSUMER_EXPORT_PATH if args.out == REGISTRY_PATH
                      else args.out.parent / CONSUMER_EXPORT_PATH.name)
    export_out.write_bytes(render_consumer_export(registry))
    print("parameters:", len(params),
          "| implemented:", sum(1 for s in params.values()
                                if s.get("implemented") is not False),
          "| unimplemented:", sum(1 for s in params.values()
                                  if s.get("implemented") is False),
          "|", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
