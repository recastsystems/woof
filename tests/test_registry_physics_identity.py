"""A physics selection binds the registry's physics, not its prose (A153).

A citation-only commit to ``woof/physics_registry_v2.json`` (the Shin-Hong
``kpbl < kte`` guard re-pinned from :1371 to :1429, the LES ``km_opt``
headers from dycore.py:1041/:1131 to :1147/:1237) moved the registry
digest every selection receipt bound, and every preparation made before it
was refused as "physics selection differs" while its forecast was byte
identical.  These tests hold both halves of the fix: documentation and
admission edits leave a selection's identity alone, and a real change to
an option, a knob, a table or a kernel identity still moves it and is
named.  A configuration an admission edit no longer admits is refused by
the preflight's recomputation, naming the rule.  Physics added off by
default (a component, option, knob or template entry a preparation's
registry lacked) resolves where the configuration leaves it off and is
named where the configuration selects it.
"""
from __future__ import annotations

from copy import deepcopy
import dataclasses
import json
from pathlib import Path
import re
import sys

import pytest

from woof import physics_registry as registry_module
from woof.physics_compat import (
    SELECTION_RECEIPT_RECORD_ONLY_FIELDS,
    THOMPSON_RTE_RRTMGP_PROFILE_ID,
    PhysicsCapabilityError,
    physics_selection_differences,
    single_domain_physics_selection,
    single_domain_runtime_switches,
    validate_physics_capabilities,
    validate_single_domain_physics_profile,
)
from woof.config import RunConfig
from woof.physics_registry import (
    DEFAULT_TEMPLATE_ID,
    REGISTRY_ADMISSION_CONSTRAINTS,
    REGISTRY_BLOCKS_OUTSIDE_PHYSICS_IDENTITY,
    REGISTRY_CONSTRAINTS_KEPT_IN_PHYSICS_IDENTITY,
    REGISTRY_DOCUMENTATION_FIELDS,
    REGISTRY_DOCUMENTATION_VALUE_MAPS,
    REGISTRY_EVIDENCE_RECORDS,
    component_off_option,
    physics_registry,
    recorded_registry_physics_parts,
    registry_knob_readers,
    registry_off_template,
    registry_physics_history,
    registry_physics_part_sources,
    registry_physics_parts,
    registry_physics_sha256,
    registry_sha256,
    setting_off_value,
    strip_registry_documentation,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_2492999CD = (ROOT / "tests" / "data" / "registry_physics"
                      / "receipts-2492999cd.json")

#: A shipped single-domain profile (the default template is a tree
#: product, not one).  YSU over the classic MM5 surface layer.
PROFILE = THOMPSON_RTE_RRTMGP_PROFILE_ID

#: The registry document 2.8.0 shipped, which 2492999cd still carried.
DOCUMENT_280 = (
    "378ade8eaa2e5cb6776141198d551b90daba1b5e987b3a64fbe2d58d9fdf3459")

#: The two citation-only documents after it (A153's two cases).
DOCUMENTS_CITATION_ONLY_SINCE_280 = (
    "a0c16909afa677e3e250864a1e2cb94f2d3a3206e21a5c926e0dc90c4a47e907",
    "d261e818de7bceff7c62e497a53106fc4f26ee61f5c891bba82ed9ed64ca4294",
)

#: Every physics part the registry changed since 2.8.0, with its cause.
#: Its citations also moved, the UW PBL became a new partner for GF and
#: the MYNN surface layer, and every template gained the urban
#: component's ``none`` option; none of that is physics.
_URBAN = (
    "the urban canopy models (sf_urban_physics), a component 2.8.0 did not "
    "have, off at its none option in every template (f21eedce3)")
#: Every scheme that finishes through woof.core.microphysics.
#: moist_physics_finish moved its restart_algorithm_identity consumer row
#: in 2.8.8 for the same cause.
_REAL_CLAMP = (
    "Its restart_algorithm_identity consumer row also advanced (2.8.8, "
    "gfix/288-mp-clamp): the shared microphysics finish clamps the heating "
    "at WRF's REAL product mp_tend_lim*dt, as classic Thompson already did")
PHYSICS_CHANGES_SINCE_280 = {
    "parameters.icloud": (
        "icloud2-binary-cldfra (golden/ports-rad): icloud gains WRF's binary "
        "cloud fraction (2); every earlier value keeps its meaning"),
    "components.radiation.options.new-goddard": (
        "ra5-goddard (golden/ports-rad): a new radiation option, WRF's 2017 "
        "Goddard pair (ra_lw/ra_sw_physics = 5); no earlier option changes"),
    **{f"parameters.{key}": (
        "aer-opt2-aod (golden/ports-rad): a user-aerosol control, read only "
        "under aer_opt = 2") for key in (
            "aer_type", "aer_aod550_opt", "aer_angexp_opt", "aer_ssa_opt",
            "aer_asy_opt", "aer_aod550_val", "aer_angexp_val", "aer_ssa_val",
            "aer_asy_val", "aer_auxinput15", "aer_auxinput15_interval")},
    "components.cumulus.options.grell-freitas": (
        "the intake carries the strict GF normalization and real downdraft origin; "
        "previous GF restart identities are refused for this declared implementation change"),
    "components.cumulus.options.tiedtke": (
        "a new cumulus option, WRF's original Tiedtke (cu_physics = 6); "
        "no earlier option or receipt changes"),
    "components.land_surface.options.noah-mp": (
        "corrected default night snow aging changes TAUSS and the land-surface "
        "restart identity; explicit wrf_461 preserves daylight-only aging"),
    "parameters.solar_generation": (
        "a new numerical generation selector, default wrf_480 (WRF 4.8 "
        "calc_coszen: julian - 0.5, 0.040849); wrf_461 reproduces 2.8.9"),
    "parameters.ysu_version": (
        "a new numerical generation selector, default wrf_480 (WRF 4.8 YSU "
        "capped top-down entrainment); wrf_461 reproduces 2.8.9's kernel"),
    "parameters.bl_mynn_tkeadvect": (
        "WRF's MYNN TKE advection switch is admitted (true advects QKE_ADV "
        "through the scalar transport); the default false is unchanged"),
    "parameters.scalar_adv_opt": (
        "WRF's scalar-array advection selector is published at its only "
        "implemented value 1, which every earlier run used"),
    "parameters.noahmp_snow_age": (
        "a new numerical generation selector, default wrf_470, which ages snow "
        "at night; old configurations without the key change generation"),
    **{f"parameters.{name}": "Explicit urban numerical generation and pinned table controls"
       for name in ("urban_ucm_generation", "urban_treeoption", "urban_ttscheme",
                    "urban_fvg", "urban_table_dir", "urban_water_balance")},
    "components.cumulus.options.new-sas": "New SAS with built-in NSCV is a new selectable scheme",
    "parameters.nsas_dx_factor": "New SAS distance-dependent trigger option",
    "parameters.sas_pgcon": "SAS convective pressure-gradient factor",
    "templates.wsm6-ysu-mm5-noah-nsas-rrtmg-legacy-v1": "New named NSAS composition",

    "components.microphysics.options.ferrier-aligo-mp5": (
        "A newly implemented native microphysics selector and its registry-bound consumers."),
    "components.microphysics.options.wsm3-mp3": (
        "new WSM3 column scheme, native selector 3, off in previous suites"),
    "components.microphysics.options.wsm7-mp24": (
        "new WSM7 execution option, with its own restart and consumer declarations"),
    **{f"parameters.{key}": "new passive lightning diagnostic control, off by default" for key in ("cellcount_method", "cldtop_adjustment", "do_radar_ref", "flashrate_factor", "iccg_method", "iccg_prescribed_den", "iccg_prescribed_num", "lightning_dt", "lightning_option", "lightning_start_seconds")},
    "components.pbl.options.ccpp-tke-edmf": (
        "the pinned upstream CCPP closure is a new non-WRF selector"),
    "templates.ccpp-tke-edmf-thompson-mm5-noah-gf-rte-v1": (
        "a new unverified composition makes the CCPP closure reachable by name"),
    "templates.thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1": (
        "a new named configuration selects the prescribed monthly surface "
        "fields and fractional sea ice; earlier templates keep their settings"),
    "templates.thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1": (
        "a new named configuration adds sun-angle albedo to the monthly "
        "surface fields; earlier templates keep their settings"),
    "components.pbl.options.mynn": (
        "mixing length 2 and local scalar diffusion are implemented; "
        "cold initialization now passes vapor to the WRF moments solve"),
    "parameters.bl_mynn_mixlength": (
        "the implemented choices are WRF mixing lengths 1 and 2"),
    "parameters.scalar_pblmix": (
        "WRF post-PBL local diffusion for MYNN aerosol number scalars, "
        "off by default"),
    **{f"components.microphysics.options.{option}": (
        "new analyzed-aerosol controls are forbidden under this package; "
        "exact prior option declarations remain compatible at admitted old "
        "settings. " + _REAL_CLAMP)
       for option in ("milbrandt2mom-mp9", "morrison-mp10", "nssl2-mp18")},
    "components.microphysics.options.fast-sbm-mp30": (
        "a new option: WRF FAST_SBM spectral bin microphysics (mp_physics 30); "
        "no earlier selection names it"),
    "components.microphysics.options.thompson-mp8": (
        "new analyzed-aerosol controls are forbidden under this package; "
        "exact prior option declarations remain compatible at admitted old "
        "settings. Its restart_algorithm_identity consumer row also changed: "
        "classic Thompson reads rain collecting graupel from the one slab WRF "
        "builds and clamps its heating at the REAL product (v5, "
        "gfix/288-thompson-classic), so a 2.8.0 mp=8 receipt is refused for it"),
    "components.microphysics.options.p3-mp50": (
        "new analyzed-aerosol controls are forbidden under this package; "
        "exact prior option declarations remain compatible at admitted old "
        "settings. Its radar_da consumer row also changed: P3 gained a "
        "scheme-diagnostic reflectivity operator and one clear-air value "
        "(acb8faf25, audit S14), so a 2.8.0 P3 receipt is refused for it. "
        + _REAL_CLAMP),
    "parameters.aer_init_opt": (
        "first-guess aerosol source value 2 is admitted with explicit analyzed "
        "selection; prior values retain their parameter-declaration identity"),
    "parameters.mp28_aerosol_source": (
        "analysis requires source aerosol on every frame; the previous three "
        "choices retain their parameter-declaration identity"),
    "parameters.use_rap_aero_icbc": (
        "analyzed aerosol input with operational monthly surface emission, "
        "new and disabled by default"),
    "parameters.moist_cq": (
        "moisture pressure correction now defaults on whenever a vapor "
        "state exists, including passive vapor with microphysics off"),
    "components.cumulus.options.kain-fritsch": (
        "the CUDA column became a line-for-line port of WRF 4.6.1 "
        "KF_eta_PARA, graded at 0 ULP by tools/kf_wrf461_oracle; its restart "
        "algorithm identity moved to kain-fritsch-wrf461-kfeta-port-v1, so a "
        "2.8.0 receipt that selected Kain-Fritsch is refused for it"),
    "components.microphysics.options.off": (
        "passive vapor retains WRF's moisture pressure correction; dry "
        "states bypass the enabled policy"),
    "components.microphysics.options.kessler-mp1": (
        "Kessler profiles apply WRF's moisture pressure correction "
        "instead of the retired launch guard. " + _REAL_CLAMP),
    "components.microphysics.options.wsm6-mp6": (
        "WSM6 profiles apply WRF's moisture pressure correction "
        "instead of the retired launch guard. " + _REAL_CLAMP),
    "components.microphysics.options.tempo": (
        "a new named microphysics scheme (TEMPO, selected as \"tempo\", "
        "WOOF id 900); every earlier option keeps its settings"),
    "templates.p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1": (
        "its explicit CQ-on setting now equals the enabled default, "
        "so template identity normalization removes that redundant pin"),
    "parameters.diff_opt": (
        "coordinate-surface and terrain-aware mixing are implemented "
        "choices at 1 and 2; default 2 retains the prior route"),
    "parameters.mix_full_fields": (
        "the WRF logical is retained under coordinate diffusion; that "
        "operator mixes theta relative to its initial field for either value"),
    "parameters.topo_wind": (
        "YSU terrain-wind correction, implemented at options 1 and 2, "
        "off at 0 (12d03dd6c)"),
    "parameters.gwd_opt": (
        "orographic drag, implemented at options 1 and 3, off at 0 "
        "(12d03dd6c)"),
    "components.pbl.options.uw": (
        "the UW PBL, a new option no 2.8.0 suite selects (51693ec76)"),
    "components.microphysics.options.wdm6-mp16": (
        "WDM6's restart algorithm identity v4: rain with no rain number no "
        "longer evaporates through the condensation cap, which changes "
        "WDM6 forecasts (A144, b814c65a0). " + _REAL_CLAMP),
    "components.urban": _URBAN,
    "components.urban.options.none": _URBAN + "; none runs nothing",
    "components.urban.options.slucm": _URBAN + "; the single-layer UCM",
    "components.urban.options.bep": _URBAN + "; BEP",
    "components.urban.options.bep-bem": _URBAN + "; BEP+BEM",
    "parameters.num_urban_hi": (
        "an urban knob only the three urban canopy options read "
        "(f21eedce3)"),
    "parameters.use_wudapt_lcz": (
        "an urban knob only the three urban canopy options read "
        "(f21eedce3)"),
    "parameters.fractional_seaice": (
        "WRF's sea-ice threshold select, declared but not implemented at "
        "2.8.0, now implemented for the RUC LSM and off at 0, the 0.5 "
        "threshold every earlier RUC run used (lane/286-veg-albedo-seaice)"),
    "parameters.slope_rad": (
        "slope-aware radiation, declared but not implemented at 2.8.0, now "
        "implemented and off at 0 (the namelist gaps, cb8af5ccb)"),
    "parameters.topo_shading": (
        "terrain shading, declared but not implemented at 2.8.0, now "
        "implemented and off at 0 (the namelist gaps, cb8af5ccb)"),
    "parameters.zadvect_implicit": (
        "WRF's implicit-explicit vertical advection, a knob 2.8.0 did not "
        "have (A158), off at 0, registered with its w solve's declared "
        "divergence from WRF v4.7.1 (A179)"),
    "parameters.zadvect_implicit_variant": (
        "the older WRF implicit split and current-mass coefficients are "
        "selectable; the wrf_471 default keeps the existing operator"),
    "parameters.sf_surface_mosaic": (
        "Noah mosaic land use, declared but not implemented at 2.8.0, now "
        "implemented and off at 0 (the Noah mosaic lane, fa2e5efd8)"),
    "parameters.mosaic_cat": (
        "the Noah mosaic tile count, a knob 2.8.0 did not have, read only "
        "with sf_surface_mosaic = 1, scoped to land_surface by component_id "
        "and read_when, and off at 3 (fa2e5efd8)"),
    "parameters.mosaic_urban_canopy": (
        "where Noah mosaic runs the urban canopy, a knob 2.8.0 did not "
        "have, off at WRF's rule \"dominant\" (c517106a8)"),
    "parameters.diff_opt": (
        "WRF's horizontal diffusion form, a knob 2.8.0 did not have "
        "(lane/282-namelist-tolerance, 679a5f5fe): 1 selects coordinate-"
        "surface diffusion for km_opt 2 and 4, and it is off at 2, the "
        "metric form every earlier build ran"),
    "components.land_surface.options.ruc-lsm": (
        "a LAKEMASK column is bypassed only when the lake model is selected "
        "(module_sf_ruclsm.F:824), so with sf_lake_physics = 0 it runs the "
        "water branch instead of never being advanced; the option's restart "
        "algorithm identity advanced to v2 with it, to v3 with the SOILPROP "
        "lineage default (ruc_soilprop) and to v4 with the irrigation "
        "default (ruc_irrigation), whose admitted names wrf_45 and wrf_461 "
        "the option declares, as it declares ruc_qvg_cold_start's air and "
        "wrf and ruc_2m_diagnostic's flux and log_profile; v5 with the "
        "snow default (ruc_snow), whose admitted names are wrf_45 and "
        "wrf_461; v6 binds the restored generic irrigation and snow "
        "defaults, wrf_461, so a v5 header which omitted wrf_45 cannot "
        "silently resume as wrf_461; v7 (2.8.8) binds WOOF's libm words in "
        "the column and the fractional_seaice gate on the ice blend"),
    "components.pbl.options.ysu": (
        "YSU's restart algorithm identity v2 (2.8.8): no FMA contraction, "
        "WOOF's float32 pow/exp, the driver's theta, the double Richardson "
        "sign test and WRF's surface-drag form change YSU forecasts, so a "
        "checkpoint of the earlier code may not resume onto it"),
    "components.pbl.options.shinhong": (
        "Shin-Hong's restart algorithm identity v2 (2.8.8): no FMA "
        "contraction and WOOF's float32 pow/exp change its forecasts"),
    **{f"components.pbl.options.{option}": (
        "the pbl.scalar_diffusion_exchange consumer row (golden/exact "
        "83a6116b2f): scalar_pblmix = 1 runs WRF's post-PBL diff4d after "
        "this scheme, or is WRF's no-op for it, where the scheme was refused "
        "before; scalar_pblmix = 0 forecasts are unchanged")
       for option in ("myj", "sase", "off")},
    **{f"components.surface_layer.options.{option}": (
        "the surface layer's restart algorithm identity v2 (2.8.8): the "
        "sfclay, Eta and MYNN surface units compile without FMA contraction "
        "and keep denormals on WOOF's libm words, which changes their "
        "forecasts")
       for option in ("revised-mm5", "eta-similarity", "mynn", "classic-mm5")},
    "components.microphysics.options.thompson-aerosol-mp28": (
        "aerosol-aware Thompson's restart algorithm identity v2 (2.8.8): "
        "WRF's arithmetic on WOOF's libm words, the 273.15 K melting fix "
        "and the rain-graupel table slab change its forecasts. "
        + _REAL_CLAMP),
    "parameters.mynn_sfclay_variant": (
        "the MYNN surface layer's generation, a knob 2.8.0 did not have: "
        "the default wrf_461 is the WRF v4.6.1 form every earlier build "
        "ran; gsl_wrf39 selects the GSL WRF 3.9 fork's form by name"),
    "parameters.ruc_soilprop": (
        "LSMRUC SOILPROP's soil-water lineage, a knob 2.8.0 did not have: "
        "the default wrf_45 diffusivity over the moisture above the residual "
        "replaces the WRF v4.6.1 form over total porosity, which stays "
        "selectable as wrf_461; every RUC configuration changes answers"),
    "parameters.thompson_version": (
        "the aerosol-aware scheme's source generation; wrf_461 preserves "
        "the previous default and wrf_39_noaa selects the fork's own tables "
        "and process rules"),
    "parameters.thompson_fork_snow_fall": (
        "the fork's singular melting-snow fall speed; the default blend "
        "keeps the later rain-share form and the fork defect is explicit"),
    "parameters.bl_mynn_cloud_tendency_form": (
        "paired source cloud conservation defects, explicitly selected and off at wrf_461"),
    "parameters.cycling": (
        "WRF's cycled start, a knob 2.8.0 did not have: false (default) "
        "cold-starts the MYNN; true keeps the start's QKE, QC_BL and "
        "CLDFRA_BL in the gsd_41 MYNN, as the operational fork does"),
    "parameters.rrtmg_aer3_type_source": (
        "the aer_opt=3 shortwave aerosol type, a knob 2.8.0 did not have: "
        "rural (default) is type 1 everywhere as every earlier build ran; "
        "landmask is the NOAA MPAS fork's rural-over-land, maritime-over-water"),
    "parameters.bl_mynn_version": (
        "the MYNN generation, a knob 2.8.0 did not have: wrf_461 (default) "
        "is WRF v4.6.1, gsd_41 the GSD MYNN v4.1 of the NOAA-EMC WRF 3.9 "
        "branch, selected by a WRF 3.x MYNN namelist"),
    "parameters.bl_mynn_gsd41_unsquared_qtke": (
        "the gsd_41 option-2 TKE conversion as written, a knob 2.8.0 did "
        "not have, off at false"),
    "templates.thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1": (
        "the explicit GSD v4.1 PBL, GSL surface, aerosol Thompson and "
        "legacy radiation composition, a template 2.8.0 did not have; "
        "generic compositions keep their original switch sets"),
    "parameters.mosaic_lu": (
        "RUC fractional land-use mosaic, implemented and off at 0"),
    "parameters.mosaic_soil": (
        "RUC fractional soil mosaic, implemented and off at 0"),
    "parameters.sf_lake_physics": (
        "WRF's CLM lake model, a knob 2.8.0 did not have, off at 0"),
    "parameters.use_lakedepth": (
        "CLM lake bathymetry switch, read only with sf_lake_physics = 1"),
    "parameters.lakedepth_default": (
        "CLM lake default depth, read only with sf_lake_physics = 1"),
    "parameters.lake_min_elev": (
        "CLM lake minimum elevation, read only with sf_lake_physics = 1"),
    "parameters.sf_ocean_physics": (
        "WRF's 1-D ocean mixed layer, a knob 2.9.0 did not have, off at 0"),
    "parameters.oml_hml0": (
        "OML initial mixed-layer depth, read only with sf_ocean_physics = 1"),
    "parameters.oml_gamma": (
        "OML deep-water lapse rate, read only with sf_ocean_physics = 1"),
    "parameters.oml_relaxation_time": (
        "OML relaxation time, read only with sf_ocean_physics = 1"),
    "parameters.mix_full_fields": (
        "WRF's full-field mixing logical, a knob 2.8.0 did not have "
        "(679a5f5fe), off at true, the full-field mixing every earlier "
        "build ran"),
    "parameters.spp_conv": (
        "WRF's stochastic parameter perturbation for Grell-Freitas, a knob "
        "2.8.0 did not have, off at 0"),
    "parameters.spp_pbl": (
        "WRF's stochastic parameter perturbation for MYNN, a knob 2.8.0 did "
        "not have, off at 0"),
    "parameters.diff_6th_form": (
        "the sixth-order filter's source form, a knob 2.8.0 did not have, "
        "at wrf_461, the filter every earlier build ran"),
    "parameters.upper_wind_limiter_form": (
        "the NOAA WRFV3.9 saved-wind limiter, absent from earlier builds, "
        "off under wrf_461"),
    "parameters.diff_6th_factor2": (
        "the NOAA WRFV3.9 fork's moisture filter factor, a knob 2.8.0 did "
        "not have, unset and refused outside the fork form"),
    "parameters.mp_zero_out": (
        "WRF's post-microphysics zero-out, a knob 2.8.0 did not have, off "
        "at 0"),
    "parameters.mp_zero_out_thresh": (
        "WRF's zero-out threshold, read only with mp_zero_out > 0"),
    "parameters.mp_zero_out_all": (
        "WRF v4.6.1's zero-out array switch, read only with mp_zero_out > 0"),
    "parameters.v_sca_adv_order": (
        "WRF's vertical scalar advection order (lane/286-vadv5), a knob "
        "2.8.0 did not have, at 3 the ladder every earlier build ran"),
    "parameters.v_mom_adv_order": (
        "WRF's vertical momentum advection order (lane/286-vadv5), a knob "
        "2.8.0 did not have, at 3 the ladder every earlier build ran"),
    "parameters.h_mom_adv_order": (
        "WRF's horizontal momentum advection order (lane/286-vadv5), "
        "declared at 5, the flux5 stencil every earlier build ran"),
    "parameters.ruc_snow": (
        "the RUC snow scheme's lineage, a knob 2.8.0 did not have: the "
        "generic default wrf_461 keeps the v4.6.1 form every earlier build "
        "ran; the HRRR namelist importer and recipe select the WRF v4.0-4.5 "
        "form wrf_45 explicitly"),
    "parameters.ruc_2m_diagnostic": (
        "SFCDIAGS_RUCLSM's 2 m form, a knob 2.8.0 did not have, at public "
        "WRF's flux form by default; log_profile adds the operational "
        "RAP/HRRR branch's logarithmic block by name"),
    "parameters.ruc_qvg_cold_start": (
        "LSMRUC's QVG cold start, a knob 2.8.0 did not have, at public "
        "WRF's form by default; air starts the ground vapour from the "
        "lowest-level air as the operational RAP/HRRR branch's fallback "
        "does, by name"),
    "parameters.ruc_irrigation": (
        "LSMRUC's irrigation rule by WRF lineage, a knob 2.8.0 did not "
        "have: the generic default wrf_461 keeps the WRF v4.6.1 per-step "
        "relaxation; the HRRR namelist importer and recipe select the "
        "crop-fraction-scaled floor wrf_45 explicitly"),
    "parameters.swint_opt": (
        "WRF's surface shortwave interpolation between radiation calls "
        "(lane/286-aer-swint), implemented at 1; 0 holds the radiation "
        "call's fluxes, as every earlier build did"),
    "parameters.rrtmg_cloud_optics_form": (
        "the legacy RRTMG cloud wrapper source form; wrf_461 is the "
        "wrapper every earlier build ran, noaa_wrf39 is source-selected"),
    "parameters.rrtmg_smoke_manifest": (
        "prescribed source smoke profiles bind their manifest and member "
        "contents; empty remains the prior radiation input path"),
    "parameters.aer_opt": (
        "WRF's radiation aerosol input (lane/286-aer-swint), implemented at "
        "3 for the legacy RRTMG shortwave; 0 runs zero aerosol, as every "
        "earlier build did"),
    "parameters.alb_sol": (
        "sun-angle-dependent land albedo, implemented at 1; 0 preserves "
        "the albedo carrier every earlier build used"),
    "components.cumulus.options.new-tiedtke": (
        "2.9.1 donor bound (hexeng/hex040-fix-global-ysu): default arithmetic "
        "scales a column's explicit rates so no layer loses more water than it "
        "holds over the step; its restart algorithm identity moved to "
        "new-tiedtke-wrf461-cumastrn-donor-bound-v2 (strict keeps v1), so a "
        "2.8.0 receipt that selected New Tiedtke is refused for it"),
    "components.surface_layer.options.eta-similarity": (
        "the Eta similarity surface layer's restart algorithm identity v2: "
        "its tables, terrain-height seed, LOG/EXP/** and FMA-free "
        "arithmetic are now WRF v4.6.1's word for word "
        "(lane/verify-eta-similarity-sfclay, "
        "tests/test_myjsfc_wrf461_parity.py), which moves MYJ forecasts in "
        "their last bits"),
}

_WDM6_SUITE = "wdm6-mp16-ysu-mm5-noah-grell-freitas-rte-rrtmgp-v1"

#: What each 2.8.0 receipt is refused for at this head, by (profile,
#: spelling).  Written out rather than derived from what changed since
#: 2.8.0, so a false refusal cannot pass by being derived: an entry here
#: is a real physics change, named with its cause in
#: :data:`PHYSICS_CHANGES_SINCE_280`.
EXPECTED_280_REFUSALS: dict[tuple[str, str], list[str]] = {
    (row["profile"], spelling): sorted([
        "registry physics of parameters.moist_cq (changed)",
        # golden/ports-rad: icloud gains WRF's binary cloud fraction (2).
        "registry physics of parameters.icloud (changed)",
        *([f"registry physics of components.microphysics.options.{option} "
           "(changed)"] if (option := {
               1: "kessler-mp1", 6: "wsm6-mp6", 8: "thompson-mp8",
               9: "milbrandt2mom-mp9", 10: "morrison-mp10",
               16: "wdm6-mp16", 18: "nssl2-mp18", 50: "p3-mp50",
           }.get(row["switches"]["mp_physics"])) else []),
        *(["registry physics of templates.p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1 "
           "(changed)"] if spelling == "named" and
          row["switches"]["mp_physics"] == 50 else []),
        *(["registry physics of components.pbl.options.mynn (changed)",
           "registry physics of parameters.bl_mynn_mixlength (changed)"]
          if row["switches"].get("bl_pbl_physics") == 5 else []),
        *(["registry physics of components.cumulus.options.kain-fritsch "
           "(changed)"] if row["switches"].get("cu_physics") == 1 else []),
        *(["registry physics of components.cumulus.options.new-tiedtke "
           "(changed)"] if row["switches"].get("cu_physics") == 16 else []),
        *(["registry physics of components.pbl.options.ysu (changed)"]
          if row["switches"].get("bl_pbl_physics") == 1 else []),
        *(["registry physics of components.pbl.options.shinhong (changed)"]
          if row["switches"].get("bl_pbl_physics") == 11 else []),
        # golden/exact 83a6116b2f: the scalar_diffusion_exchange consumer row.
        *([f"registry physics of components.pbl.options.{pbl} (changed)"]
          if (pbl := {0: "off", 2: "myj", 900: "sase"}.get(
              row["switches"].get("bl_pbl_physics"))) else []),
        *([f"registry physics of components.surface_layer.options.{layer} "
           "(changed)"] if (layer := {
               1: "revised-mm5", 2: "eta-similarity", 5: "mynn",
               91: "classic-mm5",
           }.get(row["switches"].get("sf_sfclay_physics"))) else []),
        *(["registry physics of components.microphysics.options."
           "thompson-aerosol-mp28 (changed)"]
          if row["switches"]["mp_physics"] == 28 else []),
        *(["registry physics of components.cumulus.options.grell-freitas (changed)"]
          if row["switches"].get("cu_physics") == 3 else []),
        *(["registry physics of components.cumulus.options.kain-fritsch "
           "(changed)"] if row["switches"].get("cu_physics") == 1 else []),
        *(["registry physics of components.land_surface.options.ruc-lsm "
           "(changed)",
           "registry physics of parameters.mosaic_lu (changed)",
           "registry physics of parameters.mosaic_soil (changed)"]
          if row["switches"].get("sf_surface_physics") == 3 else []),
        *(["registry physics of components.land_surface.options.noah-mp (changed)"]
          if row["switches"].get("sf_surface_physics") == 4 else []),
        *(["registry physics of components.surface_layer.options."
           "eta-similarity (changed)"]
          if row["switches"].get("sf_sfclay_physics") == 2 else []),
    ])
    for row in json.loads(RECEIPTS_2492999CD.read_text())["receipts"]
    for spelling in ("named", "unnamed")
}

# The recorded inputs remain CQ-off. These names cannot be recomputed as
# their corrected CQ-on profiles; unnamed explicit controls still resolve.
CQ_OFF_280_NAMED_PROFILES = frozenset({
    "wsm6-ysu-mm5-noah-no-radiation-v1",
    "kessler-mp1-ysu-mm5-noah-dudhia-v1",
    "wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1",
    "wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1",
    "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1",
    "wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1",
    "wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
    "wsm6-ysu-mm5-noahmp-no-radiation-expert-only-v1",
    "wsm6-mynn-mynn-noahmp-no-radiation-expert-only-v1",
    "wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1",
    "20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1",
    "wsm6-sase-revised-mm5-noah-closure-supplied-v1",
    "wsm6-pbl-off-mm5-noah-tke-1-5-order-v1",
    "wsm6-pbl-off-mm5-noah-smagorinsky-3d-v1",
    "wsm6-pbl-off-mm5-noah-constant-k-v1",
})


#: Recast WOOF: a part whose physics (documentation stripped) names this
#: package -- a consumer row's class path ``woof.core.*``, an asset root
#: ``woof_data/...`` -- holds the engine's ``gpuwm`` spelling in a 2.8.0
#: receipt, so against this tree the receipt names it changed.
_WOOF_PACKAGE = re.compile(r"\bwoof[._/]|woof_data/")


def _woof_renamed_parts(row, spelling) -> set[str]:
    """The receipt's selected parts that differ from 2.8.0 only by the package name."""
    try:
        selection = _expected_for(row, spelling)
    except ValueError:
        return set()
    renamed = {name for name, source in registry_physics_part_sources().items()
               if _WOOF_PACKAGE.search(json.dumps(strip_registry_documentation(source)))}
    return {f"registry physics of {name} (changed)"
            for name in selection["registry_physics"]["parts"] if name in renamed}


def _legacy_expected(row, spelling, *additional):
    """Known default correction plus the mutation an individual test makes."""
    return sorted({*EXPECTED_280_REFUSALS[(row["profile"], spelling)],
                   *_woof_renamed_parts(row, spelling), *additional})


def _loaded_config(switches) -> RunConfig:
    """The configuration a TOML stating ``switches`` loads to at this build:
    every setting it omits (the urban selector and knobs, slope_rad,
    topo_shading, shadlen among them for a 2.8.0 TOML) at RunConfig's
    default, which is the object the real preflight recomputes from."""
    return RunConfig(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, ztop=1.0, dt=1.0,
                     run_seconds=1.0, **switches)


def _loaded_settings(switches, **stated) -> dict[str, object]:
    """:func:`_loaded_config` as a mapping, with ``stated`` settings this
    build's RunConfig does not carry (a registry knob a lane adds)."""
    config = _loaded_config(switches)
    return {**{field.name: getattr(config, field.name)
               for field in dataclasses.fields(config)}, **stated}


def _citations(value) -> dict[str, list[str]]:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_registry_citations as checker
    finally:
        sys.path.pop(0)
    return checker.citations({"part": value})


def _part_sources(registry) -> dict[str, object]:
    """Every part the whole-registry identity binds, before stripping."""
    return registry_physics_part_sources(registry)


def _default_options(registry) -> dict[str, str]:
    return dict(registry["templates"][DEFAULT_TEMPLATE_ID]["components"])


def _changed(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {name for name in set(before) | set(after)
            if before.get(name) != after.get(name)}


def _numeric_default_knob(registry) -> str:
    return next(key for key, value in registry["parameters"].items()
                if "default" in value
                and isinstance(value["default"], (int, float))
                and not isinstance(value["default"], bool))


def _unimplemented_knob(registry) -> str:
    """An integer knob the registry declares but does not implement and
    gives to no one component, so every selection's scope gains it once it
    is implemented."""
    return next(key for key, value in registry["parameters"].items()
                if value.get("implemented") is False
                and value.get("type") == "integer"
                and "component_id" not in value)


#: A knob no registry declares, for the edit that adds one.
_ADDED_KNOB = "added_integer_knob"


# --------------------------------------------------- documentation leaves


def _move_citation(text: str, old: str, new: str) -> str:
    assert old in text, (old, text[:200])
    return text.replace(old, new)


@pytest.mark.parametrize("path,old,new", [
    # The two A153 commits, replayed backwards.
    (("components", "pbl", "options", "shinhong", "warnings", 1),
     "kernels/shinhong.cu:1429", "kernels/shinhong.cu:1371"),

    (("components", "turbulence", "options", "tke-1.5-order", "warnings", 0),
     "woof/core/dycore.py:1294", "woof/core/dycore.py:1041"),
    (("components", "turbulence", "options", "smagorinsky-3d", "warnings",
      0),
     "woof/core/dycore.py:1388", "woof/core/dycore.py:1131"),
])
def test_a_citation_edit_keeps_the_physics_identity(path, old, new):
    registry = physics_registry()
    node = registry
    for key in path[:-1]:
        node = node[key]
    # Source integration moves citation lines without changing the contract.
    citation_path = old.rsplit(":", 1)[0]
    match = re.search(re.escape(citation_path) + r":\d+", node[path[-1]])
    assert match is not None, (citation_path, node[path[-1]])
    old = match.group(0)
    assert old != new
    node[path[-1]] = _move_citation(node[path[-1]], old, new)

    assert registry_sha256(registry) != registry_sha256()
    assert registry_physics_sha256(registry) == registry_physics_sha256()
    component, option = path[1], path[3]
    options = {**_default_options(registry), component: option}
    assert registry_physics_parts(registry, options=options) \
        == registry_physics_parts(options=options)


@pytest.mark.parametrize("edit", [
    "label", "maturity", "scientific_evidence", "note", "reason",
    "unimplemented_reason", "template_warning", "evidence_axes"])
def test_labels_notes_and_evidence_keep_the_physics_identity(edit):
    registry = physics_registry()
    thompson = registry["components"]["microphysics"]["options"][
        "thompson-mp8"]
    if edit in ("label", "maturity"):
        thompson[edit] = thompson[edit] + "-edited"
    elif edit == "scientific_evidence":
        thompson[edit] = "obs-evaluated"
    elif edit == "note":
        thompson["asset_requirements"][0]["note"] = "rewritten"
    elif edit == "reason":
        registry["components"]["pbl"]["options"]["ysu"]["constraints"][
            "requires_components_reasons"]["surface_layer"] = "rewritten"
    elif edit == "unimplemented_reason":
        name = next(key for key, value in registry["parameters"].items()
                    if "unimplemented_reason" in value)
        registry["parameters"][name]["unimplemented_reason"] = "rewritten"
    elif edit == "template_warning":
        registry["templates"][DEFAULT_TEMPLATE_ID]["warnings"].append("new")
    else:
        registry["evidence_axes"]["maturity"]["axis"] = "rewritten"
    options = _default_options(registry)
    assert registry_physics_parts(
        registry, options=options, profile=DEFAULT_TEMPLATE_ID) \
        == registry_physics_parts(options=options, profile=DEFAULT_TEMPLATE_ID)


def test_no_citation_survives_into_a_physics_part():
    """Every file:line the registry carries is documentation, all 158."""
    leftovers = {
        name: sorted(found)
        for name, value in _part_sources(physics_registry()).items()
        if (found := _citations(strip_registry_documentation(value)))}
    assert leftovers == {}


def test_every_documentation_row_names_a_field_the_parts_carry():
    """A row that outlives its field stops describing anything."""
    used: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in REGISTRY_EVIDENCE_RECORDS and isinstance(
                        value, dict):
                    used.add(key)
                elif key in REGISTRY_DOCUMENTATION_FIELDS and (
                        value is None or isinstance(value, (str, list))):
                    used.add(key)
                elif key in REGISTRY_DOCUMENTATION_VALUE_MAPS:
                    used.add(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for value in _part_sources(physics_registry()).values():
        walk(value)
    declared = (set(REGISTRY_DOCUMENTATION_FIELDS)
                | set(REGISTRY_DOCUMENTATION_VALUE_MAPS)
                | set(REGISTRY_EVIDENCE_RECORDS))
    # This retired field remains classified so historical registry documents
    # still hash their explanatory prose as documentation. Removing its
    # classification would turn old aerosol notes into a false physics change.
    assert declared - used == {"withheld_aerosol_number_note"}


def _constraint_kinds(registry) -> set[str]:
    return {kind
            for component in registry["components"].values()
            for option in component["options"].values()
            for kind in (option.get("constraints") or {})}


def test_every_constraint_kind_is_classified_admission_or_kept():
    """A new kind of constraint is decided on, not bound by default.

    Admission kinds leave the identity because the preflight's
    recomputation re-checks them against the prepared configuration; a
    kind it does not re-check stays bound.  Both tables name only kinds
    the registry carries.
    """
    kinds = _constraint_kinds(physics_registry())
    assert not (set(REGISTRY_ADMISSION_CONSTRAINTS)
                & set(REGISTRY_CONSTRAINTS_KEPT_IN_PHYSICS_IDENTITY))
    assert kinds == (set(REGISTRY_ADMISSION_CONSTRAINTS)
                     | set(REGISTRY_CONSTRAINTS_KEPT_IN_PHYSICS_IDENTITY))


def test_an_options_own_required_values_stay_bound_through_its_parameters():
    """``required_settings`` leaves the identity; what it repeats does not.

    Every required value that names one of the option's own parameters
    (the Noah-MP ``opt_*``, MYNN ``bl_mynn_*`` and RUC mosaic pins among
    them) is carried with the same value in the option's parameters
    block, which stays bound.  The rest (moist, km_opt, khdif, kvdif,
    bl_pbl_physics, num_soil_layers) are settings the prepared
    configuration carries, which the recomputation checks.
    """
    for component in physics_registry()["components"].values():
        for option_id, option in component["options"].items():
            required = (option.get("constraints") or {}).get(
                "required_settings") or {}
            parameters = option.get("parameters") or {}
            assert {name: value for name, value in required.items()
                    if name in parameters} == {
                name: parameters[name] for name in required
                if name in parameters}, option_id


def test_every_top_level_block_is_classified():
    assert set(physics_registry()) == set(
        REGISTRY_BLOCKS_OUTSIDE_PHYSICS_IDENTITY)


# ------------------------------------------------------ physics still binds


def _edit_thompson_selector(registry):
    registry["components"]["microphysics"]["options"]["thompson-mp8"][
        "selectors"]["mp_physics"] = 88


def _edit_thompson_forbidden_value(registry):
    # A kept constraint: the preflight's recomputation does not read it.
    registry["components"]["microphysics"]["options"]["thompson-mp8"][
        "constraints"]["forbidden_setting_values"][
        "mp28_aerosol_source"].append("prescribed")


def _edit_noahmp_identity_value(registry):
    # As the registry builder writes it: the option's parameters block
    # and its required_settings together.  The parameters block binds it.
    option = registry["components"]["land_surface"]["options"]["noah-mp"]
    option["parameters"]["opt_sfc"] = 2
    option["constraints"]["required_settings"]["opt_sfc"] = 2


def _edit_thompson_table(registry):
    asset = registry["components"]["microphysics"]["options"][
        "thompson-mp8"]["asset_requirements"][0]["assets"][0]
    asset["sha256"] = "0" * 64


def _edit_thompson_kernel_identity(registry):
    consumers = registry["components"]["microphysics"]["options"][
        "thompson-mp8"]["consumers"]
    consumers["restart_algorithm_identity"] += "-v2"


def _edit_kf_option_parameter(registry):
    option = registry["components"]["cumulus"]["options"]["kain-fritsch"]
    option["parameters"]["cudt_minutes"] = 99.0


def _edit_parameter_default(registry):
    registry["parameters"][_numeric_default_knob(registry)]["default"] += 1


def _edit_parameter_implemented(registry):
    # A knob published as unimplemented becoming settable, as a lane that
    # implements one writes it (the Noah mosaic lane did this to
    # sf_surface_mosaic): the citation of the read that makes it true, a
    # default and its values.
    spec = registry["parameters"][_unimplemented_knob(registry)]
    spec.pop("implemented")
    spec.pop("unimplemented_reason", None)
    spec.update(consuming_read="woof/namelist_import.py", default=0,
                enum=[0, 1])


def _add_parameter(registry):
    # A knob added, as the Noah mosaic lane added mosaic_cat.
    assert _ADDED_KNOB not in registry["parameters"]
    registry["parameters"][_ADDED_KNOB] = {
        "consuming_read": "woof/core/noah_mosaic.py", "default": 3,
        "minimum": 1, "per_domain": False, "type": "integer"}


def _edit_template_parameter(registry):
    registry["templates"][DEFAULT_TEMPLATE_ID]["parameters"]["radt"] = 99.0


def _edit_selector_keys(registry):
    registry["components"]["pbl"]["selector_keys"].append("bl_pbl_extra")


_THOMPSON = "components.microphysics.options.thompson-mp8"


@pytest.mark.parametrize("edit,part,options", [
    (_edit_thompson_selector, _THOMPSON, {}),
    (_edit_thompson_forbidden_value, _THOMPSON, {}),
    (_edit_thompson_table, _THOMPSON, {}),
    (_edit_thompson_kernel_identity, _THOMPSON, {}),
    (_edit_kf_option_parameter, "components.cumulus.options.kain-fritsch",
     {}),
    (_edit_noahmp_identity_value, "components.land_surface.options.noah-mp",
     {"land_surface": "noah-mp"}),
    (_edit_parameter_default, "parameters.<numeric default knob>", {}),
    (_edit_parameter_implemented, "parameters.<unimplemented knob>", {}),
    (_add_parameter, f"parameters.{_ADDED_KNOB}", {}),
    (_edit_template_parameter, f"templates.{DEFAULT_TEMPLATE_ID}", {}),
    (_edit_selector_keys, "components.pbl", {}),
])
def test_a_real_physics_change_moves_the_identity_and_is_named(
        edit, part, options):
    """Each change moves exactly one part, the one it names.

    A knob is its own part, so a knob added or changed for one scheme is
    refused by the knob's name rather than as "parameters".
    """
    registry = physics_registry()
    options = {**_default_options(registry), **options}
    if part == "parameters.<numeric default knob>":
        part = f"parameters.{_numeric_default_knob(registry)}"
    if part == "parameters.<unimplemented knob>":
        part = f"parameters.{_unimplemented_knob(registry)}"
    edit(registry)
    before = registry_physics_parts(options=options,
                                    profile=DEFAULT_TEMPLATE_ID)
    after = registry_physics_parts(registry, options=options,
                                   profile=DEFAULT_TEMPLATE_ID)
    assert registry_physics_sha256(registry) != registry_physics_sha256()
    assert _changed(before, after) == {part}


# ------------------------------------------------ admission is re-checked


def _gf_admits_another_pbl(registry):
    # 2.8.1's UW PBL did this to GF and the MYNN surface layer.
    registry["components"]["cumulus"]["options"]["grell-freitas"][
        "constraints"]["requires_components"]["pbl"].append("pbl-new")


def _mynn_surface_admits_another_pbl(registry):
    constraints = registry["components"]["surface_layer"]["options"][
        "mynn"]["constraints"]
    constraints["requires_components"]["pbl"].append("pbl-new")
    constraints["requires_components_reasons"]["pbl"] += " Rewritten."


def _gf_drops_a_required_setting(registry):
    registry["components"]["cumulus"]["options"]["grell-freitas"][
        "constraints"]["required_settings"].pop("moist")


def _noah_admits_more_soil_layers(registry):
    registry["components"]["land_surface"]["options"]["noah"][
        "constraints"]["admitted_setting_values"]["num_soil_layers"].append(6)


@pytest.mark.parametrize("edit", [
    _gf_admits_another_pbl, _mynn_surface_admits_another_pbl,
    _gf_drops_a_required_setting, _noah_admits_more_soil_layers])
def test_an_admission_edit_keeps_the_physics_identity(edit):
    """Which configurations an option admits is not what it computes."""
    registry = physics_registry()
    options = {**_default_options(registry), "cumulus": "grell-freitas",
               "surface_layer": "mynn", "land_surface": "noah"}
    edit(registry)
    assert registry_sha256(registry) != registry_sha256()
    assert registry_physics_sha256(registry) == registry_physics_sha256()
    assert registry_physics_parts(registry, options=options) \
        == registry_physics_parts(options=options)


def _ysu_stops_admitting_classic_mm5(monkeypatch):
    registry = physics_registry()
    registry["components"]["pbl"]["options"]["ysu"]["constraints"][
        "requires_components"]["surface_layer"] = ["revised-mm5"]
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)
    return registry


def test_a_config_an_admission_edit_no_longer_admits_is_refused(monkeypatch):
    """The recomputation refuses it, naming the rule, identity unchanged.

    Replaces the v1 assertion that narrowing YSU's admitted surface
    layers moves YSU's part: that binding is what refused 18 of 2.8.0's
    56 receipts when the UW PBL was admitted beside GF and the MYNN
    surface layer.  A 2.8.0 receipt of a YSU over classic MM5 suite,
    against a build whose YSU no longer admits classic MM5, is refused
    by the named door (it raises) and by the per-domain one (it records
    the blocker, and the blocker is compared).
    """
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    options = row["named"]["components"]
    before = registry_physics_parts(options=options, profile=PROFILE)
    registry = _ysu_stops_admitting_classic_mm5(monkeypatch)
    assert registry_physics_parts(registry, options=options,
                                  profile=PROFILE) == before

    rule = r"requires surface_layer in \['revised-mm5'\], got 'classic-mm5'"
    config = _loaded_config(row["switches"])
    with pytest.raises(PhysicsCapabilityError, match=rule):
        validate_single_domain_physics_profile(PROFILE, config=config)

    expected = single_domain_physics_selection(config)
    differences = physics_selection_differences(
        row["unnamed"], expected, settings=config)
    assert any(line.startswith("domains.1.registry_blocker (prepared None")
               and re.search(rule, line) for line in differences), \
        differences


def _tke_refusal_rule(registry):
    return registry["components"]["turbulence"]["options"][
        "tke-1.5-order"]["constraints"]["refused_when"][0]


def test_a_conditional_refusal_the_preflight_evaluates_is_admission():
    """lane/282-namelist-tolerance moved the 1.5-order TKE closure's PBL-off
    requirement out of ``required_settings`` (admission) into a
    ``refused_when`` rule on ``diff_opt = 2``.  Bound by its kind, that
    rule refused every 2.8.0 and 2.8.1 preparation of the closure as
    "registry physics ... (changed)" though the closure computes what it
    did.  The preflight's recomputation evaluates a rule with no
    ``sources`` clause, so it is admission like ``required_settings``:
    widening it keeps the identity, and the configuration it refuses is
    still refused by the recomputation, naming the rule.  A ``sources``
    clause puts the same rule out of the recomputation's reach, and then
    the identity is the only place a change to it is seen."""
    registry = physics_registry()
    options = {**_default_options(registry), "turbulence": "tke-1.5-order"}
    rule = _tke_refusal_rule(registry)
    assert "sources" not in rule and rule["settings"] == {"diff_opt": [2]}
    rule["components"]["pbl"].remove("ysu")
    assert registry_sha256(registry) != registry_sha256()
    assert registry_physics_sha256(registry) == registry_physics_sha256()
    assert registry_physics_parts(registry, options=options) \
        == registry_physics_parts(options=options)

    pbl_on = _loaded_config({"km_opt": 2, "bl_pbl_physics": 1,
                             "sf_sfclay_physics": 1})
    with pytest.raises(PhysicsCapabilityError, match=r"tke-1\.5-order is "
                       r"refused here: diff_opt=2"):
        validate_physics_capabilities(pbl_on)
    assert validate_physics_capabilities(
        dataclasses.replace(pbl_on, diff_opt=1))["turbulence"] \
        == "tke-1.5-order"

    rule["sources"] = ["hrrr"]
    assert registry_physics_parts(registry, options=options) \
        != registry_physics_parts(options=options)


def test_terrain_wind_admission_preserves_every_physics_part():
    """Configuration-only terrain refusals change no prepared physics,
    including PBL off, whose option carries no constraint object."""
    registry = physics_registry()
    without = deepcopy(registry)
    removed = []
    for option in without["components"]["turbulence"]["options"].values():
        rules = option.get("constraints", {}).get("refused_when", [])
        removed.extend(rule for rule in rules
                       if {"topo_wind", "gwd_opt"}.intersection(
                           rule.get("settings", {})))
        option["constraints"]["refused_when"] = [
            rule for rule in rules
            if not {"topo_wind", "gwd_opt"}.intersection(
                rule.get("settings", {}))]
    assert removed and all("sources" not in rule for rule in removed)
    assert registry_sha256(without) != registry_sha256(registry)
    assert registry_physics_parts(without) == registry_physics_parts(registry)
    assert registry_physics_sha256(without) == registry_physics_sha256(registry)
    removed[0]["sources"] = ["hrrr"]
    a_closure = next(iter(without["components"]["turbulence"][
        "options"].values()))
    a_closure["constraints"]["refused_when"].append(removed[0])
    assert registry_physics_sha256(without) != registry_physics_sha256(registry)


@pytest.mark.parametrize("pbl,surface", [
    (0, 1), (1, 1), (2, 2), (5, 5), (9, 1), (11, 1), (900, 1),
])
@pytest.mark.parametrize("topo", [1, 2])
def test_preflight_rechecks_terrain_wind_without_source_identity(
        pbl, surface, topo):
    settings = {"bl_pbl_physics": pbl, "sf_sfclay_physics": surface,
                "topo_wind": topo}
    if pbl == 900:
        settings.update(km_opt=0, khdif=0.0, kvdif=0.0, moist=True)
    config = _loaded_config(settings)
    assert validate_physics_capabilities(dataclasses.replace(config, topo_wind=0))
    # WRF v4.6.1 hands CTOPO/CTOPO2 to YSU and to Shin-Hong, and both read
    # them (bl_ysu.F90:1308-1314; module_bl_shinhong.F:1387, :1472-1473).
    if pbl in (1, 11):
        assert validate_physics_capabilities(config)["pbl"] == {
            1: "ysu", 11: "shinhong"}[pbl]
    else:
        with pytest.raises(PhysicsCapabilityError,
                           match="topo_wind.*YSU.*Shin-Hong"):
            validate_physics_capabilities(config)


@pytest.mark.parametrize("gwd", [1, 3])
def test_preflight_rechecks_drag_requires_an_active_pbl(gwd):
    config = _loaded_config({"bl_pbl_physics": 0, "sf_sfclay_physics": 1,
                             "gwd_opt": gwd})
    assert validate_physics_capabilities(dataclasses.replace(config, gwd_opt=0))
    with pytest.raises(PhysicsCapabilityError, match="gwd_opt.*PBL driver"):
        validate_physics_capabilities(config)
    assert validate_physics_capabilities(dataclasses.replace(
        config, bl_pbl_physics=1))["pbl"] == "ysu"


def test_another_schemes_change_leaves_this_selection_alone():
    """WDM6's restart identity moving must not refuse a Thompson run."""
    registry = physics_registry()
    options = _default_options(registry)
    registry["components"]["microphysics"]["options"]["wdm6-mp16"][
        "consumers"]["restart_algorithm_identity"] += "-v4"
    assert registry_physics_sha256(registry) != registry_physics_sha256()
    assert registry_physics_parts(registry, options=options) \
        == registry_physics_parts(options=options)
    wdm6 = {**options, "microphysics": "wdm6-mp16"}
    assert registry_physics_parts(registry, options=wdm6) \
        != registry_physics_parts(options=wdm6)


# ----------------------------------------------------- receipts resolve


def _receipts_2492999cd():
    document = json.loads(RECEIPTS_2492999CD.read_text(encoding="utf-8"))
    assert document["registry_sha256"] == DOCUMENT_280
    return document["receipts"]


def test_the_history_resolves_the_280_document_and_its_successors():
    """Check the history against its committed registry documents.

    The declared registry and recorded history are the option inventory.
    Comparing both to their Git documents detects a changed part without
    requiring every new scheme to edit a second handwritten option list.
    The old receipt refusal tests below retain their independent expectations.
    """
    from tools.registry_physics_history import check

    history = registry_physics_history()
    rows = history["documents"]
    assert DOCUMENT_280 in rows
    physics_280 = rows[DOCUMENT_280]["physics_sha256"]
    for document in DOCUMENTS_CITATION_ONLY_SINCE_280:
        assert rows[document]["physics_sha256"] == physics_280
    assert registry_sha256() in rows
    assert history["physics"][rows[registry_sha256()]["physics_sha256"]] \
        == registry_physics_parts()
    failures = check()
    if failures and not (ROOT / "work" / "build_release_snapshot.py").is_file():
        # A public snapshot carries one commit per release, so the engine
        # commits and registry blobs this history cites cannot be read in it
        # (2.9.0 round 2, CI rehearsal routes B and C: 143 rows "not in this
        # repository's history").  The private clone's Stage 1 runs the check;
        # the assertions above and below hold in every tree.
        unreadable = ("is not in this repository's history",
                      "is missing or is not a registry document")
        if all(any(text in failure for text in unreadable)
               for failure in failures):
            failures = []
    assert failures == [], "recorded identity differs from its committed registry"
    for row in rows.values():
        assert re.fullmatch(r"[0-9a-f]{10}", row["commit"])
        parts = history["physics"][row["physics_sha256"]]
        assert all(re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in parts.values())


def _expected_for(row, spelling, config=None):
    """This build's receipt for a 2.8.0 receipt's configuration, as the
    real preflight recomputes it: from the TOML loaded at this build."""
    recorded = row[spelling]
    acknowledgements = tuple(recorded["acknowledgements"])
    provenance = recorded["acknowledgement_provenance"]
    config = _loaded_config(row["switches"]) if config is None else config
    if spelling == "named":
        return validate_single_domain_physics_profile(
            row["profile"], config=config,
            expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=provenance)
    return single_domain_physics_selection(
        config, expert_acknowledgements=acknowledgements,
        acknowledgement_provenance=provenance)


@pytest.mark.parametrize("row", _receipts_2492999cd(),
                         ids=lambda row: row["profile"])
@pytest.mark.parametrize("spelling", ["named", "unnamed"])
def test_a_receipt_written_at_2492999cd_resolves_where_physics_is_unchanged(
        row, spelling):
    """Every shipped suite, both receipt spellings, prepared at 2.8.0's
    registry: refused only for the declared default and option changes.
    Old CQ-off named profiles are refused during recomputation; unnamed
    explicit controls retain their saved values and bind the changed parts. The
    urban component, its knobs and the two radiation knobs implemented
    since resolve, because the 2.8.0 TOML loaded here leaves them off.
    """
    config = _loaded_config(row["switches"])
    if spelling == "named" and row["profile"] in CQ_OFF_280_NAMED_PROFILES:
        assert row["switches"]["moist_cq"] is False
        with pytest.raises(ValueError,
                           match="selected physics differs from profile") as error:
            _expected_for(row, spelling, config)
        assert ("settings={'moist_cq': {'selected': False, 'expected': True}}"
                in str(error.value))
        return
    expected = _expected_for(row, spelling, config)
    assert physics_selection_differences(
        row[spelling], expected, settings=config) \
        == _legacy_expected(row, spelling)


def test_an_unknown_registry_document_is_refused_by_name():
    expected = validate_single_domain_physics_profile(PROFILE)
    recorded = deepcopy(expected)
    recorded.pop("registry_physics")
    recorded["registry_sha256"] = "e" * 64
    differences = physics_selection_differences(recorded, expected)
    assert len(differences) == 1
    assert "registry document eeeeeeeeeeee" in differences[0]
    assert "cannot be established" in differences[0]


def test_a_legacy_receipt_names_the_part_that_changed(monkeypatch):
    """2.8.0's receipt against a build whose Thompson table changed."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    registry = physics_registry()
    _edit_thompson_kernel_identity(registry)
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)
    config = _loaded_config(row["switches"])
    expected = validate_single_domain_physics_profile(PROFILE, config=config)
    assert physics_selection_differences(
        row["named"], expected, settings=config) == _legacy_expected(row, "named",
        "registry physics of components.microphysics.options.thompson-mp8 "
        "(changed)")


def test_record_only_fields_are_the_prose_ones_and_the_rest_still_bind():
    expected = validate_single_domain_physics_profile(PROFILE)
    assert set(SELECTION_RECEIPT_RECORD_ONLY_FIELDS) == {
        "registry_sha256", "registry_physics", "maturity"}
    relabelled = deepcopy(expected)
    relabelled["registry_sha256"] = "f" * 64
    relabelled["maturity"] = "supported"
    assert physics_selection_differences(relabelled, expected) == []
    for field, value in (("selectors", {"mp_physics": 6}),
                         ("governance", {"state": "other"}),
                         ("acknowledgements", ["x"])):
        tampered = deepcopy(expected)
        tampered[field] = value
        differences = physics_selection_differences(tampered, expected)
        assert differences and all(
            line.startswith(field) for line in differences), differences
    tampered = deepcopy(expected)
    tampered["registry_physics"]["parts"]["parameters.radt"] = "0" * 64
    assert physics_selection_differences(tampered, expected) == [
        "registry physics of parameters.radt (changed)"]


def _added_knobs(monkeypatch):
    """This build's registry, where the Noah mosaic lane made
    sf_surface_mosaic settable and added mosaic_cat and
    mosaic_urban_canopy, each off at a default RunConfig shares (0, 3 and
    WRF's rule "dominant"), none of them in 2.8.0's registry."""
    registry = physics_registry()
    for knob, off in (("sf_surface_mosaic", 0), ("mosaic_cat", 3),
                      ("mosaic_urban_canopy", "dominant")):
        spec = registry["parameters"][knob]
        assert "implemented" not in spec and spec["default"] == off, knob
        assert setting_off_value(knob, registry) == off, knob
    monkeypatch.setattr(registry_module, "_REGISTRY", registry)
    return registry


@pytest.mark.parametrize("spelling", ["named", "unnamed"])
def test_a_legacy_receipt_resolves_knobs_added_at_their_off_value(
        monkeypatch, spelling):
    """v2 refused every 2.8.0 receipt by the knobs' names here."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    _added_knobs(monkeypatch)
    for settings in (
            _loaded_settings(row["switches"]),
            _loaded_settings(row["switches"], sf_surface_mosaic=0,
                             mosaic_cat=3, mosaic_urban_canopy="dominant")):
        assert physics_selection_differences(
            row[spelling], _expected_for(row, spelling, settings),
            settings=settings) == _legacy_expected(row, spelling)


@pytest.mark.parametrize("spelling", ["named", "unnamed"])
@pytest.mark.parametrize("stated,named", [
    ({"sf_surface_mosaic": 1}, ["sf_surface_mosaic"]),
    ({"sf_surface_mosaic": 1, "mosaic_cat": 7},
     ["mosaic_cat", "sf_surface_mosaic"]),
], ids=["mosaic-on", "mosaic-on-with-7-tiles"])
def test_a_legacy_receipt_names_an_added_knob_set_on(
        monkeypatch, spelling, stated, named):
    """A preparation made before a knob existed never ran it on."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    _added_knobs(monkeypatch)
    settings = _loaded_settings(row["switches"], **stated)
    assert physics_selection_differences(
        row[spelling], _expected_for(row, spelling, settings),
        settings=settings) == _legacy_expected(row, spelling, *[
        f"registry physics of parameters.{knob} (absent from the prepared "
        "registry)" for knob in named])


#: A tile count set while the tiles are off: a WRF namelist carrying
#: mosaic_cat with sf_surface_mosaic = 0 (WRF reads the count only under
#: mosaic, Registry.EM_COMMON:2537).  The canopy rule's town option is
#: refused at load without tiles and the single-layer canopy
#: (woof.config.validate_noah_mosaic_config), so it has no such case.
_UNREAD = {
    "tiles-off": {"mosaic_cat": 7},
    "tiles-off-stated": {"sf_surface_mosaic": 0, "mosaic_cat": 1},
}


@pytest.mark.parametrize("spelling", ["named", "unnamed"])
@pytest.mark.parametrize("stated", list(_UNREAD.values()), ids=list(_UNREAD))
def test_a_legacy_receipt_resolves_a_knob_its_configuration_does_not_read(
        monkeypatch, spelling, stated):
    """A153's third repair refused these by the knob's name: the knob had
    no scope the registry could read, so a value other than its default
    was taken for physics a 2.8.0 preparation never ran, though nothing
    reads it.  The row's ``read_when`` is that scope."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    _added_knobs(monkeypatch)
    settings = _loaded_settings(row["switches"], **stated)
    assert physics_selection_differences(
        row[spelling], _expected_for(row, spelling, settings),
        settings=settings) == _legacy_expected(row, spelling)


def test_the_read_condition_is_the_knobs_registry_row(monkeypatch):
    """No knob is named in the comparison's code: the condition is the
    declaration's, so a registry whose row states none refuses the same
    configuration by the knob's name again, and the rows say what WRF
    reads them under."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    registry = _added_knobs(monkeypatch)
    assert registry["parameters"]["mosaic_cat"]["component_id"] == "land_surface"
    # The owner is a registered component; its conditional read remains
    # authoritative when the component's configurable tile count is unread.
    assert "land_surface" in registry["components"]
    assert registry_knob_readers("mosaic_cat", registry) is not None
    assert registry["parameters"]["mosaic_cat"]["read_when"] == {
        "sf_surface_mosaic": 1}
    assert registry["parameters"]["mosaic_urban_canopy"]["read_when"] == {
        "sf_surface_mosaic": 1, "sf_urban_physics": 1}
    settings = _loaded_settings(row["switches"], mosaic_cat=7)
    for spelling in ("named", "unnamed"):
        assert physics_selection_differences(
            row[spelling], _expected_for(row, spelling, settings),
            settings=settings) == _legacy_expected(row, spelling)
    registry["parameters"]["mosaic_cat"].pop("read_when")
    for spelling in ("named", "unnamed"):
        assert physics_selection_differences(
            row[spelling], _expected_for(row, spelling, settings),
            settings=settings) == _legacy_expected(row, spelling,
            "registry physics of parameters.mosaic_cat (absent from the "
            "prepared registry)")


@pytest.mark.parametrize("settings,read", [
    ({}, False),
    ({"sf_surface_mosaic": 0}, False),
    ({"sf_surface_mosaic": 1}, True),
    ({"sf_surface_mosaic": True}, False),
])
def test_a_read_condition_holds_only_at_its_value(settings, read):
    """An omitted setting holds its off value; a boolean never equals 1."""
    from woof.physics_registry import registry_knob_is_read
    assert registry_knob_is_read("mosaic_cat", settings) is read
    assert registry_knob_is_read("radt", settings) is True


@pytest.mark.parametrize("knob,old_value,new_value", [
    ("aer_init_opt", 0, 2),
    ("mp28_aerosol_source", "auto", "analysis"),
])
def test_enum_extension_reuses_only_previously_admitted_values(knob, old_value, new_value):
    from woof.physics_registry import canonical_sha256, REGISTRY_PHYSICS_IDENTITY_SCHEMA
    spec = physics_registry()["parameters"][knob]
    part = "parameters." + knob
    prior = {**spec, "enum": spec["compatible_previous_enums"][0]}

    def receipt(declaration):
        return {"registry_physics": {
            "schema": REGISTRY_PHYSICS_IDENTITY_SCHEMA,
            "parts": {part: canonical_sha256(strip_registry_documentation(declaration))}}}

    before, after = receipt(prior), receipt(spec)
    assert physics_selection_differences(before, after, settings={knob: old_value}) == []
    expected = [f"registry physics of {part} (changed)"]
    assert physics_selection_differences(before, after, settings={knob: new_value}) == expected
    assert physics_selection_differences(before, after) == expected
    # A different prior default is physics, even though the enum matches.
    wrong_default = receipt({**prior, "default": new_value})
    assert physics_selection_differences(wrong_default, after,
                                         settings={knob: old_value}) == expected


def test_added_aerosol_refusals_preserve_only_admitted_prior_options():
    from woof.physics_registry import canonical_sha256, REGISTRY_PHYSICS_IDENTITY_SCHEMA
    option = physics_registry()["components"]["microphysics"]["options"]["thompson-mp8"]
    name = "components.microphysics.options.thompson-mp8"
    prior = {**option, "constraints": {**option["constraints"],
        "forbidden_setting_values": option["compatible_previous_forbidden_settings"][0]}}

    def receipt(value):
        return {"registry_physics": {"schema": REGISTRY_PHYSICS_IDENTITY_SCHEMA,
            "parts": {name: canonical_sha256(strip_registry_documentation(value))}}}

    before, after = receipt(prior), receipt(option)
    assert physics_selection_differences(before, after, settings={
        "mp28_aerosol_source": "auto", "use_rap_aero_icbc": False}) == []
    expected = [f"registry physics of {name} (changed)"]
    for settings in ({"mp28_aerosol_source": "analysis"}, {"use_rap_aero_icbc": True}):
        assert physics_selection_differences(before, after, settings=settings) == expected
    wrong_physics = receipt({**prior, "selectors": {"mp_physics": 88}})
    assert physics_selection_differences(wrong_physics, after, settings={}) == expected


def test_without_the_configuration_an_added_knob_is_named(monkeypatch):
    """Unknown values cannot be shown off: name every knob added or
    implemented since 2.8.0, including A179's registered IEVA knob,
    lane/282-namelist-tolerance's diff_opt and mix_full_fields, terrain
    drag, WRF's three advection orders, and the radiation-driver options
    swint_opt, aer_opt, alb_sol and source cloud/smoke controls,
    WRF's cycling (the gsd_41 MYNN cycled start), and the 2.9.4
    capabilities campaign's registered parameters: lane/cap-hail's
    HAILCAST selectors, lane/cap-slab-ocean's PWP controls and
    lane/cap-seaice's prognostic sea ice."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    _added_knobs(monkeypatch)
    assert physics_selection_differences(
        row["named"], _expected_for(row, "named")) == _legacy_expected(row, "named",
        "registry physics of components.microphysics.options.thompson-mp8 (changed)",
        "registry physics of parameters.aer_init_opt (changed)",
        "registry physics of parameters.mp28_aerosol_source (changed)",
        "registry physics of parameters.use_rap_aero_icbc (absent from the prepared registry)", *[
        f"registry physics of parameters.{knob} (absent from the prepared "
        "registry)"
        for knob in ('aer_angexp_opt', 'aer_angexp_val', 'aer_aod550_opt', 'aer_aod550_val', 'aer_asy_opt', 'aer_asy_val', 'aer_auxinput15', 'aer_auxinput15_interval', 'aer_opt', 'aer_ssa_opt', 'aer_ssa_val', 'aer_type', 'alb_sol', 'bl_mynn_cloud_tendency_form', 'bl_mynn_gsd41_unsquared_qtke', 'bl_mynn_version', 'cellcount_method', 'cldtop_adjustment', 'cycling', 'diff_6th_factor2', 'diff_6th_form', 'diff_opt', 'do_radar_ref', 'flashrate_factor', 'fractional_seaice', 'gwd_opt', 'h_mom_adv_order', 'hailcast_opt', 'hailcast_version', 'haildt', 'iccg_method', 'iccg_prescribed_den', 'iccg_prescribed_num', 'lake_min_elev', 'lakedepth_default', 'lightning_dt', 'lightning_option', 'lightning_start_seconds', 'mix_full_fields', 'mosaic_cat', 'mosaic_urban_canopy', 'mp_zero_out', 'mp_zero_out_all', 'mp_zero_out_thresh', 'noahmp_snow_age', 'ocean_analysis', 'ocean_analysis_source', 'ocean_levels', 'ocean_s', 'ocean_t', 'ocean_z', 'omdt', 'oml_gamma', 'oml_hml0', 'oml_relaxation_time', 'rrtmg_aer3_type_source', 'rrtmg_cloud_optics_form', 'rrtmg_smoke_manifest', 'scalar_adv_opt', 'seaice_initial_thickness', 'seaice_new_ice_thickness', 'seaice_ocean_heat_flux', 'seaice_prognostic', 'sf_lake_physics', 'sf_ocean_physics', 'sf_surface_mosaic', 'slope_rad', 'solar_generation', 'spp_conv', 'spp_pbl', 'swint_opt', 'thompson_fork_snow_fall', 'thompson_version', 'topo_shading', 'topo_wind', 'upper_wind_limiter_form', 'urban_table_dir', 'urban_water_balance', 'use_lakedepth', 'v_mom_adv_order', 'v_sca_adv_order', 'ysu_version', 'zadvect_implicit', 'zadvect_implicit_variant')])


def test_a_new_receipt_resolves_to_its_own_parts():
    expected = single_domain_physics_selection(
        single_domain_runtime_switches(PROFILE))
    assert recorded_registry_physics_parts(expected) \
        == expected["registry_physics"]["parts"]
    assert physics_selection_differences(
        json.loads(json.dumps(expected)), expected) == []


@pytest.mark.parametrize("pbl", [1, 5])
def test_changed_mynn_parameter_only_binds_a_configuration_that_reads_it(pbl):
    # Before the MYNN options were scoped, every receipt carried this part.
    # A YSU preparation must survive the scope correction; an active MYNN
    # calculation must still name a changed mixing-length declaration.
    row = next(row for row in _receipts_2492999cd()
               if row["switches"].get("bl_pbl_physics") == pbl)
    settings = _loaded_settings(row["switches"])
    expected = single_domain_physics_selection(settings)
    recorded = deepcopy(expected)
    key = "parameters.bl_mynn_mixlength"
    recorded["registry_physics"]["parts"][key] = "0" * 64
    differences = physics_selection_differences(recorded, expected, settings=settings)
    if pbl == 1:
        assert differences == []
        assert physics_selection_differences(recorded, expected)
    else:
        assert differences == [f"registry physics of {key} (changed)"]


def test_parts_in_another_identity_schema_resolve_through_the_document():
    """A receipt whose parts this build cannot read falls back to history."""
    row = next(row for row in _receipts_2492999cd()
               if row["profile"] == PROFILE)
    config = _loaded_config(row["switches"])
    expected = validate_single_domain_physics_profile(PROFILE, config=config)
    recorded = deepcopy(row["named"])
    # v1's spelling: the parameters block as one part.
    recorded["registry_physics"] = {
        "schema": "gpuwm-physics-registry-parts-v1",
        "parts": {"parameters": "0" * 64}}
    assert physics_selection_differences(
        recorded, expected, settings=config) \
        == _legacy_expected(row, "named")
    recorded["registry_sha256"] = "e" * 64
    assert "cannot be established" in physics_selection_differences(
        recorded, expected, settings=config)[0]


# ------------------------------------------- physics added since 2.8.0


def _thompson_row():
    return next(row for row in _receipts_2492999cd()
                if row["profile"] == PROFILE)


def test_an_unimplemented_knob_is_no_part():
    """No configuration can set it, so it has no physics; implementing it
    is an addition (the case of slope_rad and topo_shading since 2.8.0)."""
    registry = physics_registry()
    knob = next(name for name, spec in registry["parameters"].items()
                if spec.get("implemented") is False)
    assert f"parameters.{knob}" not in registry_physics_parts(registry)
    registry["parameters"][knob]["default"] = 99
    assert registry_physics_sha256(registry) == registry_physics_sha256()


def test_a_component_knob_only_unselected_options_read_is_outside():
    """The urban knobs are read by the three canopy options alone, so a
    selection with urban off does not bind them, and one with a canopy
    model on does."""
    registry = physics_registry()
    readers = {knob: registry_knob_readers(knob, registry)
               for knob in registry["parameters"]}
    owned = {"num_urban_hi", "use_wudapt_lcz"}
    assert all(readers[knob] == {"urban.slucm", "urban.bep",
                                 "urban.bep-bem"} for knob in owned)
    options = _default_options(registry)
    assert options["urban"] == component_off_option("urban", registry)
    parts = registry_physics_parts(registry, options=options)
    assert not {f"parameters.{knob}" for knob in owned} & set(parts)
    canopy = registry_physics_parts(
        registry, options={**options, "urban": "slucm"})
    assert {f"parameters.{knob}" for knob in owned} <= set(canopy)


def test_a_template_entry_at_its_off_value_keeps_the_template():
    """Every template gained ``urban: none``; that composes nothing."""
    registry = physics_registry()
    template = registry["templates"][PROFILE]
    assert template["components"]["urban"] == "none"
    without = deepcopy(template)
    del without["components"]["urban"]
    assert registry_off_template(template, registry) \
        == registry_off_template(without, registry)
    selected = deepcopy(template)
    selected["components"]["urban"] = "slucm"
    assert registry_off_template(selected, registry)["components"][
        "urban"] == "slucm"


@pytest.mark.parametrize("spelling", ["named", "unnamed"])
def test_a_280_receipt_resolves_where_added_physics_is_left_off(spelling):
    """Every added setting stated at its off value, and an urban knob set
    while urban is off (nothing selected reads it)."""
    row = _thompson_row()
    for stated in ({}, {"sf_urban_physics": 0, "slope_rad": 0,
                        "topo_shading": 0}, {"use_wudapt_lcz": 1}):
        settings = _loaded_settings(row["switches"], **stated)
        assert physics_selection_differences(
            row[spelling], _expected_for(row, spelling, settings),
            settings=settings) == _legacy_expected(row, spelling), stated


@pytest.mark.parametrize("spelling", ["named", "unnamed"])
@pytest.mark.parametrize("knob", ["slope_rad", "topo_shading"])
def test_a_280_receipt_names_a_knob_implemented_since_and_set_on(
        spelling, knob):
    row = _thompson_row()
    settings = _loaded_settings(row["switches"], **{knob: 1})
    assert physics_selection_differences(
        row[spelling], _expected_for(row, spelling, settings),
        settings=settings) == _legacy_expected(row, spelling,
        f"registry physics of parameters.{knob} (absent from the prepared "
        "registry)")


def test_a_280_receipt_names_a_component_added_since_and_selected():
    """A 2.8.0 preparation never ran an urban canopy model."""
    row = _thompson_row()
    settings = _loaded_settings(row["switches"], sf_urban_physics=1)
    with pytest.raises(ValueError, match="selected physics differs from "
                       "profile"):
        _expected_for(row, "named", settings)
    differences = physics_selection_differences(
        row["unnamed"], _expected_for(row, "unnamed", settings),
        settings=settings)
    for line in (
            "domains.1.components.urban (prepared absent, this build "
            "'slucm')",
            "domains.1.selectors.sf_urban_physics (prepared absent, this "
            "build 1)",
            "registry physics of components.urban (absent from the "
            "prepared registry)",
            "registry physics of components.urban.options.slucm (absent "
            "from the prepared registry)"):
        assert line in differences, differences
