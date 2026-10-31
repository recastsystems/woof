"""Phase-5 Task 1 byte-identity freeze pins (architecture section A).

``woof/config.py`` carries the Phase-5 fields plus WRF-default
``top_lid = False`` and attribution-only ``moist_cq = True``;
``load_config()`` and the legacy
``[grid]``/``[dynamics]``/``[run]`` tables are untouched.  These pins
keep every existing legacy TOML under ``configs/`` and every frozen-case
constructed RunConfig fixed except for declared default corrections --
``dataclasses.asdict`` compared field-for-field against the golden
snapshot captured from the pre-change code plus explicitly reviewed new
configuration fields (``tests/data/config_freeze_golden.json``). The CQ
defect fix intentionally enables the formerly disabled default; that one
declared difference is checked separately without rewriting the old snapshot.

Freeze discipline: a legacy TOML added to ``configs/`` without a golden
entry FAILS here -- new files are pinned consciously, never implicitly.
Experiment-shaped TOMLs ([experiment]/[[domain]]) are exercised by
tests/test_experiment.py instead.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tomllib

import pytest

from woof.config import RunConfig, load_config, _KNOWN_TABLES
from woof.experiment import is_experiment_toml

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "config_freeze_golden.json")
    .read_text())

_WIND_DEFAULTS = {"wind_farm_scheme": "off", "wind_farms": "", "wind_farm_table": "",
                  "windfarm_tke_factor": .25, "wind_farm_max_dx_m": 3000.0}


def _frozen_constructors():
    from woof.verify.cases import (hill2d, igw, moist_bubble,
                                    straka, wk82)
    constructors = {
        "straka.default_config": straka.default_config,
        "igw.default_config": igw.default_config,
        "hill2d.default_config": hill2d.default_config,
        "moist_bubble.default_config": moist_bubble.default_config,
        "wk82.default_config": wk82.default_config,
    }
    try:
        from woof.verify.cases import real74_d01
    except ImportError:
        from _woof_snapshot import left_out
        if not left_out("woof/verify/cases/real74_d01.py"):
            raise
        return constructors  # the real-case module is left out of this tree, its two constructors with it
    constructors.update({
        "real74_d01.config": real74_d01.config,
        "real74_d01.phase3_config": real74_d01.phase3_config,
    })
    return constructors


def _golden_in_this_tree() -> set:
    """The golden keys whose config file or case module this tree carries."""
    from _woof_snapshot import left_out

    def gone(key):
        if key.startswith("configs/"):
            return left_out(key)
        return left_out(f"woof/verify/cases/{key.split('.')[0]}.py")
    return {key for key in GOLDEN if not gone(key)}


def _pop_fork_dycore_defaults(actual: dict) -> bool:
    """The diff_6th_form / mp_zero_out fields at the values every earlier
    configuration ran: WRF v4.6.1's filter and no zero-out pass."""
    return (actual.pop("scalar_adv_opt") == 1
            and actual.pop("upper_wind_limiter_form") == "wrf_461"
            and actual.pop("diff_6th_form") == "wrf_461"
            and actual.pop("diff_6th_factor2") is None
            and actual.pop("mp_zero_out") == 0
            and actual.pop("mp_zero_out_thresh") == 1.0e-8
            and actual.pop("mp_zero_out_all") == 0)


def test_new_fields_are_reviewed_defaults_appended_last():
    """Phase-5 Task 1: preserve positional construction and reviewed defaults.

    Retains A158/A165 append order, the 2.8.8 terrain-clock correction,
    the 2.9.0 correct-mode defaults and the 2.9.1 ocean/WB5 additions.
    Literal default rows use equality or singleton identity as before.
    """
    from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # fix/hailcast, keyword-only and appended last: the HAILCAST generation
    # (operational HRRR v4's by default; output-only).
    assert names.pop() == "hailcast_version"
    assert RunConfig.__dataclass_fields__["hailcast_version"].default == "wrf_39_noaa"
    assert RunConfig.__dataclass_fields__["hailcast_version"].kw_only
    for name, default in reversed(tuple(_WIND_DEFAULTS.items())):
        assert names.pop() == name
        field = RunConfig.__dataclass_fields__[name]
        assert field.kw_only and field.default == default
    # Appended last after the campaign, keyword-only and off by default:
    # sea ice's real.exe threshold (4c82704f8f, Registry default 100 K
    # converts nothing) and the HRRR v4 output controls (lane/hrrr-v4-output:
    # e55b68accb, bd4a1c2090, f80ac46aeb).
    # 2.9.7, keyword-only and appended last: tc/nest's fixed-frame nest
    # accumulation archive (on; read only on a nested domain) and the HRRR v4
    # climate statistics (lane/hrrr-v4-clwrf 3aec2abd31, off).
    # 2.9.10, keyword-only and appended last: WRF's ghg_input (CAMtr gases,
    # engine default 0 = fixed gases; fix/wrf461-2dom-exact 329c723395).
    # Constant moving frames append keyword-only and are disabled by default.
    for name in ("vmove", "umove"):
        assert names.pop() == name
        assert RunConfig.__dataclass_fields__[name].default == 0.0
        assert RunConfig.__dataclass_fields__[name].kw_only
    assert names.pop() == "ghg_input"
    assert RunConfig.__dataclass_fields__["ghg_input"].default == 0
    assert RunConfig.__dataclass_fields__["ghg_input"].kw_only
    for name, default in (("nest_accumulation_archive", True),
                          ("wind_int", 5), ("diag_int", 15),
                          ("output_diagnostics", 0),
                          ("smoke_off_reason", ""),
                          ("history_interval_change_s", 0.0),
                          ("history_interval2_s", 0.0),
                          ("reset_interval1", 60),
                          ("gsd_diagnostics", 0),
                          ("prec_acc_dt1", 0.0),
                          ("prec_acc_dt", 0.0),
                          ("seaice_threshold", 100.0)):
        assert names.pop() == name
        assert RunConfig.__dataclass_fields__[name].default == default, name
        assert RunConfig.__dataclass_fields__[name].kw_only, name
    # The capabilities campaign (2.9.4), appended keyword-only in merge
    # order.  lane/cap-lake: the table-driven lake-state analysis, "auto"
    # source and no licensed file.
    for name, default in (("lake_state_accept_licence", ""),
                          ("lake_state_path", ""),
                          ("lake_state_source", "auto")):
        assert names.pop() == name
        assert RunConfig.__dataclass_fields__[name].default == default, name
        assert RunConfig.__dataclass_fields__[name].kw_only, name
    assert names.pop() == "ra_sw_eclipse"
    assert RunConfig.__dataclass_fields__["ra_sw_eclipse"].default == 1
    # lane/cap-seaice (prognostic sea ice, default on), lane/cap-hail
    # (HAILCAST at its -1 auto selector) and lane/cap-waves-couple (the
    # table-driven ocean door, no config file by default).
    for name, default in (("seaice_ocean_heat_flux", 2.0),
                          ("seaice_new_ice_thickness", 0.5),
                          ("seaice_initial_thickness", 1.0),
                          ("seaice_prognostic", True),
                          ("haildt", 60.0), ("hailcast_opt", -1),
                          ("ocean_config", "")):
        assert names.pop() == name
        assert RunConfig.__dataclass_fields__[name].default == default, name
        assert RunConfig.__dataclass_fields__[name].kw_only, name
    # lane/cap-slab-ocean: WRF's 3-D PWP ocean controls sit keyword-only
    # beside sf_ocean_physics (they are read only under sf_ocean_physics =
    # 2), so positional construction is unchanged.
    pwp = ["ocean_levels", "ocean_analysis", "ocean_analysis_source",
           "ocean_mld_climatology", "omdt", "ocean_z", "ocean_t", "ocean_s"]
    at = names.index("sf_ocean_physics") + 1
    assert names[at:at + len(pwp)] == pwp
    for name in pwp:
        assert RunConfig.__dataclass_fields__[name].kw_only, name
    del names[at:at + len(pwp)]
    from woof.config import G3_RUN_DEFAULTS
    assert [name for name in names if name in G3_RUN_DEFAULTS] == list(G3_RUN_DEFAULTS)
    for name, default in G3_RUN_DEFAULTS.items():
        field = RunConfig.__dataclass_fields__[name]
        assert field.kw_only and field.default == default
    sas = ['nsas_dx_factor', 'sas_pgcon']
    last_g3 = names.index(list(G3_RUN_DEFAULTS)[-1])
    assert names[last_g3 + 1:last_g3 + 3] == sas
    for field in sas:
        assert RunConfig.__dataclass_fields__[field].kw_only, field
    names = [name for name in names if name not in G3_RUN_DEFAULTS and name not in sas]
    assert names[-len(FIRE_RUN_FIELDS):] == list(FIRE_RUN_FIELDS)
    for field in FIRE_RUN_FIELDS:
        assert RunConfig.__dataclass_fields__[field].kw_only, field
    names = names[:-len(FIRE_RUN_FIELDS)]
    assert names[-len(CHEM_RUN_FIELDS):] == list(CHEM_RUN_FIELDS)
    names = names[:-len(CHEM_RUN_FIELDS)]
    from woof import correct_mode
    assert names[-4:] == ['sf_ocean_physics', 'oml_hml0', 'oml_gamma', 'oml_relaxation_time']
    names = names[:-4]
    # 2.9.7 HN1: keyword-only, so the positional constructor is unchanged.
    assert names.pop() == 'hydrometeor_handover_numbers'
    assert RunConfig.__dataclass_fields__['hydrometeor_handover_numbers'].kw_only
    assert names.pop() == 'wsm7_conservative_surface'
    assert names.pop() == 'wsm5_conservative_surface'
    assert names.pop() == 'wsm3_conservative_surface'
    assert names.pop() == 'thompson_cloud_surface_booking'
    assert names.pop() == 'wdm6_conservative_surface'
    assert names.pop() == 'ntiedtke_dry_mass_coupling'
    assert names.pop() == 'wsm6_conservative_surface'
    for field in ('ntiedtke_dry_mass_coupling', 'wsm6_conservative_surface',
     'wdm6_conservative_surface', 'thompson_cloud_surface_booking',
     'wsm3_conservative_surface', 'wsm5_conservative_surface',
     'wsm7_conservative_surface', 'hydrometeor_handover_numbers'):
        assert RunConfig.__dataclass_fields__[field].default is correct_mode.CORRECT_KEYS_DEFAULT
        assert correct_mode.CORRECT_KEYS_DEFAULT is (not correct_mode.STRICT)
    from woof.lightning_contract import LIGHTNING_DEFAULTS
    lightning = ('lightning_option', 'lightning_dt', 'lightning_start_seconds', 'flashrate_factor',
     'iccg_method', 'iccg_prescribed_num', 'iccg_prescribed_den', 'cellcount_method',
     'cldtop_adjustment', 'do_radar_ref')
    assert names[-len(lightning):] == list(lightning)
    for name in lightning:
        assert RunConfig.__dataclass_fields__[name].default == LIGHTNING_DEFAULTS[name]
    names = names[:-len(lightning)]
    from woof.config import AER_OPT2_RUN_KEYS
    assert names[-len(AER_OPT2_RUN_KEYS):] == list(AER_OPT2_RUN_KEYS)
    names = names[:-len(AER_OPT2_RUN_KEYS)]
    for name, default in reversed((('scalar_adv_opt', 1), ('urban_ucm_generation', 'wrf_471'),
     ('urban_treeoption', -1), ('urban_ttscheme', -1), ('urban_fvg', -1.0),
     ('urban_table_dir', ''), ('solar_generation', 'wrf_480'),
     ('ysu_version', 'wrf_480'), ('noahmp_snow_age', 'wrf_470'),
     ('urban_water_balance', 'wrf_480'))):
        assert names.pop() == name
        assert RunConfig.__dataclass_fields__[name].default == default, name
        assert not RunConfig.__dataclass_fields__[name].kw_only, name
    assert names.pop() == 'thompson_table_dir'
    assert names.pop() == 'tempo_surface_emission_before'
    assert names.pop() == 'tempo_ml_for_bl_nc'
    assert names.pop() == 'bl_mynn_mixnumcon'
    assert names.pop() == 'bl_mynn_mixaerosols'
    assert names.pop() == 'rrtmg_aer3_type_source'
    assert names.pop() == 'cycling'
    assert names.pop() == 'surface_energy_diag'
    assert names.pop() == 'rrtmg_smoke_manifest'
    assert names.pop() == 'rrtmg_cloud_optics_form'
    assert names.pop() == 'thompson_fork_snow_fall'
    assert names.pop() == 'thompson_version'
    assert names[-3:] == ['swint_opt', 'aer_opt', 'alb_sol']
    for key in names[-3:]:
        assert RunConfig.__dataclass_fields__[key].default == 0
    names = names[:-3]
    assert names.pop() == 'bl_mynn_cloud_tendency_form'
    assert names.pop() == 'bl_mynn_gsd41_unsquared_qtke'
    assert names.pop() == 'bl_mynn_version'
    assert names.pop() == 'ruc_snow'
    assert names.pop() == 'ruc_2m_diagnostic'
    assert names.pop() == 'ruc_qvg_cold_start'
    assert names.pop() == 'ruc_irrigation'
    assert names[-3:] == ['v_sca_adv_order', 'v_mom_adv_order', 'h_mom_adv_order']
    names = names[:-3]
    assert not {'grib2', 'grib2_out', 'verify_visuals'} & set(names)
    assert names.pop() == 'upper_wind_limiter_form'
    assert names[-5:] == ['diff_6th_form', 'diff_6th_factor2', 'mp_zero_out', 'mp_zero_out_thresh', 'mp_zero_out_all']
    fields = RunConfig.__dataclass_fields__
    assert (fields['mp_zero_out'].default, fields['mp_zero_out_all'].default, fields['mp_zero_out_thresh'].default) == (0, 0, 1e-08)
    names = names[:-5]
    assert names[-1] == 'terrain_clock'
    names = names[:-1]
    assert names.pop() == 'mynn_sfclay_variant'
    assert names.pop() == 'fractional_seaice'
    assert names.pop() == 'ruc_soilprop'
    assert names[-2:] == ['spp_conv', 'spp_pbl']
    names = names[:-2]
    assert names[-4:] == ['sf_lake_physics', 'use_lakedepth', 'lakedepth_default', 'lake_min_elev']
    names = names[:-4]
    assert names.pop() == 'use_rap_aero_icbc'
    assert names.pop() == 'scalar_pblmix'
    assert names[-1] == 'zadvect_implicit_variant'
    names = names[:-1]
    assert names[-2:] == ['topo_wind', 'gwd_opt']
    names = names[:-2]
    assert names[-2:] == ['diff_opt', 'mix_full_fields']
    names = names[:-2]
    mosaic_names = names[-3:]
    assert mosaic_names == ['sf_surface_mosaic', 'mosaic_cat', 'mosaic_urban_canopy']
    names = names[:-3]
    assert names[-1] == 'w_crit_cfl'
    names = names[:-1]
    assert names[-1] == 'zadvect_implicit'
    names = names[:-1]
    assert names[-3:] == ['slope_rad', 'topo_shading', 'shadlen']
    names = names[:-3]
    assert names[-3:] == ['sf_urban_physics', 'use_wudapt_lcz', 'num_urban_hi']
    every_name = set(names) - {'adaptive_nest_lattice'} | set(mosaic_names)
    names = names[:-3]
    assert names[-1] == 'adaptive_nest_lattice'
    names = names[:-1]
    assert names[-1] == 'min_time_step_sound'
    names = names[:-1]
    assert names[-2:] == ['relax_timescale_s', 'relax_w']
    assert names[-3] == 'eta_levels'
    assert names[-114:-3] == [
        'nested', 'grid_id', 'top_lid', 'moist_cq',
        'morr_rimed_ice', 'wsm6_hail_opt', 'ra_lw_physics', 'ra_sw_physics',
        'icloud', 'swrad_scat', 'wrf_rrtmg_compatibility', 'num_soil_layers',
        'ra_rrtmg_variant', 'nest_microphysics_transition', 'isftcflx', 'iz0tlnd',
        'usemonalb', 'rdlai2d', 'opt_thcnd', 'bl_mynn_closure',
        'bl_mynn_cloudpdf', 'bl_mynn_mixlength', 'bl_mynn_edmf', 'bl_mynn_edmf_mom',
        'bl_mynn_edmf_tke', 'bl_mynn_mixscalars', 'bl_mynn_cloudmix', 'bl_mynn_mixqt',
        'bl_mynn_output', 'bl_mynn_tkeadvect', 'icloud_bl', 'dveg',
        'opt_crs', 'opt_btr', 'opt_run', 'opt_sfc',
        'opt_frz', 'opt_inf', 'opt_rad', 'opt_alb',
        'opt_snf', 'opt_tbot', 'opt_stc', 'opt_gla',
        'opt_rsf', 'opt_soil', 'opt_pedo', 'opt_crop',
        'opt_irr', 'opt_irrm', 'opt_infdv', 'opt_tdrn',
        'soiltstep', 'noahmp_output', 'noahmp_acc_dt', 'mosaic_lu',
        'mosaic_soil', 'flag_sm_adj', 'spp_lsm', 'nwp_diagnostics',
        'isfflx', 'o3input', 'use_mp_re', 'seaice_albedo_default',
        'rdmaxalb', 'mix_isotropic', 'mix_upper_bound', 'tke_heat_flux',
        'tke_drag_coefficient', 'c_k', 'tke_upper_bound', 'tke_budget',
        'aer_init_opt', 'wif_input_opt', 'sase_flux_diag', 'sase_moist_n2',
        'sase_stable_dissipation', 'sase_additive_dissipation', 'hmix_k_diag', 'km_opt_zero_acknowledgement',
        'inflow_perturbation', 'inflow_perturbation_seed', 'inflow_perturbation_amplitude_scale', 'inflow_perturbation_faces',
        'moist_mix6_off', 'clos_choice', 'ishallow', 'nssl_2moment_on',
        'nssl_hail_on', 'nssl_ccn_on', 'nssl_density_on', 'nssl_3moment',
        'wdm6_hail_opt', 'wdm6_ccn_conc', 'surface_radiation_policy', 'wif_climatology_path',
        'mp28_aerosol_source', 'p3_backend', 'ntiedtke_tiedtke_closure', 'use_adaptive_time_step',
        'step_to_output_time', 'adaptation_domain', 'target_cfl', 'target_hcfl',
        'max_step_increase_pct', 'starting_time_step', 'starting_time_step_den', 'max_time_step',
        'max_time_step_den', 'min_time_step', 'min_time_step_den',
    ]
    for name, default in (('use_adaptive_time_step', False), ('step_to_output_time', True),
     ('adaptation_domain', 1), ('target_cfl', 1.2), ('target_hcfl', 0.84),
     ('max_step_increase_pct', 5), ('starting_time_step', -1),
     ('starting_time_step_den', 0), ('max_time_step', -1), ('max_time_step_den', 0),
     ('min_time_step', -1), ('min_time_step_den', 0)):
        assert RunConfig.__dataclass_fields__[name].default == default, name
    from woof.config import P3_BACKENDS
    assert P3_BACKENDS == ('cuda', 'fused', 'reference')
    from woof.config import MYNN_PBL_OPTION_IDENTITY
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    from woof.config import NOAHMP_OPTION_IDENTITY
    for name, admitted in NOAHMP_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    from woof.config import RUC_OPTION_IDENTITY
    for name, admitted in RUC_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    for entry in GOLDEN.values():
        assert entry['sf_surface_physics'] not in (3, 4)
    for key, entry in GOLDEN.items():
        assert set(entry) == every_name, key
    # equal defaults, independent of the implementation defaults.
    for name, expected in (
            ('nsas_dx_factor', 0),
            ('sas_pgcon', 0.55),
            ('sf_ocean_physics', 0),
            ('bl_mynn_mixnumcon', 0),
            ('bl_mynn_mixaerosols', 0),
            ('rrtmg_aer3_type_source', 'rural'),
            ('rrtmg_smoke_manifest', ''),
            ('rrtmg_cloud_optics_form', 'wrf_461'),
            ('thompson_version', 'wrf_461'),
            ('thompson_fork_snow_fall', 'blend'),
            ('bl_mynn_cloud_tendency_form', 'wrf_461'),
            ('bl_mynn_version', 'wrf_461'),
            ('ruc_snow', 'wrf_461'),
            ('ruc_2m_diagnostic', 'flux'),
            ('ruc_qvg_cold_start', 'wrf'),
            ('ruc_irrigation', 'wrf_461'),
            ('v_sca_adv_order', 3),
            ('v_mom_adv_order', 3),
            ('h_mom_adv_order', 5),
            ('upper_wind_limiter_form', 'wrf_461'),
            ('diff_6th_form', 'wrf_461'),
            ('terrain_clock', 'local_face'),
            ('mynn_sfclay_variant', 'wrf_461'),
            ('fractional_seaice', 0),
            ('ruc_soilprop', 'wrf_45'),
            ('spp_conv', 0),
            ('spp_pbl', 0),
            ('sf_lake_physics', 0),
            ('use_lakedepth', 1),
            ('lakedepth_default', 50.0),
            ('lake_min_elev', 5.0),
            ('scalar_pblmix', 0),
            ('zadvect_implicit_variant', 'wrf_471'),
            ('topo_wind', 0),
            ('gwd_opt', 0),
            ('diff_opt', 2),
            ('sf_surface_mosaic', 0),
            ('mosaic_cat', 3),
            ('mosaic_urban_canopy', 'dominant'),
            ('w_crit_cfl', 1.0),
            ('zadvect_implicit', 0),
            ('slope_rad', 0),
            ('topo_shading', 0),
            ('shadlen', 25000.0),
            ('sf_urban_physics', 0),
            ('use_wudapt_lcz', 0),
            ('num_urban_hi', 15),
            ('min_time_step_sound', 0),
            ('relax_timescale_s', 0.0),
            ('aer_init_opt', 0),
            ('wif_input_opt', 0),
            ('wif_climatology_path', ''),
            ('mp28_aerosol_source', 'auto'),
            ('p3_backend', 'cuda'),
            ('km_opt_zero_acknowledgement', ''),
            ('clos_choice', 0),
            ('ishallow', 0),
            ('surface_radiation_policy', 'required'),
            ('opt_thcnd', 1),
            ('grid_id', 1),
            ('morr_rimed_ice', 1),
            ('wsm6_hail_opt', 0),
            ('nest_microphysics_transition', 'same-scheme-only'),
            ('ra_lw_physics', -1),
            ('ra_sw_physics', -1),
            ('icloud', 1),
            ('swrad_scat', 1.0),
            ('wrf_rrtmg_compatibility', 'none'),
            ('num_soil_layers', 4),
            ('ra_rrtmg_variant', 'rte-rrtmgp'),
            ('nwp_diagnostics', 0),
            ('isfflx', 1),
            ('mix_isotropic', 0),
            ('mix_upper_bound', 0.1),
            ('tke_heat_flux', 0.0),
            ('tke_drag_coefficient', 0.0),
            ('c_k', 0.15),
            ('o3input', 2),
            ('use_mp_re', 1),
            ('seaice_albedo_default', 0.65),
            ('isftcflx', 0),
            ('iz0tlnd', 0),
    ):
        assert RunConfig.__dataclass_fields__[name].default == expected, name

    # singleton defaults, independent of the implementation defaults.
    for name, expected in (
            ('thompson_table_dir', None),
            ('tempo_surface_emission_before', False),
            ('tempo_ml_for_bl_nc', False),
            ('cycling', False),
            ('surface_energy_diag', False),
            ('bl_mynn_gsd41_unsquared_qtke', False),
            ('diff_6th_factor2', None),
            ('use_rap_aero_icbc', False),
            ('mix_full_fields', True),
            ('adaptive_nest_lattice', False),
            ('relax_w', False),
            ('ntiedtke_tiedtke_closure', False),
            ('hmix_k_diag', False),
            ('moist_mix6_off', False),
            ('usemonalb', False),
            ('rdlai2d', False),
            ('nested', False),
            ('top_lid', True),
            ('moist_cq', True),
            ('rdmaxalb', True),
    ):
        assert RunConfig.__dataclass_fields__[name].default is expected, name



def _legacy_run_config_tomls() -> list[Path]:
    """Every TOML in configs/ that IS a legacy RunConfig, and only those.

    "Not an experiment TOML" is not the same statement as "a legacy
    RunConfig TOML", and configs/ now holds a third kind: the
    high-resolution static demo carries only [demo] and [static.highres]
    and is read by tools/run_highres_static_demo.py.  ``load_config``
    refuses it outright -- unknown tables -- so it can never have a
    golden freeze entry, and demanding one was asking for a pin that
    cannot exist.

    The positive test is the loader's own table list rather than a
    hand-kept exclusion list, so a fourth kind of config in this
    directory does not silently join the frozen surface either.
    """

    return [path for path in sorted((REPO / "configs").glob("*.toml"))
            if not is_experiment_toml(path)
            and _declares_a_run_config_table(path)]


def _declares_a_run_config_table(path: Path) -> bool:
    import io
    import tomllib

    from woof.config_authority import read_config_authority

    raw = tomllib.load(io.BytesIO(read_config_authority(path).payload))
    return any(table in raw for table in _KNOWN_TABLES)


def _pop_radiation_port_defaults(actual: dict) -> bool:
    """The lightning and aer_opt = 2 controls (golden/ports-rad) and the
    ocean mixed layer's keys at the defaults under which nothing reads them."""
    from woof.config import AER_OPT2_RUN_KEYS
    from woof.lightning_contract import LIGHTNING_DEFAULTS
    fields = RunConfig.__dataclass_fields__
    # The ocean mixed layer's keys (2.9.1) at the off default nothing reads.
    ocean = ("sf_ocean_physics", "oml_hml0", "oml_gamma", "oml_relaxation_time")
    return (all(actual.pop(name) == default
                for name, default in LIGHTNING_DEFAULTS.items())
            and all(actual.pop(name) == fields[name].default
                    for name in AER_OPT2_RUN_KEYS)
            and all(actual.pop(name) == fields[name].default
                    for name in ocean))


def _pop_capability_campaign_defaults(actual: dict) -> bool:
    """The 2.9.4 capabilities campaign's keys, appended after the golden was
    captured, popped at their defaults: lane/cap-slab-ocean (3-D PWP ocean,
    read only under sf_ocean_physics = 2), lane/cap-waves-couple (ocean door),
    lane/cap-hail (HAILCAST), lane/cap-seaice (prognostic sea ice) and
    lane/cap-lake (lake-state analysis)."""
    expected = {
        "ocean_levels": 30, "ocean_analysis": "",
        "ocean_analysis_source": "rtofs", "ocean_mld_climatology": "woa18",
        "omdt": 1.0, "ocean_z": (), "ocean_t": (), "ocean_s": (),
        "ocean_config": "", "hailcast_opt": -1, "haildt": 60.0,
        "seaice_prognostic": True, "seaice_initial_thickness": 1.0,
        "seaice_new_ice_thickness": 0.5, "seaice_ocean_heat_flux": 2.0,
        "lake_state_source": "auto", "lake_state_path": "",
        "lake_state_accept_licence": "",
    }
    return all(actual.pop(name) == value for name, value in expected.items())


def _pop_topo_radiation_defaults(actual: dict) -> bool:
    """WRF's slope_rad/topo_shading/shadlen, appended after the golden was
    captured, popped at their defaults the way adaptive_nest_lattice is."""
    return (actual.pop("slope_rad") == 0 and actual.pop("topo_shading") == 0
            and actual.pop("shadlen") == 25000.0)


def _check_frozen_defaults(actual):
    """Check and remove reviewed additions for both frozen-config doors."""
    for name, default in _WIND_DEFAULTS.items():
        assert actual.pop(name) == default, name
    assert actual.pop("hailcast_version") == "wrf_39_noaa"
    assert actual.pop("ra_sw_eclipse") == 1
    assert _pop_capability_campaign_defaults(actual)
    # Appended after the campaign at values that change no earlier run:
    # 4c82704f8f's seaice_threshold (100 K converts nothing) and the HRRR v4
    # output controls, off (lane/hrrr-v4-output).
    for field, value in (("seaice_threshold", 100.0), ("prec_acc_dt", 0.0),
                         ("prec_acc_dt1", 0.0), ("gsd_diagnostics", 0),
                         ("reset_interval1", 60), ("history_interval2_s", 0.0),
                         ("history_interval_change_s", 0.0),
                         ("smoke_off_reason", ""),
                         # 2.9.7: the HRRR v4 climate statistics (off) and
                         # tc/nest's archive (read only on a nested domain).
                         ("output_diagnostics", 0), ("diag_int", 15),
                         ("wind_int", 5), ("nest_accumulation_archive", True),
                         # 2.9.10: WRF's ghg_input at the engine's fixed
                         # gases (fix/wrf461-2dom-exact).
                         ("ghg_input", 0), ("umove", 0.0), ("vmove", 0.0)):
        assert actual.pop(field) == value, field
    for field,value in (("urban_ucm_generation","wrf_471"),("urban_treeoption",-1),
                        ("urban_ttscheme",-1),("urban_fvg",-1.),("urban_table_dir","")):
        assert actual.pop(field) == value
    from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS
    for field in CHEM_RUN_FIELDS + FIRE_RUN_FIELDS:
        assert actual.pop(field) == RunConfig.__dataclass_fields__[field].default
    # WRF 4.7/4.8 upstream fixes, default-on (declared default changes).
    assert actual.pop("solar_generation") == "wrf_480"
    assert actual.pop("ysu_version") == "wrf_480"
    assert actual.pop("noahmp_snow_age") == "wrf_470"
    assert actual.pop("urban_water_balance") == "wrf_480"
    assert actual.pop("nsas_dx_factor") == 0
    assert actual.pop("sas_pgcon") == 0.55
    assert actual.pop("surface_energy_diag") is False
    # Declared default correction, 2.9.0: correct mode is the default.
    from woof.correct_mode import CORRECT_KEYS_DEFAULT
    assert actual.pop("wsm6_conservative_surface") is CORRECT_KEYS_DEFAULT
    assert actual.pop("ntiedtke_dry_mass_coupling") is CORRECT_KEYS_DEFAULT
    assert actual.pop("wdm6_conservative_surface") is CORRECT_KEYS_DEFAULT
    assert actual.pop("thompson_cloud_surface_booking") is CORRECT_KEYS_DEFAULT
    # 2.9.1: WB5's keys for WSM3, WSM5 and WSM7, same default.
    assert actual.pop("wsm3_conservative_surface") is CORRECT_KEYS_DEFAULT
    assert actual.pop("wsm5_conservative_surface") is CORRECT_KEYS_DEFAULT
    assert actual.pop("wsm7_conservative_surface") is CORRECT_KEYS_DEFAULT
    # 2.9.7: HN1, the hydrometeor handover numbers, same default.
    assert actual.pop("hydrometeor_handover_numbers") is CORRECT_KEYS_DEFAULT
    # Declared default correction, 2.8.8 (test above).
    assert actual.pop("terrain_clock") == "local_face"
    assert actual.pop("use_rap_aero_icbc") is False
    assert actual.pop("diff_opt") == 2
    assert actual.pop("scalar_pblmix") == 0
    assert actual.pop("mix_full_fields") is True
    assert actual.pop("adaptive_nest_lattice") is False
    assert actual.pop("zadvect_implicit") == 0
    assert actual.pop("w_crit_cfl") == 1.0
    assert actual.pop("topo_wind") == 0 and actual.pop("gwd_opt") == 0
    assert actual.pop("zadvect_implicit_variant") == "wrf_471"
    assert actual.pop("sf_lake_physics") == 0
    assert actual.pop("use_lakedepth") == 1
    assert actual.pop("lakedepth_default") == 50.0
    assert actual.pop("lake_min_elev") == 5.0
    assert actual.pop("spp_conv") == 0 and actual.pop("spp_pbl") == 0
    assert actual.pop("swint_opt") == 0 and actual.pop("aer_opt") == 0
    assert actual.pop("rrtmg_cloud_optics_form") == "wrf_461"
    assert actual.pop("rrtmg_aer3_type_source") == "rural"
    assert actual.pop("bl_mynn_mixaerosols") == 0
    assert actual.pop("bl_mynn_mixnumcon") == 0
    assert actual.pop("tempo_ml_for_bl_nc") is False
    assert actual.pop("tempo_surface_emission_before") is False
    assert actual.pop("thompson_table_dir") is None
    assert actual.pop("rrtmg_smoke_manifest") == ""
    assert actual.pop("cycling") is False
    assert actual.pop("alb_sol") == 0
    # The RUC SOILPROP lineage changed its default: a non-RUC case
    # never reads it; a RUC case changes answers (CHANGELOG).
    assert actual.pop("ruc_soilprop") == "wrf_45"
    # The Thompson generation, appended at the v4.6.1 default.
    assert actual.pop("thompson_version") == "wrf_461"
    assert actual.pop("thompson_fork_snow_fall") == "blend"
    assert actual.pop("bl_mynn_version") == "wrf_461"
    assert actual.pop("bl_mynn_gsd41_unsquared_qtke") is False
    assert actual.pop("bl_mynn_cloud_tendency_form") == "wrf_461"
    # The MYNN surface-layer generation, at the form every build ran.
    assert actual.pop("mynn_sfclay_variant") == "wrf_461"
    assert actual.pop("fractional_seaice") == 0
    # The WRF v4.6.1 filter form and no zero-out pass, as before.
    from woof.config import G3_RUN_DEFAULTS
    assert {k: actual.pop(k) for k in G3_RUN_DEFAULTS} == G3_RUN_DEFAULTS
    assert _pop_fork_dycore_defaults(actual)
    # Generic RUC preserves irrigation and snow from WRF v4.6.1.
    assert actual.pop("ruc_irrigation") == "wrf_461"
    assert actual.pop("ruc_qvg_cold_start") == "wrf"
    assert actual.pop("ruc_2m_diagnostic") == "flux"
    assert actual.pop("ruc_snow") == "wrf_461"
    assert actual.pop("v_sca_adv_order") == 3
    assert actual.pop("v_mom_adv_order") == 3
    assert actual.pop("h_mom_adv_order") == 5
    assert _pop_topo_radiation_defaults(actual)
    assert _pop_radiation_port_defaults(actual)


def test_every_existing_legacy_toml_resolves_identically():
    """Re-resolve every legacy TOML in configs/ and compare dataclasses
    (as field dicts) against the pre-change golden snapshot."""
    tomls = sorted((REPO / "configs").glob("*.toml"))
    assert tomls, "configs/ directory is empty?"
    legacy = _legacy_run_config_tomls()
    if not legacy and not any(key.startswith("configs/") for key in _golden_in_this_tree()):
        pytest.skip("every legacy TOML the golden file freezes is a case configuration this tree leaves out "
                    "(tests/_woof_left_out.txt); this case runs in the engine's tree")
    assert legacy, "no legacy TOMLs found in configs/"
    for path in legacy:
        key = f"configs/{path.name}"
        assert key in GOLDEN, (
            f"{key} has no golden freeze entry: pin new legacy TOMLs in "
            "tests/data/config_freeze_golden.json consciously.")
        cfg = load_config(path)
        actual = dataclasses.asdict(cfg)
        _check_frozen_defaults(actual)
        # These legacy files omit CQ. Its default-on defect fix is the
        # only approved change from the recorded field dictionary.
        assert actual == dict(GOLDEN[key], moist_cq=True), key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_frozen_case_constructed_configs_resolve_identically():
    """Every frozen verify-case RunConfig constructor re-resolves to the
    pre-change golden values (frozen cases never migrate to the
    experiment path -- they construct RunConfig directly)."""
    for key, ctor in _frozen_constructors().items():
        cfg = ctor()
        assert key in GOLDEN, key
        actual = dataclasses.asdict(cfg)
        _check_frozen_defaults(actual)
        # The constructors omit CQ too. Dry cases remain numerically
        # unchanged because their qv=None bypass does no CQ work.
        assert actual == dict(GOLDEN[key], moist_cq=True), key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_golden_inventory_matches_known_frozen_surface():
    """The golden file covers exactly the frozen surface: every known
    constructor plus every legacy TOML (no stale or orphaned pins)."""
    expected = set(_frozen_constructors())
    expected |= {f"configs/{p.name}"
                 for p in _legacy_run_config_tomls()}
    assert _golden_in_this_tree() == expected


def test_parameter_set_is_appended_to_the_experiment_config():
    from woof.experiment import ExperimentConfig
    names = list(ExperimentConfig.__dataclass_fields__)
    assert names[-2:] == ["smooth_cg_topo", "physics_params"]
    assert ExperimentConfig.__dataclass_fields__["physics_params"].default is None
    assert "physics_params" not in RunConfig.__dataclass_fields__


@pytest.mark.parametrize("theme", ["paper", "woof-light", "woof-dark", "inherited"])
def test_renderer_environment_preserves_emitted_config_bytes(monkeypatch, tmp_path, theme):
    """Rendering choices cannot rewrite the ordinary forecast configuration.

    A paired qualification run may provide its staging receipt to check the
    exact same emissions across the two revisions without repinning physics
    selectors when the staging branch advances.
    """
    from tools.renderer_recovery_identity import emitted_config_bytes

    before = emitted_config_bytes()
    reference = os.environ.get("WOOF_RENDERER_IDENTITY_REFERENCE")
    if reference:
        receipt = json.loads(Path(reference).read_text(encoding="utf-8"))
        assert before == receipt["identity"]["emitted_config_bytes"]
    if theme == "inherited":
        theme_file = tmp_path / "site-theme.json"
        theme_file.write_text(json.dumps({"extends": "woof-dark", "name": "site"}),
                              encoding="utf-8")
        theme = str(theme_file)
    for key, value in {
            "RUSTWX_THEME": theme, "RUSTWX_RADAR_COLORS": "nws",
            "RUSTWX_WIND_STREAMLINES": "1"}.items():
        monkeypatch.setenv(key, value)
    assert emitted_config_bytes() == before


@pytest.mark.parametrize("payload", [
    b'[grid]\nnx = 32\nny = 1\nnz = 16\ndx = 1000.0\ndy = 1000.0\nztop = 8000.0\n\n'
    b'[dynamics]\ndt = 2.0\n\n[run]\nrun_seconds = 10.0\n',
    b'[experiment]\nname = "area"\nrun_seconds = 120.0\nfeedback = 0\n\n'
    b'[shared]\nnz = 16\nztop = 8000.0\nmoist = true\nspp_conv = 0\nspp_pbl = 0\n'
    b'sf_lake_physics = 0\n\n[[domain]]\ngrid_id = 1\nparent_id = 0\n'
    b'nx = 32\nny = 24\ndx = 2000.0\ntime_step = 10\n',
], ids=["generic", "non-feature"])
def test_generic_and_non_feature_emitted_config_bytes_are_frozen(payload):
    """Formatting and ordering stay fixed as well as parsed field values.

    Catalog naming and route changes must not change a configuration emitted
    for an ordinary run, including explicitly disabled optional physics.
    These are exact UTF-8 byte pins, so an equal parsed dictionary alone is
    insufficient.
    """
    from woof.toml_document import emit_experiment_toml

    raw = tomllib.loads(payload.decode("utf-8"))
    assert emit_experiment_toml(raw).encode("utf-8") == payload


@pytest.mark.parametrize("option,default", [("grib2", False), ("verify_visuals", True)])
def test_output_choice_leaves_forecast_identity_unchanged(tmp_path, option, default):
    """The output choice changes neither forecast settings nor restart identity."""
    from types import SimpleNamespace

    from woof.core.model import experiment_fingerprint, restart_identity_payload
    from woof.io.restart import configuration_echo
    from woof.runplan import PLAN_SCHEMA, build_plan, resolve_plan
    from test_case_data import make_case_toml

    config = make_case_toml(tmp_path)
    default_plan = build_plan({
        "schema": PLAN_SCHEMA, "name": "output-choice", "route": "experiment",
        "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
    }, source="output-choice", base_dir=tmp_path, sha256="0" * 64)
    assert default_plan.run_options[option] is default
    experiments = []
    for selected in (False, True):
        plan = build_plan({
            "schema": PLAN_SCHEMA, "name": "output-choice",
            "route": "experiment", "config": {"path": str(config)},
            "output_root": str(tmp_path / "run"),
            "run_options": {option: selected},
        }, source="output-choice", base_dir=tmp_path, sha256="0" * 64)
        assert plan.run_options[option] is selected
        _, experiment, _ = resolve_plan(plan, require_inputs=False)
        experiments.append(experiment)

    without, with_export = experiments
    assert without == with_export
    assert restart_identity_payload(without) == restart_identity_payload(with_export)
    catalog = SimpleNamespace(run_provenance={"inputs": "same-forcing-window"})
    assert experiment_fingerprint(without, catalog) == \
        experiment_fingerprint(with_export, catalog)
    for original, exported in zip(without.domains, with_export.domains, strict=True):
        before = configuration_echo(original.run)
        after = configuration_echo(exported.run)
        assert json.dumps(before, allow_nan=False).encode("utf-8") == \
            json.dumps(after, allow_nan=False).encode("utf-8")
