"""A191: every checkpoint writer echoes a run's configuration the same way.

The breakage this prevents: the resident writer dropped nine default-off
RunConfig keys from a checkpoint's ``config`` echo and the streamed writer
wrote ``dataclasses.asdict(cfg)`` whole, so two checkpoints of one state,
one per memory road, compared unequal though every array matched
(tests/test_streamed_relocation_rebuild_gpu.py's
test_live_bounded_move_keeps_owner_and_continues_exactly failed on the
2.8.1 bench cards).  The out-of-core store writer had the same raw echo.

Each case writes the same NumPy-backed state through the resident writer
(``woof.io.restart.write_restart``), the streamed writer
(``tilestream.restart_stream.write_streamed_restart``) and the store writer
(``tilestream.checkpoint.write_store_restart``) and requires one echo from
all three, at the defaults and with each default-off field moved.  CPU only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.io import restart
from test_restart import (
    _cfg, _fill_setup, _identity_bound_physics_state, _shim_driver_state)


#: lane/cap-seaice: bound wherever a land surface runs prognostic sea ice.
_SEAICE_BOUND = ("seaice_prognostic", "seaice_initial_thickness",
                 "seaice_new_ice_thickness", "seaice_ocean_heat_flux")

#: Every default-off field restart._drop_default_off_run_keys drops at its
#: default, moved, as (overrides, the keys the echo must then carry).
MOVED = {
    "umove": (dict(umove=12.0), {"umove"}),
    "vmove": (dict(vmove=-8.0), {"vmove"}),
    "output_diagnostics": (dict(output_diagnostics=1), {"output_diagnostics", "diag_int", "wind_int"}),
    "diag_int": (dict(diag_int=30), {"diag_int"}),
    "wind_int": (dict(wind_int=10), {"wind_int"}),
    "thompson_version": (dict(mp_physics=28, moist=True, thompson_version="wrf_39_noaa"), {"thompson_version"}),
    "thompson_fork_snow_fall": (dict(mp_physics=28, moist=True, thompson_version="wrf_39_noaa", thompson_fork_snow_fall="wrf_39_noaa"), {"thompson_version", "thompson_fork_snow_fall"}),
    "upper_wind_limiter_form": (dict(upper_wind_limiter_form="noaa_wrf39"),
                                {"upper_wind_limiter_form"}),
    "adaptive_nest_lattice": (dict(adaptive_nest_lattice=True),
                              {"adaptive_nest_lattice"}),
    "zadvect_implicit": (dict(zadvect_implicit=1), {"zadvect_implicit"}),
    "zadvect_implicit_variant": (
        dict(zadvect_implicit=1, zadvect_implicit_variant="wrf_legacy"),
        {"zadvect_implicit", "zadvect_implicit_variant"}),
    "ruc_irrigation": (dict(ruc_irrigation="wrf_45"), {"ruc_irrigation"}),
    "ruc_qvg_cold_start": (dict(ruc_qvg_cold_start="air"),
                           {"ruc_qvg_cold_start"}),
    "ruc_2m_diagnostic": (dict(ruc_2m_diagnostic="log_profile"),
                          {"ruc_2m_diagnostic"}),
    "ruc_snow": (dict(ruc_snow="wrf_45"), {"ruc_snow"}),
    "w_crit_cfl": (dict(zadvect_implicit=1, w_crit_cfl=2.0),
                   {"zadvect_implicit", "w_crit_cfl"}),
    "slope_rad": (dict(slope_rad=1), {"slope_rad"}),
    "topo_shading": (dict(slope_rad=1, topo_shading=1, shadlen=20000.0),
                     {"slope_rad", "topo_shading", "shadlen"}),
    # YSU runs here, so its generation (default wrf_480, 2.9.0) is bound too.
    "topo_wind": (dict(bl_pbl_physics=1, sf_sfclay_physics=1, topo_wind=1),
                  {"topo_wind", "ysu_version"}),
    "gwd_opt": (dict(gwd_opt=1), {"gwd_opt"}),
    # Noah runs here, so lane/cap-seaice's default-on prognostic sea ice
    # and its parameters are read and bound too (_SEAICE_BOUND).
    "sf_surface_mosaic": (dict(sf_surface_physics=2, sf_surface_mosaic=1,
                               mosaic_cat=2),
                          {"sf_surface_mosaic", "mosaic_cat",
                           *_SEAICE_BOUND}),
    # The single-layer urban model runs here, so its water-balance
    # generation (default wrf_480, 2.9.0) is bound too.
    "mosaic_urban_canopy": (dict(sf_surface_physics=2, sf_urban_physics=1,
                                 sf_surface_mosaic=1,
                                 mosaic_urban_canopy="every_tile"),
                            {"sf_surface_mosaic", "mosaic_cat",
                             "urban_water_balance",
                             "mosaic_urban_canopy", *_SEAICE_BOUND}),
    "diff_opt": (dict(diff_opt=1), {"diff_opt", "mix_full_fields"}),
    "mix_full_fields": (dict(diff_opt=1, mix_full_fields=False),
                        {"diff_opt", "mix_full_fields"}),
    "scalar_pblmix": (dict(scalar_pblmix=1), {"scalar_pblmix"}),
    "use_rap_aero_icbc": (dict(use_rap_aero_icbc=True), {"use_rap_aero_icbc"}),
    "sf_lake_physics": (dict(sf_lake_physics=1),
                        {"sf_lake_physics", "use_lakedepth",
                         "lakedepth_default", "lake_min_elev"}),
    "spp_conv": (dict(spp_conv=1), {"spp_conv"}),
    "spp_pbl": (dict(spp_pbl=1), {"spp_pbl"}),
    "ruc_soilprop": (dict(ruc_soilprop="wrf_461"), {"ruc_soilprop"}),
    # terrain_clock is no longer default-off: from 2.8.8 its default
    # ("local_face") is echoed and "measured" drops out
    # (test_the_terrain_clock_default_is_echoed_and_measured_drops_out).
    "bl_mynn_cloud_tendency_form": (
        dict(bl_mynn_version="gsd_41", bl_mynn_cloud_tendency_form="gsd_41"),
        {"bl_mynn_version", "bl_mynn_cloud_tendency_form"}),
    "bl_mynn_version": (dict(bl_mynn_version="gsd_41"), {"bl_mynn_version"}),
    "bl_mynn_gsd41_unsquared_qtke": (dict(bl_mynn_gsd41_unsquared_qtke=True),
                                     {"bl_mynn_gsd41_unsquared_qtke"}),
    "mynn_sfclay_variant": (dict(mynn_sfclay_variant="gsl_wrf39"),
                            {"mynn_sfclay_variant"}),
    "cycling": (dict(bl_mynn_version="gsd_41", cycling=True),
                {"bl_mynn_version", "cycling"}),
    "diff_6th_form": (dict(diff_6th_form="noaa_wrf39", diff_6th_factor2=0.04),
                      {"diff_6th_form", "diff_6th_factor2"}),
    "mp_zero_out": (dict(mp_zero_out=2),
                    {"mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all"}),
    "v_sca_adv_order": (dict(v_sca_adv_order=5), {"v_sca_adv_order"}),
    "v_mom_adv_order": (dict(v_mom_adv_order=5), {"v_mom_adv_order"}),
    "rrtmg_cloud_optics_form": (
        dict(rrtmg_cloud_optics_form="noaa_wrf39"),
        {"rrtmg_cloud_optics_form"}),
    "swint_opt": (dict(swint_opt=1), {"swint_opt"}),
    "aer_opt": (dict(aer_opt=3, mp_physics=28, moist=True,
                     ra_physics=4, ra_rrtmg_variant="rrtmg_legacy"),
                {"aer_opt"}),
    "rrtmg_aer3_type_source": (
        dict(aer_opt=3, mp_physics=28, moist=True, ra_physics=4,
             ra_rrtmg_variant="rrtmg_legacy",
             rrtmg_aer3_type_source="landmask"),
        {"aer_opt", "rrtmg_aer3_type_source"}),
    "alb_sol": (dict(alb_sol=1), {"alb_sol"}),
}

#: The keys an echo at every default leaves out.
DEFAULT_OFF = frozenset({
    "output_diagnostics", "diag_int", "wind_int",
    # 2.9.10: WRF's ghg_input at the engine's fixed gases (0), absent as in
    # a header written before the key (checkpoint_identity).
    "ghg_input",
    "umove", "vmove",
    # 2.9.11: the Fitch wind-farm controls with farm physics off
    # (labs/windfarm, woof.wind_table.drop_inert_config).
    "wind_farm_scheme", "wind_farms", "wind_farm_table",
    "windfarm_tke_factor", "wind_farm_max_dx_m",
    # The ocean mixed layer (2.9.1) off: dropped, as in a header written
    # before the keys existed.
    "sf_ocean_physics", "oml_hml0", "oml_gamma", "oml_relaxation_time",
    # Inactive G3 fields preserve pre-G3 checkpoint identities.
    "cugd_avedx",
    "maxiens",
    "maxens",
    "maxens2",
    "maxens3",
    "ensdim",
    "cu_diag",
    "convtrans_avglen_m",
    # Inactive New SAS knobs (read only by cu_physics = 96) preserve
    # pre-NSAS checkpoint identities.
    "nsas_dx_factor", "sas_pgcon",
    # The aer_opt = 2 controls and the lightning package (golden/ports-rad),
    # absent while unread, as in a 2.8.9 header
    # (config.drop_inactive_radiation_port_keys).
    "aer_type", "aer_aod550_opt", "aer_angexp_opt", "aer_ssa_opt",
    "aer_asy_opt", "aer_aod550_val", "aer_angexp_val", "aer_ssa_val",
    "aer_asy_val", "aer_auxinput15", "aer_auxinput15_interval",
    "lightning_option", "lightning_dt", "lightning_start_seconds",
    "flashrate_factor", "iccg_method", "iccg_prescribed_num",
    "iccg_prescribed_den", "cellcount_method", "cldtop_adjustment",
    "do_radar_ref",
    "thompson_version", "thompson_fork_snow_fall", "thompson_table_dir",
    "upper_wind_limiter_form",
    "adaptive_nest_lattice", "zadvect_implicit", "w_crit_cfl", "slope_rad",
    "topo_shading", "shadlen", "topo_wind", "gwd_opt", "sf_surface_mosaic",
    "mosaic_cat", "mosaic_urban_canopy", "diff_opt", "mix_full_fields",
    "zadvect_implicit_variant",
    # 2.8.5: post-PBL scalar diffusion, the analyzed aerosol start and the
    # CLM lake quartet, each absent at its default as in a 2.8.4 header.
    "scalar_pblmix", "use_rap_aero_icbc", "sf_lake_physics", "use_lakedepth",
    "lakedepth_default", "lake_min_elev",
    # MYNN-EDMF's split number-species mixing keys, absent at 0.
    "bl_mynn_mixaerosols", "bl_mynn_mixnumcon",
    "spp_conv", "spp_pbl",
    "ruc_soilprop",
    # terrain_clock left this set in 2.8.8: its default, "local_face",
    # is echoed, and "measured" (every earlier header's clock) drops out.
    "bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke", "bl_mynn_cloud_tendency_form",
    "mynn_sfclay_variant",
    # TEMPO's optional coupling arms, absent while off.
    "tempo_ml_for_bl_nc", "tempo_surface_emission_before",
    # The RUC sea-ice threshold switch, absent at 0 as in a 2.8.5 header.
    "fractional_seaice",
    # real.exe's seaice_threshold at its 100 K Registry default (converts
    # nothing), and the operational output controls while off.
    "seaice_threshold", "prec_acc_dt", "prec_acc_dt1", "gsd_diagnostics",
    "reset_interval1", "history_interval2_s", "history_interval_change_s",
    # The per-run smoke-off sentence, absent while empty.
    "smoke_off_reason",
    # The WRF v4.6.1 sixth-order filter form and no mp_zero_out pass.
    "diff_6th_form", "diff_6th_factor2",
    "mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all",
    "ruc_irrigation", "ruc_qvg_cold_start", "ruc_2m_diagnostic",
    "ruc_snow",
    # WRF's advection orders, dropped by the echo at their Registry
    # defaults (3, 3, 5) as in a header written before they existed.
    "v_sca_adv_order", "v_mom_adv_order", "h_mom_adv_order",
    "swint_opt", "aer_opt", "rrtmg_cloud_optics_form", "rrtmg_smoke_manifest",
    "alb_sol", "rrtmg_aer3_type_source",
    # Output-only surface energy carriers, dropped by the echo when off.
    "surface_energy_diag",
    "cycling",
    # 2.9.0 (golden/wrf-fixes).  WRF's only implemented scalar advection
    # operator and the WRF 4.7.1 urban canopy drop at their defaults; the
    # YSU, Noah-MP snow-aging and SLUCM water selectors are scheme-scoped
    # and absent while their scheme is off (this fixture runs none of
    # them).  solar_generation is NOT here: its default, "wrf_480", is
    # echoed, and "wrf_461" (every earlier header's sun) drops out.
    "scalar_adv_opt", "urban_ucm_generation", "urban_treeoption",
    "urban_ttscheme", "urban_fvg", "urban_table_dir", "ysu_version",
    "noahmp_snow_age", "urban_water_balance",
    # 2.9.0 correct mode (woof/correct_mode.py): default on, but echoed only
    # where its scheme reads it; this fixture runs neither WSM6 nor New
    # Tiedtke, so both are absent, as in a 2.8.9 header.
    "wsm6_conservative_surface", "ntiedtke_dry_mass_coupling",
    "wdm6_conservative_surface", "thompson_cloud_surface_booking",
    "wsm3_conservative_surface", "wsm5_conservative_surface",
    "wsm7_conservative_surface",
    # 2.9.7 HN1 (hydrometeor handover numbers): read by Thompson and
    # Morrison only, so absent here, as in a 2.9.6 header.
    "hydrometeor_handover_numbers",
    # 2.9.7 tc/nest: the nest accumulation archive is echoed only on a nested
    # domain with the archive on (restart._drop_inert_nest_archive); this
    # fixture is not nested, so it is absent, as in a 2.9.6 header.
    "nest_accumulation_archive",
    # 2.9.4 capabilities campaign, each absent at its default as in a 2.9.3
    # header: lane/cap-slab-ocean's 3-D PWP controls (dropped with the
    # ocean off), lane/cap-eclipse's default-on shadow (explicit 0 binds),
    # lane/cap-waves-couple's ocean door, lane/cap-hail's output-only
    # HAILCAST selectors, lane/cap-seaice's keys (read only under a land
    # surface) and lane/cap-lake's lake-state analysis.
    "ocean_levels", "ocean_analysis", "ocean_analysis_source",
    "ocean_mld_climatology", "omdt", "ocean_z", "ocean_t", "ocean_s",
    "ra_sw_eclipse", "ocean_config", "hailcast_opt", "haildt",
    # fix/hailcast: the HAILCAST generation, absent at its default like the
    # other output-only HAILCAST selectors.
    "hailcast_version",
    "seaice_prognostic", "seaice_initial_thickness",
    "seaice_new_ice_thickness", "seaice_ocean_heat_flux",
    "lake_state_source", "lake_state_path", "lake_state_accept_licence",
})


def _checkpoint_state(cfg, monkeypatch):
    """The same bound NumPy-backed setup for echo and restore witnesses."""
    state, driver = _shim_driver_state(cfg, monkeypatch)
    if cfg.ra_rrtmg_variant == "rrtmg_legacy" and (
            cfg.ra_physics > 0 or cfg.ra_lw_physics > 0 or cfg.ra_sw_physics > 0):
        # The selected legacy adapter has a real stock identity even though
        # these CPU tests never execute radiation. Bind its required setup
        # just as the restart identity fixture binds the modern adapter.
        from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
        setup = _identity_bound_physics_state(cfg, monkeypatch)[1].radiation_callable
        radiation = object.__new__(RRTMGLegacyRadiation)
        for name in ("start_time", "latitude_deg", "longitude_deg"):
            setattr(radiation, name, getattr(setup, name))
        driver.radiation_callable = radiation
    if cfg.sf_surface_physics == 2:
        driver.noah_params = _identity_bound_physics_state(
            cfg, monkeypatch)[1].noah_params
    if cfg.sf_surface_mosaic:
        # The tile identity a mosaic header binds, without the tiles: this
        # file reads the configuration echo, not the tile carriers
        # (tests/test_noah_mosaic_driver.py checkpoints those on a card).
        from woof.core.noah_mosaic import MosaicCategories
        driver.noah_mosaic = SimpleNamespace(
            mosaic_cat=cfg.mosaic_cat, xice_threshold=0.5,
            categories=MosaicCategories(isurban=13, isice=15, iswater=17,
                                        natural=7, lcz=()))
    _fill_setup(state)
    return state, driver


def _echoes(cfg, monkeypatch, tmp_path, *, surface_receipt=None) -> dict[str, dict]:
    """The ``config`` each of the three writers puts in its header."""
    from tilestream import checkpoint, physics_inventory, restart_stream

    state, _driver = _checkpoint_state(cfg, monkeypatch)
    if cfg.umove or cfg.vmove:
        # _fill_setup deliberately writes nonzero bytes into every setup
        # array; active moving-frame fixtures must represent flat ground.
        state.ht[...] = 0.0
    if surface_receipt is not None:
        state._ensemble_surface_state = surface_receipt
    paths = {"resident": restart.write_restart(
        tmp_path / "resident.npz", state, cfg)}
    # After the resident write, which seeds diff_opt = 1's thermal
    # reference on the state it checkpoints.
    store = {key: value.copy() for key, value in
             physics_inventory.carrier_manifest(state).items()}
    scalars = physics_inventory.carrier_scalars(state)
    paths |= {
        "streamed": restart_stream.write_streamed_restart(
            tmp_path / "streamed.npz", store, cfg, scalars=scalars,
            setup=restart_stream.capture_domain_setup(state),
            template_state=state, check_pinned=False).path,
        "store": checkpoint.write_store_restart(
            tmp_path / "store.npz", store, scalars,
            checkpoint.DomainSetup.capture(state, cfg), cfg),
    }
    return {road: restart.read_restart_header(path)["config"]
            for road, path in paths.items()}


def test_every_writer_echoes_the_defaults_alike(monkeypatch, tmp_path):
    cfg = _cfg()
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
        # Every writer encodes the header with this serializer. Keep the
        # configuration member's UTF-8 bytes and key order fixed too.
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == \
            json.dumps(expected, allow_nan=False).encode("utf-8"), road
    # At the defaults every default-off key is absent, as in a header
    # written before the field existed, and every other field is echoed.
    assert not DEFAULT_OFF & set(expected)
    # Output export options are not checkpointed forecast settings.
    assert not {"grib2", "grib2_out", "verify_visuals"} & set(expected)
    from woof.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS
    assert set(expected) == ({f.name for f in dataclasses.fields(cfg)}
                             - DEFAULT_OFF - set(CHEM_RUN_FIELDS)
                             - set(FIRE_RUN_FIELDS))


def test_inactive_fire_does_not_change_prior_checkpoint_identity():
    """ifire 0 omits the whole fire block, so pre-fire headers still match."""
    from dataclasses import replace
    from woof.config import FIRE_RUN_FIELDS

    cfg = _cfg()
    prior = dataclasses.asdict(cfg)
    for name in FIRE_RUN_FIELDS:
        prior.pop(name)
    unused = replace(cfg, fire_num_ignitions=2, sr_x=4, sr_y=4)
    assert restart.configuration_echo(unused) == restart.configuration_echo(cfg)
    restart._require_config_match(prior, unused, "fire-off-checkpoint")


@pytest.mark.parametrize("live", [{"ifire": 2, "sr_x": 4, "sr_y": 4}])
def test_active_fire_refuses_a_fire_off_checkpoint_both_ways(live):
    from woof.config import FIRE_RUN_FIELDS

    off = restart.configuration_echo(_cfg())
    assert not set(FIRE_RUN_FIELDS) & set(off)
    on = _cfg(**live)
    with pytest.raises(restart.RestartMismatchError, match="ifire"):
        restart._require_config_match(off, on, "fire-off-to-on")
    on_echo = restart.configuration_echo(on)
    assert set(FIRE_RUN_FIELDS) <= set(on_echo)
    with pytest.raises(restart.RestartMismatchError, match="ifire"):
        restart._require_config_match(on_echo, _cfg(), "fire-on-to-off")


@pytest.mark.parametrize("fork", [{}, dict(simple_dir_fdb=1, enh_vermix=True,
                                         flam_part=0.75, aer_drydep_opt=111)])
def test_every_writer_echoes_active_chemistry_alike(monkeypatch, tmp_path, fork):
    from woof.config import CHEM_RUN_FIELDS

    # The smoke fork's keys (c363fd55ff, fc1e2df360) are absent at their
    # defaults so chem checkpoints written before them keep their digest
    # (tests/test_restart_287_header.py); stated, every writer echoes them.
    fork = restart.CHEM_FORK_RUN_DEFAULTS
    for cfg, fork_echoed in (
            (_cfg(chem_sets="tracer_test"), set()),
            (_cfg(chem_sets="tracer_test", aer_drydep_opt=111,
                  flam_part=0.5), {"aer_drydep_opt", "flam_part"})):
        expected = restart.configuration_echo(cfg)
        assert set(CHEM_RUN_FIELDS) - set(fork) <= expected.keys()
        assert set(fork) & expected.keys() == fork_echoed
        for road, echo in _echoes(cfg, monkeypatch, tmp_path).items():
            assert echo == expected, road


def test_inactive_chemistry_does_not_change_prior_checkpoint_identity():
    from dataclasses import replace
    from woof.config import CHEM_RUN_FIELDS

    cfg = _cfg()
    prior = dataclasses.asdict(cfg)
    for name in CHEM_RUN_FIELDS:
        prior.pop(name)
    unused = replace(cfg, chem_adv_opt=2, kemit=17, dust_alpha=3.0,
                     aer_ra_feedback=1)
    assert restart.configuration_echo(unused) == restart.configuration_echo(cfg)
    restart._require_config_match(prior, unused, "chemistry-off-checkpoint")


@pytest.mark.parametrize("chem_sets", ["", "tracer_test"])
def test_every_writer_default_fork_echo_is_admitted(
        monkeypatch, tmp_path, chem_sets):
    cfg = _cfg(chem_sets=chem_sets)
    for road, echo in _echoes(cfg, monkeypatch, tmp_path).items():
        restart._require_config_match(echo, cfg, tmp_path / f"{road}.npz")


def test_default_fork_checkpoint_restores_the_actual_resident_words(
        monkeypatch, tmp_path):
    from test_restart import _fill_serialized

    cfg = _cfg()
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20261004)
    expected = {name: getattr(state, name).tobytes()
                for name in restart.serialized_state_attrs(state)
                if getattr(state, name, None) is not None}
    path = restart.write_restart(tmp_path / "default-fork.npz", state, cfg)
    header = restart.read_restart_header(path)
    assert not {"diff_6th_form", "diff_6th_factor2", "upper_wind_limiter_form",
                "mp_zero_out", "mp_zero_out_all", "mp_zero_out_thresh"} & set(header["config"])
    fresh, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name, value in expected.items():
        assert getattr(fresh, name).tobytes() == value, name


@pytest.mark.parametrize("stored,live,field", [
    ({}, {"diff_6th_form": "noaa_wrf39"}, "diff_6th_form"),
    ({}, {"diff_6th_factor2": 0.04}, "diff_6th_factor2"),
    ({}, {"upper_wind_limiter_form": "noaa_wrf39"}, "upper_wind_limiter_form"),
    ({}, {"mp_zero_out": 1}, "mp_zero_out"),
    ({"mp_zero_out": 1}, {"mp_zero_out": 1, "mp_zero_out_all": 1}, "mp_zero_out_all"),
    ({"mp_zero_out": 1}, {"mp_zero_out": 1, "mp_zero_out_thresh": 2.0e-8}, "mp_zero_out_thresh"),
])
def test_nondefault_fork_checkpoint_changes_still_refuse(stored, live, field):
    echo = restart.configuration_echo(_cfg(**stored))
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(echo, _cfg(**live), "changed-fork-checkpoint")
    reverse = restart.configuration_echo(_cfg(**live))
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(reverse, _cfg(**stored), "changed-fork-checkpoint")


def test_inactive_zero_out_controls_follow_the_writer_omission_rule():
    cfg = _cfg(mp_zero_out=0, mp_zero_out_all=1, mp_zero_out_thresh=2.0e-8)
    echo = restart.configuration_echo(cfg)
    restart._require_config_match(echo, _cfg(), "inactive-zero-out-checkpoint")


@pytest.mark.parametrize("case", ["generic_dry", "non_feature_moist"])
@pytest.mark.parametrize("inactive_nondefaults", [False, True])
def test_chemistry_off_emitted_config_bytes_match_recovery_staging(
        case, inactive_nondefaults):
    """Actual checkpoint configuration JSON is unchanged by the opt-in block."""
    from woof.config import RunConfig

    baseline = json.loads((Path(__file__).parent / "data" /
                           "aq_off_emitted_config_bytes.json").read_text())
    assert baseline["staging_sha"] == "00bbfcf9a91723125b4a1328248e90b66e611256"
    row = baseline["cases"][case]
    # These bytes were pinned under the measured terrain clock, the default
    # until 2.8.8; the local-face default is echoed and is checked in
    # test_the_terrain_clock_default_is_echoed_and_measured_drops_out.
    # ... and under the WRF 4.6.1 solar generation, the default until 2.9.0;
    # the WRF 4.8 default is echoed (tests/test_restart.py
    # test_the_wrf480_solar_default_moves_only_the_solar_binding).
    # ... and before the 2.9.7 hydrometeor handover numbers (HN1), which
    # bind under the moist case's Thompson; off is every earlier header's ring.
    cfg = RunConfig(**{"terrain_clock": "measured",
                       "solar_generation": "wrf_461",
                       "hydrometeor_handover_numbers": False,
                       **row["constructor"]})
    if inactive_nondefaults:
        cfg = dataclasses.replace(cfg, chem_adv_opt=2, kemit=17,
                                  dust_alpha=3.0, aer_ra_feedback=1)
    encoded = json.dumps(restart.configuration_echo(cfg), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    assert encoded == row["config_json"].encode()
    assert hashlib.sha256(encoded).hexdigest() == row["config_sha256"]
    digest = json.dumps(restart._configuration_digest_values(
        dataclasses.asdict(cfg)), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()
    assert digest == row["digest_json"].encode()


def test_every_writer_echoes_prescribed_source_identity_alike(monkeypatch, tmp_path):
    """Each memory road binds the same manifest and binary member hashes."""
    from test_rrtmg_smoke_manifest import _fixture
    from woof.core.rrtmg_smoke_identity import SMOKE_IDENTITY_KEY

    source = tmp_path / "source"
    source.mkdir()
    path, raw = _fixture(source)
    # Configuration serialization uses the existing NumPy state shim. The
    # source mode admission is covered by test_rrtmg_smoke_source_doors.
    cfg = _cfg(nz=2, rrtmg_smoke_manifest=str(path))
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert json.dumps(echo).encode() == json.dumps(expected).encode(), road
        assert echo[SMOKE_IDENTITY_KEY]["frames"][0]["value"]["sha256"] == \
            raw["frames"][0]["value"]["sha256"]


def test_member_surface_provenance_does_not_change_checkpoint_config_echo_bytes(monkeypatch, tmp_path):
    cfg = _cfg()
    receipt = {"identity": {"member_id": 7, "seed": 907,
               "options": {"kind": "surface-state", "sst_offset_k": 1.0}},
               "receipt": {"realized_fp32_hex": "0000803f0000803f"}}
    expected = json.dumps(restart.configuration_echo(cfg), allow_nan=False).encode("utf-8")
    for road, echo in _echoes(cfg, monkeypatch, tmp_path, surface_receipt=receipt).items():
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == expected, road
        assert not {"ensemble", "perturbation", "seed", "member_id"} & echo.keys()


def _with_handover(cfg, kept):
    """HN1 (2.9.7) is default on and echoed where its scheme reads it
    (Thompson and Morrison), so a moved field selected under Thompson keeps
    it as well."""
    from woof.correct_mode import key_in_scope
    key = "hydrometeor_handover_numbers"
    if getattr(cfg, key) is True and key_in_scope(key, dataclasses.asdict(cfg)):
        return set(kept) | {key}
    return set(kept)


@pytest.mark.parametrize("field", sorted(MOVED))
def test_every_writer_echoes_a_moved_field_alike(monkeypatch, tmp_path, field):
    overrides, kept = MOVED[field]
    cfg = _cfg(**overrides)
    kept = _with_handover(cfg, kept)
    echoes = _echoes(cfg, monkeypatch, tmp_path)
    expected = restart.configuration_echo(cfg)
    for road, echo in echoes.items():
        assert echo == expected, road
        assert json.dumps(echo, allow_nan=False).encode("utf-8") == \
            json.dumps(expected, allow_nan=False).encode("utf-8"), road
    assert DEFAULT_OFF & set(expected) == kept
    for key in kept:
        assert expected[key] == getattr(cfg, key), key


def test_the_digest_drops_what_the_echo_drops():
    """The configuration digest reads the echo's drops, so a default-off
    field absent from one is absent from the other."""
    for overrides, kept in [({}, set())] + list(MOVED.values()):
        cfg = _cfg(**overrides)
        digest = restart._configuration_digest_values(dataclasses.asdict(cfg))
        assert DEFAULT_OFF & set(digest) == _with_handover(cfg, kept), overrides


@pytest.mark.parametrize("field", (
    "bl_mynn_version", "bl_mynn_gsd41_unsquared_qtke",
    "bl_mynn_cloud_tendency_form", "swint_opt", "aer_opt", "alb_sol",
))
def test_inherited_default_off_fields_bind_both_restart_directions(field):
    """An old header may omit the default, never an active selector."""
    default = _cfg()
    legacy = restart.configuration_echo(default)
    assert field not in legacy
    restart._require_config_match(legacy, default, "legacy-default.npz")
    overrides, _kept = MOVED[field]
    changed = _cfg(**overrides)
    active = restart.configuration_echo(changed)
    assert active[field] == getattr(changed, field)
    restart._require_config_match(active, changed, "active-selector.npz")
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(legacy, changed, "legacy-default.npz")
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(active, default, "active-selector.npz")


def test_current_echo_resumes_with_omitted_fork_dycore_defaults():
    cfg = _cfg()
    echo = restart.configuration_echo(cfg)
    assert not {"diff_6th_form", "diff_6th_factor2", "upper_wind_limiter_form",
                "mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all"} & set(echo)
    restart._require_config_match(echo, cfg, "current.npz")


def test_unread_zero_out_knobs_do_not_refuse_continuation():
    stored = _cfg(mp_zero_out=0, mp_zero_out_thresh=2.0e-5, mp_zero_out_all=1)
    live = _cfg(mp_zero_out=0, mp_zero_out_thresh=1.0e-8, mp_zero_out_all=0)
    restart._require_config_match(dataclasses.asdict(stored), live, "off.npz")
    restart._require_config_match(restart.configuration_echo(stored), live, "off-echo.npz")


@pytest.mark.parametrize("name,value", [("mp_zero_out_thresh", 2.0e-5), ("mp_zero_out_all", 1)])
def test_active_zero_out_knobs_still_bind_continuation(name, value):
    stored = _cfg(mp_zero_out=1)
    live = _cfg(mp_zero_out=1, **{name: value})
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(restart.configuration_echo(stored), live, "on.npz")


@pytest.mark.parametrize("name", ["mp_zero_out_thresh", "mp_zero_out_all"])
def test_active_zero_out_cannot_restore_an_unrecorded_read_knob(name):
    cfg = _cfg(mp_zero_out=1)
    stored = restart.configuration_echo(cfg)
    stored.pop(name)
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart._require_config_match(stored, cfg, "incomplete-on.npz")


@pytest.mark.parametrize("field", ["diff_6th_form", "mp_zero_out", "upper_wind_limiter_form"])
def test_omitted_dycore_default_still_refuses_an_active_selector(field):
    overrides, _kept = MOVED[field]
    with pytest.raises(restart.RestartMismatchError, match=field):
        restart._require_config_match(restart.configuration_echo(_cfg()), _cfg(**overrides), "default.npz")


@pytest.mark.parametrize("overrides", [{}, {"moist": True, "mp_physics": 6}])
@pytest.mark.parametrize("theme", ["paper", "woof-light", "woof-dark", "inherited"])
def test_renderer_environment_preserves_checkpoint_config_bytes(
        monkeypatch, tmp_path, overrides, theme):
    """Renderer settings must not enter any memory road's checkpoint echo."""
    cfg = _cfg(**overrides)
    before_root, after_root = tmp_path / "before", tmp_path / "after"
    before_root.mkdir()
    after_root.mkdir()
    before = _echoes(cfg, monkeypatch, before_root)
    if theme == "inherited":
        theme_file = tmp_path / "site-theme.json"
        theme_file.write_text(json.dumps({"extends": "woof-dark", "name": "site"}),
                              encoding="utf-8")
        theme = str(theme_file)
    for key, value in {
            "RUSTWX_THEME": theme, "RUSTWX_RADAR_COLORS": "nws",
            "RUSTWX_WIND_STREAMLINES": "1"}.items():
        monkeypatch.setenv(key, value)
    after = _echoes(cfg, monkeypatch, after_root)

    def payload(echo):
        return json.dumps(echo, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    expected = payload(restart.configuration_echo(cfg))
    assert {payload(echo) for echo in before.values()} == {expected}
    assert {payload(echo) for echo in after.values()} == {expected}


@pytest.mark.parametrize("overrides", [{}, {
    "diff_6th_form": "noaa_wrf39", "diff_6th_factor2": 0.04,
    "upper_wind_limiter_form": "noaa_wrf39", "mp_zero_out": 2,
    "mp_zero_out_all": 1, "mp_zero_out_thresh": 1.0e-8,
}])
def test_every_writer_echo_resumes_its_unchanged_dycore(
        monkeypatch, tmp_path, overrides):
    """A written echo must accept the configuration that produced it."""
    cfg = _cfg(**overrides)
    for road, echo in _echoes(cfg, monkeypatch, tmp_path).items():
        restart._require_config_match(echo, cfg, road)


@pytest.mark.parametrize("key,value", [
    ("diff_6th_form", "noaa_wrf39"),
    ("diff_6th_factor2", 0.04),
    ("upper_wind_limiter_form", "noaa_wrf39"),
    ("mp_zero_out", 2),
])
def test_omitted_dycore_default_refuses_an_enabled_or_changed_value(key, value):
    cfg = _cfg()
    echo = restart.configuration_echo(cfg)
    assert key not in echo
    restart._require_config_match(echo, cfg, "checkpoint")
    with pytest.raises(ValueError, match=key):
        restart._require_config_match(
            echo, dataclasses.replace(cfg, **{key: value}), "checkpoint")


def test_omitted_staged_dycore_defaults_restore_the_same_configuration(tmp_path):
    cfg = _cfg()
    restart._require_config_match(restart.configuration_echo(cfg), cfg, tmp_path / "state.npz")


@pytest.mark.parametrize("key,value", (
    ("diff_6th_form", "noaa_wrf39"), ("diff_6th_factor2", 0.04),
    ("mp_zero_out", 2), ("mp_zero_out_thresh", 1.0e-12),
    ("mp_zero_out_all", 1), ("upper_wind_limiter_form", "noaa_wrf39"),
))
def test_missing_staged_dycore_selector_refuses_a_changed_value(tmp_path, key, value):
    cfg = _cfg()
    header = restart.configuration_echo(cfg)
    overrides = {key: value}
    if key in ("mp_zero_out_thresh", "mp_zero_out_all"):
        overrides["mp_zero_out"] = 2
    changed = dataclasses.replace(cfg, **overrides)
    with pytest.raises(restart.RestartMismatchError, match=key):
        restart._require_config_match(header, changed, tmp_path / "state.npz")


def test_unread_zero_out_parameters_do_not_block_an_off_restart(tmp_path):
    cfg = _cfg(mp_zero_out=0, mp_zero_out_thresh=1.0e-12, mp_zero_out_all=1)
    restart._require_config_match(restart.configuration_echo(cfg), cfg, tmp_path / "state.npz")


@pytest.mark.parametrize("option,default", [("grib2", False), ("verify_visuals", True)])
def test_output_choices_preserve_every_actual_writer_echo(monkeypatch, tmp_path, option, default):
    """Output options stay outside the configuration in all three headers."""
    from woof.runplan import PLAN_SCHEMA, build_plan, resolve_plan
    from test_case_data import make_case_toml

    config = make_case_toml(tmp_path)
    document = {"schema": PLAN_SCHEMA, "name": "writer-output-choice", "route": "experiment",
                "config": {"path": str(config)}, "output_root": str(tmp_path / "run")}
    ordinary = build_plan(document, source="writer-output-choice", base_dir=tmp_path, sha256="0" * 64)
    assert ordinary.run_options[option] is default
    echoes = []
    for selected in (False, True):
        plan = build_plan({**document, "run_options": {option: selected}},
                          source="writer-output-choice", base_dir=tmp_path, sha256="0" * 64)
        assert plan.run_options[option] is selected
        _, experiment, _ = resolve_plan(plan, require_inputs=False)
        folder = tmp_path / str(selected)
        folder.mkdir()
        echoes.append(_echoes(experiment.domains[0].run, monkeypatch, folder))
    before, after = echoes
    for road in ("resident", "streamed", "store"):
        assert json.dumps(before[road], allow_nan=False).encode("utf-8") == \
            json.dumps(after[road], allow_nan=False).encode("utf-8"), road
        assert option not in before[road] and option not in after[road]


def test_the_terrain_clock_default_is_echoed_and_measured_drops_out():
    """From 2.8.8 the default terrain clock is "local_face", and it binds:
    a header without the key ran "measured" (every earlier header), so the
    echo keeps dropping "measured" and an earlier checkpoint resumes only
    under "measured".  Breakage the binding prevents: a head-bound tree
    re-derives its clock as each boundary interval arrives, so a resume
    under the other clock can take a different step later in the run."""
    from dataclasses import replace
    from woof.config import TERRAIN_CLOCK_RESTART_BREAK_NOTICE

    default = _cfg()
    assert default.terrain_clock == "local_face"
    echo = restart.configuration_echo(default)
    assert echo["terrain_clock"] == "local_face"
    measured = replace(default, terrain_clock="measured")
    legacy = restart.configuration_echo(measured)
    assert "terrain_clock" not in legacy
    pinned = restart.configuration_echo(replace(default, terrain_clock="pinned"))
    assert pinned["terrain_clock"] == "pinned"
    digest = restart._configuration_digest_values(dataclasses.asdict(measured))
    assert "terrain_clock" not in digest
    assert restart._configuration_digest_values(
        dataclasses.asdict(default))["terrain_clock"] == "local_face"
    # An earlier header resumes under "measured" and under nothing else;
    # the refusal under the new default says why and what to write.
    restart._require_config_match(legacy, measured, "earlier.npz")
    with pytest.raises(restart.RestartMismatchError,
                       match="terrain_clock") as caught:
        restart._require_config_match(legacy, default, "earlier.npz")
    assert TERRAIN_CLOCK_RESTART_BREAK_NOTICE in str(caught.value)
    with pytest.raises(restart.RestartMismatchError,
                       match="terrain_clock") as caught:
        restart._require_config_match(echo, measured, "new.npz")
    assert TERRAIN_CLOCK_RESTART_BREAK_NOTICE not in str(caught.value)
    restart._require_config_match(echo, default, "new.npz")
