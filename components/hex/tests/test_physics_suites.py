"""Named physics suites, per-scheme overrides and the namelist importer.

CPU only.  Holds the suite table to its own schema, the default suite to the
run a bare door made before named suites (the byte-identity half a CPU can
prove: the same backend row, the same convection request, the same driver
vector, the same configuration), every refusal to a named breakage and a
follow-up, the namelist importer to a disposition per option, and the front
doors (config, CLI, forecast door) to one resolver.  The column-suite parts
need the engine's ``woof.core.mpas_column_suite`` importable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from woof.hex import forecast_door as door
from woof.hex import physics_suites as ps
from woof.hex.errors import ConfigurationRefusal
from woof.hex.physics_namelist import import_physics_namelist

TREE = Path(__file__).resolve().parents[1]
RRFS_NAMELIST = TREE / "verification/source-namelists/rrfs-v2-hrrrv5-namelist.atmosphere"
RRFS_NAMELIST_SHA256 = "70ad70cec7354baa75fd8a670151b915a792f11f2b2c1625d76ccbde6489f299"

#: The workflow's own placeholder values for its 12 km mesh
#: (rrfs-workflow 8c2d53d config.meshdep and config.base), as exrrfs_fcst.sh
#: renders them.
RRFS_12KM = {
    "dt": "60.0", "substeps": "2", "start_time": "2026-10-03_12:00:00",
    "run_duration": "01:00:00", "pio_num_iotasks": "1", "pio_stride": "1",
    "MESH_NAME": "conus12km", "do_DAcycling": "false", "radt": "30",
    "lsm_scheme": "sf_ruc", "nsoillevels": "9", "jedi_da": "false",
    "do_sppt": "false",
}


def rendered_rrfs_namelist(**changes: str) -> str:
    text = RRFS_NAMELIST.read_text(encoding="utf-8")
    for key, value in {**RRFS_12KM, **changes}.items():
        text = text.replace("${%s}" % key, value)
    assert "${" not in text
    return text


def _needs_suite_engine():
    return pytest.importorskip("woof.core.mpas_column_suite")


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def test_the_vendored_rrfs_namelist_is_the_pinned_workflow_file():
    import hashlib

    assert hashlib.sha256(RRFS_NAMELIST.read_bytes()).hexdigest() == RRFS_NAMELIST_SHA256


def test_the_table_loads_and_names_one_default():
    table = ps.load_table()
    assert table["schema"] == ps.SCHEMA
    defaults = [row["name"] for row in table["suites"] if row.get("default")]
    assert defaults == [ps.DEFAULT_SUITE] == ["arwen_mpas_column"]
    assert {"arwen_mpas_column", "arwen_thompson_column", "rrfs_v2_dev"} <= set(ps.suite_names())


def test_every_refused_option_names_its_breakage_and_a_follow_up():
    for key, rows in ps.load_table()["options"].items():
        for value, row in rows.items():
            if "refused" in row:
                assert len(row["refused"]) > 40, (key, value)
                assert row["follow_up"].strip(), (key, value)


def test_every_value_mpas_can_resolve_has_a_row():
    table = ps.load_table()
    for name, selectors in table["mpas_suites"]["suites"].items():
        for key, value in selectors.items():
            assert value in table["options"][key], (name, key, value)


def _table_copy():
    return json.loads(json.dumps(ps._thaw(ps.load_table())))


@pytest.mark.parametrize("damage,match", [
    (lambda t: t["suites"].append(dict(t["suites"][0], name="arwen_mpas_column")), "declared twice"),
    (lambda t: next(r for r in t["suites"] if r["name"] == "arwen_thompson_column").__setitem__("default", True), "exactly one suite"),
    (lambda t: t["suites"][0]["selectors"].pop("config_pbl_scheme"), "every selector"),
    (lambda t: t["suites"][0]["selectors"].__setitem__("config_pbl_scheme", "bl_nope"), "no row"),
    (lambda t: t["options"]["config_radt_lw_scheme"]["cam_lw"].pop("follow_up"), "names no follow-up"),
    (lambda t: t["options"]["config_lsm_scheme"]["sf_noah"].pop("engine"), "an alias onto the engine's selectors"),
    (lambda t: next(r for r in t["suites"] if r["name"] == "rrfs_v2_dev").__setitem__("differences_from_shaped_after", []), "lists no"),
    (lambda t: t["suites"].append(dict(t["suites"][0], name="hrrrv5", default=False)), "collide"),
    (lambda t: t["mpas_suites"]["suites"]["hrrrv5"].__setitem__("config_microp_scheme", "mp_new"), "has no row"),
])
def test_a_malformed_table_refuses_by_name(damage, match):
    table = _table_copy()
    damage(table)
    with pytest.raises(ps.PhysicsTableError, match=match):
        ps.validate_table(table)


def test_the_hex_generation_table_is_the_engines():
    suite = _needs_suite_engine()
    engine = getattr(suite, "GENERATION_SELECTORS", None)
    if engine is None:
        pytest.skip("engine column-suite seam predates generation selectors")
    slot_names = {"microphysics": "microphysics_scheme"}
    for key, row in ps.load_table()["generation_selectors"].items():
        assert key in engine, key
        assert tuple(row["values"]) == tuple(engine[key]["values"]), key
        assert slot_names.get(row["slot"], row["slot"]) == engine[key]["requires"][0], key
        assert tuple(row["schemes"]) == tuple(engine[key]["requires"][1]), key


def test_every_carried_column_slot_is_an_engine_row():
    suite = _needs_suite_engine()
    tables = {
        "pbl_scheme": suite.PBL_SCHEMES, "surface_layer_scheme": suite.SURFACE_LAYER_SCHEMES,
        "land_surface_scheme": suite.LAND_SURFACE_SCHEMES, "radiation_scheme": suite.RADIATION_SCHEMES,
    }
    for rows in ps.load_table()["options"].values():
        for row in rows.values():
            for slot, value in row.get("carried", {}).items():
                if slot in tables:
                    assert value in tables[slot], (slot, value)
    for drag in ps.load_table()["column_suite_seam"]["drag"].values():
        assert drag in suite.DRAG_SCHEMES
    ruc = ps.load_table()["options"]["config_lsm_scheme"]["sf_ruc"]["carried"]
    assert suite.LAND_SURFACE_SCHEMES["ruc"]["num_soil_layers"] == ruc["n_soil_levels"]


# ---------------------------------------------------------------------------
# the default does not move
# ---------------------------------------------------------------------------
def test_the_default_suite_is_the_frozen_row_with_the_auto_request():
    for resolved in (ps.resolve_physics(), ps.resolve_physics("arwen_mpas_column")):
        assert resolved.is_default()
        assert (resolved.seam, resolved.backend_row, resolved.convection) == (
            "frozen", "wsm6_column", "auto")
        assert resolved.column_profile is None and not resolved.generation


def test_the_default_suite_selectors_are_the_frozen_configuration_literals():
    from woof.hex.config_v841 import V841MpasColumnPhysicsConfig

    config = V841MpasColumnPhysicsConfig()
    resolved = ps.resolve_physics()
    assert config.config_physics_suite == ps.DEFAULT_SUITE
    for key, value in resolved.selectors.items():
        if key == "config_convection_scheme":
            continue  # decided by the mesh bind (auto), not a literal
        if key == "config_gwdo_scheme":
            # the door's lane is the YSU-GWDO subtype
            from woof.hex.config_v841 import V841MpasColumnPhysicsGwdoConfig
            assert V841MpasColumnPhysicsGwdoConfig().config_gwdo_scheme == value
            continue
        assert getattr(config, key) == value, key


# ---------------------------------------------------------------------------
# suites and overrides
# ---------------------------------------------------------------------------
def test_aerosol_thompson_suite_runs_on_the_frozen_thompson_row():
    resolved = ps.resolve_physics("arwen_thompson_column")
    assert (resolved.seam, resolved.backend_row, resolved.convection) == (
        "frozen", "thompson_aero_column", "auto")
    assert not resolved.is_default()


def test_rrfs_v2_dev_at_3_km_is_the_hrrr_family_column_suite():
    _needs_suite_engine()
    resolved = ps.resolve_physics("rrfs_v2_dev", mesh="conus3km")
    assert resolved.seam == "column_suite"
    assert resolved.backend_row == "mynn_ruc_gsl_column"
    assert resolved.convection == "off"
    assert dict(resolved.column_profile["selection"]) == {
        "radiation_scheme": "rrtmg_legacy", "surface_layer_scheme": "mynn",
        "land_surface_scheme": "ruc", "pbl_scheme": "mynn", "drag_scheme": "gsl"}
    assert resolved.selectors["config_pbl_scheme"] == "bl_mynnedmf"
    assert resolved.frac_seaice is True
    assert any("TEMPO" in line for line in resolved.differences)


def test_rrfs_v2_dev_on_a_12_km_mesh_selects_the_carried_new_tiedtke_slot():
    _needs_suite_engine()
    carried = ps.resolve_physics("rrfs_v2_dev", mesh="conus12km")
    assert carried.convection == "ntiedtke"
    assert carried.selectors["config_convection_scheme"] == "cu_ntiedtke"
    resolved = ps.resolve_physics(
        "rrfs_v2_dev", {"config_convection_scheme": "off"}, mesh="conus12km")
    assert resolved.origins["config_convection_scheme"] == "override"
    assert dict(resolved.overrides) == {"config_convection_scheme": "off"}


def test_an_override_off_the_profiles_synthesizes_a_registered_row():
    _needs_suite_engine()
    from woof.hex import cuda_suite_physics_v841 as adapter
    from woof.hex.physics_backend_admission import resolve_backend

    resolved = ps.resolve_physics(
        "rrfs_v2_dev", {"config_lsm_scheme": "sf_noahmp", "config_frac_seaice": False})
    profile = resolved.column_profile
    assert profile["synthesized"] is True
    assert profile["n_soil_levels"] == 4
    import numpy as np

    # init_atmosphere's own float32 zs for Noah-MP's four layers (the third is
    # a running sum that lands on 0.70000005, measured on NOAA's conus12km init)
    assert np.array_equal(np.asarray(profile["soil_level_depths_m"], np.float32),
                          np.array([0.05, 0.25, 0.70000005, 1.5], np.float32))
    assert np.float32(profile["soil_level_depths_m"][2]) != np.float32(0.7)
    assert adapter.physics_suite(profile["id"])["backend_row"] == resolved.backend_row
    row = resolve_backend(resolved.backend_row)
    assert row.adapter_module == "woof.hex.cuda_suite_physics_v841"
    assert row.anchored is False
    assert row.driven_scalar_names[-3:] == ("ni", "nr", "nc")
    assert row.appended_scalar_names == ("nwfa", "nifa")
    # the same request resolves to the same row, every time
    again = ps.resolve_physics(
        "rrfs_v2_dev", {"config_lsm_scheme": "sf_noahmp", "config_frac_seaice": False})
    assert again.backend_row == resolved.backend_row


def test_generation_selectors_ride_the_column_suite_and_are_refused_on_the_frozen_batch():
    _needs_suite_engine()
    resolved = ps.resolve_physics(
        "rrfs_v2_dev", {"bl_mynn_version": "gsd_41", "mynn_sfclay_variant": "gsl_wrf39"})
    assert dict(resolved.generation) == {"bl_mynn_version": "gsd_41",
                                         "mynn_sfclay_variant": "gsl_wrf39"}
    assert dict(resolved.column_profile["generation"]) == dict(resolved.generation)
    # A generation of a scheme the run does not select is refused on every
    # seam: it would be recorded and never read.
    with pytest.raises(ps.PhysicsSelectionRefusal, match="never read"):
        ps.resolve_physics("arwen_mpas_column", {"bl_mynn_version": "gsd_41"})
    with pytest.raises(ps.PhysicsSelectionRefusal, match="never read"):
        ps.resolve_physics("rrfs_v2_dev", {"thompson_version": "wrf_39_noaa",
                                           "config_microp_scheme": "mp_wsm6"})
    with pytest.raises(ps.PhysicsSelectionRefusal, match="admits only"):
        ps.resolve_physics("rrfs_v2_dev", {"bl_mynn_version": "gsd_99"})


@pytest.mark.parametrize("overrides,knob", [
    # What is left to refuse once "no seam runs it" is gone (2026-10-11):
    # a name WOOF's registry does not carry, a combination that would run
    # under another name, and one decision given two values.  The options
    # this list used to hold (TEMPO on the default suite, Noah, no drag,
    # the GSL drag, Kain-Fritsch) now resolve: tests/test_arbitrary_physics.py.
    ({"config_pbl_scheme": "bl_mynn"}, "config_radt_cld_scheme"),
    ({"config_radt_lw_scheme": "rrtmgp_lw"}, "config_radt_sw_scheme"),
    ({"config_radt_lw_scheme": "cam_lw"}, "config_radt_lw_scheme"),
    ({"config_radt_cld_scheme": "cld_fraction_thompson"}, "config_radt_cld_scheme"),
    ({"config_convection_scheme": "cu_gf_monan"}, "config_convection_scheme"),
    ({"config_frac_seaice": True}, "config_frac_seaice"),
    ({"config_pbl_scheme": "bl_not_a_scheme"}, "config_pbl_scheme"),
])
def test_an_unsupported_override_is_refused_by_name(overrides, knob):
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics("arwen_mpas_column", overrides)
    assert knob in [item.knob for item in caught.value.refusals]
    assert all(item.follow_up for item in caught.value.refusals)


def test_the_ysu_drag_beside_mynn_runs_in_the_engine_not_as_two_operators():
    """bl_ysu_gwdo on a selectable boundary layer was refused because hex's
    own YSU-GWDO kernel and the engine's PBL-slot drag would both have acted
    on one momentum tendency.  It now runs as ONE operator: the engine's own
    YSU drag (WOOF gwd_opt=1) on the generic column seam, reading the ten
    statistics the mesh's static file carries; the hex-side kernel is not
    composed."""

    suite = _needs_suite_engine()
    if not hasattr(suite, "column_physics_plan"):
        pytest.skip("the importable engine predates the generic column seam")
    resolved = ps.resolve_physics("rrfs_v2_dev", {"config_gwdo_scheme": "bl_ysu_gwdo"})
    assert resolved.seam == "engine"
    assert resolved.engine_physics["gwd_opt"] == 1
    assert resolved.column_profile["selection"]["drag_scheme"] == "ysu_gwdo"
    assert resolved.column_profile["orographic_statistics"]["file"] == "static"


def test_every_refusal_of_one_request_is_named_together():
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics("arwen_mpas_column", {
            "config_radt_lw_scheme": "cam_lw",
            "config_radt_cld_scheme": "cld_fraction_thompson"})
    knobs = {item.knob for item in caught.value.refusals}
    assert {"config_radt_lw_scheme", "config_radt_cld_scheme"} <= knobs
    assert "2 physics option(s) refused" in str(caught.value)


def test_an_unknown_suite_names_the_ones_that_exist():
    with pytest.raises(ps.PhysicsSelectionRefusal, match="rrfs_v2_dev"):
        ps.resolve_physics("rrfs_v3")


def test_a_suite_override_of_suite_keeps_the_suite_value():
    resolved = ps.resolve_physics("arwen_mpas_column", {"config_pbl_scheme": "suite"})
    assert resolved.selectors["config_pbl_scheme"] == "bl_ysu"
    assert resolved.backend_row == "wsm6_column"


@pytest.mark.parametrize("text,expected", [
    ("config_pbl_scheme=bl_mynn", ("config_pbl_scheme", "bl_mynn")),
    ("bl_mynn_mixlength=2", ("bl_mynn_mixlength", 2)),
    ("config_frac_seaice=.true.", ("config_frac_seaice", True)),
    ("config_lsm_scheme='sf_ruc'", ("config_lsm_scheme", "sf_ruc")),
])
def test_cli_overrides_are_typed_by_their_rows(text, expected):
    assert ps.parse_override(text) == expected


# ---------------------------------------------------------------------------
# the configuration and the suite adapter
# ---------------------------------------------------------------------------
def test_the_configuration_records_any_table_suite_and_refuses_another():
    from dataclasses import replace

    from woof.hex.config_v841 import V841MpasColumnPhysicsConfig

    base = V841MpasColumnPhysicsConfig()
    for name in ps.recordable_suite_names():
        replace(base, config_physics_suite=name).validate()
    with pytest.raises(ConfigurationRefusal, match="suite table"):
        replace(base, config_physics_suite="bogus").validate()
    # the selectors stay exact on the frozen type
    with pytest.raises(ConfigurationRefusal, match="config_pbl_scheme"):
        replace(base, config_pbl_scheme="bl_mynn").validate()


def test_the_suite_configuration_records_the_resolved_spellings():
    _needs_suite_engine()
    from dataclasses import replace

    from woof.hex import cuda_suite_physics_v841 as adapter
    from woof.hex.config_v841 import V841MpasColumnPhysicsConfig

    resolved = ps.resolve_physics("rrfs_v2_dev", mesh="conus3km")
    selectors = {k: v for k, v in resolved.selectors.items() if k != "config_convection_scheme"}
    config = V841MpasColumnPhysicsConfig()
    recorded = adapter.suite_config(config, resolved.backend_row, mpas_selectors=selectors)
    recorded = replace(recorded, config_physics_suite="rrfs_v2_dev")
    recorded.validate()
    assert recorded.config_pbl_scheme == "bl_mynnedmf"
    assert recorded.config_sfclayer_scheme == "sf_mynnsfclay"
    assert recorded.config_radt_cld_scheme == "cld_fraction_mynn"
    assert recorded.config_gwdo_scheme == "bl_ugwp_gwdo"
    assert recorded.config_physics_suite == "rrfs_v2_dev"


def test_the_mynn_profile_records_the_cloud_scheme_it_executes():
    from woof.hex import cuda_suite_physics_v841 as adapter

    row = adapter.load_physics_suites()["mynn-ruc-gsl"]
    assert row["mpas_selectors"]["config_radt_cld_scheme"] == "cld_fraction_mynn"


def test_a_generation_reaches_the_sealed_constructor_only_when_selected():
    _needs_suite_engine()
    from test_physics_suite_row import _frozen_values, _oro  # noqa: PLC0415

    from woof.hex import cuda_suite_physics_v841 as adapter

    plain = adapter.constructor_values_for(
        _frozen_values(), row_name="mynn_ruc_gsl_column", orographic_statistics=_oro())
    assert "generation" not in plain
    sealed = adapter.seal_constructor(plain)
    assert "generation" not in sealed.arwen_kwargs()
    resolved = ps.resolve_physics("rrfs_v2_dev", {"bl_mynn_version": "gsd_41"})
    values = adapter.constructor_values_for(
        _frozen_values(), row_name=resolved.backend_row, orographic_statistics=_oro())
    assert values["generation"] == {"bl_mynn_version": "gsd_41"}
    sealed = adapter.seal_constructor(values)
    assert sealed.arwen_kwargs()["generation"] == {"bl_mynn_version": "gsd_41"}
    assert sealed.receipt()["generation"] == {"bl_mynn_version": "gsd_41"}
    assert sealed.identity_sha256 != adapter.seal_constructor(plain).identity_sha256


# ---------------------------------------------------------------------------
# the namelist importer
# ---------------------------------------------------------------------------
def test_rrfs_v2s_own_namelist_imports_with_no_unsupported_option():
    # config_mynn_mixaerosols = 1 was the last refusal: MYNN's nwfa/nifa
    # rates now leave the column seam (scalar_rates) and hex couples them.
    _needs_suite_engine()
    result = import_physics_namelist(rendered_rrfs_namelist(), mesh="conus3km")
    assert result.admitted, result.report()["refusals"]
    assert list(result.refusals) == []
    assert result.require_admitted().generation["bl_mynn_mixaerosols"] == 1
    statuses = {(d.option, d.status) for d in result.dispositions}
    assert ("config_mynn_mixaerosols", "admitted") in statuses
    assert ("config_tempo_aerosolaware", "admitted") in statuses
    assert ("config_pbl_scheme", "admitted") in statuses
    assert ("num_soil_layers", "admitted") in statuses
    assert ("config_iseed_sppt1", "ignored") in statuses
    assert ("config_pio_num_iotasks", "outside") in statuses
    assert ("config_start_time", "outside") in statuses
    assert ("config_mpas_cam_coef", "dycore") in statuses
    assert ("config_IAU_window_length_s", "ignored") in statuses
    assert result.radiation_seconds == 1800.0
    assert result.dycore_config is not None
    assert result.dycore_config.config_number_cam_damping_levels == 8
    # no option is dropped without a disposition
    from woof.hex.config import parse_fortran_namelist

    parsed = parse_fortran_namelist(rendered_rrfs_namelist())
    every = {option for group in parsed.values() for option in group}
    assert every <= {d.option for d in result.dispositions}


def test_the_rrfs_namelist_with_the_two_options_lifted_runs_the_hrrr_family_row():
    _needs_suite_engine()
    text = rendered_rrfs_namelist().replace(
        "config_physics_suite = 'hrrrv5'",
        "config_physics_suite = 'hrrrv5'\n    config_microp_scheme = 'mp_thompson_aerosols'",
    ).replace("config_mynn_mixaerosols = 1", "config_mynn_mixaerosols = 0")
    result = import_physics_namelist(text, mesh="conus3km")
    assert result.admitted, result.report()["refusals"]
    physics = result.require_admitted()
    assert physics.backend_row == "mynn_ruc_gsl_column"
    assert physics.suite == "hrrrv5" and physics.suite_kind == "mpas"
    statuses = {(d.option, d.status) for d in result.dispositions}
    assert ("config_tempo_aerosolaware", "ignored") in statuses


@pytest.mark.parametrize("edit,knob,match", [
    (lambda t: t.replace("num_soil_layers      = 9", "num_soil_layers      = 4"),
     "num_soil_layers", "integrates 9 soil"),
    (lambda t: t.replace("config_radtsw_interval = '00:30:00'", "config_radtsw_interval = '00:15:00'"),
     "config_radtlw_interval", "different intervals"),
    (lambda t: t.replace("config_mynn_mixnumcon = 0", "config_mynn_mixnumcon = 0\n    config_mynn_new_knob = 1"),
     "config_mynn_new_knob", "no row"),
])
def test_namelist_refusals_are_per_option(edit, knob, match):
    _needs_suite_engine()
    result = import_physics_namelist(edit(rendered_rrfs_namelist()), mesh="conus3km")
    hits = [item for item in result.refusals if item.knob == knob]
    assert hits and match in hits[0].reason, result.report()["refusals"]


def test_stochastic_switches_now_have_a_gpu_consumer():
    _needs_suite_engine()
    text = rendered_rrfs_namelist().replace('do_sppt = false','do_sppt = true')
    result = import_physics_namelist(text, mesh='conus3km')
    assert not [item for item in result.refusals if item.knob in ('do_sppt','do_skeb','config_stochini')]
    assert result.stochastic['sppt']
    result = import_physics_namelist(text, mesh='conus3km', overrides={'do_sppt':False})
    assert not result.stochastic['sppt']


def test_an_unrendered_workflow_template_is_refused_as_syntax():
    with pytest.raises(ConfigurationRefusal, match="namelist_syntax"):
        import_physics_namelist(RRFS_NAMELIST.read_text(encoding="utf-8"))


def test_a_physics_only_namelist_resolves_a_hex_suite():
    text = "&physics\n    config_physics_suite = 'arwen_thompson_column'\n/\n"
    result = import_physics_namelist(text)
    assert result.admitted and result.physics.backend_row == "thompson_aero_column"


@pytest.mark.parametrize("key", ("config_sst_update", "config_sstdiurn_update",
                               "config_gvf_update", "config_deepsoiltemp_update"))
def test_surface_update_switches_are_admitted_and_recorded(key):
    result = import_physics_namelist(
        "&physics config_physics_suite='arwen_mpas_column', "+key+"=.true. /\n")
    assert result.admitted
    assert result.physics.overrides[key] is True
    assert ps.parse_override(key+"=.true.") == (key, True)
    assert "follow_up" not in ps.load_table()["namelist"]["options"][key]


def test_door_namelist_switches_retain_explicit_scalar_stochastic_controls(tmp_path, door_seams):
    namelist = tmp_path / 'namelist.atmosphere'
    namelist.write_text("&physics config_physics_suite='arwen_thompson_column' /\n"
                        "&nam_stochy do_sppt=.true., do_skeb=.true. /\n")
    controls = dict(seed=711, amplitude=.4, length_m=1500000., time_s=600.)
    request = door.resolve_request(_namespace(tmp_path, 'x1.40962',
        '--physics-namelist', str(namelist), '--stochastic', json.dumps(controls)),
        registry=_registry())
    config = json.loads(request.stochastic)
    assert config['sppt'] and config['skeb'] and config['sppt_scales'] == []
    assert all(config[key] == value for key, value in controls.items())


# ---------------------------------------------------------------------------
# the doors
# ---------------------------------------------------------------------------
@pytest.fixture
def door_seams(monkeypatch):
    from woof.hex.device_admission import REFERENCE_CARD

    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))
    monkeypatch.setattr(door, "read_nvrtc_targets", lambda: (75, 80, 86, 89, 90, 100, 120, 121))
    monkeypatch.setattr(door, "read_card_profile", lambda: REFERENCE_CARD)
    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None)
    monkeypatch.setattr(door, "backend_pin_problem", lambda row, checkout: None, raising=False)
    # These request/argv checks do not decode or run table data. The dedicated
    # missing-table test below replaces this seam with a concrete refusal.
    import woof.table_assets
    monkeypatch.setattr(woof.table_assets, "require_thompson_tables", lambda: None)


def _registry():
    return {
        "x1.40962": door.MeshRow("x1.40962", 40_962, 120.0),
        "conus12km": door.MeshRow("conus12km", 157_859, 120.0),
    }


def _namespace(tmp_path: Path, mesh: str = "x1.40962", *extra: str) -> argparse.Namespace:
    grid = tmp_path / f"{mesh}.grid.nc"
    static = tmp_path / f"{mesh}.static.nc"
    init = tmp_path / "init.nc"
    checkout = tmp_path / "woof"
    for path in (grid, static, init):
        path.write_bytes(b"not really netcdf")
    checkout.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args([
        "--mesh", mesh, "--grid", str(grid), "--static", str(static),
        "--init", str(init), "--init-source", "GFS", "--hours", "1.0",
        "--history-every-minutes", "30", "--out", str(tmp_path / "out"),
        "--gpuwm-checkout", str(checkout), *extra,
    ])


def test_a_bare_door_request_is_the_request_before_named_suites(tmp_path, door_seams):
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    assert request.physics_suite is None and request.physics_overrides == ()
    assert request.physics_backend == "wsm6_column" and request.convection == "auto"
    # MPAS-A's config_radtlw/sw_interval default, 00:30:00 (woof.hex.mpas_defaults).
    assert request.radiation_seconds == 1800.0 and request.orographic_statistics is None
    argv = door.build_driver_argv(request)
    for flag in ("--physics-suite", "--physics", "--physics-mesh",
                 "--orographic-statistics", "--physics-backend"):
        assert flag not in argv
    # The schedule always reaches the driver explicitly.
    assert argv[argv.index("--radiation-seconds") + 1] == "1800.0"


def test_the_driver_re_resolves_what_the_door_passed(tmp_path, door_seams):
    from woof.hex.drivers.run_cuda_v841_forecast import parse_args, resolve_driver_physics

    request = door.resolve_request(
        _namespace(tmp_path, "x1.40962", "--physics-suite", "arwen_thompson_column"),
        registry=_registry())
    assert request.physics_backend == "thompson_aero_column"
    argv = door.build_driver_argv(request)
    parsed = parse_args(argv)
    physics = resolve_driver_physics(parsed)
    assert physics.suite == "arwen_thompson_column"
    assert physics.backend_row == parsed.physics_backend == "thompson_aero_column"
    stripped = parse_args([a for a in argv if a not in ("--physics-backend", "thompson_aero_column")])
    assert stripped.physics_backend == "wsm6_column"
    with pytest.raises(ConfigurationRefusal, match="one decision, two sources"):
        resolve_driver_physics(stripped)
    legacy = argparse.Namespace(physics_suite=None, physics=None, physics_mesh=None,
                                physics_backend="wsm6_column")
    assert resolve_driver_physics(legacy) is None


def test_the_door_finds_the_drag_statistics_beside_the_grid(tmp_path, door_seams, monkeypatch):
    _needs_suite_engine()
    arguments = _namespace(tmp_path, "conus12km", "--physics-suite", "rrfs_v2_dev",
                           "--convection", "off")
    # The suite door now checks the grid's cells, so its fixture must be a real netCDF grid.
    from netCDF4 import Dataset
    with Dataset(arguments.grid, "w") as ds:
        ds.createDimension("nCells", 2)
        for name, values in (("latCell", [0.5, 0.6]), ("lonCell", [0.1, 0.2]), ("areaCell", [1e8, 1e8])):
            ds.createVariable(name, "f8", ("nCells",))[:] = values
        ds.setncattr("sphere_radius", 6371220.0)
    # Missing geography is explicit, independent of the host's installed geography.
    from woof.hex import mesh_point
    def no_geography(explicit):
        raise mesh_point.PointPlanRefusal("no geography staged in this fixture")
    monkeypatch.setattr(mesh_point, "resolve_geog", no_geography)
    # Missing, the door builds it from the 30" terrain (woof.hex.ugwp_oro); with
    # no geography staged that build is refused by name.
    with pytest.raises(door.ForecastDoorRefusal, match="orographic statistics for conus12km could not be built"):
        door.resolve_request(arguments, registry=_registry())
    (tmp_path / "conus12km.ugwp_oro_data.nc").write_bytes(b"x")
    request = door.resolve_request(arguments, registry=_registry())
    assert request.physics_backend == "mynn_ruc_gsl_column"
    assert request.convection == "off"
    assert request.physics_overrides == (("config_convection_scheme", "off"),)
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--orographic-statistics") + 1].endswith("conus12km.ugwp_oro_data.nc")
    assert argv[argv.index("--physics-mesh") + 1] == "conus12km"


def test_the_door_refuses_two_sources_for_one_decision(tmp_path, door_seams):
    with pytest.raises(door.ForecastDoorRefusal, match="two backends"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-suite", "arwen_thompson_column",
            "--physics-backend", "mynn_ruc_gsl_column"), registry=_registry())
    with pytest.raises(door.ForecastDoorRefusal, match="two cumulus"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-suite", "arwen_mpas_column",
            "--convection", "gf", "--physics", "config_convection_scheme=off"),
            registry=_registry())


def test_the_door_takes_a_namelist_and_its_radiation_interval(tmp_path, door_seams):
    _needs_suite_engine()
    namelist = tmp_path / "namelist.atmosphere"
    # The registry row's own timestep, so config_dt agrees with the run; and
    # no lateral boundaries, because this test's row is a global stand-in
    # with no --lbc-dir (the door holds config_apply_lbcs to the run's).
    namelist.write_text(
        rendered_rrfs_namelist(dt="120.0").replace(
            "config_apply_lbcs = true", "config_apply_lbcs = false"),
        encoding="utf-8")
    (tmp_path / "conus12km.ugwp_oro_data.nc").write_bytes(b"x")
    # The bare RRFS namelist no longer refuses on its aerosol mixing; what
    # stops it here, if anything, is this machine's staged TEMPO tables.
    try:
        door.resolve_request(_namespace(tmp_path, "conus12km", "--physics-namelist", str(namelist)),
                             registry=_registry())
    except door.ForecastDoorRefusal as refusal:
        assert "config_mynn_mixaerosols" not in str(refusal)
    request = door.resolve_request(_namespace(
        tmp_path, "conus12km", "--physics-namelist", str(namelist),
        "--physics", "config_microp_scheme=mp_thompson_aerosols",
        "--physics", "config_mynn_mixaerosols=0",
        "--physics", "config_convection_scheme=off", *RRFS_12KM_DYCORE_FLAGS),
        registry=_registry())
    assert request.radiation_seconds == 1800.0
    assert request.physics_suite == "hrrrv5"
    assert request.physics_backend == "mynn_ruc_gsl_column"


#: The door flags that carry RRFS v2's 12 km &nhyd_model and &damping
#: (rrfs-workflow 8c2d53d namelist + config.meshdep), as the GPU smoke ran.
RRFS_12KM_DYCORE_FLAGS = (
    "--time-integration-order", "2", "--acoustic-substeps", "2", "--monotonic",
    "--cam-coefficient", "2.0", "--cam-damping-levels", "8", "--rayleigh-u",
    "--damping-start-m", "16000",
)

_DYNAMICS_NAMELIST = """&nhyd_model
    config_time_integration_order = 2
    config_dt = 120.0
    config_number_of_sub_steps = 2
    config_horiz_mixing = '2d_smagorinsky'
    config_monotonic = true
    config_epssm = 0.3
/
&damping
    config_mpas_cam_coef = 2.0
    config_number_cam_damping_levels = 8
    config_rayleigh_damp_u = true
    config_zd = 16000.0
/
&physics
    config_physics_suite = 'arwen_mpas_column'
/
"""


def test_the_namelists_dynamics_are_the_runs_or_the_door_refuses(tmp_path, door_seams):
    """--physics-namelist once reported &nhyd_model/&damping 'admitted' and ran
    the door's own dynamics.  Since 2026-10-10 the namelist's config_dt,
    order, sub-steps and limiter are taken when their flags are absent; every
    other option the run would not execute is named, with its flag, in one
    refusal; config_epssm is read and ignored, as MPAS-A v8.4.1 ignores it."""

    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text(_DYNAMICS_NAMELIST, encoding="utf-8")
    imported = import_physics_namelist(_DYNAMICS_NAMELIST)
    statuses = {d.option: d.status for d in imported.dispositions}
    assert statuses["config_epssm"] == "ignored"
    assert statuses["config_monotonic"] == "dycore"
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-namelist", str(namelist)), registry=_registry())
    text = str(caught.value)
    for option, flag in (("config_mpas_cam_coef", "--cam-coefficient"),
                         ("config_rayleigh_damp_u", "--rayleigh-u"),
                         ("config_zd", "--damping-start-m")):
        assert option in text and flag in text, (option, text)
    for adopted in ("config_time_integration_order", "config_number_of_sub_steps",
                    "config_monotonic", "config_dt", "config_epssm"):
        assert adopted not in text, (adopted, text)
    preflight = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-namelist", str(namelist), "--preflight"),
        registry=_registry())
    assert sum("--physics-namelist" in item for item in preflight.input_problems) == 4
    request = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-namelist", str(namelist), *RRFS_12KM_DYCORE_FLAGS),
        registry=_registry())
    assert request.physics_suite == "arwen_mpas_column"
    assert (request.time_integration_order, request.acoustic_substeps, request.monotonic) == (2, 2, True)
    assert request.dt_seconds == 120.0 and request.dt_selection["source"] == "namelist"
    assert not request.input_problems
    other_dt = tmp_path / "other-dt.atmosphere"
    other_dt.write_text(_DYNAMICS_NAMELIST.replace("config_dt = 120.0", "config_dt = 60.0")
                        .replace("config_time_integration_order = 2",
                                 "config_time_integration_order = 3"),
                        encoding="utf-8")
    adopted = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-namelist", str(other_dt),
        "--cam-coefficient", "2.0", "--cam-damping-levels", "8", "--rayleigh-u",
        "--damping-start-m", "16000"), registry=_registry())
    assert adopted.dt_seconds == 60.0 and adopted.time_integration_order == 3
    assert adopted.schedule_sources["order"] == "namelist"
    with pytest.raises(door.ForecastDoorRefusal, match="config_dt.*--dt"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-namelist", str(other_dt), "--dt", "90",
            *RRFS_12KM_DYCORE_FLAGS), registry=_registry())


def test_a_namelists_unflagged_dynamics_reach_the_driver(tmp_path, door_seams):
    """MEASURED 2026-10-10 (audit of hex/retire-cert-gates): a namelist with
    config_apvm_upwinding, config_smdiv and config_coef_3rd_order was refused
    three times with "the run has no flag for it" after the certificate
    gates over those options were retired -- free in the configuration type,
    unreachable from the door.  They now travel as --dynamics KEY=VALUE."""

    text = _DYNAMICS_NAMELIST.replace(
        "    config_epssm = 0.3\n",
        "    config_epssm = 0.3\n    config_apvm_upwinding = 0.25\n"
        "    config_smdiv = 0.2\n    config_coef_3rd_order = 1.0\n")
    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text(text, encoding="utf-8")
    request = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-namelist", str(namelist), *RRFS_12KM_DYCORE_FLAGS),
        registry=_registry())
    assert not request.input_problems
    wanted = {"config_apvm_upwinding": 0.25, "config_smdiv": 0.2, "config_coef_3rd_order": 1.0}
    held = dict(request.dynamics_overrides)
    assert {key: held[key] for key in wanted} == wanted
    assert all(request.dynamics_override_sources[key] == "namelist" for key in wanted)
    argv = door.build_driver_argv(request)
    pairs = [argv[i + 1] for i, item in enumerate(argv) if item == "--dynamics"]
    assert "config_apvm_upwinding=0.25" in pairs and "config_smdiv=0.2" in pairs

    # The driver builds the configuration the door described.
    import importlib
    driver = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    config = driver.build_forecast_config(
        dt_seconds=120.0, cam_coefficient=2.0, cam_damping_levels=8, rayleigh_u=True,
        damping_start_m=16000.0,
        dynamics_overrides=driver.parse_dynamics_overrides(pairs))
    assert (config.config_apvm_upwinding, config.config_smdiv, config.config_coef_3rd_order) == (0.25, 0.2, 1.0)

    # One option, two values: refused, as --dt against a namelist's config_dt
    # is.  The same value both ways is one decision.
    with pytest.raises(door.ForecastDoorRefusal, match="config_smdiv.*--dynamics"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-namelist", str(namelist), *RRFS_12KM_DYCORE_FLAGS,
            "--dynamics", "config_smdiv=0.15"), registry=_registry())
    agreed = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-namelist", str(namelist), *RRFS_12KM_DYCORE_FLAGS,
        "--dynamics", "config_smdiv=0.2"), registry=_registry())
    assert dict(agreed.dynamics_overrides)["config_smdiv"] == 0.2
    assert agreed.dynamics_override_sources["config_smdiv"] == "explicit"
    # --dynamics is reachable without a namelist.
    bare = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--dynamics", "config_positive_definite=true"),
        registry=_registry())
    assert dict(bare.dynamics_overrides) == {"config_positive_definite": True}

    # An option the run does not carry, or one another flag decides, is
    # refused by what would go wrong.
    with pytest.raises(door.ForecastDoorRefusal, match="would be recorded and not run"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--dynamics", "config_no_such_option=1"), registry=_registry())
    with pytest.raises(door.ForecastDoorRefusal, match="--acoustic-substeps"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--dynamics", "config_number_of_sub_steps=4"),
            registry=_registry())


def test_the_physics_door_lists_resolves_and_imports(tmp_path, capsys):
    from woof.hex.cli import main

    assert main(["physics", "resolve", "--physics-suite", "arwen_thompson_column"]) == 0
    assert json.loads(capsys.readouterr().out)["backend_row"] == "thompson_aero_column"
    # A name in a suite_routes table runs the forecast door's MPAS suite route (forecast_v842.native_route),
    # and these verbs report that route.  convection_permitting (bl_mynn with cld_fraction) was refused
    # here on the engine-MYNN pairing while the door ran it on driver_v842.
    assert main(["physics", "resolve", "--physics-suite", "hrrrv5"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert (resolved["route"], resolved["backend_row"]) == ("mpas_suite", "mpasphys_rrfs")
    assert resolved["selectors"]["microphysics"] == "mp_tempo"
    assert main(["physics", "resolve", "--physics-suite", "convection_permitting",
                 "--physics-mode", "exact"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert (resolved["backend_row"], resolved["default_backend"], resolved["physics_mode"]) == (
        "mpasphys_v842", "hex-cuda", "exact")
    assert resolved["selectors"]["cloudiness"] == "cld_fraction"
    assert main(["physics", "list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in listing["suites"]} >= set(ps.suite_names())
    namelist = tmp_path / "namelist.atmosphere"
    namelist.write_text(rendered_rrfs_namelist(), encoding="utf-8")
    pytest.importorskip("woof.core.mpas_column_suite")
    assert main(["physics", "import", str(namelist), "--mesh", "conus3km"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["refusals"] == []


def test_a_thompson_row_without_its_tables_is_refused_at_the_door(tmp_path, door_seams, monkeypatch):
    table_assets = pytest.importorskip("woof.table_assets")

    def missing(*_args, **_kwargs):
        raise table_assets.MissingTableAssets(
            "mp_physics=8 (Thompson) needs freezeH2O.dat -- run `woof fetch-tables`")

    monkeypatch.setattr(table_assets, "require_thompson_tables", missing)
    with pytest.raises(door.ForecastDoorRefusal, match="fetch-tables"):
        door.resolve_request(
            _namespace(tmp_path, "x1.40962", "--physics-suite", "arwen_thompson_column"),
            registry=_registry())
    request = door.resolve_request(
        _namespace(tmp_path, "x1.40962", "--physics-suite", "arwen_thompson_column",
                   "--preflight"), registry=_registry())
    assert any("fetch-tables" in problem for problem in request.input_problems)
    # the default row reads no Thompson table and is not asked
    assert door.resolve_request(_namespace(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path),
                                registry=_registry()).physics_backend == "wsm6_column"


# ---------------------------------------------------------------------------
# review 2026-10-07: what a namelist leaves out, the auto cumulus rule behind
# the column-suite seam, presets, and the engine's generation support
# ---------------------------------------------------------------------------
_MESOSCALE_ON_THE_FROZEN_ROW = """&physics
    config_physics_suite = 'mesoscale_reference'
    config_convection_scheme = 'cu_grell_freitas'
    config_lsm_scheme = 'sf_noahmp'
    config_frac_seaice = false
    config_microp_re = true
/
"""


def test_a_namelist_naming_no_suite_is_mpas_mesoscale_reference_not_hex_default():
    """MPAS's Registry default for config_physics_suite is mesoscale_reference
    (v8.4.1 and the RRFS fork alike).  The importer once resolved an absent
    suite to hex's own default, running WSM6/YSU/Noah-MP under a namelist
    whose MPAS run is New Tiedtke/Noah."""

    result = import_physics_namelist(
        "&physics\n    config_microp_scheme = 'mp_wsm6'\n/\n")
    assert result.suite == "mesoscale_reference"
    assert not result.admitted
    # Noah itself is no longer refused (it runs on the generic seam); what
    # refuses this namelist is what its Registry defaults ask for that
    # nothing would honour.
    assert "config_microp_re" in {r.knob for r in result.refusals}
    assert "config_lsm_scheme" not in {r.knob for r in result.refusals}
    suite = next(d for d in result.dispositions if d.option == "config_physics_suite")
    assert suite.status == "admitted" and "Registry default" in suite.detail
    # a namelist with no &physics at all reads the same way
    assert import_physics_namelist("&nhyd_model\n    config_dt = 120.0\n/\n").suite == (
        "mesoscale_reference")


def test_an_absent_option_reads_at_mpas_registry_default():
    result = import_physics_namelist(_MESOSCALE_ON_THE_FROZEN_ROW)
    assert result.admitted, result.report()["refusals"]
    assert result.physics.backend_row == "wsm6_column"
    # MPAS radiates every 30 min when the namelist says nothing, not hex's 600 s
    assert result.radiation_seconds == 1800.0
    details = {d.option: d for d in result.dispositions}
    assert "Registry default" in details["config_radtlw_interval"].detail
    # MYNN's mixing switch defaults to 1 in v8.4.1, and MPAS reads it only with MYNN
    assert details["config_mynn_mixscalars"].status == "ignored"
    # config_frac_seaice defaults to true: left out, it is refused on Noah-MP and
    # the refusal says the namelist never wrote it
    result = import_physics_namelist(
        _MESOSCALE_ON_THE_FROZEN_ROW.replace("    config_frac_seaice = false\n", ""))
    hit = [r for r in result.refusals if r.knob == "config_frac_seaice"]
    assert hit and not result.admitted
    # config_microp_re defaults to false in v8.4.1; hex always hands radiation the radii
    result = import_physics_namelist(
        _MESOSCALE_ON_THE_FROZEN_ROW.replace("    config_microp_re = true\n", ""))
    hit = [r for r in result.refusals if r.knob == "config_microp_re"]
    assert hit and "not in the namelist" in hit[0].reason


def test_a_hex_suite_in_a_namelist_reads_no_mpas_registry():
    result = import_physics_namelist(
        "&physics\n    config_physics_suite = 'arwen_mpas_column'\n/\n")
    assert result.admitted and result.radiation_seconds is None
    assert result.physics.is_default()


def test_mynn_mixing_is_held_only_where_mpas_reads_it():
    text = _MESOSCALE_ON_THE_FROZEN_ROW.replace(
        "/\n", "    config_mynn_mixscalars = 1\n    config_mynn_mixaerosols = 1\n/\n", 1)
    result = import_physics_namelist(text)
    assert result.admitted, result.report()["refusals"]
    statuses = {d.option: d.status for d in result.dispositions}
    assert statuses["config_mynn_mixscalars"] == "ignored"
    assert statuses["config_mynn_mixaerosols"] == "ignored"


def test_the_registry_defaults_are_table_rows_the_importer_reads():
    spec = ps.load_table()["namelist"]
    options = spec["options"]
    for registry in spec["registry_defaults"]["registries"].values():
        assert set(registry["values"]) <= set(options)
        assert registry["values"]["config_physics_suite"] == "mesoscale_reference"
    damaged = ps._thaw(ps.load_table())
    damaged["namelist"]["registry_defaults"]["registries"]["mpas-a-v8.4.2"]["values"][
        "config_not_read"] = 1
    with pytest.raises(ps.PhysicsTableError, match="never dispositioned"):
        ps.validate_table(damaged)


def test_auto_cumulus_behind_the_column_suite_is_decided_at_the_door():
    """``auto`` resolves to New Tiedtke at 20 km and coarser. The column-suite
    seam carries New Tiedtke since the RRFS v2 rows, so the door admits the
    request at both spacings; before that it refused the 20 km case first,
    instead of letting the sealed constructor refuse after the mesh bind."""

    mynn = {"config_pbl_scheme": "bl_mynn", "config_sfclayer_scheme": "sf_mynn",
            "config_radt_cld_scheme": "cld_fraction_mynn", "config_gwdo_scheme": "off"}
    coarse = ps.resolve_physics("arwen_mpas_column", mynn, nominal_dx_m=120_000.0)
    assert coarse.seam == "column_suite" and coarse.convection == "auto"
    _needs_suite_engine()
    resolved = ps.resolve_physics("arwen_mpas_column", mynn, nominal_dx_m=12_000.0)
    assert resolved.seam == "column_suite" and resolved.convection == "auto"


def test_a_held_preset_is_refused_on_column_suite_physics(tmp_path, door_seams):
    _needs_suite_engine()
    (tmp_path / "x1.40962.ugwp_oro_data.nc").write_bytes(b"x")
    with pytest.raises(door.ForecastDoorRefusal, match="--preset fast"):
        door.resolve_request(_namespace(
            tmp_path, "x1.40962", "--physics-suite", "rrfs_v2_dev", "--preset", "fast"),
            registry=_registry())
    request = door.resolve_request(_namespace(
        tmp_path, "x1.40962", "--physics-suite", "rrfs_v2_dev"), registry=_registry())
    preset = door._preset_receipt(request)
    assert preset["suite"] is None and preset["declared_suite"]["boundary_layer"] == "ysu"
    bare = door.resolve_request(_namespace(tmp_path), registry=_registry())
    assert door._preset_receipt(bare)["suite"]["boundary_layer"] == "ysu"


def test_generation_selectors_need_an_engine_that_takes_them(tmp_path):
    _needs_suite_engine()
    from woof.hex import cuda_suite_physics_v841 as adapter

    resolved = ps.resolve_physics("rrfs_v2_dev", {"bl_mynn_version": "gsd_41"})
    selection = resolved.column_profile["selection"]
    for relative in adapter._pinned_paths(selection):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stand-in\n", encoding="utf-8")
    problem = adapter.pin_problem(tmp_path, resolved.backend_row)
    assert problem is not None and "resolve_generation" in problem
    seam = tmp_path / "woof/core/mpas_column_suite.py"
    seam.write_text("def resolve_generation(*a, **k):\n    pass\n", encoding="utf-8")
    assert adapter.pin_problem(tmp_path, resolved.backend_row) is None
    # a generation-less row asks nothing of the seam
    seam.write_text("# no generation\n", encoding="utf-8")
    assert adapter.pin_problem(tmp_path, "mynn_ruc_gsl_column") is None
