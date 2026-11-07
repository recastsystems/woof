"""MPAS-A defaults are the defaults, and no certificate gate refuses a run.

Owner ruling 2026-10-10., and earlier "why would we ever run at not actual normal
defaults".  Three things are held here:

* a bare run gets MPAS-A v8.4.2's Registry schedule -- RK order 2, four
  acoustic sub-steps (two through v8.4.1, still selectable), three splits, the monotonic limiter, radiation every
  1800 s, surface/PBL and cumulus every step -- and a timestep of 6 s per km
  of the mesh's nominal spacing, each value read from the cited upstream line
  (verification/mpas-v8.4.2-registry/cited-lines.txt);
* the configurations the certificate gates refused (two sub-steps on the RK3
  lane, radiation 1800 s, dt 90 s at 15 km, dt 180 s at 30 km) are admitted;
* the refusals that name a concrete breakage still fire: a cadence the
  timestep cannot schedule, a clock that does not close, a step above the
  advective or acoustic stability limit, a scheme no seam runs.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

import numpy as np
import pytest

from woof.hex import dt_admission, forecast_door as door, mpas_defaults
from woof.hex.config_v841 import (
    V841MpasColumnPhysicsConfig,
    V841MpasColumnPhysicsGwdoConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from woof.hex.errors import ConfigurationRefusal
from woof.hex.timestep_admission import (
    CourantPolicy,
    ACOUSTIC_COURANT_LIMIT,
    SplitExplicitSchedule,
    TimestepAdmissionError,
    admit_timestep,
    edge_length_authority,
    one_dimensional_advective_limit,
)
from _layout import PACKAGE_DIR

TREE = Path(__file__).resolve().parents[1]
CITED = TREE / "verification" / "mpas-v8.4.2-registry" / "cited-lines.txt"

#: Measured minimum dcEdge of the staged quasi-uniform meshes (8-card RTX PRO 6000 host,
#: 2026-10-10, static.dcEdge): x1.2621442 (15 km), x1.655362 (30 km),
#: x1.163842 (60 km); nominalMinDc is 15000, 30000, 60000 m.
MIN_DC_EDGE_M = {15_000.0: 11_036.673828125, 30_000.0: 22_846.64453125,
                 60_000.0: 47_094.6484375}


# ---------------------------------------------------------------------------
# the defaults, with their citations
# ---------------------------------------------------------------------------
def _cited_blocks() -> dict[str, str]:
    blocks: dict[str, str] = {}
    current = None
    for line in CITED.read_text(encoding="utf-8").splitlines():
        if line.startswith("== "):
            current = line[3:].strip()
            blocks[current] = ""
        elif current is not None:
            blocks[current] += line + "\n"
    return blocks


def test_every_registry_default_matches_the_line_it_cites():
    blocks = _cited_blocks()
    for knob, row in mpas_defaults.REGISTRY_DEFAULTS.items():
        match = re.match(r"Registry\.xml:(\d+)", row["citation"])
        assert match, knob
        line = int(match.group(1))
        block = next(text for name, text in blocks.items()
                     if name.startswith("Registry.xml:")
                     and int(name.split(":")[1].split("-")[0]) == line)
        assert f'name="{knob}"' in block, knob
        value = row["value"]
        literal = ("true" if value else "false") if isinstance(value, bool) else (
            str(value) if not isinstance(value, float) else repr(value))
        assert f'default_value="{literal}"' in block, (knob, literal)


def test_the_every_step_cadences_cite_the_physics_manager():
    blocks = _cited_blocks()
    manager = "".join(text for name, text in blocks.items()
                      if name.startswith("mpas_atmphys_manager.F"))
    assert "dt_cu = dt" in manager and "dt_pbl = dt" in manager


def test_the_schedule_defaults_are_mpas_a_registry_defaults():
    assert (mpas_defaults.DEFAULT_TIME_INTEGRATION_ORDER,
            mpas_defaults.DEFAULT_ACOUSTIC_SUBSTEPS,
            mpas_defaults.DEFAULT_DYNAMICS_SPLITS,
            mpas_defaults.DEFAULT_RADIATION_SECONDS,
            mpas_defaults.DEFAULT_MONOTONIC) == (2, 4, 3, 1800.0, True)
    for config_type in (V841MpasColumnPhysicsConfig, V841MpasColumnPhysicsGwdoConfig,
                        V841MpasColumnPhysicsSmagorinskyGwdoConfig):
        config = config_type()
        config.validate()
        assert (config.config_time_integration_order, config.config_number_of_sub_steps,
                config.config_dynamics_split_steps, config.config_radt_seconds,
                config.config_monotonic) == (2, 4, 3, 1800.0, True)
        # surface/PBL and cumulus every step, as config_pbl_interval and
        # config_conv_interval = 'none' make them in MPAS-A
        assert config.config_bldt_seconds == config.config_dt
        assert config.config_cudt_seconds == config.config_dt


@pytest.mark.parametrize("nominal, dt", [
    (15_000.0, 90.0), (30_000.0, 180.0), (60_000.0, 360.0), (25_000.0, 150.0),
    (3_000.0, 18.0), (937.5, 5.0), (120_000.0, 600.0),
])
def test_the_default_timestep_is_six_seconds_per_km(nominal, dt):
    chosen = mpas_defaults.default_dt_seconds(nominal)
    assert chosen["dt_seconds"] == dt
    assert chosen["source"] == "mesh"
    assert 1800.0 % chosen["dt_seconds"] == 0.0


def test_the_default_timestep_divides_a_named_radiation_interval():
    assert mpas_defaults.default_dt_seconds(15_000.0, radiation_seconds=600.0)["dt_seconds"] == 75.0


def test_resolve_dt_spells_auto_row_and_seconds():
    assert mpas_defaults.resolve_dt(None, nominal_dx_m=15_000.0, row_dt_seconds=60.0)["dt_seconds"] == 90.0
    assert mpas_defaults.resolve_dt("auto", nominal_dx_m=15_000.0, row_dt_seconds=60.0)["dt_seconds"] == 90.0
    row = mpas_defaults.resolve_dt("row", nominal_dx_m=15_000.0, row_dt_seconds=60.0)
    assert (row["dt_seconds"], row["source"]) == (60.0, "row")
    explicit = mpas_defaults.resolve_dt("75", nominal_dx_m=15_000.0, row_dt_seconds=60.0)
    assert (explicit["dt_seconds"], explicit["source"]) == (75.0, "explicit")
    for bad in ("fast", "-1", "nan"):
        with pytest.raises(ValueError):
            mpas_defaults.resolve_dt(bad, nominal_dx_m=15_000.0, row_dt_seconds=60.0)


# ---------------------------------------------------------------------------
# the door: a bare request runs MPAS-A's defaults, every knob overridable
# ---------------------------------------------------------------------------
def _registry() -> dict[str, door.MeshRow]:
    return {
        "g15": door.MeshRow("g15", 2_621_442, 60.0, nominal_dx_m=15_000.0),
        "g30": door.MeshRow("g30", 655_362, 120.0, nominal_dx_m=30_000.0),
        "g60": door.MeshRow("g60", 163_842, 120.0, nominal_dx_m=60_000.0),
    }


def _request(tmp_path: Path, mesh: str, *extra: str):
    for name in ("grid.nc", "static.nc", "init.nc"):
        (tmp_path / name).write_bytes(b"not really netcdf")
    (tmp_path / "woof").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args([
        "--mesh", mesh, "--grid", str(tmp_path / "grid.nc"),
        "--static", str(tmp_path / "static.nc"), "--init", str(tmp_path / "init.nc"),
        "--init-source", "GFS 2026-10-10 00Z", "--hours", "6",
        "--history-every-minutes", "60", "--out", str(tmp_path / "out"),
        "--gpuwm-checkout", str(tmp_path / "woof"), *extra,
    ])
    return door.resolve_request(arguments, registry=_registry())


@pytest.fixture
def no_pins(monkeypatch):
    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None)
    monkeypatch.setattr(door, "backend_pin_problem", lambda row, checkout: None, raising=False)
    monkeypatch.setattr(door, "microphysics_table_problem", lambda row: None)


@pytest.fixture
def bounded_engine(monkeypatch):
    """An engine carrying both donor bounds, so the YSU long-step cap is retired.

    The default step on a YSU run is capped at 300 s on an engine without them (woof.hex.global_door.
    default_step_cap, hex 0.4.0 recut); the test below reads MPAS-A's own defaults, which is what the door
    runs once the cap retires.  The installed-pair G0 on PyPI gpuwm 2.9.0 (no bounds) found it asserting 360 s
    on the 60 km row while the door took 300 s."""
    from woof.hex import global_door

    arm = global_door.load_table()["physics"]["arwen"]
    monkeypatch.setattr(global_door, "engine_identities", lambda: frozenset(arm["max_dt_retired_by_engine"]))


@pytest.mark.parametrize("mesh, dt", [("g15", 90.0), ("g30", 180.0), ("g60", 360.0)])
def test_a_bare_door_request_runs_mpas_a_defaults(tmp_path, no_pins, bounded_engine, mesh, dt):
    request = _request(tmp_path, mesh)
    assert request.dt_seconds == dt
    assert request.dt_selection["source"] == "mesh"
    assert (request.time_integration_order, request.acoustic_substeps,
            request.radiation_seconds, request.monotonic) == (2, 4, 1800.0, True)
    assert set(request.schedule_sources.values()) == {"mpas-defaults"}
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--time-integration-order") + 1] == "2"
    assert argv[argv.index("--acoustic-substeps") + 1] == "4"
    assert argv[argv.index("--radiation-seconds") + 1] == "1800.0"
    assert "--monotonic" in argv
    assert request.steps == int(6 * 3600 / dt)


def test_ucar_apvm_choice_reaches_the_driver_config(tmp_path, no_pins):
    from woof.hex.drivers.run_cuda_v841_forecast import build_forecast_config, parse_dynamics_overrides

    request = _request(tmp_path, "g15", "--dt", "18", "--dynamics", "config_apvm_upwinding=0")
    argv = door.build_driver_argv(request)
    setting = argv[argv.index("--dynamics") + 1]
    overrides = parse_dynamics_overrides([setting])
    config = build_forecast_config(dt_seconds=request.dt_seconds,
                                   dynamics_overrides=overrides)
    config.validate()
    assert config.config_apvm_upwinding == 0.0
    assert build_forecast_config(dt_seconds=18).config_apvm_upwinding == 0.5


def test_every_schedule_knob_stays_overridable(tmp_path, no_pins):
    request = _request(tmp_path, "g15", "--dt", "60", "--time-integration-order", "3",
                       "--acoustic-substeps", "6", "--radiation-seconds", "600",
                       "--no-monotonic", "--pbl-cadence", "120")
    assert request.dt_seconds == 60.0 and request.dt_selection["source"] == "explicit"
    assert (request.time_integration_order, request.acoustic_substeps,
            request.radiation_seconds, request.monotonic) == (3, 6, 600.0, False)
    assert request.pbl_cadence == "120"
    argv = door.build_driver_argv(request)
    assert "--no-monotonic" in argv
    row = _request(tmp_path, "g15", "--dt", "row")
    assert row.dt_seconds == 60.0 and row.dt_selection["source"] == "row"


def test_local_timestep_takes_its_own_schedule_when_none_is_named(tmp_path, no_pins):
    request = _request(tmp_path, "g15", "--local-timestep")
    assert (request.time_integration_order, request.acoustic_substeps,
            request.monotonic) == (3, 6, False)
    assert request.schedule_sources["order"] == "local-timestep"


def test_an_unschedulable_radiation_interval_is_refused_at_the_door(tmp_path, no_pins):
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        _request(tmp_path, "g15", "--dt", "90", "--radiation-seconds", "600")
    assert "not a positive integer multiple" in str(caught.value)


# ---------------------------------------------------------------------------
# the configurations the gates refused, admitted
# ---------------------------------------------------------------------------
def test_two_acoustic_substeps_on_the_rk3_lane_is_admitted():
    """'config_number_of_sub_steps=2 is refused: the matched v8.4.1 real-x4
    column-physics lane is exact' -- measured on the 8-card RTX PRO 6000 host, 2026-10-10."""

    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=120.0, config_time_integration_order=3, config_number_of_sub_steps=2,
        config_radt_seconds=600.0,
    ).validate()


def test_radiation_every_1800_seconds_is_admitted_at_an_anchored_dt():
    """'config_dt=120 s is anchored, but at physics cadences this run does not
    use' -- measured on the 8-card RTX PRO 6000 host, 2026-10-10."""

    config = V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=120.0, config_time_integration_order=3, config_number_of_sub_steps=6,
        config_radt_seconds=1800.0, config_monotonic=False,
    )
    config.validate()
    record = dt_admission.anchor_record(
        120.0, radiation_seconds=1800.0, surface_pbl_seconds=120.0, cumulus_seconds=120.0)
    assert record["matched"] is False and record["row"]["radiation_seconds"] == 600.0


@pytest.mark.parametrize("nominal, dt", [(15_000.0, 90.0), (30_000.0, 180.0), (60_000.0, 360.0)])
def test_the_mpas_standard_timestep_is_admitted_on_the_real_mesh_geometry(nominal, dt):
    """'holds no dycore step certificate' refused every timestep but 15, 20
    and 60 s; the physical check admits MPAS-A's own steps on the measured
    minimum dcEdge of the staged meshes."""

    from woof.hex.step_anchors import step_anchor_record

    config = V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=dt, config_bldt_seconds=dt, config_cudt_seconds=dt)
    config.validate()
    assert step_anchor_record(config)["matched"] is False
    authority = edge_length_authority(np.asarray([MIN_DC_EDGE_M[nominal], nominal]))
    admitted = admit_timestep(dt, authority)
    assert admitted.estimated_outer_courant < admitted.admitted_courant
    assert admitted.acoustic_courant < ACOUSTIC_COURANT_LIMIT
    record = door.admit_timestep("g", dt, nominal_dx_m=nominal)
    assert record["admitted"] is True and record["anchor"]["matched"] is False


def test_the_old_rk3_six_substep_60_second_lane_still_runs():
    """The configuration the overnight 15 km run used stays reachable."""

    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=60.0, config_bldt_seconds=60.0, config_cudt_seconds=60.0,
        config_time_integration_order=3, config_number_of_sub_steps=6,
        config_radt_seconds=600.0, config_monotonic=False,
    ).validate()
    authority = edge_length_authority(np.asarray([MIN_DC_EDGE_M[15_000.0]]))
    admit_timestep(60.0, authority, schedule=SplitExplicitSchedule(3, 6))


# ---------------------------------------------------------------------------
# the refusals that name a breakage still fire
# ---------------------------------------------------------------------------
def test_a_cadence_the_timestep_cannot_schedule_is_refused():
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsConfig(
            config_dt=90.0, config_bldt_seconds=90.0, config_cudt_seconds=90.0,
            config_radt_seconds=600.0,
        ).validate()
    assert "radiation_seconds=600 s is not a positive integer multiple of dt=90 s" in str(caught.value)


def test_a_timestep_whose_clock_does_not_close_is_refused():
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsConfig(
            config_dt=0.1, config_bldt_seconds=0.1, config_cudt_seconds=None,
            config_convection_scheme="off", config_radt_seconds=1800.0,
        ).validate()
    assert "clock closure fails" in str(caught.value)


def test_an_unstable_timestep_is_refused_with_both_courant_numbers():
    authority = edge_length_authority(np.asarray([MIN_DC_EDGE_M[15_000.0]]))
    # Two acoustic sub-steps (the default through MPAS-A v8.4.1): 180 s on the
    # 15 km mesh breaks both bounds.  At the v8.4.2 default of four the
    # acoustic step is half as long and only the advective bound is broken.
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(180.0, authority, schedule=SplitExplicitSchedule(2, 2))
    message = str(caught.value)
    assert "advective Courant estimate U_max*dt/min(dcEdge)" in message
    assert "acoustic Courant c_max*dtau/min(dcEdge)" in message
    assert "von Neumann" in message and "forward-backward" in message


def test_the_advective_refusal_says_it_is_an_estimate_and_can_be_adjusted():
    """2026-10-10 audit: dt 105 s on the 15 km mesh was refused at 1.189
    against 1.160 on an assumed 125 m/s wind and a rule-of-thumb hexagon
    factor, with no way to state the case's own bound.  The refusal now says
    what is assumed, gives the wind the step is good for, and --max-wind
    (CourantPolicy.max_characteristic_speed_m_s) moves the bound."""

    authority = edge_length_authority(np.asarray([MIN_DC_EDGE_M[15_000.0]]))
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(105.0, authority, cadence_seconds=1800.0)
    message = str(caught.value)
    assert "estimate" in message and "assumed largest wind (default)" in message
    assert "not an analysis of the 2-D operator" in message
    assert "--max-wind" in message
    assert "acoustic Courant" not in message, "the acoustic limit passes at 105 s"

    wind_at_limit = (
        0.75 * one_dimensional_advective_limit(2, 0.25) * MIN_DC_EDGE_M[15_000.0] / 105.0
    )
    assert 115.0 < wind_at_limit < 125.0
    assert f"winds up to {wind_at_limit:.1f} m/s" in message
    admitted = admit_timestep(
        105.0, authority,
        policy=CourantPolicy(
            max_characteristic_speed_m_s=float(int(wind_at_limit)),
            max_characteristic_speed_source="--max-wind"),
        cadence_seconds=1800.0)
    assert admitted.resolved_dt_seconds == 105.0
    assert admitted.policy.as_dict()["max_characteristic_speed_source"] == "--max-wind"
    # A stricter bound is a stricter check.
    with pytest.raises(TimestepAdmissionError):
        admit_timestep(
            90.0, authority, policy=CourantPolicy(max_characteristic_speed_m_s=150.0))


def test_the_recommended_timestep_is_one_the_run_can_schedule():
    """2026-10-10 audit: the refusal said "declare dt <= 102.4 s"; 102.4
    does not divide the 1800 s radiation interval, so following the advice
    met the cadence refusal next."""

    from woof.hex.timestep_admission import largest_schedulable_dt

    authority = edge_length_authority(np.asarray([MIN_DC_EDGE_M[15_000.0]]))
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(105.0, authority, cadence_seconds=1800.0)
    message = str(caught.value)
    assert "--dt 100 or smaller" in message and "1800 s radiation interval" in message
    assert "102.4" not in message.split("Any one of:")[1]
    assert largest_schedulable_dt(102.4, 1800.0) == 100.0
    assert 1800.0 % largest_schedulable_dt(102.4, 1800.0) == 0.0
    admit_timestep(100.0, authority, cadence_seconds=1800.0)
    # With no cadence known the old tenth-of-a-second form stands.
    with pytest.raises(TimestepAdmissionError, match=r"dt <= 102\.4 s"):
        admit_timestep(105.0, authority)
    # The acoustic remedy is a count that passes.
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(180.0, authority, cadence_seconds=1800.0,
                       schedule=SplitExplicitSchedule(2, 2))
    assert "--acoustic-substeps 3 or more" in str(caught.value)
    # and four, the default, is one: the same step is refused for its
    # advective number alone
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(180.0, authority, cadence_seconds=1800.0)
    assert "--acoustic-substeps" not in str(caught.value)


def test_the_stability_limits_are_the_derived_numbers():
    assert one_dimensional_advective_limit(3, 0.25) == pytest.approx(1.520, abs=1e-3)
    assert one_dimensional_advective_limit(2, 0.25) == pytest.approx(1.547, abs=1e-3)
    assert one_dimensional_advective_limit(3, 0.0) == pytest.approx(1.262, abs=1e-3)
    assert one_dimensional_advective_limit(3, 1.0) == pytest.approx(1.626, abs=1e-3)
    assert ACOUSTIC_COURANT_LIMIT == pytest.approx(2.0 / np.sqrt(6.0))
    assert SplitExplicitSchedule().largest_acoustic_fraction() == pytest.approx(1.0 / 12.0)
    assert SplitExplicitSchedule(2, 2).largest_acoustic_fraction() == pytest.approx(1.0 / 6.0)
    assert SplitExplicitSchedule(3, 6).largest_acoustic_fraction() == pytest.approx(1.0 / 9.0)


def test_a_selector_no_seam_runs_is_refused_by_name():
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsConfig(config_pbl_scheme="bl_mynn").validate()
    assert "recorded in the receipt and not run" in str(caught.value)
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsConfig(config_convection_scheme="cu_never_resolved").validate()
    assert "no physics selection resolved this name" in str(caught.value)


def test_three_dynamics_splits_stay_required_where_the_cuda_subcycle_is_written_for_three():
    from woof.hex.cuda_driver import CudaDryDycoreDriver

    config = V841MpasColumnPhysicsConfig()
    CudaDryDycoreDriver._validate_config(config)
    from dataclasses import replace
    with pytest.raises(ConfigurationRefusal) as caught:
        CudaDryDycoreDriver._validate_config(replace(config, config_dynamics_split_steps=1))
    assert "implemented for three subcycles" in str(caught.value)


def test_positive_definite_and_scalar_advection_off_are_no_longer_refused():
    from dataclasses import replace
    from woof.hex.cuda_driver import CudaDryDycoreDriver

    config = V841MpasColumnPhysicsConfig()
    CudaDryDycoreDriver._validate_config(replace(config, config_positive_definite=True))
    CudaDryDycoreDriver._validate_config(replace(config, config_apvm_upwinding=0.0))
    with pytest.raises(ConfigurationRefusal):
        CudaDryDycoreDriver._validate_config(replace(config, config_apvm_upwinding=-0.1))


# ---------------------------------------------------------------------------
# the byte-identity lane is a test with its configuration passed explicitly
# ---------------------------------------------------------------------------
def test_the_x4_proof_passes_the_recorded_schedule_explicitly():
    source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py").read_text(encoding="utf-8")
    assert "V841MpasColumnPhysicsGwdoConfig(**X4_RECORDED_SCHEDULE)" in source
    assert dict(mpas_defaults.X4_RECORDED_SCHEDULE) == {
        "config_dt": 120.0, "config_time_integration_order": 3,
        "config_number_of_sub_steps": 6, "config_dynamics_split_steps": 3,
        "config_radt_seconds": 600.0, "config_monotonic": False,
    }
    recorded = V841MpasColumnPhysicsGwdoConfig(**mpas_defaults.X4_RECORDED_SCHEDULE)
    recorded.validate()
    assert recorded.recorded_lane_match()["matched"] is True
    assert V841MpasColumnPhysicsGwdoConfig().recorded_lane_match()["matched"] is False


_RETIRED_PHRASES = (
    "lane is exact",
    "holds no dycore step certificate",
    "holds no timestep anchor",
    "at physics cadences this run does not use",
    "holds no forecast mint",
    "physics nobody proved",
    "pinned WOOF source hash mismatch",
    "pinned WOOF contract surface hash mismatch",
    "has not been proved",
)


def _refusal_texts(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in ("_refuse", "_refusal"):
                target = node
        if target is None:
            continue
        text = " ".join(
            sub.value for sub in ast.walk(target)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
        )
        yield node.lineno, text


def test_the_unpinned_dynamics_options_are_settable_on_a_run():
    """The options the certificate gates used to pin are reachable: the
    driver's configuration builder takes them by their namelist names."""

    import importlib

    driver = importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")
    overrides = driver.parse_dynamics_overrides([
        "config_apvm_upwinding=0.25", "config_smdiv=0.2", "config_coef_3rd_order=1.0",
        "config_positive_definite=.true.", "config_smagorinsky_coef=0.2",
    ])
    config = driver.build_forecast_config(dt_seconds=90.0, dynamics_overrides=overrides)
    assert config.config_apvm_upwinding == 0.25 and config.config_smdiv == 0.2
    assert config.config_coef_3rd_order == 1.0 and config.config_positive_definite is True
    assert config.config_smagorinsky_coef == 0.2
    assert config.config_dt == 90.0, "the flags' knobs are untouched"

    with pytest.raises(ValueError, match="would be recorded and not run"):
        driver.build_forecast_config(dt_seconds=90.0, dynamics_overrides={"config_nope": 1})
    with pytest.raises(ValueError, match="--dt"):
        driver.build_forecast_config(dt_seconds=90.0, dynamics_overrides={"config_dt": 30.0})
    with pytest.raises(ValueError, match="not KEY=VALUE"):
        driver.parse_dynamics_overrides(["config_smdiv"])
    with pytest.raises(ValueError, match="twice"):
        driver.parse_dynamics_overrides(["config_smdiv=0.1", "config_smdiv=0.2"])


def test_the_regional_scalar_path_limits_under_either_switch():
    """Native takes the limited routine on the last stage when
    config_monotonic OR config_positive_definite is on
    (mpas_atm_time_integration.F:2953); the global CUDA path does too.  The
    regional branch tested config_monotonic alone and passed
    config_positive_definite=False, so a regional run with
    positive_definite on and monotonic off ran unlimited while its receipt
    recorded the switch (2026-10-10 audit).  Structural: this holds the
    condition and the arguments; no regional case ran on a card for it.
    """

    tree = ast.parse((PACKAGE_DIR / "cuda_driver.py").read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "stage.stage == 3" not in test:
            continue
        calls = [
            sub for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) == "advance_scalar_transport_cuda_v841"
            and any(k.arg == "bdy_mask_cell" for k in sub.keywords)
        ]
        if calls:
            hits.append((test, calls[0]))
    assert len(hits) == 1, "the regional limited-transport branch was not found"
    test, call = hits[0]
    assert "config_monotonic" in test and "config_positive_definite" in test
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert "self.config.config_positive_definite" in keywords["config_positive_definite"]
    assert "self.config.config_monotonic" in keywords["config_monotonic"]


def test_the_engine_and_source_pins_record_and_do_not_refuse_a_forecast():
    """The forecast's two byte checks are measurements (2026-10-10): this
    port's own sources against EXECUTION_SOURCE_PINS, and the engine's seam
    files against the pinned engine.  Neither may raise on a difference."""

    import importlib

    proof = importlib.import_module("woof.hex.drivers.run_cuda_v841_full_physics_x4")
    measured = proof.measure_execution_sources()
    assert set(measured) >= {"files", "sha256", "matches_pins", "moved_from_pins"}
    assert set(measured["files"]) == set(proof.EXECUTION_SOURCE_PINS)
    forecast = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    binding = (PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py").read_text(encoding="utf-8")
    for name, text in (("run_cuda_v841_forecast.py", forecast), ("mpas_mesh_binding.py", binding)):
        assert "require_frozen_execution_sources(" not in text, (
            f"{name} refuses a forecast for an edited source file")
        assert "measure_execution_sources(" in text


def test_no_runtime_refusal_on_the_forecast_path_names_a_certificate():
    """Every raise in the forecast path is scanned; none may refuse for a
    missing certificate, anchor, mint or exact parity lane."""

    paths = [
        PACKAGE_DIR / name for name in (
            "config_v841.py", "dt_admission.py", "step_anchors.py", "forecast_door.py",
            "cuda_driver.py", "cuda_regional_forecast_v841.py", "mesh_point.py",
            "mesh_row_candidate.py", "convection_admission.py", "pbl_cadence.py",
            "timestep_admission.py", "physics_backend_admission.py",
            "cuda_arwen_physics_v841.py", "engine_pin.py", "cuda_driver_lts.py",
        )
    ] + [PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py",
         PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py"]
    found = []
    for path in paths:
        for line, text in _refusal_texts(path):
            for phrase in _RETIRED_PHRASES:
                if phrase in text:
                    found.append(f"{path.name}:{line}: {phrase}")
    assert not found, found


def test_a_bare_door_request_on_an_engine_without_the_bounds_takes_the_ysu_cap(tmp_path, no_pins, monkeypatch):
    from woof.hex import global_door

    monkeypatch.setattr(global_door, "engine_identities", lambda: frozenset())
    request = _request(tmp_path, "g60")
    assert request.dt_seconds == 300.0
    assert request.dt_selection["source"] == "physics cap"
    assert request.dt_selection["mesh_dt_seconds"] == 360.0
    assert _request(tmp_path, "g30").dt_seconds == 180.0
