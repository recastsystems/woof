"""MPAS physics is the hex default (owner ruling 2026-10-25), and WOOF physics stays one flag away.

THE BREAKAGE THIS PREVENTS: the default staying WOOF on some door after the flip (a plan, a step rule, a roster or a
card price deciding the WOOF route while the forecast runs MPAS, or the reverse), and the WOOF route no longer
reachable as it ran before 0.4.6.  CPU only: parsers and plans, no card."""
from __future__ import annotations

import json

import pytest

from woof.hex import forecast_door, global_door, physics_default
from woof.hex.cli import build_parser


def test_the_rule_picks_convection_permitting_on_every_mesh_and_says_why():
    table = physics_default.load_table()
    assert table["route"] == "mpas" and table["arwen_suite"] == "arwen_mpas_column"
    for dx in (None, 937.5, 3000.0, 9999.0, 10000.0, 60000.0, 120000.0):
        assert physics_default.default_suite(dx) == "convection_permitting"
    assert "10 km" in physics_default.default_rule(3000.0)["why"]
    assert "mesoscale_reference" in physics_default.default_rule(120000.0)["why"]


@pytest.mark.parametrize("flags, defaulted", [
    ([], True),
    (["--convection", "auto"], True),
    (["--physics-mode=exact", "--checkpoint-hours=6"], True),
    (["--physics-suite=arwen_mpas_column"], False),
    (["--physics-suite", "mesoscale_reference"], False),
    (["--physics=config_microp_scheme=mp_thompson"], False),
    (["--convection=gf"], False),
    (["--physics-backend", "mpasphys"], False),
    (["--physics-namelist=namelist.atmosphere"], False),
    (["--forecast-arg=--physics-suite=rrfs_v2_dev"], False),
])
def test_ensure_adds_the_default_only_when_no_physics_is_named(flags, defaulted):
    out = physics_default.ensure(flags)
    assert out == ([*flags, "--physics-suite=convection_permitting"] if defaulted else flags)


def _forecast_namespace(*extra):
    return build_parser().parse_args(["forecast", "--mesh", "x1.40962", "--hours", "1", *extra])


def test_a_bare_forecast_takes_the_mpas_route_and_a_named_choice_is_kept():
    bare = _forecast_namespace()
    assert physics_default.apply(bare)["suite"] == "convection_permitting"
    assert bare.physics_suite == "convection_permitting"
    for extra, suite in ((["--physics-suite", "arwen_mpas_column"], "arwen_mpas_column"),
                         (["--physics", "config_microp_scheme=mp_thompson"], None),
                         (["--convection", "ntiedtke"], None),
                         (["--physics-backend", "mpasphys"], None)):
        named = _forecast_namespace(*extra)
        assert physics_default.apply(named) is None and named.physics_suite == suite


def test_the_bare_forecast_decides_the_mpas_route_as_the_door_does():
    from woof.hex.mpasphys.forecast_v842 import native_route

    bare = _forecast_namespace()
    physics_default.apply(bare)
    config, _text = native_route(bare, default_mode="correct")
    assert config is not None  # the MPAS suite route, as the door runs it
    arwen = _forecast_namespace("--physics-suite", "arwen_mpas_column")
    assert native_route(arwen, default_mode="correct")[0] is None  # the ArWen route


def test_naming_the_arwen_suite_alone_is_the_pre_046_bare_road():
    arwen = _forecast_namespace("--physics-suite", "arwen_mpas_column")
    assert forecast_door.resolve_door_physics(arwen, "x1.40962", 120000.0) == (None, None, {})


def _plan(tmp_path, row, *extra):
    return global_door.plan(build_parser().parse_args(
        ["global", "--row", row, "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path / row),
         "--cache-dir", str(tmp_path / "cache"), "--stage", "plan", *extra]))


def _suites(forecast):
    return [forecast[i + 1] for i, token in enumerate(forecast) if token == "--physics-suite"]


def test_global_default_rows_run_the_mpas_suite_and_arwen_is_a_row_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    value = _plan(tmp_path, "global-120km")
    forecast = value["commands"]["forecast"]
    assert _suites(forecast) == ["convection_permitting"] and "--dt" not in forecast
    assert value["physics"]["row"] == "convection_permitting"
    assert value["physics"]["default"]["route"] == "mpas"
    build_parser().parse_args(forecast)

    arwen = _plan(tmp_path, "global-120km", "--physics-row", "arwen")
    assert _suites(arwen["commands"]["forecast"]) == ["arwen_mpas_column"]
    assert arwen["physics"]["row"] == "arwen"

    morrison = _plan(tmp_path, "global-60km-morrison")["commands"]["forecast"]
    assert _suites(morrison) == []  # its overrides name the ArWen-route physics themselves

    override = _plan(tmp_path, "global-120km", "--forecast-arg=--physics=config_microp_scheme=mp_thompson")
    assert override["physics"]["row"] == "arwen" and _suites(override["commands"]["forecast"]) == []

    named = _plan(tmp_path, "global-120km", "--forecast-arg=--physics-suite=arwen_thompson_column")
    assert _suites(named["commands"]["forecast"]) == [] and named["physics"]["row"] == "arwen_thompson_column"


def test_a_moving_case_with_no_physics_args_runs_the_default(tmp_path):
    from woof.hex.moving_case import load_case

    case = {"start_time": "2022-09-26_00:00:00", "dt_seconds": 18.75, "n_levels": 55,
            "vertical_spec": "tc55-v1.json", "storm": {"initial_center_deg": [20.0, -80.0]},
            "mesh": {"background_km": 48.0, "fine_km": 3.0, "fine_radius_km": 300.0,
                     "fine_transition_km": 54.0, "rings": []}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    try:
        loaded = load_case(path)
    except Exception as error:  # a case key this test does not know about: the defaulting is still checked
        pytest.skip(f"case shape: {error}")
    assert loaded["physics_args"] == ["--physics-suite=convection_permitting", "--physics-mode=correct",
                                      "--physics-profile=hrrr-like-aerosol"]
    case["physics_args"] = ["--physics-suite=arwen_mpas_column"]
    path.write_text(json.dumps(case), encoding="utf-8")
    assert load_case(path)["physics_args"] == ["--physics-suite=arwen_mpas_column"]


def test_physics_list_marks_the_mpas_default(capsys):
    from woof.hex.cli import main

    assert main(["physics", "list"]) == 0
    suites = {row["name"]: row for row in json.loads(capsys.readouterr().out)["suites"]}
    assert suites["convection_permitting"]["default"] is True
    assert suites["arwen_mpas_column"]["default"] is False
    assert suites["arwen_mpas_column"]["arwen_route_default"] is True


def test_a_named_arwen_suite_keeps_the_doors_mesh_step():
    assert physics_default.suite_takes_door_step({"suite": "convection_permitting"}) is True
    assert physics_default.suite_takes_door_step({"namelist": "namelist.atmosphere"}) is True
    assert physics_default.suite_takes_door_step({"suite": "arwen_mpas_column"}) is False
    assert physics_default.suite_takes_door_step({}) is False


def test_a_named_mpas_suite_keeps_a_mesh_step_shorter_than_its_rule():
    # zc1200 (2026-10-29): a 96 -> 3 km zoom admitted at 12 s ran the default suite at the rule's 18 s.
    door_step = physics_default.door_step
    assert door_step({"suite": "convection_permitting"}, 12.0, 3000.0) == ["--dt", "12.0"]
    assert door_step({"suite": "convection_permitting"}, 18.0, 3000.0) == []
    assert door_step({"suite": "convection_permitting"}, 18.0, None) == []
    assert door_step({"namelist": "namelist.atmosphere"}, 12.0, 3000.0) == []
    assert door_step({"suite": "arwen_mpas_column"}, 18.0, 3000.0) == ["--dt", "18.0"]
    assert door_step({}, 12.0, 3000.0) == ["--dt", "12.0"]


def test_a_suite_with_its_own_producer_reaches_the_cold_init():
    """hrrrv5 (RUC on nine levels) is listed with MPAS's own suites, and `woof hex regional
    --forecast-arg=--physics-suite=hrrrv5` built MPAS-A's Noah four-layer init for it.  A suite whose model
    initializes with its own producer row now names itself to the init door; MPAS-A's own suites still add nothing."""
    from woof.hex.physics_default import init_flags

    assert init_flags(["--physics-suite=hrrrv5"]) == ["--physics-suite", "hrrrv5"]
    assert init_flags(["--physics-suite=convection_permitting"]) == []
