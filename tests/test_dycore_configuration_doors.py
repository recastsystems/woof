"""Fork dynamics are explicit at source doors; other emitted bytes stay pinned."""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from woof.experiment import build_experiment, load_experiment
from woof.namelist_import import import_namelists, parse_namelist
from woof.physics_source_defaults import with_physics_selector_comment

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/source_requests"
FORK_VALUES = {
    "diff_6th_form": "noaa_wrf39", "diff_6th_factor2": .04,
    "mp_zero_out": 2, "mp_zero_out_thresh": 1e-12,
    "mp_zero_out_all": 1, "upper_wind_limiter_form": "noaa_wrf39",
}


def assert_fork(run):
    for key, value in FORK_VALUES.items():
        assert getattr(run, key) == value, key


def test_generic_namelist_keeps_current_staging_bytes_and_legacy_defaults(tmp_path):
    from test_namelist_import import _pair
    text, _ = import_namelists(*_pair(tmp_path), name="dycore-control")
    baseline = json.loads((FIXTURE / "identity-34951539d98-namelist.json").read_text())
    from wrf_clock_lines import WRF_CLOCK_BLOCK, without_wrf_clock_lines
    assert WRF_CLOCK_BLOCK in text
    # Re-recorded by running the base 34951539d98 importer: the earlier
    # e62a72e48 fixture already failed on that base. Keep the same clock
    # and retired-selector normalization used by the earlier assertion.
    from ishmael_retirement import with_retired_line
    assert hashlib.sha256(with_retired_line(
        without_wrf_clock_lines(text)).encode()
                          ).hexdigest() == baseline["namelist"]
    raw = tomllib.loads(text)
    emitted = set(raw["shared"]) | {key for row in raw["domain"] for key in row}
    assert not emitted.intersection(FORK_VALUES)
    run = build_experiment(raw, source="generic dycore control").root.run
    assert run.diff_6th_form == run.upper_wind_limiter_form == "wrf_461"
    assert run.diff_6th_factor2 is None
    assert (run.mp_zero_out, run.mp_zero_out_all) == (0, 0)
    assert run.mp_zero_out_thresh == 1e-8
    assert run.time_step_sound == 4


@pytest.mark.parametrize("name", ("hrrr_wrf.nl", "hrrr_wrf.nl.c18c"))
def test_retained_clone_namelist_selects_the_fork_and_six_substeps(tmp_path, name):
    namelist = tmp_path / name
    namelist.write_bytes((FIXTURE / "hrrr_wrf.nl.c18c").read_bytes())
    text, _ = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    exp = build_experiment(tomllib.loads(text), source="retained dycore request")
    assert_fork(exp.root.run)
    assert exp.root.run.dt == 20
    assert exp.root.run.time_step_sound == 6


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_source_recipe_authors_the_fork_without_a_profile_override(source):
    from woof.domain_wizard import (
        experiment_from_text, render_config, resolved_physics_profile)
    text = render_config(
        name="dycore-source-request", start_time=datetime(2026, 10, 2, 21),
        hours=1, projection={"map_proj": "lambert", "ref_lat": 38.5,
            "ref_lon": -97.5, "truelat1": 38.5, "truelat2": 38.5,
            "stand_lon": -97.5}, dims=[(50, 50)], ratios=(), root_dx_m=3000,
        fetch_hints={"source": source}, case_data=None,
        profile=resolved_physics_profile(source, None))
    exp = experiment_from_text(text, source="bare dycore recipe")
    for domain in exp.domains:
        assert_fork(domain.run)


@pytest.mark.parametrize("relative", (
    "hrrr_native_3km_demo.toml", "hrrr_native_quick_demo.toml",
    "hrrr_prs_3km_demo.toml", "hrrr_prs_demo.toml",
    "recipes/hrrr_configuration_clock.toml",
))
def test_shipped_source_templates_execute_the_fork_forms(relative):
    path = ROOT / "configs" / relative
    shared = tomllib.loads(path.read_text())["shared"]
    for key, value in FORK_VALUES.items():
        assert shared[key] == value, key
    exp = load_experiment(path)
    for domain in exp.domains:
        assert_fork(domain.run)


def test_carried_legacy_limiter_overrides_factor_presence_inference(tmp_path):
    from test_diff6_fork_form import _import
    _import(tmp_path, dynamics=" diff_6th_factor2 = 0.04, 0.03,\n")
    path = tmp_path / "namelist.input"
    path.write_text(with_physics_selector_comment(
        path.read_text(), {"upper_wind_limiter_form": "wrf_461"}))
    text, _ = import_namelists(tmp_path / "namelist.wps", path)
    run = build_experiment(tomllib.loads(text), source="explicit legacy limiter").root.run
    assert run.diff_6th_form == "noaa_wrf39"
    assert run.upper_wind_limiter_form == "wrf_461"


def test_generated_legacy_limiter_survives_the_factor_presence_inference(tmp_path):
    from woof.companion_domains import candidate_wps_text
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    from woof.physics_source_defaults import read_physics_selector_comment
    path = ROOT / "configs/recipes/hrrr_configuration_clock.toml"
    exp = load_experiment(path)
    exp = replace(exp, domains=(replace(
        exp.root, run=replace(exp.root.run, upper_wind_limiter_form="wrf_461")),))
    raw = tomllib.loads(path.read_text())
    raw["shared"]["upper_wind_limiter_form"] = "wrf_461"
    output = tmp_path / "legacy-limiter.toml"
    wps = candidate_wps_text(raw, exp, exp, output)
    write_hrrr_route_inputs(
        output, exp, wps_text=wps,
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    native = tmp_path / "legacy-limiter.namelist.input"
    assert read_physics_selector_comment(native.read_text())["upper_wind_limiter_form"] == "wrf_461"
    text, _ = import_namelists(tmp_path / "legacy-limiter.namelist.wps", native)
    replay = build_experiment(tomllib.loads(text), source="generated legacy limiter")
    assert replay.root.run.diff_6th_form == "noaa_wrf39"
    assert replay.root.run.diff_6th_factor2 == .04
    assert replay.root.run.upper_wind_limiter_form == "wrf_461"


def test_named_source_keeps_explicit_zero_and_default_threshold_choices(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    raw = INPUT_TEXT.replace("&physics\n", "&physics\n mp_zero_out = 0,\n"
                            " mp_zero_out_all = 0,\n mp_zero_out_thresh = 1e-8,\n")
    wps, original = _pair(tmp_path, inp=raw)
    named = tmp_path / "hrrr_wrf.nl"
    named.write_bytes(original.read_bytes())
    text, _ = import_namelists(wps, named)
    doc = tomllib.loads(text)
    assert not {"mp_zero_out", "mp_zero_out_all", "mp_zero_out_thresh"} & set(doc["shared"])
    run = build_experiment(doc, source="named explicit zero control").root.run
    assert run.mp_zero_out == run.mp_zero_out_all == 0
    assert run.mp_zero_out_thresh == 1e-8


def test_carried_legacy_filter_refuses_a_conflicting_fork_only_factor(tmp_path):
    from test_diff6_fork_form import _import
    _import(tmp_path, dynamics=" diff_6th_factor2 = 0.04, 0.03,\n")
    path = tmp_path / "namelist.input"
    path.write_text(with_physics_selector_comment(
        path.read_text(), {"diff_6th_form": "wrf_461"}))
    with pytest.raises(ValueError, match="diff_6th_factor2.*read only"):
        import_namelists(tmp_path / "namelist.wps", path)


def test_generated_route_namelist_carries_standard_fields_and_typed_generations():
    from woof.hrrr_route_inputs import render_namelist_input
    from woof.physics_source_defaults import read_physics_selector_comment
    exp = load_experiment(ROOT / "configs/recipes/hrrr_configuration_clock.toml")
    text = render_namelist_input(exp)
    selectors = read_physics_selector_comment(text)
    assert selectors["diff_6th_form"] == "noaa_wrf39"
    assert selectors["upper_wind_limiter_form"] == "noaa_wrf39"
    # The four numeric settings have actual WRF keys, so the typed
    # comment cannot substitute for the physics/dynamics columns.
    for key in ("mp_zero_out", "mp_zero_out_thresh", "mp_zero_out_all"):
        assert key not in selectors
        assert f" {key}" in text
    assert " diff_6th_factor2" in text


def test_a_fork_filter_can_leave_its_factor_implicit_through_route_roundtrip(tmp_path):
    from woof.companion_domains import candidate_wps_text
    from woof.hrrr_route_inputs import write_hrrr_route_inputs
    path = ROOT / "configs/recipes/hrrr_configuration_clock.toml"
    exp = load_experiment(path)
    exp = replace(exp, domains=(replace(
        exp.root, run=replace(exp.root.run, diff_6th_factor2=None)),))
    raw = tomllib.loads(path.read_text())
    raw["shared"].pop("diff_6th_factor2")
    wps = candidate_wps_text(raw, exp, exp, tmp_path / "implicit.toml")
    written = write_hrrr_route_inputs(
        tmp_path / "implicit.toml", exp, wps_text=wps,
        writer=lambda path, text: path.write_text(text))
    native = next(path for path in written
                  if path.name == "implicit.namelist.input")
    assert "diff_6th_factor2" not in parse_namelist(native)["dynamics"]
