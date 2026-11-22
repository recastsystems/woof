"""HEX-PHYSICS-PARITY run group 2 #6: the moving ensemble on the MPAS suite route.

Each test is one of the defects the real run showed (2026-10-24).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import moving_members, moving_run
from woof.hex.config import ConfigurationRefusal, parse_fortran_namelist
from woof.hex.mpasphys.forecast_v842 import override_literal


def test_a_bare_character_override_reads_as_on_the_arwen_route():
    # the packaged roster wrote config_microp_scheme=mp_thompson; the parser refused every member
    value = override_literal("config_microp_scheme", "mp_thompson")
    assert parse_fortran_namelist(f"&physics\nconfig_microp_scheme={value}\n/")["physics"] == {
        "config_microp_scheme": "mp_thompson"}
    assert override_literal("config_microp_scheme", "'mp_thompson'") == "'mp_thompson'"
    assert override_literal("config_microp_re", "true") == "true"
    assert override_literal("config_gfconv_closure_deep", "0") == "0"
    assert override_literal("config_radtlw_interval", "00:15:00") == "'00:15:00'"


def test_a_bare_word_for_a_numeric_option_is_still_refused_by_name():
    value = override_literal("config_dt", "abc")
    with pytest.raises(ConfigurationRefusal):
        parse_fortran_namelist(f"&nhyd_model\nconfig_dt={value}\n/")


def test_the_default_roster_takes_the_route_s_own_physics_rows():
    default = json.loads((Path(moving_members.__file__).with_name("data") / "moving-ensemble.v1.json").read_text())
    arwen = moving_members.roster_physics_rows(default, None)
    assert [row["id"] for row in arwen] == ["wsm6", "thompson"]
    mpas = moving_members.roster_physics_rows(default, "convection_permitting")
    assert all("mp_wsm6" not in " ".join(row["forecast_flags"]) for row in mpas)
    assert len({json.dumps(row["forecast_flags"]) for row in mpas}) == 2
    assert moving_members.roster_physics_rows(default, "rrfs_v2_dev")[0]["forecast_flags"] == []


def test_a_door_refusal_before_the_first_step_is_not_retried(tmp_path):
    seg = tmp_path / "segment-000"
    (seg / "control").mkdir(parents=True)
    (seg / "fc").mkdir()
    (seg / "fc" / "timeline.jsonl").write_text('{"stage": "prep.door", "phase": "begin"}\n')
    (seg / "forecast.log").write_text("noise\nwoof hex: config_microp_scheme='mp_wsm6' is refused: ...\n")
    assert moving_run.door_refusal(seg, 2).startswith("config_microp_scheme")
    assert moving_run.door_refusal(seg, -11) is None
    (seg / "fc" / "timeline.jsonl").write_text('{"stage": "forecast.first_step", "phase": "begin"}\n')
    assert moving_run.door_refusal(seg, 2) is None


def test_a_recover_segment_holding_at_a_decided_frame_gets_its_verdict(tmp_path):
    controller = object.__new__(moving_run.Controller)
    controller.state = {"frames": [{"model_seconds": 5400.0}]}
    control = tmp_path / "control"
    control.mkdir()
    (control / "await-verdict.json").write_text("{}\n")
    assert controller.on_frame(5400.0, "history", control) is None
    assert json.loads((control / "verdict-00005400.json").read_text()) == {"stop": False}


def test_recovery_forgets_the_frames_after_its_carrier():
    controller = object.__new__(moving_run.Controller)
    controller.state = {"frames": [{"model_seconds": s} for s in (1800.0, 3600.0, 5400.0, 7200.0)]}
    controller.session = SimpleNamespace(fixes=[SimpleNamespace(seconds=5400.0)],
                                         decisions=[SimpleNamespace(seconds=5400.0)])
    assert controller.forget_after(3600.0) == 2
    assert [f["model_seconds"] for f in controller.state["frames"]] == [1800.0, 3600.0]


def test_the_ensemble_names_a_member_s_own_failure(tmp_path):
    log = tmp_path / "controller-launch.log"
    log.write_text("...\nsegment 0: config_microp_scheme='mp_wsm6' is refused: only logical ...\n")
    row = dict(status="failed", log=str(log), returncode=1)
    assert moving_members.member_failure(row).startswith("segment 0: config_microp_scheme")
