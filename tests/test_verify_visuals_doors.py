"""`--no-verify-visuals` has one spelling and one meaning on woof go, sim and run.

THE BREAKAGE THIS PREVENTS (2026-11-11, an outside runner on 2.9.13): the CPU
verification step a finished forecast starts outlived the card lease, and only
`woof go` had the switch; `woof sim` and `woof run` had nothing but the
WOOF_VERIFY_VISUALS variable, which that caller never found.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from woof.cli import build_parser, main as cli_main


@pytest.mark.parametrize("argv", [
    ["sim", "prepared", "--experiment-config", "e.toml", "--outdir", "o"],
    ["run", "--met-em", "metgrid"],
    ["go", "case.toml"],
])
def test_every_forecast_door_parses_the_switch(argv):
    parser = build_parser()
    assert parser.parse_args(argv).verify_visuals is None
    assert parser.parse_args([*argv, "--no-verify-visuals"]).verify_visuals is False


def test_run_carries_the_switch_to_its_forecast_and_gives_the_variable_back(tmp_path, monkeypatch):
    from woof import capabilities, metem_forecast, provenance_gate

    directory = tmp_path / "metgrid"
    directory.mkdir()
    (directory / "namelist.input").write_text("&physics bl_pbl_physics=16 /\n")
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a: None)
    monkeypatch.setattr(provenance_gate, "announce", lambda *a: None)
    monkeypatch.setenv("WOOF_VERIFY_VISUALS", "1")
    seen = []
    monkeypatch.setattr(metem_forecast, "run_metem_forecast",
                        lambda *a, **k: seen.append(os.environ.get("WOOF_VERIFY_VISUALS")) or 0)
    argv = ["run", "--met-em", str(directory), "--outdir", str(tmp_path / "out")]
    assert cli_main(argv) == 0
    assert cli_main([*argv, "--no-verify-visuals"]) == 0
    assert seen == ["1", "0"]
    assert os.environ["WOOF_VERIFY_VISUALS"] == "1"


def _bundle(root: Path) -> Path:
    from woof.prepared_single_domain_forecast import _PROOF_SCHEMA

    root.mkdir(parents=True, exist_ok=True)
    (root / "proof.json").write_text(json.dumps({
        "schema": _PROOF_SCHEMA["gfs"], "status": "READY",
        "initial_valid_time": "2026-05-17T18:00:00",
        "input_manifest_sha256": "11" * 32,
        "prepared_cache": {"content_sha256": "22" * 32},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("flag, expected", [([], "1"), (["--no-verify-visuals"], "0")])
def test_sim_carries_the_switch_to_its_in_process_runner(tmp_path, monkeypatch, capsys, flag, expected):
    from woof import capabilities, prepared_single_domain_forecast

    # The runner is stubbed, so the CuPy door check is too: without it the
    # test failed on a CPU box with no cupy, before the switch.
    monkeypatch.setattr(capabilities, "require_for_command", lambda _: None)
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'door8'\n", encoding="utf-8")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    monkeypatch.setenv("WOOF_VERIFY_VISUALS", "1")
    seen = []
    monkeypatch.setattr(prepared_single_domain_forecast, "main",
                        lambda argv: seen.append(os.environ.get("WOOF_VERIFY_VISUALS")) or 0)
    assert cli_main(["sim", str(_bundle(tmp_path / "prepared")), "--experiment-config", str(config),
                     "--wps-namelist", str(wps), "--outdir", str(tmp_path / "runs"), *flag]) == 0
    assert seen == [expected]
    assert os.environ["WOOF_VERIFY_VISUALS"] == "1"
    note = capsys.readouterr().out
    assert ("--no-verify-visuals skips it" in note) == (expected == "1")
