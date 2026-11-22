"""`woof check --met-em DIR` prices card memory from the met_em files alone
(door item 3, second half).

An outside runner sized its met_em forecast by importing the namelist pair
to a TOML and running ``woof check`` on that.  The TOML route asks the
WPS_GEOG tree for the geography a preparation would read, so on a machine
with none staged the check exited 6 and the runner printed "does NOT fit"
for a run that fits.  ``woof run --met-em`` reads no WPS_GEOG: the static
fields are in the met_em files.  The new door resolves the directory with the
run's own resolver and prices it through the same memory check.

CPU only: the card probe is replaced, no device is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import requires_netcdf_bridge

import woof.core.mynn_pbl_scratch as scratch
from woof.core import preflight as pf

CONFIG = (Path(__file__).resolve().parent / "data" / "check_mynn_width"
          / "metem_470x370x49_mynn.toml")
DECLARED = ("--free-gib", "15.25", "--vram-gib", "15.92")


def _args(*tokens):
    from woof.ingest.preflight import register_cli as ingest_register
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    ingest_register(sub)
    return parser.parse_args(["check", *tokens])


@pytest.fixture
def no_card_no_geography(monkeypatch):
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: None)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    # This machine has no WPS_GEOG tree: the CONFIG route's geography leg
    # refuses.
    monkeypatch.setattr("woof.go_cli.geography_refusal",
                        lambda geog_root, *, config=None: "missing x/index")


def _stub_metem(monkeypatch, directory, interval=10800):
    """The runner's 470 x 370 x 49 pair, as the met_em resolver returns it."""
    seen = []

    def resolve(path, **kw):
        seen.append(Path(path))
        return SimpleNamespace(directory=Path(path), toml_text=CONFIG.read_text(encoding="utf-8"),
                               interval_seconds=interval)

    monkeypatch.setattr("woof.metem_door.resolve_metem_run", resolve)
    return seen


def test_a_met_em_check_needs_no_geography_and_prices_as_the_config_route(
        monkeypatch, capsys, tmp_path, no_card_no_geography):
    from woof import cli

    metgrid = tmp_path / "metgrid"
    seen = _stub_metem(monkeypatch, metgrid)
    code = cli.combined_check_main(_args("--met-em", str(metgrid), "--json", *DECLARED))
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert seen == [metgrid]
    # The geography leg is not asked: the met_em files carry the statics.
    assert code == 0, captured.err
    assert report["geography_blocker"] is None
    assert "WPS_GEOG is not read" in captured.err
    assert "fits the 14.75 GiB budget" in report["phase_verdict"]
    assert report["met_em"] == str(metgrid) == report["config"]

    # The same pricing as `woof check CONFIG --vram-gib` on the imported
    # pair at the directory's interval_seconds; only the geography differs.
    config_code = cli.combined_check_main(
        _args(str(CONFIG), "--json", "--forcing-interval-s", "10800", *DECLARED))
    config_report = json.loads(capsys.readouterr().out)
    assert config_code == cli.EXIT_GEOGRAPHY_MISSING
    for key in ("peak_envelope_bytes", "scratch_arena_request_bytes",
                "envelope_budget_bytes", "memory_verdict", "phase_verdict"):
        assert report[key] == config_report[key], key


def test_a_disagreeing_forcing_interval_is_refused(monkeypatch, capsys, tmp_path,
                                                   no_card_no_geography):
    from woof import cli

    _stub_metem(monkeypatch, tmp_path)
    code = cli.combined_check_main(_args("--met-em", str(tmp_path), "--json",
                                         "--forcing-interval-s", "3600", *DECLARED))
    assert code == 2
    assert "interval_seconds = 10800" in capsys.readouterr().err


def test_config_and_met_em_together_are_refused(monkeypatch, capsys, tmp_path):
    from woof import cli

    _stub_metem(monkeypatch, tmp_path)
    code = cli.combined_check_main(_args(str(CONFIG), "--met-em", str(tmp_path), *DECLARED))
    assert code == 2
    assert "not both" in capsys.readouterr().err


def test_a_directory_the_run_would_refuse_is_refused_by_the_check(capsys, tmp_path):
    from woof import cli

    code = cli.combined_check_main(_args("--met-em", str(tmp_path), *DECLARED))
    assert code == 2
    assert "namelist.input" in capsys.readouterr().err


@requires_netcdf_bridge
def test_a_real_metgrid_directory_is_priced_from_its_files(
        monkeypatch, capsys, tmp_path, no_card_no_geography):
    from woof import cli
    from test_lake_state_door import _NAMELIST
    from test_metem_ingest import write_case

    # Two real met_em files (00 and 03 UTC) and the namelist that made them.
    directory = tmp_path / "met"
    directory.mkdir()
    for stamp in ("2021-06-01_00:00:00", "2021-06-01_03:00:00"):
        write_case(directory / f"met_em.d01.{stamp.replace(':', '_')}.nc", time=stamp,
                   global_override={"FLAG_PSFC": 1, "FLAG_SOILHGT": 1, "GRID_ID": 1,
                                    "PARENT_ID": 1, "I_PARENT_START": 1,
                                    "J_PARENT_START": 1, "PARENT_GRID_RATIO": 1})
    (directory / "namelist.input").write_text(_NAMELIST)
    code = cli.combined_check_main(_args("--met-em", str(directory), "--json", *DECLARED))
    captured = capsys.readouterr()
    # A stale static-fields bridge prints its workaround line first.
    report = json.JSONDecoder().raw_decode(captured.out, captured.out.find("{"))[0]
    assert code == 0, captured.err
    assert report["geography_blocker"] is None
    assert report["met_em"] == str(directory)
    assert report["memory_verdict"] != "fail"
