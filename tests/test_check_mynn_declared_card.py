"""`woof check` prices the MYNN column width the run walks (door item 3).

An outside runner checked a 470 x 370 x 49 MYNN met_em forecast with
``woof check from_namelist.toml --json --free-gib 15.25 --vram-gib 15.92``
and read 17.86 GiB "EXCEEDS the 14.75 GiB budget", exit 1, for a run that
then ran on that 16 GB card.  The check priced MYNN's column workspace at
the width THIS process would walk: 98,304 columns (4.79 GiB) when it read
no card, whatever the local card allowed when it read one.  So one file and
one declared card gave 17.86 GiB on one machine and 13.12 on another.

Two rules, both the run's own:

* a declared card prices the width that card walks (the target fit
  ``woof domain`` already uses), whatever card this machine holds;
* where the walked width does not fit and the narrowest does, the report
  prices the narrowest and says so, as the resident door and ``woof go``
  do (A3c): the run pins the widest width that fits.

CPU only: the card probe is replaced, no device is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import woof.core.mynn_pbl_scratch as scratch
from woof.core import preflight as pf

CONFIG = (Path(__file__).resolve().parent / "data" / "check_mynn_width"
          / "metem_470x370x49_mynn.toml")
GIB = 1024 ** 3

#: The desktop card a second machine would read (RTX 3080, 10 GB).
OTHER_CARD = {"sm_count": 68, "max_threads_per_sm": 1536,
              "registers_per_sm": 65536, "warp_size": 32,
              "free_bytes": 9_538_895_872, "total_bytes": 10_736_893_952,
              "device_name": "NVIDIA GeForce RTX 3080"}


def _check(monkeypatch, capsys, card, *flags):
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: card)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(["check", str(CONFIG), "--json",
                              "--forcing-interval-s", "10800", *flags])
    code = pf.check_main(args)
    return code, json.loads(capsys.readouterr().out)


def test_a_declared_card_is_priced_at_its_own_width_on_any_machine(
        monkeypatch, capsys):
    declared = ("--free-gib", "15.25", "--vram-gib", "15.92")
    code_none, none = _check(monkeypatch, capsys, None, *declared)
    code_other, other = _check(monkeypatch, capsys, dict(OTHER_CARD), *declared)
    # The machine running the check no longer moves the figure.
    assert none["peak_envelope_bytes"] == other["peak_envelope_bytes"]
    assert none["scratch_arena_request_bytes"] == other["scratch_arena_request_bytes"]
    # ...and the figure is the one the 16 GB card walks: it fits.  It is
    # 14.64 GiB since D2 (2026-11-12), where it was 13.34: the legacy RRTMG
    # call's own rule takes two waves with 15.25 GiB free, and the price
    # carries that width because it still fits the 14.75 GiB budget.
    assert none["legacy_radiation_widths"] == [4096]
    assert none["peak_envelope_bytes"] < 14.75 * GIB
    assert "fits the 14.75 GiB budget" in none["phase_verdict"]
    assert none["memory_verdict"] != "fail"
    assert code_none == code_other == 0
    assert "mynn_column_width" not in none


def test_the_narrowest_width_is_priced_when_the_walked_one_does_not_fit(
        monkeypatch, capsys):
    # No card capacity declared and no card read: the walked width is the
    # 98,304-column cap, which this budget does not hold.
    code, report = _check(monkeypatch, capsys, None, "--free-gib", "15.25")
    width = report["mynn_column_width"]
    assert width["walked_width_columns"] == scratch.MYNN_PBL_COLUMN_CHUNK_DEFAULT
    assert width["priced_width_columns"] == scratch.MYNN_PBL_COLUMN_CHUNK_MINIMUM
    assert pf.MYNN_NARROWED_NOTE in report["phase_verdict"]
    assert "fits the 14.75 GiB budget" in report["phase_verdict"]
    assert report["memory_verdict"] != "fail"
    assert code == 0


def test_a_budget_the_narrowest_width_cannot_hold_is_still_refused(
        monkeypatch, capsys):
    code, report = _check(monkeypatch, capsys, None, "--free-gib", "8")
    assert "mynn_column_width" not in report
    assert "EXCEEDS" in report["phase_verdict"]
    assert code != 0


def test_a_failed_allocation_leg_alone_does_not_narrow_the_width(
        monkeypatch, capsys):
    """The run narrows only when the ENVELOPE is over (``admit_resident_road``,
    ``woof go``'s gate).  The shipped HRRR recipe on a declared 100 GB card
    with 98.5 GiB free: the envelope fits at the walked width and only the
    allocation-estimate leg fails.  The run walks the full width there, so the
    check must price that width and keep the leg's failure (item 3 review:
    narrowing on the leg priced 8,192 columns and exited 0)."""
    monkeypatch.setenv("WOOF_CHAINED_PREP", "0")
    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: None)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    recipe = (Path(__file__).resolve().parents[1] / "configs" / "recipes"
              / "conus_hrrr_configuration.toml")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(["check", str(recipe), "--json", "--devices", "1",
                              "--free-gib", "98.5", "--vram-gib", "100"])
    code = pf.check_main(args)
    out = capsys.readouterr().out
    report = json.JSONDecoder().raw_decode(out, out.find("{"))[0]
    assert report["peak_envelope_bytes"] <= report["envelope_budget_bytes"]
    # The leg is named for the platform budget (wddm on Windows, vram elsewhere).
    leg = [v for k, v in report["gates"].items() if k.startswith("alloc_estimate_le_")]
    assert leg == [False], report["gates"]
    assert "mynn_column_width" not in report
    assert pf.MYNN_NARROWED_NOTE not in report["phase_verdict"]
    assert code == 1


def test_the_runner_command_reports_fit_and_geography_apart(
        monkeypatch, capsys):
    """The outside runner's whole ``woof check`` command, on a machine with
    no geography staged: the memory verdict says it fits, and the exit 6
    is named in the report as the geography leg, not as memory."""
    from woof import cli

    monkeypatch.setattr(scratch, "probe_mynn_card", lambda device=None: None)
    monkeypatch.setattr(scratch, "_RESOLVED", {})
    monkeypatch.setattr("woof.go_cli.geography_refusal",
                        lambda geog_root, *, config=None: "missing x/index")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    args = parser.parse_args(["check", str(CONFIG), "--json",
                              "--free-gib", "15.25", "--vram-gib", "15.92"])
    args.ingest_preflight_handler = lambda a: 0
    code = cli.combined_check_main(args)
    report = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_GEOGRAPHY_MISSING
    assert "fits the 14.75 GiB budget" in report["phase_verdict"]
    assert report["memory_verdict"] != "fail"
    assert report["geography_blocker"] == "missing x/index"
