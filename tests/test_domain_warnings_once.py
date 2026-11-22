"""`woof domain` says each warning about the file it writes once.

THE BREAKAGE THIS PREVENTS (3.0.0 artefact check, 2026-11-10): `woof domain
--card 32gb --ladder 12-3` on a Noah-MP suite printed "Noah-MP at N columns is
beyond the 360000-column measured width" once per size its fit loop tried and
again for every load of the written file: 27 lines then, 42 on 3.0.2, for one
configuration.
"""
from __future__ import annotations

import argparse

import pytest

from woof import domain_wizard as wizard
from woof import explain, physics_compat


def test_a_sentence_raised_three_times_prints_once_inside_the_scope(capsys):
    seen = []
    explain.add_warning_observer(seen.append)
    try:
        with explain.each_warning_once():
            for _ in range(3):
                explain.warn("the same sentence")
            explain.warn("another sentence")
        explain.warn("the same sentence")
    finally:
        explain.remove_warning_observer(seen.append)
    err = capsys.readouterr().err.splitlines()
    assert err == ["warning: the same sentence", "warning: another sentence",
                   "warning: the same sentence"]
    assert len(seen) == 5


def test_a_muted_warning_spends_no_slot_of_the_scope(capsys):
    with explain.each_warning_once():
        with explain.muted_warnings():
            explain.warn("about the written file")
        explain.warn("about the written file")
    assert capsys.readouterr().err.splitlines() == ["warning: about the written file"]


def test_the_noahmp_width_warning_prints_once_for_the_written_file(tmp_path, monkeypatch, capsys):
    # A 100 x 100 domain trips the warning once the measured width is set
    # below it, so the door is exercised without a production-width fit.
    monkeypatch.setattr(physics_compat, "NOAHMP_MEASURED_COLUMN_CEILING", 1000)
    monkeypatch.delenv(physics_compat.NOAHMP_EXPERT_COLUMN_BUDGET_ENV, raising=False)
    parser = argparse.ArgumentParser()
    wizard.register_cli(parser.add_subparsers())
    out = tmp_path / "forecast.toml"
    args = parser.parse_args([
        "domain", "--source", "gfs", "--cycle", "2026-09-05T00", "--hours", "6",
        "--root-dx", "3", "--point", "35.3,-97.5", "--point-extent-km", "300",
        "--vram-gib", "16", "--tiles", "off", "--out", str(out),
        "--physics-profile", "wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1"])
    args.explain = False
    assert wizard.domain_main(args) == 0
    lines = [line for line in capsys.readouterr().err.splitlines()
             if "measured width" in line]
    assert len(lines) == 1, lines
    assert "Noah-MP at" in lines[0]
