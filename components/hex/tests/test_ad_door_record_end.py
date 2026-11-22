"""The door records one step past its window.

THE BREAKAGE THIS HOLDS SHUT: a door-recorded 3-step plan named boundaries 1 to 3 only, so the window-end field
addresses (boundary 4, what the strict self dot weighs and what an AD_DOT_DEAD list is built from) were missing and
sixty's mkdead.py stopped with KeyError '4' on the 480 km user case (2026-11-10 03:13Z).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from woof.hex import ad_doors as d

FAKE = r'''
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); steps = int(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
(out / "plan.json").write_text(json.dumps({"blocks": {f"step_{s:02d}": [] for s in range(1, steps + 1)}}))
(out / "alloc-map.json").write_text(json.dumps({"boundaries": {str(b): {} for b in range(1, steps + 1)}}))
(out / "record-receipt.json").write_text("{}")
'''


def _setup(tmp_path: Path, monkeypatch, steps: int | None):
    (tmp_path / "grid.nc").write_text("g")
    (tmp_path / "oro.nc").write_text("o")
    (tmp_path / "init.nc").write_text("i")
    fake = tmp_path / "fake_recorder.py"
    fake.write_text(FAKE)
    monkeypatch.setenv(d.RECORDER_ENV, f"{sys.executable} {fake} {{out}} {{steps}}")
    row = {"mesh": "x1.2562", "suite": "hrrrv5", "dt_s": 900.0, "default_steps": 3,
           "assets": {"grid": "grid.nc", "oro": "oro.nc"}}
    args = argparse.Namespace(grid=None, oro=None, steps=steps, out=str(tmp_path / "run"), init=str(tmp_path / "init.nc"),
                              card=0, runner=None)
    return args, row


def test_door_records_the_window_end_boundary(tmp_path, monkeypatch):
    args, row = _setup(tmp_path, monkeypatch, None)
    plan = d.record_plan(args, tmp_path, row)
    assert d.recorded_end_boundary(plan) == 4   # 3-step window: boundaries 1..4
    assert d.record_plan(args, tmp_path, row) == plan   # a finished recording that reaches the end is reused


def test_a_short_recording_is_refused_not_reused(tmp_path, monkeypatch):
    args, row = _setup(tmp_path, monkeypatch, 2)
    d.record_plan(args, tmp_path, row)   # boundaries 1..3
    args.steps = 3
    with pytest.raises(d.AdDoorError, match="ends before boundary 4"):
        d.record_plan(args, tmp_path, row)


def test_the_door_records_in_strict_numerics(tmp_path, monkeypatch):
    """THE BREAKAGE: the door left the recorder at its fast default and the 120 km pair could not seal (box-d d120-d)."""
    args, row = _setup(tmp_path, monkeypatch, None)
    monkeypatch.delenv(d.RECORDER_ENV)
    runner = tmp_path / "ad-binder"
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o755)
    args.runner = runner
    seen = {}

    class Done:
        returncode = 0

    def run(argv, **kw):
        seen["argv"] = argv
        return Done()

    monkeypatch.setattr(d.subprocess, "run", run)
    d.record_plan(args, tmp_path, row)
    argv = seen["argv"]
    assert argv[argv.index("--numerics") + 1] == "strict"
