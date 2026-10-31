"""A Hex with Regional nest's maps say ``nest 3 km``, not ``d01 3 km``.

The breakage (WOOF, 2026-10-28, hexreg-nest canary run_9a0796b2e9c8e251): the nest is its own one-domain
WOOF run, so every nest map was headlined ``d01 3 km`` while the site files it as d05.  add-nest now runs
the nest's ``woof sim`` with ``GPUWM_GRID_NAME=nest``; the engine stamps it on the frames and the renderer
prints it in the headline (the domain folder keeps ``d01-3km``).
"""

import argparse
import datetime as dt
import json
import types
from pathlib import Path

import pytest

from woof.hex import add_nest_door as door


def _live_parent(tmp_path):
    run = tmp_path / "hex"
    (run / "forecast").mkdir(parents=True)
    (run / "forecast" / "forecast-start.json").write_text(json.dumps({
        "schema": "gpuwm-hex.forecast-start/1", "grid": "/w/g.nc", "init": "/w/i.nc",
        "start_time": "2025-05-19_12:00:00", "forecast_hours": 12.0,
        "history_every_minutes": 60.0, "history_from_minutes": 0.0}))
    return run


def test_the_nest_sim_runs_with_the_nest_grid_name(tmp_path, monkeypatch):
    run = _live_parent(tmp_path)
    start = dt.datetime(2025, 5, 19, 12)
    monkeypatch.setattr(door, "closed_frames", lambda forecast_dir: [(start, Path("h0.nc"))])
    monkeypatch.setattr(door, "disk_check", lambda *a, **k: {})
    monkeypatch.setattr(door, "nest_config", lambda *a, **k: tmp_path / "nest.toml")
    monkeypatch.setattr(door, "nest_estimate_bytes", lambda config: None)
    monkeypatch.setattr(door, "_engine", lambda name: name)
    monkeypatch.setattr(door, "CardLedger", lambda *a, **k: object())
    monkeypatch.setattr(door, "place_on_card", lambda *a, **k: {"placement": "test", "reservation": None})
    monkeypatch.setattr(door.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    monkeypatch.setenv("WOOF_TEST_KEEP", "kept")
    spawned = []

    class Spawned(Exception):
        pass

    def spawn(argv, log, *, env=None):
        argv = [str(a) for a in argv]
        spawned.append((argv, env))
        out = Path(log).parent.parent
        if "woof.local_posting" in argv:
            (out / "fetch" / "posting").mkdir(parents=True, exist_ok=True)
            (out / "fetch" / "posting" / "schedule.json").write_text("{}")
        if argv[:2] == ["woof", "prep"]:
            (out / "prepared" / "boundary-stream").mkdir(parents=True, exist_ok=True)
            (out / "prepared" / "boundary-stream" / "head.json").write_text("{}")
        if argv[:2] == ["woof", "sim"]:
            raise Spawned
        return types.SimpleNamespace(pid=1, poll=lambda: None, returncode=None)

    monkeypatch.setattr(door, "_spawn", spawn)
    parser = argparse.ArgumentParser()
    door.add_add_nest_parser(parser.add_subparsers())
    args = parser.parse_args(["add-nest", str(run), "--lat", "35", "--lon", "-97", "--wait-minutes", "0",
                              "--out", str(tmp_path / "nests"), "--start", "now", "--hours", "2"])
    with pytest.raises(Spawned):
        door.add_nest(args)
    sim_argv, sim_env = spawned[-1]
    assert sim_argv[:2] == ["woof", "sim"]
    assert sim_env["GPUWM_GRID_NAME"] == "nest"
    assert sim_env["WOOF_TEST_KEEP"] == "kept", "the sim keeps the door's own environment"
    # Only the nest's sim is named: export, posting and prep run on the door's environment.
    assert all(env is None for argv, env in spawned[:-1])
