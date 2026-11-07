"""A resume from a checkpoint after a cut lease (IFS-9KM-DEFAULT-2026-11-04).

1. Frames the cut leg wrote past its checkpoint are regenerated, so the resume sets them aside instead of
   refusing them (the never_modify history writer raised FileExistsError on each).
2. The resumed segments default maps keep the runs init: an 18Z run restarted from f015 read
   "Init 2026-11-03 09Z F009".
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from woof.hex import forecast_door
from woof.hex.mpasphys.checkpoint_v842 import set_aside_frames_after


def test_frames_after_the_checkpoint_are_set_aside_and_the_rest_kept(tmp_path: Path):
    names = ["history.2026-11-02_18.00.00.nc", "history.2026-11-03_09.00.00.nc",
             "history.2026-11-03_10.00.00.nc", "diag.2026-11-03_11.00.00.nc", "static.nc"]
    for name in names:
        (tmp_path / name).write_text(name)
    (tmp_path / "history.2026-11-03_10.00.00.nc.pre-restart").write_text("older resume")
    moved = set_aside_frames_after(tmp_path, "2026-11-03_09:00:00")
    assert moved == ["diag.2026-11-03_11.00.00.nc", "history.2026-11-03_10.00.00.nc"]
    kept = sorted(p.name for p in tmp_path.glob("*.nc"))
    assert kept == ["history.2026-11-02_18.00.00.nc", "history.2026-11-03_09.00.00.nc", "static.nc"]
    assert (tmp_path / "history.2026-11-03_10.00.00.nc.pre-restart").read_text() == "history.2026-11-03_10.00.00.nc"
    assert set_aside_frames_after(tmp_path, "2026-11-03_09:00:00") == []


def test_a_resumed_segment_labels_its_maps_with_the_run_init(tmp_path: Path, monkeypatch):
    out = tmp_path / "forecast"
    out.mkdir()
    (out / "history.2026-11-03_18.00.00.nc").write_text("")
    (out / "native-physics-request.json").write_text(json.dumps({
        "start": "2026-11-03_09:00:00", "initial_time": "2026-11-02_18:00:00",
        "grid": str(tmp_path / "grid.nc"), "init": str(tmp_path / "init.nc")}))
    seen = {}
    monkeypatch.setattr(forecast_door, "run_default_products",
                        lambda request, history, arguments: seen.update(start=request.start_time))
    monkeypatch.setattr(forecast_door, "moving_launch_switches", lambda: (), raising=False)
    forecast_door.native_default_products(Namespace(out=out, preflight=False, no_products=False))
    assert seen["start"] == "2026-11-02_18:00:00"
