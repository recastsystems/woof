"""``export --follow``: each frame is exported the moment the forecast lists it.

The breakage: a WOOF domain driven by a hex parent waited for the whole parent
forecast, because the export only read a finished run.  Here the converter is
replaced by a stand-in that writes one wrfout per history, so the test checks
the follow loop itself: order, the atomic filing, and the ready list.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from woof.hex import export_door, render_door


def _fake_convert(convert_exe, *, histories, scratch, simulation_start, **_kwargs):
    """One wrfout per history, named and dated by the history's valid time, as rw_mpas_convert does."""
    frames = Path(scratch) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rows = []
    for history in histories:
        valid = export_door.history_valid_time(Path(history))
        out = frames / f"wrfout_d01_{valid:%Y-%m-%d_%H_%M_%S}"
        out.write_bytes(b"frame " + out.name.encode())
        rows.append({"output": str(out), "output_sha256": "x" * 64, "valid_time": f"{valid:%Y-%m-%dT%H:%M:%S}"})
    return {"frames": rows}, ["fake"]


def test_follow_exports_each_listed_frame_and_lists_it_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert)
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    run = tmp_path / "run"
    (run / "forecast").mkdir(parents=True)
    stamps = ["2025-05-19_12_00_00", "2025-05-19_13_00_00", "2025-05-19_14_00_00"]

    def forecast():
        for stamp in stamps:
            time.sleep(0.05)
            history = run / "forecast" / f"history.{stamp}.nc"
            history.write_bytes(b"h")
            with (run / "forecast" / "history-ready.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": 0, "path": str(history)}) + "\n")

    writer = threading.Thread(target=forecast)
    writer.start()
    args = argparse.Namespace(
        forecast_dir=str(run), mesh="grid.nc", init="init.nc", simulation_start="2025-05-19_12:00:00",
        follow=True, follow_frames=3, follow_timeout_minutes=1.0, follow_poll_seconds=0.01,
        out=str(tmp_path / "export"), scratch=None, window="region-latlon:35.5,-97.5,1300,1300",
        remap="product", view="nest", field_set="full", nc_format="cdf5", model_label="hex",
        keep_scratch=False, overwrite=False, convert_exe=None)
    assert export_door.run_export(args) == 0
    writer.join()
    rows = [json.loads(line) for line in
            (tmp_path / "export" / export_door.READY_NAME).read_text(encoding="utf-8").splitlines()]
    assert [Path(row["frame"]).name for row in rows] == [f"wrfout_d01_{stamp}" for stamp in stamps]
    for row in rows:
        assert Path(row["frame"]).is_file() and Path(row["frame"]).parent.parent.name == "nest"
    receipt = json.loads((tmp_path / "export" / export_door.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["mode"] == "follow" and len(receipt["frames"]) == 3


def test_follow_takes_frames_by_durable_sidecar_when_no_ready_list(tmp_path, monkeypatch):
    # The variable-resolution door's executor writes <frame>.durable.json after each frame and no
    # history-ready.jsonl (2026-10-23: --follow beside it exported nothing).
    # the real history name carries dots in its time (cuda-history.2025-05-19_12.00.00.nc)
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert)
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    run = tmp_path / "run"
    run.mkdir()
    stamps = ["2025-05-19_12.00.00", "2025-05-19_13.00.00"]
    for stamp in stamps:
        (run / f"cuda-history.{stamp}.nc").write_bytes(b"hh")
    # The second frame's sidecar is not whole yet: it must not be taken, and nothing after it either.
    (run / f"cuda-history.{stamps[0]}.nc.durable.json").write_text(json.dumps({"bytes": 2}))
    (run / f"cuda-history.{stamps[1]}.nc.durable.json").write_text(json.dumps({"bytes": 9}))
    assert export_door._durable_frames(run) == [run / f"cuda-history.{stamps[0]}.nc"]

    def finish():
        time.sleep(0.05)
        (run / f"cuda-history.{stamps[1]}.nc.durable.json").write_text(json.dumps({"bytes": 2}))

    writer = threading.Thread(target=finish)
    writer.start()
    args = argparse.Namespace(
        forecast_dir=str(run), mesh="grid.nc", init="init.nc", simulation_start="2025-05-19_12:00:00",
        follow=True, follow_frames=2, follow_timeout_minutes=1.0, follow_poll_seconds=0.01,
        out=str(tmp_path / "export"), scratch=None, window="region-latlon:35.5,-97.5,1300,1300",
        remap="product", view="nest", field_set="full", nc_format="cdf5", model_label="hex",
        keep_scratch=False, overwrite=False, convert_exe=None)
    assert export_door.run_export(args) == 0
    writer.join()
    rows = [json.loads(line) for line in
            (tmp_path / "export" / export_door.READY_NAME).read_text(encoding="utf-8").splitlines()]
    assert [Path(row["history"]).name for row in rows] == [f"cuda-history.{s}.nc" for s in stamps]
