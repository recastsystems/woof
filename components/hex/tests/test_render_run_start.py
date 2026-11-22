"""Several history frames and no --simulation-start: the render door reads the run start the histories carry.

THE BREAKAGE THIS PREVENTS (0.6.1): `woof hex render --history RUN/history.*.nc` without `--simulation-start`
filed every frame as lead 0, so each frame's pictures overwrote the one before, and the door still ended
`failed=0` with one frame's files on disk.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

netCDF4 = pytest.importorskip("netCDF4")

from woof.hex import render_door  # noqa: E402


def _history(folder: Path, valid: str, start: str | None) -> Path:
    path = folder / f"history.{valid}.nc"
    with netCDF4.Dataset(str(path), "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("StrLen", 64)
        xtime = dataset.createVariable("xtime", "S1", ("Time", "StrLen"))
        xtime[0, :] = netCDF4.stringtochar(__import__("numpy").array([valid.replace(".", ":")], dtype="S64"))
        if start is not None:
            dataset.setncattr("config_start_time", start)
    return path


def test_the_histories_name_one_start_and_the_door_takes_it(tmp_path):
    frames = [_history(tmp_path, f"2026-08-12_0{h}.00.00", "2026-08-12_06:00:00") for h in (6, 7, 8)]
    assert render_door.histories_run_start(frames) == "2026-08-12_06:00:00"


def test_two_starts_are_refused_by_name(tmp_path):
    frames = [_history(tmp_path, "2026-08-12_07.00.00", "2026-08-12_06:00:00"),
              _history(tmp_path, "2026-08-13_07.00.00", "2026-08-13_06:00:00")]
    with pytest.raises(render_door.RenderDoorError, match="2 run starts"):
        render_door.histories_run_start(frames)


def test_a_history_with_no_start_is_refused_with_the_flag(tmp_path):
    frames = [_history(tmp_path, "2026-08-12_07.00.00", "2026-08-12_06:00:00"),
              _history(tmp_path, "2026-08-12_08.00.00", None)]
    with pytest.raises(render_door.RenderDoorError, match="--simulation-start"):
        render_door.histories_run_start(frames)


def test_one_frame_takes_the_start_it_carries(tmp_path):
    """0.6.2: one +3 h frame was drawn as "Init 11/19 15Z | F000" for a run started at 12Z."""
    frame = _history(tmp_path, "2026-11-19_15.00.00", "2026-11-19_12:00:00")
    assert render_door.door_run_start([frame], None) == "2026-11-19_12:00:00"


def test_one_frame_with_no_start_stays_at_lead_zero(tmp_path):
    frame = _history(tmp_path, "2026-11-19_15.00.00", None)
    assert render_door.door_run_start([frame], None) is None


def test_a_given_start_wins_and_several_frames_still_refuse_two_starts(tmp_path):
    frames = [_history(tmp_path, "2026-08-12_07.00.00", "2026-08-12_06:00:00"),
              _history(tmp_path, "2026-08-13_07.00.00", "2026-08-13_06:00:00")]
    assert render_door.door_run_start(frames, "2026-08-12_00:00:00") == "2026-08-12_00:00:00"
    with pytest.raises(render_door.RenderDoorError, match="2 run starts"):
        render_door.door_run_start(frames, None)


def test_the_door_asks_door_run_start():
    source = Path(render_door.__file__).read_text(encoding="utf-8")
    assert "args.simulation_start = door_run_start(histories, None)" in source
