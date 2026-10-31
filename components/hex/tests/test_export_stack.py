"""The export of a variable-resolution mesh writes each region at the detail the model has there.

THE BREAKAGE (an Arizona 96 -> 3 km zoom globe, 36 h, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30): the
site's ``export --window global`` laid the whole sphere down at the refined core's spacing (9 million points at
9.4 km) and wrote 252 GB, after the forecast, on the CPU while the cards sat idle. Here the mesh is a synthetic zoom
globe (a 3 km core, rings of 6, 12, 24 and 48 km, a 96 km outer globe) and the converter is a stand-in, so the tests
check the plan (which grids), the filing (``wrfout_dNN``), the follower beside the forecast, and the export step
after it taking the follower's frames instead of converting them again.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.hex import export_door, render_door


def _zoom_globe(centre=(34.06, -111.12)):
    """Cells of a 96 -> 3 km zoom globe around ``centre``: spacing by distance, as the cap-family spec makes it."""
    rng = np.random.default_rng(7)
    rings = [(783.0, 3.0, 233_000), (940.0, 6.0, 21_000), (1239.0, 12.0, 13_000), (1919.0, 24.0, 10_000),
             (3579.0, 48.0, 10_000)]
    lat, lon, spacing = [], [], []
    inner = 0.0
    clat, clon = map(math.radians, centre)
    for outer, km, n in rings:
        d = np.sqrt(rng.uniform(inner ** 2, outer ** 2, n)) / 6371.229
        az = rng.uniform(0, 2 * math.pi, n)
        la = np.arcsin(np.sin(clat) * np.cos(d) + np.cos(clat) * np.sin(d) * np.cos(az))
        lo = clon + np.arctan2(np.sin(az) * np.sin(d) * np.cos(clat), np.cos(d) - np.sin(clat) * np.sin(la))
        lat.append(np.degrees(la)), lon.append(np.degrees(lo)), spacing.append(np.full(n, km))
        inner = outer
    # the outer globe: 96 km cells everywhere else
    n = 60_000
    z = rng.uniform(-1, 1, n)
    la, lo = np.degrees(np.arcsin(z)), rng.uniform(-180, 180, n)
    lat.append(la), lon.append(lo), spacing.append(np.full(n, 96.0))
    lat, lon = np.concatenate(lat), (np.concatenate(lon) + 180.0) % 360.0 - 180.0
    return lat, lon, np.concatenate(spacing)


def test_a_zoom_globe_exports_a_stack_each_band_at_its_own_spacing(monkeypatch):
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    plan = export_door.export_plan("global", Path("grid.nc"))
    assert [e["view"] for e in plan] == ["global", "ring-48km", "ring-24km", "ring-12km", "ring-6km", "mesh"]
    assert [e["domain"] for e in plan] == [1, 2, 3, 4, 5, 6]
    assert plan[0]["window"] == "latlon:-90,90,-180,179.25,0.75"   # 96 km cells: 83 km, never coarser
    degrees = [float(e["window"].rsplit(",", 1)[1]) for e in plan[1:]]
    assert degrees == [0.4, 0.2, 0.1, 0.05, 0.025]                  # the 3 km core at 0.025 degrees (2.8 km)
    for entry in plan[1:]:
        s, n, w, e, step = map(float, entry["window"][len("latlon:"):].split(","))
        assert s < 34.06 < n and w < -111.12 < e and step * export_door.KM_PER_DEGREE <= entry["spacing_km"]
        assert max(entry["points"]) <= export_door.MAX_POINTS_PER_SIDE
    # The core box holds the 783 km core and little more: about 1,600 km a side.
    s, n, w, e, _step = map(float, plan[-1]["window"][len("latlon:"):].split(","))
    assert 13.5 < n - s < 15.0 and 16.0 < e - w < 20.0
    # Far fewer points than the 9 million of the whole sphere at the core's spacing.
    assert sum(e["points"][0] * e["points"][1] for e in plan) < 1_100_000


def test_the_core_is_a_lat_lon_box_so_it_carries_the_cf_axes(monkeypatch):
    """The independent check (RTX 5090, 2026-10-30): the core went on the converter's Lambert `mesh` window, which has
    no 1-D axes, so wrfout_d06 lacked lat, lon, Time, bottom_top, bottom_top_stag and soil_layers_stag. The
    converter writes them on every `latlon:` window, so the core is one. A polar core, which no lat-lon box holds at
    its spacing inside the converter's 2000 points a side, keeps the Lambert grid rather than being coarsened."""
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    core = export_door.export_plan("global", Path("grid.nc"))[-1]
    assert core["view"] == "mesh" and core["window"].startswith("latlon:") and core["grid_km"] < core["spacing_km"]
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe(centre=(84.0, 20.0)))
    core = export_door.export_plan("global", Path("grid.nc"))[-1]
    assert core["view"] == "mesh" and core["window"] == "mesh" and core["grid_km"] is None
    assert "Lambert" in core["why"]


def test_a_uniform_mesh_and_other_windows_keep_one_grid(monkeypatch):
    lat, lon, _ = _zoom_globe()
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: (lat, lon, np.full(lat.shape, 60.0)))
    assert export_door.export_plan("global", Path("grid.nc")) == [
        {"domain": 1, "view": "global", "window": "global-mesh", "spacing_km": None, "why": None}]
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    assert [e["window"] for e in export_door.export_plan("mesh", Path("grid.nc"))] == ["mesh"]
    assert [e["view"] for e in export_door.export_plan("global", Path("grid.nc"), view="whole")] == ["whole"]


def test_a_box_over_the_dateline_or_a_pole_stays_one_valid_window():
    window, nx, ny, degrees = export_door._latlon_box(40.0, 70.0, 170.0, 200.0, 0.1)
    s, n, w, e, step = map(float, window[len("latlon:"):].split(","))
    assert (w, e) == (170.0, 200.0) and nx == 301
    window, nx, ny, degrees = export_door._latlon_box(60.0, 95.0, -20.0, 20.0, 0.1)
    s, n, w, e, step = map(float, window[len("latlon:"):].split(","))
    # every longitude, at the spacing whose whole circle fits the converter's 2000 points a side
    assert n == 90.0 and (w, e) == (-180.0, 180.0 - step) and nx <= export_door.MAX_POINTS_PER_SIDE
    # More than the converter's 2000 points a side coarsens the box rather than being refused.
    window, nx, ny, degrees = export_door._latlon_box(-60.0, 60.0, -100.0, 100.0, 0.05)
    assert max(nx, ny) <= export_door.MAX_POINTS_PER_SIDE and degrees > 0.05


def _fake_convert(calls):
    def convert(convert_exe, *, histories, scratch, window, **_kwargs):
        calls.append((window, [Path(h).name for h in histories]))
        frames = Path(scratch) / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        rows = []
        for history in histories:
            valid = export_door.history_valid_time(Path(history))
            out = frames / f"wrfout_d01_{valid:%Y-%m-%d_%H_%M_%S}"
            out.write_bytes(b"frame " + window.encode())
            rows.append({"output": str(out), "output_sha256": "x" * 64, "valid_time": f"{valid:%Y-%m-%dT%H:%M:%S}"})
        return {"frames": rows}, ["fake", window]
    return convert


def _forecast(tmp_path, hours=3):
    forecast = tmp_path / "run" / "forecast"
    forecast.mkdir(parents=True)
    histories = []
    for h in range(hours + 1):
        history = forecast / f"history.2026-10-30_{h:02d}.00.00.nc"
        history.write_bytes(b"h" * (h + 1))
        with (forecast / "history-ready.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": h, "path": str(history)}) + "\n")
        histories.append(history)
    (tmp_path / "run" / "grid.nc").write_bytes(b"g")
    (tmp_path / "run" / "init.nc").write_bytes(b"i")
    return forecast, histories


def _args(tmp_path, forecast, **extra):
    values = dict(
        forecast_dir=str(forecast), history=None, mesh=str(tmp_path / "run" / "grid.nc"),
        init=str(tmp_path / "run" / "init.nc"), simulation_start="2026-10-30_00:00:00", out=str(tmp_path / "export"),
        scratch=None, window="global", view=None, field_set="surface", remap="product", nc_format="cdf5",
        model_label="hex", jobs=1, keep_scratch=False, overwrite=False, convert_exe=None, follow=False,
        follow_frames=None, follow_timeout_minutes=1.0, follow_poll_seconds=0.01, follow_from=None,
        follow_every_minutes=None)
    values.update(extra)
    return argparse.Namespace(**values)


def test_the_export_files_every_grid_as_its_wrf_domain(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert(calls))
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    monkeypatch.setenv("WOOF_COLUMNS", "off")
    forecast, histories = _forecast(tmp_path)
    assert export_door.run_export(_args(tmp_path, forecast)) == 0
    names = sorted(p.relative_to(tmp_path / "export").as_posix() for p in (tmp_path / "export").rglob("wrfout_*"))
    assert len(names) == 6 * 4
    assert "global/2026-10-30/wrfout_d01_2026-10-30_00_00_00" in names
    assert "mesh/2026-10-30/wrfout_d06_2026-10-30_03_00_00" in names
    assert "ring-6km/2026-10-30/wrfout_d05_2026-10-30_01_00_00" in names
    receipt = json.loads((tmp_path / "export" / export_door.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert [e["converted"] for e in receipt["stack"]] == [4] * 6 and receipt["view"] == "global"
    assert len(calls) == 6


def test_the_export_step_takes_the_frames_the_follower_wrote(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert(calls))
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    monkeypatch.setenv("WOOF_COLUMNS", "off")
    forecast, histories = _forecast(tmp_path)
    assert export_door.run_export(_args(tmp_path, forecast, follow=True, follow_frames=3)) == 0
    followed = len(calls)
    rows = [json.loads(line) for line in (tmp_path / "export" / export_door.READY_NAME).read_text().splitlines()]
    assert len(rows) == 3 * 6 and all(row["history_identity"] for row in rows)
    # The one-shot export after the forecast: the 3 followed frames are taken, only the 4th is converted.
    assert export_door.run_export(_args(tmp_path, forecast)) == 0
    after = calls[followed:]
    assert len(after) == 6 and all(names == [histories[-1].name] for _window, names in after)
    receipt = json.loads((tmp_path / "export" / export_door.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert [(e["reused"], e["converted"]) for e in receipt["stack"]] == [(3, 1)] * 6
    assert len(receipt["frames"]) == 24
    # The export step run again takes every frame; a history rewritten since is converted again, never taken stale.
    histories[0].write_bytes(b"rewritten")
    calls.clear()
    assert export_door.run_export(_args(tmp_path, forecast)) == 0
    assert len(calls) == 6 and all(names == [histories[0].name] for _window, names in calls)


def test_the_live_export_is_the_hosts_export_beside_the_forecast(tmp_path):
    forecast = tmp_path / "run" / "forecast"
    flags = ["--out", "/w/run/export", "--window", "global", "--field-set", "surface", "--follow"]
    argv = export_door.live_export_command(forecast, grid="g.nc", init="i.nc", start_time="2026-10-30_00:00:00",
                                           frames=37, environ={"WOOF_HEX_EXPORT_LIVE": json.dumps(flags)})
    tail = argv[argv.index("export"):]
    assert tail[:2] == ["export", str(forecast)] and tail.count("--follow") == 1
    assert tail[tail.index("--follow-frames") + 1] == "37" and tail[-6:] == flags[:6]
    for value in ("", "not json", json.dumps({"out": 1})):
        assert export_door.live_export_command(forecast, grid="g", init="i", start_time="s", frames=1,
                                               environ={"WOOF_HEX_EXPORT_LIVE": value}) is None
    assert export_door.LiveExport(forecast, grid="g", init="i", start_time="s", frames=1,
                                  log=tmp_path / "l.log").start().finish(forecast_ok=True) == {"status": "off"}


def test_a_frame_listed_under_two_paths_is_followed_once(tmp_path, monkeypatch):
    """RTX 5090, 2026-10-30: the forecast door's durable sidecars named f000 through a symlinked folder and its
    history-ready.jsonl (written a moment later) under the resolved path; the follower took f000 twice and refused its
    own frame ("exists"), and the live export stopped."""
    calls = []
    monkeypatch.setattr(export_door, "_mesh_cells", lambda mesh: _zoom_globe())
    monkeypatch.setattr(render_door, "convert_frames", _fake_convert(calls))
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda value: Path("fake"))
    real = tmp_path / "real"
    forecast, histories = _forecast(real, hours=1)
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    seen = [[link / "run" / "forecast" / histories[0].name], [histories[0], link / "run" / "forecast" / histories[1].name]]
    monkeypatch.setattr(export_door, "_history_ready_paths", lambda folder: seen.pop(0) if len(seen) > 1 else seen[0])
    args = _args(real, forecast, follow=True, follow_frames=2, out=str(tmp_path / "export"))
    assert export_door.run_export(args) == 0
    assert [names for _window, names in calls[::6]] == [[histories[0].name], [histories[1].name]]
